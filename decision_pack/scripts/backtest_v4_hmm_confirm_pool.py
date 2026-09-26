"""Pool-wide A/B: V4 HMM p_into_reversal entry-confirm gate vs frozen baseline.

V4 (from backtest_sh515880_regime_augment.py): entering ``up`` requires the
causal walk-forward HMM ``p_into_reversal_hmm`` >= this code's train median.
Gate applies to every code's frozen method (adx_di / ma_stack_strict / ...).

Causal: HMM refit every 63 trading days on data <= t; median from train only.
Research only — does not touch production configs or HTML.

Example::

    PYTHONPATH=. python decision_pack/scripts/backtest_v4_hmm_confirm_pool.py \
      --adaptive-dir ~/etf-daily-output/temp/plotly_outputs/20260904_from_listing --jobs 8
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runtime_paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

TRAIN_CUTOFF = "2026-04-01"
END = "2026-09-04"


def _load(mod_name: str, rel: str):
    spec = importlib.util.spec_from_file_location(mod_name, _PROJECT_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _one_code(code: str, adaptive_dir: str) -> dict[str, Any]:
    from decision_pack.src.adaptive_stage_common import prepare_features, segments
    from decision_pack.src.fair_path_band_signals import load_ohlcv_from_regime_html
    from decision_pack.src.regime_transition_board import (
        RegimeTransitionConfig,
        compute_frozen_transition_step,
        fit_frozen_regime_model,
    )

    ev = _load("ev_v4pool", "decision_pack/scripts/eval_causal_regime_switch_pool.py")
    acc = _load("acc_v4pool", "decision_pack/scripts/eval_adaptive_regime_accuracy.py")

    adir = Path(adaptive_dir)
    cfg_path = adir / "configs" / f"{code}.json"
    htmls = sorted(
        p
        for p in adir.glob(f"regime_transition_{code}_*_adaptive.html")
        if "_fig9" not in p.stem and "_fig10" not in p.stem and "_oracle" not in p.stem
    )
    if not cfg_path.exists() or not htmls:
        return {"code": code, "error": "missing cfg/html"}
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    method = cfg.get("method") or "ma_stack_strict"
    params = dict(cfg.get("method_params") or {})
    min_seg = int(params.get("min_seg", 10) or 10)

    _, ohlcv = load_ohlcv_from_regime_html(htmls[0])
    ohlcv = ohlcv.reset_index()
    if "datetime" not in ohlcv.columns:
        ohlcv = ohlcv.rename(columns={ohlcv.columns[0]: "datetime"})
    px = prepare_features(ev, ohlcv, method=None)
    as_of = pd.to_datetime(px["as_of"]).dt.normalize()
    close_full = px["$close"].to_numpy(float)
    dates = as_of.dt.strftime("%Y-%m-%d").to_numpy()

    close = pd.Series(close_full, index=as_of)
    close = close[~close.index.duplicated(keep="last")].sort_index()

    # HMM walkforward p_into_reversal (causal, frozen refits)
    hcfg = RegimeTransitionConfig()
    p_rev = pd.Series(np.nan, index=close.index, dtype=float)
    model = None
    alpha_prev = None
    last_fit_i = -1
    for i, d in enumerate(close.index):
        if i < 259:
            continue
        need = model is None or not model.fit.reliable or (i - last_fit_i) >= 63
        if need:
            try:
                model = fit_frozen_regime_model(close, d, cfg=hcfg, prev_model=model)
                alpha_prev = None
                last_fit_i = i
            except Exception:
                continue
        try:
            metrics, alpha_t, _ = compute_frozen_transition_step(
                close, d, model, cfg=hcfg, alpha_prev=alpha_prev
            )
            if alpha_t is not None:
                alpha_prev = np.asarray(alpha_t, dtype=float).ravel()
            if metrics.p_into_reversal_hmm is not None:
                p_rev.iloc[i] = float(metrics.p_into_reversal_hmm)
        except Exception:
            alpha_prev = None
            continue

    base = np.asarray(ev.run_method(method, px, params=params), dtype=object)

    train_mask = (as_of < pd.Timestamp(TRAIN_CUTOFF)).to_numpy()
    p_train = p_rev.to_numpy()[train_mask]
    p_train = p_train[np.isfinite(p_train)]
    if len(p_train) < 20:
        median = None  # too short history → no gate (V4 == V0)
    else:
        median = float(np.median(p_train))

    labs_v4 = base.copy()
    n_gated = 0
    if median is not None:
        pv = p_rev.reindex(as_of).to_numpy(float)
        # Gate on the *evolving* label: while not up, every raw-up day is a
        # fresh entry attempt that must pass the HMM check (same semantics as
        # the single-code experiment; gating only base segment starts would
        # leave the rest of a rejected segment wrongly up).
        for i in range(len(labs_v4)):
            if labs_v4[i] == "up" and (i == 0 or labs_v4[i - 1] != "up"):
                p = pv[i]
                if not np.isfinite(p) or p < median:
                    labs_v4[i] = "range"
                    n_gated += 1
        labs_v4 = ev.finalize_regime_labels(labs_v4, px, min_seg=min_seg)

    truth_retro = ev.retrospective_truth(close_full)
    truth_causal = ev.causal_confirmed_swing_truth(close_full)

    row: dict[str, Any] = {
        "code": code,
        "method": method,
        "min_seg": min_seg,
        "hmm_median": median,
        "n_gated_entries": n_gated,
    }
    masks = {
        "train": train_mask,
        "oos": ((as_of >= pd.Timestamp(TRAIN_CUTOFF)) & (as_of <= pd.Timestamp(END))).to_numpy(),
        "full": np.ones(len(px), bool),
    }
    for tag, labs in (("v0", base), ("v4", labs_v4)):
        for wtag, mask in masks.items():
            idx = np.where(mask)[0]
            if len(idx) < 10:
                continue
            met = acc._window_metrics(
                ev=ev,
                labs=labs[idx],
                close=close_full[idx],
                truth_retro=truth_retro[idx],
                truth_causal=truth_causal[idx],
                prefix=wtag,
            )
            seg = segments(labs[idx], dates[idx], close_full[idx])
            ups = seg[seg.regime == "up"]
            met[f"{wtag}_up_seg_loss"] = int((ups.ret < -0.02).sum())
            met[f"{wtag}_up_seg_n"] = int(len(ups))
            for k, v in met.items():
                row[f"{tag}_{k}"] = v
    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--adaptive-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260904_from_listing",
    )
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--code", action="append", default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    adir = args.adaptive_dir if args.adaptive_dir.is_absolute() else _PROJECT_ROOT / args.adaptive_dir
    out_dir = args.out_dir or adir / "regime_augment_pool_v4"
    out_dir = out_dir if out_dir.is_absolute() else _PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.code:
        codes = [c.upper() for c in args.code]
    else:
        codes = sorted(p.stem for p in (adir / "configs").glob("*.json"))
    if args.max_codes:
        codes = codes[: args.max_codes]
    print(f"[INFO] codes={len(codes)} jobs={args.jobs} out={out_dir}")

    rows: list[dict[str, Any]] = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_one_code, c, str(adir)): c for c in codes}
            for i, fut in enumerate(as_completed(futs), 1):
                c = futs[fut]
                try:
                    r = fut.result()
                except Exception as exc:  # noqa: BLE001
                    r = {"code": c, "error": str(exc)}
                rows.append(r)
                oos0 = r.get("v0_oos_retro_macro_f1", float("nan"))
                oos4 = r.get("v4_oos_retro_macro_f1", float("nan"))
                print(f"[{i}/{len(codes)}] {c} {r.get('method')} oos_mf1 v0={oos0:.3f} v4={oos4:.3f} err={r.get('error')}")
    else:
        for i, c in enumerate(codes, 1):
            r = _one_code(c, str(adir))
            rows.append(r)
            print(f"[{i}/{len(codes)}] {c} done err={r.get('error')}")

    df = pd.DataFrame(rows).sort_values("code")
    df.to_csv(out_dir / "metrics_by_code.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] {out_dir / 'metrics_by_code.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
