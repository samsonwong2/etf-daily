#!/usr/bin/env python3
"""Reproducible dual-truth regime accuracy for adaptive pool configs.

For each symbol, evaluate the frozen method (or all METHODS) under:
  - retro_full: retrospective_truth on full history, sliced to train/OOS
  - causal: causal_confirmed_swing_truth on the same window

Usage::

    $PY decision_pack/scripts/eval_adaptive_regime_accuracy.py \\
      --adaptive-dir ~/etf-daily-output/temp/plotly_outputs/20260806all_adaptive \\
      --end-date 2026-08-06 --oos-start 2026-04-01
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from decision_pack.src.adaptive_stage_common import (  # noqa: E402
    hold_up_compound,
    prepare_features,
    worst_up_drawdown,
)
from decision_pack.src.symbol_trend_board import load_cluster_mapping_codes  # noqa: E402
from runtime_paths import CLUSTER_MAPPING_SELECTED_TXT, PLOTLY_OUTPUTS_DIR  # noqa: E402

WARMUP_CALENDAR_DAYS = 220
DEFAULT_ADAPTIVE = (
    PLOTLY_OUTPUTS_DIR / "20260806all_adaptive"
)


def _load_ev():
    path = _PROJECT_ROOT / "decision_pack/scripts/eval_causal_regime_switch_pool.py"
    spec = importlib.util.spec_from_file_location("ev_acc_repro", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_pm():
    path = _PROJECT_ROOT / "decision_pack/scripts/plot_regime_transition_example.py"
    spec = importlib.util.spec_from_file_location("pm_acc_repro", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _window_metrics(
    *,
    ev,
    labs: np.ndarray,
    close: np.ndarray,
    truth_retro: np.ndarray,
    truth_causal: np.ndarray,
    prefix: str,
) -> dict[str, Any]:
    mr = ev.eval_labels(truth_retro, labs)
    mc = ev.eval_labels(truth_causal, labs)
    hu, _ = hold_up_compound(labs, close)
    out: dict[str, Any] = {}
    for k, v in mr.items():
        out[f"{prefix}_retro_{k}"] = v
    for k, v in mc.items():
        out[f"{prefix}_causal_{k}"] = v
    out[f"{prefix}_hold_up"] = hu
    out[f"{prefix}_worst_up_dd"] = worst_up_drawdown(labs, close)
    out[f"{prefix}_bh"] = float(close[-1] / close[0] - 1) if len(close) > 1 else 0.0
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adaptive-dir", type=Path, default=DEFAULT_ADAPTIVE)
    p.add_argument("--oos-start", default="2026-04-01")
    p.add_argument("--end-date", default="2026-08-06")
    p.add_argument("--train-cutoff", default=None, help="default = oos-start")
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--code", action="append", default=None)
    p.add_argument(
        "--all-methods",
        action="store_true",
        help="also score every METHOD (slow); default: frozen config method only",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    oos_start = pd.Timestamp(args.oos_start).strftime("%Y-%m-%d")
    end = pd.Timestamp(args.end_date).strftime("%Y-%m-%d")
    train_cutoff = pd.Timestamp(args.train_cutoff or oos_start).strftime("%Y-%m-%d")
    adaptive_dir = (
        args.adaptive_dir
        if args.adaptive_dir.is_absolute()
        else _PROJECT_ROOT / args.adaptive_dir
    )
    out_dir = args.out_dir or adaptive_dir
    out_dir = out_dir if out_dir.is_absolute() else _PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dir = adaptive_dir / "configs"
    if args.code:
        codes = [c.upper() for c in args.code]
    elif (adaptive_dir / "batch_summary.csv").exists():
        codes = pd.read_csv(adaptive_dir / "batch_summary.csv")["code"].astype(str).tolist()
    else:
        codes = list(load_cluster_mapping_codes(CLUSTER_MAPPING_SELECTED_TXT))
    if args.max_codes:
        codes = codes[: args.max_codes]

    ev = _load_ev()
    pm = _load_pm()
    pm._init_qlib(None)

    rows: list[dict] = []
    method_rows: list[dict] = []

    for i, code in enumerate(codes, 1):
        cfg_path = cfg_dir / f"{code}.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        method = cfg.get("method") or "ma_stack_strict"
        params = cfg.get("method_params") or {}
        try:
            ohlcv = pm.load_qlib_ohlcv(
                code,
                "2015-01-01",
                end,
                init_qlib=False,
                lookback_calendar_days=WARMUP_CALENDAR_DAYS,
                clip_to_window=False,
            )
            px = prepare_features(ev, ohlcv, method=None)
            as_of = px["as_of"]
            close_full = px["$close"].to_numpy(float)
            # index masks
            train_m = as_of < train_cutoff
            oos_m = (as_of >= oos_start) & (as_of <= end)
            if oos_m.sum() < 5:
                raise ValueError("short oos")

            # full-slice retrospective truth
            truth_full = ev.retrospective_truth(close_full)
            truth_causal_full = ev.causal_confirmed_swing_truth(close_full)

            methods_to_run = [method]
            if args.all_methods:
                methods_to_run = list(ev.METHODS.keys())

            frozen_labs = None
            for mname in methods_to_run:
                p = params if mname == method else {}
                labs_full = ev.run_method(mname, px, params=p)
                if mname == method:
                    frozen_labs = labs_full

                for tag, mask in (("train", train_m), ("oos", oos_m)):
                    if mask.sum() < 10:
                        continue
                    idx = np.where(mask.to_numpy())[0]
                    labs = labs_full[idx]
                    close = close_full[idx]
                    tr = truth_full[idx]
                    tc = truth_causal_full[idx]
                    met = _window_metrics(
                        ev=ev,
                        labs=labs,
                        close=close,
                        truth_retro=tr,
                        truth_causal=tc,
                        prefix=tag,
                    )
                    method_rows.append(
                        dict(
                            code=code,
                            method=mname,
                            is_frozen=(mname == method),
                            **met,
                        )
                    )

            assert frozen_labs is not None
            row = dict(
                code=code,
                method=method,
                skipped_train=cfg.get("skipped_train"),
                train_bars=cfg.get("train_bars"),
                score_mode=cfg.get("score_mode"),
            )
            for tag, mask in (("train", train_m), ("oos", oos_m)):
                idx = np.where(mask.to_numpy())[0]
                labs = frozen_labs[idx]
                close = close_full[idx]
                tr = truth_full[idx]
                tc = truth_causal_full[idx]
                row.update(
                    _window_metrics(
                        ev=ev,
                        labs=labs,
                        close=close,
                        truth_retro=tr,
                        truth_causal=tc,
                        prefix=tag,
                    )
                )
            rows.append(row)
            print(
                f"[{i}/{len(codes)}] {code} {method} "
                f"oos_retro_agree={row.get('oos_retro_agree', float('nan')):.1%} "
                f"oos_retro_mf1={row.get('oos_retro_macro_f1', float('nan')):.3f} "
                f"oos_causal_mf1={row.get('oos_causal_macro_f1', float('nan')):.3f}"
            )
        except Exception as e:
            print(f"[{i}/{len(codes)}] FAIL {code}: {e}")
            rows.append(dict(code=code, method=method, error=str(e)))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "regime_accuracy_dual.csv", index=False, encoding="utf-8-sig")
    if method_rows:
        pd.DataFrame(method_rows).to_csv(
            out_dir / "regime_accuracy_dual_by_method.csv",
            index=False,
            encoding="utf-8-sig",
        )

    ok = df.dropna(subset=["oos_retro_agree"]) if "oos_retro_agree" in df.columns else df
    report = f"""# Dual-truth regime accuracy

> adaptive_dir=`{adaptive_dir}` · OOS **{oos_start}→{end}** · train < {train_cutoff}
> n={len(ok)}

## Headline (frozen methods)

| Metric | Value |
|:--|--:|
| OOS retro agree mean | {ok['oos_retro_agree'].mean():.1%} |
| OOS retro macro-F1 mean | {ok['oos_retro_macro_f1'].mean():.3f} |
| OOS retro balanced_acc mean | {ok['oos_retro_balanced_acc'].mean():.3f} |
| OOS causal macro-F1 mean | {ok['oos_causal_macro_f1'].mean():.3f} |
| OOS retro frac_truth range | {ok['oos_retro_frac_truth_range'].mean():.1%} |
| OOS retro frac_pred down | {ok['oos_retro_frac_pred_down'].mean():.1%} |
| sticky up (worst_dd≤−12%) | {int((ok['oos_worst_up_dd'] <= -0.12).sum())} |

## Notes

- `retro_full` = retrospective_truth(full history) sliced to window (fixes short-OOS 75% range artifact).
- `causal` = causal_confirmed_swing_truth (no look-ahead).
- Detail: `regime_accuracy_dual.csv`
"""
    (out_dir / "regime_accuracy_dual_report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
