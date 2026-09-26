"""SH515880 regime 色带因果增强 A/B（研究用，不入生产）。

Question: can fig1-fig11 *causal* features (multi-scale fair-path band position,
vol regime, HMM p_into_reversal) improve regime labels over frozen adx_di?

Variants (all decisions at day t use data <= t only):
  V0  baseline frozen adx_di (thr=18, smooth=3, min_seg=10)
  V1  upper-rail reject: raw up with band_pos >= 0.9 on fig8/fig9 -> range
  V2  lower-rail assist: range + touch_lo (fig9/fig10) + prior20>0 -> up
  V3  vol-extreme gate: enter up needs 2 consecutive raw-up when vol_regime==extreme
  V4  HMM confirm: enter up only if p_into_reversal_hmm >= train median
  V5  V1 + V3

Eval: train (<2026-04-01) / OOS (2026-04-01..2026-09-04) / full, dual-truth
metrics from eval_adaptive_regime_accuracy._window_metrics + up-seg loss count.

Example::

    PYTHONPATH=. python decision_pack/scripts/backtest_sh515880_regime_augment.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runtime_paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from decision_pack.src.adaptive_stage_common import prepare_features, segments  # noqa: E402
from decision_pack.src.fair_path_band_signals import (  # noqa: E402
    FIG_PANELS,
    compute_multi_scale_frame,
    load_ohlcv_from_regime_html,
)
from decision_pack.src.regime_transition_plot import (  # noqa: E402
    VOL_REGIME_EXTREME,
    compute_volatility_regime_frame,
)

CODE = "SH515880"
HTML = (
    PLOTLY_OUTPUTS_DIR / "20260904_from_listing"
    / "regime_transition_SH515880_通信ETF国泰_20190906_20260904_adaptive.html"
)
CFG_PATH = PLOTLY_OUTPUTS_DIR / "20260904_from_listing/configs/SH515880.json"
OUT_DIR = PLOTLY_OUTPUTS_DIR / "20260904_from_listing/regime_augment_SH515880"
TRAIN_CUTOFF = "2026-04-01"
END = "2026-09-04"


def _load(mod_name: str, rel: str):
    path = _PROJECT_ROOT / rel
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_inputs() -> dict[str, Any]:
    ev = _load("ev_augment", "decision_pack/scripts/eval_causal_regime_switch_pool.py")
    acc = _load("acc_augment", "decision_pack/scripts/eval_adaptive_regime_accuracy.py")
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    _, ohlcv = load_ohlcv_from_regime_html(HTML)
    ohlcv = ohlcv.reset_index()
    if "datetime" not in ohlcv.columns:
        ohlcv = ohlcv.rename(columns={ohlcv.columns[0]: "datetime"})
    px = prepare_features(ev, ohlcv, method=None)
    close = px.set_index(pd.to_datetime(px["as_of"]).dt.normalize())["$close"].astype(float)
    close = close[~close.index.duplicated(keep="last")].sort_index()
    ms = compute_multi_scale_frame(close)
    vol = compute_volatility_regime_frame(close)
    return dict(ev=ev, acc=acc, cfg=cfg, px=px, close=close, ms=ms, vol=vol)


def hmm_walkforward_series(close: pd.Series) -> pd.DataFrame:
    """Causal p_into_reversal_hmm / p_switch_hmm per day (frozen refits, q=0.90)."""
    from decision_pack.src.regime_transition_board import (
        RegimeTransitionConfig,
        compute_frozen_transition_step,
        fit_frozen_regime_model,
    )

    cfg = RegimeTransitionConfig()
    s = close.copy()
    s.index = pd.to_datetime(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    rows: list[dict[str, Any]] = []
    model = None
    alpha_prev = None
    last_fit_i = -1
    refit_days = 63
    min_warmup = 260
    for i, d in enumerate(s.index):
        if i < max(min_warmup - 1, 2):
            continue
        need = model is None or not model.fit.reliable or (i - last_fit_i) >= refit_days
        if need:
            try:
                model = fit_frozen_regime_model(s, d, cfg=cfg, prev_model=model)
                alpha_prev = None
                last_fit_i = i
            except Exception:
                continue
        try:
            metrics, alpha_t, _step = compute_frozen_transition_step(
                s,
                d,
                model,
                cfg=cfg,
                alpha_prev=alpha_prev,
            )
            if alpha_t is not None:
                alpha_prev = np.asarray(alpha_t, dtype=float).ravel()
        except Exception:
            alpha_prev = None
            continue
        rows.append(
            {
                "as_of": pd.Timestamp(d).normalize(),
                "p_into_reversal_hmm": metrics.p_into_reversal_hmm,
                "p_switch_hmm": metrics.p_switch_hmm,
            }
        )
    return pd.DataFrame(rows).set_index("as_of")


def band_pos(ms: pd.DataFrame, key: str) -> pd.Series:
    """(close - lower) / (upper - lower); 1.0 = at upper rail."""
    up = pd.to_numeric(ms[f"{key}_upper"], errors="coerce")
    lo = pd.to_numeric(ms[f"{key}_lower"], errors="coerce")
    px = pd.to_numeric(ms["px"], errors="coerce")
    width = (up - lo).replace(0, np.nan)
    return (px - lo) / width


def apply_variant(
    ev,
    px: pd.DataFrame,
    base_raw: np.ndarray,
    feat: pd.DataFrame,
    variant: str,
    train_median_p_rev: float | None,
) -> np.ndarray:
    """Apply causal gates to finalized V0 labels, then re-finalize (min_seg)."""
    labs = np.asarray(base_raw, dtype=object).copy()
    n = len(labs)
    bp8 = feat["bp_fig8"].to_numpy(float)
    bp9 = feat["bp_fig9"].to_numpy(float)
    tlo9 = feat["fig9_touch_lo"].fillna(False).to_numpy(bool)
    tlo10 = feat["fig10_touch_lo"].fillna(False).to_numpy(bool)
    p20 = px["prior20"].to_numpy(float)
    vol_ext = feat["vol_extreme"].fillna(False).to_numpy(bool)
    prev = pd.to_numeric(px["$close"], errors="coerce").to_numpy(float)

    if variant in {"V1", "V5"}:
        hot = (np.nan_to_num(bp8, nan=0.0) >= 0.90) | (np.nan_to_num(bp9, nan=0.0) >= 0.90)
        labs[(labs == "up") & hot] = "range"

    if variant == "V2":
        cheap = tlo9 | tlo10
        turn_up = np.isfinite(p20) & (p20 > 0)
        labs[(labs == "range") & cheap & turn_up] = "up"

    if variant in {"V3", "V5"}:
        # enter up only after 2 consecutive raw-up days during extreme vol
        out = labs.copy()
        for i in range(n):
            if labs[i] == "up" and vol_ext[i]:
                prev_up = i > 0 and out[i - 1] == "up"
                if not prev_up:
                    out[i] = "range"
        labs = out

    if variant == "V4" and train_median_p_rev is not None:
        p_rev = feat["p_into_reversal_hmm"].to_numpy(float)
        # gate only *entries* into up (prev day not up)
        for i in range(n):
            if labs[i] == "up" and (i == 0 or labs[i - 1] != "up"):
                p = p_rev[i]
                if not np.isfinite(p) or p < train_median_p_rev:
                    labs[i] = "range"
        _ = prev

    return ev.finalize_regime_labels(labs, px, min_seg=10)


def up_seg_loss_count(labs: np.ndarray, dates: np.ndarray, close: np.ndarray) -> tuple[int, int]:
    seg = segments(labs, dates, close)
    ups = seg[seg.regime == "up"]
    return int((ups.ret < -0.02).sum()), int(len(ups))


def main() -> int:
    inp = load_inputs()
    ev, acc, px, close, ms, vol = inp["ev"], inp["acc"], inp["px"], inp["close"], inp["ms"], inp["vol"]
    cfg = inp["cfg"]
    params = cfg.get("method_params") or {"adx_thr": 18.0, "smooth": 3, "min_seg": 10}

    as_of = pd.to_datetime(px["as_of"]).dt.normalize()
    close_full = px["$close"].to_numpy(float)
    dates = as_of.dt.strftime("%Y-%m-%d").to_numpy()
    train_m = as_of < pd.Timestamp(TRAIN_CUTOFF)
    oos_m = (as_of >= pd.Timestamp(TRAIN_CUTOFF)) & (as_of <= pd.Timestamp(END))

    truth_retro = ev.retrospective_truth(close_full)
    truth_causal = ev.causal_confirmed_swing_truth(close_full)

    # features aligned to px
    feat = pd.DataFrame(index=as_of)
    feat["bp_fig8"] = band_pos(ms, "fig8").reindex(as_of).to_numpy()
    feat["bp_fig9"] = band_pos(ms, "fig9").reindex(as_of).to_numpy()
    feat["fig9_touch_lo"] = ms["fig9_touch_lo"].reindex(as_of).to_numpy()
    feat["fig10_touch_lo"] = ms["fig10_touch_lo"].reindex(as_of).to_numpy()
    vol_idx = vol.set_index(pd.to_datetime(vol["as_of"]).dt.normalize())["vol_regime"]
    feat["vol_extreme"] = vol_idx.reindex(as_of).eq(VOL_REGIME_EXTREME).to_numpy()

    print("[hmm] walkforward p_into_reversal (may take a minute)…")
    hmm = hmm_walkforward_series(close)
    feat["p_into_reversal_hmm"] = hmm["p_into_reversal_hmm"].reindex(as_of).to_numpy()
    train_median_p_rev = float(
        np.nanmedian(feat.loc[train_m.to_numpy(), "p_into_reversal_hmm"])
    )
    print(f"[hmm] train median p_into_reversal = {train_median_p_rev:.4f}")

    base = ev.run_method("adx_di", px, params=params)

    variants = ["V0", "V1", "V2", "V3", "V4", "V5"]
    rows: list[dict[str, Any]] = []
    for v in variants:
        labs = (
            np.asarray(base, dtype=object)
            if v == "V0"
            else apply_variant(ev, px, base, feat, v, train_median_p_rev)
        )
        row: dict[str, Any] = {"variant": v}
        for tag, mask in (("train", train_m), ("oos", oos_m), ("full", np.ones(len(px), bool))):
            idx = np.where(np.asarray(mask))[0]
            if len(idx) < 10:
                continue
            met = acc._window_metrics(
                ev=ev,
                labs=labs[idx],
                close=close_full[idx],
                truth_retro=truth_retro[idx],
                truth_causal=truth_causal[idx],
                prefix=tag,
            )
            row.update(met)
            n_bad, n_up = up_seg_loss_count(labs[idx], dates[idx], close_full[idx])
            row[f"{tag}_up_seg_loss"] = n_bad
            row[f"{tag}_up_seg_n"] = n_up
        rows.append(row)
        print(
            f"[{v}] oos_retro_mf1={row.get('oos_retro_macro_f1', float('nan')):.3f} "
            f"oos_causal_mf1={row.get('oos_causal_macro_f1', float('nan')):.3f} "
            f"oos_hold_up={row.get('oos_hold_up', float('nan')):+.1%} "
            f"oos_up_loss={row.get('oos_up_seg_loss')}/{row.get('oos_up_seg_n')}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] {OUT_DIR / 'metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
