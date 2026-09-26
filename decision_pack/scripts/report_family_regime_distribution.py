#!/usr/bin/env python3
"""Read-only family distribution for Family Regime Router (no label changes).

Design: huangtuo-master-design-20260812-171400-family-regime-router.md
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

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from decision_pack.src.adaptive_stage_common import prepare_features  # noqa: E402
from decision_pack.src.symbol_trend_board import load_cluster_mapping_codes  # noqa: E402
from runtime_paths import CLUSTER_MAPPING_SELECTED_TXT, PLOTLY_OUTPUTS_DIR  # noqa: E402

# Design placeholder defaults
RV_LOW = 0.22
RV_MID = 0.30
RV_HIGH = 0.40
BULL_MIN = 0.25
DD_DEEP = -0.15
LOOK_RV = 60
LOOK_BULL = 60
LOOK_DD = 120

ANCHORS = {
    "SH515220": "cyclical_recovery",
    "SH513650": "lowvol_grind",
    "SH588710": "impulse_trend",
    "SH512480": "impulse_trend",
}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def classify_family_row(
    rv_med: float,
    bull_frac: float,
    dd: float,
) -> str:
    if not (np.isfinite(rv_med) and np.isfinite(bull_frac) and np.isfinite(dd)):
        return "default"
    if rv_med <= RV_LOW and bull_frac >= BULL_MIN:
        return "lowvol_grind"
    if rv_med >= RV_HIGH and bull_frac >= BULL_MIN:
        return "impulse_trend"
    if rv_med >= RV_MID and dd <= DD_DEEP:
        return "cyclical_recovery"
    return "default"


def features_asof(px: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, float]:
    sub = px[px["as_of"] <= as_of].copy()
    if len(sub) == 0:
        return {"rv_med": np.nan, "bull_frac": np.nan, "dd": np.nan}
    tail_rv = sub.tail(LOOK_RV)
    rv_med = (
        float(pd.to_numeric(tail_rv["rv20"], errors="coerce").median())
        if "rv20" in tail_rv.columns
        else float("nan")
    )
    tail_b = sub.tail(LOOK_BULL)
    bull = (
        (tail_b["$close"] > tail_b["MA20"]) & (tail_b["MA20"] > tail_b["MA60"])
    ).mean()
    bull_frac = float(bull)
    tail_dd = sub.tail(LOOK_DD)
    close = pd.to_numeric(tail_dd["$close"], errors="coerce").to_numpy(float)
    if len(close) == 0 or not np.isfinite(close).any():
        dd = float("nan")
    else:
        # max drawdown from running peak within window
        peak = np.maximum.accumulate(np.where(np.isfinite(close), close, -np.inf))
        dd_path = close / peak - 1.0
        dd = float(np.nanmin(dd_path))
    return {"rv_med": rv_med, "bull_frac": bull_frac, "dd": dd}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--as-of", default="2026-08-11")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260811_family_regime_distribution",
    )
    p.add_argument("--code", action="append", default=[])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    as_of = pd.Timestamp(args.as_of)

    pm = _load("pm_fam", _ROOT / "decision_pack/scripts/plot_regime_transition_example.py")
    ev = _load("ev_fam", _ROOT / "decision_pack/scripts/eval_causal_regime_switch_pool.py")
    pm._init_qlib(None)

    codes = [str(x).upper() for x in args.code] or list(
        load_cluster_mapping_codes(CLUSTER_MAPPING_SELECTED_TXT)
    )

    rows: list[dict[str, Any]] = []
    for i, code in enumerate(codes, 1):
        ohlcv = pm.load_qlib_ohlcv(
            code,
            "2015-01-01",
            args.as_of,
            init_qlib=False,
            lookback_calendar_days=400,
            clip_to_window=False,
        )
        px = prepare_features(ev, ohlcv, method=None)
        feat = features_asof(px, as_of)
        fam = classify_family_row(feat["rv_med"], feat["bull_frac"], feat["dd"])
        expect = ANCHORS.get(code)
        rows.append(
            {
                "code": code,
                "as_of": args.as_of,
                "family": fam,
                "anchor_expect": expect,
                "anchor_match": (expect is None)
                or (fam == expect),
                **feat,
            }
        )
        print(f"[{i}/{len(codes)}] {code} -> {fam} rv={feat['rv_med']:.3f} bull={feat['bull_frac']:.2f} dd={feat['dd']:.2%}")

    df = pd.DataFrame(rows).sort_values(["family", "code"])
    df.to_csv(out_dir / "family_distribution.csv", index=False, encoding="utf-8-sig")

    counts = df["family"].value_counts().to_dict()
    anchors = df[df["code"].isin(ANCHORS)].copy()
    summary = {
        "as_of": args.as_of,
        "n_symbols": len(df),
        "thresholds": {
            "RV_LOW": RV_LOW,
            "RV_MID": RV_MID,
            "RV_HIGH": RV_HIGH,
            "BULL_MIN": BULL_MIN,
            "DD_DEEP": DD_DEEP,
        },
        "counts": counts,
        "anchors": anchors.to_dict(orient="records"),
        "anchor_all_match": bool(anchors["anchor_match"].all()) if len(anchors) else True,
        "design": "huangtuo-master-design-20260812-171400-family-regime-router.md",
        "label_changes": False,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print("DONE counts=", counts, "anchor_all_match=", summary["anchor_all_match"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
