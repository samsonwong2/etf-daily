"""Tests for fair-need gap investability helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from decision_pack.src.hold_vs_swing_bench import spread_metrics
from decision_pack.src.regime_transition_plot import compute_volatility_regime_frame
from decision_pack.src.vol_need_return_board import (
    ANCHOR_ERP_ANN,
    fair_need_ann_series,
    gap_realized_minus_need,
)

_ROOT = Path(__file__).resolve().parents[2]


def _load_gap_script():
    path = _ROOT / "decision_pack/scripts/backtest_need_gap_spread.py"
    spec = importlib.util.spec_from_file_location("need_gap_spread_mod", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gap_positive_when_return_beats_need():
    idx = pd.bdate_range("2024-01-02", periods=80)
    close = pd.Series(100 * np.exp(np.cumsum(np.full(80, 0.01))), index=idx)
    vol = compute_volatility_regime_frame(close)
    vol = vol.set_index(pd.to_datetime(vol["as_of"]).dt.normalize())
    need = fair_need_ann_series(vol["rv20"], vol["rv20"], erp_ann=ANCHOR_ERP_ANN)
    gap = gap_realized_minus_need(close, need, bars=20)
    assert float(gap["gap"].iloc[-1]) > 0


def test_assign_gap_and_need_buckets():
    mod = _load_gap_script()
    panel = pd.DataFrame(
        {
            "date": ["2024-01-31", "2024-01-31", "2024-02-29", "2024-02-29"],
            "code": ["A", "B", "A", "B"],
            "gap_20d": [0.02, -0.01, 0.0, 0.03],
            "gap_5d": [0.01, -0.02, -0.01, 0.02],
            "r_need_ann": [0.08, 0.03, 0.09, 0.04],
            "fwd_63": [0.1, 0.0, 0.05, -0.02],
        }
    )
    g = mod.assign_gap_bucket(panel, bars=20, thr=0.0)
    assert list(g["bucket"]) == ["investable", "skip", "skip", "investable"]
    n = mod.assign_need_bucket(panel)
    assert n.loc[0, "bucket"] == "high_need"
    assert n.loc[1, "bucket"] == "low_need"
    m = spread_metrics(
        g,
        horizon=63,
        min_group_n=1,
        pos_label="investable",
        neg_label="skip",
    )
    assert np.isfinite(m["spread"])
