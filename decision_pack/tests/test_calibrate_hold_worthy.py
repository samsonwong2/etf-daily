"""Tests for hold_worthy spread metrics (calibration objective)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from decision_pack.src.hold_vs_swing_bench import (
    month_end_forward_panel,
    rolling_hold_worthy,
    spread_metrics,
)
from decision_pack.src.ma20_truth_triggers import ma20_hug_truth


def test_spread_metrics_hold_beats_swing():
    idx = pd.bdate_range("2020-01-02", periods=300)
    up = pd.Series(100 * np.exp(np.cumsum(np.full(300, 0.003))), index=idx)
    dn = pd.Series(100 * np.exp(np.cumsum(np.full(300, -0.003))), index=idx)
    panels = []
    for code, close in (("UP", up), ("DN", dn)):
        truth = ma20_hug_truth(close)
        hw = rolling_hold_worthy(close, truth, window=63, hysteresis=False)
        panels.append(
            month_end_forward_panel(close, hw["bucket"], horizons=(63, 126), code=code)
        )
    panel = pd.concat(panels, ignore_index=True)
    m = spread_metrics(panel, horizon=63, min_group_n=1)
    assert m["n"] > 0
    if m["n_hold"] >= 1 and m["n_swing"] >= 1 and np.isfinite(m["spread"]):
        assert m["spread"] > 0


def test_month_end_forward_panel_causal_label():
    idx = pd.bdate_range("2024-01-02", periods=100)
    close = pd.Series(np.linspace(100, 110, 100), index=idx)
    bucket = pd.Series("hold_ok", index=idx)
    panel = month_end_forward_panel(close, bucket, horizons=(63,), code="X")
    assert not panel.empty
    assert "fwd_63" in panel.columns
    assert panel["bucket"].eq("hold_ok").all()


def test_uptrend_hold_downtrend_swing_for_spread():
    idx = pd.bdate_range("2020-01-02", periods=200)
    up = pd.Series(100 * np.exp(np.cumsum(np.full(200, 0.004))), index=idx)
    dn = pd.Series(100 * np.exp(np.cumsum(np.full(200, -0.004))), index=idx)
    hw_up = rolling_hold_worthy(up, ma20_hug_truth(up), window=63, hysteresis=False)
    hw_dn = rolling_hold_worthy(dn, ma20_hug_truth(dn), window=63, hysteresis=False)
    assert (hw_up["bucket"].iloc[80:] == "hold_ok").mean() > 0.7
    assert (hw_dn["bucket"].iloc[80:] == "swing_only").mean() > 0.7
