"""Tests for causal rolling hold_worthy and PIT path edge."""
from __future__ import annotations

import numpy as np
import pandas as pd

from etf_daily.lib.hold_vs_swing_bench import (
    decision_dates,
    label_at,
    pit_path_edge,
    rolling_hold_worthy,
    stability_stats,
)
from etf_daily.lib.ma20_truth_triggers import ma20_hug_truth


def test_rolling_hold_worthy_causal_uptrend():
    idx = pd.bdate_range("2020-01-02", periods=200)
    close = pd.Series(100 * np.exp(np.cumsum(np.full(200, 0.003))), index=idx)
    truth = ma20_hug_truth(close)
    df = rolling_hold_worthy(close, truth, window=63, hysteresis=False)
    tail = df["bucket"].iloc[80:]
    assert (tail == "hold_ok").mean() > 0.8


def test_rolling_hold_worthy_downtrend_swing():
    idx = pd.bdate_range("2020-01-02", periods=200)
    close = pd.Series(100 * np.exp(np.cumsum(np.full(200, -0.003))), index=idx)
    truth = ma20_hug_truth(close)
    df = rolling_hold_worthy(close, truth, window=63, hysteresis=False)
    tail = df["bucket"].iloc[80:]
    assert (tail == "swing_only").mean() > 0.8


def test_hysteresis_reduces_flips():
    idx = pd.bdate_range("2020-01-02", periods=250)
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.02, size=250)
    close = pd.Series(100 * np.exp(np.cumsum(rets)), index=idx)
    truth = ma20_hug_truth(close)
    raw = rolling_hold_worthy(close, truth, window=63, hysteresis=False)
    sticky = rolling_hold_worthy(close, truth, window=63, hysteresis=True)
    st_raw = stability_stats(raw["bucket"], decision_freq="D")
    st_sticky = stability_stats(sticky["bucket"], decision_freq="D")
    assert st_sticky["n_flips"] <= st_raw["n_flips"]


def test_decision_dates_month_end():
    idx = pd.bdate_range("2024-01-02", periods=60)
    dec = decision_dates(idx, freq="M")
    assert len(dec) >= 2
    assert dec.is_monotonic_increasing


def test_pit_path_edge_hold_uses_bh():
    idx = pd.bdate_range("2024-01-02", periods=80)
    close = pd.Series(np.linspace(100, 120, 80), index=idx)
    equity = pd.DataFrame(
        {"date": idx.strftime("%Y-%m-%d"), "equity": close / close.iloc[0]}
    )
    bucket = pd.Series("hold_ok", index=idx)
    pit = pit_path_edge(
        equity, close, bucket, start=str(idx[0].date()), end=str(idx[-1].date())
    )
    assert abs(pit["edge_pit"]) < 0.02
    assert pit["pit_hold_frac"] == 1.0


def test_pit_path_edge_swing_vs_cash():
    idx = pd.bdate_range("2024-01-02", periods=80)
    close = pd.Series(np.linspace(100, 80, 80), index=idx)
    equity = pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "equity": np.ones(80)})
    bucket = pd.Series("swing_only", index=idx)
    pit = pit_path_edge(
        equity, close, bucket, start=str(idx[0].date()), end=str(idx[-1].date())
    )
    assert abs(pit["edge_pit"]) < 0.02
    assert pit["pit_hold_frac"] == 0.0
    assert label_at(bucket, idx[-1]) == "swing_only"
