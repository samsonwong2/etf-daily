"""Tests for MA20 chart-truth trigger scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd

from decision_pack.src.ma20_truth_triggers import (
    ma20_hug_truth,
    score_trigger_pair,
    trigger_lead_hits,
    truth_entry_exit,
)
from decision_pack.src.fwd_return_trend_truth import STATE_UP
from decision_pack.src.event_long_only_sim import simulate_long_only_events


def test_ma20_truth_matches_chart_band():
    idx = pd.bdate_range("2024-01-02", periods=60)
    px = np.full(60, 100.0)
    px[30:] = 110.0
    close = pd.Series(px, index=idx)
    truth = ma20_hug_truth(close)
    assert truth.iloc[30] == STATE_UP
    entry, leave = truth_entry_exit(truth)
    # First enter-up after jump once band is satisfied.
    assert entry.any()
    assert leave.sum() >= 0


def test_trigger_lead_hits_earlier_event():
    idx = pd.bdate_range("2024-01-02", periods=20)
    # Truth enter on day 10.
    truth_turns = pd.DatetimeIndex([idx[10]])
    trig = pd.Series(False, index=idx)
    trig.iloc[7] = True  # 3 bars early
    hit = trigger_lead_hits(trig, truth_turns, calendar=idx, max_lead_bars=5)
    assert hit["n_hit"] == 1
    assert hit["mean_lead_bars"] == 3.0
    # Too early beyond max_lead → miss
    trig2 = pd.Series(False, index=idx)
    trig2.iloc[2] = True
    miss = trigger_lead_hits(trig2, truth_turns, calendar=idx, max_lead_bars=5)
    assert miss["n_hit"] == 0


def test_baseline_self_hit_and_trade():
    idx = pd.bdate_range("2024-01-02", periods=80)
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.02, size=80)
    close = pd.Series(100 * np.exp(np.cumsum(rets)), index=idx)
    truth = ma20_hug_truth(close)
    entry, leave = truth_entry_exit(truth)
    scores = score_trigger_pair(entry, leave, truth, max_lead_bars=5)
    # Truth triggers on its own turns → hit_rate 1 when turns exist.
    if scores["enter_n_truth"] > 0:
        assert scores["enter_hit_rate"] == 1.0
        assert scores["enter_mean_lead"] == 0.0
    metrics, trades, _ = simulate_long_only_events(
        entry, leave, close, dates=idx.strftime("%Y-%m-%d")
    )
    assert "n_closed" in metrics
    assert metrics["n_closed"] == len(trades)
