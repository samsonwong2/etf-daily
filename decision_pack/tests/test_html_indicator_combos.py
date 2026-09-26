"""Tests for causal HTML-indicator catalog and long-only event sim."""
from __future__ import annotations

import numpy as np
import pandas as pd

from decision_pack.src.event_long_only_sim import (
    COST_PER_SIDE,
    simulate_long_only_events,
)
from decision_pack.src.html_indicator_catalog import (
    EVENT_DEFS,
    LEAKAGE_OR_BANNED_EVENT_NAMES,
    build_event_frame,
    catalog_markdown,
    combine_entry_signals,
    events_in_category,
)


def _synth_close(n: int = 120, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    # Mild trend with noise so MA crosses occur.
    rets = rng.normal(0.0005, 0.015, size=n)
    px = 10.0 * np.exp(np.cumsum(rets))
    return pd.Series(px, index=idx)


def test_catalog_has_no_leakage_names():
    names = {e.name for e in EVENT_DEFS}
    assert not (names & LEAKAGE_OR_BANNED_EVENT_NAMES)
    md = catalog_markdown()
    assert "hmm_switch_up" in md
    assert "hold_up" in md.lower() or "上涨买" in md


def test_build_event_frame_ma_cross_and_shock():
    close = _synth_close()
    # Force an extreme drop day.
    close = close.copy()
    close.iloc[80] = close.iloc[79] * 0.90  # -10%
    ev = build_event_frame(close, code="TEST001", oos=None)
    assert set(e.name for e in EVENT_DEFS).issubset(ev.columns)
    assert not (set(ev.columns) & LEAKAGE_OR_BANNED_EVENT_NAMES)
    assert bool(ev["extreme_drop_le_7pct"].iloc[80])
    # Some MA cross should fire over 120 bars.
    assert ev[[c for c in ev.columns if c.startswith("ma5_")]].any().any()


def test_combine_entry_and():
    idx = pd.bdate_range("2024-01-02", periods=5)
    ev = pd.DataFrame(
        {
            "a": [False, True, True, False, True],
            "b": [False, False, True, True, True],
        },
        index=idx,
    )
    out = combine_entry_signals(ev, ["a", "b"], how="and")
    assert list(out.astype(bool)) == [False, False, True, False, True]


def test_simulate_t_plus_one_and_cost():
    # Flat then entry signal on day0 → buy day1; exit signal day2 → sell day3.
    dates = pd.bdate_range("2024-01-02", periods=6)
    close = pd.Series([100.0, 100.0, 110.0, 120.0, 120.0, 120.0], index=dates)
    entry = pd.Series([True, False, False, False, False, False], index=dates)
    exit_sig = pd.Series([False, False, True, False, False, False], index=dates)
    metrics, trades, equity = simulate_long_only_events(
        entry, exit_sig, close, dates=dates.strftime("%Y-%m-%d"), cost_per_side=0.001
    )
    assert metrics["n_closed"] == 1
    t = trades[0]
    # Buy at bar1=100, sell at bar3=120
    buy_eff = 100.0 * (1.0 + COST_PER_SIDE)
    sell_eff = 120.0 * (1.0 - COST_PER_SIDE)
    expected = sell_eff / buy_eff - 1.0
    assert abs(t["ret_net"] - expected) < 1e-12
    assert t["entry_date"] == "2024-01-03"
    assert t["exit_date"] == "2024-01-05"


def test_simulate_ignores_duplicate_entry_while_flat_pending():
    dates = pd.bdate_range("2024-01-02", periods=5)
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=dates)
    entry = pd.Series([True, True, False, False, False], index=dates)
    exit_sig = pd.Series([False, False, False, True, False], index=dates)
    metrics, trades, _ = simulate_long_only_events(
        entry, exit_sig, close, dates=dates.strftime("%Y-%m-%d")
    )
    assert metrics["n_closed"] == 1
    assert trades[0]["entry_date"] == "2024-01-03"  # first pending only


def test_events_in_category_partition():
    all_names = {e.name for e in EVENT_DEFS}
    union = set()
    for cat in ("hmm", "ma_trend", "vol", "fair_gap", "shock"):
        names = events_in_category(cat)
        assert names
        union.update(names)
    assert union == all_names
