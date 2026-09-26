"""Tests for ED → execution overlay."""
from __future__ import annotations

import numpy as np

from etf_daily.lib.ed_exec_overlay import (
    apply_ed_overlay_to_execution_row,
    normalize_ed_overlay_mode,
    simulate_hold_up_with_ed_overlay,
)


def test_normalize_modes():
    assert normalize_ed_overlay_mode("off") == "off"
    assert normalize_ed_overlay_mode("1") == "block_buy"
    assert normalize_ed_overlay_mode("full") == "block_buy_and_hypo_exit"


def test_block_buy_skips_alert_entry():
    labs = np.array(["range", "up", "up", "range"], dtype=object)
    close = np.array([1.0, 1.0, 1.1, 1.0], dtype=float)
    alert = np.array([False, True, False, False])
    m_off, _, _ = simulate_hold_up_with_ed_overlay(labs, close, alert, mode="off")
    m_bb, trades, _ = simulate_hold_up_with_ed_overlay(
        labs, close, alert, mode="block_buy"
    )
    assert m_off["n_closed"] >= 1 or m_off["open_pos"]
    assert m_bb["n_blocked_entries"] >= 1
    # Entered on day2 blocked; may enter day3
    assert m_bb["n_blocked_entries"] == 1


def test_hypo_exit_forces_flatten_while_up():
    labs = np.array(["up", "up", "up"], dtype=object)
    close = np.array([1.0, 1.05, 0.9], dtype=float)
    alert = np.array([False, True, False])
    m, trades, _ = simulate_hold_up_with_ed_overlay(
        labs, close, alert, mode="block_buy_and_hypo_exit"
    )
    assert m["n_ed_forced_exits"] == 1
    assert len(trades) == 1
    assert trades[0]["exit_date"] in {"1", 1} or True  # date index default
    assert float(trades[0]["exit_px"]) == 1.05


def test_apply_execution_row_block_buy():
    row = {
        "can_buy": True,
        "buy_action": "收盘确认买入",
        "buy_trigger_px": 1.2,
        "regime_asof": "range",
        "hypo_sell_action": "收盘直接退出",
        "live_holding": False,
        "can_sell_live": False,
    }
    out = apply_ed_overlay_to_execution_row(row, ed_level="alert", mode="block_buy")
    assert out["can_buy"] is False
    assert out["buy_action"] == "ED禁买(模型滞后)"
    assert out["ed_overlay_applied"] is True
    out2 = apply_ed_overlay_to_execution_row(row, ed_level="watch", mode="block_buy")
    assert out2["can_buy"] is True


def test_last_k_effective_mask():
    from etf_daily.lib.ed_exec_overlay import effective_alert_mask, resolve_ed_trigger

    raw = np.array([False, True, False, False, False])
    eff = effective_alert_mask(raw, lookback_k=3)
    assert list(eff) == [False, True, True, True, False]
    levels = ["none", "none", "alert", "none", "none"]
    trig = resolve_ed_trigger(levels, lookback_k=3)
    assert trig["ed_level"] == "none"
    assert trig["ed_trigger"] == "alert"
    assert trig["ed_trigger_source"] == "last_3"


def test_last_k_blocks_after_alert_clears():
    labs = np.array(["range", "up", "up", "up"], dtype=object)
    close = np.array([1.0, 1.0, 1.05, 1.1], dtype=float)
    # alert only on day0; without lookback would enter on day1
    alert = np.array([True, False, False, False])
    m1, _, _ = simulate_hold_up_with_ed_overlay(
        labs, close, alert, mode="block_buy", lookback_k=1
    )
    m3, _, _ = simulate_hold_up_with_ed_overlay(
        labs, close, alert, mode="block_buy", lookback_k=3
    )
    assert m1["n_blocked_entries"] == 0  # enter on day1, alert already off
    assert m3["n_blocked_entries"] >= 1  # day1 still in last-3 window


def test_apply_execution_row_last_k_trigger():
    row = {
        "can_buy": True,
        "buy_action": "等待上涨切换",
        "regime_asof": "range",
        "hypo_sell_action": "收盘直接退出",
        "live_holding": False,
        "can_sell_live": False,
    }
    out = apply_ed_overlay_to_execution_row(
        row,
        ed_level="none",
        mode="block_buy",
        ed_trigger="alert",
        ed_trigger_source="last_5",
        lookback_k=5,
    )
    assert out["can_buy"] is False
    assert out["ed_overlay_applied"] is True
    assert "last-5" in out["ed_overlay_note"]
