"""Tests for replay_simulation strategy engine."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from decision_pack.src.replay_simulation import (
    StrategySpec,
    decide_reduce,
    load_holdings_for_sim,
    load_replay_frames,
    simulate_portfolio,
    simulate_symbol,
    SymbolSimState,
)


def _sample_replay() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "alert_level": "green",
                "suggest_reduce_pct": 0.0,
                "pnl_pct": 1.0,
                "dd_20d_high": 0.0,
                "reasons": "",
            },
            {
                "date": "2026-06-02",
                "alert_level": "yellow",
                "suggest_reduce_pct": 0.3,
                "pnl_pct": -1.0,
                "dd_20d_high": -0.02,
                "reasons": "dd_soft|mom_fade",
            },
            {
                "date": "2026-06-03",
                "alert_level": "red",
                "suggest_reduce_pct": 0.5,
                "pnl_pct": -3.0,
                "dd_20d_high": -0.05,
                "reasons": "dd5_20d",
            },
        ]
    )


def test_decide_reduce_warning_b_limits_actions():
    row_y = pd.Series(
        {
            "date": "2026-06-02",
            "alert_level": "yellow",
            "suggest_reduce_pct": 0.3,
            "pnl_pct": -1.0,
            "reasons": "dd_soft",
        }
    )
    row_r = pd.Series(
        {
            "date": "2026-06-03",
            "alert_level": "red",
            "suggest_reduce_pct": 0.5,
            "pnl_pct": -3.0,
            "reasons": "dd5_20d",
        }
    )
    spec = StrategySpec(
        name="B",
        yellow_reduce_pct=0.30,
        red_reduce_pct=0.50,
        max_yellow_actions=1,
        max_red_actions=1,
    )
    state = SymbolSimState(remaining_shares=1000.0)
    assert decide_reduce(row_y, state, spec) == pytest.approx(0.30)
    state.yellow_actions = 1
    state.total_actions = 1
    assert decide_reduce(row_y, state, spec) == 0.0
    assert decide_reduce(row_r, state, spec) == pytest.approx(0.50)
    state.red_actions = 1
    state.total_actions = 2
    assert decide_reduce(row_r, state, spec) == 0.0


def test_red_pnl_dd_only_blocks_trend_red():
    row = pd.Series(
        {
            "date": "2026-06-02",
            "alert_level": "red",
            "suggest_reduce_pct": 0.5,
            "pnl_pct": 1.0,
            "reasons": "below_ma20_3d+",
        }
    )
    spec = StrategySpec(name="red_pnl_dd_only", red_reasons_allow=frozenset({"pnl7", "dd5_20d"}))
    state = SymbolSimState(remaining_shares=100.0)
    assert decide_reduce(row, state, spec) == 0.0


def test_simulate_symbol_cap50_cumulative():
    replay = _sample_replay()
    spec = StrategySpec(name="cap50", max_cumulative_reduce=0.50)
    result = simulate_symbol("TEST", replay, cost=10.0, shares=1000.0, spec=spec)
    assert len(result.actions) >= 1
    assert result.final_position_pct >= 50.0


def test_regression_hold_warning_ab_against_live_replay():
    live = Path("~/etf-daily-output/temp/live_holdings/20260610.csv")
    replay_dir = Path("~/etf-daily-output/temp/decision_packs_replay/all_holdings")
    if not live.exists() or not replay_dir.exists():
        pytest.skip("live holdings or replay dir missing")

    holdings = load_holdings_for_sim(live)
    frames = load_replay_frames(replay_dir, tuple(sorted(holdings.keys())))

    hold = simulate_portfolio(holdings, frames, StrategySpec(name="hold", mode="hold"))
    warn_a = simulate_portfolio(holdings, frames, StrategySpec(name="warning_A"))
    warn_b = simulate_portfolio(
        holdings,
        frames,
        StrategySpec(
            name="warning_B",
            yellow_reduce_pct=0.30,
            red_reduce_pct=0.50,
            max_yellow_actions=1,
            max_red_actions=1,
            warning_b_branch=True,
        ),
    )

    assert hold.total_pnl == pytest.approx(-27548, abs=1)
    assert warn_a.total_pnl == pytest.approx(-28635, abs=1)
    assert warn_b.total_pnl == pytest.approx(-26252, abs=1)
