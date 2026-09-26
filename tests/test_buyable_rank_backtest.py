"""Tests for buyable rank backtest accounting and selection."""
from __future__ import annotations

import math

import pandas as pd

from etf_daily.lib.buyable_rank import BuyableRankConfig, gate_to_buyable_config
from etf_daily.lib.buyable_rank_backtest import (
    StrategySpec,
    assign_segment,
    build_strategy_grid,
    open_to_close_return,
    portfolio_day_return,
    select_best_strategy,
    sharpe_ann,
    summarize_segment,
)


def test_open_to_close_return_applies_round_trip_cost():
    # 1% gross, 10bp/side → net 1% - 0.20% = 0.80%
    net = open_to_close_return(100.0, 101.0, cost_bps_per_side=10.0)
    assert net is not None
    assert abs(net - 0.008) < 1e-12


def test_cash_day_when_no_picks():
    ret, detail = portfolio_day_return(
        [],
        trade_date=pd.Timestamp("2026-07-18"),
        open_panel=pd.DataFrame(),
        close_panel=pd.DataFrame(),
        cost_bps_per_side=10.0,
    )
    assert ret == 0.0
    assert detail == []


def test_portfolio_day_return_equal_weight():
    idx = pd.DatetimeIndex([pd.Timestamp("2026-07-18")])
    open_panel = pd.DataFrame({"AAA": [10.0], "BBB": [20.0]}, index=idx)
    close_panel = pd.DataFrame({"AAA": [11.0], "BBB": [20.0]}, index=idx)
    ret, detail = portfolio_day_return(
        ["AAA", "BBB"],
        trade_date=idx[0],
        open_panel=open_panel,
        close_panel=close_panel,
        cost_bps_per_side=10.0,
    )
    # AAA: 10% - 0.2% = 9.8%; BBB: 0% - 0.2% = -0.2%; mean = 4.8%
    assert abs(ret - 0.048) < 1e-12
    assert len(detail) == 2


def test_assign_segment_boundaries():
    assert assign_segment("2026-05-31") == "train"
    assert assign_segment("2026-06-01") == "validation"
    assert assign_segment("2026-06-30") == "validation"
    assert assign_segment("2026-07-01") == "holdout"


def test_gate_configs():
    strict = gate_to_buyable_config("strict", k=8)
    assert strict.exclude_avoid_leg and strict.exclude_broken_path
    assert "below_ma20_3d+" in strict.trend_reject_flags

    no_path = gate_to_buyable_config("no_path_gate", k=8)
    assert no_path.exclude_broken_path is False

    core = gate_to_buyable_config("core_risk", k=8)
    assert core.exclude_broken_path is False
    assert core.trend_reject_flags == ("bear_stack", "deep_below_ma20")

    pos = gate_to_buyable_config("positive_score", k=8)
    assert pos.min_pick_score is not None and pos.min_pick_score > 0

    none = gate_to_buyable_config("none", k=8)
    assert none.exclude_avoid_leg is False
    assert none.exclude_trend_flags is False


def test_select_best_by_validation_sharpe():
    rows = []
    for strategy, val_sharpe, hold_sharpe, is_bench in [
        ("a__upside_x_cred__k2__c0", 0.5, 0.1, False),
        ("b__dir_x_cred__k3__c0", 1.2, -0.2, False),
        ("bench_x", 9.0, 9.0, True),
    ]:
        for segment, sharpe in [("validation", val_sharpe), ("holdout", hold_sharpe), ("full", 0.0)]:
            rows.append(
                {
                    "segment": segment,
                    "strategy": strategy,
                    "sharpe": sharpe,
                    "cum_ret": sharpe * 0.01,
                    "max_dd": -0.05,
                    "hit_rate": 0.5,
                    "avg_n_picks": 2.0,
                    "cash_day_rate": 0.1,
                    "n_days": 20,
                    "is_benchmark": is_bench,
                    "rank_rule": "upside_x_cred",
                    "gate": "strict",
                    "k": 2,
                    "min_credibility": 0.0,
                }
            )
    summary = pd.DataFrame(rows)
    best = select_best_strategy(summary)
    assert best is not None
    assert best["strategy"] == "b__dir_x_cred__k3__c0"
    assert best["holdout"]["sharpe"] == -0.2


def test_summarize_and_sharpe():
    daily = pd.DataFrame(
        {
            "strategy": ["s"] * 5,
            "net_ret": [0.01, -0.005, 0.002, 0.0, 0.003],
            "n_picks": [2, 2, 0, 1, 2],
        }
    )
    metrics = summarize_segment(daily, strategy="s", segment="validation")
    assert metrics.n_days == 5
    assert metrics.n_trade_days == 4
    assert metrics.cash_day_rate == 0.2
    assert math.isfinite(metrics.sharpe)
    assert sharpe_ann(pd.Series([0.01, 0.02, -0.005])) != 0.0


def test_strategy_grid_contains_strict_and_rules():
    grid = build_strategy_grid()
    names = {s.name for s in grid}
    assert any(s.is_benchmark for s in grid)
    assert any(s.gate == "strict" and s.rank_rule == "upside_x_cred" for s in grid)
    assert any(s.rank_rule == "composite" for s in grid)
    assert "bench_tomorrow_topk_upside_k8" in names
