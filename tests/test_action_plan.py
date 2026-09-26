"""Tests for institutional action plan helpers."""
from __future__ import annotations

import pandas as pd

from etf_daily.lib.action_plan import (
    SymbolAction,
    build_action_plan,
    build_execution_rows,
    classify_ips_stage,
    classify_nuance,
    estimate_sell_shares,
    priority_score,
)
from etf_daily.lib.portfolio import HoldingRow, PortfolioSnapshot
from etf_daily.lib.symbol_trend_board import TREND_BOARD_COLUMNS
from etf_daily.lib.tier import TierDecision


def _tier_decision(**overrides) -> TierDecision:
    base = dict(
        as_of="2026-06-11",
        benchmark="SH510300",
        r_bench=-0.0069,
        dd_port=0.0562,
        vol_percentile_120d=0.85,
        table_tier=2,
        regime="structural_weakness",
        executed_tier=2,
        current_equity_exposure=1.0,
        target_equity_exposure=0.6,
        reduce_fraction=0.4,
        triggers=("dd_port>=0.0500",),
        structural_rule_hits=(),
    )
    base.update(overrides)
    return TierDecision(**base)


def _portfolio(**overrides) -> PortfolioSnapshot:
    holdings = overrides.pop(
        "holdings",
        (
            HoldingRow(
                code="SH561560",
                name="电力ETF",
                market_value=118865.7,
                weight=0.108,
                weight_total=0.108,
                source="live",
                shares=86700,
                cost_price=1.498,
                close_price=1.371,
            ),
            HoldingRow(
                code="SH588850",
                name="科创机械",
                market_value=104940.0,
                weight=0.0954,
                weight_total=0.0954,
                source="live",
                shares=55000,
                cost_price=1.997,
                close_price=1.908,
            ),
            HoldingRow(
                code="SH588710",
                name="科半导体",
                market_value=90308.9,
                weight=0.0821,
                weight_total=0.0821,
                source="live",
                shares=31900,
                cost_price=2.654,
                close_price=2.831,
            ),
        ),
    )
    base = dict(
        as_of="2026-06-11",
        source="live",
        holdings=holdings,
        total_assets=1100549.5,
        cash_mv=0.0,
        stock_mv=1100549.5,
        equity_exposure=1.0,
    )
    base.update(overrides)
    return PortfolioSnapshot(**base)


def _row(**kwargs) -> dict:
    base = {
        "code": "SH561560",
        "name": "电力ETF",
        "weight": 0.108,
        "close": 1.371,
        "dist_ma20_pct": -4.5,
        "ret_5d": -0.0799,
        "ret_20d": -0.0359,
        "ma_stack": "mixed",
        "vol_pct_120d": 0.7,
        "vol_ann_20d": 0.36,
        "vol_ann_5d": 0.34,
        "vol_ratio_5_20": 0.94,
        "vol_level": "high",
        "dd_20d_high": -0.1016,
        "pnl_pct": -8.48,
        "days_below_ma20": 5,
        "rs_20d_vs_bench": 0.0026,
        "risk_flags": "below_ma20|below_ma20_3d+|loss|pnl7|dd5_20d|mom_fade",
    }
    base.update(kwargs)
    return base


def test_estimate_sell_shares_rounds_to_lot_size():
    assert estimate_sell_shares(86700, 0.5) == 43300
    assert estimate_sell_shares(55000, 0.5) == 27500


def test_classify_nuance_break_weak_and_strong():
    assert classify_nuance("green", 0.1, "") == "intact"
    assert classify_nuance("red", -0.01, "below_ma20|weak_rs") == "break_weak"
    assert classify_nuance("red", 0.0026, "below_ma20|pnl7") == "break_weak"
    assert classify_nuance("red", 0.1074, "below_ma20|dd5_20d") == "break_strong"


def test_classify_ips_stage_exit_reduce_observe():
    assert classify_ips_stage("red", ("pnl7", "deep_below_ma20"), 1.0) == "exit"
    assert classify_ips_stage("red", ("dd5_20d", "pnl7"), 0.5) == "reduce"
    assert classify_ips_stage("red", ("dd5_20d",), 0.3) == "reduce"
    assert classify_ips_stage("red", ("dd5_20d",), 0.0) == "observe"
    assert classify_ips_stage("yellow", ("below_ma20", "weak_rs"), 0.3) == "observe"


def test_priority_score_orders_heavy_weak_above_break_strong():
    weak = priority_score(
        weight=0.108,
        suggest_reduce_pct=0.5,
        reason_count=3,
        nuance="break_weak",
    )
    strong = priority_score(
        weight=0.0954,
        suggest_reduce_pct=0.5,
        reason_count=1,
        nuance="break_strong",
    )
    assert weak > strong


def test_build_action_plan_ranks_561560_above_588850():
    trend_board = pd.DataFrame(
        [
            _row(),
            _row(
                code="SH588850",
                name="科创机械",
                weight=0.0954,
                close=1.908,
                pnl_pct=-4.46,
                dd_20d_high=-0.0578,
                days_below_ma20=1,
                rs_20d_vs_bench=0.1074,
                risk_flags="below_ma20|high_vol|loss|dd5_20d|mom_fade",
            ),
            _row(
                code="SH588710",
                name="科半导体",
                weight=0.0821,
                close=2.831,
                dist_ma20_pct=8.44,
                pnl_pct=6.67,
                dd_20d_high=0.0,
                days_below_ma20=0,
                rs_20d_vs_bench=0.227,
                risk_flags="high_vol|mom_fade",
            ),
        ],
        columns=TREND_BOARD_COLUMNS,
    )
    plan = build_action_plan(
        decision=_tier_decision(),
        portfolio=_portfolio(),
        early_warning=None,
        trend_board=trend_board,
        pipeline_weights={"SH561560": 0.08, "SH588850": 0.0736, "SH588710": 0.09},
    )
    codes = [action.code for action in plan.ranked_actions]
    assert codes.index("SH561560") < codes.index("SH588850")
    by_code = {action.code: action for action in plan.ranked_actions}
    assert by_code["SH561560"].nuance == "break_weak"
    assert by_code["SH588850"].nuance == "break_strong"
    assert by_code["SH588850"].ips_stage == "reduce"
    assert by_code["SH588850"].suggest_reduce_pct == 0.30
    assert by_code["SH561560"].suggest_reduce_pct == 0.50
    keep_codes = {action.code for action in plan.keep_watch}
    assert "SH588710" in keep_codes
    assert plan.execution_rows[0].phase == "tier_budget"
    assert any(row.code == "SH561560" for row in plan.execution_rows if row.phase == "symbol")


def test_build_execution_rows_defers_when_budget_filled():
    actions = (
        SymbolAction(
            code="SH561560",
            name="电力ETF",
            weight=0.108,
            pipeline_weight=0.08,
            alert_level="red",
            ips_stage="reduce",
            nuance="break_weak",
            suggest_reduce_pct=0.5,
            priority_score=0.05,
            est_sell_shares=43300,
            est_sell_mv=420000.0,
            rs_20d_vs_bench=0.0026,
            reasons=("pnl7", "dd5_20d"),
        ),
        SymbolAction(
            code="SH520830",
            name="沙特ETF",
            weight=0.05,
            pipeline_weight=0.04,
            alert_level="red",
            ips_stage="reduce",
            nuance="break_strong",
            suggest_reduce_pct=0.5,
            priority_score=0.03,
            est_sell_shares=10000,
            est_sell_mv=50000.0,
            rs_20d_vs_bench=0.05,
            reasons=("dd5_20d",),
        ),
    )
    rows, cumulative, deferred = build_execution_rows(
        tier_reduce_mv=440000.0,
        ranked_actions=actions,
    )
    symbol_codes = [row.code for row in rows if row.phase == "symbol"]
    assert symbol_codes == ["SH561560"]
    assert cumulative == 420000.0
    assert len(deferred) == 1
    assert deferred[0].code == "SH520830"


def test_dd5_only_above_ma20_goes_to_keep_watch_not_l3():
    trend_board = pd.DataFrame(
        [
            _row(
                code="SH515220",
                name="煤炭ETF",
                weight=0.06,
                dist_ma20_pct=1.45,
                days_below_ma20=0,
                pnl_pct=2.0,
                dd_20d_high=-0.055,
                rs_20d_vs_bench=0.1193,
                risk_flags="dd5_20d",
            ),
        ],
        columns=TREND_BOARD_COLUMNS,
    )
    plan = build_action_plan(
        decision=_tier_decision(),
        portfolio=_portfolio(
            holdings=(
                HoldingRow(
                    code="SH515220",
                    name="煤炭ETF",
                    market_value=60000.0,
                    weight=0.06,
                    weight_total=0.06,
                    source="live",
                    shares=50000,
                    cost_price=1.0,
                    close_price=1.2,
                ),
            ),
            total_assets=1000000.0,
            stock_mv=60000.0,
        ),
        early_warning=None,
        trend_board=trend_board,
        pipeline_weights={"SH515220": 0.05},
    )
    assert plan.ranked_actions == ()
    assert len(plan.keep_watch) == 1
    assert plan.keep_watch[0].code == "SH515220"
