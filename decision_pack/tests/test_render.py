"""Report rendering for institutional sections."""
from __future__ import annotations

import pandas as pd

from decision_pack.src.action_plan import ActionPlan, ExecutionRow, SymbolAction, build_action_plan
from decision_pack.src.portfolio import HoldingRow, PortfolioSnapshot
from decision_pack.src.render import render_decision_pack_report, render_t1_execution_plan
from decision_pack.src.recovery import RecoveryUpdate, RecoveryState
from decision_pack.src.symbol_trend_board import TREND_BOARD_COLUMNS, TrendBoardVolContext
from decision_pack.src.tier import MarketSnapshot, TierDecision


def _sample_board() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "code": "SH510300",
                "name": "HS300",
                "weight": 0.5,
                "close": 1.0,
                "ma5": 1.0,
                "ma10": 1.0,
                "ma20": 1.0,
                "dist_ma20_pct": -1.0,
                "ret_5d": -0.01,
                "ret_20d": -0.02,
                "ma_stack": "mixed",
                "vol_pct_120d": 0.85,
                "vol_ann_20d": 0.2,
                "vol_ann_5d": 0.25,
                "vol_ratio_5_20": 1.25,
                "vol_level": "high",
                "dd_20d_high": -0.05,
                "pnl_pct": -2.0,
                "days_below_ma20": 2,
                "rs_20d_vs_bench": -0.03,
                "risk_flags": "below_ma20|high_vol",
            }
        ],
        columns=TREND_BOARD_COLUMNS,
    )


def _empty_plan() -> ActionPlan:
    return ActionPlan(
        tier_reduce_mv=0.0,
        ranked_actions=(),
        keep_watch=(),
        allocation_gaps=pd.DataFrame(),
        execution_rows=(ExecutionRow("tier_budget", None, None, None, None, None, 0.0, "test"),),
        summary_lines=("summary",),
    )


def test_render_trend_board_section_includes_holdings_title():
    from decision_pack.src.render import render_trend_board_section

    ctx = TrendBoardVolContext(
        bench_vol_pct=0.5,
        bench_vol_ann=0.15,
        bench_vol_level="mid",
        equity_count=2,
        equity_high_vol_count=1,
        equity_below_ma20_count=1,
    )
    text = "\n".join(render_trend_board_section(_sample_board(), vol_context=ctx))
    assert "## 持仓趋势板（v0.3 · 人工决策）" in text
    assert "SH510300" in text
    assert "权益持仓 1/2" in text


def test_render_cluster_trend_board_section_includes_pool_title():
    from decision_pack.src.render import render_cluster_trend_board_section

    ctx = TrendBoardVolContext(
        bench_vol_pct=0.5,
        bench_vol_ann=0.15,
        bench_vol_level="mid",
        equity_count=35,
        equity_high_vol_count=10,
        equity_below_ma20_count=12,
    )
    text = "\n".join(render_cluster_trend_board_section(_sample_board(), vol_context=ctx))
    assert "## 待选池趋势板（v0.3 · cluster_mapping_selected · 人工决策）" in text
    assert "cluster_mapping_selected.txt" in text
    assert "待选池 10/35" in text


def test_render_t1_execution_plan_includes_est_shares():
    action = SymbolAction(
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
        est_sell_mv=59382.3,
        rs_20d_vs_bench=0.0026,
        reasons=("dd5_20d", "pnl7"),
    )
    plan = ActionPlan(
        tier_reduce_mv=440000.0,
        ranked_actions=(action,),
        keep_watch=(),
        allocation_gaps=pd.DataFrame(),
        execution_rows=(
            ExecutionRow("tier_budget", None, None, None, None, None, 440000.0, "budget"),
            ExecutionRow(
                "symbol",
                "SH561560",
                "电力ETF",
                "reduce",
                "break_weak",
                43300,
                59382.3,
                "sell",
            ),
        ),
        summary_lines=("summary",),
    )
    text = "\n".join(render_t1_execution_plan(plan, display_only=True))
    assert "43,300" in text
    assert "SH561560" in text


def test_render_t1_execution_plan_lists_deferred_symbols():
    deferred = SymbolAction(
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
    )
    plan = ActionPlan(
        tier_reduce_mv=440000.0,
        ranked_actions=(deferred,),
        keep_watch=(),
        allocation_gaps=pd.DataFrame(),
        execution_rows=(
            ExecutionRow("tier_budget", None, None, None, None, None, 440000.0, "budget"),
            ExecutionRow(
                "symbol",
                "SH561560",
                "电力ETF",
                "reduce",
                "break_weak",
                43300,
                418000.0,
                "sell",
            ),
        ),
        summary_lines=("summary",),
        t1_cumulative_mv=418000.0,
        t1_deferred_actions=(deferred,),
    )
    text = "\n".join(render_t1_execution_plan(plan, display_only=True))
    assert "T+1 累计" in text
    assert "本轮未纳入 T+1" in text
    assert "SH520830" in text


def test_institutional_report_puts_l3_before_appendix():
    decision = TierDecision(
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
    market = MarketSnapshot(
        as_of="2026-06-11",
        benchmark="SH510300",
        r_bench=-0.0069,
        bench_3d=0.0025,
        bench_5d=-0.0355,
        close_below_ma20=True,
        vol_percentile_120d=0.85,
        dd_port=0.0562,
    )
    portfolio = PortfolioSnapshot(
        as_of="2026-06-11",
        source="live",
        holdings=(
            HoldingRow(
                code="SH561560",
                name="电力ETF",
                market_value=100.0,
                weight=1.0,
                weight_total=1.0,
                source="live",
                shares=1000,
                cost_price=1.0,
                close_price=1.0,
            ),
        ),
        total_assets=100.0,
        cash_mv=0.0,
        stock_mv=100.0,
        equity_exposure=1.0,
    )
    recovery = RecoveryUpdate(
        recovery_signal=False,
        addback_fraction=0.0,
        reason="no_recovery",
        watch_items=("days_without_tier_ge_2=0/10",),
        state=RecoveryState(
            last_updated="2026-06-11",
            days_without_tier_ge_2=0,
            peak_tier=2,
            quarter_tier2_manual_count=0,
        ),
    )
    text = render_decision_pack_report(
        decision=decision,
        market=market,
        portfolio=portfolio,
        orders=pd.DataFrame(),
        reconcile=None,
        recovery=recovery,
        system_vs_manual={"manual_tier": 2},
        orders_mode="display_only",
        trend_board=_sample_board(),
        action_plan=_empty_plan(),
    )
    assert "## PM 决策摘要" in text
    assert "## L3 单标行动清单" in text
    assert "## 附录 A · 持仓趋势板" in text
    assert text.index("## L3 单标行动清单") < text.index("## 附录 A · 持仓趋势板")
