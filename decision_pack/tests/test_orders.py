"""P1 orders: proportional sell validation."""
from __future__ import annotations

from decision_pack.src.orders import build_orders, orders_summary
from decision_pack.src.portfolio import HoldingRow, PortfolioSnapshot
from decision_pack.src.tier import TierDecision


def _portfolio_full() -> PortfolioSnapshot:
    holdings = (
        HoldingRow("SH510300", "HS300", 0.5, 0.5, 0.5, "pipeline"),
        HoldingRow("SH512100", "ZZ1000", 0.3, 0.3, 0.3, "pipeline"),
        HoldingRow("SH515050", "CHIP", 0.2, 0.2, 0.2, "pipeline"),
    )
    return PortfolioSnapshot(
        as_of="2026-06-05",
        source="pipeline",
        holdings=holdings,
        total_assets=1.0,
        cash_mv=0.0,
        stock_mv=1.0,
        equity_exposure=1.0,
    )


def _tier1_decision() -> TierDecision:
    return TierDecision(
        as_of="2026-06-05",
        benchmark="SH510300",
        r_bench=-0.018,
        dd_port=0.032,
        vol_percentile_120d=0.65,
        table_tier=1,
        regime="sentiment_shock",
        executed_tier=1,
        current_equity_exposure=1.0,
        target_equity_exposure=0.8,
        reduce_fraction=0.2,
        triggers=("dd_port>=0.03",),
        structural_rule_hits=(),
    )


def test_proportional_sell_fraction_tier1():
    orders = build_orders(_portfolio_full(), _tier1_decision())
    summary = orders_summary(orders, _portfolio_full())
    assert abs(summary["sell_fraction_of_stock"] - 0.2) < 0.01
    for _, row in orders.iterrows():
        if row["action"] == "SELL":
            assert abs(row["sell_mv"] / row["current_mv"] - 0.2) < 1e-9


def test_hold_when_no_reduction():
    decision = TierDecision(
        as_of="2026-06-05",
        benchmark="SH510300",
        r_bench=-0.018,
        dd_port=0.032,
        vol_percentile_120d=0.65,
        table_tier=1,
        regime="sentiment_shock",
        executed_tier=1,
        current_equity_exposure=0.75,
        target_equity_exposure=0.75,
        reduce_fraction=0.0,
        triggers=("dd_port>=0.03",),
        structural_rule_hits=(),
    )
    orders = build_orders(_portfolio_full(), decision)
    assert (orders["action"] == "HOLD").all()
    assert orders["sell_mv"].sum() == 0.0
