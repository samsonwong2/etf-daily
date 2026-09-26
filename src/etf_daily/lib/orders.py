"""Proportional sell orders for T+1 execution."""
from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from etf_daily.lib.portfolio import PortfolioSnapshot
from etf_daily.lib.tier import TierDecision


def build_display_only_orders(portfolio: PortfolioSnapshot) -> pd.DataFrame:
    """v0: no automatic sells; HOLD rows for audit trail."""
    rows: list[dict[str, object]] = []
    for holding in portfolio.holdings:
        rows.append(
            {
                "code": holding.code,
                "name": holding.name,
                "current_mv": holding.market_value,
                "current_weight": holding.weight,
                "target_weight": holding.weight,
                "sell_mv": 0.0,
                "sell_shares_est": None,
                "action": "HOLD",
                "reason": "display_only_v0",
            }
        )
    return pd.DataFrame(rows)


def build_orders(
    portfolio: PortfolioSnapshot,
    decision: TierDecision,
    *,
    min_sell_mv: float = 0.0,
) -> pd.DataFrame:
    """Same-ratio reduction across stock holdings when reduce_fraction > 0."""
    rows: list[dict[str, object]] = []
    reduce_fraction = float(decision.reduce_fraction)

    if reduce_fraction <= 0 or not portfolio.holdings:
        for holding in portfolio.holdings:
            rows.append(
                {
                    "code": holding.code,
                    "name": holding.name,
                    "current_mv": holding.market_value,
                    "current_weight": holding.weight,
                    "target_weight": holding.weight,
                    "sell_mv": 0.0,
                    "sell_shares_est": None,
                    "action": "HOLD",
                    "reason": "no_reduction_needed",
                }
            )
        return pd.DataFrame(rows)

    reason = f"tier{decision.executed_tier}_proportional"
    for holding in portfolio.holdings:
        sell_mv = holding.market_value * reduce_fraction
        target_mv = holding.market_value - sell_mv
        target_weight = target_mv / portfolio.stock_mv if portfolio.stock_mv > 0 else 0.0
        action = "SELL" if sell_mv >= min_sell_mv else "HOLD"
        rows.append(
            {
                "code": holding.code,
                "name": holding.name,
                "current_mv": holding.market_value,
                "current_weight": holding.weight,
                "target_weight": target_weight,
                "sell_mv": sell_mv if action == "SELL" else 0.0,
                "sell_shares_est": None,
                "action": action,
                "reason": reason if action == "SELL" else "below_min_sell_mv",
            }
        )

    return pd.DataFrame(rows)


def orders_summary(orders: pd.DataFrame, portfolio: PortfolioSnapshot) -> dict[str, float]:
    if orders.empty or portfolio.stock_mv <= 0:
        return {"total_sell_mv": 0.0, "sell_fraction_of_stock": 0.0}
    total_sell = float(orders["sell_mv"].sum())
    return {
        "total_sell_mv": total_sell,
        "sell_fraction_of_stock": total_sell / portfolio.stock_mv,
    }


def append_recovery_buy_rows(
    orders: pd.DataFrame,
    *,
    addback_fraction: float,
    portfolio: PortfolioSnapshot,
    reason: str = "recovery_addback",
) -> pd.DataFrame:
    if addback_fraction <= 0 or not portfolio.holdings:
        return orders

    buy_budget = portfolio.total_assets * addback_fraction
    if buy_budget <= 0:
        return orders

    rows = orders.to_dict(orient="records")
    for holding in portfolio.holdings:
        buy_mv = buy_budget * holding.weight
        rows.append(
            {
                "code": holding.code,
                "name": holding.name,
                "current_mv": holding.market_value,
                "current_weight": holding.weight,
                "target_weight": holding.weight,
                "sell_mv": -buy_mv,
                "sell_shares_est": None,
                "action": "BUY",
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "append_recovery_buy_rows",
    "build_display_only_orders",
    "build_orders",
    "orders_summary",
]
