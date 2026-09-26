"""P2 reconcile warnings."""
from __future__ import annotations

from etf_daily.lib.portfolio import HoldingRow, PortfolioSnapshot
from etf_daily.lib.reconcile import reconcile_portfolios


def _snapshot(code_weights: list[tuple[str, float]], source: str) -> PortfolioSnapshot:
    holdings = tuple(
        HoldingRow(code, code, weight, weight, weight, source)
        for code, weight in code_weights
    )
    stock_mv = sum(weight for _, weight in code_weights)
    return PortfolioSnapshot(
        as_of="2026-06-05",
        source=source,
        holdings=holdings,
        total_assets=stock_mv,
        cash_mv=0.0,
        stock_mv=stock_mv,
        equity_exposure=1.0,
    )


def test_reconcile_ok_within_threshold():
    live = _snapshot([("SH510300", 0.5), ("SH512100", 0.5)], "live")
    pipe = _snapshot([("SH510300", 0.49), ("SH512100", 0.51)], "pipeline")
    result = reconcile_portfolios(live, pipe)
    assert not result.warning


def test_reconcile_warning_on_symbol_diff():
    live = _snapshot([("SH510300", 0.7), ("SH512100", 0.3)], "live")
    pipe = _snapshot([("SH510300", 0.4), ("SH512100", 0.6)], "pipeline")
    result = reconcile_portfolios(live, pipe)
    assert result.warning
    assert result.mismatches
