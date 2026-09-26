from __future__ import annotations

from pathlib import Path

from etf_daily.lib.portfolio import HoldingRow, PortfolioSnapshot, load_live_holdings_csv


def test_load_live_holdings_csv_empty_file_means_flat_position(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("code,name,shares,price\n", encoding="utf-8")

    snapshot = load_live_holdings_csv(path, as_of="2026-07-17")

    assert snapshot.source == "live"
    assert snapshot.holdings == ()
    assert snapshot.stock_mv == 0.0
    assert snapshot.cash_mv == 0.0
    assert snapshot.total_assets == 0.0
    assert snapshot.equity_exposure == 0.0


def test_load_live_holdings_csv_empty_file_respects_total_assets_override(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("code,name,shares,price\n", encoding="utf-8")

    snapshot = load_live_holdings_csv(
        path,
        as_of="2026-07-17",
        total_assets_override=1_000_000.0,
    )

    assert snapshot.total_assets == 1_000_000.0
    assert snapshot.equity_exposure == 0.0


def test_portfolio_snapshot_to_frame_has_columns_when_empty() -> None:
    snapshot = PortfolioSnapshot(
        as_of="2026-07-17",
        source="live",
        holdings=(),
        total_assets=0.0,
        cash_mv=0.0,
        stock_mv=0.0,
        equity_exposure=0.0,
    )
    frame = snapshot.to_frame()

    assert frame.columns.tolist() == list(HoldingRow.__dataclass_fields__.keys())
    assert frame.empty
