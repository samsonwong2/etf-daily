"""Unit tests for HTML-parity signal board overlay."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from decision_pack.src.execution_signal_overlay import (
    build_signal_board,
    build_symbol_signal_row,
)


def test_build_symbol_signal_row_from_trades(tmp_path: Path):
    trades = tmp_path / "trades"
    trades.mkdir()
    pd.DataFrame(
        [
            {"side": "BUY", "date": "2026-05-06", "px": 1.2},
            {"side": "SELL", "date": "2026-06-12", "px": 1.3},
        ]
    ).to_csv(trades / "SH513650_trades.csv", index=False)
    pd.DataFrame(
        [
            {"date": "2026-04-29", "kind": "enter_up", "move": 0.06},
            {"date": "2026-05-18", "kind": "enter_down", "move": -0.07},
            {"date": "2026-07-01", "kind": "leave_down", "move": 0.05},
            {"date": "2026-07-10", "kind": "leave_up", "move": -0.04},
        ]
    ).to_csv(trades / "SH513650_early_regime_marks.csv", index=False)
    pd.DataFrame(
        [
            {
                "date": "2026-05-21",
                "end": "2026-06-05",
                "level": "alert",
                "reasons": "soft_bear|peak8",
            }
        ]
    ).to_csv(trades / "SH513650_early_down_marks.csv", index=False)

    row = build_symbol_signal_row(
        code="SH513650",
        name="测试",
        method="hybrid_ma_adx",
        regime_asof="range",
        trades_dir=trades,
        window_start="2026-04-01",
        window_end="2026-08-10",
        as_of="2026-05-22",
    )
    assert "▲B05-06" in row["signals_compact"]
    assert "▼S06-12" in row["signals_compact"]
    assert "◇↑" in row["signals_compact"]
    assert "◇↓" in row["signals_compact"]
    assert "◇V" in row["signals_compact"]
    assert "◇X" in row["signals_compact"]
    assert row["ed_level"] == "alert"
    assert "⚠ED" in row["signals_compact"]


def test_build_signal_board_empty_plan():
    board = build_signal_board(
        pd.DataFrame(),
        adaptive_dir=Path("."),
        window_start="2026-04-01",
        window_end="2026-08-10",
        as_of="2026-08-07",
    )
    assert board.empty
    assert "signals_compact" in board.columns
