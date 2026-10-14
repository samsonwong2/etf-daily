"""Offline tests for regime day-already-scored skip and qfq idempotence."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from etf_daily.pipeline.price_adjustments import auto_qfq_adjust_close_panel
from etf_daily.regime.backtest_regime_transition_signals import day_already_scored


def _write(path: Path, dates: list[str]) -> None:
    pd.DataFrame({"as_of": dates, "code": ["SH510300"] * len(dates)}).to_csv(path, index=False)


def test_day_already_scored_when_both_cover(tmp_path: Path):
    shard = tmp_path / "2026-09.signals.csv"
    signals = tmp_path / "signals_oos.csv"
    _write(shard, ["2026-09-29", "2026-09-30"])
    _write(signals, ["2026-09-30"])
    assert day_already_scored(shard, signals, "2026-09-30") is True
    assert day_already_scored(shard, signals, "2026-09-29") is True


def test_day_already_scored_false_when_stale_or_missing(tmp_path: Path):
    shard = tmp_path / "2026-09.signals.csv"
    signals = tmp_path / "signals_oos.csv"
    assert day_already_scored(shard, signals, "2026-09-30") is False
    _write(shard, ["2026-09-29"])
    _write(signals, ["2026-09-29"])
    assert day_already_scored(shard, signals, "2026-09-30") is False
    empty = tmp_path / "empty.csv"
    empty.write_text("as_of,code\n", encoding="utf-8")
    assert day_already_scored(empty, signals, "2026-09-29") is False


def test_qfq_twice_is_identity():
    close = pd.DataFrame({"A": [10.0, 10.0, 4.0, 4.0]})
    once, _ = auto_qfq_adjust_close_panel(close)
    twice, _ = auto_qfq_adjust_close_panel(once)
    pd.testing.assert_frame_equal(once, twice)
    assert float(once.loc[0, "A"]) != 10.0
