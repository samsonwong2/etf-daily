"""Offline unit tests for sina ETL skip / merge / dump-mode helpers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from etf_daily.etl.sina_helpers import (
    calendars_day_exists,
    choose_dump_mode,
    csv_max_date,
    merge_ohlcv,
    polite_delay_seconds,
    should_skip_fund,
)


def test_csv_max_date_missing(tmp_path: Path):
    assert csv_max_date(tmp_path / "nope.csv") is None


def test_csv_max_date_valid(tmp_path: Path):
    path = tmp_path / "sh510300.csv"
    pd.DataFrame(
        {
            "date": ["2026-09-28", "2026-09-30", "2026-09-29"],
            "close": [1.0, 1.2, 1.1],
        }
    ).to_csv(path, index=False)
    assert csv_max_date(path) == "2026-09-30"


def test_csv_max_date_empty_file(tmp_path: Path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    assert csv_max_date(path) is None


def test_should_skip_fund():
    assert should_skip_fund("2026-09-30", "2026-09-30") is True
    assert should_skip_fund("2026-09-30", "2026-09-29") is True
    assert should_skip_fund("2026-09-29", "2026-09-30") is False
    assert should_skip_fund(None, "2026-09-30") is False


def test_merge_ohlcv_dedupe_and_sort():
    old = pd.DataFrame(
        {
            "date": ["2026-09-28", "2026-09-29"],
            "open": [1.0, 1.1],
            "close": [1.0, 1.1],
            "high": [1.0, 1.1],
            "low": [1.0, 1.1],
            "volume": [10, 20],
            "code": ["sh510300", "sh510300"],
        }
    )
    new = pd.DataFrame(
        {
            "date": ["2026-09-29", "2026-09-30"],
            "open": [9.9, 1.2],
            "close": [9.9, 1.2],
            "high": [9.9, 1.2],
            "low": [9.9, 1.2],
            "volume": [99, 30],
            "code": ["sh510300", "sh510300"],
        }
    )
    out = merge_ohlcv(old, new)
    assert list(out["date"]) == ["2026-09-28", "2026-09-29", "2026-09-30"]
    assert float(out.loc[out["date"] == "2026-09-29", "close"].iloc[0]) == 9.9


def test_merge_ohlcv_empty_new_keeps_old():
    old = pd.DataFrame({"date": ["2026-09-30"], "close": [1.0]})
    out = merge_ohlcv(old, pd.DataFrame())
    assert list(out["date"]) == ["2026-09-30"]
    assert float(out["close"].iloc[0]) == 1.0


def test_choose_dump_mode():
    assert choose_dump_mode(full=True, calendars_exist=True) == "dump_all"
    assert choose_dump_mode(full=True, calendars_exist=False) == "dump_all"
    assert choose_dump_mode(full=False, calendars_exist=False) == "dump_all"
    assert choose_dump_mode(full=False, calendars_exist=True) == "dump_update"


def test_polite_delay_seconds_disabled():
    assert polite_delay_seconds(False) == 0.0
    assert polite_delay_seconds(False, rng=object()) == 0.0


def test_polite_delay_seconds_enabled_range():
    class FixedRng:
        def randint(self, a, b):
            assert a == 1 and b == 3
            return 2

    assert polite_delay_seconds(True, rng=FixedRng()) == 2.0


def test_calendars_day_exists(tmp_path: Path):
    assert calendars_day_exists(tmp_path) is False
    cal = tmp_path / "calendars"
    cal.mkdir()
    (cal / "day.txt").write_text("2026-09-30\n", encoding="utf-8")
    assert calendars_day_exists(tmp_path) is True
