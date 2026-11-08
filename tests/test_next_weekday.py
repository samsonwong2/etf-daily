"""The triggers fallback skips weekends only. It does not know exchange holidays."""

from datetime import date

from etf_daily.cli import _next_weekday


def test_next_weekday_skips_saturday_and_sunday_only():
    assert date(2026, 10, 2).weekday() == 4
    assert _next_weekday("2026-10-02") == "2026-10-05"
    assert _next_weekday("2026-10-03") == "2026-10-05"
    assert _next_weekday("2026-10-01") == "2026-10-02"
