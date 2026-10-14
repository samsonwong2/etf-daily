"""Offline tests for cluster-review single qlib read helpers."""
from __future__ import annotations

import pandas as pd

from etf_daily.pipeline.price_adjustments import auto_qfq_adjust_close_panel
from etf_daily.pool.builder.audit import clip_to_end, qfq_window, window_bounds


def _panel() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=12)
    # Jump on the last bar is outside a window that ends earlier.
    close = pd.DataFrame({"A": [10.0] * 11 + [4.0], "B": [1.0 + i * 0.01 for i in range(12)]}, index=idx)
    close.index.name = "Date"
    return close


def test_qfq_window_ignores_jump_outside_the_slice():
    close = _panel()
    start = close.index[0]
    end = close.index[8]
    window = qfq_window(close, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    direct, _ = auto_qfq_adjust_close_panel(close.loc[start:end])
    pd.testing.assert_frame_equal(window, direct)
    assert float(window["A"].iloc[0]) == 10.0
    full, _ = auto_qfq_adjust_close_panel(close)
    assert float(full["A"].iloc[0]) != 10.0


def test_clip_to_end_matches_panel_that_stopped_early():
    close = _panel()
    build_end = close.index[8].strftime("%Y-%m-%d")
    clipped = clip_to_end(close, build_end)
    stopped = close.loc[: close.index[8]]
    pd.testing.assert_frame_equal(clipped["A"].pct_change().to_frame("A"), stopped["A"].pct_change().to_frame("A"))
    assert clipped.index.max() == close.index[8]


def test_window_bounds():
    review_start, audit_start = window_bounds("2026-09-01", "2026-09-30", [5, 20, 60])
    assert audit_start == (pd.Timestamp("2026-09-01") - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    assert review_start == (
        pd.Timestamp("2026-09-30") - pd.tseries.offsets.BDay(100)
    ).strftime("%Y-%m-%d")
