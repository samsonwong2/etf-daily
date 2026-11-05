"""Qfq split-ratio snapping: round multiples keep the ex-day move."""
from __future__ import annotations

import pandas as pd
import pytest

from etf_daily.data.price_adjustments import (
    auto_qfq_adjust_close_panel,
    auto_qfq_adjust_long_frame,
)


def test_two_round_splits_compound_to_one_sixth():
    """1/3 then 1/2: history before the first split is scaled by 1/6."""
    close = pd.DataFrame({"A": [3.0, 1.0, 1.0, 0.48]})
    adj, records = auto_qfq_adjust_close_panel(close)
    assert list(records["snapped"]) == [True, True]
    assert sorted(records["split_ratio"]) == pytest.approx(sorted([1.0 / 3.0, 0.5]))
    assert adj.loc[0, "A"] == pytest.approx(3.0 / 6.0)
    # Later 1/2 scales the first ex-day; its close keeps the −4% residual vs 1/2.
    assert adj.loc[1, "A"] == pytest.approx(0.5)
    assert adj.loc[3, "A"] / adj.loc[2, "A"] - 1 == pytest.approx(0.48 / 0.5 - 1)


def test_tenth_then_ten_restores_pre_history():
    """1/10 then ×10: the cumulative factor on prices before both events is 1."""
    close = pd.DataFrame({"A": [100.0, 10.03, 10.5, 105.0]})
    adj, records = auto_qfq_adjust_close_panel(close)
    assert list(records.sort_values("datetime")["snapped"]) == [True, True]
    assert adj.loc[0, "A"] == pytest.approx(100.0)
    assert float(records.loc[records["split_ratio"] < 1, "applied_factor"].iloc[0]) == pytest.approx(1.0)


def test_non_round_cliff_keeps_close_ratio():
    """SH512200-style +170% is not within 5% of 3x, so the close ratio is used."""
    close = pd.DataFrame({"A": [0.441, 1.191]})
    adj, records = auto_qfq_adjust_close_panel(close)
    assert bool(records.iloc[0]["snapped"]) is False
    assert records.iloc[0]["split_ratio"] == pytest.approx(1.191 / 0.441)
    assert adj.loc[0, "A"] == pytest.approx(1.191)
    assert adj.loc[1, "A"] / adj.loc[0, "A"] - 1 == pytest.approx(0.0)


def test_snapped_series_is_not_rescaled():
    """A round split's leftover move is under 25%, so a second pass is a no-op."""
    close = pd.DataFrame({"A": [1.924, 0.626]})
    once, records = auto_qfq_adjust_close_panel(close)
    twice, again = auto_qfq_adjust_close_panel(once)
    assert bool(records.iloc[0]["snapped"]) is True
    assert again.empty
    pd.testing.assert_frame_equal(once, twice)
    ex_ret = float(once.loc[1, "A"] / once.loc[0, "A"] - 1)
    assert ex_ret == pytest.approx(0.626 / (1.924 / 3.0) - 1)
    assert abs(ex_ret) < 0.25


def test_open_ratio_wins_when_close_would_snap_elsewhere():
    """Open on 1/3 is used even if the close is nearer 1/2."""
    df = pd.DataFrame(
        {
            "instrument": ["SZ159320", "SZ159320"],
            "datetime": pd.to_datetime(["2026-07-14", "2026-07-15"]),
            "open": [1.867, 1.0],
            "close": [3.0, 1.56],
        }
    )
    adj, records = auto_qfq_adjust_long_frame(
        df,
        date_col="datetime",
        code_col="instrument",
        close_col="close",
        extra_price_cols=["open"],
    )
    assert bool(records.iloc[0]["snapped"]) is True
    assert records.iloc[0]["split_ratio"] == pytest.approx(1.0 / 3.0)
    assert adj["close"].iloc[0] == pytest.approx(1.0)
    assert adj["close"].iloc[1] == pytest.approx(1.56)
