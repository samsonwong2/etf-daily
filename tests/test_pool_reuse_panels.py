"""Offline tests for pool panel reuse (no qlib)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from etf_daily.pipeline.price_adjustments import auto_qfq_adjust_close_panel
from etf_daily.pool.builder.clustering import pairwise_overlap_matrices, subset_pairwise
from etf_daily.pool.builder.data_loading import (
    panel_has_codes,
    should_reinit_qlib,
    slice_loaded_codes,
)


def _returns() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=40)
    frame = pd.DataFrame(rng.normal(size=(40, 4)), index=idx, columns=list("ABCD"))
    # Short overlap so min_periods fillna is exercised on D vs the others.
    frame.iloc[:32, frame.columns.get_loc("D")] = np.nan
    return frame


def test_subset_pairwise_matches_fresh_corr():
    returns = _returns()
    corr, overlap, _ = pairwise_overlap_matrices(returns, min_overlap_days=5, low_overlap_distance=0.95)
    sub_corr, sub_overlap = subset_pairwise(corr, overlap, ["C", "A", "D"])
    fresh_corr, fresh_overlap, _ = pairwise_overlap_matrices(
        returns[["C", "A", "D"]],
        min_overlap_days=5,
        low_overlap_distance=0.95,
    )
    pd.testing.assert_frame_equal(sub_corr, fresh_corr)
    pd.testing.assert_frame_equal(sub_overlap, fresh_overlap)
    assert list(sub_corr.index) == ["C", "A", "D"]
    assert float(sub_corr.loc["C", "C"]) == 1.0


def test_panel_has_codes():
    assert panel_has_codes(["SH510300", "SZ159915"], ["SZ159915", "SH510300"]) is True
    assert panel_has_codes(["SH510300"], ["SH510300", "SZ159915"]) is False
    assert panel_has_codes(["SH510300"], []) is True


def test_slice_loaded_codes_keeps_loaded_order_and_omits_qlib_gaps():
    loaded = ["SH510300", "SZ159915", "SH518880"]
    first = ["SH510300", "SZ159915", "SH518880", "SH511010"]
    # SH511010 was requested first and qlib omitted it. SZ159915 is filtered out.
    sliced = slice_loaded_codes(loaded, ["SH518880", "SH510300", "SH511010"], first)
    assert sliced == ["SH510300", "SH518880"]


def test_slice_loaded_codes_falls_back_when_code_was_never_requested():
    loaded = ["SH510300", "SZ159915"]
    assert slice_loaded_codes(loaded, ["SH510300", "SH999999"], ["SH510300", "SZ159915"]) is None


def test_should_reinit_qlib():
    assert should_reinit_qlib(None, "/data/qlib") is True
    assert should_reinit_qlib("/data/other", "/data/qlib") is True
    assert should_reinit_qlib("/data/qlib", "/data/qlib") is False


def test_qfq_slice_matches_adjusting_the_slice():
    close = pd.DataFrame(
        {
            "A": [10.0, 10.0, 4.0],
            "B": [1.0, 1.1, 1.2],
        }
    )
    full, _ = auto_qfq_adjust_close_panel(close)
    sliced, _ = auto_qfq_adjust_close_panel(close[["A"]])
    pd.testing.assert_frame_equal(full[["A"]], sliced)
    assert float(full.loc[0, "A"]) != 10.0
