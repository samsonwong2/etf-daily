"""Dividend fingerprint vs G1 / unrecovered labels. Synthetic frames only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from decision_pack.scripts.scan_trend_vs_dividend import (
    LABEL_DIV,
    LABEL_OTHER,
    LABEL_TREND,
    classify_frame,
    is_bond_name,
)


def _frame(
    close: np.ndarray,
    *,
    lower: float = 1.0,
    upper: float = float(np.e),
    g7: float = 0.1,
    g8: float = 0.1,
    g9: float = 0.05,
    g10: float = 0.2,
    start: str = "2024-01-02",
) -> pd.DataFrame:
    n = len(close)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {
            "px": close,
            "fig7_g": g7,
            "fig8_g": g8,
            "fig9_g": g9,
            "fig10_g": g10,
            "fig9_upper": upper,
            "fig9_lower": lower,
        },
        index=idx,
    )


def test_bond_name() -> None:
    assert is_bond_name("科创债ETF嘉实")
    assert is_bond_name("30年国债ETF鹏扬")
    assert not is_bond_name("沪深300红利ETF建信")


def test_g1_from_midline() -> None:
    close = np.full(21, np.exp(0.49))
    close[-1] = np.exp(0.82)
    rec = classify_frame(_frame(close))
    assert rec["g1"] is True
    assert rec["label"] == LABEL_TREND


def test_g1_rejects_crawl_from_upper_area() -> None:
    close = np.full(21, np.exp(0.70))
    close[-1] = np.exp(0.85)
    rec = classify_frame(_frame(close))
    assert rec["g1"] is False


def test_g1_requires_long_g_positive() -> None:
    close = np.full(21, np.exp(0.49))
    close[-1] = np.exp(0.82)
    rec = classify_frame(_frame(close, g7=-0.01, g8=0.2))
    assert rec["g1"] is False
    assert rec["label"] != LABEL_TREND


def test_unrecovered_after_15_bars() -> None:
    close = np.full(40, np.exp(0.40))
    close[-17] = np.exp(0.90)
    close[-16:] = np.exp(0.70)
    rec = classify_frame(_frame(close, g10=0.01, g9=0.2))
    assert rec["g1"] is False
    assert rec["unrecovered"] is True
    assert rec["label"] == LABEL_TREND


def test_recent_high_inside_15_bars_is_not_unrecovered() -> None:
    close = np.full(30, np.exp(0.40))
    close[-10] = np.exp(0.90)
    close[-9:] = np.exp(0.57)
    rec = classify_frame(_frame(close))
    assert rec["unrecovered"] is False


def test_outside_streak_without_long_g_is_not_trend() -> None:
    close = np.full(30, np.exp(0.40))
    close[-5:] = np.exp(1.2)  # above upper = e
    rec = classify_frame(_frame(close, g7=-0.1, g8=-0.1))
    assert rec["outside5"] is True
    assert rec["unrecovered"] is False
    assert rec["label"] == LABEL_OTHER


def test_fingerprint_beats_neither_when_switch_off() -> None:
    n = 252 * 5
    idx_start = "2019-01-02"
    lower, upper = 1.0, 1.15
    mid = float(np.exp(0.5 * np.log(upper / lower)) * lower)
    close = np.full(n, mid)
    # Two near-upper edges per year, each a single bar back to mid.
    for year in range(5):
        for k in (40, 120):
            close[year * 252 + k] = upper * 0.995
    rec = classify_frame(
        _frame(close, lower=lower, upper=upper, g9=0.05, g10=0.04, start=idx_start)
    )
    assert rec["fingerprint"] is True
    assert rec["g1"] is False
    assert rec["unrecovered"] is False
    assert rec["label"] == LABEL_DIV


def test_open_episode_overrides_fingerprint() -> None:
    n = 252 * 5
    lower, upper = 1.0, 1.15
    mid = float(np.exp(0.5 * np.log(upper / lower)) * lower)
    close = np.full(n, mid)
    for year in range(5):
        for k in (40, 120):
            close[year * 252 + k] = upper * 0.995
    # Last high 20 bars ago, then parked above 0.50 and below 0.80.
    close[-21] = upper * 0.995
    close[-20:] = float(lower * (upper / lower) ** 0.70)
    rec = classify_frame(
        _frame(close, lower=lower, upper=upper, g9=0.05, g10=0.04, start="2019-01-02")
    )
    assert rec["fingerprint"] is True
    assert rec["unrecovered"] is True
    assert rec["label"] == LABEL_TREND
