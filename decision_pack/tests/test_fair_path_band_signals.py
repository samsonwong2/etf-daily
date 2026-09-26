"""Tests for fair_path_band_signals (synthetic, no Qlib)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from runtime_paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from decision_pack.src.fair_path_band_signals import (  # noqa: E402
    _plotly_x_to_index,
    _rr_reward_risk,
    detect_checklist_entries,
    detect_lower_band_entries,
    evaluate_checklist_row,
    find_regime_html,
    load_ohlcv_from_regime_html,
)


def test_rr_reward_risk_basic():
    rr = _rr_reward_risk(100.0, 130.0, 94.0)
    assert rr is not None
    assert rr == pytest.approx(0.30 / 0.063829, rel=1e-3)


def test_evaluate_checklist_falling_knife():
    row = pd.Series(
        {
            "px": 1.0,
            "fig10_touch_lo": True,
            "fig9_touch_lo": False,
            "fig8_gap": -0.2,
            "fig7_g": -0.1,
            "fig8_g": -0.05,
            "fig10_above_path": False,
            "fig11_above_path": False,
            "fig10_lower": 0.99,
            "fig10_prev_touch_lo": False,
            "fig8_path": 1.3,
            "fig10_path": 1.2,
            "fig10_g": -0.2,
            "fig10_touch_hi": False,
        },
        name=pd.Timestamp("2026-07-30"),
    )
    cr = evaluate_checklist_row(row, code="TEST", atr=0.08, regime="range", has_ed=True)
    assert cr.flags["B1_position"] is True
    assert cr.flags["B2_long_ok"] is False
    assert cr.verdict == "avoid_falling_knife"


def test_detect_lower_band_entries_returns_bool_series():
    n = 300
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = pd.Series(
        [float(x) for x in __import__("numpy").exp(__import__("numpy").linspace(0, 0.693, n))],
        index=idx,
    )
    from decision_pack.src.fair_path_band_signals import compute_multi_scale_frame

    ms = compute_multi_scale_frame(close)
    entries = detect_lower_band_entries(ms)
    assert entries.dtype == bool
    assert len(entries) == n


def test_detect_checklist_entries_modes():
    n = 400
    idx = pd.bdate_range("2023-01-03", periods=n)
    close = pd.Series(
        [float(x) for x in __import__("numpy").exp(__import__("numpy").linspace(0, 0.4, n))],
        index=idx,
    )
    from decision_pack.src.fair_path_band_signals import compute_multi_scale_frame

    ms = compute_multi_scale_frame(close)
    b12 = detect_checklist_entries(ms, mode="B12")
    b123 = detect_checklist_entries(ms, mode="B123")
    assert b12.dtype == bool
    assert b123.dtype == bool
    assert int(b123.sum()) <= int(b12.sum())


def test_load_ohlcv_from_regime_html_smoke():
    listing = PLOTLY_OUTPUTS_DIR / "20260821_from_listing"
    html = find_regime_html(listing, "SH513520")
    if html is None:
        pytest.skip("SH513520 html not present")
    close, ohlcv = load_ohlcv_from_regime_html(html)
    assert len(close) > 100
    assert len(ohlcv) == len(close)
    assert "$close" in ohlcv.columns
    assert close.iloc[-1] > 0


def test_plotly_x_to_index_mixed_iso_datetime_and_date():
    """Incremental HTML appends date-only last bar; history is ISO datetime."""
    idx = _plotly_x_to_index(
        [
            "2012-05-28T00:00:00.000000000",
            "2026-08-28T00:00:00.000000000",
            "2026-08-31",
        ]
    )
    assert list(idx.strftime("%Y-%m-%d")) == [
        "2012-05-28",
        "2026-08-28",
        "2026-08-31",
    ]
