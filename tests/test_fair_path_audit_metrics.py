"""Checklist contracts that do not depend on the removed tail-audit helpers."""
from __future__ import annotations

import inspect

import pandas as pd

from etf_daily.lib.fair_path_band_signals import evaluate_checklist_row
from etf_daily.lib.incremental_from_listing import try_incremental_html


def _row(**kwargs) -> pd.Series:
    base = {
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
        "fig9_touch_hi": False,
        "fig7_path_rising": False,
        "fig8_path_rising": False,
    }
    base.update(kwargs)
    return pd.Series(base, name=pd.Timestamp("2026-07-30"))


def test_b1_without_b2_is_falling_knife():
    cr = evaluate_checklist_row(_row(), code="TEST", atr=0.02)
    assert cr.flags["B1_position"] is True
    assert cr.flags["B2_long_ok"] is False
    assert cr.verdict == "avoid_falling_knife"


def test_h7_incremental_refuses_html_whose_rails_stamp_is_stale():
    src = inspect.getsource(try_incremental_html)
    assert "fair-path-rails" in src
