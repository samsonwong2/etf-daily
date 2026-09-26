"""Unit tests for beat-or-hold q05-floor gate (no GARCH / Qlib)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from decision_pack.scripts.backtest_beat_or_hold_513650 import (
    apply_new_veto,
    apply_old_veto,
    compare_ungated,
    resolve_out_dir,
    ticket_row_from_forecast,
)


def _tickets(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=["code", "q05", "match"])


def test_ticket_row_nan_match_enters_when_q05_finite():
    row = ticket_row_from_forecast(
        "A",
        {"garch_reliable": True, "q05_20d_garch": -0.04, "match_20d_garch": float("nan")},
        horizon=20,
    )
    assert row is not None
    assert row["code"] == "A"
    assert row["q05"] == pytest.approx(-0.04)
    assert not np.isfinite(row["match"])


def test_ticket_row_nan_q05_dropped():
    assert (
        ticket_row_from_forecast(
            "A",
            {"garch_reliable": True, "q05_20d_garch": float("nan"), "match_20d_garch": 1.2},
            horizon=20,
        )
        is None
    )


def test_ticket_row_unreliable_dropped():
    assert (
        ticket_row_from_forecast(
            "A",
            {"garch_reliable": False, "q05_20d_garch": -0.01, "match_20d_garch": 1.2},
            horizon=20,
        )
        is None
    )


def test_apply_empty_returns_empty_and_nan_floor():
    empty = pd.DataFrame(columns=["code", "q05", "match"])
    old_set, old_floor = apply_old_veto(empty)
    new_set, new_floor = apply_new_veto(empty)
    assert old_set == set() and new_set == set()
    assert not np.isfinite(old_floor) and not np.isfinite(new_floor)


def test_match_below_one_old_out_new_in():
    df = _tickets(
        {"code": "WEAK", "q05": -0.02, "match": 0.5},
        {"code": "OK1", "q05": -0.03, "match": 1.1},
        {"code": "OK2", "q05": -0.04, "match": 1.2},
        {"code": "TAIL", "q05": -0.20, "match": 1.0},
    )
    old_set, _ = apply_old_veto(df)
    new_set, _ = apply_new_veto(df)
    assert "WEAK" not in old_set
    assert "WEAK" in new_set


def test_nan_match_old_out_new_in():
    df = _tickets(
        {"code": "NANM", "q05": -0.02, "match": float("nan")},
        {"code": "OK1", "q05": -0.03, "match": 1.1},
        {"code": "OK2", "q05": -0.04, "match": 1.2},
        {"code": "TAIL", "q05": -0.20, "match": 1.0},
    )
    old_set, _ = apply_old_veto(df)
    new_set, _ = apply_new_veto(df)
    assert "NANM" not in old_set
    assert "NANM" in new_set


def test_nan_match_shifts_new_floor_off_old_floor():
    df = _tickets(
        {"code": "LOW", "q05": -0.10, "match": 1.0},
        {"code": "MID", "q05": -0.04, "match": 1.0},
        {"code": "HIGH", "q05": -0.01, "match": 1.0},
        {"code": "NANM", "q05": 0.00, "match": float("nan")},
    )
    _, floor_old = apply_old_veto(df)
    _, floor_new = apply_new_veto(df)
    assert np.isfinite(floor_old) and np.isfinite(floor_new)
    assert floor_new != floor_old


def test_resolve_out_dir_refuses_published(tmp_path: Path):
    published = tmp_path / "workspace/plotly_outputs/20260810_beat_or_hold_513650"
    published.mkdir(parents=True)
    with pytest.raises(SystemExit) as exc:
        resolve_out_dir(published, as_of="2026-08-10", root=tmp_path)
    assert exc.value.code == 2


def test_resolve_out_dir_default_q05floor_suffix(tmp_path: Path):
    got = resolve_out_dir(None, as_of="2026-08-10", root=tmp_path)
    assert got.name == "20260810_beat_or_hold_513650_q05floor"


def test_compare_ungated_ok_and_drift():
    pub = pd.DataFrame(
        {
            "strategy": ["anchor_only", "switch_mom5", "switch_mom10", "switch_mom20", "blend_mom5"],
            "cum_ret": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    same = pub.copy()
    assert compare_ungated(pub, same) == []
    drifted = same.copy()
    drifted.loc[drifted["strategy"] == "switch_mom5", "cum_ret"] = 0.9
    assert compare_ungated(pub, drifted) == ["switch_mom5"]
