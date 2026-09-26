"""Tests for extreme-drop red-triangle override."""
from __future__ import annotations

import pandas as pd
import pytest

from etf_daily.lib.extreme_drop_override import (
    SOURCE_EXTREME_DROP,
    SOURCE_QUANTILE_TAIL,
    apply_extreme_drop_override,
    coverage_split_stats,
    day_returns_from_close,
)


def test_sh588710_20260803_override_fires():
    """Regression: pred_cdf≈0.124 + day_ret≈-9.9% → effective red, not model."""
    signals = pd.DataFrame(
        [
            {
                "as_of": "2026-08-03",
                "code": "SH588710",
                "pred_cdf": 0.124483,
                "quantile_switch": False,
                "switch_side": None,
            },
            {
                "as_of": "2026-08-04",
                "code": "SH588710",
                "pred_cdf": 0.80,
                "quantile_switch": False,
                "switch_side": None,
            },
        ]
    )
    day_ret = pd.DataFrame(
        [
            {"code": "SH588710", "as_of": "2026-08-03", "day_ret": -0.099298},
            {"code": "SH588710", "as_of": "2026-08-04", "day_ret": 0.047},
        ]
    )
    out = apply_extreme_drop_override(signals, day_ret)
    row = out.loc[out["as_of"] == "2026-08-03"].iloc[0]
    assert bool(row["model_quantile_switch"]) is False
    assert bool(row["extreme_drop_switch"]) is True
    assert bool(row["quantile_switch"]) is True
    assert str(row["switch_side"]) == "down"
    assert str(row["switch_source"]) == SOURCE_EXTREME_DROP

    mild = out.loc[out["as_of"] == "2026-08-04"].iloc[0]
    assert bool(mild["extreme_drop_switch"]) is False
    assert bool(mild["quantile_switch"]) is False


def test_small_drop_does_not_override():
    signals = pd.DataFrame(
        [
            {
                "as_of": "2026-08-03",
                "code": "SH588710",
                "quantile_switch": False,
                "switch_side": None,
            }
        ]
    )
    day_ret = pd.DataFrame(
        [{"code": "SH588710", "as_of": "2026-08-03", "day_ret": -0.051}]
    )
    out = apply_extreme_drop_override(signals, day_ret)
    assert bool(out.iloc[0]["extreme_drop_switch"]) is False
    assert bool(out.iloc[0]["quantile_switch"]) is False


def test_existing_model_red_not_duplicated():
    signals = pd.DataFrame(
        [
            {
                "as_of": "2026-07-02",
                "code": "SH588710",
                "pred_cdf": 0.0089,
                "quantile_switch": True,
                "switch_side": "down",
            }
        ]
    )
    day_ret = pd.DataFrame(
        [{"code": "SH588710", "as_of": "2026-07-02", "day_ret": -0.1049}]
    )
    out = apply_extreme_drop_override(signals, day_ret)
    row = out.iloc[0]
    assert bool(row["model_quantile_switch"]) is True
    assert bool(row["extreme_drop_switch"]) is False
    assert bool(row["quantile_switch"]) is True
    assert str(row["switch_side"]) == "down"
    assert str(row["switch_source"]) == SOURCE_QUANTILE_TAIL


def test_idempotent_when_fields_present():
    signals = pd.DataFrame(
        [
            {
                "as_of": "2026-08-03",
                "code": "SH588710",
                "quantile_switch": False,
                "switch_side": None,
            }
        ]
    )
    day_ret = pd.DataFrame(
        [{"code": "SH588710", "as_of": "2026-08-03", "day_ret": -0.099}]
    )
    once = apply_extreme_drop_override(signals, day_ret)
    twice = apply_extreme_drop_override(once, day_ret)
    assert bool(twice.iloc[0]["extreme_drop_switch"]) is True
    assert bool(twice.iloc[0]["model_quantile_switch"]) is False
    assert str(twice.iloc[0]["switch_source"]) == SOURCE_EXTREME_DROP


def test_day_returns_from_close_causal():
    close = pd.Series(
        [100.0, 93.0, 95.0],
        index=pd.to_datetime(["2026-08-01", "2026-08-03", "2026-08-04"]),
    )
    ret = day_returns_from_close(close)
    assert pd.isna(ret.iloc[0])
    assert ret.iloc[1] == pytest.approx(-0.07)
    assert ret.iloc[2] == pytest.approx(95 / 93 - 1)


def test_coverage_split_stats_separates_model_and_effective():
    signals = pd.DataFrame(
        [
            {
                "as_of": "2026-08-01",
                "code": "A",
                "quantile_switch": True,
                "switch_side": "down",
            },
            {
                "as_of": "2026-08-03",
                "code": "A",
                "quantile_switch": False,
                "switch_side": None,
            },
        ]
    )
    day_ret = pd.DataFrame(
        [
            {"code": "A", "as_of": "2026-08-01", "day_ret": -0.08},
            {"code": "A", "as_of": "2026-08-03", "day_ret": -0.09},
        ]
    )
    out = apply_extreme_drop_override(signals, day_ret)
    stats = coverage_split_stats(out)
    assert stats["n_extreme_override"] == 1
    assert stats["model_switch_rate"] == pytest.approx(0.5)
    assert stats["effective_switch_rate"] == pytest.approx(1.0)
