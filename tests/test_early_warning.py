"""Early warning classification for per-symbol alerts."""
from __future__ import annotations

import pandas as pd

from etf_daily.lib.early_warning import (
    build_early_warning_frame,
    classify_symbol_alert,
    parse_risk_flags,
)


def test_parse_risk_flags_splits_pipe():
    assert parse_risk_flags("below_ma20|weak_rs") == {"below_ma20", "weak_rs"}


def test_yellow_requires_two_signals():
    row = {
        "risk_flags": "below_ma20|weak_rs",
        "ret_5d": 0.01,
        "ret_20d": 0.02,
        "vol_ratio_5_20": 1.0,
        "weight": 0.08,
    }
    alert = classify_symbol_alert(row, executed_tier=2)
    assert alert.level == "yellow"
    assert alert.suggest_reduce_pct == 0.30


def test_yellow_zero_suggest_when_executed_tier_below_threshold():
    row = {
        "risk_flags": "below_ma20|weak_rs",
        "ret_5d": 0.01,
        "ret_20d": 0.02,
        "vol_ratio_5_20": 1.0,
        "weight": 0.08,
    }
    calm = classify_symbol_alert(row, executed_tier=1)
    stress = classify_symbol_alert(row, executed_tier=2)
    assert calm.level == "yellow"
    assert calm.suggest_reduce_pct == 0.0
    assert stress.suggest_reduce_pct == 0.30


def test_red_soft_single_reason_suggests_thirty_pct():
    row = {
        "risk_flags": "below_ma20|below_ma20_3d+|loss",
        "weight": 0.08,
        "pnl_pct": -2.0,
        "days_below_ma20": 5,
    }
    alert = classify_symbol_alert(row)
    assert alert.level == "red"
    assert alert.reasons == ("below_ma20_3d+",)
    assert alert.suggest_reduce_pct == 0.30


def test_red_hard_multi_reason_or_pnl7_suggests_fifty_pct():
    multi = classify_symbol_alert(
        {
            "risk_flags": "below_ma20|below_ma20_3d+|dd5_20d|loss",
            "weight": 0.08,
            "pnl_pct": -3.0,
            "dd_20d_high": -0.06,
            "days_below_ma20": 5,
        }
    )
    pnl_only = classify_symbol_alert(
        {
            "risk_flags": "below_ma20|pnl7|loss",
            "pnl_pct": -8.0,
            "weight": 0.08,
        }
    )
    assert multi.suggest_reduce_pct == 0.50
    assert pnl_only.suggest_reduce_pct == 0.50


def test_dd5_only_above_ma20_is_observe_not_reduce():
    row = {
        "risk_flags": "dd5_20d",
        "weight": 0.08,
        "pnl_pct": 3.0,
        "dd_20d_high": -0.06,
        "dist_ma20_pct": 1.5,
        "days_below_ma20": 0,
    }
    alert = classify_symbol_alert(row)
    assert alert.level == "green"
    assert alert.suggest_reduce_pct == 0.0
    assert alert.reasons == ()


def test_red_on_pnl7():
    row = {
        "risk_flags": "below_ma20|pnl7|loss",
        "pnl_pct": -8.0,
        "weight": 0.08,
    }
    alert = classify_symbol_alert(row)
    assert alert.level == "red"
    assert "pnl7" in alert.reasons
    assert alert.suggest_reduce_pct == 0.50


def test_red_critical_pnl7_and_deep_below():
    row = {
        "risk_flags": "deep_below_ma20|pnl7|below_ma20",
        "pnl_pct": -9.0,
        "weight": 0.08,
    }
    alert = classify_symbol_alert(row)
    assert alert.level == "red"
    assert alert.suggest_reduce_pct == 1.0


def test_pipeline_zero_held_triggers_red():
    row = {
        "risk_flags": "below_ma20",
        "weight": 0.05,
        "pnl_pct": -1.0,
    }
    alert = classify_symbol_alert(row, pipeline_weight=0.0)
    assert alert.level == "red"
    assert "pipeline_zero_held" in alert.reasons


def test_pipeline_weight_missing_does_not_trigger_zero_held_red():
    row = {
        "risk_flags": "mom_fade",
        "weight": 0.08,
        "pnl_pct": 3.0,
        "dd_20d_high": 0.0,
        "dist_ma20_pct": 6.0,
    }
    alert_absent = classify_symbol_alert(row, pipeline_weight=None)
    alert_zero = classify_symbol_alert(row, pipeline_weight=0.0)
    assert alert_absent.level == "green"
    assert alert_zero.level == "red"
    assert "pipeline_zero_held" in alert_zero.reasons


def test_build_early_warning_frame_skips_green_and_zero_weight():
    trend = pd.DataFrame(
        [
            {
                "code": "SH510300",
                "name": "HS300",
                "weight": 0.10,
                "risk_flags": "below_ma20|weak_rs|bear_stack",
                "pnl_pct": -1.0,
                "dd_20d_high": -0.02,
                "dist_ma20_pct": -1.0,
                "ret_5d": -0.01,
                "ret_20d": 0.01,
                "vol_ratio_5_20": 1.0,
            },
            {
                "code": "SH561560",
                "name": "POWER",
                "weight": 0.0,
                "risk_flags": "below_ma20|weak_rs|bear_stack",
                "pnl_pct": -10.0,
                "dd_20d_high": -0.10,
                "dist_ma20_pct": -6.0,
                "ret_5d": -0.02,
                "ret_20d": 0.01,
                "vol_ratio_5_20": 1.0,
            },
        ]
    )
    frame = build_early_warning_frame(
        trend,
        None,
        pipeline_weights={"SH510300": 0.08},
        executed_tier=2,
    )
    assert len(frame) == 1
    assert frame.iloc[0]["code"] == "SH510300"
    assert frame.iloc[0]["alert_level"] == "yellow"
