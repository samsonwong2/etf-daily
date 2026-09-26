"""Unit tests for CDF + return_z triangle rules."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from etf_daily.lib.triangle_cdf_z_rules import (
    CdfZRule,
    build_rule_masks,
    direction_aligned_event_stats,
    iter_cdf_z_rules,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "as_of": pd.to_datetime(
                [
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-06",
                    "2026-01-07",
                    "2026-01-08",
                    "2026-01-09",
                ]
            ),
            "code": ["SH1"] * 6,
            "pred_cdf": [0.95, 0.85, 0.05, 0.12, 0.50, 0.82],
            "return_z": [2.5, 1.2, -2.2, -0.8, 0.1, 2.0],
            "turn_kind": [
                "top_reversal",
                "none",
                "bottom_reversal",
                "none",
                "none",
                "none",
            ],
        }
    )


def test_baseline_uses_only_cdf():
    frame = _frame()
    rule = CdfZRule(mode="baseline", baseline_q=0.90, soft_q=0.80)
    alert, up, down = build_rule_masks(frame, rule)
    assert bool(up.iloc[0])  # 0.95
    assert not bool(up.iloc[1])  # 0.85 < 0.90
    assert bool(down.iloc[2])  # 0.05
    assert not bool(down.iloc[3])  # 0.12 > 0.10
    assert int(alert.sum()) == 2


def test_and_soft_requires_both_cdf_and_z():
    frame = _frame()
    rule = CdfZRule(mode="and_soft", soft_q=0.80, z_up=1.5, z_down=1.5)
    alert, up, down = build_rule_masks(frame, rule)
    # row0: cdf 0.95 & z 2.5 -> up
    assert bool(up.iloc[0])
    # row1: cdf 0.85 ok but z 1.2 < 1.5 -> no
    assert not bool(up.iloc[1])
    # row2: cdf 0.05 & z -2.2 -> down
    assert bool(down.iloc[2])
    # row3: cdf 0.12 soft-ok but z -0.8 not extreme -> no
    assert not bool(down.iloc[3])
    # row5: cdf 0.82 & z 2.0 -> up
    assert bool(up.iloc[5])
    assert int(alert.sum()) == 3


def test_or_rescue_keeps_baseline_and_adds_soft():
    frame = _frame()
    rule = CdfZRule(
        mode="or_rescue",
        baseline_q=0.90,
        soft_q=0.80,
        z_up=1.5,
        z_down=1.5,
    )
    alert, up, down = build_rule_masks(frame, rule)
    assert bool(up.iloc[0])  # baseline
    assert bool(down.iloc[2])  # baseline
    assert bool(up.iloc[5])  # soft rescue
    assert not bool(up.iloc[1])  # soft cdf but weak z
    assert int(alert.sum()) == 3


def test_asymmetric_z_thresholds():
    frame = _frame()
    rule = CdfZRule(mode="and_soft", soft_q=0.80, z_up=2.2, z_down=0.5)
    _alert, up, down = build_rule_masks(frame, rule)
    # row0 z=2.5 passes z_up=2.2
    assert bool(up.iloc[0])
    # row5 z=2.0 fails z_up=2.2
    assert not bool(up.iloc[5])
    # row3 z=-0.8 passes z_down=0.5 and cdf soft down
    assert bool(down.iloc[3])


def test_missing_return_z_raises_for_non_baseline():
    frame = _frame().drop(columns=["return_z"])
    with pytest.raises(ValueError, match="return_z"):
        build_rule_masks(frame, CdfZRule(mode="and_soft"))


def test_missing_return_z_nan_treated_as_fail():
    frame = _frame()
    frame.loc[0, "return_z"] = float("nan")
    rule = CdfZRule(mode="and_soft", soft_q=0.80, z_up=1.5, z_down=1.5)
    _alert, up, _down = build_rule_masks(frame, rule)
    assert not bool(up.iloc[0])


def test_iter_rules_includes_baseline_and_modes():
    rules = iter_cdf_z_rules(z_values=(1.0, 2.0), modes=("and_soft", "or_rescue"))
    assert rules[0].mode == "baseline"
    labels = {r.label() for r in rules}
    assert any(x.startswith("and_soft_") for x in labels)
    assert any(x.startswith("or_rescue_") for x in labels)
    # 1 baseline + 2 modes * 2 * 2 z pairs
    assert len(rules) == 1 + 2 * 2 * 2


def test_direction_aligned_recall():
    frame = _frame()
    rule = CdfZRule(mode="and_soft", soft_q=0.80, z_up=1.5, z_down=1.5)
    _alert, up, down = build_rule_masks(frame, rule)
    events = [
        SimpleNamespace(code="SH1", onset=pd.Timestamp("2026-01-02")),
        SimpleNamespace(code="SH1", onset=pd.Timestamp("2026-01-06")),
    ]
    stats = direction_aligned_event_stats(
        events=events,
        mature=frame,
        up_mask=up,
        down_mask=down,
        lead=1,
        horizon=1,
    )
    assert stats["n_top_events"] == 1
    assert stats["n_bottom_events"] == 1
    assert stats["n_top_covered_by_up"] == 1
    assert stats["n_bottom_covered_by_down"] == 1
    assert stats["direction_recall"] == 1.0
