from __future__ import annotations

import math

import pandas as pd
import pytest

from etf_daily.lib.left_side_digest import (
    LeftSideDigestConfig,
    LeftSideTopKConfig,
    compute_compression,
    compute_left_score,
    is_jump_excluded,
    select_left_topk_picks,
    transition_credibility_today,
)


def test_compute_compression():
    assert compute_compression(0.2) == 0.8
    assert compute_compression(None) is None
    assert math.isnan(compute_compression(float("nan")))


def test_compute_left_score_with_touch_down():
    score = compute_left_score(
        0.8,
        0.4,
        0.25,
        0.5,
        rank_rule="compression_x_spike_x_down",
    )
    assert score == 0.8 * 0.4 * 0.25 * 0.5


def test_compute_left_score_transition_rule():
    score = compute_left_score(
        None,
        0.4,
        0.25,
        0.5,
        rank_rule="transition_x_spike_x_down",
        transition_score=0.48,
    )
    assert score == pytest.approx(0.48 * 0.4 * 0.25 * 0.5)


def test_transition_credibility_today():
    assert transition_credibility_today(0.8, reliable=True, regime_separated=True, cal_quality=0.5) == 0.4
    assert transition_credibility_today(0.8, reliable=False, regime_separated=True, cal_quality=0.5) == 0.0


def test_compute_left_score_without_touch_down_rule():
    score = compute_left_score(
        0.7,
        0.3,
        None,
        0.4,
        rank_rule="compression_x_spike",
    )
    assert score == 0.7 * 0.3 * 0.4


def test_is_jump_excluded_live_price_gap():
    jump = pd.DataFrame(
        [{"code": "SZ159320", "flags": "live_price_gap|recent_jump"}]
    )
    assert is_jump_excluded("SZ159320", jump, exclude_jump_flags=True)
    assert not is_jump_excluded("SH510300", jump, exclude_jump_flags=True)


def test_select_left_topk_picks_filters_and_ranks():
    digest = pd.DataFrame(
        [
            {
                "code": "A",
                "name": "A",
                "close": 1.0,
                "as_of": "2026-07-16",
                "vol5_pct_120d": "20.0%",
                "p_vol_spike_h5": "40.0%",
                "触轨↓": "20.0%",
                "q05_H": "-5.0%",
                "_compression": 0.8,
                "_combined_sort": 0.5,
                "_vol5_pct": 0.2,
                "_p_spike": 0.4,
                "_p_down": 0.2,
            },
            {
                "code": "B",
                "name": "B",
                "close": 2.0,
                "as_of": "2026-07-16",
                "vol5_pct_120d": "50.0%",
                "p_vol_spike_h5": "60.0%",
                "触轨↓": "25.0%",
                "q05_H": "-6.0%",
                "_compression": 0.5,
                "_combined_sort": 0.6,
                "_vol5_pct": 0.5,
                "_p_spike": 0.6,
                "_p_down": 0.25,
            },
        ]
    )
    cluster = pd.DataFrame(
        [
            {"code": "A", "name": "A", "weight": 0.0, "barbell_leg": "risk_leg"},
            {"code": "B", "name": "B", "weight": 0.0, "barbell_leg": "avoid"},
        ]
    )
    cfg = LeftSideDigestConfig(
        topk=LeftSideTopKConfig(
            k=1,
            rank_all=True,
            max_vol5_pct=0.35,
            require_touch_down_min=0.15,
        )
    )
    topk = select_left_topk_picks(digest, cluster, digest_cfg=cfg)
    assert len(topk) == 2
    assert topk.iloc[0]["code"] == "B"
    assert topk.iloc[1]["code"] == "A"
    assert topk.iloc[0]["equal_weight_pct"] is None
    assert topk.iloc[1]["equal_weight_pct"] == 100.0


def test_select_left_topk_transition_rank_rule():
    digest = pd.DataFrame(
        [
            {
                "code": "HIGH_VOL",
                "name": "High Vol",
                "close": 1.0,
                "as_of": "2026-06-29",
                "vol5_pct_120d": "100.0%",
                "p_vol_spike_h5": "10.0%",
                "触轨↓": "20.0%",
                "q05_H": "-5.0%",
                "_compression": 0.0,
                "_combined_sort": 0.4,
                "_vol5_pct": 1.0,
                "_p_spike": 0.1,
                "_p_down": 0.2,
                "_transition_score": 0.1,
                "_transition_cred": 0.3,
            },
            {
                "code": "SZ159647",
                "name": "中药",
                "close": 2.0,
                "as_of": "2026-06-29",
                "vol5_pct_120d": "100.0%",
                "p_vol_spike_h5": "40.0%",
                "触轨↓": "25.0%",
                "q05_H": "-6.0%",
                "_compression": 0.0,
                "_combined_sort": 0.008,
                "_vol5_pct": 1.0,
                "_p_spike": 0.4,
                "_p_down": 0.25,
                "_transition_score": 0.48,
                "_transition_cred": 0.24,
            },
        ]
    )
    cluster = pd.DataFrame(
        [
            {"code": "HIGH_VOL", "name": "High Vol", "weight": 0.0, "barbell_leg": "risk_leg"},
            {"code": "SZ159647", "name": "中药", "weight": 0.0, "barbell_leg": "risk_leg"},
        ]
    )
    cfg = LeftSideDigestConfig(
        topk=LeftSideTopKConfig(
            k=1,
            rank_all=True,
            rank_rule="transition_x_spike_x_down",
            max_vol5_pct=0.35,
            require_touch_down_min=0.15,
        )
    )
    topk = select_left_topk_picks(digest, cluster, digest_cfg=cfg)
    assert topk.iloc[0]["code"] == "SZ159647"
    assert topk.iloc[1]["code"] == "HIGH_VOL"
    assert topk.iloc[0]["equal_weight_pct"] == 100.0
