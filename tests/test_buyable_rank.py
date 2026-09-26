"""Tests for buyable rank join + hard filters."""
from __future__ import annotations

import pandas as pd

from etf_daily.lib.buyable_rank import (
    BUYABLE_COLUMNS,
    BuyableRankConfig,
    build_buyable_frame,
    evaluate_buyable_row,
)
from dataclasses import replace


def test_evaluate_rejects_avoid_jump_bear():
    cfg = BuyableRankConfig()
    ok, reason = evaluate_buyable_row(
        leg="risk_leg",
        path_state="稳定",
        risk_flags="",
        jump_flags="",
        pick_score=1e-5,
        cfg=cfg,
    )
    assert ok and reason == ""

    ok, reason = evaluate_buyable_row(
        leg="avoid",
        path_state="稳定",
        risk_flags="",
        jump_flags="",
        pick_score=1e-5,
        cfg=cfg,
    )
    assert not ok and "avoid_leg" in reason

    ok, reason = evaluate_buyable_row(
        leg="risk_leg",
        path_state="破位",
        risk_flags="",
        jump_flags="",
        pick_score=1e-5,
        cfg=cfg,
    )
    assert not ok and "path_broken" in reason

    ok, reason = evaluate_buyable_row(
        leg="risk_leg",
        path_state="稳定",
        risk_flags="below_ma20|bear_stack|weak_rs",
        jump_flags="",
        pick_score=1e-5,
        cfg=cfg,
    )
    assert not ok and "bear_stack" in reason

    ok, reason = evaluate_buyable_row(
        leg="risk_leg",
        path_state="稳定",
        risk_flags="",
        jump_flags="recent_jump",
        pick_score=1e-5,
        cfg=cfg,
    )
    assert not ok and "recent_jump" in reason


def test_build_buyable_frame_filters_and_ranks():
    tomorrow = pd.DataFrame(
        [
            {
                "rank": 1,
                "code": "SH588710",
                "name": "科创半导体设备",
                "close": 3.0,
                "as_of": "2026-07-17",
                "barbell_leg": "avoid",
                "direction_bias": 0.03,
                "combined_cred": 0.005,
                "garch_跌5%": "-7%",
                "garch_涨95%": "8%",
                "upside_skew_pct": "0.8%",
                "pick_score": 2e-5,
                "equal_weight_pct": 12.5,
            },
            {
                "rank": 2,
                "code": "SH561560",
                "name": "电力ETF",
                "close": 1.25,
                "as_of": "2026-07-17",
                "barbell_leg": "risk_leg",
                "direction_bias": 0.02,
                "combined_cred": 0.006,
                "garch_跌5%": "-2%",
                "garch_涨95%": "2%",
                "upside_skew_pct": "0.1%",
                "pick_score": 6e-6,
                "equal_weight_pct": None,
            },
            {
                "rank": 3,
                "code": "SH513310",
                "name": "中韩半导体",
                "close": 4.6,
                "as_of": "2026-07-17",
                "barbell_leg": "avoid",
                "direction_bias": 0.01,
                "combined_cred": 0.002,
                "garch_跌5%": "-6%",
                "garch_涨95%": "7%",
                "upside_skew_pct": "0.7%",
                "pick_score": 8e-6,
                "equal_weight_pct": None,
            },
        ]
    )
    trend = pd.DataFrame(
        [
            {
                "code": "SH588710",
                "ma_stack": "bear",
                "risk_flags": "deep_below_ma20|bear_stack",
            },
            {"code": "SH561560", "ma_stack": "bull", "risk_flags": ""},
            {
                "code": "SH513310",
                "ma_stack": "bear",
                "risk_flags": "deep_below_ma20|bear_stack|below_ma20_3d+",
            },
        ]
    )
    pool = pd.DataFrame(
        [
            {"code": "SH588710", "leg": "avoid", "状态": "破位"},
            {"code": "SH561560", "leg": "risk_leg", "状态": "稳定"},
            {"code": "SH513310", "leg": "avoid", "状态": "破位"},
        ]
    )
    jump = pd.DataFrame(
        [
            {"code": "SH588710", "flags": "recent_jump"},
            {"code": "SH561560", "flags": ""},
            {"code": "SH513310", "flags": "recent_jump"},
        ]
    )

    frame = build_buyable_frame(tomorrow, trend, pool, jump, cfg=BuyableRankConfig(k=8))
    assert list(frame.columns) == list(BUYABLE_COLUMNS)

    by_code = frame.set_index("code")
    assert bool(by_code.loc["SH561560", "pass_filters"]) is True
    assert int(by_code.loc["SH561560", "rank"]) == 1
    assert float(by_code.loc["SH561560", "equal_weight_pct"]) == 12.5

    assert bool(by_code.loc["SH588710", "pass_filters"]) is False
    assert "avoid_leg" in str(by_code.loc["SH588710", "reject_reason"])
    assert pd.isna(by_code.loc["SH588710", "rank"]) or by_code.loc["SH588710", "rank"] is None

    assert bool(by_code.loc["SH513310", "pass_filters"]) is False
    assert "jump:recent_jump" in str(by_code.loc["SH513310", "reject_reason"]) or "avoid" in str(
        by_code.loc["SH513310", "reject_reason"]
    )


def test_missing_pick_score_rejected():
    ok, reason = evaluate_buyable_row(
        leg="risk_leg",
        path_state="稳定",
        risk_flags="",
        jump_flags="",
        pick_score=None,
        cfg=BuyableRankConfig(),
    )
    assert not ok
    assert reason == "missing_pick_score"


def test_gate_mapping_helpers_cover_rank_rules():
    from etf_daily.lib.buyable_rank import gate_to_buyable_config
    from etf_daily.lib.buyable_rank_backtest import build_strategy_grid

    assert gate_to_buyable_config("strict", k=5).k == 5
    core = gate_to_buyable_config("core_risk", k=1)
    assert core.exclude_broken_path is False
    assert "below_ma20_3d+" not in core.trend_reject_flags
    grid = build_strategy_grid(
        include_benchmarks=False,
        k_values=(2,),
        rank_rules=("upside_x_cred", "dir_x_cred", "composite"),
        gates=("strict",),
        min_creds=(0.0,),
    )
    assert {s.rank_rule for s in grid} == {"upside_x_cred", "dir_x_cred", "composite"}


def test_recompute_dir_x_cred_and_core_risk_gate():
    from etf_daily.lib.buyable_rank import (
        gate_to_buyable_config,
        recompute_tomorrow_topk_scores,
    )

    tomorrow = pd.DataFrame(
        [
            {
                "code": "SH516310",
                "name": "银行",
                "close": 1.3,
                "as_of": "2026-07-17",
                "barbell_leg": "risk_leg",
                "direction_bias": 0.04,
                "combined_cred": 0.005,
                "garch_跌5%": "-2%",
                "garch_涨95%": "2%",
                "upside_skew_pct": "0.10%",
                "pick_score": 1e-6,  # old upside score
            },
            {
                "code": "SZ159898",
                "name": "医疗",
                "close": 1.1,
                "as_of": "2026-07-17",
                "barbell_leg": "risk_leg",
                "direction_bias": 0.01,
                "combined_cred": 0.005,
                "garch_跌5%": "-2%",
                "garch_涨95%": "3%",
                "upside_skew_pct": "0.50%",
                "pick_score": 9e-6,
            },
        ]
    )
    recomputed = recompute_tomorrow_topk_scores(tomorrow, rank_rule="dir_x_cred")
    # 0.04*0.005=2e-4 > 0.01*0.005=5e-5 → bank ranks higher under dir_x_cred
    assert float(recomputed.loc[0, "pick_score"]) == 0.0002
    assert float(recomputed.loc[1, "pick_score"]) == 5e-5

    trend = pd.DataFrame(
        [
            {"code": "SH516310", "ma_stack": "bull", "risk_flags": "below_ma20_3d+"},
            {"code": "SZ159898", "ma_stack": "bull", "risk_flags": ""},
        ]
    )
    pool = pd.DataFrame(
        [
            {"code": "SH516310", "leg": "risk_leg", "状态": "破位"},
            {"code": "SZ159898", "leg": "risk_leg", "状态": "稳定"},
        ]
    )
    jump = pd.DataFrame(
        [
            {"code": "SH516310", "flags": ""},
            {"code": "SZ159898", "flags": ""},
        ]
    )
    cfg = replace(gate_to_buyable_config("core_risk", k=1), min_credibility=0.0)
    frame = build_buyable_frame(recomputed, trend, pool, jump, cfg=cfg)
    # core_risk allows 破位 + below_ma20_3d+
    by = frame.set_index("code")
    assert bool(by.loc["SH516310", "pass_filters"]) is True
    assert int(by.loc["SH516310", "rank"]) == 1
    assert float(by.loc["SH516310", "equal_weight_pct"]) == 100.0
