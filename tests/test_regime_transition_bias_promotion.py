"""Tests for gate-driven bias promotion scorer and JSON IO."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from etf_daily.lib.regime_transition_bias_promotion import (
    attach_forward_returns,
    build_bias_promotion,
    eval_rule_bias,
    load_bias_promotion,
    lockbox_eligible,
    run_candidate_race,
    score_candidate,
    write_bias_promotion_outputs,
)
from etf_daily.lib.regime_transition_plot import (
    build_overlay_frame,
    causal_action_label,
)


def _panel_from_path(dates: list[str], code: str, opens: list[float], closes: list[float]):
    idx = pd.to_datetime(dates)
    open_panel = pd.DataFrame({code: opens}, index=idx)
    close_panel = pd.DataFrame({code: closes}, index=idx)
    return open_panel, close_panel


def test_eval_rule_bias_candidates():
    assert (
        eval_rule_bias(
            "baseline_switch_side",
            quantile_switch=True,
            switch_side="up",
        )
        == "buy_bias"
    )
    assert (
        eval_rule_bias(
            "baseline_switch_side",
            quantile_switch=True,
            switch_side="down",
        )
        == "sell_bias"
    )
    assert (
        eval_rule_bias(
            "prior_bias_strong",
            quantile_switch=True,
            switch_side="up",
            prior_ret=-0.05,
            thr_prior=0.01,
        )
        == "buy_bias"
    )
    assert (
        eval_rule_bias(
            "prior_bias_strong",
            quantile_switch=True,
            switch_side="up",
            prior_ret=-0.005,
            thr_prior=0.01,
        )
        is None
    )
    assert (
        eval_rule_bias(
            "trade_bias",
            quantile_switch=True,
            switch_side="up",
            trade_action="sell_bias",
        )
        == "sell_bias"
    )
    assert (
        eval_rule_bias("unknown", quantile_switch=True, switch_side="up") is None
    )


def test_score_candidate_pass_and_fail():
    # 25 buy labels with mostly positive returns → pass
    n = 25
    hits = [0.02] * 20 + [-0.01] * 5
    frame = pd.DataFrame(
        {
            "as_of": pd.date_range("2026-01-01", periods=n, freq="B"),
            "code": ["SH000001"] * n,
            "quantile_switch": [True] * n,
            "switch_side": ["up"] * n,
            "trade_label_mature": [True] * n,
            "r_t1_open_t5_close": hits,
            "trade_action": [None] * n,
            "causal_prior_ret": [None] * n,
            "causal_thr_prior": [None] * n,
        }
    )
    row = score_candidate(
        frame, rule="baseline_switch_side", side="buy_bias", window="holdout"
    )
    assert row["n_scored"] == 25
    assert row["pass"] is True
    assert row["hit_ci_low"] is not None and row["hit_ci_low"] > 0.5

    # too few scored → fail
    tiny = frame.iloc[:10].copy()
    row2 = score_candidate(
        tiny, rule="baseline_switch_side", side="buy_bias", window="holdout"
    )
    assert row2["n_scored"] == 10
    assert row2["pass"] is False


def test_score_candidate_skips_immature():
    frame = pd.DataFrame(
        {
            "as_of": pd.to_datetime(["2026-01-02", "2026-01-03"]),
            "code": ["SH000001", "SH000001"],
            "quantile_switch": [True, True],
            "switch_side": ["up", "up"],
            "trade_label_mature": [False, True],
            "r_t1_open_t5_close": [np.nan, 0.01],
        }
    )
    row = score_candidate(
        frame, rule="baseline_switch_side", side="buy_bias", window="holdout"
    )
    assert row["n_switch"] == 2
    assert row["n_scored"] == 1


def test_attach_forward_returns_precompute():
    dates = [f"2026-01-{d:02d}" for d in range(2, 20)]
    # skip weekends roughly via business days
    dates = pd.bdate_range("2026-01-02", periods=15).strftime("%Y-%m-%d").tolist()
    opens = [1.0 + 0.01 * i for i in range(len(dates))]
    closes = [1.0 + 0.01 * i + 0.005 for i in range(len(dates))]
    open_p, close_p = _panel_from_path(dates, "SH000001", opens, closes)
    switches = pd.DataFrame(
        {
            "as_of": [dates[0], dates[1]],
            "code": ["SH000001", "SH000001"],
            "quantile_switch": [True, True],
            "switch_side": ["up", "down"],
        }
    )
    out = attach_forward_returns(switches, open_p, close_p, hold_days=5)
    assert "r_t1_open_t5_close" in out.columns
    assert bool(out["trade_label_mature"].iloc[0]) is True
    assert pd.notna(out["r_t1_open_t5_close"].iloc[0])


def test_load_bias_promotion_missing_and_bad(tmp_path: Path):
    assert load_bias_promotion(tmp_path / "missing.json")["promotion_source"] == "none"
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_bias_promotion(bad)["promotion_source"] == "none"
    wrong_ver = tmp_path / "v.json"
    wrong_ver.write_text(
        json.dumps({"schema_version": 99, "promotion_source": "lockbox"}),
        encoding="utf-8",
    )
    assert load_bias_promotion(wrong_ver)["promotion_source"] == "none"


def test_build_promotion_holdout_informational_lockbox_gated(tmp_path: Path):
    # Fabricate race: holdout buy passes, lockbox ineligible → source none
    race = pd.DataFrame(
        [
            {
                "window": "holdout",
                "rule": "baseline_switch_side",
                "side": "buy_bias",
                "n_switch": 40,
                "n_scored": 40,
                "coverage": 1.0,
                "hit_rate": 0.7,
                "hit_ci_low": 0.55,
                "hit_ci_high": 0.8,
                "pass": True,
            },
            {
                "window": "holdout",
                "rule": "baseline_switch_side",
                "side": "sell_bias",
                "n_switch": 40,
                "n_scored": 40,
                "coverage": 1.0,
                "hit_rate": 0.4,
                "hit_ci_low": 0.3,
                "hit_ci_high": 0.5,
                "pass": False,
            },
            {
                "window": "lockbox",
                "rule": "baseline_switch_side",
                "side": "buy_bias",
                "n_switch": 5,
                "n_scored": 5,
                "coverage": 1.0,
                "hit_rate": 0.8,
                "hit_ci_low": 0.6,
                "hit_ci_high": 0.9,
                "pass": False,
            },
            {
                "window": "lockbox",
                "rule": "baseline_switch_side",
                "side": "sell_bias",
                "n_switch": 5,
                "n_scored": 2,
                "coverage": 0.4,
                "hit_rate": 0.5,
                "hit_ci_low": 0.1,
                "hit_ci_high": 0.9,
                "pass": False,
            },
        ]
    )
    switches = pd.DataFrame(
        {
            "as_of": pd.bdate_range("2026-07-21", periods=5),
            "trade_label_mature": [True] * 5,
            "r_t1_open_t5_close": [0.01] * 5,
        }
    )
    promo = build_bias_promotion(
        race,
        validation_dir=tmp_path,
        holdout_months=["2026-02", "2026-03"],
        lockbox_start="2026-07-21",
        switches_with_returns=switches,
    )
    assert promo["promotion_source"] == "none"
    assert promo["buy_rule"] is None
    assert promo["holdout_preview_buy_rule"] == "baseline_switch_side"
    assert promo["lockbox"]["eligible"] is False

    race_path, promo_path = write_bias_promotion_outputs(race, promo, tmp_path)
    assert race_path.exists() and promo_path.exists()
    loaded = load_bias_promotion(promo_path)
    assert loaded["promotion_source"] == "none"


def test_lockbox_eligible_requires_span_and_n():
    switches = pd.DataFrame(
        {
            "as_of": pd.bdate_range("2026-07-21", periods=25),
            "trade_label_mature": [True] * 25,
        }
    )
    race = pd.DataFrame(
        [
            {
                "window": "lockbox",
                "rule": "baseline_switch_side",
                "side": "buy_bias",
                "n_scored": 25,
                "pass": False,
            }
        ]
    )
    assert lockbox_eligible(
        switches, lockbox_start="2026-07-21", race=race
    )


def test_causal_action_label_promotion_sides():
    assert (
        causal_action_label(quantile_switch=True, switch_side="up")
        == "上行分位切换"
    )
    promo = {
        "promotion_source": "lockbox",
        "buy_pass_gate": True,
        "sell_pass_gate": False,
        "buy_rule": "baseline_switch_side",
        "sell_rule": None,
    }
    assert (
        causal_action_label(
            quantile_switch=True,
            switch_side="up",
            promotion=promo,
        )
        == "上行分位切换(偏买)"
    )
    assert (
        causal_action_label(
            quantile_switch=True,
            switch_side="down",
            promotion=promo,
        )
        == "下行分位切换"
    )
    # conflict → observe
    both = {
        "promotion_source": "lockbox",
        "buy_pass_gate": True,
        "sell_pass_gate": True,
        "buy_rule": "baseline_switch_side",
        "sell_rule": "prior_bias",  # down with negative prior → buy; conflict possible
    }
    # up + baseline → buy; prior_bias with prior_ret=-0.05 → buy; only buy → ok
    assert (
        causal_action_label(
            quantile_switch=True,
            switch_side="up",
            prior_ret=-0.05,
            thr_prior=0.01,
            promotion=both,
        )
        == "上行分位切换(偏买)"
    )


def test_build_overlay_respects_promotion():
    oos = pd.DataFrame(
        {
            "as_of": ["2026-06-09"],
            "code": ["SH588710"],
            "quantile_switch": [True],
            "switch_side": ["up"],
            "causal_prior_ret": [-0.05],
            "causal_thr_prior": [0.01],
        }
    )
    rev = oos.copy()
    rev["turn_kind"] = "none"
    rev["jump_role"] = "none"
    rev["label_switch_reversal"] = False
    evt = oos.copy()
    evt["matched_event_onset"] = pd.NaT
    evt["is_duplicate"] = False
    evt["is_fp"] = False
    out = build_overlay_frame(
        oos,
        rev,
        evt,
        code="SH588710",
        start_date="2026-06-01",
        end_date="2026-06-30",
    )
    assert out.iloc[0]["action_label"] == "上行分位切换"
    assert "偏买" not in out.iloc[0]["action_label"]
    out2 = build_overlay_frame(
        oos,
        rev,
        evt,
        code="SH588710",
        start_date="2026-06-01",
        end_date="2026-06-30",
        promotion={
            "promotion_source": "lockbox",
            "buy_pass_gate": True,
            "sell_pass_gate": False,
            "buy_rule": "baseline_switch_side",
            "sell_rule": None,
        },
    )
    assert out2.iloc[0]["action_label"] == "上行分位切换(偏买)"
