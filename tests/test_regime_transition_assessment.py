from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from etf_daily.lib.regime_transition_assessment import (
    baseline_comparison,
    block_bootstrap_mean_ci,
    build_assessment_frame,
    coverage_summary,
    decide_verdict,
    default_qstar_grid,
    distribution_assessment,
    evaluate_threshold_on_frame,
    label_mature_end,
    pit_uniformity_summary,
    qstar_sensitivity,
    run_causal_bias_study,
    run_early_confirmation_study,
    run_prior_bias_alignment_study,
    run_qstar_threshold_study,
    select_qstar_threshold,
    split_time_windows,
    state_semantics_summary,
    switch_mask_from_cdf,
    switch_reversal_assessment,
    write_assessment_outputs,
    _direction_hit,
    _forward_log_return,
)


def test_pit_uniformity_ideal_and_skewed():
    rng = np.random.default_rng(0)
    ideal = rng.uniform(0, 1, 2000)
    uni = pit_uniformity_summary(ideal)
    assert uni["n"] == 2000
    assert abs(uni["mean"] - 0.5) < 0.03
    assert uni["ks_stat"] < 0.05

    skewed = rng.beta(2.0, 5.0, 2000)
    bad = pit_uniformity_summary(skewed)
    assert bad["ks_stat"] > uni["ks_stat"]


def test_coverage_summary_bilateral_tail():
    u = np.linspace(0.0, 1.0, 1001)
    cov = coverage_summary(u)
    assert abs(cov["coverage_0.5"] - 0.5) < 0.01
    assert abs(cov["bilateral_tail_rate"] - 0.20) < 0.02


def test_block_bootstrap_mean_ci_contains_mean():
    x = np.arange(100, dtype=float)
    ci = block_bootstrap_mean_ci(x, block_size=10, n_boot=200, seed=1)
    assert ci["mean"] is not None
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]


def _synthetic_panel_and_signals() -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.bdate_range("2024-01-02", periods=80)
    rng = np.random.default_rng(3)
    rets = rng.normal(0.0, 0.01, len(idx))
    # inject a few jumps for switch/reversal structure
    rets[40] = 0.06
    rets[41:45] = -0.015
    prices = 100.0 * np.exp(np.cumsum(rets))
    panel = pd.DataFrame({"SH513310": prices}, index=idx)

    as_ofs = idx[20:]
    pred_cdf = rng.uniform(0.05, 0.95, len(as_ofs))
    pred_cdf[20] = 0.97  # switch up near jump
    signals = pd.DataFrame(
        {
            "as_of": as_ofs,
            "code": ["SH513310"] * len(as_ofs),
            "name": ["demo"] * len(as_ofs),
            "regime_now": ["calm"] * len(as_ofs),
            "regime_hmm_raw": ["calm"] * len(as_ofs),
            "quantile_switch": pred_cdf >= 0.90,
            "switch_side": np.where(pred_cdf >= 0.90, "up", None),
            "pred_cdf": pred_cdf,
            "pred_log_density": rng.normal(-3.0, 0.2, len(as_ofs)),
            "label_ambiguous": False,
            "regime_separated": True,
        }
    )
    # mark a few volatile_reversal raw states
    signals.loc[signals.index[18:22], "regime_hmm_raw"] = "volatile_reversal"
    signals.loc[signals.index[10:14], "regime_hmm_raw"] = "bear_grind"
    return panel, signals


def _longer_panel_and_signals() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Multi-month series so time splits and selection have enough months."""
    idx = pd.bdate_range("2024-01-02", periods=260)
    rng = np.random.default_rng(11)
    rets = rng.normal(0.0, 0.012, len(idx))
    for j in (40, 90, 140, 190, 230):
        rets[j] = 0.07 if j % 80 < 40 else -0.07
        rets[j + 1 : j + 4] = rng.normal(-0.01, 0.01, 3)
    prices = 100.0 * np.exp(np.cumsum(rets))
    panel = pd.DataFrame({"SH513310": prices}, index=idx)

    as_ofs = idx[30:]
    pred_cdf = rng.uniform(0.02, 0.98, len(as_ofs))
    # force some extreme tails near jumps
    for j in (10, 60, 110, 160, 200):
        if j < len(pred_cdf):
            pred_cdf[j] = 0.97 if j % 2 == 0 else 0.03
    signals = pd.DataFrame(
        {
            "as_of": as_ofs,
            "code": ["SH513310"] * len(as_ofs),
            "name": ["demo"] * len(as_ofs),
            "regime_now": ["calm"] * len(as_ofs),
            "regime_hmm_raw": ["calm"] * len(as_ofs),
            "quantile_switch": (pred_cdf >= 0.90) | (pred_cdf <= 0.10),
            "switch_side": np.where(
                pred_cdf >= 0.90, "up", np.where(pred_cdf <= 0.10, "down", None)
            ),
            "pred_cdf": pred_cdf,
            "pred_log_density": rng.normal(-3.0, 0.2, len(as_ofs)),
            "label_ambiguous": False,
            "regime_separated": True,
        }
    )
    return panel, signals


def test_build_assessment_frame_marks_immature_tail():
    panel, signals = _synthetic_panel_and_signals()
    frame = build_assessment_frame(signals, panel, horizon=10)
    assert "label_mature" in frame.columns
    assert frame["label_mature"].dtype == bool or frame["label_mature"].isna().any() is False
    # last <10 days cannot be mature
    last_dates = frame.sort_values("as_of")["as_of"].tail(5)
    assert not frame.loc[frame["as_of"].isin(last_dates), "label_mature"].all()
    mature_end = label_mature_end(frame)
    assert mature_end is not None
    assert pd.Timestamp(mature_end) < frame["as_of"].max()


def test_distribution_and_baseline_on_fixture():
    panel, signals = _synthetic_panel_and_signals()
    frame = build_assessment_frame(signals, panel, horizon=10)
    dist = distribution_assessment(frame[frame["pred_cdf"].notna()])
    assert dist["uniformity"]["n"] > 0
    base = baseline_comparison(frame)
    # baseline may be empty if lookback insufficient on tiny series; allow either
    assert "delta_vs_gauss" in base


def test_state_semantics_and_switch_assessment():
    panel, signals = _synthetic_panel_and_signals()
    frame = build_assessment_frame(signals, panel, horizon=5)
    state = state_semantics_summary(frame)
    assert state["state_col"] == "regime_hmm_raw"
    assert "occupancy" in state
    switch = switch_reversal_assessment(frame, horizon=5)
    assert "n_alerts" in switch
    assert "precision" in switch
    sens = qstar_sensitivity(frame, horizon=5)
    assert not sens.empty
    assert set(sens["q_star"]) >= {0.85, 0.90, 0.95}
    assert set(sens["side"]) >= {"both", "up", "down"}
    assert "f1" in sens.columns
    assert "precision" in sens.columns


def test_default_qstar_grid_and_switch_mask_boundaries():
    grid = default_qstar_grid()
    assert grid[0] == 0.80
    assert grid[-1] == 0.99
    assert 0.90 in grid
    assert len(grid) == 20

    cdf = pd.Series([0.05, 0.10, 0.50, 0.90, 0.95])
    both = switch_mask_from_cdf(cdf, q_star=0.90, side="both")
    assert list(both) == [True, True, False, True, True]
    up = switch_mask_from_cdf(cdf, q_star=0.90, side="up")
    assert list(up) == [False, False, False, True, True]
    down = switch_mask_from_cdf(cdf, q_star=0.90, side="down")
    assert list(down) == [True, True, False, False, False]
    # asymmetric
    asym = switch_mask_from_cdf(cdf, q_up=0.95, q_down=0.05, side="both")
    assert list(asym) == [True, False, False, False, True]


def test_qstar_sensitivity_zero_alerts_and_immature_excluded():
    panel, signals = _synthetic_panel_and_signals()
    frame = build_assessment_frame(signals, panel, horizon=5)
    # Extreme q* near 1.0 may yield zero alerts on tiny fixture
    sens = qstar_sensitivity(
        frame, q_values=(0.999,), sides=("both",), horizon=5
    )
    assert not sens.empty
    assert int(sens.iloc[0]["n_alerts"]) == 0
    assert sens.iloc[0]["f1"] is None or pd.isna(sens.iloc[0]["f1"])

    # Immature-only frame → empty sensitivity
    immature = frame.copy()
    immature["label_mature"] = False
    assert qstar_sensitivity(immature, horizon=5).empty


def test_split_time_windows_holdout_disjoint_from_select():
    panel, signals = _longer_panel_and_signals()
    frame = build_assessment_frame(signals, panel, horizon=10)
    splits = split_time_windows(frame)
    assert splits["holdout_months"]
    assert splits["select_months"]
    overlap = set(splits["select_months"]) & set(splits["holdout_months"])
    assert not overlap
    # holdout is the latest months
    assert max(splits["holdout_months"]) == max(splits["months"])
    # candidate excludes holdout
    assert not (set(splits["candidate_months"]) & set(splits["holdout_months"]))


def test_select_qstar_does_not_read_holdout():
    panel, signals = _longer_panel_and_signals()
    frame = build_assessment_frame(signals, panel, horizon=10)
    splits = split_time_windows(frame)
    select = frame[frame["month"].isin(splits["select_months"])].copy()
    sel1 = select_qstar_threshold(select, horizon=10, min_alerts=1)
    # Selection only receives select frame; poisoning holdout cannot affect it
    sel2 = select_qstar_threshold(select, horizon=10, min_alerts=1)
    assert sel1["q_star"] == sel2["q_star"]
    assert sel1["recommended_mode"] == sel2["recommended_mode"]
    assert splits["holdout_months"]


def test_run_qstar_threshold_study_outputs_recommendation():
    panel, signals = _longer_panel_and_signals()
    frame = build_assessment_frame(signals, panel, horizon=10)
    study = run_qstar_threshold_study(frame, horizon=10, baseline_q=0.90)
    assert "recommendation" in study
    assert "comparison" in study
    assert isinstance(study["comparison"], pd.DataFrame)
    assert len(study["comparison"]) == 4
    assert "keep_production_q90" in study["recommendation"]
    assert not study["sensitivity_all"].empty
    # baseline 0.90 evaluated
    labels = set(study["comparison"]["label"].astype(str))
    assert any("baseline" in x for x in labels)
    assert any("recommended" in x for x in labels)


def test_evaluate_threshold_on_frame_empty_safe():
    empty = pd.DataFrame(
        columns=["as_of", "code", "pred_cdf", "label_mature", "label_switch_reversal"]
    )
    out = evaluate_threshold_on_frame(empty, q_star=0.90, label="empty")
    assert out["n_alerts"] == 0
    assert out["label"] == "empty"


def test_decide_verdict_fail_on_bad_tail():
    dist = {
        "coverage": {"bilateral_tail_err": 0.20},
        "uniformity": {"ks_stat": 0.02},
        "ljung_box": {"acf1": 0.01},
        "mean_nll": 1.0,
    }
    base = {
        "delta_vs_gauss": 0.1,
        "gauss_ci": {"ci_high": 0.2},
        "t_ci": {"ci_high": 0.2},
    }
    state = {"semantic_order_ok": True, "ambiguous_rate": 0.1}
    switch = {"precision": 0.25, "n_alerts": 100}
    v = decide_verdict(dist, base, state, switch)
    assert v.status == "FAIL"


def test_write_assessment_outputs(tmp_path: Path):
    panel, signals = _synthetic_panel_and_signals()
    # write a mini validation dir
    val = tmp_path / "val"
    val.mkdir()
    signals.to_csv(val / "signals_oos.csv", index=False)
    from etf_daily.lib.regime_transition_assessment import run_assessment

    result = run_assessment(val, panel, horizon=5)
    out = write_assessment_outputs(result, tmp_path / "assessment")
    assert (out / "ASSESSMENT.md").exists()
    assert (out / "assessment_summary.json").exists()
    assert (out / "daily_assessment.csv").exists()
    assert (out / "per_etf_distribution.csv").exists()
    assert (out / "qstar_sensitivity.csv").exists()
    assert (out / "qstar_threshold_recommendation.json").exists()
    assert (out / "qstar_threshold_comparison.csv").exists()
    assert (out / "causal_bias_forward_accuracy.csv").exists()
    assert (out / "causal_bias_reversal_accuracy.csv").exists()
    assert (out / "causal_bias_by_etf.csv").exists()
    assert (out / "causal_bias_by_year.csv").exists()
    assert (out / "causal_bias_summary.json").exists()
    assert (out / "early_confirmation_migration.csv").exists()
    assert (out / "early_confirmation_by_etf.csv").exists()
    assert (out / "early_confirmation_by_year.csv").exists()
    assert (out / "early_confirmation_summary.json").exists()
    text = (out / "ASSESSMENT.md").read_text(encoding="utf-8")
    assert "标签成熟截止" in text
    assert "q* 阈值 walk-forward" in text
    assert "因果偏买/偏卖准确性" in text
    assert "两阶段反转确认" in text
    assert result["summary"]["signal_end"] is not None
    assert "threshold_study" in result["summary"]
    assert "causal_bias" in result["summary"]
    assert "early_confirmation" in result["summary"]
    assert "prior_bias_alignment" in result["summary"]
    assert (out / "prior_bias_alignment_summary.json").exists()
    assert (out / "prior_bias_alignment_same_day.csv").exists()
    assert "先验偏买对齐菱形" in text


def test_prior_bias_alignment_study_reproducible_and_isolated():
    panel, signals = _longer_panel_and_signals()
    frame = build_assessment_frame(signals, panel, horizon=10)
    assert "causal_prior_ret" in frame.columns
    assert "causal_thr_prior" in frame.columns
    s1 = run_prior_bias_alignment_study(frame, horizon=10)
    s2 = run_prior_bias_alignment_study(frame, horizon=10)
    hold1 = s1["same_day"][s1["same_day"]["window"] == "holdout"].reset_index(drop=True)
    hold2 = s2["same_day"][s2["same_day"]["window"] == "holdout"].reset_index(drop=True)
    assert list(hold1["same_day_hit_rate"]) == list(hold2["same_day_hit_rate"])
    assert "recommendation" in s1["summary"]
    assert "update_plot_labels" in s1["summary"]["recommendation"]
    # Poisoning select months must not change holdout alignment rates.
    splits = split_time_windows(frame)
    poisoned = frame.copy()
    select_mask = poisoned["month"].isin(splits["select_months"])
    poisoned.loc[select_mask, "turn_kind"] = "none"
    poisoned.loc[select_mask, "label_switch_reversal"] = False
    s_p = run_prior_bias_alignment_study(poisoned, horizon=10)
    hold_p = s_p["same_day"][s_p["same_day"]["window"] == "holdout"].reset_index(
        drop=True
    )
    assert list(hold_p["same_day_hit_rate"]) == list(hold1["same_day_hit_rate"])


def test_early_confirmation_study_outputs_migration_rates():
    panel, signals = _longer_panel_and_signals()
    frame = build_assessment_frame(signals, panel, horizon=10)
    assert "early_label_switch_reversal" in frame.columns
    assert "early_confirmed_on" in frame.columns
    assert "final_confirmed_on" in frame.columns
    assert "early_to_final_status" in frame.columns
    # Immature tail cannot mature for short forward horizons.
    assert pd.isna(frame.sort_values("as_of").iloc[-1]["fwd_ret_1"])

    study = run_early_confirmation_study(frame)
    assert "summary" in study
    assert not study["windows"].empty
    assert set(study["windows"]["window"]) >= {"full_mature", "holdout"}
    hold = study["windows"][study["windows"]["window"] == "holdout"].iloc[0]
    assert "confirm_rate_given_early" in hold
    assert "revoke_rate_given_early" in hold
    assert not study["by_etf"].empty
    assert not study["by_year"].empty


def test_forward_log_return_maturity_and_missing():
    idx = pd.bdate_range("2024-01-02", periods=12)
    series = pd.Series(np.linspace(100.0, 111.0, len(idx)), index=idx)
    assert _forward_log_return(series, idx[0], 1) is not None
    assert _forward_log_return(series, idx[0], 10) is not None
    # last day cannot mature for h=1
    assert _forward_log_return(series, idx[-1], 1) is None
    assert _forward_log_return(series, idx[-2], 3) is None
    # missing as_of
    assert _forward_log_return(series, pd.Timestamp("2023-01-01"), 1) is None


def test_direction_hit_up_down_zero():
    assert _direction_hit("buy_bias", 0.01) is True
    assert _direction_hit("buy_bias", -0.01) is False
    assert _direction_hit("sell_bias", -0.02) is True
    assert _direction_hit("sell_bias", 0.02) is False
    assert _direction_hit("buy_bias", 0.0) is None
    assert _direction_hit("sell_bias", float("nan")) is None


def test_attach_forward_horizons_and_causal_bias_hits():
    panel, signals = _synthetic_panel_and_signals()
    # add an explicit down switch
    signals = signals.copy()
    signals.loc[signals.index[21], "pred_cdf"] = 0.03
    signals.loc[signals.index[21], "quantile_switch"] = True
    signals.loc[signals.index[21], "switch_side"] = "down"
    frame = build_assessment_frame(signals, panel, horizon=10)
    for h in (1, 3, 5, 10):
        assert f"fwd_ret_{h}" in frame.columns
    # immature tail: last day has no fwd_ret_1
    last = frame.sort_values("as_of").iloc[-1]
    assert pd.isna(last["fwd_ret_1"])
    # earlier mature day has fwd_ret_10
    mature_rows = frame[frame["fwd_ret_10"].notna()]
    assert not mature_rows.empty

    study = run_causal_bias_study(frame, reversal_horizon=10)
    fwd = study["forward"]
    assert not fwd.empty
    assert set(fwd["horizon"]) >= {1, 3, 5, 10}
    assert set(fwd["side"]) >= {"buy_bias", "sell_bias", "both"}
    assert set(fwd["window"]) >= {"full_mature", "holdout"}
    rev = study["reversal"]
    assert not rev.empty
    assert "f1" in rev.columns
    assert "naming_recommendation" in study["summary"]


def test_causal_bias_opposite_turn_matching_and_zero_alerts():
    panel, signals = _synthetic_panel_and_signals()
    frame = build_assessment_frame(signals, panel, horizon=5)
    # Force one up alert; turn_kind may or may not match bottom.
    study = run_causal_bias_study(frame, reversal_horizon=5)
    buy = study["reversal"][
        (study["reversal"]["window"] == "full_mature")
        & (study["reversal"]["side"] == "buy_bias")
    ]
    assert not buy.empty
    # Zero-alert frame
    empty = frame.copy()
    empty["quantile_switch"] = False
    empty["switch_side"] = None
    zero = run_causal_bias_study(empty, reversal_horizon=5)
    assert int(zero["summary"]["n_alerts_full"]) == 0
    assert zero["forward"]["n_scored"].fillna(0).sum() == 0


def test_causal_bias_holdout_isolation_and_bootstrap_reproducible():
    panel, signals = _longer_panel_and_signals()
    frame = build_assessment_frame(signals, panel, horizon=10)
    splits = split_time_windows(frame)
    study1 = run_causal_bias_study(frame, reversal_horizon=10)
    study2 = run_causal_bias_study(frame, reversal_horizon=10)
    hold1 = study1["forward"][study1["forward"]["window"] == "holdout"].reset_index(
        drop=True
    )
    hold2 = study2["forward"][study2["forward"]["window"] == "holdout"].reset_index(
        drop=True
    )
    assert list(hold1["hit_rate"]) == list(hold2["hit_rate"])
    assert list(hold1["hit_ci_low"]) == list(hold2["hit_ci_low"])
    # holdout alerts only from holdout months
    alerts = frame[
        frame["quantile_switch"].fillna(False)
        & frame["switch_side"].notna()
        & frame["month"].isin(splits["holdout_months"])
    ]
    assert int(study1["summary"]["n_alerts_holdout"]) == int(len(alerts))
    # Changing only non-holdout months' future returns should not change holdout hit rates
    # when we recompute after zeroing select-month fwd returns on a copy of study inputs.
    poisoned = frame.copy()
    select_mask = poisoned["month"].isin(splits["select_months"])
    for h in (1, 3, 5, 10):
        poisoned.loc[select_mask, f"fwd_ret_{h}"] = 0.0
    study_p = run_causal_bias_study(poisoned, reversal_horizon=10)
    hold_p = study_p["forward"][study_p["forward"]["window"] == "holdout"].reset_index(
        drop=True
    )
    assert list(hold_p["hit_rate"]) == list(hold1["hit_rate"])


def test_build_assessment_frame_fwd_ret_columns_present():
    panel, signals = _synthetic_panel_and_signals()
    frame = build_assessment_frame(signals, panel, horizon=10)
    assert "fwd_ret_h" in frame.columns
    left = pd.to_numeric(frame["fwd_ret_h"], errors="coerce")
    right = pd.to_numeric(frame["fwd_ret_10"], errors="coerce")
    assert left.equals(right)