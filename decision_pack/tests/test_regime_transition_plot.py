from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from decision_pack.src.regime_transition_plot import (
    build_confirmation_marker_frame,
    build_overlay_frame,
    causal_action_label,
    classify_vol_pct_regime,
    collapse_reversal_marker_runs,
    compute_realized_vol,
    compute_vol5_pct_120d,
    compute_volatility_regime_frame,
    dedupe_event_rows,
    load_validation_csvs,
    overlay_marker_specs,
    posthoc_reversal_label,
    validation_action_label,
    volatility_regime_segments,
    VOL_REGIME_CALM,
    VOL_REGIME_CLUSTER,
    VOL_REGIME_EXTREME,
    VOL_REGIME_NORMAL,
    VOL_REGIME_UNKNOWN,
    VOL_REGIMES_SHADED,
)


def _sample_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oos = pd.DataFrame(
        {
            "as_of": ["2026-06-08", "2026-06-09", "2026-07-02", "2026-07-03"],
            "code": ["SH513310"] * 4,
            "name": ["中韩半导体ETF华泰柏瑞"] * 4,
            "signal_level": ["OBSERVE", "OBSERVE", "OBSERVE", "NONE"],
            "quantile_switch": [True, True, True, False],
            "switch_side": ["down", "up", "down", None],
            "pred_cdf": [0.05, 0.95, 0.04, 0.5],
            "pred_mean": [-0.003, -0.0035, -0.002, -0.001],
            "pred_mean_next": [-0.0035, -0.004, -0.003, None],
            "pred_scale_next": [0.05, 0.051, 0.049, None],
            "return_z": [-2.0, 2.1, -2.2, 0.1],
            "p_switch_hmm": [0.2, 0.3, 0.4, 0.1],
            "p_into_reversal_hmm": [0.1, 0.2, 0.3, 0.05],
        }
    )
    rev = pd.DataFrame(
        {
            "as_of": ["2026-06-08", "2026-06-09", "2026-07-02", "2026-07-03"],
            "code": ["SH513310"] * 4,
            "label_switch_reversal": [False, True, True, False],
            "turn_kind": ["none", "bottom_reversal", "top_reversal", "none"],
            "jump_role": ["none", "bounce", "break", "none"],
            "prior_ret": [-0.03, -0.05, 0.04, 0.0],
            "jump_ret": [-0.02, 0.06, -0.05, 0.0],
            "post_ret": [0.01, 0.03, -0.04, 0.0],
            "final_confirmed_on": [None, "2026-06-24", "2026-07-16", None],
            "early_label_switch_reversal": [False, True, True, False],
            "early_turn_kind": ["none", "bottom_reversal", "top_reversal", "none"],
            "early_jump_role": ["none", "bounce", "break", "none"],
            "early_prior_ret": [-0.03, -0.05, 0.04, 0.0],
            "early_jump_ret": [-0.02, 0.06, -0.05, 0.0],
            "early_post_ret": [0.01, 0.02, -0.03, 0.0],
            "early_confirmed_on": [None, "2026-06-12", "2026-07-07", None],
            "early_to_final_status": ["none", "confirmed", "confirmed", "none"],
            # Strong priors for default observe_only wording.
            "causal_prior_ret": [-0.03, -0.05, 0.04, 0.0],
            "causal_jump_ret": [-0.02, 0.06, -0.05, 0.0],
            "causal_thr_prior": [0.01, 0.01, 0.01, 0.01],
            "causal_prior_strong": [True, True, True, False],
        }
    )
    evt = pd.DataFrame(
        {
            "as_of": ["2026-06-08", "2026-06-08", "2026-06-09", "2026-07-02"],
            "code": ["SH513310"] * 4,
            "matched_event_onset": [None, None, "2026-06-09", None],
            "detection_delay": [None, None, 0, None],
            "is_duplicate": [True, True, False, False],
            "is_fp": [False, False, False, True],
        }
    )
    return oos, rev, evt


def test_causal_action_label_ignores_future_fields():
    # Default observe_only → direction only even with strong prior.
    assert (
        causal_action_label(quantile_switch=True, switch_side="up")
        == "上行分位切换"
    )
    assert (
        causal_action_label(quantile_switch=True, switch_side="down")
        == "下行分位切换"
    )
    assert (
        causal_action_label(quantile_switch=False, switch_side=None) == "无切换"
    )
    assert (
        causal_action_label(
            quantile_switch=True,
            switch_side="up",
            prior_ret=-0.05,
            thr_prior=0.01,
        )
        == "上行分位切换"
    )
    assert (
        causal_action_label(
            quantile_switch=True,
            switch_side="up",
            prior_ret=0.27,
            thr_prior=0.05,
        )
        == "上行分位切换"
    )
    # Weak prior also direction-only.
    assert (
        causal_action_label(
            quantile_switch=True,
            switch_side="up",
            prior_ret=0.01,
            thr_prior=0.05,
        )
        == "上行分位切换"
    )
    assert (
        causal_action_label(
            quantile_switch=True, switch_side="up", bias_rule="observe_only"
        )
        == "上行分位切换"
    )
    # Explicit prior_bias no longer changes main label (gate-only).
    assert (
        causal_action_label(
            quantile_switch=True,
            switch_side="up",
            prior_ret=-0.05,
            thr_prior=0.01,
            bias_rule="prior_bias",
        )
        == "上行分位切换"
    )
    assert (
        causal_action_label(
            quantile_switch=True,
            switch_side="up",
            prior_ret=0.05,
            thr_prior=0.01,
            bias_rule="prior_bias",
        )
        == "上行分位切换"
    )
    # Future reversal fields must not change causal label.
    assert (
        validation_action_label(
            quantile_switch=True,
            switch_side="up",
            turn_kind="bottom_reversal",
            jump_role="bounce",
        )
        == "上行分位切换"
    )
    assert (
        validation_action_label(
            quantile_switch=True,
            switch_side="down",
            turn_kind="top_reversal",
            jump_role="break",
        )
        == "下行分位切换"
    )


def test_posthoc_reversal_label_mapping():
    assert (
        posthoc_reversal_label(turn_kind="bottom_reversal", jump_role="bounce")
        == "最终确认:底部反转(偏买)"
    )
    assert (
        posthoc_reversal_label(
            turn_kind="bottom_reversal", jump_role="climax_bottom", stage="early"
        )
        == "早期确认:底部反转"
    )
    assert (
        posthoc_reversal_label(turn_kind="top_reversal", jump_role="break")
        == "最终确认:顶部反转(偏卖)"
    )
    assert posthoc_reversal_label(turn_kind="none") == "无反转标签"
    assert (
        posthoc_reversal_label(turn_kind="bottom_reversal", stage="revoked")
        == "早期确认已撤销(非最终反转)"
    )

def test_dedupe_event_rows_prefers_matched_non_duplicate():
    evt = pd.DataFrame(
        {
            "as_of": ["2026-06-09", "2026-06-09", "2026-06-09"],
            "code": ["SH513310"] * 3,
            "matched_event_onset": [None, "2026-06-09", "2026-06-09"],
            "is_duplicate": [False, True, False],
            "is_fp": [False, False, False],
            "detection_delay": [None, 1, 0],
        }
    )
    out = dedupe_event_rows(evt)
    assert len(out) == 1
    row = out.iloc[0]
    assert pd.notna(row["matched_event_onset"])
    assert bool(row["is_duplicate"]) is False
    assert int(row["detection_delay"]) == 0


def test_build_overlay_frame_same_day_dual_markers():
    oos, rev, evt = _sample_frames()
    oos = pd.concat(
        [
            oos,
            pd.DataFrame(
                [
                    {
                        "as_of": "2026-06-09",
                        "code": "SZ159647",
                        "name": "other",
                        "signal_level": "OBSERVE",
                        "quantile_switch": True,
                        "switch_side": "up",
                        "pred_cdf": 0.99,
                        "return_z": 3.0,
                        "p_switch_hmm": 0.5,
                        "p_into_reversal_hmm": 0.4,
                    },
                    {
                        "as_of": "2026-05-01",
                        "code": "SH513310",
                        "name": "中韩半导体ETF华泰柏瑞",
                        "signal_level": "OBSERVE",
                        "quantile_switch": True,
                        "switch_side": "up",
                        "pred_cdf": 0.99,
                        "return_z": 3.0,
                        "p_switch_hmm": 0.5,
                        "p_into_reversal_hmm": 0.4,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    out = build_overlay_frame(
        oos,
        rev,
        evt,
        code="SH513310",
        start_date="2026-06-01",
        end_date="2026-07-03",
    )
    assert list(out["as_of"].dt.strftime("%Y-%m-%d")) == [
        "2026-06-08",
        "2026-06-09",
        "2026-07-02",
        "2026-07-03",
    ]
    assert set(out["code"]) == {"SH513310"}

    row_0609 = out.loc[out["as_of"] == "2026-06-09"].iloc[0]
    assert bool(row_0609["quantile_switch"]) is True
    assert row_0609["turn_kind"] == "bottom_reversal"
    assert row_0609["switch_marker"] == "switch_up"
    assert row_0609["reversal_marker"] == "reversal_bottom"
    assert row_0609["marker_kind"] == "switch_up"  # causal alias
    assert row_0609["action_label"] == "上行分位切换"
    assert row_0609["reversal_label"] == "最终确认:底部反转(偏买)"
    assert pd.notna(row_0609["matched_event_onset"])
    assert bool(row_0609["is_duplicate"]) is False
    assert str(pd.Timestamp(row_0609["early_confirmed_on"]).date()) == "2026-06-12"
    assert str(pd.Timestamp(row_0609["final_confirmed_on"]).date()) == "2026-06-24"

    row_0702 = out.loc[out["as_of"] == "2026-07-02"].iloc[0]
    assert row_0702["switch_marker"] == "switch_down"
    assert row_0702["reversal_marker"] == "reversal_top"
    assert row_0702["action_label"] == "下行分位切换"
    assert row_0702["reversal_label"] == "最终确认:顶部反转(偏卖)"
    assert bool(row_0702["is_fp"]) is True

    row_0608 = out.loc[out["as_of"] == "2026-06-08"].iloc[0]
    assert row_0608["switch_marker"] == "switch_down"
    assert row_0608["reversal_marker"] == "none"

    row_0703 = out.loc[out["as_of"] == "2026-07-03"].iloc[0]
    assert bool(row_0703["quantile_switch"]) is False
    assert row_0703["switch_marker"] == "none"
    assert row_0703["reversal_marker"] == "none"


def test_build_overlay_frame_missing_code_raises():
    oos, rev, evt = _sample_frames()
    with pytest.raises(ValueError, match="not found"):
        build_overlay_frame(
            oos, rev, evt, code="SH999999", start_date="2026-06-01", end_date="2026-07-03"
        )


def test_build_overlay_frame_date_oob_raises():
    oos, rev, evt = _sample_frames()
    with pytest.raises(ValueError, match="no signals_oos rows"):
        build_overlay_frame(
            oos, rev, evt, code="SH513310", start_date="2030-01-01", end_date="2030-01-31"
        )


def test_load_validation_csvs_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="missing validation CSV"):
        load_validation_csvs(tmp_path)


def test_collapse_reversal_keeps_switch_triangles():
    frame = pd.DataFrame(
        {
            "as_of": pd.to_datetime(
                ["2026-06-09", "2026-06-10", "2026-06-11", "2026-06-23", "2026-06-24"]
            ),
            "switch_marker": [
                "switch_up",
                "none",
                "none",
                "switch_down",
                "none",
            ],
            "reversal_marker": [
                "reversal_bottom",
                "reversal_bottom",
                "reversal_bottom",
                "none",
                "reversal_top",
            ],
            "early_reversal_marker": ["none"] * 5,
            "early_confirmed_on": [pd.NaT] * 5,
            "final_confirmed_on": pd.to_datetime(
                ["2026-06-23", "2026-06-24", "2026-06-25", None, "2026-07-08"]
            ),
            "early_to_final_status": ["confirmed", "confirmed", "confirmed", "none", "confirmed"],
            "action_label": ["上行分位切换"] * 5,
            "reversal_label": [
                "底部反转验证(偏买)",
                "底部反转验证",
                "底部反转验证",
                "无反转标签",
                "顶部反转验证",
            ],
            "signal_level": ["OBSERVE"] * 5,
            "pred_cdf": [0.95, 0.5, 0.5, 0.05, 0.5],
            "pred_mean": [0.0] * 5,
            "pred_mean_next": [0.0] * 5,
            "return_z": [0.0] * 5,
            "p_switch_hmm": [0.0] * 5,
            "p_into_reversal_hmm": [0.0] * 5,
            "turn_kind": [
                "bottom_reversal",
                "bottom_reversal",
                "bottom_reversal",
                "none",
                "top_reversal",
            ],
            "jump_role": ["bounce", "climax_bottom", "bounce", "none", "break"],
            "prior_ret": [0.0] * 5,
            "jump_ret": [0.0] * 5,
            "post_ret": [0.0] * 5,
            "matched_event_onset": [None] * 5,
            "detection_delay": [None] * 5,
            "is_duplicate": [False] * 5,
            "is_fp": [False] * 5,
        }
    )
    out = collapse_reversal_marker_runs(frame)
    assert list(out["switch_marker"]) == [
        "switch_up",
        "none",
        "none",
        "switch_down",
        "none",
    ]
    assert list(out["reversal_marker"]) == [
        "reversal_bottom",
        "none",
        "none",
        "none",
        "reversal_top",
    ]
    specs = overlay_marker_specs(frame)
    kinds = {s["name"] for s in specs}
    assert "最终底反确认(T+10)" in kinds
    assert "上行切换" in kinds
    assert "下行切换" in kinds
    # Post-hoc verification wording must not appear in triangle legend names.
    assert not any("底部反转" in name and "切换" in name for name in kinds)
    assert not any("偏买" in name or "偏卖" in name for name in kinds)
    assert not any("switch_up_bottom" in str(s) for s in specs)


def test_switch_hover_excludes_future_fields():
    oos, rev, evt = _sample_frames()
    overlay = build_overlay_frame(
        oos, rev, evt, code="SH513310", start_date="2026-06-01", end_date="2026-07-03"
    )
    specs = overlay_marker_specs(overlay)
    switch_specs = [
        s for s in specs if s["name"] in {"上行切换", "下行切换"}
    ]
    assert switch_specs
    switch_hover = "\n".join(h for s in switch_specs for h in s["hover"])
    for banned in (
        "turn=",
        "jump_role=",
        "prior=",
        "jump=",
        "post=",
        "matched_onset=",
        "delay=",
        "duplicate=",
        "is_fp=",
        "底部反转验证",
        "顶部反转验证",
        "最终确认",
        "早期确认",
    ):
        assert banned not in switch_hover, banned
    assert "pred_mean_next=" in switch_hover
    assert "pred_mean=" in switch_hover
    assert "pred_mean_next=-0.40%" in switch_hover
    assert "无未来标签" in switch_hover
    assert "上行分位切换" in switch_hover or "下行分位切换" in switch_hover


def test_confirmation_markers_use_knowable_dates_not_origin():
    oos, rev, evt = _sample_frames()
    overlay = build_overlay_frame(
        oos, rev, evt, code="SH513310", start_date="2026-06-01", end_date="2026-07-03"
    )
    confirm = build_confirmation_marker_frame(overlay)
    early = confirm[confirm["confirm_stage"] == "early"]
    final = confirm[confirm["confirm_stage"] == "final"]
    assert set(early["as_of"].dt.strftime("%Y-%m-%d")) >= {"2026-06-12"}
    assert "2026-06-09" not in set(early["as_of"].dt.strftime("%Y-%m-%d"))
    assert set(final["as_of"].dt.strftime("%Y-%m-%d")) >= {"2026-06-24"}
    origin_0609 = confirm[confirm["origin_as_of"] == "2026-06-09"]
    assert set(origin_0609["confirm_stage"]) >= {"early", "final"}

    specs = overlay_marker_specs(overlay)
    early_specs = [s for s in specs if "早期" in s["name"]]
    final_specs = [s for s in specs if "最终" in s["name"]]
    assert early_specs
    assert final_specs
    early_xs = [pd.Timestamp(x).strftime("%Y-%m-%d") for s in early_specs for x in s["x"]]
    final_xs = [pd.Timestamp(x).strftime("%Y-%m-%d") for s in final_specs for x in s["x"]]
    assert "2026-06-12" in early_xs
    assert "2026-06-24" in final_xs
    assert "2026-06-09" not in early_xs
    assert "2026-06-09" not in final_xs


def test_revoked_early_confirmation_marker():
    oos, rev, evt = _sample_frames()
    rev = rev.copy()
    rev.loc[rev["as_of"] == "2026-06-09", "label_switch_reversal"] = False
    rev.loc[rev["as_of"] == "2026-06-09", "turn_kind"] = "none"
    rev.loc[rev["as_of"] == "2026-06-09", "jump_role"] = "none"
    rev.loc[rev["as_of"] == "2026-06-09", "early_to_final_status"] = "revoked"
    overlay = build_overlay_frame(
        oos, rev, evt, code="SH513310", start_date="2026-06-01", end_date="2026-07-03"
    )
    confirm = build_confirmation_marker_frame(overlay)
    revoked = confirm[confirm["confirm_stage"] == "revoked"]
    assert not revoked.empty
    assert str(pd.Timestamp(revoked.iloc[0]["as_of"]).date()) == "2026-06-24"
    specs = overlay_marker_specs(overlay)
    assert any(s["name"] == "早期确认已撤销" for s in specs)


def test_diamond_hover_includes_posthoc_fields():
    oos, rev, evt = _sample_frames()
    overlay = build_overlay_frame(
        oos, rev, evt, code="SH513310", start_date="2026-06-01", end_date="2026-07-03"
    )
    specs = overlay_marker_specs(overlay)
    diamond_specs = [s for s in specs if "确认" in s["name"] or "撤销" in s["name"]]
    assert diamond_specs
    diamond_hover = "\n".join(h for s in diamond_specs for h in s["hover"])
    assert "确认日=" in diamond_hover
    assert "拐点日=" in diamond_hover
    assert "turn=" in diamond_hover
    assert "jump_role=" in diamond_hover
    assert "post=" in diamond_hover
    assert "matched_onset=" in diamond_hover or "is_fp=" in diamond_hover
    # Triangle hover still must not leak future fields.
    switch_hover = "\n".join(
        h for s in specs if "切换" in s["name"] for h in s["hover"]
    )
    for banned in ("确认日=", "拐点日=", "turn=", "post=", "matched_onset="):
        assert banned not in switch_hover, banned

def test_missing_reversal_still_plots_causal_switch():
    oos, _, _ = _sample_frames()
    empty_rev = pd.DataFrame(
        columns=[
            "as_of",
            "code",
            "label_switch_reversal",
            "turn_kind",
            "jump_role",
            "prior_ret",
            "jump_ret",
            "post_ret",
        ]
    )
    empty_evt = pd.DataFrame(
        columns=[
            "as_of",
            "code",
            "matched_event_onset",
            "detection_delay",
            "is_duplicate",
            "is_fp",
        ]
    )
    out = build_overlay_frame(
        oos,
        empty_rev,
        empty_evt,
        code="SH513310",
        start_date="2026-06-01",
        end_date="2026-07-03",
    )
    assert (out["switch_marker"] == "switch_up").sum() == 1
    assert (out["switch_marker"] == "switch_down").sum() == 2
    assert (out["reversal_marker"] == "none").all()
    specs = overlay_marker_specs(out)
    assert any(s["name"] == "上行切换" for s in specs)
    assert not any("确认" in s["name"] for s in specs)


def test_triangle_labels_follow_strong_prior_without_trade_bias():
    oos, rev, evt = _sample_frames()
    out = build_overlay_frame(
        oos, rev, evt, code="SH513310", start_date="2026-06-01", end_date="2026-07-03"
    )
    row_up = out.loc[out["as_of"] == "2026-06-09"].iloc[0]
    row_down = out.loc[out["as_of"] == "2026-07-02"].iloc[0]
    assert row_up["action_label"] == "上行分位切换"
    assert row_down["action_label"] == "下行分位切换"


def test_trade_bias_conflict_does_not_override_prior_wording():
    """Model buy/sell must not override strong-prior triangle wording."""
    oos, rev, evt = _sample_frames()
    # 0609: up + strong negative prior → 偏买; model says sell_bias.
    # 0702: down + strong positive prior → 偏卖; model says buy_bias.
    trade_bias = pd.DataFrame(
        {
            "as_of": ["2026-06-09", "2026-07-02"],
            "code": ["SH513310", "SH513310"],
            "trade_action": ["sell_bias", "buy_bias"],
            "p_up": [0.28, 0.72],
            "model_train_end": ["2026-05-30", "2026-05-30"],
            "hold_days": [5, 5],
            "schema": ["regime_trade_bias_v1", "regime_trade_bias_v1"],
        }
    )
    out = build_overlay_frame(
        oos,
        rev,
        evt,
        code="SH513310",
        start_date="2026-06-01",
        end_date="2026-07-03",
        trade_bias=trade_bias,
    )
    row_up = out.loc[out["as_of"] == "2026-06-09"].iloc[0]
    row_down = out.loc[out["as_of"] == "2026-07-02"].iloc[0]
    assert row_up["action_label"] == "上行分位切换"
    assert row_down["action_label"] == "下行分位切换"
    assert row_up["trade_action"] == "sell_bias"
    assert row_down["trade_action"] == "buy_bias"
    specs = overlay_marker_specs(out)
    switch_hover = "\n".join(h for s in specs if "切换" in s["name"] for h in s["hover"])
    assert "5日模型参考=" in switch_hover
    assert "p_up=" in switch_hover
    assert "model_train_end=" in switch_hover
    assert "hold_days=" in switch_hover
    assert "仅观察" in switch_hover
    assert "lockbox 门控" in switch_hover
    assert "|prior|" not in switch_hover
    assert "涨后偏卖" not in switch_hover
    for banned in ("turn=", "post=", "matched_onset=", "trade_ret", "trade_action="):
        assert banned not in switch_hover, banned


def test_weak_prior_keeps_direction_only_label():
    oos, rev, evt = _sample_frames()
    rev = rev.copy()
    # Make 0609 prior weak relative to thr → no 偏买/偏卖 wording.
    rev.loc[rev["as_of"] == "2026-06-09", "causal_prior_ret"] = 0.005
    rev.loc[rev["as_of"] == "2026-06-09", "causal_thr_prior"] = 0.05
    rev.loc[rev["as_of"] == "2026-06-09", "causal_prior_strong"] = False
    out = build_overlay_frame(
        oos, rev, evt, code="SH513310", start_date="2026-06-01", end_date="2026-07-03"
    )
    row = out.loc[out["as_of"] == "2026-06-09"].iloc[0]
    assert row["action_label"] == "上行分位切换"


def test_final_diamond_skipped_without_confirmation_date():
    oos, rev, evt = _sample_frames()
    rev = rev.copy()
    # Keep final reversal label but wipe confirmation date → must not plot on origin.
    rev.loc[rev["as_of"] == "2026-06-09", "final_confirmed_on"] = None
    overlay = build_overlay_frame(
        oos, rev, evt, code="SH513310", start_date="2026-06-01", end_date="2026-07-03"
    )
    confirm = build_confirmation_marker_frame(overlay)
    final_0609 = confirm[
        (confirm["origin_as_of"] == "2026-06-09") & (confirm["confirm_stage"] == "final")
    ]
    assert final_0609.empty
    # Early confirmation with date still plots.
    early_0609 = confirm[
        (confirm["origin_as_of"] == "2026-06-09") & (confirm["confirm_stage"] == "early")
    ]
    assert not early_0609.empty
    assert str(pd.Timestamp(early_0609.iloc[0]["as_of"]).date()) == "2026-06-12"
    # Origin day must not appear as a final diamond x.
    specs = overlay_marker_specs(overlay)
    final_xs = [
        pd.Timestamp(x).strftime("%Y-%m-%d")
        for s in specs
        if "最终" in s["name"]
        for x in s["x"]
    ]
    assert "2026-06-09" not in final_xs


def test_causal_action_label_ignores_trade_action_conflicts():
    assert (
        causal_action_label(
            quantile_switch=True,
            switch_side="up",
            prior_ret=0.27,
            thr_prior=0.05,
            trade_action="buy_bias",
        )
        == "上行分位切换"
    )
    assert (
        causal_action_label(
            quantile_switch=True,
            switch_side="down",
            prior_ret=-0.20,
            thr_prior=0.05,
            trade_action="sell_bias",
        )
        == "下行分位切换"
    )
    assert (
        causal_action_label(
            quantile_switch=True,
            switch_side="down",
            trade_action="observe_only",
            bias_rule="observe_only",
        )
        == "下行分位切换"
    )


def _synthetic_close_for_vol_regime(n: int = 200) -> pd.Series:
    """Build a close series with calm then clustered absolute returns."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-02", periods=n)
    rets = np.zeros(n)
    # First stretch: low vol; later stretch: high vol spikes.
    cut = n // 2
    rets[1:cut] = rng.normal(0.0, 0.004, size=cut - 1)
    rets[cut:] = rng.normal(0.0, 0.035, size=n - cut)
    close = 100.0 * np.exp(np.cumsum(rets))
    return pd.Series(close, index=dates, name="close")


def test_classify_vol_pct_regime_boundaries():
    assert classify_vol_pct_regime(None) == VOL_REGIME_UNKNOWN
    assert classify_vol_pct_regime(0.30) == VOL_REGIME_CALM
    assert classify_vol_pct_regime(0.50) == VOL_REGIME_NORMAL
    assert classify_vol_pct_regime(0.70) == VOL_REGIME_CLUSTER
    assert classify_vol_pct_regime(0.89) == VOL_REGIME_CLUSTER
    assert classify_vol_pct_regime(0.90) == VOL_REGIME_EXTREME


def test_vol5_pct_120d_matches_garch_board_pointwise():
    from decision_pack.src.garch_short_horizon_board import (
        realized_vol_percentile_log,
    )

    close = _synthetic_close_for_vol_regime(180)
    frame = compute_volatility_regime_frame(close, lookback=120)
    # Sample several dates after warmup and compare to GARCH helper.
    sample_idx = [80, 100, 120, 140, 160]
    for i in sample_idx:
        end = pd.Timestamp(close.index[i])
        expected = realized_vol_percentile_log(
            close, end, vol_window=5, lookback=120
        )
        got = float(frame.loc[i, "vol5_pct_120d"])
        assert np.isfinite(expected)
        assert np.isfinite(got)
        assert got == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_volatility_percentile_uses_trailing_history_only():
    close = _synthetic_close_for_vol_regime(160)
    frame = compute_volatility_regime_frame(close, lookback=120, min_history=5)
    # Early rows lack enough rv history → unknown / NaN percentile.
    assert (frame["vol_regime"].iloc[:8] == VOL_REGIME_UNKNOWN).all()
    # rv5_for_pct needs 5 returns; then min_history=5 ⇒ first valid around idx 9+.
    known = frame[frame["vol5_pct_120d"].notna()]
    assert not known.empty
    assert set(known["vol_regime"]).issubset(
        {VOL_REGIME_CALM, VOL_REGIME_NORMAL, VOL_REGIME_CLUSTER, VOL_REGIME_EXTREME}
    )
    assert "rv20" in frame.columns
    assert frame["rv20"].notna().sum() > 0


def test_volatility_regime_has_elevated_stretch():
    close = _synthetic_close_for_vol_regime(220)
    frame = compute_volatility_regime_frame(close, lookback=120)
    late = frame.iloc[140:]
    elevated = late["vol_regime"].isin(
        {VOL_REGIME_CLUSTER, VOL_REGIME_EXTREME}
    )
    assert elevated.any()
    # Calm/normal should also appear somewhere in the series.
    assert (frame["vol_regime"] == VOL_REGIME_CALM).any() or (
        frame["vol_regime"] == VOL_REGIME_NORMAL
    ).any()


def test_volatility_future_poison_does_not_change_history():
    close = _synthetic_close_for_vol_regime(180)
    base = compute_volatility_regime_frame(close, lookback=120)
    poisoned = close.copy()
    # Blow up the last 5 closes — must not alter earlier regimes/thresholds.
    poisoned.iloc[-5:] = poisoned.iloc[-5:] * 1.5
    alt = compute_volatility_regime_frame(poisoned, lookback=120)
    cut = len(base) - 10
    for col in ("rv5", "rv20", "vol5_pct_120d"):
        assert np.allclose(
            base[col].iloc[:cut].to_numpy(dtype=float),
            alt[col].iloc[:cut].to_numpy(dtype=float),
            equal_nan=True,
        )
    assert list(base["vol_regime"].iloc[:cut]) == list(alt["vol_regime"].iloc[:cut])


def test_volatility_regime_segments_collapse_runs():
    frame = pd.DataFrame(
        {
            "as_of": pd.bdate_range("2026-06-01", periods=10),
            "vol_regime": [
                VOL_REGIME_UNKNOWN,
                VOL_REGIME_CALM,
                VOL_REGIME_CALM,
                VOL_REGIME_NORMAL,
                VOL_REGIME_CLUSTER,
                VOL_REGIME_CLUSTER,
                VOL_REGIME_EXTREME,
                VOL_REGIME_EXTREME,
                VOL_REGIME_NORMAL,
                VOL_REGIME_UNKNOWN,
            ],
        }
    )
    # Default: only elevated regimes for background.
    segs = volatility_regime_segments(frame)
    assert len(segs) == 2
    assert segs[0]["regime"] == VOL_REGIME_CLUSTER
    assert segs[1]["regime"] == VOL_REGIME_EXTREME
    assert str(pd.Timestamp(segs[0]["start"]).date()) == "2026-06-05"
    assert str(pd.Timestamp(segs[1]["end"]).date()) == "2026-06-10"
    # Explicit all-known regimes still collapse runs.
    all_segs = volatility_regime_segments(
        frame,
        regimes={
            VOL_REGIME_CALM,
            VOL_REGIME_NORMAL,
            VOL_REGIME_CLUSTER,
            VOL_REGIME_EXTREME,
        },
    )
    assert [s["regime"] for s in all_segs] == [
        VOL_REGIME_CALM,
        VOL_REGIME_NORMAL,
        VOL_REGIME_CLUSTER,
        VOL_REGIME_EXTREME,
        VOL_REGIME_NORMAL,
    ]
    assert VOL_REGIMES_SHADED == frozenset(
        {VOL_REGIME_CLUSTER, VOL_REGIME_EXTREME}
    )


def test_compute_vol5_pct_helper_inclusive_window():
    rv = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=pd.bdate_range("2026-01-01", periods=5))
    pct = compute_vol5_pct_120d(rv, lookback=5, min_history=3)
    # At last point, history=[0.1..0.5], rank of 0.5 is 5/5=1.0.
    assert pct.iloc[-1] == pytest.approx(1.0)
    # At index 2, history=[0.1,0.2,0.3], rank of 0.3 is 3/3=1.0.
    assert pct.iloc[2] == pytest.approx(1.0)


def test_build_candle_volume_vol_figure_has_three_rows_and_vol_traces():
    import importlib.util
    from pathlib import Path

    import numpy as np

    mod_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "plot_regime_transition_example.py"
    )
    spec = importlib.util.spec_from_file_location("plot_regime_example_mod", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    dates = pd.bdate_range("2026-06-01", periods=40)
    close = 10 + np.cumsum(np.random.default_rng(0).normal(0, 0.05, size=40))
    ohlcv = pd.DataFrame(
        {
            "datetime": dates,
            "$open": close,
            "$high": close + 0.1,
            "$low": close - 0.1,
            "$close": close,
            "$volume": np.full(40, 1_000_000.0),
            "$pct_change": np.r_[np.nan, np.diff(close) / close[:-1] * 100.0],
        }
    )
    # Prepend warmup history so percentile exists inside the window.
    warm_dates = pd.bdate_range("2025-06-01", periods=160)
    warm_close = 9 + np.cumsum(np.random.default_rng(1).normal(0, 0.02, size=160))
    warm = pd.Series(warm_close, index=warm_dates)
    full_close = pd.concat([warm, pd.Series(close, index=dates)])
    full_close = full_close[~full_close.index.duplicated(keep="last")]
    vol_frame = compute_volatility_regime_frame(full_close, lookback=120)

    fig = mod.build_candle_volume_vol_figure(ohlcv, vol_frame)
    assert fig.layout.grid.rows == 3 or len(fig._grid_ref) == 3
    names = {getattr(tr, "name", None) for tr in fig.data}
    assert "rv5" in names
    assert "rv20" in names
    assert "vol5_pct_120d" in names
    assert "分位70%" in names
    assert "分位90%" in names
    assert {"MA5", "MA10", "MA20"}.issubset(names)

    oos, rev, evt = _sample_frames()
    overlay = build_overlay_frame(
        oos, rev, evt, code="SH513310", start_date="2026-06-01", end_date="2026-07-03"
    )
    # Clip ohlcv/vol to overlay window for annotation.
    fig2 = mod.annotate_regime_overlays(
        fig,
        overlay,
        ohlcv,
        title="test",
        vol_frame=vol_frame,
    )
    names2 = {getattr(tr, "name", None) for tr in fig2.data}
    assert "上行切换" in names2 or "下行切换" in names2
    assert "rv5" in names2
    # Calm/normal must not appear as background legend; elevated may.
    assert "波动率平稳" not in names2
    assert "波动率正常" not in names2


def test_erp_path_symmetric_envelope_p95():
    """Causal expanding P95: asof δ≈full-sample; early rails ignore late spikes."""
    import importlib.util
    from pathlib import Path

    import numpy as np
    import pandas as pd

    path = Path("decision_pack/scripts/plot_regime_transition_example.py")
    spec = importlib.util.spec_from_file_location("pm_erp_env", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    n = 100
    idx = pd.RangeIndex(n)
    path_s = pd.Series(np.full(n, 100.0), index=idx)
    # Calm residuals then late extremes (would leak under full-window δ).
    rng = np.random.default_rng(0)
    r = rng.uniform(-0.10, 0.10, size=n)
    r[-3:] = 0.50
    r[-5:-3] = -0.40
    close = path_s * np.exp(r)
    upper, lower, delta_asof = mod.erp_path_symmetric_envelope(
        path_s, close, q=0.95, min_bars=20
    )
    assert np.isfinite(delta_asof)
    # As-of δ matches full-sample P95 (expanding at the last bar).
    assert delta_asof == pytest.approx(float(np.quantile(np.abs(r), 0.95)))
    assert delta_asof < np.log(1.50)

    # Day-wise symmetry where rails exist.
    ok = upper.notna() & lower.notna()
    assert ok.any()
    assert np.allclose(
        upper.loc[ok].to_numpy(),
        path_s.loc[ok].to_numpy() * np.exp(np.log(upper.loc[ok] / path_s.loc[ok])),
    )
    # Reconstruct δ_t from upper/path; early δ must be << late full-sample δ
    # when late spikes are excluded from the prefix.
    mid = 60
    upper_mid, lower_mid, delta_mid = mod.erp_path_symmetric_envelope(
        path_s.iloc[: mid + 1], close.iloc[: mid + 1], q=0.95, min_bars=20
    )
    assert np.isfinite(delta_mid)
    # Prefix without extremes → narrower than asof (which includes ±40%/50%).
    assert delta_mid < delta_asof - 1e-6
    # Early rail on full series equals prefix-only envelope at same t (causality).
    assert float(upper.iloc[mid]) == pytest.approx(float(upper_mid.iloc[-1]), rel=1e-12)
    assert float(lower.iloc[mid]) == pytest.approx(float(lower_mid.iloc[-1]), rel=1e-12)
    # Last-day rails use asof δ.
    assert float(upper.iloc[-1]) == pytest.approx(100.0 * np.exp(delta_asof), rel=1e-12)
    assert float(lower.iloc[-1]) == pytest.approx(100.0 * np.exp(-delta_asof), rel=1e-12)


def test_fair_need_compound_price_path():
    """Constant 4.3% need matches constant ERP path; 2× need grows faster."""
    import importlib.util
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from decision_pack.src.vol_need_return_board import ANCHOR_ERP_ANN

    path = Path("decision_pack/scripts/plot_regime_transition_example.py")
    spec = importlib.util.spec_from_file_location("pm_fair_need_path", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    n = 50
    dates = pd.bdate_range("2020-01-02", periods=n)
    close = pd.Series(np.full(n, 10.0))
    need_const = pd.Series(np.full(n, ANCHOR_ERP_ANN), index=dates)
    path_c, need_aln, _anchor = mod.fair_need_compound_price_path(
        close, dates, need_const, fallback_erp=ANCHOR_ERP_ANN
    )
    path_fixed = mod.constant_erp_price_path(close, erp_ann=ANCHOR_ERP_ANN)
    assert np.allclose(path_c.to_numpy(), path_fixed.to_numpy())
    assert np.allclose(need_aln.to_numpy(), ANCHOR_ERP_ANN)

    need_2x = pd.Series(np.full(n, ANCHOR_ERP_ANN * 2.0), index=dates)
    path_2x, _, _ = mod.fair_need_compound_price_path(
        close, dates, need_2x, fallback_erp=ANCHOR_ERP_ANN
    )
    p0 = 10.0
    expected_end = p0 * (1.0 + ANCHOR_ERP_ANN * 2.0) ** ((n - 1) / 252.0)
    assert float(path_2x.iloc[-1]) == pytest.approx(expected_end, rel=1e-9)
    assert float(path_2x.iloc[-1]) > float(path_c.iloc[-1])


def test_window_own_cagr_and_constant_path():
    """Phase A: g=(P_T/P_0)^(252/n_steps)-1; path = P0*(1+g)^(i/252)."""
    import importlib.util
    from pathlib import Path

    import numpy as np
    import pandas as pd

    path = Path("decision_pack/scripts/plot_regime_transition_example.py")
    spec = importlib.util.spec_from_file_location("pm_own_cagr", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    n = 253  # 252 steps from first to last
    close = pd.Series(np.linspace(10.0, 20.0, n))
    g = mod.window_own_cagr(close)
    expected_g = (20.0 / 10.0) ** (252.0 / 252.0) - 1.0
    assert g == pytest.approx(expected_g, rel=1e-12)
    path_s = mod.constant_erp_price_path(close, erp_ann=g)
    assert float(path_s.iloc[0]) == pytest.approx(10.0)
    assert float(path_s.iloc[-1]) == pytest.approx(20.0, rel=1e-12)
    # Midpoint matches compound formula.
    mid = n // 2
    assert float(path_s.iloc[mid]) == pytest.approx(
        10.0 * (1.0 + g) ** (mid / 252.0), rel=1e-12
    )


def test_log_trend_price_path():
    """own_log_trend: path = exp(a+b·i); g = exp(b·252)-1; short → CAGR fallback."""
    import importlib.util
    from pathlib import Path

    import numpy as np
    import pandas as pd

    path = Path("decision_pack/scripts/plot_regime_transition_example.py")
    spec = importlib.util.spec_from_file_location("pm_log_trend", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    n = 252
    # Exact geometric: P_i = 10 * (1.1)^(i/252) → log-linear, OLS recovers g=10%.
    i = np.arange(n, dtype=float)
    close = pd.Series(10.0 * np.power(1.1, i / 252.0))
    path_s, g = mod.log_trend_price_path(close, min_bars=10)
    assert g == pytest.approx(0.10, rel=1e-6)
    assert np.allclose(path_s.to_numpy(dtype=float), close.to_numpy(dtype=float), rtol=1e-6)

    # Short sample falls back to endpoint CAGR path.
    short = pd.Series([10.0, 11.0, 12.0])
    path_fb, g_fb = mod.log_trend_price_path(short, min_bars=63)
    g_ep = mod.window_own_cagr(short)
    expected = mod.constant_erp_price_path(short, erp_ann=g_ep)
    assert g_fb == pytest.approx(g_ep, rel=1e-12)
    assert np.allclose(path_fb.to_numpy(dtype=float), expected.to_numpy(dtype=float))


def test_trailing_log_trend_price_path_causal_and_adapts():
    """Trailing log OLS uses ≤t only; adapts after a late level jump."""
    import importlib.util
    from pathlib import Path

    import numpy as np
    import pandas as pd

    path = Path("decision_pack/scripts/plot_regime_transition_example.py")
    spec = importlib.util.spec_from_file_location("pm_trail_log", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    # Flat then steep melt-up: full-window fit sits above late-flat prices.
    n = 800
    close = pd.Series(np.concatenate([np.full(500, 10.0), np.linspace(10.0, 40.0, 300)]))
    full_path, _ = mod.log_trend_price_path(close, min_bars=10)
    trail_path, g_s = mod.trailing_log_trend_price_path(
        close, trail_years=2.0, min_bars=63
    )
    late_flat = 480
    # Causality: prefix trail path end equals truncated series end.
    mid = 550
    p_pref, _ = mod.trailing_log_trend_price_path(
        close.iloc[: mid + 1], trail_years=2.0, min_bars=63
    )
    assert float(trail_path.iloc[mid]) == pytest.approx(float(p_pref.iloc[-1]), rel=1e-9)
    # Full-window path is pulled up by the eventual melt-up; trailing stays near 10.
    assert float(full_path.iloc[late_flat]) > float(close.iloc[late_flat]) * 1.15
    assert abs(float(trail_path.iloc[late_flat]) / float(close.iloc[late_flat]) - 1.0) < 0.08
    # End: trailing tracks local regime better than full-window line.
    end_full = abs(float(close.iloc[-1]) / float(full_path.iloc[-1]) - 1.0)
    end_trail = abs(float(close.iloc[-1]) / float(trail_path.iloc[-1]) - 1.0)
    assert end_trail < end_full
    assert np.isfinite(g_s.iloc[-1])
    assert np.isfinite(trail_path.iloc[100:]).all()


def test_fair_path_lookback_covers_trail_years():
    import importlib.util
    from pathlib import Path

    import numpy as np

    path = Path("decision_pack/scripts/plot_regime_transition_example.py")
    spec = importlib.util.spec_from_file_location("pm_lookback", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    days = mod.fair_path_lookback_calendar_days()
    need = int(np.ceil(float(mod.FAIR_PATH_LOG_TREND_TRAIL_YEARS) * 365.25)) + int(
        mod.FAIR_PATH_LOOKBACK_BUFFER_DAYS
    )
    assert days >= need
    assert days >= int(mod.VOL_WARMUP_CALENDAR_DAYS)


def test_fig7_trail_uses_fair_path_ohlcv_history_not_plot_window_only():
    """Short OOS plot window must not annualize a 4-month slope as '2y'."""
    import importlib.util
    from pathlib import Path

    import numpy as np
    import pandas as pd

    path = Path("decision_pack/scripts/plot_regime_transition_example.py")
    spec = importlib.util.spec_from_file_location("pm_fig7_hist", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    n_hist = 600
    n_plot = 80
    dates = pd.bdate_range("2023-01-02", periods=n_hist + n_plot)
    # Steady ~10% ann log drift so trail g is stable.
    drift = np.log(1.10) / 252.0
    close = np.exp(drift * np.arange(n_hist + n_plot, dtype=float))
    full = pd.DataFrame(
        {
            "datetime": dates,
            "$open": close,
            "$high": close * 1.01,
            "$low": close * 0.99,
            "$close": close,
            "$volume": np.full(n_hist + n_plot, 1e6),
        }
    )
    plot = full.iloc[-n_plot:].reset_index(drop=True)
    from decision_pack.src.regime_transition_plot import compute_volatility_regime_frame

    close_s = mod._asof_close_series(full)
    vol_frame = compute_volatility_regime_frame(close_s, lookback=120)
    need_frame = pd.DataFrame(
        {
            "as_of": plot["datetime"],
            "r_need_ann": 0.043,
            "rv20": 0.2,
            "rv5": 0.2,
            "r_20d": 0.0,
            "gap_20d": -0.01,
            "r_need_ann_5": 0.043,
            "r_5d": 0.0,
            "gap_5d": -0.01,
        }
    )
    policy_frame = pd.DataFrame(
        {
            "as_of": plot["datetime"],
            "nav_hold_up": np.linspace(1.0, 1.1, n_plot),
            "nav_bh": np.linspace(1.0, 1.2, n_plot),
            "nav_cash": np.ones(n_plot),
            "nav_policy": np.linspace(1.0, 1.2, n_plot),
        }
    )

    path_full, g_full = mod.trailing_log_trend_price_path(
        mod._asof_close_series(full), min_bars=10
    )
    g_plot = mod._align_series_to_plot(g_full, plot)
    finite_long = g_plot.to_numpy(dtype=float)
    finite_long = finite_long[np.isfinite(finite_long)]
    assert len(finite_long) > 10
    assert abs(float(finite_long[-1]) - 0.10) < 0.03

    fig = mod.build_candle_volume_vol_figure(
        plot,
        vol_frame,
        need_frame=need_frame,
        policy_frame=policy_frame,
        policy_meta={"policy": "bh", "policy_label": "BH"},
        train_cutoff="2026-04-01",
        fair_path_ohlcv=full,
    )
    own_g = float(finite_long[-1])
    trail_y = int(round(float(mod.FAIR_PATH_LOG_TREND_TRAIL_YEARS)))
    path_prefix = f"公平路径(自有log趋势·{trail_y}年 g≈{own_g:.1%}"
    names = {getattr(tr, "name", None) for tr in fig.data}
    path_names = [
        n
        for n in names
        if isinstance(n, str) and n.startswith(path_prefix) and n.endswith(", 诊断)")
    ]
    assert path_names
    erp_tr = next(tr for tr in fig.data if getattr(tr, "name", None) in path_names)
    path_y = np.asarray(erp_tr.y, dtype=float)
    exp_y = mod._align_series_to_plot(path_full, plot).to_numpy(dtype=float)
    m = np.isfinite(path_y) & np.isfinite(exp_y)
    assert m.any()
    assert np.allclose(path_y[m], exp_y[m])
    hover = list(erp_tr.customdata or [])
    hover_g = []
    for h in hover:
        if not isinstance(h, str) or "当日路径年化" not in h:
            continue
        # "当日路径年化=" or "当日路径年化(2年)="
        frag = h.split("当日路径年化", 1)[1]
        frag = frag.split("=", 1)[1].split("<", 1)[0]
        if frag.endswith("%"):
            hover_g.append(abs(float(frag[:-1]) / 100.0))
    assert hover_g
    assert max(hover_g) < 1.0


def test_fig8_dual_trail_years_adds_row_and_dual_scale_labels():
    """fair_path_extra_trail_years=1 → 8 rows; both panels dual-annotate g."""
    import importlib.util
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from decision_pack.src.regime_transition_plot import compute_volatility_regime_frame

    path = Path("decision_pack/scripts/plot_regime_transition_example.py")
    spec = importlib.util.spec_from_file_location("pm_fig8", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    n = 400
    dates = pd.bdate_range("2024-01-02", periods=n)
    drift = np.log(1.12) / 252.0
    close = np.exp(drift * np.arange(n, dtype=float))
    ohlcv = pd.DataFrame(
        {
            "datetime": dates,
            "$open": close,
            "$high": close * 1.01,
            "$low": close * 0.99,
            "$close": close,
            "$volume": np.full(n, 1e6),
        }
    )
    vol_frame = compute_volatility_regime_frame(
        mod._asof_close_series(ohlcv), lookback=120
    )
    need_frame = pd.DataFrame({"as_of": dates, "r_need_ann": np.full(n, 0.043)})
    policy_frame = pd.DataFrame(
        {
            "as_of": dates,
            "nav_hold_up": np.linspace(1.0, 1.1, n),
            "nav_bh": np.linspace(1.0, 1.2, n),
            "nav_cash": np.ones(n),
            "nav_policy": np.linspace(1.0, 1.2, n),
        }
    )
    fig = mod.build_candle_volume_vol_figure(
        ohlcv,
        vol_frame,
        need_frame=need_frame,
        policy_frame=policy_frame,
        policy_meta={"policy": "bh", "policy_label": "BH"},
        fair_path_extra_trail_years=1.0,
    )
    assert fig.layout.height == 3000
    names = {getattr(tr, "name", None) for tr in fig.data}
    path_names = [n for n in names if isinstance(n, str) and n.startswith("公平路径")]
    assert len(path_names) >= 2
    assert any("2年" in n and "1年" in n for n in path_names)
    assert any("1年" in n and "2年" in n for n in path_names)
    assert any(isinstance(n, str) and n.startswith("上轨1年") for n in names)
    assert any(isinstance(n, str) and n.startswith("上轨2年") for n in names)
    titles = " ".join(
        getattr(a, "text", "") or "" for a in (fig.layout.annotations or [])
    )
    assert "图8" in titles or "滚动1年" in titles
    assert "因果P95+EWMA双轨" in titles
    assert any(
        isinstance(n, str) and n.startswith("上轨1年") and "EWMA残差轨" in n
        for n in names
    )


def test_fig9_six_month_trail_with_1y_extra():
    """extras 1 + 0.5 → fig8/fig9; labels use 6个月."""
    import importlib.util
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from decision_pack.src.regime_transition_plot import compute_volatility_regime_frame

    path = Path("decision_pack/scripts/plot_regime_transition_example.py")
    spec = importlib.util.spec_from_file_location("pm_fig9", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    assert mod._trail_horizon_label(0.5) == "6个月"
    assert mod._normalize_extra_trail_years("1,0.5") == [1.0, 0.5]

    n = 400
    dates = pd.bdate_range("2024-01-02", periods=n)
    drift = np.log(1.12) / 252.0
    close = np.exp(drift * np.arange(n, dtype=float))
    ohlcv = pd.DataFrame(
        {
            "datetime": dates,
            "$open": close,
            "$high": close * 1.01,
            "$low": close * 0.99,
            "$close": close,
            "$volume": np.full(n, 1e6),
        }
    )
    vol_frame = compute_volatility_regime_frame(
        mod._asof_close_series(ohlcv), lookback=120
    )
    need_frame = pd.DataFrame({"as_of": dates, "r_need_ann": np.full(n, 0.043)})
    policy_frame = pd.DataFrame(
        {
            "as_of": dates,
            "nav_hold_up": np.linspace(1.0, 1.1, n),
            "nav_bh": np.linspace(1.0, 1.2, n),
            "nav_cash": np.ones(n),
            "nav_policy": np.linspace(1.0, 1.2, n),
        }
    )
    fig = mod.build_candle_volume_vol_figure(
        ohlcv,
        vol_frame,
        need_frame=need_frame,
        policy_frame=policy_frame,
        policy_meta={"policy": "bh", "policy_label": "BH"},
        fair_path_extra_trail_years="1,0.5",
    )
    assert fig.layout.height == 2600 + 400 * 2
    names = {getattr(tr, "name", None) for tr in fig.data}
    path_names = [n for n in names if isinstance(n, str) and n.startswith("公平路径")]
    assert len(path_names) >= 3
    assert any("6个月" in n for n in path_names)
    assert any("｜6个月" in n for n in path_names)
    titles = " ".join(
        getattr(a, "text", "") or "" for a in (fig.layout.annotations or [])
    )
    assert "图9" in titles or "6个月" in titles
    assert any(
        isinstance(n, str) and n.startswith("上轨6个月") and "因果P95" in n
        for n in names
    )
    assert any(
        isinstance(n, str) and n.startswith("上轨6个月") and "EWMA残差轨" in n
        for n in names
    )


def test_fig11_one_month_trail_min_bars_and_label():
    """1/12 year → 1个月 label; min_bars clamps to ~21 so OLS can fit."""
    import importlib.util
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from decision_pack.src.regime_transition_plot import compute_volatility_regime_frame

    path = Path("decision_pack/scripts/plot_regime_transition_example.py")
    spec = importlib.util.spec_from_file_location("pm_fig11", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    assert mod._trail_horizon_label(1.0 / 12.0) == "1个月"
    assert mod._normalize_extra_trail_years("1,0.5,0.25,1/12") == [
        1.0,
        0.5,
        0.25,
        pytest.approx(1.0 / 12.0),
    ]
    assert mod._normalize_extra_trail_years("off") == []
    assert mod._normalize_extra_trail_years("none") == []
    assert mod._trail_fit_min_bars(1.0 / 12.0) == 21

    n = 400
    dates = pd.bdate_range("2024-01-02", periods=n)
    # Mild up then sharp 1m drop so short-trail g is negative at the end.
    drift = np.log(1.10) / 252.0
    close = np.exp(drift * np.arange(n, dtype=float))
    close[-21:] *= np.linspace(1.0, 0.92, 21)
    ohlcv = pd.DataFrame(
        {
            "datetime": dates,
            "$open": close,
            "$high": close * 1.01,
            "$low": close * 0.99,
            "$close": close,
            "$volume": np.full(n, 1e6),
        }
    )
    vol_frame = compute_volatility_regime_frame(
        mod._asof_close_series(ohlcv), lookback=120
    )
    need_frame = pd.DataFrame({"as_of": dates, "r_need_ann": np.full(n, 0.043)})
    policy_frame = pd.DataFrame(
        {
            "as_of": dates,
            "nav_hold_up": np.linspace(1.0, 1.1, n),
            "nav_bh": np.linspace(1.0, 1.2, n),
            "nav_cash": np.ones(n),
            "nav_policy": np.linspace(1.0, 1.2, n),
        }
    )
    fig = mod.build_candle_volume_vol_figure(
        ohlcv,
        vol_frame,
        need_frame=need_frame,
        policy_frame=policy_frame,
        policy_meta={"policy": "bh", "policy_label": "BH"},
        fair_path_extra_trail_years="1,0.5,0.25,1/12",
    )
    assert fig.layout.height == 2600 + 400 * 4
    names = {getattr(tr, "name", None) for tr in fig.data}
    path_names = [n for n in names if isinstance(n, str) and n.startswith("公平路径")]
    assert any("1个月" in n for n in path_names)
    titles = " ".join(
        getattr(a, "text", "") or "" for a in (fig.layout.annotations or [])
    )
    assert "图11" in titles

    path_s, g_s = mod.trailing_log_trend_price_path(
        mod._asof_close_series(ohlcv),
        trail_years=1.0 / 12.0,
        min_bars=mod._trail_fit_min_bars(1.0 / 12.0),
    )
    assert np.isfinite(float(g_s.iloc[-1]))
    assert float(g_s.iloc[-1]) < 0.0


def test_fig7_dual_ewma_and_fig9_fig10_ewma_rails():
    """HTML rails: fig7–10 expanding+EWMA overlay; fig11 expanding P95.

    hold_up / compute_multi_scale_frame stay expanding; this only covers drawing.
    """
    import importlib.util
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from decision_pack.src.regime_transition_plot import compute_volatility_regime_frame

    path = Path("decision_pack/scripts/plot_regime_transition_example.py")
    spec = importlib.util.spec_from_file_location("pm_rail_policy", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    assert mod._trail_envelope_policy(2.0) == "expanding_plus_ewma"
    assert mod._trail_envelope_policy(1.0) == "expanding_plus_ewma"
    assert mod._trail_envelope_policy(0.5) == "expanding_plus_ewma"
    assert mod._trail_envelope_policy(0.25) == "expanding_plus_ewma"
    assert mod._trail_envelope_policy(1.0 / 12.0) == "expanding"

    n = 400
    dates = pd.bdate_range("2024-01-02", periods=n)
    rng = np.random.default_rng(1)
    close = np.exp(np.cumsum(rng.normal(0.0004, 0.012, size=n)))
    ohlcv = pd.DataFrame(
        {
            "datetime": dates,
            "$open": close,
            "$high": close * 1.01,
            "$low": close * 0.99,
            "$close": close,
            "$volume": np.full(n, 1e6),
        }
    )
    vol_frame = compute_volatility_regime_frame(
        mod._asof_close_series(ohlcv), lookback=120
    )
    need_frame = pd.DataFrame({"as_of": dates, "r_need_ann": np.full(n, 0.043)})
    policy_frame = pd.DataFrame(
        {
            "as_of": dates,
            "nav_hold_up": np.linspace(1.0, 1.1, n),
            "nav_bh": np.linspace(1.0, 1.2, n),
            "nav_cash": np.ones(n),
            "nav_policy": np.linspace(1.0, 1.2, n),
        }
    )
    fig = mod.build_candle_volume_vol_figure(
        ohlcv,
        vol_frame,
        need_frame=need_frame,
        policy_frame=policy_frame,
        policy_meta={"policy": "bh", "policy_label": "BH"},
        fair_path_extra_trail_years="1,0.5,0.25,1/12",
    )
    names = [getattr(tr, "name", None) for tr in fig.data]
    titles = " ".join(
        getattr(a, "text", "") or "" for a in (fig.layout.annotations or [])
    )
    assert "因果P95+EWMA双轨" in titles
    assert "图8" in titles
    assert "图9" in titles
    assert "图10" in titles

    def _rails(pred):
        return [n for n in names if isinstance(n, str) and pred(n)]

    fig7_p95 = _rails(lambda n: n.startswith("上轨2年") and "因果P95" in n)
    fig7_ewma = _rails(lambda n: n.startswith("上轨2年") and "EWMA残差轨" in n)
    fig8_p95 = _rails(lambda n: n.startswith("上轨1年") and "因果P95" in n)
    fig8_ewma = _rails(lambda n: n.startswith("上轨1年") and "EWMA残差轨" in n)
    fig9_p95 = _rails(lambda n: n.startswith("上轨6个月") and "因果P95" in n)
    fig9_ewma = _rails(lambda n: n.startswith("上轨6个月") and "EWMA残差轨" in n)
    fig10_p95 = _rails(lambda n: n.startswith("上轨3个月") and "因果P95" in n)
    fig10_ewma = _rails(lambda n: n.startswith("上轨3个月") and "EWMA残差轨" in n)
    fig11_p95 = _rails(lambda n: n.startswith("上轨1个月") and "因果P95" in n)
    fig11_ewma = _rails(lambda n: n.startswith("上轨1个月") and "EWMA残差轨" in n)
    assert fig7_p95 and fig7_ewma
    assert fig8_p95 and fig8_ewma
    assert fig9_p95 and fig9_ewma
    assert fig10_p95 and fig10_ewma
    assert fig11_p95 and not fig11_ewma
    ewma_hover = next(
        tr
        for tr in fig.data
        if isinstance(getattr(tr, "name", None), str)
        and str(tr.name).startswith("上轨6个月")
        and "EWMA残差轨" in str(tr.name)
    )
    assert ewma_hover.customdata is not None
    assert any("当日" in str(x) for x in ewma_hover.customdata)


def test_causal_own_cagr_series_and_path():
    """Phase B: g_t uses ≤t only; expanding then trail; path compounds smoothly."""
    import importlib.util
    from pathlib import Path

    import numpy as np
    import pandas as pd

    path = Path("decision_pack/scripts/plot_regime_transition_example.py")
    spec = importlib.util.spec_from_file_location("pm_own_cagr_causal", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    # Steady double over 252 steps → expanding g_t rises toward 100%.
    n = 253
    close = pd.Series(np.linspace(10.0, 20.0, n))
    g_s = mod.causal_own_cagr_series(close, trail_years=5, min_bars=2)
    assert np.isfinite(g_s.iloc[1:]).all()
    # At last bar, expanding CAGR matches Phase A window CAGR.
    assert float(g_s.iloc[-1]) == pytest.approx(mod.window_own_cagr(close), rel=1e-12)
    # Causality: prefix g equals series truncated to that prefix.
    mid = 100
    g_prefix = mod.causal_own_cagr_series(close.iloc[: mid + 1], trail_years=5, min_bars=2)
    assert float(g_s.iloc[mid]) == pytest.approx(float(g_prefix.iloc[-1]), rel=1e-12)

    path_s, g_aln = mod.causal_own_cagr_price_path(close, g_s, fallback_erp=0.043)
    assert float(path_s.iloc[0]) == pytest.approx(10.0)
    assert np.isfinite(path_s.to_numpy()).all()
    # No hard jump: day-to-day path move equals (1+g_t)^(1/252)-1.
    day_ret = path_s.pct_change().iloc[1:].to_numpy(dtype=float)
    expected_ret = (1.0 + g_aln.iloc[1:].to_numpy(dtype=float)) ** (1.0 / 252.0) - 1.0
    assert np.allclose(day_ret, expected_ret, rtol=1e-10, atol=1e-12)

    # Trailing window: after trail_bars, g uses last trail_bars only (not full history).
    # Build a series that rises then flats so full-history g ≠ trailing g.
    n2 = 300
    close2 = pd.Series(np.concatenate([np.linspace(10.0, 30.0, 150), np.full(150, 30.0)]))
    trail_bars = 100
    g_trail = mod.causal_own_cagr_series(
        close2, trail_years=trail_bars / 252.0, min_bars=2
    )
    # At the end, trailing CAGR over flat window ≈ 0, full-history CAGR > 0.
    g_full = mod.window_own_cagr(close2)
    assert float(g_trail.iloc[-1]) == pytest.approx(
        mod.window_own_cagr(close2.iloc[-trail_bars:]), rel=1e-10
    )
    assert float(g_trail.iloc[-1]) < float(g_full) * 0.5


def test_fair_need_compound_price_path_two_year_rebase():
    """2y rebase resets P0 to close on the first bar of each calendar segment."""
    import importlib.util
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from decision_pack.src.vol_need_return_board import ANCHOR_ERP_ANN

    path = Path("decision_pack/scripts/plot_regime_transition_example.py")
    spec = importlib.util.spec_from_file_location("pm_fair_need_rebase", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    # ~3 calendar years of business days starting 2020-01-02.
    dates = pd.bdate_range("2020-01-02", periods=780)
    close = pd.Series(np.full(len(dates), 10.0))
    # Permanent step at calendar +2y so listing-era path would stay stuck low.
    step_cut = pd.Timestamp("2020-01-02") + pd.DateOffset(years=2)
    close.loc[dates >= step_cut] = 20.0
    need = pd.Series(np.full(len(dates), ANCHOR_ERP_ANN), index=dates)

    path_rebased, need_aln, anchors = mod.fair_need_compound_price_path(
        close,
        dates,
        need,
        fallback_erp=ANCHOR_ERP_ANN,
        rebase_years=2,
    )
    path_legacy, _, _ = mod.fair_need_compound_price_path(
        close,
        dates,
        need,
        fallback_erp=ANCHOR_ERP_ANN,
        rebase_years=None,
    )
    assert np.allclose(need_aln.to_numpy(), ANCHOR_ERP_ANN)

    # First bar of second segment: path equals stepped close (20), not legacy ~10.
    rebase_mask = np.asarray(dates >= step_cut)
    first_rebase_i = int(np.flatnonzero(rebase_mask)[0])
    assert float(close.iloc[first_rebase_i]) == pytest.approx(20.0)
    assert float(path_rebased.iloc[first_rebase_i]) == pytest.approx(20.0)
    assert float(path_legacy.iloc[first_rebase_i]) < 12.0
    assert float(path_rebased.iloc[first_rebase_i]) > float(
        path_legacy.iloc[first_rebase_i]
    )

    # Anchor trading date on the rebase bar equals that bar's date.
    assert pd.Timestamp(anchors.iloc[first_rebase_i]).normalize() == pd.Timestamp(
        dates[first_rebase_i]
    ).normalize()
    # Later bars in the same segment keep the same trade-day anchor.
    assert pd.Timestamp(anchors.iloc[first_rebase_i + 5]).normalize() == pd.Timestamp(
        dates[first_rebase_i]
    ).normalize()

    upper, lower, delta = mod.erp_path_symmetric_envelope(
        path_rebased, close, q=0.95
    )
    assert np.isfinite(delta)
    path_arr = path_rebased.to_numpy(dtype=float)
    assert np.allclose(
        np.log(upper.to_numpy() / path_arr),
        -np.log(lower.to_numpy() / path_arr),
        equal_nan=True,
    )


def test_build_candle_volume_vol_figure_optional_need_row():
    """need_frame adds rows 4–5 with fair-return line + CSI300 reference."""
    import importlib.util
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from decision_pack.src.regime_transition_plot import compute_volatility_regime_frame
    from decision_pack.src.vol_need_return_board import (
        ANCHOR_ERP_ANN,
        EQUITY_ANCHOR_LABEL,
    )

    path = Path("decision_pack/scripts/plot_regime_transition_example.py")
    spec = importlib.util.spec_from_file_location("pm_need_row", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    dates = pd.bdate_range("2025-12-01", periods=100)
    rng = np.random.default_rng(0)
    close = 10 + np.cumsum(rng.normal(0, 0.05, size=100))
    ohlcv = pd.DataFrame(
        {
            "datetime": dates,
            "$open": close,
            "$high": close * 1.01,
            "$low": close * 0.99,
            "$close": close,
            "$volume": rng.integers(1_000_000, 2_000_000, size=100),
        }
    )
    warm_dates = pd.bdate_range("2025-01-02", periods=200)
    warm_close = 9 + np.cumsum(rng.normal(0, 0.02, size=200))
    full_close = pd.concat(
        [pd.Series(warm_close, index=warm_dates), pd.Series(close, index=dates)]
    )
    full_close = full_close[~full_close.index.duplicated(keep="last")]
    vol_frame = compute_volatility_regime_frame(full_close, lookback=120)

    fig3 = mod.build_candle_volume_vol_figure(ohlcv, vol_frame)
    assert fig3.layout.grid.rows == 3 or len(fig3._grid_ref) == 3

    need_frame = pd.DataFrame(
        {
            "as_of": dates,
            "r_need_ann": np.full(len(dates), ANCHOR_ERP_ANN),
        }
    )
    fig5 = mod.build_candle_volume_vol_figure(ohlcv, vol_frame, need_frame=need_frame)
    assert fig5.layout.grid.rows == 5 or len(fig5._grid_ref) == 5

    policy_frame = pd.DataFrame(
        {
            "as_of": dates,
            "nav_hold_up": np.linspace(1.0, 1.2, len(dates)),
            "nav_bh": np.linspace(1.0, 1.4, len(dates)),
            "nav_cash": np.ones(len(dates)),
            "nav_policy": np.linspace(1.0, 1.4, len(dates)),
        }
    )
    fig7 = mod.build_candle_volume_vol_figure(
        ohlcv,
        vol_frame,
        need_frame=need_frame,
        policy_frame=policy_frame,
        policy_meta={
            "policy": "bh",
            "policy_label": "买入持有(BH)",
            "train_hold_up": 0.1,
            "train_bh": 0.2,
            "train_cutoff": "2026-04-01",
        },
        train_cutoff="2026-04-01",
    )
    assert fig7.layout.grid.rows == 7 or len(fig7._grid_ref) == 7
    assert fig7.layout.height == 2600
    names7 = {getattr(tr, "name", None) for tr in fig7.data}
    expected_path, g_s = mod.trailing_log_trend_price_path(ohlcv["$close"])
    finite_g = g_s.to_numpy(dtype=float)
    finite_g = finite_g[np.isfinite(finite_g)]
    own_g = float(finite_g[-1])
    trail_y = int(round(float(mod.FAIR_PATH_LOG_TREND_TRAIL_YEARS)))
    path_prefix = f"公平路径(自有log趋势·{trail_y}年 g≈{own_g:.1%}"
    path_names7 = [
        n
        for n in names7
        if isinstance(n, str) and n.startswith(path_prefix) and n.endswith(", 诊断)")
    ]
    assert path_names7
    assert any("log趋势" in str(n) for n in names7 if n)
    assert not any("每2年重锚" in str(n) for n in names7 if n)
    assert not any("因果滚动" in str(n) for n in names7 if n)
    erp_tr = next(tr for tr in fig7.data if getattr(tr, "name", None) in path_names7)
    path_y = np.asarray(erp_tr.y, dtype=float)
    # Warmup may leave leading NaNs; compare finite overlap.
    exp_y = expected_path.to_numpy(dtype=float)
    m = np.isfinite(path_y) & np.isfinite(exp_y)
    assert m.any()
    assert np.allclose(path_y[m], exp_y[m])
    title7 = ""
    if fig7.layout.annotations:
        texts = [getattr(a, "text", "") or "" for a in fig7.layout.annotations]
        title7 = " ".join(texts)
    assert "≠图4" in title7 or "自有log趋势" in title7
    assert "滚动" in title7
    assert "因果P95+EWMA双轨" in title7
    # Clean row-7 candles: no MA traces shared onto that panel (MA stay on row 1).
    assert "K线(净)" not in names7 or any(
        getattr(tr, "type", None) == "candlestick" for tr in fig7.data
    )
    # Causal expanding P95 envelope parallel to fair path (extremes may pierce).
    upper_s, lower_s, delta = mod.erp_path_symmetric_envelope(
        expected_path, ohlcv["$close"], q=0.95
    )
    assert np.isfinite(delta)
    band_pct = float(np.exp(delta) - 1.0)
    assert any(
        isinstance(n, str) and n.startswith("上轨") and "因果P95" in n
        for n in names7
    )
    assert any(
        isinstance(n, str) and n.startswith("下轨") and "因果P95" in n
        for n in names7
    )
    up_tr = next(
        tr
        for tr in fig7.data
        if isinstance(getattr(tr, "name", None), str)
        and str(tr.name).startswith("上轨")
        and "因果P95" in str(tr.name)
    )
    dn_tr = next(
        tr
        for tr in fig7.data
        if isinstance(getattr(tr, "name", None), str)
        and str(tr.name).startswith("下轨")
        and "因果P95" in str(tr.name)
    )
    up_y = np.asarray(up_tr.y, dtype=float)
    dn_y = np.asarray(dn_tr.y, dtype=float)
    up_exp = upper_s.to_numpy(dtype=float)
    dn_exp = lower_s.to_numpy(dtype=float)
    m_up = np.isfinite(up_y) & np.isfinite(up_exp)
    m_dn = np.isfinite(dn_y) & np.isfinite(dn_exp)
    assert m_up.any() and m_dn.any()
    assert np.allclose(up_y[m_up], up_exp[m_up])
    assert np.allclose(dn_y[m_dn], dn_exp[m_dn])
    # Symmetry: upper/lower are mirror in log space around path.
    sym_m = np.isfinite(up_y) & np.isfinite(dn_y) & np.isfinite(path_y) & (path_y > 0)
    assert sym_m.any()
    assert np.allclose(
        np.log(up_y[sym_m] / path_y[sym_m]),
        -np.log(dn_y[sym_m] / path_y[sym_m]),
    )
    assert any(
        isinstance(n, str) and n.startswith("上轨") and "EWMA残差轨" in n
        for n in names7
    )
    assert any(
        isinstance(n, str) and n.startswith("下轨") and "EWMA残差轨" in n
        for n in names7
    )
    ewma_up = next(
        tr
        for tr in fig7.data
        if isinstance(getattr(tr, "name", None), str)
        and str(tr.name).startswith("上轨")
        and "EWMA残差轨" in str(tr.name)
    )
    ewma_y = np.asarray(ewma_up.y, dtype=float)
    both = np.isfinite(up_y) & np.isfinite(ewma_y)
    assert both.any()
    assert not np.allclose(up_y[both], ewma_y[both])
    names = {getattr(tr, "name", None) for tr in fig5.data}
    assert "公平年化收益" in names
    assert f"{EQUITY_ANCHOR_LABEL}公平 {ANCHOR_ERP_ANN:.1%}" in names
    assert "20日差额(实现−公平)" not in names
    assert "5日差额(实现−公平)" not in names
    assert fig5.layout.height == 1520

    need_gap = pd.DataFrame(
        {
            "as_of": dates,
            "r_need_ann": np.full(len(dates), ANCHOR_ERP_ANN),
            "r_need_ann_5": np.full(len(dates), ANCHOR_ERP_ANN * 2.0),
            "r_need_20d": np.full(
                len(dates), ((1.0 + ANCHOR_ERP_ANN) ** (20.0 / 252.0) - 1.0)
            ),
            "r_20d": np.linspace(-0.20, 0.05, len(dates)),
            "r_need_5d": np.full(
                len(dates), ((1.0 + ANCHOR_ERP_ANN * 2.0) ** (5.0 / 252.0) - 1.0)
            ),
            "r_5d": np.linspace(-0.08, 0.02, len(dates)),
        }
    )
    need_gap["gap_20d"] = need_gap["r_20d"] - need_gap["r_need_20d"]
    need_gap["gap_5d"] = need_gap["r_5d"] - need_gap["r_need_5d"]
    fig_gap = mod.build_candle_volume_vol_figure(
        ohlcv, vol_frame, need_frame=need_gap
    )
    names_gap = {getattr(tr, "name", None) for tr in fig_gap.data}
    assert "20日差额(实现−公平)" in names_gap
    assert "5日差额(实现−公平)" in names_gap
    assert "差额=0" in names_gap
    assert "公平年化收益(rv5)" in names_gap
    row5_fair = next(
        tr for tr in fig_gap.data if getattr(tr, "name", None) == "公平年化收益(rv5)"
    )
    assert np.allclose(np.asarray(row5_fair.y, dtype=float), ANCHOR_ERP_ANN * 2.0)
    row4_fair = next(
        tr for tr in fig_gap.data if getattr(tr, "name", None) == "公平年化收益"
    )
    assert np.allclose(np.asarray(row4_fair.y, dtype=float), ANCHOR_ERP_ANN)
    assert fig_gap.layout.height == 1520


def test_trend_state_series_and_summary():
    """Per-day MA20 trend series + snapshot: up/down/hug routing."""
    import importlib.util
    from pathlib import Path

    mod_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "plot_regime_transition_example.py"
    )
    spec = importlib.util.spec_from_file_location("plot_regime_example_mod", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # 60d flat then strong rally → confirmed up at end.
    dates = pd.bdate_range("2026-01-01", periods=80)
    close = pd.Series(
        np.r_[np.full(40, 1.0), np.linspace(1.0, 1.6, 40)], index=dates
    )
    tf = mod.trend_state_series(close)
    last = tf.iloc[-1]
    assert last["state"] == "up"
    assert last["dist_pct"] > 2.0
    snap = mod.trend_state_summary(close, dates[-1])
    assert snap["state"] == "up"
    assert snap["dist_pct"] > 2.0

    # Deep drop → confirmed down.
    close_dn = pd.Series(
        np.r_[np.full(40, 2.0), np.linspace(2.0, 1.2, 40)], index=dates
    )
    tf_dn = mod.trend_state_series(close_dn)
    assert tf_dn.iloc[-1]["state"] == "down"
    snap_dn = mod.trend_state_summary(close_dn, dates[-1])
    assert snap_dn["state"] == "down"
    assert snap_dn["dist_pct"] < -2.0

    # Hugging: close within ±2% of MA20 → hug, not a confirmed trend.
    close_hug = pd.Series(1.0 + np.sin(np.arange(80)) * 0.003, index=dates)
    snap_hug = mod.trend_state_summary(close_hug, dates[-1])
    assert snap_hug["state"] == "hug"


def test_candle_hover_includes_trend_state():
    """Candle hover text must show MA20 dist + 趋势 state + 状态≠预测 note."""
    import importlib.util
    from pathlib import Path

    mod_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "plot_regime_transition_example.py"
    )
    spec = importlib.util.spec_from_file_location("plot_regime_example_mod", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    dates = pd.bdate_range("2026-04-01", periods=60)
    close = np.r_[np.full(30, 1.0), np.linspace(1.0, 1.5, 30)]
    ohlcv = pd.DataFrame(
        {
            "datetime": dates,
            "$open": close,
            "$high": close + 0.02,
            "$low": close - 0.02,
            "$close": close,
            "$volume": np.full(60, 1_000_000.0),
            "$pct_change": np.r_[np.nan, np.diff(close) / close[:-1] * 100.0],
        }
    )
    close_s = pd.Series(close, index=pd.to_datetime(dates).normalize())
    vol_frame = compute_volatility_regime_frame(close_s, lookback=30, min_history=5)
    trend_frame = mod.trend_state_series(close_s)
    fig = mod.build_candle_volume_vol_figure(ohlcv, vol_frame)
    fig = mod.enrich_candle_hover_with_pct_chg(fig, ohlcv, trend_frame=trend_frame)
    candle = next(tr for tr in fig.data if getattr(tr, "type", None) == "candlestick")
    last_text = list(candle.hovertext)[-1]
    assert "MA20=" in last_text
    assert "趋势=趋势上涨" in last_text
    assert "状态≠预测" in last_text


def _load_plot_example_mod():
    import importlib.util
    from pathlib import Path

    mod_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "plot_regime_transition_example.py"
    )
    spec = importlib.util.spec_from_file_location("plot_regime_example_mod", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _extreme_drop_ohlcv() -> pd.DataFrame:
    dates = pd.bdate_range("2026-06-01", periods=40)
    close = np.full(40, 1.0)
    close[20] = 0.92  # -8% day
    close[21:] = 0.92
    pct = np.r_[np.nan, np.diff(close) / close[:-1] * 100.0]
    return pd.DataFrame(
        {
            "datetime": dates,
            "$open": close,
            "$high": close + 0.01,
            "$low": close - 0.01,
            "$close": close,
            "$volume": np.full(40, 1_000_000.0),
            "$pct_change": pct,
        }
    )


def test_extreme_drop_marker_visual_only():
    """单日跌≤-7% → 灰色「极端跌幅」中性标注；hover-silent，不抢 K 线 hover。"""
    mod = _load_plot_example_mod()
    ohlcv = _extreme_drop_ohlcv()
    dates = pd.to_datetime(ohlcv["datetime"]).dt.normalize()
    close_s = pd.Series(ohlcv["$close"].to_numpy(), index=dates)
    vol_frame = compute_volatility_regime_frame(close_s, lookback=30, min_history=5)
    fig = mod.build_candle_volume_vol_figure(ohlcv, vol_frame)
    fig = mod.annotate_extreme_drops(fig, ohlcv)
    names = [getattr(tr, "name", None) for tr in fig.data]
    assert mod.EXTREME_DROP_LEGEND in names
    drop_tr = next(tr for tr in fig.data if getattr(tr, "name", None) == mod.EXTREME_DROP_LEGEND)
    assert len(list(drop_tr.x)) == 1
    # 纯视觉层：不得再携带 hover（hovermode=closest 下会顶掉 K 线 hover）
    assert getattr(drop_tr, "hoverinfo", None) == "skip"
    assert getattr(drop_tr, "hovertemplate", None) is None
    assert getattr(drop_tr, "customdata", None) is None

    # No big drop → no marker.
    ohlcv2 = ohlcv.copy()
    ohlcv2["$pct_change"] = np.r_[np.nan, np.full(39, 0.1)]
    fig2 = mod.build_candle_volume_vol_figure(ohlcv2, vol_frame)
    fig2 = mod.annotate_extreme_drops(fig2, ohlcv2)
    names2 = [getattr(tr, "name", None) for tr in fig2.data]
    assert mod.EXTREME_DROP_LEGEND not in names2


def test_extreme_drop_skips_red_triangle_days():
    """红三角（quantile_switch & switch_side=down）的大跌日不画黑三角。"""
    mod = _load_plot_example_mod()
    ohlcv = _extreme_drop_ohlcv()
    drop_day = pd.to_datetime(ohlcv["datetime"]).dt.normalize()[20]
    dates = pd.to_datetime(ohlcv["datetime"]).dt.normalize()

    # 大跌日已有红三角 → 剔除
    overlay_red = pd.DataFrame(
        {
            "as_of": [drop_day],
            "quantile_switch": [True],
            "switch_side": ["down"],
        }
    )
    assert mod.compute_extreme_drop_dates(ohlcv, overlay_red) == set()

    # 非红三角日（无切换 / 上行切换）→ 保留
    overlay_up = pd.DataFrame(
        {
            "as_of": [drop_day],
            "quantile_switch": [True],
            "switch_side": ["up"],
        }
    )
    assert mod.compute_extreme_drop_dates(ohlcv, overlay_up) == {drop_day}
    assert mod.compute_extreme_drop_dates(ohlcv, None) == {drop_day}

    # marker 层同步：有红三角 → 不出现黑三角 trace
    close_s = pd.Series(ohlcv["$close"].to_numpy(), index=dates)
    vol_frame = compute_volatility_regime_frame(close_s, lookback=30, min_history=5)
    fig = mod.build_candle_volume_vol_figure(ohlcv, vol_frame)
    fig = mod.annotate_extreme_drops(fig, ohlcv, overlay=overlay_red)
    names = [getattr(tr, "name", None) for tr in fig.data]
    assert mod.EXTREME_DROP_LEGEND not in names


def test_extreme_drop_note_in_candle_hover():
    """极端跌幅提示并入 K 线 hover；红三角日不带「勿当看跌信号」文案。"""
    mod = _load_plot_example_mod()
    ohlcv = _extreme_drop_ohlcv()
    dates = pd.to_datetime(ohlcv["datetime"]).dt.normalize()
    drop_day = dates[20]
    close_s = pd.Series(ohlcv["$close"].to_numpy(), index=dates)
    vol_frame = compute_volatility_regime_frame(close_s, lookback=30, min_history=5)

    def candle_texts(fig):
        tr = next(t for t in fig.data if getattr(t, "type", None) == "candlestick")
        return list(tr.text)

    # 无红三角：hover 含极端跌幅提示
    fig = mod.build_candle_volume_vol_figure(ohlcv, vol_frame)
    dates_set = mod.compute_extreme_drop_dates(ohlcv, None)
    fig = mod.enrich_candle_hover_with_pct_chg(fig, ohlcv, extreme_dates=dates_set)
    texts = candle_texts(fig)
    assert "极端跌幅标注（视觉层，非交易信号）" in texts[20]
    assert "勿当看跌信号" in texts[20]
    assert "极端跌幅标注" not in texts[19]

    # 有红三角：extreme_dates 为空 → hover 无该文案
    overlay_red = pd.DataFrame(
        {
            "as_of": [drop_day],
            "quantile_switch": [True],
            "switch_side": ["down"],
        }
    )
    fig2 = mod.build_candle_volume_vol_figure(ohlcv, vol_frame)
    dates_set2 = mod.compute_extreme_drop_dates(ohlcv, overlay_red)
    fig2 = mod.enrich_candle_hover_with_pct_chg(fig2, ohlcv, extreme_dates=dates_set2)
    texts2 = candle_texts(fig2)
    assert "极端跌幅标注" not in texts2[20]
    assert "pct_chg=-8.00" in texts2[20]


def test_plot_ohlcv_qfq_removes_exdiv_cliff():
    """Plot OHLC qfq should smooth ETF 除权 cliffs the same way as signal closes."""
    from pipeline.price_adjustments import auto_qfq_adjust_long_frame

    dates = pd.bdate_range("2026-07-10", periods=6)
    # Mimic SZ159320-style split: ~1.92 → 0.626 overnight.
    close = np.array([1.90, 1.91, 1.924, 0.626, 0.604, 0.576], dtype=float)
    factor = 0.626 / 1.924
    df = pd.DataFrame(
        {
            "instrument": ["SZ159320"] * len(dates),
            "datetime": dates,
            "$open": close * 0.99,
            "$high": close * 1.01,
            "$low": close * 0.98,
            "$close": close,
            "$volume": np.full(len(dates), 1e6),
        }
    )
    adj, records = auto_qfq_adjust_long_frame(
        df,
        date_col="datetime",
        code_col="instrument",
        close_col="$close",
        threshold=0.25,
        extra_price_cols=["$open", "$high", "$low"],
    )
    assert not records.empty
    rets = adj["$close"].pct_change().to_numpy()
    assert np.nanmax(np.abs(rets)) < 0.25
    # Pre-split closes are scaled toward post-split level.
    assert adj["$close"].iloc[2] == pytest.approx(1.924 * factor, rel=1e-6)
    assert adj["$open"].iloc[2] == pytest.approx(df["$open"].iloc[2] * factor, rel=1e-6)


def _synthetic_trail_plot_df(n_plot: int = 80, n_hist: int = 600):
    """Long history + short plot window for fig7 trail tests."""
    dates = pd.bdate_range("2023-01-02", periods=n_hist + n_plot)
    drift = np.log(1.10) / 252.0
    close = np.exp(drift * np.arange(n_hist + n_plot, dtype=float))
    full = pd.DataFrame(
        {
            "datetime": dates,
            "$open": close,
            "$high": close * 1.01,
            "$low": close * 0.99,
            "$close": close,
            "$volume": np.full(n_hist + n_plot, 1e6),
        }
    )
    plot = full.iloc[-n_plot:].reset_index(drop=True)
    return full, plot


def test_realized_cagr_chord_endpoints_match_window_closes():
    """Helper chord (not plotted) still passes through first/last closes of last trail window."""
    mod = _load_plot_example_mod()
    full, plot = _synthetic_trail_plot_df()
    close_s = mod._asof_close_series(full)
    trail_y = float(mod.FAIR_PATH_LOG_TREND_TRAIL_YEARS)
    chord = mod.realized_cagr_chord(close_s, trail_years=trail_y)
    chord_plot = mod._align_series_to_plot(chord, plot)
    finite = np.isfinite(chord_plot.to_numpy(dtype=float))
    assert finite.any()
    idx = np.flatnonzero(finite)
    j0, jT = int(idx[0]), int(idx[-1])
    close_plot = plot["$close"].to_numpy(dtype=float)
    assert chord_plot.iloc[j0] == pytest.approx(close_plot[j0], rel=1e-4)
    assert chord_plot.iloc[jT] == pytest.approx(close_plot[jT], rel=1e-4)


def test_fig7_trail_shows_realized_cagr_in_legend_and_hover():
    """Fig7 path legend/hover expose causal endpoint CAGR; no separate price chord."""
    mod = _load_plot_example_mod()
    full, plot = _synthetic_trail_plot_df()
    from decision_pack.src.regime_transition_plot import compute_volatility_regime_frame

    close_s = mod._asof_close_series(full)
    vol_frame = compute_volatility_regime_frame(close_s, lookback=120)
    n_plot = len(plot)
    need_frame = pd.DataFrame(
        {
            "as_of": plot["datetime"],
            "r_need_ann": np.full(n_plot, 0.043),
            "rv20": np.full(n_plot, 0.2),
            "rv5": np.full(n_plot, 0.2),
            "r_20d": np.zeros(n_plot),
            "gap_20d": np.full(n_plot, -0.01),
            "r_need_ann_5": np.full(n_plot, 0.043),
            "r_5d": np.zeros(n_plot),
            "gap_5d": np.full(n_plot, -0.01),
        }
    )
    policy_frame = pd.DataFrame(
        {
            "as_of": plot["datetime"],
            "nav_hold_up": np.linspace(1.0, 1.1, n_plot),
            "nav_bh": np.linspace(1.0, 1.2, n_plot),
            "nav_cash": np.ones(n_plot),
            "nav_policy": np.linspace(1.0, 1.2, n_plot),
        }
    )
    fig = mod.build_candle_volume_vol_figure(
        plot,
        vol_frame,
        need_frame=need_frame,
        policy_frame=policy_frame,
        policy_meta={"policy": "bh", "policy_label": "BH"},
        train_cutoff="2026-04-01",
        fair_path_ohlcv=full,
        fair_path_extra_trail_years="1,0.5,1/4,1/12",
        hover_asset_name="煤炭ETF国泰",
    )
    names = [getattr(tr, "name", "") or "" for tr in fig.data]
    assert not any(n.startswith("实现年化") for n in names)
    path_names = [n for n in names if "公平路径" in n and "自有log趋势" in n]
    assert path_names
    assert all("实现CAGR" in n for n in path_names)
    path_tr = next(
        tr
        for tr in fig.data
        if getattr(tr, "mode", None) == "lines"
        and tr.customdata is not None
        and "当日路径年化" in str(tr.customdata[-1])
    )
    assert "当日实现年化" in str(path_tr.customdata[-1])
    assert "相对路径因果分位=" in str(path_tr.customdata[-1])
    path_hovers = [
        str(tr.customdata[-1])
        for tr in fig.data
        if getattr(tr, "mode", None) == "lines"
        and tr.customdata is not None
        and "相对路径因果分位=" in str(tr.customdata[-1])
    ]
    assert len(path_hovers) == 5  # fig7–fig11
    for last in path_hovers:
        i_pct = last.find("相对路径因果分位=")
        i_ewma = last.find("相对路径ewma分位=")
        i_gap = last.find("煤炭 ")
        assert 0 <= i_pct < i_ewma < i_gap
        assert "约 " in last[i_gap:]
        assert "双尺度·" in last
        assert i_gap < last.find("双尺度·")
    assert "实现CAGR" in str(path_tr.name)
    # As-of 实现CAGR is legend-only; hover points must not all embed the same suffix.
    assert "实现CAGR≈" not in str(path_tr.customdata[0])
    assert "实现CAGR≈" not in str(path_tr.customdata[-1])
    # Causal daily realized CAGR present on hover.
    import re

    def _parse_real(cd: str) -> float | None:
        m = re.search(r"当日实现年化\([^)]+\)=([-\d.]+)%", str(cd))
        return float(m.group(1)) / 100.0 if m else None

    first_r = _parse_real(path_tr.customdata[0])
    last_r = _parse_real(path_tr.customdata[-1])
    assert first_r is not None and last_r is not None
    assert abs(first_r - 0.10) < 0.05
    assert abs(last_r - 0.10) < 0.05


def test_causal_path_residual_signed_percentile_sign_and_rank():
    """Above path → +P…; below → −P…; smaller |gap| ranks below a prior extreme."""
    mod = _load_plot_example_mod()
    n = 80
    path = pd.Series(np.full(n, 100.0))
    # Early extreme +10%, then calm +1% → last percentile < 1 (not always max).
    close = path.copy()
    close.iloc[:30] = 110.0
    close.iloc[30:] = 101.0
    signed = mod.causal_path_residual_signed_percentile(path, close, min_bars=20)
    assert float(signed.iloc[25]) > 0
    assert 0 < float(signed.iloc[-1]) < 1.0
    assert float(signed.iloc[-1]) < float(signed.iloc[25])
    # Below path → negative.
    close_dn = path * 0.90
    signed_dn = mod.causal_path_residual_signed_percentile(path, close_dn, min_bars=20)
    assert float(signed_dn.iloc[-1]) < 0
    assert mod._fmt_signed_path_percentile(0.723).startswith("+P")
    assert mod._fmt_signed_path_percentile(-0.451).startswith("-P")
    assert mod.hover_asset_short_name("煤炭ETF国泰") == "煤炭"
    assert (
        mod.fmt_path_gap_hover_line("煤炭ETF国泰", "2026-08-28", 1.13, 1.0)
        == "煤炭 8/28 约 +13%"
    )


def test_ewma_path_residual_signed_percentile_sign_and_rank():
    """Above path → +; below → −; smaller |z| ranks below a prior extreme."""
    mod = _load_plot_example_mod()
    from decision_pack.scripts.scan_next_day_trigger_prices import (
        _ewma_state_from_path_close,
        _last_ewma_z,
        _signed_rank_last,
    )

    rng = np.random.default_rng(2)
    n = 120
    idx = pd.bdate_range("2020-01-02", periods=n)
    path = pd.Series(100.0 * np.exp(np.linspace(0.0, 0.2, n)), index=idx)
    close = path * np.exp(rng.normal(0.0, 0.02, n))
    close = close.copy()
    close.iloc[80] = float(path.iloc[80]) * float(np.exp(-0.08))
    close.iloc[-1] = float(path.iloc[-1]) * float(np.exp(0.005))
    signed = mod.ewma_path_residual_signed_percentile(path, close)
    assert float(signed.iloc[80]) < 0
    assert float(signed.iloc[-1]) > 0
    assert abs(float(signed.iloc[-1])) < abs(float(signed.iloc[80]))
    state = _ewma_state_from_path_close(path.iloc[:-1], close.iloc[:-1], min_bars=63)
    z_last = _last_ewma_z(float(path.iloc[-1]), float(close.iloc[-1]), state)
    got = _signed_rank_last(state.abs_z, z_last, min_bars=63)
    np.testing.assert_allclose(got, float(signed.iloc[-1]), rtol=1e-10, atol=1e-12)
    # RangeIndex close must pair by position, not by label.
    signed_pos = mod.ewma_path_residual_signed_percentile(
        path, pd.Series(close.to_numpy(dtype=float))
    )
    np.testing.assert_allclose(
        signed_pos.to_numpy(dtype=float),
        signed.to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )


def test_short_plot_window_keeps_full_history_percentiles_and_rails():
    """OOS chart must not restart residual history at the left edge of the plot."""
    mod = _load_plot_example_mod()
    n_hist = 700
    n_plot = 80
    n = n_hist + n_plot
    dates = pd.bdate_range("2022-01-03", periods=n)
    drift = np.log(1.08) / 252.0
    close = np.exp(drift * np.arange(n, dtype=float)) * 10.0
    # Isolated shocks before the plot window. One bar does not move the
    # 2y OLS path, so those |r| and |z| stay in the expanding sample while
    # the chart window itself is calm except a mild last-bar gap.
    close[200:n_hist:8] *= 1.50
    close[-1] *= 1.08
    full = pd.DataFrame(
        {
            "datetime": dates,
            "$open": close,
            "$high": close * 1.01,
            "$low": close * 0.99,
            "$close": close,
            "$volume": np.full(n, 1e6),
        }
    )
    plot = full.iloc[-n_plot:].reset_index(drop=True)
    close_s = mod._asof_close_series(full)
    vol_frame = compute_volatility_regime_frame(close_s, lookback=120)
    need_frame = pd.DataFrame(
        {
            "as_of": plot["datetime"],
            "r_need_ann": np.full(n_plot, 0.043),
            "rv20": np.full(n_plot, 0.2),
            "rv5": np.full(n_plot, 0.2),
            "r_20d": np.zeros(n_plot),
            "gap_20d": np.full(n_plot, -0.01),
            "r_need_ann_5": np.full(n_plot, 0.043),
            "r_5d": np.zeros(n_plot),
            "gap_5d": np.full(n_plot, -0.01),
        }
    )
    policy_frame = pd.DataFrame(
        {
            "as_of": plot["datetime"],
            "nav_hold_up": np.linspace(1.0, 1.1, n_plot),
            "nav_bh": np.linspace(1.0, 1.2, n_plot),
            "nav_cash": np.ones(n_plot),
            "nav_policy": np.linspace(1.0, 1.2, n_plot),
        }
    )
    fig = mod.build_candle_volume_vol_figure(
        plot,
        vol_frame,
        need_frame=need_frame,
        policy_frame=policy_frame,
        policy_meta={"policy": "bh", "policy_label": "BH"},
        train_cutoff="2026-04-01",
        fair_path_ohlcv=full,
    )
    years = float(mod.FAIR_PATH_LOG_TREND_TRAIL_YEARS)
    min_b = mod._trail_fit_min_bars(years)
    close_fit = mod._asof_close_series(full)
    erp_fit, _g = mod.trailing_log_trend_price_path(
        close_fit, trail_years=years, min_bars=min_b
    )
    pct_full = mod._align_series_to_plot(
        mod.causal_path_residual_signed_percentile(
            erp_fit, close_fit, min_bars=min_b
        ),
        plot,
    )
    ewma_full = mod._align_series_to_plot(
        mod.ewma_path_residual_signed_percentile(erp_fit, close_fit),
        plot,
    )
    upper_full, _, _ = mod.erp_path_symmetric_envelope(erp_fit, close_fit, q=0.95)
    upper_aln = mod._align_series_to_plot(upper_full, plot)
    erp_plot = mod._align_series_to_plot(erp_fit, plot)
    close_plot = pd.to_numeric(plot["$close"], errors="coerce")
    pct_win = mod.causal_path_residual_signed_percentile(
        erp_plot, close_plot, min_bars=min_b
    )
    ewma_win = mod.ewma_path_residual_signed_percentile(erp_plot, close_plot)
    upper_win, _, _ = mod.erp_path_symmetric_envelope(erp_plot, close_plot, q=0.95)
    assert abs(float(pct_full.iloc[-1]) - float(pct_win.iloc[-1])) > 0.05
    assert abs(float(ewma_full.iloc[-1]) - float(ewma_win.iloc[-1])) > 0.05
    assert abs(float(upper_aln.iloc[-1]) / float(upper_win.iloc[-1]) - 1.0) > 0.02

    path_tr = next(
        tr
        for tr in fig.data
        if getattr(tr, "mode", None) == "lines"
        and tr.customdata is not None
        and "相对路径因果分位=" in str(tr.customdata[-1])
    )
    last = str(path_tr.customdata[-1])
    assert (
        f"相对路径因果分位={mod._fmt_signed_path_percentile(pct_full.iloc[-1])}"
        in last
    )
    assert (
        f"相对路径ewma分位={mod._fmt_signed_path_percentile(ewma_full.iloc[-1])}"
        in last
    )
    p95 = next(
        tr
        for tr in fig.data
        if isinstance(getattr(tr, "name", None), str)
        and str(tr.name).startswith("上轨")
        and "因果P95" in str(tr.name)
    )
    np.testing.assert_allclose(
        float(np.asarray(p95.y, dtype=float)[-1]),
        float(upper_aln.iloc[-1]),
        rtol=1e-10,
        atol=1e-8,
    )


def test_fig7_realized_cagr_toggle_off_hides_legend_and_hover(monkeypatch):
    """FAIR_PATH_SHOW_REALIZED_CAGR=0 removes realized CAGR from legend and hover."""
    monkeypatch.setenv("FAIR_PATH_SHOW_REALIZED_CAGR", "0")
    mod = _load_plot_example_mod()
    full, plot = _synthetic_trail_plot_df(n_plot=60, n_hist=500)
    from decision_pack.src.regime_transition_plot import compute_volatility_regime_frame

    close_s = mod._asof_close_series(full)
    vol_frame = compute_volatility_regime_frame(close_s, lookback=120)
    n_plot = len(plot)
    need_frame = pd.DataFrame(
        {
            "as_of": plot["datetime"],
            "r_need_ann": np.full(n_plot, 0.043),
            "rv20": np.full(n_plot, 0.2),
            "rv5": np.full(n_plot, 0.2),
            "r_20d": np.zeros(n_plot),
            "gap_20d": np.full(n_plot, -0.01),
            "r_need_ann_5": np.full(n_plot, 0.043),
            "r_5d": np.zeros(n_plot),
            "gap_5d": np.full(n_plot, -0.01),
        }
    )
    policy_frame = pd.DataFrame(
        {
            "as_of": plot["datetime"],
            "nav_hold_up": np.linspace(1.0, 1.1, n_plot),
            "nav_bh": np.linspace(1.0, 1.2, n_plot),
            "nav_cash": np.ones(n_plot),
            "nav_policy": np.linspace(1.0, 1.2, n_plot),
        }
    )
    fig = mod.build_candle_volume_vol_figure(
        plot,
        vol_frame,
        need_frame=need_frame,
        policy_frame=policy_frame,
        policy_meta={"policy": "bh", "policy_label": "BH"},
        train_cutoff="2026-04-01",
        fair_path_ohlcv=full,
    )
    names = [getattr(tr, "name", "") or "" for tr in fig.data]
    assert not any("实现CAGR" in n for n in names)
    assert not any(n.startswith("实现年化") for n in names)
    path_tr = next(
        tr
        for tr in fig.data
        if getattr(tr, "mode", None) == "lines"
        and tr.customdata is not None
        and "当日路径年化" in str(tr.customdata[-1])
    )
    assert "当日实现年化" not in str(path_tr.customdata[-1])
