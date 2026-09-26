"""Tests for tomorrow (H=1) candidate digest."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from etf_daily.lib.garch_short_horizon_board import (
    GarchShortHorizonConfig,
    GarchShortHorizonMetrics,
)
from etf_daily.lib.hmm_short_horizon import (
    HmmShortHorizonConfig,
    HmmShortHorizonMetrics,
)
from etf_daily.lib.tomorrow_digest import (
    TOMORROW_DIGEST_COLUMNS,
    TopKConfig,
    TomorrowDigestConfig,
    build_tomorrow_digest_frame,
    calibration_quality_row,
    combined_credibility,
    compute_pick_score,
    compute_upside_skew,
    credibility_today,
    pool_median_calibration,
    select_topk_picks,
)


def test_calibration_quality_monotone_in_brier():
    cfg = TomorrowDigestConfig()
    good, _ = calibration_quality_row(
        {"n_obs": 40, "brier": 0.30, "coverage_q05": 0.05, "coverage_q95": 0.95},
        pool_median=0.5,
        cfg=cfg,
    )
    bad, _ = calibration_quality_row(
        {"n_obs": 40, "brier": 0.65, "coverage_q05": 0.05, "coverage_q95": 0.95},
        pool_median=0.5,
        cfg=cfg,
    )
    assert good > bad


def test_credibility_zero_when_unreliable():
    cfg = TomorrowDigestConfig()
    cal, _ = calibration_quality_row(
        {"n_obs": 30, "brier": 0.40, "coverage_q05": 0.05, "coverage_q95": 0.95},
        pool_median=0.5,
        cfg=cfg,
    )
    assert credibility_today(p_up=0.6, p_down=0.2, reliable=False, cal_quality=cal) == 0.0
    assert credibility_today(p_up=0.6, p_down=0.2, reliable=True, cal_quality=cal) > 0.0


def test_combined_credibility_modes():
    assert combined_credibility(0.2, 0.1, garch_reliable=True, hmm_reliable=True)[0] == 0.1
    assert combined_credibility(0.2, 0.1, garch_reliable=True, hmm_reliable=True, mode="mean")[0] == pytest.approx(0.15)
    cred, label = combined_credibility(0.2, 0.0, garch_reliable=True, hmm_reliable=False)
    assert cred == 0.2
    assert label == "仅GARCH"
    cred2, label2 = combined_credibility(0.0, 0.0, garch_reliable=False, hmm_reliable=False)
    assert cred2 == 0.0
    assert label2 == "不可信"


def test_combined_credibility_three_way():
    cred, label = combined_credibility(
        0.3,
        0.2,
        garch_reliable=True,
        hmm_reliable=True,
        transition_cred=0.15,
        transition_reliable=True,
    )
    assert cred == 0.15
    assert label == "三模可信"
    cred_mean, label_mean = combined_credibility(
        0.3,
        0.2,
        garch_reliable=True,
        hmm_reliable=True,
        transition_cred=0.15,
        transition_reliable=True,
        mode="mean",
    )
    assert cred_mean == pytest.approx(0.2166666667)
    assert label_mean == "三模可信"


def test_combined_credibility_transition_only():
    cred, label = combined_credibility(
        0.0,
        0.0,
        garch_reliable=False,
        hmm_reliable=False,
        transition_cred=0.42,
        transition_reliable=True,
    )
    assert cred == 0.42
    assert label == "仅跃迁"


def test_combined_credibility_transition_none_regression():
    cred, label = combined_credibility(0.2, 0.1, garch_reliable=True, hmm_reliable=True)
    assert cred == 0.1
    assert label == "双模可信"


def test_pool_median_calibration_empty():
    cfg = TomorrowDigestConfig()
    assert pool_median_calibration(pd.DataFrame(), cfg=cfg) == 0.5


def _make_close_from_returns(returns: np.ndarray) -> pd.Series:
    prices = 100.0 * np.exp(np.cumsum(np.insert(returns, 0, 0.0)))
    return pd.Series(prices, index=pd.bdate_range("2020-01-01", periods=len(prices)))


def test_build_digest_candidates_only_and_columns(monkeypatch):
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, 0.01, size=280)
    close = _make_close_from_returns(returns)
    end = close.index[-1]

    cluster = pd.DataFrame(
        [
            {"code": "EQ1", "name": "held", "weight": 0.5, "flags": ""},
            {"code": "EQ2", "name": "cand", "weight": 0.0, "flags": ""},
        ]
    )
    panel = pd.DataFrame({c: close for c in ("EQ1", "EQ2")})

    g_metrics = GarchShortHorizonMetrics(
        horizon=1,
        sigma_garch_1d=0.02,
        sigma_garch_h=0.02,
        vol_ann_5d=0.3,
        vol_ann_20d=0.25,
        vol_ratio_5_20=1.2,
        nu_t=5.0,
        arch_pvalue=0.01,
        p_up=0.35,
        p_down=0.30,
        p_timeout=0.35,
        barrier=0.02,
        q05=-0.03,
        q50=0.0,
        q95=0.03,
        var95_cond=-0.02,
        cvar95_cond=-0.025,
        dgp_basis="GJR-t",
        reliable=True,
        basis="GARCH-DGP·GJR-t",
        n_samples=252,
    )
    h_metrics = HmmShortHorizonMetrics(
        horizon=1,
        p_high=0.8,
        p_low=0.2,
        regime_now="高波动",
        sigma_mix_1d=0.02,
        sigma_mix_h=0.02,
        mu_mix_1d=0.0,
        p_up=0.4,
        p_down=0.35,
        p_timeout=0.25,
        barrier=0.04,
        q05=-0.04,
        q50=0.0,
        q95=0.04,
        reliable=True,
        basis="HMM-2·student_t·§34",
        regime_separated=False,
    )

    def fake_garch(series, end_ts, **kwargs):
        return g_metrics

    def fake_hmm(series, end_ts, **kwargs):
        return h_metrics

    monkeypatch.setattr(
        "etf_daily.lib.tomorrow_digest.compute_garch_short_horizon_metrics",
        fake_garch,
    )
    monkeypatch.setattr(
        "etf_daily.lib.tomorrow_digest.compute_hmm_short_horizon_metrics",
        fake_hmm,
    )
    monkeypatch.setattr(
        "etf_daily.lib.tomorrow_digest.fit_garch_dynamics",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "etf_daily.lib.tomorrow_digest.fit_hmm_2state",
        lambda *a, **k: None,
    )

    cal = pd.DataFrame(
        [
            {
                "code": "EQ2",
                "n_obs": 25,
                "brier": 0.45,
                "coverage_q05": 0.06,
                "coverage_q95": 0.94,
                "model": "garch",
                "horizon": 1,
            }
        ]
    )
    digest_cfg = TomorrowDigestConfig(candidates_only=True, horizon=1)
    frame = build_tomorrow_digest_frame(
        cluster,
        panel,
        end.strftime("%Y-%m-%d"),
        digest_cfg=digest_cfg,
        garch_board_cfg=GarchShortHorizonConfig(),
        hmm_board_cfg=HmmShortHorizonConfig(),
        garch_cal=cal,
        hmm_cal=pd.DataFrame(),
    )

    assert list(frame.reindex(columns=list(TOMORROW_DIGEST_COLUMNS)).columns) == list(
        TOMORROW_DIGEST_COLUMNS
    )
    assert len(frame) == 1
    assert frame.iloc[0]["code"] == "EQ2"
    assert frame.iloc[0]["hmm_可信度"] == "0.000"
    assert frame.iloc[0]["可靠"] == "仅GARCH"
    assert "HMM塌缩" in str(frame.iloc[0]["依据"])


def test_compute_pick_score_monotone():
    cfg = TopKConfig(
        min_credibility=0.003,
        min_direction=0.0,
        rank_rule="dir_x_cred",
        require_positive_direction=False,
    )
    high = compute_pick_score(0.05, 0.02, topk_cfg=cfg, for_ranking=True)
    low = compute_pick_score(0.02, 0.02, topk_cfg=cfg, for_ranking=True)
    assert high > low
    assert not math.isfinite(compute_pick_score(-0.01, 0.02, topk_cfg=cfg, for_ranking=False))
    assert not math.isfinite(compute_pick_score(0.05, 0.001, topk_cfg=cfg, for_ranking=False))
    assert compute_pick_score(0.05, 0.001, topk_cfg=cfg, for_ranking=True) == pytest.approx(0.00005)


def test_compute_pick_score_upside_x_cred_ranks_by_upside():
    cfg = TopKConfig(rank_rule="upside_x_cred", require_positive_direction=True)
    cred = 0.02
    high_up = compute_pick_score(0.05, cred, topk_cfg=cfg, upside_skew=0.06, for_ranking=True)
    low_up = compute_pick_score(0.05, cred, topk_cfg=cfg, upside_skew=0.01, for_ranking=True)
    assert high_up > low_up
    assert high_up == pytest.approx(0.06 * cred)
    assert not math.isfinite(
        compute_pick_score(-0.01, cred, topk_cfg=cfg, upside_skew=0.06, for_ranking=True)
    )


def test_compute_pick_score_require_positive_direction():
    cfg = TopKConfig(rank_rule="dir_x_cred", require_positive_direction=True)
    assert not math.isfinite(compute_pick_score(-0.02, 0.03, topk_cfg=cfg, for_ranking=True))
    assert compute_pick_score(0.02, 0.03, topk_cfg=cfg, for_ranking=True) == pytest.approx(0.0006)


def test_compute_upside_skew():
    assert compute_upside_skew(-0.02, 0.05) == pytest.approx(0.03)
    assert compute_upside_skew(None, 0.05) is None


def test_select_topk_filters_avoid_and_lists_negative():
    digest = pd.DataFrame(
        [
            {
                "code": "A1",
                "name": "good",
                "close": 1.0,
                "as_of": "2026-07-15",
                "_g_dir": 0.05,
                "_combined_sort": 0.02,
                "_g_q05": -0.01,
                "_g_q95": 0.02,
                "_upside_skew": 0.01,
            },
            {
                "code": "A2",
                "name": "neg",
                "close": 1.0,
                "as_of": "2026-07-15",
                "_g_dir": -0.03,
                "_combined_sort": 0.02,
                "_g_q05": -0.02,
                "_g_q95": 0.01,
                "_upside_skew": -0.01,
            },
            {
                "code": "A3",
                "name": "avoid",
                "close": 1.0,
                "as_of": "2026-07-15",
                "_g_dir": 0.08,
                "_combined_sort": 0.03,
                "_g_q05": -0.01,
                "_g_q95": 0.03,
                "_upside_skew": 0.02,
            },
        ]
    )
    cluster = pd.DataFrame(
        [
            {"code": "A1", "name": "good", "weight": 0.0, "barbell_leg": "risk_leg"},
            {"code": "A2", "name": "neg", "weight": 0.0, "barbell_leg": "risk_leg"},
            {"code": "A3", "name": "avoid", "weight": 0.0, "barbell_leg": "avoid"},
        ]
    )
    cfg = TomorrowDigestConfig(
        topk=TopKConfig(
            k=5,
            min_credibility=0.003,
            candidates_only=True,
            exclude_avoid_leg=True,
            rank_all=True,
            rank_rule="dir_x_cred",
            require_positive_direction=False,
        )
    )
    picks = select_topk_picks(digest, cluster, digest_cfg=cfg)
    assert list(picks["code"]) == ["A1", "A2"]
    assert picks.iloc[0]["pick_score"] == pytest.approx(0.05 * 0.02, rel=1e-6)
    assert picks.iloc[1]["pick_score"] == pytest.approx(-0.03 * 0.02, rel=1e-6)
    assert picks.iloc[0]["equal_weight_pct"] == pytest.approx(20.0)
    assert pd.isna(picks.iloc[1]["equal_weight_pct"])


def test_select_topk_avoid_leg_penalty_preserves_order():
    digest = pd.DataFrame(
        [
            {
                "code": "C1",
                "name": "avoid-high",
                "close": 1.0,
                "as_of": "2026-07-15",
                "_g_dir": 0.10,
                "_combined_sort": 0.04,
                "_g_q05": -0.02,
                "_g_q95": 0.08,
                "_upside_skew": 0.06,
            },
            {
                "code": "C2",
                "name": "risk-mid",
                "close": 1.0,
                "as_of": "2026-07-15",
                "_g_dir": 0.08,
                "_combined_sort": 0.04,
                "_g_q05": -0.01,
                "_g_q95": 0.05,
                "_upside_skew": 0.04,
            },
            {
                "code": "C3",
                "name": "risk-low",
                "close": 1.0,
                "as_of": "2026-07-15",
                "_g_dir": 0.06,
                "_combined_sort": 0.04,
                "_g_q05": -0.01,
                "_g_q95": 0.03,
                "_upside_skew": 0.02,
            },
        ]
    )
    cluster = pd.DataFrame(
        [
            {"code": "C1", "name": "avoid-high", "weight": 0.0, "barbell_leg": "avoid"},
            {"code": "C2", "name": "risk-mid", "weight": 0.0, "barbell_leg": "risk_leg"},
            {"code": "C3", "name": "risk-low", "weight": 0.0, "barbell_leg": "risk_leg"},
        ]
    )
    cfg = TomorrowDigestConfig(
        topk=TopKConfig(
            k=3,
            rank_rule="upside_x_cred",
            require_positive_direction=True,
            exclude_avoid_leg=False,
            avoid_leg_penalty=0.5,
            rank_all=True,
        )
    )
    picks = select_topk_picks(digest, cluster, digest_cfg=cfg)
    assert list(picks["code"]) == ["C2", "C1", "C3"]
    assert picks.iloc[0]["pick_score"] == pytest.approx(0.04 * 0.04, rel=1e-6)
    assert picks.iloc[1]["pick_score"] == pytest.approx(0.5 * 0.04 * 0.06, rel=1e-6)


def test_select_topk_equal_weight_and_tiebreak():
    digest = pd.DataFrame(
        [
            {
                "code": "B2",
                "name": "b",
                "close": 1.0,
                "as_of": "2026-07-15",
                "_g_dir": 0.04,
                "_combined_sort": 0.02,
                "_g_q05": -0.01,
                "_g_q95": 0.02,
                "_upside_skew": 0.01,
            },
            {
                "code": "B1",
                "name": "a",
                "close": 1.0,
                "as_of": "2026-07-15",
                "_g_dir": 0.04,
                "_combined_sort": 0.02,
                "_g_q05": -0.01,
                "_g_q95": 0.02,
                "_upside_skew": 0.01,
            },
        ]
    )
    cluster = pd.DataFrame(
        [
            {"code": "B1", "name": "a", "weight": 0.0, "barbell_leg": "risk_leg"},
            {"code": "B2", "name": "b", "weight": 0.0, "barbell_leg": "risk_leg"},
        ]
    )
    cfg = TomorrowDigestConfig(
        topk=TopKConfig(
            k=2,
            min_credibility=0.003,
            rank_all=True,
            rank_rule="dir_x_cred",
            require_positive_direction=False,
        )
    )
    picks = select_topk_picks(digest, cluster, digest_cfg=cfg)
    assert list(picks["code"]) == ["B1", "B2"]
    assert len(picks) == 2
    assert picks["equal_weight_pct"].tolist() == [50.0, 50.0]
    assert picks["equal_weight_pct"].sum() == pytest.approx(100.0)
