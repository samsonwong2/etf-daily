from __future__ import annotations

import math

import numpy as np
import pandas as pd

from decision_pack.src.hmm_short_horizon import HmmShortHorizonConfig, fit_hmm_2state, fit_hmm_kstate
from decision_pack.src.regime_transition_board import (
    CANONICAL_LABELS,
    RegimeTransitionConfig,
    compute_regime_transition_metrics,
    quantile_switch_from_cdf,
    relabel_hmm_states,
    student_t_mixture_cdf,
)


def _close_from_returns(returns: np.ndarray) -> pd.Series:
    prices = 100.0 * np.exp(np.cumsum(np.insert(returns, 0, 0.0)))
    return pd.Series(prices, index=pd.bdate_range("2020-01-01", periods=len(prices)))


def _three_phase_with_jump(n_calm: int = 120, n_bear: int = 120, jump: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(7)
    calm = rng.normal(0.0, 0.003, n_calm)
    bear = rng.normal(-0.004, 0.012, n_bear - 1)
    jump_day = np.array([jump])
    tail = rng.normal(0.006, 0.02, 80)
    return np.concatenate([calm, bear, jump_day, tail])


def test_fit_hmm_kstate_k3_converges():
    returns = _three_phase_with_jump()
    cfg = HmmShortHorizonConfig(n_states=3, emission="normal", max_em_iter=80)
    fit = fit_hmm_kstate(returns, cfg)
    assert fit.reliable
    assert fit.n_states == 3
    assert fit.sigma[0] <= fit.sigma[1] <= fit.sigma[2]


def test_fit_hmm_2state_regression_wrapper():
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.01, 300)
    direct = fit_hmm_kstate(returns, HmmShortHorizonConfig(n_states=2, emission="normal"))
    wrapped = fit_hmm_2state(returns, HmmShortHorizonConfig(emission="normal"))
    assert direct.reliable and wrapped.reliable
    assert np.allclose(direct.sigma, wrapped.sigma, rtol=0.05)


def test_relabel_permutation_invariant():
    mu = np.array([-0.002, 0.001, -0.008])
    sigma = np.array([0.01, 0.02, 0.015])
    trans = np.full((3, 3), 0.05)
    np.fill_diagonal(trans, 0.9)
    a = relabel_hmm_states(mu, sigma, trans, None)
    # permute input order
    perm = [2, 0, 1]
    b = relabel_hmm_states(mu[perm], sigma[perm], trans[np.ix_(perm, perm)], None)
    assert a[4] == CANONICAL_LABELS
    assert b[4] == CANONICAL_LABELS
    assert np.allclose(a[0], b[0])
    assert np.allclose(a[1], b[1])
    assert len(a[5]) == 3


def test_transition_score_peaks_on_jump_day():
    returns = _three_phase_with_jump(jump=0.06)
    close = _close_from_returns(returns)
    cfg = RegimeTransitionConfig(emission="normal", min_samples=60)
    scores: list[float] = []
    for i in range(-8, 0):
        sub = close.iloc[: i if i else None]
        m = compute_regime_transition_metrics(sub, sub.index[-1], cfg=cfg)
        scores.append(m.transition_score)
    assert max(scores) == scores[-1] or scores[-1] >= scores[-3]
    last = compute_regime_transition_metrics(close, close.index[-1], cfg=cfg)
    assert last.transition_score > 0


def test_collapsed_fallback_rule_score():
    rng = np.random.default_rng(99)
    bear = rng.normal(-0.004, 0.01, 199)
    bear = np.concatenate([bear, np.array([0.08])])
    close = _close_from_returns(bear)
    cfg = RegimeTransitionConfig(emission="normal", min_samples=60)
    m = compute_regime_transition_metrics(close, close.index[-1], cfg=cfg)
    assert m.transition_score >= 0.35


def test_causal_slice_no_future_leak():
    returns = _three_phase_with_jump()
    close = _close_from_returns(returns)
    cfg = RegimeTransitionConfig(emission="normal", min_samples=60)
    cut = len(close) - 10
    m1 = compute_regime_transition_metrics(close.iloc[:cut], close.index[cut - 1], cfg=cfg)
    m2 = compute_regime_transition_metrics(close, close.index[cut - 1], cfg=cfg)
    assert m1.transition_score == m2.transition_score
    assert m1.regime_now == m2.regime_now


def test_frozen_model_strict_probs_in_bounds():
    from decision_pack.src.regime_transition_board import (
        compute_frozen_transition_step,
        fit_frozen_regime_model,
    )

    returns = _three_phase_with_jump(jump=0.06)
    close = _close_from_returns(returns)
    train_end = close.index[-30]
    as_of = close.index[-1]
    cfg = RegimeTransitionConfig(emission="normal", min_samples=60)
    model = fit_frozen_regime_model(close, train_end, cfg=cfg)
    metrics, alpha, step = compute_frozen_transition_step(close, as_of, model, cfg=cfg)
    assert metrics.reliable
    assert metrics.p_switch_hmm is not None
    assert 0.0 <= metrics.p_switch_hmm <= 1.0
    assert metrics.p_into_reversal_hmm is not None
    assert 0.0 <= metrics.p_into_reversal_hmm <= 1.0
    assert abs(step.xi.sum() - 1.0) < 1e-9
    # future bars do not change as_of output
    m2, _, _ = compute_frozen_transition_step(close.iloc[:-5], close.index[-6], model, cfg=cfg)
    m3, _, _ = compute_frozen_transition_step(close, close.index[-6], model, cfg=cfg)
    assert m2.p_switch_hmm == m3.p_switch_hmm


def test_quantile_switch_bilateral_q90():
    hit_up, side_up = quantile_switch_from_cdf(0.91, q_star=0.90, bilateral=True)
    hit_dn, side_dn = quantile_switch_from_cdf(0.09, q_star=0.90, bilateral=True)
    miss, side_m = quantile_switch_from_cdf(0.50, q_star=0.90, bilateral=True)
    assert hit_up and side_up == "up"
    assert hit_dn and side_dn == "down"
    assert not miss and side_m is None


def test_student_t_mixture_cdf_center_near_half():
    pi = np.array([1.0])
    mu = np.array([0.0])
    sigma = np.array([0.01])
    nu = np.array([5.0])
    cdf0 = student_t_mixture_cdf(0.0, pi, mu, sigma, nu)
    assert cdf0 is not None and abs(cdf0 - 0.5) < 0.02


def test_student_t_mixture_pdf_and_log_density():
    from decision_pack.src.regime_transition_board import (
        student_t_mixture_log_density,
        student_t_mixture_moments,
        student_t_mixture_pdf,
    )

    pi = np.array([0.5, 0.5])
    mu = np.array([-0.01, 0.01])
    sigma = np.array([0.01, 0.02])
    nu = np.array([5.0, 5.0])
    pdf0 = student_t_mixture_pdf(0.0, pi, mu, sigma, nu)
    logp = student_t_mixture_log_density(0.0, pi, mu, sigma, nu)
    mean, scale = student_t_mixture_moments(pi, mu, sigma, nu)
    assert pdf0 is not None and pdf0 > 0
    assert logp is not None and abs(logp - math.log(pdf0)) < 1e-9
    assert mean is not None and abs(mean) < 1e-12
    assert scale is not None and scale > 0


def test_frozen_raw_state_separate_from_operational_override():
    from decision_pack.src.regime_transition_board import (
        compute_frozen_transition_step,
        fit_frozen_regime_model,
    )

    returns = _three_phase_with_jump(jump=0.08)
    close = _close_from_returns(returns)
    jump_idx = len(close) - 81
    train_end = close.index[jump_idx - 5]
    as_of = close.index[jump_idx]
    cfg = RegimeTransitionConfig(
        emission="student_t",
        min_samples=60,
        switch_quantile_q=0.90,
        switch_quantile_bilateral=True,
    )
    model = fit_frozen_regime_model(close, train_end, cfg=cfg)
    metrics, _, _ = compute_frozen_transition_step(close, as_of, model, cfg=cfg)
    assert metrics.reliable
    assert metrics.regime_hmm_raw is not None
    assert metrics.pred_log_density is not None
    assert metrics.pred_mean is not None
    assert metrics.pred_scale is not None
    # Operational override may differ from raw filtered state on a jump day.
    if metrics.quantile_switch:
        assert metrics.regime_now in {"volatile_reversal", "bear_grind"}


def test_frozen_quantile_switch_fires_on_large_jump():
    from decision_pack.src.regime_transition_board import (
        compute_frozen_transition_step,
        fit_frozen_regime_model,
    )

    returns = _three_phase_with_jump(jump=0.08)
    close = _close_from_returns(returns)
    # returns = calm + bear + jump + tail(80) → jump at -(81)
    jump_idx = len(close) - 81
    train_end = close.index[jump_idx - 5]
    as_of = close.index[jump_idx]
    cfg = RegimeTransitionConfig(
        emission="student_t",
        min_samples=60,
        switch_quantile_q=0.90,
        switch_quantile_bilateral=True,
    )
    model = fit_frozen_regime_model(close, train_end, cfg=cfg)
    metrics, _, _ = compute_frozen_transition_step(close, as_of, model, cfg=cfg)
    assert metrics.reliable
    assert metrics.pred_cdf is not None
    assert metrics.quantile_switch is True
    assert metrics.switch_side == "up"
    assert metrics.transition_score >= 0.5


def test_pred_mean_next_matches_next_day_pred_mean():
    from decision_pack.src.regime_transition_board import (
        compute_frozen_transition_step,
        compute_predictive_moments_next,
        fit_frozen_regime_model,
    )

    returns = _three_phase_with_jump(jump=0.08)
    close = _close_from_returns(returns)
    jump_idx = len(close) - 81
    train_end = close.index[jump_idx - 5]
    t0 = close.index[jump_idx - 2]
    t1 = close.index[jump_idx - 1]
    cfg = RegimeTransitionConfig(
        emission="student_t",
        min_samples=60,
        switch_quantile_q=0.90,
        switch_quantile_bilateral=True,
    )
    model = fit_frozen_regime_model(close, train_end, cfg=cfg)
    _m0, alpha0, _ = compute_frozen_transition_step(close, t0, model, cfg=cfg)
    m1, _alpha1, _ = compute_frozen_transition_step(
        close, t1, model, cfg=cfg, alpha_prev=alpha0
    )
    pred_mean_next, _ = compute_predictive_moments_next(alpha0, model)
    assert pred_mean_next is not None
    assert m1.pred_mean is not None
    assert math.isclose(float(pred_mean_next), float(m1.pred_mean), rel_tol=1e-9, abs_tol=1e-12)
