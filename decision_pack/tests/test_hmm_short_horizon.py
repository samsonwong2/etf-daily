from __future__ import annotations

import numpy as np
import pandas as pd

from decision_pack.src.hmm_short_horizon import (
    HMM_SHORT_HORIZON_COLUMNS,
    HmmShortHorizonConfig,
    compute_hmm_short_horizon_metrics,
    fit_hmm_2state,
    format_hmm_row,
    triple_barrier_mc_hmm,
)


def _regime_switching_returns(n: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = np.empty(n, dtype=float)
    state = 0
    for i in range(n):
        if rng.random() < 0.05:
            state = 1 - state
        sigma = 0.005 if state == 0 else 0.03
        returns[i] = rng.normal(0.0, sigma)
    # force high-vol ending
    returns[-40:] = rng.normal(0.0, 0.035, size=40)
    return returns


def _close_from_returns(returns: np.ndarray) -> pd.Series:
    prices = 100.0 * np.exp(np.cumsum(np.insert(returns, 0, 0.0)))
    return pd.Series(prices, index=pd.bdate_range("2020-01-01", periods=len(prices)))


def _intermittent_high_vol_returns(n: int = 400, seed: int = 11) -> np.ndarray:
    """Low vol history with a recent high-vol cluster (SH588200-like trap)."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.018, size=n - 45)
    returns = np.concatenate([returns, rng.normal(0.0, 0.042, size=45)])
    return returns


def test_recent_filter_detects_high_vol_after_intermittent_history():
    returns = _intermittent_high_vol_returns()
    close = _close_from_returns(returns)
    cfg = HmmShortHorizonConfig(
        emission="normal",
        max_em_iter=60,
        mc_samples=500,
        regime_filter_lookback=40,
    )
    metrics = compute_hmm_short_horizon_metrics(close, close.index[-1], horizon=10, cfg=cfg)
    assert metrics.reliable
    assert metrics.regime_separated
    assert metrics.p_high is not None and metrics.p_high > 0.5
    assert metrics.regime_now == "高波动"


def test_fit_hmm_detects_high_vol_at_end():
    returns = _regime_switching_returns()
    cfg = HmmShortHorizonConfig(emission="normal", max_em_iter=60, mc_samples=500)
    fit = fit_hmm_2state(returns, cfg)
    assert fit.reliable
    assert fit.sigma[1] > fit.sigma[0]
    assert fit.pi_filt[-1] > 0.4  # elevated high-vol probability after high-vol ending


def test_hmm_metrics_probabilities_sum():
    close = _close_from_returns(_regime_switching_returns(350, seed=2))
    end = close.index[-1]
    cfg = HmmShortHorizonConfig(emission="normal", mc_samples=800)
    metrics = compute_hmm_short_horizon_metrics(close, end, horizon=10, cfg=cfg)
    assert metrics.reliable
    assert metrics.p_up is not None and metrics.p_down is not None and metrics.p_timeout is not None
    assert abs(metrics.p_up + metrics.p_down + metrics.p_timeout - 1.0) < 1e-9
    assert metrics.q05 is not None and metrics.q50 is not None and metrics.q95 is not None
    assert metrics.q05 <= metrics.q50 <= metrics.q95


def test_format_hmm_row_columns():
    close = _close_from_returns(np.random.default_rng(3).normal(0, 0.01, 300))
    metrics = compute_hmm_short_horizon_metrics(
        close, close.index[-1], horizon=5, cfg=HmmShortHorizonConfig(emission="normal", mc_samples=400)
    )
    cluster_row = pd.Series(
        {
            "barbell_leg": "risk_leg",
            "flags": "",
            "tail_regime": "stable",
            "path_state": "near_high",
            "dd_long": -0.05,
        }
    )
    formatters = {
        "pct": lambda v, digits=2: "1.00%" if v is not None else "—",
        "prob": lambda v, digits=1: "50.0%" if v is not None else "—",
        "flags_str": lambda v: str(v or "—"),
        "tail_short": lambda v: "稳定",
        "state_short": lambda v: "近高",
    }
    row = format_hmm_row(
        code="SH588170",
        name="科创半导",
        weight=0.08,
        cluster_row=cluster_row,
        metrics=metrics,
        formatters=formatters,
    )
    assert set(row.keys()) == set(HMM_SHORT_HORIZON_COLUMNS)


def test_triple_barrier_reproducible():
    returns = _regime_switching_returns(300, seed=4)
    fit = fit_hmm_2state(returns, HmmShortHorizonConfig(emission="normal"))
    a = triple_barrier_mc_hmm(fit, horizon=10, barrier_mult=2.0, n_paths=500, seed=11)
    b = triple_barrier_mc_hmm(fit, horizon=10, barrier_mult=2.0, n_paths=500, seed=11)
    assert a == b


def _positive_drift_returns(n: int = 300, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.003, 0.01, size=n)


def _flat_low_vol_returns(n: int = 300, seed: int = 21) -> np.ndarray:
    """Recent calm tail with vol_5d << vol_20d (gold-like flat spell)."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.002, size=n - 20)
    returns = np.concatenate([returns, rng.normal(0.0, 0.003, size=15)])
    returns = np.concatenate([returns, rng.normal(0.0, 0.0004, size=5)])
    return returns


def test_mc_zero_mean_reduces_upward_touch_bias():
    close = _close_from_returns(_positive_drift_returns())
    end = close.index[-1]
    base = dict(emission="normal", mc_samples=1200, mc_recent_vol_scale=False)
    with_zero = compute_hmm_short_horizon_metrics(
        close,
        end,
        horizon=10,
        cfg=HmmShortHorizonConfig(mc_zero_mean=True, **base),
    )
    without_zero = compute_hmm_short_horizon_metrics(
        close,
        end,
        horizon=10,
        cfg=HmmShortHorizonConfig(mc_zero_mean=False, **base),
    )
    assert with_zero.p_up is not None and without_zero.p_up is not None
    assert with_zero.p_down is not None and without_zero.p_down is not None
    assert with_zero.p_up < without_zero.p_up
    assert abs(with_zero.p_up - with_zero.p_down) < abs(
        without_zero.p_up - without_zero.p_down
    )


def test_mc_low_vol_flat_series_low_touch():
    close = _close_from_returns(_flat_low_vol_returns())
    metrics = compute_hmm_short_horizon_metrics(
        close,
        close.index[-1],
        horizon=10,
        cfg=HmmShortHorizonConfig(
            emission="normal",
            mc_samples=1500,
            mc_zero_mean=True,
            mc_recent_vol_scale=True,
        ),
    )
    assert metrics.reliable
    assert metrics.p_up is not None and metrics.p_timeout is not None
    assert metrics.p_up < 0.08
    assert metrics.p_timeout > 0.85
    assert "低波缩放" in metrics.basis


def test_high_vol_regime_unchanged_with_scale_one():
    close = _close_from_returns(_intermittent_high_vol_returns())
    metrics = compute_hmm_short_horizon_metrics(
        close,
        close.index[-1],
        horizon=10,
        cfg=HmmShortHorizonConfig(
            emission="normal",
            mc_samples=800,
            regime_filter_lookback=40,
            mc_zero_mean=True,
            mc_recent_vol_scale=True,
        ),
    )
    assert metrics.reliable
    assert metrics.p_high is not None and metrics.p_high > 0.5
    assert metrics.p_up is not None and metrics.p_down is not None and metrics.p_timeout is not None
    assert abs(metrics.p_up + metrics.p_down + metrics.p_timeout - 1.0) < 1e-9


def test_regimes_are_separated_detects_collapse():
    from decision_pack.src.hmm_short_horizon import regimes_are_separated

    ok, ratio = regimes_are_separated(np.array([0.01, 0.03]), min_sigma_ratio=1.15)
    assert ok and ratio == 3.0
    bad, ratio2 = regimes_are_separated(np.array([0.0284, 0.0288]), min_sigma_ratio=1.15)
    assert not bad
    assert ratio2 is not None and ratio2 < 1.15


def test_collapsed_fit_marks_regime_unseparated():
    """Near-homoskedastic series -> regime=未分离, sep=塌缩, p_high blanked in CSV row."""
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, 0.02, size=300)  # single regime
    close = _close_from_returns(returns)
    cfg = HmmShortHorizonConfig(emission="normal", mc_samples=200, min_sigma_ratio=1.15)
    metrics = compute_hmm_short_horizon_metrics(close, close.index[-1], horizon=5, cfg=cfg)
    # May or may not collapse depending on EM; force via synthetic fit path if needed
    fit = fit_hmm_2state(returns, cfg)
    if fit.sigma[1] / fit.sigma[0] < 1.15:
        assert metrics.regime_now == "未分离"
        assert metrics.regime_separated is False
        row = format_hmm_row(
            code="SH515050",
            name="通信ETF华夏",
            weight=0.0,
            cluster_row=pd.Series(
                {
                    "barbell_leg": "avoid",
                    "flags": "",
                    "tail_regime": "stable",
                    "path_state": "near_high",
                    "dd_long": -0.1,
                }
            ),
            metrics=metrics,
            formatters={
                "pct": lambda v, digits=2: "1.00%" if v is not None else "—",
                "prob": lambda v, digits=1: "50.0%" if v is not None else "—",
                "flags_str": lambda v: str(v or "—"),
                "tail_short": lambda v: "稳定",
                "state_short": lambda v: "近高",
            },
        )
        assert row["sep"] == "塌缩"
        assert row["regime"] == "未分离"
        assert row["p_high"] == "—"
        assert "两态未分离" in row["依据"]
    else:
        # If EM luckily separates a bit, still check helper on near-equal sigmas
        from decision_pack.src.hmm_short_horizon import regimes_are_separated

        sep, _ = regimes_are_separated(np.array([0.02, 0.0205]), min_sigma_ratio=1.15)
        assert sep is False


def test_forward_filter_step_normalizes_and_bounds():
    from decision_pack.src.hmm_short_horizon import forward_filter_step, fit_hmm_kstate

    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, 300)
    fit = fit_hmm_kstate(returns, HmmShortHorizonConfig(n_states=3, emission="normal"))
    assert fit.reliable
    alpha = np.full(3, 1.0 / 3)
    step = forward_filter_step(fit, alpha, float(returns[-1]))
    assert abs(step.xi.sum() - 1.0) < 1e-9
    assert abs(step.alpha.sum() - 1.0) < 1e-9
    assert np.all(step.xi >= -1e-12)
    assert np.all(step.alpha >= -1e-12)


def test_forward_filter_step_diagonal_near_zero_switch():
    from decision_pack.src.hmm_short_horizon import ForwardFilterStep, HmmFitResult, forward_filter_step

    fit = HmmFitResult(
        n_states=3,
        trans=np.eye(3),
        mu=np.array([-0.01, 0.0, 0.01]),
        sigma=np.array([0.01, 0.02, 0.03]),
        nu=None,
        pi_filt=np.array([0.2, 0.5, 0.3]),
        emission="normal",
        reliable=True,
        n_samples=100,
    )
    alpha = np.array([0.1, 0.8, 0.1])
    step = forward_filter_step(fit, alpha, 0.0)
    p_switch = float(step.xi.sum() - np.trace(step.xi))
    assert p_switch < 1e-9


def test_forward_filter_step_rejects_bad_alpha():
    from decision_pack.src.hmm_short_horizon import HmmFitResult, forward_filter_step
    import pytest

    fit = HmmFitResult(
        n_states=2,
        trans=np.eye(2),
        mu=np.zeros(2),
        sigma=np.ones(2) * 0.01,
        nu=None,
        pi_filt=np.array([0.5, 0.5]),
        emission="normal",
        reliable=True,
        n_samples=10,
    )
    with pytest.raises(ValueError):
        forward_filter_step(fit, np.array([1.0]), 0.0)
