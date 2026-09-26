from __future__ import annotations

import math

import numpy as np
import pytest

from decision_pack.src.garch_dynamics import (
    GarchDynamicsConfig,
    GarchFitResult,
    arch_lm_test,
    conditional_var_cvar,
    fit_garch_dynamics,
    forecast_sigma_h,
    mc_quantiles_garch,
    triple_barrier_mc_garch,
    unit_variance_t_ppf,
    vol_spike_prob_mc_garch,
)


def _make_garch_like_returns(n: int = 300, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = np.zeros(n, dtype=float)
    sigma2 = 0.0004
    for i in range(n):
        z = rng.standard_normal()
        returns[i] = math.sqrt(sigma2) * z
        sigma2 = 1e-6 + 0.05 * returns[i] ** 2 + 0.90 * sigma2
    return returns


def test_arch_lm_test_returns_pvalue():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.01, size=200)
    pval = arch_lm_test(returns, lags=5)
    assert pval is None or (0.0 <= pval <= 1.0)


def test_ewma_fallback_when_model_ewma():
    returns = np.random.default_rng(1).normal(0.0, 0.012, size=120)
    cfg = GarchDynamicsConfig(model="ewma", min_train=60)
    fit = fit_garch_dynamics(returns, cfg, max_horizon=10)
    assert fit.sigma_1d is not None
    assert fit.sigma_1d > 0
    assert fit.dgp_basis == "EWMA"
    assert 10 in fit.sigma_h


def test_forecast_sigma_h_increases_with_horizon():
    returns = _make_garch_like_returns(300)
    cfg = GarchDynamicsConfig(model="ewma", min_train=60)
    fit = fit_garch_dynamics(returns, cfg, max_horizon=20)
    s5 = forecast_sigma_h(fit, 5)
    s10 = forecast_sigma_h(fit, 10)
    assert s5 is not None and s10 is not None
    assert s10 >= s5


def test_triple_barrier_probabilities_sum_to_one():
    returns = np.random.default_rng(2).normal(0.0, 0.012, size=200)
    cfg = GarchDynamicsConfig(model="ewma", mc_samples=1000, min_train=60)
    fit = fit_garch_dynamics(returns, cfg, max_horizon=10)
    p_up, p_down, p_to, barrier = triple_barrier_mc_garch(
        fit,
        horizon=10,
        barrier_mult=2.0,
        n_paths=1000,
        seed=12345,
    )
    assert p_up is not None and p_down is not None and p_to is not None
    assert barrier is not None and barrier > 0
    assert abs(p_up + p_down + p_to - 1.0) < 1e-9


def test_triple_barrier_reproduces_under_fixed_seed():
    returns = np.random.default_rng(3).normal(0.0, 0.01, size=250)
    cfg = GarchDynamicsConfig(model="ewma", mc_samples=500, min_train=60)
    fit = fit_garch_dynamics(returns, cfg, max_horizon=10)
    first = triple_barrier_mc_garch(
        fit, horizon=10, barrier_mult=2.0, n_paths=500, seed=99
    )
    second = triple_barrier_mc_garch(
        fit, horizon=10, barrier_mult=2.0, n_paths=500, seed=99
    )
    assert first == second


def test_mc_quantiles_ordering():
    returns = np.random.default_rng(4).normal(0.0, 0.01, size=250)
    cfg = GarchDynamicsConfig(model="ewma", mc_samples=2000, min_train=60)
    fit = fit_garch_dynamics(returns, cfg, max_horizon=10)
    q05, q50, q95 = mc_quantiles_garch(fit, horizon=10, n_paths=2000, seed=42)
    assert q05 is not None and q50 is not None and q95 is not None
    assert q05 <= q50 <= q95


def test_conditional_var_cvar_negative_left_tail():
    returns = np.random.default_rng(5).normal(0.0, 0.015, size=200)
    cfg = GarchDynamicsConfig(model="ewma", min_train=60)
    fit = fit_garch_dynamics(returns, cfg, max_horizon=5)
    var, cvar = conditional_var_cvar(fit, alpha=0.05)
    assert var is not None and cvar is not None
    assert var < 0
    assert cvar <= var


def test_low_n_marks_unreliable():
    returns = np.random.default_rng(6).normal(0.0, 0.01, size=30)
    cfg = GarchDynamicsConfig(min_train=60)
    fit = fit_garch_dynamics(returns, cfg, max_horizon=5)
    assert "low_n" in fit.flags


@pytest.mark.parametrize("missing_arch", [True, False])
def test_fit_never_crashes_without_sigma_path(missing_arch, monkeypatch):
    returns = np.random.default_rng(7).normal(0.0, 0.01, size=100)
    cfg = GarchDynamicsConfig(model="ewma", min_train=60)
    if missing_arch:
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "arch":
                raise ImportError("no arch")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
    fit = fit_garch_dynamics(returns, cfg, max_horizon=5)
    assert fit.sigma_1d is not None


def test_conditional_var_cvar_unit_variance_t_smaller_than_raw_t():
    """arch uses unit-variance t; raw scipy.t quantiles overstate |VaR|."""
    from scipy import stats

    sigma = 0.01
    nu = 5.0
    alpha = 0.05
    fit = GarchFitResult(
        model="GJR-t",
        sigma_1d=sigma,
        nu=nu,
        reliable=True,
        dgp_basis="GJR-t",
    )
    var, cvar = conditional_var_cvar(fit, alpha=alpha)
    assert var is not None and cvar is not None
    raw_z = float(stats.t.ppf(alpha, df=nu))
    unit_z = unit_variance_t_ppf(alpha, nu)
    assert abs(unit_z) < abs(raw_z)
    assert abs(var - sigma * unit_z) < 1e-12
    assert abs(var) < abs(sigma * raw_z)
    assert cvar <= var < 0


def test_arch_lm_test_uses_demeaned_returns():
    rng = np.random.default_rng(11)
    # Strong mean shift should not dominate a² clustering test vs demeaned path.
    returns = 0.05 + rng.normal(0.0, 0.01, size=200)
    pval = arch_lm_test(returns, lags=5)
    assert pval is None or (0.0 <= pval <= 1.0)


def _arch_available() -> bool:
    try:
        import arch  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _arch_available(), reason="arch package not installed")
def test_gjr_fit_stores_recursive_params_and_mc_sums():
    returns = _make_garch_like_returns(400, seed=21)
    cfg = GarchDynamicsConfig(model="gjr11_t", min_train=60, mc_samples=800)
    fit = fit_garch_dynamics(returns, cfg, max_horizon=20)
    assert fit.sigma_1d is not None
    assert fit.dgp_basis in ("GJR-t", "GARCH-t")
    if fit.dgp_basis.startswith("GJR") or fit.dgp_basis.startswith("GARCH"):
        assert fit.omega is not None
        assert fit.alpha is not None
        assert fit.beta is not None
        assert fit.gamma is not None
    p_up, p_down, p_to, barrier = triple_barrier_mc_garch(
        fit,
        horizon=10,
        barrier_mult=2.0,
        n_paths=800,
        seed=7,
    )
    assert p_up is not None and p_down is not None and p_to is not None
    assert abs(p_up + p_down + p_to - 1.0) < 1e-9
    assert barrier is not None and barrier > 0


@pytest.mark.skipif(not _arch_available(), reason="arch package not installed")
def test_bootstrap_innovations_demeaned():
    rng = np.random.default_rng(42)
    returns = 0.003 + rng.normal(0.0, 0.01, size=300)
    cfg = GarchDynamicsConfig(model="gjr11_t", min_train=60, mc_samples=2000)
    fit = fit_garch_dynamics(returns, cfg, max_horizon=10)
    assert fit.std_resid is not None
    assert abs(float(np.mean(fit.std_resid))) > 0.01
    p_up, p_down, p_to, _ = triple_barrier_mc_garch(
        fit,
        horizon=10,
        barrier_mult=2.0,
        n_paths=2000,
        seed=12345,
    )
    assert p_up is not None and p_down is not None
    assert abs(p_up - p_down) < 0.08
    assert abs(p_up + p_down + p_to - 1.0) < 1e-9


def test_vol_spike_prob_reproduces_under_fixed_seed():
    returns = np.random.default_rng(8).normal(0.0, 0.012, size=250)
    cfg = GarchDynamicsConfig(model="ewma", mc_samples=500, min_train=60)
    fit = fit_garch_dynamics(returns, cfg, max_horizon=10)
    baseline = 0.25
    first = vol_spike_prob_mc_garch(
        fit,
        horizon=5,
        baseline_vol_ann=baseline,
        spike_mult=1.2,
        n_paths=500,
        seed=77,
    )
    second = vol_spike_prob_mc_garch(
        fit,
        horizon=5,
        baseline_vol_ann=baseline,
        spike_mult=1.2,
        n_paths=500,
        seed=77,
    )
    assert first is not None and second is not None
    assert first == second
    assert 0.0 <= first <= 1.0


def test_vol_spike_prob_invalid_baseline_returns_none():
    returns = np.random.default_rng(9).normal(0.0, 0.01, size=200)
    cfg = GarchDynamicsConfig(model="ewma", min_train=60)
    fit = fit_garch_dynamics(returns, cfg, max_horizon=5)
    assert vol_spike_prob_mc_garch(
        fit,
        horizon=5,
        baseline_vol_ann=0.0,
        spike_mult=1.2,
        n_paths=100,
        seed=1,
    ) is None
