"""GARCH-family conditional volatility for short-horizon decision boards (ftsnotes §18–21).

Causal one-step / multi-step sigma forecasts, recursive GARCH-t path simulation,
conditional VaR/CVaR (unit-variance Student-t). Falls back EWMA -> RV20 -> MAD when
``arch`` is unavailable or MLE fails.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from decision_pack.src.fat_tail_risk import compute_mad

ANNUALIZE = math.sqrt(252.0)


@dataclass(frozen=True)
class GarchDynamicsConfig:
    train_window: int = 252
    model: str = "gjr11_t"  # gjr11_t | garch11_t | ewma
    ewma_lambda: float = 0.94
    min_train: int = 60
    mc_samples: int = 2000
    mc_seed: int = 12345
    barrier_mult: float = 1.0
    variance_floor: float = 2.5e-5
    variance_cap: float = 2e-2
    arch_test: bool = True
    arch_lags: int = 5


@dataclass
class GarchFitResult:
    model: str
    sigma_1d: float | None
    sigma_h: dict[int, float] = field(default_factory=dict)
    nu: float | None = None
    arch_pvalue: float | None = None
    reliable: bool = False
    flags: tuple[str, ...] = ()
    # simulation state
    std_resid: np.ndarray | None = None
    sigma_path: np.ndarray | None = None  # analytic per-step sigma (forecast)
    dgp_basis: str = ""
    # recursive GARCH params in *return* units (None => flat-path MC fallback)
    omega: float | None = None
    alpha: float | None = None
    gamma: float | None = None
    beta: float | None = None
    variance_floor: float = 2.5e-5
    variance_cap: float = 2e-2


def garch_dynamics_config_from_raw(raw: dict[str, Any] | None) -> GarchDynamicsConfig:
    if not raw or not isinstance(raw, dict):
        return GarchDynamicsConfig()
    garch_raw = raw.get("garch")
    merged: dict[str, Any] = dict(raw)
    if isinstance(garch_raw, dict):
        for key, value in garch_raw.items():
            if value is not None:
                merged[key] = value
        merged.pop("garch", None)
    kwargs: dict[str, Any] = {}
    for key in GarchDynamicsConfig.__dataclass_fields__:
        if key in merged and merged[key] is not None:
            kwargs[key] = merged[key]
    if "horizons" in kwargs:
        kwargs.pop("horizons")
    return GarchDynamicsConfig(**kwargs)


def _clip_sigma(sigma: float, cfg: GarchDynamicsConfig) -> float:
    var = float(np.clip(sigma * sigma, cfg.variance_floor, cfg.variance_cap))
    return math.sqrt(var)


def _clip_variance(var: float, floor: float, cap: float) -> float:
    return float(np.clip(var, floor, cap))


def _clean_returns(returns: np.ndarray) -> np.ndarray:
    return returns[np.isfinite(returns)].astype(float)


def arch_lm_test(returns: np.ndarray, *, lags: int = 5) -> float | None:
    """Ljung-Box p-value on squared demeaned returns (a² proxy, ftsnotes §18.4)."""
    clean = _clean_returns(returns)
    if clean.size < lags + 10:
        return None
    # Constant-mean residual a_t = r_t - mean(r); book tests a² not raw r².
    innovations = clean - float(np.mean(clean))
    squared = innovations**2
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        result = acorr_ljungbox(squared, lags=[lags], return_df=True)
        pval = float(result["lb_pvalue"].iloc[-1])
        return pval if math.isfinite(pval) else None
    except ImportError:
        n = squared.size
        acfs = []
        centered = squared - float(np.mean(squared))
        denom = float(np.sum(centered**2))
        if denom <= 0:
            return None
        for lag in range(1, lags + 1):
            num = float(np.sum(centered[lag:] * centered[:-lag]))
            acfs.append(num / denom)
        q_stat = n * (n + 2) * sum(
            (acf**2) / (n - k - 1) for k, acf in enumerate(acfs, start=1)
        )
        pval = float(1.0 - stats.chi2.cdf(q_stat, df=lags))
        return pval if math.isfinite(pval) else None


def _ewma_sigma(returns: np.ndarray, cfg: GarchDynamicsConfig) -> float:
    squared = np.square(returns.astype(float))
    variance = float(np.nanmean(squared[: min(len(squared), 20)]))
    if not math.isfinite(variance) or variance <= 0.0:
        variance = float(np.nanmean(squared))
    for value in squared:
        variance = cfg.ewma_lambda * variance + (1.0 - cfg.ewma_lambda) * float(value)
    return _clip_sigma(math.sqrt(variance), cfg)


def _rv_sigma(returns: np.ndarray, window: int = 20) -> float | None:
    clean = _clean_returns(returns)
    if clean.size < window:
        return None
    tail = clean[-window:]
    std = float(np.std(tail, ddof=1))
    if not math.isfinite(std) or std <= 0:
        return None
    return std


def _mad_sigma(returns: np.ndarray) -> float | None:
    mad, _ = compute_mad(returns)
    if mad is None or mad <= 0:
        return None
    return mad * 1.253314  # MAD -> std for normal


def _build_sigma_path(sigma_1d: float, horizon: int) -> np.ndarray:
    """Flat sigma path for non-GARCH fallbacks."""
    h = max(1, int(horizon))
    return np.full(h, float(sigma_1d), dtype=float)


def _param_by_keys(params: pd.Series, *keys: str) -> float | None:
    lower_map = {str(k).lower(): float(v) for k, v in params.items()}
    for key in keys:
        if key in lower_map and math.isfinite(lower_map[key]):
            return lower_map[key]
    for name, value in lower_map.items():
        for key in keys:
            if key in name and math.isfinite(value):
                return value
    return None


def _extract_garch_params(
    fit: Any,
    *,
    o: int,
    cfg: GarchDynamicsConfig,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Extract ω, α, γ, β in return units from an arch fit on percent returns."""
    params = fit.params
    omega_pct = _param_by_keys(params, "omega")
    alpha = _param_by_keys(params, "alpha[1]", "alpha1", "alpha")
    beta = _param_by_keys(params, "beta[1]", "beta1", "beta")
    gamma = 0.0
    if o > 0:
        gamma_val = _param_by_keys(params, "gamma[1]", "gamma1", "gamma")
        if gamma_val is None:
            return None, None, None, None
        gamma = float(gamma_val)
    if omega_pct is None or alpha is None or beta is None:
        return None, None, None, None
    # arch fit on r*100 => variance scale *10000 relative to raw returns
    omega = float(omega_pct) / 10000.0
    if omega <= 0 or alpha < 0 or beta < 0:
        return None, None, None, None
    return omega, float(alpha), float(gamma), float(beta)


def _has_recursive_params(fit: GarchFitResult) -> bool:
    return (
        fit.omega is not None
        and fit.alpha is not None
        and fit.beta is not None
        and fit.gamma is not None
        and fit.sigma_1d is not None
        and fit.sigma_1d > 0
    )


def _fit_arch_model(
    returns: np.ndarray,
    *,
    vol: str,
    o: int,
    dist: str,
    cfg: GarchDynamicsConfig,
    max_horizon: int,
) -> GarchFitResult | None:
    try:
        from arch import arch_model
    except ImportError:
        return None

    clean = _clean_returns(returns)
    if clean.size < cfg.min_train:
        return None

    scaled = pd.Series(clean * 100.0)
    try:
        model = arch_model(
            scaled,
            mean="Zero",
            vol=vol,
            p=1,
            o=o,
            q=1,
            dist=dist,
            rescale=False,
        )
        fit = model.fit(disp="off", show_warning=False)
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
        return None

    try:
        forecast = fit.forecast(horizon=max_horizon, reindex=False)
        var_path = forecast.variance.values[-1] / 10000.0
        var_path = np.clip(var_path, cfg.variance_floor, cfg.variance_cap)
        sigma_path = np.sqrt(var_path)
        sigma_1d = float(sigma_path[0])
    except (ValueError, IndexError, TypeError):
        return None

    std_resid = np.asarray(fit.std_resid, dtype=float)
    std_resid = std_resid[np.isfinite(std_resid)]
    if std_resid.size < 10:
        return None

    nu: float | None = None
    if dist == "t":
        for key in fit.params.index:
            if "nu" in str(key).lower():
                nu_val = float(fit.params[key])
                if math.isfinite(nu_val) and nu_val > 2:
                    nu = nu_val
                break

    omega, alpha, gamma, beta = _extract_garch_params(fit, o=o, cfg=cfg)
    model_name = "GJR-t" if o > 0 and dist == "t" else f"GARCH-{dist}"
    sigma_h = {
        h: float(np.sqrt(np.sum(sigma_path[:h] ** 2))) for h in range(1, max_horizon + 1)
    }

    return GarchFitResult(
        model=model_name,
        sigma_1d=_clip_sigma(sigma_1d, cfg),
        sigma_h=sigma_h,
        nu=nu,
        reliable=True,
        flags=(),
        std_resid=std_resid,
        sigma_path=sigma_path.astype(float),
        dgp_basis=model_name,
        omega=omega,
        alpha=alpha,
        gamma=gamma if gamma is not None else (0.0 if o == 0 else None),
        beta=beta,
        variance_floor=cfg.variance_floor,
        variance_cap=cfg.variance_cap,
    )


def _fallback_fit(
    returns: np.ndarray,
    cfg: GarchDynamicsConfig,
    *,
    max_horizon: int,
    tried: tuple[str, ...],
) -> GarchFitResult:
    flags = list(tried)
    clean = _clean_returns(returns)

    sigma = _ewma_sigma(clean, cfg) if clean.size >= cfg.min_train else None
    basis = "EWMA"
    if sigma is None:
        sigma = _rv_sigma(clean, window=20)
        basis = "RV20"
        flags.append("ewma_failed")
    if sigma is None:
        sigma = _mad_sigma(clean)
        basis = "MAD"
        flags.append("rv_failed")
    if sigma is None:
        return GarchFitResult(
            model="none",
            sigma_1d=None,
            reliable=False,
            flags=tuple(flags + ["no_sigma"]),
            dgp_basis="none",
            variance_floor=cfg.variance_floor,
            variance_cap=cfg.variance_cap,
        )

    sigma = _clip_sigma(float(sigma), cfg)
    sigma_h = {h: sigma * math.sqrt(h) for h in range(1, max_horizon + 1)}
    std_resid = clean / sigma if sigma > 0 else clean
    return GarchFitResult(
        model=basis,
        sigma_1d=sigma,
        sigma_h=sigma_h,
        reliable=basis in ("EWMA", "RV20"),
        flags=tuple(flags),
        std_resid=std_resid,
        sigma_path=_build_sigma_path(sigma, max_horizon),
        dgp_basis=basis,
        variance_floor=cfg.variance_floor,
        variance_cap=cfg.variance_cap,
    )


def fit_garch_dynamics(
    returns: np.ndarray,
    cfg: GarchDynamicsConfig | None = None,
    *,
    max_horizon: int = 20,
) -> GarchFitResult:
    """Fit conditional volatility with GJR/GARCH-t fallback chain."""
    cfg = cfg or GarchDynamicsConfig()
    clean = _clean_returns(returns)
    if clean.size < cfg.min_train:
        result = _fallback_fit(clean, cfg, max_horizon=max_horizon, tried=("low_n",))
        result.reliable = False
        return result

    if cfg.train_window > 0 and clean.size > cfg.train_window:
        clean = clean[-cfg.train_window :]

    # §18.4: Ljung-Box on a² under constant-mean residual (demeaned returns).
    arch_p = arch_lm_test(clean, lags=cfg.arch_lags) if cfg.arch_test else None
    tried: list[str] = []
    preferred = str(cfg.model).lower()
    candidates: list[tuple[str, int, str]] = []
    if preferred == "gjr11_t":
        candidates = [("GARCH", 1, "t"), ("GARCH", 0, "t")]
    elif preferred == "garch11_t":
        candidates = [("GARCH", 0, "t")]
    elif preferred == "ewma":
        candidates = []

    for vol, o, dist in candidates:
        tried.append(f"{vol.lower()}{o}_{dist}")
        result = _fit_arch_model(
            clean,
            vol=vol,
            o=o,
            dist=dist,
            cfg=cfg,
            max_horizon=max_horizon,
        )
        if result is not None:
            result.arch_pvalue = arch_p
            result.dgp_basis = result.model
            return result

    result = _fallback_fit(clean, cfg, max_horizon=max_horizon, tried=tuple(tried))
    result.arch_pvalue = arch_p
    return result


def forecast_sigma_h(fit: GarchFitResult, h: int) -> float | None:
    if fit.sigma_1d is None:
        return None
    if h in fit.sigma_h:
        return fit.sigma_h[h]
    if fit.sigma_path is not None and h <= fit.sigma_path.size:
        return float(np.sqrt(np.sum(fit.sigma_path[:h] ** 2)))
    return fit.sigma_1d * math.sqrt(h)


def _sample_innovations(
    fit: GarchFitResult,
    gen: np.random.Generator,
    n: int,
) -> np.ndarray:
    resid = fit.std_resid
    if resid is not None and resid.size >= 5:
        # Demean so Zero-mean GARCH MC paths do not inherit historical drift.
        centered = np.asarray(resid, dtype=float) - float(np.mean(resid))
        return gen.choice(centered, size=n, replace=True).astype(float)
    if fit.nu is not None and fit.nu > 2:
        # Unit-variance Student-t draws (arch convention).
        raw = gen.standard_t(fit.nu, size=n)
        scale = math.sqrt(fit.nu / (fit.nu - 2.0))
        return (raw / scale).astype(float)
    return gen.standard_normal(n).astype(float)


def _simulate_recursive_paths(
    fit: GarchFitResult,
    h: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """Recursive GJR/GARCH path simulation; shape (n_paths, h) step returns."""
    assert _has_recursive_params(fit)
    omega = float(fit.omega)
    alpha = float(fit.alpha)
    gamma = float(fit.gamma)
    beta = float(fit.beta)
    floor = fit.variance_floor
    cap = fit.variance_cap

    gen = np.random.default_rng(seed)
    z = _sample_innovations(fit, gen, n_paths * h).reshape(n_paths, h)
    steps = np.zeros((n_paths, h), dtype=float)
    sigma2 = np.full(n_paths, float(fit.sigma_1d) ** 2, dtype=float)
    sigma2 = np.clip(sigma2, floor, cap)

    for t in range(h):
        sigma = np.sqrt(sigma2)
        a = sigma * z[:, t]
        steps[:, t] = a
        neg = (a < 0.0).astype(float)
        sigma2 = omega + (alpha + gamma * neg) * (a**2) + beta * sigma2
        sigma2 = np.clip(sigma2, floor, cap)
    return steps


def _simulate_flat_paths(
    fit: GarchFitResult,
    h: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """Fixed analytic sigma_path × residual bootstrap; shape (n_paths, h)."""
    gen = np.random.default_rng(seed)
    if fit.sigma_path is not None and fit.sigma_path.size >= h:
        sigma_path = fit.sigma_path[:h]
    else:
        sigma_path = _build_sigma_path(float(fit.sigma_1d), h)
    z = _sample_innovations(fit, gen, n_paths * h).reshape(n_paths, h)
    return z * sigma_path.reshape(1, h)


def simulate_step_returns(
    fit: GarchFitResult,
    h: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """Simulate per-step log returns under fitted DGP; shape (n_paths, h)."""
    if fit.sigma_1d is None or h < 1 or n_paths < 1:
        return np.zeros((0, 0), dtype=float)
    if _has_recursive_params(fit):
        return _simulate_recursive_paths(fit, h, n_paths, seed)
    return _simulate_flat_paths(fit, h, n_paths, seed)


def simulate_h_day_log_returns(
    fit: GarchFitResult,
    h: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """Simulate H-day cumulative log returns under fitted DGP."""
    steps = simulate_step_returns(fit, h, n_paths, seed)
    if steps.size == 0:
        return np.array([], dtype=float)
    return steps.sum(axis=1)


def triple_barrier_mc_garch(
    fit: GarchFitResult,
    *,
    horizon: int,
    barrier_mult: float,
    n_paths: int,
    seed: int,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Monte-Carlo triple-barrier first-touch probabilities using GARCH DGP."""
    if fit.sigma_1d is None or horizon < 1 or n_paths < 1:
        return None, None, None, None

    sigma_h = forecast_sigma_h(fit, horizon)
    if sigma_h is None or sigma_h <= 0:
        return None, None, None, None
    barrier = float(barrier_mult * sigma_h)
    if barrier <= 0:
        return None, None, None, None

    steps = simulate_step_returns(fit, horizon, n_paths, seed)
    if steps.size == 0:
        return None, None, None, None

    cum = np.cumsum(steps, axis=1)
    up_hit = cum >= barrier
    down_hit = cum <= -barrier
    any_up = up_hit.any(axis=1)
    any_down = down_hit.any(axis=1)
    first_up = np.where(any_up, up_hit.argmax(axis=1), horizon + 1)
    first_down = np.where(any_down, down_hit.argmax(axis=1), horizon + 1)

    up_wins = (first_up < first_down) & (first_up <= horizon - 1)
    down_wins = (first_down < first_up) & (first_down <= horizon - 1)
    # simultaneous first touch: count neither as exclusive; treat as timeout-split
    # by assigning to the side that appears in argmax ties — argmax picks first True,
    # so equal index means both True on same step; split half via up preference on tie
    tie = (first_up == first_down) & (first_up <= horizon - 1)
    up_wins = up_wins | tie  # first True in row is up check order; keep up on exact tie
    down_wins = down_wins & ~tie

    up_hits = int(np.sum(up_wins))
    down_hits = int(np.sum(down_wins))
    timeouts = int(n_paths - up_hits - down_hits)
    total = float(n_paths)
    return up_hits / total, down_hits / total, timeouts / total, barrier


def vol_spike_prob_mc_garch(
    fit: GarchFitResult,
    *,
    horizon: int,
    baseline_vol_ann: float,
    spike_mult: float = 1.2,
    n_paths: int,
    seed: int,
) -> float | None:
    """Fraction of MC paths whose H-day realized vol exceeds baseline × spike_mult.

    Path realized vol uses the same annualization as the GARCH board (std of daily
    log returns × √252). ``baseline_vol_ann`` is typically ``vol_ann_5d`` at as_of.
    """
    if fit.sigma_1d is None or horizon < 2 or n_paths < 1:
        return None
    if baseline_vol_ann <= 0 or not math.isfinite(baseline_vol_ann):
        return None
    if spike_mult <= 0 or not math.isfinite(spike_mult):
        return None

    threshold = float(baseline_vol_ann) * float(spike_mult)
    steps = simulate_step_returns(fit, horizon, n_paths, seed)
    if steps.size == 0 or steps.shape[1] < 2:
        return None

    path_std = np.std(steps, axis=1, ddof=1)
    path_vol_ann = path_std * ANNUALIZE
    spikes = path_vol_ann > threshold
    return float(np.sum(spikes)) / float(n_paths)


def mc_quantiles_garch(
    fit: GarchFitResult,
    *,
    horizon: int,
    n_paths: int,
    seed: int,
) -> tuple[float | None, float | None, float | None]:
    """H-day cumulative log-return quantiles from GARCH DGP simulation."""
    totals = simulate_h_day_log_returns(fit, horizon, n_paths, seed)
    if totals.size == 0:
        return None, None, None
    q05, q50, q95 = np.quantile(totals, [0.05, 0.50, 0.95])
    return float(q05), float(q50), float(q95)


def unit_variance_t_ppf(alpha: float, nu: float) -> float:
    """Percentile of Student-t standardized to unit variance (arch convention)."""
    raw = float(stats.t.ppf(alpha, df=nu))
    scale = math.sqrt(nu / (nu - 2.0))
    return raw / scale


def unit_variance_t_pdf(z: float, nu: float) -> float:
    """Density of unit-variance Student-t at z."""
    scale = math.sqrt(nu / (nu - 2.0))
    # f_unit(z) = scale * f_raw(z * scale)
    return float(scale * stats.t.pdf(z * scale, df=nu))


def conditional_var_cvar(
    fit: GarchFitResult,
    *,
    alpha: float = 0.05,
) -> tuple[float | None, float | None]:
    """1-day conditional left-tail VaR/CVaR at level ``alpha`` (negative = loss).

    When ``fit.nu`` is set, innovations follow *unit-variance* Student-t (arch),
    not scipy's raw t (variance ν/(ν−2)).
    """
    if fit.sigma_1d is None or fit.sigma_1d <= 0:
        return None, None

    sigma = fit.sigma_1d
    nu = fit.nu
    if nu is not None and nu > 2:
        z = unit_variance_t_ppf(alpha, nu)
        var = float(sigma * z)
        # ES for unit-variance t: scale the raw-t ES by 1/s where s=sqrt(ν/(ν-2)).
        # Raw-t ES at q_raw = z*s: -((ν+q_raw²)/(ν-1)) * f_raw(q_raw) / α
        # Unit ES = raw_ES / s.
        scale = math.sqrt(nu / (nu - 2.0))
        q_raw = z * scale
        pdf_raw = float(stats.t.pdf(q_raw, df=nu))
        es_raw = -((nu + q_raw**2) / (nu - 1.0)) * pdf_raw / alpha
        cvar = float(sigma * es_raw / scale)
        return var, cvar

    z = float(stats.norm.ppf(alpha))
    var = float(sigma * z)
    cvar = float(-sigma * stats.norm.pdf(z) / alpha)
    return var, cvar


__all__ = [
    "GarchDynamicsConfig",
    "GarchFitResult",
    "arch_lm_test",
    "conditional_var_cvar",
    "fit_garch_dynamics",
    "forecast_sigma_h",
    "garch_dynamics_config_from_raw",
    "mc_quantiles_garch",
    "simulate_h_day_log_returns",
    "simulate_step_returns",
    "triple_barrier_mc_garch",
    "unit_variance_t_ppf",
    "vol_spike_prob_mc_garch",
]
