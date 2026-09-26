"""2-state HMM short-horizon mixture predictive densities (ftsnotes §34).

Lightweight regime model: low/high volatility emissions (Student-t or normal).
Multi-step predictive density via regime-probability evolution + mixture MC
(§34.5.2). Independent of the GARCH board CSV schema.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from decision_pack.src.fat_tail_risk import slice_log_returns
from decision_pack.src.price_jump_flags import drop_split_log_returns

ANNUALIZE = math.sqrt(252.0)


@dataclass(frozen=True)
class HmmShortHorizonConfig:
    horizons: tuple[int, ...] = (5, 10, 20)
    n_states: int = 2
    train_window: int = 252
    min_samples: int = 60
    emission: str = "student_t"  # student_t | normal
    mc_samples: int = 2000
    barrier_mult: float = 2.0
    mc_seed: int = 12345
    drop_split_returns: bool = True
    split_threshold: float = 0.25
    max_em_iter: int = 80
    em_tol: float = 1e-6
    nu_fixed: float = 5.0  # fixed df for t-emission (stable EM)
    # If sigma_high / sigma_low < this, regimes are collapsed (p_high not interpretable).
    min_sigma_ratio: float = 1.15
    # Forward-filter only the last N returns for p_high/p_low (avoids sticky-state lock-in).
    regime_filter_lookback: int = 40
    # MC / triple-barrier: zero-mean paths (align with GARCH); mu_mix remains display-only.
    mc_zero_mean: bool = True
    mc_recent_vol_scale: bool = True
    mc_vol_short_window: int = 5
    mc_vol_long_window: int = 20
    mc_vol_scale_floor: float = 0.5


@dataclass(frozen=True)
class McSimulationOptions:
    zero_mean: bool = False
    sigma_scale: float = 1.0


@dataclass(frozen=True)
class HmmFitResult:
    n_states: int
    trans: np.ndarray  # (K, K)
    mu: np.ndarray  # (K,)
    sigma: np.ndarray  # (K,)
    nu: np.ndarray | None
    pi_filt: np.ndarray  # (K,) filtered state probs at T
    emission: str
    reliable: bool
    n_samples: int
    flags: tuple[str, ...] = ()
    loglik: float | None = None
    regime_separated: bool = True
    sigma_ratio: float | None = None
    # Mean training gamma mass per state (after final EM pass).
    state_posterior_mass: np.ndarray | None = None


@dataclass(frozen=True)
class ForwardFilterStep:
    """One causal forward-filter update under frozen HMM parameters."""

    alpha: np.ndarray  # (K,) filtered state probs at T
    xi: np.ndarray  # (K, K) joint filtered posterior P(S_{T-1}=i, S_T=j | y_1:T)
    log_evidence: float


@dataclass(frozen=True)
class HmmShortHorizonMetrics:
    horizon: int
    p_high: float | None
    p_low: float | None
    regime_now: str
    sigma_mix_1d: float | None
    sigma_mix_h: float | None
    mu_mix_1d: float | None
    p_up: float | None
    p_down: float | None
    p_timeout: float | None
    barrier: float | None
    q05: float | None
    q50: float | None
    q95: float | None
    reliable: bool
    basis: str
    n_samples: int = 0
    emission: str = "student_t"
    regime_separated: bool = True
    sigma_ratio: float | None = None


def regimes_are_separated(
    sigma: np.ndarray,
    *,
    min_sigma_ratio: float = 1.15,
) -> tuple[bool, float | None]:
    """Return (separated, sigma_high/sigma_low). Collapsed if ratio too close to 1."""
    sig = np.asarray(sigma, dtype=float).ravel()
    if sig.size < 2:
        return False, None
    low = float(np.min(sig))
    high = float(np.max(sig))
    if low <= 0 or not math.isfinite(low) or not math.isfinite(high):
        return False, None
    ratio = high / low
    return bool(ratio >= float(min_sigma_ratio)), float(ratio)


def hmm_short_horizon_config_from_raw(raw: dict[str, Any] | None) -> HmmShortHorizonConfig:
    if not raw or not isinstance(raw, dict):
        return HmmShortHorizonConfig()
    kwargs: dict[str, Any] = {}
    for key in HmmShortHorizonConfig.__dataclass_fields__:
        if key in raw and raw[key] is not None:
            val = raw[key]
            if key == "horizons":
                val = tuple(int(h) for h in val)
            kwargs[key] = val
    return HmmShortHorizonConfig(**kwargs)


def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    out = m_safe + np.log(np.sum(np.exp(a - m_safe), axis=axis, keepdims=True))
    if axis is None:
        return float(np.squeeze(out))
    return np.squeeze(out, axis=axis)


def _emission_logpdf(
    x: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    emission: str,
    nu: np.ndarray | None,
) -> np.ndarray:
    """Return (T, K) log emission densities."""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    mu = np.asarray(mu, dtype=float).reshape(1, -1)
    sigma = np.maximum(np.asarray(sigma, dtype=float).reshape(1, -1), 1e-8)
    if emission == "student_t" and nu is not None:
        nu_arr = np.asarray(nu, dtype=float).reshape(1, -1)
        z = (x - mu) / sigma
        # unit-scale t: dens = t.pdf(z, nu) / sigma
        return stats.t.logpdf(z, df=nu_arr) - np.log(sigma)
    return stats.norm.logpdf(x, loc=mu, scale=sigma)


def _forward_backward(
    log_emit: np.ndarray,
    log_trans: np.ndarray,
    log_start: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return gamma (T,K), xi (T-1,K,K), loglik."""
    t_len, k = log_emit.shape
    log_alpha = np.full((t_len, k), -np.inf)
    log_alpha[0] = log_start + log_emit[0]
    for t in range(1, t_len):
        for j in range(k):
            log_alpha[t, j] = log_emit[t, j] + _logsumexp(log_alpha[t - 1] + log_trans[:, j])

    loglik = float(_logsumexp(log_alpha[-1]))
    log_beta = np.zeros((t_len, k))
    for t in range(t_len - 2, -1, -1):
        for i in range(k):
            log_beta[t, i] = _logsumexp(log_trans[i, :] + log_emit[t + 1] + log_beta[t + 1])

    log_gamma = log_alpha + log_beta
    log_gamma = log_gamma - _logsumexp(log_gamma, axis=1)[:, None]
    gamma = np.exp(log_gamma)

    xi = np.zeros((max(t_len - 1, 0), k, k))
    for t in range(t_len - 1):
        log_xi = (
            log_alpha[t][:, None]
            + log_trans
            + log_emit[t + 1][None, :]
            + log_beta[t + 1][None, :]
        )
        log_xi = log_xi - _logsumexp(log_xi)
        xi[t] = np.exp(log_xi)
    return gamma, xi, loglik


def _init_params(
    returns: np.ndarray,
    k: int,
    emission: str,
    nu_fixed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    abs_r = np.abs(returns)
    med = float(np.median(abs_r)) if abs_r.size else 0.01
    mu = np.zeros(k, dtype=float)
    if k == 2:
        sigma = np.array([max(med * 0.7, 1e-4), max(med * 2.0, 2e-4)], dtype=float)
    elif k >= 3 and returns.size >= k * 5:
        sigma = np.zeros(k, dtype=float)
        edges = np.quantile(abs_r, np.linspace(0.0, 1.0, k + 1))
        for j in range(k):
            lo, hi = edges[j], edges[j + 1]
            mask = (abs_r >= lo) & (abs_r <= hi if j == k - 1 else abs_r < hi)
            if not np.any(mask):
                mask = abs_r <= hi
            chunk = returns[mask]
            if chunk.size:
                mu[j] = float(np.mean(chunk))
                sigma[j] = max(float(np.std(chunk, ddof=1)), 1e-4)
            else:
                sigma[j] = max(float(np.quantile(abs_r, (j + 1) / (k + 1))), 1e-4)
    else:
        qs = np.linspace(0.25, 0.75, k)
        sigma = np.maximum(np.quantile(abs_r, qs), 1e-4)
    trans = np.full((k, k), 0.1 / max(k - 1, 1))
    np.fill_diagonal(trans, 0.9)
    nu = np.full(k, nu_fixed, dtype=float) if emission == "student_t" else None
    start = np.full(k, 1.0 / k)
    return start, trans, mu, sigma, nu


def min_pairwise_sigma_ratio(sigma: np.ndarray) -> float | None:
    """Minimum σ_i/σ_j over all i<j (K-state regime separation)."""
    sig = np.asarray(sigma, dtype=float).ravel()
    if sig.size < 2:
        return None
    ratios: list[float] = []
    for i in range(sig.size):
        for j in range(i + 1, sig.size):
            low = float(min(sig[i], sig[j]))
            high = float(max(sig[i], sig[j]))
            if low <= 0 or not math.isfinite(low) or not math.isfinite(high):
                continue
            ratios.append(high / low)
    return min(ratios) if ratios else None


def fit_hmm_kstate(
    returns: np.ndarray,
    cfg: HmmShortHorizonConfig | None = None,
) -> HmmFitResult:
    cfg = cfg or HmmShortHorizonConfig()
    clean = returns[np.isfinite(returns)].astype(float)
    flags: list[str] = []
    if clean.size < cfg.min_samples:
        return HmmFitResult(
            n_states=cfg.n_states,
            trans=np.eye(cfg.n_states),
            mu=np.zeros(cfg.n_states),
            sigma=np.ones(cfg.n_states) * 0.01,
            nu=None,
            pi_filt=np.full(cfg.n_states, 1.0 / cfg.n_states),
            emission=cfg.emission,
            reliable=False,
            n_samples=int(clean.size),
            flags=("low_n",),
        )

    k = int(cfg.n_states)
    start, trans, mu, sigma, nu = _init_params(clean, k, cfg.emission, cfg.nu_fixed)
    log_start = np.log(np.maximum(start, 1e-12))
    prev_ll = -np.inf

    for _ in range(cfg.max_em_iter):
        log_trans = np.log(np.maximum(trans, 1e-12))
        log_emit = _emission_logpdf(clean, mu, sigma, emission=cfg.emission, nu=nu)
        gamma, xi, ll = _forward_backward(log_emit, log_trans, log_start)
        # M-step
        start = gamma[0] / max(float(gamma[0].sum()), 1e-12)
        log_start = np.log(np.maximum(start, 1e-12))
        xi_sum = xi.sum(axis=0)
        row_sum = xi_sum.sum(axis=1, keepdims=True)
        trans = xi_sum / np.maximum(row_sum, 1e-12)
        # sticky floor
        trans = np.maximum(trans, 1e-3)
        trans = trans / trans.sum(axis=1, keepdims=True)

        for j in range(k):
            w = gamma[:, j]
            w_sum = float(np.sum(w))
            if w_sum < 1e-8:
                continue
            mu[j] = float(np.sum(w * clean) / w_sum)
            var = float(np.sum(w * (clean - mu[j]) ** 2) / w_sum)
            floor = 1e-5
            if k >= 3:
                floor = max(float(np.median(np.abs(clean))) * 0.15, 1e-4)
            sigma[j] = max(math.sqrt(var), floor)

        # ensure state ordering: sigma[0] <= sigma[1] (low/high)
        order = np.argsort(sigma)
        mu = mu[order]
        sigma = sigma[order]
        trans = trans[np.ix_(order, order)]
        start = start[order]
        log_start = np.log(np.maximum(start, 1e-12))
        if nu is not None:
            nu = nu[order]

        if abs(ll - prev_ll) < cfg.em_tol:
            prev_ll = ll
            break
        prev_ll = ll

    # final filter probs
    log_trans = np.log(np.maximum(trans, 1e-12))
    log_emit = _emission_logpdf(clean, mu, sigma, emission=cfg.emission, nu=nu)
    gamma, _, ll = _forward_backward(log_emit, log_trans, log_start)
    pi_filt = gamma[-1] / max(float(gamma[-1].sum()), 1e-12)
    state_mass = gamma.mean(axis=0).astype(float)
    state_mass = state_mass / max(float(state_mass.sum()), 1e-12)

    if not np.all(np.isfinite(pi_filt)) or not np.all(np.isfinite(sigma)):
        flags.append("em_failed")
        return HmmFitResult(
            n_states=k,
            trans=trans,
            mu=mu,
            sigma=sigma,
            nu=nu,
            pi_filt=np.full(k, 1.0 / k),
            emission=cfg.emission,
            reliable=False,
            n_samples=int(clean.size),
            flags=tuple(flags),
            loglik=None,
            regime_separated=False,
            sigma_ratio=None,
            state_posterior_mass=None,
        )

    if k >= 3:
        sigma_ratio = min_pairwise_sigma_ratio(sigma)
        separated = bool(
            sigma_ratio is not None and sigma_ratio >= float(cfg.min_sigma_ratio)
        )
    else:
        separated, sigma_ratio = regimes_are_separated(
            sigma, min_sigma_ratio=cfg.min_sigma_ratio
        )
    if not separated:
        flags.append("regime_collapsed")

    return HmmFitResult(
        n_states=k,
        trans=trans,
        mu=mu,
        sigma=sigma,
        nu=nu,
        pi_filt=pi_filt.astype(float),
        emission=cfg.emission,
        reliable=True,
        n_samples=int(clean.size),
        flags=tuple(flags),
        loglik=float(prev_ll),
        regime_separated=separated,
        sigma_ratio=sigma_ratio,
        state_posterior_mass=state_mass,
    )


def fit_hmm_2state(
    returns: np.ndarray,
    cfg: HmmShortHorizonConfig | None = None,
) -> HmmFitResult:
    cfg = cfg or HmmShortHorizonConfig()
    if cfg.n_states != 2:
        cfg = replace(cfg, n_states=2)
    return fit_hmm_kstate(returns, cfg)


def forward_filter_step(
    fit: HmmFitResult,
    alpha_prev: np.ndarray,
    observation: float,
) -> ForwardFilterStep:
    """One-step causal filter under frozen parameters.

    Returns normalized joint posterior ``xi`` and marginal ``alpha`` at T.
    Raises ``ValueError`` for invalid shapes. If evidence is non-finite or
    zero, raises ``ValueError`` so callers can mark the update unreliable.
    """
    if not fit.reliable:
        raise ValueError("forward_filter_step requires a reliable HmmFitResult")
    k = int(fit.n_states)
    alpha_prev = np.asarray(alpha_prev, dtype=float).ravel()
    if alpha_prev.size != k:
        raise ValueError(f"alpha_prev size {alpha_prev.size} != n_states {k}")
    if not np.all(np.isfinite(alpha_prev)):
        raise ValueError("alpha_prev contains non-finite values")
    obs = float(observation)
    if not math.isfinite(obs):
        raise ValueError("observation must be finite")

    alpha_prev = np.maximum(alpha_prev, 0.0)
    alpha_prev = alpha_prev / max(float(alpha_prev.sum()), 1e-12)

    log_emit = _emission_logpdf(
        np.asarray([obs], dtype=float),
        fit.mu,
        fit.sigma,
        emission=fit.emission,
        nu=fit.nu,
    )[0]
    log_trans = np.log(np.maximum(fit.trans, 1e-12))
    log_alpha_prev = np.log(np.maximum(alpha_prev, 1e-12))
    log_xi = log_alpha_prev[:, None] + log_trans + log_emit[None, :]
    log_evidence = float(_logsumexp(log_xi))
    if not math.isfinite(log_evidence):
        raise ValueError("non-finite filter evidence")
    xi = np.exp(log_xi - log_evidence)
    if not np.all(np.isfinite(xi)):
        raise ValueError("non-finite xi after normalization")
    xi_sum = float(xi.sum())
    if xi_sum <= 0.0:
        raise ValueError("zero xi mass after normalization")
    xi = xi / xi_sum
    alpha = xi.sum(axis=0)
    alpha = alpha / max(float(alpha.sum()), 1e-12)
    return ForwardFilterStep(alpha=alpha.astype(float), xi=xi.astype(float), log_evidence=log_evidence)


def replay_forward_filter(
    fit: HmmFitResult,
    observations: np.ndarray,
    *,
    alpha_start: np.ndarray | None = None,
) -> np.ndarray:
    """Replay a frozen HMM over ``observations``; return final alpha."""
    clean = np.asarray(observations, dtype=float).ravel()
    clean = clean[np.isfinite(clean)]
    k = int(fit.n_states)
    if alpha_start is None:
        alpha = np.full(k, 1.0 / k, dtype=float)
    else:
        alpha = np.asarray(alpha_start, dtype=float).ravel()
        if alpha.size != k:
            raise ValueError(f"alpha_start size {alpha.size} != n_states {k}")
        alpha = np.maximum(alpha, 0.0)
        alpha = alpha / max(float(alpha.sum()), 1e-12)
    if clean.size == 0:
        return alpha
    for obs in clean:
        step = forward_filter_step(fit, alpha, float(obs))
        alpha = step.alpha
    return alpha


def filtered_state_probs(
    fit: HmmFitResult,
    returns: np.ndarray,
    *,
    lookback: int,
) -> np.ndarray:
    """Causal forward-filtered state probs at the end of `returns[-lookback:]`.

    Uses a uniform start prior so long-run sticky transitions cannot trap the
    current regime label in a distant historical state.
    """
    clean = returns[np.isfinite(returns)].astype(float)
    if clean.size == 0 or not fit.reliable or fit.pi_filt.size < 2:
        return fit.pi_filt.astype(float)
    n = max(1, min(int(lookback), clean.size))
    chunk = clean[-n:]
    try:
        return replay_forward_filter(fit, chunk)
    except ValueError:
        return fit.pi_filt.astype(float)


def _evolve_pi(pi: np.ndarray, trans: np.ndarray, steps: int) -> np.ndarray:
    p = pi.astype(float).copy()
    for _ in range(max(0, steps)):
        p = p @ trans
    return p


def _recent_vol_ratio(
    returns: np.ndarray,
    *,
    short: int,
    long: int,
) -> float | None:
    """std(short) / std(long) on trailing log returns (aligned with GARCH vol_ratio_5_20)."""
    clean = returns[np.isfinite(returns)].astype(float)
    if clean.size < long or long < 2 or short < 2:
        return None
    tail_short = clean[-short:]
    tail_long = clean[-long:]
    std_short = float(np.std(tail_short, ddof=1))
    std_long = float(np.std(tail_long, ddof=1))
    if std_long <= 0 or not math.isfinite(std_short) or not math.isfinite(std_long):
        return None
    return std_short / std_long


def _mc_sigma_scale(returns: np.ndarray, cfg: HmmShortHorizonConfig) -> float:
    if not cfg.mc_recent_vol_scale:
        return 1.0
    ratio = _recent_vol_ratio(
        returns,
        short=cfg.mc_vol_short_window,
        long=cfg.mc_vol_long_window,
    )
    if ratio is None or ratio >= 1.0:
        return 1.0
    return max(float(cfg.mc_vol_scale_floor), float(ratio))


def _sample_emission(
    fit: HmmFitResult,
    state: int,
    gen: np.random.Generator,
    n: int,
    *,
    mc_opts: McSimulationOptions | None = None,
) -> np.ndarray:
    opts = mc_opts or McSimulationOptions()
    mu = 0.0 if opts.zero_mean else float(fit.mu[state])
    sigma = float(fit.sigma[state]) * max(float(opts.sigma_scale), 1e-8)
    if fit.emission == "student_t" and fit.nu is not None:
        nu = float(fit.nu[state])
        # scale-t: sigma * standard_t (not unit-variance arch style; HMM emission scale)
        return mu + sigma * gen.standard_t(nu, size=n)
    return gen.normal(mu, sigma, size=n)


def simulate_hmm_paths(
    fit: HmmFitResult,
    horizon: int,
    n_paths: int,
    seed: int,
    *,
    mc_opts: McSimulationOptions | None = None,
) -> np.ndarray:
    """Simulate step returns shape (n_paths, horizon) under HMM DGP."""
    if horizon < 1 or n_paths < 1 or not fit.reliable:
        return np.zeros((0, 0), dtype=float)
    gen = np.random.default_rng(seed)
    k = fit.n_states
    # initial state from filtered pi
    states = gen.choice(k, size=n_paths, p=np.maximum(fit.pi_filt, 0) / max(fit.pi_filt.sum(), 1e-12))
    steps = np.zeros((n_paths, horizon), dtype=float)
    for t in range(horizon):
        for s in range(k):
            idx = np.where(states == s)[0]
            if idx.size == 0:
                continue
            steps[idx, t] = _sample_emission(
                fit, s, gen, idx.size, mc_opts=mc_opts
            )
        # transition
        new_states = np.empty_like(states)
        for s in range(k):
            idx = np.where(states == s)[0]
            if idx.size == 0:
                continue
            new_states[idx] = gen.choice(k, size=idx.size, p=fit.trans[s])
        states = new_states
    return steps


def triple_barrier_mc_hmm(
    fit: HmmFitResult,
    *,
    horizon: int,
    barrier_mult: float,
    n_paths: int,
    seed: int,
    mc_opts: McSimulationOptions | None = None,
) -> tuple[float | None, float | None, float | None, float | None]:
    if not fit.reliable or horizon < 1:
        return None, None, None, None
    opts = mc_opts or McSimulationOptions()
    # mixture 1-step sigma -> H-day scale ~ sigma_mix * sqrt(H) under i.i.d. within mix
    sigma_mix = float(np.dot(fit.pi_filt, fit.sigma))
    sigma_h = sigma_mix * max(float(opts.sigma_scale), 1e-8) * math.sqrt(horizon)
    barrier = float(barrier_mult * sigma_h)
    if barrier <= 0:
        return None, None, None, None
    steps = simulate_hmm_paths(fit, horizon, n_paths, seed, mc_opts=opts)
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
    tie = (first_up == first_down) & (first_up <= horizon - 1)
    up_wins = up_wins | tie
    down_wins = down_wins & ~tie
    up_hits = int(np.sum(up_wins))
    down_hits = int(np.sum(down_wins))
    timeouts = int(n_paths - up_hits - down_hits)
    total = float(n_paths)
    return up_hits / total, down_hits / total, timeouts / total, barrier


def mc_quantiles_hmm(
    fit: HmmFitResult,
    *,
    horizon: int,
    n_paths: int,
    seed: int,
    mc_opts: McSimulationOptions | None = None,
) -> tuple[float | None, float | None, float | None]:
    steps = simulate_hmm_paths(fit, horizon, n_paths, seed, mc_opts=mc_opts)
    if steps.size == 0:
        return None, None, None
    totals = steps.sum(axis=1)
    q05, q50, q95 = np.quantile(totals, [0.05, 0.50, 0.95])
    return float(q05), float(q50), float(q95)


def compute_hmm_short_horizon_metrics(
    close: pd.Series,
    end: pd.Timestamp,
    *,
    horizon: int,
    cfg: HmmShortHorizonConfig | None = None,
    fit: HmmFitResult | None = None,
) -> HmmShortHorizonMetrics:
    cfg = cfg or HmmShortHorizonConfig()
    if cfg.drop_split_returns:
        returns = drop_split_log_returns(
            close,
            end=end,
            lookback_days=cfg.train_window,
            split_threshold=cfg.split_threshold,
        )
    else:
        returns = slice_log_returns(close, end, lookback_days=cfg.train_window)

    n = int(returns.size)
    if n < 20:
        return HmmShortHorizonMetrics(
            horizon=horizon,
            p_high=None,
            p_low=None,
            regime_now="—",
            sigma_mix_1d=None,
            sigma_mix_h=None,
            mu_mix_1d=None,
            p_up=None,
            p_down=None,
            p_timeout=None,
            barrier=None,
            q05=None,
            q50=None,
            q95=None,
            reliable=False,
            basis="数据不足·仅展示",
            n_samples=n,
            emission=cfg.emission,
            regime_separated=False,
        )

    if fit is None:
        fit = fit_hmm_2state(returns, cfg)

    if not fit.reliable or fit.pi_filt.size < 2:
        return HmmShortHorizonMetrics(
            horizon=horizon,
            p_high=None,
            p_low=None,
            regime_now="—",
            sigma_mix_1d=None,
            sigma_mix_h=None,
            mu_mix_1d=None,
            p_up=None,
            p_down=None,
            p_timeout=None,
            barrier=None,
            q05=None,
            q50=None,
            q95=None,
            reliable=False,
            basis="HMM未收敛·仅展示",
            n_samples=fit.n_samples,
            emission=cfg.emission,
            regime_separated=False,
        )

    regime_pi = filtered_state_probs(
        fit, returns, lookback=cfg.regime_filter_lookback
    )
    p_low = float(regime_pi[0])
    p_high = float(regime_pi[-1])
    separated = bool(fit.regime_separated)
    sigma_ratio = fit.sigma_ratio
    if not separated:
        # Re-check in case fit was built without the flag (older callers).
        separated, sigma_ratio = regimes_are_separated(
            fit.sigma, min_sigma_ratio=cfg.min_sigma_ratio
        )

    if separated:
        regime_now = "高波动" if p_high >= p_low else "低波动"
        basis = f"HMM-2·{cfg.emission}·§34"
    else:
        # Do not label collapsed EM as 低/高波动 — p_high is not interpretable.
        regime_now = "未分离"
        ratio_txt = f"{sigma_ratio:.2f}" if sigma_ratio is not None else "—"
        basis = f"HMM-2·{cfg.emission}·§34·两态未分离(σH/σL={ratio_txt})"

    sigma_mix = float(np.dot(regime_pi, fit.sigma))
    mu_mix = float(np.dot(regime_pi, fit.mu))
    sigma_h = sigma_mix * math.sqrt(horizon)

    fit_for_mc = fit
    if not np.allclose(regime_pi, fit.pi_filt):
        fit_for_mc = replace(fit, pi_filt=regime_pi.astype(float))

    mc_opts = McSimulationOptions(
        zero_mean=cfg.mc_zero_mean,
        sigma_scale=_mc_sigma_scale(returns, cfg),
    )
    if cfg.mc_zero_mean:
        basis = f"{basis}·mc零均值"
    if mc_opts.sigma_scale < 0.999:
        basis = f"{basis}·低波缩放"

    p_up, p_down, p_timeout, barrier = triple_barrier_mc_hmm(
        fit_for_mc,
        horizon=horizon,
        barrier_mult=cfg.barrier_mult,
        n_paths=cfg.mc_samples,
        seed=cfg.mc_seed + horizon,
        mc_opts=mc_opts,
    )
    q05, q50, q95 = mc_quantiles_hmm(
        fit_for_mc,
        horizon=horizon,
        n_paths=cfg.mc_samples,
        seed=cfg.mc_seed + horizon + 1000,
        mc_opts=mc_opts,
    )

    return HmmShortHorizonMetrics(
        horizon=horizon,
        p_high=p_high,
        p_low=p_low,
        regime_now=regime_now,
        sigma_mix_1d=sigma_mix,
        sigma_mix_h=sigma_h,
        mu_mix_1d=mu_mix,
        p_up=p_up,
        p_down=p_down,
        p_timeout=p_timeout,
        barrier=barrier,
        q05=q05,
        q50=q50,
        q95=q95,
        reliable=True,
        basis=basis,
        n_samples=fit.n_samples,
        emission=cfg.emission,
        regime_separated=separated,
        sigma_ratio=sigma_ratio,
    )


HMM_SHORT_HORIZON_COLUMNS: tuple[str, ...] = (
    "code",
    "name",
    "持仓",
    "wt",
    "n",
    "H",
    "regime",
    "sep",
    "p_high",
    "p_low",
    "σ_mix_1d",
    "σ_mix_H",
    "μ_mix_1d",
    "触轨↑",
    "触轨↓",
    "到期",
    "q05_H",
    "q50_H",
    "q95_H",
    "障宽",
    "leg",
    "flags",
    "尾结构",
    "状态",
    "回撤长",
    "依据",
)


def format_hmm_row(
    *,
    code: str,
    name: str,
    weight: float,
    cluster_row: pd.Series,
    metrics: HmmShortHorizonMetrics,
    formatters: dict[str, Any],
) -> dict[str, str]:
    pct = formatters["pct"]
    prob = formatters["prob"]
    flags_str = formatters["flags_str"]
    tail_short = formatters["tail_short"](cluster_row.get("tail_regime"))
    state_short = formatters["state_short"](cluster_row.get("path_state"))
    sep = "ok" if metrics.regime_separated else "塌缩"
    # Keep numeric p_high/p_low for audit, but mark collapsed clearly in regime/sep/依据.
    return {
        "code": code,
        "name": name,
        "持仓": "持" if weight > 0 else "候选",
        "wt": f"{weight:.4f}",
        "n": str(metrics.n_samples),
        "H": str(metrics.horizon),
        "regime": metrics.regime_now,
        "sep": sep,
        "p_high": prob(metrics.p_high) if metrics.regime_separated else "—",
        "p_low": prob(metrics.p_low) if metrics.regime_separated else "—",
        "σ_mix_1d": pct(metrics.sigma_mix_1d),
        "σ_mix_H": pct(metrics.sigma_mix_h),
        "μ_mix_1d": pct(metrics.mu_mix_1d),
        "触轨↑": prob(metrics.p_up),
        "触轨↓": prob(metrics.p_down),
        "到期": prob(metrics.p_timeout),
        "q05_H": pct(metrics.q05),
        "q50_H": pct(metrics.q50),
        "q95_H": pct(metrics.q95),
        "障宽": pct(metrics.barrier),
        "leg": str(cluster_row.get("barbell_leg") or "—"),
        "flags": flags_str(cluster_row.get("flags")),
        "尾结构": tail_short,
        "状态": state_short,
        "回撤长": pct(cluster_row.get("dd_long"), digits=1),
        "依据": metrics.basis,
    }


__all__ = [
    "HMM_SHORT_HORIZON_COLUMNS",
    "ForwardFilterStep",
    "HmmFitResult",
    "HmmShortHorizonConfig",
    "HmmShortHorizonMetrics",
    "McSimulationOptions",
    "compute_hmm_short_horizon_metrics",
    "filtered_state_probs",
    "fit_hmm_2state",
    "fit_hmm_kstate",
    "forward_filter_step",
    "min_pairwise_sigma_ratio",
    "format_hmm_row",
    "hmm_short_horizon_config_from_raw",
    "mc_quantiles_hmm",
    "regimes_are_separated",
    "replay_forward_filter",
    "simulate_hmm_paths",
    "triple_barrier_mc_hmm",
]
