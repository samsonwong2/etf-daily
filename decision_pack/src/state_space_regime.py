"""Causal state-space regime labels for long-only ETF trading.

The live path uses filtering only:

    log_price -> local-linear-trend Kalman filter -> slope probability
              -> frozen two-state Student-t HMM volatility gate
              -> up / down / range

Full-sample smoothing and Viterbi decoding are intentionally absent.  A fitted
model is JSON serializable and must be frozen before an evaluation window.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from decision_pack.src.hmm_short_horizon import (
    HmmFitResult,
    HmmShortHorizonConfig,
    fit_hmm_kstate,
    forward_filter_step,
)


STATE_SPACE_METHODS = ("kalman_local_trend", "hmm_kalman_gate")


@dataclass(frozen=True)
class StateSpaceRegimeConfig:
    method: str = "hmm_kalman_gate"
    train_window: int = 756
    q_level_ratio: float = 0.02
    q_slope_ratio: float = 0.001
    direction_prob: float = 0.72
    high_vol_direction_prob: float = 0.80
    confirm_days: int = 3
    hmm_states: int = 2
    hmm_emission: str = "student_t"
    hmm_nu: float = 5.0
    hmm_min_samples: int = 80
    hmm_max_iter: int = 80
    hmm_min_sigma_ratio: float = 1.15
    fallback_method: str = "ma_stack_hyst"


@dataclass(frozen=True)
class KalmanFilterResult:
    level: np.ndarray
    slope: np.ndarray
    slope_var: np.ndarray
    innovation: np.ndarray
    innovation_var: np.ndarray


def _finite_positive(value: float, default: float) -> float:
    value = float(value)
    return value if math.isfinite(value) and value > 0 else float(default)


def fit_kalman_scale(log_price: np.ndarray) -> float:
    """Estimate observation scale from training data only."""
    y = np.asarray(log_price, dtype=float)
    dy = np.diff(y[np.isfinite(y)])
    if dy.size < 10:
        return 1e-4
    # Robust daily innovation variance; bounded away from the singular case.
    med = float(np.median(dy))
    mad = float(np.median(np.abs(dy - med)))
    sigma = max(1.4826 * mad, float(np.std(dy, ddof=1)) * 0.5, 1e-4)
    return float(sigma * sigma)


def kalman_local_linear_filter(
    log_price: np.ndarray,
    *,
    obs_var: float,
    q_level_ratio: float,
    q_slope_ratio: float,
) -> KalmanFilterResult:
    """Filter a local-linear-trend model without using future observations."""
    y = np.asarray(log_price, dtype=float)
    n = len(y)
    level = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    slope_var = np.full(n, np.nan)
    innovation = np.full(n, np.nan)
    innovation_var = np.full(n, np.nan)
    if n == 0:
        return KalmanFilterResult(level, slope, slope_var, innovation, innovation_var)

    r = _finite_positive(obs_var, 1e-4)
    q_level = max(r * float(q_level_ratio), 1e-12)
    q_slope = max(r * float(q_slope_ratio), 1e-12)
    f = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)
    h = np.array([[1.0, 0.0]], dtype=float)
    q = np.diag([q_level, q_slope])
    first = next((float(v) for v in y if np.isfinite(v)), 0.0)
    x = np.array([first, 0.0], dtype=float)
    p = np.diag([r * 100.0, r * 10.0])
    eye = np.eye(2)

    for i, obs in enumerate(y):
        x_pred = f @ x
        p_pred = f @ p @ f.T + q
        if np.isfinite(obs):
            v = float(obs - (h @ x_pred)[0])
            s = float((h @ p_pred @ h.T)[0, 0] + r)
            s = max(s, 1e-12)
            k = (p_pred @ h.T)[:, 0] / s
            x = x_pred + k * v
            # Joseph form keeps covariance positive under long replays.
            kh = np.outer(k, h[0])
            p = (eye - kh) @ p_pred @ (eye - kh).T + np.outer(k, k) * r
            innovation[i] = v
            innovation_var[i] = s
        else:
            x, p = x_pred, p_pred
        p = (p + p.T) * 0.5
        level[i] = x[0]
        slope[i] = x[1]
        slope_var[i] = max(float(p[1, 1]), 1e-12)
    return KalmanFilterResult(level, slope, slope_var, innovation, innovation_var)


def _serialize_hmm(fit: HmmFitResult) -> dict[str, Any]:
    return {
        "n_states": int(fit.n_states),
        "trans": fit.trans.tolist(),
        "mu": fit.mu.tolist(),
        "sigma": fit.sigma.tolist(),
        "nu": fit.nu.tolist() if fit.nu is not None else None,
        "pi_filt": fit.pi_filt.tolist(),
        "emission": fit.emission,
        "reliable": bool(fit.reliable),
        "n_samples": int(fit.n_samples),
        "flags": list(fit.flags),
        "loglik": fit.loglik,
        "regime_separated": bool(fit.regime_separated),
        "sigma_ratio": fit.sigma_ratio,
        "state_posterior_mass": (
            fit.state_posterior_mass.tolist()
            if fit.state_posterior_mass is not None
            else None
        ),
    }


def _deserialize_hmm(raw: dict[str, Any]) -> HmmFitResult:
    nu = raw.get("nu")
    mass = raw.get("state_posterior_mass")
    return HmmFitResult(
        n_states=int(raw["n_states"]),
        trans=np.asarray(raw["trans"], dtype=float),
        mu=np.asarray(raw["mu"], dtype=float),
        sigma=np.asarray(raw["sigma"], dtype=float),
        nu=np.asarray(nu, dtype=float) if nu is not None else None,
        pi_filt=np.asarray(raw["pi_filt"], dtype=float),
        emission=str(raw["emission"]),
        reliable=bool(raw["reliable"]),
        n_samples=int(raw["n_samples"]),
        flags=tuple(str(x) for x in raw.get("flags", [])),
        loglik=raw.get("loglik"),
        regime_separated=bool(raw.get("regime_separated", True)),
        sigma_ratio=raw.get("sigma_ratio"),
        state_posterior_mass=np.asarray(mass, dtype=float) if mass is not None else None,
    )


def fit_state_space_regime(
    px_train: pd.DataFrame,
    config: StateSpaceRegimeConfig | None = None,
) -> dict[str, Any]:
    """Fit all parameters on one training slice and return a frozen model."""
    cfg = config or StateSpaceRegimeConfig()
    if cfg.method not in STATE_SPACE_METHODS:
        raise ValueError(f"unknown state-space method: {cfg.method}")
    close = pd.to_numeric(px_train["$close"], errors="coerce").to_numpy(float)
    valid = np.isfinite(close) & (close > 0)
    if int(valid.sum()) < max(30, int(cfg.hmm_min_samples // 2)):
        return {
            "config": asdict(cfg),
            "reliable": False,
            "flags": ["low_n"],
            "fallback_method": cfg.fallback_method,
        }
    logp = np.where(valid, np.log(close), np.nan)
    fit_logp = logp[-max(int(cfg.train_window), 30) :]
    obs_var = fit_kalman_scale(fit_logp)
    model: dict[str, Any] = {
        "config": asdict(cfg),
        "obs_var": obs_var,
        "reliable": True,
        "flags": [],
        "fallback_method": cfg.fallback_method,
        "train_end": (
            str(pd.Timestamp(px_train["as_of"].iloc[-1]).date())
            if "as_of" in px_train.columns and len(px_train)
            else None
        ),
    }
    if cfg.method == "hmm_kalman_gate":
        fit_valid = np.isfinite(fit_logp)
        returns = np.diff(fit_logp[fit_valid])
        hmm_cfg = HmmShortHorizonConfig(
            n_states=int(cfg.hmm_states),
            min_samples=int(cfg.hmm_min_samples),
            emission=str(cfg.hmm_emission),
            nu_fixed=float(cfg.hmm_nu),
            max_em_iter=int(cfg.hmm_max_iter),
            min_sigma_ratio=float(cfg.hmm_min_sigma_ratio),
        )
        fit = fit_hmm_kstate(returns, hmm_cfg)
        model["hmm"] = _serialize_hmm(fit)
        if not fit.reliable or not fit.regime_separated:
            model["reliable"] = False
            model["flags"] = list(fit.flags or ("hmm_unreliable",))
    return model


def _causal_confirm(candidates: np.ndarray, confirm_days: int) -> np.ndarray:
    """Causally require repeated evidence before changing state."""
    candidates = np.asarray(candidates, dtype=object)
    if len(candidates) == 0 or confirm_days <= 1:
        return candidates.copy()
    out = np.full(len(candidates), "range", dtype=object)
    state = "range"
    pending = ""
    count = 0
    for i, candidate in enumerate(candidates):
        candidate = str(candidate)
        if candidate == state:
            pending, count = "", 0
        elif candidate == pending:
            count += 1
        else:
            pending, count = candidate, 1
        if pending and count >= confirm_days:
            state = pending
            pending, count = "", 0
        out[i] = state
    return out


def _hmm_high_prob(log_returns: np.ndarray, fit: HmmFitResult) -> np.ndarray:
    """Replay frozen HMM parameters and return high-vol posterior each day."""
    n = len(log_returns)
    out = np.full(n, np.nan)
    if not fit.reliable or fit.n_states < 2:
        return out
    high_idx = int(np.argmax(fit.sigma))
    alpha = np.full(fit.n_states, 1.0 / fit.n_states)
    for i, obs in enumerate(log_returns):
        if not np.isfinite(obs):
            out[i] = alpha[high_idx]
            continue
        try:
            alpha = forward_filter_step(fit, alpha, float(obs)).alpha
        except ValueError:
            return np.full(n, np.nan)
        out[i] = alpha[high_idx]
    return out


def apply_state_space_regime(
    px: pd.DataFrame,
    frozen_model: dict[str, Any],
) -> pd.DataFrame:
    """Apply a frozen model causally; returns labels and filter diagnostics."""
    cfg = StateSpaceRegimeConfig(**dict(frozen_model.get("config") or {}))
    if not frozen_model.get("reliable", False):
        raise ValueError(
            "unreliable state-space model: "
            + ",".join(str(x) for x in frozen_model.get("flags", []))
        )
    close = pd.to_numeric(px["$close"], errors="coerce").to_numpy(float)
    logp = np.where(np.isfinite(close) & (close > 0), np.log(close), np.nan)
    kf = kalman_local_linear_filter(
        logp,
        obs_var=float(frozen_model["obs_var"]),
        q_level_ratio=cfg.q_level_ratio,
        q_slope_ratio=cfg.q_slope_ratio,
    )
    z = kf.slope / np.sqrt(np.maximum(kf.slope_var, 1e-12))
    p_up = stats.norm.cdf(z)
    p_down = stats.norm.cdf(-z)
    high_prob = np.zeros(len(px), dtype=float)
    threshold = np.full(len(px), float(cfg.direction_prob), dtype=float)

    if cfg.method == "hmm_kalman_gate":
        fit = _deserialize_hmm(dict(frozen_model["hmm"]))
        ret = np.r_[np.nan, np.diff(logp)]
        high_prob = _hmm_high_prob(ret, fit)
        threshold = np.where(
            np.nan_to_num(high_prob, nan=0.0) >= 0.5,
            float(cfg.high_vol_direction_prob),
            float(cfg.direction_prob),
        )

    candidate = np.full(len(px), "range", dtype=object)
    candidate[p_up >= threshold] = "up"
    candidate[p_down >= threshold] = "down"
    labels = _causal_confirm(candidate, int(cfg.confirm_days))
    return pd.DataFrame(
        {
            "regime": labels,
            "state_candidate": candidate,
            "kalman_level": kf.level,
            "kalman_slope": kf.slope,
            "kalman_slope_z": z,
            "p_up_state": p_up,
            "p_down_state": p_down,
            "p_high_vol_state": high_prob,
            "state_threshold": threshold,
        },
        index=px.index,
    )


def state_space_labels_or_fallback(
    px: pd.DataFrame,
    frozen_model: dict[str, Any],
    *,
    fallback_labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return labels plus diagnostics, using an explicit reliable fallback."""
    try:
        frame = apply_state_space_regime(px, frozen_model)
        return frame["regime"].to_numpy(object), {
            "fallback": False,
            "flags": list(frozen_model.get("flags", [])),
            "diagnostics": frame,
        }
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return np.asarray(fallback_labels, dtype=object), {
            "fallback": True,
            "flags": list(frozen_model.get("flags", [])) + [str(exc)],
            "diagnostics": None,
        }


def fit_kalman_confirm_model(
    px_train: pd.DataFrame,
    *,
    q_level_ratio: float = 0.02,
    q_slope_ratio: float = 0.001,
    train_window: int = 756,
) -> dict[str, Any]:
    """Fit only the Kalman observation scale used by MA confirmation."""
    close = pd.to_numeric(px_train["$close"], errors="coerce").to_numpy(float)
    valid = np.isfinite(close) & (close > 0)
    if int(valid.sum()) < 30:
        return {
            "reliable": False,
            "flags": ["low_n"],
            "obs_var": 1e-4,
            "q_level_ratio": float(q_level_ratio),
            "q_slope_ratio": float(q_slope_ratio),
        }
    logp = np.where(valid, np.log(close), np.nan)
    fit_logp = logp[-max(int(train_window), 30) :]
    return {
        "reliable": True,
        "flags": [],
        "obs_var": fit_kalman_scale(fit_logp),
        "q_level_ratio": float(q_level_ratio),
        "q_slope_ratio": float(q_slope_ratio),
        "train_window": int(train_window),
    }


def kalman_filtered_probs(
    px: pd.DataFrame,
    frozen_model: dict[str, Any],
) -> pd.DataFrame:
    """Causal Kalman slope probabilities under a frozen observation scale."""
    close = pd.to_numeric(px["$close"], errors="coerce").to_numpy(float)
    logp = np.where(np.isfinite(close) & (close > 0), np.log(close), np.nan)
    kf = kalman_local_linear_filter(
        logp,
        obs_var=float(frozen_model.get("obs_var", 1e-4)),
        q_level_ratio=float(frozen_model.get("q_level_ratio", 0.02)),
        q_slope_ratio=float(frozen_model.get("q_slope_ratio", 0.001)),
    )
    z = kf.slope / np.sqrt(np.maximum(kf.slope_var, 1e-12))
    return pd.DataFrame(
        {
            "kalman_level": kf.level,
            "kalman_slope": kf.slope,
            "kalman_slope_z": z,
            "p_up_state": stats.norm.cdf(z),
            "p_down_state": stats.norm.cdf(-z),
        },
        index=px.index,
    )


def apply_ma_kalman_confirm(
    ma_labels: np.ndarray,
    px: pd.DataFrame,
    frozen_model: dict[str, Any],
    *,
    confirm_prob: float = 0.60,
    require_ma_bull: bool = True,
    soft_exit: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Keep MA ``up`` only when filtered Kalman slope also confirms.

    - Entry: MA==up AND p_up >= confirm_prob (and optional MA20>MA60).
    - Soft exit: once in confirmed up, leave when MA leaves up OR (soft_exit and
      p_up < 0.5). Hard exit always follows MA leaving up.
    - Non-up MA labels are unchanged (down/range pass through).
    """
    base = np.asarray(ma_labels, dtype=object).copy()
    probs = kalman_filtered_probs(px, frozen_model)
    p_up = probs["p_up_state"].to_numpy(float)
    thr = float(confirm_prob)
    if require_ma_bull and "ma_bull" in px.columns:
        ma_bull = px["ma_bull"].fillna(False).to_numpy(bool)
    elif require_ma_bull and {"MA20", "MA60"} <= set(px.columns):
        ma_bull = (px["MA20"] > px["MA60"]).fillna(False).to_numpy(bool)
    else:
        ma_bull = np.ones(len(base), dtype=bool)

    out = base.copy()
    confirmed = False
    for i, lab in enumerate(base):
        lab = str(lab)
        pup = float(p_up[i]) if np.isfinite(p_up[i]) else 0.0
        if lab == "up":
            if not confirmed:
                if pup >= thr and bool(ma_bull[i]):
                    confirmed = True
                    out[i] = "up"
                else:
                    out[i] = "range"
            else:
                if soft_exit and pup < 0.5:
                    confirmed = False
                    out[i] = "range"
                else:
                    out[i] = "up"
        else:
            confirmed = False
            out[i] = lab
    probs = probs.copy()
    probs["regime"] = out
    probs["ma_regime"] = base
    probs["confirm_prob"] = thr
    return out, probs
