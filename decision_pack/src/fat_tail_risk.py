"""Fat-tail risk metrics for ETF decision support (Taleb framework).

Uses MAD, tail index alpha, kappa, shadow/plug-in mean, parametric EVT VaR/CVaR.
Does NOT use standard deviation, empirical VaR, kurtosis, or winsorization.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

BarbellLeg = Literal["safe_leg", "risk_leg", "avoid"]

ANNUALIZE = math.sqrt(252.0)

_REGIME_CONFIG_ALIASES: dict[str, str] = {
    "alpha_recent_days": "regime_alpha_recent_days",
    "alpha_watch_ratio": "regime_alpha_watch_ratio",
    "alpha_confirm_ratio": "regime_alpha_confirm_ratio",
    "alpha_confirm_streak": "regime_alpha_confirm_streak",
    "kappa_watch_min": "regime_kappa_watch_min",
}


@dataclass(frozen=True)
class FatTailConfig:
    lookback_days: int = 756
    warmup_buffer_days: int = 40
    min_samples: int = 120
    tail_quantile: float = 0.925
    min_exceed: int = 25
    var_confidence: float = 0.99
    kappa_n: int = 30
    kappa_resamples: int = 2000
    kappa_seed: int = 12345
    ewma_span: int = 60
    alpha_thin: float = 4.0
    alpha_fat: float = 2.5
    kappa_safe_max: float = 0.08
    kappa_fat_min: float = 0.25
    cvar_red: float = 0.06
    cvar_safe_max: float = 0.02
    shrink_min_n: int = 252
    # Portfolio generation / risk control (framework §8, §9, §10.1).
    loss_budget_k: float = 0.03  # max acceptable daily portfolio left-tail loss (|CVaR99|)
    kelly_fraction: float = 0.25  # fractional-Kelly multiplier for per-name sizing cap
    n_min: int = 12  # min names (broad diversification floor, §8.3 step 0)
    n_max: int = 40  # max names cap
    concavity_ratio_warn: float = 1.6  # loss(p/2)/loss(p) >= this -> fragile/concave (§9.3.4)
    crash_q: float = 0.05  # bottom-quantile used for empirical co-crash evidence (§10.1)
    max_name_weight: float = 0.20  # upper bound for any single-name weight in proposals
    # Proposal generation objective / exclusions (report-only; production untouched).
    barbell_exclude_codes: tuple[str, ...] = ()  # codes to drop from generated proposals
    proposal_objective: str = "budget_barbell"  # "return_max" | "budget_barbell"
    # §12 stop-loss: the ONLY hard stop is on the portfolio total (HWM drawdown +
    # daily CVaR99 budget). Per-name actions come from drawdown + regime/trend +
    # position budget (kelly_cap), NOT a per-name single-day price trigger.
    trailing_stop_drawdown: float = 0.08  # high-water-mark trailing hard stop on total NAV (§12.3)
    # Per-name single-day VaR depths used ONLY as monitoring references ("a single-day
    # drop beyond these is abnormal for this name") — NOT execution triggers, no σ.
    # Names without an estimable left tail fall back to the fixed reference drops.
    exit_var_confidences: tuple[float, float, float] = (0.95, 0.99, 0.997)
    exit_drop_fallback: tuple[float, float, float] = (0.06, 0.09, 0.12)
    # §7 regime / price-trend layer drives per-name risk actions (no per-name σ-stop).
    regime_alpha_recent_days: int = 252
    regime_alpha_watch_ratio: float = 0.75
    regime_alpha_confirm_ratio: float = 0.65
    regime_alpha_confirm_streak: int = 2
    regime_kappa_watch_min: float = 0.15
    regime_confirm_step_days: int = 21  # trading days between streak checkpoints
    # Path / drawdown state (replaces MA-based price trend; book-endorsed path & ruin,
    # §12.3 high-water-mark, §9 path dependence — not mean/MA forecasting). The longer
    # window judges "trend broken"; the near window only flags a transient pullback.
    drawdown_near_days: int = 20
    drawdown_long_days: int = 120
    drawdown_pullback_pct: float = 0.10  # |dd| >= this -> pullback
    drawdown_broken_pct: float = 0.20  # |dd| over the long window >= this -> trend broken


@dataclass(frozen=True)
class TailFit:
    side: str
    threshold: float | None
    xi: float | None
    beta: float | None
    alpha: float | None
    n_exceed: int
    n_total: int
    reliable: bool
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class FatTailMetrics:
    code: str
    name: str
    weight: float
    n_samples: int
    mad: float | None
    mad_ann: float | None
    alpha_left: float | None
    alpha_right: float | None
    kappa: float | None
    raw_mean: float | None
    ewma60_mean: float | None
    shadow_mean: float | None
    shrink_factor: float | None
    shrunk_mean: float | None
    var_left: float | None
    cvar_left: float | None
    barbell_leg: BarbellLeg
    # Left-tail GPD parameters (exposed so downstream can recompute parametric
    # loss at any probability for concavity / crash-resonance aggregation).
    xi_left: float | None = None
    beta_left: float | None = None
    threshold_left: float | None = None
    nu_left: float | None = None
    flags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fat_tail_config_from_raw(raw: dict[str, Any] | None) -> FatTailConfig:
    if not raw or not isinstance(raw, dict):
        return FatTailConfig()
    merged: dict[str, Any] = dict(raw)
    regime = merged.pop("regime", None)
    if isinstance(regime, dict):
        for key, value in regime.items():
            mapped = _REGIME_CONFIG_ALIASES.get(key, key)
            if mapped in FatTailConfig.__dataclass_fields__ and value is not None:
                merged[mapped] = value
    kwargs: dict[str, Any] = {}
    for key in FatTailConfig.__dataclass_fields__:
        if key in merged and merged[key] is not None:
            kwargs[key] = merged[key]
    if "barbell_exclude_codes" in kwargs:
        kwargs["barbell_exclude_codes"] = tuple(
            str(c) for c in kwargs["barbell_exclude_codes"]
        )
    return FatTailConfig(**kwargs)


def slice_log_returns(
    close: pd.Series,
    end: pd.Timestamp,
    *,
    lookback_days: int,
) -> np.ndarray:
    available = close.loc[close.index <= end].dropna()
    if len(available) < 2:
        return np.array([], dtype=float)
    if lookback_days > 0 and len(available) > lookback_days + 1:
        available = available.tail(lookback_days + 1)
    prices = available.astype(float).values
    prices = prices[prices > 0]
    if len(prices) < 2:
        return np.array([], dtype=float)
    return np.diff(np.log(prices))


def compute_mad(returns: np.ndarray) -> tuple[float | None, float | None]:
    if returns.size == 0:
        return None, None
    med = float(np.median(returns))
    mad = float(np.mean(np.abs(returns - med)))
    if not math.isfinite(mad):
        return None, None
    return mad, mad * ANNUALIZE


def effective_min_exceed(
    n_total: int,
    *,
    tail_quantile: float,
    min_exceed: int,
) -> int:
    """Cap POT exceedance minimum to the mass available at ``tail_quantile``.

    Default ``min_exceed=25`` assumes ~756 observations (~57 exceedances at q=0.925).
    Shorter windows (e.g. 252d → ~19 exceedances) need a lower floor so GPD/VaR
    still runs while staying ≥3 for stable fits.
    """
    if n_total <= 0:
        return min_exceed
    pot_mass = max(0.0, 1.0 - float(tail_quantile))
    expected = int(max(3, round(n_total * pot_mass)))
    return min(min_exceed, expected)


def _hill_alpha_from_exceedances(exceedances: np.ndarray) -> float | None:
    """Hill tail index on POT exceedances when GPD shape ξ≤0 (bounded tail)."""
    clean = exceedances[np.isfinite(exceedances) & (exceedances > 0)]
    if clean.size < 3:
        return None
    xs = np.sort(clean)
    u = float(xs[0])
    if u <= 0:
        return None
    logs = np.log(xs / u)
    denom = float(np.sum(logs))
    if denom <= 0 or not math.isfinite(denom):
        return None
    alpha = float(clean.size / denom)
    if not math.isfinite(alpha) or alpha <= 0:
        return None
    return alpha


def _fit_gpd_exceedances(exceedances: np.ndarray) -> tuple[float | None, float | None]:
    if exceedances.size < 3:
        return None, None
    try:
        xi, _loc, beta = stats.genpareto.fit(exceedances, floc=0.0)
    except (ValueError, RuntimeError, FloatingPointError):
        return None, None
    if not math.isfinite(xi) or not math.isfinite(beta) or beta <= 0:
        return None, None
    return float(xi), float(beta)


def fit_tail(
    values: np.ndarray,
    *,
    side: str,
    tail_quantile: float,
    min_exceed: int,
) -> TailFit:
    flags: list[str] = []
    clean = values[np.isfinite(values)]
    n_total = int(clean.size)
    if n_total < min_exceed:
        return TailFit(side, None, None, None, None, 0, n_total, False, ("low_n",))

    if side == "left":
        # Standard POT: threshold on the full loss distribution (losses = -returns),
        # not the positive-loss subset, so exceedance count matches the right tail.
        losses = -clean
        threshold = float(np.quantile(losses, tail_quantile))
        exceedances = losses[losses > threshold] - threshold
    else:
        threshold = float(np.quantile(clean, tail_quantile))
        exceedances = clean[clean > threshold] - threshold

    n_exceed = int(exceedances.size)
    if n_exceed < min_exceed:
        flags.append("alpha_unreliable")
        return TailFit(
            side,
            threshold,
            None,
            None,
            None,
            n_exceed,
            n_total,
            False,
            tuple(flags),
        )

    xi, beta = _fit_gpd_exceedances(exceedances)
    if xi is None or beta is None:
        flags.append("alpha_unreliable")
        return TailFit(
            side,
            threshold,
            None,
            None,
            None,
            n_exceed,
            n_total,
            False,
            tuple(flags),
        )

    if xi <= 0:
        # Side-specific so a benign bounded RIGHT tail is never mistaken for a
        # bounded (safe) LEFT tail on the left-tail risk board.
        flags.append(f"bounded_tail_{side}")
        alpha = _hill_alpha_from_exceedances(exceedances)
    else:
        alpha = 1.0 / xi

    return TailFit(
        side,
        threshold,
        xi,
        beta,
        alpha,
        n_exceed,
        n_total,
        True,
        tuple(flags),
    )


def compute_kappa(
    returns: np.ndarray,
    *,
    n: int = 30,
    resamples: int = 2000,
    rng: np.random.Generator | None = None,
    seed: int = 12345,
) -> float | None:
    """Taleb kappa_{1,n} via iid bootstrap aggregation.

    M(k) = E|S_k - k*mu| (mean abs deviation of the sum of k iid draws).
    Drawing returns with replacement removes serial dependence (mean reversion /
    momentum), so kappa measures marginal tail-thickness rather than the speed of
    autocorrelated aggregation that overlapping/blocked sums would conflate.
    """
    if returns.size < n + 20:
        return None
    mu = float(np.mean(returns))
    m1 = float(np.mean(np.abs(returns - mu)))
    if not math.isfinite(m1) or m1 <= 0:
        return None

    gen = rng or np.random.default_rng(seed)
    idx = gen.integers(0, returns.size, size=(resamples, n))
    sums = returns[idx].sum(axis=1)
    m_n = float(np.mean(np.abs(sums - n * mu)))
    if not math.isfinite(m_n) or m_n <= 0:
        return None
    ratio = m_n / m1
    if ratio <= 1:
        return 0.0
    kappa = 2.0 - math.log(n) / math.log(ratio)
    if not math.isfinite(kappa):
        return None
    return float(np.clip(kappa, 0.0, 1.0))


def ewma_mean(returns: np.ndarray, span: int) -> float | None:
    if returns.size == 0:
        return None
    series = pd.Series(returns)
    value = series.ewm(span=span, adjust=False).mean().iloc[-1]
    if pd.isna(value) or not math.isfinite(float(value)):
        return None
    return float(value)


def _gpd_var_cvar_loss(
    *,
    threshold: float,
    xi: float,
    beta: float,
    n_exceed: int,
    n_total: int,
    confidence: float,
) -> tuple[float | None, float | None]:
    if n_total <= 0 or n_exceed <= 0 or confidence <= 0 or confidence >= 1:
        return None, None
    nu = n_exceed / n_total
    tail_prob = 1.0 - confidence
    if tail_prob <= 0 or nu <= tail_prob:
        return None, None

    if abs(xi) < 1e-8:
        y = -beta * math.log(tail_prob / nu)
    else:
        y = (beta / xi) * ((tail_prob / nu) ** (-xi) - 1.0)

    if not math.isfinite(y) or y < 0:
        return None, None

    var_loss = threshold + y
    if xi >= 1:
        cvar_loss = None
    else:
        cvar_loss = var_loss + (beta + xi * (var_loss - threshold)) / (1.0 - xi)

    return var_loss, cvar_loss


def gpd_loss_at_prob(
    *,
    xi: float | None,
    beta: float | None,
    threshold: float | None,
    nu: float | None,
    tail_prob: float,
) -> float | None:
    """Parametric POT-GPD loss (positive magnitude) exceeded with probability `tail_prob`.

    `nu` is the exceedance ratio n_exceed/n_total. Returns None when parameters are
    missing or the requested probability lies above the modelled tail (tail_prob >= nu).
    Used to recompute losses at arbitrary depths for concavity / crash aggregation.
    """
    if xi is None or beta is None or threshold is None or nu is None:
        return None
    if not (0.0 < tail_prob < 1.0) or nu <= 0.0 or nu >= 1.0:
        return None
    if tail_prob >= nu:
        return None
    if abs(xi) < 1e-8:
        y = -beta * math.log(tail_prob / nu)
    else:
        y = (beta / xi) * ((tail_prob / nu) ** (-xi) - 1.0)
    if not math.isfinite(y) or y < 0:
        return None
    loss = threshold + y
    return float(loss) if math.isfinite(loss) else None


def parametric_left_tail_risk(
    returns: np.ndarray,
    left_fit: TailFit,
    *,
    confidence: float,
) -> tuple[float | None, float | None]:
    if not left_fit.reliable or left_fit.threshold is None:
        return None, None
    if left_fit.xi is None or left_fit.beta is None:
        return None, None

    var_loss, cvar_loss = _gpd_var_cvar_loss(
        threshold=left_fit.threshold,
        xi=left_fit.xi,
        beta=left_fit.beta,
        n_exceed=left_fit.n_exceed,
        n_total=left_fit.n_total,
        confidence=confidence,
    )
    if var_loss is None:
        return None, None
    var_return = -var_loss
    cvar_return = -cvar_loss if cvar_loss is not None else None
    return var_return, cvar_return


def shadow_mean(
    returns: np.ndarray,
    left_fit: TailFit,
    right_fit: TailFit,
) -> float | None:
    if returns.size == 0:
        return None
    raw = float(np.mean(returns))
    adjustment = 0.0

    if left_fit.reliable and left_fit.threshold is not None and left_fit.xi is not None:
        if left_fit.xi < 1 and left_fit.beta is not None:
            losses = -returns
            nu = left_fit.n_exceed / max(left_fit.n_total, 1)
            gpd_tail_mean = left_fit.threshold + left_fit.beta / (1.0 - left_fit.xi)
            tail_mask = losses > left_fit.threshold
            if tail_mask.any():
                emp_tail = float(losses[tail_mask].mean())
            else:
                emp_tail = left_fit.threshold
            adjustment -= nu * (gpd_tail_mean - emp_tail)

    if right_fit.reliable and right_fit.threshold is not None and right_fit.xi is not None:
        if right_fit.xi < 1 and right_fit.beta is not None:
            nu = right_fit.n_exceed / max(right_fit.n_total, 1)
            gpd_tail_mean = right_fit.threshold + right_fit.beta / (1.0 - right_fit.xi)
            tail_mask = returns > right_fit.threshold
            if tail_mask.any():
                emp_tail = float(returns[tail_mask].mean())
            else:
                emp_tail = right_fit.threshold
            adjustment += nu * (gpd_tail_mean - emp_tail)

    value = raw + adjustment
    return value if math.isfinite(value) else raw


def shrink_factor(
    *,
    alpha_left: float | None,
    kappa: float | None,
    n_samples: int,
    cfg: FatTailConfig,
) -> float:
    s = 1.0
    if alpha_left is not None and math.isfinite(alpha_left) and alpha_left > 0:
        s *= float(np.clip((alpha_left - 1.0) / 3.0, 0.05, 1.0))
    else:
        s *= 0.35
    if kappa is not None and math.isfinite(kappa):
        s *= float(np.clip(1.0 - kappa, 0.05, 1.0))
    if n_samples < cfg.shrink_min_n:
        s *= n_samples / cfg.shrink_min_n
    return float(np.clip(s, 0.0, 1.0))


def barbell_leg(
    *,
    alpha_left: float | None,
    kappa: float | None,
    cvar_left: float | None,
    flags: tuple[str, ...],
    cfg: FatTailConfig,
    is_defensive: bool = False,
) -> BarbellLeg:
    # Genuinely fat left tail -> avoid (over-rides everything else).
    fat_left = alpha_left is not None and alpha_left < cfg.alpha_fat
    high_kappa = kappa is not None and kappa >= cfg.kappa_fat_min
    extreme_cvar = cvar_left is not None and cvar_left <= -cfg.cvar_red
    if fat_left or high_kappa or extreme_cvar:
        return "avoid"

    # Defensive mandate assets (bonds / cash proxies) anchor the safe leg
    # whenever they are not flagged as fat-tailed above.
    if is_defensive:
        return "safe_leg"

    # Unknown / unreliable estimates stay in the middle.
    if {"low_n", "alpha_unreliable", "missing_price"} & set(flags):
        return "risk_leg"

    # `bounded_tail_left` (xi_left<=0) means a sub-power-law (thin/bounded) LEFT
    # tail. A bounded RIGHT tail says nothing about left-tail safety, so only the
    # left-side flag may count toward `thin_left`.
    thin_left = "bounded_tail_left" in flags or (
        alpha_left is not None and alpha_left >= cfg.alpha_thin
    )
    low_kappa = kappa is None or kappa <= cfg.kappa_safe_max
    safe_cvar = cvar_left is None or cvar_left >= -cfg.cvar_safe_max
    if thin_left and low_kappa and safe_cvar:
        return "safe_leg"
    return "risk_leg"


def compute_fat_tail_metrics(
    code: str,
    *,
    name: str = "",
    weight: float = 0.0,
    close: pd.Series,
    end: pd.Timestamp,
    cfg: FatTailConfig | None = None,
    rng: np.random.Generator | None = None,
    is_defensive: bool = False,
) -> FatTailMetrics:
    cfg = cfg or FatTailConfig()
    returns = slice_log_returns(close, end, lookback_days=cfg.lookback_days)
    n = int(returns.size)
    flags: list[str] = []
    if n < cfg.min_samples:
        flags.append("low_n")

    mad, mad_ann = compute_mad(returns) if n > 0 else (None, None)
    min_exceed = effective_min_exceed(
        n,
        tail_quantile=cfg.tail_quantile,
        min_exceed=cfg.min_exceed,
    )
    left_fit = fit_tail(
        returns,
        side="left",
        tail_quantile=cfg.tail_quantile,
        min_exceed=min_exceed,
    )
    right_fit = fit_tail(
        returns,
        side="right",
        tail_quantile=cfg.tail_quantile,
        min_exceed=min_exceed,
    )
    flags.extend(left_fit.flags)
    flags.extend(right_fit.flags)
    flags = sorted(set(flags))

    kappa = compute_kappa(
        returns,
        n=cfg.kappa_n,
        resamples=cfg.kappa_resamples,
        rng=rng,
        seed=cfg.kappa_seed,
    )
    raw_mean = float(np.mean(returns)) if n > 0 else None
    ewma60 = ewma_mean(returns, cfg.ewma_span) if n > 0 else None
    shadow = shadow_mean(returns, left_fit, right_fit) if n > 0 else None
    s_factor = shrink_factor(
        alpha_left=left_fit.alpha,
        kappa=kappa,
        n_samples=n,
        cfg=cfg,
    )
    shrunk = raw_mean * s_factor if raw_mean is not None else None
    var_l, cvar_l = parametric_left_tail_risk(
        returns,
        left_fit,
        confidence=cfg.var_confidence,
    )

    leg = barbell_leg(
        alpha_left=left_fit.alpha,
        kappa=kappa,
        cvar_left=cvar_l,
        flags=tuple(flags),
        cfg=cfg,
        is_defensive=is_defensive,
    )

    return FatTailMetrics(
        code=code,
        name=name,
        weight=weight,
        n_samples=n,
        mad=mad,
        mad_ann=mad_ann,
        alpha_left=left_fit.alpha,
        alpha_right=right_fit.alpha,
        kappa=kappa,
        raw_mean=raw_mean,
        ewma60_mean=ewma60,
        shadow_mean=shadow,
        shrink_factor=s_factor,
        shrunk_mean=shrunk,
        var_left=var_l,
        cvar_left=cvar_l,
        barbell_leg=leg,
        xi_left=left_fit.xi if left_fit.reliable else None,
        beta_left=left_fit.beta if left_fit.reliable else None,
        threshold_left=left_fit.threshold if left_fit.reliable else None,
        nu_left=(
            left_fit.n_exceed / left_fit.n_total
            if left_fit.reliable and left_fit.n_total > 0
            else None
        ),
        flags=tuple(flags),
    )


def build_fat_tail_frame(
    codes: list[str],
    close_panel: pd.DataFrame,
    as_of: str,
    *,
    weights: dict[str, float] | None = None,
    names: dict[str, str] | None = None,
    defensive_codes: set[str] | frozenset[str] | None = None,
    cfg: FatTailConfig | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    cfg = cfg or FatTailConfig()
    end = pd.Timestamp(as_of)
    weights = weights or {}
    names = names or {}
    defensive = set(defensive_codes or set())
    rows: list[dict[str, Any]] = []

    for code in codes:
        is_defensive = code in defensive
        if code not in close_panel.columns:
            rows.append(
                {
                    **FatTailMetrics(
                        code=code,
                        name=names.get(code, code),
                        weight=float(weights.get(code, 0.0)),
                        n_samples=0,
                        mad=None,
                        mad_ann=None,
                        alpha_left=None,
                        alpha_right=None,
                        kappa=None,
                        raw_mean=None,
                        ewma60_mean=None,
                        shadow_mean=None,
                        shrink_factor=None,
                        shrunk_mean=None,
                        var_left=None,
                        cvar_left=None,
                        barbell_leg="risk_leg",
                        flags=("missing_price",),
                    ).to_dict(),
                    "is_defensive": is_defensive,
                }
            )
            continue

        series = pd.to_numeric(close_panel[code], errors="coerce").dropna()
        metrics = compute_fat_tail_metrics(
            code,
            name=names.get(code, code),
            weight=float(weights.get(code, 0.0)),
            close=series,
            end=end,
            cfg=cfg,
            rng=rng,
            is_defensive=is_defensive,
        )
        rows.append({**metrics.to_dict(), "is_defensive": is_defensive})

    columns = [
        "code",
        "name",
        "weight",
        "n_samples",
        "mad",
        "mad_ann",
        "alpha_left",
        "alpha_right",
        "kappa",
        "raw_mean",
        "ewma60_mean",
        "shadow_mean",
        "shrink_factor",
        "shrunk_mean",
        "var_left",
        "cvar_left",
        "barbell_leg",
        "xi_left",
        "beta_left",
        "threshold_left",
        "nu_left",
        "is_defensive",
        "flags",
    ]
    return pd.DataFrame(rows, columns=columns)


def portfolio_weighted_summary(
    frame: pd.DataFrame,
    *,
    weight_col: str = "weight",
) -> dict[str, Any]:
    """Portfolio-level fat-tail summary.

    Notes on aggregation:
    - alpha (tail exponent) is NOT averaged linearly: a single near-degenerate
      estimate (xi->0, alpha->inf) would dominate and make the book look thin.
      Instead we average in xi=1/alpha space and convert back, and we also report
      the fattest (smallest-alpha) held tail as the true risk driver.
    - Each weighted figure also reports its coverage (fraction of portfolio
      weight that had an estimable value), so silently-dropped holdings (e.g.
      short-history names without an alpha/CVaR fit) are visible.
    """
    empty: dict[str, Any] = {
        "weighted_kappa": None,
        "weighted_alpha_left": None,
        "weighted_cvar_left": None,
        "fattest_alpha_left": None,
        "fattest_code": None,
        "coverage_kappa": 0.0,
        "coverage_alpha_left": 0.0,
        "coverage_cvar_left": 0.0,
        "safe_leg_weight": 0.0,
        "risk_leg_weight": 0.0,
        "avoid_weight": 0.0,
    }
    if frame.empty:
        return empty

    weights = pd.to_numeric(frame[weight_col], errors="coerce").fillna(0.0)
    total = float(weights.sum())
    if total <= 0:
        weights = pd.Series([1.0 / len(frame)] * len(frame), index=frame.index)
    else:
        weights = weights / total

    def _wavg(col: str) -> tuple[float | None, float]:
        values = pd.to_numeric(frame[col], errors="coerce")
        mask = values.notna()
        covered = float(weights[mask].sum())
        if not mask.any():
            return None, 0.0
        w = weights[mask]
        w = w / w.sum()
        return float((values[mask] * w).sum()), covered

    def _wavg_alpha_xi(col: str) -> tuple[float | None, float]:
        values = pd.to_numeric(frame[col], errors="coerce")
        mask = values.notna() & (values > 0)
        covered = float(weights[mask].sum())
        if not mask.any():
            return None, 0.0
        xi = 1.0 / values[mask]
        w = weights[mask]
        w = w / w.sum()
        xi_bar = float((xi * w).sum())
        if not math.isfinite(xi_bar) or xi_bar <= 0:
            return None, covered
        return 1.0 / xi_bar, covered

    weighted_kappa, cov_kappa = _wavg("kappa")
    weighted_alpha, cov_alpha = _wavg_alpha_xi("alpha_left")
    weighted_cvar, cov_cvar = _wavg("cvar_left")

    # Fattest left tail among *held* names (smallest alpha = heaviest tail).
    held_mask = weights > 0
    if not held_mask.any():
        held_mask = pd.Series(True, index=frame.index)
    alpha_vals = pd.to_numeric(frame["alpha_left"], errors="coerce")
    amask = alpha_vals.notna() & held_mask
    if amask.any():
        fattest_idx = alpha_vals[amask].idxmin()
        fattest_alpha: float | None = float(alpha_vals.loc[fattest_idx])
        fattest_code: str | None = str(frame.loc[fattest_idx, "code"])
    else:
        fattest_alpha = None
        fattest_code = None

    leg_weights = frame.groupby("barbell_leg")[weight_col].sum()
    return {
        "weighted_kappa": weighted_kappa,
        "weighted_alpha_left": weighted_alpha,
        "weighted_cvar_left": weighted_cvar,
        "fattest_alpha_left": fattest_alpha,
        "fattest_code": fattest_code,
        "coverage_kappa": cov_kappa,
        "coverage_alpha_left": cov_alpha,
        "coverage_cvar_left": cov_cvar,
        "safe_leg_weight": float(leg_weights.get("safe_leg", 0.0)),
        "risk_leg_weight": float(leg_weights.get("risk_leg", 0.0)),
        "avoid_weight": float(leg_weights.get("avoid", 0.0)),
    }


__all__ = [
    "BarbellLeg",
    "FatTailConfig",
    "FatTailMetrics",
    "TailFit",
    "barbell_leg",
    "build_fat_tail_frame",
    "compute_fat_tail_metrics",
    "compute_kappa",
    "compute_mad",
    "effective_min_exceed",
    "fat_tail_config_from_raw",
    "fit_tail",
    "gpd_loss_at_prob",
    "parametric_left_tail_risk",
    "portfolio_weighted_summary",
    "shadow_mean",
    "shrink_factor",
    "slice_log_returns",
]
