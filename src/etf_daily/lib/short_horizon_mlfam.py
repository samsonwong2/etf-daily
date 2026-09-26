"""MLFAM short-horizon board metrics (§5, §13).

Replaces literal short-term mu / distribution point forecasts with:
- trend-scanning direction labels
- triple-barrier Monte-Carlo event probabilities
- meta-labeling bet sizing
- DGP-based cumulative return quantiles (not a point mu)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from etf_daily.lib.fat_tail_risk import compute_mad, slice_log_returns
from etf_daily.lib.indicators import slice_through

TrendLabel = Literal["up", "down", "flat"]
DirectionLabel = Literal["up", "down", "flat", "不下注"]
BarrierOutcome = Literal["up", "down", "timeout"]


@dataclass(frozen=True)
class ShortHorizonConfig:
    horizon: int = 10
    win_min: int = 5
    win_max: int = 10
    trend_win_from_horizon: bool = True
    trend_win_min_ratio: float = 0.5
    trend_win_floor: int = 5
    t_thresh: float = 2.0
    barrier_mult: float = 2.0
    mc_samples: int = 2000
    mc_seed: int = 12345
    conf_min: float = 0.15
    min_samples: int = 60
    kelly_fraction: float = 0.25
    max_name_weight: float = 0.20

    def trend_win_range(self) -> tuple[int, int]:
        """Trend-scan lookback window; scales with ``horizon`` when enabled."""
        if not self.trend_win_from_horizon:
            return (self.win_min, self.win_max)
        win_max = max(3, int(self.horizon))
        win_min = max(
            self.trend_win_floor,
            round(float(self.horizon) * float(self.trend_win_min_ratio)),
        )
        win_min = min(win_min, win_max)
        win_min = max(3, win_min)
        return (win_min, win_max)


def short_horizon_config_from_raw(raw: dict | None) -> ShortHorizonConfig:
    if not raw or not isinstance(raw, dict):
        return ShortHorizonConfig()
    kwargs: dict = {}
    for key in ShortHorizonConfig.__dataclass_fields__:
        if key in raw and raw[key] is not None:
            kwargs[key] = raw[key]
    return ShortHorizonConfig(**kwargs)


def _trend_scan_window(
    log_prices: np.ndarray,
    t_thresh: float,
) -> tuple[TrendLabel, float | None, int, float | None]:
    n = log_prices.size
    if n < 3:
        return "flat", None, n, None
    x = np.arange(n, dtype=float)
    slope, _, _, pvalue, se_slope = stats.linregress(x, log_prices)
    tval = float(slope / se_slope) if se_slope and se_slope > 0 else 0.0
    pval = float(pvalue) if math.isfinite(pvalue) else None
    if not math.isfinite(tval) or abs(tval) < t_thresh:
        return "flat", tval if math.isfinite(tval) else None, n, pval
    if slope > 0:
        return "up", tval, n, pval
    if slope < 0:
        return "down", tval, n, pval
    return "flat", tval, n, pval


def trend_scan(
    close: pd.Series,
    end: pd.Timestamp,
    *,
    win_range: tuple[int, int] = (5, 10),
    t_thresh: float = 2.0,
) -> tuple[TrendLabel, float | None, int, float | None]:
    """Trend-scanning label: pick window L* in [win_min, win_max] with min p(slope)."""
    available = slice_through(end, close)
    if available.size < win_range[0] + 1:
        return "flat", None, 0, None

    best_label: TrendLabel = "flat"
    best_t: float | None = None
    best_p: float | None = None
    best_win = 0
    best_pkey = float("inf")

    for win in range(win_range[0], win_range[1] + 1):
        if available.size < win:
            continue
        window = available.tail(win).astype(float)
        if (window <= 0).any():
            continue
        log_prices = np.log(window.values)
        label, tval, w, pval = _trend_scan_window(log_prices, t_thresh)
        pkey = float(pval) if pval is not None and math.isfinite(pval) else float("inf")
        if pkey < best_pkey or (pkey == best_pkey and w > best_win):
            best_pkey = pkey
            best_label = label
            best_t = tval
            best_p = pval if math.isfinite(pkey) else None
            best_win = w

    return best_label, best_t, best_win, best_p


def _simulate_paths(
    returns: np.ndarray,
    *,
    horizon: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """IID bootstrap cumulative log-return paths, shape (n_paths, horizon)."""
    gen = np.random.default_rng(seed)
    idx = gen.integers(0, returns.size, size=(n_paths, horizon))
    daily = returns[idx]
    return np.cumsum(daily, axis=1)


def triple_barrier_mc(
    returns: np.ndarray,
    *,
    horizon: int,
    barrier_mult: float,
    n_paths: int,
    seed: int,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Monte-Carlo triple-barrier first-touch probabilities (up / down / time-out)."""
    clean = returns[np.isfinite(returns)]
    if clean.size < 20 or horizon < 1 or n_paths < 1:
        return None, None, None, None

    mad, _ = compute_mad(clean)
    if mad is None or mad <= 0:
        return None, None, None, None
    barrier = float(barrier_mult * mad)
    if barrier <= 0:
        return None, None, None, None

    paths = _simulate_paths(clean, horizon=horizon, n_paths=n_paths, seed=seed)
    up_hits = 0
    down_hits = 0
    timeouts = 0

    for path in paths:
        hit_up = False
        hit_down = False
        for step, cum in enumerate(path, start=1):
            if cum >= barrier:
                up_hits += 1
                hit_up = True
                break
            if cum <= -barrier:
                down_hits += 1
                hit_down = True
                break
        if not hit_up and not hit_down:
            timeouts += 1

    total = float(n_paths)
    return up_hits / total, down_hits / total, timeouts / total, barrier


def mc_quantiles(
    returns: np.ndarray,
    *,
    horizon: int,
    n_paths: int,
    seed: int,
) -> tuple[float | None, float | None, float | None]:
    """DGP cumulative H-day log-return quantiles (5/50/95%), not a point mu."""
    clean = returns[np.isfinite(returns)]
    if clean.size < 20 or horizon < 1 or n_paths < 1:
        return None, None, None

    paths = _simulate_paths(clean, horizon=horizon, n_paths=n_paths, seed=seed)
    totals = paths[:, -1]
    q05, q50, q95 = np.quantile(totals, [0.05, 0.50, 0.95])
    return float(q05), float(q50), float(q95)


def realize_triple_barrier(
    forward_log_returns: np.ndarray,
    *,
    barrier: float,
) -> BarrierOutcome:
    """First-touch outcome on an realized forward log-return path."""
    if barrier <= 0 or forward_log_returns.size == 0:
        return "timeout"
    cum = 0.0
    for r in forward_log_returns:
        if not math.isfinite(float(r)):
            continue
        cum += float(r)
        if cum >= barrier:
            return "up"
        if cum <= -barrier:
            return "down"
    return "timeout"


def forward_log_returns(
    close: pd.Series,
    end: pd.Timestamp,
    *,
    horizon: int,
) -> np.ndarray | None:
    """H daily log returns strictly after ``end`` (exclusive)."""
    if horizon < 1:
        return None
    available = close.loc[close.index <= end].dropna()
    future = close.loc[close.index > end].dropna()
    if available.size < 1 or future.size < horizon:
        return None
    path = pd.concat([available.tail(1), future.head(horizon)]).astype(float)
    if (path <= 0).any():
        return None
    rets = np.diff(np.log(path.values))
    if rets.size != horizon:
        return None
    return rets.astype(float)


def forward_cumulative_log_return(forward_log_returns: np.ndarray) -> float | None:
    if forward_log_returns.size == 0:
        return None
    clean = forward_log_returns[np.isfinite(forward_log_returns)]
    if clean.size == 0:
        return None
    return float(np.sum(clean))


def meta_label(
    *,
    trend_label: TrendLabel,
    p_up: float | None,
    p_down: float | None,
    tval: float | None,
    kelly_fraction: float = 0.25,
    max_w: float = 0.20,
    conf_min: float = 0.15,
) -> tuple[DirectionLabel, float | None, float | None]:
    """Meta-labeling: direction from trend-scan; bet size from barrier confidence."""
    if p_up is None or p_down is None:
        return "不下注", None, 0.0

    conf = abs(float(p_up) - float(p_down))
    if trend_label == "flat" or tval is None or not math.isfinite(tval):
        direction: DirectionLabel = "flat"
    elif trend_label == "up":
        direction = "up"
    elif trend_label == "down":
        direction = "down"
    else:
        direction = "flat"

    if conf < conf_min or direction == "flat":
        return "不下注", conf, 0.0

    bet = float(np.clip(kelly_fraction * conf, 0.0, max_w))
    return direction, conf, bet


@dataclass(frozen=True)
class ShortHorizonMetrics:
    trend_label: TrendLabel
    trend_t: float | None
    trend_p: float | None
    trend_win: int
    p_up: float | None
    p_down: float | None
    p_timeout: float | None
    barrier: float | None
    direction: DirectionLabel
    confidence: float | None
    bet_size: float | None
    q05: float | None
    q50: float | None
    q95: float | None
    reliable: bool
    basis: str


def compute_short_horizon_metrics(
    close: pd.Series,
    end: pd.Timestamp,
    *,
    cfg: ShortHorizonConfig | None = None,
    flags: tuple[str, ...] | str = (),
) -> ShortHorizonMetrics:
    cfg = cfg or ShortHorizonConfig()
    if isinstance(flags, str):
        flag_set = {f for f in flags.split("|") if f}
    else:
        flag_set = {str(f) for f in (flags or ())}

    returns = slice_log_returns(close, end, lookback_days=756)
    n = int(returns.size)

    trend_label, trend_t, trend_win, trend_p = trend_scan(
        close,
        end,
        win_range=cfg.trend_win_range(),
        t_thresh=cfg.t_thresh,
    )

    if "missing_price" in flag_set or n < 20:
        return ShortHorizonMetrics(
            trend_label=trend_label,
            trend_t=trend_t,
            trend_p=trend_p,
            trend_win=trend_win,
            p_up=None,
            p_down=None,
            p_timeout=None,
            barrier=None,
            direction="不下注",
            confidence=None,
            bet_size=0.0,
            q05=None,
            q50=None,
            q95=None,
            reliable=False,
            basis="缺价格·仅展示" if "missing_price" in flag_set else "数据不足·仅展示",
        )

    caution: str | None = None
    if "alpha_unreliable" in flag_set:
        caution = "α不可靠"
    elif "low_n" in flag_set or n < cfg.min_samples:
        caution = "低样本"
    reliable = caution is None

    p_up, p_down, p_timeout, barrier = triple_barrier_mc(
        returns,
        horizon=cfg.horizon,
        barrier_mult=cfg.barrier_mult,
        n_paths=cfg.mc_samples,
        seed=cfg.mc_seed,
    )
    q05, q50, q95 = mc_quantiles(
        returns,
        horizon=cfg.horizon,
        n_paths=cfg.mc_samples,
        seed=cfg.mc_seed + 1,
    )
    direction, confidence, bet_size = meta_label(
        trend_label=trend_label,
        p_up=p_up,
        p_down=p_down,
        tval=trend_t,
        kelly_fraction=cfg.kelly_fraction,
        max_w=cfg.max_name_weight,
        conf_min=cfg.conf_min,
    )

    basis = "趋势扫描+三重门(MC)"
    if caution:
        basis = f"{basis}·{caution}"

    return ShortHorizonMetrics(
        trend_label=trend_label,
        trend_t=trend_t,
        trend_p=trend_p,
        trend_win=trend_win,
        p_up=p_up,
        p_down=p_down,
        p_timeout=p_timeout,
        barrier=barrier,
        direction=direction,
        confidence=confidence,
        bet_size=bet_size,
        q05=q05,
        q50=q50,
        q95=q95,
        reliable=reliable,
        basis=basis,
    )


__all__ = [
    "BarrierOutcome",
    "DirectionLabel",
    "ShortHorizonConfig",
    "ShortHorizonMetrics",
    "compute_short_horizon_metrics",
    "forward_cumulative_log_return",
    "forward_log_returns",
    "mc_quantiles",
    "meta_label",
    "realize_triple_barrier",
    "short_horizon_config_from_raw",
    "trend_scan",
    "triple_barrier_mc",
]
