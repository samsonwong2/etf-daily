"""§7 tail-structure regime + §12 path/drawdown state for fat-tail risk report.

Default stance: do not switch the fitted distribution on large moves (Peso problem).
Full-window EVT VaR tiers stay unchanged. The per-name "state" is a path / drawdown
measure (book-endorsed: §12.3 high-water-mark, §9 path dependence / ruin), NOT an
MA / 20-day-return trend (the book de-emphasizes mean-based signals as noisy under
fat tails, §3 / §7.1, and gives no MA rules, §12).
"""
from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pandas as pd

from etf_daily.lib.indicators import compute_trend_metrics, drawdown_from_high, slice_through
from etf_daily.lib.fat_tail_risk import (
    FatTailConfig,
    effective_min_exceed,
    fit_tail,
    slice_log_returns,
)

TailRegime = Literal["stable", "tail_thickening_watch", "tail_thickening_confirmed"]
PathState = Literal["near_high", "pullback", "broken"]

_TAIL_REGIME_ZH = {
    "stable": "稳定",
    "tail_thickening_watch": "尾变厚·观察",
    "tail_thickening_confirmed": "尾漂移·建议重估",
}
_PATH_STATE_ZH = {
    "near_high": "近高位",
    "pullback": "回撤中",
    "broken": "破位",
}


def fit_alpha_left_recent(
    close: pd.Series,
    end: pd.Timestamp,
    cfg: FatTailConfig | None = None,
) -> tuple[float | None, bool]:
    """Left-tail α on the recent window only (does not replace full-window α_L)."""
    cfg = cfg or FatTailConfig()
    returns = slice_log_returns(close, end, lookback_days=cfg.regime_alpha_recent_days)
    if returns.size < cfg.min_samples:
        return None, False
    fit = fit_tail(
        returns,
        side="left",
        tail_quantile=cfg.tail_quantile,
        min_exceed=effective_min_exceed(
            int(returns.size),
            tail_quantile=cfg.tail_quantile,
            min_exceed=cfg.min_exceed,
        ),
    )
    if not fit.reliable or fit.alpha is None:
        return None, False
    return float(fit.alpha), True


def _alpha_ratio_at_end(
    close: pd.Series,
    end: pd.Timestamp,
    cfg: FatTailConfig,
) -> tuple[float | None, float | None, bool]:
    """Return (anchor_alpha, recent_alpha, both_reliable) at ``end``."""
    anchor_returns = slice_log_returns(close, end, lookback_days=cfg.lookback_days)
    if anchor_returns.size < cfg.min_samples:
        return None, None, False
    anchor_fit = fit_tail(
        anchor_returns,
        side="left",
        tail_quantile=cfg.tail_quantile,
        min_exceed=effective_min_exceed(
            int(anchor_returns.size),
            tail_quantile=cfg.tail_quantile,
            min_exceed=cfg.min_exceed,
        ),
    )
    recent_alpha, recent_ok = fit_alpha_left_recent(close, end, cfg)
    if not anchor_fit.reliable or anchor_fit.alpha is None or not recent_ok:
        return (
            float(anchor_fit.alpha) if anchor_fit.alpha is not None else None,
            recent_alpha,
            False,
        )
    return float(anchor_fit.alpha), recent_alpha, True


def classify_tail_regime(
    close: pd.Series,
    end: pd.Timestamp,
    *,
    anchor_alpha: float | None,
    recent_alpha: float | None,
    anchor_reliable: bool,
    kappa: float | None,
    flags: tuple[str, ...] | str,
    cfg: FatTailConfig | None = None,
) -> TailRegime:
    """§7 default stable; upgrade only on sustained α drift (not single jumps)."""
    cfg = cfg or FatTailConfig()
    flag_set = set(str(flags).split("|")) if flags else set()
    if "missing_price" in flag_set:
        return "stable"
    if not anchor_reliable or anchor_alpha is None or recent_alpha is None:
        return "stable"
    if anchor_alpha <= 0 or recent_alpha <= 0:
        return "stable"

    ratio = recent_alpha / anchor_alpha
    watch_hit = ratio < cfg.regime_alpha_watch_ratio

  # Confirm streak: ratio below confirm threshold at consecutive checkpoints.
    confirm_hits = 0
    available = slice_through(end, close)
    step = max(1, int(cfg.regime_confirm_step_days))
    for i in range(max(1, int(cfg.regime_alpha_confirm_streak))):
        idx = len(available) - 1 - i * step
        if idx < cfg.min_samples:
            break
        checkpoint_end = available.index[idx]
        _, _, ok = _alpha_ratio_at_end(close, checkpoint_end, cfg)
        if not ok:
            break
        checkpoint_returns = slice_log_returns(close, checkpoint_end, lookback_days=cfg.lookback_days)
        checkpoint_recent = slice_log_returns(
            close, checkpoint_end, lookback_days=cfg.regime_alpha_recent_days
        )
        if checkpoint_returns.size < cfg.min_samples or checkpoint_recent.size < cfg.min_samples:
            break
        a_fit = fit_tail(
            checkpoint_returns,
            side="left",
            tail_quantile=cfg.tail_quantile,
            min_exceed=effective_min_exceed(
                int(checkpoint_returns.size),
                tail_quantile=cfg.tail_quantile,
                min_exceed=cfg.min_exceed,
            ),
        )
        r_alpha, r_ok = fit_alpha_left_recent(close, checkpoint_end, cfg)
        if not a_fit.reliable or a_fit.alpha is None or not r_ok or r_alpha is None:
            break
        if float(r_alpha) / float(a_fit.alpha) < cfg.regime_alpha_confirm_ratio:
            confirm_hits += 1
        else:
            break

    kappa_high = (
        kappa is not None
        and math.isfinite(float(kappa))
        and float(kappa) >= cfg.regime_kappa_watch_min
    )
    if confirm_hits >= cfg.regime_alpha_confirm_streak:
        return "tail_thickening_confirmed"
    if watch_hit and (kappa_high or confirm_hits > 0):
        return "tail_thickening_watch"
    if watch_hit:
        return "tail_thickening_watch"
    return "stable"


def _finite(x: Any) -> bool:
    return x is not None and pd.notna(x) and math.isfinite(float(x))


def classify_path_state(metrics: dict[str, Any], cfg: FatTailConfig | None = None) -> PathState:
    """Path / drawdown state from high-water-mark (§12.3 / §9), not MA trend.

    ``dd_long`` (longer window, e.g. 120d) is the primary "trend broken" judge so a
    transient short-window dip does not flip the state; ``dd_near`` (e.g. 20d) can at
    most mark a pullback when the longer window is still intact. Drawdowns are
    fractions <= 0 (current/peak - 1).
    """
    cfg = cfg or FatTailConfig()
    dd_long = metrics.get("dd_long")
    dd_near = metrics.get("dd_near")

    if _finite(dd_long):
        if float(dd_long) <= -cfg.drawdown_broken_pct:
            return "broken"
        if float(dd_long) <= -cfg.drawdown_pullback_pct:
            return "pullback"
        # Long window intact: a sharp near-term drop is at most a pullback, not broken.
        if _finite(dd_near) and float(dd_near) <= -cfg.drawdown_broken_pct:
            return "pullback"
        return "near_high"

    if _finite(dd_near):
        if float(dd_near) <= -cfg.drawdown_broken_pct:
            return "broken"
        if float(dd_near) <= -cfg.drawdown_pullback_pct:
            return "pullback"
    return "near_high"


def _regime_note(tail_regime: TailRegime, path_state: PathState) -> str:
    parts = [_TAIL_REGIME_ZH.get(tail_regime, tail_regime)]
    parts.append(_PATH_STATE_ZH.get(path_state, path_state))
    return "；".join(parts)


def enrich_regime_columns(
    frame: pd.DataFrame,
    close_panel: pd.DataFrame,
    as_of: str,
    cfg: FatTailConfig | None = None,
) -> pd.DataFrame:
    """Append tail_regime, path_state, drawdowns, display metrics, regime_note to ``frame``."""
    cfg = cfg or FatTailConfig()
    end = pd.Timestamp(as_of)
    out = frame.copy()
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        code = str(row["code"])
        flags = row.get("flags")
        flag_tuple: tuple[str, ...]
        if isinstance(flags, str):
            flag_tuple = tuple(f for f in flags.split("|") if f)
        elif isinstance(flags, (list, tuple)):
            flag_tuple = tuple(str(f) for f in flags)
        else:
            flag_tuple = ()

        kappa = row.get("kappa")
        kappa_val = float(kappa) if kappa is not None and pd.notna(kappa) else None
        anchor_alpha = row.get("alpha_left")
        anchor_val = (
            float(anchor_alpha) if anchor_alpha is not None and pd.notna(anchor_alpha) else None
        )

        if code not in close_panel.columns:
            rows.append(
                {
                    "alpha_left_recent": None,
                    "tail_regime": "stable",
                    "path_state": "near_high",
                    "dd_near": None,
                    "dd_long": None,
                    "dist_ma20_pct": None,
                    "ret_20d": None,
                    "ma_stack": None,
                    "regime_note": "缺价格",
                }
            )
            continue

        series = pd.to_numeric(close_panel[code], errors="coerce").dropna()
        recent_alpha, recent_ok = fit_alpha_left_recent(series, end, cfg)
        anchor_reliable = "alpha_unreliable" not in flag_tuple and anchor_val is not None
        tail_regime = classify_tail_regime(
            series,
            end,
            anchor_alpha=anchor_val,
            recent_alpha=recent_alpha,
            anchor_reliable=anchor_reliable,
            kappa=kappa_val,
            flags=flag_tuple,
            cfg=cfg,
        )
        trend_metrics = compute_trend_metrics(
            series,
            end,
            vol_window=20,
            vol_lookback=120,
            dd_lookback=cfg.drawdown_near_days,
        )
        dd_near = drawdown_from_high(series, end, lookback=cfg.drawdown_near_days)
        dd_long = drawdown_from_high(series, end, lookback=cfg.drawdown_long_days)
        path_metrics = {"dd_near": dd_near, "dd_long": dd_long}
        path_state = classify_path_state(path_metrics, cfg)
        rows.append(
            {
                "alpha_left_recent": recent_alpha,
                "tail_regime": tail_regime,
                "path_state": path_state,
                "dd_near": dd_near if math.isfinite(dd_near) else None,
                "dd_long": dd_long if math.isfinite(dd_long) else None,
                "dist_ma20_pct": trend_metrics.get("dist_ma20_pct"),
                "ret_20d": trend_metrics.get("ret_20d"),
                "ma_stack": trend_metrics.get("ma_stack"),
                "regime_note": _regime_note(tail_regime, path_state),
            }
        )

    extra = pd.DataFrame(rows, index=out.index)
    for col in extra.columns:
        out[col] = extra[col]
    return out


__all__ = [
    "classify_path_state",
    "classify_tail_regime",
    "enrich_regime_columns",
    "fit_alpha_left_recent",
]
