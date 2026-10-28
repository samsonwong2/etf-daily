"""Read-only audit helpers for fig7–11 vs the fat-tail framework.

Measurement, protocol constants, and acceptance checks. Does not change
production path, rails, hold_up, or checklist defaults.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from etf_daily.lib.fat_tail_risk import (
    compute_kappa,
    compute_mad,
    fit_tail,
    shadow_mean,
)

# Written before any diagnostic or A/B result. Do not retune after seeing OOS.
PROTOCOL: dict[str, Any] = {
    "dev_end": "2024-12-31",
    "final_start": "2025-01-01",
    "deep_gap": -0.15,
    "max_hold_days": 40,
    "target_panel": "fig10",
    "entry_mode": "B123",
    "pot_quantile": 0.90,
    "min_exceed": 25,
    "block_len": 20,
    "n_boot": 400,
    "boot_seed": 12345,
    "min_independent_days": 60,
    "h1_min_frac": 1.0 / 3.0,
    "h3_flip_min": 0.10,
    "h4_low_exceed_max": 0.50,
    "h2_min_events": 30,
    "variants": ("V0", "V1", "V2", "V3", "V4", "V5", "V6"),
    "sensitivity_not_for_selection": {
        "pot_quantile": (0.85, 0.90, 0.925),
        "ewma_lambda": (0.94,),
    },
    "rails_stamp": "fair-path-rails:legacy-20261019",
}

GOLDEN_CODES: tuple[str, ...] = (
    "SH511260",
    "SH511090",
    "SH510300",
    "SH512530",
    "SH513310",
    "SH518880",
    "SZ159502",
    "SZ159985",
)

PANELS: tuple[tuple[str, float], ...] = (
    ("fig7", 2.0),
    ("fig8", 1.0),
    ("fig9", 0.5),
    ("fig10", 0.25),
    ("fig11", 1.0 / 12.0),
)


def endpoint_leverage(n: int) -> float:
    """OLS hat diagonal at the last point of x = 0..n-1 with intercept.

    h = (4n - 2) / (n(n + 1)).
    """
    n = int(n)
    if n < 2:
        return float("nan")
    return (4.0 * n - 2.0) / (n * (n + 1.0))


def _trail_bounds(t: int, n_window: int) -> tuple[int, int]:
    start = 0 if n_window <= 0 or (t + 1) < n_window else t + 1 - n_window
    return start, t + 1


def insample_and_exante_residual(
    close: pd.Series | np.ndarray,
    *,
    trail_bars: int,
    min_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Log residual of close vs trailing OLS, in-sample and one-step ex-ante.

    In-sample fits ``[start, t]`` and evaluates at t, so today's shock moves
    the ruler. Ex-ante fits ``[start, t)`` only and extrapolates one step,
    so today's close cannot change today's path.
    """
    y = np.asarray(pd.to_numeric(close, errors="coerce"), dtype=float)
    n = int(y.size)
    r_in = np.full(n, np.nan, dtype=float)
    r_ex = np.full(n, np.nan, dtype=float)
    tb = int(trail_bars)
    mb = max(2, int(min_bars))
    for t in range(n):
        ct = float(y[t])
        if not (np.isfinite(ct) and ct > 0):
            continue
        start, end = _trail_bounds(t, tb)
        sl = y[start:end]
        ok = np.isfinite(sl) & (sl > 0)
        if int(ok.sum()) >= mb:
            jj = np.flatnonzero(ok).astype(float)
            log_y = np.log(sl[ok])
            b, a = np.polyfit(jj, log_y, 1)
            if np.isfinite(a) and np.isfinite(b):
                fitted = float(np.exp(a + b * float(jj[-1])))
                if fitted > 0:
                    r_in[t] = float(np.log(ct / fitted))
        # Window of the same length ending yesterday, predict one step ahead.
        if t < 1:
            continue
        end_ex = t  # exclusive: yesterday is t-1
        start_ex = 0 if tb <= 0 or end_ex < tb else end_ex - tb
        sl_ex = y[start_ex:end_ex]
        ok_ex = np.isfinite(sl_ex) & (sl_ex > 0)
        if int(ok_ex.sum()) < mb:
            continue
        jj_ex = np.flatnonzero(ok_ex).astype(float)
        log_ex = np.log(sl_ex[ok_ex])
        b_ex, a_ex = np.polyfit(jj_ex, log_ex, 1)
        if not (np.isfinite(a_ex) and np.isfinite(b_ex)):
            continue
        pred = float(np.exp(a_ex + b_ex * (float(jj_ex[-1]) + 1.0)))
        if pred > 0:
            r_ex[t] = float(np.log(ct / pred))
    return r_in, r_ex


def log_residual(close: pd.Series, path: pd.Series) -> np.ndarray:
    c = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(path, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(c), np.nan, dtype=float)
    ok = np.isfinite(c) & np.isfinite(p) & (c > 0) & (p > 0)
    out[ok] = np.log(c[ok] / p[ok])
    return out


def breach_rates(
    residual: np.ndarray,
    *,
    half_width: np.ndarray,
) -> dict[str, float]:
    """Left/right breach rates against a (possibly asymmetric) half-width.

    ``half_width`` is the positive log distance to the rail used on both
    sides when the rail is symmetric. For split rails pass a 2-column
    array ``(left_width, right_width)``.
    """
    r = np.asarray(residual, dtype=float)
    w = np.asarray(half_width, dtype=float)
    if w.ndim == 1:
        left_w = right_w = w
    else:
        left_w = w[:, 0]
        right_w = w[:, 1]
    ok = np.isfinite(r) & np.isfinite(left_w) & np.isfinite(right_w) & (left_w >= 0) & (right_w >= 0)
    n = int(ok.sum())
    if n == 0:
        return {"n": 0.0, "left_rate": float("nan"), "right_rate": float("nan"), "ratio": float("nan")}
    left = int(np.sum(ok & (r < -left_w)))
    right = int(np.sum(ok & (r > right_w)))
    lr = left / n
    rr = right / n
    ratio = lr / rr if rr > 0 else float("inf")
    return {"n": float(n), "left_rate": float(lr), "right_rate": float(rr), "ratio": float(ratio)}


def block_bootstrap_ci(
    daily: np.ndarray | pd.Series,
    *,
    stat: str = "mean",
    block: int = 20,
    n_boot: int = 400,
    seed: int = 12345,
    alpha: float = 0.10,
) -> dict[str, float]:
    """Circular block bootstrap CI for the mean (or sum) of a daily series."""
    x = np.asarray(daily, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < max(8, block):
        return {"n": float(n), "point": float("nan"), "lo": float("nan"), "hi": float("nan")}
    point = float(np.mean(x) if stat == "mean" else np.sum(x))
    gen = np.random.default_rng(seed)
    b = max(1, int(block))
    n_blocks = int(math.ceil(n / b))
    starts = gen.integers(0, n, size=(n_boot, n_blocks))
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = np.concatenate([
            (starts[i, k] + np.arange(b)) % n for k in range(n_blocks)
        ])[:n]
        sample = x[idx]
        stats[i] = float(np.mean(sample) if stat == "mean" else np.sum(sample))
    lo = float(np.quantile(stats, alpha / 2.0))
    hi = float(np.quantile(stats, 1.0 - alpha / 2.0))
    return {"n": float(n), "point": point, "lo": lo, "hi": hi}


def paired_block_bootstrap_diff(
    left: np.ndarray,
    right: np.ndarray,
    *,
    block: int = 20,
    n_boot: int = 400,
    seed: int = 12345,
    alpha: float = 0.10,
) -> dict[str, float]:
    """CI for mean(left - right) with shared block draws. NaNs dropped pairwise."""
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    return block_bootstrap_ci(a[m] - b[m], stat="mean", block=block, n_boot=n_boot, seed=seed, alpha=alpha)


def position_cap(budget: float, extreme_drop: float) -> float:
    """Single-name cap = loss budget / assumed extreme drop (framework §12.2).

    ``budget`` and ``extreme_drop`` are positive fractions (0.01 and 0.50).
    """
    b = float(budget)
    d = abs(float(extreme_drop))
    if not (b > 0 and d > 0):
        return float("nan")
    return b / d


def dip_credit_allowed(
    alpha_now: float | None,
    alpha_prev: float | None,
) -> bool:
    """No dip-buy credit while the left tail is thickening (alpha falling)."""
    if alpha_now is None or alpha_prev is None:
        return False
    if not (math.isfinite(alpha_now) and math.isfinite(alpha_prev)):
        return False
    return not (alpha_now < alpha_prev)


def pnl_distribution(rets: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """Trade-level P&L summary. Replaces win-rate as the decision metric."""
    x = np.asarray(list(rets), dtype=float)
    x = x[np.isfinite(x)]
    out: dict[str, Any] = {"n": int(x.size)}
    if x.size == 0:
        return out
    mad, _ = compute_mad(x)
    order = np.sort(x)
    k = min(5, int(order.size))
    worst_k = order[:k]
    q = float(np.quantile(order, 0.05))
    tail = order[order <= q]
    out.update(
        {
            "median": float(np.median(x)),
            "mean": float(np.mean(x)),
            "mad": mad,
            "worst5_mean": float(np.mean(worst_k)),
            "worst5pct_mean": float(np.mean(tail)) if tail.size else float(order[0]),
            "max_loss": float(order[0]),
            "empirical_cvar05": float(np.mean(tail)) if tail.size else float("nan"),
        }
    )
    left = fit_tail(x, side="left", tail_quantile=float(PROTOCOL["pot_quantile"]), min_exceed=8)
    right = fit_tail(x, side="right", tail_quantile=float(PROTOCOL["pot_quantile"]), min_exceed=8)
    out["alpha_left"] = left.alpha
    out["alpha_right"] = right.alpha
    out["shadow_mean"] = shadow_mean(x, left, right)
    total = float(np.sum(x))
    if total > 0 and x.size >= 5:
        top = np.sort(x)[::-1]
        out["top1_share"] = float(top[0] / total)
        out["top3_share"] = float(np.sum(top[:3]) / total)
        out["top5_share"] = float(np.sum(top[:5]) / total)
    else:
        out["top1_share"] = out["top3_share"] = out["top5_share"] = None
    return out


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce")
    return c.shift(-int(horizon)) / c - 1.0


def g_sign_lift(
    g: pd.Series,
    fwd: pd.Series,
    *,
    block: int = 20,
    n_boot: int = 400,
    seed: int = 12345,
) -> dict[str, float]:
    """P(fwd>0 | g>0) - P(fwd>0), with a block-bootstrap CI on the difference."""
    gg = pd.to_numeric(g, errors="coerce").to_numpy(dtype=float)
    ff = pd.to_numeric(fwd, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(gg) & np.isfinite(ff)
    if int(ok.sum()) < 30:
        return {"n": float(ok.sum()), "lift": float("nan"), "lo": float("nan"), "hi": float("nan")}
    base = (ff[ok] > 0).astype(float)
    cond = np.where(gg[ok] > 0, base, np.nan)
    # Difference of means via paired series aligned on ok rows: indicator(g>0)*aligned trick.
    # Use E[1_{fwd>0}|g>0] - E[1_{fwd>0}] estimated by resampling rows.
    g_pos = gg[ok] > 0
    if int(g_pos.sum()) < 15 or int((~g_pos).sum()) < 15:
        return {"n": float(ok.sum()), "lift": float("nan"), "lo": float("nan"), "hi": float("nan")}
    point = float(base[g_pos].mean() - base.mean())
    gen = np.random.default_rng(seed)
    n = int(base.size)
    b = max(1, int(block))
    n_blocks = int(math.ceil(n / b))
    starts = gen.integers(0, n, size=(n_boot, n_blocks))
    offs = np.arange(b)
    idx = ((starts[:, :, None] + offs[None, None, :]) % n).reshape(n_boot, n_blocks * b)[:, :n]
    bb = base[idx]
    gp = g_pos[idx]
    counts = gp.sum(axis=1)
    means = np.divide(
        (bb * gp).sum(axis=1),
        np.maximum(counts, 1),
    )
    lifts = np.where(counts >= 5, means - bb.mean(axis=1), np.nan)
    finite = lifts[np.isfinite(lifts)]
    if finite.size < 20:
        return {"n": float(n), "lift": point, "lo": float("nan"), "hi": float("nan")}
    return {
        "n": float(n),
        "lift": point,
        "lo": float(np.quantile(finite, 0.05)),
        "hi": float(np.quantile(finite, 0.95)),
        "rows": float(n),
    }


def evaluate_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered stop rules in plan section 4.6.

    ``summary`` keys (all optional; missing counts as not established):
      h1_left_dominant_frac, h2_ewma_only_not_better, h2_inconclusive,
      h3_short_window_flip, h4_low_exceed_frac, h4_pool_possible
    """
    h1 = float(summary.get("h1_left_dominant_frac", 0.0) or 0.0) > float(PROTOCOL["h1_min_frac"])
    h2_inc = bool(summary.get("h2_inconclusive", True))
    h2 = (not h2_inc) and bool(summary.get("h2_ewma_only_not_better", False))
    h3 = bool(summary.get("h3_short_window_flip", False))
    low = summary.get("h4_low_exceed_frac")
    low_f = float(low) if low is not None else 1.0
    pool_ok = bool(summary.get("h4_pool_possible", False))
    change_rails = bool(h1 or h2 or h3)
    use_pot = bool(change_rails and (low_f <= float(PROTOCOL["h4_low_exceed_max"]) or pool_ok))
    if not change_rails:
        action = "docs_only"
    elif use_pot:
        action = "rails_split_and_pot"
    else:
        action = "rails_split_empirical_only"
    return {
        "h1": h1,
        "h2": h2,
        "h2_inconclusive": h2_inc,
        "h3": h3,
        "change_rails": change_rails,
        "use_pot": use_pot,
        "action": action,
    }


def acceptance_vs_legacy(
    legacy: Mapping[str, Any],
    challenger: Mapping[str, Any],
    *,
    independent_days: int,
    worst_tol: float = 0.0,
) -> dict[str, Any]:
    """Pre-registered A/B gates. ``worst_tol`` is the allowed worsening (≤0 means none)."""
    reasons: list[str] = []
    if int(independent_days) < int(PROTOCOL["min_independent_days"]):
        reasons.append("independent_days")
    for key in ("worst5pct_mean", "max_loss"):
        a = legacy.get(key)
        b = challenger.get(key)
        if a is None or b is None or not (math.isfinite(float(a)) and math.isfinite(float(b))):
            reasons.append(key)
            continue
        # More negative is worse. Challenger must be >= legacy + tol (tol ≤ 0).
        if float(b) < float(a) + float(worst_tol):
            reasons.append(key)
    for key in ("median", "shadow_mean"):
        diff_lo = challenger.get(f"{key}_diff_lo")
        if diff_lo is None or not math.isfinite(float(diff_lo)):
            reasons.append(f"{key}_ci")
            continue
        if float(diff_lo) < 0:
            # Interval extends below 0: not "significantly negative" is the
            # plan's bar only when the whole CI is negative. Fail when hi < 0.
            diff_hi = challenger.get(f"{key}_diff_hi")
            if diff_hi is not None and math.isfinite(float(diff_hi)) and float(diff_hi) < 0:
                reasons.append(key)
    return {"pass": not reasons, "reasons": reasons}


def date_block_mean_diff(
    dates_a: Sequence[pd.Timestamp] | pd.Series,
    rets_a: Sequence[float] | np.ndarray,
    dates_b: Sequence[pd.Timestamp] | pd.Series,
    rets_b: Sequence[float] | np.ndarray,
    *,
    block: int = 20,
    n_boot: int = 400,
    seed: int = 12345,
) -> dict[str, float]:
    """CI for mean(b) - mean(a), resampling contiguous calendar blocks.

    Trades that share an entry date stay together. A block that is drawn
    again duplicates every trade opened inside it.
    """
    a_dates = pd.to_datetime(pd.Series(list(dates_a)))
    b_dates = pd.to_datetime(pd.Series(list(dates_b)))
    a_ret = np.asarray(list(rets_a), dtype=float)
    b_ret = np.asarray(list(rets_b), dtype=float)
    if a_ret.size == 0 or b_ret.size == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan")}
    calendar = np.array(sorted(set(a_dates.tolist()) | set(b_dates.tolist())))
    n = int(calendar.size)
    if n < 8:
        diff = float(np.mean(b_ret) - np.mean(a_ret))
        return {"diff": diff, "lo": float("nan"), "hi": float("nan")}
    pos = {pd.Timestamp(d): i for i, d in enumerate(calendar)}
    a_idx = np.array([pos[pd.Timestamp(d)] for d in a_dates], dtype=int)
    b_idx = np.array([pos[pd.Timestamp(d)] for d in b_dates], dtype=int)
    point = float(np.mean(b_ret) - np.mean(a_ret))
    median_point = float(np.median(b_ret) - np.median(a_ret))
    gen = np.random.default_rng(seed)
    b = max(1, int(block))
    n_blocks = int(math.ceil(n / b))
    starts = gen.integers(0, n, size=(n_boot, n_blocks))
    offs = np.arange(b)
    drawn = ((starts[:, :, None] + offs[None, None, :]) % n).reshape(n_boot, -1)[:, :n]
    # Count how many times each calendar day is drawn, per bootstrap.
    diffs = np.empty(n_boot, dtype=float)
    median_diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        counts = np.bincount(drawn[i], minlength=n)
        wa = counts[a_idx].astype(float)
        wb = counts[b_idx].astype(float)
        if float(wa.sum()) <= 0 or float(wb.sum()) <= 0:
            diffs[i] = np.nan
            median_diffs[i] = np.nan
            continue
        diffs[i] = float(np.average(b_ret, weights=wb) - np.average(a_ret, weights=wa))
        median_diffs[i] = _weighted_median(b_ret, wb) - _weighted_median(a_ret, wa)
    finite = diffs[np.isfinite(diffs)]
    finite_med = median_diffs[np.isfinite(median_diffs)]
    if finite.size < 20:
        return {
            "diff": point,
            "lo": float("nan"),
            "hi": float("nan"),
            "median_diff": median_point,
            "median_lo": float("nan"),
            "median_hi": float("nan"),
        }
    return {
        "diff": point,
        "lo": float(np.quantile(finite, 0.05)),
        "hi": float(np.quantile(finite, 0.95)),
        "median_diff": median_point,
        "median_lo": float(np.quantile(finite_med, 0.05)) if finite_med.size >= 20 else float("nan"),
        "median_hi": float(np.quantile(finite_med, 0.95)) if finite_med.size >= 20 else float("nan"),
    }


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v = np.asarray(values, dtype=float)[order]
    w = np.asarray(weights, dtype=float)[order]
    cutoff = 0.5 * float(w.sum())
    return float(v[int(np.searchsorted(np.cumsum(w), cutoff, side="left"))])


def kappa_of(values: np.ndarray) -> float | None:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return compute_kappa(v, n=30, resamples=400, seed=int(PROTOCOL["boot_seed"]))
