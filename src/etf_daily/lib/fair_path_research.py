"""Research-only alternatives for the fig7–fig11 fair path / bands.

Addresses three documented limitations of the production implementation
(``trailing_log_trend_price_path`` + ``erp_path_symmetric_envelope``):

1. Trend lag — trailing OLS extrapolation lags at true trend reversals.
   ``kalman_trend_path`` fits a local-linear-trend state-space model whose
   filtered level adapts continuously instead of by window refit.
2. Expanding-window inertia — the causal expanding P95 of |ln(c/p)| stays
   wide long after a high-vol regime ends. ``vol_scaled_envelope``
   standardizes residuals by rv5 so bands tighten in calm regimes and
   widen in storms.
3. P95 != prediction interval — the band is an unconditional historical
   quantile, not a forward coverage statement. ``coverage_report``
   measures realized coverage (overall / per-year / per-vol-regime /
   post-touch return) so the gap is quantified instead of assumed.

Research only: production path/band logic and all trading signals are
untouched. All series here are causal (value at t uses data ≤ t only),
with one documented compromise: Kalman noise parameters are estimated
once on the first ``fit_bars`` observations and frozen afterwards.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

DEFAULT_ENVELOPE_Q = 0.95
DEFAULT_MIN_BARS = 63
DEFAULT_FIT_BARS = 504  # ~2 trading years
DEFAULT_Z_LOOKBACK = 504  # trailing |z| window for delta* (≈2y); 0 = expanding
LOOKBACK_3Y = 756  # ~3 trading years for rolling residual P95
LOOKBACK_5Y = 1260  # ~5 trading years
DEFAULT_EWMA_LAMBDA = 0.94  # RiskMetrics daily
DEFAULT_CLAMP_FLOOR_Q = 0.50  # collapse floor only; never paired with cap
DEFAULT_CLAMP_CAP_Q = 0.95  # calm |r| P95; applied only while over-wide
DEFAULT_CLAMP_COV_LOOKBACK = 60  # trailing emitted calm bars for the 98% detector
TRADING_DAYS_PER_YEAR = 252.0
# Causal expanding P80 of |ln(c/p)| used as a log-space width floor so
# calm-regime rails do not collapse below a typical (not median) historical
# deviation. P50 was too low on SH512530 (calm coverage stayed ~83%).
DEFAULT_FLOOR_Q = 0.80  # legacy global residual quantile; unused when vol_regime is set
DEFAULT_CALM_COVERAGE_Q = 0.95
DEFAULT_CALM_FLOOR_MIN_BARS = 20
DEFAULT_CALM_ONLINE = True
DEFAULT_CALM_ALPHA_MAX = 4.0  # numerical cap; α never shrinks, so unbound P95(scores) can explode
MAX_LOG_HALF_WIDTH = 5.0  # clip before exp; e^5 ≈ 148x, avoids overflow on tiny w_raw
CALM_SCORE_MIN_WIDTH = 1e-8  # skip conformal scores when w_raw is numerically empty
COVERAGE_PASS_LO = 0.90
COVERAGE_PASS_HI = 0.98
CALM_REGIME_LABEL = "波动率平稳"
NORMAL_REGIME_LABEL = "波动率正常"
CLUSTER_REGIME_LABEL = "波动率聚集"
EXTREME_REGIME_LABEL = "波动率极端聚集"
# Calm/normal keep an upper bound (over-wide bands in quiet regimes mean the
# channel is stale). Cluster/extreme have no upper bound: high coverage there
# is the goal of vol-scaling, not a failure.
REGIMES_WITH_HI = frozenset({CALM_REGIME_LABEL, NORMAL_REGIME_LABEL})
REGIMES_NO_HI = frozenset({CLUSTER_REGIME_LABEL, EXTREME_REGIME_LABEL})
B2_MODES = ("ols", "kalman", "both")


def residual_p95_envelope(
    path: pd.Series,
    close: pd.Series,
    *,
    q: float = DEFAULT_ENVELOPE_Q,
    min_bars: int = DEFAULT_MIN_BARS,
    lookback: int = 0,
) -> tuple[pd.Series, pd.Series, float]:
    """Causal P95 of ``|ln(close/path)|``; expanding or trailing.

    Same residual and rails as production ``erp_path_symmetric_envelope``:
    ``δ_t = quantile(|r_s|, q)``, ``upper = path·exp(δ)``. ``lookback=0``
    uses all history ``s ≤ t`` (expanding). ``lookback > 0`` uses only
    the last ``lookback`` valid residuals so a dead high-vol regime
    falls out of the window (the documented 3–5y rolling P95). Warmup
    still requires ``min_bars`` residuals in the active window.

    Research only: does not import or call the production envelope.
    """
    p = pd.to_numeric(path, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    upper = pd.Series(np.nan, index=p.index, dtype=float)
    lower = pd.Series(np.nan, index=p.index, dtype=float)
    qq = float(q)
    min_b = max(2, int(min_bars))
    lb = int(lookback)
    if p.empty or not np.isfinite(qq) or not (0.0 < qq < 1.0):
        return upper, lower, float("nan")
    path_arr = p.to_numpy(dtype=float)
    close_arr = c.to_numpy(dtype=float)
    n = len(path_arr)
    upper_arr = np.full(n, np.nan, dtype=float)
    lower_arr = np.full(n, np.nan, dtype=float)
    delta_arr = np.full(n, np.nan, dtype=float)
    abs_hist: list[float] = []
    for t in range(n):
        pt = float(path_arr[t])
        ct = float(close_arr[t])
        if np.isfinite(pt) and pt > 0 and np.isfinite(ct) and ct > 0:
            rt = float(np.log(ct / pt))
            if np.isfinite(rt):
                abs_hist.append(abs(rt))
        win = abs_hist if lb <= 0 else abs_hist[-lb:]
        if len(win) < min_b:
            continue
        if not (np.isfinite(pt) and pt > 0):
            continue
        delta_t = float(np.quantile(np.asarray(win, dtype=float), qq))
        if not np.isfinite(delta_t) or delta_t < 0:
            continue
        delta_arr[t] = delta_t
        upper_arr[t] = pt * float(np.exp(delta_t))
        lower_arr[t] = pt * float(np.exp(-delta_t))
    upper.iloc[:] = upper_arr
    lower.iloc[:] = lower_arr
    finite_d = delta_arr[np.isfinite(delta_arr)]
    delta_asof = float(finite_d[-1]) if finite_d.size else float("nan")
    return upper, lower, delta_asof


def ewma_residual_sigma(
    residual: pd.Series,
    *,
    lam: float = DEFAULT_EWMA_LAMBDA,
) -> pd.Series:
    """Causal EWMA std of path residuals; ``σ_t`` uses ``r_{t-1}`` only.

    ``σ²_t = λ σ²_{t-1} + (1-λ) r_{t-1}²``. First bar is NaN (no lag).
    The first valid lagged residual seeds ``σ²``; subsequent bars
    recurse. Today's ``r_t`` is stored for tomorrow and never enters
    ``σ_t``.
    """
    r = pd.to_numeric(residual, errors="coerce").to_numpy(dtype=float)
    n = r.size
    out = np.full(n, np.nan, dtype=float)
    lam_f = float(lam)
    if not (0.0 < lam_f < 1.0) or n == 0:
        return pd.Series(out, index=residual.index, dtype=float)
    one_l = 1.0 - lam_f
    last_var: float | None = None
    last_r2: float | None = None
    for t in range(n):
        if last_var is None:
            if last_r2 is not None and last_r2 > 0:
                last_var = last_r2
                out[t] = float(np.sqrt(last_var))
        else:
            if last_r2 is not None and np.isfinite(last_r2):
                last_var = lam_f * last_var + one_l * last_r2
            if last_var is not None and last_var > 0:
                out[t] = float(np.sqrt(last_var))
        rt = float(r[t])
        if np.isfinite(rt):
            last_r2 = rt * rt
    return pd.Series(out, index=residual.index, dtype=float)


def ewma_scaled_envelope(
    path: pd.Series,
    close: pd.Series,
    *,
    q: float = DEFAULT_ENVELOPE_Q,
    min_bars: int = DEFAULT_MIN_BARS,
    lam: float = DEFAULT_EWMA_LAMBDA,
) -> tuple[pd.Series, pd.Series, float]:
    """Causal envelope: expanding P95 of ``|r|/σ_ewma`` times current σ.

    Width tracks the current residual-vol cluster (forgets a dead storm
    as EWMA σ decays). No calm residual floor — that floor is what
    blocked trailing-|z| from tightening after a storm.
    """
    p = pd.to_numeric(path, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    upper = pd.Series(np.nan, index=p.index, dtype=float)
    lower = pd.Series(np.nan, index=p.index, dtype=float)
    qq = float(q)
    min_b = max(2, int(min_bars))
    if p.empty or not np.isfinite(qq) or not (0.0 < qq < 1.0):
        return upper, lower, float("nan")
    r = pd.Series(np.nan, index=p.index, dtype=float)
    ok = (p > 0) & (c > 0) & p.notna() & c.notna()
    r.loc[ok] = np.log(c.loc[ok].to_numpy(dtype=float) / p.loc[ok].to_numpy(dtype=float))
    sigma = ewma_residual_sigma(r, lam=lam)
    path_arr = p.to_numpy(dtype=float)
    r_arr = r.to_numpy(dtype=float)
    s_arr = sigma.to_numpy(dtype=float)
    n = len(path_arr)
    upper_arr = np.full(n, np.nan, dtype=float)
    lower_arr = np.full(n, np.nan, dtype=float)
    delta_arr = np.full(n, np.nan, dtype=float)
    abs_z: list[float] = []
    for t in range(n):
        pt = float(path_arr[t])
        rt = float(r_arr[t])
        st = float(s_arr[t])
        ok_p = np.isfinite(pt) and pt > 0
        ok_s = np.isfinite(st) and st > 0
        if ok_s and np.isfinite(rt):
            zt = rt / st
            if np.isfinite(zt):
                abs_z.append(abs(zt))
        if len(abs_z) < min_b or not (ok_p and ok_s):
            continue
        delta_t = float(np.quantile(np.asarray(abs_z, dtype=float), qq))
        if not np.isfinite(delta_t) or delta_t < 0:
            continue
        delta_arr[t] = delta_t
        width = delta_t * st
        upper_arr[t] = pt * float(np.exp(width))
        lower_arr[t] = pt * float(np.exp(-width))
    upper.iloc[:] = upper_arr
    lower.iloc[:] = lower_arr
    finite_d = delta_arr[np.isfinite(delta_arr)]
    delta_asof = float(finite_d[-1]) if finite_d.size else float("nan")
    return upper, lower, delta_asof


def ewma_clamped_envelope(
    path: pd.Series,
    close: pd.Series,
    *,
    q: float = DEFAULT_ENVELOPE_Q,
    min_bars: int = DEFAULT_MIN_BARS,
    lam: float = DEFAULT_EWMA_LAMBDA,
    vol_regime: pd.Series | None = None,
    calm_label: str = CALM_REGIME_LABEL,
    floor_q: float | None = DEFAULT_CLAMP_FLOOR_Q,
    cap_q: float = DEFAULT_CLAMP_CAP_Q,
    coverage_hi: float = COVERAGE_PASS_HI,
    coverage_lookback: int = DEFAULT_CLAMP_COV_LOOKBACK,
    calm_min_bars: int = DEFAULT_CALM_FLOOR_MIN_BARS,
) -> tuple[pd.Series, pd.Series, float]:
    """EWMA envelope; one-sided calm floor / over-wide cap, never both.

    Storm / cluster / normal days keep ``w = δ*_ewma · σ_ewma``. On calm
    days, after ≥ ``calm_min_bars`` calm residuals:

    - **Cap** (only while *recent* emitted calm coverage exceeds
      ``coverage_hi``, default 98% over the last ``coverage_lookback``
      calm bars, ``s < t``): ``w = min(w, P_cap)`` of expanding calm
      ``|ln(c/p)|`` (default P95). Trailing, not lifetime, so a post-storm
      100% streak is not diluted by older ~95% days. Idle when recent
      coverage is already inside the gate.
    - **Floor** (collapse only, and never on a bar that is being capped):
      ``w = max(w, P50)``. Floor and cap are mutually exclusive so they
      cannot sandwich the band.

    ``vol_regime is None`` → identical to :func:`ewma_scaled_envelope`.
    """
    p = pd.to_numeric(path, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    upper = pd.Series(np.nan, index=p.index, dtype=float)
    lower = pd.Series(np.nan, index=p.index, dtype=float)
    qq = float(q)
    fq = None if floor_q is None else float(floor_q)
    cq = float(cap_q)
    hi = float(coverage_hi)
    cov_lb = int(coverage_lookback)
    min_b = max(2, int(min_bars))
    min_calm = max(2, int(calm_min_bars))
    if p.empty or not np.isfinite(qq) or not (0.0 < qq < 1.0):
        return upper, lower, float("nan")
    if fq is not None and not (0.0 < fq < 1.0):
        raise ValueError(f"floor_q must be in (0, 1), got {floor_q}")
    if not (0.0 < cq < 1.0):
        raise ValueError(f"cap_q must be in (0, 1), got {cap_q}")
    if not (0.0 < hi <= 1.0):
        raise ValueError(f"coverage_hi must be in (0, 1], got {coverage_hi}")
    if fq is not None and fq > cq:
        raise ValueError(f"floor_q ({fq}) must be <= cap_q ({cq})")
    r = pd.Series(np.nan, index=p.index, dtype=float)
    ok = (p > 0) & (c > 0) & p.notna() & c.notna()
    r.loc[ok] = np.log(c.loc[ok].to_numpy(dtype=float) / p.loc[ok].to_numpy(dtype=float))
    sigma = ewma_residual_sigma(r, lam=lam)
    if vol_regime is None:
        regime_arr = np.array([""] * len(p), dtype=object)
        use_clamp = False
    else:
        regime_arr = (
            pd.Series(vol_regime).reindex(p.index).astype(str).fillna("").to_numpy()
        )
        use_clamp = True
    path_arr = p.to_numpy(dtype=float)
    r_arr = r.to_numpy(dtype=float)
    s_arr = sigma.to_numpy(dtype=float)
    n = len(path_arr)
    upper_arr = np.full(n, np.nan, dtype=float)
    lower_arr = np.full(n, np.nan, dtype=float)
    delta_arr = np.full(n, np.nan, dtype=float)
    abs_z: list[float] = []
    abs_r_calm: list[float] = []
    calm_inside: list[float] = []  # emitted calm coverage, s < t
    for t in range(n):
        pt = float(path_arr[t])
        rt = float(r_arr[t])
        st = float(s_arr[t])
        ok_p = np.isfinite(pt) and pt > 0
        ok_s = np.isfinite(st) and st > 0
        is_calm = use_clamp and str(regime_arr[t]) == str(calm_label)
        if np.isfinite(rt):
            if is_calm:
                abs_r_calm.append(abs(rt))
            if ok_s:
                zt = rt / st
                if np.isfinite(zt):
                    abs_z.append(abs(zt))
        if len(abs_z) < min_b or not (ok_p and ok_s):
            continue
        delta_t = float(np.quantile(np.asarray(abs_z, dtype=float), qq))
        if not np.isfinite(delta_t) or delta_t < 0:
            continue
        delta_arr[t] = delta_t
        width = delta_t * st
        if is_calm and len(abs_r_calm) >= min_calm:
            calm_arr = np.asarray(abs_r_calm, dtype=float)
            recent = (
                calm_inside[-cov_lb:] if cov_lb > 0 else calm_inside
            )
            past_cov = (
                float(np.mean(recent))
                if len(recent) >= min_calm
                else float("nan")
            )
            over_wide = bool(np.isfinite(past_cov) and past_cov > hi)
            if over_wide:
                cap_w = float(np.quantile(calm_arr, cq))
                if np.isfinite(cap_w) and cap_w >= 0:
                    width = min(width, cap_w)
            elif fq is not None:
                floor_w = float(np.quantile(calm_arr, fq))
                if np.isfinite(floor_w) and floor_w >= 0:
                    width = max(width, floor_w)
        upper_arr[t] = pt * float(np.exp(width))
        lower_arr[t] = pt * float(np.exp(-width))
        if is_calm and np.isfinite(rt):
            calm_inside.append(1.0 if abs(rt) <= width else 0.0)
    upper.iloc[:] = upper_arr
    lower.iloc[:] = lower_arr
    finite_d = delta_arr[np.isfinite(delta_arr)]
    delta_asof = float(finite_d[-1]) if finite_d.size else float("nan")
    return upper, lower, delta_asof


def vol_scaled_envelope(
    path: pd.Series,
    close: pd.Series,
    sigma: pd.Series,
    *,
    q: float = DEFAULT_ENVELOPE_Q,
    min_bars: int = DEFAULT_MIN_BARS,
    floor_q: float | None = None,
    abs_floor: float | None = None,
    vol_regime: pd.Series | None = None,
    calm_label: str = CALM_REGIME_LABEL,
    calm_coverage_q: float = DEFAULT_CALM_COVERAGE_Q,
    calm_floor_min_bars: int = DEFAULT_CALM_FLOOR_MIN_BARS,
    z_lookback: int = 0,
    calm_online: bool | None = None,
) -> tuple[pd.Series, pd.Series, float]:
    """Causal expanding P95 envelope on vol-standardized residuals.

    Residuals are ``z_s = ln(close_s / path_s) / sigma_s`` for valid
    ``s <= t``; ``delta*_t = quantile(|z|, q)`` over either all history
    (expanding, ``z_lookback=0``) or the last ``z_lookback`` valid z's
    (trailing, e.g. ~504 bars = 2y). A trailing window lets ``delta*``
    forget a dead high-vol regime so bands tighten once the storm ends;
    the calm floor below keeps them from collapsing. Half-width in log
    space is ``w_t = max(delta*_t * sigma_t, floor_t)``.

    Floor (first match wins, can stack):

    - **Per-symbol causal calm pin** (preferred): when ``vol_regime`` is
      given, ``floor_t`` is the expanding ``calm_coverage_q`` (default
      0.95) of ``|ln(c/p)|`` on calm days ``s <= t`` only. Each ticker
      gets its own floor from its own history; no shared ``floor_q``.
    - Legacy ``floor_q``: expanding quantile of all-history ``|r|``.
    - Constant ``abs_floor``.

    Storm width still comes from ``delta*σ``. Calm days cannot collapse
    below that ticker's own causal calm P95 residual.

    **Online calm coverage pin** (``calm_online``, default True when
    ``vol_regime`` is set): after the raw width is formed, collect
    conformal scores ``|r_s| / w_raw_s`` on past calm days ``s < t``
    only. Inflation runs only while past *emitted* calm coverage is
    below ``calm_coverage_q`` (already-on-target tickers stay put, so
    the [90%, 98%] upper bound is not ratcheted through). Then
    ``alpha_t = min(alpha_max, max(1, quantile(scores, q)))``. On a
    calm day, ``w_t = alpha_t * w_raw_t``; storm days keep ``w_raw``.
    Log half-width is clipped at ``MAX_LOG_HALF_WIDTH`` before ``exp``.

    Warmup: fewer than ``min_bars`` valid standardized residuals, or a
    missing/non-positive ``sigma_t``, yields NaN rails that day.
    Returns ``(upper, lower, delta_asof)`` where ``delta_asof`` is the
    last finite ``delta*_t`` (standardized units, for legend/hover).
    """
    p = pd.to_numeric(path, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    s = pd.to_numeric(sigma, errors="coerce").reindex(p.index)
    upper = pd.Series(np.nan, index=p.index, dtype=float)
    lower = pd.Series(np.nan, index=p.index, dtype=float)
    qq = float(q)
    min_b = max(2, int(min_bars))
    fq = None if floor_q is None else float(floor_q)
    af = None if abs_floor is None else float(abs_floor)
    cq = float(calm_coverage_q)
    min_calm = max(2, int(calm_floor_min_bars))
    if p.empty or not np.isfinite(qq) or not (0.0 < qq < 1.0):
        return upper, lower, float("nan")
    if fq is not None and not (0.0 < fq < 1.0):
        raise ValueError(f"floor_q must be in (0, 1), got {floor_q}")
    if not (0.0 < cq < 1.0):
        raise ValueError(f"calm_coverage_q must be in (0, 1), got {calm_coverage_q}")
    path_arr = p.to_numpy(dtype=float)
    close_arr = c.to_numpy(dtype=float)
    sig_arr = s.to_numpy(dtype=float)
    zlb = int(z_lookback)
    if vol_regime is None:
        regime_arr = np.array([""] * len(p), dtype=object)
        use_calm = False
    else:
        regime_arr = (
            pd.Series(vol_regime).reindex(p.index).astype(str).fillna("").to_numpy()
        )
        use_calm = True
    do_online = bool(use_calm if calm_online is None else calm_online)
    n = len(path_arr)
    upper_arr = np.full(n, np.nan, dtype=float)
    lower_arr = np.full(n, np.nan, dtype=float)
    delta_arr = np.full(n, np.nan, dtype=float)
    abs_z_hist: list[float] = []
    abs_r_hist: list[float] = []
    abs_r_calm: list[float] = []
    calm_scores: list[float] = []  # |r|/w_raw on past calm days, s < t
    calm_inside: list[float] = []  # 1 if |r_s| <= emitted w_s, s < t
    alpha_max = float(DEFAULT_CALM_ALPHA_MAX)
    for t in range(n):
        pt = float(path_arr[t])
        ct = float(close_arr[t])
        st = float(sig_arr[t])
        ok_p = np.isfinite(pt) and pt > 0
        ok_c = np.isfinite(ct) and ct > 0
        ok_s = np.isfinite(st) and st > 0
        rt = float("nan")
        if ok_p and ok_c:
            rt = float(np.log(ct / pt))
            if np.isfinite(rt):
                abs_r_hist.append(abs(rt))
                if use_calm and str(regime_arr[t]) == str(calm_label):
                    abs_r_calm.append(abs(rt))
                if ok_s:
                    zt = rt / st
                    if np.isfinite(zt):
                        abs_z_hist.append(abs(zt))
        if len(abs_z_hist) < min_b or not (ok_p and ok_s):
            continue
        z_win = abs_z_hist if zlb <= 0 else abs_z_hist[-zlb:]
        delta_t = float(np.quantile(np.asarray(z_win, dtype=float), qq))
        if not np.isfinite(delta_t) or delta_t < 0:
            continue
        delta_arr[t] = delta_t
        width = delta_t * st
        if use_calm and len(abs_r_calm) >= min_calm:
            floor_t = float(np.quantile(np.asarray(abs_r_calm, dtype=float), cq))
            if np.isfinite(floor_t):
                width = max(width, floor_t)
        elif fq is not None and abs_r_hist:
            floor_t = float(np.quantile(np.asarray(abs_r_hist, dtype=float), fq))
            if np.isfinite(floor_t):
                width = max(width, floor_t)
        if af is not None and np.isfinite(af) and af > 0:
            width = max(width, af)
        width_raw = width
        is_calm = use_calm and str(regime_arr[t]) == str(calm_label)
        # Causal online pin: inflate calm width only while past emitted
        # coverage is below q. Storm days keep the raw width.
        # Scores are |r_s| / w_raw_s on emitted calm bars s < t only;
        # today's score is appended after alpha_t is applied.
        if (
            do_online
            and is_calm
            and len(calm_scores) >= min_calm
            and len(calm_inside) >= min_calm
        ):
            realized = float(np.mean(calm_inside))
            if np.isfinite(realized) and realized < cq:
                alpha = float(np.quantile(np.asarray(calm_scores, dtype=float), cq))
                if np.isfinite(alpha):
                    width = width_raw * min(alpha_max, max(1.0, alpha))
        if np.isfinite(width):
            width = min(float(width), MAX_LOG_HALF_WIDTH)
        upper_arr[t] = pt * float(np.exp(width))
        lower_arr[t] = pt * float(np.exp(-width))
        if do_online and is_calm and width_raw >= CALM_SCORE_MIN_WIDTH and np.isfinite(rt):
            calm_scores.append(abs(rt) / width_raw)
            calm_inside.append(1.0 if abs(rt) <= width else 0.0)
    upper.iloc[:] = upper_arr
    lower.iloc[:] = lower_arr
    finite_d = delta_arr[np.isfinite(delta_arr)]
    delta_asof = float(finite_d[-1]) if finite_d.size else float("nan")
    return upper, lower, delta_asof


def _estimate_llt_noises(dy: np.ndarray) -> tuple[float, float, float]:
    """Moment-based noise variances for a local-linear-trend model.

    For y_t = l_t + e_t, l_t = l_{t-1} + b_{t-1} + n_l, b_t = b_{t-1} + n_b:
    ``dy_t = b + n_l + e_t - e_{t-1}`` implies cov(dy_t, dy_{t-1}) ~= -r
    and var(dy) ~= q_b + q_l + 2r on detrended increments. This is a
    heuristic (b is a random walk, so moments are not stationary); it is
    a documented research compromise that avoids full-sample MLE leakage.
    Returns ``(r_obs, q_level, q_slope)``.
    """
    dy = np.asarray(dy, dtype=float)
    dy = dy[np.isfinite(dy)]
    if dy.size < 20:
        return 1e-4, 1e-5, 1e-8
    dyc = dy - float(np.mean(dy))
    acov1 = float(np.mean(dyc[1:] * dyc[:-1]))
    r_obs = max(-acov1, 1e-10)
    var_dy = float(np.var(dyc))
    q_level = max(var_dy - 2.0 * r_obs, 1e-10)
    ddy = np.diff(dyc)
    q_slope = max(float(np.var(ddy)) - 2.0 * q_level - 6.0 * r_obs, 1e-12)
    return r_obs, q_level, q_slope


def kalman_trend_path(
    close: pd.Series,
    *,
    fit_bars: int = DEFAULT_FIT_BARS,
    trading_days_per_year: float = TRADING_DAYS_PER_YEAR,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Causal local-linear-trend Kalman filter on log(close).

    State x = [level, slope], F = [[1, 1], [0, 1]], H = [1, 0].
    Noise variances are estimated ONCE from increments of the first
    ``fit_bars`` valid observations via :func:`_estimate_llt_noises` and
    frozen; the filter then runs forward causally (no smoother, so no
    future data enters any estimate).

    Returns ``(path, g, pred_std, pred_mean_log)`` where
    ``path[t] = exp(level_{t|t})``, ``g[t] = exp(slope_{t|t} * tdy) - 1``
    (annualized drift), and ``pred_mean_log`` / ``pred_std`` are the
    one-step-ahead forecast mean and std of log close at t given data
    <= t-1 (NaN at the first valid bar). Forecast bands must be centered
    on ``pred_mean_log`` — centering on the filtered level would leak
    the current observation and fake ~100% coverage.
    """
    c = pd.to_numeric(close, errors="coerce").astype(float)
    carr = c.to_numpy(dtype=float)
    n = carr.shape[0]
    y = np.full(n, np.nan, dtype=float)
    pos = np.isfinite(carr) & (carr > 0)
    y[pos] = np.log(carr[pos])
    path = pd.Series(np.nan, index=c.index, dtype=float)
    g = pd.Series(np.nan, index=c.index, dtype=float)
    pred_std = pd.Series(np.nan, index=c.index, dtype=float)
    pred_mean = pd.Series(np.nan, index=c.index, dtype=float)
    tdy = float(trading_days_per_year)
    valid_idx = np.flatnonzero(np.isfinite(y))
    if valid_idx.size < 3 or not np.isfinite(tdy) or tdy <= 0:
        return path, g, pred_std, pred_mean

    fit_n = min(max(int(fit_bars), 3), valid_idx.size)
    dy_fit = np.diff(y[valid_idx[:fit_n]])
    r_obs, q_level, q_slope = _estimate_llt_noises(dy_fit)

    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.diag([q_level, q_slope])
    I2 = np.eye(2)

    f0 = int(valid_idx[0])
    slope0 = float(np.mean(dy_fit)) if dy_fit.size else 0.0
    x = np.array([y[f0], slope0])
    P = np.diag([4.0 * r_obs, max(10.0 * q_slope, 1e-8)])
    path.iloc[f0] = float(np.exp(x[0]))
    g.iloc[f0] = float(np.exp(x[1] * tdy) - 1.0)

    path_arr = path.to_numpy(dtype=float)
    g_arr = g.to_numpy(dtype=float)
    std_arr = pred_std.to_numpy(dtype=float)
    mean_arr = pred_mean.to_numpy(dtype=float)
    for t in range(f0 + 1, n):
        x = F @ x
        P = F @ P @ F.T + Q
        mean_arr[t] = float(x[0])
        std_arr[t] = float(np.sqrt(max(P[0, 0] + r_obs, 0.0)))
        if np.isfinite(y[t]):
            innov = y[t] - float((H @ x)[0])
            s_t = float((H @ P @ H.T)[0, 0] + r_obs)
            K = (P @ H.T) / s_t
            x = x + K[:, 0] * innov
            P = (I2 - np.outer(K[:, 0], H)) @ P
        path_arr[t] = float(np.exp(x[0]))
        g_arr[t] = float(np.exp(x[1] * tdy) - 1.0)
    path.iloc[:] = path_arr
    g.iloc[:] = g_arr
    pred_std.iloc[:] = std_arr
    pred_mean.iloc[:] = mean_arr
    return path, g, pred_std, pred_mean


def kalman_forecast_band(
    pred_mean_log: pd.Series,
    pred_std: pd.Series,
    *,
    z: float = 1.6448536269514722,
) -> tuple[pd.Series, pd.Series]:
    """One-step-ahead predictive band from Kalman forecast stats.

    ``upper/lower = exp(pred_mean_log ± z * pred_std)`` (z=1.645 ~ 90%
    Gaussian interval on log close), centered on the pre-observation
    forecast mean so realized coverage is measurable. Unlike quantile
    envelopes this is a genuine model-based forward interval, though it
    inherits Gaussian + frozen-parameter assumptions.
    """
    mu = pd.to_numeric(pred_mean_log, errors="coerce")
    s = pd.to_numeric(pred_std, errors="coerce").reindex(mu.index)
    mu_arr = mu.to_numpy(dtype=float)
    s_arr = s.to_numpy(dtype=float)
    ok = np.isfinite(mu_arr) & np.isfinite(s_arr) & (s_arr > 0)
    upper = pd.Series(np.where(ok, np.exp(mu_arr + float(z) * s_arr), np.nan), index=mu.index)
    lower = pd.Series(np.where(ok, np.exp(mu_arr - float(z) * s_arr), np.nan), index=mu.index)
    return upper, lower


def coverage_report(
    close: pd.Series,
    bands: Mapping[str, tuple[pd.Series, pd.Series]],
    *,
    vol_regime: pd.Series | None = None,
    horizons: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """Realized-coverage diagnostics for named (upper, lower) band sets.

    For each band set: overall coverage (share of valid days with
    ``lower <= close <= upper``), per-calendar-year coverage, per
    ``vol_regime`` coverage (when provided), and post-touch return rate:
    among days closing above the upper rail (touch_hi) or below the
    lower rail (touch_lo), the share back inside the band within h days.

    Returns a tidy DataFrame with columns
    ``band, dimension, bucket, horizon, n, coverage``.
    """
    c = pd.to_numeric(close, errors="coerce").astype(float)
    n_rows: list[dict] = []
    if not isinstance(c.index, pd.DatetimeIndex):
        c = c.copy()
        c.index = pd.DatetimeIndex(pd.to_datetime(c.index))
    regime = None
    if vol_regime is not None:
        regime = pd.Series(vol_regime).reindex(c.index).astype(str)

    c_arr = c.to_numpy(dtype=float)
    for name, pair in bands.items():
        up = pd.to_numeric(pair[0], errors="coerce").reindex(c.index)
        lo = pd.to_numeric(pair[1], errors="coerce").reindex(c.index)
        up_arr = up.to_numpy(dtype=float)
        lo_arr = lo.to_numpy(dtype=float)
        valid = np.isfinite(up_arr) & np.isfinite(lo_arr) & np.isfinite(c_arr)
        inside = valid & (c_arr <= up_arr) & (c_arr >= lo_arr)
        n_valid = int(valid.sum())
        cov = float(inside[valid].mean()) if n_valid else float("nan")
        n_rows.append(
            dict(band=name, dimension="overall", bucket="all",
                 horizon=np.nan, n=n_valid, coverage=cov)
        )
        # Per calendar year.
        years = c.index.year.to_numpy()
        for yr in sorted({int(y) for y in years[valid]}):
            m = valid & (years == yr)
            n_m = int(m.sum())
            n_rows.append(
                dict(band=name, dimension="year", bucket=str(yr),
                     horizon=np.nan, n=n_m,
                     coverage=float(inside[m].mean()) if n_m else float("nan"))
            )
        # Per vol regime.
        if regime is not None:
            rg = regime.to_numpy()
            for val in sorted({str(v) for v in rg[valid]}):
                m = valid & (rg == val)
                n_m = int(m.sum())
                n_rows.append(
                    dict(band=name, dimension="vol_regime", bucket=val,
                         horizon=np.nan, n=n_m,
                         coverage=float(inside[m].mean()) if n_m else float("nan"))
                )
        # Post-touch return-to-band rate.
        touch = {
            "touch_hi": valid & (c_arr > up_arr),
            "touch_lo": valid & (c_arr < lo_arr),
        }
        n_tot = len(c_arr)
        for side, tmask in touch.items():
            t_idx = np.flatnonzero(tmask)
            for h in horizons:
                h = int(h)
                if h <= 0 or t_idx.size == 0:
                    continue
                returned = 0
                evaluated = 0
                for t0 in t_idx:
                    end = min(t0 + h, n_tot - 1)
                    if end <= t0:
                        continue  # not enough forward data to judge
                    evaluated += 1
                    if bool(inside[t0 + 1 : end + 1].any()):
                        returned += 1
                n_rows.append(
                    dict(band=name, dimension=side, bucket=f"return_{h}d",
                         horizon=h, n=evaluated,
                         coverage=(returned / evaluated) if evaluated else float("nan"))
                )
    return pd.DataFrame(
        n_rows,
        columns=["band", "dimension", "bucket", "horizon", "n", "coverage"],
    )


def regime_pass_table(
    coverage: pd.DataFrame,
    *,
    lo: float = COVERAGE_PASS_LO,
    hi: float = COVERAGE_PASS_HI,
    min_n: int = 20,
) -> pd.DataFrame:
    """Flag each (band, vol_regime) row against the coverage gate.

    Calm/normal regimes must sit in ``[lo, hi]`` (an over-wide quiet-
    regime channel is a stale, meaningless ruler). Cluster/extreme only
    require ``coverage >= lo``: high storm coverage is the point of
    vol-scaling and must not be penalized by an upper bound.
    """
    sub = coverage[coverage["dimension"] == "vol_regime"].copy()
    if sub.empty:
        return sub
    n = pd.to_numeric(sub["n"], errors="coerce")
    cov = pd.to_numeric(sub["coverage"], errors="coerce")
    bucket = sub["bucket"].astype(str)
    eligible = n >= int(min_n)
    has_hi = bucket.isin(REGIMES_WITH_HI)
    ok = cov >= float(lo)
    ok_hi = ~has_hi | (cov <= float(hi))
    sub["pass"] = eligible & ok & ok_hi
    sub["eligible"] = eligible
    sub["hi_applies"] = has_hi
    return sub


def build_research_panels(
    close: pd.Series,
    *,
    floor_q: float | None = None,
    vol_regime: pd.Series | None = None,
    calm_coverage_q: float = DEFAULT_CALM_COVERAGE_Q,
    z_lookback: int = 0,
    calm_online: bool | None = None,
) -> dict[str, pd.DataFrame]:
    """Per-panel production path/P95 + vol-scaled envelope.

    When ``vol_regime`` is omitted, it is computed causally from
    ``close`` via ``compute_volatility_regime_frame``. The research
    envelope then uses that ticker's expanding calm-residual P95 as
    the width floor (not a pool-wide ``floor_q``), then (default)
    a causal online scale ``alpha_t`` that pins realized calm
    coverage. ``z_lookback`` > 0 makes ``delta*`` a trailing-window
    quantile so it forgets a dead high-vol regime.

    Imports production OLS/envelope via ``fair_path_band_signals``.
    """
    from etf_daily.lib.fair_path_band_signals import FIG_PANELS, _load_plot_mod
    from etf_daily.lib.regime_transition_plot import (
        compute_realized_vol,
        compute_volatility_regime_frame,
    )

    pm = _load_plot_mod()
    rv5 = compute_realized_vol(close)
    if vol_regime is None:
        vol = compute_volatility_regime_frame(close)
        vol_regime = vol.set_index("as_of")["vol_regime"].reindex(close.index)
    panels: dict[str, pd.DataFrame] = {}
    for key, years, label in FIG_PANELS:
        path, g = pm.trailing_log_trend_price_path(
            close, trail_years=float(years), min_bars=pm._trail_fit_min_bars(float(years))
        )
        upper, lower, delta = pm.erp_path_symmetric_envelope(path, close)
        vs_upper, vs_lower, vs_delta = vol_scaled_envelope(
            path,
            close,
            rv5,
            floor_q=floor_q,
            vol_regime=vol_regime,
            calm_coverage_q=calm_coverage_q,
            z_lookback=z_lookback,
            calm_online=calm_online,
        )
        frame = pd.DataFrame(
            {
                "path": path,
                "g": g,
                "upper": upper,
                "lower": lower,
                "vs_upper": vs_upper,
                "vs_lower": vs_lower,
            },
            index=close.index,
        )
        frame.attrs["label"] = label
        frame.attrs["delta"] = delta
        frame.attrs["vs_delta"] = vs_delta
        panels[key] = frame
    return panels


def build_inertia_panels(close: pd.Series) -> dict[str, pd.DataFrame]:
    """Per-panel OLS path + expanding / roll3y / roll5y / EWMA / clamp envelopes.

    Isolated inertia experiment: same production OLS path. Clamp uses
    the causal ``vol_regime`` from ``compute_volatility_regime_frame``
    (P95 cap only while recent emitted calm coverage > 98%; P50 floor
    only when not capping). Imports production OLS only.
    """
    from etf_daily.lib.fair_path_band_signals import FIG_PANELS, _load_plot_mod
    from etf_daily.lib.regime_transition_plot import compute_volatility_regime_frame

    pm = _load_plot_mod()
    vol = compute_volatility_regime_frame(close)
    regime_s = vol.set_index("as_of")["vol_regime"].reindex(close.index)
    panels: dict[str, pd.DataFrame] = {}
    for key, years, label in FIG_PANELS:
        path, g = pm.trailing_log_trend_price_path(
            close, trail_years=float(years), min_bars=pm._trail_fit_min_bars(float(years))
        )
        u0, l0, d0 = residual_p95_envelope(path, close, lookback=0)
        u3, l3, d3 = residual_p95_envelope(path, close, lookback=LOOKBACK_3Y)
        u5, l5, d5 = residual_p95_envelope(path, close, lookback=LOOKBACK_5Y)
        ue, le, de = ewma_scaled_envelope(path, close)
        uc, lc, dc = ewma_clamped_envelope(path, close, vol_regime=regime_s)
        frame = pd.DataFrame(
            {
                "path": path,
                "g": g,
                "upper": u0,
                "lower": l0,
                "roll3y_upper": u3,
                "roll3y_lower": l3,
                "roll5y_upper": u5,
                "roll5y_lower": l5,
                "ewma_upper": ue,
                "ewma_lower": le,
                "ewma_clamp_upper": uc,
                "ewma_clamp_lower": lc,
            },
            index=close.index,
        )
        frame.attrs["label"] = label
        frame.attrs["delta"] = d0
        frame.attrs["roll3y_delta"] = d3
        frame.attrs["roll5y_delta"] = d5
        frame.attrs["ewma_delta"] = de
        frame.attrs["ewma_clamp_delta"] = dc
        panels[key] = frame
    return panels


def overlay_ewma_rails(
    ms: pd.DataFrame,
    close: pd.Series,
    *,
    min_bars: int = DEFAULT_MIN_BARS,
) -> pd.DataFrame:
    """Copy of ``ms`` with fig7–11 rails swapped to ``ewma_scaled_envelope``.

    Path, ``g``, gap, and ``above_path`` stay production OLS. Only upper/lower
    and touch flags change, so B1/S1 see the adaptive ruler while B2 still
    uses the long OLS trend. Research only: does not touch production HTML.
    """
    from etf_daily.lib.fair_path_band_signals import FIG_PANELS

    out = ms.copy()
    px = pd.to_numeric(close, errors="coerce").reindex(out.index)
    for key, _, _ in FIG_PANELS:
        path_col = f"{key}_path"
        if path_col not in out.columns:
            continue
        path = pd.to_numeric(out[path_col], errors="coerce")
        upper, lower, _delta = ewma_scaled_envelope(path, px, min_bars=int(min_bars))
        out[f"{key}_upper"] = upper
        out[f"{key}_lower"] = lower
        out[f"{key}_touch_lo"] = px <= lower
        out[f"{key}_touch_hi"] = px >= upper
    return out


def apply_research_bands_to_ms(
    ms: pd.DataFrame,
    panels: Mapping[str, pd.DataFrame],
    close: pd.Series,
) -> pd.DataFrame:
    """Copy of ``ms`` with fig9/fig10 upper/lower swapped to vol-scaled rails.

    Path and ``g`` stay production (B2/B3 path-rising still uses OLS).
    Touch flags and ``above_path`` are recomputed from the new rails.
    """
    out = ms.copy()
    px = pd.to_numeric(close, errors="coerce").reindex(out.index)
    for key in ("fig9", "fig10"):
        if key not in panels:
            continue
        pf = panels[key]
        out[f"{key}_upper"] = pf["vs_upper"].reindex(out.index)
        out[f"{key}_lower"] = pf["vs_lower"].reindex(out.index)
        lo = out[f"{key}_lower"]
        hi = out[f"{key}_upper"]
        out[f"{key}_touch_lo"] = px <= lo
        out[f"{key}_touch_hi"] = px >= hi
    return out


def kalman_long_ok(g: pd.Series, *, window: int = 20) -> pd.Series:
    """B2-style long-side OK from Kalman annualized drift.

    True when ``g > 0`` or the filtered path rose over ``window`` bars —
    the Kalman analogue of production B2 (fig7/fig8 OLS g>0 or rising).
    """
    gg = pd.to_numeric(g, errors="coerce")
    rising = gg > gg.shift(int(window))
    return ((gg > 0) | rising).fillna(False).astype(bool)


def b1_entries_variant(
    ms: pd.DataFrame,
    *,
    touch_lo: pd.Series,
    kalman_g: pd.Series | None = None,
    b2_mode: str = "ols",
    vol_regime: pd.Series | None = None,
    block_extreme: bool = False,
    deep_gap: float = -0.15,
) -> pd.Series:
    """B1 ablation: touch fig9/fig10 lower OR deep fig8 discount, with B2 choice.

    ``touch_lo`` is the lower-band touch from whichever rails the caller
    picked (production or research). ``b2_mode`` picks the long-side
    filter: ``ols`` (production fig7/fig8 g>0/path rising), ``kalman``
    (``kalman_long_ok``), or ``both``. ``block_extreme`` drops signals
    on 极端聚集 days, the falling-knife regime.
    """
    from etf_daily.lib.fair_path_band_signals import enrich_frame_for_checklist

    if b2_mode not in B2_MODES:
        raise ValueError(f"b2_mode must be one of {B2_MODES}, got {b2_mode}")
    frame = enrich_frame_for_checklist(ms.copy())
    touch = pd.to_numeric(touch_lo, errors="coerce").reindex(frame.index).fillna(0) > 0
    gap8 = pd.to_numeric(frame.get("fig8_gap"), errors="coerce")
    deep_disc = gap8 <= float(deep_gap)
    b1 = (touch | deep_disc).fillna(False).astype(bool)

    def _gpos(col: str) -> pd.Series:
        return (pd.to_numeric(frame.get(col), errors="coerce") > 0).fillna(False)

    def _rising(col: str) -> pd.Series:
        return frame.get(col, pd.Series(False, index=frame.index)).fillna(False).astype(bool)

    b2_ols = (
        _gpos("fig7_g") | _gpos("fig8_g") | _rising("fig7_path_rising") | _rising("fig8_path_rising")
    )
    if b2_mode == "ols":
        b2 = b2_ols
    else:
        if kalman_g is None:
            raise ValueError("kalman_g required for b2_mode != 'ols'")
        b2_k = kalman_long_ok(
            pd.to_numeric(kalman_g, errors="coerce").reindex(frame.index)
        )
        b2 = b2_k if b2_mode == "kalman" else (b2_ols & b2_k)
    out = (b1 & b2).fillna(False).astype(bool)
    if block_extreme and vol_regime is not None:
        rg = pd.Series(vol_regime).reindex(frame.index).astype(str)
        out = out & (rg != EXTREME_REGIME_LABEL)
    return out.fillna(False).astype(bool)


__all__ = [
    "B2_MODES",
    "COVERAGE_PASS_HI",
    "COVERAGE_PASS_LO",
    "CALM_REGIME_LABEL",
    "CLUSTER_REGIME_LABEL",
    "DEFAULT_CALM_ALPHA_MAX",
    "DEFAULT_CALM_COVERAGE_Q",
    "DEFAULT_CALM_FLOOR_MIN_BARS",
    "DEFAULT_CALM_ONLINE",
    "DEFAULT_ENVELOPE_Q",
    "DEFAULT_CLAMP_CAP_Q",
    "DEFAULT_CLAMP_COV_LOOKBACK",
    "DEFAULT_CLAMP_FLOOR_Q",
    "DEFAULT_EWMA_LAMBDA",
    "DEFAULT_FIT_BARS",
    "DEFAULT_FLOOR_Q",
    "DEFAULT_MIN_BARS",
    "DEFAULT_Z_LOOKBACK",
    "EXTREME_REGIME_LABEL",
    "LOOKBACK_3Y",
    "LOOKBACK_5Y",
    "NORMAL_REGIME_LABEL",
    "REGIMES_NO_HI",
    "REGIMES_WITH_HI",
    "TRADING_DAYS_PER_YEAR",
    "apply_research_bands_to_ms",
    "b1_entries_variant",
    "build_inertia_panels",
    "build_research_panels",
    "coverage_report",
    "ewma_clamped_envelope",
    "ewma_residual_sigma",
    "ewma_scaled_envelope",
    "kalman_forecast_band",
    "overlay_ewma_rails",
    "kalman_long_ok",
    "kalman_trend_path",
    "regime_pass_table",
    "residual_p95_envelope",
    "vol_scaled_envelope",
]
