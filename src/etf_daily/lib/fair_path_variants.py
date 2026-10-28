"""Challenger rails for the fig7–11 fat-tail audit.

Production HTML stays on the legacy symmetric expanding P95. These
functions are the pre-registered challenger used by the A/B script:

- touch is judged on the one-step ex-ante residual (today's close does
  not move the ruler);
- the left rail is the parametric 5% loss when the POT/GPD fit is
  reliable, otherwise the expanding 5% quantile of the signed residual.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from etf_daily.lib.fair_path_tail_audit import insample_and_exante_residual
from etf_daily.lib.fat_tail_risk import fit_tail, gpd_loss_at_prob

REFIT_EVERY = 63
TAIL_PROB = 0.05
BODY_Q = 0.90
MIN_EXCEED = 25


def split_signed_envelope(
    path: pd.Series,
    close: pd.Series,
    *,
    q: float = 0.95,
    min_bars: int = 63,
) -> tuple[pd.Series, pd.Series]:
    """Expanding left/right quantiles of the signed log residual.

    Lower rail uses the ``1-q`` quantile, upper rail the ``q`` quantile.
    The two sides are not forced to the same width.
    """
    p = pd.to_numeric(path, errors="coerce").to_numpy(dtype=float)
    c = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    n = len(p)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    hist: list[float] = []
    qq = float(q)
    mb = max(2, int(min_bars))
    for t in range(n):
        pt = float(p[t])
        ct = float(c[t])
        if np.isfinite(pt) and pt > 0 and np.isfinite(ct) and ct > 0:
            rt = float(np.log(ct / pt))
            if np.isfinite(rt):
                hist.append(rt)
        if len(hist) < mb or not (np.isfinite(pt) and pt > 0):
            continue
        arr = np.asarray(hist, dtype=float)
        lo_q = float(np.quantile(arr, 1.0 - qq))
        hi_q = float(np.quantile(arr, qq))
        lower[t] = pt * float(np.exp(min(lo_q, 0.0)))
        upper[t] = pt * float(np.exp(max(hi_q, 0.0)))
    idx = path.index
    return (
        pd.Series(upper, index=idx, dtype=float),
        pd.Series(lower, index=idx, dtype=float),
    )


def _pot_width(values: np.ndarray, *, side: str) -> float | None:
    fit = fit_tail(
        values,
        side=side,
        tail_quantile=BODY_Q,
        min_exceed=MIN_EXCEED,
    )
    if not fit.reliable or fit.threshold is None or fit.xi is None or fit.beta is None:
        return None
    if fit.n_total <= 0:
        return None
    nu = fit.n_exceed / fit.n_total
    return gpd_loss_at_prob(
        xi=fit.xi,
        beta=fit.beta,
        threshold=fit.threshold,
        nu=nu,
        tail_prob=TAIL_PROB,
    )


def exante_pot_lower_touch(
    close: pd.Series,
    *,
    trail_bars: int,
    min_bars: int,
) -> np.ndarray:
    """True when the ex-ante residual is at or below the left rail.

    Left rail = POT 5% loss refit every ``REFIT_EVERY`` bars, falling
    back to the expanding 5% quantile of the ex-ante residual.
    """
    _, rex = insample_and_exante_residual(
        close, trail_bars=int(trail_bars), min_bars=int(min_bars)
    )
    n = len(rex)
    touch = np.zeros(n, dtype=bool)
    hist: list[float] = []
    width = float("nan")
    mb = max(2, int(min_bars))
    for t in range(n):
        rt = float(rex[t])
        if np.isfinite(rt):
            hist.append(rt)
        if len(hist) < mb or not np.isfinite(rt):
            continue
        if not np.isfinite(width) or (t % REFIT_EVERY == 0):
            arr = np.asarray(hist, dtype=float)
            pot = _pot_width(arr, side="left")
            emp = float(-np.quantile(arr, TAIL_PROB))
            width = float(pot) if pot is not None and pot > 0 else max(emp, 0.0)
        touch[t] = bool(rt <= -width)
    return touch
