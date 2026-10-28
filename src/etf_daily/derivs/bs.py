"""Black-Scholes prices and implied-vol inversion. Dividend yield is zero."""
from __future__ import annotations

import math

# Ledger check: S = K = 100, r = 0, q = 0, sigma = 0.20, tau = 1.
ORACLE_CALL = 7.9656
ORACLE_TOL = 0.01
DEFAULT_R = 0.015


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1(spot: float, strike: float, rate: float, sigma: float, tau: float) -> float:
    vol = sigma * math.sqrt(tau)
    return (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * tau) / vol


def bs_call(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    tau: float,
) -> float:
    """European call. q = 0. ``rate`` is a continuous rate."""
    if spot <= 0 or strike <= 0 or sigma <= 0 or tau <= 0:
        raise ValueError("spot, strike, sigma, and tau must be positive")
    d1 = _d1(spot, strike, rate, sigma, tau)
    d2 = d1 - sigma * math.sqrt(tau)
    return spot * norm_cdf(d1) - strike * math.exp(-rate * tau) * norm_cdf(d2)


def bs_put(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    tau: float,
) -> float:
    """European put. q = 0."""
    if spot <= 0 or strike <= 0 or sigma <= 0 or tau <= 0:
        raise ValueError("spot, strike, sigma, and tau must be positive")
    d1 = _d1(spot, strike, rate, sigma, tau)
    d2 = d1 - sigma * math.sqrt(tau)
    return strike * math.exp(-rate * tau) * norm_cdf(-d2) - spot * norm_cdf(-d1)


def bs_delta(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    tau: float,
    *,
    kind: str,
) -> float:
    """Call delta is N(d1). Put delta is N(d1) - 1. q = 0."""
    if kind not in ("call", "put"):
        raise ValueError("kind must be call or put")
    d1 = _d1(spot, strike, rate, sigma, tau)
    call = norm_cdf(d1)
    return call if kind == "call" else call - 1.0


def invert_iv(
    price: float,
    spot: float,
    strike: float,
    rate: float,
    tau: float,
    *,
    kind: str = "call",
    lo: float = 1e-4,
    hi: float = 5.0,
    price_tol: float = 1e-8,
    max_iter: int = 80,
) -> float | None:
    """Positive sigma whose model price matches ``price``, or None.

    Failure is a non-positive quote, a quote outside the model range, or a
    root that still misses ``price`` by more than ``price_tol``.
    """
    if kind not in ("call", "put"):
        raise ValueError("kind must be call or put")
    pricer = bs_call if kind == "call" else bs_put
    try:
        mark = float(price)
        spot_f = float(spot)
        strike_f = float(strike)
        tau_f = float(tau)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (mark, spot_f, strike_f, tau_f, rate)):
        return None
    if mark <= 0 or spot_f <= 0 or strike_f <= 0 or tau_f <= 0:
        return None
    try:
        low_px = pricer(spot_f, strike_f, rate, lo, tau_f)
        high_px = pricer(spot_f, strike_f, rate, hi, tau_f)
    except ValueError:
        return None
    if mark < low_px - price_tol or mark > high_px + price_tol:
        return None
    left, right = lo, hi
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        model = pricer(spot_f, strike_f, rate, mid, tau_f)
        if abs(model - mark) <= price_tol:
            return mid
        if model < mark:
            left = mid
        else:
            right = mid
    sigma = 0.5 * (left + right)
    if abs(pricer(spot_f, strike_f, rate, sigma, tau_f) - mark) > price_tol:
        return None
    return sigma
