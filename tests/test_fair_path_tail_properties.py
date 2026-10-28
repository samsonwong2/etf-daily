"""Property tests for fig7–11 rails. Some xfail: they state the desired
fat-tail property and are red on the legacy implementation on purpose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_daily.lib.fair_path_research import ewma_residual_sigma, ewma_scaled_envelope
from etf_daily.lib.fair_path_tail_audit import (
    endpoint_leverage,
    insample_and_exante_residual,
)
from etf_daily.lib.fair_path_variants import split_signed_envelope
from etf_daily.lib.fat_tail_risk import fit_tail
from etf_daily.plots.plot_regime_transition_example import (
    erp_path_symmetric_envelope,
    trailing_log_trend_price_path,
)


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2016-01-01", periods=n)


def _gaussian_close(n: int = 2500, seed: int = 3, sigma: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    shock = rng.normal(0.00015, sigma, size=n)
    return pd.Series(100.0 * np.exp(np.cumsum(shock)), index=_idx(n))


def _asymmetric_close(n: int = 3000, seed: int = 5) -> pd.Series:
    """Trend plus iid noise: heavy left tail, thin right tail."""
    rng = np.random.default_rng(seed)
    left = -rng.pareto(1.8, size=n) * 0.008
    right = rng.pareto(6.0, size=n) * 0.004
    sign = rng.random(n) < 0.5
    noise = np.where(sign, left, right)
    i = np.arange(n, dtype=float)
    log_p = np.log(100.0) + 0.0001 * i + noise
    return pd.Series(np.exp(log_p), index=_idx(n))


def _breach(close: pd.Series, path: pd.Series, upper: pd.Series, lower: pd.Series) -> tuple[float, float, int]:
    c = close.to_numpy(dtype=float)
    p = path.to_numpy(dtype=float)
    u = upper.to_numpy(dtype=float)
    lo = lower.to_numpy(dtype=float)
    ok = np.isfinite(c) & np.isfinite(p) & np.isfinite(u) & np.isfinite(lo) & (p > 0)
    n = int(ok.sum())
    left = float(np.mean(c[ok] < lo[ok])) if n else float("nan")
    right = float(np.mean(c[ok] > u[ok])) if n else float("nan")
    return left, right, n


def test_endpoint_leverage_formula():
    # h = (4n-2)/(n(n+1)); about 4/n on the short windows.
    assert endpoint_leverage(21) == pytest.approx((4 * 21 - 2) / (21 * 22))
    assert endpoint_leverage(21) == pytest.approx(0.177, abs=0.01)
    assert endpoint_leverage(63) == pytest.approx(0.062, abs=0.005)
    assert endpoint_leverage(252) == pytest.approx(0.016, abs=0.002)


def test_gaussian_p95_coverage_near_nominal():
    close = _gaussian_close()
    path, _ = trailing_log_trend_price_path(close, trail_years=1.0, min_bars=63)
    up, lo, _ = erp_path_symmetric_envelope(path, close, q=0.95, min_bars=63)
    left, right, n = _breach(close, path, up, lo)
    assert n > 500
    # Expanding P95 includes each point in its own threshold, so the
    # realized breach rate sits a bit above the nominal 2.5% per side.
    assert 0.005 < left < 0.09
    assert 0.005 < right < 0.09
    assert abs(left - right) < 0.025


def test_legacy_symmetric_rails_over_breach_the_heavy_side():
    """Documents the violation: one |r| quantile cannot match both tails."""
    close = _asymmetric_close()
    path, _ = trailing_log_trend_price_path(close, trail_years=1.0, min_bars=63)
    up, lo, _ = erp_path_symmetric_envelope(path, close, q=0.95, min_bars=63)
    left, right, n = _breach(close, path, up, lo)
    assert n > 500
    assert left > right * 1.3


def test_split_rails_shrink_the_left_right_breach_gap():
    close = _asymmetric_close()
    path, _ = trailing_log_trend_price_path(close, trail_years=1.0, min_bars=63)
    up_s, lo_s, _ = erp_path_symmetric_envelope(path, close, q=0.95, min_bars=63)
    up_a, lo_a = split_signed_envelope(path, close, q=0.95, min_bars=63)
    left_s, right_s, n = _breach(close, path, up_s, lo_s)
    left_a, right_a, _ = _breach(close, path, up_a, lo_a)
    assert n > 500
    assert abs(left_a - right_a) < abs(left_s - right_s)


@pytest.mark.xfail(
    reason="expanding quantiles overshoot a flat 2.5% even after the sides are split",
    strict=False,
)
def test_split_rails_restore_nominal_coverage_on_asymmetric_noise():
    """Desired property. Red until left and right rails are estimated apart."""
    close = _asymmetric_close()
    path, _ = trailing_log_trend_price_path(close, trail_years=1.0, min_bars=63)
    up, lo = split_signed_envelope(path, close, q=0.95, min_bars=63)
    left, right, n = _breach(close, path, up, lo)
    assert n > 500
    assert abs(left - 0.025) < 0.015
    assert abs(right - 0.025) < 0.015


def test_legacy_ewma_sigma_forgets_a_shock_within_sixty_bars():
    """The L2 recursion itself (half-life ~11 bars) drops most of the shock by bar +60.

    The drawn rail is σ times an expanding |z| quantile, so the rail width
    can stay wide after σ has already snapped back. The fast decay is in σ.
    """
    rng = np.random.default_rng(1)
    n = 400
    r = pd.Series(rng.normal(0.0, 0.004, size=n), index=_idx(n))
    shock_at = 250
    r.iloc[shock_at] = -0.10
    sigma = ewma_residual_sigma(r, lam=0.94).to_numpy(dtype=float)
    pre = float(sigma[shock_at - 1])
    peak = float(np.nanmax(sigma[shock_at : shock_at + 5]))
    later = float(sigma[shock_at + 60])
    assert peak > pre * 3
    assert later < pre + 0.5 * (peak - pre)


@pytest.mark.xfail(
    reason="legacy EWMA width is an IGARCH variance recursion and decays in ~11 bars",
    strict=False,
)
def test_rail_width_does_not_snap_back_after_a_shock():
    """Desired §10.3 property: the L2 scale may widen in one bar, but must not decay in ~60."""
    rng = np.random.default_rng(1)
    n = 400
    r = pd.Series(rng.normal(0.0, 0.004, size=n), index=_idx(n))
    shock_at = 250
    r.iloc[shock_at] = -0.10
    sigma = ewma_residual_sigma(r, lam=0.94).to_numpy(dtype=float)
    peak = float(np.nanmax(sigma[shock_at : shock_at + 5]))
    later = float(sigma[shock_at + 60])
    assert later > 0.8 * peak


def test_insample_residual_is_shrunk_vs_exante():
    rng = np.random.default_rng(4)
    n_win = 63
    sigma = 0.02
    # Long enough that the ex-ante window of ``n_win`` bars is full.
    n = 400
    log_y = 0.0002 * np.arange(n) + rng.normal(0.0, sigma, size=n)
    close = pd.Series(np.exp(log_y))
    rin, rex = insample_and_exante_residual(close, trail_bars=n_win, min_bars=n_win)
    m = np.isfinite(rin) & np.isfinite(rex)
    assert int(m.sum()) > 200
    h = endpoint_leverage(n_win)
    # In-sample endpoint residual variance is (1-h) σ^2. Ex-ante is larger.
    assert float(np.mean(rin[m] ** 2)) == pytest.approx((1.0 - h) * sigma ** 2, rel=0.25)
    assert float(np.mean(rin[m] ** 2)) < float(np.mean(rex[m] ** 2)) * 0.95


def test_crash_day_insample_residual_shrinks():
    rng = np.random.default_rng(8)
    n = 80
    log_y = np.log(100.0) + np.cumsum(rng.normal(0.0, 0.01, size=n))
    log_y[-1] = log_y[-2] - 0.15
    close = pd.Series(np.exp(log_y))
    rin, rex = insample_and_exante_residual(close, trail_bars=63, min_bars=40)
    assert abs(rin[-1]) < abs(rex[-1])


def test_gpd_recovers_pareto_alpha_when_n_is_large():
    rng = np.random.default_rng(12)
    # Pareto exceedances already above 0; fit_tail thresholds the sample.
    xm = 1.0
    alpha = 2.5
    n = 2000
    u = rng.random(n)
    sample = xm / (1.0 - u) ** (1.0 / alpha)
    # Symmetric body plus a Pareto right tail so the quantile threshold is interior.
    body = rng.normal(0.0, 0.5, size=n)
    values = np.concatenate([body, sample])
    fit = fit_tail(values, side="right", tail_quantile=0.90, min_exceed=25)
    assert fit.reliable
    assert fit.alpha is not None
    assert 1.5 < fit.alpha < 4.5


def test_gpd_refuses_when_exceedances_are_scarce():
    rng = np.random.default_rng(13)
    values = rng.normal(0.0, 1.0, size=80)
    fit = fit_tail(values, side="left", tail_quantile=0.925, min_exceed=25)
    assert fit.reliable is False
    assert fit.alpha is None


def test_short_window_g_is_noise_annualized():
    # Sign across independent series: one path's 2-year slope stays one sign
    # for a long time, so the coin-flip check has to be across series.
    signs = []
    for i in range(60):
        rng = np.random.default_rng(1000 + i)
        n = 40
        close = pd.Series(
            100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, size=n))),
            index=_idx(n),
        )
        _, g = trailing_log_trend_price_path(close, trail_years=21 / 252, min_bars=21)
        last = float(g.dropna().iloc[-1])
        signs.append(last > 0)
    assert 0.30 < float(np.mean(signs)) < 0.70

    rng = np.random.default_rng(15)
    n = 1600
    close = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, size=n))),
        index=_idx(n),
    )
    med = {}
    for bars, years in ((21, 21 / 252), (504, 2.0)):
        _, g = trailing_log_trend_price_path(close, trail_years=years, min_bars=min(63, bars))
        gv = g.to_numpy(dtype=float)
        gv = gv[np.isfinite(gv)]
        med[bars] = float(np.median(np.abs(gv)))
    # Annualizing a 21-bar slope inflates |g| relative to a 2-year slope.
    assert med[21] > 3.0 * med[504]


def test_rails_bracket_path_and_p99_is_wider():
    close = _gaussian_close(800, seed=2)
    path, _ = trailing_log_trend_price_path(close, trail_years=0.5, min_bars=63)
    up95, lo95, _ = erp_path_symmetric_envelope(path, close, q=0.95, min_bars=63)
    up99, lo99, _ = erp_path_symmetric_envelope(path, close, q=0.99, min_bars=63)
    m = up95.notna() & lo95.notna() & path.notna()
    assert bool(m.any())
    assert bool((lo95[m] < path[m]).all())
    assert bool((path[m] < up95[m]).all())
    both = m & up99.notna()
    assert bool((up99[both] >= up95[both] - 1e-9).all())
    assert bool((lo99[both] <= lo95[both] + 1e-9).all())
    # Log-symmetric by construction.
    width_up = np.log(up95[m].to_numpy() / path[m].to_numpy())
    width_dn = np.log(path[m].to_numpy() / lo95[m].to_numpy())
    np.testing.assert_allclose(width_up, width_dn, atol=1e-10)


def test_warmup_rails_are_nan_and_short_listing_does_not_crash():
    close = _gaussian_close(40, seed=1)
    path, g = trailing_log_trend_price_path(close, trail_years=2.0, min_bars=63)
    # Fewer than 63 bars: path falls back to endpoint CAGR, rails stay NaN.
    assert path.notna().any()
    assert g.notna().any()
    up, lo, delta = erp_path_symmetric_envelope(path, close, q=0.95, min_bars=63)
    assert not up.notna().any()
    assert not lo.notna().any()
    assert not np.isfinite(delta)


def test_constant_and_nan_close_do_not_crash():
    idx = _idx(80)
    const = pd.Series(3.0, index=idx)
    path, g = trailing_log_trend_price_path(const, trail_years=0.25, min_bars=21)
    assert path.dropna().iloc[-1] == pytest.approx(3.0, rel=1e-6)
    assert g.dropna().iloc[-1] == pytest.approx(0.0, abs=1e-8)
    dirty = const.copy()
    dirty.iloc[10:15] = np.nan
    path_d, _ = trailing_log_trend_price_path(dirty, trail_years=0.25, min_bars=21)
    assert path_d.notna().sum() > 20


def test_ewma_sigma_uses_lagged_residual_only():
    idx = _idx(30)
    r = pd.Series(0.0, index=idx)
    r.iloc[10] = 0.2
    sigma = ewma_residual_sigma(r, lam=0.94)
    # Shock at t=10 must not enter σ_10.
    assert not np.isfinite(sigma.iloc[10]) or sigma.iloc[10] < 0.2
    assert sigma.iloc[11] == pytest.approx(0.2, rel=1e-6)
