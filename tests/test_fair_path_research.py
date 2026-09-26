"""Tests for fair_path_research (synthetic, no Qlib, no HTML)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from etf_daily.lib.fair_path_research import (  # noqa: E402
    CALM_REGIME_LABEL,
    EXTREME_REGIME_LABEL,
    apply_research_bands_to_ms,
    b1_entries_variant,
    coverage_report,
    ewma_clamped_envelope,
    ewma_residual_sigma,
    ewma_scaled_envelope,
    kalman_forecast_band,
    kalman_long_ok,
    kalman_trend_path,
    regime_pass_table,
    residual_p95_envelope,
    vol_scaled_envelope,
)


def _trend_close(n: int = 400, g_ann: float = 0.08, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    drift = np.log(1.0 + g_ann) / 252.0
    shocks = rng.normal(0.0, 0.01, size=n)
    log_px = np.log(100.0) + np.cumsum(drift + shocks)
    return pd.Series(np.exp(log_px), index=idx)


def _ols_path(close: pd.Series, window: int = 60) -> pd.Series:
    """Simple trailing log-OLS path for envelope tests."""
    y = np.log(close.to_numpy(dtype=float))
    out = np.full(len(y), np.nan)
    for t in range(window - 1, len(y)):
        sl = y[t + 1 - window : t + 1]
        b, a = np.polyfit(np.arange(window, dtype=float), sl, 1)
        out[t] = np.exp(a + b * (window - 1))
    return pd.Series(out, index=close.index)


# ---------- vol_scaled_envelope ----------


def test_vol_scaled_envelope_causality():
    """Truncating the series must not change earlier rails."""
    close = _trend_close(300)
    path = _ols_path(close)
    sigma = pd.Series(0.15, index=close.index)
    up_full, lo_full, _ = vol_scaled_envelope(path, close, sigma, min_bars=20)
    cut = 200
    up_cut, lo_cut, _ = vol_scaled_envelope(
        path.iloc[:cut], close.iloc[:cut], sigma.iloc[:cut], min_bars=20
    )
    m = up_full.iloc[:cut].notna() & up_cut.notna()
    assert m.sum() > 50
    np.testing.assert_allclose(
        up_full.iloc[:cut][m].to_numpy(), up_cut[m].to_numpy(), rtol=1e-10
    )
    np.testing.assert_allclose(
        lo_full.iloc[:cut][m].to_numpy(), lo_cut[m].to_numpy(), rtol=1e-10
    )


def test_vol_scaled_envelope_adapts_to_vol_regime():
    """Same path, two vol regimes: storm rails wider than calm rails."""
    n_calm, n_storm = 300, 60
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2023-01-01", periods=n_calm + n_storm)
    shocks = np.concatenate(
        [rng.normal(0, 0.003, n_calm), rng.normal(0, 0.03, n_storm)]
    )
    log_px = np.log(100.0) + np.cumsum(0.0002 + shocks)
    close = pd.Series(np.exp(log_px), index=idx)
    path = pd.Series(np.exp(log_px - shocks), index=idx)  # smooth proxy
    sigma = pd.Series(
        np.concatenate([np.full(n_calm, 0.05), np.full(n_storm, 0.50)]), index=idx
    )
    up, lo, _ = vol_scaled_envelope(path, close, sigma, min_bars=20)
    width = (up / lo).apply(np.log)
    calm_w = width.iloc[n_calm - 10 : n_calm].mean()
    storm_w = width.iloc[-5:].mean()
    assert np.isfinite(calm_w) and np.isfinite(storm_w)
    assert storm_w > calm_w * 3.0


def test_vol_scaled_envelope_warmup_nan():
    close = _trend_close(50)
    path = _ols_path(close, window=10)
    sigma = pd.Series(0.15, index=close.index)
    up, lo, delta = vol_scaled_envelope(path, close, sigma, min_bars=40)
    assert up.iloc[:40].isna().all()


def test_vol_scaled_envelope_floor_widens_calm():
    """Floor bites when current sigma is far below the sigma that set δ*."""
    rng = np.random.default_rng(9)
    n_storm, n_calm = 150, 80
    idx = pd.bdate_range("2024-01-01", periods=n_storm + n_calm)
    # Storm residuals ~5% at sigma=0.30; calm residuals ~1% at sigma=0.02.
    r = np.concatenate(
        [rng.normal(0, 0.05, n_storm), rng.normal(0, 0.01, n_calm)]
    )
    close = pd.Series(100.0 * np.exp(r), index=idx)
    path = pd.Series(100.0, index=idx)
    sigma = pd.Series(
        np.concatenate([np.full(n_storm, 0.30), np.full(n_calm, 0.02)]), index=idx
    )
    up0, lo0, _ = vol_scaled_envelope(path, close, sigma, min_bars=20, floor_q=None)
    upf, lof, _ = vol_scaled_envelope(path, close, sigma, min_bars=20, floor_q=0.80)
    # Compare last 20 calm days.
    sl = slice(-20, None)
    m = up0.iloc[sl].notna() & upf.iloc[sl].notna()
    w0 = np.log(up0.iloc[sl][m] / lo0.iloc[sl][m]).mean()
    wf = np.log(upf.iloc[sl][m] / lof.iloc[sl][m]).mean()
    assert np.isfinite(w0) and np.isfinite(wf)
    assert wf > w0 * 1.5


def test_vol_scaled_envelope_floor_still_widens_in_storm():
    """Floor must not erase the storm > calm width ordering."""
    n_calm, n_storm = 300, 60
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2023-01-01", periods=n_calm + n_storm)
    shocks = np.concatenate(
        [rng.normal(0, 0.003, n_calm), rng.normal(0, 0.03, n_storm)]
    )
    log_px = np.log(100.0) + np.cumsum(0.0002 + shocks)
    close = pd.Series(np.exp(log_px), index=idx)
    path = pd.Series(np.exp(log_px - shocks), index=idx)
    sigma = pd.Series(
        np.concatenate([np.full(n_calm, 0.05), np.full(n_storm, 0.50)]), index=idx
    )
    up, lo, _ = vol_scaled_envelope(path, close, sigma, min_bars=20, floor_q=0.80)
    width = (up / lo).apply(np.log)
    calm_w = width.iloc[n_calm - 10 : n_calm].mean()
    storm_w = width.iloc[-5:].mean()
    assert np.isfinite(calm_w) and np.isfinite(storm_w)
    assert storm_w > calm_w * 2.0


def test_vol_scaled_envelope_abs_floor():
    close = _trend_close(200)
    path = _ols_path(close)
    sigma = pd.Series(0.001, index=close.index)
    up, lo, _ = vol_scaled_envelope(
        path, close, sigma, min_bars=20, floor_q=None, abs_floor=0.05
    )
    m = up.notna()
    half = np.log(up[m] / path[m])
    assert (half >= 0.05 - 1e-12).all()


def test_calm_pin_floor_causal_and_lifts_calm_coverage():
    """Calm-day expanding P95 floor is causal and equals that quantile."""
    rng = np.random.default_rng(4)
    n_calm, n_storm = 250, 80
    idx = pd.bdate_range("2023-01-01", periods=n_calm + n_storm)
    r = np.concatenate(
        [rng.normal(0, 0.02, n_calm), rng.normal(0, 0.08, n_storm)]
    )
    close = pd.Series(100.0 * np.exp(r), index=idx)
    path = pd.Series(100.0, index=idx)
    sigma = pd.Series(0.50, index=idx)
    regime = pd.Series(
        [CALM_REGIME_LABEL] * n_calm + ["波动率极端聚集"] * n_storm, index=idx
    )
    upf, lof, _ = vol_scaled_envelope(
        path,
        close,
        sigma,
        min_bars=20,
        vol_regime=regime,
        calm_floor_min_bars=20,
        calm_online=False,
    )
    cut = n_calm
    up_cut, _, _ = vol_scaled_envelope(
        path.iloc[:cut],
        close.iloc[:cut],
        sigma.iloc[:cut],
        min_bars=20,
        vol_regime=regime.iloc[:cut],
        calm_floor_min_bars=20,
        calm_online=False,
    )
    m = upf.iloc[:cut].notna() & up_cut.notna()
    np.testing.assert_allclose(
        upf.iloc[:cut][m].to_numpy(), up_cut[m].to_numpy(), rtol=1e-10
    )
    calm = regime == CALM_REGIME_LABEL
    validf = upf.notna() & calm
    covf = float(((close <= upf) & (close >= lof))[validf].mean())
    assert covf >= 0.90
    # Last calm bar: floor is P95 of |r| on calm days ≤ t.
    last_calm = int(np.flatnonzero(calm.to_numpy() & upf.notna().to_numpy())[-1])
    r_calm = np.abs(np.log(close.iloc[: last_calm + 1][calm.iloc[: last_calm + 1]] / 100.0))
    expected = float(np.quantile(r_calm.to_numpy(dtype=float), 0.95))
    half = float(np.log(upf.iloc[last_calm] / 100.0))
    assert half == pytest.approx(expected, rel=1e-6)


def test_calm_online_scale_causal_lifts_coverage_storm_unscaled():
    """α_t from past calm |r|/w_raw pins coverage; storms stay at w_raw."""
    rng = np.random.default_rng(12)
    n_calm, n_storm = 400, 80
    idx = pd.bdate_range("2022-01-01", periods=n_calm + n_storm)
    t = np.arange(n_calm, dtype=float)
    scale = 0.008 + 0.06 * (t / n_calm) ** 2
    r_calm = rng.normal(0.0, 1.0, n_calm) * scale
    r_storm = rng.normal(0.0, 0.09, n_storm)
    r = np.concatenate([r_calm, r_storm])
    close = pd.Series(100.0 * np.exp(r), index=idx)
    path = pd.Series(100.0, index=idx)
    # Tiny sigma so δ*σ never binds; width is the calm residual floor.
    sigma = pd.Series(1e-3, index=idx)
    regime = pd.Series(
        [CALM_REGIME_LABEL] * n_calm + [EXTREME_REGIME_LABEL] * n_storm,
        index=idx,
    )
    kw = dict(min_bars=20, vol_regime=regime, calm_floor_min_bars=20)
    up0, lo0, _ = vol_scaled_envelope(path, close, sigma, calm_online=False, **kw)
    up1, lo1, _ = vol_scaled_envelope(path, close, sigma, calm_online=True, **kw)
    cut = 280
    up_cut, _, _ = vol_scaled_envelope(
        path.iloc[:cut],
        close.iloc[:cut],
        sigma.iloc[:cut],
        min_bars=20,
        vol_regime=regime.iloc[:cut],
        calm_floor_min_bars=20,
        calm_online=True,
    )
    m = up1.iloc[:cut].notna() & up_cut.notna()
    assert m.sum() > 50
    np.testing.assert_allclose(
        up1.iloc[:cut][m].to_numpy(), up_cut[m].to_numpy(), rtol=1e-10
    )
    storm = slice(n_calm, None)
    ms = up0.iloc[storm].notna() & up1.iloc[storm].notna()
    np.testing.assert_allclose(
        up0.iloc[storm][ms].to_numpy(),
        up1.iloc[storm][ms].to_numpy(),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        lo0.iloc[storm][ms].to_numpy(),
        lo1.iloc[storm][ms].to_numpy(),
        rtol=1e-10,
    )
    calm = (regime == CALM_REGIME_LABEL) & up0.notna() & up1.notna()
    inside0 = (close <= up0) & (close >= lo0)
    inside1 = (close <= up1) & (close >= lo1)
    cov0 = float(inside0[calm].mean())
    cov1 = float(inside1[calm].mean())
    assert cov1 >= cov0
    assert cov1 >= 0.90
    # Growing residuals: α must actually widen some later calm bars.
    later = calm & (np.arange(len(close)) >= n_calm // 2)
    w0 = np.log(up0[later] / lo0[later])
    w1 = np.log(up1[later] / lo1[later])
    assert float(w1.mean()) > float(w0.mean()) * 1.02
    # α never shrinks: online width ≥ offline wherever both exist.
    both = up0.notna() & up1.notna()
    assert (up1[both] >= up0[both] - 1e-12).all()


def test_calm_online_does_not_ratchet_when_raw_already_covers():
    """iid residuals: raw P95 already ~95%, so α must not keep inflating."""
    rng = np.random.default_rng(0)
    n = 400
    idx = pd.bdate_range("2023-01-01", periods=n)
    r = rng.normal(0.0, 0.02, n)
    close = pd.Series(100.0 * np.exp(r), index=idx)
    path = pd.Series(100.0, index=idx)
    sigma = pd.Series(1e-3, index=idx)
    regime = pd.Series([CALM_REGIME_LABEL] * n, index=idx)
    kw = dict(min_bars=20, vol_regime=regime, calm_floor_min_bars=20)
    up0, lo0, _ = vol_scaled_envelope(path, close, sigma, calm_online=False, **kw)
    up1, lo1, _ = vol_scaled_envelope(path, close, sigma, calm_online=True, **kw)
    both = up0.notna() & up1.notna()
    assert both.sum() > 200
    tail = both.to_numpy() & (np.arange(n) >= n - 120)
    ratio = (up1.to_numpy()[tail] / up0.to_numpy()[tail])
    assert float(np.mean(ratio)) < 1.02
    calm = (regime == CALM_REGIME_LABEL) & both
    cov0 = float(((close <= up0) & (close >= lo0))[calm].mean())
    cov1 = float(((close <= up1) & (close >= lo1))[calm].mean())
    assert cov1 <= 0.98 or cov0 > 0.98


# ---------- kalman_trend_path ----------


def test_kalman_tracks_linear_trend():
    """Filtered path on a clean log-linear trend stays near the true level."""
    n = 400
    idx = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(11)
    g_ann = 0.10
    drift = np.log(1.0 + g_ann) / 252.0
    log_px = np.log(50.0) + drift * np.arange(n) + rng.normal(0, 0.004, n)
    close = pd.Series(np.exp(log_px), index=idx)
    path, g, std, mean = kalman_trend_path(close, fit_bars=200)
    tail = slice(-20, None)
    rel_err = np.abs(np.log(path.iloc[tail].to_numpy()) - log_px[tail])
    assert np.mean(rel_err) < 0.02
    g_tail = g.iloc[tail].to_numpy()
    assert np.abs(np.nanmean(g_tail) - g_ann) < 0.08
    assert std.iloc[-20:].notna().all()
    assert (std.iloc[-20:] > 0).all()
    assert mean.iloc[-20:].notna().all()


def test_kalman_causality_after_fit_window():
    """With frozen params, earlier outputs must not depend on later data."""
    close = _trend_close(400)
    p_full, g_full, s_full, m_full = kalman_trend_path(close, fit_bars=150)
    cut = 300  # beyond fit window: params identical, pure causal filtering
    p_cut, g_cut, s_cut, m_cut = kalman_trend_path(close.iloc[:cut], fit_bars=150)
    m = p_full.iloc[:cut].notna() & p_cut.notna()
    assert m.sum() > 100
    np.testing.assert_allclose(
        p_full.iloc[:cut][m].to_numpy(), p_cut[m].to_numpy(), rtol=1e-8
    )
    np.testing.assert_allclose(
        g_full.iloc[:cut][m].to_numpy(), g_cut[m].to_numpy(), rtol=1e-8
    )
    np.testing.assert_allclose(
        m_full.iloc[:cut][m].to_numpy(), m_cut[m].to_numpy(), rtol=1e-8
    )


def test_kalman_forecast_band_ordering_and_coverage():
    """Band is centered on the pre-observation forecast, so 90% coverage
    on iid log-noise prices should land near (not at) nominal."""
    rng = np.random.default_rng(21)
    n = 600
    idx = pd.bdate_range("2023-01-01", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    path, g, std, mean = kalman_trend_path(close, fit_bars=200)
    up, lo = kalman_forecast_band(mean, std)
    m = up.notna() & lo.notna()
    assert m.sum() > 200
    assert (up[m] > lo[m]).all()
    up95, _ = kalman_forecast_band(mean, std, z=1.96)
    assert (up95[m] > up[m]).all()
    rep = coverage_report(close, {"kalman_90": (up, lo)})
    cov = rep[rep["dimension"] == "overall"]["coverage"].iloc[0]
    assert 0.70 <= cov <= 1.0


# ---------- coverage_report ----------


def test_coverage_report_overall_and_dimensions():
    rng = np.random.default_rng(5)
    n = 300
    idx = pd.bdate_range("2023-01-01", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    mid = close.rolling(20, min_periods=1).mean()
    up = mid * 1.05
    lo = mid * 0.95
    regime = pd.Series(
        np.where(np.arange(n) < n // 2, "波动率平稳", "波动率聚集"), index=idx
    )
    rep = coverage_report(close, {"band_a": (up, lo)}, vol_regime=regime)
    overall = rep[(rep["dimension"] == "overall")].iloc[0]
    assert 0.0 <= overall["coverage"] <= 1.0
    assert overall["n"] == n
    dims = set(rep["dimension"])
    assert {"overall", "year", "vol_regime", "touch_hi", "touch_lo"} <= dims
    rg = rep[rep["dimension"] == "vol_regime"]
    assert set(rg["bucket"]) == {"波动率平稳", "波动率聚集"}


def test_coverage_report_wider_band_higher_coverage():
    close = _trend_close(200)
    mid = close.rolling(10, min_periods=1).mean()
    narrow = (mid * 1.01, mid * 0.99)
    wide = (mid * 1.20, mid * 0.80)
    rep = coverage_report(close, {"narrow": narrow, "wide": wide})
    cov = rep[rep["dimension"] == "overall"].set_index("band")["coverage"]
    assert cov["wide"] >= cov["narrow"]
    assert cov["wide"] == pytest.approx(1.0)


def test_coverage_report_touch_return():
    n = 120
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = pd.Series(100.0, index=idx)
    close.iloc[50] = 120.0  # single spike above upper rail
    up = pd.Series(105.0, index=idx)
    lo = pd.Series(95.0, index=idx)
    rep = coverage_report(close, {"flat": (up, lo)}, horizons=(5,))
    th = rep[(rep["dimension"] == "touch_hi") & (rep["horizon"] == 5)].iloc[0]
    assert th["n"] == 1
    assert th["coverage"] == pytest.approx(1.0)  # back inside next day


def test_regime_pass_table_gate():
    idx = pd.bdate_range("2024-01-01", periods=80)
    close = pd.Series(100.0, index=idx)
    up = pd.Series(110.0, index=idx)
    lo = pd.Series(90.0, index=idx)
    regime = pd.Series(["波动率平稳"] * 40 + ["波动率聚集"] * 40, index=idx)
    rep = coverage_report(close, {"a": (up, lo)}, vol_regime=regime)
    tbl = regime_pass_table(rep, lo=0.90, hi=0.98, min_n=20)
    by_bucket = tbl.set_index("bucket")["pass"]
    # Calm keeps the upper bound: 100% coverage in quiet regime fails.
    assert not bool(by_bucket["波动率平稳"])
    # Cluster has no upper bound: 100% coverage in a storm passes.
    assert bool(by_bucket["波动率聚集"])
    # A band that never contains the price fails the lower bound everywhere.
    rep2 = coverage_report(
        close,
        {"a": (pd.Series(50.0, index=idx), pd.Series(40.0, index=idx))},
        vol_regime=regime,
    )
    tbl2 = regime_pass_table(rep2, lo=0.90, min_n=20)
    assert bool((~tbl2["pass"]).all())


def test_z_lookback_causality_and_forgets_storm():
    """Trailing |z| must be causal and tighten after a dead storm regime."""
    rng = np.random.default_rng(8)
    n_calm, n_storm = 250, 80
    idx = pd.bdate_range("2023-01-01", periods=n_calm + n_storm)
    r = np.concatenate(
        [rng.normal(0, 0.01, n_calm), rng.normal(0, 0.06, n_storm)]
    )
    close = pd.Series(100.0 * np.exp(r), index=idx)
    path = pd.Series(100.0, index=idx)
    sigma = pd.Series(
        np.concatenate([np.full(n_calm, 0.30), np.full(n_storm, 0.30)]), index=idx
    )
    # Trailing window shorter than the calm stretch, so the storm falls out.
    zlb = 60
    up_full, lo_full, _ = vol_scaled_envelope(
        path, close, sigma, min_bars=20, z_lookback=zlb
    )
    cut = n_calm + 40
    up_cut, lo_cut, _ = vol_scaled_envelope(
        path.iloc[:cut], close.iloc[:cut], sigma.iloc[:cut],
        min_bars=20, z_lookback=zlb,
    )
    m = up_full.iloc[:cut].notna() & up_cut.notna()
    np.testing.assert_allclose(
        up_full.iloc[:cut][m].to_numpy(), up_cut[m].to_numpy(), rtol=1e-10
    )
    # After >zlb calm bars, delta* only sees calm z's → narrower than a
    # delta* that still remembers the storm. Build the expanding variant.
    up_exp, lo_exp, _ = vol_scaled_envelope(
        path, close, sigma, min_bars=20, z_lookback=0
    )
    # Width at a point deep into the post-storm calm tail of a longer run:
    # extend with more calm bars so the trailing window is storm-free.
    extra = 100
    idx2 = pd.bdate_range(idx[-1] + pd.Timedelta(days=1), periods=extra)
    r2 = np.concatenate([r, rng.normal(0, 0.01, extra)])
    idx_all = idx.append(idx2)
    close2 = pd.Series(100.0 * np.exp(r2), index=idx_all)
    path2 = pd.Series(100.0, index=idx_all)
    sigma2 = pd.Series(0.30, index=idx_all)
    up_t, lo_t, _ = vol_scaled_envelope(
        path2, close2, sigma2, min_bars=20, z_lookback=zlb
    )
    up_e, lo_e, _ = vol_scaled_envelope(
        path2, close2, sigma2, min_bars=20, z_lookback=0
    )
    tail = slice(-30, None)
    w_t = np.log(up_t.iloc[tail] / lo_t.iloc[tail]).mean()
    w_e = np.log(up_e.iloc[tail] / lo_e.iloc[tail]).mean()
    assert np.isfinite(w_t) and np.isfinite(w_e)
    assert w_t < w_e  # trailing forgot the storm → tighter


def test_apply_research_bands_to_ms_swaps_touch():
    idx = pd.bdate_range("2024-01-01", periods=5)
    close = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=idx)
    ms = pd.DataFrame(
        {
            "px": close,
            "fig9_upper": 101.0,
            "fig9_lower": 99.0,
            "fig9_touch_lo": False,
            "fig9_touch_hi": False,
            "fig10_upper": 101.0,
            "fig10_lower": 99.0,
            "fig10_touch_lo": False,
            "fig10_touch_hi": False,
        },
        index=idx,
    )
    vs = pd.Series([90.0, 90.0, 90.0, 90.0, 90.0], index=idx)
    hi = pd.Series([91.0, 91.0, 91.0, 91.0, 91.0], index=idx)
    panels = {
        "fig9": pd.DataFrame({"vs_upper": hi, "vs_lower": vs}, index=idx),
        "fig10": pd.DataFrame({"vs_upper": hi, "vs_lower": vs}, index=idx),
    }
    out = apply_research_bands_to_ms(ms, panels, close)
    assert bool(out["fig9_touch_hi"].all())
    assert not bool(out["fig9_touch_lo"].any())


# ---------- B1 variant / Kalman B2 ----------


def _ms_frame(close: pd.Series, g7: float = 0.05, g8: float = 0.04) -> pd.DataFrame:
    idx = close.index
    return pd.DataFrame(
        {
            "px": close,
            "fig7_g": g7,
            "fig8_g": g8,
            "fig8_gap": 0.0,
            "fig9_touch_lo": False,
            "fig10_touch_lo": False,
        },
        index=idx,
    )


def test_kalman_long_ok_basic():
    idx = pd.bdate_range("2024-01-01", periods=50)
    g = pd.Series(np.linspace(-0.2, 0.3, 50), index=idx)
    ok = kalman_long_ok(g, window=10)
    assert ok.iloc[0] == False  # g<0 and no rising history
    assert ok.iloc[-1] == True  # g>0
    g_neg = pd.Series(-0.1, index=idx)
    assert not kalman_long_ok(g_neg, window=10).any()


def test_b1_variant_modes_and_extreme_block():
    n = 40
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = pd.Series(100.0, index=idx)
    ms = _ms_frame(close, g7=0.05, g8=0.05)  # OLS B2 passes
    touch = pd.Series([False] * 5 + [True] + [False] * (n - 6), index=idx)
    kal_g = pd.Series(0.10, index=idx)  # Kalman B2 passes
    ent_ols = b1_entries_variant(ms, touch_lo=touch, kalman_g=kal_g, b2_mode="ols")
    ent_k = b1_entries_variant(ms, touch_lo=touch, kalman_g=kal_g, b2_mode="kalman")
    ent_b = b1_entries_variant(ms, touch_lo=touch, kalman_g=kal_g, b2_mode="both")
    assert ent_ols.sum() == ent_k.sum() == ent_b.sum() == 1
    # Kalman g <= 0 kills kalman/both but not ols.
    kal_g_neg = pd.Series(-0.05, index=idx)
    assert b1_entries_variant(ms, touch_lo=touch, kalman_g=kal_g_neg, b2_mode="ols").sum() == 1
    assert b1_entries_variant(ms, touch_lo=touch, kalman_g=kal_g_neg, b2_mode="kalman").sum() == 0
    assert b1_entries_variant(ms, touch_lo=touch, kalman_g=kal_g_neg, b2_mode="both").sum() == 0
    # Extreme regime blocks the signal.
    regime = pd.Series(["波动率平稳"] * n, index=idx)
    regime.iloc[5] = EXTREME_REGIME_LABEL
    blocked = b1_entries_variant(
        ms, touch_lo=touch, kalman_g=kal_g, b2_mode="ols",
        vol_regime=regime, block_extreme=True,
    )
    assert blocked.sum() == 0


# ---------- residual rolling P95 / EWMA inertia ----------


def _storm_then_calm_residual(n_calm1=80, n_storm=80, n_calm2=120, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_calm1 + n_storm + n_calm2)
    r = np.concatenate(
        [
            rng.normal(0, 0.01, n_calm1),
            rng.normal(0, 0.06, n_storm),
            rng.normal(0, 0.01, n_calm2),
        ]
    )
    close = pd.Series(100.0 * np.exp(r), index=idx)
    path = pd.Series(100.0, index=idx)
    return close, path, n_calm1, n_storm, n_calm2


def test_residual_p95_lookback0_matches_production():
    close = _trend_close(250)
    path = _ols_path(close, window=40)
    up_r, lo_r, d_r = residual_p95_envelope(path, close, lookback=0, min_bars=20)
    from etf_daily.lib.fair_path_band_signals import _load_plot_mod

    pm = _load_plot_mod()
    up_p, lo_p, d_p = pm.erp_path_symmetric_envelope(path, close, min_bars=20)
    m = up_r.notna() & up_p.notna()
    assert m.sum() > 100
    np.testing.assert_allclose(up_r[m].to_numpy(), up_p[m].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(lo_r[m].to_numpy(), lo_p[m].to_numpy(), rtol=1e-12)
    assert d_r == pytest.approx(d_p, rel=1e-12)


def test_residual_p95_rolling_causal_and_forgets_storm():
    close, path, n_calm1, n_storm, n_calm2 = _storm_then_calm_residual()
    zlb = 50  # shorter than post-storm calm
    up_full, lo_full, _ = residual_p95_envelope(path, close, min_bars=20, lookback=zlb)
    cut = n_calm1 + n_storm + 40
    up_cut, lo_cut, _ = residual_p95_envelope(
        path.iloc[:cut], close.iloc[:cut], min_bars=20, lookback=zlb
    )
    m = up_full.iloc[:cut].notna() & up_cut.notna()
    np.testing.assert_allclose(
        up_full.iloc[:cut][m].to_numpy(), up_cut[m].to_numpy(), rtol=1e-10
    )
    up_exp, lo_exp, _ = residual_p95_envelope(path, close, min_bars=20, lookback=0)
    tail = slice(-30, None)
    w_r = np.log(up_full.iloc[tail] / lo_full.iloc[tail]).mean()
    w_e = np.log(up_exp.iloc[tail] / lo_exp.iloc[tail]).mean()
    assert np.isfinite(w_r) and np.isfinite(w_e)
    assert w_r < w_e  # rolling forgot the storm


def test_ewma_sigma_causal_ignores_today_residual():
    idx = pd.bdate_range("2024-01-01", periods=80)
    r = pd.Series(0.01, index=idx)
    sig = ewma_residual_sigma(r)
    assert np.isnan(sig.iloc[0])
    r_shock = r.copy()
    r_shock.iloc[40] = 0.50
    sig_s = ewma_residual_sigma(r_shock)
    # Today’s shock must not move σ_t; it shows up at t+1.
    assert sig_s.iloc[40] == pytest.approx(sig.iloc[40], rel=1e-12)
    assert sig_s.iloc[41] > sig.iloc[41] * 2.0
    cut = 50
    sig_cut = ewma_residual_sigma(r.iloc[:cut])
    m = sig.iloc[:cut].notna() & sig_cut.notna()
    np.testing.assert_allclose(
        sig.iloc[:cut][m].to_numpy(), sig_cut[m].to_numpy(), rtol=1e-12
    )


def test_ewma_envelope_causal_and_forgets_storm():
    close, path, n_calm1, n_storm, n_calm2 = _storm_then_calm_residual()
    up_full, lo_full, _ = ewma_scaled_envelope(path, close, min_bars=20)
    cut = n_calm1 + n_storm + 40
    up_cut, _, _ = ewma_scaled_envelope(
        path.iloc[:cut], close.iloc[:cut], min_bars=20
    )
    m = up_full.iloc[:cut].notna() & up_cut.notna()
    np.testing.assert_allclose(
        up_full.iloc[:cut][m].to_numpy(), up_cut[m].to_numpy(), rtol=1e-10
    )
    up_exp, lo_exp, _ = residual_p95_envelope(path, close, min_bars=20, lookback=0)
    tail = slice(-30, None)
    w_e = np.log(up_full.iloc[tail] / lo_full.iloc[tail]).mean()
    w_x = np.log(up_exp.iloc[tail] / lo_exp.iloc[tail]).mean()
    assert np.isfinite(w_e) and np.isfinite(w_x)
    assert w_e < w_x  # EWMA σ decayed; expanding |r| P95 did not


def test_overlay_ewma_rails_swaps_touch_flags():
    from etf_daily.lib.fair_path_research import overlay_ewma_rails

    close, path, *_ = _storm_then_calm_residual()
    up_x, lo_x, _ = residual_p95_envelope(path, close, min_bars=20, lookback=0)
    up_e, lo_e, _ = ewma_scaled_envelope(path, close, min_bars=20)
    ms = pd.DataFrame(
        {
            "px": close,
            "fig9_path": path,
            "fig9_upper": up_x,
            "fig9_lower": lo_x,
            "fig9_touch_lo": close <= lo_x,
            "fig9_touch_hi": close >= up_x,
        },
        index=close.index,
    )
    out = overlay_ewma_rails(ms, close, min_bars=20)
    m = lo_e.notna()
    np.testing.assert_allclose(out.loc[m, "fig9_lower"], lo_e.loc[m], rtol=1e-10)
    np.testing.assert_allclose(out.loc[m, "fig9_upper"], up_e.loc[m], rtol=1e-10)
    assert bool((out.loc[m, "fig9_touch_lo"] != ms.loc[m, "fig9_touch_lo"]).any())


def test_ewma_clamped_matches_unclamped_without_regime():
    close, path, *_ = _storm_then_calm_residual()
    up0, lo0, d0 = ewma_scaled_envelope(path, close, min_bars=20)
    up1, lo1, d1 = ewma_clamped_envelope(path, close, min_bars=20, vol_regime=None)
    m = up0.notna() & up1.notna()
    np.testing.assert_allclose(up0[m].to_numpy(), up1[m].to_numpy(), rtol=1e-12)
    assert d0 == pytest.approx(d1, rel=1e-12)


def test_ewma_clamped_storm_unscaled_overwide_cap_causal():
    close, path, n_calm1, n_storm, n_calm2 = _storm_then_calm_residual()
    idx = close.index
    regime = pd.Series(
        [CALM_REGIME_LABEL] * n_calm1
        + [EXTREME_REGIME_LABEL] * n_storm
        + [CALM_REGIME_LABEL] * n_calm2,
        index=idx,
    )
    kw = dict(min_bars=20, calm_min_bars=20)
    up0, lo0, _ = ewma_scaled_envelope(path, close, min_bars=20)
    up1, lo1, _ = ewma_clamped_envelope(
        path, close, vol_regime=regime, **kw
    )
    cut = n_calm1 + n_storm + 40
    up_cut, _, _ = ewma_clamped_envelope(
        path.iloc[:cut],
        close.iloc[:cut],
        vol_regime=regime.iloc[:cut],
        **kw,
    )
    m = up1.iloc[:cut].notna() & up_cut.notna()
    np.testing.assert_allclose(
        up1.iloc[:cut][m].to_numpy(), up_cut[m].to_numpy(), rtol=1e-10
    )
    storm = slice(n_calm1, n_calm1 + n_storm)
    ms = up0.iloc[storm].notna() & up1.iloc[storm].notna()
    np.testing.assert_allclose(
        up0.iloc[storm][ms].to_numpy(), up1.iloc[storm][ms].to_numpy(), rtol=1e-10
    )
    later_calm = slice(-40, None)
    w0 = np.log(up0.iloc[later_calm] / lo0.iloc[later_calm])
    w1 = np.log(up1.iloc[later_calm] / lo1.iloc[later_calm])
    assert float(w1.mean()) <= float(w0.mean()) + 1e-12


def test_ewma_clamped_idle_when_not_overwide():
    """coverage_hi=1: cap never fires; with floor off, rails match unclamped."""
    rng = np.random.default_rng(0)
    n = 400
    idx = pd.bdate_range("2023-01-01", periods=n)
    r = rng.normal(0.0, 0.02, n)
    close = pd.Series(100.0 * np.exp(r), index=idx)
    path = pd.Series(100.0, index=idx)
    regime = pd.Series([CALM_REGIME_LABEL] * n, index=idx)
    up0, lo0, _ = ewma_scaled_envelope(path, close, min_bars=20)
    up1, lo1, _ = ewma_clamped_envelope(
        path,
        close,
        min_bars=20,
        vol_regime=regime,
        calm_min_bars=20,
        floor_q=None,
        coverage_hi=1.0,
    )
    both = up0.notna() & up1.notna()
    np.testing.assert_allclose(up0[both].to_numpy(), up1[both].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(lo0[both].to_numpy(), lo1[both].to_numpy(), rtol=1e-12)


def test_ewma_clamped_caps_only_when_overwide():
    """Storm leftover: trailing calm coverage >98% then P95 cap binds."""
    n_calm1, n_storm, n_calm2 = 40, 80, 200
    close, path, *_ = _storm_then_calm_residual(
        n_calm1=n_calm1, n_storm=n_storm, n_calm2=n_calm2, seed=3
    )
    idx = close.index
    regime = pd.Series(
        [CALM_REGIME_LABEL] * n_calm1
        + [EXTREME_REGIME_LABEL] * n_storm
        + [CALM_REGIME_LABEL] * n_calm2,
        index=idx,
    )
    up0, lo0, _ = ewma_scaled_envelope(path, close, min_bars=20)
    up1, lo1, _ = ewma_clamped_envelope(
        path, close, min_bars=20, vol_regime=regime, calm_min_bars=20, floor_q=None
    )
    # Trailing 60 needs ~40 post-storm calm bars to exceed 98%; σ still elevated.
    a = n_calm1 + n_storm + 40
    b = n_calm1 + n_storm + 80
    w0 = np.log(up0.iloc[a:b] / lo0.iloc[a:b])
    w1 = np.log(up1.iloc[a:b] / lo1.iloc[a:b])
    assert float(w1.mean()) < float(w0.mean())
    inside1 = (close <= up1) & (close >= lo1)
    calm_win = regime.eq(CALM_REGIME_LABEL) & up1.notna()
    calm_win = calm_win & (np.arange(len(close)) >= a) & (np.arange(len(close)) < b)
    cov1 = float(inside1[calm_win].mean())
    assert cov1 <= 0.98 + 1e-9


