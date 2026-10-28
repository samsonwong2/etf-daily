"""Causality of fig7–11 path, rails, and percentiles. No Qlib."""
from __future__ import annotations

import numpy as np
import pandas as pd

from etf_daily.lib.fair_path_band_signals import path_rising
from etf_daily.lib.fair_path_research import ewma_scaled_envelope
from etf_daily.plots.plot_regime_transition_example import (
    causal_path_residual_signed_percentile,
    erp_path_symmetric_envelope,
    ewma_path_residual_signed_percentile,
    trailing_log_trend_price_path,
)


def _close(n: int = 400, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    shock = rng.normal(0.0002, 0.012, size=n)
    return pd.Series(100.0 * np.exp(np.cumsum(shock)), index=idx)


def _assert_prefix_equal(full: pd.Series, cut: pd.Series, n: int) -> None:
    a = full.iloc[:n].to_numpy(dtype=float)
    b = cut.to_numpy(dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    assert int(m.sum()) > 30
    np.testing.assert_allclose(a[m], b[m], rtol=0, atol=1e-12)


def test_trailing_log_path_and_g_are_causal():
    close = _close()
    cut = 250
    path, g = trailing_log_trend_price_path(close, trail_years=1.0, min_bars=63)
    path_c, g_c = trailing_log_trend_price_path(close.iloc[:cut], trail_years=1.0, min_bars=63)
    _assert_prefix_equal(path, path_c, cut)
    _assert_prefix_equal(g, g_c, cut)


def test_p95_rails_are_causal():
    close = _close()
    path, _ = trailing_log_trend_price_path(close, trail_years=1.0, min_bars=63)
    cut = 250
    up, lo, _ = erp_path_symmetric_envelope(path, close, q=0.95, min_bars=63)
    up_c, lo_c, _ = erp_path_symmetric_envelope(
        path.iloc[:cut], close.iloc[:cut], q=0.95, min_bars=63
    )
    _assert_prefix_equal(up, up_c, cut)
    _assert_prefix_equal(lo, lo_c, cut)


def test_ewma_rails_are_causal():
    close = _close()
    path, _ = trailing_log_trend_price_path(close, trail_years=1.0, min_bars=63)
    cut = 280
    up, lo, _ = ewma_scaled_envelope(path, close, min_bars=63)
    up_c, lo_c, _ = ewma_scaled_envelope(path.iloc[:cut], close.iloc[:cut], min_bars=63)
    _assert_prefix_equal(up, up_c, cut)
    _assert_prefix_equal(lo, lo_c, cut)


def test_signed_percentiles_are_causal():
    close = _close()
    path, _ = trailing_log_trend_price_path(close, trail_years=1.0, min_bars=63)
    cut = 280
    p95 = causal_path_residual_signed_percentile(path, close, min_bars=63)
    p95_c = causal_path_residual_signed_percentile(
        path.iloc[:cut], close.iloc[:cut], min_bars=63
    )
    ewma = ewma_path_residual_signed_percentile(path, close, min_bars=63)
    ewma_c = ewma_path_residual_signed_percentile(
        path.iloc[:cut], close.iloc[:cut], min_bars=63
    )
    _assert_prefix_equal(p95, p95_c, cut)
    _assert_prefix_equal(ewma, ewma_c, cut)


def test_path_rising_is_causal():
    close = _close()
    path, _ = trailing_log_trend_price_path(close, trail_years=1.0, min_bars=63)
    cut = 250
    full = path_rising(path, window=20)
    part = path_rising(path.iloc[:cut], window=20)
    _assert_prefix_equal(full.astype(float), part.astype(float), cut)
