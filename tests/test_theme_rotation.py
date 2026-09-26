"""Unit tests for theme rotation helpers (no qlib)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from etf_daily.scripts import backtest_theme_rotation as mod


def _synthetic_prices(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-02", periods=n)
    cols = [f"C{i}" for i in range(8)]
    # Two return regimes → should tend to separate under clustering
    data = {}
    for i, c in enumerate(cols):
        shock = rng.normal(0.001 if i < 4 else -0.0005, 0.01, size=n)
        data[c] = 100 * np.cumprod(1.0 + shock)
    return pd.DataFrame(data, index=idx)


def test_causal_k_buckets_returns_four_nonempty():
    px = _synthetic_prices()
    asof = px.index[-1]
    codes = list(px.columns)
    buckets = mod.causal_k_buckets(px, codes, asof, n_themes=4, lookback=63)
    assert len(buckets) == 4
    assert sum(len(v) for v in buckets.values()) == len(codes)
    flat = [c for v in buckets.values() for c in v]
    assert sorted(flat) == sorted(codes)


def test_top_bucket_weights_causal_top1():
    px = _synthetic_prices()
    asof = px.index[-1]
    w, err = mod.top_bucket_weights(
        px,
        list(px.columns),
        asof,
        mode="causal",
        n_themes=4,
        cluster_lookback=63,
        mom_lookback=20,
        top_k=1,
    )
    assert err is None
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(v > 0 for v in w.values())


def test_fixed_theme_covers_pack_codes():
    # Spot-check mapping keys exist for known semis / rare-metal
    assert mod.FIXED_THEME["SH513310"] == "tech_semi"
    assert mod.FIXED_THEME["SH561800"] == "resources"
    buckets = mod.fixed_theme_buckets(list(mod.FIXED_THEME.keys()))
    assert set(buckets.keys()) == set(mod.THEME_ORDER)
