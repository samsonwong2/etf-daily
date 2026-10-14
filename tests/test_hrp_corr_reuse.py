"""Offline tests for HRP correlation reuse and close-panel cache matching."""
from __future__ import annotations

import numpy as np
import pandas as pd

from etf_daily.hrp.dendrogram import (
    close_cache_usable,
    compute_cluster_representatives,
    domain_ward_labels,
    ward_labels,
)


def _returns() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 40
    a = rng.normal(size=n)
    b = a + rng.normal(scale=0.05, size=n)
    c = rng.normal(size=n)
    d = c + rng.normal(scale=0.05, size=n)
    return pd.DataFrame({"A": a, "B": b, "C": c, "D": d})


def test_ward_labels_match_when_corr_is_passed():
    returns = _returns()
    fresh = ward_labels(returns, dist_t=0.4)
    reused = ward_labels(returns, dist_t=0.4, corr=returns.corr())
    pd.testing.assert_series_equal(fresh, reused)


def test_domain_ward_labels_are_stable():
    returns = _returns()
    names = {"A": "通信ETF", "B": "半导体ETF", "C": "黄金ETF", "D": "白银ETF"}
    first, _, _ = domain_ward_labels(returns, names, dist_t=0.4, min_corr=0.0)
    second, _, _ = domain_ward_labels(returns, names, dist_t=0.4, min_corr=0.0)
    pd.testing.assert_series_equal(first, second)


def test_representatives_keep_domain_when_labels_are_passed():
    returns = _returns()
    names = {"A": "通信ETF", "B": "半导体ETF", "C": "黄金ETF", "D": "白银ETF"}
    labels, domains, corr = domain_ward_labels(returns, names, dist_t=0.4, min_corr=0.0)
    close = (1.0 + returns).cumprod()
    reps = compute_cluster_representatives(
        returns,
        close,
        name_map=names,
        dist_t=0.4,
        min_corr=0.0,
        labels=labels,
        corr=corr,
        domains=domains,
    )
    assert not reps.empty
    assert set(reps["domain"]).issubset({"bond", "overseas", "domestic"})


def test_close_cache_usable():
    meta = {"asof": "2026-09-30", "lookback": 252, "calendar_last": "2026-09-30"}
    assert close_cache_usable(meta, "2026-09-30", 252, "2026-09-30") is True
    assert close_cache_usable(meta, "2026-09-29", 252, "2026-09-30") is False
    assert close_cache_usable(meta, "2026-09-30", 120, "2026-09-30") is False
    assert close_cache_usable(meta, "2026-09-30", 252, "2026-10-09") is False
    assert close_cache_usable(meta, "2026-09-30", 252, "") is False
