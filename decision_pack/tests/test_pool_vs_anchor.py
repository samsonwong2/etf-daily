"""Tests for pool-vs-anchor equal-weight gate helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]


def _load_mod():
    path = _ROOT / "decision_pack/scripts/backtest_pool_vs_anchor.py"
    spec = importlib.util.spec_from_file_location("pool_vs_anchor_mod", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_listed_on_excludes_future_listings():
    mod = _load_mod()
    listings = {"A": "2024-06-01", "B": "2023-01-01"}
    got = mod.listed_on(["A", "B"], listings, pd.Timestamp("2024-03-01"))
    assert got == ["B"]
    got2 = mod.listed_on(["A", "B"], listings, pd.Timestamp("2024-06-01"))
    assert set(got2) == {"A", "B"}


def test_equal_and_cluster_weights():
    mod = _load_mod()
    assert mod.equal_weights(["X", "Y"]) == {"X": 0.5, "Y": 0.5}
    mem = pd.DataFrame(
        {"code": ["A", "B", "C"], "cluster": [1, 1, 2]}
    )
    w = mod.cluster_balanced_weights(["A", "B", "C"], mem)
    assert abs(w["A"] - 0.25) < 1e-9
    assert abs(w["B"] - 0.25) < 1e-9
    assert abs(w["C"] - 0.5) < 1e-9
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_ew_sim_two_identical_series():
    mod = _load_mod()
    idx = pd.bdate_range("2024-01-02", periods=80)
    # Both rise ~ same path
    px = pd.Series(100 * np.exp(np.cumsum(np.full(80, 0.002))), index=idx)
    prices = pd.DataFrame({"A": px, "B": px})
    anchor = px.copy()
    listings = {"A": "2024-01-02", "B": "2024-01-02"}

    def ew_fn(listed, _asof):
        return mod.equal_weights(listed), None

    eq, meta = mod.simulate_monthly_rebalance(
        prices,
        anchor,
        weight_fn=ew_fn,
        start="2024-01-02",
        end=idx[-1].strftime("%Y-%m-%d"),
        listings=listings,
        cost_per_side=0.0,
        scheme="equal",
    )
    assert len(eq) > 10
    # With zero cost and identical assets, portfolio ≈ asset ≈ anchor → excess ~0
    assert abs(meta["excess"]) < 0.02
    assert meta["bh"] > 0


def test_layer1_need_period():
    mod = _load_mod()
    need = mod.period_need_from_ann(0.043, 252)
    assert abs(need - 0.043) < 1e-6
    idx = pd.bdate_range("2024-01-02", periods=50)
    close = pd.Series(np.linspace(100, 110, 50), index=idx)
    l1 = mod.layer1_anchor(
        close,
        train_start="2024-01-02",
        train_end="2024-02-29",
        oos_start="2024-03-01",
        oos_end="2024-03-29",
        erp_ann=0.043,
    )
    assert set(l1["split"]) == {"train", "oos"}
    assert l1.loc[l1["split"] == "train", "bh"].iloc[0] > 0
