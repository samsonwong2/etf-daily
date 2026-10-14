"""Unit tests for HRP domain / min-corr / medoid helpers (no qlib)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from etf_daily.hrp.dendrogram import (
    DOMAIN_BOND,
    DOMAIN_DOMESTIC,
    DOMAIN_OVERSEAS,
    classify_cluster_domain,
    corr_medoid,
    split_by_min_corr,
    ward_labels,
)


def test_classify_cluster_domain():
    assert classify_cluster_domain("SH511010", "国债ETF国泰") == DOMAIN_BOND
    assert classify_cluster_domain("SH513100", "纳指ETF国泰") == DOMAIN_OVERSEAS
    assert classify_cluster_domain("SZ159509", "纳指科技ETF景顺") == DOMAIN_OVERSEAS
    assert classify_cluster_domain("SH513650", "标普500ETF南方") == DOMAIN_OVERSEAS
    assert classify_cluster_domain("SH520870", "巴西ETF易方达") == DOMAIN_OVERSEAS
    assert classify_cluster_domain("SH513050", "中概互联网ETF易方达") == DOMAIN_DOMESTIC
    assert classify_cluster_domain("SH513190", "港股通金融ETF华夏") == DOMAIN_DOMESTIC
    assert classify_cluster_domain("SH515880", "通信ETF国泰") == DOMAIN_DOMESTIC
    assert classify_cluster_domain("SH518880", "黄金ETF华安") == DOMAIN_DOMESTIC


def test_corr_medoid_picks_center():
    # A and B highly correlated; C weakly correlated to both → medoid of {A,B,C} is A or B
    rng = np.random.default_rng(0)
    n = 80
    a = rng.normal(size=n)
    b = a + rng.normal(scale=0.1, size=n)
    c = rng.normal(size=n)
    returns = pd.DataFrame({"A": a, "B": b, "C": c})
    corr = returns.corr()
    med = corr_medoid(["A", "B", "C"], corr)
    assert med in {"A", "B"}
    assert corr_medoid(["C"], corr) == "C"


def test_split_by_min_corr_breaks_weak_cluster():
    # Cluster 1: A-B strong, C weak to both → should split C out
    corr = pd.DataFrame(
        {
            "A": [1.0, 0.9, 0.2],
            "B": [0.9, 1.0, 0.25],
            "C": [0.2, 0.25, 1.0],
        },
        index=["A", "B", "C"],
    )
    labels = pd.Series({"A": 1, "B": 1, "C": 1})
    out = split_by_min_corr(labels, corr, min_corr=0.55)
    assert out["A"] == out["B"]
    assert out["C"] != out["A"]
    # disabled
    same = split_by_min_corr(labels, corr, min_corr=0.0)
    assert same.nunique() == 1


def test_ward_labels_singleton_and_pair():
    rng = np.random.default_rng(1)
    n = 60
    x = rng.normal(size=n)
    y = x + rng.normal(scale=0.05, size=n)
    returns = pd.DataFrame({"X": x, "Y": y})
    lab = ward_labels(returns, dist_t=0.4)
    assert set(lab.index) == {"X", "Y"}
    assert lab.nunique() >= 1
    one = ward_labels(pd.DataFrame({"X": x}), dist_t=0.4)
    assert list(one.values) == [1]
