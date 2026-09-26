"""Tests for A→B peak-sell scanner."""
from __future__ import annotations

import pandas as pd

from decision_pack.src.regime_transition_ab_sell_scan import (
    ABSellScanConfig,
    flag_a_candidates,
    flag_b_candidates,
    merge_oos_with_causal_prior,
    scan_ab_sell_candidates,
    write_ab_sell_scan_outputs,
)


def _frame() -> pd.DataFrame:
    # Mirrors SH588710-like peak pairs + distractors.
    return pd.DataFrame(
        {
            "as_of": [
                "2026-06-29",  # A1
                "2026-07-01",  # weak up, not A
                "2026-07-02",  # B1
                "2026-07-09",  # A2
                "2026-07-10",  # B2
                "2026-07-20",  # lone B, no prior A in gap
            ],
            "code": ["SH588710"] * 6,
            "name": ["科创半导体设备ETF华泰柏瑞"] * 6,
            "quantile_switch": [True, True, True, True, True, True],
            "switch_side": ["up", "up", "down", "up", "down", "down"],
            "pred_cdf": [0.976, 0.80, 0.009, 0.948, 0.023, 0.05],
            "return_z": [2.39, 1.2, -2.97, 2.11, -1.68, -1.8],
            "causal_prior_strong": [True, False, True, True, True, True],
            "signal_level": ["OBSERVE"] * 6,
        }
    )


def test_flag_a_b_thresholds():
    cfg = ABSellScanConfig()
    frame = _frame()
    a = flag_a_candidates(frame, cfg)
    b = flag_b_candidates(frame, cfg)
    assert list(frame.loc[a, "as_of"].astype(str)) == [
        "2026-06-29",
        "2026-07-09",
    ]
    assert list(frame.loc[b, "as_of"].astype(str)) == [
        "2026-07-02",
        "2026-07-10",
        "2026-07-20",
    ]


def test_scan_pairs_sh588710_style():
    a_hits, b_hits, pairs = scan_ab_sell_candidates(_frame())
    assert len(a_hits) == 2
    assert len(b_hits) == 3
    assert len(pairs) == 2
    assert list(pairs["a_as_of"].astype(str)) == ["2026-06-29", "2026-07-09"]
    assert list(pairs["b_as_of"].astype(str)) == ["2026-07-02", "2026-07-10"]
    assert pairs["gap_days"].tolist() == [3, 1]
    assert a_hits["paired"].tolist() == [True, True]
    assert b_hits["paired"].tolist() == [True, True, False]


def test_max_gap_excludes_distant_b():
    frame = pd.DataFrame(
        {
            "as_of": ["2026-06-01", "2026-06-20"],
            "code": ["SH1", "SH1"],
            "name": ["x", "x"],
            "quantile_switch": [True, True],
            "switch_side": ["up", "down"],
            "pred_cdf": [0.95, 0.05],
            "return_z": [2.2, -2.0],
        }
    )
    _, _, pairs = scan_ab_sell_candidates(
        frame, cfg=ABSellScanConfig(max_gap_days=5)
    )
    assert pairs.empty
    _, _, pairs2 = scan_ab_sell_candidates(
        frame, cfg=ABSellScanConfig(max_gap_days=20)
    )
    assert len(pairs2) == 1


def test_require_strong_prior_filters():
    frame = _frame()
    frame.loc[frame["as_of"] == "2026-07-09", "causal_prior_strong"] = False
    a_hits, _, pairs = scan_ab_sell_candidates(
        frame, cfg=ABSellScanConfig(require_strong_prior=True)
    )
    assert list(a_hits["as_of"].astype(str)) == ["2026-06-29"]
    assert len(pairs) == 1
    assert str(pairs.iloc[0]["b_as_of"].date()) == "2026-07-02"


def test_merge_oos_with_causal_prior():
    oos = pd.DataFrame(
        {
            "as_of": ["2026-06-29"],
            "code": ["sh588710"],
            "quantile_switch": [True],
            "switch_side": ["up"],
            "pred_cdf": [0.97],
            "return_z": [2.4],
        }
    )
    rev = pd.DataFrame(
        {
            "as_of": ["2026-06-29"],
            "code": ["SH588710"],
            "causal_prior_ret": [0.27],
            "causal_thr_prior": [0.05],
        }
    )
    merged = merge_oos_with_causal_prior(oos, rev)
    assert bool(merged.iloc[0]["causal_prior_strong"]) is True


def test_write_outputs(tmp_path):
    a, b, pairs = scan_ab_sell_candidates(_frame())
    paths = write_ab_sell_scan_outputs(
        tmp_path, a_hits=a, b_hits=b, pairs=pairs, cfg=ABSellScanConfig()
    )
    assert paths["pairs"].exists()
    assert "A→B" in paths["summary"].read_text(encoding="utf-8")
    loaded = pd.read_csv(paths["pairs"])
    assert len(loaded) == 2


def test_buy_mirror_pairs_and_forward_metrics():
    from decision_pack.src.regime_transition_ab_sell_scan import (
        ABBuyScanConfig,
        forward_pair_metrics,
        scan_ab_buy_candidates,
        summarize_pair_metrics,
    )

    frame = pd.DataFrame(
        {
            "as_of": ["2026-07-02", "2026-07-09", "2026-07-15", "2026-07-21"],
            "code": ["SH588710"] * 4,
            "name": ["x"] * 4,
            "quantile_switch": [True, True, True, True],
            "switch_side": ["down", "up", "down", "up"],
            "pred_cdf": [0.009, 0.95, 0.04, 0.996],
            "return_z": [-3.0, 2.1, -1.51, 2.8],
        }
    )
    _, _, pairs = scan_ab_buy_candidates(
        frame, cfg=ABBuyScanConfig(a_max_return_z=-1.5, max_gap_days=7)
    )
    assert len(pairs) == 2
    assert str(pairs.iloc[0]["a_as_of"].date()) == "2026-07-02"
    assert str(pairs.iloc[1]["b_as_of"].date()) == "2026-07-21"

    idx = pd.to_datetime(
        ["2026-07-02", "2026-07-03", "2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09"]
    )
    close = pd.Series([3.8, 3.7, 3.6, 3.7, 3.9, 4.3], index=idx)
    scored = forward_pair_metrics(pairs.iloc[[0]], {"SH588710": close}, side="buy")
    sm = summarize_pair_metrics(scored, side="buy")
    assert sm["n_pairs"] == 1
    assert "mean_ret_1d" in sm
