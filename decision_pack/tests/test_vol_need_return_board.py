"""Tests for dual-track vol need-return board."""
from __future__ import annotations

import numpy as np
import pandas as pd

from decision_pack.src.vol_need_return_board import (
    ANCHOR_ERP_ANN,
    EQUITY_ANCHOR,
    HORIZON_5,
    HORIZON_20,
    TARGET_SHARPE_LIKE,
    ann_to_h_sigma,
    build_vol_need_board,
    downside_bands,
    fair_need_ann_series,
    gap_realized_minus_need,
    garch_h_to_ann,
    implied_sharpe_from_anchor,
    need_returns,
    period_return_from_ann,
    realized_simple_return,
    ticket_stats,
)


def test_implied_sharpe_from_anchor():
    assert abs(implied_sharpe_from_anchor(0.20, erp_ann=0.07) - 0.35) < 1e-12
    assert np.isnan(implied_sharpe_from_anchor(0.0, erp_ann=0.07))
    assert np.isnan(implied_sharpe_from_anchor(float("nan"), erp_ann=0.07))


def test_fair_need_ann_series_scales_with_vol_ratio():
    idx = pd.bdate_range("2026-01-02", periods=5)
    anchor = pd.Series([0.20, 0.20, 0.20, 0.20, 0.20], index=idx)
    same = fair_need_ann_series(anchor, anchor, erp_ann=0.07)
    assert np.allclose(same.to_numpy(dtype=float), 0.07)
    hi = pd.Series([0.40, 0.40, 0.40, 0.40, 0.40], index=idx)
    doubled = fair_need_ann_series(hi, anchor, erp_ann=0.07)
    assert np.allclose(doubled.to_numpy(dtype=float), 0.14)
    bad_anchor = pd.Series([0.20, 0.0, np.nan, -0.1, 0.20], index=idx)
    mixed = fair_need_ann_series(hi, bad_anchor, erp_ann=0.07)
    assert abs(float(mixed.iloc[0]) - 0.14) < 1e-12
    assert np.isnan(float(mixed.iloc[1]))
    assert np.isnan(float(mixed.iloc[2]))
    assert np.isnan(float(mixed.iloc[3]))
    assert abs(float(mixed.iloc[4]) - 0.14) < 1e-12


def test_gap_realized_minus_need_is_causal_20d():
    idx = pd.bdate_range("2026-06-01", periods=25)
    close = pd.Series(np.linspace(100.0, 75.0, 25), index=idx)
    r_need_ann = pd.Series(np.full(25, 0.342), index=idx)
    out = gap_realized_minus_need(close, r_need_ann, bars=HORIZON_20)
    assert np.isnan(float(out["r_realized"].iloc[19]))
    r20 = float(close.iloc[24] / close.iloc[4] - 1.0)
    need20 = float((1.0 + 0.342) ** (20.0 / 252.0) - 1.0)
    assert abs(float(out["r_realized"].iloc[24]) - r20) < 1e-12
    assert abs(float(out["r_need_period"].iloc[24]) - need20) < 1e-12
    assert abs(float(out["gap"].iloc[24]) - (r20 - need20)) < 1e-12
    assert r20 < 0.0
    assert float(out["gap"].iloc[24]) < r20  # need is positive, so gap is more negative
    direct = realized_simple_return(close, bars=20)
    assert abs(float(direct.iloc[24]) - r20) < 1e-12
    p = period_return_from_ann(r_need_ann, bars=20)
    assert abs(float(p.iloc[0]) - need20) < 1e-12
    out5 = gap_realized_minus_need(close, r_need_ann, bars=HORIZON_5)
    r5 = float(close.iloc[24] / close.iloc[19] - 1.0)
    need5 = float((1.0 + 0.342) ** (5.0 / 252.0) - 1.0)
    assert abs(float(out5["r_realized"].iloc[24]) - r5) < 1e-12
    assert abs(float(out5["gap"].iloc[24]) - (r5 - need5)) < 1e-12


def test_build_vol_need_board_spx_erp_scales_need():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2024-01-02", periods=400)
    rets_spx = 0.0002 + 0.01 * rng.standard_normal(400)
    rets_hi = 0.0002 + 0.02 * rng.standard_normal(400)
    px_spx = pd.Series(100.0 * np.cumprod(1.0 + rets_spx), index=idx)
    px_hi = pd.Series(100.0 * np.cumprod(1.0 + rets_hi), index=idx)
    px_bond = pd.Series(100.0 * np.cumprod(1.0 + 0.00005 + 0.001 * rng.standard_normal(400)), index=idx)
    board, meta = build_vol_need_board(
        {EQUITY_ANCHOR: px_spx, "SHAAA": px_hi, "SH511260": px_bond},
        idx[-1],
        bond_anchor="SH511260",
        defensive_codes={"SH511260"},
        include_mc=False,
    )
    assert meta["sharpe_mode"] == "spx_erp"
    assert meta["erp_ann"] == ANCHOR_ERP_ANN
    spx = board.set_index("code").loc[EQUITY_ANCHOR]
    hi = board.set_index("code").loc["SHAAA"]
    bond = board.set_index("code").loc["SH511260"]
    assert abs(float(spx["r_need_ann_20_garch"]) - ANCHOR_ERP_ANN) < 1e-10
    expected_hi = ANCHOR_ERP_ANN * float(hi["rv20_garch_ann"]) / float(spx["rv20_garch_ann"])
    assert abs(float(hi["r_need_ann_20_garch"]) - expected_hi) < 1e-10
    assert np.isnan(float(bond["r_need_ann_20_garch"]))
    assert np.isnan(float(bond["p_ge_need_20d_garch"]))


def test_build_vol_need_board_spx_mode_requires_anchor():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2024-01-02", periods=400)
    rets = 0.0002 + 0.01 * rng.standard_normal(400)
    px = pd.Series(100.0 * np.cumprod(1.0 + rets), index=idx)
    try:
        build_vol_need_board({"SHAAA": px}, idx[-1], include_mc=False)
    except ValueError as exc:
        assert EQUITY_ANCHOR in str(exc)
    else:
        raise AssertionError("expected missing-anchor ValueError")


def test_need_returns_ann_is_sharpe_times_sigma():
    out = need_returns(0.20, bars=20, sharpe_like=1.33)
    assert abs(out["r_need_ann"] - 1.33 * 0.20) < 1e-12


def test_need_returns_period_roundtrip():
    sigma = 0.25
    bars = 5
    out = need_returns(sigma, bars=bars, sharpe_like=TARGET_SHARPE_LIKE)
    r_ann = out["r_need_ann"]
    period = out["r_need_period"]
    back = (1.0 + period) ** (252.0 / bars) - 1.0
    assert abs(back - r_ann) < 1e-10


def test_garch_h_to_ann():
    # H-day cum sigma 0.05 over 5 days → ann = 0.05 * sqrt(252/5)
    got = garch_h_to_ann(0.05, bars=5)
    assert abs(got - 0.05 * np.sqrt(252.0 / 5.0)) < 1e-12
    assert np.isnan(garch_h_to_ann(None, bars=5))


def test_ann_to_h_sigma_roundtrip():
    ann = 0.40
    h = ann_to_h_sigma(ann, bars=5)
    assert abs(garch_h_to_ann(h, bars=5) - ann) < 1e-12


def test_downside_bands_expm1():
    bands = downside_bands(0.10, z_var=1.65)
    assert abs(bands["down_1sig"] - np.expm1(-0.10)) < 1e-12
    assert abs(bands["down_var95"] - np.expm1(-1.65 * 0.10)) < 1e-12
    assert abs(bands["down_2sig"] - np.expm1(-0.20)) < 1e-12


def test_build_vol_need_board_persist_columns():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2024-01-02", periods=400)
    rets = 0.0002 + 0.01 * rng.standard_normal(400)
    px = pd.Series(100.0 * np.cumprod(1.0 + rets), index=idx)
    board, meta = build_vol_need_board(
        {"SHAAA": px, "SH511260": px * 1.0},
        idx[-1],
        bond_anchor="SH511260",
        defensive_codes={"SH511260"},
        names={"SHAAA": "测试", "SH511260": "债"},
        sharpe_like=1.33,
        include_mc=False,
    )
    assert meta["sharpe_target"] == 1.33
    assert meta["include_mc"] is False
    assert {
        "rv5",
        "rv20",
        "r_need_ann_5_persist",
        "r_need_5d_persist",
        "r_need_ann_20_persist",
        "r_need_20d_persist",
        "down_var95_5d_persist",
        "down_var95_20d_persist",
        "rv5_garch_ann",
        "rv20_garch_ann",
        "r_need_ann_5_garch",
        "r_need_5d_garch",
        "r_need_ann_20_garch",
        "r_need_20d_garch",
        "down_var95_5d_garch",
        "down_var95_20d_garch",
        "q05_5d_garch",
        "q05_20d_garch",
        "p_up_5d_garch",
        "e_up_5d_garch",
        "e_down_5d_garch",
        "match_5d_garch",
        "p_ge_need_5d_garch",
        "p_up_20d_garch",
        "e_up_20d_garch",
        "e_down_20d_garch",
        "match_20d_garch",
        "p_ge_need_20d_garch",
    } <= set(board.columns)
    row = board.set_index("code").loc["SHAAA"]
    assert abs(float(row["r_need_ann_5_persist"]) - 1.33 * float(row["rv5"])) < 1e-10
    assert abs(float(row["r_need_ann_20_persist"]) - 1.33 * float(row["rv20"])) < 1e-10
    # persist downside consistent with rv → H-day σ
    s5 = ann_to_h_sigma(float(row["rv5"]), bars=5)
    assert abs(float(row["sigma5_persist"]) - s5) < 1e-10
    assert abs(float(row["down_var95_5d_persist"]) - float(np.expm1(-1.65 * s5))) < 1e-10
    # skip-mc → ticket NaN
    assert np.isnan(float(row["p_up_5d_garch"]))


def test_ticket_stats_basic():
    rng = np.random.default_rng(0)
    r = rng.normal(0.01, 0.05, size=5000)
    t = ticket_stats(r, need=0.0)
    assert 0.4 < t["p_up"] < 0.8
    assert t["e_up"] > 0
    assert t["e_down"] < 0
    assert t["match"] > 0
    assert 0.0 < t["p_ge_need"] < 1.0


def test_build_vol_need_board_ticket_with_mc():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2024-01-02", periods=400)
    rets = 0.0002 + 0.01 * rng.standard_normal(400)
    px = pd.Series(100.0 * np.cumprod(1.0 + rets), index=idx)
    board, meta = build_vol_need_board(
        {"SHAAA": px},
        idx[-1],
        bond_anchor="SH511260",
        defensive_codes=set(),
        include_mc=True,
        sharpe_like=1.33,
    )
    assert meta["include_mc"] is True
    row = board.iloc[0]
    assert 0.0 <= float(row["p_up_5d_garch"]) <= 1.0
    assert float(row["e_up_5d_garch"]) > 0
    assert float(row["e_down_5d_garch"]) < 0
    assert float(row["q05_5d_garch"]) < 0
    assert float(row["match_5d_garch"]) > 0
    assert 0.0 <= float(row["p_ge_need_5d_garch"]) <= 1.0
