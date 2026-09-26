"""Tests for dual-horizon pool market-price-of-risk vol board."""
from __future__ import annotations

import numpy as np
import pandas as pd

from decision_pack.src.vol_return_board import (
    asof_metrics,
    build_pool_board,
    compute_market_price_of_risk,
    required_annual_return,
    slice_close_through,
    trailing_realized_annual_return,
)


def _gbm(
    n: int,
    *,
    seed: int,
    mu: float,
    sigma: float,
    start: str = "2024-01-02",
) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    rets = mu / 252.0 + sigma / np.sqrt(252.0) * rng.standard_normal(n)
    px = 100.0 * np.cumprod(1.0 + rets)
    return pd.Series(px, index=idx, name="close")


def test_slice_close_is_causal():
    close = _gbm(300, seed=1, mu=0.05, sigma=0.10)
    as_of = close.index[200]
    s = slice_close_through(close, as_of)
    assert s.index.max() <= pd.Timestamp(as_of).normalize()
    poisoned = close.copy()
    poisoned.iloc[-1] = poisoned.iloc[-1] * 10.0
    m0 = asof_metrics(close, as_of)
    m1 = asof_metrics(poisoned, as_of)
    for k in ("rv5", "rv20", "rv60", "r_5a", "r_20a", "r_60a"):
        assert m0[k] == m1[k]


def test_trailing_60d_annualized():
    idx = pd.bdate_range("2024-01-02", periods=61)
    # +10% over 60 intervals → annualized (1.1)**(252/60) - 1
    px = pd.Series(np.linspace(100.0, 110.0, 61), index=idx)
    r = trailing_realized_annual_return(px, bars=60)
    expected = 1.1 ** (252.0 / 60.0) - 1.0
    assert abs(r - expected) < 1e-9


def test_required_return_is_lambda_times_vol():
    lam = 0.4
    assert abs(required_annual_return(vol=0.5, lambda_mkt=lam) - 0.2) < 1e-12


def test_compute_market_price_of_risk_median():
    rows = [
        {"r_60a": 0.10, "rv60": 0.20},
        {"r_60a": 0.20, "rv60": 0.20},
        {"r_60a": 0.30, "rv60": 0.20},
    ]
    por = compute_market_price_of_risk(rows, ret_key="r_60a", vol_key="rv60")
    assert abs(por["lambda_mkt"] - 1.0) < 1e-12


def test_build_pool_board_dual_gaps():
    bond = _gbm(400, seed=2, mu=0.03, sigma=0.04)
    eq_low = _gbm(400, seed=3, mu=0.06, sigma=0.08)
    eq_high = _gbm(400, seed=4, mu=0.00, sigma=0.25)
    as_of = bond.index[-1]
    board, meta = build_pool_board(
        {"SH511260": bond, "SHAAA": eq_low, "SHBBB": eq_high},
        as_of,
        bond_anchor="SH511260",
        defensive_codes={"SH511260"},
    )
    assert {
        "gap_5",
        "gap_20",
        "gap_60",
        "gap_incep",
        "rv5",
        "rv_incep",
        "r_5a",
        "r_incep",
        "r_req_5",
        "r_req_incep",
    } <= set(board.columns)
    lam60 = float(meta["lambda_mkt_60"])
    lam20 = float(meta["lambda_mkt_20"])
    lam5 = float(meta["lambda_mkt_5"])
    lam_i = float(meta["lambda_mkt_incep"])
    req = board.set_index("code")
    for code in ("SHAAA", "SHBBB"):
        assert abs(float(req.loc[code, "r_req_60"]) - lam60 * float(req.loc[code, "rv60"])) < 1e-10
        assert abs(float(req.loc[code, "r_req_20"]) - lam20 * float(req.loc[code, "rv20"])) < 1e-10
        assert abs(float(req.loc[code, "r_req_5"]) - lam5 * float(req.loc[code, "rv5"])) < 1e-10
        assert abs(float(req.loc[code, "r_req_incep"]) - lam_i * float(req.loc[code, "rv_incep"])) < 1e-10
        assert abs(float(req.loc[code, "gap_60"]) - (float(req.loc[code, "r_60a"]) - float(req.loc[code, "r_req_60"]))) < 1e-10
        assert abs(float(req.loc[code, "gap_20"]) - (float(req.loc[code, "r_20a"]) - float(req.loc[code, "r_req_20"]))) < 1e-10
        assert abs(float(req.loc[code, "gap_5"]) - (float(req.loc[code, "r_5a"]) - float(req.loc[code, "r_req_5"]))) < 1e-10
        assert abs(float(req.loc[code, "gap_incep"]) - (float(req.loc[code, "r_incep"]) - float(req.loc[code, "r_req_incep"]))) < 1e-10
    # Compat columns = since-inception set (not 60d).
    assert abs(float(req.loc["SHAAA", "gap"]) - float(req.loc["SHAAA", "gap_incep"])) < 1e-12
    assert abs(float(req.loc["SHAAA", "r_1y"]) - float(req.loc["SHAAA", "r_incep"])) < 1e-12
    assert abs(float(req.loc["SHAAA", "sharpe_like"]) - float(req.loc["SHAAA", "sharpe_like_incep"])) < 1e-12


def test_build_pool_board_without_bond_anchor():
    eq_a = _gbm(400, seed=5, mu=0.08, sigma=0.15)
    eq_b = _gbm(400, seed=6, mu=0.04, sigma=0.20)
    as_of = eq_a.index[-1]
    board, meta = build_pool_board(
        {"SHAAA": eq_a, "SHBBB": eq_b},
        as_of,
        bond_anchor="SH511260",
        defensive_codes={"SH511260"},
    )
    assert meta["bond_present"] is False
    assert np.isfinite(meta["lambda_mkt_60"]) and np.isfinite(meta["lambda_mkt_20"])
    assert np.isfinite(meta["lambda_mkt_5"])
    assert board["vol_vs_bond60"].isna().all()
    assert board["r_req_60"].notna().all()
    assert board["r_req_20"].notna().all()
    assert board["r_req_5"].notna().all()
