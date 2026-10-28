"""Pure tests for basis, Black-Scholes, and the overlay decision rule."""
from __future__ import annotations

import numpy as np
import pandas as pd

from etf_daily.derivs.bs import ORACLE_CALL, ORACLE_TOL, bs_call, bs_put, invert_iv
from etf_daily.derivs.features import (
    annualize_basis,
    atm_iv_from_marks,
    basis,
    build_panel,
    days_to_front_expiry,
    front_expiry,
    parse_contract,
    rolling_percentile,
    summarize_risk_day,
    third_friday,
)
from etf_daily.derivs.overlay import adopted_variants, block_masks, oos_improved


def test_oracle_call_price():
    price = bs_call(100, 100, 0.0, 0.20, 1.0)
    assert abs(price - ORACLE_CALL) <= ORACLE_TOL
    assert abs(bs_put(100, 100, 0.0, 0.20, 1.0) - price) < 1e-9


def test_invert_recovers_oracle_vol():
    price = bs_call(100, 100, 0.0, 0.20, 1.0)
    sigma = invert_iv(price, 100, 100, 0.0, 1.0, kind="call")
    assert sigma is not None
    assert abs(sigma - 0.20) < 1e-3
    again = bs_call(100, 100, 0.0, sigma, 1.0)
    assert abs(again - price) <= ORACLE_TOL


def test_invert_rejects_bad_prices():
    assert invert_iv(-1, 100, 100, 0.0, 1.0) is None
    assert invert_iv(0, 100, 100, 0.0, 1.0) is None
    assert invert_iv(float("nan"), 100, 100, 0.0, 1.0) is None
    assert invert_iv(250, 100, 100, 0.0, 1.0) is None


def test_atm_iv_from_marks_matches_known_vol():
    spot, strike, rate, tau, sigma = 3.0, 3.0, 0.015, 30 / 365, 0.22
    call = bs_call(spot, strike, rate, sigma, tau)
    put = bs_put(spot, strike, rate, sigma, tau)
    found = atm_iv_from_marks(call, put, spot, strike, tau, rate)
    assert found is not None
    assert abs(found - sigma) < 1e-3


def test_basis_and_annualize():
    assert abs(basis(101.0, 100.0) - 0.01) < 1e-12
    assert np.isnan(basis(0.0, 100.0))
    assert abs(annualize_basis(0.01, 365) - 0.01) < 1e-12
    day = pd.Timestamp("2024-03-01")
    assert third_friday(2024, 3) == pd.Timestamp("2024-03-15")
    assert front_expiry(day) == pd.Timestamp("2024-03-15")
    assert days_to_front_expiry(day) == 14
    assert front_expiry(pd.Timestamp("2024-03-14")) == pd.Timestamp("2024-04-19")


def test_percentile_ignores_the_future():
    values = pd.Series(np.linspace(-0.02, 0.03, 80))
    window, min_periods = 20, 20
    left = rolling_percentile(values, window, min_periods)
    shocked = values.copy()
    shocked.iloc[-1] = 10.0
    right = rolling_percentile(shocked, window, min_periods)
    pd.testing.assert_series_equal(left.iloc[:-1], right.iloc[:-1])
    at = 40
    window_values = values.iloc[at - window + 1 : at + 1].to_numpy()
    expected = float(np.mean(window_values <= window_values[-1]))
    assert abs(left.iloc[at] - expected) < 1e-12


def test_roll_day_is_blank():
    index = pd.bdate_range("2024-06-03", periods=6)
    future = pd.Series([100, 101, 112, 112, 113, 113], index=index, dtype=float)
    spot = pd.Series([100, 101, 101, 102, 102, 103], index=index, dtype=float)
    etf = spot.copy()
    panel = build_panel(etf, future, spot, window=5, min_periods=3)
    assert bool(panel["roll"].iloc[2])
    assert np.isnan(panel["basis"].iloc[2])
    assert np.isnan(panel["basis_ann"].iloc[2])
    assert abs(panel["basis"].iloc[1] - (101 / 101 - 1)) < 1e-12


def test_chain_picks_front_month_atm_and_skew():
    rows = pd.DataFrame(
        [
            _row("510050C2407M02500", "2024-07-01", "0.50", "0.20"),
            _row("510050C2407M02700", "2024-07-01", "0.25", "0.22"),
            _row("510050P2407M02500", "2024-07-01", "-0.50", "0.21"),
            _row("510050P2407M02300", "2024-07-01", "-0.25", "0.28"),
            _row("510050C2408M02500", "2024-07-01", "0.50", "0.40"),
        ]
    )
    summary = summarize_risk_day(rows)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["underlying"] == "510050"
    assert abs(row["atm_iv"] - 0.205) < 1e-12
    assert abs(row["skew"] - 0.06) < 1e-12
    parsed = parse_contract("510050C2407M02250")
    assert parsed is not None
    assert parsed["strike"] == 2.25
    assert parsed["expiry"] == pd.Timestamp("2024-07-19")


def test_block_mask_leaves_missing_features_alone():
    index = pd.bdate_range("2024-01-01", periods=3)
    panel = pd.DataFrame(
        {
            "iv_pct": [0.9, np.nan, 0.1],
            "basis_pct": [0.5, 0.1, np.nan],
            "witching": [False, False, True],
            "limit_day": [False, True, False],
        },
        index=index,
    )
    masks = block_masks(panel)
    assert masks["挡住高隐含波动率"].tolist() == [True, False, False]
    assert masks["挡住深贴水"].tolist() == [False, True, False]
    assert masks["挡住三巫日或涨跌停"].tolist() == [False, True, True]


def test_month_ends_start_at_the_first_option_month():
    from etf_daily.derivs.fetch import month_ends

    dates = pd.to_datetime(["2015-01-30", "2015-02-02", "2015-02-27", "2015-03-02"])
    assert month_ends(dates, start="2015-02-01") == [
        pd.Timestamp("2015-02-27"),
        pd.Timestamp("2015-03-02"),
    ]


def test_oos_rule_needs_a_real_improvement_and_enough_trades():
    base = {"n_trades": 10, "mdd": -0.30, "calmar": 0.40}
    shallower = {"n_trades": 8, "mdd": -0.20, "calmar": 0.40}
    richer = {"n_trades": 6, "mdd": -0.30, "calmar": 0.46}
    gutted = {"n_trades": 4, "mdd": -0.05, "calmar": 2.0}
    flat = {"n_trades": 10, "mdd": -0.295, "calmar": 0.44}
    assert oos_improved(base, shallower)
    assert oos_improved(base, richer)
    assert not oos_improved(base, gutted)
    assert not oos_improved(base, flat)
    improved = {
        "SH510300": {"基线": False, "挡住高隐含波动率": True},
        "SH510050": {"基线": False, "挡住高隐含波动率": True},
        "SH510500": {"基线": False, "挡住高隐含波动率": False},
    }
    assert adopted_variants(improved) == ["挡住高隐含波动率"]
    improved["SH510050"]["挡住高隐含波动率"] = False
    assert adopted_variants(improved) == []


def _row(contract: str, day: str, delta: str, iv: str) -> dict[str, str]:
    return {
        "CONTRACT_ID": contract,
        "TRADE_DATE": day,
        "DELTA_VALUE": delta,
        "IMPLC_VOLATLTY": iv,
    }
