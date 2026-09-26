from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from decision_pack.src.garch_dynamics import GarchDynamicsConfig, fit_garch_dynamics
from decision_pack.src.garch_price_targets import (
    build_garch_price_targets_frame,
    build_price_target_row,
    price_from_log_return,
)
from decision_pack.src.garch_short_horizon_board import (
    GarchShortHorizonConfig,
    GarchShortHorizonMetrics,
    compute_garch_short_horizon_metrics,
)


def _close_from_returns(returns: np.ndarray) -> pd.Series:
    prices = 100.0 * np.exp(np.cumsum(np.insert(returns, 0, 0.0)))
    return pd.Series(prices, index=pd.bdate_range("2020-01-01", periods=len(prices)))


def test_price_from_log_return():
    got = price_from_log_return(1.214, 0.0754)
    assert got is not None
    assert got == pytest.approx(1.214 * math.exp(0.0754), rel=1e-4)


def test_build_price_target_row_manual():
    close = 1.214
    barrier = 0.3215
    metrics = GarchShortHorizonMetrics(
        horizon=10,
        sigma_garch_1d=0.05,
        sigma_garch_h=0.16,
        vol_ann_5d=None,
        vol_ann_20d=None,
        vol_ratio_5_20=None,
        nu_t=5.0,
        arch_pvalue=None,
        p_up=0.114,
        p_down=0.002,
        p_timeout=0.884,
        barrier=barrier,
        q05=-0.1666,
        q50=0.0754,
        q95=0.4014,
        var95_cond=None,
        cvar95_cond=None,
        dgp_basis="GJR-t",
        reliable=True,
        basis="GARCH-DGP·GJR-t",
    )
    row = build_price_target_row(
        as_of="2026-07-14",
        code="SH588170",
        name="科创半导",
        weight=0.0835,
        close=close,
        horizon=10,
        metrics=metrics,
    )
    assert row["tp_price"] == pytest.approx(round(close * math.exp(barrier), 4), rel=1e-4)
    assert row["sl_price"] == pytest.approx(round(close * math.exp(-barrier), 4), rel=1e-4)
    assert row["median_price"] == pytest.approx(round(close * math.exp(0.0754), 4), rel=1e-4)
    assert row["p_tp"] == 0.114
    assert row["p_median"] == 0.5


def test_h1_probabilities_sum():
    rng = np.random.default_rng(9)
    returns = rng.normal(0.0, 0.015, size=280)
    close = _close_from_returns(returns)
    end = close.index[-1]
    cfg = GarchShortHorizonConfig(
        price_target_horizons=(1, 5, 10, 20),
        garch=GarchDynamicsConfig(model="ewma", mc_samples=800, min_train=60),
    )
    fit = fit_garch_dynamics(returns[-252:], cfg.garch, max_horizon=20)
    metrics = compute_garch_short_horizon_metrics(
        close, end, horizon=1, cfg=cfg, fit=fit, max_horizon=20
    )
    assert metrics.barrier is not None
    assert metrics.p_up is not None and metrics.p_down is not None and metrics.p_timeout is not None
    assert abs(metrics.p_up + metrics.p_down + metrics.p_timeout - 1.0) < 1e-9


def test_build_frame_four_horizons():
    rng = np.random.default_rng(10)
    returns = rng.normal(0.0, 0.012, size=320)
    close = _close_from_returns(returns)
    cluster = pd.DataFrame([{"code": "TEST01", "name": "Test", "weight": 0.05, "flags": ""}])
    panel = pd.DataFrame({"TEST01": close})
    end = close.index[-1]
    cfg = GarchShortHorizonConfig(
        price_target_horizons=(1, 5, 10, 20),
        garch=GarchDynamicsConfig(model="ewma", mc_samples=500, min_train=60),
    )
    fit = fit_garch_dynamics(returns[-252:], cfg.garch, max_horizon=20)
    frame = build_garch_price_targets_frame(
        cluster,
        panel,
        str(end.date()),
        board_cfg=cfg,
        fit_cache={"TEST01": fit},
        horizons=(1, 5, 10, 20),
        max_horizon=20,
    )
    assert len(frame) == 4
    assert set(frame["H"].tolist()) == {1, 5, 10, 20}
    for _, row in frame.iterrows():
        assert row["tp_price"] > row["close"]
        assert row["sl_price"] < row["close"]
        assert abs(row["p_tp"] + row["p_sl"] + row["p_timeout"] - 1.0) < 1e-9


def test_tp_matches_close_times_exp_barrier():
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0, 0.01, size=300)
    close = _close_from_returns(returns)
    end = close.index[-1]
    p0 = float(close.iloc[-1])
    cfg = GarchShortHorizonConfig(garch=GarchDynamicsConfig(model="ewma", mc_samples=600))
    fit = fit_garch_dynamics(returns[-252:], cfg.garch, max_horizon=10)
    metrics = compute_garch_short_horizon_metrics(
        close, end, horizon=10, cfg=cfg, fit=fit, max_horizon=10
    )
    row = build_price_target_row(
        as_of=str(end.date()),
        code="TEST01",
        name="Test",
        weight=0.0,
        close=p0,
        horizon=10,
        metrics=metrics,
    )
    assert metrics.barrier is not None
    assert row["tp_price"] == round(p0 * math.exp(metrics.barrier), 4)
