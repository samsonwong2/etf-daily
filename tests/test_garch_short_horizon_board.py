from __future__ import annotations

import numpy as np
import pandas as pd

from etf_daily.lib.fat_tail_risk import slice_log_returns
from etf_daily.lib.garch_dynamics import fit_garch_dynamics
from etf_daily.lib.garch_short_horizon_board import (
    GARCH_SHORT_HORIZON_COLUMNS,
    GarchShortHorizonConfig,
    compute_garch_short_horizon_metrics,
    format_garch_row,
    realized_vol_annualized_log,
)


def _make_close_from_returns(returns: np.ndarray, start: str = "2020-01-01") -> pd.Series:
    prices = 100.0 * np.exp(np.cumsum(np.insert(returns, 0, 0.0)))
    index = pd.bdate_range(start, periods=len(prices))
    return pd.Series(prices, index=index)


def test_compute_garch_short_horizon_metrics_basic():
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.012, size=300)
    close = _make_close_from_returns(returns)
    end = close.index[-1]
    cfg = GarchShortHorizonConfig()
    metrics = compute_garch_short_horizon_metrics(
        close, end, horizon=10, cfg=cfg
    )
    assert metrics.horizon == 10
    assert metrics.n_samples >= 60
    assert metrics.sigma_garch_1d is not None
    assert metrics.p_up is not None
    assert metrics.p_down is not None
    assert metrics.p_timeout is not None
    assert abs(metrics.p_up + metrics.p_down + metrics.p_timeout - 1.0) < 1e-9
    assert metrics.q05 is not None and metrics.q50 is not None and metrics.q95 is not None
    assert metrics.q05 <= metrics.q50 <= metrics.q95
    if metrics.vol5_pct_120d is not None:
        assert 0.0 <= metrics.vol5_pct_120d <= 1.0
    if metrics.p_vol_spike_h5 is not None:
        assert 0.0 <= metrics.p_vol_spike_h5 <= 1.0


def test_format_garch_row_columns():
    rng = np.random.default_rng(0)
    close = _make_close_from_returns(rng.normal(0, 0.01, 300))
    end = close.index[-1]
    metrics = compute_garch_short_horizon_metrics(close, end, horizon=5)
    cluster_row = pd.Series(
        {
            "barbell_leg": "risk_leg",
            "flags": "low_n",
            "tail_regime": "stable",
            "path_state": "pullback",
            "dd_long": -0.12,
        }
    )
    formatters = {
        "pct": lambda v, digits=2: "1.00%" if v is not None else "—",
        "prob": lambda v, digits=1: "50.0%" if v is not None else "—",
        "num": lambda v, digits=2: "1.00" if v is not None else "—",
        "pval": lambda v: "0.0100" if v is not None else "—",
        "flags_str": lambda v: str(v),
        "tail_short": lambda v: "稳定",
        "state_short": lambda v: "回撤",
    }
    row = format_garch_row(
        code="SH510300",
        name="沪深300",
        weight=0.1,
        cluster_row=cluster_row,
        metrics=metrics,
        formatters=formatters,
    )
    assert set(row.keys()) == set(GARCH_SHORT_HORIZON_COLUMNS)
    assert row["持仓"] == "持"
    assert row["H"] == "5"


def test_missing_price_basis():
    empty = pd.Series(dtype=float)
    end = pd.Timestamp("2024-06-01")
    metrics = compute_garch_short_horizon_metrics(
        empty,
        end,
        horizon=10,
        flags=("missing_price",),
    )
    assert metrics.basis == "缺价格·仅展示"
    assert metrics.reliable is False


def test_realized_vol_uses_log_returns():
    rng = np.random.default_rng(9)
    close = _make_close_from_returns(rng.normal(0, 0.01, 120))
    end = close.index[-1]
    vol = realized_vol_annualized_log(close, end, vol_window=20)
    assert np.isfinite(vol)
    assert vol > 0


def test_shared_fit_across_horizons_consistent_sigma1d():
    rng = np.random.default_rng(12)
    close = _make_close_from_returns(rng.normal(0, 0.012, 300))
    end = close.index[-1]
    cfg = GarchShortHorizonConfig()
    returns = slice_log_returns(close, end, lookback_days=cfg.garch.train_window)
    fit = fit_garch_dynamics(returns, cfg.garch, max_horizon=20)
    m5 = compute_garch_short_horizon_metrics(
        close, end, horizon=5, cfg=cfg, fit=fit, max_horizon=20
    )
    m10 = compute_garch_short_horizon_metrics(
        close, end, horizon=10, cfg=cfg, fit=fit, max_horizon=20
    )
    m20 = compute_garch_short_horizon_metrics(
        close, end, horizon=20, cfg=cfg, fit=fit, max_horizon=20
    )
    assert m5.sigma_garch_1d == m10.sigma_garch_1d == m20.sigma_garch_1d
    assert m5.sigma_garch_h is not None and m20.sigma_garch_h is not None
    assert m20.sigma_garch_h >= m5.sigma_garch_h
    assert m5.dgp_basis == m10.dgp_basis == m20.dgp_basis
