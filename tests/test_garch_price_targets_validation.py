from __future__ import annotations

import numpy as np
import pandas as pd

from etf_daily.lib.garch_dynamics import GarchDynamicsConfig
from etf_daily.lib.garch_price_targets_validation import (
    _simple_return,
    _strategy_log_return,
    run_garch_price_targets_validation,
)
from etf_daily.lib.garch_short_horizon_board import GarchShortHorizonConfig
from etf_daily.lib.short_horizon_validation import build_eval_dates


def _close_from_returns(returns: np.ndarray) -> pd.Series:
    prices = 100.0 * np.exp(np.cumsum(np.insert(returns, 0, 0.0)))
    return pd.Series(prices, index=pd.bdate_range("2020-01-01", periods=len(prices)))


def test_strategy_log_return_outcomes():
    assert _strategy_log_return("up", 0.1, 0.05) == 0.1
    assert _strategy_log_return("down", 0.1, 0.05) == -0.1
    assert _strategy_log_return("timeout", 0.1, 0.05) == 0.05


def test_walk_forward_produces_rows():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.012, size=400)
    close = _close_from_returns(returns)
    panel = pd.DataFrame({"TEST01": close})
    eval_dates = build_eval_dates(
        panel,
        eval_start=close.index[200],
        eval_end=close.index[-10],
        step_days=10,
    )
    cfg = GarchShortHorizonConfig(
        garch=GarchDynamicsConfig(model="ewma", mc_samples=300, min_train=60),
    )
    detail, summary = run_garch_price_targets_validation(
        panel,
        eval_dates=eval_dates,
        horizon=5,
        cfg=cfg,
    )
    assert summary.n_obs > 0
    assert not detail.empty
    assert abs(detail["p_tp"] + detail["p_sl"] + detail["p_timeout"] - 1.0).max() < 1e-9
