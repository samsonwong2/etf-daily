from __future__ import annotations

import numpy as np
import pandas as pd

from etf_daily.lib.hmm_short_horizon import HmmShortHorizonConfig
from etf_daily.lib.hmm_short_horizon_validation import (
    run_hmm_walk_forward_validation,
)
from etf_daily.lib.short_horizon_validation import build_eval_dates


def _panel_with_regimes(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = np.empty(n, dtype=float)
    state = 0
    for i in range(n):
        if rng.random() < 0.04:
            state = 1 - state
        sigma = 0.006 if state == 0 else 0.028
        rets[i] = rng.normal(0.0, sigma)
    prices = 100.0 * np.exp(np.cumsum(np.insert(rets, 0, 0.0)))
    idx = pd.bdate_range("2020-01-01", periods=len(prices))
    return pd.DataFrame({"TEST": prices}, index=idx)


def test_hmm_walk_forward_runs_and_sums():
    panel = _panel_with_regimes()
    eval_dates = build_eval_dates(
        panel,
        eval_start=pd.Timestamp("2021-06-01"),
        eval_end=panel.index[-15],
        step_days=10,
    )
    cfg = HmmShortHorizonConfig(
        emission="normal",
        mc_samples=300,
        max_em_iter=40,
        min_samples=60,
        train_window=200,
    )
    detail, summary = run_hmm_walk_forward_validation(
        panel,
        eval_dates=eval_dates,
        horizon=5,
        cfg=cfg,
    )
    assert summary.n_obs > 0
    assert 0.0 <= summary.brier_score <= 2.0
    assert abs(summary.mean_pred_p_up + summary.mean_pred_p_down + summary.mean_pred_p_timeout - 1.0) < 1e-6
    assert set(detail["actual_outcome"].unique()).issubset({"up", "down", "timeout"})
