from __future__ import annotations

import numpy as np
import pandas as pd

from decision_pack.src.state_space_regime import (
    StateSpaceRegimeConfig,
    apply_ma_kalman_confirm,
    apply_state_space_regime,
    fit_kalman_confirm_model,
    fit_state_space_regime,
)
from decision_pack.src.state_space_selection import (
    beta_binomial_rate,
    choose_candidate,
    evaluate_candidate_folds,
    expanding_splits,
    fit_final_candidate,
    grouped_block_bootstrap_delta,
    hold_up_trade_metrics,
)


def _frame(returns: np.ndarray, start: str = "2020-01-01") -> pd.DataFrame:
    close = 100.0 * np.exp(np.cumsum(np.r_[0.0, returns]))
    return pd.DataFrame(
        {
            "as_of": pd.bdate_range(start, periods=len(close)),
            "$close": close,
        }
    )


def test_kalman_local_trend_recognizes_persistent_direction():
    rng = np.random.default_rng(7)
    returns = np.r_[
        rng.normal(0.003, 0.002, 120),
        rng.normal(0.0, 0.001, 80),
        rng.normal(-0.003, 0.002, 120),
    ]
    px = _frame(returns)
    train = px.iloc[:220]
    cfg = StateSpaceRegimeConfig(
        method="kalman_local_trend",
        direction_prob=0.65,
        confirm_days=2,
    )
    model = fit_state_space_regime(train, cfg)
    out = apply_state_space_regime(px, model)
    assert (out["regime"].iloc[40:110] == "up").mean() > 0.7
    assert (out["regime"].iloc[-80:] == "down").mean() > 0.7


def test_appending_future_does_not_change_filtered_history():
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0005, 0.012, 360)
    px = _frame(returns)
    cfg = StateSpaceRegimeConfig(method="hmm_kalman_gate", hmm_emission="normal")
    model = fit_state_space_regime(px.iloc[:240], cfg)
    first = apply_state_space_regime(px.iloc[:300], model)
    full = apply_state_space_regime(px, model)
    cols = ["regime", "kalman_slope_z", "p_up_state", "p_high_vol_state"]
    for col in cols:
        if col == "regime":
            assert first[col].tolist() == full[col].iloc[:300].tolist()
        else:
            assert np.allclose(
                first[col].to_numpy(float),
                full[col].iloc[:300].to_numpy(float),
                equal_nan=True,
            )


def test_hmm_collapse_is_unreliable_and_requests_fallback():
    px = _frame(np.zeros(180))
    cfg = StateSpaceRegimeConfig(
        method="hmm_kalman_gate",
        hmm_emission="normal",
        hmm_min_sigma_ratio=1.5,
    )
    model = fit_state_space_regime(px, cfg)
    assert not model["reliable"]
    assert model["fallback_method"] == "ma_stack_hyst"


def test_short_history_is_unreliable():
    px = _frame(np.array([0.01, -0.01, 0.0]))
    model = fit_state_space_regime(px)
    assert not model["reliable"]
    assert "low_n" in model["flags"]


def test_trade_metrics_exclude_open_trade_from_win_and_charge_cost():
    close = np.array([100.0, 110.0, 108.0, 120.0])
    labels = np.array(["up", "range", "up", "up"], dtype=object)
    metrics, trades, _ = hold_up_trade_metrics(labels, close, cost_per_side=0.001)
    assert metrics["n_closed"] == 1
    assert metrics["wins"] == 1
    assert metrics["win"] == 1.0
    assert metrics["open_pos"]
    assert 0.09 < trades[0]["ret_net"] < 0.10


def test_beta_shrinkage_prevents_one_trade_perfect_score():
    one = beta_binomial_rate(1, 1, prior_rate=0.5, prior_strength=5)
    many = beta_binomial_rate(8, 10, prior_rate=0.5, prior_strength=5)
    assert one < many < 0.8


def test_constrained_choice_falls_back_when_no_candidate_is_eligible():
    aggregate = pd.DataFrame(
        [
            {
                "candidate": "new",
                "n_closed": 1,
                "posterior_win": 0.8,
                "mean_ret_net": 0.1,
                "mean_edge": 0.1,
                "worst_drawdown": -0.1,
            }
        ]
    )
    winner, meta = choose_candidate(
        aggregate, baseline_candidate="baseline", min_closed=2
    )
    assert winner == "baseline"
    assert not meta["accepted"]


def test_grouped_bootstrap_is_reproducible():
    selected = pd.DataFrame(
        {"block_id": ["A", "B", "C"], "win": [1.0, 0.5, 0.7]}
    )
    baseline = pd.DataFrame(
        {"block_id": ["A", "B", "C"], "win": [0.5, 0.5, 0.4]}
    )
    a = grouped_block_bootstrap_delta(
        selected, baseline, metric="win", n_boot=100, seed=3
    )
    b = grouped_block_bootstrap_delta(
        selected, baseline, metric="win", n_boot=100, seed=3
    )
    assert a == b
    assert a["delta"] > 0


def test_fold_boundaries_are_strictly_chronological():
    for train, validation in expanding_splits(
        400, n_folds=4, min_train=160, val_size=50
    ):
        assert train[-1] < validation[0]
        assert not set(train).intersection(validation)


def test_fold_evaluation_refits_and_freezes_state_model():
    class FakeEv:
        METHODS = {"ma_stack_hyst": object()}

        @staticmethod
        def run_method(name, px, params=None):
            close = px["$close"].to_numpy(float)
            ret20 = pd.Series(close).pct_change(20).fillna(0.0).to_numpy()
            return np.where(
                ret20 > 0.02, "up", np.where(ret20 < -0.02, "down", "range")
            )

    rng = np.random.default_rng(31)
    px = _frame(rng.normal(0.0004, 0.008, 300))
    candidates = [
        {
            "key": "ma_stack_hyst",
            "kind": "baseline",
            "method": "ma_stack_hyst",
            "config": {},
        },
        {
            "key": "kalman",
            "kind": "state_space",
            "config": {
                **StateSpaceRegimeConfig(method="kalman_local_trend").__dict__,
            },
        },
    ]
    rows = evaluate_candidate_folds(FakeEv(), px, candidates)
    assert len(rows) >= 4
    assert (rows["train_end_i"] < rows["val_start_i"]).all()
    method, params = fit_final_candidate(FakeEv(), px, candidates[1])
    assert method == "kalman_local_trend"
    assert len(params["model_hash"]) == 16


def test_missing_close_is_filtered_without_future_fill():
    rng = np.random.default_rng(41)
    px = _frame(rng.normal(0.0002, 0.01, 240))
    px.loc[150:153, "$close"] = np.nan
    model = fit_state_space_regime(
        px.iloc[:140],
        StateSpaceRegimeConfig(method="kalman_local_trend"),
    )
    out = apply_state_space_regime(px, model)
    assert len(out) == len(px)
    assert out["regime"].notna().all()


def test_ma_kalman_confirm_blocks_unconfirmed_up():
    rng = np.random.default_rng(51)
    # Strong up then noise: MA marks all up, Kalman should delay early bars.
    returns = np.r_[rng.normal(0.0, 0.01, 40), rng.normal(0.004, 0.002, 120)]
    px = _frame(returns)
    px["MA20"] = px["$close"].rolling(20, min_periods=5).mean()
    px["MA60"] = px["$close"].rolling(60, min_periods=10).mean()
    px["ma_bull"] = px["MA20"] > px["MA60"]
    ma = np.array(["up"] * len(px), dtype=object)
    model = fit_kalman_confirm_model(px.iloc[:100])
    gated, diag = apply_ma_kalman_confirm(
        ma, px, model, confirm_prob=0.65, require_ma_bull=True, soft_exit=True
    )
    assert (gated[:30] == "range").mean() > 0.5
    assert (gated[-40:] == "up").mean() > 0.7
    assert "p_up_state" in diag.columns
