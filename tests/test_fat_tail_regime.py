"""Tests for §7 tail-structure regime / §12.3 path-state layer and per-name action."""
from __future__ import annotations

import numpy as np
import pandas as pd

from etf_daily.lib.fat_tail_portfolio import (
    _single_day_drop_tiers,
    is_thin_left_tail,
    name_risk_action,
)
from etf_daily.lib.fat_tail_regime import classify_path_state, classify_tail_regime
from etf_daily.lib.fat_tail_risk import FatTailConfig, slice_log_returns


def _thin_row(
    *,
    tail_regime: str = "stable",
    path_state: str = "near_high",
    shrunk: float = 0.001,
    ewma: float = 0.001,
) -> dict:
    return {
        "alpha_left": 5.0,
        "flags": "",
        "shrunk_mean": shrunk,
        "ewma60_mean": ewma,
        "tail_regime": tail_regime,
        "path_state": path_state,
        "xi_left": -0.1,
        "beta_left": 0.01,
        "threshold_left": 0.02,
        "nu_left": 0.075,
    }


def test_action_stable_near_high_holds():
    action, _ = name_risk_action(
        {"tail_regime": "stable", "path_state": "near_high", "oversized": False}
    )
    assert action == "持有"


def test_action_broken_oversized_trims_to_cap():
    action, _ = name_risk_action(
        {"tail_regime": "stable", "path_state": "broken", "oversized": True}
    )
    assert action == "减至预算上限"


def test_action_broken_not_oversized_watch_trim():
    action, _ = name_risk_action(
        {"tail_regime": "stable", "path_state": "broken", "oversized": False}
    )
    assert action == "观察/部分减"


def test_action_confirmed_overrides_to_derisk():
    action, _ = name_risk_action(
        {"tail_regime": "tail_thickening_confirmed", "path_state": "near_high", "oversized": False}
    )
    assert action == "降risk·重估尾"


def test_action_watch_no_add():
    action, _ = name_risk_action(
        {"tail_regime": "tail_thickening_watch", "path_state": "near_high", "oversized": False}
    )
    assert action == "不加仓·盯回撤"


def test_action_oversized_near_high_trims_on_strength():
    action, _ = name_risk_action(
        {"tail_regime": "stable", "path_state": "near_high", "oversized": True}
    )
    assert action == "逢强减至上限"


def test_action_pullback_holds_when_long_window_intact():
    """A transient pullback (long window intact) is not a reduction trigger."""
    action, _ = name_risk_action(
        {"tail_regime": "stable", "path_state": "pullback", "oversized": False}
    )
    assert action == "持有"


def test_action_ignores_cost_basis_and_ma_trend():
    """Action must ignore cost-basis P&L and MA/return signals (no anchoring/forecast)."""
    base = {"tail_regime": "stable", "path_state": "near_high", "oversized": False}
    noisy = {**base, "unrealized_return": -0.5, "ret_20d": -0.3, "dist_ma20_pct": -10.0}
    assert name_risk_action(base) == name_risk_action(noisy)


def test_classify_path_state_broken_pullback_near():
    cfg = FatTailConfig()
    assert classify_path_state({"dd_long": -0.25, "dd_near": -0.05}, cfg) == "broken"
    assert classify_path_state({"dd_long": -0.12, "dd_near": -0.05}, cfg) == "pullback"
    assert classify_path_state({"dd_long": -0.03, "dd_near": -0.01}, cfg) == "near_high"


def test_classify_path_state_near_term_dip_is_pullback_not_broken():
    """Sharp 20d drop but 120d high intact -> pullback, not broken."""
    cfg = FatTailConfig()
    assert classify_path_state({"dd_long": -0.03, "dd_near": -0.25}, cfg) == "pullback"


def test_single_day_spike_does_not_force_watch_regime():
    """One -8% day should not by itself flip tail_regime away from stable."""
    cfg = FatTailConfig(
        lookback_days=400,
        regime_alpha_recent_days=252,
        min_samples=80,
        min_exceed=15,
    )
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.bdate_range("2023-01-01", periods=n)
    rets = rng.normal(0.0002, 0.012, size=n)
    rets[-1] = -0.08
    prices = 100.0 * np.exp(np.cumsum(rets))
    close = pd.Series(prices, index=dates)

    end = dates[-1]
    anchor_returns = slice_log_returns(close, end, lookback_days=cfg.lookback_days)
    anchor_fit_alpha = None
    from etf_daily.lib.fat_tail_risk import effective_min_exceed, fit_tail

    anchor_fit = fit_tail(
        anchor_returns,
        side="left",
        tail_quantile=cfg.tail_quantile,
        min_exceed=effective_min_exceed(
            int(anchor_returns.size),
            tail_quantile=cfg.tail_quantile,
            min_exceed=cfg.min_exceed,
        ),
    )
    if anchor_fit.reliable and anchor_fit.alpha is not None:
        anchor_fit_alpha = float(anchor_fit.alpha)

    from etf_daily.lib.fat_tail_regime import fit_alpha_left_recent

    recent_alpha, recent_ok = fit_alpha_left_recent(close, end, cfg)
    regime = classify_tail_regime(
        close,
        end,
        anchor_alpha=anchor_fit_alpha,
        recent_alpha=recent_alpha if recent_ok else None,
        anchor_reliable=anchor_fit.reliable,
        kappa=0.05,
        flags=(),
        cfg=cfg,
    )
    assert regime == "stable"


def test_var_tiers_unaffected_by_tail_regime_column():
    """drop_tiers depend on full-window GPD params, not tail_regime."""
    cfg = FatTailConfig()
    row_stable = _thin_row(tail_regime="stable")
    row_watch = _thin_row(tail_regime="tail_thickening_watch")
    drops_stable, _ = _single_day_drop_tiers(row_stable, cfg)
    drops_watch, _ = _single_day_drop_tiers(row_watch, cfg)
    assert drops_stable == drops_watch


def test_is_thin_left_tail_high_alpha():
    cfg = FatTailConfig(alpha_thin=4.0)
    assert is_thin_left_tail({"alpha_left": 5.0, "flags": ""}, cfg)


def test_fat_tail_config_regime_aliases():
    from etf_daily.lib.fat_tail_risk import fat_tail_config_from_raw

    cfg = fat_tail_config_from_raw(
        {
            "regime": {
                "alpha_recent_days": 200,
                "kappa_watch_min": 0.2,
                "drawdown_long_days": 150,
                "drawdown_broken_pct": 0.25,
            }
        }
    )
    assert cfg.regime_alpha_recent_days == 200
    assert cfg.regime_kappa_watch_min == 0.2
    assert cfg.drawdown_long_days == 150
    assert cfg.drawdown_broken_pct == 0.25
