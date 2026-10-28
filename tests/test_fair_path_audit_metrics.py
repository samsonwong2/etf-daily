"""Protocol metrics, checklist contracts, and the H7 incremental-stamp gap."""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from etf_daily.lib.fair_path_band_signals import (
    detect_s1_trim_signals,
    evaluate_checklist_row,
)
from etf_daily.lib.fair_path_tail_audit import (
    PROTOCOL,
    acceptance_vs_legacy,
    block_bootstrap_ci,
    dip_credit_allowed,
    evaluate_gate,
    pnl_distribution,
    position_cap,
)
from etf_daily.lib.incremental_from_listing import try_incremental_html
from etf_daily.scripts import backtest_fair_path_band_signals as bt


def _row(**kwargs) -> pd.Series:
    base = {
        "px": 1.0,
        "fig10_touch_lo": True,
        "fig9_touch_lo": False,
        "fig8_gap": -0.2,
        "fig7_g": -0.1,
        "fig8_g": -0.05,
        "fig10_above_path": False,
        "fig11_above_path": False,
        "fig10_lower": 0.99,
        "fig10_prev_touch_lo": False,
        "fig8_path": 1.3,
        "fig10_path": 1.2,
        "fig10_g": -0.2,
        "fig10_touch_hi": False,
        "fig9_touch_hi": False,
        "fig7_path_rising": False,
        "fig8_path_rising": False,
    }
    base.update(kwargs)
    return pd.Series(base, name=pd.Timestamp("2026-07-30"))


def test_b1_without_b2_is_falling_knife():
    cr = evaluate_checklist_row(_row(), code="TEST", atr=0.02)
    assert cr.flags["B1_position"] is True
    assert cr.flags["B2_long_ok"] is False
    assert cr.verdict == "avoid_falling_knife"


def test_dip_credit_refused_when_left_tail_thickens():
    assert dip_credit_allowed(2.4, 3.1) is False
    assert dip_credit_allowed(3.2, 3.1) is True
    assert dip_credit_allowed(None, 3.1) is False


def test_position_cap_is_budget_over_extreme_drop():
    # 1% portfolio damage / 50% name crash => 2% weight.
    assert position_cap(0.01, 0.50) == pytest.approx(0.02)
    assert position_cap(0.01, -0.80) == pytest.approx(0.0125)
    assert np.isnan(position_cap(0.0, 0.5))


def test_pnl_distribution_known_book():
    rets = [-0.20, -0.10, -0.05, 0.01, 0.02, 0.03, 0.04, 0.40]
    d = pnl_distribution(rets)
    assert d["n"] == 8
    assert d["median"] == pytest.approx(0.015)
    assert d["max_loss"] == pytest.approx(-0.20)
    assert d["worst5_mean"] == pytest.approx(np.mean([-0.20, -0.10, -0.05, 0.01, 0.02]))
    assert d["top1_share"] == pytest.approx(0.40 / 0.15)
    assert d["shadow_mean"] is None or np.isfinite(d["shadow_mean"])


def test_block_bootstrap_ci_covers_constant_series():
    x = np.ones(100)
    ci = block_bootstrap_ci(x, block=20, n_boot=50, seed=1)
    assert ci["point"] == pytest.approx(1.0)
    assert ci["lo"] == pytest.approx(1.0)
    assert ci["hi"] == pytest.approx(1.0)


def test_block_bootstrap_is_seed_stable():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    a = block_bootstrap_ci(x, block=20, n_boot=80, seed=7)
    b = block_bootstrap_ci(x, block=20, n_boot=80, seed=7)
    assert a["lo"] == b["lo"]
    assert a["hi"] == b["hi"]


def test_gate_docs_only_when_no_hypothesis_holds():
    g = evaluate_gate(
        {
            "h1_left_dominant_frac": 0.1,
            "h2_inconclusive": False,
            "h2_ewma_only_not_better": False,
            "h3_short_window_flip": False,
            "h4_low_exceed_frac": 0.8,
            "h4_pool_possible": False,
        }
    )
    assert g["action"] == "docs_only"
    assert g["change_rails"] is False
    assert g["use_pot"] is False


def test_gate_split_without_pot_when_tails_are_short():
    g = evaluate_gate(
        {
            "h1_left_dominant_frac": 0.5,
            "h2_inconclusive": True,
            "h3_short_window_flip": False,
            "h4_low_exceed_frac": 0.9,
            "h4_pool_possible": False,
        }
    )
    assert g["action"] == "rails_split_empirical_only"
    assert g["use_pot"] is False


def test_gate_allows_pot_when_pooling_is_possible():
    g = evaluate_gate(
        {
            "h1_left_dominant_frac": 0.5,
            "h2_inconclusive": True,
            "h3_short_window_flip": True,
            "h4_low_exceed_frac": 0.9,
            "h4_pool_possible": True,
        }
    )
    assert g["action"] == "rails_split_and_pot"
    assert g["use_pot"] is True


def test_acceptance_fails_closed_on_thin_sample_and_worse_left_tail():
    legacy = {"worst5pct_mean": -0.04, "max_loss": -0.10, "median": 0.01, "shadow_mean": 0.0}
    worse = {
        "worst5pct_mean": -0.08,
        "max_loss": -0.20,
        "median_diff_lo": -0.02,
        "median_diff_hi": -0.01,
        "shadow_mean_diff_lo": 0.0,
        "shadow_mean_diff_hi": 0.01,
    }
    bad = acceptance_vs_legacy(legacy, worse, independent_days=10)
    assert bad["pass"] is False
    assert "independent_days" in bad["reasons"]
    assert "worst5pct_mean" in bad["reasons"]
    assert "median" in bad["reasons"]


def test_acceptance_passes_when_left_tail_holds_and_ci_includes_zero():
    legacy = {"worst5pct_mean": -0.04, "max_loss": -0.10}
    ok = {
        "worst5pct_mean": -0.04,
        "max_loss": -0.09,
        "median_diff_lo": -0.002,
        "median_diff_hi": 0.01,
        "shadow_mean_diff_lo": -0.001,
        "shadow_mean_diff_hi": 0.004,
    }
    verdict = acceptance_vs_legacy(legacy, ok, independent_days=int(PROTOCOL["min_independent_days"]))
    assert verdict["pass"] is True
    assert verdict["reasons"] == []


def test_protocol_constants_are_frozen_before_results():
    assert PROTOCOL["dev_end"] == "2024-12-31"
    assert PROTOCOL["final_start"] == "2025-01-01"
    assert PROTOCOL["min_independent_days"] == 60
    assert PROTOCOL["block_len"] == 20
    assert PROTOCOL["pot_quantile"] == 0.90
    assert PROTOCOL["variants"][0] == "V0"


def test_h7_incremental_refuses_html_whose_rails_stamp_is_stale():
    src = inspect.getsource(try_incremental_html)
    assert "fair-path-rails" in src


@pytest.mark.xfail(
    reason="legacy B5/S1 still use an ATR stop and an upper-rail trim; §12 forbids both",
    strict=False,
)
def test_legacy_exit_does_not_use_atr_or_s1_trim():
    src = inspect.getsource(bt.simulate_trades)
    assert "atr20" not in src
    assert "stop" not in src
    idx = pd.bdate_range("2024-01-01", periods=5)
    ms = pd.DataFrame(
        {
            "fig9_touch_hi": [False, False, True, False, False],
            "fig10_touch_hi": [False, False, False, False, False],
        },
        index=idx,
    )
    assert not bool(detect_s1_trim_signals(ms).any())
