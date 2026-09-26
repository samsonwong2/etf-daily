"""Tests for per-code self-train policy."""
from __future__ import annotations

import numpy as np
import pandas as pd

from etf_daily.lib.self_train_policy import (
    POLICY_BH,
    POLICY_CASH,
    POLICY_HOLD_UP,
    evaluate_self_train_policy,
    pick_policy,
)


def test_pick_policy_argmax_and_tiebreak():
    assert pick_policy(0.1, 0.2, 0.0) == POLICY_BH
    assert pick_policy(0.3, 0.1, 0.0) == POLICY_HOLD_UP
    assert pick_policy(-0.1, -0.2, 0.0) == POLICY_CASH
    # tie → bh preferred
    assert pick_policy(0.5, 0.5, 0.0) == POLICY_BH


def test_evaluate_self_train_policy_picks_on_train_only():
    idx = pd.bdate_range("2025-01-02", periods=80)
    # Rising then flat: BH wins on full; craft labs so hold_up loses on train
    close = np.linspace(1.0, 2.0, len(idx))
    labs = np.array(["up"] * 40 + ["range"] * 40, dtype=object)
    # Cut after 50 days so train includes up+range
    ev = evaluate_self_train_policy(
        idx, close, labs, train_cutoff="2025-03-15"
    )
    assert ev["policy"] in {POLICY_HOLD_UP, POLICY_BH, POLICY_CASH}
    assert len(ev["nav_bh"]) == len(idx)
    assert abs(ev["nav_bh"][-1] - close[-1] / close[0]) < 1e-9
    block = {
        k: v
        for k, v in ev.items()
        if k not in {"nav_hold_up", "nav_bh", "nav_cash", "nav_policy", "dates"}
    }
    assert "policy" in block
    assert "train_hold_up" in block
