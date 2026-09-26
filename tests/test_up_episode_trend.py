"""Tests for fwd-return truth and causal up-episode machines."""
from __future__ import annotations

import numpy as np
import pandas as pd

from etf_daily.lib.causal_trend_machines import (
    MACHINE_IDS,
    machine_above_ma20,
    machine_ma20_hug,
    pred_to_entry_exit,
    run_machine,
)
from etf_daily.lib.event_long_only_sim import simulate_long_only_events
from etf_daily.lib.fwd_return_trend_truth import (
    STATE_DOWN,
    STATE_RANGE,
    STATE_UP,
    classification_scores,
    fwd_return_trend_labels,
    turn_hit_rate,
    up_episode_turns,
)


def test_fwd_labels_on_synthetic_ramp():
    idx = pd.bdate_range("2024-01-02", periods=40)
    # Steady +0.3%/day → 20d forward ~ +6% → up for early bars.
    px = 100.0 * np.cumprod(np.r_[1.0, np.full(39, 1.003)])
    close = pd.Series(px, index=idx)
    lab = fwd_return_trend_labels(close, horizon=20, up_thr=0.02, down_thr=0.02)
    assert lab.iloc[:15].eq(STATE_UP).all()
    assert lab.iloc[-20:].isna().all()


def test_fwd_labels_down_and_range():
    idx = pd.bdate_range("2024-01-02", periods=40)
    px_down = 100.0 * np.cumprod(np.r_[1.0, np.full(39, 0.997)])
    lab_d = fwd_return_trend_labels(
        pd.Series(px_down, index=idx), horizon=20, up_thr=0.02, down_thr=0.02
    )
    assert lab_d.iloc[:10].eq(STATE_DOWN).all()
    flat = pd.Series(100.0, index=idx)
    lab_f = fwd_return_trend_labels(flat, horizon=20)
    assert lab_f.iloc[:15].eq(STATE_RANGE).all()


def test_up_episode_turns_and_hit_rate():
    idx = pd.bdate_range("2024-01-02", periods=12)
    lab = pd.Series(
        [
            STATE_RANGE,
            STATE_RANGE,
            STATE_UP,
            STATE_UP,
            STATE_UP,
            STATE_RANGE,
            STATE_RANGE,
            STATE_UP,
            STATE_UP,
            STATE_DOWN,
            STATE_DOWN,
            STATE_RANGE,
        ],
        index=idx,
    )
    turns = up_episode_turns(lab)
    assert list(turns["enter_up"]) == [idx[2], idx[7]]
    assert list(turns["leave_up"]) == [idx[5], idx[9]]
    hit = turn_hit_rate(
        [idx[2], idx[8]],
        turns["enter_up"],
        tolerance_bars=3,
        calendar=idx,
    )
    assert hit["n_hit"] == 2
    assert hit["hit_rate"] == 1.0


def test_ma20_hug_and_above_ma20():
    idx = pd.bdate_range("2024-01-02", periods=60)
    # Flat then jump 10% and stay — after MA catches up, above_ma20 should be up.
    px = np.full(60, 100.0)
    px[30:] = 110.0
    close = pd.Series(px, index=idx)
    above = machine_above_ma20(close)
    # Once MA20 has fully rolled into 110, close==ma → not strictly >; use later bars
    # where close still 110 and ma still climbing from 100.
    assert above.iloc[35:45].eq(STATE_UP).any()
    hug = machine_ma20_hug(close)
    # Right after jump, dist to MA20 is large → up.
    assert hug.iloc[30] == STATE_UP


def test_pred_to_entry_exit_drives_sim():
    idx = pd.bdate_range("2024-01-02", periods=8)
    pred = pd.Series(
        [
            STATE_RANGE,
            STATE_UP,
            STATE_UP,
            STATE_UP,
            STATE_RANGE,
            STATE_RANGE,
            STATE_UP,
            STATE_UP,
        ],
        index=idx,
    )
    entry, exit_sig = pred_to_entry_exit(pred)
    assert list(entry.astype(bool)) == [False, True, False, False, False, False, True, False]
    assert list(exit_sig.astype(bool)) == [
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    close = pd.Series([100, 101, 102, 103, 104, 105, 106, 107], index=idx, dtype=float)
    metrics, trades, _ = simulate_long_only_events(
        entry, exit_sig, close, dates=idx.strftime("%Y-%m-%d")
    )
    assert metrics["n_closed"] >= 1
    # enter signal on 2024-01-03 (idx[1]) → fill next bar 2024-01-04
    assert trades[0]["entry_date"] == "2024-01-04"


def test_all_machines_registered():
    assert len(MACHINE_IDS) == 6
    close = pd.Series(
        100 * np.cumprod(np.r_[1.0, np.full(79, 1.001)]),
        index=pd.bdate_range("2024-01-02", periods=80),
    )
    for mid in ("ma20_hug", "above_ma20", "above_ma10", "vol_calm_ma20"):
        pred = run_machine(mid, close)
        assert len(pred) == len(close)
        assert set(pred.dropna().unique()).issubset({STATE_UP, STATE_DOWN, STATE_RANGE})


def test_classification_scores():
    idx = pd.bdate_range("2024-01-02", periods=5)
    truth = pd.Series([STATE_UP, STATE_UP, STATE_RANGE, STATE_DOWN, np.nan], index=idx)
    pred = pd.Series([STATE_UP, STATE_RANGE, STATE_RANGE, STATE_DOWN, STATE_UP], index=idx)
    s = classification_scores(pred, truth)
    assert s["n_scored"] == 4
    assert abs(s["accuracy"] - 0.75) < 1e-12
