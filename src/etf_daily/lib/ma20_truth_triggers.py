"""MA20±2% chart trend as truth + lead/lag trigger scoring.

Truth matches the adaptive HTML row-1 trend state (趋势上涨/下跌/缠绕).
Triggers are causal catalog events scored for hitting enter-up / leave-up
with optional lead (bars before the truth turn).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from etf_daily.lib.causal_trend_machines import machine_ma20_hug, pred_to_entry_exit
from etf_daily.lib.fwd_return_trend_truth import (
    STATE_UP,
    turn_hit_rate,
)
from etf_daily.lib.html_indicator_catalog import TREND_HUG_BAND_PCT


def ma20_hug_truth(
    close: pd.Series,
    *,
    band_pct: float = TREND_HUG_BAND_PCT,
) -> pd.Series:
    """Chart-aligned three-state truth (same definition as HTML trend)."""
    return machine_ma20_hug(close, band_pct=band_pct).rename("ma20_hug_truth")


def truth_entry_exit(truth: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Enter-up / leave-up boolean series from MA20 truth labels."""
    return pred_to_entry_exit(truth)


def _bar_positions(calendar: pd.DatetimeIndex) -> dict[pd.Timestamp, int]:
    cal = pd.DatetimeIndex(pd.to_datetime(calendar)).normalize().unique().sort_values()
    return {d: i for i, d in enumerate(cal)}


def trigger_lead_hits(
    trigger: pd.Series,
    truth_turns: pd.DatetimeIndex | list,
    *,
    calendar: pd.DatetimeIndex,
    max_lead_bars: int = 5,
) -> dict[str, Any]:
    """Hit truth turns if trigger fires in (prev_turn, turn] ∩ [turn-max_lead, turn].

    Using the previous truth turn as a lower bound stops an earlier episode's
    trigger from inflating lead on the next turn. ``mean_lead_bars`` is
    ``turn_pos - earliest_trigger_pos`` on hits (≥0 ⇒ on or before the turn).
    """
    truth = (
        pd.DatetimeIndex(pd.to_datetime(list(truth_turns))).normalize().unique().sort_values()
    )
    trig = trigger.fillna(False).astype(bool)
    trig.index = pd.to_datetime(trig.index).normalize()
    pos = _bar_positions(calendar)
    trig_pos = sorted({pos[d] for d, v in trig.items() if v and d in pos})
    if len(truth) == 0:
        return {
            "n_truth": 0,
            "n_trigger": int(trig.sum()),
            "n_hit": 0,
            "hit_rate": float("nan"),
            "mean_lead_bars": float("nan"),
            "median_lead_bars": float("nan"),
        }
    leads: list[int] = []
    hits = 0
    truth_pos_list = [pos[t] for t in truth if t in pos]
    for i, tp in enumerate(truth_pos_list):
        prev_bound = truth_pos_list[i - 1] + 1 if i > 0 else -10**9
        lo = max(tp - int(max_lead_bars), prev_bound)
        matched = [pp for pp in trig_pos if lo <= pp <= tp]
        if matched:
            hits += 1
            leads.append(int(tp - min(matched)))
    return {
        "n_truth": int(len(truth_pos_list)),
        "n_trigger": int(trig.sum()),
        "n_hit": int(hits),
        "hit_rate": float(hits / len(truth_pos_list)) if truth_pos_list else float("nan"),
        "mean_lead_bars": float(np.mean(leads)) if leads else float("nan"),
        "median_lead_bars": float(np.median(leads)) if leads else float("nan"),
    }


def false_alarm_rate(
    trigger: pd.Series,
    truth_turns: pd.DatetimeIndex | list,
    *,
    calendar: pd.DatetimeIndex,
    max_lead_bars: int = 5,
) -> dict[str, Any]:
    """Fraction of trigger fires that are near some truth turn."""
    truth = (
        pd.DatetimeIndex(pd.to_datetime(list(truth_turns))).normalize().unique().sort_values()
    )
    trig = trigger.fillna(False).astype(bool)
    trig.index = pd.to_datetime(trig.index).normalize()
    pos = _bar_positions(calendar)
    truth_pos = sorted({pos[d] for d in truth if d in pos})
    n_trig = 0
    n_near = 0
    for d, v in trig.items():
        if not v or d not in pos:
            continue
        n_trig += 1
        pp = pos[d]
        if any(abs(pp - tp) <= int(max_lead_bars) for tp in truth_pos):
            n_near += 1
    return {
        "n_trigger": int(n_trig),
        "n_near_truth": int(n_near),
        "precision_vs_turns": float(n_near / n_trig) if n_trig else float("nan"),
    }


def score_trigger_pair(
    entry_trig: pd.Series,
    exit_trig: pd.Series,
    truth: pd.Series,
    *,
    max_lead_bars: int = 5,
) -> dict[str, Any]:
    """Lead-hit scores for enter/leave against MA20 truth turns.

    Truth turns are taken from the same enter/leave definition used for the
    baseline trade book (``truth_entry_exit``), so baseline self-score has
    lead 0 and hit_rate 1 when turns exist.
    """
    idx = pd.DatetimeIndex(truth.index).normalize()
    truth_entry, truth_leave = truth_entry_exit(truth)
    enter_turns = pd.DatetimeIndex(idx[truth_entry.reindex(idx).fillna(False).astype(bool)])
    leave_turns = pd.DatetimeIndex(idx[truth_leave.reindex(idx).fillna(False).astype(bool)])
    enter_lead = trigger_lead_hits(
        entry_trig.reindex(idx).fillna(False),
        enter_turns,
        calendar=idx,
        max_lead_bars=max_lead_bars,
    )
    leave_lead = trigger_lead_hits(
        exit_trig.reindex(idx).fillna(False),
        leave_turns,
        calendar=idx,
        max_lead_bars=max_lead_bars,
    )
    enter_fa = false_alarm_rate(
        entry_trig.reindex(idx).fillna(False),
        enter_turns,
        calendar=idx,
        max_lead_bars=max_lead_bars,
    )
    leave_fa = false_alarm_rate(
        exit_trig.reindex(idx).fillna(False),
        leave_turns,
        calendar=idx,
        max_lead_bars=max_lead_bars,
    )
    enter_sym = turn_hit_rate(
        idx[entry_trig.reindex(idx).fillna(False).astype(bool)],
        enter_turns,
        tolerance_bars=max_lead_bars,
        calendar=idx,
    )
    leave_sym = turn_hit_rate(
        idx[exit_trig.reindex(idx).fillna(False).astype(bool)],
        leave_turns,
        tolerance_bars=max_lead_bars,
        calendar=idx,
    )
    return {
        "enter_hit_rate": enter_lead["hit_rate"],
        "enter_mean_lead": enter_lead["mean_lead_bars"],
        "enter_precision": enter_fa["precision_vs_turns"],
        "enter_n_truth": enter_lead["n_truth"],
        "enter_n_trigger": enter_lead["n_trigger"],
        "leave_hit_rate": leave_lead["hit_rate"],
        "leave_mean_lead": leave_lead["mean_lead_bars"],
        "leave_precision": leave_fa["precision_vs_turns"],
        "leave_n_truth": leave_lead["n_truth"],
        "leave_n_trigger": leave_lead["n_trigger"],
        "enter_sym_hit_rate": enter_sym["hit_rate"],
        "leave_sym_hit_rate": leave_sym["hit_rate"],
        "truth_n_up_days": int(truth.eq(STATE_UP).sum()),
    }


__all__ = [
    "false_alarm_rate",
    "ma20_hug_truth",
    "score_trigger_pair",
    "trigger_lead_hits",
    "truth_entry_exit",
]
