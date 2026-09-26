"""Forward-return three-state trend truth (labeling only; not causal).

For each bar T with a close at T+H:
  r = close[T+H] / close[T] - 1
  r > +up_thr  -> up
  r < -down_thr -> down
  else         -> range

Used only for scoring classifiers; trading signals must not use these labels.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

STATE_UP = "up"
STATE_DOWN = "down"
STATE_RANGE = "range"
STATES = (STATE_UP, STATE_DOWN, STATE_RANGE)

DEFAULT_HORIZON = 20
DEFAULT_UP_THR = 0.02
DEFAULT_DOWN_THR = 0.02


def fwd_return_trend_labels(
    close: pd.Series,
    *,
    horizon: int = DEFAULT_HORIZON,
    up_thr: float = DEFAULT_UP_THR,
    down_thr: float = DEFAULT_DOWN_THR,
) -> pd.Series:
    """Three-state labels from H-bar forward simple return.

    Last ``horizon`` bars are NaN (no future close). Index matches ``close``.
    """
    s = pd.to_numeric(close, errors="coerce").copy()
    s.index = pd.to_datetime(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be positive")
    fwd = s.shift(-h) / s - 1.0
    out = pd.Series(STATE_RANGE, index=s.index, dtype=object)
    out[fwd > float(up_thr)] = STATE_UP
    out[fwd < -float(down_thr)] = STATE_DOWN
    out[fwd.isna()] = np.nan
    out.name = f"fwd{h}_trend"
    return out


def up_episode_turns(labels: pd.Series) -> dict[str, pd.DatetimeIndex]:
    """Enter-up / leave-up dates from a three-state label series.

    Enter-up: non-up -> up. Leave-up: up -> non-up.
    NaN labels break continuity (no turn across NaN).
    """
    lab = labels.copy()
    lab.index = pd.to_datetime(lab.index).normalize()
    is_up = lab.eq(STATE_UP)
    valid = lab.notna()
    prev_up = is_up.shift(1)
    prev_valid = valid.shift(1)
    prev_up_b = prev_up.fillna(False).astype(bool)
    prev_valid_b = prev_valid.fillna(False).astype(bool)
    enter = is_up & (~prev_up_b) & valid & prev_valid_b
    leave = (~is_up) & prev_up_b & valid & prev_valid_b
    return {
        "enter_up": pd.DatetimeIndex(lab.index[enter]),
        "leave_up": pd.DatetimeIndex(lab.index[leave]),
    }


def turn_hit_rate(
    pred_turns: pd.DatetimeIndex | list,
    truth_turns: pd.DatetimeIndex | list,
    *,
    tolerance_bars: int = 3,
    calendar: pd.DatetimeIndex | None = None,
) -> dict[str, Any]:
    """Fraction of truth turns matched by a pred turn within ±tolerance bars.

    If ``calendar`` is given, tolerance is in trading-bar distance on that index;
    otherwise calendar-day distance is used (less precise).
    """
    truth = pd.DatetimeIndex(pd.to_datetime(list(truth_turns))).normalize().unique().sort_values()
    pred = pd.DatetimeIndex(pd.to_datetime(list(pred_turns))).normalize().unique().sort_values()
    if len(truth) == 0:
        return {
            "n_truth": 0,
            "n_pred": int(len(pred)),
            "n_hit": 0,
            "hit_rate": float("nan"),
        }
    if calendar is not None:
        cal = pd.DatetimeIndex(pd.to_datetime(calendar)).normalize().unique().sort_values()
        pos = {d: i for i, d in enumerate(cal)}
        pred_pos = sorted(pos[d] for d in pred if d in pos)
        hits = 0
        for t in truth:
            if t not in pos:
                continue
            tp = pos[t]
            for pp in pred_pos:
                if abs(pp - tp) <= int(tolerance_bars):
                    hits += 1
                    break
    else:
        hits = 0
        for t in truth:
            if any(abs((p - t).days) <= int(tolerance_bars) for p in pred):
                hits += 1
    return {
        "n_truth": int(len(truth)),
        "n_pred": int(len(pred)),
        "n_hit": int(hits),
        "hit_rate": float(hits / len(truth)),
    }


def classification_scores(
    pred: pd.Series,
    truth: pd.Series,
) -> dict[str, Any]:
    """Accuracy / up precision / up recall where both pred and truth are finite."""
    p = pred.reindex(truth.index)
    t = truth
    mask = t.notna() & p.notna()
    if int(mask.sum()) == 0:
        return {
            "n_scored": 0,
            "accuracy": float("nan"),
            "up_precision": float("nan"),
            "up_recall": float("nan"),
            "up_f1": float("nan"),
        }
    pv = p.loc[mask].astype(str)
    tv = t.loc[mask].astype(str)
    acc = float((pv == tv).mean())
    pred_up = pv.eq(STATE_UP)
    truth_up = tv.eq(STATE_UP)
    tp = int((pred_up & truth_up).sum())
    fp = int((pred_up & ~truth_up).sum())
    fn = int((~pred_up & truth_up).sum())
    prec = float(tp / (tp + fp)) if (tp + fp) else float("nan")
    rec = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0:
        f1 = float(2.0 * prec * rec / (prec + rec))
    else:
        f1 = float("nan")
    return {
        "n_scored": int(mask.sum()),
        "accuracy": acc,
        "up_precision": prec,
        "up_recall": rec,
        "up_f1": f1,
    }


__all__ = [
    "DEFAULT_DOWN_THR",
    "DEFAULT_HORIZON",
    "DEFAULT_UP_THR",
    "STATE_DOWN",
    "STATE_RANGE",
    "STATE_UP",
    "STATES",
    "classification_scores",
    "fwd_return_trend_labels",
    "turn_hit_rate",
    "up_episode_turns",
]
