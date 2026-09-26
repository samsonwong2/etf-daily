"""Per-code self-train trade policy: hold_up vs buy&hold vs cash.

Each ticker picks using **only its own train window** (no pool-wide threshold).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

POLICY_HOLD_UP = "hold_up"
POLICY_BH = "bh"
POLICY_CASH = "cash"
POLICY_LABELS = {
    POLICY_HOLD_UP: "只做上涨态(hold_up)",
    POLICY_BH: "买入持有(BH)",
    POLICY_CASH: "空仓(cash)",
}


def _hold_up_nav_clean(close: np.ndarray, labs: np.ndarray) -> np.ndarray:
    """Long-only NAV: earn close-to-close return while prior bar was up."""
    close = np.asarray(close, dtype=float)
    labs = np.asarray(labs, dtype=object)
    n = len(close)
    nav = np.ones(n, dtype=float)
    if n == 0:
        return nav
    wealth = 1.0
    for i in range(1, n):
        if (
            labs[i - 1] == "up"
            and np.isfinite(close[i - 1])
            and close[i - 1] > 0
            and np.isfinite(close[i])
        ):
            wealth *= float(close[i]) / float(close[i - 1])
        nav[i] = wealth
    return nav


def hold_up_nav(close: np.ndarray, labs: np.ndarray) -> np.ndarray:
    return _hold_up_nav_clean(close, labs)


def bh_nav(close: np.ndarray) -> np.ndarray:
    close = np.asarray(close, dtype=float)
    n = len(close)
    if n == 0:
        return np.asarray([], dtype=float)
    c0 = float(close[0])
    if not (np.isfinite(c0) and c0 > 0):
        return np.ones(n, dtype=float)
    out = close / c0
    out = np.where(np.isfinite(out), out, 1.0)
    return out.astype(float)


def cash_nav(n: int) -> np.ndarray:
    return np.ones(int(n), dtype=float)


def compound_from_nav(nav: np.ndarray) -> float:
    if len(nav) < 1 or not np.isfinite(nav[-1]):
        return float("nan")
    return float(nav[-1] - 1.0)


def pick_policy(
    train_hold_up: float,
    train_bh: float,
    train_cash: float = 0.0,
) -> str:
    """Argmax train compound; ties prefer bh > hold_up > cash."""
    cands = {
        POLICY_BH: float(train_bh),
        POLICY_HOLD_UP: float(train_hold_up),
        POLICY_CASH: float(train_cash),
    }
    rank = {POLICY_BH: 2, POLICY_HOLD_UP: 1, POLICY_CASH: 0}
    return max(rank, key=lambda k: (cands[k], rank[k]))


def evaluate_self_train_policy(
    dates: pd.DatetimeIndex | np.ndarray,
    close: np.ndarray,
    labs: np.ndarray,
    *,
    train_cutoff: str,
) -> dict[str, Any]:
    """Pick policy on train (< cutoff); return metrics + full-window NAV curves."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    close = np.asarray(close, dtype=float)
    labs = np.asarray(labs, dtype=object)
    if len(dates) != len(close) or len(close) != len(labs):
        raise ValueError("dates/close/labs length mismatch")

    cut = pd.Timestamp(train_cutoff).normalize()
    train_mask = dates < cut
    if int(train_mask.sum()) < 2:
        policy = POLICY_HOLD_UP
        tr_hu = tr_bh = tr_cash = float("nan")
    else:
        c_tr = close[train_mask]
        l_tr = labs[train_mask]
        tr_hu = compound_from_nav(_hold_up_nav_clean(c_tr, l_tr))
        tr_bh = compound_from_nav(bh_nav(c_tr))
        tr_cash = 0.0
        policy = pick_policy(tr_hu, tr_bh, tr_cash)

    nav_hu = _hold_up_nav_clean(close, labs)
    nav_bh_ = bh_nav(close)
    nav_cash_ = cash_nav(len(close))
    nav_policy = {
        POLICY_HOLD_UP: nav_hu,
        POLICY_BH: nav_bh_,
        POLICY_CASH: nav_cash_,
    }[policy]

    oos_mask = dates >= cut

    def _split_comp(nav: np.ndarray, mask: np.ndarray) -> float:
        idx = np.flatnonzero(mask)
        if len(idx) < 2:
            return float("nan")
        a, b = float(nav[idx[0]]), float(nav[idx[-1]])
        if not (np.isfinite(a) and a > 0 and np.isfinite(b)):
            return float("nan")
        return float(b / a - 1.0)

    return {
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "train_cutoff": str(cut.date()),
        "train_hold_up": tr_hu,
        "train_bh": tr_bh,
        "train_cash": 0.0 if np.isfinite(tr_hu) else float("nan"),
        "train_edge_hu_vs_bh": (
            float(tr_hu - tr_bh)
            if np.isfinite(tr_hu) and np.isfinite(tr_bh)
            else float("nan")
        ),
        "oos_hold_up": _split_comp(nav_hu, oos_mask),
        "oos_bh": _split_comp(nav_bh_, oos_mask),
        "oos_cash": 0.0 if int(oos_mask.sum()) else float("nan"),
        "oos_policy": _split_comp(nav_policy, oos_mask),
        "source": "argmax_train_compound",
        "nav_hold_up": nav_hu,
        "nav_bh": nav_bh_,
        "nav_cash": nav_cash_,
        "nav_policy": nav_policy,
        "dates": dates,
    }


def policy_frame_from_eval(ev: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "as_of": ev["dates"],
            "nav_hold_up": ev["nav_hold_up"],
            "nav_bh": ev["nav_bh"],
            "nav_cash": ev["nav_cash"],
            "nav_policy": ev["nav_policy"],
        }
    )


def config_block(ev: dict[str, Any]) -> dict[str, Any]:
    skip = {"nav_hold_up", "nav_bh", "nav_cash", "nav_policy", "dates"}
    out = {k: v for k, v in ev.items() if k not in skip}
    for k, v in list(out.items()):
        if isinstance(v, (np.floating, float)):
            out[k] = float(v) if np.isfinite(v) else None
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
    return out


__all__ = [
    "POLICY_BH",
    "POLICY_CASH",
    "POLICY_HOLD_UP",
    "POLICY_LABELS",
    "bh_nav",
    "cash_nav",
    "compound_from_nav",
    "config_block",
    "evaluate_self_train_policy",
    "hold_up_nav",
    "pick_policy",
    "policy_frame_from_eval",
]
