"""Long-only discrete-event simulator (T+1 close, fixed cost).

Independent of adaptive hold_up / stage-rule books. Signal on day T uses
information through T's close; execution is the next bar's close.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

COST_PER_SIDE = 0.001


def simulate_long_only_events(
    entry: pd.Series | np.ndarray,
    exit_sig: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    *,
    dates: pd.Series | np.ndarray | None = None,
    cost_per_side: float = COST_PER_SIDE,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    """Simulate long-only book from boolean entry/exit signals.

    Rules
    -----
    - Signal at bar ``i`` executes at bar ``i+1`` close (T+1).
    - Flat: ignore exit; ignore duplicate entry while already marking a pending buy
      or while in a position (position starts at the fill bar).
    - In position: ignore entry; exit on next exit signal's T+1 bar.
    - Same bar cannot both open and close; fill order is open then check exit only
      on later bars.
    - Open position at end is MTM into equity / compound; not counted in ``n_closed``.
    """
    px = np.asarray(pd.to_numeric(pd.Series(close), errors="coerce"), dtype=float)
    n = len(px)
    ent = np.asarray(pd.Series(entry).fillna(False).astype(bool), dtype=bool)
    ex = np.asarray(pd.Series(exit_sig).fillna(False).astype(bool), dtype=bool)
    if len(ent) != n or len(ex) != n:
        raise ValueError("entry/exit/close length mismatch")
    if dates is None:
        date_arr = np.asarray([str(i) for i in range(n)], dtype=object)
    else:
        date_arr = np.asarray(pd.Series(dates).astype(str), dtype=object)
        if len(date_arr) != n:
            raise ValueError("dates length mismatch")

    if n == 0:
        empty = pd.DataFrame(columns=["date", "equity", "drawdown", "exposed"])
        return (
            {
                "n_closed": 0,
                "wins": 0,
                "win": float("nan"),
                "mean_ret_net": 0.0,
                "compound": 0.0,
                "bh": 0.0,
                "edge": 0.0,
                "max_drawdown": 0.0,
                "exposure": 0.0,
                "open_pos": False,
            },
            [],
            empty,
        )

    wealth = 1.0
    entry_i = -1
    entry_px = np.nan
    entry_effective = np.nan
    pending_buy = -1
    pending_sell = -1
    trades: list[dict[str, Any]] = []
    equities = np.ones(n, dtype=float)
    exposed = np.zeros(n, dtype=bool)

    for i in range(n):
        # Execute pending fills at today's close.
        if pending_buy == i and entry_i < 0 and np.isfinite(px[i]) and px[i] > 0:
            entry_i = i
            entry_px = float(px[i])
            entry_effective = entry_px * (1.0 + float(cost_per_side))
            pending_buy = -1
        elif pending_buy == i:
            pending_buy = -1

        if pending_sell == i and entry_i >= 0 and np.isfinite(px[i]) and px[i] > 0:
            exit_effective = float(px[i]) * (1.0 - float(cost_per_side))
            net = exit_effective / entry_effective - 1.0
            wealth *= 1.0 + net
            trades.append(
                {
                    "entry_date": str(date_arr[entry_i]),
                    "exit_date": str(date_arr[i]),
                    "entry_px": float(entry_px),
                    "exit_px": float(px[i]),
                    "ret_net": float(net),
                    "bars": int(i - entry_i),
                }
            )
            entry_i = -1
            entry_px = entry_effective = np.nan
            pending_sell = -1
        elif pending_sell == i:
            pending_sell = -1

        # Schedule new signals for next bar (need i+1).
        if i + 1 < n:
            if entry_i < 0 and pending_buy < 0 and bool(ent[i]):
                pending_buy = i + 1
            if entry_i >= 0 and pending_sell < 0 and bool(ex[i]):
                # Do not schedule sell on the same bar we just bought.
                if entry_i != i:
                    pending_sell = i + 1

        if entry_i >= 0:
            exposed[i] = True
            if np.isfinite(px[i]) and px[i] > 0 and np.isfinite(entry_effective):
                liquidation = float(px[i]) * (1.0 - float(cost_per_side))
                equities[i] = wealth * liquidation / entry_effective
            else:
                equities[i] = wealth
        else:
            equities[i] = wealth

    peak = np.maximum.accumulate(equities)
    drawdown = equities / np.maximum(peak, 1e-12) - 1.0
    rets = np.asarray([t["ret_net"] for t in trades], dtype=float)
    if n > 1 and np.isfinite(px[0]) and px[0] > 0 and np.isfinite(px[-1]):
        bh = float(px[-1] / px[0] - 1.0)
    else:
        bh = 0.0
    metrics = {
        "n_closed": int(len(trades)),
        "wins": int(np.sum(rets > 0)) if len(rets) else 0,
        "win": float(np.mean(rets > 0)) if len(rets) else float("nan"),
        "mean_ret_net": float(np.mean(rets)) if len(rets) else 0.0,
        "compound": float(equities[-1] - 1.0),
        "bh": bh,
        "edge": float(equities[-1] - 1.0 - bh),
        "max_drawdown": float(np.min(drawdown)),
        "exposure": float(np.mean(exposed)),
        "open_pos": bool(entry_i >= 0),
    }
    equity = pd.DataFrame(
        {
            "date": date_arr,
            "equity": equities,
            "drawdown": drawdown,
            "exposed": exposed,
        }
    )
    return metrics, trades, equity


__all__ = ["COST_PER_SIDE", "simulate_long_only_events"]
