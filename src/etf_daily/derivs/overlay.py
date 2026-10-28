"""Compare rules7 with a few fixed filters. No threshold search."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from etf_daily.lib import rules7

COST = 0.0005
HIGH_IV_PCT = 0.80
DEEP_BASIS_PCT = 0.20
HORIZONS = (5, 10, 20)
MDD_IMPROVE = 0.01
CALMAR_IMPROVE = 0.05
MIN_TRADE_FRACTION = 0.5
MIN_TRADES = 5


def entry_fills(result: dict[str, Any]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Signal date and fill date for each rules7 entry, including an open trade."""
    rows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for trade in result["trades"]:
        rows.append((pd.Timestamp(trade["buy"]).normalize(), pd.Timestamp(trade["fill"]).normalize()))
    state = result["state"]
    if state.get("holding") and state.get("buy_date") is not None and state.get("fill_date") is not None:
        rows.append(
            (pd.Timestamp(state["buy_date"]).normalize(), pd.Timestamp(state["fill_date"]).normalize())
        )
    seen: set[pd.Timestamp] = set()
    unique: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for signal, fill in rows:
        if signal in seen:
            continue
        seen.add(signal)
        unique.append((signal, fill))
    return unique


def path_stats(close: pd.Series, fill: pd.Timestamp, horizon: int) -> tuple[float, float] | None:
    """Return and max drawdown from the fill close through ``horizon`` sessions."""
    index = pd.DatetimeIndex(pd.to_datetime(close.index)).normalize()
    series = pd.Series(close.to_numpy(), index=index)
    fill = pd.Timestamp(fill).normalize()
    if fill not in series.index:
        return None
    loc = int(series.index.get_loc(fill))
    end = loc + int(horizon)
    if end >= len(series):
        return None
    path = series.iloc[loc : end + 1].astype(float)
    if not np.isfinite(path.iloc[0]) or path.iloc[0] <= 0:
        return None
    ret = float(path.iloc[-1] / path.iloc[0] - 1.0)
    return ret, rules7.max_drawdown(path)


def tercile(percentile: float) -> str | None:
    if percentile is None or not np.isfinite(percentile):
        return None
    if percentile <= 1.0 / 3.0:
        return "低"
    if percentile <= 2.0 / 3.0:
        return "中"
    return "高"


def run_window(
    close: pd.Series,
    buy: pd.Series,
    sell: pd.Series,
    *,
    g1: pd.Series | None,
    trail: float | None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """rules7 state machine on one slice, starting flat. Cost is 5bp a side."""
    if start is None and end is None:
        view = close
    elif start is None:
        view = close.loc[:end]
    elif end is None:
        view = close.loc[start:]
    else:
        view = close.loc[start:end]
    buy_view = buy.reindex(view.index).fillna(False)
    sell_view = sell.reindex(view.index).fillna(False)
    g1_view = None if g1 is None else g1.reindex(view.index).fillna(False)
    return rules7.run_state_machine(
        view,
        buy_view,
        sell_view,
        g1=g1_view,
        trail=trail,
        cost=COST,
    )


def oos_improved(base: dict[str, Any], alt: dict[str, Any]) -> bool:
    """True when the back half is better and the trade count is still usable.

    Better means max drawdown at least 1 percentage point shallower, or
    Calmar at least 0.05 higher. Trade count must stay at least half of the
    baseline and at least 5.
    """
    base_n = int(base["n_trades"])
    alt_n = int(alt["n_trades"])
    if base_n <= 0:
        return False
    if alt_n < max(MIN_TRADES, math.ceil(MIN_TRADE_FRACTION * base_n)):
        return False
    mdd_better = float(alt["mdd"]) >= float(base["mdd"]) + MDD_IMPROVE
    base_calmar = float(base["calmar"])
    alt_calmar = float(alt["calmar"])
    calmar_better = (
        math.isfinite(base_calmar)
        and math.isfinite(alt_calmar)
        and alt_calmar >= base_calmar + CALMAR_IMPROVE
    )
    return bool(mdd_better or calmar_better)


def block_masks(panel: pd.DataFrame) -> dict[str, pd.Series]:
    """Fixed filters. A missing feature does not block the buy."""
    index = panel.index
    high_iv = panel["iv_pct"].ge(HIGH_IV_PCT).fillna(False)
    deep = panel["basis_pct"].le(DEEP_BASIS_PCT).fillna(False)
    calendar = (panel["witching"].astype(bool) | panel["limit_day"].astype(bool)).fillna(False)
    none = pd.Series(False, index=index)
    return {
        "基线": none,
        "挡住高隐含波动率": high_iv,
        "挡住深贴水": deep,
        "挡住三巫日或涨跌停": calendar,
    }


def adopted_variants(improved: dict[str, dict[str, bool]], minimum_codes: int = 2) -> list[str]:
    """Variants that improve at least ``minimum_codes`` symbols out of sample."""
    names = [name for name in next(iter(improved.values())) if name != "基线"] if improved else []
    kept = []
    for name in names:
        hits = sum(1 for code in improved if improved[code].get(name))
        if hits >= minimum_codes:
            kept.append(name)
    return kept
