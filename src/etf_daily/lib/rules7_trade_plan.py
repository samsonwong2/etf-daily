"""P95 signals scored one at a time: enter next close, hold 20 sessions, stop at -10%.

Nothing here is a live account. A later signal is not skipped just because an
earlier one is still open.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

HOLD_DAYS = 20
STOP_PCT = 0.10
# Observation tier: same buy mask, lower rail recomputed at P90 instead of P95.
WATCH_Q = 0.90
WATCH_SINCE = "2026-02-02"
WATCH_TARGET_TRADES = 200
# Same-day P95 buy count, divided by the applicable pool with enough history.
# Calibrated on closed trades: share >= 8% won about 67%, share < 5% about 36%.
BREADTH_MIN_BARS = 250
BREADTH_STRONG = 0.08
BREADTH_WEAK = 0.05
CLUSTER_PACK_ROOTS = (
    Path.home() / "temp" / "decision_packs",
    Path.home() / "etf-daily-output" / "temp" / "decision_packs",
)


def _iso(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _blank_trade(signal_date: str) -> dict:
    return {
        "signal_date": signal_date,
        "entry_date": None,
        "entry_px": None,
        "exit_date": None,
        "exit_px": None,
        "ret": None,
        "reason": None,
        "status": "pending",
        "days_held": None,
        "days_left": None,
        "stop_px": None,
        "pnl": None,
        "dist_stop": None,
        "sell_tomorrow": False,
        "bars_from_end": None,
    }


def _one_signal(
    prices: pd.Series,
    dates: list[pd.Timestamp],
    sig_i: int,
    *,
    hold_days: int,
    stop_pct: float,
) -> dict | None:
    """One signal, independent of every other signal.

    Bar *sig_i* fills at the next close. From the bar after entry, a close at
    or below the entry times ``1 - stop_pct`` sells at the following close.
    Otherwise the close ``hold_days - 1`` bars after entry is the scheduled
    exit and sells on the ``hold_days``-th bar after entry.
    """
    n = len(prices)
    last = n - 1
    trade = _blank_trade(_iso(dates[sig_i]))
    if sig_i >= last:
        return trade
    entry_i = sig_i + 1
    entry_px = float(prices.iloc[entry_i])
    if not np.isfinite(entry_px) or entry_px <= 0:
        return None
    stop_px = entry_px * (1.0 - stop_pct)
    pending: str | None = None
    exit_i: int | None = None
    reason: str | None = None
    for i in range(entry_i, n):
        price = float(prices.iloc[i])
        if pending is not None:
            exit_i = i
            reason = pending
            break
        days = i - entry_i
        stop_hit = days >= 1 and np.isfinite(price) and price <= stop_px
        expiry = days >= hold_days - 1
        if stop_hit or expiry:
            pending = "stop" if stop_hit else "expiry"
    last_px = float(prices.iloc[last])
    trade.update(
        {
            "entry_date": _iso(dates[entry_i]),
            "entry_px": entry_px,
            "stop_px": stop_px,
            "bars_from_end": last - entry_i,
        }
    )
    if exit_i is not None and reason is not None:
        exit_px = float(prices.iloc[exit_i])
        ret = exit_px / entry_px - 1.0
        trade.update(
            {
                "status": "closed",
                "reason": reason,
                "exit_date": _iso(dates[exit_i]),
                "exit_px": exit_px,
                "ret": ret,
                "pnl": ret,
                "days_held": exit_i - entry_i,
                "days_left": None,
                "dist_stop": (exit_px / stop_px - 1.0) if stop_px else None,
                "sell_tomorrow": False,
            }
        )
        return trade
    days_held = last - entry_i
    trade.update(
        {
            "status": "open",
            "reason": pending,
            "pnl": (last_px / entry_px - 1.0) if entry_px else None,
            "days_held": days_held,
            "days_left": hold_days - days_held,
            "dist_stop": (last_px / stop_px - 1.0) if stop_px else None,
            "sell_tomorrow": pending is not None,
        }
    )
    return trade


def signal_trades(
    px: pd.Series,
    buy: pd.Series,
    *,
    hold_days: int = HOLD_DAYS,
    stop_pct: float = STOP_PCT,
) -> list[dict]:
    """Every buy signal is its own trade. Later signals are not skipped."""
    prices = pd.to_numeric(pd.Series(px), errors="coerce").ffill()
    if prices.empty:
        return []
    prices = prices.sort_index()
    signals = pd.Series(buy).reindex(prices.index).fillna(False).astype(bool)
    dates = [pd.Timestamp(ts) for ts in prices.index]
    trades: list[dict] = []
    for sig_i in np.flatnonzero(signals.to_numpy(bool)):
        trade = _one_signal(prices, dates, int(sig_i), hold_days=hold_days, stop_pct=stop_pct)
        if trade is not None:
            trades.append(trade)
    return trades


def trade_stats(trades: Iterable[dict] | None, since: str | None = None) -> tuple[int, float, float]:
    """Closed trades only. ``since`` compares the signal date, inclusive."""
    rets: list[float] = []
    for trade in trades or []:
        if not isinstance(trade, dict) or trade.get("status") != "closed":
            continue
        if since and str(trade.get("signal_date") or "") < str(since):
            continue
        ret = trade.get("ret")
        if isinstance(ret, (int, float)) and np.isfinite(ret):
            rets.append(float(ret))
    if not rets:
        return 0, float("nan"), float("nan")
    arr = np.asarray(rets, dtype=float)
    return int(len(arr)), float((arr > 0).mean()), float(arr.mean())


def _pack_rets(rets: list[float]) -> tuple[int, float, float]:
    if not rets:
        return 0, float("nan"), float("nan")
    arr = np.asarray(rets, dtype=float)
    return int(len(arr)), float((arr > 0).mean()), float(arr.mean())


def tier_history(rows: Iterable[dict]) -> dict[str, tuple[int, float, float]]:
    """Tag each closed signal by that day's pool-wide buy share.

    A name is in the pool on a date when the rule applies and the date is
    between its 250th bar and its last bar. The numerator is names in the
    pool with a signal that day. Trades from before a name joined the pool
    are left out of the three buckets.
    """
    usable = [
        row for row in rows
        if row.get("applicable") and row.get("bar250_date") and row.get("last_date")
    ]
    day_buys: dict[str, int] = {}
    for row in usable:
        bar250 = str(row["bar250_date"])
        last = str(row["last_date"])
        seen: set[str] = set()
        for trade in row.get("trades") or []:
            day = str(trade.get("signal_date") or "")
            if not day or day < bar250 or day > last or day in seen:
                continue
            seen.add(day)
            day_buys[day] = day_buys.get(day, 0) + 1
    day_tier: dict[str, str] = {}
    for day, n_buy in day_buys.items():
        pool = sum(1 for row in usable if str(row["bar250_date"]) <= day <= str(row["last_date"]))
        share = (n_buy / pool) if pool else float("nan")
        day_tier[day] = breadth_tier(n_buy, share)
    buckets: dict[str, list[float]] = {"集中": [], "一般": [], "单独": []}
    for row in usable:
        bar250 = str(row["bar250_date"])
        for trade in row.get("trades") or []:
            if trade.get("status") != "closed":
                continue
            day = str(trade.get("signal_date") or "")
            tier = day_tier.get(day)
            ret = trade.get("ret")
            if day >= bar250 and tier in buckets and isinstance(ret, (int, float)) and np.isfinite(ret):
                buckets[tier].append(float(ret))
    return {key: _pack_rets(vals) for key, vals in buckets.items()}


def breadth_tier(n_buy: int, share: float) -> str:
    """集中 / 一般 / 单独 / 无买点. A day with no raw buys is not a weak day."""
    if int(n_buy) <= 0 or not np.isfinite(share):
        return "无买点"
    if share >= BREADTH_STRONG:
        return "集中"
    if share >= BREADTH_WEAK:
        return "一般"
    return "单独"


def _flag(value: object) -> bool:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip() in {"是", "1", "True", "true", "yes"}
    return bool(value)


def breadth_stats(df: pd.DataFrame) -> dict:
    """Pool-wide P95 buy share for one as-of day.

    The pool is names the rule applies to, with at least ``BREADTH_MIN_BARS``
    bars. The numerator is the raw P95 buy mask, including names already held,
    so the 8% and 5% cutoffs match the closed-trade study. Missing ``_buy_raw``
    falls back to the checklist column ``买入信号``; missing ``_n_bars`` treats
    every applicable name as long enough, and ``degraded`` is set.
    """
    work = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    degraded = False
    if "_plan_applicable" in work.columns:
        applicable = work["_plan_applicable"].map(_flag)
    else:
        applicable = pd.Series(True, index=work.index)
    if "_n_bars" in work.columns:
        bars = pd.to_numeric(work["_n_bars"], errors="coerce").fillna(0)
    else:
        degraded = True
        bars = pd.Series(float(BREADTH_MIN_BARS), index=work.index)
    pool_mask = applicable & (bars >= BREADTH_MIN_BARS)
    if "_buy_raw" in work.columns:
        buy = work["_buy_raw"].map(_flag)
    else:
        degraded = True
        buy = work["买入信号"].map(_flag) if "买入信号" in work.columns else pd.Series(False, index=work.index)
    n_buy = int((pool_mask & buy).sum())
    pool = int(pool_mask.sum())
    share = (n_buy / pool) if pool else float("nan")
    return {
        "n_buy": n_buy,
        "pool": pool,
        "share": float(share),
        "tier": breadth_tier(n_buy, share),
        "degraded": degraded,
    }


def d080_cluster_csv(
    as_of: pd.Timestamp | str,
    roots: Iterable[Path] | None = None,
) -> tuple[Path | None, pd.Timestamp | None]:
    """d080 cluster file for ``as_of``, or the latest earlier day."""
    tag = pd.Timestamp(as_of).strftime("%Y%m%d")
    found: list[tuple[str, Path]] = []
    for root in CLUSTER_PACK_ROOTS if roots is None else roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for child in root.iterdir():
            name = child.name
            if not child.is_dir() or len(name) != 8 or not name.isdigit() or name > tag:
                continue
            path = child / f"cluster_representatives_{name}_d080.csv"
            if path.is_file():
                found.append((name, path))
    if not found:
        return None, None
    found.sort()
    for name, path in found:
        if name == tag:
            return path, pd.Timestamp(name)
    name, path = found[-1]
    return path, pd.Timestamp(name)
