"""Shared causal market indicators for decision_pack."""
from __future__ import annotations

import math

import pandas as pd


def slice_through(end: pd.Timestamp, close: pd.Series) -> pd.Series:
    available = close.loc[close.index <= end].dropna()
    return available


def window_return(close: pd.Series, end: pd.Timestamp, days: int) -> float:
    available = slice_through(end, close)
    if len(available) < days + 1:
        return float("nan")
    window = available.tail(days + 1)
    start = float(window.iloc[0])
    finish = float(window.iloc[-1])
    if start <= 0 or not math.isfinite(start) or not math.isfinite(finish):
        return float("nan")
    return finish / start - 1.0


def daily_return(close: pd.Series, end: pd.Timestamp) -> float:
    available = slice_through(end, close)
    if len(available) < 2:
        return float("nan")
    prev = float(available.iloc[-2])
    last = float(available.iloc[-1])
    if prev <= 0:
        return float("nan")
    return last / prev - 1.0


def moving_average(close: pd.Series, end: pd.Timestamp, window: int) -> float:
    available = slice_through(end, close)
    if len(available) < window:
        return float("nan")
    return float(available.tail(window).mean())


def close_below_ma(close: pd.Series, end: pd.Timestamp, window: int) -> bool:
    ma = moving_average(close, end, window)
    available = slice_through(end, close)
    if len(available) < window or not math.isfinite(ma):
        return False
    last = float(available.iloc[-1])
    return last < ma


def days_below_ma(close: pd.Series, end: pd.Timestamp, window: int = 20) -> int:
    """Consecutive trading days (ending at `end`) with close strictly below MA(window)."""
    available = slice_through(end, close)
    if len(available) < window:
        return 0
    count = 0
    for end_idx in range(len(available) - 1, window - 2, -1):
        window_slice = available.iloc[: end_idx + 1]
        ma = float(window_slice.tail(window).mean())
        current = float(window_slice.iloc[-1])
        if not math.isfinite(ma) or not math.isfinite(current):
            break
        if current < ma:
            count += 1
        else:
            break
    return count


def vol_percentile(
    close: pd.Series,
    end: pd.Timestamp,
    *,
    vol_window: int,
    lookback: int,
) -> float:
    available = slice_through(end, close)
    if len(available) < vol_window + 5:
        return float("nan")
    returns = available.pct_change().dropna()
    realized = returns.rolling(vol_window).std() * math.sqrt(252.0)
    realized = realized.dropna()
    if realized.empty:
        return float("nan")
    current = float(realized.iloc[-1])
    history = realized.tail(lookback)
    if len(history) < 5:
        return float("nan")
    return float((history <= current).sum()) / float(len(history))


def realized_vol_annualized(close: pd.Series, end: pd.Timestamp, *, vol_window: int) -> float:
    available = slice_through(end, close)
    if len(available) < vol_window + 1:
        return float("nan")
    returns = available.pct_change().dropna()
    window = returns.tail(vol_window)
    if len(window) < vol_window:
        return float("nan")
    std = float(window.std())
    if not math.isfinite(std):
        return float("nan")
    return std * math.sqrt(252.0)


def vol_level_label(
    vol_pct: float | None,
    *,
    low_max: float = 0.33,
    mid_max: float = 0.80,
) -> str | None:
    if vol_pct is None or not math.isfinite(vol_pct):
        return None
    if vol_pct <= low_max:
        return "low"
    if vol_pct <= mid_max:
        return "mid"
    return "high"


def drawdown_from_high(close: pd.Series, end: pd.Timestamp, *, lookback: int) -> float:
    available = slice_through(end, close)
    if len(available) < 2:
        return float("nan")
    window = available.tail(lookback)
    peak = float(window.max())
    current = float(window.iloc[-1])
    if peak <= 0 or not math.isfinite(peak) or not math.isfinite(current):
        return float("nan")
    return current / peak - 1.0


def ma_stack_label(ma5: float, ma10: float, ma20: float) -> str:
    if not all(math.isfinite(x) for x in (ma5, ma10, ma20)):
        return "mixed"
    if ma5 > ma10 > ma20:
        return "bull"
    if ma5 < ma10 < ma20:
        return "bear"
    return "mixed"


def ma20_trend_state(
    close: pd.Series,
    as_of: pd.Timestamp,
    *,
    window: int = 20,
) -> dict[str, object]:
    """Causal MA(window) trend state through ``as_of``.

    ``trend_intact``: close(as_of) >= MA and a contiguous run of closes at/above MA
    ending at as_of. ``trend_since`` is the first day of that run (reclaim day).
    """
    as_of = pd.Timestamp(as_of).normalize()
    available = slice_through(as_of, close)
    empty: dict[str, object] = {
        "trend_intact": False,
        "trend_since": None,
        "days_above": 0,
        "last_px": None,
        "ma20": None,
        "dist_ma20_pct": None,
        "last_break": None,
        "above_ma20": None,
    }
    if len(available) < window:
        return empty

    # Build aligned close vs rolling MA ending each day.
    above_flags: list[bool] = []
    dates: list[pd.Timestamp] = []
    ma_at: list[float] = []
    px_at: list[float] = []
    for end_idx in range(window - 1, len(available)):
        window_slice = available.iloc[: end_idx + 1]
        ma = float(window_slice.tail(window).mean())
        px = float(window_slice.iloc[-1])
        d = pd.Timestamp(window_slice.index[end_idx]).normalize()
        if not math.isfinite(ma) or not math.isfinite(px) or ma <= 0:
            continue
        dates.append(d)
        ma_at.append(ma)
        px_at.append(px)
        above_flags.append(px >= ma)

    if not dates:
        return empty

    # Ensure we evaluate at last available bar <= as_of
    last_i = len(dates) - 1
    last_px = px_at[last_i]
    last_ma = ma_at[last_i]
    dist = (last_px / last_ma - 1.0) * 100.0
    above_now = above_flags[last_i]

    if not above_now:
        # Start of the current below-MA streak (not as_of itself).
        run_start_i = last_i
        for i in range(last_i, -1, -1):
            if not above_flags[i]:
                run_start_i = i
            else:
                break
        return {
            "trend_intact": False,
            "trend_since": None,
            "days_above": 0,
            "last_px": last_px,
            "ma20": last_ma,
            "dist_ma20_pct": dist,
            "last_break": str(dates[run_start_i].date()),
            "above_ma20": False,
        }

    # Walk back while still above MA to find reclaim / run start.
    run_start_i = last_i
    for i in range(last_i, -1, -1):
        if above_flags[i]:
            run_start_i = i
        else:
            break
    days_above = last_i - run_start_i + 1
    trend_since = str(dates[run_start_i].date())
    # Most recent below day before the current above run (prior break).
    last_break: str | None = None
    for i in range(run_start_i - 1, -1, -1):
        if not above_flags[i]:
            last_break = str(dates[i].date())
            break
    return {
        "trend_intact": True,
        "trend_since": trend_since,
        "days_above": days_above,
        "last_px": last_px,
        "ma20": last_ma,
        "dist_ma20_pct": dist,
        "last_break": last_break,
        "above_ma20": True,
    }


def compute_trend_metrics(
    close: pd.Series,
    end: pd.Timestamp,
    *,
    vol_window: int,
    vol_lookback: int,
    dd_lookback: int = 20,
) -> dict[str, float | str | None]:
    available = slice_through(end, close)
    last_close = float(available.iloc[-1]) if not available.empty else float("nan")

    ma5 = moving_average(close, end, 5)
    ma10 = moving_average(close, end, 10)
    ma20 = moving_average(close, end, 20)

    dist_ma20_pct = float("nan")
    if math.isfinite(last_close) and math.isfinite(ma20) and ma20 > 0:
        dist_ma20_pct = (last_close / ma20 - 1.0) * 100.0

    vol_pct = vol_percentile(
        close,
        end,
        vol_window=vol_window,
        lookback=vol_lookback,
    )
    vol_ann_20 = realized_vol_annualized(close, end, vol_window=vol_window)
    vol_ann_5 = realized_vol_annualized(close, end, vol_window=5)
    vol_ratio_5_20 = float("nan")
    if math.isfinite(vol_ann_5) and math.isfinite(vol_ann_20) and vol_ann_20 > 0:
        vol_ratio_5_20 = vol_ann_5 / vol_ann_20
    ret_5d = window_return(close, end, 5)
    ret_20d = window_return(close, end, 20)
    dd_20d = drawdown_from_high(close, end, lookback=dd_lookback)

    def _opt(value: float) -> float | None:
        return value if math.isfinite(value) else None

    return {
        "close": _opt(last_close),
        "ma5": _opt(ma5),
        "ma10": _opt(ma10),
        "ma20": _opt(ma20),
        "vol_pct_120d": _opt(vol_pct),
        "vol_ann_20d": _opt(vol_ann_20),
        "vol_ann_5d": _opt(vol_ann_5),
        "vol_ratio_5_20": _opt(vol_ratio_5_20),
        "dist_ma20_pct": _opt(dist_ma20_pct),
        "ret_5d": _opt(ret_5d),
        "ret_20d": _opt(ret_20d),
        "ma_stack": ma_stack_label(ma5, ma10, ma20),
        "dd_20d_high": _opt(dd_20d),
    }


__all__ = [
    "close_below_ma",
    "days_below_ma",
    "compute_trend_metrics",
    "daily_return",
    "drawdown_from_high",
    "ma20_trend_state",
    "ma_stack_label",
    "moving_average",
    "realized_vol_annualized",
    "vol_level_label",
    "vol_percentile",
    "window_return",
]
