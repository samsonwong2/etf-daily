"""Basis, percentile, calendar flags, and option-chain summaries.

Percentiles use a trailing window that ends on the same day. A later
observation cannot change an earlier percentile.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from etf_daily.derivs.bs import DEFAULT_R, invert_iv

CONTRACT_RE = re.compile(r"^(\d{6})([CP])(\d{2})(\d{2})[A-Z](\d+)$")
ROLL_THRESHOLD = 0.015
ROLL_BUFFER_DAYS = 5
PCT_WINDOW = 250
PCT_MIN_PERIODS = 60
SKEW_CARRY_LIMIT = 25
WITCHING_MONTHS = {3, 6, 9, 12}
LIMIT_MOVE = 0.095


@dataclass(frozen=True)
class PairSpec:
    """ETF, its index future, the cash index, and the QVIX series name."""

    future: str
    index: str
    qvix: str


PAIRS: dict[str, PairSpec] = {
    "SH510300": PairSpec("IF", "000300", "300etf"),
    "SH510050": PairSpec("IH", "000016", "50etf"),
    "SH510500": PairSpec("IC", "000905", "500etf"),
}


def basis(future: float, spot: float) -> float:
    """Raw basis F / S - 1. Both prices must be positive."""
    if not np.isfinite(future) or not np.isfinite(spot) or future <= 0 or spot <= 0:
        return float("nan")
    return float(future) / float(spot) - 1.0


def annualize_basis(raw: float, days: int) -> float:
    """Scale a raw basis by 365 / days. ``days`` is at least 1."""
    if not np.isfinite(raw) or days <= 0:
        return float("nan")
    return float(raw) * 365.0 / int(days)


def third_friday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (4 - int(first.weekday())) % 7
    return first + pd.Timedelta(days=offset + 14)


def front_expiry(day: pd.Timestamp, buffer: int = ROLL_BUFFER_DAYS) -> pd.Timestamp:
    """Third Friday of the front month.

    Inside ``buffer`` calendar days of expiry, the next month is used.
    That matches a dominant contract that has already rolled.
    """
    day = pd.Timestamp(day).normalize()
    expiry = third_friday(day.year, day.month)
    if (expiry - day).days < buffer:
        year, month = day.year, day.month + 1
        if month == 13:
            year, month = year + 1, 1
        expiry = third_friday(year, month)
    return expiry


def days_to_front_expiry(day: pd.Timestamp, buffer: int = ROLL_BUFFER_DAYS) -> int:
    day = pd.Timestamp(day).normalize()
    return max(int((front_expiry(day, buffer) - day).days), 1)


def rolling_percentile(
    values: pd.Series,
    window: int = PCT_WINDOW,
    min_periods: int = PCT_MIN_PERIODS,
) -> pd.Series:
    """Fraction of the trailing window that is less than or equal to today.

    NaN rows do not count. The window stops at today.
    """

    def _rank(window_values: np.ndarray) -> float:
        current = window_values[-1]
        if not np.isfinite(current):
            return float("nan")
        finite = window_values[np.isfinite(window_values)]
        if len(finite) == 0:
            return float("nan")
        return float(np.mean(finite <= current))

    return values.rolling(int(window), min_periods=int(min_periods)).apply(_rank, raw=True)


def roll_jump_mask(
    future: pd.Series,
    spot: pd.Series,
    threshold: float = ROLL_THRESHOLD,
) -> pd.Series:
    """True when the future jumps relative to the cash index.

    A roll changes F without a matching move in S. Those days are left
    blank in the basis series.
    """
    aligned_f, aligned_s = future.align(spot, join="inner")
    gap = np.log(aligned_f.astype(float)).diff() - np.log(aligned_s.astype(float)).diff()
    mask = gap.abs() > float(threshold)
    mask.iloc[0] = False
    return mask.fillna(False)


def parse_contract(contract_id: str) -> dict[str, object] | None:
    """Parse an SSE/ETF option id such as ``510050C2407M02250``."""
    match = CONTRACT_RE.match(str(contract_id).strip())
    if match is None:
        return None
    underlying, right, year, month, strike_raw = match.groups()
    return {
        "underlying": underlying,
        "cp": right,
        "expiry": third_friday(2000 + int(year), int(month)),
        "strike": int(strike_raw) / 1000.0,
    }


def _nearest(frame: pd.DataFrame, target: float) -> pd.Series | None:
    if frame.empty:
        return None
    delta = pd.to_numeric(frame["delta"], errors="coerce")
    score = (delta - target).abs()
    if not score.notna().any():
        return None
    return frame.loc[score.idxmin()]


def summarize_risk_day(frame: pd.DataFrame) -> pd.DataFrame:
    """Near-month ATM IV and 25-delta skew for one SSE risk snapshot.

    ATM is the contract whose delta is closest to 0.5 (call) or -0.5 (put).
    Skew is the 25-delta put IV minus the 25-delta call IV. IV is the
    exchange value on the row. Rows without a parseable contract id are dropped.
    """
    if frame is None or frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    work = frame.copy()
    parsed = work["CONTRACT_ID"].map(parse_contract) if "CONTRACT_ID" in work.columns else None
    if parsed is None:
        return pd.DataFrame()
    keep = parsed.notna()
    work = work.loc[keep].copy()
    parsed = parsed.loc[keep]
    work["underlying"] = parsed.map(lambda item: item["underlying"])
    work["cp"] = parsed.map(lambda item: item["cp"])
    work["expiry"] = parsed.map(lambda item: item["expiry"])
    work["delta"] = (
        pd.to_numeric(work["DELTA_VALUE"], errors="coerce")
        if "DELTA_VALUE" in work.columns
        else np.nan
    )
    work["iv"] = (
        pd.to_numeric(work["IMPLC_VOLATLTY"], errors="coerce")
        if "IMPLC_VOLATLTY" in work.columns
        else np.nan
    )
    if "TRADE_DATE" not in work.columns or work.empty:
        return pd.DataFrame()
    trade_date = pd.to_datetime(work["TRADE_DATE"].iloc[0]).normalize()
    for underlying, group in work.groupby("underlying", sort=True):
        expiry = group.loc[group["expiry"] >= trade_date, "expiry"]
        if expiry.empty:
            continue
        front = expiry.min()
        front_rows = group.loc[group["expiry"] == front]
        calls = front_rows.loc[front_rows["cp"] == "C"]
        puts = front_rows.loc[front_rows["cp"] == "P"]
        atm_call = _nearest(calls, 0.50)
        atm_put = _nearest(puts, -0.50)
        call_25 = _nearest(calls, 0.25)
        put_25 = _nearest(puts, -0.25)
        call_iv = float(atm_call["iv"]) if atm_call is not None else float("nan")
        put_iv = float(atm_put["iv"]) if atm_put is not None else float("nan")
        ivs = [v for v in (call_iv, put_iv) if np.isfinite(v)]
        skew = float("nan")
        if call_25 is not None and put_25 is not None:
            if np.isfinite(call_25["iv"]) and np.isfinite(put_25["iv"]):
                skew = float(put_25["iv"]) - float(call_25["iv"])
        rows.append(
            {
                "date": trade_date,
                "underlying": underlying,
                "expiry": front,
                "atm_call_iv": call_iv,
                "atm_put_iv": put_iv,
                "atm_iv": float(np.mean(ivs)) if ivs else float("nan"),
                "iv25_call": float(call_25["iv"]) if call_25 is not None else float("nan"),
                "iv25_put": float(put_25["iv"]) if put_25 is not None else float("nan"),
                "skew": skew,
                "n_contracts": int(len(front_rows)),
            }
        )
    return pd.DataFrame(rows)


def atm_iv_from_marks(
    call_price: float,
    put_price: float,
    spot: float,
    strike: float,
    tau: float,
    rate: float = DEFAULT_R,
) -> float | None:
    """Average of the call IV and the put IV inverted from marks. q = 0."""
    call_iv = invert_iv(call_price, spot, strike, rate, tau, kind="call")
    put_iv = invert_iv(put_price, spot, strike, rate, tau, kind="put")
    found = [v for v in (call_iv, put_iv) if v is not None]
    if not found:
        return None
    return float(np.mean(found))


def _as_close(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").astype(float)
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).normalize()
    return out[~out.index.duplicated(keep="last")].sort_index()


def _calendar_flags(index: pd.DatetimeIndex) -> pd.DataFrame:
    delivery = []
    witching = []
    for day in index:
        stamp = pd.Timestamp(day).normalize()
        is_expiry = stamp == third_friday(stamp.year, stamp.month)
        delivery.append(bool(is_expiry))
        witching.append(bool(is_expiry and stamp.month in WITCHING_MONTHS))
    return pd.DataFrame(
        {"delivery": delivery, "witching": witching},
        index=index,
    )


def build_panel(
    etf_close: pd.Series,
    future_close: pd.Series,
    index_close: pd.Series,
    qvix_close: pd.Series | None = None,
    chain: pd.DataFrame | None = None,
    *,
    underlying: str = "",
    window: int = PCT_WINDOW,
    min_periods: int = PCT_MIN_PERIODS,
    roll_threshold: float = ROLL_THRESHOLD,
    roll_buffer_days: int = ROLL_BUFFER_DAYS,
) -> pd.DataFrame:
    """Daily features on the ETF calendar. Missing inputs stay NaN.

    ``qvix_close`` is the published vol index in percent (28.6 means 28.6%).
    It is stored as a decimal. Chain IV is joined by date and not filled
    into the filter columns. ``skew_carry`` repeats the last chain skew for
    at most ``SKEW_CARRY_LIMIT`` sessions so a monthly snapshot can be read
    on later days. It is not a trading filter.
    """
    close = _as_close(etf_close)
    future = _as_close(future_close).reindex(close.index)
    spot = _as_close(index_close).reindex(close.index)
    raw = pd.Series(
        [basis(f, s) for f, s in zip(future.to_numpy(), spot.to_numpy())],
        index=close.index,
        dtype=float,
    )
    roll = roll_jump_mask(future, spot, roll_threshold).reindex(close.index).fillna(False)
    days = [days_to_front_expiry(day, roll_buffer_days) for day in close.index]
    ann = pd.Series(
        [annualize_basis(value, day_count) for value, day_count in zip(raw.to_numpy(), days)],
        index=close.index,
        dtype=float,
    )
    ann = ann.mask(roll.to_numpy())
    out = pd.DataFrame(
        {
            "px": close,
            "future_close": future,
            "index_close": spot,
            "basis": raw.mask(roll.to_numpy()),
            "basis_ann": ann,
            "roll": roll.astype(bool),
        },
        index=close.index,
    )
    out["basis_pct"] = rolling_percentile(out["basis_ann"], window, min_periods)
    if qvix_close is None or len(qvix_close) == 0:
        out["iv"] = np.nan
    else:
        iv = _as_close(qvix_close).reindex(close.index) / 100.0
        out["iv"] = iv
    out["iv_pct"] = rolling_percentile(out["iv"], window, min_periods)
    hv = close.pct_change(fill_method=None).rolling(20).std() * np.sqrt(250.0)
    out["hv20"] = hv
    out["iv_hv"] = out["iv"] / hv
    out["atm_iv"] = np.nan
    out["skew"] = np.nan
    if chain is not None and not chain.empty and underlying:
        part = chain.loc[chain["underlying"].astype(str) == str(underlying)].copy()
        if not part.empty:
            part["date"] = pd.to_datetime(part["date"]).dt.normalize()
            part = part.drop_duplicates("date", keep="last").set_index("date").sort_index()
            out["atm_iv"] = pd.to_numeric(part["atm_iv"], errors="coerce").reindex(close.index)
            out["skew"] = pd.to_numeric(part["skew"], errors="coerce").reindex(close.index)
    out["skew_carry"] = out["skew"].ffill(limit=SKEW_CARRY_LIMIT)
    flags = _calendar_flags(close.index)
    out["delivery"] = flags["delivery"].to_numpy()
    out["witching"] = flags["witching"].to_numpy()
    move = close.pct_change(fill_method=None).abs()
    out["limit_day"] = move.ge(LIMIT_MOVE).fillna(False)
    return out
