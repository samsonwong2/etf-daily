"""Download futures, cash indexes, QVIX, and month-end option snapshots.

A failed download keeps the previous CSV and does not raise. Nothing here
is called by the daily etl.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

from etf_daily.derivs.features import summarize_risk_day
from etf_daily.paths import WORKSPACE_DIR

CACHE_DIR = WORKSPACE_DIR / "derivs_cache"
FRESH_DAYS = 4

INDEX_SYMBOLS = {
    "000300": "sh000300",
    "000016": "sh000016",
    "000905": "sh000905",
}
FUTURE_SYMBOLS = {"IF": "IF0", "IH": "IH0", "IC": "IC0"}
QVIX_NAMES = ("50etf", "300etf", "500etf")
CHAIN_UNDERLYINGS = ("510050", "510300", "510500")

SSE_URL = "http://query.sse.com.cn/commonQuery.do"
SSE_SQL = "SSE_ZQPZ_YSP_GGQQZSXT_YSHQ_QQFXZB_DATE_L"


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _fresh(path: Path, today: pd.Timestamp | None = None) -> bool:
    if not path.is_file():
        return False
    frame = _read(path)
    if frame.empty or "date" not in frame.columns:
        return False
    last = pd.to_datetime(frame["date"]).max()
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today).normalize()
    return pd.Timestamp(last).normalize() >= today - pd.Timedelta(days=FRESH_DAYS)


def _normalize_close(frame: pd.DataFrame, date_col: str, close_col: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_col]).dt.normalize(),
            "close": pd.to_numeric(frame[close_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["date", "close"])
    return out.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def _load_or_fetch(path: Path, fetcher, *, refresh: bool) -> pd.DataFrame:
    if path.is_file() and not refresh and _fresh(path):
        return _read(path)
    try:
        fresh = fetcher()
    except Exception as exc:  # network or vendor schema
        print(f"derivs: {path.name} failed ({exc})", file=sys.stderr)
        if path.is_file():
            return _read(path)
        return pd.DataFrame(columns=["date", "close"])
    if fresh is None or fresh.empty:
        print(f"derivs: {path.name} came back empty", file=sys.stderr)
        if path.is_file():
            return _read(path)
        return pd.DataFrame(columns=["date", "close"])
    _write(path, fresh)
    return fresh


def _fetch_index(code: str) -> pd.DataFrame:
    import akshare as ak

    symbol = INDEX_SYMBOLS[code]
    raw = ak.stock_zh_index_daily(symbol=symbol)
    return _normalize_close(raw, "date", "close")


def _fetch_future(name: str) -> pd.DataFrame:
    import akshare as ak

    raw = ak.futures_main_sina(symbol=FUTURE_SYMBOLS[name], start_date="20100101")
    return _normalize_close(raw, "日期", "收盘价")


def _fetch_qvix(name: str) -> pd.DataFrame:
    import akshare as ak

    getter = {
        "50etf": ak.index_option_50etf_qvix,
        "300etf": ak.index_option_300etf_qvix,
        "500etf": ak.index_option_500etf_qvix,
    }[name]
    return _normalize_close(getter(), "date", "close")


def refresh_market(*, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Indexes, dominant futures, and QVIX. Keys are ``index:000300`` and so on."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    jobs: list[tuple[str, Path, object]] = []
    for code in INDEX_SYMBOLS:
        jobs.append((f"index:{code}", CACHE_DIR / f"index_{code}.csv", lambda code=code: _fetch_index(code)))
    for name in FUTURE_SYMBOLS:
        jobs.append((f"future:{name}", CACHE_DIR / f"future_{name}.csv", lambda name=name: _fetch_future(name)))
    for name in QVIX_NAMES:
        jobs.append((f"qvix:{name}", CACHE_DIR / f"qvix_{name}.csv", lambda name=name: _fetch_qvix(name)))
    for key, path, fetcher in jobs:
        out[key] = _load_or_fetch(path, fetcher, refresh=refresh)
        print(f"derivs: {key} {len(out[key])} rows")
    return out


def fetch_sse_risk(trade_date: str) -> pd.DataFrame:
    """One day of SSE option greeks. Empty frame when the vendor has no rows."""
    import requests

    response = requests.get(
        SSE_URL,
        params={
            "isPagination": "false",
            "trade_date": trade_date,
            "sqlId": SSE_SQL,
            "contractSymbol": "",
        },
        headers={"Referer": "http://www.sse.com.cn/", "User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    rows = response.json().get("result") or []
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _attempted_path() -> Path:
    return CACHE_DIR / "option_chain_attempted.txt"


def _attempted() -> set[str]:
    path = _attempted_path()
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _mark_attempted(day: str) -> None:
    path = _attempted_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    done = _attempted()
    if day in done:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(day + "\n")


def month_ends(dates: pd.Series, start: str = "2015-02-01") -> list[pd.Timestamp]:
    """Last available session of each month, from the first 50ETF option month."""
    index = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().unique().sort_values()
    index = index[index >= pd.Timestamp(start)]
    if len(index) == 0:
        return []
    series = pd.Series(index, index=index)
    grouped = series.groupby([index.year, index.month]).max()
    return [pd.Timestamp(value).normalize() for value in grouped.to_list()]


def refresh_chain(
    dates: list[pd.Timestamp],
    *,
    pause: float = 0.2,
) -> pd.DataFrame:
    """Append month-end ATM IV and skew. Dates already tried are skipped."""
    path = CACHE_DIR / "option_chain.csv"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    existing = _read(path) if path.is_file() else pd.DataFrame()
    have: set[str] = set()
    if not existing.empty and "date" in existing.columns:
        have = {pd.Timestamp(day).strftime("%Y%m%d") for day in existing["date"]}
    done = _attempted() | have
    for day in dates:
        stamp = pd.Timestamp(day).normalize()
        key = stamp.strftime("%Y%m%d")
        if key in done:
            continue
        try:
            raw = fetch_sse_risk(key)
            summary = summarize_risk_day(raw)
        except Exception as exc:
            print(f"derivs: chain {key} failed ({exc})", file=sys.stderr)
            time.sleep(pause)
            continue
        if summary.empty:
            _mark_attempted(key)
            print(f"derivs: chain {key} empty")
            time.sleep(pause)
            continue
        summary = summary.loc[summary["underlying"].isin(CHAIN_UNDERLYINGS)].copy()
        if summary.empty:
            _mark_attempted(key)
            time.sleep(pause)
            continue
        existing = pd.concat([existing, summary], ignore_index=True) if not existing.empty else summary
        existing["date"] = pd.to_datetime(existing["date"]).dt.normalize()
        existing = existing.drop_duplicates(["date", "underlying"], keep="last")
        existing = existing.sort_values(["date", "underlying"]).reset_index(drop=True)
        _write(path, existing)
        _mark_attempted(key)
        print(f"derivs: chain {key} {len(summary)} underlyings", flush=True)
        time.sleep(pause)
    return existing


def load_chain() -> pd.DataFrame:
    path = CACHE_DIR / "option_chain.csv"
    if not path.is_file():
        return pd.DataFrame()
    return _read(path)


def close_series(frame: pd.DataFrame) -> pd.Series:
    """Date-indexed close from a cache frame. Empty series when the frame is empty."""
    if frame is None or frame.empty or "date" not in frame.columns:
        return pd.Series(dtype=float)
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last")
    series = pd.Series(pd.to_numeric(out["close"], errors="coerce").to_numpy(), index=pd.DatetimeIndex(out["date"]))
    return series.sort_index()
