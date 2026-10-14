"""Pure helpers for sina ETL incremental skip / merge / dump mode.

Kept free of akshare / network so unit tests stay offline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

POLITE_DELAY_MIN_S = 1
POLITE_DELAY_MAX_S = 3


def csv_max_date(path: str | Path) -> Optional[str]:
    """Return max YYYY-MM-DD in CSV ``date`` column, or None if missing/empty."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        df = pd.read_csv(p, usecols=["date"])
    except (ValueError, pd.errors.EmptyDataError, OSError):
        return None
    if df.empty or "date" not in df.columns:
        return None
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.max().strftime("%Y-%m-%d")


def should_skip_fund(max_date: Optional[str], as_of: str) -> bool:
    """Skip network fetch when local CSV already covers ``as_of`` (inclusive)."""
    if max_date is None:
        return False
    return str(max_date)[:10] >= str(as_of)[:10]


def merge_ohlcv(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Concat OHLCV frames, drop duplicate dates (keep last), sort by date."""
    if new_df is None or new_df.empty:
        return old_df.copy() if old_df is not None else pd.DataFrame()
    if old_df is None or old_df.empty:
        out = new_df.copy()
    else:
        out = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    if "date" not in out.columns:
        return out
    out = out.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.dropna(subset=["date"])
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out.sort_values("date").reset_index(drop=True)


def choose_dump_mode(full: bool, calendars_exist: bool) -> str:
    """Return ``dump_all`` or ``dump_update`` for qlib dump_bin.py."""
    if full or not calendars_exist:
        return "dump_all"
    return "dump_update"


def polite_delay_seconds(
    enabled: bool,
    rng: Any = None,
    *,
    min_s: int = POLITE_DELAY_MIN_S,
    max_s: int = POLITE_DELAY_MAX_S,
) -> float:
    """Return sleep seconds; 0 when polite delay is disabled."""
    if not enabled:
        return 0.0
    if rng is None:
        import random

        return float(random.randint(min_s, max_s))
    if hasattr(rng, "integers"):
        return float(rng.integers(min_s, max_s + 1))
    return float(rng.randint(min_s, max_s))


def calendars_day_exists(qlib_dir: str | Path) -> bool:
    return Path(qlib_dir).joinpath("calendars", "day.txt").is_file()
