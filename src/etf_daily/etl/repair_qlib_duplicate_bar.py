"""Drop one duplicated Qlib day bar when that realigns OHLC to change_csv.

``dump_update`` can append a session that is already in the bin when
``instruments/all.txt`` lags the features. The calendar only grows by truly
new dates, so every later bar shifts forward by one and the real last day
sits past the calendar. This repairs a symbol only when deleting that one
repeated bar makes open/high/low/close match ``change_csv``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_FIELDS = ("open", "high", "low", "close")
ALL_FIELDS = PRICE_FIELDS + ("volume",)
PRICE_ABS_TOL = 1.5e-3
TAIL_SCAN = 40


def _read_bin(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<f4")


def _prices_match(
    values: np.ndarray,
    start: int,
    calendar: pd.DatetimeIndex,
    csv_col: pd.Series,
    n_calendar: int,
) -> bool:
    dates = calendar[start : start + n_calendar]
    aligned = pd.Series(values[:n_calendar], index=dates)
    common = aligned.index.intersection(csv_col.index)
    if len(common) < 5:
        return False
    diff = np.abs(
        aligned.reindex(common).to_numpy(dtype=float)
        - csv_col.reindex(common).to_numpy(dtype=float)
    )
    finite = np.isfinite(diff)
    if int(finite.sum()) < 5:
        return False
    return bool(np.nanmax(diff[finite]) <= PRICE_ABS_TOL)


def duplicate_drop_index(
    bins: dict[str, np.ndarray],
    *,
    start: int,
    calendar: pd.DatetimeIndex,
    csv: pd.DataFrame,
) -> int | None:
    """Index of the inserted duplicate, or None when no safe drop exists.

    ``bins`` values include the leading calendar-start float.
    """
    close = bins["close"][1:]
    n = len(close)
    n_calendar = len(calendar) - start
    if n_calendar <= 0 or n != n_calendar + 1:
        return None
    if all(
        _prices_match(bins[field][1:], start, calendar, csv[field], n_calendar)
        for field in PRICE_FIELDS
    ):
        return None
    begin = max(1, n - TAIL_SCAN)
    for i in range(n - 1, begin - 1, -1):
        if abs(float(close[i]) - float(close[i - 1])) > PRICE_ABS_TOL:
            continue
        ok = True
        for field in PRICE_FIELDS:
            data = bins[field][1:]
            trial = np.concatenate([data[:i], data[i + 1 :]])
            if len(trial) != n_calendar or not _prices_match(
                trial, start, calendar, csv[field], n_calendar
            ):
                ok = False
                break
        if ok:
            return i
    return None


def _write_dropped(path: Path, raw: np.ndarray, drop_data_index: int) -> None:
    """Rewrite a bin without data index ``drop_data_index`` (0 = first price)."""
    file_index = drop_data_index + 1
    kept = np.concatenate([raw[:file_index], raw[file_index + 1 :]]).astype("<f4")
    tmp = path.with_suffix(path.suffix + ".repairing")
    kept.tofile(tmp)
    tmp.replace(path)


def repair_provider(provider: Path, *, symbols: list[str] | None = None) -> dict[str, int]:
    """Repair shifted bins under a Qlib provider. Returns action counts."""
    provider = Path(provider)
    calendar = pd.DatetimeIndex(
        pd.to_datetime(pd.read_csv(provider / "calendars" / "day.txt", header=None)[0])
    )
    features = provider / "features"
    csv_dir = provider / "change_csv"
    counts = {"aligned": 0, "repaired": 0, "skipped": 0}
    names = symbols or sorted(p.name for p in features.iterdir() if (p / "close.day.bin").is_file())
    for name in names:
        feat = features / name
        close_path = feat / "close.day.bin"
        if not close_path.is_file():
            counts["skipped"] += 1
            continue
        raws = {field: _read_bin(feat / f"{field}.day.bin") for field in ALL_FIELDS}
        if any(len(raws[field]) != len(raws["close"]) for field in ALL_FIELDS):
            counts["skipped"] += 1
            continue
        start = int(round(float(raws["close"][0])))
        n_calendar = len(calendar) - start
        n = len(raws["close"]) - 1
        if n == n_calendar:
            counts["aligned"] += 1
            continue
        csv_path = csv_dir / f"{name}.csv"
        if n != n_calendar + 1 or not csv_path.is_file():
            counts["skipped"] += 1
            continue
        csv = pd.read_csv(csv_path, usecols=["date", *PRICE_FIELDS], parse_dates=["date"])
        csv = csv.dropna(subset=["date"]).drop_duplicates("date").set_index("date").sort_index()
        drop_at = duplicate_drop_index(raws, start=start, calendar=calendar, csv=csv)
        if drop_at is None:
            counts["skipped"] += 1
            continue
        for field in ALL_FIELDS:
            _write_dropped(feat / f"{field}.day.bin", raws[field], drop_at)
        counts["repaired"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--symbol", action="append", default=None)
    args = parser.parse_args()
    counts = repair_provider(args.provider, symbols=args.symbol)
    print(counts)


if __name__ == "__main__":
    main()
