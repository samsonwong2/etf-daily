"""Benchmark and portfolio market metrics for tier evaluation."""
from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from decision_pack.src.config import DecisionPackConfig, load_decision_pack_config
from decision_pack.src.indicators import close_below_ma, daily_return, vol_percentile, window_return
from decision_pack.src.tier import MarketSnapshot
from pipeline.benchmark_regime import load_hs300_close
from runtime_paths import QLIB_PROVIDER_URI
from workspace.scripts.generate_daily_mu_position_report import (  # noqa: E402
    day_rows,
    first_existing_column,
    read_csv,
)


def _as_timestamp(report_date: str) -> pd.Timestamp:
    return pd.Timestamp(report_date)


def load_nav_series(daily_nav_csv: Path, report_date: str) -> pd.Series:
    frame = read_csv(daily_nav_csv)
    date_col = first_existing_column(frame, ["datetime", "date", "trade_date"])
    nav_col = first_existing_column(frame, ["nav", "portfolio_nav", "net_value"])
    if nav_col is None:
        raise ValueError(f"No nav column in {daily_nav_csv}")

    if date_col is not None:
        series = pd.to_numeric(frame[nav_col], errors="coerce")
        index = pd.to_datetime(frame[date_col], errors="coerce")
        nav = pd.Series(series.values, index=index).dropna().sort_index()
    else:
        nav = pd.to_numeric(frame[nav_col], errors="coerce").dropna()
        nav.index = pd.RangeIndex(len(nav))

    if "datetime" in frame.columns:
        day = day_rows(frame, report_date)
        if not day.empty:
            value = pd.to_numeric(day[nav_col], errors="coerce").dropna()
            if not value.empty:
                end_ts = _as_timestamp(report_date)
                nav.loc[end_ts] = float(value.iloc[-1])
    return nav.sort_index()


def compute_drawdown(
    nav: pd.Series,
    end: pd.Timestamp,
    *,
    lookback_days: int,
) -> float:
    if nav.empty:
        return float("nan")
    if isinstance(nav.index, pd.DatetimeIndex):
        available = nav.loc[nav.index <= end].dropna()
    else:
        available = nav.dropna()
    if available.empty:
        return float("nan")
    window = available.tail(lookback_days)
    peak = float(window.max())
    current = float(window.iloc[-1])
    if peak <= 0 or not math.isfinite(peak) or not math.isfinite(current):
        return float("nan")
    return max(0.0, (peak - current) / peak)


def build_market_snapshot(
    report_date: str,
    *,
    daily_nav_csv: Path | None = None,
    provider_uri: str | Path | None = None,
    config: DecisionPackConfig | None = None,
    benchmark_close: pd.Series | None = None,
    nav_series: pd.Series | None = None,
) -> MarketSnapshot:
    cfg = config or load_decision_pack_config()
    end = _as_timestamp(report_date)
    uri = str(provider_uri or QLIB_PROVIDER_URI)

    if benchmark_close is None:
        warmup_start = (end - pd.tseries.offsets.BDay(cfg.vol_lookback + cfg.ma_window + 30)).strftime(
            "%Y-%m-%d"
        )
        benchmark_close = load_hs300_close(uri, warmup_start, report_date)

    bench = benchmark_close
    if cfg.benchmark_code not in getattr(bench, "name", cfg.benchmark_code):
        pass

    r_bench = daily_return(bench, end)
    bench_3d = window_return(bench, end, 3)
    bench_5d = window_return(bench, end, 5)
    close_below_ma20 = close_below_ma(bench, end, cfg.ma_window)
    vol_pct = vol_percentile(
        bench,
        end,
        vol_window=cfg.vol_window,
        lookback=cfg.vol_lookback,
    )

    dd_port = float("nan")
    if nav_series is not None:
        dd_port = compute_drawdown(nav_series, end, lookback_days=cfg.dd_lookback_days)
    elif daily_nav_csv is not None and daily_nav_csv.exists():
        dd_port = compute_drawdown(
            load_nav_series(daily_nav_csv, report_date),
            end,
            lookback_days=cfg.dd_lookback_days,
        )

    return MarketSnapshot(
        as_of=report_date,
        benchmark=cfg.benchmark_code,
        r_bench=float(r_bench) if math.isfinite(r_bench) else 0.0,
        bench_3d=float(bench_3d) if math.isfinite(bench_3d) else 0.0,
        bench_5d=float(bench_5d) if math.isfinite(bench_5d) else 0.0,
        close_below_ma20=close_below_ma20,
        vol_percentile_120d=float(vol_pct) if math.isfinite(vol_pct) else 0.0,
        dd_port=float(dd_port) if math.isfinite(dd_port) else 0.0,
    )


def market_snapshot_to_dict(snapshot: MarketSnapshot) -> dict[str, Any]:
    return asdict(snapshot)


def write_market_snapshot(path: Path, snapshot: MarketSnapshot) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(market_snapshot_to_dict(snapshot), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_bridge_l3_row(bridge_l3_csv: Path, report_date: str) -> dict[str, float | str | None]:
    frame = read_csv(bridge_l3_csv)
    day = day_rows(frame, report_date)
    if day.empty:
        return {"exposure_scale": None, "risk_macro_budget": None}
    row = day.iloc[-1]
    exposure = pd.to_numeric(row.get("exposure_scale"), errors="coerce")
    budget = pd.to_numeric(row.get("risk_macro_budget"), errors="coerce")
    return {
        "exposure_scale": float(exposure) if pd.notna(exposure) else None,
        "risk_macro_budget": float(budget) if pd.notna(budget) else None,
    }


__all__ = [
    "build_market_snapshot",
    "compute_drawdown",
    "load_bridge_l3_row",
    "load_nav_series",
    "market_snapshot_to_dict",
    "write_market_snapshot",
]
