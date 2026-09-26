#!/usr/bin/env python3
"""Find unselected overflow diversifier candidates from the full qlib universe."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .strategy_layer_audit import (
        DEFAULT_ALL_TXT,
        DEFAULT_END_DATE,
        DEFAULT_FUND_LIST_CSV,
        DEFAULT_OUT_DIR,
        DEFAULT_PROVIDER_URI,
        DEFAULT_SELECTED_TXT,
        DEFAULT_START_DATE,
        build_name_map,
        load_qlib_close,
        normalize_code,
        numeric_series,
        read_csv_columns,
        read_txt_codes,
        safe_float,
        write_table,
    )
except ImportError:  # pragma: no cover - supports direct script execution from etf_daily.pipeline/.
    from strategy_layer_audit import (  # type: ignore
        DEFAULT_ALL_TXT,
        DEFAULT_END_DATE,
        DEFAULT_FUND_LIST_CSV,
        DEFAULT_OUT_DIR,
        DEFAULT_PROVIDER_URI,
        DEFAULT_SELECTED_TXT,
        DEFAULT_START_DATE,
        build_name_map,
        load_qlib_close,
        normalize_code,
        numeric_series,
        read_csv_columns,
        read_txt_codes,
        safe_float,
        write_table,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit unselected Overflow Bucket candidates with low selected-pool correlation."
    )
    parser.add_argument("--provider-uri", default=DEFAULT_PROVIDER_URI)
    parser.add_argument("--selected-txt", default=DEFAULT_SELECTED_TXT)
    parser.add_argument("--all-txt", default=DEFAULT_ALL_TXT)
    parser.add_argument("--fund-list-csv", default=DEFAULT_FUND_LIST_CSV)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--lookback-days", type=int, default=20)
    parser.add_argument("--forward-days", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-avg-corr", type=float, default=0.60)
    parser.add_argument("--min-corr-obs", type=int, default=10)
    parser.add_argument("--rank-mode", choices=["return", "forecast_mean"], default="return")
    parser.add_argument("--forecast-csv", default="")
    parser.add_argument("--forecast-date", default="")
    return parser.parse_args(argv)


def corr_pair(left: pd.Series, right: pd.Series, min_obs: int) -> float:
    paired = pd.concat([numeric_series(left), numeric_series(right)], axis=1).dropna()
    if len(paired) < min_obs:
        return np.nan
    value = paired.iloc[:, 0].corr(paired.iloc[:, 1])
    return safe_float(value)


def selected_corr_metrics(
    returns_window: pd.DataFrame,
    code: str,
    selected_codes: list[str],
    min_corr_obs: int,
) -> tuple[float, float]:
    if code not in returns_window.columns:
        return np.nan, np.nan
    selected_available = [selected_code for selected_code in selected_codes if selected_code in returns_window.columns]
    if not selected_available:
        return np.nan, np.nan
    candidate = returns_window[code]
    corr_values: list[float] = []
    for selected_code in selected_available:
        corr_value = corr_pair(candidate, returns_window[selected_code], min_corr_obs)
        if np.isfinite(corr_value):
            corr_values.append(abs(corr_value))
    selected_equal = returns_window[selected_available].mean(axis=1)
    corr_to_equal = corr_pair(candidate, selected_equal, min_corr_obs)
    avg_abs_corr = float(np.nanmean(corr_values)) if corr_values else np.nan
    return avg_abs_corr, corr_to_equal


def forward_return(close: pd.DataFrame, code: str, position: int, forward_days: int) -> float:
    end_position = position + int(forward_days)
    if code not in close.columns or end_position >= len(close):
        return np.nan
    start_value = safe_float(close.iloc[position][code])
    end_value = safe_float(close.iloc[end_position][code])
    if not np.isfinite(start_value) or not np.isfinite(end_value) or start_value <= 0:
        return np.nan
    return float(end_value / start_value - 1.0)


def selected_equal_forward_return(
    close: pd.DataFrame,
    selected_codes: list[str],
    position: int,
    forward_days: int,
) -> float:
    selected_available = [code for code in selected_codes if code in close.columns]
    if not selected_available or position + int(forward_days) >= len(close):
        return np.nan
    start_values = close.iloc[position][selected_available].apply(safe_float)
    end_values = close.iloc[position + int(forward_days)][selected_available].apply(safe_float)
    returns = end_values / start_values.replace(0.0, np.nan) - 1.0
    return float(numeric_series(returns).mean())


def trailing_return_rankings(close: pd.DataFrame, position: int, lookback_days: int) -> pd.Series:
    start_position = position - int(lookback_days)
    if start_position < 0:
        return pd.Series(dtype=float)
    start_values = close.iloc[start_position].apply(safe_float)
    end_values = close.iloc[position].apply(safe_float)
    rankings = end_values / start_values.replace(0.0, np.nan) - 1.0
    return numeric_series(rankings).dropna().sort_values(ascending=False)


def forecast_rankings_by_date(forecast_csv: str | Path, forecast_date: str) -> dict[pd.Timestamp, pd.Series]:
    path = Path(forecast_csv).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Forecast CSV does not exist: {path}")
    requested_columns = [
        "instrument",
        "datetime",
        "forecast_mean_return_robust",
        "forecast_mean_return",
        "forecast_mean",
    ]
    forecast = read_csv_columns(path, requested_columns)
    if "instrument" not in forecast.columns or "datetime" not in forecast.columns:
        raise ValueError("Forecast CSV must include instrument and datetime columns")
    mu_col = next(
        (column for column in ["forecast_mean_return_robust", "forecast_mean_return", "forecast_mean"] if column in forecast.columns),
        None,
    )
    if mu_col is None:
        raise ValueError("Forecast CSV must include a forecast mean column")
    forecast[mu_col] = numeric_series(forecast[mu_col])
    if forecast_date:
        target_date = pd.Timestamp(forecast_date)
        forecast = forecast[pd.to_datetime(forecast["datetime"]) == target_date]
    rankings: dict[pd.Timestamp, pd.Series] = {}
    for date_value, day_frame in forecast.groupby("datetime"):
        series = day_frame.groupby("instrument")[mu_col].last().dropna().sort_values(ascending=False)
        rankings[pd.Timestamp(date_value)] = series
    return rankings


def candidate_rows_for_date(
    date_value: pd.Timestamp,
    position: int,
    ranking: pd.Series,
    close: pd.DataFrame,
    returns: pd.DataFrame,
    selected_codes: list[str],
    unselected_codes: set[str],
    name_map: dict[str, str],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    returns_window = returns.iloc[position - args.lookback_days + 1: position + 1]
    selected_future = selected_equal_forward_return(close, selected_codes, position, args.forward_days)
    rows: list[dict[str, object]] = []
    rank_index = 0
    for raw_code, score in ranking.items():
        code = normalize_code(raw_code)
        if code not in unselected_codes:
            continue
        avg_abs_corr, corr_to_equal = selected_corr_metrics(returns_window, code, selected_codes, args.min_corr_obs)
        if not np.isfinite(avg_abs_corr) or avg_abs_corr >= args.max_avg_corr:
            continue
        future = forward_return(close, code, position, args.forward_days)
        if not np.isfinite(future):
            continue
        rank_index += 1
        rows.append(
            {
                "selection_date": date_value.strftime("%Y-%m-%d"),
                "rank_mode": args.rank_mode,
                "rank_order": rank_index,
                "instrument": code,
                "name": name_map.get(code, ""),
                "candidate_tag": "Diversifier_Candidates",
                "score": float(score),
                "avg_abs_corr_to_selected": avg_abs_corr,
                "corr_to_selected_equal": corr_to_equal,
                "future_return": future,
                "selected_equal_future_return": selected_future,
                "excess_future_return": future - selected_future if np.isfinite(selected_future) else np.nan,
                "win_vs_selected_equal": bool(np.isfinite(selected_future) and future > selected_future),
                "positive_future_return": bool(future > 0),
            }
        )
        if rank_index >= args.top_k:
            break
    return rows


def build_summary(candidates: pd.DataFrame, forward_days: int) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(
            [
                {
                    "scope": "overall",
                    "candidate_count": 0,
                    "unique_instrument_count": 0,
                    "future_days": forward_days,
                }
            ]
        )
    top_counts = candidates["instrument"].value_counts().head(10)
    rows = [
        {
            "scope": "overall",
            "candidate_count": int(len(candidates)),
            "unique_instrument_count": int(candidates["instrument"].nunique()),
            "future_days": forward_days,
            "avg_score": float(numeric_series(candidates["score"]).mean()),
            "avg_abs_corr_to_selected": float(numeric_series(candidates["avg_abs_corr_to_selected"]).mean()),
            "avg_corr_to_selected_equal": float(numeric_series(candidates["corr_to_selected_equal"]).mean()),
            "avg_future_return": float(numeric_series(candidates["future_return"]).mean()),
            "avg_selected_equal_future_return": float(numeric_series(candidates["selected_equal_future_return"]).mean()),
            "avg_excess_future_return": float(numeric_series(candidates["excess_future_return"]).mean()),
            "win_rate_vs_selected_equal": float(candidates["win_vs_selected_equal"].mean()),
            "positive_future_share": float(candidates["positive_future_return"].mean()),
            "top_instruments_by_count": "; ".join(f"{code}:{count}" for code, count in top_counts.items()),
        }
    ]
    for rank_order, frame in candidates.groupby("rank_order"):
        rows.append(
            {
                "scope": f"rank_{int(rank_order)}",
                "candidate_count": int(len(frame)),
                "unique_instrument_count": int(frame["instrument"].nunique()),
                "future_days": forward_days,
                "avg_score": float(numeric_series(frame["score"]).mean()),
                "avg_abs_corr_to_selected": float(numeric_series(frame["avg_abs_corr_to_selected"]).mean()),
                "avg_corr_to_selected_equal": float(numeric_series(frame["corr_to_selected_equal"]).mean()),
                "avg_future_return": float(numeric_series(frame["future_return"]).mean()),
                "avg_selected_equal_future_return": float(numeric_series(frame["selected_equal_future_return"]).mean()),
                "avg_excess_future_return": float(numeric_series(frame["excess_future_return"]).mean()),
                "win_rate_vs_selected_equal": float(frame["win_vs_selected_equal"].mean()),
                "positive_future_share": float(frame["positive_future_return"].mean()),
                "top_instruments_by_count": "",
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    selected_codes = read_txt_codes(args.selected_txt)
    all_codes = read_txt_codes(args.all_txt)
    unselected_codes = set(all_codes).difference(selected_codes)
    close = load_qlib_close(args.provider_uri, all_codes, args.start_date, args.end_date)
    returns = close.pct_change(fill_method=None)
    name_map = build_name_map(args.fund_list_csv)
    dates = [pd.Timestamp(date_value) for date_value in close.index]
    date_positions = {date_value: index for index, date_value in enumerate(dates)}
    rows: list[dict[str, object]] = []

    if args.rank_mode == "forecast_mean":
        if not args.forecast_csv:
            raise ValueError("--rank-mode forecast_mean requires --forecast-csv")
        rankings_by_date = forecast_rankings_by_date(args.forecast_csv, args.forecast_date)
        for date_value, ranking in sorted(rankings_by_date.items()):
            if date_value not in date_positions:
                continue
            position = date_positions[date_value]
            if position < args.lookback_days or position + args.forward_days >= len(close):
                continue
            rows.extend(
                candidate_rows_for_date(
                    date_value, position, ranking, close, returns, selected_codes, unselected_codes, name_map, args
                )
            )
    else:
        for position in range(args.lookback_days, len(close) - args.forward_days):
            date_value = dates[position]
            ranking = trailing_return_rankings(close, position, args.lookback_days)
            rows.extend(
                candidate_rows_for_date(
                    date_value, position, ranking, close, returns, selected_codes, unselected_codes, name_map, args
                )
            )

    candidates = pd.DataFrame(rows)
    if not candidates.empty:
        candidates = candidates.sort_values(["selection_date", "rank_order", "instrument"])
    summary = build_summary(candidates, args.forward_days)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_table(candidates, out_dir, "overflow_diversifier_candidates.csv")
    write_table(summary, out_dir, "overflow_diversifier_summary.csv")
    return {
        "overflow_diversifier_candidates": candidates,
        "overflow_diversifier_summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
