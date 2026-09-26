"""Build Route-A-style override files from causal rolling historical means.

This is the production-safe counterpart to the oracle mean override builder.
For each forecast date, it writes the mean return that would have been known as
of that date, instead of copying one full-window realized mean across the whole
backtest.

Example
-------
python src/etf_daily/pool/build_rolling_mean_route_a_override.py \
    --forecast-start-date 2025-05-01 \
    --forecast-end-date 2026-04-30 \
    --initial-annualized-csv ~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/annualized_returns_20240501_20250430.csv \
    --output-dir ~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/route_a_rolling_prior_20240501_20260430 \
    --window 252 \
    --asof-lag 1 \
    --overwrite
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from etf_daily.pipeline.price_adjustments import (
    auto_qfq_adjust_close_panel,
    log_qfq_adjustments,
)

try:
    from etf_daily.pool import build_oracle_route_a_override as static_builder
except ModuleNotFoundError:  # Allows direct execution from the fund_pool_builder directory.
    import build_oracle_route_a_override as static_builder  # type: ignore[no-redef]


DEFAULT_PROVIDER_URI = static_builder.DEFAULT_PROVIDER_URI
DEFAULT_CLUSTER_MAPPING = static_builder.DEFAULT_CLUSTER_MAPPING
DEFAULT_TEMP_DIR = static_builder.DEFAULT_TEMP_DIR
TRADING_DAYS_PER_YEAR = static_builder.TRADING_DAYS_PER_YEAR
MEAN_MODEL_CHOICES = (
    "rolling_mean",
    "ewma",
    "multi_window_mean",
    "momentum",
    "vol_adjusted_momentum",
    "ic_weighted_ensemble",
    "regime_ic_weighted_ensemble",
    "regime_switch_ewma_multi_window",
    "regime_switch_ewma_shrink",
)
SHRINK_TARGET_CHOICES = ("none", "zero", "cross_sectional_median")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Route-A override files whose mu is a date-varying rolling "
            "historical mean computed without future returns."
        )
    )
    parser.add_argument("--forecast-start-date", required=True, help="First forecast/trading date.")
    parser.add_argument("--forecast-end-date", required=True, help="Last forecast/trading date.")
    parser.add_argument(
        "--initial-annualized-csv",
        default=None,
        help=(
            "Optional prior CSV with instrument, mean_daily_return and annualized_return. "
            "The date range is inferred from its filename when --history-start-date is omitted. "
            "Values are used only as a fallback for missing early rolling means."
        ),
    )
    parser.add_argument(
        "--history-start-date",
        default=None,
        help=(
            "First qlib close date to load for the return history. Defaults to the first "
            "YYYYMMDD in --initial-annualized-csv, or forecast_start minus about 370 calendar days."
        ),
    )
    parser.add_argument(
        "--history-end-date",
        default=None,
        help="Last history date before forecasts. Defaults to the second YYYYMMDD in the initial CSV.",
    )
    parser.add_argument("--cluster-mapping", default=DEFAULT_CLUSTER_MAPPING)
    parser.add_argument("--provider-uri", default=DEFAULT_PROVIDER_URI)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output Route-A override directory. Defaults under ~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=252,
        help="Rolling return window in trading days (default: 252).",
    )
    parser.add_argument(
        "--mean-model",
        default="rolling_mean",
        choices=MEAN_MODEL_CHOICES,
        help="Causal mean model used to generate daily mu. Default keeps the original rolling mean baseline.",
    )
    parser.add_argument(
        "--ewma-span",
        type=int,
        default=60,
        help="EWMA span for --mean-model ewma (default: 60).",
    )
    parser.add_argument(
        "--multi-windows",
        default="20,60,120,252",
        help="Comma-separated windows for multi-window and ensemble models.",
    )
    parser.add_argument(
        "--multi-window-weights",
        default="",
        help="Optional comma-separated weights matching --multi-windows; empty means equal weights.",
    )
    parser.add_argument(
        "--momentum-window",
        type=int,
        default=60,
        help="Lookback window for momentum mean models (default: 60).",
    )
    parser.add_argument(
        "--vol-window",
        type=int,
        default=60,
        help="Volatility window for vol-adjusted momentum (default: 60).",
    )
    parser.add_argument(
        "--ensemble-ewma-spans",
        default="20,60,120,252",
        help="Comma-separated EWMA spans included in IC-weighted ensembles.",
    )
    parser.add_argument(
        "--ensemble-momentum-windows",
        default="20,60,120",
        help="Comma-separated momentum windows included in IC-weighted ensembles.",
    )
    parser.add_argument(
        "--ensemble-vol-adjusted-windows",
        default="60,120",
        help="Comma-separated vol-adjusted momentum windows included in IC-weighted ensembles.",
    )
    parser.add_argument(
        "--ic-lookback",
        type=int,
        default=60,
        help="Rolling IC lookback for IC-weighted ensembles (default: 60 forecast dates).",
    )
    parser.add_argument(
        "--ic-min-periods",
        type=int,
        default=20,
        help="Minimum IC observations before ensemble weights become active (default: 20).",
    )
    parser.add_argument(
        "--regime-benchmark",
        default="SH510300",
        help="Benchmark instrument used by regime-conditioned ensembles (default: SH510300).",
    )
    parser.add_argument(
        "--regime-fast-ma",
        type=int,
        default=20,
        help="Fast benchmark moving average for regime-conditioned ensembles.",
    )
    parser.add_argument(
        "--regime-slow-ma",
        type=int,
        default=60,
        help="Slow benchmark moving average for regime-conditioned ensembles.",
    )
    parser.add_argument(
        "--regime-vol-window",
        type=int,
        default=20,
        help="Benchmark volatility window for regime-conditioned ensembles.",
    )
    parser.add_argument(
        "--regime-vol-lookback",
        type=int,
        default=120,
        help="Lookback used to define high-volatility benchmark regimes.",
    )
    parser.add_argument(
        "--regime-risk-off-shrink-alpha",
        type=float,
        default=0.50,
        help=(
            "Effective shrink alpha for risk-off branch in regime_switch_ewma_shrink. "
            "With --shrink-target zero, trend-on dates use --shrink-alpha while "
            "risk-off dates use this value."
        ),
    )
    parser.add_argument(
        "--min-periods",
        type=int,
        default=20,
        help="Minimum historical returns required before a rolling mean is considered valid.",
    )
    parser.add_argument(
        "--asof-lag",
        type=int,
        default=1,
        help=(
            "Trading-day lag for available returns. 1 means forecast_date t uses returns through "
            "the prior trading day; 0 means same-day close return is allowed. Default: 1."
        ),
    )
    parser.add_argument(
        "--mu-scale",
        type=float,
        default=1.0,
        help="Multiply the rolling daily mu before writing override files.",
    )
    parser.add_argument(
        "--shrink-target",
        default="none",
        choices=SHRINK_TARGET_CHOICES,
        help="Optional shrinkage target applied after model construction and scaling.",
    )
    parser.add_argument(
        "--shrink-alpha",
        type=float,
        default=1.0,
        help="Reliability alpha for shrinkage: 1=no shrink, 0=full target (default: 1).",
    )
    parser.add_argument(
        "--winsor-quantile",
        type=float,
        default=None,
        help="Optional per-date two-sided cross-sectional winsor quantile, e.g. 0.01.",
    )
    parser.add_argument(
        "--no-initial-fallback",
        action="store_true",
        help="Do not fill missing early rolling means from --initial-annualized-csv.",
    )
    parser.add_argument(
        "--missing-mu-fill",
        default="none",
        choices=["none", "zero"],
        help="Optional fallback for forecast dates where the causal mean model is unavailable.",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="Optional first-N ticker limit for smoke tests.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete output dir first if it exists.")
    parser.add_argument(
        "--raw-panel-path",
        default=None,
        help=(
            "Optional raw panel CSV with spot-appended close. When qlib history ends before "
            "--forecast-end-date, supplement the close panel from this file for the missing date."
        ),
    )
    return parser.parse_args()


def compact_date(value: str) -> str:
    return value.replace("-", "")


def dashed_date(value: str) -> str:
    value = value.replace("-", "")
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def infer_window_from_name(path: str | Path | None) -> tuple[str | None, str | None]:
    if not path:
        return None, None
    match = re.search(r"(\d{8})_(\d{8})", Path(path).name)
    if not match:
        return None, None
    return dashed_date(match.group(1)), dashed_date(match.group(2))


def default_history_start(forecast_start_date: str) -> str:
    start = pd.Timestamp(forecast_start_date) - pd.Timedelta(days=370)
    return start.strftime("%Y-%m-%d")


def default_output_dir(
    *,
    history_start_date: str,
    forecast_start_date: str,
    forecast_end_date: str,
) -> Path:
    return Path(DEFAULT_TEMP_DIR) / (
        "route_a_rolling_prior_"
        f"{compact_date(history_start_date)}_"
        f"{compact_date(forecast_start_date)}_{compact_date(forecast_end_date)}"
    )


def load_close_panel(
    tickers: list[str],
    *,
    provider_uri: str,
    start_date: str,
    end_date: str,
    return_qfq_records: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=provider_uri, region="cn")
    raw = D.features(tickers, ["$close"], start_time=start_date, end_time=end_date)
    if raw.empty:
        raise ValueError("qlib returned no close data for the requested history window")
    close = raw["$close"].unstack(level="instrument").sort_index()
    close.index = pd.to_datetime(close.index)
    close.columns = close.columns.astype(str).str.upper()
    close = close.reindex(columns=tickers)
    adjusted, records = auto_qfq_adjust_close_panel(close)
    if not records.empty:
        log_qfq_adjustments(records, stage="load_close_panel")
    if return_qfq_records:
        return adjusted, records
    return adjusted


def supplement_close_from_raw_panel(
    close: pd.DataFrame,
    raw_panel_path: Path,
    *,
    forecast_end_date: str,
) -> pd.DataFrame:
    """Append missing forecast-end close row from a raw panel (e.g. spot-appended)."""
    end_ts = pd.Timestamp(forecast_end_date).normalize()
    if close.index.max() >= end_ts:
        return close
    if not raw_panel_path.exists():
        return close

    raw = pd.read_csv(raw_panel_path, usecols=["instrument", "datetime", "close"], parse_dates=["datetime"])
    raw["instrument"] = raw["instrument"].astype(str).str.upper()
    raw["datetime"] = pd.to_datetime(raw["datetime"]).dt.normalize()
    spot_day = raw.loc[raw["datetime"] == end_ts]
    if spot_day.empty:
        return close

    new_row = pd.Series(index=close.columns, dtype=float)
    for ticker in close.columns:
        values = pd.to_numeric(
            spot_day.loc[spot_day["instrument"] == ticker, "close"],
            errors="coerce",
        )
        if not values.empty and pd.notna(values.iloc[0]):
            new_row[ticker] = float(values.iloc[0])
    if new_row.notna().sum() == 0:
        return close

    supplemented = pd.concat([close, pd.DataFrame([new_row], index=[end_ts])]).sort_index()
    print(
        f"[INFO] Supplemented close panel through {end_ts.date()} from raw panel "
        f"({int(new_row.notna().sum())}/{len(close.columns)} instruments)"
    )
    return supplemented


def load_initial_mu(csv_path: Path | None) -> pd.Series:
    if csv_path is None:
        return pd.Series(dtype=float, name="initial_daily_mu")
    return static_builder.load_daily_mu(csv_path, "mean_daily_return", 1.0).rename("initial_daily_mu")


def rolling_return_count(daily_returns: pd.DataFrame, *, window: int, min_periods: int, asof_lag: int) -> pd.DataFrame:
    counts = daily_returns.rolling(window=int(window), min_periods=int(min_periods)).count()
    return counts.shift(int(asof_lag)) if int(asof_lag) > 0 else counts


def rolling_daily_mu(daily_returns: pd.DataFrame, *, window: int, min_periods: int, asof_lag: int) -> pd.DataFrame:
    rolling = daily_returns.rolling(window=int(window), min_periods=int(min_periods)).mean()
    return rolling.shift(int(asof_lag)) if int(asof_lag) > 0 else rolling


def parse_int_list(value: str | None) -> list[int]:
    if value is None or str(value).strip() == "":
        return []
    out: list[int] = []
    for item in str(value).split(","):
        stripped = item.strip()
        if not stripped:
            continue
        parsed = int(stripped)
        if parsed <= 0:
            raise ValueError(f"Window/span values must be positive: {value}")
        out.append(parsed)
    return out


def parse_float_list(value: str | None) -> list[float]:
    if value is None or str(value).strip() == "":
        return []
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def shift_asof(frame: pd.DataFrame | pd.Series, asof_lag: int) -> pd.DataFrame | pd.Series:
    return frame.shift(int(asof_lag)) if int(asof_lag) > 0 else frame


def ewma_daily_mu(daily_returns: pd.DataFrame, *, span: int, min_periods: int, asof_lag: int) -> pd.DataFrame:
    ewma = daily_returns.ewm(span=int(span), min_periods=int(min_periods), adjust=False).mean()
    return shift_asof(ewma, asof_lag)  # type: ignore[return-value]


def momentum_daily_mu(close: pd.DataFrame, *, window: int, asof_lag: int) -> pd.DataFrame:
    total_return = close / close.shift(int(window)) - 1.0
    daily_mu = np.expm1(np.log1p(total_return.clip(lower=-0.999999)) / float(window))
    return shift_asof(daily_mu, asof_lag)  # type: ignore[return-value]


def vol_adjusted_momentum_daily_mu(
    close: pd.DataFrame,
    daily_returns: pd.DataFrame,
    *,
    momentum_window: int,
    vol_window: int,
    min_periods: int,
    asof_lag: int,
) -> pd.DataFrame:
    momentum_mu = momentum_daily_mu(close, window=int(momentum_window), asof_lag=int(asof_lag))
    vol = daily_returns.rolling(window=int(vol_window), min_periods=int(min_periods)).std()
    vol = shift_asof(vol, asof_lag)  # type: ignore[assignment]
    vol_median = vol.median(axis=1).replace(0.0, np.nan)
    score = momentum_mu / vol.replace(0.0, np.nan)
    return score.mul(vol_median, axis=0)


def weighted_average_panels(panels: list[pd.DataFrame], weights: list[float] | None = None) -> pd.DataFrame:
    if not panels:
        raise ValueError("No panels supplied for weighted average")
    if weights is None or not weights:
        weights = [1.0 / len(panels)] * len(panels)
    if len(weights) != len(panels):
        raise ValueError("--multi-window-weights length must match --multi-windows")
    weight_sum = float(sum(weights))
    if not np.isfinite(weight_sum) or abs(weight_sum) <= 1e-12:
        raise ValueError("Mean model weights must sum to a non-zero value")
    normalized = [float(weight) / weight_sum for weight in weights]
    numerator = pd.DataFrame(0.0, index=panels[0].index, columns=panels[0].columns)
    denominator = pd.DataFrame(0.0, index=panels[0].index, columns=panels[0].columns)
    for panel, weight in zip(panels, normalized):
        aligned = panel.reindex(index=numerator.index, columns=numerator.columns)
        valid = aligned.notna().astype(float)
        numerator = numerator.add(aligned.fillna(0.0) * float(weight), fill_value=0.0)
        denominator = denominator.add(valid * float(weight), fill_value=0.0)
    return numerator / denominator.replace(0.0, np.nan)


def forward_return_panel(close: pd.DataFrame) -> pd.DataFrame:
    return close.shift(-1) / close - 1.0


def cross_sectional_spearman_ic(signal: pd.DataFrame, forward_returns: pd.DataFrame) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    shared_dates = signal.index.intersection(forward_returns.index)
    for date_value in shared_dates:
        work = pd.DataFrame(
            {
                "signal": pd.to_numeric(signal.loc[date_value], errors="coerce"),
                "forward_return": pd.to_numeric(forward_returns.loc[date_value], errors="coerce"),
            }
        ).dropna()
        if len(work) < 3 or work["signal"].nunique() < 2 or work["forward_return"].nunique() < 2:
            values[pd.Timestamp(date_value)] = math.nan
            continue
        value = work["signal"].corr(work["forward_return"], method="spearman")
        values[pd.Timestamp(date_value)] = float(value) if pd.notna(value) else math.nan
    return pd.Series(values, name="spearman_ic").sort_index()


def build_component_panels(
    daily_returns: pd.DataFrame,
    close: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, pd.DataFrame]:
    components: dict[str, pd.DataFrame] = {}
    windows = sorted(set(parse_int_list(args.multi_windows) + [int(args.window)]))
    for window in windows:
        components[f"rolling_{window}"] = rolling_daily_mu(
            daily_returns,
            window=window,
            min_periods=min(int(args.min_periods), window),
            asof_lag=int(args.asof_lag),
        )
    for span in sorted(set(parse_int_list(args.ensemble_ewma_spans) + [int(args.ewma_span)])):
        components[f"ewma_{span}"] = ewma_daily_mu(
            daily_returns,
            span=span,
            min_periods=min(int(args.min_periods), span),
            asof_lag=int(args.asof_lag),
        )
    for window in sorted(set(parse_int_list(args.ensemble_momentum_windows) + [int(args.momentum_window)])):
        components[f"momentum_{window}"] = momentum_daily_mu(close, window=window, asof_lag=int(args.asof_lag))
    for window in sorted(set(parse_int_list(args.ensemble_vol_adjusted_windows) + [int(args.momentum_window)])):
        components[f"vol_adj_momentum_{window}"] = vol_adjusted_momentum_daily_mu(
            close,
            daily_returns,
            momentum_window=window,
            vol_window=int(args.vol_window),
            min_periods=min(int(args.min_periods), int(args.vol_window)),
            asof_lag=int(args.asof_lag),
        )
    return components


def regime_component_multipliers(component_names: list[str], close: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    index = close.index
    multipliers = pd.DataFrame(1.0, index=index, columns=component_names)
    benchmark = str(args.regime_benchmark).upper().strip()
    if benchmark not in close.columns:
        return multipliers
    bench_close = close[benchmark]
    available_close = shift_asof(bench_close, int(args.asof_lag))
    fast_ma = shift_asof(bench_close.rolling(int(args.regime_fast_ma), min_periods=5).mean(), int(args.asof_lag))
    slow_ma = shift_asof(bench_close.rolling(int(args.regime_slow_ma), min_periods=10).mean(), int(args.asof_lag))
    bench_return = bench_close.pct_change(fill_method=None)
    bench_vol = shift_asof(
        bench_return.rolling(int(args.regime_vol_window), min_periods=5).std(),
        int(args.asof_lag),
    )
    vol_reference = bench_vol.rolling(int(args.regime_vol_lookback), min_periods=20).median().shift(1)
    trend_on = (available_close > fast_ma) & (available_close > slow_ma)
    high_vol = bench_vol > vol_reference
    risk_on = trend_on.fillna(False) & ~high_vol.fillna(False)
    risk_off = ~risk_on
    for name in component_names:
        short_or_momentum = (
            name.startswith("momentum_")
            or name.startswith("vol_adj_momentum_")
            or name in {"ewma_20", "ewma_60", "rolling_20", "rolling_60"}
        )
        long_or_anchor = name in {"rolling_120", "rolling_252", "ewma_120", "ewma_252"}
        if short_or_momentum:
            multipliers.loc[risk_on, name] = 1.25
            multipliers.loc[risk_off, name] = 0.50
        elif long_or_anchor:
            multipliers.loc[risk_on, name] = 0.85
            multipliers.loc[risk_off, name] = 1.25
    return multipliers


def benchmark_risk_on_mask(close: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    benchmark = str(args.regime_benchmark).upper().strip()
    if benchmark not in close.columns:
        return pd.Series(True, index=close.index, dtype=bool)
    bench_close = close[benchmark]
    available_close = shift_asof(bench_close, int(args.asof_lag))
    fast_ma = shift_asof(bench_close.rolling(int(args.regime_fast_ma), min_periods=5).mean(), int(args.asof_lag))
    slow_ma = shift_asof(bench_close.rolling(int(args.regime_slow_ma), min_periods=10).mean(), int(args.asof_lag))
    bench_return = bench_close.pct_change(fill_method=None)
    bench_vol = shift_asof(
        bench_return.rolling(int(args.regime_vol_window), min_periods=5).std(),
        int(args.asof_lag),
    )
    vol_reference = bench_vol.rolling(int(args.regime_vol_lookback), min_periods=20).median().shift(1)
    trend_on = (available_close > fast_ma) & (available_close > slow_ma)
    high_vol = bench_vol > vol_reference
    return (trend_on.fillna(False) & ~high_vol.fillna(False)).reindex(close.index).fillna(False).astype(bool)


def where_by_date(mask: pd.Series, trend_panel: pd.DataFrame, fallback_panel: pd.DataFrame) -> pd.DataFrame:
    aligned_mask = mask.reindex(trend_panel.index).fillna(False).astype(bool)
    mask_frame = pd.DataFrame(
        np.repeat(aligned_mask.to_numpy(dtype=bool)[:, None], len(trend_panel.columns), axis=1),
        index=trend_panel.index,
        columns=trend_panel.columns,
    )
    return trend_panel.where(mask_frame, fallback_panel.reindex(index=trend_panel.index, columns=trend_panel.columns))


def ic_weighted_ensemble_panel(
    component_panels: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    args: argparse.Namespace,
    *,
    regime_conditioned: bool,
) -> pd.DataFrame:
    if not component_panels:
        raise ValueError("IC-weighted ensemble requires at least one component")
    forward_returns = forward_return_panel(close)
    component_names = list(component_panels)
    ic_frame = pd.DataFrame(index=close.index, columns=component_names, dtype=float)
    for name, panel in component_panels.items():
        ic_frame[name] = cross_sectional_spearman_ic(panel, forward_returns).reindex(close.index)
    ic_lag = max(1, int(args.asof_lag) + 1)
    weights = ic_frame.rolling(int(args.ic_lookback), min_periods=int(args.ic_min_periods)).mean().shift(ic_lag)
    weights = weights.clip(lower=0.0).fillna(0.0)
    if regime_conditioned:
        weights = weights * regime_component_multipliers(component_names, close, args)

    fallback_name = f"rolling_{int(args.window)}"
    fallback = component_panels.get(fallback_name)
    if fallback is None:
        fallback = next(iter(component_panels.values()))
    weighted_sum = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for name, panel in component_panels.items():
        weighted_sum = weighted_sum.add(panel.mul(weights[name], axis=0), fill_value=0.0)
    weight_sum = weights.sum(axis=1).replace(0.0, np.nan)
    ensemble = weighted_sum.div(weight_sum, axis=0)
    return ensemble.where(weight_sum.notna(), fallback)


def apply_shrinkage(frame: pd.DataFrame, *, target: str, alpha: float) -> pd.DataFrame:
    if target == "none":
        return frame
    clipped_alpha = min(max(float(alpha), 0.0), 1.0)
    if target == "zero":
        target_frame = pd.DataFrame(0.0, index=frame.index, columns=frame.columns)
    elif target == "cross_sectional_median":
        row_median = frame.median(axis=1)
        target_frame = pd.DataFrame({column: row_median for column in frame.columns}, index=frame.index)
    else:
        raise ValueError(f"Unsupported shrink target: {target}")
    return target_frame + clipped_alpha * (frame - target_frame)


def build_mean_model_panel(
    daily_returns: pd.DataFrame,
    close: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    mean_model = str(args.mean_model)
    if mean_model == "rolling_mean":
        raw_mu = rolling_daily_mu(
            daily_returns,
            window=int(args.window),
            min_periods=int(args.min_periods),
            asof_lag=int(args.asof_lag),
        )
    elif mean_model == "ewma":
        raw_mu = ewma_daily_mu(
            daily_returns,
            span=int(args.ewma_span),
            min_periods=min(int(args.min_periods), int(args.ewma_span)),
            asof_lag=int(args.asof_lag),
        )
    elif mean_model == "multi_window_mean":
        windows = parse_int_list(args.multi_windows)
        raw_mu = weighted_average_panels(
            [
                rolling_daily_mu(
                    daily_returns,
                    window=window,
                    min_periods=min(int(args.min_periods), window),
                    asof_lag=int(args.asof_lag),
                )
                for window in windows
            ],
            parse_float_list(args.multi_window_weights),
        )
    elif mean_model == "momentum":
        raw_mu = momentum_daily_mu(close, window=int(args.momentum_window), asof_lag=int(args.asof_lag))
    elif mean_model == "vol_adjusted_momentum":
        raw_mu = vol_adjusted_momentum_daily_mu(
            close,
            daily_returns,
            momentum_window=int(args.momentum_window),
            vol_window=int(args.vol_window),
            min_periods=min(int(args.min_periods), int(args.vol_window)),
            asof_lag=int(args.asof_lag),
        )
    elif mean_model in {"ic_weighted_ensemble", "regime_ic_weighted_ensemble"}:
        raw_mu = ic_weighted_ensemble_panel(
            build_component_panels(daily_returns, close, args),
            close,
            args,
            regime_conditioned=mean_model == "regime_ic_weighted_ensemble",
        )
    elif mean_model == "regime_switch_ewma_multi_window":
        trend_mu = ewma_daily_mu(
            daily_returns,
            span=int(args.ewma_span),
            min_periods=min(int(args.min_periods), int(args.ewma_span)),
            asof_lag=int(args.asof_lag),
        )
        windows = parse_int_list(args.multi_windows)
        fallback_mu = weighted_average_panels(
            [
                rolling_daily_mu(
                    daily_returns,
                    window=window,
                    min_periods=min(int(args.min_periods), window),
                    asof_lag=int(args.asof_lag),
                )
                for window in windows
            ],
            parse_float_list(args.multi_window_weights),
        )
        raw_mu = where_by_date(benchmark_risk_on_mask(close, args), trend_mu, fallback_mu)
    elif mean_model == "regime_switch_ewma_shrink":
        trend_mu = ewma_daily_mu(
            daily_returns,
            span=int(args.ewma_span),
            min_periods=min(int(args.min_periods), int(args.ewma_span)),
            asof_lag=int(args.asof_lag),
        )
        trend_alpha = max(float(args.shrink_alpha), 1e-12)
        risk_off_alpha = min(max(float(args.regime_risk_off_shrink_alpha), 0.0), 1.0)
        fallback_mu = trend_mu * (risk_off_alpha / trend_alpha)
        raw_mu = where_by_date(benchmark_risk_on_mask(close, args), trend_mu, fallback_mu)
    else:
        raise ValueError(f"Unsupported mean model: {mean_model}")

    scaled_mu = raw_mu * float(args.mu_scale)
    shrunk_mu = apply_shrinkage(scaled_mu, target=str(args.shrink_target), alpha=float(args.shrink_alpha))
    final_mu = apply_datewise_winsor(shrunk_mu, args.winsor_quantile)
    max_count_window = max(
        [int(args.window), int(args.ewma_span), int(args.momentum_window), int(args.vol_window)]
        + parse_int_list(args.multi_windows)
        + parse_int_list(args.ensemble_ewma_spans)
        + parse_int_list(args.ensemble_momentum_windows)
        + parse_int_list(args.ensemble_vol_adjusted_windows)
    )
    count_panel = daily_returns.rolling(max_count_window, min_periods=1).count()
    count_panel = shift_asof(count_panel, int(args.asof_lag))  # type: ignore[assignment]
    source_panel = pd.DataFrame(mean_model, index=final_mu.index, columns=final_mu.columns).where(final_mu.notna(), "missing")
    metadata = {
        "mean_model": mean_model,
        "ewma_span": int(args.ewma_span),
        "multi_windows": parse_int_list(args.multi_windows),
        "multi_window_weights": parse_float_list(args.multi_window_weights),
        "momentum_window": int(args.momentum_window),
        "vol_window": int(args.vol_window),
        "ic_lookback": int(args.ic_lookback),
        "ic_min_periods": int(args.ic_min_periods),
        "shrink_target": str(args.shrink_target),
        "shrink_alpha": float(args.shrink_alpha),
        "regime_risk_off_shrink_alpha": float(args.regime_risk_off_shrink_alpha),
    }
    return final_mu, count_panel, source_panel, metadata


def apply_datewise_winsor(frame: pd.DataFrame, quantile: float | None) -> pd.DataFrame:
    if quantile is None:
        return frame
    q = float(quantile)
    if q <= 0.0 or q >= 0.5:
        raise ValueError("--winsor-quantile must be in (0, 0.5)")

    def clip_row(row: pd.Series) -> pd.Series:
        finite = row.dropna()
        if finite.empty:
            return row
        lower = float(finite.quantile(q))
        upper = float(finite.quantile(1.0 - q))
        return row.clip(lower=lower, upper=upper)

    return frame.apply(clip_row, axis=1)


def build_params_frame_variable(
    forecast_dates: list[pd.Timestamp],
    mu_by_date: pd.Series,
    count_by_date: pd.Series,
    source_by_date: pd.Series,
) -> pd.DataFrame:
    date_index = pd.Index([pd.Timestamp(value) for value in forecast_dates], name="forecast_date")
    mu = pd.to_numeric(mu_by_date.reindex(date_index), errors="coerce")
    count = pd.to_numeric(count_by_date.reindex(date_index), errors="coerce")
    source = source_by_date.reindex(date_index).fillna("missing")
    date_text = [value.strftime("%Y-%m-%d") for value in date_index]
    target_dates = static_builder.make_target_dates(list(date_index))
    frame = pd.DataFrame(
        {
            "target_date": target_dates,
            "forecast_date": date_text,
            "mu_raw": mu.to_numpy(dtype=float),
            "mu": mu.to_numpy(dtype=float),
            "mu_smooth": mu.to_numpy(dtype=float),
            "mu_path": mu.to_numpy(dtype=float),
            "posterior_mu_1d": mu.to_numpy(dtype=float),
            "posterior_mu_20d": mu.to_numpy(dtype=float),
            "posterior_mu_std": 0.0,
            "pred_mean_20": mu.to_numpy(dtype=float),
            "real_mean_20": mu.to_numpy(dtype=float),
            "scale_20": 1.0,
            "pred_ann20": mu.to_numpy(dtype=float) * TRADING_DAYS_PER_YEAR,
            "real_ann20": mu.to_numpy(dtype=float) * TRADING_DAYS_PER_YEAR,
            "sigma": pd.NA,
            "nu": 8.0,
            "mu_shrink": mu.to_numpy(dtype=float),
            "signal_strength": mu.abs().to_numpy(dtype=float),
            "fit_success": mu.notna().to_numpy(dtype=bool),
            "fit_fallback": source.astype(str).isin(["initial_csv", "zero_fallback", "missing"]).to_numpy(dtype=bool),
            "fit_message": "rolling_mean_prior_" + source.astype(str),
            "skew": pd.NA,
            "kurt": pd.NA,
            "rolling_return_count": count.to_numpy(dtype=float),
            "rolling_mu_source": source.astype(str).to_numpy(),
        }
    )
    return frame[static_builder.PARAMS_COLUMNS + ["rolling_return_count", "rolling_mu_source"]]


def write_rolling_override_files(
    *,
    tickers: list[str],
    mu_panel: pd.DataFrame,
    count_panel: pd.DataFrame,
    source_panel: pd.DataFrame,
    close_panel: pd.DataFrame,
    output_dir: Path,
    start_label: str,
    end_label: str,
    forecast_start_date: str,
    forecast_end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    daily_records: list[pd.DataFrame] = []
    forecast_mask = (close_panel.index >= pd.Timestamp(forecast_start_date)) & (
        close_panel.index <= pd.Timestamp(forecast_end_date)
    )
    for ticker in tickers:
        ticker_dates = [
            pd.Timestamp(value)
            for value in close_panel.index[forecast_mask & close_panel[ticker].notna()]
        ]
        status = "ok"
        message = ""
        if not ticker_dates:
            status = "missing_dates"
            message = "no forecast dates available"
            missing_mu_rows = 0
            seed_filled_rows = 0
            rolling_rows = 0
        else:
            date_index = pd.Index(ticker_dates, name="forecast_date")
            mu_by_date = mu_panel[ticker].reindex(date_index)
            count_by_date = count_panel[ticker].reindex(date_index)
            source_by_date = source_panel[ticker].reindex(date_index)
            missing_mu_rows = int(mu_by_date.isna().sum())
            source_text = source_by_date.astype(str)
            seed_filled_rows = int((source_text == "initial_csv").sum())
            zero_filled_rows = int((source_text == "zero_fallback").sum())
            rolling_rows = int((~source_text.isin(["initial_csv", "zero_fallback", "missing"])).sum())
            if missing_mu_rows:
                status = "missing_mu_dates"
                message = f"{missing_mu_rows} forecast dates have no rolling or fallback mu"
            else:
                params = build_params_frame_variable(ticker_dates, mu_by_date, count_by_date, source_by_date)
                eval_frame = static_builder.build_eval_frame(params)
                params_path = output_dir / f"route_a_params_{ticker}_{start_label}_{end_label}.csv"
                eval_path = output_dir / f"route_a_eval_daily_{ticker}_{start_label}_{end_label}.csv"
                params.to_csv(params_path, index=False)
                eval_frame.to_csv(eval_path, index=False)
                message = f"wrote {len(params)} forecast rows"
                daily_records.append(
                    pd.DataFrame(
                        {
                            "instrument": ticker,
                            "forecast_date": [value.strftime("%Y-%m-%d") for value in date_index],
                            "mean_daily_return": mu_by_date.to_numpy(dtype=float),
                            "annualized_return": mu_by_date.to_numpy(dtype=float) * TRADING_DAYS_PER_YEAR,
                            "rolling_return_count": count_by_date.to_numpy(dtype=float),
                            "mu_source": source_by_date.astype(str).to_numpy(),
                            "mean_model": source_by_date.astype(str).to_numpy(),
                        }
                    )
                )
        first_mu = mu_panel[ticker].dropna().iloc[0] if ticker in mu_panel and mu_panel[ticker].notna().any() else math.nan
        last_mu = mu_panel[ticker].dropna().iloc[-1] if ticker in mu_panel and mu_panel[ticker].notna().any() else math.nan
        rows.append(
            {
                "instrument": ticker,
                "status": status,
                "message": message,
                "forecast_rows": len(ticker_dates),
                "rolling_rows": rolling_rows,
                "initial_csv_filled_rows": seed_filled_rows,
                "zero_filled_rows": zero_filled_rows if ticker_dates else 0,
                "missing_mu_rows": missing_mu_rows,
                "first_daily_mu": first_mu,
                "last_daily_mu": last_mu,
                "first_annualized_mu_arithmetic": first_mu * TRADING_DAYS_PER_YEAR if pd.notna(first_mu) else math.nan,
                "last_annualized_mu_arithmetic": last_mu * TRADING_DAYS_PER_YEAR if pd.notna(last_mu) else math.nan,
            }
        )
    daily_prior = pd.concat(daily_records, ignore_index=True) if daily_records else pd.DataFrame()
    return pd.DataFrame(rows), daily_prior


def main() -> int:
    args = parse_args()
    inferred_history_start, inferred_history_end = infer_window_from_name(args.initial_annualized_csv)
    history_start_date = args.history_start_date or inferred_history_start or default_history_start(args.forecast_start_date)
    history_end_date = args.history_end_date or inferred_history_end or args.forecast_start_date
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output_dir(
            history_start_date=history_start_date,
            forecast_start_date=args.forecast_start_date,
            forecast_end_date=args.forecast_end_date,
        )
    )
    cluster_mapping = Path(args.cluster_mapping).expanduser().resolve()
    initial_csv = Path(args.initial_annualized_csv).expanduser().resolve() if args.initial_annualized_csv else None

    if initial_csv is not None and not initial_csv.exists():
        raise FileNotFoundError(f"initial annualized CSV not found: {initial_csv}")
    if not cluster_mapping.exists():
        raise FileNotFoundError(f"cluster mapping not found: {cluster_mapping}")
    if int(args.window) <= 0:
        raise ValueError("--window must be positive")
    if int(args.min_periods) <= 0 or int(args.min_periods) > int(args.window):
        raise ValueError("--min-periods must be in [1, window]")
    if int(args.asof_lag) < 0:
        raise ValueError("--asof-lag must be >= 0")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output dir exists; pass --overwrite to replace: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers = static_builder.read_cluster_tickers(cluster_mapping)
    if args.max_tickers is not None:
        tickers = tickers[: int(args.max_tickers)]
    load_tickers = list(tickers)
    benchmark = str(getattr(args, "regime_benchmark", "")).upper().strip()
    regime_models = {"regime_ic_weighted_ensemble", "regime_switch_ewma_multi_window", "regime_switch_ewma_shrink"}
    if args.mean_model in regime_models and benchmark and benchmark not in load_tickers:
        load_tickers.append(benchmark)
    load_start = min(pd.Timestamp(history_start_date), pd.Timestamp(args.forecast_start_date)).strftime("%Y-%m-%d")
    close, qfq_records = load_close_panel(
        load_tickers,
        provider_uri=args.provider_uri,
        start_date=load_start,
        end_date=args.forecast_end_date,
        return_qfq_records=True,
    )
    if not qfq_records.empty:
        qfq_path = output_dir / "qfq_adjustments.csv"
        qfq_records.to_csv(qfq_path, index=False)
        print(f"[INFO] Wrote qfq adjustment audit: {qfq_path}")
    if args.raw_panel_path:
        close = supplement_close_from_raw_panel(
            close,
            Path(args.raw_panel_path).expanduser().resolve(),
            forecast_end_date=args.forecast_end_date,
        )
    daily_returns = close.pct_change(fill_method=None)
    rolling_mu, rolling_count, source_panel, model_metadata = build_mean_model_panel(daily_returns, close, args)

    initial_mu = load_initial_mu(initial_csv)
    if initial_csv is not None and not args.no_initial_fallback:
        for ticker in tickers:
            seed_value = initial_mu.get(ticker, math.nan)
            if pd.isna(seed_value):
                continue
            missing_mask = rolling_mu[ticker].isna()
            rolling_mu.loc[missing_mask, ticker] = float(seed_value) * float(args.mu_scale)
            rolling_count.loc[missing_mask, ticker] = rolling_count.loc[missing_mask, ticker].fillna(0.0)
            source_panel.loc[missing_mask, ticker] = "initial_csv"

    if args.missing_mu_fill == "zero":
        forecast_mask = (close.index >= pd.Timestamp(args.forecast_start_date)) & (
            close.index <= pd.Timestamp(args.forecast_end_date)
        )
        for ticker in tickers:
            missing_mask = forecast_mask & close[ticker].notna() & rolling_mu[ticker].isna()
            rolling_mu.loc[missing_mask, ticker] = 0.0
            rolling_count.loc[missing_mask, ticker] = rolling_count.loc[missing_mask, ticker].fillna(0.0)
            source_panel.loc[missing_mask, ticker] = "zero_fallback"

    start_label = compact_date(args.forecast_start_date)
    end_label = compact_date(args.forecast_end_date)
    status, daily_prior = write_rolling_override_files(
        tickers=tickers,
        mu_panel=rolling_mu,
        count_panel=rolling_count,
        source_panel=source_panel,
        close_panel=close,
        output_dir=output_dir,
        start_label=start_label,
        end_label=end_label,
        forecast_start_date=args.forecast_start_date,
        forecast_end_date=args.forecast_end_date,
    )
    status_path = output_dir / "rolling_mean_override_status.csv"
    daily_prior_path = output_dir / "rolling_mean_prior_daily.csv"
    status.to_csv(status_path, index=False)
    daily_prior.to_csv(daily_prior_path, index=False)

    manifest = {
        "mode": "causal_rolling_mean_prior_route_a_override",
        "mean_model": str(args.mean_model),
        "forecast_start_date": args.forecast_start_date,
        "forecast_end_date": args.forecast_end_date,
        "history_start_date": history_start_date,
        "history_end_date": history_end_date,
        "initial_annualized_csv": str(initial_csv) if initial_csv is not None else None,
        "cluster_mapping": str(cluster_mapping),
        "provider_uri": str(args.provider_uri),
        "output_dir": str(output_dir),
        "window": int(args.window),
        "min_periods": int(args.min_periods),
        "asof_lag": int(args.asof_lag),
        "mu_scale": float(args.mu_scale),
        "missing_mu_fill": str(args.missing_mu_fill),
        "winsor_quantile": args.winsor_quantile,
        "model_metadata": model_metadata,
        "initial_fallback": bool(initial_csv is not None and not args.no_initial_fallback),
        "tickers_requested": int(len(tickers)),
        "tickers_ok": int((status["status"].astype(str) == "ok").sum()),
        "params_files": int(len(list(output_dir.glob("route_a_params_*.csv")))),
        "eval_files": int(len(list(output_dir.glob("route_a_eval_daily_*.csv")))),
        "daily_prior_rows": int(len(daily_prior)),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote rolling Route-A override to: {output_dir}")
    print(f"Status CSV: {status_path}")
    print(f"Daily prior CSV: {daily_prior_path}")
    print(f"Manifest: {manifest_path}")
    print(status["status"].value_counts(dropna=False).to_string())
    if not (status["status"].astype(str) == "ok").all():
        print(status[status["status"].astype(str) != "ok"].head(20).to_string(index=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())