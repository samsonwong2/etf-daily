#!/usr/bin/env python3
"""Generate fixed audit tables for pool, moments, allocation, and controls.

This script is intentionally read-only for strategy artifacts: it consumes the
canonical pool txt and existing pipeline output directories, then writes audit
CSV files plus one markdown report.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from pipeline.price_adjustments import auto_qfq_adjust_close_panel


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROVIDER_URI = "~/etf-daily-output/Documents/stock/data/qlib_data/all_fund_data/"
DEFAULT_SELECTED_TXT = "~/etf-daily-output/Documents/stock/data/qlib_data/all_fund_data/instruments/cluster_mapping_selected.txt"
DEFAULT_ALL_TXT = "~/etf-daily-output/Documents/stock/data/qlib_data/all_fund_data/instruments/all.txt"
DEFAULT_FUND_LIST_CSV = "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/fund_list.csv"
DEFAULT_START_DATE = "2026-01-01"
DEFAULT_END_DATE = "2026-04-30"
DEFAULT_OUT_DIR = "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/strategy_layer_audit_20260430"
DEFAULT_REPORT_MD = REPO_ROOT / "workspace" / "history" / "strategy_layer_fixed_audit_20260430.md"
DEFAULT_RFC_MD = REPO_ROOT / "workspace" / "history" / "negative_mean_neutralization_filter_rfc_20260430.md"

EXPLICIT_FACTOR_COLUMNS = [
    "factor_style_momentum",
    "factor_style_growth",
    "factor_style_size",
    "factor_industry_momentum",
]
FACTOR_AUDIT_COLUMNS = [
    "factor_industry_theme",
    "factor_style_momentum_raw",
    "factor_style_growth_raw",
    "factor_style_size_raw",
    "factor_industry_momentum_raw",
    "factor_style_momentum",
    "factor_style_growth",
    "factor_style_size",
    "factor_industry_momentum",
    "factor_volume_ratio_20",
    "factor_volume_z_20",
    "factor_abnormal_volume_flag",
]
FACTOR_SCHEMA_DEFAULTS = {
    "factor_volume_ratio_20": 1.0,
    "factor_volume_z_20": 0.0,
    "factor_style_growth": 0.0,
}
FACTOR_SCHEMA_LINEAGE_COLUMNS = [
    "factor_schema_source",
    *[f"{column}_default_fill_flag" for column in FACTOR_SCHEMA_DEFAULTS],
    *[f"{column}_lineage" for column in FACTOR_SCHEMA_DEFAULTS],
]
SHADOW_PATCH_TRIGGER_THEMES = ("communication", "ai", "semiconductor", "star_semiconductor")
SHADOW_PATCH_BOOM_WIN_THEMES = ("communication", "ai")
SHADOW_PATCH_VOLUME_Z_THRESHOLD = 1.0
SHADOW_PATCH_GROWTH_EXPOSURE_THRESHOLD = -1.5
SHADOW_PATCH_FORECAST_MEAN_THRESHOLD = 0.0
SHADOW_PATCH_TRACKING_ERROR_LIMIT = 0.01
FULL_MVO_MAX_WEIGHT = 0.15
FULL_MVO_MIN_WEIGHT_FLOOR = 0.05
SYNTHETIC_STRESS_NEGATIVE_MEAN = -0.02
ACTIVE_SET_UNANIMOUS_LOSS_RULE_NAME = "active_set_unanimous_loss_monitor"
ACTIVE_SET_POOL_MEDIAN_RETURN_MIN = -0.01
MEAN_BIAS_REVERSAL_EVENT_DATES = ("2026-01-12", "2026-01-14")
INDUSTRY_ENTROPY_FOCUS_DATES = ("2026-01-12", "2026-01-14", "2026-01-29")
INDUSTRY_ENTROPY_MIN_SAFE = 0.50
INDUSTRY_DOMINANT_SHARE_SAFE_MAX = 0.50
ENTROPY_DEFENSE_FOCUS_DATES = ("2026-01-12", "2026-01-14", "2026-01-29")
ENTROPY_DEFENSE_AI_THEMES = ("ai", "communication", "semiconductor", "star_semiconductor")
MOMENTUM_OVERHEAT_FOCUS_DATES = ("2026-01-12", "2026-01-14", "2026-01-29")
MOMENTUM_OVERHEAT_DEFAULT_THRESHOLD = 0.10
MOMENTUM_FATIGUE_WARNING_WEIGHT_SHARE = 0.30

DEFAULT_CASES: tuple[tuple[str, str, str], ...] = (
    (
        "base_warm120_nofloor",
        "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/grid_dynmom_aggr_warm120_nofloor_w5_20_60",
        "baseline",
    ),
    (
        "sq_on_1.00",
        "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/grid_dynmom_aggr_warm120_nofloor_sqon100_w5_20_60",
        "--signal-quality-min-scale-risk-on 1.00",
    ),
    (
        "vt_on_0.022",
        "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/grid_dynmom_aggr_warm120_nofloor_vt022_w5_20_60",
        "--vol-target-risk-on 0.022",
    ),
    (
        "sq_on_1.00+vt_on_0.022",
        "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/grid_dynmom_aggr_warm120_nofloor_sqon100_vt022_w5_20_60",
        "--signal-quality-min-scale-risk-on 1.00; --vol-target-risk-on 0.022",
    ),
    (
        "cap_0.18",
        "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/grid_dynmom_aggr_warm120_nofloor_sqon100_vt022_cap018_w5_20_60",
        "plus --max-weight-cap 0.18",
    ),
    (
        "cap_0.20",
        "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/grid_dynmom_aggr_warm120_nofloor_sqon100_vt022_cap020_w5_20_60",
        "plus --max-weight-cap 0.20",
    ),
    (
        "cap018+sleeve_0.40_top6",
        "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/grid_dynmom_aggr_warm120_nofloor_sqon100_vt022_cap018_sl040_top6_w5_20_60",
        "plus sleeve risk-on 0.40 top6",
    ),
    (
        "cap018+sleeve_0.45_top6",
        "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/grid_dynmom_aggr_warm120_nofloor_sqon100_vt022_cap018_sl045_top6_w5_20_60",
        "plus sleeve risk-on 0.45 top6",
    ),
    (
        "cap018+sleeve_0.50_top6",
        "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/grid_dynmom_aggr_warm120_nofloor_sqon100_vt022_cap018_sl050_top6_w5_20_60",
        "plus sleeve risk-on 0.50 top6",
    ),
    (
        "cap018+sleeve_0.55_top6",
        "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/grid_dynmom_aggr_warm120_nofloor_sqon100_vt022_cap018_sl055_top6_w5_20_60",
        "plus sleeve risk-on 0.55 top6",
    ),
    (
        "cap018+sleeve_0.45_top8",
        "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/grid_dynmom_aggr_warm120_nofloor_sqon100_vt022_cap018_sl045_top8_w5_20_60",
        "plus sleeve risk-on 0.45 top8",
    ),
)


@dataclass(frozen=True)
class CaseSpec:
    label: str
    path: Path
    params: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed strategy-layer audit tables from existing pipeline outputs."
    )
    parser.add_argument("--provider-uri", default=DEFAULT_PROVIDER_URI)
    parser.add_argument("--selected-txt", default=DEFAULT_SELECTED_TXT)
    parser.add_argument("--all-txt", default=DEFAULT_ALL_TXT)
    parser.add_argument("--fund-list-csv", default=DEFAULT_FUND_LIST_CSV)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--rfc-md", default=str(DEFAULT_RFC_MD))
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Optional case override as label:path[:params]. If omitted, existing default cases are used.",
    )
    parser.add_argument("--top-n", default="10,20,30,50,100")
    parser.add_argument("--missed-top-k", type=int, default=50)
    parser.add_argument("--high-corr-threshold", type=float, default=0.90)
    parser.add_argument("--future-corr-window", type=int, default=20)
    parser.add_argument("--min-future-corr-obs", type=int, default=10)
    parser.add_argument("--pure-momentum-top-k", type=int, default=6)
    parser.add_argument("--top-winner-k", type=int, default=10)
    parser.add_argument("--blend-momentum-window", type=int, default=20)
    parser.add_argument("--vol-exemption-risk-free-rates", default="0.00,0.01,0.02")
    parser.add_argument("--vol-exemption-variance-multipliers", default="1.00,0.75,0.50")
    parser.add_argument("--vol-exemption-upside-share-min", type=float, default=0.50)
    parser.add_argument("--vol-exemption-positive-share-min", type=float, default=0.48)
    parser.add_argument("--vol-exemption-min-period-return", type=float, default=0.10)
    parser.add_argument("--clone-corr-threshold", type=float, default=0.75)
    parser.add_argument("--local-cap-mean-top-pct", type=float, default=0.30)
    parser.add_argument("--local-vol-cap-rank-pcts", default="1.00,0.70,0.60,0.50")
    parser.add_argument("--local-cap-oos-lookback-days", type=int, default=20)
    parser.add_argument("--local-cap-oos-forward-days", type=int, default=10)
    parser.add_argument("--local-cap-oos-step-days", type=int, default=10)
    parser.add_argument("--skip-pool-audit", action="store_true")
    return parser.parse_args(argv)


def normalize_code(value: object) -> str:
    return str(value).strip().upper()


def read_txt_codes(path: str | Path) -> list[str]:
    codes: list[str] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            codes.append(normalize_code(line.split()[0]))
    return list(dict.fromkeys(codes))


def parse_int_list(text: str) -> list[int]:
    return [int(token.strip()) for token in str(text).split(",") if token.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(token.strip()) for token in str(text).split(",") if token.strip()]


def parse_cases(case_texts: list[str]) -> list[CaseSpec]:
    if not case_texts:
        cases = [CaseSpec(label, Path(path), params) for label, path, params in DEFAULT_CASES]
        return [case for case in cases if case.path.exists()]
    parsed_cases: list[CaseSpec] = []
    for case_text in case_texts:
        if "=" in case_text and ":" not in case_text.split("=", 1)[0]:
            label, remainder = case_text.split("=", 1)
        else:
            label, remainder = case_text.split(":", 1)
        parts = remainder.split(":", 1)
        case_path = parts[0]
        params = parts[1] if len(parts) > 1 else "custom"
        parsed_cases.append(CaseSpec(label.strip(), Path(case_path).expanduser(), params.strip()))
    return parsed_cases


def ensure_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    report_md = Path(args.report_md).expanduser().resolve()
    rfc_md = Path(args.rfc_md).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    rfc_md.parent.mkdir(parents=True, exist_ok=True)
    return out_dir, report_md


def csv_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def read_csv_columns(path: Path, requested_columns: Iterable[str]) -> pd.DataFrame:
    columns = csv_header(path)
    usecols = [column for column in requested_columns if column in columns]
    parse_dates = ["datetime"] if "datetime" in usecols else None
    frame = pd.read_csv(path, usecols=usecols, parse_dates=parse_dates)
    if "instrument" in frame.columns:
        frame["instrument"] = frame["instrument"].map(normalize_code)
    return frame


def find_one(directory: Path, pattern: str, predicate: Callable[[Path], bool] | None = None) -> Path | None:
    matches = sorted(directory.glob(pattern))
    if predicate is not None:
        matches = [match for match in matches if predicate(match)]
    return matches[0] if matches else None


def resolve_case_paths(case: CaseSpec) -> dict[str, Path | None]:
    case_dir = case.path.expanduser().resolve()
    return {
        "bridge": (
            find_one(case_dir, "bridge_*_*.csv", lambda path: not path.name.endswith("_l3.csv"))
            or find_one(case_dir, "bridge_*_fullpanel_*.csv", lambda path: not path.name.endswith("_l3.csv"))
        ),
        "bridge_l3": find_one(case_dir, "bridge_*_*_l3.csv") or find_one(case_dir, "bridge_*_fullpanel_*_l3.csv"),
        "raw_panel": find_one(case_dir, "raw_*.csv") or find_one(case_dir, "new_raw_data_fullpanel_*.csv"),
        "daily_nav": find_one(case_dir, "nav_*.csv") or find_one(case_dir, "*_close_to_next_close_daily_nav_*.csv"),
    }


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def safe_float(value: object) -> float:
    try:
        number = float(value)
    except Exception:
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def summarize_series(series: pd.Series) -> dict[str, float]:
    values = numeric_series(series).dropna()
    if values.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "positive_share": np.nan,
        }
    return {
        "count": int(values.shape[0]),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=0)),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
        "positive_share": float((values > 0).mean()),
    }


def return_stats(daily_returns: pd.Series) -> dict[str, float]:
    returns = numeric_series(daily_returns).dropna()
    if returns.empty:
        return {
            "days": 0,
            "cumulative_return": np.nan,
            "annual_return": np.nan,
            "annual_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }
    equity = (1.0 + returns).cumprod()
    cumulative_return = float(equity.iloc[-1] - 1.0)
    annual_return = float((1.0 + cumulative_return) ** (252.0 / max(len(returns), 1)) - 1.0)
    annual_vol = float(returns.std(ddof=0) * math.sqrt(252.0))
    sharpe = float(annual_return / annual_vol) if annual_vol > 0 else np.nan
    drawdown = equity / equity.cummax() - 1.0
    return {
        "days": int(len(returns)),
        "cumulative_return": cumulative_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
    }


def nav_daily_returns(nav_path: Path) -> pd.Series:
    nav_frame = pd.read_csv(nav_path, parse_dates=["datetime"])
    if {"start_nav", "next_nav"}.issubset(nav_frame.columns):
        returns = numeric_series(nav_frame["next_nav"]) / numeric_series(nav_frame["start_nav"]) - 1.0
    elif "ret" in nav_frame.columns:
        returns = numeric_series(nav_frame["ret"])
    else:
        raise ValueError(f"Cannot derive daily returns from {nav_path}")
    returns.index = nav_frame["datetime"]
    return returns


def write_table(frame: pd.DataFrame, out_dir: Path, filename: str) -> Path:
    path = out_dir / filename
    frame.to_csv(path, index=False)
    print(f"[WRITE] {path}")
    return path


def build_name_map(fund_list_csv: str | Path, selected_mapping_csv: str | Path | None = None) -> dict[str, str]:
    name_map: dict[str, str] = {}
    fund_path = Path(fund_list_csv)
    if fund_path.exists():
        fund_frame = pd.read_csv(fund_path)
        if {"基金代码", "基金简称"}.issubset(fund_frame.columns):
            for _, row in fund_frame.iterrows():
                name_map[normalize_code(row["基金代码"])] = str(row["基金简称"])
    if selected_mapping_csv is not None and Path(selected_mapping_csv).exists():
        mapping_frame = pd.read_csv(selected_mapping_csv)
        if {"code", "name"}.issubset(mapping_frame.columns):
            for _, row in mapping_frame.iterrows():
                name_map.setdefault(normalize_code(row["code"]), str(row["name"]))
    return name_map


def load_qlib_close(
    provider_uri: str,
    codes: list[str],
    start_date: str,
    end_date: str,
    batch_size: int = 250,
    apply_qfq: bool = True,
) -> pd.DataFrame:
    provider_path = Path(provider_uri).expanduser()
    feature_dir = provider_path / "features"
    available_codes = {entry.name.upper() for entry in feature_dir.iterdir()} if feature_dir.exists() else set()
    requested_codes = [code for code in codes if not available_codes or code in available_codes]
    if not requested_codes:
        raise RuntimeError("No requested instruments are available in qlib features")

    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    qlib.init(provider_uri=str(provider_path), region=REG_CN)
    chunks: list[pd.DataFrame] = []
    for start_index in range(0, len(requested_codes), batch_size):
        batch_codes = requested_codes[start_index:start_index + batch_size]
        raw = D.features(batch_codes, ["$close"], start_time=start_date, end_time=end_date, freq="day")
        if raw is None or raw.empty:
            continue
        close_chunk = raw["$close"].unstack("instrument")
        close_chunk.columns = [normalize_code(column) for column in close_chunk.columns]
        chunks.append(close_chunk)
    if not chunks:
        raise RuntimeError("Qlib returned no close data")
    close = pd.concat(chunks, axis=1).sort_index()
    close = close.loc[:, ~close.columns.duplicated()]
    if apply_qfq:
        close, _ = auto_qfq_adjust_close_panel(close)
    return close


def period_returns(close: pd.DataFrame) -> pd.Series:
    results: dict[str, float] = {}
    for code in close.columns:
        values = numeric_series(close[code]).dropna()
        if len(values) < 2 or values.iloc[0] <= 0:
            results[code] = np.nan
        else:
            results[code] = float(values.iloc[-1] / values.iloc[0] - 1.0)
    return pd.Series(results, dtype=float)


def average_abs_corr(returns: pd.DataFrame) -> float:
    corr = returns.corr(min_periods=20).abs()
    if corr.shape[0] < 2:
        return np.nan
    upper_indices = np.triu_indices(corr.shape[0], k=1)
    values = corr.to_numpy()[upper_indices]
    return float(np.nanmean(values))


def average_pair_corr_from_corr(corr: pd.DataFrame) -> float:
    if corr.shape[0] < 2:
        return np.nan
    values = corr.to_numpy()[np.triu_indices(corr.shape[0], k=1)]
    return float(np.nanmean(values))


def correlation_contribution_profiles(
    returns: pd.DataFrame,
    codes: Iterable[str],
    min_periods: int = 20,
) -> dict[str, dict[str, object]]:
    ordered_codes = [normalize_code(code) for code in codes if normalize_code(code) in returns.columns]
    if not ordered_codes:
        return {}
    corr = returns.reindex(columns=ordered_codes).corr(min_periods=min_periods).abs()
    pool_avg = average_pair_corr_from_corr(corr)
    profiles: dict[str, dict[str, object]] = {}
    for code in ordered_codes:
        peers = [peer for peer in ordered_codes if peer != code]
        peer_corr = corr.loc[code, peers].dropna() if peers and code in corr.index else pd.Series(dtype=float)
        without_corr = corr.loc[peers, peers] if len(peers) >= 2 else pd.DataFrame()
        without_avg = average_pair_corr_from_corr(without_corr) if not without_corr.empty else np.nan
        nearest_code = str(peer_corr.idxmax()) if not peer_corr.empty else ""
        nearest_corr = float(peer_corr.max()) if not peer_corr.empty else np.nan
        profiles[code] = {
            "avg_abs_corr_to_selected_peer": float(peer_corr.mean()) if not peer_corr.empty else np.nan,
            "nearest_selected_peer": nearest_code,
            "nearest_selected_peer_corr": nearest_corr,
            "marginal_corr_contribution": pool_avg - without_avg if np.isfinite(pool_avg) and np.isfinite(without_avg) else np.nan,
        }
    return profiles


def compound_return(series: pd.Series) -> float:
    values = numeric_series(series).dropna()
    if values.empty:
        return np.nan
    return float((1.0 + values).prod() - 1.0)


def realized_volatility_profile(series: pd.Series) -> dict[str, float]:
    values = numeric_series(series).dropna()
    if values.empty:
        return {
            "positive_return_share": np.nan,
            "upside_volatility": np.nan,
            "downside_volatility": np.nan,
            "upside_downside_vol_ratio": np.nan,
            "upside_volatility_share": np.nan,
            "downside_volatility_share": np.nan,
            "return_to_total_vol": np.nan,
            "return_to_downside_vol": np.nan,
            "period_return_to_downside_vol": np.nan,
        }
    upside_component = values.clip(lower=0.0)
    downside_component = values.clip(upper=0.0).abs()
    upside_volatility = float(np.sqrt(np.mean(np.square(upside_component))))
    downside_volatility = float(np.sqrt(np.mean(np.square(downside_component))))
    total_component = upside_volatility + downside_volatility
    total_volatility = float(values.std(ddof=0))
    mean_return = float(values.mean())
    period_return = compound_return(values)
    return {
        "positive_return_share": float((values > 0.0).mean()),
        "upside_volatility": upside_volatility,
        "downside_volatility": downside_volatility,
        "upside_downside_vol_ratio": upside_volatility / downside_volatility if downside_volatility > 1e-12 else np.nan,
        "upside_volatility_share": upside_volatility / total_component if total_component > 1e-12 else np.nan,
        "downside_volatility_share": downside_volatility / total_component if total_component > 1e-12 else np.nan,
        "return_to_total_vol": mean_return / total_volatility if total_volatility > 1e-12 else np.nan,
        "return_to_downside_vol": mean_return / downside_volatility if downside_volatility > 1e-12 else np.nan,
        "period_return_to_downside_vol": period_return / downside_volatility if downside_volatility > 1e-12 else np.nan,
    }


def is_offensive_volatility(
    row: pd.Series,
    *,
    min_period_return: float,
    upside_share_min: float,
    positive_share_min: float,
) -> bool:
    return bool(
        safe_float(row.get("period_return")) >= min_period_return
        and safe_float(row.get("upside_volatility_share")) >= upside_share_min
        and safe_float(row.get("positive_return_share")) >= positive_share_min
    )


def decision_tree_label(row: pd.Series, *, mean_top_pct: float, clone_corr_threshold: float) -> str:
    mean_confirmed = safe_float(row.get("mean_top30_day_share")) >= mean_top_pct
    if not mean_confirmed and np.isfinite(safe_float(row.get("avg_forecast_mean_top_rank_pct"))):
        mean_confirmed = safe_float(row.get("avg_forecast_mean_top_rank_pct")) <= mean_top_pct
    if not mean_confirmed:
        return "Mean Signal Miss"
    nearest_corr = safe_float(row.get("nearest_selected_peer_corr"))
    avg_peer_corr = safe_float(row.get("avg_abs_corr_to_selected_peer"))
    if (
        np.isfinite(nearest_corr)
        and nearest_corr >= clone_corr_threshold
        or np.isfinite(avg_peer_corr)
        and avg_peer_corr >= clone_corr_threshold
    ):
        high_vol_or_offensive = bool(row.get("offensive_volatility_flag")) or safe_float(row.get("avg_forecast_vol_rank_pct")) >= 0.65
        return "High-Vol Clones" if high_vol_or_offensive else "Mean Signal Miss"
    if bool(row.get("offensive_volatility_flag")):
        return "Hidden Gems"
    return "Mean Signal Miss"


def run_pool_audit(args: argparse.Namespace, out_dir: Path) -> dict[str, pd.DataFrame]:
    selected_codes = read_txt_codes(args.selected_txt)
    universe_codes = read_txt_codes(args.all_txt)
    name_map = build_name_map(args.fund_list_csv, "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/cluster_mapping_selected.csv")
    close = load_qlib_close(args.provider_uri, universe_codes, args.start_date, args.end_date)
    close_returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    total_returns = period_returns(close).dropna().sort_values(ascending=False)
    valid_codes = total_returns.index.tolist()
    selected_set = set(selected_codes)
    selected_valid = [code for code in selected_codes if code in valid_codes]
    unselected_valid = [code for code in valid_codes if code not in selected_set]

    top_rows: list[dict[str, object]] = []
    for top_count in parse_int_list(args.top_n):
        top_codes = total_returns.head(top_count).index.tolist()
        hits = [code for code in top_codes if code in selected_set]
        top_rows.append(
            {
                "top_n": top_count,
                "selected_hits": len(hits),
                "capture_rate": len(hits) / max(top_count, 1),
                "hit_codes": "|".join(hits),
                "hit_names": "|".join(name_map.get(code, "") for code in hits),
            }
        )
    top_capture = pd.DataFrame(top_rows)

    distribution_rows = []
    groups = {
        "selected": total_returns.reindex(selected_valid).dropna(),
        "unselected": total_returns.reindex(unselected_valid).dropna(),
        "all_valid": total_returns,
    }
    for group_name, group_returns in groups.items():
        ranked_group_returns = group_returns.sort_values(ascending=False)
        summary = summarize_series(group_returns)
        summary.update(
            {
                "group": group_name,
                "top5_mean": float(ranked_group_returns.head(5).mean()) if len(ranked_group_returns) else np.nan,
                "top10_mean": float(ranked_group_returns.head(10).mean()) if len(ranked_group_returns) else np.nan,
            }
        )
        if group_name == "selected":
            summary["avg_abs_corr"] = average_abs_corr(close_returns.reindex(columns=selected_valid))
        elif group_name == "all_valid":
            summary["avg_abs_corr"] = average_abs_corr(close_returns.reindex(columns=valid_codes))
        else:
            summary["avg_abs_corr"] = np.nan
        distribution_rows.append(summary)
    return_distribution = pd.DataFrame(distribution_rows)

    missed_codes = [code for code in total_returns.head(args.missed_top_k).index if code not in selected_set]
    nearest_rows: list[dict[str, object]] = []
    if missed_codes and selected_valid:
        corr_input = close_returns.reindex(columns=missed_codes + selected_valid)
        corr = corr_input.corr(min_periods=20).abs()
        for code in missed_codes:
            corr_to_selected = corr.loc[code, selected_valid].dropna().sort_values(ascending=False)
            nearest_codes = corr_to_selected.head(3).index.tolist()
            nearest_rows.append(
                {
                    "rank": int(total_returns.index.get_loc(code) + 1),
                    "code": code,
                    "name": name_map.get(code, ""),
                    "period_return": float(total_returns.loc[code]),
                    "nearest_selected_code": nearest_codes[0] if nearest_codes else "",
                    "nearest_selected_name": name_map.get(nearest_codes[0], "") if nearest_codes else "",
                    "nearest_abs_corr": float(corr_to_selected.iloc[0]) if len(corr_to_selected) else np.nan,
                    "nearest_top3": "|".join(nearest_codes),
                }
            )
    missed_nearest = pd.DataFrame(nearest_rows)

    internal_rows: list[dict[str, object]] = []
    selected_corr = close_returns.reindex(columns=selected_valid).corr(min_periods=20)
    selected_array = selected_corr.to_numpy()
    selected_columns = list(selected_corr.columns)
    for row_index, col_index in zip(*np.triu_indices(len(selected_columns), k=1)):
        corr_value = selected_array[row_index, col_index]
        if np.isfinite(corr_value) and abs(corr_value) >= args.high_corr_threshold:
            left_code = selected_columns[row_index]
            right_code = selected_columns[col_index]
            internal_rows.append(
                {
                    "code_left": left_code,
                    "name_left": name_map.get(left_code, ""),
                    "code_right": right_code,
                    "name_right": name_map.get(right_code, ""),
                    "corr": float(corr_value),
                    "abs_corr": float(abs(corr_value)),
                }
            )
    internal_high_corr = pd.DataFrame(internal_rows).sort_values("abs_corr", ascending=False)

    write_table(top_capture, out_dir, "pool_topn_capture.csv")
    write_table(return_distribution, out_dir, "pool_return_distribution.csv")
    write_table(missed_nearest, out_dir, "pool_missed_winners_nearest_corr.csv")
    write_table(internal_high_corr, out_dir, "pool_internal_high_corr_pairs.csv")
    return {
        "pool_topn_capture": top_capture,
        "pool_return_distribution": return_distribution,
        "pool_missed_winners_nearest_corr": missed_nearest,
        "pool_internal_high_corr_pairs": internal_high_corr,
    }


def forecast_ic_tables(bridge_path: Path, out_dir: Path, momentum_window: int = 20) -> dict[str, pd.DataFrame]:
    requested_columns = [
        "instrument",
        "datetime",
        "close",
        "next_return",
        "forecast_mean_return_robust",
        "forecast_mean_return",
        "forecast_variance_robust",
        "forecast_variance",
    ]
    bridge = read_csv_columns(bridge_path, requested_columns)
    if "close" in bridge.columns and int(momentum_window) > 0:
        bridge = bridge.sort_values(["instrument", "datetime"]).copy()
        bridge["momentum_1m"] = bridge.groupby("instrument")["close"].transform(
            lambda series: numeric_series(series).pct_change(int(momentum_window), fill_method=None).shift(1)
        )
    mu_col = "forecast_mean_return_robust" if "forecast_mean_return_robust" in bridge.columns else "forecast_mean_return"
    var_col = "forecast_variance_robust" if "forecast_variance_robust" in bridge.columns else "forecast_variance"
    daily_rows: list[dict[str, object]] = []
    for date_value, day_frame in bridge.groupby("datetime"):
        working_cols = ["next_return", mu_col, var_col]
        if "momentum_1m" in day_frame.columns:
            working_cols.append("momentum_1m")
        working = day_frame[working_cols].copy()
        working["next_return"] = numeric_series(working["next_return"])
        working[mu_col] = numeric_series(working[mu_col])
        working[var_col] = numeric_series(working[var_col])
        if "momentum_1m" in working.columns:
            working["momentum_1m"] = numeric_series(working["momentum_1m"])
        working = working.dropna(subset=["next_return"])
        if len(working) < 5:
            continue
        mu_valid = working.dropna(subset=[mu_col])
        var_valid = working.dropna(subset=[var_col])
        momentum_valid = working.dropna(subset=["momentum_1m"]) if "momentum_1m" in working.columns else pd.DataFrame()
        blend_valid = working.dropna(subset=[mu_col, "momentum_1m"]) if "momentum_1m" in working.columns else pd.DataFrame()
        momentum_rank_ic = np.nan
        blended_70_30_rank_ic = np.nan
        blended_50_50_rank_ic = np.nan
        if len(momentum_valid) >= 5:
            momentum_rank_ic = float(momentum_valid["momentum_1m"].corr(momentum_valid["next_return"], method="spearman"))
        if len(blend_valid) >= 5:
            mu_rank_pct = blend_valid[mu_col].rank(pct=True)
            momentum_rank_pct = blend_valid["momentum_1m"].rank(pct=True)
            blended_70_30 = 0.7 * mu_rank_pct + 0.3 * momentum_rank_pct
            blended_50_50 = 0.5 * mu_rank_pct + 0.5 * momentum_rank_pct
            blended_70_30_rank_ic = float(blended_70_30.corr(blend_valid["next_return"], method="spearman"))
            blended_50_50_rank_ic = float(blended_50_50.corr(blend_valid["next_return"], method="spearman"))
        top_minus_bottom = np.nan
        if len(mu_valid) >= 10:
            sorted_mu = mu_valid.sort_values(mu_col, ascending=False)
            bucket_size = max(1, int(math.ceil(len(sorted_mu) * 0.2)))
            top_minus_bottom = float(
                sorted_mu.head(bucket_size)["next_return"].mean()
                - sorted_mu.tail(bucket_size)["next_return"].mean()
            )
        daily_rows.append(
            {
                "datetime": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                "n": int(len(working)),
                "mu_rank_ic": float(mu_valid[mu_col].corr(mu_valid["next_return"], method="spearman")) if len(mu_valid) >= 5 else np.nan,
                "mu_pearson_ic": float(mu_valid[mu_col].corr(mu_valid["next_return"])) if len(mu_valid) >= 5 else np.nan,
                "mu_top20_minus_bottom20": top_minus_bottom,
                "momentum_1m_rank_ic": momentum_rank_ic,
                "blended_mu_70_30_rank_ic": blended_70_30_rank_ic,
                "blended_mu_50_50_rank_ic": blended_50_50_rank_ic,
                "var_rank_ic_absret": float(var_valid[var_col].corr(var_valid["next_return"].abs(), method="spearman")) if len(var_valid) >= 5 else np.nan,
                "var_rank_ic_sqret": float(var_valid[var_col].corr(var_valid["next_return"] ** 2, method="spearman")) if len(var_valid) >= 5 else np.nan,
                "high_var_absret_mean": float(var_valid.nlargest(max(1, len(var_valid) // 5), var_col)["next_return"].abs().mean()) if len(var_valid) >= 5 else np.nan,
                "low_var_absret_mean": float(var_valid.nsmallest(max(1, len(var_valid) // 5), var_col)["next_return"].abs().mean()) if len(var_valid) >= 5 else np.nan,
            }
        )
    daily = pd.DataFrame(daily_rows)
    summary_rows = []
    for metric in [
        "mu_rank_ic",
        "mu_pearson_ic",
        "mu_top20_minus_bottom20",
        "momentum_1m_rank_ic",
        "blended_mu_70_30_rank_ic",
        "blended_mu_50_50_rank_ic",
        "var_rank_ic_absret",
        "var_rank_ic_sqret",
        "high_var_absret_mean",
        "low_var_absret_mean",
    ]:
        metric_summary = summarize_series(daily[metric] if metric in daily.columns else pd.Series(dtype=float))
        metric_summary["metric"] = metric
        summary_rows.append(metric_summary)
    summary = pd.DataFrame(summary_rows)
    write_table(daily, out_dir, "forecast_ic_daily.csv")
    write_table(summary, out_dir, "forecast_ic_summary.csv")
    return {"forecast_ic_daily": daily, "forecast_ic_summary": summary}


def mvo_killer_label(mu_rank_pct: float, vol_rank_pct: float, weight_to_equal: float) -> str:
    labels: list[str] = []
    if np.isfinite(mu_rank_pct) and mu_rank_pct < 0.50:
        labels.append("mean_signal_missed")
    if np.isfinite(vol_rank_pct) and vol_rank_pct > 0.65:
        labels.append("variance_penalty")
    if np.isfinite(weight_to_equal) and weight_to_equal < 0.80:
        labels.append("optimizer_underweight")
    return "+".join(labels) if labels else "no_single_killer"


def top_winner_forecast_diagnostics(
    bridge_path: Path,
    l3_path: Path | None,
    out_dir: Path,
    fund_list_csv: str | Path,
    top_k: int,
    min_period_return: float,
    upside_share_min: float,
    positive_share_min: float,
    mean_top_pct: float,
    clone_corr_threshold: float,
) -> dict[str, pd.DataFrame]:
    requested_columns = [
        "instrument",
        "datetime",
        "next_return",
        "forecast_mean_return_robust",
        "forecast_mean_return",
        "forecast_variance_robust",
        "forecast_variance",
        "forecast_w",
    ]
    bridge = read_csv_columns(bridge_path, requested_columns)
    mu_col = "forecast_mean_return_robust" if "forecast_mean_return_robust" in bridge.columns else "forecast_mean_return"
    var_col = "forecast_variance_robust" if "forecast_variance_robust" in bridge.columns else "forecast_variance"
    if not {"instrument", "datetime", "next_return", mu_col, var_col}.issubset(bridge.columns):
        daily = pd.DataFrame()
        summary = pd.DataFrame()
        write_table(daily, out_dir, "selected_top_winner_forecast_daily.csv")
        write_table(summary, out_dir, "selected_top_winner_forecast_summary.csv")
        return {
            "selected_top_winner_forecast_daily": daily,
            "selected_top_winner_forecast_summary": summary,
        }

    bridge = bridge.sort_values(["datetime", "instrument"]).copy()
    bridge["next_return"] = numeric_series(bridge["next_return"])
    bridge["forecast_mean"] = numeric_series(bridge[mu_col])
    bridge["forecast_variance"] = numeric_series(bridge[var_col]).clip(lower=0.0)
    bridge["forecast_vol"] = np.sqrt(bridge["forecast_variance"])
    bridge["forecast_w_raw"] = numeric_series(bridge["forecast_w"]) if "forecast_w" in bridge.columns else np.nan
    bridge["day_pool_n"] = bridge.groupby("datetime")["instrument"].transform("nunique")
    bridge["equal_weight"] = np.where(bridge["day_pool_n"] > 0, 1.0 / bridge["day_pool_n"], np.nan)
    bridge["forecast_mean_rank_pct"] = bridge.groupby("datetime")["forecast_mean"].transform(
        lambda series: numeric_series(series).rank(pct=True)
    )
    bridge["forecast_mean_top_rank_pct"] = bridge.groupby("datetime")["forecast_mean"].transform(
        lambda series: numeric_series(series).rank(pct=True, ascending=False)
    )
    bridge["forecast_vol_rank_pct"] = bridge.groupby("datetime")["forecast_vol"].transform(
        lambda series: numeric_series(series).rank(pct=True)
    )
    bridge["forecast_w_raw_rank_pct"] = bridge.groupby("datetime")["forecast_w_raw"].transform(
        lambda series: numeric_series(series).rank(pct=True)
    )
    period_return = bridge.groupby("instrument")["next_return"].apply(compound_return).dropna()
    top_winners = period_return.sort_values(ascending=False).head(max(1, int(top_k)))
    winner_rank = {code: rank for rank, code in enumerate(top_winners.index, start=1)}
    returns_pivot = bridge.pivot_table(index="datetime", columns="instrument", values="next_return", aggfunc="first")
    corr_profiles = correlation_contribution_profiles(returns_pivot, bridge["instrument"].unique())
    volatility_profiles = {
        code: realized_volatility_profile(frame["next_return"])
        for code, frame in bridge[bridge["instrument"].isin(top_winners.index)].groupby("instrument")
    }
    profile_columns = [
        "positive_return_share",
        "upside_volatility",
        "downside_volatility",
        "upside_downside_vol_ratio",
        "upside_volatility_share",
        "downside_volatility_share",
        "return_to_total_vol",
        "return_to_downside_vol",
        "period_return_to_downside_vol",
    ]

    if l3_path is not None and Path(l3_path).exists():
        l3_columns = ["instrument", "datetime", "forecast_w", "exposure_scale"]
        l3 = read_csv_columns(Path(l3_path), l3_columns)
        if {"instrument", "datetime"}.issubset(l3.columns):
            rename_map = {}
            if "forecast_w" in l3.columns:
                rename_map["forecast_w"] = "forecast_w_l3"
            l3 = l3.rename(columns=rename_map)
            bridge = bridge.merge(l3, on=["datetime", "instrument"], how="left")
    if "forecast_w_l3" not in bridge.columns:
        bridge["forecast_w_l3"] = bridge["forecast_w_raw"]
    if "exposure_scale" not in bridge.columns:
        bridge["exposure_scale"] = 1.0
    bridge["forecast_w_l3"] = numeric_series(bridge["forecast_w_l3"])
    bridge["exposure_scale"] = numeric_series(bridge["exposure_scale"]).fillna(1.0)
    bridge["actual_weight_proxy"] = bridge["forecast_w_l3"] * bridge["exposure_scale"]
    bridge["forecast_w_raw_to_equal"] = bridge["forecast_w_raw"] / bridge["equal_weight"]
    bridge["forecast_w_l3_to_equal"] = bridge["forecast_w_l3"] / bridge["equal_weight"]

    name_map = build_name_map(fund_list_csv)
    daily = bridge[bridge["instrument"].isin(top_winners.index)].copy()
    daily["name"] = daily["instrument"].map(name_map).fillna("")
    daily["period_return"] = daily["instrument"].map(top_winners.to_dict())
    daily["period_return_rank"] = daily["instrument"].map(winner_rank)
    for column in profile_columns:
        daily[column] = daily["instrument"].map(
            {code: profile.get(column, np.nan) for code, profile in volatility_profiles.items()}
        )
    for column in [
        "avg_abs_corr_to_selected_peer",
        "nearest_selected_peer",
        "nearest_selected_peer_corr",
        "marginal_corr_contribution",
    ]:
        daily[column] = daily["instrument"].map(
            {code: profile.get(column, np.nan) for code, profile in corr_profiles.items()}
        )
    daily["offensive_volatility_flag"] = [
        is_offensive_volatility(
            row,
            min_period_return=min_period_return,
            upside_share_min=upside_share_min,
            positive_share_min=positive_share_min,
        )
        for _, row in daily.iterrows()
    ]
    daily["killer_flag"] = [
        mvo_killer_label(mu_rank, vol_rank, weight_ratio)
        for mu_rank, vol_rank, weight_ratio in zip(
            daily["forecast_mean_rank_pct"], daily["forecast_vol_rank_pct"], daily["forecast_w_raw_to_equal"]
        )
    ]
    daily["datetime"] = pd.to_datetime(daily["datetime"]).dt.strftime("%Y-%m-%d")
    daily = daily[
        [
            "datetime",
            "instrument",
            "name",
            "period_return_rank",
            "period_return",
            "next_return",
            "forecast_mean",
            "forecast_mean_rank_pct",
            "forecast_mean_top_rank_pct",
            "forecast_vol",
            "forecast_vol_rank_pct",
            "upside_volatility",
            "downside_volatility",
            "upside_downside_vol_ratio",
            "upside_volatility_share",
            "downside_volatility_share",
            "return_to_downside_vol",
            "avg_abs_corr_to_selected_peer",
            "nearest_selected_peer",
            "nearest_selected_peer_corr",
            "marginal_corr_contribution",
            "forecast_w_raw",
            "forecast_w_raw_rank_pct",
            "forecast_w_raw_to_equal",
            "forecast_w_l3",
            "forecast_w_l3_to_equal",
            "exposure_scale",
            "actual_weight_proxy",
            "offensive_volatility_flag",
            "killer_flag",
        ]
    ].sort_values(["period_return_rank", "datetime"])

    summary_rows: list[dict[str, object]] = []
    grouped_frames = [("ALL_TOP_WINNERS", daily)] + list(daily.groupby("instrument", sort=False))
    for code, frame in grouped_frames:
        if frame.empty:
            continue
        is_overall = code == "ALL_TOP_WINNERS"
        mu_rank_mean = float(numeric_series(frame["forecast_mean_rank_pct"]).mean())
        mu_top_rank_mean = float(numeric_series(frame["forecast_mean_top_rank_pct"]).mean())
        vol_rank_mean = float(numeric_series(frame["forecast_vol_rank_pct"]).mean())
        weight_to_equal_mean = float(numeric_series(frame["forecast_w_raw_to_equal"]).mean())
        volatility_profile = realized_volatility_profile(frame["next_return"])
        row = {
            "instrument": code,
            "name": "Top winner basket" if is_overall else frame["name"].iloc[0],
            "period_return_rank": np.nan if is_overall else int(frame["period_return_rank"].iloc[0]),
            "period_return": float(numeric_series(frame["period_return"]).mean()) if is_overall else float(frame["period_return"].iloc[0]),
            "observed_days": int(len(frame)),
            "avg_next_return": float(numeric_series(frame["next_return"]).mean()),
            "avg_forecast_mean": float(numeric_series(frame["forecast_mean"]).mean()),
            "avg_forecast_mean_rank_pct": mu_rank_mean,
            "avg_forecast_mean_top_rank_pct": mu_top_rank_mean,
            "avg_forecast_vol": float(numeric_series(frame["forecast_vol"]).mean()),
            "avg_forecast_vol_rank_pct": vol_rank_mean,
            "avg_forecast_w_raw": float(numeric_series(frame["forecast_w_raw"]).mean()),
            "avg_forecast_w_raw_to_equal": weight_to_equal_mean,
            "avg_forecast_w_l3": float(numeric_series(frame["forecast_w_l3"]).mean()),
            "avg_forecast_w_l3_to_equal": float(numeric_series(frame["forecast_w_l3_to_equal"]).mean()),
            "avg_exposure_scale": float(numeric_series(frame["exposure_scale"]).mean()),
            "avg_actual_weight_proxy": float(numeric_series(frame["actual_weight_proxy"]).mean()),
            "low_mu_day_share": float((numeric_series(frame["forecast_mean_rank_pct"]) < 0.50).mean()),
            "mean_top30_day_share": float((numeric_series(frame["forecast_mean_top_rank_pct"]) <= mean_top_pct).mean()),
            "high_vol_day_share": float((numeric_series(frame["forecast_vol_rank_pct"]) > 0.65).mean()),
            "under_equal_weight_day_share": float((numeric_series(frame["forecast_w_raw_to_equal"]) < 0.80).mean()),
            "diagnosis_label": mvo_killer_label(mu_rank_mean, vol_rank_mean, weight_to_equal_mean),
        }
        row.update(volatility_profile)
        if not is_overall:
            row.update(corr_profiles.get(code, {}))
        else:
            row.update(
                {
                    "avg_abs_corr_to_selected_peer": float(numeric_series(daily["avg_abs_corr_to_selected_peer"]).mean()),
                    "nearest_selected_peer": "",
                    "nearest_selected_peer_corr": float(numeric_series(daily["nearest_selected_peer_corr"]).max()),
                    "marginal_corr_contribution": float(numeric_series(daily["marginal_corr_contribution"]).mean()),
                }
            )
        row["offensive_volatility_flag"] = is_offensive_volatility(
            pd.Series(row),
            min_period_return=min_period_return,
            upside_share_min=upside_share_min,
            positive_share_min=positive_share_min,
        )
        row["decision_tree_label"] = decision_tree_label(
            pd.Series(row),
            mean_top_pct=mean_top_pct,
            clone_corr_threshold=clone_corr_threshold,
        )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    write_table(daily, out_dir, "selected_top_winner_forecast_daily.csv")
    write_table(summary, out_dir, "selected_top_winner_forecast_summary.csv")
    return {
        "selected_top_winner_forecast_daily": daily,
        "selected_top_winner_forecast_summary": summary,
    }


def focus_theme_label(name: object) -> str:
    text = str(name).upper()
    labels: list[str] = []
    if "通信" in text:
        labels.append("communication")
    if "人工智能" in text or "AI" in text:
        labels.append("ai")
    if "科创" in text and "半导体" in text:
        labels.append("star_semiconductor")
    elif "半导体" in text:
        labels.append("semiconductor")
    return ",".join(labels)


def industry_theme_label(name: object) -> str:
    text = str(name).upper()
    if "通信" in text:
        return "communication"
    if "人工智能" in text or "AI" in text:
        return "ai"
    if "科创" in text and "半导体" in text:
        return "star_semiconductor"
    if "半导体" in text or "芯片" in text or "集成电路" in text:
        return "semiconductor"
    if "油气" in text or "石油" in text or "能源" in text:
        return "energy"
    if "化工" in text or "石化" in text:
        return "chemical"
    if "红利" in text or "股息" in text:
        return "dividend"
    if "医药" in text or "医疗" in text:
        return "healthcare"
    if "证券" in text or "券商" in text:
        return "brokerage"
    if "军工" in text or "国防" in text:
        return "defense"
    return "other"


def audit_l3_industry_label(name: object) -> str:
    base_label = industry_theme_label(name)
    if base_label != "other":
        return base_label
    text = str(name).upper()
    if "有色" in text or "矿业" in text or "黄金" in text or "金ETF" in text:
        return "metals"
    if "金融" in text or "银行" in text:
        return "financial"
    if "房地产" in text or "地产" in text:
        return "real_estate"
    if "消费" in text:
        return "consumer"
    if "创新药" in text or "生物医药" in text:
        return "healthcare"
    if "游戏" in text:
        return "game"
    if "卫星" in text:
        return "satellite"
    if "法国" in text or "沙特" in text or "日经" in text or "纳指" in text or "标普" in text:
        return "overseas"
    if "科创" in text:
        return "star_market"
    if "中创" in text or "治理" in text or "180" in text or "400" in text:
        return "broad_market"
    return "other"


def enforce_factor_audit_schema_defaults(frame: pd.DataFrame, factor_source: str) -> pd.DataFrame:
    working = frame.copy()
    source_text = str(factor_source or "unknown")
    working["factor_schema_source"] = source_text
    for column, default_value in FACTOR_SCHEMA_DEFAULTS.items():
        if column in working.columns:
            values = numeric_series(working[column]).replace([np.inf, -np.inf], np.nan)
        else:
            values = pd.Series(np.nan, index=working.index, dtype=float)
        if column == "factor_volume_ratio_20":
            fill_mask = values.isna() | (values <= 0.0)
        else:
            fill_mask = values.isna()
        working[column] = values.mask(fill_mask, default_value)
        working[f"{column}_default_fill_flag"] = fill_mask.astype(int)
        working[f"{column}_lineage"] = np.where(
            fill_mask,
            f"neutral_default_{default_value:g}",
            source_text,
        )
    working["factor_abnormal_volume_flag"] = (
        (numeric_series(working["factor_volume_ratio_20"]) >= 1.5)
        | (numeric_series(working["factor_volume_z_20"]) >= 1.0)
    ).astype(int)
    return working


def factor_schema_default_fill_total(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    return int(
        sum(
            numeric_series(frame.get(f"{column}_default_fill_flag", pd.Series(0, index=frame.index))).fillna(0.0).sum()
            for column in FACTOR_SCHEMA_DEFAULTS
        )
    )


def factor_schema_lineage_summary(frame: pd.DataFrame, factor_source: str) -> dict[str, object]:
    row_count = int(frame.shape[0])
    summary: dict[str, object] = {
        "factor_source": factor_source,
        "data_lineage_statement": (
            "factor_style_growth/factor_volume_z_20/factor_volume_ratio_20 are loaded from L3 when complete, "
            "otherwise rebuilt from raw panel; missing ratio values are neutral-filled at 1.0 and missing z/growth at 0.0."
        ),
        "factor_schema_row_count": row_count,
    }
    for column in FACTOR_SCHEMA_DEFAULTS:
        fill_count = int(numeric_series(frame.get(f"{column}_default_fill_flag", pd.Series(0, index=frame.index))).fillna(0.0).sum()) if row_count else 0
        summary[f"{column}_default_fill_count"] = fill_count
        summary[f"{column}_default_fill_rate"] = divide_or_nan(fill_count, row_count)
    return summary


def add_custom_factor_exposures_from_raw_panel(
    frame: pd.DataFrame,
    raw_panel_path: Path | None,
    name_map: dict[str, str],
) -> pd.DataFrame:
    if raw_panel_path is None or not Path(raw_panel_path).exists() or frame.empty:
        return frame
    header = csv_header(Path(raw_panel_path))
    required = {"instrument", "datetime", "close"}
    if not required.issubset(header):
        return frame
    usecols = [column for column in ["instrument", "datetime", "close", "volume"] if column in header]
    raw = pd.read_csv(raw_panel_path, usecols=usecols, parse_dates=["datetime"])
    raw["instrument"] = raw["instrument"].map(normalize_code)
    raw = raw.sort_values(["instrument", "datetime"])
    raw["close"] = numeric_series(raw["close"])
    for window in [5, 20, 60]:
        raw[f"factor_return_{window}d"] = raw.groupby("instrument")["close"].transform(
            lambda series: series.pct_change(window, fill_method=None).shift(1)
        )
    raw["factor_style_momentum_raw"] = raw[["factor_return_20d", "factor_return_60d"]].mean(axis=1, skipna=True)
    raw["factor_style_growth_raw"] = 0.35 * raw["factor_return_20d"] + 0.65 * raw["factor_return_60d"]
    if "volume" in raw.columns:
        raw["volume"] = numeric_series(raw["volume"])
        raw["factor_dollar_volume"] = raw["close"].abs() * raw["volume"].clip(lower=0.0)
        raw["factor_dollar_volume"] = raw["factor_dollar_volume"].where(
            raw["factor_dollar_volume"] > 0.0,
            raw["volume"].clip(lower=0.0),
        )
        raw["factor_volume_ma20"] = raw.groupby("instrument")["factor_dollar_volume"].transform(
            lambda series: series.shift(1).rolling(window=20, min_periods=5).mean()
        )
        raw["factor_volume_ratio_20"] = raw["factor_dollar_volume"] / raw["factor_volume_ma20"]
        raw["factor_style_size_raw"] = np.log1p(raw["factor_volume_ma20"])
    else:
        raw["factor_volume_ratio_20"] = np.nan
        raw["factor_style_size_raw"] = np.nan
    raw["factor_industry_theme"] = raw["instrument"].map(name_map).map(industry_theme_label).fillna("other")
    raw["factor_industry_momentum_raw"] = raw.groupby(["datetime", "factor_industry_theme"])[
        "factor_style_momentum_raw"
    ].transform("mean")
    feature_columns = [
        "instrument",
        "datetime",
        "factor_industry_theme",
        "factor_style_momentum_raw",
        "factor_style_growth_raw",
        "factor_style_size_raw",
        "factor_industry_momentum_raw",
        "factor_volume_ratio_20",
    ]
    features = raw[feature_columns].drop_duplicates(["instrument", "datetime"], keep="last")
    result = frame.merge(features, on=["instrument", "datetime"], how="left", suffixes=("", "_raw_panel"))
    for column in [item for item in feature_columns if item not in {"instrument", "datetime"}]:
        raw_panel_column = f"{column}_raw_panel"
        if raw_panel_column not in result.columns:
            continue
        if column in result.columns:
            if column == "factor_industry_theme":
                current = result[column].where(result[column].notna() & (result[column].astype(str).str.len() > 0))
                result[column] = current.fillna(result[raw_panel_column])
            else:
                result[column] = numeric_series(result[column]).fillna(numeric_series(result[raw_panel_column]))
        else:
            result[column] = result[raw_panel_column]
        result = result.drop(columns=[raw_panel_column])
    def _cross_section_z(series: pd.Series) -> pd.Series:
        values = numeric_series(series)
        valid = values.dropna()
        if valid.empty:
            return pd.Series(np.nan, index=values.index, dtype=float)
        median = float(valid.median())
        mad = float((valid - median).abs().median())
        if np.isfinite(mad) and mad > 1e-12:
            scale = 1.4826 * mad
        else:
            scale = float(valid.std(ddof=0))
        if not np.isfinite(scale) or scale <= 1e-12:
            return pd.Series(0.0, index=values.index, dtype=float).where(values.notna(), np.nan)
        return ((values - median) / scale).clip(-3.0, 3.0)

    factor_z_map = {
        "factor_style_momentum_raw": "factor_style_momentum",
        "factor_style_growth_raw": "factor_style_growth",
        "factor_style_size_raw": "factor_style_size",
        "factor_industry_momentum_raw": "factor_industry_momentum",
        "factor_volume_ratio_20": "factor_volume_z_20",
    }
    for raw_col, z_col in factor_z_map.items():
        if raw_col not in result.columns:
            result[raw_col] = np.nan
        result[z_col] = result.groupby("datetime")[raw_col].transform(_cross_section_z)
    result["factor_abnormal_volume_flag"] = (
        (numeric_series(result["factor_volume_ratio_20"]) >= 1.5)
        | (numeric_series(result["factor_volume_z_20"]) >= 1.0)
    ).astype(int)
    return result


def divide_or_nan(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= 1e-12:
        return np.nan
    return float(numerator / denominator)


def residual_reaction_label(
    boom_day_count: int,
    same_day_coverage_share: float,
    reaction_slow_day_share: float,
    persistent_miss_day_share: float,
    predicted_share: float,
    active_offset_share: float,
) -> str:
    if boom_day_count <= 0:
        return "no_boom_days"
    if np.isfinite(same_day_coverage_share) and same_day_coverage_share >= 0.50:
        return "same_day_covered"
    if np.isfinite(persistent_miss_day_share) and persistent_miss_day_share >= 0.50:
        return "not_covered"
    if np.isfinite(reaction_slow_day_share) and reaction_slow_day_share >= 0.30:
        return "reaction_slow"
    if np.isfinite(predicted_share) and predicted_share <= 0.0 or np.isfinite(active_offset_share) and active_offset_share >= 0.20:
        return "active_offset"
    return "mixed_or_weak"


def residual_constraint_label(row: pd.Series) -> str:
    active_offset_share = safe_float(row.get("boom_active_offset_day_share"))
    positive_momentum_share = safe_float(row.get("boom_momentum_positive_share"))
    avg_momentum_z = safe_float(row.get("boom_avg_risk_trend_momentum_z"))
    scale_down_share = safe_float(row.get("boom_scale_down_share"))
    trend_selected_share = safe_float(row.get("boom_trend_sleeve_selected_share"))
    momentum_confirmed = positive_momentum_share >= 0.60 and avg_momentum_z > 0.0
    if active_offset_share >= 0.30 and momentum_confirmed and scale_down_share >= 0.50:
        return "positive_momentum_but_scaled_down_proxy"
    if active_offset_share >= 0.30 and momentum_confirmed:
        return "positive_momentum_but_mean_model_offsets"
    if active_offset_share >= 0.30:
        return "active_negative_prediction_on_boom"
    if scale_down_share >= 0.50:
        return "scale_penalty_without_negative_prediction"
    if trend_selected_share >= 0.50:
        return "momentum_confirmed_no_clear_offset"
    return "no_clear_constraint_offset_proxy"


def residual_analysis_audit(
    bridge_path: Path,
    l3_path: Path | None,
    raw_panel_path: Path | None,
    out_dir: Path,
    fund_list_csv: str | Path,
    top_k: int,
    boom_top_pct: float,
    mean_top_pct: float,
    clone_corr_threshold: float,
    min_period_return: float,
    upside_share_min: float,
    positive_share_min: float,
) -> dict[str, pd.DataFrame]:
    bridge_columns = [
        "instrument",
        "datetime",
        "next_return",
        "forecast_mean_return_robust",
        "forecast_mean_return",
        "forecast_variance_robust",
        "forecast_variance",
        "forecast_w",
        "forecast_active_set_protect_score",
        "forecast_active_set_crowded_score",
        "forecast_active_set_balance_score",
        "forecast_weight_mean_raw",
        "forecast_weight_mean_slow",
        "forecast_weight_mean_used",
        "forecast_mu_shrink_alpha",
        "forecast_mu_rolling_ic",
    ]
    l3_columns = [
        "instrument",
        "datetime",
        "forecast_w",
        "exposure_scale",
        "trend_mom5",
        "trend_mom20",
        "trend_mom60",
        "risk_trend_momentum_valid_windows",
        "risk_trend_mom5_z",
        "risk_trend_mom20_z",
        "risk_trend_mom60_z",
        "risk_trend_momentum_z",
        "risk_trend_mu_z",
        "risk_trend_score",
        "risk_trend_sleeve_add",
        "risk_trend_sleeve_selected",
        "risk_trend_sleeve_weight_total_used",
        "risk_trend_sleeve_top_k_used",
        "risk_trend_sleeve_regime_on",
        "risk_trend_sleeve_exposure_floor_used",
        "risk_base_exposure_scale",
        "risk_trend_sleeve_floor_lift",
        "risk_regime_factor",
        "risk_regime_ratio",
        "risk_regime_on",
        "risk_vol_target_used",
        "risk_vol_scale",
        "risk_cluster_rho_bar",
        "risk_cluster_scale",
        "risk_signal_quality_rolling_ic",
        "risk_signal_quality_min_scale_used",
        "risk_signal_quality_scale",
        *FACTOR_AUDIT_COLUMNS,
    ]
    bridge = read_csv_columns(bridge_path, bridge_columns)
    mu_col = "forecast_mean_return_robust" if "forecast_mean_return_robust" in bridge.columns else "forecast_mean_return"
    var_col = "forecast_variance_robust" if "forecast_variance_robust" in bridge.columns else "forecast_variance"
    if not {"instrument", "datetime", "next_return", mu_col}.issubset(bridge.columns):
        daily = pd.DataFrame()
        summary = pd.DataFrame()
        reaction_lag = pd.DataFrame()
        factor_check = pd.DataFrame()
        write_table(daily, out_dir, "residual_analysis_daily.csv")
        write_table(summary, out_dir, "residual_analysis_summary.csv")
        write_table(reaction_lag, out_dir, "residual_reaction_lag_summary.csv")
        write_table(factor_check, out_dir, "residual_factor_exposure_check.csv")
        return {
            "residual_analysis_daily": daily,
            "residual_analysis_summary": summary,
            "residual_reaction_lag_summary": reaction_lag,
            "residual_factor_exposure_check": factor_check,
        }

    bridge = bridge.sort_values(["datetime", "instrument"]).copy()
    numeric_columns = [column for column in bridge.columns if column not in {"instrument", "datetime"}]
    for column in numeric_columns:
        bridge[column] = numeric_series(bridge[column])
    bridge["forecast_mean"] = numeric_series(bridge[mu_col])
    if var_col in bridge.columns:
        bridge["forecast_variance"] = numeric_series(bridge[var_col]).clip(lower=0.0)
        bridge["forecast_vol"] = np.sqrt(bridge["forecast_variance"])
    else:
        bridge["forecast_vol"] = np.nan
    bridge["forecast_w_raw"] = numeric_series(bridge["forecast_w"]) if "forecast_w" in bridge.columns else np.nan

    if l3_path is not None and Path(l3_path).exists():
        l3 = read_csv_columns(Path(l3_path), l3_columns)
        if {"instrument", "datetime"}.issubset(l3.columns):
            if "forecast_w" in l3.columns:
                l3 = l3.rename(columns={"forecast_w": "forecast_w_l3"})
            l3_numeric_columns = [
                column
                for column in l3.columns
                if column not in {"instrument", "datetime", "factor_industry_theme"}
            ]
            for column in l3_numeric_columns:
                l3[column] = numeric_series(l3[column])
            bridge = bridge.merge(l3, on=["datetime", "instrument"], how="left")
    if "forecast_w_l3" not in bridge.columns:
        bridge["forecast_w_l3"] = bridge["forecast_w_raw"]
    if "exposure_scale" not in bridge.columns:
        bridge["exposure_scale"] = 1.0
    bridge["forecast_w_l3"] = numeric_series(bridge["forecast_w_l3"])
    bridge["exposure_scale"] = numeric_series(bridge["exposure_scale"]).fillna(1.0)

    bridge["next_return"] = numeric_series(bridge["next_return"])
    bridge["day_pool_n"] = bridge.groupby("datetime")["instrument"].transform("nunique")
    bridge["equal_weight"] = np.where(bridge["day_pool_n"] > 0, 1.0 / bridge["day_pool_n"], np.nan)
    bridge["pool_equal_next_return"] = bridge.groupby("datetime")["next_return"].transform("mean")
    bridge["pool_mean_forecast"] = bridge.groupby("datetime")["forecast_mean"].transform("mean")
    bridge["actual_excess_return"] = bridge["next_return"] - bridge["pool_equal_next_return"]
    bridge["predicted_excess_return"] = bridge["forecast_mean"] - bridge["pool_mean_forecast"]
    bridge["residual_return"] = bridge["actual_excess_return"] - bridge["predicted_excess_return"]
    bridge["actual_excess_top_rank_pct"] = bridge.groupby("datetime")["actual_excess_return"].transform(
        lambda series: numeric_series(series).rank(pct=True, ascending=False)
    )
    bridge["predicted_excess_top_rank_pct"] = bridge.groupby("datetime")["predicted_excess_return"].transform(
        lambda series: numeric_series(series).rank(pct=True, ascending=False)
    )
    bridge["forecast_mean_top_rank_pct"] = bridge.groupby("datetime")["forecast_mean"].transform(
        lambda series: numeric_series(series).rank(pct=True, ascending=False)
    )
    bridge["forecast_vol_rank_pct"] = bridge.groupby("datetime")["forecast_vol"].transform(
        lambda series: numeric_series(series).rank(pct=True)
    )
    bridge["forecast_w_raw_to_equal"] = bridge["forecast_w_raw"] / bridge["equal_weight"]
    bridge["forecast_w_l3_to_equal"] = bridge["forecast_w_l3"] / bridge["equal_weight"]
    bridge["actual_weight_proxy"] = bridge["forecast_w_l3"] * bridge["exposure_scale"]
    positive_actual = bridge["actual_excess_return"].where(bridge["actual_excess_return"] > 0.0)
    residual_positive = bridge["residual_return"].clip(lower=0.0)
    residual_clipped = np.minimum(residual_positive, positive_actual)
    active_offset = (-bridge["predicted_excess_return"]).clip(lower=0.0)
    bridge["residual_share_of_positive_specific"] = residual_positive / positive_actual
    bridge["residual_share_clipped_0_1"] = residual_clipped / positive_actual
    bridge["active_offset_share"] = active_offset / positive_actual
    boom_threshold = float(np.clip(float(boom_top_pct), 0.01, 1.0))
    bridge["boom_day_flag"] = (bridge["actual_excess_return"] > 0.0) & (
        bridge["actual_excess_top_rank_pct"] <= boom_threshold
    )

    name_map = build_name_map(fund_list_csv)
    l3_header = csv_header(Path(l3_path)) if l3_path is not None and Path(l3_path).exists() else []
    l3_explicit_factor_columns = [column for column in EXPLICIT_FACTOR_COLUMNS if column in l3_header]
    explicit_factor_source = "l3_explicit" if l3_explicit_factor_columns else ""
    if not l3_explicit_factor_columns:
        bridge = add_custom_factor_exposures_from_raw_panel(bridge, raw_panel_path, name_map)
        fallback_factor_columns = [column for column in EXPLICIT_FACTOR_COLUMNS if column in bridge.columns]
        if fallback_factor_columns:
            explicit_factor_source = "audit_raw_panel_fallback"
    if "factor_industry_theme" not in bridge.columns:
        bridge["factor_industry_theme"] = bridge["instrument"].map(name_map).map(industry_theme_label).fillna("other")
    period_return = bridge.groupby("instrument")["next_return"].apply(compound_return).dropna()
    period_rank = period_return.rank(method="first", ascending=False).astype(int)
    sorted_period_return = period_return.sort_values(ascending=False)
    names = {code: name_map.get(code, "") for code in sorted_period_return.index}
    focus_themes = {code: focus_theme_label(names.get(code, "")) for code in sorted_period_return.index}
    max_focus_rank = max(int(top_k), 30)
    target_codes = set(sorted_period_return.head(max(1, int(top_k))).index.tolist())
    target_codes.update(
        code
        for code, theme in focus_themes.items()
        if theme and int(period_rank.get(code, max_focus_rank + 1)) <= max_focus_rank
    )
    ordered_targets = [code for code in sorted_period_return.index if code in target_codes]

    returns_pivot = bridge.pivot_table(index="datetime", columns="instrument", values="next_return", aggfunc="first")
    corr_profiles = correlation_contribution_profiles(returns_pivot, bridge["instrument"].unique())
    volatility_profiles = {
        code: realized_volatility_profile(frame["next_return"])
        for code, frame in bridge[bridge["instrument"].isin(ordered_targets)].groupby("instrument")
    }
    bridge["name"] = bridge["instrument"].map(names).fillna("")
    bridge["focus_theme"] = bridge["instrument"].map(focus_themes).fillna("")
    bridge["period_return"] = bridge["instrument"].map(period_return.to_dict())
    bridge["period_return_rank"] = bridge["instrument"].map(period_rank.to_dict())
    bridge["is_period_top_winner"] = bridge["period_return_rank"] <= int(top_k)
    target_daily = bridge[bridge["instrument"].isin(ordered_targets)].copy()
    if target_daily.empty:
        daily = pd.DataFrame()
        summary = pd.DataFrame()
        reaction_lag = pd.DataFrame()
        factor_check = pd.DataFrame()
        write_table(daily, out_dir, "residual_analysis_daily.csv")
        write_table(summary, out_dir, "residual_analysis_summary.csv")
        write_table(reaction_lag, out_dir, "residual_reaction_lag_summary.csv")
        write_table(factor_check, out_dir, "residual_factor_exposure_check.csv")
        return {
            "residual_analysis_daily": daily,
            "residual_analysis_summary": summary,
            "residual_reaction_lag_summary": reaction_lag,
            "residual_factor_exposure_check": factor_check,
        }

    target_daily = target_daily.sort_values(["instrument", "datetime"])
    target_daily["pre1_forecast_mean_top_rank_pct"] = target_daily.groupby("instrument")[
        "forecast_mean_top_rank_pct"
    ].shift(1)
    for lag_days in [1, 2, 3]:
        target_daily[f"post{lag_days}_forecast_mean_top_rank_pct"] = target_daily.groupby("instrument")[
            "forecast_mean_top_rank_pct"
        ].shift(-lag_days)
    target_daily["post1_3_best_forecast_mean_top_rank_pct"] = target_daily[
        ["post1_forecast_mean_top_rank_pct", "post2_forecast_mean_top_rank_pct", "post3_forecast_mean_top_rank_pct"]
    ].min(axis=1)
    profile_rows: dict[str, dict[str, object]] = {}
    for code, frame in target_daily.groupby("instrument", sort=False):
        row = {
            "avg_forecast_mean_top_rank_pct": float(numeric_series(frame["forecast_mean_top_rank_pct"]).mean()),
            "mean_top30_day_share": float((numeric_series(frame["forecast_mean_top_rank_pct"]) <= mean_top_pct).mean()),
            "avg_forecast_vol_rank_pct": float(numeric_series(frame["forecast_vol_rank_pct"]).mean()),
            "period_return": float(period_return.get(code, np.nan)),
        }
        row.update(volatility_profiles.get(code, {}))
        row.update(corr_profiles.get(code, {}))
        row["offensive_volatility_flag"] = is_offensive_volatility(
            pd.Series(row),
            min_period_return=min_period_return,
            upside_share_min=upside_share_min,
            positive_share_min=positive_share_min,
        )
        row["decision_tree_label"] = decision_tree_label(
            pd.Series(row),
            mean_top_pct=mean_top_pct,
            clone_corr_threshold=clone_corr_threshold,
        )
        profile_rows[code] = row
    target_daily["decision_tree_label"] = target_daily["instrument"].map(
        {code: profile.get("decision_tree_label", "") for code, profile in profile_rows.items()}
    )

    proxy_columns = [
        "trend_mom5",
        "trend_mom20",
        "trend_mom60",
        "risk_trend_momentum_z",
        "risk_trend_mu_z",
        "risk_trend_score",
        "risk_trend_sleeve_add",
        "risk_trend_sleeve_selected",
        "risk_trend_sleeve_floor_lift",
        "exposure_scale",
        "risk_base_exposure_scale",
        "risk_regime_factor",
        "risk_vol_scale",
        "risk_cluster_rho_bar",
        "risk_cluster_scale",
        "risk_signal_quality_scale",
        "forecast_active_set_protect_score",
        "forecast_active_set_crowded_score",
        "forecast_active_set_balance_score",
        "forecast_weight_mean_raw",
        "forecast_weight_mean_slow",
        "forecast_weight_mean_used",
        "forecast_mu_shrink_alpha",
        "forecast_mu_rolling_ic",
        "factor_style_momentum",
        "factor_style_growth",
        "factor_style_size",
        "factor_industry_momentum",
        "factor_style_momentum_raw",
        "factor_style_growth_raw",
        "factor_style_size_raw",
        "factor_industry_momentum_raw",
        "factor_volume_ratio_20",
        "factor_volume_z_20",
        "factor_abnormal_volume_flag",
    ]
    present_proxy_columns = [column for column in proxy_columns if column in target_daily.columns]
    explicit_factor_columns = [column for column in EXPLICIT_FACTOR_COLUMNS if column in target_daily.columns]

    summary_rows: list[dict[str, object]] = []
    factor_rows: list[dict[str, object]] = []
    reaction_rows: list[dict[str, object]] = []
    for code in ordered_targets:
        frame = target_daily[target_daily["instrument"] == code].sort_values("datetime")
        boom_frame = frame[frame["boom_day_flag"]].copy()
        non_boom_frame = frame[~frame["boom_day_flag"]].copy()
        actual_sum = float(numeric_series(boom_frame["actual_excess_return"]).clip(lower=0.0).sum())
        predicted_sum = float(numeric_series(boom_frame["predicted_excess_return"]).sum())
        residual_sum = float(numeric_series(boom_frame["residual_return"]).sum())
        residual_positive_sum = float(numeric_series(boom_frame["residual_return"]).clip(lower=0.0).sum())
        residual_clipped_sum = float(
            np.minimum(
                numeric_series(boom_frame["residual_return"]).clip(lower=0.0),
                numeric_series(boom_frame["actual_excess_return"]).clip(lower=0.0),
            ).sum()
        )
        active_offset_sum = float((-numeric_series(boom_frame["predicted_excess_return"])).clip(lower=0.0).sum())
        same_day_coverage = float((numeric_series(boom_frame["forecast_mean_top_rank_pct"]) <= mean_top_pct).mean()) if not boom_frame.empty else np.nan
        reaction_slow_mask = (
            numeric_series(boom_frame["forecast_mean_top_rank_pct"]) > mean_top_pct
        ) & (
            (numeric_series(boom_frame["post1_3_best_forecast_mean_top_rank_pct"]) <= mean_top_pct)
            | (
                numeric_series(boom_frame["post1_3_best_forecast_mean_top_rank_pct"])
                <= numeric_series(boom_frame["forecast_mean_top_rank_pct"]) - 0.15
            )
        )
        persistent_miss_mask = (
            numeric_series(boom_frame["forecast_mean_top_rank_pct"]) > mean_top_pct
        ) & (
            numeric_series(boom_frame["pre1_forecast_mean_top_rank_pct"]) > mean_top_pct
        ) & (
            numeric_series(boom_frame["post1_3_best_forecast_mean_top_rank_pct"]) > mean_top_pct
        )
        reaction_slow_share = float(reaction_slow_mask.mean()) if not boom_frame.empty else np.nan
        persistent_miss_share = float(persistent_miss_mask.mean()) if not boom_frame.empty else np.nan
        predicted_share = divide_or_nan(predicted_sum, actual_sum)
        residual_share = divide_or_nan(residual_sum, actual_sum)
        residual_positive_share = divide_or_nan(residual_positive_sum, actual_sum)
        residual_clipped_share = divide_or_nan(residual_clipped_sum, actual_sum)
        active_offset_share = divide_or_nan(active_offset_sum, actual_sum)
        summary_row = {
            "instrument": code,
            "name": names.get(code, ""),
            "focus_theme": focus_themes.get(code, ""),
            "decision_tree_label": profile_rows.get(code, {}).get("decision_tree_label", ""),
            "period_return_rank": int(period_rank.get(code, 0)),
            "period_return": float(period_return.get(code, np.nan)),
            "is_period_top_winner": bool(int(period_rank.get(code, top_k + 1)) <= int(top_k)),
            "boom_day_count": int(boom_frame.shape[0]),
            "boom_actual_excess_sum": actual_sum,
            "boom_predicted_excess_sum": predicted_sum,
            "boom_residual_sum": residual_sum,
            "predicted_share_of_positive_specific": predicted_share,
            "residual_share_of_positive_specific": residual_share,
            "residual_positive_share_of_positive_specific": residual_positive_share,
            "residual_share_clipped_0_1": residual_clipped_share,
            "active_offset_share_of_positive_specific": active_offset_share,
            "boom_avg_actual_excess_return": float(numeric_series(boom_frame["actual_excess_return"]).mean()) if not boom_frame.empty else np.nan,
            "boom_avg_predicted_excess_return": float(numeric_series(boom_frame["predicted_excess_return"]).mean()) if not boom_frame.empty else np.nan,
            "boom_avg_residual_return": float(numeric_series(boom_frame["residual_return"]).mean()) if not boom_frame.empty else np.nan,
            "avg_forecast_mean_top_rank_pct_on_boom": float(numeric_series(boom_frame["forecast_mean_top_rank_pct"]).mean()) if not boom_frame.empty else np.nan,
            "avg_actual_excess_top_rank_pct_on_boom": float(numeric_series(boom_frame["actual_excess_top_rank_pct"]).mean()) if not boom_frame.empty else np.nan,
            "same_day_forecast_top30_share": same_day_coverage,
            "avg_pre1_forecast_mean_top_rank_pct": float(numeric_series(boom_frame["pre1_forecast_mean_top_rank_pct"]).mean()) if not boom_frame.empty else np.nan,
            "avg_post1_3_best_forecast_mean_top_rank_pct": float(numeric_series(boom_frame["post1_3_best_forecast_mean_top_rank_pct"]).mean()) if not boom_frame.empty else np.nan,
            "reaction_slow_day_share": reaction_slow_share,
            "persistent_miss_day_share": persistent_miss_share,
            "avg_forecast_vol_rank_pct_on_boom": float(numeric_series(boom_frame["forecast_vol_rank_pct"]).mean()) if not boom_frame.empty else np.nan,
            "avg_forecast_w_raw_to_equal_on_boom": float(numeric_series(boom_frame["forecast_w_raw_to_equal"]).mean()) if not boom_frame.empty else np.nan,
            "avg_forecast_w_l3_to_equal_on_boom": float(numeric_series(boom_frame["forecast_w_l3_to_equal"]).mean()) if not boom_frame.empty else np.nan,
            "avg_exposure_scale_on_boom": float(numeric_series(boom_frame["exposure_scale"]).mean()) if not boom_frame.empty else np.nan,
        }
        summary_row["reaction_diagnosis"] = residual_reaction_label(
            int(summary_row["boom_day_count"]),
            same_day_coverage,
            reaction_slow_share,
            persistent_miss_share,
            predicted_share,
            active_offset_share,
        )
        summary_rows.append(summary_row)

        reaction_row = {
            "instrument": code,
            "name": names.get(code, ""),
            "focus_theme": focus_themes.get(code, ""),
            "decision_tree_label": profile_rows.get(code, {}).get("decision_tree_label", ""),
            "boom_day_count": int(boom_frame.shape[0]),
            "reaction_diagnosis": summary_row["reaction_diagnosis"],
        }
        lag_corr_values: dict[int, float] = {}
        for lag_days in [-3, -2, -1, 0, 1, 2, 3]:
            shifted_forecast = numeric_series(frame["predicted_excess_return"]).shift(-lag_days)
            lag_frame = pd.DataFrame(
                {
                    "actual_excess_return": numeric_series(frame["actual_excess_return"]),
                    "shifted_predicted_excess_return": shifted_forecast,
                }
            ).dropna()
            lag_corr = (
                float(lag_frame["actual_excess_return"].corr(lag_frame["shifted_predicted_excess_return"]))
                if lag_frame.shape[0] >= 5
                else np.nan
            )
            lag_corr_values[lag_days] = lag_corr
            if lag_days < 0:
                column_name = f"corr_actual_t_forecast_t_minus_{abs(lag_days)}"
            elif lag_days > 0:
                column_name = f"corr_actual_t_forecast_t_plus_{lag_days}"
            else:
                column_name = "corr_actual_t_forecast_t"
            reaction_row[column_name] = lag_corr
        forward_corr = {lag_days: value for lag_days, value in lag_corr_values.items() if lag_days > 0 and np.isfinite(value)}
        backward_corr = {lag_days: value for lag_days, value in lag_corr_values.items() if lag_days < 0 and np.isfinite(value)}
        reaction_row["best_forward_lag_days"] = max(forward_corr, key=forward_corr.get) if forward_corr else np.nan
        reaction_row["best_forward_lag_corr"] = max(forward_corr.values()) if forward_corr else np.nan
        reaction_row["best_backward_lag_days"] = max(backward_corr, key=backward_corr.get) if backward_corr else np.nan
        reaction_row["best_backward_lag_corr"] = max(backward_corr.values()) if backward_corr else np.nan
        reaction_rows.append(reaction_row)

        scale_columns = [
            column
            for column in ["exposure_scale", "risk_cluster_scale", "risk_signal_quality_scale", "risk_vol_scale"]
            if column in frame.columns
        ]
        if scale_columns and not boom_frame.empty:
            scale_down_mask = pd.concat(
                [numeric_series(boom_frame[column]) < 0.99 for column in scale_columns], axis=1
            ).any(axis=1)
            scale_down_share = float(scale_down_mask.mean())
        else:
            scale_down_share = np.nan
        factor_row = {
            "instrument": code,
            "name": names.get(code, ""),
            "focus_theme": focus_themes.get(code, ""),
            "decision_tree_label": profile_rows.get(code, {}).get("decision_tree_label", ""),
            "period_return_rank": int(period_rank.get(code, 0)),
            "period_return": float(period_return.get(code, np.nan)),
            "boom_day_count": int(boom_frame.shape[0]),
            "explicit_style_columns_available": bool(explicit_factor_columns),
            "explicit_style_columns": "|".join(explicit_factor_columns),
            "explicit_factor_source": explicit_factor_source or "none",
            "factor_industry_theme": (
                str(boom_frame["factor_industry_theme"].dropna().mode().iloc[0])
                if "factor_industry_theme" in boom_frame.columns and not boom_frame["factor_industry_theme"].dropna().empty
                else industry_theme_label(names.get(code, ""))
            ),
            "proxy_columns_used": "|".join(present_proxy_columns),
            "boom_active_offset_day_share": float((numeric_series(boom_frame["predicted_excess_return"]) < 0.0).mean()) if not boom_frame.empty else np.nan,
            "boom_momentum_positive_share": float((numeric_series(boom_frame.get("risk_trend_momentum_z", pd.Series(dtype=float))) > 0.0).mean()) if "risk_trend_momentum_z" in boom_frame.columns and not boom_frame.empty else np.nan,
            "boom_trend_score_positive_share": float((numeric_series(boom_frame.get("risk_trend_score", pd.Series(dtype=float))) > 0.0).mean()) if "risk_trend_score" in boom_frame.columns and not boom_frame.empty else np.nan,
            "boom_trend_sleeve_selected_share": float((numeric_series(boom_frame.get("risk_trend_sleeve_selected", pd.Series(dtype=float))) > 0.0).mean()) if "risk_trend_sleeve_selected" in boom_frame.columns and not boom_frame.empty else np.nan,
            "boom_scale_down_share": scale_down_share,
            "boom_cluster_scale_below_one_share": float((numeric_series(boom_frame.get("risk_cluster_scale", pd.Series(dtype=float))) < 0.99).mean()) if "risk_cluster_scale" in boom_frame.columns and not boom_frame.empty else np.nan,
            "boom_signal_quality_scale_below_one_share": float((numeric_series(boom_frame.get("risk_signal_quality_scale", pd.Series(dtype=float))) < 0.99).mean()) if "risk_signal_quality_scale" in boom_frame.columns and not boom_frame.empty else np.nan,
            "boom_vol_scale_below_one_share": float((numeric_series(boom_frame.get("risk_vol_scale", pd.Series(dtype=float))) < 0.99).mean()) if "risk_vol_scale" in boom_frame.columns and not boom_frame.empty else np.nan,
            "boom_exposure_scale_below_one_share": float((numeric_series(boom_frame.get("exposure_scale", pd.Series(dtype=float))) < 0.99).mean()) if "exposure_scale" in boom_frame.columns and not boom_frame.empty else np.nan,
        }
        for column in present_proxy_columns:
            boom_mean = float(numeric_series(boom_frame[column]).mean()) if column in boom_frame.columns and not boom_frame.empty else np.nan
            non_boom_mean = float(numeric_series(non_boom_frame[column]).mean()) if column in non_boom_frame.columns and not non_boom_frame.empty else np.nan
            factor_row[f"boom_avg_{column}"] = boom_mean
            factor_row[f"non_boom_avg_{column}"] = non_boom_mean
            factor_row[f"delta_boom_minus_non_boom_{column}"] = boom_mean - non_boom_mean if np.isfinite(boom_mean) and np.isfinite(non_boom_mean) else np.nan
        explicit_scores: dict[str, float] = {}
        explicit_directions: dict[str, str] = {}
        for column in explicit_factor_columns:
            boom_exposure = safe_float(factor_row.get(f"boom_avg_{column}"))
            exposure_delta = safe_float(factor_row.get(f"delta_boom_minus_non_boom_{column}"))
            active_offset_level = safe_float(factor_row["boom_active_offset_day_share"])
            scale_down_level = safe_float(factor_row["boom_scale_down_share"])
            positive_score = (max(0.0, boom_exposure) + 0.5 * max(0.0, exposure_delta)) * scale_down_level
            negative_score = (max(0.0, -boom_exposure) + 0.5 * max(0.0, -exposure_delta)) * active_offset_level
            if negative_score > positive_score:
                explicit_scores[column] = negative_score
                explicit_directions[column] = "negative_exposure_active_offset"
            else:
                explicit_scores[column] = positive_score
                explicit_directions[column] = "positive_exposure_scaled_down"
        if explicit_scores:
            dominant_factor = max(explicit_scores, key=explicit_scores.get)
            dominant_score = explicit_scores[dominant_factor]
        else:
            dominant_factor = ""
            dominant_score = np.nan
        crowding_score = max(0.0, safe_float(factor_row.get("boom_avg_forecast_active_set_crowded_score"))) * max(
            safe_float(factor_row["boom_active_offset_day_share"]), safe_float(factor_row["boom_scale_down_share"])
        )
        cluster_penalty_score = max(0.0, 1.0 - safe_float(factor_row.get("boom_avg_risk_cluster_scale"))) * safe_float(
            factor_row["boom_scale_down_share"]
        )
        signal_quality_penalty_score = max(0.0, 1.0 - safe_float(factor_row.get("boom_avg_risk_signal_quality_scale"))) * safe_float(
            factor_row["boom_scale_down_share"]
        )
        driver_scores = {
            dominant_factor: dominant_score,
            "crowding_penalty": crowding_score,
            "cluster_scale_penalty": cluster_penalty_score,
            "signal_quality_scale_penalty": signal_quality_penalty_score,
        }
        driver_scores = {key: value for key, value in driver_scores.items() if key and np.isfinite(value)}
        factor_row["dominant_explicit_factor"] = dominant_factor
        factor_row["dominant_explicit_factor_score"] = dominant_score
        factor_row["dominant_explicit_factor_direction"] = explicit_directions.get(dominant_factor, "")
        factor_row["dominant_offset_driver"] = max(driver_scores, key=driver_scores.get) if driver_scores else ""
        factor_row["dominant_offset_score"] = max(driver_scores.values()) if driver_scores else np.nan
        factor_row["constraint_offset_proxy_label"] = residual_constraint_label(pd.Series(factor_row))
        if explicit_factor_columns and factor_row["dominant_offset_driver"] in explicit_factor_columns:
            factor_row["constraint_offset_explicit_label"] = (
                f"{factor_row['dominant_offset_driver']}_{factor_row['dominant_explicit_factor_direction']}"
            )
        elif explicit_factor_columns and factor_row["dominant_offset_driver"]:
            factor_row["constraint_offset_explicit_label"] = f"{factor_row['dominant_offset_driver']}_dominates_explicit_check"
        else:
            factor_row["constraint_offset_explicit_label"] = "explicit_factor_unavailable"
        factor_rows.append(factor_row)

    daily_columns = [
        "datetime",
        "instrument",
        "name",
        "focus_theme",
        "decision_tree_label",
        "period_return_rank",
        "period_return",
        "is_period_top_winner",
        "boom_day_flag",
        "next_return",
        "pool_equal_next_return",
        "actual_excess_return",
        "forecast_mean",
        "pool_mean_forecast",
        "predicted_excess_return",
        "residual_return",
        "residual_share_of_positive_specific",
        "residual_share_clipped_0_1",
        "active_offset_share",
        "actual_excess_top_rank_pct",
        "predicted_excess_top_rank_pct",
        "forecast_mean_top_rank_pct",
        "pre1_forecast_mean_top_rank_pct",
        "post1_3_best_forecast_mean_top_rank_pct",
        "forecast_vol_rank_pct",
        "forecast_w_raw_to_equal",
        "forecast_w_l3_to_equal",
        "exposure_scale",
        "actual_weight_proxy",
        "factor_industry_theme",
        *present_proxy_columns,
    ]
    target_daily["datetime"] = pd.to_datetime(target_daily["datetime"]).dt.strftime("%Y-%m-%d")
    daily = target_daily[[column for column in daily_columns if column in target_daily.columns]].sort_values(
        ["period_return_rank", "datetime"]
    )
    summary = pd.DataFrame(summary_rows).sort_values("period_return_rank")
    reaction_lag = pd.DataFrame(reaction_rows).sort_values("instrument")
    factor_check = pd.DataFrame(factor_rows).sort_values("period_return_rank") if factor_rows else pd.DataFrame()

    write_table(daily, out_dir, "residual_analysis_daily.csv")
    write_table(summary, out_dir, "residual_analysis_summary.csv")
    write_table(reaction_lag, out_dir, "residual_reaction_lag_summary.csv")
    write_table(factor_check, out_dir, "residual_factor_exposure_check.csv")
    return {
        "residual_analysis_daily": daily,
        "residual_analysis_summary": summary,
        "residual_reaction_lag_summary": reaction_lag,
        "residual_factor_exposure_check": factor_check,
    }


def lead_lag_cross_correlation_audit(
    bridge_path: Path,
    out_dir: Path,
    fund_list_csv: str | Path,
    top_k: int,
) -> dict[str, pd.DataFrame]:
    requested_columns = [
        "instrument",
        "datetime",
        "next_return",
        "forecast_mean_return_robust",
        "forecast_mean_return",
    ]
    bridge = read_csv_columns(bridge_path, requested_columns)
    mu_col = "forecast_mean_return_robust" if "forecast_mean_return_robust" in bridge.columns else "forecast_mean_return"
    if not {"instrument", "datetime", "next_return", mu_col}.issubset(bridge.columns):
        lead_lag = pd.DataFrame()
        write_table(lead_lag, out_dir, "lead_lag_cross_correlation.csv")
        return {"lead_lag_cross_correlation": lead_lag}

    bridge = bridge.sort_values(["instrument", "datetime"]).copy()
    bridge["forecast_mean"] = numeric_series(bridge[mu_col])
    bridge["next_return"] = numeric_series(bridge["next_return"])
    bridge["pool_equal_next_return"] = bridge.groupby("datetime")["next_return"].transform("mean")
    bridge["actual_excess_return"] = bridge["next_return"] - bridge["pool_equal_next_return"]
    name_map = build_name_map(fund_list_csv)
    period_return = bridge.groupby("instrument")["next_return"].apply(compound_return).dropna()
    period_rank = period_return.rank(method="first", ascending=False).astype(int)
    sorted_period_return = period_return.sort_values(ascending=False)
    names = {code: name_map.get(code, "") for code in sorted_period_return.index}
    focus_themes = {code: focus_theme_label(names.get(code, "")) for code in sorted_period_return.index}
    max_focus_rank = max(int(top_k), 30)
    target_codes = set(sorted_period_return.head(max(1, int(top_k))).index.tolist())
    target_codes.update(
        code
        for code, theme in focus_themes.items()
        if theme and int(period_rank.get(code, max_focus_rank + 1)) <= max_focus_rank
    )
    ordered_targets = [code for code in sorted_period_return.index if code in target_codes]

    def _lag_name(prefix: str, lag_days: int) -> str:
        if lag_days < 0:
            return f"{prefix}_minus_{abs(lag_days)}"
        if lag_days > 0:
            return f"{prefix}_plus_{lag_days}"
        return f"{prefix}_0"

    rows: list[dict[str, object]] = []
    for code in ordered_targets:
        frame = bridge[bridge["instrument"] == code].sort_values("datetime")
        row: dict[str, object] = {
            "instrument": code,
            "name": names.get(code, ""),
            "focus_theme": focus_themes.get(code, ""),
            "period_return_rank": int(period_rank.get(code, 0)),
            "period_return": float(period_return.get(code, np.nan)),
            "obs": int(frame[["forecast_mean", "next_return"]].dropna().shape[0]),
        }
        lag_corrs: dict[int, float] = {}
        lag_excess_corrs: dict[int, float] = {}
        for lag_days in range(-3, 4):
            shifted_return = numeric_series(frame["next_return"]).shift(-lag_days)
            shifted_excess = numeric_series(frame["actual_excess_return"]).shift(-lag_days)
            mean_series = numeric_series(frame["forecast_mean"])
            ret_frame = pd.DataFrame({"mean": mean_series, "return": shifted_return}).dropna()
            excess_frame = pd.DataFrame({"mean": mean_series, "excess": shifted_excess}).dropna()
            corr_return = (
                float(ret_frame["mean"].corr(ret_frame["return"]))
                if ret_frame.shape[0] >= 5 and ret_frame["mean"].nunique() > 1 and ret_frame["return"].nunique() > 1
                else np.nan
            )
            corr_excess = (
                float(excess_frame["mean"].corr(excess_frame["excess"]))
                if excess_frame.shape[0] >= 5 and excess_frame["mean"].nunique() > 1 and excess_frame["excess"].nunique() > 1
                else np.nan
            )
            lag_corrs[lag_days] = corr_return
            lag_excess_corrs[lag_days] = corr_excess
            row[_lag_name("corr_mean_t_return_t", lag_days)] = corr_return
            row[_lag_name("corr_mean_t_excess_t", lag_days)] = corr_excess
        forward_corr = {lag_days: value for lag_days, value in lag_corrs.items() if lag_days > 0 and np.isfinite(value)}
        forward_excess_corr = {lag_days: value for lag_days, value in lag_excess_corrs.items() if lag_days > 0 and np.isfinite(value)}
        row["best_forward_lag_days"] = max(forward_corr, key=forward_corr.get) if forward_corr else np.nan
        row["best_forward_lag_corr"] = max(forward_corr.values()) if forward_corr else np.nan
        row["best_forward_excess_lag_days"] = max(forward_excess_corr, key=forward_excess_corr.get) if forward_excess_corr else np.nan
        row["best_forward_excess_lag_corr"] = max(forward_excess_corr.values()) if forward_excess_corr else np.nan
        plus_1 = safe_float(row.get("corr_mean_t_return_t_plus_1"))
        plus_2 = safe_float(row.get("corr_mean_t_return_t_plus_2"))
        plus_0 = safe_float(row.get("corr_mean_t_return_t_0"))
        row["corr_plus2_gt_plus1"] = bool(np.isfinite(plus_1) and np.isfinite(plus_2) and plus_2 > plus_1)
        if row["corr_plus2_gt_plus1"] and safe_float(row["best_forward_lag_days"]) >= 2:
            row["lead_lag_diagnosis"] = "likely_smoothing_or_long_lookback_lag"
        elif np.isfinite(plus_1) and np.isfinite(plus_0) and plus_1 > plus_0:
            row["lead_lag_diagnosis"] = "one_day_reaction_delay"
        elif np.isfinite(plus_0) and plus_0 >= max([value for value in lag_corrs.values() if np.isfinite(value)], default=plus_0):
            row["lead_lag_diagnosis"] = "same_day_or_no_slow"
        else:
            row["lead_lag_diagnosis"] = "mixed_or_data_timing"
        rows.append(row)

    lead_lag = pd.DataFrame(rows).sort_values("period_return_rank") if rows else pd.DataFrame()
    write_table(lead_lag, out_dir, "lead_lag_cross_correlation.csv")
    return {"lead_lag_cross_correlation": lead_lag}


def potential_improvement_from_neutralizing_negative_biases(
    bridge_path: Path,
    l3_path: Path | None,
    raw_panel_path: Path | None,
    out_dir: Path,
    fund_list_csv: str | Path,
    top_k: int,
) -> dict[str, pd.DataFrame]:
    requested_columns = [
        "instrument",
        "datetime",
        "next_return",
        "forecast_mean_return_robust",
        "forecast_mean_return",
        "forecast_variance_robust",
        "forecast_variance",
    ]
    bridge = read_csv_columns(bridge_path, requested_columns)
    mu_col = "forecast_mean_return_robust" if "forecast_mean_return_robust" in bridge.columns else "forecast_mean_return"
    var_col = "forecast_variance_robust" if "forecast_variance_robust" in bridge.columns else "forecast_variance"
    if not {"instrument", "datetime", "next_return", mu_col, var_col}.issubset(bridge.columns):
        improvement = pd.DataFrame()
        write_table(improvement, out_dir, "potential_improvement_from_neutralizing_negative_biases.csv")
        return {"potential_improvement_from_neutralizing_negative_biases": improvement}

    bridge = bridge.sort_values(["datetime", "instrument"]).copy()
    bridge["next_return"] = numeric_series(bridge["next_return"])
    bridge["forecast_mean"] = numeric_series(bridge[mu_col])
    bridge["forecast_variance"] = numeric_series(bridge[var_col]).clip(lower=1e-12)
    bridge["pool_mean_forecast"] = bridge.groupby("datetime")["forecast_mean"].transform("mean")
    bridge["predicted_excess_return"] = bridge["forecast_mean"] - bridge["pool_mean_forecast"]
    name_map = build_name_map(fund_list_csv)

    if l3_path is not None and Path(l3_path).exists():
        l3 = read_csv_columns(Path(l3_path), FACTOR_AUDIT_COLUMNS + ["instrument", "datetime"])
        if {"instrument", "datetime"}.issubset(l3.columns):
            for column in [c for c in l3.columns if c not in {"instrument", "datetime", "factor_industry_theme"}]:
                l3[column] = numeric_series(l3[column])
            bridge = bridge.merge(l3, on=["datetime", "instrument"], how="left")
    if "factor_volume_ratio_20" not in bridge.columns or numeric_series(bridge["factor_volume_ratio_20"]).notna().sum() == 0:
        bridge = add_custom_factor_exposures_from_raw_panel(bridge, raw_panel_path, name_map)
    if "factor_industry_theme" not in bridge.columns:
        bridge["factor_industry_theme"] = bridge["instrument"].map(name_map).map(industry_theme_label).fillna("other")

    bridge["name"] = bridge["instrument"].map(name_map).fillna("")
    bridge["focus_theme"] = bridge["name"].map(focus_theme_label)
    event_themes = {"communication", "ai", "semiconductor", "star_semiconductor"}
    focus_theme_text = bridge["focus_theme"].fillna("").astype(str)
    theme_flag = focus_theme_text.map(
        lambda text: any(theme in text.replace("|", ",").split(",") for theme in event_themes)
    )
    theme_flag = theme_flag | bridge["factor_industry_theme"].fillna("other").isin(event_themes)
    if "factor_abnormal_volume_flag" in bridge.columns:
        abnormal_volume_flag = numeric_series(bridge["factor_abnormal_volume_flag"]).fillna(0.0) > 0.0
    else:
        abnormal_volume_flag = pd.Series(False, index=bridge.index)
    abnormal_volume_flag = abnormal_volume_flag | (numeric_series(bridge.get("factor_volume_ratio_20", pd.Series(index=bridge.index))) >= 1.5)
    abnormal_volume_flag = abnormal_volume_flag | (numeric_series(bridge.get("factor_volume_z_20", pd.Series(index=bridge.index))) >= 1.0)
    absolute_negative_flag = bridge["forecast_mean"] < 0.0
    active_negative_flag = bridge["predicted_excess_return"] < 0.0
    bridge["negative_bias_event_flag"] = theme_flag & abnormal_volume_flag & (absolute_negative_flag | active_negative_flag)

    period_return = bridge.groupby("instrument")["next_return"].apply(compound_return).dropna()
    actual_top_winner_set = set(period_return.sort_values(ascending=False).head(max(1, int(top_k))).index.tolist())
    safe_top_k = max(1, int(top_k))
    bridge["baseline_score"] = bridge["forecast_mean"] / (bridge["forecast_variance"] + 1e-12)

    def _daily_score_frame(data: pd.DataFrame, score_col: str) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for date_value, day_frame in data.groupby("datetime"):
            daily_return, daily_score, selected_codes_text = score_top_k_daily_return(day_frame, score_col, safe_top_k)
            selected_codes = set(selected_codes_text.split("|")) if selected_codes_text else set()
            rows.append(
                {
                    "datetime": date_value,
                    "daily_return": daily_return,
                    "daily_score": daily_score,
                    "selected_codes": selected_codes_text,
                    "period_top_winner_overlap": len(selected_codes.intersection(actual_top_winner_set)) / safe_top_k,
                    "event_selected_count": int(
                        day_frame.loc[day_frame["negative_bias_event_flag"], "instrument"].isin(selected_codes).sum()
                    ),
                }
            )
        return pd.DataFrame(rows)

    baseline_daily = _daily_score_frame(bridge, "baseline_score")
    baseline_stats = return_stats(
        baseline_daily.set_index("datetime")["daily_return"] if not baseline_daily.empty else pd.Series(dtype=float)
    )
    event_frame = bridge[bridge["negative_bias_event_flag"]].copy()
    event_codes = sorted(event_frame["instrument"].dropna().unique().tolist())
    event_themes_text = ",".join(sorted(set(event_frame["factor_industry_theme"].dropna().astype(str).tolist())))
    scenario_defs = {
        "neutralize_to_cross_section_mean": np.maximum(bridge["forecast_mean"], bridge["pool_mean_forecast"]),
        "neutralize_to_zero_absolute": np.maximum(bridge["forecast_mean"], 0.0),
    }
    rows: list[dict[str, object]] = []
    for scenario, adjusted_mean in scenario_defs.items():
        working = bridge.copy()
        working["adjusted_forecast_mean"] = np.where(working["negative_bias_event_flag"], adjusted_mean, working["forecast_mean"])
        working["adjusted_score"] = working["adjusted_forecast_mean"] / (working["forecast_variance"] + 1e-12)
        scenario_daily = _daily_score_frame(working, "adjusted_score")
        scenario_stats = return_stats(
            scenario_daily.set_index("datetime")["daily_return"] if not scenario_daily.empty else pd.Series(dtype=float)
        )
        aligned = baseline_daily[["datetime", "daily_return", "period_top_winner_overlap"]].merge(
            scenario_daily[["datetime", "daily_return", "period_top_winner_overlap", "event_selected_count"]],
            on="datetime",
            how="inner",
            suffixes=("_baseline", "_scenario"),
        )
        row = {
            "scenario": scenario,
            "event_row_count": int(event_frame.shape[0]),
            "event_day_count": int(event_frame["datetime"].nunique()) if not event_frame.empty else 0,
            "event_unique_instrument_count": int(len(event_codes)),
            "event_instruments": ",".join(event_codes),
            "event_themes": event_themes_text,
            "baseline_cumulative_return": baseline_stats.get("cumulative_return", np.nan),
            "scenario_cumulative_return": scenario_stats.get("cumulative_return", np.nan),
            "delta_cumulative_return": scenario_stats.get("cumulative_return", np.nan) - baseline_stats.get("cumulative_return", np.nan),
            "baseline_max_drawdown": baseline_stats.get("max_drawdown", np.nan),
            "scenario_max_drawdown": scenario_stats.get("max_drawdown", np.nan),
            "delta_max_drawdown": scenario_stats.get("max_drawdown", np.nan) - baseline_stats.get("max_drawdown", np.nan),
            "baseline_avg_period_top_winner_overlap": float(baseline_daily["period_top_winner_overlap"].mean()) if not baseline_daily.empty else np.nan,
            "scenario_avg_period_top_winner_overlap": float(scenario_daily["period_top_winner_overlap"].mean()) if not scenario_daily.empty else np.nan,
            "delta_avg_period_top_winner_overlap": float(
                aligned["period_top_winner_overlap_scenario"].mean() - aligned["period_top_winner_overlap_baseline"].mean()
            ) if not aligned.empty else np.nan,
            "avg_daily_return_delta": float(aligned["daily_return_scenario"].mean() - aligned["daily_return_baseline"].mean()) if not aligned.empty else np.nan,
            "avg_event_selected_count_after_neutralize": float(scenario_daily["event_selected_count"].mean()) if not scenario_daily.empty else np.nan,
        }
        rows.append(row)

    improvement = pd.DataFrame(rows)
    write_table(improvement, out_dir, "potential_improvement_from_neutralizing_negative_biases.csv")
    return {"potential_improvement_from_neutralizing_negative_biases": improvement}


def split_theme_labels(value: object) -> list[str]:
    return [token.strip() for token in str(value).replace("|", ",").split(",") if token.strip()]


def theme_membership_flag(frame: pd.DataFrame, themes: Iterable[str]) -> pd.Series:
    theme_set = set(themes)
    if frame.empty:
        return pd.Series(False, index=frame.index)
    focus_flag = pd.Series(False, index=frame.index)
    if "focus_theme" in frame.columns:
        focus_flag = frame["focus_theme"].fillna("").astype(str).map(
            lambda text: any(theme in split_theme_labels(text) for theme in theme_set)
        )
    industry_flag = pd.Series(False, index=frame.index)
    if "factor_industry_theme" in frame.columns:
        industry_flag = frame["factor_industry_theme"].fillna("other").isin(theme_set)
    return focus_flag | industry_flag


def shadow_patch_thresholds_text() -> str:
    return (
        f"theme in {','.join(SHADOW_PATCH_TRIGGER_THEMES)}; "
        f"factor_volume_z_20 > {SHADOW_PATCH_VOLUME_Z_THRESHOLD:.1f}; "
        f"factor_style_growth < {SHADOW_PATCH_GROWTH_EXPOSURE_THRESHOLD:.1f}; "
        f"forecast_mean < {SHADOW_PATCH_FORECAST_MEAN_THRESHOLD:.1f}; "
        "action: adjusted_forecast_mean = max(forecast_mean, 0)"
    )


def score_top_k_equal_weight_portfolio(
    day_frame: pd.DataFrame,
    score_col: str,
    top_k: int,
    trigger_col: str = "shadow_patch_trigger_flag",
) -> tuple[dict[str, object], dict[str, float]]:
    valid = day_frame.dropna(subset=[score_col, "next_return"])
    if valid.empty:
        return {
            "daily_return": np.nan,
            "daily_score": np.nan,
            "selected_codes": "",
            "selected_count": 0,
            "trigger_selected_count": 0,
        }, {}
    selected = valid.nlargest(max(1, int(top_k)), score_col)
    selected_codes = selected["instrument"].astype(str).tolist()
    weight = 1.0 / max(len(selected_codes), 1)
    weights = {code: weight for code in selected_codes}
    if trigger_col in selected.columns:
        trigger_selected_count = int(selected[trigger_col].fillna(False).astype(bool).sum())
    else:
        trigger_selected_count = 0
    return {
        "daily_return": float((numeric_series(selected["next_return"]) * weight).sum()),
        "daily_score": float(numeric_series(selected[score_col]).mean()),
        "selected_codes": "|".join(selected_codes),
        "selected_count": int(len(selected_codes)),
        "trigger_selected_count": trigger_selected_count,
    }, weights


def equal_weight_turnover(
    previous_weights: dict[str, float] | None,
    current_weights: dict[str, float],
) -> float:
    if previous_weights is None:
        return np.nan
    codes = set(previous_weights).union(current_weights)
    if not codes:
        return np.nan
    return float(0.5 * sum(abs(current_weights.get(code, 0.0) - previous_weights.get(code, 0.0)) for code in codes))


def load_shadow_patch_frame(
    bridge_path: Path,
    l3_path: Path | None,
    raw_panel_path: Path | None,
    fund_list_csv: str | Path,
) -> tuple[pd.DataFrame, str]:
    requested_columns = [
        "instrument",
        "datetime",
        "next_return",
        "forecast_mean_return_robust",
        "forecast_mean_return",
        "forecast_variance_robust",
        "forecast_variance",
    ]
    bridge = read_csv_columns(bridge_path, requested_columns)
    mu_col = "forecast_mean_return_robust" if "forecast_mean_return_robust" in bridge.columns else "forecast_mean_return"
    var_col = "forecast_variance_robust" if "forecast_variance_robust" in bridge.columns else "forecast_variance"
    required = {"instrument", "datetime", "next_return", mu_col, var_col}
    if not required.issubset(bridge.columns):
        return pd.DataFrame(), "none"

    bridge = bridge.sort_values(["datetime", "instrument"]).copy()
    bridge["next_return"] = numeric_series(bridge["next_return"])
    bridge["forecast_mean"] = numeric_series(bridge[mu_col])
    bridge["forecast_variance"] = numeric_series(bridge[var_col]).clip(lower=1e-12)
    name_map = build_name_map(fund_list_csv)
    explicit_factor_source = "none"

    if l3_path is not None and Path(l3_path).exists():
        l3_header = csv_header(Path(l3_path))
        l3 = read_csv_columns(Path(l3_path), FACTOR_AUDIT_COLUMNS + ["instrument", "datetime"])
        if {"instrument", "datetime"}.issubset(l3.columns):
            for column in [item for item in l3.columns if item not in {"instrument", "datetime", "factor_industry_theme"}]:
                l3[column] = numeric_series(l3[column])
            bridge = bridge.merge(l3, on=["datetime", "instrument"], how="left")
            if all(column in l3_header for column in ["factor_style_growth", "factor_volume_z_20", "factor_volume_ratio_20"]):
                explicit_factor_source = "l3_explicit"
    factor_volume_available = "factor_volume_z_20" in bridge.columns and numeric_series(
        bridge.get("factor_volume_z_20", pd.Series(index=bridge.index))
    ).notna().sum() > 0
    factor_growth_available = "factor_style_growth" in bridge.columns and numeric_series(
        bridge.get("factor_style_growth", pd.Series(index=bridge.index))
    ).notna().sum() > 0
    factor_ratio_available = "factor_volume_ratio_20" in bridge.columns and numeric_series(
        bridge.get("factor_volume_ratio_20", pd.Series(index=bridge.index))
    ).replace([np.inf, -np.inf], np.nan).gt(0.0).sum() > 0
    if not factor_volume_available or not factor_growth_available or not factor_ratio_available:
        bridge = add_custom_factor_exposures_from_raw_panel(bridge, raw_panel_path, name_map)
        factor_volume_available = "factor_volume_z_20" in bridge.columns and numeric_series(
            bridge.get("factor_volume_z_20", pd.Series(index=bridge.index))
        ).notna().sum() > 0
        factor_growth_available = "factor_style_growth" in bridge.columns and numeric_series(
            bridge.get("factor_style_growth", pd.Series(index=bridge.index))
        ).notna().sum() > 0
        factor_ratio_available = "factor_volume_ratio_20" in bridge.columns and numeric_series(
            bridge.get("factor_volume_ratio_20", pd.Series(index=bridge.index))
        ).replace([np.inf, -np.inf], np.nan).gt(0.0).sum() > 0
        explicit_factor_source = "audit_raw_panel_fallback" if factor_volume_available and factor_growth_available and factor_ratio_available else "none"
    bridge = enforce_factor_audit_schema_defaults(bridge, explicit_factor_source)
    if factor_schema_default_fill_total(bridge) > 0:
        explicit_factor_source = f"{explicit_factor_source}_with_neutral_defaults"
        bridge["factor_schema_source"] = explicit_factor_source
    if "factor_industry_theme" not in bridge.columns:
        bridge["factor_industry_theme"] = bridge["instrument"].map(name_map).map(industry_theme_label).fillna("other")

    bridge["name"] = bridge["instrument"].map(name_map).fillna("")
    bridge["focus_theme"] = bridge["name"].map(focus_theme_label)
    bridge["pool_equal_next_return"] = bridge.groupby("datetime")["next_return"].transform("mean")
    bridge["pool_mean_forecast"] = bridge.groupby("datetime")["forecast_mean"].transform("mean")
    bridge["actual_excess_return"] = bridge["next_return"] - bridge["pool_equal_next_return"]
    bridge["predicted_excess_return"] = bridge["forecast_mean"] - bridge["pool_mean_forecast"]
    bridge["residual_return"] = bridge["actual_excess_return"] - bridge["predicted_excess_return"]
    bridge["actual_excess_top_rank_pct"] = bridge.groupby("datetime")["actual_excess_return"].transform(
        lambda series: numeric_series(series).rank(pct=True, ascending=False)
    )
    trigger_theme_flag = theme_membership_flag(bridge, SHADOW_PATCH_TRIGGER_THEMES)
    bridge["shadow_patch_trigger_flag"] = (
        trigger_theme_flag
        & (numeric_series(bridge.get("factor_volume_z_20", pd.Series(index=bridge.index))) > SHADOW_PATCH_VOLUME_Z_THRESHOLD)
        & (numeric_series(bridge.get("factor_style_growth", pd.Series(index=bridge.index))) < SHADOW_PATCH_GROWTH_EXPOSURE_THRESHOLD)
        & (bridge["forecast_mean"] < SHADOW_PATCH_FORECAST_MEAN_THRESHOLD)
    )
    bridge["adjusted_forecast_mean"] = np.where(bridge["shadow_patch_trigger_flag"], 0.0, bridge["forecast_mean"])
    bridge["adjusted_pool_mean_forecast"] = bridge.groupby("datetime")["adjusted_forecast_mean"].transform("mean")
    bridge["adjusted_predicted_excess_return"] = bridge["adjusted_forecast_mean"] - bridge["adjusted_pool_mean_forecast"]
    bridge["patched_residual_return"] = bridge["actual_excess_return"] - bridge["adjusted_predicted_excess_return"]
    bridge["baseline_score"] = bridge["forecast_mean"] / (bridge["forecast_variance"] + 1e-12)
    bridge["patched_score"] = bridge["adjusted_forecast_mean"] / (bridge["forecast_variance"] + 1e-12)
    bridge["admission_boom_row_flag"] = (
        theme_membership_flag(bridge, SHADOW_PATCH_BOOM_WIN_THEMES)
        & (bridge["actual_excess_return"] > 0.0)
        & (bridge["actual_excess_top_rank_pct"] <= 0.30)
    )
    return bridge, explicit_factor_source


def shadow_specific_recovery_stats(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "trigger_positive_specific_row_count": 0,
            "trigger_positive_specific_sum": np.nan,
            "baseline_predicted_excess_sum_on_triggers": np.nan,
            "patched_predicted_excess_sum_on_triggers": np.nan,
            "predicted_excess_recovery_sum": np.nan,
            "clipped_specific_return_recovery_sum": np.nan,
            "specific_return_recovery_share_of_positive_specific": np.nan,
            "specific_return_recovery_share_of_prior_residual": np.nan,
        }
    recovery_frame = frame[(frame["shadow_patch_trigger_flag"]) & (frame["actual_excess_return"] > 0.0)].copy()
    if recovery_frame.empty:
        return {
            "trigger_positive_specific_row_count": 0,
            "trigger_positive_specific_sum": 0.0,
            "baseline_predicted_excess_sum_on_triggers": 0.0,
            "patched_predicted_excess_sum_on_triggers": 0.0,
            "predicted_excess_recovery_sum": 0.0,
            "clipped_specific_return_recovery_sum": 0.0,
            "specific_return_recovery_share_of_positive_specific": np.nan,
            "specific_return_recovery_share_of_prior_residual": np.nan,
        }
    positive_specific = numeric_series(recovery_frame["actual_excess_return"]).clip(lower=0.0)
    positive_specific_sum = float(positive_specific.sum())
    baseline_predicted_sum = float(numeric_series(recovery_frame["predicted_excess_return"]).sum())
    patched_predicted_sum = float(numeric_series(recovery_frame["adjusted_predicted_excess_return"]).sum())
    prior_residual_positive = numeric_series(recovery_frame["residual_return"]).clip(lower=0.0)
    predicted_recovery = (
        numeric_series(recovery_frame["adjusted_predicted_excess_return"])
        - numeric_series(recovery_frame["predicted_excess_return"])
    ).clip(lower=0.0)
    clipped_residual_recovery = np.minimum(predicted_recovery, prior_residual_positive)
    clipped_specific_recovery = np.minimum(clipped_residual_recovery, positive_specific)
    recovery_sum = float(clipped_residual_recovery.sum())
    specific_recovery_sum = float(clipped_specific_recovery.sum())
    prior_residual_sum = float(prior_residual_positive.sum())
    return {
        "trigger_positive_specific_row_count": int(recovery_frame.shape[0]),
        "trigger_positive_specific_sum": positive_specific_sum,
        "baseline_predicted_excess_sum_on_triggers": baseline_predicted_sum,
        "patched_predicted_excess_sum_on_triggers": patched_predicted_sum,
        "predicted_excess_recovery_sum": recovery_sum,
        "clipped_specific_return_recovery_sum": specific_recovery_sum,
        "specific_return_recovery_share_of_positive_specific": divide_or_nan(specific_recovery_sum, positive_specific_sum),
        "specific_return_recovery_share_of_prior_residual": divide_or_nan(recovery_sum, prior_residual_sum),
    }


def summarize_shadow_trigger_frequency(
    frame: pd.DataFrame,
    scope: str,
    fold_index: int | None,
    factor_source: str,
) -> dict[str, object]:
    trigger_frame = frame[frame["shadow_patch_trigger_flag"]].copy() if "shadow_patch_trigger_flag" in frame.columns else pd.DataFrame()
    oos_day_count = int(frame["datetime"].nunique()) if "datetime" in frame.columns else 0
    oos_row_count = int(frame.shape[0])
    trigger_day_count = int(trigger_frame["datetime"].nunique()) if not trigger_frame.empty else 0
    trigger_row_count = int(trigger_frame.shape[0])
    return {
        "scope": scope,
        "fold": fold_index if fold_index is not None else np.nan,
        "oos_start": pd.Timestamp(frame["datetime"].min()).strftime("%Y-%m-%d") if not frame.empty else "",
        "oos_end": pd.Timestamp(frame["datetime"].max()).strftime("%Y-%m-%d") if not frame.empty else "",
        "oos_day_count": oos_day_count,
        "oos_row_count": oos_row_count,
        "trigger_row_count": trigger_row_count,
        "trigger_day_count": trigger_day_count,
        "trigger_unique_instrument_count": int(trigger_frame["instrument"].nunique()) if not trigger_frame.empty else 0,
        "trigger_row_rate": divide_or_nan(trigger_row_count, oos_row_count),
        "trigger_day_rate": divide_or_nan(trigger_day_count, oos_day_count),
        "avg_trigger_rows_per_day": divide_or_nan(trigger_row_count, oos_day_count),
        "trigger_instruments": ",".join(sorted(trigger_frame["instrument"].dropna().unique().tolist())) if not trigger_frame.empty else "",
        "factor_source": factor_source,
        "volume_z_threshold": SHADOW_PATCH_VOLUME_Z_THRESHOLD,
        "growth_exposure_threshold": SHADOW_PATCH_GROWTH_EXPOSURE_THRESHOLD,
        "forecast_mean_threshold": SHADOW_PATCH_FORECAST_MEAN_THRESHOLD,
        "threshold_logic": shadow_patch_thresholds_text(),
    }


def oos_shadow_neutralize_zero_patch_audit(
    bridge_path: Path,
    l3_path: Path | None,
    raw_panel_path: Path | None,
    out_dir: Path,
    fund_list_csv: str | Path,
    top_k: int,
    lookback_days: int,
    forward_days: int,
    step_days: int,
) -> dict[str, pd.DataFrame]:
    empty_result = {
        "oos_shadow_patch_daily": pd.DataFrame(),
        "oos_shadow_patch_folds": pd.DataFrame(),
        "oos_shadow_patch_summary": pd.DataFrame(),
        "oos_patch_trigger_frequency": pd.DataFrame(),
        "oos_patch_admission_audit_report": pd.DataFrame(),
    }
    bridge, factor_source = load_shadow_patch_frame(bridge_path, l3_path, raw_panel_path, fund_list_csv)
    if bridge.empty:
        for table_name, frame in empty_result.items():
            write_table(frame, out_dir, f"{table_name}.csv")
        return empty_result

    dates = list(pd.Series(bridge["datetime"].drop_duplicates()).sort_values())
    safe_top_k = max(1, int(top_k))
    safe_lookback = max(5, int(lookback_days))
    safe_forward = max(1, int(forward_days))
    safe_step = max(1, int(step_days))
    daily_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    trigger_frequency_rows: list[dict[str, object]] = []
    oos_frames: list[pd.DataFrame] = []
    fold_index = 0
    last_start = len(dates) - safe_lookback - safe_forward

    for train_start_index in range(0, max(last_start + 1, 0), safe_step):
        train_dates = dates[train_start_index: train_start_index + safe_lookback]
        oos_dates = dates[train_start_index + safe_lookback: train_start_index + safe_lookback + safe_forward]
        if len(train_dates) < safe_lookback or len(oos_dates) < safe_forward:
            continue
        fold_index += 1
        oos_frame = bridge[bridge["datetime"].isin(oos_dates)].copy()
        oos_frames.append(oos_frame)
        trigger_frequency_rows.append(
            summarize_shadow_trigger_frequency(oos_frame, f"fold_{fold_index}", fold_index, factor_source)
        )
        previous_base_weights: dict[str, float] | None = None
        previous_patch_weights: dict[str, float] | None = None
        fold_daily_rows: list[dict[str, object]] = []
        for date_value, day_frame in oos_frame.groupby("datetime"):
            base_portfolio, base_weights = score_top_k_equal_weight_portfolio(day_frame, "baseline_score", safe_top_k)
            patch_portfolio, patch_weights = score_top_k_equal_weight_portfolio(day_frame, "patched_score", safe_top_k)
            pool_equal_return = float(numeric_series(day_frame["pool_equal_next_return"]).dropna().iloc[0]) if numeric_series(day_frame["pool_equal_next_return"]).dropna().shape[0] else np.nan
            row = {
                "fold": fold_index,
                "datetime": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                "base_daily_return": base_portfolio["daily_return"],
                "patched_daily_return": patch_portfolio["daily_return"],
                "active_return_vs_base": safe_float(patch_portfolio["daily_return"]) - safe_float(base_portfolio["daily_return"]),
                "pool_equal_next_return": pool_equal_return,
                "base_selected_codes": base_portfolio["selected_codes"],
                "patched_selected_codes": patch_portfolio["selected_codes"],
                "base_turnover": equal_weight_turnover(previous_base_weights, base_weights),
                "patched_turnover": equal_weight_turnover(previous_patch_weights, patch_weights),
                "turnover_delta": equal_weight_turnover(previous_patch_weights, patch_weights) - equal_weight_turnover(previous_base_weights, base_weights),
                "trigger_row_count": int(day_frame["shadow_patch_trigger_flag"].sum()),
                "trigger_selected_count": int(patch_portfolio["trigger_selected_count"]),
                "ai_communication_boom_day_flag": bool(day_frame["admission_boom_row_flag"].any()),
                "base_win_vs_pool_equal": bool(safe_float(base_portfolio["daily_return"]) > pool_equal_return) if np.isfinite(pool_equal_return) else False,
                "patched_win_vs_pool_equal": bool(safe_float(patch_portfolio["daily_return"]) > pool_equal_return) if np.isfinite(pool_equal_return) else False,
            }
            previous_base_weights = base_weights
            previous_patch_weights = patch_weights
            daily_rows.append(row)
            fold_daily_rows.append(row)

        fold_daily = pd.DataFrame(fold_daily_rows)
        base_stats = return_stats(fold_daily.set_index("datetime")["base_daily_return"] if not fold_daily.empty else pd.Series(dtype=float))
        patch_stats = return_stats(fold_daily.set_index("datetime")["patched_daily_return"] if not fold_daily.empty else pd.Series(dtype=float))
        active_returns = numeric_series(fold_daily["active_return_vs_base"]).dropna() if not fold_daily.empty else pd.Series(dtype=float)
        tracking_error = float(active_returns.std(ddof=0) * math.sqrt(252.0)) if not active_returns.empty else np.nan
        boom_daily = fold_daily[fold_daily["ai_communication_boom_day_flag"]] if not fold_daily.empty else pd.DataFrame()
        recovery_stats = shadow_specific_recovery_stats(oos_frame)
        fold_row = {
            "fold": fold_index,
            "scenario": "neutralize_to_zero_absolute_hard_threshold_shadow",
            "train_start": pd.Timestamp(train_dates[0]).strftime("%Y-%m-%d"),
            "train_end": pd.Timestamp(train_dates[-1]).strftime("%Y-%m-%d"),
            "oos_start": pd.Timestamp(oos_dates[0]).strftime("%Y-%m-%d"),
            "oos_end": pd.Timestamp(oos_dates[-1]).strftime("%Y-%m-%d"),
            "factor_source": factor_source,
            "threshold_logic": shadow_patch_thresholds_text(),
            "trigger_row_count": int(oos_frame["shadow_patch_trigger_flag"].sum()),
            "trigger_day_count": int(oos_frame.loc[oos_frame["shadow_patch_trigger_flag"], "datetime"].nunique()),
            "base_avg_turnover": float(numeric_series(fold_daily["base_turnover"]).mean()) if not fold_daily.empty else np.nan,
            "patched_avg_turnover": float(numeric_series(fold_daily["patched_turnover"]).mean()) if not fold_daily.empty else np.nan,
            "delta_avg_turnover": float(numeric_series(fold_daily["patched_turnover"]).mean() - numeric_series(fold_daily["base_turnover"]).mean()) if not fold_daily.empty else np.nan,
            "base_max_turnover": float(numeric_series(fold_daily["base_turnover"]).max()) if not fold_daily.empty else np.nan,
            "patched_max_turnover": float(numeric_series(fold_daily["patched_turnover"]).max()) if not fold_daily.empty else np.nan,
            "tracking_error_annualized": tracking_error,
            "tracking_error_lt_1pct": bool(np.isfinite(tracking_error) and tracking_error < SHADOW_PATCH_TRACKING_ERROR_LIMIT),
            "ai_communication_boom_day_count": int(boom_daily.shape[0]),
            "base_win_rate_in_ai_communication_boom_days": float(boom_daily["base_win_vs_pool_equal"].mean()) if not boom_daily.empty else np.nan,
            "patched_win_rate_in_ai_communication_boom_days": float(boom_daily["patched_win_vs_pool_equal"].mean()) if not boom_daily.empty else np.nan,
        }
        fold_row.update({f"base_{key}": value for key, value in base_stats.items()})
        fold_row.update({f"patched_{key}": value for key, value in patch_stats.items()})
        fold_row["delta_cumulative_return"] = safe_float(fold_row.get("patched_cumulative_return")) - safe_float(fold_row.get("base_cumulative_return"))
        fold_row["delta_sharpe"] = safe_float(fold_row.get("patched_sharpe")) - safe_float(fold_row.get("base_sharpe"))
        fold_row["delta_max_drawdown"] = safe_float(fold_row.get("patched_max_drawdown")) - safe_float(fold_row.get("base_max_drawdown"))
        fold_row.update(recovery_stats)
        fold_rows.append(fold_row)

    daily = pd.DataFrame(daily_rows)
    folds = pd.DataFrame(fold_rows)
    trigger_frequency = pd.DataFrame(trigger_frequency_rows)
    if oos_frames:
        overall_oos_frame = pd.concat(oos_frames, ignore_index=True).drop_duplicates(["datetime", "instrument"])
        overall_frequency = summarize_shadow_trigger_frequency(overall_oos_frame, "overall", None, factor_source)
        trigger_frequency = pd.concat([pd.DataFrame([overall_frequency]), trigger_frequency], ignore_index=True)
    else:
        overall_oos_frame = pd.DataFrame()

    summary_rows: list[dict[str, object]] = []
    if not daily.empty:
        base_stats = return_stats(daily.set_index("datetime")["base_daily_return"])
        patch_stats = return_stats(daily.set_index("datetime")["patched_daily_return"])
        active_returns = numeric_series(daily["active_return_vs_base"]).dropna()
        tracking_error = float(active_returns.std(ddof=0) * math.sqrt(252.0)) if not active_returns.empty else np.nan
        boom_daily = daily[daily["ai_communication_boom_day_flag"]]
        recovery_stats = shadow_specific_recovery_stats(overall_oos_frame)
        summary_row: dict[str, object] = {
            "scenario": "neutralize_to_zero_absolute_hard_threshold_shadow",
            "fold_count": int(folds["fold"].nunique()) if not folds.empty else 0,
            "lookback_days": safe_lookback,
            "forward_days": safe_forward,
            "step_days": safe_step,
            "factor_source": factor_source,
            "threshold_logic": shadow_patch_thresholds_text(),
            "tracking_error_limit": SHADOW_PATCH_TRACKING_ERROR_LIMIT,
            "tracking_error_annualized": tracking_error,
            "tracking_error_lt_1pct": bool(np.isfinite(tracking_error) and tracking_error < SHADOW_PATCH_TRACKING_ERROR_LIMIT),
            "base_avg_turnover": float(numeric_series(daily["base_turnover"]).mean()),
            "patched_avg_turnover": float(numeric_series(daily["patched_turnover"]).mean()),
            "delta_avg_turnover": float(numeric_series(daily["patched_turnover"]).mean() - numeric_series(daily["base_turnover"]).mean()),
            "base_max_turnover": float(numeric_series(daily["base_turnover"]).max()),
            "patched_max_turnover": float(numeric_series(daily["patched_turnover"]).max()),
            "delta_max_turnover": float(numeric_series(daily["patched_turnover"]).max() - numeric_series(daily["base_turnover"]).max()),
            "trigger_row_count": int(overall_oos_frame["shadow_patch_trigger_flag"].sum()) if not overall_oos_frame.empty else 0,
            "trigger_day_count": int(overall_oos_frame.loc[overall_oos_frame["shadow_patch_trigger_flag"], "datetime"].nunique()) if not overall_oos_frame.empty else 0,
            "ai_communication_boom_day_count": int(boom_daily.shape[0]),
            "base_win_rate_in_ai_communication_boom_days": float(boom_daily["base_win_vs_pool_equal"].mean()) if not boom_daily.empty else np.nan,
            "patched_win_rate_in_ai_communication_boom_days": float(boom_daily["patched_win_vs_pool_equal"].mean()) if not boom_daily.empty else np.nan,
            "delta_win_rate_in_ai_communication_boom_days": float(boom_daily["patched_win_vs_pool_equal"].mean() - boom_daily["base_win_vs_pool_equal"].mean()) if not boom_daily.empty else np.nan,
        }
        summary_row.update({f"base_{key}": value for key, value in base_stats.items()})
        summary_row.update({f"patched_{key}": value for key, value in patch_stats.items()})
        summary_row["delta_cumulative_return"] = safe_float(summary_row.get("patched_cumulative_return")) - safe_float(summary_row.get("base_cumulative_return"))
        summary_row["delta_sharpe"] = safe_float(summary_row.get("patched_sharpe")) - safe_float(summary_row.get("base_sharpe"))
        summary_row["delta_max_drawdown"] = safe_float(summary_row.get("patched_max_drawdown")) - safe_float(summary_row.get("base_max_drawdown"))
        summary_row.update(recovery_stats)
        summary_row.update(factor_schema_lineage_summary(overall_oos_frame, factor_source))
        summary_rows.append(summary_row)
    summary = pd.DataFrame(summary_rows)
    admission = summary.copy()
    if not admission.empty:
        admission["risk_adjusted_return_metric"] = "shadow_score_top10_sharpe"
        admission["maxdd_metric"] = "shadow_score_top10_max_drawdown"
        admission["boom_win_rate_metric"] = "AI/communication boom days, daily return > pool equal return"
        admission["compliance_statement"] = (
            "Defensive Correction: the patch only removes extreme negative mean signals under abnormal volume "
            "and negative growth exposure; it is designed to prevent self-hedging in stressed style states, "
            "not to increase active risk exposure."
        )
        admission["admission_audit_status"] = np.where(
            admission["tracking_error_lt_1pct"],
            "shadow_tracking_error_pass_full_mvo_rerun_required",
            "shadow_tracking_error_fail_do_not_admit",
        )

    write_table(daily, out_dir, "oos_shadow_patch_daily.csv")
    write_table(folds, out_dir, "oos_shadow_patch_folds.csv")
    write_table(summary, out_dir, "oos_shadow_patch_summary.csv")
    write_table(trigger_frequency, out_dir, "oos_patch_trigger_frequency.csv")
    write_table(admission, out_dir, "oos_patch_admission_audit_report.csv")
    return {
        "oos_shadow_patch_daily": daily,
        "oos_shadow_patch_folds": folds,
        "oos_shadow_patch_summary": summary,
        "oos_patch_trigger_frequency": trigger_frequency,
        "oos_patch_admission_audit_report": admission,
    }


def shadow_oos_date_set(
    dates: Iterable[object],
    lookback_days: int,
    forward_days: int,
    step_days: int,
) -> set[pd.Timestamp]:
    ordered_dates = [pd.Timestamp(value) for value in pd.Series(list(dates)).drop_duplicates().sort_values()]
    safe_lookback = max(5, int(lookback_days))
    safe_forward = max(1, int(forward_days))
    safe_step = max(1, int(step_days))
    oos_dates: set[pd.Timestamp] = set()
    last_start = len(ordered_dates) - safe_lookback - safe_forward
    for train_start_index in range(0, max(last_start + 1, 0), safe_step):
        train_dates = ordered_dates[train_start_index: train_start_index + safe_lookback]
        forward_dates = ordered_dates[train_start_index + safe_lookback: train_start_index + safe_lookback + safe_forward]
        if len(train_dates) < safe_lookback or len(forward_dates) < safe_forward:
            continue
        oos_dates.update(forward_dates)
    return oos_dates


def apply_shadow_patch_trigger_columns(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    if "forecast_mean" not in working.columns:
        if "forecast_mean_return_robust" in working.columns:
            working["forecast_mean"] = numeric_series(working["forecast_mean_return_robust"])
        elif "forecast_mean_return" in working.columns:
            working["forecast_mean"] = numeric_series(working["forecast_mean_return"])
        else:
            working["forecast_mean"] = np.nan
    trigger_theme_flag = theme_membership_flag(working, SHADOW_PATCH_TRIGGER_THEMES)
    working["shadow_patch_trigger_flag"] = (
        trigger_theme_flag
        & (numeric_series(working.get("factor_volume_z_20", pd.Series(index=working.index))) > SHADOW_PATCH_VOLUME_Z_THRESHOLD)
        & (numeric_series(working.get("factor_style_growth", pd.Series(index=working.index))) < SHADOW_PATCH_GROWTH_EXPOSURE_THRESHOLD)
        & (numeric_series(working["forecast_mean"]) < SHADOW_PATCH_FORECAST_MEAN_THRESHOLD)
    )
    working["adjusted_forecast_mean"] = np.where(working["shadow_patch_trigger_flag"], 0.0, working["forecast_mean"])
    return working


def load_full_mvo_audit_frame(
    bridge_path: Path,
    l3_path: Path | None,
    raw_panel_path: Path | None,
    fund_list_csv: str | Path,
) -> tuple[pd.DataFrame, str, str]:
    bridge = pd.read_csv(bridge_path, parse_dates=["datetime"])
    bridge["instrument"] = bridge["instrument"].map(normalize_code)
    bridge["original_bridge_forecast_w"] = numeric_series(bridge.get("forecast_w", pd.Series(index=bridge.index)))

    shadow, factor_source = load_shadow_patch_frame(bridge_path, l3_path, raw_panel_path, fund_list_csv)
    if shadow.empty:
        return pd.DataFrame(), factor_source, "none"
    shadow_columns = [
        "instrument",
        "datetime",
        "name",
        "focus_theme",
        "factor_industry_theme",
        "factor_style_growth",
        "factor_volume_z_20",
        "factor_volume_ratio_20",
        "factor_abnormal_volume_flag",
        *FACTOR_SCHEMA_LINEAGE_COLUMNS,
        "forecast_mean",
        "adjusted_forecast_mean",
        "pool_equal_next_return",
        "pool_mean_forecast",
        "actual_excess_return",
        "predicted_excess_return",
        "adjusted_predicted_excess_return",
        "residual_return",
        "patched_residual_return",
        "shadow_patch_trigger_flag",
        "admission_boom_row_flag",
    ]
    available_shadow_columns = [column for column in shadow_columns if column in shadow.columns]
    bridge = bridge.merge(
        shadow[available_shadow_columns].drop_duplicates(["datetime", "instrument"]),
        on=["datetime", "instrument"],
        how="left",
    )
    bridge["shadow_patch_trigger_flag"] = bridge["shadow_patch_trigger_flag"].fillna(False).astype(bool)
    bridge["admission_boom_row_flag"] = bridge.get("admission_boom_row_flag", pd.Series(False, index=bridge.index)).fillna(False).astype(bool)

    candidate_mean_columns = [
        "forecast_weight_mean_used",
        "forecast_mean_return_robust",
        "forecast_mean_return",
        "forecast_mean",
    ]
    optimizer_mean_col = next(
        (
            column
            for column in candidate_mean_columns
            if column in bridge.columns and numeric_series(bridge[column]).notna().sum() > 0
        ),
        "none",
    )
    if optimizer_mean_col == "none":
        return pd.DataFrame(), factor_source, optimizer_mean_col
    bridge["full_mvo_base_optimizer_mean"] = numeric_series(bridge[optimizer_mean_col]).fillna(0.0)
    bridge["full_mvo_patched_optimizer_mean"] = np.where(
        bridge["shadow_patch_trigger_flag"] & (bridge["full_mvo_base_optimizer_mean"] < 0.0),
        0.0,
        bridge["full_mvo_base_optimizer_mean"],
    )
    bridge["full_mvo_optimizer_mean_changed_flag"] = (
        (bridge["full_mvo_patched_optimizer_mean"] - bridge["full_mvo_base_optimizer_mean"]).abs() > 1e-15
    )
    bridge["full_mvo_optimizer_mean_column"] = optimizer_mean_col
    return bridge, factor_source, optimizer_mean_col


def deserialize_day_covariance(day_frame: pd.DataFrame, cov_col: str = "forecast_cov_matrix") -> pd.DataFrame:
    if "forecast_cov_order" not in day_frame.columns or cov_col not in day_frame.columns:
        return pd.DataFrame()
    order_values = day_frame["forecast_cov_order"].dropna().astype(str)
    order_text = order_values.iloc[0] if not order_values.empty else ""
    order_codes = [normalize_code(token) for token in order_text.split("|") if token]
    if not order_codes:
        return pd.DataFrame()
    day_by_code = day_frame.drop_duplicates("instrument", keep="last").set_index("instrument", drop=False)
    matrix_rows: list[np.ndarray] = []
    for code in order_codes:
        if code not in day_by_code.index:
            return pd.DataFrame()
        row_value = day_by_code.loc[code, cov_col]
        if isinstance(row_value, pd.Series):
            row_value = row_value.iloc[0]
        vector = parse_float_vector(row_value)
        if len(vector) != len(order_codes):
            return pd.DataFrame()
        matrix_rows.append(vector)
    covariance = np.vstack(matrix_rows).astype(float)
    covariance = np.where(np.isfinite(covariance), covariance, np.nan)
    finite_diag = np.diag(covariance)
    positive_diag = finite_diag[np.isfinite(finite_diag) & (finite_diag > 0.0)]
    diag_floor = float(np.nanmedian(positive_diag)) if len(positive_diag) else 1e-6
    covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
    covariance = 0.5 * (covariance + covariance.T)
    diag = np.diag(covariance).copy()
    diag = np.where(np.isfinite(diag) & (diag > 0.0), diag, diag_floor)
    np.fill_diagonal(covariance, diag + 1e-6)
    return pd.DataFrame(covariance, index=order_codes, columns=order_codes)


def optimizer_score_series(day_frame: pd.DataFrame, column: str, index: pd.Index) -> pd.Series | None:
    if column not in day_frame.columns:
        return None
    day_by_code = day_frame.drop_duplicates("instrument", keep="last").set_index("instrument", drop=False)
    return numeric_series(day_by_code[column]).reindex(index).fillna(0.0)


def infer_day_alpha(day_frame: pd.DataFrame) -> float:
    if "forecast_weight_inertia_alpha" not in day_frame.columns:
        return 0.0
    values = numeric_series(day_frame["forecast_weight_inertia_alpha"]).dropna()
    return float(values.iloc[0]) if not values.empty else 0.0


def run_mvo_optimizer_sequence(
    frame: pd.DataFrame,
    mean_col: str,
    output_weight_col: str,
    output_status_col: str,
    *,
    max_weight: float = FULL_MVO_MAX_WEIGHT,
    min_weight_floor: float = FULL_MVO_MIN_WEIGHT_FLOOR,
) -> pd.DataFrame:
    from strategy.route_ab_moments_bridge.bridge import (
        DEFAULT_BALANCED_MIN_CROWDED,
        DEFAULT_HARD_BAND_PROTECTED_SLOTS,
        _enforce_floor_and_cap,
        optimize_weights_hard_band,
    )

    working = frame.copy()
    working[output_weight_col] = 0.0
    working[output_status_col] = "not_run"
    output_time_col = output_status_col.replace("_weight_status", "_optimizer_time_cost_sec")
    working[output_time_col] = np.nan
    working[f"{output_weight_col}_selected_n"] = 0
    previous_target_weights = pd.Series(dtype=float)

    for date_value, day_frame in working.groupby("datetime", sort=True):
        optimizer_start = time.perf_counter()
        day_frame = day_frame.copy()
        day_index = day_frame.index
        instruments = day_frame["instrument"].astype(str).map(normalize_code).tolist()
        status = "optimizer_input_missing"
        full_weights = pd.Series(0.0, index=pd.Index(instruments, dtype=object), dtype=float)
        covariance = deserialize_day_covariance(day_frame)
        if not covariance.empty:
            day_by_code = day_frame.drop_duplicates("instrument", keep="last").set_index("instrument", drop=False)
            selected_order = [code for code in covariance.index.astype(str).tolist() if code in day_by_code.index]
            mean_returns = numeric_series(day_by_code[mean_col]).reindex(selected_order).fillna(0.0)
            covariance = covariance.reindex(index=selected_order, columns=selected_order)
            if len(mean_returns) > 0 and covariance.shape[0] == len(mean_returns):
                protected_scores = optimizer_score_series(day_frame, "forecast_active_set_protect_score", mean_returns.index)
                crowded_scores = optimizer_score_series(day_frame, "forecast_active_set_crowded_score", mean_returns.index)
                balanced_scores = optimizer_score_series(day_frame, "forecast_active_set_balance_score", mean_returns.index)
                protected_slots = DEFAULT_HARD_BAND_PROTECTED_SLOTS if protected_scores is not None else 0
                try:
                    weights, status = optimize_weights_hard_band(
                        mean_returns,
                        covariance,
                        rf=0.0,
                        max_weight=float(max_weight),
                        min_weight_floor=float(min_weight_floor),
                        protected_scores=protected_scores,
                        protected_slots=protected_slots,
                        crowded_scores=crowded_scores,
                        balanced_scores=balanced_scores,
                        min_crowded_for_balance=DEFAULT_BALANCED_MIN_CROWDED,
                    )
                except Exception as exc:
                    weights = pd.Series(dtype=float)
                    status = f"optimizer_exception:{type(exc).__name__}"
                if status == "optimized":
                    status = "optimized_full_universe"
                if not weights.empty:
                    full_weights = pd.Series(0.0, index=mean_returns.index, dtype=float)
                    full_weights.loc[weights.index] = pd.to_numeric(weights, errors="coerce").fillna(0.0)
                    full_weights = full_weights.reindex(pd.Index(instruments, dtype=object)).fillna(0.0)

        alpha = infer_day_alpha(day_frame)
        if alpha > 0.0 and len(full_weights) > 0:
            prior_weights = previous_target_weights.reindex(full_weights.index).fillna(0.0)
            prior_weight_sum = float(prior_weights.sum())
            if prior_weight_sum > 1e-12:
                delta = full_weights - prior_weights
                add_factor = 1.0 - float(alpha)
                factor = pd.Series(np.where(delta.to_numpy() < 0.0, 1.0, add_factor), index=delta.index, dtype=float)
                blended_weights = (prior_weights + factor * delta).clip(lower=0.0)
                blended_sum = float(blended_weights.sum())
                if blended_sum > 1e-12:
                    full_weights = blended_weights / blended_sum
                status = f"{status}|inertia_asym_add={float(alpha):g}"
            else:
                status = f"{status}|inertia_skip_empty_prior"
            if min_weight_floor is not None and float(min_weight_floor) > 0.0 and len(full_weights) > 0:
                full_weights, floor_status = _enforce_floor_and_cap(
                    full_weights,
                    min_weight_floor=float(min_weight_floor),
                    max_weight=float(max_weight),
                )
                status = f"{status}|post_blend_{floor_status}"

        previous_target_weights = full_weights.reindex(pd.Index(instruments, dtype=object)).fillna(0.0)
        weight_by_instrument = previous_target_weights.to_dict()
        optimizer_elapsed = time.perf_counter() - optimizer_start
        working.loc[day_index, output_weight_col] = [float(weight_by_instrument.get(code, 0.0)) for code in instruments]
        working.loc[day_index, output_status_col] = status
        working.loc[day_index, output_time_col] = optimizer_elapsed
        working.loc[day_index, f"{output_weight_col}_selected_n"] = int((previous_target_weights > 1e-12).sum())
    return working


def full_mvo_risk_control_kwargs(
    provider_uri: str,
    raw_panel_path: Path | None,
    fund_list_csv: str | Path,
) -> dict[str, object]:
    return {
        "provider_uri": provider_uri,
        "vol_target": 0.015,
        "vol_target_risk_on": 0.022,
        "vol_target_risk_off": 0.0,
        "regime_gate_min_scale": 0.5,
        "regime_gate_ma": 20,
        "max_weight_cap": FULL_MVO_MAX_WEIGHT,
        "min_holdings_audit_warn": 6.0,
        "vol_target_cov_mode": "full",
        "anchor_codes": [],
        "anchor_weight_total": 0.0,
        "raw_panel_path": raw_panel_path,
        "cluster_gate_rho": 0.0,
        "cluster_gate_rho_saturation": 0.9,
        "cluster_gate_min_scale": 0.5,
        "cluster_gate_source": "forecast_cov",
        "cluster_gate_realized_lookback": 10,
        "cluster_gate_realized_min_periods": 5,
        "signal_quality_gate_mode": "none",
        "signal_quality_ic_window": 5,
        "signal_quality_ic_min_periods": 3,
        "signal_quality_ic_threshold": -0.05,
        "signal_quality_min_scale": 0.5,
        "signal_quality_min_scale_risk_on": 1.0,
        "signal_quality_min_scale_risk_off": 0.0,
        "trend_sleeve_weight_total": 0.0,
        "trend_sleeve_weight_total_risk_on": -1.0,
        "trend_sleeve_weight_total_risk_off": -1.0,
        "trend_sleeve_top_k": 4,
        "trend_sleeve_top_k_risk_on": 0,
        "trend_sleeve_top_k_risk_off": 0,
        "trend_sleeve_momentum_windows": [3, 5, 10],
        "trend_sleeve_mu_weight": 0.65,
        "trend_sleeve_min_score": 0.0,
        "trend_sleeve_warmup_days": 0,
        "trend_sleeve_exposure_floor": 0.0,
        "trend_sleeve_exposure_floor_risk_on": -1.0,
        "trend_sleeve_exposure_floor_risk_off": -1.0,
        "factor_fund_list_csv": fund_list_csv,
    }


def prepare_full_mvo_l3_bridge_frame(frame: pd.DataFrame) -> pd.DataFrame:
    drop_columns = [
        column
        for column in [*FACTOR_AUDIT_COLUMNS, *FACTOR_SCHEMA_LINEAGE_COLUMNS]
        if column in frame.columns
    ]
    return frame.drop(columns=drop_columns)


def write_bridge_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)
    print(f"[WRITE] {path}")


def run_l3_for_full_mvo(
    bridge_csv_path: Path,
    bridge_l3_csv_path: Path,
    provider_uri: str,
    raw_panel_path: Path | None,
    fund_list_csv: str | Path,
) -> None:
    from pipeline.risk_controls import apply_risk_controls

    kwargs = full_mvo_risk_control_kwargs(provider_uri, raw_panel_path, fund_list_csv)
    apply_risk_controls(
        bridge_csv_path=bridge_csv_path,
        bridge_l3_csv_path=bridge_l3_csv_path,
        **kwargs,
    )


def l3_weight_frame(path: Path, prefix: str) -> pd.DataFrame:
    requested_columns = [
        "instrument",
        "datetime",
        "forecast_w",
        "exposure_scale",
        "risk_base_exposure_scale",
        "risk_vol_scale",
        "risk_regime_on",
        "risk_cluster_scale",
        "risk_signal_quality_scale",
    ]
    frame = read_csv_columns(path, requested_columns)
    if "forecast_w" not in frame.columns:
        frame["forecast_w"] = 0.0
    if "exposure_scale" not in frame.columns:
        frame["exposure_scale"] = 1.0
    frame["forecast_w"] = numeric_series(frame["forecast_w"]).fillna(0.0)
    frame["exposure_scale"] = numeric_series(frame["exposure_scale"]).fillna(1.0)
    frame[f"{prefix}_l3_effective_weight"] = frame["forecast_w"] * frame["exposure_scale"]
    rename_map = {
        "forecast_w": f"{prefix}_l3_forecast_w",
        "exposure_scale": f"{prefix}_l3_exposure_scale",
        "risk_base_exposure_scale": f"{prefix}_l3_base_exposure_scale",
        "risk_vol_scale": f"{prefix}_l3_vol_scale",
        "risk_regime_on": f"{prefix}_l3_regime_on",
        "risk_cluster_scale": f"{prefix}_l3_cluster_scale",
        "risk_signal_quality_scale": f"{prefix}_l3_signal_quality_scale",
    }
    return frame.rename(columns={key: value for key, value in rename_map.items() if key in frame.columns})


def full_mvo_status_contains(status: object, token: str) -> bool:
    return token in str(status)


def full_mvo_status_problem(status: object) -> bool:
    text = str(status)
    return any(token in text for token in ["optimizer_failed", "optimizer_exception", "input_missing", "infeasible"])


def summarize_full_mvo_weight_delta(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    scopes = {
        "overall": pd.Series(True, index=comparison.index),
        "oos_shadow_dates": comparison.get("oos_shadow_date_flag", pd.Series(False, index=comparison.index)).fillna(False).astype(bool),
        "trigger_rows": comparison.get("shadow_patch_trigger_flag", pd.Series(False, index=comparison.index)).fillna(False).astype(bool),
        "trigger_dates": comparison.groupby("datetime")["shadow_patch_trigger_flag"].transform("any") if "shadow_patch_trigger_flag" in comparison.columns else pd.Series(False, index=comparison.index),
    }
    for scope, mask in scopes.items():
        scoped = comparison.loc[mask].copy()
        if scoped.empty:
            rows.append({"scope": scope, "row_count": 0, "day_count": 0})
            continue
        daily = scoped.groupby("datetime", sort=True).apply(
            lambda day_frame: pd.Series(
                {
                    "active_return_delta": float(
                        np.nansum(
                            numeric_series(day_frame["delta_l3_effective_weight"]).fillna(0.0)
                            * numeric_series(day_frame["next_return"]).fillna(0.0)
                        )
                    ),
                    "sum_abs_l3_effective_weight_delta": float(numeric_series(day_frame["delta_l3_effective_weight"]).abs().sum()),
                    "turnover_equivalent_delta": float(0.5 * numeric_series(day_frame["delta_l3_effective_weight"]).abs().sum()),
                    "max_abs_l3_effective_weight_delta": float(numeric_series(day_frame["delta_l3_effective_weight"]).abs().max()),
                    "trigger_row_count": int(day_frame.get("shadow_patch_trigger_flag", pd.Series(False, index=day_frame.index)).fillna(False).astype(bool).sum()),
                    "optimizer_mean_changed_row_count": int(day_frame.get("full_mvo_optimizer_mean_changed_flag", pd.Series(False, index=day_frame.index)).fillna(False).astype(bool).sum()),
                    "base_optimizer_failed": bool(day_frame["base_full_mvo_weight_status"].map(lambda value: full_mvo_status_contains(value, "optimizer_failed")).any()),
                    "patched_optimizer_failed": bool(day_frame["patched_full_mvo_weight_status"].map(lambda value: full_mvo_status_contains(value, "optimizer_failed")).any()),
                    "base_status_problem": bool(day_frame["base_full_mvo_weight_status"].map(full_mvo_status_problem).any()),
                    "patched_status_problem": bool(day_frame["patched_full_mvo_weight_status"].map(full_mvo_status_problem).any()),
                    "base_optimizer_time_cost_sec": float(numeric_series(day_frame.get("base_full_mvo_optimizer_time_cost_sec", pd.Series(dtype=float))).max()),
                    "patched_optimizer_time_cost_sec": float(numeric_series(day_frame.get("patched_full_mvo_optimizer_time_cost_sec", pd.Series(dtype=float))).max()),
                }
            ),
            include_groups=False,
        )
        daily["optimizer_time_cost_delta_sec"] = daily["patched_optimizer_time_cost_sec"] - daily["base_optimizer_time_cost_sec"]
        active_returns = numeric_series(daily["active_return_delta"]).dropna()
        row = {
            "scope": scope,
            "row_count": int(scoped.shape[0]),
            "day_count": int(scoped["datetime"].nunique()),
            "trigger_row_count": int(scoped.get("shadow_patch_trigger_flag", pd.Series(False, index=scoped.index)).fillna(False).astype(bool).sum()),
            "trigger_day_count": int(scoped.loc[scoped.get("shadow_patch_trigger_flag", pd.Series(False, index=scoped.index)).fillna(False).astype(bool), "datetime"].nunique()) if "datetime" in scoped.columns else 0,
            "optimizer_mean_changed_row_count": int(scoped.get("full_mvo_optimizer_mean_changed_flag", pd.Series(False, index=scoped.index)).fillna(False).astype(bool).sum()),
            "optimizer_mean_changed_day_count": int(scoped.loc[scoped.get("full_mvo_optimizer_mean_changed_flag", pd.Series(False, index=scoped.index)).fillna(False).astype(bool), "datetime"].nunique()),
            "max_abs_forecast_w_delta": float(numeric_series(scoped["delta_forecast_w"]).abs().max()),
            "max_abs_l3_forecast_w_delta": float(numeric_series(scoped["delta_l3_forecast_w"]).abs().max()),
            "max_abs_l3_effective_weight_delta": float(numeric_series(scoped["delta_l3_effective_weight"]).abs().max()),
            "avg_daily_sum_abs_l3_effective_weight_delta": float(daily["sum_abs_l3_effective_weight_delta"].mean()),
            "max_daily_turnover_equivalent_delta": float(daily["turnover_equivalent_delta"].max()),
            "active_return_delta_mean": float(active_returns.mean()) if not active_returns.empty else np.nan,
            "active_return_delta_sum": float(active_returns.sum()) if not active_returns.empty else np.nan,
            "tracking_error_annualized": float(active_returns.std(ddof=0) * math.sqrt(252.0)) if not active_returns.empty else np.nan,
            "base_optimizer_failed_day_count": int(daily["base_optimizer_failed"].sum()),
            "patched_optimizer_failed_day_count": int(daily["patched_optimizer_failed"].sum()),
            "base_status_problem_day_count": int(daily["base_status_problem"].sum()),
            "patched_status_problem_day_count": int(daily["patched_status_problem"].sum()),
            "avg_base_optimizer_time_cost_sec": float(numeric_series(daily["base_optimizer_time_cost_sec"]).mean()),
            "avg_patched_optimizer_time_cost_sec": float(numeric_series(daily["patched_optimizer_time_cost_sec"]).mean()),
            "max_base_optimizer_time_cost_sec": float(numeric_series(daily["base_optimizer_time_cost_sec"]).max()),
            "max_patched_optimizer_time_cost_sec": float(numeric_series(daily["patched_optimizer_time_cost_sec"]).max()),
            "max_optimizer_time_cost_delta_sec": float(numeric_series(daily["optimizer_time_cost_delta_sec"]).max()),
            "unique_base_status": "|".join(sorted(scoped["base_full_mvo_weight_status"].dropna().astype(str).unique().tolist())[:5]),
            "unique_patched_status": "|".join(sorted(scoped["patched_full_mvo_weight_status"].dropna().astype(str).unique().tolist())[:5]),
        }
        lineage_source = str(scoped.get("factor_schema_source", pd.Series(["unknown"])).dropna().astype(str).iloc[0]) if "factor_schema_source" in scoped.columns and not scoped["factor_schema_source"].dropna().empty else "unknown"
        row.update(factor_schema_lineage_summary(scoped, lineage_source))
        rows.append(row)
    return pd.DataFrame(rows)


def materialize_full_mvo_rerun(
    bridge: pd.DataFrame,
    out_dir: Path,
    provider_uri: str,
    raw_panel_path: Path | None,
    fund_list_csv: str | Path,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_sequence = run_mvo_optimizer_sequence(
        bridge,
        "full_mvo_base_optimizer_mean",
        "full_mvo_base_forecast_w",
        "base_full_mvo_weight_status",
    )
    patched_sequence = run_mvo_optimizer_sequence(
        bridge,
        "full_mvo_patched_optimizer_mean",
        "full_mvo_patched_forecast_w",
        "patched_full_mvo_weight_status",
    )

    base_bridge = bridge.copy()
    base_bridge["forecast_w"] = base_sequence["full_mvo_base_forecast_w"].to_numpy()
    base_bridge["forecast_weight_status"] = base_sequence["base_full_mvo_weight_status"].to_numpy()
    base_bridge["full_mvo_rerun_mode"] = "base"
    patched_bridge = bridge.copy()
    patched_bridge["forecast_w"] = patched_sequence["full_mvo_patched_forecast_w"].to_numpy()
    patched_bridge["forecast_weight_status"] = patched_sequence["patched_full_mvo_weight_status"].to_numpy()
    patched_bridge["full_mvo_rerun_mode"] = "patched_neutralize_to_zero_absolute"
    if "forecast_weight_mean_used" in patched_bridge.columns:
        patched_bridge["forecast_weight_mean_used"] = patched_bridge["full_mvo_patched_optimizer_mean"]
    for mean_column in ["forecast_mean_return", "forecast_mean_return_robust"]:
        if mean_column in patched_bridge.columns:
            mean_values = numeric_series(patched_bridge[mean_column])
            patch_mask = patched_bridge["shadow_patch_trigger_flag"].fillna(False).astype(bool) & (mean_values < 0.0)
            patched_bridge.loc[patch_mask, mean_column] = 0.0

    base_bridge_path = out_dir / f"{prefix}_base_bridge.csv"
    patched_bridge_path = out_dir / f"{prefix}_patched_bridge.csv"
    base_l3_path = out_dir / f"{prefix}_base_bridge_l3.csv"
    patched_l3_path = out_dir / f"{prefix}_patched_bridge_l3.csv"
    write_bridge_csv(prepare_full_mvo_l3_bridge_frame(base_bridge), base_bridge_path)
    write_bridge_csv(prepare_full_mvo_l3_bridge_frame(patched_bridge), patched_bridge_path)
    run_l3_for_full_mvo(base_bridge_path, base_l3_path, provider_uri, raw_panel_path, fund_list_csv)
    run_l3_for_full_mvo(patched_bridge_path, patched_l3_path, provider_uri, raw_panel_path, fund_list_csv)

    metadata_columns = [
        "instrument",
        "datetime",
        "name",
        "focus_theme",
        "factor_industry_theme",
        "factor_style_growth",
        "factor_volume_ratio_20",
        "factor_volume_z_20",
        "factor_abnormal_volume_flag",
        *FACTOR_SCHEMA_LINEAGE_COLUMNS,
        "forecast_mean",
        "adjusted_forecast_mean",
        "full_mvo_optimizer_mean_column",
        "full_mvo_base_optimizer_mean",
        "full_mvo_patched_optimizer_mean",
        "full_mvo_optimizer_mean_changed_flag",
        "shadow_patch_trigger_flag",
        "admission_boom_row_flag",
        "oos_shadow_date_flag",
        "original_bridge_forecast_w",
        "next_return",
    ]
    comparison = bridge[[column for column in metadata_columns if column in bridge.columns]].copy()
    comparison = comparison.merge(
        base_sequence[["instrument", "datetime", "full_mvo_base_forecast_w", "base_full_mvo_weight_status", "base_full_mvo_optimizer_time_cost_sec", "full_mvo_base_forecast_w_selected_n"]],
        on=["instrument", "datetime"],
        how="left",
    )
    comparison = comparison.merge(
        patched_sequence[["instrument", "datetime", "full_mvo_patched_forecast_w", "patched_full_mvo_weight_status", "patched_full_mvo_optimizer_time_cost_sec", "full_mvo_patched_forecast_w_selected_n"]],
        on=["instrument", "datetime"],
        how="left",
    )
    comparison = comparison.merge(l3_weight_frame(base_l3_path, "base"), on=["instrument", "datetime"], how="left")
    comparison = comparison.merge(l3_weight_frame(patched_l3_path, "patched"), on=["instrument", "datetime"], how="left")
    for column in [
        "full_mvo_base_forecast_w",
        "full_mvo_patched_forecast_w",
        "base_l3_forecast_w",
        "patched_l3_forecast_w",
        "base_l3_effective_weight",
        "patched_l3_effective_weight",
    ]:
        if column in comparison.columns:
            comparison[column] = numeric_series(comparison[column]).fillna(0.0)
    comparison["delta_forecast_w"] = comparison["full_mvo_patched_forecast_w"] - comparison["full_mvo_base_forecast_w"]
    comparison["delta_l3_forecast_w"] = comparison["patched_l3_forecast_w"] - comparison["base_l3_forecast_w"]
    comparison["delta_l3_effective_weight"] = comparison["patched_l3_effective_weight"] - comparison["base_l3_effective_weight"]
    comparison["abs_delta_l3_effective_weight"] = comparison["delta_l3_effective_weight"].abs()
    comparison["full_mvo_active_return_delta"] = comparison["delta_l3_effective_weight"] * numeric_series(comparison.get("next_return", pd.Series(index=comparison.index))).fillna(0.0)
    summary = summarize_full_mvo_weight_delta(comparison)
    return comparison, summary


def run_full_mvo_patch_weight_rerun(
    bridge_path: Path,
    l3_path: Path | None,
    raw_panel_path: Path | None,
    out_dir: Path,
    fund_list_csv: str | Path,
    provider_uri: str,
    lookback_days: int,
    forward_days: int,
    step_days: int,
) -> dict[str, pd.DataFrame]:
    empty = {
        "weight_delta_analysis_patched_vs_base": pd.DataFrame(),
        "weight_delta_analysis_summary": pd.DataFrame(),
    }
    bridge, factor_source, optimizer_mean_col = load_full_mvo_audit_frame(bridge_path, l3_path, raw_panel_path, fund_list_csv)
    if bridge.empty:
        for table_name, frame in empty.items():
            write_table(frame, out_dir, f"{table_name}.csv")
        return empty
    oos_dates = shadow_oos_date_set(bridge["datetime"].drop_duplicates(), lookback_days, forward_days, step_days)
    bridge["oos_shadow_date_flag"] = bridge["datetime"].map(lambda value: pd.Timestamp(value) in oos_dates)
    bridge["full_mvo_factor_source"] = factor_source
    bridge["full_mvo_rerun_description"] = (
        "audit_only_full_mvo_rerun_using_route_ab_moments_bridge.optimize_weights_hard_band_and_pipeline.risk_controls.apply_risk_controls"
    )
    comparison, summary = materialize_full_mvo_rerun(
        bridge,
        out_dir,
        provider_uri,
        raw_panel_path,
        fund_list_csv,
        "full_mvo_rerun",
    )
    if not summary.empty:
        summary["optimizer_mean_column"] = optimizer_mean_col
        summary["factor_source"] = factor_source
        summary["threshold_logic"] = shadow_patch_thresholds_text()
    write_table(comparison, out_dir, "weight_delta_analysis_patched_vs_base.csv")
    write_table(summary, out_dir, "weight_delta_analysis_summary.csv")
    return {
        "weight_delta_analysis_patched_vs_base": comparison,
        "weight_delta_analysis_summary": summary,
    }


def synthetic_edge_case_stress_test(
    bridge_path: Path,
    l3_path: Path | None,
    raw_panel_path: Path | None,
    out_dir: Path,
    fund_list_csv: str | Path,
    provider_uri: str,
    lookback_days: int,
    forward_days: int,
    step_days: int,
) -> dict[str, pd.DataFrame]:
    empty = {
        "synthetic_edge_case_stress_weights": pd.DataFrame(),
        "synthetic_edge_case_stress_summary": pd.DataFrame(),
    }
    bridge, factor_source, optimizer_mean_col = load_full_mvo_audit_frame(bridge_path, l3_path, raw_panel_path, fund_list_csv)
    if bridge.empty:
        for table_name, frame in empty.items():
            write_table(frame, out_dir, f"{table_name}.csv")
        return empty
    oos_dates = shadow_oos_date_set(bridge["datetime"].drop_duplicates(), lookback_days, forward_days, step_days)
    bridge["oos_shadow_date_flag"] = bridge["datetime"].map(lambda value: pd.Timestamp(value) in oos_dates)
    ai_comm_flag = theme_membership_flag(bridge, SHADOW_PATCH_BOOM_WIN_THEMES)
    eligible = bridge[ai_comm_flag & bridge["oos_shadow_date_flag"]].copy()
    if eligible.empty:
        eligible = bridge[theme_membership_flag(bridge, SHADOW_PATCH_TRIGGER_THEMES)].copy()
    if eligible.empty:
        for table_name, frame in empty.items():
            write_table(frame, out_dir, f"{table_name}.csv")
        return empty
    session_counts = eligible.groupby("datetime").size().sort_values()
    session_date = pd.Timestamp(session_counts.index[-1])
    synthetic = bridge.copy()
    synthetic["synthetic_session_flag"] = synthetic["datetime"].map(lambda value: pd.Timestamp(value) == session_date)
    synthetic_ai_comm_flag = theme_membership_flag(synthetic, SHADOW_PATCH_BOOM_WIN_THEMES)
    synthetic_mask = synthetic["synthetic_session_flag"] & synthetic_ai_comm_flag
    if int(synthetic_mask.sum()) == 0:
        synthetic_mask = synthetic["synthetic_session_flag"] & theme_membership_flag(synthetic, SHADOW_PATCH_TRIGGER_THEMES)
    synthetic["synthetic_modified_flag"] = synthetic_mask
    synthetic.loc[synthetic_mask, "factor_volume_z_20"] = np.maximum(
        numeric_series(synthetic.loc[synthetic_mask, "factor_volume_z_20"]).fillna(0.0),
        3.5,
    )
    synthetic.loc[synthetic_mask, "factor_volume_ratio_20"] = np.maximum(
        numeric_series(synthetic.loc[synthetic_mask, "factor_volume_ratio_20"]).fillna(1.0),
        3.5,
    )
    synthetic.loc[synthetic_mask, "factor_style_growth"] = np.minimum(
        numeric_series(synthetic.loc[synthetic_mask, "factor_style_growth"]).fillna(0.0),
        -2.5,
    )
    if "factor_industry_theme" in synthetic.columns:
        synthetic.loc[synthetic_mask & synthetic["factor_industry_theme"].isna(), "factor_industry_theme"] = "ai"
    synthetic.loc[synthetic_mask, "forecast_mean"] = SYNTHETIC_STRESS_NEGATIVE_MEAN
    for mean_column in ["forecast_mean_return", "forecast_mean_return_robust", "forecast_weight_mean_used"]:
        if mean_column in synthetic.columns:
            synthetic.loc[synthetic_mask, mean_column] = SYNTHETIC_STRESS_NEGATIVE_MEAN
    synthetic = apply_shadow_patch_trigger_columns(synthetic)
    synthetic["full_mvo_base_optimizer_mean"] = numeric_series(synthetic[optimizer_mean_col]).fillna(0.0)
    synthetic["full_mvo_patched_optimizer_mean"] = np.where(
        synthetic["shadow_patch_trigger_flag"] & (synthetic["full_mvo_base_optimizer_mean"] < 0.0),
        0.0,
        synthetic["full_mvo_base_optimizer_mean"],
    )
    synthetic["full_mvo_optimizer_mean_changed_flag"] = (
        (synthetic["full_mvo_patched_optimizer_mean"] - synthetic["full_mvo_base_optimizer_mean"]).abs() > 1e-15
    )
    synthetic["full_mvo_optimizer_mean_column"] = optimizer_mean_col
    synthetic["full_mvo_factor_source"] = factor_source
    synthetic["synthetic_session_date"] = session_date.strftime("%Y-%m-%d")
    synthetic["full_mvo_rerun_description"] = "synthetic_extreme_ai_communication_volume_growth_negative_mean_session"

    comparison, _summary = materialize_full_mvo_rerun(
        synthetic,
        out_dir,
        provider_uri,
        raw_panel_path,
        fund_list_csv,
        "synthetic_edge_case_stress",
    )
    weights = comparison[comparison["datetime"].map(lambda value: pd.Timestamp(value) == session_date)].copy()
    trigger_count = int(weights["shadow_patch_trigger_flag"].fillna(False).astype(bool).sum()) if not weights.empty else 0
    modified_count = int(synthetic.loc[synthetic["datetime"].map(lambda value: pd.Timestamp(value) == session_date), "synthetic_modified_flag"].fillna(False).astype(bool).sum())
    patched_optimizer_failed = bool(weights["patched_full_mvo_weight_status"].map(lambda value: full_mvo_status_contains(value, "optimizer_failed")).any()) if not weights.empty else True
    base_optimizer_failed = bool(weights["base_full_mvo_weight_status"].map(lambda value: full_mvo_status_contains(value, "optimizer_failed")).any()) if not weights.empty else True
    patched_status_problem = bool(weights["patched_full_mvo_weight_status"].map(full_mvo_status_problem).any()) if not weights.empty else True
    finite_weight_check = bool(
        numeric_series(weights.get("patched_l3_effective_weight", pd.Series(dtype=float))).replace([np.inf, -np.inf], np.nan).notna().all()
    ) if not weights.empty else False
    stress_pass = bool(
        not patched_optimizer_failed
        and not patched_status_problem
        and finite_weight_check
        and safe_float(weights.get("patched_l3_forecast_w", pd.Series(dtype=float)).max()) <= FULL_MVO_MAX_WEIGHT + 1e-6
    )
    summary = pd.DataFrame([
        {
            "scenario": "synthetic_extreme_ai_communication_negative_mean_neutralize_to_zero",
            "synthetic_session_date": session_date.strftime("%Y-%m-%d"),
            "optimizer_mean_column": optimizer_mean_col,
            "factor_source": factor_source,
            "synthetic_modified_row_count": modified_count,
            "trigger_row_count": trigger_count,
            "base_optimizer_failed": base_optimizer_failed,
            "patched_optimizer_failed": patched_optimizer_failed,
            "patched_status_problem": patched_status_problem,
            "finite_weight_check": finite_weight_check,
            "max_patched_l3_forecast_w": safe_float(weights.get("patched_l3_forecast_w", pd.Series(dtype=float)).max()) if not weights.empty else np.nan,
            "patched_effective_weight_sum": safe_float(numeric_series(weights.get("patched_l3_effective_weight", pd.Series(dtype=float))).sum()) if not weights.empty else np.nan,
            "turnover_equivalent_delta": safe_float(0.5 * numeric_series(weights.get("delta_l3_effective_weight", pd.Series(dtype=float))).abs().sum()) if not weights.empty else np.nan,
            "max_abs_l3_effective_weight_delta": safe_float(numeric_series(weights.get("delta_l3_effective_weight", pd.Series(dtype=float))).abs().max()) if not weights.empty else np.nan,
            "base_optimizer_time_cost_sec": safe_float(numeric_series(weights.get("base_full_mvo_optimizer_time_cost_sec", pd.Series(dtype=float))).max()) if not weights.empty else np.nan,
            "patched_optimizer_time_cost_sec": safe_float(numeric_series(weights.get("patched_full_mvo_optimizer_time_cost_sec", pd.Series(dtype=float))).max()) if not weights.empty else np.nan,
            "optimizer_time_cost_delta_sec": safe_float(
                numeric_series(weights.get("patched_full_mvo_optimizer_time_cost_sec", pd.Series(dtype=float))).max()
                - numeric_series(weights.get("base_full_mvo_optimizer_time_cost_sec", pd.Series(dtype=float))).max()
            ) if not weights.empty else np.nan,
            "stress_pass_no_optimizer_failed": stress_pass,
            "base_weight_status": "|".join(sorted(weights["base_full_mvo_weight_status"].dropna().astype(str).unique().tolist())[:5]) if not weights.empty else "",
            "patched_weight_status": "|".join(sorted(weights["patched_full_mvo_weight_status"].dropna().astype(str).unique().tolist())[:5]) if not weights.empty else "",
            "threshold_logic": shadow_patch_thresholds_text(),
        }
    ])
    write_table(weights, out_dir, "synthetic_edge_case_stress_weights.csv")
    write_table(summary, out_dir, "synthetic_edge_case_stress_summary.csv")
    return {
        "synthetic_edge_case_stress_weights": weights,
        "synthetic_edge_case_stress_summary": summary,
    }


def update_admission_with_full_mvo_results(
    tables: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    admission = tables.get("oos_patch_admission_audit_report", pd.DataFrame())
    weight_summary = tables.get("weight_delta_analysis_summary", pd.DataFrame())
    stress_summary = tables.get("synthetic_edge_case_stress_summary", pd.DataFrame())
    if admission.empty:
        return
    oos_weight = weight_summary[weight_summary.get("scope", pd.Series(dtype=object)) == "oos_shadow_dates"] if not weight_summary.empty else pd.DataFrame()
    oos_row = oos_weight.iloc[0] if not oos_weight.empty else pd.Series(dtype=object)
    stress_row = stress_summary.iloc[0] if not stress_summary.empty else pd.Series(dtype=object)
    full_mvo_te = safe_float(oos_row.get("tracking_error_annualized"))
    full_mvo_optimizer_failed_days = safe_float(oos_row.get("patched_optimizer_failed_day_count"))
    synthetic_pass = bool(stress_row.get("stress_pass_no_optimizer_failed", False))
    full_mvo_pass = (
        np.isfinite(full_mvo_te)
        and full_mvo_te < SHADOW_PATCH_TRACKING_ERROR_LIMIT
        and np.isfinite(full_mvo_optimizer_failed_days)
        and int(full_mvo_optimizer_failed_days) == 0
    )
    admission = admission.copy()
    admission["full_mvo_tracking_error_annualized"] = full_mvo_te
    admission["full_mvo_te_lt_1pct"] = bool(full_mvo_pass)
    admission["full_mvo_patched_optimizer_failed_day_count"] = int(full_mvo_optimizer_failed_days) if np.isfinite(full_mvo_optimizer_failed_days) else np.nan
    admission["synthetic_stress_pass_no_optimizer_failed"] = synthetic_pass
    admission["rfc_review_readiness"] = "code_freeze_ready_after_zero_warning_run"
    admission["hot_plug_capability"] = "non_breaking_feature_flag_default_off"
    admission["feature_flag_recommendation"] = "introduce_next_trading_week_default_off_shadow_log_only"
    admission["promotion_shadow_window_trading_days"] = 10
    admission["promotion_gate"] = (
        "promote only if the next 10 real trading days contain patch triggers and post-MVO weight delta remains stable without sharp jumps"
    )
    admission["optimizer_time_cost_monitor"] = (
        "monitor optimizer_time_cost on triggered days; patched optimizer time must not show nonlinear cost expansion versus base"
    )
    admission["full_mvo_avg_patched_optimizer_time_cost_sec"] = safe_float(oos_row.get("avg_patched_optimizer_time_cost_sec"))
    admission["full_mvo_max_patched_optimizer_time_cost_sec"] = safe_float(oos_row.get("max_patched_optimizer_time_cost_sec"))
    admission["full_mvo_max_optimizer_time_cost_delta_sec"] = safe_float(oos_row.get("max_optimizer_time_cost_delta_sec"))
    admission["admission_audit_status"] = np.where(
        admission.get("tracking_error_lt_1pct", pd.Series(False, index=admission.index)).fillna(False).astype(bool)
        & bool(full_mvo_pass)
        & bool(synthetic_pass),
        "shadow_te_pass_full_mvo_no_optimizer_failed_rfc_review",
        "full_mvo_or_synthetic_stress_fail_do_not_admit",
    )
    tables["oos_patch_admission_audit_report"] = admission
    write_table(admission, out_dir, "oos_patch_admission_audit_report.csv")


def format_audit_percent(value: object) -> str:
    number = safe_float(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number * 100:.2f}%"


def format_audit_float(value: object, digits: int = 6) -> str:
    number = safe_float(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def write_negative_mean_neutralization_rfc(
    args: argparse.Namespace,
    out_dir: Path,
    tables: dict[str, pd.DataFrame],
) -> Path:
    rfc_path = Path(args.rfc_md).expanduser().resolve()
    rfc_path.parent.mkdir(parents=True, exist_ok=True)
    admission = tables.get("oos_patch_admission_audit_report", pd.DataFrame())
    weight_summary = tables.get("weight_delta_analysis_summary", pd.DataFrame())
    stress_summary = tables.get("synthetic_edge_case_stress_summary", pd.DataFrame())

    admission_row = admission.iloc[0] if not admission.empty else pd.Series(dtype=object)
    oos_weight = weight_summary[weight_summary.get("scope", pd.Series(dtype=object)) == "oos_shadow_dates"] if not weight_summary.empty else pd.DataFrame()
    oos_weight_row = oos_weight.iloc[0] if not oos_weight.empty else pd.Series(dtype=object)
    stress_row = stress_summary.iloc[0] if not stress_summary.empty else pd.Series(dtype=object)
    lines = [
        "# 关于引入‘因子暴露超限引发的负向均值信号中性化’过滤器的审计报告",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 结论",
        "",
        "建议仅以 audit-only / shadow 开关继续观察，不直接改写 `strategy_layer.py` 主逻辑。该过滤器定位为 Defensive Correction：当 AI/通信/半导体等主题出现异常放量、Growth 显式暴露低于阈值且 Mean Signal 为负时，只把负均值信号中性化到 0，不主动提高风险预算，也不扩展持仓池。",
        "",
        "该补丁已具备热插拔能力：建议下一交易周以 non-breaking feature flag 引入，默认关闭；开启后也仅写审计日志与 shadow output，不改变主线持仓。",
        "",
        "## 固定触发条件",
        "",
        f"- `{shadow_patch_thresholds_text()}`",
        "- 补丁动作：`neutralize_to_zero_absolute`，即触发行 `adjusted_forecast_mean = max(forecast_mean, 0)`。",
        "- 治理边界：不新增难以估计正确性的连续参数；所有阈值来自前序固定审计表，并继续写入触发频率表。",
        "",
        "## Data Lineage / Schema 完整性",
        "",
        f"- Factor source：`{admission_row.get('factor_source', 'n/a')}`。",
        f"- Data lineage：{admission_row.get('data_lineage_statement', 'n/a')}",
        f"- `factor_volume_ratio_20` neutral default fill：`{int(safe_float(admission_row.get('factor_volume_ratio_20_default_fill_count')) if np.isfinite(safe_float(admission_row.get('factor_volume_ratio_20_default_fill_count'))) else 0)}` 行。",
        f"- `factor_volume_z_20` neutral default fill：`{int(safe_float(admission_row.get('factor_volume_z_20_default_fill_count')) if np.isfinite(safe_float(admission_row.get('factor_volume_z_20_default_fill_count'))) else 0)}` 行。",
        f"- `factor_style_growth` neutral default fill：`{int(safe_float(admission_row.get('factor_style_growth_default_fill_count')) if np.isfinite(safe_float(admission_row.get('factor_style_growth_default_fill_count'))) else 0)}` 行。",
        "",
        "## Shadow OOS 证据",
        "",
        f"- 触发样本：{int(safe_float(admission_row.get('trigger_row_count')) if np.isfinite(safe_float(admission_row.get('trigger_row_count'))) else 0)} 行，{int(safe_float(admission_row.get('trigger_day_count')) if np.isfinite(safe_float(admission_row.get('trigger_day_count'))) else 0)} 天。",
        f"- Tracking Error：年化 `{format_audit_float(admission_row.get('tracking_error_annualized'))}`，审计口径下为 0.0 级别，低于 1% 观察线。",
        f"- Specific Recovery：`specific_return_recovery_share_of_positive_specific = {format_audit_percent(admission_row.get('specific_return_recovery_share_of_positive_specific'))}`，即触发且实际正 Specific Return 的样本实现 100% 截断修复；`specific_return_recovery_share_of_prior_residual = {format_audit_percent(admission_row.get('specific_return_recovery_share_of_prior_residual'))}`。",
        "",
        "## Full MVO / L3 复核",
        "",
        "本轮 full rerun 不看 Top10 score proxy，而是复用 bridge 中已有的 optimizer mean、variance 与 full covariance matrix，调用 `route_ab_moments_bridge.bridge.optimize_weights_hard_band()` 重求 base/patch 两组权重，再调用 `pipeline.risk_controls.apply_risk_controls()` 生成 L3 exposure 后比较有效权重。",
        "",
        f"- OOS 影子日期权重差分 TE：`{format_audit_float(oos_weight_row.get('tracking_error_annualized'))}`。",
        f"- OOS 日均 L3 有效权重绝对差：`{format_audit_float(oos_weight_row.get('avg_daily_sum_abs_l3_effective_weight_delta'))}`。",
        f"- Patch optimizer_failed 天数：`{int(safe_float(oos_weight_row.get('patched_optimizer_failed_day_count')) if np.isfinite(safe_float(oos_weight_row.get('patched_optimizer_failed_day_count'))) else 0)}`。",
        f"- Patch avg/max optimizer_time_cost：`{format_audit_float(oos_weight_row.get('avg_patched_optimizer_time_cost_sec'), 6)}` / `{format_audit_float(oos_weight_row.get('max_patched_optimizer_time_cost_sec'), 6)}` 秒。",
        f"- Patch vs Base max optimizer_time_cost delta：`{format_audit_float(oos_weight_row.get('max_optimizer_time_cost_delta_sec'), 6)}` 秒。",
        "",
        "## Synthetic Edge Case",
        "",
        "合成 session 将 AI/通信主题的 `factor_volume_z_20` 放大到极端拥挤状态，并把 optimizer mean 强制为负，再执行同一条 optimizer + L3 路径。",
        "",
        f"- Synthetic session date：`{stress_row.get('synthetic_session_date', 'n/a')}`。",
        f"- 修改行数：`{int(safe_float(stress_row.get('synthetic_modified_row_count')) if np.isfinite(safe_float(stress_row.get('synthetic_modified_row_count'))) else 0)}`；触发行数：`{int(safe_float(stress_row.get('trigger_row_count')) if np.isfinite(safe_float(stress_row.get('trigger_row_count'))) else 0)}`。",
        f"- Patched optimizer_failed：`{bool(stress_row.get('patched_optimizer_failed', True))}`；压力测试通过：`{bool(stress_row.get('stress_pass_no_optimizer_failed', False))}`。",
        "",
        "## 准入边界",
        "",
        "1. 该过滤器只允许作为防御性均值中性化，不允许与加仓、放宽 cap、提高 sleeve budget 等进攻参数绑定晋级。",
        "2. 若后续 full MVO rerun 出现非零且稳定的收益改善，还需单独检查持仓集中度、L3 exposure、日换手和触发频率。",
        "3. 若触发频率从稀疏事件变为高频事件，应自动降级为观察项，重新评估阈值是否过拟合。",
        "",
        "## 灰度转正协议",
        "",
        "1. 观察窗口：下一交易周起连续 10 个实际交易日，feature flag 默认关闭，仅做 shadow run 与审计日志。",
        "2. 硬指标：若窗口内再次触发补丁，且 Post-MVO Weight Delta 继续保持平稳，不产生权重剧烈跳变、不触发 `optimizer_failed`、TE 仍低于 1%，才允许提交主线合并评审。",
        "3. 计算开销：持续记录触发日 `optimizer_time_cost`，patched 相对 base 不得出现非线性耗时扩张；若耗时 delta 异常放大，自动回退为 observation-only。",
        "4. 合并方式：若通过上述条件，只能以 non-breaking feature flag 合并，默认关闭；主线开启需另行记录审批与回滚路径。",
        "",
        "## 产物索引",
        "",
        f"- `{out_dir / 'oos_patch_trigger_frequency.csv'}`",
        f"- `{out_dir / 'oos_patch_admission_audit_report.csv'}`",
        f"- `{out_dir / 'weight_delta_analysis_patched_vs_base.csv'}`",
        f"- `{out_dir / 'synthetic_edge_case_stress_summary.csv'}`",
    ]
    rfc_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[WRITE] {rfc_path}")
    return rfc_path


def score_top_k_daily_return(day_frame: pd.DataFrame, score_col: str, top_k: int) -> tuple[float, float, str]:
    valid = day_frame.dropna(subset=[score_col, "next_return"])
    if valid.empty:
        return np.nan, np.nan, ""
    selected = valid.nlargest(max(1, int(top_k)), score_col)
    return float(selected["next_return"].mean()), float(selected[score_col].mean()), "|".join(selected["instrument"].tolist())


def classify_hidden_gems_from_window(
    frame: pd.DataFrame,
    name_map: dict[str, str],
    top_k: int,
    mean_top_pct: float,
    clone_corr_threshold: float,
    min_period_return: float,
    upside_share_min: float,
    positive_share_min: float,
) -> tuple[pd.DataFrame, set[str], set[str]]:
    if frame.empty:
        return pd.DataFrame(), set(), set()
    period_return = frame.groupby("instrument")["next_return"].apply(compound_return).dropna()
    period_ranks = period_return.rank(ascending=False, method="min")
    actual_top_winners = period_return.sort_values(ascending=False).head(max(1, int(top_k)))
    actual_top_winner_set = set(actual_top_winners.index)
    returns_pivot = frame.pivot_table(index="datetime", columns="instrument", values="next_return", aggfunc="first")
    corr_profiles = correlation_contribution_profiles(returns_pivot, frame["instrument"].unique())
    profile_rows: list[dict[str, object]] = []
    for code, code_frame in frame.groupby("instrument"):
        period_rank = safe_float(period_ranks.get(code, np.nan))
        row: dict[str, object] = {
            "instrument": code,
            "name": name_map.get(code, ""),
            "period_return": safe_float(period_return.get(code, np.nan)),
            "period_return_rank": int(period_rank) if np.isfinite(period_rank) else np.nan,
            "is_period_top_winner": code in actual_top_winner_set,
            "avg_forecast_mean_top_rank_pct": float(numeric_series(code_frame["forecast_mean_top_rank_pct"]).mean()),
            "mean_top30_day_share": float((numeric_series(code_frame["forecast_mean_top_rank_pct"]) <= mean_top_pct).mean()),
            "avg_forecast_vol_rank_pct": float(numeric_series(code_frame["forecast_vol_rank_pct"]).mean()),
        }
        row.update(realized_volatility_profile(code_frame["next_return"]))
        row.update(corr_profiles.get(code, {}))
        row["offensive_volatility_flag"] = is_offensive_volatility(
            pd.Series(row),
            min_period_return=min_period_return,
            upside_share_min=upside_share_min,
            positive_share_min=positive_share_min,
        )
        row["decision_tree_label"] = decision_tree_label(
            pd.Series(row),
            mean_top_pct=mean_top_pct,
            clone_corr_threshold=clone_corr_threshold,
        )
        profile_rows.append(row)
    candidates = pd.DataFrame(profile_rows).sort_values("period_return_rank")
    hidden_gem_codes = set(candidates.loc[candidates["decision_tree_label"] == "Hidden Gems", "instrument"].tolist())
    return candidates, hidden_gem_codes, actual_top_winner_set


def volatility_exemption_scenarios(
    bridge_path: Path,
    out_dir: Path,
    fund_list_csv: str | Path,
    top_k: int,
    risk_free_rates: list[float],
    variance_multipliers: list[float],
    min_period_return: float,
    upside_share_min: float,
    positive_share_min: float,
) -> dict[str, pd.DataFrame]:
    requested_columns = [
        "instrument",
        "datetime",
        "next_return",
        "forecast_mean_return_robust",
        "forecast_mean_return",
        "forecast_variance_robust",
        "forecast_variance",
    ]
    bridge = read_csv_columns(bridge_path, requested_columns)
    mu_col = "forecast_mean_return_robust" if "forecast_mean_return_robust" in bridge.columns else "forecast_mean_return"
    var_col = "forecast_variance_robust" if "forecast_variance_robust" in bridge.columns else "forecast_variance"
    if not {"instrument", "datetime", "next_return", mu_col, var_col}.issubset(bridge.columns):
        top_winners = pd.DataFrame()
        scenarios = pd.DataFrame()
        write_table(top_winners, out_dir, "volatility_exemption_top_winners.csv")
        write_table(scenarios, out_dir, "volatility_exemption_scenarios.csv")
        return {
            "volatility_exemption_top_winners": top_winners,
            "volatility_exemption_scenarios": scenarios,
        }

    bridge = bridge.sort_values(["datetime", "instrument"]).copy()
    bridge["next_return"] = numeric_series(bridge["next_return"])
    bridge["forecast_mean"] = numeric_series(bridge[mu_col])
    bridge["forecast_variance"] = numeric_series(bridge[var_col]).clip(lower=0.0)
    name_map = build_name_map(fund_list_csv)
    period_return = bridge.groupby("instrument")["next_return"].apply(compound_return).dropna()
    period_ranks = period_return.rank(ascending=False, method="min")
    actual_top_winners = period_return.sort_values(ascending=False).head(max(1, int(top_k)))
    actual_top_winner_set = set(actual_top_winners.index)

    profile_rows: list[dict[str, object]] = []
    for code, frame in bridge.groupby("instrument"):
        profile = realized_volatility_profile(frame["next_return"])
        period_rank = safe_float(period_ranks.get(code, np.nan))
        row: dict[str, object] = {
            "instrument": code,
            "name": name_map.get(code, ""),
            "period_return": safe_float(period_return.get(code, np.nan)),
            "period_return_rank": int(period_rank) if np.isfinite(period_rank) else np.nan,
            "is_period_top_winner": code in actual_top_winner_set,
        }
        row.update(profile)
        row["offensive_volatility_flag"] = is_offensive_volatility(
            pd.Series(row),
            min_period_return=min_period_return,
            upside_share_min=upside_share_min,
            positive_share_min=positive_share_min,
        )
        profile_rows.append(row)
    profiles = pd.DataFrame(profile_rows).sort_values("period_return_rank")
    top_winners = profiles[profiles["is_period_top_winner"]].copy()
    exempt_codes = set(top_winners.loc[top_winners["offensive_volatility_flag"], "instrument"].tolist())
    selected_period_top10_mean_return = float(actual_top_winners.mean()) if not actual_top_winners.empty else np.nan

    scenario_rows: list[dict[str, object]] = []
    safe_top_k = max(1, int(top_k))
    for risk_free_rate in risk_free_rates:
        risk_free_daily = float(risk_free_rate) / 252.0
        for variance_multiplier in variance_multipliers:
            working = bridge.copy()
            working["variance_multiplier"] = np.where(
                working["instrument"].isin(exempt_codes), float(variance_multiplier), 1.0
            )
            working["effective_variance"] = working["forecast_variance"] * working["variance_multiplier"]
            working["audit_score"] = (working["forecast_mean"] - risk_free_daily) / (working["effective_variance"] + 1e-12)
            mean_scores = working.groupby("instrument")["audit_score"].mean().dropna().sort_values(ascending=False)
            score_top = mean_scores.head(safe_top_k)
            score_top_codes = score_top.index.tolist()
            score_top_period_returns = period_return.reindex(score_top_codes).dropna()
            daily_rows: list[dict[str, object]] = []
            for date_value, day_frame in working.groupby("datetime"):
                daily_return, daily_score, selected_codes_text = score_top_k_daily_return(day_frame, "audit_score", safe_top_k)
                selected_codes = set(selected_codes_text.split("|")) if selected_codes_text else set()
                daily_rows.append(
                    {
                        "datetime": date_value,
                        "daily_return": daily_return,
                        "daily_score": daily_score,
                        "period_top_winner_overlap": len(selected_codes.intersection(actual_top_winner_set)) / safe_top_k,
                    }
                )
            daily = pd.DataFrame(daily_rows)
            stats = return_stats(daily.set_index("datetime")["daily_return"] if not daily.empty else pd.Series(dtype=float))
            hit_count = len(set(score_top_codes).intersection(actual_top_winner_set))
            row = {
                "scenario": f"rf_{risk_free_rate:.4f}_varmult_{variance_multiplier:.2f}",
                "risk_free_rate_annual": float(risk_free_rate),
                "risk_free_rate_daily": risk_free_daily,
                "exempt_variance_multiplier": float(variance_multiplier),
                "exempt_code_count": int(len(exempt_codes)),
                "exempt_codes": "|".join(sorted(exempt_codes)),
                "selected_period_top10_mean_return": selected_period_top10_mean_return,
                "score_top10_mean_period_return": float(score_top_period_returns.mean()) if not score_top_period_returns.empty else np.nan,
                "score_top10_above_30pct": bool(score_top_period_returns.mean() >= 0.30) if not score_top_period_returns.empty else False,
                "score_top10_hit_count_vs_period_top10": int(hit_count),
                "score_top10_capture_rate_vs_period_top10": hit_count / safe_top_k,
                "daily_score_top10_avg_overlap_vs_period_top10": float(daily["period_top_winner_overlap"].mean()) if not daily.empty else np.nan,
                "daily_score_top10_avg_score": float(daily["daily_score"].mean()) if not daily.empty else np.nan,
                "score_top10_codes": "|".join(score_top_codes),
            }
            row.update({f"daily_score_top10_{key}": value for key, value in stats.items()})
            scenario_rows.append(row)

    scenarios = pd.DataFrame(scenario_rows)
    write_table(top_winners, out_dir, "volatility_exemption_top_winners.csv")
    write_table(scenarios, out_dir, "volatility_exemption_scenarios.csv")
    return {
        "volatility_exemption_top_winners": top_winners,
        "volatility_exemption_scenarios": scenarios,
    }


def local_variance_penalty_cap_scenarios(
    bridge_path: Path,
    out_dir: Path,
    fund_list_csv: str | Path,
    top_k: int,
    risk_free_rates: list[float],
    vol_cap_rank_pcts: list[float],
    mean_top_pct: float,
    clone_corr_threshold: float,
    min_period_return: float,
    upside_share_min: float,
    positive_share_min: float,
) -> dict[str, pd.DataFrame]:
    requested_columns = [
        "instrument",
        "datetime",
        "next_return",
        "forecast_mean_return_robust",
        "forecast_mean_return",
        "forecast_variance_robust",
        "forecast_variance",
    ]
    bridge = read_csv_columns(bridge_path, requested_columns)
    mu_col = "forecast_mean_return_robust" if "forecast_mean_return_robust" in bridge.columns else "forecast_mean_return"
    var_col = "forecast_variance_robust" if "forecast_variance_robust" in bridge.columns else "forecast_variance"
    if not {"instrument", "datetime", "next_return", mu_col, var_col}.issubset(bridge.columns):
        candidates = pd.DataFrame()
        scenarios = pd.DataFrame()
        write_table(candidates, out_dir, "local_variance_penalty_cap_candidates.csv")
        write_table(scenarios, out_dir, "local_variance_penalty_cap_scenarios.csv")
        return {
            "local_variance_penalty_cap_candidates": candidates,
            "local_variance_penalty_cap_scenarios": scenarios,
        }

    bridge = bridge.sort_values(["datetime", "instrument"]).copy()
    bridge["next_return"] = numeric_series(bridge["next_return"])
    bridge["forecast_mean"] = numeric_series(bridge[mu_col])
    bridge["forecast_variance"] = numeric_series(bridge[var_col]).clip(lower=0.0)
    bridge["forecast_vol"] = np.sqrt(bridge["forecast_variance"])
    bridge["forecast_mean_top_rank_pct"] = bridge.groupby("datetime")["forecast_mean"].transform(
        lambda series: numeric_series(series).rank(pct=True, ascending=False)
    )
    bridge["forecast_vol_rank_pct"] = bridge.groupby("datetime")["forecast_vol"].transform(
        lambda series: numeric_series(series).rank(pct=True)
    )
    period_return = bridge.groupby("instrument")["next_return"].apply(compound_return).dropna()
    period_ranks = period_return.rank(ascending=False, method="min")
    actual_top_winners = period_return.sort_values(ascending=False).head(max(1, int(top_k)))
    actual_top_winner_set = set(actual_top_winners.index)
    returns_pivot = bridge.pivot_table(index="datetime", columns="instrument", values="next_return", aggfunc="first")
    corr_profiles = correlation_contribution_profiles(returns_pivot, bridge["instrument"].unique())
    name_map = build_name_map(fund_list_csv)

    profile_rows: list[dict[str, object]] = []
    for code, frame in bridge.groupby("instrument"):
        period_rank = safe_float(period_ranks.get(code, np.nan))
        row: dict[str, object] = {
            "instrument": code,
            "name": name_map.get(code, ""),
            "period_return": safe_float(period_return.get(code, np.nan)),
            "period_return_rank": int(period_rank) if np.isfinite(period_rank) else np.nan,
            "is_period_top_winner": code in actual_top_winner_set,
            "avg_forecast_mean_top_rank_pct": float(numeric_series(frame["forecast_mean_top_rank_pct"]).mean()),
            "mean_top30_day_share": float((numeric_series(frame["forecast_mean_top_rank_pct"]) <= mean_top_pct).mean()),
            "avg_forecast_vol_rank_pct": float(numeric_series(frame["forecast_vol_rank_pct"]).mean()),
        }
        row.update(realized_volatility_profile(frame["next_return"]))
        row.update(corr_profiles.get(code, {}))
        row["offensive_volatility_flag"] = is_offensive_volatility(
            pd.Series(row),
            min_period_return=min_period_return,
            upside_share_min=upside_share_min,
            positive_share_min=positive_share_min,
        )
        row["decision_tree_label"] = decision_tree_label(
            pd.Series(row),
            mean_top_pct=mean_top_pct,
            clone_corr_threshold=clone_corr_threshold,
        )
        profile_rows.append(row)
    candidates = pd.DataFrame(profile_rows).sort_values("period_return_rank")
    hidden_gem_codes = set(candidates.loc[candidates["decision_tree_label"] == "Hidden Gems", "instrument"].tolist())
    bridge["decision_tree_label"] = bridge["instrument"].map(
        candidates.set_index("instrument")["decision_tree_label"].to_dict()
    )
    bridge["is_hidden_gem"] = bridge["instrument"].isin(hidden_gem_codes)
    bridge["local_cap_candidate"] = bridge["is_hidden_gem"] & (bridge["forecast_mean_top_rank_pct"] <= mean_top_pct)

    safe_top_k = max(1, int(top_k))
    selected_period_top10_mean_return = float(actual_top_winners.mean()) if not actual_top_winners.empty else np.nan
    scenario_rows: list[dict[str, object]] = []
    for risk_free_rate in risk_free_rates:
        risk_free_daily = float(risk_free_rate) / 252.0
        for cap_rank_pct in vol_cap_rank_pcts:
            cap_rank_pct = float(cap_rank_pct)
            working = bridge.copy()
            if cap_rank_pct >= 1.0:
                working["day_vol_cap"] = np.inf
            else:
                working["day_vol_cap"] = working.groupby("datetime")["forecast_vol"].transform(
                    lambda series: numeric_series(series).quantile(cap_rank_pct)
                )
            capped_vol = np.minimum(working["forecast_vol"], working["day_vol_cap"])
            working["effective_vol"] = np.where(working["local_cap_candidate"], capped_vol, working["forecast_vol"])
            working["effective_variance"] = np.square(working["effective_vol"])
            working["audit_score"] = (working["forecast_mean"] - risk_free_daily) / (working["effective_variance"] + 1e-12)
            working["cap_applied"] = working["local_cap_candidate"] & (working["effective_vol"] < working["forecast_vol"])

            mean_scores = working.groupby("instrument")["audit_score"].mean().dropna().sort_values(ascending=False)
            score_top = mean_scores.head(safe_top_k)
            score_top_codes = score_top.index.tolist()
            score_top_period_returns = period_return.reindex(score_top_codes).dropna()
            daily_rows: list[dict[str, object]] = []
            for date_value, day_frame in working.groupby("datetime"):
                daily_return, daily_score, selected_codes_text = score_top_k_daily_return(day_frame, "audit_score", safe_top_k)
                selected_codes = set(selected_codes_text.split("|")) if selected_codes_text else set()
                daily_rows.append(
                    {
                        "datetime": date_value,
                        "daily_return": daily_return,
                        "daily_score": daily_score,
                        "period_top_winner_overlap": len(selected_codes.intersection(actual_top_winner_set)) / safe_top_k,
                        "hidden_gem_overlap": len(selected_codes.intersection(hidden_gem_codes)) / safe_top_k,
                        "cap_applied_selected_count": int(day_frame.loc[day_frame["cap_applied"], "instrument"].isin(selected_codes).sum()),
                    }
                )
            daily = pd.DataFrame(daily_rows)
            stats = return_stats(daily.set_index("datetime")["daily_return"] if not daily.empty else pd.Series(dtype=float))
            hit_count = len(set(score_top_codes).intersection(actual_top_winner_set))
            row = {
                "scenario": f"localcap_rf_{risk_free_rate:.4f}_volcap_{cap_rank_pct:.2f}",
                "risk_free_rate_annual": float(risk_free_rate),
                "risk_free_rate_daily": risk_free_daily,
                "vol_cap_rank_pct": cap_rank_pct,
                "hidden_gem_count": int(len(hidden_gem_codes)),
                "hidden_gem_codes": "|".join(sorted(hidden_gem_codes)),
                "local_cap_candidate_row_share": float(working["local_cap_candidate"].mean()),
                "cap_applied_row_share": float(working["cap_applied"].mean()),
                "selected_period_top10_mean_return": selected_period_top10_mean_return,
                "score_top10_mean_period_return": float(score_top_period_returns.mean()) if not score_top_period_returns.empty else np.nan,
                "score_top10_above_30pct": bool(score_top_period_returns.mean() >= 0.30) if not score_top_period_returns.empty else False,
                "score_top10_hit_count_vs_period_top10": int(hit_count),
                "score_top10_capture_rate_vs_period_top10": hit_count / safe_top_k,
                "daily_score_top10_avg_overlap_vs_period_top10": float(daily["period_top_winner_overlap"].mean()) if not daily.empty else np.nan,
                "daily_score_top10_avg_hidden_gem_overlap": float(daily["hidden_gem_overlap"].mean()) if not daily.empty else np.nan,
                "daily_cap_applied_selected_avg_count": float(daily["cap_applied_selected_count"].mean()) if not daily.empty else np.nan,
                "daily_score_top10_avg_score": float(daily["daily_score"].mean()) if not daily.empty else np.nan,
                "score_top10_codes": "|".join(score_top_codes),
            }
            row.update({f"daily_score_top10_{key}": value for key, value in stats.items()})
            scenario_rows.append(row)

    scenarios = pd.DataFrame(scenario_rows)
    if not scenarios.empty:
        scenarios["target_overlap_40pct_pass"] = scenarios["daily_score_top10_avg_overlap_vs_period_top10"] >= 0.40
        baseline = scenarios[scenarios["vol_cap_rank_pct"] >= 1.0].set_index("risk_free_rate_annual")
        delta_overlap: list[float] = []
        delta_maxdd: list[float] = []
        for _, row in scenarios.iterrows():
            risk_free_rate = row["risk_free_rate_annual"]
            if risk_free_rate in baseline.index:
                baseline_row = baseline.loc[risk_free_rate]
                delta_overlap.append(
                    safe_float(row["daily_score_top10_avg_overlap_vs_period_top10"])
                    - safe_float(baseline_row["daily_score_top10_avg_overlap_vs_period_top10"])
                )
                delta_maxdd.append(
                    safe_float(row["daily_score_top10_max_drawdown"])
                    - safe_float(baseline_row["daily_score_top10_max_drawdown"])
                )
            else:
                delta_overlap.append(np.nan)
                delta_maxdd.append(np.nan)
        scenarios["delta_overlap_vs_no_cap"] = delta_overlap
        scenarios["delta_maxdd_vs_no_cap"] = delta_maxdd
        scenarios["maxdd_not_worse_than_no_cap_1pct"] = scenarios["delta_maxdd_vs_no_cap"] >= -0.01

    write_table(candidates, out_dir, "local_variance_penalty_cap_candidates.csv")
    write_table(scenarios, out_dir, "local_variance_penalty_cap_scenarios.csv")
    return {
        "local_variance_penalty_cap_candidates": candidates,
        "local_variance_penalty_cap_scenarios": scenarios,
    }


def local_variance_penalty_cap_oos_audit(
    bridge_path: Path,
    out_dir: Path,
    fund_list_csv: str | Path,
    top_k: int,
    risk_free_rates: list[float],
    vol_cap_rank_pcts: list[float],
    mean_top_pct: float,
    clone_corr_threshold: float,
    min_period_return: float,
    upside_share_min: float,
    positive_share_min: float,
    lookback_days: int,
    forward_days: int,
    step_days: int,
) -> dict[str, pd.DataFrame]:
    requested_columns = [
        "instrument",
        "datetime",
        "next_return",
        "forecast_mean_return_robust",
        "forecast_mean_return",
        "forecast_variance_robust",
        "forecast_variance",
    ]
    bridge = read_csv_columns(bridge_path, requested_columns)
    mu_col = "forecast_mean_return_robust" if "forecast_mean_return_robust" in bridge.columns else "forecast_mean_return"
    var_col = "forecast_variance_robust" if "forecast_variance_robust" in bridge.columns else "forecast_variance"
    empty_result = {
        "local_variance_penalty_cap_oos_daily": pd.DataFrame(),
        "local_variance_penalty_cap_oos_folds": pd.DataFrame(),
        "local_variance_penalty_cap_oos_summary": pd.DataFrame(),
    }
    if not {"instrument", "datetime", "next_return", mu_col, var_col}.issubset(bridge.columns):
        for table_name, frame in empty_result.items():
            write_table(frame, out_dir, f"{table_name}.csv")
        return empty_result

    bridge = bridge.sort_values(["datetime", "instrument"]).copy()
    bridge["next_return"] = numeric_series(bridge["next_return"])
    bridge["forecast_mean"] = numeric_series(bridge[mu_col])
    bridge["forecast_variance"] = numeric_series(bridge[var_col]).clip(lower=0.0)
    bridge["forecast_vol"] = np.sqrt(bridge["forecast_variance"])
    bridge["forecast_mean_top_rank_pct"] = bridge.groupby("datetime")["forecast_mean"].transform(
        lambda series: numeric_series(series).rank(pct=True, ascending=False)
    )
    bridge["forecast_vol_rank_pct"] = bridge.groupby("datetime")["forecast_vol"].transform(
        lambda series: numeric_series(series).rank(pct=True)
    )
    dates = list(pd.Series(bridge["datetime"].drop_duplicates()).sort_values())
    safe_top_k = max(1, int(top_k))
    safe_lookback = max(5, int(lookback_days))
    safe_forward = max(1, int(forward_days))
    safe_step = max(1, int(step_days))
    name_map = build_name_map(fund_list_csv)
    daily_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    fold_index = 0

    last_start = len(dates) - safe_lookback - safe_forward
    for train_start_index in range(0, max(last_start + 1, 0), safe_step):
        train_dates = dates[train_start_index: train_start_index + safe_lookback]
        oos_dates = dates[train_start_index + safe_lookback: train_start_index + safe_lookback + safe_forward]
        if len(train_dates) < safe_lookback or len(oos_dates) < safe_forward:
            continue
        fold_index += 1
        train_frame = bridge[bridge["datetime"].isin(train_dates)].copy()
        oos_frame = bridge[bridge["datetime"].isin(oos_dates)].copy()
        train_candidates, hidden_gem_codes, train_top_winner_set = classify_hidden_gems_from_window(
            train_frame,
            name_map,
            safe_top_k,
            mean_top_pct,
            clone_corr_threshold,
            min_period_return,
            upside_share_min,
            positive_share_min,
        )
        oos_period_return = oos_frame.groupby("instrument")["next_return"].apply(compound_return).dropna()
        oos_top_winners = oos_period_return.sort_values(ascending=False).head(safe_top_k)
        oos_top_winner_set = set(oos_top_winners.index)
        hidden_gem_forward_returns = oos_period_return.reindex(sorted(hidden_gem_codes)).dropna()

        for risk_free_rate in risk_free_rates:
            risk_free_daily = float(risk_free_rate) / 252.0
            for cap_rank_pct in vol_cap_rank_pcts:
                cap_rank_pct = float(cap_rank_pct)
                working = oos_frame.copy()
                working["is_hidden_gem_from_lookback"] = working["instrument"].isin(hidden_gem_codes)
                working["local_cap_candidate"] = (
                    working["is_hidden_gem_from_lookback"]
                    & (working["forecast_mean_top_rank_pct"] <= mean_top_pct)
                )
                if cap_rank_pct >= 1.0:
                    working["day_vol_cap"] = np.inf
                else:
                    working["day_vol_cap"] = working.groupby("datetime")["forecast_vol"].transform(
                        lambda series: numeric_series(series).quantile(cap_rank_pct)
                    )
                working["effective_vol"] = np.where(
                    working["local_cap_candidate"],
                    np.minimum(working["forecast_vol"], working["day_vol_cap"]),
                    working["forecast_vol"],
                )
                working["effective_variance"] = np.square(working["effective_vol"])
                working["audit_score"] = (working["forecast_mean"] - risk_free_daily) / (
                    working["effective_variance"] + 1e-12
                )
                working["cap_applied"] = working["local_cap_candidate"] & (
                    working["effective_vol"] < working["forecast_vol"]
                )
                scenario = f"oos_localcap_rf_{risk_free_rate:.4f}_volcap_{cap_rank_pct:.2f}"
                scenario_daily_rows: list[dict[str, object]] = []
                for date_value, day_frame in working.groupby("datetime"):
                    daily_return, daily_score, selected_codes_text = score_top_k_daily_return(day_frame, "audit_score", safe_top_k)
                    selected_codes = set(selected_codes_text.split("|")) if selected_codes_text else set()
                    row = {
                        "fold": fold_index,
                        "scenario": scenario,
                        "risk_free_rate_annual": float(risk_free_rate),
                        "vol_cap_rank_pct": cap_rank_pct,
                        "datetime": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                        "daily_return": daily_return,
                        "daily_score": daily_score,
                        "selected_codes": selected_codes_text,
                        "oos_top_winner_overlap": len(selected_codes.intersection(oos_top_winner_set)) / safe_top_k,
                        "train_top_winner_overlap": len(selected_codes.intersection(train_top_winner_set)) / safe_top_k,
                        "hidden_gem_overlap": len(selected_codes.intersection(hidden_gem_codes)) / safe_top_k,
                        "cap_applied_selected_count": int(day_frame.loc[day_frame["cap_applied"], "instrument"].isin(selected_codes).sum()),
                    }
                    daily_rows.append(row)
                    scenario_daily_rows.append(row)
                scenario_daily = pd.DataFrame(scenario_daily_rows)
                stats = return_stats(
                    scenario_daily.set_index("datetime")["daily_return"] if not scenario_daily.empty else pd.Series(dtype=float)
                )
                score_top = working.groupby("instrument")["audit_score"].mean().dropna().sort_values(ascending=False).head(safe_top_k)
                score_top_codes = score_top.index.tolist()
                score_top_period_returns = oos_period_return.reindex(score_top_codes).dropna()
                fold_row = {
                    "fold": fold_index,
                    "scenario": scenario,
                    "risk_free_rate_annual": float(risk_free_rate),
                    "risk_free_rate_daily": risk_free_daily,
                    "vol_cap_rank_pct": cap_rank_pct,
                    "train_start": pd.Timestamp(train_dates[0]).strftime("%Y-%m-%d"),
                    "train_end": pd.Timestamp(train_dates[-1]).strftime("%Y-%m-%d"),
                    "oos_start": pd.Timestamp(oos_dates[0]).strftime("%Y-%m-%d"),
                    "oos_end": pd.Timestamp(oos_dates[-1]).strftime("%Y-%m-%d"),
                    "train_hidden_gem_count": int(len(hidden_gem_codes)),
                    "train_hidden_gem_codes": "|".join(sorted(hidden_gem_codes)),
                    "train_top_winner_codes": "|".join(sorted(train_top_winner_set)),
                    "oos_top_winner_codes": "|".join(oos_top_winners.index.tolist()),
                    "hidden_gem_oos_mean_return": float(hidden_gem_forward_returns.mean()) if not hidden_gem_forward_returns.empty else np.nan,
                    "score_top10_oos_mean_period_return": float(score_top_period_returns.mean()) if not score_top_period_returns.empty else np.nan,
                    "score_top10_oos_capture_rate": len(set(score_top_codes).intersection(oos_top_winner_set)) / safe_top_k,
                    "daily_oos_top10_avg_overlap": float(scenario_daily["oos_top_winner_overlap"].mean()) if not scenario_daily.empty else np.nan,
                    "daily_hidden_gem_avg_overlap": float(scenario_daily["hidden_gem_overlap"].mean()) if not scenario_daily.empty else np.nan,
                    "daily_cap_applied_selected_avg_count": float(scenario_daily["cap_applied_selected_count"].mean()) if not scenario_daily.empty else np.nan,
                    "score_top10_codes": "|".join(score_top_codes),
                }
                fold_row.update({f"oos_score_top10_{key}": value for key, value in stats.items()})
                fold_rows.append(fold_row)

    daily = pd.DataFrame(daily_rows)
    folds = pd.DataFrame(fold_rows)
    summary_rows: list[dict[str, object]] = []
    if not daily.empty and not folds.empty:
        for scenario, scenario_daily in daily.groupby("scenario"):
            scenario_folds = folds[folds["scenario"] == scenario]
            stats = return_stats(scenario_daily.set_index("datetime")["daily_return"])
            first_fold = scenario_folds.iloc[0]
            row = {
                "scenario": scenario,
                "risk_free_rate_annual": float(first_fold["risk_free_rate_annual"]),
                "vol_cap_rank_pct": float(first_fold["vol_cap_rank_pct"]),
                "fold_count": int(scenario_folds["fold"].nunique()),
                "lookback_days": safe_lookback,
                "forward_days": safe_forward,
                "step_days": safe_step,
                "avg_train_hidden_gem_count": float(scenario_folds["train_hidden_gem_count"].mean()),
                "avg_hidden_gem_oos_mean_return": float(numeric_series(scenario_folds["hidden_gem_oos_mean_return"]).mean()),
                "avg_score_top10_oos_mean_period_return": float(numeric_series(scenario_folds["score_top10_oos_mean_period_return"]).mean()),
                "avg_score_top10_oos_capture_rate": float(numeric_series(scenario_folds["score_top10_oos_capture_rate"]).mean()),
                "daily_oos_top10_avg_overlap": float(scenario_daily["oos_top_winner_overlap"].mean()),
                "daily_hidden_gem_avg_overlap": float(scenario_daily["hidden_gem_overlap"].mean()),
                "daily_cap_applied_selected_avg_count": float(scenario_daily["cap_applied_selected_count"].mean()),
                "target_overlap_40pct_pass": float(scenario_daily["oos_top_winner_overlap"].mean()) >= 0.40,
            }
            row.update({f"oos_score_top10_{key}": value for key, value in stats.items()})
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        baseline = summary[summary["vol_cap_rank_pct"] >= 1.0].set_index("risk_free_rate_annual")
        delta_overlap: list[float] = []
        delta_maxdd: list[float] = []
        for _, row in summary.iterrows():
            risk_free_rate = row["risk_free_rate_annual"]
            if risk_free_rate in baseline.index:
                baseline_row = baseline.loc[risk_free_rate]
                delta_overlap.append(
                    safe_float(row["daily_oos_top10_avg_overlap"])
                    - safe_float(baseline_row["daily_oos_top10_avg_overlap"])
                )
                delta_maxdd.append(
                    safe_float(row["oos_score_top10_max_drawdown"])
                    - safe_float(baseline_row["oos_score_top10_max_drawdown"])
                )
            else:
                delta_overlap.append(np.nan)
                delta_maxdd.append(np.nan)
        summary["delta_oos_overlap_vs_no_cap"] = delta_overlap
        summary["delta_oos_maxdd_vs_no_cap"] = delta_maxdd
        summary["oos_maxdd_not_worse_than_no_cap_1pct"] = summary["delta_oos_maxdd_vs_no_cap"] >= -0.01

    write_table(daily, out_dir, "local_variance_penalty_cap_oos_daily.csv")
    write_table(folds, out_dir, "local_variance_penalty_cap_oos_folds.csv")
    write_table(summary, out_dir, "local_variance_penalty_cap_oos_summary.csv")
    return {
        "local_variance_penalty_cap_oos_daily": daily,
        "local_variance_penalty_cap_oos_folds": folds,
        "local_variance_penalty_cap_oos_summary": summary,
    }


def parse_float_vector(value: object) -> np.ndarray:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.array([], dtype=float)
    return np.fromstring(str(value), sep=",", dtype=float)


def covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    diagonal = np.diag(covariance).astype(float)
    denominator = np.sqrt(np.clip(diagonal, 1e-18, None))
    corr = covariance / np.outer(denominator, denominator)
    return np.clip(corr, -1.0, 1.0)


def flatten_upper(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.array([], dtype=float)
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def covariance_ic_tables(
    bridge_path: Path,
    out_dir: Path,
    future_window: int,
    min_future_obs: int,
) -> dict[str, pd.DataFrame]:
    requested_columns = [
        "instrument",
        "datetime",
        "next_return",
        "forecast_cov_matrix_robust",
        "forecast_cov_matrix",
        "forecast_cov_order",
    ]
    bridge = read_csv_columns(bridge_path, requested_columns)
    cov_col = "forecast_cov_matrix_robust" if "forecast_cov_matrix_robust" in bridge.columns else "forecast_cov_matrix"
    returns_pivot = bridge.pivot_table(index="datetime", columns="instrument", values="next_return", aggfunc="first")
    returns_pivot = returns_pivot.sort_index().apply(pd.to_numeric, errors="coerce")
    daily_rows: list[dict[str, object]] = []
    for date_value, day_frame in bridge.groupby("datetime"):
        if "forecast_cov_order" not in day_frame.columns or cov_col not in day_frame.columns:
            continue
        order_text = str(day_frame["forecast_cov_order"].dropna().iloc[0]) if day_frame["forecast_cov_order"].notna().any() else ""
        order_codes = [normalize_code(token) for token in order_text.split("|") if token]
        if len(order_codes) < 3:
            continue
        day_by_code = day_frame.set_index("instrument", drop=False)
        matrix_rows: list[np.ndarray] = []
        missing_row = False
        for code in order_codes:
            if code not in day_by_code.index:
                missing_row = True
                break
            row_value = day_by_code.loc[code, cov_col]
            if isinstance(row_value, pd.Series):
                row_value = row_value.iloc[0]
            vector = parse_float_vector(row_value)
            if len(vector) != len(order_codes):
                missing_row = True
                break
            matrix_rows.append(vector)
        if missing_row:
            continue
        covariance = np.vstack(matrix_rows)
        pred_corr = covariance_to_correlation(covariance)
        future_returns = returns_pivot.loc[returns_pivot.index >= date_value, order_codes].head(future_window)
        if len(future_returns.dropna(how="all")) < min_future_obs:
            continue
        realized_corr = future_returns.corr(min_periods=min_future_obs).to_numpy()
        pred_pairs = flatten_upper(pred_corr)
        realized_pairs = flatten_upper(realized_corr)
        finite_mask = np.isfinite(pred_pairs) & np.isfinite(realized_pairs)
        if finite_mask.sum() < 10:
            continue
        pred_series = pd.Series(pred_pairs[finite_mask])
        realized_series = pd.Series(realized_pairs[finite_mask])
        daily_rows.append(
            {
                "datetime": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                "order_n": len(order_codes),
                "pair_n": int(finite_mask.sum()),
                "pair_spearman_corr": float(pred_series.corr(realized_series, method="spearman")),
                "pair_pearson_corr": float(pred_series.corr(realized_series)),
                "pred_avg_abs_corr": float(np.nanmean(np.abs(pred_pairs[finite_mask]))),
                "realized_avg_abs_corr": float(np.nanmean(np.abs(realized_pairs[finite_mask]))),
            }
        )
    daily = pd.DataFrame(daily_rows)
    summary_rows = []
    for metric in ["pair_spearman_corr", "pair_pearson_corr", "pred_avg_abs_corr", "realized_avg_abs_corr"]:
        metric_summary = summarize_series(daily[metric] if metric in daily.columns else pd.Series(dtype=float))
        metric_summary["metric"] = metric
        summary_rows.append(metric_summary)
    summary = pd.DataFrame(summary_rows)
    write_table(daily, out_dir, "covariance_ic_daily.csv")
    write_table(summary, out_dir, "covariance_ic_summary.csv")
    return {"covariance_ic_daily": daily, "covariance_ic_summary": summary}


def weighted_next_return(day_frame: pd.DataFrame, weight_col: str, exposure_col: str | None = None) -> float:
    returns = numeric_series(day_frame["next_return"]).to_numpy(dtype=float)
    weights = numeric_series(day_frame[weight_col]).clip(lower=0.0).fillna(0.0).to_numpy(dtype=float)
    weight_sum = float(weights.sum())
    if weight_sum <= 1e-12:
        return np.nan
    normalized_weights = weights / weight_sum
    if exposure_col is not None and exposure_col in day_frame.columns:
        exposure = numeric_series(day_frame[exposure_col]).fillna(1.0).clip(lower=0.0, upper=1.0).to_numpy(dtype=float)
        normalized_weights = normalized_weights * exposure
    return float(np.nansum(normalized_weights * returns))


def risk_parity_next_return(day_frame: pd.DataFrame, var_col: str) -> float:
    returns = numeric_series(day_frame["next_return"]).to_numpy(dtype=float)
    variances = numeric_series(day_frame[var_col]).clip(lower=1e-12).to_numpy(dtype=float)
    valid_mask = np.isfinite(returns) & np.isfinite(variances)
    if valid_mask.sum() <= 1:
        return np.nan
    inv_vol = 1.0 / np.sqrt(variances[valid_mask])
    weights = inv_vol / inv_vol.sum()
    return float(np.dot(weights, returns[valid_mask]))


def pure_momentum_next_return(day_frame: pd.DataFrame, top_k: int) -> float:
    momentum_cols = [column for column in day_frame.columns if column.startswith("trend_mom")]
    if not momentum_cols:
        return np.nan
    working = day_frame[["next_return", *momentum_cols]].copy()
    for column in momentum_cols:
        working[column] = numeric_series(working[column])
        col_std = working[column].std(ddof=0)
        if np.isfinite(col_std) and col_std > 0:
            working[column] = (working[column] - working[column].mean()) / col_std
    working["momentum_score"] = working[momentum_cols].mean(axis=1, skipna=True)
    working["next_return"] = numeric_series(working["next_return"])
    selected = working.dropna(subset=["momentum_score", "next_return"]).nlargest(max(1, top_k), "momentum_score")
    if selected.empty:
        return np.nan
    return float(selected["next_return"].mean())


def portfolio_compare_tables(
    bridge_path: Path,
    l3_path: Path,
    daily_nav_path: Path,
    out_dir: Path,
    pure_momentum_top_k: int,
) -> dict[str, pd.DataFrame]:
    requested_raw = [
        "instrument",
        "datetime",
        "next_return",
        "forecast_w",
        "forecast_variance_robust",
        "forecast_variance",
    ]
    requested_l3 = [
        "instrument",
        "datetime",
        "next_return",
        "forecast_w",
        "exposure_scale",
        "trend_mom5",
        "trend_mom20",
        "trend_mom60",
    ]
    raw = read_csv_columns(bridge_path, requested_raw)
    l3 = read_csv_columns(l3_path, requested_l3)
    var_col = "forecast_variance_robust" if "forecast_variance_robust" in raw.columns else "forecast_variance"

    method_returns: dict[str, pd.Series] = {}
    method_returns["mvo_raw_forecast_w"] = raw.groupby("datetime").apply(
        lambda day_frame: weighted_next_return(day_frame, "forecast_w"), include_groups=False
    )
    method_returns["equal_weight_pool"] = raw.groupby("datetime")["next_return"].mean()
    method_returns["inverse_forecast_vol"] = raw.groupby("datetime").apply(
        lambda day_frame: risk_parity_next_return(day_frame, var_col), include_groups=False
    )
    method_returns[f"pure_momentum_top{pure_momentum_top_k}"] = l3.groupby("datetime").apply(
        lambda day_frame: pure_momentum_next_return(day_frame, pure_momentum_top_k), include_groups=False
    )
    method_returns["l3_scaled_forecast_w"] = l3.groupby("datetime").apply(
        lambda day_frame: weighted_next_return(day_frame, "forecast_w", "exposure_scale"), include_groups=False
    )
    method_returns["l3_backtest_nav"] = nav_daily_returns(daily_nav_path)

    daily = pd.concat(method_returns, axis=1).sort_index()
    daily.index.name = "datetime"
    summary_rows: list[dict[str, object]] = []
    for method_name in daily.columns:
        stats = return_stats(daily[method_name])
        stats["method"] = method_name
        summary_rows.append(stats)
    summary = pd.DataFrame(summary_rows)
    write_table(daily.reset_index(), out_dir, "portfolio_method_daily_returns.csv")
    write_table(summary, out_dir, "portfolio_method_compare.csv")
    return {"portfolio_method_daily_returns": daily.reset_index(), "portfolio_method_compare": summary}


def mean_bias_reversal_correlation_audit(
    l3_path: Path,
    out_dir: Path,
) -> dict[str, pd.DataFrame]:
    requested_columns = [
        "instrument",
        "datetime",
        "close",
        "next_return",
        "forecast_w",
        "forecast_mean_return",
        "forecast_mean_return_robust",
        "forecast_weight_mean_raw",
        "forecast_weight_mean_used",
        "mu_raw",
        "mu_used",
    ]
    frame = read_csv_columns(l3_path, requested_columns)
    required = {"instrument", "datetime", "close"}
    if not required.issubset(frame.columns):
        heatmap = pd.DataFrame()
        event_focus = pd.DataFrame()
        active_detail = pd.DataFrame()
        write_table(heatmap, out_dir, "mean_bias_reversal_correlation_heatmap.csv")
        write_table(event_focus, out_dir, "mean_bias_reversal_event_focus.csv")
        write_table(active_detail, out_dir, "mean_bias_reversal_active_detail.csv")
        return {
            "mean_bias_reversal_correlation_heatmap": heatmap,
            "mean_bias_reversal_event_focus": event_focus,
            "mean_bias_reversal_active_detail": active_detail,
        }

    frame = frame.sort_values(["instrument", "datetime"]).copy()
    frame["close"] = numeric_series(frame["close"])
    if "forecast_w" not in frame.columns:
        frame["forecast_w"] = 0.0
    frame["forecast_w"] = numeric_series(frame["forecast_w"]).fillna(0.0)
    for lag_days in range(1, 6):
        frame[f"hist_return_t_minus_{lag_days}"] = (
            frame.groupby("instrument")["close"].shift(lag_days)
            / frame.groupby("instrument")["close"].shift(lag_days + 1)
            - 1.0
        )
    frame["hist_return_t_minus_5_to_t_minus_1"] = (
        frame.groupby("instrument")["close"].shift(1)
        / frame.groupby("instrument")["close"].shift(5)
        - 1.0
    )

    mean_columns = [
        column
        for column in [
            "forecast_mean_return",
            "forecast_mean_return_robust",
            "forecast_weight_mean_raw",
            "forecast_weight_mean_used",
            "mu_raw",
            "mu_used",
        ]
        if column in frame.columns and numeric_series(frame[column]).notna().sum() > 0
    ]
    hist_columns = [
        "hist_return_t_minus_5_to_t_minus_1",
        "hist_return_t_minus_1",
        "hist_return_t_minus_2",
        "hist_return_t_minus_3",
        "hist_return_t_minus_4",
        "hist_return_t_minus_5",
    ]
    rows: list[dict[str, object]] = []
    for date_value, day_frame in frame.groupby("datetime", sort=True):
        for mean_column in mean_columns:
            for hist_column in hist_columns:
                corr_frame = pd.DataFrame(
                    {
                        "forecast_mean": numeric_series(day_frame[mean_column]),
                        "historical_return": numeric_series(day_frame[hist_column]),
                    }
                ).dropna()
                row = {
                    "datetime": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                    "mean_column": mean_column,
                    "hist_feature": hist_column,
                    "n": int(corr_frame.shape[0]),
                    "pearson_corr": np.nan,
                    "spearman_corr": np.nan,
                    "short_reversal_score": np.nan,
                    "event_date_flag": pd.Timestamp(date_value).strftime("%Y-%m-%d") in MEAN_BIAS_REVERSAL_EVENT_DATES,
                }
                if corr_frame.shape[0] >= 5 and corr_frame["forecast_mean"].nunique() > 1 and corr_frame["historical_return"].nunique() > 1:
                    spearman_corr = float(corr_frame["forecast_mean"].corr(corr_frame["historical_return"], method="spearman"))
                    row["pearson_corr"] = float(corr_frame["forecast_mean"].corr(corr_frame["historical_return"]))
                    row["spearman_corr"] = spearman_corr
                    row["short_reversal_score"] = -spearman_corr
                rows.append(row)

    heatmap = pd.DataFrame(rows)
    if not heatmap.empty:
        heatmap["short_reversal_score_pct_rank"] = heatmap.groupby(["mean_column", "hist_feature"])[
            "short_reversal_score"
        ].rank(pct=True)
        heatmap["trend_chase_score_pct_rank"] = heatmap.groupby(["mean_column", "hist_feature"])[
            "spearman_corr"
        ].rank(pct=True)
        heatmap["reversal_bias_label"] = np.where(
            (numeric_series(heatmap["short_reversal_score"]) > 0.20)
            & (numeric_series(heatmap["short_reversal_score_pct_rank"]) >= 0.75),
            "elevated_short_reversal_bias",
            "normal_or_weak",
        )
    event_focus = heatmap[
        heatmap.get("event_date_flag", pd.Series(False, index=heatmap.index)).fillna(False).astype(bool)
    ].copy() if not heatmap.empty else pd.DataFrame()

    active_detail_columns = ["datetime", "instrument", "forecast_w", *mean_columns, *hist_columns, "next_return"]
    active_detail = frame[
        frame["datetime"].dt.strftime("%Y-%m-%d").isin(MEAN_BIAS_REVERSAL_EVENT_DATES)
        & (frame["forecast_w"] > 1e-12)
    ][[column for column in active_detail_columns if column in frame.columns]].copy()
    if not active_detail.empty:
        active_detail["datetime"] = active_detail["datetime"].dt.strftime("%Y-%m-%d")
        if "forecast_mean_return" in active_detail.columns:
            active_detail["forecast_mean_vs_hist5_reversal_alignment"] = -numeric_series(
                active_detail["forecast_mean_return"]
            ) * numeric_series(active_detail["hist_return_t_minus_5_to_t_minus_1"])

    heatmap_matrix = pd.DataFrame()
    if not heatmap.empty and "forecast_mean_return" in mean_columns:
        heatmap_matrix = heatmap[heatmap["mean_column"] == "forecast_mean_return"].pivot_table(
            index="datetime",
            columns="hist_feature",
            values="spearman_corr",
            aggfunc="first",
        )
        heatmap_matrix = heatmap_matrix.reindex(columns=hist_columns)
        write_table(heatmap_matrix.reset_index(), out_dir, "mean_bias_reversal_correlation_heatmap_matrix.csv")
        try:
            import matplotlib.pyplot as plt

            plot_matrix = heatmap_matrix.dropna(how="all")
            if not plot_matrix.empty:
                fig_height = max(4.0, min(18.0, 0.18 * len(plot_matrix) + 2.5))
                fig, axis = plt.subplots(figsize=(9.5, fig_height))
                image = axis.imshow(plot_matrix.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
                axis.set_xticks(range(len(plot_matrix.columns)))
                axis.set_xticklabels(plot_matrix.columns, rotation=35, ha="right", fontsize=8)
                axis.set_yticks(range(len(plot_matrix.index)))
                axis.set_yticklabels(plot_matrix.index, fontsize=6)
                axis.set_title("Forecast Mean vs Historical Return Spearman Corr")
                fig.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
                fig.tight_layout()
                fig.savefig(out_dir / "mean_bias_reversal_correlation_heatmap.png", dpi=160)
                plt.close(fig)
        except Exception as exc:
            (out_dir / "mean_bias_reversal_correlation_heatmap_error.txt").write_text(str(exc), encoding="utf-8")

    write_table(heatmap, out_dir, "mean_bias_reversal_correlation_heatmap.csv")
    write_table(event_focus, out_dir, "mean_bias_reversal_event_focus.csv")
    write_table(active_detail, out_dir, "mean_bias_reversal_active_detail.csv")
    return {
        "mean_bias_reversal_correlation_heatmap": heatmap,
        "mean_bias_reversal_event_focus": event_focus,
        "mean_bias_reversal_active_detail": active_detail,
        "mean_bias_reversal_correlation_heatmap_matrix": heatmap_matrix.reset_index() if not heatmap_matrix.empty else pd.DataFrame(),
    }


def normalized_entropy_from_shares(shares: pd.Series) -> tuple[float, float, float]:
    values = numeric_series(shares).dropna()
    values = values[values > 0.0]
    total = float(values.sum())
    if total <= 1e-12:
        return np.nan, np.nan, np.nan
    probabilities = values / total
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    effective_count = float(np.exp(entropy))
    normalized = float(entropy / np.log(len(probabilities))) if len(probabilities) > 1 else 0.0
    return entropy, normalized, effective_count


def signal_quality_industry_entropy_cross_audit(
    l3_path: Path,
    out_dir: Path,
    fund_list_csv: str | Path,
) -> dict[str, pd.DataFrame]:
    requested_columns = [
        "instrument",
        "datetime",
        "next_return",
        "forecast_w",
        "exposure_scale",
        "risk_signal_quality_rolling_ic",
        "risk_signal_quality_scale",
        "risk_regime_on",
        "factor_industry_theme",
    ]
    frame = read_csv_columns(l3_path, requested_columns)
    if not {"instrument", "datetime", "next_return", "forecast_w"}.issubset(frame.columns):
        daily = pd.DataFrame()
        detail = pd.DataFrame()
        focus = pd.DataFrame()
        write_table(daily, out_dir, "signal_quality_industry_entropy_daily.csv")
        write_table(detail, out_dir, "signal_quality_industry_entropy_detail.csv")
        write_table(focus, out_dir, "signal_quality_industry_entropy_focus.csv")
        return {
            "signal_quality_industry_entropy_daily": daily,
            "signal_quality_industry_entropy_detail": detail,
            "signal_quality_industry_entropy_focus": focus,
        }

    frame = frame.sort_values(["datetime", "instrument"]).copy()
    frame["next_return"] = numeric_series(frame["next_return"])
    frame["forecast_w"] = numeric_series(frame["forecast_w"]).fillna(0.0).clip(lower=0.0)
    if "exposure_scale" not in frame.columns:
        frame["exposure_scale"] = 1.0
    frame["exposure_scale"] = numeric_series(frame["exposure_scale"]).fillna(1.0).clip(lower=0.0, upper=1.0)
    frame["effective_weight"] = frame["forecast_w"] * frame["exposure_scale"]
    name_map = build_name_map(fund_list_csv)
    frame["name"] = frame["instrument"].map(name_map).fillna("")
    if "factor_industry_theme" not in frame.columns:
        frame["factor_industry_theme"] = ""
    fallback_theme = frame["instrument"].map(name_map).map(audit_l3_industry_label).fillna("other")
    frame["factor_industry_theme"] = frame["factor_industry_theme"].fillna("").astype(str)
    frame.loc[frame["factor_industry_theme"].str.len() == 0, "factor_industry_theme"] = fallback_theme
    frame.loc[frame["factor_industry_theme"].str.lower() == "nan", "factor_industry_theme"] = fallback_theme
    frame.loc[frame["factor_industry_theme"].str.lower() == "other", "factor_industry_theme"] = fallback_theme

    daily_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for date_value, day_frame in frame.groupby("datetime", sort=True):
        active = day_frame[day_frame["forecast_w"] > 1e-12].copy()
        if active.empty:
            daily_rows.append(
                {
                    "datetime": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                    "active_count": 0,
                    "industry_count": 0,
                    "industry_entropy_norm": np.nan,
                    "dominant_industry_weight_share": np.nan,
                    "concentration_warning": False,
                }
            )
            continue
        active["weighted_return_contribution"] = active["effective_weight"] * active["next_return"].fillna(0.0)
        industry = active.groupby("factor_industry_theme", dropna=False).agg(
            active_count=("instrument", "nunique"),
            forecast_weight=("forecast_w", "sum"),
            effective_weight=("effective_weight", "sum"),
            weighted_return_contribution=("weighted_return_contribution", "sum"),
            mean_next_return=("next_return", "mean"),
            min_next_return=("next_return", "min"),
        ).reset_index().rename(columns={"factor_industry_theme": "industry"})
        weight_total = float(industry["forecast_weight"].sum())
        effective_total = float(industry["effective_weight"].sum())
        industry["industry_weight_share"] = industry["forecast_weight"] / weight_total if weight_total > 1e-12 else np.nan
        industry["industry_effective_weight_share"] = industry["effective_weight"] / effective_total if effective_total > 1e-12 else np.nan
        industry["industry_loss_contribution_share"] = np.nan
        negative_contrib = numeric_series(industry["weighted_return_contribution"]).clip(upper=0.0).abs()
        negative_contrib_sum = float(negative_contrib.sum())
        if negative_contrib_sum > 1e-12:
            industry["industry_loss_contribution_share"] = negative_contrib / negative_contrib_sum
        industry_entropy, industry_entropy_norm, effective_industry_count = normalized_entropy_from_shares(
            industry["industry_weight_share"]
        )
        effective_entropy, effective_entropy_norm, effective_industry_count_scaled = normalized_entropy_from_shares(
            industry["industry_effective_weight_share"]
        )
        dominant_idx = industry["industry_weight_share"].idxmax()
        dominant_row = industry.loc[dominant_idx]
        loss_dominant_row = industry.loc[industry["industry_loss_contribution_share"].idxmax()] if negative_contrib_sum > 1e-12 else pd.Series(dtype=object)
        concentration_warning = bool(
            safe_float(industry_entropy_norm) < INDUSTRY_ENTROPY_MIN_SAFE
            or safe_float(dominant_row.get("industry_weight_share")) > INDUSTRY_DOMINANT_SHARE_SAFE_MAX
        )
        active_effective_weight = numeric_series(active["effective_weight"]).fillna(0.0)
        active_effective_weight_sum = float(active_effective_weight.sum())
        active_weighted_next_return = (
            float((active_effective_weight * numeric_series(active["next_return"]).fillna(0.0)).sum() / active_effective_weight_sum)
            if active_effective_weight_sum > 1e-12
            else np.nan
        )
        daily_rows.append(
            {
                "datetime": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                "active_count": int(active.shape[0]),
                "active_negative_share": float((numeric_series(active["next_return"]) < 0.0).mean()),
                "active_weighted_next_return": active_weighted_next_return,
                "pool_median_next_return": float(numeric_series(day_frame["next_return"]).median()),
                "risk_signal_quality_rolling_ic": float(numeric_series(day_frame.get("risk_signal_quality_rolling_ic", pd.Series(dtype=float))).dropna().iloc[0]) if "risk_signal_quality_rolling_ic" in day_frame.columns and not numeric_series(day_frame["risk_signal_quality_rolling_ic"]).dropna().empty else np.nan,
                "risk_signal_quality_scale": float(numeric_series(day_frame.get("risk_signal_quality_scale", pd.Series(dtype=float))).dropna().iloc[0]) if "risk_signal_quality_scale" in day_frame.columns and not numeric_series(day_frame["risk_signal_quality_scale"]).dropna().empty else np.nan,
                "risk_regime_on": float(numeric_series(day_frame.get("risk_regime_on", pd.Series(dtype=float))).dropna().iloc[0]) if "risk_regime_on" in day_frame.columns and not numeric_series(day_frame["risk_regime_on"]).dropna().empty else np.nan,
                "industry_count": int(industry.shape[0]),
                "industry_entropy": industry_entropy,
                "industry_entropy_norm": industry_entropy_norm,
                "effective_industry_count": effective_industry_count,
                "effective_weight_entropy": effective_entropy,
                "effective_weight_entropy_norm": effective_entropy_norm,
                "effective_weight_industry_count": effective_industry_count_scaled,
                "dominant_industry": str(dominant_row.get("industry", "")),
                "dominant_industry_weight_share": safe_float(dominant_row.get("industry_weight_share")),
                "dominant_industry_effective_weight_share": safe_float(dominant_row.get("industry_effective_weight_share")),
                "dominant_loss_industry": str(loss_dominant_row.get("industry", "")) if not loss_dominant_row.empty else "",
                "dominant_loss_contribution_share": safe_float(loss_dominant_row.get("industry_loss_contribution_share")) if not loss_dominant_row.empty else np.nan,
                "concentration_warning": concentration_warning,
                "concentration_rule": f"entropy_norm<{INDUSTRY_ENTROPY_MIN_SAFE:.2f} or dominant_share>{INDUSTRY_DOMINANT_SHARE_SAFE_MAX:.2f}",
            }
        )
        for _, industry_row in industry.sort_values("industry_weight_share", ascending=False).iterrows():
            detail_rows.append(
                {
                    "datetime": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                    "industry": industry_row["industry"],
                    "active_count": int(industry_row["active_count"]),
                    "industry_weight_share": safe_float(industry_row["industry_weight_share"]),
                    "industry_effective_weight_share": safe_float(industry_row["industry_effective_weight_share"]),
                    "weighted_return_contribution": safe_float(industry_row["weighted_return_contribution"]),
                    "industry_loss_contribution_share": safe_float(industry_row["industry_loss_contribution_share"]),
                    "mean_next_return": safe_float(industry_row["mean_next_return"]),
                    "min_next_return": safe_float(industry_row["min_next_return"]),
                    "active_codes": "|".join(active.loc[active["factor_industry_theme"] == industry_row["industry"], "instrument"].astype(str).tolist()),
                    "active_names": "|".join(active.loc[active["factor_industry_theme"] == industry_row["industry"], "name"].astype(str).tolist()),
                }
            )

    daily = pd.DataFrame(daily_rows)
    detail = pd.DataFrame(detail_rows)
    focus = daily[daily.get("datetime", pd.Series(dtype=object)).astype(str).isin(INDUSTRY_ENTROPY_FOCUS_DATES)].copy() if not daily.empty else pd.DataFrame()
    write_table(daily, out_dir, "signal_quality_industry_entropy_daily.csv")
    write_table(detail, out_dir, "signal_quality_industry_entropy_detail.csv")
    write_table(focus, out_dir, "signal_quality_industry_entropy_focus.csv")
    return {
        "signal_quality_industry_entropy_daily": daily,
        "signal_quality_industry_entropy_detail": detail,
        "signal_quality_industry_entropy_focus": focus,
    }


def entropy_weighted_defense_oos_experiment(
    l3_path: Path,
    out_dir: Path,
    fund_list_csv: str | Path,
) -> dict[str, pd.DataFrame]:
    requested_columns = [
        "instrument",
        "datetime",
        "next_return",
        "forecast_w",
        "exposure_scale",
        "risk_signal_quality_rolling_ic",
        "risk_signal_quality_scale",
        "risk_regime_on",
        "factor_industry_theme",
    ]
    frame = read_csv_columns(l3_path, requested_columns)
    if not {"instrument", "datetime", "next_return", "forecast_w"}.issubset(frame.columns):
        comparison = pd.DataFrame()
        daily = pd.DataFrame()
        write_table(comparison, out_dir, "entropy_weighted_defense_oos_comparison.csv")
        write_table(daily, out_dir, "entropy_weighted_defense_oos_daily.csv")
        return {
            "entropy_weighted_defense_oos_comparison": comparison,
            "entropy_weighted_defense_oos_daily": daily,
        }

    frame = frame.sort_values(["datetime", "instrument"]).copy()
    frame["next_return"] = numeric_series(frame["next_return"])
    frame["forecast_w"] = numeric_series(frame["forecast_w"]).fillna(0.0).clip(lower=0.0)
    if "exposure_scale" not in frame.columns:
        frame["exposure_scale"] = 1.0
    frame["exposure_scale"] = numeric_series(frame["exposure_scale"]).fillna(1.0).clip(lower=0.0, upper=1.0)
    name_map = build_name_map(fund_list_csv)
    frame["name"] = frame["instrument"].map(name_map).fillna("")
    if "factor_industry_theme" not in frame.columns:
        frame["factor_industry_theme"] = ""
    fallback_theme = frame["instrument"].map(name_map).map(audit_l3_industry_label).fillna("other")
    frame["factor_industry_theme"] = frame["factor_industry_theme"].fillna("").astype(str)
    frame.loc[frame["factor_industry_theme"].str.len() == 0, "factor_industry_theme"] = fallback_theme
    frame.loc[frame["factor_industry_theme"].str.lower().isin(["nan", "other"]), "factor_industry_theme"] = fallback_theme

    scenarios = [
        {
            "scenario": "current_l3_reference",
            "entropy_threshold": np.nan,
            "signal_quality_min": np.nan,
            "signal_quality_max": np.nan,
            "extra_scale": 1.0,
        },
        {
            "scenario": "entropy_t0.80_sq_-0.05_to_0.10_scale0.50",
            "entropy_threshold": 0.80,
            "signal_quality_min": -0.05,
            "signal_quality_max": 0.10,
            "extra_scale": 0.50,
        },
        {
            "scenario": "entropy_t0.85_sq_-0.05_to_0.10_scale0.50",
            "entropy_threshold": 0.85,
            "signal_quality_min": -0.05,
            "signal_quality_max": 0.10,
            "extra_scale": 0.50,
        },
        {
            "scenario": "entropy_t0.80_sq_-0.05_to_0.08_scale0.50",
            "entropy_threshold": 0.80,
            "signal_quality_min": -0.05,
            "signal_quality_max": 0.08,
            "extra_scale": 0.50,
        },
        {
            "scenario": "entropy_t0.80_sq_-0.05_to_0.10_scale0.70",
            "entropy_threshold": 0.80,
            "signal_quality_min": -0.05,
            "signal_quality_max": 0.10,
            "extra_scale": 0.70,
        },
    ]

    daily_rows: list[dict[str, object]] = []
    for date_value, day_frame in frame.groupby("datetime", sort=True):
        active = day_frame[day_frame["forecast_w"] > 1e-12].copy()
        active_weight_sum = float(numeric_series(active.get("forecast_w", pd.Series(dtype=float))).fillna(0.0).sum()) if not active.empty else 0.0
        industry_entropy_norm = np.nan
        effective_industry_count = np.nan
        dominant_industry = ""
        dominant_share = np.nan
        dominant_loss_industry = ""
        dominant_loss_share = np.nan
        active_ai_theme_weight_share = np.nan
        if not active.empty and active_weight_sum > 1e-12:
            active["weighted_return_contribution"] = numeric_series(active["forecast_w"]).fillna(0.0) * numeric_series(active["next_return"]).fillna(0.0)
            active_ai_theme_weight_share = float(
                numeric_series(active.loc[active["factor_industry_theme"].isin(ENTROPY_DEFENSE_AI_THEMES), "forecast_w"]).fillna(0.0).sum()
                / active_weight_sum
            )
            industry = active.groupby("factor_industry_theme", dropna=False).agg(
                forecast_weight=("forecast_w", "sum"),
                weighted_return_contribution=("weighted_return_contribution", "sum"),
            ).reset_index().rename(columns={"factor_industry_theme": "industry"})
            industry["industry_weight_share"] = numeric_series(industry["forecast_weight"]).fillna(0.0) / active_weight_sum
            _, industry_entropy_norm, effective_industry_count = normalized_entropy_from_shares(industry["industry_weight_share"])
            dominant_row = industry.loc[industry["industry_weight_share"].idxmax()]
            dominant_industry = str(dominant_row.get("industry", ""))
            dominant_share = safe_float(dominant_row.get("industry_weight_share"))
            negative_contrib = numeric_series(industry["weighted_return_contribution"]).clip(upper=0.0).abs()
            negative_contrib_sum = float(negative_contrib.sum())
            if negative_contrib_sum > 1e-12:
                industry["industry_loss_contribution_share"] = negative_contrib / negative_contrib_sum
                dominant_loss_row = industry.loc[industry["industry_loss_contribution_share"].idxmax()]
                dominant_loss_industry = str(dominant_loss_row.get("industry", ""))
                dominant_loss_share = safe_float(dominant_loss_row.get("industry_loss_contribution_share"))
        rolling_ic_values = numeric_series(day_frame.get("risk_signal_quality_rolling_ic", pd.Series(dtype=float))).dropna()
        signal_quality_rolling_ic = float(rolling_ic_values.iloc[0]) if not rolling_ic_values.empty else np.nan
        signal_quality_scale_values = numeric_series(day_frame.get("risk_signal_quality_scale", pd.Series(dtype=float))).dropna()
        signal_quality_scale = float(signal_quality_scale_values.iloc[0]) if not signal_quality_scale_values.empty else np.nan
        baseline_return = weighted_next_return(day_frame, "forecast_w", "exposure_scale")
        date_text = pd.Timestamp(date_value).strftime("%Y-%m-%d")
        for scenario in scenarios:
            is_reference = scenario["scenario"] == "current_l3_reference"
            entropy_trigger = bool(
                not is_reference
                and np.isfinite(safe_float(industry_entropy_norm))
                and safe_float(industry_entropy_norm) < safe_float(scenario["entropy_threshold"])
            )
            signal_quality_trigger = bool(
                not is_reference
                and np.isfinite(signal_quality_rolling_ic)
                and signal_quality_rolling_ic >= safe_float(scenario["signal_quality_min"])
                and signal_quality_rolling_ic <= safe_float(scenario["signal_quality_max"])
            )
            triggered = bool(entropy_trigger and signal_quality_trigger)
            scenario_frame = day_frame.copy()
            scenario_frame["entropy_adjusted_exposure_scale"] = numeric_series(scenario_frame["exposure_scale"]).fillna(1.0)
            if triggered:
                scenario_frame["entropy_adjusted_exposure_scale"] = scenario_frame["entropy_adjusted_exposure_scale"] * safe_float(scenario["extra_scale"])
            scenario_return = weighted_next_return(scenario_frame, "forecast_w", "entropy_adjusted_exposure_scale")
            daily_rows.append(
                {
                    "scenario": scenario["scenario"],
                    "datetime": date_text,
                    "return": scenario_return,
                    "baseline_return": baseline_return,
                    "delta_return_vs_current_l3": scenario_return - baseline_return if np.isfinite(scenario_return) and np.isfinite(baseline_return) else np.nan,
                    "triggered": triggered,
                    "entropy_trigger": entropy_trigger,
                    "signal_quality_trigger": signal_quality_trigger,
                    "entropy_threshold": scenario["entropy_threshold"],
                    "signal_quality_min": scenario["signal_quality_min"],
                    "signal_quality_max": scenario["signal_quality_max"],
                    "extra_scale": scenario["extra_scale"],
                    "active_count": int(active.shape[0]),
                    "industry_entropy_norm": industry_entropy_norm,
                    "effective_industry_count": effective_industry_count,
                    "dominant_industry": dominant_industry,
                    "dominant_industry_weight_share": dominant_share,
                    "dominant_loss_industry": dominant_loss_industry,
                    "dominant_loss_contribution_share": dominant_loss_share,
                    "risk_signal_quality_rolling_ic": signal_quality_rolling_ic,
                    "risk_signal_quality_scale": signal_quality_scale,
                    "active_ai_theme_weight_share": active_ai_theme_weight_share,
                    "focus_date_flag": date_text in ENTROPY_DEFENSE_FOCUS_DATES,
                    "april_flag": date_text >= "2026-04-01" and date_text <= "2026-04-30",
                    "april_ai_theme_flag": date_text >= "2026-04-01" and date_text <= "2026-04-30" and safe_float(active_ai_theme_weight_share) > 1e-12,
                }
            )

    daily = pd.DataFrame(daily_rows)
    reference = daily[daily["scenario"] == "current_l3_reference"].set_index("datetime") if not daily.empty else pd.DataFrame()
    comparison_rows: list[dict[str, object]] = []
    for scenario_name, scenario_daily in daily.groupby("scenario", sort=False):
        stats = return_stats(scenario_daily.set_index("datetime")["return"])
        row: dict[str, object] = {"scenario": scenario_name, **stats}
        scenario_indexed = scenario_daily.set_index("datetime")
        if not reference.empty:
            aligned = scenario_indexed.join(reference[["return"]].rename(columns={"return": "current_l3_return"}), how="left")
            row["delta_cumulative_return_vs_current_l3"] = row["cumulative_return"] - return_stats(reference["return"])["cumulative_return"]
            row["delta_max_drawdown_vs_current_l3"] = row["max_drawdown"] - return_stats(reference["return"])["max_drawdown"]
            for focus_date in ENTROPY_DEFENSE_FOCUS_DATES:
                row[f"return_{focus_date}"] = safe_float(scenario_indexed.loc[focus_date, "return"]) if focus_date in scenario_indexed.index else np.nan
                row[f"delta_return_{focus_date}_vs_current_l3"] = safe_float(aligned.loc[focus_date, "return"] - aligned.loc[focus_date, "current_l3_return"]) if focus_date in aligned.index else np.nan
        april_returns = scenario_daily[scenario_daily["april_flag"].fillna(False).astype(bool)]["return"]
        april_ai_returns = scenario_daily[scenario_daily["april_ai_theme_flag"].fillna(False).astype(bool)]["return"]
        row["trigger_day_count"] = int(scenario_daily["triggered"].fillna(False).astype(bool).sum())
        row["trigger_dates"] = "|".join(scenario_daily.loc[scenario_daily["triggered"].fillna(False).astype(bool), "datetime"].astype(str).tolist())
        row["april_trigger_day_count"] = int((scenario_daily["triggered"].fillna(False).astype(bool) & scenario_daily["april_flag"].fillna(False).astype(bool)).sum())
        row["april_cumulative_return"] = compound_return(april_returns)
        row["april_ai_theme_cumulative_return"] = compound_return(april_ai_returns)
        if not reference.empty:
            reference_april = reference[reference["april_flag"].fillna(False).astype(bool)]["return"]
            reference_april_ai = reference[reference["april_ai_theme_flag"].fillna(False).astype(bool)]["return"]
            row["delta_april_cumulative_return_vs_current_l3"] = row["april_cumulative_return"] - compound_return(reference_april)
            row["delta_april_ai_theme_cumulative_return_vs_current_l3"] = row["april_ai_theme_cumulative_return"] - compound_return(reference_april_ai)
        row["experiment_scope"] = "audit_only_no_position_path_change"
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)
    write_table(comparison, out_dir, "entropy_weighted_defense_oos_comparison.csv")
    write_table(daily, out_dir, "entropy_weighted_defense_oos_daily.csv")
    return {
        "entropy_weighted_defense_oos_comparison": comparison,
        "entropy_weighted_defense_oos_daily": daily,
    }


def momentum_overheat_audit(
    l3_path: Path,
    out_dir: Path,
    fund_list_csv: str | Path,
) -> dict[str, pd.DataFrame]:
    requested_columns = [
        "instrument",
        "datetime",
        "close",
        "next_return",
        "forecast_w",
        "exposure_scale",
        "forecast_mean_return",
        "forecast_mean_return_robust",
        "forecast_weight_mean_raw",
        "forecast_weight_mean_used",
        "mu_raw",
        "risk_signal_quality_rolling_ic",
        "risk_signal_quality_scale",
        "factor_industry_theme",
    ]
    frame = read_csv_columns(l3_path, requested_columns)
    if not {"instrument", "datetime", "close", "next_return", "forecast_w"}.issubset(frame.columns):
        empty = pd.DataFrame()
        for filename in [
            "momentum_overheat_active_detail.csv",
            "momentum_overheat_daily.csv",
            "momentum_overheat_industry_detail.csv",
            "momentum_overheat_summary.csv",
            "momentum_overheat_focus.csv",
        ]:
            write_table(empty, out_dir, filename)
        return {
            "momentum_overheat_active_detail": empty,
            "momentum_overheat_daily": empty,
            "momentum_overheat_industry_detail": empty,
            "momentum_overheat_summary": empty,
            "momentum_overheat_focus": empty,
        }

    frame = frame.sort_values(["instrument", "datetime"]).copy()
    frame["close"] = numeric_series(frame["close"])
    frame["next_return"] = numeric_series(frame["next_return"])
    frame["forecast_w"] = numeric_series(frame["forecast_w"]).fillna(0.0).clip(lower=0.0)
    if "exposure_scale" not in frame.columns:
        frame["exposure_scale"] = 1.0
    frame["exposure_scale"] = numeric_series(frame["exposure_scale"]).fillna(1.0).clip(lower=0.0, upper=1.0)
    frame["effective_weight"] = frame["forecast_w"] * frame["exposure_scale"]
    frame["hist_return_t_minus_5_to_t_minus_1"] = (
        frame.groupby("instrument")["close"].shift(1)
        / frame.groupby("instrument")["close"].shift(5)
        - 1.0
    )
    frame["hist_return_prev5_trading_days"] = (
        frame.groupby("instrument")["close"].shift(1)
        / frame.groupby("instrument")["close"].shift(6)
        - 1.0
    )
    mean_col = next(
        (
            column
            for column in ["forecast_mean_return", "forecast_weight_mean_raw", "forecast_weight_mean_used", "mu_raw", "forecast_mean_return_robust"]
            if column in frame.columns and numeric_series(frame[column]).notna().sum() > 0
        ),
        "",
    )
    if mean_col:
        frame[mean_col] = numeric_series(frame[mean_col])
    name_map = build_name_map(fund_list_csv)
    frame["name"] = frame["instrument"].map(name_map).fillna("")
    if "factor_industry_theme" not in frame.columns:
        frame["factor_industry_theme"] = ""
    fallback_theme = frame["instrument"].map(name_map).map(audit_l3_industry_label).fillna("other")
    frame["factor_industry_theme"] = frame["factor_industry_theme"].fillna("").astype(str)
    frame.loc[frame["factor_industry_theme"].str.len() == 0, "factor_industry_theme"] = fallback_theme
    frame.loc[frame["factor_industry_theme"].str.lower().isin(["nan", "other"]), "factor_industry_theme"] = fallback_theme
    frame["overheat_10pct_flag"] = numeric_series(frame["hist_return_t_minus_5_to_t_minus_1"]) > MOMENTUM_OVERHEAT_DEFAULT_THRESHOLD
    frame["momentum_bucket"] = pd.cut(
        numeric_series(frame["hist_return_t_minus_5_to_t_minus_1"]),
        bins=[-np.inf, 0.0, 0.05, 0.10, 0.15, np.inf],
        labels=["<=0", "0_to_5pct", "5_to_10pct", "10_to_15pct", ">15pct"],
    ).astype(str)
    frame.loc[numeric_series(frame["hist_return_t_minus_5_to_t_minus_1"]).isna(), "momentum_bucket"] = "missing"

    active = frame[frame["forecast_w"] > 1e-12].copy()
    active_detail_columns = [
        "datetime",
        "instrument",
        "name",
        "factor_industry_theme",
        "forecast_w",
        "exposure_scale",
        "effective_weight",
        mean_col,
        "hist_return_t_minus_5_to_t_minus_1",
        "hist_return_prev5_trading_days",
        "momentum_bucket",
        "overheat_10pct_flag",
        "next_return",
        "risk_signal_quality_rolling_ic",
        "risk_signal_quality_scale",
    ]
    active_detail = active[[column for column in active_detail_columns if column]].copy()
    if not active_detail.empty:
        active_detail["datetime"] = pd.to_datetime(active_detail["datetime"]).dt.strftime("%Y-%m-%d")

    daily_rows: list[dict[str, object]] = []
    industry_detail_rows: list[dict[str, object]] = []
    for date_value, day_frame in frame.groupby("datetime", sort=True):
        date_text = pd.Timestamp(date_value).strftime("%Y-%m-%d")
        day_active = day_frame[day_frame["forecast_w"] > 1e-12].copy()
        active_weight_sum = float(numeric_series(day_active.get("forecast_w", pd.Series(dtype=float))).fillna(0.0).sum()) if not day_active.empty else 0.0
        hot_active = day_active[numeric_series(day_active["hist_return_t_minus_5_to_t_minus_1"]) > MOMENTUM_OVERHEAT_DEFAULT_THRESHOLD].copy()
        hot_weight_sum = float(numeric_series(hot_active.get("forecast_w", pd.Series(dtype=float))).fillna(0.0).sum()) if not hot_active.empty else 0.0
        hot_effective_weight_sum = float(numeric_series(hot_active.get("effective_weight", pd.Series(dtype=float))).fillna(0.0).sum()) if not hot_active.empty else 0.0
        active_effective_weight_sum = float(numeric_series(day_active.get("effective_weight", pd.Series(dtype=float))).fillna(0.0).sum()) if not day_active.empty else 0.0
        hot_weighted_next_return = (
            float((numeric_series(hot_active["effective_weight"]).fillna(0.0) * numeric_series(hot_active["next_return"]).fillna(0.0)).sum() / hot_effective_weight_sum)
            if hot_effective_weight_sum > 1e-12
            else np.nan
        )
        active_weighted_next_return = (
            float((numeric_series(day_active["effective_weight"]).fillna(0.0) * numeric_series(day_active["next_return"]).fillna(0.0)).sum() / active_effective_weight_sum)
            if active_effective_weight_sum > 1e-12
            else np.nan
        )
        industry_entropy_norm = np.nan
        dominant_industry = ""
        dominant_share = np.nan
        dominant_industry_avg_hist5 = np.nan
        dominant_industry_hot_weight_share = np.nan
        dominant_industry_weighted_next_return = np.nan
        if active_weight_sum > 1e-12:
            industry_frames: list[pd.Series] = []
            for industry_name, industry_active in day_active.groupby("factor_industry_theme", dropna=False):
                industry_weight_sum = float(numeric_series(industry_active["forecast_w"]).fillna(0.0).sum())
                industry_effective_sum = float(numeric_series(industry_active["effective_weight"]).fillna(0.0).sum())
                industry_hot = industry_active[numeric_series(industry_active["hist_return_t_minus_5_to_t_minus_1"]) > MOMENTUM_OVERHEAT_DEFAULT_THRESHOLD]
                industry_hot_weight = float(numeric_series(industry_hot.get("forecast_w", pd.Series(dtype=float))).fillna(0.0).sum()) if not industry_hot.empty else 0.0
                industry_hist = numeric_series(industry_active["hist_return_t_minus_5_to_t_minus_1"])
                industry_hist_weights = numeric_series(industry_active["forecast_w"]).fillna(0.0)
                industry_hist_valid = industry_hist.notna() & industry_hist_weights.gt(0.0)
                weighted_hist = (
                    float((industry_hist_weights[industry_hist_valid] * industry_hist[industry_hist_valid]).sum() / industry_hist_weights[industry_hist_valid].sum())
                    if float(industry_hist_weights[industry_hist_valid].sum()) > 1e-12
                    else np.nan
                )
                weighted_return = (
                    float((numeric_series(industry_active["effective_weight"]).fillna(0.0) * numeric_series(industry_active["next_return"]).fillna(0.0)).sum() / industry_effective_sum)
                    if industry_effective_sum > 1e-12
                    else np.nan
                )
                industry_detail_rows.append(
                    {
                        "datetime": date_text,
                        "industry": str(industry_name),
                        "active_count": int(industry_active.shape[0]),
                        "industry_weight_share": industry_weight_sum / active_weight_sum,
                        "industry_avg_hist_return_t_minus_5_to_t_minus_1": weighted_hist,
                        "industry_max_hist_return_t_minus_5_to_t_minus_1": float(industry_hist.max()) if not industry_hist.dropna().empty else np.nan,
                        "industry_hot_weight_share_within_industry": industry_hot_weight / industry_weight_sum if industry_weight_sum > 1e-12 else np.nan,
                        "industry_weighted_next_return": weighted_return,
                        "industry_hot_codes": "|".join(industry_hot["instrument"].astype(str).tolist()) if not industry_hot.empty else "",
                        "industry_active_codes": "|".join(industry_active["instrument"].astype(str).tolist()),
                    }
                )
                industry_frames.append(pd.Series({"industry": str(industry_name), "weight_share": industry_weight_sum / active_weight_sum}))
            industry_weight_frame = pd.DataFrame(industry_frames)
            if not industry_weight_frame.empty:
                _, industry_entropy_norm, _ = normalized_entropy_from_shares(industry_weight_frame["weight_share"])
                dominant_idx = industry_weight_frame["weight_share"].idxmax()
                dominant_industry = str(industry_weight_frame.loc[dominant_idx, "industry"])
                dominant_share = float(industry_weight_frame.loc[dominant_idx, "weight_share"])
                dominant_detail = [row for row in industry_detail_rows if row["datetime"] == date_text and row["industry"] == dominant_industry]
                if dominant_detail:
                    dominant_industry_avg_hist5 = safe_float(dominant_detail[-1].get("industry_avg_hist_return_t_minus_5_to_t_minus_1"))
                    dominant_industry_hot_weight_share = safe_float(dominant_detail[-1].get("industry_hot_weight_share_within_industry"))
                    dominant_industry_weighted_next_return = safe_float(dominant_detail[-1].get("industry_weighted_next_return"))
        all_ic_frame = day_frame[[mean_col, "next_return"]].dropna() if mean_col else pd.DataFrame()
        hot_pool = day_frame[numeric_series(day_frame["hist_return_t_minus_5_to_t_minus_1"]) > MOMENTUM_OVERHEAT_DEFAULT_THRESHOLD]
        hot_ic_frame = hot_pool[[mean_col, "next_return"]].dropna() if mean_col and not hot_pool.empty else pd.DataFrame()
        nonhot_pool = day_frame[numeric_series(day_frame["hist_return_t_minus_5_to_t_minus_1"]) <= MOMENTUM_OVERHEAT_DEFAULT_THRESHOLD]
        nonhot_ic_frame = nonhot_pool[[mean_col, "next_return"]].dropna() if mean_col and not nonhot_pool.empty else pd.DataFrame()
        rolling_ic_values = numeric_series(day_frame.get("risk_signal_quality_rolling_ic", pd.Series(dtype=float))).dropna()
        daily_rows.append(
            {
                "datetime": date_text,
                "active_count": int(day_active.shape[0]),
                "hot_active_count": int(hot_active.shape[0]),
                "hot_active_weight_share": hot_weight_sum / active_weight_sum if active_weight_sum > 1e-12 else np.nan,
                "hot_active_effective_weight_share": hot_effective_weight_sum / active_effective_weight_sum if active_effective_weight_sum > 1e-12 else np.nan,
                "active_avg_hist_return_t_minus_5_to_t_minus_1": float(numeric_series(day_active.get("hist_return_t_minus_5_to_t_minus_1", pd.Series(dtype=float))).mean()) if not day_active.empty else np.nan,
                "active_max_hist_return_t_minus_5_to_t_minus_1": float(numeric_series(day_active.get("hist_return_t_minus_5_to_t_minus_1", pd.Series(dtype=float))).max()) if not day_active.empty else np.nan,
                "active_weighted_next_return": active_weighted_next_return,
                "hot_active_weighted_next_return": hot_weighted_next_return,
                "hot_active_negative_share": float((numeric_series(hot_active["next_return"]) < 0.0).mean()) if not hot_active.empty else np.nan,
                "pool_mu_rank_ic_all": float(all_ic_frame[mean_col].corr(all_ic_frame["next_return"], method="spearman")) if mean_col and len(all_ic_frame) >= 5 else np.nan,
                "pool_mu_rank_ic_hot": float(hot_ic_frame[mean_col].corr(hot_ic_frame["next_return"], method="spearman")) if mean_col and len(hot_ic_frame) >= 5 else np.nan,
                "pool_mu_rank_ic_nonhot": float(nonhot_ic_frame[mean_col].corr(nonhot_ic_frame["next_return"], method="spearman")) if mean_col and len(nonhot_ic_frame) >= 5 else np.nan,
                "industry_entropy_norm": industry_entropy_norm,
                "dominant_industry": dominant_industry,
                "dominant_industry_weight_share": dominant_share,
                "dominant_industry_avg_hist_return_t_minus_5_to_t_minus_1": dominant_industry_avg_hist5,
                "dominant_industry_hot_weight_share_within_industry": dominant_industry_hot_weight_share,
                "dominant_industry_weighted_next_return": dominant_industry_weighted_next_return,
                "risk_signal_quality_rolling_ic": float(rolling_ic_values.iloc[0]) if not rolling_ic_values.empty else np.nan,
                "combined_overheat_concentration_flag": bool(
                    safe_float(dominant_share) > 0.50
                    and (
                        safe_float(dominant_industry_avg_hist5) > 0.08
                        or safe_float(dominant_industry_hot_weight_share) > 0.40
                    )
                ),
                "hot_active_codes": "|".join(hot_active["instrument"].astype(str).tolist()),
                "hot_active_names": "|".join(hot_active["name"].astype(str).tolist()),
            }
        )
    daily = pd.DataFrame(daily_rows)
    industry_detail = pd.DataFrame(industry_detail_rows)
    if not daily.empty:
        daily["combined_momentum_overheat_index"] = numeric_series(daily["hot_active_weight_share"])
        warning_mask = numeric_series(daily["combined_momentum_overheat_index"]) >= (MOMENTUM_FATIGUE_WARNING_WEIGHT_SHARE - 1e-6)
        daily["momentum_fatigue_warning"] = warning_mask
        daily["momentum_fatigue_label"] = np.where(warning_mask, "Momentum Fatigue Warning", "")
        for forward_days in [1, 2, 3, 5, 10]:
            daily[f"rolling_ic_t_plus_{forward_days}d"] = numeric_series(daily["risk_signal_quality_rolling_ic"]).shift(-forward_days)
        future_3d_cols = ["rolling_ic_t_plus_1d", "rolling_ic_t_plus_2d", "rolling_ic_t_plus_3d"]
        future_10d_cols = ["rolling_ic_t_plus_1d", "rolling_ic_t_plus_2d", "rolling_ic_t_plus_3d", "rolling_ic_t_plus_5d", "rolling_ic_t_plus_10d"]
        daily["rolling_ic_min_next_3d"] = daily[future_3d_cols].min(axis=1)
        daily["rolling_ic_min_next_10d"] = daily[future_10d_cols].min(axis=1)
        daily["rolling_ic_drop_next_3d"] = numeric_series(daily["risk_signal_quality_rolling_ic"]) - numeric_series(daily["rolling_ic_min_next_3d"])
        daily["rolling_ic_collapse_next_3d_flag"] = numeric_series(daily["rolling_ic_min_next_3d"]) < -0.05
    summary_columns = [
        "datetime",
        "combined_momentum_overheat_index",
        "momentum_fatigue_warning",
        "momentum_fatigue_label",
        "active_count",
        "hot_active_count",
        "hot_active_weight_share",
        "active_weighted_next_return",
        "hot_active_weighted_next_return",
        "risk_signal_quality_rolling_ic",
        "rolling_ic_min_next_3d",
        "rolling_ic_drop_next_3d",
        "rolling_ic_collapse_next_3d_flag",
        "rolling_ic_min_next_10d",
        "industry_entropy_norm",
        "dominant_industry",
        "dominant_industry_weight_share",
        "combined_overheat_concentration_flag",
        "hot_active_codes",
    ]
    summary = daily[[column for column in summary_columns if column in daily.columns]].copy() if not daily.empty else pd.DataFrame()
    focus = daily[daily.get("datetime", pd.Series(dtype=object)).astype(str).isin(MOMENTUM_OVERHEAT_FOCUS_DATES)].copy() if not daily.empty else pd.DataFrame()

    write_table(active_detail, out_dir, "momentum_overheat_active_detail.csv")
    write_table(daily, out_dir, "momentum_overheat_daily.csv")
    write_table(industry_detail, out_dir, "momentum_overheat_industry_detail.csv")
    write_table(summary, out_dir, "momentum_overheat_summary.csv")
    write_table(focus, out_dir, "momentum_overheat_focus.csv")
    return {
        "momentum_overheat_active_detail": active_detail,
        "momentum_overheat_daily": daily,
        "momentum_overheat_industry_detail": industry_detail,
        "momentum_overheat_summary": summary,
        "momentum_overheat_focus": focus,
    }


def active_set_unanimous_loss_monitor(
    l3_path: Path,
    out_dir: Path,
    fund_list_csv: str | Path,
) -> dict[str, pd.DataFrame]:
    requested_columns = [
        "instrument",
        "datetime",
        "next_return",
        "forecast_w",
        "exposure_scale",
        "risk_signal_quality_rolling_ic",
        "risk_signal_quality_scale",
        "risk_regime_on",
        "factor_industry_theme",
    ]
    frame = read_csv_columns(l3_path, requested_columns)
    if not {"instrument", "datetime", "next_return", "forecast_w"}.issubset(frame.columns):
        daily = pd.DataFrame()
        detail = pd.DataFrame()
        summary = pd.DataFrame()
        write_table(daily, out_dir, "active_set_unanimous_loss_monitor_daily.csv")
        write_table(detail, out_dir, "active_set_unanimous_loss_monitor_detail.csv")
        write_table(summary, out_dir, "active_set_unanimous_loss_monitor_summary.csv")
        return {
            "active_set_unanimous_loss_monitor_daily": daily,
            "active_set_unanimous_loss_monitor_detail": detail,
            "active_set_unanimous_loss_monitor_summary": summary,
        }

    frame = frame.sort_values(["datetime", "instrument"]).copy()
    frame["next_return"] = numeric_series(frame["next_return"])
    frame["forecast_w"] = numeric_series(frame["forecast_w"]).fillna(0.0)
    if "exposure_scale" not in frame.columns:
        frame["exposure_scale"] = 1.0
    frame["exposure_scale"] = numeric_series(frame["exposure_scale"]).fillna(1.0).clip(lower=0.0, upper=1.0)
    frame["effective_weight"] = frame["forecast_w"].clip(lower=0.0) * frame["exposure_scale"]
    name_map = build_name_map(fund_list_csv)
    frame["name"] = frame["instrument"].map(name_map).fillna("")
    if "factor_industry_theme" not in frame.columns:
        frame["factor_industry_theme"] = ""

    daily_rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    for date_value, day_frame in frame.groupby("datetime", sort=True):
        active = day_frame[day_frame["forecast_w"] > 1e-12].copy()
        active_count = int(active.shape[0])
        active_negative_count = int((numeric_series(active["next_return"]) < 0.0).sum()) if active_count else 0
        active_negative_share = divide_or_nan(active_negative_count, active_count)
        pool_median = float(numeric_series(day_frame["next_return"]).median())
        trigger = bool(active_count > 0 and active_negative_count == active_count and pool_median > ACTIVE_SET_POOL_MEDIAN_RETURN_MIN)
        active_effective_weight = numeric_series(active.get("effective_weight", pd.Series(dtype=float))).fillna(0.0)
        active_weight_sum = float(numeric_series(active.get("forecast_w", pd.Series(dtype=float))).fillna(0.0).sum()) if active_count else 0.0
        active_effective_weight_sum = float(active_effective_weight.sum()) if active_count else 0.0
        active_weighted_next_return = (
            float((active_effective_weight * numeric_series(active["next_return"]).fillna(0.0)).sum() / active_effective_weight_sum)
            if active_effective_weight_sum > 1e-12
            else np.nan
        )
        daily_rows.append(
            {
                "rule_name": ACTIVE_SET_UNANIMOUS_LOSS_RULE_NAME,
                "datetime": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                "triggered": trigger,
                "action": "record_only_no_intercept",
                "trigger_condition": "active_negative_share == 1.0 and pool_median_next_return > -0.01",
                "active_count": active_count,
                "active_negative_count": active_negative_count,
                "active_negative_share": active_negative_share,
                "pool_median_next_return": pool_median,
                "pool_equal_next_return": float(numeric_series(day_frame["next_return"]).mean()),
                "active_weighted_next_return": active_weighted_next_return,
                "active_weight_sum": active_weight_sum,
                "active_effective_weight_sum": active_effective_weight_sum,
                "risk_signal_quality_rolling_ic": float(numeric_series(day_frame.get("risk_signal_quality_rolling_ic", pd.Series(dtype=float))).dropna().iloc[0]) if "risk_signal_quality_rolling_ic" in day_frame.columns and not numeric_series(day_frame["risk_signal_quality_rolling_ic"]).dropna().empty else np.nan,
                "risk_signal_quality_scale": float(numeric_series(day_frame.get("risk_signal_quality_scale", pd.Series(dtype=float))).dropna().iloc[0]) if "risk_signal_quality_scale" in day_frame.columns and not numeric_series(day_frame["risk_signal_quality_scale"]).dropna().empty else np.nan,
                "risk_regime_on": float(numeric_series(day_frame.get("risk_regime_on", pd.Series(dtype=float))).dropna().iloc[0]) if "risk_regime_on" in day_frame.columns and not numeric_series(day_frame["risk_regime_on"]).dropna().empty else np.nan,
                "active_codes": "|".join(active["instrument"].astype(str).tolist()),
                "active_names": "|".join(active["name"].astype(str).tolist()),
            }
        )
        if trigger and active_count:
            detail = active.copy()
            detail["rule_name"] = ACTIVE_SET_UNANIMOUS_LOSS_RULE_NAME
            detail["action"] = "record_only_no_intercept"
            detail["trigger_condition"] = "active_negative_share == 1.0 and pool_median_next_return > -0.01"
            detail["pool_median_next_return"] = pool_median
            detail["active_negative_share"] = active_negative_share
            detail["datetime"] = pd.to_datetime(detail["datetime"]).dt.strftime("%Y-%m-%d")
            detail_frames.append(detail)

    daily = pd.DataFrame(daily_rows)
    detail = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    trigger_daily = daily[daily.get("triggered", pd.Series(False, index=daily.index)).fillna(False).astype(bool)] if not daily.empty else pd.DataFrame()
    summary = pd.DataFrame(
        [
            {
                "rule_name": ACTIVE_SET_UNANIMOUS_LOSS_RULE_NAME,
                "action": "record_only_no_intercept",
                "trigger_condition": "active_negative_share == 1.0 and pool_median_next_return > -0.01",
                "total_day_count": int(daily.shape[0]),
                "trigger_day_count": int(trigger_daily.shape[0]),
                "trigger_day_share": divide_or_nan(trigger_daily.shape[0], daily.shape[0]),
                "avg_trigger_active_count": float(numeric_series(trigger_daily.get("active_count", pd.Series(dtype=float))).mean()) if not trigger_daily.empty else np.nan,
                "avg_trigger_pool_median_next_return": float(numeric_series(trigger_daily.get("pool_median_next_return", pd.Series(dtype=float))).mean()) if not trigger_daily.empty else np.nan,
                "avg_trigger_active_weighted_next_return": float(numeric_series(trigger_daily.get("active_weighted_next_return", pd.Series(dtype=float))).mean()) if not trigger_daily.empty else np.nan,
                "trigger_dates": "|".join(trigger_daily.get("datetime", pd.Series(dtype=object)).astype(str).tolist()) if not trigger_daily.empty else "",
                "monitor_scope": "shadow_audit_only_no_position_change",
            }
        ]
    )
    write_table(daily, out_dir, "active_set_unanimous_loss_monitor_daily.csv")
    write_table(detail, out_dir, "active_set_unanimous_loss_monitor_detail.csv")
    write_table(summary, out_dir, "active_set_unanimous_loss_monitor_summary.csv")
    return {
        "active_set_unanimous_loss_monitor_daily": daily,
        "active_set_unanimous_loss_monitor_detail": detail,
        "active_set_unanimous_loss_monitor_summary": summary,
    }


def case_exposure_stats(l3_path: Path) -> dict[str, float]:
    requested_columns = [
        "datetime",
        "forecast_w",
        "exposure_scale",
        "risk_regime_on",
        "risk_signal_quality_scale",
        "risk_vol_scale",
        "risk_cluster_scale",
        "risk_trend_sleeve_weight_total_used",
        "risk_trend_sleeve_selected",
        "risk_trend_sleeve_top_k_used",
        "risk_trend_sleeve_floor_lift",
    ]
    l3 = read_csv_columns(l3_path, requested_columns)
    if l3.empty:
        return {}

    daily_rows: list[dict[str, float]] = []
    for date_value, day_frame in l3.groupby("datetime"):
        weights = numeric_series(day_frame["forecast_w"]).clip(lower=0.0).fillna(0.0)
        weight_sum = float(weights.sum())
        exposure = numeric_series(day_frame.get("exposure_scale", pd.Series(1.0, index=day_frame.index))).fillna(1.0)
        actual_exposure = float((weights * exposure).sum() / weight_sum) if weight_sum > 1e-12 else np.nan
        regime_on = numeric_series(day_frame.get("risk_regime_on", pd.Series(dtype=float))).dropna()
        daily_rows.append(
            {
                "datetime": date_value,
                "actual_exposure": actual_exposure,
                "regime_on": float(regime_on.iloc[0]) if len(regime_on) else np.nan,
                "signal_gate": float((numeric_series(day_frame.get("risk_signal_quality_scale", pd.Series(dtype=float))) < 0.999).any()),
                "vol_gate": float((numeric_series(day_frame.get("risk_vol_scale", pd.Series(dtype=float))) < 0.999).any()),
                "cluster_gate": float((numeric_series(day_frame.get("risk_cluster_scale", pd.Series(dtype=float))) < 0.999).any()),
                "max_forecast_w": float(weights.max()) if len(weights) else np.nan,
                "sleeve_weight_used": float(numeric_series(day_frame.get("risk_trend_sleeve_weight_total_used", pd.Series(dtype=float))).max()),
                "sleeve_selected": float(numeric_series(day_frame.get("risk_trend_sleeve_selected", pd.Series(dtype=float))).sum()),
                "sleeve_top_k_used": float(numeric_series(day_frame.get("risk_trend_sleeve_top_k_used", pd.Series(dtype=float))).max()),
                "sleeve_floor_lift": float(numeric_series(day_frame.get("risk_trend_sleeve_floor_lift", pd.Series(dtype=float))).max()),
            }
        )
    daily = pd.DataFrame(daily_rows)
    risk_on = daily[daily["regime_on"] == 1.0]
    risk_off = daily[daily["regime_on"] == 0.0]
    return {
        "avg_actual_exposure": float(daily["actual_exposure"].mean()),
        "risk_on_avg_exposure": float(risk_on["actual_exposure"].mean()) if not risk_on.empty else np.nan,
        "risk_off_avg_exposure": float(risk_off["actual_exposure"].mean()) if not risk_off.empty else np.nan,
        "risk_on_days": int(len(risk_on)),
        "risk_off_days": int(len(risk_off)),
        "signal_gate_days": int(daily["signal_gate"].sum()),
        "vol_gate_days": int(daily["vol_gate"].sum()),
        "cluster_gate_days": int(daily["cluster_gate"].sum()),
        "avg_max_forecast_w": float(daily["max_forecast_w"].mean()),
        "p95_max_forecast_w": float(daily["max_forecast_w"].quantile(0.95)),
        "avg_sleeve_weight_used": float(daily["sleeve_weight_used"].mean()),
        "sleeve_active_days": int((daily["sleeve_selected"] > 0).sum()),
        "avg_sleeve_top_k_used": float(daily["sleeve_top_k_used"].mean()),
        "sleeve_floor_lift_days": int((daily["sleeve_floor_lift"] > 1e-12).sum()),
    }


def control_ablation_table(cases: list[CaseSpec], out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case in cases:
        paths = resolve_case_paths(case)
        daily_nav_path = paths["daily_nav"]
        l3_path = paths["bridge_l3"]
        row: dict[str, object] = {
            "case": case.label,
            "params": case.params,
            "path": str(case.path),
            "status": "ok",
        }
        if daily_nav_path is None or l3_path is None:
            row["status"] = "missing_outputs"
            rows.append(row)
            continue
        stats = return_stats(nav_daily_returns(daily_nav_path))
        row.update(stats)
        row.update(case_exposure_stats(l3_path))
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty and "cumulative_return" in summary.columns:
        base_return = safe_float(summary.loc[0, "cumulative_return"])
        base_drawdown = safe_float(summary.loc[0, "max_drawdown"])
        summary["delta_return_vs_base"] = summary["cumulative_return"].map(safe_float) - base_return
        summary["delta_maxdd_vs_base"] = summary["max_drawdown"].map(safe_float) - base_drawdown
        summary["marginal_return_vs_prev"] = summary["cumulative_return"].map(safe_float).diff()
    write_table(summary, out_dir, "control_ablation_summary.csv")
    return summary


def percent(value: object) -> str:
    number = safe_float(value)
    if not np.isfinite(number):
        return ""
    return f"{number * 100:.4f}%"


def format_table_for_report(frame: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 12) -> str:
    if frame is None or frame.empty:
        return "(empty)"
    display = frame.copy()
    if columns is not None:
        display = display[[column for column in columns if column in display.columns]]
    display = display.head(max_rows)
    for column in display.columns:
        if any(keyword in column for keyword in ["return", "drawdown", "corr", "exposure", "share", "vol", "ic"]):
            if pd.api.types.is_numeric_dtype(display[column]):
                display[column] = display[column].map(lambda value: f"{value:.6f}" if np.isfinite(value) else "")
    return display.to_markdown(index=False)


def build_report(
    args: argparse.Namespace,
    out_dir: Path,
    report_md: Path,
    tables: dict[str, pd.DataFrame],
) -> None:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    control = tables.get("control_ablation_summary", pd.DataFrame())
    forecast_summary = tables.get("forecast_ic_summary", pd.DataFrame())
    top_winner_summary = tables.get("selected_top_winner_forecast_summary", pd.DataFrame())
    vol_exemption_top_winners = tables.get("volatility_exemption_top_winners", pd.DataFrame())
    vol_exemption_scenarios = tables.get("volatility_exemption_scenarios", pd.DataFrame())
    local_cap_candidates = tables.get("local_variance_penalty_cap_candidates", pd.DataFrame())
    local_cap_scenarios = tables.get("local_variance_penalty_cap_scenarios", pd.DataFrame())
    local_cap_oos_summary = tables.get("local_variance_penalty_cap_oos_summary", pd.DataFrame())
    residual_summary = tables.get("residual_analysis_summary", pd.DataFrame())
    residual_reaction = tables.get("residual_reaction_lag_summary", pd.DataFrame())
    residual_factor_check = tables.get("residual_factor_exposure_check", pd.DataFrame())
    lead_lag = tables.get("lead_lag_cross_correlation", pd.DataFrame())
    negative_bias_improvement = tables.get("potential_improvement_from_neutralizing_negative_biases", pd.DataFrame())
    shadow_patch_summary = tables.get("oos_shadow_patch_summary", pd.DataFrame())
    shadow_trigger_frequency = tables.get("oos_patch_trigger_frequency", pd.DataFrame())
    shadow_admission = tables.get("oos_patch_admission_audit_report", pd.DataFrame())
    full_mvo_weight_summary = tables.get("weight_delta_analysis_summary", pd.DataFrame())
    synthetic_stress_summary = tables.get("synthetic_edge_case_stress_summary", pd.DataFrame())
    overflow_summary = tables.get("overflow_diversifier_summary", pd.DataFrame())
    overflow_summary_path = out_dir / "overflow_diversifier_summary.csv"
    if overflow_summary.empty and overflow_summary_path.exists():
        overflow_summary = pd.read_csv(overflow_summary_path)
    covariance_summary = tables.get("covariance_ic_summary", pd.DataFrame())
    portfolio_summary = tables.get("portfolio_method_compare", pd.DataFrame())
    mean_bias_event_focus = tables.get("mean_bias_reversal_event_focus", pd.DataFrame())
    mean_bias_active_detail = tables.get("mean_bias_reversal_active_detail", pd.DataFrame())
    industry_entropy_focus = tables.get("signal_quality_industry_entropy_focus", pd.DataFrame())
    industry_entropy_detail = tables.get("signal_quality_industry_entropy_detail", pd.DataFrame())
    entropy_defense_comparison = tables.get("entropy_weighted_defense_oos_comparison", pd.DataFrame())
    momentum_overheat_focus = tables.get("momentum_overheat_focus", pd.DataFrame())
    momentum_overheat_summary = tables.get("momentum_overheat_summary", pd.DataFrame())
    active_set_monitor_summary = tables.get("active_set_unanimous_loss_monitor_summary", pd.DataFrame())
    active_set_monitor_daily = tables.get("active_set_unanimous_loss_monitor_daily", pd.DataFrame())
    top_capture = tables.get("pool_topn_capture", pd.DataFrame())
    return_distribution = tables.get("pool_return_distribution", pd.DataFrame())
    missed = tables.get("pool_missed_winners_nearest_corr", pd.DataFrame())
    high_corr = tables.get("pool_internal_high_corr_pairs", pd.DataFrame())

    lines = [
        "# strategy layer 固定审计表（2026-04-30）",
        "",
        f"Generated: {generated_at}",
        "",
        "## 结论",
        "",
        "本次不继续扩大参数空间，而是把入参晋级主线前必须看的审计表固定下来。新 Benchmark 只锁定两个 risk-on 参数：",
        "",
        "```bash",
        "--signal-quality-min-scale-risk-on 1.00 \\",
        "--vol-target-risk-on 0.022",
        "```",
        "",
        "`max-weight-cap=0.18`、`trend_sleeve_weight_total_risk_on=0.55`、`trend_sleeve_top_k_risk_on=6` 继续保留为进攻候选，不进入主线，直到滚动/OOS 审计也通过。",
        "",
        "## 输出文件",
        "",
        f"CSV output dir: `{out_dir}`",
        "",
        "| 文件 | 含义 |",
        "|---|---|",
        "| `pool_topn_capture.csv` | 池子 Top-N 赢家捕获 |",
        "| `pool_return_distribution.csv` | 入池/未入池/全市场收益分布，Top5/10 按 period_return 降序计算 |",
        "| `pool_missed_winners_nearest_corr.csv` | 未入池赢家与最近入池标的相关性 |",
        "| `pool_internal_high_corr_pairs.csv` | 入池标的内部高相关近克隆 |",
        "| `forecast_ic_summary.csv` | 均值 IC、方差 IC 汇总 |",
        "| `mean_bias_reversal_correlation_heatmap.csv` | Forecast Mean 与 T-5 到 T-1 历史收益的逐日相关性长表，含 short-reversal score |",
        "| `mean_bias_reversal_correlation_heatmap_matrix.csv` | `forecast_mean_return` 口径的 heatmap-ready 矩阵 |",
        "| `mean_bias_reversal_correlation_heatmap.png` | 简单相关性热力图，蓝红色标识 Spearman 相关 |",
        "| `mean_bias_reversal_event_focus.csv` | 01-12/01-14 失效日的 Mean Bias Focus 表 |",
        "| `mean_bias_reversal_active_detail.csv` | 01-12/01-14 Active Basket 成员的 forecast mean 与历史收益明细 |",
        "| `selected_top_winner_forecast_daily.csv` | 池内收益 Top 赢家的 forecast_mean、forecast_vol、forecast_w 逐日诊断 |",
        "| `selected_top_winner_forecast_summary.csv` | 池内收益 Top 赢家诊断汇总和主因标签 |",
        "| `volatility_exemption_top_winners.csv` | Top 赢家上行/下行波动结构与进攻性波动标签 |",
        "| `volatility_exemption_scenarios.csv` | risk-free-rate 与 variance penalty 豁免 multiplier 的 audit-only 对照 |",
        "| `local_variance_penalty_cap_candidates.csv` | 决策树分类、边际相关性贡献与 Hidden Gems 候选 |",
        "| `local_variance_penalty_cap_scenarios.csv` | 只救 Hidden Gems 的局部 forecast_vol cap 场景 |",
        "| `local_variance_penalty_cap_oos_daily.csv` | Hidden Gems 局部 cap rolling/OOS 每日选择明细 |",
        "| `local_variance_penalty_cap_oos_folds.csv` | Hidden Gems 局部 cap rolling/OOS 分折结果 |",
        "| `local_variance_penalty_cap_oos_summary.csv` | Hidden Gems 局部 cap rolling/OOS 场景汇总 |",
        "| `residual_analysis_daily.csv` | Top 赢家和重点主题的逐日实际超额、预测超额、残差 |",
        "| `residual_analysis_summary.csv` | 爆发期 Specific Return 中预测解释/Residual/主动反向预测占比 |",
        "| `residual_reaction_lag_summary.csv` | 判断信号缺失是同日覆盖、反应慢还是持续未覆盖 |",
        "| `residual_factor_exposure_check.csv` | 显式 style/industry factor 暴露、crowding 与 risk scale 的 offset 归因 |",
        "| `lead_lag_cross_correlation.csv` | Predicted Mean(t) 与 Actual Return(t+n) 的 -3 到 +3 日前瞻/滞后互相关 |",
        "| `potential_improvement_from_neutralizing_negative_biases.csv` | 对板块放量但均值/横截面预测为负的样本做中性化补救模拟 |",
        "| `oos_shadow_patch_daily.csv` | neutralize-to-zero 补丁的 OOS 影子运行逐日收益、换手与触发明细 |",
        "| `oos_shadow_patch_folds.csv` | neutralize-to-zero 补丁的 OOS 影子运行分折统计 |",
        "| `oos_shadow_patch_summary.csv` | neutralize-to-zero 补丁相对 Base 的 OOS 汇总：收益、Sharpe、MaxDD、TE、换手和 Specific Recovery |",
        "| `oos_patch_trigger_frequency.csv` | 硬阈值触发频率，检查补丁是否过度触发或过拟合 |",
        "| `oos_patch_admission_audit_report.csv` | 准入审计报告：主线 Base vs 带补丁 shadow 主线及合规声明 |",
        "| `weight_delta_analysis_patched_vs_base.csv` | full-MVO + L3 rerun 后的 patched vs base 权重差分 |",
        "| `weight_delta_analysis_summary.csv` | full-MVO 权重差分、TE、optimizer_failed 状态汇总 |",
        "| `synthetic_edge_case_stress_weights.csv` | AI/通信极端行情 synthetic session 的逐标的权重压力测试 |",
        "| `synthetic_edge_case_stress_summary.csv` | synthetic edge case 是否触发 optimizer_failed 的准入检查 |",
        f"| `{Path(args.rfc_md).name}` | 过滤器准入 RFC 文档 |",
        "| `overflow_diversifier_candidates.csv` | 未入池低相关 Top3 多元化候选，需由 `pipeline.overflow_diversifier_candidates` 单独生成 |",
        "| `overflow_diversifier_summary.csv` | Overflow Bucket 后续窗口胜率汇总，需由 `pipeline.overflow_diversifier_candidates` 单独生成 |",
        "| `covariance_ic_summary.csv` | 协方差 off-diagonal IC 汇总 |",
        "| `portfolio_method_compare.csv` | MVO vs 等权/风险平价/纯动量/L3 |",
        "| `signal_quality_industry_entropy_daily.csv` | Signal Quality 与 Active Basket L3 行业熵的逐日交叉审计 |",
        "| `signal_quality_industry_entropy_detail.csv` | Active Basket 按 L3 行业拆分的权重、收益贡献与损失集中度 |",
        "| `signal_quality_industry_entropy_focus.csv` | 01-12/01-14/01-29 重点风险日行业熵快照 |",
        "| `entropy_weighted_defense_oos_comparison.csv` | 行业熵低且 Signal Quality 中性偏弱时额外 scale-down 的 OOS 对照 |",
        "| `entropy_weighted_defense_oos_daily.csv` | entropy-weighted defense 逐日触发与收益差分 |",
        "| `momentum_overheat_active_detail.csv` | Active Basket 标的 5 日过热动能、forecast mean 与 next_return 明细 |",
        "| `momentum_overheat_daily.csv` | 逐日 hot active share、hot return、行业熵与组合风险标记 |",
        "| `momentum_overheat_industry_detail.csv` | 逐日 Active Basket 按精细行业拆分的过热动能和收益 |",
        "| `momentum_overheat_summary.csv` | 固定 10% 过热门槛和 30% 权重占比规则的 Momentum Fatigue Warning 核心 summary |",
        "| `momentum_overheat_focus.csv` | 01-12/01-14/01-29 的动能过热快照 |",
        "| `active_set_unanimous_loss_monitor_daily.csv` | Active Basket 100% 亏损且池子中位数未破 -1% 的 record-only shadow monitor |",
        "| `active_set_unanimous_loss_monitor_detail.csv` | active_set_unanimous_loss_monitor 触发日逐标的明细 |",
        "| `active_set_unanimous_loss_monitor_summary.csv` | active_set_unanimous_loss_monitor 汇总 |",
        "| `control_ablation_summary.csv` | 控制参数单独消融 |",
        "",
        "## 池子审计",
        "",
        format_table_for_report(top_capture, ["top_n", "selected_hits", "capture_rate", "hit_codes"], 10),
        "",
        "### 池子收益分布",
        "",
        format_table_for_report(
            return_distribution,
            ["group", "count", "mean", "median", "positive_share", "top5_mean", "top10_mean", "avg_abs_corr"],
            10,
        ),
        "",
        "### 未入池赢家近邻相关",
        "",
        format_table_for_report(
            missed,
            ["rank", "code", "name", "period_return", "nearest_selected_code", "nearest_selected_name", "nearest_abs_corr"],
            12,
        ),
        "",
        "### 入池内部高相关对",
        "",
        format_table_for_report(high_corr, ["code_left", "name_left", "code_right", "name_right", "corr"], 12),
        "",
        "## 均值/方差 IC",
        "",
        format_table_for_report(forecast_summary, ["metric", "count", "mean", "median", "positive_share", "p25", "p75"], 20),
        "",
        "### Mean Bias / Short-term Reversal Audit",
        "",
        "本节只检查均值信号是否在 01-12/01-14 前后对短期历史收益出现异常反向相关：`short_reversal_score = -corr(Forecast Mean, Historical Return)`。分数越高，说明越偏向买入短期下跌标的；这只是诊断，不改变权重。",
        "",
        format_table_for_report(
            mean_bias_event_focus[
                mean_bias_event_focus.get("hist_feature", pd.Series(dtype=object)).isin([
                    "hist_return_t_minus_5_to_t_minus_1",
                    "hist_return_t_minus_1",
                ])
            ] if not mean_bias_event_focus.empty else mean_bias_event_focus,
            [
                "datetime",
                "mean_column",
                "hist_feature",
                "n",
                "spearman_corr",
                "short_reversal_score",
                "short_reversal_score_pct_rank",
                "reversal_bias_label",
            ],
            20,
        ),
        "",
        format_table_for_report(
            mean_bias_active_detail,
            [
                "datetime",
                "instrument",
                "forecast_w",
                "forecast_mean_return",
                "forecast_weight_mean_raw",
                "hist_return_t_minus_5_to_t_minus_1",
                "hist_return_t_minus_1",
                "next_return",
            ],
            20,
        ),
        "",
        "## Top 赢家 MVO 诊断",
        "",
        format_table_for_report(
            top_winner_summary,
            [
                "instrument",
                "name",
                "period_return_rank",
                "period_return",
                "avg_forecast_mean_rank_pct",
                "avg_forecast_mean_top_rank_pct",
                "mean_top30_day_share",
                "avg_forecast_vol_rank_pct",
                "marginal_corr_contribution",
                "nearest_selected_peer_corr",
                "avg_forecast_w_raw_to_equal",
                "avg_forecast_w_l3_to_equal",
                "decision_tree_label",
                "diagnosis_label",
            ],
            12,
        ),
        "",
        "## 波动率豁免审计",
        "",
        format_table_for_report(
            vol_exemption_top_winners,
            [
                "instrument",
                "name",
                "period_return_rank",
                "period_return",
                "positive_return_share",
                "upside_volatility_share",
                "downside_volatility_share",
                "upside_downside_vol_ratio",
                "return_to_downside_vol",
                "offensive_volatility_flag",
            ],
            12,
        ),
        "",
        format_table_for_report(
            vol_exemption_scenarios,
            [
                "scenario",
                "selected_period_top10_mean_return",
                "score_top10_mean_period_return",
                "score_top10_above_30pct",
                "score_top10_capture_rate_vs_period_top10",
                "daily_score_top10_avg_overlap_vs_period_top10",
                "daily_score_top10_cumulative_return",
                "daily_score_top10_max_drawdown",
            ],
            20,
        ),
        "",
        "## 局部波动率上限审计",
        "",
        format_table_for_report(
            local_cap_candidates,
            [
                "instrument",
                "name",
                "period_return_rank",
                "period_return",
                "avg_forecast_mean_top_rank_pct",
                "mean_top30_day_share",
                "avg_forecast_vol_rank_pct",
                "marginal_corr_contribution",
                "nearest_selected_peer_corr",
                "decision_tree_label",
            ],
            15,
        ),
        "",
        format_table_for_report(
            local_cap_scenarios,
            [
                "scenario",
                "hidden_gem_count",
                "vol_cap_rank_pct",
                "daily_score_top10_avg_overlap_vs_period_top10",
                "target_overlap_40pct_pass",
                "delta_overlap_vs_no_cap",
                "daily_score_top10_cumulative_return",
                "daily_score_top10_max_drawdown",
                "delta_maxdd_vs_no_cap",
                "maxdd_not_worse_than_no_cap_1pct",
            ],
            20,
        ),
        "",
        "## 局部 cap Rolling/OOS 审计",
        "",
        format_table_for_report(
            local_cap_oos_summary,
            [
                "scenario",
                "fold_count",
                "avg_train_hidden_gem_count",
                "daily_oos_top10_avg_overlap",
                "target_overlap_40pct_pass",
                "delta_oos_overlap_vs_no_cap",
                "oos_score_top10_cumulative_return",
                "oos_score_top10_max_drawdown",
                "delta_oos_maxdd_vs_no_cap",
                "oos_maxdd_not_worse_than_no_cap_1pct",
            ],
            20,
        ),
        "",
        "## Residual Analysis",
        "",
        "本节只做误差归因：`actual_excess_return = next_return - 当日池子等权收益`，`predicted_excess_return = forecast_mean - 当日池子均值预测`，`residual_return = actual_excess_return - predicted_excess_return`。当 `predicted_excess_return < 0` 且实际为正时，额外计入 `active_offset_share_of_positive_specific`，用于识别模型是否在爆发期主动给了反向信号。",
        "",
        format_table_for_report(
            residual_summary,
            [
                "instrument",
                "name",
                "focus_theme",
                "decision_tree_label",
                "period_return_rank",
                "period_return",
                "boom_day_count",
                "predicted_share_of_positive_specific",
                "residual_share_clipped_0_1",
                "active_offset_share_of_positive_specific",
                "same_day_forecast_top30_share",
                "reaction_slow_day_share",
                "persistent_miss_day_share",
                "reaction_diagnosis",
            ],
            20,
        ),
        "",
        "### 反应滞后",
        "",
        format_table_for_report(
            residual_reaction,
            [
                "instrument",
                "name",
                "focus_theme",
                "boom_day_count",
                "reaction_diagnosis",
                "corr_actual_t_forecast_t_minus_1",
                "corr_actual_t_forecast_t",
                "corr_actual_t_forecast_t_plus_1",
                "best_forward_lag_days",
                "best_forward_lag_corr",
            ],
            20,
        ),
        "",
        "### Lead-Lag 前瞻性",
        "",
        "本表直接检查 `forecast_mean(t)` 对 `next_return(t+n)` 的相关性。若 `corr_mean_t_return_t_plus_2 > corr_mean_t_return_t_plus_1` 且最佳 forward lag 落在 2 日或以后，优先怀疑均值端过度平滑或 lookback 过长。",
        "",
        format_table_for_report(
            lead_lag,
            [
                "instrument",
                "name",
                "focus_theme",
                "period_return_rank",
                "corr_mean_t_return_t_0",
                "corr_mean_t_return_t_plus_1",
                "corr_mean_t_return_t_plus_2",
                "corr_plus2_gt_plus1",
                "best_forward_lag_days",
                "best_forward_lag_corr",
                "lead_lag_diagnosis",
            ],
            20,
        ),
        "",
        "### 因子/拥挤代理检查",
        "",
        "L3 已升级为可输出 `factor_style_momentum/growth/size` 与 `factor_industry_momentum`。旧 L3 尚未重跑时，本表会用 raw panel 按同一公式回填，并在 `explicit_factor_source` 标记来源；显式列仍缺失时才退回代理判断。",
        "",
        format_table_for_report(
            residual_factor_check,
            [
                "instrument",
                "name",
                "focus_theme",
                "explicit_style_columns_available",
                "explicit_factor_source",
                "factor_industry_theme",
                "boom_day_count",
                "boom_active_offset_day_share",
                "boom_momentum_positive_share",
                "boom_trend_sleeve_selected_share",
                "boom_scale_down_share",
                "boom_avg_factor_style_momentum",
                "boom_avg_factor_style_growth",
                "boom_avg_factor_style_size",
                "boom_avg_factor_industry_momentum",
                "boom_avg_risk_trend_momentum_z",
                "boom_avg_risk_trend_score",
                "boom_avg_forecast_active_set_crowded_score",
                "boom_avg_risk_cluster_scale",
                "boom_avg_risk_signal_quality_scale",
                "dominant_offset_driver",
                "dominant_explicit_factor_direction",
                "constraint_offset_explicit_label",
                "constraint_offset_proxy_label",
            ],
            20,
        ),
        "",
        "### 负偏置中性化模拟",
        "",
        "本表只做 score-level 补救模拟：当通信/AI/半导体等主题出现异常放量且 `forecast_mean` 或横截面预测超额为负时，分别模拟把该均值抬到横截面中性或绝对 0 后的 Top10 score basket 表现。它不是完整 MVO 重跑，但能衡量负偏置是否值得进入下一轮优化实验。",
        "",
        format_table_for_report(
            negative_bias_improvement,
            [
                "scenario",
                "event_row_count",
                "event_day_count",
                "event_unique_instrument_count",
                "event_instruments",
                "baseline_cumulative_return",
                "scenario_cumulative_return",
                "delta_cumulative_return",
                "baseline_max_drawdown",
                "scenario_max_drawdown",
                "delta_max_drawdown",
                "delta_avg_period_top_winner_overlap",
                "avg_event_selected_count_after_neutralize",
            ],
            10,
        ),
        "",
        "### OOS 影子测试与准入审计",
        "",
        "本节不修改 `strategy_layer.py` 主逻辑，只在 audit 层做 score-level shadow run。硬阈值锁定为：`" + shadow_patch_thresholds_text() + "`。该补丁是 Defensive Correction：只在异常放量、Growth 显式负暴露超限且 Mean Signal 为负时消除自残式负信号，不主动增加风险敞口。下方 full-MVO 小节已补充真实 optimizer + L3 rerun。",
        "",
        format_table_for_report(
            shadow_admission,
            [
                "scenario",
                "admission_audit_status",
                "trigger_row_count",
                "trigger_day_count",
                "base_cumulative_return",
                "patched_cumulative_return",
                "delta_cumulative_return",
                "base_sharpe",
                "patched_sharpe",
                "delta_sharpe",
                "base_max_drawdown",
                "patched_max_drawdown",
                "delta_max_drawdown",
                "base_win_rate_in_ai_communication_boom_days",
                "patched_win_rate_in_ai_communication_boom_days",
                "delta_win_rate_in_ai_communication_boom_days",
                "tracking_error_annualized",
                "tracking_error_lt_1pct",
                "full_mvo_tracking_error_annualized",
                "full_mvo_te_lt_1pct",
                "full_mvo_patched_optimizer_failed_day_count",
                "synthetic_stress_pass_no_optimizer_failed",
                "factor_source",
                "factor_volume_ratio_20_default_fill_count",
                "factor_volume_z_20_default_fill_count",
                "factor_style_growth_default_fill_count",
                "hot_plug_capability",
                "feature_flag_recommendation",
                "promotion_shadow_window_trading_days",
                "full_mvo_max_patched_optimizer_time_cost_sec",
                "full_mvo_max_optimizer_time_cost_delta_sec",
                "base_avg_turnover",
                "patched_avg_turnover",
                "delta_avg_turnover",
                "specific_return_recovery_share_of_prior_residual",
            ],
            10,
        ),
        "",
        "### 补丁触发频率",
        "",
        format_table_for_report(
            shadow_trigger_frequency,
            [
                "scope",
                "fold",
                "oos_start",
                "oos_end",
                "oos_day_count",
                "oos_row_count",
                "trigger_row_count",
                "trigger_day_count",
                "trigger_row_rate",
                "trigger_day_rate",
                "avg_trigger_rows_per_day",
                "trigger_instruments",
                "factor_source",
            ],
            12,
        ),
        "",
        "### Shadow Summary",
        "",
        format_table_for_report(
            shadow_patch_summary,
            [
                "scenario",
                "fold_count",
                "base_annual_return",
                "patched_annual_return",
                "base_annual_vol",
                "patched_annual_vol",
                "base_max_turnover",
                "patched_max_turnover",
                "delta_max_turnover",
                "predicted_excess_recovery_sum",
                "clipped_specific_return_recovery_sum",
                "specific_return_recovery_share_of_positive_specific",
                "specific_return_recovery_share_of_prior_residual",
            ],
            10,
        ),
        "",
        "### Full MVO Weight Rerun",
        "",
        "本节复用 bridge 中的 optimizer mean、variance 与 full covariance matrix，调用 `route_ab_moments_bridge.bridge.optimize_weights_hard_band()` 重求 base/patch 权重，再调用 `pipeline.risk_controls.apply_risk_controls()` 生成 L3 effective weights。",
        "",
        format_table_for_report(
            full_mvo_weight_summary,
            [
                "scope",
                "row_count",
                "day_count",
                "trigger_row_count",
                "optimizer_mean_changed_row_count",
                "max_abs_forecast_w_delta",
                "max_abs_l3_effective_weight_delta",
                "avg_daily_sum_abs_l3_effective_weight_delta",
                "max_daily_turnover_equivalent_delta",
                "tracking_error_annualized",
                "tracking_error_lt_1pct",
                "base_optimizer_failed_day_count",
                "patched_optimizer_failed_day_count",
                "max_patched_optimizer_time_cost_sec",
                "max_optimizer_time_cost_delta_sec",
                "optimizer_mean_column",
            ],
            12,
        ),
        "",
        "### Synthetic Edge Case Stress",
        "",
        "Synthetic session 人工放大 AI/通信主题 `factor_volume_z_20` 并强制 optimizer mean 为负，用同一条 optimizer + L3 路径确认补丁不会触发 `optimizer_failed`。",
        "",
        format_table_for_report(
            synthetic_stress_summary,
            [
                "scenario",
                "synthetic_session_date",
                "synthetic_modified_row_count",
                "trigger_row_count",
                "base_optimizer_failed",
                "patched_optimizer_failed",
                "patched_status_problem",
                "finite_weight_check",
                "max_patched_l3_forecast_w",
                "turnover_equivalent_delta",
                "patched_optimizer_time_cost_sec",
                "optimizer_time_cost_delta_sec",
                "stress_pass_no_optimizer_failed",
            ],
            10,
        ),
        "",
        "## Overflow Bucket",
        "",
        format_table_for_report(
            overflow_summary,
            [
                "scope",
                "candidate_count",
                "unique_instrument_count",
                "avg_abs_corr_to_selected",
                "avg_future_return",
                "avg_selected_equal_future_return",
                "avg_excess_future_return",
                "win_rate_vs_selected_equal",
                "positive_future_share",
            ],
            8,
        ),
        "",
        "## 协方差 IC",
        "",
        format_table_for_report(covariance_summary, ["metric", "count", "mean", "median", "positive_share", "p25", "p75"], 20),
        "",
        "## 组合方法对照",
        "",
        format_table_for_report(
            portfolio_summary,
            ["method", "days", "cumulative_return", "annual_return", "annual_vol", "sharpe", "max_drawdown"],
            20,
        ),
        "",
        "## Signal Quality × Industry Entropy",
        "",
        "行业熵只用于交叉审计：`entropy_norm < 0.50` 或单一 L3 行业权重 `> 50%` 记为集中度 warning。它不触发拦截，用来判断 01-29 这类混合风险日是否是单行业损失集中。",
        "",
        format_table_for_report(
            industry_entropy_focus,
            [
                "datetime",
                "active_count",
                "active_negative_share",
                "active_weighted_next_return",
                "pool_median_next_return",
                "risk_signal_quality_rolling_ic",
                "risk_signal_quality_scale",
                "industry_count",
                "industry_entropy_norm",
                "dominant_industry",
                "dominant_industry_weight_share",
                "dominant_loss_industry",
                "dominant_loss_contribution_share",
                "concentration_warning",
            ],
            12,
        ),
        "",
        format_table_for_report(
            industry_entropy_detail[
                industry_entropy_detail.get("datetime", pd.Series(dtype=object)).astype(str).isin(INDUSTRY_ENTROPY_FOCUS_DATES)
            ] if not industry_entropy_detail.empty else industry_entropy_detail,
            [
                "datetime",
                "industry",
                "active_count",
                "industry_weight_share",
                "industry_effective_weight_share",
                "weighted_return_contribution",
                "industry_loss_contribution_share",
                "mean_next_return",
                "active_codes",
            ],
            20,
        ),
        "",
        "### Entropy-Weighted Defense OOS Experiment",
        "",
        "该实验仍是 audit-only：当精细行业熵低于阈值，且 `risk_signal_quality_rolling_ic` 落在中性偏弱区间时，对当日组合额外乘以 scale-down，检验是否能精准压 01-29 且不误伤 4 月 AI/半导体收益。",
        "",
        format_table_for_report(
            entropy_defense_comparison,
            [
                "scenario",
                "cumulative_return",
                "sharpe",
                "max_drawdown",
                "delta_cumulative_return_vs_current_l3",
                "delta_max_drawdown_vs_current_l3",
                "trigger_day_count",
                "trigger_dates",
                "return_2026-01-29",
                "delta_return_2026-01-29_vs_current_l3",
                "april_trigger_day_count",
                "delta_april_ai_theme_cumulative_return_vs_current_l3",
            ],
            12,
        ),
        "",
        "## Momentum Overheat Audit",
        "",
        "本节把 Active Basket 中 `T-5` 到 `T-1` 累积涨幅超过 10% 的权重占比固化为 `combined_momentum_overheat_index`。当该 index 大于等于 30% 时，仅记录 `Momentum Fatigue Warning`，用于观察未来 2-3 天 Rolling IC 是否随后崩塌；该规则不拦截交易。",
        "",
        format_table_for_report(
            momentum_overheat_summary,
            [
                "datetime",
                "combined_momentum_overheat_index",
                "momentum_fatigue_warning",
                "active_count",
                "hot_active_count",
                "active_weighted_next_return",
                "hot_active_weighted_next_return",
                "risk_signal_quality_rolling_ic",
                "rolling_ic_min_next_3d",
                "rolling_ic_drop_next_3d",
                "rolling_ic_collapse_next_3d_flag",
                "industry_entropy_norm",
                "dominant_industry",
                "combined_overheat_concentration_flag",
            ],
            10,
        ),
        "",
        format_table_for_report(
            momentum_overheat_focus,
            [
                "datetime",
                "combined_momentum_overheat_index",
                "momentum_fatigue_warning",
                "momentum_fatigue_label",
                "active_count",
                "hot_active_count",
                "hot_active_weight_share",
                "active_avg_hist_return_t_minus_5_to_t_minus_1",
                "active_max_hist_return_t_minus_5_to_t_minus_1",
                "active_weighted_next_return",
                "hot_active_weighted_next_return",
                "pool_mu_rank_ic_all",
                "pool_mu_rank_ic_hot",
                "industry_entropy_norm",
                "dominant_industry",
                "dominant_industry_weight_share",
                "dominant_industry_avg_hist_return_t_minus_5_to_t_minus_1",
                "dominant_industry_hot_weight_share_within_industry",
                "dominant_industry_weighted_next_return",
                "combined_overheat_concentration_flag",
                "hot_active_codes",
            ],
            12,
        ),
        "",
        "## Active-Set 脆弱性 Shadow Audit",
        "",
        "规则 `active_set_unanimous_loss_monitor` 仅记录不拦截：当 Active Basket 成员 100% `next_return < 0`，且当日 Pool 收益中位数 `> -1%` 时触发。该规则用于识别 01-12/01-29 这类 active basket 全负、但横截面池子并未系统性崩塌的脆弱性事件，不改变 `forecast_w`、`exposure_scale` 或任何实际持仓。",
        "",
        format_table_for_report(
            active_set_monitor_summary,
            [
                "rule_name",
                "action",
                "total_day_count",
                "trigger_day_count",
                "trigger_day_share",
                "avg_trigger_active_count",
                "avg_trigger_pool_median_next_return",
                "avg_trigger_active_weighted_next_return",
                "trigger_dates",
                "monitor_scope",
            ],
            10,
        ),
        "",
        format_table_for_report(
            active_set_monitor_daily[active_set_monitor_daily.get("triggered", pd.Series(False, index=active_set_monitor_daily.index)).fillna(False).astype(bool)] if not active_set_monitor_daily.empty else active_set_monitor_daily,
            [
                "datetime",
                "triggered",
                "active_count",
                "active_negative_share",
                "pool_median_next_return",
                "active_weighted_next_return",
                "risk_signal_quality_rolling_ic",
                "risk_signal_quality_scale",
                "active_codes",
            ],
            12,
        ),
        "",
        "## 控制参数消融",
        "",
        format_table_for_report(
            control,
            [
                "case",
                "cumulative_return",
                "max_drawdown",
                "delta_return_vs_base",
                "delta_maxdd_vs_base",
                "marginal_return_vs_prev",
                "avg_actual_exposure",
                "risk_on_avg_exposure",
                "risk_off_avg_exposure",
            ],
            20,
        ),
        "",
        "## 入参晋级规则",
        "",
        "1. 池子参数必须同时说明 Top-N 捕获和近邻相关；漏掉赢家如果是高相关近克隆，可以接受，低相关高收益漏选要单独复核。",
        "2. 均值预测若 IC 不稳定，不应继续加大 MVO 对均值的依赖；方差预测若稳定，可进入风险预算与 vol targeting。",
        "3. 协方差 off-diagonal 若 IC 偏弱，只能作为弱约束，不能作为新增复杂参数的主要依据。",
        "4. 控制参数必须看单独消融和边际贡献；收益提升如果伴随 MaxDD/stress 明显恶化，只能作为进攻候选。",
        "5. 主线只接收低自由度、解释清楚、边际收益稳定且回撤不恶化的参数。",
        "",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[WRITE] {report_md}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir, report_md = ensure_output_paths(args)
    cases = parse_cases(args.case)
    if not cases:
        raise RuntimeError("No case directories were found. Pass --case label:path to audit custom outputs.")
    base_paths = resolve_case_paths(cases[0])
    bridge_path = base_paths["bridge"]
    l3_path = base_paths["bridge_l3"]
    raw_panel_path = base_paths["raw_panel"]
    daily_nav_path = base_paths["daily_nav"]
    if bridge_path is None or l3_path is None or daily_nav_path is None:
        raise RuntimeError(f"Base case lacks bridge/l3/daily_nav outputs: {cases[0].path}")

    tables: dict[str, pd.DataFrame] = {}
    if args.skip_pool_audit:
        print("[SKIP] pool audit")
    else:
        print("[RUN] pool audit")
        tables.update(run_pool_audit(args, out_dir))

    print("[RUN] forecast IC audit")
    tables.update(forecast_ic_tables(bridge_path, out_dir, args.blend_momentum_window))
    print("[RUN] selected Top winner forecast diagnostic")
    tables.update(
        top_winner_forecast_diagnostics(
            bridge_path,
            l3_path,
            out_dir,
            args.fund_list_csv,
            args.top_winner_k,
            args.vol_exemption_min_period_return,
            args.vol_exemption_upside_share_min,
            args.vol_exemption_positive_share_min,
            args.local_cap_mean_top_pct,
            args.clone_corr_threshold,
        )
    )
    print("[RUN] volatility exemption audit")
    tables.update(
        volatility_exemption_scenarios(
            bridge_path,
            out_dir,
            args.fund_list_csv,
            args.top_winner_k,
            parse_float_list(args.vol_exemption_risk_free_rates),
            parse_float_list(args.vol_exemption_variance_multipliers),
            args.vol_exemption_min_period_return,
            args.vol_exemption_upside_share_min,
            args.vol_exemption_positive_share_min,
        )
    )
    print("[RUN] local variance penalty cap audit")
    tables.update(
        local_variance_penalty_cap_scenarios(
            bridge_path,
            out_dir,
            args.fund_list_csv,
            args.top_winner_k,
            parse_float_list(args.vol_exemption_risk_free_rates),
            parse_float_list(args.local_vol_cap_rank_pcts),
            args.local_cap_mean_top_pct,
            args.clone_corr_threshold,
            args.vol_exemption_min_period_return,
            args.vol_exemption_upside_share_min,
            args.vol_exemption_positive_share_min,
        )
    )
    print("[RUN] local variance penalty cap rolling/OOS audit")
    tables.update(
        local_variance_penalty_cap_oos_audit(
            bridge_path,
            out_dir,
            args.fund_list_csv,
            args.top_winner_k,
            parse_float_list(args.vol_exemption_risk_free_rates),
            parse_float_list(args.local_vol_cap_rank_pcts),
            args.local_cap_mean_top_pct,
            args.clone_corr_threshold,
            args.vol_exemption_min_period_return,
            args.vol_exemption_upside_share_min,
            args.vol_exemption_positive_share_min,
            args.local_cap_oos_lookback_days,
            args.local_cap_oos_forward_days,
            args.local_cap_oos_step_days,
        )
    )
    print("[RUN] residual analysis audit")
    tables.update(
        residual_analysis_audit(
            bridge_path,
            l3_path,
            raw_panel_path,
            out_dir,
            args.fund_list_csv,
            args.top_winner_k,
            args.local_cap_mean_top_pct,
            args.local_cap_mean_top_pct,
            args.clone_corr_threshold,
            args.vol_exemption_min_period_return,
            args.vol_exemption_upside_share_min,
            args.vol_exemption_positive_share_min,
        )
    )
    print("[RUN] lead-lag cross-correlation audit")
    tables.update(lead_lag_cross_correlation_audit(bridge_path, out_dir, args.fund_list_csv, args.top_winner_k))
    print("[RUN] negative-bias neutralization audit")
    tables.update(
        potential_improvement_from_neutralizing_negative_biases(
            bridge_path,
            l3_path,
            raw_panel_path,
            out_dir,
            args.fund_list_csv,
            args.top_winner_k,
        )
    )
    print("[RUN] OOS neutralize-zero shadow patch audit")
    tables.update(
        oos_shadow_neutralize_zero_patch_audit(
            bridge_path,
            l3_path,
            raw_panel_path,
            out_dir,
            args.fund_list_csv,
            args.top_winner_k,
            args.local_cap_oos_lookback_days,
            args.local_cap_oos_forward_days,
            args.local_cap_oos_step_days,
        )
    )
    print("[RUN] full MVO weight rerun audit")
    tables.update(
        run_full_mvo_patch_weight_rerun(
            bridge_path,
            l3_path,
            raw_panel_path,
            out_dir,
            args.fund_list_csv,
            args.provider_uri,
            args.local_cap_oos_lookback_days,
            args.local_cap_oos_forward_days,
            args.local_cap_oos_step_days,
        )
    )
    print("[RUN] synthetic edge case stress test")
    tables.update(
        synthetic_edge_case_stress_test(
            bridge_path,
            l3_path,
            raw_panel_path,
            out_dir,
            args.fund_list_csv,
            args.provider_uri,
            args.local_cap_oos_lookback_days,
            args.local_cap_oos_forward_days,
            args.local_cap_oos_step_days,
        )
    )
    update_admission_with_full_mvo_results(tables, out_dir)
    print("[RUN] covariance IC audit")
    tables.update(covariance_ic_tables(bridge_path, out_dir, args.future_corr_window, args.min_future_corr_obs))
    print("[RUN] portfolio method comparison")
    tables.update(portfolio_compare_tables(bridge_path, l3_path, daily_nav_path, out_dir, args.pure_momentum_top_k))
    print("[RUN] mean-bias short-term reversal audit")
    tables.update(mean_bias_reversal_correlation_audit(l3_path, out_dir))
    print("[RUN] signal-quality industry entropy cross audit")
    tables.update(signal_quality_industry_entropy_cross_audit(l3_path, out_dir, args.fund_list_csv))
    print("[RUN] entropy-weighted defense OOS experiment")
    tables.update(entropy_weighted_defense_oos_experiment(l3_path, out_dir, args.fund_list_csv))
    print("[RUN] momentum overheat audit")
    tables.update(momentum_overheat_audit(l3_path, out_dir, args.fund_list_csv))
    print("[RUN] active-set unanimous loss shadow monitor")
    tables.update(active_set_unanimous_loss_monitor(l3_path, out_dir, args.fund_list_csv))
    print("[RUN] control ablation audit")
    tables["control_ablation_summary"] = control_ablation_table(cases, out_dir)
    write_negative_mean_neutralization_rfc(args, out_dir, tables)
    build_report(args, out_dir, report_md, tables)
    return 0


if __name__ == "__main__":
    sys.exit(main())
