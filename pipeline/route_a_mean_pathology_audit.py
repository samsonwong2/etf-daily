"""Route A mean pathology audit.

This module joins Route A per-instrument mean layers with the existing residual
and full-MVO audit outputs, so mean misses can be attributed before any model
change is promoted.
"""
from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_DIR = Path("~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/strategy_layer_audit_20260430")
DEFAULT_ROUTE_A_DIR = REPO_ROOT / "strategy" / "route_a_core" / "outputs_route_a_quick_nll_fullpanel_20260101_20260413"
DEFAULT_OUT_DIR = DEFAULT_AUDIT_DIR
DEFAULT_REPORT_MD = REPO_ROOT / "workspace" / "history" / "route_a_mean_pathology_audit_20260501.md"
DEFAULT_FINAL_REPORT_MD = REPO_ROOT / "workspace" / "history" / "route_a_mean_pathology_audit_FINAL_20260501.md"
DEFAULT_PROVIDER_URI = "~/etf-daily-output/Documents/stock/data/qlib_data/all_fund_data/"
DEFAULT_ROUTE_A_START_DATE = "2026-01-01"
DEFAULT_ROUTE_A_END_DATE = "2026-04-13"
DEFAULT_ROUTE_A_TRAIN_WINDOW = 37
DEFAULT_ROUTE_A_RETRAIN_EVERY = 20
DEFAULT_ROUTE_A_TRAINING_OBJECTIVE = "nll-only"
DEFAULT_FORMULA_STRESS_INSTRUMENT = "SH588170"
DEFAULT_POST_BACKFILL_BRIDGE = DEFAULT_AUDIT_DIR / "backfilled_bridge_no_overlay_20260101_20260413.csv"
DEFAULT_MOMENTUM_OVERLAY_ALPHA = 0.02
ALPHA_WEIGHT_SCAN_LEVELS = tuple(round(i * 0.005, 6) for i in range(11))
V2_T3_MIN_EFFECTIVE_PATCH = 0.005
FINAL_SUMMARY_NAME = "route_a_mean_pathology_FINAL_summary.csv"
SHADOW_PNL_EQUITY_CURVE_NAME = "route_a_shadow_pnl_equity_curve_V2_T3_MIN_EFFECTIVE_PATCH.csv"
SHADOW_PNL_SUMMARY_NAME = "route_a_shadow_pnl_summary_V2_T3_MIN_EFFECTIVE_PATCH.csv"
ROUTE_A_INTERMEDIATE_OUTPUT_NAMES = (
    "route_a_mean_pathology_daily.csv",
    "route_a_mean_pathology_summary.csv",
    "route_a_mean_pathology_missing_route_a_files.csv",
    "student_t_formula_stress_test.csv",
    "required_momentum_boost_to_flip_sign.csv",
    "backfill_coverage_weight_impact.csv",
    "treatment_simulation_v1.csv",
    "v2_trigger_sensitivity.csv",
    "alpha_to_weight_elasticity.csv",
)
BACKFILL_TARGET_INSTRUMENTS = ("SH515880", "SZ159363", "SH515980")
TARGET_THEMES = ("communication", "ai", "semiconductor", "star_semiconductor")
NEGATIVE_EPS = -1e-12
ZERO_EPS = 1e-12


@dataclass(frozen=True)
class RouteAFileInfo:
    instrument: str
    path: Path | None
    source: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join Route A mean layers with residual/MVO audit outputs for pathology attribution."
    )
    parser.add_argument("--audit-dir", default=str(DEFAULT_AUDIT_DIR))
    parser.add_argument("--route-a-dir", default=str(DEFAULT_ROUTE_A_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument(
        "--target-instrument",
        action="append",
        default=None,
        help="Optional instrument code to audit. Repeatable. Defaults to mean-miss/theme targets from residual summary.",
    )
    parser.add_argument("--min-residual-share", type=float, default=0.80)
    parser.add_argument("--min-active-offset-share", type=float, default=0.20)
    parser.add_argument("--provider-uri", default=DEFAULT_PROVIDER_URI)
    parser.add_argument("--route-a-start-date", default=DEFAULT_ROUTE_A_START_DATE)
    parser.add_argument("--route-a-end-date", default=DEFAULT_ROUTE_A_END_DATE)
    parser.add_argument("--route-a-train-window", type=int, default=DEFAULT_ROUTE_A_TRAIN_WINDOW)
    parser.add_argument("--route-a-retrain-every", type=int, default=DEFAULT_ROUTE_A_RETRAIN_EVERY)
    parser.add_argument(
        "--route-a-training-objective",
        default=DEFAULT_ROUTE_A_TRAINING_OBJECTIVE,
        choices=("mixed", "nll-only"),
    )
    parser.add_argument("--formula-stress-instrument", default=DEFAULT_FORMULA_STRESS_INSTRUMENT)
    parser.add_argument("--post-backfill-bridge", default=str(DEFAULT_POST_BACKFILL_BRIDGE))
    parser.add_argument("--momentum-overlay-alpha", type=float, default=DEFAULT_MOMENTUM_OVERLAY_ALPHA)
    parser.add_argument(
        "--final-freeze",
        action="store_true",
        help="Write the final freeze report/summary and remove Route A pathology intermediate CSVs.",
    )
    return parser.parse_args(argv)


def normalize_code(value: object) -> str:
    return str(value).strip().upper()


def numeric_series(value: object, index: pd.Index | None = None) -> pd.Series:
    if isinstance(value, pd.Series):
        return pd.to_numeric(value, errors="coerce")
    if index is None:
        return pd.to_numeric(pd.Series(value), errors="coerce")
    return pd.to_numeric(pd.Series(value, index=index), errors="coerce")


def safe_float(value: object, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def ratio_or_nan(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= ZERO_EPS:
        return np.nan
    return float(numerator / denominator)


def read_csv_if_exists(path: Path, parse_dates: Iterable[str] = ()) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    present_dates = []
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        present_dates = [column for column in parse_dates if column in header]
        return pd.read_csv(path, parse_dates=present_dates)
    except Exception:
        return pd.DataFrame()


def write_table(frame: pd.DataFrame, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    frame.to_csv(path, index=False)
    print(f"[INFO] wrote {path} rows={len(frame)}")
    return path


def theme_tokens(value: object) -> set[str]:
    return {token.strip() for token in str(value).replace("|", ",").split(",") if token.strip()}


def has_target_theme(value: object) -> bool:
    return bool(theme_tokens(value).intersection(TARGET_THEMES))


def select_target_instruments(
    residual_summary: pd.DataFrame,
    explicit_targets: list[str] | None,
    min_residual_share: float,
    min_active_offset_share: float,
) -> list[str]:
    if explicit_targets:
        return sorted({normalize_code(code) for code in explicit_targets if str(code).strip()})
    if residual_summary.empty or "instrument" not in residual_summary.columns:
        return []
    work = residual_summary.copy()
    work["instrument"] = work["instrument"].map(normalize_code)
    focus_theme = work.get("focus_theme", pd.Series("", index=work.index)).fillna("")
    decision_label = work.get("decision_tree_label", pd.Series("", index=work.index)).fillna("")
    residual_share = numeric_series(work.get("residual_share_clipped_0_1", pd.Series(np.nan, index=work.index)))
    active_offset = numeric_series(
        work.get("active_offset_share_of_positive_specific", pd.Series(np.nan, index=work.index))
    )
    selector = (
        focus_theme.map(has_target_theme)
        | decision_label.eq("Mean Signal Miss")
        | ((residual_share >= float(min_residual_share)) & (active_offset >= float(min_active_offset_share)))
    )
    selected = work.loc[selector, "instrument"].dropna().tolist()
    return list(dict.fromkeys(selected))


def find_route_a_file(route_a_dir: Path, instrument: str) -> RouteAFileInfo:
    params_matches = sorted(route_a_dir.glob(f"route_a_params_{instrument}_*.csv"))
    if params_matches:
        return RouteAFileInfo(instrument=instrument, path=params_matches[-1], source="params")
    eval_matches = sorted(route_a_dir.glob(f"route_a_eval_daily_{instrument}_*.csv"))
    if eval_matches:
        return RouteAFileInfo(instrument=instrument, path=eval_matches[-1], source="eval_daily")
    return RouteAFileInfo(instrument=instrument, path=None, source="missing")


def load_route_a_layers(route_a_dir: Path, instruments: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    for instrument in instruments:
        file_info = find_route_a_file(route_a_dir, normalize_code(instrument))
        coverage_rows.append(
            {
                "instrument": file_info.instrument,
                "route_a_file_found": file_info.path is not None,
                "route_a_file_source": file_info.source,
                "route_a_file_path": str(file_info.path) if file_info.path is not None else "",
            }
        )
        if file_info.path is None:
            continue
        frame = read_csv_if_exists(file_info.path, parse_dates=("target_date", "forecast_date"))
        if frame.empty:
            continue
        frame = frame.copy()
        frame["instrument"] = file_info.instrument
        frame["route_a_file_source"] = file_info.source
        frame["route_a_file_path"] = str(file_info.path)
        for date_column in ["target_date", "forecast_date"]:
            if date_column in frame.columns:
                frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
        frames.append(frame)
    layers = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    coverage = pd.DataFrame(coverage_rows)
    return layers, coverage


def prepare_residual_daily(residual_daily: pd.DataFrame) -> pd.DataFrame:
    if residual_daily.empty:
        return pd.DataFrame()
    work = residual_daily.copy()
    work["instrument"] = work["instrument"].map(normalize_code)
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
    numeric_columns = [column for column in work.columns if column not in {"instrument", "datetime", "name", "focus_theme", "decision_tree_label", "factor_industry_theme"}]
    for column in numeric_columns:
        converted = pd.to_numeric(work[column], errors="coerce")
        if converted.notna().sum() > 0 or work[column].notna().sum() == 0:
            work[column] = converted
    return work


def bool_series(value: object, index: pd.Index | None = None) -> pd.Series:
    source = value if isinstance(value, pd.Series) else pd.Series(value, index=index)
    return source.map(lambda item: bool(item) if pd.notna(item) else False).astype(bool)


def prepare_full_mvo(full_mvo: pd.DataFrame) -> pd.DataFrame:
    if full_mvo.empty:
        return pd.DataFrame()
    keep_columns = [
        "instrument",
        "datetime",
        "full_mvo_optimizer_mean_column",
        "full_mvo_base_optimizer_mean",
        "full_mvo_patched_optimizer_mean",
        "full_mvo_optimizer_mean_changed_flag",
        "shadow_patch_trigger_flag",
        "admission_boom_row_flag",
        "oos_shadow_date_flag",
        "full_mvo_base_forecast_w",
        "full_mvo_patched_forecast_w",
        "base_l3_effective_weight",
        "patched_l3_effective_weight",
        "delta_l3_effective_weight",
        "abs_delta_l3_effective_weight",
        "full_mvo_active_return_delta",
    ]
    present = [column for column in keep_columns if column in full_mvo.columns]
    work = full_mvo[present].copy()
    if work.empty:
        return work
    work["instrument"] = work["instrument"].map(normalize_code)
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
    return work


def merge_observable_layers(
    residual_daily: pd.DataFrame,
    route_a_layers: pd.DataFrame,
    route_a_coverage: pd.DataFrame,
    full_mvo: pd.DataFrame,
) -> pd.DataFrame:
    daily = prepare_residual_daily(residual_daily)
    if daily.empty:
        return pd.DataFrame()
    route_a = route_a_layers.copy()
    if not route_a.empty:
        route_a["instrument"] = route_a["instrument"].map(normalize_code)
        route_a["forecast_date"] = pd.to_datetime(route_a["forecast_date"], errors="coerce")
        route_a = route_a.rename(columns={"forecast_date": "datetime", "target_date": "route_a_target_date"})
        route_a_prefixed_columns = {
            column: f"route_a_{column}"
            for column in route_a.columns
            if column not in {"instrument", "datetime", "route_a_target_date", "route_a_file_source", "route_a_file_path"}
        }
        route_a = route_a.rename(columns=route_a_prefixed_columns)
    merged = daily.merge(route_a, on=["instrument", "datetime"], how="left") if not route_a.empty else daily.copy()
    coverage = route_a_coverage.copy()
    if not coverage.empty:
        coverage["instrument"] = coverage["instrument"].map(normalize_code)
        merged = merged.merge(coverage, on="instrument", how="left", suffixes=("", "_coverage"))
        for column in ["route_a_file_source", "route_a_file_path"]:
            coverage_column = f"{column}_coverage"
            if coverage_column in merged.columns:
                if column in merged.columns:
                    merged[column] = merged[column].fillna(merged[coverage_column])
                else:
                    merged[column] = merged[coverage_column]
                merged = merged.drop(columns=[coverage_column])
        if "route_a_file_found" in merged.columns:
            merged["route_a_file_found"] = bool_series(merged["route_a_file_found"])
    else:
        merged["route_a_file_found"] = False
    mvo = prepare_full_mvo(full_mvo)
    if not mvo.empty:
        merged = merged.merge(mvo, on=["instrument", "datetime"], how="left", suffixes=("", "_full_mvo"))
    return merged


def sign_label(value: object) -> str:
    number = safe_float(value)
    if not np.isfinite(number):
        return "missing"
    if number < NEGATIVE_EPS:
        return "negative"
    if number > ZERO_EPS:
        return "positive"
    return "zero"


def add_pathology_columns(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    layer_columns = [
        "route_a_mu_raw",
        "route_a_mu",
        "route_a_mu_smooth",
        "route_a_mu_path",
        "route_a_posterior_mu_1d",
        "route_a_posterior_mu_20d",
        "forecast_mean",
        "forecast_weight_mean_raw",
        "forecast_weight_mean_slow",
        "forecast_weight_mean_used",
        "full_mvo_base_optimizer_mean",
    ]
    for column in layer_columns:
        if column not in work.columns:
            work[column] = np.nan
        work[f"{column}_sign"] = work[column].map(sign_label)
    work["actual_positive_specific_flag"] = numeric_series(work.get("actual_excess_return", pd.Series(index=work.index))) > 0.0
    work["bridge_used_zero_or_negative_flag"] = numeric_series(work["forecast_weight_mean_used"]) <= ZERO_EPS
    work["bridge_raw_positive_used_zero_flag"] = (
        numeric_series(work["forecast_weight_mean_raw"]) > ZERO_EPS
    ) & work["bridge_used_zero_or_negative_flag"]
    work["route_a_mu_raw_negative_flag"] = numeric_series(work["route_a_mu_raw"]) < 0.0
    work["route_a_mu_negative_flag"] = numeric_series(work["route_a_mu"]) < 0.0
    work["route_a_smooth_negative_flag"] = numeric_series(work["route_a_mu_smooth"]) < 0.0
    work["route_a_posterior_1d_negative_flag"] = numeric_series(work["route_a_posterior_mu_1d"]) < 0.0
    work["route_a_posterior_20d_negative_flag"] = numeric_series(work["route_a_posterior_mu_20d"]) < 0.0
    work["predicted_excess_negative_flag"] = numeric_series(work.get("predicted_excess_return", pd.Series(index=work.index))) < 0.0
    work["route_a_mu_shrink_delta"] = numeric_series(work["route_a_mu"]) - numeric_series(work["route_a_mu_raw"])
    work["route_a_smooth_delta"] = numeric_series(work["route_a_mu_smooth"]) - numeric_series(work["route_a_mu"])
    work["route_a_path_delta"] = numeric_series(work["route_a_mu_path"]) - numeric_series(work["route_a_mu_smooth"])
    work["route_a_kalman_1d_delta"] = numeric_series(work["route_a_posterior_mu_1d"]) - numeric_series(work["route_a_mu"])
    work["route_a_posterior20_delta"] = numeric_series(work["route_a_posterior_mu_20d"]) - numeric_series(work["route_a_posterior_mu_1d"])
    work["bridge_raw_to_used_delta"] = numeric_series(work["forecast_weight_mean_used"]) - numeric_series(work["forecast_weight_mean_raw"])
    work["primary_pathology"] = work.apply(classify_pathology_row, axis=1)
    return work


def classify_pathology_row(row: pd.Series) -> str:
    if not bool(row.get("route_a_file_found", False)):
        return "route_a_output_missing"
    if pd.isna(row.get("route_a_mu_raw")) and pd.isna(row.get("route_a_posterior_mu_1d")):
        return "route_a_layer_not_observable"
    if bool(row.get("actual_positive_specific_flag", False)) and safe_float(row.get("route_a_mu_raw")) < 0.0:
        return "student_t_raw_mean_negative_on_positive_specific"
    if bool(row.get("actual_positive_specific_flag", False)) and safe_float(row.get("route_a_mu")) < 0.0:
        return "shrunk_mean_negative_on_positive_specific"
    if safe_float(row.get("route_a_mu")) > 0.0 and safe_float(row.get("route_a_mu_smooth")) < 0.0:
        return "ewma_smoothing_flipped_positive_mu"
    if safe_float(row.get("route_a_mu")) > 0.0 and safe_float(row.get("route_a_posterior_mu_1d")) < 0.0:
        return "kalman_posterior_flipped_positive_mu"
    if safe_float(row.get("route_a_posterior_mu_1d")) > 0.0 and safe_float(row.get("route_a_posterior_mu_20d")) < 0.0:
        return "posterior_20d_lag_flipped_positive_1d"
    if bool(row.get("bridge_raw_positive_used_zero_flag", False)):
        return "bridge_transform_zeroed_positive_weight_mean"
    if bool(row.get("actual_positive_specific_flag", False)) and bool(row.get("predicted_excess_negative_flag", False)):
        return "cross_section_active_offset_negative"
    if bool(row.get("boom_day_flag", False)) and safe_float(row.get("forecast_mean_top_rank_pct")) > 0.30:
        return "rank_miss_on_boom_day"
    return "mixed_or_not_negative"


def mean_on_mask(frame: pd.DataFrame, column: str, mask: pd.Series) -> float:
    if column not in frame.columns:
        return np.nan
    values = numeric_series(frame.loc[mask, column]).dropna()
    return float(values.mean()) if not values.empty else np.nan


def share_on_mask(frame: pd.DataFrame, column: str, mask: pd.Series) -> float:
    if column not in frame.columns or int(mask.sum()) == 0:
        return np.nan
    values = frame.loc[mask, column]
    return float(values.fillna(False).astype(bool).mean())


def mode_text(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    if values.empty:
        return ""
    mode = values.mode()
    return str(mode.iloc[0]) if not mode.empty else str(values.iloc[0])


def summarize_pathology(daily: pd.DataFrame, residual_summary: pd.DataFrame, factor_check: pd.DataFrame, lead_lag: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    residual_lookup = residual_summary.copy()
    if not residual_lookup.empty and "instrument" in residual_lookup.columns:
        residual_lookup["instrument"] = residual_lookup["instrument"].map(normalize_code)
        residual_lookup = residual_lookup.set_index("instrument", drop=False)
    factor_lookup = factor_check.copy()
    if not factor_lookup.empty and "instrument" in factor_lookup.columns:
        factor_lookup["instrument"] = factor_lookup["instrument"].map(normalize_code)
        factor_lookup = factor_lookup.set_index("instrument", drop=False)
    lead_lookup = lead_lag.copy()
    if not lead_lookup.empty and "instrument" in lead_lookup.columns:
        lead_lookup["instrument"] = lead_lookup["instrument"].map(normalize_code)
        lead_lookup = lead_lookup.set_index("instrument", drop=False)
    rows: list[dict[str, object]] = []
    for instrument, frame in daily.groupby("instrument", sort=False):
        boom_mask = frame.get("boom_day_flag", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
        positive_mask = numeric_series(frame.get("actual_excess_return", pd.Series(index=frame.index))) > 0.0
        boom_positive_mask = boom_mask & positive_mask
        route_a_observed_mask = frame.get("route_a_mu_raw", pd.Series(np.nan, index=frame.index)).notna() | frame.get(
            "route_a_posterior_mu_1d", pd.Series(np.nan, index=frame.index)
        ).notna()
        observed_boom_mask = boom_mask & route_a_observed_mask
        observed_positive_mask = boom_positive_mask & route_a_observed_mask
        summary_row: dict[str, object] = {
            "instrument": instrument,
            "name": mode_text(frame.get("name", pd.Series(dtype=object))),
            "focus_theme": mode_text(frame.get("focus_theme", pd.Series(dtype=object))),
            "decision_tree_label": mode_text(frame.get("decision_tree_label", pd.Series(dtype=object))),
            "period_return_rank": int(safe_float(frame.get("period_return_rank", pd.Series([np.nan])).dropna().iloc[0])) if frame.get("period_return_rank", pd.Series(dtype=float)).dropna().shape[0] else np.nan,
            "period_return": mean_on_mask(frame, "period_return", pd.Series(True, index=frame.index)),
            "route_a_file_found": bool(frame.get("route_a_file_found", pd.Series(False, index=frame.index)).fillna(False).astype(bool).any()),
            "route_a_file_source": mode_text(frame.get("route_a_file_source", pd.Series(dtype=object))),
            "route_a_file_path": mode_text(frame.get("route_a_file_path", pd.Series(dtype=object))),
            "row_count": int(frame.shape[0]),
            "boom_day_count": int(boom_mask.sum()),
            "boom_positive_specific_row_count": int(boom_positive_mask.sum()),
            "route_a_observed_row_count": int(route_a_observed_mask.sum()),
            "route_a_observed_boom_row_count": int((route_a_observed_mask & boom_mask).sum()),
            "route_a_observed_positive_specific_row_count": int(observed_positive_mask.sum()),
            "route_a_boom_coverage_rate": ratio_or_nan(float((route_a_observed_mask & boom_mask).sum()), float(boom_mask.sum())),
            "boom_actual_excess_sum": float(numeric_series(frame.loc[boom_mask, "actual_excess_return"]).clip(lower=0.0).sum()) if "actual_excess_return" in frame.columns else np.nan,
            "boom_predicted_excess_sum": float(numeric_series(frame.loc[boom_mask, "predicted_excess_return"]).sum()) if "predicted_excess_return" in frame.columns else np.nan,
            "boom_residual_sum": float(numeric_series(frame.loc[boom_mask, "residual_return"]).sum()) if "residual_return" in frame.columns else np.nan,
            "boom_primary_pathology": mode_text(frame.loc[boom_mask, "primary_pathology"]) if int(boom_mask.sum()) else "",
            "positive_specific_primary_pathology": mode_text(frame.loc[boom_positive_mask, "primary_pathology"]) if int(boom_positive_mask.sum()) else "",
            "observed_boom_primary_pathology": mode_text(frame.loc[observed_boom_mask, "primary_pathology"]) if int(observed_boom_mask.sum()) else "",
            "observed_positive_specific_primary_pathology": mode_text(frame.loc[observed_positive_mask, "primary_pathology"]) if int(observed_positive_mask.sum()) else "",
        }
        layer_mean_columns = [
            "route_a_mu_raw",
            "route_a_mu",
            "route_a_mu_smooth",
            "route_a_mu_path",
            "route_a_posterior_mu_1d",
            "route_a_posterior_mu_20d",
            "forecast_mean",
            "forecast_weight_mean_raw",
            "forecast_weight_mean_slow",
            "forecast_weight_mean_used",
            "full_mvo_base_optimizer_mean",
        ]
        for column in layer_mean_columns:
            summary_row[f"boom_avg_{column}"] = mean_on_mask(frame, column, boom_mask)
            summary_row[f"positive_specific_avg_{column}"] = mean_on_mask(frame, column, boom_positive_mask)
            summary_row[f"observed_boom_avg_{column}"] = mean_on_mask(frame, column, observed_boom_mask)
            summary_row[f"observed_positive_specific_avg_{column}"] = mean_on_mask(frame, column, observed_positive_mask)
        flag_columns = [
            "route_a_mu_raw_negative_flag",
            "route_a_mu_negative_flag",
            "route_a_smooth_negative_flag",
            "route_a_posterior_1d_negative_flag",
            "route_a_posterior_20d_negative_flag",
            "predicted_excess_negative_flag",
            "bridge_used_zero_or_negative_flag",
            "bridge_raw_positive_used_zero_flag",
        ]
        for column in flag_columns:
            summary_row[f"boom_share_{column}"] = share_on_mask(frame, column, boom_mask)
            summary_row[f"positive_specific_share_{column}"] = share_on_mask(frame, column, boom_positive_mask)
            summary_row[f"observed_boom_share_{column}"] = share_on_mask(frame, column, observed_boom_mask)
            summary_row[f"observed_positive_specific_share_{column}"] = share_on_mask(frame, column, observed_positive_mask)
        for source_frame, prefix, columns in [
            (
                residual_lookup,
                "residual",
                [
                    "predicted_share_of_positive_specific",
                    "residual_share_clipped_0_1",
                    "active_offset_share_of_positive_specific",
                    "same_day_forecast_top30_share",
                    "reaction_diagnosis",
                ],
            ),
            (
                factor_lookup,
                "factor",
                [
                    "boom_avg_factor_style_growth",
                    "boom_avg_factor_style_momentum",
                    "boom_avg_factor_style_size",
                    "boom_avg_forecast_weight_mean_raw",
                    "boom_avg_forecast_weight_mean_used",
                    "dominant_offset_driver",
                    "constraint_offset_explicit_label",
                    "constraint_offset_proxy_label",
                ],
            ),
            (
                lead_lookup,
                "lead_lag",
                [
                    "best_forward_lag_days",
                    "best_forward_lag_corr",
                    "corr_plus2_gt_plus1",
                    "lead_lag_diagnosis",
                ],
            ),
        ]:
            if not source_frame.empty and instrument in source_frame.index:
                source_row = source_frame.loc[instrument]
                if isinstance(source_row, pd.DataFrame):
                    source_row = source_row.iloc[0]
                for column in columns:
                    if column in source_row.index:
                        summary_row[f"{prefix}_{column}"] = source_row[column]
        rows.append(summary_row)
    summary = pd.DataFrame(rows)
    if "period_return_rank" in summary.columns:
        summary = summary.sort_values(["period_return_rank", "instrument"], na_position="last")
    return summary


def build_missing_coverage_table(coverage: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if coverage.empty:
        return pd.DataFrame()
    result = coverage.copy()
    if not summary.empty:
        extra_columns = [
            "instrument",
            "name",
            "focus_theme",
            "period_return_rank",
            "period_return",
            "boom_day_count",
            "residual_predicted_share_of_positive_specific",
            "residual_residual_share_clipped_0_1",
            "residual_active_offset_share_of_positive_specific",
        ]
        result = result.merge(summary[[column for column in extra_columns if column in summary.columns]], on="instrument", how="left")
    return result.sort_values(["route_a_file_found", "period_return_rank", "instrument"], ascending=[True, True, True])


def load_legacy_route_a_module() -> Any:
    module_path = REPO_ROOT / "strategy" / "route_a_core" / "route_a_student_t.py"
    spec = importlib.util.spec_from_file_location("route_a_student_t_legacy_for_audit", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Route A legacy module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def route_a_quick_profile_payload(legacy: Any, training_objective: str) -> dict[str, Any]:
    defaults = getattr(legacy, "RUNTIME_PROFILE_DEFAULTS", {}).get("quick-calibration", {})
    payload = {"training": dict(defaults.get("training", {}))}
    payload["training"]["training_objective"] = training_objective
    return payload


def target_residual_frame(residual_daily: pd.DataFrame) -> pd.DataFrame:
    residual = prepare_residual_daily(residual_daily)
    if residual.empty:
        return pd.DataFrame()
    keep_columns = [
        "instrument",
        "datetime",
        "name",
        "focus_theme",
        "boom_day_flag",
        "next_return",
        "actual_excess_return",
        "predicted_excess_return",
        "residual_return",
        "factor_industry_momentum",
        "factor_industry_momentum_raw",
        "factor_volume_ratio_20",
        "factor_volume_z_20",
        "forecast_weight_mean_used",
    ]
    present = [column for column in keep_columns if column in residual.columns]
    target = residual[present].copy()
    target = target.rename(columns={"datetime": "route_a_target_date"})
    target = target.rename(
        columns={column: f"target_{column}" for column in target.columns if column not in {"instrument", "route_a_target_date"}}
    )
    target["route_a_target_date"] = pd.to_datetime(target["route_a_target_date"], errors="coerce")
    return target


def select_formula_stress_sample(daily: pd.DataFrame, residual_daily: pd.DataFrame, instrument: str) -> pd.Series | None:
    if daily.empty or "route_a_target_date" not in daily.columns:
        return None
    work = daily.copy()
    work["instrument"] = work["instrument"].map(normalize_code)
    work["route_a_target_date"] = pd.to_datetime(work["route_a_target_date"], errors="coerce")
    target = target_residual_frame(residual_daily)
    if not target.empty:
        work = work.merge(target, on=["instrument", "route_a_target_date"], how="left")
    work = work[work["instrument"].eq(normalize_code(instrument))].copy()
    if work.empty:
        return None

    raw_negative = numeric_series(work.get("route_a_mu_raw", pd.Series(index=work.index))) < 0.0
    target_positive = numeric_series(work.get("target_actual_excess_return", pd.Series(index=work.index))) > 0.0
    forecast_positive = numeric_series(work.get("actual_excess_return", pd.Series(index=work.index))) > 0.0
    target_boom = bool_series(work.get("target_boom_day_flag", pd.Series(False, index=work.index)), index=work.index)
    forecast_boom = bool_series(work.get("boom_day_flag", pd.Series(False, index=work.index)), index=work.index)

    priority_masks = [
        raw_negative & target_positive & target_boom,
        raw_negative & target_positive,
        raw_negative & forecast_positive & forecast_boom,
        raw_negative & forecast_positive,
        raw_negative,
    ]
    sort_columns = ["target_actual_excess_return", "actual_excess_return", "datetime"]
    for mask in priority_masks:
        subset = work.loc[mask].copy()
        if subset.empty:
            continue
        for column in sort_columns[:2]:
            if column not in subset.columns:
                subset[column] = np.nan
        subset = subset.sort_values(sort_columns, ascending=[False, False, True], na_position="last")
        return subset.iloc[0]
    return None


def build_student_t_formula_stress_test(
    daily: pd.DataFrame,
    residual_daily: pd.DataFrame,
    *,
    route_a_dir: Path,
    provider_uri: str,
    route_a_start_date: str,
    route_a_end_date: str,
    train_window: int,
    retrain_every: int,
    training_objective: str,
    instrument: str,
) -> pd.DataFrame:
    sample = select_formula_stress_sample(daily, residual_daily, instrument)
    if sample is None:
        return pd.DataFrame()

    legacy = load_legacy_route_a_module()
    payload = route_a_quick_profile_payload(legacy, training_objective)
    config = legacy.route_a_config_from_payload(
        provider_uri=provider_uri,
        output_dir=str(route_a_dir),
        payload=payload,
    )
    daily_mean = legacy.load_from_qlib(
        provider_uri=config.provider_uri,
        benchmark=[normalize_code(instrument)],
        start_date=route_a_start_date,
        end_date=route_a_end_date,
        pct_change_unit="auto",
    )
    df_feat = legacy.build_features(daily_mean, config)
    if df_feat.empty:
        return pd.DataFrame()

    effective_train_window, _warning = legacy.resolve_effective_train_window(train_window, len(df_feat), config)
    feature_cols = [column for column in df_feat.columns if column not in {"target", "target_date"}]
    x_all = np.column_stack([np.ones(len(df_feat)), df_feat[feature_cols].values])
    y_all = df_feat["target"].to_numpy(dtype=float)
    forecast_dates = pd.to_datetime(df_feat.index)
    forecast_date = pd.Timestamp(sample.get("datetime"))
    matches = np.flatnonzero(forecast_dates == forecast_date)
    if matches.size == 0:
        return pd.DataFrame()
    sample_idx = int(matches[0])
    if sample_idx < effective_train_window:
        return pd.DataFrame()

    fit = None
    init_params = None
    for idx in range(effective_train_window, sample_idx + 1):
        if (idx - effective_train_window) % int(retrain_every) == 0 or fit is None:
            start = idx - effective_train_window
            end = idx
            fit = legacy.fit_student_t_mu_sigma(
                x_all[start:end],
                y_all[start:end],
                config,
                init_params=init_params,
            )
            init_params = fit.optimizer_params.copy() if fit.optimizer_params is not None else None
    if fit is None:
        return pd.DataFrame()

    x_one = x_all[sample_idx : sample_idx + 1]
    x_scaled = x_one.copy()
    x_scaled[:, 1:] = (x_scaled[:, 1:] - fit.feature_mean) / fit.feature_scale
    mu, sigma, nu = legacy.predict_params(x_one, fit)
    mu_raw = float((x_scaled @ fit.beta)[0] * fit.target_scale)
    mu_shrunk = float(mu[0])
    sigma_value = float(sigma[0])
    nu_value = float(nu[0])
    y_true = float(y_all[sample_idx])
    nll_at_mu_raw = -float(legacy.student_t.logpdf(y_true, df=nu_value, loc=mu_raw, scale=sigma_value))
    nll_at_zero = -float(legacy.student_t.logpdf(y_true, df=nu_value, loc=0.0, scale=sigma_value))
    nll_at_actual = -float(legacy.student_t.logpdf(y_true, df=nu_value, loc=y_true, scale=sigma_value))
    standardized_residual_z = (y_true - mu_raw) / max(sigma_value, 1e-12)

    names = ["intercept", *feature_cols]
    rows: list[dict[str, object]] = []
    for idx, feature in enumerate(names):
        if idx == 0:
            x_raw = 1.0
            feature_mean = 0.0
            feature_scale = 1.0
        else:
            feature_idx = idx - 1
            x_raw = safe_float(df_feat.iloc[sample_idx][feature])
            feature_mean = safe_float(fit.feature_mean[feature_idx])
            feature_scale = safe_float(fit.feature_scale[feature_idx])
        x_scaled_value = safe_float(x_scaled[0, idx])
        beta_value = safe_float(fit.beta[idx])
        contribution = x_scaled_value * beta_value * fit.target_scale
        rows.append(
            {
                "instrument": normalize_code(instrument),
                "forecast_date": forecast_date,
                "route_a_target_date": pd.Timestamp(sample.get("route_a_target_date")),
                "feature": feature,
                "x_raw": x_raw,
                "feature_mean": feature_mean,
                "feature_scale": feature_scale,
                "x_scaled": x_scaled_value,
                "beta_scaled": beta_value,
                "target_scale": fit.target_scale,
                "contribution_to_mu_raw": contribution,
                "contribution_sign": sign_label(contribution),
                "route_a_mu_raw_file": safe_float(sample.get("route_a_mu_raw")),
                "mu_raw_recomputed": mu_raw,
                "mu_raw_file_diff": mu_raw - safe_float(sample.get("route_a_mu_raw")),
                "mu_shrink": fit.mu_shrink,
                "mu_after_shrink_recomputed": mu_shrunk,
                "sigma_recomputed": sigma_value,
                "nu_recomputed": nu_value,
                "y_true_route_a_next_return": y_true,
                "target_actual_excess_return": safe_float(sample.get("target_actual_excess_return")),
                "target_boom_day_flag": bool(sample.get("target_boom_day_flag", False)),
                "standardized_residual_z": standardized_residual_z,
                "nll_at_mu_raw": nll_at_mu_raw,
                "nll_at_zero_mu": nll_at_zero,
                "nll_at_actual_return_mu": nll_at_actual,
                "nll_improvement_if_mu_zero": nll_at_mu_raw - nll_at_zero,
                "nll_improvement_if_mu_actual": nll_at_mu_raw - nll_at_actual,
                "fit_success": bool(fit.success),
                "fit_fallback": bool(fit.fallback_used),
                "fit_message": fit.message,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["abs_contribution_rank"] = result["contribution_to_mu_raw"].abs().rank(
            ascending=False,
            method="dense",
        ).astype(int)
        result = result.sort_values(["contribution_to_mu_raw", "feature"], ascending=[True, True])
    return result


def required_alpha_to_flip(mu: pd.Series, momentum: pd.Series) -> pd.Series:
    mu_values = numeric_series(mu)
    momentum_values = numeric_series(momentum)
    result = pd.Series(np.nan, index=mu_values.index, dtype=float)
    result.loc[mu_values >= 0.0] = 0.0
    feasible = (mu_values < 0.0) & (momentum_values > ZERO_EPS)
    result.loc[feasible] = (ZERO_EPS - mu_values.loc[feasible]) / momentum_values.loc[feasible]
    return result.replace([np.inf, -np.inf], np.nan)


def build_required_momentum_boost_to_flip_sign(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    work = daily.copy()
    for column in [
        "route_a_mu_raw",
        "route_a_mu",
        "factor_industry_momentum",
        "factor_industry_momentum_raw",
        "actual_excess_return",
        "boom_day_flag",
        "route_a_file_found",
    ]:
        if column not in work.columns:
            work[column] = np.nan
    work["required_alpha_gain_to_flip_raw_mu"] = required_alpha_to_flip(
        work["route_a_mu_raw"],
        work["factor_industry_momentum"],
    )
    work["required_alpha_gain_raw_momentum_to_flip_raw_mu"] = required_alpha_to_flip(
        work["route_a_mu_raw"],
        work["factor_industry_momentum_raw"],
    )
    work["required_alpha_gain_to_flip_shrunk_mu"] = required_alpha_to_flip(
        work["route_a_mu"],
        work["factor_industry_momentum"],
    )
    work["momentum_boost_feasible_flag"] = work["required_alpha_gain_to_flip_raw_mu"].notna()
    work["raw_momentum_boost_feasible_flag"] = work["required_alpha_gain_raw_momentum_to_flip_raw_mu"].notna()
    work["actual_positive_specific_flag"] = numeric_series(work.get("actual_excess_return", pd.Series(index=work.index))) > 0.0
    work["route_a_raw_negative_flag"] = numeric_series(work["route_a_mu_raw"]) < 0.0
    selector = bool_series(work["route_a_file_found"], index=work.index) & work["route_a_raw_negative_flag"]
    result = work.loc[selector].copy()
    keep_columns = [
        "datetime",
        "instrument",
        "name",
        "focus_theme",
        "boom_day_flag",
        "actual_positive_specific_flag",
        "actual_excess_return",
        "route_a_target_date",
        "route_a_mu_raw",
        "route_a_mu",
        "factor_industry_momentum",
        "factor_industry_momentum_raw",
        "factor_volume_z_20",
        "required_alpha_gain_to_flip_raw_mu",
        "required_alpha_gain_raw_momentum_to_flip_raw_mu",
        "required_alpha_gain_to_flip_shrunk_mu",
        "momentum_boost_feasible_flag",
        "raw_momentum_boost_feasible_flag",
        "forecast_weight_mean_used",
        "primary_pathology",
    ]
    present = [column for column in keep_columns if column in result.columns]
    result = result[present]
    if not result.empty:
        result = result.sort_values(
            ["actual_positive_specific_flag", "boom_day_flag", "required_alpha_gain_to_flip_raw_mu", "instrument", "datetime"],
            ascending=[False, False, True, True, True],
            na_position="last",
        )
    return result


def weight_allocation_bucket(weight: object) -> str:
    value = safe_float(weight)
    if not np.isfinite(value) or value <= ZERO_EPS:
        return "zero"
    if value < 0.0001:
        return "tiny_lt_1bp"
    if value < 0.01:
        return "small_lt_1pct"
    return "material_ge_1pct"


def build_backfill_coverage_weight_impact(daily: pd.DataFrame, post_backfill_bridge_path: Path) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    focus = daily.loc[daily["instrument"].isin(BACKFILL_TARGET_INSTRUMENTS)].copy()
    if focus.empty:
        return pd.DataFrame()
    focus["datetime"] = pd.to_datetime(focus["datetime"])
    focus = focus.loc[bool_series(focus.get("route_a_file_found", pd.Series(False, index=focus.index)), index=focus.index)].copy()
    if focus.empty:
        return pd.DataFrame()
    bridge_columns = [
        "datetime",
        "instrument",
        "route_a_status",
        "route_a_message",
        "forecast_weight_mean_raw",
        "forecast_weight_mean_slow",
        "forecast_weight_mean_used",
        "forecast_w",
        "forecast_weight_status",
        "forecast_weight_selected_n",
        "forecast_mean_return_direct_source",
    ]
    post_bridge = read_csv_if_exists(post_backfill_bridge_path, parse_dates=("datetime",))
    if post_bridge.empty:
        focus["backfill_bridge_found"] = False
        focus["backfill_bridge_path"] = str(post_backfill_bridge_path)
        return focus
    present_bridge_columns = [column for column in bridge_columns if column in post_bridge.columns]
    post_bridge = post_bridge[present_bridge_columns].copy()
    post_bridge["instrument"] = post_bridge["instrument"].map(normalize_code)
    post_bridge["datetime"] = pd.to_datetime(post_bridge["datetime"])
    rename_map = {
        column: f"backfilled_bridge_{column}"
        for column in present_bridge_columns
        if column not in {"datetime", "instrument"}
    }
    post_bridge = post_bridge.rename(columns=rename_map)
    merged = focus.merge(post_bridge, on=["instrument", "datetime"], how="left")
    merged["backfill_bridge_found"] = merged.get("backfilled_bridge_route_a_status", pd.Series(index=merged.index)).notna()
    merged["backfill_bridge_path"] = str(post_backfill_bridge_path)
    merged["comparison_window"] = f"{DEFAULT_ROUTE_A_START_DATE}_{DEFAULT_ROUTE_A_END_DATE}"
    merged["no_momentum_overlay_flag"] = True
    route_a_mu_raw = numeric_series(merged.get("route_a_mu_raw", pd.Series(index=merged.index)))
    backfilled_weight = numeric_series(merged.get("backfilled_bridge_forecast_w", pd.Series(index=merged.index)))
    merged["post_backfill_allocated_weight_flag"] = backfilled_weight > ZERO_EPS
    merged["post_backfill_material_weight_flag"] = backfilled_weight >= 0.01
    merged["negative_raw_mu_allocated_weight_flag"] = (route_a_mu_raw < 0.0) & (backfilled_weight > ZERO_EPS)
    merged["backfilled_bridge_weight_bucket"] = backfilled_weight.map(weight_allocation_bucket)
    keep_columns = [
        "datetime",
        "instrument",
        "name",
        "focus_theme",
        "boom_day_flag",
        "actual_positive_specific_flag",
        "actual_excess_return",
        "route_a_target_date",
        "route_a_file_found",
        "route_a_mu_raw",
        "route_a_mu",
        "factor_industry_momentum",
        "factor_volume_z_20",
        "forecast_weight_mean_used",
        "backfilled_bridge_route_a_status",
        "backfilled_bridge_route_a_message",
        "backfilled_bridge_forecast_weight_mean_raw",
        "backfilled_bridge_forecast_weight_mean_slow",
        "backfilled_bridge_forecast_weight_mean_used",
        "backfilled_bridge_forecast_w",
        "backfilled_bridge_forecast_weight_status",
        "backfilled_bridge_forecast_weight_selected_n",
        "backfilled_bridge_forecast_mean_return_direct_source",
        "backfill_bridge_found",
        "post_backfill_allocated_weight_flag",
        "post_backfill_material_weight_flag",
        "negative_raw_mu_allocated_weight_flag",
        "backfilled_bridge_weight_bucket",
        "backfill_bridge_path",
        "comparison_window",
        "no_momentum_overlay_flag",
        "primary_pathology",
    ]
    present = [column for column in keep_columns if column in merged.columns]
    return merged[present].sort_values(["instrument", "datetime"])


def build_treatment_candidate_frame(daily: pd.DataFrame, alpha_injection: float) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    work = daily.copy()
    for column in ["route_a_mu_raw", "factor_industry_momentum", "factor_volume_z_20", "actual_excess_return"]:
        if column not in work.columns:
            work[column] = np.nan
    work["route_a_raw_negative_flag"] = numeric_series(work["route_a_mu_raw"]) < 0.0
    work["industry_momentum_std_gt_1_flag"] = numeric_series(work["factor_industry_momentum"]) > 1.0
    work["volume_z_20_gt_1_5_flag"] = numeric_series(work["factor_volume_z_20"]) > 1.5
    work["actual_positive_specific_flag"] = numeric_series(work.get("actual_excess_return", pd.Series(index=work.index))) > 0.0
    work["required_alpha_gain_to_flip_raw_mu"] = required_alpha_to_flip(
        work["route_a_mu_raw"],
        work["factor_industry_momentum"],
    )
    work["treatment_candidate_flag"] = (
        bool_series(work.get("route_a_file_found", pd.Series(False, index=work.index)), index=work.index)
        & work["route_a_raw_negative_flag"]
        & work["required_alpha_gain_to_flip_raw_mu"].notna()
    )
    work["treatment_v1_strict_trigger_flag"] = (
        work["treatment_candidate_flag"]
        & work["industry_momentum_std_gt_1_flag"]
        & work["volume_z_20_gt_1_5_flag"]
    )
    route_a_mu_raw = numeric_series(work["route_a_mu_raw"])
    injected_mu = route_a_mu_raw + float(alpha_injection)
    work["alpha_injection"] = float(alpha_injection)
    work["mu_patched_policy_v1"] = route_a_mu_raw.where(~work["treatment_v1_strict_trigger_flag"], np.maximum(0.0, injected_mu))
    work["mu_patched_candidate_scope"] = route_a_mu_raw.where(~work["treatment_candidate_flag"], np.maximum(0.0, injected_mu))
    work["policy_v1_recovered_to_nonnegative_flag"] = work["treatment_v1_strict_trigger_flag"] & (
        numeric_series(work["mu_patched_policy_v1"]) >= 0.0
    )
    work["candidate_scope_recovered_to_nonnegative_flag"] = work["treatment_candidate_flag"] & (
        numeric_series(work["mu_patched_candidate_scope"]) >= 0.0
    )
    selector = work["treatment_candidate_flag"] | work["treatment_v1_strict_trigger_flag"]
    return work.loc[selector].copy().sort_values(
        ["treatment_v1_strict_trigger_flag", "actual_positive_specific_flag", "boom_day_flag", "instrument", "datetime"],
        ascending=[False, False, False, True, True],
    )


def replay_student_t_nll_for_treatment_rows(
    treatment: pd.DataFrame,
    route_a_dir: Path,
    provider_uri: str,
    route_a_start_date: str,
    route_a_end_date: str,
    train_window: int,
    retrain_every: int,
    training_objective: str,
) -> pd.DataFrame:
    if treatment.empty:
        return treatment
    legacy = load_legacy_route_a_module()
    payload = route_a_quick_profile_payload(legacy, training_objective)
    config = legacy.route_a_config_from_payload(
        provider_uri=provider_uri,
        output_dir=str(route_a_dir),
        payload=payload,
    )
    records: list[dict[str, Any]] = []
    for instrument, rows in treatment.groupby("instrument", sort=False):
        try:
            daily_mean = legacy.load_from_qlib(
                provider_uri=config.provider_uri,
                benchmark=[normalize_code(instrument)],
                start_date=route_a_start_date,
                end_date=route_a_end_date,
                pct_change_unit="auto",
            )
            df_feat = legacy.build_features(daily_mean, config)
            if df_feat.empty:
                raise ValueError("empty treatment feature frame")
            effective_train_window, _warning = legacy.resolve_effective_train_window(train_window, len(df_feat), config)
            if effective_train_window < 1 or len(df_feat) <= effective_train_window:
                raise ValueError("insufficient feature rows for treatment replay")
            feature_cols = [column for column in df_feat.columns if column not in {"target", "target_date"}]
            x_all = np.column_stack([np.ones(len(df_feat)), df_feat[feature_cols].values])
            y_all = df_feat["target"].to_numpy(dtype=float)
            date_to_index = {pd.Timestamp(date): index for index, date in enumerate(pd.to_datetime(df_feat.index))}
            sample_indices = {
                row_index: date_to_index[pd.Timestamp(row["datetime"])]
                for row_index, row in rows.iterrows()
                if pd.Timestamp(row["datetime"]) in date_to_index
            }
            if not sample_indices:
                raise ValueError("forecast dates missing from treatment feature frame")
            max_sample_index = max(sample_indices.values())
            fit: Any | None = None
            init_params: np.ndarray | None = None
            fits_by_index: dict[int, Any] = {}
            for sample_index in range(effective_train_window, max_sample_index + 1):
                need_fit = fit is None or ((sample_index - effective_train_window) % int(retrain_every) == 0)
                if need_fit:
                    train_slice = slice(sample_index - effective_train_window, sample_index)
                    fit = legacy.fit_student_t_mu_sigma(
                        x_all[train_slice],
                        y_all[train_slice],
                        config,
                        init_params=init_params,
                    )
                    init_params = fit.optimizer_params.copy() if fit.optimizer_params is not None else None
                if sample_index in sample_indices.values():
                    fits_by_index[sample_index] = fit
            for row_index, row in rows.iterrows():
                sample_index = sample_indices.get(row_index)
                base_record = {
                    "row_index": row_index,
                    "nll_replay_status": "ok",
                    "nll_replay_message": "ok",
                }
                if sample_index is None or sample_index not in fits_by_index:
                    base_record.update(
                        {
                            "nll_replay_status": "missing_sample",
                            "nll_replay_message": "forecast date not replayed",
                        }
                    )
                    records.append(base_record)
                    continue
                fit = fits_by_index[sample_index]
                x_one = x_all[sample_index : sample_index + 1]
                x_scaled = x_one.copy()
                x_scaled[:, 1:] = (x_scaled[:, 1:] - fit.feature_mean) / fit.feature_scale
                mu_raw_recomputed = float((x_scaled @ fit.beta)[0] * fit.target_scale)
                _mu, sigma, nu = legacy.predict_params(x_one, fit)
                sigma_value = float(sigma[0])
                nu_value = float(nu[0])
                y_true = float(y_all[sample_index])
                file_mu_raw = safe_float(row.get("route_a_mu_raw"), mu_raw_recomputed)
                policy_mu = safe_float(row.get("mu_patched_policy_v1"), file_mu_raw)
                candidate_mu = safe_float(row.get("mu_patched_candidate_scope"), file_mu_raw)
                nll_raw = float(-legacy.student_t.logpdf(y_true, df=nu_value, loc=file_mu_raw, scale=sigma_value))
                nll_policy = float(-legacy.student_t.logpdf(y_true, df=nu_value, loc=policy_mu, scale=sigma_value))
                nll_candidate = float(-legacy.student_t.logpdf(y_true, df=nu_value, loc=candidate_mu, scale=sigma_value))
                records.append(
                    {
                        **base_record,
                        "mu_raw_recomputed": mu_raw_recomputed,
                        "mu_raw_file_minus_recomputed": file_mu_raw - mu_raw_recomputed,
                        "sigma_recomputed": sigma_value,
                        "nu_recomputed": nu_value,
                        "y_true_route_a_next_return": y_true,
                        "nll_at_route_a_mu_raw": nll_raw,
                        "nll_at_policy_v1_mu": nll_policy,
                        "nll_at_candidate_scope_mu": nll_candidate,
                        "nll_improvement_policy_v1": nll_raw - nll_policy,
                        "nll_improvement_candidate_scope": nll_raw - nll_candidate,
                    }
                )
        except Exception as exc:
            for row_index in rows.index:
                records.append(
                    {
                        "row_index": row_index,
                        "nll_replay_status": "error",
                        "nll_replay_message": str(exc),
                    }
                )
    replay = pd.DataFrame(records).drop_duplicates(subset=["row_index"], keep="last")
    result = treatment.reset_index(names="row_index").merge(replay, on="row_index", how="left").drop(columns=["row_index"])
    return result


def build_treatment_simulation_v1(
    daily: pd.DataFrame,
    alpha_injection: float,
    route_a_dir: Path,
    provider_uri: str,
    route_a_start_date: str,
    route_a_end_date: str,
    train_window: int,
    retrain_every: int,
    training_objective: str,
) -> pd.DataFrame:
    treatment = build_treatment_candidate_frame(daily, alpha_injection)
    if treatment.empty:
        return treatment
    treatment = replay_student_t_nll_for_treatment_rows(
        treatment,
        route_a_dir,
        provider_uri,
        route_a_start_date,
        route_a_end_date,
        train_window,
        retrain_every,
        training_objective,
    )
    keep_columns = [
        "datetime",
        "instrument",
        "name",
        "focus_theme",
        "boom_day_flag",
        "actual_positive_specific_flag",
        "actual_excess_return",
        "route_a_target_date",
        "route_a_mu_raw",
        "route_a_mu",
        "factor_industry_momentum",
        "factor_volume_z_20",
        "required_alpha_gain_to_flip_raw_mu",
        "alpha_injection",
        "industry_momentum_std_gt_1_flag",
        "volume_z_20_gt_1_5_flag",
        "treatment_candidate_flag",
        "treatment_v1_strict_trigger_flag",
        "mu_patched_policy_v1",
        "mu_patched_candidate_scope",
        "policy_v1_recovered_to_nonnegative_flag",
        "candidate_scope_recovered_to_nonnegative_flag",
        "forecast_weight_mean_used",
        "primary_pathology",
        "nll_replay_status",
        "nll_replay_message",
        "mu_raw_recomputed",
        "mu_raw_file_minus_recomputed",
        "sigma_recomputed",
        "nu_recomputed",
        "y_true_route_a_next_return",
        "nll_at_route_a_mu_raw",
        "nll_at_policy_v1_mu",
        "nll_at_candidate_scope_mu",
        "nll_improvement_policy_v1",
        "nll_improvement_candidate_scope",
    ]
    present = [column for column in keep_columns if column in treatment.columns]
    return treatment[present].sort_values(
        ["treatment_v1_strict_trigger_flag", "actual_positive_specific_flag", "boom_day_flag", "instrument", "datetime"],
        ascending=[False, False, False, True, True],
    )


def build_v2_trigger_sensitivity(treatment_simulation: pd.DataFrame) -> pd.DataFrame:
    if treatment_simulation.empty:
        return pd.DataFrame()
    work = treatment_simulation.copy()
    for column in [
        "route_a_mu_raw",
        "factor_industry_momentum",
        "factor_volume_z_20",
        "nll_improvement_candidate_scope",
    ]:
        if column not in work.columns:
            work[column] = np.nan
    candidate_flag = bool_series(work.get("treatment_candidate_flag", pd.Series(False, index=work.index)), index=work.index)
    raw_negative_flag = numeric_series(work["route_a_mu_raw"]) < 0.0
    industry = numeric_series(work["factor_industry_momentum"])
    volume = numeric_series(work["factor_volume_z_20"])
    actual_positive = bool_series(work.get("actual_positive_specific_flag", pd.Series(False, index=work.index)), index=work.index)
    boom = bool_series(work.get("boom_day_flag", pd.Series(False, index=work.index)), index=work.index)
    scenarios = [
        {
            "trigger_name": "T1_Strict_industry_gt_1_volume_gt_1_5",
            "strictness_rank": 1,
            "industry_momentum_threshold": 1.0,
            "volume_z_20_threshold": 1.5,
            "volume_gate_enabled": True,
        },
        {
            "trigger_name": "T2_Moderate_industry_gt_0_5_volume_gt_1",
            "strictness_rank": 2,
            "industry_momentum_threshold": 0.5,
            "volume_z_20_threshold": 1.0,
            "volume_gate_enabled": True,
        },
        {
            "trigger_name": "T3_Momentum_Only_industry_gt_0",
            "strictness_rank": 3,
            "industry_momentum_threshold": 0.0,
            "volume_z_20_threshold": np.nan,
            "volume_gate_enabled": False,
        },
    ]
    records: list[dict[str, Any]] = []
    for scenario in scenarios:
        trigger = candidate_flag & raw_negative_flag & (industry > float(scenario["industry_momentum_threshold"]))
        if bool(scenario["volume_gate_enabled"]):
            trigger = trigger & (volume > float(scenario["volume_z_20_threshold"]))
        positive_specific_trigger = trigger & actual_positive
        positive_boom_trigger = trigger & actual_positive & boom
        nll_all = numeric_series(work.loc[trigger, "nll_improvement_candidate_scope"])
        nll_positive = numeric_series(work.loc[positive_specific_trigger, "nll_improvement_candidate_scope"])
        nll_positive_boom = numeric_series(work.loc[positive_boom_trigger, "nll_improvement_candidate_scope"])
        trigger_row_count = int(trigger.sum())
        positive_specific_count = int(positive_specific_trigger.sum())
        positive_boom_count = int(positive_boom_trigger.sum())
        mean_positive_boom_nll = float(nll_positive_boom.mean()) if not nll_positive_boom.dropna().empty else np.nan
        row_count_target_pass = 10 <= trigger_row_count <= 20
        nll_target_pass = bool(np.isfinite(mean_positive_boom_nll) and mean_positive_boom_nll > 5.0)
        records.append(
            {
                **scenario,
                "raw_mu_negative_required": True,
                "trigger_row_count": trigger_row_count,
                "trigger_positive_specific_count": positive_specific_count,
                "trigger_positive_boom_count": positive_boom_count,
                "mean_nll_improvement_all_triggered": float(nll_all.mean()) if not nll_all.dropna().empty else np.nan,
                "mean_nll_improvement_positive_specific": float(nll_positive.mean()) if not nll_positive.dropna().empty else np.nan,
                "mean_nll_improvement_positive_boom": mean_positive_boom_nll,
                "median_required_alpha_to_flip_raw_mu": float(
                    numeric_series(work.loc[trigger, "required_alpha_gain_to_flip_raw_mu"]).median()
                )
                if trigger.any() and "required_alpha_gain_to_flip_raw_mu" in work.columns
                else np.nan,
                "coverage_target_10_20_flag": row_count_target_pass,
                "nll_positive_boom_gt_5_flag": nll_target_pass,
                "mainline_candidate_flag": bool(row_count_target_pass and nll_target_pass),
                "selection_distance_to_15_rows": abs(trigger_row_count - 15),
            }
        )
    sensitivity = pd.DataFrame(records)
    sensitivity["recommended_threshold_flag"] = False
    candidates = sensitivity[sensitivity["mainline_candidate_flag"]].copy()
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["selection_distance_to_15_rows", "mean_nll_improvement_positive_boom", "strictness_rank"],
            ascending=[True, False, True],
        )
        sensitivity.loc[candidates.index[0], "recommended_threshold_flag"] = True
    return sensitivity


def parse_float_vector(value: object) -> np.ndarray:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.array([], dtype=float)
    return np.fromstring(str(value), sep=",", dtype=float)


def deserialize_day_covariance(day_frame: pd.DataFrame, cov_col: str) -> pd.DataFrame:
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


def optimize_single_day_weights(day_frame: pd.DataFrame, mean_col: str, cov_col: str) -> tuple[pd.Series, str, int]:
    from strategy.route_ab_moments_bridge.bridge import (
        DEFAULT_BALANCED_MIN_CROWDED,
        DEFAULT_HARD_BAND_PROTECTED_SLOTS,
        optimize_weights_hard_band,
    )

    instruments = day_frame["instrument"].astype(str).map(normalize_code).tolist()
    full_weights = pd.Series(0.0, index=pd.Index(instruments, dtype=object), dtype=float)
    covariance = deserialize_day_covariance(day_frame, cov_col)
    if covariance.empty:
        return full_weights, "optimizer_input_missing", 0
    day_by_code = day_frame.drop_duplicates("instrument", keep="last").set_index("instrument", drop=False)
    selected_order = [code for code in covariance.index.astype(str).tolist() if code in day_by_code.index]
    mean_returns = numeric_series(day_by_code[mean_col]).reindex(selected_order).fillna(0.0)
    covariance = covariance.reindex(index=selected_order, columns=selected_order)
    if len(mean_returns) == 0 or covariance.shape[0] != len(mean_returns):
        return full_weights, "optimizer_input_missing", 0
    protected_scores = optimizer_score_series(day_frame, "forecast_active_set_protect_score", mean_returns.index)
    crowded_scores = optimizer_score_series(day_frame, "forecast_active_set_crowded_score", mean_returns.index)
    balanced_scores = optimizer_score_series(day_frame, "forecast_active_set_balance_score", mean_returns.index)
    protected_slots = DEFAULT_HARD_BAND_PROTECTED_SLOTS if protected_scores is not None else 0
    try:
        weights, status = optimize_weights_hard_band(
            mean_returns,
            covariance,
            rf=0.0,
            max_weight=0.15,
            min_weight_floor=0.05,
            protected_scores=protected_scores,
            protected_slots=protected_slots,
            crowded_scores=crowded_scores,
            balanced_scores=balanced_scores,
            min_crowded_for_balance=DEFAULT_BALANCED_MIN_CROWDED,
        )
    except Exception as exc:
        return full_weights, f"optimizer_exception:{type(exc).__name__}", 0
    if not weights.empty:
        full_weights = pd.Series(0.0, index=mean_returns.index, dtype=float)
        full_weights.loc[weights.index] = pd.to_numeric(weights, errors="coerce").fillna(0.0)
        full_weights = full_weights.reindex(pd.Index(instruments, dtype=object)).fillna(0.0)
    return full_weights, str(status), int((full_weights > 1e-12).sum())


def optimize_weight_frame_for_dates(frame: pd.DataFrame, mean_col: str, cov_col: str, weight_col: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    status_col = f"{weight_col}_status"
    selected_col = f"{weight_col}_selected_n"
    for date_value, day_frame in frame.groupby("datetime", sort=True):
        day_frame = day_frame.copy()
        weights, status, selected_n = optimize_single_day_weights(day_frame, mean_col, cov_col)
        for code, weight in weights.items():
            records.append(
                {
                    "datetime": pd.Timestamp(date_value),
                    "instrument": normalize_code(code),
                    weight_col: float(weight),
                    status_col: status,
                    selected_col: int(selected_n),
                }
            )
    return pd.DataFrame(records)


def is_ai_communication_theme(value: object) -> bool:
    text = str(value).lower()
    return "ai" in text or "communication" in text


def build_alpha_to_weight_elasticity(treatment_simulation: pd.DataFrame, post_backfill_bridge_path: Path) -> pd.DataFrame:
    if treatment_simulation.empty:
        return pd.DataFrame()
    candidate_flag = bool_series(
        treatment_simulation.get("treatment_candidate_flag", pd.Series(False, index=treatment_simulation.index)),
        index=treatment_simulation.index,
    )
    candidates = treatment_simulation.loc[candidate_flag].copy()
    if candidates.empty:
        return pd.DataFrame()
    bridge = read_csv_if_exists(post_backfill_bridge_path, parse_dates=("datetime",))
    if bridge.empty:
        return pd.DataFrame()
    bridge["instrument"] = bridge["instrument"].map(normalize_code)
    bridge["datetime"] = pd.to_datetime(bridge["datetime"])
    mean_col = "forecast_weight_mean_used" if "forecast_weight_mean_used" in bridge.columns else "forecast_mean_return_robust"
    if mean_col not in bridge.columns:
        return pd.DataFrame()
    cov_col = "forecast_cov_matrix_robust" if "forecast_cov_matrix_robust" in bridge.columns else "forecast_cov_matrix"
    if cov_col not in bridge.columns:
        return pd.DataFrame()
    candidates["instrument"] = candidates["instrument"].map(normalize_code)
    candidates["datetime"] = pd.to_datetime(candidates["datetime"])
    candidate_keys = pd.MultiIndex.from_frame(candidates[["datetime", "instrument"]].drop_duplicates())
    relevant_dates = candidates["datetime"].dropna().drop_duplicates()
    bridge_focus = bridge.loc[bridge["datetime"].isin(relevant_dates)].copy()
    if bridge_focus.empty:
        return pd.DataFrame()
    bridge_focus["alpha_scan_base_mean"] = numeric_series(bridge_focus[mean_col]).fillna(0.0)
    base_weights = optimize_weight_frame_for_dates(
        bridge_focus,
        mean_col="alpha_scan_base_mean",
        cov_col=cov_col,
        weight_col="base_resim_forecast_w",
    )
    base_context_columns = [
        "datetime",
        "instrument",
        mean_col,
        "forecast_w",
        "forecast_weight_status",
        "forecast_weight_selected_n",
    ]
    base_context_columns = [column for column in base_context_columns if column in bridge_focus.columns]
    base_context = bridge_focus[base_context_columns].drop_duplicates(["datetime", "instrument"], keep="last")
    candidate_columns = [
        "datetime",
        "instrument",
        "name",
        "focus_theme",
        "boom_day_flag",
        "actual_positive_specific_flag",
        "route_a_mu_raw",
        "factor_industry_momentum",
        "factor_volume_z_20",
        "required_alpha_gain_to_flip_raw_mu",
        "treatment_v1_strict_trigger_flag",
    ]
    candidate_context = candidates[[column for column in candidate_columns if column in candidates.columns]].drop_duplicates(
        ["datetime", "instrument"],
        keep="last",
    )
    records: list[pd.DataFrame] = []
    for alpha_level in ALPHA_WEIGHT_SCAN_LEVELS:
        scan_frame = bridge_focus.copy()
        scan_frame["alpha_scan_optimizer_mean"] = scan_frame["alpha_scan_base_mean"]
        scan_keys = pd.MultiIndex.from_frame(scan_frame[["datetime", "instrument"]])
        patch_mask = scan_keys.isin(candidate_keys)
        scan_frame.loc[patch_mask, "alpha_scan_optimizer_mean"] = float(alpha_level)
        patched_weights = optimize_weight_frame_for_dates(
            scan_frame,
            mean_col="alpha_scan_optimizer_mean",
            cov_col=cov_col,
            weight_col="patched_forecast_w",
        )
        current = candidate_context.merge(base_context, on=["datetime", "instrument"], how="left")
        current = current.merge(base_weights, on=["datetime", "instrument"], how="left")
        current = current.merge(patched_weights, on=["datetime", "instrument"], how="left")
        current["mu_patch_level"] = float(alpha_level)
        current["base_optimizer_mean"] = numeric_series(current.get(mean_col, pd.Series(index=current.index)))
        current["patched_optimizer_mean"] = float(alpha_level)
        current["existing_bridge_forecast_w"] = numeric_series(current.get("forecast_w", pd.Series(index=current.index))).fillna(0.0)
        current["base_resim_forecast_w"] = numeric_series(current.get("base_resim_forecast_w", pd.Series(index=current.index))).fillna(0.0)
        current["patched_forecast_w"] = numeric_series(current.get("patched_forecast_w", pd.Series(index=current.index))).fillna(0.0)
        current["marginal_weight"] = current["patched_forecast_w"] - current["base_resim_forecast_w"]
        current["patched_allocated_flag"] = current["patched_forecast_w"] > ZERO_EPS
        current["is_ai_communication_flag"] = current.get("focus_theme", pd.Series(index=current.index)).map(is_ai_communication_theme)
        current["daily_candidate_weight_sum"] = current.groupby("datetime")["patched_forecast_w"].transform("sum")
        current["daily_candidate_marginal_weight_sum"] = current.groupby("datetime")["marginal_weight"].transform("sum")
        ai_weight = current["patched_forecast_w"].where(current["is_ai_communication_flag"], 0.0)
        ai_marginal = current["marginal_weight"].where(current["is_ai_communication_flag"], 0.0)
        current["daily_ai_communication_weight_sum"] = ai_weight.groupby(current["datetime"]).transform("sum")
        current["daily_ai_communication_marginal_weight_sum"] = ai_marginal.groupby(current["datetime"]).transform("sum")
        current["level_candidate_weight_sum"] = float(current["patched_forecast_w"].sum())
        current["level_candidate_marginal_weight_sum"] = float(current["marginal_weight"].sum())
        current["level_ai_communication_weight_sum"] = float(ai_weight.sum())
        current["level_ai_communication_marginal_weight_sum"] = float(ai_marginal.sum())
        current["level_allocated_candidate_rows"] = int(current["patched_allocated_flag"].sum())
        current["level_allocated_ai_communication_rows"] = int(
            (current["patched_allocated_flag"] & current["is_ai_communication_flag"]).sum()
        )
        records.append(current)
    result = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    rename_map = {
        mean_col: "bridge_optimizer_mean_source_value",
        "forecast_weight_status": "existing_bridge_weight_status",
        "forecast_weight_selected_n": "existing_bridge_selected_n",
        "base_resim_forecast_w_status": "base_resim_optimizer_status",
        "base_resim_forecast_w_selected_n": "base_resim_selected_n",
        "patched_forecast_w_status": "patched_optimizer_status",
        "patched_forecast_w_selected_n": "patched_selected_n",
    }
    result = result.rename(columns={key: value for key, value in rename_map.items() if key in result.columns})
    keep_columns = [
        "mu_patch_level",
        "datetime",
        "instrument",
        "name",
        "focus_theme",
        "is_ai_communication_flag",
        "boom_day_flag",
        "actual_positive_specific_flag",
        "route_a_mu_raw",
        "factor_industry_momentum",
        "factor_volume_z_20",
        "required_alpha_gain_to_flip_raw_mu",
        "treatment_v1_strict_trigger_flag",
        "bridge_optimizer_mean_source_value",
        "base_optimizer_mean",
        "patched_optimizer_mean",
        "existing_bridge_forecast_w",
        "base_resim_forecast_w",
        "patched_forecast_w",
        "marginal_weight",
        "patched_allocated_flag",
        "existing_bridge_weight_status",
        "existing_bridge_selected_n",
        "base_resim_optimizer_status",
        "base_resim_selected_n",
        "patched_optimizer_status",
        "patched_selected_n",
        "daily_candidate_weight_sum",
        "daily_candidate_marginal_weight_sum",
        "daily_ai_communication_weight_sum",
        "daily_ai_communication_marginal_weight_sum",
        "level_candidate_weight_sum",
        "level_candidate_marginal_weight_sum",
        "level_ai_communication_weight_sum",
        "level_ai_communication_marginal_weight_sum",
        "level_allocated_candidate_rows",
        "level_allocated_ai_communication_rows",
    ]
    present = [column for column in keep_columns if column in result.columns]
    return result[present].sort_values(["mu_patch_level", "datetime", "instrument"])


def summarize_alpha_weight_elasticity(alpha_weight: pd.DataFrame) -> pd.DataFrame:
    if alpha_weight.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for alpha_level, group in alpha_weight.groupby("mu_patch_level", sort=True):
        rows.append(
            {
                "mu_patch_level": float(alpha_level),
                "candidate_row_count": int(group.shape[0]),
                "candidate_allocated_rows": int(bool_series(group.get("patched_allocated_flag", pd.Series(dtype=bool))).sum()),
                "ai_communication_allocated_rows": int(
                    (
                        bool_series(group.get("patched_allocated_flag", pd.Series(dtype=bool)))
                        & bool_series(group.get("is_ai_communication_flag", pd.Series(dtype=bool)))
                    ).sum()
                ),
                "candidate_weight_sum": float(numeric_series(group.get("patched_forecast_w", pd.Series(dtype=float))).sum()),
                "candidate_marginal_weight_sum": float(numeric_series(group.get("marginal_weight", pd.Series(dtype=float))).sum()),
                "ai_communication_weight_sum": float(
                    numeric_series(
                        group.get("patched_forecast_w", pd.Series(dtype=float)).where(
                            bool_series(group.get("is_ai_communication_flag", pd.Series(dtype=bool))),
                            0.0,
                        )
                    ).sum()
                ),
                "ai_communication_marginal_weight_sum": float(
                    numeric_series(
                        group.get("marginal_weight", pd.Series(dtype=float)).where(
                            bool_series(group.get("is_ai_communication_flag", pd.Series(dtype=bool))),
                            0.0,
                        )
                    ).sum()
                ),
                "max_single_candidate_weight": float(numeric_series(group.get("patched_forecast_w", pd.Series(dtype=float))).max()),
            }
        )
    return pd.DataFrame(rows)


def v2_t3_min_effective_trigger_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(False, index=frame.index)
    candidate_flag = bool_series(frame.get("treatment_candidate_flag", pd.Series(False, index=frame.index)), index=frame.index)
    raw_negative = numeric_series(frame.get("route_a_mu_raw", pd.Series(np.nan, index=frame.index))) < 0.0
    industry_positive = numeric_series(frame.get("factor_industry_momentum", pd.Series(np.nan, index=frame.index))) > 0.0
    return candidate_flag & raw_negative & industry_positive


def build_shadow_pnl_backtest(
    treatment_simulation: pd.DataFrame,
    post_backfill_bridge_path: Path,
    patch_level: float = V2_T3_MIN_EFFECTIVE_PATCH,
) -> pd.DataFrame:
    if treatment_simulation.empty:
        return pd.DataFrame()
    trigger_mask = v2_t3_min_effective_trigger_mask(treatment_simulation)
    triggers = treatment_simulation.loc[trigger_mask].copy()
    if triggers.empty:
        return pd.DataFrame()
    bridge = read_csv_if_exists(post_backfill_bridge_path, parse_dates=("datetime",))
    if bridge.empty or "next_return" not in bridge.columns:
        return pd.DataFrame()
    bridge = bridge.copy()
    bridge["instrument"] = bridge["instrument"].map(normalize_code)
    bridge["datetime"] = pd.to_datetime(bridge["datetime"])
    mean_col = "forecast_weight_mean_used" if "forecast_weight_mean_used" in bridge.columns else "forecast_mean_return_robust"
    cov_col = "forecast_cov_matrix_robust" if "forecast_cov_matrix_robust" in bridge.columns else "forecast_cov_matrix"
    if mean_col not in bridge.columns or cov_col not in bridge.columns:
        return pd.DataFrame()

    triggers["instrument"] = triggers["instrument"].map(normalize_code)
    triggers["datetime"] = pd.to_datetime(triggers["datetime"])
    trigger_columns = [
        "datetime",
        "instrument",
        "name",
        "focus_theme",
        "boom_day_flag",
        "actual_positive_specific_flag",
        "route_a_target_date",
        "route_a_mu_raw",
        "factor_industry_momentum",
        "factor_volume_z_20",
        "required_alpha_gain_to_flip_raw_mu",
    ]
    trigger_context = triggers[[column for column in trigger_columns if column in triggers.columns]].drop_duplicates(
        ["datetime", "instrument"],
        keep="last",
    )
    trigger_context["v2_patch_name"] = "V2_T3_MIN_EFFECTIVE_PATCH"
    trigger_context["v2_patch_level"] = float(patch_level)
    trigger_context["v2_t3_min_effective_trigger_flag"] = True
    trigger_context["is_ai_communication_flag"] = trigger_context.get(
        "focus_theme",
        pd.Series("", index=trigger_context.index),
    ).map(is_ai_communication_theme)
    trigger_keys = pd.MultiIndex.from_frame(trigger_context[["datetime", "instrument"]])

    bridge["shadow_base_optimizer_mean"] = numeric_series(bridge[mean_col]).fillna(0.0)
    bridge_keys = pd.MultiIndex.from_frame(bridge[["datetime", "instrument"]])
    bridge["v2_t3_min_effective_trigger_flag"] = bridge_keys.isin(trigger_keys)
    bridge["shadow_patched_optimizer_mean"] = bridge["shadow_base_optimizer_mean"]
    patch_mask = bool_series(bridge["v2_t3_min_effective_trigger_flag"], index=bridge.index)
    bridge.loc[patch_mask, "shadow_patched_optimizer_mean"] = np.maximum(
        numeric_series(bridge.loc[patch_mask, "shadow_base_optimizer_mean"]),
        float(patch_level),
    )

    base_weights = optimize_weight_frame_for_dates(
        bridge,
        mean_col="shadow_base_optimizer_mean",
        cov_col=cov_col,
        weight_col="shadow_base_w",
    )
    patched_weights = optimize_weight_frame_for_dates(
        bridge,
        mean_col="shadow_patched_optimizer_mean",
        cov_col=cov_col,
        weight_col="shadow_patched_w",
    )
    working_columns = [
        "datetime",
        "instrument",
        "next_return",
        "shadow_base_optimizer_mean",
        "shadow_patched_optimizer_mean",
        "v2_t3_min_effective_trigger_flag",
    ]
    working = bridge[[column for column in working_columns if column in bridge.columns]].copy()
    working = working.merge(base_weights, on=["datetime", "instrument"], how="left")
    working = working.merge(patched_weights, on=["datetime", "instrument"], how="left")
    working = working.merge(trigger_context, on=["datetime", "instrument"], how="left", suffixes=("", "_trigger"))
    if "v2_t3_min_effective_trigger_flag_trigger" in working.columns:
        trigger_override = working["v2_t3_min_effective_trigger_flag_trigger"]
        working["v2_t3_min_effective_trigger_flag"] = bool_series(
            trigger_override.where(trigger_override.notna(), working["v2_t3_min_effective_trigger_flag"]),
            index=working.index,
        )
        working = working.drop(columns=["v2_t3_min_effective_trigger_flag_trigger"])
    working["is_ai_communication_flag"] = bool_series(
        working.get("is_ai_communication_flag", pd.Series(False, index=working.index)),
        index=working.index,
    )

    daily_rows: list[dict[str, Any]] = []
    for date_value, day_frame in working.groupby("datetime", sort=True):
        returns = numeric_series(day_frame.get("next_return", pd.Series(index=day_frame.index))).fillna(0.0)
        base_weight = numeric_series(day_frame.get("shadow_base_w", pd.Series(index=day_frame.index))).fillna(0.0)
        patched_weight = numeric_series(day_frame.get("shadow_patched_w", pd.Series(index=day_frame.index))).fillna(0.0)
        weight_delta = patched_weight - base_weight
        trigger = bool_series(day_frame.get("v2_t3_min_effective_trigger_flag", pd.Series(False, index=day_frame.index)), index=day_frame.index)
        ai_trigger = trigger & bool_series(day_frame.get("is_ai_communication_flag", pd.Series(False, index=day_frame.index)), index=day_frame.index)
        base_daily_return = float((base_weight * returns).sum())
        patched_daily_return = float((patched_weight * returns).sum())
        active_return = patched_daily_return - base_daily_return
        trigger_names = day_frame.loc[trigger, "name"].dropna().astype(str).unique().tolist() if "name" in day_frame.columns else []
        trigger_codes = day_frame.loc[trigger, "instrument"].dropna().astype(str).unique().tolist()
        trigger_targets = (
            pd.to_datetime(day_frame.loc[trigger, "route_a_target_date"], errors="coerce")
            .dropna()
            .dt.strftime("%Y-%m-%d")
            .unique()
            .tolist()
            if "route_a_target_date" in day_frame.columns
            else []
        )
        candidate_alpha = float((weight_delta.where(trigger, 0.0) * returns).sum())
        ai_alpha = float((weight_delta.where(ai_trigger, 0.0) * returns).sum())
        daily_rows.append(
            {
                "datetime": pd.Timestamp(date_value),
                "patch_name": "V2_T3_MIN_EFFECTIVE_PATCH",
                "patch_level": float(patch_level),
                "trigger_name": "T3_Momentum_Only_industry_gt_0",
                "base_daily_return": base_daily_return,
                "patched_daily_return": patched_daily_return,
                "active_return": active_return,
                "base_weight_sum": float(base_weight.sum()),
                "patched_weight_sum": float(patched_weight.sum()),
                "t3_trigger_row_count": int(trigger.sum()),
                "t3_trigger_instruments": "|".join(sorted(trigger_codes)),
                "t3_trigger_names": "|".join(sorted(trigger_names)),
                "t3_trigger_target_dates": "|".join(sorted(trigger_targets)),
                "ai_communication_trigger_row_count": int(ai_trigger.sum()),
                "base_t3_trigger_weight": float(base_weight.where(trigger, 0.0).sum()),
                "patched_t3_trigger_weight": float(patched_weight.where(trigger, 0.0).sum()),
                "delta_t3_trigger_weight": float(weight_delta.where(trigger, 0.0).sum()),
                "base_ai_communication_weight": float(base_weight.where(ai_trigger, 0.0).sum()),
                "patched_ai_communication_weight": float(patched_weight.where(ai_trigger, 0.0).sum()),
                "delta_ai_communication_weight": float(weight_delta.where(ai_trigger, 0.0).sum()),
                "t3_trigger_base_pnl_contribution": float((base_weight.where(trigger, 0.0) * returns).sum()),
                "t3_trigger_patched_pnl_contribution": float((patched_weight.where(trigger, 0.0) * returns).sum()),
                "t3_trigger_alpha_attribution": candidate_alpha,
                "ai_communication_alpha_attribution": ai_alpha,
                "non_trigger_reallocation_attribution": active_return - candidate_alpha,
                "trigger_next_return_mean": float(returns.loc[trigger].mean()) if trigger.any() else np.nan,
                "trigger_next_return_max": float(returns.loc[trigger].max()) if trigger.any() else np.nan,
                "ai_communication_next_return_mean": float(returns.loc[ai_trigger].mean()) if ai_trigger.any() else np.nan,
                "base_optimizer_status": mode_text(day_frame.get("shadow_base_w_status", pd.Series(dtype=object))),
                "patched_optimizer_status": mode_text(day_frame.get("shadow_patched_w_status", pd.Series(dtype=object))),
                "base_selected_n": int(numeric_series(day_frame.get("shadow_base_w_selected_n", pd.Series(dtype=float))).max())
                if "shadow_base_w_selected_n" in day_frame.columns
                else 0,
                "patched_selected_n": int(numeric_series(day_frame.get("shadow_patched_w_selected_n", pd.Series(dtype=float))).max())
                if "shadow_patched_w_selected_n" in day_frame.columns
                else 0,
            }
        )
    equity = pd.DataFrame(daily_rows)
    if equity.empty:
        return equity
    equity = equity.sort_values("datetime")
    equity["base_nav"] = (1.0 + numeric_series(equity["base_daily_return"]).fillna(0.0)).cumprod()
    equity["patched_nav"] = (1.0 + numeric_series(equity["patched_daily_return"]).fillna(0.0)).cumprod()
    equity["active_nav_delta"] = equity["patched_nav"] - equity["base_nav"]
    equity["cumulative_active_return"] = equity["patched_nav"] / equity["base_nav"] - 1.0
    equity["trigger_day_flag"] = numeric_series(equity["t3_trigger_row_count"]).fillna(0.0) > 0.0
    equity["ai_burst_window_flag"] = equity["datetime"].between(pd.Timestamp("2026-04-02"), pd.Timestamp("2026-04-10"))
    return equity


def max_drawdown_from_returns(returns: pd.Series) -> float:
    values = numeric_series(returns).dropna()
    if values.empty:
        return np.nan
    nav = (1.0 + values).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    return float(drawdown.min())


def summarize_shadow_pnl_backtest(shadow_pnl: pd.DataFrame) -> pd.DataFrame:
    if shadow_pnl.empty:
        return pd.DataFrame()

    def summarize_scope(scope: str, frame: pd.DataFrame) -> dict[str, Any]:
        base_returns = numeric_series(frame.get("base_daily_return", pd.Series(dtype=float))).dropna()
        patched_returns = numeric_series(frame.get("patched_daily_return", pd.Series(dtype=float))).dropna()
        active_returns = numeric_series(frame.get("active_return", pd.Series(dtype=float))).dropna()
        base_cum = float((1.0 + base_returns).prod() - 1.0) if not base_returns.empty else np.nan
        patched_cum = float((1.0 + patched_returns).prod() - 1.0) if not patched_returns.empty else np.nan
        active_std = float(active_returns.std(ddof=0)) if not active_returns.empty else np.nan
        active_mean = float(active_returns.mean()) if not active_returns.empty else np.nan
        return {
            "scope": scope,
            "day_count": int(frame.shape[0]),
            "trigger_day_count": int(bool_series(frame.get("trigger_day_flag", pd.Series(dtype=bool))).sum()),
            "base_cumulative_return": base_cum,
            "patched_cumulative_return": patched_cum,
            "delta_cumulative_return": patched_cum - base_cum if np.isfinite(base_cum) and np.isfinite(patched_cum) else np.nan,
            "sum_active_return": float(active_returns.sum()) if not active_returns.empty else np.nan,
            "active_return_mean": active_mean,
            "active_return_sharpe_like": float(active_mean / active_std * np.sqrt(252.0)) if np.isfinite(active_mean) and np.isfinite(active_std) and active_std > ZERO_EPS else np.nan,
            "base_max_drawdown": max_drawdown_from_returns(base_returns),
            "patched_max_drawdown": max_drawdown_from_returns(patched_returns),
            "sum_t3_trigger_alpha_attribution": float(numeric_series(frame.get("t3_trigger_alpha_attribution", pd.Series(dtype=float))).sum()),
            "sum_ai_communication_alpha_attribution": float(numeric_series(frame.get("ai_communication_alpha_attribution", pd.Series(dtype=float))).sum()),
            "sum_non_trigger_reallocation_attribution": float(numeric_series(frame.get("non_trigger_reallocation_attribution", pd.Series(dtype=float))).sum()),
            "max_delta_t3_trigger_weight": float(numeric_series(frame.get("delta_t3_trigger_weight", pd.Series(dtype=float))).max()),
            "max_delta_ai_communication_weight": float(numeric_series(frame.get("delta_ai_communication_weight", pd.Series(dtype=float))).max()),
        }

    rows = [summarize_scope("overall", shadow_pnl)]
    trigger_frame = shadow_pnl.loc[bool_series(shadow_pnl.get("trigger_day_flag", pd.Series(False, index=shadow_pnl.index)), index=shadow_pnl.index)]
    if not trigger_frame.empty:
        rows.append(summarize_scope("trigger_dates_only", trigger_frame))
    ai_burst = shadow_pnl.loc[bool_series(shadow_pnl.get("ai_burst_window_flag", pd.Series(False, index=shadow_pnl.index)), index=shadow_pnl.index)]
    if not ai_burst.empty:
        rows.append(summarize_scope("ai_burst_window_20260402_20260410", ai_burst))
    ai_trigger = shadow_pnl.loc[numeric_series(shadow_pnl.get("ai_communication_trigger_row_count", pd.Series(0, index=shadow_pnl.index))) > 0.0]
    if not ai_trigger.empty:
        rows.append(summarize_scope("ai_communication_trigger_dates", ai_trigger))
    return pd.DataFrame(rows)


def build_final_freeze_summary(
    v2_trigger_sensitivity: pd.DataFrame,
    alpha_weight_elasticity: pd.DataFrame,
    shadow_pnl_summary: pd.DataFrame,
) -> pd.DataFrame:
    alpha_summary = summarize_alpha_weight_elasticity(alpha_weight_elasticity)
    min_level = alpha_summary[
        numeric_series(alpha_summary.get("ai_communication_weight_sum", pd.Series(dtype=float))) > ZERO_EPS
    ] if not alpha_summary.empty else pd.DataFrame()
    recommended = v2_trigger_sensitivity[
        bool_series(v2_trigger_sensitivity.get("recommended_threshold_flag", pd.Series(dtype=bool)))
    ] if not v2_trigger_sensitivity.empty else pd.DataFrame()
    trigger_row = recommended.iloc[0].to_dict() if not recommended.empty else {}
    shadow_overall = shadow_pnl_summary.loc[shadow_pnl_summary.get("scope", pd.Series(dtype=object)).eq("overall")]
    shadow_trigger = shadow_pnl_summary.loc[shadow_pnl_summary.get("scope", pd.Series(dtype=object)).eq("trigger_dates_only")]
    overall_row = shadow_overall.iloc[0].to_dict() if not shadow_overall.empty else {}
    trigger_scope_row = shadow_trigger.iloc[0].to_dict() if not shadow_trigger.empty else {}
    min_level_row = min_level.iloc[0].to_dict() if not min_level.empty else {}
    return pd.DataFrame(
        [
            {
                "final_recommendation": "recommend_mainline_defensive_industry_momentum_mean_correction_layer",
                "patch_name": "V2_T3_MIN_EFFECTIVE_PATCH",
                "patch_level": V2_T3_MIN_EFFECTIVE_PATCH,
                "trigger_name": trigger_row.get("trigger_name", "T3_Momentum_Only_industry_gt_0"),
                "trigger_logic": "route_a_mu_raw < 0 and factor_industry_momentum > 0; no volume gate",
                "min_effective_scan_level": min_level_row.get("mu_patch_level", np.nan),
                "min_effective_ai_communication_allocated_rows": min_level_row.get("ai_communication_allocated_rows", np.nan),
                "min_effective_ai_communication_weight_sum": min_level_row.get("ai_communication_weight_sum", np.nan),
                "v2_trigger_row_count": trigger_row.get("trigger_row_count", np.nan),
                "v2_positive_boom_count": trigger_row.get("trigger_positive_boom_count", np.nan),
                "v2_mean_nll_improvement_positive_boom": trigger_row.get("mean_nll_improvement_positive_boom", np.nan),
                "shadow_overall_delta_cumulative_return": overall_row.get("delta_cumulative_return", np.nan),
                "shadow_trigger_dates_delta_cumulative_return": trigger_scope_row.get("delta_cumulative_return", np.nan),
                "shadow_trigger_dates_ai_communication_alpha_attribution": trigger_scope_row.get(
                    "sum_ai_communication_alpha_attribution",
                    np.nan,
                ),
                "mainline_posture": "feature_flag_default_off_then_shadow_monitor_before_enablement",
                "governance_note": "minimum effective dose chosen from real hard-band MVO response; avoids aggressive 0.015 exposure jump",
            }
        ]
    )


def cleanup_route_a_intermediate_outputs(out_dir: Path) -> None:
    for name in ROUTE_A_INTERMEDIATE_OUTPUT_NAMES:
        path = out_dir / name
        if path.exists():
            path.unlink()
            print(f"[INFO] removed intermediate {path}")


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._\n"
    present = [column for column in columns if column in frame.columns]
    if not present:
        return "_No requested columns._\n"
    sample = frame[present].head(max_rows).copy()
    for column in sample.columns:
        if pd.api.types.is_float_dtype(sample[column]):
            sample[column] = sample[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.6g}")
        else:
            sample[column] = sample[column].fillna("").astype(str)
    header = "| " + " | ".join(sample.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(sample.columns)) + " |"
    body = ["| " + " | ".join(str(row[column]) for column in sample.columns) + " |" for _, row in sample.iterrows()]
    return "\n".join([header, divider, *body]) + "\n"


def write_report(
    report_path: Path,
    summary: pd.DataFrame,
    missing: pd.DataFrame,
    out_dir: Path,
    formula_stress: pd.DataFrame | None = None,
    momentum_boost: pd.DataFrame | None = None,
    backfill_impact: pd.DataFrame | None = None,
    treatment_simulation: pd.DataFrame | None = None,
    v2_trigger_sensitivity: pd.DataFrame | None = None,
    alpha_weight_elasticity: pd.DataFrame | None = None,
    shadow_pnl: pd.DataFrame | None = None,
    shadow_pnl_summary: pd.DataFrame | None = None,
    final_summary: pd.DataFrame | None = None,
    final_freeze: bool = False,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    formula_stress = formula_stress if formula_stress is not None else pd.DataFrame()
    momentum_boost = momentum_boost if momentum_boost is not None else pd.DataFrame()
    backfill_impact = backfill_impact if backfill_impact is not None else pd.DataFrame()
    treatment_simulation = treatment_simulation if treatment_simulation is not None else pd.DataFrame()
    v2_trigger_sensitivity = v2_trigger_sensitivity if v2_trigger_sensitivity is not None else pd.DataFrame()
    alpha_weight_elasticity = alpha_weight_elasticity if alpha_weight_elasticity is not None else pd.DataFrame()
    shadow_pnl = shadow_pnl if shadow_pnl is not None else pd.DataFrame()
    shadow_pnl_summary = shadow_pnl_summary if shadow_pnl_summary is not None else pd.DataFrame()
    final_summary = final_summary if final_summary is not None else pd.DataFrame()
    alpha_weight_summary = summarize_alpha_weight_elasticity(alpha_weight_elasticity)
    row_count = int(summary.shape[0]) if not summary.empty else 0
    missing_count = int((~bool_series(missing.get("route_a_file_found", pd.Series(dtype=bool)))).sum()) if not missing.empty else 0
    observed_count = int(bool_series(summary.get("route_a_file_found", pd.Series(dtype=bool))).sum()) if not summary.empty else 0
    observed_raw_negative_count = int(
        summary.get("observed_positive_specific_primary_pathology", pd.Series(dtype=object))
        .fillna("")
        .eq("student_t_raw_mean_negative_on_positive_specific")
        .sum()
    ) if not summary.empty else 0
    low_route_a_boom_coverage_count = int(
        (numeric_series(summary.get("route_a_boom_coverage_rate", pd.Series(dtype=float))) < 0.50).sum()
    ) if not summary.empty else 0
    boost_candidate_count = int(momentum_boost.shape[0]) if not momentum_boost.empty else 0
    boost_feasible_count = int(
        bool_series(momentum_boost.get("momentum_boost_feasible_flag", pd.Series(dtype=bool))).sum()
    ) if not momentum_boost.empty else 0
    backfill_rows = int(backfill_impact.shape[0]) if not backfill_impact.empty else 0
    backfill_bridge_found_rows = int(
        bool_series(backfill_impact.get("backfill_bridge_found", pd.Series(dtype=bool))).sum()
    ) if not backfill_impact.empty else 0
    backfill_allocated_rows = int(
        bool_series(backfill_impact.get("post_backfill_allocated_weight_flag", pd.Series(dtype=bool))).sum()
    ) if not backfill_impact.empty else 0
    backfill_negative_allocated_rows = int(
        bool_series(backfill_impact.get("negative_raw_mu_allocated_weight_flag", pd.Series(dtype=bool))).sum()
    ) if not backfill_impact.empty else 0
    backfill_negative_raw_mask = (
        numeric_series(backfill_impact.get("route_a_mu_raw", pd.Series(dtype=float))) < 0.0
    ) if not backfill_impact.empty else pd.Series(dtype=bool)
    backfill_negative_raw_rows = int(backfill_negative_raw_mask.sum()) if not backfill_impact.empty else 0
    backfill_negative_raw_weight_sum = (
        float(numeric_series(backfill_impact.loc[backfill_negative_raw_mask, "backfilled_bridge_forecast_w"]).fillna(0.0).sum())
        if not backfill_impact.empty and "backfilled_bridge_forecast_w" in backfill_impact.columns
        else np.nan
    )
    treatment_candidate_count = int(
        bool_series(treatment_simulation.get("treatment_candidate_flag", pd.Series(dtype=bool))).sum()
    ) if not treatment_simulation.empty else 0
    treatment_trigger_count = int(
        bool_series(treatment_simulation.get("treatment_v1_strict_trigger_flag", pd.Series(dtype=bool))).sum()
    ) if not treatment_simulation.empty else 0
    treatment_policy_recovered_count = int(
        bool_series(treatment_simulation.get("policy_v1_recovered_to_nonnegative_flag", pd.Series(dtype=bool))).sum()
    ) if not treatment_simulation.empty else 0
    treatment_candidate_recovered_count = int(
        bool_series(treatment_simulation.get("candidate_scope_recovered_to_nonnegative_flag", pd.Series(dtype=bool))).sum()
    ) if not treatment_simulation.empty else 0
    treatment_positive_boom = treatment_simulation[
        bool_series(treatment_simulation.get("actual_positive_specific_flag", pd.Series(dtype=bool)))
        & bool_series(treatment_simulation.get("boom_day_flag", pd.Series(dtype=bool)))
    ] if not treatment_simulation.empty else pd.DataFrame()
    treatment_policy_triggered_positive_boom = treatment_positive_boom[
        bool_series(treatment_positive_boom.get("treatment_v1_strict_trigger_flag", pd.Series(dtype=bool)))
    ] if not treatment_positive_boom.empty else pd.DataFrame()
    policy_positive_boom_mean_nll_improvement = (
        float(numeric_series(treatment_policy_triggered_positive_boom["nll_improvement_policy_v1"]).mean())
        if not treatment_policy_triggered_positive_boom.empty and "nll_improvement_policy_v1" in treatment_policy_triggered_positive_boom.columns
        else np.nan
    )
    candidate_positive_boom_mean_nll_improvement = (
        float(numeric_series(treatment_positive_boom["nll_improvement_candidate_scope"]).mean())
        if not treatment_positive_boom.empty and "nll_improvement_candidate_scope" in treatment_positive_boom.columns
        else np.nan
    )
    treatment_alpha = DEFAULT_MOMENTUM_OVERLAY_ALPHA
    if not treatment_simulation.empty and "alpha_injection" in treatment_simulation.columns:
        alpha_values = numeric_series(treatment_simulation["alpha_injection"]).dropna()
        if not alpha_values.empty:
            treatment_alpha = float(alpha_values.iloc[0])
    recommended_v2 = v2_trigger_sensitivity[
        bool_series(v2_trigger_sensitivity.get("recommended_threshold_flag", pd.Series(dtype=bool)))
    ] if not v2_trigger_sensitivity.empty else pd.DataFrame()
    recommended_trigger_name = (
        str(recommended_v2["trigger_name"].iloc[0]) if not recommended_v2.empty and "trigger_name" in recommended_v2.columns else "none"
    )
    first_ai_weight = alpha_weight_summary[
        numeric_series(alpha_weight_summary.get("ai_communication_weight_sum", pd.Series(dtype=float))) > ZERO_EPS
    ] if not alpha_weight_summary.empty else pd.DataFrame()
    first_ai_mu_patch = (
        float(first_ai_weight["mu_patch_level"].iloc[0]) if not first_ai_weight.empty and "mu_patch_level" in first_ai_weight.columns else np.nan
    )
    min_effective_row = alpha_weight_summary[
        np.isclose(numeric_series(alpha_weight_summary.get("mu_patch_level", pd.Series(dtype=float))), V2_T3_MIN_EFFECTIVE_PATCH)
    ] if not alpha_weight_summary.empty else pd.DataFrame()
    min_effective_ai_rows = (
        int(safe_float(min_effective_row["ai_communication_allocated_rows"].iloc[0], 0.0))
        if not min_effective_row.empty and "ai_communication_allocated_rows" in min_effective_row.columns
        else 0
    )
    min_effective_ai_weight = (
        float(min_effective_row["ai_communication_weight_sum"].iloc[0])
        if not min_effective_row.empty and "ai_communication_weight_sum" in min_effective_row.columns
        else np.nan
    )
    shadow_overall = shadow_pnl_summary[
        shadow_pnl_summary.get("scope", pd.Series(dtype=object)).eq("overall")
    ] if not shadow_pnl_summary.empty else pd.DataFrame()
    shadow_trigger_dates = shadow_pnl_summary[
        shadow_pnl_summary.get("scope", pd.Series(dtype=object)).eq("trigger_dates_only")
    ] if not shadow_pnl_summary.empty else pd.DataFrame()
    shadow_burst_window = shadow_pnl_summary[
        shadow_pnl_summary.get("scope", pd.Series(dtype=object)).eq("ai_burst_window_20260402_20260410")
    ] if not shadow_pnl_summary.empty else pd.DataFrame()
    shadow_trigger_delta_cum = (
        float(shadow_trigger_dates["delta_cumulative_return"].iloc[0])
        if not shadow_trigger_dates.empty and "delta_cumulative_return" in shadow_trigger_dates.columns
        else np.nan
    )
    shadow_trigger_ai_attr = (
        float(shadow_trigger_dates["sum_ai_communication_alpha_attribution"].iloc[0])
        if not shadow_trigger_dates.empty and "sum_ai_communication_alpha_attribution" in shadow_trigger_dates.columns
        else np.nan
    )
    shadow_burst_delta_cum = (
        float(shadow_burst_window["delta_cumulative_return"].iloc[0])
        if not shadow_burst_window.empty and "delta_cumulative_return" in shadow_burst_window.columns
        else np.nan
    )
    coverage_finding = (
        "Route A selected-pool coverage is now backfilled for the treatment target set; the earlier gap was an output completeness issue, not a selected-pool threshold failure."
        if missing_count == 0
        else "Route A output coverage remains incomplete for some selected-pool targets, so missing coverage must stay explicit in the audit trail."
    )
    stress_finding = (
        "The SH588170 formula replay decomposes `mu_raw = target_scale * Σ(scaled_feature * beta)` and identifies the largest negative linear terms directly."
        if not formula_stress.empty
        else "Formula stress replay did not find an eligible SH588170 sample."
    )
    output_lines = [
        f"- `{out_dir / 'route_a_mean_pathology_daily.csv'}`",
        f"- `{out_dir / 'route_a_mean_pathology_summary.csv'}`",
        f"- `{out_dir / 'route_a_mean_pathology_missing_route_a_files.csv'}`",
        f"- `{out_dir / 'student_t_formula_stress_test.csv'}`",
        f"- `{out_dir / 'required_momentum_boost_to_flip_sign.csv'}`",
        f"- `{out_dir / 'backfill_coverage_weight_impact.csv'}`",
        f"- `{out_dir / 'treatment_simulation_v1.csv'}`",
        f"- `{out_dir / 'v2_trigger_sensitivity.csv'}`",
        f"- `{out_dir / 'alpha_to_weight_elasticity.csv'}`",
        f"- `{out_dir / SHADOW_PNL_EQUITY_CURVE_NAME}`",
        f"- `{out_dir / SHADOW_PNL_SUMMARY_NAME}`",
    ]
    if final_freeze:
        output_lines = [
            f"- `{out_dir / FINAL_SUMMARY_NAME}`",
            f"- `{out_dir / SHADOW_PNL_EQUITY_CURVE_NAME}`",
            f"- `{out_dir / SHADOW_PNL_SUMMARY_NAME}`",
            f"- `{report_path}`",
        ]
    title = "# Route A Mean Pathology Audit FINAL（2026-05-01）" if final_freeze else "# Route A Mean Pathology Audit（2026-05-01）"
    command_line = "python -m pipeline.route_a_mean_pathology_audit"
    if final_freeze:
        command_line = f"python -m pipeline.route_a_mean_pathology_audit --report-md {report_path} --final-freeze"
    treatment_heading = "## Treatment_Simulation_v1（historical clinical stress, not final dose）" if final_freeze else "## Treatment_Simulation_v1"
    treatment_intro = (
        "Historical default-off clinical stress. The v1 NLL replay used a larger provisional candidate-scope neutralization stress; the final production candidate is the smaller `V2_T3_MIN_EFFECTIVE_PATCH = 0.005` optimizer-mean floor documented above."
        if final_freeze
        else "Default-off audit candidate. Trigger: `industry_momentum_std > 1.0`, `raw_mu < 0`, `volume_z_20 > 1.5`. Patch: `mu_patched = max(0, raw_mu + alpha_injection)`."
    )
    interpretation_text = (
        f"For final review, use `{out_dir / FINAL_SUMMARY_NAME}` for the frozen recommendation and `{out_dir / SHADOW_PNL_EQUITY_CURVE_NAME}` for the day-by-day Base vs Patched equity curve."
        if final_freeze
        else "Use `route_a_mean_pathology_daily.csv` to inspect the exact layer where an event row becomes negative or gets zeroed: raw Student-t mean, shrunk mean, EWMA/path/Kalman posterior, bridge weight mean, or cross-sectional active offset."
    )
    content = [
        title,
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Command",
        "",
        "```bash",
        command_line,
        "```",
        "",
        "## Scope",
        "",
        "This is an audit-only implementation step. It does not modify `strategy_layer.py`, Route A training, bridge optimization, or L3 risk controls.",
        "",
        "## Outputs",
        "",
        *output_lines,
        "",
        "## Coverage",
        "",
        f"- Target instruments: `{row_count}`",
        f"- Route A files found: `{observed_count}`",
        f"- Route A files missing: `{missing_count}`",
        f"- Route A boom coverage below 50%: `{low_route_a_boom_coverage_count}` instruments",
        f"- Observed positive-specific rows with raw Student-t negative mean: `{observed_raw_negative_count}` instruments",
        f"- Raw-negative rows tested for momentum boost: `{boost_candidate_count}`",
        f"- Rows with positive standardized industry momentum and finite required alpha: `{boost_feasible_count}`",
        f"- Backfilled AI/communication target audit rows: `{backfill_rows}`",
        f"- Backfilled rows found in bridge rerun: `{backfill_bridge_found_rows}`",
        f"- Backfilled rows receiving nonzero MVO weight: `{backfill_allocated_rows}`",
        f"- Backfilled rows with negative raw mean: `{backfill_negative_raw_rows}`",
        f"- Backfilled rows with negative raw mean and nonzero MVO weight: `{backfill_negative_allocated_rows}`",
        "",
        "## Key Findings",
        "",
        "1. The first observable failure on overlapping Route A rows is the raw Student-t mean itself turning negative on positive-specific boom rows.",
        f"2. {coverage_finding}",
        f"3. {stress_finding}",
        "4. Bridge/MVO still receives mostly zero or negative effective mean for these rows, consistent with the earlier full-MVO finding that downstream optimization cannot create missing alpha.",
        "5. Final freeze recommendation: introduce an Industry Momentum based mean-correction layer into the mainline as a defensive correction for Student-t reversal bias during explosive regimes, with feature flag default-off for the first production shadow window.",
        "",
        "## Final Freeze Decision",
        "",
        "Recommendation: adopt `V2_T3_MIN_EFFECTIVE_PATCH` as the mainline candidate for the event-driven mean correction layer.",
        "",
        f"- Patch name: `V2_T3_MIN_EFFECTIVE_PATCH`",
        f"- Locked patch level: `{V2_T3_MIN_EFFECTIVE_PATCH:.6g}`",
        "- Trigger: `route_a_mu_raw < 0` and `factor_industry_momentum > 0`; no volume gate.",
        "- Patch action: on triggered rows, floor the optimizer mean to the minimum effective level, not to an aggressive alpha level.",
        f"- MVO evidence at locked level: `{min_effective_ai_rows}` AI/communication event rows enter with total event-row weight `{min_effective_ai_weight:.6g}`.",
        "- Governance note: `0.015` already allocates all candidate rows, so it is intentionally rejected as too aggressive for first mainline admission.",
        f"- Shadow trigger-window cumulative PnL delta: `{shadow_trigger_delta_cum:.6g}`; AI/communication alpha attribution: `{shadow_trigger_ai_attr:.6g}`.",
        f"- AI burst-window cumulative PnL delta: `{shadow_burst_delta_cum:.6g}`.",
        "",
        markdown_table(
            final_summary,
            [
                "final_recommendation",
                "patch_name",
                "patch_level",
                "trigger_name",
                "min_effective_scan_level",
                "min_effective_ai_communication_allocated_rows",
                "min_effective_ai_communication_weight_sum",
                "v2_trigger_row_count",
                "v2_mean_nll_improvement_positive_boom",
                "shadow_trigger_dates_delta_cumulative_return",
                "shadow_trigger_dates_ai_communication_alpha_attribution",
                "mainline_posture",
            ],
            max_rows=4,
        ),
        "",
        "## Summary",
        "",
        markdown_table(
            summary,
            [
                "instrument",
                "name",
                "focus_theme",
                "period_return_rank",
                "period_return",
                "boom_day_count",
                "route_a_boom_coverage_rate",
                "boom_primary_pathology",
                "observed_boom_primary_pathology",
                "positive_specific_primary_pathology",
                "observed_positive_specific_primary_pathology",
                "residual_predicted_share_of_positive_specific",
                "residual_residual_share_clipped_0_1",
                "lead_lag_lead_lag_diagnosis",
            ],
        ),
        "",
        "## Backfilled Rerun Weight Impact",
        "",
        "The backfilled Route A files are mounted into the bridge/MVO path with no momentum overlay. This table is the direct test for whether coverage alone buys the AI/communication names.",
        "",
        f"- Bridge-covered target rows: `{backfill_bridge_found_rows}` / `{backfill_rows}`",
        f"- Negative raw-mu rows in the backfilled target set: `{backfill_negative_raw_rows}`",
        f"- Sum of MVO `forecast_w` on those negative raw-mu rows: `{backfill_negative_raw_weight_sum:.6g}`",
        "",
        markdown_table(
            backfill_impact,
            [
                "datetime",
                "instrument",
                "name",
                "boom_day_flag",
                "actual_positive_specific_flag",
                "route_a_mu_raw",
                "backfilled_bridge_forecast_weight_mean_used",
                "backfilled_bridge_forecast_w",
                "backfilled_bridge_route_a_status",
                "negative_raw_mu_allocated_weight_flag",
                "backfilled_bridge_weight_bucket",
            ],
            max_rows=24,
        ),
        "",
        "## Student-t Formula Stress Test",
        "",
        markdown_table(
            formula_stress.sort_values("contribution_to_mu_raw", ascending=True) if not formula_stress.empty else pd.DataFrame(),
            [
                "instrument",
                "forecast_date",
                "route_a_target_date",
                "feature",
                "x_scaled",
                "beta_scaled",
                "target_scale",
                "contribution_to_mu_raw",
                "mu_raw_recomputed",
                "route_a_mu_raw_file",
                "standardized_residual_z",
                "nll_improvement_if_mu_zero",
            ],
            max_rows=12,
        ),
        "",
        treatment_heading,
        "",
        treatment_intro,
        "",
        f"- Alpha injection: `{treatment_alpha:.6g}`",
        f"- Candidate rows with positive standardized industry momentum: `{treatment_candidate_count}`",
        f"- Strict-trigger rows after volume gate: `{treatment_trigger_count}`",
        f"- Strict-trigger rows recovered to nonnegative mu: `{treatment_policy_recovered_count}`",
        f"- Candidate-scope rows recovered if volume gate is removed: `{treatment_candidate_recovered_count}`",
        f"- Mean NLL improvement on strict-trigger positive boom rows: `{policy_positive_boom_mean_nll_improvement:.6g}`",
        f"- Mean NLL improvement on candidate-scope positive boom rows: `{candidate_positive_boom_mean_nll_improvement:.6g}`",
        "",
        markdown_table(
            treatment_simulation,
            [
                "datetime",
                "instrument",
                "name",
                "boom_day_flag",
                "actual_positive_specific_flag",
                "route_a_mu_raw",
                "factor_industry_momentum",
                "factor_volume_z_20",
                "required_alpha_gain_to_flip_raw_mu",
                "treatment_v1_strict_trigger_flag",
                "mu_patched_policy_v1",
                "nll_improvement_policy_v1",
                "nll_improvement_candidate_scope",
            ],
            max_rows=18,
        ),
        "",
        "## 事件驱动型均值修正（Event-Driven Mean Correction）",
        "",
        "The proposed treatment is an event-row correction, not a long-term always-on mean factor: it only touches rows where Route A raw Student-t mean is negative while same-theme industry momentum is already positive. The final V2_T3 form intentionally removes the volume gate because the sensitivity scan showed that the strict and moderate volume gates were too sparse for mainline candidacy.",
        "",
        f"- Recommended v2 trigger by the audit rule: `{recommended_trigger_name}`",
        f"- Locked patch parameter: `V2_T3_MIN_EFFECTIVE_PATCH = {V2_T3_MIN_EFFECTIVE_PATCH:.6g}`",
        "- Selection rule: prefer triggers with 10-20 candidate rows and mean positive-boom NLL improvement above 5; among passing rows, choose the count closest to 15, then higher NLL improvement, then stricter trigger.",
        f"- First scan level where AI/communication candidates receive nonzero hard-band MVO weight: `{first_ai_mu_patch:.6g}`",
        "- Alpha-to-weight scan sets candidate optimizer mean directly to each `mu_patch_level` and reruns the same hard-band MVO optimizer on the affected dates only; it does not invoke L3 controls or modify production code.",
        "",
        "### v2 Trigger Sensitivity",
        "",
        markdown_table(
            v2_trigger_sensitivity,
            [
                "trigger_name",
                "trigger_row_count",
                "trigger_positive_specific_count",
                "trigger_positive_boom_count",
                "mean_nll_improvement_all_triggered",
                "mean_nll_improvement_positive_boom",
                "coverage_target_10_20_flag",
                "nll_positive_boom_gt_5_flag",
                "mainline_candidate_flag",
                "recommended_threshold_flag",
            ],
            max_rows=8,
        ),
        "",
        "### Alpha-to-Weight Elasticity",
        "",
        markdown_table(
            alpha_weight_summary,
            [
                "mu_patch_level",
                "candidate_allocated_rows",
                "ai_communication_allocated_rows",
                "candidate_weight_sum",
                "candidate_marginal_weight_sum",
                "ai_communication_weight_sum",
                "ai_communication_marginal_weight_sum",
                "max_single_candidate_weight",
            ],
            max_rows=12,
        ),
        "",
        "## Shadow PnL Backtest",
        "",
        "This time-axis audit reruns the full hard-band MVO universe for Base and Patched. Base uses the original negative/zero optimizer mean state. Patched applies `V2_T3_MIN_EFFECTIVE_PATCH` only to T3 event rows. Daily returns use the bridge `next_return`, so a forecast date such as 2026-04-02 captures the realized close-to-next-close move into 2026-04-03.",
        "",
        markdown_table(
            shadow_pnl_summary,
            [
                "scope",
                "day_count",
                "trigger_day_count",
                "base_cumulative_return",
                "patched_cumulative_return",
                "delta_cumulative_return",
                "sum_active_return",
                "sum_t3_trigger_alpha_attribution",
                "sum_ai_communication_alpha_attribution",
                "sum_non_trigger_reallocation_attribution",
                "max_delta_ai_communication_weight",
            ],
            max_rows=8,
        ),
        "",
        markdown_table(
            shadow_pnl.loc[
                bool_series(shadow_pnl.get("trigger_day_flag", pd.Series(False, index=shadow_pnl.index)), index=shadow_pnl.index)
                | bool_series(shadow_pnl.get("ai_burst_window_flag", pd.Series(False, index=shadow_pnl.index)), index=shadow_pnl.index)
            ] if not shadow_pnl.empty else pd.DataFrame(),
            [
                "datetime",
                "t3_trigger_instruments",
                "t3_trigger_target_dates",
                "base_daily_return",
                "patched_daily_return",
                "active_return",
                "base_nav",
                "patched_nav",
                "t3_trigger_alpha_attribution",
                "ai_communication_alpha_attribution",
                "non_trigger_reallocation_attribution",
                "delta_ai_communication_weight",
            ],
            max_rows=16,
        ),
        "",
        "## Momentum Injection Prototype",
        "",
        markdown_table(
            momentum_boost,
            [
                "datetime",
                "instrument",
                "name",
                "boom_day_flag",
                "actual_positive_specific_flag",
                "route_a_mu_raw",
                "factor_industry_momentum",
                "required_alpha_gain_to_flip_raw_mu",
                "required_alpha_gain_raw_momentum_to_flip_raw_mu",
                "primary_pathology",
            ],
            max_rows=12,
        ),
        "",
        "## Missing Route A Coverage",
        "",
        markdown_table(
            missing[~bool_series(missing.get("route_a_file_found", pd.Series(dtype=bool)))] if not missing.empty else pd.DataFrame(),
            [
                "instrument",
                "name",
                "focus_theme",
                "period_return_rank",
                "period_return",
                "boom_day_count",
                "route_a_file_source",
            ],
        ),
        "",
        "## Interpretation",
        "",
        interpretation_text,
        "",
    ]
    report_path.write_text("\n".join(content), encoding="utf-8")
    print(f"[INFO] wrote {report_path}")


def run_audit(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    audit_dir = Path(args.audit_dir).expanduser().resolve()
    route_a_dir = Path(args.route_a_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    report_md = Path(args.report_md).expanduser().resolve()
    residual_daily = read_csv_if_exists(audit_dir / "residual_analysis_daily.csv", parse_dates=("datetime",))
    residual_summary = read_csv_if_exists(audit_dir / "residual_analysis_summary.csv")
    factor_check = read_csv_if_exists(audit_dir / "residual_factor_exposure_check.csv")
    lead_lag = read_csv_if_exists(audit_dir / "lead_lag_cross_correlation.csv")
    full_mvo = read_csv_if_exists(audit_dir / "weight_delta_analysis_patched_vs_base.csv", parse_dates=("datetime",))
    instruments = select_target_instruments(
        residual_summary,
        args.target_instrument,
        args.min_residual_share,
        args.min_active_offset_share,
    )
    if not instruments:
        raise SystemExit("No target instruments selected. Check residual_analysis_summary.csv or pass --target-instrument.")
    route_a_layers, coverage = load_route_a_layers(route_a_dir, instruments)
    daily = merge_observable_layers(residual_daily, route_a_layers, coverage, full_mvo)
    daily = daily[daily["instrument"].isin(instruments)].copy() if not daily.empty else daily
    daily = add_pathology_columns(daily) if not daily.empty else daily
    summary = summarize_pathology(daily, residual_summary, factor_check, lead_lag)
    missing = build_missing_coverage_table(coverage, summary)
    formula_stress = build_student_t_formula_stress_test(
        daily,
        residual_daily,
        route_a_dir=route_a_dir,
        provider_uri=str(args.provider_uri),
        route_a_start_date=str(args.route_a_start_date),
        route_a_end_date=str(args.route_a_end_date),
        train_window=int(args.route_a_train_window),
        retrain_every=int(args.route_a_retrain_every),
        training_objective=str(args.route_a_training_objective),
        instrument=str(args.formula_stress_instrument),
    )
    momentum_boost = build_required_momentum_boost_to_flip_sign(daily)
    backfill_impact = build_backfill_coverage_weight_impact(daily, Path(args.post_backfill_bridge).expanduser().resolve())
    treatment_simulation = build_treatment_simulation_v1(
        daily,
        alpha_injection=float(args.momentum_overlay_alpha),
        route_a_dir=route_a_dir,
        provider_uri=str(args.provider_uri),
        route_a_start_date=str(args.route_a_start_date),
        route_a_end_date=str(args.route_a_end_date),
        train_window=int(args.route_a_train_window),
        retrain_every=int(args.route_a_retrain_every),
        training_objective=str(args.route_a_training_objective),
    )
    v2_trigger_sensitivity = build_v2_trigger_sensitivity(treatment_simulation)
    alpha_weight_elasticity = build_alpha_to_weight_elasticity(
        treatment_simulation,
        Path(args.post_backfill_bridge).expanduser().resolve(),
    )
    shadow_pnl = build_shadow_pnl_backtest(
        treatment_simulation,
        Path(args.post_backfill_bridge).expanduser().resolve(),
        patch_level=V2_T3_MIN_EFFECTIVE_PATCH,
    )
    shadow_pnl_summary = summarize_shadow_pnl_backtest(shadow_pnl)
    final_summary = build_final_freeze_summary(v2_trigger_sensitivity, alpha_weight_elasticity, shadow_pnl_summary)
    if args.final_freeze:
        write_table(final_summary, out_dir, FINAL_SUMMARY_NAME)
        write_table(shadow_pnl, out_dir, SHADOW_PNL_EQUITY_CURVE_NAME)
        write_table(shadow_pnl_summary, out_dir, SHADOW_PNL_SUMMARY_NAME)
    else:
        write_table(daily, out_dir, "route_a_mean_pathology_daily.csv")
        write_table(summary, out_dir, "route_a_mean_pathology_summary.csv")
        write_table(missing, out_dir, "route_a_mean_pathology_missing_route_a_files.csv")
        write_table(formula_stress, out_dir, "student_t_formula_stress_test.csv")
        write_table(momentum_boost, out_dir, "required_momentum_boost_to_flip_sign.csv")
        write_table(backfill_impact, out_dir, "backfill_coverage_weight_impact.csv")
        write_table(treatment_simulation, out_dir, "treatment_simulation_v1.csv")
        write_table(v2_trigger_sensitivity, out_dir, "v2_trigger_sensitivity.csv")
        write_table(alpha_weight_elasticity, out_dir, "alpha_to_weight_elasticity.csv")
        write_table(shadow_pnl, out_dir, SHADOW_PNL_EQUITY_CURVE_NAME)
        write_table(shadow_pnl_summary, out_dir, SHADOW_PNL_SUMMARY_NAME)
    write_report(
        report_md,
        summary,
        missing,
        out_dir,
        formula_stress,
        momentum_boost,
        backfill_impact,
        treatment_simulation,
        v2_trigger_sensitivity,
        alpha_weight_elasticity,
        shadow_pnl,
        shadow_pnl_summary,
        final_summary,
        bool(args.final_freeze),
    )
    if args.final_freeze:
        cleanup_route_a_intermediate_outputs(out_dir)
    return {
        "daily": daily,
        "summary": summary,
        "missing": missing,
        "formula_stress": formula_stress,
        "momentum_boost": momentum_boost,
        "backfill_impact": backfill_impact,
        "treatment_simulation": treatment_simulation,
        "v2_trigger_sensitivity": v2_trigger_sensitivity,
        "alpha_weight_elasticity": alpha_weight_elasticity,
        "shadow_pnl": shadow_pnl,
        "shadow_pnl_summary": shadow_pnl_summary,
        "final_summary": final_summary,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_audit(args)


if __name__ == "__main__":
    main()
