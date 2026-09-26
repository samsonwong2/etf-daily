from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategy.route_ab_moments_bridge import bridge as bridge_core

DEFAULT_BRIDGE_L3_CSV = (
    "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/warmup_route_a_20251001_sq_ic3_shadow/"
    "bridge_posterior_mu1d_softshrink_fullpanel_20260101_20260430_l3.csv"
)
DEFAULT_PARAMETER_AUDIT_CSV = (
    "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/warmup_route_a_20251001_sq_ic3_shadow/"
    "parameter_effect_audit_posterior_mu1d_softshrink_fullpanel_20260101_20260430/"
    "per_instrument_daily_parameter_effects.csv"
)
DEFAULT_OUTPUT_DIR = "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/smart_mu_patch_audit_20260101_20260430"
DEFAULT_START_DATE = "2026-01-01"
DEFAULT_END_DATE = "2026-04-30"
SMART_MU_PATCH_MULTIPLIER = 0.3
OVERHEAT_COLUMNS = (
    "route_a_params_feature_momentum_overheat_5d_gt_10pct",
    "route_a_eval_feature_momentum_overheat_5d_gt_10pct",
)


@dataclass(frozen=True)
class SmartMuPatchConfig:
    start_date: str
    end_date: str
    max_weight: float
    min_weight_floor: float
    patch_multiplier: float
    communication_theme: str


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _compound_return(values: pd.Series) -> float:
    series = _numeric(values).fillna(0.0)
    if series.empty:
        return float("nan")
    return float((1.0 + series).prod() - 1.0)


def _max_drawdown(values: pd.Series) -> float:
    series = _numeric(values).fillna(0.0)
    if series.empty:
        return float("nan")
    nav = (1.0 + series).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    return float(drawdown.min())


def _sharpe(values: pd.Series) -> float:
    series = _numeric(values).dropna()
    if len(series) < 2:
        return float("nan")
    vol = float(series.std(ddof=1))
    if vol <= 1e-12:
        return float("nan")
    return float(series.mean() / vol * np.sqrt(252.0))


def _markdown_table(frame: pd.DataFrame, max_rows: int = 12) -> str:
    if frame.empty:
        return "No rows."
    shown = frame.head(max_rows).copy()
    columns = [str(column) for column in shown.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in shown.iterrows():
        values = []
        for column in shown.columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.8g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    if len(frame) > len(shown):
        lines.append(f"| ... {len(frame) - len(shown)} more rows | " + " | ".join([""] * (len(columns) - 1)) + " |")
    return "\n".join(lines)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_bridge_frame(path: str | Path, start_date: str, end_date: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["datetime"])
    required = {
        "instrument",
        "datetime",
        "close",
        "next_return",
        "forecast_w",
        "forecast_cov_order",
        "forecast_cov_matrix_robust",
        "forecast_weight_mean_used",
        "exposure_scale",
        "factor_industry_theme",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"bridge L3 CSV missing required columns: {missing}")
    frame["instrument"] = frame["instrument"].astype(str).str.upper()
    frame = frame[(frame["datetime"] >= pd.Timestamp(start_date)) & (frame["datetime"] <= pd.Timestamp(end_date))].copy()
    frame = frame.sort_values(["datetime", "instrument"]).reset_index(drop=True)
    for column in ["close", "next_return", "forecast_w", "forecast_weight_mean_used", "exposure_scale"]:
        frame[column] = _numeric(frame[column])
    return frame


def _load_overheat_flags(parameter_audit_csv: str | Path, start_date: str, end_date: str) -> tuple[pd.DataFrame, str]:
    path = Path(parameter_audit_csv)
    if not path.exists():
        return pd.DataFrame(columns=["datetime", "instrument", "smart_mu_patch_trigger_route_a_overheat_5d"]), "missing_parameter_audit"

    header = pd.read_csv(path, nrows=0)
    selected_column = next((column for column in OVERHEAT_COLUMNS if column in header.columns), None)
    if selected_column is None:
        return pd.DataFrame(columns=["datetime", "instrument", "smart_mu_patch_trigger_route_a_overheat_5d"]), "missing_overheat_column"

    usecols = ["datetime", "instrument", selected_column]
    frame = pd.read_csv(path, usecols=usecols, parse_dates=["datetime"])
    frame["instrument"] = frame["instrument"].astype(str).str.upper()
    frame = frame[(frame["datetime"] >= pd.Timestamp(start_date)) & (frame["datetime"] <= pd.Timestamp(end_date))].copy()
    frame["smart_mu_patch_trigger_route_a_overheat_5d"] = _numeric(frame[selected_column]).fillna(0.0) > 0.0
    frame = frame[["datetime", "instrument", "smart_mu_patch_trigger_route_a_overheat_5d"]]
    return frame.drop_duplicates(["datetime", "instrument"]), f"parameter_audit:{selected_column}"


def _raw_overheat_from_close(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame[["instrument", "datetime", "close"]].copy()
    working = working.sort_values(["instrument", "datetime"])
    working["smart_mu_patch_raw_close_return_5d"] = working.groupby("instrument")["close"].pct_change(5, fill_method=None)
    working["smart_mu_patch_trigger_raw_overheat_5d"] = working["smart_mu_patch_raw_close_return_5d"] > 0.10
    return working[["smart_mu_patch_raw_close_return_5d", "smart_mu_patch_trigger_raw_overheat_5d"]].reindex(frame.index)


def _attach_patch_flags(
    bridge_frame: pd.DataFrame,
    parameter_audit_csv: str | Path,
    config: SmartMuPatchConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    working = bridge_frame.copy()
    raw_overheat = _raw_overheat_from_close(working)
    working["smart_mu_patch_raw_close_return_5d"] = _numeric(raw_overheat["smart_mu_patch_raw_close_return_5d"])
    working["smart_mu_patch_trigger_raw_overheat_5d"] = raw_overheat["smart_mu_patch_trigger_raw_overheat_5d"].fillna(False).astype(bool)
    overheat_flags, overheat_source = _load_overheat_flags(parameter_audit_csv, config.start_date, config.end_date)
    if overheat_flags.empty:
        working["smart_mu_patch_trigger_route_a_overheat_5d"] = False
        if overheat_source == "missing_parameter_audit":
            overheat_source = "raw_l3_close_pct_change_5d_parameter_audit_missing"
        else:
            overheat_source = "raw_l3_close_pct_change_5d_overheat_column_missing"
    else:
        working = working.merge(overheat_flags, on=["datetime", "instrument"], how="left")
        working["smart_mu_patch_trigger_route_a_overheat_5d"] = working["smart_mu_patch_trigger_route_a_overheat_5d"].fillna(False).astype(bool)
        overheat_source = f"raw_l3_close_pct_change_5d_or_{overheat_source}"

    working["smart_mu_patch_trigger_overheat_5d"] = (
        working["smart_mu_patch_trigger_raw_overheat_5d"] | working["smart_mu_patch_trigger_route_a_overheat_5d"]
    )

    theme = working["factor_industry_theme"].astype(str).str.lower().str.strip()
    target_theme = str(config.communication_theme).lower().strip()
    working["smart_mu_patch_trigger_warmup_communication"] = theme.eq(target_theme)
    working["smart_mu_patch_trigger_any"] = (
        working["smart_mu_patch_trigger_overheat_5d"] | working["smart_mu_patch_trigger_warmup_communication"]
    )
    working["smart_mu_before"] = _numeric(working["forecast_weight_mean_used"]).fillna(0.0)
    working["smart_mu_after"] = working["smart_mu_before"]
    trigger = working["smart_mu_patch_trigger_any"].astype(bool)
    working.loc[trigger, "smart_mu_after"] = working.loc[trigger, "smart_mu_before"] * float(config.patch_multiplier)
    working["smart_mu_delta"] = working["smart_mu_after"] - working["smart_mu_before"]
    metadata = {
        "overheat_source": overheat_source,
        "raw_overheat_trigger_rows": int(working["smart_mu_patch_trigger_raw_overheat_5d"].sum()),
        "route_a_overheat_trigger_rows": int(working["smart_mu_patch_trigger_route_a_overheat_5d"].sum()),
        "overheat_trigger_rows": int(working["smart_mu_patch_trigger_overheat_5d"].sum()),
        "warmup_communication_trigger_rows": int(working["smart_mu_patch_trigger_warmup_communication"].sum()),
        "any_trigger_rows": int(working["smart_mu_patch_trigger_any"].sum()),
    }
    return working, metadata


def _parse_cov_matrix(day_frame: pd.DataFrame) -> pd.DataFrame:
    order_values = day_frame["forecast_cov_order"].dropna()
    if order_values.empty:
        return pd.DataFrame()
    order = str(order_values.iloc[0]).split("|")
    vector_by_instrument: dict[str, list[float]] = {}
    for _, row in day_frame.iterrows():
        text = str(row.get("forecast_cov_matrix_robust", ""))
        values = [float(value) for value in text.split(",") if value != ""]
        if len(values) == len(order):
            vector_by_instrument[str(row["instrument"])] = values
    rows: list[list[float]] = []
    valid_order: list[str] = []
    for instrument in order:
        if instrument in vector_by_instrument:
            valid_order.append(instrument)
            rows.append(vector_by_instrument[instrument])
    cov = pd.DataFrame(rows, index=valid_order, columns=order).reindex(columns=valid_order)
    cov = cov.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if not cov.empty:
        cov = 0.5 * (cov + cov.T)
        np.fill_diagonal(cov.values, np.maximum(np.diag(cov.values), 1e-12))
    return cov


def _apply_inertia_and_floor(
    weights: pd.Series,
    previous_weights: pd.Series,
    sample_index: pd.Index,
    alpha: float,
    min_weight_floor: float,
    max_weight: float,
) -> tuple[pd.Series, str]:
    out = weights.reindex(sample_index).fillna(0.0)
    status = ""
    if float(alpha) > 0.0 and len(out) > 0:
        prior = previous_weights.reindex(sample_index).fillna(0.0)
        if float(prior.sum()) > 1e-12:
            delta = out - prior
            factor = pd.Series(np.where(delta.to_numpy() < 0.0, 1.0, 1.0 - float(alpha)), index=sample_index)
            out = (prior + factor * delta).clip(lower=0.0)
            total = float(out.sum())
            if total > 1e-12:
                out = out / total
            status = f"inertia_asym_add={float(alpha):g}"
        else:
            status = "inertia_skip_empty_prior"
        out, floor_status = bridge_core._enforce_floor_and_cap(
            out,
            min_weight_floor=min_weight_floor,
            max_weight=max_weight,
        )
        status = f"{status}|post_blend_{floor_status}" if status else f"post_blend_{floor_status}"
    return out.reindex(sample_index).fillna(0.0), status


def _optimize_day(
    day_frame: pd.DataFrame,
    mean_column: str,
    previous_weights: pd.Series,
    config: SmartMuPatchConfig,
) -> tuple[pd.Series, str]:
    sample = day_frame.set_index("instrument", drop=False).sort_index()
    sample_index = pd.Index(sample.index.astype(str), dtype=object)
    cov = _parse_cov_matrix(day_frame).reindex(index=sample_index, columns=sample_index).fillna(0.0)
    mean = _numeric(sample[mean_column]).fillna(0.0).reindex(sample_index)
    protected = _numeric(sample.get("forecast_active_set_protect_score", pd.Series(index=sample_index))).fillna(0.0).reindex(sample_index)
    crowded = _numeric(sample.get("forecast_active_set_crowded_score", pd.Series(index=sample_index))).fillna(0.0).reindex(sample_index)
    balanced = _numeric(sample.get("forecast_active_set_balance_score", pd.Series(index=sample_index))).fillna(0.0).reindex(sample_index)
    weights, status = bridge_core.optimize_weights_hard_band(
        mean,
        cov,
        rf=0.0,
        max_weight=config.max_weight,
        min_weight_floor=config.min_weight_floor,
        protected_scores=protected,
        protected_slots=bridge_core.DEFAULT_HARD_BAND_PROTECTED_SLOTS,
        crowded_scores=crowded,
        balanced_scores=balanced,
        min_crowded_for_balance=bridge_core.DEFAULT_BALANCED_MIN_CROWDED,
    )
    alpha_values = _numeric(day_frame.get("forecast_weight_inertia_alpha", pd.Series(dtype=float))).dropna()
    alpha = float(alpha_values.iloc[0]) if not alpha_values.empty else 0.0
    weights, inertia_status = _apply_inertia_and_floor(
        weights,
        previous_weights,
        sample_index,
        alpha,
        config.min_weight_floor,
        config.max_weight,
    )
    if inertia_status:
        status = f"{status}|{inertia_status}"
    return weights, status


def _trigger_case(row: pd.Series) -> str:
    overheat = bool(row.get("smart_mu_patch_trigger_overheat_5d", False))
    communication = bool(row.get("smart_mu_patch_trigger_warmup_communication", False))
    if overheat and communication:
        return "both_overheat_and_warmup_communication"
    if overheat:
        return "overheat_only"
    if communication:
        return "warmup_communication_only"
    return "not_triggered"


def _summarize_daily(daily: pd.DataFrame) -> dict[str, Any]:
    replay_diff = _numeric(daily["old_mvo_effective_return"]) - _numeric(daily["published_l3_effective_return"])
    replay_corr = _numeric(daily["old_mvo_effective_return"]).corr(_numeric(daily["published_l3_effective_return"])) if len(daily) >= 2 else np.nan
    worst_idx = _numeric(daily["smart_mu_patch_effective_return"]).idxmin() if not daily.empty else None
    best_delta_idx = _numeric(daily["delta_return_vs_old_mvo"]).idxmax() if not daily.empty else None
    worst_delta_idx = _numeric(daily["delta_return_vs_old_mvo"]).idxmin() if not daily.empty else None
    return {
        "days": int(daily.shape[0]),
        "old_mvo_cumulative_return": _compound_return(daily["old_mvo_effective_return"]),
        "smart_mu_patch_cumulative_return": _compound_return(daily["smart_mu_patch_effective_return"]),
        "delta_cumulative_return_vs_old_mvo": _compound_return(daily["smart_mu_patch_effective_return"]) - _compound_return(daily["old_mvo_effective_return"]),
        "published_l3_cumulative_return": _compound_return(daily["published_l3_effective_return"]),
        "old_mvo_daily_return_sum": float(_numeric(daily["old_mvo_effective_return"]).sum()),
        "smart_mu_patch_daily_return_sum": float(_numeric(daily["smart_mu_patch_effective_return"]).sum()),
        "delta_daily_return_sum_vs_old_mvo": float(_numeric(daily["delta_return_vs_old_mvo"]).sum()),
        "old_mvo_max_drawdown": _max_drawdown(daily["old_mvo_effective_return"]),
        "smart_mu_patch_max_drawdown": _max_drawdown(daily["smart_mu_patch_effective_return"]),
        "old_mvo_sharpe": _sharpe(daily["old_mvo_effective_return"]),
        "smart_mu_patch_sharpe": _sharpe(daily["smart_mu_patch_effective_return"]),
        "trigger_active_days": int((_numeric(daily["triggered_rows"]).fillna(0.0) > 0.0).sum()),
        "avg_gross_abs_weight_delta": float(_numeric(daily["gross_abs_weight_delta"]).mean()),
        "max_gross_abs_weight_delta": float(_numeric(daily["gross_abs_weight_delta"]).max()),
        "old_replay_vs_published_cumulative_delta": _compound_return(daily["old_mvo_effective_return"]) - _compound_return(daily["published_l3_effective_return"]),
        "old_replay_vs_published_daily_corr": float(replay_corr) if pd.notna(replay_corr) else np.nan,
        "old_replay_vs_published_max_abs_daily_diff": float(replay_diff.abs().max()),
        "worst_smart_mu_patch_day": str(daily.loc[worst_idx, "datetime"]) if worst_idx is not None else "",
        "worst_smart_mu_patch_return": float(daily.loc[worst_idx, "smart_mu_patch_effective_return"]) if worst_idx is not None else np.nan,
        "best_delta_day": str(daily.loc[best_delta_idx, "datetime"]) if best_delta_idx is not None else "",
        "best_delta_return_vs_old_mvo": float(daily.loc[best_delta_idx, "delta_return_vs_old_mvo"]) if best_delta_idx is not None else np.nan,
        "worst_delta_day": str(daily.loc[worst_delta_idx, "datetime"]) if worst_delta_idx is not None else "",
        "worst_delta_return_vs_old_mvo": float(daily.loc[worst_delta_idx, "delta_return_vs_old_mvo"]) if worst_delta_idx is not None else np.nan,
    }


def run_smart_mu_patch_audit(
    bridge_l3_csv: str | Path,
    parameter_audit_csv: str | Path,
    output_dir: str | Path,
    config: SmartMuPatchConfig,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    bridge_frame = _load_bridge_frame(bridge_l3_csv, config.start_date, config.end_date)
    working, trigger_metadata = _attach_patch_flags(bridge_frame, parameter_audit_csv, config)

    previous_old = pd.Series(dtype=float)
    previous_patch = pd.Series(dtype=float)
    daily_rows: list[dict[str, Any]] = []
    weight_frames: list[pd.DataFrame] = []

    for date_value, day_frame in working.groupby("datetime", sort=True):
        date_text = pd.Timestamp(date_value).strftime("%Y-%m-%d")
        old_weights, old_status = _optimize_day(day_frame, "smart_mu_before", previous_old, config)
        patch_weights, patch_status = _optimize_day(day_frame, "smart_mu_after", previous_patch, config)
        sample = day_frame.set_index("instrument", drop=False).reindex(old_weights.index)
        next_return = _numeric(sample["next_return"]).fillna(0.0)
        exposure_scale = _numeric(sample["exposure_scale"]).fillna(1.0)
        published_w = _numeric(sample["forecast_w"]).fillna(0.0)
        old_effective = old_weights * exposure_scale
        patch_effective = patch_weights * exposure_scale
        published_effective = published_w * exposure_scale
        old_return = float((old_effective * next_return).sum())
        patch_return = float((patch_effective * next_return).sum())
        published_return = float((published_effective * next_return).sum())
        delta_w = patch_weights - old_weights
        contribution_delta = delta_w * exposure_scale * next_return
        trigger_any = sample["smart_mu_patch_trigger_any"].fillna(False).astype(bool)
        overheat = sample["smart_mu_patch_trigger_overheat_5d"].fillna(False).astype(bool)
        raw_overheat = sample["smart_mu_patch_trigger_raw_overheat_5d"].fillna(False).astype(bool)
        route_a_overheat = sample["smart_mu_patch_trigger_route_a_overheat_5d"].fillna(False).astype(bool)
        communication = sample["smart_mu_patch_trigger_warmup_communication"].fillna(False).astype(bool)
        day_weights = pd.DataFrame(
            {
                "datetime": date_text,
                "instrument": old_weights.index,
                "factor_industry_theme": sample["factor_industry_theme"].astype(str).to_numpy(),
                "trigger_case": sample.apply(_trigger_case, axis=1).to_numpy(),
                "smart_mu_patch_trigger_overheat_5d": overheat.to_numpy(),
                "smart_mu_patch_trigger_raw_overheat_5d": raw_overheat.to_numpy(),
                "smart_mu_patch_trigger_route_a_overheat_5d": route_a_overheat.to_numpy(),
                "smart_mu_patch_trigger_warmup_communication": communication.to_numpy(),
                "smart_mu_patch_trigger_any": trigger_any.to_numpy(),
                "old_mvo_weight": old_weights.to_numpy(),
                "smart_mu_patch_mvo_weight": patch_weights.reindex(old_weights.index).fillna(0.0).to_numpy(),
                "delta_mvo_weight": delta_w.to_numpy(),
                "old_effective_weight": old_effective.to_numpy(),
                "smart_mu_patch_effective_weight": patch_effective.reindex(old_weights.index).fillna(0.0).to_numpy(),
                "published_l3_weight": published_w.to_numpy(),
                "published_l3_effective_weight": published_effective.to_numpy(),
                "smart_mu_before": sample["smart_mu_before"].to_numpy(),
                "smart_mu_after": sample["smart_mu_after"].to_numpy(),
                "smart_mu_delta": sample["smart_mu_delta"].to_numpy(),
                "smart_mu_patch_raw_close_return_5d": sample["smart_mu_patch_raw_close_return_5d"].to_numpy(),
                "next_return": next_return.to_numpy(),
                "exposure_scale": exposure_scale.to_numpy(),
                "delta_return_contribution": contribution_delta.to_numpy(),
            }
        )
        weight_frames.append(day_weights)
        daily_rows.append(
            {
                "datetime": date_text,
                "triggered_rows": int(trigger_any.sum()),
                "overheat_trigger_rows": int(overheat.sum()),
                "raw_overheat_trigger_rows": int(raw_overheat.sum()),
                "route_a_overheat_trigger_rows": int(route_a_overheat.sum()),
                "warmup_communication_trigger_rows": int(communication.sum()),
                "active_old_count": int((old_weights > 1e-12).sum()),
                "active_patch_count": int((patch_weights > 1e-12).sum()),
                "changed_weight_count": int((delta_w.abs() > 1e-8).sum()),
                "gross_abs_weight_delta": float(delta_w.abs().sum()),
                "max_abs_weight_delta": float(delta_w.abs().max()) if len(delta_w) else 0.0,
                "smart_mu_delta_abs_sum": float(_numeric(sample["smart_mu_delta"]).abs().sum()),
                "old_mvo_effective_return": old_return,
                "smart_mu_patch_effective_return": patch_return,
                "delta_return_vs_old_mvo": patch_return - old_return,
                "published_l3_effective_return": published_return,
                "delta_return_vs_published_l3": patch_return - published_return,
                "old_status": old_status,
                "patch_status": patch_status,
                "top_weight_delta_codes": "|".join(delta_w.abs().sort_values(ascending=False).head(8).index.astype(str).tolist()),
                "top_delta_contribution_codes": "|".join(contribution_delta.abs().sort_values(ascending=False).head(8).index.astype(str).tolist()),
            }
        )
        previous_old = old_weights.copy()
        previous_patch = patch_weights.copy()

    daily = pd.DataFrame(daily_rows)
    weights = pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    if not daily.empty:
        daily["old_mvo_nav_proxy"] = (1.0 + _numeric(daily["old_mvo_effective_return"]).fillna(0.0)).cumprod()
        daily["smart_mu_patch_nav_proxy"] = (1.0 + _numeric(daily["smart_mu_patch_effective_return"]).fillna(0.0)).cumprod()
        daily["published_l3_nav_proxy"] = (1.0 + _numeric(daily["published_l3_effective_return"]).fillna(0.0)).cumprod()
        daily["delta_nav_vs_old_mvo"] = daily["smart_mu_patch_nav_proxy"] - daily["old_mvo_nav_proxy"]

    summary_payload = _summarize_daily(daily)
    summary_payload.update(
        {
            "audit_scope": "smart_mu_patch_audit_only_no_live_weight_change",
            "bridge_l3_csv": str(bridge_l3_csv),
            "parameter_audit_csv": str(parameter_audit_csv),
            "start_date": config.start_date,
            "end_date": config.end_date,
            "patch_multiplier": float(config.patch_multiplier),
            "communication_theme": config.communication_theme,
            **trigger_metadata,
        }
    )
    summary = pd.DataFrame([summary_payload])

    trigger_breakdown = pd.DataFrame()
    if not weights.empty:
        trigger_breakdown = (
            weights.groupby(["trigger_case", "factor_industry_theme"], dropna=False)
            .agg(
                rows=("instrument", "size"),
                instruments=("instrument", "nunique"),
                active_old_rows=("old_mvo_weight", lambda values: int((_numeric(values) > 1e-12).sum())),
                active_patch_rows=("smart_mu_patch_mvo_weight", lambda values: int((_numeric(values) > 1e-12).sum())),
                smart_mu_before_mean=("smart_mu_before", "mean"),
                smart_mu_after_mean=("smart_mu_after", "mean"),
                smart_mu_delta_abs_sum=("smart_mu_delta", lambda values: float(_numeric(values).abs().sum())),
                delta_mvo_weight_abs_sum=("delta_mvo_weight", lambda values: float(_numeric(values).abs().sum())),
                delta_return_contribution_sum=("delta_return_contribution", "sum"),
            )
            .reset_index()
            .sort_values(["delta_return_contribution_sum", "rows"], ascending=[False, False])
        )

    top_positive_days = daily.sort_values("delta_return_vs_old_mvo", ascending=False).head(10) if not daily.empty else pd.DataFrame()
    top_negative_days = daily.sort_values("delta_return_vs_old_mvo", ascending=True).head(10) if not daily.empty else pd.DataFrame()
    top_weight_delta = weights.reindex(weights["delta_mvo_weight"].abs().sort_values(ascending=False).index).head(100) if not weights.empty else pd.DataFrame()
    top_contribution_delta = weights.reindex(weights["delta_return_contribution"].abs().sort_values(ascending=False).index).head(100) if not weights.empty else pd.DataFrame()

    daily_path = output_path / "smart_mu_patch_daily.csv"
    weights_path = output_path / "smart_mu_patch_weights.csv"
    summary_path = output_path / "smart_mu_patch_summary.csv"
    trigger_breakdown_path = output_path / "smart_mu_patch_trigger_breakdown.csv"
    top_weight_delta_path = output_path / "smart_mu_patch_top_weight_delta.csv"
    top_contribution_delta_path = output_path / "smart_mu_patch_top_contribution_delta.csv"
    report_path = output_path / "smart_mu_patch_report.md"
    manifest_path = output_path / "manifest.json"

    daily.to_csv(daily_path, index=False)
    weights.to_csv(weights_path, index=False)
    summary.to_csv(summary_path, index=False)
    trigger_breakdown.to_csv(trigger_breakdown_path, index=False)
    top_weight_delta.to_csv(top_weight_delta_path, index=False)
    top_contribution_delta.to_csv(top_contribution_delta_path, index=False)
    manifest_path.write_text(
        json.dumps(
            _json_ready(
                {
                    "scope": "smart_mu_patch_audit_only_no_live_weight_change",
                    "config": config.__dict__,
                    "bridge_l3_csv": str(bridge_l3_csv),
                    "parameter_audit_csv": str(parameter_audit_csv),
                    "trigger_metadata": trigger_metadata,
                    "outputs": {
                        "daily": daily_path,
                        "weights": weights_path,
                        "summary": summary_path,
                        "trigger_breakdown": trigger_breakdown_path,
                        "top_weight_delta": top_weight_delta_path,
                        "top_contribution_delta": top_contribution_delta_path,
                        "report": report_path,
                    },
                }
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "# Smart-mu Patch Audit Replay",
        "",
        "Audit-only replay. Production Student-t, bridge, optimizer defaults, and L3 risk controls are not modified.",
        "",
        "## Summary",
        "",
        _markdown_table(summary),
        "",
        "## Trigger Breakdown",
        "",
        _markdown_table(trigger_breakdown),
        "",
        "## Top Positive Delta Days",
        "",
        _markdown_table(top_positive_days[["datetime", "triggered_rows", "delta_return_vs_old_mvo", "smart_mu_patch_effective_return", "old_mvo_effective_return", "top_delta_contribution_codes"]] if not top_positive_days.empty else top_positive_days),
        "",
        "## Top Negative Delta Days",
        "",
        _markdown_table(top_negative_days[["datetime", "triggered_rows", "delta_return_vs_old_mvo", "smart_mu_patch_effective_return", "old_mvo_effective_return", "top_delta_contribution_codes"]] if not top_negative_days.empty else top_negative_days),
        "",
        "## Interpretation Boundary",
        "",
        "The patch multiplies forecast_weight_mean_used by 0.3 only when the row is 5-day overheat or communication-theme in the warm-up Route A run's trading window. The replay preserves covariance, next returns, and exposure_scale from the input L3 artifact.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {
        "daily": daily_path,
        "weights": weights_path,
        "summary": summary_path,
        "trigger_breakdown": trigger_breakdown_path,
        "top_weight_delta": top_weight_delta_path,
        "top_contribution_delta": top_contribution_delta_path,
        "report": report_path,
        "manifest": manifest_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit-only Smart-mu patch replay.")
    parser.add_argument("--bridge-l3-csv", default=DEFAULT_BRIDGE_L3_CSV)
    parser.add_argument("--parameter-audit-csv", default=DEFAULT_PARAMETER_AUDIT_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--max-weight", type=float, default=bridge_core.DEFAULT_MAX_WEIGHT)
    parser.add_argument("--min-weight-floor", type=float, default=bridge_core.DEFAULT_MIN_WEIGHT_FLOOR)
    parser.add_argument("--patch-multiplier", type=float, default=SMART_MU_PATCH_MULTIPLIER)
    parser.add_argument("--communication-theme", default="communication")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SmartMuPatchConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        max_weight=float(args.max_weight),
        min_weight_floor=float(args.min_weight_floor),
        patch_multiplier=float(args.patch_multiplier),
        communication_theme=str(args.communication_theme),
    )
    outputs = run_smart_mu_patch_audit(args.bridge_l3_csv, args.parameter_audit_csv, args.output_dir, config)
    print(f"[INFO] wrote {outputs['summary']}")
    print(f"[INFO] report {outputs['report']}")


if __name__ == "__main__":
    main()