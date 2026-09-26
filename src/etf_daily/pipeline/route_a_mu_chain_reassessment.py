from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from etf_daily.pipeline.smart_mu_patch_audit import (
    DEFAULT_BRIDGE_L3_CSV,
    DEFAULT_PARAMETER_AUDIT_CSV,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    SmartMuPatchConfig,
    _compound_return,
    _json_ready,
    _load_bridge_frame,
    _max_drawdown,
    _numeric,
    _optimize_day,
    _sharpe,
)
from strategy.route_ab_moments_bridge import bridge as bridge_core

DEFAULT_OUTPUT_DIR = "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/route_a_mu_chain_reassessment_20260101_20260430"
DEFAULT_TRANSACTION_COST_BPS = 15.0
INTERNAL_LAYER_ORDER = (
    "route_a_mu_raw",
    "route_a_mu",
    "route_a_mu_smooth",
    "route_a_mu_path",
    "route_a_posterior_mu_1d",
    "route_a_posterior_mu_20d",
    "forecast_weight_mean_used",
)
SIMPLIFIED_LAYER_ORDER = (
    "route_a_mu_raw",
    "route_a_mu_path_refined",
    "route_a_used_simple_pulse",
)
SIMPLE_CHAIN_PULSE_CAP_QUANTILE = 0.99
SIMPLE_CHAIN_MAX_PENALTY_INTENSITY = 0.40
SIMPLE_CHAIN_NU_WEIGHT = 0.20
SIMPLE_CHAIN_CROWD_WEIGHT = 0.20
SIMPLE_CHAIN_REGIME_WEIGHT = 0.05
SIMPLE_CHAIN_CLUSTER_WEIGHT = 0.05
SIMPLE_CHAIN_PATH_OPTIMISM_WEIGHT = 0.15


@dataclass(frozen=True)
class LayerSpec:
    name: str
    source_columns: tuple[str, ...]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Route A mu chain reassessment. "
            "Generates layer-level signal diagnostics and mean-column replay ablation."
        )
    )
    parser.add_argument("--bridge-l3-csv", default=str(DEFAULT_BRIDGE_L3_CSV))
    parser.add_argument("--parameter-audit-csv", default=str(DEFAULT_PARAMETER_AUDIT_CSV))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--max-weight", type=float, default=float(bridge_core.DEFAULT_MAX_WEIGHT))
    parser.add_argument("--min-weight-floor", type=float, default=float(bridge_core.DEFAULT_MIN_WEIGHT_FLOOR))
    parser.add_argument("--transaction-cost-bps", type=float, default=DEFAULT_TRANSACTION_COST_BPS)
    parser.add_argument("--focus-instrument", default="SH513350")
    parser.add_argument("--focus-start-date", default="2026-04-15")
    parser.add_argument("--focus-end-date", default="2026-04-30")
    return parser.parse_args(argv)


def _infer_layers() -> list[LayerSpec]:
    return [
        LayerSpec("route_a_mu_raw", ("mu_raw", "route_a_mu_raw")),
        LayerSpec("route_a_mu", ("route_a_mu", "mu_used", "mu")),
        LayerSpec("route_a_mu_path_refined", ("route_a_mu_path_refined",)),
        LayerSpec("route_a_used_simple_pulse", ("route_a_used_simple_pulse",)),
        LayerSpec("route_a_mu_smooth", ("route_a_mu_smooth", "mu_smooth")),
        LayerSpec("route_a_mu_path", ("route_a_mu_path", "mu_path")),
        LayerSpec("route_a_posterior_mu_1d", ("route_a_posterior_mu_1d", "posterior_mu_1d")),
        LayerSpec("route_a_posterior_mu_20d", ("route_a_posterior_mu_20d", "posterior_mu_20d")),
        LayerSpec("forecast_mean_return_direct", ("forecast_mean_return_direct",)),
        LayerSpec("forecast_weight_mean_raw", ("forecast_weight_mean_raw",)),
        LayerSpec("forecast_weight_mean_slow", ("forecast_weight_mean_slow",)),
        LayerSpec("forecast_weight_mean_used", ("forecast_weight_mean_used",)),
    ]


def _resolve_layer_series(frame: pd.DataFrame, spec: LayerSpec) -> tuple[pd.Series, str] | None:
    for column in spec.source_columns:
        if column in frame.columns:
            return _numeric(frame[column]), column
    return None


def _rank_ic_by_date(mu: pd.Series, y: pd.Series, dates: pd.Series) -> pd.Series:
    work = pd.DataFrame({"mu": _numeric(mu), "y": _numeric(y), "datetime": pd.to_datetime(dates)})
    work = work.dropna(subset=["mu", "y", "datetime"])
    if work.empty:
        return pd.Series(dtype=float)
    values: dict[pd.Timestamp, float] = {}
    for date_value, group in work.groupby("datetime", sort=True):
        if len(group) < 3 or float(group["mu"].std(ddof=0)) <= 1e-12 or float(group["y"].std(ddof=0)) <= 1e-12:
            values[pd.Timestamp(date_value)] = float("nan")
            continue
        value = group["mu"].corr(group["y"], method="spearman")
        values[pd.Timestamp(date_value)] = float(value) if pd.notna(value) else float("nan")
    return pd.Series(values)


def _ic_by_date(mu: pd.Series, y: pd.Series, dates: pd.Series) -> pd.Series:
    work = pd.DataFrame({"mu": _numeric(mu), "y": _numeric(y), "datetime": pd.to_datetime(dates)})
    work = work.dropna(subset=["mu", "y", "datetime"])
    if work.empty:
        return pd.Series(dtype=float)
    values: dict[pd.Timestamp, float] = {}
    for date_value, group in work.groupby("datetime", sort=True):
        if len(group) < 3 or float(group["mu"].std(ddof=0)) <= 1e-12 or float(group["y"].std(ddof=0)) <= 1e-12:
            values[pd.Timestamp(date_value)] = float("nan")
            continue
        value = group["mu"].corr(group["y"], method="pearson")
        values[pd.Timestamp(date_value)] = float(value) if pd.notna(value) else float("nan")
    return pd.Series(values)


def _layer_diagnostics(frame: pd.DataFrame, layers: list[LayerSpec]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    y = _numeric(frame["next_return"]).fillna(0.0)
    dates = pd.to_datetime(frame["datetime"])
    abs_effective_weight = (_numeric(frame["forecast_w"]).fillna(0.0) * _numeric(frame["exposure_scale"]).fillna(0.0)).abs()

    summary_rows: list[dict[str, Any]] = []
    daily_rows: list[pd.DataFrame] = []
    source_map: dict[str, str] = {}

    for spec in layers:
        resolved = _resolve_layer_series(frame, spec)
        if resolved is None:
            continue
        mu, source_col = resolved
        source_map[spec.name] = source_col
        mu = mu.fillna(0.0)
        mu_error = mu - y
        valid_direction = mu.abs() > 1e-12
        direction_correct = ((np.sign(mu) == np.sign(y)) & valid_direction).astype(float)
        direction_correct = direction_correct.where(valid_direction, np.nan)
        positive_mu_negative_return = ((mu > 0.0) & (y < 0.0)).astype(float)
        rank_ic = _rank_ic_by_date(mu, y, dates)
        ic = _ic_by_date(mu, y, dates)

        summary_rows.append(
            {
                "layer": spec.name,
                "source_column": source_col,
                "rows": int(len(frame)),
                "avg_mu": float(mu.mean()),
                "avg_next_return": float(y.mean()),
                "avg_mu_error": float(mu_error.mean()),
                "mae_mu_error": float(mu_error.abs().mean()),
                "direction_accuracy": float(direction_correct.mean()),
                "positive_mu_negative_return_rate": float(positive_mu_negative_return.mean()),
                "weighted_mu_error_sum": float((mu_error * abs_effective_weight).sum()),
                "weighted_mu_error_per_abs_weight": float((mu_error * abs_effective_weight).sum() / max(abs_effective_weight.sum(), 1e-12)),
                "rank_ic_mean_by_date": float(rank_ic.mean()),
                "rank_ic_median_by_date": float(rank_ic.median()),
                "ic_mean_by_date": float(ic.mean()),
                "ic_median_by_date": float(ic.median()),
            }
        )

        daily = (
            pd.DataFrame(
                {
                    "datetime": pd.to_datetime(rank_ic.index),
                    "layer": spec.name,
                    "rank_ic": rank_ic.values,
                    "ic": ic.reindex(rank_ic.index).values,
                }
            )
            .sort_values("datetime")
            .reset_index(drop=True)
        )
        daily_rows.append(daily)

    summary = pd.DataFrame(summary_rows).sort_values("weighted_mu_error_sum", ascending=False)
    daily = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
    return summary, daily, source_map


def _run_replay_ablation(
    frame: pd.DataFrame,
    layers: list[LayerSpec],
    config: SmartMuPatchConfig,
    transaction_cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(pd.to_datetime(frame["datetime"]).drop_duplicates())
    daily_rows: list[dict[str, Any]] = []

    tc = float(transaction_cost_bps) / 1e4

    for spec in layers:
        resolved = _resolve_layer_series(frame, spec)
        if resolved is None:
            continue
        mu_values, source_col = resolved
        working = frame.copy()
        working["__mu_layer"] = _numeric(mu_values).fillna(0.0)

        prev_weights = pd.Series(dtype=float)
        layer_daily_returns: list[float] = []
        layer_net_daily_returns: list[float] = []
        layer_turnover: list[float] = []

        for date_value in dates:
            day = working[pd.to_datetime(working["datetime"]).eq(date_value)].copy()
            if day.empty:
                continue
            weights, status = _optimize_day(
                day,
                mean_column="__mu_layer",
                previous_weights=prev_weights,
                config=config,
            )
            sample_index = day["instrument"].astype(str)
            mvo_weight = weights.reindex(sample_index).fillna(0.0)
            exposure = pd.Series(_numeric(day["exposure_scale"]).fillna(0.0).to_numpy(), index=sample_index)
            next_return = pd.Series(_numeric(day["next_return"]).fillna(0.0).to_numpy(), index=sample_index)
            effective = mvo_weight * exposure
            gross_return = float((effective * next_return).sum())

            if prev_weights.empty:
                turnover = float(mvo_weight.abs().sum())
            else:
                union = prev_weights.index.union(mvo_weight.index)
                turnover = float((mvo_weight.reindex(union).fillna(0.0) - prev_weights.reindex(union).fillna(0.0)).abs().sum())
            tcost = tc * turnover
            net_return = gross_return - tcost

            layer_daily_returns.append(gross_return)
            layer_net_daily_returns.append(net_return)
            layer_turnover.append(turnover)
            prev_weights = mvo_weight.copy()

            daily_rows.append(
                {
                    "datetime": pd.Timestamp(date_value),
                    "layer": spec.name,
                    "source_column": source_col,
                    "optimizer_status": status,
                    "daily_return": gross_return,
                    "daily_return_after_tcost": net_return,
                    "one_way_turnover": turnover,
                    "transaction_cost_return": tcost,
                }
            )

    replay_daily = pd.DataFrame(daily_rows).sort_values(["datetime", "layer"]).reset_index(drop=True)
    if replay_daily.empty:
        return pd.DataFrame(), replay_daily

    summary_rows: list[dict[str, Any]] = []
    for layer_name, group in replay_daily.groupby("layer", sort=False):
        source_col = str(group["source_column"].dropna().iloc[0]) if group["source_column"].notna().any() else ""
        gross = _numeric(group["daily_return"])
        net = _numeric(group["daily_return_after_tcost"])
        turnover = _numeric(group["one_way_turnover"])
        summary_rows.append(
            {
                "layer": str(layer_name),
                "source_column": source_col,
                "days": int(len(group)),
                "cumulative_return": _compound_return(gross),
                "net_cumulative_return_after_tcost": _compound_return(net),
                "daily_return_sum": float(gross.sum()),
                "net_daily_return_sum_after_tcost": float(net.sum()),
                "max_drawdown": _max_drawdown(gross),
                "net_max_drawdown_after_tcost": _max_drawdown(net),
                "sharpe": _sharpe(gross),
                "net_sharpe_after_tcost": _sharpe(net),
                "total_one_way_turnover": float(turnover.sum()),
                "avg_one_way_turnover": float(turnover.mean()),
            }
        )
    replay_summary = pd.DataFrame(summary_rows).sort_values("net_cumulative_return_after_tcost", ascending=False)
    return replay_summary, replay_daily


def _build_ordered_layer_focus(
    layer_summary: pd.DataFrame,
    replay_summary: pd.DataFrame,
    layer_order: tuple[str, ...],
) -> pd.DataFrame:
    if layer_summary.empty:
        return pd.DataFrame()

    internal_order = {layer: idx for idx, layer in enumerate(layer_order)}
    signal_focus = layer_summary[layer_summary["layer"].isin(layer_order)].copy()
    if signal_focus.empty:
        return pd.DataFrame()

    replay_cols = [
        "layer",
        "net_cumulative_return_after_tcost",
        "net_daily_return_sum_after_tcost",
        "net_max_drawdown_after_tcost",
        "net_sharpe_after_tcost",
        "avg_one_way_turnover",
    ]
    replay_focus = replay_summary[replay_cols].copy() if not replay_summary.empty else pd.DataFrame(columns=replay_cols)
    focus = signal_focus.merge(replay_focus, on="layer", how="left")
    focus["stage_order"] = focus["layer"].map(internal_order)
    focus = focus.sort_values("stage_order").reset_index(drop=True)

    delta_columns = [
        "weighted_mu_error_sum",
        "mae_mu_error",
        "direction_accuracy",
        "positive_mu_negative_return_rate",
        "rank_ic_mean_by_date",
        "ic_mean_by_date",
        "net_cumulative_return_after_tcost",
        "net_max_drawdown_after_tcost",
        "net_sharpe_after_tcost",
        "avg_one_way_turnover",
    ]
    for column in delta_columns:
        if column in focus.columns:
            focus[f"{column}_delta_vs_prev"] = focus[column].diff()

    focus["weighted_mu_error_rank"] = focus["weighted_mu_error_sum"].rank(method="dense", ascending=True)
    focus["net_return_rank"] = focus["net_cumulative_return_after_tcost"].rank(method="dense", ascending=False)
    return focus


def _build_internal_layer_focus(
    layer_summary: pd.DataFrame,
    replay_summary: pd.DataFrame,
) -> pd.DataFrame:
    return _build_ordered_layer_focus(layer_summary, replay_summary, INTERNAL_LAYER_ORDER)


def _build_simplified_layer_focus(
    layer_summary: pd.DataFrame,
    replay_summary: pd.DataFrame,
) -> pd.DataFrame:
    return _build_ordered_layer_focus(layer_summary, replay_summary, SIMPLIFIED_LAYER_ORDER)


def _build_internal_bad_day_attribution(
    frame: pd.DataFrame,
    layers: list[LayerSpec],
    layer_daily: pd.DataFrame,
    replay_daily: pd.DataFrame,
    parameter_audit: pd.DataFrame,
    layer_order: tuple[str, ...] = INTERNAL_LAYER_ORDER,
    final_layer: str = "forecast_weight_mean_used",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    internal_specs = [spec for spec in layers if spec.name in layer_order]
    if not internal_specs:
        return pd.DataFrame(), pd.DataFrame()

    working = frame.copy()
    if "sigma_route_a" not in working.columns and not parameter_audit.empty:
        audit_meta_columns = ["datetime", "instrument"]
        audit_source_map = {
            "sigma_route_a": "route_a_params_sigma",
            "nu_route_a": "route_a_params_nu",
            "signal_strength_route_a": "route_a_params_signal_strength",
        }
        available = [source for source in audit_source_map.values() if source in parameter_audit.columns]
        if available:
            audit_meta = parameter_audit[audit_meta_columns + available].copy()
            rename_map = {source: target for target, source in audit_source_map.items() if source in audit_meta.columns}
            audit_meta = audit_meta.rename(columns=rename_map).drop_duplicates(["datetime", "instrument"])
            working = working.merge(audit_meta, on=["datetime", "instrument"], how="left")

    working["abs_effective_weight"] = (
        _numeric(working["forecast_w"]).fillna(0.0) * _numeric(working["exposure_scale"]).fillna(0.0)
    ).abs()
    sigma = _numeric(working.get("sigma_route_a", np.nan))

    route_a_mu_resolved = _resolve_layer_series(working, LayerSpec("route_a_mu", ("route_a_mu", "mu_used", "mu")))
    route_a_mu = route_a_mu_resolved[0] if route_a_mu_resolved is not None else pd.Series(np.nan, index=working.index)
    if sigma.notna().any() and (sigma.abs() > 1e-12).any():
        baseline_ratio = np.where(sigma.abs() > 1e-12, route_a_mu.abs() / sigma.abs(), np.nan)
        certainty_threshold = float(pd.Series(baseline_ratio).dropna().quantile(0.75))
        certainty_mode = "mu_over_sigma_q75_from_route_a_mu"
    else:
        certainty_threshold = float(route_a_mu.abs().dropna().quantile(0.75)) if route_a_mu.notna().any() else np.nan
        certainty_mode = "abs_mu_q75_fallback"

    replay_lookup = replay_daily[["datetime", "layer", "daily_return_after_tcost", "one_way_turnover"]].copy() if not replay_daily.empty else pd.DataFrame(columns=["datetime", "layer", "daily_return_after_tcost", "one_way_turnover"])
    metric_lookup = layer_daily[["datetime", "layer", "rank_ic", "ic"]].copy() if not layer_daily.empty else pd.DataFrame(columns=["datetime", "layer", "rank_ic", "ic"])
    order_map = {layer: idx for idx, layer in enumerate(layer_order)}

    daily_rows: list[dict[str, Any]] = []
    for spec in internal_specs:
        resolved = _resolve_layer_series(working, spec)
        if resolved is None:
            continue
        mu_layer, source_column = resolved
        layer_frame = working[["datetime", "instrument", "factor_industry_theme", "next_return", "abs_effective_weight"]].copy()
        layer_frame["mu_layer"] = _numeric(mu_layer).fillna(0.0)
        layer_frame["weighted_mu_error"] = (layer_frame["mu_layer"] - _numeric(working["next_return"]).fillna(0.0)) * layer_frame["abs_effective_weight"]
        layer_frame["positive_mu_negative_return_flag"] = (layer_frame["mu_layer"] > 0.0) & (_numeric(working["next_return"]).fillna(0.0) < 0.0)

        if certainty_mode == "mu_over_sigma_q75_from_route_a_mu":
            layer_frame["certainty_ratio"] = np.where(sigma.abs() > 1e-12, layer_frame["mu_layer"].abs() / sigma.abs(), np.nan)
            layer_frame["high_confidence_wrong_flag"] = layer_frame["positive_mu_negative_return_flag"] & (layer_frame["certainty_ratio"] >= certainty_threshold)
        else:
            layer_frame["certainty_ratio"] = layer_frame["mu_layer"].abs()
            layer_frame["high_confidence_wrong_flag"] = layer_frame["positive_mu_negative_return_flag"] & (layer_frame["mu_layer"].abs() >= certainty_threshold)

        layer_frame["high_confidence_wrong_abs_weight"] = layer_frame["abs_effective_weight"].where(layer_frame["high_confidence_wrong_flag"], 0.0)
        layer_frame["high_confidence_wrong_weighted_mu_error"] = layer_frame["weighted_mu_error"].where(layer_frame["high_confidence_wrong_flag"], 0.0)

        for date_value, group in layer_frame.groupby("datetime", sort=True):
            wrong_group = group[group["high_confidence_wrong_flag"]].copy()
            top_wrong_instrument = ""
            top_wrong_theme = ""
            top_wrong_abs_weight = 0.0
            top_wrong_weighted_mu_error = 0.0
            if not wrong_group.empty:
                top_row = wrong_group.sort_values(["high_confidence_wrong_abs_weight", "weighted_mu_error"], ascending=False).iloc[0]
                top_wrong_instrument = str(top_row["instrument"])
                top_wrong_theme = str(top_row.get("factor_industry_theme", ""))
                top_wrong_abs_weight = float(top_row["high_confidence_wrong_abs_weight"])
                top_wrong_weighted_mu_error = float(top_row["weighted_mu_error"])

            daily_rows.append(
                {
                    "datetime": pd.Timestamp(date_value),
                    "layer": spec.name,
                    "stage_order": order_map.get(spec.name),
                    "source_column": source_column,
                    "certainty_threshold": certainty_threshold,
                    "certainty_mode": certainty_mode,
                    "positive_mu_negative_return_count": int(group["positive_mu_negative_return_flag"].sum()),
                    "high_confidence_wrong_count": int(group["high_confidence_wrong_flag"].sum()),
                    "high_confidence_wrong_abs_weight_sum": float(group["high_confidence_wrong_abs_weight"].sum()),
                    "high_confidence_wrong_weighted_mu_error_sum": float(group["high_confidence_wrong_weighted_mu_error"].sum()),
                    "weighted_mu_error_sum": float(group["weighted_mu_error"].sum()),
                    "top_wrong_instrument": top_wrong_instrument,
                    "top_wrong_theme": top_wrong_theme,
                    "top_wrong_abs_weight": top_wrong_abs_weight,
                    "top_wrong_weighted_mu_error": top_wrong_weighted_mu_error,
                    "max_certainty_ratio": float(group.loc[group["high_confidence_wrong_flag"], "certainty_ratio"].max()) if wrong_group.shape[0] else np.nan,
                }
            )

    bad_day_daily = pd.DataFrame(daily_rows)
    if bad_day_daily.empty:
        return bad_day_daily, pd.DataFrame()

    bad_day_daily = bad_day_daily.merge(metric_lookup, on=["datetime", "layer"], how="left")
    bad_day_daily = bad_day_daily.merge(replay_lookup, on=["datetime", "layer"], how="left")
    bad_day_daily = bad_day_daily.sort_values(["datetime", "stage_order"]).reset_index(drop=True)

    for column in [
        "high_confidence_wrong_count",
        "high_confidence_wrong_abs_weight_sum",
        "high_confidence_wrong_weighted_mu_error_sum",
        "weighted_mu_error_sum",
        "daily_return_after_tcost",
        "rank_ic",
        "ic",
    ]:
        bad_day_daily[f"{column}_delta_vs_prev"] = bad_day_daily.groupby("datetime")[column].diff()

    summary_rows: list[dict[str, Any]] = []
    for date_value, group in bad_day_daily.groupby("datetime", sort=True):
        group = group.sort_values("stage_order").reset_index(drop=True)
        first_bad = group[group["high_confidence_wrong_count"] > 0].head(1)
        first_incremental = group[
            (group["high_confidence_wrong_count"] > 0)
            & (group["high_confidence_wrong_count_delta_vs_prev"].fillna(group["high_confidence_wrong_count"]) > 0)
        ].head(1)
        final_row = group[group["layer"].eq(final_layer)].tail(1)

        row: dict[str, Any] = {
            "datetime": pd.Timestamp(date_value),
            "bad_day_final_used_flag": bool(final_row["high_confidence_wrong_count"].iloc[0] > 0) if not final_row.empty else False,
            "first_high_confidence_wrong_layer": "",
            "first_high_confidence_wrong_stage_order": np.nan,
            "first_high_confidence_wrong_top_instrument": "",
            "first_high_confidence_wrong_top_theme": "",
            "first_incremental_bad_layer": "",
            "first_incremental_bad_stage_order": np.nan,
            "first_incremental_bad_top_instrument": "",
            "final_used_high_confidence_wrong_count": int(final_row["high_confidence_wrong_count"].iloc[0]) if not final_row.empty else 0,
            "final_used_high_confidence_wrong_abs_weight_sum": float(final_row["high_confidence_wrong_abs_weight_sum"].iloc[0]) if not final_row.empty else 0.0,
            "final_used_weighted_mu_error_sum": float(final_row["weighted_mu_error_sum"].iloc[0]) if not final_row.empty else 0.0,
            "final_used_daily_return_after_tcost": float(final_row["daily_return_after_tcost"].iloc[0]) if not final_row.empty and pd.notna(final_row["daily_return_after_tcost"].iloc[0]) else np.nan,
            "final_used_rank_ic": float(final_row["rank_ic"].iloc[0]) if not final_row.empty and pd.notna(final_row["rank_ic"].iloc[0]) else np.nan,
            "final_used_ic": float(final_row["ic"].iloc[0]) if not final_row.empty and pd.notna(final_row["ic"].iloc[0]) else np.nan,
        }
        if not first_bad.empty:
            row["first_high_confidence_wrong_layer"] = str(first_bad["layer"].iloc[0])
            row["first_high_confidence_wrong_stage_order"] = int(first_bad["stage_order"].iloc[0])
            row["first_high_confidence_wrong_top_instrument"] = str(first_bad["top_wrong_instrument"].iloc[0])
            row["first_high_confidence_wrong_top_theme"] = str(first_bad["top_wrong_theme"].iloc[0])
        if not first_incremental.empty:
            row["first_incremental_bad_layer"] = str(first_incremental["layer"].iloc[0])
            row["first_incremental_bad_stage_order"] = int(first_incremental["stage_order"].iloc[0])
            row["first_incremental_bad_top_instrument"] = str(first_incremental["top_wrong_instrument"].iloc[0])
        summary_rows.append(row)

    bad_day_summary = pd.DataFrame(summary_rows).sort_values("datetime").reset_index(drop=True)
    return bad_day_daily, bad_day_summary


def _coalesce_numeric(frame: pd.DataFrame, columns: tuple[str, ...], default: float = np.nan) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in columns:
        if column in frame.columns:
            result = result.where(result.notna(), _numeric(frame[column]))
    return result.fillna(default)


def _safe_quantile(values: pd.Series, quantile: float, default: float = 0.0) -> float:
    clean = _numeric(values).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return float(default)
    return float(clean.quantile(float(quantile)))


def _scaled_excess(values: pd.Series, low: float, high: float) -> pd.Series:
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return pd.Series(0.0, index=values.index, dtype=float)
    return ((_numeric(values) - float(low)) / float(high - low)).clip(lower=0.0, upper=1.0).fillna(0.0)


def _date_theme_positive_mu_unit(base_mu: pd.Series, dates: pd.Series, themes: pd.Series) -> pd.Series:
    work = pd.DataFrame(
        {
            "datetime": pd.to_datetime(dates),
            "theme": themes.astype(str).fillna(""),
            "positive_mu": _numeric(base_mu).where(_numeric(base_mu) > 0.0),
        },
        index=base_mu.index,
    )
    theme_unit = work.groupby(["datetime", "theme"])["positive_mu"].transform("median")
    date_unit = work.groupby("datetime")["positive_mu"].transform("median")
    global_unit = _safe_quantile(work["positive_mu"], 0.50, default=0.0)
    return theme_unit.fillna(date_unit).fillna(global_unit).fillna(0.0)


def _attach_simplified_shadow_chain(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    working = frame.copy()
    base_mu = _coalesce_numeric(working, ("route_a_mu", "mu_used", "mu"), default=0.0)
    mu_path = _coalesce_numeric(working, ("route_a_mu_path", "mu_path"), default=np.nan)
    nu = _coalesce_numeric(working, ("nu_route_a", "route_a_nu", "route_a_params_nu"), default=np.nan)
    crowd_score = _coalesce_numeric(working, ("forecast_active_set_crowded_score",), default=np.nan)
    risk_regime_on = _coalesce_numeric(working, ("risk_regime_on",), default=0.0)
    risk_regime_factor = _coalesce_numeric(working, ("risk_regime_factor",), default=1.0)
    cluster_scale = _coalesce_numeric(working, ("risk_cluster_scale",), default=1.0)

    nu_q75 = _safe_quantile(nu, 0.75, default=np.nan)
    nu_q95 = _safe_quantile(nu, 0.95, default=np.nan)
    crowd_q75 = _safe_quantile(crowd_score, 0.75, default=np.nan)
    crowd_q95 = _safe_quantile(crowd_score, 0.95, default=np.nan)
    nu_excess = _scaled_excess(nu, nu_q75, nu_q95)
    crowd_excess = _scaled_excess(crowd_score, crowd_q75, crowd_q95)
    regime_flag = ((risk_regime_on > 0.0) | (risk_regime_factor < 0.999)).astype(float)
    cluster_flag = (cluster_scale < 0.999).astype(float)

    theme_unit = _date_theme_positive_mu_unit(base_mu, working["datetime"], working["factor_industry_theme"])
    path_optimism = (mu_path - base_mu).clip(lower=0.0).where(base_mu > 0.0, 0.0).fillna(0.0)
    penalty_intensity = (
        SIMPLE_CHAIN_NU_WEIGHT * nu_excess
        + SIMPLE_CHAIN_CROWD_WEIGHT * crowd_excess
        + SIMPLE_CHAIN_REGIME_WEIGHT * regime_flag
        + SIMPLE_CHAIN_CLUSTER_WEIGHT * cluster_flag
    ).clip(lower=0.0, upper=SIMPLE_CHAIN_MAX_PENALTY_INTENSITY)
    additive_bias = theme_unit * penalty_intensity + SIMPLE_CHAIN_PATH_OPTIMISM_WEIGHT * path_optimism

    refined = base_mu.copy()
    positive_mask = refined > 0.0
    refined.loc[positive_mask] = np.maximum(refined.loc[positive_mask] - additive_bias.loc[positive_mask], 0.0)

    cap_by_date = refined.abs().groupby(pd.to_datetime(working["datetime"])).transform(
        lambda values: float(values.quantile(SIMPLE_CHAIN_PULSE_CAP_QUANTILE)) if values.notna().any() else np.nan
    )
    fallback_cap = _safe_quantile(refined.abs(), SIMPLE_CHAIN_PULSE_CAP_QUANTILE, default=0.0)
    cap = cap_by_date.fillna(fallback_cap).clip(lower=1e-8)
    simple_used = refined.clip(lower=-cap, upper=cap)

    working["route_a_mu_path_refined_base"] = base_mu
    working["route_a_mu_path_refined_nu_excess"] = nu_excess
    working["route_a_mu_path_refined_crowd_excess"] = crowd_excess
    working["route_a_mu_path_refined_regime_flag"] = regime_flag
    working["route_a_mu_path_refined_cluster_flag"] = cluster_flag
    working["route_a_mu_path_refined_theme_unit"] = theme_unit
    working["route_a_mu_path_refined_path_optimism"] = path_optimism
    working["route_a_mu_path_refined_penalty_intensity"] = penalty_intensity
    working["route_a_mu_path_refined_bg"] = additive_bias
    working["route_a_mu_path_refined"] = refined
    working["route_a_used_simple_pulse_cap"] = cap
    working["route_a_used_simple_pulse"] = simple_used
    return working


def _load_parameter_audit_frame(path: str | Path, start_date: str, end_date: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(file_path, parse_dates=["datetime"])
    frame["instrument"] = frame["instrument"].astype(str).str.upper()
    frame = frame[
        (frame["datetime"] >= pd.Timestamp(start_date)) & (frame["datetime"] <= pd.Timestamp(end_date))
    ].copy()
    return frame.sort_values(["datetime", "instrument"]).reset_index(drop=True)


def _enrich_bridge_frame_with_parameter_layers(
    bridge_frame: pd.DataFrame,
    parameter_audit: pd.DataFrame,
) -> pd.DataFrame:
    if bridge_frame.empty or parameter_audit.empty:
        return bridge_frame

    layer_sources = {
        "route_a_mu_smooth": ("route_a_params_mu_smooth", "route_a_eval_mu_smooth"),
        "route_a_mu_path": ("route_a_params_mu_path", "route_a_eval_mu_path"),
        "route_a_posterior_mu_1d": ("route_a_params_posterior_mu_1d", "route_a_eval_posterior_mu_1d"),
        "route_a_posterior_mu_20d": ("route_a_params_posterior_mu_20d", "route_a_eval_posterior_mu_20d"),
        "sigma_route_a": ("route_a_params_sigma",),
        "nu_route_a": ("route_a_params_nu",),
        "signal_strength_route_a": ("route_a_params_signal_strength", "route_a_eval_signal_strength"),
    }
    if not any(any(source in parameter_audit.columns for source in sources) for sources in layer_sources.values()):
        return bridge_frame

    join_frame = parameter_audit[["datetime", "instrument"]].copy()
    for target_column, source_candidates in layer_sources.items():
        if target_column in bridge_frame.columns:
            continue
        series = pd.Series(np.nan, index=parameter_audit.index, dtype=float)
        for source_column in source_candidates:
            if source_column in parameter_audit.columns:
                source_values = _numeric(parameter_audit[source_column])
                series = series.where(series.notna(), source_values)
        join_frame[target_column] = series

    added_columns = [column for column in join_frame.columns if column not in {"datetime", "instrument"}]
    if not added_columns:
        return bridge_frame

    join_frame = join_frame.drop_duplicates(["datetime", "instrument"])
    merged = bridge_frame.merge(join_frame, on=["datetime", "instrument"], how="left")
    return merged


def _phase2_decomposition(
    parameter_audit: pd.DataFrame,
    focus_instrument: str,
    focus_start_date: str,
    focus_end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if parameter_audit.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    work = parameter_audit.copy()
    work["datetime"] = pd.to_datetime(work["datetime"])  # guard for caller overrides
    work["instrument"] = work["instrument"].astype(str).str.upper()

    work["mu_route_a"] = _numeric(work.get("route_a_params_mu", 0.0)).fillna(0.0)
    work["mu_used_bridge"] = _numeric(work.get("forecast_weight_mean_used", 0.0)).fillna(0.0)
    work["sigma_route_a"] = _numeric(work.get("route_a_params_sigma", np.nan))
    work["nu_route_a"] = _numeric(work.get("route_a_params_nu", np.nan))
    work["signal_strength_route_a"] = _numeric(work.get("route_a_params_signal_strength", np.nan))
    work["nll_route_a"] = _numeric(work.get("route_a_eval_nll", np.nan))
    work["nll_raw_route_a"] = _numeric(work.get("route_a_eval_nll_raw", np.nan))
    work["pred_mean_20"] = _numeric(work.get("route_a_params_pred_mean_20", np.nan))
    work["real_mean_20"] = _numeric(work.get("route_a_params_real_mean_20", np.nan))
    work["pred_ann20"] = _numeric(work.get("route_a_params_pred_ann20", np.nan))
    work["real_ann20"] = _numeric(work.get("route_a_params_real_ann20", np.nan))
    work["cum_gap_path"] = _numeric(work.get("route_a_eval_cum_gap_path", np.nan))
    work["cum_gap_mu"] = _numeric(work.get("route_a_eval_cum_gap_mu", np.nan))
    work["next_return"] = _numeric(work.get("next_return", 0.0)).fillna(0.0)
    work["forecast_w"] = _numeric(work.get("forecast_w", 0.0)).fillna(0.0)
    work["exposure_scale"] = _numeric(work.get("exposure_scale", 0.0)).fillna(0.0)

    work["effective_weight"] = work["forecast_w"] * work["exposure_scale"]
    work["abs_effective_weight"] = work["effective_weight"].abs()
    work["realized_contribution"] = work["effective_weight"] * work["next_return"]
    work["forecast_mu_contribution"] = work["effective_weight"] * work["mu_used_bridge"]
    work["weighted_mu_error"] = (work["mu_used_bridge"] - work["next_return"]) * work["abs_effective_weight"]

    work["path_gap_mean20_abs"] = (work["pred_mean_20"] - work["real_mean_20"]).abs()
    work["path_gap_ann20_abs"] = (work["pred_ann20"] - work["real_ann20"]).abs()
    work["path_gap_cum_path_abs"] = work["cum_gap_path"].abs()
    work["path_gap_cum_mu_abs"] = work["cum_gap_mu"].abs()

    work["mu_sigma_abs_ratio"] = np.where(
        work["sigma_route_a"].fillna(0.0).abs() > 1e-12,
        work["mu_route_a"].abs() / work["sigma_route_a"].abs(),
        np.nan,
    )
    work["posterior_std"] = _numeric(work.get("route_a_params_posterior_mu_std", np.nan))
    work["nu_available"] = work["nu_route_a"].notna()
    work["assessment_reason"] = "not_focus_row"
    work["nu_diagnosis_note"] = np.where(
        work["nu_available"],
        "nu_available_direct_evidence_use_nu_route_a_for_tail_heaviness",
        "nu_not_available_in_current_artifacts_use_sigma_and_signal_strength_proxy",
    )

    valid_nll = work["nll_route_a"].replace([np.inf, -np.inf], np.nan).dropna()
    valid_path = work["path_gap_mean20_abs"].replace([np.inf, -np.inf], np.nan).dropna()
    nll_good_th = float(valid_nll.quantile(0.30)) if not valid_nll.empty else np.nan
    path_bad_th = float(valid_path.quantile(0.70)) if not valid_path.empty else np.nan

    work["nll_good_flag"] = work["nll_route_a"] <= nll_good_th if np.isfinite(nll_good_th) else False
    work["path_bad_flag"] = work["path_gap_mean20_abs"] >= path_bad_th if np.isfinite(path_bad_th) else False
    work["nll_good_but_path_bad_flag"] = work["nll_good_flag"] & work["path_bad_flag"]

    nll_local_good_th = work.groupby("instrument")["nll_route_a"].transform(
        lambda s: float(s.quantile(0.50)) if s.notna().any() else np.nan
    )
    path_local_bad_th = work.groupby("instrument")["path_gap_mean20_abs"].transform(
        lambda s: float(s.quantile(0.70)) if s.notna().any() else np.nan
    )
    work["nll_good_local_flag"] = work["nll_route_a"] <= nll_local_good_th
    work["path_bad_local_flag"] = work["path_gap_mean20_abs"] >= path_local_bad_th
    work["nll_good_but_path_bad_local_flag"] = work["nll_good_local_flag"] & work["path_bad_local_flag"]

    ratio_q75 = float(work["mu_sigma_abs_ratio"].dropna().quantile(0.75)) if work["mu_sigma_abs_ratio"].notna().any() else np.nan
    work["high_certainty_proxy_flag"] = (
        (work["mu_sigma_abs_ratio"] >= ratio_q75) if np.isfinite(ratio_q75) else False
    )

    summary = (
        work.groupby("instrument", as_index=False)
        .agg(
            rows=("instrument", "size"),
            dates=("datetime", "nunique"),
            avg_mu_route_a=("mu_route_a", "mean"),
            avg_sigma_route_a=("sigma_route_a", "mean"),
            avg_nu_route_a=("nu_route_a", "mean"),
            avg_nll=("nll_route_a", "mean"),
            avg_path_gap_mean20_abs=("path_gap_mean20_abs", "mean"),
            avg_path_gap_ann20_abs=("path_gap_ann20_abs", "mean"),
            nll_good_but_path_bad_rows=("nll_good_but_path_bad_flag", "sum"),
            nll_good_but_path_bad_local_rows=("nll_good_but_path_bad_local_flag", "sum"),
            high_certainty_proxy_rows=("high_certainty_proxy_flag", "sum"),
            weighted_mu_error_sum=("weighted_mu_error", "sum"),
            realized_contribution_sum=("realized_contribution", "sum"),
            forecast_mu_contribution_sum=("forecast_mu_contribution", "sum"),
        )
        .sort_values("weighted_mu_error_sum", ascending=False)
        .reset_index(drop=True)
    )

    focus_code = str(focus_instrument).upper().strip()
    focus = work[
        work["instrument"].eq(focus_code)
        & (work["datetime"] >= pd.Timestamp(focus_start_date))
        & (work["datetime"] <= pd.Timestamp(focus_end_date))
    ].copy()
    focus = focus.sort_values("datetime")
    focus_negative = focus[focus["realized_contribution"] < 0.0].copy()
    focus_negative["assessment_reason"] = np.where(
        (focus_negative["mu_route_a"] > 0.0)
        & (focus_negative["next_return"] < 0.0)
        & (focus_negative["high_certainty_proxy_flag"].fillna(False)),
        "mu_positive_but_realized_negative_with_high_certainty_proxy",
        np.where(
            (focus_negative["mu_route_a"] > 0.0) & (focus_negative["next_return"] < 0.0),
            "mu_positive_but_realized_negative",
            "other",
        ),
    )
    focus_negative["nu_diagnosis_note"] = np.where(
        focus_negative["nu_available"],
        "nu_available_direct_evidence_use_nu_route_a_for_tail_heaviness",
        "nu_not_available_in_current_artifacts_use_sigma_and_signal_strength_proxy",
    )

    detail_columns = [
        "datetime",
        "instrument",
        "factor_industry_theme",
        "mu_route_a",
        "mu_used_bridge",
        "sigma_route_a",
        "nu_route_a",
        "signal_strength_route_a",
        "mu_sigma_abs_ratio",
        "posterior_std",
        "nll_route_a",
        "nll_raw_route_a",
        "path_gap_mean20_abs",
        "path_gap_ann20_abs",
        "path_gap_cum_path_abs",
        "path_gap_cum_mu_abs",
        "next_return",
        "effective_weight",
        "realized_contribution",
        "forecast_mu_contribution",
        "weighted_mu_error",
        "nll_good_but_path_bad_flag",
        "nll_good_but_path_bad_local_flag",
        "high_certainty_proxy_flag",
        "assessment_reason",
        "nu_available",
        "nu_diagnosis_note",
    ]
    detail = work[detail_columns].sort_values(["datetime", "instrument"]).reset_index(drop=True)
    focus_output = focus_negative[detail_columns].sort_values("realized_contribution").reset_index(drop=True)
    return detail, summary, focus_output


def run_route_a_mu_chain_reassessment(
    bridge_l3_csv: str | Path,
    parameter_audit_csv: str | Path,
    output_dir: str | Path,
    start_date: str,
    end_date: str,
    max_weight: float,
    min_weight_floor: float,
    transaction_cost_bps: float,
    focus_instrument: str,
    focus_start_date: str,
    focus_end_date: str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frame = _load_bridge_frame(bridge_l3_csv, start_date, end_date)
    parameter_audit = _load_parameter_audit_frame(parameter_audit_csv, start_date, end_date)
    frame = _enrich_bridge_frame_with_parameter_layers(frame, parameter_audit)
    frame = _attach_simplified_shadow_chain(frame)
    layers = _infer_layers()

    layer_summary, layer_daily, source_map = _layer_diagnostics(frame, layers)

    replay_config = SmartMuPatchConfig(
        start_date=start_date,
        end_date=end_date,
        max_weight=float(max_weight),
        min_weight_floor=float(min_weight_floor),
        patch_multiplier=1.0,
        communication_theme="communication",
    )
    replay_summary, replay_daily = _run_replay_ablation(
        frame,
        layers,
        config=replay_config,
        transaction_cost_bps=float(transaction_cost_bps),
    )
    internal_focus = _build_internal_layer_focus(layer_summary, replay_summary)
    simplified_focus = _build_simplified_layer_focus(layer_summary, replay_summary)
    internal_bad_day_daily, internal_bad_day_summary = _build_internal_bad_day_attribution(
        frame=frame,
        layers=layers,
        layer_daily=layer_daily,
        replay_daily=replay_daily,
        parameter_audit=parameter_audit,
    )
    simplified_bad_day_daily, simplified_bad_day_summary = _build_internal_bad_day_attribution(
        frame=frame,
        layers=layers,
        layer_daily=layer_daily,
        replay_daily=replay_daily,
        parameter_audit=parameter_audit,
        layer_order=SIMPLIFIED_LAYER_ORDER,
        final_layer="route_a_used_simple_pulse",
    )

    phase2_detail, phase2_summary, phase2_focus = _phase2_decomposition(
        parameter_audit=parameter_audit,
        focus_instrument=focus_instrument,
        focus_start_date=focus_start_date,
        focus_end_date=focus_end_date,
    )

    layer_summary_path = output_path / "route_a_mu_chain_layer_metrics.csv"
    layer_daily_path = output_path / "route_a_mu_chain_layer_daily_metrics.csv"
    replay_summary_path = output_path / "route_a_mu_chain_replay_summary.csv"
    replay_daily_path = output_path / "route_a_mu_chain_replay_daily.csv"
    internal_focus_path = output_path / "route_a_mu_chain_internal_layer_focus.csv"
    simplified_focus_path = output_path / "route_a_mu_chain_simplified_layer_focus.csv"
    internal_bad_day_daily_path = output_path / "route_a_mu_chain_internal_bad_day_daily.csv"
    internal_bad_day_summary_path = output_path / "route_a_mu_chain_internal_bad_day_summary.csv"
    simplified_bad_day_daily_path = output_path / "route_a_mu_chain_simplified_bad_day_daily.csv"
    simplified_bad_day_summary_path = output_path / "route_a_mu_chain_simplified_bad_day_summary.csv"
    phase2_detail_path = output_path / "route_a_mu_chain_phase2_loss_detail.csv"
    phase2_summary_path = output_path / "route_a_mu_chain_phase2_loss_summary.csv"
    phase2_focus_path = output_path / "route_a_mu_chain_phase2_bad_date_drilldown.csv"
    report_path = output_path / "route_a_mu_chain_reassessment_report.md"
    manifest_path = output_path / "manifest.json"

    layer_summary.to_csv(layer_summary_path, index=False)
    layer_daily.to_csv(layer_daily_path, index=False)
    replay_summary.to_csv(replay_summary_path, index=False)
    replay_daily.to_csv(replay_daily_path, index=False)
    internal_focus.to_csv(internal_focus_path, index=False)
    simplified_focus.to_csv(simplified_focus_path, index=False)
    internal_bad_day_daily.to_csv(internal_bad_day_daily_path, index=False)
    internal_bad_day_summary.to_csv(internal_bad_day_summary_path, index=False)
    simplified_bad_day_daily.to_csv(simplified_bad_day_daily_path, index=False)
    simplified_bad_day_summary.to_csv(simplified_bad_day_summary_path, index=False)
    phase2_detail.to_csv(phase2_detail_path, index=False)
    phase2_summary.to_csv(phase2_summary_path, index=False)
    phase2_focus.to_csv(phase2_focus_path, index=False)

    report_lines = [
        "# Route A Mu Chain Reassessment",
        "",
        "Read-only audit. No production Route A, bridge, covariance, MVO constraints, or L3 logic were modified.",
        "",
        "## Layer Metrics",
        "",
        layer_summary.head(30).to_markdown(index=False) if not layer_summary.empty else "No rows.",
        "",
        "## Replay Ablation Summary",
        "",
        replay_summary.head(30).to_markdown(index=False) if not replay_summary.empty else "No rows.",
        "",
        "## Internal Layer Focus",
        "",
        internal_focus.to_markdown(index=False) if not internal_focus.empty else "No rows.",
        "",
        "## Simplified Three-Layer Shadow Chain",
        "",
        simplified_focus.to_markdown(index=False) if not simplified_focus.empty else "No rows.",
        "",
        "## Internal Bad-Day Attribution",
        "",
        internal_bad_day_summary.head(30).to_markdown(index=False) if not internal_bad_day_summary.empty else "No rows.",
        "",
        "## Simplified Bad-Day Attribution",
        "",
        simplified_bad_day_summary.head(30).to_markdown(index=False) if not simplified_bad_day_summary.empty else "No rows.",
        "",
        "## Phase-2 Loss Decomposition (NLL vs Path)",
        "",
        phase2_summary.head(30).to_markdown(index=False) if not phase2_summary.empty else "No rows.",
        "",
        f"## Phase-2 Bad-Date Drilldown ({str(focus_instrument).upper().strip()})",
        "",
        phase2_focus.head(40).to_markdown(index=False) if not phase2_focus.empty else "No rows.",
        "",
        "## Notes",
        "",
        "- `layer` = candidate mean vector injected before optimizer.",
        "- Replay ablation keeps covariance/L3 columns unchanged; only mean column differs.",
        "- `source_column` records the resolved column in bridge L3 CSV.",
        "- `nu_route_a` is sourced from `route_a_params_nu` when present. If missing, diagnostics fall back to sigma and signal-strength proxies.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest = {
        "scope": "route_a_mu_chain_reassessment_read_only",
        "config": {
            "bridge_l3_csv": str(bridge_l3_csv),
            "parameter_audit_csv": str(parameter_audit_csv),
            "output_dir": str(output_path),
            "start_date": start_date,
            "end_date": end_date,
            "max_weight": float(max_weight),
            "min_weight_floor": float(min_weight_floor),
            "transaction_cost_bps": float(transaction_cost_bps),
            "focus_instrument": str(focus_instrument).upper().strip(),
            "focus_start_date": focus_start_date,
            "focus_end_date": focus_end_date,
        },
        "layer_source_map": source_map,
        "outputs": {
            "layer_metrics": str(layer_summary_path),
            "layer_daily_metrics": str(layer_daily_path),
            "replay_summary": str(replay_summary_path),
            "replay_daily": str(replay_daily_path),
            "internal_layer_focus": str(internal_focus_path),
            "simplified_layer_focus": str(simplified_focus_path),
            "internal_bad_day_daily": str(internal_bad_day_daily_path),
            "internal_bad_day_summary": str(internal_bad_day_summary_path),
            "simplified_bad_day_daily": str(simplified_bad_day_daily_path),
            "simplified_bad_day_summary": str(simplified_bad_day_summary_path),
            "phase2_loss_detail": str(phase2_detail_path),
            "phase2_loss_summary": str(phase2_summary_path),
            "phase2_bad_date_drilldown": str(phase2_focus_path),
            "report": str(report_path),
        },
    }
    manifest_path.write_text(json.dumps(_json_ready(manifest), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[INFO] wrote {layer_summary_path}")
    print(f"[INFO] wrote {replay_summary_path}")
    print(f"[INFO] report {report_path}")
    return {
        "layer_metrics": layer_summary_path,
        "layer_daily_metrics": layer_daily_path,
        "replay_summary": replay_summary_path,
        "replay_daily": replay_daily_path,
        "internal_layer_focus": internal_focus_path,
        "simplified_layer_focus": simplified_focus_path,
        "internal_bad_day_daily": internal_bad_day_daily_path,
        "internal_bad_day_summary": internal_bad_day_summary_path,
        "simplified_bad_day_daily": simplified_bad_day_daily_path,
        "simplified_bad_day_summary": simplified_bad_day_summary_path,
        "phase2_loss_detail": phase2_detail_path,
        "phase2_loss_summary": phase2_summary_path,
        "phase2_bad_date_drilldown": phase2_focus_path,
        "report": report_path,
        "manifest": manifest_path,
    }


def main(argv: list[str] | None = None) -> dict[str, Path]:
    args = _parse_args(argv)
    return run_route_a_mu_chain_reassessment(
        bridge_l3_csv=args.bridge_l3_csv,
        parameter_audit_csv=args.parameter_audit_csv,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        max_weight=args.max_weight,
        min_weight_floor=args.min_weight_floor,
        transaction_cost_bps=args.transaction_cost_bps,
        focus_instrument=args.focus_instrument,
        focus_start_date=args.focus_start_date,
        focus_end_date=args.focus_end_date,
    )


if __name__ == "__main__":
    main()
