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

NEW_MODEL_SIGNAL_FEATURE_FLAG_DEFAULT = False
DEFAULT_BRIDGE_L3_CSV = "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/bridge_posterior_mu1d_softshrink_fullpanel_20260101_20260430_l3.csv"
DEFAULT_COEFFICIENTS_CSV = "archive/audits/20260504_route_a_q1_coefficients/route_a_q1_feature_coefficients.csv"
DEFAULT_OUTPUT_DIR = "archive/audits/20260504_new_model_signal_shadow_oos"
DEFAULT_PROVIDER_URI = "~/etf-daily-output/Documents/stock/data/qlib_data/all_fund_data/"
MOMENTUM_DECAY_WINDOW = 5
MOMENTUM_DECAY_THRESHOLD = 0.10
MOMENTUM_DECAY_FEATURE_NAME = "momentum_decay_5d_over_10pct"


@dataclass(frozen=True)
class ReplayConfig:
    start_date: str
    end_date: str
    warmup_start_date: str
    max_weight: float
    min_weight_floor: float
    provider_uri: str
    enable_new_model_shadow: bool


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
    peak = nav.cummax()
    drawdown = nav / peak - 1.0
    return float(drawdown.min())


def _sharpe(values: pd.Series) -> float:
    series = _numeric(values).dropna()
    if len(series) < 2:
        return float("nan")
    vol = float(series.std(ddof=1))
    if vol <= 1e-12:
        return float("nan")
    return float(series.mean() / vol * np.sqrt(252.0))


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in frame.columns) + " |")
    return "\n".join(lines)


def _load_coefficients(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"instrument", "feature", "final_mu_coefficient_after_shrink"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"coefficient CSV missing required columns: {missing}")
    frame["instrument"] = frame["instrument"].astype(str).str.upper()
    frame["feature"] = frame["feature"].astype(str)
    return frame


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
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"bridge L3 CSV missing required columns: {missing}")
    frame["instrument"] = frame["instrument"].astype(str).str.upper()
    frame = frame[(frame["datetime"] >= pd.Timestamp(start_date)) & (frame["datetime"] <= pd.Timestamp(end_date))].copy()
    frame = frame.sort_values(["datetime", "instrument"]).reset_index(drop=True)
    for column in ["next_return", "forecast_w", "forecast_weight_mean_used", "exposure_scale"]:
        frame[column] = _numeric(frame[column])
    return frame


def _load_decay_features_from_qlib(
    instruments: list[str],
    provider_uri: str,
    warmup_start_date: str,
    end_date: str,
) -> pd.DataFrame:
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    qlib.init(provider_uri=provider_uri, region=REG_CN)
    raw = D.features(instruments, fields=["$close"], start_time=warmup_start_date, end_time=end_date)
    frame = raw.reset_index()
    date_col = next((column for column in ["datetime", "date", "trade_date"] if column in frame.columns), None)
    code_col = next((column for column in ["instrument", "code", "symbol"] if column in frame.columns), None)
    if date_col is None or code_col is None:
        raise ValueError(f"Cannot infer qlib columns from {list(frame.columns)}")
    close_col = "$close" if "$close" in frame.columns else "close"
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame[code_col] = frame[code_col].astype(str).str.upper()
    frame = frame.sort_values([code_col, date_col]).copy()
    frame["return"] = frame.groupby(code_col)[close_col].pct_change(fill_method=None)
    frame["momentum_5d_compound"] = (
        (1.0 + frame["return"])
        .groupby(frame[code_col])
        .rolling(MOMENTUM_DECAY_WINDOW)
        .apply(np.prod, raw=True)
        .reset_index(level=0, drop=True)
        - 1.0
    )
    excess = (frame["momentum_5d_compound"] - MOMENTUM_DECAY_THRESHOLD).clip(lower=0.0)
    frame[MOMENTUM_DECAY_FEATURE_NAME] = -np.square(excess / MOMENTUM_DECAY_THRESHOLD)
    return frame[[code_col, date_col, MOMENTUM_DECAY_FEATURE_NAME, "momentum_5d_compound"]].rename(
        columns={code_col: "instrument", date_col: "datetime"}
    )


def _attach_new_model_mu(
    bridge_frame: pd.DataFrame,
    coefficients: pd.DataFrame,
    provider_uri: str,
    warmup_start_date: str,
    end_date: str,
    enable_new_model_shadow: bool,
) -> pd.DataFrame:
    working = bridge_frame.copy()
    instruments = sorted(working["instrument"].unique().tolist())
    decay_features = _load_decay_features_from_qlib(instruments, provider_uri, warmup_start_date, end_date)
    coeff = coefficients[coefficients["feature"] == MOMENTUM_DECAY_FEATURE_NAME][
        ["instrument", "final_mu_coefficient_after_shrink"]
    ].rename(columns={"final_mu_coefficient_after_shrink": "momentum_decay_final_coef"})
    working = working.merge(decay_features, on=["instrument", "datetime"], how="left")
    working = working.merge(coeff, on="instrument", how="left")
    working["momentum_decay_final_coef"] = _numeric(working["momentum_decay_final_coef"]).fillna(0.0)
    working[MOMENTUM_DECAY_FEATURE_NAME] = _numeric(working[MOMENTUM_DECAY_FEATURE_NAME]).fillna(0.0)
    working["new_model_mu_delta"] = working[MOMENTUM_DECAY_FEATURE_NAME] * working["momentum_decay_final_coef"]
    working["old_route_a_signal"] = _numeric(working["forecast_weight_mean_used"]).fillna(0.0)
    if enable_new_model_shadow:
        working["new_model_shadow_signal"] = working["old_route_a_signal"] + working["new_model_mu_delta"]
    else:
        working["new_model_shadow_signal"] = working["old_route_a_signal"]
    working["new_model_shadow_enabled"] = bool(enable_new_model_shadow)
    return working


def _parse_cov_matrix(day_frame: pd.DataFrame) -> pd.DataFrame:
    order = str(day_frame["forecast_cov_order"].dropna().iloc[0]).split("|")
    vector_by_instrument: dict[str, list[float]] = {}
    for _, row in day_frame.iterrows():
        text = str(row.get("forecast_cov_matrix_robust", ""))
        values = [float(value) for value in text.split(",") if value != ""]
        if len(values) == len(order):
            vector_by_instrument[str(row["instrument"])] = values
    rows = []
    valid_order = []
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
    config: ReplayConfig,
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


def run_shadow_oos(
    bridge_l3_csv: str | Path,
    coefficients_csv: str | Path,
    output_dir: str | Path,
    config: ReplayConfig,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    coefficients = _load_coefficients(coefficients_csv)
    bridge_frame = _load_bridge_frame(bridge_l3_csv, config.start_date, config.end_date)
    working = _attach_new_model_mu(
        bridge_frame,
        coefficients,
        config.provider_uri,
        config.warmup_start_date,
        config.end_date,
        config.enable_new_model_shadow,
    )

    previous_old = pd.Series(dtype=float)
    previous_new = pd.Series(dtype=float)
    daily_rows: list[dict[str, Any]] = []
    weight_frames: list[pd.DataFrame] = []

    for date_value, day_frame in working.groupby("datetime", sort=True):
        date_text = pd.Timestamp(date_value).strftime("%Y-%m-%d")
        old_weights, old_status = _optimize_day(day_frame, "old_route_a_signal", previous_old, config)
        new_weights, new_status = _optimize_day(day_frame, "new_model_shadow_signal", previous_new, config)
        sample = day_frame.set_index("instrument", drop=False).reindex(old_weights.index)
        next_return = _numeric(sample["next_return"]).fillna(0.0)
        exposure_scale = _numeric(sample["exposure_scale"]).fillna(1.0)
        published_w = _numeric(sample["forecast_w"]).fillna(0.0)
        old_effective = old_weights * exposure_scale
        new_effective = new_weights * exposure_scale
        published_effective = published_w * exposure_scale
        old_return = float((old_effective * next_return).sum())
        new_return = float((new_effective * next_return).sum())
        published_return = float((published_effective * next_return).sum())
        delta_w = new_weights - old_weights
        day_weights = pd.DataFrame(
            {
                "datetime": date_text,
                "instrument": old_weights.index,
                "old_mvo_weight": old_weights.to_numpy(),
                "new_model_shadow_mvo_weight": new_weights.reindex(old_weights.index).fillna(0.0).to_numpy(),
                "delta_mvo_weight": delta_w.to_numpy(),
                "old_effective_weight": old_effective.to_numpy(),
                "new_model_shadow_effective_weight": new_effective.reindex(old_weights.index).fillna(0.0).to_numpy(),
                "published_l3_weight": published_w.to_numpy(),
                "published_l3_effective_weight": published_effective.to_numpy(),
                "old_route_a_signal": sample["old_route_a_signal"].to_numpy(),
                "new_model_shadow_signal": sample["new_model_shadow_signal"].to_numpy(),
                "new_model_mu_delta": sample["new_model_mu_delta"].to_numpy(),
                MOMENTUM_DECAY_FEATURE_NAME: sample[MOMENTUM_DECAY_FEATURE_NAME].to_numpy(),
                "momentum_5d_compound": sample["momentum_5d_compound"].to_numpy(),
                "momentum_decay_final_coef": sample["momentum_decay_final_coef"].to_numpy(),
                "next_return": next_return.to_numpy(),
                "exposure_scale": exposure_scale.to_numpy(),
            }
        )
        weight_frames.append(day_weights)
        mu_delta_abs = _numeric(sample["new_model_mu_delta"]).abs()
        daily_rows.append(
            {
                "datetime": date_text,
                "new_model_shadow_enabled": bool(config.enable_new_model_shadow),
                "active_old_count": int((old_weights > 1e-12).sum()),
                "active_new_count": int((new_weights > 1e-12).sum()),
                "changed_weight_count": int((delta_w.abs() > 1e-8).sum()),
                "gross_abs_weight_delta": float(delta_w.abs().sum()),
                "max_abs_weight_delta": float(delta_w.abs().max()) if len(delta_w) else 0.0,
                "mu_delta_nonzero_count": int((mu_delta_abs > 1e-12).sum()),
                "mu_delta_min": float(_numeric(sample["new_model_mu_delta"]).min()),
                "mu_delta_mean_active_old": float(_numeric(sample.loc[old_weights > 1e-12, "new_model_mu_delta"]).mean()) if (old_weights > 1e-12).any() else np.nan,
                "old_mvo_effective_return": old_return,
                "new_model_shadow_effective_return": new_return,
                "delta_return_vs_old_mvo": new_return - old_return,
                "published_l3_effective_return": published_return,
                "delta_return_vs_published_l3": new_return - published_return,
                "old_status": old_status,
                "new_status": new_status,
                "top_weight_delta_codes": "|".join(delta_w.abs().sort_values(ascending=False).head(8).index.astype(str).tolist()),
            }
        )
        previous_old = old_weights.copy()
        previous_new = new_weights.copy()

    daily = pd.DataFrame(daily_rows)
    weights = pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    if not daily.empty:
        daily["old_mvo_nav_proxy"] = (1.0 + _numeric(daily["old_mvo_effective_return"]).fillna(0.0)).cumprod()
        daily["new_model_shadow_nav_proxy"] = (1.0 + _numeric(daily["new_model_shadow_effective_return"]).fillna(0.0)).cumprod()
        daily["published_l3_nav_proxy"] = (1.0 + _numeric(daily["published_l3_effective_return"]).fillna(0.0)).cumprod()
        daily["delta_nav_vs_old_mvo"] = daily["new_model_shadow_nav_proxy"] - daily["old_mvo_nav_proxy"]
        daily["decay_mu_active_day"] = _numeric(daily["mu_delta_nonzero_count"]).fillna(0.0) > 0.0

    focus_dates = ["2026-01-12", "2026-01-14", "2026-01-28", "2026-01-29", "2026-03-02", "2026-03-06"]
    focus = daily[daily.get("datetime", pd.Series(dtype=object)).astype(str).isin(focus_dates)].copy() if not daily.empty else pd.DataFrame()
    top_weight_delta = weights.reindex(weights["delta_mvo_weight"].abs().sort_values(ascending=False).index).head(80) if not weights.empty else pd.DataFrame()
    jan29_top_delta = pd.DataFrame()
    if not weights.empty:
        jan29_top_delta = weights[weights["datetime"].astype(str) == "2026-01-29"].copy()
        if not jan29_top_delta.empty:
            jan29_top_delta["abs_delta_mvo_weight"] = jan29_top_delta["delta_mvo_weight"].abs()
            jan29_top_delta = jan29_top_delta.sort_values("abs_delta_mvo_weight", ascending=False).head(20)
    active_day_mask = daily["decay_mu_active_day"] if "decay_mu_active_day" in daily.columns else pd.Series(dtype=bool)
    calm_day_mask = ~active_day_mask if len(active_day_mask) else pd.Series(dtype=bool)
    replay_vs_published = _numeric(daily["old_mvo_effective_return"]) - _numeric(daily["published_l3_effective_return"])
    replay_corr = _numeric(daily["old_mvo_effective_return"]).corr(_numeric(daily["published_l3_effective_return"])) if len(daily) >= 2 else np.nan

    summary = pd.DataFrame(
        [
            {
                "audit_scope": "shadow_only_no_live_weight_change",
                "new_model_feature_flag_default": NEW_MODEL_SIGNAL_FEATURE_FLAG_DEFAULT,
                "new_model_shadow_enabled_for_this_run": bool(config.enable_new_model_shadow),
                "bridge_l3_csv": str(bridge_l3_csv),
                "coefficients_csv": str(coefficients_csv),
                "start_date": config.start_date,
                "end_date": config.end_date,
                "days": int(daily.shape[0]),
                "old_mvo_cumulative_return": _compound_return(daily["old_mvo_effective_return"]),
                "new_model_shadow_cumulative_return": _compound_return(daily["new_model_shadow_effective_return"]),
                "delta_cumulative_return_vs_old_mvo": _compound_return(daily["new_model_shadow_effective_return"]) - _compound_return(daily["old_mvo_effective_return"]),
                "published_l3_cumulative_return": _compound_return(daily["published_l3_effective_return"]),
                "old_mvo_sharpe": _sharpe(daily["old_mvo_effective_return"]),
                "new_model_shadow_sharpe": _sharpe(daily["new_model_shadow_effective_return"]),
                "old_mvo_max_drawdown": _max_drawdown(daily["old_mvo_effective_return"]),
                "new_model_shadow_max_drawdown": _max_drawdown(daily["new_model_shadow_effective_return"]),
                "return_2026_01_29_old_mvo": float(daily.loc[daily["datetime"] == "2026-01-29", "old_mvo_effective_return"].iloc[0]) if "2026-01-29" in set(daily["datetime"]) else np.nan,
                "return_2026_01_29_new_model_shadow": float(daily.loc[daily["datetime"] == "2026-01-29", "new_model_shadow_effective_return"].iloc[0]) if "2026-01-29" in set(daily["datetime"]) else np.nan,
                "delta_return_2026_01_29": (
                    float(daily.loc[daily["datetime"] == "2026-01-29", "new_model_shadow_effective_return"].iloc[0])
                    - float(daily.loc[daily["datetime"] == "2026-01-29", "old_mvo_effective_return"].iloc[0])
                ) if "2026-01-29" in set(daily["datetime"]) else np.nan,
                "avg_gross_abs_weight_delta": float(_numeric(daily["gross_abs_weight_delta"]).mean()),
                "max_gross_abs_weight_delta": float(_numeric(daily["gross_abs_weight_delta"]).max()),
                "avg_delta_return_vs_old_mvo": float(_numeric(daily["delta_return_vs_old_mvo"]).mean()),
                "worst_delta_return_vs_old_mvo": float(_numeric(daily["delta_return_vs_old_mvo"]).min()),
                "decay_mu_active_days": int(active_day_mask.sum()) if len(active_day_mask) else 0,
                "decay_mu_calm_days": int(calm_day_mask.sum()) if len(calm_day_mask) else 0,
                "active_days_old_cumulative_return": _compound_return(daily.loc[active_day_mask, "old_mvo_effective_return"]) if len(active_day_mask) else np.nan,
                "active_days_new_cumulative_return": _compound_return(daily.loc[active_day_mask, "new_model_shadow_effective_return"]) if len(active_day_mask) else np.nan,
                "active_days_delta_cumulative_return": (
                    _compound_return(daily.loc[active_day_mask, "new_model_shadow_effective_return"])
                    - _compound_return(daily.loc[active_day_mask, "old_mvo_effective_return"])
                ) if len(active_day_mask) else np.nan,
                "calm_days_old_cumulative_return": _compound_return(daily.loc[calm_day_mask, "old_mvo_effective_return"]) if len(calm_day_mask) else np.nan,
                "calm_days_new_cumulative_return": _compound_return(daily.loc[calm_day_mask, "new_model_shadow_effective_return"]) if len(calm_day_mask) else np.nan,
                "calm_days_delta_cumulative_return": (
                    _compound_return(daily.loc[calm_day_mask, "new_model_shadow_effective_return"])
                    - _compound_return(daily.loc[calm_day_mask, "old_mvo_effective_return"])
                ) if len(calm_day_mask) else np.nan,
                "calm_days_avg_delta_return": float(_numeric(daily.loc[calm_day_mask, "delta_return_vs_old_mvo"]).mean()) if len(calm_day_mask) else np.nan,
                "calm_days_negative_delta_count": int((_numeric(daily.loc[calm_day_mask, "delta_return_vs_old_mvo"]) < 0.0).sum()) if len(calm_day_mask) else 0,
                "old_replay_vs_published_cumulative_delta": _compound_return(daily["old_mvo_effective_return"]) - _compound_return(daily["published_l3_effective_return"]),
                "old_replay_vs_published_daily_corr": float(replay_corr) if pd.notna(replay_corr) else np.nan,
                "old_replay_vs_published_max_abs_daily_diff": float(replay_vs_published.abs().max()),
            }
        ]
    )

    daily_path = output_path / "new_model_signal_shadow_oos_daily.csv"
    weights_path = output_path / "new_model_signal_shadow_oos_weights.csv"
    focus_path = output_path / "new_model_signal_shadow_oos_focus.csv"
    summary_path = output_path / "new_model_signal_shadow_oos_summary.csv"
    top_delta_path = output_path / "new_model_signal_shadow_oos_top_weight_delta.csv"
    jan29_top_delta_path = output_path / "new_model_signal_shadow_oos_20260129_top_weight_delta.csv"
    report_path = output_path / "new_model_signal_shadow_oos_report.md"
    manifest_path = output_path / "release_candidate_manifest.json"

    daily.to_csv(daily_path, index=False)
    weights.to_csv(weights_path, index=False)
    focus.to_csv(focus_path, index=False)
    summary.to_csv(summary_path, index=False)
    top_weight_delta.to_csv(top_delta_path, index=False)
    jan29_top_delta.to_csv(jan29_top_delta_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "scope": "shadow_only_no_live_weight_change",
                "new_model_feature_flag_default": NEW_MODEL_SIGNAL_FEATURE_FLAG_DEFAULT,
                "new_model_shadow_enabled_for_this_run": bool(config.enable_new_model_shadow),
                "bridge_l3_csv": str(bridge_l3_csv),
                "coefficients_csv": str(coefficients_csv),
                "outputs": {
                    "daily": str(daily_path),
                    "weights": str(weights_path),
                    "focus": str(focus_path),
                    "summary": str(summary_path),
                    "top_weight_delta": str(top_delta_path),
                    "jan29_top_weight_delta": str(jan29_top_delta_path),
                    "report": str(report_path),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "# New Model Signal Shadow OOS",
        "",
        "This audit replays 2026 Q1 with the current bridge/L3 artifact as the old Route A baseline and adds only the momentum-decay coefficient contribution from the Q1 coefficient table.",
        "",
        "## Summary",
        "",
        _markdown_table(summary),
        "",
        "## Focus Dates",
        "",
        _markdown_table(focus),
        "",
        "## 2026-01-29 Top Weight Deltas",
        "",
        _markdown_table(
            jan29_top_delta[
                [
                    "instrument",
                    "old_mvo_weight",
                    "new_model_shadow_mvo_weight",
                    "delta_mvo_weight",
                    "old_route_a_signal",
                    "new_model_shadow_signal",
                    "new_model_mu_delta",
                    MOMENTUM_DECAY_FEATURE_NAME,
                    "momentum_5d_compound",
                    "next_return",
                ]
            ].head(12)
            if not jan29_top_delta.empty
            else jan29_top_delta
        ),
        "",
        "## Interpretation Boundary",
        "",
        "The run is shadow-only. Production Route A remains default-off for the new feature; production bridge inputs continue to use the existing Route A output directory unless a future promotion gate changes that explicitly.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {
        "daily": daily_path,
        "weights": weights_path,
        "focus": focus_path,
        "summary": summary_path,
        "top_weight_delta": top_delta_path,
        "report": report_path,
        "manifest": manifest_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit-only Route A new-model signal shadow OOS replay.")
    parser.add_argument("--bridge-l3-csv", default=DEFAULT_BRIDGE_L3_CSV)
    parser.add_argument("--coefficients-csv", default=DEFAULT_COEFFICIENTS_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-03-31")
    parser.add_argument("--warmup-start-date", default="2025-11-01")
    parser.add_argument("--provider-uri", default=DEFAULT_PROVIDER_URI)
    parser.add_argument("--max-weight", type=float, default=bridge_core.DEFAULT_MAX_WEIGHT)
    parser.add_argument("--min-weight-floor", type=float, default=bridge_core.DEFAULT_MIN_WEIGHT_FLOOR)
    parser.add_argument(
        "--enable-new-model-shadow",
        action="store_true",
        default=NEW_MODEL_SIGNAL_FEATURE_FLAG_DEFAULT,
        help="Opt in to applying the new coefficient-derived mu delta. Default false for release-candidate safety.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ReplayConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        warmup_start_date=args.warmup_start_date,
        max_weight=float(args.max_weight),
        min_weight_floor=float(args.min_weight_floor),
        provider_uri=args.provider_uri,
        enable_new_model_shadow=bool(args.enable_new_model_shadow),
    )
    outputs = run_shadow_oos(args.bridge_l3_csv, args.coefficients_csv, args.output_dir, config)
    print(f"[INFO] wrote {outputs['summary']}")
    print(f"[INFO] report {outputs['report']}")


if __name__ == "__main__":
    main()
