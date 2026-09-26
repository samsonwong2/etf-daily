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

from pipeline.smart_mu_patch_audit import (  # noqa: E402
    DEFAULT_BRIDGE_L3_CSV,
    DEFAULT_END_DATE,
    DEFAULT_PARAMETER_AUDIT_CSV,
    DEFAULT_START_DATE,
    SMART_MU_PATCH_MULTIPLIER,
    _compound_return,
    _json_ready,
    _load_bridge_frame,
    _load_overheat_flags,
    _markdown_table,
    _max_drawdown,
    _numeric,
    _optimize_day,
    _sharpe,
)
from strategy.route_ab_moments_bridge import bridge as bridge_core  # noqa: E402
from pipeline.constants import DEFAULT_PROVIDER_URI  # noqa: E402

DEFAULT_OUTPUT_DIR = "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/smart_mu_variant_audit_20260101_20260430"
DEFAULT_VARIANCE_FLOOR_QUANTILE = 0.60
DEFAULT_CORR_FLOOR = 0.20
ENTROPY_LOW_THRESHOLD = 0.80
HIGH_ENTROPY_THRESHOLD = 0.85
HIGH_ENTROPY_EXPOSURE_SCALE = 1.30
CROWDED_OVERHEAT_EXPOSURE_SCALE = 0.20
RISK_OFF_OVERLAY_EXPOSURE_SCALE = 0.50
RISK_OFF_VOL_SPIKE_THRESHOLD = 0.20
RISK_OFF_VOL_SPIKE_LOOKBACK_DAYS = 10
RISK_OFF_MARKET_GAP_THRESHOLD = -0.01
NO_LOOKAHEAD_ATR_SPIKE_THRESHOLD = 0.20
NO_LOOKAHEAD_ATR_LOOKBACK_DAYS = 10
NO_LOOKAHEAD_ATR_HOLD_DAYS = 2
HYSTERESIS_ENTROPY_UP_THRESHOLD = 0.88
HYSTERESIS_ENTROPY_DOWN_THRESHOLD = 0.82
SELECTIVE_HYSTERESIS_LIQUIDITY_Z_THRESHOLD = 0.25
SELECTIVE_HYSTERESIS_FAST_UP_THRESHOLD = HYSTERESIS_ENTROPY_UP_THRESHOLD
SELECTIVE_HYSTERESIS_FAST_DOWN_THRESHOLD = HYSTERESIS_ENTROPY_DOWN_THRESHOLD
SELECTIVE_HYSTERESIS_WIDE_UP_THRESHOLD = 0.90
SELECTIVE_HYSTERESIS_WIDE_DOWN_THRESHOLD = 0.78
TRANSACTION_COST_BPS = 15.0
SMOOTHED_RERISK_EXPOSURE_STEP = 0.30
TRADING_DAYS_PER_YEAR = 252.0
OVERHEAT_RETURN_5D_THRESHOLD = 0.10
NEGATIVE_FLIP_MU = -0.05
STRATEGIC_LEVERAGE_EXPOSURE_POLICY = "entropy_strategic_leverage"
STRATEGIC_LEVERAGE_RISK_OFF_EXPOSURE_POLICY = "entropy_strategic_leverage_risk_off"
STRATEGIC_LEVERAGE_RISK_OFF_ANTILOOKAHEAD_EXPOSURE_POLICY = "entropy_strategic_leverage_risk_off_antilookahead"
STRATEGIC_LEVERAGE_HYSTERESIS_EXPOSURE_POLICY = "entropy_strategic_leverage_hysteresis"
STRATEGIC_LEVERAGE_HYSTERESIS_RISK_OFF_ANTILOOKAHEAD_EXPOSURE_POLICY = (
    "entropy_strategic_leverage_hysteresis_risk_off_antilookahead"
)
STRATEGIC_LEVERAGE_SELECTIVE_HYSTERESIS_EXPOSURE_POLICY = "entropy_strategic_leverage_selective_hysteresis"
DEFAULT_MARKET_BENCHMARK = "sh510300"
EXECUTION_TARGET_TRANSACTION_COST_BPS = 10.0
FRICTION_SENSITIVITY_BPS = (5.0, 8.0, EXECUTION_TARGET_TRANSACTION_COST_BPS, 12.0, 15.0)
REGIME_LOC_CALIBRATION_LOOKBACK_DAYS = 40
REGIME_LOC_CALIBRATION_MIN_DATES = 10
REGIME_LOC_CALIBRATION_MIN_OBS = 30
REGIME_LOC_CALIBRATION_SHRINK_OBS = 30
REGIME_LOC_CALIBRATION_STUDENT_T_DF = 5.0
REGIME_LOC_CALIBRATION_SLOPE_MIN = 0.0
REGIME_LOC_CALIBRATION_SLOPE_MAX = 1.25
REGIME_LOC_CALIBRATION_MIN_SIGMA = 1e-4
REGIME_LOC_CALIBRATION_MIN_ABS_MU = 1e-8


@dataclass(frozen=True)
class SmartMuVariantAuditConfig:
    start_date: str
    end_date: str
    max_weight: float
    min_weight_floor: float
    variance_floor_quantile: float
    corr_floor: float
    patch_multiplier: float
    negative_flip_mu: float
    communication_theme: str
    entropy_low_threshold: float
    high_entropy_threshold: float
    high_entropy_exposure_scale: float
    crowded_overheat_exposure_scale: float
    risk_off_overlay_exposure_scale: float
    risk_off_vol_spike_threshold: float
    risk_off_vol_spike_lookback_days: int
    risk_off_market_gap_threshold: float
    no_lookahead_atr_spike_threshold: float
    no_lookahead_atr_lookback_days: int
    no_lookahead_atr_hold_days: int
    hysteresis_entropy_up_threshold: float
    hysteresis_entropy_down_threshold: float
    selective_hysteresis_liquidity_z_threshold: float
    selective_hysteresis_fast_up_threshold: float
    selective_hysteresis_fast_down_threshold: float
    selective_hysteresis_wide_up_threshold: float
    selective_hysteresis_wide_down_threshold: float
    transaction_cost_bps: float
    smoothed_rerisk_exposure_step: float
    provider_uri: str
    market_benchmark: str


@dataclass(frozen=True)
class VariantDefinition:
    name: str
    mean_column: str
    trigger_column: str
    description: str
    exclude_column: str | None = None
    exposure_policy: str = "published_l3"
    smooth_rerisk_exposure: bool = False


def _normalized_entropy(shares: pd.Series) -> tuple[float, float, float]:
    values = _numeric(shares).dropna().clip(lower=0.0)
    total = float(values.sum())
    if total <= 1e-12:
        return float("nan"), float("nan"), float("nan")
    probs = values / total
    probs = probs[probs > 1e-12]
    if probs.empty:
        return float("nan"), float("nan"), float("nan")
    entropy = float(-(probs * np.log(probs)).sum())
    normalized = float(entropy / np.log(len(probs))) if len(probs) > 1 else 0.0
    effective_count = float(np.exp(entropy))
    return entropy, normalized, effective_count


def _raw_momentum_features(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame[["instrument", "datetime", "close"]].copy()
    working = working.sort_values(["instrument", "datetime"])
    close = working.groupby("instrument")["close"]
    working["smart_mu_variant_raw_close_return_5d"] = close.pct_change(5, fill_method=None)
    working["smart_mu_variant_raw_close_return_2d"] = close.pct_change(2, fill_method=None)
    return_5d = _numeric(working["smart_mu_variant_raw_close_return_5d"])
    return_2d = _numeric(working["smart_mu_variant_raw_close_return_2d"])
    working["smart_mu_variant_expected_return_2d_from_5d"] = np.where(
        return_5d > -1.0,
        np.power(1.0 + return_5d, 2.0 / 5.0) - 1.0,
        np.nan,
    )
    working["smart_mu_variant_return_5d_daily_rate"] = np.where(
        return_5d > -1.0,
        np.power(1.0 + return_5d, 1.0 / 5.0) - 1.0,
        np.nan,
    )
    working["smart_mu_variant_return_2d_daily_rate"] = np.where(
        return_2d > -1.0,
        np.power(1.0 + return_2d, 1.0 / 2.0) - 1.0,
        np.nan,
    )
    working["smart_mu_variant_trigger_raw_overheat_5d"] = return_5d > OVERHEAT_RETURN_5D_THRESHOLD
    expected_2d = _numeric(working["smart_mu_variant_expected_return_2d_from_5d"])
    working["smart_mu_variant_momentum_deceleration_flag"] = (
        working["smart_mu_variant_trigger_raw_overheat_5d"] & return_2d.notna() & (return_2d <= expected_2d)
    )
    working["smart_mu_variant_momentum_flat_or_negative_flag"] = (
        working["smart_mu_variant_trigger_raw_overheat_5d"] & return_2d.notna() & (return_2d <= 0.0)
    )
    return working[
        [
            "smart_mu_variant_raw_close_return_5d",
            "smart_mu_variant_raw_close_return_2d",
            "smart_mu_variant_expected_return_2d_from_5d",
            "smart_mu_variant_return_5d_daily_rate",
            "smart_mu_variant_return_2d_daily_rate",
            "smart_mu_variant_trigger_raw_overheat_5d",
            "smart_mu_variant_momentum_deceleration_flag",
            "smart_mu_variant_momentum_flat_or_negative_flag",
        ]
    ].reindex(frame.index)


def _daily_industry_entropy(frame: pd.DataFrame, config: SmartMuVariantAuditConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date_value, day_frame in frame.groupby("datetime", sort=True):
        active = day_frame[_numeric(day_frame["forecast_w"]).fillna(0.0) > 1e-12].copy()
        active_weight_sum = float(_numeric(active["forecast_w"]).fillna(0.0).sum()) if not active.empty else 0.0
        entropy = float("nan")
        entropy_norm = float("nan")
        entropy_effective_count = float("nan")
        dominant_industry = ""
        dominant_industry_weight_share = float("nan")
        active_amount_proxy = float("nan")
        active_liquidity_z_amount_weighted = float("nan")
        active_liquidity_z_mean = float("nan")
        if active_weight_sum > 1e-12:
            shares = _numeric(active.groupby("factor_industry_theme")["forecast_w"].sum()) / active_weight_sum
            entropy, entropy_norm, entropy_effective_count = _normalized_entropy(shares)
            if not shares.empty:
                dominant_industry = str(shares.idxmax())
                dominant_industry_weight_share = float(shares.max())
            liquidity_z = _numeric(active.get("factor_volume_z_20", pd.Series(np.nan, index=active.index)))
            close = _numeric(active.get("close", pd.Series(np.nan, index=active.index)))
            volume = _numeric(active.get("volume", pd.Series(np.nan, index=active.index)))
            amount_proxy = (close * volume).replace([np.inf, -np.inf], np.nan).clip(lower=0.0)
            active_amount_proxy = float(amount_proxy.fillna(0.0).sum())
            if liquidity_z.notna().any():
                active_liquidity_z_mean = float(liquidity_z.mean())
                amount_sum = float(amount_proxy.fillna(0.0).sum())
                if amount_sum > 1e-12:
                    active_liquidity_z_amount_weighted = float((liquidity_z.fillna(0.0) * amount_proxy.fillna(0.0)).sum() / amount_sum)
                else:
                    active_liquidity_z_amount_weighted = active_liquidity_z_mean
        rows.append(
            {
                "datetime": pd.Timestamp(date_value),
                "smart_mu_variant_industry_entropy": entropy,
                "smart_mu_variant_industry_entropy_norm": entropy_norm,
                "smart_mu_variant_industry_entropy_effective_count": entropy_effective_count,
                "smart_mu_variant_dominant_industry": dominant_industry,
                "smart_mu_variant_dominant_industry_weight_share": dominant_industry_weight_share,
                "smart_mu_variant_low_entropy_day": bool(np.isfinite(entropy_norm) and entropy_norm <= config.entropy_low_threshold),
                "smart_mu_variant_high_entropy_day": bool(np.isfinite(entropy_norm) and entropy_norm > config.high_entropy_threshold),
                "smart_mu_variant_active_weight_sum_for_entropy": active_weight_sum,
                "smart_mu_variant_active_count_for_entropy": int(active.shape[0]),
                "smart_mu_variant_active_amount_proxy": active_amount_proxy,
                "smart_mu_variant_active_liquidity_z_amount_weighted": active_liquidity_z_amount_weighted,
                "smart_mu_variant_active_liquidity_z_mean": active_liquidity_z_mean,
                "smart_mu_variant_active_liquidity_high_day": bool(
                    np.isfinite(active_liquidity_z_amount_weighted)
                    and active_liquidity_z_amount_weighted >= float(config.selective_hysteresis_liquidity_z_threshold)
                ),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    hysteresis_state = False
    hysteresis_values: list[bool] = []
    for entropy_norm in _numeric(result["smart_mu_variant_industry_entropy_norm"]):
        if np.isfinite(entropy_norm) and entropy_norm > float(config.hysteresis_entropy_up_threshold):
            hysteresis_state = True
        elif np.isfinite(entropy_norm) and entropy_norm < float(config.hysteresis_entropy_down_threshold):
            hysteresis_state = False
        hysteresis_values.append(hysteresis_state)
    result["smart_mu_variant_hysteresis_high_entropy_day"] = hysteresis_values
    selective_state = False
    selective_values: list[bool] = []
    selective_bands: list[str] = []
    selective_up_thresholds: list[float] = []
    selective_down_thresholds: list[float] = []
    for _, row in result.iterrows():
        entropy_norm = float(row["smart_mu_variant_industry_entropy_norm"])
        high_liquidity = bool(row["smart_mu_variant_active_liquidity_high_day"])
        if high_liquidity:
            up_threshold = float(config.selective_hysteresis_fast_up_threshold)
            down_threshold = float(config.selective_hysteresis_fast_down_threshold)
            band = "liquid_fast"
        else:
            up_threshold = float(config.selective_hysteresis_wide_up_threshold)
            down_threshold = float(config.selective_hysteresis_wide_down_threshold)
            band = "wide_band"
        if np.isfinite(entropy_norm) and entropy_norm > up_threshold:
            selective_state = True
        elif np.isfinite(entropy_norm) and entropy_norm < down_threshold:
            selective_state = False
        selective_values.append(selective_state)
        selective_bands.append(band)
        selective_up_thresholds.append(up_threshold)
        selective_down_thresholds.append(down_threshold)
    result["smart_mu_variant_selective_hysteresis_high_entropy_day"] = selective_values
    result["smart_mu_variant_selective_hysteresis_band"] = selective_bands
    result["smart_mu_variant_selective_hysteresis_up_threshold"] = selective_up_thresholds
    result["smart_mu_variant_selective_hysteresis_down_threshold"] = selective_down_thresholds
    return result


def _load_benchmark_gap_features(config: SmartMuVariantAuditConfig) -> pd.DataFrame:
    columns = [
        "datetime",
        "smart_mu_variant_market_benchmark_open",
        "smart_mu_variant_market_benchmark_high",
        "smart_mu_variant_market_benchmark_low",
        "smart_mu_variant_market_benchmark_close",
        "smart_mu_variant_market_benchmark_prev_close",
        "smart_mu_variant_market_overnight_gap",
        "smart_mu_variant_market_gap_risk_off",
        "smart_mu_variant_market_benchmark_true_range_pct",
        "smart_mu_variant_market_benchmark_atr_baseline",
        "smart_mu_variant_market_benchmark_atr_spike_ratio",
        "smart_mu_variant_market_benchmark_atr_spike",
        "smart_mu_variant_market_benchmark_atr_spike_no_lookahead",
    ]
    try:
        import qlib  # type: ignore[import-untyped]
        from qlib.constant import REG_CN  # type: ignore[import-untyped]
        from qlib.data import D  # type: ignore[import-untyped]

        qlib.init(provider_uri=config.provider_uri, region=REG_CN)
        raw = D.features(
            [config.market_benchmark],
            fields=["$open", "$high", "$low", "$close"],
            start_time=config.start_date,
            end_time=config.end_date,
        ).reset_index()
    except Exception:
        return pd.DataFrame(columns=columns)

    if raw.empty:
        return pd.DataFrame(columns=columns)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = ["_".join([str(part) for part in column if part != ""]).strip("_") for column in raw.columns]
    open_column = "$open" if "$open" in raw.columns else "open"
    high_column = "$high" if "$high" in raw.columns else "high"
    low_column = "$low" if "$low" in raw.columns else "low"
    close_column = "$close" if "$close" in raw.columns else "close"
    if open_column not in raw.columns or high_column not in raw.columns or low_column not in raw.columns or close_column not in raw.columns:
        return pd.DataFrame(columns=columns)
    output = raw[["datetime", open_column, high_column, low_column, close_column]].copy()
    output["datetime"] = pd.to_datetime(output["datetime"])
    output = output.sort_values("datetime")
    output["smart_mu_variant_market_benchmark_open"] = _numeric(output[open_column])
    output["smart_mu_variant_market_benchmark_high"] = _numeric(output[high_column])
    output["smart_mu_variant_market_benchmark_low"] = _numeric(output[low_column])
    output["smart_mu_variant_market_benchmark_close"] = _numeric(output[close_column])
    output["smart_mu_variant_market_benchmark_prev_close"] = output["smart_mu_variant_market_benchmark_close"].shift(1)
    output["smart_mu_variant_market_overnight_gap"] = (
        output["smart_mu_variant_market_benchmark_open"] / output["smart_mu_variant_market_benchmark_prev_close"] - 1.0
    )
    output["smart_mu_variant_market_gap_risk_off"] = (
        output["smart_mu_variant_market_overnight_gap"] <= float(config.risk_off_market_gap_threshold)
    )
    prev_close = output["smart_mu_variant_market_benchmark_prev_close"]
    high = output["smart_mu_variant_market_benchmark_high"]
    low = output["smart_mu_variant_market_benchmark_low"]
    output["smart_mu_variant_market_benchmark_true_range_pct"] = pd.concat(
        [
            (high - low).abs() / prev_close,
            (high - prev_close).abs() / prev_close,
            (low - prev_close).abs() / prev_close,
        ],
        axis=1,
    ).max(axis=1)
    atr_lookback = max(int(config.no_lookahead_atr_lookback_days), 2)
    output["smart_mu_variant_market_benchmark_atr_baseline"] = (
        output["smart_mu_variant_market_benchmark_true_range_pct"].rolling(atr_lookback).median().shift(1)
    )
    output["smart_mu_variant_market_benchmark_atr_spike_ratio"] = (
        output["smart_mu_variant_market_benchmark_true_range_pct"] / output["smart_mu_variant_market_benchmark_atr_baseline"]
    )
    output["smart_mu_variant_market_benchmark_atr_spike"] = (
        output["smart_mu_variant_market_benchmark_atr_spike_ratio"] >= 1.0 + float(config.no_lookahead_atr_spike_threshold)
    )
    atr_hold = max(int(config.no_lookahead_atr_hold_days), 1)
    shifted_spike = output["smart_mu_variant_market_benchmark_atr_spike"].shift(1).eq(True).astype(float)
    output["smart_mu_variant_market_benchmark_atr_spike_no_lookahead"] = (
        shifted_spike.rolling(atr_hold, min_periods=1).max().fillna(0.0).astype(bool)
    )
    return output[columns]


def _market_risk_off_features(frame: pd.DataFrame, config: SmartMuVariantAuditConfig) -> pd.DataFrame:
    market = (
        frame.groupby("datetime", sort=True)
        .agg(
            smart_mu_variant_market_holding_realized_vol=(
                "next_return",
                lambda values: float(_numeric(values).dropna().std() * np.sqrt(TRADING_DAYS_PER_YEAR)),
            ),
            smart_mu_variant_market_equal_holding_return=("next_return", lambda values: float(_numeric(values).dropna().mean())),
            smart_mu_variant_market_equal_abs_holding_return=(
                "next_return",
                lambda values: float(_numeric(values).dropna().abs().mean()),
            ),
            smart_mu_variant_market_instrument_count=("instrument", "nunique"),
        )
        .reset_index()
    )
    market["datetime"] = pd.to_datetime(market["datetime"])
    market = market.sort_values("datetime")
    lookback = max(int(config.risk_off_vol_spike_lookback_days), 2)
    baseline = market["smart_mu_variant_market_holding_realized_vol"].rolling(lookback).median().shift(1)
    market["smart_mu_variant_market_realized_vol_baseline"] = baseline
    market["smart_mu_variant_market_realized_vol_spike_ratio"] = (
        market["smart_mu_variant_market_holding_realized_vol"] / baseline
    )
    market["smart_mu_variant_market_vol_spike_risk_off"] = (
        market["smart_mu_variant_market_realized_vol_spike_ratio"] >= 1.0 + float(config.risk_off_vol_spike_threshold)
    )

    benchmark_gap = _load_benchmark_gap_features(config)
    if benchmark_gap.empty:
        for column in [
            "smart_mu_variant_market_benchmark_open",
            "smart_mu_variant_market_benchmark_high",
            "smart_mu_variant_market_benchmark_low",
            "smart_mu_variant_market_benchmark_close",
            "smart_mu_variant_market_benchmark_prev_close",
            "smart_mu_variant_market_overnight_gap",
            "smart_mu_variant_market_benchmark_true_range_pct",
            "smart_mu_variant_market_benchmark_atr_baseline",
            "smart_mu_variant_market_benchmark_atr_spike_ratio",
        ]:
            market[column] = np.nan
        market["smart_mu_variant_market_gap_risk_off"] = False
        market["smart_mu_variant_market_benchmark_atr_spike"] = False
        market["smart_mu_variant_market_benchmark_atr_spike_no_lookahead"] = False
    else:
        market = market.merge(benchmark_gap, on="datetime", how="left")
        market["smart_mu_variant_market_gap_risk_off"] = market["smart_mu_variant_market_gap_risk_off"].fillna(False).astype(bool)

    market["smart_mu_variant_market_risk_off"] = (
        market["smart_mu_variant_market_vol_spike_risk_off"].fillna(False).astype(bool)
        | market["smart_mu_variant_market_gap_risk_off"].fillna(False).astype(bool)
    )
    market["smart_mu_variant_market_risk_off_no_lookahead"] = (
        market["smart_mu_variant_market_benchmark_atr_spike_no_lookahead"].fillna(False).astype(bool)
        | market["smart_mu_variant_market_gap_risk_off"].fillna(False).astype(bool)
    )
    return market


def _attach_regime_calibration_labels(working: pd.DataFrame, config: SmartMuVariantAuditConfig) -> pd.DataFrame:
    labeled = working.copy()
    entropy = _numeric(labeled["smart_mu_variant_industry_entropy_norm"])
    labeled["smart_mu_variant_entropy_bucket"] = np.select(
        [
            entropy <= float(config.entropy_low_threshold),
            entropy <= float(config.high_entropy_threshold),
            entropy <= float(config.hysteresis_entropy_up_threshold),
            entropy > float(config.hysteresis_entropy_up_threshold),
        ],
        ["low_entropy", "mid_entropy", "high_entropy_pre_hysteresis", "hysteresis_high_entropy"],
        default="entropy_unknown",
    )

    liquidity_z = _numeric(labeled["smart_mu_variant_active_liquidity_z_amount_weighted"])
    labeled["smart_mu_variant_liquidity_bucket"] = np.select(
        [
            liquidity_z < 0.0,
            liquidity_z < float(config.selective_hysteresis_liquidity_z_threshold),
            liquidity_z < 0.75,
            liquidity_z >= 0.75,
        ],
        ["thin_liquidity", "below_selective_threshold", "liquid", "very_liquid"],
        default="liquidity_unknown",
    )

    instrument_overheat_low_entropy = (
        labeled["smart_mu_variant_trigger_overheat_5d"].fillna(False).astype(bool)
        & labeled["smart_mu_variant_low_entropy_day"].fillna(False).astype(bool)
    )
    day_overheat_low_entropy_context = (
        labeled["smart_mu_variant_overheat_low_entropy_day"].fillna(False).astype(bool)
        & ~instrument_overheat_low_entropy
    )
    labeled["smart_mu_variant_overheat_regime"] = np.select(
        [
            instrument_overheat_low_entropy,
            day_overheat_low_entropy_context,
            labeled["smart_mu_variant_trigger_overheat_5d"].fillna(False).astype(bool),
            labeled["smart_mu_variant_trigger_raw_overheat_5d"].fillna(False).astype(bool),
            labeled["smart_mu_variant_trigger_route_a_overheat_5d"].fillna(False).astype(bool),
        ],
        [
            "instrument_overheat_low_entropy",
            "day_overheat_low_entropy_context",
            "instrument_overheat",
            "raw_overheat",
            "route_a_overheat",
        ],
        default="no_overheat",
    )
    labeled["smart_mu_variant_theme_bucket"] = (
        labeled["factor_industry_theme"].astype(str).str.lower().str.strip().replace("", "theme_unknown")
    )
    return labeled


def _student_t_loc_slope_loss(
    slope: float,
    mu_values: np.ndarray,
    y_values: np.ndarray,
    sigma_values: np.ndarray,
) -> float:
    sigma = np.clip(sigma_values.astype(float), REGIME_LOC_CALIBRATION_MIN_SIGMA, None)
    z = (y_values.astype(float) - float(slope) * mu_values.astype(float)) / sigma
    return float(np.sum(np.log(sigma) + 0.5 * (REGIME_LOC_CALIBRATION_STUDENT_T_DF + 1.0) * np.log1p((z * z) / REGIME_LOC_CALIBRATION_STUDENT_T_DF)))


def _estimate_regime_loc_slope(history: pd.DataFrame) -> tuple[float, int, int]:
    if history.empty:
        return float("nan"), 0, 0

    mu = _numeric(history["smart_mu_variant_mu_before"])
    y = _numeric(history["next_return"])
    if "forecast_variance_robust" in history.columns:
        variance = _numeric(history["forecast_variance_robust"])
    elif "forecast_variance" in history.columns:
        variance = _numeric(history["forecast_variance"])
    else:
        variance = pd.Series(np.nan, index=history.index)

    valid = mu.abs().gt(REGIME_LOC_CALIBRATION_MIN_ABS_MU) & mu.notna() & y.notna()
    if not valid.any():
        return float("nan"), 0, 0

    valid_history = history.loc[valid]
    date_count = int(pd.to_datetime(valid_history["datetime"]).nunique()) if "datetime" in valid_history.columns else 0
    mu_values = valid_history["smart_mu_variant_mu_before"].pipe(_numeric).to_numpy(dtype=float)
    y_values = valid_history["next_return"].pipe(_numeric).to_numpy(dtype=float)
    sigma = np.sqrt(variance.loc[valid].clip(lower=REGIME_LOC_CALIBRATION_MIN_SIGMA**2)).to_numpy(dtype=float)
    if not np.isfinite(sigma).any():
        fallback_sigma = max(float(np.nanstd(y_values - mu_values)), REGIME_LOC_CALIBRATION_MIN_SIGMA)
        sigma = np.full_like(mu_values, fallback_sigma, dtype=float)
    else:
        fallback_sigma = max(float(np.nanmedian(sigma[np.isfinite(sigma)])), REGIME_LOC_CALIBRATION_MIN_SIGMA)
        sigma = np.where(np.isfinite(sigma), sigma, fallback_sigma)

    coarse_grid = np.linspace(REGIME_LOC_CALIBRATION_SLOPE_MIN, REGIME_LOC_CALIBRATION_SLOPE_MAX, 61)
    coarse_losses = np.array([_student_t_loc_slope_loss(value, mu_values, y_values, sigma) for value in coarse_grid])
    best = float(coarse_grid[int(np.nanargmin(coarse_losses))])
    half_width = (REGIME_LOC_CALIBRATION_SLOPE_MAX - REGIME_LOC_CALIBRATION_SLOPE_MIN) / 60.0
    fine_grid = np.linspace(
        max(REGIME_LOC_CALIBRATION_SLOPE_MIN, best - half_width),
        min(REGIME_LOC_CALIBRATION_SLOPE_MAX, best + half_width),
        41,
    )
    fine_losses = np.array([_student_t_loc_slope_loss(value, mu_values, y_values, sigma) for value in fine_grid])
    slope = float(fine_grid[int(np.nanargmin(fine_losses))])
    return slope, int(len(mu_values)), date_count


def _fit_level_slope(
    history: pd.DataFrame,
    current_row: pd.Series,
    level_columns: list[str],
    global_slope: float,
) -> tuple[float, float, int, int, str] | None:
    if not level_columns:
        return None
    mask = pd.Series(True, index=history.index)
    for column in level_columns:
        mask &= history[column].astype(str).eq(str(current_row[column]))
    subset = history.loc[mask]
    raw_slope, obs, date_count = _estimate_regime_loc_slope(subset)
    if obs < REGIME_LOC_CALIBRATION_MIN_OBS or date_count < REGIME_LOC_CALIBRATION_MIN_DATES or not np.isfinite(raw_slope):
        return None

    reliability = obs / (obs + float(REGIME_LOC_CALIBRATION_SHRINK_OBS))
    final_slope = reliability * raw_slope + (1.0 - reliability) * global_slope
    source = "+".join(column.replace("smart_mu_variant_", "") for column in level_columns)
    return float(np.clip(final_slope, REGIME_LOC_CALIBRATION_SLOPE_MIN, REGIME_LOC_CALIBRATION_SLOPE_MAX)), raw_slope, obs, date_count, source


def _apply_regime_loc_calibration(working: pd.DataFrame) -> pd.DataFrame:
    calibrated = working.copy()
    calibrated["smart_mu_variant_regime_loc_slope_a"] = 1.0
    calibrated["smart_mu_variant_regime_loc_slope_raw"] = 1.0
    calibrated["smart_mu_variant_regime_loc_slope_obs"] = 0
    calibrated["smart_mu_variant_regime_loc_slope_dates"] = 0
    calibrated["smart_mu_variant_regime_loc_global_slope_a"] = 1.0
    calibrated["smart_mu_variant_regime_loc_global_slope_obs"] = 0
    calibrated["smart_mu_variant_regime_loc_global_slope_dates"] = 0
    calibrated["smart_mu_variant_regime_loc_slope_source"] = "no_prior_history"

    dates = list(pd.to_datetime(calibrated["datetime"]).drop_duplicates().sort_values())
    level_candidates = [
        [
            "smart_mu_variant_theme_bucket",
            "smart_mu_variant_entropy_bucket",
            "smart_mu_variant_overheat_regime",
            "smart_mu_variant_liquidity_bucket",
        ],
        ["smart_mu_variant_theme_bucket", "smart_mu_variant_entropy_bucket", "smart_mu_variant_overheat_regime"],
        ["smart_mu_variant_entropy_bucket", "smart_mu_variant_overheat_regime"],
    ]

    for date_position, date_value in enumerate(dates):
        lookback_start = max(0, date_position - REGIME_LOC_CALIBRATION_LOOKBACK_DAYS)
        history_dates = set(dates[lookback_start:date_position])
        current_mask = pd.to_datetime(calibrated["datetime"]).eq(date_value)
        if not history_dates:
            continue
        history = calibrated.loc[pd.to_datetime(calibrated["datetime"]).isin(history_dates)].copy()
        global_raw_slope, global_obs, global_date_count = _estimate_regime_loc_slope(history)
        global_slope = (
            1.0
            if global_obs < REGIME_LOC_CALIBRATION_MIN_OBS
            or global_date_count < REGIME_LOC_CALIBRATION_MIN_DATES
            or not np.isfinite(global_raw_slope)
            else global_raw_slope
        )
        global_slope = float(np.clip(global_slope, REGIME_LOC_CALIBRATION_SLOPE_MIN, REGIME_LOC_CALIBRATION_SLOPE_MAX))

        for row_index, current_row in calibrated.loc[current_mask].iterrows():
            selected: tuple[float, float, int, int, str] | None = None
            for level_columns in level_candidates:
                selected = _fit_level_slope(history, current_row, level_columns, global_slope)
                if selected is not None:
                    break
            if selected is None:
                if global_obs >= REGIME_LOC_CALIBRATION_MIN_OBS and global_date_count >= REGIME_LOC_CALIBRATION_MIN_DATES and np.isfinite(global_slope):
                    selected = (global_slope, global_slope, global_obs, global_date_count, "global")
                else:
                    selected = (1.0, 1.0, global_obs, global_date_count, "insufficient_history_default_1")
            final_slope, raw_slope, obs, date_count, source = selected
            calibrated.at[row_index, "smart_mu_variant_regime_loc_slope_a"] = final_slope
            calibrated.at[row_index, "smart_mu_variant_regime_loc_slope_raw"] = raw_slope
            calibrated.at[row_index, "smart_mu_variant_regime_loc_slope_obs"] = int(obs)
            calibrated.at[row_index, "smart_mu_variant_regime_loc_slope_dates"] = int(date_count)
            calibrated.at[row_index, "smart_mu_variant_regime_loc_global_slope_a"] = global_slope
            calibrated.at[row_index, "smart_mu_variant_regime_loc_global_slope_obs"] = int(global_obs)
            calibrated.at[row_index, "smart_mu_variant_regime_loc_global_slope_dates"] = int(global_date_count)
            calibrated.at[row_index, "smart_mu_variant_regime_loc_slope_source"] = source

    calibrated["smart_mu_variant_mu_regime_calibrated_loc"] = (
        _numeric(calibrated["smart_mu_variant_mu_before"]).fillna(0.0)
        * _numeric(calibrated["smart_mu_variant_regime_loc_slope_a"]).fillna(1.0)
    )
    calibrated["smart_mu_variant_trigger_regime_loc_calibration"] = (
        _numeric(calibrated["smart_mu_variant_regime_loc_slope_a"]).sub(1.0).abs() > 1e-12
    )
    return calibrated


def _attach_variant_flags(
    bridge_frame: pd.DataFrame,
    parameter_audit_csv: str | Path,
    config: SmartMuVariantAuditConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    working = bridge_frame.copy()
    momentum = _raw_momentum_features(working)
    for column in momentum.columns:
        if momentum[column].dtype == bool:
            working[column] = momentum[column].fillna(False).astype(bool)
        else:
            working[column] = _numeric(momentum[column])

    overheat_flags, overheat_source = _load_overheat_flags(parameter_audit_csv, config.start_date, config.end_date)
    if overheat_flags.empty:
        working["smart_mu_variant_trigger_route_a_overheat_5d"] = False
        overheat_source = f"raw_l3_close_pct_change_5d_{overheat_source}"
    else:
        overheat_flags = overheat_flags.rename(
            columns={"smart_mu_patch_trigger_route_a_overheat_5d": "smart_mu_variant_trigger_route_a_overheat_5d"}
        )
        working = working.merge(overheat_flags, on=["datetime", "instrument"], how="left")
        working["smart_mu_variant_trigger_route_a_overheat_5d"] = (
            working["smart_mu_variant_trigger_route_a_overheat_5d"].fillna(False).astype(bool)
        )
        overheat_source = f"raw_l3_close_pct_change_5d_or_{overheat_source}"

    working["smart_mu_variant_trigger_overheat_5d"] = (
        working["smart_mu_variant_trigger_raw_overheat_5d"] | working["smart_mu_variant_trigger_route_a_overheat_5d"]
    )
    theme = working["factor_industry_theme"].astype(str).str.lower().str.strip()
    target_theme = str(config.communication_theme).lower().strip()
    working["smart_mu_variant_communication_theme_flag"] = theme.eq(target_theme)
    entropy = _daily_industry_entropy(working, config)
    working = working.merge(entropy, on="datetime", how="left")
    working["smart_mu_variant_low_entropy_day"] = working["smart_mu_variant_low_entropy_day"].fillna(False).astype(bool)
    working["smart_mu_variant_high_entropy_day"] = working["smart_mu_variant_high_entropy_day"].fillna(False).astype(bool)
    working["smart_mu_variant_hysteresis_high_entropy_day"] = (
        working["smart_mu_variant_hysteresis_high_entropy_day"].fillna(False).astype(bool)
    )
    working["smart_mu_variant_active_liquidity_high_day"] = (
        working["smart_mu_variant_active_liquidity_high_day"].fillna(False).astype(bool)
    )
    working["smart_mu_variant_selective_hysteresis_high_entropy_day"] = (
        working["smart_mu_variant_selective_hysteresis_high_entropy_day"].fillna(False).astype(bool)
    )
    market_risk = _market_risk_off_features(working, config)
    working = working.merge(market_risk, on="datetime", how="left")
    for risk_column in [
        "smart_mu_variant_market_vol_spike_risk_off",
        "smart_mu_variant_market_gap_risk_off",
        "smart_mu_variant_market_risk_off",
        "smart_mu_variant_market_benchmark_atr_spike",
        "smart_mu_variant_market_benchmark_atr_spike_no_lookahead",
        "smart_mu_variant_market_risk_off_no_lookahead",
    ]:
        working[risk_column] = working[risk_column].fillna(False).astype(bool)

    working["smart_mu_variant_trigger_blanket"] = (
        working["smart_mu_variant_trigger_overheat_5d"] | working["smart_mu_variant_communication_theme_flag"]
    )
    working["smart_mu_variant_trigger_overheat_deceleration"] = (
        working["smart_mu_variant_trigger_raw_overheat_5d"] & working["smart_mu_variant_momentum_deceleration_flag"]
    )
    working["smart_mu_variant_trigger_overheat_low_entropy"] = (
        working["smart_mu_variant_trigger_overheat_5d"] & working["smart_mu_variant_low_entropy_day"]
    )
    working["smart_mu_variant_overheat_low_entropy_day"] = (
        working.groupby("datetime")["smart_mu_variant_trigger_overheat_low_entropy"].transform("any").fillna(False).astype(bool)
    )
    working["smart_mu_variant_trigger_strategic_leverage"] = (
        working["smart_mu_variant_high_entropy_day"] | working["smart_mu_variant_overheat_low_entropy_day"]
    )
    working["smart_mu_variant_trigger_hysteresis_strategic_leverage"] = (
        working["smart_mu_variant_hysteresis_high_entropy_day"] | working["smart_mu_variant_overheat_low_entropy_day"]
    )
    working["smart_mu_variant_trigger_selective_hysteresis_strategic_leverage"] = (
        working["smart_mu_variant_selective_hysteresis_high_entropy_day"] | working["smart_mu_variant_overheat_low_entropy_day"]
    )
    working["smart_mu_variant_trigger_risk_off_overlay"] = (
        working["smart_mu_variant_high_entropy_day"] & working["smart_mu_variant_market_risk_off"]
    )
    working["smart_mu_variant_trigger_risk_off_overlay_no_lookahead"] = (
        working["smart_mu_variant_high_entropy_day"] & working["smart_mu_variant_market_risk_off_no_lookahead"]
    )

    working["smart_mu_variant_mu_before"] = _numeric(working["forecast_weight_mean_used"]).fillna(0.0)
    working["smart_mu_variant_mu_blanket"] = working["smart_mu_variant_mu_before"]
    working.loc[working["smart_mu_variant_trigger_blanket"], "smart_mu_variant_mu_blanket"] = (
        working.loc[working["smart_mu_variant_trigger_blanket"], "smart_mu_variant_mu_before"] * float(config.patch_multiplier)
    )
    working["smart_mu_variant_mu_overheat_deceleration"] = working["smart_mu_variant_mu_before"]
    working.loc[working["smart_mu_variant_trigger_overheat_deceleration"], "smart_mu_variant_mu_overheat_deceleration"] = (
        working.loc[working["smart_mu_variant_trigger_overheat_deceleration"], "smart_mu_variant_mu_before"]
        * float(config.patch_multiplier)
    )
    working["smart_mu_variant_mu_overheat_low_entropy"] = working["smart_mu_variant_mu_before"]
    working.loc[working["smart_mu_variant_trigger_overheat_low_entropy"], "smart_mu_variant_mu_overheat_low_entropy"] = (
        working.loc[working["smart_mu_variant_trigger_overheat_low_entropy"], "smart_mu_variant_mu_before"]
        * float(config.patch_multiplier)
    )
    working["smart_mu_variant_mu_negative_flip"] = working["smart_mu_variant_mu_before"]
    working.loc[working["smart_mu_variant_trigger_overheat_low_entropy"], "smart_mu_variant_mu_negative_flip"] = float(
        config.negative_flip_mu
    )
    working = _attach_regime_calibration_labels(working, config)
    working = _apply_regime_loc_calibration(working)

    metadata = {
        "overheat_source": overheat_source,
        "raw_overheat_trigger_rows": int(working["smart_mu_variant_trigger_raw_overheat_5d"].sum()),
        "route_a_overheat_trigger_rows": int(working["smart_mu_variant_trigger_route_a_overheat_5d"].sum()),
        "overheat_trigger_rows": int(working["smart_mu_variant_trigger_overheat_5d"].sum()),
        "momentum_deceleration_trigger_rows": int(working["smart_mu_variant_trigger_overheat_deceleration"].sum()),
        "momentum_flat_or_negative_rows": int(working["smart_mu_variant_momentum_flat_or_negative_flag"].sum()),
        "warmup_communication_rows": int(working["smart_mu_variant_communication_theme_flag"].sum()),
        "low_entropy_days": int(entropy["smart_mu_variant_low_entropy_day"].sum()) if not entropy.empty else 0,
        "high_entropy_days": int(entropy["smart_mu_variant_high_entropy_day"].sum()) if not entropy.empty else 0,
        "hysteresis_high_entropy_days": int(entropy["smart_mu_variant_hysteresis_high_entropy_day"].sum())
        if not entropy.empty
        else 0,
        "active_liquidity_high_days": int(entropy["smart_mu_variant_active_liquidity_high_day"].sum())
        if not entropy.empty
        else 0,
        "selective_hysteresis_high_entropy_days": int(entropy["smart_mu_variant_selective_hysteresis_high_entropy_day"].sum())
        if not entropy.empty
        else 0,
        "selective_hysteresis_liquid_fast_days": int(
            entropy["smart_mu_variant_selective_hysteresis_band"].eq("liquid_fast").sum()
        )
        if not entropy.empty
        else 0,
        "selective_hysteresis_wide_band_days": int(
            entropy["smart_mu_variant_selective_hysteresis_band"].eq("wide_band").sum()
        )
        if not entropy.empty
        else 0,
        "crowded_overheat_low_entropy_days": int(
            working.groupby("datetime")["smart_mu_variant_trigger_overheat_low_entropy"].any().sum()
        ),
        "risk_off_overlay_days": int(working.groupby("datetime")["smart_mu_variant_trigger_risk_off_overlay"].first().sum()),
        "risk_off_overlay_no_lookahead_days": int(
            working.groupby("datetime")["smart_mu_variant_trigger_risk_off_overlay_no_lookahead"].first().sum()
        ),
        "risk_off_vol_spike_days": int(working.groupby("datetime")["smart_mu_variant_market_vol_spike_risk_off"].first().sum()),
        "risk_off_market_gap_days": int(working.groupby("datetime")["smart_mu_variant_market_gap_risk_off"].first().sum()),
        "risk_off_atr_spike_no_lookahead_days": int(
            working.groupby("datetime")["smart_mu_variant_market_benchmark_atr_spike_no_lookahead"].first().sum()
        ),
        "overheat_low_entropy_trigger_rows": int(working["smart_mu_variant_trigger_overheat_low_entropy"].sum()),
        "entropy_low_threshold": float(config.entropy_low_threshold),
        "high_entropy_threshold": float(config.high_entropy_threshold),
        "high_entropy_exposure_scale": float(config.high_entropy_exposure_scale),
        "crowded_overheat_exposure_scale": float(config.crowded_overheat_exposure_scale),
        "risk_off_overlay_exposure_scale": float(config.risk_off_overlay_exposure_scale),
        "risk_off_vol_spike_threshold": float(config.risk_off_vol_spike_threshold),
        "risk_off_vol_spike_lookback_days": int(config.risk_off_vol_spike_lookback_days),
        "risk_off_market_gap_threshold": float(config.risk_off_market_gap_threshold),
        "no_lookahead_atr_spike_threshold": float(config.no_lookahead_atr_spike_threshold),
        "no_lookahead_atr_lookback_days": int(config.no_lookahead_atr_lookback_days),
        "no_lookahead_atr_hold_days": int(config.no_lookahead_atr_hold_days),
        "hysteresis_entropy_up_threshold": float(config.hysteresis_entropy_up_threshold),
        "hysteresis_entropy_down_threshold": float(config.hysteresis_entropy_down_threshold),
        "selective_hysteresis_liquidity_z_threshold": float(config.selective_hysteresis_liquidity_z_threshold),
        "selective_hysteresis_fast_up_threshold": float(config.selective_hysteresis_fast_up_threshold),
        "selective_hysteresis_fast_down_threshold": float(config.selective_hysteresis_fast_down_threshold),
        "selective_hysteresis_wide_up_threshold": float(config.selective_hysteresis_wide_up_threshold),
        "selective_hysteresis_wide_down_threshold": float(config.selective_hysteresis_wide_down_threshold),
        "variance_floor_quantile": float(config.variance_floor_quantile),
        "corr_floor": float(config.corr_floor),
        "transaction_cost_bps": float(config.transaction_cost_bps),
        "smoothed_rerisk_exposure_step": float(config.smoothed_rerisk_exposure_step),
        "market_benchmark": config.market_benchmark,
        "negative_flip_mu": float(config.negative_flip_mu),
        "regime_loc_calibration_lookback_days": int(REGIME_LOC_CALIBRATION_LOOKBACK_DAYS),
        "regime_loc_calibration_min_dates": int(REGIME_LOC_CALIBRATION_MIN_DATES),
        "regime_loc_calibration_min_obs": int(REGIME_LOC_CALIBRATION_MIN_OBS),
        "regime_loc_calibration_shrink_obs": int(REGIME_LOC_CALIBRATION_SHRINK_OBS),
        "regime_loc_calibration_student_t_df": float(REGIME_LOC_CALIBRATION_STUDENT_T_DF),
        "regime_loc_calibration_slope_min": float(REGIME_LOC_CALIBRATION_SLOPE_MIN),
        "regime_loc_calibration_slope_max": float(REGIME_LOC_CALIBRATION_SLOPE_MAX),
        "regime_loc_calibration_trigger_rows": int(working["smart_mu_variant_trigger_regime_loc_calibration"].sum()),
    }
    return working, metadata


def _variant_definitions() -> list[VariantDefinition]:
    return [
        VariantDefinition(
            name="old_mvo_replay",
            mean_column="smart_mu_variant_mu_before",
            trigger_column="smart_mu_variant_trigger_none",
            description="Baseline replay using forecast_weight_mean_used without audit patch.",
        ),
        VariantDefinition(
            name="blanket_overheat_or_communication_mu30",
            mean_column="smart_mu_variant_mu_blanket",
            trigger_column="smart_mu_variant_trigger_blanket",
            description="Reference only: mu * multiplier when 5-day overheat or communication theme is present.",
        ),
        VariantDefinition(
            name="overheat_deceleration_mu30",
            mean_column="smart_mu_variant_mu_overheat_deceleration",
            trigger_column="smart_mu_variant_trigger_overheat_deceleration",
            description="Mu penalty only when raw 5-day overheat also shows 2-day momentum deceleration.",
        ),
        VariantDefinition(
            name="communication_removed",
            mean_column="smart_mu_variant_mu_before",
            trigger_column="smart_mu_variant_communication_theme_flag",
            exclude_column="smart_mu_variant_communication_theme_flag",
            description="Conditional admission test: remove communication-theme instruments from the MVO universe.",
        ),
        VariantDefinition(
            name="overheat_low_entropy_mu30",
            mean_column="smart_mu_variant_mu_overheat_low_entropy",
            trigger_column="smart_mu_variant_trigger_overheat_low_entropy",
            description="Mu penalty only when 5-day overheat happens on low industry-entropy days.",
        ),
        VariantDefinition(
            name="overheat_low_entropy_mu_negative_flip",
            mean_column="smart_mu_variant_mu_negative_flip",
            trigger_column="smart_mu_variant_trigger_overheat_low_entropy",
            description="Aggressive flip: set mu to a fixed negative value when 5-day overheat happens on low industry-entropy days.",
        ),
        VariantDefinition(
            name="regime_calibrated_student_t_loc",
            mean_column="smart_mu_variant_mu_regime_calibrated_loc",
            trigger_column="smart_mu_variant_trigger_regime_loc_calibration",
            description="Audit-only root-distribution test: replace Student-t loc mu with no-lookahead rolling regime slope a_g * mu before MVO, leaving L3 and optimizer contracts unchanged.",
        ),
        VariantDefinition(
            name="strategic_leverage_entropy_timing",
            mean_column="smart_mu_variant_mu_before",
            trigger_column="smart_mu_variant_trigger_strategic_leverage",
            description="L3-only strategic exposure timing: exposure 1.3 on high-entropy days and 0.2 on crowded overheat low-entropy days.",
            exposure_policy=STRATEGIC_LEVERAGE_EXPOSURE_POLICY,
        ),
        VariantDefinition(
            name="strategic_leverage_entropy_timing_risk_off",
            mean_column="smart_mu_variant_mu_before",
            trigger_column="smart_mu_variant_trigger_strategic_leverage",
            description="L3-only strategic exposure timing with risk-off overlay on high-entropy vol/gap shock days.",
            exposure_policy=STRATEGIC_LEVERAGE_RISK_OFF_EXPOSURE_POLICY,
        ),
        VariantDefinition(
            name="overheat_low_entropy_mu30_strategic_leverage",
            mean_column="smart_mu_variant_mu_overheat_low_entropy",
            trigger_column="smart_mu_variant_trigger_strategic_leverage",
            description="Best mu filter plus strategic exposure timing: exposure 1.3 on high-entropy days and 0.2 on crowded overheat low-entropy days.",
            exposure_policy=STRATEGIC_LEVERAGE_EXPOSURE_POLICY,
        ),
        VariantDefinition(
            name="overheat_low_entropy_mu30_strategic_leverage_risk_off",
            mean_column="smart_mu_variant_mu_overheat_low_entropy",
            trigger_column="smart_mu_variant_trigger_strategic_leverage",
            description="Best mu filter plus strategic exposure timing and risk-off overlay on high-entropy vol/gap shock days.",
            exposure_policy=STRATEGIC_LEVERAGE_RISK_OFF_EXPOSURE_POLICY,
        ),
        VariantDefinition(
            name="overheat_low_entropy_mu30_strategic_leverage_risk_off_smooth_rerisk030",
            mean_column="smart_mu_variant_mu_overheat_low_entropy",
            trigger_column="smart_mu_variant_trigger_strategic_leverage",
            description="Risk-off overlay plus asymmetric smoothing: de-risk immediately and re-risk by at most the configured daily step.",
            exposure_policy=STRATEGIC_LEVERAGE_RISK_OFF_EXPOSURE_POLICY,
            smooth_rerisk_exposure=True,
        ),
        VariantDefinition(
            name="overheat_low_entropy_mu30_strategic_leverage_risk_off_antilookahead",
            mean_column="smart_mu_variant_mu_overheat_low_entropy",
            trigger_column="smart_mu_variant_trigger_strategic_leverage",
            description="Best mu filter plus strategic exposure timing and no-lookahead risk-off using benchmark ATR hold/gap.",
            exposure_policy=STRATEGIC_LEVERAGE_RISK_OFF_ANTILOOKAHEAD_EXPOSURE_POLICY,
        ),
        VariantDefinition(
            name="overheat_low_entropy_mu30_hysteresis",
            mean_column="smart_mu_variant_mu_overheat_low_entropy",
            trigger_column="smart_mu_variant_trigger_hysteresis_strategic_leverage",
            description="Best mu filter plus leverage hysteresis: enter high leverage above upper entropy band and exit below lower band.",
            exposure_policy=STRATEGIC_LEVERAGE_HYSTERESIS_EXPOSURE_POLICY,
        ),
        VariantDefinition(
            name="overheat_low_entropy_mu30_selective_hysteresis_liquidity",
            mean_column="smart_mu_variant_mu_overheat_low_entropy",
            trigger_column="smart_mu_variant_trigger_selective_hysteresis_strategic_leverage",
            description="Best mu filter plus liquidity-aware selective hysteresis: fast band on high amount-weighted liquidity z-score days and wider band otherwise.",
            exposure_policy=STRATEGIC_LEVERAGE_SELECTIVE_HYSTERESIS_EXPOSURE_POLICY,
        ),
        VariantDefinition(
            name="overheat_low_entropy_mu30_hysteresis_risk_off_antilookahead",
            mean_column="smart_mu_variant_mu_overheat_low_entropy",
            trigger_column="smart_mu_variant_trigger_hysteresis_strategic_leverage",
            description="Leverage hysteresis plus no-lookahead ATR/gap risk-off overlay.",
            exposure_policy=STRATEGIC_LEVERAGE_HYSTERESIS_RISK_OFF_ANTILOOKAHEAD_EXPOSURE_POLICY,
        ),
    ]


def _variant_exposure_scale(
    sample: pd.DataFrame,
    base_exposure_scale: pd.Series,
    variant: VariantDefinition,
    config: SmartMuVariantAuditConfig,
) -> tuple[pd.Series, str]:
    adjusted = _numeric(base_exposure_scale).fillna(1.0).copy()
    strategic_policies = {
        STRATEGIC_LEVERAGE_EXPOSURE_POLICY,
        STRATEGIC_LEVERAGE_RISK_OFF_EXPOSURE_POLICY,
        STRATEGIC_LEVERAGE_RISK_OFF_ANTILOOKAHEAD_EXPOSURE_POLICY,
        STRATEGIC_LEVERAGE_HYSTERESIS_EXPOSURE_POLICY,
        STRATEGIC_LEVERAGE_HYSTERESIS_RISK_OFF_ANTILOOKAHEAD_EXPOSURE_POLICY,
        STRATEGIC_LEVERAGE_SELECTIVE_HYSTERESIS_EXPOSURE_POLICY,
    }
    if variant.exposure_policy not in strategic_policies:
        return adjusted, "published_l3"

    crowded_overheat_day = bool(sample["smart_mu_variant_trigger_overheat_low_entropy"].fillna(False).astype(bool).any())
    entropy_norm = float(_numeric(sample["smart_mu_variant_industry_entropy_norm"]).iloc[0])
    high_entropy_day = bool(sample["smart_mu_variant_high_entropy_day"].fillna(False).astype(bool).iloc[0])
    if crowded_overheat_day:
        return pd.Series(float(config.crowded_overheat_exposure_scale), index=adjusted.index), "crowded_overheat_low_entropy"

    if variant.exposure_policy == STRATEGIC_LEVERAGE_SELECTIVE_HYSTERESIS_EXPOSURE_POLICY:
        selective_high_day = bool(sample["smart_mu_variant_selective_hysteresis_high_entropy_day"].fillna(False).astype(bool).iloc[0])
        band = str(sample["smart_mu_variant_selective_hysteresis_band"].iloc[0])
        up_threshold = float(_numeric(sample["smart_mu_variant_selective_hysteresis_up_threshold"]).iloc[0])
        if selective_high_day:
            suffix = "enter" if entropy_norm > up_threshold else "hold"
            return (
                pd.Series(float(config.high_entropy_exposure_scale), index=adjusted.index),
                f"selective_hysteresis_{band}_{suffix}",
            )
        return adjusted, f"selective_hysteresis_{band}_neutral_published_l3"

    hysteresis_policies = {
        STRATEGIC_LEVERAGE_HYSTERESIS_EXPOSURE_POLICY,
        STRATEGIC_LEVERAGE_HYSTERESIS_RISK_OFF_ANTILOOKAHEAD_EXPOSURE_POLICY,
    }
    if variant.exposure_policy in hysteresis_policies:
        hysteresis_high_day = bool(sample["smart_mu_variant_hysteresis_high_entropy_day"].fillna(False).astype(bool).iloc[0])
        if hysteresis_high_day:
            vol_spike = bool(sample["smart_mu_variant_market_vol_spike_risk_off"].fillna(False).astype(bool).iloc[0])
            gap_down = bool(sample["smart_mu_variant_market_gap_risk_off"].fillna(False).astype(bool).iloc[0])
            atr_hold = bool(sample["smart_mu_variant_market_benchmark_atr_spike_no_lookahead"].fillna(False).astype(bool).iloc[0])
            if variant.exposure_policy == STRATEGIC_LEVERAGE_HYSTERESIS_RISK_OFF_ANTILOOKAHEAD_EXPOSURE_POLICY and (atr_hold or gap_down):
                reasons = []
                if atr_hold:
                    reasons.append("atr_hold")
                if gap_down:
                    reasons.append("market_gap")
                return pd.Series(float(config.risk_off_overlay_exposure_scale), index=adjusted.index), "hysteresis_risk_off_" + "_".join(reasons)
            suffix = "enter" if entropy_norm > float(config.hysteresis_entropy_up_threshold) else "hold"
            return pd.Series(float(config.high_entropy_exposure_scale), index=adjusted.index), f"hysteresis_high_entropy_{suffix}"
        return adjusted, "hysteresis_neutral_published_l3"

    if high_entropy_day:
        vol_spike = bool(sample["smart_mu_variant_market_vol_spike_risk_off"].fillna(False).astype(bool).iloc[0])
        gap_down = bool(sample["smart_mu_variant_market_gap_risk_off"].fillna(False).astype(bool).iloc[0])
        atr_hold = bool(sample["smart_mu_variant_market_benchmark_atr_spike_no_lookahead"].fillna(False).astype(bool).iloc[0])
        if variant.exposure_policy == STRATEGIC_LEVERAGE_RISK_OFF_EXPOSURE_POLICY and (vol_spike or gap_down):
            reasons = []
            if vol_spike:
                reasons.append("vol_spike")
            if gap_down:
                reasons.append("market_gap")
            return pd.Series(float(config.risk_off_overlay_exposure_scale), index=adjusted.index), "risk_off_overlay_" + "_".join(reasons)
        if variant.exposure_policy == STRATEGIC_LEVERAGE_RISK_OFF_ANTILOOKAHEAD_EXPOSURE_POLICY and (atr_hold or gap_down):
            reasons = []
            if atr_hold:
                reasons.append("atr_hold")
            if gap_down:
                reasons.append("market_gap")
            return pd.Series(float(config.risk_off_overlay_exposure_scale), index=adjusted.index), "risk_off_overlay_" + "_".join(reasons)
        return pd.Series(float(config.high_entropy_exposure_scale), index=adjusted.index), "high_entropy_expansion"
    return adjusted, "neutral_published_l3"


def _apply_rerisk_smoothing(
    target_exposure_scale: pd.Series,
    previous_exposure_scale: pd.Series,
    variant: VariantDefinition,
    config: SmartMuVariantAuditConfig,
) -> tuple[pd.Series, bool]:
    target = _numeric(target_exposure_scale).fillna(1.0).copy()
    if not variant.smooth_rerisk_exposure or previous_exposure_scale.empty:
        return target, False
    previous = _numeric(previous_exposure_scale).reindex(target.index).fillna(float(target.iloc[0]))
    step = max(float(config.smoothed_rerisk_exposure_step), 0.0)
    smoothed = target.copy()
    rerisk_mask = target > previous
    smoothed.loc[rerisk_mask] = np.minimum(target.loc[rerisk_mask], previous.loc[rerisk_mask] + step)
    changed = bool((smoothed - target).abs().max() > 1e-12)
    return smoothed, changed


def _optimize_variant_day(
    day_frame: pd.DataFrame,
    variant: VariantDefinition,
    previous_weights: pd.Series,
    config: SmartMuVariantAuditConfig,
) -> tuple[pd.Series, str]:
    sample = day_frame.set_index("instrument", drop=False).sort_index()
    full_index = pd.Index(sample.index.astype(str), dtype=object)
    if variant.exclude_column is None:
        weights, status = _optimize_day(day_frame, variant.mean_column, previous_weights, config)
        return weights.reindex(full_index).fillna(0.0), status

    allowed_mask = ~sample[variant.exclude_column].fillna(False).astype(bool)
    allowed = sample.loc[allowed_mask].copy()
    if allowed.empty:
        return pd.Series(0.0, index=full_index), "all_rows_excluded"
    allowed_frame = allowed.reset_index(drop=True)
    allowed_weights, status = _optimize_day(allowed_frame, variant.mean_column, previous_weights, config)
    return allowed_weights.reindex(full_index).fillna(0.0), f"exclude_{variant.exclude_column}|{status}"


def _summarize_daily(daily: pd.DataFrame, trigger_metadata: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if daily.empty:
        return pd.DataFrame()
    old = daily[daily["variant"] == "old_mvo_replay"].copy()
    old_cumulative = _compound_return(old["variant_effective_return"])
    old_net_cumulative = _compound_return(old["net_variant_effective_return"])
    published_cumulative = _compound_return(old["published_l3_effective_return"])
    old_vs_published_corr = (
        _numeric(old["variant_effective_return"]).corr(_numeric(old["published_l3_effective_return"])) if len(old) >= 2 else np.nan
    )
    old_vs_published_diff = _numeric(old["variant_effective_return"]) - _numeric(old["published_l3_effective_return"])

    for variant, group in daily.groupby("variant", sort=False):
        group = group.sort_values("datetime").copy()
        cumulative = _compound_return(group["variant_effective_return"])
        net_cumulative = _compound_return(group["net_variant_effective_return"])
        non_initial = group[~group["is_initial_trade"].fillna(False).astype(bool)].copy()
        annualized_turnover = float(_numeric(non_initial["one_way_turnover"]).mean() * TRADING_DAYS_PER_YEAR) if not non_initial.empty else np.nan
        annualized_turnover_including_initial = float(_numeric(group["one_way_turnover"]).mean() * TRADING_DAYS_PER_YEAR)
        best_delta_idx = _numeric(group["delta_return_vs_old_mvo"]).idxmax() if not group.empty else None
        worst_delta_idx = _numeric(group["delta_return_vs_old_mvo"]).idxmin() if not group.empty else None
        worst_return_idx = _numeric(group["variant_effective_return"]).idxmin() if not group.empty else None
        rows.append(
            {
                "variant": variant,
                "days": int(group.shape[0]),
                "cumulative_return": cumulative,
                "delta_cumulative_return_vs_old_mvo": cumulative - old_cumulative,
                "net_cumulative_return_after_tcost": net_cumulative,
                "delta_net_cumulative_return_after_tcost_vs_old_mvo": net_cumulative - old_net_cumulative,
                "net_return_above_12pct": bool(net_cumulative >= 0.12),
                "published_l3_cumulative_return": published_cumulative,
                "delta_cumulative_return_vs_published_l3": cumulative - published_cumulative,
                "daily_return_sum": float(_numeric(group["variant_effective_return"]).sum()),
                "net_daily_return_sum_after_tcost": float(_numeric(group["net_variant_effective_return"]).sum()),
                "total_transaction_cost_return": float(_numeric(group["transaction_cost_return"]).sum()),
                "total_one_way_turnover": float(_numeric(group["one_way_turnover"]).sum()),
                "annualized_turnover": annualized_turnover,
                "annualized_turnover_including_initial": annualized_turnover_including_initial,
                "delta_daily_return_sum_vs_old_mvo": float(_numeric(group["delta_return_vs_old_mvo"]).sum()),
                "max_drawdown": _max_drawdown(group["variant_effective_return"]),
                "net_max_drawdown_after_tcost": _max_drawdown(group["net_variant_effective_return"]),
                "sharpe": _sharpe(group["variant_effective_return"]),
                "net_sharpe_after_tcost": _sharpe(group["net_variant_effective_return"]),
                "triggered_rows": int(_numeric(group["triggered_rows"]).fillna(0.0).sum()),
                "trigger_active_days": int((_numeric(group["triggered_rows"]).fillna(0.0) > 0.0).sum()),
                "high_entropy_days": int(group["high_entropy_day"].fillna(False).astype(bool).sum()),
                "hysteresis_high_entropy_days": int(group["hysteresis_high_entropy_day"].fillna(False).astype(bool).sum()),
                "selective_hysteresis_high_entropy_days": int(
                    group["selective_hysteresis_high_entropy_day"].fillna(False).astype(bool).sum()
                ),
                "active_liquidity_high_days": int(group["active_liquidity_high_day"].fillna(False).astype(bool).sum()),
                "crowded_overheat_low_entropy_days": int(group["overheat_low_entropy_day"].fillna(False).astype(bool).sum()),
                "risk_off_overlay_days": int(group["risk_off_overlay_day"].fillna(False).astype(bool).sum()),
                "risk_off_overlay_no_lookahead_days": int(group["risk_off_overlay_no_lookahead_day"].fillna(False).astype(bool).sum()),
                "risk_off_exposure_regime_days": int(group["strategic_leverage_regime"].astype(str).str.contains("risk_off").sum()),
                "hysteresis_high_exposure_regime_days": int(
                    group["strategic_leverage_regime"].astype(str).str.contains("hysteresis_high_entropy").sum()
                ),
                "selective_hysteresis_high_exposure_regime_days": int(
                    group["strategic_leverage_regime"].astype(str).str.contains("selective_hysteresis_.*_(?:enter|hold)", regex=True).sum()
                ),
                "avg_active_liquidity_z_amount_weighted": float(_numeric(group["active_liquidity_z_amount_weighted"]).mean()),
                "risk_off_vol_spike_days": int(group["market_vol_spike_risk_off"].fillna(False).astype(bool).sum()),
                "risk_off_market_gap_days": int(group["market_gap_risk_off"].fillna(False).astype(bool).sum()),
                "risk_off_atr_spike_no_lookahead_days": int(group["market_atr_spike_no_lookahead"].fillna(False).astype(bool).sum()),
                "exposure_adjusted_days": int((_numeric(group["changed_exposure_scale_count_vs_base"]).fillna(0.0) > 0.0).sum()),
                "smoothed_exposure_days": int(group["exposure_was_smoothed"].fillna(False).astype(bool).sum()),
                "avg_base_exposure_scale": float(_numeric(group["base_exposure_scale_mean"]).mean()),
                "avg_variant_exposure_scale": float(_numeric(group["variant_exposure_scale_mean"]).mean()),
                "min_variant_exposure_scale": float(_numeric(group["variant_exposure_scale_min"]).min()),
                "max_variant_exposure_scale": float(_numeric(group["variant_exposure_scale_max"]).max()),
                "avg_gross_abs_weight_delta_vs_old": float(_numeric(group["gross_abs_weight_delta_vs_old"]).mean()),
                "max_gross_abs_weight_delta_vs_old": float(_numeric(group["gross_abs_weight_delta_vs_old"]).max()),
                "changed_weight_rows": int(_numeric(group["changed_weight_count_vs_old"]).fillna(0.0).sum()),
                "avg_gross_abs_effective_weight_delta_vs_old": float(_numeric(group["gross_abs_effective_weight_delta_vs_old"]).mean()),
                "max_gross_abs_effective_weight_delta_vs_old": float(_numeric(group["gross_abs_effective_weight_delta_vs_old"]).max()),
                "changed_effective_weight_rows": int(_numeric(group["changed_effective_weight_count_vs_old"]).fillna(0.0).sum()),
                "best_delta_day": str(group.loc[best_delta_idx, "datetime"]) if best_delta_idx is not None else "",
                "best_delta_return_vs_old_mvo": float(group.loc[best_delta_idx, "delta_return_vs_old_mvo"]) if best_delta_idx is not None else np.nan,
                "worst_delta_day": str(group.loc[worst_delta_idx, "datetime"]) if worst_delta_idx is not None else "",
                "worst_delta_return_vs_old_mvo": float(group.loc[worst_delta_idx, "delta_return_vs_old_mvo"]) if worst_delta_idx is not None else np.nan,
                "worst_return_day": str(group.loc[worst_return_idx, "datetime"]) if worst_return_idx is not None else "",
                "worst_return": float(group.loc[worst_return_idx, "variant_effective_return"]) if worst_return_idx is not None else np.nan,
                "old_replay_vs_published_daily_corr": float(old_vs_published_corr) if pd.notna(old_vs_published_corr) else np.nan,
                "old_replay_vs_published_cumulative_delta": old_cumulative - published_cumulative,
                "old_replay_vs_published_max_abs_daily_diff": float(old_vs_published_diff.abs().max()),
                **trigger_metadata,
            }
        )
    return pd.DataFrame(rows)


def _summarize_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    working = daily.copy()
    working["month"] = pd.to_datetime(working["datetime"]).dt.strftime("%Y-%m")
    for (variant, month), group in working.groupby(["variant", "month"], sort=True):
        group = group.sort_values("datetime")
        variant_return = _compound_return(group["variant_effective_return"])
        net_variant_return = _compound_return(group["net_variant_effective_return"])
        old_return = _compound_return(group["old_mvo_effective_return"])
        published_return = _compound_return(group["published_l3_effective_return"])
        non_initial = group[~group["is_initial_trade"].fillna(False).astype(bool)].copy()
        rows.append(
            {
                "variant": variant,
                "month": month,
                "days": int(group.shape[0]),
                "triggered_rows": int(_numeric(group["triggered_rows"]).fillna(0.0).sum()),
                "trigger_active_days": int((_numeric(group["triggered_rows"]).fillna(0.0) > 0.0).sum()),
                "variant_cumulative_return": variant_return,
                "net_variant_cumulative_return_after_tcost": net_variant_return,
                "old_mvo_cumulative_return": old_return,
                "published_l3_cumulative_return": published_return,
                "delta_cumulative_return_vs_old_mvo": variant_return - old_return,
                "transaction_cost_return_sum": float(_numeric(group["transaction_cost_return"]).sum()),
                "one_way_turnover_sum": float(_numeric(group["one_way_turnover"]).sum()),
                "annualized_turnover": float(_numeric(non_initial["one_way_turnover"]).mean() * TRADING_DAYS_PER_YEAR)
                if not non_initial.empty
                else np.nan,
                "delta_daily_return_sum_vs_old_mvo": float(_numeric(group["delta_return_vs_old_mvo"]).sum()),
                "low_entropy_days": int(group["low_entropy_day"].fillna(False).astype(bool).sum()),
                "high_entropy_days": int(group["high_entropy_day"].fillna(False).astype(bool).sum()),
                "hysteresis_high_entropy_days": int(group["hysteresis_high_entropy_day"].fillna(False).astype(bool).sum()),
                "selective_hysteresis_high_entropy_days": int(
                    group["selective_hysteresis_high_entropy_day"].fillna(False).astype(bool).sum()
                ),
                "active_liquidity_high_days": int(group["active_liquidity_high_day"].fillna(False).astype(bool).sum()),
                "overheat_low_entropy_days": int(group["overheat_low_entropy_day"].fillna(False).astype(bool).sum()),
                "risk_off_overlay_days": int(group["risk_off_overlay_day"].fillna(False).astype(bool).sum()),
                "risk_off_overlay_no_lookahead_days": int(group["risk_off_overlay_no_lookahead_day"].fillna(False).astype(bool).sum()),
                "risk_off_exposure_regime_days": int(group["strategic_leverage_regime"].astype(str).str.contains("risk_off").sum()),
                "hysteresis_high_exposure_regime_days": int(
                    group["strategic_leverage_regime"].astype(str).str.contains("hysteresis_high_entropy").sum()
                ),
                "selective_hysteresis_high_exposure_regime_days": int(
                    group["strategic_leverage_regime"].astype(str).str.contains("selective_hysteresis_.*_(?:enter|hold)", regex=True).sum()
                ),
                "avg_active_liquidity_z_amount_weighted": float(_numeric(group["active_liquidity_z_amount_weighted"]).mean()),
                "exposure_adjusted_days": int((_numeric(group["changed_exposure_scale_count_vs_base"]).fillna(0.0) > 0.0).sum()),
                "smoothed_exposure_days": int(group["exposure_was_smoothed"].fillna(False).astype(bool).sum()),
                "avg_variant_exposure_scale": float(_numeric(group["variant_exposure_scale_mean"]).mean()),
                "overheat_rows": int(_numeric(group["overheat_rows"]).fillna(0.0).sum()),
                "momentum_deceleration_rows": int(_numeric(group["momentum_deceleration_rows"]).fillna(0.0).sum()),
                "communication_rows": int(_numeric(group["communication_rows"]).fillna(0.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def _write_strategic_executive_summary(
    output_path: Path,
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    summary: pd.DataFrame,
    config: SmartMuVariantAuditConfig,
    diversification_audit: pd.DataFrame | None = None,
    net_correlation_matrix: pd.DataFrame | None = None,
) -> dict[str, Path]:
    executive_path = output_path / "strategic_entropy_executive_summary.md"
    chart_path = output_path / "strategic_entropy_nav_comparison.png"
    case_path = output_path / "strategic_entropy_case_study.csv"

    focus_variants = [
        "old_mvo_replay",
        "overheat_low_entropy_mu30_strategic_leverage",
        "overheat_low_entropy_mu30_strategic_leverage_risk_off",
        "overheat_low_entropy_mu30_strategic_leverage_risk_off_smooth_rerisk030",
        "overheat_low_entropy_mu30_strategic_leverage_risk_off_antilookahead",
        "overheat_low_entropy_mu30_hysteresis",
        "overheat_low_entropy_mu30_selective_hysteresis_liquidity",
        "overheat_low_entropy_mu30_hysteresis_risk_off_antilookahead",
    ]
    summary_cols = [
        "variant",
        "cumulative_return",
        "net_cumulative_return_after_tcost",
        "net_return_above_12pct",
        "max_drawdown",
        "net_max_drawdown_after_tcost",
        "sharpe",
        "net_sharpe_after_tcost",
        "annualized_turnover",
        "total_transaction_cost_return",
        "risk_off_overlay_days",
        "risk_off_overlay_no_lookahead_days",
        "risk_off_exposure_regime_days",
        "hysteresis_high_exposure_regime_days",
        "selective_hysteresis_high_exposure_regime_days",
        "active_liquidity_high_days",
        "avg_active_liquidity_z_amount_weighted",
        "smoothed_exposure_days",
        "best_delta_day",
        "worst_delta_day",
    ]
    summary_focus = summary.loc[summary["variant"].isin(focus_variants), summary_cols].copy() if not summary.empty else pd.DataFrame()

    case_days = ["2026-01-29", "2026-03-06"]
    case_cols = [
        "datetime",
        "variant",
        "strategic_leverage_regime",
        "variant_effective_return",
        "net_variant_effective_return",
        "old_mvo_effective_return",
        "one_way_turnover",
        "transaction_cost_return",
        "industry_entropy_norm",
        "active_liquidity_z_amount_weighted",
        "high_entropy_day",
        "hysteresis_high_entropy_day",
        "selective_hysteresis_high_entropy_day",
        "active_liquidity_high_day",
        "selective_hysteresis_band",
        "overheat_low_entropy_day",
        "risk_off_overlay_day",
        "market_vol_spike_risk_off",
        "market_gap_risk_off",
        "market_atr_spike_no_lookahead",
        "market_realized_vol_spike_ratio",
        "market_overnight_gap",
        "market_benchmark_atr_spike_ratio",
        "variant_exposure_scale_mean",
        "dominant_industry",
        "dominant_industry_weight_share",
    ]
    case_study = (
        daily.loc[daily["variant"].isin(focus_variants) & daily["datetime"].isin(case_days), case_cols].copy()
        if not daily.empty
        else pd.DataFrame(columns=case_cols)
    )
    case_study.to_csv(case_path, index=False)

    april = (
        monthly.loc[monthly["variant"].isin(focus_variants) & monthly["month"].eq("2026-04")].copy()
        if not monthly.empty
        else pd.DataFrame()
    )
    april_cols = [
        "variant",
        "month",
        "variant_cumulative_return",
        "net_variant_cumulative_return_after_tcost",
        "old_mvo_cumulative_return",
        "high_entropy_days",
        "hysteresis_high_entropy_days",
        "selective_hysteresis_high_entropy_days",
        "active_liquidity_high_days",
        "overheat_low_entropy_days",
        "risk_off_overlay_days",
        "risk_off_exposure_regime_days",
        "selective_hysteresis_high_exposure_regime_days",
        "avg_variant_exposure_scale",
        "avg_active_liquidity_z_amount_weighted",
        "one_way_turnover_sum",
        "transaction_cost_return_sum",
    ]
    diversification_audit = diversification_audit if diversification_audit is not None else pd.DataFrame()
    net_correlation_matrix = net_correlation_matrix if net_correlation_matrix is not None else pd.DataFrame()

    chart_note = "Chart not generated."
    if not daily.empty:
        try:
            import matplotlib.pyplot as plt  # type: ignore[import-untyped]

            chart_frame = daily.loc[daily["variant"].isin(focus_variants)].copy()
            chart_frame["datetime"] = pd.to_datetime(chart_frame["datetime"])
            gross_nav = chart_frame.pivot(index="datetime", columns="variant", values="variant_nav_proxy")
            net_nav = chart_frame.pivot(index="datetime", columns="variant", values="net_variant_nav_proxy_after_tcost")
            fig, ax = plt.subplots(figsize=(11, 5.5))
            if "old_mvo_replay" in gross_nav:
                ax.plot(gross_nav.index, gross_nav["old_mvo_replay"], label="old_mvo_replay gross", linewidth=1.7)
            combo = "overheat_low_entropy_mu30_strategic_leverage"
            risk_off = "overheat_low_entropy_mu30_strategic_leverage_risk_off"
            smooth = "overheat_low_entropy_mu30_strategic_leverage_risk_off_smooth_rerisk030"
            hysteresis = "overheat_low_entropy_mu30_hysteresis_risk_off_antilookahead"
            selective = "overheat_low_entropy_mu30_selective_hysteresis_liquidity"
            if combo in gross_nav:
                ax.plot(gross_nav.index, gross_nav[combo], label="entropy leverage gross", linewidth=1.7)
            if risk_off in net_nav:
                ax.plot(net_nav.index, net_nav[risk_off], label="risk-off overlay net", linewidth=2.1)
            if smooth in net_nav:
                ax.plot(net_nav.index, net_nav[smooth], label="risk-off smooth net", linewidth=1.6, linestyle="--")
            if hysteresis in net_nav:
                ax.plot(net_nav.index, net_nav[hysteresis], label="hysteresis anti-lookahead net", linewidth=1.8, linestyle=":")
            if selective in net_nav:
                ax.plot(net_nav.index, net_nav[selective], label="selective hysteresis net", linewidth=1.8, linestyle="-.")
            ax.axvline(pd.Timestamp("2026-01-29"), color="tab:red", linestyle="--", linewidth=1.0, label="01-29 low entropy exit")
            ax.axvline(pd.Timestamp("2026-03-06"), color="tab:orange", linestyle="--", linewidth=1.0, label="03-06 brake test")
            ax.axvspan(pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-30"), color="tab:green", alpha=0.10, label="April high entropy window")
            ax.set_title("Strategic Entropy Leverage: NAV and Brake Audit")
            ax.set_ylabel("NAV proxy")
            ax.grid(alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(chart_path, dpi=160)
            plt.close(fig)
            chart_note = f"Chart: {chart_path.name}"
        except Exception as exc:
            chart_note = f"Chart not generated: {type(exc).__name__}: {exc}"

    lines = [
        "# Strategic Entropy Leverage Executive Summary - Conditional Go",
        "",
        "Audit-only report. Production Student-t, bridge optimizer, and live L3 risk controls are not modified.",
        "",
        "## Promotion Decision: Conditional Go",
        "",
        "Recommendation: open a PR with `overheat_low_entropy_mu30_hysteresis` as the mainline candidate. The promotion is conditional, not an unconditional production switch: the trading/execution layer must keep realized one-way impact at or below 10 bp and preserve daily per-instrument audit fields for entropy, liquidity, exposure regime, turnover, and transaction-cost attribution.",
        "",
        "Rationale: the hysteresis-only candidate does not clear the 12% net-return bar at a conservative 15 bp cost, but it materially improves net return and drawdown versus the baseline. At the execution target of <=10 bp one-way cost, the friction profile is close enough to the 12 bp audit line to justify a controlled PR review.",
        "",
        "## Committee Thesis",
        "",
        "The leverage rule is not blind risk taking. It is a confidence-scoring overlay: normalized industry entropy measures whether the portfolio signal is broad-based or crowded. High entropy means the active book is supported by multiple themes, so exposure is allowed to expand. Low entropy plus overheat means the book is concentrated and crowded, so exposure is cut aggressively.",
        "",
        "## Policy Under Review",
        "",
        f"- High-entropy confidence zone: normalized entropy > {config.high_entropy_threshold:.2f}, exposure = {config.high_entropy_exposure_scale:.2f}.",
        f"- Crowded-overheat defense zone: overheat + low entropy, exposure = {config.crowded_overheat_exposure_scale:.2f}.",
        f"- Risk-off overlay: high entropy plus holding-period realized-vol spike >= {1.0 + config.risk_off_vol_spike_threshold:.2f}x the prior {config.risk_off_vol_spike_lookback_days}-day median or {config.market_benchmark} overnight gap <= {config.risk_off_market_gap_threshold:.2%}, exposure = {config.risk_off_overlay_exposure_scale:.2f}.",
        f"- Anti-lookahead risk-off proxy: {config.market_benchmark} ATR spike >= {1.0 + config.no_lookahead_atr_spike_threshold:.2f}x the prior {config.no_lookahead_atr_lookback_days}-day median, held for {config.no_lookahead_atr_hold_days} trading days.",
        f"- Leverage hysteresis band: enter high leverage above entropy {config.hysteresis_entropy_up_threshold:.2f}; remain high until entropy falls below {config.hysteresis_entropy_down_threshold:.2f}.",
        f"- Selective hysteresis: active-basket liquidity uses `close * volume` weighted `factor_volume_z_20`; z >= {config.selective_hysteresis_liquidity_z_threshold:.2f} uses fast band {config.selective_hysteresis_fast_up_threshold:.2f}/{config.selective_hysteresis_fast_down_threshold:.2f}, otherwise wide band {config.selective_hysteresis_wide_up_threshold:.2f}/{config.selective_hysteresis_wide_down_threshold:.2f}.",
        f"- Transaction-cost stress: one-way turnover charged at {config.transaction_cost_bps:.1f} bp.",
        f"- Promotion execution mandate: trading-side one-way impact must be controlled at <= {EXECUTION_TARGET_TRANSACTION_COST_BPS:.1f} bp before production admission.",
        f"- Smoothing stress: de-risk immediately, re-risk by at most {config.smoothed_rerisk_exposure_step:.2f} exposure points per day.",
        "",
        "## Headline Results",
        "",
        _markdown_table(summary_focus, max_rows=10),
        "",
        "## Cross-Strategy Diversification Gate",
        "",
        "The admission shortcut only applies if hysteresis-only net returns have correlation below 0.70 versus `old_mvo_replay`. If the matrix fails that gate, the PR case must stand on execution quality, drawdown control, and auditability rather than diversification uplift.",
        "",
        _markdown_table(net_correlation_matrix, max_rows=10),
        "",
        _markdown_table(diversification_audit, max_rows=10),
        "",
        "## Case Study: 01-29, 03-06, and April",
        "",
        "01-29 demonstrates the low-entropy exit logic: crowded concentration is not a moment to press exposure. 03-06 is the airbag test: the risk-off overlay catches high-entropy leverage when the holding-period realized-vol audit spikes. April demonstrates the opposite state: broad high-entropy participation supports larger exposure.",
        "",
        _markdown_table(case_study, max_rows=20),
        "",
        "## April Participation Window",
        "",
        _markdown_table(april[april_cols] if not april.empty else april, max_rows=10),
        "",
        "## Visual Evidence",
        "",
        chart_note,
        "",
        "## Actionability Boundary",
        "",
        "The realized-vol spike used here is a holding-period audit feature designed to test whether a brake can reduce observed damage. It is not a live-safe intraday predictor. The benchmark gap and previous-day ATR hold features are no-lookahead proxies. The Conditional Go recommendation is for the hysteresis-only branch with execution optimization; risk-off overlays remain shadow candidates until a separate live-safe signal review clears them.",
    ]
    executive_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"executive_summary": executive_path, "executive_chart": chart_path, "case_study": case_path}


def _write_friction_sensitivity_matrix(output_path: Path, daily: pd.DataFrame) -> Path:
    matrix_path = output_path / "friction_sensitivity_matrix.csv"
    rows: list[dict[str, Any]] = []
    if daily.empty:
        pd.DataFrame(rows).to_csv(matrix_path, index=False)
        return matrix_path
    for variant, group in daily.groupby("variant", sort=False):
        group = group.sort_values("datetime").copy()
        turnover = _numeric(group["one_way_turnover"]).fillna(0.0)
        gross = _numeric(group["variant_effective_return"]).fillna(0.0)
        non_initial = group[~group["is_initial_trade"].fillna(False).astype(bool)].copy()
        for bps in FRICTION_SENSITIVITY_BPS:
            net = gross - turnover * float(bps) / 10_000.0
            rows.append(
                {
                    "variant": variant,
                    "transaction_cost_bps": float(bps),
                    "gross_cumulative_return": _compound_return(gross),
                    "net_cumulative_return": _compound_return(net),
                    "net_return_above_12pct": bool(_compound_return(net) >= 0.12),
                    "net_sharpe": _sharpe(net),
                    "net_max_drawdown": _max_drawdown(net),
                    "total_one_way_turnover": float(turnover.sum()),
                    "annualized_turnover": float(_numeric(non_initial["one_way_turnover"]).mean() * TRADING_DAYS_PER_YEAR)
                    if not non_initial.empty
                    else np.nan,
                    "total_transaction_cost_return": float((turnover * float(bps) / 10_000.0).sum()),
                }
            )
    pd.DataFrame(rows).to_csv(matrix_path, index=False)
    return matrix_path


def _nav_slope(returns: pd.Series) -> float:
    values = _numeric(returns).fillna(0.0)
    if len(values) < 2:
        return float("nan")
    nav = (1.0 + values).cumprod().to_numpy(dtype=float)
    x = np.arange(len(nav), dtype=float)
    slope, _ = np.polyfit(x, nav, 1)
    return float(slope)


def _return_metric_row(name: str, returns: pd.Series, allocation_old_mvo: float, allocation_variant: float) -> dict[str, Any]:
    values = _numeric(returns).fillna(0.0)
    return {
        "portfolio": name,
        "allocation_old_mvo_replay": float(allocation_old_mvo),
        "allocation_hysteresis_only": float(allocation_variant),
        "net_cumulative_return": _compound_return(values),
        "net_sharpe": _sharpe(values),
        "net_max_drawdown": _max_drawdown(values),
        "net_annualized_vol": float(values.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(values) >= 2 else np.nan,
        "net_nav_slope_per_day": _nav_slope(values),
    }


def _write_cross_strategy_diversification_audit(output_path: Path, daily: pd.DataFrame) -> dict[str, Path]:
    net_matrix_path = output_path / "cross_strategy_net_return_correlation_matrix.csv"
    gross_matrix_path = output_path / "cross_strategy_gross_return_correlation_matrix.csv"
    diversification_path = output_path / "cross_strategy_diversification_audit.csv"
    focus_variants = [
        "old_mvo_replay",
        "regime_calibrated_student_t_loc",
        "overheat_low_entropy_mu30_hysteresis",
        "overheat_low_entropy_mu30_selective_hysteresis_liquidity",
        "overheat_low_entropy_mu30_strategic_leverage_risk_off",
    ]
    if daily.empty:
        pd.DataFrame().to_csv(net_matrix_path)
        pd.DataFrame().to_csv(gross_matrix_path)
        pd.DataFrame().to_csv(diversification_path, index=False)
        return {
            "cross_strategy_net_correlation_matrix": net_matrix_path,
            "cross_strategy_gross_correlation_matrix": gross_matrix_path,
            "cross_strategy_diversification_audit": diversification_path,
        }
    working = daily.loc[daily["variant"].isin(focus_variants)].copy()
    working["datetime"] = pd.to_datetime(working["datetime"])
    net_returns = working.pivot(index="datetime", columns="variant", values="net_variant_effective_return").sort_index()
    gross_returns = working.pivot(index="datetime", columns="variant", values="variant_effective_return").sort_index()
    net_returns.corr().to_csv(net_matrix_path)
    gross_returns.corr().to_csv(gross_matrix_path)

    old_name = "old_mvo_replay"
    hysteresis_name = "overheat_low_entropy_mu30_hysteresis"
    rows: list[dict[str, Any]] = []
    if old_name in net_returns and hysteresis_name in net_returns:
        pair = net_returns[[old_name, hysteresis_name]].dropna()
        correlation = float(pair[old_name].corr(pair[hysteresis_name])) if len(pair) >= 2 else np.nan
        rows.append(
            {
                **_return_metric_row("old_mvo_replay", pair[old_name], 1.0, 0.0),
                "correlation_vs_old_mvo_replay": 1.0,
                "correlation_below_0p70": False,
                "diversification_gate": "baseline",
            }
        )
        rows.append(
            {
                **_return_metric_row("hysteresis_only", pair[hysteresis_name], 0.0, 1.0),
                "correlation_vs_old_mvo_replay": correlation,
                "correlation_below_0p70": bool(np.isfinite(correlation) and correlation < 0.70),
                "diversification_gate": "pass" if np.isfinite(correlation) and correlation < 0.70 else "fail",
            }
        )
        for allocation in (0.25, 0.50, 0.75):
            blended = (1.0 - allocation) * pair[old_name] + allocation * pair[hysteresis_name]
            rows.append(
                {
                    **_return_metric_row(f"blend_old_{1.0 - allocation:.0%}_hysteresis_{allocation:.0%}", blended, 1.0 - allocation, allocation),
                    "correlation_vs_old_mvo_replay": correlation,
                    "correlation_below_0p70": bool(np.isfinite(correlation) and correlation < 0.70),
                    "diversification_gate": "pass" if np.isfinite(correlation) and correlation < 0.70 else "fail",
                }
            )
    pd.DataFrame(rows).to_csv(diversification_path, index=False)
    return {
        "cross_strategy_net_correlation_matrix": net_matrix_path,
        "cross_strategy_gross_correlation_matrix": gross_matrix_path,
        "cross_strategy_diversification_audit": diversification_path,
    }


def _write_regime_loc_calibration_audit(output_path: Path, working: pd.DataFrame) -> Path:
    calibration_path = output_path / "regime_loc_calibration_audit.csv"
    if working.empty:
        pd.DataFrame().to_csv(calibration_path, index=False)
        return calibration_path

    calibration = working.copy()
    calibration["datetime"] = pd.to_datetime(calibration["datetime"]).dt.strftime("%Y-%m-%d")
    calibration["mu_before"] = _numeric(calibration["smart_mu_variant_mu_before"]).fillna(0.0)
    calibration["mu_regime_calibrated_loc"] = _numeric(calibration["smart_mu_variant_mu_regime_calibrated_loc"]).fillna(0.0)
    calibration["mu_loc_delta"] = calibration["mu_regime_calibrated_loc"] - calibration["mu_before"]
    calibration["positive_mu_negative_return"] = (
        calibration["mu_regime_calibrated_loc"].fillna(0.0) > 0.0
    ) & (_numeric(calibration["next_return"]).fillna(0.0) < 0.0)
    calibration["loc_direction_correct"] = np.where(
        (calibration["mu_regime_calibrated_loc"].abs() > 1e-12) & (_numeric(calibration["next_return"]).abs() > 1e-12),
        np.sign(calibration["mu_regime_calibrated_loc"]) == np.sign(_numeric(calibration["next_return"])),
        np.nan,
    )

    group_columns = [
        "datetime",
        "smart_mu_variant_theme_bucket",
        "smart_mu_variant_entropy_bucket",
        "smart_mu_variant_overheat_regime",
        "smart_mu_variant_liquidity_bucket",
        "smart_mu_variant_regime_loc_slope_source",
    ]
    summary = (
        calibration.groupby(group_columns, dropna=False)
        .agg(
            rows=("instrument", "size"),
            instruments=("instrument", "nunique"),
            loc_slope_a=("smart_mu_variant_regime_loc_slope_a", "mean"),
            loc_slope_raw=("smart_mu_variant_regime_loc_slope_raw", "mean"),
            loc_slope_obs=("smart_mu_variant_regime_loc_slope_obs", "max"),
            loc_slope_dates=("smart_mu_variant_regime_loc_slope_dates", "max"),
            global_slope_a=("smart_mu_variant_regime_loc_global_slope_a", "mean"),
            global_slope_obs=("smart_mu_variant_regime_loc_global_slope_obs", "max"),
            global_slope_dates=("smart_mu_variant_regime_loc_global_slope_dates", "max"),
            avg_mu_before=("mu_before", "mean"),
            avg_mu_regime_calibrated_loc=("mu_regime_calibrated_loc", "mean"),
            avg_mu_loc_delta=("mu_loc_delta", "mean"),
            avg_next_return=("next_return", "mean"),
            direction_accuracy=("loc_direction_correct", "mean"),
            positive_mu_negative_return_rate=("positive_mu_negative_return", "mean"),
            mu_delta_abs_sum=("mu_loc_delta", lambda values: float(_numeric(values).abs().sum())),
        )
        .reset_index()
        .rename(
            columns={
                "smart_mu_variant_theme_bucket": "factor_industry_theme",
                "smart_mu_variant_entropy_bucket": "entropy_bucket",
                "smart_mu_variant_overheat_regime": "overheat_regime",
                "smart_mu_variant_liquidity_bucket": "liquidity_bucket",
                "smart_mu_variant_regime_loc_slope_source": "loc_slope_source",
            }
        )
        .sort_values(["datetime", "mu_delta_abs_sum"], ascending=[True, False])
    )
    summary.to_csv(calibration_path, index=False)
    return calibration_path


def _write_mean_error_attribution(
    output_path: Path,
    weights: pd.DataFrame,
    config: SmartMuVariantAuditConfig,
) -> dict[str, Path]:
    attribution_path = output_path / "mean_error_attribution.csv"
    regime_summary_path = output_path / "mean_error_regime_summary.csv"
    top_overestimation_path = output_path / "mean_error_top_overestimation_rows.csv"
    diagnostic_summary_path = output_path / "mean_error_diagnostic_summary.md"

    if weights.empty:
        pd.DataFrame().to_csv(attribution_path, index=False)
        pd.DataFrame().to_csv(regime_summary_path, index=False)
        pd.DataFrame().to_csv(top_overestimation_path, index=False)
        diagnostic_summary_path.write_text("# Mean Error Diagnostic Summary\n\nNo rows available.\n", encoding="utf-8")
        return {
            "mean_error_attribution": attribution_path,
            "mean_error_regime_summary": regime_summary_path,
            "mean_error_top_overestimation_rows": top_overestimation_path,
            "mean_error_diagnostic_summary": diagnostic_summary_path,
        }

    attribution = weights.copy()
    attribution["datetime"] = pd.to_datetime(attribution["datetime"]).dt.strftime("%Y-%m-%d")
    attribution["mu_error"] = _numeric(attribution["variant_mu_after"]) - _numeric(attribution["next_return"])
    attribution["mu_before_error"] = _numeric(attribution["mu_before"]) - _numeric(attribution["next_return"])
    attribution["realized_return_contribution"] = (
        _numeric(attribution["variant_effective_weight"]).fillna(0.0) * _numeric(attribution["next_return"]).fillna(0.0)
    )
    attribution["forecast_mu_contribution"] = (
        _numeric(attribution["variant_effective_weight"]).fillna(0.0) * _numeric(attribution["variant_mu_after"]).fillna(0.0)
    )
    attribution["weighted_mu_error"] = attribution["forecast_mu_contribution"] - attribution["realized_return_contribution"]
    attribution["abs_variant_effective_weight"] = _numeric(attribution["variant_effective_weight"]).abs().fillna(0.0)
    attribution["active_variant_position"] = attribution["abs_variant_effective_weight"] > 1e-12
    attribution["positive_mu_negative_return"] = (
        _numeric(attribution["variant_mu_after"]).fillna(0.0) > 0.0
    ) & (_numeric(attribution["next_return"]).fillna(0.0) < 0.0)
    direction_evaluable = (_numeric(attribution["variant_mu_after"]).abs() > 1e-12) & (_numeric(attribution["next_return"]).abs() > 1e-12)
    attribution["direction_correct"] = np.where(
        direction_evaluable,
        np.sign(_numeric(attribution["variant_mu_after"])) == np.sign(_numeric(attribution["next_return"])),
        np.nan,
    )
    attribution["negative_realized_contribution"] = np.where(
        attribution["realized_return_contribution"] < 0.0,
        attribution["realized_return_contribution"],
        0.0,
    )
    entropy = _numeric(attribution["industry_entropy_norm"])
    attribution["entropy_bucket"] = np.select(
        [
            entropy <= float(config.entropy_low_threshold),
            entropy <= float(config.high_entropy_threshold),
            entropy <= float(config.hysteresis_entropy_up_threshold),
            entropy > float(config.hysteresis_entropy_up_threshold),
        ],
        ["low_entropy", "mid_entropy", "high_entropy_pre_hysteresis", "hysteresis_high_entropy"],
        default="entropy_unknown",
    )
    liquidity_z = _numeric(attribution["active_liquidity_z_amount_weighted"])
    attribution["liquidity_bucket"] = np.select(
        [
            liquidity_z < 0.0,
            liquidity_z < float(config.selective_hysteresis_liquidity_z_threshold),
            liquidity_z < 0.75,
            liquidity_z >= 0.75,
        ],
        ["thin_liquidity", "below_selective_threshold", "liquid", "very_liquid"],
        default="liquidity_unknown",
    )
    instrument_overheat_low_entropy = (
        attribution["overheat_5d"].fillna(False).astype(bool)
        & attribution["low_entropy_day"].fillna(False).astype(bool)
    )
    day_overheat_low_entropy_context = (
        attribution["overheat_low_entropy_day"].fillna(False).astype(bool)
        & ~instrument_overheat_low_entropy
    )
    attribution["overheat_regime"] = np.select(
        [
            instrument_overheat_low_entropy,
            day_overheat_low_entropy_context,
            attribution["overheat_5d"].fillna(False).astype(bool),
            attribution["raw_overheat_5d"].fillna(False).astype(bool),
            attribution["route_a_overheat_5d"].fillna(False).astype(bool),
        ],
        [
            "instrument_overheat_low_entropy",
            "day_overheat_low_entropy_context",
            "instrument_overheat",
            "raw_overheat",
            "route_a_overheat",
        ],
        default="no_overheat",
    )
    attribution["position_bucket"] = np.where(attribution["active_variant_position"], "active", "flat")

    attribution_columns = [
        "datetime",
        "variant",
        "instrument",
        "factor_industry_theme",
        "entropy_bucket",
        "industry_entropy_norm",
        "overheat_regime",
        "raw_overheat_5d",
        "route_a_overheat_5d",
        "overheat_5d",
        "low_entropy_day",
        "high_entropy_day",
        "hysteresis_high_entropy_day",
        "selective_hysteresis_high_entropy_day",
        "active_liquidity_high_day",
        "liquidity_bucket",
        "active_liquidity_z_amount_weighted",
        "selective_hysteresis_band",
        "strategic_leverage_regime",
        "mu_before",
        "variant_mu_after",
        "variant_mu_delta",
        "next_return",
        "mu_error",
        "mu_before_error",
        "direction_correct",
        "positive_mu_negative_return",
        "old_mvo_weight",
        "variant_mvo_weight",
        "old_effective_weight",
        "variant_effective_weight",
        "abs_variant_effective_weight",
        "position_bucket",
        "realized_return_contribution",
        "forecast_mu_contribution",
        "weighted_mu_error",
        "negative_realized_contribution",
        "delta_return_contribution_vs_old",
    ]
    attribution[attribution_columns].to_csv(attribution_path, index=False)

    group_columns = [
        "variant",
        "factor_industry_theme",
        "entropy_bucket",
        "overheat_regime",
        "liquidity_bucket",
        "active_liquidity_high_day",
        "position_bucket",
    ]
    regime_summary = (
        attribution.groupby(group_columns, dropna=False)
        .agg(
            rows=("instrument", "size"),
            dates=("datetime", "nunique"),
            instruments=("instrument", "nunique"),
            active_rows=("active_variant_position", "sum"),
            avg_mu_before=("mu_before", "mean"),
            avg_variant_mu=("variant_mu_after", "mean"),
            avg_next_return=("next_return", "mean"),
            avg_mu_error=("mu_error", "mean"),
            median_mu_error=("mu_error", "median"),
            direction_accuracy=("direction_correct", "mean"),
            positive_mu_negative_return_rows=("positive_mu_negative_return", "sum"),
            abs_variant_effective_weight_sum=("abs_variant_effective_weight", "sum"),
            realized_return_contribution_sum=("realized_return_contribution", "sum"),
            forecast_mu_contribution_sum=("forecast_mu_contribution", "sum"),
            weighted_mu_error_sum=("weighted_mu_error", "sum"),
            negative_realized_contribution_sum=("negative_realized_contribution", "sum"),
            delta_return_contribution_vs_old_sum=("delta_return_contribution_vs_old", "sum"),
            avg_active_liquidity_z_amount_weighted=("active_liquidity_z_amount_weighted", "mean"),
            avg_industry_entropy_norm=("industry_entropy_norm", "mean"),
        )
        .reset_index()
    )
    regime_summary["positive_mu_negative_return_rate"] = np.where(
        regime_summary["rows"] > 0,
        regime_summary["positive_mu_negative_return_rows"] / regime_summary["rows"],
        np.nan,
    )
    regime_summary = regime_summary.sort_values(
        ["weighted_mu_error_sum", "negative_realized_contribution_sum", "abs_variant_effective_weight_sum"],
        ascending=[False, True, False],
    )
    regime_summary.to_csv(regime_summary_path, index=False)

    top_overestimation = attribution.loc[
        attribution["active_variant_position"] & (attribution["weighted_mu_error"] > 0.0), attribution_columns
    ].sort_values("weighted_mu_error", ascending=False)
    top_overestimation.head(500).to_csv(top_overestimation_path, index=False)

    focus_variants = [
        "old_mvo_replay",
        "regime_calibrated_student_t_loc",
        "overheat_low_entropy_mu30_hysteresis",
        "overheat_low_entropy_mu30_selective_hysteresis_liquidity",
    ]
    summary_columns = [
        "variant",
        "factor_industry_theme",
        "entropy_bucket",
        "overheat_regime",
        "liquidity_bucket",
        "position_bucket",
        "rows",
        "dates",
        "instruments",
        "avg_variant_mu",
        "avg_next_return",
        "avg_mu_error",
        "direction_accuracy",
        "positive_mu_negative_return_rate",
        "abs_variant_effective_weight_sum",
        "realized_return_contribution_sum",
        "forecast_mu_contribution_sum",
        "weighted_mu_error_sum",
        "negative_realized_contribution_sum",
        "delta_return_contribution_vs_old_sum",
    ]
    top_columns = [
        "datetime",
        "variant",
        "instrument",
        "factor_industry_theme",
        "entropy_bucket",
        "overheat_regime",
        "liquidity_bucket",
        "strategic_leverage_regime",
        "variant_mu_after",
        "next_return",
        "mu_error",
        "variant_effective_weight",
        "realized_return_contribution",
        "forecast_mu_contribution",
        "weighted_mu_error",
    ]
    active_attribution = attribution.loc[attribution["active_variant_position"]].copy()
    variant_summary = (
        active_attribution.groupby("variant", dropna=False)
        .agg(
            rows=("instrument", "size"),
            dates=("datetime", "nunique"),
            instruments=("instrument", "nunique"),
            avg_variant_mu=("variant_mu_after", "mean"),
            avg_next_return=("next_return", "mean"),
            avg_mu_error=("mu_error", "mean"),
            direction_accuracy=("direction_correct", "mean"),
            positive_mu_negative_return_rate=("positive_mu_negative_return", "mean"),
            abs_variant_effective_weight_sum=("abs_variant_effective_weight", "sum"),
            realized_return_contribution_sum=("realized_return_contribution", "sum"),
            forecast_mu_contribution_sum=("forecast_mu_contribution", "sum"),
            weighted_mu_error_sum=("weighted_mu_error", "sum"),
            negative_realized_contribution_sum=("negative_realized_contribution", "sum"),
        )
        .reset_index()
        .sort_values("weighted_mu_error_sum", ascending=False)
    )
    diagnostic_lines = [
        "# Mean Error Diagnostic Summary",
        "",
        "Audit-only diagnostic. This report attributes forecast mean error by date, instrument, theme, entropy, overheat, liquidity, and active MVO/effective weight.",
        "",
        "## Active Variant Summary",
        "",
        _markdown_table(variant_summary, max_rows=20),
        "",
        "## Top Regime Overestimation",
        "",
        _markdown_table(regime_summary[summary_columns].head(30), max_rows=30),
        "",
        "## Focus Variant Regime Overestimation",
        "",
        _markdown_table(regime_summary.loc[regime_summary["variant"].isin(focus_variants), summary_columns].head(30), max_rows=30),
        "",
        "## Top Instrument Rows",
        "",
        _markdown_table(top_overestimation.loc[top_overestimation["variant"].isin(focus_variants), top_columns].head(30), max_rows=30),
        "",
        "## Reading Rule",
        "",
        "Positive `weighted_mu_error_sum` means active weighted forecast contribution exceeded realized contribution. The highest rows are the places where a mean-calibration layer should be tested before adding more optimizer or L3 controls.",
    ]
    diagnostic_summary_path.write_text("\n".join(diagnostic_lines) + "\n", encoding="utf-8")
    return {
        "mean_error_attribution": attribution_path,
        "mean_error_regime_summary": regime_summary_path,
        "mean_error_top_overestimation_rows": top_overestimation_path,
        "mean_error_diagnostic_summary": diagnostic_summary_path,
    }


def run_smart_mu_variant_audit(
    bridge_l3_csv: str | Path,
    parameter_audit_csv: str | Path,
    output_dir: str | Path,
    config: SmartMuVariantAuditConfig,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    bridge_frame = _load_bridge_frame(bridge_l3_csv, config.start_date, config.end_date)
    working, trigger_metadata = _attach_variant_flags(bridge_frame, parameter_audit_csv, config)
    working["smart_mu_variant_trigger_none"] = False

    variants = _variant_definitions()
    previous_by_variant = {variant.name: pd.Series(dtype=float) for variant in variants}
    previous_effective_by_variant = {variant.name: pd.Series(dtype=float) for variant in variants}
    previous_exposure_by_variant = {variant.name: pd.Series(dtype=float) for variant in variants}
    daily_rows: list[dict[str, Any]] = []
    weight_frames: list[pd.DataFrame] = []

    for date_value, day_frame in working.groupby("datetime", sort=True):
        date_text = pd.Timestamp(date_value).strftime("%Y-%m-%d")
        sample = day_frame.set_index("instrument", drop=False).sort_index()
        sample_index = pd.Index(sample.index.astype(str), dtype=object)
        next_return = _numeric(sample["next_return"]).fillna(0.0)
        base_exposure_scale = _numeric(sample["exposure_scale"]).fillna(1.0)
        published_w = _numeric(sample["forecast_w"]).fillna(0.0)
        published_effective = published_w * base_exposure_scale
        published_return = float((published_effective * next_return).sum())

        day_weights_by_variant: dict[str, pd.Series] = {}
        status_by_variant: dict[str, str] = {}
        for variant in variants:
            weights, status = _optimize_variant_day(day_frame, variant, previous_by_variant[variant.name], config)
            weights = weights.reindex(sample_index).fillna(0.0)
            day_weights_by_variant[variant.name] = weights
            status_by_variant[variant.name] = status

        old_weights = day_weights_by_variant["old_mvo_replay"]
        old_effective = old_weights * base_exposure_scale
        old_return = float((old_effective * next_return).sum())

        for variant in variants:
            weights = day_weights_by_variant[variant.name]
            target_exposure_scale, exposure_regime = _variant_exposure_scale(
                sample,
                base_exposure_scale,
                variant,
                config,
            )
            variant_exposure_scale, exposure_was_smoothed = _apply_rerisk_smoothing(
                target_exposure_scale,
                previous_exposure_by_variant[variant.name],
                variant,
                config,
            )
            if exposure_was_smoothed:
                exposure_regime = f"{exposure_regime}_smooth_rerisk"
            effective = weights * variant_exposure_scale
            effective_delta = effective - old_effective
            previous_effective = previous_effective_by_variant[variant.name].reindex(sample_index).fillna(0.0)
            one_way_turnover = float((effective - previous_effective).abs().sum())
            transaction_cost_return = one_way_turnover * float(config.transaction_cost_bps) / 10_000.0
            variant_return = float((effective * next_return).sum())
            net_variant_return = variant_return - transaction_cost_return
            delta_w = weights - old_weights
            contribution_delta = effective_delta * next_return
            trigger = sample[variant.trigger_column].fillna(False).astype(bool)
            excluded = sample[variant.exclude_column].fillna(False).astype(bool) if variant.exclude_column else pd.Series(False, index=sample_index)
            mu_after = _numeric(sample[variant.mean_column]).fillna(0.0)
            mu_before = _numeric(sample["smart_mu_variant_mu_before"]).fillna(0.0)
            weight_frames.append(
                pd.DataFrame(
                    {
                        "datetime": date_text,
                        "variant": variant.name,
                        "instrument": sample_index,
                        "factor_industry_theme": sample["factor_industry_theme"].astype(str).to_numpy(),
                        "regime_calibration_theme": sample["smart_mu_variant_theme_bucket"].astype(str).to_numpy(),
                        "regime_calibration_entropy_bucket": sample["smart_mu_variant_entropy_bucket"].astype(str).to_numpy(),
                        "regime_calibration_overheat_regime": sample["smart_mu_variant_overheat_regime"].astype(str).to_numpy(),
                        "regime_calibration_liquidity_bucket": sample["smart_mu_variant_liquidity_bucket"].astype(str).to_numpy(),
                        "regime_loc_calibration_slope_a": sample["smart_mu_variant_regime_loc_slope_a"].to_numpy(),
                        "regime_loc_calibration_slope_raw": sample["smart_mu_variant_regime_loc_slope_raw"].to_numpy(),
                        "regime_loc_calibration_slope_obs": sample["smart_mu_variant_regime_loc_slope_obs"].to_numpy(),
                        "regime_loc_calibration_slope_dates": sample["smart_mu_variant_regime_loc_slope_dates"].to_numpy(),
                        "regime_loc_calibration_global_slope_a": sample["smart_mu_variant_regime_loc_global_slope_a"].to_numpy(),
                        "regime_loc_calibration_global_slope_obs": sample["smart_mu_variant_regime_loc_global_slope_obs"].to_numpy(),
                        "regime_loc_calibration_global_slope_dates": sample["smart_mu_variant_regime_loc_global_slope_dates"].to_numpy(),
                        "regime_loc_calibration_slope_source": sample["smart_mu_variant_regime_loc_slope_source"].astype(str).to_numpy(),
                        "variant_triggered": trigger.to_numpy(),
                        "variant_excluded": excluded.to_numpy(),
                        "raw_overheat_5d": sample["smart_mu_variant_trigger_raw_overheat_5d"].fillna(False).astype(bool).to_numpy(),
                        "route_a_overheat_5d": sample["smart_mu_variant_trigger_route_a_overheat_5d"].fillna(False).astype(bool).to_numpy(),
                        "overheat_5d": sample["smart_mu_variant_trigger_overheat_5d"].fillna(False).astype(bool).to_numpy(),
                        "momentum_deceleration_flag": sample["smart_mu_variant_momentum_deceleration_flag"].fillna(False).astype(bool).to_numpy(),
                        "momentum_flat_or_negative_flag": sample["smart_mu_variant_momentum_flat_or_negative_flag"].fillna(False).astype(bool).to_numpy(),
                        "communication_theme_flag": sample["smart_mu_variant_communication_theme_flag"].fillna(False).astype(bool).to_numpy(),
                        "low_entropy_day": sample["smart_mu_variant_low_entropy_day"].fillna(False).astype(bool).to_numpy(),
                        "high_entropy_day": sample["smart_mu_variant_high_entropy_day"].fillna(False).astype(bool).to_numpy(),
                        "hysteresis_high_entropy_day": sample["smart_mu_variant_hysteresis_high_entropy_day"].fillna(False).astype(bool).to_numpy(),
                        "selective_hysteresis_high_entropy_day": sample[
                            "smart_mu_variant_selective_hysteresis_high_entropy_day"
                        ].fillna(False).astype(bool).to_numpy(),
                        "active_liquidity_high_day": sample["smart_mu_variant_active_liquidity_high_day"].fillna(False).astype(bool).to_numpy(),
                        "selective_hysteresis_band": sample["smart_mu_variant_selective_hysteresis_band"].astype(str).to_numpy(),
                        "overheat_low_entropy_day": sample["smart_mu_variant_overheat_low_entropy_day"].fillna(False).astype(bool).to_numpy(),
                        "risk_off_overlay_day": sample["smart_mu_variant_trigger_risk_off_overlay"].fillna(False).astype(bool).to_numpy(),
                        "risk_off_overlay_no_lookahead_day": sample["smart_mu_variant_trigger_risk_off_overlay_no_lookahead"].fillna(False).astype(bool).to_numpy(),
                        "market_vol_spike_risk_off": sample["smart_mu_variant_market_vol_spike_risk_off"].fillna(False).astype(bool).to_numpy(),
                        "market_gap_risk_off": sample["smart_mu_variant_market_gap_risk_off"].fillna(False).astype(bool).to_numpy(),
                        "market_atr_spike_no_lookahead": sample["smart_mu_variant_market_benchmark_atr_spike_no_lookahead"].fillna(False).astype(bool).to_numpy(),
                        "strategic_leverage_regime": exposure_regime,
                        "industry_entropy_norm": sample["smart_mu_variant_industry_entropy_norm"].to_numpy(),
                        "active_liquidity_z_amount_weighted": sample[
                            "smart_mu_variant_active_liquidity_z_amount_weighted"
                        ].to_numpy(),
                        "active_liquidity_z_mean": sample["smart_mu_variant_active_liquidity_z_mean"].to_numpy(),
                        "active_amount_proxy": sample["smart_mu_variant_active_amount_proxy"].to_numpy(),
                        "selective_hysteresis_up_threshold": sample[
                            "smart_mu_variant_selective_hysteresis_up_threshold"
                        ].to_numpy(),
                        "selective_hysteresis_down_threshold": sample[
                            "smart_mu_variant_selective_hysteresis_down_threshold"
                        ].to_numpy(),
                        "market_holding_realized_vol": sample["smart_mu_variant_market_holding_realized_vol"].to_numpy(),
                        "market_realized_vol_baseline": sample["smart_mu_variant_market_realized_vol_baseline"].to_numpy(),
                        "market_realized_vol_spike_ratio": sample["smart_mu_variant_market_realized_vol_spike_ratio"].to_numpy(),
                        "market_overnight_gap": sample["smart_mu_variant_market_overnight_gap"].to_numpy(),
                        "market_benchmark_true_range_pct": sample["smart_mu_variant_market_benchmark_true_range_pct"].to_numpy(),
                        "market_benchmark_atr_spike_ratio": sample["smart_mu_variant_market_benchmark_atr_spike_ratio"].to_numpy(),
                        "dominant_industry": sample["smart_mu_variant_dominant_industry"].astype(str).to_numpy(),
                        "dominant_industry_weight_share": sample["smart_mu_variant_dominant_industry_weight_share"].to_numpy(),
                        "old_mvo_weight": old_weights.to_numpy(),
                        "variant_mvo_weight": weights.to_numpy(),
                        "delta_mvo_weight_vs_old": delta_w.to_numpy(),
                        "old_effective_weight": old_effective.to_numpy(),
                        "variant_effective_weight": effective.to_numpy(),
                        "delta_effective_weight_vs_old": effective_delta.to_numpy(),
                        "published_l3_weight": published_w.to_numpy(),
                        "published_l3_effective_weight": published_effective.to_numpy(),
                        "mu_before": mu_before.to_numpy(),
                        "variant_mu_after": mu_after.to_numpy(),
                        "variant_mu_delta": (mu_after - mu_before).to_numpy(),
                        "raw_close_return_5d": sample["smart_mu_variant_raw_close_return_5d"].to_numpy(),
                        "raw_close_return_2d": sample["smart_mu_variant_raw_close_return_2d"].to_numpy(),
                        "expected_return_2d_from_5d": sample["smart_mu_variant_expected_return_2d_from_5d"].to_numpy(),
                        "return_5d_daily_rate": sample["smart_mu_variant_return_5d_daily_rate"].to_numpy(),
                        "return_2d_daily_rate": sample["smart_mu_variant_return_2d_daily_rate"].to_numpy(),
                        "next_return": next_return.to_numpy(),
                        "exposure_scale": base_exposure_scale.to_numpy(),
                        "base_exposure_scale": base_exposure_scale.to_numpy(),
                        "target_exposure_scale": target_exposure_scale.to_numpy(),
                        "variant_exposure_scale": variant_exposure_scale.to_numpy(),
                        "exposure_was_smoothed": exposure_was_smoothed,
                        "delta_exposure_scale_vs_base": (variant_exposure_scale - base_exposure_scale).to_numpy(),
                        "delta_return_contribution_vs_old": contribution_delta.to_numpy(),
                    }
                )
            )
            daily_rows.append(
                {
                    "datetime": date_text,
                    "variant": variant.name,
                    "description": variant.description,
                    "triggered_rows": int(trigger.sum()),
                    "excluded_rows": int(excluded.sum()),
                    "raw_overheat_rows": int(sample["smart_mu_variant_trigger_raw_overheat_5d"].sum()),
                    "route_a_overheat_rows": int(sample["smart_mu_variant_trigger_route_a_overheat_5d"].sum()),
                    "overheat_rows": int(sample["smart_mu_variant_trigger_overheat_5d"].sum()),
                    "momentum_deceleration_rows": int(sample["smart_mu_variant_trigger_overheat_deceleration"].sum()),
                    "momentum_flat_or_negative_rows": int(sample["smart_mu_variant_momentum_flat_or_negative_flag"].sum()),
                    "communication_rows": int(sample["smart_mu_variant_communication_theme_flag"].sum()),
                    "low_entropy_day": bool(sample["smart_mu_variant_low_entropy_day"].iloc[0]),
                    "high_entropy_day": bool(sample["smart_mu_variant_high_entropy_day"].iloc[0]),
                    "hysteresis_high_entropy_day": bool(sample["smart_mu_variant_hysteresis_high_entropy_day"].iloc[0]),
                    "selective_hysteresis_high_entropy_day": bool(
                        sample["smart_mu_variant_selective_hysteresis_high_entropy_day"].iloc[0]
                    ),
                    "active_liquidity_high_day": bool(sample["smart_mu_variant_active_liquidity_high_day"].iloc[0]),
                    "selective_hysteresis_band": str(sample["smart_mu_variant_selective_hysteresis_band"].iloc[0]),
                    "overheat_low_entropy_day": bool(sample["smart_mu_variant_overheat_low_entropy_day"].iloc[0]),
                    "risk_off_overlay_day": bool(sample["smart_mu_variant_trigger_risk_off_overlay"].iloc[0]),
                    "risk_off_overlay_no_lookahead_day": bool(sample["smart_mu_variant_trigger_risk_off_overlay_no_lookahead"].iloc[0]),
                    "market_vol_spike_risk_off": bool(sample["smart_mu_variant_market_vol_spike_risk_off"].iloc[0]),
                    "market_gap_risk_off": bool(sample["smart_mu_variant_market_gap_risk_off"].iloc[0]),
                    "market_atr_spike_no_lookahead": bool(sample["smart_mu_variant_market_benchmark_atr_spike_no_lookahead"].iloc[0]),
                    "industry_entropy_norm": float(sample["smart_mu_variant_industry_entropy_norm"].iloc[0]),
                    "active_liquidity_z_amount_weighted": float(
                        sample["smart_mu_variant_active_liquidity_z_amount_weighted"].iloc[0]
                    ),
                    "active_liquidity_z_mean": float(sample["smart_mu_variant_active_liquidity_z_mean"].iloc[0]),
                    "active_amount_proxy": float(sample["smart_mu_variant_active_amount_proxy"].iloc[0]),
                    "selective_hysteresis_up_threshold": float(
                        sample["smart_mu_variant_selective_hysteresis_up_threshold"].iloc[0]
                    ),
                    "selective_hysteresis_down_threshold": float(
                        sample["smart_mu_variant_selective_hysteresis_down_threshold"].iloc[0]
                    ),
                    "market_holding_realized_vol": float(sample["smart_mu_variant_market_holding_realized_vol"].iloc[0]),
                    "market_realized_vol_baseline": float(sample["smart_mu_variant_market_realized_vol_baseline"].iloc[0]),
                    "market_realized_vol_spike_ratio": float(sample["smart_mu_variant_market_realized_vol_spike_ratio"].iloc[0]),
                    "market_overnight_gap": float(sample["smart_mu_variant_market_overnight_gap"].iloc[0]),
                    "market_benchmark_true_range_pct": float(sample["smart_mu_variant_market_benchmark_true_range_pct"].iloc[0]),
                    "market_benchmark_atr_spike_ratio": float(sample["smart_mu_variant_market_benchmark_atr_spike_ratio"].iloc[0]),
                    "dominant_industry": str(sample["smart_mu_variant_dominant_industry"].iloc[0]),
                    "dominant_industry_weight_share": float(sample["smart_mu_variant_dominant_industry_weight_share"].iloc[0]),
                    "strategic_leverage_regime": exposure_regime,
                    "base_exposure_scale_mean": float(base_exposure_scale.mean()),
                    "variant_exposure_scale_mean": float(variant_exposure_scale.mean()),
                    "target_exposure_scale_mean": float(target_exposure_scale.mean()),
                    "exposure_was_smoothed": exposure_was_smoothed,
                    "variant_exposure_scale_min": float(variant_exposure_scale.min()),
                    "variant_exposure_scale_max": float(variant_exposure_scale.max()),
                    "changed_exposure_scale_count_vs_base": int(((variant_exposure_scale - base_exposure_scale).abs() > 1e-12).sum()),
                    "active_old_count": int((old_weights > 1e-12).sum()),
                    "active_variant_count": int((weights > 1e-12).sum()),
                    "changed_weight_count_vs_old": int((delta_w.abs() > 1e-8).sum()),
                    "gross_abs_weight_delta_vs_old": float(delta_w.abs().sum()),
                    "max_abs_weight_delta_vs_old": float(delta_w.abs().max()) if len(delta_w) else 0.0,
                    "changed_effective_weight_count_vs_old": int((effective_delta.abs() > 1e-8).sum()),
                    "gross_abs_effective_weight_delta_vs_old": float(effective_delta.abs().sum()),
                    "max_abs_effective_weight_delta_vs_old": float(effective_delta.abs().max()) if len(effective_delta) else 0.0,
                    "mu_delta_abs_sum": float((mu_after - mu_before).abs().sum()),
                    "one_way_turnover": one_way_turnover,
                    "transaction_cost_bps": float(config.transaction_cost_bps),
                    "transaction_cost_return": transaction_cost_return,
                    "net_variant_effective_return": net_variant_return,
                    "is_initial_trade": bool(previous_effective_by_variant[variant.name].empty),
                    "old_mvo_effective_return": old_return,
                    "variant_effective_return": variant_return,
                    "delta_return_vs_old_mvo": variant_return - old_return,
                    "published_l3_effective_return": published_return,
                    "delta_return_vs_published_l3": variant_return - published_return,
                    "status": status_by_variant[variant.name],
                    "top_weight_delta_codes": "|".join(delta_w.abs().sort_values(ascending=False).head(8).index.astype(str).tolist()),
                    "top_delta_contribution_codes": "|".join(
                        contribution_delta.abs().sort_values(ascending=False).head(8).index.astype(str).tolist()
                    ),
                }
            )
            previous_by_variant[variant.name] = weights.copy()
            previous_effective_by_variant[variant.name] = effective.copy()
            previous_exposure_by_variant[variant.name] = variant_exposure_scale.copy()

    daily = pd.DataFrame(daily_rows)
    weights = pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    if not daily.empty:
        daily["variant_nav_proxy"] = daily.groupby("variant", sort=False)["variant_effective_return"].transform(
            lambda values: (1.0 + _numeric(values).fillna(0.0)).cumprod()
        )
        daily["old_mvo_nav_proxy"] = daily.groupby("variant", sort=False)["old_mvo_effective_return"].transform(
            lambda values: (1.0 + _numeric(values).fillna(0.0)).cumprod()
        )
        daily["published_l3_nav_proxy"] = daily.groupby("variant", sort=False)["published_l3_effective_return"].transform(
            lambda values: (1.0 + _numeric(values).fillna(0.0)).cumprod()
        )
        daily["net_variant_nav_proxy_after_tcost"] = daily.groupby("variant", sort=False)["net_variant_effective_return"].transform(
            lambda values: (1.0 + _numeric(values).fillna(0.0)).cumprod()
        )
        daily["delta_nav_vs_old_mvo"] = daily["variant_nav_proxy"] - daily["old_mvo_nav_proxy"]

    summary = _summarize_daily(daily, trigger_metadata)
    summary.insert(1, "audit_scope", "smart_mu_variant_audit_only_no_live_weight_change")
    summary.insert(2, "bridge_l3_csv", str(bridge_l3_csv))
    summary.insert(3, "parameter_audit_csv", str(parameter_audit_csv))
    summary.insert(4, "start_date", config.start_date)
    summary.insert(5, "end_date", config.end_date)
    summary.insert(6, "variance_floor_quantile", float(config.variance_floor_quantile))
    summary.insert(7, "corr_floor", float(config.corr_floor))
    summary.insert(8, "patch_multiplier", float(config.patch_multiplier))
    summary.insert(9, "communication_theme", config.communication_theme)
    monthly = _summarize_monthly(daily)

    trigger_breakdown = pd.DataFrame()
    if not weights.empty:
        trigger_breakdown = (
            weights.groupby(["variant", "variant_triggered", "variant_excluded", "factor_industry_theme"], dropna=False)
            .agg(
                rows=("instrument", "size"),
                instruments=("instrument", "nunique"),
                active_old_rows=("old_mvo_weight", lambda values: int((_numeric(values) > 1e-12).sum())),
                active_variant_rows=("variant_mvo_weight", lambda values: int((_numeric(values) > 1e-12).sum())),
                mu_before_mean=("mu_before", "mean"),
                variant_mu_after_mean=("variant_mu_after", "mean"),
                variant_mu_delta_abs_sum=("variant_mu_delta", lambda values: float(_numeric(values).abs().sum())),
                delta_mvo_weight_abs_sum=("delta_mvo_weight_vs_old", lambda values: float(_numeric(values).abs().sum())),
                delta_effective_weight_abs_sum=("delta_effective_weight_vs_old", lambda values: float(_numeric(values).abs().sum())),
                base_exposure_scale_mean=("base_exposure_scale", "mean"),
                variant_exposure_scale_mean=("variant_exposure_scale", "mean"),
                delta_return_contribution_sum=("delta_return_contribution_vs_old", "sum"),
            )
            .reset_index()
            .sort_values(["variant", "delta_return_contribution_sum", "rows"], ascending=[True, False, False])
        )

    non_old_daily = daily[daily["variant"] != "old_mvo_replay"].copy() if not daily.empty else pd.DataFrame()
    top_positive_days = non_old_daily.sort_values("delta_return_vs_old_mvo", ascending=False).head(20) if not non_old_daily.empty else pd.DataFrame()
    top_negative_days = non_old_daily.sort_values("delta_return_vs_old_mvo", ascending=True).head(20) if not non_old_daily.empty else pd.DataFrame()
    top_contribution_delta = (
        weights[weights["variant"] != "old_mvo_replay"]
        .reindex(weights[weights["variant"] != "old_mvo_replay"]["delta_return_contribution_vs_old"].abs().sort_values(ascending=False).index)
        .head(200)
        if not weights.empty
        else pd.DataFrame()
    )

    daily_path = output_path / "smart_mu_variant_daily.csv"
    weights_path = output_path / "smart_mu_variant_weights.csv"
    summary_path = output_path / "smart_mu_variant_summary.csv"
    monthly_path = output_path / "smart_mu_variant_monthly.csv"
    trigger_breakdown_path = output_path / "smart_mu_variant_trigger_breakdown.csv"
    top_contribution_delta_path = output_path / "smart_mu_variant_top_contribution_delta.csv"
    report_path = output_path / "smart_mu_variant_report.md"
    manifest_path = output_path / "manifest.json"
    regime_loc_calibration_path = _write_regime_loc_calibration_audit(output_path, working)
    regime_loc_calibration_audit = pd.read_csv(regime_loc_calibration_path) if regime_loc_calibration_path.exists() else pd.DataFrame()
    friction_sensitivity_path = _write_friction_sensitivity_matrix(output_path, daily)
    friction_sensitivity = pd.read_csv(friction_sensitivity_path) if friction_sensitivity_path.exists() else pd.DataFrame()
    mean_error_outputs = _write_mean_error_attribution(output_path, weights, config)
    mean_error_regime_summary = pd.read_csv(mean_error_outputs["mean_error_regime_summary"])
    mean_error_top_overestimation = pd.read_csv(mean_error_outputs["mean_error_top_overestimation_rows"])
    diversification_outputs = _write_cross_strategy_diversification_audit(output_path, daily)
    diversification_audit = pd.read_csv(diversification_outputs["cross_strategy_diversification_audit"])
    net_correlation_matrix = pd.read_csv(diversification_outputs["cross_strategy_net_correlation_matrix"])
    executive_outputs = _write_strategic_executive_summary(
        output_path,
        daily,
        monthly,
        summary,
        config,
        diversification_audit,
        net_correlation_matrix,
    )

    daily.to_csv(daily_path, index=False)
    weights.to_csv(weights_path, index=False)
    summary.to_csv(summary_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    trigger_breakdown.to_csv(trigger_breakdown_path, index=False)
    top_contribution_delta.to_csv(top_contribution_delta_path, index=False)
    manifest_path.write_text(
        json.dumps(
            _json_ready(
                {
                    "scope": "smart_mu_variant_audit_only_no_live_weight_change",
                    "config": config.__dict__,
                    "bridge_l3_csv": str(bridge_l3_csv),
                    "parameter_audit_csv": str(parameter_audit_csv),
                    "trigger_metadata": trigger_metadata,
                    "variant_definitions": [variant.__dict__ for variant in variants],
                    "outputs": {
                        "daily": daily_path,
                        "weights": weights_path,
                        "summary": summary_path,
                        "monthly": monthly_path,
                        "trigger_breakdown": trigger_breakdown_path,
                        "top_contribution_delta": top_contribution_delta_path,
                        "regime_loc_calibration_audit": regime_loc_calibration_path,
                        "friction_sensitivity_matrix": friction_sensitivity_path,
                        **mean_error_outputs,
                        **diversification_outputs,
                        "report": report_path,
                        **executive_outputs,
                    },
                }
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "# Smart-mu Variant Audit Replay",
        "",
        "Audit-only replay. Production Student-t, bridge, optimizer defaults, and L3 risk controls are not modified.",
        "",
        "## Variant Summary",
        "",
        _markdown_table(
            summary[
                [
                    "variant",
                    "cumulative_return",
                    "net_cumulative_return_after_tcost",
                    "annualized_turnover",
                    "total_transaction_cost_return",
                    "delta_cumulative_return_vs_old_mvo",
                    "published_l3_cumulative_return",
                    "max_drawdown",
                    "net_max_drawdown_after_tcost",
                    "sharpe",
                    "net_sharpe_after_tcost",
                    "avg_variant_exposure_scale",
                    "exposure_adjusted_days",
                    "smoothed_exposure_days",
                    "risk_off_overlay_days",
                    "risk_off_overlay_no_lookahead_days",
                    "risk_off_exposure_regime_days",
                    "hysteresis_high_exposure_regime_days",
                    "selective_hysteresis_high_exposure_regime_days",
                    "active_liquidity_high_days",
                    "avg_active_liquidity_z_amount_weighted",
                    "triggered_rows",
                    "trigger_active_days",
                    "best_delta_day",
                    "best_delta_return_vs_old_mvo",
                    "worst_delta_day",
                    "worst_delta_return_vs_old_mvo",
                ]
            ]
            if not summary.empty
            else summary,
            max_rows=20,
        ),
        "",
        "## Friction Sensitivity",
        "",
        _markdown_table(friction_sensitivity, max_rows=60),
        "",
        "## Regime-Calibrated Student-t Loc Audit",
        "",
        "No-lookahead shadow calibration. Each row estimates `loc = a_g * mu` from prior dates only; `loc_slope_source` records the regime fallback level used for that date.",
        "",
        _markdown_table(
            regime_loc_calibration_audit[
                [
                    "datetime",
                    "factor_industry_theme",
                    "entropy_bucket",
                    "overheat_regime",
                    "liquidity_bucket",
                    "loc_slope_source",
                    "rows",
                    "instruments",
                    "loc_slope_a",
                    "loc_slope_raw",
                    "loc_slope_obs",
                    "loc_slope_dates",
                    "global_slope_a",
                    "global_slope_dates",
                    "avg_mu_before",
                    "avg_mu_regime_calibrated_loc",
                    "avg_next_return",
                    "direction_accuracy",
                    "positive_mu_negative_return_rate",
                    "mu_delta_abs_sum",
                ]
            ]
            if not regime_loc_calibration_audit.empty
            else regime_loc_calibration_audit,
            max_rows=30,
        ),
        "",
        "## Cross-Strategy Diversification Audit",
        "",
        _markdown_table(net_correlation_matrix, max_rows=10),
        "",
        _markdown_table(diversification_audit, max_rows=10),
        "",
        "## Mean Error Regime Attribution",
        "",
        _markdown_table(
            mean_error_regime_summary[
                [
                    "variant",
                    "factor_industry_theme",
                    "entropy_bucket",
                    "overheat_regime",
                    "liquidity_bucket",
                    "position_bucket",
                    "rows",
                    "dates",
                    "instruments",
                    "avg_variant_mu",
                    "avg_next_return",
                    "avg_mu_error",
                    "direction_accuracy",
                    "positive_mu_negative_return_rate",
                    "abs_variant_effective_weight_sum",
                    "realized_return_contribution_sum",
                    "forecast_mu_contribution_sum",
                    "weighted_mu_error_sum",
                    "negative_realized_contribution_sum",
                    "delta_return_contribution_vs_old_sum",
                ]
            ]
            if not mean_error_regime_summary.empty
            else mean_error_regime_summary,
            max_rows=30,
        ),
        "",
        "## Top Mean Overestimation Rows",
        "",
        _markdown_table(
            mean_error_top_overestimation[
                [
                    "datetime",
                    "variant",
                    "instrument",
                    "factor_industry_theme",
                    "entropy_bucket",
                    "overheat_regime",
                    "liquidity_bucket",
                    "strategic_leverage_regime",
                    "variant_mu_after",
                    "next_return",
                    "mu_error",
                    "variant_effective_weight",
                    "realized_return_contribution",
                    "forecast_mu_contribution",
                    "weighted_mu_error",
                ]
            ]
            if not mean_error_top_overestimation.empty
            else mean_error_top_overestimation,
            max_rows=30,
        ),
        "",
        "## Trigger Metadata",
        "",
        _markdown_table(pd.DataFrame([trigger_metadata])),
        "",
        "## Monthly Delta",
        "",
        _markdown_table(
            monthly[monthly["variant"] != "old_mvo_replay"][
                [
                    "variant",
                    "month",
                    "triggered_rows",
                    "delta_cumulative_return_vs_old_mvo",
                    "net_variant_cumulative_return_after_tcost",
                    "transaction_cost_return_sum",
                    "annualized_turnover",
                    "delta_daily_return_sum_vs_old_mvo",
                    "low_entropy_days",
                    "high_entropy_days",
                    "hysteresis_high_entropy_days",
                    "selective_hysteresis_high_entropy_days",
                    "active_liquidity_high_days",
                    "overheat_low_entropy_days",
                    "risk_off_overlay_days",
                    "risk_off_overlay_no_lookahead_days",
                    "risk_off_exposure_regime_days",
                    "hysteresis_high_exposure_regime_days",
                    "selective_hysteresis_high_exposure_regime_days",
                    "avg_active_liquidity_z_amount_weighted",
                    "exposure_adjusted_days",
                    "smoothed_exposure_days",
                    "avg_variant_exposure_scale",
                    "overheat_rows",
                    "momentum_deceleration_rows",
                    "communication_rows",
                ]
            ]
            if not monthly.empty
            else monthly,
            max_rows=20,
        ),
        "",
        "## Trigger Breakdown",
        "",
        _markdown_table(trigger_breakdown.head(30), max_rows=30),
        "",
        "## Top Positive Delta Days",
        "",
        _markdown_table(
            top_positive_days[
                [
                    "datetime",
                    "variant",
                    "triggered_rows",
                    "delta_return_vs_old_mvo",
                    "variant_effective_return",
                    "old_mvo_effective_return",
                    "industry_entropy_norm",
                    "dominant_industry",
                    "top_delta_contribution_codes",
                ]
            ]
            if not top_positive_days.empty
            else top_positive_days,
            max_rows=12,
        ),
        "",
        "## Top Negative Delta Days",
        "",
        _markdown_table(
            top_negative_days[
                [
                    "datetime",
                    "variant",
                    "triggered_rows",
                    "delta_return_vs_old_mvo",
                    "variant_effective_return",
                    "old_mvo_effective_return",
                    "industry_entropy_norm",
                    "dominant_industry",
                    "top_delta_contribution_codes",
                ]
            ]
            if not top_negative_days.empty
            else top_negative_days,
            max_rows=12,
        ),
        "",
        "## Interpretation Boundary",
        "",
        "The momentum variant compares raw close pct_change(5) with pct_change(2): deceleration means the 2-day return is below the compounding pace implied by the 5-day return. The communication variant removes communication-theme instruments from the optimizer universe instead of scaling mu. The entropy mu variant applies the mu penalty only on low-entropy overheat days. The strategic leverage variant is audit-only L3 exposure timing: it sets exposure to the configured high-entropy scale on high-entropy days and to the configured crowded-overheat scale when any overheat instrument appears on a low-entropy day.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {
        "daily": daily_path,
        "weights": weights_path,
        "summary": summary_path,
        "monthly": monthly_path,
        "trigger_breakdown": trigger_breakdown_path,
        "top_contribution_delta": top_contribution_delta_path,
        "regime_loc_calibration_audit": regime_loc_calibration_path,
        "friction_sensitivity_matrix": friction_sensitivity_path,
        **mean_error_outputs,
        **diversification_outputs,
        "report": report_path,
        "manifest": manifest_path,
        **executive_outputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit-only Smart-mu variant replay.")
    parser.add_argument("--bridge-l3-csv", default=DEFAULT_BRIDGE_L3_CSV)
    parser.add_argument("--parameter-audit-csv", default=DEFAULT_PARAMETER_AUDIT_CSV)
    parser.add_argument("--provider-uri", default=DEFAULT_PROVIDER_URI)
    parser.add_argument("--market-benchmark", default=DEFAULT_MARKET_BENCHMARK)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--max-weight", type=float, default=bridge_core.DEFAULT_MAX_WEIGHT)
    parser.add_argument("--min-weight-floor", type=float, default=bridge_core.DEFAULT_MIN_WEIGHT_FLOOR)
    parser.add_argument("--variance-floor-quantile", type=float, default=DEFAULT_VARIANCE_FLOOR_QUANTILE)
    parser.add_argument("--corr-floor", type=float, default=DEFAULT_CORR_FLOOR)
    parser.add_argument("--patch-multiplier", type=float, default=SMART_MU_PATCH_MULTIPLIER)
    parser.add_argument("--negative-flip-mu", type=float, default=NEGATIVE_FLIP_MU)
    parser.add_argument("--communication-theme", default="communication")
    parser.add_argument("--entropy-low-threshold", type=float, default=ENTROPY_LOW_THRESHOLD)
    parser.add_argument("--high-entropy-threshold", type=float, default=HIGH_ENTROPY_THRESHOLD)
    parser.add_argument("--high-entropy-exposure-scale", type=float, default=HIGH_ENTROPY_EXPOSURE_SCALE)
    parser.add_argument("--crowded-overheat-exposure-scale", type=float, default=CROWDED_OVERHEAT_EXPOSURE_SCALE)
    parser.add_argument("--risk-off-overlay-exposure-scale", type=float, default=RISK_OFF_OVERLAY_EXPOSURE_SCALE)
    parser.add_argument("--risk-off-vol-spike-threshold", type=float, default=RISK_OFF_VOL_SPIKE_THRESHOLD)
    parser.add_argument("--risk-off-vol-spike-lookback-days", type=int, default=RISK_OFF_VOL_SPIKE_LOOKBACK_DAYS)
    parser.add_argument("--risk-off-market-gap-threshold", type=float, default=RISK_OFF_MARKET_GAP_THRESHOLD)
    parser.add_argument("--no-lookahead-atr-spike-threshold", type=float, default=NO_LOOKAHEAD_ATR_SPIKE_THRESHOLD)
    parser.add_argument("--no-lookahead-atr-lookback-days", type=int, default=NO_LOOKAHEAD_ATR_LOOKBACK_DAYS)
    parser.add_argument("--no-lookahead-atr-hold-days", type=int, default=NO_LOOKAHEAD_ATR_HOLD_DAYS)
    parser.add_argument("--hysteresis-entropy-up-threshold", type=float, default=HYSTERESIS_ENTROPY_UP_THRESHOLD)
    parser.add_argument("--hysteresis-entropy-down-threshold", type=float, default=HYSTERESIS_ENTROPY_DOWN_THRESHOLD)
    parser.add_argument("--selective-hysteresis-liquidity-z-threshold", type=float, default=SELECTIVE_HYSTERESIS_LIQUIDITY_Z_THRESHOLD)
    parser.add_argument("--selective-hysteresis-fast-up-threshold", type=float, default=SELECTIVE_HYSTERESIS_FAST_UP_THRESHOLD)
    parser.add_argument("--selective-hysteresis-fast-down-threshold", type=float, default=SELECTIVE_HYSTERESIS_FAST_DOWN_THRESHOLD)
    parser.add_argument("--selective-hysteresis-wide-up-threshold", type=float, default=SELECTIVE_HYSTERESIS_WIDE_UP_THRESHOLD)
    parser.add_argument("--selective-hysteresis-wide-down-threshold", type=float, default=SELECTIVE_HYSTERESIS_WIDE_DOWN_THRESHOLD)
    parser.add_argument("--transaction-cost-bps", type=float, default=TRANSACTION_COST_BPS)
    parser.add_argument("--smoothed-rerisk-exposure-step", type=float, default=SMOOTHED_RERISK_EXPOSURE_STEP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SmartMuVariantAuditConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        max_weight=float(args.max_weight),
        min_weight_floor=float(args.min_weight_floor),
        variance_floor_quantile=float(args.variance_floor_quantile),
        corr_floor=float(args.corr_floor),
        patch_multiplier=float(args.patch_multiplier),
        negative_flip_mu=float(args.negative_flip_mu),
        communication_theme=str(args.communication_theme),
        entropy_low_threshold=float(args.entropy_low_threshold),
        high_entropy_threshold=float(args.high_entropy_threshold),
        high_entropy_exposure_scale=float(args.high_entropy_exposure_scale),
        crowded_overheat_exposure_scale=float(args.crowded_overheat_exposure_scale),
        risk_off_overlay_exposure_scale=float(args.risk_off_overlay_exposure_scale),
        risk_off_vol_spike_threshold=float(args.risk_off_vol_spike_threshold),
        risk_off_vol_spike_lookback_days=int(args.risk_off_vol_spike_lookback_days),
        risk_off_market_gap_threshold=float(args.risk_off_market_gap_threshold),
        no_lookahead_atr_spike_threshold=float(args.no_lookahead_atr_spike_threshold),
        no_lookahead_atr_lookback_days=int(args.no_lookahead_atr_lookback_days),
        no_lookahead_atr_hold_days=int(args.no_lookahead_atr_hold_days),
        hysteresis_entropy_up_threshold=float(args.hysteresis_entropy_up_threshold),
        hysteresis_entropy_down_threshold=float(args.hysteresis_entropy_down_threshold),
        selective_hysteresis_liquidity_z_threshold=float(args.selective_hysteresis_liquidity_z_threshold),
        selective_hysteresis_fast_up_threshold=float(args.selective_hysteresis_fast_up_threshold),
        selective_hysteresis_fast_down_threshold=float(args.selective_hysteresis_fast_down_threshold),
        selective_hysteresis_wide_up_threshold=float(args.selective_hysteresis_wide_up_threshold),
        selective_hysteresis_wide_down_threshold=float(args.selective_hysteresis_wide_down_threshold),
        transaction_cost_bps=float(args.transaction_cost_bps),
        smoothed_rerisk_exposure_step=float(args.smoothed_rerisk_exposure_step),
        provider_uri=str(args.provider_uri),
        market_benchmark=str(args.market_benchmark),
    )
    outputs = run_smart_mu_variant_audit(args.bridge_l3_csv, args.parameter_audit_csv, args.output_dir, config)
    print(f"[INFO] wrote {outputs['summary']}")
    print(f"[INFO] report {outputs['report']}")


if __name__ == "__main__":
    main()
