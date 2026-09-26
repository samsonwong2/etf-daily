"""Post-processing risk layers: L2 cap + HHI audit + L3 regime/vol/cluster + anchor overlay."""
from __future__ import annotations

from pathlib import Path

import qlib
from qlib.data import D

from . import constants
from .constants import DEFAULT_FUND_LIST_CSV
from .benchmark_regime import (
    BenchmarkRegimeConfig,
    build_macro_budget_series,
    compute_benchmark_features,
    load_hs300_close,
    vol_target_for_macro_budget,
)
from .price_adjustments import (
    auto_qfq_adjust_close_panel,
    auto_qfq_adjust_price_columns_by_group,
)


def apply_risk_controls(
    bridge_csv_path: Path,
    provider_uri: str,
    bridge_l3_csv_path: Path | None = None,
    vol_target: float = 0.0,
    vol_target_risk_on: float = 0.0,
    vol_target_risk_off: float = 0.0,
    regime_gate_min_scale: float = 1.0,
    regime_gate_ma: int = 20,
    regime_gate_mode: str = "binary",
    macro_budget_min: float = 0.50,
    macro_budget_trend_fast_ma: int = 20,
    macro_budget_trend_slow_ma: int = 60,
    macro_budget_vol_window: int = 20,
    macro_budget_vol_lookback: int = 120,
    vol_target_offensive: float = 0.0,
    vol_target_neutral: float = 0.0,
    vol_target_defensive: float = 0.0,
    max_weight_cap: float = 0.0,
    min_holdings_audit_warn: float = 0.0,
    vol_target_cov_mode: str = "full",
    anchor_codes: list[str] | None = None,
    anchor_weight_total: float = 0.0,
    raw_panel_path: Path | None = None,
    cluster_gate_rho: float = 0.0,
    cluster_gate_rho_saturation: float = 0.9,
    cluster_gate_min_scale: float = 0.5,
    cluster_gate_source: str = "forecast_cov",
    cluster_gate_realized_lookback: int = 10,
    cluster_gate_realized_min_periods: int = 5,
    signal_quality_gate_mode: str = "none",
    signal_quality_ic_window: int = 5,
    signal_quality_ic_min_periods: int = 3,
    signal_quality_ic_threshold: float = -0.05,
    signal_quality_min_scale: float = 0.5,
    signal_quality_min_scale_risk_on: float = 0.0,
    signal_quality_min_scale_risk_off: float = 0.0,
    trend_sleeve_weight_total: float = 0.0,
    trend_sleeve_weight_total_risk_on: float = -1.0,
    trend_sleeve_weight_total_risk_off: float = -1.0,
    trend_sleeve_top_k: int = 4,
    trend_sleeve_top_k_risk_on: int = 0,
    trend_sleeve_top_k_risk_off: int = 0,
    trend_sleeve_momentum_windows: list[int] | None = None,
    trend_sleeve_mu_weight: float = 0.65,
    trend_sleeve_min_score: float = 0.0,
    trend_sleeve_warmup_days: int = 0,
    trend_sleeve_exposure_floor: float = 0.0,
    trend_sleeve_exposure_floor_risk_on: float = -1.0,
    trend_sleeve_exposure_floor_risk_off: float = -1.0,
    factor_fund_list_csv: str | Path | None = DEFAULT_FUND_LIST_CSV,
) -> None:
    """Post-process forecast_w in the bridge CSV with three independent risk layers.

    Layer 2 (hard constraints, applied first):
      - max_weight_cap: cap per-asset weight via capped-simplex projection
        (total weight preserved; weights re-distributed to unconstrained assets).
      - min_holdings_audit_warn: audit-only, prints warnings when 1/HHI drops below
        the threshold on any day. NOTE: this is a soft warn, NOT a hard constraint;
        it does not modify weights.

    Layer 3 (post-processing, applied after):
      - Regime gate: HS300 close / MA(regime_gate_ma); binary scale
        (1.0 above MA, regime_gate_min_scale below).
    - Vol-targeting: scale weights so portfolio daily std ≤ vol_target.
    - Momentum/trend sleeve: optional pre-cap weight reserve for fast model+momentum names.

    Phase 4 anchor overlay (applied FIRST, before cap / vol-target):
      - Original MVO weights scaled by (1 - anchor_weight_total)
      - Anchor rows injected with fixed weight = anchor_weight_total / N_active_anchors
      - Anchors get a valid forecast_variance estimated from rolling 252d realized
        variance (via raw_panel_path) so vol-targeting properly accounts for them.

    L3 de-leveraging is applied multiplicatively via exposure_scale; pre-cap overlays
    modify forecast_w and preserve each day's total raw weight.
    """
    import numpy as np
    import pandas as pd

    vol_target_enabled = max(
        float(vol_target),
        float(vol_target_risk_on),
        float(vol_target_risk_off),
    ) > 0.0
    vol_target_dynamic_enabled = (
        float(vol_target_risk_on) > 0.0 or float(vol_target_risk_off) > 0.0
    )
    regime_gate_mode = str(regime_gate_mode or "binary").strip().lower()
    budget_v2_enabled = regime_gate_mode == "budget_v2"
    macro_budget_min = float(np.clip(float(macro_budget_min), 0.0, 1.0))
    regime_gate_enabled = budget_v2_enabled or float(regime_gate_min_scale) < 1.0
    if budget_v2_enabled:
        vol_target_enabled = max(
            float(vol_target),
            float(vol_target_risk_on),
            float(vol_target_risk_off),
            float(vol_target_offensive),
            float(vol_target_neutral),
            float(vol_target_defensive),
        ) > 0.0
    weight_cap_enabled = float(max_weight_cap) > 0.0
    hhi_audit_enabled = float(min_holdings_audit_warn) > 0.0
    anchor_codes_active = [str(c).upper() for c in (anchor_codes or [])]
    anchor_overlay_enabled = bool(anchor_codes_active) and float(anchor_weight_total) > 0.0
    factor_export_enabled = raw_panel_path is not None and Path(raw_panel_path).exists()
    cluster_gate_source = str(cluster_gate_source or "forecast_cov").lower()
    signal_quality_gate_mode = str(signal_quality_gate_mode or "none").lower()
    signal_quality_min_scale = float(np.clip(float(signal_quality_min_scale), 0.0, 1.0))
    signal_quality_min_scale_risk_on = float(np.clip(float(signal_quality_min_scale_risk_on), 0.0, 1.0))
    signal_quality_min_scale_risk_off = float(np.clip(float(signal_quality_min_scale_risk_off), 0.0, 1.0))
    signal_quality_effective_min_scales = [signal_quality_min_scale]
    if signal_quality_min_scale_risk_on > 0.0:
        signal_quality_effective_min_scales.append(signal_quality_min_scale_risk_on)
    if signal_quality_min_scale_risk_off > 0.0:
        signal_quality_effective_min_scales.append(signal_quality_min_scale_risk_off)
    cluster_gate_enabled = (
        float(cluster_gate_rho) > 0.0
        and float(cluster_gate_min_scale) < 1.0
        and float(cluster_gate_rho_saturation) > float(cluster_gate_rho)
    )
    signal_quality_gate_enabled = (
        signal_quality_gate_mode == "rolling_ic"
        and min(signal_quality_effective_min_scales) < 1.0
        and int(signal_quality_ic_window) > 0
    )
    trend_sleeve_windows = [
        int(w) for w in (trend_sleeve_momentum_windows or [3, 5, 10]) if int(w) > 0
    ]
    trend_sleeve_weight_total = float(np.clip(float(trend_sleeve_weight_total), 0.0, 1.0))
    trend_sleeve_weight_total_risk_on = float(trend_sleeve_weight_total_risk_on)
    trend_sleeve_weight_total_risk_off = float(trend_sleeve_weight_total_risk_off)
    trend_sleeve_top_k = int(trend_sleeve_top_k)
    trend_sleeve_top_k_risk_on = int(trend_sleeve_top_k_risk_on)
    trend_sleeve_top_k_risk_off = int(trend_sleeve_top_k_risk_off)
    trend_sleeve_warmup_days = max(0, int(trend_sleeve_warmup_days))
    trend_sleeve_exposure_floor = float(np.clip(float(trend_sleeve_exposure_floor), 0.0, 1.0))
    trend_sleeve_exposure_floor_risk_on = float(trend_sleeve_exposure_floor_risk_on)
    trend_sleeve_exposure_floor_risk_off = float(trend_sleeve_exposure_floor_risk_off)
    trend_sleeve_dynamic_enabled = (
        trend_sleeve_weight_total_risk_on >= 0.0
        or trend_sleeve_weight_total_risk_off >= 0.0
        or trend_sleeve_top_k_risk_on > 0
        or trend_sleeve_top_k_risk_off > 0
        or trend_sleeve_exposure_floor_risk_on >= 0.0
        or trend_sleeve_exposure_floor_risk_off >= 0.0
    )
    trend_sleeve_exposure_floor_enabled = (
        trend_sleeve_exposure_floor > 0.0
        or trend_sleeve_exposure_floor_risk_on >= 0.0
        or trend_sleeve_exposure_floor_risk_off >= 0.0
    )
    trend_sleeve_weight_candidates = [trend_sleeve_weight_total]
    if trend_sleeve_weight_total_risk_on >= 0.0:
        trend_sleeve_weight_candidates.append(
            float(np.clip(trend_sleeve_weight_total_risk_on, 0.0, 1.0))
        )
    if trend_sleeve_weight_total_risk_off >= 0.0:
        trend_sleeve_weight_candidates.append(
            float(np.clip(trend_sleeve_weight_total_risk_off, 0.0, 1.0))
        )
    trend_sleeve_mu_weight = float(np.clip(float(trend_sleeve_mu_weight), 0.0, 1.0))
    trend_sleeve_enabled = (
        max(trend_sleeve_weight_candidates) > 0.0
        and trend_sleeve_top_k > 0
        and bool(trend_sleeve_windows)
    )

    trend_sleeve_desc = "disabled"
    if trend_sleeve_enabled:
        trend_sleeve_desc = (
            "%.3f@top%d mom=%s mu_weight=%.2f"
            % (
                trend_sleeve_weight_total,
                trend_sleeve_top_k,
                ",".join(map(str, trend_sleeve_windows)),
                trend_sleeve_mu_weight,
            )
        )
        if trend_sleeve_dynamic_enabled:
            risk_on_weight = (
                trend_sleeve_weight_total
                if trend_sleeve_weight_total_risk_on < 0.0
                else float(np.clip(trend_sleeve_weight_total_risk_on, 0.0, 1.0))
            )
            risk_off_weight = (
                trend_sleeve_weight_total
                if trend_sleeve_weight_total_risk_off < 0.0
                else float(np.clip(trend_sleeve_weight_total_risk_off, 0.0, 1.0))
            )
            risk_on_top_k = trend_sleeve_top_k_risk_on if trend_sleeve_top_k_risk_on > 0 else trend_sleeve_top_k
            risk_off_top_k = trend_sleeve_top_k_risk_off if trend_sleeve_top_k_risk_off > 0 else trend_sleeve_top_k
            risk_on_floor = (
                trend_sleeve_exposure_floor
                if trend_sleeve_exposure_floor_risk_on < 0.0
                else float(np.clip(trend_sleeve_exposure_floor_risk_on, 0.0, 1.0))
            )
            risk_off_floor = (
                trend_sleeve_exposure_floor
                if trend_sleeve_exposure_floor_risk_off < 0.0
                else float(np.clip(trend_sleeve_exposure_floor_risk_off, 0.0, 1.0))
            )
            trend_sleeve_desc += (
                " (risk_on=%.3f@top%d floor=%.2f, risk_off=%.3f@top%d floor=%.2f)"
                % (risk_on_weight, risk_on_top_k, risk_on_floor, risk_off_weight, risk_off_top_k, risk_off_floor)
            )
        elif trend_sleeve_exposure_floor_enabled:
            trend_sleeve_desc += " (floor=%.2f)" % trend_sleeve_exposure_floor
        if trend_sleeve_warmup_days > 0:
            trend_sleeve_desc += " warmup=%dd" % trend_sleeve_warmup_days

    if not any([
        vol_target_enabled,
        regime_gate_enabled,
        weight_cap_enabled,
        hhi_audit_enabled,
        anchor_overlay_enabled,
        cluster_gate_enabled,
        signal_quality_gate_enabled,
        trend_sleeve_enabled,
        factor_export_enabled,
    ]):
        return

    if constants._DRY_RUN:
        print(
            f"[DRY-RUN] Skipping apply_risk_controls "
            f"(would read {bridge_csv_path} and write {bridge_l3_csv_path or bridge_csv_path})"
        )
        return

    print(
        f"[INFO] apply_risk_controls: "
        f"anchor_overlay={'%s@%.3f' % (','.join(anchor_codes_active), anchor_weight_total) if anchor_overlay_enabled else 'disabled'}, "
        f"factor_export={'enabled' if factor_export_enabled else 'disabled'}, "
        f"trend_sleeve={trend_sleeve_desc}, "
        f"max_weight_cap={'%.3f' % max_weight_cap if weight_cap_enabled else 'disabled'}, "
        f"min_holdings_audit_warn={'%.1f' % min_holdings_audit_warn if hhi_audit_enabled else 'disabled'}, "
        f"cluster_gate={'%s:ρ>%.2f→%.2f@ρ=%.2f' % (cluster_gate_source, cluster_gate_rho, cluster_gate_min_scale, cluster_gate_rho_saturation) if cluster_gate_enabled else 'disabled'}, "
        f"signal_quality_gate={'rolling_ic%d<%.3f→%.2f%s' % (signal_quality_ic_window, signal_quality_ic_threshold, signal_quality_min_scale, (' (risk_on=%.2f, risk_off=%.2f)' % (signal_quality_min_scale_risk_on or signal_quality_min_scale, signal_quality_min_scale_risk_off or signal_quality_min_scale) if (signal_quality_min_scale_risk_on > 0.0 or signal_quality_min_scale_risk_off > 0.0) else '')) if signal_quality_gate_enabled else 'disabled'}, "
        f"regime_gate_min_scale={regime_gate_min_scale if regime_gate_enabled else 'disabled'} (ma={regime_gate_ma}), "
        f"vol_target={'%.4f' % vol_target if vol_target_enabled else 'disabled'}"
        f"{' (risk_on=%.4f, risk_off=%.4f)' % (vol_target_risk_on, vol_target_risk_off) if vol_target_dynamic_enabled else ''} "
        f"(cov_mode={vol_target_cov_mode})"
    )

    # ── Inline capped-simplex projection (adapted from compare_weight_overlay_variants.py) ──
    def _project_capped(values: np.ndarray, target_sum: float, cap: float) -> np.ndarray:
        """Project a non-negative vector onto {w: sum(w)=target_sum, 0<=w<=cap}.

        Preserves ranking and total weight. Uses bisection on shift parameter.
        """
        if target_sum <= 0 or cap <= 0:
            return np.zeros_like(values, dtype=float)
        v = np.clip(values.astype(float), 0.0, None)
        n = len(v)
        max_sum = cap * n
        if max_sum + 1e-12 < target_sum:
            # Infeasible (cap too tight for this total). Fall back to uniform at cap.
            return np.full(n, min(cap, target_sum / n), dtype=float)
        # Fast path: already feasible after simple clip.
        clipped = np.clip(v, 0.0, cap)
        if float(clipped.sum()) <= target_sum + 1e-12:
            residual = target_sum - float(clipped.sum())
            if residual > 1e-12:
                room = cap - clipped
                for idx in np.argsort(-room):
                    if residual <= 1e-12:
                        break
                    add = min(room[idx], residual)
                    clipped[idx] += add
                    residual -= add
            return clipped
        # Bisect on shift m such that sum(clip(v - m, 0, cap)) == target_sum.
        lo = float(v.min() - cap)
        hi = float(v.max())
        for _ in range(100):
            mid = (lo + hi) / 2.0
            projected = np.clip(v - mid, 0.0, cap)
            total = projected.sum()
            if abs(total - target_sum) <= 1e-10:
                break
            if total > target_sum:
                lo = mid
            else:
                hi = mid
        projected = np.clip(v - (lo + hi) / 2.0, 0.0, cap)
        total = float(projected.sum())
        if total != 0:
            projected *= target_sum / total
        projected = np.clip(projected, 0.0, cap)
        # Fix small rounding residual.
        residual = target_sum - float(projected.sum())
        if abs(residual) > 1e-8:
            room = cap - projected
            for idx in np.argsort(-room):
                if residual <= 1e-10:
                    break
                add = min(room[idx], residual)
                projected[idx] += add
                residual -= add
        return projected

    df = pd.read_csv(bridge_csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["instrument"] = df["instrument"].astype(str).str.upper()
    dates = sorted(df["datetime"].unique())

    def _robust_z(values: pd.Series) -> pd.Series:
        series = pd.to_numeric(values, errors="coerce")
        valid = series.dropna()
        if valid.empty:
            return pd.Series(np.nan, index=series.index, dtype=float)
        median = float(valid.median())
        mad = float((valid - median).abs().median())
        if not np.isfinite(mad) or mad <= 1e-12:
            std = float(valid.std(ddof=0))
            if not np.isfinite(std) or std <= 1e-12:
                return pd.Series(0.0, index=series.index, dtype=float).where(series.notna(), np.nan)
            scale = std
        else:
            scale = 1.4826 * mad
        return ((series - median) / scale).clip(-3.0, 3.0)

    def _normalize_factor_code(value: object) -> str:
        return str(value).strip().upper()

    def _load_factor_name_map(path_value: str | Path | None) -> dict[str, str]:
        if path_value is None:
            return {}
        path = Path(path_value).expanduser()
        if not path.exists():
            return {}
        try:
            frame = pd.read_csv(path)
        except Exception:
            return {}
        code_col = next((c for c in ["基金代码", "code", "instrument", "symbol"] if c in frame.columns), None)
        name_col = next((c for c in ["基金简称", "基金名称", "name", "short_name"] if c in frame.columns), None)
        if code_col is None or name_col is None:
            return {}
        return {
            _normalize_factor_code(row[code_col]): str(row[name_col])
            for _, row in frame[[code_col, name_col]].dropna(subset=[code_col]).iterrows()
        }

    def _factor_industry_theme(name: object) -> str:
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

    if factor_export_enabled:
        try:
            raw_for_factors = pd.read_csv(raw_panel_path, parse_dates=["datetime"])
            raw_for_factors["instrument"] = raw_for_factors["instrument"].astype(str).str.upper()
            raw_for_factors = raw_for_factors.sort_values(["instrument", "datetime"])
            if "close" not in raw_for_factors.columns:
                print("[WARN] factor export skipped: raw panel lacks close column")
            else:
                raw_for_factors["close"] = pd.to_numeric(raw_for_factors["close"], errors="coerce")
                for window in [5, 20, 60]:
                    raw_for_factors[f"factor_return_{window}d"] = raw_for_factors.groupby("instrument")["close"].transform(
                        lambda series: series.pct_change(window, fill_method=None).shift(1)
                    )
                raw_for_factors["factor_style_momentum_raw"] = raw_for_factors[
                    ["factor_return_20d", "factor_return_60d"]
                ].mean(axis=1, skipna=True)
                raw_for_factors["factor_style_growth_raw"] = (
                    0.35 * raw_for_factors["factor_return_20d"] + 0.65 * raw_for_factors["factor_return_60d"]
                )

                volume_available = "volume" in raw_for_factors.columns
                if volume_available:
                    raw_for_factors["volume"] = pd.to_numeric(raw_for_factors["volume"], errors="coerce")
                    raw_for_factors["factor_dollar_volume"] = raw_for_factors["close"].abs() * raw_for_factors["volume"].clip(lower=0.0)
                    raw_for_factors["factor_dollar_volume"] = raw_for_factors["factor_dollar_volume"].where(
                        raw_for_factors["factor_dollar_volume"] > 0.0,
                        raw_for_factors["volume"].clip(lower=0.0),
                    )
                    raw_for_factors["factor_volume_ma20"] = raw_for_factors.groupby("instrument")["factor_dollar_volume"].transform(
                        lambda series: series.shift(1).rolling(window=20, min_periods=5).mean()
                    )
                    raw_for_factors["factor_volume_ratio_20"] = raw_for_factors["factor_dollar_volume"] / raw_for_factors[
                        "factor_volume_ma20"
                    ]
                    raw_for_factors["factor_style_size_raw"] = np.log1p(raw_for_factors["factor_volume_ma20"])
                else:
                    raw_for_factors["factor_volume_ratio_20"] = np.nan
                    raw_for_factors["factor_style_size_raw"] = np.nan

                name_map = _load_factor_name_map(factor_fund_list_csv)
                raw_for_factors["factor_industry_theme"] = raw_for_factors["instrument"].map(name_map).map(_factor_industry_theme)
                raw_for_factors["factor_industry_theme"] = raw_for_factors["factor_industry_theme"].fillna("other")
                raw_for_factors["factor_industry_momentum_raw"] = raw_for_factors.groupby(
                    ["datetime", "factor_industry_theme"]
                )["factor_style_momentum_raw"].transform("mean")

                factor_feature_columns = [
                    "instrument",
                    "datetime",
                    "factor_industry_theme",
                    "factor_style_momentum_raw",
                    "factor_style_growth_raw",
                    "factor_style_size_raw",
                    "factor_industry_momentum_raw",
                    "factor_volume_ratio_20",
                ]
                factor_features = raw_for_factors[factor_feature_columns].drop_duplicates(
                    ["instrument", "datetime"], keep="last"
                )
                df = df.merge(factor_features, on=["instrument", "datetime"], how="left", suffixes=("", "_factor_export"))
                for column in [item for item in factor_feature_columns if item not in {"instrument", "datetime"}]:
                    export_column = f"{column}_factor_export"
                    if export_column not in df.columns:
                        continue
                    if column in df.columns:
                        if column == "factor_industry_theme":
                            current = df[column].where(df[column].notna() & (df[column].astype(str).str.len() > 0))
                            df[column] = current.fillna(df[export_column])
                        else:
                            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(
                                pd.to_numeric(df[export_column], errors="coerce")
                            )
                    else:
                        df[column] = df[export_column]
                    df = df.drop(columns=[export_column])
                if "factor_volume_ratio_20" not in df.columns:
                    df["factor_volume_ratio_20"] = 1.0
                ratio_values = pd.to_numeric(df["factor_volume_ratio_20"], errors="coerce").replace([np.inf, -np.inf], np.nan)
                df["factor_volume_ratio_20"] = ratio_values.mask(ratio_values <= 0.0).fillna(1.0)
                factor_z_map = {
                    "factor_style_momentum_raw": "factor_style_momentum",
                    "factor_style_growth_raw": "factor_style_growth",
                    "factor_style_size_raw": "factor_style_size",
                    "factor_industry_momentum_raw": "factor_industry_momentum",
                    "factor_volume_ratio_20": "factor_volume_z_20",
                }
                for raw_col, z_col in factor_z_map.items():
                    if raw_col not in df.columns:
                        df[raw_col] = np.nan
                    df[z_col] = df.groupby("datetime")[raw_col].transform(_robust_z).fillna(0.0)
                df["factor_abnormal_volume_flag"] = (
                    (pd.to_numeric(df["factor_volume_ratio_20"], errors="coerce") >= 1.5)
                    | (pd.to_numeric(df["factor_volume_z_20"], errors="coerce") >= 1.0)
                ).astype(int)
                print(
                    "[INFO] factor export: wrote explicit factor_style_momentum/growth/size "
                    "and factor_industry_momentum columns to L3 output"
                )
        except Exception as exc:
            print(f"[WARN] factor export skipped: {exc}")

    signal_quality_daily_ic = pd.Series(dtype=float)
    signal_quality_rolling_ic = pd.Series(dtype=float)
    if signal_quality_gate_enabled:
        required_cols = {"forecast_mean_return", "next_return"}
        if not required_cols.issubset(df.columns):
            print(
                "[WARN] signal-quality gate disabled: bridge CSV lacks "
                f"{sorted(required_cols - set(df.columns))}"
            )
            signal_quality_gate_enabled = False
        else:
            daily_ic_values: dict = {}
            for current_dt, ic_day in df.groupby("datetime", sort=True):
                ic_frame = pd.DataFrame(
                    {
                        "mu": pd.to_numeric(ic_day["forecast_mean_return"], errors="coerce"),
                        "next_return": pd.to_numeric(ic_day["next_return"], errors="coerce"),
                    }
                ).dropna()
                if (
                    len(ic_frame) >= 5
                    and ic_frame["mu"].nunique() > 1
                    and ic_frame["next_return"].nunique() > 1
                ):
                    daily_ic_values[current_dt] = float(
                        ic_frame["mu"].corr(ic_frame["next_return"], method="spearman")
                    )
                else:
                    daily_ic_values[current_dt] = float("nan")
            signal_quality_daily_ic = pd.Series(daily_ic_values, dtype=float).sort_index()
            signal_quality_rolling_ic = (
                signal_quality_daily_ic.shift(1)
                .rolling(
                    window=int(signal_quality_ic_window),
                    min_periods=max(1, int(signal_quality_ic_min_periods)),
                )
                .mean()
            )

    realized_return_panel: pd.DataFrame | None = None
    if cluster_gate_enabled and cluster_gate_source == "realized":
        if raw_panel_path is None or not Path(raw_panel_path).exists():
            print(f"[WARN] realized cluster-gate disabled: raw_panel_path unavailable ({raw_panel_path})")
        else:
            raw_for_cluster = pd.read_csv(raw_panel_path, parse_dates=["datetime"])
            raw_for_cluster["instrument"] = raw_for_cluster["instrument"].astype(str).str.upper()
            raw_for_cluster = raw_for_cluster.sort_values(["instrument", "datetime"])
            if "return" in raw_for_cluster.columns:
                raw_for_cluster["return_for_corr"] = pd.to_numeric(
                    raw_for_cluster["return"], errors="coerce"
                )
            else:
                raw_for_cluster, _ = auto_qfq_adjust_price_columns_by_group(
                    raw_for_cluster,
                    group_col="instrument",
                    date_col="datetime",
                    price_cols=["close"],
                )
                raw_for_cluster["return_for_corr"] = raw_for_cluster.groupby("instrument")["close"].pct_change()
            if raw_for_cluster["return_for_corr"].notna().sum() == 0 and "close" in raw_for_cluster.columns:
                raw_for_cluster, _ = auto_qfq_adjust_price_columns_by_group(
                    raw_for_cluster,
                    group_col="instrument",
                    date_col="datetime",
                    price_cols=["close"],
                )
                raw_for_cluster["return_for_corr"] = raw_for_cluster.groupby("instrument")["close"].pct_change()
            realized_return_panel = (
                raw_for_cluster.pivot_table(
                    index="datetime",
                    columns="instrument",
                    values="return_for_corr",
                    aggfunc="last",
                )
                .sort_index()
            )

    def _weighted_offdiag_corr(corr_matrix: np.ndarray, aligned_weights: np.ndarray) -> float | None:
        if corr_matrix.shape[0] < 2:
            return None
        weight_outer = np.outer(aligned_weights, aligned_weights)
        off_diag_mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        valid_mask = off_diag_mask & np.isfinite(corr_matrix)
        denom = float(weight_outer[valid_mask].sum())
        if denom <= 1e-18:
            return None
        return float((corr_matrix[valid_mask] * weight_outer[valid_mask]).sum() / denom)

    def _rho_bar_from_forecast_cov(day_frame: pd.DataFrame, day_weights: np.ndarray) -> float | None:
        if "forecast_cov_matrix" not in day_frame.columns or "forecast_cov_order" not in day_frame.columns:
            return None
        instruments_day = day_frame["instrument"].astype(str).str.upper().tolist()
        cov_order_raw = day_frame["forecast_cov_order"].dropna().astype(str)
        cov_order = cov_order_raw.iloc[0].split("|") if len(cov_order_raw) > 0 else []
        if not cov_order:
            return None
        try:
            cov_lookup = dict(zip(instruments_day, day_frame["forecast_cov_matrix"].astype(str).tolist()))
            weight_by_inst = dict(zip(instruments_day, day_weights.tolist()))
            cov_rows: list[list[float]] = []
            for inst in cov_order:
                raw = cov_lookup.get(inst, "")
                if not raw or raw.lower() == "nan":
                    return None
                cov_rows.append([float(x) for x in raw.split(",")])
            if len(cov_rows) != len(cov_order):
                return None
            sigma = np.asarray(cov_rows, dtype=float)
            aligned_weights = np.asarray([weight_by_inst.get(inst, 0.0) for inst in cov_order], dtype=float)
            diag = np.sqrt(np.clip(np.diag(sigma), 1e-18, None))
            corr_matrix = sigma / np.outer(diag, diag)
            np.fill_diagonal(corr_matrix, 1.0)
            return _weighted_offdiag_corr(corr_matrix, aligned_weights)
        except (ValueError, TypeError):
            return None

    def _rho_bar_from_realized(day_frame: pd.DataFrame, day_weights: np.ndarray, current_dt) -> float | None:
        if realized_return_panel is None:
            return None
        active_mask = day_weights > 1e-12
        if int(active_mask.sum()) < 2:
            return None
        active_names = day_frame.loc[active_mask, "instrument"].astype(str).str.upper().tolist()
        active_weights = day_weights[active_mask]
        available = [name for name in active_names if name in realized_return_panel.columns]
        if len(available) < 2:
            return None
        weight_lookup = dict(zip(active_names, active_weights.tolist()))
        aligned_weights = np.asarray([weight_lookup[name] for name in available], dtype=float)
        history = realized_return_panel.loc[
            realized_return_panel.index < pd.Timestamp(current_dt), available
        ].tail(max(1, int(cluster_gate_realized_lookback)))
        if len(history) < max(2, int(cluster_gate_realized_min_periods)):
            return None
        corr_frame = history.corr(min_periods=max(2, int(cluster_gate_realized_min_periods)))
        corr_matrix = corr_frame.reindex(index=available, columns=available).to_numpy(dtype=float)
        np.fill_diagonal(corr_matrix, 1.0)
        return _weighted_offdiag_corr(corr_matrix, aligned_weights)

    # ── Phase 4: Anchor overlay — inject hedge reserve before cap / vol-target ──
    # Anchors are NOT in Route A / bridge universe. We inject rows here with:
    #  - forecast_w = anchor_weight_total / N_active_anchors
    #  - forecast_variance = rolling 252d realized variance (from raw_panel)
    #  - other cov columns left NaN (vol-target will fall back to diag for anchors)
    # Original rows are scaled by (1 - anchor_weight_total) to preserve total weight.
    if anchor_overlay_enabled:
        if raw_panel_path is None or not Path(raw_panel_path).exists():
            print(f"[WARN] anchor overlay disabled: raw_panel_path unavailable ({raw_panel_path})")
            anchor_overlay_enabled = False

    if anchor_overlay_enabled:
        raw_panel = pd.read_csv(raw_panel_path, parse_dates=["datetime"])
        raw_panel["instrument"] = raw_panel["instrument"].astype(str).str.upper()
        anchor_panel = raw_panel[raw_panel["instrument"].isin(anchor_codes_active)].copy()
        anchor_panel = anchor_panel.sort_values(["instrument", "datetime"]).reset_index(drop=True)
        if anchor_panel.empty:
            print(f"[WARN] anchor overlay disabled: no raw data for anchors {anchor_codes_active}")
            anchor_overlay_enabled = False

    if anchor_overlay_enabled:
        # Drop any existing anchor rows that bridge.py may have produced (bridge uses
        # raw_panel as its universe, which includes anchors). We take full control
        # over their weights here. BUT first record the day's total w (incl. anchors)
        # so we can preserve it when re-normalizing.
        df_inst_upper = df["instrument"].astype(str).str.upper()
        w_total_by_date: dict = {}
        for dt in dates:
            mask_dt = df["datetime"] == dt
            w_sum_full = float(
                df.loc[mask_dt, "forecast_w"].fillna(0.0).clip(lower=0.0).sum()
            )
            w_total_by_date[dt] = w_sum_full
        n_dropped = int(df_inst_upper.isin(anchor_codes_active).sum())
        if n_dropped > 0:
            df = df[~df_inst_upper.isin(anchor_codes_active)].reset_index(drop=True)
            dates = sorted(df["datetime"].unique())
            print(
                f"[INFO] anchor overlay: removed {n_dropped} pre-existing anchor rows "
                f"from bridge output; re-normalizing non-anchor weights to preserve "
                f"daily w_total before re-injecting fixed-share anchors."
            )
            # Re-normalize each day's non-anchor weights so that their sum equals
            # w_total_full * (1 - anchor_weight_total). This undoes bridge's
            # allocation to anchors and replaces it with our fixed overlay share.
            anchor_frac = float(anchor_weight_total)
            for dt in dates:
                mask_dt = df["datetime"] == dt
                w_nonanchor_sum = float(
                    df.loc[mask_dt, "forecast_w"].fillna(0.0).clip(lower=0.0).sum()
                )
                target_nonanchor_sum = w_total_by_date.get(dt, 0.0) * (1.0 - anchor_frac)
                if w_nonanchor_sum > 1e-12 and target_nonanchor_sum > 0:
                    rescale = target_nonanchor_sum / w_nonanchor_sum
                    df.loc[mask_dt, "forecast_w"] = (
                        df.loc[mask_dt, "forecast_w"].fillna(0.0) * rescale
                    )
        else:
            # No anchor pre-existing; just scale all original weights by (1 - frac)
            df["forecast_w"] = df["forecast_w"].fillna(0.0) * (1.0 - float(anchor_weight_total))

    if anchor_overlay_enabled:
        # Compute per-(instrument, datetime) realized 252d variance of close returns.
        anchor_panel, _ = auto_qfq_adjust_price_columns_by_group(
            anchor_panel,
            group_col="instrument",
            date_col="datetime",
            price_cols=["close"],
        )
        anchor_panel["ret"] = anchor_panel.groupby("instrument")["close"].pct_change()
        # Use shift(1) so variance at date t uses returns strictly before t.
        anchor_panel["ret_for_var"] = anchor_panel.groupby("instrument")["ret"].shift(1)
        anchor_panel["rolling_var"] = (
            anchor_panel.groupby("instrument")["ret_for_var"]
            .transform(lambda s: s.rolling(window=252, min_periods=20).var())
        )
        # Fallback for early dates: if NaN, use expanding variance with min_periods=5.
        anchor_panel["rolling_var"] = anchor_panel["rolling_var"].fillna(
            anchor_panel.groupby("instrument")["ret_for_var"].transform(
                lambda s: s.expanding(min_periods=5).var()
            )
        )
        # Final fallback: reasonable default (2% daily vol ≈ 0.0004).
        anchor_panel["rolling_var"] = anchor_panel["rolling_var"].fillna(0.0004)

        anchor_var_lookup = {
            (row.instrument, row.datetime): float(row.rolling_var)
            for row in anchor_panel.itertuples(index=False)
        }

        # Build anchor rows to append. Use w_total_by_date (from before the drop)
        # so anchors are sized relative to the day's original total allocation.
        new_rows = []
        anchor_frac = float(anchor_weight_total)
        n_anchors = len(anchor_codes_active)
        per_anchor_frac = anchor_frac / n_anchors
        for dt in dates:
            w_total_full = w_total_by_date.get(dt, 0.0)
            # If original total was already ~0 (e.g. fully risk-off day), don't inject.
            if w_total_full < 1e-9:
                continue
            anchor_budget_each = per_anchor_frac * w_total_full
            for code in anchor_codes_active:
                var_val = anchor_var_lookup.get((code, pd.Timestamp(dt)))
                if var_val is None or not np.isfinite(var_val) or var_val <= 0:
                    var_val = 0.0004  # safe default (~2% daily vol)
                new_rows.append(
                    {
                        "instrument": code,
                        "datetime": dt,
                        "forecast_w": anchor_budget_each,
                        "forecast_variance": var_val,
                        "forecast_variance_robust": var_val,
                        "forecast_mean_return": 0.0,
                        "forecast_mean_return_robust": 0.0,
                    }
                )

        if new_rows:
            anchor_df = pd.DataFrame(new_rows)
            # Align columns with df schema (fill missing as NaN).
            for col in df.columns:
                if col not in anchor_df.columns:
                    anchor_df[col] = np.nan
            anchor_df = anchor_df[df.columns]
            df = pd.concat([df, anchor_df], ignore_index=True)
            df = df.sort_values(["datetime", "instrument"]).reset_index(drop=True)
            print(
                f"[INFO] Phase 4 anchor overlay: injected {len(new_rows)} rows "
                f"across {len(dates)} days for anchors {anchor_codes_active} "
                f"(per-anchor share = {per_anchor_frac:.3f} × daily w_total)"
            )

    # ── Regime gate: load HS300 benchmark close via qlib ─────────────────────
    # Load before the trend sleeve so dynamic risk-on/off sleeve overrides use
    # the same regime state as L3 regime gate and vol-targeting.
    regime_series: pd.Series | None = None
    regime_ratio_series: pd.Series | None = None
    regime_on_series: pd.Series | None = None
    macro_budget_series: pd.Series | None = None
    bench_state_series: pd.Series | None = None
    bench_trend_on_series: pd.Series | None = None
    bench_vol_stress_series: pd.Series | None = None
    bench_trend_score_series: pd.Series | None = None
    benchmark_regime_config = BenchmarkRegimeConfig(
        fast_ma=int(macro_budget_trend_fast_ma),
        slow_ma=int(macro_budget_trend_slow_ma),
        vol_window=int(macro_budget_vol_window),
        vol_lookback=int(macro_budget_vol_lookback),
        budget_min=macro_budget_min,
    )
    need_benchmark_regime = (
        regime_gate_enabled or vol_target_dynamic_enabled or trend_sleeve_dynamic_enabled or budget_v2_enabled
    )
    if need_benchmark_regime:
        try:
            start_str = df["datetime"].min().strftime("%Y-%m-%d")
            end_str = df["datetime"].max().strftime("%Y-%m-%d")
            if budget_v2_enabled:
                bench_close = load_hs300_close(
                    provider_uri,
                    start_str,
                    end_str,
                    warmup_bdays=max(180, int(macro_budget_vol_lookback) + int(macro_budget_trend_slow_ma)),
                )
                features = compute_benchmark_features(bench_close, config=benchmark_regime_config)
                macro_frame = build_macro_budget_series(features, config=benchmark_regime_config)
                macro_frame = macro_frame.set_index("datetime")
                macro_budget_series = macro_frame["risk_macro_budget"]
                bench_state_series = macro_frame["risk_bench_state_label"].astype(str)
                regime_on_series = macro_frame["risk_regime_on"].astype(bool)
                regime_series = macro_budget_series
                regime_ratio_series = features["risk_bench_trend_score"]
                bench_trend_on_series = features["risk_bench_trend_on"]
                bench_vol_stress_series = features["risk_bench_vol_stress"]
                bench_trend_score_series = features["risk_bench_trend_score"]
                state_counts = bench_state_series.value_counts().to_dict()
                print(
                    f"[INFO] Regime gate (budget_v2): macro_budget in "
                    f"[{macro_budget_series.min():.2f}, {macro_budget_series.max():.2f}], "
                    f"states={state_counts}"
                )
            else:
                from qlib.constant import REG_CN
                qlib.init(provider_uri=provider_uri, region=REG_CN)
                bench_raw: pd.DataFrame = D.features(
                    ["sh510300"],
                    ["$close"],
                    start_time=start_str,
                    end_time=end_str,
                )
                bench_close = (
                    bench_raw["$close"]
                    .reset_index(level=0, drop=True)
                    .rename("hs300_close")
                    .sort_index()
                )
                bench_frame = bench_close.to_frame(name="hs300_close")
                bench_frame, _ = auto_qfq_adjust_close_panel(bench_frame)
                bench_close = bench_frame["hs300_close"]
                bench_close = bench_close.reindex(
                    pd.date_range(bench_close.index.min(), bench_close.index.max(), freq="B")
                ).ffill()
                ma = bench_close.rolling(window=int(regime_gate_ma), min_periods=1).mean()
                ratio = bench_close / ma
                regime_ratio_series = ratio
                regime_on_series = ratio >= 1.0
                regime_series = ratio.apply(
                    lambda x: 1.0 if x >= 1.0 else float(regime_gate_min_scale)
                )
                n_below = (ratio < 1.0).sum()
                print(
                    f"[INFO] Regime gate (binary): HS300 below MA({regime_gate_ma}) on {n_below}/{len(ratio)} days "
                    f"→ scale={regime_gate_min_scale}; above MA → scale=1.0"
                )
        except Exception as exc:
            print(f"[WARN] apply_risk_controls: regime-dependent controls disabled — could not load HS300: {exc}")
            regime_series = None
            regime_ratio_series = None
            regime_on_series = None
            macro_budget_series = None
            bench_state_series = None
            bench_trend_on_series = None
            bench_vol_stress_series = None
            bench_trend_score_series = None

    # ── Momentum/trend sleeve: reserve a small budget for fast trend names ──
    # This is applied before cap / vol-target so the downstream hard cap and L3
    # de-risking still govern the combined book.
    trend_sleeve_log: list[dict] = []
    if trend_sleeve_enabled:
        if raw_panel_path is None or not Path(raw_panel_path).exists():
            print(f"[WARN] trend sleeve disabled: raw_panel_path unavailable ({raw_panel_path})")
            trend_sleeve_enabled = False

    if trend_sleeve_enabled:
        raw_for_trend = pd.read_csv(raw_panel_path, parse_dates=["datetime"])
        raw_for_trend["instrument"] = raw_for_trend["instrument"].astype(str).str.upper()
        raw_for_trend = raw_for_trend.sort_values(["instrument", "datetime"])
        if "close" not in raw_for_trend.columns:
            print("[WARN] trend sleeve disabled: raw panel lacks close column")
            trend_sleeve_enabled = False

    if trend_sleeve_enabled and trend_sleeve_warmup_days > 0:
        try:
            from qlib.constant import REG_CN
            qlib.init(provider_uri=provider_uri, region=REG_CN)
            warmup_start = (
                pd.Timestamp(df["datetime"].min())
                - pd.tseries.offsets.BDay(trend_sleeve_warmup_days)
            ).strftime("%Y-%m-%d")
            warmup_end = pd.Timestamp(df["datetime"].max()).strftime("%Y-%m-%d")
            instruments = sorted(df["instrument"].astype(str).str.upper().unique().tolist())
            warmup_raw: pd.DataFrame = D.features(
                instruments,
                ["$close"],
                start_time=warmup_start,
                end_time=warmup_end,
            )
            warmup_close = (
                warmup_raw.reset_index()
                .rename(columns={"$close": "close"})[["instrument", "datetime", "close"]]
            )
            warmup_close["instrument"] = warmup_close["instrument"].astype(str).str.upper()
            warmup_close["datetime"] = pd.to_datetime(warmup_close["datetime"])
            raw_for_trend = pd.concat(
                [warmup_close, raw_for_trend[["instrument", "datetime", "close"]]],
                ignore_index=True,
            )
            raw_for_trend = raw_for_trend.drop_duplicates(
                ["instrument", "datetime"], keep="last"
            ).sort_values(["instrument", "datetime"])
            print(
                f"[INFO] trend sleeve warm-up: loaded qlib close history from {warmup_start} "
                f"for {len(instruments)} instruments (raw rows={len(raw_for_trend)})"
            )
        except Exception as exc:
            print(f"[WARN] trend sleeve warm-up unavailable; using raw panel only: {exc}")

    if trend_sleeve_enabled:
        raw_for_trend["close"] = pd.to_numeric(raw_for_trend["close"], errors="coerce")
        raw_for_trend, _ = auto_qfq_adjust_price_columns_by_group(
            raw_for_trend,
            group_col="instrument",
            date_col="datetime",
            price_cols=["close"],
        )
        mom_cols: list[str] = []
        for window in trend_sleeve_windows:
            col = f"trend_mom{window}"
            raw_for_trend[col] = raw_for_trend.groupby("instrument")["close"].transform(
                lambda series: series.pct_change(window, fill_method=None).shift(1)
            )
            mom_cols.append(col)
        trend_features = raw_for_trend[["instrument", "datetime", *mom_cols]].drop_duplicates(
            ["instrument", "datetime"], keep="last"
        )
        trend_features["risk_trend_momentum_valid_windows"] = trend_features[mom_cols].notna().sum(axis=1)
        df = df.merge(trend_features, on=["instrument", "datetime"], how="left")
        for col in mom_cols + ["forecast_mean_return"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in mom_cols:
            df[f"risk_{col}_z"] = df.groupby("datetime")[col].transform(_robust_z)
        df["risk_trend_momentum_z"] = df[[f"risk_{col}_z" for col in mom_cols]].mean(axis=1)
        if "forecast_mean_return" in df.columns:
            df["risk_trend_mu_z"] = df.groupby("datetime")["forecast_mean_return"].transform(_robust_z)
        else:
            df["risk_trend_mu_z"] = 0.0
        df["risk_trend_score"] = (
            trend_sleeve_mu_weight * df["risk_trend_mu_z"].fillna(0.0)
            + (1.0 - trend_sleeve_mu_weight) * df["risk_trend_momentum_z"].fillna(0.0)
        )
        df["risk_trend_sleeve_add"] = 0.0
        df["risk_trend_sleeve_selected"] = 0
        df["risk_trend_sleeve_weight_total_used"] = 0.0
        df["risk_trend_sleeve_top_k_used"] = 0
        df["risk_trend_sleeve_regime_on"] = float("nan")
        df["risk_trend_sleeve_exposure_floor_used"] = 0.0

        for dt in dates:
            mask_dt = df["datetime"] == dt
            day = df.loc[mask_dt].copy()
            w_nonneg = day["forecast_w"].fillna(0.0).clip(lower=0.0)
            total_w = float(w_nonneg.sum())
            if total_w <= 1e-12:
                continue
            sleeve_weight_total_used = float(trend_sleeve_weight_total)
            sleeve_top_k_used = int(trend_sleeve_top_k)
            sleeve_exposure_floor_used = float(trend_sleeve_exposure_floor)
            sleeve_regime_on_used = float("nan")
            if trend_sleeve_dynamic_enabled and regime_on_series is not None:
                ts = pd.Timestamp(dt)
                available = regime_on_series.index[regime_on_series.index <= ts]
                if len(available) > 0:
                    sleeve_regime_on_used = float(bool(regime_on_series.loc[available[-1]]))
                    if bool(sleeve_regime_on_used):
                        if trend_sleeve_weight_total_risk_on >= 0.0:
                            sleeve_weight_total_used = float(
                                np.clip(trend_sleeve_weight_total_risk_on, 0.0, 1.0)
                            )
                        if trend_sleeve_top_k_risk_on > 0:
                            sleeve_top_k_used = int(trend_sleeve_top_k_risk_on)
                        if trend_sleeve_exposure_floor_risk_on >= 0.0:
                            sleeve_exposure_floor_used = float(
                                np.clip(trend_sleeve_exposure_floor_risk_on, 0.0, 1.0)
                            )
                    else:
                        if trend_sleeve_weight_total_risk_off >= 0.0:
                            sleeve_weight_total_used = float(
                                np.clip(trend_sleeve_weight_total_risk_off, 0.0, 1.0)
                            )
                        if trend_sleeve_top_k_risk_off > 0:
                            sleeve_top_k_used = int(trend_sleeve_top_k_risk_off)
                        if trend_sleeve_exposure_floor_risk_off >= 0.0:
                            sleeve_exposure_floor_used = float(
                                np.clip(trend_sleeve_exposure_floor_risk_off, 0.0, 1.0)
                            )
            sleeve_weight_total_used = float(np.clip(sleeve_weight_total_used, 0.0, 1.0))
            sleeve_top_k_used = int(sleeve_top_k_used)
            sleeve_exposure_floor_used = float(np.clip(sleeve_exposure_floor_used, 0.0, 1.0))
            df.loc[mask_dt, "risk_trend_sleeve_weight_total_used"] = sleeve_weight_total_used
            df.loc[mask_dt, "risk_trend_sleeve_top_k_used"] = sleeve_top_k_used
            df.loc[mask_dt, "risk_trend_sleeve_regime_on"] = sleeve_regime_on_used
            df.loc[mask_dt, "risk_trend_sleeve_exposure_floor_used"] = sleeve_exposure_floor_used
            if sleeve_weight_total_used <= 1e-12 or sleeve_top_k_used <= 0:
                continue
            candidate_mask = (
                day["risk_trend_score"].replace([np.inf, -np.inf], np.nan).notna()
                & (day["risk_trend_score"] >= float(trend_sleeve_min_score))
                & (~day["instrument"].isin(anchor_codes_active))
            )
            sort_columns = ["risk_trend_score", "risk_trend_momentum_z"]
            if "forecast_mean_return" in day.columns:
                sort_columns.append("forecast_mean_return")
            candidates = day.loc[candidate_mask].sort_values(sort_columns, ascending=False)
            if candidates.empty:
                continue
            selected = candidates.head(sleeve_top_k_used)
            selected_index = selected.index
            sleeve_budget = total_w * sleeve_weight_total_used
            if sleeve_budget <= 1e-12:
                continue
            base_scale = max(0.0, 1.0 - sleeve_weight_total_used)
            df.loc[mask_dt, "forecast_w"] = df.loc[mask_dt, "forecast_w"].fillna(0.0) * base_scale
            sleeve_each = sleeve_budget / len(selected_index)
            df.loc[selected_index, "forecast_w"] = df.loc[selected_index, "forecast_w"].fillna(0.0) + sleeve_each
            df.loc[selected_index, "risk_trend_sleeve_add"] = sleeve_each
            df.loc[selected_index, "risk_trend_sleeve_selected"] = 1
            trend_sleeve_log.append(
                {
                    "datetime": dt,
                    "regime_on": sleeve_regime_on_used,
                    "selected": len(selected_index),
                    "weight_total": sleeve_weight_total_used,
                    "top_k": sleeve_top_k_used,
                    "exposure_floor": sleeve_exposure_floor_used,
                    "sleeve_budget": sleeve_budget,
                    "top_instruments": "|".join(selected["instrument"].astype(str).tolist()),
                    "top_score": float(selected["risk_trend_score"].iloc[0]),
                }
            )

    # ── Per-day scaling ───────────────────────────────────────────────────────
    scale_log: list[dict] = []
    cap_trigger_count = 0
    hhi_warn_count = 0
    # Pre-seed exposure_scale column so every row has a value even if some days
    # are skipped (e.g. zero weights). Engine defaults to 1.0 on missing/NaN.
    if "exposure_scale" not in df.columns:
        df["exposure_scale"] = 1.0
    for dt in dates:
        mask = df["datetime"] == dt
        day = df.loc[mask].copy()
        w = day["forecast_w"].fillna(0.0).to_numpy()

        # ── Layer 2a: per-asset weight cap (capped-simplex projection) ────────
        cap_triggered = False
        if weight_cap_enabled:
            w_nonneg = np.clip(w, 0.0, None)
            total = float(w_nonneg.sum())
            if total > 1e-12:
                pre_max = float(w_nonneg.max())
                if pre_max > max_weight_cap + 1e-9:
                    w_capped = _project_capped(w_nonneg, target_sum=total, cap=float(max_weight_cap))
                    w = w_capped
                    df.loc[mask, "forecast_w"] = w
                    cap_trigger_count += 1
                    cap_triggered = True

        # ── Layer 2b: HHI audit (log-only) ───────────────────────────────────
        effective_n = float("nan")
        if hhi_audit_enabled:
            w_nonneg = np.clip(w, 0.0, None)
            total = float(w_nonneg.sum())
            if total > 1e-12:
                hhi = float(np.sum((w_nonneg / total) ** 2))
                if hhi > 0:
                    effective_n = 1.0 / hhi
                    if effective_n < float(min_holdings_audit_warn):
                        hhi_warn_count += 1

        # Regime factor / macro budget
        regime_ratio_used = float("nan")
        regime_on_used = float("nan")
        macro_budget_used = float("nan")
        bench_state_used = ""
        bench_trend_on_used = float("nan")
        bench_vol_stress_used = float("nan")
        bench_trend_score_used = float("nan")
        if regime_series is not None:
            ts = pd.Timestamp(dt)
            available = regime_series.index[regime_series.index <= ts]
            regime_factor = float(regime_series.loc[available[-1]]) if len(available) > 0 else 1.0
            macro_budget_used = float(regime_factor)
            if regime_ratio_series is not None and len(available) > 0:
                regime_ratio_used = float(regime_ratio_series.loc[available[-1]])
                bench_trend_score_used = regime_ratio_used
            if regime_on_series is not None and len(available) > 0:
                regime_on_used = float(bool(regime_on_series.loc[available[-1]]))
            if bench_state_series is not None and len(available) > 0:
                bench_state_used = str(bench_state_series.loc[available[-1]])
            if bench_trend_on_series is not None and len(available) > 0:
                bench_trend_on_used = float(bench_trend_on_series.loc[available[-1]])
            if bench_vol_stress_series is not None and len(available) > 0:
                bench_vol_stress_used = float(bench_vol_stress_series.loc[available[-1]])
        else:
            regime_factor = 1.0
            macro_budget_used = 1.0
            bench_state_used = "offensive"

        vol_target_used = float(vol_target)
        if budget_v2_enabled and np.isfinite(macro_budget_used):
            offensive_target = float(vol_target_offensive or vol_target_risk_on or vol_target)
            neutral_target = float(vol_target_neutral or vol_target)
            defensive_target = float(vol_target_defensive or vol_target_risk_off or vol_target)
            vol_target_used = vol_target_for_macro_budget(
                macro_budget_used,
                offensive=offensive_target,
                neutral=neutral_target,
                defensive=defensive_target,
            )
        elif vol_target_dynamic_enabled and np.isfinite(regime_on_used):
            if bool(regime_on_used) and float(vol_target_risk_on) > 0.0:
                vol_target_used = float(vol_target_risk_on)
            elif (not bool(regime_on_used)) and float(vol_target_risk_off) > 0.0:
                vol_target_used = float(vol_target_risk_off)
        if vol_target_used <= 0.0:
            vol_target_used = max(
                float(vol_target_risk_on),
                float(vol_target_risk_off),
                float(vol_target_offensive),
                float(vol_target_neutral),
                float(vol_target_defensive),
                float(vol_target),
                0.0,
            )

        # Vol scale (full covariance by default; diagonal available for back-compat)
        # 'full' uses w'Σw from bridge's forecast_cov_matrix/order (captures correlation risk).
        # 'diagonal' uses Σwᵢ²σᵢ² (legacy; ignores correlations — underestimates sector-concentrated books).
        vol_scale = 1.0
        port_vol_used = float("nan")
        if vol_target_enabled:
            port_var: float | None = None
            use_full = (
                vol_target_cov_mode == "full"
                and "forecast_cov_matrix" in day.columns
                and "forecast_cov_order" in day.columns
            )
            if use_full:
                # Build full Σ from serialized rows.
                instruments_day = day["instrument"].astype(str).tolist()
                cov_order_raw = day["forecast_cov_order"].dropna().astype(str)
                cov_order = cov_order_raw.iloc[0].split("|") if len(cov_order_raw) > 0 else []
                try:
                    cov_rows: list[list[float]] = []
                    # forecast_cov_matrix is one row per instrument (in same order as cov_order).
                    cov_lookup = dict(zip(instruments_day, day["forecast_cov_matrix"].astype(str).tolist()))
                    w_by_inst = dict(zip(instruments_day, w.tolist()))
                    if cov_order:
                        for inst in cov_order:
                            raw = cov_lookup.get(inst, "")
                            if not raw or raw.lower() == "nan":
                                cov_rows = []
                                break
                            cov_rows.append([float(x) for x in raw.split(",")])
                    if cov_rows and len(cov_rows) == len(cov_order):
                        Sigma = np.asarray(cov_rows, dtype=float)
                        w_aligned = np.asarray([w_by_inst.get(inst, 0.0) for inst in cov_order], dtype=float)
                        port_var = float(w_aligned @ Sigma @ w_aligned)
                except (ValueError, TypeError):
                    port_var = None

            if port_var is None:
                # Fallback: diagonal approx.
                if "forecast_variance" in day.columns:
                    var_vals = pd.to_numeric(day["forecast_variance"], errors="coerce")
                elif "forecast_variance_robust" in day.columns:
                    var_vals = pd.to_numeric(day["forecast_variance_robust"], errors="coerce")
                elif "annual_forecast_variance" in day.columns:
                    var_vals = pd.to_numeric(day["annual_forecast_variance"], errors="coerce") / 252.0
                else:
                    var_vals = pd.Series(dtype=float)
                if not var_vals.empty:
                    if "forecast_variance_robust" in day.columns and "forecast_variance" in day.columns:
                        fb = pd.to_numeric(day["forecast_variance_robust"], errors="coerce")
                        var_vals = var_vals.fillna(fb)
                    var_arr = var_vals.fillna(0.0).clip(lower=0.0).to_numpy()
                    port_var = float(np.dot(w ** 2, var_arr))

            if port_var is not None and port_var > 0.0:
                port_vol_used = float(np.sqrt(port_var))
                vol_scale = min(1.0, float(vol_target_used) / port_vol_used)

        # ── Phase 3 Path A: cluster-gate (weighted-avg pairwise correlation) ─
        # Compute ρ̄ on active holdings (w>0). The default uses bridge Σ; the
        # realized source uses trailing raw-panel returns and is less affected by
        # corr_floor flattening.
        cluster_scale = 1.0
        rho_bar_used = float("nan")
        if cluster_gate_enabled:
            w_nonneg = np.clip(w, 0.0, None)
            total_w = float(w_nonneg.sum())
            rho_bar: float | None = None
            if total_w > 1e-12:
                if cluster_gate_source == "realized":
                    rho_bar = _rho_bar_from_realized(day, w_nonneg, dt)
                if rho_bar is None:
                    rho_bar = _rho_bar_from_forecast_cov(day, w_nonneg)

            if rho_bar is not None and np.isfinite(rho_bar):
                rho_bar_used = float(np.clip(rho_bar, -1.0, 1.0))
                if rho_bar_used > float(cluster_gate_rho):
                    span = float(cluster_gate_rho_saturation) - float(cluster_gate_rho)
                    t = (rho_bar_used - float(cluster_gate_rho)) / span
                    t = float(np.clip(t, 0.0, 1.0))
                    cluster_scale = 1.0 - (1.0 - float(cluster_gate_min_scale)) * t

        signal_quality_scale = 1.0
        signal_quality_ic_used = float("nan")
        signal_quality_min_scale_used = float(signal_quality_min_scale)
        if signal_quality_gate_enabled and np.isfinite(regime_on_used):
            if bool(regime_on_used) and signal_quality_min_scale_risk_on > 0.0:
                signal_quality_min_scale_used = float(signal_quality_min_scale_risk_on)
            elif (not bool(regime_on_used)) and signal_quality_min_scale_risk_off > 0.0:
                signal_quality_min_scale_used = float(signal_quality_min_scale_risk_off)
        if signal_quality_gate_enabled:
            ts = pd.Timestamp(dt)
            if ts in signal_quality_rolling_ic.index:
                signal_quality_ic_used = float(signal_quality_rolling_ic.loc[ts])
            if (
                np.isfinite(signal_quality_ic_used)
                and signal_quality_ic_used < float(signal_quality_ic_threshold)
            ):
                signal_quality_scale = float(signal_quality_min_scale_used)

        portfolio_scale = float(vol_scale) * float(cluster_scale)
        signal_scale = float(signal_quality_scale)
        if budget_v2_enabled:
            macro_budget = float(macro_budget_used if np.isfinite(macro_budget_used) else regime_factor)
            effective_exposure = macro_budget * portfolio_scale * signal_scale
            effective_exposure = float(np.clip(effective_exposure, macro_budget_min, macro_budget))
            final_scale = effective_exposure
            regime_factor = macro_budget
        else:
            effective_exposure = float(regime_factor) * portfolio_scale * signal_scale
            final_scale = effective_exposure
        row_exposure_scale = np.full(len(day), final_scale, dtype=float)
        sleeve_floor_lift = 0.0
        if (
            not budget_v2_enabled
            and trend_sleeve_exposure_floor_enabled
            and "risk_trend_sleeve_add" in day.columns
            and "risk_trend_sleeve_exposure_floor_used" in day.columns
        ):
            post_w_nonneg = np.clip(w, 0.0, None)
            sleeve_add = pd.to_numeric(
                day["risk_trend_sleeve_add"], errors="coerce"
            ).fillna(0.0).clip(lower=0.0).to_numpy()
            floor_arr = pd.to_numeric(
                day["risk_trend_sleeve_exposure_floor_used"], errors="coerce"
            ).fillna(0.0).clip(lower=0.0, upper=1.0).to_numpy()
            sleeve_component = np.minimum(sleeve_add, post_w_nonneg)
            sleeve_floor_scale = np.maximum(final_scale, floor_arr)
            core_component = np.maximum(post_w_nonneg - sleeve_component, 0.0)
            numerator = core_component * final_scale + sleeve_component * sleeve_floor_scale
            row_exposure_scale = np.full(len(day), final_scale, dtype=float)
            np.divide(
                numerator,
                post_w_nonneg,
                out=row_exposure_scale,
                where=post_w_nonneg > 1e-12,
            )
            sleeve_floor_lift = float(np.dot(post_w_nonneg, row_exposure_scale - final_scale))
        # forecast_w keeps the capped values (no multiplicative scaling); the
        # Layer-3 de-leveraging is transmitted to the backtest engine via the
        # exposure_scale column, so undeployed capital becomes residual cash
        # rather than being normalized away.
        df.loc[mask, "exposure_scale"] = row_exposure_scale
        df.loc[mask, "risk_base_exposure_scale"] = final_scale
        df.loc[mask, "effective_exposure"] = final_scale
        df.loc[mask, "risk_trend_sleeve_floor_lift"] = sleeve_floor_lift
        df.loc[mask, "risk_regime_factor"] = regime_factor
        df.loc[mask, "risk_macro_budget"] = macro_budget_used
        df.loc[mask, "risk_bench_state_label"] = bench_state_used
        df.loc[mask, "risk_bench_trend_on"] = bench_trend_on_used
        df.loc[mask, "risk_bench_vol_stress"] = bench_vol_stress_used
        df.loc[mask, "risk_bench_trend_score"] = bench_trend_score_used
        df.loc[mask, "risk_portfolio_scale"] = portfolio_scale
        df.loc[mask, "risk_signal_scale"] = signal_scale
        df.loc[mask, "risk_regime_ratio"] = regime_ratio_used
        df.loc[mask, "risk_regime_on"] = regime_on_used
        df.loc[mask, "risk_vol_target_used"] = vol_target_used
        df.loc[mask, "risk_vol_scale"] = vol_scale
        df.loc[mask, "risk_cluster_rho_bar"] = rho_bar_used
        df.loc[mask, "risk_cluster_scale"] = cluster_scale
        df.loc[mask, "risk_signal_quality_rolling_ic"] = signal_quality_ic_used
        df.loc[mask, "risk_signal_quality_min_scale_used"] = signal_quality_min_scale_used
        df.loc[mask, "risk_signal_quality_scale"] = signal_quality_scale
        scale_log.append(
            {
                "datetime": dt,
                "cap_triggered": cap_triggered,
                "effective_n": effective_n,
                "port_vol_used": port_vol_used,
                "rho_bar": rho_bar_used,
                "regime_factor": regime_factor,
                "macro_budget": macro_budget_used,
                "bench_state_label": bench_state_used,
                "regime_ratio": regime_ratio_used,
                "regime_on": regime_on_used,
                "vol_target_used": vol_target_used,
                "vol_scale": vol_scale,
                "portfolio_scale": portfolio_scale,
                "cluster_scale": cluster_scale,
                "signal_quality_ic": signal_quality_ic_used,
                "signal_quality_min_scale_used": signal_quality_min_scale_used,
                "signal_quality_scale": signal_quality_scale,
                "signal_scale": signal_scale,
                "final_scale": final_scale,
                "effective_scale_mean": float(np.mean(row_exposure_scale)) if len(row_exposure_scale) else final_scale,
                "sleeve_floor_lift": sleeve_floor_lift,
            }
        )

    # ── Summary log ──────────────────────────────────────────────────────────
    log_df = pd.DataFrame(scale_log)
    active = log_df[log_df["final_scale"] < 0.999]
    regime_active = log_df[log_df["regime_factor"] < 0.999]
    vol_active = log_df[log_df["vol_scale"] < 0.999]
    cluster_active = log_df[log_df["cluster_scale"] < 0.999] if "cluster_scale" in log_df.columns else log_df.iloc[:0]
    signal_quality_active = log_df[log_df["signal_quality_scale"] < 0.999] if "signal_quality_scale" in log_df.columns else log_df.iloc[:0]
    print(
        f"[INFO] apply_risk_controls summary: {len(log_df)} days total"
    )
    if weight_cap_enabled:
        print(
            f"  max_weight_cap triggered: {cap_trigger_count} days (cap={max_weight_cap})"
        )
    if hhi_audit_enabled and "effective_n" in log_df.columns:
        en = log_df["effective_n"].dropna()
        if not en.empty:
            print(
                f"  effective_n (1/HHI): mean={en.mean():.2f}, min={en.min():.2f}, "
                f"< {min_holdings_audit_warn} on {hhi_warn_count} days"
            )
    if regime_gate_enabled:
        print(
            f"  regime_gate triggered: {len(regime_active)} days (factor={regime_gate_min_scale})"
        )
    if vol_target_enabled:
        print(
            f"  vol_targeting triggered: {len(vol_active)} days (vol_target={vol_target:.4f})"
        )
        if vol_target_dynamic_enabled and "vol_target_used" in log_df.columns:
            print(
                f"  regime-conditioned vol_target: mean={log_df['vol_target_used'].mean():.4f}, "
                f"min={log_df['vol_target_used'].min():.4f}, max={log_df['vol_target_used'].max():.4f}"
            )
    if trend_sleeve_enabled:
        trend_log_df = pd.DataFrame(trend_sleeve_log)
        if not trend_log_df.empty:
            print(
                f"  trend_sleeve applied: {len(trend_log_df)} days "
                f"({trend_sleeve_desc})"
            )
            if "weight_total" in trend_log_df.columns:
                print(
                    "  trend_sleeve effective settings: "
                    f"weight_total mean={trend_log_df['weight_total'].mean():.3f}, "
                    f"min={trend_log_df['weight_total'].min():.3f}, "
                    f"max={trend_log_df['weight_total'].max():.3f}; "
                    f"top_k mean={trend_log_df['top_k'].mean():.2f}"
                )
            if "exposure_floor" in trend_log_df.columns and trend_log_df["exposure_floor"].max() > 0:
                print(
                    "  trend_sleeve exposure floor: "
                    f"mean={trend_log_df['exposure_floor'].mean():.3f}, "
                    f"min={trend_log_df['exposure_floor'].min():.3f}, "
                    f"max={trend_log_df['exposure_floor'].max():.3f}"
                )
            print(trend_log_df.tail(10).to_string(index=False))
    if cluster_gate_enabled and "rho_bar" in log_df.columns:
        rb = log_df["rho_bar"].dropna()
        if not rb.empty:
            print(
                f"  cluster_gate triggered: {len(cluster_active)} days "
                f"(source={cluster_gate_source}, ρ̄ mean={rb.mean():.3f}, max={rb.max():.3f}, "
                f"min_scale={cluster_gate_min_scale}, threshold={cluster_gate_rho})"
            )
    if signal_quality_gate_enabled and "signal_quality_ic" in log_df.columns:
        ic_vals = log_df["signal_quality_ic"].dropna()
        if not ic_vals.empty:
            print(
                f"  signal_quality_gate triggered: {len(signal_quality_active)} days "
                f"(rolling_ic mean={ic_vals.mean():.3f}, min={ic_vals.min():.3f}, "
                f"threshold={signal_quality_ic_threshold}, min_scale={signal_quality_min_scale}"
                f"{', risk_on_min_scale=%.2f, risk_off_min_scale=%.2f' % (signal_quality_min_scale_risk_on or signal_quality_min_scale, signal_quality_min_scale_risk_off or signal_quality_min_scale) if (signal_quality_min_scale_risk_on > 0.0 or signal_quality_min_scale_risk_off > 0.0) else ''})"
            )
    print(
        f"  combined final_scale < 1.0: {len(active)} days, "
        f"mean={log_df['final_scale'].mean():.3f}, min={log_df['final_scale'].min():.3f}"
    )
    if "sleeve_floor_lift" in log_df.columns and log_df["sleeve_floor_lift"].max() > 1e-12:
        lifted = log_df[log_df["sleeve_floor_lift"] > 1e-12]
        print(
            f"  trend_sleeve exposure floor lifted {len(lifted)} days: "
            f"mean_additional_exposure={lifted['sleeve_floor_lift'].mean():.4f}, "
            f"max={lifted['sleeve_floor_lift'].max():.4f}"
        )
    if not active.empty:
        print(active.drop(columns=["effective_n"], errors="ignore").to_string(index=False))
    if len(active) == 0:
        print(
            "[WARN] L3 is a NO-OP on every day (exposure_scale==1.0 for all days). "
            "Check --vol-target / --regime-gate-min-scale / --cluster-gate-* configuration."
        )

    out_path = bridge_l3_csv_path if bridge_l3_csv_path is not None else bridge_csv_path
    df.to_csv(out_path, index=False)
    if out_path != bridge_csv_path:
        print(f"[INFO] apply_risk_controls: kept L1+L2 snapshot → {bridge_csv_path}")
        print(f"[INFO] apply_risk_controls: wrote L3 (with exposure_scale) → {out_path}")
    else:
        print(f"[INFO] apply_risk_controls: wrote modified bridge CSV → {out_path}")
