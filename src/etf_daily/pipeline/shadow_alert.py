from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import DEFAULT_FUND_LIST_CSV, DEFAULT_PROVIDER_URI

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
ENTROPY_DEFENSE_THRESHOLD = 0.80
ENTROPY_DEFENSE_SQ_LOW = -0.05
ENTROPY_DEFENSE_SQ_HIGH = 0.10
ENTROPY_DEFENSE_SCALE = 0.50
MOMENTUM_FATIGUE_RETURN_THRESHOLD = 0.10
MOMENTUM_FATIGUE_WEIGHT_SHARE = 0.30
MOMENTUM_FATIGUE_TOLERANCE = 1e-6
NEW_MODEL_SIGNAL_FEATURE_FLAG_DEFAULT = False
DEFAULT_NEW_MODEL_COEFFICIENTS_CSV = "archive/audits/20260504_route_a_q1_coefficients/route_a_q1_feature_coefficients.csv"
DEFAULT_NEW_MODEL_PROVIDER_URI = DEFAULT_PROVIDER_URI


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _json_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _json_scalar(value: object) -> object:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)):
        return _json_float(value)
    return value


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


def _required_columns() -> set[str]:
    return {"instrument", "datetime", "close", "forecast_w"}


def _normalize_code(value: object) -> str:
    text = str(value).strip().upper()
    if text.startswith("SH") or text.startswith("SZ"):
        return text
    if len(text) == 6 and text.isdigit():
        prefix = "SH" if text.startswith(("5", "6")) else "SZ"
        return f"{prefix}{text}"
    return text


def _load_name_map(fund_list_csv: str | Path | None) -> dict[str, str]:
    if not fund_list_csv:
        return {}
    path = Path(fund_list_csv)
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {}
    code_col = "基金代码" if "基金代码" in frame.columns else "code" if "code" in frame.columns else "instrument" if "instrument" in frame.columns else ""
    name_col = "基金简称" if "基金简称" in frame.columns else "name" if "name" in frame.columns else ""
    if not code_col or not name_col:
        return {}
    return {
        _normalize_code(code): str(name)
        for code, name in zip(frame[code_col], frame[name_col])
        if str(code).strip() and str(name).strip()
    }


def _industry_theme_label(name: object) -> str:
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


def _refined_industry_label(name: object) -> str:
    base_label = _industry_theme_label(name)
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


def load_shadow_l3_frame(l3_csv_path: str | Path, fund_list_csv: str | Path | None = DEFAULT_FUND_LIST_CSV) -> pd.DataFrame:
    path = Path(l3_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"L3 bridge CSV not found: {path}")
    frame = pd.read_csv(path, parse_dates=["datetime"])
    missing = sorted(_required_columns() - set(frame.columns))
    if missing:
        raise ValueError(f"L3 bridge CSV missing required columns: {missing}")
    frame = frame.sort_values(["instrument", "datetime"]).copy()
    frame["instrument"] = frame["instrument"].astype(str)
    frame["forecast_w"] = _numeric(frame["forecast_w"]).fillna(0.0).clip(lower=0.0)
    frame["close"] = _numeric(frame["close"])
    if "next_return" not in frame.columns:
        frame["next_return"] = np.nan
    frame["next_return"] = _numeric(frame["next_return"])
    if "exposure_scale" not in frame.columns:
        frame["exposure_scale"] = 1.0
    frame["exposure_scale"] = _numeric(frame["exposure_scale"]).fillna(1.0).clip(lower=0.0, upper=1.0)
    if "risk_signal_quality_rolling_ic" not in frame.columns:
        frame["risk_signal_quality_rolling_ic"] = np.nan
    frame["risk_signal_quality_rolling_ic"] = _numeric(frame["risk_signal_quality_rolling_ic"])
    if "factor_industry_theme" not in frame.columns:
        frame["factor_industry_theme"] = "other"
    frame["factor_industry_theme"] = frame["factor_industry_theme"].fillna("other").astype(str)
    frame.loc[frame["factor_industry_theme"].str.len() == 0, "factor_industry_theme"] = "other"
    name_map = _load_name_map(fund_list_csv)
    if name_map:
        fallback_theme = frame["instrument"].map(lambda value: name_map.get(_normalize_code(value), "")).map(_refined_industry_label)
        coarse_mask = frame["factor_industry_theme"].str.lower().isin(["", "nan", "other"])
        frame.loc[coarse_mask, "factor_industry_theme"] = fallback_theme[coarse_mask]
    frame["mainline_effective_weight"] = frame["forecast_w"] * frame["exposure_scale"]
    frame["momentum_return_t_minus_5_to_t_minus_1"] = (
        frame.groupby("instrument")["close"].shift(1)
        / frame.groupby("instrument")["close"].shift(5)
        - 1.0
    )
    return frame


def _first_finite(values: pd.Series) -> float:
    finite = _numeric(values).dropna()
    return float(finite.iloc[0]) if not finite.empty else float("nan")


def build_shadow_alert_history(l3_csv_path: str | Path, fund_list_csv: str | Path | None = DEFAULT_FUND_LIST_CSV) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = load_shadow_l3_frame(l3_csv_path, fund_list_csv=fund_list_csv)
    rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []

    for date_value, day_frame in frame.groupby("datetime", sort=True):
        date_text = pd.Timestamp(date_value).strftime("%Y-%m-%d")
        active = day_frame[day_frame["forecast_w"] > 1e-12].copy()
        active_weight_sum = float(active["forecast_w"].sum()) if not active.empty else 0.0
        mainline_effective_sum = float(active["mainline_effective_weight"].sum()) if not active.empty else 0.0
        sq_ic = _first_finite(day_frame["risk_signal_quality_rolling_ic"])

        entropy = float("nan")
        entropy_norm = float("nan")
        entropy_effective_count = float("nan")
        dominant_industry = ""
        dominant_industry_weight_share = float("nan")
        if active_weight_sum > 1e-12:
            shares = active.groupby("factor_industry_theme")["forecast_w"].sum() / active_weight_sum
            entropy, entropy_norm, entropy_effective_count = _normalized_entropy(shares)
            if not shares.empty:
                dominant_industry = str(shares.idxmax())
                dominant_industry_weight_share = float(shares.max())

        hot_active = active[_numeric(active["momentum_return_t_minus_5_to_t_minus_1"]) > MOMENTUM_FATIGUE_RETURN_THRESHOLD]
        hot_weight_sum = float(hot_active["forecast_w"].sum()) if not hot_active.empty else 0.0
        combined_momentum_overheat_index = hot_weight_sum / active_weight_sum if active_weight_sum > 1e-12 else float("nan")
        momentum_fatigue_warning = bool(
            np.isfinite(combined_momentum_overheat_index)
            and combined_momentum_overheat_index >= MOMENTUM_FATIGUE_WEIGHT_SHARE - MOMENTUM_FATIGUE_TOLERANCE
        )

        entropy_threshold_trigger = bool(np.isfinite(entropy_norm) and entropy_norm <= ENTROPY_DEFENSE_THRESHOLD)
        entropy_defense_trigger = bool(
            entropy_threshold_trigger
            and np.isfinite(sq_ic)
            and ENTROPY_DEFENSE_SQ_LOW <= sq_ic <= ENTROPY_DEFENSE_SQ_HIGH
        )
        entropy_shadow_scale = ENTROPY_DEFENSE_SCALE if entropy_defense_trigger else 1.0

        active["entropy_shadow_suggested_effective_weight"] = active["mainline_effective_weight"] * entropy_shadow_scale
        active["entropy_shadow_weight_delta"] = (
            active["entropy_shadow_suggested_effective_weight"] - active["mainline_effective_weight"]
        )
        detail_frames.append(
            active[
                [
                    "datetime",
                    "instrument",
                    "factor_industry_theme",
                    "forecast_w",
                    "exposure_scale",
                    "mainline_effective_weight",
                    "entropy_shadow_suggested_effective_weight",
                    "entropy_shadow_weight_delta",
                    "next_return",
                    "momentum_return_t_minus_5_to_t_minus_1",
                ]
            ].copy()
        )

        mainline_return = float((active["mainline_effective_weight"] * active["next_return"].fillna(0.0)).sum()) if not active.empty else 0.0
        entropy_shadow_return = float((active["entropy_shadow_suggested_effective_weight"] * active["next_return"].fillna(0.0)).sum()) if not active.empty else 0.0
        rows.append(
            {
                "datetime": date_text,
                "active_count": int(active.shape[0]),
                "active_weight_sum": active_weight_sum,
                "mainline_effective_weight_sum": mainline_effective_sum,
                "industry_entropy_norm": entropy_norm,
                "industry_entropy_effective_count": entropy_effective_count,
                "dominant_industry": dominant_industry,
                "dominant_industry_weight_share": dominant_industry_weight_share,
                "risk_signal_quality_rolling_ic": sq_ic,
                "entropy_threshold_trigger": entropy_threshold_trigger,
                "entropy_defense_trigger": entropy_defense_trigger,
                "entropy_shadow_scale": entropy_shadow_scale,
                "combined_momentum_overheat_index": combined_momentum_overheat_index,
                "momentum_fatigue_warning": momentum_fatigue_warning,
                "momentum_fatigue_label": "Momentum Fatigue Warning" if momentum_fatigue_warning else "",
                "hot_active_count": int(hot_active.shape[0]),
                "hot_active_weight_share": combined_momentum_overheat_index,
                "hot_active_codes": "|".join(hot_active["instrument"].astype(str).tolist()),
                "mainline_effective_return": mainline_return,
                "entropy_shadow_effective_return": entropy_shadow_return,
                "entropy_shadow_delta_return": entropy_shadow_return - mainline_return,
                "entropy_shadow_abs_weight_delta": float(active["entropy_shadow_weight_delta"].abs().sum()) if not active.empty else 0.0,
            }
        )

    history = pd.DataFrame(rows)
    detail = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    if not history.empty:
        history["entropy_shadow_delta_return_cumulative"] = _numeric(history["entropy_shadow_delta_return"]).fillna(0.0).cumsum()
        history["mainline_cumulative_return_proxy"] = (1.0 + _numeric(history["mainline_effective_return"]).fillna(0.0)).cumprod() - 1.0
        history["entropy_shadow_cumulative_return_proxy"] = (1.0 + _numeric(history["entropy_shadow_effective_return"]).fillna(0.0)).cumprod() - 1.0
    if not detail.empty:
        detail["datetime_text"] = pd.to_datetime(detail["datetime"]).dt.strftime("%Y-%m-%d")
        for source_col, output_col in [
            ("mainline_effective_weight", "mainline_effective_turnover_proxy"),
            ("entropy_shadow_suggested_effective_weight", "entropy_shadow_effective_turnover_proxy"),
        ]:
            wide = detail.pivot_table(index="datetime_text", columns="instrument", values=source_col, aggfunc="sum").fillna(0.0)
            turnover = wide.diff().abs().sum(axis=1)
            if len(wide) > 0:
                turnover.iloc[0] = wide.iloc[0].abs().sum()
            history[output_col] = history["datetime"].map(turnover.to_dict())
    return history, detail


def _records_for_latest_detail(detail: pd.DataFrame, alert_date: str, limit: int = 12) -> list[dict[str, Any]]:
    if detail.empty:
        return []
    working = detail[pd.to_datetime(detail["datetime"]).dt.strftime("%Y-%m-%d") == alert_date].copy()
    if working.empty:
        return []
    working["abs_delta"] = _numeric(working["entropy_shadow_weight_delta"]).abs()
    working = working.sort_values(["abs_delta", "mainline_effective_weight"], ascending=[False, False]).head(limit)
    records: list[dict[str, Any]] = []
    for _, row in working.iterrows():
        records.append(
            {
                "instrument": str(row.get("instrument", "")),
                "industry": str(row.get("factor_industry_theme", "")),
                "forecast_w": _json_float(row.get("forecast_w")),
                "exposure_scale": _json_float(row.get("exposure_scale")),
                "mainline_effective_weight": _json_float(row.get("mainline_effective_weight")),
                "entropy_shadow_suggested_effective_weight": _json_float(row.get("entropy_shadow_suggested_effective_weight")),
                "entropy_shadow_weight_delta": _json_float(row.get("entropy_shadow_weight_delta")),
                "next_return": _json_float(row.get("next_return")),
                "momentum_return_t_minus_5_to_t_minus_1": _json_float(row.get("momentum_return_t_minus_5_to_t_minus_1")),
            }
        )
    return records


def _resolve_repo_relative_path(path_value: str | Path | None, default_value: str) -> Path:
    path = Path(path_value or default_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _records_for_new_model_weight_delta(weights: pd.DataFrame, alert_date: str, limit: int = 12) -> list[dict[str, Any]]:
    if weights.empty:
        return []
    working = weights[weights["datetime"].astype(str) == alert_date].copy()
    if working.empty:
        return []
    working["abs_delta"] = _numeric(working["delta_mvo_weight"]).abs()
    working = working.sort_values("abs_delta", ascending=False).head(limit)
    records: list[dict[str, Any]] = []
    for _, row in working.iterrows():
        records.append(
            {
                "instrument": str(row.get("instrument", "")),
                "old_mvo_weight": _json_float(row.get("old_mvo_weight")),
                "new_model_shadow_mvo_weight": _json_float(row.get("new_model_shadow_mvo_weight")),
                "delta_mvo_weight": _json_float(row.get("delta_mvo_weight")),
                "old_route_a_signal": _json_float(row.get("old_route_a_signal")),
                "new_model_shadow_signal": _json_float(row.get("new_model_shadow_signal")),
                "new_model_mu_delta": _json_float(row.get("new_model_mu_delta")),
                "momentum_decay_5d_over_10pct": _json_float(row.get("momentum_decay_5d_over_10pct")),
                "next_return": _json_float(row.get("next_return")),
            }
        )
    return records


def _build_new_model_signal_shadow_payload(
    l3_csv_path: str | Path,
    output_json_path: str | Path,
    history: pd.DataFrame,
    alert_date: str,
    enable_new_model_signal_shadow: bool,
    new_model_coefficients_csv: str | Path | None,
    new_model_shadow_output_dir: str | Path | None,
    provider_uri: str,
) -> dict[str, Any]:
    coefficients_path = _resolve_repo_relative_path(new_model_coefficients_csv, DEFAULT_NEW_MODEL_COEFFICIENTS_CSV)
    if not enable_new_model_signal_shadow:
        return {
            "enabled": False,
            "feature_flag_default": NEW_MODEL_SIGNAL_FEATURE_FLAG_DEFAULT,
            "coefficients_csv": str(coefficients_path),
            "scope": "shadow_only_no_live_weight_change",
            "status": "disabled_by_default",
        }
    try:
        from etf_daily.pipeline.route_a_new_model_shadow_oos import ReplayConfig, run_shadow_oos

        start_date = str(history["datetime"].iloc[0]) if not history.empty else alert_date
        output_dir = Path(new_model_shadow_output_dir) if new_model_shadow_output_dir else Path(output_json_path).with_name("new_model_signal_shadow_latest")
        config = ReplayConfig(
            start_date=start_date,
            end_date=alert_date,
            warmup_start_date="2025-11-01",
            max_weight=0.15,
            min_weight_floor=0.05,
            provider_uri=provider_uri,
            enable_new_model_shadow=True,
        )
        outputs = run_shadow_oos(l3_csv_path, coefficients_path, output_dir, config)
        daily = pd.read_csv(outputs["daily"])
        weights = pd.read_csv(outputs["weights"])
        latest_daily = daily[daily["datetime"].astype(str) == alert_date]
        latest_row = latest_daily.iloc[-1].to_dict() if not latest_daily.empty else {}
        return {
            "enabled": True,
            "feature_flag_default": NEW_MODEL_SIGNAL_FEATURE_FLAG_DEFAULT,
            "coefficients_csv": str(coefficients_path),
            "alert_date": alert_date,
            "latest": {key: _json_scalar(value) for key, value in latest_row.items()},
            "top_abs_weight_delta": _records_for_new_model_weight_delta(weights, alert_date),
            "files": {key: str(value) for key, value in outputs.items()},
            "scope": "shadow_only_no_live_weight_change",
            "status": "ok",
        }
    except Exception as exc:
        return {
            "enabled": True,
            "feature_flag_default": NEW_MODEL_SIGNAL_FEATURE_FLAG_DEFAULT,
            "coefficients_csv": str(coefficients_path),
            "alert_date": alert_date,
            "status": "error",
            "error": str(exc),
            "scope": "shadow_only_no_live_weight_change",
        }


def generate_daily_shadow_alert(
    l3_csv_path: str | Path,
    output_json_path: str | Path,
    history_csv_path: str | Path | None = None,
    alert_date: str | None = None,
    fund_list_csv: str | Path | None = DEFAULT_FUND_LIST_CSV,
    enable_new_model_signal_shadow: bool = NEW_MODEL_SIGNAL_FEATURE_FLAG_DEFAULT,
    new_model_coefficients_csv: str | Path | None = DEFAULT_NEW_MODEL_COEFFICIENTS_CSV,
    new_model_shadow_output_dir: str | Path | None = None,
    new_model_provider_uri: str = DEFAULT_NEW_MODEL_PROVIDER_URI,
) -> dict[str, Any]:
    output_path = Path(output_json_path)
    history_path = Path(history_csv_path) if history_csv_path is not None else output_path.with_name("daily_shadow_alert_history.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any]
    try:
        history, detail = build_shadow_alert_history(l3_csv_path, fund_list_csv=fund_list_csv)
        if history.empty:
            raise ValueError("No daily rows available in L3 bridge CSV")
        history.to_csv(history_path, index=False)
        if alert_date is None:
            latest = history.iloc[-1]
        else:
            date_text = pd.Timestamp(alert_date).strftime("%Y-%m-%d")
            matched = history[history["datetime"].astype(str) <= date_text]
            latest = matched.iloc[-1] if not matched.empty else history.iloc[-1]
        latest_date = str(latest["datetime"])
        latest_dict = {key: _json_scalar(value) for key, value in latest.to_dict().items()}
        new_model_signal_shadow = _build_new_model_signal_shadow_payload(
            l3_csv_path=l3_csv_path,
            output_json_path=output_json_path,
            history=history,
            alert_date=latest_date,
            enable_new_model_signal_shadow=bool(enable_new_model_signal_shadow),
            new_model_coefficients_csv=new_model_coefficients_csv,
            new_model_shadow_output_dir=new_model_shadow_output_dir,
            provider_uri=new_model_provider_uri,
        )
        payload = {
            "status": "ok",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_l3_csv": str(Path(l3_csv_path)),
            "fund_list_csv": str(Path(fund_list_csv)) if fund_list_csv else "",
            "alert_date": latest_date,
            "thresholds": {
                "entropy_threshold": ENTROPY_DEFENSE_THRESHOLD,
                "entropy_sq_range": [ENTROPY_DEFENSE_SQ_LOW, ENTROPY_DEFENSE_SQ_HIGH],
                "entropy_shadow_scale": ENTROPY_DEFENSE_SCALE,
                "momentum_return_threshold": MOMENTUM_FATIGUE_RETURN_THRESHOLD,
                "momentum_weight_share_threshold": MOMENTUM_FATIGUE_WEIGHT_SHARE,
            },
            "latest": latest_dict,
            "cumulative": {
                "history_days": int(history.shape[0]),
                "entropy_defense_trigger_days": int(history["entropy_defense_trigger"].sum()),
                "entropy_threshold_trigger_days": int(history["entropy_threshold_trigger"].sum()),
                "momentum_fatigue_warning_days": int(history["momentum_fatigue_warning"].sum()),
                "entropy_shadow_delta_return_cumulative": _json_float(history["entropy_shadow_delta_return"].fillna(0.0).sum()),
                "mainline_cumulative_return_proxy": _json_float(history["mainline_cumulative_return_proxy"].iloc[-1]),
                "entropy_shadow_cumulative_return_proxy": _json_float(history["entropy_shadow_cumulative_return_proxy"].iloc[-1]),
            },
            "weight_comparison_top_abs_delta": _records_for_latest_detail(detail, latest_date),
            "new_model_signal_shadow": new_model_signal_shadow,
            "files": {
                "history_csv": str(history_path),
                "json": str(output_path),
            },
            "scope": "shadow_only_no_live_weight_change",
        }
    except Exception as exc:
        payload = {
            "status": "error",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_l3_csv": str(Path(l3_csv_path)),
            "fund_list_csv": str(Path(fund_list_csv)) if fund_list_csv else "",
            "error": str(exc),
            "scope": "shadow_only_no_live_weight_change",
        }

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
