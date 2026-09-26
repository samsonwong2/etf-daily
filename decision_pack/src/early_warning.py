"""Per-symbol yellow/red early warning for decision_pack (display_only overlay)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from decision_pack.src.config import DecisionPackConfig, load_decision_pack_config

AlertLevel = Literal["green", "yellow", "red"]

YELLOW_SIGNALS = frozenset({"below_ma20", "weak_rs", "bear_stack", "mom_fade", "vol_accel", "dd_soft"})
RED_FLAGS = frozenset(
    {
        "deep_below_ma20",
        "below_ma20_3d+",
        "dd5_20d",
        "pnl7",
        "pipeline_zero_held",
    }
)


@dataclass(frozen=True)
class EarlyWarningThresholds:
    yellow_min_signals: int
    yellow_min_executed_tier: int
    red_pnl_pct: float
    red_dd_20d_high: float
    yellow_dd_20d_high: float
    vol_accel_ratio: float
    suggest_yellow: float
    suggest_red_soft: float
    suggest_red: float
    suggest_red_critical: float


@dataclass(frozen=True)
class AlertResult:
    level: AlertLevel
    reasons: tuple[str, ...]
    suggest_reduce_pct: float
    yellow_signal_count: int


def early_warning_thresholds(config: DecisionPackConfig | None = None) -> EarlyWarningThresholds:
    cfg = config or load_decision_pack_config()
    raw = cfg.raw.get("early_warning") or {}
    if not isinstance(raw, dict):
        raw = {}
    suggest = raw.get("suggest_reduce") or {}
    if not isinstance(suggest, dict):
        suggest = {}
    return EarlyWarningThresholds(
        yellow_min_signals=int(raw.get("yellow_min_signals", 2)),
        yellow_min_executed_tier=int(raw.get("yellow_min_executed_tier", 2)),
        red_pnl_pct=float(raw.get("red_pnl_pct", -7.0)),
        red_dd_20d_high=float(raw.get("red_dd_20d_high", -0.05)),
        yellow_dd_20d_high=float(raw.get("yellow_dd_20d_high", -0.02)),
        vol_accel_ratio=float(raw.get("vol_accel_ratio", 1.2)),
        suggest_yellow=float(suggest.get("yellow", 0.30)),
        suggest_red_soft=float(suggest.get("red_soft", 0.30)),
        suggest_red=float(suggest.get("red", 0.50)),
        suggest_red_critical=float(suggest.get("red_critical", 1.00)),
    )


def parse_risk_flags(flags: str | None) -> set[str]:
    return {part for part in str(flags or "").split("|") if part}


def _float_or_none(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not pd.notna(out):
        return None
    return out


def _yellow_signals_from_row(row: dict[str, Any] | pd.Series, flags: set[str]) -> set[str]:
    hits = set(YELLOW_SIGNALS & flags)
    ret_5d = _float_or_none(row.get("ret_5d"))
    ret_20d = _float_or_none(row.get("ret_20d"))
    vol_ratio = _float_or_none(row.get("vol_ratio_5_20"))
    dd = _float_or_none(row.get("dd_20d_high"))
    thresholds = early_warning_thresholds()
    if ret_5d is not None and ret_20d is not None and ret_5d < ret_20d:
        hits.add("mom_fade")
    if vol_ratio is not None and vol_ratio > thresholds.vol_accel_ratio:
        hits.add("vol_accel")
    if dd is not None and dd <= thresholds.yellow_dd_20d_high:
        hits.add("dd_soft")
    return hits


def _is_above_ma20(row: dict[str, Any] | pd.Series) -> bool:
    days_below = _float_or_none(row.get("days_below_ma20"))
    if days_below is not None and days_below > 0:
        return False
    dist = _float_or_none(row.get("dist_ma20_pct"))
    if dist is not None and dist > 0:
        return True
    return days_below == 0


def _red_reasons_from_row(
    row: dict[str, Any] | pd.Series,
    flags: set[str],
    *,
    pipeline_weight: float | None,
    live_weight: float | None,
    thresholds: EarlyWarningThresholds,
) -> list[str]:
    reasons: list[str] = []
    for flag in sorted(RED_FLAGS & flags):
        reasons.append(flag)

    dd = _float_or_none(row.get("dd_20d_high"))
    if dd is not None and dd <= thresholds.red_dd_20d_high and "dd5_20d" not in reasons:
        reasons.append("dd5_20d")

    pnl = _float_or_none(row.get("pnl_pct"))
    if pnl is not None and pnl <= thresholds.red_pnl_pct and "pnl7" not in reasons:
        reasons.append("pnl7")

    lw = live_weight if live_weight is not None else _float_or_none(row.get("weight"))
    pw = pipeline_weight
    if pw is not None and lw is not None and pw <= 0 and lw > 0:
        reasons.append("pipeline_zero_held")

    return reasons


def _red_suggest_pct(
    red_reasons: list[str],
    flags: set[str],
    thresholds: EarlyWarningThresholds,
) -> float:
    if "pnl7" in red_reasons and "deep_below_ma20" in flags:
        return thresholds.suggest_red_critical
    if len(red_reasons) >= 2 or "pnl7" in red_reasons:
        return thresholds.suggest_red
    return thresholds.suggest_red_soft


def _apply_executed_tier_gating(
    level: AlertLevel,
    suggest: float,
    *,
    executed_tier: int | None,
    thresholds: EarlyWarningThresholds,
) -> float:
    if level == "yellow" and suggest > 0:
        min_tier = thresholds.yellow_min_executed_tier
        if executed_tier is None or executed_tier < min_tier:
            return 0.0
    return suggest


def classify_symbol_alert(
    row: dict[str, Any] | pd.Series,
    *,
    pipeline_weight: float | None = None,
    executed_tier: int | None = None,
    config: DecisionPackConfig | None = None,
) -> AlertResult:
    thresholds = early_warning_thresholds(config)
    flags = parse_risk_flags(row.get("risk_flags"))
    yellow_hits = _yellow_signals_from_row(row, flags)
    red_reasons = _red_reasons_from_row(
        row,
        flags,
        pipeline_weight=pipeline_weight,
        live_weight=_float_or_none(row.get("weight")),
        thresholds=thresholds,
    )

    if red_reasons:
        if red_reasons == ["dd5_20d"] and _is_above_ma20(row):
            return AlertResult(
                level="green",
                reasons=(),
                suggest_reduce_pct=0.0,
                yellow_signal_count=len(yellow_hits),
            )
        suggest = _red_suggest_pct(red_reasons, flags, thresholds)
        return AlertResult(
            level="red",
            reasons=tuple(red_reasons),
            suggest_reduce_pct=suggest,
            yellow_signal_count=len(yellow_hits),
        )

    if len(yellow_hits) >= thresholds.yellow_min_signals:
        suggest = _apply_executed_tier_gating(
            "yellow",
            thresholds.suggest_yellow,
            executed_tier=executed_tier,
            thresholds=thresholds,
        )
        return AlertResult(
            level="yellow",
            reasons=tuple(sorted(yellow_hits)),
            suggest_reduce_pct=suggest,
            yellow_signal_count=len(yellow_hits),
        )

    return AlertResult(
        level="green",
        reasons=(),
        suggest_reduce_pct=0.0,
        yellow_signal_count=len(yellow_hits),
    )


def build_early_warning_frame(
    trend_board: pd.DataFrame | None,
    cluster_trend_board: pd.DataFrame | None,
    *,
    pipeline_weights: dict[str, float] | None = None,
    executed_tier: int | None = None,
    config: DecisionPackConfig | None = None,
) -> pd.DataFrame:
    """Merge holdings + cluster boards; classify alerts for held symbols (weight > 0)."""
    frames: list[pd.DataFrame] = []
    if trend_board is not None and not trend_board.empty:
        frames.append(trend_board.copy())
    if cluster_trend_board is not None and not cluster_trend_board.empty:
        frames.append(cluster_trend_board.copy())

    if not frames:
        return pd.DataFrame(
            columns=[
                "code",
                "name",
                "weight",
                "pipeline_weight",
                "alert_level",
                "reasons",
                "red_reason_count",
                "suggest_reduce_pct",
                "pnl_pct",
                "dd_20d_high",
                "dist_ma20_pct",
                "rs_20d_vs_bench",
                "days_below_ma20",
                "ma_stack",
                "risk_flags",
            ]
        )

    merged = pd.concat(frames, ignore_index=True)
    merged["code"] = merged["code"].astype(str)
    merged = merged.drop_duplicates(subset=["code"], keep="first")

    pipe = pipeline_weights or {}
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        weight = _float_or_none(row.get("weight")) or 0.0
        if weight <= 0:
            continue
        code = str(row["code"])
        pw = pipe.get(code)
        alert = classify_symbol_alert(
            row,
            pipeline_weight=pw,
            executed_tier=executed_tier,
            config=config,
        )
        if alert.level == "green":
            continue
        rows.append(
            {
                "code": code,
                "name": row.get("name"),
                "weight": weight,
                "pipeline_weight": pw,
                "alert_level": alert.level,
                "reasons": "|".join(alert.reasons),
                "red_reason_count": len(alert.reasons) if alert.level == "red" else 0,
                "suggest_reduce_pct": alert.suggest_reduce_pct,
                "pnl_pct": row.get("pnl_pct"),
                "dd_20d_high": row.get("dd_20d_high"),
                "dist_ma20_pct": row.get("dist_ma20_pct"),
                "rs_20d_vs_bench": row.get("rs_20d_vs_bench"),
                "days_below_ma20": row.get("days_below_ma20"),
                "ma_stack": row.get("ma_stack"),
                "risk_flags": row.get("risk_flags"),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    level_order = {"red": 0, "yellow": 1}
    frame["_sort"] = frame["alert_level"].map(level_order)
    frame = frame.sort_values(["_sort", "suggest_reduce_pct"], ascending=[True, False]).drop(
        columns=["_sort"]
    )
    return frame.reset_index(drop=True)


__all__ = [
    "AlertResult",
    "AlertLevel",
    "EarlyWarningThresholds",
    "build_early_warning_frame",
    "classify_symbol_alert",
    "early_warning_thresholds",
    "parse_risk_flags",
]
