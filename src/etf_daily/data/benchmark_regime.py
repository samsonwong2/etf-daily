"""Shared HS300 benchmark regime features for L3 risk controls and forecast-mu layers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .price_adjustments import auto_qfq_adjust_close_panel

BENCHMARK_CODE = "sh510300"

STATE_OFFENSIVE = "offensive"
STATE_CAUTION = "caution"
STATE_DEFENSIVE = "defensive"
STATE_STRESS = "stress"

STATE_RANK = {
    STATE_OFFENSIVE: 0,
    STATE_CAUTION: 1,
    STATE_DEFENSIVE: 2,
    STATE_STRESS: 3,
}


@dataclass(frozen=True)
class BenchmarkRegimeConfig:
    fast_ma: int = 20
    slow_ma: int = 60
    vol_window: int = 20
    vol_lookback: int = 120
    budget_min: float = 0.50
    budget_offensive: float = 1.00
    budget_caution: float = 0.92
    budget_defensive: float = 0.82
    budget_stress: float = 0.65
    drawdown_window: int = 60
    drawdown_stress_threshold: float = 0.10
    downgrade_confirm_days: int = 3
    upgrade_confirm_days: int = 1
    upgrade_confirm_days_from_stress: int = 2
    stress_min_hold_days: int = 3


def compute_benchmark_features(
    bench_close: pd.Series,
    *,
    config: BenchmarkRegimeConfig | None = None,
) -> pd.DataFrame:
    """Compute causal benchmark trend/vol/drawdown features indexed by business dates."""
    cfg = config or BenchmarkRegimeConfig()
    close = pd.to_numeric(bench_close, errors="coerce").sort_index()
    close = close.reindex(pd.date_range(close.index.min(), close.index.max(), freq="B")).ffill()

    fast_ma = close.rolling(int(cfg.fast_ma), min_periods=max(5, int(cfg.fast_ma) // 4)).mean()
    slow_ma = close.rolling(int(cfg.slow_ma), min_periods=max(10, int(cfg.slow_ma) // 6)).mean()
    bench_return = close.pct_change(fill_method=None)
    bench_vol = bench_return.rolling(int(cfg.vol_window), min_periods=max(5, int(cfg.vol_window) // 4)).std()
    vol_reference = bench_vol.rolling(int(cfg.vol_lookback), min_periods=20).median().shift(1)
    rolling_high = close.rolling(int(cfg.drawdown_window), min_periods=max(10, int(cfg.drawdown_window) // 6)).max()
    drawdown = 1.0 - (close / rolling_high)

    trend_on = (close > fast_ma) & (close > slow_ma)
    trend_structure_up = fast_ma > slow_ma
    high_vol = bench_vol > vol_reference
    drawdown_stress = drawdown >= float(cfg.drawdown_stress_threshold)
    close_fast_ratio = close / fast_ma

    out = pd.DataFrame(
        {
            "hs300_close": close,
            "risk_bench_trend_on": trend_on.astype(float),
            "risk_bench_trend_structure_up": trend_structure_up.astype(float),
            "risk_bench_vol_stress": high_vol.astype(float),
            "risk_bench_drawdown_stress": drawdown_stress.astype(float),
            "risk_bench_trend_score": close_fast_ratio,
            "risk_bench_vol_20d": bench_vol,
            "risk_bench_vol_ref": vol_reference,
            "risk_bench_drawdown": drawdown,
        },
        index=close.index,
    )
    return out


def macro_budget_from_features(
    trend_on: bool,
    high_vol: bool,
    drawdown_stress: bool,
    *,
    trend_structure_up: bool | None = None,
    config: BenchmarkRegimeConfig | None = None,
) -> tuple[float, str]:
    """Map instantaneous benchmark features to macro budget tier (pre-hysteresis)."""
    cfg = config or BenchmarkRegimeConfig()
    structure_up = trend_on if trend_structure_up is None else trend_structure_up
    trend_weak = not structure_up
    stress_signals = [trend_weak, high_vol, drawdown_stress]
    stress_count = sum(stress_signals)

    if stress_count >= 2:
        if drawdown_stress:
            return float(cfg.budget_min), STATE_STRESS
        return float(cfg.budget_stress), STATE_STRESS

    if structure_up and not drawdown_stress and not high_vol:
        return float(cfg.budget_offensive), STATE_OFFENSIVE
    if trend_on and not high_vol:
        return float(cfg.budget_offensive), STATE_OFFENSIVE
    if trend_on and high_vol:
        return float(cfg.budget_caution), STATE_CAUTION
    if structure_up and not drawdown_stress and high_vol:
        return float(cfg.budget_offensive), STATE_OFFENSIVE
    if not trend_on and not high_vol:
        return float(cfg.budget_defensive), STATE_DEFENSIVE
    return float(cfg.budget_stress), STATE_STRESS


def _apply_state_hysteresis(
    raw_labels: list[str],
    *,
    config: BenchmarkRegimeConfig | None = None,
) -> list[str]:
    """Smooth raw macro states: slower downgrade, faster upgrade, min stress hold."""
    cfg = config or BenchmarkRegimeConfig()
    if not raw_labels:
        return []

    current = raw_labels[0]
    downgrade_streak = 0
    upgrade_streak = 0
    stress_hold_remaining = 0
    smoothed: list[str] = []

    for raw_label in raw_labels:
        raw_rank = STATE_RANK[raw_label]
        cur_rank = STATE_RANK[current]

        if raw_rank > cur_rank:
            downgrade_streak += 1
            upgrade_streak = 0
            if downgrade_streak >= int(cfg.downgrade_confirm_days):
                current = raw_label
                downgrade_streak = 0
                if current == STATE_STRESS:
                    stress_hold_remaining = int(cfg.stress_min_hold_days)
        elif raw_rank < cur_rank:
            upgrade_streak += 1
            downgrade_streak = 0
            confirm_days = (
                int(cfg.upgrade_confirm_days_from_stress)
                if cur_rank >= STATE_RANK[STATE_DEFENSIVE]
                else int(cfg.upgrade_confirm_days)
            )
            if stress_hold_remaining > 0:
                upgrade_streak = 0
            elif upgrade_streak >= confirm_days:
                current = raw_label
                upgrade_streak = 0
        else:
            downgrade_streak = 0
            upgrade_streak = 0

        if stress_hold_remaining > 0:
            stress_hold_remaining = max(0, stress_hold_remaining - 1)

        smoothed.append(current)

    return smoothed


def _budget_for_state(label: str, *, config: BenchmarkRegimeConfig | None = None) -> float:
    cfg = config or BenchmarkRegimeConfig()
    mapping = {
        STATE_OFFENSIVE: cfg.budget_offensive,
        STATE_CAUTION: cfg.budget_caution,
        STATE_DEFENSIVE: cfg.budget_defensive,
        STATE_STRESS: cfg.budget_stress,
    }
    return float(mapping.get(label, cfg.budget_stress))


def build_macro_budget_series(
    features: pd.DataFrame,
    *,
    config: BenchmarkRegimeConfig | None = None,
) -> pd.DataFrame:
    cfg = config or BenchmarkRegimeConfig()
    raw_rows: list[dict[str, object]] = []
    raw_labels: list[str] = []

    for ts, row in features.iterrows():
        trend_on = bool(row.get("risk_bench_trend_on", 0.0))
        trend_structure_up = bool(row.get("risk_bench_trend_structure_up", trend_on))
        high_vol = bool(row.get("risk_bench_vol_stress", 0.0))
        drawdown_stress = bool(row.get("risk_bench_drawdown_stress", 0.0))
        budget, label = macro_budget_from_features(
            trend_on,
            high_vol,
            drawdown_stress,
            trend_structure_up=trend_structure_up,
            config=cfg,
        )
        raw_labels.append(label)
        raw_rows.append(
            {
                "datetime": pd.Timestamp(ts),
                "risk_macro_budget_raw": budget,
                "risk_bench_state_label_raw": label,
                "drawdown_stress": drawdown_stress,
                "trend_on": trend_on,
                "high_vol": high_vol,
            }
        )

    smoothed_labels = _apply_state_hysteresis(raw_labels, config=cfg)
    rows: list[dict[str, object]] = []
    for raw, label in zip(raw_rows, smoothed_labels, strict=True):
        drawdown_stress = bool(raw["drawdown_stress"])
        trend_on = bool(raw["trend_on"])
        high_vol = bool(raw["high_vol"])
        budget = _budget_for_state(label, config=cfg)
        if label == STATE_STRESS and drawdown_stress:
            budget = float(cfg.budget_min)
        rows.append(
            {
                "datetime": raw["datetime"],
                "risk_macro_budget": budget,
                "risk_bench_state_label": label,
                "risk_regime_on": float(trend_on and not high_vol and not drawdown_stress),
                "risk_macro_budget_raw": raw["risk_macro_budget_raw"],
                "risk_bench_state_label_raw": raw["risk_bench_state_label_raw"],
            }
        )
    return pd.DataFrame(rows)


def vol_target_for_macro_budget(
    macro_budget: float,
    *,
    offensive: float,
    neutral: float,
    defensive: float,
    offensive_threshold: float = 0.90,
    stress_threshold: float = 0.65,
) -> float:
    """Map macro budget to vol target; only deep stress uses the defensive vol tier."""
    budget = float(macro_budget)
    if budget >= float(offensive_threshold):
        return float(offensive)
    if budget > float(stress_threshold):
        return float(neutral)
    return float(defensive)


def load_hs300_close(
    provider_uri: str,
    start_date: str,
    end_date: str,
    *,
    warmup_bdays: int = 180,
) -> pd.Series:
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    qlib.init(provider_uri=provider_uri, region=REG_CN)
    warmup_start = (pd.Timestamp(start_date) - pd.tseries.offsets.BDay(int(warmup_bdays))).strftime("%Y-%m-%d")
    bench_raw: pd.DataFrame = D.features(
        [BENCHMARK_CODE],
        ["$close"],
        start_time=warmup_start,
        end_time=end_date,
    )
    bench_close = (
        bench_raw["$close"]
        .reset_index(level=0, drop=True)
        .rename("hs300_close")
        .sort_index()
    )
    bench_close.index = pd.to_datetime(bench_close.index)
    bench_frame = bench_close.to_frame(name=BENCHMARK_CODE)
    adjusted, _ = auto_qfq_adjust_close_panel(bench_frame)
    bench_close = adjusted[BENCHMARK_CODE]
    return bench_close.reindex(pd.date_range(bench_close.index.min(), bench_close.index.max(), freq="B")).ffill()


def lookup_series_value(series: pd.Series | None, dt: pd.Timestamp) -> float:
    if series is None or series.empty:
        return float("nan")
    available = series.index[series.index <= pd.Timestamp(dt)]
    if len(available) == 0:
        return float("nan")
    value = series.loc[available[-1]]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def lookup_label_value(labels: pd.Series | None, dt: pd.Timestamp) -> str:
    if labels is None or labels.empty:
        return ""
    available = labels.index[labels.index <= pd.Timestamp(dt)]
    if len(available) == 0:
        return ""
    return str(labels.loc[available[-1]])
