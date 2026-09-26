"""Walk-forward validation for GARCH price targets (TP/SL/median)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from decision_pack.src.garch_price_targets import price_from_log_return
from decision_pack.src.garch_short_horizon_board import (
    GarchShortHorizonConfig,
    compute_garch_short_horizon_metrics,
)
from decision_pack.src.garch_short_horizon_validation import _multiclass_brier
from decision_pack.src.short_horizon_mlfam import (
    forward_cumulative_log_return,
    forward_log_returns,
    realize_triple_barrier,
)


@dataclass(frozen=True)
class GarchPriceTargetsValidationSummary:
    horizon: int
    n_obs: int
    n_symbols: int
    eval_start: str
    eval_end: str
    mean_pred_p_tp: float
    mean_pred_p_sl: float
    mean_pred_p_timeout: float
    actual_tp_rate: float
    actual_sl_rate: float
    actual_timeout_rate: float
    brier_score: float
    median_coverage: float
    median_mae_pct: float
    q05_coverage: float
    q95_coverage: float
    strategy_mean_simple_ret: float
    buy_hold_mean_simple_ret: float
    strategy_win_rate: float
    buy_hold_win_rate: float
    timeout_mean_simple_ret: float


def _strategy_log_return(outcome: str, barrier: float, cum_h: float) -> float:
    if outcome == "up":
        return float(barrier)
    if outcome == "down":
        return float(-barrier)
    return float(cum_h)


def _simple_return(log_ret: float) -> float:
    return math.expm1(log_ret)


def run_garch_price_targets_validation(
    close_panel: pd.DataFrame,
    *,
    eval_dates: pd.DatetimeIndex,
    horizon: int,
    cfg: GarchShortHorizonConfig | None = None,
    flags_by_code: dict[str, tuple[str, ...]] | None = None,
) -> tuple[pd.DataFrame, GarchPriceTargetsValidationSummary]:
    cfg = cfg or GarchShortHorizonConfig()
    flags_by_code = flags_by_code or {}
    min_samples = max(cfg.min_samples, cfg.garch.min_train)
    rows: list[dict[str, Any]] = []

    for code in close_panel.columns:
        series = pd.to_numeric(close_panel[code], errors="coerce").dropna()
        if series.size < min_samples + horizon + 5:
            continue
        flag_tuple = flags_by_code.get(str(code), ())

        for end in eval_dates:
            if end > series.index.max():
                continue
            ts_end = pd.Timestamp(end)
            available = series.loc[series.index <= ts_end].dropna()
            if available.empty:
                continue
            close = float(available.iloc[-1])
            if close <= 0 or not math.isfinite(close):
                continue

            metrics = compute_garch_short_horizon_metrics(
                series,
                ts_end,
                horizon=horizon,
                cfg=cfg,
                flags=flag_tuple,
            )
            if metrics.barrier is None or metrics.p_up is None or metrics.q50 is None:
                continue
            if metrics.q05 is None or metrics.q95 is None:
                continue

            fwd = forward_log_returns(series, ts_end, horizon=horizon)
            if fwd is None:
                continue
            outcome = realize_triple_barrier(fwd, barrier=float(metrics.barrier))
            cum_h = forward_cumulative_log_return(fwd)
            if cum_h is None:
                continue

            barrier = float(metrics.barrier)
            tp_price = price_from_log_return(close, barrier)
            sl_price = price_from_log_return(close, -barrier)
            median_price = price_from_log_return(close, metrics.q50)
            q05_price = price_from_log_return(close, metrics.q05)
            q95_price = price_from_log_return(close, metrics.q95)
            actual_end_price = close * math.exp(cum_h)

            strat_log = _strategy_log_return(outcome, barrier, cum_h)
            median_mae_pct = (
                abs(actual_end_price - median_price) / close if median_price else np.nan
            )

            rows.append(
                {
                    "code": str(code),
                    "as_of": ts_end.strftime("%Y-%m-%d"),
                    "H": horizon,
                    "close": close,
                    "tp_price": tp_price,
                    "p_tp": metrics.p_up,
                    "sl_price": sl_price,
                    "p_sl": metrics.p_down,
                    "median_price": median_price,
                    "p_timeout": metrics.p_timeout,
                    "barrier_log": barrier,
                    "q05_log": metrics.q05,
                    "q50_log": metrics.q50,
                    "q95_log": metrics.q95,
                    "q05_price": q05_price,
                    "q95_price": q95_price,
                    "actual_outcome": outcome,
                    "actual_cum_h": cum_h,
                    "actual_end_price": actual_end_price,
                    "actual_above_median": float(cum_h >= metrics.q50),
                    "strategy_log_ret": strat_log,
                    "buy_hold_log_ret": cum_h,
                    "strategy_simple_ret": _simple_return(strat_log),
                    "buy_hold_simple_ret": _simple_return(cum_h),
                    "median_mae_pct": median_mae_pct,
                    "reliable": metrics.reliable,
                }
            )

    frame = pd.DataFrame(rows)
    empty = GarchPriceTargetsValidationSummary(
        horizon=horizon,
        n_obs=0,
        n_symbols=0,
        eval_start="",
        eval_end="",
        mean_pred_p_tp=0.0,
        mean_pred_p_sl=0.0,
        mean_pred_p_timeout=0.0,
        actual_tp_rate=0.0,
        actual_sl_rate=0.0,
        actual_timeout_rate=0.0,
        brier_score=0.0,
        median_coverage=0.0,
        median_mae_pct=0.0,
        q05_coverage=0.0,
        q95_coverage=0.0,
        strategy_mean_simple_ret=0.0,
        buy_hold_mean_simple_ret=0.0,
        strategy_win_rate=0.0,
        buy_hold_win_rate=0.0,
        timeout_mean_simple_ret=0.0,
    )
    if frame.empty:
        return frame, empty

    actual = frame["actual_outcome"].to_numpy()
    timeout_mask = actual == "timeout"
    summary = GarchPriceTargetsValidationSummary(
        horizon=horizon,
        n_obs=len(frame),
        n_symbols=int(frame["code"].nunique()),
        eval_start=str(frame["as_of"].min()),
        eval_end=str(frame["as_of"].max()),
        mean_pred_p_tp=float(frame["p_tp"].mean()),
        mean_pred_p_sl=float(frame["p_sl"].mean()),
        mean_pred_p_timeout=float(frame["p_timeout"].mean()),
        actual_tp_rate=float((actual == "up").mean()),
        actual_sl_rate=float((actual == "down").mean()),
        actual_timeout_rate=float((actual == "timeout").mean()),
        brier_score=_multiclass_brier(
            frame["p_tp"].astype(float).to_numpy(),
            frame["p_sl"].astype(float).to_numpy(),
            frame["p_timeout"].astype(float).to_numpy(),
            actual,
        ),
        median_coverage=float(frame["actual_above_median"].mean()),
        median_mae_pct=float(frame["median_mae_pct"].mean()),
        q05_coverage=float((frame["actual_cum_h"] <= frame["q05_log"]).mean()),
        q95_coverage=float((frame["actual_cum_h"] <= frame["q95_log"]).mean()),
        strategy_mean_simple_ret=float(frame["strategy_simple_ret"].mean()),
        buy_hold_mean_simple_ret=float(frame["buy_hold_simple_ret"].mean()),
        strategy_win_rate=float((frame["strategy_simple_ret"] > 0).mean()),
        buy_hold_win_rate=float((frame["buy_hold_simple_ret"] > 0).mean()),
        timeout_mean_simple_ret=float(frame.loc[timeout_mask, "buy_hold_simple_ret"].mean())
        if timeout_mask.any()
        else 0.0,
    )
    return frame, summary


def format_price_targets_validation_report(
    summaries: dict[int, GarchPriceTargetsValidationSummary],
    *,
    pack_dir: str,
    as_of_board: str,
    step_days: int,
) -> str:
    lines = [
        "# GARCH Price Targets Walk-Forward Backtest",
        "",
        f"**Board as_of:** {as_of_board}  ",
        f"**Pack dir:** `{pack_dir}`  ",
        f"**Step:** {step_days} trading days  ",
        "",
        "模拟规则：在 `as_of` 收盘价建仓；H 日内**先触** tp_price / sl_price（对数障宽）",
        "则按障宽止盈/止损平仓；否则 H 日到期按实际收盘价平仓。",
        "",
        "## 1. 三重门概率校准（预测 vs 实际）",
        "",
        "| H | 预测止盈 | 实际止盈 | 预测止损 | 实际止损 | 预测到期 | 实际到期 | Brier |",
        "|---|----------|----------|----------|----------|----------|----------|-------|",
    ]
    for h, s in sorted(summaries.items()):
        if s.n_obs == 0:
            continue
        lines.append(
            f"| {h} | {s.mean_pred_p_tp:.1%} | {s.actual_tp_rate:.1%} | "
            f"{s.mean_pred_p_sl:.1%} | {s.actual_sl_rate:.1%} | "
            f"{s.mean_pred_p_timeout:.1%} | {s.actual_timeout_rate:.1%} | "
            f"{s.brier_score:.3f} |"
        )
    lines.extend(
        [
            "",
            "> Brier 越低越好（均匀猜≈0.67）。预测/实际接近表示触轨概率可用。",
            "",
            "## 2. 中位数价位（q50）校准",
            "",
            "| H | 目标覆盖率 | 实际≥中位价 | 中位价 MAE(% of close) | n_obs |",
            "|---|------------|-------------|------------------------|-------|",
        ]
    )
    for h, s in sorted(summaries.items()):
        if s.n_obs == 0:
            continue
        lines.append(
            f"| {h} | 50% | {s.median_coverage:.1%} | {s.median_mae_pct:.2%} | {s.n_obs} |"
        )
    lines.extend(
        [
            "",
            "## 3. 分位覆盖（q05 / q95 对数收益）",
            "",
            "| H | q05 目标 | q05 实际 | q95 目标 | q95 实际 |",
            "|---|----------|----------|----------|----------|",
        ]
    )
    for h, s in sorted(summaries.items()):
        if s.n_obs == 0:
            continue
        lines.append(
            f"| {h} | 5% | {s.q05_coverage:.1%} | 95% | {s.q95_coverage:.1%} |"
        )
    lines.extend(
        [
            "",
            "## 4. 按价位表止损止盈 vs 买入持有",
            "",
            "| H | 策略均收益 | 持有均收益 | 策略胜率 | 持有胜率 | 到期子样本均收益 |",
            "|---|------------|------------|----------|----------|------------------|",
        ]
    )
    for h, s in sorted(summaries.items()):
        if s.n_obs == 0:
            continue
        lines.append(
            f"| {h} | {s.strategy_mean_simple_ret:.2%} | {s.buy_hold_mean_simple_ret:.2%} | "
            f"{s.strategy_win_rate:.1%} | {s.buy_hold_win_rate:.1%} | "
            f"{s.timeout_mean_simple_ret:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 解读",
            "",
            "- **H=1**：下一交易日 tp/sl；**H=5/10/20**：对应持有窗口。",
            "- 策略收益：先触上轨 = +障宽简单收益；先触下轨 = -障宽；到期 = H 日实际收益。",
            "- 若实际触轨率系统性低于预测，说明障宽偏宽或 MC 仍偏乐观。",
            "- 中位价 MAE 反映 q50 点估计精度；覆盖率偏离 50% 表示分布偏斜。",
            "",
            "---",
            "",
            "*由 `decision_pack/scripts/validate_garch_price_targets.py` 生成。*",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "GarchPriceTargetsValidationSummary",
    "format_price_targets_validation_report",
    "run_garch_price_targets_validation",
]
