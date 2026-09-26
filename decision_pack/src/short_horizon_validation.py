"""Walk-forward validation for MLFAM short-horizon board metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from decision_pack.src.short_horizon_mlfam import (
    ShortHorizonConfig,
    compute_short_horizon_metrics,
    forward_cumulative_log_return,
    forward_log_returns,
    realize_triple_barrier,
)


@dataclass(frozen=True)
class ValidationSummary:
    n_obs: int
    n_symbols: int
    eval_start: str
    eval_end: str
    horizon: int
    step_days: int
    # triple-barrier
    actual_up_rate: float
    actual_down_rate: float
    actual_timeout_rate: float
    mean_pred_p_up: float
    mean_pred_p_down: float
    mean_pred_p_timeout: float
    brier_score: float
    # quantile coverage (target 5/50/95)
    coverage_q05: float
    coverage_q50: float
    coverage_q95: float
    # trend scan (H-day cumulative log return sign)
    trend_up_pos_rate: float
    trend_down_neg_rate: float
    trend_flat_abs_rate: float
    n_trend_up: int
    n_trend_down: int
    n_trend_flat: int
    # meta-label bets
    n_bets: int
    bet_win_rate: float
    bet_mean_h_return: float


def _multiclass_brier(
    pred_up: np.ndarray,
    pred_down: np.ndarray,
    pred_timeout: np.ndarray,
    actual: np.ndarray,
) -> float:
    one_hot = np.zeros((actual.size, 3), dtype=float)
    for i, label in enumerate(actual):
        if label == "up":
            one_hot[i, 0] = 1.0
        elif label == "down":
            one_hot[i, 1] = 1.0
        else:
            one_hot[i, 2] = 1.0
    pred = np.column_stack([pred_up, pred_down, pred_timeout])
    return float(np.mean(np.sum((pred - one_hot) ** 2, axis=1)))


def run_walk_forward_validation(
    close_panel: pd.DataFrame,
    *,
    eval_dates: pd.DatetimeIndex,
    cfg: ShortHorizonConfig | None = None,
    flags_by_code: dict[str, tuple[str, ...]] | None = None,
) -> tuple[pd.DataFrame, ValidationSummary]:
    """Compare board metrics at each eval date with realized H-day outcomes."""
    cfg = cfg or ShortHorizonConfig()
    flags_by_code = flags_by_code or {}

    rows: list[dict[str, Any]] = []
    for code in close_panel.columns:
        series = pd.to_numeric(close_panel[code], errors="coerce").dropna()
        if series.size < cfg.min_samples + cfg.horizon + 5:
            continue
        flag_tuple = flags_by_code.get(str(code), ())

        for end in eval_dates:
            if end not in series.index and end > series.index.max():
                continue
            if end > series.index.max():
                continue

            metrics = compute_short_horizon_metrics(
                series,
                pd.Timestamp(end),
                cfg=cfg,
                flags=flag_tuple,
            )
            if not metrics.reliable or metrics.barrier is None:
                continue

            fwd = forward_log_returns(series, pd.Timestamp(end), horizon=cfg.horizon)
            if fwd is None:
                continue

            outcome = realize_triple_barrier(fwd, barrier=float(metrics.barrier))
            cum_h = forward_cumulative_log_return(fwd)
            if cum_h is None:
                continue

            win = False
            if metrics.direction == "up":
                win = cum_h > 0
            elif metrics.direction == "down":
                win = cum_h < 0

            rows.append(
                {
                    "code": str(code),
                    "as_of": pd.Timestamp(end).strftime("%Y-%m-%d"),
                    "trend_label": metrics.trend_label,
                    "trend_t": metrics.trend_t,
                    "trend_p": metrics.trend_p,
                    "p_up": metrics.p_up,
                    "p_down": metrics.p_down,
                    "p_timeout": metrics.p_timeout,
                    "barrier": metrics.barrier,
                    "direction": metrics.direction,
                    "confidence": metrics.confidence,
                    "bet_size": metrics.bet_size,
                    "q05": metrics.q05,
                    "q50": metrics.q50,
                    "q95": metrics.q95,
                    "actual_outcome": outcome,
                    "actual_cum_h": cum_h,
                    "bet_win": win if metrics.direction in ("up", "down") else np.nan,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        empty = ValidationSummary(
            n_obs=0,
            n_symbols=0,
            eval_start="",
            eval_end="",
            horizon=cfg.horizon,
            step_days=0,
            actual_up_rate=0.0,
            actual_down_rate=0.0,
            actual_timeout_rate=0.0,
            mean_pred_p_up=0.0,
            mean_pred_p_down=0.0,
            mean_pred_p_timeout=0.0,
            brier_score=0.0,
            coverage_q05=0.0,
            coverage_q50=0.0,
            coverage_q95=0.0,
            trend_up_pos_rate=0.0,
            trend_down_neg_rate=0.0,
            trend_flat_abs_rate=0.0,
            n_trend_up=0,
            n_trend_down=0,
            n_trend_flat=0,
            n_bets=0,
            bet_win_rate=0.0,
            bet_mean_h_return=0.0,
        )
        return frame, empty

    actual = frame["actual_outcome"].to_numpy()
    pred_up = frame["p_up"].astype(float).to_numpy()
    pred_down = frame["p_down"].astype(float).to_numpy()
    pred_timeout = frame["p_timeout"].astype(float).to_numpy()

    up_mask = frame["trend_label"] == "up"
    down_mask = frame["trend_label"] == "down"
    flat_mask = frame["trend_label"] == "flat"

    bets = frame[frame["direction"].isin(["up", "down"])].copy()

    summary = ValidationSummary(
        n_obs=len(frame),
        n_symbols=int(frame["code"].nunique()),
        eval_start=str(frame["as_of"].min()),
        eval_end=str(frame["as_of"].max()),
        horizon=cfg.horizon,
        step_days=0,
        actual_up_rate=float((actual == "up").mean()),
        actual_down_rate=float((actual == "down").mean()),
        actual_timeout_rate=float((actual == "timeout").mean()),
        mean_pred_p_up=float(pred_up.mean()),
        mean_pred_p_down=float(pred_down.mean()),
        mean_pred_p_timeout=float(pred_timeout.mean()),
        brier_score=_multiclass_brier(pred_up, pred_down, pred_timeout, actual),
        coverage_q05=float((frame["actual_cum_h"] <= frame["q05"]).mean()),
        coverage_q50=float((frame["actual_cum_h"] <= frame["q50"]).mean()),
        coverage_q95=float((frame["actual_cum_h"] <= frame["q95"]).mean()),
        trend_up_pos_rate=float((frame.loc[up_mask, "actual_cum_h"] > 0).mean())
        if up_mask.any()
        else 0.0,
        trend_down_neg_rate=float((frame.loc[down_mask, "actual_cum_h"] < 0).mean())
        if down_mask.any()
        else 0.0,
        trend_flat_abs_rate=float((frame.loc[flat_mask, "actual_cum_h"].abs() < 0.01).mean())
        if flat_mask.any()
        else 0.0,
        n_trend_up=int(up_mask.sum()),
        n_trend_down=int(down_mask.sum()),
        n_trend_flat=int(flat_mask.sum()),
        n_bets=len(bets),
        bet_win_rate=float(bets["bet_win"].mean()) if len(bets) else 0.0,
        bet_mean_h_return=float(bets["actual_cum_h"].mean()) if len(bets) else 0.0,
    )
    return frame, summary


def build_eval_dates(
    close_panel: pd.DataFrame,
    *,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    step_days: int,
) -> pd.DatetimeIndex:
    """Trading dates in [eval_start, eval_end] sampled every ``step_days``."""
    idx = close_panel.index
    idx = idx[(idx >= eval_start) & (idx <= eval_end)]
    if idx.empty:
        return pd.DatetimeIndex([])
    if step_days <= 1:
        return idx
    return idx[::step_days]


def per_symbol_validation(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate validation metrics per symbol."""
    if frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for code, grp in frame.groupby("code"):
        actual = grp["actual_outcome"].to_numpy()
        bets = grp[grp["direction"].isin(["up", "down"])]
        rows.append(
            {
                "code": code,
                "n_obs": len(grp),
                "actual_up_rate": float((actual == "up").mean()),
                "mean_pred_p_up": float(grp["p_up"].mean()),
                "brier": _multiclass_brier(
                    grp["p_up"].astype(float).to_numpy(),
                    grp["p_down"].astype(float).to_numpy(),
                    grp["p_timeout"].astype(float).to_numpy(),
                    actual,
                ),
                "coverage_q05": float((grp["actual_cum_h"] <= grp["q05"]).mean()),
                "coverage_q50": float((grp["actual_cum_h"] <= grp["q50"]).mean()),
                "coverage_q95": float((grp["actual_cum_h"] <= grp["q95"]).mean()),
                "trend_up_pos_rate": float(
                    (grp.loc[grp["trend_label"] == "up", "actual_cum_h"] > 0).mean()
                )
                if (grp["trend_label"] == "up").any()
                else np.nan,
                "n_bets": len(bets),
                "bet_win_rate": float(bets["bet_win"].mean()) if len(bets) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("n_obs", ascending=False)


def format_validation_report(
    summary: ValidationSummary,
    *,
    pack_dir: str,
    as_of_board: str,
    per_symbol: pd.DataFrame,
    step_days: int = 1,
) -> str:
    lines = [
        "# Short-Horizon Board Walk-Forward Validation",
        "",
        f"**Board as_of:** {as_of_board}  ",
        f"**Pack dir:** `{pack_dir}`  ",
        f"**Horizon H:** {summary.horizon} trading days  ",
        f"**Eval window:** {summary.eval_start} → {summary.eval_end} (step={step_days}d)  ",
        f"**Observations:** {summary.n_obs} ({summary.n_symbols} symbols)  ",
        "",
        "## 1. Triple-Barrier Calibration",
        "",
        "Compare predicted MC probabilities vs realized first-touch frequencies.",
        "",
        "| 事件 | 预测均值 | 实际频率 | 理想校准 |",
        "|------|----------|----------|----------|",
        f"| 触轨↑ | {summary.mean_pred_p_up:.1%} | {summary.actual_up_rate:.1%} | 接近 |",
        f"| 触轨↓ | {summary.mean_pred_p_down:.1%} | {summary.actual_down_rate:.1%} | 接近 |",
        f"| 到期 | {summary.mean_pred_p_timeout:.1%} | {summary.actual_timeout_rate:.1%} | 接近 |",
        f"| **Brier score** | — | **{summary.brier_score:.4f}** | 越低越好（0=完美） |",
        "",
        "> Brier 对三分类概率评分；0.67≈随机猜，<0.60 通常有一定校准价值。",
        "",
        "## 2. DGP Quantile Coverage",
        "",
        "若 MC 分位数校准良好，实际 H 日累计收益应约有 5%/50%/95% 落在 q05/q50/q95 以下。",
        "",
        "| 分位 | 目标覆盖率 | 实际覆盖率 |",
        "|------|------------|------------|",
        f"| q05 | 5% | **{summary.coverage_q05:.1%}** |",
        f"| q50 | 50% | **{summary.coverage_q50:.1%}** |",
        f"| q95 | 95% | **{summary.coverage_q95:.1%}** |",
        "",
        "## 3. Trend Scan (H-day cumulative return)",
        "",
        "| 趋势标签 | 样本数 | 命中率 | 定义 |",
        "|----------|--------|--------|------|",
        f"| 上升 | {summary.n_trend_up} | {summary.trend_up_pos_rate:.1%} | H 日收益>0 |",
        f"| 下降 | {summary.n_trend_down} | {summary.trend_down_neg_rate:.1%} | H 日收益<0 |",
        f"| 无趋势 | {summary.n_trend_flat} | {summary.trend_flat_abs_rate:.1%} | |H 日收益|<1% |",
        "",
        "## 4. Meta-Label Bets (direction ≠ 不下注)",
        "",
        f"- 下注次数: **{summary.n_bets}**",
        f"- 方向胜率（H 日收益与方向一致）: **{summary.bet_win_rate:.1%}**",
        f"- 下注样本 H 日平均累计收益: **{summary.bet_mean_h_return:.2%}**",
        "",
        "## 5. Per-Symbol Summary (top 15 by n_obs)",
        "",
    ]
    if not per_symbol.empty:
        lines.append("| code | n | actual↑ | pred↑ | Brier | cov q50 | bet win |")
        lines.append("|------|---|---------|-------|-------|---------|---------|")
        for _, row in per_symbol.head(15).iterrows():
            bwr = row["bet_win_rate"]
            bwr_s = f"{bwr:.1%}" if pd.notna(bwr) else "—"
            lines.append(
                f"| {row['code']} | {int(row['n_obs'])} | "
                f"{row['actual_up_rate']:.1%} | {row['mean_pred_p_up']:.1%} | "
                f"{row['brier']:.3f} | {row['coverage_q50']:.1%} | {bwr_s} |"
            )
    else:
        lines.append("_No per-symbol data._")

    lines.extend(
        [
            "",
            "## 解读说明",
            "",
            "- 本验证用 **walk-forward**：每个 as_of 日只用该日之前数据算指标，再对比之后 H 日真实路径。",
            "- **2026-06-25 当日 board 行本身无法验证**（需等 H 个交易日后的未来数据）。",
            "- MC 使用 IID bootstrap，不嵌入 momentum；校准偏差在趋势突变期可能变大。",
            "",
            "---",
            "",
            "*由 `decision_pack/scripts/validate_short_horizon_board.py` 生成。*",
            "",
        ]
    )
    return "\n".join(lines)
