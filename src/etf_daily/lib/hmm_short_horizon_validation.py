"""Walk-forward validation for HMM short-horizon board metrics (ftsnotes §34)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from etf_daily.lib.hmm_short_horizon import (
    HmmShortHorizonConfig,
    compute_hmm_short_horizon_metrics,
)
from etf_daily.lib.short_horizon_mlfam import (
    forward_cumulative_log_return,
    forward_log_returns,
    realize_triple_barrier,
)
from etf_daily.lib.short_horizon_validation import _multiclass_brier


@dataclass(frozen=True)
class HmmValidationSummary:
    n_obs: int
    n_symbols: int
    eval_start: str
    eval_end: str
    horizon: int
    step_days: int
    actual_up_rate: float
    actual_down_rate: float
    actual_timeout_rate: float
    mean_pred_p_up: float
    mean_pred_p_down: float
    mean_pred_p_timeout: float
    brier_score: float
    coverage_q05: float
    coverage_q50: float
    coverage_q95: float
    mean_p_high: float
    high_regime_rate: float
    # When predicted high-vol regime: mean |H-day cum ret| vs low-regime
    mean_abs_h_when_high: float
    mean_abs_h_when_low: float
    mean_sigma_mix_h: float
    mean_actual_abs_h: float
    reliable_rate: float


def run_hmm_walk_forward_validation(
    close_panel: pd.DataFrame,
    *,
    eval_dates: pd.DatetimeIndex,
    horizon: int,
    cfg: HmmShortHorizonConfig | None = None,
) -> tuple[pd.DataFrame, HmmValidationSummary]:
    cfg = cfg or HmmShortHorizonConfig()
    min_samples = cfg.min_samples

    rows: list[dict[str, Any]] = []
    for code in close_panel.columns:
        series = pd.to_numeric(close_panel[code], errors="coerce").dropna()
        if series.size < min_samples + horizon + 5:
            continue

        for end in eval_dates:
            if end > series.index.max():
                continue
            ts_end = pd.Timestamp(end)
            metrics = compute_hmm_short_horizon_metrics(
                series,
                ts_end,
                horizon=horizon,
                cfg=cfg,
            )
            if metrics.barrier is None or metrics.p_up is None:
                continue

            fwd = forward_log_returns(series, ts_end, horizon=horizon)
            if fwd is None:
                continue
            outcome = realize_triple_barrier(fwd, barrier=float(metrics.barrier))
            cum_h = forward_cumulative_log_return(fwd)
            if cum_h is None:
                continue

            rows.append(
                {
                    "code": str(code),
                    "as_of": ts_end.strftime("%Y-%m-%d"),
                    "horizon": horizon,
                    "regime": metrics.regime_now,
                    "p_high": metrics.p_high,
                    "p_low": metrics.p_low,
                    "sigma_mix_1d": metrics.sigma_mix_1d,
                    "sigma_mix_h": metrics.sigma_mix_h,
                    "mu_mix_1d": metrics.mu_mix_1d,
                    "p_up": metrics.p_up,
                    "p_down": metrics.p_down,
                    "p_timeout": metrics.p_timeout,
                    "barrier": metrics.barrier,
                    "q05": metrics.q05,
                    "q50": metrics.q50,
                    "q95": metrics.q95,
                    "actual_outcome": outcome,
                    "actual_cum_h": cum_h,
                    "reliable": metrics.reliable,
                }
            )

    frame = pd.DataFrame(rows)
    empty = HmmValidationSummary(
        n_obs=0,
        n_symbols=0,
        eval_start="",
        eval_end="",
        horizon=horizon,
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
        mean_p_high=0.0,
        high_regime_rate=0.0,
        mean_abs_h_when_high=0.0,
        mean_abs_h_when_low=0.0,
        mean_sigma_mix_h=0.0,
        mean_actual_abs_h=0.0,
        reliable_rate=0.0,
    )
    if frame.empty:
        return frame, empty

    actual = frame["actual_outcome"].to_numpy()
    pred_up = frame["p_up"].astype(float).to_numpy()
    pred_down = frame["p_down"].astype(float).to_numpy()
    pred_timeout = frame["p_timeout"].astype(float).to_numpy()
    p_high = frame["p_high"].astype(float)
    high_mask = p_high >= 0.5
    low_mask = ~high_mask
    abs_h = frame["actual_cum_h"].abs()

    summary = HmmValidationSummary(
        n_obs=len(frame),
        n_symbols=int(frame["code"].nunique()),
        eval_start=str(frame["as_of"].min()),
        eval_end=str(frame["as_of"].max()),
        horizon=horizon,
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
        mean_p_high=float(p_high.mean()),
        high_regime_rate=float(high_mask.mean()),
        mean_abs_h_when_high=float(abs_h.loc[high_mask].mean()) if high_mask.any() else 0.0,
        mean_abs_h_when_low=float(abs_h.loc[low_mask].mean()) if low_mask.any() else 0.0,
        mean_sigma_mix_h=float(frame["sigma_mix_h"].astype(float).mean()),
        mean_actual_abs_h=float(abs_h.mean()),
        reliable_rate=float(frame["reliable"].astype(bool).mean()),
    )
    return frame, summary


def per_symbol_hmm_validation(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for code, grp in frame.groupby("code"):
        actual = grp["actual_outcome"].to_numpy()
        rows.append(
            {
                "code": code,
                "n_obs": len(grp),
                "actual_up_rate": float((actual == "up").mean()),
                "mean_pred_p_up": float(grp["p_up"].mean()),
                "mean_p_high": float(grp["p_high"].mean()),
                "brier": _multiclass_brier(
                    grp["p_up"].astype(float).to_numpy(),
                    grp["p_down"].astype(float).to_numpy(),
                    grp["p_timeout"].astype(float).to_numpy(),
                    actual,
                ),
                "coverage_q05": float((grp["actual_cum_h"] <= grp["q05"]).mean()),
                "coverage_q50": float((grp["actual_cum_h"] <= grp["q50"]).mean()),
                "coverage_q95": float((grp["actual_cum_h"] <= grp["q95"]).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("n_obs", ascending=False)


def format_hmm_validation_report(
    summary: HmmValidationSummary,
    *,
    pack_dir: str,
    as_of_board: str,
    per_symbol: pd.DataFrame,
    step_days: int = 1,
    garch_brier: float | None = None,
    mlfam_brier: float | None = None,
) -> str:
    lines = [
        "# HMM Short-Horizon Board Walk-Forward Validation",
        "",
        f"**Board as_of:** {as_of_board}  ",
        f"**Pack dir:** `{pack_dir}`  ",
        f"**Horizon H:** {summary.horizon} trading days  ",
        f"**Eval window:** {summary.eval_start} → {summary.eval_end} (step={step_days}d)  ",
        f"**Observations:** {summary.n_obs} ({summary.n_symbols} symbols)  ",
        f"**Reliable rate:** {summary.reliable_rate:.1%}  ",
        "",
        "## 1. Triple-Barrier Calibration (HMM mixture MC · §34.5.2)",
        "",
        "| 事件 | 预测均值 | 实际频率 |",
        "|------|----------|----------|",
        f"| 触轨↑ | {summary.mean_pred_p_up:.1%} | {summary.actual_up_rate:.1%} |",
        f"| 触轨↓ | {summary.mean_pred_p_down:.1%} | {summary.actual_down_rate:.1%} |",
        f"| 到期 | {summary.mean_pred_p_timeout:.1%} | {summary.actual_timeout_rate:.1%} |",
        f"| **Brier score** | — | **{summary.brier_score:.4f}** |",
        "",
    ]
    if garch_brier is not None or mlfam_brier is not None:
        lines.append("> 对照：")
        if garch_brier is not None:
            delta = summary.brier_score - garch_brier
            better = "HMM 更好" if delta < 0 else "GARCH 更好" if delta > 0 else "相当"
            lines.append(f"> - GARCH Brier={garch_brier:.4f}；Δ={delta:+.4f}（{better}）")
        if mlfam_brier is not None:
            delta = summary.brier_score - mlfam_brier
            better = "HMM 更好" if delta < 0 else "MLFAM 更好" if delta > 0 else "相当"
            lines.append(f"> - MLFAM Brier={mlfam_brier:.4f}；Δ={delta:+.4f}（{better}）")
        lines.append("")

    lines.extend(
        [
            "> Brier 三分类概率评分；≈0.67 为均匀瞎猜，越低越好。",
            "",
            "## 2. H-Day Quantile Coverage (HMM mixture simulation)",
            "",
            "| 分位 | 目标 | 实际覆盖率 | 偏差 |",
            "|------|------|------------|------|",
            f"| q05 | 5% | {summary.coverage_q05:.1%} | {summary.coverage_q05 - 0.05:+.1%} |",
            f"| q50 | 50% | {summary.coverage_q50:.1%} | {summary.coverage_q50 - 0.50:+.1%} |",
            f"| q95 | 95% | {summary.coverage_q95:.1%} | {summary.coverage_q95 - 0.95:+.1%} |",
            "",
            "## 3. Regime Separation (高波动滤波是否对应更大实现波动)",
            "",
            f"- 平均 p_high: **{summary.mean_p_high:.1%}**；判为高波动 (p_high≥50%) 占比: **{summary.high_regime_rate:.1%}**",
            f"- 高波动日的平均 |H 日累计收益|: **{summary.mean_abs_h_when_high:.2%}**",
            f"- 低波动日的平均 |H 日累计收益|: **{summary.mean_abs_h_when_low:.2%}**",
        ]
    )
    if summary.mean_abs_h_when_low > 0:
        ratio = summary.mean_abs_h_when_high / summary.mean_abs_h_when_low
        lines.append(f"- 高/低 |actual| 比值: **{ratio:.2f}**（>1 表示 regime 有分离力）")
    else:
        lines.append("- 高/低 |actual| 比值: —（低波动样本不足）")

    lines.extend(
        [
            "",
            "## 4. Volatility Scale Sanity",
            "",
            f"- 平均 σ_mix_H: **{summary.mean_sigma_mix_h:.2%}**",
            f"- 平均 |实际 H 日累计收益|: **{summary.mean_actual_abs_h:.2%}**",
        ]
    )
    if summary.mean_sigma_mix_h > 0:
        lines.append(
            f"- 比值 |actual|/σ_mix_H: **{summary.mean_actual_abs_h / summary.mean_sigma_mix_h:.2f}**"
        )
    else:
        lines.append("- 比值: —")

    lines.extend(
        [
            "",
            "## 5. Per-Symbol (top 15 by n_obs)",
            "",
        ]
    )
    if not per_symbol.empty:
        lines.append("| code | n | actual↑ | pred↑ | mean p_high | Brier | cov q50 |")
        lines.append("|------|---|---------|-------|-------------|-------|---------|")
        for _, row in per_symbol.head(15).iterrows():
            lines.append(
                f"| {row['code']} | {int(row['n_obs'])} | "
                f"{row['actual_up_rate']:.1%} | {row['mean_pred_p_up']:.1%} | "
                f"{row['mean_p_high']:.1%} | {row['brier']:.3f} | {row['coverage_q50']:.1%} |"
            )
    else:
        lines.append("_No per-symbol data._")

    lines.extend(
        [
            "",
            "## 解读",
            "",
            "- Walk-forward：每个 as_of 仅用该日及之前数据拟合 2-state HMM，再生成混合 MC。",
            "- 三重门 Brier / 分位覆盖衡量**混合预测密度**校准；regime 分离衡量**高/低波动状态**是否有用。",
            "- 与 GARCH 对照时：两者障宽定义不同（σ√H vs √Σσ²），Brier 可比方向，不宜逐格等同。",
            "- 本验证**不是**交易 PnL 回测；不替代肥尾 leg / Kelly 定仓。",
            "",
            "---",
            "",
            "*由 `src/etf_daily/scripts/validate_hmm_short_horizon_board.py` 生成。*",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "HmmValidationSummary",
    "format_hmm_validation_report",
    "per_symbol_hmm_validation",
    "run_hmm_walk_forward_validation",
]
