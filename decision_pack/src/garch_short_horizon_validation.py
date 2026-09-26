"""Walk-forward validation for GARCH short-horizon board metrics."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from decision_pack.src.fat_tail_risk import ANNUALIZE
from decision_pack.src.garch_short_horizon_board import (
    GarchShortHorizonConfig,
    compute_garch_short_horizon_metrics,
)
from decision_pack.src.short_horizon_mlfam import (
    forward_cumulative_log_return,
    forward_log_returns,
    realize_triple_barrier,
)
from decision_pack.src.short_horizon_validation import (
    _multiclass_brier,
    build_eval_dates,
)


def _binary_brier(pred: np.ndarray, actual: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(actual)
    if not mask.any():
        return 0.0
    p = pred[mask].astype(float)
    y = actual[mask].astype(float)
    return float(np.mean((p - y) ** 2))


def forward_realized_vol_ann_log(
    close: pd.Series,
    end: pd.Timestamp,
    *,
    horizon: int,
) -> float | None:
    fwd = forward_log_returns(close, end, horizon=horizon)
    if fwd is None or fwd.size < 2:
        return None
    std = float(np.std(fwd, ddof=1))
    if not math.isfinite(std):
        return None
    return std * ANNUALIZE


@dataclass(frozen=True)
class GarchValidationSummary:
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
    var95_breach_rate: float
    cvar95_mean_exceedance: float
    mean_dgp_basis_gjr: float
    mean_sigma_garch_h: float
    mean_actual_abs_h: float
    spike_brier_score: float
    actual_spike_rate: float
    mean_pred_p_vol_spike: float
    spike_obs_count: int


def _next_day_log_return(close: pd.Series, end: pd.Timestamp) -> float | None:
    fwd = forward_log_returns(close, end, horizon=1)
    if fwd is None or fwd.size != 1:
        return None
    return float(fwd[0])


def run_garch_walk_forward_validation(
    close_panel: pd.DataFrame,
    *,
    eval_dates: pd.DatetimeIndex,
    horizon: int,
    cfg: GarchShortHorizonConfig | None = None,
    flags_by_code: dict[str, tuple[str, ...]] | None = None,
) -> tuple[pd.DataFrame, GarchValidationSummary]:
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
            metrics = compute_garch_short_horizon_metrics(
                series,
                ts_end,
                horizon=horizon,
                cfg=cfg,
                flags=flag_tuple,
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

            ret_1d = _next_day_log_return(series, ts_end)
            var_breach = np.nan
            cvar_exceed = np.nan
            if ret_1d is not None and metrics.var95_cond is not None:
                var_breach = float(ret_1d <= metrics.var95_cond)
                if metrics.cvar95_cond is not None and ret_1d <= metrics.var95_cond:
                    cvar_exceed = float(ret_1d - metrics.cvar95_cond)

            baseline_vol = metrics.vol_ann_5d
            fwd_vol = forward_realized_vol_ann_log(series, ts_end, horizon=horizon)
            actual_spike = np.nan
            if (
                baseline_vol is not None
                and fwd_vol is not None
                and math.isfinite(baseline_vol)
                and math.isfinite(fwd_vol)
                and baseline_vol > 0
            ):
                actual_spike = float(fwd_vol > baseline_vol * cfg.spike_mult)

            rows.append(
                {
                    "code": str(code),
                    "as_of": ts_end.strftime("%Y-%m-%d"),
                    "horizon": horizon,
                    "dgp_basis": metrics.dgp_basis,
                    "sigma_garch_1d": metrics.sigma_garch_1d,
                    "sigma_garch_h": metrics.sigma_garch_h,
                    "vol_ann_5d": metrics.vol_ann_5d,
                    "vol5_pct_120d": metrics.vol5_pct_120d,
                    "p_vol_spike_h5": metrics.p_vol_spike_h5,
                    "actual_vol_spike": actual_spike,
                    "forward_vol_ann": fwd_vol,
                    "p_up": metrics.p_up,
                    "p_down": metrics.p_down,
                    "p_timeout": metrics.p_timeout,
                    "barrier": metrics.barrier,
                    "q05": metrics.q05,
                    "q50": metrics.q50,
                    "q95": metrics.q95,
                    "var95_cond": metrics.var95_cond,
                    "cvar95_cond": metrics.cvar95_cond,
                    "actual_outcome": outcome,
                    "actual_cum_h": cum_h,
                    "actual_1d": ret_1d,
                    "var95_breach": var_breach,
                    "cvar95_exceed": cvar_exceed,
                    "reliable": metrics.reliable,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, GarchValidationSummary(
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
            var95_breach_rate=0.0,
            cvar95_mean_exceedance=0.0,
            mean_dgp_basis_gjr=0.0,
            mean_sigma_garch_h=0.0,
            mean_actual_abs_h=0.0,
            spike_brier_score=0.0,
            actual_spike_rate=0.0,
            mean_pred_p_vol_spike=0.0,
            spike_obs_count=0,
        )

    actual = frame["actual_outcome"].to_numpy()
    pred_up = frame["p_up"].astype(float).to_numpy()
    pred_down = frame["p_down"].astype(float).to_numpy()
    pred_timeout = frame["p_timeout"].astype(float).to_numpy()
    var_breaches = frame["var95_breach"].dropna()

    spike_mask = frame["actual_vol_spike"].notna() & frame["p_vol_spike_h5"].notna()
    spike_obs = int(spike_mask.sum())
    if spike_obs > 0:
        spike_brier = _binary_brier(
            frame.loc[spike_mask, "p_vol_spike_h5"].astype(float).to_numpy(),
            frame.loc[spike_mask, "actual_vol_spike"].astype(float).to_numpy(),
        )
        actual_spike_rate = float(frame.loc[spike_mask, "actual_vol_spike"].mean())
        mean_pred_spike = float(frame.loc[spike_mask, "p_vol_spike_h5"].mean())
    else:
        spike_brier = 0.0
        actual_spike_rate = 0.0
        mean_pred_spike = 0.0

    summary = GarchValidationSummary(
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
        var95_breach_rate=float(var_breaches.mean()) if len(var_breaches) else 0.0,
        cvar95_mean_exceedance=float(frame["cvar95_exceed"].dropna().mean())
        if frame["cvar95_exceed"].notna().any()
        else 0.0,
        mean_dgp_basis_gjr=float((frame["dgp_basis"] == "GJR-t").mean()),
        mean_sigma_garch_h=float(frame["sigma_garch_h"].astype(float).mean()),
        mean_actual_abs_h=float(frame["actual_cum_h"].abs().mean()),
        spike_brier_score=spike_brier,
        actual_spike_rate=actual_spike_rate,
        mean_pred_p_vol_spike=mean_pred_spike,
        spike_obs_count=spike_obs,
    )
    return frame, summary


def per_symbol_garch_validation(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for code, grp in frame.groupby("code"):
        actual = grp["actual_outcome"].to_numpy()
        spike_grp = grp.dropna(subset=["actual_vol_spike", "p_vol_spike_h5"])
        spike_brier_val = np.nan
        if not spike_grp.empty:
            spike_brier_val = _binary_brier(
                spike_grp["p_vol_spike_h5"].astype(float).to_numpy(),
                spike_grp["actual_vol_spike"].astype(float).to_numpy(),
            )
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
                "var95_breach_rate": float(grp["var95_breach"].dropna().mean())
                if grp["var95_breach"].notna().any()
                else np.nan,
                "spike_brier": spike_brier_val,
            }
        )
    return pd.DataFrame(rows).sort_values("n_obs", ascending=False)


def format_garch_validation_report(
    summary: GarchValidationSummary,
    *,
    pack_dir: str,
    as_of_board: str,
    per_symbol: pd.DataFrame,
    step_days: int = 1,
    mlfam_brier: float | None = None,
) -> str:
    lines = [
        "# GARCH Short-Horizon Board Walk-Forward Validation",
        "",
        f"**Board as_of:** {as_of_board}  ",
        f"**Pack dir:** `{pack_dir}`  ",
        f"**Horizon H:** {summary.horizon} trading days  ",
        f"**Eval window:** {summary.eval_start} → {summary.eval_end} (step={step_days}d)  ",
        f"**Observations:** {summary.n_obs} ({summary.n_symbols} symbols)  ",
        f"**GJR-t fit rate:** {summary.mean_dgp_basis_gjr:.1%}  ",
        "",
        "## 1. Triple-Barrier Calibration (GARCH-DGP MC)",
        "",
        "| 事件 | 预测均值 | 实际频率 |",
        "|------|----------|----------|",
        f"| 触轨↑ | {summary.mean_pred_p_up:.1%} | {summary.actual_up_rate:.1%} |",
        f"| 触轨↓ | {summary.mean_pred_p_down:.1%} | {summary.actual_down_rate:.1%} |",
        f"| 到期 | {summary.mean_pred_p_timeout:.1%} | {summary.actual_timeout_rate:.1%} |",
        f"| **Brier score** | — | **{summary.brier_score:.4f}** |",
        "",
    ]
    if mlfam_brier is not None:
        delta = summary.brier_score - mlfam_brier
        better = "GARCH 更好" if delta < 0 else "MLFAM 更好" if delta > 0 else "相当"
        lines.extend(
            [
                f"> 对照 MLFAM(IID) Brier={mlfam_brier:.4f}；Δ={delta:+.4f}（{better}）",
                "",
            ]
        )
    lines.extend(
        [
            "> Brier 三分类概率评分；0.67≈均匀猜，越低越好。",
            "",
            "## 2. H-Day Quantile Coverage (GARCH-t simulation)",
            "",
            "| 分位 | 目标 | 实际覆盖率 | 偏差 |",
            "|------|------|------------|------|",
            f"| q05 | 5% | {summary.coverage_q05:.1%} | {summary.coverage_q05 - 0.05:+.1%} |",
            f"| q50 | 50% | {summary.coverage_q50:.1%} | {summary.coverage_q50 - 0.50:+.1%} |",
            f"| q95 | 95% | {summary.coverage_q95:.1%} | {summary.coverage_q95 - 0.95:+.1%} |",
            "",
            "## 3. 1-Day Conditional VaR95 (GARCH, §24.5)",
            "",
            f"- 次日收益 ≤ VaR95_cond 的频率: **{summary.var95_breach_rate:.1%}**（目标 ≈5%）",
            f"-  breach 样本平均超出 CVaR95 的深度: **{summary.cvar95_mean_exceedance:.2%}**",
            "",
            "## 4. Volatility Scale Sanity",
            "",
            f"- 平均 σ_garch_H: **{summary.mean_sigma_garch_h:.2%}**",
            f"- 平均 |实际 H 日累计收益|: **{summary.mean_actual_abs_h:.2%}**",
            f"- 比值 |actual|/σ_H: **{summary.mean_actual_abs_h / summary.mean_sigma_garch_h:.2f}**"
            if summary.mean_sigma_garch_h > 0
            else "- 比值: —",
            "",
            "## 5. Vol Spike Probability (GARCH MC)",
            "",
            f"- 标签：未来 H 日 realized vol > vol_ann_5d(T) × spike_mult  ",
            f"- 观测数: **{summary.spike_obs_count}**  ",
            f"| 预测均值 | 实际频率 | Brier |",
            f"|----------|----------|-------|",
            f"| {summary.mean_pred_p_vol_spike:.1%} | {summary.actual_spike_rate:.1%} | **{summary.spike_brier_score:.4f}** |",
            "",
            "> 二分类 Brier；0.25≈均匀猜 50%，越低越好。",
            "",
            "## 6. Per-Symbol (top 15 by n_obs)",
            "",
        ]
    )
    if not per_symbol.empty:
        lines.append("| code | n | actual↑ | pred↑ | Brier | cov q50 | VaR95 breach |")
        lines.append("|------|---|---------|-------|-------|---------|--------------|")
        for _, row in per_symbol.head(15).iterrows():
            vbr = row["var95_breach_rate"]
            vbr_s = f"{vbr:.1%}" if pd.notna(vbr) else "—"
            lines.append(
                f"| {row['code']} | {int(row['n_obs'])} | "
                f"{row['actual_up_rate']:.1%} | {row['mean_pred_p_up']:.1%} | "
                f"{row['brier']:.3f} | {row['coverage_q50']:.1%} | {vbr_s} |"
            )
    else:
        lines.append("_No per-symbol data._")

    lines.extend(
        [
            "",
            "## 解读",
            "",
            "- Walk-forward：每个 as_of 仅用该日之前数据拟合 GARCH 并生成 MC 概率/分位数。",
            "- 分位覆盖率偏离 5/50/95 表示条件分布偏宽或偏窄；Brier 衡量三重门概率校准。",
            "- VaR95_cond 是 1 日 GARCH 条件左尾，与 risk_252 的 EVT 静态 VaR 不同口径。",
            "",
            "---",
            "",
            "*由 `decision_pack/scripts/validate_garch_short_horizon_board.py` 生成。*",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "GarchValidationSummary",
    "format_garch_validation_report",
    "per_symbol_garch_validation",
    "run_garch_walk_forward_validation",
]
