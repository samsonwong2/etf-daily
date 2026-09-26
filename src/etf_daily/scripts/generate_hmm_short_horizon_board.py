#!/usr/bin/env python3
"""Generate HMM regime-mixture short-horizon board (ftsnotes §34).

Independent of GARCH CSV schema. Default H=5, 10, 20.

Example:
  cd ~/gitee/skfolio_csi300/etf_strategy_clean
  PY=~/etf-daily-output/python/envs/py312/bin/python
  $PY src/etf_daily/scripts/generate_hmm_short_horizon_board.py \\
    --pack-dir ~/temp/decision_packs/20260709
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.scripts.generate_fat_tail_risk_report import (
    _PATH_STATE_SHORT,
    _TAIL_REGIME_SHORT,
    _flags_str,
    _pct,
    load_fat_tail_data,
)
from etf_daily.lib.config import load_decision_pack_config
from etf_daily.lib.fat_tail_risk import FatTailConfig, fat_tail_config_from_raw
from etf_daily.lib.hmm_short_horizon import (
    HMM_SHORT_HORIZON_COLUMNS,
    HmmFitResult,
    HmmShortHorizonConfig,
    HmmShortHorizonMetrics,
    compute_hmm_short_horizon_metrics,
    fit_hmm_2state,
    format_hmm_row,
    hmm_short_horizon_config_from_raw,
)
from etf_daily.lib.pack_live_prices import (
    apply_pack_live_closes,
    load_live_close_from_pack,
)
from etf_daily.lib.price_jump_flags import drop_split_log_returns

DEFAULT_HORIZONS: tuple[int, ...] = (5, 10, 20)


def hmm_short_horizon_csv_path(pack_dir: Path, horizon: int) -> Path:
    return pack_dir / f"cluster_mapping_selected_hmm_short_horizon_h{horizon}.csv"


def hmm_short_horizon_report_path(pack_dir: Path, horizon: int) -> Path:
    return pack_dir / f"HMM_SHORT_HORIZON_REPORT_h{horizon}.md"


def _num(value: float | None, *, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _prob(value: float | None, *, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def _tail_short(regime: Any) -> str:
    return _TAIL_REGIME_SHORT.get(str(regime or "stable"), "稳定")


def _state_short(state: Any) -> str:
    return _PATH_STATE_SHORT.get(str(state or ""), "—")


def _sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    is_holding = 0 if row.get("_is_holding") else 1
    unreliable = 1 if row.get("_unreliable") else 0
    p_high = row.get("_p_high") or 0.0
    return (is_holding, unreliable, -p_high, str(row.get("code", "")))


def _fit_cache_for_panel(
    cluster_frame: pd.DataFrame,
    close_panel: pd.DataFrame,
    as_of: str,
    *,
    board_cfg: HmmShortHorizonConfig,
) -> dict[str, HmmFitResult]:
    end = pd.Timestamp(as_of)
    cache: dict[str, HmmFitResult] = {}
    for _, row in cluster_frame.iterrows():
        code = str(row["code"])
        if code not in close_panel.columns:
            continue
        series = pd.to_numeric(close_panel[code], errors="coerce").dropna()
        if board_cfg.drop_split_returns:
            returns = drop_split_log_returns(
                series,
                end=end,
                lookback_days=board_cfg.train_window,
                split_threshold=board_cfg.split_threshold,
            )
        else:
            from etf_daily.lib.fat_tail_risk import slice_log_returns

            returns = slice_log_returns(series, end, lookback_days=board_cfg.train_window)
        if returns.size < 20:
            continue
        cache[code] = fit_hmm_2state(returns, board_cfg)
    return cache


def build_hmm_short_horizon_csv(
    cluster_frame: pd.DataFrame,
    close_panel: pd.DataFrame,
    as_of: str,
    *,
    board_cfg: HmmShortHorizonConfig,
    horizon: int,
    fit_cache: dict[str, HmmFitResult] | None = None,
) -> pd.DataFrame:
    if cluster_frame.empty:
        return pd.DataFrame()

    end = pd.Timestamp(as_of)
    formatters = {
        "pct": _pct,
        "prob": _prob,
        "num": _num,
        "flags_str": _flags_str,
        "tail_short": _tail_short,
        "state_short": _state_short,
    }
    rows: list[dict[str, Any]] = []
    fit_cache = fit_cache or {}

    for _, row in cluster_frame.iterrows():
        code = str(row["code"])
        weight = float(row["weight"])
        if code in close_panel.columns:
            series = pd.to_numeric(close_panel[code], errors="coerce").dropna()
            metrics = compute_hmm_short_horizon_metrics(
                series,
                end,
                horizon=horizon,
                cfg=board_cfg,
                fit=fit_cache.get(code),
            )
        else:
            metrics = HmmShortHorizonMetrics(
                horizon=horizon,
                p_high=None,
                p_low=None,
                regime_now="—",
                sigma_mix_1d=None,
                sigma_mix_h=None,
                mu_mix_1d=None,
                p_up=None,
                p_down=None,
                p_timeout=None,
                barrier=None,
                q05=None,
                q50=None,
                q95=None,
                reliable=False,
                basis="缺价格·仅展示",
                regime_separated=False,
            )

        out = format_hmm_row(
            code=code,
            name=str(row["name"]),
            weight=weight,
            cluster_row=row,
            metrics=metrics,
            formatters=formatters,
        )
        out["_is_holding"] = weight > 0
        out["_unreliable"] = (not metrics.reliable) or (not metrics.regime_separated)
        # Collapsed regimes: do not sort as if p_high were meaningful.
        out["_p_high"] = (
            (metrics.p_high or 0.0) if metrics.regime_separated else -1.0
        )
        rows.append(out)

    frame = pd.DataFrame(rows)
    frame = frame.iloc[frame.apply(_sort_key, axis=1).argsort()].reset_index(drop=True)
    return frame.reindex(columns=list(HMM_SHORT_HORIZON_COLUMNS))


def build_hmm_short_horizon_report(
    as_of: str,
    pack_dir: Path,
    *,
    board_cfg: HmmShortHorizonConfig,
    horizon: int,
    n_symbols: int,
    live_close_count: int = 0,
) -> str:
    lines = [
        "# HMM Regime-Mixture Short-Horizon Board",
        "",
        f"**As of:** {as_of}  ",
        f"**数据来源:** `{pack_dir}`  ",
        f"**标的数:** {n_symbols}  ",
        f"**Pack live 价注入:** {live_close_count} 只  ",
        "**理论依据:** ftsnotes §34 隐马氏模型 / 观测值预测分布（§34.5.2）；"
        "**非** §30–32 线性高斯 Kalman",
        "",
        "## 方法学声明",
        "",
        "本板用 **2-regime HMM**（低/高波动）给出 H 日 **混合预测密度** 摘要，",
        "与 GARCH 条件波动板、MLFAM 板**并列互补**：",
        "",
        f"- **发射分布:** {board_cfg.emission}（各 regime 内近似 i.i.d.）",
        "- **当前 regime:** 滤波概率 `p_high` / `p_low`；"
        f"若 σ_high/σ_low < {board_cfg.min_sigma_ratio:.2f} 则标 **未分离/塌缩**，"
        "`p_high` 显示为 —（不可解读为低波动）",
        "- **多步预测:** 转移矩阵演化 + 混合 MC 抽样（§34.5.2）",
        "- **三重门 / 分位数 MC:** "
        f"{'零均值' if board_cfg.mc_zero_mean else 'EM 均值'}；"
        f"{'当 vol_5d/vol_20d<1 时 σ 缩放至 max(' + str(board_cfg.mc_vol_scale_floor) + ', ratio)' if board_cfg.mc_recent_vol_scale else '无低波缩放'}；"
        f"障宽 = barrier_mult × σ_mix × scale × √H",
        "- **`μ_mix_1d`:** EM 展示量，不参与触轨 MC（与 GARCH Zero mean 对齐）",
        f"- **拆分日:** {'训练时剔除' if board_cfg.drop_split_returns else '保留'}（仅影响本板，不影响 GARCH CSV）",
        "",
        f"**Horizon H = {horizon}**；训练窗 = {board_cfg.train_window}；"
        f"MC = {board_cfg.mc_samples}；barrier_mult = {board_cfg.barrier_mult}。",
        "",
        "## 列图例",
        "",
        "| 列 | 含义 |",
        "|----|------|",
        "| regime | 高波动 / 低波动；两态 σ 分不开时为 **未分离** |",
        "| sep | `ok`=两态可分；`塌缩`=σ 几乎相同，勿读 p_high |",
        "| p_high / p_low | 滤波状态概率（塌缩时为 —） |",
        "| σ_mix_1d / σ_mix_H | 混合条件波动（1 日 / √H 缩放，**未**含低波 MC 缩放）；塌缩时仍可读 |",
        "| 触轨↑/↓/到期 | 混合 DGP 三重门首触概率（MC 零均值 + 可选低波缩放） |",
        "| q05/q50/q95 | H 日累积对数收益混合分位数（同上 MC 规则） |",
        "| μ_mix_1d | EM 漂移展示；**不**驱动触轨 MC |",
        "",
        "**排序**：持仓置顶 → 可靠且两态可分优先 → p_high 降序。",
        "",
        "---",
        "",
        "*由 `src/etf_daily/scripts/generate_hmm_short_horizon_board.py` 自动生成。*",
        "",
    ]
    return "\n".join(lines)


def build_hmm_short_horizon_board(
    *,
    pack_dir: Path,
    cluster_mapping_path: Path | None = None,
    provider_uri: str | Path | None = None,
    fat_cfg: FatTailConfig | None = None,
    board_cfg: HmmShortHorizonConfig | None = None,
    horizons: tuple[int, ...] | list[int] = DEFAULT_HORIZONS,
) -> tuple[str, dict[int, pd.DataFrame], dict[int, str], int]:
    pack_cfg = load_decision_pack_config()
    fat_cfg = fat_cfg or fat_tail_config_from_raw(pack_cfg.raw.get("fat_tail"))
    board_cfg = board_cfg or hmm_short_horizon_config_from_raw(
        pack_cfg.raw.get("hmm_short_horizon")
    )

    defensive_codes = frozenset(c for c in pack_cfg.defensive_codes)
    as_of, _, cluster_frame, _ = load_fat_tail_data(
        pack_dir,
        cluster_mapping_path=cluster_mapping_path,
        provider_uri=provider_uri,
        defensive_codes=defensive_codes,
        cfg=fat_cfg,
    )

    from etf_daily.scripts.generate_fat_tail_risk_report import (
        build_universe_codes,
        load_holdings_from_pack,
        warmup_start,
    )
    from etf_daily.lib.symbol_trend_board import load_cluster_mapping_codes
    from etf_daily.pipeline.price_adjustments import auto_qfq_adjust_close_panel
    from etf_daily.pipeline.strategy_layer_audit import load_qlib_close
    from etf_daily.paths import CLUSTER_MAPPING_SELECTED_TXT, QLIB_PROVIDER_URI

    holdings = load_holdings_from_pack(pack_dir)
    cluster_path = Path(cluster_mapping_path or CLUSTER_MAPPING_SELECTED_TXT)
    cluster_codes = load_cluster_mapping_codes(cluster_path) if cluster_path.exists() else ()
    universe = build_universe_codes(holdings, cluster_codes, defensive_codes)
    uri = str(provider_uri or QLIB_PROVIDER_URI)
    start = warmup_start(as_of, fat_cfg)
    panel = load_qlib_close(uri, universe, start, as_of)
    panel, _ = auto_qfq_adjust_close_panel(panel)
    live_closes = load_live_close_from_pack(pack_dir)
    panel, live_stats = apply_pack_live_closes(panel, as_of, live_closes)

    horizon_list = tuple(int(h) for h in horizons)
    fit_cache = _fit_cache_for_panel(
        cluster_frame, panel, as_of, board_cfg=board_cfg
    )

    csv_by_horizon: dict[int, pd.DataFrame] = {}
    report_by_horizon: dict[int, str] = {}
    for horizon in horizon_list:
        csv_frame = build_hmm_short_horizon_csv(
            cluster_frame,
            panel,
            as_of,
            board_cfg=board_cfg,
            horizon=int(horizon),
            fit_cache=fit_cache,
        )
        markdown = build_hmm_short_horizon_report(
            as_of,
            pack_dir,
            board_cfg=board_cfg,
            horizon=int(horizon),
            n_symbols=len(csv_frame),
            live_close_count=live_stats.injected,
        )
        csv_by_horizon[int(horizon)] = csv_frame
        report_by_horizon[int(horizon)] = markdown
    return as_of, csv_by_horizon, report_by_horizon, live_stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate HMM regime-mixture short-horizon board CSV",
    )
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--cluster-mapping", type=Path, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=None)
    parser.add_argument("--provider-uri", type=Path, default=None)
    return parser


def _resolve_horizons(args: argparse.Namespace) -> tuple[int, ...]:
    if args.horizon is not None and args.horizons:
        raise SystemExit("[ERR] --horizon 与 --horizons 不能同时使用")
    if args.horizon is not None:
        return (int(args.horizon),)
    if args.horizons:
        return tuple(int(h) for h in args.horizons)
    return DEFAULT_HORIZONS


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    pack_dir = args.pack_dir.expanduser().resolve()
    horizons = _resolve_horizons(args)
    single_horizon = len(horizons) == 1

    as_of, csv_by_horizon, report_by_horizon, live_stats = build_hmm_short_horizon_board(
        pack_dir=pack_dir,
        cluster_mapping_path=args.cluster_mapping,
        provider_uri=args.provider_uri,
        horizons=horizons,
    )

    for horizon in horizons:
        csv_path = (
            args.output.expanduser().resolve()
            if single_horizon and args.output
            else hmm_short_horizon_csv_path(pack_dir, horizon)
        )
        report_path = (
            args.report_output.expanduser().resolve()
            if single_horizon and args.report_output
            else hmm_short_horizon_report_path(pack_dir, horizon)
        )
        csv_frame = csv_by_horizon[horizon]
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
        report_path.write_text(report_by_horizon[horizon], encoding="utf-8")
        print(
            f"[OK] hmm short-horizon h{horizon} -> {csv_path} "
            f"({len(csv_frame)} rows, as_of={as_of}, "
            f"live_prices={live_stats.fetched}, injected={live_stats.injected}, "
            f"skipped_same={live_stats.skipped_same})"
        )
        print(f"[OK] report h{horizon}         -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
