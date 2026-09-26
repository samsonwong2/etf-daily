#!/usr/bin/env python3
"""Generate GARCH conditional-volatility short-horizon board CSV from decision_pack artifacts.

By default emits H=5, H=10, and H=20 boards in one run.

Example:
  cd ~/gitee/skfolio_csi300/etf_strategy_clean
  PY=~/etf-daily-output/python/envs/py312/bin/python
  $PY src/etf_daily/scripts/generate_garch_short_horizon_board.py \\
    --pack-dir ~/temp/decision_packs/20260708

Outputs (under pack-dir):
  - cluster_mapping_selected_garch_short_horizon_h5.csv
  - cluster_mapping_selected_garch_short_horizon_h10.csv
  - cluster_mapping_selected_garch_short_horizon_h20.csv
  - GARCH_SHORT_HORIZON_REPORT_h{H}.md
  - garch_price_targets.csv (with --with-price-targets)
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
from etf_daily.lib.fat_tail_risk import FatTailConfig, fat_tail_config_from_raw, slice_log_returns
from etf_daily.lib.garch_dynamics import GarchFitResult, fit_garch_dynamics
from etf_daily.lib.garch_short_horizon_board import (
    GARCH_SHORT_HORIZON_COLUMNS,
    GarchShortHorizonConfig,
    GarchShortHorizonMetrics,
    compute_garch_short_horizon_metrics,
    format_garch_row,
    garch_short_horizon_config_from_raw,
)
from etf_daily.lib.pack_live_prices import (
    LiveCloseInjectionStats,
    apply_pack_live_closes,
    load_live_close_from_pack,
)
from etf_daily.lib.price_jump_flags import (
    build_jump_diagnostics_frame,
    format_jump_diagnostics_report_section,
    jump_flag_config_from_raw,
)
from etf_daily.lib.garch_price_targets import (
    GARCH_PRICE_TARGET_CSV,
    build_garch_price_targets_frame,
)

DEFAULT_HORIZONS: tuple[int, ...] = (5, 10, 20)
GARCH_JUMP_DIAGNOSTICS_CSV = "garch_jump_diagnostics.csv"


def garch_short_horizon_csv_path(pack_dir: Path, horizon: int) -> Path:
    return pack_dir / f"cluster_mapping_selected_garch_short_horizon_h{horizon}.csv"


def garch_short_horizon_report_path(pack_dir: Path, horizon: int) -> Path:
    return pack_dir / f"GARCH_SHORT_HORIZON_REPORT_h{horizon}.md"


def _num(value: float | None, *, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _prob(value: float | None, *, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def _pval(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    p = float(value)
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


def _tail_short(regime: Any) -> str:
    return _TAIL_REGIME_SHORT.get(str(regime or "stable"), "稳定")


def _state_short(state: Any) -> str:
    return _PATH_STATE_SHORT.get(str(state or ""), "—")


def _sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    is_holding = 0 if row.get("_is_holding") else 1
    unreliable = 1 if row.get("_unreliable") else 0
    sigma_h = row.get("_sigma_h") or 0.0
    return (is_holding, unreliable, -sigma_h, str(row.get("code", "")))


def _flag_tuple(flags: Any) -> tuple[str, ...]:
    if isinstance(flags, str):
        return tuple(f for f in flags.split("|") if f)
    if isinstance(flags, (list, tuple)):
        return tuple(str(f) for f in flags)
    return ()


def _fit_cache_for_panel(
    cluster_frame: pd.DataFrame,
    close_panel: pd.DataFrame,
    as_of: str,
    *,
    board_cfg: GarchShortHorizonConfig,
    max_horizon: int,
) -> dict[str, GarchFitResult]:
    """Fit each symbol once at max_horizon; reused across H=5/10/20."""
    end = pd.Timestamp(as_of)
    gcfg = board_cfg.garch
    cache: dict[str, GarchFitResult] = {}
    for _, row in cluster_frame.iterrows():
        code = str(row["code"])
        if code not in close_panel.columns:
            continue
        series = pd.to_numeric(close_panel[code], errors="coerce").dropna()
        returns = slice_log_returns(series, end, lookback_days=gcfg.train_window)
        if returns.size < 20:
            continue
        cache[code] = fit_garch_dynamics(returns, gcfg, max_horizon=max_horizon)
    return cache


def build_garch_short_horizon_csv(
    cluster_frame: pd.DataFrame,
    close_panel: pd.DataFrame,
    as_of: str,
    *,
    board_cfg: GarchShortHorizonConfig,
    horizon: int,
    fit_cache: dict[str, GarchFitResult] | None = None,
    max_horizon: int | None = None,
) -> pd.DataFrame:
    if cluster_frame.empty:
        return pd.DataFrame()

    end = pd.Timestamp(as_of)
    formatters = {
        "pct": _pct,
        "prob": _prob,
        "num": _num,
        "pval": _pval,
        "flags_str": _flags_str,
        "tail_short": _tail_short,
        "state_short": _state_short,
    }
    rows: list[dict[str, Any]] = []
    fit_cache = fit_cache or {}

    for _, row in cluster_frame.iterrows():
        code = str(row["code"])
        weight = float(row["weight"])
        flag_tuple = _flag_tuple(row.get("flags"))

        if code in close_panel.columns:
            series = pd.to_numeric(close_panel[code], errors="coerce").dropna()
            metrics = compute_garch_short_horizon_metrics(
                series,
                end,
                horizon=horizon,
                cfg=board_cfg,
                flags=flag_tuple,
                fit=fit_cache.get(code),
                max_horizon=max_horizon,
            )
        else:
            metrics = GarchShortHorizonMetrics(
                horizon=horizon,
                sigma_garch_1d=None,
                sigma_garch_h=None,
                vol_ann_5d=None,
                vol_ann_20d=None,
                vol_ratio_5_20=None,
                nu_t=None,
                arch_pvalue=None,
                p_up=None,
                p_down=None,
                p_timeout=None,
                barrier=None,
                q05=None,
                q50=None,
                q95=None,
                var95_cond=None,
                cvar95_cond=None,
                dgp_basis="none",
                reliable=False,
                basis="缺价格·仅展示",
            )

        out = format_garch_row(
            code=code,
            name=str(row["name"]),
            weight=weight,
            cluster_row=row,
            metrics=metrics,
            formatters=formatters,
        )
        out["_is_holding"] = weight > 0
        out["_unreliable"] = not metrics.reliable
        out["_sigma_h"] = metrics.sigma_garch_h or 0.0
        rows.append(out)

    frame = pd.DataFrame(rows)
    frame = frame.iloc[frame.apply(_sort_key, axis=1).argsort()].reset_index(drop=True)
    return frame.reindex(columns=list(GARCH_SHORT_HORIZON_COLUMNS))


def build_garch_short_horizon_report(
    as_of: str,
    pack_dir: Path,
    *,
    board_cfg: GarchShortHorizonConfig,
    horizon: int,
    n_symbols: int,
    live_stats: LiveCloseInjectionStats | None = None,
    jump_diagnostics: pd.DataFrame | None = None,
) -> str:
    gcfg = board_cfg.garch
    lines = [
        "# GARCH Conditional-Volatility Short-Horizon Board",
        "",
        f"**As of:** {as_of}  ",
        f"**数据来源:** `{pack_dir}`  ",
        f"**标的数:** {n_symbols}  ",
    ]
    if live_stats is not None:
        lines.append(
            f"**Pack live 价:** 取数 {live_stats.fetched} 只 · 注入 panel {live_stats.injected} 只 · "
            f"跳过同价 {live_stats.skipped_same} 只（`universe_live_close.csv` @ as_of）  "
        )
    else:
        lines.append("**Pack live 价:** —  ")
    lines.extend(
        [
            "**理论依据:** ftsnotes（Tsay）§18–21 条件波动率、§20.4/§24.1 多步波动与期限结构；"
            "三重门语义见 MLFAM §5.3",
            "",
            "## 方法学声明",
            "",
            "本板刻画**当前条件波动率 regime** 下 H 日收益分布，与 MLFAM 短周期板、"
            "`cluster_mapping_selected_pool_risk_*.csv` 长窗 EVT 肥尾板**并列互补**：",
            "",
            "- **GJR-GARCH(1,1)-t**（§21.2 + §20.5）：杠杆效应 + 厚尾创新；`mean=Zero` 简化；"
            "失败则 EWMA→RV20→MAD",
            "- **ARCH 效应检验**（§18.4）：对 demeaned 收益平方 a² 的 Ljung-Box p 值（非 Engle LM 全式）",
            "- **条件 VaR95/CVaR95**（§20.5 区间角色 + 实现扩展）：1 日单位方差 t 条件左尾，"
            "**非** EVT 静态 VaR；CVaR 为板内扩展",
            "- **三重门 MC**（MLFAM §5.3）：障宽 = barrier_mult × σ_garch_H；"
            "路径为递归 GARCH-t DGP（参数更新 σ），fallback 用平坦 σ 路径",
            "- **q05/q50/q95**：H 日累积对数收益分位数（条件分布摘要，非 μ 点估计）",
            "- **σ_garch_H**（§24.1）：√(Σσ²_step) 累积条件波动",
            "- **拆分 vs 真实跳跃**：旁路诊断（见下节 / `garch_jump_diagnostics.csv`）；"
            "**不改变本板主 CSV 列集**；拟合仍用 qfq 后序列（默认不剔除拆分日收益）",
            "",
            f"**Horizon H = {horizon}** 交易日；训练窗 = {gcfg.train_window} 日；",
            f"barrier_mult = {gcfg.barrier_mult}；MC 路径 = {gcfg.mc_samples}。",
            "",
            "## 与 risk_252 / MLFAM 的分工",
            "",
            "| 板 | 回答的问题 |",
            "|----|-----------|",
            "| risk_252 | 结构尾风险：leg、α、κ、EVT VaR99（能不能碰） |",
            "| **本板 (garch_h*)** | 当前 σ regime 下 H 日分布与三重门（怎么动） |",
            "| short_horizon_h* | MLFAM meta-label 下注信号（原脚本，未改动） |",
            "| hmm_short_horizon_h* | §34 机制转换混合预测密度（独立脚本） |",
            "| garch_price_targets.csv | H=1/5/10/20 止盈/止损/中位**价位**（见同目录长表） |",
            "",
            "## 列图例",
            "",
            "| 列 | 含义 |",
            "|----|------|",
            f"| H | 前瞻 horizon = {horizon} 交易日 |",
            "| σ_garch_1d | 下一日条件波动（对数收益，GARCH 预测） |",
            "| σ_garch_H | H 日累积条件波动 √(Σσ²_step)（§24.1） |",
            "| vol_ann_5d / vol_ann_20d | 对数收益实现波动率（与 GARCH 同口径） |",
            "| vol_ratio_5_20 | 5日/20日 vol；>1 表示近期波动加速 |",
            "| arch_p | a² Ljung-Box p 值（§18.4）；越小越显著 |",
            "| VaR95_cond / CVaR95_cond | 1 日单位方差 t 条件左尾（监控用） |",
            "| 触轨↑/↓/到期 | 三重门 MC 首触概率（和为 100%） |",
            f"| 障宽 | {gcfg.barrier_mult} × σ_garch_H |",
            "| DGP | 实际使用的数据生成过程（GJR-t/EWMA/RV20/MAD） |",
            "",
            "**排序**：持仓置顶 → 可靠估计优先 → σ_garch_H 降序。",
            "",
        ]
    )
    lines.append(format_jump_diagnostics_report_section(jump_diagnostics))
    lines.extend(
        [
            "---",
            "",
            "*由 `src/etf_daily/scripts/generate_garch_short_horizon_board.py` 自动生成。*",
            "",
        ]
    )
    return "\n".join(lines)



def build_garch_short_horizon_board(
    *,
    pack_dir: Path,
    cluster_mapping_path: Path | None = None,
    provider_uri: str | Path | None = None,
    fat_cfg: FatTailConfig | None = None,
    board_cfg: GarchShortHorizonConfig | None = None,
    horizons: tuple[int, ...] | list[int] = DEFAULT_HORIZONS,
) -> tuple[
    str,
    dict[int, pd.DataFrame],
    dict[int, str],
    LiveCloseInjectionStats,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, GarchFitResult],
    GarchShortHorizonConfig,
]:
    pack_cfg = load_decision_pack_config()
    fat_cfg = fat_cfg or fat_tail_config_from_raw(pack_cfg.raw.get("fat_tail"))
    board_cfg = board_cfg or garch_short_horizon_config_from_raw(
        pack_cfg.raw.get("garch_short_horizon")
    )
    garch_raw = pack_cfg.raw.get("garch_short_horizon") or {}
    jump_cfg = jump_flag_config_from_raw(
        garch_raw.get("jump") if isinstance(garch_raw, dict) else None
    )

    defensive_codes = frozenset(c for c in pack_cfg.defensive_codes)
    # Note: load_fat_tail_data also loads qlib internally for cluster flags;
    # we reload below to inject pack live closes (same pattern as MLFAM board).
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
    panel_raw = load_qlib_close(uri, universe, start, as_of)
    live_closes = load_live_close_from_pack(pack_dir)
    name_map = {
        str(row["code"]): str(row.get("name") or row["code"])
        for _, row in cluster_frame.iterrows()
    }
    # Sidecar diagnostics on raw closes (before qfq) so corp_split remains visible.
    jump_diagnostics = build_jump_diagnostics_frame(
        panel_raw,
        as_of=as_of,
        codes=[str(c) for c in cluster_frame["code"].tolist()],
        names=name_map,
        live_closes=live_closes,
        cfg=jump_cfg,
    )

    panel, _ = auto_qfq_adjust_close_panel(panel_raw)
    panel, live_stats = apply_pack_live_closes(panel, as_of, live_closes)

    horizon_list = tuple(int(h) for h in horizons)
    pt_horizons = board_cfg.price_target_horizons
    max_h = max(
        max(horizon_list) if horizon_list else max(DEFAULT_HORIZONS),
        max(pt_horizons) if pt_horizons else 20,
    )
    fit_cache = _fit_cache_for_panel(
        cluster_frame,
        panel,
        as_of,
        board_cfg=board_cfg,
        max_horizon=max_h,
    )

    csv_by_horizon: dict[int, pd.DataFrame] = {}
    report_by_horizon: dict[int, str] = {}
    for horizon in horizon_list:
        csv_frame = build_garch_short_horizon_csv(
            cluster_frame,
            panel,
            as_of,
            board_cfg=board_cfg,
            horizon=int(horizon),
            fit_cache=fit_cache,
            max_horizon=max_h,
        )
        markdown = build_garch_short_horizon_report(
            as_of,
            pack_dir,
            board_cfg=board_cfg,
            horizon=int(horizon),
            n_symbols=len(csv_frame),
            live_stats=live_stats,
            jump_diagnostics=jump_diagnostics,
        )
        csv_by_horizon[int(horizon)] = csv_frame
        report_by_horizon[int(horizon)] = markdown
    return (
        as_of,
        csv_by_horizon,
        report_by_horizon,
        live_stats,
        jump_diagnostics,
        cluster_frame,
        panel,
        fit_cache,
        board_cfg,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate GARCH conditional-volatility short-horizon board CSV",
    )
    parser.add_argument(
        "--pack-dir",
        type=Path,
        required=True,
        help="decision_pack generate 输出目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="单 horizon 模式下的 CSV 路径（与 --horizon 联用）",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help="单 horizon 模式下的 MD 路径（与 --horizon 联用）",
    )
    parser.add_argument(
        "--cluster-mapping",
        type=Path,
        default=None,
        help="cluster_mapping_selected.txt（默认 runtime_paths）",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="仅生成指定 H（省略则同时生成 h5 + h10 + h20）",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=None,
        help="自定义多个 H，如 --horizons 5 10 20（与 --horizon 互斥）",
    )
    parser.add_argument(
        "--provider-uri",
        type=Path,
        default=None,
        help="qlib provider URI override",
    )
    parser.add_argument(
        "--with-price-targets",
        action="store_true",
        help=f"同时写入 {GARCH_PRICE_TARGET_CSV}（H=1/5/10/20 价位目标长表）",
    )
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

    (
        as_of,
        csv_by_horizon,
        report_by_horizon,
        live_stats,
        jump_diagnostics,
        cluster_frame,
        panel,
        fit_cache,
        board_cfg,
    ) = build_garch_short_horizon_board(
        pack_dir=pack_dir,
        cluster_mapping_path=args.cluster_mapping,
        provider_uri=args.provider_uri,
        horizons=horizons,
    )

    diag_path = pack_dir / GARCH_JUMP_DIAGNOSTICS_CSV
    jump_diagnostics.to_csv(diag_path, index=False, encoding="utf-8-sig")
    print(f"[OK] jump diagnostics      -> {diag_path} ({len(jump_diagnostics)} rows)")

    for horizon in horizons:
        csv_path = (
            args.output.expanduser().resolve()
            if single_horizon and args.output
            else garch_short_horizon_csv_path(pack_dir, horizon)
        )
        report_path = (
            args.report_output.expanduser().resolve()
            if single_horizon and args.report_output
            else garch_short_horizon_report_path(pack_dir, horizon)
        )

        csv_frame = csv_by_horizon[horizon]
        markdown = report_by_horizon[horizon]
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
        report_path.write_text(markdown, encoding="utf-8")
        print(
            f"[OK] garch short-horizon h{horizon} -> {csv_path} "
            f"({len(csv_frame)} rows, as_of={as_of}, "
            f"live_prices={live_stats.fetched}, injected={live_stats.injected}, "
            f"skipped_same={live_stats.skipped_same})"
        )
        print(f"[OK] report h{horizon}           -> {report_path}")

    if args.with_price_targets:
        pt_horizons = board_cfg.price_target_horizons
        max_h = max(
            max(horizons) if horizons else 20,
            max(pt_horizons) if pt_horizons else 20,
        )
        pt_frame = build_garch_price_targets_frame(
            cluster_frame,
            panel,
            as_of,
            board_cfg=board_cfg,
            fit_cache=fit_cache,
            horizons=pt_horizons,
            max_horizon=max_h,
        )
        pt_path = pack_dir / GARCH_PRICE_TARGET_CSV
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        pt_frame.to_csv(pt_path, index=False, encoding="utf-8-sig")
        print(
            f"[OK] garch price targets -> {pt_path} "
            f"({len(pt_frame)} rows, as_of={as_of})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
