#!/usr/bin/env python3
"""Generate MLFAM short-horizon board CSV from decision_pack artifacts.

By default emits **both** H=5 and H=10 boards in one run.

Example:
  cd ~/gitee/skfolio_csi300/etf_strategy_clean
  PY=~/etf-daily-output/python/envs/py312/bin/python
  $PY decision_pack/scripts/generate_short_horizon_board.py \\
    --pack-dir ~/temp/decision_packs/20260623

Outputs (under pack-dir):
  - cluster_mapping_selected_short_horizon_h5.csv
  - cluster_mapping_selected_short_horizon_h10.csv
  - SHORT_HORIZON_MLFAM_REPORT_h5.md
  - SHORT_HORIZON_MLFAM_REPORT_h10.md

Price panel: qlib daily close through as_of, then ``universe_live_close.csv``
(from pack generate: portfolio live > AkShare spot > qlib) is appended/overwritten
on the as_of row for cluster universe symbols.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from decision_pack.scripts.generate_fat_tail_risk_report import (
    _TAIL_REGIME_SHORT,
    _PATH_STATE_SHORT,
    _pct,
    _flags_str,
    load_fat_tail_data,
)
from decision_pack.src.config import load_decision_pack_config
from decision_pack.src.fat_tail_risk import FatTailConfig, fat_tail_config_from_raw
from decision_pack.src.indicators import window_return
from decision_pack.src.pack_live_prices import (
    apply_pack_live_closes,
    load_live_close_from_pack,
)
from decision_pack.src.short_horizon_mlfam import (
    ShortHorizonConfig,
    ShortHorizonMetrics,
    compute_short_horizon_metrics,
    short_horizon_config_from_raw,
)


DEFAULT_HORIZONS: tuple[int, ...] = (5, 10)

SHORT_HORIZON_COLUMNS: tuple[str, ...] = (
    "code",
    "name",
    "持仓",
    "wt",
    "n",
    "H",
    "行动",
    "信号",
    "方向",
    "建议下注",
    "不下注原因",
    "期望EV%",
    "盈亏比",
    "置信",
    "趋势标签",
    "趋势t值",
    "趋势p值",
    "现价",
    "ret5d",
    "ret10d",
    "ret20d",
    "上轨价",
    "下轨价",
    "目标价",
    "止损参考价",
    "触轨↑",
    "触轨↓",
    "到期",
    "q50_H",
    "障宽",
    "leg",
    "尾结构",
    "状态",
    "回撤长",
    "flags",
    "可靠性",
    "依据",
)


def short_horizon_csv_path(pack_dir: Path, horizon: int) -> Path:
    return pack_dir / f"cluster_mapping_selected_short_horizon_h{horizon}.csv"


def short_horizon_report_path(pack_dir: Path, horizon: int) -> Path:
    return pack_dir / f"SHORT_HORIZON_MLFAM_REPORT_h{horizon}.md"


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


def _trend_label_zh(label: str) -> str:
    return {"up": "上升", "down": "下降", "flat": "无趋势"}.get(label, label)


def _direction_zh(label: str) -> str:
    return {
        "up": "多",
        "down": "空",
        "flat": "观望",
        "不下注": "不下注",
    }.get(label, label)


def _price(value: float | None, *, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _ret_pct(value: float | None, *, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def _ev_pct(value: float | None, *, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def _ratio(value: float | None, *, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _price_from_logret(close: float | None, logret: float | None) -> float | None:
    if close is None or logret is None or pd.isna(close) or pd.isna(logret):
        return None
    price = float(close)
    if price <= 0 or not math.isfinite(price):
        return None
    return price * math.exp(float(logret))


def _triple_barrier_ev(
    p_up: float | None,
    p_down: float | None,
    barrier: float | None,
) -> float | None:
    if p_up is None or p_down is None or barrier is None:
        return None
    return (float(p_up) - float(p_down)) * float(barrier)


def _risk_reward(p_up: float | None, p_down: float | None) -> float | None:
    if p_up is None or p_down is None:
        return None
    down = float(p_down)
    if down <= 0:
        return None
    return float(p_up) / down


def _reliability_tag(basis: str) -> str:
    if "α不可靠" in basis:
        return "α不可靠"
    if "低样本" in basis:
        return "低样本"
    if basis in ("缺价格·仅展示", "数据不足·仅展示"):
        return "—"
    return "正常"


def _signal_tier_core(
    metrics: ShortHorizonMetrics,
    *,
    sh_cfg: ShortHorizonConfig,
) -> tuple[str, str]:
    direction = str(metrics.direction)
    if direction in ("up", "down") and metrics.bet_size is not None and metrics.bet_size > 0:
        return "下注", ""

    if metrics.basis in ("缺价格·仅展示", "数据不足·仅展示"):
        return "回避", "缺价格" if "缺价格" in metrics.basis else "数据不足"

    if metrics.trend_label == "flat":
        return "回避", "趋势flat"

    tval = metrics.trend_t
    if tval is not None and abs(float(tval)) >= sh_cfg.t_thresh:
        if direction == "不下注":
            conf = metrics.confidence
            if conf is not None and conf < sh_cfg.conf_min:
                pct = float(conf) * 100.0
                return "观察", f"置信{pct:.1f}%<{sh_cfg.conf_min * 100:.0f}%"
            return "观察", "趋势与MC分歧"
        return "观察", "置信不足"

    return "回避", "趋势flat"


def _signal_tier(
    metrics: ShortHorizonMetrics,
    *,
    sh_cfg: ShortHorizonConfig,
) -> tuple[str, str]:
    caution = _reliability_tag(metrics.basis)
    if caution in ("α不可靠", "低样本"):
        signal, reason = _signal_tier_core(metrics, sh_cfg=sh_cfg)
        if signal == "下注":
            return "观察", caution
        if signal == "观察":
            return "观察", caution if not reason else f"{reason}·{caution}"
        return "观察", caution
    return _signal_tier_core(metrics, sh_cfg=sh_cfg)


def _action_label(
    metrics: ShortHorizonMetrics,
    *,
    is_holding: bool,
    path_state: str,
    signal: str,
) -> str:
    direction = str(metrics.direction)
    trend = metrics.trend_label
    broken = path_state == "破位"

    if is_holding:
        if direction == "down" or trend == "down" or broken:
            return "减/防守"
        if direction == "up" and signal == "下注":
            return "持有可加"
        return "持有观察"

    if signal == "回避" or direction == "down" or trend == "down":
        return "回避"
    if direction == "up" and signal == "下注":
        return "关注买入"
    return "观察"


def _action_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    is_holding = 0 if row.get("_is_holding") else 1
    signal_order = {"下注": 0, "观察": 1, "回避": 2}.get(str(row.get("_signal", "回避")), 2)
    ev_abs = float(row.get("_ev_abs") or 0.0)
    unreliable = 1 if row.get("_unreliable") else 0
    return (is_holding, signal_order, -ev_abs, unreliable, str(row.get("code", "")))


def _latest_close(series: pd.Series, end: pd.Timestamp) -> float | None:
    available = series.loc[series.index <= end].dropna()
    if available.empty:
        return None
    value = float(available.iloc[-1])
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def build_short_horizon_csv(
    cluster_frame: pd.DataFrame,
    close_panel: pd.DataFrame,
    as_of: str,
    *,
    sh_cfg: ShortHorizonConfig,
) -> pd.DataFrame:
    """Build actionable short-horizon board CSV for cluster universe."""
    if cluster_frame.empty:
        return pd.DataFrame()

    end = pd.Timestamp(as_of)
    rows: list[dict[str, Any]] = []

    for _, row in cluster_frame.iterrows():
        code = str(row["code"])
        weight = float(row["weight"])
        is_holding = weight > 0
        flags = row.get("flags")
        flag_tuple: tuple[str, ...]
        if isinstance(flags, str):
            flag_tuple = tuple(f for f in flags.split("|") if f)
        elif isinstance(flags, (list, tuple)):
            flag_tuple = tuple(str(f) for f in flags)
        else:
            flag_tuple = ()

        close_price: float | None = None
        if code in close_panel.columns:
            series = pd.to_numeric(close_panel[code], errors="coerce").dropna()
            close_price = _latest_close(series, end)
            metrics = compute_short_horizon_metrics(
                series,
                end,
                cfg=sh_cfg,
                flags=flag_tuple,
            )
            ret5 = window_return(series, end, 5)
            ret10 = window_return(series, end, 10)
            ret20 = window_return(series, end, 20)
        else:
            metrics = ShortHorizonMetrics(
                trend_label="flat",
                trend_t=None,
                trend_p=None,
                trend_win=0,
                p_up=None,
                p_down=None,
                p_timeout=None,
                barrier=None,
                direction="不下注",
                confidence=None,
                bet_size=0.0,
                q05=None,
                q50=None,
                q95=None,
                reliable=False,
                basis="缺价格·仅展示",
            )
            ret5 = ret10 = ret20 = float("nan")

        tail_short = _TAIL_REGIME_SHORT.get(str(row.get("tail_regime") or "stable"), "稳定")
        state_short = _PATH_STATE_SHORT.get(str(row.get("path_state") or ""), "—")
        signal, no_bet_reason = _signal_tier(metrics, sh_cfg=sh_cfg)
        action = _action_label(
            metrics,
            is_holding=is_holding,
            path_state=state_short,
            signal=signal,
        )
        ev = _triple_barrier_ev(metrics.p_up, metrics.p_down, metrics.barrier)
        rr = _risk_reward(metrics.p_up, metrics.p_down)
        barrier = metrics.barrier

        upper = _price_from_logret(close_price, barrier)
        lower = _price_from_logret(close_price, -barrier if barrier is not None else None)
        target = _price_from_logret(close_price, metrics.q95)
        stop_ref = _price_from_logret(close_price, metrics.q05)

        unreliable = _reliability_tag(metrics.basis) in ("α不可靠", "低样本", "—")

        rows.append(
            {
                "code": code,
                "name": str(row["name"]),
                "持仓": "持" if is_holding else "候选",
                "wt": f"{weight:.4f}",
                "n": str(int(row["n_samples"])),
                "H": str(sh_cfg.horizon),
                "行动": action,
                "信号": signal,
                "方向": _direction_zh(str(metrics.direction)),
                "建议下注": _pct(metrics.bet_size) if metrics.bet_size is not None else "—",
                "不下注原因": no_bet_reason or "—",
                "期望EV%": _ev_pct(ev),
                "盈亏比": _ratio(rr),
                "置信": _prob(metrics.confidence),
                "趋势标签": _trend_label_zh(metrics.trend_label),
                "趋势t值": _num(metrics.trend_t, digits=2),
                "趋势p值": _pval(metrics.trend_p),
                "现价": _price(close_price),
                "ret5d": _ret_pct(ret5),
                "ret10d": _ret_pct(ret10),
                "ret20d": _ret_pct(ret20),
                "上轨价": _price(upper),
                "下轨价": _price(lower),
                "目标价": _price(target),
                "止损参考价": _price(stop_ref),
                "触轨↑": _prob(metrics.p_up),
                "触轨↓": _prob(metrics.p_down),
                "到期": _prob(metrics.p_timeout),
                "q50_H": _pct(metrics.q50),
                "障宽": _pct(barrier) if barrier is not None else "—",
                "leg": str(row.get("barbell_leg") or "—"),
                "flags": _flags_str(row.get("flags")),
                "尾结构": tail_short,
                "状态": state_short,
                "回撤长": _pct(row.get("dd_long"), digits=1),
                "可靠性": _reliability_tag(metrics.basis),
                "依据": metrics.basis,
                "_is_holding": is_holding,
                "_signal": signal,
                "_ev_abs": abs(ev) if ev is not None else 0.0,
                "_unreliable": unreliable,
            }
        )

    frame = pd.DataFrame(rows)
    frame = frame.iloc[
        frame.apply(_action_sort_key, axis=1).argsort()
    ].reset_index(drop=True)
    return frame.reindex(columns=list(SHORT_HORIZON_COLUMNS))


def build_short_horizon_report(
    as_of: str,
    pack_dir: Path,
    *,
    sh_cfg: ShortHorizonConfig,
    n_symbols: int,
    live_close_count: int = 0,
) -> str:
    win_lo, win_hi = sh_cfg.trend_win_range()
    trend_win_desc = (
        f"{win_lo}–{win_hi} 日窗内 |t(slope)| 最大的趋势扫描结果"
        if sh_cfg.trend_win_from_horizon
        else f"{win_lo}–{win_hi} 日窗内 |t(slope)| 最大的趋势扫描结果（固定窗口）"
    )
    lines = [
        "# MLFAM Short-Horizon Board",
        "",
        f"**As of:** {as_of}  ",
        f"**数据来源:** `{pack_dir}`  ",
        f"**标的数:** {n_symbols}  ",
        f"**Pack live 价注入:** {live_close_count} 只（`universe_live_close.csv` @ as_of）  ",
        "**框架:** [mlfam-ml-mean-risk-framework.md](~/etf-daily-output/books/MLFAM/mlfam-ml-mean-risk-framework.md) §13",
        "",
        "## 方法学声明（§13）",
        "",
        "本书**反对**把 5–10 个交易日的短期收益均值 / 分布当作可重仓信用的点预测。"
        "本板改用：",
        "",
        "- **趋势扫描**（§5）：方向标签 + t 值，不估收益大小",
        "- **三重门 Monte-Carlo**（§5 + 附录 A DGP）：触上轨 / 触下轨 / 到期概率",
        "- **meta-labeling**（§5.5）：方向与下注大小解耦；置信不足输出「不下注」",
        "- **DGP 分位数** q05/q50/q95：H 日累计对数收益分布摘要，**不是** μ 点估计",
        "- **结构突变代理**（§7）：复用 `尾结构` / `状态` / `回撤长` 列",
        "",
        "## 红线（§11）",
        "",
        "- 任何策略评估须报告试验次数 N 并使用通缩 Sharpe（DSR）",
        "- 单次历史回测只能证伪，不能证真；本板为**决策辅助展示**，非自动下单",
        "",
        "## 列图例",
        "",
        "| 列 | 含义 |",
        "|----|------|",
        f"| H | 三重门 / MC 前瞻 horizon = {sh_cfg.horizon} 交易日 |",
        "| 持仓 | 持=当前持仓(wt>0)；候选=待选池 |",
        "| 行动 | 可执行建议：关注买入/持有可加/持有观察/减/防守/观察/回避 |",
        "| 信号 | 三档：下注 / 观察 / 回避 |",
        "| 可靠性 | 正常 / **α不可靠** / **低样本**（仍展示 MC 结果，信号最高「观察」，不下正式注） |",
        "| 依据 | 计算来源；带 `·α不可靠` / `·低样本` 后缀表示尾部分布或样本量需谨慎 |",
        "| 不下注原因 | 信号非「下注」时的主因（置信不足、α不可靠、低样本等） |",
        "| 期望EV% | (触轨↑−触轨↓)×障宽；>0 偏多，量纲为 H 日对数收益% |",
        "| 盈亏比 | 触轨↑ / 触轨↓；>1 偏多 |",
        "| 现价 / ret5d / ret10d / ret20d | 盘中/最新价与真实简单涨幅（**非**趋势 t 值） |",
        "| 上轨价 / 下轨价 | 现价×exp(±障宽)；三重门首触参考价 |",
        "| 目标价 / 止损参考价 | 现价×exp(q95/q05)；MC 分位数对应价格 |",
        f"| 趋势标签 / 趋势t值 | {trend_win_desc}；按 p 值最小选窗 |",
        "| 趋势p值 | 选中窗口斜率显著性 p 值 |",
        "| 触轨↑ / 触轨↓ / 到期 | 三重门 MC 首触概率（和为 100%） |",
        "| 障宽 | barrier_mult × 日 MAD 的对数收益阈值 |",
        "| 方向 / 置信 / 建议下注 | meta-labeling：趋势方向 + |P↑−P↓| 置信 → 分数 Kelly 仓位 |",
        "| q50_H | H 日累计对数收益 DGP 中位数（非 μ 点估计） |",
        "",
        "**ETF 只做多口径**：方向「空」对持仓=减/防守，对候选=回避；不做融券做空。",
        "",
        "**排序**：持仓置顶 → 信号(下注→观察→回避) → |期望EV%| 降序 → α不可靠/低样本下沉。",
        "",
        "---",
        "",
        "*由 `decision_pack/scripts/generate_short_horizon_board.py` 自动生成。*",
        "",
    ]
    return "\n".join(lines)


def build_short_horizon_board(
    *,
    pack_dir: Path,
    cluster_mapping_path: Path | None = None,
    provider_uri: str | Path | None = None,
    fat_cfg: FatTailConfig | None = None,
    sh_cfg: ShortHorizonConfig | None = None,
    horizons: tuple[int, ...] | list[int] = DEFAULT_HORIZONS,
) -> tuple[str, dict[int, pd.DataFrame], dict[int, str], int]:
    pack_cfg = load_decision_pack_config()
    fat_cfg = fat_cfg or fat_tail_config_from_raw(pack_cfg.raw.get("fat_tail"))
    base_sh_cfg = sh_cfg or short_horizon_config_from_raw(pack_cfg.raw.get("short_horizon"))
    # Align Kelly cap with fat-tail config when not overridden.
    if base_sh_cfg.kelly_fraction == ShortHorizonConfig.kelly_fraction:
        base_sh_cfg = ShortHorizonConfig(
            **{**base_sh_cfg.__dict__, "kelly_fraction": fat_cfg.kelly_fraction}
        )
    if base_sh_cfg.max_name_weight == ShortHorizonConfig.max_name_weight:
        base_sh_cfg = ShortHorizonConfig(
            **{**base_sh_cfg.__dict__, "max_name_weight": fat_cfg.max_name_weight}
        )

    defensive_codes = frozenset(c for c in pack_cfg.defensive_codes)
    as_of, _, cluster_frame, _ = load_fat_tail_data(
        pack_dir,
        cluster_mapping_path=cluster_mapping_path,
        provider_uri=provider_uri,
        defensive_codes=defensive_codes,
        cfg=fat_cfg,
    )

    from decision_pack.scripts.generate_fat_tail_risk_report import (
        build_universe_codes,
        load_holdings_from_pack,
        warmup_start,
    )
    from decision_pack.src.symbol_trend_board import load_cluster_mapping_codes
    from pipeline.price_adjustments import auto_qfq_adjust_close_panel
    from pipeline.strategy_layer_audit import load_qlib_close
    from runtime_paths import CLUSTER_MAPPING_SELECTED_TXT, QLIB_PROVIDER_URI

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

    csv_by_horizon: dict[int, pd.DataFrame] = {}
    report_by_horizon: dict[int, str] = {}
    for horizon in horizons:
        sh_cfg = ShortHorizonConfig(**{**base_sh_cfg.__dict__, "horizon": int(horizon)})
        csv_frame = build_short_horizon_csv(
            cluster_frame,
            panel,
            as_of,
            sh_cfg=sh_cfg,
        )
        markdown = build_short_horizon_report(
            as_of,
            pack_dir,
            sh_cfg=sh_cfg,
            n_symbols=len(csv_frame),
            live_close_count=live_stats.injected,
        )
        csv_by_horizon[int(horizon)] = csv_frame
        report_by_horizon[int(horizon)] = markdown
    return as_of, csv_by_horizon, report_by_horizon, live_stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate MLFAM short-horizon board CSV from decision_pack output",
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
        help="仅生成指定 H（省略则同时生成 h5 + h10）",
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

    as_of, csv_by_horizon, report_by_horizon, live_stats = build_short_horizon_board(
        pack_dir=pack_dir,
        cluster_mapping_path=args.cluster_mapping,
        provider_uri=args.provider_uri,
        horizons=horizons,
    )

    for horizon in horizons:
        csv_path = (
            args.output.expanduser().resolve()
            if single_horizon and args.output
            else short_horizon_csv_path(pack_dir, horizon)
        )
        report_path = (
            args.report_output.expanduser().resolve()
            if single_horizon and args.report_output
            else short_horizon_report_path(pack_dir, horizon)
        )

        csv_frame = csv_by_horizon[horizon]
        markdown = report_by_horizon[horizon]
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
        report_path.write_text(markdown, encoding="utf-8")
        print(
            f"[OK] short-horizon h{horizon} -> {csv_path} "
            f"({len(csv_frame)} rows, as_of={as_of}, "
            f"live_prices={live_stats.fetched}, injected={live_stats.injected}, "
            f"skipped_same={live_stats.skipped_same})"
        )
        print(f"[OK] disclaimer h{horizon}  -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
