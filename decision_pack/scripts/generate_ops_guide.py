#!/usr/bin/env python3
"""Generate T+1 operational guidance markdown from decision_pack artifacts.

Example:
  cd ~/gitee/skfolio_csi300/etf_strategy_clean
  PY=~/etf-daily-output/python/envs/py312/bin/python
  $PY decision_pack/scripts/generate_ops_guide.py \\
    --pack-dir ~/temp/decision_packs/20260612 \\
    --notes workspace/notes/20260612.md \\
    --output ~/temp/decision_packs/20260612/OPS_GUIDE.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from decision_pack.src.config import load_decision_pack_config


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float | None, *, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def _wan(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / 10000:.1f} 万"


def _read_notes_excerpt(notes_path: Path | None) -> dict[str, str]:
    if notes_path is None or not notes_path.exists():
        return {}
    text = notes_path.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip().lstrip(">").strip()
        if ("非调仓日" in stripped or "调仓日" in stripped) and "距上次调仓" in stripped:
            out["rebalance_line"] = stripped
            break
    m = re.search(r"(\d+)\s*只持仓权重合计[^\n。]+", text)
    if m:
        out["pipeline_weight_line"] = m.group(0).strip()
    mu_rows = re.findall(r"^\|\s*\d+\s*\|\s*(SH\d+|SZ\d+)", text, re.MULTILINE)
    if mu_rows:
        out["pipeline_holdings_count"] = str(len(set(mu_rows)))
    return out


def _regime_zh(regime: str) -> str:
    return {
        "structural_weakness": "结构弱市",
        "sentiment_shock": "情绪急跌",
    }.get(regime, regime)


def _ips_zh(stage: str | None) -> str:
    return {
        "observe": "观察",
        "reduce": "减仓",
        "exit": "清仓",
    }.get(stage or "", stage or "—")


def _nuance_zh(nuance: str | None) -> str:
    return {
        "break_weak": "破且弱",
        "break_strong": "破但强",
        "intact": "趋势完好",
    }.get(nuance or "", nuance or "—")


def _reasons_zh(reasons: list[str]) -> str:
    mapping = {
        "below_ma20_3d+": "破 MA20≥3 天",
        "dd5_20d": "20 日高点回撤≥5%",
        "pnl7": "浮亏≥7%",
        "deep_below_ma20": "深度破 MA20",
    }
    if not reasons:
        return "—"
    return " + ".join(mapping.get(r, r) for r in reasons)


def build_ops_guide_markdown(
    *,
    pack_dir: Path,
    notes_path: Path | None = None,
) -> str:
    tier_path = pack_dir / "tier_decision.json"
    plan_path = pack_dir / "action_plan.json"
    recovery_path = pack_dir / "recovery_state.json"

    for required in (tier_path, plan_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing required artifact: {required}")

    tier = _load_json(tier_path)
    plan = _load_json(plan_path)
    recovery = _load_json(recovery_path) if recovery_path.exists() else {}

    cfg = load_decision_pack_config()
    defensive = cfg.raw.get("defensive_codes") or ["SH511260", "SH511010"]

    as_of = tier.get("as_of", "unknown")
    executed = int(tier.get("executed_tier", 0))
    table = int(tier.get("table_tier", executed))
    regime = str(tier.get("regime", ""))
    current = float(tier.get("current_equity_exposure", 0.0))
    target = tier.get("target_equity_exposure")
    target_f = float(target) if target is not None else None
    reduce_frac = float(tier.get("reduce_fraction", 0.0))
    tier_mv = float(plan.get("tier_reduce_mv", 0.0))
    t1_cum = float(plan.get("t1_cumulative_mv", 0.0))
    triggers = tier.get("triggers") or []
    structural = tier.get("structural_rule_hits") or []

    execution_rows = [
        row for row in plan.get("execution_rows", []) if row.get("phase") == "symbol"
    ]
    deferred = plan.get("t1_deferred_actions") or []
    keep_watch = plan.get("keep_watch") or []

    recovery_signal = bool(recovery.get("recovery_signal", False))
    addback = float(recovery.get("addback_fraction", 0.0))
    recovery_reason = str(recovery.get("reason", "n/a"))
    watch_items = recovery.get("watch_items") or []

    notes_info = _read_notes_excerpt(notes_path)
    budget_ratio = (t1_cum / tier_mv * 100) if tier_mv > 0 else 0.0

    lines: list[str] = [
        f"# T+1 操作指引",
        "",
        f"**As of:** {as_of}  ",
        f"**数据来源:** `{pack_dir}`",
    ]
    if notes_path and notes_path.exists():
        lines.append(f"**笔记:** `{notes_path}`")
    lines.extend(["", "---", ""])

    lines.extend(
        [
            "## 1. 组合状态（L1）",
            "",
            "| 维度 | 结论 |",
            "|------|------|",
            f"| 执行档位 | Tier **{executed}**（查表 Tier {table}） |",
            f"| 市场环境 | {_regime_zh(regime)} |",
            f"| 权益暴露 | {_pct(current)} → {_pct(target_f)} |",
            f"| 整体减仓比例 | **{_pct(reduce_frac)}** |",
            f"| 需释放股票 MV | **{_wan(tier_mv)}**（约 {tier_mv:,.0f} 元） |",
            f"| 组合 20 日回撤 dd_port | {_pct(float(tier.get('dd_port', 0.0)))} |",
            f"| 基准当日 r_bench | {_pct(float(tier.get('r_bench', 0.0)))} |",
            f"| L1 触发 | {', '.join(triggers) if triggers else '—'} |",
            f"| 结构弱市规则 | {', '.join(structural) if structural else '—'} |",
            "",
        ]
    )

    lines.extend(
        [
            "## 2. 恢复与调仓",
            "",
            f"- **recovery_signal:** `{recovery_signal}`（addback {_pct(addback)}，reason={recovery_reason}）",
        ]
    )
    for item in watch_items:
        lines.append(f"- {item}")
    if notes_info.get("rebalance_line"):
        lines.append(f"- **笔记:** {notes_info['rebalance_line']}")
    pipeline_line = notes_info.get("pipeline_weight_line")
    rebalance_line = notes_info.get("rebalance_line", "")
    if pipeline_line and pipeline_line not in rebalance_line:
        lines.append(f"- **笔记 pipeline 权重:** {pipeline_line}")
    lines.append("")

    lines.extend(
        [
            "## 3. 今日 T+1 卖出清单（人工确认后下单）",
            "",
            f"> display_only 参考卖单；累计约 **{_wan(t1_cum)}**（Tier 预算 {_wan(tier_mv)} 的 **{budget_ratio:.1f}%**）",
            "",
            "| # | 代码 | 名称 | IPS | 强弱 | suggest↓ | 股数 | 约 MV | 原因 |",
            "|---|------|------|-----|------|----------|------|-------|------|",
        ]
    )
    for idx, row in enumerate(execution_rows, start=1):
        code = row.get("code", "")
        name = row.get("name", "")
        ips = _ips_zh(row.get("ips_stage"))
        nuance = _nuance_zh(row.get("nuance"))
        note = str(row.get("note", ""))
        suggest_pct = ""
        m = re.search(r"suggest↓=(\d+)%", note)
        if m:
            suggest_pct = f"{m.group(1)}%"
        reasons_m = re.search(r"reasons=([^,]+(?:\|[^,]+)*)", note)
        reasons_raw = reasons_m.group(1) if reasons_m else "—"
        reasons_list = [p for p in reasons_raw.split("|") if p and p != "—"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(code),
                    str(name),
                    ips,
                    nuance,
                    suggest_pct or "—",
                    str(row.get("est_sell_shares") or "—"),
                    _wan(float(row.get("est_sell_mv") or 0.0)),
                    _reasons_zh(reasons_list) if reasons_list else reasons_raw,
                ]
            )
            + " |"
        )
    if not execution_rows:
        lines.append("| — | — | 无纳入 T+1 的卖单 | — | — | — | — | — | — |")
    lines.append("")

    lines.extend(["## 4. 保留观察（今日不卖）", ""])
    if keep_watch:
        lines.extend(
            [
                "| 代码 | 名称 | 权重 | rs_20d |",
                "|------|------|------|--------|",
            ]
        )
        for row in keep_watch:
            rs = row.get("rs_20d_vs_bench")
            rs_s = _pct(float(rs)) if rs is not None else "n/a"
            lines.append(
                f"| {row.get('code')} | {row.get('name')} | "
                f"{float(row.get('weight', 0.0)):.2%} | {rs_s} |"
            )
    else:
        lines.append("（无）")
    lines.append("")

    lines.extend(["## 5. 预算已满 · 可择机补卖", ""])
    if deferred:
        lines.extend(
            [
                "| 代码 | 名称 | IPS | suggest↓ | 约 MV | 原因 |",
                "|------|------|-----|----------|-------|------|",
            ]
        )
        for row in deferred:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("code", "")),
                        str(row.get("name", "")),
                        _ips_zh(row.get("ips_stage")),
                        _pct(float(row.get("suggest_reduce_pct", 0.0))),
                        _wan(float(row.get("est_sell_mv", 0.0))),
                        _reasons_zh(list(row.get("reasons") or [])),
                    ]
                )
                + " |"
            )
    else:
        lines.append("（无 deferred）")
    lines.append("")

    cash_wan = tier_mv / 10000.0
    post_stock_wan = (
        (tier_mv / reduce_frac * target_f) / 10000.0
        if reduce_frac > 0 and target_f is not None
        else None
    )
    lines.extend(
        [
            "## 6. 卖出回款怎么处理",
            "",
            "### 结论：**今天以持有现金为主，不买新 ETF**",
            "",
            "理由：",
            "",
            f"1. L1 目标权益 **{_pct(target_f)}** — 减 **{_pct(reduce_frac)}** 后应约 "
            f"**{_wan(post_stock_wan) if post_stock_wan else 'n/a'} 股票 + {_wan(cash_wan)} 现金**（风险上限，不是换仓信号）。",
            f"2. recovery 未开启（`{recovery_reason}`），addback **{_pct(addback)}**。",
            "3. 结构弱市 + 高波动环境下，不宜按 μ 排名开新仓。",
            "4. decision_pack 当前为 display_only，**无自动 BUY 指令**。",
            "",
            "| 做法 | 说明 |",
            "|------|------|",
            f"| **A. 留现金（推荐）** | 卖完后核对 `stock_mv/total ≈ {_pct(target_f)}` |",
            f"| **B. 配国债（Playbook）** | 约 {_wan(tier_mv)} 买入 "
            + " / ".join(str(c) for c in defensive)
            + " |",
            "| **不建议** | 用现金加仓保留观察名单或按 μ 表买未持仓标的 |",
            "",
        ]
    )

    lines.extend(
        [
            "## 7. 建议的一日流程",
            "",
            "```text",
            "开盘前",
            "  ① 核对 pack 日期与 live 持仓一致",
            "  ② 按 §3 执行卖出（优先 break_weak / exit）",
            "  ③ 不动 §4 保留观察",
            "  ④ 回款留现金（或可选配国债）",
            "",
            "不做",
            "  ✗ 不因 μ 排名买入未持仓标的",
            "  ✗ 不因 rs 强给 §3 中仍建议减仓的标的加仓",
            "  ✗ 非调仓日不按 pipeline 目标一次性对齐",
            "",
            "收盘后",
            "  ⑤ 核对权益暴露是否 ≈ " + _pct(target_f),
            "  ⑥ 可选：补卖 §5 deferred",
            "```",
            "",
        ]
    )

    summary = plan.get("summary_lines") or []
    if summary:
        lines.extend(["## 8. PM 摘要（action_plan）", ""])
        for item in summary:
            lines.append(f"- {item}")
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            "*由 `decision_pack/scripts/generate_ops_guide.py` 自动生成；卖单仅供人工确认。*",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate T+1 operational guidance markdown from decision_pack output",
    )
    parser.add_argument(
        "--pack-dir",
        type=Path,
        required=True,
        help="decision_pack generate 输出目录（含 tier_decision.json、action_plan.json）",
    )
    parser.add_argument(
        "--notes",
        type=Path,
        default=None,
        help="可选：workspace/notes/YYYYMMDD.md，用于补充调仓日 / pipeline 说明",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 MD 路径（默认：<pack-dir>/OPS_GUIDE.md）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    pack_dir = args.pack_dir.expanduser().resolve()
    notes_path = args.notes.expanduser().resolve() if args.notes else None
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else pack_dir / "OPS_GUIDE.md"
    )

    markdown = build_ops_guide_markdown(pack_dir=pack_dir, notes_path=notes_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"[OK] ops guide -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
