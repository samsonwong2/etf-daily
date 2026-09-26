"""Write human-readable Decision Pack artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from decision_pack.src.action_plan import ActionPlan
from decision_pack.src.orders import orders_summary
from decision_pack.src.portfolio import PortfolioSnapshot
from decision_pack.src.recovery import RecoveryUpdate
from decision_pack.src.reconcile import ReconcileResult
from decision_pack.src.symbol_trend_board import TrendBoardVolContext
from decision_pack.src.tier import MarketSnapshot, TierDecision


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _num(value: float | None, *, pct: bool = False, digits: int = 2) -> str:
    if value is None:
        return "—"
    if pct:
        return f"{value * 100:.{digits}f}%"
    return f"{value:.{digits}f}"


def _vol_level_zh(level: str | None) -> str:
    mapping = {"low": "低", "mid": "中", "high": "高"}
    if level is None:
        return "—"
    return mapping.get(str(level), str(level))


def _parse_flags(flags: str | None) -> set[str]:
    return {part for part in str(flags or "").split("|") if part}


def _trend_row_mark(row: pd.Series) -> str:
    flags = _parse_flags(row.get("risk_flags"))
    dist = row.get("dist_ma20_pct")
    if "deep_below_ma20" in flags or any(f.startswith("below_ma20_") and f.endswith("d+") for f in flags):
        return "⚠⚠ "
    if "below_ma20" in flags and "high_vol" in flags:
        return "⚠⚠ "
    if dist is not None and dist < 0:
        return "⚠ "
    return ""


def _render_vol_context_lines(
    ctx: TrendBoardVolContext | None,
    *,
    universe_label: str = "权益持仓",
) -> list[str]:
    if ctx is None or ctx.equity_count == 0:
        return []
    bench_pct = "—" if ctx.bench_vol_pct is None else f"{ctx.bench_vol_pct * 100:.0f}%"
    bench_ann = _num(ctx.bench_vol_ann, pct=True)
    lines = [
        "> **波动环境**：基准 HS300 vol 分位 "
        f"{bench_pct}（{_vol_level_zh(ctx.bench_vol_level)}），"
        f"20 日年化实现波动 {bench_ann}；"
        f"{universe_label} {ctx.equity_high_vol_count}/{ctx.equity_count} 只高波动，"
        f"{ctx.equity_below_ma20_count}/{ctx.equity_count} 只破 MA20。",
    ]
    if (
        ctx.bench_vol_level == "high"
        and ctx.equity_count > 0
        and ctx.equity_high_vol_count == ctx.equity_count
    ):
        lines.append(
            "> 解读：全市场与持仓普遍高波动，`high_vol` 标签区分度低，优先看 MA20 / PnL / rs_20d。"
        )
    return lines


def _render_trend_board_table(
    trend_board: pd.DataFrame,
    *,
    empty_message: str = "(no symbols)",
) -> list[str]:
    lines = [
        "| code | name | wt | close | dist_ma20% | ret_5d | ret_20d | stack | vol | ann20 | ann5 | r5/20 | dd_20d | pnl% | <ma20d | rs_20d | flags |",
        "|------|------|----|-------|------------|--------|---------|-------|-----|-------|------|-------|--------|------|-------|--------|-------|",
    ]
    if trend_board.empty:
        lines.append(f"| — | {empty_message} | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |")
        return lines

    for _, row in trend_board.iterrows():
        mark = _trend_row_mark(row)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{mark}{row['code']}",
                    str(row["name"]),
                    f"{float(row['weight']):.4f}",
                    _num(row.get("close")),
                    _num(row.get("dist_ma20_pct")),
                    _num(row.get("ret_5d"), pct=True),
                    _num(row.get("ret_20d"), pct=True),
                    str(row.get("ma_stack") or "—"),
                    _vol_level_zh(row.get("vol_level")),
                    _num(row.get("vol_ann_20d"), pct=True),
                    _num(row.get("vol_ann_5d"), pct=True),
                    _num(row.get("vol_ratio_5_20"), digits=2),
                    _num(row.get("dd_20d_high"), pct=True),
                    _num(row.get("pnl_pct")),
                    _num(row.get("days_below_ma20"), digits=0),
                    _num(row.get("rs_20d_vs_bench"), pct=True),
                    str(row.get("risk_flags") or "—"),
                ]
            )
            + " |"
        )
    return lines


def render_trend_board_section(
    trend_board: pd.DataFrame,
    *,
    title: str = "## 持仓趋势板（v0.3 · 人工决策）",
    intro_lines: tuple[str, ...] = (
        "> 以下仅为指标展示，不构成自动卖单。Tier 与组合指标见上文。",
        "> v0.3：5/20 日年化波动率及比值（>1 表示近期波动加速）；⚠⚠ = 破 MA20 且高波动。",
    ),
    universe_label: str = "权益持仓",
    empty_message: str = "(no equity holdings)",
    vol_context: TrendBoardVolContext | None = None,
) -> list[str]:
    lines = [title, ""]
    lines.extend(intro_lines)
    lines.append("")
    lines.extend(_render_vol_context_lines(vol_context, universe_label=universe_label))
    if lines[-1] != "":
        lines.append("")
    lines.extend(_render_trend_board_table(trend_board, empty_message=empty_message))
    lines.append("")
    return lines


def render_cluster_trend_board_section(
    trend_board: pd.DataFrame,
    *,
    vol_context: TrendBoardVolContext | None = None,
    title: str = "## 待选池趋势板（v0.3 · cluster_mapping_selected · 人工决策）",
) -> list[str]:
    return render_trend_board_section(
        trend_board,
        title=title,
        intro_lines=(
            "> 覆盖 `cluster_mapping_selected.txt` 全部标的；未持仓 wt=0，pnl% 为空。",
            "> v0.3：5/20 日年化波动率及比值（>1 表示近期波动加速）；⚠⚠ = 破 MA20 且高波动。",
        ),
        universe_label="待选池",
        empty_message="(no cluster symbols)",
        vol_context=vol_context,
    )


def render_early_warning_section(early_warning: pd.DataFrame) -> list[str]:
    lines = [
        "## 标的预警清单（v0.3 · 黄/红灯 · 人工决策）",
        "",
        "> 黄灯：≥2 条弱化信号；红灯：深破 MA20 / 连续破线 / 回撤 / 浮亏≥7% / 模型零仓仍持。",
        "> 建议减仓比例为参考值，不构成自动卖单。",
        "",
        "| code | name | wt | level | suggest↓ | pnl% | dd_20d | dist_ma20% | reasons |",
        "|------|------|----|-------|----------|------|--------|------------|---------|",
    ]
    if early_warning is None or early_warning.empty:
        lines.append("| — | (no yellow/red alerts) | — | — | — | — | — | — | — |")
        lines.append("")
        return lines

    for _, row in early_warning.iterrows():
        level = str(row.get("alert_level") or "—")
        mark = "🔴 " if level == "red" else "🟡 "
        suggest = row.get("suggest_reduce_pct")
        suggest_s = "—" if suggest is None or pd.isna(suggest) else f"{float(suggest) * 100:.0f}%"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{mark}{row['code']}",
                    str(row.get("name") or "—"),
                    f"{float(row['weight']):.4f}",
                    level,
                    suggest_s,
                    _num(row.get("pnl_pct")),
                    _num(row.get("dd_20d_high"), pct=True),
                    _num(row.get("dist_ma20_pct")),
                    str(row.get("reasons") or "—"),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def render_pm_summary(action_plan: ActionPlan) -> list[str]:
    lines = ["## PM 决策摘要", ""]
    for item in action_plan.summary_lines:
        lines.append(f"- {item}")
    lines.append("")
    return lines


def render_l1_risk_budget(
    decision: TierDecision,
    market: MarketSnapshot,
    action_plan: ActionPlan | None,
) -> list[str]:
    tier_reduce_mv = action_plan.tier_reduce_mv if action_plan is not None else None
    lines = [
        "## L1 组合风险预算",
        "",
        (
            f"- executed_tier={decision.executed_tier} "
            f"(table={decision.table_tier}), regime={decision.regime}"
        ),
        f"- current_equity_exposure={_pct(decision.current_equity_exposure)}",
        f"- target_equity_exposure={_pct(decision.target_equity_exposure)}",
        f"- reduce_fraction={_pct(decision.reduce_fraction)}",
    ]
    if tier_reduce_mv is not None:
        lines.append(f"- tier_reduce_mv={tier_reduce_mv:.2f}（约 {tier_reduce_mv / 10000:.1f} 万）")
    lines.extend(
        [
            "",
            "### 市场快照",
            "",
            f"- benchmark: {market.benchmark}",
            f"- r_bench: {_pct(market.r_bench)}",
            f"- bench_3d: {_pct(market.bench_3d)}",
            f"- bench_5d: {_pct(market.bench_5d)}",
            f"- close_below_ma20: {market.close_below_ma20}",
            f"- vol_percentile_120d: {market.vol_percentile_120d:.2f}",
            f"- dd_port: {_pct(market.dd_port)}",
            f"- triggers: {', '.join(decision.triggers) or '(none)'}",
            "",
        ]
    )
    return lines


def render_l2_allocation(allocation_gaps: pd.DataFrame) -> list[str]:
    lines = [
        "## L2 配置对账",
        "",
        "> live 权重 vs pipeline 目标；`pipeline_zero_held` = 模型零仓仍持有。",
        "",
        "| code | name | live_wt | pipe_wt | gap | zero_held |",
        "|------|------|---------|---------|-----|-----------|",
    ]
    if allocation_gaps.empty:
        lines.append("| — | (no holdings) | — | — | — | — |")
        lines.append("")
        return lines

    for _, row in allocation_gaps.head(16).iterrows():
        pw = row.get("pipeline_weight")
        pw_s = "—" if pw is None or pd.isna(pw) else f"{float(pw):.4f}"
        zero = "Y" if row.get("pipeline_zero_held") else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["code"]),
                    str(row.get("name") or "—"),
                    f"{float(row['live_weight']):.4f}",
                    pw_s,
                    f"{float(row['gap']):+.4f}",
                    zero,
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def render_l3_action_plan(action_plan: ActionPlan) -> list[str]:
    lines = [
        "## L3 单标行动清单",
        "",
        "> IPS：observe=观察 / reduce=减30–50%（软红30%、硬红50%）/ exit=清仓；"
        "黄灯仅 Tier≥2 执行 30%，否则只观察；nuance：break_weak=破且弱 / break_strong=破但强。",
        "",
        "| priority | code | name | wt | ips | nuance | suggest↓ | est_shares | rs_20d | reasons |",
        "|----------|------|------|----|-----|--------|----------|------------|--------|---------|",
    ]
    if not action_plan.ranked_actions:
        lines.append("| — | — | (no yellow/red alerts) | — | — | — | — | — | — | — |")
        lines.append("")
        return lines

    for action in action_plan.ranked_actions:
        mark = "🔴 " if action.alert_level == "red" else "🟡 "
        shares_s = (
            "—"
            if action.est_sell_shares is None
            else f"{action.est_sell_shares:,}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{action.priority_score:.4f}",
                    f"{mark}{action.code}",
                    action.name,
                    f"{action.weight:.4f}",
                    action.ips_stage,
                    action.nuance,
                    f"{action.suggest_reduce_pct * 100:.0f}%",
                    shares_s,
                    _num(action.rs_20d_vs_bench, pct=True),
                    "|".join(action.reasons) or "—",
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def render_keep_watch(action_plan: ActionPlan) -> list[str]:
    lines = ["## 保留观察", ""]
    if not action_plan.keep_watch:
        lines.extend(["- (none)", ""])
        return lines
    for action in action_plan.keep_watch:
        rs_s = _num(action.rs_20d_vs_bench, pct=True)
        lines.append(
            f"- {action.code} {action.name}：wt={action.weight:.2f}, rs_20d={rs_s}, stack=intact"
        )
    lines.append("")
    return lines


def render_t1_execution_plan(
    action_plan: ActionPlan,
    *,
    display_only: bool,
) -> list[str]:
    lines = ["## T+1 执行清单", ""]
    if display_only:
        lines.append(
            "- **v0 展示模式**：以下为参考卖单，请人工确认后下单；`orders_T+1.csv` 仍为 HOLD 占位。"
        )
        lines.append("")
    lines.extend(
        [
            "| phase | code | name | ips | nuance | est_shares | est_mv | note |",
            "|-------|------|------|-----|--------|------------|--------|------|",
        ]
    )
    for row in action_plan.execution_rows:
        code = row.code or "—"
        name = row.name or "—"
        ips = row.ips_stage or "—"
        nuance = row.nuance or "—"
        shares = (
            "—"
            if row.est_sell_shares is None
            else f"{row.est_sell_shares:,}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    row.phase,
                    code,
                    name,
                    ips,
                    nuance,
                    shares,
                    f"{row.est_sell_mv:.2f}",
                    row.note,
                ]
            )
            + " |"
        )
    lines.append("")
    budget = action_plan.tier_reduce_mv
    cumulative = action_plan.t1_cumulative_mv
    if budget > 0 and cumulative > 0:
        fill_pct = cumulative / budget * 100
        lines.append(
            f"- **T+1 累计**：参考卖单合计约 {cumulative / 10000:.1f} 万 MV"
            f"（Tier 预算 {budget / 10000:.1f} 万的 {fill_pct:.1f}%）。"
        )
        lines.append(
            f"- **截断规则**：按 L3 优先级累加至预算的 95% 后停止纳入（最后一笔可略超 95%）。"
        )
    deferred = action_plan.t1_deferred_actions
    if deferred:
        parts = [
            f"{action.code}({action.name}, {action.ips_stage}, "
            f"suggest↓={action.suggest_reduce_pct * 100:.0f}%)"
            for action in deferred
        ]
        lines.append(
            f"- **本轮未纳入 T+1**（L3 仍有告警但预算已满）：{'、'.join(parts)}"
        )
    lines.append("")
    return lines


def render_holdings_summary(portfolio: PortfolioSnapshot) -> list[str]:
    lines = [
        "### 持仓摘要",
        "",
        f"- source: {portfolio.source}",
        f"- total_assets: {portfolio.total_assets:.4f}",
        f"- stock_mv: {portfolio.stock_mv:.4f}",
        f"- cash_mv: {portfolio.cash_mv:.4f}",
        f"- equity_exposure: {_pct(portfolio.equity_exposure)}",
        "",
        "| code | name | weight | market_value | cost | close |",
        "|------|------|--------|--------------|------|-------|",
    ]
    for row in portfolio.holdings:
        cost = "—" if row.cost_price is None else f"{row.cost_price:.4f}"
        close = "—" if row.close_price is None else f"{row.close_price:.4f}"
        lines.append(
            f"| {row.code} | {row.name} | {row.weight:.4f} | {row.market_value:.2f} | {cost} | {close} |"
        )
    lines.append("")
    return lines


def _render_footer_sections(
    *,
    decision: TierDecision,
    market: MarketSnapshot,
    portfolio: PortfolioSnapshot,
    orders: pd.DataFrame,
    recovery: RecoveryUpdate,
    system_vs_manual: dict[str, Any],
    summary: dict[str, float],
    display_only: bool,
) -> list[str]:
    lines = [
        "## 系统 vs 手动",
        "",
        f"- exposure_scale: {system_vs_manual.get('exposure_scale')}",
        f"- risk_macro_budget: {system_vs_manual.get('risk_macro_budget')}",
        f"- manual_tier: {system_vs_manual.get('manual_tier')}",
        f"- manual_target_exposure: {_pct(system_vs_manual.get('manual_target_exposure'))}",
        f"- l3_gap: {system_vs_manual.get('l3_gap')}",
        f"- quarter_tier2_manual_count: {system_vs_manual.get('quarter_tier2_manual_count')}",
        "",
    ]
    if not display_only:
        lines.extend(
            [
                "## T+1 执行清单",
                "",
                "- 竞价或开盘前 30 分钟按 `orders_T+1.csv` 同比例减仓。",
                f"- 预计卖出占股票市值比例: {_pct(summary.get('sell_fraction_of_stock'))}",
                f"- 预计卖出总市值: {summary.get('total_sell_mv', 0.0):.4f}",
                "- 已低于目标暴露时不主动加仓，仅观望。",
                "",
            ]
        )

    lines.extend(
        [
            "## 恢复监视",
            "",
            f"- recovery_signal: {recovery.recovery_signal}",
            f"- addback_fraction: {_pct(recovery.addback_fraction)}",
            f"- reason: {recovery.reason}",
        ]
    )
    for item in recovery.watch_items:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Override 日志模板",
            "",
            "```json",
            json.dumps(
                {
                    "date": decision.as_of,
                    "system": (
                        f"exposure_scale={system_vs_manual.get('exposure_scale')}, "
                        f"risk_macro_budget={system_vs_manual.get('risk_macro_budget')}"
                    ),
                    "manual": (
                        f"tier{decision.executed_tier} "
                        f"{_pct(decision.current_equity_exposure)}→{_pct(decision.target_equity_exposure)}"
                    ),
                    "trigger": (
                        f"dd_port={market.dd_port:.4f} bench={market.r_bench:.4f}"
                    ),
                },
                ensure_ascii=False,
            ),
            "```",
            "",
        ]
    )

    if system_vs_manual.get("quarter_l3_gap_alert"):
        lines.extend(
            [
                "> 脚注：本季度 manual_tier≥2 且 exposure_scale≈1.0 次数已达阈值，"
                "建议评估 L3 tail 触发改造。",
                "",
            ]
        )
    return lines


def render_decision_pack_report(
    *,
    decision: TierDecision,
    market: MarketSnapshot,
    portfolio: PortfolioSnapshot,
    orders: pd.DataFrame,
    reconcile: ReconcileResult | None,
    recovery: RecoveryUpdate,
    system_vs_manual: dict[str, Any],
    orders_mode: str = "proportional_legacy",
    trend_board: pd.DataFrame | None = None,
    trend_vol_context: TrendBoardVolContext | None = None,
    cluster_trend_board: pd.DataFrame | None = None,
    cluster_vol_context: TrendBoardVolContext | None = None,
    early_warning: pd.DataFrame | None = None,
    action_plan: ActionPlan | None = None,
) -> str:
    summary = orders_summary(orders, portfolio)
    display_only = orders_mode == "display_only"
    action = "人工决策（展示模式）" if display_only else ("SELL" if decision.reduce_fraction > 0 else "HOLD")
    if not display_only and recovery.recovery_signal:
        action = "BUY (recovery)"

    lines = [
        "# Decision Pack Report",
        "",
        f"**As of:** {decision.as_of}",
        "",
    ]
    if reconcile is not None and reconcile.warning:
        lines.extend(["## ⚠️ RECONCILE_WARNING", "", reconcile.message, ""])

    if display_only and action_plan is not None:
        lines.extend(render_pm_summary(action_plan))
        lines.extend(render_l1_risk_budget(decision, market, action_plan))
        lines.extend(render_holdings_summary(portfolio))
        lines.extend(render_l2_allocation(action_plan.allocation_gaps))
        lines.extend(render_l3_action_plan(action_plan))
        lines.extend(render_t1_execution_plan(action_plan, display_only=True))
        lines.extend(render_keep_watch(action_plan))

        if trend_board is not None:
            lines.extend(
                render_trend_board_section(
                    trend_board,
                    title="## 附录 A · 持仓趋势板（v0.3 · 人工决策）",
                    vol_context=trend_vol_context,
                )
            )
        if cluster_trend_board is not None:
            lines.extend(
                render_cluster_trend_board_section(
                    cluster_trend_board,
                    vol_context=cluster_vol_context,
                    title="## 附录 B · 待选池趋势板（v0.3 · cluster_mapping_selected · 人工决策）",
                )
            )

        lines.extend(
            _render_footer_sections(
                decision=decision,
                market=market,
                portfolio=portfolio,
                orders=orders,
                recovery=recovery,
                system_vs_manual=system_vs_manual,
                summary=summary,
                display_only=True,
            )
        )
        return "\n".join(lines)

    lines.extend(
        [
            "## 一句话结论",
            "",
            (
                f"- executed_tier={decision.executed_tier} "
                f"(table={decision.table_tier}), regime={decision.regime}"
            ),
            f"- current_equity_exposure={_pct(decision.current_equity_exposure)}",
            f"- target_equity_exposure={_pct(decision.target_equity_exposure)}",
            f"- reduce_fraction={_pct(decision.reduce_fraction)}",
            f"- **T+1 action:** {action}",
            "",
            "## 市场快照",
            "",
            f"- benchmark: {market.benchmark}",
            f"- r_bench: {_pct(market.r_bench)}",
            f"- bench_3d: {_pct(market.bench_3d)}",
            f"- bench_5d: {_pct(market.bench_5d)}",
            f"- close_below_ma20: {market.close_below_ma20}",
            f"- vol_percentile_120d: {market.vol_percentile_120d:.2f}",
            f"- dd_port: {_pct(market.dd_port)}",
            f"- triggers: {', '.join(decision.triggers) or '(none)'}",
            "",
            "## 持仓摘要",
            "",
            f"- source: {portfolio.source}",
            f"- total_assets: {portfolio.total_assets:.4f}",
            f"- stock_mv: {portfolio.stock_mv:.4f}",
            f"- cash_mv: {portfolio.cash_mv:.4f}",
            f"- equity_exposure: {_pct(portfolio.equity_exposure)}",
            "",
            "| code | name | weight | market_value | cost | close |",
            "|------|------|--------|--------------|------|-------|",
        ]
    )

    for row in portfolio.holdings:
        cost = "—" if row.cost_price is None else f"{row.cost_price:.4f}"
        close = "—" if row.close_price is None else f"{row.close_price:.4f}"
        lines.append(
            f"| {row.code} | {row.name} | {row.weight:.4f} | {row.market_value:.2f} | {cost} | {close} |"
        )

    if trend_board is not None:
        lines.extend([""] + render_trend_board_section(trend_board, vol_context=trend_vol_context))

    if cluster_trend_board is not None:
        lines.extend(
            [""]
            + render_cluster_trend_board_section(
                cluster_trend_board,
                vol_context=cluster_vol_context,
            )
        )

    if early_warning is not None:
        lines.extend([""] + render_early_warning_section(early_warning))

    lines.extend(
        _render_footer_sections(
            decision=decision,
            market=market,
            portfolio=portfolio,
            orders=orders,
            recovery=recovery,
            system_vs_manual=system_vs_manual,
            summary=summary,
            display_only=display_only,
        )
    )
    if display_only:
        lines.insert(
            lines.index("## 系统 vs 手动"),
            "- **v0 展示模式**：不生成自动卖单，请结合「持仓趋势板」与「待选池趋势板」人工决策。\n"
            "- `orders_T+1.csv` 仅含 HOLD 占位行（reason=display_only_v0）。\n",
        )

    return "\n".join(lines)


def write_pack_outputs(
    output_dir: Path,
    *,
    decision: TierDecision,
    market: MarketSnapshot,
    portfolio: PortfolioSnapshot,
    orders: pd.DataFrame,
    reconcile: ReconcileResult | None,
    reconcile_md: str | None,
    recovery: RecoveryUpdate,
    system_vs_manual: dict[str, Any],
    inputs_manifest: dict[str, Any],
    orders_mode: str = "proportional_legacy",
    trend_board: pd.DataFrame | None = None,
    trend_vol_context: TrendBoardVolContext | None = None,
    cluster_trend_board: pd.DataFrame | None = None,
    cluster_vol_context: TrendBoardVolContext | None = None,
    early_warning: pd.DataFrame | None = None,
    action_plan: ActionPlan | None = None,
    universe_live_close: pd.DataFrame | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    report_path = output_dir / "DECISION_PACK_REPORT.md"
    report_path.write_text(
        render_decision_pack_report(
            decision=decision,
            market=market,
            portfolio=portfolio,
            orders=orders,
            reconcile=reconcile,
            recovery=recovery,
            system_vs_manual=system_vs_manual,
            orders_mode=orders_mode,
            trend_board=trend_board,
            trend_vol_context=trend_vol_context,
            cluster_trend_board=cluster_trend_board,
            cluster_vol_context=cluster_vol_context,
            early_warning=early_warning,
            action_plan=action_plan,
        ),
        encoding="utf-8",
    )
    written["report"] = report_path

    tier_path = output_dir / "tier_decision.json"
    tier_path.write_text(
        json.dumps(decision.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    written["tier_decision"] = tier_path

    market_path = output_dir / "market_snapshot.json"
    market_path.write_text(
        json.dumps(
            {
                "as_of": market.as_of,
                "benchmark": market.benchmark,
                "r_bench": market.r_bench,
                "bench_3d": market.bench_3d,
                "bench_5d": market.bench_5d,
                "close_below_ma20": market.close_below_ma20,
                "vol_percentile_120d": market.vol_percentile_120d,
                "dd_port": market.dd_port,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    written["market_snapshot"] = market_path

    portfolio_path = output_dir / "portfolio_snapshot.csv"
    portfolio.to_frame().to_csv(portfolio_path, index=False)
    written["portfolio_snapshot"] = portfolio_path

    orders_path = output_dir / "orders_T+1.csv"
    orders.to_csv(orders_path, index=False)
    written["orders"] = orders_path

    if trend_board is not None:
        trend_path = output_dir / "holdings_trend_board.csv"
        trend_board.to_csv(trend_path, index=False)
        written["holdings_trend_board"] = trend_path

    if cluster_trend_board is not None:
        cluster_path = output_dir / "cluster_trend_board.csv"
        cluster_trend_board.to_csv(cluster_path, index=False)
        written["cluster_trend_board"] = cluster_path

    if universe_live_close is not None and not universe_live_close.empty:
        from decision_pack.src.pack_live_prices import UNIVERSE_LIVE_CLOSE_CSV

        live_path = output_dir / UNIVERSE_LIVE_CLOSE_CSV
        universe_live_close.to_csv(live_path, index=False, encoding="utf-8-sig")
        written["universe_live_close"] = live_path

    if early_warning is not None:
        ew_path = output_dir / "early_warning.csv"
        early_warning.to_csv(ew_path, index=False)
        written["early_warning"] = ew_path

    if action_plan is not None:
        plan_path = output_dir / "action_plan.json"
        plan_path.write_text(
            json.dumps(action_plan.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written["action_plan"] = plan_path

    if reconcile is not None and reconcile_md is not None:
        reconcile_path = output_dir / "reconcile_report.md"
        reconcile_path.write_text(reconcile_md, encoding="utf-8")
        written["reconcile_report"] = reconcile_path

    recovery_path = output_dir / "recovery_state.json"
    recovery_path.write_text(
        json.dumps(recovery.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    written["recovery_state"] = recovery_path

    system_path = output_dir / "system_vs_manual.json"
    system_path.write_text(
        json.dumps(system_vs_manual, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    written["system_vs_manual"] = system_path

    manifest_path = output_dir / "inputs_manifest.json"
    manifest_path.write_text(
        json.dumps(inputs_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    written["inputs_manifest"] = manifest_path

    return written


__all__ = [
    "render_cluster_trend_board_section",
    "render_decision_pack_report",
    "render_early_warning_section",
    "render_keep_watch",
    "render_l1_risk_budget",
    "render_l2_allocation",
    "render_l3_action_plan",
    "render_pm_summary",
    "render_t1_execution_plan",
    "render_trend_board_section",
    "write_pack_outputs",
]
