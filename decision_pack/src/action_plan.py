"""Institutional-style action plan: L2 allocation + L3 ranked symbol actions."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import pandas as pd

from decision_pack.src.config import DecisionPackConfig
from decision_pack.src.early_warning import classify_symbol_alert, parse_risk_flags
from decision_pack.src.portfolio import PortfolioSnapshot
from decision_pack.src.tier import TierDecision

IpsStage = Literal["observe", "reduce", "exit"]
Nuance = Literal["break_weak", "break_strong", "intact"]

ETF_LOT_SIZE = 100
HEAVY_WEIGHT_THRESHOLD = 0.10
KEEP_WATCH_RS_THRESHOLD = 0.05


@dataclass(frozen=True)
class SymbolAction:
    code: str
    name: str
    weight: float
    pipeline_weight: float | None
    alert_level: str
    ips_stage: IpsStage
    nuance: Nuance
    suggest_reduce_pct: float
    priority_score: float
    est_sell_shares: int | None
    est_sell_mv: float
    rs_20d_vs_bench: float | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


@dataclass(frozen=True)
class ExecutionRow:
    phase: str
    code: str | None
    name: str | None
    ips_stage: str | None
    nuance: str | None
    est_sell_shares: int | None
    est_sell_mv: float
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionPlan:
    tier_reduce_mv: float
    ranked_actions: tuple[SymbolAction, ...]
    keep_watch: tuple[SymbolAction, ...]
    allocation_gaps: pd.DataFrame
    execution_rows: tuple[ExecutionRow, ...]
    summary_lines: tuple[str, ...]
    t1_cumulative_mv: float = 0.0
    t1_deferred_actions: tuple[SymbolAction, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier_reduce_mv": self.tier_reduce_mv,
            "ranked_actions": [row.to_dict() for row in self.ranked_actions],
            "keep_watch": [row.to_dict() for row in self.keep_watch],
            "execution_rows": [row.to_dict() for row in self.execution_rows],
            "summary_lines": list(self.summary_lines),
            "t1_cumulative_mv": self.t1_cumulative_mv,
            "t1_deferred_actions": [row.to_dict() for row in self.t1_deferred_actions],
        }


def _float_or_none(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not pd.notna(out):
        return None
    return out


def _parse_reasons(reasons: str | None) -> tuple[str, ...]:
    if not reasons:
        return ()
    return tuple(part for part in str(reasons).split("|") if part)


def estimate_sell_shares(shares: float | None, suggest_reduce_pct: float) -> int | None:
    if shares is None or suggest_reduce_pct <= 0:
        return None
    raw = int(shares * suggest_reduce_pct)
    if raw <= 0:
        return None
    return (raw // ETF_LOT_SIZE) * ETF_LOT_SIZE


def classify_ips_stage(
    alert_level: str,
    reasons: tuple[str, ...],
    suggest_reduce_pct: float,
) -> IpsStage:
    if suggest_reduce_pct >= 0.999:
        return "exit"
    if alert_level == "yellow":
        return "observe"
    if alert_level == "red":
        if suggest_reduce_pct > 0:
            return "reduce"
        return "observe"
    return "observe"


def classify_nuance(
    alert_level: str,
    rs_20d_vs_bench: float | None,
    risk_flags: str | None,
) -> Nuance:
    if alert_level == "green":
        return "intact"
    flags = parse_risk_flags(risk_flags)
    if "deep_below_ma20" in flags:
        return "break_weak"
    rs = rs_20d_vs_bench if rs_20d_vs_bench is not None else 0.0
    if rs < KEEP_WATCH_RS_THRESHOLD or "bear_stack" in flags or "weak_rs" in flags:
        return "break_weak"
    return "break_strong"


def priority_score(
    *,
    weight: float,
    suggest_reduce_pct: float,
    reason_count: int,
    nuance: Nuance,
) -> float:
    if suggest_reduce_pct <= 0:
        return 0.0
    nuance_mult = {
        "break_weak": 1.2,
        "break_strong": 0.8,
        "intact": 0.3,
    }[nuance]
    heavy_mult = 1.1 if weight >= HEAVY_WEIGHT_THRESHOLD else 1.0
    return weight * suggest_reduce_pct * (1.0 + 0.5 * reason_count) * nuance_mult * heavy_mult


def build_allocation_gaps(
    trend_board: pd.DataFrame | None,
    pipeline_weights: dict[str, float] | None,
) -> pd.DataFrame:
    if trend_board is None or trend_board.empty:
        return pd.DataFrame(
            columns=[
                "code",
                "name",
                "live_weight",
                "pipeline_weight",
                "gap",
                "pipeline_zero_held",
            ]
        )

    pipe = pipeline_weights or {}
    rows: list[dict[str, Any]] = []
    for _, row in trend_board.iterrows():
        weight = _float_or_none(row.get("weight")) or 0.0
        if weight <= 0:
            continue
        code = str(row["code"])
        pw = _float_or_none(pipe.get(code))
        pipeline_zero = pw is not None and pw <= 0
        gap = weight - (pw if pw is not None else 0.0)
        rows.append(
            {
                "code": code,
                "name": row.get("name"),
                "live_weight": weight,
                "pipeline_weight": pw,
                "gap": gap,
                "pipeline_zero_held": pipeline_zero,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["abs_gap"] = frame["gap"].abs()
    return frame.sort_values("abs_gap", ascending=False).drop(columns=["abs_gap"]).reset_index(drop=True)


def _holding_lookup(portfolio: PortfolioSnapshot) -> dict[str, Any]:
    return {h.code: h for h in portfolio.holdings}


def _symbol_action_from_row(
    row: pd.Series,
    *,
    pipeline_weight: float | None,
    holding: Any | None,
    executed_tier: int | None,
    config: DecisionPackConfig | None,
) -> SymbolAction:
    alert = classify_symbol_alert(
        row,
        pipeline_weight=pipeline_weight,
        executed_tier=executed_tier,
        config=config,
    )
    reasons = alert.reasons
    flags = str(row.get("risk_flags") or "")
    rs = _float_or_none(row.get("rs_20d_vs_bench"))
    nuance = classify_nuance(alert.level, rs, flags)
    ips = classify_ips_stage(alert.level, reasons, alert.suggest_reduce_pct)
    reason_count = len(reasons) if alert.level == "red" else len(reasons)
    score = priority_score(
        weight=float(row.get("weight") or 0.0),
        suggest_reduce_pct=alert.suggest_reduce_pct,
        reason_count=reason_count,
        nuance=nuance,
    )
    shares = holding.shares if holding is not None else None
    close = _float_or_none(row.get("close")) or (
        holding.close_price if holding is not None else None
    )
    est_shares = estimate_sell_shares(shares, alert.suggest_reduce_pct)
    est_mv = 0.0
    if est_shares is not None and close is not None and close > 0:
        est_mv = est_shares * close
    elif alert.suggest_reduce_pct > 0 and holding is not None:
        est_mv = holding.market_value * alert.suggest_reduce_pct

    return SymbolAction(
        code=str(row["code"]),
        name=str(row.get("name") or row["code"]),
        weight=float(row.get("weight") or 0.0),
        pipeline_weight=pipeline_weight,
        alert_level=alert.level,
        ips_stage=ips,
        nuance=nuance,
        suggest_reduce_pct=alert.suggest_reduce_pct,
        priority_score=score,
        est_sell_shares=est_shares,
        est_sell_mv=est_mv,
        rs_20d_vs_bench=rs,
        reasons=reasons,
    )


T1_BUDGET_FILL_THRESHOLD = 0.95


def build_execution_rows(
    *,
    tier_reduce_mv: float,
    ranked_actions: tuple[SymbolAction, ...],
) -> tuple[tuple[ExecutionRow, ...], float, tuple[SymbolAction, ...]]:
    rows: list[ExecutionRow] = [
        ExecutionRow(
            phase="tier_budget",
            code=None,
            name=None,
            ips_stage=None,
            nuance=None,
            est_sell_shares=None,
            est_sell_mv=tier_reduce_mv,
            note="L1 组合层：按 Tier reduce_fraction 需释放的总股票市值",
        )
    ]
    cumulative = 0.0
    included_codes: set[str] = set()
    for action in ranked_actions:
        if action.suggest_reduce_pct <= 0:
            continue
        if tier_reduce_mv > 0 and cumulative >= tier_reduce_mv * T1_BUDGET_FILL_THRESHOLD:
            break
        cumulative += action.est_sell_mv
        included_codes.add(action.code)
        rows.append(
            ExecutionRow(
                phase="symbol",
                code=action.code,
                name=action.name,
                ips_stage=action.ips_stage,
                nuance=action.nuance,
                est_sell_shares=action.est_sell_shares,
                est_sell_mv=action.est_sell_mv,
                note=(
                    f"suggest↓={action.suggest_reduce_pct * 100:.0f}%, "
                    f"priority={action.priority_score:.4f}, "
                    f"reasons={'|'.join(action.reasons) or '—'}"
                ),
            )
        )
    deferred = tuple(
        action
        for action in ranked_actions
        if action.suggest_reduce_pct > 0 and action.code not in included_codes
    )
    return tuple(rows), cumulative, deferred


def _format_mv_wan(value: float) -> str:
    return f"{value / 10000:.1f} 万"


def build_summary_lines(
    *,
    decision: TierDecision,
    tier_reduce_mv: float,
    ranked_actions: tuple[SymbolAction, ...],
    keep_watch: tuple[SymbolAction, ...],
) -> tuple[str, ...]:
    target = (
        f"{decision.target_equity_exposure * 100:.0f}%"
        if decision.target_equity_exposure is not None
        else "n/a"
    )
    lines = [
        (
            f"Tier {decision.executed_tier} {decision.regime}：权益 "
            f"{decision.current_equity_exposure * 100:.0f}%→{target}，"
            f"需释放约 {_format_mv_wan(tier_reduce_mv)} MV"
        ),
    ]

    actionable = [a for a in ranked_actions if a.suggest_reduce_pct > 0]
    if actionable:
        top = actionable[:5]
        parts = [
            f"{a.code}({a.name}, {a.ips_stage}/{a.nuance}, "
            f"{a.suggest_reduce_pct * 100:.0f}%)"
            for a in top
        ]
        lines.append(f"优先减仓：{'、'.join(parts)}")

    if keep_watch:
        keep_parts = []
        for action in keep_watch[:5]:
            rs_s = (
                f"rs+{action.rs_20d_vs_bench * 100:.1f}%"
                if action.rs_20d_vs_bench is not None
                else "rs n/a"
            )
            keep_parts.append(f"{action.code}({action.name}, {rs_s})")
        lines.append(f"保留观察：{'、'.join(keep_parts)}")

    lines.append(
        "纪律：卖出决策基于趋势/相对强弱/回撤，成本价仅作 pnl7 风控参考，不作等回本依据"
    )
    return tuple(lines)


def build_action_plan(
    *,
    decision: TierDecision,
    portfolio: PortfolioSnapshot,
    early_warning: pd.DataFrame | None,
    trend_board: pd.DataFrame | None,
    pipeline_weights: dict[str, float] | None = None,
    config: DecisionPackConfig | None = None,
) -> ActionPlan:
    tier_reduce_mv = portfolio.stock_mv * float(decision.reduce_fraction)
    allocation_gaps = build_allocation_gaps(trend_board, pipeline_weights)
    holdings = _holding_lookup(portfolio)
    pipe = pipeline_weights or {}

    if trend_board is None or trend_board.empty:
        execution_rows, t1_cumulative, t1_deferred = build_execution_rows(
            tier_reduce_mv=tier_reduce_mv,
            ranked_actions=(),
        )
        return ActionPlan(
            tier_reduce_mv=tier_reduce_mv,
            ranked_actions=(),
            keep_watch=(),
            allocation_gaps=allocation_gaps,
            execution_rows=execution_rows,
            summary_lines=build_summary_lines(
                decision=decision,
                tier_reduce_mv=tier_reduce_mv,
                ranked_actions=(),
                keep_watch=(),
            ),
            t1_cumulative_mv=t1_cumulative,
            t1_deferred_actions=t1_deferred,
        )

    alert_actions: list[SymbolAction] = []
    keep_watch: list[SymbolAction] = []

    for _, row in trend_board.iterrows():
        weight = _float_or_none(row.get("weight")) or 0.0
        if weight <= 0:
            continue
        code = str(row["code"])
        holding = holdings.get(code)
        pw = _float_or_none(pipe.get(code))
        action = _symbol_action_from_row(
            row,
            pipeline_weight=pw,
            holding=holding,
            executed_tier=decision.executed_tier,
            config=config,
        )
        if action.suggest_reduce_pct <= 0:
            if action.alert_level == "green" and action.nuance == "intact":
                rs = action.rs_20d_vs_bench
                if rs is not None and rs >= KEEP_WATCH_RS_THRESHOLD:
                    keep_watch.append(action)
            continue
        if action.alert_level in {"red", "yellow"}:
            alert_actions.append(action)

    alert_actions.sort(key=lambda item: item.priority_score, reverse=True)
    keep_watch.sort(
        key=lambda item: item.rs_20d_vs_bench if item.rs_20d_vs_bench is not None else -999,
        reverse=True,
    )

    ranked = tuple(alert_actions)
    keep = tuple(keep_watch)
    execution_rows, t1_cumulative, t1_deferred = build_execution_rows(
        tier_reduce_mv=tier_reduce_mv,
        ranked_actions=ranked,
    )
    summary_lines = build_summary_lines(
        decision=decision,
        tier_reduce_mv=tier_reduce_mv,
        ranked_actions=ranked,
        keep_watch=keep,
    )

    return ActionPlan(
        tier_reduce_mv=tier_reduce_mv,
        ranked_actions=ranked,
        keep_watch=keep,
        allocation_gaps=allocation_gaps,
        execution_rows=execution_rows,
        summary_lines=summary_lines,
        t1_cumulative_mv=t1_cumulative,
        t1_deferred_actions=t1_deferred,
    )


__all__ = [
    "ActionPlan",
    "ExecutionRow",
    "SymbolAction",
    "build_action_plan",
    "build_allocation_gaps",
    "build_execution_rows",
    "build_summary_lines",
    "classify_ips_stage",
    "classify_nuance",
    "estimate_sell_shares",
    "priority_score",
]
