"""Orchestrate decision pack generation."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from decision_pack.src.config import DecisionPackConfig, load_decision_pack_config
from decision_pack.src.inputs import (
    DEFAULT_STATE_PATH,
    RunInputs,
    build_inputs_manifest,
    resolve_run_inputs,
)
from decision_pack.src.market import build_market_snapshot, load_bridge_l3_row
from decision_pack.src.orders import append_recovery_buy_rows, build_display_only_orders, build_orders
from decision_pack.src.portfolio import PortfolioSnapshot, load_portfolio_snapshot
from decision_pack.src.recovery import (
    RecoveryUpdate,
    build_system_vs_manual,
    load_recovery_state,
    save_recovery_state,
    update_recovery_state,
)
from decision_pack.src.reconcile import reconcile_portfolios, render_reconcile_report
from decision_pack.src.action_plan import ActionPlan, build_action_plan
from decision_pack.src.early_warning import build_early_warning_frame
from decision_pack.src.pack_live_prices import (
    UNIVERSE_LIVE_CLOSE_CSV,
    build_universe_live_close_frame,
)
from decision_pack.src.render import write_pack_outputs
from decision_pack.src.symbol_trend_board import (
    build_cluster_trend_board,
    build_trend_board,
    load_cluster_mapping_codes,
)
from decision_pack.src.tier import MarketSnapshot, TierDecision, decide_tier
from runtime_paths import CLUSTER_MAPPING_SELECTED_TXT, FUND_LIST_CSV, WORKSPACE_DIR
from workspace.scripts.generate_daily_mu_position_report import load_fund_name_map, normalize_code


@dataclass(frozen=True)
class PackResult:
    output_dir: Path
    decision: TierDecision
    market: MarketSnapshot
    portfolio: PortfolioSnapshot
    recovery: RecoveryUpdate
    written: dict[str, Path]


def default_output_dir(report_date: str) -> Path:
    compact = report_date.replace("-", "")
    return WORKSPACE_DIR / "decision_packs" / compact


def generate_decision_pack(
    *,
    run_dir: Path | str,
    report_date: str,
    output_dir: Path | str | None = None,
    live_holdings: Path | str | None = None,
    cluster_mapping_path: Path | str | None = None,
    write_cluster_trend_board: bool = True,
    preferred_start: str = "20260401",
    ensure_audit: bool = True,
    dry_run: bool = False,
    config: DecisionPackConfig | None = None,
    market_snapshot: MarketSnapshot | None = None,
    state_path: Path | None = None,
) -> PackResult:
    cfg = config or load_decision_pack_config()
    inputs = resolve_run_inputs(
        run_dir,
        report_date,
        preferred_start=preferred_start,
        ensure_audit=ensure_audit,
    )
    out_dir = Path(output_dir) if output_dir is not None else default_output_dir(inputs.report_date)
    live_path = Path(live_holdings).expanduser().resolve() if live_holdings else None

    pipeline_portfolio = load_portfolio_snapshot(
        as_of=inputs.report_date,
        pipeline_weights_csv=inputs.paths.portfolio_weights_csv,
        fund_list_csv=FUND_LIST_CSV if FUND_LIST_CSV.exists() else None,
    )
    portfolio = (
        load_portfolio_snapshot(
            as_of=inputs.report_date,
            live_holdings_path=live_path,
            fund_list_csv=FUND_LIST_CSV if FUND_LIST_CSV.exists() else None,
            config=cfg,
        )
        if live_path is not None
        else pipeline_portfolio
    )

    market = market_snapshot or build_market_snapshot(
        inputs.report_date,
        daily_nav_csv=inputs.daily_nav_csv,
        config=cfg,
    )
    decision = decide_tier(market, current_equity_exposure=portfolio.equity_exposure, config=cfg)
    display_only = cfg.orders_mode == "display_only"
    cluster_trend_board = None
    cluster_vol_context = None
    early_warning = None
    action_plan: ActionPlan | None = None
    pipeline_weights: dict[str, float] | None = None
    if display_only:
        orders = build_display_only_orders(portfolio)
        trend_result = build_trend_board(portfolio, config=cfg)
        trend_board = trend_result.board
        trend_vol_context = trend_result.vol_context
        cluster_path = Path(cluster_mapping_path).expanduser() if cluster_mapping_path else CLUSTER_MAPPING_SELECTED_TXT
        if write_cluster_trend_board and cluster_path.exists():
            fund_list = FUND_LIST_CSV if FUND_LIST_CSV.exists() else None
            cluster_result = build_cluster_trend_board(
                portfolio,
                cluster_path,
                fund_list_csv=fund_list,
                config=cfg,
            )
            cluster_trend_board = cluster_result.board
            cluster_vol_context = cluster_result.vol_context
        pipeline_weights = {
            normalize_code(h.code): h.weight_total for h in pipeline_portfolio.holdings
        }
        early_warning = build_early_warning_frame(
            trend_board,
            cluster_trend_board,
            pipeline_weights=pipeline_weights,
            executed_tier=decision.executed_tier,
            config=cfg,
        )
        action_plan = build_action_plan(
            decision=decision,
            portfolio=portfolio,
            early_warning=early_warning,
            trend_board=trend_board,
            pipeline_weights=pipeline_weights,
            config=cfg,
        )
    else:
        orders = build_orders(portfolio, decision)
        trend_board = None
        trend_vol_context = None

    reconcile = None
    reconcile_md = None
    if live_path is not None:
        reconcile = reconcile_portfolios(portfolio, pipeline_portfolio, cfg)
        reconcile_md = render_reconcile_report(reconcile)

    bridge_row: dict[str, float | str | None] = {}
    if inputs.bridge_l3_csv is not None and inputs.bridge_l3_csv.exists():
        bridge_row = load_bridge_l3_row(inputs.bridge_l3_csv, inputs.report_date)

    prev_state = load_recovery_state(state_path or DEFAULT_STATE_PATH)
    recovery = update_recovery_state(
        prev_state,
        snapshot=market,
        decision=decision,
        exposure_scale=bridge_row.get("exposure_scale"),
        config=cfg,
    )
    if recovery.recovery_signal and not display_only:
        orders = append_recovery_buy_rows(
            orders,
            addback_fraction=recovery.addback_fraction,
            portfolio=portfolio,
            reason=recovery.reason,
        )

    system_vs_manual = build_system_vs_manual(
        report_date=inputs.report_date,
        decision=decision,
        exposure_scale=bridge_row.get("exposure_scale"),
        risk_macro_budget=bridge_row.get("risk_macro_budget"),
        recovery_state=recovery.state,
        config=cfg,
    )

    manifest = build_inputs_manifest(
        inputs,
        live_holdings=live_path,
        output_dir=out_dir,
        extra={"dry_run": dry_run},
    )

    universe_live_close = None
    cluster_path = Path(cluster_mapping_path).expanduser() if cluster_mapping_path else CLUSTER_MAPPING_SELECTED_TXT
    if cluster_path.exists():
        from decision_pack.scripts.generate_fat_tail_risk_report import build_universe_codes

        cluster_codes = load_cluster_mapping_codes(cluster_path)
        holdings_frame = portfolio.to_frame()
        universe = build_universe_codes(holdings_frame, cluster_codes, cfg.defensive_codes)
        name_map = load_fund_name_map(FUND_LIST_CSV) if FUND_LIST_CSV.exists() else {}
        for row in portfolio.holdings:
            if row.name:
                name_map[normalize_code(row.code)] = row.name
        portfolio_closes = {
            normalize_code(row.code): float(row.close_price)
            for row in portfolio.holdings
            if row.close_price is not None and row.close_price > 0
        }
        universe_live_close = build_universe_live_close_frame(
            as_of=inputs.report_date,
            universe_codes=universe,
            names=name_map,
            portfolio_closes=portfolio_closes,
            config=cfg,
        )

    written = write_pack_outputs(
        out_dir,
        decision=decision,
        market=market,
        portfolio=portfolio,
        orders=orders,
        reconcile=reconcile,
        reconcile_md=reconcile_md,
        recovery=recovery,
        system_vs_manual=system_vs_manual,
        inputs_manifest=manifest,
        orders_mode=cfg.orders_mode,
        trend_board=trend_board,
        trend_vol_context=trend_vol_context,
        cluster_trend_board=cluster_trend_board,
        cluster_vol_context=cluster_vol_context,
        early_warning=early_warning,
        action_plan=action_plan,
        universe_live_close=universe_live_close,
    )

    if not dry_run:
        save_recovery_state(recovery.state, state_path or DEFAULT_STATE_PATH)

    return PackResult(
        output_dir=out_dir,
        decision=decision,
        market=market,
        portfolio=portfolio,
        recovery=recovery,
        written=written,
    )


__all__ = [
    "PackResult",
    "default_output_dir",
    "generate_decision_pack",
]
