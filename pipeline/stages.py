"""Backtest + monitor stages."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .constants import DEFAULT_FUND_LIST_CSV, REPO_ROOT
from .daily_parameter_effect_audit import generate_daily_parameter_effect_audit
from .infra import _run_command
from .shadow_alert import generate_daily_shadow_alert


def run_backtest(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    # Prefer the L3 bridge (contains exposure_scale); fall back to L2 if not produced.
    bridge_input = paths["bridge_l3_out"] if paths["bridge_l3_out"].exists() else paths["bridge_out"]
    command = [
        sys.executable,
        "-m",
        "close_to_next_close_backtest.cli",
        "--input-csv",
        str(bridge_input),
        "--market-data-csv",
        str(paths["raw_panel_out"]),
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--init-cash",
        str(args.init_cash),
        "--provider-uri",
        args.provider_uri,
        "--benchmark",
        args.benchmark,
        "--rebalance-mode",
        args.rebalance_mode,
        "--rebalance-every-n",
        str(args.rebalance_every_n),
        "--rebalance-weekday",
        str(args.rebalance_weekday),
        "--rebalance-min-gap-days",
        str(args.rebalance_min_gap_days),
        "--rebalance-max-gap-days",
        str(args.rebalance_max_gap_days),
        "--rebalance-drift-threshold",
        str(args.rebalance_drift_threshold),
        "--rebalance-exposure-change-threshold",
        str(args.rebalance_exposure_change_threshold),
        "--rebalance-macro-budget-change-threshold",
        str(args.rebalance_macro_budget_change_threshold),
        "--rebalance-force-on-macro-state-flip" if args.rebalance_force_on_macro_state_flip else "--no-rebalance-force-on-macro-state-flip",
        "--rebalance-risk-change-threshold",
        str(args.rebalance_risk_change_threshold),
        "--rebalance-score-threshold",
        str(args.rebalance_score_threshold),
        "--rebalance-min-expected-return-gain",
        str(args.rebalance_min_expected_return_gain),
        "--rebalance-risk-aversion",
        str(args.rebalance_risk_aversion),
        "--rebalance-corr-bonus-weight",
        str(args.rebalance_corr_bonus_weight),
        "--rebalance-turnover-cost",
        str(args.rebalance_turnover_cost),
        "--rebalance-score-gate-mode",
        args.rebalance_score_gate_mode,
        "--weights-out",
        str(paths["weights_out"]),
        "--daily-nav-out",
        str(paths["daily_nav_out"]),
        "--backtest-with-name-out",
        str(paths["backtest_with_name_out"]),
        "--weights-full-out",
        str(paths["weights_full_out"]),
    ]
    _run_command(command, REPO_ROOT / "backtest", "close_to_next_close_backtest")


def run_monitor(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    # Auto-detect actual Route A labels (same logic as run_bridge)
    start_label = (getattr(args, "route_a_start_date", None) or args.start_date).replace("-", "")
    end_label = args.end_date.replace("-", "")
    if paths["route_a_output_dir"].exists():
        existing_params = list(paths["route_a_output_dir"].glob("route_a_params_*.csv"))
        if existing_params:
            parts = existing_params[0].stem.rsplit("_", 2)
            if len(parts) == 3:
                if parts[1] != start_label:
                    start_label = parts[1]
                if parts[2] != end_label:
                    end_label = parts[2]

    command = [
        sys.executable,
        str(REPO_ROOT / "strategy" / "route_ab_moments_bridge" / "run_mu_monitor_report.py"),
        "--route-a-output-dir",
        str(paths["route_a_output_dir"]),
        "--route-a-start-label",
        start_label,
        "--route-a-end-label",
        end_label,
        "--bridge-csv",
        str(paths["bridge_l3_out"] if paths["bridge_l3_out"].exists() else paths["bridge_out"]),
        "--daily-nav-csv",
        str(paths["daily_nav_out"]),
        "--output-dir",
        str(paths["monitor_out"]),
    ]
    # Auto-pass --pending-date when this run used --include-spot and end_date is
    # today. Otherwise today's row is a spot row with no next_close, the backtest
    # computes 0 P&L for it, and the monitor would report the day as 'healthy'.
    # Mark it as 'pending' instead so it is not confused with a confirmed result.
    if getattr(args, "include_spot", False):
        import pandas as pd
        today = pd.Timestamp.today().normalize()
        target = pd.Timestamp(args.end_date).normalize()
        if today == target:
            command.extend(["--pending-date", args.end_date])
    _run_command(command, REPO_ROOT, "monitor")


def run_shadow_alert(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    bridge_input = paths["bridge_l3_out"] if paths["bridge_l3_out"].exists() else paths["bridge_out"]
    payload = generate_daily_shadow_alert(
        l3_csv_path=bridge_input,
        output_json_path=paths["daily_shadow_alert_json"],
        history_csv_path=paths["daily_shadow_alert_history"],
        alert_date=args.end_date,
        fund_list_csv=getattr(args, "mean_overlay_factor_fund_list_csv", DEFAULT_FUND_LIST_CSV),
    )
    print(f"[INFO] Daily shadow alert: {paths['daily_shadow_alert_json']} status={payload.get('status')}")


def run_parameter_effect_audit(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    if getattr(args, "dry_run", False):
        print(f"[DRY-RUN] Skipping daily parameter effect audit -> {paths['parameter_effect_audit_dir']}")
        return
    bridge_input = paths["bridge_l3_out"] if paths["bridge_l3_out"].exists() else paths["bridge_out"]
    outputs = generate_daily_parameter_effect_audit(
        bridge_csv_path=bridge_input,
        output_dir=paths["parameter_effect_audit_dir"],
        cluster_mapping_path=Path(args.cluster_mapping_path).expanduser().resolve(),
        cluster_mapping_full_csv_path=(
            Path(args.cluster_mapping_full_csv).expanduser().resolve()
            if getattr(args, "cluster_mapping_full_csv", None)
            else None
        ),
        route_a_output_dir=paths["route_a_output_dir"],
        raw_panel_path=paths["raw_panel_out"],
        daily_nav_path=paths["daily_nav_out"],
        weights_path=paths["weights_out"],
        weights_full_path=paths["weights_full_out"],
        start_date=args.start_date,
        end_date=args.end_date,
        provider_uri=args.provider_uri,
        run_config=vars(args),
        audit_level=getattr(args, "audit_level", "simple"),
    )
    print(f"[INFO] Simple audit README: {outputs['readme_md']}")
    print(f"[INFO] Latest portfolio weights: {outputs['simple_latest_weights']}")
    print(f"[INFO] Daily decision summary: {outputs['simple_daily_summary']}")
    if getattr(args, "audit_level", "simple") == "full":
        print(f"[INFO] Full audit report: {outputs['report_md']}")
        print(f"[INFO] Full decision panel: {outputs['decision_panel']}")
        print(f"[INFO] Layer quality report: {outputs['layer_quality_report']}")
        print(f"[INFO] Layer quality panel: {outputs['layer_quality_panel']}")
