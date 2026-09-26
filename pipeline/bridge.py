"""bridge.py subprocess invocation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import constants
from .constants import REPO_ROOT
from .infra import _run_command


def run_bridge(
    args: argparse.Namespace,
    paths: dict[str, Path],
    monitor_csv: Path | None = None,
) -> None:
    stage_name = "bridge (with monitor overlay)" if monitor_csv is not None else "bridge"
    command = [
        sys.executable,
        str(REPO_ROOT / "strategy" / "route_ab_moments_bridge" / "bridge.py"),
        "--input-path",
        str(paths["raw_panel_out"]),
        "--output-path",
        str(paths["bridge_out"]),
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--mu-preset",
        args.mu_preset,
        "--route-b-model-type",
        args.route_b_model_type,
        "--route-a-train-window",
        str(args.route_a_train_window),
        "--route-b-train-window",
        str(args.route_b_train_window),
        "--route-a-retrain-every",
        str(args.route_a_retrain_every),
        "--route-b-retrain-every",
        str(args.route_b_retrain_every),
        "--route-a-mc-samples",
        str(args.route_a_mc_samples),
        "--route-a-optimizer-maxiter",
        str(args.route_a_optimizer_maxiter),
        "--route-a-optimizer-maxfun",
        str(args.route_a_optimizer_maxfun),
        "--route-a-optimizer-ftol",
        str(args.route_a_optimizer_ftol),
        "--route-a-training-objective",
        args.route_a_training_objective,
        "--route-max-workers",
        str(args.route_max_workers),
        "--provider-uri",
        str(args.provider_uri),
        "--variance-floor-mode",
        str(args.variance_floor_mode),
        "--variance-floor-quantile",
        str(args.variance_floor_quantile),
        "--corr-floor",
        str(args.corr_floor),
        "--mu-override",
        str(args.mu_override),
        "--mean-overlay-mode",
        str(args.mean_overlay_mode),
        "--mean-overlay-patch-level",
        str(args.mean_overlay_patch_level),
        "--mean-overlay-patch-name",
        str(args.mean_overlay_patch_name),
        "--mean-overlay-theme-filter",
        str(args.mean_overlay_theme_filter),
        "--mean-overlay-factor-fund-list-csv",
        str(args.mean_overlay_factor_fund_list_csv),
    ]
    variance_floor_dynamic = False
    if getattr(args, "variance_floor_quantile_risk_on", None) is not None:
        command.extend(["--variance-floor-quantile-risk-on", str(float(args.variance_floor_quantile_risk_on))])
        variance_floor_dynamic = True
    if getattr(args, "variance_floor_quantile_risk_off", None) is not None:
        command.extend(["--variance-floor-quantile-risk-off", str(float(args.variance_floor_quantile_risk_off))])
        variance_floor_dynamic = True
    if variance_floor_dynamic:
        command.extend(["--variance-floor-regime-ma", str(int(args.regime_gate_ma))])
        command.extend(["--variance-floor-regime-source", str(args.variance_floor_regime_source)])
        command.extend(["--variance-floor-opportunity-top-k", str(int(args.variance_floor_opportunity_top_k))])
        if getattr(args, "variance_floor_opportunity_threshold", None) is not None:
            command.extend([
                "--variance-floor-opportunity-threshold",
                str(float(args.variance_floor_opportunity_threshold)),
            ])
    if args.mu_shrink_alpha is not None:
        command.extend(["--mu-shrink-alpha", str(args.mu_shrink_alpha)])
    if args.mu_state_mode is not None:
        command.extend(["--mu-state-mode", str(args.mu_state_mode)])
    if args.mu_rank_buckets is not None:
        command.extend(["--mu-rank-buckets", str(args.mu_rank_buckets)])
    if args.mu_rank_active_buckets is not None:
        command.extend(["--mu-rank-active-buckets", str(args.mu_rank_active_buckets)])
    if args.route_a_config_path:
        command.extend(["--route-a-config-path", args.route_a_config_path])
    # T3-3/T3-4: explicitly pass hard weight cap + weight inertia alpha to
    # bridge subprocess so pipeline-level defaults govern (no silent drift
    # from bridge's internal defaults).
    if getattr(args, "max_weight_cap", None) is not None and float(args.max_weight_cap) > 0:
        command.extend(["--max-weight", str(float(args.max_weight_cap))])
    if getattr(args, "min_weight_floor", None) is not None and float(args.min_weight_floor) >= 0:
        command.extend(["--min-weight-floor", str(float(args.min_weight_floor))])
    if getattr(args, "weight_inertia_alpha", None) is not None:
        command.extend(["--weight-inertia-alpha", str(float(args.weight_inertia_alpha))])
    # T1-3: optional MZ calibration directory passthrough.
    if getattr(args, "route_b_calibration_dir", None):
        command.extend(["--route-b-calibration-dir", str(args.route_b_calibration_dir)])
    route_a_output_dir = paths["route_a_output_dir"]
    should_pass_route_a_dir = route_a_output_dir.exists() or (
        constants._DRY_RUN
        and getattr(args, "mode", None) == "reuse-route-a"
        and bool(getattr(args, "route_a_output_dir", None))
    )
    if should_pass_route_a_dir:
        command.extend(["--route-a-output-dir", str(route_a_output_dir)])
        route_a_start_label = (getattr(args, "route_a_start_date", None) or args.start_date).replace("-", "")
        if route_a_start_label != args.start_date.replace("-", ""):
            command.extend(["--route-a-label-start-date", route_a_start_label])
        # Auto-detect the actual Route A file end-label when the directory was produced
        # from a different (earlier) end date (e.g. --end-date 2026-04-22 but files say
        # _20260417). Pass --route-a-label-end-date so bridge finds the right files.
        expected_end_label = args.end_date.replace("-", "")
        existing_params = list(route_a_output_dir.glob("route_a_params_*.csv")) if route_a_output_dir.exists() else []
        if existing_params:
            sample_stem = existing_params[0].stem  # e.g. route_a_params_SH513030_20260101_20260417
            parts = sample_stem.rsplit("_", 2)  # ['route_a_params_SH513030', '20260101', '20260417']
            if len(parts) == 3:
                actual_end_label = parts[2]
                if actual_end_label != expected_end_label:
                    print(
                        f"[INFO] Route A files have end_label={actual_end_label}, "
                        f"bridge end_date label would be {expected_end_label}; "
                        f"passing --route-a-label-end-date {actual_end_label}"
                    )
                    command.extend(["--route-a-label-end-date", actual_end_label])
    if monitor_csv is not None:
        command.extend(["--monitor-csv", str(monitor_csv)])
        command.extend(["--monitor-alert-max-weight", str(args.monitor_alert_max_weight)])
        command.extend(["--monitor-watch-max-weight", str(args.monitor_watch_max_weight)])
    _run_command(command, REPO_ROOT, stage_name)
