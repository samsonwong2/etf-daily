"""Orchestrator path helpers."""
from __future__ import annotations

import argparse
from pathlib import Path

from .constants import REPO_ROOT


def _suffix(start_date: str, end_date: str) -> str:
    return f"{start_date.replace('-', '')}_{end_date.replace('-', '')}"


def _default_route_a_output_dir(suffix: str) -> Path:
    return REPO_ROOT / "strategy" / "route_a_core" / f"outputs_route_a_quick_nll_fullpanel_{suffix}"


_PRESET_SLUG_ALIASES = {
    "simple_pulse_mu": "spm",
    "trend_mu_default": "trend",
    "legacy_raw_mu": "rawmu",
    "path_trend_default": "path",
    "mu_smooth_trend_default": "smooth",
    "posterior_mu1d_default": "pmu1d",
    "posterior_mu20_default": "pmu20",
    "posterior_mu1d_softshrink_default": "pmu1d_ss",
    "posterior_mu1d_softshrink_signgate": "pmu1d_sg",
    "posterior_mu1d_softshrink_stoploss": "pmu1d_sl",
}


def _preset_slug(preset: str) -> str:
    """Derive a short filename-safe slug from the mu preset name."""
    if preset in _PRESET_SLUG_ALIASES:
        return _PRESET_SLUG_ALIASES[preset]
    # Keep only [A-Za-z0-9_-] - paranoid guard, preset names are normally clean.
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in preset) or "preset"


def _resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    suffix = _suffix(args.start_date, args.end_date)
    route_a_start_date = getattr(args, "route_a_start_date", None) or args.start_date
    route_a_suffix = _suffix(route_a_start_date, args.end_date)
    temp_dir = Path(args.temp_dir).expanduser().resolve()
    route_a_output_dir = (
        Path(args.route_a_output_dir).expanduser().resolve()
        if args.route_a_output_dir
        else _default_route_a_output_dir(route_a_suffix)
    )
    slug = _preset_slug(args.mu_preset)
    monitor_dir = temp_dir / f"monitor_{slug}_{suffix}"
    audit_dir = temp_dir / f"audit_{slug}_{suffix}"
    return {
        "temp_dir": temp_dir,
        "codes_out": temp_dir / f"codes_{suffix}.txt",
        "raw_panel_out": temp_dir / f"raw_{suffix}.csv",
        "route_a_output_dir": route_a_output_dir,
        "bridge_out": temp_dir / f"bridge_{slug}_{suffix}.csv",
        "bridge_l3_out": temp_dir / f"bridge_{slug}_{suffix}_l3.csv",
        "weights_out": temp_dir / f"weights_{slug}_{suffix}.csv",
        "daily_nav_out": temp_dir / f"nav_{slug}_{suffix}.csv",
        "backtest_with_name_out": temp_dir / f"bt_{slug}_{suffix}.csv",
        "weights_full_out": temp_dir / f"weights_full_{slug}_{suffix}.csv",
        "monitor_out": monitor_dir,
        "monitor_csv": monitor_dir / "raw_mu_alert_daily_summary.csv",
        "daily_shadow_alert_json": temp_dir / "shadow_alert.json",
        "daily_shadow_alert_history": temp_dir / "shadow_alert_history.csv",
        "parameter_effect_audit_dir": audit_dir,
        "parameter_effect_simple_latest_weights_csv": audit_dir / "portfolio_latest.csv",
        "parameter_effect_simple_daily_summary_csv": audit_dir / "decision_daily.csv",
        "parameter_effect_simple_instrument_drilldown_csv": audit_dir / "decision_inst.csv",
        "parameter_effect_simple_manifest_json": audit_dir / "run_manifest.json",
        "parameter_effect_simple_readme_md": audit_dir / "README.md",
        "parameter_effect_instrument_daily_csv": audit_dir / "param_inst_daily.csv",
        "parameter_effect_decision_panel_csv": audit_dir / "decision_panel.csv",
        "parameter_effect_portfolio_weights_csv": audit_dir / "portfolio_daily.csv",
        "parameter_effect_latest_weights_csv": audit_dir / "latest_weights.csv",
        "parameter_effect_latest_weights_dated_csv": audit_dir / f"latest_weights_{args.end_date.replace('-', '')}.csv",
        "parameter_effect_rebalance_event_log_csv": audit_dir / "rebalance_log.csv",
        "parameter_effect_schema_json": audit_dir / "decision_schema.json",
        "parameter_effect_daily_summary_csv": audit_dir / "param_daily.csv",
        "parameter_effect_instrument_summary_csv": audit_dir / "param_inst.csv",
        "parameter_effect_layer_quality_panel_csv": audit_dir / "layer_panel.csv",
        "parameter_effect_layer_quality_daily_summary_csv": audit_dir / "layer_daily.csv",
        "parameter_effect_layer_quality_instrument_summary_csv": audit_dir / "layer_inst.csv",
        "parameter_effect_layer_quality_latest_pending_csv": audit_dir / "layer_latest_pending.csv",
        "parameter_effect_layer_quality_schema_json": audit_dir / "layer_schema.json",
        "parameter_effect_layer_quality_report_md": audit_dir / "layer_report.md",
        "parameter_effect_report_md": audit_dir / "param_report.md",
    }
