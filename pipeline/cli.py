"""Command-line entry point; glues all pipeline stages together."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

from . import constants
from .constants import (
    DEFAULT_CLUSTER_MAPPING,
    DEFAULT_FUND_LIST_CSV,
    DEFAULT_INIT_CASH,
    DEFAULT_MU_PRESET,
    DEFAULT_PROVIDER_URI,
    DEFAULT_TEMP_DIR,
    PRODUCTION_FORECAST_MU_METHOD,
)
from .paths import _preset_slug, _resolve_paths
from .infra import _pipeline_lock
from runtime_paths import (
    CLUSTER_MAPPING_SELECTED_TXT,
    FUND_LIST_CSV,
    LOG_DIR,
    PIPELINE_RUN_LOG_DIR,
    PROJECT_ROOT,
    QLIB_PROVIDER_URI,
    TEMP_DIR,
    WORKSPACE_DIR,
    WORKSPACE_ROOT,
)
from .data_prep import (
    generate_code_list,
    generate_raw_panel,
    append_spot_to_raw_panel,
)
from .route_a import run_route_a_full, run_route_a_rolling_refresh
from .bridge import run_bridge
from .mean_override import normalize_mean_override_args, prepare_mean_override_route_a
from .forecast_mu import (
    FORECAST_MU_METHOD_CHOICES,
    FORECAST_MU_MODEL_CHOICES,
    normalize_forecast_mu_args,
    prepare_forecast_mu_route_a,
)
from .risk_controls import apply_risk_controls
from .stages import run_backtest, run_monitor, run_shadow_alert, run_parameter_effect_audit


RUN_PROFILE_DEBUG = "debug_full_controls"
RUN_PROFILE_BALANCED = "simple_pulse_adaptive_balanced_v1"
RUN_PROFILE_BALANCED_V2 = "simple_pulse_adaptive_balanced_v2"
RUN_PROFILE_DEFENSIVE = "simple_pulse_adaptive_defensive_v1"
RUN_PROFILE_CHOICES = (
    RUN_PROFILE_DEBUG,
    RUN_PROFILE_BALANCED,
    RUN_PROFILE_BALANCED_V2,
    RUN_PROFILE_DEFENSIVE,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "production_regime_switch_ewma_shrink.json"


RUN_PROFILE_CONFIGS: dict[str, dict[str, object]] = {
    RUN_PROFILE_DEBUG: {},
    RUN_PROFILE_BALANCED: {
        "forecast_mu_method": "ewma60",
        "mu_preset": "simple_pulse_mu",
        "mean_overlay_mode": "shadow",
        "max_weight_cap": 0.25,
        "min_weight_floor": 0.05,
        "regime_gate_min_scale": 0.50,
        "variance_floor_mode": "cross_sectional",
        "variance_floor_quantile": 0.25,
        "corr_floor": 0.10,
        "rebalance_mode": "adaptive",
        "rebalance_min_gap_days": 10,
        "rebalance_max_gap_days": 18,
        "rebalance_drift_threshold": 1.00,
        "rebalance_exposure_change_threshold": 0.10,
        "rebalance_risk_change_threshold": 0.10,
        "rebalance_score_threshold": 0.0002,
        "rebalance_min_expected_return_gain": 0.0,
        "rebalance_risk_aversion": 1.0,
        "rebalance_corr_bonus_weight": 0.0002,
        "rebalance_turnover_cost": 0.0001,
        "rebalance_score_gate_mode": "off",
        "vol_target": 0.015,
        "vol_target_risk_on": 0.022,
        "vol_target_risk_off": 0.0,
        "weight_inertia_alpha": 0.30,
        "audit_level": "simple",
    },
    RUN_PROFILE_BALANCED_V2: {
        "forecast_mu_method": "ewma60",
        "mu_preset": "simple_pulse_mu",
        "mean_overlay_mode": "shadow",
        "max_weight_cap": 0.25,
        "min_weight_floor": 0.05,
        "regime_gate_mode": "budget_v2",
        "regime_gate_min_scale": 0.50,
        "macro_budget_min": 0.50,
        "macro_budget_trend_fast_ma": 20,
        "macro_budget_trend_slow_ma": 60,
        "macro_budget_vol_window": 20,
        "macro_budget_vol_lookback": 120,
        "variance_floor_mode": "cross_sectional",
        "variance_floor_quantile": 0.25,
        "corr_floor": 0.10,
        "rebalance_mode": "adaptive",
        "rebalance_min_gap_days": 10,
        "rebalance_max_gap_days": 18,
        "rebalance_drift_threshold": 1.00,
        "rebalance_exposure_change_threshold": 0.05,
        "rebalance_macro_budget_change_threshold": 0.15,
        "rebalance_force_on_macro_state_flip": True,
        "rebalance_risk_change_threshold": 0.10,
        "rebalance_score_threshold": 0.0002,
        "rebalance_min_expected_return_gain": 0.0,
        "rebalance_risk_aversion": 1.0,
        "rebalance_corr_bonus_weight": 0.0002,
        "rebalance_turnover_cost": 0.0001,
        "rebalance_score_gate_mode": "off",
        "vol_target": 0.015,
        "vol_target_risk_on": 0.022,
        "vol_target_risk_off": 0.012,
        "vol_target_offensive": 0.022,
        "vol_target_neutral": 0.015,
        "vol_target_defensive": 0.012,
        "cluster_gate_source": "realized",
        "cluster_gate_rho": 0.5,
        "cluster_gate_rho_saturation": 0.85,
        "cluster_gate_min_scale": 0.4,
        "cluster_gate_realized_lookback": 10,
        "signal_quality_gate_mode": "rolling_ic",
        "signal_quality_ic_threshold": -0.05,
        "signal_quality_min_scale": 0.5,
        "trend_sleeve_exposure_floor": 0.0,
        "weight_inertia_alpha": 0.30,
        "audit_level": "simple",
    },
    RUN_PROFILE_DEFENSIVE: {
        "forecast_mu_method": "ewma60",
        "mu_preset": "simple_pulse_mu",
        "mean_overlay_mode": "shadow",
        "max_weight_cap": 0.25,
        "min_weight_floor": 0.05,
        "regime_gate_min_scale": 0.50,
        "variance_floor_mode": "cross_sectional",
        "variance_floor_quantile": 0.05,
        "corr_floor": 0.10,
        "rebalance_mode": "adaptive",
        "rebalance_min_gap_days": 10,
        "rebalance_max_gap_days": 18,
        "rebalance_drift_threshold": 1.00,
        "rebalance_exposure_change_threshold": 0.10,
        "rebalance_risk_change_threshold": 0.10,
        "rebalance_score_threshold": 0.0002,
        "rebalance_min_expected_return_gain": 0.0,
        "rebalance_risk_aversion": 1.0,
        "rebalance_corr_bonus_weight": 0.0002,
        "rebalance_turnover_cost": 0.0001,
        "rebalance_score_gate_mode": "off",
        "vol_target": 0.015,
        "vol_target_risk_on": 0.022,
        "vol_target_risk_off": 0.0,
        "weight_inertia_alpha": 0.30,
        "audit_level": "simple",
    },
}


def _config_context(start_date: str | None, end_date: str | None) -> dict[str, str]:
    start_date = start_date or ""
    end_date = end_date or ""
    return {
        "project_root": str(PROJECT_ROOT),
        "workspace_root": str(WORKSPACE_ROOT),
        "temp_dir": str(TEMP_DIR),
        "workspace_dir": str(WORKSPACE_DIR),
        "log_dir": str(LOG_DIR),
        "provider_uri": str(QLIB_PROVIDER_URI),
        "cluster_mapping_path": str(CLUSTER_MAPPING_SELECTED_TXT),
        "fund_list_csv": str(FUND_LIST_CSV),
        "start_date": start_date,
        "end_date": end_date,
        "start_yyyymmdd": start_date.replace("-", ""),
        "end_yyyymmdd": end_date.replace("-", ""),
    }


def _format_config_value(value: object, context: dict[str, str]) -> object:
    if isinstance(value, str):
        return value.format(**context)
    if isinstance(value, list):
        return [_format_config_value(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _format_config_value(item, context) for key, item in value.items()}
    return value


def _load_pipeline_config(config_path: str | Path, start_date: str | None, end_date: str | None) -> dict[str, object]:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = payload.get("pipeline_args", payload)
    if not isinstance(config, dict):
        raise ValueError(f"Pipeline config must be a JSON object: {path}")
    context = _config_context(start_date, end_date)
    return {str(key): _format_config_value(value, context) for key, value in config.items()}


def _option_for_action(action: argparse.Action) -> str:
    long_options = [option for option in action.option_strings if option.startswith("--")]
    if long_options:
        return long_options[0]
    return f"--{action.dest.replace('_', '-')}"


def _config_dict_to_argv(parser: argparse.ArgumentParser, config: dict[str, object]) -> list[str]:
    action_by_dest = {action.dest: action for action in parser._actions}
    expanded: list[str] = []
    unknown_keys = sorted(set(config) - set(action_by_dest))
    if unknown_keys:
        raise ValueError(f"Unknown config option(s): {', '.join(unknown_keys)}")
    for dest, value in config.items():
        action = action_by_dest[dest]
        option = _option_for_action(action)
        if isinstance(action, argparse._StoreTrueAction):
            if bool(value):
                expanded.append(option)
            continue
        if isinstance(action, argparse._StoreFalseAction):
            if not bool(value):
                expanded.append(option)
            continue
        if isinstance(action, argparse.BooleanOptionalAction):
            expanded.append(option if bool(value) else f"--no-{dest.replace('_', '-')}")
            continue
        if isinstance(value, list):
            expanded.append(option)
            expanded.extend(str(item) for item in value)
            continue
        if value is None:
            continue
        expanded.extend([option, str(value)])
    return expanded


def _expand_config_argv(parser: argparse.ArgumentParser, argv: list[str]) -> list[str]:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_parser.add_argument("--start-date", default=None)
    pre_parser.add_argument("--end-date", default=None)
    known, _ = pre_parser.parse_known_args(argv)
    if not known.config:
        return argv
    defaults = {action.dest: action.default for action in parser._actions}
    start_date = known.start_date or defaults.get("start_date")
    end_date = known.end_date or defaults.get("end_date")
    config = _load_pipeline_config(known.config, start_date, end_date)
    return _config_dict_to_argv(parser, config) + argv


def _collect_explicit_option_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest
    explicit_dests: set[str] = set()
    for token in argv:
        option = str(token).split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest:
            explicit_dests.add(dest)
    return explicit_dests


def _apply_run_profile(args: argparse.Namespace, explicit_dests: set[str]) -> None:
    profile = str(getattr(args, "run_profile", RUN_PROFILE_DEBUG) or RUN_PROFILE_DEBUG)
    config = RUN_PROFILE_CONFIGS.get(profile)
    if config is None:
        raise ValueError(f"Unknown run_profile: {profile}")
    applied: dict[str, object] = {}
    overrides: dict[str, object] = {}
    for dest, value in config.items():
        if dest in explicit_dests:
            overrides[dest] = getattr(args, dest, None)
            continue
        if hasattr(args, dest):
            setattr(args, dest, value)
            applied[dest] = value
    if getattr(args, "audit_level", None) is None:
        args.audit_level = "full" if profile == RUN_PROFILE_DEBUG else "simple"
        applied["audit_level"] = args.audit_level
    args.run_profile_applied = applied
    args.run_profile_overrides = overrides


def _risk_control_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "vol_target": args.vol_target,
        "vol_target_risk_on": args.vol_target_risk_on,
        "vol_target_risk_off": args.vol_target_risk_off,
        "vol_target_offensive": args.vol_target_offensive,
        "vol_target_neutral": args.vol_target_neutral,
        "vol_target_defensive": args.vol_target_defensive,
        "regime_gate_min_scale": args.regime_gate_min_scale,
        "regime_gate_ma": args.regime_gate_ma,
        "regime_gate_mode": args.regime_gate_mode,
        "macro_budget_min": args.macro_budget_min,
        "macro_budget_trend_fast_ma": args.macro_budget_trend_fast_ma,
        "macro_budget_trend_slow_ma": args.macro_budget_trend_slow_ma,
        "macro_budget_vol_window": args.macro_budget_vol_window,
        "macro_budget_vol_lookback": args.macro_budget_vol_lookback,
        "max_weight_cap": args.max_weight_cap,
        "min_holdings_audit_warn": args.min_holdings_audit_warn,
        "vol_target_cov_mode": args.vol_target_cov_mode,
        "cluster_gate_rho": args.cluster_gate_rho,
        "cluster_gate_rho_saturation": args.cluster_gate_rho_saturation,
        "cluster_gate_min_scale": args.cluster_gate_min_scale,
        "cluster_gate_source": args.cluster_gate_source,
        "cluster_gate_realized_lookback": args.cluster_gate_realized_lookback,
        "cluster_gate_realized_min_periods": args.cluster_gate_realized_min_periods,
        "signal_quality_gate_mode": args.signal_quality_gate_mode,
        "signal_quality_ic_window": args.signal_quality_ic_window,
        "signal_quality_ic_min_periods": args.signal_quality_ic_min_periods,
        "signal_quality_ic_threshold": args.signal_quality_ic_threshold,
        "signal_quality_min_scale": args.signal_quality_min_scale,
        "signal_quality_min_scale_risk_on": args.signal_quality_min_scale_risk_on,
        "signal_quality_min_scale_risk_off": args.signal_quality_min_scale_risk_off,
        "trend_sleeve_weight_total": args.trend_sleeve_weight_total,
        "trend_sleeve_weight_total_risk_on": args.trend_sleeve_weight_total_risk_on,
        "trend_sleeve_weight_total_risk_off": args.trend_sleeve_weight_total_risk_off,
        "trend_sleeve_top_k": args.trend_sleeve_top_k,
        "trend_sleeve_top_k_risk_on": args.trend_sleeve_top_k_risk_on,
        "trend_sleeve_top_k_risk_off": args.trend_sleeve_top_k_risk_off,
        "trend_sleeve_momentum_windows": args.trend_sleeve_momentum_windows,
        "trend_sleeve_mu_weight": args.trend_sleeve_mu_weight,
        "trend_sleeve_min_score": args.trend_sleeve_min_score,
        "trend_sleeve_warmup_days": args.trend_sleeve_warmup_days,
        "trend_sleeve_exposure_floor": args.trend_sleeve_exposure_floor,
        "trend_sleeve_exposure_floor_risk_on": args.trend_sleeve_exposure_floor_risk_on,
        "trend_sleeve_exposure_floor_risk_off": args.trend_sleeve_exposure_floor_risk_off,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-shot simple-pulse mean pipeline: code list -> raw panel -> Route A(optional) -> bridge -> backtest -> monitor"
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "JSON config with pipeline_args. Relative paths are resolved from the project root. "
            f"Production preset: {DEFAULT_CONFIG_PATH.relative_to(PROJECT_ROOT)}"
        ),
    )
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-04-13")
    parser.add_argument(
        "--run-profile",
        default=RUN_PROFILE_BALANCED,
        choices=list(RUN_PROFILE_CHOICES),
        help=(
            "High-level profile that fills the noisy production knobs. "
            "Use simple_pulse_adaptive_balanced_v1 for the default fuzzy-correct path; "
            "simple_pulse_adaptive_balanced_v2 enables budget_v2 L3 macro risk budget; "
            "simple_pulse_adaptive_defensive_v1 keeps the q05 variance floor; "
            "debug_full_controls leaves low-level defaults exposed. Explicit CLI flags override profile values."
        ),
    )
    parser.add_argument(
        "--audit-level",
        default=None,
        choices=["simple", "full"],
        help="simple writes a compact daily decision audit; full also writes the engineering-level long tables.",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "reuse-route-a", "rolling-refresh"],
        default="reuse-route-a",
        help=(
            "full: rerun Route A from scratch; "
            "reuse-route-a: reuse existing Route A output dir for the current window; "
            "rolling-refresh: reuse the most recent prior Route A dir with the same "
            "start_label and only short-window-retrain the trailing segment per code "
            "(falls back to full if no prior dir is found)."
        ),
    )
    parser.add_argument("--provider-uri", default=DEFAULT_PROVIDER_URI)
    parser.add_argument("--cluster-mapping-path", default=DEFAULT_CLUSTER_MAPPING)
    parser.add_argument(
        "--cluster-mapping-full-csv",
        default=None,
        help="Optional full cluster_mapping.csv baseline used by the full audit to compare selected vs unselected names.",
    )
    parser.add_argument("--temp-dir", default=DEFAULT_TEMP_DIR)
    parser.add_argument("--mu-preset", default=DEFAULT_MU_PRESET)
    parser.add_argument("--route-a-output-dir", default=None)
    parser.add_argument(
        "--forecast-mu-method",
        default=PRODUCTION_FORECAST_MU_METHOD,
        choices=list(FORECAST_MU_METHOD_CHOICES),
        help=(
            "Production forecast mean source. Default ewma60 generates a causal "
            "EWMA60 Route-A override with shrink_alpha locked at 0.75; route_a "
            "keeps the legacy explicit Route-A/full/reuse workflow."
        ),
    )
    parser.add_argument(
        "--forecast-mu-model",
        default="ewma",
        choices=list(FORECAST_MU_MODEL_CHOICES),
        help=(
            "Causal forecast-mu model used when --forecast-mu-method ewma60 generates "
            "Route-A override files. Default ewma preserves the existing EWMA60 path; "
            "regime_ic_weighted_ensemble and regime_switch_ewma_shrink are trend-aware candidates."
        ),
    )
    parser.add_argument(
        "--forecast-mu-start-date",
        default=None,
        help="Label/start date for production forecast-mu files. Defaults to --route-a-start-date or --start-date.",
    )
    parser.add_argument(
        "--forecast-mu-output-dir",
        default=None,
        help="Directory for generated production forecast-mu Route-A-style files. Defaults under --temp-dir.",
    )
    parser.add_argument(
        "--forecast-mu-history-start-date",
        default=None,
        help="Optional first qlib close date loaded for production EWMA60 history.",
    )
    parser.add_argument(
        "--forecast-mu-window",
        type=int,
        default=252,
        help="Rolling window used by rolling/multi-window forecast-mu models (default: 252).",
    )
    parser.add_argument(
        "--forecast-mu-ewma-span",
        type=int,
        default=60,
        help="EWMA span used by ewma/regime forecast-mu models (default: 60).",
    )
    parser.add_argument(
        "--forecast-mu-multi-windows",
        default="20,60,120,252",
        help="Comma-separated windows for multi-window and ensemble forecast-mu models.",
    )
    parser.add_argument(
        "--forecast-mu-multi-window-weights",
        default="",
        help="Optional comma-separated weights matching --forecast-mu-multi-windows.",
    )
    parser.add_argument(
        "--forecast-mu-momentum-window",
        type=int,
        default=60,
        help="Lookback window for momentum forecast-mu models (default: 60).",
    )
    parser.add_argument(
        "--forecast-mu-vol-window",
        type=int,
        default=60,
        help="Volatility window for vol-adjusted momentum forecast-mu models (default: 60).",
    )
    parser.add_argument(
        "--forecast-mu-ensemble-ewma-spans",
        default="20,60,120,252",
        help="Comma-separated EWMA spans included in IC-weighted forecast-mu ensembles.",
    )
    parser.add_argument(
        "--forecast-mu-ensemble-momentum-windows",
        default="20,60,120",
        help="Comma-separated momentum windows included in IC-weighted forecast-mu ensembles.",
    )
    parser.add_argument(
        "--forecast-mu-ensemble-vol-adjusted-windows",
        default="60,120",
        help="Comma-separated vol-adjusted momentum windows included in IC-weighted forecast-mu ensembles.",
    )
    parser.add_argument(
        "--forecast-mu-ic-lookback",
        type=int,
        default=60,
        help="Rolling IC lookback for IC-weighted forecast-mu ensembles (default: 60).",
    )
    parser.add_argument(
        "--forecast-mu-ic-min-periods",
        type=int,
        default=20,
        help="Minimum IC observations before ensemble weights become active (default: 20).",
    )
    parser.add_argument(
        "--forecast-mu-regime-benchmark",
        default="SH510300",
        help="Benchmark instrument used by regime-conditioned forecast-mu models (default: SH510300).",
    )
    parser.add_argument(
        "--forecast-mu-regime-fast-ma",
        type=int,
        default=20,
        help="Fast benchmark moving average for regime-conditioned forecast-mu models.",
    )
    parser.add_argument(
        "--forecast-mu-regime-slow-ma",
        type=int,
        default=60,
        help="Slow benchmark moving average for regime-conditioned forecast-mu models.",
    )
    parser.add_argument(
        "--forecast-mu-regime-vol-window",
        type=int,
        default=20,
        help="Benchmark volatility window for regime-conditioned forecast-mu models.",
    )
    parser.add_argument(
        "--forecast-mu-regime-vol-lookback",
        type=int,
        default=120,
        help="Lookback used to define high-volatility regimes for forecast-mu models.",
    )
    parser.add_argument(
        "--forecast-mu-regime-risk-off-shrink-alpha",
        type=float,
        default=0.50,
        help="Risk-off shrink alpha used by regime_switch_ewma_shrink forecast-mu model.",
    )
    parser.add_argument(
        "--forecast-mu-scale",
        type=float,
        default=1.0,
        help="Multiply generated forecast-mu values before writing Route-A override files.",
    )
    parser.add_argument(
        "--forecast-mu-shrink-target",
        default="zero",
        choices=["none", "zero", "cross_sectional_median"],
        help="Shrinkage target for generated forecast-mu values (default: zero).",
    )
    parser.add_argument(
        "--forecast-mu-shrink-alpha",
        type=float,
        default=0.75,
        help="Reliability alpha for generated forecast-mu shrinkage (default: 0.75).",
    )
    parser.add_argument(
        "--forecast-mu-winsor-quantile",
        type=float,
        default=None,
        help="Optional per-date two-sided winsor quantile for generated forecast-mu values, e.g. 0.01.",
    )
    parser.add_argument(
        "--forecast-mu-min-periods",
        type=int,
        default=20,
        help="Minimum historical observations required by generated forecast-mu models (default: 20).",
    )
    parser.add_argument(
        "--forecast-mu-asof-lag",
        type=int,
        default=1,
        help="Trading-day lag for generated forecast-mu models; 1 uses returns through prior trading day.",
    )
    parser.add_argument(
        "--forecast-mu-missing-fill",
        default="zero",
        choices=["none", "zero"],
        help="Fallback for missing generated forecast-mu values (default: zero).",
    )
    parser.add_argument(
        "--forecast-mu-initial-annualized-csv",
        default=None,
        help="Optional fallback prior CSV for early EWMA60 gaps; normally unnecessary with enough qlib history.",
    )
    parser.add_argument(
        "--forecast-mu-no-initial-fallback",
        action="store_true",
        help="Do not fill missing early production EWMA60 means from --forecast-mu-initial-annualized-csv.",
    )
    parser.add_argument(
        "--forecast-mu-overwrite",
        action="store_true",
        help="Rebuild the generated production forecast-mu directory if it already exists.",
    )
    parser.add_argument(
        "--forecast-mu-allow-incomplete",
        action="store_true",
        help="Allow incomplete production forecast-mu coverage. Default is strict 100%% coverage.",
    )
    parser.add_argument(
        "--route-a-start-date",
        default=None,
        help=(
            "Optional earlier Route A training start date. Trading/backtest outputs still use "
            "--start-date; only Route A files and downstream Route A label lookup use this date."
        ),
    )
    parser.add_argument("--route-a-config-path", default=None)
    parser.add_argument(
        "--mean-override-csv",
        default=None,
        help=(
            "External/full-coverage expected-mean CSV with instrument plus "
            "mean_daily_return or annualized_return. When supplied, the pipeline "
            "generates Route-A-style override files and consumes them through "
            "simple_pulse_mu instead of rerunning Route A."
        ),
    )
    parser.add_argument(
        "--mean-override-start-date",
        default=None,
        help="Label/start date for the mean override files. Defaults to --route-a-start-date or --start-date.",
    )
    parser.add_argument(
        "--mean-override-end-date",
        default=None,
        help="Label/end date for the mean override files. Defaults to --end-date.",
    )
    parser.add_argument(
        "--mean-override-output-dir",
        default=None,
        help="Directory for generated Route-A-style mean override files. Defaults under --temp-dir.",
    )
    parser.add_argument(
        "--mean-override-mu-source",
        default="mean_daily_return",
        choices=["mean_daily_return", "annualized_arithmetic", "annualized_geometric"],
        help="How to derive daily mu from --mean-override-csv (default mean_daily_return).",
    )
    parser.add_argument(
        "--mean-override-mu-scale",
        type=float,
        default=1.0,
        help="Multiply the derived daily mean before writing Route-A override files.",
    )
    parser.add_argument(
        "--mean-override-winsor-quantile",
        type=float,
        default=None,
        help="Optional two-sided cross-sectional winsor quantile for external means, e.g. 0.01.",
    )
    parser.add_argument(
        "--mean-override-date-source",
        default="qlib",
        choices=["qlib", "business"],
        help="Use qlib close-date coverage or business dates for override forecast rows.",
    )
    parser.add_argument(
        "--mean-override-overwrite",
        action="store_true",
        help="Rebuild the generated mean override directory if it already exists.",
    )
    parser.add_argument(
        "--mean-override-allow-incomplete",
        action="store_true",
        help="Allow missing instruments/dates in generated mean override files. Default is strict 100%% coverage.",
    )
    parser.add_argument("--route-a-train-window", type=int, default=37)
    parser.add_argument("--route-b-train-window", type=int, default=37)
    parser.add_argument("--route-a-retrain-every", type=int, default=20)
    parser.add_argument("--route-b-retrain-every", type=int, default=20)
    parser.add_argument("--route-a-mc-samples", type=int, default=48)
    parser.add_argument("--route-a-optimizer-maxiter", type=int, default=450)
    parser.add_argument("--route-a-optimizer-maxfun", type=int, default=9000)
    parser.add_argument("--route-a-optimizer-ftol", type=float, default=1e-6)
    parser.add_argument("--route-a-training-objective", default="nll-only")
    parser.add_argument(
        "--route-max-workers",
        type=int,
        default=1,
        help=(
            "Maximum parallel workers for Route A full retraining and bridge forecast assembly. "
            "Default 1 preserves deterministic serial execution; use 4 or 6 for faster full reruns."
        ),
    )
    parser.add_argument("--benchmark", default="sh510300")
    parser.add_argument("--init-cash", type=float, default=DEFAULT_INIT_CASH)
    parser.add_argument(
        "--rebalance-mode",
        default="daily",
        choices=["daily", "every_n", "weekly", "adaptive"],
        help=(
            "Backtest rebalance schedule. 'daily' preserves the historical pipeline behavior; "
            "'every_n' rebalances every --rebalance-every-n trading days; 'weekly' rebalances "
            "on --rebalance-weekday; 'adaptive' uses low-frequency guardrails plus risk/weight "
            "drift triggers. Non-rebalance days carry forward existing holdings."
        ),
    )
    parser.add_argument(
        "--rebalance-every-n",
        type=int,
        default=10,
        help="Trading-day interval for --rebalance-mode every_n (default: 10).",
    )
    parser.add_argument(
        "--rebalance-weekday",
        type=int,
        default=0,
        help="Weekday for --rebalance-mode weekly; 0=Monday ... 6=Sunday (default: 0).",
    )
    parser.add_argument(
        "--rebalance-min-gap-days",
        type=int,
        default=10,
        help="Minimum trading-day gap between adaptive early rebalances (default: 10).",
    )
    parser.add_argument(
        "--rebalance-max-gap-days",
        type=int,
        default=18,
        help="Maximum trading-day gap before adaptive forces a scheduled rebalance (default: 18).",
    )
    parser.add_argument(
        "--rebalance-drift-threshold",
        type=float,
        default=1.00,
        help="Adaptive trigger threshold for L1 drift between current holdings and target effective weights.",
    )
    parser.add_argument(
        "--rebalance-exposure-change-threshold",
        type=float,
        default=0.10,
        help="Adaptive trigger threshold for target exposure changes since the last rebalance.",
    )
    parser.add_argument(
        "--rebalance-macro-budget-change-threshold",
        type=float,
        default=0.15,
        help="Adaptive trigger threshold for macro risk budget changes since the last rebalance.",
    )
    parser.add_argument(
        "--rebalance-force-on-macro-state-flip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When enabled, offensive/caution -> defensive/stress macro state flips bypass min_gap.",
    )
    parser.add_argument(
        "--rebalance-risk-change-threshold",
        type=float,
        default=0.10,
        help="Adaptive trigger threshold for L3 risk scale changes since the last rebalance.",
    )
    parser.add_argument(
        "--rebalance-score-threshold",
        type=float,
        default=0.0002,
        help="Adaptive net score threshold for score-triggered rebalances and optional score gate (default: 0.0002).",
    )
    parser.add_argument(
        "--rebalance-min-expected-return-gain",
        type=float,
        default=0.0,
        help="Minimum forecast daily expected-return gain required by adaptive score gate (default: 0).",
    )
    parser.add_argument(
        "--rebalance-risk-aversion",
        type=float,
        default=1.0,
        help="Adaptive score penalty/bonus weight for forecast portfolio variance change (default: 1).",
    )
    parser.add_argument(
        "--rebalance-corr-bonus-weight",
        type=float,
        default=0.0002,
        help="Adaptive score reward per unit decrease in weighted average forecast correlation (default: 0.0002).",
    )
    parser.add_argument(
        "--rebalance-turnover-cost",
        type=float,
        default=0.0001,
        help="Adaptive score transaction-cost penalty per unit turnover (default: 0.0001).",
    )
    parser.add_argument(
        "--rebalance-score-gate-mode",
        default="off",
        choices=["off", "soft", "hard"],
        help="Adaptive score gate mode: off only uses score as a trigger; soft lets max_gap override failed score; hard requires score pass even at max_gap.",
    )
    parser.add_argument(
        "--keep-route-a-output-dir",
        action="store_true",
        help="In full mode, keep existing Route A output dir contents instead of deleting first",
    )
    parser.add_argument(
        "--lock-wait",
        action="store_true",
        help="Block waiting for pipeline lock if another run holds it (default: fail fast)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print every subprocess stage command (Route A, bridge, backtest, monitor) "
            "without executing. Orchestrator Python code (path resolution, code list, "
            "raw panel, risk controls) still runs so resolved paths are surfaced. "
            "Intended for plan inspection and code review."
        ),
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help=(
            "Directory to tee per-stage subprocess stdout/stderr into "
            f"stage_<name>.log files. Default: {PIPELINE_RUN_LOG_DIR}/"
            "<preset>_<suffix>_<timestamp>. Pass 'none' to disable tee and keep "
            "only the orchestrator's stdout (legacy behavior)."
        ),
    )
    # garch11_norm + monitor overlay
    parser.add_argument(
        "--route-b-model-type",
        default="garch11_norm",
        choices=["ewma", "rolling_logvar", "egarch11_norm", "garch11_norm", "gjr11_t"],
        help="Route B variance model (default: garch11_norm; use ewma to restore old baseline)",
    )
    parser.add_argument(
        "--monitor-overlay",
        action="store_true",
        help="Enable monitor-linked max_weight overlay: runs bridge twice (once to build monitor, once with overlay applied)",
    )
    parser.add_argument(
        "--monitor-alert-max-weight",
        type=float,
        default=0.20,
        help="Per-name weight cap when overall_status==alert (default 0.20)",
    )
    parser.add_argument(
        "--monitor-watch-max-weight",
        type=float,
        default=0.25,
        help="Per-name weight cap when overall_status==watch (default 0.25)",
    )
    parser.add_argument(
        "--include-spot",
        action="store_true",
        help="Append today's realtime spot data (AkShare) to raw panel after generation. Only applies when end_date == today.",
    )
    parser.add_argument(
        "--spot-csv",
        default=str(TEMP_DIR / "fund_etf_category_sina_df.csv"),
        help=f"Temporary path for AkShare spot data CSV (default: {TEMP_DIR / 'fund_etf_category_sina_df.csv'})",
    )
    # ── Risk control post-processing ─────────────────────────────────────────
    parser.add_argument(
        "--vol-target",
        type=float,
        default=0.015,
        help=(
            "Daily portfolio volatility target (std dev). 0 = disabled. "
            "Default 0.015 = 1.5%% daily vol target (~24%% annualised), the Pareto "
            "knee observed on 2026-01-01 → 2026-04-22 (Sharpe 1.80, MaxDD -9.17%%). "
            "vt in [0.008, 0.015] keeps Sharpe ~1.80; >=0.020 breaks Sharpe. "
            "Uses Route B per-instrument variance predictions via exposure_scale."
        ),
    )
    parser.add_argument(
        "--vol-target-risk-on",
        type=float,
        default=0.022,
        help=(
            "Optional daily vol target when HS300 is at/above its MA. "
            "0 = use --vol-target. Mainline benchmark default 0.022 is locked "
            "from the 2026-04-30 fixed audit."
        ),
    )
    parser.add_argument(
        "--vol-target-risk-off",
        type=float,
        default=0.0,
        help=(
            "Optional daily vol target when HS300 is below its MA. "
            "0 = use --vol-target. Typical defensive value: 0.014."
        ),
    )
    parser.add_argument(
        "--regime-gate-min-scale",
        type=float,
        default=0.5,
        help=(
            "Minimum weight scale factor when HS300 is below its moving average. "
            "1.0 = disabled. Default 0.5 = reduce exposure by up to 50%% when benchmark "
            "is at its moving-average floor. Scale is interpolated continuously."
        ),
    )
    parser.add_argument(
        "--regime-gate-ma",
        type=int,
        default=20,
        help="Number of trading days for the HS300 moving-average used in the regime gate (default: 20).",
    )
    parser.add_argument(
        "--regime-gate-mode",
        default="binary",
        choices=["binary", "budget_v2"],
        help=(
            "L3 regime gate mode. binary keeps the legacy HS300/MA20 switch; "
            "budget_v2 uses trend+vol macro risk budget."
        ),
    )
    parser.add_argument(
        "--macro-budget-min",
        type=float,
        default=0.50,
        help="Minimum macro risk budget for budget_v2 mode (default: 0.50).",
    )
    parser.add_argument(
        "--macro-budget-trend-fast-ma",
        type=int,
        default=20,
        help="Fast MA window for budget_v2 benchmark trend (default: 20).",
    )
    parser.add_argument(
        "--macro-budget-trend-slow-ma",
        type=int,
        default=60,
        help="Slow MA window for budget_v2 benchmark trend (default: 60).",
    )
    parser.add_argument(
        "--macro-budget-vol-window",
        type=int,
        default=20,
        help="Benchmark vol window for budget_v2 mode (default: 20).",
    )
    parser.add_argument(
        "--macro-budget-vol-lookback",
        type=int,
        default=120,
        help="Benchmark vol reference lookback for budget_v2 mode (default: 120).",
    )
    parser.add_argument(
        "--vol-target-offensive",
        type=float,
        default=0.0,
        help="Daily vol target when macro budget >= 0.85 in budget_v2 mode.",
    )
    parser.add_argument(
        "--vol-target-neutral",
        type=float,
        default=0.0,
        help="Daily vol target when 0.50 <= macro budget < 0.85 in budget_v2 mode.",
    )
    parser.add_argument(
        "--vol-target-defensive",
        type=float,
        default=0.0,
        help="Daily vol target when macro budget < 0.50 in budget_v2 mode.",
    )
    # ── Hard weight constraints (layer 2) ─────────────────────────────────────
    parser.add_argument(
        "--max-weight-cap",
        type=float,
        default=0.15,
        help=(
            "Hard per-asset weight cap applied as a post-processing projection "
            "onto the capped simplex. 0 = disabled. Default 0.15 caps each ETF "
            "at 15%% of the daily total weight (total weight preserved)."
        ),
    )
    parser.add_argument(
        "--min-weight-floor",
        type=float,
        default=0.05,
        help=(
            "Hard lower bound for nonzero bridge weights before L3 risk scaling. "
            "Passed through to route_ab_moments_bridge/bridge.py. Default 0.05."
        ),
    )
    parser.add_argument(
        "--min-holdings-audit-warn",
        "--min-holdings-effective",  # DEPRECATED alias, kept for back-compat
        dest="min_holdings_audit_warn",
        type=float,
        default=6.0,
        help=(
            "Audit-only threshold for the minimum effective number of holdings "
            "(≈ 1/HHI where HHI = sum(w_i^2)). 0 = disabled. Default 6 prints a "
            "warning when effective holdings drop below 6 on any day. Does NOT "
            "modify weights. The legacy flag --min-holdings-effective is still "
            "accepted but deprecated; the new name reflects the soft-warn nature."
        ),
    )
    # ── Variance floor (layer 1: fixes Route B variance input) ────────────────
    parser.add_argument(
        "--variance-floor-mode",
        type=str,
        default="cross_sectional",
        choices=["none", "cross_sectional", "percentile"],
        help=(
            "Phase 2 variance-floor applied inside bridge.py BEFORE MVO. "
            "'none' disables; 'cross_sectional' (default) clips each day's forecast_variance "
            "to the q-quantile of that day's cross-section; 'percentile' clips each "
            "instrument's forecast_variance to the q-quantile of its own rolling-20d "
            "realized variance over the last 252 trading days."
        ),
    )
    parser.add_argument(
        "--variance-floor-quantile",
        type=float,
        default=0.60,
        help="Quantile used by --variance-floor-mode (default 0.60).",
    )
    parser.add_argument(
        "--variance-floor-quantile-risk-on",
        type=float,
        default=None,
        help=(
            "Optional variance-floor quantile when HS300 is at/above --regime-gate-ma. "
            "Omit to use --variance-floor-quantile."
        ),
    )
    parser.add_argument(
        "--variance-floor-quantile-risk-off",
        type=float,
        default=None,
        help=(
            "Optional variance-floor quantile when HS300 is below --regime-gate-ma. "
            "Omit to use --variance-floor-quantile."
        ),
    )
    parser.add_argument(
        "--variance-floor-regime-source",
        type=str,
        default="benchmark_ma",
        choices=["benchmark_ma", "opportunity_strength"],
        help=(
            "Source used by risk-on/off variance-floor quantile overrides. "
            "benchmark_ma uses HS300 vs --regime-gate-ma; opportunity_strength "
            "uses the daily top-K forecast mean strength before MVO."
        ),
    )
    parser.add_argument(
        "--variance-floor-opportunity-top-k",
        type=int,
        default=5,
        help="Top-K forecast means averaged by --variance-floor-regime-source opportunity_strength.",
    )
    parser.add_argument(
        "--variance-floor-opportunity-threshold",
        type=float,
        default=None,
        help="Risk-on threshold for top-K forecast mean strength when --variance-floor-regime-source opportunity_strength.",
    )
    parser.add_argument(
        "--corr-floor",
        type=float,
        default=0.2,
        help=(
            "Phase 3: correlation floor inside bridge.py. Non-diagonal correlation "
            "entries below this value (post-shrinkage) are clipped up. Tier2 整改 "
            "2026-04-24 (T2-3)：0.30 → 0.40；oracle bottleneck grid 后降至 0.20。"
            "与 DEFAULT_COV_CORR_SHRINK=0.50 配合。0 disables."
        ),
    )
    parser.add_argument(
        "--mu-override",
        type=str,
        default="none",
        choices=["none", "flip", "zero"],
        help=(
            "Phase 5 diagnostic: override Route A mu. 'none' (default) uses as-is; "
            "'flip' multiplies mu by -1 (test if Route A is systematically reversed); "
            "'zero' forces mu=0 so MVO degenerates to pure min-variance."
        ),
    )
    parser.add_argument(
        "--weight-inertia-alpha",
        type=float,
        default=0.30,
        help=(
            "Tier3 (T3-4) 权重惯性：w_t = alpha * w_{t-1} + (1-alpha) * w_new。"
            "默认 0.30 — 降日换手、稳定 MVO 在含噪 μ 下的结果。0 = 禁用。"
        ),
    )
    parser.add_argument(
        "--route-b-calibration-dir",
        type=str,
        default=None,
        help=(
            "T1-3: directory containing route_b_variance_eval_daily_calibrated_*.csv "
            "produced by `route_b_variance_multi_instrument/variance_calibrator.py`. "
            "When set, bridge overlays Mincer-Zarnowitz rolling-calibrated "
            "variance_pred_cal on top of inline Route B variance (by instrument + "
            "forecast_date). Raw values preserved in route_b_variance_raw."
        ),
    )
    parser.add_argument(
        "--mu-shrink-alpha",
        type=float,
        default=None,
        help=(
            "Phase 5: override preset's mu_shrink_alpha (signal *= alpha before MVO). "
            "Lower values (e.g. 0.1~0.2) make MVO more variance-driven. "
            "When omitted, inherits from --mu-preset."
        ),
    )
    parser.add_argument(
        "--mu-state-mode",
        type=str,
        default=None,
        choices=["raw", "sign", "threshold_gate", "rank_bucket"],
        help=(
            "Phase 5: post-smoothing transform of optimizer mu input. "
            "'rank_bucket' with --mu-rank-active-buckets='5' approximates top-quintile gating."
        ),
    )
    parser.add_argument(
        "--mu-rank-buckets",
        type=int,
        default=None,
        help="Phase 5: number of cross-sectional buckets when --mu-state-mode=rank_bucket.",
    )
    parser.add_argument(
        "--mu-rank-active-buckets",
        type=str,
        default=None,
        help=(
            "Phase 5: comma-separated 1-indexed bucket ids kept non-zero, e.g. '5' for top-20%% "
            "when --mu-rank-buckets=5. Ignored unless --mu-state-mode=rank_bucket."
        ),
    )
    parser.add_argument(
        "--mean-overlay-mode",
        type=str,
        default="shadow",
        choices=["off", "shadow", "enabled"],
        help=(
            "Route A industry-momentum mean overlay. Production default 'shadow' logs "
            "the hypothetical patch without changing MVO; 'enabled' feeds the patched "
            "optimizer mean into bridge MVO."
        ),
    )
    parser.add_argument(
        "--mean-overlay-patch-level",
        type=float,
        default=0.015,
        help="Optimizer-mean floor for V2_T3_MIN_EFFECTIVE_PATCH (default 0.015).",
    )
    parser.add_argument(
        "--mean-overlay-patch-name",
        type=str,
        default="V2_T3_MIN_EFFECTIVE_PATCH",
        help="Audit label for the Route A mean overlay patch.",
    )
    parser.add_argument(
        "--mean-overlay-theme-filter",
        type=str,
        default="communication,ai,semiconductor,star_semiconductor",
        help=(
            "Comma-separated factor_industry_theme values eligible for the Route A "
            "mean overlay. Empty string disables the theme gate."
        ),
    )
    parser.add_argument(
        "--mean-overlay-factor-fund-list-csv",
        type=str,
        default=DEFAULT_FUND_LIST_CSV,
        help="fund_list CSV used to derive industry theme before bridge MVO for the mean overlay.",
    )
    # ── Phase 3 Path A: cluster-gate (layer-3 correlation-based de-risking) ──
    parser.add_argument(
        "--cluster-gate-rho",
        type=float,
        default=0.0,
        help=(
            "Phase 3 Path A: threshold on the weighted-average pairwise correlation "
            "of active holdings (w>0). When the portfolio's weighted-avg ρ̄ exceeds "
            "this value, scale all weights down linearly toward --cluster-gate-min-scale "
            "at ρ̄=--cluster-gate-rho-saturation. 0 disables (default). Typical: 0.6."
        ),
    )
    parser.add_argument(
        "--cluster-gate-rho-saturation",
        type=float,
        default=0.9,
        help=(
            "Phase 3 Path A: ρ̄ at which cluster-gate fully engages (reaches --cluster-gate-min-scale). "
            "Default 0.9. Must be > --cluster-gate-rho."
        ),
    )
    parser.add_argument(
        "--cluster-gate-min-scale",
        type=float,
        default=0.5,
        help=(
            "Phase 3 Path A: minimum scale applied when ρ̄ ≥ saturation. "
            "0.5 halves weights at full engagement. 1.0 disables."
        ),
    )
    parser.add_argument(
        "--cluster-gate-source",
        type=str,
        default="forecast_cov",
        choices=["forecast_cov", "realized"],
        help=(
            "Source for cluster-gate ρ̄. 'forecast_cov' uses bridge forecast_cov_matrix; "
            "'realized' uses trailing realized returns from the raw panel."
        ),
    )
    parser.add_argument(
        "--cluster-gate-realized-lookback",
        type=int,
        default=10,
        help="Trailing trading days used when --cluster-gate-source=realized (default 10).",
    )
    parser.add_argument(
        "--cluster-gate-realized-min-periods",
        type=int,
        default=5,
        help="Minimum pairwise observations for realized cluster correlations (default 5).",
    )
    parser.add_argument(
        "--signal-quality-gate-mode",
        type=str,
        default="none",
        choices=["none", "rolling_ic"],
        help=(
            "Optional L3 signal-quality gate. 'rolling_ic' scales exposure when the "
            "lagged rolling cross-sectional IC of forecast_mean_return is weak."
        ),
    )
    parser.add_argument(
        "--signal-quality-ic-window",
        type=int,
        default=5,
        help="Rolling window for --signal-quality-gate-mode=rolling_ic (default 5).",
    )
    parser.add_argument(
        "--signal-quality-ic-min-periods",
        type=int,
        default=3,
        help="Minimum historical IC observations for signal-quality gate (default 3).",
    )
    parser.add_argument(
        "--signal-quality-ic-threshold",
        type=float,
        default=-0.05,
        help="Trigger threshold for lagged rolling IC; below this value applies min-scale.",
    )
    parser.add_argument(
        "--signal-quality-min-scale",
        type=float,
        default=0.5,
        help="Multiplicative exposure scale when signal-quality gate triggers (default 0.5).",
    )
    parser.add_argument(
        "--signal-quality-min-scale-risk-on",
        type=float,
        default=1.0,
        help=(
            "Optional signal-quality gate scale when HS300 is at/above MA. "
            "0 = use --signal-quality-min-scale. Mainline benchmark default 1.00 "
            "is locked from the 2026-04-30 fixed audit."
        ),
    )
    parser.add_argument(
        "--signal-quality-min-scale-risk-off",
        type=float,
        default=0.0,
        help=(
            "Optional signal-quality gate scale when HS300 is below MA. "
            "0 = use --signal-quality-min-scale. Typical defensive value: 0.50."
        ),
    )
    # ── Vol-targeting covariance mode (layer 3 upgrade) ──────────────────────
    parser.add_argument(
        "--vol-target-cov-mode",
        type=str,
        default="full",
        choices=["diagonal", "full"],
        help=(
            "Covariance form used by vol-targeting in apply_risk_controls. "
            "'diagonal' uses Σwᵢ²σᵢ² (legacy, ignores correlations); "
            "'full' (default) uses w'Σw from bridge's forecast_cov_matrix/order, "
            "properly accounting for correlation-driven risk."
        ),
    )
    # ── Trend sleeve (pre-cap raw-weight overlay) ───────────────────────────
    parser.add_argument(
        "--trend-sleeve-weight-total",
        type=float,
        default=0.0,
        help=(
            "Reserve this fraction of each day's raw weight for top model+momentum "
            "trend names before cap/vol-target. 0 disables. Typical: 0.15~0.25."
        ),
    )
    parser.add_argument(
        "--trend-sleeve-weight-total-risk-on",
        type=float,
        default=-1.0,
        help=(
            "Optional trend sleeve budget when HS300 is at/above MA. "
            "Negative = use --trend-sleeve-weight-total; 0 explicitly disables in risk-on."
        ),
    )
    parser.add_argument(
        "--trend-sleeve-weight-total-risk-off",
        type=float,
        default=-1.0,
        help=(
            "Optional trend sleeve budget when HS300 is below MA. "
            "Negative = use --trend-sleeve-weight-total; 0 explicitly disables in risk-off."
        ),
    )
    parser.add_argument(
        "--trend-sleeve-top-k",
        type=int,
        default=4,
        help="Number of top trend names to receive the reserved sleeve (default: 4).",
    )
    parser.add_argument(
        "--trend-sleeve-top-k-risk-on",
        type=int,
        default=0,
        help="Optional top-K for the trend sleeve in risk-on. 0 = use --trend-sleeve-top-k.",
    )
    parser.add_argument(
        "--trend-sleeve-top-k-risk-off",
        type=int,
        default=0,
        help="Optional top-K for the trend sleeve in risk-off. 0 = use --trend-sleeve-top-k.",
    )
    parser.add_argument(
        "--trend-sleeve-momentum-windows",
        nargs="+",
        type=int,
        default=[3, 5, 10],
        help="Causal trailing close-return windows for trend sleeve scoring (default: 3 5 10).",
    )
    parser.add_argument(
        "--trend-sleeve-mu-weight",
        type=float,
        default=0.65,
        help=(
            "Weight on robust-z Route A forecast_mean_return in trend score; "
            "remaining weight goes to robust-z momentum average (default: 0.65)."
        ),
    )
    parser.add_argument(
        "--trend-sleeve-min-score",
        type=float,
        default=0.0,
        help="Minimum model+momentum score required for sleeve selection (default: 0.0).",
    )
    parser.add_argument(
        "--trend-sleeve-warmup-days",
        type=int,
        default=0,
        help=(
            "Extra business-day close history loaded from qlib before start-date for "
            "trend sleeve momentum. 0 keeps raw-panel-only behavior."
        ),
    )
    parser.add_argument(
        "--trend-sleeve-exposure-floor",
        type=float,
        default=0.0,
        help=(
            "Independent exposure-scale floor for the sleeve-add portion of selected "
            "trend names. 0 disables."
        ),
    )
    parser.add_argument(
        "--trend-sleeve-exposure-floor-risk-on",
        type=float,
        default=-1.0,
        help=(
            "Optional sleeve exposure floor when HS300 is at/above MA. "
            "Negative = use --trend-sleeve-exposure-floor; 0 explicitly disables."
        ),
    )
    parser.add_argument(
        "--trend-sleeve-exposure-floor-risk-off",
        type=float,
        default=-1.0,
        help=(
            "Optional sleeve exposure floor when HS300 is below MA. "
            "Negative = use --trend-sleeve-exposure-floor; 0 explicitly disables."
        ),
    )
    # ── Phase 4: Anchor ETF overlay (independent hedge reserve) ──────────────
    parser.add_argument(
        "--anchor-codes",
        nargs="*",
        default=[],
        help=(
            "Phase 4 anchor ETFs (e.g. SH518880 SH515180 for gold + dividend). "
            "Added to raw panel for backtest but excluded from Route A training. "
            "In apply_risk_controls they are injected with a fixed weight reserve "
            "before cap/vol-target, providing a low/negative-correlated hedge "
            "against sustained drawdowns. Empty list disables."
        ),
    )
    parser.add_argument(
        "--anchor-weight-total",
        type=float,
        default=0.0,
        help=(
            "Phase 4 total weight reserved for anchor ETFs (split equally among "
            "available anchors). Original MVO weights are scaled by "
            "(1 - anchor_weight_total) before anchor rows are inserted. "
            "Typical: 0.15 (15%%). 0 disables."
        ),
    )
    return parser


def validate_scalar_args(args: argparse.Namespace) -> None:
    if getattr(args, "audit_level", None) not in {"simple", "full"}:
        raise ValueError(f"audit_level must be one of simple/full, got {getattr(args, 'audit_level', None)}")
    if int(args.rebalance_every_n) < 1:
        raise ValueError(f"rebalance_every_n must be >= 1, got {args.rebalance_every_n}")
    if not 0 <= int(args.rebalance_weekday) <= 6:
        raise ValueError(f"rebalance_weekday must be in [0, 6], got {args.rebalance_weekday}")
    if int(args.rebalance_min_gap_days) < 1:
        raise ValueError(f"rebalance_min_gap_days must be >= 1, got {args.rebalance_min_gap_days}")
    if int(args.rebalance_max_gap_days) < 1:
        raise ValueError(f"rebalance_max_gap_days must be >= 1, got {args.rebalance_max_gap_days}")
    if int(args.rebalance_min_gap_days) > int(args.rebalance_max_gap_days):
        raise ValueError(
            "rebalance_min_gap_days must be <= rebalance_max_gap_days, "
            f"got {args.rebalance_min_gap_days} > {args.rebalance_max_gap_days}"
        )
    for name in [
        "rebalance_drift_threshold",
        "rebalance_exposure_change_threshold",
        "rebalance_macro_budget_change_threshold",
        "rebalance_risk_change_threshold",
        "rebalance_risk_aversion",
        "rebalance_corr_bonus_weight",
        "rebalance_turnover_cost",
    ]:
        if float(getattr(args, name)) < 0:
            raise ValueError(f"{name} must be >= 0, got {getattr(args, name)}")
    if args.rebalance_score_gate_mode not in {"off", "soft", "hard"}:
        raise ValueError(
            "rebalance_score_gate_mode must be one of off, soft, hard, "
            f"got {args.rebalance_score_gate_mode}"
        )
    for name in [
        "forecast_mu_window",
        "forecast_mu_ewma_span",
        "forecast_mu_momentum_window",
        "forecast_mu_vol_window",
        "forecast_mu_ic_lookback",
        "forecast_mu_ic_min_periods",
        "forecast_mu_regime_fast_ma",
        "forecast_mu_regime_slow_ma",
        "forecast_mu_regime_vol_window",
        "forecast_mu_regime_vol_lookback",
        "forecast_mu_min_periods",
    ]:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be >= 1, got {getattr(args, name)}")
    if int(args.forecast_mu_min_periods) > int(args.forecast_mu_window):
        raise ValueError(
            "forecast_mu_min_periods must be <= forecast_mu_window, "
            f"got {args.forecast_mu_min_periods} > {args.forecast_mu_window}"
        )
    if int(args.forecast_mu_asof_lag) < 0:
        raise ValueError(f"forecast_mu_asof_lag must be >= 0, got {args.forecast_mu_asof_lag}")
    for name in ["forecast_mu_shrink_alpha", "forecast_mu_regime_risk_off_shrink_alpha"]:
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if float(args.forecast_mu_scale) < 0:
        raise ValueError(f"forecast_mu_scale must be >= 0, got {args.forecast_mu_scale}")
    if args.forecast_mu_winsor_quantile is not None:
        value = float(args.forecast_mu_winsor_quantile)
        if not 0.0 < value < 0.5:
            raise ValueError(f"forecast_mu_winsor_quantile must be in (0, 0.5), got {value}")
    if float(args.max_weight_cap) < 0:
        raise ValueError(f"max_weight_cap must be >= 0, got {args.max_weight_cap}")
    if float(args.min_weight_floor) < 0:
        raise ValueError(f"min_weight_floor must be >= 0, got {args.min_weight_floor}")
    if float(args.max_weight_cap) > 0 and float(args.min_weight_floor) > float(args.max_weight_cap):
        raise ValueError(
            "min_weight_floor must be <= max_weight_cap when max_weight_cap is enabled, "
            f"got min_weight_floor={args.min_weight_floor} > max_weight_cap={args.max_weight_cap}"
        )
    if not 0.0 <= float(args.regime_gate_min_scale) <= 1.0:
        raise ValueError(f"regime_gate_min_scale must be in [0, 1], got {args.regime_gate_min_scale}")
    if not 0.0 <= float(args.macro_budget_min) <= 1.0:
        raise ValueError(f"macro_budget_min must be in [0, 1], got {args.macro_budget_min}")
    if args.regime_gate_mode not in {"binary", "budget_v2"}:
        raise ValueError(f"regime_gate_mode must be binary or budget_v2, got {args.regime_gate_mode}")
    for name in [
        "macro_budget_trend_fast_ma",
        "macro_budget_trend_slow_ma",
        "macro_budget_vol_window",
        "macro_budget_vol_lookback",
    ]:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be >= 1, got {getattr(args, name)}")
    for name in [
        "variance_floor_quantile",
        "variance_floor_quantile_risk_on",
        "variance_floor_quantile_risk_off",
    ]:
        value = getattr(args, name, None)
        if value is not None and not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if int(args.variance_floor_opportunity_top_k) < 1:
        raise ValueError(
            "variance_floor_opportunity_top_k must be >= 1, "
            f"got {args.variance_floor_opportunity_top_k}"
        )
    variance_floor_dynamic = (
        args.variance_floor_quantile_risk_on is not None
        or args.variance_floor_quantile_risk_off is not None
    )
    if (
        variance_floor_dynamic
        and args.variance_floor_regime_source == "opportunity_strength"
        and args.variance_floor_opportunity_threshold is None
    ):
        raise ValueError(
            "variance_floor_opportunity_threshold is required when "
            "--variance-floor-regime-source opportunity_strength is used with risk-on/off quantiles"
        )


def _should_check_route_a_reuse_dir_early(args: argparse.Namespace) -> bool:
    """Return True when Route-A reuse dir is built before raw panel (mean override)."""
    if getattr(args, "mean_override_csv", None):
        return True
    if getattr(args, "forecast_mu_method", "route_a") != "route_a":
        return False
    return True


def _validate_route_a_reuse_dir(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    if args.mode != "reuse-route-a" or paths["route_a_output_dir"].exists():
        return
    if constants._DRY_RUN:
        print(
            f"[DRY-RUN] Note: Route A output dir for reuse mode does not exist: "
            f"{paths['route_a_output_dir']} (would fail in real run)"
        )
        return
    raise FileNotFoundError(
        f"Route A output dir not found for reuse mode: {paths['route_a_output_dir']}"
    )


def validate_args(
    args: argparse.Namespace,
    paths: dict[str, Path],
    *,
    check_route_a_reuse_dir: bool = True,
) -> None:
    validate_scalar_args(args)
    cluster_mapping_path = Path(args.cluster_mapping_path).expanduser().resolve()
    if not cluster_mapping_path.exists():
        raise FileNotFoundError(f"cluster_mapping file not found: {cluster_mapping_path}")
    if getattr(args, "cluster_mapping_full_csv", None):
        cluster_mapping_full_csv = Path(args.cluster_mapping_full_csv).expanduser().resolve()
        if not cluster_mapping_full_csv.exists():
            raise FileNotFoundError(f"cluster_mapping full CSV not found: {cluster_mapping_full_csv}")
    if args.route_a_start_date:
        route_a_start = _dt.date.fromisoformat(args.route_a_start_date)
        trading_start = _dt.date.fromisoformat(args.start_date)
        end_date = _dt.date.fromisoformat(args.end_date)
        if route_a_start > trading_start:
            raise ValueError(
                f"route_a_start_date must be <= start_date, got {args.route_a_start_date} > {args.start_date}"
            )
        if route_a_start > end_date:
            raise ValueError(
                f"route_a_start_date must be <= end_date, got {args.route_a_start_date} > {args.end_date}"
            )
    if check_route_a_reuse_dir:
        _validate_route_a_reuse_dir(args, paths)


def main(argv: list[str] | None = None) -> int:
    # dry-run + log-dir state lives on pipeline.constants module
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    expanded_argv = _expand_config_argv(parser, raw_argv)
    explicit_dests = _collect_explicit_option_dests(parser, expanded_argv)
    args = parser.parse_args(expanded_argv)
    _apply_run_profile(args, explicit_dests)
    normalize_mean_override_args(args)
    normalize_forecast_mu_args(args)
    constants._DRY_RUN = bool(getattr(args, "dry_run", False))
    if constants._DRY_RUN:
        print("[DRY-RUN] Subprocess stages will be printed but NOT executed.")
    print(
        f"[INFO] Run profile: {args.run_profile} "
        f"(applied={len(getattr(args, 'run_profile_applied', {}))}, "
        f"overrides={len(getattr(args, 'run_profile_overrides', {}))})"
    )
    paths = _resolve_paths(args)
    start_label = args.start_date.replace("-", "")
    end_label = args.end_date.replace("-", "")
    route_a_start_date = args.route_a_start_date or args.start_date
    # Structured logging: unless explicitly disabled (--log-dir none) or in
    # dry-run mode, tee every stage's stdout/stderr into per-stage log files
    # so crashes/output can be inspected after the fact without scrolling
    # through interleaved terminal output.
    log_dir_arg = getattr(args, "log_dir", None)
    if constants._DRY_RUN or (log_dir_arg is not None and str(log_dir_arg).lower() == "none"):
        constants._LOG_DIR = None
    else:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = _preset_slug(args.mu_preset)
        if log_dir_arg:
            constants._LOG_DIR = Path(log_dir_arg).expanduser().resolve()
        else:
            constants._LOG_DIR = PIPELINE_RUN_LOG_DIR / f"{slug}_{start_label}_{end_label}_{ts}"
        constants._LOG_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Per-stage logs will be written to: {constants._LOG_DIR}")
    lock_path = (
        Path(args.temp_dir).expanduser().resolve()
        / f".pipeline_{args.mu_preset}_{start_label}_{end_label}.lock"
    )
    try:
        lock_ctx = _pipeline_lock(lock_path, wait=bool(getattr(args, "lock_wait", False)))
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    try:
        lock_ctx.__enter__()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print(f"[INFO] Acquired pipeline lock: {lock_path}")
    try:
        validate_scalar_args(args)
        prepare_mean_override_route_a(args, paths)
        validate_args(args, paths, check_route_a_reuse_dir=_should_check_route_a_reuse_dir_early(args))
        codes = generate_code_list(
            Path(args.cluster_mapping_path).expanduser().resolve(),
            paths["codes_out"],
        )
        # Phase 4: anchor codes are loaded into raw_panel (so backtest can price them)
        # but are NOT included in codes passed to Route A / bridge. They are injected
        # in apply_risk_controls as an independent hedge reserve.
        anchor_codes_requested = [str(c).upper() for c in (args.anchor_codes or [])]
        anchor_codes_requested = [c for c in anchor_codes_requested if c]
        codes_for_panel = list(codes) + [c for c in anchor_codes_requested if c not in set(codes)]
        available_codes_all = generate_raw_panel(
            provider_uri=args.provider_uri,
            codes=codes_for_panel,
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=paths["raw_panel_out"],
        )
        available_set = set(available_codes_all)
        available_anchor_codes = [c for c in anchor_codes_requested if c in available_set]
        if anchor_codes_requested:
            missing_anchors = [c for c in anchor_codes_requested if c not in available_set]
            print(
                f"[INFO] Phase 4 anchor overlay: {len(available_anchor_codes)}/{len(anchor_codes_requested)} "
                f"anchors available ({', '.join(available_anchor_codes)})"
                + (f"; missing: {', '.join(missing_anchors)}" if missing_anchors else "")
            )
        # available_codes for Route A / bridge excludes anchors.
        available_codes = [c for c in available_codes_all if c not in set(available_anchor_codes)]
        missing_codes = [code for code in codes if code not in set(available_codes)]
        if missing_codes:
            print(
                f"[INFO] Dropping {len(missing_codes)} codes with no raw data in window: "
                f"{', '.join(missing_codes[:10])}"
                + (" ..." if len(missing_codes) > 10 else "")
            )
            codes = available_codes
            paths["codes_out"].write_text("\n".join(codes) + "\n", encoding="utf-8")
        if args.include_spot:
            append_spot_to_raw_panel(
                raw_panel_path=paths["raw_panel_out"],
                spot_csv_path=Path(args.spot_csv).expanduser().resolve(),
                end_date=args.end_date,
            )
        # Build forecast-mu after raw panel (+ optional spot) so intraday runs can
        # supplement qlib close with today's spot close for the last forecast date.
        prepare_forecast_mu_route_a(args, paths)
        if not _should_check_route_a_reuse_dir_early(args):
            _validate_route_a_reuse_dir(args, paths)
        if args.mode == "full":
            run_route_a_full(
                codes=codes,
                output_dir=paths["route_a_output_dir"],
                start_date=route_a_start_date,
                end_date=args.end_date,
                train_window=args.route_a_train_window,
                retrain_every=args.route_a_retrain_every,
                training_objective=args.route_a_training_objective,
                keep_output_dir=args.keep_route_a_output_dir,
                route_max_workers=args.route_max_workers,
            )
        elif args.mode == "rolling-refresh":
            run_route_a_rolling_refresh(
                codes=codes,
                output_dir=paths["route_a_output_dir"],
                start_date=route_a_start_date,
                end_date=args.end_date,
                train_window=args.route_a_train_window,
                retrain_every=args.route_a_retrain_every,
                training_objective=args.route_a_training_objective,
                keep_output_dir=args.keep_route_a_output_dir,
                route_max_workers=args.route_max_workers,
            )
        if args.monitor_overlay:
            # Phase 1: bridge without overlay (needed to produce monitor status)
            run_bridge(args, paths)
            apply_risk_controls(
                bridge_csv_path=paths["bridge_out"],
                bridge_l3_csv_path=paths["bridge_l3_out"],
                provider_uri=args.provider_uri,
                anchor_codes=available_anchor_codes,
                anchor_weight_total=args.anchor_weight_total,
                raw_panel_path=paths["raw_panel_out"],
                factor_fund_list_csv=getattr(args, "mean_overlay_factor_fund_list_csv", None),
                **_risk_control_kwargs(args),
            )
            run_backtest(args, paths)
            run_monitor(args, paths)
            # Phase 2: bridge with monitor overlay applied
            print("\n[INFO] Monitor overlay: rerunning bridge and backtest with overlay applied")
            run_bridge(args, paths, monitor_csv=paths["monitor_csv"])
            apply_risk_controls(
                bridge_csv_path=paths["bridge_out"],
                bridge_l3_csv_path=paths["bridge_l3_out"],
                provider_uri=args.provider_uri,
                anchor_codes=available_anchor_codes,
                anchor_weight_total=args.anchor_weight_total,
                raw_panel_path=paths["raw_panel_out"],
                factor_fund_list_csv=getattr(args, "mean_overlay_factor_fund_list_csv", None),
                **_risk_control_kwargs(args),
            )
            run_backtest(args, paths)
            run_shadow_alert(args, paths)
            run_parameter_effect_audit(args, paths)
        else:
            run_bridge(args, paths)
            apply_risk_controls(
                bridge_csv_path=paths["bridge_out"],
                bridge_l3_csv_path=paths["bridge_l3_out"],
                provider_uri=args.provider_uri,
                anchor_codes=available_anchor_codes,
                anchor_weight_total=args.anchor_weight_total,
                raw_panel_path=paths["raw_panel_out"],
                factor_fund_list_csv=getattr(args, "mean_overlay_factor_fund_list_csv", None),
                **_risk_control_kwargs(args),
            )
            run_backtest(args, paths)
            run_monitor(args, paths)
            run_shadow_alert(args, paths)
            run_parameter_effect_audit(args, paths)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        lock_ctx.__exit__(None, None, None)
        return 1

    lock_ctx.__exit__(None, None, None)
    print("\n[INFO] Pipeline completed successfully")
    print(f"[INFO] L1/L2 bridge snapshot (intermediate): {paths['bridge_out']}")
    if paths["bridge_l3_out"].exists():
        print(f"[INFO] FINAL bridge for backtest/trading (with exposure_scale): {paths['bridge_l3_out']}")
    print(f"[INFO] Backtest daily NAV: {paths['daily_nav_out']}")
    print(f"[INFO] Monitor output: {paths['monitor_out']}")
    print(f"[INFO] Daily shadow alert: {paths['daily_shadow_alert_json']}")
    print(f"[INFO] Simple audit README: {paths['parameter_effect_simple_readme_md']}")
    print(f"[INFO] Latest portfolio weights: {paths['parameter_effect_simple_latest_weights_csv']}")
    print(f"[INFO] Daily decision summary: {paths['parameter_effect_simple_daily_summary_csv']}")
    if getattr(args, "audit_level", "simple") == "full":
        print(f"[INFO] Full audit report: {paths['parameter_effect_report_md']}")
        print(f"[INFO] Full decision panel: {paths['parameter_effect_decision_panel_csv']}")
        print(f"[INFO] Layer quality report: {paths['parameter_effect_layer_quality_report_md']}")
        print(f"[INFO] Layer quality panel: {paths['parameter_effect_layer_quality_panel_csv']}")
    return 0
