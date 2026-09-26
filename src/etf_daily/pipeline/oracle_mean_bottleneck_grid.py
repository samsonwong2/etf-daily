"""Run and summarize oracle-mean bottleneck experiments.

This module is intentionally an experiment orchestrator, not a production path.
It reuses existing pipeline CLI flags to isolate the impact of mean scaling,
MVO constraints, L3 exposure controls, and variance/correlation settings after
oracle mean data has been converted to a Route-A override directory.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_ROUTE_A_OUTPUT_DIR = "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/route_a_oracle_mean_{suffix}"
DEFAULT_TEMP_ROOT = "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/oracle_bottleneck_grid_{suffix}"
DEFAULT_ANNUALIZED_CSV = "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/annualized_returns_{suffix}.csv"
PIPELINE_ENTRY = "run.py"


@dataclass(frozen=True)
class Scenario:
    name: str
    profile: str
    mu_preset: str
    route_a_output_dir: Path
    temp_dir: Path
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    override_mu_scale: float | None = None
    override_winsor_quantile: float | None = None
    note: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run/summarize oracle mean bottleneck grids across mean, MVO, L3, variance, and covariance settings."
    )
    parser.add_argument("--start-date", default="2025-05-01")
    parser.add_argument("--end-date", default="2026-04-30")
    parser.add_argument(
        "--profiles",
        default="quick",
        help="Comma-separated profiles: quick,baselines,mean-scale,mvo,l3,cov,production,all.",
    )
    parser.add_argument("--route-max-workers", type=int, default=4)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--base-route-a-output-dir", default=None)
    parser.add_argument("--annualized-csv", default=None)
    parser.add_argument("--temp-root", default=None)
    parser.add_argument("--execute", action="store_true", help="Actually run missing scenarios. Default only writes plan/summary.")
    parser.add_argument("--rerun-existing", action="store_true", help="Rerun scenarios even when daily NAV already exists.")
    parser.add_argument("--overwrite-overrides", action="store_true", help="Rebuild scaled override dirs if they exist.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap on the number of scenarios to execute.")
    parser.add_argument("--only", default="", help="Comma-separated scenario names to include after profile expansion.")
    return parser.parse_args()


def compact_date(value: str) -> str:
    return value.replace("-", "")


def suffix(start_date: str, end_date: str) -> str:
    return f"{compact_date(start_date)}_{compact_date(end_date)}"


def safe_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def repo_root_from_args(args: argparse.Namespace) -> Path:
    if args.repo_root:
        return Path(args.repo_root).expanduser().resolve()
    return next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())


def expected_slug(mu_preset: str) -> str:
    aliases = {
        "simple_pulse_mu": "spm",
        "posterior_mu1d_softshrink_default": "pmu1d_ss",
    }
    if mu_preset in aliases:
        return aliases[mu_preset]
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in mu_preset) or "preset"


def base_paths(args: argparse.Namespace) -> tuple[str, Path, Path, Path]:
    run_suffix = suffix(args.start_date, args.end_date)
    route_a_output_dir = Path(
        (args.base_route_a_output_dir or DEFAULT_ROUTE_A_OUTPUT_DIR.format(suffix=run_suffix))
    ).expanduser().resolve()
    annualized_csv = Path(
        (args.annualized_csv or DEFAULT_ANNUALIZED_CSV.format(suffix=run_suffix))
    ).expanduser().resolve()
    temp_root = Path((args.temp_root or DEFAULT_TEMP_ROOT.format(suffix=run_suffix))).expanduser().resolve()
    return run_suffix, route_a_output_dir, annualized_csv, temp_root


def scenario_temp_dir(temp_root: Path, run_suffix: str, name: str) -> Path:
    if name == "pure_oracle_raw":
        return Path(f"~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/oracle_mean_replay_raw_{run_suffix}")
    if name == "production_simple_pulse_overlay":
        return Path(f"~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/oracle_mean_replay_simple_pulse_{run_suffix}")
    return temp_root / name


def scaled_override_dir(temp_root: Path, run_suffix: str, scale: float, winsor: float | None) -> Path:
    scale_tag = f"scale{scale:g}".replace(".", "p")
    if winsor is None:
        return temp_root / f"route_a_oracle_mean_{run_suffix}_{scale_tag}"
    winsor_tag = f"winsor{winsor:g}".replace(".", "p")
    return temp_root / f"route_a_oracle_mean_{run_suffix}_{scale_tag}_{winsor_tag}"


def raw_oracle_args(*, overlay: str = "off", shrink_alpha: float = 1.0) -> tuple[str, ...]:
    return (
        "--mu-shrink-alpha", str(float(shrink_alpha)),
        "--mean-overlay-mode", overlay,
    )


def simple_pulse_args(*, overlay: str = "enabled", patch_level: float = 0.015) -> tuple[str, ...]:
    return (
        "--mean-overlay-mode", overlay,
        "--mean-overlay-patch-level", str(float(patch_level)),
    )


def build_scenarios(args: argparse.Namespace) -> list[Scenario]:
    run_suffix, base_route_a_output_dir, _annualized_csv, temp_root = base_paths(args)
    profiles = set(split_csv(args.profiles))
    if "all" in profiles:
        profiles = {"baselines", "mean-scale", "mvo", "l3", "cov", "production"}
    if "quick" in profiles:
        profiles.update({"baselines", "mean-scale-quick", "mvo-quick", "l3-quick", "cov-quick"})
        profiles.discard("quick")

    scenarios: list[Scenario] = []

    def add(name: str, profile: str, mu_preset: str, route_dir: Path, extra: Iterable[str], note: str = "", scale: float | None = None, winsor: float | None = None) -> None:
        scenarios.append(
            Scenario(
                name=name,
                profile=profile,
                mu_preset=mu_preset,
                route_a_output_dir=route_dir,
                temp_dir=scenario_temp_dir(temp_root, run_suffix, name),
                extra_args=tuple(extra),
                override_mu_scale=scale,
                override_winsor_quantile=winsor,
                note=note,
            )
        )

    if "baselines" in profiles:
        add(
            "pure_oracle_raw",
            "baselines",
            "legacy_raw_mu",
            base_route_a_output_dir,
            raw_oracle_args(overlay="off", shrink_alpha=1.0),
            "Clean oracle mean: no overlay, no shrink.",
        )
        add(
            "production_simple_pulse_overlay",
            "baselines",
            "simple_pulse_mu",
            base_route_a_output_dir,
            simple_pulse_args(overlay="enabled", patch_level=0.015),
            "Current production-compatible stack with oracle Route-A files.",
        )

    mean_scales = [0.5, 1.0, 2.0, 4.0] if "mean-scale" in profiles else []
    if "mean-scale-quick" in profiles:
        mean_scales.extend([2.0])
    for scale in sorted(set(mean_scales)):
        route_dir = scaled_override_dir(temp_root, run_suffix, scale, winsor=None)
        add(
            f"mean_scale_{scale:g}_raw",
            "mean-scale",
            "legacy_raw_mu",
            route_dir,
            raw_oracle_args(overlay="off", shrink_alpha=1.0),
            f"Pure oracle with daily mu multiplied by {scale:g}.",
            scale=scale,
        )

    if "mean-scale" in profiles:
        for scale, winsor in [(2.0, 0.01), (4.0, 0.01)]:
            route_dir = scaled_override_dir(temp_root, run_suffix, scale, winsor=winsor)
            add(
                f"mean_scale_{scale:g}_winsor{winsor:g}_raw",
                "mean-scale",
                "legacy_raw_mu",
                route_dir,
                raw_oracle_args(overlay="off", shrink_alpha=1.0),
                f"Pure oracle with scale={scale:g} and cross-sectional winsor={winsor:g}.",
                scale=scale,
                winsor=winsor,
            )

    mvo_grid = [(0.20, 0.03)] if "mvo-quick" in profiles else []
    if "mvo" in profiles:
        mvo_grid.extend(itertools.product([0.15, 0.20, 0.25], [0.02, 0.03, 0.05]))
    for max_weight, min_floor in sorted(set(mvo_grid)):
        add(
            f"mvo_cap{max_weight:g}_floor{min_floor:g}",
            "mvo",
            "legacy_raw_mu",
            base_route_a_output_dir,
            raw_oracle_args(overlay="off", shrink_alpha=1.0) + (
                "--max-weight-cap", str(float(max_weight)),
                "--min-weight-floor", str(float(min_floor)),
            ),
            f"MVO hard-band sensitivity: cap={max_weight:g}, floor={min_floor:g}.",
        )

    l3_grid = [(0.8, 0.020, 0.026)] if "l3-quick" in profiles else []
    if "l3" in profiles:
        l3_grid.extend(itertools.product([0.5, 0.8, 1.0], [0.015, 0.020, 0.025], [0.022, 0.026]))
    for regime_min, vol_target, vol_risk_on in sorted(set(l3_grid)):
        add(
            f"l3_reg{regime_min:g}_vt{vol_target:g}_on{vol_risk_on:g}",
            "l3",
            "legacy_raw_mu",
            base_route_a_output_dir,
            raw_oracle_args(overlay="off", shrink_alpha=1.0) + (
                "--regime-gate-min-scale", str(float(regime_min)),
                "--vol-target", str(float(vol_target)),
                "--vol-target-risk-on", str(float(vol_risk_on)),
            ),
            f"L3 exposure sensitivity: regime_min={regime_min:g}, vol_target={vol_target:g}, risk_on={vol_risk_on:g}.",
        )

    cov_grid = [(0.60, 0.20)] if "cov-quick" in profiles else []
    if "cov" in profiles:
        cov_grid.extend(itertools.product([0.50, 0.60, 0.75], [0.10, 0.20, 0.40]))
    for var_q, corr_floor in sorted(set(cov_grid)):
        add(
            f"cov_varq{var_q:g}_corr{corr_floor:g}",
            "cov",
            "legacy_raw_mu",
            base_route_a_output_dir,
            raw_oracle_args(overlay="off", shrink_alpha=1.0) + (
                "--variance-floor-quantile", str(float(var_q)),
                "--corr-floor", str(float(corr_floor)),
            ),
            f"Variance/correlation sensitivity: variance_floor_quantile={var_q:g}, corr_floor={corr_floor:g}.",
        )

    if "production" in profiles:
        for patch_level in [0.005, 0.010, 0.015]:
            add(
                f"production_patch{patch_level:g}",
                "production",
                "simple_pulse_mu",
                base_route_a_output_dir,
                simple_pulse_args(overlay="enabled", patch_level=patch_level),
                f"Production-compatible oracle with overlay patch={patch_level:g}.",
            )

    only = set(split_csv(args.only))
    if only:
        scenarios = [scenario for scenario in scenarios if scenario.name in only]

    unique: dict[str, Scenario] = {}
    for scenario in scenarios:
        unique[scenario.name] = scenario
    return list(unique.values())


def nav_path_for(scenario: Scenario, start_date: str, end_date: str) -> Path:
    run_suffix = suffix(start_date, end_date)
    slug = expected_slug(scenario.mu_preset)
    path = scenario.temp_dir / f"nav_{slug}_{run_suffix}.csv"
    if path.exists():
        return path
    return scenario.temp_dir / f"{slug}_close_to_next_close_daily_nav_{run_suffix}.csv"


def bridge_l3_path_for(scenario: Scenario, start_date: str, end_date: str) -> Path:
    run_suffix = suffix(start_date, end_date)
    slug = expected_slug(scenario.mu_preset)
    path = scenario.temp_dir / f"bridge_{slug}_{run_suffix}_l3.csv"
    if path.exists():
        return path
    return scenario.temp_dir / f"bridge_{slug}_fullpanel_{run_suffix}_l3.csv"


def audit_daily_summary_path_for(scenario: Scenario, start_date: str, end_date: str) -> Path:
    run_suffix = suffix(start_date, end_date)
    slug = expected_slug(scenario.mu_preset)
    path = scenario.temp_dir / f"audit_{slug}_{run_suffix}" / "param_daily.csv"
    if path.exists():
        return path
    return scenario.temp_dir / f"parameter_effect_audit_{slug}_fullpanel_{run_suffix}" / "daily_parameter_effect_summary.csv"


def build_override_command(args: argparse.Namespace, scenario: Scenario, annualized_csv: Path) -> list[str] | None:
    if scenario.override_mu_scale is None and scenario.override_winsor_quantile is None:
        return None
    repo_root = repo_root_from_args(args)
    command = [
        args.python_exe,
        str(repo_root / "src/etf_daily/pool" / "build_oracle_route_a_override.py"),
        "--start-date", args.start_date,
        "--end-date", args.end_date,
        "--annualized-csv", str(annualized_csv),
        "--output-dir", str(scenario.route_a_output_dir),
        "--mu-scale", str(float(scenario.override_mu_scale or 1.0)),
    ]
    if scenario.override_winsor_quantile is not None:
        command.extend(["--winsor-quantile", str(float(scenario.override_winsor_quantile))])
    if args.overwrite_overrides:
        command.append("--overwrite")
    return command


def build_pipeline_command(args: argparse.Namespace, scenario: Scenario) -> list[str]:
    repo_root = repo_root_from_args(args)
    command = [
        args.python_exe,
        str(repo_root / PIPELINE_ENTRY),
        "pipeline",
        "--start-date", args.start_date,
        "--end-date", args.end_date,
        "--mode", "reuse-route-a",
        "--route-a-output-dir", str(scenario.route_a_output_dir),
        "--temp-dir", str(scenario.temp_dir),
        "--mu-preset", scenario.mu_preset,
        "--route-max-workers", str(int(args.route_max_workers)),
        "--log-dir", str(scenario.temp_dir / "logs"),
    ]
    command.extend(scenario.extra_args)
    return command


def run_command(command: list[str], cwd: Path) -> None:
    print(" ".join(command), flush=True)
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit_code={completed.returncode}: {' '.join(command)}")


def summarize_nav(path: Path) -> dict[str, float | int | str]:
    if not path.exists():
        return {"status": "missing_nav"}
    nav = pd.read_csv(path, parse_dates=["datetime"])
    if "nav" not in nav.columns:
        return {"status": "missing_nav_column", "nav_rows": len(nav)}
    values = pd.to_numeric(nav["nav"], errors="coerce").dropna()
    if values.empty:
        return {"status": "empty_nav", "nav_rows": len(nav)}
    returns = values.pct_change().dropna()
    cum_return = float(values.iloc[-1] / values.iloc[0] - 1.0)
    sharpe = float("nan")
    if len(returns) > 1 and float(returns.std()) > 0.0:
        sharpe = float(returns.mean() / returns.std() * math.sqrt(252.0))
    max_drawdown = float((values / values.cummax() - 1.0).min())
    return {
        "status": "ok",
        "nav_rows": int(len(nav)),
        "cum_return": cum_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "final_nav": float(values.iloc[-1]),
    }


def summarize_bridge(path: Path) -> dict[str, float | int | str]:
    if not path.exists():
        return {"bridge_status": "missing_bridge"}
    columns = {
        "datetime",
        "instrument",
        "forecast_w",
        "exposure_scale",
        "forecast_mean_source",
        "route_a_status",
        "mean_overlay_trigger_flag",
    }
    bridge = pd.read_csv(path, usecols=lambda column: column in columns)
    out: dict[str, float | int | str] = {"bridge_status": "ok", "bridge_rows": int(len(bridge))}
    if "route_a_status" in bridge.columns:
        out["route_a_ok_rate"] = float((bridge["route_a_status"] == "ok").mean())
    if "forecast_mean_source" in bridge.columns:
        out["route_a_mu_source_rate"] = float((bridge["forecast_mean_source"] == "route_a_mu").mean())
    if "forecast_w" in bridge.columns and "datetime" in bridge.columns:
        w = pd.to_numeric(bridge["forecast_w"], errors="coerce").fillna(0.0)
        out["active_rows"] = int((w > 0.0).sum())
        out["avg_gross_weight"] = float(w.groupby(bridge["datetime"]).sum().mean())
        active = bridge.loc[w > 0.0]
        if not active.empty:
            out["avg_active_names"] = float(active.groupby("datetime")["instrument"].nunique().mean())
    if "exposure_scale" in bridge.columns and "datetime" in bridge.columns:
        exposure = pd.to_numeric(bridge["exposure_scale"], errors="coerce")
        out["avg_exposure_scale"] = float(exposure.groupby(bridge["datetime"]).mean().mean())
    if "mean_overlay_trigger_flag" in bridge.columns:
        overlay = bridge["mean_overlay_trigger_flag"].astype(str).str.lower().isin({"true", "1"})
        out["overlay_trigger_rows"] = int(overlay.sum())
    return out


def summarize_audit(path: Path) -> dict[str, float | int | str]:
    if not path.exists():
        return {"audit_status": "missing_audit"}
    daily = pd.read_csv(path)
    out: dict[str, float | int | str] = {"audit_status": "ok", "audit_days": int(len(daily))}
    for column in [
        "portfolio_expected_mean_contribution",
        "portfolio_realized_return_contribution",
        "l3_gross_weight_delta",
        "l3_return_delta_proxy_sum",
        "variance_floor_trigger_rate",
        "risk_base_exposure_scale",
        "risk_regime_factor",
        "risk_vol_scale",
        "effective_holdings_pre_l3",
        "weight_top1",
        "weight_top5",
        "covariance_condition_number",
        "covariance_effective_rank",
        "sigma_port_pre_l3",
        "sigma_port_effective_l3",
    ]:
        if column in daily.columns:
            series = pd.to_numeric(daily[column], errors="coerce")
            out[f"avg_{column}"] = float(series.mean())
            out[f"median_{column}"] = float(series.median())
    if "l3_gross_weight_delta" in daily.columns:
        series = pd.to_numeric(daily["l3_gross_weight_delta"], errors="coerce").fillna(0.0)
        out["l3_cut_days"] = int((series.abs() > 1e-12).sum())
        out["l3_cut_abs_sum"] = float(series.abs().sum())
    return out


def scenario_plan_rows(scenarios: list[Scenario], args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        rows.append(
            {
                "name": scenario.name,
                "profile": scenario.profile,
                "mu_preset": scenario.mu_preset,
                "route_a_output_dir": str(scenario.route_a_output_dir),
                "temp_dir": str(scenario.temp_dir),
                "nav_path": str(nav_path_for(scenario, args.start_date, args.end_date)),
                "override_mu_scale": scenario.override_mu_scale,
                "override_winsor_quantile": scenario.override_winsor_quantile,
                "extra_args": " ".join(scenario.extra_args),
                "note": scenario.note,
            }
        )
    return rows


def summarize_scenarios(scenarios: list[Scenario], args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        row: dict[str, object] = {
            "name": scenario.name,
            "profile": scenario.profile,
            "mu_preset": scenario.mu_preset,
            "temp_dir": str(scenario.temp_dir),
            "route_a_output_dir": str(scenario.route_a_output_dir),
            "note": scenario.note,
        }
        row.update(summarize_nav(nav_path_for(scenario, args.start_date, args.end_date)))
        row.update(summarize_bridge(bridge_l3_path_for(scenario, args.start_date, args.end_date)))
        row.update(summarize_audit(audit_daily_summary_path_for(scenario, args.start_date, args.end_date)))
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(scenarios: list[Scenario], args: argparse.Namespace, temp_root: Path) -> tuple[Path, Path, Path]:
    temp_root.mkdir(parents=True, exist_ok=True)
    plan_path = temp_root / "oracle_bottleneck_grid_plan.csv"
    summary_path = temp_root / "oracle_bottleneck_grid_summary.csv"
    manifest_path = temp_root / "oracle_bottleneck_grid_manifest.json"
    pd.DataFrame(scenario_plan_rows(scenarios, args)).to_csv(plan_path, index=False)
    summary = summarize_scenarios(scenarios, args)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "profiles": args.profiles,
        "execute": bool(args.execute),
        "scenario_count": len(scenarios),
        "plan_path": str(plan_path),
        "summary_path": str(summary_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return plan_path, summary_path, manifest_path


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_args(args)
    run_suffix, _base_route_a_output_dir, annualized_csv, temp_root = base_paths(args)
    scenarios = build_scenarios(args)
    if args.max_runs is not None:
        scenarios = scenarios[: int(args.max_runs)]
    plan_path, summary_path, manifest_path = write_outputs(scenarios, args, temp_root)
    print(f"Plan: {plan_path}")
    print(f"Summary: {summary_path}")
    print(f"Manifest: {manifest_path}")
    print(pd.read_csv(summary_path).sort_values(["status", "cum_return"], ascending=[True, False]).head(20).to_string(index=False))

    if not args.execute:
        print("Dry run only. Pass --execute to run missing scenarios.")
        for scenario in scenarios:
            override_command = build_override_command(args, scenario, annualized_csv)
            if override_command is not None:
                print("[override]", " ".join(override_command))
            print("[pipeline]", " ".join(build_pipeline_command(args, scenario)))
        return 0

    executed = 0
    for scenario in scenarios:
        nav_path = nav_path_for(scenario, args.start_date, args.end_date)
        if nav_path.exists() and not args.rerun_existing:
            print(f"[SKIP] {scenario.name}: existing NAV {nav_path}")
            continue
        override_command = build_override_command(args, scenario, annualized_csv)
        if override_command is not None:
            if scenario.route_a_output_dir.exists() and not args.overwrite_overrides:
                print(f"[SKIP] override exists for {scenario.name}: {scenario.route_a_output_dir}")
            else:
                run_command(override_command, cwd=repo_root)
        run_command(build_pipeline_command(args, scenario), cwd=repo_root)
        executed += 1
        write_outputs(scenarios, args, temp_root)

    print(f"Executed scenarios: {executed}")
    _, summary_path, _ = write_outputs(scenarios, args, temp_root)
    print(pd.read_csv(summary_path).sort_values("cum_return", ascending=False, na_position="last").head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
