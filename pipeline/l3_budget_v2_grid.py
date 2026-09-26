"""Compare legacy binary L3, vol-off patch, and budget_v2 profiles across dual windows."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .constants import REPO_ROOT

PIPELINE_ENTRY = REPO_ROOT / "run.py"


@dataclass(frozen=True)
class WindowSpec:
    name: str
    start_date: str
    end_date: str
    cluster_mapping_path: str
    route_a_output_dir: str


@dataclass(frozen=True)
class Scenario:
    name: str
    profile: str
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""


DEFAULT_WINDOWS = (
    WindowSpec(
        name="2023_2024",
        start_date="2023-09-01",
        end_date="2024-08-31",
        cluster_mapping_path="~/etf-daily-output/Documents/stock/data/qlib_data/all_fund_data/instruments/cluster_mapping_selected_20230901_20240831.txt",
        route_a_output_dir="~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/forecast_mu_model_grid_20260514/2023_ewma/route_a_forecast_mu_ewma60_shrink0p75_20230901_20240831",
    ),
    WindowSpec(
        name="2025_2026",
        start_date="2025-05-01",
        end_date="2026-04-30",
        cluster_mapping_path="~/etf-daily-output/Documents/stock/data/qlib_data/all_fund_data/instruments/cluster_mapping_selected_20250501_20260430.txt",
        route_a_output_dir="~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/route_a_forecast_mu_ewma60_shrink0p75_20250501_20260430",
    ),
)

DEFAULT_SCENARIOS = (
    Scenario("balanced_binary", "simple_pulse_adaptive_balanced_v1", note="Legacy HS300/MA20 binary gate"),
    Scenario(
        "balanced_vol_off_patch",
        "simple_pulse_adaptive_balanced_v1",
        ("--vol-target-risk-off", "0.015"),
        note="Binary gate + enabled risk-off vol target",
    ),
    Scenario("balanced_budget_v2", "simple_pulse_adaptive_balanced_v2", note="Macro budget + always-on vol tiers"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or execute L3 budget_v2 ablation grid.")
    parser.add_argument("--temp-root", default="~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/l3_budget_v2_ablation")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--execute", action="store_true", help="Run missing scenarios.")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--only", default="", help="Comma-separated scenario names.")
    parser.add_argument("--only-window", default="", help="Comma-separated window names (e.g. 2025_2026).")
    return parser.parse_args()


def _compact(start: str, end: str) -> str:
    return f"{start.replace('-', '')}_{end.replace('-', '')}"


def _scenario_dir(temp_root: Path, window: WindowSpec, scenario: Scenario) -> Path:
    return temp_root / window.name / scenario.name


def _nav_metrics(nav_path: Path) -> dict[str, float]:
    if not nav_path.exists():
        return {}
    nav = pd.read_csv(nav_path)
    if nav.empty or "nav" not in nav.columns:
        return {}
    series = pd.to_numeric(nav["nav"], errors="coerce").dropna()
    if series.empty:
        return {}
    rets = series.pct_change().dropna()
    total_return = float(series.iloc[-1] / series.iloc[0] - 1.0)
    max_dd = float((series / series.cummax() - 1.0).min())
    vol = float(rets.std() * math.sqrt(252)) if len(rets) > 1 else float("nan")
    sharpe = float(rets.mean() / rets.std() * math.sqrt(252)) if len(rets) > 1 and rets.std() > 0 else float("nan")
    avg_exposure = float("nan")
    if "trigger_exposure_change" in nav.columns:
        pass
    return {
        "total_return": total_return,
        "max_drawdown": max_dd,
        "annual_vol": vol,
        "sharpe": sharpe,
    }


def _summarize_runs(temp_root: Path, windows: tuple[WindowSpec, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for window in windows:
        for scenario in DEFAULT_SCENARIOS:
            run_dir = _scenario_dir(temp_root, window, scenario)
            nav_path = run_dir / f"nav_spm_{_compact(window.start_date, window.end_date)}.csv"
            if not nav_path.exists():
                nav_path = run_dir / f"simple_pulse_mu_close_to_next_close_daily_nav_{_compact(window.start_date, window.end_date)}.csv"
            metrics = _nav_metrics(nav_path)
            l3_path = next(run_dir.glob("bridge_*_l3.csv"), None)
            avg_macro = float("nan")
            if l3_path and l3_path.exists():
                l3 = pd.read_csv(l3_path, usecols=lambda c: c in {"datetime", "risk_macro_budget", "exposure_scale"})
                if "risk_macro_budget" in l3.columns:
                    avg_macro = float(pd.to_numeric(l3["risk_macro_budget"], errors="coerce").mean())
                elif "exposure_scale" in l3.columns:
                    avg_macro = float(pd.to_numeric(l3["exposure_scale"], errors="coerce").mean())
            rows.append(
                {
                    "window": window.name,
                    "scenario": scenario.name,
                    "profile": scenario.profile,
                    "note": scenario.note,
                    "run_dir": str(run_dir),
                    "nav_exists": nav_path.exists(),
                    "avg_macro_or_exposure": avg_macro,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _build_command(python_exe: str, window: WindowSpec, scenario: Scenario, temp_dir: Path) -> list[str]:
    command = [
        python_exe,
        str(PIPELINE_ENTRY),
        "pipeline",
        "--start-date",
        window.start_date,
        "--end-date",
        window.end_date,
        "--cluster-mapping-path",
        window.cluster_mapping_path,
        "--route-a-output-dir",
        window.route_a_output_dir,
        "--temp-dir",
        str(temp_dir),
        "--run-profile",
        scenario.profile,
        "--audit-level",
        "simple",
        "--forecast-mu-method",
        "route_a",
    ]
    command.extend(scenario.extra_args)
    return command


def main() -> int:
    args = parse_args()
    temp_root = Path(args.temp_root).expanduser().resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    selected = {
        item.strip()
        for item in str(args.only).split(",")
        if item.strip()
    }
    selected_windows = {
        item.strip()
        for item in str(args.only_window).split(",")
        if item.strip()
    }
    scenarios = [s for s in DEFAULT_SCENARIOS if not selected or s.name in selected]
    windows = [w for w in DEFAULT_WINDOWS if not selected_windows or w.name in selected_windows]
    plan_rows: list[dict[str, object]] = []
    for window in windows:
        for scenario in scenarios:
            run_dir = _scenario_dir(temp_root, window, scenario)
            nav_path = run_dir / f"nav_spm_{_compact(window.start_date, window.end_date)}.csv"
            if not nav_path.exists():
                nav_path = run_dir / f"simple_pulse_mu_close_to_next_close_daily_nav_{_compact(window.start_date, window.end_date)}.csv"
            command = _build_command(args.python_exe, window, scenario, run_dir)
            plan_rows.append(
                {
                    "window": window.name,
                    "scenario": scenario.name,
                    "run_dir": str(run_dir),
                    "command": " ".join(command),
                    "nav_exists": nav_path.exists(),
                }
            )
            if args.execute and (args.rerun_existing or not nav_path.exists()):
                run_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(command, cwd=str(REPO_ROOT), check=True)
    plan_path = temp_root / "l3_budget_v2_plan.json"
    summary_path = temp_root / "l3_budget_v2_summary.csv"
    report_path = temp_root / "l3_budget_v2_report.md"
    plan_path.write_text(json.dumps(plan_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = _summarize_runs(temp_root, tuple(windows))
    summary.to_csv(summary_path, index=False)
    report_path.write_text(
        "\n".join(
            [
                "# L3 budget_v2 ablation",
                "",
                f"- Plan: `{plan_path}`",
                f"- Summary: `{summary_path}`",
                "",
                "## Scenarios",
                "",
                "| name | profile | note |",
                "| --- | --- | --- |",
                *[f"| {s.name} | {s.profile} | {s.note} |" for s in scenarios],
                "",
                "## Summary",
                "",
                summary.to_markdown(index=False) if not summary.empty else "No runs found.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[INFO] wrote plan -> {plan_path}")
    print(f"[INFO] wrote summary -> {summary_path}")
    print(f"[INFO] wrote report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
