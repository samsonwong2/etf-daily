#!/usr/bin/env python
"""Unified runner for the ETF strategy workflow.

Daily use:
    python run.py pipeline --start-date 2026-04-01 --end-date 2026-05-29

Full refresh:
    python run.py all --start-date 2026-04-01 --end-date 2026-05-29

Daily health check:
    python run.py health --temp-dir temp/runs/20260529/prod
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "production_regime_switch_ewma_shrink.json"


def _run(command: list[str], *, cwd: Path = PROJECT_ROOT) -> int:
    print("\n== " + " ".join(command[1:]) + " ==")
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    return int(completed.returncode)


def _run_etl(extra_args: list[str]) -> int:
    script = PROJECT_ROOT / "data_ingestion" / "1_old_parallel_data_preparation_fund_all_2_sina.py"
    return _run([sys.executable, str(script), *extra_args])


def _run_pool(extra_args: list[str]) -> int:
    script = PROJECT_ROOT / "fund_pool_builder" / "生成cluster_mapping_selected_最短命令清单.py"
    return _run([sys.executable, str(script), *extra_args])


def _run_pipeline(args: argparse.Namespace, extra_args: list[str]) -> int:
    from pipeline.cli import main as pipeline_main

    return pipeline_main(
        [
            "--config",
            str(Path(args.config).expanduser().resolve()),
            "--start-date",
            args.start_date,
            "--end-date",
            args.end_date,
            *extra_args,
        ]
    )


def _run_health(args: argparse.Namespace, extra_args: list[str]) -> int:
    from pipeline.mu_health_check import main as health_main

    health_args = [
        "--temp-dir",
        args.temp_dir,
        "--mu-preset",
        args.mu_preset,
        "--tail",
        str(args.tail),
    ]
    if args.start_date:
        health_args.extend(["--start-date", args.start_date])
    if args.end_date:
        health_args.extend(["--end-date", args.end_date])
    health_args.extend(extra_args)
    return health_main(health_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified entry point: etl -> pool -> pipeline -> health.",
        epilog="Use '--' after a subcommand to pass through extra args to the underlying legacy script.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("etl", help="Refresh fund_list.csv and qlib binary data.")
    subparsers.add_parser("pool", help="Generate cluster_mapping_selected.txt.")
    health = subparsers.add_parser("health", help="Print the latest mu monitor and layer-quality rows.")
    health.add_argument("--temp-dir", required=True, help="Pipeline temp_dir containing monitor/audit outputs.")
    health.add_argument("--mu-preset", default="simple_pulse_mu")
    health.add_argument("--start-date", default=None, help="YYYY-MM-DD; optional if manifest/artifacts exist.")
    health.add_argument("--end-date", default=None, help="YYYY-MM-DD; optional if manifest/artifacts exist.")
    health.add_argument("--tail", type=int, default=5, help="Rows to show per table.")

    pipeline = subparsers.add_parser("pipeline", help="Run the main portfolio pipeline.")
    pipeline.add_argument("--start-date", required=True)
    pipeline.add_argument("--end-date", required=True)
    pipeline.add_argument("--config", default=str(DEFAULT_CONFIG))

    all_cmd = subparsers.add_parser("all", help="Run etl, pool, then the main pipeline.")
    all_cmd.add_argument("--start-date", required=True)
    all_cmd.add_argument("--end-date", required=True)
    all_cmd.add_argument("--config", default=str(DEFAULT_CONFIG))
    all_cmd.add_argument("--skip-etl", action="store_true", help="Skip the data refresh stage.")
    all_cmd.add_argument("--skip-pool", action="store_true", help="Skip the pool-building stage.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extra_args = parser.parse_known_args(argv)

    if args.command == "etl":
        return _run_etl(extra_args)
    if args.command == "pool":
        return _run_pool(extra_args)
    if args.command == "pipeline":
        return _run_pipeline(args, extra_args)
    if args.command == "health":
        return _run_health(args, extra_args)
    if args.command == "all":
        if not args.skip_etl:
            rc = _run_etl([])
            if rc != 0:
                return rc
        if not args.skip_pool:
            rc = _run_pool([])
            if rc != 0:
                return rc
        return _run_pipeline(args, extra_args)

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
