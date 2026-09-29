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
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "production_regime_switch_ewma_shrink.json"


def _run(command: list[str], *, cwd: Path = PROJECT_ROOT) -> int:
    print("\n== " + " ".join(command[1:]) + " ==")
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    return int(completed.returncode)


def _run_etl(extra_args: list[str]) -> int:
    script = PROJECT_ROOT / "src/etf_daily/etl" / "1_old_parallel_data_preparation_fund_all_2_sina.py"
    return _run([sys.executable, str(script), *extra_args])


def _run_pool(extra_args: list[str]) -> int:
    script = PROJECT_ROOT / "src/etf_daily/pool" / "生成cluster_mapping_selected_最短命令清单.py"
    return _run([sys.executable, str(script), *extra_args])


def _run_pipeline(args: argparse.Namespace, extra_args: list[str]) -> int:
    from etf_daily.pipeline.cli import main as pipeline_main

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
    from etf_daily.pipeline.mu_health_check import main as health_main

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


def _config_value(key: str, default: str = "") -> str:
    current = os.environ.get(key, "").strip()
    if current:
        return current
    env_file = PROJECT_ROOT / "config.env"
    if not env_file.is_file():
        return default
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, item = line.split("=", 1)
        if name.strip() != key:
            continue
        value = item.strip().strip('"').strip("'")
        return value or default
    return default


def _run_shell(script: str, extra: dict[str, str]) -> int:
    env = os.environ.copy()
    env.update(extra)
    completed = subprocess.run(
        [str(PROJECT_ROOT / "scripts" / script)],
        cwd=str(PROJECT_ROOT),
        env=env,
        check=False,
    )
    return int(completed.returncode)


def _cluster_review(rest: list[str]) -> int:
    from etf_daily.pool.builder.daily_review import main as review_main

    result = review_main(rest)
    return int(result) if isinstance(result, int) else 0


def _dated(rest: list[str], prog: str, script: str) -> int:
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--as-of", default=date.today().isoformat())
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--intraday",
        action="store_true",
        help="Live session: AkShare snapshot, clock-named pack. Do not pass a stamp.",
    )
    regime = script.startswith("daily_regime")
    listing = "from_listing" in script
    if regime:
        parser.add_argument("--skip-rebuild", action="store_true")
        parser.add_argument("--skip-html", action="store_true")
        parser.add_argument("--force-month-rebuild", action="store_true")
    args = parser.parse_args(rest)
    if args.intraday and script == "daily_adaptive_stage_html.sh":
        script = "daily_adaptive_stage_html_intraday.sh"
    extra = {"AS_OF": args.as_of, "JOBS": str(args.jobs)}
    if regime:
        extra["SKIP_REBUILD"] = "1" if args.skip_rebuild else "0"
        extra["SKIP_HTML"] = "1" if args.skip_html else "0"
        extra["FORCE_MONTH_REBUILD"] = "1" if args.force_month_rebuild else "0"
    if args.intraday and (regime or listing):
        extra["INTRADAY"] = "1"
    return _run_shell(script, extra)


def _hrp(rest: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="etf-daily hrp")
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--asof-date", default=None)
    parser.add_argument("--lookback-days", type=int, default=252)
    parser.add_argument("--dist-t", type=float, default=None)
    args = parser.parse_args(rest)
    if args.as_of and args.asof_date and args.as_of != args.asof_date:
        parser.error("--as-of and --asof-date differ")
    asof_date = args.as_of or args.asof_date or date.today().isoformat()
    tag = asof_date.replace("-", "")
    cut = ""
    if args.dist_t is not None:
        cut = f"_d{int(round(args.dist_t * 100)):03d}"
    out_root = _config_value(
        "HRP_OUTPUT_DIR",
        str(Path.home() / "etf-daily-output" / "decision_packs"),
    )
    output = Path(out_root).expanduser() / tag / f"hrp_dendrogram_{tag}{cut}.html"
    argv = [
        "--lookback-days",
        str(args.lookback_days),
        "--asof-date",
        asof_date,
        "--output",
        str(output),
    ]
    if args.dist_t is not None:
        argv.extend(["--dist-t", str(args.dist_t)])
    from etf_daily.hrp.dendrogram import main as hrp_main

    result = hrp_main(argv)
    return int(result) if isinstance(result, int) else 0


def _next_weekday(iso: str) -> str:
    day = date.fromisoformat(iso) + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day.isoformat()


def _as_of_from_listing(path: Path) -> str | None:
    digits = ""
    for char in path.name:
        if char.isdigit():
            digits += char
            if len(digits) == 8:
                break
        else:
            digits = ""
    if len(digits) != 8:
        return None
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"


def _resolve_listing(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_dir():
        return path
    root = _config_value("PLOTLY_ROOT")
    if root:
        candidate = Path(root).expanduser() / path.name
        if candidate.is_dir():
            return candidate
    return path


def _triggers(rest: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="etf-daily triggers")
    parser.add_argument("--listing-dir", required=True)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--next-day", default=None)
    parser.add_argument("--jobs", type=int, default=8)
    args, unknown = parser.parse_known_args(rest)
    listing = _resolve_listing(args.listing_dir)
    as_of = args.as_of or _as_of_from_listing(listing)
    if not as_of:
        print("pass --as-of YYYY-MM-DD", file=sys.stderr)
        return 2
    next_day = args.next_day or _next_weekday(as_of)
    argv = [
        "--listing-dir",
        str(listing),
        "--as-of",
        as_of,
        "--next-day",
        next_day,
        "--jobs",
        str(args.jobs),
        *unknown,
    ]
    from etf_daily.triggers.scan import main as scan_main

    result = scan_main(argv)
    return int(result) if isinstance(result, int) else 0


def _b1235(rest: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="etf-daily b1235")
    parser.add_argument("--listing-dir", required=True)
    parser.add_argument("--as-of", default=None)
    args, unknown = parser.parse_known_args(rest)
    listing = _resolve_listing(args.listing_dir)
    as_of = args.as_of or _as_of_from_listing(listing)
    if not as_of:
        print("pass --as-of YYYY-MM-DD", file=sys.stderr)
        return 2
    argv = [
        "--listing-dir",
        str(listing),
        "--as-of",
        as_of,
        *unknown,
    ]
    from etf_daily.scripts.scan_b1235_checklist import main as b1235_main

    result = b1235_main(argv)
    return int(result) if isinstance(result, int) else 0


def _b1235_backtest(rest: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="etf-daily b1235-backtest", add_help=False)
    parser.add_argument("--listing-dir", default=None)
    args, unknown = parser.parse_known_args(rest)
    argv = list(unknown)
    if args.listing_dir is not None:
        argv = ["--listing-dir", str(_resolve_listing(args.listing_dir)), *argv]
    from etf_daily.scripts.backtest_b1235 import main as bt_main

    result = bt_main(argv)
    return int(result) if isinstance(result, int) else 0


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0] in {"-h", "--help"}:
        print(
            "etf-daily commands: etl, pool, cluster-review, regime, adaptive, listing, hrp, triggers, b1235, "
            "b1235-backtest, morning"
        )
        return 0
    command, rest = raw[0], raw[1:]
    if command == "cluster-review":
        return _cluster_review(rest)
    if command == "regime":
        return _dated(rest, "etf-daily regime", "daily_regime_transition_validation.sh")
    if command == "adaptive":
        return _dated(rest, "etf-daily adaptive", "daily_adaptive_stage_html.sh")
    if command == "listing":
        return _dated(rest, "etf-daily listing", "daily_adaptive_from_listing.sh")
    if command == "hrp":
        return _hrp(rest)
    if command == "triggers":
        return _triggers(rest)
    if command == "b1235":
        return _b1235(rest)
    if command == "b1235-backtest":
        return _b1235_backtest(rest)
    if command == "morning":
        from etf_daily.morning import main as morning_main

        return morning_main(rest)
    parser = build_parser()
    args, extra_args = parser.parse_known_args(raw)

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
