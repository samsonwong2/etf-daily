#!/usr/bin/env python
"""Daily ETF commands: etl, pool, regime, listing, hrp, and rules7."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path


PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()), Path.cwd())
COMMANDS = (
    "etl, pool, cluster-review, regime, adaptive, listing, hrp, triggers, b1235, "
    "b1235-backtest, rules7, forward, forward-page, morning"
)


def _strip_passthrough(rest: list[str]) -> list[str]:
    if rest[:1] == ["--"]:
        return rest[1:]
    return rest


def _run_etl(extra_args: list[str]) -> int:
    from etf_daily.etl.sina_fund_etl import main as etl_main

    result = etl_main(_strip_passthrough(extra_args))
    return int(result) if isinstance(result, int) else 0


def _run_pool(extra_args: list[str]) -> int:
    from etf_daily.pool.builder.cli import main as pool_main

    result = pool_main(_strip_passthrough(extra_args))
    return int(result) if isinstance(result, int) else 0


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
    parser.add_argument("--min-corr", type=float, default=None)
    args, unknown = parser.parse_known_args(rest)
    if args.as_of and args.asof_date and args.as_of != args.asof_date:
        parser.error("--as-of and --asof-date differ")
    asof_date = args.as_of or args.asof_date or date.today().isoformat()
    tag = asof_date.replace("-", "")
    cut = ""
    if args.dist_t is not None:
        cut = f"_d{int(round(args.dist_t * 100)):03d}"
    out_root = _config_value(
        "HRP_OUTPUT_DIR",
        str(Path.home() / "temp" / "decision_packs"),
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
    if args.min_corr is not None:
        argv.extend(["--min-corr", str(args.min_corr)])
    argv.extend(unknown)
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
        return path.resolve()
    root = _config_value("PLOTLY_ROOT")
    candidate = Path(root).expanduser() / path.name if root else None
    if candidate is not None and candidate.is_dir():
        print(f"[INFO] listing-dir {path} -> {candidate}")
        return candidate.resolve()
    tried = [str(path)]
    if candidate is not None and str(candidate) not in tried:
        tried.append(str(candidate))
    hint = ""
    if root:
        siblings = sorted(Path(root).expanduser().glob("*_from_listing"))
        if siblings:
            hint = f" Latest on disk: {siblings[-1]}"
    raise SystemExit(
        "listing directory not found: "
        + " ; ".join(tried)
        + "."
        + hint
        + " Export PLOTLY_ROOT or pass an existing --listing-dir."
    )


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


def _rules7(rest: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="etf-daily rules7")
    parser.add_argument("--listing-dir", required=True)
    parser.add_argument("--as-of", default=None)
    args, unknown = parser.parse_known_args(rest)
    listing = _resolve_listing(args.listing_dir)
    as_of = args.as_of or _as_of_from_listing(listing)
    if not as_of:
        print("pass --as-of YYYY-MM-DD", file=sys.stderr)
        return 2
    argv = ["--listing-dir", str(listing), "--as-of", as_of, *unknown]
    from etf_daily.scripts.scan_rules7_checklist import main as rules7_main

    result = rules7_main(argv)
    return int(result) if isinstance(result, int) else 0


def _forward(rest: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="etf-daily forward")
    parser.add_argument("--listing-dir", required=True)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--out-dir", default=None)
    args, unknown = parser.parse_known_args(rest)
    if unknown:
        print(f"unknown arguments: {' '.join(unknown)}", file=sys.stderr)
        return 2
    listing = _resolve_listing(args.listing_dir)
    as_of = args.as_of or _as_of_from_listing(listing)
    if not as_of:
        print("pass --as-of YYYY-MM-DD", file=sys.stderr)
        return 2
    from etf_daily.lib.rules7_forward import run

    try:
        result = run(listing, as_of, out_dir=Path(args.out_dir) if args.out_dir else None)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    if result.get("code"):
        print(result.get("error", "forward failed"), file=sys.stderr)
        return int(result["code"])
    verdict = result["verdict"]
    print(
        f"[OK] {result['report']}  names={result['n_names']} rows={result['n_rows']} "
        f"裁决={verdict['overall']}"
    )
    for line in result.get("errors") or []:
        print(f"[skip] {line}", file=sys.stderr)
    return 0


def _forward_page(rest: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="etf-daily forward-page")
    parser.add_argument("--listing-dir", required=True)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--codes", default=None, help="comma-separated codes; default is the 7")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out-dir", default=None)
    args, unknown = parser.parse_known_args(rest)
    if unknown:
        print(f"unknown arguments: {' '.join(unknown)}", file=sys.stderr)
        return 2
    if args.all and args.codes:
        print("pass --all or --codes, not both", file=sys.stderr)
        return 2
    listing = _resolve_listing(args.listing_dir)
    as_of = args.as_of or _as_of_from_listing(listing)
    if not as_of:
        print("pass --as-of YYYY-MM-DD", file=sys.stderr)
        return 2
    from etf_daily.plots.plot_rules7_forward import write_pages

    codes = None if args.codes is None else [item.strip() for item in args.codes.split(",") if item.strip()]
    try:
        result = write_pages(
            listing,
            as_of,
            codes=codes,
            all_names=bool(args.all),
            dest_dir=Path(args.out_dir) if args.out_dir else None,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    for path in result["written"]:
        print(f"[OK] {path}")
    if result["missing"]:
        print(
            "no listing html for "
            + ", ".join(result["missing"])
            + ". Run etf-daily listing first. This command does not draw those pages.",
            file=sys.stderr,
        )
    for line in result.get("errors") or []:
        print(f"[skip] {line}", file=sys.stderr)
    return 0 if not result["missing"] and not result.get("errors") else 2


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
        print(f"etf-daily commands: {COMMANDS}")
        return 0
    command, rest = raw[0], raw[1:]
    if command == "etl":
        return _run_etl(rest)
    if command == "pool":
        return _run_pool(rest)
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
    if command == "rules7":
        return _rules7(rest)
    if command == "forward":
        return _forward(rest)
    if command == "forward-page":
        return _forward_page(rest)
    if command == "morning":
        from etf_daily.morning import main as morning_main

        return morning_main(rest)
    print(f"unknown command: {command}", file=sys.stderr)
    print(f"etf-daily commands: {COMMANDS}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
