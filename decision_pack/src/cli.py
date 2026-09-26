"""CLI entrypoints for decision_pack."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from decision_pack.src.inputs import DEFAULT_STATE_PATH
from decision_pack.src.pack import generate_decision_pack
from decision_pack.src.recovery import load_recovery_state


def _build_generate_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("generate", help="Generate a decision pack for a trading day")
    parser.add_argument("--date", required=True, help="Report date YYYY-MM-DD or YYYYMMDD")
    parser.add_argument("--run-dir", type=Path, required=True, help="Pipeline run output directory")
    parser.add_argument("--live-holdings", type=Path, default=None, help="Optional broker export CSV")
    parser.add_argument(
        "--cluster-mapping",
        type=Path,
        default=None,
        help="cluster_mapping_selected.txt for full-universe trend board (default: runtime_paths)",
    )
    parser.add_argument(
        "--no-cluster-trend-board",
        action="store_true",
        help="Skip cluster_trend_board.csv (holdings board still generated in display_only mode)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output directory override")
    parser.add_argument("--preferred-start", default="20260401", help="Preferred audit start date")
    parser.add_argument(
        "--no-regenerate-audit",
        action="store_true",
        help="Do not auto-generate missing audit outputs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write pack artifacts but do not persist recovery state",
    )
    parser.add_argument("--state-path", type=Path, default=None, help="Recovery state JSON path")


def _build_status_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("status", help="Show current recovery state")
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="decision_pack", description="Post-close ETF decision pack")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _build_generate_parser(subparsers)
    _build_status_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "status":
        state_path = args.state_path or DEFAULT_STATE_PATH
        state = load_recovery_state(state_path)
        print(json.dumps(state.to_dict(), indent=2, ensure_ascii=False))
        return 0

    if args.command == "generate":
        result = generate_decision_pack(
            run_dir=args.run_dir,
            report_date=args.date,
            output_dir=args.output,
            live_holdings=args.live_holdings,
            cluster_mapping_path=args.cluster_mapping,
            write_cluster_trend_board=not args.no_cluster_trend_board,
            preferred_start=args.preferred_start,
            ensure_audit=not args.no_regenerate_audit,
            dry_run=args.dry_run,
            state_path=args.state_path,
        )
        print(f"[OK] decision pack -> {result.output_dir}")
        print(
            f"     tier={result.decision.executed_tier} "
            f"target={result.decision.target_equity_exposure} "
            f"reduce={result.decision.reduce_fraction:.4f}"
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
