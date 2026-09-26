"""Top-level orchestrator for generating ``cluster_mapping_selected.txt``.

Replaces ``2_filter/0.1run_all_fund_herc.py``. Runs the two core stages:

1. ``select`` — produce ``cluster_mapping.csv`` / ``cluster_mapping_selected.csv``
   (and the benchmark variant) via :mod:`pool_builder.selector`.
2. ``filter-txt`` — turn the selected CSV into the canonical instruments
   ``cluster_mapping_selected.txt`` via :mod:`pool_builder.filter_txt`.

Optional stages that depended on ``cluster_pool_rolling_eval/`` (one-shot
forward evaluation + auto-apply best pool) are out of scope of this package;
a thin hook is provided via ``--production-eval-cmd`` / ``--production-apply-cmd``
to keep the door open without hard-coding a missing repo path.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys

from . import constants as C
from .filter_txt import filter_all_txt
from .selector import select_pool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate cluster_mapping_selected.txt: select → filter_txt [→ optional one-shot eval/apply]"
    )
    parser.add_argument(
        "--shared-config",
        default=None,
        help="Path to shared_filter_config.json (default: fund_pool_builder/shared_filter_config.json)",
    )
    parser.add_argument(
        "--inception-cutoff",
        default=None,
        help="Override fund-inception cutoff date (default: test_period[1] from shared config)",
    )
    parser.add_argument(
        "--skip-select",
        action="store_true",
        help="Skip the clustering/selection stage (reuse existing cluster_mapping_selected.csv)",
    )
    parser.add_argument(
        "--skip-filter-txt",
        action="store_true",
        help="Skip the canonical-txt stage",
    )
    parser.add_argument(
        "--auto-selected-suffix",
        default=None,
        metavar="YYYYMMDD",
        help=(
            "Write machine-selected rows to cluster_mapping_selected_{suffix}.csv under "
            f"{C.OUT_DIR} and do not overwrite {C.CSV_SELECTED_OUT}."
        ),
    )
    parser.add_argument(
        "--csv-path",
        default=C.CSV_SELECTED_OUT,
        help=f"Input CSV for filter-txt stage (default: {C.CSV_SELECTED_OUT})",
    )
    parser.add_argument(
        "--all-path",
        default=C.DEFAULT_ALL_TXT,
        help=f"qlib all.txt path (default: {C.DEFAULT_ALL_TXT})",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help=(
            "Output canonical txt. Defaults to "
            f"{C.DEFAULT_CANONICAL_TXT} when --csv-path is left at its default, "
            "else <csv_path>.with_suffix('.txt')."
        ),
    )
    parser.add_argument(
        "--code-column",
        default=None,
        help="Explicit code column name in --csv-path.",
    )
    parser.add_argument(
        "--production-eval-cmd",
        default=None,
        help=(
            "Optional shell command for the one-shot production evaluation stage "
            "(e.g. 'python /path/to/production_cluster_pool_eval.py --eval-start 2025-04-01 "
            "--eval-end 2026-03-31'). If unset, the stage is skipped."
        ),
    )
    parser.add_argument(
        "--production-apply-cmd",
        default=None,
        help=(
            "Optional shell command for the production apply stage "
            "(e.g. 'python /path/to/production_cluster_pool_apply.py'). If unset, skipped."
        ),
    )
    return parser


def _run_shell(command: str, stage_name: str) -> None:
    print(f"\n== {stage_name} ==")
    print(command)
    completed = subprocess.run(shlex.split(command), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Stage failed: {stage_name}, exit_code={completed.returncode}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.skip_select:
        print("== select ==")
        select_pool(
            shared_config_path=args.shared_config,
            inception_cutoff=args.inception_cutoff,
            auto_selected_suffix=args.auto_selected_suffix,
        )

    if not args.skip_filter_txt:
        print("\n== filter_txt ==")
        filter_all_txt(
            csv_path=args.csv_path,
            all_path=args.all_path,
            output_path=args.output_path,
            code_column=args.code_column,
        )

    if args.production_eval_cmd:
        _run_shell(args.production_eval_cmd, "production_one_shot_eval")

    if args.production_apply_cmd:
        _run_shell(args.production_apply_cmd, "apply_best_pool")

    print("\n[INFO] fund-pool pipeline completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
