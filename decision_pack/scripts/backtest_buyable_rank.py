#!/usr/bin/env python3
"""Walk-forward backtest for buyable Top-K selection rules.

Example:
  cd ~/gitee/skfolio_csi300/etf_strategy_clean
  PY=~/etf-daily-output/python/envs/py312/bin/python
  $PY decision_pack/scripts/backtest_buyable_rank.py \\
    --pack-dir ~/temp/decision_packs/20260717 \\
    --eval-start 2026-04-01 \\
    --eval-end 2026-07-17 \\
    --cost-bps-per-side 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from decision_pack.src.buyable_rank_backtest import (
    StrategySpec,
    build_strategy_grid,
    run_buyable_rank_backtest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backtest buyable Top-K rules (next open→close, costed)",
    )
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--eval-start", type=str, default="2026-04-01")
    parser.add_argument("--eval-end", type=str, default="2026-07-17")
    parser.add_argument("--cost-bps-per-side", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--provider-uri", type=Path, default=None)
    parser.add_argument("--cluster-mapping", type=Path, default=None)
    parser.add_argument(
        "--write-daily-buyable",
        action="store_true",
        help="Write per-day buyable_topk_picks.csv under output_dir/daily/",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny grid + short window for smoke testing",
    )
    parser.add_argument("--progress-every", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pack_dir = args.pack_dir.expanduser().resolve()

    strategies: list[StrategySpec] | None = None
    eval_start = args.eval_start
    eval_end = args.eval_end
    if args.smoke:
        eval_start = "2026-07-10"
        eval_end = "2026-07-17"
        strategies = build_strategy_grid(
            include_benchmarks=True,
            k_values=(2, 8),
            rank_rules=("upside_x_cred",),
            gates=("strict", "core_risk"),
            min_creds=(0.0, 0.003),
        )

    result = run_buyable_rank_backtest(
        pack_dir=pack_dir,
        eval_start=eval_start,
        eval_end=eval_end,
        cost_bps_per_side=float(args.cost_bps_per_side),
        strategies=strategies,
        provider_uri=args.provider_uri,
        cluster_mapping_path=args.cluster_mapping,
        write_daily_buyable=bool(args.write_daily_buyable),
        output_dir=args.output_dir,
        progress_every=int(args.progress_every),
    )
    out_dir = result["output_dir"]
    best = result["best"]
    print(f"[OK] output -> {out_dir}", flush=True)
    print(f"[OK] signal days = {result['n_signal_days']}", flush=True)
    if best:
        print(f"[OK] best (validation Sharpe) -> {best['strategy']}", flush=True)
        print(json.dumps(best, ensure_ascii=False, indent=2), flush=True)
    else:
        print("[WARN] no best strategy selected", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
