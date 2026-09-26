#!/usr/bin/env python3
"""Scan pool for A→B peak-sell pairs (冲高提醒 / 回落确认).

Example:
  python decision_pack/scripts/scan_regime_transition_ab_sell.py \\
    --validation-dir workspace/decision_packs/20260720/regime_transition_validation_q90_to0720 \\
    --start-date 2026-05-01 --end-date 2026-07-22 \\
    --output-dir workspace/decision_packs/20260720/regime_transition_validation_q90_to0720/assessment/ab_sell_scan
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from decision_pack.src.regime_transition_ab_sell_scan import (
    ABSellScanConfig,
    merge_oos_with_causal_prior,
    scan_ab_sell_candidates,
    write_ab_sell_scan_outputs,
)
from decision_pack.src.regime_transition_plot import DEFAULT_VALIDATION_DIR, load_validation_csvs
from decision_pack.src.symbol_trend_board import load_cluster_mapping_codes
from runtime_paths import CLUSTER_MAPPING_SELECTED_TXT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scan A→B peak-sell candidates across the validation pool"
    )
    p.add_argument(
        "--validation-dir",
        type=Path,
        default=DEFAULT_VALIDATION_DIR,
        help="directory with signals_oos.csv (+ optional reversal labels)",
    )
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--end-date", type=str, default=None)
    p.add_argument(
        "--code",
        action="append",
        default=None,
        help="optional code filter; repeatable. Default: cluster_mapping ∩ oos",
    )
    p.add_argument(
        "--cluster-mapping",
        type=Path,
        default=None,
        help="cluster_mapping_selected.txt (default: runtime_paths)",
    )
    p.add_argument("--a-min-pred-cdf", type=float, default=0.90)
    p.add_argument("--a-min-return-z", type=float, default=2.0)
    p.add_argument("--b-max-pred-cdf", type=float, default=0.10)
    p.add_argument("--b-max-return-z", type=float, default=-1.5)
    p.add_argument("--max-gap-days", type=int, default=5)
    p.add_argument(
        "--require-strong-prior",
        action="store_true",
        help="require causal_prior_strong on both A and B",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: <validation-dir>/assessment/ab_sell_scan",
    )
    return p.parse_args(argv)


def _resolve(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


def resolve_codes(
    oos: pd.DataFrame,
    *,
    codes: list[str] | None,
    cluster_mapping: Path | None,
) -> list[str]:
    oos_set = set(oos["code"].astype(str).str.upper().unique())
    if codes:
        wanted = [str(c).upper() for c in codes]
        missing = [c for c in wanted if c not in oos_set]
        if missing:
            raise ValueError(f"codes not in signals_oos: {', '.join(missing)}")
        return wanted
    mapping = Path(cluster_mapping or CLUSTER_MAPPING_SELECTED_TXT)
    if not mapping.is_absolute():
        mapping = _PROJECT_ROOT / mapping if not mapping.exists() else mapping
    if mapping.exists():
        pool = [c for c in load_cluster_mapping_codes(mapping) if c in oos_set]
        if pool:
            return pool
    return sorted(oos_set)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validation_dir = _resolve(args.validation_dir)
    oos, rev, _evt = load_validation_csvs(validation_dir)
    signals = merge_oos_with_causal_prior(oos, rev)
    codes = resolve_codes(
        signals,
        codes=args.code,
        cluster_mapping=args.cluster_mapping,
    )
    cfg = ABSellScanConfig(
        a_min_pred_cdf=args.a_min_pred_cdf,
        a_min_return_z=args.a_min_return_z,
        b_max_pred_cdf=args.b_max_pred_cdf,
        b_max_return_z=args.b_max_return_z,
        max_gap_days=args.max_gap_days,
        require_strong_prior=args.require_strong_prior,
    )
    a_hits, b_hits, pairs = scan_ab_sell_candidates(
        signals,
        cfg=cfg,
        codes=codes,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    out_dir = _resolve(
        args.output_dir
        if args.output_dir is not None
        else validation_dir / "assessment" / "ab_sell_scan"
    )
    paths = write_ab_sell_scan_outputs(
        out_dir, a_hits=a_hits, b_hits=b_hits, pairs=pairs, cfg=cfg
    )

    print(f"[INFO] pool codes={len(codes)} window=[{args.start_date} → {args.end_date}]")
    print(
        f"[INFO] A={len(a_hits)} B={len(b_hits)} pairs={len(pairs)} "
        f"cfg={cfg}"
    )
    if not pairs.empty:
        show = pairs.head(30)
        print("[INFO] pairs (head):")
        print(show.to_string(index=False))
        sh = pairs[pairs["code"] == "SH588710"]
        if not sh.empty:
            print("[INFO] SH588710 pairs:")
            print(sh.to_string(index=False))
    for key, path in paths.items():
        print(f"[OK] {key} → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
