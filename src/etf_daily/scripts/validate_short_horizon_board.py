#!/usr/bin/env python3
"""Walk-forward validation for MLFAM short-horizon board (H=10 default).

Example:
  cd ~/gitee/skfolio_csi300/etf_strategy_clean
  PY=~/etf-daily-output/python/envs/py312/bin/python
  $PY src/etf_daily/scripts/validate_short_horizon_board.py \\
    --pack-dir ~/temp/decision_packs/20260625
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.scripts.generate_fat_tail_risk_report import (
    build_universe_codes,
    load_as_of,
    load_holdings_from_pack,
    warmup_start,
)
from etf_daily.lib.config import load_decision_pack_config
from etf_daily.lib.fat_tail_risk import FatTailConfig, fat_tail_config_from_raw
from etf_daily.lib.short_horizon_mlfam import ShortHorizonConfig, short_horizon_config_from_raw
from etf_daily.lib.short_horizon_validation import (
    build_eval_dates,
    format_validation_report,
    per_symbol_validation,
    run_walk_forward_validation,
)
from etf_daily.lib.symbol_trend_board import load_cluster_mapping_codes
from etf_daily.pipeline.price_adjustments import auto_qfq_adjust_close_panel
from etf_daily.pipeline.strategy_layer_audit import load_qlib_close
from etf_daily.paths import CLUSTER_MAPPING_SELECTED_TXT, QLIB_PROVIDER_URI


def _flags_by_code(cluster_frame: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for _, row in cluster_frame.iterrows():
        flags = row.get("flags")
        if isinstance(flags, str) and flags:
            out[str(row["code"])] = tuple(f for f in flags.split("|") if f)
        elif isinstance(flags, (list, tuple)):
            out[str(row["code"])] = tuple(str(f) for f in flags)
        else:
            out[str(row["code"])] = ()
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Walk-forward validate short-horizon board metrics",
    )
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument(
        "--eval-start",
        type=str,
        default="2024-01-01",
        help="Walk-forward start date (default 2024-01-01)",
    )
    parser.add_argument(
        "--eval-end",
        type=str,
        default=None,
        help="Walk-forward end date (default: board as_of minus H trading days)",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=5,
        help="Sample every N trading days (default 5)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Override H (default config short_horizon.horizon)",
    )
    parser.add_argument(
        "--output-detail",
        type=Path,
        default=None,
        help="Optional CSV of every (code, as_of) observation",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Markdown report path",
    )
    parser.add_argument("--cluster-mapping", type=Path, default=None)
    parser.add_argument("--provider-uri", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pack_dir = args.pack_dir.expanduser().resolve()
    pack_cfg = load_decision_pack_config()
    fat_cfg = fat_tail_config_from_raw(pack_cfg.raw.get("fat_tail"))
    sh_cfg = short_horizon_config_from_raw(pack_cfg.raw.get("short_horizon"))
    if args.horizon is not None:
        sh_cfg = ShortHorizonConfig(**{**sh_cfg.__dict__, "horizon": args.horizon})

    as_of = load_as_of(pack_dir)
    defensive = frozenset(pack_cfg.defensive_codes)
    holdings = load_holdings_from_pack(pack_dir)
    cluster_path = Path(args.cluster_mapping or CLUSTER_MAPPING_SELECTED_TXT)
    cluster_codes = load_cluster_mapping_codes(cluster_path) if cluster_path.exists() else ()
    universe = build_universe_codes(holdings, cluster_codes, defensive)

    # Reuse fat-tail frame for flags only.
    from etf_daily.scripts.generate_fat_tail_risk_report import load_fat_tail_data

    _, _, cluster_frame, _ = load_fat_tail_data(
        pack_dir,
        cluster_mapping_path=cluster_path,
        provider_uri=args.provider_uri,
        defensive_codes=defensive,
        cfg=fat_cfg,
    )
    flags_map = _flags_by_code(cluster_frame)

    uri = str(args.provider_uri or QLIB_PROVIDER_URI)
    start = warmup_start(as_of, fat_cfg)
    panel = load_qlib_close(uri, universe, start, as_of)
    panel, _ = auto_qfq_adjust_close_panel(panel)

    eval_end = (
        pd.Timestamp(args.eval_end)
        if args.eval_end
        else panel.index[-1 - sh_cfg.horizon]
    )
    eval_start = pd.Timestamp(args.eval_start)
    eval_dates = build_eval_dates(
        panel,
        eval_start=eval_start,
        eval_end=eval_end,
        step_days=max(1, args.step_days),
    )
    if eval_dates.empty:
        print("[ERR] No eval dates in range", file=sys.stderr)
        return 1

    detail, summary = run_walk_forward_validation(
        panel,
        eval_dates=eval_dates,
        cfg=sh_cfg,
        flags_by_code=flags_map,
    )
    per_sym = per_symbol_validation(detail)

    report_path = (
        args.output_report.expanduser().resolve()
        if args.output_report
        else pack_dir / "SHORT_HORIZON_VALIDATION.md"
    )
    detail_path = (
        args.output_detail.expanduser().resolve()
        if args.output_detail
        else pack_dir / "short_horizon_validation_detail.csv"
    )

    report = format_validation_report(
        summary,
        pack_dir=str(pack_dir),
        as_of_board=as_of,
        per_symbol=per_sym,
        step_days=max(1, args.step_days),
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    if not detail.empty:
        detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print(f"[OK] validation report -> {report_path}")
    print(f"[OK] detail CSV        -> {detail_path} ({len(detail)} rows)")
    print(
        f"[SUMMARY] H={summary.horizon} obs={summary.n_obs} symbols={summary.n_symbols} "
        f"Brier={summary.brier_score:.4f} "
        f"cov(q05/q50/q95)={summary.coverage_q05:.1%}/{summary.coverage_q50:.1%}/{summary.coverage_q95:.1%} "
        f"bet_win={summary.bet_win_rate:.1%} (n={summary.n_bets})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
