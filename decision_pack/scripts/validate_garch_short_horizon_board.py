#!/usr/bin/env python3
"""Walk-forward validation for GARCH short-horizon board.

Example:
  cd ~/gitee/skfolio_csi300/etf_strategy_clean
  PY=~/etf-daily-output/python/envs/py312/bin/python
  $PY decision_pack/scripts/validate_garch_short_horizon_board.py \\
    --pack-dir ~/temp/decision_packs/20260708
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from decision_pack.scripts.generate_fat_tail_risk_report import (
    build_universe_codes,
    load_as_of,
    load_holdings_from_pack,
    warmup_start,
)
from decision_pack.src.config import load_decision_pack_config
from decision_pack.src.fat_tail_risk import FatTailConfig, fat_tail_config_from_raw
from decision_pack.src.garch_short_horizon_board import (
    GarchShortHorizonConfig,
    garch_short_horizon_config_from_raw,
)
from decision_pack.src.garch_short_horizon_validation import (
    format_garch_validation_report,
    per_symbol_garch_validation,
    run_garch_walk_forward_validation,
)
from decision_pack.src.tomorrow_digest import write_calibration_cache
from decision_pack.src.short_horizon_mlfam import ShortHorizonConfig, short_horizon_config_from_raw
from decision_pack.src.short_horizon_validation import (
    build_eval_dates,
    run_walk_forward_validation,
)
from decision_pack.src.symbol_trend_board import load_cluster_mapping_codes
from pipeline.price_adjustments import auto_qfq_adjust_close_panel
from pipeline.strategy_layer_audit import load_qlib_close
from runtime_paths import CLUSTER_MAPPING_SELECTED_TXT, QLIB_PROVIDER_URI

DEFAULT_HORIZONS: tuple[int, ...] = (5, 10, 20)


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
        description="Walk-forward validate GARCH short-horizon board metrics",
    )
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--eval-start", type=str, default="2024-01-01")
    parser.add_argument("--eval-end", type=str, default=None)
    parser.add_argument("--step-days", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=None,
        help="Validate multiple H (default 5 10 20)",
    )
    parser.add_argument("--compare-mlfam", action="store_true", default=True)
    parser.add_argument("--no-compare-mlfam", action="store_false", dest="compare_mlfam")
    parser.add_argument("--output-detail", type=Path, default=None)
    parser.add_argument("--output-report", type=Path, default=None)
    parser.add_argument(
        "--output-cache",
        type=Path,
        default=None,
        help="Write per-symbol calibration CSV (code,n_obs,brier,coverage_q05,coverage_q95)",
    )
    parser.add_argument("--cluster-mapping", type=Path, default=None)
    parser.add_argument("--provider-uri", type=Path, default=None)
    return parser


def _resolve_horizons(args: argparse.Namespace) -> tuple[int, ...]:
    if args.horizon is not None and args.horizons:
        raise SystemExit("[ERR] --horizon 与 --horizons 不能同时使用")
    if args.horizon is not None:
        return (int(args.horizon),)
    if args.horizons:
        return tuple(int(h) for h in args.horizons)
    return DEFAULT_HORIZONS


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pack_dir = args.pack_dir.expanduser().resolve()
    horizons = _resolve_horizons(args)

    pack_cfg = load_decision_pack_config()
    fat_cfg = fat_tail_config_from_raw(pack_cfg.raw.get("fat_tail"))
    board_cfg = garch_short_horizon_config_from_raw(pack_cfg.raw.get("garch_short_horizon"))
    sh_cfg = short_horizon_config_from_raw(pack_cfg.raw.get("short_horizon"))

    as_of = load_as_of(pack_dir)
    defensive = frozenset(pack_cfg.defensive_codes)
    holdings = load_holdings_from_pack(pack_dir)
    cluster_path = Path(args.cluster_mapping or CLUSTER_MAPPING_SELECTED_TXT)
    cluster_codes = load_cluster_mapping_codes(cluster_path) if cluster_path.exists() else ()
    universe = build_universe_codes(holdings, cluster_codes, defensive)

    from decision_pack.scripts.generate_fat_tail_risk_report import load_fat_tail_data

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

    max_h = max(horizons)
    eval_end = (
        pd.Timestamp(args.eval_end)
        if args.eval_end
        else panel.index[-1 - max_h]
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

    all_summaries: list[tuple[int, object, pd.DataFrame]] = []
    for horizon in horizons:
        detail, summary = run_garch_walk_forward_validation(
            panel,
            eval_dates=eval_dates,
            horizon=horizon,
            cfg=board_cfg,
            flags_by_code=flags_map,
        )
        per_sym = per_symbol_garch_validation(detail)

        mlfam_brier: float | None = None
        if args.compare_mlfam:
            mlfam_cfg = ShortHorizonConfig(**{**sh_cfg.__dict__, "horizon": horizon})
            _, mlfam_summary = run_walk_forward_validation(
                panel,
                eval_dates=eval_dates,
                cfg=mlfam_cfg,
                flags_by_code=flags_map,
            )
            if mlfam_summary.n_obs > 0:
                mlfam_brier = mlfam_summary.brier_score

        single = len(horizons) == 1
        report_path = (
            args.output_report.expanduser().resolve()
            if single and args.output_report
            else pack_dir / f"GARCH_SHORT_HORIZON_VALIDATION_h{horizon}.md"
        )
        detail_path = (
            args.output_detail.expanduser().resolve()
            if single and args.output_detail
            else pack_dir / f"garch_short_horizon_validation_detail_h{horizon}.csv"
        )

        report = format_garch_validation_report(
            summary,
            pack_dir=str(pack_dir),
            as_of_board=as_of,
            per_symbol=per_sym,
            step_days=max(1, args.step_days),
            mlfam_brier=mlfam_brier,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        if not detail.empty:
            detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

        if args.output_cache is not None:
            cache_path = args.output_cache.expanduser().resolve()
            if len(horizons) > 1:
                stem = cache_path.stem
                cache_path = cache_path.with_name(f"{stem}_h{horizon}{cache_path.suffix}")
            write_calibration_cache(
                per_sym,
                cache_path,
                model="garch",
                horizon=horizon,
            )
            print(f"[OK] calibration cache h{horizon} -> {cache_path}")

        print(f"[OK] validation h{horizon} report -> {report_path}")
        print(f"[OK] detail h{horizon}         -> {detail_path} ({len(detail)} rows)")
        mlfam_note = f" vs MLFAM Brier={mlfam_brier:.4f}" if mlfam_brier is not None else ""
        print(
            f"[SUMMARY h{horizon}] obs={summary.n_obs} symbols={summary.n_symbols} "
            f"Brier={summary.brier_score:.4f}{mlfam_note} "
            f"cov={summary.coverage_q05:.1%}/{summary.coverage_q50:.1%}/{summary.coverage_q95:.1%} "
            f"VaR95_breach={summary.var95_breach_rate:.1%}"
        )
        all_summaries.append((horizon, summary, detail))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
