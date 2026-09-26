#!/usr/bin/env python3
"""Walk-forward backtest for garch_price_targets.csv (TP/SL/median accuracy).

Example:
  cd ~/gitee/skfolio_csi300/etf_strategy_clean
  PY=~/etf-daily-output/python/envs/py312/bin/python
  $PY decision_pack/scripts/validate_garch_price_targets.py \\
    --pack-dir ~/temp/decision_packs/20260714
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
    load_fat_tail_data,
    load_holdings_from_pack,
    warmup_start,
)
from decision_pack.scripts.validate_garch_short_horizon_board import _flags_by_code
from decision_pack.src.config import load_decision_pack_config
from decision_pack.src.fat_tail_risk import fat_tail_config_from_raw
from decision_pack.src.garch_price_targets_validation import (
    format_price_targets_validation_report,
    run_garch_price_targets_validation,
)
from decision_pack.src.garch_short_horizon_board import garch_short_horizon_config_from_raw
from decision_pack.src.short_horizon_validation import build_eval_dates
from decision_pack.src.symbol_trend_board import load_cluster_mapping_codes
from pipeline.price_adjustments import auto_qfq_adjust_close_panel
from pipeline.strategy_layer_audit import load_qlib_close
from runtime_paths import CLUSTER_MAPPING_SELECTED_TXT, QLIB_PROVIDER_URI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backtest GARCH price targets (TP/SL/median) walk-forward",
    )
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--eval-start", type=str, default="2024-01-01")
    parser.add_argument("--eval-end", type=str, default=None)
    parser.add_argument("--step-days", type=int, default=5)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=None,
        help="Default: price_target_horizons from config (1 5 10 20)",
    )
    parser.add_argument("--output-report", type=Path, default=None)
    parser.add_argument("--output-detail", type=Path, default=None)
    parser.add_argument("--cluster-mapping", type=Path, default=None)
    parser.add_argument("--provider-uri", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pack_dir = args.pack_dir.expanduser().resolve()

    pack_cfg = load_decision_pack_config()
    fat_cfg = fat_tail_config_from_raw(pack_cfg.raw.get("fat_tail"))
    board_cfg = garch_short_horizon_config_from_raw(pack_cfg.raw.get("garch_short_horizon"))
    horizons = tuple(
        int(h) for h in (args.horizons or board_cfg.price_target_horizons)
    )

    as_of = load_as_of(pack_dir)
    defensive = frozenset(pack_cfg.defensive_codes)
    holdings = load_holdings_from_pack(pack_dir)
    cluster_path = Path(args.cluster_mapping or CLUSTER_MAPPING_SELECTED_TXT)
    cluster_codes = load_cluster_mapping_codes(cluster_path) if cluster_path.exists() else ()
    universe = build_universe_codes(holdings, cluster_codes, defensive)

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
        pd.Timestamp(args.eval_end) if args.eval_end else panel.index[-1 - max_h]
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

    summaries: dict[int, object] = {}
    all_detail: list[pd.DataFrame] = []
    for horizon in horizons:
        detail, summary = run_garch_price_targets_validation(
            panel,
            eval_dates=eval_dates,
            horizon=int(horizon),
            cfg=board_cfg,
            flags_by_code=flags_map,
        )
        summaries[int(horizon)] = summary
        if not detail.empty:
            all_detail.append(detail)
        print(
            f"[H={horizon}] n={summary.n_obs} "
            f"pred_tp={summary.mean_pred_p_tp:.1%} act_tp={summary.actual_tp_rate:.1%} "
            f"pred_sl={summary.mean_pred_p_sl:.1%} act_sl={summary.actual_sl_rate:.1%} "
            f"med_cov={summary.median_coverage:.1%} "
            f"strat={summary.strategy_mean_simple_ret:.2%} bh={summary.buy_hold_mean_simple_ret:.2%} "
            f"Brier={summary.brier_score:.3f}"
        )

    report = format_price_targets_validation_report(
        summaries,
        pack_dir=str(pack_dir),
        as_of_board=as_of,
        step_days=max(1, args.step_days),
    )
    report_path = args.output_report or (pack_dir / "GARCH_PRICE_TARGETS_BACKTEST.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"[OK] report -> {report_path}")

    if all_detail:
        detail_frame = pd.concat(all_detail, ignore_index=True)
        detail_path = args.output_detail or (pack_dir / "garch_price_targets_backtest_detail.csv")
        detail_frame.to_csv(detail_path, index=False, encoding="utf-8-sig")
        print(f"[OK] detail -> {detail_path} ({len(detail_frame)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
