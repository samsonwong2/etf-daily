#!/usr/bin/env python3
"""Generate tomorrow (H=1) candidate-pool digest CSV + report.

Sidecar to GARCH/HMM short-horizon boards; does not modify their CSV columns.

Example:
  cd ~/gitee/skfolio_csi300/etf_strategy_clean
  PY=~/etf-daily-output/python/envs/py312/bin/python
  $PY decision_pack/scripts/generate_tomorrow_digest.py \\
    --pack-dir ~/temp/decision_packs/20260715

Weekly calibration refresh (recommended):
  $PY decision_pack/scripts/validate_garch_short_horizon_board.py \\
    --pack-dir ~/temp/decision_packs/20260715 --horizon 1 \\
    --output-cache ~/temp/decision_packs/calibration/garch_h1_per_symbol.csv
  $PY decision_pack/scripts/validate_hmm_short_horizon_board.py \\
    --pack-dir ~/temp/decision_packs/20260715 --horizon 1 \\
    --output-cache ~/temp/decision_packs/calibration/hmm_h1_per_symbol.csv
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from decision_pack.scripts.generate_fat_tail_risk_report import (
    build_universe_codes,
    load_fat_tail_data,
    load_holdings_from_pack,
    warmup_start,
)
from decision_pack.src.config import load_decision_pack_config
from decision_pack.src.fat_tail_risk import fat_tail_config_from_raw
from decision_pack.src.garch_short_horizon_board import garch_short_horizon_config_from_raw
from decision_pack.src.hmm_short_horizon import hmm_short_horizon_config_from_raw
from decision_pack.src.pack_live_prices import (
    apply_pack_live_closes,
    load_live_close_from_pack,
)
from decision_pack.src.symbol_trend_board import load_cluster_mapping_codes
from decision_pack.src.tomorrow_digest import (
    TOMORROW_DIGEST_COLUMNS,
    TOMORROW_TOPK_COLUMNS,
    build_tomorrow_digest_frame,
    format_tomorrow_digest_report,
    load_calibration_cache,
    resolve_cache_path,
    select_topk_picks,
    tomorrow_digest_config_from_raw,
)
from pipeline.price_adjustments import auto_qfq_adjust_close_panel
from pipeline.strategy_layer_audit import load_qlib_close
from runtime_paths import CLUSTER_MAPPING_SELECTED_TXT, QLIB_PROVIDER_URI

TOMORROW_DIGEST_CSV = "tomorrow_candidates_digest.csv"
TOMORROW_TOPK_CSV = "tomorrow_topk_picks.csv"
TOMORROW_DIGEST_REPORT = "TOMORROW_DIGEST.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate H=1 tomorrow candidate digest (GARCH+HMM merged)",
    )
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help="Directory with garch_h1_per_symbol.csv and hmm_h1_per_symbol.csv",
    )
    parser.add_argument("--garch-cal", type=Path, default=None, help="Override GARCH calibration CSV")
    parser.add_argument("--hmm-cal", type=Path, default=None, help="Override HMM calibration CSV")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--cluster-mapping", type=Path, default=None)
    parser.add_argument("--provider-uri", type=Path, default=None)
    parser.add_argument(
        "--topk-k",
        type=int,
        default=None,
        help="Override topk.k from config (5–10 typical)",
    )
    parser.add_argument(
        "--no-topk",
        action="store_true",
        help="Skip tomorrow_topk_picks.csv generation",
    )
    return parser


def _default_calibration_dir(pack_dir: Path) -> Path:
    return pack_dir.parent / "calibration"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pack_dir = args.pack_dir.expanduser().resolve()

    pack_cfg = load_decision_pack_config()
    fat_cfg = fat_tail_config_from_raw(pack_cfg.raw.get("fat_tail"))
    digest_cfg = tomorrow_digest_config_from_raw(pack_cfg.raw.get("tomorrow_digest"))
    if args.no_topk:
        digest_cfg = replace(digest_cfg, topk=replace(digest_cfg.topk, enabled=False))
    elif args.topk_k is not None:
        digest_cfg = replace(digest_cfg, topk=replace(digest_cfg.topk, k=int(args.topk_k)))
    garch_cfg = garch_short_horizon_config_from_raw(pack_cfg.raw.get("garch_short_horizon"))
    hmm_cfg = hmm_short_horizon_config_from_raw(pack_cfg.raw.get("hmm_short_horizon"))

    cluster_path = Path(args.cluster_mapping or CLUSTER_MAPPING_SELECTED_TXT)
    defensive = frozenset(pack_cfg.defensive_codes)
    as_of, _, cluster_frame, _ = load_fat_tail_data(
        pack_dir,
        cluster_mapping_path=cluster_path,
        provider_uri=args.provider_uri,
        defensive_codes=defensive,
        cfg=fat_cfg,
    )

    holdings = load_holdings_from_pack(pack_dir)
    cluster_codes = load_cluster_mapping_codes(cluster_path) if cluster_path.exists() else ()
    universe = build_universe_codes(holdings, cluster_codes, defensive)
    uri = str(args.provider_uri or QLIB_PROVIDER_URI)
    start = warmup_start(as_of, fat_cfg)
    panel = load_qlib_close(uri, universe, start, as_of)
    panel, _ = auto_qfq_adjust_close_panel(panel)
    live_closes = load_live_close_from_pack(pack_dir)
    panel, live_stats = apply_pack_live_closes(panel, as_of, live_closes)

    cal_dir = (
        args.calibration_dir.expanduser().resolve()
        if args.calibration_dir
        else _default_calibration_dir(pack_dir)
    )
    horizon = int(digest_cfg.horizon)
    garch_cal_path = (
        args.garch_cal.expanduser().resolve()
        if args.garch_cal
        else resolve_cache_path(cal_dir, "garch", horizon)
    )
    hmm_cal_path = (
        args.hmm_cal.expanduser().resolve()
        if args.hmm_cal
        else resolve_cache_path(cal_dir, "hmm", horizon)
    )
    garch_cal = load_calibration_cache(garch_cal_path)
    hmm_cal = load_calibration_cache(hmm_cal_path)
    pooled_warning = garch_cal.empty and hmm_cal.empty

    digest_frame = build_tomorrow_digest_frame(
        cluster_frame,
        panel,
        as_of,
        digest_cfg=digest_cfg,
        garch_board_cfg=garch_cfg,
        hmm_board_cfg=hmm_cfg,
        garch_cal=garch_cal,
        hmm_cal=hmm_cal,
    )

    out_csv = args.output.expanduser().resolve() if args.output else pack_dir / TOMORROW_DIGEST_CSV
    out_report = (
        args.report_output.expanduser().resolve()
        if args.report_output
        else pack_dir / TOMORROW_DIGEST_REPORT
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    digest_frame.reindex(columns=list(TOMORROW_DIGEST_COLUMNS)).to_csv(
        out_csv, index=False, encoding="utf-8-sig"
    )

    topk_frame = pd.DataFrame(columns=list(TOMORROW_TOPK_COLUMNS))
    out_topk = pack_dir / TOMORROW_TOPK_CSV
    if digest_cfg.topk.enabled:
        topk_frame = select_topk_picks(
            digest_frame,
            cluster_frame,
            digest_cfg=digest_cfg,
        )
        topk_frame.to_csv(out_topk, index=False, encoding="utf-8-sig")

    report = format_tomorrow_digest_report(
        digest_frame,
        as_of=as_of,
        pack_dir=str(pack_dir),
        pooled_warning=pooled_warning,
        topk_frame=topk_frame if digest_cfg.topk.enabled else None,
        topk_cfg=digest_cfg.topk,
    )
    out_report.write_text(report, encoding="utf-8")

    n_cand = len(digest_frame)
    print(
        f"[OK] tomorrow digest -> {out_csv} "
        f"({n_cand} candidates, as_of={as_of}, "
        f"live_prices={live_stats.fetched}, injected={live_stats.injected}, "
        f"skipped_same={live_stats.skipped_same})"
    )
    if digest_cfg.topk.enabled:
        print(
            f"[OK] topk picks         -> {out_topk} "
            f"({len(topk_frame)} ranked, equal-weight ref k={digest_cfg.topk.k})"
        )
    print(f"[OK] report            -> {out_report}")
    if pooled_warning:
        print(
            "[WARN] No H=1 calibration cache found; using pooled median. "
            f"Expected: {garch_cal_path} and {hmm_cal_path}",
            file=sys.stderr,
        )
    elif garch_cal.empty or hmm_cal.empty:
        missing = []
        if garch_cal.empty:
            missing.append(str(garch_cal_path))
        if hmm_cal.empty:
            missing.append(str(hmm_cal_path))
        print(f"[WARN] Partial calibration cache missing: {', '.join(missing)}", file=sys.stderr)

    assert list(digest_frame.reindex(columns=list(TOMORROW_DIGEST_COLUMNS)).columns) == list(
        TOMORROW_DIGEST_COLUMNS
    ) or digest_frame.empty
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
