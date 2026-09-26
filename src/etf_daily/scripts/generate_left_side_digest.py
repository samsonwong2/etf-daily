#!/usr/bin/env python3
"""Generate left-side (H=5 vol compression + spike) digest CSV + report.

Example:
  cd ~/gitee/skfolio_csi300/etf_strategy_clean
  PY=~/etf-daily-output/python/envs/py312/bin/python
  $PY src/etf_daily/scripts/generate_left_side_digest.py \\
    --pack-dir ~/temp/decision_packs/20260716
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.scripts.generate_fat_tail_risk_report import (
    build_universe_codes,
    load_fat_tail_data,
    load_holdings_from_pack,
    warmup_start,
)
from etf_daily.lib.config import load_decision_pack_config
from etf_daily.lib.fat_tail_risk import fat_tail_config_from_raw
from etf_daily.lib.garch_short_horizon_board import garch_short_horizon_config_from_raw
from etf_daily.lib.hmm_short_horizon import hmm_short_horizon_config_from_raw
from etf_daily.lib.left_side_digest import (
    LEFT_SIDE_DIGEST_COLUMNS,
    LEFT_SIDE_TOPK_COLUMNS,
    REGIME_TRANSITION_CSV,
    build_left_side_digest_frame,
    format_left_side_digest_report,
    left_side_digest_config_from_raw,
    load_regime_transition_frame,
    select_left_topk_picks,
)
from etf_daily.lib.pack_live_prices import (
    apply_pack_live_closes,
    load_live_close_from_pack,
)
from etf_daily.lib.symbol_trend_board import load_cluster_mapping_codes
from etf_daily.lib.tomorrow_digest import load_calibration_cache, resolve_cache_path
from etf_daily.pipeline.price_adjustments import auto_qfq_adjust_close_panel
from etf_daily.pipeline.strategy_layer_audit import load_qlib_close
from etf_daily.paths import CLUSTER_MAPPING_SELECTED_TXT, QLIB_PROVIDER_URI

LEFT_SIDE_DIGEST_CSV = "left_side_candidates_digest.csv"
LEFT_SIDE_TOPK_CSV = "left_side_topk_picks.csv"
LEFT_SIDE_DIGEST_REPORT = "LEFT_SIDE_DIGEST.md"
GARCH_JUMP_DIAGNOSTICS_CSV = "garch_jump_diagnostics.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate H=5 left-side vol compression + spike digest",
    )
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, default=None)
    parser.add_argument("--garch-cal", type=Path, default=None)
    parser.add_argument("--hmm-cal", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--cluster-mapping", type=Path, default=None)
    parser.add_argument("--provider-uri", type=Path, default=None)
    parser.add_argument("--transition-csv", type=Path, default=None)
    parser.add_argument("--topk-k", type=int, default=None)
    parser.add_argument("--no-topk", action="store_true")
    return parser


def _default_calibration_dir(pack_dir: Path) -> Path:
    return pack_dir.parent / "calibration"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pack_dir = args.pack_dir.expanduser().resolve()

    pack_cfg = load_decision_pack_config()
    fat_cfg = fat_tail_config_from_raw(pack_cfg.raw.get("fat_tail"))
    digest_cfg = left_side_digest_config_from_raw(pack_cfg.raw.get("left_side_digest"))
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
    panel, _ = apply_pack_live_closes(panel, as_of, live_closes)

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

    transition_path = (
        args.transition_csv.expanduser().resolve()
        if args.transition_csv
        else pack_dir / REGIME_TRANSITION_CSV
    )
    transition_frame = load_regime_transition_frame(transition_path)

    digest_frame = build_left_side_digest_frame(
        cluster_frame,
        panel,
        as_of,
        digest_cfg=digest_cfg,
        garch_board_cfg=garch_cfg,
        hmm_board_cfg=hmm_cfg,
        garch_cal=garch_cal,
        hmm_cal=hmm_cal,
        transition_frame=transition_frame,
    )

    out_csv = args.output.expanduser().resolve() if args.output else pack_dir / LEFT_SIDE_DIGEST_CSV
    out_report = (
        args.report_output.expanduser().resolve()
        if args.report_output
        else pack_dir / LEFT_SIDE_DIGEST_REPORT
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    digest_frame.reindex(columns=list(LEFT_SIDE_DIGEST_COLUMNS)).to_csv(
        out_csv, index=False, encoding="utf-8-sig"
    )

    jump_path = pack_dir / GARCH_JUMP_DIAGNOSTICS_CSV
    jump_frame = (
        pd.read_csv(jump_path)
        if jump_path.exists()
        else pd.DataFrame()
    )

    topk_frame = pd.DataFrame(columns=list(LEFT_SIDE_TOPK_COLUMNS))
    out_topk = pack_dir / LEFT_SIDE_TOPK_CSV
    if digest_cfg.topk.enabled:
        topk_frame = select_left_topk_picks(
            digest_frame,
            cluster_frame,
            digest_cfg=digest_cfg,
            jump_frame=jump_frame if not jump_frame.empty else None,
        )
        topk_frame.to_csv(out_topk, index=False, encoding="utf-8-sig")

    report = format_left_side_digest_report(
        digest_frame,
        as_of=as_of,
        pack_dir=str(pack_dir),
        pooled_warning=pooled_warning,
        topk_frame=topk_frame if digest_cfg.topk.enabled else None,
        topk_cfg=digest_cfg.topk,
    )
    out_report.write_text(report, encoding="utf-8")
    print(f"[OK] left digest         -> {out_csv} ({len(digest_frame)} rows)")
    if digest_cfg.topk.enabled:
        print(f"[OK] left topk           -> {out_topk} ({len(topk_frame)} rows)")
    print(f"[OK] left report         -> {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
