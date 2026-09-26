#!/usr/bin/env python3
"""Generate GARCH price targets CSV (TP/SL/median) from decision_pack artifacts.

Example:
  cd ~/gitee/skfolio_csi300/etf_strategy_clean
  PY=~/etf-daily-output/python/envs/py312/bin/python
  $PY src/etf_daily/scripts/generate_garch_price_targets.py \\
    --pack-dir ~/temp/decision_packs/20260714

Output: {pack_dir}/garch_price_targets.csv (long table, H=1/5/10/20 per symbol)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.scripts.generate_garch_short_horizon_board import (
    build_garch_short_horizon_board,
)
from etf_daily.lib.config import load_decision_pack_config
from etf_daily.lib.fat_tail_risk import fat_tail_config_from_raw
from etf_daily.lib.garch_price_targets import (
    GARCH_PRICE_TARGET_CSV,
    build_garch_price_targets_frame,
)
from etf_daily.lib.garch_short_horizon_board import garch_short_horizon_config_from_raw


def write_garch_price_targets(
    pack_dir: Path,
    *,
    cluster_mapping_path: Path | None = None,
    provider_uri: str | Path | None = None,
    output: Path | None = None,
) -> tuple[str, pd.DataFrame, int]:
    pack_cfg = load_decision_pack_config()
    fat_cfg = fat_tail_config_from_raw(pack_cfg.raw.get("fat_tail"))
    board_cfg = garch_short_horizon_config_from_raw(pack_cfg.raw.get("garch_short_horizon"))

    (
        as_of,
        _csv_by_h,
        _report_by_h,
        live_close_count,
        _jump_diag,
        cluster_frame,
        panel,
        fit_cache,
        board_cfg,
    ) = build_garch_short_horizon_board(
        pack_dir=pack_dir,
        cluster_mapping_path=cluster_mapping_path,
        provider_uri=provider_uri,
        fat_cfg=fat_cfg,
        board_cfg=board_cfg,
        horizons=board_cfg.horizons,
    )

    pt_horizons = board_cfg.price_target_horizons
    max_h = max(
        max(board_cfg.horizons) if board_cfg.horizons else 20,
        max(pt_horizons) if pt_horizons else 20,
    )
    frame = build_garch_price_targets_frame(
        cluster_frame,
        panel,
        as_of,
        board_cfg=board_cfg,
        fit_cache=fit_cache,
        horizons=pt_horizons,
        max_horizon=max_h,
    )
    out_path = output or (pack_dir / GARCH_PRICE_TARGET_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False, encoding="utf-8-sig")
    return as_of, frame, live_close_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate GARCH price targets CSV (TP/SL/median per H)",
    )
    parser.add_argument(
        "--pack-dir",
        type=Path,
        required=True,
        help="decision_pack generate 输出目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"输出 CSV 路径（默认 pack-dir/{GARCH_PRICE_TARGET_CSV}）",
    )
    parser.add_argument(
        "--cluster-mapping",
        type=Path,
        default=None,
        help="cluster_mapping_selected.txt（默认 runtime_paths）",
    )
    parser.add_argument(
        "--provider-uri",
        type=Path,
        default=None,
        help="qlib provider URI override",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pack_dir = args.pack_dir.expanduser().resolve()
    as_of, frame, live_count = write_garch_price_targets(
        pack_dir,
        cluster_mapping_path=args.cluster_mapping,
        provider_uri=args.provider_uri,
        output=args.output,
    )
    out_path = args.output or (pack_dir / GARCH_PRICE_TARGET_CSV)
    print(
        f"[OK] garch price targets -> {out_path} "
        f"({len(frame)} rows, as_of={as_of}, live_closes={live_count})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
