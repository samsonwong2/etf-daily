#!/usr/bin/env python3
"""Generate buyable topk from pack CSVs (trend / fat-tail / jump gates + tomorrow pick_score).

Examples:
  cd ~/gitee/skfolio_csi300/etf_strategy_clean
  PY=~/etf-daily-output/python/envs/py312/bin/python

  # Default strict gates + pack tomorrow pick_score
  $PY src/etf_daily/scripts/generate_buyable_rank.py \\
    --pack-dir ~/temp/decision_packs/20260717

  # Use walk-forward best strategy (gate / rank_rule / k / min_cred)
  $PY src/etf_daily/scripts/generate_buyable_rank.py \\
    --pack-dir ~/temp/decision_packs/20260717 \\
    --from-best-strategy ~/temp/decision_packs/20260717/buyable_rank_backtest/best_strategy.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.lib.buyable_rank import (
    BUYABLE_COLUMNS,
    BUYABLE_RANK_REPORT,
    BUYABLE_TOPK_CSV,
    GateName,
    build_buyable_frame,
    buyable_rank_config_from_raw,
    format_buyable_rank_report,
    gate_to_buyable_config,
    load_best_strategy_spec,
    load_pack_buyable_inputs,
    recompute_tomorrow_topk_scores,
)
from etf_daily.lib.config import load_decision_pack_config
from etf_daily.lib.tomorrow_digest import RankRule

BUYABLE_TOPK_BEST_CSV = "buyable_topk_picks_best.csv"
BUYABLE_RANK_BEST_REPORT = "BUYABLE_RANK_BEST.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Buyable rank: hard-filter tomorrow topk by trend/risk/jump",
    )
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--topk-k", type=int, default=None, help="Equal-weight top K")
    parser.add_argument(
        "--include-avoid",
        action="store_true",
        help="Do not exclude fat-tail avoid legs",
    )
    parser.add_argument(
        "--gate",
        type=str,
        default=None,
        choices=["strict", "no_path_gate", "core_risk", "positive_score", "none"],
        help="Hard-filter preset (overrides default.yaml buyable_rank flags)",
    )
    parser.add_argument(
        "--rank-rule",
        type=str,
        default=None,
        choices=["upside_x_cred", "dir_x_cred", "composite"],
        help="Recompute pick_score from direction_bias/cred/upside before ranking",
    )
    parser.add_argument(
        "--min-credibility",
        type=float,
        default=None,
        help="Min combined_cred for equal-weight suggestion (default 0)",
    )
    parser.add_argument(
        "--from-best-strategy",
        type=Path,
        default=None,
        help="Load gate/rank_rule/k/min_credibility from best_strategy.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pack_dir = args.pack_dir.expanduser().resolve()

    pack_cfg = load_decision_pack_config()
    cfg = buyable_rank_config_from_raw(pack_cfg.raw.get("buyable_rank"))
    rank_rule: RankRule | None = None
    gate: GateName | None = None
    strategy_name: str | None = None
    used_best = False

    if args.from_best_strategy is not None:
        used_best = True
        best = load_best_strategy_spec(args.from_best_strategy)
        strategy_name = best["strategy"] or None
        gate = best["gate"]  # type: ignore[assignment]
        rank_rule = best["rank_rule"]  # type: ignore[assignment]
        cfg = gate_to_buyable_config(gate, k=int(best["k"]))
        cfg = replace(cfg, min_credibility=float(best["min_credibility"]))

    if args.gate is not None:
        gate = args.gate  # type: ignore[assignment]
        k_for_gate = int(args.topk_k) if args.topk_k is not None else int(cfg.k)
        prev_min_cred = float(cfg.min_credibility)
        cfg = gate_to_buyable_config(gate, k=k_for_gate)
        cfg = replace(cfg, min_credibility=prev_min_cred)
    if args.rank_rule is not None:
        rank_rule = args.rank_rule  # type: ignore[assignment]
    if args.topk_k is not None:
        cfg = replace(cfg, k=int(args.topk_k))
    if args.min_credibility is not None:
        cfg = replace(cfg, min_credibility=float(args.min_credibility))
    if args.include_avoid:
        cfg = replace(cfg, exclude_avoid_leg=False)

    inputs = load_pack_buyable_inputs(pack_dir, cfg=cfg)
    tomorrow = inputs["tomorrow_topk"]  # type: ignore[assignment]
    if rank_rule is not None:
        tomorrow = recompute_tomorrow_topk_scores(tomorrow, rank_rule=rank_rule)

    frame = build_buyable_frame(
        tomorrow,
        inputs["trend"],  # type: ignore[arg-type]
        inputs["pool_risk"],  # type: ignore[arg-type]
        inputs["jump"],  # type: ignore[arg-type]
        cfg=cfg,
    )

    if args.output is not None:
        out_csv = args.output.expanduser().resolve()
    elif used_best:
        out_csv = pack_dir / BUYABLE_TOPK_BEST_CSV
    else:
        out_csv = pack_dir / BUYABLE_TOPK_CSV

    if args.report_output is not None:
        out_report = args.report_output.expanduser().resolve()
    elif used_best:
        out_report = pack_dir / BUYABLE_RANK_BEST_REPORT
    else:
        out_report = pack_dir / BUYABLE_RANK_REPORT

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.reindex(columns=list(BUYABLE_COLUMNS)).to_csv(
        out_csv, index=False, encoding="utf-8-sig"
    )

    as_of = "—"
    if not frame.empty and "as_of" in frame.columns:
        vals = frame["as_of"].dropna()
        if not vals.empty:
            as_of = str(vals.iloc[0])
    pool_path = inputs.get("pool_risk_path")
    pool_name = Path(pool_path).name if pool_path else None
    report = format_buyable_rank_report(
        frame,
        as_of=as_of,
        pack_dir=str(pack_dir),
        pool_risk_source=pool_name,
        cfg=cfg,
        rank_rule=rank_rule,
        gate=gate,
        strategy_name=strategy_name,
    )
    out_report.write_text(report, encoding="utf-8")

    n_pass = int(frame["pass_filters"].sum()) if not frame.empty else 0
    n_total = len(frame)
    meta: dict[str, Any] = {
        "as_of": as_of,
        "gate": gate,
        "rank_rule": rank_rule,
        "k": cfg.k,
        "min_credibility": cfg.min_credibility,
        "strategy": strategy_name,
        "passed": n_pass,
        "total": n_total,
        "csv": str(out_csv),
        "report": str(out_report),
    }
    print(
        f"[OK] buyable rank -> {out_csv} "
        f"(passed={n_pass}/{n_total}, as_of={as_of})",
        flush=True,
    )
    if gate or rank_rule or strategy_name:
        print(f"[OK] params       -> {json.dumps(meta, ensure_ascii=False)}", flush=True)
    print(f"[OK] report       -> {out_report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
