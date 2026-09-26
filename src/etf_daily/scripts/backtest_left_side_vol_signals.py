#!/usr/bin/env python3
"""Backtest left-side vol signals (compression × spike × touch-down) from GARCH H=5 board."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.scripts.backtest_garch_topk_selection import (
    build_metrics_board,
    max_drawdown,
    period_simple_return,
    select_codes,
)
from etf_daily.scripts.generate_fat_tail_risk_report import (
    build_universe_codes,
    load_as_of,
    load_fat_tail_data,
    load_holdings_from_pack,
    warmup_start,
)
from etf_daily.scripts.validate_garch_short_horizon_board import _flags_by_code
from etf_daily.lib.config import load_decision_pack_config
from etf_daily.lib.fat_tail_risk import fat_tail_config_from_raw
from etf_daily.lib.garch_short_horizon_board import (
    GarchShortHorizonConfig,
    GarchShortHorizonMetrics,
    garch_short_horizon_config_from_raw,
)
from etf_daily.lib.left_side_digest import compute_compression, compute_left_score
from etf_daily.lib.short_horizon_validation import build_eval_dates
from etf_daily.lib.symbol_trend_board import load_cluster_mapping_codes
from etf_daily.lib.tomorrow_digest import credibility_today
from etf_daily.pipeline.price_adjustments import auto_qfq_adjust_close_panel
from etf_daily.pipeline.strategy_layer_audit import load_qlib_close
from etf_daily.paths import CLUSTER_MAPPING_SELECTED_TXT, QLIB_PROVIDER_URI


@dataclass(frozen=True)
class PickRule:
    name: str
    description: str
    score_fn: Callable[[GarchShortHorizonMetrics], float]
    ascending: bool = False
    filter_fn: Callable[[str, pd.Series], bool] | None = None


def _safe(x: float | None, default: float = float("-inf")) -> float:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return default
    return float(x)


def _g_cred(m: GarchShortHorizonMetrics) -> float:
    return credibility_today(
        p_up=m.p_up,
        p_down=m.p_down,
        reliable=m.reliable,
        cal_quality=1.0,
    )


def build_left_rules(*, max_vol5_pct: float = 0.35) -> list[PickRule]:
    def _compression_x_spike(m: GarchShortHorizonMetrics) -> float:
        comp = compute_compression(m.vol5_pct_120d)
        if comp is None or m.p_vol_spike_h5 is None:
            return float("-inf")
        return comp * float(m.p_vol_spike_h5)

    def _compression_x_spike_x_down(m: GarchShortHorizonMetrics) -> float:
        score = compute_left_score(
            compute_compression(m.vol5_pct_120d),
            m.p_vol_spike_h5,
            m.p_down,
            _g_cred(m),
            rank_rule="compression_x_spike_x_down",
        )
        return score if np.isfinite(score) else float("-inf")

    return [
        PickRule(
            "compression_x_spike",
            "压缩度 × spike 概率",
            _compression_x_spike,
        ),
        PickRule(
            "compression_x_spike_x_down",
            "压缩 × spike × 触轨↓ × 可信度",
            _compression_x_spike_x_down,
        ),
        PickRule(
            "low_vol5_pct",
            "5日波动历史分位最低",
            lambda m: -float(m.vol5_pct_120d)
            if m.vol5_pct_120d is not None
            else float("-inf"),
            ascending=True,
        ),
        PickRule(
            "high_p_spike",
            "波动放大概率最高",
            lambda m: _safe(m.p_vol_spike_h5, float("-inf")),
        ),
    ]


def build_left_rules_filtered(*, max_vol5_pct: float = 0.35) -> list[PickRule]:
    return build_left_rules(max_vol5_pct=max_vol5_pct)


def run_backtest(
    panel: pd.DataFrame,
    eval_dates: pd.DatetimeIndex,
    *,
    horizon: int,
    k: int,
    cfg: GarchShortHorizonConfig,
    cluster_frame: pd.DataFrame,
    flags_map: dict[str, tuple[str, ...]],
    rules: list[PickRule],
    rng: np.random.Generator,
    max_vol5_pct: float = 0.35,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    period_rows: list[dict] = []
    picks_rows: list[dict] = []

    for i, start in enumerate(eval_dates[:-1]):
        end = eval_dates[i + 1]
        board = build_metrics_board(
            panel,
            pd.Timestamp(start),
            horizon=horizon,
            cfg=cfg,
            cluster_frame=cluster_frame,
            flags_map=flags_map,
        )
        if board.empty:
            continue
        vol_mask = [
            r.metrics.vol5_pct_120d is not None
            and float(r.metrics.vol5_pct_120d) <= max_vol5_pct
            for r in board.itertuples()
        ]
        board = board.loc[vol_mask]
        universe = board["code"].tolist()
        if len(universe) < k:
            continue

        bench: dict[str, list[str]] = {
            "random_k": list(rng.choice(universe, size=k, replace=False)),
        }

        for rule in rules:
            sel = select_codes(board, rule, k)
            if len(sel) < max(1, min(k, 3)):
                continue
            bench[f"rule:{rule.name}"] = sel

        for label, codes in bench.items():
            rets = []
            for code in codes:
                r = period_simple_return(panel[code], pd.Timestamp(start), pd.Timestamp(end))
                if r is not None:
                    rets.append(r)
            if not rets:
                continue
            port_ret = float(np.mean(rets))
            period_rows.append(
                {
                    "start": start.strftime("%Y-%m-%d"),
                    "end": end.strftime("%Y-%m-%d"),
                    "H": horizon,
                    "k": k,
                    "strategy": label,
                    "n_picks": len(codes),
                    "port_ret": port_ret,
                }
            )
            if label.startswith("rule:"):
                picks_rows.append(
                    {
                        "start": start.strftime("%Y-%m-%d"),
                        "H": horizon,
                        "k": k,
                        "rule": label.replace("rule:", ""),
                        "codes": "|".join(codes),
                    }
                )

    return pd.DataFrame(period_rows), pd.DataFrame(picks_rows)


def summarize(periods: pd.DataFrame) -> pd.DataFrame:
    if periods.empty:
        return periods
    rows = []
    for (h, k, strat), g in periods.groupby(["H", "k", "strategy"]):
        rets = g["port_ret"]
        rows.append(
            {
                "H": h,
                "k": k,
                "strategy": strat,
                "n_periods": len(g),
                "cum_ret": float((1 + rets).prod() - 1),
                "mean_period_ret": float(rets.mean()),
                "hit_rate": float((rets > 0).mean()),
                "max_dd": max_drawdown(rets),
            }
        )
    return pd.DataFrame(rows).sort_values(["H", "cum_ret"], ascending=[True, False])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest left-side GARCH H=5 vol signals")
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--eval-start", type=str, default="2024-06-01")
    parser.add_argument("--eval-end", type=str, default=None)
    parser.add_argument("--step-days", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--max-vol5-pct", type=float, default=0.35)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--output-periods", type=Path, default=None)
    args = parser.parse_args(argv)

    pack_dir = args.pack_dir.expanduser().resolve()
    pack_cfg = load_decision_pack_config()
    fat_cfg = fat_tail_config_from_raw(pack_cfg.raw.get("fat_tail"))
    board_cfg = garch_short_horizon_config_from_raw(pack_cfg.raw.get("garch_short_horizon"))

    as_of = load_as_of(pack_dir)
    holdings = load_holdings_from_pack(pack_dir)
    cluster_path = Path(CLUSTER_MAPPING_SELECTED_TXT)
    cluster_codes = load_cluster_mapping_codes(cluster_path) if cluster_path.exists() else ()
    defensive = frozenset(pack_cfg.defensive_codes)
    universe = build_universe_codes(holdings, cluster_codes, defensive)
    _, _, cluster_frame, _ = load_fat_tail_data(
        pack_dir,
        cluster_mapping_path=cluster_path,
        defensive_codes=defensive,
        cfg=fat_cfg,
    )
    flags_map = _flags_by_code(cluster_frame)

    start = warmup_start(as_of, fat_cfg)
    panel = load_qlib_close(str(QLIB_PROVIDER_URI), universe, start, as_of)
    panel, _ = auto_qfq_adjust_close_panel(panel)

    horizon = int(args.horizon)
    eval_end = pd.Timestamp(args.eval_end) if args.eval_end else panel.index[-1 - horizon]
    eval_dates = build_eval_dates(
        panel,
        eval_start=pd.Timestamp(args.eval_start),
        eval_end=eval_end,
        step_days=max(1, args.step_days),
    )
    rules = build_left_rules_filtered(max_vol5_pct=float(args.max_vol5_pct))
    rng = np.random.default_rng(42)

    print(f"[INFO] left-side backtest H={horizon} k={args.k} periods={len(eval_dates)-1}")
    periods, picks = run_backtest(
        panel,
        eval_dates,
        horizon=horizon,
        k=args.k,
        cfg=board_cfg,
        cluster_frame=cluster_frame,
        flags_map=flags_map,
        rules=rules,
        rng=rng,
        max_vol5_pct=float(args.max_vol5_pct),
    )
    summary = summarize(periods)
    print(summary.to_string(index=False))

    out_sum = args.output_summary or pack_dir / f"left_side_backtest_h{horizon}_summary.csv"
    out_per = args.output_periods or pack_dir / f"left_side_backtest_h{horizon}_periods.csv"
    summary.to_csv(out_sum, index=False, encoding="utf-8-sig")
    periods.to_csv(out_per, index=False, encoding="utf-8-sig")
    print(f"[OK] summary -> {out_sum}")
    print(f"[OK] periods -> {out_per}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
