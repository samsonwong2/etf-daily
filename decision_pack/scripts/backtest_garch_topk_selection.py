#!/usr/bin/env python3
"""Backtest equal-weight top-K picks from GARCH short-horizon board metrics."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
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
from decision_pack.src.garch_short_horizon_board import (
    GarchShortHorizonConfig,
    GarchShortHorizonMetrics,
    compute_garch_short_horizon_metrics,
    garch_short_horizon_config_from_raw,
)
from decision_pack.src.short_horizon_validation import build_eval_dates
from decision_pack.src.symbol_trend_board import load_cluster_mapping_codes
from pipeline.price_adjustments import auto_qfq_adjust_close_panel
from pipeline.strategy_layer_audit import load_qlib_close
from decision_pack.src.tomorrow_digest import credibility_today
from runtime_paths import CLUSTER_MAPPING_SELECTED_TXT, QLIB_PROVIDER_URI


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


def _g_dir(m: GarchShortHorizonMetrics) -> float:
    return _safe(m.p_up, 0) - _safe(m.p_down, 0)


def _g_cred(m: GarchShortHorizonMetrics) -> float:
    return credibility_today(
        p_up=m.p_up,
        p_down=m.p_down,
        reliable=m.reliable,
        cal_quality=1.0,
    )


def _upside_skew(m: GarchShortHorizonMetrics) -> float:
    return _safe(m.q95, -999) - abs(_safe(m.q05, 0))


def build_rules(*, horizon: int | None = None) -> list[PickRule]:
    rules = [
        PickRule("max_q50", "q50_H 最高（中位收益预期）", lambda m: _safe(m.q50, -999)),
        PickRule("max_p_up", "触轨↑ 最高", lambda m: _safe(m.p_up, -1)),
        PickRule("min_p_down", "触轨↓ 最低", lambda m: _safe(m.p_down, 999), ascending=True),
        PickRule("max_net_prob", "触轨↑−触轨↓ 最大", lambda m: _safe(m.p_up, 0) - _safe(m.p_down, 0)),
        PickRule("min_sigma_1d", "σ_garch_1d 最低（低波）", lambda m: _safe(m.sigma_garch_1d, 999), ascending=True),
        PickRule("max_cvar", "CVaR95 最高（尾风险最小）", lambda m: _safe(m.cvar95_cond, -999)),
        PickRule("upside_skew", "q95−|q05| 最大（上行偏度）", lambda m: _safe(m.q95, -999) - abs(_safe(m.q05, 0))),
        PickRule(
            "q50_minus_pdown",
            "q50 − 触轨↓（收益−下行概率）",
            lambda m: _safe(m.q50, -999) - 0.05 * _safe(m.p_down, 0),
        ),
        PickRule(
            "q50_low_timeout",
            "q50 高且到期概率低",
            lambda m: _safe(m.q50, -999) - 0.02 * _safe(m.p_timeout, 0),
        ),
        PickRule(
            "risk_leg_max_q50",
            "risk_leg 内 q50 最高",
            lambda m: _safe(m.q50, -999),
            filter_fn=lambda code, row: str(row.get("barbell_leg") or "") == "risk_leg",
        ),
        PickRule(
            "ex_safe_max_q50",
            "排除 safe_leg 后 q50 最高",
            lambda m: _safe(m.q50, -999),
            filter_fn=lambda code, row: str(row.get("barbell_leg") or "") not in ("safe_leg",),
        ),
        PickRule(
            "avoid_max_q50",
            "avoid 腿 q50 最高",
            lambda m: _safe(m.q50, -999),
            filter_fn=lambda code, row: str(row.get("barbell_leg") or "") == "avoid",
        ),
        PickRule(
            "low_vol_ratio_q50",
            "vol_ratio 低 + q50 高",
            lambda m: _safe(m.q50, -999) - 0.01 * _safe(m.vol_ratio_5_20, 2),
        ),
    ]
    if horizon == 1:
        def _score_dir_x_cred(m: GarchShortHorizonMetrics) -> float:
            d = _g_dir(m)
            if d <= 0:
                return float("-inf")
            return d * _g_cred(m)

        def _score_upside_x_cred(m: GarchShortHorizonMetrics) -> float:
            if _g_dir(m) <= 0:
                return float("-inf")
            return _upside_skew(m) * _g_cred(m)

        def _score_composite(m: GarchShortHorizonMetrics) -> float:
            d = max(0.0, _g_dir(m))
            if d <= 0:
                return float("-inf")
            return d * _g_cred(m) * _upside_skew(m)

        rules.extend(
            [
                PickRule(
                    "dir_x_cred_h1",
                    "H=1: 方向偏×可信度（看涨过滤）",
                    _score_dir_x_cred,
                ),
                PickRule(
                    "upside_x_cred_h1",
                    "H=1: upside_skew×可信度（看涨过滤）",
                    _score_upside_x_cred,
                ),
                PickRule(
                    "composite_h1",
                    "H=1: max(0,方向偏)×可信度×upside_skew",
                    _score_composite,
                ),
            ]
        )
    return rules


def period_simple_return(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    s = series.dropna()
    s = s[(s.index >= start) & (s.index <= end)]
    if s.empty:
        return None
    p0 = float(s.iloc[0])
    p1 = float(s.iloc[-1])
    if p0 <= 0 or p1 <= 0:
        return None
    return p1 / p0 - 1.0


def build_metrics_board(
    panel: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    horizon: int,
    cfg: GarchShortHorizonConfig,
    cluster_frame: pd.DataFrame,
    flags_map: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    cluster_by_code = cluster_frame.set_index("code")
    rows: list[dict] = []
    for code in panel.columns:
        if code not in cluster_by_code.index:
            continue
        series = pd.to_numeric(panel[code], errors="coerce").dropna()
        if series.empty:
            continue
        metrics = compute_garch_short_horizon_metrics(
            series,
            as_of,
            horizon=horizon,
            cfg=cfg,
            flags=flags_map.get(str(code), ()),
            max_horizon=horizon,
        )
        if not metrics.reliable or metrics.q50 is None:
            continue
        crow = cluster_by_code.loc[code]
        rows.append({"code": str(code), "metrics": metrics, "cluster": crow})
    return pd.DataFrame(rows)


def select_codes(
    board: pd.DataFrame,
    rule: PickRule,
    k: int,
) -> list[str]:
    if board.empty:
        return []
    candidates = board
    if rule.filter_fn is not None:
        mask = [rule.filter_fn(str(r.code), r.cluster) for r in board.itertuples()]
        candidates = board.loc[mask]
    if candidates.empty:
        return []
    scores = candidates["metrics"].map(rule.score_fn)
    order = scores.sort_values(ascending=rule.ascending).index
    picked = candidates.loc[order].head(k)
    return picked["code"].tolist()


def max_drawdown(rets: pd.Series) -> float:
    eq = (1 + rets.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


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
        universe = board["code"].tolist()
        if len(universe) < k:
            continue

        # benchmarks
        bench: dict[str, list[str]] = {
            "all_equal": universe,
            "random_k": list(rng.choice(universe, size=k, replace=False)),
        }
        held = cluster_frame.loc[cluster_frame["weight"] > 0, "code"].astype(str).tolist()
        held = [c for c in held if c in universe]
        if len(held) >= 1:
            bench["holdings_equal"] = held

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
                "mean_next_day_ret": float(rets.mean()) if h == 1 else None,
                "hit_rate": float((rets > 0).mean()),
                "win_rate": float((rets > 0).mean()),
                "max_dd": max_drawdown(rets),
                "vol_ann": float(rets.std() * np.sqrt(252 / 5)) if rets.std() > 0 else 0.0,
                "sharpe_simple": (
                    float(rets.mean() / rets.std() * np.sqrt(252 / 5)) if rets.std() > 0 else 0.0
                ),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["H", "cum_ret"], ascending=[True, False])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest GARCH board top-K equal-weight picks")
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--eval-start", type=str, default="2024-06-01")
    parser.add_argument("--eval-end", type=str, default=None)
    parser.add_argument("--step-days", type=int, default=5)
    parser.add_argument("--horizons", type=int, nargs="+", default=None)
    parser.add_argument("--horizon", type=int, default=None, help="Single horizon (overrides --horizons)")
    parser.add_argument("--k", type=int, default=6, help="Number of picks (5-7)")
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

    horizons = (
        (int(args.horizon),)
        if args.horizon is not None
        else tuple(int(h) for h in (args.horizons or [5, 10, 20]))
    )
    max_h = max(horizons)
    eval_end = pd.Timestamp(args.eval_end) if args.eval_end else panel.index[-1 - max_h]
    rng = np.random.default_rng(42)

    all_periods: list[pd.DataFrame] = []
    all_picks: list[pd.DataFrame] = []
    for h in horizons:
        step_days = 1 if h == 1 else max(1, args.step_days)
        eval_dates = build_eval_dates(
            panel,
            eval_start=pd.Timestamp(args.eval_start),
            eval_end=eval_end,
            step_days=step_days,
        )
        rules = build_rules(horizon=h)
        print(f"[INFO] backtest H={h} k={args.k} step={step_days} periods={len(eval_dates)-1}")
        periods, picks = run_backtest(
            panel,
            eval_dates,
            horizon=h,
            k=args.k,
            cfg=board_cfg,
            cluster_frame=cluster_frame,
            flags_map=flags_map,
            rules=rules,
            rng=rng,
        )
        all_periods.append(periods)
        all_picks.append(picks)

    period_frame = pd.concat(all_periods, ignore_index=True)
    pick_frame = pd.concat(all_picks, ignore_index=True)
    summary = summarize(period_frame)

    out_summary = args.output_summary or (pack_dir / "garch_topk_selection_backtest_summary.csv")
    out_periods = args.output_periods or (pack_dir / "garch_topk_selection_backtest_periods.csv")
    summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
    period_frame.to_csv(out_periods, index=False, encoding="utf-8-sig")
    pick_frame.to_csv(pack_dir / "garch_topk_selection_backtest_picks.csv", index=False, encoding="utf-8-sig")

    print(f"[OK] summary -> {out_summary}")
    print(f"[OK] periods -> {out_periods}")
    for h in horizons:
        sub = summary[summary["H"] == h].head(8)
        print(f"\n=== H={h} top strategies ===")
        print(sub.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
