#!/usr/bin/env python3
"""Walk-forward ranking accuracy for left_side_topk_picks (left_score)."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.regime.backtest_regime_transition_signals import structural_change_label
from etf_daily.scripts.generate_fat_tail_risk_report import (
    build_universe_codes,
    load_as_of,
    load_fat_tail_data,
    load_holdings_from_pack,
    warmup_start,
)
from etf_daily.lib.config import load_decision_pack_config
from etf_daily.lib.fat_tail_risk import fat_tail_config_from_raw
from etf_daily.lib.garch_short_horizon_board import (
    compute_garch_short_horizon_metrics,
    garch_short_horizon_config_from_raw,
)
from etf_daily.lib.garch_short_horizon_validation import forward_realized_vol_ann_log
from etf_daily.lib.hmm_short_horizon import hmm_short_horizon_config_from_raw
from etf_daily.lib.left_side_digest import (
    build_left_side_digest_frame,
    compute_left_score,
    left_side_digest_config_from_raw,
    select_left_topk_picks,
)
from etf_daily.lib.regime_transition_board import (
    compute_regime_transition_metrics,
    format_regime_transition_row,
    regime_transition_config_from_raw,
)
from etf_daily.lib.short_horizon_mlfam import forward_log_returns, realize_triple_barrier
from etf_daily.lib.short_horizon_validation import build_eval_dates
from etf_daily.lib.symbol_trend_board import load_cluster_mapping_codes
from etf_daily.pipeline.price_adjustments import auto_qfq_adjust_close_panel
from etf_daily.pipeline.strategy_layer_audit import load_qlib_close
from etf_daily.paths import CLUSTER_MAPPING_SELECTED_TXT, QLIB_PROVIDER_URI


def _forward_labels(
    close: pd.Series,
    as_of: pd.Timestamp,
    *,
    horizon: int,
    spike_mult: float,
    barrier: float | None,
    baseline_vol: float | None,
) -> dict[str, bool | None]:
    ts = pd.Timestamp(as_of)
    structural = structural_change_label(close, ts, horizon=horizon)

    fwd = forward_log_returns(close, ts, horizon=horizon)
    touch_down: bool | None = None
    if fwd is not None and barrier is not None and math.isfinite(float(barrier)):
        outcome = realize_triple_barrier(fwd, barrier=float(barrier))
        touch_down = outcome == "down"

    vol_spike: bool | None = None
    fwd_vol = forward_realized_vol_ann_log(close, ts, horizon=horizon)
    if (
        baseline_vol is not None
        and fwd_vol is not None
        and math.isfinite(float(baseline_vol))
        and math.isfinite(float(fwd_vol))
        and float(baseline_vol) > 0
    ):
        vol_spike = bool(float(fwd_vol) > float(baseline_vol) * float(spike_mult))

    return {
        "label_structural_change": structural,
        "label_vol_spike": vol_spike,
        "label_touch_down": touch_down,
    }


def _build_transition_frame(
    panel: pd.DataFrame,
    cluster_frame: pd.DataFrame,
    as_of: str,
    *,
    rt_cfg,
) -> pd.DataFrame:
    rows: list[dict] = []
    ts = pd.Timestamp(as_of)
    for _, row in cluster_frame.iterrows():
        code = str(row["code"])
        name = str(row.get("name", code))
        if code not in panel.columns:
            continue
        close = pd.to_numeric(panel[code], errors="coerce").dropna()
        if close.empty:
            continue
        available = close.loc[:ts]
        if available.empty:
            continue
        close_val = float(available.iloc[-1])
        metrics = compute_regime_transition_metrics(close, ts, cfg=rt_cfg)
        rows.append(
            format_regime_transition_row(
                code=code,
                name=name,
                close=close_val if math.isfinite(close_val) else None,
                as_of=as_of,
                metrics=metrics,
            )
        )
    return pd.DataFrame(rows)


def _precision_at_top(frame: pd.DataFrame, label_col: str, top_n: int) -> float | None:
    valid = frame.dropna(subset=["left_score", label_col])
    if valid.empty:
        return None
    top = valid.sort_values("left_score", ascending=False, kind="stable").head(top_n)
    if top.empty:
        return None
    return float(top[label_col].astype(float).mean())


def _spearman(frame: pd.DataFrame, label_col: str) -> float | None:
    valid = frame.dropna(subset=["left_score", label_col])
    if len(valid) < 3:
        return None
    rho, _ = stats.spearmanr(valid["left_score"].astype(float), valid[label_col].astype(float))
    if not math.isfinite(float(rho)):
        return None
    return float(rho)


def _score_maps_from_digest(digest: pd.DataFrame, rank_rule: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for _, row in digest.iterrows():
        code = str(row["code"])
        rank_cred = row.get("_combined_sort")
        if rank_rule == "transition_x_spike_x_down":
            t_cred = row.get("_transition_cred")
            if t_cred is not None and math.isfinite(float(t_cred)) and float(t_cred) > 0:
                rank_cred = float(t_cred)
        score = compute_left_score(
            row.get("_compression"),
            row.get("_p_spike"),
            row.get("_p_down"),
            rank_cred,
            rank_rule=rank_rule,  # type: ignore[arg-type]
            transition_score=row.get("_transition_score"),
        )
        if score is not None and math.isfinite(float(score)):
            scores[code] = float(score)
    return scores


def _daily_metrics(
    day_df: pd.DataFrame,
    *,
    as_of_s: str,
    top_k: int,
    top_pct: float,
    rank_rule: str,
) -> list[dict]:
    n = len(day_df)
    n_top_decile = max(1, int(math.ceil(n * float(top_pct))))
    base = {
        "as_of": as_of_s,
        "n_scored": n,
        "top_k": top_k,
        "top_decile_n": n_top_decile,
        "rank_rule": rank_rule,
    }
    rows: list[dict] = []
    for label_col in (
        "label_structural_change",
        "label_vol_spike",
        "label_touch_down",
    ):
        valid = day_df.dropna(subset=[label_col])
        base_rate = float(valid[label_col].astype(float).mean()) if len(valid) else float("nan")
        prec_k = _precision_at_top(day_df, label_col, top_k)
        prec_dec = _precision_at_top(day_df, label_col, n_top_decile)
        rho = _spearman(day_df, label_col)
        rows.append(
            {
                **base,
                "label": label_col.replace("label_", ""),
                "base_rate": base_rate,
                f"precision_top{top_k}": prec_k,
                "precision_top_decile": prec_dec,
                "spearman_rho": rho,
                "lift_top_decile": (
                    prec_dec / base_rate
                    if prec_dec is not None and base_rate and math.isfinite(base_rate) and base_rate > 0
                    else float("nan")
                ),
            }
        )
    return rows


def run_ranking_backtest(
    panel: pd.DataFrame,
    cluster_frame: pd.DataFrame,
    eval_dates: pd.DatetimeIndex,
    *,
    digest_cfg,
    garch_board_cfg,
    hmm_board_cfg,
    rt_cfg,
    jump_frame: pd.DataFrame | None,
    spike_mult: float,
    top_k: int,
    top_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signal_rows: list[dict] = []
    daily_rows: list[dict] = []

    horizon = int(digest_cfg.horizon)
    for as_of in eval_dates:
        as_of_s = pd.Timestamp(as_of).strftime("%Y-%m-%d")
        transition_frame = _build_transition_frame(
            panel, cluster_frame, as_of_s, rt_cfg=rt_cfg
        )
        digest = build_left_side_digest_frame(
            cluster_frame,
            panel,
            as_of_s,
            digest_cfg=digest_cfg,
            garch_board_cfg=garch_board_cfg,
            hmm_board_cfg=hmm_board_cfg,
            garch_cal=None,
            hmm_cal=None,
            transition_frame=transition_frame,
        )
        if digest.empty:
            continue

        topk = select_left_topk_picks(
            digest,
            cluster_frame,
            digest_cfg=digest_cfg,
            jump_frame=jump_frame,
        )
        if topk.empty:
            continue

        rank_map = topk.set_index("code")["rank"].to_dict()
        score_map = topk.set_index("code")["left_score"].to_dict()
        comp_score_map = _score_maps_from_digest(digest, "compression_x_spike_x_down")

        day_records: list[dict] = []
        cluster_idx = cluster_frame.set_index("code")
        for _, drow in digest.iterrows():
            code = str(drow["code"])
            if code not in panel.columns:
                continue
            close = pd.to_numeric(panel[code], errors="coerce").dropna()
            flag_tuple: tuple[str, ...] = ()
            if code in cluster_idx.index:
                flags = cluster_idx.loc[code].get("flags")
                if isinstance(flags, str) and flags:
                    flag_tuple = tuple(f for f in flags.split("|") if f)
            g_m = compute_garch_short_horizon_metrics(
                close,
                pd.Timestamp(as_of),
                horizon=horizon,
                cfg=garch_board_cfg,
                flags=flag_tuple,
            )
            labels = _forward_labels(
                close,
                pd.Timestamp(as_of),
                horizon=horizon,
                spike_mult=spike_mult,
                barrier=g_m.barrier,
                baseline_vol=g_m.vol_ann_5d,
            )
            left_score = score_map.get(code)
            if left_score is None:
                continue
            rec = {
                "as_of": as_of_s,
                "code": code,
                "name": drow["name"],
                "rank": rank_map.get(code),
                "left_score": left_score,
                "compression_score": comp_score_map.get(code),
                **labels,
            }
            day_records.append(rec)
            signal_rows.append(rec)

        if not day_records:
            continue
        day_df = pd.DataFrame(day_records)
        rank_rule = digest_cfg.topk.rank_rule
        daily_rows.extend(
            _daily_metrics(
                day_df.rename(columns={"left_score": "left_score"}),
                as_of_s=as_of_s,
                top_k=top_k,
                top_pct=top_pct,
                rank_rule=rank_rule,
            )
        )
        if comp_score_map:
            comp_day = day_df.copy()
            comp_day["left_score"] = comp_day["code"].map(comp_score_map)
            comp_day = comp_day.dropna(subset=["left_score"])
            if not comp_day.empty:
                daily_rows.extend(
                    _daily_metrics(
                        comp_day,
                        as_of_s=as_of_s,
                        top_k=top_k,
                        top_pct=top_pct,
                        rank_rule="compression_x_spike_x_down",
                    )
                )

    signals = pd.DataFrame(signal_rows)
    daily = pd.DataFrame(daily_rows)
    aggregate = _aggregate_daily(daily, top_k=top_k, top_pct=top_pct)
    return signals, daily, aggregate


def _aggregate_daily(daily: pd.DataFrame, *, top_k: int, top_pct: float) -> pd.DataFrame:
    if daily.empty:
        return daily
    rows = []
    for (label, rank_rule), g in daily.groupby(["label", "rank_rule"]):
        rows.append(
            {
                "label": label,
                "rank_rule": rank_rule,
                "n_eval_days": len(g),
                "top_k": top_k,
                "top_pct": top_pct,
                "mean_base_rate": float(g["base_rate"].mean()),
                f"mean_precision_top{top_k}": float(g[f"precision_top{top_k}"].mean()),
                "mean_precision_top_decile": float(g["precision_top_decile"].mean()),
                "median_precision_top_decile": float(g["precision_top_decile"].median()),
                "mean_spearman_rho": float(g["spearman_rho"].mean()),
                "mean_lift_top_decile": float(g["lift_top_decile"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backtest left_side_topk_picks ranking accuracy (walk-forward)",
    )
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--eval-start", type=str, default="2026-01-01")
    parser.add_argument("--eval-end", type=str, default="2026-07-16")
    parser.add_argument("--step-days", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=None, help="Override config topk.k")
    parser.add_argument("--top-pct", type=float, default=0.10)
    parser.add_argument("--output-signals", type=Path, default=None)
    parser.add_argument("--output-daily", type=Path, default=None)
    parser.add_argument("--output-aggregate", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pack_dir = args.pack_dir.expanduser().resolve()
    pack_cfg = load_decision_pack_config()
    fat_cfg = fat_tail_config_from_raw(pack_cfg.raw.get("fat_tail"))
    digest_cfg = left_side_digest_config_from_raw(pack_cfg.raw.get("left_side_digest"))
    garch_board_cfg = garch_short_horizon_config_from_raw(pack_cfg.raw.get("garch_short_horizon"))
    hmm_board_cfg = hmm_short_horizon_config_from_raw(pack_cfg.raw.get("hmm_short_horizon"))
    rt_cfg = regime_transition_config_from_raw(pack_cfg.raw.get("regime_transition"))
    spike_mult = float(garch_board_cfg.spike_mult)

    top_k = int(args.top_k) if args.top_k is not None else int(digest_cfg.topk.k)
    rank_rule = digest_cfg.topk.rank_rule

    as_of = load_as_of(pack_dir)
    holdings = load_holdings_from_pack(pack_dir)
    cluster_path = Path(CLUSTER_MAPPING_SELECTED_TXT)
    cluster_codes = load_cluster_mapping_codes(cluster_path) if cluster_path.exists() else ()
    defensive = frozenset(pack_cfg.defensive_codes)
    universe = build_universe_codes(holdings, cluster_codes, defensive)
    _, _, cluster_frame, jump_frame = load_fat_tail_data(
        pack_dir,
        cluster_mapping_path=cluster_path,
        defensive_codes=defensive,
        cfg=fat_cfg,
    )

    start = warmup_start(as_of, fat_cfg)
    panel = load_qlib_close(str(QLIB_PROVIDER_URI), universe, start, as_of)
    panel, _ = auto_qfq_adjust_close_panel(panel)

    horizon = int(digest_cfg.horizon)
    eval_end_requested = pd.Timestamp(args.eval_end)
    if len(panel.index) <= horizon:
        raise SystemExit(f"[ERR] panel too short for H={horizon}")
    last_label_date = pd.Timestamp(panel.index[-1 - horizon])
    eval_end = min(eval_end_requested, last_label_date)
    if eval_end < eval_end_requested:
        print(
            f"[WARN] eval-end clipped to {eval_end.date()} "
            f"(need {horizon} forward trading days for labels)"
        )
    eval_dates = build_eval_dates(
        panel,
        eval_start=pd.Timestamp(args.eval_start),
        eval_end=eval_end,
        step_days=max(1, int(args.step_days)),
    )

    print(
        f"[INFO] left_side topk ranking backtest "
        f"rule={rank_rule} H={horizon} k={top_k} "
        f"eval={args.eval_start}..{args.eval_end} step={args.step_days} "
        f"days={len(eval_dates)}"
    )

    signals, daily, aggregate = run_ranking_backtest(
        panel,
        cluster_frame,
        eval_dates,
        digest_cfg=digest_cfg,
        garch_board_cfg=garch_board_cfg,
        hmm_board_cfg=hmm_board_cfg,
        rt_cfg=rt_cfg,
        jump_frame=jump_frame,
        spike_mult=spike_mult,
        top_k=top_k,
        top_pct=float(args.top_pct),
    )
    if not aggregate.empty:
        pass  # rank_rule already in daily rows

    out_signals = args.output_signals or pack_dir / "left_side_topk_ranking_signals.csv"
    out_daily = args.output_daily or pack_dir / "left_side_topk_ranking_daily.csv"
    out_agg = args.output_aggregate or pack_dir / "left_side_topk_ranking_aggregate.csv"

    signals.to_csv(out_signals, index=False, encoding="utf-8-sig")
    daily.to_csv(out_daily, index=False, encoding="utf-8-sig")
    aggregate.to_csv(out_agg, index=False, encoding="utf-8-sig")

    print("\n=== Aggregate ranking accuracy ===")
    if aggregate.empty:
        print("(no results)")
    else:
        print(aggregate.to_string(index=False))

    print(f"\n[OK] signals -> {out_signals} ({len(signals)} rows)")
    print(f"[OK] daily -> {out_daily} ({len(daily)} rows)")
    print(f"[OK] aggregate -> {out_agg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
