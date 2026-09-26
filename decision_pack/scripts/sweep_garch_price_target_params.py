#!/usr/bin/env python3
"""Sweep H × barrier_mult for all garch_price_targets universe symbols."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

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
from decision_pack.src.fat_tail_risk import fat_tail_config_from_raw, slice_log_returns
from decision_pack.src.garch_dynamics import (
    fit_garch_dynamics,
    forecast_sigma_h,
    simulate_step_returns,
)
from decision_pack.src.garch_price_targets_validation import (
    _simple_return,
    _strategy_log_return,
)
from decision_pack.src.garch_short_horizon_board import (
    GarchShortHorizonConfig,
    garch_short_horizon_config_from_raw,
)
from decision_pack.src.short_horizon_mlfam import (
    forward_cumulative_log_return,
    forward_log_returns,
    realize_triple_barrier,
)
from decision_pack.src.short_horizon_validation import build_eval_dates
from decision_pack.src.symbol_trend_board import load_cluster_mapping_codes
from pipeline.price_adjustments import auto_qfq_adjust_close_panel
from pipeline.strategy_layer_audit import load_qlib_close
from runtime_paths import CLUSTER_MAPPING_SELECTED_TXT, QLIB_PROVIDER_URI


def _mc_probs(steps: np.ndarray, barrier: float) -> tuple[float, float, float]:
    cum = np.cumsum(steps, axis=1)
    n_paths = steps.shape[0]
    up_wins = down_wins = 0
    for i in range(n_paths):
        path = cum[i]
        up_hit = path >= barrier
        down_hit = path <= -barrier
        if not up_hit.any() and not down_hit.any():
            continue
        first_up = int(np.argmax(up_hit)) if up_hit.any() else path.size + 1
        first_down = int(np.argmax(down_hit)) if down_hit.any() else path.size + 1
        if first_up < first_down:
            up_wins += 1
        elif first_down < first_up:
            down_wins += 1
        else:
            up_wins += 1
    timeout = n_paths - up_wins - down_wins
    total = float(n_paths)
    return up_wins / total, down_wins / total, timeout / total


def sweep(
    close_panel: pd.DataFrame,
    *,
    eval_dates: pd.DatetimeIndex,
    horizons: tuple[int, ...],
    barrier_mults: tuple[float, ...],
    cfg: GarchShortHorizonConfig,
    flags_by_code: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    gcfg = cfg.garch
    rows: list[dict] = []
    max_h = max(horizons)

    for code in close_panel.columns:
        series = pd.to_numeric(close_panel[code], errors="coerce").dropna()
        if series.size < cfg.min_samples + max_h + 5:
            continue
        flag_tuple = flags_by_code.get(str(code), ())

        for end in eval_dates:
            if end > series.index.max():
                continue
            ts_end = pd.Timestamp(end)
            available = series.loc[series.index <= ts_end].dropna()
            if available.empty:
                continue
            close = float(available.iloc[-1])
            if close <= 0 or not math.isfinite(close):
                continue

            for horizon in horizons:
                fwd = forward_log_returns(series, ts_end, horizon=horizon)
                if fwd is None:
                    continue
                cum_h = forward_cumulative_log_return(fwd)
                if cum_h is None:
                    continue

                rets = slice_log_returns(series, ts_end, lookback_days=gcfg.train_window)
                fit = fit_garch_dynamics(rets, gcfg, max_horizon=max_h)
                if not fit.reliable or fit.sigma_1d is None:
                    continue
                sigma_h = forecast_sigma_h(fit, horizon)
                if sigma_h is None or sigma_h <= 0:
                    continue

                steps = simulate_step_returns(
                    fit,
                    int(horizon),
                    gcfg.mc_samples,
                    gcfg.mc_seed + int(horizon),
                )
                if steps.size == 0:
                    continue

                for bm in barrier_mults:
                    barrier = float(bm * sigma_h)
                    if barrier <= 0:
                        continue
                    outcome = realize_triple_barrier(fwd, barrier=barrier)
                    p_up, p_down, p_timeout = _mc_probs(steps, barrier)
                    strat_log = _strategy_log_return(outcome, barrier, cum_h)
                    bh_log = float(cum_h)
                    rows.append(
                        {
                            "code": str(code),
                            "as_of": ts_end.strftime("%Y-%m-%d"),
                            "H": int(horizon),
                            "barrier_mult": float(bm),
                            "close": close,
                            "barrier_log": barrier,
                            "sl_dist_pct": (math.exp(-barrier) - 1.0) * 100.0,
                            "p_tp": p_up,
                            "p_sl": p_down,
                            "p_timeout": p_timeout,
                            "actual_outcome": outcome,
                            "strategy_simple_ret": _simple_return(strat_log),
                            "buy_hold_simple_ret": _simple_return(bh_log),
                        }
                    )

    return pd.DataFrame(rows)


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    records = []
    for (h, bm), g in frame.groupby(["H", "barrier_mult"]):
        loss = g[g["buy_hold_simple_ret"] < 0]
        down = g[g["actual_outcome"] == "down"]
        actual = g["actual_outcome"].to_numpy()
        records.append(
            {
                "H": int(h),
                "barrier_mult": float(bm),
                "n_obs": len(g),
                "n_symbols": int(g["code"].nunique()),
                "pred_p_sl": float(g["p_sl"].mean()),
                "act_sl_rate": float((actual == "down").mean()),
                "pred_p_tp": float(g["p_tp"].mean()),
                "act_tp_rate": float((actual == "up").mean()),
                "timeout_rate": float((actual == "timeout").mean()),
                "sl_width_pct": float((-g["sl_dist_pct"]).mean()),
                "strat_mean_ret": float(g["strategy_simple_ret"].mean()),
                "bh_mean_ret": float(g["buy_hold_simple_ret"].mean()),
                "excess_mean_ret": float(
                    (g["strategy_simple_ret"] - g["buy_hold_simple_ret"]).mean()
                ),
                "strat_win_rate": float((g["strategy_simple_ret"] > 0).mean()),
                "bh_win_rate": float((g["buy_hold_simple_ret"] > 0).mean()),
                "loss_day_n": int(len(loss)),
                "loss_day_saved_pp": float(
                    (loss["strategy_simple_ret"] - loss["buy_hold_simple_ret"]).mean()
                    if len(loss)
                    else 0.0
                ),
                "sl_day_n": int(len(down)),
                "sl_day_saved_pp": float(
                    (down["strategy_simple_ret"] - down["buy_hold_simple_ret"]).mean()
                    if len(down)
                    else 0.0
                ),
                "brier": float(
                    np.mean(
                        (g["p_tp"] - (actual == "up")) ** 2
                        + (g["p_sl"] - (actual == "down")) ** 2
                        + (g["p_timeout"] - (actual == "timeout")) ** 2
                    )
                ),
            }
        )
    out = pd.DataFrame(records)
    return out.sort_values(
        by=["loss_day_saved_pp", "excess_mean_ret", "brier"],
        ascending=[False, False, True],
    )


def per_symbol_best(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    rows = []
    for code, g in frame.groupby("code"):
        g = g.copy()
        g["score"] = (
            g.groupby(["H", "barrier_mult"])["strategy_simple_ret"].transform("mean")
            - g.groupby(["H", "barrier_mult"])["buy_hold_simple_ret"].transform("mean")
        )
        agg = (
            g.groupby(["H", "barrier_mult"])
            .agg(
                n=("strategy_simple_ret", "size"),
                excess=("strategy_simple_ret", lambda s: s.mean())
                - g.loc[s.index, "buy_hold_simple_ret"].mean(),
                loss_saved=(
                    "strategy_simple_ret",
                    lambda s: (
                        (s[g.loc[s.index, "buy_hold_simple_ret"] < 0]
                         - g.loc[s.index, "buy_hold_simple_ret"][g.loc[s.index, "buy_hold_simple_ret"] < 0]
                        ).mean()
                        if (g.loc[s.index, "buy_hold_simple_ret"] < 0).any()
                        else 0.0
                    ),
                ),
            )
            .reset_index()
        )
        best = agg.sort_values(["loss_saved", "excess"], ascending=False).iloc[0]
        rows.append({"code": code, **best.to_dict()})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep GARCH price-target params")
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--eval-start", type=str, default="2024-06-01")
    parser.add_argument("--eval-end", type=str, default=None)
    parser.add_argument("--step-days", type=int, default=5)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--barrier-mults", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--output-detail", type=Path, default=None)
    parser.add_argument("--output-per-symbol", type=Path, default=None)
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

    horizons = tuple(int(h) for h in args.horizons)
    barrier_mults = tuple(float(b) for b in args.barrier_mults)
    max_h = max(horizons)
    eval_end = pd.Timestamp(args.eval_end) if args.eval_end else panel.index[-1 - max_h]
    eval_dates = build_eval_dates(
        panel,
        eval_start=pd.Timestamp(args.eval_start),
        eval_end=eval_end,
        step_days=max(1, args.step_days),
    )

    print(
        f"[INFO] symbols={len(panel.columns)} eval_dates={len(eval_dates)} "
        f"range={eval_dates.min().date()}..{eval_dates.max().date()} "
        f"H={horizons} barrier_mult={barrier_mults}"
    )

    detail = sweep(
        panel,
        eval_dates=eval_dates,
        horizons=horizons,
        barrier_mults=barrier_mults,
        cfg=board_cfg,
        flags_by_code=flags_map,
    )
    summary = aggregate(detail)

    out_summary = args.output_summary or (pack_dir / "garch_price_targets_param_sweep_summary.csv")
    out_detail = args.output_detail or (pack_dir / "garch_price_targets_param_sweep_detail.csv")
    out_ps = args.output_per_symbol or (pack_dir / "garch_price_targets_param_sweep_per_symbol.csv")

    summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
    detail.to_csv(out_detail, index=False, encoding="utf-8-sig")

    ps_rows = []
    for code, g in detail.groupby("code"):
        agg = (
            g.groupby(["H", "barrier_mult"])
            .apply(
                lambda x: pd.Series(
                    {
                        "n": len(x),
                        "excess_mean_ret": (
                            x["strategy_simple_ret"].mean() - x["buy_hold_simple_ret"].mean()
                        ),
                        "loss_saved_pp": (
                            (
                                x.loc[x["buy_hold_simple_ret"] < 0, "strategy_simple_ret"]
                                - x.loc[x["buy_hold_simple_ret"] < 0, "buy_hold_simple_ret"]
                            ).mean()
                            if (x["buy_hold_simple_ret"] < 0).any()
                            else 0.0
                        ),
                        "act_sl_rate": (x["actual_outcome"] == "down").mean(),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
        best = agg.sort_values(["loss_saved_pp", "excess_mean_ret"], ascending=False).iloc[0]
        ps_rows.append({"code": code, **best.to_dict()})
    pd.DataFrame(ps_rows).to_csv(out_ps, index=False, encoding="utf-8-sig")

    print(f"[OK] summary -> {out_summary}")
    print(f"[OK] detail  -> {out_detail} ({len(detail)} rows)")
    print(f"[OK] per_symbol -> {out_ps}")
    print("\n=== SUMMARY (sorted by loss_day_saved) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
