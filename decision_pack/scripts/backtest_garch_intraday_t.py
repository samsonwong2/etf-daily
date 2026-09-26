#!/usr/bin/env python3
"""Intraday T (做T) backtest for a held symbol using GARCH TP/SL rails + daily high/low."""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from decision_pack.src.config import load_decision_pack_config
from decision_pack.src.garch_price_targets import price_from_log_return
from decision_pack.src.garch_short_horizon_board import (
    GarchShortHorizonConfig,
    compute_garch_short_horizon_metrics,
    garch_short_horizon_config_from_raw,
)
from decision_pack.src.intraday_t import IntradayTConfig, intraday_t_config_from_raw
from decision_pack.src.short_horizon_mlfam import realize_triple_barrier
from decision_pack.src.symbol_trend_board import load_cluster_mapping_codes
from pipeline.price_adjustments import auto_qfq_adjust_price_columns_by_group
from runtime_paths import CLUSTER_MAPPING_SELECTED_TXT, QLIB_PROVIDER_URI

CORE4_CODES = ("SH588170", "SH588710", "SH588200", "SZ159502")


@dataclass(frozen=True)
class TIntradayRules:
    name: str
    t_frac: float = 0.33
    allow_zhen_t: bool = True  # 正T: high 触轨↑ 先卖后买
    allow_fan_t: bool = True  # 反T: low 触轨↓ 先买后卖
    min_p_tp: float = 0.0
    min_p_sl: float = 0.0
    rebuy_on_sl: bool = True  # 正T后若也触下轨，在 sl 买回
    max_hold_ret_for_zhen_t: float | None = None  # 震荡过滤：收盘涨幅低于该阈值(%)才做正T


def load_qlib_ohlc(
    provider_uri: str,
    codes: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    qlib.init(provider_uri=str(Path(provider_uri).expanduser()), region=REG_CN)
    raw = D.features(codes, ["$open", "$high", "$low", "$close"], start_time=start_date, end_time=end_date)
    raw.columns = ["open", "high", "low", "close"]
    raw = raw.sort_index().reset_index()
    raw["instrument"] = raw["instrument"].astype(str).str.upper()
    raw, _ = auto_qfq_adjust_price_columns_by_group(
        raw,
        group_col="instrument",
        date_col="datetime",
        price_cols=["open", "high", "low", "close"],
    )
    return raw.set_index(["instrument", "datetime"]).sort_index()


def intraday_barrier_outcome(
    *,
    ref_close: float,
    open_px: float,
    high_px: float,
    low_px: float,
    barrier_log: float,
) -> tuple[str, float | None, float | None]:
    """First-touch on day using high/low vs ref_close rails."""
    tp = ref_close * math.exp(barrier_log)
    sl = ref_close * math.exp(-barrier_log)
    up_hit = high_px >= tp
    down_hit = low_px <= sl
    if not up_hit and not down_hit:
        return "timeout", None, None
    if up_hit and not down_hit:
        return "up", tp, None
    if down_hit and not up_hit:
        return "down", None, sl
    # both hit: open-distance heuristic for which rail touched first
    dist_up = abs(tp - open_px)
    dist_dn = abs(open_px - sl)
    if dist_dn <= dist_up:
        return "down", tp, sl
    return "up", tp, sl


def t_overlay_return(
    *,
    ref_close: float,
    close_px: float,
    open_px: float,
    high_px: float,
    low_px: float,
    barrier_log: float,
    p_tp: float | None,
    p_sl: float | None,
    rules: TIntradayRules,
) -> tuple[float, str, float | None, float | None, float | None]:
    """Additive daily overlay on top of close-to-close hold return (fraction of ref)."""
    outcome, tp_px, sl_px = intraday_barrier_outcome(
        ref_close=ref_close,
        open_px=open_px,
        high_px=high_px,
        low_px=low_px,
        barrier_log=barrier_log,
    )
    hold_ret = close_px / ref_close - 1.0
    f = rules.t_frac
    overlay = 0.0
    action = "hold_only"

    ptp = float(p_tp) if p_tp is not None else 0.0
    psl = float(p_sl) if p_sl is not None else 0.0

    hold_ret_pct = hold_ret * 100.0
    if f > 0 and outcome == "up" and rules.allow_zhen_t and ptp >= rules.min_p_tp and tp_px is not None:
        chop_skip = (
            rules.max_hold_ret_for_zhen_t is not None
            and hold_ret_pct >= float(rules.max_hold_ret_for_zhen_t)
        )
        if chop_skip:
            action = "zhen_t_chop_skip"
        else:
            buy_px = sl_px if (rules.rebuy_on_sl and sl_px is not None) else close_px
            overlay = f * (tp_px - buy_px) / ref_close
            action = "zhen_t_tp_sl" if sl_px is not None and rules.rebuy_on_sl else "zhen_t_tp_close"
    elif f > 0 and outcome == "down" and rules.allow_fan_t and psl >= rules.min_p_sl and sl_px is not None:
        sell_px = tp_px if tp_px is not None else close_px
        overlay = f * (sell_px - sl_px) / ref_close
        action = "fan_t_sl_tp" if tp_px is not None else "fan_t_sl_close"
    elif outcome == "timeout":
        action = "timeout"

    return hold_ret + overlay, action, tp_px, sl_px, overlay


def max_drawdown(rets: pd.Series) -> float:
    eq = (1 + rets.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def run_backtest(
    ohlc: pd.DataFrame,
    *,
    code: str,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    horizon: int,
    barrier_mult_override: float | None,
    board_cfg: GarchShortHorizonConfig,
    flags: tuple[str, ...],
    rules_list: list[TIntradayRules],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = pd.to_numeric(ohlc["close"], errors="coerce")
    open_ = pd.to_numeric(ohlc["open"], errors="coerce")
    high = pd.to_numeric(ohlc["high"], errors="coerce")
    low = pd.to_numeric(ohlc["low"], errors="coerce")

    rows: list[dict] = []
    daily_ledgers: list[dict] = []

    trading = close.index[(close.index >= eval_start) & (close.index <= eval_end)]
    if len(trading) < 2:
        return pd.DataFrame(), pd.DataFrame()

    for i in range(1, len(trading)):
        day = trading[i]
        prev = trading[i - 1]
        ref = float(close.loc[prev])
        if ref <= 0:
            continue

        metrics = compute_garch_short_horizon_metrics(
            close,
            pd.Timestamp(prev),
            horizon=horizon,
            cfg=board_cfg,
            flags=flags,
            max_horizon=horizon,
        )
        if metrics.barrier is None or metrics.p_up is None or metrics.p_down is None:
            continue

        barrier = float(metrics.barrier)
        if barrier_mult_override is not None:
            # rescale barrier keeping sigma ratio
            default_bm = float(board_cfg.garch.barrier_mult)
            barrier = barrier * (float(barrier_mult_override) / default_bm)

        tp_price = price_from_log_return(ref, barrier)
        sl_price = price_from_log_return(ref, -barrier)

        o = float(open_.loc[day])
        h = float(high.loc[day])
        l = float(low.loc[day])
        c = float(close.loc[day])

        bar_outcome, _, _ = intraday_barrier_outcome(
            ref_close=ref,
            open_px=o,
            high_px=h,
            low_px=l,
            barrier_log=barrier,
        )

        # close-path outcome for comparison (H=1 forward single day)
        day_log = math.log(c / ref) if c > 0 else 0.0
        close_path_outcome = realize_triple_barrier(np.array([day_log]), barrier=barrier)

        row_base = {
            "code": code,
            "as_of": prev.strftime("%Y-%m-%d"),
            "trade_date": day.strftime("%Y-%m-%d"),
            "H": horizon,
            "ref_close": round(ref, 4),
            "open": round(o, 4),
            "high": round(h, 4),
            "low": round(l, 4),
            "close": round(c, 4),
            "tp_price": round(tp_price, 4) if tp_price else None,
            "sl_price": round(sl_price, 4) if sl_price else None,
            "p_tp": round(float(metrics.p_up), 4),
            "p_sl": round(float(metrics.p_down), 4),
            "p_timeout": round(float(metrics.p_timeout), 4),
            "barrier_log": round(barrier, 6),
            "barrier_pct": round((math.exp(barrier) - 1) * 100, 2),
            "intraday_outcome": bar_outcome,
            "close_path_outcome": close_path_outcome,
            "hold_ret_pct": round((c / ref - 1) * 100, 4),
        }
        rows.append(row_base)

        bh_ret = c / ref - 1.0
        for rules in rules_list:
            total_ret, action, _, _, overlay = t_overlay_return(
                ref_close=ref,
                close_px=c,
                open_px=o,
                high_px=h,
                low_px=l,
                barrier_log=barrier,
                p_tp=metrics.p_up,
                p_sl=metrics.p_down,
                rules=rules,
            )
            daily_ledgers.append(
                {
                    **row_base,
                    "strategy": rules.name,
                    "t_frac": rules.t_frac,
                    "action": action,
                    "overlay_ret_pct": round(overlay * 100, 4),
                    "total_ret_pct": round(total_ret * 100, 4),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(daily_ledgers)


def summarize(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return daily
    records = []
    for strat, g in daily.groupby("strategy"):
        rets = g["total_ret_pct"] / 100.0
        bh = g["hold_ret_pct"] / 100.0
        records.append(
            {
                "strategy": strat,
                "n_days": len(g),
                "cum_ret_pct": round(((1 + rets).prod() - 1) * 100, 2),
                "bh_cum_ret_pct": round(((1 + bh).prod() - 1) * 100, 2),
                "excess_vs_bh_pct": round(((1 + rets).prod() - (1 + bh).prod()) * 100, 2),
                "max_dd_pct": round(max_drawdown(rets) * 100, 2),
                "win_rate_pct": round(float((rets > 0).mean()) * 100, 1),
                "t_days_pct": round(
                    float(g["action"].str.startswith(("zhen_t_", "fan_t_")).mean()) * 100, 1
                ),
                "t_trigger_days": int(g["action"].str.startswith(("zhen_t_", "fan_t_")).sum()),
                "avg_overlay_pp": round(float(g["overlay_ret_pct"].mean()), 3),
            }
        )
    return pd.DataFrame(records).sort_values("cum_ret_pct", ascending=False)


def load_universe_codes(universe: str, explicit: list[str] | None) -> list[str]:
    if explicit:
        return [c.upper() for c in explicit]
    if universe == "core4":
        return list(CORE4_CODES)
    if universe == "cluster":
        path = Path(CLUSTER_MAPPING_SELECTED_TXT)
        if not path.exists():
            raise FileNotFoundError(f"cluster mapping not found: {path}")
        return list(load_cluster_mapping_codes(path))
    raise ValueError(f"unknown universe: {universe}")


def rules_from_intraday_t_config(it_cfg: IntradayTConfig, *, hold_pct: float | None = None) -> TIntradayRules:
    return TIntradayRules(
        it_cfg.strategy,
        t_frac=float(it_cfg.t_frac),
        allow_zhen_t=bool(it_cfg.allow_zhen_t),
        allow_fan_t=bool(it_cfg.allow_fan_t),
        min_p_tp=float(it_cfg.min_p_tp),
        min_p_sl=float(it_cfg.min_p_sl),
        rebuy_on_sl=bool(it_cfg.rebuy_on_sl),
        max_hold_ret_for_zhen_t=float(
            hold_pct if hold_pct is not None else it_cfg.max_hold_ret_for_zhen_t_pct
        ),
    )


def default_rules_list(
    *,
    sweep: bool = False,
    chop_hold_pcts: tuple[float, ...] | None = None,
    it_cfg: IntradayTConfig | None = None,
) -> list[TIntradayRules]:
    cfg = it_cfg or IntradayTConfig()
    rules: list[TIntradayRules] = [
        TIntradayRules("buy_hold", t_frac=0.0, allow_zhen_t=False, allow_fan_t=False),
    ]
    holds = chop_hold_pcts if chop_hold_pcts is not None else cfg.chop_hold_pct_sweep
    if sweep:
        for hold_pct in holds:
            suffix = f"_{int(hold_pct)}" if hold_pct == int(hold_pct) else f"_{hold_pct}"
            rules.append(replace(rules_from_intraday_t_config(cfg, hold_pct=hold_pct), name=f"zhen_t_chop_filter{suffix}"))
        return rules

    rules.append(rules_from_intraday_t_config(cfg))
    rules.extend(
        [
            TIntradayRules("t_both_33", t_frac=0.33, allow_zhen_t=True, allow_fan_t=True),
            TIntradayRules("t_both_50", t_frac=0.50, allow_zhen_t=True, allow_fan_t=True),
            TIntradayRules("zhen_t_only_33", t_frac=0.33, allow_zhen_t=True, allow_fan_t=False),
            TIntradayRules("fan_t_only_33", t_frac=0.33, allow_zhen_t=False, allow_fan_t=True),
            TIntradayRules("t_both_33_pfilter", t_frac=0.33, min_p_tp=0.03, min_p_sl=0.015),
            TIntradayRules("zhen_t_33_pfilter", t_frac=0.33, allow_fan_t=False, min_p_tp=0.03),
        ]
    )
    return rules


def run_symbol_backtest(
    code: str,
    ohlc: pd.DataFrame,
    *,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    horizon: int,
    bm_list: list[float],
    board_cfg: GarchShortHorizonConfig,
    rules_list: list[TIntradayRules],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_daily: list[pd.DataFrame] = []
    all_bar: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []

    for bm in bm_list:
        bars, daily = run_backtest(
            ohlc,
            code=code,
            eval_start=eval_start,
            eval_end=eval_end,
            horizon=horizon,
            barrier_mult_override=bm,
            board_cfg=board_cfg,
            flags=(),
            rules_list=rules_list,
        )
        if bars.empty:
            continue
        bars["barrier_mult"] = bm
        daily["barrier_mult"] = bm
        all_bar.append(bars)
        all_daily.append(daily)
        sm = summarize(daily)
        sm["code"] = code
        sm["barrier_mult"] = bm
        sm["H"] = horizon
        all_summary.append(sm)

    if not all_daily:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    bar_frame = pd.concat(all_bar, ignore_index=True)
    daily_frame = pd.concat(all_daily, ignore_index=True)
    summary_frame = pd.concat(all_summary, ignore_index=True)
    cal = (
        bar_frame.groupby("barrier_mult")
        .agg(
            n=("intraday_outcome", "count"),
            intraday_up=("intraday_outcome", lambda s: (s == "up").mean()),
            intraday_down=("intraday_outcome", lambda s: (s == "down").mean()),
            intraday_timeout=("intraday_outcome", lambda s: (s == "timeout").mean()),
            close_up=("close_path_outcome", lambda s: (s == "up").mean()),
            close_down=("close_path_outcome", lambda s: (s == "down").mean()),
            mean_p_tp=("p_tp", "mean"),
            mean_p_sl=("p_sl", "mean"),
            mean_barrier_pct=("barrier_pct", "mean"),
        )
        .reset_index()
    )
    cal["code"] = code
    return bar_frame, daily_frame, summary_frame, cal


def write_symbol_outputs(
    out_dir: Path,
    code: str,
    horizon: int,
    bar_frame: pd.DataFrame,
    daily_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    cal: pd.DataFrame,
) -> None:
    tag = f"{code}_h{horizon}"
    bar_frame.to_csv(out_dir / f"garch_intraday_t_bars_{tag}.csv", index=False, encoding="utf-8-sig")
    daily_frame.to_csv(out_dir / f"garch_intraday_t_daily_{tag}.csv", index=False, encoding="utf-8-sig")
    summary_frame.to_csv(out_dir / f"garch_intraday_t_summary_{tag}.csv", index=False, encoding="utf-8-sig")
    cal.to_csv(out_dir / f"garch_intraday_t_calibration_{tag}.csv", index=False, encoding="utf-8-sig")


def build_compare_frame(summary_frame: pd.DataFrame, *, horizon: int, barrier_mult: float) -> pd.DataFrame:
    sub = summary_frame[
        (summary_frame["H"] == horizon) & (summary_frame["barrier_mult"] == barrier_mult)
    ].copy()
    if sub.empty:
        return sub
    pivot = sub.pivot_table(
        index="code",
        columns="strategy",
        values=["cum_ret_pct", "excess_vs_bh_pct", "t_trigger_days"],
        aggfunc="first",
    )
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()
    cols = ["code"]
    for strat in ("buy_hold", "zhen_t_chop_filter"):
        for metric in ("cum_ret_pct", "excess_vs_bh_pct", "t_trigger_days"):
            col = f"{metric}_{strat}"
            if col in pivot.columns:
                cols.append(col)
    return pivot.reindex(columns=[c for c in cols if c in pivot.columns])


def build_barrier_optimize_summary(
    summary_frame: pd.DataFrame,
    cal_frame: pd.DataFrame,
    *,
    horizon: int,
    chop_strategies: tuple[str, ...],
) -> pd.DataFrame:
    chop = summary_frame[
        summary_frame["strategy"].isin(chop_strategies) & (summary_frame["H"] == horizon)
    ].copy()
    if chop.empty:
        return chop
    cal = cal_frame[cal_frame["H"] == horizon] if "H" in cal_frame.columns else cal_frame.copy()
    cal_mean = (
        cal.groupby("barrier_mult")
        .agg(
            mean_barrier_pct=("mean_barrier_pct", "mean"),
            intraday_up=("intraday_up", "mean"),
            intraday_down=("intraday_down", "mean"),
            intraday_timeout=("intraday_timeout", "mean"),
            mean_p_tp=("mean_p_tp", "mean"),
            mean_p_sl=("mean_p_sl", "mean"),
        )
        .reset_index()
    )
    rows: list[dict] = []
    for (bm, strat), g in chop.groupby(["barrier_mult", "strategy"]):
        bh = summary_frame[
            (summary_frame["strategy"] == "buy_hold")
            & (summary_frame["barrier_mult"] == bm)
            & (summary_frame["H"] == horizon)
        ]
        bh_map = bh.set_index("code")["cum_ret_pct"].to_dict() if not bh.empty else {}
        g = g.copy()
        g["bh_cum"] = g["code"].map(bh_map)
        g = g.dropna(subset=["bh_cum"])
        if g.empty:
            continue
        cal_row = cal_mean[cal_mean["barrier_mult"] == bm]
        rows.append(
            {
                "barrier_mult": float(bm),
                "strategy": strat,
                "n_symbols": len(g),
                "mean_excess_pp": round(float(g["excess_vs_bh_pct"].mean()), 2),
                "median_excess_pp": round(float(g["excess_vs_bh_pct"].median()), 2),
                "beat_bh_symbols": int((g["excess_vs_bh_pct"] > 0).sum()),
                "beat_bh_rate_pct": round(float((g["excess_vs_bh_pct"] > 0).mean()) * 100, 1),
                "mean_chop_cum_pct": round(float(g["cum_ret_pct"].mean()), 2),
                "mean_bh_cum_pct": round(float(g["bh_cum"].mean()), 2),
                "mean_t_days": round(float(g["t_trigger_days"].mean()), 1),
                "mean_barrier_pct": round(float(cal_row["mean_barrier_pct"].iloc[0]), 2)
                if not cal_row.empty
                else np.nan,
                "intraday_up_rate_pct": round(float(cal_row["intraday_up"].iloc[0]) * 100, 1)
                if not cal_row.empty
                else np.nan,
                "intraday_down_rate_pct": round(float(cal_row["intraday_down"].iloc[0]) * 100, 1)
                if not cal_row.empty
                else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["mean_excess_pp", "beat_bh_rate_pct"], ascending=[False, False])


def build_per_symbol_best_bm(
    summary_frame: pd.DataFrame,
    *,
    horizon: int,
    chop_strategy: str = "zhen_t_chop_filter",
) -> pd.DataFrame:
    chop = summary_frame[
        (summary_frame["strategy"] == chop_strategy) & (summary_frame["H"] == horizon)
    ].copy()
    if chop.empty:
        # fallback: any zhen_t_chop_filter* strategy
        chop = summary_frame[
            summary_frame["strategy"].str.startswith("zhen_t_chop_filter") & (summary_frame["H"] == horizon)
        ].copy()
    if chop.empty:
        return chop
    idx = chop.groupby("code")["excess_vs_bh_pct"].idxmax()
    best = chop.loc[idx].copy()
    cols = [
        "code",
        "barrier_mult",
        "strategy",
        "cum_ret_pct",
        "bh_cum_ret_pct",
        "excess_vs_bh_pct",
        "t_trigger_days",
        "max_dd_pct",
    ]
    return best.reindex(columns=[c for c in cols if c in best.columns]).sort_values(
        "excess_vs_bh_pct", ascending=False
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GARCH intraday T backtest")
    parser.add_argument("--code", type=str, default="SH588710")
    parser.add_argument(
        "--codes",
        type=str,
        nargs="*",
        default=None,
        help="Batch symbols (overrides --universe/--code)",
    )
    parser.add_argument(
        "--universe",
        type=str,
        choices=("single", "core4", "cluster"),
        default="single",
        help="cluster = full cluster_mapping_selected pool (36)",
    )
    parser.add_argument("--eval-start", type=str, default="2026-06-01")
    parser.add_argument("--eval-end", type=str, default="2026-07-13")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--barrier-mult", type=float, default=None, help="Single barrier_mult")
    parser.add_argument(
        "--barrier-mults",
        type=float,
        nargs="*",
        default=None,
        help="Sweep barrier_mult grid for TP/SL width optimization",
    )
    parser.add_argument(
        "--chop-hold-pcts",
        type=float,
        nargs="*",
        default=None,
        help="Sweep max_hold_ret_for_zhen_t (%%) for chop filter, e.g. 2 3 4 5",
    )
    parser.add_argument(
        "--sweep-mode",
        action="store_true",
        help="Only buy_hold + zhen_t_chop_filter variants (faster universe run)",
    )
    parser.add_argument(
        "--write-symbol-files",
        action="store_true",
        help="Write per-symbol bars/daily CSVs (off by default for cluster sweep)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    pack_cfg = load_decision_pack_config()
    board_cfg = garch_short_horizon_config_from_raw(pack_cfg.raw.get("garch_short_horizon"))
    it_cfg = intraday_t_config_from_raw(pack_cfg.raw.get("intraday_t"))

    if args.codes:
        codes = load_universe_codes("single", args.codes)
    elif args.universe == "single":
        codes = [args.code.upper()]
    else:
        codes = load_universe_codes(args.universe, None)

    eval_start = pd.Timestamp(args.eval_start)
    eval_end = pd.Timestamp(args.eval_end)
    warmup_start = (eval_start - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    chop_holds = tuple(args.chop_hold_pcts) if args.chop_hold_pcts else it_cfg.chop_hold_pct_sweep
    sweep = bool(args.sweep_mode or args.universe == "cluster" or args.barrier_mults)
    rules_list = default_rules_list(sweep=sweep, chop_hold_pcts=chop_holds, it_cfg=it_cfg)

    if args.barrier_mults:
        bm_list = [float(x) for x in args.barrier_mults]
    elif args.barrier_mult is not None:
        bm_list = [float(args.barrier_mult)]
    elif sweep:
        bm_list = list(it_cfg.barrier_mult_sweep)
    else:
        bm_list = [float(it_cfg.barrier_mult)]

    horizon = int(args.horizon)

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_symbol_files = bool(args.write_symbol_files or (not sweep and len(codes) <= 8))
    compare_bm = float(args.barrier_mult) if args.barrier_mult is not None else float(it_cfg.barrier_mult)

    print(
        f"[INFO] universe={len(codes)} symbols | eval={args.eval_start}..{args.eval_end} | "
        f"H={horizon} bm={bm_list} chop<{it_cfg.max_hold_ret_for_zhen_t_pct}% | sweep={sweep}"
    )

    ohlc_raw = load_qlib_ohlc(str(QLIB_PROVIDER_URI), codes, warmup_start, args.eval_end)
    all_summary: list[pd.DataFrame] = []
    all_cal: list[pd.DataFrame] = []

    for code in codes:
        if code not in ohlc_raw.index.get_level_values(0):
            print(f"[WARN] skip {code}: no OHLC data", file=sys.stderr)
            continue
        ohlc = ohlc_raw.xs(code, level=0).copy()
        ohlc.index = pd.to_datetime(ohlc.index)
        bar_frame, daily_frame, summary_frame, cal = run_symbol_backtest(
            code,
            ohlc,
            eval_start=eval_start,
            eval_end=eval_end,
            horizon=horizon,
            bm_list=bm_list,
            board_cfg=board_cfg,
            rules_list=rules_list,
        )
        if summary_frame.empty:
            print(f"[WARN] skip {code}: no backtest rows", file=sys.stderr)
            continue
        if write_symbol_files:
            write_symbol_outputs(out_dir, code, horizon, bar_frame, daily_frame, summary_frame, cal)
        all_summary.append(summary_frame)
        cal_h = cal.copy()
        cal_h["H"] = horizon
        all_cal.append(cal_h)
        if not sweep:
            print(f"[OK] {code} summary rows={len(summary_frame)}")
            chop = summary_frame[
                (summary_frame["strategy"].str.startswith("zhen_t_chop_filter"))
                & (summary_frame["barrier_mult"] == compare_bm)
            ]
            bh = summary_frame[
                (summary_frame["strategy"] == "buy_hold") & (summary_frame["barrier_mult"] == compare_bm)
            ]
            if not chop.empty and not bh.empty:
                row = chop.sort_values("excess_vs_bh_pct", ascending=False).iloc[0]
                print(
                    f"     bm={compare_bm}: buy_hold {bh.iloc[0]['cum_ret_pct']:.2f}% | "
                    f"{row['strategy']} {row['cum_ret_pct']:.2f}% "
                    f"(+{row['excess_vs_bh_pct']:.2f}pp, T={int(row['t_trigger_days'])}d)"
                )

    if not all_summary:
        print("[ERR] no backtest rows", file=sys.stderr)
        return 1

    merged_summary = pd.concat(all_summary, ignore_index=True)
    merged_cal = pd.concat(all_cal, ignore_index=True)
    tag = f"h{horizon}_{args.eval_start}_{args.eval_end}"
    merged_summary.to_csv(out_dir / f"garch_intraday_t_summary_all_{tag}.csv", index=False, encoding="utf-8-sig")

    chop_strats = tuple(s.name for s in rules_list if s.name.startswith("zhen_t_chop_filter"))
    opt = build_barrier_optimize_summary(
        merged_summary, merged_cal, horizon=horizon, chop_strategies=chop_strats
    )
    opt_path = out_dir / f"garch_intraday_t_barrier_optimize_{tag}.csv"
    opt.to_csv(opt_path, index=False, encoding="utf-8-sig")

    primary_chop = (
        "zhen_t_chop_filter"
        if "zhen_t_chop_filter" in chop_strats
        else (chop_strats[0] if chop_strats else "zhen_t_chop_filter")
    )
    per_sym = build_per_symbol_best_bm(
        merged_summary, horizon=horizon, chop_strategy=primary_chop
    )
    per_sym_path = out_dir / f"garch_intraday_t_per_symbol_best_bm_{tag}.csv"
    per_sym.to_csv(per_sym_path, index=False, encoding="utf-8-sig")

    compare = build_compare_frame(merged_summary, horizon=horizon, barrier_mult=compare_bm)
    if not compare.empty and len(codes) <= 12:
        compare_path = out_dir / f"garch_intraday_t_compare_{tag}_bm{compare_bm}.csv"
        compare.to_csv(compare_path, index=False, encoding="utf-8-sig")

    print(f"\n[OK] summary_all -> {out_dir / f'garch_intraday_t_summary_all_{tag}.csv'}")
    print(f"[OK] barrier_optimize -> {opt_path}")
    print(f"[OK] per_symbol_best -> {per_sym_path}")
    print("\n=== Barrier mult optimization (universe aggregate) ===")
    print(opt.to_string(index=False))
    if not per_sym.empty:
        bm_counts = per_sym["barrier_mult"].value_counts().sort_index()
        print("\n=== Best barrier_mult per symbol (counts) ===")
        print(bm_counts.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
