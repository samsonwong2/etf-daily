#!/usr/bin/env python3
"""红三角买点参数组合全池回测。

Universe: 红三角日（signals_oos: quantile_switch & switch_side=down，含 extreme_drop）
Features: day_ret / close_eq_low / below_ma20 / vol5_pct_120d / prior10 /
          range_pos=(c-l)/(h-l) / rv_expand
Sweep: 买点过滤 × 止盈(高开/浮盈/持有天数) × 止损(破位缓冲/浮亏)
Entry: T 收盘；Exit: 先止盈后止损（与三件套一致）
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from decision_pack.src.regime_transition_plot import (
    compute_volatility_regime_frame,
    load_validation_csvs,
)
from decision_pack.src.symbol_trend_board import load_cluster_mapping_codes
from runtime_paths import CLUSTER_MAPPING_SELECTED_TXT

CLOSE_EQ_LOW_TOL = 1e-4
MIN_N = 15


def _load_plot_mod():
    path = _PROJECT_ROOT / "decision_pack" / "scripts" / "plot_regime_transition_example.py"
    spec = importlib.util.spec_from_file_location("plot_red_sweep", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_rules_mod():
    path = _PROJECT_ROOT / "decision_pack" / "scripts" / "backtest_triangle_decision_rules.py"
    spec = importlib.util.spec_from_file_location("rules_red_sweep", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ohlcv_frame(ohlcv: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.copy()
    df["as_of"] = pd.to_datetime(df["datetime"]).dt.normalize()
    df = df.sort_values("as_of")
    df = df[~df["as_of"].duplicated(keep="last")].reset_index(drop=True)
    for col in ("$open", "$high", "$low", "$close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    vf = vol.copy()
    vf["as_of"] = pd.to_datetime(vf["as_of"]).dt.normalize()
    out = df.merge(
        vf[["as_of", "rv5", "rv20", "vol5_pct_120d", "vol_regime"]],
        on="as_of",
        how="left",
    )
    out["ret1"] = out["$close"].pct_change(fill_method=None)
    out["MA20"] = out["$close"].rolling(20, min_periods=20).mean()
    out["below_ma20"] = out["$close"] / out["MA20"] - 1.0
    out["close_eq_low"] = (
        (out["$close"] - out["$low"]).abs() / out["$close"].clip(lower=1e-8) <= CLOSE_EQ_LOW_TOL
    )
    rng = (out["$high"] - out["$low"]).replace(0, np.nan)
    out["range_pos"] = (out["$close"] - out["$low"]) / rng
    out["rv_expand"] = out["rv5"] > out["rv20"]
    # prior10: close / close_10ago - 1
    out["prior10"] = out["$close"] / out["$close"].shift(10) - 1.0
    return out


def collect_red_events(
    codes: list[str],
    oos: pd.DataFrame,
    plot_mod: Any,
    rules_mod: Any,
    *,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return red-triangle event table + per-code px frames."""
    px_map: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    oos = oos.copy()
    oos["as_of"] = pd.to_datetime(oos["as_of"]).dt.normalize()
    oos["code"] = oos["code"].astype(str).str.upper()

    for code in codes:
        sub = oos[oos["code"] == code]
        if sub.empty:
            continue
        ohlcv = plot_mod.load_qlib_ohlcv(
            code, start, end, init_qlib=False, lookback_calendar_days=220, clip_to_window=False
        )
        ohlcv = plot_mod.attach_sma_columns(ohlcv)
        close = ohlcv.set_index(pd.to_datetime(ohlcv["datetime"]).dt.normalize())["$close"]
        close = close[~close.index.duplicated(keep="last")]
        vol = compute_volatility_regime_frame(close)
        px = _ohlcv_frame(ohlcv, vol)
        px_map[code] = px

        # red triangles from oos
        red = sub[
            sub["quantile_switch"].fillna(False).astype(bool)
            & (sub["switch_side"].astype(str).str.lower() == "down")
        ].copy()
        if red.empty:
            print(f"[OK] {code} reds=0")
            continue

        pxi = px.set_index("as_of")
        for _, r in red.iterrows():
            d = pd.Timestamp(r["as_of"]).normalize()
            if d not in pxi.index:
                continue
            bar = pxi.loc[d]
            if isinstance(bar, pd.DataFrame):
                bar = bar.iloc[-1]
            rows.append(
                {
                    "code": code,
                    "signal_date": d,
                    "day_ret": float(bar["ret1"]) if pd.notna(bar["ret1"]) else float("nan"),
                    "close": float(bar["$close"]),
                    "high": float(bar["$high"]),
                    "low": float(bar["$low"]),
                    "open": float(bar["$open"]),
                    "close_eq_low": bool(bar["close_eq_low"]),
                    "below_ma20": float(bar["below_ma20"]) if pd.notna(bar["below_ma20"]) else np.nan,
                    "vol_pct": float(bar["vol5_pct_120d"]) if pd.notna(bar["vol5_pct_120d"]) else np.nan,
                    "prior10": float(bar["prior10"]) if pd.notna(bar["prior10"]) else np.nan,
                    "range_pos": float(bar["range_pos"]) if pd.notna(bar["range_pos"]) else np.nan,
                    "rv_expand": bool(bar["rv_expand"]) if pd.notna(bar["rv_expand"]) else False,
                    "switch_source": str(r.get("switch_source") or ""),
                    "extreme_drop": bool(r.get("extreme_drop_switch")) if pd.notna(r.get("extreme_drop_switch")) else False,
                }
            )
        print(f"[OK] {code} reds={len(red)} feats={sum(1 for x in rows if x['code']==code)}")

    return pd.DataFrame(rows), px_map


def pass_buy_filter(row: pd.Series, filt: dict[str, Any]) -> bool:
    if filt.get("require_floor") and not bool(row["close_eq_low"]):
        return False
    if filt.get("day_ret_max") is not None:
        if pd.isna(row["day_ret"]) or float(row["day_ret"]) > float(filt["day_ret_max"]):
            return False
    if filt.get("below_ma20_max") is not None:
        if pd.isna(row["below_ma20"]) or float(row["below_ma20"]) > float(filt["below_ma20_max"]):
            return False
    if filt.get("vol_pct_min") is not None:
        if pd.isna(row["vol_pct"]) or float(row["vol_pct"]) < float(filt["vol_pct_min"]):
            return False
    if filt.get("prior10_max") is not None:
        if pd.isna(row["prior10"]) or float(row["prior10"]) > float(filt["prior10_max"]):
            return False
    if filt.get("range_pos_max") is not None:
        if pd.isna(row["range_pos"]) or float(row["range_pos"]) > float(filt["range_pos_max"]):
            return False
    if filt.get("require_rv_expand") and not bool(row["rv_expand"]):
        return False
    return True


def simulate_one(
    px: pd.DataFrame,
    signal_date: pd.Timestamp,
    entry: float,
    buy_low: float,
    *,
    tp_gap: float,
    float_tp: float | None,
    max_hold: int,
    sl_buf: float,
    sl_pct: float,
) -> dict[str, Any] | None:
    pxi = px.reset_index(drop=True)
    idx = {pd.Timestamp(a): i for i, a in enumerate(pxi["as_of"])}
    if signal_date not in idx:
        return None
    i = idx[signal_date]
    if i + 1 >= len(pxi):
        return None
    stop_line = buy_low * (1.0 - sl_buf)
    last_i = min(i + max_hold, len(pxi) - 1)
    for j in range(i + 1, last_i + 1):
        bar = pxi.iloc[j]
        o, h, c = float(bar["$open"]), float(bar["$high"]), float(bar["$close"])
        if o / entry - 1.0 >= tp_gap:
            return {
                "exit_why": "止盈_高开",
                "exit_px": o,
                "hold_days": j - i,
                "fwd_ret": o / entry - 1.0,
                "exit_date": str(pd.Timestamp(bar["as_of"]).date()),
            }
        if float_tp is not None and h / entry - 1.0 >= float_tp:
            px_out = entry * (1.0 + float_tp)
            return {
                "exit_why": "止盈_浮盈",
                "exit_px": px_out,
                "hold_days": j - i,
                "fwd_ret": float_tp,
                "exit_date": str(pd.Timestamp(bar["as_of"]).date()),
            }
        if c < stop_line:
            return {
                "exit_why": "止损_破买点低",
                "exit_px": c,
                "hold_days": j - i,
                "fwd_ret": c / entry - 1.0,
                "exit_date": str(pd.Timestamp(bar["as_of"]).date()),
            }
        if c / entry - 1.0 <= sl_pct:
            return {
                "exit_why": "止损_浮亏",
                "exit_px": c,
                "hold_days": j - i,
                "fwd_ret": c / entry - 1.0,
                "exit_date": str(pd.Timestamp(bar["as_of"]).date()),
            }
        if j == last_i:
            ret = c / entry - 1.0
            why = "止盈_时间" if ret > 0 else "到期_亏损"
            return {
                "exit_why": why,
                "exit_px": c,
                "hold_days": j - i,
                "fwd_ret": ret,
                "exit_date": str(pd.Timestamp(bar["as_of"]).date()),
            }
    return None


BUY_FILTERS: list[dict[str, Any]] = [
    {"name": "红三角_全量", },
    {"name": "跌≤-5%", "day_ret_max": -0.05},
    {"name": "跌≤-7%", "day_ret_max": -0.07},
    {"name": "跌≤-8%", "day_ret_max": -0.08},
    {"name": "跌≤-7%_地板", "day_ret_max": -0.07, "require_floor": True},
    {"name": "跌≤-5%_地板", "day_ret_max": -0.05, "require_floor": True},
    {"name": "跌≤-7%_近低位", "day_ret_max": -0.07, "range_pos_max": 0.15},
    {"name": "跌≤-7%_破MA20_10%", "day_ret_max": -0.07, "below_ma20_max": -0.10},
    {"name": "跌≤-7%_破MA20_15%", "day_ret_max": -0.07, "below_ma20_max": -0.15},
    {"name": "跌≤-7%_vol≥0.7", "day_ret_max": -0.07, "vol_pct_min": 0.70},
    {"name": "跌≤-7%_vol≥0.8", "day_ret_max": -0.07, "vol_pct_min": 0.80},
    {"name": "跌≤-7%_vol≥0.9", "day_ret_max": -0.07, "vol_pct_min": 0.90},
    {"name": "跌≤-7%_prior10≤-8%", "day_ret_max": -0.07, "prior10_max": -0.08},
    {"name": "跌≤-7%_prior10≤-10%", "day_ret_max": -0.07, "prior10_max": -0.10},
    {"name": "跌≤-7%_地板_vol≥0.7", "day_ret_max": -0.07, "require_floor": True, "vol_pct_min": 0.70},
    {"name": "跌≤-7%_地板_vol≥0.8", "day_ret_max": -0.07, "require_floor": True, "vol_pct_min": 0.80},
    {"name": "跌≤-7%_地板_破MA15%", "day_ret_max": -0.07, "require_floor": True, "below_ma20_max": -0.15},
    {"name": "跌≤-7%_地板_prior≤-8%", "day_ret_max": -0.07, "require_floor": True, "prior10_max": -0.08},
    {"name": "跌≤-7%_近低_vol≥0.7", "day_ret_max": -0.07, "range_pos_max": 0.15, "vol_pct_min": 0.70},
    {"name": "跌≤-7%_破MA15%_vol≥0.7", "day_ret_max": -0.07, "below_ma20_max": -0.15, "vol_pct_min": 0.70},
    {"name": "跌≤-7%_地板_近低", "day_ret_max": -0.07, "require_floor": True, "range_pos_max": 0.05},
    {"name": "跌≤-7%_rv扩张", "day_ret_max": -0.07, "require_rv_expand": True},
    {"name": "跌≤-7%_地板_rv扩张", "day_ret_max": -0.07, "require_floor": True, "require_rv_expand": True},
    {"name": "跌≤-7%_vol≥0.8_破MA10%", "day_ret_max": -0.07, "vol_pct_min": 0.80, "below_ma20_max": -0.10},
]


def exit_grid() -> list[dict[str, Any]]:
    grid = []
    for tp_gap, float_tp, max_hold, sl_buf, sl_pct in itertools.product(
        (0.02, 0.03, 0.05),
        (None, 0.08),
        (5, 10),
        (0.01, 0.02),
        (-0.06, -0.08),
    ):
        name = f"gap{int(tp_gap*100)}_ft{'NA' if float_tp is None else int(float_tp*100)}_h{max_hold}_sb{int(sl_buf*100)}_sl{int(abs(sl_pct)*100)}"
        grid.append(
            {
                "name": name,
                "tp_gap": tp_gap,
                "float_tp": float_tp,
                "max_hold": max_hold,
                "sl_buf": sl_buf,
                "sl_pct": sl_pct,
            }
        )
    return grid


def run_sweep(events: pd.DataFrame, px_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    exits = exit_grid()
    summary_rows: list[dict[str, Any]] = []
    # Pre-index events by filter for speed
    for bf in BUY_FILTERS:
        mask = events.apply(lambda r: pass_buy_filter(r, bf), axis=1)
        cand = events[mask].copy()
        n_sig = len(cand)
        if n_sig == 0:
            continue
        for ex in exits:
            trades = []
            # non-overlap per code
            busy: dict[str, pd.Timestamp] = {}
            for _, ev in cand.sort_values(["code", "signal_date"]).iterrows():
                code = ev["code"]
                sd = pd.Timestamp(ev["signal_date"])
                if code in busy and sd <= busy[code]:
                    continue
                px = px_map.get(code)
                if px is None:
                    continue
                sim = simulate_one(
                    px,
                    sd,
                    float(ev["close"]),
                    float(ev["low"]),
                    tp_gap=ex["tp_gap"],
                    float_tp=ex["float_tp"],
                    max_hold=ex["max_hold"],
                    sl_buf=ex["sl_buf"],
                    sl_pct=ex["sl_pct"],
                )
                if sim is None:
                    continue
                trades.append(sim)
                busy[code] = pd.Timestamp(sim["exit_date"])
            if not trades:
                continue
            tdf = pd.DataFrame(trades)
            n = len(tdf)
            win = float((tdf["fwd_ret"] > 0).mean())
            mean_r = float(tdf["fwd_ret"].mean())
            med_r = float(tdf["fwd_ret"].median())
            # composite: emphasize mean with sample size penalty
            score = mean_r * np.sqrt(n) if n >= MIN_N else -999.0
            summary_rows.append(
                {
                    "buy_filter": bf["name"],
                    "exit_rule": ex["name"],
                    "tp_gap": ex["tp_gap"],
                    "float_tp": ex["float_tp"] if ex["float_tp"] is not None else np.nan,
                    "max_hold": ex["max_hold"],
                    "sl_buf": ex["sl_buf"],
                    "sl_pct": ex["sl_pct"],
                    "n_signals": n_sig,
                    "n_trades": n,
                    "win_rate": win,
                    "mean_ret": mean_r,
                    "median_ret": med_r,
                    "p25_ret": float(tdf["fwd_ret"].quantile(0.25)),
                    "mean_hold": float(tdf["hold_days"].mean()),
                    "tp_gap_rate": float(tdf["exit_why"].eq("止盈_高开").mean()),
                    "sl_rate": float(tdf["exit_why"].str.startswith("止损").mean()),
                    "score": score,
                }
            )
    return pd.DataFrame(summary_rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--validation-dir",
        type=Path,
        default=_PROJECT_ROOT
        / "workspace/decision_packs/20260720/regime_transition_validation_q90_to0720",
    )
    p.add_argument("--start-date", default="2024-01-01")
    p.add_argument("--end-date", default="2026-08-05")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    val = args.validation_dir if args.validation_dir.is_absolute() else _PROJECT_ROOT / args.validation_dir
    out_dir = args.out_dir or (val / "triangle_decision_backtest_red_buy_sweep")
    if not out_dir.is_absolute():
        out_dir = _PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    oos, _rev, _evt = load_validation_csvs(val)
    codes = [c.upper() for c in load_cluster_mapping_codes(CLUSTER_MAPPING_SELECTED_TXT)]
    codes = [c for c in codes if c in set(oos["code"].astype(str).str.upper())]
    plot_mod = _load_plot_mod()
    rules_mod = _load_rules_mod()
    plot_mod._init_qlib(None)

    print(f"[INFO] red-buy sweep codes={len(codes)} {args.start_date}->{args.end_date}")
    events, px_map = collect_red_events(
        codes, oos, plot_mod, rules_mod, start=args.start_date, end=args.end_date
    )
    events_path = out_dir / "red_events_features.csv"
    events.to_csv(events_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] red events={len(events)} -> {events_path}")

    if events.empty:
        print("[WARN] no red events")
        return 0

    print(f"[INFO] sweeping buy×exit combos ...")
    summ = run_sweep(events, px_map)
    summ_path = out_dir / "red_buy_sweep_summary.csv"
    summ.to_csv(summ_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] wrote {summ_path} rows={len(summ)}")

    valid = summ[summ["n_trades"] >= MIN_N].copy()
    if valid.empty:
        valid = summ.copy()

    top_score = valid.sort_values(["score", "mean_ret", "win_rate"], ascending=False).head(15)
    top_win = valid.sort_values(["win_rate", "mean_ret", "n_trades"], ascending=False).head(15)
    top_mean = valid.sort_values(["mean_ret", "win_rate", "n_trades"], ascending=False).head(15)

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 25)
    print("\n=== TOP by score (mean*sqrt(n), n>=%d) ===" % MIN_N)
    print(top_score.to_string(index=False))
    print("\n=== TOP by win_rate ===")
    print(top_win.to_string(index=False))
    print("\n=== TOP by mean_ret ===")
    print(top_mean.to_string(index=False))

    best = top_score.iloc[0].to_dict() if len(top_score) else None
    md = [
        "# 红三角买点参数组合全池回测",
        "",
        f"- 区间：{args.start_date} → {args.end_date}",
        f"- 红三角事件数：{len(events)}",
        f"- 组合数：{len(summ)}；入选门槛 n_trades≥{MIN_N}",
        "",
        "## 最优（score=mean×√n）",
        "",
    ]
    if best:
        md.extend(
            [
                f"- **买点过滤**：`{best['buy_filter']}`",
                f"- **退出**：`{best['exit_rule']}`",
                f"- n={int(best['n_trades'])} 胜率={best['win_rate']:.1%} 均值={best['mean_ret']:+.2%} 中位={best['median_ret']:+.2%}",
                f"- 高开止盈占比={best['tp_gap_rate']:.0%} 止损占比={best['sl_rate']:.0%} 均持仓={best['mean_hold']:.1f}日",
                "",
            ]
        )
    md.append("## Top15 by score")
    md.append("")
    md.append(top_score.to_markdown(index=False))
    md.append("")
    md.append("## Top15 by win_rate")
    md.append("")
    md.append(top_win.to_markdown(index=False))
    md.append("")
    md.append("## Top15 by mean_ret")
    md.append("")
    md.append(top_mean.to_markdown(index=False))
    (out_dir / "red_buy_sweep_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[INFO] wrote {out_dir / 'red_buy_sweep_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
