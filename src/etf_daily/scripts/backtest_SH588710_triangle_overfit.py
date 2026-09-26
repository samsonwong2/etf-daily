#!/usr/bin/env python3
"""SH588710 单票过拟合：红/绿三角 + 均线/波动/量价 → 最优买卖点。

明确目标：在 2025-06-04 → 2026-08-05 上网格搜索，找出样本内最佳
买点过滤 × 止盈/止损组合（过拟合优先，不做外推验证）。
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

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.lib.regime_transition_plot import (
    compute_volatility_regime_frame,
    load_validation_csvs,
)

CLOSE_EQ_TOL = 1e-4
MIN_N = 3  # 单票过拟合：至少 3 笔才进排行


def _load_plot_mod():
    path = _PROJECT_ROOT / "decision_pack" / "scripts" / "plot_regime_transition_example.py"
    spec = importlib.util.spec_from_file_location("plot_overfit_588710", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_frame(
    ohlcv: pd.DataFrame,
    oos: pd.DataFrame,
    *,
    code: str,
) -> pd.DataFrame:
    df = ohlcv.copy()
    df["as_of"] = pd.to_datetime(df["datetime"]).dt.normalize()
    df = df.sort_values("as_of")
    df = df[~df["as_of"].duplicated(keep="last")].reset_index(drop=True)
    for c in ("$open", "$high", "$low", "$close", "$volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    close = df.set_index("as_of")["$close"]
    vol = compute_volatility_regime_frame(close)
    vf = vol.copy()
    vf["as_of"] = pd.to_datetime(vf["as_of"]).dt.normalize()
    out = df.merge(
        vf[["as_of", "rv5", "rv20", "vol5_pct_120d", "vol_regime"]],
        on="as_of",
        how="left",
    )

    out["ret1"] = out["$close"].pct_change(fill_method=None)
    out["MA5"] = out["$close"].rolling(5, min_periods=5).mean()
    out["MA10"] = out["$close"].rolling(10, min_periods=10).mean()
    out["MA20"] = out["$close"].rolling(20, min_periods=20).mean()
    out["below_ma5"] = out["$close"] / out["MA5"] - 1.0
    out["below_ma10"] = out["$close"] / out["MA10"] - 1.0
    out["below_ma20"] = out["$close"] / out["MA20"] - 1.0
    out["ma_bull"] = (out["$close"] > out["MA5"]) & (out["MA5"] > out["MA10"])
    out["ma_bear"] = (out["$close"] < out["MA5"]) & (out["MA5"] < out["MA10"])
    out["close_eq_low"] = (
        (out["$close"] - out["$low"]).abs() / out["$close"].clip(lower=1e-8) <= CLOSE_EQ_TOL
    )
    out["close_eq_high"] = (
        (out["$high"] - out["$close"]).abs() / out["$close"].clip(lower=1e-8) <= CLOSE_EQ_TOL
    )
    rng = (out["$high"] - out["$low"]).replace(0, np.nan)
    out["range_pos"] = (out["$close"] - out["$low"]) / rng
    out["rv_expand"] = out["rv5"] > out["rv20"]
    out["vol_ma20"] = out["$volume"].rolling(20, min_periods=10).mean()
    out["vol_ratio"] = out["$volume"] / out["vol_ma20"]
    out["prior5"] = out["$close"] / out["$close"].shift(5) - 1.0
    out["prior10"] = out["$close"] / out["$close"].shift(10) - 1.0

    # triangles from oos
    sub = oos[oos["code"].astype(str).str.upper() == code.upper()].copy()
    sub["as_of"] = pd.to_datetime(sub["as_of"]).dt.normalize()
    red = set(
        pd.to_datetime(
            sub.loc[
                sub["quantile_switch"].fillna(False).astype(bool)
                & (sub["switch_side"].astype(str).str.lower() == "down"),
                "as_of",
            ]
        ).dt.normalize()
    )
    green = set(
        pd.to_datetime(
            sub.loc[
                sub["quantile_switch"].fillna(False).astype(bool)
                & (sub["switch_side"].astype(str).str.lower() == "up"),
                "as_of",
            ]
        ).dt.normalize()
    )
    out["is_red"] = out["as_of"].isin(red)
    out["is_green"] = out["as_of"].isin(green)

    # days since last opposite triangle
    last_red = np.nan
    last_green = np.nan
    dsr, dsg = [], []
    for _, r in out.iterrows():
        if bool(r["is_red"]):
            last_red = 0.0
        elif pd.notna(last_red):
            last_red += 1.0
        if bool(r["is_green"]):
            last_green = 0.0
        elif pd.notna(last_green):
            last_green += 1.0
        dsr.append(last_red)
        dsg.append(last_green)
    out["days_since_red"] = dsr
    out["days_since_green"] = dsg
    return out


def pass_filter(row: pd.Series, filt: dict[str, Any]) -> bool:
    # universe
    uni = filt.get("universe", "red")
    if uni == "red" and not bool(row["is_red"]):
        return False
    if uni == "green" and not bool(row["is_green"]):
        return False
    if uni == "red_or_green" and not (bool(row["is_red"]) or bool(row["is_green"])):
        return False
    if uni == "green_after_red":
        if not bool(row["is_green"]):
            return False
        dsr = row["days_since_red"]
        max_gap = float(filt.get("after_red_max", 10))
        if pd.isna(dsr) or not (1.0 <= float(dsr) <= max_gap):
            return False
    if uni == "red_after_green":
        if not bool(row["is_red"]):
            return False
        dsg = row["days_since_green"]
        max_gap = float(filt.get("after_green_max", 10))
        if pd.isna(dsg) or not (1.0 <= float(dsg) <= max_gap):
            return False

    if filt.get("require_floor") and not bool(row["close_eq_low"]):
        return False
    if filt.get("require_ceiling") and not bool(row["close_eq_high"]):
        return False
    if filt.get("require_ma_bull") and not bool(row["ma_bull"]):
        return False
    if filt.get("require_ma_bear") and not bool(row["ma_bear"]):
        return False
    if filt.get("require_rv_expand") and not bool(row["rv_expand"]):
        return False

    if filt.get("day_ret_max") is not None:
        if pd.isna(row["ret1"]) or float(row["ret1"]) > float(filt["day_ret_max"]):
            return False
    if filt.get("day_ret_min") is not None:
        if pd.isna(row["ret1"]) or float(row["ret1"]) < float(filt["day_ret_min"]):
            return False
    if filt.get("below_ma20_max") is not None:
        if pd.isna(row["below_ma20"]) or float(row["below_ma20"]) > float(filt["below_ma20_max"]):
            return False
    if filt.get("below_ma20_min") is not None:
        if pd.isna(row["below_ma20"]) or float(row["below_ma20"]) < float(filt["below_ma20_min"]):
            return False
    if filt.get("vol_pct_min") is not None:
        if pd.isna(row["vol5_pct_120d"]) or float(row["vol5_pct_120d"]) < float(filt["vol_pct_min"]):
            return False
    if filt.get("vol_pct_max") is not None:
        if pd.isna(row["vol5_pct_120d"]) or float(row["vol5_pct_120d"]) > float(filt["vol_pct_max"]):
            return False
    if filt.get("vol_ratio_min") is not None:
        if pd.isna(row["vol_ratio"]) or float(row["vol_ratio"]) < float(filt["vol_ratio_min"]):
            return False
    if filt.get("range_pos_max") is not None:
        if pd.isna(row["range_pos"]) or float(row["range_pos"]) > float(filt["range_pos_max"]):
            return False
    if filt.get("range_pos_min") is not None:
        if pd.isna(row["range_pos"]) or float(row["range_pos"]) < float(filt["range_pos_min"]):
            return False
    if filt.get("prior10_max") is not None:
        if pd.isna(row["prior10"]) or float(row["prior10"]) > float(filt["prior10_max"]):
            return False
    return True


def simulate(
    px: pd.DataFrame,
    i: int,
    entry: float,
    buy_low: float,
    *,
    tp_gap: float,
    float_tp: float | None,
    max_hold: int,
    sl_buf: float,
    sl_pct: float,
    exit_on_green: bool,
    exit_on_red: bool,
) -> dict[str, Any] | None:
    if i + 1 >= len(px):
        return None
    stop_line = buy_low * (1.0 - sl_buf)
    last_i = min(i + max_hold, len(px) - 1)
    for j in range(i + 1, last_i + 1):
        bar = px.iloc[j]
        o, h, c = float(bar["$open"]), float(bar["$high"]), float(bar["$close"])
        d = str(pd.Timestamp(bar["as_of"]).date())
        if o / entry - 1.0 >= tp_gap:
            return {
                "exit_date": d,
                "exit_why": "止盈_高开",
                "exit_px": o,
                "hold_days": j - i,
                "fwd_ret": o / entry - 1.0,
            }
        if float_tp is not None and h / entry - 1.0 >= float_tp:
            return {
                "exit_date": d,
                "exit_why": "止盈_浮盈",
                "exit_px": entry * (1.0 + float_tp),
                "hold_days": j - i,
                "fwd_ret": float_tp,
            }
        if exit_on_green and bool(bar["is_green"]):
            return {
                "exit_date": d,
                "exit_why": "止盈_绿三角",
                "exit_px": c,
                "hold_days": j - i,
                "fwd_ret": c / entry - 1.0,
            }
        if exit_on_red and bool(bar["is_red"]):
            return {
                "exit_date": d,
                "exit_why": "止损_红三角",
                "exit_px": c,
                "hold_days": j - i,
                "fwd_ret": c / entry - 1.0,
            }
        if c < stop_line:
            return {
                "exit_date": d,
                "exit_why": "止损_破买点低",
                "exit_px": c,
                "hold_days": j - i,
                "fwd_ret": c / entry - 1.0,
            }
        if c / entry - 1.0 <= sl_pct:
            return {
                "exit_date": d,
                "exit_why": "止损_浮亏",
                "exit_px": c,
                "hold_days": j - i,
                "fwd_ret": c / entry - 1.0,
            }
        if j == last_i:
            ret = c / entry - 1.0
            return {
                "exit_date": d,
                "exit_why": "止盈_时间" if ret > 0 else "到期_亏损",
                "exit_px": c,
                "hold_days": j - i,
                "fwd_ret": ret,
            }
    return None


def buy_filters() -> list[dict[str, Any]]:
    """Curated overfit filter set for single-name search."""
    out: list[dict[str, Any]] = []

    # --- RED mean-reversion family ---
    red_base = [
        {"name": "红_全量", "universe": "red"},
        {"name": "红_跌≤-5%", "universe": "red", "day_ret_max": -0.05},
        {"name": "红_跌≤-7%", "universe": "red", "day_ret_max": -0.07},
        {"name": "红_跌≤-7%_地板", "universe": "red", "day_ret_max": -0.07, "require_floor": True},
        {"name": "红_地板", "universe": "red", "require_floor": True},
        {"name": "红_近低", "universe": "red", "range_pos_max": 0.15},
        {"name": "红_跌≤-5%_近低", "universe": "red", "day_ret_max": -0.05, "range_pos_max": 0.20},
        {"name": "红_vol≥0.7", "universe": "red", "vol_pct_min": 0.70},
        {"name": "红_vol≥0.8", "universe": "red", "vol_pct_min": 0.80},
        {"name": "红_跌≤-7%_vol≥0.7", "universe": "red", "day_ret_max": -0.07, "vol_pct_min": 0.70},
        {"name": "红_跌≤-7%_地板_vol≥0.7", "universe": "red", "day_ret_max": -0.07, "require_floor": True, "vol_pct_min": 0.70},
        {"name": "红_破MA20_10%", "universe": "red", "below_ma20_max": -0.10},
        {"name": "红_破MA20_15%", "universe": "red", "below_ma20_max": -0.15},
        {"name": "红_跌≤-5%_破MA10%", "universe": "red", "day_ret_max": -0.05, "below_ma20_max": -0.10},
        {"name": "红_量比≥1.5", "universe": "red", "vol_ratio_min": 1.5},
        {"name": "红_量比≥2", "universe": "red", "vol_ratio_min": 2.0},
        {"name": "红_跌≤-5%_量比≥1.5", "universe": "red", "day_ret_max": -0.05, "vol_ratio_min": 1.5},
        {"name": "红_跌≤-7%_量比≥1.5", "universe": "red", "day_ret_max": -0.07, "vol_ratio_min": 1.5},
        {"name": "红_熊均线", "universe": "red", "require_ma_bear": True},
        {"name": "红_跌≤-5%_熊均线", "universe": "red", "day_ret_max": -0.05, "require_ma_bear": True},
        {"name": "红_rv扩张", "universe": "red", "require_rv_expand": True},
        {"name": "红_跌≤-7%_rv扩张", "universe": "red", "day_ret_max": -0.07, "require_rv_expand": True},
        {"name": "红_prior10≤-10%", "universe": "red", "prior10_max": -0.10},
        {"name": "红_跌≤-5%_prior10≤-10%", "universe": "red", "day_ret_max": -0.05, "prior10_max": -0.10},
        {"name": "红_后绿10日内", "universe": "red", "after_green_max": 10},  # uses red_after_green? need universe
    ]
    # fix last one
    red_base[-1] = {"name": "红_距绿≤10日", "universe": "red_after_green", "after_green_max": 10}
    red_base.append({"name": "红_距绿≤5日", "universe": "red_after_green", "after_green_max": 5})
    red_base.append(
        {
            "name": "红_跌≤-5%_近低_vol≥0.7",
            "universe": "red",
            "day_ret_max": -0.05,
            "range_pos_max": 0.20,
            "vol_pct_min": 0.70,
        }
    )
    red_base.append(
        {
            "name": "红_跌≤-7%_近低_量比≥1.5",
            "universe": "red",
            "day_ret_max": -0.07,
            "range_pos_max": 0.20,
            "vol_ratio_min": 1.5,
        }
    )
    out.extend(red_base)

    # --- GREEN momentum family ---
    green_base = [
        {"name": "绿_全量", "universe": "green"},
        {"name": "绿_涨≥+3%", "universe": "green", "day_ret_min": 0.03},
        {"name": "绿_涨≥+5%", "universe": "green", "day_ret_min": 0.05},
        {"name": "绿_天花板", "universe": "green", "require_ceiling": True},
        {"name": "绿_涨≥+3%_天花板", "universe": "green", "day_ret_min": 0.03, "require_ceiling": True},
        {"name": "绿_近高", "universe": "green", "range_pos_min": 0.80},
        {"name": "绿_牛均线", "universe": "green", "require_ma_bull": True},
        {"name": "绿_涨≥+3%_牛均线", "universe": "green", "day_ret_min": 0.03, "require_ma_bull": True},
        {"name": "绿_站上MA20", "universe": "green", "below_ma20_min": 0.0},
        {"name": "绿_涨≥+3%_站上MA20", "universe": "green", "day_ret_min": 0.03, "below_ma20_min": 0.0},
        {"name": "绿_vol≥0.7", "universe": "green", "vol_pct_min": 0.70},
        {"name": "绿_涨≥+3%_vol≥0.7", "universe": "green", "day_ret_min": 0.03, "vol_pct_min": 0.70},
        {"name": "绿_量比≥1.5", "universe": "green", "vol_ratio_min": 1.5},
        {"name": "绿_涨≥+3%_量比≥1.5", "universe": "green", "day_ret_min": 0.03, "vol_ratio_min": 1.5},
        {"name": "绿_量比≥2", "universe": "green", "vol_ratio_min": 2.0},
        {"name": "绿_rv扩张", "universe": "green", "require_rv_expand": True},
        {"name": "绿_距红≤10日", "universe": "green_after_red", "after_red_max": 10},
        {"name": "绿_距红≤5日", "universe": "green_after_red", "after_red_max": 5},
        {"name": "绿_涨≥+3%_距红≤10日", "universe": "green_after_red", "day_ret_min": 0.03, "after_red_max": 10},
        {
            "name": "绿_涨≥+3%_近高_量比≥1.5",
            "universe": "green",
            "day_ret_min": 0.03,
            "range_pos_min": 0.80,
            "vol_ratio_min": 1.5,
        },
        {
            "name": "绿_涨≥+5%_牛均线_量比≥1.5",
            "universe": "green",
            "day_ret_min": 0.05,
            "require_ma_bull": True,
            "vol_ratio_min": 1.5,
        },
    ]
    out.extend(green_base)

    # --- either triangle ---
    out.extend(
        [
            {"name": "红或绿_全量", "universe": "red_or_green"},
            {"name": "红或绿_地板或天花", "universe": "red_or_green"},  # too loose; add specific:
        ]
    )
    out[-1] = {"name": "红或绿_vol≥0.8", "universe": "red_or_green", "vol_pct_min": 0.80}
    out.append({"name": "红或绿_量比≥2", "universe": "red_or_green", "vol_ratio_min": 2.0})
    return out


def exit_grid() -> list[dict[str, Any]]:
    grid = []
    for tp_gap, float_tp, max_hold, sl_buf, sl_pct, eg, er in itertools.product(
        (0.02, 0.03, 0.05, 0.08),
        (None, 0.08, 0.12),
        (3, 5, 10),
        (0.01, 0.02),
        (-0.05, -0.06, -0.08),
        (False, True),
        (False,),  # exit_on_red usually bad for long; keep False only to cut grid
    ):
        # skip if both float and huge gap redundant noise — keep all for overfit
        name = (
            f"gap{int(tp_gap*100)}_ft{'NA' if float_tp is None else int(float_tp*100)}"
            f"_h{max_hold}_sb{int(sl_buf*100)}_sl{int(abs(sl_pct)*100)}"
            f"{'_eg' if eg else ''}{'_er' if er else ''}"
        )
        grid.append(
            {
                "name": name,
                "tp_gap": tp_gap,
                "float_tp": float_tp,
                "max_hold": max_hold,
                "sl_buf": sl_buf,
                "sl_pct": sl_pct,
                "exit_on_green": eg,
                "exit_on_red": er,
            }
        )
    return grid


def run_one(
    px: pd.DataFrame,
    filt: dict[str, Any],
    ex: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    trades = []
    busy_until = -1
    for i in range(len(px)):
        if i <= busy_until:
            continue
        row = px.iloc[i]
        if not pass_filter(row, filt):
            continue
        if pd.isna(row["$close"]) or pd.isna(row["$low"]):
            continue
        sim = simulate(
            px,
            i,
            float(row["$close"]),
            float(row["$low"]),
            tp_gap=ex["tp_gap"],
            float_tp=ex["float_tp"],
            max_hold=ex["max_hold"],
            sl_buf=ex["sl_buf"],
            sl_pct=ex["sl_pct"],
            exit_on_green=ex["exit_on_green"],
            exit_on_red=ex["exit_on_red"],
        )
        if sim is None:
            continue
        trades.append(
            {
                "signal_date": str(pd.Timestamp(row["as_of"]).date()),
                "entry": float(row["$close"]),
                "is_red": bool(row["is_red"]),
                "is_green": bool(row["is_green"]),
                "day_ret": float(row["ret1"]) if pd.notna(row["ret1"]) else np.nan,
                "vol_pct": float(row["vol5_pct_120d"]) if pd.notna(row["vol5_pct_120d"]) else np.nan,
                "vol_ratio": float(row["vol_ratio"]) if pd.notna(row["vol_ratio"]) else np.nan,
                "below_ma20": float(row["below_ma20"]) if pd.notna(row["below_ma20"]) else np.nan,
                **sim,
            }
        )
        busy_until = i + int(sim["hold_days"])

    if not trades:
        return pd.DataFrame(), None
    tdf = pd.DataFrame(trades)
    n = len(tdf)
    mean_r = float(tdf["fwd_ret"].mean())
    win = float((tdf["fwd_ret"] > 0).mean())
    compound = float((1.0 + tdf["fwd_ret"]).prod() - 1.0)
    summary = {
        "buy_filter": filt["name"],
        "exit_rule": ex["name"],
        "tp_gap": ex["tp_gap"],
        "float_tp": ex["float_tp"] if ex["float_tp"] is not None else np.nan,
        "max_hold": ex["max_hold"],
        "sl_buf": ex["sl_buf"],
        "sl_pct": ex["sl_pct"],
        "exit_on_green": ex["exit_on_green"],
        "n_trades": n,
        "win_rate": win,
        "mean_ret": mean_r,
        "median_ret": float(tdf["fwd_ret"].median()),
        "compound": compound,
        "mean_hold": float(tdf["hold_days"].mean()),
        "tp_rate": float(tdf["exit_why"].str.startswith("止盈").mean()),
        "sl_rate": float(tdf["exit_why"].str.startswith("止损").mean()),
        # overfit scores
        "score_compound": compound if n >= MIN_N else -999.0,
        "score_mean_sqrtn": mean_r * np.sqrt(n) if n >= MIN_N else -999.0,
    }
    return tdf, summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code", default="SH588710")
    p.add_argument("--start-date", default="2025-06-04")
    p.add_argument("--end-date", default="2026-08-05")
    p.add_argument(
        "--validation-dir",
        type=Path,
        default=_PROJECT_ROOT
        / "runtime/decision_packs/20260720/regime_transition_validation_q90_to0720",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    val = args.validation_dir if args.validation_dir.is_absolute() else _PROJECT_ROOT / args.validation_dir
    out_dir = args.out_dir or (val / f"triangle_overfit_{args.code}_{args.start_date.replace('-','')}_{args.end_date.replace('-','')}")
    if not out_dir.is_absolute():
        out_dir = _PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_mod = _load_plot_mod()
    plot_mod._init_qlib(None)
    oos, _, _ = load_validation_csvs(val)
    ohlcv = plot_mod.load_qlib_ohlcv(
        args.code,
        args.start_date,
        args.end_date,
        init_qlib=False,
        lookback_calendar_days=220,
        clip_to_window=False,
    )
    # clip to requested window for trading, keep lookback features via merge already in load
    px = build_frame(ohlcv, oos, code=args.code)
    px = px[(px["as_of"] >= pd.Timestamp(args.start_date)) & (px["as_of"] <= pd.Timestamp(args.end_date))].reset_index(drop=True)

    n_red = int(px["is_red"].sum())
    n_green = int(px["is_green"].sum())
    bh = float(px.iloc[-1]["$close"] / px.iloc[0]["$close"] - 1.0)
    print(f"[INFO] {args.code} bars={len(px)} red={n_red} green={n_green} buy&hold={bh:+.2%}")

    px.to_csv(out_dir / "bars_features.csv", index=False, encoding="utf-8-sig")

    filters = buy_filters()
    exits = exit_grid()
    print(f"[INFO] filters={len(filters)} exits={len(exits)} combos={len(filters)*len(exits)}")

    summaries: list[dict[str, Any]] = []
    best_trades: dict[str, pd.DataFrame] = {}
    best_by_compound: dict[str, Any] | None = None
    best_by_win: dict[str, Any] | None = None
    best_by_mean: dict[str, Any] | None = None

    for fi, filt in enumerate(filters):
        for ex in exits:
            tdf, summ = run_one(px, filt, ex)
            if summ is None:
                continue
            summaries.append(summ)
            # track bests
            if best_by_compound is None or summ["score_compound"] > best_by_compound["score_compound"]:
                best_by_compound = summ
                best_trades["compound"] = tdf
            if summ["n_trades"] >= MIN_N:
                if best_by_win is None or (
                    summ["win_rate"], summ["compound"], summ["n_trades"]
                ) > (best_by_win["win_rate"], best_by_win["compound"], best_by_win["n_trades"]):
                    best_by_win = summ
                    best_trades["win"] = tdf
                if best_by_mean is None or (
                    summ["mean_ret"], summ["compound"], summ["n_trades"]
                ) > (best_by_mean["mean_ret"], best_by_mean["compound"], best_by_mean["n_trades"]):
                    best_by_mean = summ
                    best_trades["mean"] = tdf
        if (fi + 1) % 10 == 0:
            print(f"[INFO] progress filters {fi+1}/{len(filters)} summaries={len(summaries)}")

    summ_df = pd.DataFrame(summaries)
    summ_path = out_dir / "overfit_summary.csv"
    summ_df.to_csv(summ_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] wrote {summ_path} rows={len(summ_df)}")

    valid = summ_df[summ_df["n_trades"] >= MIN_N].copy()
    top_c = valid.sort_values(["compound", "win_rate", "n_trades"], ascending=False).head(15)
    top_w = valid.sort_values(["win_rate", "compound", "n_trades"], ascending=False).head(15)
    top_m = valid.sort_values(["mean_ret", "compound", "n_trades"], ascending=False).head(15)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("\n=== TOP compound (n>=%d) ===" % MIN_N)
    print(top_c[["buy_filter", "exit_rule", "n_trades", "win_rate", "mean_ret", "compound", "mean_hold"]].to_string(index=False))
    print("\n=== TOP win_rate ===")
    print(top_w[["buy_filter", "exit_rule", "n_trades", "win_rate", "mean_ret", "compound", "mean_hold"]].to_string(index=False))
    print("\n=== TOP mean_ret ===")
    print(top_m[["buy_filter", "exit_rule", "n_trades", "win_rate", "mean_ret", "compound", "mean_hold"]].to_string(index=False))

    # save best trade lists
    for key, tdf in best_trades.items():
        tdf.to_csv(out_dir / f"best_trades_{key}.csv", index=False, encoding="utf-8-sig")

    def fmt(s: dict[str, Any] | None) -> list[str]:
        if not s:
            return ["(无)"]
        return [
            f"- 买点：`{s['buy_filter']}`",
            f"- 卖点：`{s['exit_rule']}`",
            f"- n={int(s['n_trades'])} 胜率={s['win_rate']:.1%} 均值={s['mean_ret']:+.2%} 复利={s['compound']:+.2%} 均持仓={s['mean_hold']:.1f}日",
        ]

    md = [
        f"# {args.code} 红绿三角过拟合回测",
        "",
        f"- 区间：{args.start_date} → {args.end_date}",
        f"- bars={len(px)} 红三角={n_red} 绿三角={n_green}",
        f"- 买入持有基准：{bh:+.2%}",
        f"- 组合数：{len(summ_df)}；门槛 n≥{MIN_N}",
        f"- **性质：样本内过拟合，不可直接外推**",
        "",
        "## 最优（复利）",
        "",
        *fmt(best_by_compound),
        "",
        "## 最优（胜率）",
        "",
        *fmt(best_by_win),
        "",
        "## 最优（均值）",
        "",
        *fmt(best_by_mean),
        "",
        "## Top15 compound",
        "",
        top_c.to_markdown(index=False),
        "",
        "## Top15 win_rate",
        "",
        top_w.to_markdown(index=False),
        "",
        "## Top15 mean_ret",
        "",
        top_m.to_markdown(index=False),
    ]
    if "compound" in best_trades and not best_trades["compound"].empty:
        md.extend(["", "## 复利最优交易明细", "", best_trades["compound"].to_markdown(index=False)])
    (out_dir / "overfit_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[INFO] wrote {out_dir / 'overfit_summary.md'}")

    # print best trade detail
    if "compound" in best_trades:
        print("\n=== 复利最优交易明细 ===")
        print(best_trades["compound"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
