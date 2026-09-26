#!/usr/bin/env python3
"""SH513310 中韩半导体：全指标样本内过拟合买卖点。

区间：有行情日起（约 2022-12-22）→ 2026-08-05。
指标：红/绿三角、均线(MA5/10/20/60)、金叉/死叉、波动分位、量比、
      日内位置、深跌地板、prior 动量、rv 扩张。

两段网格：
  A) 三角过滤 × 短持仓止盈止损（高换手过拟合）
  B) 结构买点底座组合 × 结构/回撤卖点（趋势友好）

明确标注：样本内过拟合，不保证外推。
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
MIN_N_A = 5
MIN_N_B = 3
CODE_DEFAULT = "SH513310"
END_DEFAULT = "2026-08-05"


def _load_plot_mod():
    path = _PROJECT_ROOT / "decision_pack" / "scripts" / "plot_regime_transition_example.py"
    spec = importlib.util.spec_from_file_location("plot_513310_overfit", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_frame(ohlcv: pd.DataFrame, oos: pd.DataFrame, *, code: str) -> pd.DataFrame:
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
    for w in (5, 10, 20, 60):
        out[f"MA{w}"] = out["$close"].rolling(w, min_periods=max(3, w // 2)).mean()
    out["below_ma5"] = out["$close"] / out["MA5"] - 1.0
    out["below_ma10"] = out["$close"] / out["MA10"] - 1.0
    out["below_ma20"] = out["$close"] / out["MA20"] - 1.0
    out["above_ma5"] = out["$close"] > out["MA5"]
    out["above_ma10"] = out["$close"] > out["MA10"]
    out["above_ma20"] = out["$close"] > out["MA20"]
    out["above_ma60"] = out["$close"] > out["MA60"]
    out["ma_bull"] = (out["$close"] > out["MA5"]) & (out["MA5"] > out["MA10"])
    out["ma_bear"] = (out["$close"] < out["MA5"]) & (out["MA5"] < out["MA10"])
    out["golden"] = (out["MA5"] > out["MA20"]) & (out["MA5"].shift(1) <= out["MA20"].shift(1))
    out["death"] = (out["MA5"] < out["MA20"]) & (out["MA5"].shift(1) >= out["MA20"].shift(1))
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
    out["prior60"] = out["$close"] / out["$close"].shift(60) - 1.0

    sub = oos[oos["code"].astype(str).str.upper() == code.upper()].copy()
    sub["as_of"] = pd.to_datetime(sub["as_of"]).dt.normalize()
    keep = [
        c
        for c in (
            "as_of",
            "quantile_switch",
            "switch_side",
            "pred_cdf",
            "p_trend_cal",
            "regime_now",
            "extreme_drop_switch",
        )
        if c in sub.columns
    ]
    out = out.merge(sub[keep], on="as_of", how="left")
    qs = out["quantile_switch"].fillna(False)
    if qs.dtype == object:
        qs = qs.astype(bool)
    side = out["switch_side"].astype(str).str.lower()
    out["is_red"] = qs.astype(bool) & (side == "down")
    out["is_green"] = qs.astype(bool) & (side == "up")

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


# ---------- A: short-horizon triangle overfit ----------


def pass_filter(row: pd.Series, filt: dict[str, Any]) -> bool:
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
    if uni == "golden" and not bool(row["golden"]):
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
    if filt.get("require_above_ma20") and not bool(row["above_ma20"]):
        return False

    for key, col, op in (
        ("day_ret_max", "ret1", "le"),
        ("day_ret_min", "ret1", "ge"),
        ("below_ma20_max", "below_ma20", "le"),
        ("below_ma20_min", "below_ma20", "ge"),
        ("vol_pct_min", "vol5_pct_120d", "ge"),
        ("vol_pct_max", "vol5_pct_120d", "le"),
        ("vol_ratio_min", "vol_ratio", "ge"),
        ("range_pos_max", "range_pos", "le"),
        ("range_pos_min", "range_pos", "ge"),
        ("prior10_max", "prior10", "le"),
        ("prior10_min", "prior10", "ge"),
        ("pred_cdf_min", "pred_cdf", "ge"),
        ("pred_cdf_max", "pred_cdf", "le"),
        ("p_trend_min", "p_trend_cal", "ge"),
    ):
        if filt.get(key) is None:
            continue
        v = row.get(col)
        if pd.isna(v):
            return False
        thr = float(filt[key])
        fv = float(v)
        if op == "le" and not (fv <= thr):
            return False
        if op == "ge" and not (fv >= thr):
            return False
    return True


def simulate_short(
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
            return {"exit_date": d, "exit_why": "止盈_高开", "exit_px": o, "hold_days": j - i, "fwd_ret": o / entry - 1.0}
        if float_tp is not None and h / entry - 1.0 >= float_tp:
            return {
                "exit_date": d,
                "exit_why": "止盈_浮盈",
                "exit_px": entry * (1.0 + float_tp),
                "hold_days": j - i,
                "fwd_ret": float_tp,
            }
        if exit_on_green and bool(bar["is_green"]):
            return {"exit_date": d, "exit_why": "止盈_绿三角", "exit_px": c, "hold_days": j - i, "fwd_ret": c / entry - 1.0}
        if c < stop_line:
            return {"exit_date": d, "exit_why": "止损_破买点低", "exit_px": c, "hold_days": j - i, "fwd_ret": c / entry - 1.0}
        if c / entry - 1.0 <= sl_pct:
            return {"exit_date": d, "exit_why": "止损_浮亏", "exit_px": c, "hold_days": j - i, "fwd_ret": c / entry - 1.0}
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


def buy_filters_a() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    out.extend(
        [
            {"name": "红_全量", "universe": "red"},
            {"name": "红_跌≤-5%", "universe": "red", "day_ret_max": -0.05},
            {"name": "红_跌≤-7%", "universe": "red", "day_ret_max": -0.07},
            {"name": "红_跌≤-7%_地板", "universe": "red", "day_ret_max": -0.07, "require_floor": True},
            {"name": "红_地板", "universe": "red", "require_floor": True},
            {"name": "红_近低", "universe": "red", "range_pos_max": 0.15},
            {"name": "红_仍上MA20", "universe": "red", "require_above_ma20": True},
            {"name": "红_vol≥0.7", "universe": "red", "vol_pct_min": 0.70},
            {"name": "红_vol≥0.8", "universe": "red", "vol_pct_min": 0.80},
            {"name": "红_跌≤-7%_vol≥0.7", "universe": "red", "day_ret_max": -0.07, "vol_pct_min": 0.70},
            {"name": "红_量比≥1.5", "universe": "red", "vol_ratio_min": 1.5},
            {"name": "红_量比≥2", "universe": "red", "vol_ratio_min": 2.0},
            {"name": "红_跌≤-5%_量比≥1.5", "universe": "red", "day_ret_max": -0.05, "vol_ratio_min": 1.5},
            {"name": "红_熊均线", "universe": "red", "require_ma_bear": True},
            {"name": "红_rv扩张", "universe": "red", "require_rv_expand": True},
            {"name": "红_prior10≤-10%", "universe": "red", "prior10_max": -0.10},
            {"name": "红_距绿≤10日", "universe": "red_after_green", "after_green_max": 10},
            {"name": "红_跌≤-5%_近低_vol≥0.7", "universe": "red", "day_ret_max": -0.05, "range_pos_max": 0.20, "vol_pct_min": 0.70},
            {"name": "绿_全量", "universe": "green"},
            {"name": "绿_涨≥+3%", "universe": "green", "day_ret_min": 0.03},
            {"name": "绿_涨≥+5%", "universe": "green", "day_ret_min": 0.05},
            {"name": "绿_牛均线", "universe": "green", "require_ma_bull": True},
            {"name": "绿_涨≥+3%_牛均线", "universe": "green", "day_ret_min": 0.03, "require_ma_bull": True},
            {"name": "绿_站上MA20", "universe": "green", "below_ma20_min": 0.0},
            {"name": "绿_vol≥0.7", "universe": "green", "vol_pct_min": 0.70},
            {"name": "绿_涨≥+3%_vol≥0.7", "universe": "green", "day_ret_min": 0.03, "vol_pct_min": 0.70},
            {"name": "绿_量比≥1.5", "universe": "green", "vol_ratio_min": 1.5},
            {"name": "绿_牛均线_vol≥0.5", "universe": "green", "require_ma_bull": True, "vol_pct_min": 0.50},
            {"name": "绿_牛均线_量比≥1.5", "universe": "green", "require_ma_bull": True, "vol_ratio_min": 1.5},
            {"name": "绿_距红≤10日", "universe": "green_after_red", "after_red_max": 10},
            {"name": "绿_涨≥+3%_近高_量比≥1.5", "universe": "green", "day_ret_min": 0.03, "range_pos_min": 0.80, "vol_ratio_min": 1.5},
            {"name": "金叉_全量", "universe": "golden"},
            {"name": "金叉_vol≥0.5", "universe": "golden", "vol_pct_min": 0.50},
            {"name": "金叉_vol≥0.7", "universe": "golden", "vol_pct_min": 0.70},
            {"name": "金叉_量比≥1.2", "universe": "golden", "vol_ratio_min": 1.2},
            {"name": "红或绿_vol≥0.8", "universe": "red_or_green", "vol_pct_min": 0.80},
            {"name": "红或绿_量比≥2", "universe": "red_or_green", "vol_ratio_min": 2.0},
        ]
    )
    return out


def exit_grid_a() -> list[dict[str, Any]]:
    grid = []
    for tp_gap, float_tp, max_hold, sl_buf, sl_pct, eg in itertools.product(
        (0.03, 0.05, 0.08),
        (None, 0.08, 0.12, 0.20),
        (5, 10, 20),
        (0.01, 0.02),
        (-0.06, -0.08, -0.12),
        (False, True),
    ):
        name = (
            f"gap{int(tp_gap*100)}_ft{'NA' if float_tp is None else int(float_tp*100)}"
            f"_h{max_hold}_sb{int(sl_buf*100)}_sl{int(abs(sl_pct)*100)}"
            f"{'_eg' if eg else ''}"
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
            }
        )
    return grid


def run_a(px: pd.DataFrame, filt: dict[str, Any], ex: dict[str, Any]) -> tuple[pd.DataFrame, dict | None]:
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
        sim = simulate_short(
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
        )
        if sim is None:
            continue
        trades.append(
            {
                "signal_date": str(pd.Timestamp(row["as_of"]).date()),
                "entry": float(row["$close"]),
                "is_red": bool(row["is_red"]),
                "is_green": bool(row["is_green"]),
                "golden": bool(row["golden"]),
                "day_ret": float(row["ret1"]) if pd.notna(row["ret1"]) else np.nan,
                "vol_pct": float(row["vol5_pct_120d"]) if pd.notna(row["vol5_pct_120d"]) else np.nan,
                "vol_ratio": float(row["vol_ratio"]) if pd.notna(row["vol_ratio"]) else np.nan,
                **sim,
            }
        )
        busy_until = i + int(sim["hold_days"])
    if not trades:
        return pd.DataFrame(), None
    tdf = pd.DataFrame(trades)
    n = len(tdf)
    mean_r = float(tdf["fwd_ret"].mean())
    compound = float((1.0 + tdf["fwd_ret"]).prod() - 1.0)
    summary = {
        "family": "A_短持仓",
        "buy_filter": filt["name"],
        "exit_rule": ex["name"],
        "n_trades": n,
        "win_rate": float((tdf["fwd_ret"] > 0).mean()),
        "mean_ret": mean_r,
        "median_ret": float(tdf["fwd_ret"].median()),
        "compound": compound,
        "mean_hold": float(tdf["hold_days"].mean()),
        "score_compound": compound if n >= MIN_N_A else -999.0,
    }
    return tdf, summary


# ---------- B: structural buy bases × exits ----------


def struct_buy_why(row: pd.Series, cfg: dict[str, Any]) -> str | None:
    vol_min = cfg.get("vol_pct_min")
    if vol_min is not None:
        if pd.isna(row["vol5_pct_120d"]) or float(row["vol5_pct_120d"]) < float(vol_min):
            return None
    vr_min = cfg.get("vol_ratio_min")
    if vr_min is not None:
        if pd.isna(row["vol_ratio"]) or float(row["vol_ratio"]) < float(vr_min):
            return None

    if cfg.get("use_green_bull") and bool(row["is_green"]) and bool(row["ma_bull"]):
        return "绿三角_牛均线"
    if cfg.get("use_red_ma20") and bool(row["is_red"]) and bool(row["above_ma20"]):
        return "红三角_仍上MA20"
    if (
        cfg.get("use_red_floor")
        and bool(row["is_red"])
        and pd.notna(row["ret1"])
        and float(row["ret1"]) <= float(cfg.get("floor_ret", -0.07))
        and bool(row["close_eq_low"])
    ):
        return "红三角_深跌地板"
    if cfg.get("use_golden") and bool(row["golden"]):
        return "金叉_MA5上穿MA20"
    return None


def struct_should_exit(
    bar: pd.Series,
    *,
    exit_mode: str,
    trail_pct: float,
    peak: float,
    hold_days: int,
) -> tuple[bool, str]:
    c = float(bar["$close"])
    if hold_days <= 0:
        return False, ""
    if exit_mode == "trail":
        if c <= peak * (1.0 - trail_pct):
            return True, f"峰值回撤{int(trail_pct * 100)}%"
        return False, ""
    if exit_mode == "red_ma10":
        if bool(bar["is_red"]) and (not bool(bar["above_ma10"])):
            return True, "红破MA10"
        return False, ""
    if exit_mode == "red_ma20":
        if bool(bar["is_red"]) and (not bool(bar["above_ma20"])):
            return True, "红破MA20"
        return False, ""
    if exit_mode == "break_ma20":
        if not bool(bar["above_ma20"]):
            return True, "破MA20"
        return False, ""
    if exit_mode == "death":
        if bool(bar["death"]):
            return True, "死叉_MA5下穿MA20"
        return False, ""
    if exit_mode == "trail_or_red_ma10":
        if c <= peak * (1.0 - trail_pct):
            return True, f"峰值回撤{int(trail_pct * 100)}%"
        if bool(bar["is_red"]) and (not bool(bar["above_ma10"])):
            return True, "红破MA10"
        return False, ""
    if exit_mode == "trail_or_red_ma20":
        if c <= peak * (1.0 - trail_pct):
            return True, f"峰值回撤{int(trail_pct * 100)}%"
        if bool(bar["is_red"]) and (not bool(bar["above_ma20"])):
            return True, "红破MA20"
        return False, ""
    raise ValueError(exit_mode)


def run_b(px: pd.DataFrame, buy_cfg: dict[str, Any], exit_mode: str, trail_pct: float) -> tuple[pd.DataFrame, dict | None]:
    cash = 1.0
    pos = 0.0
    entry = None
    entry_i = None
    peak = None
    entry_why = ""
    events: list[dict] = []

    for i in range(len(px)):
        bar = px.iloc[i]
        c = float(bar["$close"])
        h = float(bar["$high"])
        d = str(pd.Timestamp(bar["as_of"]).date())
        if pos == 0:
            why = struct_buy_why(bar, buy_cfg)
            if why:
                entry = c
                entry_i = i
                peak = h
                entry_why = why
                pos = cash / c
                cash = 0.0
                events.append({"side": "BUY", "date": d, "px": c, "why": why})
            continue
        peak = max(peak, h)
        ok, why = struct_should_exit(
            bar, exit_mode=exit_mode, trail_pct=trail_pct, peak=peak, hold_days=i - entry_i
        )
        if ok:
            cash = pos * c
            ret = c / entry - 1.0
            events.append(
                {
                    "side": "SELL",
                    "date": d,
                    "px": c,
                    "why": why,
                    "entry_date": str(pd.Timestamp(px.iloc[entry_i]["as_of"]).date()),
                    "entry_px": entry,
                    "entry_why": entry_why,
                    "hold_days": i - entry_i,
                    "fwd_ret": ret,
                }
            )
            pos = 0.0
            entry = entry_i = peak = None
            entry_why = ""

    if pos > 0:
        c = float(px.iloc[-1]["$close"])
        cash = pos * c
        events.append(
            {
                "side": "SELL",
                "date": str(pd.Timestamp(px.iloc[-1]["as_of"]).date()),
                "px": c,
                "why": "期末强平",
                "entry_date": str(pd.Timestamp(px.iloc[entry_i]["as_of"]).date()),
                "entry_px": entry,
                "entry_why": entry_why,
                "hold_days": len(px) - 1 - entry_i,
                "fwd_ret": c / entry - 1.0,
            }
        )
        pos = 0.0

    sells = [e for e in events if e["side"] == "SELL" and "fwd_ret" in e]
    if not sells:
        return pd.DataFrame(events), None
    tdf = pd.DataFrame(sells)
    n = len(tdf)
    compound = float(cash - 1.0)
    bh = float(px.iloc[-1]["$close"] / px.iloc[0]["$close"] - 1.0)
    buy_name = buy_cfg["name"]
    exit_name = exit_mode if not exit_mode.startswith("trail") else f"{exit_mode}{int(trail_pct*100)}"
    if exit_mode.startswith("trail_or"):
        exit_name = f"{exit_mode}_t{int(trail_pct*100)}"
    summary = {
        "family": "B_结构",
        "buy_filter": buy_name,
        "exit_rule": exit_name,
        "n_trades": n,
        "win_rate": float((tdf["fwd_ret"] > 0).mean()),
        "mean_ret": float(tdf["fwd_ret"].mean()),
        "median_ret": float(tdf["fwd_ret"].median()),
        "compound": compound,
        "bh": bh,
        "edge": compound - bh,
        "mean_hold": float(tdf["hold_days"].mean()),
        "score_compound": compound if n >= MIN_N_B else -999.0,
        "score_edge": (compound - bh) if n >= MIN_N_B else -999.0,
    }
    return pd.DataFrame(events), summary


def buy_cfgs_b() -> list[dict[str, Any]]:
    cfgs = []
    # all subsets of 4 bases
    bases = ["use_green_bull", "use_red_ma20", "use_red_floor", "use_golden"]
    for mask in range(1, 16):
        flags = {b: bool(mask & (1 << i)) for i, b in enumerate(bases)}
        label_parts = []
        if flags["use_green_bull"]:
            label_parts.append("绿牛")
        if flags["use_red_ma20"]:
            label_parts.append("红上MA20")
        if flags["use_red_floor"]:
            label_parts.append("红地板")
        if flags["use_golden"]:
            label_parts.append("金叉")
        for vol in (None, 0.5, 0.7):
            for vr in (None, 1.2, 1.5):
                name = "+".join(label_parts)
                if vol is not None:
                    name += f"_vol≥{vol}"
                if vr is not None:
                    name += f"_量比≥{vr}"
                cfgs.append({**flags, "name": name, "vol_pct_min": vol, "vol_ratio_min": vr})
    return cfgs


def exit_cfgs_b() -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for mode in ("red_ma10", "red_ma20", "break_ma20", "death"):
        out.append((mode, 0.25))
    for t in (0.15, 0.20, 0.25, 0.30, 0.35):
        out.append(("trail", t))
        out.append(("trail_or_red_ma10", t))
        out.append(("trail_or_red_ma20", t))
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code", default=CODE_DEFAULT)
    p.add_argument("--start-date", default=None, help="默认=行情首日")
    p.add_argument("--end-date", default=END_DEFAULT)
    p.add_argument(
        "--validation-dir",
        type=Path,
        default=_PROJECT_ROOT
        / "runtime/decision_packs/20260720/regime_transition_validation_q90_to0720",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--skip-a", action="store_true", help="跳过短持仓网格")
    p.add_argument("--skip-b", action="store_true", help="跳过结构网格")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    val = args.validation_dir if args.validation_dir.is_absolute() else _PROJECT_ROOT / args.validation_dir
    plot_mod = _load_plot_mod()
    plot_mod._init_qlib(None)
    oos, _, _ = load_validation_csvs(val)

    probe = plot_mod.load_qlib_ohlcv(
        args.code, "2020-01-01", args.end_date, init_qlib=False, lookback_calendar_days=30, clip_to_window=True
    )
    data_start = str(pd.to_datetime(probe["datetime"]).min().date())
    start = args.start_date or data_start
    print(f"[INFO] data_start={data_start} fit_start={start} end={args.end_date}")

    out_dir = args.out_dir or (
        val / f"triangle_overfit_{args.code}_{start.replace('-', '')}_{args.end_date.replace('-', '')}"
    )
    if not out_dir.is_absolute():
        out_dir = _PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ohlcv = plot_mod.load_qlib_ohlcv(
        args.code, start, args.end_date, init_qlib=False, lookback_calendar_days=220, clip_to_window=False
    )
    px_full = build_frame(ohlcv, oos, code=args.code)
    px = px_full[
        (px_full["as_of"] >= pd.Timestamp(start)) & (px_full["as_of"] <= pd.Timestamp(args.end_date))
    ].reset_index(drop=True)
    bh = float(px.iloc[-1]["$close"] / px.iloc[0]["$close"] - 1.0)
    print(
        f"[INFO] {args.code} bars={len(px)} red={int(px.is_red.sum())} green={int(px.is_green.sum())} "
        f"golden={int(px.golden.sum())} BH={bh:+.2%}"
    )
    px.to_csv(out_dir / "bars_features.csv", index=False, encoding="utf-8-sig")

    summaries: list[dict] = []
    best_store: dict[str, pd.DataFrame] = {}

    if not args.skip_a:
        filters = buy_filters_a()
        exits = exit_grid_a()
        print(f"[INFO] A-grid filters={len(filters)} exits={len(exits)} combos={len(filters)*len(exits)}")
        best_a = None
        best_a_tr = None
        for fi, filt in enumerate(filters):
            for ex in exits:
                tdf, summary = run_a(px, filt, ex)
                if summary is None:
                    continue
                summaries.append(summary)
                if best_a is None or summary["score_compound"] > best_a["score_compound"]:
                    best_a = summary
                    best_a_tr = tdf
            if (fi + 1) % 10 == 0:
                print(f"  A progress {fi+1}/{len(filters)} best_compound={best_a['compound'] if best_a else None}")
        if best_a_tr is not None:
            best_store["A_best"] = best_a_tr
            best_a_tr.to_csv(out_dir / "A_best_trades.csv", index=False, encoding="utf-8-sig")
            print(f"[INFO] A best: {best_a['buy_filter']} | {best_a['exit_rule']} compound={best_a['compound']:+.2%} n={best_a['n_trades']}")

    if not args.skip_b:
        buys = buy_cfgs_b()
        exits = exit_cfgs_b()
        print(f"[INFO] B-grid buys={len(buys)} exits={len(exits)} combos={len(buys)*len(exits)}")
        best_b = None
        best_b_edge = None
        best_b_ev = None
        best_b_edge_ev = None
        for bi, buy_cfg in enumerate(buys):
            for exit_mode, trail in exits:
                ev, summary = run_b(px, buy_cfg, exit_mode, trail)
                if summary is None:
                    continue
                summaries.append(summary)
                if best_b is None or summary["score_compound"] > best_b["score_compound"]:
                    best_b = summary
                    best_b_ev = ev
                if best_b_edge is None or summary["score_edge"] > best_b_edge["score_edge"]:
                    best_b_edge = summary
                    best_b_edge_ev = ev
            if (bi + 1) % 30 == 0:
                print(
                    f"  B progress {bi+1}/{len(buys)} best_compound={best_b['compound'] if best_b else None} "
                    f"best_edge={best_b_edge['edge'] if best_b_edge else None}"
                )
        if best_b_ev is not None:
            best_store["B_best_compound"] = best_b_ev
            best_b_ev.to_csv(out_dir / "B_best_compound_trades.csv", index=False, encoding="utf-8-sig")
            print(
                f"[INFO] B best compound: {best_b['buy_filter']} | {best_b['exit_rule']} "
                f"compound={best_b['compound']:+.2%} edge={best_b['edge']:+.2%} n={best_b['n_trades']}"
            )
        if best_b_edge_ev is not None:
            best_store["B_best_edge"] = best_b_edge_ev
            best_b_edge_ev.to_csv(out_dir / "B_best_edge_trades.csv", index=False, encoding="utf-8-sig")
            print(
                f"[INFO] B best edge: {best_b_edge['buy_filter']} | {best_b_edge['exit_rule']} "
                f"compound={best_b_edge['compound']:+.2%} edge={best_b_edge['edge']:+.2%} n={best_b_edge['n_trades']}"
            )

    sdf = pd.DataFrame(summaries)
    if not sdf.empty:
        sdf = sdf.sort_values("compound", ascending=False)
        sdf.to_csv(out_dir / "sweep_summary.csv", index=False, encoding="utf-8-sig")

    # markdown report
    lines = [
        f"# SH513310 中韩半导体：全指标过拟合买卖点",
        "",
        f"> **样本内过拟合**，不保证外推。",
        f"> 行情首日 `{data_start}`；拟合窗口 `{start}` → `{args.end_date}`；买入持有 **{bh:+.2%}**。",
        f"> 三角信号 OOS 约自 2024-06 起；此前仅金叉/均线类买点可触发。",
        "",
        "## 指标池",
        "- 红/绿三角、金叉/死叉、MA5/10/20/60、牛/熊均线",
        "- vol5_pct_120d、量比、rv扩张、日内位置、深跌地板、prior5/10/60",
        "",
    ]
    if not sdf.empty:
        top_a = sdf[sdf.family == "A_短持仓"].head(8) if "family" in sdf.columns else pd.DataFrame()
        top_b = sdf[sdf.family == "B_结构"].head(8) if "family" in sdf.columns else pd.DataFrame()
        beat = sdf[(sdf.family == "B_结构") & (sdf.get("edge", pd.Series(dtype=float)) > 0)].head(10)

        lines.append("## A 族 Top（短持仓三角/金叉过滤）")
        lines.append("")
        if not top_a.empty:
            lines.append("| 买点 | 卖点 | n | 胜率 | 复利 |")
            lines.append("|---|---|---:|---:|---:|")
            for _, r in top_a.iterrows():
                lines.append(
                    f"| {r.buy_filter} | `{r.exit_rule}` | {int(r.n_trades)} | {r.win_rate:.0%} | {r.compound:+.1%} |"
                )
        lines.append("")
        lines.append("## B 族 Top（结构买点 × 结构/回撤卖）")
        lines.append("")
        if not top_b.empty:
            lines.append("| 买点 | 卖点 | n | 胜率 | 复利 | vs持有 |")
            lines.append("|---|---|---:|---:|---:|---:|")
            for _, r in top_b.iterrows():
                edge = r.edge if pd.notna(r.get("edge", np.nan)) else np.nan
                lines.append(
                    f"| {r.buy_filter} | `{r.exit_rule}` | {int(r.n_trades)} | {r.win_rate:.0%} | "
                    f"{r.compound:+.1%} | {edge:+.1%} |"
                )
        lines.append("")
        lines.append("## 推荐定稿（过拟合最优）")
        lines.append("")
        # pick overall: prefer B that beats BH if any, else max compound among B, else A
        pick = None
        if not beat.empty:
            pick = beat.iloc[0].to_dict()
            lines.append("优先：**能跑赢持有的结构组合**（样本内）。")
        elif not top_b.empty:
            pick = top_b.iloc[0].to_dict()
            lines.append("未稳定跑赢持有；取 **B 复利最高**。")
        elif not top_a.empty:
            pick = top_a.iloc[0].to_dict()
            lines.append("取 **A 复利最高**（短持仓；通常难跑赢长牛）。")
        if pick:
            lines.extend(
                [
                    "",
                    f"- **买**：`{pick['buy_filter']}`",
                    f"- **卖**：`{pick['exit_rule']}`",
                    f"- **n**={int(pick['n_trades'])} 胜率={pick['win_rate']:.0%} 复利={pick['compound']:+.2%}",
                ]
            )
            if pd.notna(pick.get("edge", np.nan)):
                lines.append(f"- **vs 持有**={pick['edge']:+.2%}（持有 {bh:+.2%}）")

            # attach trade table from matching best file
            key = None
            if pick.get("family") == "B_结构":
                if best_store.get("B_best_edge") is not None and pick.get("edge", -999) == best_b_edge.get("edge"):
                    key = "B_best_edge"
                else:
                    key = "B_best_compound"
            else:
                key = "A_best"
            tr = best_store.get(key)
            if tr is not None and not tr.empty:
                lines.append("")
                lines.append("### 交易明细")
                lines.append("")
                if "side" in tr.columns:
                    lines.append("| 买卖 | 日期 | 价 | 原因 | 收益 |")
                    lines.append("|---|---|---:|---|---:|")
                    for _, r in tr.iterrows():
                        if r["side"] == "BUY":
                            lines.append(f"| 买 | {r['date']} | {float(r['px']):.3f} | {r['why']} | — |")
                        else:
                            ret = r.get("fwd_ret", np.nan)
                            ret_s = f"{float(ret):+.1%}" if pd.notna(ret) else "—"
                            lines.append(f"| 卖 | {r['date']} | {float(r['px']):.3f} | {r['why']} | {ret_s} |")
                else:
                    lines.append("| 信号日 | 入场 | 出场日 | 原因 | 收益 |")
                    lines.append("|---|---:|---|---|---:|")
                    for _, r in tr.iterrows():
                        lines.append(
                            f"| {r['signal_date']} | {float(r['entry']):.3f} | {r['exit_date']} | "
                            f"{r['exit_why']} | {float(r['fwd_ret']):+.1%} |"
                        )

        lines.extend(
            [
                "",
                "## 产物",
                "",
                f"```text",
                f"{out_dir.relative_to(_PROJECT_ROOT)}/",
                f"  sweep_summary.csv",
                f"  bars_features.csv",
                f"  A_best_trades.csv / B_best_*_trades.csv",
                f"  summary.md",
                f"```",
                "",
                "## 版本",
                "",
                "| 项 | 值 |",
                "|---|---|",
                f"| 标的 | SH513310 中韩半导体ETF华泰柏瑞 |",
                f"| 窗口 | {start} → {args.end_date} |",
                f"| 状态 | 单票样本内过拟合 |",
            ]
        )

    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # also copy to documents
    doc = _PROJECT_ROOT / "documents" / "SH513310_全指标过拟合买卖点.md"
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] wrote {summary_path}")
    print(f"[INFO] wrote {doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
