#!/usr/bin/env python3
"""全品种结构买卖点回测（支持分轨卖点）。

买（定稿）：
  - 金叉 MA5 上穿 MA20，且 vol5_pct_120d ≥ 0.5
  - 或 绿三角且牛均线 C>MA5>MA10，且 vol5_pct_120d ≥ 0.5

卖：
  - ma10：红三角且破 MA10（旧定稿）
  - ma20：红三角且破 MA20
  - adaptive（分轨，默认）：开仓当日按趋势/波动强度锁定卖点均线
      * 强趋势轨 → 红破 MA20
      * 否则波段轨 → 红破 MA10

强趋势判定（开仓日、因果）：
  prior60≥15% 且 C>MA60 且 MA20>MA60
  或（高波动趋势）vol5_pct_120d≥0.70 且 C>MA60 且 prior60≥5%
"""
from __future__ import annotations

import argparse
import importlib.util
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

VOL_PCT_MIN = 0.50
PRIOR60_STRONG = 0.15
PRIOR60_HIVOL = 0.05
VOL_PCT_HIVOL = 0.70


def _load_plot_mod():
    path = _PROJECT_ROOT / "decision_pack" / "scripts" / "plot_regime_transition_example.py"
    spec = importlib.util.spec_from_file_location("plot_struct_pool", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_px(code: str, ohlcv: pd.DataFrame, oos_code: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.copy()
    df["as_of"] = pd.to_datetime(df["datetime"]).dt.normalize()
    df = df.sort_values("as_of")
    df = df[~df["as_of"].duplicated(keep="last")].reset_index(drop=True)
    for c in ("$open", "$high", "$low", "$close", "$volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    close = df.set_index("as_of")["$close"]
    vol = compute_volatility_regime_frame(close)
    vf = vol.copy()
    vf["as_of"] = pd.to_datetime(vf["as_of"]).dt.normalize()
    px = df.merge(
        vf[["as_of", "rv5", "rv20", "vol5_pct_120d", "vol_regime"]],
        on="as_of",
        how="left",
    )
    px["ret1"] = px["$close"].pct_change(fill_method=None)
    for w in (5, 10, 20, 60):
        px[f"MA{w}"] = px["$close"].rolling(w, min_periods=max(3, w // 2)).mean()
    px["above_ma5"] = px["$close"] > px["MA5"]
    px["above_ma10"] = px["$close"] > px["MA10"]
    px["above_ma20"] = px["$close"] > px["MA20"]
    px["above_ma60"] = px["$close"] > px["MA60"]
    px["ma_bull"] = (px["$close"] > px["MA5"]) & (px["MA5"] > px["MA10"])
    px["golden"] = (px["MA5"] > px["MA20"]) & (px["MA5"].shift(1) <= px["MA20"].shift(1))
    px["prior60"] = px["$close"] / px["$close"].shift(60) - 1.0
    px["ma20_above_ma60"] = px["MA20"] > px["MA60"]
    px["rv_expand"] = px["rv5"] > px["rv20"]

    sub = oos_code.copy()
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
    px = px.merge(sub[keep], on="as_of", how="left")
    qs = px["quantile_switch"].fillna(False)
    if qs.dtype == object:
        qs = qs.astype(bool)
    side = px["switch_side"].astype(str).str.lower()
    px["is_red"] = qs.astype(bool) & (side == "down")
    px["is_green"] = qs.astype(bool) & (side == "up")
    return px


def buy_why(row: pd.Series) -> str | None:
    vol_ok = pd.notna(row["vol5_pct_120d"]) and float(row["vol5_pct_120d"]) >= VOL_PCT_MIN
    if not vol_ok:
        return None
    if bool(row["is_green"]) and bool(row["ma_bull"]):
        return "绿三角_牛均线"
    if bool(row["golden"]):
        return "金叉_MA5上穿MA20"
    return None


def classify_sell_track(row: pd.Series) -> tuple[int, str]:
    """开仓日锁定卖点轨道：返回 (ma_n, reason)。"""
    prior60 = float(row["prior60"]) if pd.notna(row["prior60"]) else np.nan
    vol_pct = float(row["vol5_pct_120d"]) if pd.notna(row["vol5_pct_120d"]) else np.nan
    above60 = bool(row["above_ma60"]) if pd.notna(row["above_ma60"]) else False
    ma20_gt_60 = bool(row["ma20_above_ma60"]) if pd.notna(row["ma20_above_ma60"]) else False

    strong = (
        pd.notna(prior60)
        and prior60 >= PRIOR60_STRONG
        and above60
        and ma20_gt_60
    )
    hivol_trend = (
        above60
        and pd.notna(vol_pct)
        and vol_pct >= VOL_PCT_HIVOL
        and pd.notna(prior60)
        and prior60 >= PRIOR60_HIVOL
    )
    if strong:
        return 20, "强趋势轨_红破MA20"
    if hivol_trend:
        return 20, "高波趋势轨_红破MA20"
    return 10, "波段轨_红破MA10"


def should_sell(row: pd.Series, sell_ma: int) -> bool:
    if not bool(row["is_red"]):
        return False
    if sell_ma <= 10:
        return not bool(row["above_ma10"])
    return not bool(row["above_ma20"])


def simulate_code(
    px: pd.DataFrame,
    *,
    start: str,
    end: str,
    sell_mode: str = "adaptive",
) -> tuple[list[dict], dict[str, Any]]:
    work = px[(px["as_of"] >= pd.Timestamp(start)) & (px["as_of"] <= pd.Timestamp(end))].reset_index(
        drop=True
    )
    if work.empty:
        return [], {"n_bars": 0, "compound": 0.0, "bh": np.nan, "n_trades": 0}

    cash = 1.0
    pos = 0.0
    entry = None
    entry_i = None
    entry_why = ""
    sell_ma = 10
    sell_track = ""
    trades: list[dict] = []
    track_counts = {"ma10": 0, "ma20": 0}

    for i in range(len(work)):
        bar = work.iloc[i]
        c = float(bar["$close"])
        d = str(pd.Timestamp(bar["as_of"]).date())
        if pos == 0:
            why = buy_why(bar)
            if why:
                if sell_mode == "ma10":
                    sell_ma, sell_track = 10, "固定_红破MA10"
                elif sell_mode == "ma20":
                    sell_ma, sell_track = 20, "固定_红破MA20"
                elif sell_mode == "adaptive":
                    sell_ma, sell_track = classify_sell_track(bar)
                else:
                    raise ValueError(f"unknown sell_mode={sell_mode}")
                track_counts["ma10" if sell_ma == 10 else "ma20"] += 1
                entry = c
                entry_i = i
                entry_why = why
                pos = cash / c
                cash = 0.0
                trades.append(
                    {
                        "side": "BUY",
                        "date": d,
                        "px": c,
                        "why": why,
                        "sell_ma": sell_ma,
                        "sell_track": sell_track,
                        "prior60": float(bar["prior60"]) if pd.notna(bar["prior60"]) else np.nan,
                        "vol_pct": float(bar["vol5_pct_120d"])
                        if pd.notna(bar["vol5_pct_120d"])
                        else np.nan,
                        "above_ma60": bool(bar["above_ma60"]),
                    }
                )
            continue

        if i > entry_i and should_sell(bar, sell_ma):
            cash = pos * c
            trades.append(
                {
                    "side": "SELL",
                    "date": d,
                    "px": c,
                    "why": f"红破MA{sell_ma}",
                    "sell_ma": sell_ma,
                    "sell_track": sell_track,
                    "entry_date": str(pd.Timestamp(work.iloc[entry_i]["as_of"]).date()),
                    "entry_px": entry,
                    "entry_why": entry_why,
                    "hold_days": i - entry_i,
                    "fwd_ret": c / entry - 1.0,
                }
            )
            pos = 0.0
            entry = None
            entry_i = None
            entry_why = ""
            sell_track = ""

    if pos > 0:
        c = float(work.iloc[-1]["$close"])
        cash = pos * c
        trades.append(
            {
                "side": "SELL",
                "date": str(pd.Timestamp(work.iloc[-1]["as_of"]).date()),
                "px": c,
                "why": "期末强平",
                "sell_ma": sell_ma,
                "sell_track": sell_track,
                "entry_date": str(pd.Timestamp(work.iloc[entry_i]["as_of"]).date()),
                "entry_px": entry,
                "entry_why": entry_why,
                "hold_days": len(work) - 1 - entry_i,
                "fwd_ret": c / entry - 1.0,
            }
        )

    compound = cash - 1.0
    bh = float(work.iloc[-1]["$close"] / work.iloc[0]["$close"] - 1.0)
    sells = [t for t in trades if t["side"] == "SELL"]
    win = float(np.mean([t["fwd_ret"] > 0 for t in sells])) if sells else float("nan")
    mean_r = float(np.mean([t["fwd_ret"] for t in sells])) if sells else float("nan")
    return trades, {
        "n_bars": len(work),
        "first": str(pd.Timestamp(work.iloc[0]["as_of"]).date()),
        "last": str(pd.Timestamp(work.iloc[-1]["as_of"]).date()),
        "compound": compound,
        "bh": bh,
        "edge": compound - bh if pd.notna(bh) else np.nan,
        "n_trades": len(sells),
        "win_rate": win,
        "mean_ret": mean_r,
        "beat_bh": bool(compound > bh) if pd.notna(bh) else False,
        "n_track_ma10": track_counts["ma10"],
        "n_track_ma20": track_counts["ma20"],
        "sell_mode": sell_mode,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start-date", default="2024-01-01")
    p.add_argument("--end-date", default="2026-08-05")
    p.add_argument(
        "--sell-mode",
        default="adaptive",
        choices=("adaptive", "ma10", "ma20"),
        help="adaptive=分轨；ma10/ma20=固定红破对应均线",
    )
    p.add_argument(
        "--codes",
        default="",
        help="逗号分隔代码；空=全池",
    )
    p.add_argument(
        "--validation-dir",
        type=Path,
        default=_PROJECT_ROOT
        / "workspace/decision_packs/20260720/regime_transition_validation_q90_to0720",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    val = args.validation_dir if args.validation_dir.is_absolute() else _PROJECT_ROOT / args.validation_dir
    tag = args.sell_mode
    out_dir = args.out_dir or (
        val
        / f"triangle_struct_pool_{tag}_{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}"
    )
    if not out_dir.is_absolute():
        out_dir = _PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_mod = _load_plot_mod()
    plot_mod._init_qlib(None)
    oos, _, _ = load_validation_csvs(val)
    oos["code"] = oos["code"].astype(str).str.upper()
    name_map = (
        oos.drop_duplicates("code").set_index("code")["name"].to_dict()
        if "name" in oos.columns
        else {}
    )

    if args.codes.strip():
        codes = [c.strip().upper() for c in args.codes.split(",") if c.strip()]
    else:
        codes = [c.upper() for c in load_cluster_mapping_codes(CLUSTER_MAPPING_SELECTED_TXT)]
        codes = [c for c in codes if c in set(oos["code"])]
    print(f"[INFO] sell_mode={args.sell_mode} codes={len(codes)} {args.start_date}->{args.end_date}")

    all_trades: list[dict] = []
    summaries: list[dict] = []

    for i, code in enumerate(codes, 1):
        try:
            ohlcv = plot_mod.load_qlib_ohlcv(
                code,
                args.start_date,
                args.end_date,
                init_qlib=False,
                lookback_calendar_days=220,
                clip_to_window=False,
            )
            px = build_px(code, ohlcv, oos[oos["code"] == code])
            trades, summ = simulate_code(
                px, start=args.start_date, end=args.end_date, sell_mode=args.sell_mode
            )
            for t in trades:
                t["code"] = code
            all_trades.extend(trades)
            summaries.append({"code": code, "name": name_map.get(code, ""), **summ})
            print(
                f"[{i:02d}/{len(codes)}] {code} n={summ['n_trades']} "
                f"win={summ['win_rate']:.1%} cmp={summ['compound']:+.2%} "
                f"bh={summ['bh']:+.2%} edge={summ['edge']:+.2%} "
                f"track10/20={summ['n_track_ma10']}/{summ['n_track_ma20']}"
            )
        except Exception as e:
            print(f"[{i:02d}/{len(codes)}] {code} ERROR {e}")
            if not args.continue_on_error:
                raise
            summaries.append(
                {
                    "code": code,
                    "name": name_map.get(code, ""),
                    "n_bars": 0,
                    "compound": np.nan,
                    "bh": np.nan,
                    "edge": np.nan,
                    "n_trades": 0,
                    "win_rate": np.nan,
                    "mean_ret": np.nan,
                    "beat_bh": False,
                    "n_track_ma10": 0,
                    "n_track_ma20": 0,
                    "sell_mode": args.sell_mode,
                    "error": str(e),
                }
            )

    tdf = pd.DataFrame(all_trades)
    sdf = pd.DataFrame(summaries)
    tdf.to_csv(out_dir / "all_trades.csv", index=False, encoding="utf-8-sig")
    sdf.to_csv(out_dir / "per_code_summary.csv", index=False, encoding="utf-8-sig")

    sells = tdf[tdf["side"] == "SELL"].copy() if not tdf.empty else pd.DataFrame()
    ok = sdf[sdf["n_trades"] > 0].copy()
    if len(ok):
        ew_compound = float(ok["compound"].mean())
        median_compound = float(ok["compound"].median())
        beat_rate = float(ok["beat_bh"].mean())
        trade_win = float((sells["fwd_ret"] > 0).mean()) if len(sells) else float("nan")
        trade_mean = float(sells["fwd_ret"].mean()) if len(sells) else float("nan")
        trade_med = float(sells["fwd_ret"].median()) if len(sells) else float("nan")
        n10 = int(ok["n_track_ma10"].sum())
        n20 = int(ok["n_track_ma20"].sum())
    else:
        ew_compound = median_compound = beat_rate = trade_win = trade_mean = trade_med = float("nan")
        n10 = n20 = 0

    focus = ["SH588710", "SH513310", "SH515050", "SH588850"]
    md = [
        f"# 结构买卖点全品种回测（sell_mode={args.sell_mode}）",
        "",
        "## 规则",
        "",
        "- **买**：金叉或绿三角+牛均线；`vol5_pct_120d≥0.5`",
        f"- **卖模式**：`{args.sell_mode}`",
        "",
    ]
    if args.sell_mode == "adaptive":
        md.extend(
            [
                "### 分轨卖点（开仓日锁定）",
                "",
                f"- **强趋势轨 → 红破 MA20**：prior60≥{PRIOR60_STRONG:.0%} 且 C>MA60 且 MA20>MA60",
                f"- **高波趋势轨 → 红破 MA20**：vol分位≥{VOL_PCT_HIVOL:.0%} 且 C>MA60 且 prior60≥{PRIOR60_HIVOL:.0%}",
                "- **否则波段轨 → 红破 MA10**",
                "",
            ]
        )
    md.extend(
        [
            f"- 区间：{args.start_date} → {args.end_date}",
            f"- 标的数：{len(codes)}（有成交 {len(ok)}）",
            f"- 总卖出笔数：{len(sells)}；开仓轨 MA10/MA20 = {n10}/{n20}",
            "",
            "## 汇总",
            "",
            f"- 单笔胜率：{trade_win:.1%}",
            f"- 单笔均值 / 中位：{trade_mean:+.2%} / {trade_med:+.2%}",
            f"- 逐标的复利等权 / 中位：{ew_compound:+.2%} / {median_compound:+.2%}",
            f"- 跑赢持有占比：{beat_rate:.1%}",
            "",
            "## 焦点四只",
            "",
            "| 代码 | 名称 | 策略 | 持有 | 超额 | 笔数 | 胜率 | 轨MA10 | 轨MA20 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for code in focus:
        sub = sdf[sdf["code"] == code]
        if sub.empty:
            continue
        r = sub.iloc[0]
        md.append(
            f"| {code} | {r.get('name','')} | {r['compound']:+.1%} | {r['bh']:+.1%} | "
            f"{r['edge']:+.1%}pct | {int(r['n_trades'])} | {r['win_rate']:.0%} | "
            f"{int(r['n_track_ma10'])} | {int(r['n_track_ma20'])} |"
        )
    md.extend(
        [
            "",
            "## 逐标的（按 compound 降序）",
            "",
            ok.sort_values("compound", ascending=False).to_markdown(index=False)
            if len(ok)
            else "(无)",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")

    print("\n=== POOL SUMMARY ===")
    print(f"sell_mode={args.sell_mode} traded={len(ok)}/{len(codes)} sells={len(sells)} track10/20={n10}/{n20}")
    print(f"trade_win={trade_win:.1%} mean={trade_mean:+.2%} med={trade_med:+.2%}")
    print(f"ew_compound={ew_compound:+.2%} median={median_compound:+.2%} beat_bh={beat_rate:.1%}")
    print("\nFocus 4:")
    cols = ["code", "name", "n_trades", "win_rate", "compound", "bh", "edge", "n_track_ma10", "n_track_ma20"]
    print(sdf[sdf.code.isin(focus)][cols].to_string(index=False))
    print(f"[INFO] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
