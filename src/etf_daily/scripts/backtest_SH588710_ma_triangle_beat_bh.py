#!/usr/bin/env python3
"""SH588710：均线过滤红/绿三角 + 金叉辅助，目标跑赢买入持有。

样本：2025-06-04 → 2026-08-05（样本内过拟合定稿）。

定稿规则：
  买（满足任一）：
     A. 绿三角且牛均线(C>MA5>MA10)
     B. 红三角且收盘仍>MA20（趋势内回撤）
     C. 红三角且跌≤-7%且收盘=最低（趋势外地板）
     D. MA5 上穿 MA20 金叉（辅助，无需三角）
  卖（定稿）：
     收盘相对持仓期最高价回撤 ≥25%（移动止盈）
     （备选：红三角且破 MA10，或收盘破 MA20）
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.lib.regime_transition_plot import (
    compute_volatility_regime_frame,
    load_validation_csvs,
)

CLOSE_EQ_TOL = 1e-4


def _load_plot_mod():
    path = _PROJECT_ROOT / "decision_pack" / "scripts" / "plot_regime_transition_example.py"
    spec = importlib.util.spec_from_file_location("plot_ma_bh", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_px(code: str, start: str, end: str, validation_dir: Path) -> pd.DataFrame:
    plot_mod = _load_plot_mod()
    plot_mod._init_qlib(None)
    oos, _, _ = load_validation_csvs(validation_dir)
    ohlcv = plot_mod.load_qlib_ohlcv(
        code, start, end, init_qlib=False, lookback_calendar_days=220, clip_to_window=False
    )
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
    px = df.merge(vf[["as_of", "rv5", "rv20", "vol5_pct_120d"]], on="as_of", how="left")
    px["ret1"] = px["$close"].pct_change(fill_method=None)
    for w in (5, 10, 20, 60):
        px[f"MA{w}"] = px["$close"].rolling(w, min_periods=max(3, w // 2)).mean()
    px["above_ma5"] = px["$close"] > px["MA5"]
    px["above_ma10"] = px["$close"] > px["MA10"]
    px["above_ma20"] = px["$close"] > px["MA20"]
    px["ma_bull"] = (px["$close"] > px["MA5"]) & (px["MA5"] > px["MA10"])
    px["close_eq_low"] = (
        (px["$close"] - px["$low"]).abs() / px["$close"].clip(lower=1e-8) <= CLOSE_EQ_TOL
    )
    # 金叉：MA5 上穿 MA20
    px["golden_ma5_ma20"] = (px["MA5"] > px["MA20"]) & (px["MA5"].shift(1) <= px["MA20"].shift(1))

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
    px["is_red"] = px["as_of"].isin(red)
    px["is_green"] = px["as_of"].isin(green)
    px = px[(px["as_of"] >= pd.Timestamp(start)) & (px["as_of"] <= pd.Timestamp(end))].reset_index(
        drop=True
    )
    return px


def is_buy(row: pd.Series, *, use_golden: bool = True) -> tuple[bool, str]:
    """买点：均线过滤红/绿三角 + 可选金叉辅助。"""
    r, g = bool(row["is_red"]), bool(row["is_green"])
    ret = float(row["ret1"]) if pd.notna(row["ret1"]) else 0.0
    if g and bool(row["ma_bull"]):
        return True, "绿三角_牛均线"
    if r and bool(row["above_ma20"]):
        return True, "红三角_仍上MA20"
    if r and ret <= -0.07 and bool(row["close_eq_low"]):
        return True, "红三角_深跌地板"
    if use_golden and bool(row.get("golden_ma5_ma20", False)):
        return True, "金叉_MA5上穿MA20"
    return False, ""


def should_exit(
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
    if exit_mode == "red_ma10_or_ma20":
        if bool(bar["is_red"]) and (not bool(bar["above_ma10"])):
            return True, "红三角且破MA10"
        if not bool(bar["above_ma20"]):
            return True, "破MA20"
        return False, ""
    if exit_mode == "trail_or_red_ma10_or_ma20":
        if c <= peak * (1.0 - trail_pct):
            return True, f"峰值回撤{int(trail_pct * 100)}%"
        if bool(bar["is_red"]) and (not bool(bar["above_ma10"])):
            return True, "红三角且破MA10"
        if not bool(bar["above_ma20"]):
            return True, "破MA20"
        return False, ""
    raise ValueError(f"unknown exit_mode={exit_mode}")


def run_strategy(
    px: pd.DataFrame,
    *,
    trail_pct: float = 0.25,
    use_golden: bool = True,
    exit_mode: str = "trail",
) -> tuple[float, float, list[dict]]:
    cash = 1.0
    pos = 0.0
    entry = None
    entry_i = None
    peak = None
    entry_why = ""
    trades: list[dict] = []

    for i in range(len(px)):
        bar = px.iloc[i]
        c = float(bar["$close"])
        h = float(bar["$high"])
        d = str(pd.Timestamp(bar["as_of"]).date())

        if pos == 0:
            ok, why = is_buy(bar, use_golden=use_golden)
            if ok:
                entry = c
                entry_i = i
                peak = h
                entry_why = why
                pos = cash / c
                cash = 0.0
                trades.append(
                    {
                        "side": "BUY",
                        "date": d,
                        "px": c,
                        "why": why,
                        "is_red": bool(bar["is_red"]),
                        "is_green": bool(bar["is_green"]),
                        "golden": bool(bar.get("golden_ma5_ma20", False)),
                        "above_ma20": bool(bar["above_ma20"]),
                        "ma_bull": bool(bar["ma_bull"]),
                    }
                )
            continue

        peak = max(peak, h)
        hold_days = i - entry_i
        exited, why = should_exit(
            bar,
            exit_mode=exit_mode,
            trail_pct=trail_pct,
            peak=peak,
            hold_days=hold_days,
        )
        if exited:
            cash = pos * c
            trades.append(
                {
                    "side": "SELL",
                    "date": d,
                    "px": c,
                    "why": why,
                    "entry_date": str(pd.Timestamp(px.iloc[entry_i]["as_of"]).date()),
                    "entry_px": entry,
                    "entry_why": entry_why,
                    "hold_days": hold_days,
                    "fwd_ret": c / entry - 1.0,
                    "peak": peak,
                }
            )
            pos = 0.0
            entry = None
            entry_i = None
            peak = None
            entry_why = ""

    if pos > 0:
        c = float(px.iloc[-1]["$close"])
        d = str(pd.Timestamp(px.iloc[-1]["as_of"]).date())
        cash = pos * c
        trades.append(
            {
                "side": "SELL",
                "date": d,
                "px": c,
                "why": "期末强平",
                "entry_date": str(pd.Timestamp(px.iloc[entry_i]["as_of"]).date()),
                "entry_px": entry,
                "entry_why": entry_why,
                "hold_days": len(px) - 1 - entry_i,
                "fwd_ret": c / entry - 1.0,
                "peak": peak,
            }
        )

    compound = cash - 1.0
    bh = float(px.iloc[-1]["$close"] / px.iloc[0]["$close"] - 1.0)
    return compound, bh, trades


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code", default="SH588710")
    p.add_argument("--start-date", default="2025-06-04")
    p.add_argument("--end-date", default="2026-08-05")
    p.add_argument("--trail-pct", type=float, default=0.25)
    p.add_argument(
        "--exit-mode",
        default="trail",
        choices=("trail", "red_ma10_or_ma20", "trail_or_red_ma10_or_ma20"),
        help="定稿默认 trail（金叉+峰值回撤25% 样本内最优）",
    )
    p.add_argument("--no-golden", action="store_true", help="关闭金叉辅助买点")
    p.add_argument(
        "--validation-dir",
        type=Path,
        default=_PROJECT_ROOT
        / "runtime/decision_packs/20260720/regime_transition_validation_q90_to0720",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--compare", action="store_true", help="打印有/无金叉与多退出模式对比")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    val = args.validation_dir if args.validation_dir.is_absolute() else _PROJECT_ROOT / args.validation_dir
    out_dir = args.out_dir or (
        val
        / f"triangle_ma_beat_bh_{args.code}_{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}"
    )
    if not out_dir.is_absolute():
        out_dir = _PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    px = build_px(args.code, args.start_date, args.end_date, val)
    use_golden = not args.no_golden

    if args.compare:
        print("[COMPARE] golden × exit_mode")
        for g in (False, True):
            for em in ("trail", "red_ma10_or_ma20", "trail_or_red_ma10_or_ma20"):
                c, bh, tr = run_strategy(
                    px, trail_pct=args.trail_pct, use_golden=g, exit_mode=em
                )
                n = sum(1 for t in tr if t["side"] == "SELL")
                print(
                    f"  golden={str(g):5} exit={em:28s} compound={c:+.2%} edge={c-bh:+.2%} n={n}"
                )

    compound, bh, trades = run_strategy(
        px,
        trail_pct=args.trail_pct,
        use_golden=use_golden,
        exit_mode=args.exit_mode,
    )
    tdf = pd.DataFrame(trades)
    tdf.to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")
    px.to_csv(out_dir / "bars.csv", index=False, encoding="utf-8-sig")

    sells = tdf[tdf["side"] == "SELL"] if not tdf.empty else tdf
    win = float((sells["fwd_ret"] > 0).mean()) if len(sells) else float("nan")
    n_golden_buys = (
        int((tdf["side"].eq("BUY") & tdf["why"].eq("金叉_MA5上穿MA20")).sum()) if not tdf.empty else 0
    )

    exit_desc = {
        "trail": f"峰值回撤 {args.trail_pct:.0%}",
        "red_ma10_or_ma20": "红三角且破MA10，或收盘破MA20",
        "trail_or_red_ma10_or_ma20": f"峰值回撤{args.trail_pct:.0%} 或 红破MA10/破MA20",
    }[args.exit_mode]

    md = [
        f"# {args.code} 均线过滤红绿三角 + 金叉辅助（跑赢持有定稿）",
        "",
        f"- 区间：{args.start_date} → {args.end_date}",
        f"- 买入持有：{bh:+.2%}",
        f"- 策略复利：{compound:+.2%}",
        f"- 相对持有：{compound - bh:+.2%}",
        f"- 交易笔数：{len(sells)}；胜率：{win:.1%}",
        f"- 金叉辅助：{'开' if use_golden else '关'}（金叉触发买入 {n_golden_buys} 次）",
        f"- 卖点模式：{exit_desc}",
        "",
        "## 买点",
        "",
        "1. 绿三角 **且** 牛均线（C>MA5>MA10）",
        "2. 红三角 **且** 收盘仍 **> MA20**（趋势内回撤买）",
        "3. 红三角 **且** 跌≤−7% **且** 收盘=最低（趋势外地板）",
        "4. **金叉辅助**：MA5 上穿 MA20（无需当日三角）",
        "",
        "## 卖点",
        "",
        f"- {exit_desc}",
        "- 不用固定 +3%/+8% 止盈（会砍主升浪）",
        "",
        "## 交易明细",
        "",
        tdf.to_markdown(index=False) if not tdf.empty else "(无)",
        "",
        "> 样本内过拟合结果；换区间需重估。",
    ]
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")

    print(f"[INFO] BH={bh:+.2%} strategy={compound:+.2%} edge={compound - bh:+.2%}")
    print(f"[INFO] exit_mode={args.exit_mode} golden={use_golden}")
    print(tdf.to_string(index=False))
    print(f"[INFO] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
