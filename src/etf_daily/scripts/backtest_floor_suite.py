#!/usr/bin/env python3
"""Full-pool backtest: §1b 深跌地板试多 · 三件套

① 买点：T 日跌幅≤-7%（深跌红代理）→ T 收盘买入；地板(C=L)同理
② 止盈：T+1起 开盘相对成本高开≥gap → 开盘止盈；
         或盘中触及 +float_tp → 按目标价止盈；
         或持有满 max_hold 日且浮盈 → 收盘时间止盈
③ 止损：收盘 < 买点日最低×(1-sl_buf) → 收盘止损；
         或收盘浮亏 ≤ sl_pct → 收盘止损
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.lib.symbol_trend_board import load_cluster_mapping_codes
from etf_daily.paths import CLUSTER_MAPPING_SELECTED_TXT

DROP_TH = -0.07
CLOSE_EQ_LOW_TOL = 1e-4


def _load_plot_mod():
    path = _PROJECT_ROOT / "decision_pack" / "scripts" / "plot_regime_transition_example.py"
    spec = importlib.util.spec_from_file_location("plot_regime_suite_bt", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ohlcv_to_px(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.copy()
    df["as_of"] = pd.to_datetime(df["datetime"]).dt.normalize()
    df = df.sort_values("as_of")
    df = df[~df["as_of"].duplicated(keep="last")].reset_index(drop=True)
    out = pd.DataFrame(
        {
            "as_of": df["as_of"],
            "open": pd.to_numeric(df["$open"], errors="coerce"),
            "high": pd.to_numeric(df["$high"], errors="coerce"),
            "low": pd.to_numeric(df["$low"], errors="coerce"),
            "close": pd.to_numeric(df["$close"], errors="coerce"),
        }
    )
    out["ret1"] = out["close"].pct_change(fill_method=None)
    out["close_eq_low"] = (out["close"] - out["low"]).abs() / out["close"].clip(lower=1e-8) <= CLOSE_EQ_LOW_TOL
    return out.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def simulate_trade(
    px: pd.DataFrame,
    signal_i: int,
    *,
    tp_gap: float,
    float_tp: float | None,
    max_hold: int,
    sl_buf: float,
    sl_pct: float,
    time_tp_only_if_profit: bool = True,
) -> dict[str, Any] | None:
    """Enter at T close; manage exits from T+1 .. T+max_hold."""
    if signal_i + 1 >= len(px):
        return None
    t = px.iloc[signal_i]
    entry = float(t["close"])
    buy_low = float(t["low"])
    buy_high = float(t["high"])
    stop_line = buy_low * (1.0 - sl_buf)
    floor = bool(t["close_eq_low"])
    signal_ret = float(t["ret1"]) if pd.notna(t["ret1"]) else np.nan

    exit_px = np.nan
    exit_i = None
    exit_why = None
    exit_side = None  # open/close/target

    last_i = min(signal_i + max_hold, len(px) - 1)
    for j in range(signal_i + 1, last_i + 1):
        bar = px.iloc[j]
        o, h, l, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
        held = j - signal_i  # trading days held after entry

        # ② 开盘止盈（相对成本）
        gap_from_entry = o / entry - 1.0
        if gap_from_entry >= tp_gap:
            exit_px, exit_i, exit_why, exit_side = o, j, "止盈_高开", "open"
            break

        # ② 盘中浮盈触及
        if float_tp is not None and h / entry - 1.0 >= float_tp:
            exit_px, exit_i, exit_why, exit_side = entry * (1.0 + float_tp), j, "止盈_浮盈", "target"
            break

        # ③ 收盘止损：破买点日低（放宽）或浮亏阈值
        if c < stop_line:
            exit_px, exit_i, exit_why, exit_side = c, j, "止损_破买点低", "close"
            break
        if c / entry - 1.0 <= sl_pct:
            exit_px, exit_i, exit_why, exit_side = c, j, "止损_浮亏", "close"
            break

        # ② 时间止盈：满 max_hold
        if j == last_i:
            ret = c / entry - 1.0
            if (not time_tp_only_if_profit) or ret > 0:
                exit_px, exit_i, exit_why, exit_side = c, j, "止盈_时间", "close"
            else:
                exit_px, exit_i, exit_why, exit_side = c, j, "到期_亏损", "close"
            break

    if exit_i is None or not np.isfinite(exit_px):
        return None

    fwd = float(exit_px) / entry - 1.0
    return {
        "signal_date": str(pd.Timestamp(t["as_of"]).date()),
        "signal_ret": signal_ret,
        "buy_type": "地板" if floor else "深跌红",
        "close_eq_low": floor,
        "buy_low": buy_low,
        "buy_high": buy_high,
        "entry_date": str(pd.Timestamp(t["as_of"]).date()),
        "entry_px": entry,
        "exit_date": str(pd.Timestamp(px.iloc[exit_i]["as_of"]).date()),
        "exit_px": float(exit_px),
        "exit_why": exit_why,
        "exit_side": exit_side,
        "hold_days": int(exit_i - signal_i),
        "fwd_ret": fwd,
        "stop_line": stop_line,
    }


def evaluate_code(
    code: str,
    px: pd.DataFrame,
    *,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    cfg: dict[str, Any],
    buy_mode: str,
) -> list[dict[str, Any]]:
    """buy_mode: deep=all ret<=-7%; floor=only C=L; both=deep (tag floor)."""
    rows: list[dict[str, Any]] = []
    busy_until = -1
    for i in range(len(px) - 1):
        if i <= busy_until:
            continue
        t = px.iloc[i]
        t_d = pd.Timestamp(t["as_of"])
        if t_d < eval_start or t_d > eval_end:
            continue
        ret = t["ret1"]
        if pd.isna(ret) or float(ret) > DROP_TH:
            continue
        floor = bool(t["close_eq_low"])
        if buy_mode == "floor" and not floor:
            continue
        # buy_mode deep/both: all deep drops

        sim = simulate_trade(px, i, **cfg)
        if sim is None:
            continue
        sim["code"] = code
        sim["buy_mode"] = buy_mode
        rows.append(sim)
        # block overlap until exit
        exit_d = pd.Timestamp(sim["exit_date"])
        # find exit index
        busy_until = i
        for k in range(i + 1, len(px)):
            if pd.Timestamp(px.iloc[k]["as_of"]) >= exit_d:
                busy_until = k
                break
    return rows


def summarize(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in events.groupby(["rule", "buy_mode"], sort=True):
        rule, buy_mode = key
        n = len(g)
        rows.append(
            {
                "rule": rule,
                "buy_mode": buy_mode,
                "n": n,
                "win_rate": float((g["fwd_ret"] > 0).mean()),
                "mean_ret": float(g["fwd_ret"].mean()),
                "median_ret": float(g["fwd_ret"].median()),
                "p25_ret": float(g["fwd_ret"].quantile(0.25)),
                "p75_ret": float(g["fwd_ret"].quantile(0.75)),
                "mean_hold": float(g["hold_days"].mean()),
                "tp_gap_rate": float(g["exit_why"].eq("止盈_高开").mean()),
                "tp_float_rate": float(g["exit_why"].eq("止盈_浮盈").mean()),
                "tp_time_rate": float(g["exit_why"].eq("止盈_时间").mean()),
                "sl_rate": float(g["exit_why"].str.startswith("止损").mean()),
                "loss_expiry_rate": float(g["exit_why"].eq("到期_亏损").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["buy_mode", "rule"])


RULE_CFGS: dict[str, dict[str, Any]] = {
    # 主规则候选（文档默认）
    "三件套_主": dict(tp_gap=0.03, float_tp=0.08, max_hold=10, sl_buf=0.01, sl_pct=-0.06),
    "三件套_高开2%": dict(tp_gap=0.02, float_tp=0.08, max_hold=10, sl_buf=0.01, sl_pct=-0.06),
    "三件套_高开5%": dict(tp_gap=0.05, float_tp=0.08, max_hold=10, sl_buf=0.01, sl_pct=-0.06),
    "三件套_无浮盈止盈": dict(tp_gap=0.03, float_tp=None, max_hold=10, sl_buf=0.01, sl_pct=-0.06),
    "三件套_hold5": dict(tp_gap=0.03, float_tp=0.08, max_hold=5, sl_buf=0.01, sl_pct=-0.06),
    "三件套_止损-8%": dict(tp_gap=0.03, float_tp=0.08, max_hold=10, sl_buf=0.01, sl_pct=-0.08),
    "三件套_止损仅破位": dict(tp_gap=0.03, float_tp=0.08, max_hold=10, sl_buf=0.01, sl_pct=-0.99),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start-date", default="2024-01-01")
    p.add_argument("--end-date", default="2026-08-05")
    p.add_argument("--code", action="append", default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_PROJECT_ROOT
        / "runtime/decision_packs/20260720/regime_transition_validation_q90_to0720"
        / "triangle_decision_backtest_floor_suite",
    )
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.out_dir if args.out_dir.is_absolute() else _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    codes = (
        [c.upper() for c in args.code]
        if args.code
        else [c.upper() for c in load_cluster_mapping_codes(CLUSTER_MAPPING_SELECTED_TXT)]
    )
    plot_mod = _load_plot_mod()
    plot_mod._init_qlib(None)
    eval_start, eval_end = pd.Timestamp(args.start_date), pd.Timestamp(args.end_date)

    print(f"[INFO] suite backtest codes={len(codes)} {args.start_date}->{args.end_date}")

    # load all px once
    px_map: dict[str, pd.DataFrame] = {}
    failed: list[tuple[str, str]] = []
    for code in codes:
        try:
            ohlcv = plot_mod.load_qlib_ohlcv(
                code,
                args.start_date,
                args.end_date,
                init_qlib=False,
                lookback_calendar_days=220,
                clip_to_window=False,
            )
            px_map[code] = _ohlcv_to_px(ohlcv)
            print(f"[OK] load {code} bars={len(px_map[code])}")
        except Exception as exc:  # noqa: BLE001
            failed.append((code, str(exc)))
            print(f"[ERROR] {code}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                return 2

    all_rows: list[dict[str, Any]] = []
    for buy_mode in ("deep", "floor"):
        for rule_name, cfg in RULE_CFGS.items():
            for code, px in px_map.items():
                ev = evaluate_code(
                    code,
                    px,
                    eval_start=eval_start,
                    eval_end=eval_end,
                    cfg=cfg,
                    buy_mode=buy_mode,
                )
                for r in ev:
                    r["rule"] = rule_name
                all_rows.extend(ev)

    events = pd.DataFrame(all_rows)
    events_path = out_dir / "floor_suite_events.csv"
    events.to_csv(events_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] wrote {events_path} rows={len(events)}")

    if events.empty:
        print("[WARN] no events")
        return 0

    summ = summarize(events)
    summ_path = out_dir / "floor_suite_summary.csv"
    summ.to_csv(summ_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] wrote {summ_path}")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print("\n=== 全池三件套汇总 ===")
    print(summ.to_string(index=False))

    # Focus main rule
    main = summ[(summ["rule"] == "三件套_主")]
    print("\n=== 主规则 三件套_主 ===")
    print(main.to_string(index=False))

    # SH588710 focus
    sh = events[(events["code"] == "SH588710") & (events["rule"] == "三件套_主") & (events["buy_mode"] == "deep")]
    if not sh.empty:
        print("\n=== SH588710 三件套_主 deep ===")
        cols = [
            "signal_date",
            "buy_type",
            "entry_px",
            "exit_date",
            "exit_px",
            "exit_why",
            "hold_days",
            "fwd_ret",
        ]
        print(sh[cols].to_string(index=False))

    # exit why breakdown for main+deep
    sub = events[(events["rule"] == "三件套_主") & (events["buy_mode"] == "deep")]
    if not sub.empty:
        print("\n=== 主规则 deep 退出原因 ===")
        print(sub["exit_why"].value_counts().to_string())
        print(f"win={ (sub['fwd_ret']>0).mean():.1%} mean={sub['fwd_ret'].mean():+.2%} n={len(sub)}")

    md_lines = [
        "# 深跌地板试多 · 三件套全池回测",
        "",
        f"- 池：{len(codes)} 只 cluster_mapping_selected",
        f"- 区间：{args.start_date} → {args.end_date}",
        "- 买点：T 日跌幅≤-7%（deep）；floor=另需 C=L",
        "- 入场：T 收盘",
        "- 止盈：开盘相对成本≥tp_gap；可选浮盈触及；满 max_hold 日有盈则时间止盈",
        "- 止损：收盘破买点日低×(1-sl_buf)；或收盘浮亏≤sl_pct",
        "",
        "## 汇总",
        "",
        summ.to_markdown(index=False),
        "",
    ]
    (out_dir / "floor_suite_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[INFO] wrote {out_dir / 'floor_suite_summary.md'}")
    if failed:
        print(f"[WARN] failed={failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
