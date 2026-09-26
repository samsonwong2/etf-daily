#!/usr/bin/env python3
"""Full-pool backtest: §1b 深跌地板试多 (experimental).

Signal day T:
  - day_ret <= -7%
  - close == low (capitulation / floor close)
Optional contrast: any day_ret <= -7% without close==low.

Entry T+1:
  - 开盘: buy next open if gap < +2%; else skip
  - 限价带: if open <= T_close*1.01 → fill open;
            elif low <= T_close → fill T_close;
            else miss. Also skip if gap >= +2% (no chase).

Exit: stop at T low (intraday low <= stop → fill stop), else hold H days close.
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

from decision_pack.src.symbol_trend_board import load_cluster_mapping_codes
from runtime_paths import CLUSTER_MAPPING_SELECTED_TXT

DROP_TH = -0.07
GAP_SKIP = 0.02
LIMIT_BAND = 0.01
CLOSE_EQ_LOW_TOL = 1e-4  # relative |c-l|/c
HOLD_DAYS = (2, 5, 10, 20)


def _load_plot_mod():
    path = _PROJECT_ROOT / "decision_pack" / "scripts" / "plot_regime_transition_example.py"
    spec = importlib.util.spec_from_file_location("plot_regime_floor_bt", path)
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
    out["ret1"] = out["close"].pct_change()
    out["ma20"] = out["close"].rolling(20, min_periods=20).mean()
    out["below_ma20"] = out["close"] / out["ma20"] - 1.0
    out["close_eq_low"] = (out["close"] - out["low"]).abs() / out["close"].clip(lower=1e-8) <= CLOSE_EQ_LOW_TOL
    return out.dropna(subset=["open", "high", "low", "close"])


def _simulate_hold(
    px: pd.DataFrame,
    exec_i: int,
    exec_px: float,
    stop: float,
    hold: int,
) -> dict[str, Any]:
    """exec_i = index of entry bar in px (T+1)."""
    fut = px.iloc[exec_i : exec_i + hold]
    if fut.empty:
        return {
            "executed": False,
            "exit_why": "无未来",
            "fwd_ret": np.nan,
            "complete": False,
            "exit_date": None,
            "exit_px": np.nan,
        }
    exit_px = float(fut.iloc[-1]["close"])
    exit_d = pd.Timestamp(fut.iloc[-1]["as_of"])
    exit_why = "到期"
    for _, r in fut.iterrows():
        if float(r["low"]) <= float(stop):
            exit_px = float(stop)
            exit_d = pd.Timestamp(r["as_of"])
            exit_why = "止损"
            break
    return {
        "executed": True,
        "exit_why": exit_why,
        "fwd_ret": exit_px / float(exec_px) - 1.0,
        "complete": bool(exit_why == "止损" or len(fut) >= hold),
        "exit_date": str(exit_d.date()),
        "exit_px": exit_px,
    }


def evaluate_code(
    code: str,
    px: pd.DataFrame,
    *,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = len(px)
    for i in range(n - 1):
        t = px.iloc[i]
        t1 = px.iloc[i + 1]
        t_d = pd.Timestamp(t["as_of"])
        if t_d < eval_start or t_d > eval_end:
            continue
        ret = t["ret1"]
        if pd.isna(ret) or float(ret) > DROP_TH:
            continue

        t_close = float(t["close"])
        t_low = float(t["low"])
        next_o = float(t1["open"])
        next_l = float(t1["low"])
        gap = next_o / t_close - 1.0
        floor = bool(t["close_eq_low"])
        below = float(t["below_ma20"]) if pd.notna(t["below_ma20"]) else np.nan

        variants: list[tuple[str, bool, float | None, str]] = []

        # Baseline: any extreme → next open (no gap gate)
        variants.append(("裸大跌_次日开盘", True, next_o, "open"))

        # §1b variants only when T close == low (floor)
        if floor:
            if gap < GAP_SKIP:
                variants.append(("地板试多_开盘", True, next_o, "open"))
            else:
                variants.append(("地板试多_开盘", False, None, "高开跳过"))

            if gap >= GAP_SKIP:
                variants.append(("地板试多_限价带", False, None, "高开跳过"))
            elif next_o <= t_close * (1.0 + LIMIT_BAND):
                variants.append(("地板试多_限价带", True, next_o, "open"))
            elif next_l <= t_close:
                variants.append(("地板试多_限价带", True, t_close, "limit@Tclose"))
            else:
                variants.append(("地板试多_限价带", False, None, "未触及限价"))

            if pd.notna(below) and below <= -0.15:
                if gap < GAP_SKIP:
                    variants.append(("地板试多_深贴均线_开盘", True, next_o, "open"))
                else:
                    variants.append(("地板试多_深贴均线_开盘", False, None, "高开跳过"))

        for pattern, ok, exec_px, how in variants:
            base = {
                "code": code,
                "pattern": pattern,
                "signal_date": str(t_d.date()),
                "signal_ret": float(ret),
                "signal_close": t_close,
                "signal_low": t_low,
                "close_eq_low": floor,
                "below_ma20": below,
                "next_open": next_o,
                "next_low": next_l,
                "gap": gap,
                "fill_how": how,
                "stop": t_low,
            }
            if not ok or exec_px is None:
                for hold in HOLD_DAYS:
                    rows.append(
                        {
                            **base,
                            "hold": hold,
                            "executed": False,
                            "exec_date": str(pd.Timestamp(t1["as_of"]).date()),
                            "exec_px": np.nan,
                            "exit_why": how,
                            "fwd_ret": np.nan,
                            "complete": False,
                            "exit_date": None,
                            "exit_px": np.nan,
                        }
                    )
                continue
            for hold in HOLD_DAYS:
                sim = _simulate_hold(px, i + 1, float(exec_px), t_low, hold)
                rows.append(
                    {
                        **base,
                        "hold": hold,
                        "executed": sim["executed"],
                        "exec_date": str(pd.Timestamp(t1["as_of"]).date()),
                        "exec_px": float(exec_px),
                        "exit_why": sim["exit_why"],
                        "fwd_ret": sim["fwd_ret"],
                        "complete": sim["complete"],
                        "exit_date": sim["exit_date"],
                        "exit_px": sim["exit_px"],
                    }
                )
    return rows


def summarize(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pattern, hold), g in events.groupby(["pattern", "hold"], sort=True):
        n_sig = int(g["signal_date"].nunique()) if "signal_date" in g else len(g)
        # each signal×hold is one row; count signals as unique code+signal_date for pattern
        n_events = int(len(g[["code", "signal_date"]].drop_duplicates()))
        ex = g[g["executed"] == True]  # noqa: E712
        n_exec = int(len(ex))
        use = ex[ex["complete"] == True] if (ex["complete"] == True).sum() >= 3 else ex  # noqa: E712
        rows.append(
            {
                "pattern": pattern,
                "hold": int(hold),
                "n_signal_days": n_events,
                "n_executed": n_exec,
                "exec_rate": n_exec / n_events if n_events else np.nan,
                "n_scored": int(len(use)),
                "win_rate": float((use["fwd_ret"] > 0).mean()) if len(use) else np.nan,
                "mean_ret": float(use["fwd_ret"].mean()) if len(use) else np.nan,
                "median_ret": float(use["fwd_ret"].median()) if len(use) else np.nan,
                "p25_ret": float(use["fwd_ret"].quantile(0.25)) if len(use) else np.nan,
                "p75_ret": float(use["fwd_ret"].quantile(0.75)) if len(use) else np.nan,
                "stop_rate": float((use["exit_why"] == "止损").mean()) if len(use) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["pattern", "hold"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start-date", default="2024-01-01")
    p.add_argument("--end-date", default="2026-08-05")
    p.add_argument("--code", action="append", default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_PROJECT_ROOT
        / "workspace/decision_packs/20260720/regime_transition_validation_q90_to0720"
        / "triangle_decision_backtest_floor_probe",
    )
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = _PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.code:
        codes = [c.upper() for c in args.code]
    else:
        codes = [c.upper() for c in load_cluster_mapping_codes(CLUSTER_MAPPING_SELECTED_TXT)]

    plot_mod = _load_plot_mod()
    plot_mod._init_qlib(None)

    eval_start = pd.Timestamp(args.start_date)
    eval_end = pd.Timestamp(args.end_date)
    print(
        f"[INFO] floor-probe backtest codes={len(codes)} "
        f"eval={args.start_date}->{args.end_date} "
        f"drop<={DROP_TH:.0%} gap_skip>={GAP_SKIP:.0%} holds={HOLD_DAYS}"
    )

    all_rows: list[dict[str, Any]] = []
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
            px = _ohlcv_to_px(ohlcv)
            ev = evaluate_code(code, px, eval_start=eval_start, eval_end=eval_end)
            all_rows.extend(ev)
            n_floor = sum(
                1
                for r in ev
                if r["pattern"] == "地板试多_开盘" and r["hold"] == HOLD_DAYS[0] and r["close_eq_low"]
            )
            n_exec = sum(
                1
                for r in ev
                if r["pattern"] == "地板试多_开盘" and r["hold"] == HOLD_DAYS[0] and r["executed"]
            )
            print(f"[OK] {code} floor_signals≈{n_floor} exec={n_exec}")
        except Exception as exc:  # noqa: BLE001
            failed.append((code, str(exc)))
            print(f"[ERROR] {code}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                return 2

    events = pd.DataFrame(all_rows)
    events_path = out_dir / "floor_probe_events.csv"
    events.to_csv(events_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] wrote {events_path} rows={len(events)}")

    if not events.empty:
        summ = summarize(events)
        summ_path = out_dir / "floor_probe_summary.csv"
        summ.to_csv(summ_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] wrote {summ_path}")
        pd.set_option("display.width", 220)
        pd.set_option("display.max_columns", 20)
        print("\n=== 全池汇总（完整样本优先） ===")
        print(summ.to_string(index=False))

        # Focus: §1b patterns hold=2
        focus = summ[summ["hold"] == 2]
        print("\n=== hold=2 速览（对齐 SH588710 两日窗口） ===")
        print(focus.to_string(index=False))

        sh = events[(events["code"] == "SH588710") & (events["executed"]) & (events["hold"] == 2)]
        if not sh.empty:
            print("\n=== SH588710 hold=2 已执行 ===")
            cols = [
                "pattern",
                "signal_date",
                "exec_date",
                "exec_px",
                "fill_how",
                "exit_why",
                "fwd_ret",
            ]
            print(sh[cols].to_string(index=False))

        md = [
            "# 深跌地板试多 · 全池回测摘要",
            "",
            f"- 池：cluster_mapping_selected（{len(codes)} 只）",
            f"- 区间：{args.start_date} → {args.end_date}",
            f"- 信号：日跌 ≤ {DROP_TH:.0%}；地板变体另需 close==low",
            f"- 入场：次日开盘 / 限价带；高开 ≥ {GAP_SKIP:.0%} 跳过",
            f"- 止损：信号日最低价；持有 {list(HOLD_DAYS)} 日",
            "",
            "## 汇总表",
            "",
            summ.to_markdown(index=False),
            "",
            "## 解读备忘",
            "",
            "- `裸大跌_次日开盘`：无地板过滤的对照基线",
            "- `地板试多_开盘` / `地板试多_限价带`：§1b 实验主规则",
            "- `地板试多_深贴均线_开盘`：另加 below MA20 ≤ −15%",
            "- 胜率 = 已执行且（完整样本优先）`fwd_ret > 0` 占比",
            "",
        ]
        md_path = out_dir / "floor_probe_summary.md"
        md_path.write_text("\n".join(md), encoding="utf-8")
        print(f"[INFO] wrote {md_path}")

    if failed:
        print(f"[WARN] failed ({len(failed)}): {failed[:5]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
