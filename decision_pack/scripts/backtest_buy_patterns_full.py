#!/usr/bin/env python3
"""Full-universe / full-OOS backtest: which buy patterns are identifiable?

Motivated by SH515050 after 2026-07-30: deep red (prior10≪-8%) blocked from
「买观察」, reclaim of red-day high not yet hit. Question: across the whole
selected pool and OOS window, do any of these patterns have positive edge?

Patterns evaluated (same reclaim / next-open simulation as production):
  A. 生产买观察     — annotate action==买观察, then reclaim high → next open
  B. 生产买确认     — annotate action==买确认候选, next open
  C. 深跌红收复     — ALL reds with prior10 < -8% (production floors these out),
                     reclaim high → next open  ← SH515050-like
  D. 中段红收复     — reds with prior10 in [-8%, 0], reclaim (baseline contrast)
  E. 深跌后绿确认   — greens with prior10 < -8%, next open (deep bounce green)

Hold = RuleConfig.hold_eval_days (20); stop = red-day low / recent 5d low.
Summary uses complete samples only (stopped or full hold window).
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


def _load_rules_mod():
    path = _PROJECT_ROOT / "decision_pack" / "scripts" / "backtest_triangle_decision_rules.py"
    spec = importlib.util.spec_from_file_location("triangle_decision_rules_bt2", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _prior_bucket(p: Any) -> str:
    if p is None or pd.isna(p):
        return "未知"
    v = float(p)
    if v < -0.08:
        return "深跌(<-8%)"
    if v > 0.0:
        return "翻红(>0%)"
    return "中段(-8%~0%)"


def _path_pnl(px: pd.DataFrame, rules: Any, exec_d, exec_px, stop, hold: int) -> dict[str, Any]:
    fut = px[px["as_of"] >= exec_d].head(hold)
    exit_px = float(fut.iloc[-1]["$close"]) if not fut.empty else float(exec_px)
    exit_d = pd.Timestamp(fut.iloc[-1]["as_of"]) if not fut.empty else pd.Timestamp(exec_d)
    exit_why = "到期"
    for _, r in fut.iterrows():
        if stop is not None and float(r["$low"]) <= float(stop):
            exit_px = float(stop)
            exit_d = pd.Timestamp(r["as_of"])
            exit_why = "止损"
            break
    st = rules.forward_stats(px, exec_d, float(exec_px), hold)
    return {
        "executed": True,
        "exec_date": str(pd.Timestamp(exec_d).date()),
        "exec_px": float(exec_px),
        "stop": None if stop is None else float(stop),
        "exit_date": str(exit_d.date()),
        "exit_why": exit_why,
        "fwd_ret": exit_px / float(exec_px) - 1.0,
        "fwd_mfe": st["mfe"],
        "fwd_mae": st["mae"],
        "complete": bool(exit_why == "止损" or len(fut) >= hold),
    }


def evaluate_code(
    code: str,
    oos: pd.DataFrame,
    rules: Any,
    plot_mod: Any,
    cfg: Any,
    *,
    end_date: str,
) -> list[dict[str, Any]]:
    code_u = str(code).upper()
    sub = oos[oos["code"].astype(str).str.upper() == code_u].copy()
    if sub.empty:
        return []
    sig_start = pd.Timestamp(sub["as_of"].min()).strftime("%Y-%m-%d")
    ohlcv = plot_mod.load_qlib_ohlcv(
        code_u,
        sig_start,
        end_date,
        init_qlib=False,
        lookback_calendar_days=220,
        clip_to_window=False,
    )
    ohlcv = plot_mod.attach_sma_columns(ohlcv)
    close = ohlcv.set_index(pd.to_datetime(ohlcv["datetime"]).dt.normalize())["$close"]
    close = close[~close.index.duplicated(keep="last")]
    vol = compute_volatility_regime_frame(close)
    px = rules.build_price_frame(ohlcv, vol)
    px = px[(px["as_of"] >= sig_start) & (px["as_of"] <= end_date)].copy()

    sw = sub[sub["quantile_switch"].fillna(False).astype(bool)].copy()
    sw["as_of"] = pd.to_datetime(sw["as_of"]).dt.normalize()
    sw = sw.merge(
        px[
            [
                "as_of",
                "$close",
                "$high",
                "$low",
                "MA5",
                "MA20",
                "rv5",
                "rv20",
                "vol5_pct_120d",
                "vol_regime",
            ]
        ],
        on="as_of",
        how="left",
    )
    ann = rules.annotate_switches(sw, px, cfg)
    if ann.empty:
        return []

    events: list[dict[str, Any]] = []

    def base_row(s: pd.Series, pattern: str) -> dict[str, Any]:
        return {
            "code": code_u,
            "signal_date": str(pd.Timestamp(s["as_of"]).date()),
            "pattern": pattern,
            "side": s["side"],
            "ann_action": s["action"],
            "prior10": s["prior10"],
            "prior10_bucket": _prior_bucket(s["prior10"]),
            "vol_pct": s["vol5_pct_120d"],
            "above_ma20": (
                bool(s["above_ma20"])
                if s["above_ma20"] is not None and not pd.isna(s["above_ma20"])
                else None
            ),
            "signal_px": float(s["close"]),
        }

    # A / B — production actions
    for _, s in ann.iterrows():
        if s["action"] == "买观察":
            sim = rules.simulate_buy_watch(px, s, cfg)
            row = base_row(s, "A.生产买观察")
            if sim["executed"]:
                row.update(_path_pnl(px, rules, sim["exec_date"], sim["exec_px"], sim["stop"], cfg.hold_eval_days))
            else:
                row.update(
                    {
                        "executed": False,
                        "exec_date": None,
                        "exec_px": None,
                        "stop": sim["stop"],
                        "exit_date": None,
                        "exit_why": sim["reason"],
                        "fwd_ret": np.nan,
                        "fwd_mfe": np.nan,
                        "fwd_mae": np.nan,
                        "complete": False,
                    }
                )
            events.append(row)
        elif s["action"] == "买确认候选":
            sim = rules.simulate_buy_confirm(px, s)
            row = base_row(s, "B.生产买确认")
            if sim["executed"]:
                row.update(_path_pnl(px, rules, sim["exec_date"], sim["exec_px"], sim["stop"], cfg.hold_eval_days))
            else:
                row.update(
                    {
                        "executed": False,
                        "exec_date": None,
                        "exec_px": None,
                        "stop": None,
                        "exit_date": None,
                        "exit_why": "无次日",
                        "fwd_ret": np.nan,
                        "fwd_mfe": np.nan,
                        "fwd_mae": np.nan,
                        "complete": False,
                    }
                )
            events.append(row)

    # C / D — all reds forced reclaim, split by prior bucket (incl. ones production floors)
    for _, s in ann[ann["side"] == "down"].iterrows():
        p10 = s["prior10"]
        if p10 is None or pd.isna(p10):
            continue
        p10 = float(p10)
        if p10 < -0.08:
            pattern = "C.深跌红收复"
        elif p10 <= 0.0:
            pattern = "D.中段红收复"
        else:
            continue  # skip 翻红 reds here
        # Skip if already counted as production 买观察 (avoid double-count in C/D)
        if s["action"] == "买观察":
            continue
        sim = rules.simulate_buy_watch(px, s, cfg)
        row = base_row(s, pattern)
        if sim["executed"]:
            row.update(_path_pnl(px, rules, sim["exec_date"], sim["exec_px"], sim["stop"], cfg.hold_eval_days))
        else:
            row.update(
                {
                    "executed": False,
                    "exec_date": None,
                    "exec_px": None,
                    "stop": sim["stop"],
                    "exit_date": None,
                    "exit_why": sim["reason"],
                    "fwd_ret": np.nan,
                    "fwd_mfe": np.nan,
                    "fwd_mae": np.nan,
                    "complete": False,
                }
            )
        events.append(row)

    # E — deep greens forced buy-confirm
    for _, s in ann[ann["side"] == "up"].iterrows():
        p10 = s["prior10"]
        if p10 is None or pd.isna(p10) or float(p10) >= -0.08:
            continue
        if s["action"] == "买确认候选":
            continue
        sim = rules.simulate_buy_confirm(px, s)
        row = base_row(s, "E.深跌后绿确认")
        if sim["executed"]:
            row.update(_path_pnl(px, rules, sim["exec_date"], sim["exec_px"], sim["stop"], cfg.hold_eval_days))
        else:
            row.update(
                {
                    "executed": False,
                    "exec_date": None,
                    "exec_px": None,
                    "stop": None,
                    "exit_date": None,
                    "exit_why": "无次日",
                    "fwd_ret": np.nan,
                    "fwd_mfe": np.nan,
                    "fwd_mae": np.nan,
                    "complete": False,
                }
            )
        events.append(row)

    return events


def summarize(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pattern, g0 in events.groupby("pattern"):
        n_sig = len(g0)
        ex = g0[g0["executed"] == True]  # noqa: E712
        n_exec = len(ex)
        comp = ex[ex["complete"] == True]  # noqa: E712
        use = comp if len(comp) >= 5 else ex  # fall back if too few complete
        rows.append(
            {
                "pattern": pattern,
                "n_signals": n_sig,
                "n_executed": n_exec,
                "exec_rate": n_exec / n_sig if n_sig else np.nan,
                "n_complete": int(len(comp)),
                "n_scored": int(len(use)),
                "win_rate": float((use["fwd_ret"] > 0).mean()) if len(use) else np.nan,
                "mean_ret": float(use["fwd_ret"].mean()) if len(use) else np.nan,
                "median_ret": float(use["fwd_ret"].median()) if len(use) else np.nan,
                "mean_mfe": float(use["fwd_mfe"].mean()) if len(use) else np.nan,
                "mean_mae": float(use["fwd_mae"].mean()) if len(use) else np.nan,
                "stop_rate": float((use["exit_why"] == "止损").mean()) if len(use) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("pattern")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--validation-dir",
        type=Path,
        default=_PROJECT_ROOT
        / "workspace/decision_packs/20260720/regime_transition_validation_q90_to0720",
    )
    p.add_argument("--end-date", default="2026-08-03")
    p.add_argument("--code", action="append", default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    val = args.validation_dir
    if not val.is_absolute():
        val = _PROJECT_ROOT / val
    out_dir = args.out_dir or (val / "triangle_decision_backtest_buy_patterns")
    if not out_dir.is_absolute():
        out_dir = _PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    oos, _rev, _evt = load_validation_csvs(val)
    oos["as_of"] = pd.to_datetime(oos["as_of"]).dt.normalize()
    oos["code"] = oos["code"].astype(str).str.upper()

    if args.code:
        codes = [c.upper() for c in args.code]
    else:
        codes = [
            c
            for c in load_cluster_mapping_codes(CLUSTER_MAPPING_SELECTED_TXT)
            if c in set(oos["code"])
        ]
        if not codes:
            codes = sorted(oos["code"].unique().tolist())

    rules = _load_rules_mod()
    cfg = rules.RuleConfig()
    plot_mod = rules._load_plot_mod()
    plot_mod._init_qlib(None)

    print(
        f"[INFO] buy-pattern backtest codes={len(codes)} end={args.end_date} "
        f"reclaim_days={cfg.buy_reclaim_days} hold={cfg.hold_eval_days}"
    )
    print(f"[INFO] OOS span {oos['as_of'].min().date()} -> {oos['as_of'].max().date()}")

    all_events: list[pd.DataFrame] = []
    failed: list[tuple[str, str]] = []
    for code in codes:
        try:
            ev = evaluate_code(code, oos, rules, plot_mod, cfg, end_date=args.end_date)
            if ev:
                all_events.append(pd.DataFrame(ev))
            print(f"[OK] {code} events={len(ev)}")
        except Exception as exc:  # noqa: BLE001
            failed.append((code, str(exc)))
            print(f"[ERROR] {code}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                return 2

    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    events_path = out_dir / "buy_pattern_events.csv"
    events.to_csv(events_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] wrote {events_path} rows={len(events)}")

    if not events.empty:
        summ = summarize(events)
        summ_path = out_dir / "buy_pattern_summary.csv"
        summ.to_csv(summ_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] wrote {summ_path}")
        pd.set_option("display.width", 220)
        pd.set_option("display.max_columns", 20)
        print("\n=== 全品种全时段汇总（完整样本优先） ===")
        print(summ.to_string(index=False))

        # SH515050 focus
        focus = events[events["code"] == "SH515050"]
        if not focus.empty:
            print("\n=== SH515050 个案 ===")
            cols = [
                "pattern",
                "signal_date",
                "ann_action",
                "prior10",
                "executed",
                "exec_date",
                "exit_why",
                "fwd_ret",
                "complete",
            ]
            print(focus[cols].to_string(index=False))

    if failed:
        print(f"[WARN] failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
