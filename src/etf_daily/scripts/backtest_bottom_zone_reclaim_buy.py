#!/usr/bin/env python3
"""Backtest: bottom alternating-zone red triangle -> reclaim high -> next-open buy.

Question (from OPS miss on SZ159509 2026-07-28):
    底部交替区里的红三角，若随后收复该红三角日高点，次日开盘试多是否有正期望？
    现生产逻辑只在「顶部区已失效」分支做收复买，底部交替区一律只观察。
    本脚本不改任何生产逻辑，只对历史 OOS 信号做事件回测，供是否扩展决策。

Groups compared (same reclaim simulation everywhere):
    - 底部区-末红:  red legs that are the LAST red leg of a 底部交替区 (SZ159509 case)
    - 底部区-非末红: earlier red legs inside a 底部交替区
    - 非底部区红:    red triangles not belonging to any 底部交替区 (baseline)

Simulation matches production simulate_buy_watch + buy-watch exit path:
    reclaim window = cfg.buy_reclaim_days; entry = next open after first close
    above the red-day high; stop = red-day low; exit = stop hit (low<=stop) or
    cfg.hold_eval_days expiry.
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

from etf_daily.lib.regime_transition_plot import (
    compute_volatility_regime_frame,
    load_validation_csvs,
)
from etf_daily.lib.symbol_trend_board import load_cluster_mapping_codes
from etf_daily.paths import CLUSTER_MAPPING_SELECTED_TXT


def _load_rules_mod():
    """Load sibling backtest_triangle_decision_rules.py as a module."""
    path = _PROJECT_ROOT / "decision_pack" / "scripts" / "backtest_triangle_decision_rules.py"
    spec = importlib.util.spec_from_file_location("triangle_decision_rules_bt", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    # dataclass processing resolves cls.__module__ via sys.modules — register first.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _prior10_bucket(prior10: Any) -> str:
    if prior10 is None or pd.isna(prior10):
        return "未知"
    v = float(prior10)
    if v < -0.08:
        return "深跌(<-8%)"
    if v > 0.0:
        return "翻红(>0%)"
    return "中段(-8%~0%)"


def evaluate_code_reclaims(
    code: str,
    oos: pd.DataFrame,
    rules: Any,
    plot_mod: Any,
    cfg: Any,
    *,
    end_date: str,
) -> list[dict[str, Any]]:
    """All red-triangle reclaim simulations for one code, tagged by zone context."""
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
    zones = rules.detect_alternating_zones(ann, cfg)
    bottom = (
        zones[zones["kind"] == "底部交替区"].copy()
        if not zones.empty
        else zones
    )

    reds = ann[ann["side"] == "down"].copy()
    events: list[dict[str, Any]] = []
    for _, s in reds.sort_values("as_of").iterrows():
        d = pd.Timestamp(s["as_of"]).normalize()
        group = "非底部区红"
        zone_pat = ""
        if bottom is not None and not bottom.empty:
            hit = bottom[
                (pd.to_datetime(bottom["zone_start"]).dt.normalize() <= d)
                & (pd.to_datetime(bottom["zone_end"]).dt.normalize() >= d)
            ]
            if not hit.empty:
                z = hit.iloc[0]
                zone_pat = str(z["pattern"])
                zone_reds = reds[
                    (reds["as_of"] >= pd.Timestamp(z["zone_start"]).normalize())
                    & (reds["as_of"] <= pd.Timestamp(z["zone_end"]).normalize())
                ]
                last_red_d = pd.Timestamp(zone_reds["as_of"].max()).normalize()
                group = "底部区-末红" if d == last_red_d else "底部区-非末红"

        sim = rules.simulate_buy_watch(px, s, cfg)
        base = {
            "code": code_u,
            "signal_date": str(d.date()),
            "group": group,
            "zone_pattern": zone_pat,
            "ann_action": s["action"],
            "prior10": s["prior10"],
            "prior10_bucket": _prior10_bucket(s["prior10"]),
            "above_ma20": bool(s["above_ma20"])
            if s["above_ma20"] is not None and not pd.isna(s["above_ma20"])
            else None,
            "vol_pct": s["vol5_pct_120d"],
            "signal_px": float(s["close"]),
            "executed": bool(sim["executed"]),
            "exec_date": None
            if sim["exec_date"] is None
            else str(pd.Timestamp(sim["exec_date"]).date()),
            "exec_px": sim["exec_px"],
            "reason": sim["reason"],
        }
        if sim["executed"]:
            fut = px[px["as_of"] >= sim["exec_date"]].head(cfg.hold_eval_days)
            exit_px = float(fut.iloc[-1]["$close"]) if not fut.empty else sim["exec_px"]
            exit_why = "到期"
            for _, r in fut.iterrows():
                if sim["stop"] is not None and float(r["$low"]) <= float(sim["stop"]):
                    exit_px = float(sim["stop"])
                    exit_why = "止损"
                    break
            ret = exit_px / float(sim["exec_px"]) - 1.0
            st = rules.forward_stats(px, sim["exec_date"], float(sim["exec_px"]), cfg.hold_eval_days)
            events.append(
                {
                    **base,
                    "exit_why": exit_why,
                    "fwd_ret": ret,
                    "fwd_mfe": st["mfe"],
                    "fwd_mae": st["mae"],
                    # 完整样本:走满持有期或中途止损(未被 end_date 截断)
                    "complete": bool(exit_why == "止损" or len(fut) >= cfg.hold_eval_days),
                }
            )
        else:
            st = rules.forward_stats(px, d, float(s["close"]), cfg.hold_eval_days)
            events.append(
                {
                    **base,
                    "exit_why": "未收复",
                    "fwd_ret": np.nan,
                    "fwd_mfe": st["mfe"],
                    "fwd_mae": st["mae"],
                    "complete": False,
                }
            )
    return events


def summarize(events: pd.DataFrame, *, complete_only: bool) -> pd.DataFrame:
    ex = events[events["executed"] == True]  # noqa: E712
    if complete_only:
        ex = ex[ex["complete"] == True]  # noqa: E712
    rows = []
    for (group, bucket), g in ex.groupby(["group", "prior10_bucket"]):
        rows.append(
            {
                "group": group,
                "prior10_bucket": bucket,
                "n_exec": int(len(g)),
                "win_rate": float((g["fwd_ret"] > 0).mean()),
                "mean_ret": float(g["fwd_ret"].mean()),
                "median_ret": float(g["fwd_ret"].median()),
                "mean_mfe": float(g["fwd_mfe"].mean()),
                "mean_mae": float(g["fwd_mae"].mean()),
                "stop_rate": float((g["exit_why"] == "止损").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["group", "prior10_bucket"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--validation-dir",
        type=Path,
        default=_PROJECT_ROOT
        / "runtime/decision_packs/20260720/regime_transition_validation_q90_to0720",
    )
    p.add_argument("--end-date", default="2026-08-03")
    p.add_argument("--code", action="append", default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--include-truncated",
        action="store_true",
        help="汇总时包含持有期被 end-date 截断的样本（默认只统计完整样本）",
    )
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    val = args.validation_dir
    if not val.is_absolute():
        val = _PROJECT_ROOT / val
    out_dir = args.out_dir or (val / "triangle_decision_backtest_bottom_reclaim")
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
        f"[INFO] bottom-zone reclaim backtest codes={len(codes)} end={args.end_date} "
        f"reclaim_days={cfg.buy_reclaim_days} hold={cfg.hold_eval_days}"
    )
    print(
        f"[INFO] OOS span {oos['as_of'].min().date()} -> {oos['as_of'].max().date()}"
    )

    all_events: list[pd.DataFrame] = []
    failed: list[tuple[str, str]] = []
    for code in codes:
        try:
            ev = evaluate_code_reclaims(
                code, oos, rules, plot_mod, cfg, end_date=args.end_date
            )
            if ev:
                all_events.append(pd.DataFrame(ev))
            print(f"[OK] {code} reclaim_events={len(ev)}")
        except Exception as exc:  # noqa: BLE001
            failed.append((code, str(exc)))
            print(f"[ERROR] {code}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                return 2

    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    events_path = out_dir / "bottom_reclaim_events.csv"
    events.to_csv(events_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] wrote {events_path} rows={len(events)}")

    if not events.empty:
        summ = summarize(events, complete_only=not args.include_truncated)
        summ_path = out_dir / "bottom_reclaim_summary.csv"
        summ.to_csv(summ_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] wrote {summ_path}")
        pd.set_option("display.width", 200)
        print(summ.to_string(index=False))

    if failed:
        print(f"[WARN] failed codes: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
