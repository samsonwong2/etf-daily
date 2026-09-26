#!/usr/bin/env python3
"""Research backtest: fig7–11 checklist auxiliary buy/sell rules.

Does NOT change production hold_up. Default data source is frozen regime HTML
(fig7 OHLCV) to match report diagnostics without Qlib.

Example::

    PYTHONPATH=. python src/etf_daily/scripts/backtest_fair_path_band_signals.py \\
      --listing-dir ~/etf-daily-output/temp/plotly_outputs/20260821_from_listing \\
      --data-source html --as-of 2026-08-21 --train-cutoff 2025-01-01 \\
      --entry-mode B123 --ablation \\
      --out-dir ~/etf-daily-output/temp/plotly_outputs/20260821_fair_path_band_bt
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from etf_daily.lib.fair_path_band_signals import (  # noqa: E402
    ENTRY_MODES,
    atr20,
    build_checklist_flags_frame,
    compute_multi_scale_frame,
    detect_checklist_entries,
    detect_lower_band_entries,
    detect_s1_trim_signals,
    entries_from_flags,
    find_regime_html,
    load_ohlcv_from_regime_html,
    load_qfq_close,
)

SKIP_CODES_NO_FIG7 = {"SH511010", "SH511260"}

_SYMBOL_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}
_FLAGS_CACHE: dict[tuple[str, str, str, float, str, str], pd.DataFrame] = {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--listing-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260821_from_listing",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260821_fair_path_band_bt",
    )
    p.add_argument("--as-of", default="2026-08-21")
    p.add_argument("--train-cutoff", default="2025-01-01", help="OOS entries on/after this")
    p.add_argument("--max-hold-days", type=int, default=40)
    p.add_argument("--target-panel", default="fig10", choices=["fig8", "fig9", "fig10"])
    p.add_argument(
        "--entry-mode",
        default="B123",
        choices=list(ENTRY_MODES),
    )
    p.add_argument(
        "--data-source",
        default="html",
        choices=["html", "qlib"],
    )
    p.add_argument("--deep-gap", type=float, default=-0.15)
    p.add_argument(
        "--touch-panels",
        default="fig9,fig10",
        help="comma-separated panels for B1 touch (fig9,fig10 or fig10)",
    )
    p.add_argument(
        "--b3-mode",
        default="default",
        choices=["default", "fig10_only", "reclaim_from_lo"],
    )
    p.add_argument("--signal-gap-days", type=int, default=5)
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--code", action="append", default=None, help="limit to code(s)")
    return p.parse_args(argv)


def _load_marks(trades_dir: Path, code: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {"ed": set(), "ru": set(), "buy": set(), "sell": set()}
    base = trades_dir
    ed = base / f"{code}_early_down_marks.csv"
    if ed.exists():
        df = pd.read_csv(ed)
        for _, r in df.iterrows():
            out["ed"].add(pd.Timestamp(r["date"]).strftime("%Y-%m-%d"))
    ru = base / f"{code}_recovery_up_marks.csv"
    if ru.exists():
        df = pd.read_csv(ru)
        for _, r in df.iterrows():
            d0 = pd.Timestamp(r["date"]).strftime("%Y-%m-%d")
            d1 = pd.Timestamp(r["end"]).strftime("%Y-%m-%d")
            for d in pd.date_range(d0, d1, freq="B"):
                out["ru"].add(d.strftime("%Y-%m-%d"))
    tr = base / f"{code}_trades.csv"
    if tr.exists():
        df = pd.read_csv(tr)
        for _, r in df.iterrows():
            ds = pd.Timestamp(r["date"]).strftime("%Y-%m-%d")
            side = str(r["side"]).upper()
            if side == "BUY":
                out["buy"].add(ds)
            elif side == "SELL":
                out["sell"].add(ds)
    return out


def _load_regime_segments(trades_dir: Path, code: str) -> pd.Series:
    seg_path = trades_dir / f"{code}_segments.csv"
    if not seg_path.exists():
        return pd.Series(dtype=object)
    seg = pd.read_csv(seg_path)
    seg["start"] = pd.to_datetime(seg["start"])
    seg["end"] = pd.to_datetime(seg["end"])
    pieces: list[pd.Series] = []
    for _, r in seg.iterrows():
        idx = pd.date_range(r["start"], r["end"], freq="B")
        pieces.append(pd.Series(str(r["regime"]), index=idx))
    if not pieces:
        return pd.Series(dtype=object)
    s = pd.concat(pieces)
    return s[~s.index.duplicated(keep="last")]


def _dedupe_signals(entries: pd.Series, *, gap_days: int) -> pd.Series:
    sig_idx = entries.index[entries]
    keep = pd.Series(False, index=entries.index)
    last = None
    for d in sig_idx:
        if last is None or (d - last).days >= gap_days:
            keep.loc[d] = True
            last = d
    return entries & keep


def simulate_trades(
    ms: pd.DataFrame,
    ohlcv: pd.DataFrame,
    *,
    entries: pd.Series,
    target_panel: str,
    max_hold: int,
) -> list[dict[str, Any]]:
    close = ms["px"]
    low = ohlcv["$low"].reindex(ms.index).astype(float)
    atr_s = atr20(ohlcv).reindex(ms.index)
    path_col = f"{target_panel}_path"
    trades: list[dict[str, Any]] = []
    i = 0
    n = len(ms)
    while i < n - 1:
        if not bool(entries.iloc[i]):
            i += 1
            continue
        entry_i = i + 1
        if entry_i >= n:
            break
        entry_date = ms.index[entry_i].strftime("%Y-%m-%d")
        entry_px = float(close.iloc[entry_i])
        atr_v = (
            float(atr_s.iloc[entry_i])
            if pd.notna(atr_s.iloc[entry_i])
            else entry_px * 0.1
        )
        stop = entry_px - atr_v
        target = (
            float(ms[path_col].iloc[entry_i])
            if pd.notna(ms[path_col].iloc[entry_i])
            else float("nan")
        )
        exit_reason = None
        exit_px = entry_px
        exit_i = entry_i
        for j in range(entry_i + 1, n):
            px = float(close.iloc[j])
            lo = float(low.iloc[j])
            hold_days = j - entry_i
            if np.isfinite(stop) and lo <= stop:
                exit_reason = "stop"
                exit_px = stop
                exit_i = j
                break
            if np.isfinite(target) and px >= target:
                exit_reason = "target_path"
                exit_px = px
                exit_i = j
                break
            if hold_days >= max_hold:
                exit_reason = "max_hold"
                exit_px = px
                exit_i = j
                break
        if exit_reason is None:
            exit_reason = "max_hold"
            exit_px = float(close.iloc[n - 1])
            exit_i = n - 1
        signal_date = ms.index[i].strftime("%Y-%m-%d")
        fwd: dict[str, float | None] = {}
        for horizon in (5, 10, 20):
            j = entry_i + horizon
            if j < n:
                fwd[f"fwd_ret_{horizon}d"] = float(close.iloc[j] / entry_px - 1.0)
            else:
                fwd[f"fwd_ret_{horizon}d"] = None
        trades.append(
            dict(
                signal_date=signal_date,
                entry_date=entry_date,
                exit_date=ms.index[exit_i].strftime("%Y-%m-%d"),
                entry_px=entry_px,
                exit_px=exit_px,
                ret=exit_px / entry_px - 1.0,
                hold_days=int(exit_i - entry_i),
                exit_reason=exit_reason,
                target=target,
                stop=stop,
                **fwd,
            )
        )
        i = exit_i + 1
    return trades


def _lead_rate(trades: list[dict[str, Any]], buy_dates: set[str], *, window: int = 20) -> float | None:
    if not trades or not buy_dates:
        return None
    hits = 0
    for t in trades:
        sig = pd.Timestamp(t["signal_date"])
        for b in buy_dates:
            bd = pd.Timestamp(b)
            delta = (bd - sig).days
            if 0 <= delta <= window:
                hits += 1
                break
    return hits / len(trades)


def _pool_summary(trades_df: pd.DataFrame, *, n_codes: int, label: str) -> dict[str, Any]:
    pool: dict[str, Any] = {
        "label": label,
        "n_codes": n_codes,
        "n_trades": len(trades_df),
    }
    if trades_df.empty:
        return pool
    pool.update(
        dict(
            win_rate=float((trades_df["ret"] > 0).mean()),
            mean_ret=float(trades_df["ret"].mean()),
            compound=float((1 + trades_df["ret"]).prod() - 1),
            target_hit_rate=float((trades_df["exit_reason"] == "target_path").mean()),
        )
    )
    for h in (5, 10, 20):
        col = f"fwd_ret_{h}d"
        if col in trades_df.columns:
            s = trades_df[col].dropna()
            if len(s):
                pool[f"fwd_hit_{h}d"] = float((s > 0).mean())
                pool[f"fwd_mean_{h}d"] = float(s.mean())
    if "lead_rate_20d" in trades_df.columns:
        lr = trades_df["lead_rate_20d"].dropna()
        if len(lr):
            pool["lead_rate_20d"] = float(lr.mean())
    return pool


def _load_symbol_data(
    code: str,
    *,
    listing_dir: Path,
    listing_start: str,
    as_of: str,
    data_source: str,
    qlib_inited: bool,
) -> tuple[pd.Series, pd.DataFrame, bool]:
    if data_source == "html":
        html_path = find_regime_html(listing_dir, code)
        if html_path is None:
            raise FileNotFoundError(f"no html for {code}")
        close, ohlcv = load_ohlcv_from_regime_html(html_path)
        if as_of:
            cut = pd.Timestamp(as_of)
            close = close[close.index <= cut]
            ohlcv = ohlcv[ohlcv.index <= cut]
        return close, ohlcv, qlib_inited
    init_qlib = not qlib_inited
    close, ohlcv = load_qfq_close(code, listing_start, as_of, init_qlib=init_qlib)
    return close, ohlcv, init_qlib or qlib_inited


def run_backtest_for_config(
    *,
    listing_dir: Path,
    listing: pd.DataFrame,
    codes: list[str],
    as_of: str,
    train_cutoff: str,
    data_source: str,
    entry_mode: str,
    deep_gap: float,
    touch_panels: tuple[str, ...],
    b3_mode: str,
    target_panel: str,
    max_hold_days: int,
    signal_gap_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    trades_dir = listing_dir / "trades"
    all_trades: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    s1_rows: list[dict[str, Any]] = []
    qlib_inited = False
    cut = pd.Timestamp(train_cutoff)

    for code in codes:
        if code in SKIP_CODES_NO_FIG7:
            stats_rows.append({"code": code, "skip": "no_fig7_panels"})
            continue
        sub = listing[listing["code"].astype(str).str.upper() == code]
        if sub.empty:
            continue
        listing_start = pd.Timestamp(sub.iloc[0]["listing"]).strftime("%Y-%m-%d")
        try:
            cache_key = (code, as_of, data_source)
            cached = _SYMBOL_CACHE.get(cache_key)
            if cached is None:
                close, ohlcv, qlib_inited = _load_symbol_data(
                    code,
                    listing_dir=listing_dir,
                    listing_start=listing_start,
                    as_of=as_of,
                    data_source=data_source,
                    qlib_inited=qlib_inited,
                )
                if len(close) < 120:
                    stats_rows.append({"code": code, "skip": "short_history"})
                    continue
                ms = compute_multi_scale_frame(close)
                marks = _load_marks(trades_dir, code)
                regime_s = _load_regime_segments(trades_dir, code)
                atr_s = atr20(ohlcv).reindex(ms.index)
                cached = dict(
                    close=close,
                    ohlcv=ohlcv,
                    ms=ms,
                    marks=marks,
                    regime_s=regime_s,
                    atr_s=atr_s,
                    listing_start=listing_start,
                )
                _SYMBOL_CACHE[cache_key] = cached
            else:
                close = cached["close"]
                ohlcv = cached["ohlcv"]
                ms = cached["ms"]
                marks = cached["marks"]
                regime_s = cached["regime_s"]
                atr_s = cached["atr_s"]
                listing_start = cached["listing_start"]
            entries: pd.Series
            if entry_mode == "baseline_touch":
                entries = detect_lower_band_entries(ms, panels=touch_panels)
            else:
                fkey = (
                    code,
                    as_of,
                    data_source,
                    deep_gap,
                    ",".join(touch_panels),
                    b3_mode,
                )
                flags = _FLAGS_CACHE.get(fkey)
                if flags is None:
                    flags = build_checklist_flags_frame(
                        ms,
                        code=code,
                        atr_s=atr_s,
                        regime_s=regime_s,
                        ed_dates=marks["ed"],
                        ru_dates=marks["ru"],
                        buy_dates=marks["buy"],
                        sell_dates=marks["sell"],
                        listing_start=listing_start,
                        deep_gap=deep_gap,
                        touch_panels=touch_panels,
                        b3_mode=b3_mode,  # type: ignore[arg-type]
                    )
                    _FLAGS_CACHE[fkey] = flags
                entries = entries_from_flags(flags, mode=entry_mode)
            entries = entries & (ms.index >= cut)
            entries = _dedupe_signals(entries, gap_days=signal_gap_days)
            trades = simulate_trades(
                ms,
                ohlcv,
                entries=entries,
                target_panel=target_panel,
                max_hold=max_hold_days,
            )
            lr = _lead_rate(trades, marks["buy"], window=20)
            for t in trades:
                t["code"] = code
                t["lead_rate_20d"] = lr
            st: dict[str, Any] = {
                "code": code,
                "n_signals": int(entries.sum()),
                "n_trades": len(trades),
                "lead_rate_20d": lr,
            }
            if trades:
                rets = pd.Series([t["ret"] for t in trades])
                st.update(
                    dict(
                        win_rate=float((rets > 0).mean()),
                        mean_ret=float(rets.mean()),
                        compound=float((1 + rets).prod() - 1),
                        target_hit_rate=float(
                            sum(1 for t in trades if t["exit_reason"] == "target_path")
                            / len(trades)
                        ),
                    )
                )
            all_trades.extend(trades)
            stats_rows.append(st)

            # S1 trim evaluation while regime==up
            s1 = detect_s1_trim_signals(ms) & (ms.index >= cut)
            if not regime_s.empty:
                aligned = regime_s.reindex(ms.index).fillna("range")
                s1 = s1 & aligned.eq("up")
            for dt in ms.index[s1]:
                i = ms.index.get_loc(dt)
                row: dict[str, Any] = {"code": code, "date": pd.Timestamp(dt).strftime("%Y-%m-%d")}
                for h in (5, 10):
                    j = i + h
                    if j < len(ms):
                        row[f"fwd_ret_{h}d"] = float(ms["px"].iloc[j] / ms["px"].iloc[i] - 1.0)
                    else:
                        row[f"fwd_ret_{h}d"] = None
                s1_rows.append(row)
        except Exception as exc:  # noqa: BLE001
            stats_rows.append({"code": code, "skip": str(exc)})
            continue

    trades_df = pd.DataFrame(all_trades)
    stats_df = pd.DataFrame(stats_rows)
    s1_df = pd.DataFrame(s1_rows)
    s1_summary: dict[str, Any] = {"n_signals": len(s1_df)}
    if not s1_df.empty:
        for h in (5, 10):
            col = f"fwd_ret_{h}d"
            s = s1_df[col].dropna()
            if len(s):
                s1_summary[f"fwd_hit_{h}d"] = float((s < 0).mean())
                s1_summary[f"fwd_mean_{h}d"] = float(s.mean())
    label = (
        f"mode={entry_mode}|deep_gap={deep_gap}|touch={','.join(touch_panels)}|"
        f"b3={b3_mode}|target={target_panel}|hold={max_hold_days}"
    )
    pool = _pool_summary(trades_df, n_codes=len(codes), label=label)
    return trades_df, stats_df, pool, [s1_summary]


def _ablation_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Focused B123 grid: threshold sweep + exit sweep."""
    grid: list[dict[str, Any]] = []
    for deep_gap, touch_panels, b3_mode in itertools.product(
        [-0.10, -0.15, -0.20],
        [("fig9", "fig10"), ("fig10",)],
        ["default", "fig10_only", "reclaim_from_lo"],
    ):
        grid.append(
            dict(
                entry_mode="B123",
                deep_gap=deep_gap,
                touch_panels=touch_panels,
                b3_mode=b3_mode,
                target_panel="fig10",
                max_hold_days=40,
            )
        )
    for target_panel, max_hold_days in itertools.product(
        ["fig8", "fig9", "fig10"],
        [20, 40],
    ):
        grid.append(
            dict(
                entry_mode="B123",
                deep_gap=-0.15,
                touch_panels=("fig9", "fig10"),
                b3_mode="default",
                target_panel=target_panel,
                max_hold_days=max_hold_days,
            )
        )
    return grid


def _write_report(
    out_dir: Path,
    *,
    pool: dict[str, Any],
    variant_pools: list[dict[str, Any]],
    ablation_df: pd.DataFrame | None,
    s1_summary: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# 图7–11 辅助买卖规则回测报告",
        "",
        f"- listing-dir: `{args.listing_dir}`",
        f"- as-of: {args.as_of}",
        f"- OOS cutoff: {args.train_cutoff}",
        f"- data-source: {args.data_source}",
        "",
        "## 主配置结果",
        "",
        f"| 指标 | 值 |",
        f"|---|---|",
    ]
    for k in (
        "label",
        "n_codes",
        "n_trades",
        "win_rate",
        "mean_ret",
        "compound",
        "target_hit_rate",
        "fwd_hit_5d",
        "fwd_hit_10d",
        "fwd_hit_20d",
        "lead_rate_20d",
    ):
        if k in pool:
            v = pool[k]
            if isinstance(v, float) and k not in {"n_trades", "n_codes"}:
                lines.append(f"| {k} | {v:.2%} |" if "rate" in k or k.startswith("fwd_hit") else f"| {k} | {v:.4f} |")
            else:
                lines.append(f"| {k} | {v} |")

    if variant_pools:
        lines.extend(["", "## 入场变体对比", "", "| mode | n_trades | win_rate | mean_ret | target_hit |", "|---|---:|---:|---:|---:|"])
        for vp in variant_pools:
            wr = vp.get("win_rate")
            mr = vp.get("mean_ret")
            th = vp.get("target_hit_rate")
            lines.append(
                f"| {vp.get('entry_mode', vp.get('label', ''))} | {vp.get('n_trades', 0)} | "
                f"{wr:.1%} | {mr:.2%} | {th:.1%} |"
                if wr is not None
                else f"| {vp.get('entry_mode', '')} | {vp.get('n_trades', 0)} | — | — | — |"
            )

    if s1_summary.get("n_signals"):
        lines.extend(
            [
                "",
                "## S1 近上轨减仓（regime=up）",
                "",
                f"- n_signals: {s1_summary['n_signals']}",
            ]
        )
        for h in (5, 10):
            hk = f"fwd_hit_{h}d"
            mn = f"fwd_mean_{h}d"
            if hk in s1_summary:
                lines.append(
                    f"- {h}日后下跌比例: {s1_summary[hk]:.1%}，均收益: {s1_summary.get(mn, 0):.2%}"
                )

    if ablation_df is not None and not ablation_df.empty:
        rec = ablation_df[ablation_df["n_trades"] >= 20].sort_values(
            ["mean_ret", "win_rate"], ascending=[False, False]
        )
        lines.extend(["", "## 消融推荐（n_trades≥20，按 mean_ret）", ""])
        if rec.empty:
            lines.append("_无足够样本组合_")
        else:
            top = rec.head(8)
            lines.extend(
                [
                    "| entry_mode | deep_gap | touch | b3 | target | hold | n_trades | win_rate | mean_ret |",
                    "|---|---:|---|---|---|---:|---:|---:|---:|",
                ]
            )
            for _, r in top.iterrows():
                lines.append(
                    f"| {r['entry_mode']} | {r['deep_gap']:.0%} | {r['touch_panels']} | {r['b3_mode']} | "
                    f"{r['target_panel']} | {int(r['max_hold_days'])} | {int(r['n_trades'])} | "
                    f"{r['win_rate']:.1%} | {r['mean_ret']:.2%} |"
                )
        lines.extend(
            [
                "",
                "## 规则调整建议（回测修订）",
                "",
                "- **弃用** `baseline_touch`（仅触轨+B2）作主买点：OOS 胜率通常低于完整 B123。",
                "- **保留 B3**：强制短尺收复/止跌后再入场，过滤「还在下坠」的触轨。",
                "- **deep_gap**：若 B123 信号过少，可试 −20%；若误报多，收至 −15%～−10%。",
                "- **目标 path**：优先 fig10；长持有(40d) 通常优于 20d。",
                "- **B5/B4**：仅作加严过滤器；样本少时不必默认开启。",
                "- **S1**：在 up 态近上轨后，短尺收益偏负时可作减仓参考，非硬卖。",
            ]
        )

    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    listing_dir = args.listing_dir
    if not listing_dir.is_absolute():
        listing_dir = _PROJECT_ROOT / listing_dir
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = _PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ld = listing_dir / "listing_dates.csv"
    if not ld.exists():
        raise SystemExit(f"missing {ld}")
    listing = pd.read_csv(ld)
    if args.code:
        codes = [c.upper() for c in args.code]
    else:
        codes = listing["code"].astype(str).str.upper().tolist()
    if args.max_codes:
        codes = codes[: int(args.max_codes)]

    touch_panels = tuple(p.strip() for p in args.touch_panels.split(",") if p.strip())

    trades_df, stats_df, pool, s1_parts = run_backtest_for_config(
        listing_dir=listing_dir,
        listing=listing,
        codes=codes,
        as_of=args.as_of,
        train_cutoff=args.train_cutoff,
        data_source=args.data_source,
        entry_mode=args.entry_mode,
        deep_gap=args.deep_gap,
        touch_panels=touch_panels,
        b3_mode=args.b3_mode,
        target_panel=args.target_panel,
        max_hold_days=args.max_hold_days,
        signal_gap_days=args.signal_gap_days,
    )
    trades_df.to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")
    stats_df.to_csv(out_dir / "by_code.csv", index=False, encoding="utf-8-sig")
    (out_dir / "pool_summary.json").write_text(
        json.dumps(pool, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    variant_pools: list[dict[str, Any]] = []
    if args.entry_mode == "B123":
        for mode in ENTRY_MODES:
            if mode == args.entry_mode:
                continue
            tdf, _, vp, _ = run_backtest_for_config(
                listing_dir=listing_dir,
                listing=listing,
                codes=codes,
                as_of=args.as_of,
                train_cutoff=args.train_cutoff,
                data_source=args.data_source,
                entry_mode=mode,
                deep_gap=args.deep_gap,
                touch_panels=touch_panels,
                b3_mode=args.b3_mode,
                target_panel=args.target_panel,
                max_hold_days=args.max_hold_days,
                signal_gap_days=args.signal_gap_days,
            )
            vp["entry_mode"] = mode
            variant_pools.append(vp)
            tdf.to_csv(out_dir / f"trades_{mode}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(variant_pools).to_csv(out_dir / "variant_summary.csv", index=False, encoding="utf-8-sig")

    ablation_df: pd.DataFrame | None = None
    if args.ablation:
        ab_rows: list[dict[str, Any]] = []
        for cfg in _ablation_grid(args):
            if cfg["entry_mode"] != "B123":
                continue  # ablation focused on B123 thresholds per plan
            _, _, ab_pool, _ = run_backtest_for_config(
                listing_dir=listing_dir,
                listing=listing,
                codes=codes,
                as_of=args.as_of,
                train_cutoff=args.train_cutoff,
                data_source=args.data_source,
                signal_gap_days=args.signal_gap_days,
                **cfg,
            )
            ab_rows.append({**cfg, **ab_pool})
        ablation_df = pd.DataFrame(ab_rows)
        ablation_df.to_csv(out_dir / "ablation.csv", index=False, encoding="utf-8-sig")

    s1_summary = s1_parts[0] if s1_parts else {}
    _write_report(
        out_dir,
        pool=pool,
        variant_pools=variant_pools,
        ablation_df=ablation_df,
        s1_summary=s1_summary,
        args=args,
    )

    print(f"[OK] fair_path_band_bt n_trades={len(trades_df)} out={out_dir}")
    if pool.get("win_rate") is not None:
        print(
            f"  pool win_rate={pool['win_rate']:.1%} "
            f"target_hit={pool.get('target_hit_rate', 0):.1%} "
            f"mean_ret={pool.get('mean_ret', 0):.2%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
