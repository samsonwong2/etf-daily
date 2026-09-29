#!/usr/bin/env python3
"""B1235 checklist backtest: per-condition ablation + threshold sensitivity.

Pools every equity ETF, scores B1/B2/B3/B5 on each historical bar exactly like
``etf-daily b1235`` (causal fig7–11 paths/bands), and measures outcomes from the
next bar's open: forward 5/10/20/40d return, hit rate, excess vs the same-date
universe mean, 20d max adverse excursion, and an ATR-stop / fig8-target exit sim.
Stats are split in-sample / out-of-sample at ``--split-date``; pick variants on
IS and read OOS.

Data sources (one of)::

    etf-daily b1235-backtest --listing-dir "$PLOTLY_ROOT/20260929_from_listing"
    etf-daily b1235-backtest --ohlcv-dir /path/to/csvs   # CODE.csv: date,open,high,low,close
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.lib.b1235_backtest import (  # noqa: E402
    B1235Params,
    add_universe_baseline,
    basket_curve,
    build_symbol_panel,
    curve_stats,
    dedupe_signals,
    run_variant,
    signal_mask,
    simulate_exits,
    summarize,
)
from etf_daily.lib.fair_path_band_signals import load_ohlcv_from_regime_html  # noqa: E402

MIN_BARS = 250
MIN_PRICE = 0.5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--listing-dir", type=Path)
    src.add_argument("--ohlcv-dir", type=Path)
    p.add_argument("--out-dir", type=Path, default=None, help="default: {source}/b1235_backtest")
    p.add_argument("--start", default="2014-01-01")
    p.add_argument("--split-date", default="2022-01-01", help="OOS starts here")
    p.add_argument("--end", default=None, help="last signal date (default: data end)")
    p.add_argument("--cooldown", type=int, default=10, help="bars between kept signals per code")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args(argv)


def _load_listing_dates(listing: Path) -> dict[str, str]:
    ld = listing / "listing_dates.csv"
    if not ld.exists():
        return {}
    df = pd.read_csv(ld)
    return {str(r["code"]).upper(): str(r["listing"]) for _, r in df.iterrows()}


def _clean(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Drop the leading stretch where qfq prices < MIN_PRICE (3-dp rounding noise)."""
    low_px = ohlcv.index[ohlcv["$close"] < MIN_PRICE]
    if len(low_px):
        ohlcv = ohlcv.loc[ohlcv.index > low_px.max()]
    return ohlcv


def _sources(args: argparse.Namespace) -> list[tuple[str, str, str | None]]:
    """(code, path, listing_start) triples."""
    if args.listing_dir is not None:
        from etf_daily.scripts import backtest_fig12_six_states_pool as pool

        dates = _load_listing_dates(args.listing_dir)
        out = []
        for html in pool.list_equity_html(args.listing_dir):
            code, _ = pool.parse_html_label(html)
            out.append((code, str(html), dates.get(code.upper())))
        return out
    return [(f.stem.upper(), str(f), None) for f in sorted(args.ohlcv_dir.glob("*.csv"))]


def _read_ohlcv(path: str) -> pd.DataFrame:
    if path.endswith(".html"):
        _, ohlcv = load_ohlcv_from_regime_html(path)
        ohlcv.index = pd.DatetimeIndex(ohlcv.index).normalize()
        return ohlcv
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return df.rename(columns={c: f"${c}" for c in ("open", "high", "low", "close")})[
        ["$open", "$high", "$low", "$close"]
    ].astype(float)


def _build_one(item: tuple[str, str, str | None]) -> tuple[str, pd.DataFrame | None, pd.DataFrame | None, str]:
    code, path, listing_start = item
    try:
        raw = _read_ohlcv(path)
        first_bar = raw.index[0]
        ohlcv = _clean(raw)
        if len(ohlcv) < MIN_BARS:
            return code, None, None, "short_history"
        start = listing_start or first_bar
        return code, build_symbol_panel(code, ohlcv, listing_start=start), ohlcv, ""
    except Exception as exc:  # noqa: BLE001
        return code, None, None, f"{type(exc).__name__}: {exc}"


def load_panel(args: argparse.Namespace, out_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    cache = out_dir / "panel_cache.pkl"
    if cache.exists() and not args.no_cache:
        obj = pd.read_pickle(cache)
        return obj["panel"], obj["ohlcv"]
    items = _sources(args)
    panels, ohlcvs = [], {}
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for code, pn, oh, err in ex.map(_build_one, items):
            if pn is None:
                print(f"[skip] {code}: {err}", flush=True)
                continue
            panels.append(pn)
            ohlcvs[code] = oh
    panel = add_universe_baseline(pd.concat(panels).sort_index(kind="mergesort"))
    pd.to_pickle({"panel": panel, "ohlcv": ohlcvs}, cache)
    return panel, ohlcvs


def variant_grid() -> list[tuple[str, str, B1235Params]]:
    """(group, name, params). First row per group is its reference."""
    base = B1235Params()
    g: list[tuple[str, str, B1235Params]] = [
        ("基准", "全样本(无条件)", base.with_(use=())),
        ("基准", "B1235(现行)", base),
    ]
    for drop in ("B1", "B2", "B3", "B5"):
        use = tuple(c for c in base.use if c != drop)
        g.append(("留一消融", f"去掉{drop}", base.with_(use=use)))
    for c in ("B1", "B2", "B3", "B5"):
        g.append(("单条件", f"仅{c}", base.with_(use=(c,))))
    g.append(("单条件", "B1∧B2", base.with_(use=("B1", "B2"))))
    g.append(("单条件", "B1∧B2∧B3", base.with_(use=("B1", "B2", "B3"))))

    for parts, label in (
        (("touch9",), "B1=仅图9触下轨"),
        (("touch10",), "B1=仅图10触下轨"),
        (("deep",), "B1=仅图8深折价"),
        (("touch9", "touch10"), "B1=触轨(无深折价)"),
    ):
        g.append(("B1拆分", label, base.with_(b1_parts=parts)))
    for dg in (-0.05, -0.10, -0.20, -0.25, -0.30):
        g.append(("B1深折价阈值", f"deep_gap={dg:.0%}", base.with_(deep_gap=dg)))

    for parts, label in (
        (("g7",), "B2=仅图7 g>0"),
        (("g8",), "B2=仅图8 g>0"),
        (("p8up",), "B2=仅图8路径20日抬升"),
        (("p7up",), "B2=仅图7路径20日抬升"),
        (("g7", "g8"), "B2=图7或图8 g>0"),
    ):
        g.append(("B2拆分", label, base.with_(b2_parts=parts)))
    g.append(("B2拆分", "B2加严=图7且图8 g>0", base.with_(extra=("g7_and_g8_pos",))))

    for parts, label in (
        (("above10",), "B3=仅站上图10"),
        (("above11",), "B3=仅站上图11"),
        (("reclaim10",), "B3=仅图10昨触今收回"),
        (("above10", "reclaim10"), "B3=图10站上或收回"),
        (("above11", "reclaim10"), "B3=图11站上或图10收回"),
    ):
        g.append(("B3拆分", label, base.with_(b3_parts=parts)))

    for rr in (0.5, 1.0, 1.5, 2.5, 3.0, 4.0):
        g.append(("B5盈亏比阈值", f"min_rr={rr}", base.with_(min_rr=rr)))
    for k in (0.5, 1.5, 2.0):
        g.append(("B5止损ATR倍数", f"stop={k}×ATR", base.with_(stop_atr=k)))
    g.append(("B5目标", "B5目标=仅图8", base.with_(b5_targets=("fig8",))))
    g.append(("B5目标", "B5目标=仅图10", base.with_(b5_targets=("fig10",))))

    for ex, label in (
        ("age_ge_1y", "+上市≥1年"),
        ("fig10_g_pos", "+图10 g>0"),
        ("not_fig10_touch", "+当日不触图10下轨"),
    ):
        g.append(("新增过滤", label, base.with_(extra=(ex,))))

    g += proposal_grid()
    return g


B3_NO_ABOVE10 = ("above11", "reclaim10")


def proposal_grid() -> list[tuple[str, str, B1235Params]]:
    base = B1235Params()
    p1 = base.with_(b3_parts=B3_NO_ABOVE10)
    return [
        ("组合建议", "P1=B3去掉站上图10", p1),
        ("组合建议", "P2=P1+上市≥1年", p1.with_(extra=("age_ge_1y",))),
        ("组合建议", "P3=B1∧B3'+上市≥1年(去B2/B5)", p1.with_(use=("B1", "B3"), extra=("age_ge_1y",))),
        ("组合建议", "P4=P2+止损2×ATR", p1.with_(extra=("age_ge_1y",), stop_atr=2.0)),
    ]


def yearly_table(
    panel: pd.DataFrame,
    variants: list[tuple[str, str, B1235Params]],
    *,
    cooldown: int,
    start: str,
) -> pd.DataFrame:
    rows = []
    for _, name, p in variants:
        mask = dedupe_signals(panel, signal_mask(panel, p), cooldown=cooldown)
        sig = panel.loc[mask.to_numpy() & (panel.index >= pd.Timestamp(start))]
        for y, sub in sig.groupby(sig.index.year):
            rows.append({"variant": name, "year": int(y), **summarize(sub)})
    return pd.DataFrame(rows)


def basket_table(
    panel: pd.DataFrame,
    ohlcvs: dict[str, pd.DataFrame],
    variants: list[tuple[str, str, B1235Params]],
    *,
    cooldown: int,
    periods: dict[str, tuple[str, str]],
    hold: int = 20,
) -> pd.DataFrame:
    rows = []
    for _, name, p in variants:
        mask = dedupe_signals(panel, signal_mask(panel, p), cooldown=cooldown)
        daily = basket_curve(panel, ohlcvs, mask, hold=hold)
        for period, (a, b) in periods.items():
            d = daily.loc[a:b]
            rows.append({"variant": name, "period": period, **curve_stats(d)})
    return pd.DataFrame(rows)


def exit_stats(
    panel: pd.DataFrame,
    ohlcvs: dict[str, pd.DataFrame],
    mask: pd.Series,
    *,
    stop_atr: float,
    periods: dict[str, tuple[str, str]],
) -> dict[str, dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    m = pd.Series(mask.to_numpy(), index=panel.index)
    for code, pc in panel.groupby("code", sort=False):
        sig = m[(panel["code"] == code).to_numpy()]
        if not sig.any():
            continue
        trades.extend(simulate_exits(pc, ohlcvs[code], sig, stop_atr=stop_atr))
    df = pd.DataFrame(trades)
    out: dict[str, dict[str, Any]] = {}
    for name, (a, b) in periods.items():
        sub = df[(df["signal_date"] >= a) & (df["signal_date"] <= b)] if len(df) else df
        if sub.empty:
            out[name] = {"trades": 0}
            continue
        out[name] = {
            "trades": int(len(sub)),
            "win": float((sub["ret"] > 0).mean()),
            "avg": float(sub["ret"].mean()),
            "target_share": float((sub["exit"] == "target").mean()),
            "stop_share": float((sub["exit"] == "stop").mean()),
            "hold": float(sub["hold"].mean()),
        }
    return out


def run_grid(
    panel: pd.DataFrame,
    ohlcvs: dict[str, pd.DataFrame],
    *,
    periods: dict[str, tuple[str, str]],
    cooldown: int,
    grid: list[tuple[str, str, B1235Params]] | None = None,
) -> pd.DataFrame:
    rows = []
    for group, name, p in grid or variant_grid():
        res = run_variant(panel, p, periods=periods, cooldown=cooldown)
        mask = dedupe_signals(panel, signal_mask(panel, p), cooldown=cooldown)
        ex = exit_stats(panel, ohlcvs, mask, stop_atr=p.stop_atr, periods=periods) if p.use else {}
        for period, st in res.items():
            rows.append(
                {
                    "group": group,
                    "variant": name,
                    "period": period,
                    **st,
                    **{f"sim_{k}": v for k, v in ex.get(period, {}).items()},
                }
            )
    return pd.DataFrame(rows)


def _fmt_pct(v: Any, d: int = 2) -> str:
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v * 100:.{d}f}%"


def render_markdown(df: pd.DataFrame, periods: dict[str, tuple[str, str]]) -> str:
    lines = [
        "# B1235 回测（自动生成）",
        "",
        "入场=信号次日开盘；fwd=收盘/入场−1；超额=减同日全池等权均值；"
        "t=按信号日聚合后的超额20日 t 值；MAE=20日内最低价回撤；"
        "模拟=1×ATR 止损/图8路径止盈/40日到期。",
        "",
    ]
    for period, (a, b) in periods.items():
        sub = df[df["period"] == period]
        lines += [
            f"## {period}（{a} ~ {b}）",
            "",
            "| 组 | 变体 | 信号数 | 信号日 | 20日均收益 | 20日胜率 | 20日超额 | t | 40日均收益 | MAE20 | 模拟胜率 | 模拟均收益 | 止盈占比 | 止损占比 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for _, r in sub.iterrows():
            t = r.get("ex_20_t")
            lines.append(
                f"| {r['group']} | {r['variant']} | {int(r['n'])} | {int(r['n_dates'])} | "
                f"{_fmt_pct(r.get('mean_20'))} | {_fmt_pct(r.get('hit_20'), 1)} | {_fmt_pct(r.get('ex_20'))} | "
                f"{'—' if t is None or not np.isfinite(t) else f'{t:.2f}'} | {_fmt_pct(r.get('mean_40'))} | "
                f"{_fmt_pct(r.get('mae_20'))} | {_fmt_pct(r.get('sim_win'), 1)} | {_fmt_pct(r.get('sim_avg'))} | "
                f"{_fmt_pct(r.get('sim_target_share'), 1)} | {_fmt_pct(r.get('sim_stop_share'), 1)} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def render_extras(yearly: pd.DataFrame, basket: pd.DataFrame) -> str:
    lines = [
        "## 等权篮子（每信号持有20日，空仓=0）",
        "",
        "| 变体 | 区间 | 年化 | 最大回撤 | Sharpe | 持仓天数占比 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, r in basket.iterrows():
        sh = r.get("sharpe")
        lines.append(
            f"| {r['variant']} | {r['period']} | {_fmt_pct(r.get('cagr'))} | {_fmt_pct(r.get('max_dd'))} | "
            f"{'—' if sh is None or not np.isfinite(sh) else f'{sh:.2f}'} | {_fmt_pct(r.get('exposure'), 1)} |"
        )
    lines += ["", "## 分年（20日）", "", "| 变体 | 年 | 信号数 | 20日均收益 | 20日胜率 | 20日超额 |", "|---|---:|---:|---:|---:|---:|"]
    for _, r in yearly.iterrows():
        lines.append(
            f"| {r['variant']} | {int(r['year'])} | {int(r['n'])} | {_fmt_pct(r.get('mean_20'))} | "
            f"{_fmt_pct(r.get('hit_20'), 1)} | {_fmt_pct(r.get('ex_20'))} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    src = args.listing_dir or args.ohlcv_dir
    out_dir = args.out_dir or (src / "b1235_backtest")
    out_dir.mkdir(parents=True, exist_ok=True)

    panel, ohlcvs = load_panel(args, out_dir)
    end = args.end or panel.index.max().strftime("%Y-%m-%d")
    split = pd.Timestamp(args.split_date)
    periods = {
        "IS": (args.start, (split - pd.Timedelta(days=1)).strftime("%Y-%m-%d")),
        "OOS": (args.split_date, end),
    }
    print(f"panel rows={len(panel)} codes={panel['code'].nunique()} {panel.index.min().date()}..{end}", flush=True)

    df = run_grid(panel, ohlcvs, periods=periods, cooldown=args.cooldown)
    df.to_csv(out_dir / "b1235_ablation.csv", index=False, encoding="utf-8-sig")

    base = variant_grid()[:2]
    key = base + proposal_grid()
    yearly = yearly_table(panel, key[1:], cooldown=args.cooldown, start=args.start)
    yearly.to_csv(out_dir / "b1235_yearly.csv", index=False, encoding="utf-8-sig")
    basket = basket_table(panel, ohlcvs, key, cooldown=args.cooldown, periods=periods)
    basket.to_csv(out_dir / "b1235_basket.csv", index=False, encoding="utf-8-sig")

    md = render_markdown(df, periods) + render_extras(yearly, basket)
    (out_dir / "b1235_ablation.md").write_text(md, encoding="utf-8")
    (out_dir / "b1235_meta.json").write_text(
        json.dumps({"periods": periods, "cooldown": args.cooldown, "codes": sorted(ohlcvs)}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"[OK] {out_dir / 'b1235_ablation.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
