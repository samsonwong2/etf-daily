#!/usr/bin/env python3
"""Export pool HTML: hybrid_ma_adx regimes + fixed stage buy/sell overlays.

Daily usage (via wrapper)::

    AS_OF=2026-08-06 PY=~/etf-daily-output/python/envs/py312/bin/python \\
      ./scripts/daily_hybrid_ma_adx_html.sh

Or directly::

    $PY decision_pack/scripts/plot_hybrid_ma_adx_stage_pool.py \\
      --end-date 2026-08-06 --start-date 2026-04-01 \\
      --html-out-dir ~/etf-daily-output/temp/plotly_outputs/20260806all_hybrid_ma_adx

Rules (causal day-by-day; rule set fixed from SH588710 in-sample research):
  - Regime: hybrid_ma_adx (no look-ahead)
  - Up:   (绿牛 ∨ 金叉 ∨ 红上MA20) → 红破MA10 (also exit on leave-up)
  - Down: 红近低 → tp10 / sl6 / hold≤10
  - Range: (红近低 ∨ 绿牛) → tp8 / sl5 / hold≤10

Open positions at window end are marked 持仓未平 (not a sell triangle).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from decision_pack.src.regime_transition_live_snapshot import (  # noqa: E402
    load_live_snapshot,
)
from decision_pack.src.regime_transition_plot import (  # noqa: E402
    VOL_REGIME_CLUSTER,
    VOL_REGIME_EXTREME,
    load_validation_csvs,
)
from decision_pack.src.symbol_trend_board import load_cluster_mapping_codes  # noqa: E402
from runtime_paths import CLUSTER_MAPPING_SELECTED_TXT, PLOTLY_OUTPUTS_DIR  # noqa: E402

DEFAULT_VALIDATION_DIR = (
    _PROJECT_ROOT
    / "workspace/decision_packs/20260720/regime_transition_validation_q90_to0720"
)
DEFAULT_START = "2026-04-01"
WARMUP_CALENDAR_DAYS = 220


def _load_ev():
    path = _PROJECT_ROOT / "decision_pack/scripts/eval_causal_regime_switch_pool.py"
    spec = importlib.util.spec_from_file_location("ev_hybrid_pool", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_pm():
    path = _PROJECT_ROOT / "decision_pack/scripts/plot_regime_transition_example.py"
    spec = importlib.util.spec_from_file_location("pm_hybrid_pool", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _segments(regime: np.ndarray, dates: np.ndarray, close: np.ndarray) -> pd.DataFrame:
    labs = list(regime)
    rows: list[dict[str, Any]] = []
    s = 0
    for i in range(1, len(labs) + 1):
        if i == len(labs) or labs[i] != labs[s]:
            rows.append(
                dict(
                    regime=labs[s],
                    start=dates[s],
                    end=dates[i - 1],
                    n_bars=i - s,
                    ret=float(close[i - 1] / close[s] - 1),
                )
            )
            s = i
    return pd.DataFrame(rows)


def _prepare_px(ev, ohlcv: pd.DataFrame) -> pd.DataFrame:
    px = ev.build_px(ohlcv)
    px["as_of"] = pd.to_datetime(px["as_of"]).dt.normalize()
    close = px["$close"].astype(float)
    high = px["$high"].astype(float)
    low = px["$low"].astype(float)
    open_ = px["$open"].astype(float)
    px["MA5"] = close.rolling(5, min_periods=5).mean()
    px["MA10"] = close.rolling(10, min_periods=10).mean()
    px["is_green"] = close > open_
    px["is_red"] = close < open_
    px["ma_bull"] = px["MA20"] > px["MA60"]
    px["above_ma10"] = close > px["MA10"]
    px["above_ma20"] = close > px["MA20"]
    px["golden"] = (px["MA5"] > px["MA20"]) & (px["MA5"].shift(1) <= px["MA20"].shift(1))
    roll_lo = low.rolling(20, min_periods=5).min()
    roll_hi = high.rolling(20, min_periods=5).max()
    px["range_pos"] = (close - roll_lo) / (roll_hi - roll_lo).replace(0, np.nan)
    px["regime"] = ev.METHODS["hybrid_ma_adx"](px)
    return px


def _simulate_stage_trades(px_w: pd.DataFrame) -> tuple[list[dict], dict, dict | None]:
    """Return (closed events, stats, open_mtm_or_None). Never emits 期末 sells."""
    close_w = px_w["$close"].to_numpy(float)
    high_w = px_w["$high"].to_numpy(float)
    bh = float(close_w[-1] / close_w[0] - 1) if len(close_w) > 1 else 0.0
    dates = px_w.as_of.dt.strftime("%Y-%m-%d").to_numpy()
    regime = px_w.regime.to_numpy()

    g = px_w.is_green.to_numpy(bool)
    bull = px_w.ma_bull.to_numpy(bool)
    gold = px_w.golden.fillna(False).to_numpy(bool)
    red = px_w.is_red.to_numpy(bool)
    a20 = px_w.above_ma20.to_numpy(bool)
    a10 = px_w.above_ma10.to_numpy(bool)
    rp = px_w.range_pos.to_numpy(float)

    up_buy = ((g & bull) | gold | (red & a20)) & (regime == "up")
    up_why = np.array([""] * len(px_w), dtype=object)
    m = np.zeros(len(px_w), dtype=bool)
    x = g & bull & ~m & (regime == "up")
    up_why[x] = "绿牛"
    m |= x
    x = gold & ~m & (regime == "up")
    up_why[x] = "金叉"
    m |= x
    x = red & a20 & ~m & (regime == "up")
    up_why[x] = "红上MA20"
    m |= x

    dn_buy = red & np.isfinite(rp) & (rp <= 0.2) & (regime == "down")
    dn_why = np.where(dn_buy, "红近低", "")

    rg_buy = ((red & np.isfinite(rp) & (rp <= 0.25)) | (g & bull)) & (regime == "range")
    rg_why = np.array([""] * len(px_w), dtype=object)
    m = np.zeros(len(px_w), dtype=bool)
    x = red & np.isfinite(rp) & (rp <= 0.25) & ~m & (regime == "range")
    rg_why[x] = "红近低"
    m |= x
    x = g & bull & ~m & (regime == "range")
    rg_why[x] = "绿牛"
    m |= x

    cash = 1.0
    pos = 0.0
    entry = None
    ei = None
    peak = None
    entry_rg = ""
    aev: list[dict] = []
    open_mtm = None

    for i in range(len(px_w)):
        rg = regime[i]
        c = close_w[i]
        h = high_w[i]
        if pos == 0:
            if rg == "up" and up_buy[i]:
                entry = c
                ei = i
                peak = h
                entry_rg = "up"
                pos = cash / c
                cash = 0.0
                aev.append(
                    dict(side="BUY", date=dates[i], px=c, why=f"up:{up_why[i]}", regime="up")
                )
            elif rg == "down" and dn_buy[i]:
                entry = c
                ei = i
                peak = h
                entry_rg = "down"
                pos = cash / c
                cash = 0.0
                aev.append(
                    dict(side="BUY", date=dates[i], px=c, why=f"down:{dn_why[i]}", regime="down")
                )
            elif rg == "range" and rg_buy[i]:
                entry = c
                ei = i
                peak = h
                entry_rg = "range"
                pos = cash / c
                cash = 0.0
                aev.append(
                    dict(side="BUY", date=dates[i], px=c, why=f"range:{rg_why[i]}", regime="range")
                )
            continue

        peak = max(peak, h)
        hold = i - ei
        if hold <= 0:
            continue
        why = None
        px_e = c
        if entry_rg == "up":
            if red[i] and (not a10[i]):
                why = "红破MA10"
            elif rg != "up":
                why = f"离上涨→{rg}"
        elif entry_rg == "down":
            if (h / entry - 1) >= 0.10:
                why = "tp10"
                px_e = entry * 1.10
            elif (c / entry - 1) <= -0.06:
                why = "sl6"
            elif hold >= 10:
                why = "hold10"
        else:
            if (h / entry - 1) >= 0.08:
                why = "tp8"
                px_e = entry * 1.08
            elif (c / entry - 1) <= -0.05:
                why = "sl5"
            elif hold >= 10:
                why = "hold10"
        if why:
            cash = pos * px_e
            aev.append(
                dict(
                    side="SELL",
                    date=dates[i],
                    px=px_e,
                    why=why,
                    entry_date=dates[ei],
                    fwd_ret=px_e / entry - 1,
                    regime=entry_rg,
                )
            )
            pos = 0.0
            entry = ei = peak = None
            entry_rg = ""

    if pos > 0:
        open_mtm = dict(
            entry_date=dates[ei],
            entry_px=float(entry),
            regime=entry_rg,
            last_date=dates[-1],
            last_px=float(close_w[-1]),
            unrealized=float(close_w[-1] / entry - 1),
        )
        equity = cash + pos * close_w[-1]
    else:
        equity = cash

    sells = [e for e in aev if e["side"] == "SELL"]
    stats = dict(
        compound=float(equity - 1),
        bh=bh,
        edge=float(equity - 1) - bh,
        n=len(sells),
        win=float(np.mean([e["fwd_ret"] > 0 for e in sells])) if sells else float("nan"),
        open_pos=bool(open_mtm),
        open_unrealized=(open_mtm["unrealized"] if open_mtm else None),
    )
    return aev, stats, open_mtm


def _annotate_figure(
    *,
    pm,
    fig: go.Figure,
    ohlcv: pd.DataFrame,
    seg_df: pd.DataFrame,
    aev: list[dict],
    open_mtm: dict | None,
    stats: dict,
    start_date: str,
    end_date: str,
) -> go.Figure:
    vol_legend = {VOL_REGIME_CLUSTER, VOL_REGIME_EXTREME}
    fig.data = tuple(t for t in fig.data if getattr(t, "name", None) not in vol_legend)

    ohlcv_w = ohlcv[
        (pd.to_datetime(ohlcv["datetime"]) >= pd.Timestamp(start_date))
        & (pd.to_datetime(ohlcv["datetime"]) <= pd.Timestamp(end_date))
    ].copy()
    ohlcv_w["as_of"] = pd.to_datetime(ohlcv_w["datetime"]).dt.normalize()
    pxmap = ohlcv_w.set_index("as_of")
    lo = float(pd.to_numeric(ohlcv_w["$low"], errors="coerce").min())
    hi = float(pd.to_numeric(ohlcv_w["$high"], errors="coerce").max())
    span = max(hi - lo, 1e-6)
    buy_off = span * 0.035

    color_map = {
        "up": "rgba(18,161,80,0.12)",
        "down": "rgba(214,39,40,0.10)",
        "range": "rgba(148,148,148,0.14)",
    }
    shapes = [
        dict(
            type="rect",
            xref="x",
            yref="y",
            x0=srow["start"],
            x1=srow["end"],
            y0=lo - span * 0.02,
            y1=hi + span * 0.05,
            fillcolor=color_map.get(srow["regime"], "rgba(0,0,0,0.05)"),
            line=dict(width=0),
            layer="below",
        )
        for _, srow in seg_df.iterrows()
    ]
    fig.update_layout(shapes=shapes)
    for name, color in (
        ("上涨态", "rgba(18,161,80,0.55)"),
        ("下跌态", "rgba(214,39,40,0.55)"),
        ("横盘态", "rgba(120,120,120,0.55)"),
    ):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color=color, symbol="square"),
                name=name,
            ),
            row=1,
            col=1,
        )

    def y_buy(d: str) -> float | None:
        ts = pd.Timestamp(d)
        return None if ts not in pxmap.index else float(pxmap.loc[ts, "$low"]) - buy_off

    def y_sell(d: str) -> float | None:
        ts = pd.Timestamp(d)
        return None if ts not in pxmap.index else float(pxmap.loc[ts, "$high"]) + buy_off

    style = {
        "up": dict(bc="#12a150", bl="#064d1f", sc="#d62728", sl="#6b1010", bn="上涨买", sn="上涨卖"),
        "down": dict(bc="#2ca02c", bl="#145214", sc="#ff7f0e", sl="#8c4500", bn="下跌买", sn="下跌卖"),
        "range": dict(bc="#1f77b4", bl="#0b3d5c", sc="#9467bd", sl="#4b2e66", bn="横盘买", sn="横盘卖"),
    }
    tr = pd.DataFrame(aev)
    if not tr.empty:
        for rg, st in style.items():
            sub = tr[tr.regime.astype(str) == rg]
            if sub.empty:
                continue
            bx, by, bt = [], [], []
            for _, r in sub[sub.side == "BUY"].iterrows():
                y = y_buy(r["date"])
                if y is None:
                    continue
                bx.append(r["date"])
                by.append(y)
                bt.append(f"买 {r['date']}<br>{r['why']}<br>价={float(r['px']):.3f}")
            sx, sy, stxt = [], [], []
            for _, r in sub[sub.side == "SELL"].iterrows():
                y = y_sell(r["date"])
                if y is None:
                    continue
                ret = r.get("fwd_ret", np.nan)
                ret_s = f"<br>收益={float(ret):+.1%}" if pd.notna(ret) else ""
                sx.append(r["date"])
                sy.append(y)
                stxt.append(f"卖 {r['date']}<br>{r['why']}<br>价={float(r['px']):.3f}{ret_s}")
            if bx:
                fig.add_trace(
                    go.Scatter(
                        x=bx,
                        y=by,
                        mode="markers+text",
                        name=st["bn"],
                        text=["B"] * len(bx),
                        textposition="bottom center",
                        textfont=dict(size=11, color=st["bl"], family="Arial Black"),
                        marker=dict(
                            symbol="triangle-up",
                            size=14,
                            color=st["bc"],
                            line=dict(width=1.2, color=st["bl"]),
                        ),
                        hovertext=bt,
                        hoverinfo="text",
                    ),
                    row=1,
                    col=1,
                )
            if sx:
                fig.add_trace(
                    go.Scatter(
                        x=sx,
                        y=sy,
                        mode="markers+text",
                        name=st["sn"],
                        text=["S"] * len(sx),
                        textposition="top center",
                        textfont=dict(size=11, color=st["sl"], family="Arial Black"),
                        marker=dict(
                            symbol="triangle-down",
                            size=14,
                            color=st["sc"],
                            line=dict(width=1.2, color=st["sl"]),
                        ),
                        hovertext=stxt,
                        hoverinfo="text",
                    ),
                    row=1,
                    col=1,
                )

    if open_mtm is not None:
        fig.add_trace(
            go.Scatter(
                x=[open_mtm["last_date"]],
                y=[float(open_mtm["last_px"])],
                mode="markers+text",
                name="持仓未平",
                text=["持"],
                textposition="top center",
                textfont=dict(size=11, color="#333", family="Arial Black"),
                marker=dict(
                    symbol="diamond",
                    size=12,
                    color="#f0ad4e",
                    line=dict(width=1.2, color="#8a6d3b"),
                ),
                hovertext=[
                    f"持仓未平<br>买入 {open_mtm['entry_date']} @ {open_mtm['entry_px']:.3f}"
                    f"<br>状态={open_mtm['regime']}<br>浮盈={open_mtm['unrealized']:+.1%}"
                ],
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )

    old = fig.layout.title.text if fig.layout.title else ""
    seg_txt = " | ".join(
        f"{r.regime} {str(r.start)[5:]}~{str(r.end)[5:]} {r.ret:+.0%}" for _, r in seg_df.iterrows()
    )
    win_s = f"{stats['win']:.0%}" if stats["win"] == stats["win"] else "n/a"
    open_note = ""
    if open_mtm:
        open_note = (
            f"｜窗口截止仍持仓（自{open_mtm['entry_date']}，"
            f"浮盈{open_mtm['unrealized']:+.1%}，非卖点）"
        )
    note = (
        f"<br><span style='font-size:12px;color:#333'>"
        f"<b>hybrid_ma_adx</b> 状态底色 + 分阶段规则（{start_date}→{end_date}）"
        f"<br>涨：绿牛∨金叉∨红上MA20 → 红破MA10｜跌：红近低 → tp10/sl6/h10｜盘：红近低∨绿牛 → tp8/sl5/h10"
        f"<br>已平仓口径（含未平仓按收盘盯市）{stats['compound']:+.1%}｜持有 {stats['bh']:+.1%}｜"
        f"edge {stats['edge']:+.1%}｜已平仓笔数 n={stats['n']} 胜率 {win_s}{open_note}"
        f"<br><span style='font-size:11px;color:#555'>{seg_txt}</span></span>"
    )
    fig.update_layout(title=dict(text=(old or "") + note))
    return fig


def process_one(
    *,
    code: str,
    start_date: str,
    end_date: str,
    ev,
    pm,
    oos: pd.DataFrame,
    rev: pd.DataFrame,
    evt: pd.DataFrame,
    name_map: dict,
    out_dir: Path,
    live_snapshot: pd.DataFrame | None = None,
) -> dict:
    ohlcv = pm.load_qlib_ohlcv(
        code,
        start_date,
        end_date,
        init_qlib=False,
        lookback_calendar_days=WARMUP_CALENDAR_DAYS,
        clip_to_window=False,
    )
    if ohlcv is None or ohlcv.empty:
        raise ValueError("empty ohlcv")

    px = _prepare_px(ev, ohlcv)
    px_w = px[(px.as_of >= start_date) & (px.as_of <= end_date)].reset_index(drop=True)
    if len(px_w) < 5:
        raise ValueError(f"too few bars {len(px_w)}")

    close_w = px_w["$close"].to_numpy(float)
    dates = px_w.as_of.dt.strftime("%Y-%m-%d").to_numpy()
    seg_df = _segments(px_w.regime.to_numpy(), dates, close_w)
    aev, stats, open_mtm = _simulate_stage_trades(px_w)

    fig, _overlay, name = pm.build_figure(
        code=code,
        start_date=start_date,
        end_date=end_date,
        oos=oos,
        rev=rev,
        evt=evt,
        provider_uri=None,
        display_name=None,
        name_map=name_map,
        template="plotly_white",
        init_qlib=False,
        kline_plotter_cls=None,
        live_snapshot=live_snapshot,
    )
    fig = _annotate_figure(
        pm=pm,
        fig=fig,
        ohlcv=ohlcv,
        seg_df=seg_df,
        aev=aev,
        open_mtm=open_mtm,
        stats=stats,
        start_date=start_date,
        end_date=end_date,
    )

    start_tag = pd.Timestamp(start_date).strftime("%Y%m%d")
    end_tag = pd.Timestamp(end_date).strftime("%Y%m%d")
    safe = pm.sanitize_name_for_filename(name)
    out = out_dir / f"regime_transition_{code}_{safe}_{start_tag}_{end_tag}_hybrid_ma_adx.html"
    fig.write_html(str(out), include_plotlyjs="cdn")

    # per-code trade dump
    trades_dir = out_dir / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(aev).to_csv(trades_dir / f"{code}_trades.csv", index=False, encoding="utf-8-sig")
    seg_df.to_csv(trades_dir / f"{code}_segments.csv", index=False, encoding="utf-8-sig")

    return dict(
        code=code,
        name=name,
        out=out.name,
        bars=len(px_w),
        compound=stats["compound"],
        bh=stats["bh"],
        edge=stats["edge"],
        n_closed=stats["n"],
        win=stats["win"],
        open_pos=stats["open_pos"],
        open_unrealized=stats["open_unrealized"],
        n_seg=len(seg_df),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code", action="append", default=None, help="optional; repeatable. Default: pool")
    p.add_argument("--start-date", default=DEFAULT_START)
    p.add_argument("--end-date", required=True, help="AS_OF / plot end date YYYY-MM-DD")
    p.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    p.add_argument("--cluster-mapping", type=Path, default=None)
    p.add_argument(
        "--html-out-dir",
        type=Path,
        default=None,
        help="default: ~/etf-daily-output/temp/plotly_outputs/{YYYYMMDD}all_hybrid_ma_adx",
    )
    p.add_argument("--live-snapshot", type=Path, default=None)
    p.add_argument("--continue-on-error", action="store_true", default=True)
    p.add_argument("--fail-fast", action="store_true", help="abort on first error")
    p.add_argument("--max-codes", type=int, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    end = pd.Timestamp(args.end_date).strftime("%Y-%m-%d")
    start = pd.Timestamp(args.start_date).strftime("%Y-%m-%d")
    end_tag = pd.Timestamp(end).strftime("%Y%m%d")

    out_dir = args.html_out_dir
    if out_dir is None:
        out_dir = PLOTLY_OUTPUTS_DIR / f"{end_tag}all_hybrid_ma_adx"
    out_dir = out_dir if out_dir.is_absolute() else _PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    val_dir = args.validation_dir if args.validation_dir.is_absolute() else _PROJECT_ROOT / args.validation_dir
    mapping = Path(args.cluster_mapping) if args.cluster_mapping else CLUSTER_MAPPING_SELECTED_TXT
    if not mapping.is_absolute():
        mapping = _PROJECT_ROOT / mapping

    codes = [c.upper() for c in (args.code or [])] or list(load_cluster_mapping_codes(mapping))
    if args.max_codes:
        codes = codes[: args.max_codes]

    ev = _load_ev()
    pm = _load_pm()
    pm._init_qlib(None)
    oos, rev, evt = load_validation_csvs(val_dir)
    name_map = pm.name_map_from_oos(oos)
    live = None
    if args.live_snapshot is not None:
        snap_path = args.live_snapshot
        if not snap_path.is_absolute():
            snap_path = _PROJECT_ROOT / snap_path
        live = load_live_snapshot(snap_path, as_of=end)

    print(f"codes={len(codes)} window={start}→{end} out={out_dir}")
    rows: list[dict] = []
    errors: list[dict] = []
    fail_fast = bool(args.fail_fast)

    for i, code in enumerate(codes, 1):
        try:
            r = process_one(
                code=code,
                start_date=start,
                end_date=end,
                ev=ev,
                pm=pm,
                oos=oos,
                rev=rev,
                evt=evt,
                name_map=name_map,
                out_dir=out_dir,
                live_snapshot=live,
            )
            rows.append(r)
            op = "OPEN" if r["open_pos"] else "flat"
            print(f"[{i}/{len(codes)}] OK {code} closed={r['n_closed']} {op} edge={r['edge']:+.1%}")
        except Exception as e:
            errors.append(dict(code=code, error=str(e)))
            print(f"[{i}/{len(codes)}] FAIL {code}: {e}")
            if fail_fast:
                traceback.print_exc()
                return 1
            if not args.continue_on_error:
                traceback.print_exc()
                return 1

    pd.DataFrame(rows).to_csv(out_dir / "batch_summary.csv", index=False, encoding="utf-8-sig")
    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "batch_errors.csv", index=False, encoding="utf-8-sig")

    readme = f"""# hybrid_ma_adx 分阶段 HTML

> 窗口 **{start} → {end}** · 成功 {len(rows)}/{len(codes)} · 失败 {len(errors)}

## 规则（因果执行；规则集固定）

- 分段：`hybrid_ma_adx`（无前瞻）
- 上涨：绿牛 ∨ 金叉 ∨ 红上MA20 → 红破 MA10（离上涨亦卖）
- 下跌：红近低 → tp10 / sl6 / ≤10 日
- 横盘：红近低 ∨ 绿牛 → tp8 / sl5 / ≤10 日

## 标注

- 不画「期末卖出」；窗口截止仍持仓标菱形「持仓未平」
- 明细：`trades/` · 汇总：`batch_summary.csv`

## 每日命令

```bash
AS_OF={end} PY=~/etf-daily-output/python/envs/py312/bin/python \\
  ./scripts/daily_hybrid_ma_adx_html.sh
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"DONE ok={len(rows)} fail={len(errors)} out={out_dir}")
    return 0 if not errors else 0  # continue-on-error default soft-ok


if __name__ == "__main__":
    raise SystemExit(main())
