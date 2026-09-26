#!/usr/bin/env python3
"""Reusable MA/gap rules → 极值点 + 持仓状态机（可选第二刀）.

Base (推荐第一步):
  买候选 = 硬买规则 ∧ 近 L 日因果局部最低（只用过去，不含未来）
  卖候选 = 硬卖规则 ∧ 近 L 日因果局部最高
  状态机: 空仓才买、持仓才卖；信号日收盘确认 → 次日开盘/收盘成交（默认次日收盘）

硬规则（两边一致可复用）:
  买: close<MA20 ∧ MA5<MA20 ∧ gap5≤0 ∧ gap20≤0 ∧ ¬S1
  卖: close>MA20 ∧ MA5>MA20 ∧ gap5≥0 ∧ gap20≥0

第二刀（可选，--second-knife）:
  none     — 只用极值+状态机
  gap      — 买再要求 gap20≤−2%；卖再要求 gap20≥+2%
  touch    — 买再要求 短尺触下∨图7折价≤−5%；卖再要求 S1
  gap_touch — gap 与 touch 同时要求

Example::

    PYTHONPATH=. python src/etf_daily/scripts/run_reusable_extrema_state_machine.py \\
      --code SH518880 SH512530 --second-knife none gap touch
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
_SCRIPTS = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from etf_daily.lib.fair_path_band_signals import (  # noqa: E402
    build_quiet_discount_frame,
    find_regime_html,
    load_ohlcv_from_regime_html,
)
from etf_daily.lib.plotly_html_autoscale import write_adaptive_html  # noqa: E402
from etf_daily.lib.regime_transition_plot import (  # noqa: E402
    compute_volatility_regime_frame,
)
from etf_daily.lib.vol_need_return_board import (  # noqa: E402
    ANCHOR_ERP_ANN,
    fair_need_ann_series,
    gap_realized_minus_need,
)

DEFAULT_LISTING = PLOTLY_OUTPUTS_DIR / "20260824_from_listing"
KNIVES = ("none", "gap", "touch", "gap_touch")


def _load_overlay():
    path = _SCRIPTS / "overlay_oracle_pivots_on_regime_html.py"
    spec = importlib.util.spec_from_file_location("overlay_oracle_pivots", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--code", nargs="+", required=True)
    p.add_argument("--listing-dir", type=Path, default=DEFAULT_LISTING)
    p.add_argument("--anchor-code", default="SH510300")
    p.add_argument("--lookback", type=int, default=10, help="causal local extremum window")
    p.add_argument(
        "--second-knife",
        nargs="+",
        default=["none", "gap", "touch"],
        choices=list(KNIVES),
        help="one or more second-knife modes to run",
    )
    p.add_argument("--skip-html", action="store_true")
    p.add_argument(
        "--html-knife",
        default="none",
        choices=list(KNIVES),
        help="which knife variant to draw on HTML (default: none = 极值+状态机)",
    )
    return p.parse_args(argv)


def _resolve(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def causal_local_extrema(close: pd.Series, lookback: int) -> tuple[pd.Series, pd.Series]:
    """True when today's close is min/max of [t-lookback, t] (no future bars)."""
    px = pd.to_numeric(close, errors="coerce").astype(float).to_numpy()
    n = len(px)
    is_min = np.zeros(n, dtype=bool)
    is_max = np.zeros(n, dtype=bool)
    for i in range(lookback, n):
        w = px[i - lookback : i + 1]
        if not np.isfinite(px[i]):
            continue
        if px[i] == np.nanmin(w):
            is_min[i] = True
        if px[i] == np.nanmax(w):
            is_max[i] = True
    idx = close.index
    return pd.Series(is_min, index=idx), pd.Series(is_max, index=idx)


def build_feature_frame_from_series(
    close: pd.Series,
    ohlcv: pd.DataFrame,
    anchor_close: pd.Series,
    *,
    lookback: int,
) -> pd.DataFrame:
    """Feature frame from close series (no HTML). Same columns as ``build_feature_frame``."""
    vf_a = compute_volatility_regime_frame(anchor_close)
    vf_a["as_of"] = pd.to_datetime(vf_a["as_of"]).dt.normalize()
    vf_a = vf_a.set_index("as_of")
    frame = build_quiet_discount_frame(
        close,
        ohlcv,
        rv20_anchor=pd.to_numeric(vf_a["rv20"], errors="coerce"),
        rv5_anchor=pd.to_numeric(vf_a["rv5"], errors="coerce"),
    )
    vol = compute_volatility_regime_frame(close)
    vol["as_of"] = pd.to_datetime(vol["as_of"]).dt.normalize()
    vol = vol.set_index("as_of").reindex(close.index)
    need5 = fair_need_ann_series(
        pd.to_numeric(vol["rv5"], errors="coerce"),
        pd.to_numeric(vf_a["rv5"], errors="coerce").reindex(close.index),
        erp_ann=ANCHOR_ERP_ANN,
    )
    g5 = gap_realized_minus_need(close, need5, bars=5)
    frame["gap_5d"] = pd.to_numeric(g5["gap"], errors="coerce").reindex(close.index)
    frame["ma5"] = close.rolling(5).mean()
    frame["ma20"] = close.rolling(20).mean()
    frame["s1"] = (
        frame["fig9_touch_hi"].fillna(False) | frame["fig10_touch_hi"].fillna(False)
    )
    frame["s_lo"] = (
        frame["fig9_touch_lo"].fillna(False) | frame["fig10_touch_lo"].fillna(False)
    )
    frame["disc5"] = frame["fig7_gap"] <= -0.05
    locmin, locmax = causal_local_extrema(close, lookback)
    frame["locmin"] = locmin
    frame["locmax"] = locmax

    # hard reusable state
    frame["hard_buy"] = (
        (close < frame["ma20"])
        & (frame["ma5"] < frame["ma20"])
        & (frame["gap_5d"] <= 0)
        & (frame["gap_20d"] <= 0)
        & (~frame["s1"])
    ).fillna(False)
    frame["hard_sell"] = (
        (close > frame["ma20"])
        & (frame["ma5"] > frame["ma20"])
        & (frame["gap_5d"] >= 0)
        & (frame["gap_20d"] >= 0)
    ).fillna(False)

    # second-knife predicates
    frame["knife_gap_buy"] = frame["gap_20d"] <= -0.02
    frame["knife_gap_sell"] = frame["gap_20d"] >= 0.02
    frame["knife_touch_buy"] = frame["s_lo"] | frame["disc5"].fillna(False)
    frame["knife_touch_sell"] = frame["s1"]

    return frame


def build_feature_frame(
    html: Path, listing: Path, anchor_code: str, *, lookback: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    close, ohlcv = load_ohlcv_from_regime_html(html)
    anchor_html = find_regime_html(listing, anchor_code)
    if anchor_html is None:
        raise FileNotFoundError(f"anchor {anchor_code} missing under {listing}")
    close_a, _ = load_ohlcv_from_regime_html(anchor_html)
    frame = build_feature_frame_from_series(
        close, ohlcv, close_a, lookback=lookback
    )
    return frame, ohlcv


def apply_second_knife(
    frame: pd.DataFrame, knife: str
) -> tuple[pd.Series, pd.Series]:
    buy = frame["hard_buy"] & frame["locmin"]
    sell = frame["hard_sell"] & frame["locmax"]
    if knife == "none":
        pass
    elif knife == "gap":
        buy = buy & frame["knife_gap_buy"].fillna(False)
        sell = sell & frame["knife_gap_sell"].fillna(False)
    elif knife == "touch":
        buy = buy & frame["knife_touch_buy"].fillna(False)
        sell = sell & frame["knife_touch_sell"].fillna(False)
    elif knife == "gap_touch":
        buy = buy & frame["knife_gap_buy"].fillna(False) & frame["knife_touch_buy"].fillna(
            False
        )
        sell = sell & frame["knife_gap_sell"].fillna(False) & frame[
            "knife_touch_sell"
        ].fillna(False)
    else:
        raise ValueError(knife)
    return buy.fillna(False), sell.fillna(False)


def run_state_machine(
    frame: pd.DataFrame,
    buy_cand: pd.Series,
    sell_cand: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Signal at close → fill next bar close. Long-only."""
    close = frame["px"].astype(float)
    idx = frame.index
    n = len(frame)
    buy_sig = np.zeros(n, dtype=bool)
    sell_sig = np.zeros(n, dtype=bool)
    trades: list[dict[str, Any]] = []
    in_pos = False
    entry_i: int | None = None
    entry_sig_i: int | None = None

    for i in range(n - 1):
        if not in_pos:
            if bool(buy_cand.iloc[i]):
                buy_sig[i] = True
                entry_sig_i = i
                entry_i = i + 1  # next-bar fill
                in_pos = True
        else:
            if bool(sell_cand.iloc[i]):
                sell_sig[i] = True
                assert entry_i is not None and entry_sig_i is not None
                exit_i = i + 1
                if exit_i >= n:
                    break
                ep = float(close.iloc[entry_i])
                xp = float(close.iloc[exit_i])
                trades.append(
                    {
                        "signal_buy": idx[entry_sig_i].strftime("%Y-%m-%d"),
                        "entry": idx[entry_i].strftime("%Y-%m-%d"),
                        "signal_sell": idx[i].strftime("%Y-%m-%d"),
                        "exit": idx[exit_i].strftime("%Y-%m-%d"),
                        "entry_px": ep,
                        "exit_px": xp,
                        "ret": xp / ep - 1.0,
                        "hold_days": int(exit_i - entry_i),
                    }
                )
                in_pos = False
                entry_i = None
                entry_sig_i = None

    # open trade mark if still in pos
    if in_pos and entry_i is not None and entry_sig_i is not None:
        trades.append(
            {
                "signal_buy": idx[entry_sig_i].strftime("%Y-%m-%d"),
                "entry": idx[entry_i].strftime("%Y-%m-%d"),
                "signal_sell": None,
                "exit": None,
                "entry_px": float(close.iloc[entry_i]),
                "exit_px": float(close.iloc[-1]),
                "ret": float(close.iloc[-1] / close.iloc[entry_i] - 1.0),
                "hold_days": int(n - 1 - entry_i),
                "open": True,
            }
        )

    tdf = pd.DataFrame(trades)
    return (
        tdf,
        pd.Series(buy_sig, index=idx),
        pd.Series(sell_sig, index=idx),
    )


def summarize_trades(tdf: pd.DataFrame) -> dict[str, Any]:
    if tdf.empty:
        closed = tdf
        n_open = 0
    else:
        is_open = (
            tdf["open"].fillna(False).astype(bool)
            if "open" in tdf.columns
            else pd.Series(False, index=tdf.index)
        )
        closed = tdf[~is_open & tdf["exit"].notna()].copy()
        n_open = int(is_open.sum())
    out: dict[str, Any] = {
        "n_trades": int(len(closed)),
        "n_open": n_open,
    }
    if closed.empty:
        out.update(win_rate=None, mean_ret=None, compound=None, mean_hold=None)
        return out
    out.update(
        win_rate=float((closed["ret"] > 0).mean()),
        mean_ret=float(closed["ret"].mean()),
        compound=float((1.0 + closed["ret"]).prod() - 1.0),
        mean_hold=float(closed["hold_days"].mean()),
    )
    return out


def oracle_overlap(
    buy_sig: pd.Series,
    sell_sig: pd.Series,
    piv: pd.DataFrame,
    *,
    tol: int = 5,
) -> dict[str, Any]:
    idx = buy_sig.index
    pos = {pd.Timestamp(d).normalize().strftime("%Y-%m-%d"): i for i, d in enumerate(idx)}
    ob = set(piv.loc[piv["kind"] == "buy", "date"].astype(str))
    os_ = set(piv.loc[piv["kind"] == "sell", "date"].astype(str))
    sb = set(pd.DatetimeIndex(idx[buy_sig]).strftime("%Y-%m-%d"))
    ss = set(pd.DatetimeIndex(idx[sell_sig]).strftime("%Y-%m-%d"))

    def near(a: set[str], b: set[str]) -> int:
        hit = 0
        for d in a:
            i = pos.get(d)
            if i is None:
                continue
            for k in range(max(0, i - tol), min(len(idx), i + tol + 1)):
                if idx[k].strftime("%Y-%m-%d") in b:
                    hit += 1
                    break
        return hit

    return {
        "n_sig_buy": len(sb),
        "n_sig_sell": len(ss),
        "n_oracle_buy": len(ob),
        "n_oracle_sell": len(os_),
        "buy_exact": len(sb & ob),
        "sell_exact": len(ss & os_),
        "buy_near_oracle_±5d": near(sb, ob),
        "sell_near_oracle_±5d": near(ss, os_),
        "oracle_buy_near_sig_±5d": near(ob, sb),
        "oracle_sell_near_sig_±5d": near(os_, ss),
    }


def overlay_signals_html(
    *,
    overlay_mod: Any,
    html_path: Path,
    ohlcv: pd.DataFrame,
    buy_sig: pd.Series,
    sell_sig: pd.Series,
    out_path: Path,
    title_note: str,
) -> Path:
    html_txt = html_path.read_text(encoding="utf-8")
    traces = overlay_mod._extract_plotly_traces(html_txt)
    layout = overlay_mod.extract_layout(html_txt)
    # Strip fig1 overlays + shapes like oracle overlay, but draw our signals
    kept = [tr for tr in traces if not overlay_mod._is_fig1_overlay_marker(tr)]
    shapes = layout.get("shapes")
    if isinstance(shapes, list):
        kept_shapes = [
            s
            for s in shapes
            if not (
                isinstance(s, dict)
                and str(s.get("yref") or "y") in {"y", "y1"}
                and str(s.get("xref") or "x") in {"x", "x1"}
            )
        ]
        layout = {**layout, "shapes": kept_shapes}

    px_by_date = ohlcv.copy()
    px_by_date.index = pd.DatetimeIndex(pd.to_datetime(px_by_date.index)).normalize()

    buys = pd.DatetimeIndex(buy_sig.index[buy_sig]).normalize()
    sells = pd.DatetimeIndex(sell_sig.index[sell_sig]).normalize()

    def xy(dates: pd.DatetimeIndex, y: float, side: str, panel: str):
        xs, ys, hs = [], [], []
        for d in dates:
            if d not in px_by_date.index:
                continue
            xs.append(d)
            ys.append(y)
            hs.append(
                f"{panel} 规则{side}<br>date={d.strftime('%Y-%m-%d')}"
                f"<br>close={float(px_by_date.loc[d,'$close']):.4f}"
            )
        return xs, ys, hs

    def add_pair(panel, yaxis, xaxis, ra, rb, showlegend):
        bx, by, bh = xy(buys, ra, "买", panel)
        sx, sy, sh = xy(sells, rb, "卖", panel)
        kept.append(
            dict(
                type="scatter",
                x=bx,
                y=by,
                mode="markers",
                name=f"规则买({len(bx)})" if showlegend else f"{panel}规则买",
                text=bh,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    symbol="triangle-up",
                    size=12 if showlegend else 10,
                    color="#1f77b4",
                    line=dict(width=0.5, color="#0b3d5c"),
                ),
                yaxis=yaxis,
                xaxis=xaxis,
                showlegend=showlegend,
                legendgroup="rule_buy",
            )
        )
        kept.append(
            dict(
                type="scatter",
                x=sx,
                y=sy,
                mode="markers",
                name=f"规则卖({len(sx)})" if showlegend else f"{panel}规则卖",
                text=sh,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    symbol="triangle-down",
                    size=12 if showlegend else 10,
                    color="#ff7f0e",
                    line=dict(width=0.5, color="#a34e00"),
                ),
                yaxis=yaxis,
                xaxis=xaxis,
                showlegend=showlegend,
                legendgroup="rule_sell",
            )
        )

    ra, rb = overlay_mod._rail_ys(ohlcv)
    add_pair("图1", "y", "x", ra, rb, True)
    for panel, yaxis, xaxis in overlay_mod.FIG_PANEL_AXES:
        pra, prb = overlay_mod._panel_rail_ys(kept, yaxis, layout)
        add_pair(panel, yaxis, xaxis, pra, prb, False)

    title = layout.get("title")
    note = f"<br><span style='font-size:12px;color:#333'>{title_note}</span>"
    if isinstance(title, dict):
        layout = {**layout, "title": {**title, "text": str(title.get("text") or "") + note}}
    elif isinstance(title, str):
        layout = {**layout, "title": title + note}
    else:
        layout = {**layout, "title": title_note}

    fig = go.Figure(data=kept, layout=layout)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_adaptive_html(fig, out_path, include_plotlyjs="cdn")
    return out_path


def process_code(
    code: str,
    *,
    listing: Path,
    anchor: str,
    lookback: int,
    knives: list[str],
    html_knife: str,
    skip_html: bool,
    overlay_mod: Any,
) -> dict[str, Any]:
    code = code.upper()
    html = find_regime_html(listing, code)
    if html is None:
        raise FileNotFoundError(f"no HTML for {code}")
    out_dir = listing / f"fig3_11_bt_{code}" / "reusable_extrema_sm"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame, ohlcv = build_feature_frame(html, listing, anchor, lookback=lookback)
    piv_path = (
        listing / f"fig3_11_bt_{code}" / "oracle_fig1_11_common_markers" / "oracle_pivots.csv"
    )
    piv = pd.read_csv(piv_path) if piv_path.exists() else pd.DataFrame(columns=["kind", "date"])

    knife_results: dict[str, Any] = {}
    html_paths: dict[str, str] = {}

    for knife in knives:
        buy_c, sell_c = apply_second_knife(frame, knife)
        # candidate counts before SM
        n_buy_cand = int(buy_c.sum())
        n_sell_cand = int(sell_c.sum())
        trades, buy_sig, sell_sig = run_state_machine(frame, buy_c, sell_c)
        trades_path = out_dir / f"trades_{knife}.csv"
        trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
        summ = summarize_trades(trades)
        ov = oracle_overlap(buy_sig, sell_sig, piv) if not piv.empty else {}
        knife_results[knife] = {
            "n_buy_candidates": n_buy_cand,
            "n_sell_candidates": n_sell_cand,
            "n_sig_buy": int(buy_sig.sum()),
            "n_sig_sell": int(sell_sig.sum()),
            "trades": summ,
            "oracle": ov,
            "trades_csv": str(trades_path),
        }
        # also dump signal dates
        sig_df = pd.DataFrame(
            {
                "date": frame.index.strftime("%Y-%m-%d"),
                "buy_cand": buy_c.astype(int).to_numpy(),
                "sell_cand": sell_c.astype(int).to_numpy(),
                "buy_sig": buy_sig.astype(int).to_numpy(),
                "sell_sig": sell_sig.astype(int).to_numpy(),
            }
        )
        sig_df.to_csv(out_dir / f"signals_{knife}.csv", index=False, encoding="utf-8-sig")

        if not skip_html and knife == html_knife:
            note = (
                f"｜极值+状态机"
                f"{'' if knife == 'none' else f'+第二刀[{knife}]'}"
                f"｜买{int(buy_sig.sum())} 卖{int(sell_sig.sum())}"
                f"｜L={lookback}｜蓝▲买 橙▼卖"
            )
            out_html = html.with_name(
                html.stem + f"_extrema_sm_{knife}.html"
            )
            overlay_signals_html(
                overlay_mod=overlay_mod,
                html_path=html,
                ohlcv=ohlcv,
                buy_sig=buy_sig,
                sell_sig=sell_sig,
                out_path=out_html,
                title_note=note,
            )
            html_paths[knife] = str(out_html)
            print(f"[{code}/{knife}] wrote {out_html.name}")

        t = summ
        cp = f"{t['compound']:.1%}" if t.get("compound") is not None else "n/a"
        print(
            f"[{code}/{knife}] cand buy/sell={n_buy_cand}/{n_sell_cand} "
            f"sig={int(buy_sig.sum())}/{int(sell_sig.sum())} "
            f"trades={t['n_trades']} compound={cp}"
        )

    summary = {
        "code": code,
        "html": str(html),
        "lookback": lookback,
        "rule": {
            "buy_hard": "close<MA20 & MA5<MA20 & gap5≤0 & gap20≤0 & not S1",
            "sell_hard": "close>MA20 & MA5>MA20 & gap5≥0 & gap20≥0",
            "timing": f"causal local min/max lookback={lookback}",
            "fill": "signal close → next bar close",
            "state_machine": "flat→buy, long→sell",
        },
        "knives": knife_results,
        "html_paths": html_paths,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def write_combined_md(results: list[dict[str, Any]], knives: list[str], path: Path) -> None:
    lines = [
        "# 可复用规则：极值点 + 持仓状态机",
        "",
        "> 第一步：硬规则 ∧ 近10日**因果**局部极值 + 空仓买/持仓卖。",
        "> 第二刀：`gap`（gap20≤−2% / ≥+2%）或 `touch`（买触下∨折价≤−5%；卖 S1）。",
        "> **不做**「状态日去重」当主手段。成交：信号日收盘 → 次日收盘。",
        "",
        "## 结果总表",
        "",
        "| 标的 | 第二刀 | 买候选 | 卖候选 | 状态机买/卖 | 成交数 | 胜率 | 复合 | 均持仓 | 事后买近邻±5d | 事后卖近邻±5d |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        code = r["code"]
        for knife in knives:
            k = r["knives"][knife]
            t = k["trades"]
            o = k.get("oracle") or {}
            wr = f"{t['win_rate']:.0%}" if t.get("win_rate") is not None else "—"
            cp = f"{t['compound']:.1%}" if t.get("compound") is not None else "—"
            mh = f"{t['mean_hold']:.1f}" if t.get("mean_hold") is not None else "—"
            lines.append(
                f"| {code} | {knife} | {k['n_buy_candidates']} | {k['n_sell_candidates']} | "
                f"{k['n_sig_buy']}/{k['n_sig_sell']} | {t['n_trades']} | {wr} | {cp} | {mh} | "
                f"{o.get('oracle_buy_near_sig_±5d', '—')}/{o.get('n_oracle_buy', '—')} | "
                f"{o.get('oracle_sell_near_sig_±5d', '—')}/{o.get('n_oracle_sell', '—')} |"
            )
    lines += [
        "",
        "## 规则备忘",
        "",
        "- **硬买**：收盘&lt;MA20 ∧ MA5&lt;MA20 ∧ gap5≤0 ∧ gap20≤0 ∧ 非短尺触上",
        "- **硬卖**：收盘&gt;MA20 ∧ MA5&gt;MA20 ∧ gap5≥0 ∧ gap20≥0",
        "- **极值**：今日收盘 = 过去 L 日（含今日）最低/最高（无未来）",
        "- **状态机**：空仓遇买候选 → 买信号；持仓遇卖候选 → 卖信号",
        "",
        "## 产物路径",
        "",
    ]
    for r in results:
        code = r["code"]
        lines.append(f"### {code}")
        lines.append("")
        base = f"~/etf-daily-output/temp/plotly_outputs/20260824_from_listing/fig3_11_bt_{code}/reusable_extrema_sm/"
        lines.append(f"- 目录：`{base}`")
        for knife, hp in (r.get("html_paths") or {}).items():
            lines.append(f"- HTML({knife})：`{hp}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    listing = _resolve(args.listing_dir)
    overlay_mod = None if args.skip_html else _load_overlay()
    results = []
    for code in args.code:
        results.append(
            process_code(
                code,
                listing=listing,
                anchor=args.anchor_code,
                lookback=args.lookback,
                knives=list(args.second_knife),
                html_knife=args.html_knife,
                skip_html=args.skip_html,
                overlay_mod=overlay_mod,
            )
        )
    # Also generate HTML for each knife if user asked multiple? Only html_knife by default.
    # Optionally overlay all knives when --html-knife matches each run — already done.

    # If user wants HTML for all knives, regenerate — keep simple: one html_knife.
    # Also write HTML for gap/touch if they are in knives and html_knife is none only one.
    # Extend: write HTML for every knife in second-knife when not skip.
    if not args.skip_html and overlay_mod is not None:
        for r in results:
            code = r["code"]
            html = Path(r["html"])
            frame, ohlcv = build_feature_frame(
                html, listing, args.anchor_code, lookback=args.lookback
            )
            for knife in args.second_knife:
                if knife == args.html_knife and knife in r.get("html_paths", {}):
                    continue  # already written
                buy_c, sell_c = apply_second_knife(frame, knife)
                _, buy_sig, sell_sig = run_state_machine(frame, buy_c, sell_c)
                note = (
                    f"｜极值+状态机"
                    f"{'' if knife == 'none' else f'+第二刀[{knife}]'}"
                    f"｜买{int(buy_sig.sum())} 卖{int(sell_sig.sum())}"
                    f"｜L={args.lookback}｜蓝▲买 橙▼卖"
                )
                out_html = html.with_name(html.stem + f"_extrema_sm_{knife}.html")
                overlay_signals_html(
                    overlay_mod=overlay_mod,
                    html_path=html,
                    ohlcv=ohlcv,
                    buy_sig=buy_sig,
                    sell_sig=sell_sig,
                    out_path=out_html,
                    title_note=note,
                )
                r.setdefault("html_paths", {})[knife] = str(out_html)
                print(f"[{code}/{knife}] wrote {out_html.name}")

    md = _PROJECT_ROOT / "documents" / "可复用规则_极值点持仓状态机.md"
    write_combined_md(results, list(args.second_knife), md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
