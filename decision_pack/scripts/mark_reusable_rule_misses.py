#!/usr/bin/env python3
"""Score reusable MA/gap rules vs oracle pivots; mark misses on HTML.

Hard rules from documents/SH518880_SH512530_事后买卖点关键数字对照.md
「两边一致（可复用）」——触下/折价、短尺溢价只作诊断，不进硬 AND
（否则买侧触下仅 ~20%，会漏掉大半谷底）。

Buy hard AND:
  close < MA20 AND MA5 < MA20 AND gap5 ≤ 0 AND gap20 ≤ 0 AND not S1(短尺触上)

Sell hard AND:
  close > MA20 AND MA5 > MA20 AND gap5 ≥ 0 AND gap20 ≥ 0

Missed oracle pivots get star markers on fig1–11 (caught keep green/red triangles).

Example::

    PYTHONPATH=. python decision_pack/scripts/mark_reusable_rule_misses.py \\
      --code SH518880 SH512530
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

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runtime_paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from decision_pack.src.fair_path_band_signals import (  # noqa: E402
    build_quiet_discount_frame,
    find_regime_html,
    load_ohlcv_from_regime_html,
)
from decision_pack.src.regime_transition_plot import (  # noqa: E402
    compute_volatility_regime_frame,
)
from decision_pack.src.vol_need_return_board import (  # noqa: E402
    ANCHOR_ERP_ANN,
    fair_need_ann_series,
    gap_realized_minus_need,
)

DEFAULT_LISTING = PLOTLY_OUTPUTS_DIR / "20260824_from_listing"


def _load_overlay():
    path = _SCRIPTS / "overlay_oracle_pivots_on_regime_html.py"
    spec = importlib.util.spec_from_file_location("overlay_oracle_pivots", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code", nargs="+", required=True)
    p.add_argument("--listing-dir", type=Path, default=DEFAULT_LISTING)
    p.add_argument("--anchor-code", default="SH510300")
    return p.parse_args(argv)


def _resolve(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def build_frame(html: Path, listing: Path, anchor_code: str) -> pd.DataFrame:
    close, ohlcv = load_ohlcv_from_regime_html(html)
    anchor_html = find_regime_html(listing, anchor_code)
    if anchor_html is None:
        raise FileNotFoundError(f"anchor {anchor_code} missing under {listing}")
    close_a, _ = load_ohlcv_from_regime_html(anchor_html)
    vf_a = compute_volatility_regime_frame(close_a)
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
    frame["below_ma20"] = close < frame["ma20"]
    frame["above_ma20"] = close > frame["ma20"]
    frame["ma5_lt_ma20"] = frame["ma5"] < frame["ma20"]
    frame["ma5_gt_ma20"] = frame["ma5"] > frame["ma20"]
    frame["s1_hi"] = (
        frame["fig9_touch_hi"].fillna(False) | frame["fig10_touch_hi"].fillna(False)
    )
    frame["s_lo"] = (
        frame["fig9_touch_lo"].fillna(False) | frame["fig10_touch_lo"].fillna(False)
    )
    frame["fig7_discount_5"] = frame["fig7_gap"] <= -0.05
    frame["fig7_premium_0"] = frame["fig7_gap"] >= 0
    # short-scale premium (fig10 or fig11 gap >= 0)
    frame["short_premium"] = (frame["fig10_gap"] >= 0) | (frame["fig11_gap"] >= 0)
    return frame, ohlcv


def score_pivots(piv: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, r in piv.iterrows():
        d = pd.Timestamp(r["date"]).normalize()
        kind = str(r["kind"]).lower()
        if d not in frame.index:
            continue
        f = frame.loc[d]
        below = bool(f["below_ma20"]) if pd.notna(f["below_ma20"]) else False
        above = bool(f["above_ma20"]) if pd.notna(f["above_ma20"]) else False
        ma5_lt = bool(f["ma5_lt_ma20"]) if pd.notna(f["ma5_lt_ma20"]) else False
        ma5_gt = bool(f["ma5_gt_ma20"]) if pd.notna(f["ma5_gt_ma20"]) else False
        gap5 = float(f["gap_5d"]) if pd.notna(f["gap_5d"]) else np.nan
        gap20 = float(f["gap_20d"]) if pd.notna(f["gap_20d"]) else np.nan
        s1 = bool(f["s1_hi"])
        s_lo = bool(f["s_lo"])
        disc = bool(f["fig7_discount_5"]) if pd.notna(f.get("fig7_discount_5")) else False
        prem0 = bool(f["fig7_premium_0"]) if pd.notna(f.get("fig7_premium_0")) else False
        short_prem = bool(f["short_premium"]) if pd.notna(f.get("short_premium")) else False

        if kind == "buy":
            hard = (
                below
                and ma5_lt
                and np.isfinite(gap5)
                and gap5 <= 0
                and np.isfinite(gap20)
                and gap20 <= 0
                and (not s1)
            )
            fail = []
            if not below:
                fail.append("非收盘<MA20")
            if not ma5_lt:
                fail.append("非MA5<MA20")
            if not (np.isfinite(gap5) and gap5 <= 0):
                fail.append("非gap5≤0")
            if not (np.isfinite(gap20) and gap20 <= 0):
                fail.append("非gap20≤0")
            if s1:
                fail.append("短尺触上S1")
            soft_ok = s_lo or disc
        else:
            hard = (
                above
                and ma5_gt
                and np.isfinite(gap5)
                and gap5 >= 0
                and np.isfinite(gap20)
                and gap20 >= 0
            )
            fail = []
            if not above:
                fail.append("非收盘>MA20")
            if not ma5_gt:
                fail.append("非MA5>MA20")
            if not (np.isfinite(gap5) and gap5 >= 0):
                fail.append("非gap5≥0")
            if not (np.isfinite(gap20) and gap20 >= 0):
                fail.append("非gap20≥0")
            soft_ok = s1 or short_prem or prem0

        rows.append(
            {
                "kind": kind,
                "date": d.strftime("%Y-%m-%d"),
                "px": float(r["px"]) if pd.notna(r.get("px")) else float(f["px"]),
                "hard_hit": bool(hard),
                "missed": not bool(hard),
                "fail_reasons": ";".join(fail),
                "soft_touch_or_premium": bool(soft_ok),
                "below_ma20": below,
                "above_ma20": above,
                "ma5_lt_ma20": ma5_lt,
                "ma5_gt_ma20": ma5_gt,
                "gap5": None if not np.isfinite(gap5) else gap5,
                "gap20": None if not np.isfinite(gap20) else gap20,
                "s1_hi": s1,
                "s_lo": s_lo,
                "fig7_discount_5": disc,
            }
        )
    return pd.DataFrame(rows)


def build_figure_with_misses(
    *,
    overlay_mod: Any,
    traces: list[dict[str, Any]],
    layout: dict[str, Any],
    pivots: pd.DataFrame,
    scored: pd.DataFrame,
    ohlcv: pd.DataFrame,
) -> go.Figure:
    """Green/red triangles = all oracle; gold stars = missed by hard rule."""
    fig = overlay_mod.build_figure(
        traces=traces, layout=layout, pivots=pivots, ohlcv=ohlcv
    )
    # build_figure already wrote adaptive figure; we need to add miss traces
    # Re-implement: call internals by rebuilding from scored
    # Easier: mutate fig after build_figure
    px_by_date = ohlcv.copy()
    px_by_date.index = pd.DatetimeIndex(pd.to_datetime(px_by_date.index)).normalize()
    missed_buy = scored[(scored["kind"] == "buy") & scored["missed"]]
    missed_sell = scored[(scored["kind"] == "sell") & scored["missed"]]

    def _xy_miss(df: pd.DataFrame, y_rail: float, side: str, panel: str):
        xs, ys, hs = [], [], []
        for _, r in df.iterrows():
            d = pd.Timestamp(r["date"]).normalize()
            if d not in px_by_date.index:
                continue
            close_px = float(px_by_date.loc[d, "$close"])
            xs.append(d)
            ys.append(y_rail)
            hs.append(
                f"{panel} 规则漏{side}<br>date={d.strftime('%Y-%m-%d')}"
                f"<br>close={close_px:.4f}<br>fail={r['fail_reasons']}"
            )
        return xs, ys, hs

    # Collect data list from fig
    data = list(fig.data)
    layout_d = fig.layout.to_plotly_json() if hasattr(fig.layout, "to_plotly_json") else dict(fig.layout)

    # Need rails again — use overlay helpers on current traces (as dicts)
    tr_dicts = [t.to_plotly_json() if hasattr(t, "to_plotly_json") else dict(t) for t in data]

    def add_miss_pair(panel, yaxis, xaxis, ra, rb, showlegend):
        bx, by, bh = _xy_miss(missed_buy, ra, "买", panel)
        sx, sy, sh = _xy_miss(missed_sell, rb, "卖", panel)
        data.append(
            go.Scatter(
                x=bx,
                y=by,
                mode="markers",
                name=f"漏买({len(bx)})" if showlegend else f"{panel}漏买",
                text=bh,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    symbol="star",
                    size=14 if showlegend else 11,
                    color="#ff7f0e",
                    line=dict(width=1, color="#a34e00"),
                ),
                yaxis=yaxis,
                xaxis=xaxis,
                showlegend=showlegend,
                legendgroup="miss_buy",
            )
        )
        data.append(
            go.Scatter(
                x=sx,
                y=sy,
                mode="markers",
                name=f"漏卖({len(sx)})" if showlegend else f"{panel}漏卖",
                text=sh,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    symbol="star",
                    size=14 if showlegend else 11,
                    color="#9467bd",
                    line=dict(width=1, color="#5a3d7a"),
                ),
                yaxis=yaxis,
                xaxis=xaxis,
                showlegend=showlegend,
                legendgroup="miss_sell",
            )
        )

    rail_above, rail_below = overlay_mod._rail_ys(ohlcv)
    add_miss_pair("图1", "y", "x", rail_above, rail_below, True)
    for panel, yaxis, xaxis in overlay_mod.FIG_PANEL_AXES:
        ra, rb = overlay_mod._panel_rail_ys(tr_dicts, yaxis, layout_d)
        add_miss_pair(panel, yaxis, xaxis, ra, rb, False)

    # Title note
    n_mb = int(missed_buy.shape[0])
    n_ms = int(missed_sell.shape[0])
    n_b = int((scored["kind"] == "buy").sum())
    n_s = int((scored["kind"] == "sell").sum())
    note = (
        f"｜可复用硬规则漏检：买{n_mb}/{n_b} 卖{n_ms}/{n_s}"
        "（橙星=漏买，紫星=漏卖；绿/红三角=全部事后点）"
    )
    title = layout_d.get("title")
    if isinstance(title, dict):
        text = str(title.get("text") or "")
        title = {
            **title,
            "text": text + f"<br><span style='font-size:12px;color:#333'>{note}</span>",
        }
        layout_d = {**layout_d, "title": title}
    elif isinstance(title, str):
        layout_d = {**layout_d, "title": title + note}
    else:
        layout_d = {**layout_d, "title": note}

    return go.Figure(data=data, layout=layout_d)


def process_one(code: str, listing: Path, anchor_code: str, overlay_mod: Any) -> dict[str, Any]:
    code = code.upper()
    html = find_regime_html(listing, code)
    if html is None:
        raise FileNotFoundError(f"no HTML for {code} under {listing}")
    out_dir = listing / f"fig3_11_bt_{code}" / "oracle_fig1_11_common_markers"
    piv_path = out_dir / "oracle_pivots.csv"
    if not piv_path.exists():
        raise FileNotFoundError(
            f"missing {piv_path}; run run_oracle_fig1_11.py --code {code} first"
        )
    piv = pd.read_csv(piv_path)
    frame, ohlcv = build_frame(html, listing, anchor_code)
    scored = score_pivots(piv, frame)
    scored.to_csv(out_dir / "reusable_rule_score.csv", index=False, encoding="utf-8-sig")

    n_buy = int((scored["kind"] == "buy").sum())
    n_sell = int((scored["kind"] == "sell").sum())
    miss_buy = scored[(scored["kind"] == "buy") & scored["missed"]]
    miss_sell = scored[(scored["kind"] == "sell") & scored["missed"]]
    hit_buy = n_buy - len(miss_buy)
    hit_sell = n_sell - len(miss_sell)

    # fail reason tallies
    def tallies(df: pd.DataFrame) -> dict[str, int]:
        c: dict[str, int] = {}
        for s in df["fail_reasons"]:
            for part in str(s).split(";"):
                if part:
                    c[part] = c.get(part, 0) + 1
        return c

    summary = {
        "code": code,
        "html": str(html),
        "rule": {
            "buy_hard": "close<MA20 & MA5<MA20 & gap5≤0 & gap20≤0 & not S1",
            "sell_hard": "close>MA20 & MA5>MA20 & gap5≥0 & gap20≥0",
            "soft_not_required": "买侧触下/折价；卖侧短尺触上/溢价",
        },
        "n_oracle_buy": n_buy,
        "n_oracle_sell": n_sell,
        "n_hit_buy": hit_buy,
        "n_hit_sell": hit_sell,
        "n_miss_buy": int(len(miss_buy)),
        "n_miss_sell": int(len(miss_sell)),
        "miss_buy_rate": float(len(miss_buy) / n_buy) if n_buy else None,
        "miss_sell_rate": float(len(miss_sell) / n_sell) if n_sell else None,
        "miss_buy_fail_tallies": tallies(miss_buy),
        "miss_sell_fail_tallies": tallies(miss_sell),
        "miss_buy_dates": miss_buy["date"].tolist(),
        "miss_sell_dates": miss_sell["date"].tolist(),
    }
    (out_dir / "reusable_rule_miss_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # HTML overlay with miss stars
    html_txt = html.read_text(encoding="utf-8")
    traces = overlay_mod._extract_plotly_traces(html_txt)
    layout = overlay_mod.extract_layout(html_txt)
    fig = build_figure_with_misses(
        overlay_mod=overlay_mod,
        traces=traces,
        layout=layout,
        pivots=piv,
        scored=scored,
        ohlcv=ohlcv,
    )
    out_html = html.with_name(html.stem + "_oracle_pivots_rule_miss.html")
    from decision_pack.src.plotly_html_autoscale import write_adaptive_html

    write_adaptive_html(fig, out_html, include_plotlyjs="cdn")
    summary["out_html"] = str(out_html)
    print(
        f"[{code}] buy hit {hit_buy}/{n_buy} miss {len(miss_buy)} "
        f"({100*len(miss_buy)/n_buy:.0f}%); "
        f"sell hit {hit_sell}/{n_sell} miss {len(miss_sell)} "
        f"({100*len(miss_sell)/n_sell:.0f}%); wrote {out_html.name}"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    listing = _resolve(args.listing_dir)
    overlay_mod = _load_overlay()
    all_sum: list[dict[str, Any]] = []
    for code in args.code:
        all_sum.append(process_one(code, listing, args.anchor_code, overlay_mod))

    # Combined md next to documents or listing
    lines = [
        "# 可复用硬规则 vs 事后买卖点：漏检统计",
        "",
        "> 硬规则来自 `documents/SH518880_SH512530_事后买卖点关键数字对照.md`「两边一致」。",
        "> **买**：收盘&lt;MA20 ∧ MA5&lt;MA20 ∧ gap5≤0 ∧ gap20≤0 ∧ 非短尺触上(S1)。",
        "> **卖**：收盘&gt;MA20 ∧ MA5&gt;MA20 ∧ gap5≥0 ∧ gap20≥0。",
        "> 「触下/折价」「短尺触上/溢价」只作画像抬升，**不进硬 AND**（否则买侧触下仅~20%会漏大半）。",
        "",
        "| 标的 | 事后买 | 规则命中买 | **漏买** | 事后卖 | 规则命中卖 | **漏卖** |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in all_sum:
        lines.append(
            f"| {s['code']} | {s['n_oracle_buy']} | {s['n_hit_buy']} | "
            f"**{s['n_miss_buy']}** ({100*s['miss_buy_rate']:.0f}%) | "
            f"{s['n_oracle_sell']} | {s['n_hit_sell']} | "
            f"**{s['n_miss_sell']}** ({100*s['miss_sell_rate']:.0f}%) |"
        )
    lines += ["", "## 漏买原因计数", ""]
    for s in all_sum:
        lines.append(f"### {s['code']}")
        lines.append("")
        if not s["miss_buy_fail_tallies"]:
            lines.append("- （无漏买）")
        else:
            for k, v in sorted(s["miss_buy_fail_tallies"].items(), key=lambda kv: -kv[1]):
                lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append(f"漏买日期: {', '.join(s['miss_buy_dates']) or '—'}")
        lines.append("")
    lines += ["## 漏卖原因计数", ""]
    for s in all_sum:
        lines.append(f"### {s['code']}")
        lines.append("")
        if not s["miss_sell_fail_tallies"]:
            lines.append("- （无漏卖）")
        else:
            for k, v in sorted(s["miss_sell_fail_tallies"].items(), key=lambda kv: -kv[1]):
                lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append(f"漏卖日期: {', '.join(s['miss_sell_dates']) or '—'}")
        lines.append("")
    lines += [
        "## 叠图 HTML（橙星=漏买，紫星=漏卖；绿/红三角=全部事后点）",
        "",
    ]
    for s in all_sum:
        lines.append(f"- `{s['out_html']}`")
    lines.append("")

    md_path = listing / "reusable_rule_miss_SH518880_SH512530.md"
    # if only these two codes, use that name; else generic
    codes = [s["code"] for s in all_sum]
    if set(codes) == {"SH518880", "SH512530"}:
        md_path = (
            _PROJECT_ROOT
            / "documents"
            / "SH518880_SH512530_可复用硬规则漏检.md"
        )
    else:
        md_path = listing / f"reusable_rule_miss_{'_'.join(codes)}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
