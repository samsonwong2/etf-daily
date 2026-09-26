#!/usr/bin/env python3
"""Overlay fig9 gap≤-5% buy / reclaim-path sell on an existing regime HTML.

Buy: close vs fig9 (6m) path gap ≤ -5%, only when flat.
Sell: close back on/above the fig9 path while long.
Markers are drawn on fig1–11 (green ▲ / red ▼). Production traces stay.

Example::

    PYTHONPATH=. python src/etf_daily/plots/overlay_fig9_gap_rule_on_regime_html.py \\
      --html ~/etf-daily-output/temp/plotly_outputs/20260909_from_listing/regime_transition_SH512700_银行ETF南方_20170726_20260909_adaptive.html
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from etf_daily.lib.fair_path_band_signals import (  # noqa: E402
    _extract_plotly_traces,
    compute_multi_scale_frame,
    load_ohlcv_from_regime_html,
)
from etf_daily.lib.plotly_html_autoscale import write_adaptive_html  # noqa: E402

DEFAULT_HTML = (
    PLOTLY_OUTPUTS_DIR / "20260909_from_listing"
    / "regime_transition_SH512700_银行ETF南方_20170726_20260909_adaptive.html"
)
DEFAULT_GAP = -0.05


def _load_oracle_mod():
    path = _PROJECT_ROOT / "src/etf_daily/plots/overlay_oracle_pivots_on_regime_html.py"
    spec = importlib.util.spec_from_file_location("oracle_overlay", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fig9_gap_trades(close: pd.Series, *, gap_th: float = DEFAULT_GAP) -> pd.DataFrame:
    """Causal paired trades: buy when fig9 gap ≤ threshold, sell on path reclaim."""
    ms = compute_multi_scale_frame(close)
    px = pd.to_numeric(ms["px"], errors="coerce")
    path = pd.to_numeric(ms["fig9_path"], errors="coerce")
    gap = pd.to_numeric(ms["fig9_gap"], errors="coerce")
    rows: list[dict[str, Any]] = []
    long = False
    for dt in ms.index:
        p = float(path.loc[dt]) if pd.notna(path.loc[dt]) else float("nan")
        g = float(gap.loc[dt]) if pd.notna(gap.loc[dt]) else float("nan")
        c = float(px.loc[dt]) if pd.notna(px.loc[dt]) else float("nan")
        if not (p > 0 and pd.notna(g) and pd.notna(c) and c > 0):
            continue
        ds = pd.Timestamp(dt).strftime("%Y-%m-%d")
        if not long and g <= gap_th:
            rows.append(
                {
                    "kind": "buy",
                    "date": ds,
                    "px": c,
                    "path": p,
                    "gap": g,
                }
            )
            long = True
        elif long and c >= p:
            rows.append(
                {
                    "kind": "sell",
                    "date": ds,
                    "px": c,
                    "path": p,
                    "gap": g,
                }
            )
            long = False
    return pd.DataFrame(rows)


def overlay_fig9_rule(
    *,
    traces: list[dict[str, Any]],
    layout: dict[str, Any],
    pivots: pd.DataFrame,
    ohlcv: pd.DataFrame,
    gap_th: float,
    title_note: str | None = None,
    glide_windows: pd.DataFrame | None = None,
) -> go.Figure:
    om = _load_oracle_mod()
    kept = list(traces)  # keep production triangles / ED / RU
    px_by_date = ohlcv.copy()
    px_by_date.index = pd.DatetimeIndex(pd.to_datetime(px_by_date.index)).normalize()
    buys = (
        pivots[pivots["kind"].astype(str).str.lower() == "buy"]
        if not pivots.empty
        else pivots
    )
    sells = (
        pivots[pivots["kind"].astype(str).str.lower() == "sell"]
        if not pivots.empty
        else pivots
    )
    n_buy = int(len(buys))
    n_sell = int(len(sells))
    gap_pct = f"{100.0 * float(gap_th):.0f}"

    def _xy(
        df: pd.DataFrame, y_rail: float, *, side: str, panel: str
    ) -> tuple[list[pd.Timestamp], list[float], list[str]]:
        xs: list[pd.Timestamp] = []
        ys: list[float] = []
        hovers: list[str] = []
        if df is None or df.empty:
            return xs, ys, hovers
        for _, r in df.iterrows():
            d = pd.Timestamp(r["date"]).normalize()
            if d not in px_by_date.index:
                continue
            close_px = float(px_by_date.loc[d, "$close"])
            xs.append(d)
            ys.append(y_rail)
            gap_s = ""
            if "gap" in r and pd.notna(r.get("gap")):
                gap_s = f"<br>fig9_gap={float(r['gap']):+.2%}"
            path_s = ""
            if "path" in r and pd.notna(r.get("path")):
                path_s = f"<br>fig9_path={float(r['path']):.4f}"
            hovers.append(
                f"{panel} 图9规则{side}<br>"
                f"date={d.strftime('%Y-%m-%d')}<br>close={close_px:.4f}{gap_s}{path_s}"
            )
        return xs, ys, hovers

    def _add_pair(
        *,
        panel: str,
        yaxis: str,
        xaxis: str,
        rail_above: float,
        rail_below: float,
        showlegend: bool,
    ) -> None:
        bx, by, bh = _xy(buys, rail_above, side="买", panel=panel)
        sx, sy, sh = _xy(sells, rail_below, side="卖", panel=panel)
        kept.append(
            dict(
                type="scatter",
                x=bx,
                y=by,
                mode="markers",
                name=f"图9折价买({n_buy})" if showlegend else f"{panel}图9折价买",
                text=bh,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    symbol="triangle-up",
                    size=11 if showlegend else 10,
                    color="#2ca02c",
                    line=dict(width=0.5, color="#145214"),
                ),
                yaxis=yaxis,
                xaxis=xaxis,
                showlegend=showlegend,
                legendgroup="fig9_gap_buy",
            )
        )
        kept.append(
            dict(
                type="scatter",
                x=sx,
                y=sy,
                mode="markers",
                name=f"图9回路径卖({n_sell})" if showlegend else f"{panel}图9回路径卖",
                text=sh,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    symbol="triangle-down",
                    size=11 if showlegend else 10,
                    color="#d62728",
                    line=dict(width=0.5, color="#7f1416"),
                ),
                yaxis=yaxis,
                xaxis=xaxis,
                showlegend=showlegend,
                legendgroup="fig9_gap_sell",
            )
        )

    ra, rb = om._panel_rail_ys(kept, "y", layout)
    _add_pair(
        panel="图1",
        yaxis="y",
        xaxis="x",
        rail_above=ra,
        rail_below=rb,
        showlegend=True,
    )
    for panel, yaxis, xaxis in om.FIG_PANEL_AXES:
        ra, rb = om._panel_rail_ys(kept, yaxis, layout)
        _add_pair(
            panel=panel,
            yaxis=yaxis,
            xaxis=xaxis,
            rail_above=ra,
            rail_below=rb,
            showlegend=False,
        )

    note = title_note or (
        f"｜图1–11 图9规则三角：买=折价≤{gap_pct}% 卖=站回6个月路径"
        f"（叠加在原图上，非生产 hold_up）｜买{n_buy} 卖{n_sell}"
    )

    if glide_windows is not None and not glide_windows.empty:
        spans = layout.get("shapes") or []
        spans = [s for s in spans if s.get("name") != "fig9_glide_warn"]
        added = 0
        for _, r in glide_windows.iterrows():
            x0 = pd.Timestamp(r["start"])
            x1 = pd.Timestamp(r["end"])
            spans.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=x0,
                    x1=x1,
                    y0=0,
                    y1=1,
                    fillcolor="#ff7f0e",
                    opacity=0.12,
                    line=dict(width=0),
                    name="fig9_glide_warn",
                )
            )
            added += 1
        layout = {**layout, "shapes": spans}
        note = note + f"｜橙底=贴轨阴跌({added}段，仅警示不改买卖)"

    title = layout.get("title")
    if isinstance(title, dict):
        text = str(title.get("text") or "")
        text = re.sub(
            r"<br><span style='font-size:12px;color:#333'>.*?图9规则.*?</span>",
            "",
            text,
        )
        layout = {
            **layout,
            "title": {
                **title,
                "text": text + f"<br><span style='font-size:12px;color:#333'>{note}</span>",
            },
        }
    elif isinstance(title, str):
        layout = {**layout, "title": title + note}
    else:
        layout = {**layout, "title": note}

    print(f"[INFO] annotated fig1–11 with buy={n_buy} sell={n_sell}")
    return go.Figure(data=kept, layout=layout)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--html", type=Path, default=DEFAULT_HTML)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--gap", type=float, default=DEFAULT_GAP)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    om = _load_oracle_mod()
    html_path = Path(args.html)
    if not html_path.is_absolute():
        html_path = _PROJECT_ROOT / html_path
    if not html_path.exists():
        raise FileNotFoundError(html_path)
    out_path = args.out
    if out_path is None:
        out_path = html_path.with_name(html_path.stem + "_fig9_gap5.html")
    else:
        out_path = Path(out_path)
        if not out_path.is_absolute():
            out_path = _PROJECT_ROOT / out_path

    close, ohlcv = load_ohlcv_from_regime_html(html_path)
    pivots = fig9_gap_trades(close, gap_th=float(args.gap))
    n_buy = int((pivots["kind"] == "buy").sum()) if not pivots.empty else 0
    n_sell = int((pivots["kind"] == "sell").sum()) if not pivots.empty else 0
    print(f"[INFO] fig9 gap<={args.gap:.0%}  buy={n_buy} sell={n_sell}")
    csv_path = out_path.with_suffix(".csv")
    if not pivots.empty:
        pivots.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] wrote {csv_path}")
        print(pivots.to_string(index=False))

    html_txt = html_path.read_text(encoding="utf-8")
    traces = _extract_plotly_traces(html_txt)
    layout = om.extract_layout(html_txt)
    fig = overlay_fig9_rule(
        traces=traces,
        layout=layout,
        pivots=pivots,
        ohlcv=ohlcv,
        gap_th=float(args.gap),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_adaptive_html(fig, out_path, include_plotlyjs="cdn")
    print(f"[OK] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
