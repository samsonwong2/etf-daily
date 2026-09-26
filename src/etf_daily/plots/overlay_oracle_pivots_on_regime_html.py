#!/usr/bin/env python3
"""Rebuild SH518880 regime HTML: fig1 only oracle buy/sell triangles; keep fig2–11.

Reads an existing adaptive regime_transition HTML, strips every fig1 overlay
marker/label (quantile switch, hold_up BUY/SELL, ED/RU, confirmation diamonds,
regime legend squares, …) and fig1 region shapes, then draws the same oracle
pivots on **fig1–11**:

  - green triangle-up near the top of each panel = oracle buy
  - red triangle-down near the bottom of each panel = oracle sell

Oracle pivots default to the CSV from the look-ahead fig1–11 study; if missing,
they are recomputed with the same ±10 / +40d / 8% rule.

Example::

    PYTHONPATH=. python src/etf_daily/plots/overlay_oracle_pivots_on_regime_html.py \\
      --html ~/etf-daily-output/temp/plotly_outputs/20260824_from_listing/regime_transition_SH518880_黄金ETF华安_20130729_20260824_adaptive.html
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from etf_daily.lib.fair_path_band_signals import (  # noqa: E402
    _extract_plotly_traces,
    _plotly_x_to_index,
    load_ohlcv_from_regime_html,
)
from etf_daily.lib.plotly_html_autoscale import write_adaptive_html  # noqa: E402

DEFAULT_HTML = (
    PLOTLY_OUTPUTS_DIR / "20260824_from_listing"
    / "regime_transition_SH518880_黄金ETF华安_20130729_20260824_adaptive.html"
)
DEFAULT_PIVOTS = (
    PLOTLY_OUTPUTS_DIR / "20260824_from_listing"
    / "fig3_11_bt_SH518880"
    / "oracle_fig1_11_common_markers"
    / "oracle_pivots.csv"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--html", type=Path, default=DEFAULT_HTML)
    p.add_argument("--pivots", type=Path, default=DEFAULT_PIVOTS)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output HTML (default: <stem>_oracle_pivots.html beside source)",
    )
    return p.parse_args(argv)


def _extract_balanced(html: str, start: int) -> tuple[int, str]:
    opener = html[start]
    closer = "]" if opener == "[" else "}"
    depth = 0
    in_str = False
    esc = False
    i = start
    while i < len(html):
        c = html[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    return i + 1, html[start : i + 1]
        i += 1
    raise ValueError("unclosed JSON blob in HTML")


def _loads_js_json(raw: str) -> Any:
    fixed = re.sub(r"\bNaN\b", "null", raw)
    fixed = re.sub(r"\b-?Infinity\b", "null", fixed)
    return json.loads(fixed)


def extract_layout(html: str) -> dict[str, Any]:
    m = re.search(r'Plotly\.newPlot\(\s*"[^"]+"\s*,\s*(\[)', html)
    if not m:
        raise ValueError("Plotly.newPlot data not found")
    data_end, _ = _extract_balanced(html, m.start(1))
    rest = html[data_end:]
    m2 = re.match(r"\s*,\s*(\{)", rest)
    if not m2:
        raise ValueError("Plotly layout object not found")
    layout_start = data_end + m2.start(1)
    _, raw = _extract_balanced(html, layout_start)
    layout = _loads_js_json(raw)
    if not isinstance(layout, dict):
        raise TypeError("layout is not an object")
    return layout


def _is_fig1_overlay_marker(tr: dict[str, Any]) -> bool:
    """True for fig1 scatter overlays (triangles/diamonds/legend squares)."""
    yaxis = str(tr.get("yaxis") or "y")
    if yaxis not in {"y", "y1"}:
        return False
    typ = str(tr.get("type") or "scatter")
    if typ != "scatter":
        return False
    mode = str(tr.get("mode") or "")
    if "markers" in mode:
        return True
    # legend-only dummy squares sometimes use markers with null x/y
    marker = tr.get("marker")
    if isinstance(marker, dict) and marker.get("symbol") and "lines" not in mode:
        # keep MA line traces (mode=lines)
        if mode == "" and tr.get("x") is None:
            return True
    return False


def _recompute_oracle_pivots(close: pd.Series) -> pd.DataFrame:
    """Same rule as the look-ahead study: ±10 local extremum + 40d ±8%."""
    left = right = 10
    fwd = 40
    thr = 0.08
    px = pd.to_numeric(close, errors="coerce").astype(float).to_numpy()
    idx = pd.DatetimeIndex(pd.to_datetime(close.index)).normalize()
    n = len(px)
    is_buy = np.zeros(n, dtype=bool)
    is_sell = np.zeros(n, dtype=bool)
    for i in range(left, n - max(right, 1)):
        j1 = min(n, i + 1 + fwd)
        window = px[i - left : i + right + 1]
        if not np.isfinite(px[i]) or j1 <= i + 1:
            continue
        fut = px[i + 1 : j1]
        up = float(np.nanmax(fut) / px[i] - 1.0)
        dn = float(np.nanmin(fut) / px[i] - 1.0)
        if px[i] == np.nanmin(window) and up >= thr:
            is_buy[i] = True
        if px[i] == np.nanmax(window) and dn <= -thr:
            is_sell[i] = True

    def thin(mask: np.ndarray, gap: int = 8) -> np.ndarray:
        out = np.zeros_like(mask)
        last = -10**9
        for i, v in enumerate(mask):
            if v and i - last >= gap:
                out[i] = True
                last = i
        return out

    is_buy = thin(is_buy)
    is_sell = thin(is_sell)
    rows: list[dict[str, Any]] = []
    for i, d in enumerate(idx):
        if is_buy[i]:
            rows.append({"kind": "buy", "date": d.strftime("%Y-%m-%d"), "px": float(px[i])})
        if is_sell[i]:
            rows.append({"kind": "sell", "date": d.strftime("%Y-%m-%d"), "px": float(px[i])})
    return pd.DataFrame(rows)


def load_pivots(path: Path, close: pd.Series) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        need = {"kind", "date"}
        if not need.issubset(df.columns):
            raise ValueError(f"{path} missing columns {need}")
        return df
    print(f"[WARN] pivots missing at {path}; recomputing")
    return _recompute_oracle_pivots(close)


def _rail_ys(ohlcv: pd.DataFrame) -> tuple[float, float]:
    hi = float(pd.to_numeric(ohlcv["$high"], errors="coerce").max())
    lo = float(pd.to_numeric(ohlcv["$low"], errors="coerce").min())
    span = max(hi - lo, abs(hi) * 0.02, 1e-6)
    return hi + span * 0.10, lo - span * 0.10


# Primary y/x axis per subplot row for fig2–11 (skip overlay right axes).
# fig2 volume; fig3 vol; fig4 need20; fig5 need5; fig6 equity; fig7–11 paths.
FIG_PANEL_AXES: tuple[tuple[str, str, str], ...] = (
    ("图2", "y2", "x2"),
    ("图3", "y3", "x3"),
    ("图4", "y5", "x4"),
    ("图5", "y7", "x5"),
    ("图6", "y9", "x6"),
    ("图7", "y10", "x7"),
    ("图8", "y11", "x8"),
    ("图9", "y12", "x9"),
    ("图10", "y13", "x10"),
    ("图11", "y14", "x11"),
)


def _finite_vals(arr: Any) -> np.ndarray:
    if arr is None:
        return np.asarray([], dtype=float)
    a = np.asarray(arr, dtype=float).ravel()
    return a[np.isfinite(a)]


def _panel_rail_ys(
    traces: list[dict[str, Any]],
    yaxis: str,
    layout: dict[str, Any],
) -> tuple[float, float]:
    """Top/bottom marker rails for one panel from its trace data (or layout range)."""
    vals: list[float] = []
    for tr in traces:
        ykey = str(tr.get("yaxis") or "y")
        if ykey != yaxis:
            continue
        typ = str(tr.get("type") or "")
        if typ == "candlestick":
            for col in ("high", "low", "open", "close"):
                vals.extend(_finite_vals(tr.get(col)).tolist())
        elif typ == "bar":
            vals.extend(_finite_vals(tr.get("y")).tolist())
        else:
            vals.extend(_finite_vals(tr.get("y")).tolist())

    layout_key = "yaxis" if yaxis in {"y", "y1"} else f"yaxis{yaxis[1:]}"
    ylayout = layout.get(layout_key) if isinstance(layout.get(layout_key), dict) else {}
    rng = (ylayout or {}).get("range")
    if isinstance(rng, (list, tuple)) and len(rng) >= 2:
        try:
            r0, r1 = float(rng[0]), float(rng[1])
            if np.isfinite(r0) and np.isfinite(r1):
                vals.extend([r0, r1])
        except (TypeError, ValueError):
            pass

    if not vals:
        return 1.0, 0.0
    hi = float(np.nanmax(vals))
    lo = float(np.nanmin(vals))
    if not np.isfinite(hi) or not np.isfinite(lo):
        return 1.0, 0.0
    span = max(hi - lo, abs(hi) * 0.02 if hi != 0 else 0.02, 1e-9)
    # Slightly inside the panel so markers stay visible without blowing the scale.
    return hi + span * 0.06, lo - span * 0.06


def build_figure(
    *,
    traces: list[dict[str, Any]],
    layout: dict[str, Any],
    pivots: pd.DataFrame,
    ohlcv: pd.DataFrame,
) -> go.Figure:
    kept = [tr for tr in traces if not _is_fig1_overlay_marker(tr)]
    removed = len(traces) - len(kept)
    print(f"[INFO] kept {len(kept)} traces, removed {removed} fig1 overlay markers")

    px_by_date = ohlcv.copy()
    px_by_date.index = pd.DatetimeIndex(pd.to_datetime(px_by_date.index)).normalize()

    buys = pivots[pivots["kind"].astype(str).str.lower() == "buy"]
    sells = pivots[pivots["kind"].astype(str).str.lower() == "sell"]

    def _xy(
        df: pd.DataFrame, y_rail: float, *, side: str, panel: str
    ) -> tuple[list[pd.Timestamp], list[float], list[str]]:
        xs: list[pd.Timestamp] = []
        ys: list[float] = []
        hovers: list[str] = []
        for _, r in df.iterrows():
            d = pd.Timestamp(r["date"]).normalize()
            if d not in px_by_date.index:
                continue
            close_px = float(px_by_date.loc[d, "$close"])
            xs.append(d)
            ys.append(y_rail)
            extra = ""
            if "fwd_up_40" in r and pd.notna(r.get("fwd_up_40")):
                extra += f"<br>fwd_up_40={float(r['fwd_up_40']):.1%}"
            if "fwd_dn_40" in r and pd.notna(r.get("fwd_dn_40")):
                extra += f"<br>fwd_dn_40={float(r['fwd_dn_40']):.1%}"
            hovers.append(
                f"{panel} 事后{side}点<br>"
                f"date={d.strftime('%Y-%m-%d')}<br>close={close_px:.4f}{extra}"
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
    ) -> tuple[int, int]:
        bx, by, bh = _xy(buys, rail_above, side="买", panel=panel)
        sx, sy, sh = _xy(sells, rail_below, side="卖", panel=panel)
        kept.append(
            dict(
                type="scatter",
                x=bx,
                y=by,
                mode="markers",
                name=f"事后买({len(bx)})" if showlegend else f"{panel}事后买",
                text=bh,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    symbol="triangle-up",
                    size=10 if not showlegend else 12,
                    color="#2ca02c",
                    line=dict(width=0.5, color="#145214"),
                ),
                yaxis=yaxis,
                xaxis=xaxis,
                showlegend=showlegend,
                legendgroup="oracle_buy",
            )
        )
        kept.append(
            dict(
                type="scatter",
                x=sx,
                y=sy,
                mode="markers",
                name=f"事后卖({len(sx)})" if showlegend else f"{panel}事后卖",
                text=sh,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    symbol="triangle-down",
                    size=10 if not showlegend else 12,
                    color="#d62728",
                    line=dict(width=0.5, color="#7f1416"),
                ),
                yaxis=yaxis,
                xaxis=xaxis,
                showlegend=showlegend,
                legendgroup="oracle_sell",
            )
        )
        return len(bx), len(sx)

    # Fig1
    rail_above, rail_below = _rail_ys(ohlcv)
    n_buy, n_sell = _add_pair(
        panel="图1",
        yaxis="y",
        xaxis="x",
        rail_above=rail_above,
        rail_below=rail_below,
        showlegend=True,
    )

    # Fig2–11 on each row's primary y-axis
    for panel, yaxis, xaxis in FIG_PANEL_AXES:
        ra, rb = _panel_rail_ys(kept, yaxis, layout)
        _add_pair(
            panel=panel,
            yaxis=yaxis,
            xaxis=xaxis,
            rail_above=ra,
            rail_below=rb,
            showlegend=False,
        )
    print(f"[INFO] annotated fig1–11 with buy={n_buy} sell={n_sell} pivots")

    # Drop fig1 regime / vol shaded regions and divider rects (all yref=y shapes).
    shapes = layout.get("shapes")
    if isinstance(shapes, list) and shapes:
        kept_shapes = [
            s
            for s in shapes
            if not (
                isinstance(s, dict)
                and str(s.get("yref") or "y") in {"y", "y1"}
                and str(s.get("xref") or "x") in {"x", "x1"}
            )
        ]
        print(
            f"[INFO] removed {len(shapes) - len(kept_shapes)} fig1 region shapes "
            f"(kept {len(kept_shapes)})"
        )
        layout = {**layout, "shapes": kept_shapes}

    # Title note: these are look-ahead pivots, not production hold_up.
    title = layout.get("title")
    note = (
        "｜图1–11事后买卖三角（未来窗标定；非生产 hold_up）"
        f"｜买{n_buy} 卖{n_sell}"
    )
    if isinstance(title, dict):
        text = str(title.get("text") or "")
        # Replace older fig1-only note if present.
        if "事后买卖" in text:
            text = re.sub(
                r"<br><span style='font-size:12px;color:#333'>.*?事后.*?</span>",
                "",
                text,
            )
        title = {
            **title,
            "text": text + f"<br><span style='font-size:12px;color:#333'>{note}</span>",
        }
        layout = {**layout, "title": title}
    elif isinstance(title, str):
        layout = {**layout, "title": title + note}
    else:
        layout = {**layout, "title": note}

    fig = go.Figure(data=kept, layout=layout)
    return fig


def run_overlay(
    *,
    html: Path,
    pivots: Path,
    out: Path | None = None,
) -> Path:
    html_path = Path(html)
    if not html_path.is_absolute():
        html_path = _PROJECT_ROOT / html_path
    if not html_path.exists():
        raise FileNotFoundError(html_path)

    out_path = out
    if out_path is None:
        out_path = html_path.with_name(html_path.stem + "_oracle_pivots.html")
    else:
        out_path = Path(out_path)
        if not out_path.is_absolute():
            out_path = _PROJECT_ROOT / out_path

    pivots_path = Path(pivots)
    if not pivots_path.is_absolute():
        pivots_path = _PROJECT_ROOT / pivots_path

    html_txt = html_path.read_text(encoding="utf-8")
    traces = _extract_plotly_traces(html_txt)
    layout = extract_layout(html_txt)
    close, ohlcv = load_ohlcv_from_regime_html(html_path)
    pivots_df = load_pivots(pivots_path, close)
    print(
        f"[INFO] pivots buy={(pivots_df['kind']=='buy').sum()} "
        f"sell={(pivots_df['kind']=='sell').sum()} from "
        f"{pivots_path if pivots_path.exists() else 'recompute'}"
    )

    fig = build_figure(traces=traces, layout=layout, pivots=pivots_df, ohlcv=ohlcv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_adaptive_html(fig, out_path, include_plotlyjs="cdn")
    print(f"[OK] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_overlay(html=args.html, pivots=args.pivots, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
