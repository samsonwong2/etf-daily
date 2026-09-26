#!/usr/bin/env python3
"""Research overlay: expanding vs rolling vs EWMA residual envelopes.

Isolated inertia experiment on the production OLS path. Default ticker is
SH510300 (history through 2015). Production HTML/signals are untouched.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runtime_paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from decision_pack.src.fair_path_band_signals import (  # noqa: E402
    FIG_PANELS,
    load_ohlcv_from_regime_html,
)
from decision_pack.src.fair_path_research import (  # noqa: E402
    build_inertia_panels,
    coverage_report,
)
from decision_pack.src.plotly_html_autoscale import write_adaptive_html  # noqa: E402
from decision_pack.src.regime_transition_plot import (  # noqa: E402
    compute_volatility_regime_frame,
)

DEFAULT_HTML = (
    PLOTLY_OUTPUTS_DIR / "20260908_from_listing"
    / "regime_transition_SH510300_沪深300ETF华泰柏瑞_20120528_20260908_adaptive.html"
)
DEFAULT_OUT_DIR = PLOTLY_OUTPUTS_DIR / "research"

# Post-storm calm years used for the forget diagnostic.
FORGET_WINDOWS = (
    ("after_2015", "2016-01-01", "2016-12-31"),
    ("2015_left_3y", "2019-01-01", "2019-12-31"),
    ("after_2020", "2021-01-01", "2021-12-31"),
)


def _half_width(upper: pd.Series, lower: pd.Series) -> pd.Series:
    u = pd.to_numeric(upper, errors="coerce")
    lo = pd.to_numeric(lower, errors="coerce")
    out = np.log(u / lo)
    out = out.where(u.notna() & lo.notna() & (u > 0) & (lo > 0))
    return out


def forget_diagnostic(fig7: pd.DataFrame) -> pd.DataFrame:
    """Mean fig7 log half-width in post-storm calm years vs expanding."""
    rows: list[dict[str, object]] = []
    w_prod = _half_width(fig7["upper"], fig7["lower"])
    widths = {
        "prod": w_prod,
        "roll3y": _half_width(fig7["roll3y_upper"], fig7["roll3y_lower"]),
        "roll5y": _half_width(fig7["roll5y_upper"], fig7["roll5y_lower"]),
        "ewma": _half_width(fig7["ewma_upper"], fig7["ewma_lower"]),
        "ewma_clamp": _half_width(fig7["ewma_clamp_upper"], fig7["ewma_clamp_lower"]),
    }
    idx = fig7.index
    for name, start, end in FORGET_WINDOWS:
        m = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
        prod_m = float(w_prod.loc[m].mean()) if m.any() else float("nan")
        row: dict[str, object] = {
            "window": name,
            "start": start,
            "end": end,
            "n": int(m.sum()),
            "prod_width": prod_m,
        }
        for band, series in widths.items():
            if band == "prod":
                continue
            wm = float(series.loc[m].mean()) if m.any() else float("nan")
            row[f"{band}_width"] = wm
            row[f"{band}_forget_ok"] = bool(
                np.isfinite(wm) and np.isfinite(prod_m) and wm < prod_m
            )
        rows.append(row)
    return pd.DataFrame(rows)


def render_html(
    ohlcv: pd.DataFrame,
    close: pd.Series,
    panels: dict[str, pd.DataFrame],
    coverage: pd.DataFrame,
    out_path: Path,
    *,
    title_code: str,
) -> None:
    fig7 = panels["fig7"]
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.50, 0.28, 0.22],
        subplot_titles=(
            f"{title_code} fig7 滚动2年｜expanding P95 vs roll3y/5y vs EWMA vs clamp",
            "fig7 对数半宽（上轨/下轨）— 风暴后应收窄",
            "覆盖率（valid日内 close∈[下轨,上轨]）",
        ),
        specs=[[{}], [{}], [{"type": "table"}]],
    )
    x = close.index
    o = ohlcv["$open"].to_numpy(dtype=float)
    h = ohlcv["$high"].to_numpy(dtype=float)
    l = ohlcv["$low"].to_numpy(dtype=float)
    c = ohlcv["$close"].to_numpy(dtype=float)
    fig.add_trace(
        go.Candlestick(
            x=x, open=o, high=h, low=l, close=c, name="K线",
            increasing_line_color="#d62728", decreasing_line_color="#2ca02c",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=fig7["path"], name="生产公平路径(log OLS)",
            line=dict(color="#7f7f7f", width=1.2, dash="dash"),
        ),
        row=1, col=1,
    )
    traces = (
        ("upper", "lower", "expanding P95", "#bdbdbd", "dot"),
        ("roll3y_upper", "roll3y_lower", "roll 3y P95", "#1f77b4", "solid"),
        ("roll5y_upper", "roll5y_lower", "roll 5y P95", "#2ca02c", "solid"),
        ("ewma_upper", "ewma_lower", "EWMA σ×P95(|z|)", "#9467bd", "solid"),
        ("ewma_clamp_upper", "ewma_clamp_lower", "EWMA clamp", "#d62728", "solid"),
    )
    for hi, lo, name, color, dash in traces:
        fig.add_trace(
            go.Scatter(
                x=x, y=fig7[hi], name=f"{name} 上",
                line=dict(color=color, width=1.2, dash=dash),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=fig7[lo], name=f"{name} 下",
                line=dict(color=color, width=1.2, dash=dash),
                showlegend=False,
            ),
            row=1, col=1,
        )
    width_traces = (
        ("prod", "upper", "lower", "expanding", "#bdbdbd"),
        ("roll3y", "roll3y_upper", "roll3y_lower", "roll3y", "#1f77b4"),
        ("roll5y", "roll5y_upper", "roll5y_lower", "roll5y", "#2ca02c"),
        ("ewma", "ewma_upper", "ewma_lower", "ewma", "#9467bd"),
        ("ewma_clamp", "ewma_clamp_upper", "ewma_clamp_lower", "ewma_clamp", "#d62728"),
    )
    for _key, hi, lo, name, color in width_traces:
        w = _half_width(fig7[hi], fig7[lo])
        fig.add_trace(
            go.Scatter(x=x, y=w, name=f"半宽 {name}", line=dict(color=color, width=1.2)),
            row=2, col=1,
        )
    overall = coverage[coverage["dimension"] == "overall"].copy()
    overall["coverage"] = overall["coverage"].map(
        lambda v: f"{v:.1%}" if pd.notna(v) else "—"
    )
    fig.add_trace(
        go.Table(
            header=dict(
                values=["面板", "轨道", "valid天数", "覆盖率"],
                fill_color="#f0f0f0",
            ),
            cells=dict(
                values=[
                    overall["panel"] if "panel" in overall.columns else overall["band"],
                    overall["band"],
                    overall["n"],
                    overall["coverage"],
                ]
            ),
        ),
        row=3, col=1,
    )
    fig.update_layout(
        title=f"{title_code} expanding 惯性对照（研究轨；生产未改）",
        height=1100,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="log 半宽", row=2, col=1)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    write_adaptive_html(fig, out_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--html", type=Path, default=DEFAULT_HTML)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--code", default=None)
    args = ap.parse_args(argv)

    close, ohlcv = load_ohlcv_from_regime_html(args.html)
    close = close.sort_index()
    ohlcv = ohlcv.reindex(close.index)
    code = args.code
    if not code:
        m = re.search(r"regime_transition_([A-Za-z0-9]+)_", args.html.stem)
        code = m.group(1).upper() if m else "UNK"

    panels = build_inertia_panels(close)
    vol = compute_volatility_regime_frame(close)
    regime_s = vol.set_index("as_of")["vol_regime"].reindex(close.index)

    reports: list[pd.DataFrame] = []
    for key, _, label in FIG_PANELS:
        pf = panels[key]
        bands = {
            "prod_p95": (pf["upper"], pf["lower"]),
            "roll3y": (pf["roll3y_upper"], pf["roll3y_lower"]),
            "roll5y": (pf["roll5y_upper"], pf["roll5y_lower"]),
            "ewma": (pf["ewma_upper"], pf["ewma_lower"]),
            "ewma_clamp": (pf["ewma_clamp_upper"], pf["ewma_clamp_lower"]),
        }
        rep = coverage_report(close, bands, vol_regime=regime_s)
        rep.insert(0, "panel", f"{key}({label})")
        reports.append(rep)
    coverage = pd.concat(reports, ignore_index=True)
    forget = forget_diagnostic(panels["fig7"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = close.index[-1].strftime("%Y%m%d")
    html_path = args.out_dir / f"fair_path_inertia_{code}_{stamp}.html"
    csv_path = args.out_dir / f"fair_path_inertia_coverage_{code}_{stamp}.csv"
    forget_path = args.out_dir / f"fair_path_inertia_forget_{code}_{stamp}.csv"
    coverage.to_csv(csv_path, index=False, encoding="utf-8-sig")
    forget.to_csv(forget_path, index=False, encoding="utf-8-sig")
    render_html(
        ohlcv, close, panels, coverage, html_path, title_code=code,
    )
    print(f"wrote {html_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {forget_path}")
    print(forget.to_string(index=False))
    fig7_over = coverage[
        (coverage["panel"].astype(str).str.startswith("fig7"))
        & (coverage["dimension"] == "overall")
    ][["band", "n", "coverage"]]
    print(fig7_over.to_string(index=False))
    (args.out_dir / f"fair_path_inertia_forget_{code}_{stamp}.json").write_text(
        json.dumps(forget.to_dict(orient="records"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
