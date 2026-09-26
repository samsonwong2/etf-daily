#!/usr/bin/env python3
"""Research overlay for fig7–fig11 fair-path improvements (read-only).

Loads K-line data from an existing regime_transition HTML (no Qlib),
recomputes the production fair path / P95 envelope per scale, and adds
two research alternatives side by side:

- vol_scaled envelope: expanding P95 on rv5-standardized residuals, so
  band width follows the vol regime (calm -> tight, storm -> wide).
- Kalman local-linear-trend path + model-based 90% forecast band, an
  adaptive trend that reacts faster at genuine turning points.

Outputs a standalone research HTML plus a coverage_report.csv comparing
realized coverage of production vs research bands. Production HTML
generation and all trading signals are untouched.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from etf_daily.lib.fair_path_band_signals import (  # noqa: E402
    FIG_PANELS,
    load_ohlcv_from_regime_html,
)
from etf_daily.lib.fair_path_research import (  # noqa: E402
    build_research_panels,
    coverage_report,
    kalman_forecast_band,
    kalman_trend_path,
)
from etf_daily.lib.plotly_html_autoscale import write_adaptive_html  # noqa: E402
from etf_daily.lib.regime_transition_plot import (  # noqa: E402
    compute_volatility_regime_frame,
)

DEFAULT_HTML = (
    PLOTLY_OUTPUTS_DIR / "20260908_from_listing"
    / "regime_transition_SH512530_沪深300红利ETF建信_20190923_20260908_adaptive.html"
)
DEFAULT_OUT_DIR = PLOTLY_OUTPUTS_DIR / "research"


def build_research_frame(close: pd.Series) -> dict[str, pd.DataFrame]:
    """Per-panel production + floored vol-scaled path/band series."""
    return build_research_panels(close)


def render_research_html(
    ohlcv: pd.DataFrame,
    close: pd.Series,
    panels: dict[str, pd.DataFrame],
    kalman: tuple[pd.Series, pd.Series, pd.Series, pd.Series],
    coverage: pd.DataFrame,
    out_path: Path,
    *,
    title_code: str,
) -> None:
    k_path, k_g, k_std, k_mean = kalman
    k_up, k_lo = kalman_forecast_band(k_mean, k_std)
    keys = [k for k, _, _ in FIG_PANELS]
    n_rows = len(keys) + 1
    subplot_titles = tuple(
        f"{key} 滚动{panels[key].attrs['label']}｜生产P95 vs vol标准化 vs Kalman"
        for key in keys
    ) + ("覆盖率汇总（valid日内 close∈[下轨,上轨] 的占比）",)
    row_heights = [0.17] * len(keys) + [0.15]
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
        specs=[[{}] for _ in range(len(keys))] + [[{"type": "table"}]],
    )
    x = close.index
    o = ohlcv["$open"].to_numpy(dtype=float)
    h = ohlcv["$high"].to_numpy(dtype=float)
    l = ohlcv["$low"].to_numpy(dtype=float)
    c = ohlcv["$close"].to_numpy(dtype=float)

    for i, key in enumerate(keys):
        row = i + 1
        pf = panels[key]
        legend_group = key
        fig.add_trace(
            go.Candlestick(
                x=x, open=o, high=h, low=l, close=c,
                name="K线", legendgroup="kline", showlegend=(i == 0),
                increasing_line_color="#d62728", decreasing_line_color="#2ca02c",
            ),
            row=row, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=pf["path"], name="生产公平路径(log OLS)",
                legendgroup="prod_path", showlegend=(i == 0),
                line=dict(color="#7f7f7f", width=1.2, dash="dash"),
            ),
            row=row, col=1,
        )
        for col, nm, color in (
            ("upper", "生产上轨(P95)", "#bdbdbd"),
            ("lower", "生产下轨(P95)", "#bdbdbd"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=x, y=pf[col], name=nm,
                    legendgroup=f"prod_{col}", showlegend=(i == 0),
                    line=dict(color=color, width=1, dash="dot"),
                ),
                row=row, col=1,
            )
        for col, nm in (("vs_upper", "vol标准化上轨"), ("vs_lower", "vol标准化下轨")):
            fig.add_trace(
                go.Scatter(
                    x=x, y=pf[col], name=nm,
                    legendgroup=f"vs_{col}", showlegend=(i == 0),
                    line=dict(color="#1f77b4", width=1.2),
                ),
                row=row, col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=x, y=k_path, name="Kalman趋势路径",
                legendgroup="kalman_path", showlegend=(i == 0),
                line=dict(color="#9467bd", width=1.4),
            ),
            row=row, col=1,
        )
        for series, nm in ((k_up, "Kalman 90%预测带"), (k_lo, "Kalman 90%预测带")):
            fig.add_trace(
                go.Scatter(
                    x=x, y=series, name=nm,
                    legendgroup="kalman_band", showlegend=(i == 0 and series is k_up),
                    line=dict(color="#c5b0d5", width=0.8, dash="dash"),
                ),
                row=row, col=1,
            )
        fig.update_yaxes(title_text=panels[key].attrs["label"], row=row, col=1)
        _ = legend_group

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
                    overall["panel"] if "panel" in overall else overall["band"],
                    overall["band"],
                    overall["n"],
                    overall["coverage"],
                ]
            ),
        ),
        row=n_rows, col=1,
    )
    fig.update_layout(
        title=f"{title_code} 公平路径研究对照（生产P95 vs vol标准化 vs Kalman；纯研究）",
        height=360 * n_rows + 200,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    write_adaptive_html(fig, out_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--html", type=Path, default=DEFAULT_HTML,
                    help="既有 regime_transition adaptive HTML（解析其K线）")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--code", default=None, help="输出文件名前缀（默认从HTML文件名推断）")
    args = ap.parse_args(argv)

    close, ohlcv = load_ohlcv_from_regime_html(args.html)
    close = close.sort_index()
    ohlcv = ohlcv.reindex(close.index)
    code = args.code
    if not code:
        m = re.search(r"regime_transition_([A-Za-z0-9]+)_", args.html.stem)
        code = m.group(1).upper() if m else "UNK"

    panels = build_research_frame(close)
    kalman = kalman_trend_path(close)
    k_up, k_lo = kalman_forecast_band(kalman[3], kalman[2])

    vol_frame = compute_volatility_regime_frame(close)
    regime_s = vol_frame.set_index("as_of")["vol_regime"].reindex(close.index)

    reports: list[pd.DataFrame] = []
    for key, _, label in FIG_PANELS:
        pf = panels[key]
        bands = {
            "prod_p95": (pf["upper"], pf["lower"]),
            "vol_scaled": (pf["vs_upper"], pf["vs_lower"]),
        }
        rep = coverage_report(close, bands, vol_regime=regime_s)
        rep.insert(0, "panel", f"{key}({label})")
        reports.append(rep)
    kalman_rep = coverage_report(close, {"kalman_90": (k_up, k_lo)}, vol_regime=regime_s)
    kalman_rep.insert(0, "panel", "kalman(scale-free)")
    reports.append(kalman_rep)
    coverage = pd.concat(reports, ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = close.index[-1].strftime("%Y%m%d")
    html_path = args.out_dir / f"fair_path_research_{code}_{stamp}.html"
    csv_path = args.out_dir / f"fair_path_research_coverage_{code}_{stamp}.csv"
    coverage.to_csv(csv_path, index=False)
    render_research_html(
        ohlcv, close, panels, kalman, coverage, html_path, title_code=code,
    )
    print(f"wrote {html_path}")
    print(f"wrote {csv_path}")
    overall = coverage[coverage["dimension"] == "overall"][
        ["panel", "band", "n", "coverage"]
    ]
    print(overall.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
