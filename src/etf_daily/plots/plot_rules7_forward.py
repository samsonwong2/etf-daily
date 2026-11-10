"""A separate HTML page: hold_up, rules7, and buy-and-hold on one screen.

Reads a listing page and writes ``{YYYYMMDD}_forward/rules7_forward_*.html``
next to the listing directory, never inside it. Does not import the listing
plotter and does not redraw ``regime_transition_*_adaptive.html``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from etf_daily.lib.fair_path_research import ewma_scaled_envelope
from etf_daily.lib.rules7_forward import (
    ANCHOR_CODE,
    DEFAULT_PAGE_CODES,
    FORWARD_START,
    FREEZE_DATE,
    PSEUDO_FORWARD_START,
    holdout_line,
    load_prepared,
    load_specs,
    output_dir,
    page_dir,
    replay_frame,
)

PANELS = (
    ("fig7", "图7", "2年"),
    ("fig8", "图8", "1年"),
    ("fig9", "图9", "6个月"),
    ("fig10", "图10", "3个月"),
)
_MARKER_ROWS = (1, 4, 5, 6, 7)
_MARKER_STYLE = {
    "aux_edge": ("triangle-up", "#2ca02c"),
    "B1": ("diamond", "#1f77b4"),
    "卖点": ("triangle-down", "#d62728"),
    "回落": ("x", "#ff7f0e"),
    "待成交": ("circle-open", "#2ca02c"),
}


def _norm(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).normalize()
    return out[~out.index.duplicated(keep="last")].sort_index()


def _at(close: pd.Series, day: object) -> float:
    key = pd.Timestamp(day).normalize()
    if key not in close.index:
        return float("nan")
    value = close.loc[key]
    if isinstance(value, pd.Series):
        value = value.iloc[-1]
    return float(value)


def _touch_rate(close: pd.Series, rail: pd.Series, *, side: str) -> float:
    ok = close.notna() & rail.notna() & (rail > 0)
    if side == "lo":
        hit = close <= rail * 1.01
    else:
        hit = close >= rail * 0.99
    if not bool(ok.any()):
        return float("nan")
    return float(hit.loc[ok].mean())


def _rate_label(prefix: str, rate: float) -> str:
    if not np.isfinite(rate):
        return prefix
    return f"{prefix}，历史触碰 {rate:.1%}"


def _entry_kind(masks: dict[str, pd.Series], day: pd.Timestamp) -> str:
    key = pd.Timestamp(day).normalize()

    def _flag(name: str) -> bool:
        series = masks.get(name)
        if series is None or key not in series.index:
            return False
        value = series.loc[key]
        if isinstance(value, pd.Series):
            value = value.iloc[-1]
        return bool(value)

    if _flag("ae_buy"):
        return "aux_edge"
    if _flag("b1"):
        return "B1"
    return "aux_edge"


def _hover_entry(row: dict[str, Any], kind: str) -> str:
    fill = row.get("fill_date") or "还没有"
    px = row.get("fill_px")
    px_txt = "还没有" if px == "" or px is None else f"{float(px):.4f}"
    return f"{kind}<br>信号日 {row['signal_date']}<br>成交日 {fill}<br>成交价 {px_txt}"


def _hover_exit(row: dict[str, Any]) -> str:
    px = row.get("exit_fill_px")
    px_txt = "还没有" if px == "" or px is None else f"{float(px):.4f}"
    return (
        f"{row['exit_reason']}<br>信号日 {row['exit_signal_date']}"
        f"<br>成交日 {row.get('exit_fill_date') or '还没有'}<br>成交价 {px_txt}"
    )


def depth_score(frame: pd.DataFrame) -> pd.Series:
    """How many of 图7 to 图10 are near a rail. Negative is 贴下轨, positive is 贴上轨."""
    close = frame["px"].astype(float)
    score = pd.Series(0, index=frame.index, dtype=int)
    for key, _label, _window in PANELS:
        lower = pd.to_numeric(frame[f"{key}_lower"], errors="coerce")
        upper = pd.to_numeric(frame[f"{key}_upper"], errors="coerce")
        score = score - (close <= lower * 1.01).fillna(False).astype(int)
        score = score + (close >= upper * 0.99).fillna(False).astype(int)
    return score


def page_header(replay: dict[str, Any], verdict: dict[str, Any] | None) -> str:
    result = replay["result"]
    text = (
        f"{replay['code']} {replay['name']}"
        f"  样本内 {result['total']:+.1%} / 回撤 {result['mdd']:.1%}"
        f"  买入持有 {replay['bh_total']:+.1%}"
        f"  已平仓 {len(result['trades'])} 笔"
    )
    if not replay["applicable"]:
        text += "  规则不适用"
    text += f"<br>冻结日 {FREEZE_DATE}  前向从 {FORWARD_START} 起"
    if verdict and verdict.get("spec_fp"):
        text += f"  裁决 {verdict.get('overall')}（账本截至 {verdict.get('as_of')}）"
    text += f"<br>{holdout_line(replay['type'])}"
    if not (verdict and verdict.get("spec_fp")):
        text += "。先跑 etf-daily forward 才有 K1 到 K3。"
    return text


def _add_markers(fig: go.Figure, replay: dict[str, Any]) -> None:
    close = replay["frame"]["px"].astype(float)
    groups: dict[str, list[tuple[pd.Timestamp, str]]] = {name: [] for name in _MARKER_STYLE}
    for row in replay["rows"]:
        if row.get("exit_reason") == "待成交":
            groups["待成交"].append((pd.Timestamp(row["signal_date"]), _hover_entry(row, "待成交")))
            continue
        kind = _entry_kind(replay["masks"], pd.Timestamp(row["signal_date"]))
        groups[kind].append((pd.Timestamp(row["signal_date"]), _hover_entry(row, kind)))
        if row.get("exit_reason") in {"卖点", "回落"} and row.get("exit_signal_date"):
            groups[row["exit_reason"]].append((pd.Timestamp(row["exit_signal_date"]), _hover_exit(row)))
    for name, points in groups.items():
        if not points:
            continue
        symbol, color = _MARKER_STYLE[name]
        xs = [item[0] for item in points]
        texts = [item[1] for item in points]
        ys = [_at(close, item) for item in xs]
        first = True
        for row_i in _MARKER_ROWS:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    name=name,
                    legendgroup=name,
                    showlegend=first,
                    marker=dict(symbol=symbol, size=9, color=color, line=dict(width=0.5, color="#222")),
                    text=texts,
                    hovertemplate="%{text}<extra>" + name + "</extra>",
                ),
                row=row_i,
                col=1,
            )
            first = False


def _add_rails(fig: go.Figure, frame: pd.DataFrame, row: int, key: str, label: str, window: str) -> None:
    close = frame["px"].astype(float)
    path = pd.to_numeric(frame[f"{key}_path"], errors="coerce")
    upper = pd.to_numeric(frame[f"{key}_upper"], errors="coerce")
    lower = pd.to_numeric(frame[f"{key}_lower"], errors="coerce")
    fig.add_trace(
        go.Scatter(x=frame.index, y=path, name=f"{label} 中线 {window}", mode="lines",
                   line=dict(color="#d62728", width=1, dash="dash"), legendgroup=f"{key}-path"),
        row=row, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame.index, y=upper, mode="lines",
            name=_rate_label(f"{label} 上轨 P95", _touch_rate(close, upper, side="hi")),
            line=dict(color="#7f7f7f", width=1, dash="dot"), legendgroup=f"{key}-up",
        ),
        row=row, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame.index, y=lower, mode="lines",
            name=_rate_label(f"{label} 下轨 P95", _touch_rate(close, lower, side="lo")),
            line=dict(color="#7f7f7f", width=1, dash="dot"), legendgroup=f"{key}-lo",
        ),
        row=row, col=1,
    )
    _hi, lo, _delta = ewma_scaled_envelope(path, close)
    ewma_up, ewma_lo = _hi, lo
    fig.add_trace(
        go.Scatter(
            x=frame.index, y=ewma_up, mode="lines", name=f"{label} EWMA 上轨",
            visible="legendonly", line=dict(color="#17becf", width=1, dash="dashdot"),
            legendgroup=f"{key}-ewma-up",
        ),
        row=row, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame.index, y=ewma_lo, mode="lines", name=f"{label} EWMA 下轨",
            visible="legendonly", line=dict(color="#e377c2", width=1, dash="dashdot"),
            legendgroup=f"{key}-ewma-lo",
        ),
        row=row, col=1,
    )


def build_figure(
    replay: dict[str, Any],
    ohlcv: pd.DataFrame,
    hold_up: pd.Series | None,
    header: str,
) -> go.Figure:
    frame = _norm(replay["frame"])
    replay = {**replay, "frame": frame, "masks": {k: _norm(v.to_frame(name="v"))["v"] for k, v in replay["masks"].items()}}
    ohlc = _norm(ohlcv)
    close = frame["px"].astype(float)
    for src, col in (("px", "$open"), ("px", "$high"), ("px", "$low"), ("px", "$close")):
        if col not in ohlc.columns:
            ohlc[col] = close
    ohlc = ohlc.reindex(frame.index)
    for col in ("$open", "$high", "$low", "$close"):
        ohlc[col] = pd.to_numeric(ohlc[col], errors="coerce").fillna(close)

    fig = make_subplots(
        rows=7,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.20, 0.16, 0.10, 0.135, 0.135, 0.135, 0.135],
        subplot_titles=(
            "收盘与买卖标记",
            "净值：rules7、hold_up、买入持有",
            "图7到图10 贴轨张数（负=贴下轨，正=贴上轨）",
            "图7 2年",
            "图8 1年",
            "图9 6个月",
            "图10 3个月",
        ),
    )
    fig.add_trace(
        go.Scatter(x=frame.index, y=close, name="收盘", mode="lines", line=dict(color="#444", width=1)),
        row=1, col=1,
    )
    equity = replay["result"]["equity"]
    equity.index = pd.DatetimeIndex(pd.to_datetime(equity.index)).normalize()
    bh = close / float(close.iloc[0])
    fig.add_trace(
        go.Scatter(x=equity.index, y=equity, name="rules7 净值", mode="lines", line=dict(color="#1f77b4", width=1.6)),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=bh.index, y=bh, name="买入持有", mode="lines", line=dict(color="#2ca02c", width=1)),
        row=2, col=1,
    )
    if hold_up is not None and len(hold_up):
        nav = hold_up.copy()
        nav.index = pd.DatetimeIndex(pd.to_datetime(nav.index)).normalize()
        nav = nav[~nav.index.duplicated(keep="last")].sort_index()
        nav = nav.loc[nav.index <= frame.index[-1]]
        fig.add_trace(
            go.Scatter(x=nav.index, y=nav, name="hold_up 净值", mode="lines", line=dict(color="#ff7f0e", width=1)),
            row=2, col=1,
        )
    score = depth_score(frame)
    colors = ["#2ca02c" if v < 0 else "#d62728" if v > 0 else "#bbbbbb" for v in score.to_numpy()]
    fig.add_trace(
        go.Bar(x=score.index, y=score, name="贴轨张数", marker_color=colors, hovertemplate="%{x|%Y-%m-%d} %{y}<extra>贴轨张数</extra>"),
        row=3, col=1,
    )
    for offset, (key, label, window) in enumerate(PANELS):
        row = 4 + offset
        fig.add_trace(
            go.Candlestick(
                x=frame.index,
                open=ohlc["$open"],
                high=ohlc["$high"],
                low=ohlc["$low"],
                close=ohlc["$close"],
                name=f"{label} K线",
                showlegend=False,
                increasing_line_color="#d62728",
                decreasing_line_color="#2ca02c",
            ),
            row=row, col=1,
        )
        _add_rails(fig, frame, row, key, label, window)
    _add_markers(fig, replay)
    for row in replay["rows"]:
        if not row.get("fill_date"):
            continue
        x1 = row.get("exit_fill_date") or frame.index[-1]
        fig.add_vrect(
            x0=row["fill_date"], x1=str(pd.Timestamp(x1).date()),
            fillcolor="rgba(44,160,44,0.08)", line_width=0, row=1, col=1,
        )
    last = pd.Timestamp(frame.index[-1])
    start = pd.Timestamp(frame.index[0])
    pseudo = pd.Timestamp(PSEUDO_FORWARD_START)
    forward = pd.Timestamp(FORWARD_START)
    if start < pseudo:
        fig.add_vrect(
            x0=str(start.date()), x1=PSEUDO_FORWARD_START,
            fillcolor="rgba(0,0,0,0.04)", line_width=0, row=2, col=1,
            annotation_text="样本内", annotation_position="top left",
        )
    if start < forward and pseudo < forward:
        fig.add_vrect(
            x0=str(max(start, pseudo).date()), x1=FORWARD_START,
            fillcolor="rgba(200,160,0,0.10)", line_width=0, row=2, col=1,
            annotation_text="假前向", annotation_position="top left",
        )
    if last >= pd.Timestamp(FORWARD_START):
        fig.add_vrect(
            x0=FORWARD_START, x1=str(last.date()),
            fillcolor="rgba(0,100,200,0.08)", line_width=0, row=2, col=1,
            annotation_text="前向", annotation_position="top left",
        )
    fig.add_shape(
        type="line", x0=FREEZE_DATE, x1=FREEZE_DATE, y0=0, y1=1, yref="paper",
        line=dict(color="#333333", width=1, dash="dash"),
    )
    fig.update_yaxes(range=[-4.5, 4.5], row=3, col=1)
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(
        title=dict(text=header, font=dict(size=13)),
        height=1680,
        width=1100,
        margin=dict(t=120, l=60, r=20, b=40),
        legend=dict(orientation="h", y=-0.02, font=dict(size=10)),
        hovermode="closest",
        barmode="relative",
    )
    return fig


def write_page(fig: go.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=True, full_html=True)
    return path


def _nav_from_html(path: Path, name: str) -> pd.Series | None:
    from etf_daily.lib.fair_path_band_signals import _extract_plotly_traces, _plotly_x_to_index

    traces = _extract_plotly_traces(Path(path).read_text(encoding="utf-8"))
    for trace in traces:
        if trace.get("name") != name or trace.get("y") is None:
            continue
        idx = _plotly_x_to_index(trace.get("x"))
        series = pd.Series(np.asarray(trace["y"], dtype=float), index=pd.DatetimeIndex(idx).normalize())
        return series[~series.index.duplicated(keep="last")].sort_index()
    return None


def _refuse_listing_write(listing: Path, dest: Path) -> None:
    listing = Path(listing).resolve()
    dest = Path(dest).resolve()
    if dest == listing or listing in dest.parents:
        raise ValueError(f"refusing to write inside the listing directory: {dest}")
    if "from_listing" in dest.name:
        raise ValueError(f"refusing a from_listing output name: {dest}")


def _read_verdict(listing: Path) -> dict[str, Any] | None:
    import json

    from etf_daily.lib.rules7_forward import spec_fingerprint

    path = output_dir(listing) / "rules7_forward_verdict.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("spec_fp") != spec_fingerprint():
        return None
    return data


def write_code_page(
    html: Path,
    listing: Path,
    as_of: object,
    dest_dir: Path,
    *,
    verdict: dict[str, Any] | None = None,
    anchor: str = ANCHOR_CODE,
) -> Path:
    from etf_daily.lib.fair_path_band_signals import load_ohlcv_from_regime_html
    from etf_daily.scripts.backtest_fig12_six_states_pool import parse_html_label

    code, name = parse_html_label(Path(html))
    frame = load_prepared(html, listing, as_of, anchor=anchor)
    replay = replay_frame(code, name, frame, specs=load_specs())
    _close, ohlcv = load_ohlcv_from_regime_html(html)
    cut = pd.Timestamp(as_of).normalize()
    ohlcv = ohlcv.loc[pd.DatetimeIndex(pd.to_datetime(ohlcv.index)).normalize() <= cut]
    hold_up = _nav_from_html(Path(html), "hold_up净值")
    if hold_up is not None:
        hold_up = hold_up.loc[hold_up.index <= cut]
    fig = build_figure(replay, ohlcv, hold_up, page_header(replay, verdict))
    day = pd.Timestamp(as_of).strftime("%Y%m%d")
    safe = str(name).replace("/", "_").replace("\\", "_")
    return write_page(fig, Path(dest_dir) / f"rules7_forward_{code.upper()}_{safe}_{day}.html")


def write_pages(
    listing: Path,
    as_of: object,
    *,
    codes: list[str] | None = None,
    all_names: bool = False,
    dest_dir: Path | None = None,
    anchor: str = ANCHOR_CODE,
) -> dict[str, Any]:
    """Write one page per requested code. Returns paths and any names with no HTML."""
    from etf_daily.lib.fair_path_band_signals import find_regime_html
    from etf_daily.scripts.backtest_fig12_six_states_pool import list_equity_html, parse_html_label

    listing = Path(listing)
    if not listing.is_dir():
        raise FileNotFoundError(listing)
    dest = Path(dest_dir) if dest_dir is not None else page_dir(listing, as_of)
    _refuse_listing_write(listing, dest)
    if all_names:
        wanted = [parse_html_label(path)[0].upper() for path in list_equity_html(listing)]
    else:
        wanted = [code.upper() for code in (codes or list(DEFAULT_PAGE_CODES))]
    verdict = _read_verdict(listing)
    written: list[Path] = []
    missing: list[str] = []
    errors: list[str] = []
    for code in wanted:
        html = find_regime_html(listing, code)
        if html is None:
            missing.append(code)
            continue
        try:
            written.append(write_code_page(html, listing, as_of, dest, verdict=verdict, anchor=anchor))
        except Exception as exc:  # noqa: BLE001 — keep going so one page does not drop the rest
            errors.append(f"{code}: {exc}")
    return {
        "dest": dest,
        "written": written,
        "missing": missing,
        "errors": errors,
        "code": 2 if missing or errors else 0,
    }
