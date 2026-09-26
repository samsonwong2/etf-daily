"""Causal six-state classifier and fig12 (clean K + state vrects).

Priority chain matches documents/六状态趋势划分_八品种验证.md §1:
  ①过热 → ②急跌刀锋 → ③趋势下行 → ⑤趋势上行 → ④底部修复 → ⑥震荡
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd
import plotly.graph_objects as go

STATE_WARMUP = "数据不足"
STATE_OVERHEAT = "①过热"
STATE_KNIFE = "②急跌刀锋"
STATE_DOWN = "③趋势下行"
STATE_REPAIR = "④底部修复"
STATE_UP = "⑤趋势上行"
STATE_CHOP = "⑥震荡"

STATE_ORDER: tuple[str, ...] = (
    STATE_OVERHEAT,
    STATE_KNIFE,
    STATE_DOWN,
    STATE_REPAIR,
    STATE_UP,
    STATE_CHOP,
)

# Fill colors: 红=①过热、深绿=②急跌刀锋、浅绿=③下行、浅蓝=④修复、浅红=⑤上行、灰=⑥震荡
STATE_COLORS: dict[str, str] = {
    STATE_OVERHEAT: "rgba(214,39,40,0.32)",
    STATE_KNIFE: "rgba(0,100,0,0.30)",
    STATE_DOWN: "rgba(60,179,113,0.16)",
    STATE_REPAIR: "rgba(100,149,237,0.22)",
    STATE_UP: "rgba(255,120,120,0.20)",
    STATE_CHOP: "rgba(150,150,150,0.13)",
}

STATE_LEGEND_COLORS: dict[str, str] = {
    STATE_OVERHEAT: "rgba(214,39,40,0.85)",
    STATE_KNIFE: "rgba(0,100,0,0.85)",
    STATE_DOWN: "rgba(60,179,113,0.85)",
    STATE_REPAIR: "rgba(100,149,237,0.85)",
    STATE_UP: "rgba(255,120,120,0.85)",
    STATE_CHOP: "rgba(150,150,150,0.85)",
}

_KNIFE_BARS = 10


def classify_six_states(ms: pd.DataFrame) -> pd.Series:
    """Label each row of a multi-scale frame with one of the six states.

    ``ms`` is the output of ``compute_multi_scale_frame`` (optionally
    ``enrich_frame_for_checklist``). Warmup days (no close or no fig10 path)
    are tagged ``数据不足``.
    """
    required = (
        "px",
        "fig8_g",
        "fig9_g",
        "fig10_g",
        "fig10_path",
        "fig9_touch_hi",
        "fig10_touch_hi",
    )
    missing = [c for c in required if c not in ms.columns]
    if missing:
        raise KeyError(f"classify_six_states missing columns: {missing}")

    px = pd.to_numeric(ms["px"], errors="coerce")
    p10 = pd.to_numeric(ms["fig10_path"], errors="coerce")
    g8 = pd.to_numeric(ms["fig8_g"], errors="coerce")
    g9 = pd.to_numeric(ms["fig9_g"], errors="coerce")
    g10 = pd.to_numeric(ms["fig10_g"], errors="coerce")
    touch_hi = ms["fig9_touch_hi"].eq(True) | ms["fig10_touch_hi"].eq(True)

    below = (px < p10).fillna(False)
    below10_run = below.astype(int).groupby((~below).cumsum()).cumsum()

    warmup = ~(px.notna() & p10.notna())
    dual_up = g8.gt(0) & g9.gt(0)
    dual_down = g10.le(0) & g9.le(0)

    # Lowest priority first so later masks win.
    out = pd.Series(STATE_CHOP, index=ms.index, dtype=object)
    out = out.mask(px.ge(p10), STATE_REPAIR)
    out = out.mask(dual_up, STATE_UP)
    out = out.mask(dual_down, STATE_DOWN)
    out = out.mask(below10_run.ge(_KNIFE_BARS), STATE_KNIFE)
    out = out.mask(touch_hi.fillna(False), STATE_OVERHEAT)
    out = out.mask(warmup, STATE_WARMUP)
    out.name = "six_state"
    return out


def current_state(states: pd.Series) -> str:
    """Last-bar label (including warmup)."""
    if states.empty:
        return STATE_WARMUP
    return str(states.iloc[-1])


def _ohlc_cols(ohlcv: pd.DataFrame) -> tuple[str, str, str, str]:
    mapping = {
        "open": ("$open", "open"),
        "high": ("$high", "high"),
        "low": ("$low", "low"),
        "close": ("$close", "close"),
    }
    found: dict[str, str] = {}
    lower = {str(c).lower(): c for c in ohlcv.columns}
    for key, aliases in mapping.items():
        for alias in aliases:
            if alias in ohlcv.columns:
                found[key] = alias
                break
            if alias.lower() in lower:
                found[key] = lower[alias.lower()]
                break
        else:
            raise KeyError(f"ohlcv missing {key} column")
    return found["open"], found["high"], found["low"], found["close"]


def build_fig12_figure(
    ohlcv: pd.DataFrame,
    states: pd.Series,
    code_name: str,
    *,
    colors: Mapping[str, str] | None = None,
) -> go.Figure:
    """Clean candlesticks + consecutive-state vrects. No rangeslider, no trade labels."""
    fill = dict(STATE_COLORS if colors is None else colors)
    o_col, h_col, l_col, c_col = _ohlc_cols(ohlcv)
    idx = pd.DatetimeIndex(pd.to_datetime(ohlcv.index)).normalize()
    frame = pd.DataFrame(
        {
            "open": pd.to_numeric(ohlcv[o_col], errors="coerce").to_numpy(),
            "high": pd.to_numeric(ohlcv[h_col], errors="coerce").to_numpy(),
            "low": pd.to_numeric(ohlcv[l_col], errors="coerce").to_numpy(),
            "close": pd.to_numeric(ohlcv[c_col], errors="coerce").to_numpy(),
        },
        index=idx,
    )
    st = states.copy()
    st.index = pd.DatetimeIndex(pd.to_datetime(st.index)).normalize()
    st = st.reindex(frame.index)
    st = st.fillna(STATE_WARMUP)

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=frame.index,
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            increasing_line_color="#d62728",
            decreasing_line_color="#2ca02c",
            name="K线",
            showlegend=False,
        )
    )

    runs = (st != st.shift()).cumsum()
    for _, seg in st.groupby(runs, sort=False):
        label = str(seg.iloc[0])
        color = fill.get(label)
        if color is None:
            continue
        x0 = pd.Timestamp(seg.index[0])
        x1 = pd.Timestamp(seg.index[-1]) + pd.Timedelta(days=1)
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=color,
            line_width=0,
            layer="below",
        )

    for label in STATE_ORDER:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color=STATE_LEGEND_COLORS.get(label, fill[label])),
                name=label,
            )
        )

    title = f"图12 六状态背景色 · {code_name}" if code_name else "图12 六状态背景色"
    fig.update_layout(
        title=title,
        height=520,
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=20, t=50, b=30),
        legend=dict(orientation="h", y=1.08),
    )
    fig.update_xaxes(rangebreaks=[])
    return fig


def six_state_span_table(states: pd.Series) -> pd.DataFrame:
    """Consecutive runs as start/end/label rows (debug helper)."""
    if states.empty:
        return pd.DataFrame(columns=["start", "end", "state", "n"])
    runs = (states != states.shift()).cumsum()
    rows: list[dict[str, object]] = []
    for _, seg in states.groupby(runs, sort=False):
        rows.append(
            {
                "start": pd.Timestamp(seg.index[0]).strftime("%Y-%m-%d"),
                "end": pd.Timestamp(seg.index[-1]).strftime("%Y-%m-%d"),
                "state": str(seg.iloc[0]),
                "n": int(len(seg)),
            }
        )
    return pd.DataFrame(rows)
