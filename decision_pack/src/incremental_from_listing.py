"""Reuse yesterday's from_listing HTML and append one causal bar.

Fair-path OLS, expanding P95 rails, and fig7–10 EWMA residual rails are
causal in ``t``, so overlapping history can be kept when qfq closes match.
Falls back to a full rebuild when the previous HTML is missing or inconsistent.

Does not change production hold_up method selection.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from decision_pack.src.fair_path_band_signals import (
    FIG_PANELS,
    _decode_plotly_bdata,
    _extract_plotly_traces,
    _plotly_x_to_index,
    load_ohlcv_from_regime_html,
)
from decision_pack.src.plotly_html_autoscale import write_adaptive_html

_OVERLAY_HTML_MARK = re.compile(
    r"_(fig\d|oracle|aux_edge|hard_touch|near_edge)", re.IGNORECASE
)

TRAIL_YAXIS: dict[float, str] = {
    2.0: "y10",
    1.0: "y11",
    0.5: "y12",
    0.25: "y13",
    1.0 / 12.0: "y14",
}

CANDLE_YAXES = ("y", "y1", "y10", "y11", "y12", "y13", "y14")


def find_listing_html(listing_dir: Path, code: str) -> Path | None:
    """Primary adaptive HTML (not research overlays)."""
    code = str(code).upper()
    hits = []
    for p in listing_dir.glob(f"regime_transition_{code}_*_adaptive.html"):
        if _OVERLAY_HTML_MARK.search(p.stem):
            continue
        hits.append(p)
    if not hits:
        return None
    return sorted(hits)[0]


def _as_1d_list(arr: Any) -> list[Any]:
    if arr is None:
        return []
    if isinstance(arr, dict) and "bdata" in arr and "dtype" in arr:
        arr = _decode_plotly_bdata(arr)
    a = np.asarray(arr)
    if a.ndim == 0:
        return [a.item()]
    return a.ravel().tolist()


def _x_values(tr: dict[str, Any]) -> np.ndarray:
    return np.asarray(_as_1d_list(tr.get("x")))


def _trace_dates(tr: dict[str, Any]) -> pd.DatetimeIndex:
    x = tr.get("x")
    if x is None:
        return pd.DatetimeIndex([])
    if isinstance(x, dict) and "bdata" in x:
        x = _decode_plotly_bdata(x)
    return _plotly_x_to_index(x).normalize()


def _fig7_candle(traces: list[dict[str, Any]]) -> dict[str, Any] | None:
    for tr in traces:
        if tr.get("type") != "candlestick":
            continue
        if str(tr.get("yaxis") or "") == "y10":
            return tr
    for tr in traces:
        if tr.get("type") == "candlestick" and str(tr.get("name", "")).startswith("K线"):
            yax = str(tr.get("yaxis") or "y")
            if yax in {"y10", "y11", "y12", "y13", "y14"}:
                return tr
    return None


def overlap_close_ok(
    prev_close: pd.Series,
    new_close: pd.Series,
    *,
    tol: float,
) -> tuple[bool, str]:
    prev_close = pd.to_numeric(prev_close, errors="coerce")
    new_close = pd.to_numeric(new_close, errors="coerce")
    prev_close.index = pd.DatetimeIndex(pd.to_datetime(prev_close.index)).normalize()
    new_close.index = pd.DatetimeIndex(pd.to_datetime(new_close.index)).normalize()
    common = prev_close.index.intersection(new_close.index)
    if len(common) < 5:
        return False, "overlap_short"
    a = prev_close.reindex(common).to_numpy(float)
    b = new_close.reindex(common).to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(a) > 1e-12)
    if int(ok.sum()) < 5:
        return False, "overlap_nan"
    rel = np.abs(a[ok] / b[ok] - 1.0)
    if float(np.nanmax(rel)) > float(tol):
        return False, "qfq"
    return True, "ok"


def classify_date_gap(
    prev_last: pd.Timestamp,
    new_index: pd.DatetimeIndex,
) -> str:
    """Return plus_one | same | gap | empty."""
    if new_index.empty:
        return "empty"
    new_last = pd.Timestamp(new_index[-1]).normalize()
    prev_last = pd.Timestamp(prev_last).normalize()
    if new_last == prev_last:
        return "same"
    after = new_index[new_index > prev_last]
    if len(after) == 1:
        return "plus_one"
    if len(after) == 0:
        return "same"
    return "gap"


def _ols_last_path_g(
    close: pd.Series,
    *,
    trail_years: float,
    min_bars: int,
    trading_days_per_year: float = 252.0,
) -> tuple[float, float]:
    """Causal trailing log OLS at the last bar only."""
    y = pd.to_numeric(close, errors="coerce").to_numpy(float)
    n = len(y)
    tdy = float(trading_days_per_year)
    trail_bars = int(round(float(trail_years) * tdy)) if trail_years > 0 else 0
    t = n - 1
    start = 0
    if trail_bars > 0 and (t + 1) >= trail_bars:
        start = t + 1 - trail_bars
    sl = y[start : t + 1]
    ok = np.isfinite(sl) & (sl > 0)
    n_ok = int(ok.sum())
    if n_ok < max(2, int(min_bars)):
        if n_ok < 2:
            return float("nan"), float("nan")
        idx = np.flatnonzero(ok)
        p0 = float(sl[int(idx[0])])
        p1 = float(sl[int(idx[-1])])
        n_steps = int(idx[-1] - idx[0])
        g = (p1 / p0) ** (tdy / n_steps) - 1.0 if n_steps > 0 and p0 > 0 else float("nan")
        path = p0 * ((1.0 + float(g)) ** (n_steps / tdy)) if n_steps > 0 and np.isfinite(g) else p1
        return float(path), float(g)
    jj = np.flatnonzero(ok).astype(float)
    log_y = np.log(sl[ok])
    b, a = np.polyfit(jj, log_y, 1)
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan"), float("nan")
    path = float(np.exp(a + b * float(jj[-1])))
    g = float(np.exp(b * tdy) - 1.0)
    return path, g


def _new_envelope(
    path_hist: np.ndarray,
    close_hist: np.ndarray,
    *,
    path_new: float,
    close_new: float,
    q: float = 0.95,
    min_bars: int = 63,
) -> tuple[float, float, float]:
    abs_hist: list[float] = []
    for pt, ct in zip(path_hist, close_hist, strict=False):
        pt = float(pt)
        ct = float(ct)
        if np.isfinite(pt) and pt > 0 and np.isfinite(ct) and ct > 0:
            rt = float(np.log(ct / pt))
            if np.isfinite(rt):
                abs_hist.append(abs(rt))
    if np.isfinite(path_new) and path_new > 0 and np.isfinite(close_new) and close_new > 0:
        rt = float(np.log(close_new / path_new))
        if np.isfinite(rt):
            abs_hist.append(abs(rt))
    if len(abs_hist) < max(2, int(min_bars)):
        return float("nan"), float("nan"), float("nan")
    delta = float(np.quantile(np.asarray(abs_hist, dtype=float), float(q)))
    if not np.isfinite(delta) or delta < 0:
        return float("nan"), float("nan"), float("nan")
    upper = float(path_new * np.exp(delta))
    lower = float(path_new * np.exp(-delta))
    return upper, lower, delta


def _horizon_label(years: float) -> str:
    y = float(years)
    months = y * 12.0
    if abs(months - round(months)) < 1e-3 and 0.5 <= months < 12:
        return f"{int(round(months))}个月"
    if abs(y - round(y)) < 1e-6:
        return f"{int(round(y))}年"
    return f"{y:g}年"


def _find_scatter(
    traces: list[dict[str, Any]],
    *,
    yaxis: str,
    name_contains: str,
) -> dict[str, Any] | None:
    for tr in traces:
        if str(tr.get("yaxis") or "") != yaxis:
            continue
        if tr.get("type") not in (None, "scatter"):
            continue
        name = str(tr.get("name") or "")
        if name_contains in name:
            return tr
    return None


def _find_rail(
    traces: list[dict[str, Any]],
    *,
    yaxis: str,
    hz: str,
    upper: bool,
    kind: str,
) -> dict[str, Any] | None:
    """P95 vs EWMA rail. ``kind`` is ``p95`` or ``ewma``."""
    prefix = f"{'上轨' if upper else '下轨'}{hz}"
    for tr in traces:
        if str(tr.get("yaxis") or "") != yaxis:
            continue
        if tr.get("type") not in (None, "scatter"):
            continue
        name = str(tr.get("name") or "")
        if prefix not in name:
            continue
        is_ewma = "EWMA残差轨" in name
        if kind == "ewma" and is_ewma:
            return tr
        if kind == "p95" and not is_ewma:
            return tr
    return None


def _ewma_rail_hover(
    upper_v: float,
    lower_v: float,
    path_v: float,
    *,
    is_upper: bool,
) -> str:
    if not (np.isfinite(path_v) and path_v > 0 and np.isfinite(upper_v)):
        return "上轨(EWMA)=na" if is_upper else "下轨(EWMA)=na"
    half = float(upper_v) / float(path_v) - 1.0
    if is_upper:
        return f"上轨(EWMA 当日{half:+.1%})={upper_v:.4f}"
    return f"下轨(EWMA 当日{-half:+.1%})={lower_v:.4f}"


def _extend_ewma_rail(
    tr: dict[str, Any],
    *,
    x_src: dict[str, Any],
    y_new: np.ndarray,
    hovers: list[str],
) -> None:
    """Append missing trailing EWMA points so the rail matches path length."""
    ys = _as_1d_list(tr.get("y"))
    xs = _as_1d_list(tr.get("x"))
    x_ref = _as_1d_list(x_src.get("x"))
    start = len(ys)
    target = int(len(y_new))
    if start >= target:
        return
    for i in range(start, target):
        if i < len(x_ref):
            xs.append(x_ref[i])
        ys.append(float(y_new[i]) if np.isfinite(y_new[i]) else float("nan"))
        tr["x"] = xs
        tr["y"] = ys
        hover = hovers[i] if i < len(hovers) else "上轨(EWMA)=na"
        _align_customdata_after_y(tr, hover)
    tr["x"] = xs
    tr["y"] = ys


def _catch_up_ewma_rails(
    traces: list[dict[str, Any]],
    *,
    yaxis: str,
    hz: str,
    path_tr: dict[str, Any],
    candle_tr: dict[str, Any] | None,
) -> None:
    """Extend fig7–10 EWMA rails to the current path length (incl. missed days)."""
    up_ew = _find_rail(traces, yaxis=yaxis, hz=hz, upper=True, kind="ewma")
    lo_ew = _find_rail(traces, yaxis=yaxis, hz=hz, upper=False, kind="ewma")
    if up_ew is None and lo_ew is None:
        return
    from decision_pack.src.fair_path_research import ewma_scaled_envelope

    path_arr = np.asarray(_as_1d_list(path_tr.get("y")), dtype=float)
    if candle_tr is not None:
        close_arr = np.asarray(_as_1d_list(candle_tr.get("close")), dtype=float)
    else:
        close_arr = np.full(len(path_arr), np.nan, dtype=float)
    m = min(len(path_arr), len(close_arr))
    if m < 2:
        return
    eu, el, _delta = ewma_scaled_envelope(
        pd.Series(path_arr[:m]),
        pd.Series(close_arr[:m]),
    )
    hovers_up: list[str] = []
    hovers_lo: list[str] = []
    for i in range(m):
        uv = float(eu.iloc[i])
        lv = float(el.iloc[i])
        pv = float(path_arr[i])
        hovers_up.append(_ewma_rail_hover(uv, lv, pv, is_upper=True))
        hovers_lo.append(_ewma_rail_hover(uv, lv, pv, is_upper=False))
    if up_ew is not None:
        _extend_ewma_rail(
            up_ew, x_src=path_tr, y_new=eu.to_numpy(dtype=float)[:m], hovers=hovers_up
        )
    if lo_ew is not None:
        _extend_ewma_rail(
            lo_ew, x_src=path_tr, y_new=el.to_numpy(dtype=float)[:m], hovers=hovers_lo
        )


def _panel_candle(
    traces: list[dict[str, Any]], yaxis: str
) -> dict[str, Any] | None:
    for tr in traces:
        if tr.get("type") == "candlestick" and str(tr.get("yaxis") or "") == yaxis:
            return tr
    return None


def _catch_up_all_ewma_rails(
    traces: list[dict[str, Any]], trails: list[float]
) -> int:
    """Fill short EWMA rails on all trail panels. Returns panels repaired."""
    n_fixed = 0
    for years in trails:
        yax = TRAIL_YAXIS.get(float(years))
        if yax is None:
            continue
        hz = _horizon_label(float(years))
        path_tr = _find_scatter(traces, yaxis=yax, name_contains="公平路径")
        if path_tr is None:
            continue
        up_ew = _find_rail(traces, yaxis=yax, hz=hz, upper=True, kind="ewma")
        n_path = len(_as_1d_list(path_tr.get("y")))
        n_ew = len(_as_1d_list(up_ew.get("y"))) if up_ew is not None else n_path
        _catch_up_ewma_rails(
            traces,
            yaxis=yax,
            hz=hz,
            path_tr=path_tr,
            candle_tr=_panel_candle(traces, yax),
        )
        if up_ew is not None and n_ew < n_path:
            n_fixed += 1
    return n_fixed


def _find_named(
    traces: list[dict[str, Any]],
    *,
    name: str,
    yaxis: str,
) -> dict[str, Any] | None:
    for tr in traces:
        if str(tr.get("name") or "") != name:
            continue
        if str(tr.get("yaxis") or "") != yaxis:
            continue
        return tr
    return None


def _append_scalar(tr: dict[str, Any], field: str, value: Any) -> None:
    vals = _as_1d_list(tr.get(field))
    vals.append(value)
    tr[field] = vals


def _encode_new_x(tr: dict[str, Any], new_ts: pd.Timestamp) -> Any:
    xs = _as_1d_list(tr.get("x"))
    new_ts = pd.Timestamp(new_ts).normalize()
    if xs and isinstance(xs[0], (int, float, np.floating)) and not isinstance(xs[0], bool):
        v0 = float(xs[0])
        if v0 > 1e11:
            return int(new_ts.value // 1_000_000)
        return int(new_ts.timestamp() * 1000)
    sample = str(xs[0]) if xs else ""
    if "T" in sample:
        return f"{new_ts.strftime('%Y-%m-%d')}T{sample.split('T', 1)[1]}"
    return new_ts.strftime("%Y-%m-%d")


def _append_new_x(tr: dict[str, Any], new_ts: pd.Timestamp) -> None:
    xs = _as_1d_list(tr.get("x"))
    xs.append(_encode_new_x(tr, new_ts))
    tr["x"] = xs


def _replace_last_x(tr: dict[str, Any], new_ts: pd.Timestamp) -> None:
    xs = _as_1d_list(tr.get("x"))
    if not xs:
        return
    xs[-1] = _encode_new_x(tr, new_ts)
    tr["x"] = xs


def _expected_trails(extra_trail_years: str | None) -> list[float]:
    from decision_pack.scripts.plot_regime_transition_example import (
        FAIR_PATH_LOG_TREND_TRAIL_YEARS,
        _normalize_extra_trail_years,
    )

    extra = _normalize_extra_trail_years(extra_trail_years)
    return [float(FAIR_PATH_LOG_TREND_TRAIL_YEARS)] + list(extra)


def traces_have_fair_panels(
    traces: list[dict[str, Any]],
    trails: list[float],
) -> bool:
    for y in trails:
        yax = TRAIL_YAXIS.get(float(y))
        if yax is None:
            continue
        hz = _horizon_label(float(y))
        path = _find_scatter(traces, yaxis=yax, name_contains="公平路径")
        up = _find_scatter(traces, yaxis=yax, name_contains=f"上轨{hz}")
        lo = _find_scatter(traces, yaxis=yax, name_contains=f"下轨{hz}")
        if path is None or up is None or lo is None:
            return False
    return True


def bump_html_end_tag(src_name: str, new_end: str) -> str:
    end_tag = pd.Timestamp(new_end).strftime("%Y%m%d")
    return re.sub(r"_(\d{8})_adaptive\.html$", f"_{end_tag}_adaptive.html", src_name)


def extract_layout(html: str) -> dict[str, Any]:
    from decision_pack.scripts.overlay_oracle_pivots_on_regime_html import (
        extract_layout as _el,
    )

    return _el(html)


def _update_title_g(layout: dict[str, Any], g_by_years: dict[float, float], deltas: dict[float, float]) -> None:
    title = layout.get("title")
    text = ""
    if isinstance(title, dict):
        text = str(title.get("text") or "")
    elif isinstance(title, str):
        text = title
    for years, g in g_by_years.items():
        if not np.isfinite(g):
            continue
        hz = _horizon_label(years)
        pat = re.compile(rf"({re.escape(hz)}\s*g≈)-?[\d.]+%")
        text = pat.sub(rf"\g<1>{g:.1%}", text)
        text = text.replace(f"{hz} g≈{g:.1%}".replace("%", "％"), f"{hz} g≈{g:.1%}")
    for years, d in deltas.items():
        if not np.isfinite(d):
            continue
        hz = _horizon_label(years)
        pct = f"{d:.1%}"
        text = re.sub(rf"(上轨{re.escape(hz)}\s*)\+[\d.]+%", rf"\g<1>+{pct}", text)
        text = re.sub(rf"(下轨{re.escape(hz)}\s*)-[\d.]+%", rf"\g<1>-{pct}", text)
    if isinstance(title, dict):
        layout["title"] = {**title, "text": text}
    else:
        layout["title"] = text


def _customdata_text(cell: Any) -> str:
    if cell is None:
        return ""
    if isinstance(cell, (list, tuple)):
        return "" if not cell else _customdata_text(cell[0])
    return str(cell)


def _append_customdata(tr: dict[str, Any], text: str) -> None:
    raw = tr.get("customdata")
    if raw is None:
        tr["customdata"] = [text]
        return
    vals = _as_1d_list(raw)
    if vals and isinstance(vals[0], (list, tuple)):
        vals.append([text])
    else:
        vals.append(text)
    tr["customdata"] = vals


def _align_customdata_after_y(tr: dict[str, Any], text: str) -> None:
    raw = tr.get("customdata")
    if raw is None:
        return
    vals = _as_1d_list(raw)
    ys = _as_1d_list(tr.get("y"))
    nested = bool(vals) and isinstance(vals[0], (list, tuple))
    cell: Any = [text] if nested else text
    if len(vals) == len(ys) - 1:
        vals.append(cell)
    elif len(vals) == len(ys) and vals:
        vals[-1] = cell
    else:
        return
    tr["customdata"] = vals


def _align_bar_color_after_y(tr: dict[str, Any], color: str) -> None:
    marker = tr.get("marker")
    if not isinstance(marker, dict):
        return
    colors = marker.get("color")
    if not isinstance(colors, list):
        return
    ys = _as_1d_list(tr.get("y"))
    if len(colors) == len(ys) - 1:
        colors.append(color)
    elif len(colors) == len(ys) and colors:
        colors[-1] = color
    else:
        return
    marker["color"] = colors
    tr["marker"] = marker


def _last_finite(series: pd.Series) -> float:
    v = pd.to_numeric(series, errors="coerce")
    v = v[np.isfinite(v.to_numpy(dtype=float))]
    if v.empty:
        return float("nan")
    return float(v.iloc[-1])


def _fig1_5_last_values(
    close: pd.Series,
    ohlcv_last: pd.Series,
    *,
    rv20_anchor: pd.Series | None = None,
    rv5_anchor: pd.Series | None = None,
    need_ann_fallback: float | None = None,
    need_ann_5_fallback: float | None = None,
    erp_ann: float | None = None,
) -> dict[str, Any]:
    """Causal last-bar values for fig1–5 overlays (MA / volume / vol / need-gap)."""
    from decision_pack.src.regime_transition_plot import compute_volatility_regime_frame
    from decision_pack.src.vol_need_return_board import (
        ANCHOR_ERP_ANN,
        HORIZON_5,
        HORIZON_20,
        fair_need_ann_series,
        gap_realized_minus_need,
    )

    out: dict[str, Any] = {}
    prem = float(ANCHOR_ERP_ANN if erp_ann is None else erp_ann)
    c = pd.to_numeric(close, errors="coerce")
    c.index = pd.DatetimeIndex(pd.to_datetime(c.index)).normalize()
    c = c[~c.index.duplicated(keep="last")].sort_index()
    for w in (5, 10, 20):
        out[f"MA{w}"] = float(c.rolling(int(w), min_periods=int(w)).mean().iloc[-1]) if len(c) else float("nan")
    vol_raw = ohlcv_last.get("$volume") if ohlcv_last is not None else None
    out["成交量"] = float(pd.to_numeric(pd.Series([vol_raw]), errors="coerce").iloc[0])
    vf = compute_volatility_regime_frame(c)
    if vf is None or vf.empty:
        return out
    vf = vf.copy()
    vf["as_of"] = pd.to_datetime(vf["as_of"]).dt.normalize()
    last = vf.iloc[-1]
    out["rv5"] = float(last.get("rv5", float("nan")))
    out["rv20"] = float(last.get("rv20", float("nan")))
    out["vol5_pct_120d"] = float(last.get("vol5_pct_120d", float("nan")))
    out["vol_regime"] = str(last.get("vol_regime") or "")
    indexed = vf.set_index("as_of")
    need = None
    if rv20_anchor is not None and not getattr(rv20_anchor, "empty", True):
        need = fair_need_ann_series(indexed["rv20"], rv20_anchor, erp_ann=prem)
    if need is None or need.empty or not np.isfinite(float(need.iloc[-1]) if len(need) else float("nan")):
        fb = need_ann_fallback if need_ann_fallback is not None else prem
        need = pd.Series(float(fb), index=c.index, dtype=float)
    need5 = need
    if (
        rv5_anchor is not None
        and not getattr(rv5_anchor, "empty", True)
        and "rv5" in indexed.columns
    ):
        need5_try = fair_need_ann_series(indexed["rv5"], rv5_anchor, erp_ann=prem)
        if need5_try is not None and not need5_try.empty:
            need5 = need5_try.reindex(need.index)
    elif need_ann_5_fallback is not None:
        need5 = pd.Series(float(need_ann_5_fallback), index=need.index, dtype=float)
    gap20 = gap_realized_minus_need(c, need, bars=HORIZON_20)
    gap5 = gap_realized_minus_need(c, need5, bars=HORIZON_5)
    out["公平年化收益"] = _last_finite(need)
    out["公平年化收益(rv5)"] = _last_finite(need5)
    out["20日差额(实现−公平)"] = _last_finite(gap20["gap"])
    out["5日差额(实现−公平)"] = _last_finite(gap5["gap"])
    out["r_need_20d"] = _last_finite(gap20["r_need_period"])
    out["r_20d"] = _last_finite(gap20["r_realized"])
    out["r_need_5d"] = _last_finite(gap5["r_need_period"])
    out["r_5d"] = _last_finite(gap5["r_realized"])
    return out


def _gap_bar_color(gap: float) -> str:
    if np.isfinite(gap) and float(gap) < 0:
        return "rgba(214,39,40,0.55)"
    return "rgba(44,160,44,0.50)"


def _append_named_overlay(
    traces: list[dict[str, Any]],
    *,
    name: str,
    yaxis: str,
    new_ts: pd.Timestamp,
    value: float,
    expected_len: int,
    customdata: str | None = None,
    bar_color: str | None = None,
) -> None:
    tr = _find_named(traces, name=name, yaxis=yaxis)
    if tr is None:
        return
    xs = _as_1d_list(tr.get("x"))
    ys = _as_1d_list(tr.get("y"))
    if len(xs) != int(expected_len) or len(ys) != int(expected_len):
        return
    _append_new_x(tr, new_ts)
    _append_scalar(tr, "y", value)
    if customdata is not None:
        _align_customdata_after_y(tr, customdata)
    if bar_color is not None:
        _align_bar_color_after_y(tr, bar_color)


def _extend_fig15_ref_lines(traces: list[dict[str, Any]], new_ts: pd.Timestamp) -> None:
    from decision_pack.src.vol_need_return_board import EQUITY_ANCHOR_LABEL

    for tr in traces:
        name = str(tr.get("name") or "")
        xs = _as_1d_list(tr.get("x"))
        if len(xs) != 2:
            continue
        if name in {"分位30%", "分位70%", "分位90%", "差额=0"} or (
            name.startswith(f"{EQUITY_ANCHOR_LABEL}公平")
        ):
            _replace_last_x(tr, new_ts)


def _append_fig1_5_overlays(
    traces: list[dict[str, Any]],
    *,
    close: pd.Series,
    ohlcv_last: pd.Series,
    new_ts: pd.Timestamp,
    expected_len: int,
    rv20_anchor: pd.Series | None = None,
    rv5_anchor: pd.Series | None = None,
    erp_ann: float | None = None,
) -> None:
    """Recompute fig1–5 last-bar overlays. Do not copy yesterday's y."""
    from decision_pack.scripts.plot_regime_transition_example import (
        _fmt_pct_rank,
        _fmt_vol,
    )
    from decision_pack.src.vol_need_return_board import ANCHOR_ERP_ANN, EQUITY_ANCHOR_LABEL

    need_tr = _find_named(traces, name="公平年化收益", yaxis="y5")
    need5_tr = _find_named(traces, name="公平年化收益(rv5)", yaxis="y7")
    need_fb = None
    need5_fb = None
    if need_tr is not None:
        ys = pd.to_numeric(pd.Series(_as_1d_list(need_tr.get("y")), dtype=object), errors="coerce")
        if len(ys) and np.isfinite(float(ys.iloc[-1])):
            need_fb = float(ys.iloc[-1])
    if need5_tr is not None:
        ys = pd.to_numeric(pd.Series(_as_1d_list(need5_tr.get("y")), dtype=object), errors="coerce")
        if len(ys) and np.isfinite(float(ys.iloc[-1])):
            need5_fb = float(ys.iloc[-1])
    vals = _fig1_5_last_values(
        close,
        ohlcv_last,
        rv20_anchor=rv20_anchor,
        rv5_anchor=rv5_anchor,
        need_ann_fallback=need_fb,
        need_ann_5_fallback=need5_fb,
        erp_ann=erp_ann,
    )
    prem = float(ANCHOR_ERP_ANN if erp_ann is None else erp_ann)
    erp_txt = f"{prem:.1%}"
    date_s = pd.Timestamp(new_ts).strftime("%Y-%m-%d")
    rv5_hover = (
        f"日期={date_s}<br>"
        f"vol_regime={vals.get('vol_regime', '')}<br>"
        f"rv5={_fmt_vol(vals.get('rv5'))}<br>"
        f"rv20={_fmt_vol(vals.get('rv20'))}<br>"
        f"vol5_pct_120d={_fmt_pct_rank(vals.get('vol5_pct_120d'))}<br>"
        f"说明=分位用近120日rv5历史（含当日），无未来数据"
    )

    def _need_hover(bars: int, *, rv5: bool) -> str:
        key_need = "r_need_5d" if bars == 5 else "r_need_20d"
        key_real = "r_5d" if bars == 5 else "r_20d"
        key_gap = "5日差额(实现−公平)" if bars == 5 else "20日差额(实现−公平)"
        key_ann = "公平年化收益(rv5)" if rv5 else "公平年化收益"
        vol_tag = "rv5" if rv5 else "rv20"
        return (
            f"日期={date_s}<br>"
            f"公平年化={_fmt_vol(vals.get(key_ann))}<br>"
            f"公平{bars}日={_fmt_vol(vals.get(key_need))}<br>"
            f"已实现{bars}日={_fmt_vol(vals.get(key_real))}<br>"
            f"差额=已实现−公平{bars}日={_fmt_vol(vals.get(key_gap))}<br>"
            f"公式={erp_txt}×{vol_tag}/{vol_tag}_{EQUITY_ANCHOR_LABEL}（实现波动）"
        )

    def _gap_hover(bars: int) -> str:
        key_need = "r_need_5d" if bars == 5 else "r_need_20d"
        key_real = "r_5d" if bars == 5 else "r_20d"
        key_gap = "5日差额(实现−公平)" if bars == 5 else "20日差额(实现−公平)"
        g = vals.get(key_gap)
        if np.isfinite(float(g)) if g is not None else False:
            gf = float(g)
            note = (
                f"近{bars}日没赚够波动补偿"
                if gf < 0
                else f"近{bars}日覆盖了波动补偿"
            )
        else:
            note = "差额=—"
        return (
            f"日期={date_s}<br>"
            f"已实现{bars}日={_fmt_vol(vals.get(key_real))}<br>"
            f"公平{bars}日={_fmt_vol(vals.get(key_need))}<br>"
            f"差额={_fmt_vol(vals.get(key_gap))}<br>"
            f"{note}"
        )

    for name, yax in (("MA5", "y"), ("MA10", "y"), ("MA20", "y")):
        _append_named_overlay(
            traces, name=name, yaxis=yax, new_ts=new_ts,
            value=float(vals.get(name, float("nan"))), expected_len=expected_len,
        )
    _append_named_overlay(
        traces, name="成交量", yaxis="y2", new_ts=new_ts,
        value=float(vals.get("成交量", float("nan"))), expected_len=expected_len,
    )
    _append_named_overlay(
        traces, name="rv5", yaxis="y3", new_ts=new_ts,
        value=float(vals.get("rv5", float("nan"))), expected_len=expected_len,
        customdata=rv5_hover,
    )
    _append_named_overlay(
        traces, name="rv20", yaxis="y3", new_ts=new_ts,
        value=float(vals.get("rv20", float("nan"))), expected_len=expected_len,
    )
    _append_named_overlay(
        traces, name="vol5_pct_120d", yaxis="y4", new_ts=new_ts,
        value=float(vals.get("vol5_pct_120d", float("nan"))), expected_len=expected_len,
    )
    g20 = float(vals.get("20日差额(实现−公平)", float("nan")))
    g5 = float(vals.get("5日差额(实现−公平)", float("nan")))
    _append_named_overlay(
        traces, name="公平年化收益", yaxis="y5", new_ts=new_ts,
        value=float(vals.get("公平年化收益", float("nan"))), expected_len=expected_len,
        customdata=_need_hover(20, rv5=False),
    )
    _append_named_overlay(
        traces, name="20日差额(实现−公平)", yaxis="y6", new_ts=new_ts,
        value=g20, expected_len=expected_len,
        customdata=_gap_hover(20), bar_color=_gap_bar_color(g20),
    )
    _append_named_overlay(
        traces, name="公平年化收益(rv5)", yaxis="y7", new_ts=new_ts,
        value=float(vals.get("公平年化收益(rv5)", float("nan"))), expected_len=expected_len,
        customdata=_need_hover(5, rv5=True),
    )
    _append_named_overlay(
        traces, name="5日差额(实现−公平)", yaxis="y8", new_ts=new_ts,
        value=g5, expected_len=expected_len,
        customdata=_gap_hover(5), bar_color=_gap_bar_color(g5),
    )
    _extend_fig15_ref_lines(traces, new_ts)


def _append_fig6_nav(
    traces: list[dict[str, Any]],
    *,
    new_ts: pd.Timestamp,
    expected_len: int,
    nav_last: dict[str, float] | None,
) -> None:
    if not nav_last:
        return
    for name, key in (
        ("hold_up净值", "hold_up净值"),
        ("BH净值", "BH净值"),
        ("cash净值", "cash净值"),
    ):
        if key not in nav_last:
            continue
        _append_named_overlay(
            traces,
            name=name,
            yaxis="y9",
            new_ts=new_ts,
            value=float(nav_last[key]),
            expected_len=expected_len,
        )
    policy_v = nav_last.get("nav_policy")
    if policy_v is None:
        return
    for tr in traces:
        if str(tr.get("yaxis") or "") != "y9":
            continue
        if not str(tr.get("name") or "").startswith("本票自选"):
            continue
        xs = _as_1d_list(tr.get("x"))
        ys = _as_1d_list(tr.get("y"))
        if len(xs) != int(expected_len) or len(ys) != int(expected_len):
            continue
        _append_new_x(tr, new_ts)
        _append_scalar(tr, "y", float(policy_v))
        break


def _patch_trail_path_hover(
    prev: str,
    *,
    new_ts: pd.Timestamp,
    path_new: float,
    close_new: float,
    g_new: float,
    g_by: dict[float, float],
    signed_pct: float,
    ewma_pct: float,
    asset_name: str | None,
) -> str:
    from decision_pack.scripts.plot_regime_transition_example import (
        _fmt_signed_path_percentile,
        _fmt_vol,
        fmt_path_gap_hover_line,
    )

    s = prev or ""
    s = re.sub(r"日期=\d{4}-\d{2}-\d{2}", f"日期={new_ts.date()}", s, count=1)
    s = re.sub(r"(诊断\))=[-+\d.]+", rf"\g<1>={path_new:.4f}", s, count=1)
    s = re.sub(
        r"(当日路径年化\([^)]+\)=)-?[\d.]+%",
        lambda m: f"{m.group(1)}{_fmt_vol(g_new)}",
        s,
        count=1,
    )
    pct_txt = f"相对路径因果分位={_fmt_signed_path_percentile(signed_pct)}"
    s = re.sub(r"相对路径因果分位=[^<]+", pct_txt, s, count=1)
    ewma_txt = f"相对路径ewma分位={_fmt_signed_path_percentile(ewma_pct)}"
    if re.search(r"相对路径ewma分位=[^<]+", s):
        s = re.sub(r"相对路径ewma分位=[^<]+", ewma_txt, s, count=1)
    else:
        s = s.replace(pct_txt, f"{pct_txt}<br>{ewma_txt}", 1)
    gap_line = fmt_path_gap_hover_line(asset_name, new_ts, close_new, path_new)
    if re.search(r"<br>[^<]* 约 (?:[+-]\d+%|—)", s):
        s = re.sub(r"<br>[^<]* 约 (?:[+-]\d+%|—)", f"<br>{gap_line}", s, count=1)
    else:
        anchor = ewma_txt if ewma_txt in s else pct_txt
        s = s.replace(anchor, f"{anchor}<br>{gap_line}", 1)
    for years, g in g_by.items():
        hz = _horizon_label(years)
        s = re.sub(
            rf"(双尺度·{re.escape(hz)}路径年化=)-?[\d.]+%",
            lambda m, gg=g: f"{m.group(1)}{_fmt_vol(gg)}",
            s,
        )
    return s


def append_fair_path_bar(
    traces: list[dict[str, Any]],
    *,
    close: pd.Series,
    ohlcv_last: pd.Series,
    new_ts: pd.Timestamp,
    trails: list[float],
    asset_name: str | None = None,
    rv20_anchor: pd.Series | None = None,
    rv5_anchor: pd.Series | None = None,
    nav_last: dict[str, float] | None = None,
    erp_ann: float | None = None,
) -> tuple[dict[float, float], dict[float, float]]:
    """Append last-bar path/rails/candles. Returns g and delta by trail years."""
    from decision_pack.scripts.plot_regime_transition_example import (
        _trail_fit_min_bars,
        causal_path_residual_signed_percentile,
        ewma_path_residual_signed_percentile,
    )

    px_new = float(ohlcv_last["$close"])
    g_by: dict[float, float] = {}
    d_by: dict[float, float] = {}
    candle_payload = dict(
        open=float(ohlcv_last["$open"]),
        high=float(ohlcv_last["$high"]),
        low=float(ohlcv_last["$low"]),
        close=px_new,
    )
    n_prev = None
    fig7 = _fig7_candle(traces)
    if fig7 is not None:
        n_prev = len(_as_1d_list(fig7.get("close")))

    for tr in traces:
        if tr.get("type") != "candlestick":
            continue
        yax = str(tr.get("yaxis") or "y")
        if yax not in CANDLE_YAXES:
            continue
        _append_new_x(tr, new_ts)
        for k, v in candle_payload.items():
            _append_scalar(tr, k, v)

    for years in trails:
        yax = TRAIL_YAXIS.get(float(years))
        if yax is None:
            continue
        min_b = _trail_fit_min_bars(float(years))
        path_new, g_new = _ols_last_path_g(close, trail_years=float(years), min_bars=min_b)
        g_by[float(years)] = g_new
        hz = _horizon_label(float(years))
        path_tr = _find_scatter(traces, yaxis=yax, name_contains="公平路径")
        up_tr = _find_rail(traces, yaxis=yax, hz=hz, upper=True, kind="p95")
        lo_tr = _find_rail(traces, yaxis=yax, hz=hz, upper=False, kind="p95")
        if path_tr is None:
            continue
        path_hist = np.asarray(_as_1d_list(path_tr.get("y")), dtype=float)
        c_tr = None
        for tr in traces:
            if tr.get("type") == "candlestick" and str(tr.get("yaxis") or "") == yax:
                c_tr = tr
                break
        if c_tr is not None:
            close_hist = np.asarray(_as_1d_list(c_tr.get("close"))[:-1], dtype=float)
        else:
            close_hist = np.full_like(path_hist, np.nan)
        n = min(len(path_hist), len(close_hist))
        upper, lower, delta = _new_envelope(
            path_hist[:n],
            close_hist[:n],
            path_new=path_new,
            close_new=px_new,
            min_bars=min_b,
        )
        d_by[float(years)] = delta
        _append_new_x(path_tr, new_ts)
        _append_scalar(path_tr, "y", path_new)
        if up_tr is not None:
            _append_new_x(up_tr, new_ts)
            _append_scalar(up_tr, "y", upper)
        if lo_tr is not None:
            _append_new_x(lo_tr, new_ts)
            _append_scalar(lo_tr, "y", lower)
        _catch_up_ewma_rails(
            traces,
            yaxis=yax,
            hz=hz,
            path_tr=path_tr,
            candle_tr=c_tr,
        )

    for years in trails:
        yax = TRAIL_YAXIS.get(float(years))
        if yax is None:
            continue
        path_tr = _find_scatter(traces, yaxis=yax, name_contains="公平路径")
        if path_tr is None or path_tr.get("customdata") is None:
            continue
        cd = _as_1d_list(path_tr.get("customdata"))
        ys = _as_1d_list(path_tr.get("y"))
        if len(cd) != len(ys) - 1 or not cd:
            continue
        c_tr = None
        for tr in traces:
            if tr.get("type") == "candlestick" and str(tr.get("yaxis") or "") == yax:
                c_tr = tr
                break
        close_arr = (
            np.asarray(_as_1d_list(c_tr.get("close")), dtype=float)
            if c_tr is not None
            else np.full(len(ys), np.nan)
        )
        path_arr = np.asarray(ys, dtype=float)
        n = min(len(path_arr), len(close_arr))
        signed = causal_path_residual_signed_percentile(
            pd.Series(path_arr[:n]),
            pd.Series(close_arr[:n]),
            min_bars=_trail_fit_min_bars(float(years)),
        )
        last_pct = float(signed.iloc[-1]) if len(signed) else float("nan")
        ewma_signed = ewma_path_residual_signed_percentile(
            pd.Series(path_arr[:n]),
            pd.Series(close_arr[:n]),
        )
        last_ewma = float(ewma_signed.iloc[-1]) if len(ewma_signed) else float("nan")
        g_this = float(g_by.get(float(years), float("nan")))
        new_hover = _patch_trail_path_hover(
            _customdata_text(cd[-1]),
            new_ts=new_ts,
            path_new=float(path_arr[-1]) if len(path_arr) else float("nan"),
            close_new=px_new,
            g_new=g_this,
            g_by=g_by,
            signed_pct=last_pct,
            ewma_pct=last_ewma,
            asset_name=asset_name,
        )
        _append_customdata(path_tr, new_hover)

    overlay_len = n_prev
    if overlay_len is None:
        for tr in traces:
            if tr.get("type") != "candlestick":
                continue
            if str(tr.get("yaxis") or "y") == "y":
                n_c = len(_as_1d_list(tr.get("close")))
                overlay_len = max(0, n_c - 1)
                break
    if overlay_len is not None:
        _append_fig1_5_overlays(
            traces,
            close=close,
            ohlcv_last=ohlcv_last,
            new_ts=new_ts,
            expected_len=int(overlay_len),
            rv20_anchor=rv20_anchor,
            rv5_anchor=rv5_anchor,
            erp_ann=erp_ann,
        )
        _append_fig6_nav(
            traces,
            new_ts=new_ts,
            expected_len=int(overlay_len),
            nav_last=nav_last,
        )
    return g_by, d_by


_TRADE_STYLES = {
    "up": dict(bc="#12a150", bl="#064d1f", sc="#d62728", sl="#6b1010", bn="上涨买", sn="上涨卖"),
    "down": dict(bc="#2ca02c", bl="#145214", sc="#ff7f0e", sl="#8c4500", bn="下跌买", sn="下跌卖"),
    "range": dict(bc="#1f77b4", bl="#0b3d5c", sc="#9467bd", sl="#4b2e66", bn="横盘买", sn="横盘卖"),
}


def _add_trade_markers(
    traces: list[dict[str, Any]],
    *,
    aev: list[dict[str, Any]],
    new_date: str,
    ohlcv_last: pd.Series,
    prev_close: pd.Series | None = None,
) -> None:
    """Append same-day B/S to fig1 only, reusing full-build names/styles.

    Full build (``annotate_figure``) draws 上涨买/上涨卖 with B/S text on the
    price panel only; incremental must match — no ``+`` names, no fig2-11 copies.
    """
    new_ev = [e for e in aev if str(e.get("date")) == new_date]
    if not new_ev:
        return
    hi = float(ohlcv_last["$high"])
    lo = float(ohlcv_last["$low"])
    span = max(hi - lo, 1e-6)
    if prev_close is not None:
        pc = pd.to_numeric(prev_close, errors="coerce").dropna()
        if len(pc):
            span = max(float(pc.max() - pc.min()), 1e-6)
    off = span * 0.035
    ts = pd.Timestamp(new_date)
    for e in new_ev:
        st = _TRADE_STYLES.get(str(e.get("regime") or "up"), _TRADE_STYLES["up"])
        is_buy = str(e.get("side")) == "BUY"
        px = float(e.get("px", np.nan))
        if is_buy:
            hover = f"买 {new_date}<br>{e.get('why', '')}<br>价={px:.3f}"
            _append_or_create_marker(
                traces, name=st["bn"], x_ts=ts, y=lo - off,
                symbol="triangle-up", color=st["bc"], line_color=st["bl"],
                size=14, line_width=1.2, text="B", hover=hover,
                textposition="bottom center",
            )
        else:
            ret = e.get("fwd_ret")
            ret_s = f"<br>收益={float(ret):+.1%}" if ret is not None and pd.notna(ret) else ""
            hover = f"卖 {new_date}<br>{e.get('why', '')}<br>价={px:.3f}{ret_s}"
            _append_or_create_marker(
                traces, name=st["sn"], x_ts=ts, y=hi + off,
                symbol="triangle-down", color=st["sc"], line_color=st["sl"],
                size=14, line_width=1.2, text="S", hover=hover,
                textposition="top center",
            )


def _trace_x_append(tr: dict[str, Any], x_ts: pd.Timestamp) -> None:
    """Append x keeping the trace's existing encoding (str date vs epoch ms)."""
    xs = _as_1d_list(tr.get("x"))
    if xs and isinstance(xs[0], (int, float)) and not isinstance(xs[0], bool):
        xs.append(int(x_ts.value // 10**6))
    else:
        xs.append(x_ts.strftime("%Y-%m-%d"))
    tr["x"] = xs


def _append_or_create_marker(
    traces: list[dict[str, Any]],
    *,
    name: str,
    x_ts: pd.Timestamp,
    y: float,
    symbol: str,
    color: str,
    line_color: str,
    size: float,
    line_width: float,
    text: str | None = None,
    hover: str | None = None,
    rail_y: float | None = None,
    textposition: str | None = None,
) -> None:
    """Append a point to the existing fig1 trace named ``name`` or create it.

    Rail traces (switch triangles / confirmation diamonds) keep their constant
    y so the in-HTML rail re-racking JS still treats them as rails.
    """
    for tr in traces:
        if str(tr.get("name") or "") != name:
            continue
        if str(tr.get("yaxis") or "y") not in {"y", "y1"}:
            continue
        _trace_x_append(tr, x_ts)
        ys = _as_1d_list(tr.get("y"))
        ys.append(float(rail_y if rail_y is not None else y))
        tr["y"] = ys
        if "text" in tr:
            tv = _as_1d_list(tr.get("text"))
            tv.append(text if text is not None else "")
            tr["text"] = tv
        if "hovertext" in tr:
            hv = _as_1d_list(tr.get("hovertext"))
            hv.append(hover or "")
            tr["hovertext"] = hv
        if "customdata" in tr:
            cv = _as_1d_list(tr.get("customdata"))
            cv.append(hover or "")
            tr["customdata"] = cv
        return
    tr: dict[str, Any] = dict(
        type="scatter",
        x=[x_ts.strftime("%Y-%m-%d")],
        y=[float(rail_y if rail_y is not None else y)],
        mode="markers+text" if text else "markers",
        name=name,
        marker=dict(
            symbol=symbol,
            size=size,
            color=color,
            line=dict(width=line_width, color=line_color),
        ),
        yaxis="y",
        xaxis="x",
        showlegend=True,
        cliponaxis=False,
    )
    if text:
        tr["text"] = [text]
        tr["textposition"] = textposition or "top center"
        tr["textfont"] = dict(size=10, color=line_color, family="Arial Black")
    if hover:
        tr["hovertext"] = [hover]
        tr["hoverinfo"] = "text"
    traces.append(tr)


# Styles mirror annotate_figure / overlay_marker_specs (full build).
_EARLY_STYLES = {
    "leave_down": dict(name="提前离↓(V)", text="V", color="#e67e22", line="#8a4b08", side="buy"),
    "enter_up": dict(name="提前进↑", text="↑", color="#27ae60", line="#145a32", side="buy"),
    "leave_up": dict(name="提前离↑(X)", text="X", color="#8e44ad", line="#4a235a", side="sell"),
    "enter_down": dict(name="提前进↓", text="↓", color="#c0392b", line="#7b241c", side="sell"),
}
_ED_STYLES = {
    "ed_alert": dict(name="ED告警(模型滞后)", color="#c0392b", line="#641e16"),
    "ed_watch": dict(name="ED观察(已对齐↓)", color="#7f8c8d", line="#2c3e50"),
}
_RU_STYLE = dict(name="RU恢复上涨(诊断)", color="#1abc9c", line="#0e6655")
_SWITCH_STYLES = {
    "up": dict(name="上行切换", symbol="triangle-up", color="#2ca02c"),
    "down": dict(name="下行切换", symbol="triangle-down", color="#d62728"),
}
_DIAMOND_STYLES = {
    "early_reversal_bottom": dict(name="早期底反确认(T+3)", color="#9ecae1", side="buy"),
    "early_reversal_top": dict(name="早期顶反确认(T+3)", color="#fdae6b", side="sell"),
    "final_reversal_bottom": dict(name="最终底反确认(T+10)", color="#1f77b4", side="buy"),
    "final_reversal_top": dict(name="最终顶反确认(T+10)", color="#ff7f0e", side="sell"),
    "reversal_revoked": dict(name="早期确认已撤销", color="#7f7f7f", side="sell"),
}


def _add_newday_overlay_markers(
    traces: list[dict[str, Any]],
    *,
    new_date: str,
    ohlcv_last: pd.Series,
    prev_close: pd.Series,
    early_marks: list[dict[str, Any]] | None,
    ed_marks: list[dict[str, Any]] | None,
    ru_marks: list[dict[str, Any]] | None,
    switch_side: str | None,
    diamonds: list[dict[str, Any]] | None,
) -> None:
    """Draw fig1 overlay markers that fire on the incremental new day.

    Full build draws these from the overlay/adaptive pipeline; incremental mode
    previously appended only B/S trades, so same-day triangles / early marks /
    ED / RU / confirmation diamonds were silently missing.
    """
    ts = pd.Timestamp(new_date)
    hi = float(ohlcv_last["$high"])
    lo = float(ohlcv_last["$low"])
    pc = pd.to_numeric(prev_close, errors="coerce").dropna()
    span = max(float(pc.max() - pc.min()), 1e-6) if len(pc) else max(hi - lo, 1e-6)
    off = span * 0.035
    y_buy, y_sell = lo - off, hi + off

    for m in early_marks or []:
        if str(m.get("date")) != new_date:
            continue
        st = _EARLY_STYLES.get(str(m.get("kind") or ""))
        if st is None:
            continue
        mv = m.get("move", np.nan)
        mv_s = f"{float(mv):+.1%}" if pd.notna(mv) else "?"
        hover = (
            f"{st['name']} {new_date}<br>变动{mv_s}（非交易信号）"
            f"<br>价={float(m.get('px', np.nan)):.3f}<br>{m.get('why', '')}"
        )
        _append_or_create_marker(
            traces, name=st["name"], x_ts=ts,
            y=y_buy if st["side"] == "buy" else y_sell,
            symbol="diamond-open", color=st["color"], line_color=st["line"],
            size=13, line_width=2.0, text=str(m.get("symbol") or st["text"]),
            hover=hover,
        )

    for m in ed_marks or []:
        if str(m.get("date")) != new_date:
            continue
        st = _ED_STYLES.get(str(m.get("kind") or ""))
        if st is None:
            continue
        end_s = str(m.get("end") or new_date)
        hover = (
            f"{st['name']} {new_date}→{end_s}<br>"
            f"n={int(m.get('n_bars') or 1)} · regime={m.get('regime')}<br>"
            f"reasons={m.get('reasons') or '—'}<br>{m.get('why', '')}<br>诊断层·不改买卖"
        )
        _append_or_create_marker(
            traces, name=st["name"], x_ts=ts, y=y_sell + span * 0.01,
            symbol="hexagon", color=st["color"], line_color=st["line"],
            size=14, line_width=1.5, text=str(m.get("symbol") or "ED"),
            hover=hover,
        )

    for m in ru_marks or []:
        if str(m.get("date")) != new_date:
            continue
        end_s = str(m.get("end") or new_date)
        hover = (
            f"{_RU_STYLE['name']} {new_date}→{end_s}<br>"
            f"n={int(m.get('n_bars') or 1)} · regime={m.get('regime')}<br>"
            f"reasons={m.get('reasons') or '—'}<br>{m.get('why', '')}<br>诊断双通道·不改买卖"
        )
        _append_or_create_marker(
            traces, name=_RU_STYLE["name"], x_ts=ts, y=y_sell + span * 0.01,
            symbol="diamond", color=_RU_STYLE["color"], line_color=_RU_STYLE["line"],
            size=14, line_width=1.5, text=str(m.get("symbol") or "RU"),
            hover=hover,
        )

    if switch_side in _SWITCH_STYLES:
        st = _SWITCH_STYLES[switch_side]
        _append_or_create_marker(
            traces, name=st["name"], x_ts=ts,
            y=y_sell if switch_side == "up" else y_buy,
            symbol=st["symbol"], color=st["color"], line_color="#222",
            size=16, line_width=1.5,
            hover=f"日期={new_date}<br>动作={'上行分位切换' if switch_side == 'up' else '下行分位切换'}",
        )

    for d in diamonds or []:
        kind = str(d.get("reversal_marker") or "")
        st = _DIAMOND_STYLES.get(kind)
        if st is None:
            continue
        _append_or_create_marker(
            traces, name=st["name"], x_ts=ts,
            y=y_buy if st["side"] == "buy" else y_sell,
            symbol="diamond-open", color=st["color"], line_color="#222",
            size=16, line_width=1.5,
            hover=str(d.get("hover") or ""),
        )


# Fig1 background fills — keep in lockstep with adaptive_stage_common.apply_adaptive_overlays.
FIG1_REGIME_FILL = {
    "up": "rgba(18,161,80,0.12)",
    "down": "rgba(214,39,40,0.10)",
    "range": "rgba(148,148,148,0.14)",
}
_FIG1_ED_FILL = {
    "ed_alert": "rgba(192,57,43,0.10)",
    "ed_watch": "rgba(127,140,141,0.08)",
}
_FIG1_ED_LINE = {
    "ed_alert": "#641e16",
    "ed_watch": "#2c3e50",
}
_FIG1_RU_FILL = "rgba(26,188,156,0.12)"
_FIG1_RU_LINE = "#0e6655"
_FIG1_REGIME_FILLS = frozenset(FIG1_REGIME_FILL.values())
_FIG1_ED_RU_FILLS = frozenset(_FIG1_ED_FILL.values()) | {_FIG1_RU_FILL}


def _shape_day(v: Any) -> str:
    return pd.Timestamp(v).strftime("%Y-%m-%d")


def _is_fig1_price_rect(shape: Any) -> bool:
    if not isinstance(shape, dict):
        return False
    if str(shape.get("type") or "rect") != "rect":
        return False
    yref = str(shape.get("yref") or "y")
    xref = str(shape.get("xref") or "x")
    return yref in {"y", "y1"} and xref in {"x", "x1"}


def _is_fig1_regime_rect(shape: Any) -> bool:
    return _is_fig1_price_rect(shape) and str(shape.get("fillcolor") or "") in _FIG1_REGIME_FILLS


def _is_fig1_ed_ru_rect(shape: Any) -> bool:
    return _is_fig1_price_rect(shape) and str(shape.get("fillcolor") or "") in _FIG1_ED_RU_FILLS


def _fig1_band_y(ohlcv: pd.DataFrame, old_shapes: list[Any]) -> tuple[float, float]:
    lo = hi = float("nan")
    if "$low" in ohlcv.columns and "$high" in ohlcv.columns:
        lows = pd.to_numeric(ohlcv["$low"], errors="coerce")
        highs = pd.to_numeric(ohlcv["$high"], errors="coerce")
        if len(lows):
            lo = float(lows.min())
        if len(highs):
            hi = float(highs.max())
    if not np.isfinite(lo) or not np.isfinite(hi):
        ys0 = [s.get("y0") for s in old_shapes if _is_fig1_regime_rect(s)]
        ys1 = [s.get("y1") for s in old_shapes if _is_fig1_regime_rect(s)]
        if ys0 and ys1:
            return float(ys0[0]), float(ys1[0])
        return 0.0, 1.0
    span = max(hi - lo, 1e-6)
    return lo - span * 0.02, hi + span * 0.05


def _fig1_rect(
    *,
    start: Any,
    end: Any,
    fillcolor: str,
    y0: float,
    y1: float,
    line: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return dict(
        type="rect",
        xref="x",
        yref="y",
        x0=_shape_day(start),
        x1=_shape_day(end),
        y0=y0,
        y1=y1,
        fillcolor=fillcolor,
        line=line if line is not None else dict(width=0),
        layer="below",
    )


def _clip_span(
    start: Any,
    end: Any,
    *,
    start_date: str | None,
    end_date: str | None,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    ts0, ts1 = pd.Timestamp(start), pd.Timestamp(end)
    if start_date:
        ts0 = max(ts0, pd.Timestamp(start_date))
    if end_date:
        ts1 = min(ts1, pd.Timestamp(end_date))
    if ts1 < ts0:
        return None
    return ts0, ts1


def sync_fig1_regime_shapes(
    layout: dict[str, Any],
    *,
    seg_df: pd.DataFrame | None,
    ohlcv: pd.DataFrame,
    ed_marks: list[dict[str, Any]] | None = None,
    ru_marks: list[dict[str, Any]] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> bool:
    """Rewrite fig1 涨跌/横盘 (and ED/RU) rects from the current book.

    Incremental HTML copies yesterday's ``layout.shapes``. Without this the
    last regime band stays frozen at the last full-rebuild date.
    """
    if seg_df is None or getattr(seg_df, "empty", True):
        return False
    old = list(layout.get("shapes") or [])
    y0, y1 = _fig1_band_y(ohlcv, old)
    rebuild_bands = ed_marks is not None or ru_marks is not None
    kept: list[Any] = []
    for s in old:
        if _is_fig1_regime_rect(s):
            continue
        if rebuild_bands and _is_fig1_ed_ru_rect(s):
            continue
        kept.append(s)
    new_rects: list[dict[str, Any]] = []
    for _, row in seg_df.iterrows():
        fill = FIG1_REGIME_FILL.get(str(row.get("regime") or ""))
        if fill is None:
            continue
        span = _clip_span(
            row.get("start"), row.get("end"), start_date=start_date, end_date=end_date
        )
        if span is None:
            continue
        new_rects.append(
            _fig1_rect(start=span[0], end=span[1], fillcolor=fill, y0=y0, y1=y1)
        )
    if rebuild_bands:
        for m in ed_marks or []:
            kind = str(m.get("kind") or "ed_alert")
            fill = _FIG1_ED_FILL.get(kind) or _FIG1_ED_FILL["ed_alert"]
            line_c = _FIG1_ED_LINE.get(kind) or _FIG1_ED_LINE["ed_alert"]
            d0 = m.get("date") or ""
            d1 = m.get("end") or d0
            span = _clip_span(d0, d1, start_date=start_date, end_date=end_date)
            if span is None:
                continue
            new_rects.append(
                _fig1_rect(
                    start=span[0],
                    end=span[1],
                    fillcolor=fill,
                    y0=y0,
                    y1=y1,
                    line=dict(width=1.0, color=line_c, dash="dot"),
                )
            )
        for m in ru_marks or []:
            d0 = m.get("date") or ""
            d1 = m.get("end") or d0
            span = _clip_span(d0, d1, start_date=start_date, end_date=end_date)
            if span is None:
                continue
            new_rects.append(
                _fig1_rect(
                    start=span[0],
                    end=span[1],
                    fillcolor=_FIG1_RU_FILL,
                    y0=y0,
                    y1=y1,
                    line=dict(width=1.0, color=_FIG1_RU_LINE, dash="dot"),
                )
            )
    new_shapes = kept + new_rects
    old_regime_x1 = [
        _shape_day(s.get("x1"))
        for s in old
        if _is_fig1_regime_rect(s)
    ]
    new_regime_x1 = [
        _shape_day(s.get("x1"))
        for s in new_rects
        if str(s.get("fillcolor") or "") in _FIG1_REGIME_FILLS
    ]
    changed = (max(old_regime_x1, default="") != max(new_regime_x1, default="")) or (
        len(old_regime_x1) != len(new_regime_x1)
    )
    if rebuild_bands:
        old_band_n = sum(1 for s in old if _is_fig1_ed_ru_rect(s))
        new_band_n = sum(
            1 for s in new_rects if str(s.get("fillcolor") or "") in _FIG1_ED_RU_FILLS
        )
        if old_band_n != new_band_n:
            changed = True
        else:
            old_band_x1 = max(
                (_shape_day(s.get("x1")) for s in old if _is_fig1_ed_ru_rect(s)),
                default="",
            )
            new_band_x1 = max(
                (
                    _shape_day(s.get("x1"))
                    for s in new_rects
                    if str(s.get("fillcolor") or "") in _FIG1_ED_RU_FILLS
                ),
                default="",
            )
            if old_band_x1 != new_band_x1:
                changed = True
    if not changed:
        return False
    layout["shapes"] = new_shapes
    return True


def try_incremental_html(
    *,
    code: str,
    prev_dir: Path,
    out_dir: Path,
    ohlcv: pd.DataFrame,
    end_date: str,
    close_tol: float,
    extra_trail_years: str | None,
    aev: list[dict[str, Any]] | None = None,
    display_name: str | None = None,
    rv20_anchor: pd.Series | None = None,
    rv5_anchor: pd.Series | None = None,
    nav_last: dict[str, float] | None = None,
    erp_ann: float | None = None,
    early_marks: list[dict[str, Any]] | None = None,
    ed_marks: list[dict[str, Any]] | None = None,
    ru_marks: list[dict[str, Any]] | None = None,
    switch_side: str | None = None,
    diamonds: list[dict[str, Any]] | None = None,
    seg_df: pd.DataFrame | None = None,
) -> tuple[Path | None, str]:
    """Patch yesterday HTML. Returns (out_path, reason). reason ok|same|*fallback*."""
    prev = find_listing_html(prev_dir, code)
    if prev is None or not prev.exists():
        return None, "no_prev_html"
    try:
        prev_close, _ = load_ohlcv_from_regime_html(prev)
    except Exception:
        return None, "prev_html_ohlcv"
    work = ohlcv.copy()
    if "datetime" not in work.columns:
        work = work.reset_index()
        if "datetime" not in work.columns and "index" in work.columns:
            work = work.rename(columns={"index": "datetime"})
        if "datetime" not in work.columns:
            work = work.rename(columns={work.columns[0]: "datetime"})
    new_close = work.set_index(pd.to_datetime(work["datetime"]).dt.normalize())["$close"]
    new_close = pd.to_numeric(new_close, errors="coerce")
    new_close = new_close[~new_close.index.duplicated(keep="last")].sort_index()
    ok, why = overlap_close_ok(prev_close, new_close, tol=close_tol)
    if not ok:
        return None, why

    html_txt = prev.read_text(encoding="utf-8")
    try:
        traces = _extract_plotly_traces(html_txt)
        layout = extract_layout(html_txt)
    except Exception:
        return None, "parse_html"

    trails = _expected_trails(extra_trail_years)
    if not traces_have_fair_panels(traces, trails):
        return None, "missing_panels"

    prev_last = pd.Timestamp(prev_close.index.max()).normalize()
    gap = classify_date_gap(prev_last, new_close.index)
    safe = (display_name or code).replace("/", "")
    start_in_name = prev.name.split("_")
    # regime_transition_CODE_NAME_START_END_adaptive.html — keep start from prev
    out_name = bump_html_end_tag(prev.name, end_date)
    out_path = out_dir / out_name
    plot_start = pd.to_datetime(work["datetime"]).min().strftime("%Y-%m-%d")

    def _sync_fig1_shapes() -> bool:
        return sync_fig1_regime_shapes(
            layout,
            seg_df=seg_df,
            ohlcv=work,
            ed_marks=ed_marks,
            ru_marks=ru_marks,
            start_date=plot_start,
            end_date=end_date,
        )

    if gap == "same":
        changed = _catch_up_all_ewma_rails(traces, trails)
        changed = _sync_fig1_shapes() or changed
        if changed:
            fig = go.Figure(data=traces, layout=layout)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            write_adaptive_html(fig, out_path)
            return out_path, "ok"
        shutil.copy2(prev, out_path)
        return out_path, "same"
    if gap != "plus_one":
        return None, gap

    new_ts = pd.Timestamp(new_close.index[-1]).normalize()
    ohlcv = work.copy()
    ohlcv["_d"] = pd.to_datetime(ohlcv["datetime"]).dt.normalize()
    last_row = ohlcv.loc[ohlcv["_d"].eq(new_ts)].iloc[-1]
    g_by, d_by = append_fair_path_bar(
        traces,
        close=new_close,
        ohlcv_last=last_row,
        new_ts=new_ts,
        trails=trails,
        asset_name=display_name,
        rv20_anchor=rv20_anchor,
        rv5_anchor=rv5_anchor,
        nav_last=nav_last,
        erp_ann=erp_ann,
    )
    _update_title_g(layout, g_by, d_by)
    if aev:
        _add_trade_markers(
            traces,
            aev=aev,
            new_date=new_ts.strftime("%Y-%m-%d"),
            ohlcv_last=last_row,
            prev_close=prev_close,
        )
    _add_newday_overlay_markers(
        traces,
        new_date=new_ts.strftime("%Y-%m-%d"),
        ohlcv_last=last_row,
        prev_close=prev_close,
        early_marks=early_marks,
        ed_marks=ed_marks,
        ru_marks=ru_marks,
        switch_side=switch_side,
        diamonds=diamonds,
    )
    _sync_fig1_shapes()
    fig = go.Figure(data=traces, layout=layout)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_adaptive_html(fig, out_path)
    _ = safe
    _ = start_in_name
    return out_path, "ok"


def write_book_csvs(
    *,
    pool: Any,
    code: str,
    cfg: dict,
    ohlcv: pd.DataFrame,
    start_date: str,
    end_date: str,
    out_dir: Path,
    ev: Any,
    hrp_mode_by_code: dict[str, str] | None,
    enable_ru_diag: bool,
    enable_shallow_up_trade: bool,
    name: str,
) -> dict[str, Any]:
    """Regime labels + hold_up book without rebuilding the figure."""
    import numpy as np
    from decision_pack.src.adaptive_stage_common import (
        FIXED_HYBRID_RULES,
        detect_early_regime_marks,
        prepare_features,
        segments,
        simulate_adaptive_book,
    )
    from decision_pack.src.early_down_alert import detect_early_down_marks
    from decision_pack.src.early_down_alert import early_down_flags_frame
    from decision_pack.src.hrp_cluster_router import mode_allows_trade_overlay
    from decision_pack.src.recovery_up_overlay import (
        DIAGNOSTIC_N,
        SHALLOW_N,
        SHALLOW_P20_MIN_DEFAULT,
        apply_recovery_up,
        detect_recovery_up_marks,
        ed_alert_bool_from_frame,
        latch_recovery_up_mask,
        recovery_up_mask,
    )
    from decision_pack.src.self_train_policy import (
        config_block as self_train_config_block,
        evaluate_self_train_policy,
        policy_frame_from_eval,
    )

    method = cfg["method"]
    method_params = cfg.get("method_params") or {}
    px = prepare_features(ev, ohlcv, method=method, method_params=method_params)
    native_labs = np.asarray(px.regime.to_numpy(), dtype=object)
    mode = (hrp_mode_by_code or {}).get(str(code).upper(), "default")
    if enable_shallow_up_trade and mode_allows_trade_overlay(mode):
        ed_full = early_down_flags_frame(px, regimes=native_labs)
        ed_bool, ed_ok = ed_alert_bool_from_frame(ed_full, len(px))
        su_mask = recovery_up_mask(
            px, native_labs, ed_bool if ed_ok else None, N=SHALLOW_N, p20_min=SHALLOW_P20_MIN_DEFAULT
        )
        if ed_ok:
            su_mask = latch_recovery_up_mask(px, native_labs, su_mask, ed_bool)
        px = px.copy()
        px["regime"] = apply_recovery_up(native_labs, su_mask)
    px_w = px[(px.as_of >= start_date) & (px.as_of <= end_date)].reset_index(drop=True)
    if len(px_w) < 5:
        raise ValueError(f"too few OOS bars {len(px_w)}")
    close_w = px_w["$close"].to_numpy(float)
    dates = px_w.as_of.dt.strftime("%Y-%m-%d").to_numpy()
    seg_df = segments(px_w.regime.to_numpy(), dates, close_w)
    baseline_method = str((cfg.get("method_select_meta") or {}).get("baseline") or method)
    if baseline_method in ev.METHODS:
        baseline_full = ev.run_method(baseline_method, px)
        window_mask = ((px.as_of >= start_date) & (px.as_of <= end_date)).to_numpy()
        baseline_w = np.asarray(baseline_full, dtype=object)[window_mask]
    else:
        baseline_w = px_w.regime.to_numpy(object)
    baseline_seg_df = segments(baseline_w, dates, close_w)
    baseline_diff = float(np.mean(baseline_w != px_w.regime.to_numpy(object)))
    aev, stats, _open_mtm = simulate_adaptive_book(px_w, rules=pool._rules_for_sim(cfg))
    px_fix = prepare_features(ev, ohlcv, method=FIXED_HYBRID_RULES["method"])
    px_fx_w = px_fix[(px_fix.as_of >= start_date) & (px_fix.as_of <= end_date)].reset_index(drop=True)
    _, stats_fx, _ = simulate_adaptive_book(
        px_fx_w, rules=pool._rules_for_sim({"rules": FIXED_HYBRID_RULES})
    )
    train_cutoff = str(cfg.get("train_cutoff") or start_date)
    stp_eval = evaluate_self_train_policy(
        px["as_of"],
        px["$close"].to_numpy(dtype=float),
        px["regime"].to_numpy(dtype=object),
        train_cutoff=train_cutoff,
    )
    cfg["self_train_policy"] = self_train_config_block(stp_eval)
    policy_frame = policy_frame_from_eval(stp_eval)
    lo = pd.Timestamp(start_date).normalize()
    hi = pd.Timestamp(end_date).normalize()
    policy_frame = policy_frame.copy()
    policy_frame["as_of"] = pd.to_datetime(policy_frame["as_of"]).dt.normalize()
    policy_frame = policy_frame[
        (policy_frame["as_of"] >= lo) & (policy_frame["as_of"] <= hi)
    ]
    nav_last: dict[str, float] = {}
    if not policy_frame.empty:
        last_nav = policy_frame.iloc[-1]
        nav_last = {
            "hold_up净值": float(last_nav.get("nav_hold_up", float("nan"))),
            "BH净值": float(last_nav.get("nav_bh", float("nan"))),
            "cash净值": float(last_nav.get("nav_cash", float("nan"))),
            "nav_policy": float(last_nav.get("nav_policy", float("nan"))),
        }
    agg_marks = detect_early_regime_marks(px_w, regimes=px_w.regime.to_numpy())
    ed_marks = detect_early_down_marks(px_w, regimes=px_w.regime.to_numpy())
    ru_marks: list[dict] = []
    if enable_ru_diag:
        try:
            ed_full = early_down_flags_frame(px, regimes=native_labs)
            ed_bool, ed_ok = ed_alert_bool_from_frame(ed_full, len(px))
            ru_all = detect_recovery_up_marks(
                px, native_labs, ed_bool if ed_ok else None, N=DIAGNOSTIC_N
            )
            lo_ts, hi_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
            for m in ru_all:
                d0 = pd.Timestamp(str(m.get("date") or ""))
                d1 = pd.Timestamp(str(m.get("end") or m.get("date") or ""))
                if d1 < lo_ts or d0 > hi_ts:
                    continue
                m2 = dict(m)
                m2["hrp_mode"] = mode
                ru_marks.append(m2)
        except Exception:
            ru_marks = []

    trades_dir = out_dir / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(aev).to_csv(trades_dir / f"{code}_trades.csv", index=False, encoding="utf-8-sig")
    seg_df.to_csv(trades_dir / f"{code}_segments.csv", index=False, encoding="utf-8-sig")
    baseline_seg_df.to_csv(
        trades_dir / f"{code}_baseline_segments.csv", index=False, encoding="utf-8-sig"
    )
    if agg_marks:
        pd.DataFrame(agg_marks).to_csv(
            trades_dir / f"{code}_early_regime_marks.csv", index=False, encoding="utf-8-sig"
        )
    if ed_marks:
        pd.DataFrame(ed_marks).to_csv(
            trades_dir / f"{code}_early_down_marks.csv", index=False, encoding="utf-8-sig"
        )
    if ru_marks:
        pd.DataFrame(ru_marks).to_csv(
            trades_dir / f"{code}_recovery_up_marks.csv", index=False, encoding="utf-8-sig"
        )
    ru = cfg["rules"]
    cfg_out = dict(cfg)
    cfg_out["oos"] = dict(
        start=start_date,
        end=end_date,
        compound=stats["compound"],
        bh=stats["bh"],
        edge=stats["edge"],
        n_closed=stats["n"],
        fixed_hybrid_edge=stats_fx["edge"],
    )
    (out_dir / "configs").mkdir(parents=True, exist_ok=True)
    (out_dir / "configs" / f"{code}.json").write_text(
        json.dumps(cfg_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return dict(
        code=code,
        name=name,
        out="",
        method=method,
        train_end=cfg.get("train_end"),
        train_bars=cfg.get("train_bars"),
        skipped_train=cfg.get("skipped_train"),
        up_buy=ru["up"]["buy"],
        up_exit=ru["up"]["exit"],
        up_n=ru["up"].get("n_trades"),
        up_post=ru["up"].get("posterior"),
        down_buy=ru["down"]["buy"],
        down_exit=ru["down"]["exit"],
        down_n=ru["down"].get("n_trades"),
        range_buy=ru["range"]["buy"],
        range_exit=ru["range"]["exit"],
        range_n=ru["range"].get("n_trades"),
        bars=len(px_w),
        compound=stats["compound"],
        bh=stats["bh"],
        edge=stats["edge"],
        n_closed=stats["n"],
        win=stats["win"],
        open_pos=stats["open_pos"],
        open_unrealized=stats["open_unrealized"],
        fixed_hybrid_edge=stats_fx["edge"],
        edge_vs_fixed=float(stats["edge"] - stats_fx["edge"]),
        n_seg=len(seg_df),
        baseline_method=baseline_method,
        baseline_regime_diff=baseline_diff,
        aev=aev,
        nav_last=nav_last,
        agg_marks=agg_marks,
        ed_marks=ed_marks,
        ru_marks=ru_marks,
        seg_df=seg_df,
    )
