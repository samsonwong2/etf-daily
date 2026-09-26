"""Incremental from_listing HTML: overlap/qfq gate and last-bar fair path."""
from __future__ import annotations

from pathlib import Path

from runtime_paths import PLOTLY_OUTPUTS_DIR

import numpy as np
import pandas as pd
import pytest

from decision_pack.src.fair_path_band_signals import (
    _extract_plotly_traces,
    load_ohlcv_from_regime_html,
)
from decision_pack.src.incremental_from_listing import (
    FIG1_REGIME_FILL,
    _append_new_x,
    _as_1d_list,
    _catch_up_ewma_rails,
    _fig7_candle,
    _find_named,
    _find_rail,
    _find_scatter,
    _horizon_label,
    _is_fig1_regime_rect,
    _ols_last_path_g,
    _trace_dates,
    _patch_trail_path_hover,
    append_fair_path_bar,
    classify_date_gap,
    extract_layout,
    find_listing_html,
    overlap_close_ok,
    sync_fig1_regime_shapes,
    try_incremental_html,
)

_LISTING_27 = (
    PLOTLY_OUTPUTS_DIR / "20260827_from_listing"
)
_EXTRA = "1,0.5,0.25,1/12"
_TRAILS = [2.0, 1.0, 0.5, 0.25, 1.0 / 12.0]


def _sz159502_html() -> Path:
    html = find_listing_html(_LISTING_27, "SZ159502")
    if html is None or not html.exists():
        pytest.skip("SZ159502 from_listing HTML missing")
    return html


def test_overlap_close_ok_and_qfq():
    idx = pd.date_range("2024-01-02", periods=20, freq="B")
    a = pd.Series(np.linspace(1.0, 1.2, len(idx)), index=idx)
    b = a.copy()
    ok, why = overlap_close_ok(a, b, tol=1e-6)
    assert ok and why == "ok"
    b.iloc[-1] *= 1.01
    ok, why = overlap_close_ok(a, b, tol=1e-6)
    assert not ok and why == "qfq"


def test_append_new_x_keeps_existing_iso_datetime_shape():
    tr = {"x": ["2026-08-27T00:00:00.000000000", "2026-08-28T00:00:00.000000000"]}
    _append_new_x(tr, pd.Timestamp("2026-08-31"))
    assert tr["x"][-1] == "2026-08-31T00:00:00.000000000"


def test_classify_date_gap():
    idx = pd.DatetimeIndex(pd.date_range("2024-01-02", periods=5, freq="B"))
    assert classify_date_gap(idx[-2], idx) == "plus_one"
    assert classify_date_gap(idx[-1], idx) == "same"
    extra = pd.DatetimeIndex([idx[-1] + pd.Timedelta(days=3), idx[-1] + pd.Timedelta(days=4)])
    wider = idx.append(extra)
    assert classify_date_gap(idx[-1], wider) == "gap"


def test_ols_last_matches_full_trail():
    from decision_pack.scripts.plot_regime_transition_example import (
        _trail_fit_min_bars,
        trailing_log_trend_price_path,
    )

    rng = np.random.default_rng(0)
    n = 400
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))
    years = 0.5
    min_b = _trail_fit_min_bars(years)
    path, g = trailing_log_trend_price_path(px, trail_years=years, min_bars=min_b)
    last_p, last_g = _ols_last_path_g(px, trail_years=years, min_bars=min_b)
    assert np.isfinite(last_p) and np.isfinite(path.iloc[-1])
    assert abs(last_p / float(path.iloc[-1]) - 1.0) < 1e-8
    assert abs(last_g - float(g.iloc[-1])) < 1e-10


def _pop_len(traces: list, n: int) -> None:
    for tr in traces:
        for f in ("x", "y", "open", "high", "low", "close"):
            if f not in tr or tr[f] is None:
                continue
            vals = _as_1d_list(tr[f])
            if len(vals) == n:
                tr[f] = vals[:-1]


def _last2(arr) -> tuple[float, float]:
    v = _as_1d_list(arr)
    s = pd.to_numeric(pd.Series(v, dtype=object), errors="coerce")
    return float(s.iloc[-2]), float(s.iloc[-1])


def test_append_last_bar_matches_sz159502_html_last_two_days():
    html = _sz159502_html()
    text = html.read_text(encoding="utf-8")
    orig = _extract_plotly_traces(text)
    fig70 = _fig7_candle(orig)
    assert fig70 is not None
    n = len(_as_1d_list(fig70.get("close")))
    assert n > 20
    orig_path9 = _find_scatter(orig, yaxis="y12", name_contains="公平路径")
    orig_up9 = _find_scatter(orig, yaxis="y12", name_contains=f"上轨{_horizon_label(0.5)}")
    orig_lo9 = _find_scatter(orig, yaxis="y12", name_contains=f"下轨{_horizon_label(0.5)}")
    assert orig_path9 is not None and orig_up9 is not None and orig_lo9 is not None
    exp_c = _last2(fig70["close"])
    exp_p = _last2(orig_path9["y"])
    exp_u = _last2(orig_up9["y"])
    exp_l = _last2(orig_lo9["y"])

    traces = _extract_plotly_traces(text)
    fig7 = _fig7_candle(traces)
    _pop_len(traces, n)
    close_full, ohlcv = load_ohlcv_from_regime_html(html)
    new_ts = pd.Timestamp(close_full.index[-1]).normalize()
    last = ohlcv.iloc[-1]
    append_fair_path_bar(
        traces,
        close=close_full,
        ohlcv_last=last,
        new_ts=new_ts,
        trails=_TRAILS,
    )
    fig7b = _fig7_candle(traces)
    path9 = _find_scatter(traces, yaxis="y12", name_contains="公平路径")
    up9 = _find_scatter(traces, yaxis="y12", name_contains=f"上轨{_horizon_label(0.5)}")
    lo9 = _find_scatter(traces, yaxis="y12", name_contains=f"下轨{_horizon_label(0.5)}")
    got_c = _last2(fig7b["close"])
    got_p = _last2(path9["y"])
    got_u = _last2(up9["y"])
    got_l = _last2(lo9["y"])
    for a, b, tol in (
        (got_c[0], exp_c[0], 1e-8),
        (got_c[1], exp_c[1], 1e-8),
        (got_p[0], exp_p[0], 1e-4),
        (got_p[1], exp_p[1], 1e-4),
        (got_u[0], exp_u[0], 2e-3),
        (got_u[1], exp_u[1], 2e-3),
        (got_l[0], exp_l[0], 2e-3),
        (got_l[1], exp_l[1], 2e-3),
    ):
        assert abs(a / b - 1.0) < tol, (a, b, fig7)
    _ = fig7


def test_catch_up_ewma_rails_fills_short_traces():
    from decision_pack.src.fair_path_research import ewma_scaled_envelope

    n = 80
    rng = np.random.default_rng(0)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    path = close * 1.001
    xs = pd.bdate_range("2024-01-02", periods=n).strftime("%Y-%m-%d").tolist()
    eu, el, _ = ewma_scaled_envelope(pd.Series(path), pd.Series(close))
    traces = [
        {
            "type": "candlestick",
            "yaxis": "y12",
            "x": list(xs),
            "close": close.tolist(),
            "open": close.tolist(),
            "high": (close * 1.01).tolist(),
            "low": (close * 0.99).tolist(),
        },
        {
            "type": "scatter",
            "yaxis": "y12",
            "name": "公平路径(自有log趋势·6个月 g≈1%)",
            "x": list(xs),
            "y": path.tolist(),
        },
        {
            "type": "scatter",
            "yaxis": "y12",
            "name": "上轨6个月 EWMA残差轨 +5%(当日)",
            "x": xs[:-3],
            "y": eu.iloc[:-3].tolist(),
            "customdata": ["h"] * (n - 3),
        },
        {
            "type": "scatter",
            "yaxis": "y12",
            "name": "下轨6个月 EWMA残差轨 -5%(当日)",
            "x": xs[:-3],
            "y": el.iloc[:-3].tolist(),
            "customdata": ["h"] * (n - 3),
        },
    ]
    _catch_up_ewma_rails(
        traces,
        yaxis="y12",
        hz="6个月",
        path_tr=traces[1],
        candle_tr=traces[0],
    )
    up = _find_rail(traces, yaxis="y12", hz="6个月", upper=True, kind="ewma")
    lo = _find_rail(traces, yaxis="y12", hz="6个月", upper=False, kind="ewma")
    assert up is not None and lo is not None
    assert len(_as_1d_list(up["y"])) == n
    assert len(_as_1d_list(lo["y"])) == n
    assert abs(float(up["y"][-1]) / float(eu.iloc[-1]) - 1.0) < 1e-9
    assert abs(float(lo["y"][-1]) / float(el.iloc[-1]) - 1.0) < 1e-9
    assert _as_1d_list(up["x"])[-1] == xs[-1]


def test_append_catches_up_ewma_on_from_listing_html():
    listing = PLOTLY_OUTPUTS_DIR / "20260915_from_listing"
    html = find_listing_html(listing, "SH510300")
    if html is None or not html.exists():
        pytest.skip("SH510300 20260915 from_listing HTML missing")
    text = html.read_text(encoding="utf-8")
    traces = _extract_plotly_traces(text)
    fig7 = _fig7_candle(traces)
    n = len(_as_1d_list(fig7.get("close")))
    hz = _horizon_label(0.5)
    ewma_up0 = _find_rail(traces, yaxis="y12", hz=hz, upper=True, kind="ewma")
    if ewma_up0 is None:
        pytest.skip("fig9 EWMA rail missing")
    n_ewma = len(_as_1d_list(ewma_up0.get("y")))
    if n_ewma >= n:
        pytest.skip("EWMA already same length as candles")
    _pop_len(traces, n)
    close_full, ohlcv = load_ohlcv_from_regime_html(html)
    new_ts = pd.Timestamp(close_full.index[-1]).normalize()
    last = ohlcv.iloc[-1]
    append_fair_path_bar(
        traces,
        close=close_full,
        ohlcv_last=last,
        new_ts=new_ts,
        trails=_TRAILS,
    )
    fig7b = _fig7_candle(traces)
    n_after = len(_as_1d_list(fig7b.get("close")))
    assert n_after == n
    for years, yax in ((2.0, "y10"), (1.0, "y11"), (0.5, "y12"), (0.25, "y13")):
        h = _horizon_label(years)
        up = _find_rail(traces, yaxis=yax, hz=h, upper=True, kind="ewma")
        lo = _find_rail(traces, yaxis=yax, hz=h, upper=False, kind="ewma")
        assert up is not None and lo is not None
        assert len(_as_1d_list(up["y"])) == n_after
        assert len(_as_1d_list(lo["y"])) == n_after
        last_d = _trace_dates(up)
        assert str(last_d[-1].date()) == str(new_ts.date())


def test_try_incremental_qfq_fallback(tmp_path: Path):
    src = _sz159502_html()
    prev_dir = tmp_path / "prev"
    prev_dir.mkdir()
    dest = prev_dir / src.name
    dest.write_bytes(src.read_bytes())
    _close, ohlcv = load_ohlcv_from_regime_html(src)
    df = ohlcv.reset_index()
    if "datetime" not in df.columns:
        df = df.rename(columns={df.columns[0]: "datetime"})
    df.loc[df.index[-1], "$close"] = float(df["$close"].iloc[-1]) * 1.01
    out, reason = try_incremental_html(
        code="SZ159502",
        prev_dir=prev_dir,
        out_dir=tmp_path / "out",
        ohlcv=df,
        end_date="2026-08-27",
        close_tol=1e-6,
        extra_trail_years=_EXTRA,
    )
    assert out is None
    assert reason == "qfq"


def test_try_incremental_html_roundtrip_last_two(tmp_path: Path):
    import plotly.graph_objects as go

    from decision_pack.src.plotly_html_autoscale import write_adaptive_html

    src = _sz159502_html()
    text = src.read_text(encoding="utf-8")
    traces = _extract_plotly_traces(text)
    layout = extract_layout(text)
    fig7 = _fig7_candle(traces)
    n = len(_as_1d_list(fig7["close"]))
    _pop_len(traces, n)
    prev_dir = tmp_path / "prev"
    prev_dir.mkdir()
    stripped = prev_dir / src.name
    write_adaptive_html(go.Figure(data=traces, layout=layout), stripped)

    close_full, ohlcv = load_ohlcv_from_regime_html(src)
    df = ohlcv.reset_index()
    if "datetime" not in df.columns:
        df = df.rename(columns={df.columns[0]: "datetime"})
    out_dir = tmp_path / "out"
    html_out, reason = try_incremental_html(
        code="SZ159502",
        prev_dir=prev_dir,
        out_dir=out_dir,
        ohlcv=df,
        end_date=pd.Timestamp(close_full.index[-1]).strftime("%Y-%m-%d"),
        close_tol=1e-6,
        extra_trail_years=_EXTRA,
    )
    assert reason == "ok" and html_out is not None and html_out.exists()

    orig = _extract_plotly_traces(text)
    got = _extract_plotly_traces(html_out.read_text(encoding="utf-8"))
    o7 = _fig7_candle(orig)
    g7 = _fig7_candle(got)
    o9 = _find_scatter(orig, yaxis="y12", name_contains="公平路径")
    g9 = _find_scatter(got, yaxis="y12", name_contains="公平路径")
    ou = _find_scatter(orig, yaxis="y12", name_contains=f"上轨{_horizon_label(0.5)}")
    gu = _find_scatter(got, yaxis="y12", name_contains=f"上轨{_horizon_label(0.5)}")
    ol = _find_scatter(orig, yaxis="y12", name_contains=f"下轨{_horizon_label(0.5)}")
    gl = _find_scatter(got, yaxis="y12", name_contains=f"下轨{_horizon_label(0.5)}")
    assert g9 is not None and gu is not None and gl is not None
    for label, got_tr, exp_tr, tol in (
        ("close", g7["close"], o7["close"], 1e-6),
        ("path9", g9["y"], o9["y"], 1e-4),
        ("up9", gu["y"], ou["y"], 2e-3),
        ("lo9", gl["y"], ol["y"], 2e-3),
    ):
        got_pair = _last2(got_tr)
        exp_pair = _last2(exp_tr)
        assert abs(got_pair[0] / exp_pair[0] - 1.0) < tol, (label, got_pair, exp_pair)
        assert abs(got_pair[1] / exp_pair[1] - 1.0) < tol, (label, got_pair, exp_pair)


def _anchor_rv_from_listing(listing_dir: Path) -> tuple[pd.Series, pd.Series]:
    from decision_pack.src.regime_transition_plot import compute_volatility_regime_frame

    html = find_listing_html(listing_dir, "SH510300")
    if html is None or not html.exists():
        pytest.skip("SH510300 from_listing HTML missing")
    close, _ = load_ohlcv_from_regime_html(html)
    vf = compute_volatility_regime_frame(close)
    vf["as_of"] = pd.to_datetime(vf["as_of"]).dt.normalize()
    indexed = vf.set_index("as_of")
    return indexed["rv20"], indexed["rv5"]


def test_append_fig1_5_last_bar_not_copied_from_prev_day():
    """Fig1–5 last overlay must be recomputed, not yesterday's y copied forward."""
    html = _sz159502_html()
    text = html.read_text(encoding="utf-8")
    orig = _extract_plotly_traces(text)
    fig70 = _fig7_candle(orig)
    n = len(_as_1d_list(fig70.get("close")))
    ma5_o = _find_named(orig, name="MA5", yaxis="y")
    vol_o = _find_named(orig, name="成交量", yaxis="y2")
    rv5_o = _find_named(orig, name="rv5", yaxis="y3")
    gap20_o = _find_named(orig, name="20日差额(实现−公平)", yaxis="y6")
    assert ma5_o is not None and vol_o is not None and rv5_o is not None
    exp_ma5 = _last2(ma5_o["y"])
    exp_vol = _last2(vol_o["y"])
    exp_rv5 = _last2(rv5_o["y"])
    exp_gap20 = _last2(gap20_o["y"]) if gap20_o is not None else None
    # Copy-forward would make last == prev. Real 8/27 vs 8/26 should differ.
    assert exp_ma5[0] != exp_ma5[1]
    assert exp_vol[0] != exp_vol[1]
    assert exp_rv5[0] != exp_rv5[1]

    traces = _extract_plotly_traces(text)
    _pop_len(traces, n)
    close_full, ohlcv = load_ohlcv_from_regime_html(html)
    ohlcv = ohlcv.copy()
    ohlcv["$volume"] = np.asarray(_as_1d_list(vol_o["y"]), dtype=float)
    new_ts = pd.Timestamp(close_full.index[-1]).normalize()
    last = ohlcv.iloc[-1]
    rv20_a, rv5_a = _anchor_rv_from_listing(_LISTING_27)
    append_fair_path_bar(
        traces,
        close=close_full,
        ohlcv_last=last,
        new_ts=new_ts,
        trails=_TRAILS,
        rv20_anchor=rv20_a,
        rv5_anchor=rv5_a,
    )
    ma5_g = _find_named(traces, name="MA5", yaxis="y")
    vol_g = _find_named(traces, name="成交量", yaxis="y2")
    rv5_g = _find_named(traces, name="rv5", yaxis="y3")
    gap20_g = _find_named(traces, name="20日差额(实现−公平)", yaxis="y6")
    got_ma5 = _last2(ma5_g["y"])
    got_vol = _last2(vol_g["y"])
    got_rv5 = _last2(rv5_g["y"])
    assert abs(got_ma5[1] / exp_ma5[1] - 1.0) < 1e-8, (got_ma5, exp_ma5)
    assert abs(got_vol[1] / exp_vol[1] - 1.0) < 1e-8, (got_vol, exp_vol)
    assert abs(got_rv5[1] / exp_rv5[1] - 1.0) < 1e-8, (got_rv5, exp_rv5)
    assert got_ma5[1] != got_ma5[0]
    assert got_vol[1] != got_vol[0]
    if exp_gap20 is not None and gap20_g is not None:
        got_gap = _last2(gap20_g["y"])
        assert abs(got_gap[1] - exp_gap20[1]) < 2e-4, (got_gap, exp_gap20)


def _stale_fig1_layout() -> dict:
    gray = FIG1_REGIME_FILL["range"]
    red = FIG1_REGIME_FILL["down"]
    return {
        "shapes": [
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0="2026-08-13",
                x1="2026-09-07",
                fillcolor=red,
                line=dict(width=0),
                layer="below",
                y0=0.75,
                y1=1.29,
            ),
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0="2026-09-08",
                x1="2026-09-10",
                fillcolor=gray,
                line=dict(width=0),
                layer="below",
                y0=0.75,
                y1=1.29,
            ),
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=0,
                x1=1,
                fillcolor="rgba(0,0,0,0.05)",
            ),
        ]
    }


def test_sync_fig1_regime_shapes_extends_stale_last_band():
    layout = _stale_fig1_layout()
    seg = pd.DataFrame(
        [
            dict(regime="down", start="2026-08-13", end="2026-09-07", n_bars=18, ret=-0.03),
            dict(regime="range", start="2026-09-08", end="2026-09-17", n_bars=8, ret=-0.01),
        ]
    )
    ohlcv = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-08-13", "2026-09-10", "2026-09-17"]),
            "$low": [1.0, 0.95, 0.90],
            "$high": [1.2, 1.15, 1.10],
            "$close": [1.1, 1.0, 0.98],
            "$open": [1.05, 1.02, 1.00],
        }
    )
    assert sync_fig1_regime_shapes(layout, seg_df=seg, ohlcv=ohlcv) is True
    regime = [s for s in layout["shapes"] if _is_fig1_regime_rect(s)]
    assert {(s["x0"], s["x1"], s["fillcolor"]) for s in regime} == {
        ("2026-08-13", "2026-09-07", FIG1_REGIME_FILL["down"]),
        ("2026-09-08", "2026-09-17", FIG1_REGIME_FILL["range"]),
    }
    assert any(s.get("yref") == "paper" for s in layout["shapes"])
    assert sync_fig1_regime_shapes(layout, seg_df=seg, ohlcv=ohlcv) is False


def test_sync_fig1_regime_shapes_adds_new_regime_on_switch():
    layout = _stale_fig1_layout()
    seg = pd.DataFrame(
        [
            dict(regime="down", start="2026-08-13", end="2026-09-07"),
            dict(regime="range", start="2026-09-08", end="2026-09-10"),
            dict(regime="up", start="2026-09-11", end="2026-09-17"),
        ]
    )
    ohlcv = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-08-13", "2026-09-17"]),
            "$low": [1.0, 0.9],
            "$high": [1.2, 1.3],
            "$close": [1.1, 1.2],
            "$open": [1.0, 1.1],
        }
    )
    assert sync_fig1_regime_shapes(layout, seg_df=seg, ohlcv=ohlcv) is True
    last = [s for s in layout["shapes"] if _is_fig1_regime_rect(s)][-1]
    assert last["x0"] == "2026-09-11"
    assert last["x1"] == "2026-09-17"
    assert last["fillcolor"] == FIG1_REGIME_FILL["up"]


def test_sync_fig1_regime_shapes_matches_sz159611_segments_end():
    html = Path(
        "~/etf-daily-output/temp/plotly_outputs/20260917_from_listing/"
        "regime_transition_SZ159611_电力ETF广发_20220107_20260917_adaptive.html"
    )
    seg_path = Path(
        "~/etf-daily-output/temp/plotly_outputs/20260917_from_listing/trades/SZ159611_segments.csv"
    )
    if not html.exists() or not seg_path.exists():
        pytest.skip("SZ159611 20260917 from_listing HTML missing")
    layout = extract_layout(html.read_text(encoding="utf-8"))
    seg = pd.read_csv(seg_path)
    last_end = str(seg.iloc[-1]["end"])[:10]
    last_rg = str(seg.iloc[-1]["regime"])
    close, ohlcv = load_ohlcv_from_regime_html(html)
    ohlcv = ohlcv.reset_index()
    if "datetime" not in ohlcv.columns:
        ohlcv = ohlcv.rename(columns={ohlcv.columns[0]: "datetime"})
    sync_fig1_regime_shapes(layout, seg_df=seg, ohlcv=ohlcv)
    fresh = [s for s in layout["shapes"] if _is_fig1_regime_rect(s)]
    assert fresh
    assert max(str(s.get("x1"))[:10] for s in fresh) == last_end
    last_rects = [s for s in fresh if str(s.get("x1"))[:10] == last_end]
    assert last_rects[-1]["fillcolor"] == FIG1_REGIME_FILL[last_rg]
    _ = close


def test_patch_trail_path_hover_inserts_and_updates_ewma_percentile():
    prev = (
        "日期=2026-09-22<br>"
        "公平路径(自有log趋势·6个月 g≈1.0%, 诊断)=1.2000<br>"
        "当日路径年化(6个月)=1.0%<br>"
        "相对路径因果分位=+P70.0<br>"
        "煤炭 9/22 约 +3%"
    )
    kwargs = dict(
        new_ts=pd.Timestamp("2026-09-23"),
        path_new=1.25,
        close_new=1.10,
        g_new=0.02,
        g_by={},
        signed_pct=0.55,
        ewma_pct=-0.412,
        asset_name="煤炭ETF国泰",
    )
    inserted = _patch_trail_path_hover(prev, **kwargs)
    i_pct = inserted.find("相对路径因果分位=+P55.0")
    i_ewma = inserted.find("相对路径ewma分位=-P41.2")
    i_gap = inserted.find("煤炭 9/23 约 ")
    assert 0 <= i_pct < i_ewma < i_gap
    updated = _patch_trail_path_hover(inserted, **{**kwargs, "ewma_pct": 0.801})
    assert "相对路径ewma分位=+P80.1" in updated
    assert updated.count("相对路径ewma分位=") == 1
    assert updated.find("相对路径因果分位=") < updated.find("相对路径ewma分位=")
