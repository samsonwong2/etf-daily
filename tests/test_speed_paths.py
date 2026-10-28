"""Equivalence checks for the listing/regime speed paths."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from etf_daily.lib.incremental_from_listing import (
    _indexed_ohlc,
    end_ohlc_matches,
    ohlcv_fingerprint,
    overlap_ohlc_matches,
    stale_end_bar_codes,
)
from etf_daily.lib.plotly_html_autoscale import write_adaptive_html
from etf_daily.lib.six_states import (
    STATE_COLORS,
    STATE_WARMUP,
    build_fig12_figure,
)
from etf_daily.plots.add_fig12_six_states import FIG12_FP_RE, add_fig12_to_html
from etf_daily.plots.overlay_oracle_pivots_on_regime_html import extract_layout
from etf_daily.regime.backtest_regime_transition_signals import _signal_levels


def _reference_overlap(html: pd.DataFrame, official: pd.DataFrame, tol: float) -> bool:
    left = _indexed_ohlc(html)
    right = _indexed_ohlc(official)
    if left.empty or right.empty:
        return True
    common = left.index.intersection(right.index)
    if len(common) == 0:
        return True
    for day in common:
        if not end_ohlc_matches(left, right, str(pd.Timestamp(day).date()), tol=tol):
            return False
    return True


def test_overlap_ohlc_matches_reference_including_gaps():
    idx = pd.bdate_range("2024-01-02", periods=40)
    html = pd.DataFrame(
        {
            "$open": np.linspace(1.0, 1.4, len(idx)),
            "$high": np.linspace(1.1, 1.5, len(idx)),
            "$low": np.linspace(0.9, 1.3, len(idx)),
            "$close": np.linspace(1.05, 1.45, len(idx)),
        },
        index=idx,
    )
    official = html.iloc[:-3].copy()
    assert overlap_ohlc_matches(html, official, tol=1e-5) is True
    assert _reference_overlap(html, official, 1e-5) is True
    official = html.copy()
    official.iloc[7, official.columns.get_loc("$close")] *= 1.01
    assert overlap_ohlc_matches(html, official, tol=1e-5) is False
    assert overlap_ohlc_matches(html, official, tol=1e-5) == _reference_overlap(
        html, official, 1e-5
    )
    broken = html.copy()
    broken.iloc[3, broken.columns.get_loc("$low")] = np.nan
    assert overlap_ohlc_matches(broken, html, tol=1e-5) is False
    assert overlap_ohlc_matches(broken, html, tol=1e-5) == _reference_overlap(
        broken, html, 1e-5
    )
    missing = html.drop(columns=["$open"])
    assert overlap_ohlc_matches(missing, html, tol=1e-5) is False
    assert overlap_ohlc_matches(missing, html, tol=1e-5) == _reference_overlap(
        missing, html, 1e-5
    )


def test_ohlcv_fingerprint_skips_unreadable_html_when_it_matches(tmp_path: Path):
    idx = pd.to_datetime(["2026-10-22"])
    official = pd.DataFrame(
        {"$open": [1.0], "$high": [1.1], "$low": [0.9], "$close": [1.05]},
        index=idx,
    )
    page = tmp_path / "regime_transition_SH510300_NAME_20100101_20261022_adaptive.html"
    page.write_text("not html", encoding="utf-8")
    fp = ohlcv_fingerprint(official)
    stale = stale_end_bar_codes(
        {"SH510300": page},
        lambda _code: official,
        "2026-10-22",
        tol=1e-5,
        fingerprints={"SH510300": fp},
    )
    assert stale == set()
    verified: dict[str, str] = {}
    stale_again = stale_end_bar_codes(
        {"SH510300": page},
        lambda _code: official,
        "2026-10-22",
        tol=1e-5,
        fingerprints={"SH510300": "deadbeef"},
        verified_fp=verified,
    )
    assert stale_again == {"SH510300"}
    assert verified == {}


def test_fig12_shapes_match_add_vrect():
    idx = pd.bdate_range("2024-01-02", periods=12)
    ohlcv = pd.DataFrame(
        {
            "$open": np.linspace(1, 2, len(idx)),
            "$high": np.linspace(1.1, 2.1, len(idx)),
            "$low": np.linspace(0.9, 1.9, len(idx)),
            "$close": np.linspace(1.05, 2.05, len(idx)),
        },
        index=idx,
    )
    labels = ["⑤趋势上行"] * 4 + [STATE_WARMUP] * 3 + ["⑥震荡"] * 5
    states = pd.Series(labels, index=idx)
    got = build_fig12_figure(ohlcv, states, "X")

    legacy = go.Figure()
    legacy.add_trace(
        go.Candlestick(
            x=idx,
            open=ohlcv["$open"],
            high=ohlcv["$high"],
            low=ohlcv["$low"],
            close=ohlcv["$close"],
            increasing_line_color="#d62728",
            decreasing_line_color="#2ca02c",
            name="K线",
            showlegend=False,
        )
    )
    runs = (states != states.shift()).cumsum()
    for _, seg in states.groupby(runs, sort=False):
        label = str(seg.iloc[0])
        color = STATE_COLORS.get(label)
        if color is None:
            continue
        legacy.add_vrect(
            x0=pd.Timestamp(seg.index[0]),
            x1=pd.Timestamp(seg.index[-1]) + pd.Timedelta(days=1),
            fillcolor=color,
            line_width=0,
            layer="below",
        )
    got_shapes = [shape.to_plotly_json() for shape in got.layout.shapes]
    legacy_shapes = [shape.to_plotly_json() for shape in legacy.layout.shapes]
    assert got_shapes == legacy_shapes
    assert len(got_shapes) == 2


def test_fig12_fingerprint_skips_second_write(tmp_path: Path):
    idx = pd.bdate_range("2020-01-02", periods=30)
    fig = go.Figure(
        go.Candlestick(
            x=idx,
            open=np.linspace(1, 2, len(idx)),
            high=np.linspace(1.1, 2.2, len(idx)),
            low=np.linspace(0.9, 1.8, len(idx)),
            close=np.linspace(1.05, 2.1, len(idx)),
            yaxis="y10",
            name="K线",
        )
    )
    path = tmp_path / "regime_transition_SH510300_NAME_20200102_20200214_adaptive.html"
    write_adaptive_html(fig, path)
    first = add_fig12_to_html(path)
    assert first["ok"] == "1"
    assert first["skipped"] == "0"
    text = path.read_text(encoding="utf-8")
    assert FIG12_FP_RE.search(text)
    stamp = path.stat().st_mtime_ns
    second = add_fig12_to_html(path)
    assert second["skipped"] == "1"
    assert path.read_text(encoding="utf-8") == text
    assert path.stat().st_mtime_ns == stamp


def test_layout_decodes_nan_as_null():
    html = 'Plotly.newPlot("gd", [{"y": [1, 2]}], {"title": {"text": NaN}, "x": -Infinity}, {});'
    layout = extract_layout(html)
    assert layout["title"]["text"] is None
    assert layout["x"] is None


def test_signal_levels_cover_each_branch():
    frame = pd.DataFrame(
        [
            dict(
                p_trend_cal=0.9,
                p_into_reversal_adj=0.0,
                p_into_reversal_hmm=0.0,
                transition_score=0.1,
                return_z=0.0,
                pred_cdf=0.5,
                label_ambiguous=False,
                transition_direction="up",
                is_candidate=True,
                quantile_switch=True,
                switch_side="up",
                direction_up=True,
            ),
            dict(
                p_trend_cal=0.1,
                p_into_reversal_adj=0.0,
                p_into_reversal_hmm=0.0,
                transition_score=0.0,
                return_z=0.0,
                pred_cdf=0.2,
                label_ambiguous=False,
                transition_direction="down",
                is_candidate=False,
                quantile_switch=True,
                switch_side="down",
                direction_up=False,
            ),
            dict(
                p_trend_cal=0.1,
                p_into_reversal_adj=0.0,
                p_into_reversal_hmm=0.5,
                transition_score=0.0,
                return_z=0.0,
                pred_cdf=0.2,
                label_ambiguous=False,
                transition_direction="mixed",
                is_candidate=False,
                quantile_switch=False,
                switch_side="",
                direction_up=False,
            ),
            dict(
                p_trend_cal=0.1,
                p_into_reversal_adj=0.0,
                p_into_reversal_hmm=0.0,
                transition_score=0.0,
                return_z=0.0,
                pred_cdf=0.2,
                label_ambiguous=False,
                transition_direction="mixed",
                is_candidate=False,
                quantile_switch=False,
                switch_side="",
                direction_up=False,
            ),
            dict(
                p_trend_cal=np.nan,
                p_into_reversal_adj=0.0,
                p_into_reversal_hmm=0.9,
                transition_score=0.0,
                return_z=0.0,
                pred_cdf=0.99,
                label_ambiguous=False,
                transition_direction="up",
                is_candidate=True,
                quantile_switch=True,
                switch_side="up",
                direction_up=True,
            ),
            dict(
                p_trend_cal=np.nan,
                p_into_reversal_adj=0.0,
                p_into_reversal_hmm=0.9,
                transition_score=0.0,
                return_z=0.0,
                pred_cdf=0.2,
                label_ambiguous=False,
                transition_direction="mixed",
                is_candidate=False,
                quantile_switch=False,
                switch_side="",
                direction_up=False,
            ),
            dict(
                p_trend_cal=0.1,
                p_into_reversal_adj=0.0,
                p_into_reversal_hmm=0.0,
                transition_score=0.8,
                return_z=2.5,
                pred_cdf=0.2,
                label_ambiguous=False,
                transition_direction="up",
                is_candidate=True,
                quantile_switch=False,
                switch_side="",
                direction_up=True,
            ),
        ]
    )
    levels = list(
        _signal_levels(
            frame,
            meta={
                "q_confirm": 0.8,
                "z_confirm": None,
                "cdf_confirm": None,
                "n_crossfit_months": 8,
            },
            q_hmm=0.2,
            q_observe=0.28,
        )
    )
    assert levels == [
        "CONFIRM",
        "OBSERVE",
        "REJECT_AS_ONE_DAY_JUMP",
        "NONE",
        "OBSERVE",
        "NONE",
        "OBSERVE",
    ]


def test_rails_stamp_is_written_with_the_page(tmp_path: Path):
    fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4], name="t"))
    out = tmp_path / "page.html"
    write_adaptive_html(fig, out, rails_stamp="fair-path-rails:abcdef0123456789")
    text = out.read_text(encoding="utf-8")
    assert "<!-- fair-path-rails:abcdef0123456789 -->\n</body>" in text
    assert text.count("fair-path-rails:") == 1
