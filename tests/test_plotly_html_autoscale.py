"""Tests for Plotly HTML Y-autoscale helper."""
from __future__ import annotations

from etf_daily.lib.plotly_html_autoscale import (
    Y_AUTOSCALE_ON_X_ZOOM_JS,
    write_adaptive_html,
)


def test_y_autoscale_js_has_relayout_hook():
    assert "plotly_relayout" in Y_AUTOSCALE_ON_X_ZOOM_JS
    assert "candlestick" in Y_AUTOSCALE_ON_X_ZOOM_JS
    assert "yaxis.range" in Y_AUTOSCALE_ON_X_ZOOM_JS or "'.range'" in Y_AUTOSCALE_ON_X_ZOOM_JS
    assert 'type = \'date\'' in Y_AUTOSCALE_ON_X_ZOOM_JS or 'type = "date"' in Y_AUTOSCALE_ON_X_ZOOM_JS
    assert "plot-start-date" in Y_AUTOSCALE_ON_X_ZOOM_JS
    assert "plot-end-date" in Y_AUTOSCALE_ON_X_ZOOM_JS
    assert "start-date" in Y_AUTOSCALE_ON_X_ZOOM_JS
    assert "end-date" in Y_AUTOSCALE_ON_X_ZOOM_JS
    assert "remapPriceMarkers" in Y_AUTOSCALE_ON_X_ZOOM_JS
    assert "_origY" in Y_AUTOSCALE_ON_X_ZOOM_JS
    assert "Plotly.restyle" in Y_AUTOSCALE_ON_X_ZOOM_JS


def test_write_adaptive_html_embeds_post_script(tmp_path):
    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name="t"))
    out = tmp_path / "t.html"
    write_adaptive_html(fig, out)
    text = out.read_text(encoding="utf-8")
    assert "plotly_relayout" in text
    assert "candlestick" in text
    assert "plot-start-date" in text
    assert "plot-end-date" in text
    assert "adaptive-date-window" in text
    assert "unpackBdata" in text
    assert "Date.UTC" in text


def test_js_parses_ns_timestamps_and_bdata():
    js = Y_AUTOSCALE_ON_X_ZOOM_JS
    assert "000000000" in js
    assert "unpackBdata" in js
    assert "Date.UTC" in js
    assert "asArray" in js
