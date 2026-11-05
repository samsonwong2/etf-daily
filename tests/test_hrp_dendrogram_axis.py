"""Offline checks for the HRP heatmap=False distance axis."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from etf_daily.hrp.dendrogram import style_hrp_no_heatmap


def test_style_hrp_no_heatmap_marks_center_pair():
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=[5, 15, 25, 35],
            ticktext=["甲", "乙", "丙", "丁"],
        ),
        yaxis_title="Distance",
    )
    corr = np.array(
        [
            [1.0, 0.2, 0.1, 0.0],
            [0.2, 1.0, 0.8, 0.3],
            [0.1, 0.8, 1.0, 0.4],
            [0.0, 0.3, 0.4, 1.0],
        ]
    )
    style_hrp_no_heatmap(fig, codependence=corr, names=["甲", "乙", "丙", "丁"])

    assert fig.layout.yaxis.dtick == 0.1
    assert fig.layout.yaxis.tick0 == 0
    assert fig.layout.yaxis.minor.dtick == 0.05
    assert fig.layout.yaxis.title.text == "Distance"

    assert len(fig.layout.shapes) == 1
    line = fig.layout.shapes[0]
    assert line.type == "line"
    assert line.line.dash == "dash"
    assert line.y0 == line.y1
    assert abs(float(line.y0) - np.sqrt(0.1)) < 1e-12
    assert float(line.x0) == 5
    assert float(line.x1) == 35

    assert len(fig.layout.annotations) == 1
    note = fig.layout.annotations[0]
    assert note.text == "乙 / 丙  ρ=0.80"
    assert note.bgcolor == "white"
    assert float(note.x) == 20
