"""The forward page is a new HTML file. It does not touch a listing directory."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from etf_daily.lib import rules7  # noqa: E402
from etf_daily.lib.rules7_forward import FREEZE_DATE, replay_frame  # noqa: E402
from etf_daily.plots.plot_rules7_forward import build_figure, page_header, write_page  # noqa: E402
from etf_daily.scripts.backtest_fig9_touch import prepare_fig9_touch_frame  # noqa: E402


def _spec() -> rules7.RuleSpec:
    return rules7.OVERRIDES["SH512530"][1]


def _long_frame(n: int = 80) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=n)
    px = np.linspace(10.0, 12.0, n)
    px[0] = 10.0
    frame = pd.DataFrame(index=idx)
    frame["px"] = px
    frame["ma20"] = px + 1.0
    frame["ma5"] = px + 0.5
    frame["gap_5d"] = -0.01
    frame["gap_20d"] = -0.01
    lower = np.full(n, 8.0)
    lower[0] = 10.0
    frame["fig9_lower"] = lower
    frame["fig9_upper"] = 20.0
    frame["fig10_lower"] = 8.0
    frame["fig10_upper"] = 20.0
    frame["fig8_upper"] = 30.0
    frame["fig7_lower"] = 7.0
    frame["fig7_upper"] = 16.0
    frame["fig8_lower"] = 7.5
    for key, level in (("fig7_path", 11.0), ("fig8_path", 11.0), ("fig9_path", 11.0), ("fig10_path", 11.0)):
        frame[key] = level
    frame["fig7_gap"] = 0.0
    frame["fig8_gap"] = 0.0
    for key in (7, 8, 9, 10):
        frame[f"fig{key}_g"] = 0.01
    frame["fig9_touch_hi"] = False
    frame["fig10_touch_hi"] = False
    frame["locmax"] = False
    frame["locmin"] = False
    frame["vol5_pct_120d"] = 0.1
    return prepare_fig9_touch_frame(frame)


def test_page_module_does_not_import_the_listing_plotter():
    src = (_ROOT / "src/etf_daily/plots/plot_rules7_forward.py").read_text(encoding="utf-8")
    assert "plot_regime_transition_example" not in src
    assert "plot_adaptive_from_listing" not in src
    assert "incremental_from_listing" not in src


def test_figure_hides_ewma_and_marks_the_freeze_date():
    frame = _long_frame()
    replay = replay_frame("SH512530", "红利", frame, typ="平稳震荡", spec=_spec())
    ohlcv = pd.DataFrame(
        {
            "$open": frame["px"],
            "$high": frame["px"] + 0.1,
            "$low": frame["px"] - 0.1,
            "$close": frame["px"],
        },
        index=frame.index,
    )
    fig = build_figure(replay, ohlcv, None, page_header(replay, None))
    ewma = [trace for trace in fig.data if trace.name and "EWMA" in trace.name]
    assert len(ewma) == 8
    assert all(trace.visible == "legendonly" for trace in ewma)
    candles = [trace for trace in fig.data if trace.type == "candlestick"]
    assert len(candles) == 4
    assert any(trace.name and "贴轨张数" in trace.name for trace in fig.data)
    assert any(str(shape.x0).startswith(FREEZE_DATE) for shape in fig.layout.shapes)
    assert "冻结日" in fig.layout.title.text
    assert "先跑 etf-daily forward" in fig.layout.title.text


def test_write_page_does_not_touch_the_listing_dir(tmp_path: Path):
    listing = tmp_path / "20261109_from_listing"
    listing.mkdir()
    marker = listing / "regime_transition_SH512530_keep_adaptive.html"
    marker.write_text("keep", encoding="utf-8")
    before = (marker.read_text(encoding="utf-8"), marker.stat().st_mtime_ns, marker.stat().st_size)
    names_before = sorted(path.name for path in listing.iterdir())

    frame = _long_frame()
    replay = replay_frame("SH512530", "红利", frame, typ="平稳震荡", spec=_spec())
    ohlcv = pd.DataFrame({"$close": frame["px"], "$open": frame["px"], "$high": frame["px"], "$low": frame["px"]}, index=frame.index)
    fig = build_figure(replay, ohlcv, None, "header")
    out = tmp_path / "20261109_forward" / "rules7_forward_SH512530_红利_20261109.html"
    write_page(fig, out)

    after = (marker.read_text(encoding="utf-8"), marker.stat().st_mtime_ns, marker.stat().st_size)
    assert before == after
    assert sorted(path.name for path in listing.iterdir()) == names_before
    assert out.is_file()
    assert out.stat().st_size > 1000
    text = out.read_text(encoding="utf-8")
    assert "regime_transition_" not in out.name
    assert "Plotly.newPlot" in text or "plotly" in text.lower()
