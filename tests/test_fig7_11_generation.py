"""Fig7–11 generation: frozen 2026-10-19 curves, incremental==full, labels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from etf_daily.lib.fair_path_band_signals import (
    FIG_PANELS,
    SKIP_CODES_NO_FIG7,
    find_regime_html,
    load_ohlcv_from_regime_html,
)
from etf_daily.lib.fair_path_research import ewma_scaled_envelope, linear_quantile_sorted
from etf_daily.lib.incremental_from_listing import (
    append_fair_path_bar,
    read_fair_path_rails_stamp,
    try_incremental_html,
)
from etf_daily.plots.plot_regime_transition_example import (
    _short_window_note,
    _trail_envelope_policy,
    _trail_fit_min_bars,
    causal_path_residual_signed_percentile,
    erp_path_symmetric_envelope,
    fair_path_rails_stamp,
    rail_min_bars,
    trailing_log_trend_price_path,
)

_BASELINE = Path(__file__).resolve().parent / "fixtures" / "fig7_11_20261019_baseline.json"
_PANELS = (
    ("y10", 2.0, True),
    ("y11", 1.0, True),
    ("y12", 0.5, True),
    ("y13", 0.25, True),
    ("y14", 1.0 / 12.0, False),
)


def _recompute(close: pd.Series, years: float, *, with_ewma: bool) -> dict[str, np.ndarray]:
    min_path = _trail_fit_min_bars(years)
    min_rail = rail_min_bars(years)
    path, _g = trailing_log_trend_price_path(
        close, trail_years=years, min_bars=min_path
    )
    upper, lower, _ = erp_path_symmetric_envelope(
        path, close, q=0.95, min_bars=min_rail
    )
    out = {
        "path": path.to_numpy(dtype=float),
        "p95_up": upper.to_numpy(dtype=float),
        "p95_lo": lower.to_numpy(dtype=float),
    }
    if with_ewma:
        eu, el, _ = ewma_scaled_envelope(path, close, min_bars=min_rail)
        out["ewma_up"] = eu.to_numpy(dtype=float)
        out["ewma_lo"] = el.to_numpy(dtype=float)
    return out


def test_linear_quantile_sorted_matches_numpy():
    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 21, 63, 100, 400):
        x = rng.normal(size=n).astype(np.float64)
        ordered = [float(v) for v in np.sort(x)]
        for q in (0.95, 0.5, 0.9):
            assert linear_quantile_sorted(ordered, q) == float(np.quantile(x, q))


def test_policy_comes_from_fig_panels():
    assert _trail_envelope_policy(2.0) == "expanding_plus_ewma"
    assert _trail_envelope_policy(0.25) == "expanding_plus_ewma"
    assert _trail_envelope_policy(1.0 / 12.0) == "expanding"
    assert rail_min_bars(1.0 / 12.0) == 63
    assert _trail_fit_min_bars(1.0 / 12.0) == 21
    assert "fig11" in {key for key, _, _ in FIG_PANELS}
    assert SKIP_CODES_NO_FIG7 == frozenset({"SH511010", "SH511260"})


def test_fig11_hover_warmup_matches_rail_min_bars():
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 80))))
    years = 1.0 / 12.0
    path, _g = trailing_log_trend_price_path(
        close, trail_years=years, min_bars=_trail_fit_min_bars(years)
    )
    early = causal_path_residual_signed_percentile(
        path, close, min_bars=_trail_fit_min_bars(years)
    )
    aligned = causal_path_residual_signed_percentile(
        path, close, min_bars=rail_min_bars(years)
    )
    early_only = np.isfinite(early.to_numpy()) & ~np.isfinite(aligned.to_numpy())
    assert int(early_only.sum()) == 42
    both = np.isfinite(early.to_numpy()) & np.isfinite(aligned.to_numpy())
    assert np.array_equal(early.to_numpy()[both], aligned.to_numpy()[both])


def test_short_window_note_and_annualized_label():
    assert _short_window_note(165, 2.0) == "窗口不足165根"
    assert _short_window_note(600, 2.0) == ""
    assert _short_window_note(200, 1.0) == "窗口不足200根"


def test_append_last_bar_matches_full_recompute():
    rng = np.random.default_rng(2)
    n = 160
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n))), index=idx)
    years = 0.5
    full = _recompute(close, years, with_ewma=False)
    xs = [d.strftime("%Y-%m-%d") for d in idx[:-1]]
    path_hist = full["path"][:-1]
    up_hist = full["p95_up"][:-1]
    lo_hist = full["p95_lo"][:-1]
    traces = [
        {
            "type": "candlestick",
            "yaxis": "y12",
            "name": "K线(净)",
            "x": list(xs),
            "open": close.iloc[:-1].tolist(),
            "high": close.iloc[:-1].tolist(),
            "low": close.iloc[:-1].tolist(),
            "close": close.iloc[:-1].tolist(),
        },
        {
            "type": "scatter",
            "yaxis": "y12",
            "name": "公平路径(自有log趋势·6个月 年化外推g≈1.0%, 诊断)",
            "x": list(xs),
            "y": path_hist.tolist(),
            "customdata": ["日期=2024-01-02<br>当日路径年化外推(6个月)=1.0%"] * (n - 1),
        },
        {
            "type": "scatter",
            "yaxis": "y12",
            "name": "上轨6个月 +1.0% (因果P95)",
            "x": list(xs),
            "y": up_hist.tolist(),
        },
        {
            "type": "scatter",
            "yaxis": "y12",
            "name": "下轨6个月 -1.0% (因果P95)",
            "x": list(xs),
            "y": lo_hist.tolist(),
        },
    ]
    last = pd.Series(
        {"$open": close.iloc[-1], "$high": close.iloc[-1], "$low": close.iloc[-1], "$close": close.iloc[-1]}
    )
    append_fair_path_bar(
        traces,
        close=close,
        ohlcv_last=last,
        new_ts=idx[-1],
        trails=[years],
        asset_name="测试ETF",
    )
    got_path = traces[1]["y"][-1]
    got_up = traces[2]["y"][-1]
    got_lo = traces[3]["y"][-1]
    assert got_path == pytest.approx(float(full["path"][-1]), rel=1e-12)
    assert got_up == pytest.approx(float(full["p95_up"][-1]), rel=1e-12)
    assert got_lo == pytest.approx(float(full["p95_lo"][-1]), rel=1e-12)


def test_incremental_rejects_missing_rails_stamp(tmp_path: Path):
    import plotly.graph_objects as go

    from etf_daily.lib.plotly_html_autoscale import write_adaptive_html

    prev = tmp_path / "prev"
    prev.mkdir()
    idx = pd.bdate_range("2026-10-01", periods=8)
    px = np.linspace(1.0, 1.1, len(idx))
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=idx,
                open=px,
                high=px,
                low=px,
                close=px,
                name="K线(净)",
                yaxis="y10",
            )
        ]
    )
    html = prev / "regime_transition_SH510300_测试_20120528_20261016_adaptive.html"
    write_adaptive_html(fig, html)
    assert read_fair_path_rails_stamp(html.read_text(encoding="utf-8")) is None
    ohlcv = pd.DataFrame(
        {
            "datetime": idx,
            "$open": px,
            "$high": px,
            "$low": px,
            "$close": px,
        }
    )
    out, reason = try_incremental_html(
        code="SH510300",
        prev_dir=prev,
        out_dir=tmp_path / "out",
        ohlcv=ohlcv,
        end_date="2026-10-12",
        close_tol=1e-6,
        extra_trail_years="1,0.5,0.25,1/12",
    )
    assert out is None
    assert reason == "algo_version"
    assert "fair-path-rails" in fair_path_rails_stamp()


def _html_series(traces: list, yaxis: str) -> dict[str, np.ndarray | None]:
    from etf_daily.lib.fair_path_band_signals import _decode_plotly_bdata

    ts = [t for t in traces if t.get("yaxis") == yaxis and t.get("type") == "scatter"]

    def arr(t):
        v = t["y"]
        if isinstance(v, dict) and "bdata" in v:
            v = _decode_plotly_bdata(v)
        return np.array([np.nan if x is None else float(x) for x in v], dtype=np.float64)

    def pick(pred):
        hits = [t for t in ts if pred(str(t.get("name") or ""))]
        return arr(hits[0]) if hits else None

    return {
        "path": pick(lambda n: n.startswith("公平路径")),
        "p95_up": pick(lambda n: n.startswith("上轨") and "P95" in n),
        "p95_lo": pick(lambda n: n.startswith("下轨") and "P95" in n),
        "ewma_up": pick(lambda n: n.startswith("上轨") and "EWMA" in n),
        "ewma_lo": pick(lambda n: n.startswith("下轨") and "EWMA" in n),
    }


def _close_enough(got: np.ndarray, ref: np.ndarray) -> bool:
    if got.shape != ref.shape:
        return False
    if not np.array_equal(np.isfinite(got), np.isfinite(ref)):
        return False
    m = np.isfinite(got)
    if not m.any():
        return True
    return bool(np.allclose(got[m], ref[m], rtol=1e-7, atol=1e-8))


def test_listing_20261019_matches_frozen_baseline():
    if not _BASELINE.exists():
        pytest.skip("fig7-11 baseline fixture missing")
    from etf_daily.lib.fair_path_band_signals import _extract_plotly_traces

    doc = json.loads(_BASELINE.read_text(encoding="utf-8"))
    root = Path(doc["listing_dir"]).expanduser()
    if not root.is_dir():
        from etf_daily.paths import PLOTLY_OUTPUTS_DIR

        root = PLOTLY_OUTPUTS_DIR / doc["listing_dir"]
    if not root.is_dir():
        pytest.skip(f"listing HTML dir missing: {root}")
    mismatches: list[str] = []
    for code, body in doc["codes"].items():
        html = find_regime_html(root, code)
        if html is None:
            mismatches.append(f"{code} html missing")
            continue
        close, _ohlcv = load_ohlcv_from_regime_html(html)
        traces = _extract_plotly_traces(html.read_text(encoding="utf-8"))
        for ya, years, with_ewma in _PANELS:
            panel = body["panels"][ya]
            got = _recompute(close, years, with_ewma=with_ewma)
            html_series = _html_series(traces, ya)
            for key, arr in got.items():
                spec = panel["series"].get(key)
                ref = html_series.get(key)
                if spec is None or ref is None:
                    mismatches.append(f"{code} {ya} {key} missing")
                    continue
                if not _close_enough(arr, ref):
                    mismatches.append(f"{code} {ya} {key} html")
                    continue
                finite = arr[np.isfinite(arr)]
                last = float(finite[-1]) if finite.size else None
                if spec["last"] is None or last is None:
                    if spec["last"] != last:
                        mismatches.append(f"{code} {ya} {key} last")
                elif abs(last / float(spec["last"]) - 1.0) > 1e-7:
                    mismatches.append(f"{code} {ya} {key} last")
    assert mismatches == [], mismatches[:12]
