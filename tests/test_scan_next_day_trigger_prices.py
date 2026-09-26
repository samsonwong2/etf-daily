"""Next-day trigger scan: last-row EWMA rails and hard-touch markdown."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from etf_daily.triggers.scan import (
    HARD_TOUCH_COLS,
    _ewma_state_from_path_close,
    _last_ewma_envelope,
    _last_ewma_z,
    _path_residual,
    _signed_rank_last,
    write_hard_touch_csv,
    write_markdown,
)
from etf_daily.plots.plot_regime_transition_example import (
    causal_path_residual_signed_percentile,
)
from etf_daily.lib.fair_path_research import ewma_scaled_envelope, ewma_residual_sigma


def test_last_ewma_envelope_matches_full_last_bar() -> None:
    rng = np.random.default_rng(0)
    n = 120
    idx = pd.bdate_range("2020-01-02", periods=n)
    path = pd.Series(100.0 * np.exp(np.linspace(0.0, 0.2, n)), index=idx)
    close = path * np.exp(rng.normal(0.0, 0.02, n))
    up, lo, _ = ewma_scaled_envelope(path, close, min_bars=63)
    state = _ewma_state_from_path_close(path.iloc[:-1], close.iloc[:-1], min_bars=63)
    last_up, last_lo = _last_ewma_envelope(
        float(path.iloc[-1]), float(close.iloc[-1]), state, min_bars=63
    )
    assert np.isfinite(up.iloc[-1])
    np.testing.assert_allclose(last_up, float(up.iloc[-1]), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(last_lo, float(lo.iloc[-1]), rtol=1e-10, atol=1e-12)


def test_signed_rank_last_matches_causal_path_percentile() -> None:
    rng = np.random.default_rng(1)
    n = 90
    idx = pd.bdate_range("2020-01-02", periods=n)
    path = pd.Series(100.0 * np.exp(np.linspace(0.0, 0.15, n)), index=idx)
    close = path * np.exp(rng.normal(0.0, 0.03, n))
    signed = causal_path_residual_signed_percentile(path, close, min_bars=63)
    r_hist = np.log(close.iloc[:-1].to_numpy(dtype=float) / path.iloc[:-1].to_numpy(dtype=float))
    abs_hist = np.abs(r_hist[np.isfinite(r_hist)])
    r_last = _path_residual(float(path.iloc[-1]), float(close.iloc[-1]))
    got = _signed_rank_last(abs_hist, r_last, min_bars=63)
    np.testing.assert_allclose(got, float(signed.iloc[-1]), rtol=1e-10, atol=1e-12)


def test_ewma_signed_rank_last_matches_abs_z_percentile() -> None:
    rng = np.random.default_rng(2)
    n = 120
    idx = pd.bdate_range("2020-01-02", periods=n)
    path = pd.Series(100.0 * np.exp(np.linspace(0.0, 0.2, n)), index=idx)
    close = path * np.exp(rng.normal(0.0, 0.02, n))
    r = np.log(close.to_numpy(dtype=float) / path.to_numpy(dtype=float))
    sigma = ewma_residual_sigma(pd.Series(r, index=idx))
    z = r / sigma.to_numpy(dtype=float)
    abs_z = np.abs(z[np.isfinite(z)])
    state = _ewma_state_from_path_close(path.iloc[:-1], close.iloc[:-1], min_bars=63)
    z_last = _last_ewma_z(float(path.iloc[-1]), float(close.iloc[-1]), state)
    got = _signed_rank_last(state.abs_z, z_last, min_bars=63)
    q = float(np.mean(abs_z <= abs(z_last)))
    want = q if z_last > 0 else (-q if z_last < 0 else 0.0)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def test_write_markdown_hard_touch_has_ewma_and_width(tmp_path: Path) -> None:
    out = tmp_path / "next_day_triggers.md"
    write_markdown(
        out,
        as_of="2026-09-21",
        next_day="2026-09-22",
        rows=[
            {
                "ok": True,
                "code": "SH515220",
                "name": "煤炭ETF国泰",
                "px": 1.25,
                "state": "⑤趋势上行",
                "in_pos": False,
                "nearest_kind": "watch_lo",
                "nearest_abs": 0.10,
                "fig9_lo_px": 1.10,
                "fig9_lo_now": False,
                "fig9_hi_px": 1.40,
                "fig9_hi_now": False,
                "fig10_lo_px": 1.20,
                "fig10_lo_now": False,
                "fig10_hi_px": 1.49,
                "fig10_hi_now": False,
                "fig9_ewma_lo_px": 1.11,
                "fig9_ewma_lo_now": False,
                "fig9_ewma_hi_px": 1.39,
                "fig9_ewma_hi_now": False,
                "fig10_ewma_lo_px": 1.205,
                "fig10_ewma_lo_now": False,
                "fig10_ewma_hi_px": 1.50,
                "fig10_ewma_hi_now": False,
                "fig9_p95_width": 0.136,
                "fig9_ewma_width": 0.126,
                "fig10_p95_width": 0.113,
                "fig10_ewma_width": 0.114,
                "fig9_p95_pct": 0.508,
                "fig9_ewma_pct": 0.412,
                "fig10_p95_pct": -0.160,
                "fig10_ewma_pct": -0.155,
                "note": "图9空仓",
            }
        ],
        skipped_bonds=[],
        listing_dir=tmp_path,
        jobs=8,
    )
    text = out.read_text(encoding="utf-8")
    assert "## 图9/图10 硬触轨（T+1 收盘）" in text
    assert "图9EWMA下" in text
    assert "图9EWMA上" in text
    assert "图10EWMA下" in text
    assert "图10EWMA上" in text
    assert "图9 P95宽" in text
    assert "图9 因果分位P" in text
    assert "图9 EWMA宽" in text
    assert "图9 EWMA分位P" in text
    assert "图10 P95宽" in text
    assert "图10 因果分位P" in text
    assert "图10 EWMA宽" in text
    assert "图10 EWMA分位P" in text
    header = [ln for ln in text.splitlines() if ln.startswith("| 代码 | 名称 | T收盘 | 图9下轨 |")][0]
    cols = [c.strip() for c in header.strip("|").split("|")]
    assert cols == [
        "代码",
        "名称",
        "T收盘",
        "图9下轨",
        "图9上轨",
        "图9 P95宽",
        "图9 因果分位P",
        "图9EWMA下",
        "图9EWMA上",
        "图9 EWMA宽",
        "图9 EWMA分位P",
        "图10下轨",
        "图10上轨",
        "图10 P95宽",
        "图10 因果分位P",
        "图10EWMA下",
        "图10EWMA上",
        "图10 EWMA宽",
        "图10 EWMA分位P",
    ]
    assert "P95 与 EWMA 分开扫 T+1 硬触" in text
    assert "13.6%" in text
    assert "12.6%" in text
    assert "+50.8" in text
    assert "-16.0" in text
    assert "+P50.8" not in text
    csv_path = tmp_path / "next_day_triggers_hard_touch.csv"
    write_hard_touch_csv(
        csv_path,
        [
            {
                "ok": True,
                "code": "SH515220",
                "name": "煤炭ETF国泰",
                "px": 1.25,
                "fig9_lo_px": 1.10,
                "fig9_lo_now": False,
                "fig9_hi_px": 1.40,
                "fig9_hi_now": False,
                "fig10_lo_px": 1.20,
                "fig10_lo_now": False,
                "fig10_hi_px": 1.49,
                "fig10_hi_now": False,
                "fig9_ewma_lo_px": 1.11,
                "fig9_ewma_lo_now": False,
                "fig9_ewma_hi_px": 1.39,
                "fig9_ewma_hi_now": False,
                "fig10_ewma_lo_px": 1.205,
                "fig10_ewma_lo_now": False,
                "fig10_ewma_hi_px": 1.50,
                "fig10_ewma_hi_now": False,
                "fig9_p95_width": 0.136,
                "fig9_ewma_width": 0.126,
                "fig10_p95_width": 0.113,
                "fig10_ewma_width": 0.114,
                "fig9_p95_pct": 0.508,
                "fig9_ewma_pct": 0.412,
                "fig10_p95_pct": -0.160,
                "fig10_ewma_pct": -0.155,
            }
        ],
    )
    hard = pd.read_csv(csv_path, dtype=str)
    assert list(hard.columns) == list(HARD_TOUCH_COLS)
    row = hard.iloc[0]
    assert row["代码"] == "SH515220"
    assert row["名称"] == "煤炭ETF国泰"
    assert row["T收盘"] == "1.2500"
    assert row["图9 P95宽"] == "13.6%"
    assert row["图9 因果分位P"] == "+50.8"
    assert row["图10 因果分位P"] == "-16.0"
    assert row["图9 EWMA分位P"] == "+41.2"
