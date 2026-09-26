"""Per-code enter-up params: wiring + B0 joint train selection."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from etf_daily.paths import PLOTLY_OUTPUTS_DIR

import numpy as np
import pandas as pd

from etf_daily.lib.adaptive_stage_common import (
    enter_up_param_grid,
    select_regime_method_legacy,
)


def _load_ev():
    path = Path(__file__).resolve().parents[1] / "scripts/eval_causal_regime_switch_pool.py"
    spec = importlib.util.spec_from_file_location("ev_enter_up", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _synthetic_px(n: int = 220, *, seed: int = 0, mode: str = "trend") -> pd.DataFrame:
    """Build a minimal px frame with MA/prior features for method tests."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    if mode == "trend":
        # Two clear legs so retrospective_truth is not all-range.
        close = np.empty(n, dtype=float)
        close[:80] = 1.0 + 0.004 * t[:80]
        close[80:110] = close[79] * np.linspace(1.0, 0.75, 30)
        close[110:] = close[109] * np.cumprod(
            1.0 + 0.008 + 0.001 * rng.standard_normal(n - 110)
        )
    else:
        close = 1.0 + 0.03 * np.sin(np.linspace(0, 24, n))
        close = close * (1.0 + 0.002 * rng.standard_normal(n))
    high = close * 1.015
    low = close * 0.985
    open_ = close * (1.0 + 0.001 * rng.standard_normal(n))
    vol = np.full(n, 1e6)
    dates = pd.bdate_range("2025-01-02", periods=n)
    ohlcv = pd.DataFrame(
        {
            "datetime": dates,
            "$open": open_,
            "$high": high,
            "$low": low,
            "$close": close,
            "$volume": vol,
        }
    )
    ev = _load_ev()
    return ev.build_px(ohlcv)


def test_empty_params_match_default_strict():
    ev = _load_ev()
    px = _synthetic_px()
    a = ev.run_method("ma_stack_strict", px, params={})
    b = ev.run_method("ma_stack_strict", px, params=None)
    c = ev.method_ma_stack_strict(px)
    # finalize with default min_seg applied in run_method
    c2 = ev.finalize_regime_labels(c, px, min_seg=10)
    assert list(a) == list(b) == list(c2)


def test_strict_confirm_1_enters_up_earlier_than_confirm_3():
    ev = _load_ev()
    px = _synthetic_px(seed=7)
    labs1 = ev.run_method(
        "ma_stack_strict",
        px,
        params={"p60_thr": 0.03, "p20_vrev": 0.03, "confirm": 1, "min_seg": 5},
    )
    labs3 = ev.run_method(
        "ma_stack_strict",
        px,
        params={"p60_thr": 0.03, "p20_vrev": 0.03, "confirm": 3, "min_seg": 5},
    )
    up1 = np.where(labs1 == "up")[0]
    up3 = np.where(labs3 == "up")[0]
    assert len(up1) > 0
    # Faster confirm should not enter later than slow confirm's first up.
    if len(up3) > 0:
        assert int(up1[0]) <= int(up3[0])


def test_strict_washout_vrev_rejects_far_stack_crash():
    """Crash-day p60<=-8% with MA20 still far below MA60 is not an up V.

    SH588850 2026-08-19 pattern: prior20 still green from the bounce, the
    red day itself pushes prior60 through -8%, stack gap ~-9%.
    """
    ev = _load_ev()
    n = 40
    dates = pd.bdate_range("2026-06-01", periods=n)
    close = np.linspace(1.70, 1.82, n)
    close[-6] = 1.969
    close[-5] = 1.787  # crash
    close[-4:] = [1.818, 1.827, 1.787, 1.762]
    ma20 = np.linspace(1.70, 1.75, n)
    ma60 = np.linspace(1.90, 1.93, n)
    p20 = np.full(n, 0.08)
    p60 = np.full(n, -0.04)
    p60[-5] = -0.092
    p60[-3] = -0.080
    px = pd.DataFrame(
        {
            "as_of": dates,
            "datetime": dates,
            "$open": close,
            "$high": close * 1.01,
            "$low": close * 0.99,
            "$close": close,
            "MA20": ma20,
            "MA60": ma60,
            "prior20": p20,
            "prior60": p60,
            "band60": np.full(n, 0.50),
        }
    )
    labs = ev.run_method(
        "ma_stack_strict",
        px,
        params={"p60_thr": 0.06, "p20_vrev": 0.03, "confirm": 1, "min_seg": 5},
    )
    assert not (labs[-8:] == "up").any()


def test_strict_washout_vrev_keeps_near_stack_green_bar():
    """Near-stack washout on a green bar may still enter (SH588850 2025-12-19)."""
    ev = _load_ev()
    n = 40
    dates = pd.bdate_range("2025-11-03", periods=n)
    close = np.linspace(1.25, 1.28, n - 8)
    close = np.concatenate([close, np.linspace(1.281, 1.330, 8)])
    ma20 = np.full(n, 1.270)
    ma60 = np.full(n, 1.309)
    p20 = np.full(n, 0.05)
    p60 = np.full(n, -0.07)
    p60[-8:] = -0.083
    px = pd.DataFrame(
        {
            "as_of": dates,
            "datetime": dates,
            "$open": close,
            "$high": close * 1.01,
            "$low": close * 0.99,
            "$close": close,
            "MA20": ma20,
            "MA60": ma60,
            "prior20": p20,
            "prior60": p60,
            "band60": np.full(n, 0.40),
        }
    )
    labs = ev.run_method(
        "ma_stack_strict",
        px,
        params={"p60_thr": 0.06, "p20_vrev": 0.03, "confirm": 1, "min_seg": 5},
    )
    assert "up" in set(labs[-5:])


def test_strict_sh588850_aug2026_bounce_stays_range():
    html = (
        PLOTLY_OUTPUTS_DIR / "20260904_from_listing"
        / "regime_transition_SH588850_科创机械ETF嘉实_20250424_20260904_adaptive.html"
    )
    if not html.is_file():
        return
    from etf_daily.lib.adaptive_stage_common import segments
    from etf_daily.lib.fair_path_band_signals import load_ohlcv_from_regime_html

    ev = _load_ev()
    _, ohlcv = load_ohlcv_from_regime_html(html)
    px = ev.build_px(ohlcv.reset_index())
    labs = ev.run_method(
        "ma_stack_strict",
        px,
        params={"p60_thr": 0.06, "p20_vrev": 0.03, "confirm": 1, "min_seg": 5},
    )
    seg = segments(
        labs,
        px["as_of"].dt.strftime("%Y-%m-%d").to_numpy(),
        px["$close"].to_numpy(),
    )
    late = seg[seg["start"] >= "2026-08-01"]
    assert not (
        (late["regime"] == "up")
        & (late["start"] <= "2026-08-20")
        & (late["end"] >= "2026-08-20")
    ).any()


def test_enter_up_param_grid_includes_defaults_and_strict_keys():
    g = enter_up_param_grid("ma_stack_strict")
    assert {} in g
    assert any(
        p.get("p60_thr") == 0.06 and p.get("p20_vrev") == 0.06 and p.get("confirm") == 3
        for p in g
        if p
    )


def test_legacy_select_joint_search_on_directional_train():
    ev = _load_ev()
    px_a = _synthetic_px(seed=1, mode="trend")
    px_b = _synthetic_px(seed=2, mode="chop")
    m_a, p_a, meta_a = select_regime_method_legacy(ev, px_a, trade_mode="hold_up")
    m_b, p_b, meta_b = select_regime_method_legacy(ev, px_b, trade_mode="hold_up")
    assert meta_a.get("joint_param_search") is True
    assert meta_a.get("hold_up_tiebreak") is False
    assert isinstance(p_a, dict)
    assert m_a in ev.METHODS
    assert int(meta_a.get("grid_tried") or 0) > len(ev.METHODS)
    # Chop series may fall back to degenerate hyst; trend must joint-search.
    assert m_a in ev.METHODS and m_b in ev.METHODS
    _ = (m_b, p_b, meta_b)