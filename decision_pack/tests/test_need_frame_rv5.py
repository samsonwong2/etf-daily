"""Row-5 fair-need uses rv5 / rv5_anchor, not the rv20 annualized series."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from decision_pack.src.regime_transition_plot import compute_volatility_regime_frame
from decision_pack.src.vol_need_return_board import (
    ANCHOR_ERP_ANN,
    fair_need_ann_series,
    period_return_from_ann,
)


def _load_pool():
    path = Path("decision_pack/scripts/plot_adaptive_stage_pool.py")
    spec = importlib.util.spec_from_file_location("plot_adaptive_pool_rv5", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ohlcv_with_late_spike(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2026-04-01", periods=n)
    rets = rng.normal(0.0, 0.008, size=n)
    rets[-5:] = np.array([0.08, -0.07, 0.09, -0.06, 0.05])
    close = 10.0 * np.cumprod(1.0 + rets)
    return pd.DataFrame(
        {
            "datetime": dates,
            "$open": close,
            "$high": close * 1.01,
            "$low": close * 0.99,
            "$close": close,
            "$volume": np.full(n, 1_000_000.0),
        }
    )


def test_need_frame_r_need_ann_5_scales_with_rv5_ratio():
    pool = _load_pool()
    ohlcv = _ohlcv_with_late_spike()
    close = ohlcv.set_index(pd.to_datetime(ohlcv["datetime"]).dt.normalize())["$close"]
    vf = compute_volatility_regime_frame(close)
    vf["as_of"] = pd.to_datetime(vf["as_of"]).dt.normalize()
    indexed = vf.set_index("as_of")
    rv20_anchor = pd.Series(0.20, index=indexed.index, name="rv20")
    rv5_anchor = pd.Series(0.10, index=indexed.index, name="rv5")
    out = pool._need_frame_vs_anchor(
        ohlcv, rv20_anchor, rv5_anchor=rv5_anchor, erp_ann=ANCHOR_ERP_ANN
    )
    assert out is not None
    assert "r_need_ann_5" in out.columns
    last = out.dropna(subset=["r_need_ann", "r_need_ann_5"]).iloc[-1]
    as_of = pd.Timestamp(last["as_of"]).normalize()
    expected20 = float(
        fair_need_ann_series(indexed["rv20"], rv20_anchor, erp_ann=ANCHOR_ERP_ANN).loc[as_of]
    )
    expected5 = float(
        fair_need_ann_series(indexed["rv5"], rv5_anchor, erp_ann=ANCHOR_ERP_ANN).loc[as_of]
    )
    assert abs(float(last["r_need_ann"]) - expected20) < 1e-10
    assert abs(float(last["r_need_ann_5"]) - expected5) < 1e-10
    assert abs(expected5 - expected20) > 1e-4
    fair5 = float(
        period_return_from_ann(
            pd.Series([expected5], index=[as_of]), bars=5
        ).iloc[0]
    )
    assert abs(float(last["r_need_5d"]) - fair5) < 1e-10
    assert abs(float(last["gap_5d"]) - (float(last["r_5d"]) - fair5)) < 1e-10


def test_need_frame_falls_back_to_rv20_ann_without_rv5_anchor():
    pool = _load_pool()
    ohlcv = _ohlcv_with_late_spike()
    out = pool._need_frame_vs_anchor(
        ohlcv,
        pd.Series(0.20, index=pd.bdate_range("2026-04-01", periods=80), name="rv20"),
        rv5_anchor=None,
        erp_ann=ANCHOR_ERP_ANN,
    )
    assert out is not None
    both = out.dropna(subset=["r_need_ann", "r_need_ann_5"])
    assert np.allclose(
        both["r_need_ann"].to_numpy(dtype=float),
        both["r_need_ann_5"].to_numpy(dtype=float),
        equal_nan=True,
    )
