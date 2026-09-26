"""Bond fair-need anchor is SH511090; defensive codes get need panels."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from etf_daily.lib.vol_need_return_board import (
    BOND_ANCHOR,
    BOND_ERP_ANN,
    is_defensive_code,
)
from etf_daily.lib.vol_return_board import (
    DEFAULT_BOND_ANCHOR,
    DEFAULT_DEFENSIVE_CODES,
)


def _load_pool():
    path = Path("src/etf_daily/plots/plot_adaptive_stage_pool.py")
    spec = importlib.util.spec_from_file_location("plot_adaptive_pool_bond", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_bond_anchor_is_sh511090():
    assert DEFAULT_BOND_ANCHOR == "SH511090"
    assert BOND_ANCHOR == "SH511090"
    assert {"SH511090", "SH511010", "SH511260"} <= {
        str(c).upper() for c in DEFAULT_DEFENSIVE_CODES
    }
    assert is_defensive_code("SH511090")
    assert is_defensive_code("sh511260")
    assert not is_defensive_code("SH515880")
    assert float(BOND_ERP_ANN) == 0.051


def test_need_frame_vs_bond_anchor_scales_with_vol_ratio():
    pool = _load_pool()
    n = 80
    dates = pd.bdate_range("2025-11-13", periods=n)
    rng = np.random.default_rng(3)
    # Subject ~2x bond vol → need ≈ 2 × BOND_ERP
    bond_rets = rng.normal(0.0, 0.002, size=n)
    subj_rets = rng.normal(0.0, 0.004, size=n)
    bond_close = 100.0 * np.cumprod(1.0 + bond_rets)
    subj_close = 100.0 * np.cumprod(1.0 + subj_rets)
    ohlcv = pd.DataFrame(
        {
            "datetime": dates,
            "$open": subj_close,
            "$high": subj_close * 1.001,
            "$low": subj_close * 0.999,
            "$close": subj_close,
            "$volume": np.full(n, 1e6),
        }
    )
    from etf_daily.lib.regime_transition_plot import compute_volatility_regime_frame

    vf_b = compute_volatility_regime_frame(
        pd.Series(bond_close, index=dates, name="$close")
    )
    assert vf_b is not None and not vf_b.empty
    vf_b = vf_b.copy()
    vf_b["as_of"] = pd.to_datetime(vf_b["as_of"]).dt.normalize()
    indexed = vf_b.set_index("as_of")
    out = pool._need_frame_vs_anchor(
        ohlcv,
        indexed["rv20"],
        rv5_anchor=indexed["rv5"],
        erp_ann=BOND_ERP_ANN,
    )
    assert out is not None and not out.empty
    last = float(out["r_need_ann"].iloc[-1])
    assert np.isfinite(last)
    # Roughly near 2x premium (noise allowed)
    assert 0.06 < last < 0.18
