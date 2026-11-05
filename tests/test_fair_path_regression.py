"""Legacy reproduction and fig7–11 smoke. Skips the golden if it was not frozen."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from plotly.subplots import make_subplots

from etf_daily.lib.fair_path_band_signals import (
    build_checklist_flags_frame,
    compute_multi_scale_frame,
)
from etf_daily.lib.fair_path_research import ewma_scaled_envelope
from etf_daily.paths import PLOTLY_OUTPUTS_DIR

GOLDEN_CODES = (
    "SH511260",
    "SH511090",
    "SH510300",
    "SH512530",
    "SH513310",
    "SH518880",
    "SZ159502",
    "SZ159985",
)
from etf_daily.plots.plot_regime_transition_example import (
    _add_own_log_trail_kline_row,
    causal_path_residual_signed_percentile,
)

GOLDEN = PLOTLY_OUTPUTS_DIR / "research" / "fair_path_fat_tail_audit" / "golden"


def _close(n: int = 180, seed: int = 21) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    shock = rng.normal(0.0002, 0.01, size=n)
    return pd.Series(50.0 * np.exp(np.cumsum(shock)), index=idx)


def test_synthetic_five_panel_frame_stays_under_budget():
    close = _close(800)
    t0 = time.perf_counter()
    ms = compute_multi_scale_frame(close)
    elapsed = time.perf_counter() - t0
    assert ms.shape[0] == 800
    assert elapsed < 5.0


def test_fig_row_legend_has_path_and_causal_p95():
    close = _close(120)
    df = pd.DataFrame(
        {
            "datetime": close.index,
            "$open": close.to_numpy(),
            "$high": close.to_numpy() * 1.01,
            "$low": close.to_numpy() * 0.99,
            "$close": close.to_numpy(),
        }
    )
    fig = make_subplots(rows=1, cols=1)
    _add_own_log_trail_kline_row(
        fig,
        df,
        row=1,
        close_fit=close,
        trail_years=0.25,
        erp_ann=0.04,
        align_from_long_history=False,
        hover_asset_name="TEST",
    )
    names = " ".join(str(t.name or "") for t in fig.data)
    blob = names + " ".join(
        str(item)
        for t in fig.data
        if getattr(t, "customdata", None) is not None
        for item in list(t.customdata)[:2]
    )
    assert "公平路径" in names
    assert "因果P95" in names
    assert "相对路径因果分位" in blob


@pytest.mark.skipif(not (GOLDEN / "SH511090.parquet").is_file(), reason="golden not frozen")
def test_legacy_frame_matches_frozen_golden():
    """Bit-exact against the frozen legacy baseline. One mid-length name."""
    code = "SH511090"
    assert code in GOLDEN_CODES
    frozen = pd.read_parquet(GOLDEN / f"{code}.parquet")
    close = frozen["px"]
    close.index = pd.to_datetime(frozen.index)
    ms = compute_multi_scale_frame(close)
    for col in ("fig7_path", "fig8_g", "fig10_lower", "fig11_upper"):
        got = ms[col].to_numpy(dtype=float)
        exp = frozen[col].to_numpy(dtype=float)
        np.testing.assert_allclose(got, exp, rtol=0, atol=0, equal_nan=True)
    pct = causal_path_residual_signed_percentile(ms["fig9_path"], close)
    np.testing.assert_allclose(
        pct.to_numpy(dtype=float),
        frozen["fig9_p95_pct"].to_numpy(dtype=float),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    up, lo, _ = ewma_scaled_envelope(ms["fig10_path"], close)
    np.testing.assert_allclose(
        up.to_numpy(dtype=float),
        frozen["fig10_ewma_upper"].to_numpy(dtype=float),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        lo.to_numpy(dtype=float),
        frozen["fig10_ewma_lower"].to_numpy(dtype=float),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    flags = build_checklist_flags_frame(ms, code=code)
    np.testing.assert_array_equal(
        flags["B1"].to_numpy(),
        frozen["B1"].to_numpy(),
    )
