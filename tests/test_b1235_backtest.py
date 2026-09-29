"""Tests for the vectorized B1235 backtest (synthetic, no Qlib / HTML)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from etf_daily.lib.b1235_backtest import (  # noqa: E402
    B1235Params,
    add_universe_baseline,
    basket_curve,
    build_symbol_panel,
    condition_frame,
    dedupe_signals,
    signal_mask,
)
from etf_daily.lib.fair_path_band_signals import (  # noqa: E402
    atr20,
    compute_multi_scale_frame,
    enrich_frame_for_checklist,
    evaluate_checklist_row,
)


def _synthetic_ohlcv(n: int = 700, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0008, 0.018, n)
    r[300:340] -= 0.012  # a drawdown so lower rails get touched
    close = 2.0 * np.exp(np.cumsum(r))
    open_ = close * np.exp(rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    idx = pd.bdate_range("2020-01-02", periods=n)
    return pd.DataFrame({"$open": open_, "$high": high, "$low": low, "$close": close}, index=idx)


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    return _synthetic_ohlcv()


@pytest.fixture(scope="module")
def panel(ohlcv: pd.DataFrame) -> pd.DataFrame:
    listing = ohlcv.index[0] - pd.Timedelta(days=200)
    return build_symbol_panel("SH000TST", ohlcv, listing_start=listing)


@pytest.mark.parametrize(
    "b3_mode,b3_parts",
    [
        ("default", ("above10", "above11", "reclaim10")),
        ("fig10_only", ("above10", "reclaim10")),
        ("fig11_or_reclaim", ("above11", "reclaim10")),
    ],
)
def test_condition_frame_matches_row_checklist(ohlcv, panel, b3_mode, b3_parts):
    ms = enrich_frame_for_checklist(compute_multi_scale_frame(ohlcv["$close"]))
    atr = atr20(ohlcv)
    listing = ohlcv.index[0] - pd.Timedelta(days=200)
    cf = condition_frame(panel, B1235Params(b3_parts=b3_parts))
    assert int(cf.all(axis=1).sum()) > 20
    for dt, row in ms.iterrows():
        a = float(atr.loc[dt])
        cr = evaluate_checklist_row(
            row,
            code="SH000TST",
            atr=a if np.isfinite(a) else None,
            listing_age_years=(dt - listing).days / 365.25,
            b3_mode=b3_mode,
        )
        want = (
            cr.flags["B1_position"],
            cr.flags["B2_long_ok"],
            cr.flags["B3_short_reclaim"],
            cr.flags["B5_rr_ok"],
        )
        got = tuple(bool(x) for x in cf.loc[dt, ["B1", "B2", "B3", "B5"]])
        assert got == want, dt


def test_forward_return_starts_next_open(ohlcv, panel):
    i = 400
    want = ohlcv["$close"].iloc[i + 20] / ohlcv["$open"].iloc[i + 1] - 1.0
    assert panel["fwd_20"].iloc[i] == pytest.approx(want)
    assert panel["fwd_20"].iloc[-20:].isna().all()


def test_dedupe_cooldown(panel):
    mask = pd.Series(False, index=panel.index)
    mask.iloc[[100, 103, 109, 110, 125]] = True
    kept = dedupe_signals(panel, mask, cooldown=10)
    assert list(np.flatnonzero(kept.to_numpy())) == [100, 110, 125]


def test_basket_single_signal_compounds_to_fwd20(ohlcv, panel):
    pn = add_universe_baseline(panel)
    mask = pd.Series(False, index=pn.index)
    mask.iloc[400] = True
    daily = basket_curve(pn, {"SH000TST": ohlcv}, mask, hold=20)
    assert (1.0 + daily).prod() - 1.0 == pytest.approx(pn["fwd_20"].iloc[400])
    assert int((daily != 0).sum()) == 20


def test_signal_mask_extra_filter(panel):
    base = signal_mask(panel, B1235Params(use=()))
    young = signal_mask(panel, B1235Params(use=(), extra=("age_ge_1y",)))
    assert young.sum() < base.sum()
    assert not young[panel["age_years"] < 1.0].any()


def test_cli_dispatches_b1235_backtest(monkeypatch, tmp_path):
    from etf_daily import cli
    from etf_daily.scripts import backtest_b1235

    seen: list[list[str]] = []
    monkeypatch.setattr(backtest_b1235, "main", lambda argv: seen.append(argv) or 0)
    assert cli.main(["b1235-backtest", "--listing-dir", str(tmp_path), "--split-date", "2021-01-01"]) == 0
    assert seen == [["--listing-dir", str(tmp_path), "--split-date", "2021-01-01"]]
