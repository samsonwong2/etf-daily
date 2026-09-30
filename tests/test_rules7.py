"""Tests for the rules7 state machine (synthetic series, no HTML)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from etf_daily.lib.rules7 import (  # noqa: E402
    TYPE_CYCLE,
    TYPE_HIVOL,
    TYPE_SHORT,
    TYPE_SMOOTH,
    TYPE_TREND,
    RuleSpec,
    buy_sell_masks,
    classify,
    run_state_machine,
)


def _series(values, dtype=float) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=dtype), index=pd.bdate_range("2024-01-01", periods=len(values)))


def _mask(n: int, on: list[int]) -> pd.Series:
    a = np.zeros(n, bool)
    a[on] = True
    return _series(a, bool)


def test_trail_peak_starts_at_fill_bar():
    close = _series([10.0, 8.0, 7.6, 9.0, 8.0, 8.0])
    res = run_state_machine(close, _mask(6, [0]), _mask(6, []), trail=0.10)
    # 7.6 is -24% from the signal close but only -5% from the fill; no exit there.
    assert len(res["trades"]) == 1
    t = res["trades"][0]
    assert t["sell"] == close.index[4] and t["why"] == "回落"
    assert np.isclose(t["ret"], 8.0 / 8.0 - 1)


def test_g1_suppresses_rail_sell():
    close = _series(np.linspace(10, 12, 8))
    buy, sell = _mask(8, [0]), _mask(8, [5])
    plain = run_state_machine(close, buy, sell)
    assert plain["trades"][0]["sell"] == close.index[5]
    held = run_state_machine(close, buy, sell, g1=_mask(8, [3]))
    assert held["trades"] == []
    assert held["state"]["holding"] and held["state"]["g1_seen"]
    assert held["state"]["action"] == "持有"


def test_buy_on_last_bar_is_today_action():
    close = _series([10.0, 10.1, 10.2, 10.3])
    res = run_state_machine(close, _mask(4, [3]), _mask(4, []), trail=0.15)
    st = res["state"]
    assert st["action"] == "买"
    assert not st["holding"]
    assert st["fill_px"] is None
    assert res["exposure"] == 0.0


def test_fast_pattern_blocks_buy():
    n = 5
    masks = {
        "ae_buy": _mask(n, [1, 3]),
        "b1": _mask(n, [4]),
        "fast": _mask(n, [1, 4]),
        "ae_sell": _mask(n, []),
        "fig8_sell": _mask(n, []),
    }
    buy, _ = buy_sell_masks(masks, RuleSpec(b1=True, fast_block=True, sell="fig9", g1_hold=True, trail=0.15))
    assert list(np.flatnonzero(buy.to_numpy())) == [3]
    buy, _ = buy_sell_masks(masks, RuleSpec(b1=True, fast_block=False, sell="fig9", g1_hold=True, trail=0.15))
    assert list(np.flatnonzero(buy.to_numpy())) == [1, 3, 4]


def test_classify_order():
    base = dict(bars=1000, vol=0.20, bh_cagr=0.12, bh_mdd=-0.25, cross_per_year=8)
    assert classify({**base, "bars": 100}) == TYPE_SHORT
    assert classify({**base, "vol": 0.35, "bh_mdd": -0.47}) == TYPE_HIVOL
    assert classify({**base, "bh_mdd": -0.48}) == TYPE_CYCLE
    assert classify({**base, "bh_cagr": 0.02}) == TYPE_CYCLE
    assert classify({**base, "vol": 0.16, "cross_per_year": 10}) == TYPE_SMOOTH
    assert classify(base) == TYPE_TREND
