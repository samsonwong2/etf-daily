"""The ±10% scenario matches a full feature rebuild, and the exit math is exact."""
from __future__ import annotations

import numpy as np
import pandas as pd

from etf_daily.lib import rules7
from etf_daily.lib.rules7_scenario import position_exits, scan_signals
from etf_daily.scripts.backtest_fig9_touch import prepare_fig9_touch_frame
from etf_daily.scripts.run_reusable_extrema_state_machine import build_feature_frame_from_series


def _series(seed: int, n: int = 400, drift: float = 0.0002) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(drift + 0.01 * rng.standard_normal(n))), index=idx)


def _ohlcv(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {"$open": close, "$high": close, "$low": close, "$close": close}, index=close.index
    )


def _frame(close: pd.Series, anchor: pd.Series) -> pd.DataFrame:
    return prepare_fig9_touch_frame(
        build_feature_frame_from_series(close, _ohlcv(close), anchor, lookback=10)
    )


def _full_buy(close: pd.Series, anchor: pd.Series, move: float, code: str) -> bool:
    p = float(close.iloc[-1]) * (1 + move)
    idx = close.index[-1] + pd.tseries.offsets.BDay(1)
    extended = pd.concat([close, pd.Series([p], index=[idx])])
    anchor_ext = pd.concat([anchor, pd.Series([float(anchor.iloc[-1])], index=[idx])])
    frame = _frame(extended, anchor_ext)
    return bool(rules7.evaluate(code, frame)["buy"].iloc[-1])


def test_buy_ranges_match_full_rebuild():
    close, anchor = _series(7), _series(11, drift=0.0001)
    code = "SH510500"
    scanned = scan_signals(code, _frame(close, anchor), anchor)
    assert scanned is not None
    ranges = scanned["buy_ranges"]
    for move in (-0.09, -0.06, -0.04, -0.02, 0.0, 0.03, 0.07):
        in_range = any(lo <= move <= hi for lo, hi in ranges)
        assert in_range == _full_buy(close, anchor, move, code), move


def test_position_exits_stop_expiry_and_a_second_open_trade():
    exits = position_exits(
        [
            {"status": "open", "entry_px": 1.0, "days_left": 5, "signal_date": "2026-10-01"},
            {"status": "open", "entry_px": 2.0, "days_left": 1, "signal_date": "2026-10-15"},
            {"status": "closed", "entry_px": 1.0, "days_left": 5, "signal_date": "2026-09-01"},
        ],
        last_px=1.05,
    )
    assert len(exits) == 2
    assert exits[0]["sell_kind"] == "stop"
    assert abs(exits[0]["sell_move"] - (0.9 / 1.05 - 1)) < 1e-12
    assert exits[1]["sell_kind"] == "expiry"
    assert exits[1]["signal_date"] == "2026-10-15"
