"""Fig9 aux_fail_lo: re-touch lower rail sells; latch until near-hi."""
from __future__ import annotations

import numpy as np
import pandas as pd

from etf_daily.scripts.backtest_fig9_touch import (
    fail_lo_eod_flags,
    leave_hi_confirmed,
    leave_lo_confirmed,
    run_aux_fail_lo_state_machine,
    tag_sell_kind,
)


def _frame() -> pd.DataFrame:
    n = 30
    idx = pd.bdate_range("2021-03-01", periods=n)
    lo = np.zeros(n, dtype=bool)
    hi = np.zeros(n, dtype=bool)
    lo[:3] = True  # buy streak
    lo[8:12] = True  # failed re-touch
    hi[20] = True  # re-arm
    lo[24:26] = True  # next buy after re-arm
    px = np.linspace(2.0, 1.5, n)
    return pd.DataFrame(
        {
            "px": px,
            "fig9_near_lo": lo,
            "fig9_near_hi": hi,
            "fig9_mid_stretch_sell": False,
            "aux_buy": lo,
            "aux_sell": False,
            "locmin": False,
            "locmax": False,
            "fig9_g": 0.20,
        },
        index=idx,
    )


def test_fail_lo_sells_on_retouch_and_latches_until_near_hi():
    frame = _frame()
    trades, buy_sig, sell_sig = run_aux_fail_lo_state_machine(frame)
    trades = tag_sell_kind(trades, frame)
    assert list(trades["signal_buy"]) == ["2021-03-01", "2021-04-02"]
    assert trades.iloc[0]["signal_sell"] == "2021-03-11"
    assert trades.iloc[0]["sell_kind"] == "再触下轨"
    assert not bool(buy_sig.loc["2021-03-12"])
    assert bool(buy_sig.loc["2021-04-02"])
    in_pos, buy_today, sell_today = fail_lo_eod_flags(frame, trades)
    assert in_pos is True
    assert buy_today is False
    assert sell_today is False


def test_fail_lo_skips_buy_when_fig9_g_negative():
    frame = _frame()
    frame["fig9_g"] = -0.24
    trades, buy_sig, _ = run_aux_fail_lo_state_machine(frame)
    assert trades.empty
    assert not bool(buy_sig.any())


def test_leave_lo_confirmed_after_three_off_days():
    idx = pd.bdate_range("2023-03-20", periods=8)
    lo = pd.Series([True, False, False, False, False, False, False, False], index=idx)
    g = pd.Series(0.10, index=idx)
    sig = leave_lo_confirmed(lo, g, off_days=3, g_max=0.25)
    # 3/20 near_lo, 3/21 leave, 3/22-23 off → confirm on third off day
    assert list(idx[sig].strftime("%Y-%m-%d")) == ["2023-03-23"]
    g_steep = pd.Series(0.50, index=idx)
    assert not bool(leave_lo_confirmed(lo, g_steep, off_days=3, g_max=0.25).any())


def test_leave_hi_confirmed_after_three_off_days():
    idx = pd.bdate_range("2024-01-10", periods=10)
    # 4-day near_hi ride through 1/15, leave 1/16 → default off_days=1 sells on leave day
    hi = pd.Series(
        [True, True, True, True, False, False, False, False, False, False],
        index=idx,
    )
    sig = leave_hi_confirmed(hi, off_days=1, min_near_days=4)
    assert list(idx[sig].strftime("%Y-%m-%d")) == ["2024-01-16"]
    # optional 3-day off confirmation
    sig3 = leave_hi_confirmed(hi, off_days=3, min_near_days=4)
    assert list(idx[sig3].strftime("%Y-%m-%d")) == ["2024-01-18"]
    # short kiss (<4 days) must not confirm
    hi_short = pd.Series(
        [True, True, False, False, False, False, False, False, False, False],
        index=idx,
    )
    assert not bool(leave_hi_confirmed(hi_short, off_days=1, min_near_days=4).any())
