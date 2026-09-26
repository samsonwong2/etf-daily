"""Unit tests for recovery_up overlay (no qlib)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from etf_daily.lib.recovery_up_overlay import (
    KEEP,
    PROMOTE,
    apply_recovery_up,
    decide_recommendation,
    ed_alert_bool_from_frame,
    first_up_date,
    recovery_up_mask,
    trading_days_between,
    up_frac_in_window,
)


def _synth(
    n: int = 80,
    *,
    trough_at: int = 20,
    recover_from: int = 40,
) -> pd.DataFrame:
    """Synthetic deep pit then recovery above MA20 with rising prior20."""
    rng = np.random.default_rng(0)
    close = np.full(n, 1.0, dtype=float)
    # grind down into pit
    for i in range(trough_at + 1):
        close[i] = 1.2 - 0.15 * (i / max(trough_at, 1))
    close[trough_at] = 0.85
    for i in range(trough_at + 1, recover_from):
        close[i] = 0.85 + 0.01 * (i - trough_at)
    for i in range(recover_from, n):
        close[i] = 0.95 + 0.02 * (i - recover_from + 1)
    # add tiny noise
    close = close + rng.normal(0, 0.001, size=n)
    ma20 = pd.Series(close).rolling(20, min_periods=1).mean().to_numpy()
    ma60 = pd.Series(close).rolling(60, min_periods=1).mean().to_numpy()
    prior20 = np.full(n, -0.1)
    for i in range(n):
        j = max(0, i - 20)
        if close[j] > 0:
            prior20[i] = close[i] / close[j] - 1.0
    as_of = pd.bdate_range("2026-01-02", periods=n)
    return pd.DataFrame(
        {
            "as_of": as_of,
            "$close": close,
            "MA20": ma20,
            "MA60": ma60,
            "prior20": prior20,
        }
    )


def test_ed_frame_missing_fail_closed():
    alert, ok = ed_alert_bool_from_frame(None, 10)
    assert alert is None and ok is False
    alert, ok = ed_alert_bool_from_frame(pd.DataFrame({"x": [1]}), 1)
    assert alert is None and ok is False


def test_ed_frame_alert_bool():
    ed = pd.DataFrame({"level": ["none", "alert", "watch"]})
    alert, ok = ed_alert_bool_from_frame(ed, 3)
    assert ok
    assert list(alert) == [False, True, False]


def test_fail_closed_none_ed_no_upgrades():
    px = _synth()
    labs = np.array(["range"] * len(px), dtype=object)
    m = recovery_up_mask(px, labs, None, L=30, k=3, N=2, R=0.05, rec_min=0.05)
    assert not m.any()


def test_immutability_apply():
    labs = np.array(["range", "down", "up", "range"], dtype=object)
    mask = np.array([True, True, True, False])
    out = apply_recovery_up(labs, mask)
    assert list(labs) == ["range", "down", "up", "range"]
    assert list(out) == ["up", "down", "up", "range"]


def test_range_only_upgrade():
    labs = np.array(["down", "range"], dtype=object)
    out = apply_recovery_up(labs, np.array([True, True]))
    assert list(out) == ["down", "up"]


def test_predicate_fires_after_recovery():
    px = _synth(n=90, trough_at=25, recover_from=55)
    n = len(px)
    labs = np.array(["range"] * n, dtype=object)
    ed = np.zeros(n, dtype=bool)
    m = recovery_up_mask(
        px,
        labs,
        ed,
        L=40,
        k=3,
        N=3,
        R=0.08,
        rec_min=0.08,
        p20_min=0.02,
        k_ed=1,
    )
    assert m.any(), "expected some recovery_up bars after climb"
    # early pit bars should not fire
    assert not m[:30].any()


def test_ed_alert_blocks():
    px = _synth(n=90, trough_at=25, recover_from=55)
    n = len(px)
    labs = np.array(["range"] * n, dtype=object)
    ed = np.zeros(n, dtype=bool)
    base = recovery_up_mask(px, labs, ed, L=40, k=3, N=3, R=0.08, rec_min=0.08, p20_min=0.02)
    assert base.any()
    ed2 = base.copy()  # alert exactly on would-be upgrade bars
    blocked = recovery_up_mask(px, labs, ed2, L=40, k=3, N=3, R=0.08, rec_min=0.08, p20_min=0.02)
    assert not blocked.any()


def test_latch_holds_while_above_ma20():
    from etf_daily.lib.recovery_up_overlay import latch_recovery_up_mask

    n = 6
    px = pd.DataFrame(
        {
            "as_of": pd.bdate_range("2026-07-30", periods=n),
            "$close": np.array([1.05, 1.04, 1.03, 1.02, 0.90, 1.06]),
            "MA20": np.array([1.00, 1.00, 1.00, 1.00, 1.00, 1.00]),
            "MA60": np.ones(n),
        }
    )
    labs = np.array(["range"] * n, dtype=object)
    fire = np.array([True, False, False, False, False, True])
    ed = np.zeros(n, dtype=bool)
    latched = latch_recovery_up_mask(px, labs, fire, ed)
    assert list(latched) == [True, True, True, True, False, True]


def test_latch_fail_closed_without_ed():
    from etf_daily.lib.recovery_up_overlay import latch_recovery_up_mask

    px = pd.DataFrame(
        {
            "as_of": pd.bdate_range("2026-07-30", periods=2),
            "$close": [1.1, 1.1],
            "MA20": [1.0, 1.0],
            "MA60": [1.0, 1.0],
        }
    )
    labs = np.array(["range", "range"], dtype=object)
    fire = np.array([True, True])
    assert not latch_recovery_up_mask(px, labs, fire, None).any()
    assert decide_recommendation(True, True, True) == PROMOTE
    assert decide_recommendation(True, True, False) == KEEP
    assert decide_recommendation(False, True, True) == KEEP


def test_decide_recommendation():
    assert decide_recommendation(True, True, True) == PROMOTE
    assert decide_recommendation(True, True, False) == KEEP
    assert decide_recommendation(False, True, True) == KEEP


def test_decide_diagnostic_promote():
    from etf_daily.lib.recovery_up_overlay import (
        DIAGNOSTIC_PROMOTE,
        decide_diagnostic_promote,
    )

    assert decide_diagnostic_promote(focus_ok=True, flip_ok=True) == DIAGNOSTIC_PROMOTE
    assert decide_diagnostic_promote(focus_ok=True, flip_ok=False) == KEEP


def test_first_up_and_delay_helpers():
    as_of = pd.bdate_range("2026-07-20", periods=10)
    labs = np.array(["range"] * 10, dtype=object)
    labs[3] = "up"
    assert first_up_date(as_of, labs) == as_of[3].strftime("%Y-%m-%d")
    d0 = as_of[0].strftime("%Y-%m-%d")
    d1 = as_of[3].strftime("%Y-%m-%d")
    assert trading_days_between(as_of, d0, d1) == 3
    frac = up_frac_in_window(as_of, labs, d0, as_of[-1].strftime("%Y-%m-%d"))
    assert abs(frac - 0.1) < 1e-9
