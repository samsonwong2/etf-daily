from __future__ import annotations

import numpy as np
import pandas as pd

from decision_pack.src.price_jump_flags import (
    JumpFlagConfig,
    diagnose_symbol_jumps,
    drop_split_log_returns,
    split_day_mask,
)


def _series_with_split_and_jump() -> pd.Series:
    # 20 quiet days, then -11% jump, then quiet, then 1:3 split
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.01, size=40)
    rets[25] = -0.1125  # genuine jump
    prices = 100.0 * np.exp(np.cumsum(np.insert(rets, 0, 0.0)))
    # apply 1:3 split on last day: price /= 3
    prices[-1] = prices[-2] / 3.0
    idx = pd.bdate_range("2025-01-01", periods=len(prices))
    return pd.Series(prices, index=idx)


def test_diagnose_split_vs_jump():
    close = _series_with_split_and_jump()
    end = close.index[-1]
    diag = diagnose_symbol_jumps(close, code="SH588170", end=end)
    assert "corp_split" in diag.flags
    assert diag.n_splits >= 1
    assert diag.n_jumps_20d >= 1 or "recent_jump" in diag.flags or "jump_day_hist" in diag.flags
    assert diag.last_jump_ret is not None
    assert abs(diag.last_jump_ret) >= 0.08


def test_genuine_jump_not_marked_split():
    rng = np.random.default_rng(1)
    rets = rng.normal(0, 0.01, size=60)
    rets[-1] = -0.11
    prices = 100.0 * np.exp(np.cumsum(np.insert(rets, 0, 0.0)))
    idx = pd.bdate_range("2025-01-01", periods=len(prices))
    close = pd.Series(prices, index=idx)
    diag = diagnose_symbol_jumps(close, code="X", end=close.index[-1])
    assert "corp_split" not in diag.flags
    assert "recent_jump" in diag.flags
    assert abs(diag.last_jump_ret) >= 0.08


def test_drop_split_log_returns_removes_split_day():
    close = _series_with_split_and_jump()
    mask = split_day_mask(close)
    assert bool(mask.iloc[-1])
    cleaned = drop_split_log_returns(
        close, end=close.index[-1], lookback_days=100, split_threshold=0.25
    )
    raw = np.diff(np.log(close.to_numpy(dtype=float)))
    assert cleaned.size == raw.size - int(mask.iloc[1:].sum())


def test_live_price_gap_flag():
    close = pd.Series(
        [1.0, 1.01, 1.02],
        index=pd.bdate_range("2025-01-01", periods=3),
    )
    diag = diagnose_symbol_jumps(
        close,
        code="Y",
        end=close.index[-1],
        live_close=3.0,  # ~3x gap
        cfg=JumpFlagConfig(live_gap_threshold=0.25),
    )
    assert "live_price_gap" in diag.flags
