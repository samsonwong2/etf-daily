from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from decision_pack.src.short_horizon_mlfam import (
    ShortHorizonConfig,
    compute_short_horizon_metrics,
    forward_cumulative_log_return,
    forward_log_returns,
    mc_quantiles,
    meta_label,
    realize_triple_barrier,
    trend_scan,
    triple_barrier_mc,
)


def _make_close_from_returns(returns: np.ndarray, start: str = "2020-01-01") -> pd.Series:
    prices = 100.0 * np.exp(np.cumsum(np.insert(returns, 0, 0.0)))
    index = pd.bdate_range(start, periods=len(prices))
    return pd.Series(prices, index=index)


def test_trend_scan_deterministic_and_upward_trend():
    rng = np.random.default_rng(42)
    drift = np.linspace(0.001, 0.003, 300)
    noise = rng.normal(0.0, 0.002, size=300)
    close = _make_close_from_returns(drift + noise)
    end = close.index[-1]
    label1, t1, w1, p1 = trend_scan(close, end, win_range=(5, 10), t_thresh=1.5)
    label2, t2, w2, p2 = trend_scan(close, end, win_range=(5, 10), t_thresh=1.5)
    assert label1 == label2
    assert t1 == t2
    assert w1 == w2
    assert p1 == p2
    assert label1 == "up"
    assert t1 is not None and t1 > 0


def test_triple_barrier_probabilities_sum_to_one():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.012, size=400)
    p_up, p_down, p_to, barrier = triple_barrier_mc(
        returns,
        horizon=10,
        barrier_mult=2.0,
        n_paths=1000,
        seed=12345,
    )
    assert p_up is not None and p_down is not None and p_to is not None
    assert barrier is not None and barrier > 0
    assert abs(p_up + p_down + p_to - 1.0) < 1e-9


def test_triple_barrier_reproduces_under_fixed_seed():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0, 0.01, size=500)
    first = triple_barrier_mc(
        returns, horizon=8, barrier_mult=2.0, n_paths=500, seed=99
    )
    second = triple_barrier_mc(
        returns, horizon=8, barrier_mult=2.0, n_paths=500, seed=99
    )
    assert first == second


def test_meta_label_no_bet_when_confidence_low():
    direction, conf, bet = meta_label(
        trend_label="up",
        p_up=0.40,
        p_down=0.35,
        tval=3.0,
        conf_min=0.15,
    )
    assert direction == "不下注"
    assert conf == pytest.approx(0.05)
    assert bet == 0.0


def test_meta_label_bets_when_confidence_high():
    direction, conf, bet = meta_label(
        trend_label="down",
        p_up=0.10,
        p_down=0.55,
        tval=-2.5,
        kelly_fraction=0.25,
        max_w=0.20,
        conf_min=0.15,
    )
    assert direction == "down"
    assert conf == pytest.approx(0.45)
    assert bet == pytest.approx(0.1125)


def test_mc_quantiles_ordering():
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0, 0.01, size=300)
    q05, q50, q95 = mc_quantiles(returns, horizon=10, n_paths=800, seed=7)
    assert q05 is not None and q50 is not None and q95 is not None
    assert q05 <= q50 <= q95


def test_low_n_fallback_basis():
    rng = np.random.default_rng(4)
    returns = rng.normal(0.0, 0.01, size=40)
    close = _make_close_from_returns(returns)
    end = close.index[-1]
    metrics = compute_short_horizon_metrics(
        close,
        end,
        cfg=ShortHorizonConfig(min_samples=120, horizon=10),
        flags=("low_n",),
    )
    assert metrics.reliable is False
    assert metrics.basis == "趋势扫描+三重门(MC)·低样本"
    assert metrics.p_up is not None
    assert metrics.p_down is not None


@pytest.mark.parametrize(
    "flags",
    [
        ("alpha_unreliable",),
        ["alpha_unreliable"],
        "alpha_unreliable",
        "low_n|alpha_unreliable",
    ],
)
def test_unreliable_flag_downgrades_even_with_enough_samples(flags):
    # Sample count is well above min_samples; MC still runs but reliability is downgraded.
    rng = np.random.default_rng(7)
    returns = rng.normal(0.001, 0.02, size=300)
    close = _make_close_from_returns(returns)
    end = close.index[-1]
    metrics = compute_short_horizon_metrics(
        close,
        end,
        cfg=ShortHorizonConfig(min_samples=60, horizon=10),
        flags=flags,
    )
    assert metrics.reliable is False
    assert "α不可靠" in metrics.basis or "低样本" in metrics.basis
    assert metrics.p_up is not None
    assert metrics.p_down is not None


def test_missing_price_hard_blocks_mc():
    rng = np.random.default_rng(7)
    returns = rng.normal(0.001, 0.02, size=300)
    close = _make_close_from_returns(returns)
    end = close.index[-1]
    for flags in (("low_n", "missing_price"), ("missing_price",)):
        metrics = compute_short_horizon_metrics(
            close,
            end,
            cfg=ShortHorizonConfig(min_samples=60, horizon=10),
            flags=flags,
        )
        assert metrics.reliable is False
        assert metrics.basis == "缺价格·仅展示"
        assert metrics.p_up is None
        assert metrics.bet_size == 0.0


def test_clean_flags_with_enough_samples_stay_reliable():
    rng = np.random.default_rng(7)
    returns = rng.normal(0.001, 0.02, size=300)
    close = _make_close_from_returns(returns)
    end = close.index[-1]
    for flags in ((), ("bounded_tail_right",), "", "bounded_tail_left"):
        metrics = compute_short_horizon_metrics(
            close,
            end,
            cfg=ShortHorizonConfig(min_samples=60, horizon=10),
            flags=flags,
        )
        assert metrics.reliable is True
        assert metrics.basis == "趋势扫描+三重门(MC)"
        assert metrics.p_up is not None


def test_realize_triple_barrier_up_on_first_day():
    assert realize_triple_barrier(np.array([0.05, -0.01]), barrier=0.03) == "up"


def test_realize_triple_barrier_down_before_up():
    assert realize_triple_barrier(np.array([-0.02, -0.02, 0.10]), barrier=0.03) == "down"


def test_realize_triple_barrier_timeout():
    assert realize_triple_barrier(np.array([0.01, -0.01, 0.005]), barrier=0.03) == "timeout"


def test_forward_log_returns_and_cumulative():
    idx = pd.bdate_range("2024-01-01", periods=20)
    prices = pd.Series(np.linspace(100, 110, 20), index=idx)
    end = idx[9]
    fwd = forward_log_returns(prices, end, horizon=5)
    assert fwd is not None and fwd.size == 5
    cum = forward_cumulative_log_return(fwd)
    assert cum is not None
    expected = float(np.log(prices.loc[idx[14]] / prices.loc[end]))
    assert cum == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "horizon,expected",
    [
        (5, (5, 5)),
        (10, (5, 10)),
        (20, (10, 20)),
    ],
)
def test_trend_win_range_scales_with_horizon(horizon, expected):
    cfg = ShortHorizonConfig(horizon=horizon)
    assert cfg.trend_win_range() == expected


def test_trend_win_range_legacy_fixed_when_disabled():
    cfg = ShortHorizonConfig(
        horizon=5,
        trend_win_from_horizon=False,
        win_min=5,
        win_max=10,
    )
    assert cfg.trend_win_range() == (5, 10)


def test_trend_t_differs_between_h5_and_h10():
    # Flat mid-window with a late ramp: 3–5d window vs 5–10d window pick different |t|.
    rng = np.random.default_rng(99)
    base = rng.normal(0.0, 0.001, size=290)
    mid = np.full(5, -0.008)
    ramp = np.linspace(0.006, 0.014, 5)
    returns = np.concatenate([base, mid, ramp])
    close = _make_close_from_returns(returns)
    end = close.index[-1]
    m5 = compute_short_horizon_metrics(
        close,
        end,
        cfg=ShortHorizonConfig(horizon=5, min_samples=60),
    )
    m10 = compute_short_horizon_metrics(
        close,
        end,
        cfg=ShortHorizonConfig(horizon=10, min_samples=60),
    )
    assert m5.trend_t is not None and m10.trend_t is not None
    assert m5.trend_t != m10.trend_t


def test_trend_scan_picks_min_p_not_max_abs_t():
    import math
    from scipy import stats

    from decision_pack.src.indicators import slice_through

    # Last 3 days: tiny clean bounce (high |t|, weaker p); longer windows: clearer drift.
    rng = np.random.default_rng(21)
    base = rng.normal(0.0, 0.012, size=293)
    mid = np.full(4, -0.006)
    bounce = np.array([0.004, 0.0045, 0.005])
    returns = np.concatenate([base, mid, bounce])
    close = _make_close_from_returns(returns)
    end = close.index[-1]
    win_range = (3, 7)
    t_thresh = 1.5

    available = slice_through(end, close)
    expected_win = 0
    expected_p = float("inf")
    for win in range(win_range[0], win_range[1] + 1):
        window = available.tail(win).astype(float)
        if (window <= 0).any():
            continue
        log_prices = np.log(window.values)
        x = np.arange(log_prices.size, dtype=float)
        _, _, _, pvalue, _ = stats.linregress(x, log_prices)
        pkey = float(pvalue) if math.isfinite(pvalue) else float("inf")
        if pkey < expected_p or (pkey == expected_p and win > expected_win):
            expected_p = pkey
            expected_win = win

    label, tval, win, pval = trend_scan(close, end, win_range=win_range, t_thresh=t_thresh)
    assert win == expected_win
    assert pval is not None
    assert label in ("up", "down", "flat")


def test_trend_p_present_when_reliable():
    rng = np.random.default_rng(7)
    returns = rng.normal(0.001, 0.02, size=300)
    close = _make_close_from_returns(returns)
    end = close.index[-1]
    metrics = compute_short_horizon_metrics(
        close,
        end,
        cfg=ShortHorizonConfig(min_samples=60, horizon=10),
    )
    assert metrics.reliable is True
    assert metrics.trend_p is not None


def test_trend_p_still_computed_when_unreliable():
    rng = np.random.default_rng(4)
    returns = rng.normal(0.0, 0.01, size=40)
    close = _make_close_from_returns(returns)
    end = close.index[-1]
    metrics = compute_short_horizon_metrics(
        close,
        end,
        cfg=ShortHorizonConfig(min_samples=120, horizon=10),
        flags=("low_n",),
    )
    assert metrics.reliable is False
    assert metrics.trend_p is not None or metrics.trend_win == 0
    assert metrics.p_up is not None
