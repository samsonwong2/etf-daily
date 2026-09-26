from __future__ import annotations

import math

import pytest

from etf_daily.scripts.generate_short_horizon_board import (
    _action_label,
    _action_sort_key,
    _price_from_logret,
    _risk_reward,
    _signal_tier,
    _triple_barrier_ev,
)
from etf_daily.lib.short_horizon_mlfam import ShortHorizonConfig, ShortHorizonMetrics


def _metrics(**kwargs) -> ShortHorizonMetrics:
    defaults = dict(
        trend_label="flat",
        trend_t=None,
        trend_p=None,
        trend_win=5,
        p_up=0.4,
        p_down=0.35,
        p_timeout=0.25,
        barrier=0.02,
        direction="不下注",
        confidence=0.05,
        bet_size=0.0,
        q05=-0.03,
        q50=0.0,
        q95=0.03,
        reliable=True,
        basis="趋势扫描+三重门(MC)",
    )
    defaults.update(kwargs)
    return ShortHorizonMetrics(**defaults)


def test_price_from_logret():
    assert _price_from_logret(100.0, 0.05) == pytest.approx(100.0 * math.exp(0.05))
    assert _price_from_logret(100.0, -0.02) == pytest.approx(100.0 * math.exp(-0.02))
    assert _price_from_logret(None, 0.05) is None
    assert _price_from_logret(0.0, 0.05) is None


def test_triple_barrier_ev_and_risk_reward():
    assert _triple_barrier_ev(0.5, 0.3, 0.02) == pytest.approx(0.004)
    assert _triple_barrier_ev(None, 0.3, 0.02) is None
    assert _risk_reward(0.5, 0.25) == pytest.approx(2.0)
    assert _risk_reward(0.5, 0.0) is None


def test_signal_tier_bet():
    cfg = ShortHorizonConfig(conf_min=0.15, t_thresh=2.0)
    m = _metrics(direction="up", bet_size=0.05, trend_label="up", trend_t=3.0, confidence=0.2)
    assert _signal_tier(m, sh_cfg=cfg) == ("下注", "")


def test_signal_tier_observe_low_confidence():
    cfg = ShortHorizonConfig(conf_min=0.15, t_thresh=2.0)
    m = _metrics(
        direction="不下注",
        trend_label="up",
        trend_t=5.95,
        confidence=0.024,
    )
    signal, reason = _signal_tier(m, sh_cfg=cfg)
    assert signal == "观察"
    assert "置信" in reason


def test_signal_tier_avoid_flat():
    cfg = ShortHorizonConfig()
    m = _metrics(trend_label="flat", direction="不下注")
    assert _signal_tier(m, sh_cfg=cfg) == ("回避", "趋势flat")


def test_signal_tier_caution_alpha_unreliable():
    cfg = ShortHorizonConfig()
    m = _metrics(
        reliable=False,
        basis="趋势扫描+三重门(MC)·α不可靠",
        direction="up",
        bet_size=0.05,
        trend_label="up",
        trend_t=2.5,
    )
    assert _signal_tier(m, sh_cfg=cfg) == ("观察", "α不可靠")


def test_signal_tier_hard_block_missing_price():
    cfg = ShortHorizonConfig()
    m = _metrics(reliable=False, basis="缺价格·仅展示")
    assert _signal_tier(m, sh_cfg=cfg) == ("回避", "缺价格")


def test_action_label_holding_reduce():
    m = _metrics(direction="down", trend_label="down")
    assert _action_label(m, is_holding=True, path_state="回撤中", signal="观察") == "减/防守"


def test_action_label_holding_add():
    m = _metrics(direction="up", trend_label="up", bet_size=0.05)
    assert _action_label(m, is_holding=True, path_state="近高位", signal="下注") == "持有可加"


def test_action_label_candidate_buy():
    m = _metrics(direction="up", trend_label="up", bet_size=0.04)
    assert _action_label(m, is_holding=False, path_state="近高位", signal="下注") == "关注买入"


def test_action_label_candidate_avoid():
    m = _metrics(direction="down", trend_label="down")
    assert _action_label(m, is_holding=False, path_state="破位", signal="回避") == "回避"


def test_action_sort_key_holding_first():
    holding_bet = {
        "code": "A",
        "_is_holding": True,
        "_signal": "下注",
        "_ev_abs": 0.01,
        "_unreliable": False,
    }
    candidate_bet = {
        "code": "B",
        "_is_holding": False,
        "_signal": "下注",
        "_ev_abs": 0.05,
        "_unreliable": False,
    }
    assert _action_sort_key(holding_bet) < _action_sort_key(candidate_bet)


def test_action_sort_key_bet_before_observe():
    bet = {"code": "A", "_is_holding": False, "_signal": "下注", "_ev_abs": 0.01, "_unreliable": False}
    observe = {"code": "B", "_is_holding": False, "_signal": "观察", "_ev_abs": 0.05, "_unreliable": False}
    assert _action_sort_key(bet) < _action_sort_key(observe)
