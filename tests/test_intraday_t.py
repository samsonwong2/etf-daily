"""Tests for intraday_t config."""
from __future__ import annotations

from etf_daily.lib.config import load_decision_pack_config
from etf_daily.lib.intraday_t import IntradayTConfig, intraday_t_config_from_raw


def test_intraday_t_defaults_from_default_yaml():
    pack = load_decision_pack_config()
    cfg = intraday_t_config_from_raw(pack.raw.get("intraday_t"))
    assert cfg.enabled is True
    assert cfg.strategy == "zhen_t_chop_filter"
    assert cfg.horizon == 1
    assert cfg.barrier_mult == 1.0
    assert cfg.max_hold_ret_for_zhen_t_pct == 2.0
    assert cfg.t_frac == 0.33
    assert cfg.allow_fan_t is False
    assert 1.0 in cfg.barrier_mult_sweep
    assert 2.0 in cfg.chop_hold_pct_sweep


def test_intraday_t_config_from_raw_empty():
    cfg = intraday_t_config_from_raw(None)
    assert isinstance(cfg, IntradayTConfig)
    assert cfg.barrier_mult == 1.0
