"""Intraday T (做T) config — separate from garch_short_horizon holding rails."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IntradayTConfig:
    enabled: bool = True
    strategy: str = "zhen_t_chop_filter"
    horizon: int = 1
    barrier_mult: float = 1.0
    barrier_mult_sweep: tuple[float, ...] = (0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5)
    t_frac: float = 0.33
    min_p_tp: float = 0.03
    min_p_sl: float = 0.015
    max_hold_ret_for_zhen_t_pct: float = 2.0
    chop_hold_pct_sweep: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0)
    allow_zhen_t: bool = True
    allow_fan_t: bool = False
    rebuy_on_sl: bool = True


def intraday_t_config_from_raw(raw: dict[str, Any] | None) -> IntradayTConfig:
    if not raw or not isinstance(raw, dict):
        return IntradayTConfig()
    kwargs: dict[str, Any] = {}
    for key in IntradayTConfig.__dataclass_fields__:
        if key not in raw or raw[key] is None:
            continue
        val = raw[key]
        if key in ("barrier_mult_sweep", "chop_hold_pct_sweep"):
            kwargs[key] = tuple(float(x) for x in val)
        else:
            kwargs[key] = val
    return IntradayTConfig(**kwargs)


__all__ = [
    "IntradayTConfig",
    "intraday_t_config_from_raw",
]
