"""Causal three-state trend machines from from_listing chart series.

Each machine emits daily labels in {up, down, range} using only closes
through T. Trading mask is ``pred == up`` (enter on non-up→up, leave on up→non-up).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from etf_daily.lib.fwd_return_trend_truth import (
    STATE_DOWN,
    STATE_RANGE,
    STATE_UP,
)
from etf_daily.lib.html_indicator_catalog import TREND_HUG_BAND_PCT
from etf_daily.lib.regime_transition_plot import (
    VOL_REGIME_CALM,
    compute_volatility_regime_frame,
)
from etf_daily.lib.regime_transition_walkforward import (
    extend_validation_with_pre_oos_walkforward,
)
from etf_daily.lib.vol_need_return_board import (
    ANCHOR_ERP_ANN,
    HORIZON_20,
    fair_need_ann_series,
    gap_realized_minus_need,
)

MachineFn = Callable[..., pd.Series]


def _norm_close(close: pd.Series) -> pd.Series:
    s = pd.to_numeric(close, errors="coerce").copy()
    s.index = pd.to_datetime(s.index).normalize()
    return s[~s.index.duplicated(keep="last")].sort_index()


def _sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()


def machine_ma20_hug(
    close: pd.Series,
    *,
    band_pct: float = TREND_HUG_BAND_PCT,
    **_kwargs: Any,
) -> pd.Series:
    """Chart MA20 trend: ≥+band up, ≤−band down, else range."""
    s = _norm_close(close)
    ma = _sma(s, 20)
    dist = (s / ma - 1.0) * 100.0
    out = pd.Series(STATE_RANGE, index=s.index, dtype=object)
    out[ma.isna()] = np.nan
    out[dist >= float(band_pct)] = STATE_UP
    out[dist <= -float(band_pct)] = STATE_DOWN
    out.name = "ma20_hug"
    return out


def machine_above_ma20(close: pd.Series, **_kwargs: Any) -> pd.Series:
    """Close > MA20 → up, else range (down folded into range for long-only)."""
    s = _norm_close(close)
    ma = _sma(s, 20)
    out = pd.Series(STATE_RANGE, index=s.index, dtype=object)
    out[ma.isna()] = np.nan
    out[s > ma] = STATE_UP
    out.name = "above_ma20"
    return out


def machine_above_ma10(close: pd.Series, **_kwargs: Any) -> pd.Series:
    s = _norm_close(close)
    ma = _sma(s, 10)
    out = pd.Series(STATE_RANGE, index=s.index, dtype=object)
    out[ma.isna()] = np.nan
    out[s > ma] = STATE_UP
    out.name = "above_ma10"
    return out


def _hmm_switch_series(
    close: pd.Series,
    *,
    code: str,
    oos: pd.DataFrame | None,
    start_date: str | None,
    end_date: str | None,
) -> tuple[pd.Series, pd.Series]:
    s = _norm_close(close)
    up = pd.Series(False, index=s.index)
    down = pd.Series(False, index=s.index)
    if oos is None or oos.empty:
        return up, down
    start = str(pd.Timestamp(start_date or s.index.min()).date())
    end = str(pd.Timestamp(end_date or s.index.max()).date())
    empty_rev = pd.DataFrame(columns=["as_of", "code"])
    oos_ext, _ = extend_validation_with_pre_oos_walkforward(
        oos,
        empty_rev,
        code=code,
        close=s,
        start_date=start,
        end_date=end,
    )
    if oos_ext.empty:
        return up, down
    work = oos_ext.copy()
    work["as_of"] = pd.to_datetime(work["as_of"], errors="coerce").dt.normalize()
    work["code"] = work["code"].astype(str).str.upper()
    work = work[work["code"] == str(code).upper()]
    if "quantile_switch" in work.columns:
        work = work[work["quantile_switch"].fillna(False).astype(bool)]
    if work.empty or "switch_side" not in work.columns:
        return up, down
    for as_of, side in zip(work["as_of"], work["switch_side"].astype(str)):
        if as_of not in up.index:
            continue
        if side == "up":
            up.loc[as_of] = True
        elif side == "down":
            down.loc[as_of] = True
    return up, down


def machine_hmm_sticky(
    close: pd.Series,
    *,
    code: str = "UNKNOWN",
    oos: pd.DataFrame | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    **_kwargs: Any,
) -> pd.Series:
    """Sticky state from last causal HMM switch (up until down, else range)."""
    s = _norm_close(close)
    up_sw, down_sw = _hmm_switch_series(
        s, code=code, oos=oos, start_date=start_date, end_date=end_date
    )
    out = pd.Series(STATE_RANGE, index=s.index, dtype=object)
    state = STATE_RANGE
    for d in s.index:
        if bool(up_sw.loc[d]):
            state = STATE_UP
        elif bool(down_sw.loc[d]):
            state = STATE_RANGE
        out.loc[d] = state
    out.name = "hmm_sticky"
    return out


def machine_gap20_pos(
    close: pd.Series,
    *,
    rv20_anchor: pd.Series | None = None,
    erp_ann: float = ANCHOR_ERP_ANN,
    **_kwargs: Any,
) -> pd.Series:
    """gap_20d > 0 → up, else range."""
    s = _norm_close(close)
    out = pd.Series(STATE_RANGE, index=s.index, dtype=object)
    if rv20_anchor is None or rv20_anchor.empty:
        out[:] = np.nan
        out.name = "gap20_pos"
        return out
    vol = compute_volatility_regime_frame(s)
    vol["as_of"] = pd.to_datetime(vol["as_of"]).dt.normalize()
    vol = vol.set_index("as_of").reindex(s.index)
    a20 = pd.to_numeric(rv20_anchor, errors="coerce")
    a20.index = pd.to_datetime(a20.index).normalize()
    a20 = a20[~a20.index.duplicated(keep="last")].sort_index()
    need = fair_need_ann_series(
        pd.to_numeric(vol["rv20"], errors="coerce"), a20, erp_ann=erp_ann
    ).reindex(s.index)
    gap = gap_realized_minus_need(s, need, bars=HORIZON_20)
    g = pd.to_numeric(gap["gap"], errors="coerce").reindex(s.index)
    out[g.isna()] = np.nan
    out[g > 0] = STATE_UP
    out.name = "gap20_pos"
    return out


def machine_vol_calm_ma20(close: pd.Series, **_kwargs: Any) -> pd.Series:
    """Vol calm AND close > MA20 → up, else range."""
    s = _norm_close(close)
    ma = _sma(s, 20)
    vol = compute_volatility_regime_frame(s)
    vol["as_of"] = pd.to_datetime(vol["as_of"]).dt.normalize()
    vol = vol.set_index("as_of").reindex(s.index)
    calm = vol["vol_regime"].astype(str).eq(VOL_REGIME_CALM)
    out = pd.Series(STATE_RANGE, index=s.index, dtype=object)
    out[ma.isna() | vol["vol_regime"].isna()] = np.nan
    out[calm & (s > ma)] = STATE_UP
    out.name = "vol_calm_ma20"
    return out


MACHINES: dict[str, MachineFn] = {
    "ma20_hug": machine_ma20_hug,
    "above_ma20": machine_above_ma20,
    "above_ma10": machine_above_ma10,
    "hmm_sticky": machine_hmm_sticky,
    "gap20_pos": machine_gap20_pos,
    "vol_calm_ma20": machine_vol_calm_ma20,
}

MACHINE_IDS: tuple[str, ...] = tuple(MACHINES.keys())


def run_machine(machine_id: str, close: pd.Series, **kwargs: Any) -> pd.Series:
    if machine_id not in MACHINES:
        raise KeyError(f"unknown machine {machine_id}")
    return MACHINES[machine_id](close, **kwargs)


def pred_to_entry_exit(pred: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Boolean enter-up / leave-up signals from daily pred states."""
    p = pred.copy()
    p = p.where(p.notna(), STATE_RANGE)
    is_up = p.eq(STATE_UP)
    prev_raw = is_up.shift(1)
    prev = prev_raw.where(prev_raw.notna(), False).astype(bool)
    entry = (is_up & ~prev).fillna(False).astype(bool)
    exit_sig = ((~is_up) & prev).fillna(False).astype(bool)
    entry.name = "enter_up"
    exit_sig.name = "leave_up"
    return entry, exit_sig


def machine_descriptions() -> list[dict[str, str]]:
    return [
        {"id": "ma20_hug", "description": "MA20 ±2% hug band → up/down/range"},
        {"id": "above_ma20", "description": "close>MA20 → up else range"},
        {"id": "above_ma10", "description": "close>MA10 → up else range"},
        {
            "id": "hmm_sticky",
            "description": "sticky up after HMM up-switch until down-switch",
        },
        {"id": "gap20_pos", "description": "fair-need gap_20d>0 → up else range"},
        {
            "id": "vol_calm_ma20",
            "description": "vol calm AND close>MA20 → up else range",
        },
    ]


__all__ = [
    "MACHINE_IDS",
    "MACHINES",
    "machine_above_ma10",
    "machine_above_ma20",
    "machine_descriptions",
    "machine_gap20_pos",
    "machine_hmm_sticky",
    "machine_ma20_hug",
    "machine_vol_calm_ma20",
    "pred_to_entry_exit",
    "run_machine",
]
