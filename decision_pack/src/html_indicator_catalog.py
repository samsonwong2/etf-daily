"""Causal event catalog from adaptive from_listing chart indicators.

Rebuilds discrete buy/sell candidate events from the same series the HTML
plots (MA / trend / vol / fair-gap / HMM triangles / extreme drop). Post-hoc
reversal diamonds and adaptive hold_up trade marks are intentionally excluded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from decision_pack.src.regime_transition_plot import (
    DEFAULT_VOL_CALM_PCT,
    DEFAULT_VOL_CLUSTER_PCT,
    DEFAULT_VOL_EXTREME_PCT,
    VOL_REGIME_CALM,
    VOL_REGIME_CLUSTER,
    VOL_REGIME_EXTREME,
    compute_volatility_regime_frame,
)
from decision_pack.src.regime_transition_walkforward import (
    extend_validation_with_pre_oos_walkforward,
)
from decision_pack.src.vol_need_return_board import (
    ANCHOR_ERP_ANN,
    HORIZON_5,
    HORIZON_20,
    fair_need_ann_series,
    gap_realized_minus_need,
)

# Chart-aligned constants (see plot_regime_transition_example).
TREND_HUG_BAND_PCT = 2.0
EXTREME_DROP_PCT = -7.0

# Names that must never appear as tradeable catalog events (leakage / old rules).
LEAKAGE_OR_BANNED_EVENT_NAMES = frozenset(
    {
        "early_reversal_bottom",
        "early_reversal_top",
        "final_reversal_bottom",
        "final_reversal_top",
        "reversal_revoked",
        "label_switch_reversal",
        "early_label_switch_reversal",
        "hold_up_enter",
        "hold_up_leave",
        "上涨买",
        "上涨卖",
        "early_leave_down",
        "early_enter_up",
        "ed_alert",
        "ru_recovery",
        "trade_bias",
    }
)


@dataclass(frozen=True)
class EventDef:
    name: str
    category: str
    description: str


EVENT_DEFS: tuple[EventDef, ...] = (
    # HMM切换
    EventDef("hmm_switch_up", "hmm", "上行分位切换（含 walkforward / 极端跌幅兜底）"),
    EventDef("hmm_switch_down", "hmm", "下行分位切换（含 walkforward / 极端跌幅兜底）"),
    # 均线趋势
    EventDef("ma5_cross_above_ma10", "ma_trend", "MA5 上穿 MA10"),
    EventDef("ma5_cross_below_ma10", "ma_trend", "MA5 下穿 MA10"),
    EventDef("ma5_cross_above_ma20", "ma_trend", "MA5 上穿 MA20"),
    EventDef("ma5_cross_below_ma20", "ma_trend", "MA5 下穿 MA20"),
    EventDef("ma10_cross_above_ma20", "ma_trend", "MA10 上穿 MA20"),
    EventDef("ma10_cross_below_ma20", "ma_trend", "MA10 下穿 MA20"),
    EventDef("close_cross_above_ma5", "ma_trend", "收盘上穿 MA5"),
    EventDef("close_cross_below_ma5", "ma_trend", "收盘下穿 MA5"),
    EventDef("close_cross_above_ma10", "ma_trend", "收盘上穿 MA10"),
    EventDef("close_cross_below_ma10", "ma_trend", "收盘下穿 MA10"),
    EventDef("close_cross_above_ma20", "ma_trend", "收盘上穿 MA20"),
    EventDef("close_cross_below_ma20", "ma_trend", "收盘下穿 MA20"),
    EventDef("trend_enter_up", "ma_trend", "MA20 趋势进入上涨态"),
    EventDef("trend_leave_up", "ma_trend", "MA20 趋势离开上涨态"),
    EventDef("trend_enter_down", "ma_trend", "MA20 趋势进入下跌态"),
    # 波动率
    EventDef("vol_pct_cross_above_30", "vol", "vol5_pct_120d 上穿 30%"),
    EventDef("vol_pct_cross_below_30", "vol", "vol5_pct_120d 下穿 30%"),
    EventDef("vol_pct_cross_above_70", "vol", "vol5_pct_120d 上穿 70%"),
    EventDef("vol_pct_cross_below_70", "vol", "vol5_pct_120d 下穿 70%"),
    EventDef("vol_pct_cross_above_90", "vol", "vol5_pct_120d 上穿 90%"),
    EventDef("vol_pct_cross_below_90", "vol", "vol5_pct_120d 下穿 90%"),
    EventDef("vol_enter_cluster", "vol", "进入波动率聚集"),
    EventDef("vol_enter_extreme", "vol", "进入波动率极端聚集"),
    EventDef("vol_enter_calm", "vol", "进入波动率平稳"),
    # 公平差额
    EventDef("gap_5d_cross_above_0", "fair_gap", "5日差额由负转正"),
    EventDef("gap_5d_cross_below_0", "fair_gap", "5日差额由正转负"),
    EventDef("gap_20d_cross_above_0", "fair_gap", "20日差额由负转正"),
    EventDef("gap_20d_cross_below_0", "fair_gap", "20日差额由正转负"),
    EventDef("r_need_ann_cross_above_erp", "fair_gap", "公平年化需求上穿 4.3% 锚"),
    EventDef("r_need_ann_cross_below_erp", "fair_gap", "公平年化需求下穿 4.3% 锚"),
    EventDef("r_need_ann_5_cross_above_erp", "fair_gap", "rv5 公平年化需求上穿 4.3% 锚"),
    EventDef("r_need_ann_5_cross_below_erp", "fair_gap", "rv5 公平年化需求下穿 4.3% 锚"),
    # 价格冲击
    EventDef("extreme_drop_le_7pct", "shock", "单日跌幅 ≤ −7%"),
)

EVENT_BY_NAME: dict[str, EventDef] = {e.name: e for e in EVENT_DEFS}
CATEGORIES: tuple[str, ...] = ("hmm", "ma_trend", "vol", "fair_gap", "shock")


def events_in_category(category: str) -> list[str]:
    return [e.name for e in EVENT_DEFS if e.category == category]


def catalog_markdown() -> str:
    lines = [
        "# Causal event catalog (from_listing indicators)",
        "",
        "All events are knowable from closes through T. Excluded: T+3/T+10",
        "reversal diamonds, early-confirm revoke, hold_up 上涨买/卖, ED/RU,",
        "early regime marks, trade-bias hover.",
        "",
        "| category | event | description |",
        "|---|---|---|",
    ]
    for e in EVENT_DEFS:
        lines.append(f"| `{e.category}` | `{e.name}` | {e.description} |")
    lines.append("")
    lines.append("## Banned / leakage names (must not appear)")
    lines.append("")
    for name in sorted(LEAKAGE_OR_BANNED_EVENT_NAMES):
        lines.append(f"- `{name}`")
    lines.append("")
    return "\n".join(lines)


def _cross_above(left: pd.Series, right: pd.Series | float) -> pd.Series:
    a = pd.to_numeric(left, errors="coerce")
    if isinstance(right, (int, float)):
        b = pd.Series(float(right), index=a.index)
    else:
        b = pd.to_numeric(right, errors="coerce").reindex(a.index)
    prev_a = a.shift(1)
    prev_b = b.shift(1)
    out = (prev_a <= prev_b) & (a > b)
    return out.fillna(False).astype(bool)


def _cross_below(left: pd.Series, right: pd.Series | float) -> pd.Series:
    a = pd.to_numeric(left, errors="coerce")
    if isinstance(right, (int, float)):
        b = pd.Series(float(right), index=a.index)
    else:
        b = pd.to_numeric(right, errors="coerce").reindex(a.index)
    prev_a = a.shift(1)
    prev_b = b.shift(1)
    out = (prev_a >= prev_b) & (a < b)
    return out.fillna(False).astype(bool)


def _enter_state(state: pd.Series, value: str) -> pd.Series:
    s = state.astype(object)
    prev = s.shift(1)
    out = (s == value) & (prev != value)
    return out.fillna(False).astype(bool)


def _leave_state(state: pd.Series, value: str) -> pd.Series:
    s = state.astype(object)
    prev = s.shift(1)
    out = (prev == value) & (s != value)
    return out.fillna(False).astype(bool)


def _trend_state_series(close: pd.Series, *, band_pct: float = TREND_HUG_BAND_PCT) -> pd.DataFrame:
    series = pd.Series(pd.to_numeric(close, errors="coerce"), copy=False)
    series.index = pd.to_datetime(series.index).normalize()
    ma = series.rolling(window=20, min_periods=20).mean()
    dist = (series / ma - 1.0) * 100.0
    state = pd.Series("hug", index=series.index, dtype=object)
    state[ma.isna()] = None
    state[dist >= band_pct] = "up"
    state[dist <= -band_pct] = "down"
    return pd.DataFrame({"ma20": ma, "dist_pct": dist, "state": state}, index=series.index)


def _sma(close: pd.Series, window: int) -> pd.Series:
    return pd.to_numeric(close, errors="coerce").rolling(window, min_periods=window).mean()


def _hmm_switch_masks(
    close: pd.Series,
    *,
    code: str,
    oos: pd.DataFrame | None,
    start_date: str | None,
    end_date: str | None,
    walkforward_pre_oos: bool = True,
) -> tuple[pd.Series, pd.Series]:
    idx = pd.to_datetime(close.index).normalize()
    up = pd.Series(False, index=idx)
    down = pd.Series(False, index=idx)
    if oos is None or oos.empty:
        return up, down
    start = str(pd.Timestamp(start_date or idx.min()).date())
    end = str(pd.Timestamp(end_date or idx.max()).date())
    if walkforward_pre_oos:
        empty_rev = pd.DataFrame(columns=["as_of", "code"])
        oos_ext, _rev = extend_validation_with_pre_oos_walkforward(
            oos,
            empty_rev,
            code=code,
            close=close,
            start_date=start,
            end_date=end,
        )
    else:
        oos_ext = oos
    if oos_ext is None or oos_ext.empty:
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


def build_event_frame(
    close: pd.Series,
    *,
    code: str,
    oos: pd.DataFrame | None = None,
    rv20_anchor: pd.Series | None = None,
    rv5_anchor: pd.Series | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    erp_ann: float = ANCHOR_ERP_ANN,
    extreme_drop_pct: float = EXTREME_DROP_PCT,
    walkforward_pre_oos: bool = True,
) -> pd.DataFrame:
    """Build daily boolean event columns aligned to ``close`` index.

    Parameters
    ----------
    close
        Price series (DatetimeIndex).
    oos
        Official ``signals_oos`` (all codes OK); optionally extended with
        pre-OOS walkforward triangles when ``walkforward_pre_oos`` is True.
    rv20_anchor / rv5_anchor
        SH510300 vol series for fair-need events; if missing, fair_gap cols are False.
    walkforward_pre_oos
        If False, use official OOS switches only (much faster for pool scans).
    """
    s = pd.to_numeric(close, errors="coerce").copy()
    s.index = pd.to_datetime(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if start_date is not None:
        s = s[s.index >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        s = s[s.index <= pd.Timestamp(end_date).normalize()]
    if s.empty:
        return pd.DataFrame(columns=[e.name for e in EVENT_DEFS])

    out = pd.DataFrame(index=s.index)
    for e in EVENT_DEFS:
        out[e.name] = False

    ma5 = _sma(s, 5)
    ma10 = _sma(s, 10)
    ma20 = _sma(s, 20)
    out["ma5_cross_above_ma10"] = _cross_above(ma5, ma10)
    out["ma5_cross_below_ma10"] = _cross_below(ma5, ma10)
    out["ma5_cross_above_ma20"] = _cross_above(ma5, ma20)
    out["ma5_cross_below_ma20"] = _cross_below(ma5, ma20)
    out["ma10_cross_above_ma20"] = _cross_above(ma10, ma20)
    out["ma10_cross_below_ma20"] = _cross_below(ma10, ma20)
    out["close_cross_above_ma5"] = _cross_above(s, ma5)
    out["close_cross_below_ma5"] = _cross_below(s, ma5)
    out["close_cross_above_ma10"] = _cross_above(s, ma10)
    out["close_cross_below_ma10"] = _cross_below(s, ma10)
    out["close_cross_above_ma20"] = _cross_above(s, ma20)
    out["close_cross_below_ma20"] = _cross_below(s, ma20)

    trend = _trend_state_series(s)
    out["trend_enter_up"] = _enter_state(trend["state"], "up")
    out["trend_leave_up"] = _leave_state(trend["state"], "up")
    out["trend_enter_down"] = _enter_state(trend["state"], "down")

    vol = compute_volatility_regime_frame(s)
    vol = vol.copy()
    vol["as_of"] = pd.to_datetime(vol["as_of"]).dt.normalize()
    vol = vol.set_index("as_of")
    vol = vol.reindex(s.index)
    pct = pd.to_numeric(vol["vol5_pct_120d"], errors="coerce")
    out["vol_pct_cross_above_30"] = _cross_above(pct, DEFAULT_VOL_CALM_PCT)
    out["vol_pct_cross_below_30"] = _cross_below(pct, DEFAULT_VOL_CALM_PCT)
    out["vol_pct_cross_above_70"] = _cross_above(pct, DEFAULT_VOL_CLUSTER_PCT)
    out["vol_pct_cross_below_70"] = _cross_below(pct, DEFAULT_VOL_CLUSTER_PCT)
    out["vol_pct_cross_above_90"] = _cross_above(pct, DEFAULT_VOL_EXTREME_PCT)
    out["vol_pct_cross_below_90"] = _cross_below(pct, DEFAULT_VOL_EXTREME_PCT)
    out["vol_enter_cluster"] = _enter_state(vol["vol_regime"], VOL_REGIME_CLUSTER)
    out["vol_enter_extreme"] = _enter_state(vol["vol_regime"], VOL_REGIME_EXTREME)
    out["vol_enter_calm"] = _enter_state(vol["vol_regime"], VOL_REGIME_CALM)

    day_pct = s.pct_change() * 100.0
    out["extreme_drop_le_7pct"] = (day_pct <= float(extreme_drop_pct)).fillna(False)

    if rv20_anchor is not None and not rv20_anchor.empty:
        a20 = pd.to_numeric(rv20_anchor, errors="coerce")
        a20.index = pd.to_datetime(a20.index).normalize()
        a20 = a20[~a20.index.duplicated(keep="last")].sort_index()
        rv20 = pd.to_numeric(vol["rv20"], errors="coerce")
        need = fair_need_ann_series(rv20, a20, erp_ann=erp_ann).reindex(s.index)
        gap20 = gap_realized_minus_need(s, need, bars=HORIZON_20)
        gap5_need = need
        if rv5_anchor is not None and not rv5_anchor.empty and "rv5" in vol.columns:
            a5 = pd.to_numeric(rv5_anchor, errors="coerce")
            a5.index = pd.to_datetime(a5.index).normalize()
            a5 = a5[~a5.index.duplicated(keep="last")].sort_index()
            need5 = fair_need_ann_series(
                pd.to_numeric(vol["rv5"], errors="coerce"), a5, erp_ann=erp_ann
            ).reindex(s.index)
            if need5 is not None and not need5.empty:
                gap5_need = need5
        gap5 = gap_realized_minus_need(s, gap5_need, bars=HORIZON_5)
        g20 = pd.to_numeric(gap20["gap"], errors="coerce").reindex(s.index)
        g5 = pd.to_numeric(gap5["gap"], errors="coerce").reindex(s.index)
        out["gap_5d_cross_above_0"] = _cross_above(g5, 0.0)
        out["gap_5d_cross_below_0"] = _cross_below(g5, 0.0)
        out["gap_20d_cross_above_0"] = _cross_above(g20, 0.0)
        out["gap_20d_cross_below_0"] = _cross_below(g20, 0.0)
        out["r_need_ann_cross_above_erp"] = _cross_above(need, float(erp_ann))
        out["r_need_ann_cross_below_erp"] = _cross_below(need, float(erp_ann))
        out["r_need_ann_5_cross_above_erp"] = _cross_above(gap5_need, float(erp_ann))
        out["r_need_ann_5_cross_below_erp"] = _cross_below(gap5_need, float(erp_ann))

    up, down = _hmm_switch_masks(
        s,
        code=code,
        oos=oos,
        start_date=start_date,
        end_date=end_date,
        walkforward_pre_oos=walkforward_pre_oos,
    )
    out["hmm_switch_up"] = up.reindex(s.index).fillna(False).astype(bool)
    out["hmm_switch_down"] = down.reindex(s.index).fillna(False).astype(bool)

    for col in out.columns:
        out[col] = out[col].fillna(False).astype(bool)
    banned = set(out.columns) & LEAKAGE_OR_BANNED_EVENT_NAMES
    if banned:
        raise ValueError(f"leakage/banned events in catalog: {sorted(banned)}")
    return out


def combine_entry_signals(
    events: pd.DataFrame,
    entry_names: Iterable[str],
    *,
    how: str = "and",
) -> pd.Series:
    names = [str(n) for n in entry_names]
    if not names:
        return pd.Series(False, index=events.index)
    missing = [n for n in names if n not in events.columns]
    if missing:
        raise KeyError(f"unknown entry events: {missing}")
    mats = [events[n].fillna(False).astype(bool) for n in names]
    if how == "and":
        out = mats[0]
        for m in mats[1:]:
            out = out & m
        return out
    if how == "or":
        out = mats[0]
        for m in mats[1:]:
            out = out | m
        return out
    raise ValueError(f"unsupported how={how}")


def event_meta_rows() -> list[dict[str, Any]]:
    return [
        {"name": e.name, "category": e.category, "description": e.description}
        for e in EVENT_DEFS
    ]


__all__ = [
    "CATEGORIES",
    "EVENT_BY_NAME",
    "EVENT_DEFS",
    "EventDef",
    "EXTREME_DROP_PCT",
    "LEAKAGE_OR_BANNED_EVENT_NAMES",
    "TREND_HUG_BAND_PCT",
    "build_event_frame",
    "catalog_markdown",
    "combine_entry_signals",
    "event_meta_rows",
    "events_in_category",
]
