"""Vectorized B1235 checklist backtest (research only).

Builds one pooled daily panel (code × date) with the raw fig7–11 readings that
``evaluate_checklist_row`` scores, plus forward outcomes measured from the
next bar's open. Variants (ablation / threshold sweeps) are then cheap boolean
expressions over that panel.

Causality: fig paths are trailing OLS, bands are expanding P95, ATR20 is
trailing, listing age is measured at each bar (not at the last bar). Outcomes
start at ``open[t+1]`` so a signal read after the T close is tradeable.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable

import numpy as np
import pandas as pd

from etf_daily.lib.fair_path_band_signals import (
    DEFAULT_DEEP_GAP,
    DEFAULT_MIN_RR,
    atr20,
    compute_multi_scale_frame,
    enrich_frame_for_checklist,
)

HORIZONS = (5, 10, 20, 40)
MAE_HORIZON = 20
YOUNG_LISTING_YEARS = 1.9

B1_PARTS = ("touch9", "touch10", "deep")
B2_PARTS = ("g7", "g8", "p8up", "p7up")
B3_PARTS = ("above10", "above11", "reclaim10")
B5_TARGETS = ("fig8", "fig10")


@dataclass(frozen=True)
class B1235Params:
    """Knobs for one variant. Defaults reproduce ``etf-daily b1235``."""

    deep_gap: float = DEFAULT_DEEP_GAP
    min_rr: float = DEFAULT_MIN_RR
    stop_atr: float = 1.0
    b1_parts: tuple[str, ...] = B1_PARTS
    b2_parts: tuple[str, ...] = B2_PARTS
    b3_parts: tuple[str, ...] = B3_PARTS
    b5_targets: tuple[str, ...] = B5_TARGETS
    use: tuple[str, ...] = ("B1", "B2", "B3", "B5")
    extra: tuple[str, ...] = field(default_factory=tuple)

    def with_(self, **kw: Any) -> "B1235Params":
        return replace(self, **kw)


def build_symbol_panel(
    code: str,
    ohlcv: pd.DataFrame,
    *,
    listing_start: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Per-date raw readings + forward outcomes for one symbol.

    ``ohlcv`` needs ``$open/$high/$low/$close`` and a DatetimeIndex.
    """
    ohlcv = ohlcv.sort_index()
    ohlcv = ohlcv[~ohlcv.index.duplicated(keep="last")]
    close = ohlcv["$close"].astype(float)
    ms = enrich_frame_for_checklist(compute_multi_scale_frame(close))
    atr = atr20(ohlcv).reindex(ms.index)

    start = pd.Timestamp(listing_start) if listing_start is not None else ms.index[0]
    age = (ms.index - start).days / 365.25

    px = ms["px"].astype(float)
    out = pd.DataFrame(index=ms.index)
    out["code"] = code
    out["px"] = px
    out["atr"] = atr
    out["age_years"] = np.asarray(age, dtype=float)
    for key in ("fig7", "fig8", "fig9", "fig10", "fig11"):
        out[f"{key}_path"] = ms[f"{key}_path"].astype(float)
        out[f"{key}_g"] = ms[f"{key}_g"].astype(float)
    out["fig8_gap"] = ms["fig8_gap"].astype(float)
    out["fig10_lower"] = ms["fig10_lower"].astype(float)
    out["touch9"] = ms["fig9_touch_lo"].astype(bool)
    out["touch10"] = ms["fig10_touch_lo"].astype(bool)
    out["above10"] = ms["fig10_above_path"].astype(bool)
    out["above11"] = ms["fig11_above_path"].astype(bool)
    out["p7up"] = ms["fig7_path_rising"].astype(bool)
    out["p8up"] = ms["fig8_path_rising"].astype(bool)
    out["reclaim10"] = ms["fig10_prev_touch_lo"].astype(bool) & (px > out["fig10_lower"])
    out["touch9_hi"] = ms["fig9_touch_hi"].astype(bool)
    out["touch10_hi"] = ms["fig10_touch_hi"].astype(bool)

    o = ohlcv["$open"].reindex(ms.index).astype(float)
    lo = ohlcv["$low"].reindex(ms.index).astype(float)
    entry = o.shift(-1)
    out["entry_px"] = entry
    for h in HORIZONS:
        out[f"fwd_{h}"] = px.shift(-h) / entry - 1.0
    # min low over bars t+1..t+MAE_HORIZON
    fut_low = lo[::-1].rolling(MAE_HORIZON, min_periods=MAE_HORIZON).min()[::-1].shift(-1)
    out[f"mae_{MAE_HORIZON}"] = fut_low / entry - 1.0
    out.index.name = "date"
    return out


def add_universe_baseline(panel: pd.DataFrame) -> pd.DataFrame:
    """Excess forward return vs the same-date equal-weight universe mean."""
    out = panel.copy()
    dates = out.index
    for h in HORIZONS:
        col = f"fwd_{h}"
        mean_by_date = out.groupby(level=0)[col].transform("mean")
        out[f"ex_{h}"] = out[col] - mean_by_date
    out.index = dates
    return out


def _rr(px: pd.Series, target: pd.Series, stop: pd.Series) -> pd.Series:
    ok = (px > 0) & (stop > 0) & (stop < px) & (target > px)
    rr = (target / px - 1.0) / (px / stop - 1.0)
    return rr.where(ok)


def condition_frame(panel: pd.DataFrame, p: B1235Params) -> pd.DataFrame:
    """B1/B2/B3/B5 booleans for ``p`` (mirrors ``evaluate_checklist_row``)."""
    px = panel["px"]
    use7 = panel["age_years"] >= YOUNG_LISTING_YEARS
    parts: dict[str, pd.Series] = {
        "touch9": panel["touch9"],
        "touch10": panel["touch10"],
        "deep": panel["fig8_gap"].le(p.deep_gap).fillna(False),
        "g7": use7 & panel["fig7_g"].gt(0).fillna(False),
        "g8": panel["fig8_g"].gt(0).fillna(False),
        "p8up": panel["p8up"],
        "p7up": use7 & panel["p7up"],
        "above10": panel["above10"],
        "above11": panel["above11"],
        "reclaim10": panel["reclaim10"],
    }

    def _any(names: Iterable[str]) -> pd.Series:
        acc = pd.Series(False, index=panel.index)
        for n in names:
            acc = acc | parts[n]
        return acc.astype(bool)

    atr = panel["atr"]
    stop = (px - p.stop_atr * atr).where(atr.notna(), px * 0.94)
    b5 = pd.Series(False, index=panel.index)
    for t in p.b5_targets:
        rr = _rr(px, panel[f"{t}_path"], stop)
        b5 = b5 | rr.ge(p.min_rr).fillna(False)

    out = pd.DataFrame(
        {
            "B1": _any(p.b1_parts),
            "B2": _any(p.b2_parts),
            "B3": _any(p.b3_parts),
            "B5": b5.astype(bool),
        },
        index=panel.index,
    )
    return out


EXTRA_FILTERS: dict[str, Any] = {
    "age_ge_1y": lambda pn: pn["age_years"] >= 1.0,
    "g7_and_g8_pos": lambda pn: pn["fig7_g"].gt(0).fillna(False) & pn["fig8_g"].gt(0).fillna(False),
    "g8_pos": lambda pn: pn["fig8_g"].gt(0).fillna(False),
    "fig10_g_pos": lambda pn: pn["fig10_g"].gt(0).fillna(False),
    "not_fig10_touch": lambda pn: ~pn["touch10"],
}


def signal_mask(panel: pd.DataFrame, p: B1235Params) -> pd.Series:
    cf = condition_frame(panel, p)
    ok = pd.Series(True, index=panel.index)
    for c in p.use:
        ok = ok & cf[c]
    for name in p.extra:
        ok = ok & EXTRA_FILTERS[name](panel).astype(bool)
    return ok & panel["entry_px"].notna()


def dedupe_signals(panel: pd.DataFrame, mask: pd.Series, *, cooldown: int) -> pd.Series:
    """Keep a code's signal only if its previous kept signal is ≥ cooldown bars back."""
    if cooldown <= 1:
        return mask
    keep = np.zeros(len(mask), dtype=bool)
    m = mask.to_numpy()
    codes = panel["code"].to_numpy()
    pos = pd.Series(np.arange(len(mask))).groupby(codes).cumcount().to_numpy()
    last: dict[str, int] = {}
    for i in np.flatnonzero(m):
        c = codes[i]
        prev = last.get(c)
        if prev is None or pos[i] - prev >= cooldown:
            keep[i] = True
            last[c] = pos[i]
    return pd.Series(keep, index=mask.index)


def summarize(sig: pd.DataFrame) -> dict[str, Any]:
    """Pooled stats; t-stat averages excess per signal date first (dates cluster)."""
    res: dict[str, Any] = {
        "n": int(len(sig)),
        "n_dates": int(sig.index.nunique()),
        "n_codes": int(sig["code"].nunique()) if len(sig) else 0,
    }
    if sig.empty:
        return res
    for h in HORIZONS:
        s = sig[f"fwd_{h}"].dropna()
        if len(s):
            res[f"mean_{h}"] = float(s.mean())
            res[f"hit_{h}"] = float((s > 0).mean())
        e = sig[f"ex_{h}"].dropna()
        if len(e):
            res[f"ex_{h}"] = float(e.mean())
    s20 = sig["fwd_20"].dropna()
    if len(s20):
        res["med_20"] = float(s20.median())
        res["p10_20"] = float(s20.quantile(0.10))
    mae = sig[f"mae_{MAE_HORIZON}"].dropna()
    if len(mae):
        res["mae_20"] = float(mae.mean())
        res["mae_20_p10"] = float(mae.quantile(0.10))
    by_date = sig["ex_20"].dropna().groupby(level=0).mean()
    if len(by_date) >= 3:
        sd = float(by_date.std(ddof=1))
        res["ex_20_t"] = float(by_date.mean() / (sd / np.sqrt(len(by_date)))) if sd > 0 else float("nan")
    return res


def run_variant(
    panel: pd.DataFrame,
    p: B1235Params,
    *,
    periods: dict[str, tuple[str, str]],
    cooldown: int = 10,
) -> dict[str, dict[str, Any]]:
    mask = dedupe_signals(panel, signal_mask(panel, p), cooldown=cooldown)
    sig = panel.loc[mask.to_numpy()]
    out: dict[str, dict[str, Any]] = {}
    for name, (a, b) in periods.items():
        sub = sig.loc[(sig.index >= pd.Timestamp(a)) & (sig.index <= pd.Timestamp(b))]
        out[name] = summarize(sub)
    return out


def basket_curve(
    panel: pd.DataFrame,
    ohlcvs: dict[str, pd.DataFrame],
    mask: pd.Series,
    *,
    hold: int = 20,
) -> pd.Series:
    """Daily return of an equal-weight basket of open positions (0 when flat).

    Each signal buys at ``open[t+1]`` and holds ``hold`` bars; the first day
    earns close/open, later days close/close.
    """
    m = pd.Series(mask.to_numpy(), index=panel.index)
    contrib: list[pd.DataFrame] = []
    for code, pc in panel.groupby("code", sort=False):
        sig = m[(panel["code"] == code).to_numpy()].to_numpy(bool)
        if not sig.any():
            continue
        oh = ohlcvs[code].reindex(pc.index)
        o, c = oh["$open"].to_numpy(float), oh["$close"].to_numpy(float)
        n = len(pc)
        day_ret = np.full(n, np.nan)
        day_ret[1:] = c[1:] / c[:-1] - 1.0
        first = c / o - 1.0
        rets, idx = [], []
        for i in np.flatnonzero(sig):
            e = i + 1
            for j in range(e, min(n, e + hold)):
                rets.append(first[j] if j == e else day_ret[j])
                idx.append(pc.index[j])
        if rets:
            contrib.append(pd.DataFrame({"date": idx, "r": rets}))
    if not contrib:
        return pd.Series(dtype=float)
    allr = pd.concat(contrib)
    daily = allr.groupby("date")["r"].mean()
    cal = panel.index.unique().sort_values()
    return daily.reindex(cal).fillna(0.0)


def curve_stats(daily: pd.Series) -> dict[str, float]:
    if daily.empty:
        return {}
    eq = (1.0 + daily).cumprod()
    years = max(len(daily) / 252.0, 1e-9)
    dd = eq / eq.cummax() - 1.0
    sd = float(daily.std(ddof=1))
    return {
        "cagr": float(eq.iloc[-1] ** (1.0 / years) - 1.0),
        "max_dd": float(dd.min()),
        "sharpe": float(daily.mean() / sd * np.sqrt(252.0)) if sd > 0 else float("nan"),
        "exposure": float((daily != 0).mean()),
    }


def simulate_exits(
    panel_code: pd.DataFrame,
    ohlcv: pd.DataFrame,
    signals: pd.Series,
    *,
    stop_atr: float = 1.0,
    target: str = "fig8",
    max_hold: int = 40,
) -> list[dict[str, Any]]:
    """Enter at next open; exit on ATR stop (intraday low), close ≥ target path, or max_hold."""
    idx = panel_code.index
    o = ohlcv["$open"].reindex(idx).to_numpy(float)
    lo = ohlcv["$low"].reindex(idx).to_numpy(float)
    c = ohlcv["$close"].reindex(idx).to_numpy(float)
    atr = panel_code["atr"].to_numpy(float)
    tgt = panel_code[f"{target}_path"].to_numpy(float)
    sig = signals.reindex(idx).fillna(False).to_numpy(bool)
    n = len(idx)
    trades: list[dict[str, Any]] = []
    i = 0
    while i < n - 1:
        if not sig[i]:
            i += 1
            continue
        e = i + 1
        entry = o[e]
        if not np.isfinite(entry) or entry <= 0:
            i += 1
            continue
        stop = entry - stop_atr * atr[i] if np.isfinite(atr[i]) else entry * 0.94
        tp = tgt[i]
        reason, exit_px, x = "open", c[n - 1], n - 1
        for j in range(e, n):
            if lo[j] <= stop:
                reason, exit_px, x = "stop", min(stop, o[j]), j
                break
            if np.isfinite(tp) and c[j] >= tp:
                reason, exit_px, x = "target", c[j], j
                break
            if j - e + 1 >= max_hold:
                reason, exit_px, x = "max_hold", c[j], j
                break
        trades.append(
            {
                "signal_date": idx[i],
                "ret": float(exit_px / entry - 1.0),
                "hold": int(x - e + 1),
                "exit": reason,
            }
        )
        i = x + 1
    return trades
