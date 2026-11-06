"""What tomorrow's close would do to the rules7 plan.

The report's actions are decided at a close and filled at the next close, so
"an opportunity within ±10%" means: if tomorrow's close is x% from today's,
tomorrow's report would show a buy, a sell, or a continued hold.

Only the last bar changes, and every feature rules7 reads is causal, so the
appended bar is recomputed from the cached history instead of rebuilding the
whole frame. The anchor (CSI300) is assumed unchanged tomorrow.

This is a scenario scan, not a forecast.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from etf_daily.lib import rules7
from etf_daily.lib.fair_path_band_signals import FIG_PANELS
from etf_daily.lib.rules7_trade_plan import STOP_PCT, WATCH_Q
from etf_daily.lib.vol_need_return_board import ANCHOR_ERP_ANN

MOVE_MIN = -0.10
MOVE_MAX = 0.10
MOVE_STEP = 0.001
RAIL_Q = 0.95
NEAR = 0.01
RAIL_MIN_BARS = 63
ANN_BARS = 252.0
GAP_WINDOWS = (5, 20)

_ANCHOR_CACHE: dict[tuple[str, str], pd.Series] = {}


def move_grid() -> np.ndarray:
    n = int(round((MOVE_MAX - MOVE_MIN) / MOVE_STEP))
    return np.round(MOVE_MIN + np.arange(n + 1) * MOVE_STEP, 3)


def load_anchor_close(listing: Path, anchor_code: str) -> pd.Series:
    """Anchor close from the listing HTML, cached per process."""
    key = (str(listing), anchor_code.upper())
    cached = _ANCHOR_CACHE.get(key)
    if cached is not None:
        return cached
    from etf_daily.lib.fair_path_band_signals import (
        find_regime_html,
        load_ohlcv_from_regime_html,
    )

    html = find_regime_html(Path(listing), anchor_code.upper())
    if html is None:
        raise FileNotFoundError(f"anchor {anchor_code} missing under {listing}")
    close, _ohlcv = load_ohlcv_from_regime_html(html)
    _ANCHOR_CACHE[key] = close.astype(float)
    return _ANCHOR_CACHE[key]


def _compress(mask: np.ndarray, grid: np.ndarray) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            ranges.append((float(grid[start]), float(grid[i - 1])))
            start = None
    if start is not None:
        ranges.append((float(grid[start]), float(grid[-1])))
    return ranges


def _prep(frame: pd.DataFrame, anchor_close: pd.Series) -> dict | None:
    px = pd.to_numeric(frame["px"], errors="coerce").astype(float)
    if len(px) < GAP_WINDOWS[-1] + 1 or not np.isfinite(px.iloc[-1]):
        return None
    c = px.to_numpy(dtype=float)
    panels = {}
    for key, years, _label in FIG_PANELS:
        if key == "fig11":
            continue
        path = pd.to_numeric(frame.get(f"{key}_path"), errors="coerce").to_numpy(dtype=float)
        both = np.isfinite(c) & (c > 0) & np.isfinite(path) & (path > 0)
        panels[key] = {
            "years": float(years),
            "abs_hist": np.abs(np.log(c[both] / path[both])),
            "path_last": float(path[-1]) if np.isfinite(path[-1]) else float("nan"),
        }
    if any(len(p["abs_hist"]) < RAIL_MIN_BARS for p in panels.values()):
        return None
    fig7_gap = pd.to_numeric(frame["fig7_gap"], errors="coerce").to_numpy(dtype=float)
    above = fig7_gap > 0
    run = c[rules7.FAST_RUN_BARS:] / c[:-rules7.FAST_RUN_BARS] - 1
    rets = pd.Series(c).pct_change(fill_method=None).dropna().to_numpy(dtype=float)
    peak = np.maximum.accumulate(np.where(np.isfinite(c), c, -np.inf))
    anchor = pd.to_numeric(anchor_close, errors="coerce").astype(float).dropna()
    return {
        "c": c,
        "last": float(c[-1]),
        "panels": panels,
        "log_ret": np.diff(np.log(c)),
        "anchor_log_ret": np.diff(np.log(anchor.to_numpy(dtype=float))),
        "ret_sum": float(rets.sum()),
        "ret_sumsq": float((rets ** 2).sum()),
        "ret_count": int(len(rets)),
        "peak_end": float(peak[-1]),
        "mdd": float((c / peak - 1).min()),
        "base_cross": int((above[1:] != above[:-1]).sum()),
        "last_gap_pos": bool(above[-1]),
        "run_hist_max": float(run[-(rules7.FAST_LOOKBACK - 1):].max()) if len(run) else float("nan"),
        "c_fast": float(c[-rules7.FAST_RUN_BARS]),
        "near_today": {
            RAIL_Q: bool(c[-1] <= float(frame["fig9_lower"].iloc[-1]) * (1 + NEAR)),
        },
    }


def _rv(log_ret: np.ndarray, new_log_ret: float, window: int) -> float:
    buf = np.append(log_ret[-(window - 1):], new_log_ret)
    if len(buf) < window or not np.isfinite(buf).all():
        return float("nan")
    return float(buf.std(ddof=0) * np.sqrt(ANN_BARS))


def _gap(prep: dict, p: float, new_log_ret: float, bars: int) -> float:
    rv = _rv(prep["log_ret"], new_log_ret, bars)
    arv = _rv(prep["anchor_log_ret"], 0.0, bars)
    if not (np.isfinite(rv) and np.isfinite(arv) and arv > 0):
        return float("nan")
    need = ANCHOR_ERP_ANN * rv / arv
    if need <= -1:
        return float("nan")
    realized = p / float(prep["c"][-bars]) - 1
    return float(realized - ((1 + need) ** (bars / ANN_BARS) - 1))


def _path_new(c: np.ndarray, p: float, years: float) -> float:
    from etf_daily.plots.plot_regime_transition_example import _trail_fit_min_bars

    trail = int(round(years * ANN_BARS))
    n = len(c)
    start = n + 1 - trail if trail > 0 and (n + 1) >= trail else 0
    sl = np.append(c[start:], p)
    ok = np.isfinite(sl) & (sl > 0)
    if int(ok.sum()) < _trail_fit_min_bars(years):
        return float("nan")
    jj = np.flatnonzero(ok).astype(float)
    slope, intercept = np.polyfit(jj, np.log(sl[ok]), 1)
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        return float("nan")
    return float(np.exp(intercept + slope * jj[-1]))


def _rail(abs_hist: np.ndarray, path_new: float, p: float, q: float) -> tuple[float, float]:
    if not (np.isfinite(path_new) and path_new > 0 and p > 0):
        return float("nan"), float("nan")
    hist = np.append(abs_hist, abs(np.log(p / path_new)))
    if len(hist) < RAIL_MIN_BARS:
        return float("nan"), float("nan")
    delta = float(np.quantile(hist, q))
    return path_new * float(np.exp(-delta)), path_new * float(np.exp(delta))


def _spec_at(code: str, prep: dict, p: float, move: float, fig7_path_new: float):
    m = prep["ret_count"] + 1
    total_sum = prep["ret_sum"] + move
    var = (prep["ret_sumsq"] + move ** 2 - total_sum ** 2 / m) / (m - 1)
    vol = float(np.sqrt(var) * np.sqrt(rules7.BARS_PER_YEAR)) if var > 0 else 0.0
    total = p / float(prep["c"][0]) - 1
    new_peak = max(prep["peak_end"], p)
    mdd = min(prep["mdd"], p / new_peak - 1)
    new_gap_pos = bool(np.isfinite(fig7_path_new) and fig7_path_new > 0 and p / fig7_path_new - 1 > 0)
    crossings = prep["base_cross"] + int(new_gap_pos != prep["last_gap_pos"])
    bars = len(prep["c"]) + 1
    feat = {
        "bars": bars,
        "vol": vol,
        "bh_total": float(total),
        "bh_cagr": rules7.cagr(float(total), bars),
        "bh_mdd": float(mdd),
        "cross_per_year": crossings / (bars / rules7.BARS_PER_YEAR),
    }
    _typ, _source, spec = rules7.resolve_spec(code, feat)
    return spec


def _buy_at(prep: dict, spec, p: float, lower: float, aux: bool, disc: bool, near_today: bool) -> bool:
    near = bool(np.isfinite(lower) and p <= lower * (1 + NEAR))
    run_new = p / prep["c_fast"] - 1
    hist = prep["run_hist_max"]
    fast = bool((np.isfinite(hist) and hist >= rules7.FAST_RUN) or run_new >= rules7.FAST_RUN)
    buy = ((near and not near_today) and aux) or (bool(spec.b1) and disc and near)
    if spec.fast_block and fast:
        return False
    return bool(buy)


def scan_signals(code: str, frame: pd.DataFrame, anchor_close: pd.Series) -> dict | None:
    """Buy-signal ranges over the ±10% grid, for the P95 rail and the P90 rail."""
    prep = _prep(frame, anchor_close)
    if prep is None:
        return None
    p90_lower_today, _ = _rail(prep["panels"]["fig9"]["abs_hist"], prep["panels"]["fig9"]["path_last"],
                               prep["last"], WATCH_Q)
    prep["near_today"][WATCH_Q] = bool(np.isfinite(p90_lower_today)
                                       and prep["last"] <= p90_lower_today * (1 + NEAR))
    grid = move_grid()
    buy95 = np.zeros(len(grid), dtype=bool)
    buy90 = np.zeros(len(grid), dtype=bool)
    for i, move in enumerate(grid):
        p = prep["last"] * (1 + move)
        new_log_ret = float(np.log(1 + move))
        paths = {key: _path_new(prep["c"], p, panel["years"]) for key, panel in prep["panels"].items()}
        gap7 = p / paths["fig7"] - 1 if paths["fig7"] > 0 else float("nan")
        gap8 = p / paths["fig8"] - 1 if paths["fig8"] > 0 else float("nan")
        disc = bool((np.isfinite(gap7) and gap7 <= rules7.B1_GAP) or (np.isfinite(gap8) and gap8 <= rules7.B1_GAP))
        gap5 = _gap(prep, p, new_log_ret, 5)
        gap20 = _gap(prep, p, new_log_ret, 20)
        ma5 = float((prep["c"][-4:].sum() + p) / 5)
        ma20 = float((prep["c"][-19:].sum() + p) / 20)
        _lo10, up10 = _rail(prep["panels"]["fig10"]["abs_hist"], paths["fig10"], p, RAIL_Q)
        aux = bool(p < ma20 and ma5 < ma20 and np.isfinite(gap5) and gap5 <= 0
                   and np.isfinite(gap20) and gap20 <= 0
                   and not (np.isfinite(up10) and p >= up10))
        spec = _spec_at(code, prep, p, move, paths["fig7"])
        for q, out in ((RAIL_Q, buy95), (WATCH_Q, buy90)):
            lower, _ = _rail(prep["panels"]["fig9"]["abs_hist"], paths["fig9"], p, q)
            out[i] = _buy_at(prep, spec, p, lower, aux, disc, prep["near_today"][q])
    base_spec = _spec_at(code, prep, prep["last"], 0.0, prep["panels"]["fig7"]["path_last"])
    blocked = bool(base_spec.fast_block and np.isfinite(prep["run_hist_max"])
                   and prep["run_hist_max"] >= rules7.FAST_RUN)
    return {
        "last": prep["last"],
        "buy_ranges": _compress(buy95, grid),
        "p90_ranges": _compress(buy90, grid),
        "block": "快形态挡住" if blocked else None,
    }


def position_exits(positions: list | None, last_px: float) -> list[dict]:
    """Stop or expiry for every still-open signal. One dict per position."""
    exits: list[dict] = []
    for pos in positions or []:
        if not isinstance(pos, dict) or pos.get("status") != "open":
            continue
        item = {
            "signal_date": pos.get("signal_date"),
            "entry_date": pos.get("entry_date"),
            "entry_px": pos.get("entry_px"),
            "days_left": pos.get("days_left"),
            "sell_kind": None,
            "sell_move": None,
            "hold_lo": None,
            "hold_hi": None,
        }
        days_left = pos.get("days_left")
        if days_left is not None and days_left <= 1:
            item["sell_kind"] = "expiry"
            exits.append(item)
            continue
        entry = pos.get("entry_px")
        if entry and np.isfinite(last_px) and last_px > 0:
            stop_px = float(entry) * (1 - STOP_PCT)
            item["sell_kind"] = "stop"
            item["sell_move"] = stop_px / last_px - 1
            item["hold_lo"] = item["sell_move"]
            item["hold_hi"] = MOVE_MAX
        exits.append(item)
    return exits


def scenario(code: str, frame: pd.DataFrame, anchor_close: pd.Series, positions: list | None = None,
             *, applicable: bool = True) -> dict:
    """Buy ranges for any close, plus an exit for every open signal."""
    empty = {"last": None, "buy_ranges": [], "p90_ranges": [], "block": None, "exits": []}
    if not applicable:
        empty["block"] = "规则不适用"
        return empty
    try:
        scanned = scan_signals(code, frame, anchor_close)
    except Exception:  # noqa: BLE001
        scanned = None
    if scanned is None:
        return empty
    last = scanned["last"]
    scanned["exits"] = position_exits(positions, float(last) if last is not None else float("nan"))
    return scanned
