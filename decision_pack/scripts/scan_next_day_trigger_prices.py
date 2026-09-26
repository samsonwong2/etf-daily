#!/usr/bin/env python3
"""Next-session trigger prices from a from_listing EOD batch.

As-of close T is known. For each equity ETF, binary-search the T+1 close that
would fire fig9 aux_edge buy/sell and six-state switches (exit ② / enter ② /
① overheat). History features are computed once; each hypothetical close only
updates the synthetic T+1 last row (same keys as a full causal rebuild).

Writes into --listing-dir:
    next_day_triggers_{next-day:%Y%m%d}.md
    next_day_triggers_{next-day:%Y%m%d}.csv
    next_day_triggers_{next-day:%Y%m%d}_hard_touch.csv
        （「图9/图10 硬触轨」表单独一份，列与格子与 md 该节相同）

Research scan only (does not change production hold_up). Signal close →
next-bar close fill.

Run from repo root (this project's Python)::

    cd ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean
    PYTHONPATH=. ~/etf-daily-output/python/envs/py312/bin/python \\
      decision_pack/scripts/scan_next_day_trigger_prices.py \\
      --listing-dir workspace/plotly_outputs/20260918_from_listing \\
      --as-of 2026-09-18 --next-day 2026-09-21 --jobs 8

换日：改 --listing-dir、--as-of（T 收盘日）、--next-day（下一交易日，跳过周末/节假日）。
需已有该日 from_listing 的 adaptive HTML（与 scan_fig9_aux_edge_eod 同一目录）。
listing 若在 ~/etf-daily-output/temp/plotly_outputs/{asof}_from_listing，把 --listing-dir 换成绝对路径即可。
"""
from __future__ import annotations

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import backtest_fig12_six_states_pool as pool  # noqa: E402
import backtest_fig9_touch as bt  # noqa: E402
import run_reusable_extrema_state_machine as sm  # noqa: E402
from decision_pack.src.fair_path_band_signals import (  # noqa: E402
    FIG_PANELS,
    find_regime_html,
    load_ohlcv_from_regime_html,
)
from decision_pack.src.fair_path_research import DEFAULT_EWMA_LAMBDA  # noqa: E402
from decision_pack.src.regime_transition_plot import (  # noqa: E402
    DEFAULT_VOL_PCT_LOOKBACK,
    DEFAULT_VOL_PCT_MIN_HISTORY,
    TRADING_DAYS_PER_YEAR,
    compute_realized_vol,
)
from decision_pack.src.six_states import (  # noqa: E402
    STATE_CHOP,
    STATE_DOWN,
    STATE_KNIFE,
    STATE_OVERHEAT,
    STATE_REPAIR,
    STATE_UP,
    STATE_WARMUP,
    classify_six_states,
)
from decision_pack.src.vol_need_return_board import ANCHOR_ERP_ANN  # noqa: E402

NEAR_PCT = 0.01
BISECT_ITERS = 12
GRID_RETS = (
    -0.15,
    -0.10,
    -0.07,
    -0.05,
    -0.03,
    -0.02,
    -0.01,
    0.0,
    0.01,
    0.02,
    0.03,
    0.05,
    0.07,
    0.10,
    0.15,
    0.20,
)
FOCUS_ABS = 0.03
LOOKBACK = 10
ANCHOR_DEFAULT = "SH510300"
TDY = float(TRADING_DAYS_PER_YEAR)
ENV_Q = 0.95
ENV_MIN_BARS = 63
EWMA_LAM = float(DEFAULT_EWMA_LAMBDA)
EWMA_KEYS = ("fig9", "fig10")
POS_MIN = 0.80
FIG10_POS_MIN = 0.90
FIG7_PREM = 0.05
VOL_CLUSTER = 0.70
_KNIFE_BARS = 10
_ANCHOR_CLOSE: dict[tuple[str, str, str], pd.Series] = {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--listing-dir", type=Path, required=True)
    p.add_argument("--as-of", required=True, help="last known close date (T)")
    p.add_argument("--next-day", required=True, help="hypothetical session (T+1)")
    p.add_argument("--anchor-code", default=ANCHOR_DEFAULT)
    p.add_argument("--lookback", type=int, default=LOOKBACK)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--include-bonds", action="store_true")
    p.add_argument(
        "--rewrite-md-from-csv",
        action="store_true",
        help="Skip rescan; rewrite md from existing next_day_triggers_{next-day}.csv",
    )
    return p.parse_args(argv)


def _resolve(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def _norm_close(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce").astype(float).copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index).normalize())
    return out[~out.index.duplicated(keep="last")].sort_index()


def _norm_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index).normalize())
    return out[~out.index.duplicated(keep="last")].sort_index()


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return x


def _bool(v: Any) -> bool:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return False
    return bool(v)


def _open_trade(trades: pd.DataFrame) -> dict[str, Any] | None:
    if trades is None or trades.empty:
        return None
    last = trades.iloc[-1]
    if bool(last.get("open", False)) or pd.isna(last.get("signal_sell")):
        return last.to_dict()
    return None


def _knife_run(px: pd.Series, p10: pd.Series) -> pd.Series:
    below = (pd.to_numeric(px, errors="coerce") < pd.to_numeric(p10, errors="coerce")).fillna(
        False
    )
    return below.astype(int).groupby((~below).cumsum()).cumsum()


def _append_bar(
    close: pd.Series, ohlcv: pd.DataFrame, day: pd.Timestamp, px: float
) -> tuple[pd.Series, pd.DataFrame]:
    d = pd.Timestamp(day).normalize()
    c = close.copy()
    c.loc[d] = float(px)
    o = ohlcv.copy()
    row = {col: np.nan for col in o.columns}
    for col in ("$open", "$high", "$low", "$close"):
        if col in o.columns:
            row[col] = float(px)
    o.loc[d] = row
    return c.sort_index(), o.sort_index()


def _extend_flat(close: pd.Series, day: pd.Timestamp) -> pd.Series:
    d = pd.Timestamp(day).normalize()
    out = close.copy()
    if d not in out.index:
        out.loc[d] = float(out.iloc[-1])
    return out.sort_index()


def _fmt_px_ret(px: float | None, px0: float | None) -> str:
    if px is None or px0 is None or px0 == 0:
        return "—"
    ret = px / px0 - 1.0
    return f"{px:.4f} ({ret:+.1%})"


def _fmt_ret(px: float | None, px0: float | None) -> str:
    if px is None or px0 is None or px0 == 0:
        return "—"
    return f"{px / px0 - 1.0:+.1%}"


def _fmt_rail(px: float | None, px0: float | None, *, already: bool) -> str:
    """Hard-touch rail: 已触 if T+1 flat still touches, else % to first touch."""
    body = _fmt_ret(px, px0)
    if body == "—":
        return "—"
    return f"已触 {body}" if already else body


def _pct(x: float | None, digits: int = 1) -> str:
    v = _f(x)
    if v is None:
        return "—"
    return f"{v:.{digits}%}"


def _truthy_in_pos(v: Any) -> bool:
    if v is True:
        return True
    if v is False or v is None:
        return False
    try:
        if pd.isna(v):
            return False
    except (TypeError, ValueError):
        pass
    return str(v).strip().lower() in {"true", "1", "yes"}


def _state_band(r: Any) -> str:
    a = _md_cell(r.get("state_m3"))
    b = _md_cell(r.get("state_flat"))
    c = _md_cell(r.get("state_p3"))
    if not (a or b or c):
        return "—"
    return f"{a or '—'} / {b or '—'} / {c or '—'}"


def _md_cell(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    if s.lower() in {"nan", "none", "nat", "<na>"}:
        return ""
    return s


def eval_next(
    *,
    close: pd.Series,
    ohlcv: pd.DataFrame,
    anchor_ext: pd.Series,
    next_day: pd.Timestamp,
    px: float,
    lookback: int,
) -> dict[str, Any]:
    c_ext, o_ext = _append_bar(close, ohlcv, next_day, px)
    frame = sm.build_feature_frame_from_series(
        c_ext, o_ext, anchor_ext, lookback=lookback
    )
    frame = bt.prepare_fig9_touch_frame(frame)
    if next_day not in frame.index:
        raise KeyError(f"synthetic bar {next_day.date()} missing from frame")
    buy_c, sell_c, _ = bt.variant_masks(frame, "aux_edge")
    states = classify_six_states(frame)
    row = frame.loc[next_day]
    run = _knife_run(frame["px"], frame["fig10_path"])
    return {
        "px": float(px),
        "state": str(states.loc[next_day]),
        "buy": bool(buy_c.loc[next_day]),
        "sell": bool(sell_c.loc[next_day]),
        "near_lo": _bool(row.get("fig9_near_lo")),
        "near_hi": _bool(row.get("fig9_near_hi")),
        "aux_buy": _bool(row.get("aux_buy")),
        "aux_sell": _bool(row.get("aux_sell")),
        "mid_stretch": _bool(row.get("fig9_mid_stretch_sell")),
        "touch_hi": _bool(row.get("fig9_touch_hi")) or _bool(row.get("fig10_touch_hi")),
        "fig9_touch_lo": _bool(row.get("fig9_touch_lo")),
        "fig9_touch_hi": _bool(row.get("fig9_touch_hi")),
        "fig10_touch_lo": _bool(row.get("fig10_touch_lo")),
        "fig10_touch_hi": _bool(row.get("fig10_touch_hi")),
        "fig9_pos": _f(row.get("fig9_pos")),
        "fig9_lower": _f(row.get("fig9_lower")),
        "fig9_upper": _f(row.get("fig9_upper")),
        "fig10_path": _f(row.get("fig10_path")),
        "fig10_lower": _f(row.get("fig10_lower")),
        "fig10_upper": _f(row.get("fig10_upper")),
        "knife_run": int(run.loc[next_day]) if pd.notna(run.loc[next_day]) else 0,
        "px_ge_fig10": bool(
            pd.to_numeric(row.get("px"), errors="coerce")
            >= pd.to_numeric(row.get("fig10_path"), errors="coerce")
        ),
    }


def _cached_eval(fn: Callable[[float], dict[str, Any]]) -> Callable[[float], dict[str, Any]]:
    cache: dict[float, dict[str, Any]] = {}

    def inner(px: float) -> dict[str, Any]:
        key = round(float(px), 8)
        hit = cache.get(key)
        if hit is None:
            hit = fn(px)
            cache[key] = hit
        return hit

    inner.cache = cache  # type: ignore[attr-defined]
    return inner


def _highest_true(
    pred: Callable[[float], bool], lo: float, hi: float, *, n: int = BISECT_ITERS
) -> float | None:
    """Largest x in [lo, hi] with pred True. None if never True at lo."""
    if hi < lo:
        lo, hi = hi, lo
    if not pred(lo):
        return None
    if pred(hi):
        return hi
    a, b = lo, hi
    for _ in range(n):
        mid = 0.5 * (a + b)
        if pred(mid):
            a = mid
        else:
            b = mid
    return a


def _lowest_true(
    pred: Callable[[float], bool], lo: float, hi: float, *, n: int = BISECT_ITERS
) -> float | None:
    """Smallest x in [lo, hi] with pred True. None if never True at hi."""
    if hi < lo:
        lo, hi = hi, lo
    if not pred(hi):
        return None
    if pred(lo):
        return lo
    a, b = lo, hi
    for _ in range(n):
        mid = 0.5 * (a + b)
        if pred(mid):
            b = mid
        else:
            a = mid
    return b


def _search_span(px0: float, guesses: list[float | None]) -> tuple[float, float]:
    lo = px0 * 0.80
    hi = px0 * 1.25
    for g in guesses:
        if g is None or not np.isfinite(g) or g <= 0:
            continue
        lo = min(lo, g * 0.97, px0 * 0.70)
        hi = max(hi, g * 1.03, px0 * 1.40)
    return max(lo, px0 * 0.40), min(hi, px0 * 2.50)


@dataclass
class _AnchorNext:
    rv5: float
    rv20: float


@dataclass
class EwmaState:
    """Causal EWMA envelope state through T (for a last-row T+1 step)."""

    abs_z: np.ndarray
    last_var: float | None
    last_r2: float | None
    last_upper: float
    last_lower: float


@dataclass
class FastCtx:
    close_arr: np.ndarray
    px0: float
    lookback: int
    log_ret: np.ndarray
    rv5_ddof1_hist: np.ndarray
    close_m4: np.ndarray
    close_m19: np.ndarray
    close_lag5: float
    close_lag20: float
    loc_window: np.ndarray
    abs_hist: dict[str, np.ndarray]
    trail_years: dict[str, float]
    near_lo_t: bool
    near_hi_t: bool
    below_t: bool
    run_t: int
    rv5_anchor_next: float
    rv20_anchor_next: float
    ewma_state: dict[str, EwmaState]


_ANCHOR_NEXT: dict[tuple[str, str, str], _AnchorNext] = {}


def _trail_bars(years: float) -> int:
    y = float(years)
    if not np.isfinite(y) or y <= 0 or TDY <= 0:
        return 0
    return int(round(y * TDY))


def _fit_min_bars(years: float) -> int:
    trail_bars = _trail_bars(years)
    base = 63
    if trail_bars <= 0:
        return max(2, base)
    return max(2, min(base, trail_bars))


def _log_ret_arr(px: np.ndarray) -> np.ndarray:
    out = np.empty(px.size, dtype=float)
    if px.size == 0:
        return out
    out[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = np.log(px[1:] / px[:-1])
    return out


def _rv_last(log_ret: np.ndarray, new_ret: float, window: int, ddof: int) -> float:
    need = int(window)
    if need < 2 or log_ret.size < need - 1:
        return float("nan")
    sl = np.empty(need, dtype=float)
    sl[:-1] = log_ret[-(need - 1) :]
    sl[-1] = float(new_ret)
    if not np.all(np.isfinite(sl)):
        return float("nan")
    return float(np.std(sl, ddof=int(ddof)) * np.sqrt(TDY))


def _abs_hist_from_base(base: pd.DataFrame, key: str) -> np.ndarray:
    px = pd.to_numeric(base["px"], errors="coerce").to_numpy(dtype=float)
    path = pd.to_numeric(base[f"{key}_path"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(path) & (path > 0) & np.isfinite(px) & (px > 0)
    if not ok.any():
        return np.empty(0, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rt = np.log(px[ok] / path[ok])
    rt = rt[np.isfinite(rt)]
    return np.abs(rt).astype(float, copy=False)


def _get_anchor_close(listing: Path, anchor_code: str, as_of: pd.Timestamp) -> pd.Series:
    day = pd.Timestamp(as_of).normalize().strftime("%Y-%m-%d")
    key = (str(listing), str(anchor_code).upper(), day)
    hit = _ANCHOR_CLOSE.get(key)
    if hit is not None:
        return hit
    html = find_regime_html(listing, anchor_code)
    if html is None:
        raise FileNotFoundError(f"anchor {anchor_code} missing")
    close_a, _ = load_ohlcv_from_regime_html(html)
    close_a = _norm_close(close_a)
    close_a = close_a.loc[close_a.index <= pd.Timestamp(as_of).normalize()]
    if close_a.empty:
        raise ValueError("anchor empty")
    _ANCHOR_CLOSE[key] = close_a
    return close_a


def _anchor_vol_cache(listing: Path, anchor_code: str, as_of: pd.Timestamp) -> _AnchorNext:
    """Worker-level cache: T+1 flat-extended anchor rv5/rv20 (appends a 0 log-return)."""
    day = pd.Timestamp(as_of).normalize().strftime("%Y-%m-%d")
    key = (str(listing), str(anchor_code).upper(), day)
    hit = _ANCHOR_NEXT.get(key)
    if hit is not None:
        return hit
    close_a = _get_anchor_close(listing, anchor_code, as_of)
    log_ret = _log_ret_arr(close_a.to_numpy(dtype=float))
    out = _AnchorNext(
        rv5=_rv_last(log_ret, 0.0, 5, 0),
        rv20=_rv_last(log_ret, 0.0, 20, 0),
    )
    _ANCHOR_NEXT[key] = out
    return out


def build_fast_ctx(
    *,
    close: pd.Series,
    ohlcv: pd.DataFrame,  # kept for call-site parity with eval_next
    base: pd.DataFrame,
    listing: Path,
    anchor_code: str,
    as_of: pd.Timestamp,
    lookback: int,
) -> FastCtx:
    c = pd.to_numeric(close, errors="coerce").astype(float)
    arr = c.to_numpy(dtype=float)
    n = int(arr.size)
    if n < 1:
        raise ValueError("empty close")
    log_ret = _log_ret_arr(arr)
    rv5_d1 = compute_realized_vol(c, window=5, ddof=1).to_numpy(dtype=float)
    last = base.iloc[-1]
    px_t = pd.to_numeric(last.get("px"), errors="coerce")
    p10_t = pd.to_numeric(last.get("fig10_path"), errors="coerce")
    below_t = bool(pd.notna(px_t) and pd.notna(p10_t) and float(px_t) < float(p10_t))
    run_t = int(_knife_run(base["px"], base["fig10_path"]).iloc[-1])
    av = _anchor_vol_cache(listing, anchor_code, as_of)
    abs_hist = {key: _abs_hist_from_base(base, key) for key, _years, _ in FIG_PANELS}
    trail_years = {key: float(years) for key, years, _ in FIG_PANELS}
    ewma_state = {
        key: _ewma_state_from_path_close(base[f"{key}_path"], base["px"])
        if f"{key}_path" in base.columns
        else _empty_ewma_state()
        for key in EWMA_KEYS
    }
    return FastCtx(
        close_arr=arr,
        px0=float(arr[-1]),
        lookback=int(lookback),
        log_ret=log_ret,
        rv5_ddof1_hist=rv5_d1,
        close_m4=arr[-4:] if n >= 4 else arr.copy(),
        close_m19=arr[-19:] if n >= 19 else arr.copy(),
        close_lag5=float(arr[-5]) if n >= 5 else float("nan"),
        close_lag20=float(arr[-20]) if n >= 20 else float("nan"),
        loc_window=arr[-int(lookback) :] if n >= int(lookback) else arr.copy(),
        abs_hist=abs_hist,
        trail_years=trail_years,
        near_lo_t=_bool(last.get("fig9_near_lo")),
        near_hi_t=_bool(last.get("fig9_near_hi")),
        below_t=below_t,
        run_t=run_t,
        rv5_anchor_next=float(av.rv5),
        rv20_anchor_next=float(av.rv20),
        ewma_state=ewma_state,
    )


def _last_path_g(close_arr: np.ndarray, c: float, years: float) -> tuple[float, float]:
    trail_bars = _trail_bars(years)
    min_b = _fit_min_bars(years)
    n = int(close_arr.size)
    t = n
    start = 0
    if trail_bars > 0 and (t + 1) >= trail_bars:
        start = t + 1 - trail_bars
    sl = np.empty(n - start + 1, dtype=float)
    sl[:-1] = close_arr[start:]
    sl[-1] = float(c)
    ok = np.isfinite(sl) & (sl > 0)
    n_ok = int(ok.sum())
    if n_ok < min_b:
        return float("nan"), float("nan")
    jj = np.flatnonzero(ok).astype(float)
    log_y = np.log(sl[ok])
    b, a = np.polyfit(jj, log_y, 1)
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan"), float("nan")
    path = float(np.exp(a + b * float(jj[-1])))
    g = float(np.exp(b * TDY) - 1.0)
    return path, g


def _last_envelope(path: float, c: float, abs_hist: np.ndarray) -> tuple[float, float]:
    hist = abs_hist
    if np.isfinite(path) and path > 0 and np.isfinite(c) and c > 0:
        rt = float(np.log(c / path))
        if np.isfinite(rt):
            hist = np.concatenate([abs_hist, np.asarray([abs(rt)], dtype=float)])
    if hist.size < ENV_MIN_BARS or not (np.isfinite(path) and path > 0):
        return float("nan"), float("nan")
    delta = float(np.quantile(hist, ENV_Q))
    if not np.isfinite(delta) or delta < 0:
        return float("nan"), float("nan")
    return path * float(np.exp(delta)), path * float(np.exp(-delta))


def _empty_ewma_state() -> EwmaState:
    return EwmaState(
        abs_z=np.empty(0, dtype=float),
        last_var=None,
        last_r2=None,
        last_upper=float("nan"),
        last_lower=float("nan"),
    )


def _ewma_state_from_path_close(
    path: pd.Series,
    close: pd.Series,
    *,
    q: float = ENV_Q,
    min_bars: int = ENV_MIN_BARS,
    lam: float = EWMA_LAM,
) -> EwmaState:
    """Walk ``ewma_scaled_envelope`` through T; keep state for a T+1 last row."""
    p = pd.to_numeric(path, errors="coerce").to_numpy(dtype=float)
    c = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    n = int(min(p.size, c.size))
    if n == 0:
        return _empty_ewma_state()
    lam_f = float(lam)
    one_l = 1.0 - lam_f
    qq = float(q)
    min_b = max(2, int(min_bars))
    last_var: float | None = None
    last_r2: float | None = None
    abs_z: list[float] = []
    last_upper = float("nan")
    last_lower = float("nan")
    for t in range(n):
        pt = float(p[t])
        ct = float(c[t])
        rt = float("nan")
        if np.isfinite(pt) and pt > 0 and np.isfinite(ct) and ct > 0:
            rt = float(np.log(ct / pt))
        st = float("nan")
        if last_var is None:
            if last_r2 is not None and last_r2 > 0:
                last_var = last_r2
                st = float(np.sqrt(last_var))
        else:
            if last_r2 is not None and np.isfinite(last_r2):
                last_var = lam_f * last_var + one_l * last_r2
            if last_var is not None and last_var > 0:
                st = float(np.sqrt(last_var))
        ok_p = np.isfinite(pt) and pt > 0
        ok_s = np.isfinite(st) and st > 0
        if ok_s and np.isfinite(rt):
            zt = rt / st
            if np.isfinite(zt):
                abs_z.append(abs(zt))
        if len(abs_z) >= min_b and ok_p and ok_s:
            delta_t = float(np.quantile(np.asarray(abs_z, dtype=float), qq))
            if np.isfinite(delta_t) and delta_t >= 0:
                width = delta_t * st
                last_upper = pt * float(np.exp(width))
                last_lower = pt * float(np.exp(-width))
        if np.isfinite(rt):
            last_r2 = rt * rt
    return EwmaState(
        abs_z=np.asarray(abs_z, dtype=float),
        last_var=last_var,
        last_r2=last_r2,
        last_upper=last_upper,
        last_lower=last_lower,
    )


def _last_ewma_envelope(
    path: float,
    c: float,
    state: EwmaState,
    *,
    q: float = ENV_Q,
    min_bars: int = ENV_MIN_BARS,
    lam: float = EWMA_LAM,
) -> tuple[float, float]:
    """One-step EWMA rails for hypothetical T+1 close. Matches last bar of full envelope."""
    st = _next_ewma_sigma(state, lam=lam)
    pt = float(path)
    rt = _path_residual(pt, c)
    zt = rt / st if np.isfinite(st) and st > 0 and np.isfinite(rt) else float("nan")
    abs_z = state.abs_z
    if np.isfinite(zt):
        if abs_z.size:
            abs_z = np.concatenate([abs_z, np.asarray([abs(zt)], dtype=float)])
        else:
            abs_z = np.asarray([abs(zt)], dtype=float)
    min_b = max(2, int(min_bars))
    if abs_z.size < min_b or not (np.isfinite(pt) and pt > 0 and np.isfinite(st) and st > 0):
        return float("nan"), float("nan")
    delta_t = float(np.quantile(abs_z, float(q)))
    if not np.isfinite(delta_t) or delta_t < 0:
        return float("nan"), float("nan")
    width = delta_t * st
    return pt * float(np.exp(width)), pt * float(np.exp(-width))


def _path_residual(path: float, c: float) -> float:
    pt = float(path)
    ct = float(c)
    if not (np.isfinite(pt) and pt > 0 and np.isfinite(ct) and ct > 0):
        return float("nan")
    rt = float(np.log(ct / pt))
    return rt if np.isfinite(rt) else float("nan")


def _next_ewma_sigma(state: EwmaState, *, lam: float = EWMA_LAM) -> float:
    lam_f = float(lam)
    one_l = 1.0 - lam_f
    last_var = state.last_var
    last_r2 = state.last_r2
    st = float("nan")
    if last_var is None:
        if last_r2 is not None and last_r2 > 0:
            st = float(np.sqrt(last_r2))
    else:
        var = float(last_var)
        if last_r2 is not None and np.isfinite(last_r2):
            var = lam_f * var + one_l * last_r2
        if var > 0:
            st = float(np.sqrt(var))
    return st


def _last_ewma_z(path: float, c: float, state: EwmaState, *, lam: float = EWMA_LAM) -> float:
    st = _next_ewma_sigma(state, lam=lam)
    rt = _path_residual(path, c)
    if not (np.isfinite(st) and st > 0 and np.isfinite(rt)):
        return float("nan")
    zt = rt / st
    return zt if np.isfinite(zt) else float("nan")


def _signed_rank_last(
    hist: np.ndarray,
    value: float,
    *,
    min_bars: int = ENV_MIN_BARS,
) -> float:
    """Causal signed empirical rank of ``|value|`` vs ``hist`` plus today (HTML hover)."""
    if not np.isfinite(value):
        return float("nan")
    abs_v = abs(float(value))
    if hist.size:
        allh = np.concatenate([hist, np.asarray([abs_v], dtype=float)])
    else:
        allh = np.asarray([abs_v], dtype=float)
    if allh.size < max(2, int(min_bars)):
        return float("nan")
    q_t = float(np.mean(allh <= abs_v))
    if not np.isfinite(q_t):
        return float("nan")
    if value > 0:
        return q_t
    if value < 0:
        return -q_t
    return 0.0


def _fmt_signed_pct(x: Any) -> str:
    v = _f(x)
    if v is None:
        return "—"
    if v == 0.0:
        return "0.0"
    sign = "+" if v > 0 else "-"
    return f"{sign}{abs(v) * 100.0:.1f}"


def _half_width(upper: float, path: float) -> float:
    if not (np.isfinite(upper) and np.isfinite(path) and path > 0):
        return float("nan")
    return float(upper / path - 1.0)


def _ewma_rail_guesses(ctx: FastCtx) -> list[float | None]:
    out: list[float | None] = []
    for key in EWMA_KEYS:
        st = ctx.ewma_state.get(key)
        if st is None:
            continue
        out.append(_f(st.last_upper))
        out.append(_f(st.last_lower))
    return out


def _band_pos_last(px: float, lower: float, upper: float) -> float:
    if not (np.isfinite(px) and px > 0 and np.isfinite(lower) and lower > 0
            and np.isfinite(upper) and upper > 0):
        return float("nan")
    lo = float(np.log(lower))
    hi = float(np.log(upper))
    den = hi - lo
    if not np.isfinite(den) or den == 0:
        return float("nan")
    return (float(np.log(px)) - lo) / den


def _fair_need_ann(rv: float, rv_anchor: float) -> float:
    prem = float(ANCHOR_ERP_ANN)
    if not (
        np.isfinite(prem)
        and np.isfinite(rv)
        and np.isfinite(rv_anchor)
        and rv_anchor > 0.0
        and rv >= 0.0
    ):
        return float("nan")
    return prem * rv / rv_anchor


def _period_need(r_ann: float, bars: int) -> float:
    if not np.isfinite(r_ann) or r_ann <= -1.0 or int(bars) <= 0:
        return float("nan")
    return float((1.0 + r_ann) ** (float(bars) / TDY) - 1.0)


def _classify_last(
    *,
    px: float,
    p10: float,
    g8: float,
    g9: float,
    g10: float,
    touch_hi: bool,
    knife_run: int,
) -> str:
    warmup = not (np.isfinite(px) and np.isfinite(p10))
    dual_up = np.isfinite(g8) and np.isfinite(g9) and g8 > 0 and g9 > 0
    dual_down = np.isfinite(g10) and np.isfinite(g9) and g10 <= 0 and g9 <= 0
    label = STATE_CHOP
    if np.isfinite(px) and np.isfinite(p10) and px >= p10:
        label = STATE_REPAIR
    if dual_up:
        label = STATE_UP
    if dual_down:
        label = STATE_DOWN
    if int(knife_run) >= _KNIFE_BARS:
        label = STATE_KNIFE
    if touch_hi:
        label = STATE_OVERHEAT
    if warmup:
        label = STATE_WARMUP
    return str(label)


def eval_next_fast(ctx: FastCtx, px: float) -> dict[str, Any]:
    """Last-row-only eval for a hypothetical T+1 close. Keys match ``eval_next``."""
    c = float(px)
    figs: dict[str, dict[str, float]] = {}
    for key, years, _ in FIG_PANELS:
        path, g = _last_path_g(ctx.close_arr, c, years)
        upper, lower = _last_envelope(path, c, ctx.abs_hist[key])
        figs[key] = {"path": path, "g": g, "upper": upper, "lower": lower}

    f9 = figs["fig9"]
    f10 = figs["fig10"]
    f8 = figs["fig8"]
    f7 = figs["fig7"]
    fig9_touch_lo = bool(np.isfinite(f9["lower"]) and c <= f9["lower"])
    fig9_touch_hi = bool(np.isfinite(f9["upper"]) and c >= f9["upper"])
    fig10_touch_lo = bool(np.isfinite(f10["lower"]) and c <= f10["lower"])
    fig10_touch_hi = bool(np.isfinite(f10["upper"]) and c >= f10["upper"])
    ewma_rails: dict[str, tuple[float, float]] = {}
    for key in EWMA_KEYS:
        st = ctx.ewma_state.get(key)
        if st is None:
            ewma_rails[key] = (float("nan"), float("nan"))
        else:
            ewma_rails[key] = _last_ewma_envelope(figs[key]["path"], c, st)
    e9u, e9l = ewma_rails["fig9"]
    e10u, e10l = ewma_rails["fig10"]
    fig9_ewma_touch_lo = bool(np.isfinite(e9l) and c <= e9l)
    fig9_ewma_touch_hi = bool(np.isfinite(e9u) and c >= e9u)
    fig10_ewma_touch_lo = bool(np.isfinite(e10l) and c <= e10l)
    fig10_ewma_touch_hi = bool(np.isfinite(e10u) and c >= e10u)
    fig9_p95_width = _half_width(f9["upper"], f9["path"])
    fig10_p95_width = _half_width(f10["upper"], f10["path"])
    fig9_ewma_width = _half_width(e9u, f9["path"])
    fig10_ewma_width = _half_width(e10u, f10["path"])
    r9 = _path_residual(f9["path"], c)
    r10 = _path_residual(f10["path"], c)
    z9 = (
        _last_ewma_z(f9["path"], c, ctx.ewma_state["fig9"])
        if "fig9" in ctx.ewma_state
        else float("nan")
    )
    z10 = (
        _last_ewma_z(f10["path"], c, ctx.ewma_state["fig10"])
        if "fig10" in ctx.ewma_state
        else float("nan")
    )
    fig9_p95_pct = _signed_rank_last(ctx.abs_hist.get("fig9", np.empty(0)), r9)
    fig10_p95_pct = _signed_rank_last(ctx.abs_hist.get("fig10", np.empty(0)), r10)
    fig9_ewma_pct = _signed_rank_last(
        ctx.ewma_state["fig9"].abs_z if "fig9" in ctx.ewma_state else np.empty(0),
        z9,
    )
    fig10_ewma_pct = _signed_rank_last(
        ctx.ewma_state["fig10"].abs_z if "fig10" in ctx.ewma_state else np.empty(0),
        z10,
    )
    near_lo = bool(np.isfinite(f9["lower"]) and c <= f9["lower"] * (1.0 + NEAR_PCT))
    near_hi = bool(np.isfinite(f9["upper"]) and c >= f9["upper"] * (1.0 - NEAR_PCT))
    fig9_pos = _band_pos_last(c, f9["lower"], f9["upper"])
    fig10_pos = _band_pos_last(c, f10["lower"], f10["upper"])
    fig7_gap = (
        c / f7["path"] - 1.0
        if np.isfinite(f7["path"]) and f7["path"] > 0 and np.isfinite(c)
        else float("nan")
    )

    n_hist = int(ctx.close_arr.size)
    ma5 = (
        float((ctx.close_m4.sum() + c) / 5.0)
        if ctx.close_m4.size == 4 and np.all(np.isfinite(ctx.close_m4)) and np.isfinite(c)
        else float("nan")
    )
    ma20 = (
        float((ctx.close_m19.sum() + c) / 20.0)
        if ctx.close_m19.size == 19 and np.all(np.isfinite(ctx.close_m19)) and np.isfinite(c)
        else float("nan")
    )

    new_ret = (
        float(np.log(c / ctx.px0))
        if np.isfinite(c) and c > 0 and np.isfinite(ctx.px0) and ctx.px0 > 0
        else float("nan")
    )
    rv5 = _rv_last(ctx.log_ret, new_ret, 5, 0)
    rv20 = _rv_last(ctx.log_ret, new_ret, 20, 0)
    rv5_d1 = _rv_last(ctx.log_ret, new_ret, 5, 1)
    vol5_pct = _vol_pct_last(ctx.rv5_ddof1_hist, rv5_d1)

    r5 = c / ctx.close_lag5 - 1.0 if np.isfinite(c) and np.isfinite(ctx.close_lag5) and ctx.close_lag5 != 0 else float("nan")
    r20 = (
        c / ctx.close_lag20 - 1.0
        if np.isfinite(c) and np.isfinite(ctx.close_lag20) and ctx.close_lag20 != 0
        else float("nan")
    )
    need5 = _period_need(_fair_need_ann(rv5, ctx.rv5_anchor_next), 5)
    need20 = _period_need(_fair_need_ann(rv20, ctx.rv20_anchor_next), 20)
    gap_5d = r5 - need5 if np.isfinite(r5) and np.isfinite(need5) else float("nan")
    gap_20d = r20 - need20 if np.isfinite(r20) and np.isfinite(need20) else float("nan")

    locmin = False
    locmax = False
    lb = int(ctx.lookback)
    if n_hist >= lb and ctx.loc_window.size == lb and np.isfinite(c):
        w = np.empty(lb + 1, dtype=float)
        w[:-1] = ctx.loc_window
        w[-1] = c
        if np.isfinite(c) and c == np.nanmin(w):
            locmin = True
        if np.isfinite(c) and c == np.nanmax(w):
            locmax = True

    aux_buy = (
        np.isfinite(ma20)
        and np.isfinite(ma5)
        and c < ma20
        and ma5 < ma20
        and np.isfinite(gap_5d)
        and gap_5d <= 0
        and np.isfinite(gap_20d)
        and gap_20d <= 0
        and (not fig10_touch_hi)
    )
    aux_sell = (
        np.isfinite(ma20)
        and np.isfinite(ma5)
        and c > ma20
        and ma5 > ma20
        and np.isfinite(gap_5d)
        and gap_5d >= 0
        and np.isfinite(gap_20d)
        and gap_20d >= 0
    )
    s1 = fig9_touch_hi or fig10_touch_hi
    mid_stretch = (
        aux_sell
        and locmax
        and np.isfinite(vol5_pct)
        and vol5_pct >= VOL_CLUSTER
        and np.isfinite(f9["g"])
        and f9["g"] > 0
        and (not s1)
        and np.isfinite(fig9_pos)
        and fig9_pos >= POS_MIN
        and (
            (np.isfinite(fig7_gap) and fig7_gap >= FIG7_PREM)
            or (np.isfinite(fig10_pos) and fig10_pos >= FIG10_POS_MIN)
        )
    )
    edge_lo = near_lo and (not ctx.near_lo_t)
    edge_hi = near_hi and (not ctx.near_hi_t)
    buy = bool(edge_lo and aux_buy)
    sell = bool((edge_hi and aux_sell) or mid_stretch)

    below = bool(np.isfinite(c) and np.isfinite(f10["path"]) and c < f10["path"])
    if below:
        knife_run = (int(ctx.run_t) + 1) if ctx.below_t else 1
    else:
        knife_run = 0
    state = _classify_last(
        px=c,
        p10=f10["path"],
        g8=f8["g"],
        g9=f9["g"],
        g10=f10["g"],
        touch_hi=s1,
        knife_run=knife_run,
    )
    px_ge_fig10 = bool(np.isfinite(c) and np.isfinite(f10["path"]) and c >= f10["path"])
    return {
        "px": float(c),
        "state": str(state),
        "buy": buy,
        "sell": sell,
        "near_lo": near_lo,
        "near_hi": near_hi,
        "aux_buy": bool(aux_buy),
        "aux_sell": bool(aux_sell),
        "mid_stretch": bool(mid_stretch),
        "touch_hi": bool(s1),
        "fig9_touch_lo": fig9_touch_lo,
        "fig9_touch_hi": fig9_touch_hi,
        "fig10_touch_lo": fig10_touch_lo,
        "fig10_touch_hi": fig10_touch_hi,
        "fig9_ewma_touch_lo": fig9_ewma_touch_lo,
        "fig9_ewma_touch_hi": fig9_ewma_touch_hi,
        "fig10_ewma_touch_lo": fig10_ewma_touch_lo,
        "fig10_ewma_touch_hi": fig10_ewma_touch_hi,
        "fig9_pos": _f(fig9_pos),
        "fig9_lower": _f(f9["lower"]),
        "fig9_upper": _f(f9["upper"]),
        "fig10_path": _f(f10["path"]),
        "fig10_lower": _f(f10["lower"]),
        "fig10_upper": _f(f10["upper"]),
        "fig9_ewma_lower": _f(e9l),
        "fig9_ewma_upper": _f(e9u),
        "fig10_ewma_lower": _f(e10l),
        "fig10_ewma_upper": _f(e10u),
        "fig9_p95_width": _f(fig9_p95_width),
        "fig9_ewma_width": _f(fig9_ewma_width),
        "fig10_p95_width": _f(fig10_p95_width),
        "fig10_ewma_width": _f(fig10_ewma_width),
        "fig9_p95_pct": _f(fig9_p95_pct),
        "fig9_ewma_pct": _f(fig9_ewma_pct),
        "fig10_p95_pct": _f(fig10_p95_pct),
        "fig10_ewma_pct": _f(fig10_ewma_pct),
        "knife_run": int(knife_run),
        "px_ge_fig10": px_ge_fig10,
    }


def _vol_pct_last(
    hist_rv: np.ndarray,
    new_rv: float,
    *,
    lookback: int = DEFAULT_VOL_PCT_LOOKBACK,
    min_history: int = DEFAULT_VOL_PCT_MIN_HISTORY,
) -> float:
    if not np.isfinite(new_rv):
        return float("nan")
    n = int(hist_rv.size)
    start = max(0, n + 1 - int(lookback))
    window = np.empty(n - start + 1, dtype=float)
    window[:-1] = hist_rv[start:]
    window[-1] = float(new_rv)
    finite = window[np.isfinite(window)]
    if finite.size < int(min_history):
        return float("nan")
    return float(np.sum(finite <= float(new_rv))) / float(finite.size)


def process_one(payload: dict[str, Any]) -> dict[str, Any]:
    html = Path(payload["html"])
    code = str(payload["code"])
    name = str(payload["name"])
    listing = Path(payload["listing"])
    as_of = pd.Timestamp(payload["as_of"]).normalize()
    next_day = pd.Timestamp(payload["next_day"]).normalize()
    lookback = int(payload["lookback"])
    anchor_code = str(payload["anchor_code"])
    try:
        close, ohlcv = load_ohlcv_from_regime_html(html)
        close = _norm_close(close)
        ohlcv = _norm_ohlcv(ohlcv)
        close = close.loc[close.index <= as_of]
        ohlcv = ohlcv.loc[ohlcv.index <= as_of]
        if close.empty:
            raise ValueError(f"no bars on/before {as_of.date()}")
        if close.index[-1] != as_of:
            # last available bar is still T if the file ends on as-of
            as_of_used = pd.Timestamp(close.index[-1]).normalize()
        else:
            as_of_used = as_of
        px0 = float(close.iloc[-1])
        if not np.isfinite(px0) or px0 <= 0:
            raise ValueError("invalid last close")

        close_a = _get_anchor_close(listing, anchor_code, as_of_used)
        close_a = close_a.loc[close_a.index <= as_of_used]
        if close_a.empty:
            raise ValueError("anchor empty")

        base = sm.build_feature_frame_from_series(
            close, ohlcv, close_a, lookback=lookback
        )
        base = bt.prepare_fig9_touch_frame(base)
        base = base.loc[base.index <= as_of_used]
        if base.empty:
            raise ValueError("empty feature frame")
        buy_hist, sell_hist, _ = bt.variant_masks(base, "aux_edge")
        trades, _, _ = sm.run_state_machine(base, buy_hist, sell_hist)
        open_tr = _open_trade(trades)
        in_pos = open_tr is not None
        last = base.iloc[-1]
        states0 = classify_six_states(base)
        state0 = str(states0.iloc[-1])
        run0 = int(_knife_run(base["px"], base["fig10_path"]).iloc[-1])
        near_lo0 = _bool(last.get("fig9_near_lo"))
        near_hi0 = _bool(last.get("fig9_near_hi"))
        aux_b0 = _bool(last.get("aux_buy"))
        aux_s0 = _bool(last.get("aux_sell"))
        pos0 = _f(last.get("fig9_pos"))
        lo0 = _f(last.get("fig9_lower"))
        hi0 = _f(last.get("fig9_upper"))
        p10_0 = _f(last.get("fig10_path"))
        fig10_lo0 = _f(last.get("fig10_lower"))
        fig9_hi0 = _f(last.get("fig9_upper"))
        fig10_hi0 = _f(last.get("fig10_upper"))
        overheat_guess = None
        for u in (fig9_hi0, fig10_hi0):
            if u is not None:
                overheat_guess = u if overheat_guess is None else min(overheat_guess, u)
        lo_guess = None if lo0 is None else lo0 * (1.0 + NEAR_PCT)
        hi_guess = None if hi0 is None else hi0 * (1.0 - NEAR_PCT)

        ctx = build_fast_ctx(
            close=close,
            ohlcv=ohlcv,
            base=base,
            listing=listing,
            anchor_code=anchor_code,
            as_of=as_of_used,
            lookback=lookback,
        )

        ev = _cached_eval(lambda px: eval_next_fast(ctx, px))
        span_lo, span_hi = _search_span(
            px0,
            [
                lo_guess,
                hi_guess,
                p10_0,
                overheat_guess,
                lo0,
                fig10_lo0,
                fig9_hi0,
                fig10_hi0,
                *_ewma_rail_guesses(ctx),
            ],
        )

        # Seed the cache with a coarse grid so bisect starts near flips.
        grid_px = sorted(
            {max(span_lo, min(span_hi, px0 * (1.0 + r))) for r in GRID_RETS}
        )
        for gpx in grid_px:
            ev(gpx)

        buy_px = None
        watch_lo_px = None
        if not near_lo0:
            watch_lo_px = _highest_true(lambda x: ev(x)["near_lo"], span_lo, px0)
            if not in_pos:
                buy_px = _highest_true(lambda x: ev(x)["buy"], span_lo, px0)
        sell_px = None
        if in_pos:
            sell_px = _lowest_true(lambda x: ev(x)["sell"], px0, span_hi)
        exit_knife_px = None
        enter_knife_px = None
        if state0 == STATE_KNIFE:
            exit_knife_px = _lowest_true(
                lambda x: ev(x)["state"] != STATE_KNIFE, px0 * 0.995, span_hi
            )
        elif run0 == 9:
            enter_knife_px = _highest_true(
                lambda x: ev(x)["state"] == STATE_KNIFE, span_lo, px0
            )
        overheat_px = None
        if state0 != STATE_OVERHEAT:
            overheat_px = _lowest_true(
                lambda x: ev(x)["state"] == STATE_OVERHEAT or ev(x)["touch_hi"],
                px0,
                span_hi,
            )

        fig9_lo_px = _highest_true(
            lambda x: ev(x)["fig9_touch_lo"], span_lo, span_hi
        )
        fig9_hi_px = _lowest_true(
            lambda x: ev(x)["fig9_touch_hi"], span_lo, span_hi
        )
        fig10_lo_px = _highest_true(
            lambda x: ev(x)["fig10_touch_lo"], span_lo, span_hi
        )
        fig10_hi_px = _lowest_true(
            lambda x: ev(x)["fig10_touch_hi"], span_lo, span_hi
        )
        fig9_ewma_lo_px = _highest_true(
            lambda x: ev(x)["fig9_ewma_touch_lo"], span_lo, span_hi
        )
        fig9_ewma_hi_px = _lowest_true(
            lambda x: ev(x)["fig9_ewma_touch_hi"], span_lo, span_hi
        )
        fig10_ewma_lo_px = _highest_true(
            lambda x: ev(x)["fig10_ewma_touch_lo"], span_lo, span_hi
        )
        fig10_ewma_hi_px = _lowest_true(
            lambda x: ev(x)["fig10_ewma_touch_hi"], span_lo, span_hi
        )

        at0 = ev(px0)
        fig9_lo_now = bool(at0["fig9_touch_lo"])
        fig9_hi_now = bool(at0["fig9_touch_hi"])
        fig10_lo_now = bool(at0["fig10_touch_lo"])
        fig10_hi_now = bool(at0["fig10_touch_hi"])
        fig9_ewma_lo_now = bool(at0["fig9_ewma_touch_lo"])
        fig9_ewma_hi_now = bool(at0["fig9_ewma_touch_hi"])
        fig10_ewma_lo_now = bool(at0["fig10_ewma_touch_lo"])
        fig10_ewma_hi_now = bool(at0["fig10_ewma_touch_hi"])
        notes: list[str] = []
        if near_lo0:
            notes.append("已贴下轨，缺上跳沿")
            if not aux_b0:
                notes.append("缺aux买")
        elif watch_lo_px is not None and buy_px is None and not in_pos:
            notes.append("可贴下轨但 aux_edge 买未齐")
        if near_hi0:
            notes.append("已贴上轨，缺上跳沿")
            if not aux_s0:
                notes.append("缺aux卖")
        if in_pos:
            notes.append("图9持仓")
        else:
            notes.append("图9空仓")
        if state0 == STATE_KNIFE:
            notes.append(f"②已{run0}日破路径")
        elif run0 == 9:
            notes.append("再破路径一日进②")

        triggers: list[tuple[str, float]] = []
        for label, val in (
            ("buy", buy_px),
            ("sell", sell_px),
            ("exit_knife", exit_knife_px),
            ("enter_knife", enter_knife_px),
            ("overheat", overheat_px),
            ("watch_lo", watch_lo_px),
        ):
            if val is not None:
                triggers.append((label, abs(val / px0 - 1.0)))
        nearest_kind = ""
        nearest_abs = None
        if triggers:
            nearest_kind, nearest_abs = min(triggers, key=lambda t: t[1])

        last_buy = ""
        unreal = None
        if open_tr is not None:
            last_buy = str(open_tr.get("signal_buy") or "")
            unreal = _f(open_tr.get("ret"))

        def _state_at(ret: float) -> str:
            return str(ev(px0 * (1.0 + ret))["state"])

        return {
            "ok": True,
            "code": code,
            "name": name,
            "as_of": as_of_used.strftime("%Y-%m-%d"),
            "next_day": next_day.strftime("%Y-%m-%d"),
            "n": int(len(base)),
            "px": round(px0, 4),
            "state": state0,
            "in_pos": in_pos,
            "fig9_pos": None if pos0 is None else round(pos0, 4),
            "near_lo": near_lo0,
            "near_hi": near_hi0,
            "aux_buy": aux_b0,
            "aux_sell": aux_s0,
            "knife_run": run0,
            "buy_px": None if buy_px is None else round(buy_px, 4),
            "buy_ret": None if buy_px is None else round(buy_px / px0 - 1.0, 4),
            "watch_lo_px": None if watch_lo_px is None else round(watch_lo_px, 4),
            "watch_lo_ret": None
            if watch_lo_px is None
            else round(watch_lo_px / px0 - 1.0, 4),
            "sell_px": None if sell_px is None else round(sell_px, 4),
            "sell_ret": None if sell_px is None else round(sell_px / px0 - 1.0, 4),
            "exit_knife_px": None if exit_knife_px is None else round(exit_knife_px, 4),
            "exit_knife_ret": None
            if exit_knife_px is None
            else round(exit_knife_px / px0 - 1.0, 4),
            "enter_knife_px": None if enter_knife_px is None else round(enter_knife_px, 4),
            "enter_knife_ret": None
            if enter_knife_px is None
            else round(enter_knife_px / px0 - 1.0, 4),
            "overheat_px": None if overheat_px is None else round(overheat_px, 4),
            "overheat_ret": None
            if overheat_px is None
            else round(overheat_px / px0 - 1.0, 4),
            "fig9_lo_px": None if fig9_lo_px is None else round(fig9_lo_px, 4),
            "fig9_lo_ret": None
            if fig9_lo_px is None
            else round(fig9_lo_px / px0 - 1.0, 4),
            "fig9_lo_now": fig9_lo_now,
            "fig9_hi_px": None if fig9_hi_px is None else round(fig9_hi_px, 4),
            "fig9_hi_ret": None
            if fig9_hi_px is None
            else round(fig9_hi_px / px0 - 1.0, 4),
            "fig9_hi_now": fig9_hi_now,
            "fig10_lo_px": None if fig10_lo_px is None else round(fig10_lo_px, 4),
            "fig10_lo_ret": None
            if fig10_lo_px is None
            else round(fig10_lo_px / px0 - 1.0, 4),
            "fig10_lo_now": fig10_lo_now,
            "fig10_hi_px": None if fig10_hi_px is None else round(fig10_hi_px, 4),
            "fig10_hi_ret": None
            if fig10_hi_px is None
            else round(fig10_hi_px / px0 - 1.0, 4),
            "fig10_hi_now": fig10_hi_now,
            "fig9_ewma_lo_px": None if fig9_ewma_lo_px is None else round(fig9_ewma_lo_px, 4),
            "fig9_ewma_lo_ret": None
            if fig9_ewma_lo_px is None
            else round(fig9_ewma_lo_px / px0 - 1.0, 4),
            "fig9_ewma_lo_now": fig9_ewma_lo_now,
            "fig9_ewma_hi_px": None if fig9_ewma_hi_px is None else round(fig9_ewma_hi_px, 4),
            "fig9_ewma_hi_ret": None
            if fig9_ewma_hi_px is None
            else round(fig9_ewma_hi_px / px0 - 1.0, 4),
            "fig9_ewma_hi_now": fig9_ewma_hi_now,
            "fig10_ewma_lo_px": None if fig10_ewma_lo_px is None else round(fig10_ewma_lo_px, 4),
            "fig10_ewma_lo_ret": None
            if fig10_ewma_lo_px is None
            else round(fig10_ewma_lo_px / px0 - 1.0, 4),
            "fig10_ewma_lo_now": fig10_ewma_lo_now,
            "fig10_ewma_hi_px": None if fig10_ewma_hi_px is None else round(fig10_ewma_hi_px, 4),
            "fig10_ewma_hi_ret": None
            if fig10_ewma_hi_px is None
            else round(fig10_ewma_hi_px / px0 - 1.0, 4),
            "fig10_ewma_hi_now": fig10_ewma_hi_now,
            "fig9_p95_width": None
            if _f(at0.get("fig9_p95_width")) is None
            else round(float(at0["fig9_p95_width"]), 4),
            "fig9_ewma_width": None
            if _f(at0.get("fig9_ewma_width")) is None
            else round(float(at0["fig9_ewma_width"]), 4),
            "fig10_p95_width": None
            if _f(at0.get("fig10_p95_width")) is None
            else round(float(at0["fig10_p95_width"]), 4),
            "fig10_ewma_width": None
            if _f(at0.get("fig10_ewma_width")) is None
            else round(float(at0["fig10_ewma_width"]), 4),
            "fig9_p95_pct": None
            if _f(at0.get("fig9_p95_pct")) is None
            else round(float(at0["fig9_p95_pct"]), 4),
            "fig9_ewma_pct": None
            if _f(at0.get("fig9_ewma_pct")) is None
            else round(float(at0["fig9_ewma_pct"]), 4),
            "fig10_p95_pct": None
            if _f(at0.get("fig10_p95_pct")) is None
            else round(float(at0["fig10_p95_pct"]), 4),
            "fig10_ewma_pct": None
            if _f(at0.get("fig10_ewma_pct")) is None
            else round(float(at0["fig10_ewma_pct"]), 4),
            "state_m3": _state_at(-0.03),
            "state_flat": at0["state"],
            "state_p3": _state_at(0.03),
            "unrealized": None if unreal is None else round(unreal, 4),
            "last_buy_signal": last_buy,
            "nearest_kind": nearest_kind,
            "nearest_abs": None if nearest_abs is None else round(nearest_abs, 4),
            "note": "；".join(notes),
            "error": "",
            "html": html.name,
            "n_eval": len(ev.cache),  # type: ignore[attr-defined]
        }
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "fig7 candlestick not found" in msg:
            msg = "HTML 无 fig7 K 线（国债类，图9规则不适用）"
        else:
            msg = f"{type(exc).__name__}: {exc}"
        return {
            "ok": False,
            "code": code,
            "name": name,
            "action": "ERROR",
            "error": msg,
            "trace": traceback.format_exc(),
            "html": html.name,
        }


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    return process_one(payload)


def _sort_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    key = pd.to_numeric(out.get("nearest_abs"), errors="coerce")
    out = out.assign(_k=key)
    return out.sort_values(
        by=["_k", "code"],
        ascending=[True, True],
        na_position="last",
        kind="mergesort",
    ).drop(columns=["_k"])


def _sort_by_fig9_lo(df: pd.DataFrame) -> pd.DataFrame:
    """Most negative fig9-lower return first (largest drop to touch); already-touch last among those."""
    if df.empty:
        return df
    out = df.copy()
    key = pd.to_numeric(out.get("fig9_lo_ret"), errors="coerce")
    out = out.assign(_k=key)
    return out.sort_values(
        by=["_k", "code"],
        ascending=[True, True],
        na_position="last",
        kind="mergesort",
    ).drop(columns=["_k"])


def _repro_cmd(listing: Path, as_of: str, next_day: str, *, jobs: int = 8) -> str:
    loc = str(listing)
    return (
        f"PYTHONPATH=. python3 \\\n"
        "  decision_pack/scripts/scan_next_day_trigger_prices.py \\\n"
        f"  --listing-dir {loc} \\\n"
        f"  --as-of {as_of} --next-day {next_day} --jobs {jobs}"
    )


def _rail_cell(r: Any, px0: float | None, lohi: str) -> str:
    return _fmt_rail(
        _f(r.get(f"{lohi}_px")),
        px0,
        already=_truthy_in_pos(r.get(f"{lohi}_now")),
    )


HARD_TOUCH_COLS = (
    "代码",
    "名称",
    "T收盘",
    "图9下轨",
    "图9上轨",
    "图9 P95宽",
    "图9 因果分位P",
    "图9EWMA下",
    "图9EWMA上",
    "图9 EWMA宽",
    "图9 EWMA分位P",
    "图10下轨",
    "图10上轨",
    "图10 P95宽",
    "图10 因果分位P",
    "图10EWMA下",
    "图10EWMA上",
    "图10 EWMA宽",
    "图10 EWMA分位P",
)


def _ok_err_frames(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(rows)
    if "ok" in df.columns:
        ok_mask = df["ok"].map(_truthy_in_pos)
        return df.loc[ok_mask], df.loc[~ok_mask]
    return df, df.iloc[0:0]


def _hard_touch_record(r: Any) -> dict[str, str]:
    px0 = _f(r.get("px"))
    return {
        "代码": str(r["code"]),
        "名称": _md_cell(r.get("name")),
        "T收盘": "" if px0 is None else f"{px0:.4f}",
        "图9下轨": _rail_cell(r, px0, "fig9_lo"),
        "图9上轨": _rail_cell(r, px0, "fig9_hi"),
        "图9 P95宽": _pct(r.get("fig9_p95_width")),
        "图9 因果分位P": _fmt_signed_pct(r.get("fig9_p95_pct")),
        "图9EWMA下": _rail_cell(r, px0, "fig9_ewma_lo"),
        "图9EWMA上": _rail_cell(r, px0, "fig9_ewma_hi"),
        "图9 EWMA宽": _pct(r.get("fig9_ewma_width")),
        "图9 EWMA分位P": _fmt_signed_pct(r.get("fig9_ewma_pct")),
        "图10下轨": _rail_cell(r, px0, "fig10_lo"),
        "图10上轨": _rail_cell(r, px0, "fig10_hi"),
        "图10 P95宽": _pct(r.get("fig10_p95_width")),
        "图10 因果分位P": _fmt_signed_pct(r.get("fig10_p95_pct")),
        "图10EWMA下": _rail_cell(r, px0, "fig10_ewma_lo"),
        "图10EWMA上": _rail_cell(r, px0, "fig10_ewma_hi"),
        "图10 EWMA宽": _pct(r.get("fig10_ewma_width")),
        "图10 EWMA分位P": _fmt_signed_pct(r.get("fig10_ewma_pct")),
    }


def hard_touch_table(ok: pd.DataFrame) -> pd.DataFrame:
    recs = [_hard_touch_record(r) for _, r in _sort_by_fig9_lo(ok).iterrows()]
    return pd.DataFrame(recs, columns=list(HARD_TOUCH_COLS))


def write_hard_touch_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ok, _err = _ok_err_frames(rows)
    hard_touch_table(ok).to_csv(path, index=False, encoding="utf-8-sig")


def write_markdown(
    path: Path,
    *,
    as_of: str,
    next_day: str,
    rows: list[dict[str, Any]],
    skipped_bonds: list[str],
    listing_dir: Path | None = None,
    jobs: int = 8,
) -> None:
    df = pd.DataFrame(rows)
    if "ok" in df.columns:
        ok_mask = df["ok"].map(_truthy_in_pos)
        ok = df.loc[ok_mask]
        err = df.loc[~ok_mask]
    else:
        ok = df
        err = df.iloc[0:0]
    n_ok = int(len(ok))
    n_err = int(len(err))
    listing_s = str(listing_dir) if listing_dir is not None else ""
    lines = [
        f"# 次日触发价扫描（T={as_of} → T+1={next_day}）",
        "",
    ]
    if listing_s:
        lines += [
            "复现：",
            "",
            "```",
            _repro_cmd(Path(listing_s), as_of, next_day, jobs=jobs),
            "```",
            "",
            "换日：改 `--listing-dir`、`--as-of`（T 收盘日）、`--next-day`（下一交易日）。"
            "需已有该日 from_listing adaptive HTML（与 `scan_fig9_aux_edge_eod.py` 同一目录）。",
            "",
        ]
    lines += [
        "以 T 日收盘为已知，求解 **T+1 收盘涨/跌到什么价** 会触发：",
        "",
        "- 图9 `aux_edge` 买：`rising_edge(fig9_near_lo) ∧ aux_buy`（仅空仓）",
        "- 图9 `aux_edge` 卖：`rising_edge(fig9_near_hi) ∧ aux_sell ∨ mid_stretch`（仅持仓）",
        "- 六状态：退出②（收盘重新站上 fig10 路径）、进入②（已连续 9 日破路径再破一日）、①过热（触 fig9/fig10 上轨）",
        "- 硬触轨：T+1 收盘 `≤下轨` / `≥上轨`（图9、图10 分开；**已触** = 明天平盘仍在该轨上或外侧）",
        "",
        "方法：把假设收盘价作为 T+1 合成 K 线（开高低收相同）喂回因果管线整帧重算，再二分找触发价。",
        f"锚 **SH510300** 在 T+1 **按 T 收盘持平延长一天**（否则 `gap_5d` 的 need 在新日期为 NaN，aux 条件全灭）。",
        "**信号日收盘确认，下一交易日收盘成交**。研究扫描，不是生产 `hold_up`。",
        "",
        f"扫描成功 {n_ok} 只"
        + (f"，失败 {n_err} 只" if n_err else "")
        + "。全表按「距最近触发」从小到大。",
        "",
        "## 明日重点（阈值距收盘 ±3% 以内）",
        "",
    ]
    focus = ok.copy()
    if not focus.empty and "nearest_abs" in focus.columns:
        na = pd.to_numeric(focus["nearest_abs"], errors="coerce")
        focus = focus.loc[na.le(FOCUS_ABS)]
    else:
        focus = focus.iloc[0:0]
    if focus.empty:
        lines += ["（没有阈值落在 ±3% 内的标的）", ""]
    else:
        lines += [
            "| 代码 | 名称 | T收盘 | 状态 | 持仓 | 最近触发 | 买点 | 卖点 | ②结束 | 进② | ①触上 | 图9下轨 | 图9上轨 | 图10下轨 | 图10上轨 | T+1状态(-3%/平/+3%) | 备注 |",
            "|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for _, r in _sort_rows(focus).iterrows():
            px0 = _f(r.get("px"))
            kind = _md_cell(r.get("nearest_kind"))
            kind_map = {
                "buy": "买点",
                "sell": "卖点",
                "exit_knife": "②结束",
                "enter_knife": "进②",
                "overheat": "①过热",
                "watch_lo": "贴下轨",
            }
            kind_px = {
                "buy": r.get("buy_px"),
                "sell": r.get("sell_px"),
                "exit_knife": r.get("exit_knife_px"),
                "enter_knife": r.get("enter_knife_px"),
                "overheat": r.get("overheat_px"),
                "watch_lo": r.get("watch_lo_px"),
            }.get(kind)
            kind_fmt = (
                _fmt_ret if kind in {"enter_knife", "overheat"} else _fmt_px_ret
            )
            lines.append(
                f"| {r['code']} | {_md_cell(r.get('name'))} | "
                f"{'—' if px0 is None else f'{px0:.4f}'} | "
                f"{_md_cell(r.get('state'))} | {'是' if _truthy_in_pos(r.get('in_pos')) else ''} | "
                f"{kind_map.get(kind, kind)} {kind_fmt(_f(kind_px), px0)} | "
                f"{_fmt_px_ret(_f(r.get('buy_px')), px0)} | "
                f"{_fmt_px_ret(_f(r.get('sell_px')), px0)} | "
                f"{_fmt_px_ret(_f(r.get('exit_knife_px')), px0)} | "
                f"{_fmt_ret(_f(r.get('enter_knife_px')), px0)} | "
                f"{_fmt_ret(_f(r.get('overheat_px')), px0)} | "
                f"{_rail_cell(r, px0, 'fig9_lo')} | "
                f"{_rail_cell(r, px0, 'fig9_hi')} | "
                f"{_rail_cell(r, px0, 'fig10_lo')} | "
                f"{_rail_cell(r, px0, 'fig10_hi')} | "
                f"{_state_band(r)} | "
                f"{_md_cell(r.get('note'))} |"
            )
        lines.append("")

    lines += [
        "## 图9/图10 硬触轨（T+1 收盘）",
        "",
        "触及 = `close ≤ lower` 或 `close ≥ upper`（不是 1% near）。"
        "**已触** = 明天平盘仍在该轨上或外侧；格子只写相对 T 收盘的涨跌幅。"
        "本表按 **图9下轨涨跌幅升序**（跌最多才触下轨的在前，已在下轨外的排后）。",
        "P95 与 EWMA 分开扫 T+1 硬触。"
        "带宽 = 平盘半宽 `upper/path−1`（对应 HTML：因果 P95 / EWMA 当日）。"
        "分位 = 相对路径符号分位（路径上 `+72`、下 `−45`；因果用 `|ln(c/path)|`，EWMA 用 `|z|=|r|/σ`；`±95` 约贴轨）。",
        "",
        "| " + " | ".join(HARD_TOUCH_COLS) + " |",
        "|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rec in hard_touch_table(ok).to_dict(orient="records"):
        cells = [rec.get(c, "") or ("—" if c == "T收盘" else "") for c in HARD_TOUCH_COLS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines += [
        "## 全表",
        "",
        "| 代码 | 名称 | T收盘 | 状态 | 持仓 | 买点触发价 | 卖点触发价 | ②结束收复价 | 进② | ①触上轨价 | 图9下轨 | 图9上轨 | 图10下轨 | 图10上轨 | T+1状态(-3%/平/+3%) | 备注 |",
        "|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in _sort_rows(ok).iterrows():
        px0 = _f(r.get("px"))
        lines.append(
            f"| {r['code']} | {_md_cell(r.get('name'))} | "
            f"{'—' if px0 is None else f'{px0:.4f}'} | "
            f"{_md_cell(r.get('state'))} | {'是' if _truthy_in_pos(r.get('in_pos')) else ''} | "
            f"{_fmt_px_ret(_f(r.get('buy_px')), px0)} | "
            f"{_fmt_px_ret(_f(r.get('sell_px')), px0)} | "
            f"{_fmt_px_ret(_f(r.get('exit_knife_px')), px0)} | "
            f"{_fmt_ret(_f(r.get('enter_knife_px')), px0)} | "
            f"{_fmt_ret(_f(r.get('overheat_px')), px0)} | "
            f"{_rail_cell(r, px0, 'fig9_lo')} | "
            f"{_rail_cell(r, px0, 'fig9_hi')} | "
            f"{_rail_cell(r, px0, 'fig10_lo')} | "
            f"{_rail_cell(r, px0, 'fig10_hi')} | "
            f"{_state_band(r)} | "
            f"{_md_cell(r.get('note'))} |"
        )
    lines.append("")

    if not err.empty:
        lines += ["## 失败", "", "| 代码 | 名称 | 错误 |", "|---|---|---|"]
        for _, r in err.iterrows():
            lines.append(
                f"| {r.get('code') or ''} | {_md_cell(r.get('name'))} | {_md_cell(r.get('error'))} |"
            )
        lines.append("")
    if skipped_bonds:
        lines += ["跳过债券：" + "；".join(skipped_bonds), ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(argv: list[str] | None = None) -> pd.DataFrame:
    args = parse_args(argv)
    listing = _resolve(args.listing_dir)
    as_of = pd.Timestamp(args.as_of).normalize()
    next_day = pd.Timestamp(args.next_day).normalize()
    tag = next_day.strftime("%Y%m%d")
    out_csv = listing / f"next_day_triggers_{tag}.csv"
    out_md = listing / f"next_day_triggers_{tag}.md"
    out_hard = listing / f"next_day_triggers_{tag}_hard_touch.csv"
    skipped: list[str] = []
    if not args.include_bonds:
        for p in sorted(listing.glob("regime_transition_*_adaptive.html")):
            code, name = pool.parse_html_label(p)
            if pool.is_bond(code, name):
                skipped.append(f"{code} {name}")

    if args.rewrite_md_from_csv:
        if not out_csv.is_file():
            raise FileNotFoundError(out_csv)
        df = pd.read_csv(out_csv)
        rows = df.to_dict(orient="records")
        print(f"[INFO] rewrite md from {out_csv} rows={len(rows)}", flush=True)
        write_markdown(
            out_md,
            as_of=as_of.strftime("%Y-%m-%d"),
            next_day=next_day.strftime("%Y-%m-%d"),
            rows=rows,
            skipped_bonds=skipped,
            listing_dir=listing,
            jobs=int(args.jobs),
        )
        write_hard_touch_csv(out_hard, rows)
        print(f"[OK] {out_md}", flush=True)
        print(f"[OK] {out_hard}", flush=True)
        return df

    htmls = (
        pool.list_equity_html(listing)
        if not args.include_bonds
        else sorted(listing.glob("regime_transition_*_adaptive.html"))
    )
    payloads = []
    for html in htmls:
        code, name = pool.parse_html_label(html)
        payloads.append(
            {
                "html": str(html),
                "code": code,
                "name": name,
                "listing": str(listing),
                "as_of": as_of.strftime("%Y-%m-%d"),
                "next_day": next_day.strftime("%Y-%m-%d"),
                "lookback": int(args.lookback),
                "anchor_code": str(args.anchor_code).upper(),
            }
        )

    rows: list[dict[str, Any]] = []
    jobs = max(1, int(args.jobs))
    print(f"[INFO] {len(payloads)} names  jobs={jobs}  {as_of.date()}→{next_day.date()}", flush=True)
    if jobs == 1:
        for payload in payloads:
            print(f"[scan] {payload['code']}", flush=True)
            row = process_one(payload)
            rows.append(row)
            print(
                f"  -> {row.get('state') or row.get('error')} "
                f"buy={row.get('buy_px')} sell={row.get('sell_px')} "
                f"exit②={row.get('exit_knife_px')} n_eval={row.get('n_eval')}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_worker, p): p["code"] for p in payloads}
            for fut in as_completed(futs):
                code = futs[fut]
                try:
                    row = fut.result()
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "ok": False,
                        "code": code,
                        "name": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                rows.append(row)
                print(
                    f"[scan] {code} -> {row.get('state') or row.get('error')} "
                    f"buy={row.get('buy_px')} sell={row.get('sell_px')} "
                    f"exit②={row.get('exit_knife_px')} n_eval={row.get('n_eval')}",
                    flush=True,
                )

    tag = next_day.strftime("%Y%m%d")
    out_csv = listing / f"next_day_triggers_{tag}.csv"
    out_md = listing / f"next_day_triggers_{tag}.md"
    out_hard = listing / f"next_day_triggers_{tag}_hard_touch.csv"
    df = pd.DataFrame(rows)
    drop_cols = [c for c in ("trace",) if c in df.columns]
    df.drop(columns=drop_cols).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] {out_csv}", flush=True)
    write_markdown(
        out_md,
        as_of=as_of.strftime("%Y-%m-%d"),
        next_day=next_day.strftime("%Y-%m-%d"),
        rows=rows,
        skipped_bonds=skipped,
        listing_dir=listing,
        jobs=jobs,
    )
    write_hard_touch_csv(out_hard, rows)
    print(f"[OK] {out_md}", flush=True)
    print(f"[OK] {out_hard}", flush=True)
    focus = df.loc[
        df.get("ok", True)
        & pd.to_numeric(df.get("nearest_abs"), errors="coerce").le(FOCUS_ABS)
    ] if "nearest_abs" in df.columns else df.iloc[0:0]
    print(f"FOCUS ±3% ({len(focus)}): {', '.join(focus['code'].astype(str)) or '—'}", flush=True)
    return df


def main(argv: list[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
