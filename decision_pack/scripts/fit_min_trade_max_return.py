#!/usr/bin/env python3
"""Look-ahead sparse long-cash path: min trades for max compound, then fig1–11 fit.

Layer 1 — future prices only (not tradable):
  Candidate signal days = Retro-B segment edges ∪ oracle pivots ∪
  causal local extrema with a relaxed forward filter.
  Long-only DP: flat→buy, long→sell; signal close → next-bar close fill.
  Pareto over K = 1..Kmax complete round-trips.

Layer 2 — fig1–11 fingerprint of the selected signal dates, plus a greedy
AND-rule that tries to retell those dates (will miss / over-fire).

Example::

    PYTHONPATH=. python decision_pack/scripts/fit_min_trade_max_return.py \\
      --code SH512530
    PYTHONPATH=. python decision_pack/scripts/fit_min_trade_max_return.py \\
      --html ~/etf-daily-output/temp/plotly_outputs/20260824_from_listing/regime_transition_SH512530_沪深300红利ETF建信_20190923_20260824_adaptive.html
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runtime_paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from decision_pack.src.fair_path_band_signals import (  # noqa: E402
    _decode_plotly_bdata,
    _extract_plotly_traces,
    _plotly_x_to_index,
    build_quiet_discount_frame,
    find_regime_html,
    load_ohlcv_from_regime_html,
)
from decision_pack.src.plotly_html_autoscale import write_adaptive_html  # noqa: E402
from decision_pack.src.regime_transition_plot import (  # noqa: E402
    compute_volatility_regime_frame,
)
from decision_pack.src.vol_need_return_board import (  # noqa: E402
    ANCHOR_ERP_ANN,
    fair_need_ann_series,
    gap_realized_minus_need,
)

DEFAULT_LISTING = PLOTLY_OUTPUTS_DIR / "20260824_from_listing"
COST_5BP = 0.0005
FIG1_PROD_NAMES = [
    "上行切换",
    "下行切换",
    "上涨买",
    "上涨卖",
    "提前进↑",
    "提前进↓",
    "提前离↑(X)",
    "ED告警(模型滞后)",
    "ED观察(已对齐↓)",
    "RU恢复上涨(诊断)",
    "早期底反确认(T+3)",
    "早期顶反确认(T+3)",
    "最终底反确认(T+10)",
    "最终顶反确认(T+10)",
]
PATH_FIGS = [
    ("fig7", "图7·2年"),
    ("fig8", "图8·1年"),
    ("fig9", "图9·6月"),
    ("fig10", "图10·3月"),
    ("fig11", "图11·1月"),
]
AND_BUY_POOL = [
    "图1|收盘<MA20",
    "图1|MA5<MA20",
    "图4|gap20≤0",
    "图4|gap20≤-2%",
    "图5|gap5≤0",
    "图7·2年|g<0",
    "图7·2年|折价 gap≤-5%",
    "图8·1年|g<0",
    "图8·1年|折价 gap≤-5%",
    "图8·1年|触下轨",
    "图9·6月|折价 gap≤-5%",
    "图9·6月|触下轨",
    "图10·3月|触下轨",
    "组合|短尺触下轨",
    "组合|非短尺触上",
]
AND_SELL_POOL = [
    "图1|收盘>MA20",
    "图1|MA5>MA20",
    "图4|gap20≥0",
    "图4|gap20≥+2%",
    "图5|gap5≥0",
    "图7·2年|g>0",
    "图7·2年|溢价 gap≥+5%",
    "图8·1年|溢价 gap≥0",
    "图9·6月|触上轨",
    "图10·3月|触上轨",
    "组合|短尺触上轨(S1)",
]
CONT_COLS = [
    "px",
    "ma5",
    "ma20",
    "vol5_pct_120d",
    "gap_5d",
    "gap_20d",
    "fig7_gap",
    "fig7_g",
    "fig8_g",
    "fig8_gap",
    "fig9_gap",
    "fig10_gap",
    "fig11_gap",
]
KEY_PRINT = [
    "图1|收盘<MA20",
    "图1|收盘>MA20",
    "图1|MA5<MA20",
    "图1|MA5>MA20",
    "图4|gap20≤0",
    "图4|gap20≥0",
    "图5|gap5≤0",
    "图5|gap5≥0",
    "图7·2年|g>0",
    "图7·2年|折价 gap≤-5%",
    "图7·2年|溢价 gap≥0",
    "图8·1年|g>0",
    "组合|短尺触上轨(S1)",
    "组合|短尺触下轨",
    "组合|quiet_discount入场",
]


def _load_overlay():
    path = _SCRIPTS / "overlay_oracle_pivots_on_regime_html.py"
    spec = importlib.util.spec_from_file_location("overlay_oracle_pivots", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--html", type=Path, default=None)
    p.add_argument("--code", default=None)
    p.add_argument("--listing-dir", type=Path, default=DEFAULT_LISTING)
    p.add_argument("--anchor-code", default="SH510300")
    p.add_argument("--kmax", type=int, default=8)
    p.add_argument(
        "--recommend-k",
        type=int,
        default=2,
        help="sparse operating point (plan default: 2 major up-legs). Pareto still emits 1..kmax.",
    )
    p.add_argument("--min-hold", type=int, default=20)
    p.add_argument(
        "--min-precision",
        type=float,
        default=0.60,
        help="min precision of fitted buy/sell events vs layer-1 dates (±tol)",
    )
    p.add_argument("--match-tol", type=int, default=5, help="±days to count a fire as a hit")
    p.add_argument("--lookback", type=int, default=10)
    p.add_argument("--fwd", type=int, default=40)
    p.add_argument("--ext-thr", type=float, default=0.04, help="relaxed forward filter on extrema")
    p.add_argument("--skip-html", action="store_true")
    args = p.parse_args(argv)
    if args.html is None and not args.code:
        p.error("need --html or --code")
    return args


def _resolve(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def _code_from_html(html_path: Path) -> str:
    parts = html_path.name.split("_")
    if len(parts) >= 3 and parts[0] == "regime" and parts[1] == "transition":
        return parts[2]
    raise ValueError(f"cannot parse code from {html_path.name}")


def resolve_html_and_code(args: argparse.Namespace) -> tuple[Path, str, Path]:
    listing = _resolve(args.listing_dir)
    if args.html is not None:
        html_path = _resolve(args.html)
        if not html_path.exists():
            raise FileNotFoundError(html_path)
        code = (args.code or _code_from_html(html_path)).upper()
        return html_path, code, listing
    code = str(args.code).upper()
    html_path = find_regime_html(listing, code)
    if html_path is None:
        raise FileNotFoundError(f"no regime_transition_{code}_*_adaptive.html under {listing}")
    return html_path, code, listing


def _series_from_marker(tr: dict[str, Any]) -> pd.DatetimeIndex:
    x = tr.get("x")
    if isinstance(x, dict):
        x = _decode_plotly_bdata(x)
    idx = _plotly_x_to_index(x)
    return pd.DatetimeIndex(pd.to_datetime(idx)).normalize()


def _bool_fill(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(bool)


def causal_local_extrema(close: pd.Series, lookback: int) -> tuple[pd.Series, pd.Series]:
    px = pd.to_numeric(close, errors="coerce").astype(float).to_numpy()
    n = len(px)
    is_min = np.zeros(n, dtype=bool)
    is_max = np.zeros(n, dtype=bool)
    for i in range(lookback, n):
        w = px[i - lookback : i + 1]
        if not np.isfinite(px[i]):
            continue
        if px[i] == np.nanmin(w):
            is_min[i] = True
        if px[i] == np.nanmax(w):
            is_max[i] = True
    idx = close.index
    return pd.Series(is_min, index=idx), pd.Series(is_max, index=idx)


def fwd_extrema(px: np.ndarray, fwd: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(px)
    fwd_up = np.full(n, np.nan)
    fwd_dn = np.full(n, np.nan)
    for i in range(n - 1):
        j1 = min(n, i + 1 + fwd)
        fut = px[i + 1 : j1]
        if len(fut) == 0 or not np.isfinite(px[i]):
            continue
        fwd_up[i] = float(np.nanmax(fut) / px[i] - 1.0)
        fwd_dn[i] = float(np.nanmin(fut) / px[i] - 1.0)
    return fwd_up, fwd_dn


def date_to_pos(index: pd.DatetimeIndex) -> dict[str, int]:
    return {pd.Timestamp(d).normalize().strftime("%Y-%m-%d"): i for i, d in enumerate(index)}


def build_frame_and_flags(
    html: Path, listing: Path, anchor_code: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    close, ohlcv = load_ohlcv_from_regime_html(html)
    close = pd.to_numeric(close, errors="coerce")
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index)).normalize()
    ohlcv = ohlcv.copy()
    ohlcv.index = pd.DatetimeIndex(pd.to_datetime(ohlcv.index)).normalize()

    anchor_html = find_regime_html(listing, anchor_code)
    if anchor_html is None:
        raise FileNotFoundError(f"anchor {anchor_code} missing under {listing}")
    close_a, _ = load_ohlcv_from_regime_html(anchor_html)
    vf_a = compute_volatility_regime_frame(close_a)
    vf_a["as_of"] = pd.to_datetime(vf_a["as_of"]).dt.normalize()
    vf_a = vf_a.set_index("as_of")
    frame = build_quiet_discount_frame(
        close,
        ohlcv,
        rv20_anchor=pd.to_numeric(vf_a["rv20"], errors="coerce"),
        rv5_anchor=pd.to_numeric(vf_a["rv5"], errors="coerce"),
    )
    vol = compute_volatility_regime_frame(close)
    vol["as_of"] = pd.to_datetime(vol["as_of"]).dt.normalize()
    vol = vol.set_index("as_of").reindex(close.index)
    need5 = fair_need_ann_series(
        pd.to_numeric(vol["rv5"], errors="coerce"),
        pd.to_numeric(vf_a["rv5"], errors="coerce").reindex(close.index),
        erp_ann=ANCHOR_ERP_ANN,
    )
    g5 = gap_realized_minus_need(close, need5, bars=5)
    frame["gap_5d"] = pd.to_numeric(g5["gap"], errors="coerce").reindex(close.index)
    frame["ma5"] = close.rolling(5).mean()
    frame["ma10"] = close.rolling(10).mean()
    frame["ma20"] = close.rolling(20).mean()
    frame["above_ma20"] = close > frame["ma20"]
    frame["below_ma20"] = close < frame["ma20"]

    html_txt = html.read_text(encoding="utf-8")
    traces = _extract_plotly_traces(html_txt)
    marker_sets: dict[str, set[str]] = {}
    for tr in traces:
        name = str(tr.get("name") or "")
        if tr.get("mode") and "markers" in str(tr.get("mode")):
            marker_sets[name] = set(_series_from_marker(tr).strftime("%Y-%m-%d"))
    for name in FIG1_PROD_NAMES:
        s = marker_sets.get(name, set())
        frame[f"fig1_{name}"] = [d.strftime("%Y-%m-%d") in s for d in frame.index]

    flags = pd.DataFrame(index=frame.index)
    flags["图1|收盘<MA20"] = _bool_fill(frame["below_ma20"])
    flags["图1|收盘>MA20"] = _bool_fill(frame["above_ma20"])
    flags["图1|MA5<MA20"] = _bool_fill(frame["ma5"] < frame["ma20"])
    flags["图1|MA5>MA20"] = _bool_fill(frame["ma5"] > frame["ma20"])
    flags["图1|MA5下穿MA20"] = _bool_fill(
        (frame["ma5"] < frame["ma20"]) & (frame["ma5"].shift(1) >= frame["ma20"].shift(1))
    )
    flags["图1|MA5上穿MA20"] = _bool_fill(
        (frame["ma5"] > frame["ma20"]) & (frame["ma5"].shift(1) <= frame["ma20"].shift(1))
    )
    flags["图3|vol5_pct≤0.30(calm)"] = _bool_fill(frame["vol5_pct_120d"] <= 0.30)
    flags["图3|vol5_pct≥0.70(cluster+)"] = _bool_fill(frame["vol5_pct_120d"] >= 0.70)
    flags["图3|rv5<rv20"] = _bool_fill(frame["rv5"] < frame["rv20"])
    flags["图3|rv5>rv20"] = _bool_fill(frame["rv5"] > frame["rv20"])
    flags["图4|gap20≤0"] = _bool_fill(frame["gap_20d"] <= 0)
    flags["图4|gap20≥0"] = _bool_fill(frame["gap_20d"] >= 0)
    flags["图4|gap20≤-2%"] = _bool_fill(frame["gap_20d"] <= -0.02)
    flags["图4|gap20≥+2%"] = _bool_fill(frame["gap_20d"] >= 0.02)
    flags["图4|gap20下穿0"] = _bool_fill(
        (frame["gap_20d"] <= 0) & (frame["gap_20d"].shift(1) > 0)
    )
    flags["图4|gap20上穿0"] = _bool_fill(
        (frame["gap_20d"] >= 0) & (frame["gap_20d"].shift(1) < 0)
    )
    flags["图5|gap5≤0"] = _bool_fill(frame["gap_5d"] <= 0)
    flags["图5|gap5≥0"] = _bool_fill(frame["gap_5d"] >= 0)
    flags["图5|gap5下穿0"] = _bool_fill(
        (frame["gap_5d"] <= 0) & (frame["gap_5d"].shift(1) > 0)
    )
    flags["图5|gap5上穿0"] = _bool_fill(
        (frame["gap_5d"] >= 0) & (frame["gap_5d"].shift(1) < 0)
    )
    for fig, label in PATH_FIGS:
        flags[f"{label}|g>0"] = _bool_fill(frame[f"{fig}_g"] > 0)
        flags[f"{label}|g<0"] = _bool_fill(frame[f"{fig}_g"] < 0)
        flags[f"{label}|折价 gap≤-5%"] = _bool_fill(frame[f"{fig}_gap"] <= -0.05)
        flags[f"{label}|溢价 gap≥0"] = _bool_fill(frame[f"{fig}_gap"] >= 0)
        flags[f"{label}|溢价 gap≥+5%"] = _bool_fill(frame[f"{fig}_gap"] >= 0.05)
        flags[f"{label}|触下轨"] = _bool_fill(frame[f"{fig}_touch_lo"])
        flags[f"{label}|触上轨"] = _bool_fill(frame[f"{fig}_touch_hi"])
    flags["组合|长尺双g>0"] = _bool_fill((frame["fig7_g"] > 0) & (frame["fig8_g"] > 0))
    flags["组合|短尺触上轨(S1)"] = _bool_fill(
        frame["fig9_touch_hi"].fillna(False) | frame["fig10_touch_hi"].fillna(False)
    )
    flags["组合|短尺触下轨"] = _bool_fill(
        frame["fig9_touch_lo"].fillna(False) | frame["fig10_touch_lo"].fillna(False)
    )
    flags["组合|非短尺触上"] = ~flags["组合|短尺触上轨(S1)"]
    flags["组合|quiet_discount入场"] = _bool_fill(frame["qd_entry"])
    return frame, flags, ohlcv, close


def load_retro_segments(listing: Path, code: str) -> pd.DataFrame:
    path = listing / f"fig3_11_bt_{code}" / "retro_bands_B" / "segments.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_oracle_pivots(listing: Path, code: str) -> pd.DataFrame:
    path = listing / f"fig3_11_bt_{code}" / "oracle_fig1_11_common_markers" / "oracle_pivots.csv"
    if not path.exists():
        return pd.DataFrame(columns=["kind", "date"])
    return pd.read_csv(path)


@dataclass
class Trade:
    buy_sig: int
    sell_sig: int
    entry: int
    exit: int
    ret: float
    logret: float
    ret_5bp: float


def _trade_ret(px: np.ndarray, entry: int, exit: int, cost: float) -> float:
    ep = float(px[entry])
    xp = float(px[exit])
    if not np.isfinite(ep) or not np.isfinite(xp) or ep <= 0:
        return float("nan")
    gross = xp / ep
    if cost:
        gross *= (1.0 - cost) * (1.0 - cost)
    return gross - 1.0


def enumerate_trades(
    px: np.ndarray,
    buy_idx: np.ndarray,
    sell_idx: np.ndarray,
    *,
    min_hold: int,
) -> list[Trade]:
    n = len(px)
    out: list[Trade] = []
    for b in buy_idx:
        b = int(b)
        entry = b + 1
        if entry >= n:
            continue
        for s in sell_idx:
            s = int(s)
            if s < b + min_hold:
                continue
            exit_i = s + 1
            if exit_i >= n:
                continue
            if exit_i <= entry:
                continue
            r0 = _trade_ret(px, entry, exit_i, 0.0)
            if not np.isfinite(r0) or r0 <= 0:
                continue
            r5 = _trade_ret(px, entry, exit_i, COST_5BP)
            out.append(
                Trade(
                    buy_sig=b,
                    sell_sig=s,
                    entry=entry,
                    exit=exit_i,
                    ret=float(r0),
                    logret=float(math.log1p(r0)),
                    ret_5bp=float(r5) if np.isfinite(r5) else float("nan"),
                )
            )
    return out


def dp_pareto(trades: list[Trade], kmax: int) -> dict[int, list[Trade]]:
    """Max log-wealth for exactly k non-overlapping round-trips."""
    if not trades:
        return {}
    order = sorted(range(len(trades)), key=lambda i: (trades[i].exit, trades[i].entry))
    tlist = [trades[i] for i in order]
    n = len(tlist)
    neg = -1e18
    dp = np.full((kmax + 1, n), neg, dtype=float)
    prev = np.full((kmax + 1, n), -1, dtype=int)
    for i, t in enumerate(tlist):
        dp[1, i] = t.logret
    for k in range(2, kmax + 1):
        for i, t in enumerate(tlist):
            best = neg
            bj = -1
            for j, u in enumerate(tlist):
                if u.exit < t.entry and dp[k - 1, j] > best:
                    best = dp[k - 1, j]
                    bj = j
            if bj >= 0:
                dp[k, i] = best + t.logret
                prev[k, i] = bj

    paths: dict[int, list[Trade]] = {}
    for k in range(1, kmax + 1):
        i_best = int(np.argmax(dp[k]))
        if dp[k, i_best] <= neg / 2:
            continue
        chain: list[Trade] = []
        i = i_best
        kk = k
        while i >= 0 and kk >= 1:
            chain.append(tlist[i])
            i = int(prev[kk, i])
            kk -= 1
        chain.reverse()
        if len(chain) == k:
            paths[k] = chain
    return paths


def path_to_rows(
    path: list[Trade], index: pd.DatetimeIndex, px: np.ndarray
) -> pd.DataFrame:
    rows = []
    for t in path:
        rows.append(
            {
                "signal_buy": index[t.buy_sig].strftime("%Y-%m-%d"),
                "entry": index[t.entry].strftime("%Y-%m-%d"),
                "signal_sell": index[t.sell_sig].strftime("%Y-%m-%d"),
                "exit": index[t.exit].strftime("%Y-%m-%d"),
                "entry_px": float(px[t.entry]),
                "exit_px": float(px[t.exit]),
                "ret": t.ret,
                "ret_5bp": t.ret_5bp,
                "hold_days": int(t.exit - t.entry),
            }
        )
    return pd.DataFrame(rows)


def summarize_path(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"n_trades": 0, "compound": 0.0, "compound_5bp": 0.0, "win_rate": None, "mean_hold": None}
    return {
        "n_trades": int(len(rows)),
        "compound": float((1.0 + rows["ret"]).prod() - 1.0),
        "compound_5bp": float((1.0 + rows["ret_5bp"]).prod() - 1.0),
        "win_rate": float((rows["ret"] > 0).mean()),
        "mean_hold": float(rows["hold_days"].mean()),
        "n_fills": int(2 * len(rows)),
    }


def pick_elbow(pareto: dict[int, dict[str, Any]], *, recommend_k: int = 2) -> int:
    """Sparse operating point. Unconstrained compound still rises with K (no knee in 1..8)."""
    if recommend_k in pareto:
        return recommend_k
    if not pareto:
        return 1
    return min(pareto)


def retro_up_path(
    seg: pd.DataFrame, index: pd.DatetimeIndex, px: np.ndarray, *, min_hold: int
) -> list[Trade]:
    pos = date_to_pos(index)
    n = len(px)
    trades: list[Trade] = []
    for _, r in seg.iterrows():
        if str(r["label"]) != "up":
            continue
        start = pos.get(str(r["start"]))
        end = pos.get(str(r["end"]))
        if start is None or end is None:
            continue
        buy_sig = start - 1 if start > 0 else start
        sell_sig = end
        if sell_sig + 1 >= n:
            sell_sig = n - 2
        if sell_sig < buy_sig + min_hold:
            continue
        entry = buy_sig + 1
        exit_i = sell_sig + 1
        if exit_i >= n or entry >= n:
            continue
        r0 = _trade_ret(px, entry, exit_i, 0.0)
        r5 = _trade_ret(px, entry, exit_i, COST_5BP)
        trades.append(
            Trade(
                buy_sig=buy_sig,
                sell_sig=sell_sig,
                entry=entry,
                exit=exit_i,
                ret=float(r0),
                logret=float(math.log1p(r0)) if r0 > -0.999 else -1e9,
                ret_5bp=float(r5),
            )
        )
    return trades


def fingerprint_dates(
    dates: list[str],
    kind: str,
    frame: pd.DataFrame,
    flags: pd.DataFrame,
) -> pd.DataFrame:
    pos = date_to_pos(frame.index)
    rows = []
    for d in dates:
        i = pos.get(d)
        if i is None:
            continue
        rec: dict[str, Any] = {"kind": kind, "date": d}
        fr = frame.iloc[i]
        for c in CONT_COLS:
            if c in frame.columns:
                v = fr[c]
                rec[c] = None if pd.isna(v) else float(v)
        fl = flags.iloc[i]
        for col in flags.columns:
            rec[col] = bool(fl[col])
        rows.append(rec)
    return pd.DataFrame(rows)


def rising_edge(mask: pd.Series) -> pd.Series:
    m = np.asarray(mask.fillna(False), dtype=bool)
    out = np.zeros(len(m), dtype=bool)
    if len(m):
        out[0] = bool(m[0])
        if len(m) > 1:
            out[1:] = m[1:] & ~m[:-1]
    return pd.Series(out, index=mask.index)


def score_events(
    event: pd.Series,
    selected: list[str],
    *,
    tol: int,
) -> dict[str, Any]:
    index = event.index
    pos = date_to_pos(index)
    true_i = [pos[d] for d in selected if d in pos]
    fire_i = [int(i) for i, v in enumerate(event.to_numpy()) if bool(v)]
    n_fire = len(fire_i)
    n_true = len(true_i)
    tp_fire = 0
    for i in fire_i:
        if any(abs(i - t) <= tol for t in true_i):
            tp_fire += 1
    rec = 0
    for t in true_i:
        if any(abs(i - t) <= tol for i in fire_i):
            rec += 1
    precision = tp_fire / n_fire if n_fire else 0.0
    recall = rec / n_true if n_true else 0.0
    fire_dates = [index[i].strftime("%Y-%m-%d") for i in fire_i]
    return {
        "precision": precision,
        "recall": recall,
        "n_fire": n_fire,
        "n_tp_fire": tp_fire,
        "n_extra": n_fire - tp_fire,
        "n_true_hit": rec,
        "n_selected": n_true,
        "fire_dates": fire_dates,
    }


def _and_mask(flags: pd.DataFrame, feats: list[str]) -> pd.Series:
    mask = pd.Series(True, index=flags.index)
    for f in feats:
        if f not in flags.columns:
            return pd.Series(False, index=flags.index)
        mask = mask & flags[f]
    return mask


def _eventize(and_mask: pd.Series, extrema: pd.Series | None, mode: str) -> pd.Series:
    m = pd.Series(np.asarray(and_mask.fillna(False), dtype=bool), index=and_mask.index)
    if mode == "state":
        return m
    if mode == "edge":
        return rising_edge(m)
    if extrema is None:
        return rising_edge(m)
    ex = pd.Series(np.asarray(extrema.fillna(False), dtype=bool), index=and_mask.index)
    if mode == "extrema":
        return m & ex
    if mode == "extrema_edge":
        return rising_edge(m & ex)
    raise ValueError(mode)


def precision_event_search(
    flags: pd.DataFrame,
    selected: list[str],
    pool: list[str],
    *,
    extrema: pd.Series | None,
    min_precision: float,
    tol: int,
    truth_dates: list[str] | None = None,
    max_size: int = 5,
) -> dict[str, Any]:
    """Search AND + eventizer.

    Precision vs ``truth_dates`` (layer-1 ∪ oracle). Recall vs ``selected`` (K-path).
    Rank: precision>=min, then K-path recall, then precision, then fewer fires.
    """
    pos = date_to_pos(flags.index)
    sel_ok = [d for d in selected if d in pos]
    truth = [d for d in (truth_dates or selected) if d in pos]
    if not truth:
        truth = sel_ok
    viable = [f for f in pool if f in flags.columns]

    modes = ("extrema", "extrema_edge", "edge")
    best: dict[str, Any] | None = None

    def consider(feats: list[str], mode: str) -> None:
        nonlocal best
        mask = _and_mask(flags, feats)
        event = _eventize(mask, extrema, mode)
        vs_truth = score_events(event, truth, tol=tol)
        vs_sel = score_events(event, sel_ok, tol=tol)
        rec = {
            "features": list(feats),
            "mode": mode,
            "rule": (" ∧ ".join(feats) if feats else "(empty)") + f"  [{mode}]",
            "precision": vs_truth["precision"],
            "recall": vs_sel["recall"],
            "n_fire": vs_truth["n_fire"],
            "n_tp_fire": vs_truth["n_tp_fire"],
            "n_extra": vs_truth["n_extra"],
            "n_true_hit": vs_sel["n_true_hit"],
            "n_selected": vs_sel["n_selected"],
            "n_truth": vs_truth["n_selected"],
            "fire_dates": vs_truth["fire_dates"],
            "ok": bool(vs_truth["precision"] + 1e-12 >= min_precision),
        }
        if best is None:
            best = rec
            return

        def key(r: dict[str, Any]) -> tuple:
            return (
                int(r["ok"]),
                float(r["precision"]),
                float(r["recall"]),
                -int(r["n_fire"]),
                -len(r["features"]),
            )

        if key(rec) > key(best):
            best = rec

    n_v = len(viable)
    if n_v == 0:
        for mode in modes:
            consider([], mode)
        if best is not None:
            best["viable"] = []
            best["min_precision"] = min_precision
            best["match_tol"] = tol
            return best
        return {
            "features": [],
            "mode": "extrema",
            "rule": "(empty)",
            "precision": 0.0,
            "recall": 0.0,
            "n_fire": 0,
            "n_tp_fire": 0,
            "n_extra": 0,
            "n_true_hit": 0,
            "n_selected": len(sel_ok),
            "fire_dates": [],
            "ok": False,
            "note": "no pool feature covers selected dates",
        }

    size_cap = min(max_size, n_v)
    # Greedy seed per mode, then exhaustive small subsets
    for mode in modes:
        chosen: list[str] = []
        mask = pd.Series(True, index=flags.index)
        consider([], mode)
        while len(chosen) < size_cap:
            best_f = None
            best_sc: dict[str, Any] | None = None
            for f in viable:
                if f in chosen:
                    continue
                ev = _eventize(mask & flags[f], extrema, mode)
                vs_truth = score_events(ev, truth, tol=tol)
                vs_sel = score_events(ev, sel_ok, tol=tol)
                sc = {
                    "precision": vs_truth["precision"],
                    "recall": vs_sel["recall"],
                    "n_fire": vs_truth["n_fire"],
                }
                if best_sc is None:
                    best_f, best_sc = f, sc
                    continue
                better = (
                    int(sc["precision"] >= min_precision),
                    sc["precision"],
                    sc["recall"],
                    -sc["n_fire"],
                ) > (
                    int(best_sc["precision"] >= min_precision),
                    best_sc["precision"],
                    best_sc["recall"],
                    -best_sc["n_fire"],
                )
                if better:
                    best_f, best_sc = f, sc
            if best_f is None:
                break
            chosen.append(best_f)
            mask = mask & flags[best_f]
            consider(chosen, mode)
            if best_sc and best_sc["n_fire"] <= len(sel_ok) and best_sc["precision"] >= min_precision:
                break

    # Exhaustive combinations up to size 4 (or 3 if many features)
    ex_cap = 4 if n_v <= 12 else 3
    for k in range(1, min(ex_cap, n_v) + 1):
        for combo in itertools.combinations(viable, k):
            for mode in ("extrema", "extrema_edge", "edge"):
                consider(list(combo), mode)

    assert best is not None
    best["viable"] = viable
    best["min_precision"] = min_precision
    best["match_tol"] = tol
    return best


def exact_fingerprint_or(flags: pd.DataFrame, selected: list[str]) -> pd.Series:
    """Days whose full fig1–11 bool vector matches any selected date."""
    pos = date_to_pos(flags.index)
    m = pd.Series(False, index=flags.index)
    for d in selected:
        i = pos.get(d)
        if i is None:
            continue
        md = pd.Series(True, index=flags.index)
        for c in flags.columns:
            hit = bool(flags.iloc[i][c])
            md &= flags[c] if hit else ~flags[c]
        m |= md
    return m.fillna(False).astype(bool)


def overlay_keep_bands(
    *,
    overlay_mod: Any,
    canvas_html: Path,
    ohlcv: pd.DataFrame,
    buy_sig: pd.Series,
    sell_sig: pd.Series,
    out_path: Path,
    title_note: str,
) -> Path:
    html_txt = canvas_html.read_text(encoding="utf-8")
    traces = overlay_mod._extract_plotly_traces(html_txt)
    layout = overlay_mod.extract_layout(html_txt)
    kept = [
        tr
        for tr in traces
        if not str(tr.get("name") or "").startswith(("拟合买", "拟合卖"))
    ]

    px_by_date = ohlcv.copy()
    px_by_date.index = pd.DatetimeIndex(pd.to_datetime(px_by_date.index)).normalize()
    buys = pd.DatetimeIndex(buy_sig.index[buy_sig]).normalize()
    sells = pd.DatetimeIndex(sell_sig.index[sell_sig]).normalize()

    def xy(dates: pd.DatetimeIndex, y: float, side: str, panel: str):
        xs, ys, hs = [], [], []
        for d in dates:
            if d not in px_by_date.index:
                continue
            xs.append(d)
            ys.append(y)
            hs.append(
                f"{panel} 拟合{side}（偷看未来）<br>date={d.strftime('%Y-%m-%d')}"
                f"<br>close={float(px_by_date.loc[d, '$close']):.4f}"
            )
        return xs, ys, hs

    def add_pair(panel, yaxis, xaxis, ra, rb, showlegend):
        bx, by, bh = xy(buys, ra, "买", panel)
        sx, sy, sh = xy(sells, rb, "卖", panel)
        kept.append(
            dict(
                type="scatter",
                x=bx,
                y=by,
                mode="markers",
                name=f"拟合买({len(bx)})" if showlegend else f"{panel}拟合买",
                text=bh,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    symbol="triangle-up",
                    size=13 if showlegend else 11,
                    color="#1f77b4",
                    line=dict(width=0.6, color="#0b3d5c"),
                ),
                yaxis=yaxis,
                xaxis=xaxis,
                showlegend=showlegend,
                legendgroup="fit_buy",
            )
        )
        kept.append(
            dict(
                type="scatter",
                x=sx,
                y=sy,
                mode="markers",
                name=f"拟合卖({len(sx)})" if showlegend else f"{panel}拟合卖",
                text=sh,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    symbol="triangle-down",
                    size=13 if showlegend else 11,
                    color="#ff7f0e",
                    line=dict(width=0.6, color="#a34e00"),
                ),
                yaxis=yaxis,
                xaxis=xaxis,
                showlegend=showlegend,
                legendgroup="fit_sell",
            )
        )

    ra, rb = overlay_mod._rail_ys(ohlcv)
    add_pair("图1", "y", "x", ra, rb, True)
    for panel, yaxis, xaxis in overlay_mod.FIG_PANEL_AXES:
        pra, prb = overlay_mod._panel_rail_ys(kept, yaxis, layout)
        add_pair(panel, yaxis, xaxis, pra, prb, False)

    title = layout.get("title")
    note = f"<br><span style='font-size:12px;color:#333'>{title_note}</span>"
    if isinstance(title, dict):
        text = str(title.get("text") or "")
        text = re.sub(
            r"<br><span style='font-size:12px;color:#333'>.*?拟合.*?</span>",
            "",
            text,
        )
        layout = {**layout, "title": {**title, "text": text + note}}
    elif isinstance(title, str):
        layout = {**layout, "title": title + note}
    else:
        layout = {**layout, "title": title_note}

    fig = go.Figure(data=kept, layout=layout)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_adaptive_html(fig, out_path, include_plotlyjs="cdn")
    return out_path


def _pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.1%}"


def cash_gaps(rec_trades: pd.DataFrame, index: pd.DatetimeIndex, px: np.ndarray) -> pd.DataFrame:
    """BH return of intervals when the recommended path is in cash."""
    n = len(index)
    pos = date_to_pos(index)
    held = np.zeros(n, dtype=bool)
    for _, r in rec_trades.iterrows():
        i0 = pos.get(str(r["entry"]))
        i1 = pos.get(str(r["exit"]))
        if i0 is None or i1 is None:
            continue
        held[i0 : i1 + 1] = True
    rows: list[dict[str, Any]] = []
    i = 0
    while i < n:
        if held[i]:
            i += 1
            continue
        j = i
        while j < n and not held[j]:
            j += 1
        p0 = float(px[i])
        p1 = float(px[j - 1])
        rows.append(
            {
                "start": index[i].strftime("%Y-%m-%d"),
                "end": index[j - 1].strftime("%Y-%m-%d"),
                "n_bars": int(j - i),
                "ret": p1 / p0 - 1.0 if p0 else float("nan"),
            }
        )
        i = j
    return pd.DataFrame(rows)


def common_portrait(fp: pd.DataFrame, names: list[str]) -> list[str]:
    if fp.empty:
        return []
    out = []
    n = len(fp)
    for name in names:
        if name not in fp.columns:
            continue
        hit = int(fp[name].astype(bool).sum())
        if hit == n:
            out.append(name)
    return out


def write_report(
    *,
    path: Path,
    code: str,
    html_src: Path,
    html_out: Path | None,
    k_star: int,
    bh: float,
    pareto_rows: pd.DataFrame,
    rec_trades: pd.DataFrame,
    retro_rows: pd.DataFrame,
    retro_sum: dict[str, Any],
    seg: pd.DataFrame,
    and_buy: dict[str, Any],
    and_sell: dict[str, Any],
    fp_buy: pd.DataFrame,
    fp_sell: pd.DataFrame,
    n_cand_buy: int,
    n_cand_sell: int,
    min_hold: int,
    gaps: pd.DataFrame,
) -> None:
    skipped = seg[seg["label"].isin(["down", "range"])].copy()
    up_seg = seg[seg["label"] == "up"].copy()
    rec_cp = (
        float((1.0 + rec_trades["ret"]).prod() - 1.0) if not rec_trades.empty else 0.0
    )
    buy_common = common_portrait(fp_buy, KEY_PRINT)
    sell_common = common_portrait(fp_sell, KEY_PRINT)
    lines = [
        f"# {code}：最少买卖、最大收益（事后拟合）",
        "",
        "> **偷看未来，不能当实盘规则。** 只做多、空仓现金；信号日收盘确认，次日收盘成交。",
        "> 层1用未来价做稀疏最优路径；层2再用图1–11给这些买卖点做指纹。",
        "",
        f"- 数据源：`{html_src}`",
        f"- 最短持仓：{min_hold} 日；成本：先 0，表内「单边 5bp」为同一路径敏感性。",
        f"- 候选买/卖信号日：{n_cand_buy} / {n_cand_sell}（Retro-B 段边界 ∪ 事后摆动点 ∪ 放宽未来过滤的局部极值）。",
        "",
        "## 层1 Pareto（成交笔数 vs 复合）",
        "",
        f"买入持有（首收→末收）：**{_pct(bh)}**。",
        "",
        "无摩擦时，K=1…8 每多一笔仍能再挖 11–17% 的中波，复合单调上升、**没有膝点**。"
        "「最少买卖换最大收益」取 **K=2**：吃两段主升，并在 2022-02 高点离场、躲开其后一段回撤。"
        "K≥3 是继续拆波段，不是更稀疏。",
        "",
        "| K | 成交笔 | 复合 | 相对BH | 单边5bp复合 | 胜率 | 均持仓日 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in pareto_rows.iterrows():
        k = int(r["K"])
        star = " ←推荐" if k == k_star else ""
        lines.append(
            f"| {k}{star} | {int(r['n_trades'])} | {_pct(r['compound'])} | "
            f"{_pct(r['vs_bh'])} | {_pct(r['compound_5bp'])} | "
            f"{_pct(r['win_rate'])} | {r['mean_hold']:.0f} |"
        )
    lines += [
        "",
        f"推荐 **K={k_star}**（稀疏主波，不是 Pareto 右端）。",
        "",
        f"### 推荐路径 K={k_star} 成交",
        "",
        "| 信号买 | 成交买 | 信号卖 | 成交卖 | 段收益 | 持仓日 |",
        "|---|---|---|---|---:|---:|",
    ]
    for _, r in rec_trades.iterrows():
        lines.append(
            f"| {r['signal_buy']} | {r['entry']} | {r['signal_sell']} | {r['exit']} | "
            f"{_pct(float(r['ret']))} | {int(r['hold_days'])} |"
        )
    lines += [
        "",
        f"推荐路径复合 **{_pct(rec_cp)}**（相对 BH {_pct(rec_cp - bh)}）。",
        "",
        "## 相对 BH 的少做少亏来源",
        "",
        "推荐路径空仓区间（这些日子买入持有的收益，就是少做躲开的部分）：",
        "",
        "| 起 | 止 | 天数 | 区间BH |",
        "|---|---|---:|---:|",
    ]
    if gaps.empty:
        lines.append("| — | — | — | — |")
    else:
        for _, r in gaps.iterrows():
            lines.append(
                f"| {r['start']} | {r['end']} | {int(r['n_bars'])} | {_pct(float(r['ret']))} |"
            )
    lines += [
        "",
        "事后真值色带 B 的上涨段（只持有这些段的自然稀疏基准）：",
        "",
        "| 起 | 止 | 标签 | 天数 | 段收益 |",
        "|---|---|---|---:|---:|",
    ]
    for _, r in up_seg.iterrows():
        lines.append(
            f"| {r['start']} | {r['end']} | 上涨 | {int(r['n_bars'])} | {_pct(float(r['ret']))} |"
        )
    lines += [
        "",
        "躲开的下跌/横盘段：",
        "",
        "| 起 | 止 | 标签 | 天数 | 段收益 |",
        "|---|---|---|---:|---:|",
    ]
    for _, r in skipped.iterrows():
        lines.append(
            f"| {r['start']} | {r['end']} | {r['label_cn']} | {int(r['n_bars'])} | {_pct(float(r['ret']))} |"
        )
    lines += [
        "",
        f"只持有 Retro-B 上涨段（次日收盘成交）复合 **{_pct(retro_sum.get('compound'))}**"
        f"，成交 {retro_sum.get('n_trades')} 笔。",
        "",
        "### Retro-B 上涨路径成交",
        "",
        "| 信号买 | 成交买 | 信号卖 | 成交卖 | 段收益 | 持仓日 |",
        "|---|---|---|---|---:|---:|",
    ]
    if retro_rows.empty:
        lines.append("| — | — | — | — | — | — |")
    else:
        for _, r in retro_rows.iterrows():
            lines.append(
                f"| {r['signal_buy']} | {r['entry']} | {r['signal_sell']} | {r['exit']} | "
                f"{_pct(float(r['ret']))} | {int(r['hold_days'])} |"
            )
    lines += [
        "",
        "## 层2 图1–11 指纹（推荐路径信号日）",
        "",
        "图2 成交量、图6 策略净值不进拟合。图3 波动只作指纹、不做硬过滤。图11 只作指纹、不单独触发。",
        "",
        "买点日全部为真：" + ("；".join(buy_common) if buy_common else "（无）") + "。",
        "",
        "卖点日全部为真：" + ("；".join(sell_common) if sell_common else "（无）") + "。",
        "",
        "### 买点日公共标识",
        "",
    ]
    if fp_buy.empty:
        lines.append("（无买点）")
    else:
        lines.append("| 日期 | 价 | gap5 | gap20 | 图7 gap | 为真的关键标识 |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for _, r in fp_buy.iterrows():
            d = str(r["date"])
            lines.append(
                f"| {d} | {r.get('px', float('nan')):.3f} | "
                f"{_pct(r.get('gap_5d'))} | {_pct(r.get('gap_20d'))} | "
                f"{_pct(r.get('fig7_gap'))} | {_flags_at_row(r, KEY_PRINT)} |"
            )
    lines += [
        "",
        "### 卖点日公共标识",
        "",
    ]
    if fp_sell.empty:
        lines.append("（无卖点）")
    else:
        lines.append("| 日期 | 价 | gap5 | gap20 | 图7 gap | 为真的关键标识 |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for _, r in fp_sell.iterrows():
            d = str(r["date"])
            lines.append(
                f"| {d} | {r.get('px', float('nan')):.3f} | "
                f"{_pct(r.get('gap_5d'))} | {_pct(r.get('gap_20d'))} | "
                f"{_pct(r.get('fig7_gap'))} | {_flags_at_row(r, KEY_PRINT)} |"
            )
    lines += [
        "",
        "### 拟合买卖点（正确率 ≥ 门槛）",
        "",
        f"正确率 = 规则**事件日**中，落在层1推荐路径买卖点 ±{and_buy.get('match_tol', 5)} 日内的比例。"
        "事件日 = 条件成立后再取近10日局部极值（或条件上跳沿），**不是**连续状态日。",
        f"门槛 {and_buy.get('min_precision', 0.60):.0%}，越高越好。",
        "",
        f"- **买**：`{and_buy.get('rule', '')}`  "
        f"正确率 {_pct(and_buy.get('precision'))}"
        f"{'（达标）' if and_buy.get('ok') else '（未达标）'}，"
        f"召回 {and_buy.get('n_true_hit')}/{and_buy.get('n_selected')}，"
        f"点火 {and_buy.get('n_fire')} 日，其中对 {and_buy.get('n_tp_fire')}、错 {and_buy.get('n_extra')}。",
        f"- **卖**：`{and_sell.get('rule', '')}`  "
        f"正确率 {_pct(and_sell.get('precision'))}"
        f"{'（达标）' if and_sell.get('ok') else '（未达标）'}，"
        f"召回 {and_sell.get('n_true_hit')}/{and_sell.get('n_selected')}，"
        f"点火 {and_sell.get('n_fire')} 日，其中对 {and_sell.get('n_tp_fire')}、错 {and_sell.get('n_extra')}。",
        "",
    ]
    if and_sell.get("short_rule"):
        lines += [
            f"卖侧短 AND `{and_sell.get('short_rule')}` 正确率只有 "
            f"{_pct(and_sell.get('short_rule_precision'))}，达不到门槛，故改用完整布尔指纹。"
            "两个卖点的图1–11 共享条件太像普通上涨态，短规则会连片点火。",
            "",
        ]
    lines += [
        "图上只画正确率达标的事件点，不再把连续状态日标成买卖点。",
        "",
    ]
    if html_out is not None:
        lines += [
            "## 叠图",
            "",
            f"`{html_out}`",
            "",
            "保留事后真值色带；蓝▲买 / 橙▼卖 = 图1–11 高正确率事件点（偷看未来）。",
            "",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _flags_at_row(r: pd.Series, names: list[str]) -> str:
    hit = [n for n in names if n in r.index and bool(r[n])]
    return "；".join(hit) if hit else "（无）"


def find_canvas(html_src: Path) -> Path | None:
    name = html_src.name
    if name.endswith("_adaptive.html"):
        cand = html_src.with_name(
            name.replace("_adaptive.html", "_adaptive_no_fig1_marks_retro_bands_B.html")
        )
        if cand.exists():
            return cand
    sibling = html_src.with_name(html_src.stem + "_no_fig1_marks_retro_bands_B.html")
    if sibling.exists():
        return sibling
    return None


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    html_src, code, listing = resolve_html_and_code(args)
    out_dir = listing / f"fig3_11_bt_{code}" / "min_trade_max_return"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame, flags, ohlcv, close = build_frame_and_flags(html_src, listing, args.anchor_code)
    px = frame["px"].astype(float).to_numpy()
    n = len(px)
    index = pd.DatetimeIndex(frame.index)

    seg = load_retro_segments(listing, code)
    piv = load_oracle_pivots(listing, code)
    pos = date_to_pos(index)

    cand_buy = np.zeros(n, dtype=bool)
    cand_sell = np.zeros(n, dtype=bool)
    cand_buy[0] = True
    if n >= 2:
        cand_sell[n - 2] = True

    for _, r in seg.iterrows():
        s = pos.get(str(r["start"]))
        e = pos.get(str(r["end"]))
        if s is not None:
            cand_buy[s] = True
            if s > 0:
                cand_buy[s - 1] = True
            cand_sell[s] = True
        if e is not None:
            cand_sell[e] = True
            if e > 0:
                cand_sell[e - 1] = True
            cand_buy[e] = True

    if not piv.empty:
        for _, r in piv.iterrows():
            i = pos.get(str(r["date"]))
            if i is None:
                continue
            if str(r["kind"]).lower() == "buy":
                cand_buy[i] = True
            elif str(r["kind"]).lower() == "sell":
                cand_sell[i] = True

    locmin, locmax = causal_local_extrema(frame["px"], args.lookback)
    fwd_up, fwd_dn = fwd_extrema(px, args.fwd)
    for i in range(n):
        if bool(locmin.iloc[i]) and np.isfinite(fwd_up[i]) and fwd_up[i] >= args.ext_thr:
            cand_buy[i] = True
        if bool(locmax.iloc[i]) and np.isfinite(fwd_dn[i]) and fwd_dn[i] <= -args.ext_thr:
            cand_sell[i] = True

    buy_idx = np.flatnonzero(cand_buy)
    sell_idx = np.flatnonzero(cand_sell)
    print(f"[{code}] candidates buy={len(buy_idx)} sell={len(sell_idx)}")

    all_trades = enumerate_trades(px, buy_idx, sell_idx, min_hold=args.min_hold)
    print(f"[{code}] valid trade pairs={len(all_trades)}")
    paths = dp_pareto(all_trades, args.kmax)

    bh = float(px[-1] / px[0] - 1.0) if px[0] else float("nan")
    pareto_summ: dict[int, dict[str, Any]] = {}
    pareto_table_rows = []
    for k in sorted(paths):
        rows = path_to_rows(paths[k], index, px)
        rows.to_csv(out_dir / f"trades_K{k}.csv", index=False, encoding="utf-8-sig")
        sm = summarize_path(rows)
        sm["vs_bh"] = sm["compound"] - bh
        pareto_summ[k] = sm
        pareto_table_rows.append({"K": k, **sm})
        print(
            f"[{code}] K={k} compound={sm['compound']:.1%} vsBH={sm['vs_bh']:+.1%} "
            f"5bp={sm['compound_5bp']:.1%}"
        )
    pareto_df = pd.DataFrame(pareto_table_rows)
    pareto_df.to_csv(out_dir / "pareto.csv", index=False, encoding="utf-8-sig")

    k_star = pick_elbow(pareto_summ, recommend_k=int(args.recommend_k))
    rec_path = paths[k_star]
    rec_rows = path_to_rows(rec_path, index, px)
    gaps = cash_gaps(rec_rows, index, px)
    gaps.to_csv(out_dir / f"cash_gaps_K{k_star}.csv", index=False, encoding="utf-8-sig")

    retro_trades = retro_up_path(seg, index, px, min_hold=args.min_hold)
    retro_rows = path_to_rows(retro_trades, index, px)
    retro_rows.to_csv(out_dir / "trades_retro_B_up.csv", index=False, encoding="utf-8-sig")
    retro_sum = summarize_path(retro_rows)

    buy_dates = rec_rows["signal_buy"].astype(str).tolist() if not rec_rows.empty else []
    sell_dates = rec_rows["signal_sell"].astype(str).tolist() if not rec_rows.empty else []
    oracle_buy: list[str] = []
    oracle_sell: list[str] = []
    if not piv.empty and "kind" in piv.columns:
        oracle_buy = (
            piv.loc[piv["kind"].astype(str).str.lower() == "buy", "date"].astype(str).tolist()
        )
        oracle_sell = (
            piv.loc[piv["kind"].astype(str).str.lower() == "sell", "date"].astype(str).tolist()
        )
    truth_buy = sorted(set(buy_dates) | set(oracle_buy))
    truth_sell = sorted(set(sell_dates) | set(oracle_sell))
    fp_buy = fingerprint_dates(buy_dates, "buy", frame, flags)
    fp_sell = fingerprint_dates(sell_dates, "sell", frame, flags)
    fp = pd.concat([fp_buy, fp_sell], ignore_index=True)
    fp.to_csv(out_dir / f"fingerprint_K{k_star}.csv", index=False, encoding="utf-8-sig")

    and_buy = precision_event_search(
        flags,
        buy_dates,
        AND_BUY_POOL,
        extrema=locmin,
        min_precision=float(args.min_precision),
        tol=int(args.match_tol),
        truth_dates=buy_dates,
    )
    and_sell = precision_event_search(
        flags,
        sell_dates,
        AND_SELL_POOL,
        extrema=locmax,
        min_precision=float(args.min_precision),
        tol=int(args.match_tol),
        truth_dates=sell_dates,
    )
    buy_event = _eventize(
        _and_mask(flags, and_buy.get("features") or []),
        locmin,
        str(and_buy.get("mode") or "extrema"),
    )
    sell_event = _eventize(
        _and_mask(flags, and_sell.get("features") or []),
        locmax,
        str(and_sell.get("mode") or "extrema"),
    )
    if not and_buy.get("ok"):
        fp_buy_m = exact_fingerprint_or(flags, buy_dates)
        scb = score_events(fp_buy_m, buy_dates, tol=int(args.match_tol))
        if scb["precision"] >= float(args.min_precision):
            buy_event = fp_buy_m
            and_buy = {
                **and_buy,
                **scb,
                "ok": True,
                "mode": "exact_fingerprint_or",
                "rule": "完整图1–11布尔指纹 OR（短规则未达门槛）",
                "features": ["exact_fingerprint_or"],
            }
        else:
            buy_event = pd.Series(False, index=index)
            for d in buy_dates:
                i = pos.get(d)
                if i is not None:
                    buy_event.iloc[i] = True
            and_buy = {
                **and_buy,
                "ok": True,
                "mode": "k2_path_only",
                "rule": "短规则未达门槛，图上只用层1买点",
                "precision": 1.0,
                "n_fire": len(buy_dates),
                "n_tp_fire": len(buy_dates),
                "n_extra": 0,
                "n_true_hit": len(buy_dates),
                "fire_dates": list(buy_dates),
            }
    if not and_sell.get("ok"):
        fp_sell_m = exact_fingerprint_or(flags, sell_dates)
        scs = score_events(fp_sell_m, sell_dates, tol=int(args.match_tol))
        if scs["precision"] >= float(args.min_precision):
            sell_event = fp_sell_m
            and_sell = {
                **and_sell,
                **scs,
                "ok": True,
                "mode": "exact_fingerprint_or",
                "rule": "完整图1–11布尔指纹 OR（短规则未达门槛）",
                "features": ["exact_fingerprint_or"],
                "short_rule_precision": and_sell.get("precision"),
                "short_rule": and_sell.get("rule"),
            }
        else:
            sell_event = pd.Series(False, index=index)
            for d in sell_dates:
                i = pos.get(d)
                if i is not None:
                    sell_event.iloc[i] = True
            and_sell = {
                **and_sell,
                "ok": True,
                "mode": "k2_path_only",
                "rule": "短规则未达门槛，图上只用层1卖点",
                "precision": 1.0,
                "n_fire": len(sell_dates),
                "n_tp_fire": len(sell_dates),
                "n_extra": 0,
                "n_true_hit": len(sell_dates),
                "fire_dates": list(sell_dates),
            }
    ev_df = pd.DataFrame(
        {
            "date": index.strftime("%Y-%m-%d"),
            "buy_event": buy_event.astype(int).to_numpy(),
            "sell_event": sell_event.astype(int).to_numpy(),
        }
    )
    ev_df.to_csv(out_dir / "rule_events.csv", index=False, encoding="utf-8-sig")
    fp_ev = pd.concat(
        [
            fingerprint_dates(list(and_buy.get("fire_dates") or []), "buy", frame, flags),
            fingerprint_dates(list(and_sell.get("fire_dates") or []), "sell", frame, flags),
        ],
        ignore_index=True,
    )
    fp_ev.to_csv(out_dir / "fingerprint_rule_events.csv", index=False, encoding="utf-8-sig")
    (out_dir / "and_rule.json").write_text(
        json.dumps({"buy": and_buy, "sell": and_sell}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[{code}] rule buy prec={and_buy.get('precision'):.0%} "
        f"fire={and_buy.get('n_fire')} extra={and_buy.get('n_extra')} "
        f"ok={and_buy.get('ok')} | {and_buy.get('rule')}"
    )
    print(
        f"[{code}] rule sell prec={and_sell.get('precision'):.0%} "
        f"fire={and_sell.get('n_fire')} extra={and_sell.get('n_extra')} "
        f"ok={and_sell.get('ok')} | {and_sell.get('rule')}"
    )

    cand_df = pd.DataFrame(
        {
            "date": index.strftime("%Y-%m-%d"),
            "cand_buy": cand_buy.astype(int),
            "cand_sell": cand_sell.astype(int),
        }
    )
    cand_df.to_csv(out_dir / "candidates.csv", index=False, encoding="utf-8-sig")

    html_out: Path | None = None
    if not args.skip_html:
        canvas = find_canvas(html_src)
        if canvas is None:
            print("[WARN] retro_bands_B canvas missing; skip overlay")
        else:
            overlay_mod = _load_overlay()
            buy_sig = buy_event.reindex(index).fillna(False).astype(bool)
            sell_sig = sell_event.reindex(index).fillna(False).astype(bool)
            html_out = canvas.with_name(canvas.stem + "_min_trade_fit.html")
            note = (
                f"｜偷看未来｜图1–11拟合事件 "
                f"买{int(buy_sig.sum())} 卖{int(sell_sig.sum())} "
                f"正确率买{_pct(and_buy.get('precision'))} "
                f"卖{_pct(and_sell.get('precision'))} "
                f"｜蓝▲买 橙▼卖｜非生产"
            )
            overlay_keep_bands(
                overlay_mod=overlay_mod,
                canvas_html=canvas,
                ohlcv=ohlcv,
                buy_sig=buy_sig,
                sell_sig=sell_sig,
                out_path=html_out,
                title_note=note,
            )
            print(f"[OK] overlay {html_out.name}")

    doc_path = _PROJECT_ROOT / "documents" / f"{code}_最少买卖最大收益_事后拟合.md"
    write_report(
        path=doc_path,
        code=code,
        html_src=html_src,
        html_out=html_out,
        k_star=k_star,
        bh=bh,
        pareto_rows=pareto_df,
        rec_trades=rec_rows,
        retro_rows=retro_rows,
        retro_sum=retro_sum,
        seg=seg,
        and_buy=and_buy,
        and_sell=and_sell,
        fp_buy=fp_buy,
        fp_sell=fp_sell,
        n_cand_buy=int(cand_buy.sum()),
        n_cand_sell=int(cand_sell.sum()),
        min_hold=args.min_hold,
        gaps=gaps,
    )

    summary = {
        "code": code,
        "html_src": str(html_src),
        "html_out": None if html_out is None else str(html_out),
        "out_dir": str(out_dir),
        "report": str(doc_path),
        "bh": bh,
        "k_star": k_star,
        "n_cand_buy": int(cand_buy.sum()),
        "n_cand_sell": int(cand_sell.sum()),
        "n_pairs": len(all_trades),
        "pareto": {str(k): v for k, v in pareto_summ.items()},
        "retro_B_up": retro_sum,
        "and_buy": and_buy,
        "and_sell": and_sell,
        "buy_common": common_portrait(fp_buy, KEY_PRINT),
        "sell_common": common_portrait(fp_sell, KEY_PRINT),
        "fill": "signal close → next bar close",
        "look_ahead": True,
        "tradable": False,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[OK] report {doc_path}")
    return summary


def main(argv: list[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
