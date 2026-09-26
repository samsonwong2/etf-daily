#!/usr/bin/env python3
"""全池因果状态划分评估：无前瞻方法 vs 回顾峰谷真值。

目标：在所有可测标的、全部可用历史上，比较多种仅用当日及历史信息的
上涨/下跌/横盘划分法，量化：
  - 与回顾主波段标签的一致率
  - 切换点检出（命中/延迟）
  - 各类别召回

回顾真值（仅评估用，含前瞻）：大幅摆动峰谷切上涨/下跌，其余为横盘。
候选方法均为因果。
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.lib.symbol_trend_board import load_cluster_mapping_codes
from etf_daily.paths import CLUSTER_MAPPING_SELECTED_TXT

LABS = ("up", "down", "range")
LAB_TO_I = {k: i for i, k in enumerate(LABS)}


def _load_plot_mod():
    path = _PROJECT_ROOT / "decision_pack" / "scripts" / "plot_regime_transition_example.py"
    spec = importlib.util.spec_from_file_location("plot_causal_regime", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rolling_hist_percentile(
    s: pd.Series, *, window: int = 252, min_periods: int = 60
) -> pd.Series:
    """Causal rank of s[t] within s[t-window+1:t] (inclusive), in [0, 1]."""
    arr = pd.to_numeric(s, errors="coerce").to_numpy(float)
    out = np.full(len(arr), np.nan, dtype=float)
    for i in range(len(arr)):
        v = arr[i]
        if not np.isfinite(v):
            continue
        lo = max(0, i - window + 1)
        w = arr[lo : i + 1]
        w = w[np.isfinite(w)]
        if len(w) < min_periods:
            continue
        out[i] = float(np.mean(w <= v))
    return pd.Series(out, index=s.index, dtype=float)


def build_px(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.copy()
    df["as_of"] = pd.to_datetime(df["datetime"]).dt.normalize()
    df = df.sort_values("as_of")
    df = df[~df["as_of"].duplicated(keep="last")].reset_index(drop=True)
    for c in ("$open", "$high", "$low", "$close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    close = df["$close"]
    for w in (5, 10, 20, 60):
        df[f"MA{w}"] = close.rolling(w, min_periods=max(3, w // 2)).mean()
    df["prior20"] = close / close.shift(20) - 1.0
    df["prior60"] = close / close.shift(60) - 1.0
    df["dist_ma20"] = close / df["MA20"] - 1.0
    df["ma20_ma60"] = df["MA20"] / df["MA60"] - 1.0
    df["roll_hi60"] = df["$high"].rolling(60, min_periods=20).max()
    df["roll_lo60"] = df["$low"].rolling(60, min_periods=20).min()
    df["band60"] = (df["roll_hi60"] - df["roll_lo60"]) / close
    # causal ATR-like vol
    tr = pd.concat(
        [
            (df["$high"] - df["$low"]),
            (df["$high"] - close.shift(1)).abs(),
            (df["$low"] - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr20"] = tr.rolling(20, min_periods=10).mean() / close
    # Realized vol + causal historical percentile (range ≈ prolonged low pctile).
    df["rv20"] = close.pct_change().rolling(20, min_periods=10).std() * np.sqrt(252.0)
    df["rv20_pct_252"] = _rolling_hist_percentile(df["rv20"], window=252, min_periods=60)
    # ADX proxy: DI+/DI- smoothed (simplified)
    up_move = df["$high"].diff()
    down_move = -df["$low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = tr.rolling(14, min_periods=7).mean().replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14, min_periods=7).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14, min_periods=7).mean() / atr
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    df["adx14"] = dx.rolling(14, min_periods=7).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    return df


def retrospective_truth(
    close: np.ndarray,
    *,
    prom: float = 0.18,
    mindist: int = 25,
    up_ret: float = 0.22,
    down_ret: float = -0.22,
    min_seg: int = 30,
) -> np.ndarray:
    """Look-ahead major swing labels: up/down/range. Evaluation only."""
    n = len(close)
    if n < 60:
        return np.array(["range"] * n, dtype=object)
    smooth = pd.Series(close).rolling(8, min_periods=1).mean().to_numpy()
    pivots: list[tuple[int, str, float]] = []
    last_i, last_px, last_t = 0, float(smooth[0]), None
    for i in range(1, n - 1):
        is_h = smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]
        is_l = smooth[i] <= smooth[i - 1] and smooth[i] <= smooth[i + 1]
        if not (is_h or is_l):
            continue
        typ = "H" if is_h else "L"
        if last_t is None:
            last_t, last_i, last_px = typ, i, float(smooth[i])
            continue
        if typ == last_t:
            if (typ == "H" and smooth[i] >= last_px) or (typ == "L" and smooth[i] <= last_px):
                last_i, last_px = i, float(smooth[i])
            continue
        move = abs(smooth[i] - last_px) / max(last_px, 1e-8)
        if move >= prom and (i - last_i) >= mindist:
            pivots.append((last_i, last_t, float(close[last_i])))
            last_t, last_i, last_px = typ, i, float(smooth[i])
    pivots.append((last_i, last_t or "L", float(close[last_i])))
    if pivots[0][0] > 0:
        pivots.insert(0, (0, "L" if pivots[0][1] == "H" else "H", float(close[0])))
    if pivots[-1][0] < n - 1:
        pivots.append((n - 1, "H" if pivots[-1][1] == "L" else "L", float(close[-1])))

    legs: list[list[Any]] = []
    for a, b in zip(pivots[:-1], pivots[1:]):
        i0, t0, p0 = a
        i1, t1, p1 = b
        if i1 <= i0:
            continue
        ret = p1 / p0 - 1.0
        if t0 == "L" and t1 == "H" and ret >= up_ret:
            lab = "up"
        elif t0 == "H" and t1 == "L" and ret <= down_ret:
            lab = "down"
        else:
            lab = "range"
        legs.append([i0, i1, lab])

    # merge adjacent same
    merged: list[list[Any]] = []
    for leg in legs:
        if merged and merged[-1][2] == leg[2]:
            merged[-1][1] = leg[1]
        else:
            merged.append(list(leg))

    # merge tiny into previous
    out_legs: list[list[Any]] = []
    for i0, i1, lab in merged:
        nb = i1 - i0 + 1
        if out_legs and nb < min_seg:
            out_legs[-1][1] = i1
        else:
            out_legs.append([i0, i1, lab])
    # re-merge same after tiny merge
    merged2: list[list[Any]] = []
    for leg in out_legs:
        if merged2 and merged2[-1][2] == leg[2]:
            merged2[-1][1] = leg[1]
        else:
            merged2.append(list(leg))

    truth = np.array(["range"] * n, dtype=object)
    for i0, i1, lab in merged2:
        truth[i0 : i1 + 1] = lab
    return truth


def retrospective_truth_full_slice(
    close_full: np.ndarray,
    *,
    start_idx: int = 0,
    end_idx: int | None = None,
    prom: float = 0.18,
    mindist: int = 25,
    up_ret: float = 0.22,
    down_ret: float = -0.22,
    min_seg: int = 30,
) -> np.ndarray:
    """Retrospective truth on full history, then slice [start_idx, end_idx].

    Avoids short-window artifact where ~87-bar OOS alone becomes ~75% range.
    Still look-ahead within the full series — evaluation / method-selection only.
    """
    truth_full = retrospective_truth(
        close_full,
        prom=prom,
        mindist=mindist,
        up_ret=up_ret,
        down_ret=down_ret,
        min_seg=min_seg,
    )
    if end_idx is None:
        end_idx = len(truth_full) - 1
    start_idx = max(0, int(start_idx))
    end_idx = min(len(truth_full) - 1, int(end_idx))
    if end_idx < start_idx:
        return np.array([], dtype=object)
    return truth_full[start_idx : end_idx + 1].copy()


def causal_confirmed_swing_truth(
    close: np.ndarray,
    *,
    prom: float = 0.12,
    mindist: int = 15,
    up_ret: float = 0.12,
    down_ret: float = -0.12,
    min_seg: int = 10,
) -> np.ndarray:
    """Causal delayed swing labels: only past bars; unconfirmed legs stay range.

    Pivot candidates are local extrema on causal rolling mean; a pivot is confirmed
    only after ``mindist`` subsequent bars. When an opposite pivot confirms, the
    completed leg is labeled (up/down/range by return thresholds).
    """
    n = len(close)
    if n < 40:
        return np.array(["range"] * n, dtype=object)
    # causal smooth: trailing mean only
    smooth = pd.Series(close).rolling(8, min_periods=1).mean().to_numpy(float)
    truth = np.array(["range"] * n, dtype=object)

    # confirmed pivots: (idx, typ H/L, px)
    confirmed: list[tuple[int, str, float]] = []
    # pending candidate
    cand_i: int | None = None
    cand_t: str | None = None
    cand_px: float | None = None

    def _try_confirm(i: int) -> None:
        nonlocal cand_i, cand_t, cand_px
        if cand_i is None or cand_t is None or cand_px is None:
            return
        if i - cand_i < mindist:
            return
        # confirm candidate
        if confirmed and confirmed[-1][1] == cand_t:
            # same type: keep more extreme
            prev_i, prev_t, prev_px = confirmed[-1]
            if (cand_t == "H" and cand_px >= prev_px) or (cand_t == "L" and cand_px <= prev_px):
                confirmed[-1] = (cand_i, cand_t, float(close[cand_i]))
        else:
            if confirmed:
                i0, t0, p0 = confirmed[-1]
                i1, t1, p1 = cand_i, cand_t, float(close[cand_i])
                if i1 > i0:
                    ret = p1 / max(p0, 1e-8) - 1.0
                    if t0 == "L" and t1 == "H" and ret >= up_ret:
                        lab = "up"
                    elif t0 == "H" and t1 == "L" and ret <= down_ret:
                        lab = "down"
                    else:
                        lab = "range"
                    # only label if segment long enough, else keep range
                    if (i1 - i0 + 1) >= min_seg or lab == "range":
                        truth[i0 : i1 + 1] = lab
                    else:
                        truth[i0 : i1 + 1] = "range"
            confirmed.append((cand_i, cand_t, float(close[cand_i])))
        cand_i = cand_t = cand_px = None

    for i in range(1, n):
        # detect extrema using only past+today (compare i-1 with neighbors in past)
        j = i - 1
        if j >= 1:
            is_h = smooth[j] >= smooth[j - 1] and smooth[j] >= smooth[i]
            is_l = smooth[j] <= smooth[j - 1] and smooth[j] <= smooth[i]
            if is_h or is_l:
                typ = "H" if is_h else "L"
                pxj = float(smooth[j])
                move_ok = True
                if cand_i is not None and cand_t is not None and cand_px is not None:
                    if typ == cand_t:
                        if (typ == "H" and pxj >= cand_px) or (typ == "L" and pxj <= cand_px):
                            cand_i, cand_px = j, pxj
                        move_ok = False
                    else:
                        move = abs(pxj - cand_px) / max(cand_px, 1e-8)
                        if move < prom or (j - cand_i) < mindist:
                            move_ok = False
                        else:
                            # confirm previous candidate first if aged
                            _try_confirm(i)
                if move_ok and (cand_i is None or typ != cand_t):
                    # start / replace candidate of opposite type after enough move
                    if cand_i is not None and cand_t is not None and cand_px is not None and typ != cand_t:
                        move = abs(pxj - cand_px) / max(cand_px, 1e-8)
                        if move >= prom and (j - cand_i) >= mindist:
                            _try_confirm(i)
                            cand_i, cand_t, cand_px = j, typ, pxj
                        elif typ == "H" and (cand_t != "H" or pxj >= (cand_px or -1e9)):
                            pass
                    if cand_i is None:
                        cand_i, cand_t, cand_px = j, typ, pxj
                elif cand_i is None:
                    cand_i, cand_t, cand_px = j, typ, pxj
        _try_confirm(i)

    # final confirm at end if aged
    if n > 0:
        _try_confirm(n - 1 + mindist)
    return truth


def merge_short_segments(labels: np.ndarray, *, min_bars: int = 10) -> np.ndarray:
    """Merge trend segments shorter than min_bars into neighboring labels (prefer range)."""
    labs = np.asarray(labels, dtype=object).copy()
    n = len(labs)
    if n == 0 or min_bars <= 1:
        return labs
    # Two passes: first absorb short trends; second cleans residuals after merges.
    for _ in range(2):
        segs = segment_bounds(labs)
        out = labs.copy()
        changed = False
        for s, e, lab in segs:
            nb = e - s + 1
            if lab == "range" or nb >= min_bars:
                continue
            if s > 0:
                fill = out[s - 1]
            elif e + 1 < n:
                fill = out[e + 1]
            else:
                fill = "range"
            out[s : e + 1] = fill
            changed = True
        labs = out
        if not changed:
            break
    return np.asarray(labs, dtype=object)


def apply_lowvol_range_bias(
    labels: np.ndarray,
    vol_pct: np.ndarray,
    prior60: np.ndarray,
    prior20: np.ndarray,
    *,
    low_pct: float = 0.35,
    look: int = 15,
    persist: int = 10,
    escape_p60: float = 0.12,
    escape_p20: float = 0.08,
) -> np.ndarray:
    """Force range when realized-vol stays in a low historical percentile.

    Sideways episodes are typically prolonged low ``rv20`` pctile; MA-stack alone
    still flickers into short up/down there. Strong momentum can escape.
    """
    labs = np.asarray(labels, dtype=object).copy()
    n = len(labs)
    if n == 0:
        return labs
    vp = np.asarray(vol_pct, dtype=float)
    p60 = np.asarray(prior60, dtype=float)
    p20 = np.asarray(prior20, dtype=float)
    for i in range(n):
        lo = max(0, i - look + 1)
        window = vp[lo : i + 1]
        n_low = int(np.sum(np.isfinite(window) & (window < low_pct)))
        if n_low < persist:
            continue
        a60 = abs(p60[i]) if np.isfinite(p60[i]) else 0.0
        a20 = abs(p20[i]) if np.isfinite(p20[i]) else 0.0
        if a60 >= escape_p60 or a20 >= escape_p20:
            continue
        labs[i] = "range"
    return labs


def finalize_regime_labels(labels: np.ndarray, px: pd.DataFrame, *, min_seg: int = 10) -> np.ndarray:
    """Shared post-process: low-vol → range, then squash short directional blips."""
    labs = np.asarray(labels, dtype=object)
    if "rv20_pct_252" in px.columns:
        labs = apply_lowvol_range_bias(
            labs,
            px["rv20_pct_252"].to_numpy(float),
            px["prior60"].to_numpy(float),
            px["prior20"].to_numpy(float),
        )
    ms = int(min_seg or 0)
    if ms > 1:
        labs = merge_short_segments(labs, min_bars=ms)
    return np.asarray(labs, dtype=object)


def causal_smooth(labs: list[str], w: int = 5) -> list[str]:
    out = []
    for i in range(len(labs)):
        sl = labs[max(0, i - w + 1) : i + 1]
        vals, cnts = np.unique(sl, return_counts=True)
        out.append(str(vals[np.argmax(cnts)]))
    return out


def apply_hysteresis(
    raw: list[str],
    *,
    above20: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    close: np.ndarray | None = None,
    confirm: int = 2,
    soft_leave_above20: int = 2,
    soft_leave_bounce: float = 0.04,
    soft_leave_below20: int = 2,
    soft_leave_drawdown: float = 0.06,
    reenter_down_guard: int = 3,
) -> list[str]:
    """Leave trend on MA20 break; soft-leave on V-reversal / pullback.

    Soft leave (down → range) when still bear-stacked but price has clearly
    reversed — avoids sticky-down on sharp recoveries (e.g. SH516310 after
    2026-06-30):
      - ``soft_leave_above20`` consecutive closes above MA20, or
      - close rebound from trough-since-enter-down ≥ ``soft_leave_bounce``.

    After soft-leave-down, ``reenter_down_guard`` bars of C<MA20 are required
    before bear-stack alone can push range→down again (SH513080 May–Jun 2026:
    soft-leave then immediate re-entry wiped the bounce).

    Soft leave (up → range) when still bull-stacked but short trend broken
    (SH513080 Jun–Jul 2026 false up while drifting under MA20):
      - ``soft_leave_below20`` consecutive closes below MA20, or
      - drawdown from peak-since-enter-up ≥ ``soft_leave_drawdown``.
    """
    state = "range"
    below = above = 0
    trough = np.nan
    peak = np.nan
    soft_guard = 0
    if close is None:
        close = np.full(len(raw), np.nan)
    out = []
    for i, lab in enumerate(raw):
        if above20[i]:
            above += 1
            below = 0
        else:
            below += 1
            above = 0
        m20, m60 = ma20[i], ma60[i]
        c = float(close[i]) if np.isfinite(close[i]) else np.nan
        bull_stack = np.isfinite(m20) and np.isfinite(m60) and m20 > m60
        bear_stack = np.isfinite(m20) and np.isfinite(m60) and m20 < m60
        if state == "up":
            if np.isfinite(c):
                peak = c if (not np.isfinite(peak) or c > peak) else peak
            dd = (
                (c / peak - 1.0)
                if (np.isfinite(c) and np.isfinite(peak) and peak > 0)
                else 0.0
            )
            soft_up = (soft_leave_below20 > 0 and below >= soft_leave_below20) or (
                soft_leave_drawdown > 0 and dd <= -soft_leave_drawdown
            )
            if lab == "down" and (below >= confirm or bear_stack):
                state = "down"
                trough = c if np.isfinite(c) else np.nan
                peak = np.nan
                soft_guard = 0
            elif soft_up or (lab == "range" and below >= confirm + 1):
                state = "range"
                trough = np.nan
                peak = np.nan
        elif state == "down":
            if np.isfinite(c):
                trough = c if (not np.isfinite(trough) or c < trough) else trough
            bounce = (
                (c / trough - 1.0)
                if (np.isfinite(c) and np.isfinite(trough) and trough > 0)
                else 0.0
            )
            soft_leave = (
                soft_leave_above20 > 0 and above >= soft_leave_above20
            ) or (soft_leave_bounce > 0 and bounce >= soft_leave_bounce)
            if lab == "up" and (above >= confirm or bull_stack):
                state = "up"
                trough = np.nan
                peak = c if np.isfinite(c) else np.nan
                soft_guard = 0
            elif soft_leave or (lab == "range" and above >= confirm):
                state = "range"
                soft_guard = max(reenter_down_guard, 0)
                # keep trough: only a new low can resume down after soft leave
                peak = np.nan
        else:
            if soft_guard > 0:
                soft_guard -= 1
            if lab == "up" and (above >= confirm or bull_stack):
                state = "up"
                trough = np.nan
                peak = c if np.isfinite(c) else np.nan
                soft_guard = 0
            elif lab == "down" and (below >= confirm or bear_stack):
                # After V soft-leave, ignore bare bear-stack while price holds
                # above the down-trough (SH513080 Jun-2026 higher-low bounce).
                higher_low = (
                    np.isfinite(trough)
                    and np.isfinite(c)
                    and c > trough
                )
                if higher_low:
                    pass
                elif soft_guard > 0 and below < max(confirm, reenter_down_guard):
                    pass
                else:
                    state = "down"
                    trough = c if np.isfinite(c) else np.nan
                    peak = np.nan
                    soft_guard = 0
        out.append(state)
    return out


def apply_hysteresis_ma60(
    raw: list[str],
    *,
    close: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    confirm_leave: int = 3,
    confirm_enter: int = 2,
    soft_leave_below20: int = 3,
    soft_leave_drawdown: float = 0.12,
    soft_leave_bounce: float = 0.04,
) -> list[str]:
    """Leave up/down primarily on MA20/MA60 stack flips.

    Soft leave (up → range) also triggers when still bull-stacked but price has
    clearly broken the short trend — avoids sticky-up on deep pullbacks
    (e.g. SZ159502 after 2026-07-10):
      - ``soft_leave_below20`` consecutive closes below MA20, or
      - close drawdown from peak-since-enter-up ≥ ``soft_leave_drawdown``.

    Soft leave (down → range) on V-reversals:
      - ``soft_leave_below20`` consecutive closes above MA20, or
      - rebound from trough-since-enter-down ≥ ``soft_leave_bounce``.

    Re-entry into up from range additionally requires ``confirm_enter`` closes
    back above MA20 (bull stack alone is not enough after a soft leave).
    """
    state = "range"
    bull_n = bear_n = 0
    below20_n = above20_n = 0
    peak = np.nan
    trough = np.nan
    out: list[str] = []
    for i, lab in enumerate(raw):
        c, m20, m60 = close[i], ma20[i], ma60[i]
        ok = np.isfinite(m20) and np.isfinite(m60)
        if ok and m20 > m60:
            bull_n += 1
            bear_n = 0
        elif ok and m20 < m60:
            bear_n += 1
            bull_n = 0
        else:
            bull_n = bear_n = 0

        above20 = np.isfinite(c) and np.isfinite(m20) and c > m20
        if above20:
            above20_n += 1
            below20_n = 0
        else:
            below20_n += 1
            above20_n = 0

        if state == "up":
            if np.isfinite(c):
                peak = c if (not np.isfinite(peak) or c > peak) else peak
            dd = (c / peak - 1.0) if (np.isfinite(c) and np.isfinite(peak) and peak > 0) else 0.0
            soft_leave = (soft_leave_below20 > 0 and below20_n >= soft_leave_below20) or (
                soft_leave_drawdown > 0 and dd <= -soft_leave_drawdown
            )
            if bear_n >= confirm_leave:
                state = "down"
                peak = np.nan
                trough = c if np.isfinite(c) else np.nan
            elif soft_leave or (lab == "range" and bear_n >= confirm_enter):
                state = "range"
                peak = np.nan
                trough = np.nan
        elif state == "down":
            if np.isfinite(c):
                trough = c if (not np.isfinite(trough) or c < trough) else trough
            bounce = (
                (c / trough - 1.0)
                if (np.isfinite(c) and np.isfinite(trough) and trough > 0)
                else 0.0
            )
            soft_leave_down = (
                soft_leave_below20 > 0 and above20_n >= soft_leave_below20
            ) or (soft_leave_bounce > 0 and bounce >= soft_leave_bounce)
            if bull_n >= confirm_leave and above20_n >= confirm_enter:
                state = "up"
                peak = c if np.isfinite(c) else np.nan
                trough = np.nan
            elif soft_leave_down or (lab == "range" and bull_n >= confirm_enter):
                state = "range"
                peak = np.nan
                trough = np.nan
        else:
            # range: need price back above MA20, not just MA20>MA60 structure
            if lab == "up" and bull_n >= confirm_enter and above20_n >= confirm_enter:
                state = "up"
                peak = c if np.isfinite(c) else np.nan
                trough = np.nan
            elif lab == "down" and bear_n >= confirm_enter:
                state = "down"
                peak = np.nan
                trough = c if np.isfinite(c) else np.nan
        out.append(state)
    return out


def _ma_stack_raw_labels(px: pd.DataFrame, *, soft_hold60: bool) -> list[str]:
    """Raw MA-stack labels. soft_hold60: pullbacks between MA20/MA60 stay with structure.

    Range-first (conservative): wider chop/hug bands and no bare C≥MA20→up fallback,
    so low-vol chop stays range instead of flickering into short up/down legs.
    """
    raw: list[str] = []
    for _, r in px.iterrows():
        if pd.isna(r["MA20"]) or pd.isna(r["MA60"]):
            raw.append("range")
            continue
        c, m20, m60 = float(r["$close"]), float(r["MA20"]), float(r["MA60"])
        p20 = float(r["prior20"]) if pd.notna(r["prior20"]) else 0.0
        p60 = float(r["prior60"]) if pd.notna(r["prior60"]) else 0.0
        band = float(r["band60"]) if pd.notna(r["band60"]) else np.nan
        hug = abs(m20 - m60) / c < 0.02
        # Wider chop than before (0.18/0.08): low-vol ETFs default to range.
        chop = (pd.notna(band) and band < 0.22 and abs(p60) < 0.10) or (
            hug and abs(p60) < 0.12
        )
        if soft_hold60:
            # Need stack + above MA20 (not just MA60) and some momentum.
            up = (
                m20 > m60
                and c > m20
                and c >= m60
                and (p60 >= 0.02 or p20 >= 0.02)
            )
            down = (
                m20 < m60
                and c < m20
                and c <= m60
                and (p60 <= -0.02 or p20 <= -0.02)
            )
        else:
            up = (m20 > m60 and c > m20 and p60 >= 0.03) or (
                m20 > m60 and c > m20 and p20 >= 0.03
            )
            # Gated V-reversal (same gate as strict): washout or thrust, not mild chop.
            if c > m20 and p20 >= 0.06 and (
                m20 > m60 or p60 <= -0.08 or p60 >= 0.08
            ):
                up = True
            down = (m20 < m60 and c < m20 and p60 <= -0.03) or (
                m20 < m60 and c < m20 and p20 <= -0.03
            )
        # Directional stack wins over chop. No bare C vs MA20 fallback — that
        # painted low-vol bounces as up/down (France/HK chop).
        if up and not down:
            lab = "up"
        elif down and not up:
            lab = "down"
        elif chop:
            lab = "range"
        else:
            lab = "range"
        raw.append(lab)
    return raw


def method_ma_stack_params(
    px: pd.DataFrame,
    *,
    confirm: int = 3,
    smooth: int = 3,
) -> np.ndarray:
    """MA stack + hysteresis; range-first enter (default confirm=3)."""
    raw = _ma_stack_raw_labels(px, soft_hold60=False)
    above20 = (px["$close"] > px["MA20"]).fillna(False).to_numpy(bool)
    labs = apply_hysteresis(
        raw,
        above20=above20,
        ma20=px["MA20"].to_numpy(float),
        ma60=px["MA60"].to_numpy(float),
        close=px["$close"].to_numpy(float),
        confirm=int(confirm),
    )
    return np.array(causal_smooth(labs, max(1, int(smooth))), dtype=object)


def method_ma_stack(px: pd.DataFrame) -> np.ndarray:
    """MA stack + hysteresis; range-first enter (confirm=3)."""
    return method_ma_stack_params(px)


def method_ma_stack_struct_params(
    px: pd.DataFrame,
    *,
    confirm_enter: int = 3,
    confirm_leave: int = 3,
    smooth: int = 3,
) -> np.ndarray:
    """MA stack + prior; pullbacks above MA60 stay up (parametrized confirms)."""
    raw = _ma_stack_raw_labels(px, soft_hold60=True)
    labs = apply_hysteresis_ma60(
        raw,
        close=px["$close"].to_numpy(float),
        ma20=px["MA20"].to_numpy(float),
        ma60=px["MA60"].to_numpy(float),
        confirm_leave=int(confirm_leave),
        confirm_enter=int(confirm_enter),
        soft_leave_below20=3,
        soft_leave_drawdown=0.12,
    )
    return np.array(causal_smooth(labs, max(1, int(smooth))), dtype=object)


def method_ma_stack_struct(px: pd.DataFrame) -> np.ndarray:
    """MA stack + prior; pullbacks above MA60 stay up.

    Leave up on MA20/MA60 flip **or** soft break: 3d C<MA20 / ≥12% peak drawdown.
    Enter up needs 3 confirms (range-first).
    """
    return method_ma_stack_struct_params(px)


# Washout V-rev only if MA20 is already near MA60. A crash that itself
# pushes prior60 through -8% while MA20 is still far below MA60 is not a V
# (SH588850 2026-08-19: -9% red day, then a losing "up" 08-20..09-02).
VREV_WASHOUT_STACK_GAP = -0.05


def method_ma_stack_strict_params(
    px: pd.DataFrame,
    *,
    p60_thr: float = 0.06,
    band_chop: float = 0.26,
    p60_chop: float = 0.12,
    p20_vrev: float = 0.06,
    confirm: int = 3,
    smooth: int = 3,
) -> np.ndarray:
    """Stricter momentum gates; ``p20_vrev`` / ``confirm`` tunable per symbol."""
    raw = []
    p20_vrev = float(p20_vrev)
    close_arr = pd.to_numeric(px["$close"], errors="coerce").to_numpy(float)
    for i, (_, r) in enumerate(px.iterrows()):
        if pd.isna(r["MA20"]) or pd.isna(r["MA60"]):
            raw.append("range")
            continue
        c, m20, m60 = float(close_arr[i]), float(r["MA20"]), float(r["MA60"])
        p20 = float(r["prior20"]) if pd.notna(r["prior20"]) else 0.0
        p60 = float(r["prior60"]) if pd.notna(r["prior60"]) else 0.0
        band = float(r["band60"]) if pd.notna(r["band60"]) else np.nan
        chop = pd.notna(band) and band < band_chop and abs(p60) < p60_chop
        prev_c = close_arr[i - 1] if i > 0 else np.nan
        recovering = (not np.isfinite(prev_c)) or (c >= prev_c)
        near_stack = m60 > 0 and (m20 / m60 - 1.0) > VREV_WASHOUT_STACK_GAP
        washout = p60 <= -0.08 and near_stack and recovering
        up = (m20 > m60 and c > m20 and p60 >= p60_thr and p20 >= 0.0) or (
            # V-reversal: prior20 above MA20; restored stack / near-stack
            # washout / thrust. Far-below-MA60 crash days are not a V.
            c > m20
            and p20 >= p20_vrev
            and (m20 > m60 or washout or p60 >= 0.08)
        ) or (
            m20 > m60 and c > m20 and p60 >= max(0.18, p60_thr * 3) and p20 >= -0.08
        )
        down = m20 < m60 and c < m20 and p60 <= -p60_thr and p20 <= 0.0
        if up:
            lab = "up"
        elif down:
            lab = "down"
        elif chop:
            lab = "range"
        else:
            lab = "range"
        raw.append(lab)
    above20 = (px["$close"] > px["MA20"]).fillna(False).to_numpy(bool)
    labs = apply_hysteresis(
        raw,
        above20=above20,
        ma20=px["MA20"].to_numpy(float),
        ma60=px["MA60"].to_numpy(float),
        close=px["$close"].to_numpy(float),
        confirm=int(confirm),
    )
    return np.array(causal_smooth(labs, max(1, int(smooth))), dtype=object)


def method_ma_stack_strict(px: pd.DataFrame) -> np.ndarray:
    """Stricter momentum gates; range-first for low-vol.

    Classic up needs clearer prior60; V-reversal recovery via strong prior20
    above MA20 (stack restored, or washout only if MA20 is already near MA60
    on a non-down bar); strong intermediate trend (p60≥18%) may enter while
    prior20 is still mildly negative (SH513310 Apr-2026 recovery).
    """
    return method_ma_stack_strict_params(px)


def method_dual_ma_cross_params(
    px: pd.DataFrame,
    *,
    confirm_enter: int = 3,
    confirm_leave: int = 3,
    smooth: int = 3,
) -> np.ndarray:
    """MA20/MA60 regime; range-first hug + stronger momentum escape."""
    raw: list[str] = []
    for _, r in px.iterrows():
        if pd.isna(r["MA20"]) or pd.isna(r["MA60"]):
            raw.append("range")
            continue
        c, m20, m60 = float(r["$close"]), float(r["MA20"]), float(r["MA60"])
        p20 = float(r["prior20"]) if pd.notna(r["prior20"]) else 0.0
        p60 = float(r["prior60"]) if pd.notna(r["prior60"]) else 0.0
        hug = abs(m20 - m60) / c < 0.025
        strong_up = c > m20 and (p20 >= 0.04 or p60 >= 0.06)
        strong_dn = c < m20 and (p20 <= -0.04 or p60 <= -0.06)
        if hug and not strong_up and not strong_dn:
            lab = "range"
        elif (m20 > m60 and c > m20 and c >= m60) or strong_up:
            lab = "up"
        elif (m20 < m60 and c < m20 and c <= m60) or strong_dn:
            lab = "down"
        else:
            lab = "range"
        raw.append(lab)
    labs = apply_hysteresis_ma60(
        raw,
        close=px["$close"].to_numpy(float),
        ma20=px["MA20"].to_numpy(float),
        ma60=px["MA60"].to_numpy(float),
        confirm_leave=int(confirm_leave),
        confirm_enter=int(confirm_enter),
    )
    return np.array(causal_smooth(labs, max(1, int(smooth))), dtype=object)


def method_dual_ma_cross(px: pd.DataFrame) -> np.ndarray:
    """MA20/MA60 regime; range-first hug + stronger momentum escape."""
    return method_dual_ma_cross_params(px)


def method_adx_di(px: pd.DataFrame) -> np.ndarray:
    """ADX trend filter + DI direction; MA+prior escape when ADX is weak.

    Low-ADX grinds (e.g. SZ159502 May–Jun 2026) were painted all-range while
    price + MA stack were clearly trending. Escape the ADX<18 range gate when
    stack+prior confirm; smooth 3.
    """
    raw = []
    for _, r in px.iterrows():
        adx = float(r["adx14"]) if pd.notna(r["adx14"]) else 0.0
        pdi = float(r["plus_di"]) if pd.notna(r["plus_di"]) else 0.0
        mdi = float(r["minus_di"]) if pd.notna(r["minus_di"]) else 0.0
        c = float(r["$close"])
        m20 = float(r["MA20"]) if pd.notna(r["MA20"]) else np.nan
        m60 = float(r["MA60"]) if pd.notna(r["MA60"]) else np.nan
        p20 = float(r["prior20"]) if pd.notna(r["prior20"]) else 0.0
        p60 = float(r["prior60"]) if pd.notna(r["prior60"]) else 0.0
        if adx < 18:
            if (
                np.isfinite(m20)
                and np.isfinite(m60)
                and m20 > m60
                and c >= m20
                and (p20 >= 0.02 or p60 >= 0.03)
            ):
                lab = "up"
            elif (
                np.isfinite(m20)
                and np.isfinite(m60)
                and m20 < m60
                and c <= m20
                and (p20 <= -0.02 or p60 <= -0.03)
            ):
                lab = "down"
            else:
                lab = "range"
        elif pdi > mdi:
            lab = "up"
        elif mdi > pdi:
            lab = "down"
        else:
            lab = "range"
        raw.append(lab)
    above20 = (px["$close"] > px["MA20"]).fillna(False).to_numpy(bool)
    labs = apply_hysteresis(
        raw,
        above20=above20,
        ma20=px["MA20"].to_numpy(float),
        ma60=px["MA60"].to_numpy(float),
        close=px["$close"].to_numpy(float),
        confirm=2,
    )
    return np.array(causal_smooth(labs, 3), dtype=object)


def method_prior60_band(px: pd.DataFrame) -> np.ndarray:
    """prior60 direction + band60 chop (range-first: wider chop, higher up gate)."""
    raw = []
    for _, r in px.iterrows():
        p60 = float(r["prior60"]) if pd.notna(r["prior60"]) else 0.0
        band = float(r["band60"]) if pd.notna(r["band60"]) else np.nan
        c_above = bool(r["$close"] > r["MA20"]) if pd.notna(r["MA20"]) else False
        if pd.notna(band) and band < 0.22 and abs(p60) < 0.10:
            lab = "range"
        elif p60 >= 0.10 and c_above:
            lab = "up"
        elif p60 <= -0.10 and not c_above:
            lab = "down"
        elif p60 >= 0.05 and c_above:
            lab = "up"
        elif p60 <= -0.05 and not c_above:
            lab = "down"
        else:
            lab = "range"
        raw.append(lab)
    return np.array(causal_smooth(raw, 5), dtype=object)


def method_hybrid_v1_params(
    px: pd.DataFrame,
    *,
    confirm: int = 3,
    smooth: int = 3,
) -> np.ndarray:
    """MA stack + ADX confirm with optional enter confirm / smooth."""
    base = method_ma_stack_params(px, confirm=confirm, smooth=smooth)
    out = []
    for i, lab in enumerate(base):
        adx = float(px.iloc[i]["adx14"]) if pd.notna(px.iloc[i]["adx14"]) else 0.0
        p20 = float(px.iloc[i]["prior20"]) if pd.notna(px.iloc[i]["prior20"]) else 0.0
        p60 = float(px.iloc[i]["prior60"]) if pd.notna(px.iloc[i]["prior60"]) else 0.0
        c = float(px.iloc[i]["$close"])
        m20 = float(px.iloc[i]["MA20"]) if pd.notna(px.iloc[i]["MA20"]) else np.nan
        mom = abs(p60) >= 0.06 or abs(p20) >= 0.04
        above = np.isfinite(m20) and c > m20
        if lab in ("up", "down") and adx < 15 and not mom and not (
            lab == "up" and above and p20 >= 0.02
        ):
            out.append("range")
        else:
            out.append(lab)
    return np.asarray(out, dtype=object)


def method_hybrid_v1(px: pd.DataFrame) -> np.ndarray:
    """MA stack + ADX confirm: weaken weak-ADX trends unless momentum is clear.

    Do **not** re-smooth: ``method_ma_stack`` already smooths; a second
    ``causal_smooth(5)`` delayed July-2026 ups on SZ159985 until the spike top
    and held soft-leave through the crash.
    """
    return method_hybrid_v1_params(px)


METHODS = {
    "ma_stack_hyst": method_ma_stack,  # leave on 2d C<MA20
    "ma_stack_struct": method_ma_stack_struct,  # leave on MA20/MA60 flip (less bull chop)
    "ma_stack_strict": method_ma_stack_strict,
    "dual_ma_cross": method_dual_ma_cross,
    "adx_di": method_adx_di,
    "prior60_band": method_prior60_band,
    "hybrid_ma_adx": method_hybrid_v1,
}


def method_adx_di_params(px: pd.DataFrame, *, adx_thr: float = 18, smooth: int = 5) -> np.ndarray:
    raw = []
    for _, r in px.iterrows():
        adx = float(r["adx14"]) if pd.notna(r["adx14"]) else 0.0
        pdi = float(r["plus_di"]) if pd.notna(r["plus_di"]) else 0.0
        mdi = float(r["minus_di"]) if pd.notna(r["minus_di"]) else 0.0
        if adx < adx_thr:
            lab = "range"
        elif pdi > mdi:
            lab = "up"
        elif mdi > pdi:
            lab = "down"
        else:
            lab = "range"
        raw.append(lab)
    above20 = (px["$close"] > px["MA20"]).fillna(False).to_numpy(bool)
    labs = apply_hysteresis(
        raw,
        above20=above20,
        ma20=px["MA20"].to_numpy(float),
        ma60=px["MA60"].to_numpy(float),
        close=px["$close"].to_numpy(float),
        confirm=2,
    )
    return np.array(causal_smooth(labs, max(1, int(smooth))), dtype=object)


def run_method(
    name: str,
    px: pd.DataFrame,
    *,
    params: dict[str, Any] | None = None,
    min_seg: int = 10,
) -> np.ndarray:
    """Run a named METHOD optionally with per-symbol params + low-vol/short-seg finalize."""
    params = dict(params or {})
    # Keys that change the label generator (not just finalize min_seg).
    label_keys = {
        "p60_thr",
        "band_chop",
        "p60_chop",
        "p20_vrev",
        "confirm",
        "confirm_enter",
        "confirm_leave",
        "smooth",
        "adx_thr",
    }
    use_params = bool(label_keys.intersection(params))
    if name == "adx_di" and use_params:
        labs = method_adx_di_params(
            px,
            adx_thr=float(params.get("adx_thr", 18)),
            smooth=int(params.get("smooth", 5)),
        )
    elif name == "ma_stack_strict" and use_params:
        labs = method_ma_stack_strict_params(
            px,
            p60_thr=float(params.get("p60_thr", 0.06)),
            band_chop=float(params.get("band_chop", 0.26)),
            p60_chop=float(params.get("p60_chop", 0.12)),
            p20_vrev=float(params.get("p20_vrev", 0.06)),
            confirm=int(params.get("confirm", 3)),
            smooth=int(params.get("smooth", 3)),
        )
    elif name == "ma_stack_hyst" and use_params:
        labs = method_ma_stack_params(
            px,
            confirm=int(params.get("confirm", 3)),
            smooth=int(params.get("smooth", 3)),
        )
    elif name == "ma_stack_struct" and use_params:
        conf = int(params.get("confirm", params.get("confirm_enter", 3)))
        labs = method_ma_stack_struct_params(
            px,
            confirm_enter=int(params.get("confirm_enter", conf)),
            confirm_leave=int(params.get("confirm_leave", conf)),
            smooth=int(params.get("smooth", 3)),
        )
    elif name == "dual_ma_cross" and use_params:
        conf = int(params.get("confirm", params.get("confirm_enter", 3)))
        labs = method_dual_ma_cross_params(
            px,
            confirm_enter=int(params.get("confirm_enter", conf)),
            confirm_leave=int(params.get("confirm_leave", conf)),
            smooth=int(params.get("smooth", 3)),
        )
    elif name == "hybrid_ma_adx" and use_params:
        labs = method_hybrid_v1_params(
            px,
            confirm=int(params.get("confirm", 3)),
            smooth=int(params.get("smooth", 3)),
        )
    else:
        if name not in METHODS:
            raise KeyError(name)
        labs = METHODS[name](px)
    ms = int(params.get("min_seg", min_seg) or 0)
    return finalize_regime_labels(labs, px, min_seg=ms)


def segment_bounds(labels: np.ndarray) -> list[tuple[int, int, str]]:
    segs = []
    s = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[s]:
            segs.append((s, i - 1, str(labels[s])))
            s = i
    return segs


def switch_indices(labels: np.ndarray) -> list[int]:
    return [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]


def _f1(prec: float, rec: float) -> float:
    if not (np.isfinite(prec) and np.isfinite(rec)) or (prec + rec) <= 0:
        return float("nan")
    return float(2 * prec * rec / (prec + rec))


def segment_overlap_score(truth: np.ndarray, pred: np.ndarray) -> float:
    """Mean IoU of same-label overlapping segments (0..1)."""
    t_segs = segment_bounds(truth)
    p_segs = segment_bounds(pred)
    if not t_segs or not p_segs:
        return float("nan")
    ious: list[float] = []
    for ts, te, tlab in t_segs:
        best = 0.0
        for ps, pe, plab in p_segs:
            if plab != tlab:
                continue
            inter = max(0, min(te, pe) - max(ts, ps) + 1)
            union = (te - ts + 1) + (pe - ps + 1) - inter
            if union > 0:
                best = max(best, inter / union)
        ious.append(best)
    return float(np.mean(ious)) if ious else float("nan")


def eval_labels(truth: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    n = len(truth)
    agree = float((truth == pred).mean()) if n else 0.0
    per: dict[str, float] = {}
    recalls: list[float] = []
    f1s: list[float] = []
    for lab in LABS:
        mask = truth == lab
        pred_m = pred == lab
        per[f"frac_truth_{lab}"] = float(mask.mean()) if n else float("nan")
        per[f"frac_pred_{lab}"] = float(pred_m.mean()) if n else float("nan")
        if mask.sum() == 0:
            per[f"recall_{lab}"] = np.nan
            per[f"prec_{lab}"] = (
                float(((truth == lab) & pred_m).sum() / pred_m.sum()) if pred_m.sum() else np.nan
            )
            per[f"f1_{lab}"] = np.nan
        else:
            rec = float(((pred == lab) & mask).sum() / mask.sum())
            prec = (
                float(((truth == lab) & pred_m).sum() / pred_m.sum()) if pred_m.sum() else np.nan
            )
            per[f"recall_{lab}"] = rec
            per[f"prec_{lab}"] = prec
            f1 = _f1(prec, rec)
            per[f"f1_{lab}"] = f1
            recalls.append(rec)
            if np.isfinite(f1):
                f1s.append(f1)

    # switch detection: for each truth switch, nearest pred switch within +/- W
    W = 10
    t_sw = switch_indices(truth)
    p_sw = switch_indices(pred)
    hits = 0
    delays = []
    matched_p = set()
    for t in t_sw:
        best = None
        best_d = 10**9
        for j, p in enumerate(p_sw):
            if j in matched_p:
                continue
            d = abs(p - t)
            if d <= W and d < best_d:
                best_d = d
                best = j
        if best is not None:
            hits += 1
            delays.append(p_sw[best] - t)  # + means pred late
            matched_p.add(best)
    switch_hit = hits / len(t_sw) if t_sw else np.nan
    false_sw = (len(p_sw) - hits) / max(len(p_sw), 1)

    # balanced accuracy = mean recall over classes present in truth
    bal_acc = float(np.mean(recalls)) if recalls else float("nan")
    macro_f1 = float(np.mean(f1s)) if f1s else float("nan")

    # Cohen's kappa vs chance agreement from marginals
    p_e = 0.0
    for lab in LABS:
        p_e += float((truth == lab).mean()) * float((pred == lab).mean())
    kappa = (
        float((agree - p_e) / (1.0 - p_e)) if n and (1.0 - p_e) > 1e-12 else float("nan")
    )

    # trend-day subset (truth != range)
    trend_m = truth != "range"
    if trend_m.any():
        trend_agree = float((truth[trend_m] == pred[trend_m]).mean())
        # binary F1 on major trend days: treat up/down as positive classes separately averaged
        trend_f1s = []
        for lab in ("up", "down"):
            tm = truth == lab
            if not tm.any():
                continue
            pm = pred == lab
            rec = float((pm & tm).sum() / tm.sum())
            prec = float((tm & pm).sum() / pm.sum()) if pm.any() else float("nan")
            trend_f1s.append(_f1(prec, rec))
        finite = [x for x in trend_f1s if np.isfinite(x)]
        trend_f1 = float(np.mean(finite)) if finite else float("nan")
    else:
        trend_agree = float("nan")
        trend_f1 = float("nan")

    seg_ov = segment_overlap_score(truth, pred)
    majority = max(float((truth == lab).mean()) for lab in LABS) if n else float("nan")
    agree_vs_majority = float(agree - majority) if np.isfinite(majority) else float("nan")

    return dict(
        agree=agree,
        n=n,
        n_truth_sw=len(t_sw),
        n_pred_sw=len(p_sw),
        switch_hit=switch_hit,
        switch_false=false_sw,
        switch_delay_mean=float(np.mean(delays)) if delays else np.nan,
        switch_delay_med=float(np.median(delays)) if delays else np.nan,
        balanced_acc=bal_acc,
        macro_f1=macro_f1,
        kappa=kappa,
        trend_agree=trend_agree,
        trend_f1=trend_f1,
        seg_overlap=seg_ov,
        majority_frac=majority,
        agree_vs_majority=agree_vs_majority,
        **per,
    )


def score_row(r: dict[str, Any]) -> float:
    """Composite: agreement + switch hit - false switches; prefer low |delay|."""
    agree = r.get("agree", 0) or 0
    hit = r.get("switch_hit", 0) or 0
    false = r.get("switch_false", 1) or 1
    delay = r.get("switch_delay_mean", 0)
    delay_pen = 0.0 if not np.isfinite(delay) else min(abs(delay), 10) / 20.0
    # balanced recalls
    recalls = [r.get(f"recall_{lab}", np.nan) for lab in LABS]
    rec = float(np.nanmean(recalls)) if any(np.isfinite(x) for x in recalls) else 0.0
    return 0.35 * agree + 0.25 * hit + 0.20 * rec - 0.15 * false - 0.05 * delay_pen


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--end-date", default="2026-08-05")
    p.add_argument("--start-date", default="2015-01-01", help="load floor; actual first bar per code")
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_PROJECT_ROOT
        / "runtime/decision_packs/20260720/regime_transition_validation_q90_to0720/causal_regime_switch_eval",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.out_dir if args.out_dir.is_absolute() else _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_mod = _load_plot_mod()
    plot_mod._init_qlib(None)
    codes = list(load_cluster_mapping_codes(CLUSTER_MAPPING_SELECTED_TXT))
    if args.max_codes:
        codes = codes[: args.max_codes]

    rows: list[dict[str, Any]] = []
    per_code_best: list[dict[str, Any]] = []

    for ci, code in enumerate(codes):
        try:
            ohlcv = plot_mod.load_qlib_ohlcv(
                code,
                args.start_date,
                args.end_date,
                init_qlib=False,
                lookback_calendar_days=30,
                clip_to_window=True,
            )
        except Exception as exc:
            print(f"[SKIP] {code}: {exc}")
            continue
        if ohlcv is None or len(ohlcv) < 120:
            print(f"[SKIP] {code}: too short")
            continue
        px = build_px(ohlcv)
        close = px["$close"].to_numpy(float)
        truth = retrospective_truth(close)
        first = str(pd.Timestamp(px.iloc[0]["as_of"]).date())
        last = str(pd.Timestamp(px.iloc[-1]["as_of"]).date())
        bh = float(close[-1] / close[0] - 1)

        best_m = None
        best_sc = -1e9
        for name, fn in METHODS.items():
            pred = fn(px)
            # drop warmup where MA60 nan
            valid = px["MA60"].notna().to_numpy(bool)
            if valid.sum() < 80:
                continue
            ev = eval_labels(truth[valid], pred[valid])
            row = dict(code=code, method=name, first=first, last=last, bh=bh, **ev)
            row["score"] = score_row(row)
            rows.append(row)
            if row["score"] > best_sc:
                best_sc = row["score"]
                best_m = row
        if best_m:
            per_code_best.append(best_m)
        if (ci + 1) % 5 == 0 or ci == 0:
            print(f"[{ci+1}/{len(codes)}] {code} bars={len(px)} best={best_m['method'] if best_m else None} score={best_sc:.3f}")

    sdf = pd.DataFrame(rows)
    if sdf.empty:
        print("[ERROR] no results")
        return 1
    sdf.to_csv(out_dir / "per_code_method_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_code_best).to_csv(out_dir / "per_code_best_method.csv", index=False, encoding="utf-8-sig")

    # aggregate by method
    agg = (
        sdf.groupby("method")
        .agg(
            n_codes=("code", "nunique"),
            agree=("agree", "mean"),
            switch_hit=("switch_hit", "mean"),
            switch_false=("switch_false", "mean"),
            switch_delay_mean=("switch_delay_mean", "mean"),
            recall_up=("recall_up", "mean"),
            recall_down=("recall_down", "mean"),
            recall_range=("recall_range", "mean"),
            score=("score", "mean"),
        )
        .reset_index()
        .sort_values("score", ascending=False)
    )
    agg.to_csv(out_dir / "method_aggregate.csv", index=False, encoding="utf-8-sig")

    # method win count
    win = pd.DataFrame(per_code_best)["method"].value_counts().rename_axis("method").reset_index(name="n_best")
    win.to_csv(out_dir / "method_win_counts.csv", index=False, encoding="utf-8-sig")

    best_method = str(agg.iloc[0]["method"])
    # Prefer balanced 3-state method when score leader barely labels range.
    hyst_rows = agg[agg["method"] == "ma_stack_hyst"]
    if best_method == "adx_di" and len(hyst_rows) and float(agg.iloc[0]["recall_range"]) < 0.15:
        best_method = "ma_stack_hyst"
        print("OVERRIDE recommend -> ma_stack_hyst (adx_di range recall too low)")
    print("\n=== AGG ===")
    print(agg.to_string(index=False))
    print("wins\n", win.to_string(index=False))
    print("BEST", best_method)

    # detailed description of best method + switch rule
    lines = [
        "# 全池因果状态划分评估：如何无前瞻地切上涨/下跌/横盘",
        "",
        "> 问题：在**不知道未来**的前提下，如何划分状态并找到切换点？",
        f"> 样本：池内 **{sdf['code'].nunique()}** 只标的 × 各自全部可用日K（至 `{args.end_date}`）。",
        "> **真值（仅评估）**：回顾大幅摆动峰谷切主升/主跌，其余为横盘（含前瞻，不当交易信号）。",
        "> **候选方法**：全部只用当日及历史价/均线/波动，因果可实盘。",
        "",
        "## 1. 评估指标",
        "",
        "| 指标 | 含义 |",
        "|---|---|",
        "| agree | 日标签与回顾真值一致率 |",
        "| recall_up/down/range | 各类召回 |",
        "| switch_hit | 真值切换点在 ±10 日内被预测切换命中的比例 |",
        "| switch_false | 预测切换中未命中真值切换的比例 |",
        "| switch_delay_mean | 命中切换的平均延迟（日，+晚/−早） |",
        "| score | 综合分：一致率+切换命中+召回 − 假切换 − 延迟惩罚 |",
        "",
        "## 2. 候选方法",
        "",
        "| 方法 | 核心规则 |",
        "|---|---|",
        "| `ma_stack_hyst` | MA20/MA60 多空堆叠 + prior + 波幅收窄判横盘 + 破/站 MA20 滞后确认 + 5日多数票 |",
        "| `ma_stack_struct` | 同堆叠，但 **跌破 MA20 未破结构仍算上涨**；离场看 MA20/MA60 翻转（≥3日），主升段更不易切碎 |",
        "| `ma_stack_strict` | 更严 prior 门槛，横盘更多，确认 3 日 |",
        "| `dual_ma_cross` | 仅 MA20 vs MA60；贴近则横盘 |",
        "| `adx_di` | ADX&lt;18 横盘，否则 DI+/DI− 定方向 + 滞后 |",
        "| `prior60_band` | prior60 方向 + 60日波幅收窄 |",
        "| `hybrid_ma_adx` | ma_stack 结果再用 ADX/prior 弱化假趋势 |",
        "",
        "## 3. 全池汇总（按 score）",
        "",
        "| 方法 | 标的数 | agree | switch_hit | false_sw | delay | recall_up | recall_down | recall_range | score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| `{r.method}` | {int(r.n_codes)} | {r.agree:.1%} | {r.switch_hit:.1%} | {r.switch_false:.1%} | "
            f"{r.switch_delay_mean:+.1f} | {r.recall_up:.1%} | {r.recall_down:.1%} | {r.recall_range:.1%} | {r.score:.3f} |"
        )

    lines += [
        "",
        "## 4. 各标的最优方法票数",
        "",
        "| 方法 | 成为该票最优的次数 |",
        "|---|---:|",
    ]
    for _, r in win.iterrows():
        lines.append(f"| `{r.method}` | {int(r.n_best)} |")

    lines += [
        "",
        f"## 5. 推荐定稿：`{best_method}`",
        "",
    ]

    if best_method == "ma_stack_hyst":
        lines += [
            "### 日状态（因果）",
            "",
            "1. 计算 `MA20/MA60`、`prior20/prior60`、`band60`、`rv20` 历史分位",
            "2. **横盘候选**：`rv20` 长时间处历史低分位（默认 <35% 且持续约10/15日），"
            "或 `band60` 收窄且 `|prior60|` 小 / MA20≈MA60；短于10日的方向段并入邻居",
            "3. **上涨候选**：`MA20>MA60` 且 `C>MA20`，且 prior 偏强",
            "4. **下跌候选**：`MA20<MA60` 且 `C<MA20`，且 prior 偏弱",
            "5. 冲突时：收盘相对 MA20 倾斜；强横盘优先",
            "",
            "### 切换点（滞后状态机，防抖）",
            "",
            "- 当前 **上涨**：",
            "  - 出现下跌标签 且（连续≥2日破 MA20 **或** MA20&lt;MA60）→ 切 **下跌**",
            "  - 或 横盘标签 且连续≥3日破 MA20 → 切 **横盘**",
            "- 当前 **下跌**：对称（站上 MA20 / MA20&gt;MA60）→ 上涨或横盘",
            "- 当前 **横盘**：出现明确上涨/下跌标签并经确认 → 切入趋势",
            "- 最后对状态做 **过去 5 日多数票**（不含未来）",
            "",
            "### 切换点操作定义（实盘）",
            "",
            "```text",
            "切换日 = 状态机输出标签首次不同于昨日的交易日",
            "（已含 2～3 日确认，故切换相对「真拐点」通常晚数日，换稳定性）",
            "```",
            "",
            "> 若主升段被 MA20 回调切碎（如 SH588710），改用 **`ma_stack_struct`**：离场看 MA20/MA60 结构翻转，不因短破 MA20 切下跌。",
            "",
        ]
    elif best_method == "ma_stack_struct":
        lines += [
            "### 日状态（因果）",
            "",
            "1. 同 `ma_stack_hyst` 的均线/prior/band 特征",
            "2. **上涨**：`MA20>MA60` 且 `C≥MA60`（跌破 MA20 但未破结构仍算上涨回撤）",
            "3. **下跌**：对称",
            "4. **离场**：仅当 MA20/MA60 堆叠翻转持续≥3 日（+5 日多数票）",
            "",
            "### 切换点操作定义（实盘）",
            "",
            "```text",
            "切换日 = 状态机输出标签首次不同于昨日的交易日",
            "```",
            "",
        ]
    else:
        lines += [
            f"详见代码 `METHODS['{best_method}']`。",
            "",
        ]

    lines += [
        "## 6. 解读与使用建议",
        "",
        "- **不可能**在无前瞻下与回顾峰谷 100% 对齐；目标是高一致率 + 可接受的切换延迟。",
        "- 切换点会系统性 **偏晚**（确认代价）；做「上涨持有」时，偏晚离场通常优于假突破频繁进出。",
        "- 横盘召回与趋势召回存在权衡：`strict` 横盘更多，趋势段更短。",
        "- 强趋势标的若被 `ma_stack_hyst` 用 MA20 回调切碎，改用 **`ma_stack_struct`**（结构离场）；全池 switch_hit 会变差，但段更干净、上涨持有更稳。",
        "- 建议生产：**默认 `ma_stack_hyst` 打日状态**；单票主升切碎时切 `ma_stack_struct`；买卖规则仍按状态族匹配。",
        "- 回顾峰谷分段只用于研究归因与本文评估，不进实盘。",
        "",
        "## 7. 产物",
        "",
        f"`{out_dir.relative_to(_PROJECT_ROOT)}/`",
        "",
        "- `method_aggregate.csv`",
        "- `per_code_method_metrics.csv`",
        "- `per_code_best_method.csv`",
        "- `method_win_counts.csv`",
        "- `summary.md`",
        "",
        "## 8. 版本",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 标的池 | {sdf['code'].nunique()} |",
        f"| 截止 | {args.end_date} |",
        f"| 推荐方法 | `{best_method}` |",
        "| 状态 | 全池因果划分评估 |",
    ]

    text = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    doc = _PROJECT_ROOT / "documents" / "全池因果状态划分与切换点评估.md"
    doc.write_text(text, encoding="utf-8")
    print("WROTE", doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
