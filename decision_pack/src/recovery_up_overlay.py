"""Causal recovery_up overlay: range→up after production finalize.

Design: huangtuo-master-design-20260812-161837-recovery-up-overlay.md
Eng plan: huangtuo-master-eng-plan-20260812-165855-recovery-up-overlay.md

Does not latch; does not require MA20>MA60; upgrades only native ``range``.
ED unavailable ⇒ fail-closed (no upgrade) for affected bars / series.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# --- Frozen defaults (single source; harness echoes into acceptance.json) ---
HUMAN_FLIP_DATE = "2026-07-22"
FOCUS_CODE = "SH515220"
FOCUS_WINDOW_START = "2026-06-30"
FOCUS_WINDOW_END = "2026-08-11"
DEFAULT_L = 60
DEFAULT_K_PIT = 5
DEFAULT_R = 0.10
DEFAULT_N_STREAK = 3
DEFAULT_P20_MIN = 0.03
DEFAULT_REC_MIN = 0.10
DEFAULT_K_ED = 1
DEFAULT_D_FLIP = 5

PROMOTE = "promote_candidate"
KEEP = "keep_baseline"
DIAGNOSTIC_PROMOTE = "diagnostic_promote"

# Diagnostic dual-channel default (C8 / HRP recovery_up mode)
DIAGNOSTIC_N = 2

# C17 shallow_up trade overlay (design 20260813-104222)
SHALLOW_FOCUS_CODE = "SH513080"
SHALLOW_PEER_CODE = "SZ159561"
SHALLOW_TROUGH_DATE = "2026-07-24"
SHALLOW_FLIP_DATE = "2026-07-30"  # first causal C>MA20 streak; user accepted
SHALLOW_FLIP_DEADLINE = "2026-07-30"
SHALLOW_N = 2
SHALLOW_P20_GRID = (0.0, 0.015, 0.03)
SHALLOW_P20_MIN_DEFAULT = 0.0  # needed to fire on 2026-07-30 (p20≈1.5%)


def constants_snapshot() -> dict[str, Any]:
    return {
        "HUMAN_FLIP_DATE": HUMAN_FLIP_DATE,
        "FOCUS_CODE": FOCUS_CODE,
        "FOCUS_WINDOW": [FOCUS_WINDOW_START, FOCUS_WINDOW_END],
        "L": DEFAULT_L,
        "k": DEFAULT_K_PIT,
        "R": DEFAULT_R,
        "N": DEFAULT_N_STREAK,
        "p20_min": DEFAULT_P20_MIN,
        "rec_min": DEFAULT_REC_MIN,
        "k_ed": DEFAULT_K_ED,
        "D_flip": DEFAULT_D_FLIP,
    }


def ed_alert_bool_from_frame(
    ed_frame: pd.DataFrame | None,
    n: int,
) -> tuple[np.ndarray | None, bool]:
    """Map early_down_flags_frame → per-bar alert mask.

    Returns ``(alert_bool, ed_ok)``. If ``ed_ok`` is False, caller must
    fail-closed (treat as no upgrades / pass None into ``recovery_up_mask``).
    """
    if n <= 0:
        return np.zeros(0, dtype=bool), True
    if ed_frame is None or len(ed_frame) == 0:
        return None, False
    if "level" not in ed_frame.columns:
        return None, False
    if len(ed_frame) != n:
        return None, False
    level = ed_frame["level"].astype(str).to_numpy()
    return (level == "alert"), True


def _rolling_any(flag: np.ndarray, k: int) -> np.ndarray:
    """True if any of the last ``k`` bars (inclusive) is True."""
    n = len(flag)
    out = np.zeros(n, dtype=bool)
    if n == 0 or k <= 0:
        return out
    pref = np.zeros(n + 1, dtype=int)
    for i in range(n):
        pref[i + 1] = pref[i] + (1 if flag[i] else 0)
    for i in range(n):
        lo = max(0, i - k + 1)
        out[i] = (pref[i + 1] - pref[lo]) > 0
    return out


def _has_pit_run(
    below60: np.ndarray,
    i: int,
    L: int,
    k: int,
) -> bool:
    """Within [i-L+1, i], exists consecutive ≥k days with C<MA60."""
    lo = i - L + 1
    if lo < 0:
        return False
    run = 0
    for j in range(lo, i + 1):
        if below60[j]:
            run += 1
            if run >= k:
                return True
        else:
            run = 0
    return False


def recovery_up_mask(
    px: pd.DataFrame,
    native_regime: np.ndarray | list[str],
    ed_alert_bool: np.ndarray | None,
    *,
    L: int = DEFAULT_L,
    k: int = DEFAULT_K_PIT,
    R: float = DEFAULT_R,
    N: int = DEFAULT_N_STREAK,
    p20_min: float = DEFAULT_P20_MIN,
    rec_min: float = DEFAULT_REC_MIN,
    k_ed: int = DEFAULT_K_ED,
) -> np.ndarray:
    """Causal bar-level recovery_up predicate (no latch).

    If ``ed_alert_bool`` is None ⇒ fail-closed (all False).
    Requires a full L-day window; earlier bars stay False.
    Constraint: ``rec_min <= R`` is caller's responsibility (defaults equal).
    """
    n = len(px)
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out

    labs = np.asarray(native_regime, dtype=object)
    if len(labs) != n:
        raise ValueError("native_regime length mismatch")

    if ed_alert_bool is None:
        return out  # fail-closed: no upgrades
    ed = np.asarray(ed_alert_bool, dtype=bool)
    if len(ed) != n:
        raise ValueError("ed_alert_bool length mismatch")
    ed_block = _rolling_any(ed, int(k_ed))

    close = px["$close"].to_numpy(float)
    ma20 = px["MA20"].to_numpy(float)
    ma60 = px["MA60"].to_numpy(float)
    if "prior20" in px.columns:
        p20 = pd.to_numeric(px["prior20"], errors="coerce").to_numpy(float)
    else:
        p20 = np.full(n, np.nan)

    below60 = (
        np.isfinite(close)
        & np.isfinite(ma60)
        & (close < ma60)
    )
    above20 = (
        np.isfinite(close)
        & np.isfinite(ma20)
        & (close > ma20)
    )

    streak = np.zeros(n, dtype=int)
    run = 0
    for i in range(n):
        if above20[i]:
            run += 1
        else:
            run = 0
        streak[i] = run

    L = int(L)
    k = int(k)
    N = int(N)
    for i in range(n):
        if i < L - 1:
            continue
        if str(labs[i]) != "range":
            continue
        if ed_block[i]:
            continue
        if streak[i] < N:
            continue
        window = close[i - L + 1 : i + 1]
        if not np.isfinite(window).all() or not np.isfinite(close[i]):
            continue
        trough = float(np.min(window))
        if trough <= 0 or not np.isfinite(trough):
            continue
        rec = float(close[i] / trough - 1.0)
        pit_ok = _has_pit_run(below60, i, L, k)
        if not (pit_ok or rec >= float(R)):
            continue
        p20i = float(p20[i]) if np.isfinite(p20[i]) else float("-inf")
        if not (p20i >= float(p20_min) or rec >= float(rec_min)):
            continue
        out[i] = True
    return out


def latch_recovery_up_mask(
    px: pd.DataFrame,
    native_regime: np.ndarray | list[str],
    fire_mask: np.ndarray,
    ed_alert_bool: np.ndarray | None,
    *,
    k_ed: int = DEFAULT_K_ED,
) -> np.ndarray:
    """Hold a recovery_up fire until C≤MA20, ED, or native leaves range.

    Causal: bar *i* is latched iff fire_mask[i] or (latched[i-1] and still
    native range, C>MA20, not ED). Does not rewrite  native ``up``/``down``.
    ``ed_alert_bool is None`` ⇒ fail-closed (return zeros).
    """
    n = len(px)
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    labs = np.asarray(native_regime, dtype=object)
    fire = np.asarray(fire_mask, dtype=bool)
    if len(labs) != n or len(fire) != n:
        raise ValueError("latch length mismatch")
    if ed_alert_bool is None:
        return out
    ed = np.asarray(ed_alert_bool, dtype=bool)
    if len(ed) != n:
        raise ValueError("ed_alert_bool length mismatch")
    ed_block = _rolling_any(ed, int(k_ed))
    close = px["$close"].to_numpy(float)
    ma20 = px["MA20"].to_numpy(float)
    above20 = np.isfinite(close) & np.isfinite(ma20) & (close > ma20)
    held = False
    for i in range(n):
        if str(labs[i]) != "range" or ed_block[i] or not above20[i]:
            held = False
            out[i] = False
            continue
        if fire[i] or held:
            held = True
            out[i] = True
        else:
            out[i] = False
    return out


def apply_recovery_up(
    labels: np.ndarray | list[str],
    mask: np.ndarray,
) -> np.ndarray:
    """Return a **new** label array; upgrade ``range``→``up`` where mask is True."""
    labs = np.asarray(labels, dtype=object).copy()
    m = np.asarray(mask, dtype=bool)
    if len(labs) != len(m):
        raise ValueError("labels/mask length mismatch")
    for i in range(len(labs)):
        if m[i] and str(labs[i]) == "range":
            labs[i] = "up"
    return labs


def decide_recommendation(
    pool_ok: bool,
    focus_ok: bool,
    flip_ok: bool,
) -> str:
    if pool_ok and focus_ok and flip_ok:
        return PROMOTE
    return KEEP


def decide_diagnostic_promote(*, focus_ok: bool, flip_ok: bool) -> str:
    """Trade labels unchanged; promote diagnostic channel only."""
    if focus_ok and flip_ok:
        return DIAGNOSTIC_PROMOTE
    return KEEP


def detect_recovery_up_marks(
    px: pd.DataFrame,
    native_regime: np.ndarray | list[str],
    ed_alert_bool: np.ndarray | None,
    *,
    N: int = DIAGNOSTIC_N,
    L: int = DEFAULT_L,
    k: int = DEFAULT_K_PIT,
    R: float = DEFAULT_R,
    p20_min: float = DEFAULT_P20_MIN,
    rec_min: float = DEFAULT_REC_MIN,
    k_ed: int = DEFAULT_K_ED,
) -> list[dict[str, Any]]:
    """Visual-only marks at the start of each recovery_up mask run.

    Does **not** change regime labels or trades. ``kind`` is ``ru_alert``.
    """
    mask = recovery_up_mask(
        px,
        native_regime,
        ed_alert_bool,
        L=L,
        k=k,
        R=R,
        N=N,
        p20_min=p20_min,
        rec_min=rec_min,
        k_ed=k_ed,
    )
    labs = np.asarray(native_regime, dtype=object)
    if "as_of" in px.columns:
        dates = pd.to_datetime(px["as_of"]).dt.normalize()
    else:
        dates = pd.Series(pd.RangeIndex(len(px)))
    close = px["$close"].to_numpy(float) if "$close" in px.columns else np.full(len(px), np.nan)

    marks: list[dict[str, Any]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        d0 = pd.Timestamp(dates.iloc[i]).strftime("%Y-%m-%d")
        d1 = pd.Timestamp(dates.iloc[j]).strftime("%Y-%m-%d")
        marks.append(
            {
                "date": d0,
                "end": d1,
                "kind": "ru_alert",
                "level": "alert",
                "symbol": "RU",
                "title": "RU恢复上涨(诊断)",
                "why": "深坑恢复命中且生产仍为 range；诊断双通道，不改买卖",
                "reasons": f"recovery_up N={int(N)}",
                "n_bars": int(j - i + 1),
                "px": float(close[i]) if np.isfinite(close[i]) else None,
                "regime": str(labs[i]) if i < len(labs) else "",
            }
        )
        i = j + 1
    return marks


def first_up_date(
    as_of: pd.Series | np.ndarray,
    labels: np.ndarray,
    *,
    start: str | None = None,
    end: str | None = None,
) -> str | None:
    """First date with label == up in optional [start, end] window."""
    dates = pd.to_datetime(pd.Series(as_of)).dt.normalize()
    labs = np.asarray(labels, dtype=object)
    m = np.ones(len(labs), dtype=bool)
    if start is not None:
        m &= dates >= pd.Timestamp(start)
    if end is not None:
        m &= dates <= pd.Timestamp(end)
    for d, lab in zip(dates[m], labs[m]):
        if str(lab) == "up":
            return pd.Timestamp(d).strftime("%Y-%m-%d")
    return None


def trading_days_between(
    as_of: pd.Series | np.ndarray,
    d0: str,
    d1: str,
) -> int | None:
    """Index distance on the ``as_of`` calendar: idx(d1) - idx(d0).

    Same-day ⇒ 0. Missing either date ⇒ None.
    """
    dates = pd.to_datetime(pd.Series(as_of)).dt.normalize()
    uniq = dates.drop_duplicates().sort_values().reset_index(drop=True)
    idx = {pd.Timestamp(x): i for i, x in enumerate(uniq)}
    t0, t1 = pd.Timestamp(d0), pd.Timestamp(d1)
    if t0 not in idx or t1 not in idx:
        return None
    return int(idx[t1] - idx[t0])


def up_frac_in_window(
    as_of: pd.Series | np.ndarray,
    labels: np.ndarray,
    start: str,
    end: str,
) -> float:
    dates = pd.to_datetime(pd.Series(as_of)).dt.normalize()
    labs = np.asarray(labels, dtype=object)
    m = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    L = labs[m.to_numpy()]
    if len(L) == 0:
        return float("nan")
    return float((L == "up").mean())
