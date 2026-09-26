"""Hold-suitable vs swing-only bucketing + dual economic benchmarks.

Classification uses **train-window only** MA20±2% truth quality (no OOS peek):
- ``hold_ok``: BH > 0 and enough time in chart-up → opportunity cost is BH
- ``swing_only``: otherwise → opportunity cost is cash (0); also report up-capture

Up-capture = strategy compound / ideal MA20 enter-up/leave-up compound (same costs).
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from etf_daily.lib.event_long_only_sim import (
    COST_PER_SIDE,
    simulate_long_only_events,
)
from etf_daily.lib.fwd_return_trend_truth import STATE_DOWN, STATE_UP
from etf_daily.lib.ma20_truth_triggers import truth_entry_exit

Bucket = Literal["hold_ok", "swing_only"]

# Train-only defaults (tuned for ETF pool; override via classify kwargs).
HOLD_BH_MIN = 0.0
HOLD_UP_FRAC_MIN = 0.20


def _window_slice(
    close: pd.Series,
    truth: pd.Series,
    start: str,
    end: str,
) -> tuple[pd.Series, pd.Series]:
    idx = pd.DatetimeIndex(pd.to_datetime(close.index)).normalize()
    lo = pd.Timestamp(start).normalize()
    hi = pd.Timestamp(end).normalize()
    mask = (idx >= lo) & (idx <= hi)
    sub_c = pd.to_numeric(close, errors="coerce").reindex(idx)[mask]
    sub_t = truth.reindex(idx)[mask]
    return sub_c, sub_t


def trend_quality_metrics(
    close: pd.Series,
    truth: pd.Series,
    *,
    start: str,
    end: str,
) -> dict[str, Any]:
    """Regime / trend quality on [start, end] from MA20 truth + price path."""
    sub_c, sub_t = _window_slice(close, truth, start, end)
    n = int(len(sub_c))
    if n < 2:
        return {
            "n_bars": n,
            "bh": float("nan"),
            "up_frac": float("nan"),
            "down_frac": float("nan"),
            "n_up_episodes": 0,
            "mean_up_episode_bars": float("nan"),
            "up_path_ret": float("nan"),
        }
    px0 = float(sub_c.iloc[0])
    px1 = float(sub_c.iloc[-1])
    bh = float(px1 / px0 - 1.0) if np.isfinite(px0) and px0 > 0 and np.isfinite(px1) else float("nan")
    lab = sub_t.dropna()
    up_frac = float(lab.eq(STATE_UP).mean()) if len(lab) else float("nan")
    down_frac = float(lab.eq(STATE_DOWN).mean()) if len(lab) else float("nan")

    is_up = sub_t.eq(STATE_UP).fillna(False).to_numpy(dtype=bool)
    # Episode lengths while labeled up.
    lengths: list[int] = []
    run = 0
    for v in is_up:
        if v:
            run += 1
        elif run:
            lengths.append(run)
            run = 0
    if run:
        lengths.append(run)

    # Sum of close-to-close returns on bars that start already in up (path inside up).
    rets = sub_c.pct_change(fill_method=None)
    up_path = rets.where(sub_t.shift(1).eq(STATE_UP), 0.0).fillna(0.0)
    up_path_ret = float((1.0 + up_path).prod() - 1.0)

    return {
        "n_bars": n,
        "bh": bh,
        "up_frac": up_frac,
        "down_frac": down_frac,
        "n_up_episodes": int(len(lengths)),
        "mean_up_episode_bars": float(np.mean(lengths)) if lengths else 0.0,
        "up_path_ret": up_path_ret,
    }


def classify_hold_bucket(
    metrics: dict[str, Any],
    *,
    bh_min: float = HOLD_BH_MIN,
    up_frac_min: float = HOLD_UP_FRAC_MIN,
) -> Bucket:
    """Map train trend-quality metrics → hold_ok | swing_only."""
    bh = metrics.get("bh", float("nan"))
    up_frac = metrics.get("up_frac", float("nan"))
    if not np.isfinite(bh) or not np.isfinite(up_frac):
        return "swing_only"
    if float(bh) > float(bh_min) and float(up_frac) >= float(up_frac_min):
        return "hold_ok"
    return "swing_only"


def ideal_up_compound(
    close: pd.Series,
    truth: pd.Series,
    *,
    start: str,
    end: str,
    cost_per_side: float = COST_PER_SIDE,
) -> float:
    """Compound of trading truth enter-up / leave-up on the window (same costs)."""
    sub_c, sub_t = _window_slice(close, truth, start, end)
    if len(sub_c) < 2:
        return float("nan")
    entry, exit_sig = truth_entry_exit(sub_t)
    metrics, _, _ = simulate_long_only_events(
        entry,
        exit_sig,
        sub_c,
        dates=pd.DatetimeIndex(sub_c.index).strftime("%Y-%m-%d"),
        cost_per_side=cost_per_side,
    )
    return float(metrics["compound"])


def attach_dual_benchmarks(
    sim_metrics: dict[str, Any],
    *,
    close: pd.Series,
    truth: pd.Series,
    start: str,
    end: str,
    cost_per_side: float = COST_PER_SIDE,
) -> dict[str, Any]:
    """Add edge_bh, edge_cash, ideal_up_compound, up_capture to sim metrics."""
    out = dict(sim_metrics)
    compound = float(out.get("compound", float("nan")))
    bh = float(out.get("bh", float("nan")))
    out["edge_bh"] = float(compound - bh) if np.isfinite(compound) and np.isfinite(bh) else float("nan")
    # Keep legacy ``edge`` = vs BH for backward compatibility.
    out["edge"] = out["edge_bh"]
    out["edge_cash"] = compound if np.isfinite(compound) else float("nan")
    ideal = ideal_up_compound(
        close, truth, start=start, end=end, cost_per_side=cost_per_side
    )
    out["ideal_up_compound"] = ideal
    if np.isfinite(compound) and np.isfinite(ideal) and abs(ideal) > 1e-12:
        out["up_capture"] = float(compound / ideal)
    else:
        out["up_capture"] = float("nan")
    return out


def gate_edge_key(bucket: Bucket) -> str:
    """Which per-code edge feeds the economic gate for this bucket."""
    return "edge_bh" if bucket == "hold_ok" else "edge_cash"


def label_at(
    bucket: pd.Series,
    when: str | pd.Timestamp,
    *,
    default: Bucket = "swing_only",
) -> Bucket:
    """Causal label at ``when`` (last available bar on or before when)."""
    s = bucket.copy()
    s.index = pd.to_datetime(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    ts = pd.Timestamp(when).normalize()
    prior = s.loc[s.index <= ts]
    if prior.empty:
        return default
    lab = str(prior.iloc[-1])
    return "hold_ok" if lab == "hold_ok" else "swing_only"


def pit_path_edge(
    equity: pd.Series | pd.DataFrame,
    close: pd.Series,
    bucket: pd.Series,
    *,
    start: str,
    end: str,
) -> dict[str, Any]:
    """Month-by-month dual benchmark along the path.

    At each month-end decision date d in (start, end], use bucket at d (≤d info)
    as the opportunity-cost regime for the month ending at d:
    - hold_ok → month_edge = strat_ret − bh_ret
    - swing_only → month_edge = strat_ret − 0

    ``edge_pit`` compounds those monthly edges. Also returns hold_frac on
    month-ends and n_months.
    """
    if isinstance(equity, pd.DataFrame):
        if "equity" in equity.columns:
            eq = pd.to_numeric(equity["equity"], errors="coerce").copy()
            if "date" in equity.columns:
                eq.index = pd.DatetimeIndex(pd.to_datetime(equity["date"])).normalize()
        else:
            raise ValueError("equity DataFrame needs equity column")
    else:
        eq = pd.to_numeric(equity, errors="coerce").copy()
        eq.index = pd.DatetimeIndex(pd.to_datetime(eq.index)).normalize()
    eq = eq[~eq.index.duplicated(keep="last")].sort_index()

    px = pd.to_numeric(close, errors="coerce").copy()
    px.index = pd.DatetimeIndex(pd.to_datetime(px.index)).normalize()
    px = px[~px.index.duplicated(keep="last")].sort_index()

    buck = bucket.copy()
    buck.index = pd.DatetimeIndex(pd.to_datetime(buck.index)).normalize()
    buck = buck[~buck.index.duplicated(keep="last")].sort_index()

    lo = pd.Timestamp(start).normalize()
    hi = pd.Timestamp(end).normalize()
    mask = (eq.index >= lo) & (eq.index <= hi)
    eq = eq.loc[mask]
    if len(eq) < 2:
        return {
            "edge_pit": float("nan"),
            "pit_hold_frac": float("nan"),
            "pit_n_months": 0,
            "pit_mean_month_edge": float("nan"),
        }

    dec = decision_dates(eq.index, freq="M")
    dec = dec[(dec >= eq.index[0]) & (dec <= eq.index[-1])]
    if len(dec) < 2:
        # Fall back to full-window single shot using label at end.
        lab = label_at(buck, hi)
        strat = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
        bh = float(px.reindex(eq.index).ffill().iloc[-1] / px.reindex(eq.index).ffill().iloc[0] - 1.0)
        edge = strat - bh if lab == "hold_ok" else strat
        return {
            "edge_pit": float(edge),
            "pit_hold_frac": 1.0 if lab == "hold_ok" else 0.0,
            "pit_n_months": 1,
            "pit_mean_month_edge": float(edge),
        }

    month_edges: list[float] = []
    hold_flags: list[bool] = []
    px_w = px.reindex(eq.index).ffill()
    for i in range(1, len(dec)):
        d0 = dec[i - 1]
        d1 = dec[i]
        if d0 not in eq.index or d1 not in eq.index:
            continue
        if d0 not in px_w.index or d1 not in px_w.index:
            continue
        e0, e1 = float(eq.loc[d0]), float(eq.loc[d1])
        p0, p1 = float(px_w.loc[d0]), float(px_w.loc[d1])
        if not (np.isfinite(e0) and np.isfinite(e1) and e0 > 0 and np.isfinite(p0) and p0 > 0):
            continue
        strat_m = e1 / e0 - 1.0
        bh_m = p1 / p0 - 1.0
        lab = label_at(buck, d1)
        hold_flags.append(lab == "hold_ok")
        month_edges.append(strat_m - bh_m if lab == "hold_ok" else strat_m)

    if not month_edges:
        return {
            "edge_pit": float("nan"),
            "pit_hold_frac": float("nan"),
            "pit_n_months": 0,
            "pit_mean_month_edge": float("nan"),
        }
    wealth = 1.0
    for m in month_edges:
        wealth *= 1.0 + float(m)
    return {
        "edge_pit": float(wealth - 1.0),
        "pit_hold_frac": float(np.mean(hold_flags)),
        "pit_n_months": int(len(month_edges)),
        "pit_mean_month_edge": float(np.mean(month_edges)),
    }


# --- Causal point-in-time hold_worthy ---------------------------------

# Defaults for rolling decision rule (trading-day lookback).
ROLL_WINDOW = 126  # ~6 months
ROLL_BH_MIN = HOLD_BH_MIN
ROLL_UP_FRAC_MIN = HOLD_UP_FRAC_MIN
# Hysteresis exits (stay hold_ok until clearly broken).
ROLL_BH_EXIT = -0.05
ROLL_UP_FRAC_EXIT = 0.15


def rolling_hold_features(
    close: pd.Series,
    truth: pd.Series,
    *,
    window: int = ROLL_WINDOW,
) -> pd.DataFrame:
    """Causal trailing-window BH / up_frac at each bar (uses only ≤ t)."""
    s = pd.to_numeric(close, errors="coerce").copy()
    s.index = pd.to_datetime(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    t = truth.reindex(s.index)
    w = int(window)
    if w < 2:
        raise ValueError("window must be >= 2")
    up = t.eq(STATE_UP).astype(float)
    up = up.where(t.notna(), np.nan)
    # Trailing BH: close[t] / close[t-w+1] - 1
    bh = s / s.shift(w - 1) - 1.0
    up_frac = up.rolling(w, min_periods=w).mean()
    n_bars = s.notna().astype(int).rolling(w, min_periods=1).sum()
    down = t.eq(STATE_DOWN).astype(float).where(t.notna(), np.nan)
    down_frac = down.rolling(w, min_periods=w).mean()
    out = pd.DataFrame(
        {
            "bh": bh,
            "up_frac": up_frac,
            "down_frac": down_frac,
            "n_bars": n_bars,
        },
        index=s.index,
    )
    # Incomplete window → NaN features
    out.loc[n_bars < w, ["bh", "up_frac", "down_frac"]] = np.nan
    return out


def _raw_hold_mask(
    feat: pd.DataFrame,
    *,
    bh_min: float,
    up_frac_min: float,
) -> pd.Series:
    ok = (
        feat["bh"].gt(float(bh_min))
        & feat["up_frac"].ge(float(up_frac_min))
        & feat["bh"].notna()
        & feat["up_frac"].notna()
    )
    return ok.fillna(False)


def rolling_hold_worthy(
    close: pd.Series,
    truth: pd.Series,
    *,
    window: int = ROLL_WINDOW,
    bh_min: float = ROLL_BH_MIN,
    up_frac_min: float = ROLL_UP_FRAC_MIN,
    hysteresis: bool = True,
    bh_exit: float = ROLL_BH_EXIT,
    up_frac_exit: float = ROLL_UP_FRAC_EXIT,
) -> pd.DataFrame:
    """Point-in-time hold_ok / swing_only series + features.

    Without hysteresis: hold_ok iff trailing BH > bh_min and up_frac ≥ min.
    With hysteresis: enter on the same rule; leave hold_ok only when
    BH ≤ bh_exit **or** up_frac < up_frac_exit (reduces day-to-day flips).
    Warm-up (incomplete window) → swing_only.
    """
    feat = rolling_hold_features(close, truth, window=window)
    enter = _raw_hold_mask(feat, bh_min=bh_min, up_frac_min=up_frac_min)
    if not hysteresis:
        bucket = np.where(enter.to_numpy(), "hold_ok", "swing_only")
    else:
        leave = (
            feat["bh"].le(float(bh_exit))
            | feat["up_frac"].lt(float(up_frac_exit))
            | feat["bh"].isna()
            | feat["up_frac"].isna()
        )
        leave = leave.fillna(True)
        state = False
        labels: list[str] = []
        for i in range(len(feat)):
            if not state:
                state = bool(enter.iloc[i])
            else:
                if bool(leave.iloc[i]):
                    state = False
                # else stay hold_ok even if enter is momentarily false
            labels.append("hold_ok" if state else "swing_only")
        bucket = np.asarray(labels, dtype=object)
    out = feat.copy()
    out["hold_raw"] = enter.astype(bool)
    out["bucket"] = bucket
    return out


def decision_dates(
    index: pd.DatetimeIndex,
    *,
    freq: str = "M",
) -> pd.DatetimeIndex:
    """Subsample calendar to decision dates (default month-end in index)."""
    idx = pd.DatetimeIndex(pd.to_datetime(index)).normalize().unique().sort_values()
    if len(idx) == 0:
        return idx
    if freq in ("D", "B"):
        return idx
    # Month-end: last available bar in each calendar month.
    s = pd.Series(1, index=idx)
    # 'ME' is month-end in newer pandas; 'M' still works as alias in many versions.
    try:
        ends = s.resample("ME").last().dropna().index
    except ValueError:
        ends = s.resample("M").last().dropna().index
    # Map to actual bars present in idx (resample may keep period ends).
    out = []
    for d in ends:
        d = pd.Timestamp(d).normalize()
        if d in idx:
            out.append(d)
        else:
            prior = idx[idx <= d]
            if len(prior):
                out.append(prior[-1])
    return pd.DatetimeIndex(out).unique().sort_values()


def stability_stats(
    bucket: pd.Series,
    *,
    decision_freq: str = "M",
) -> dict[str, Any]:
    """Flip / spell stats on decision dates (causal labels already assigned)."""
    s = bucket.copy()
    s.index = pd.to_datetime(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    dec = decision_dates(s.index, freq=decision_freq)
    sub = s.reindex(dec).dropna()
    if len(sub) == 0:
        return {
            "n_decisions": 0,
            "hold_frac": float("nan"),
            "n_flips": 0,
            "flip_rate": float("nan"),
            "mean_spell_decisions": float("nan"),
            "n_hold_spells": 0,
            "n_swing_spells": 0,
        }
    vals = sub.astype(str).to_numpy()
    flips = int(np.sum(vals[1:] != vals[:-1])) if len(vals) > 1 else 0
    # Spell lengths in decision steps
    spells: list[tuple[str, int]] = []
    cur = vals[0]
    run = 1
    for v in vals[1:]:
        if v == cur:
            run += 1
        else:
            spells.append((str(cur), run))
            cur = v
            run = 1
    spells.append((str(cur), run))
    hold_spells = [n for lab, n in spells if lab == "hold_ok"]
    swing_spells = [n for lab, n in spells if lab == "swing_only"]
    return {
        "n_decisions": int(len(sub)),
        "hold_frac": float(np.mean(vals == "hold_ok")),
        "n_flips": flips,
        "flip_rate": float(flips / max(len(sub) - 1, 1)),
        "mean_spell_decisions": float(np.mean([n for _, n in spells])),
        "n_hold_spells": int(len(hold_spells)),
        "n_swing_spells": int(len(swing_spells)),
        "mean_hold_spell": float(np.mean(hold_spells)) if hold_spells else 0.0,
        "mean_swing_spell": float(np.mean(swing_spells)) if swing_spells else 0.0,
    }


def month_end_forward_panel(
    close: pd.Series,
    bucket: pd.Series,
    *,
    horizons: tuple[int, ...] = (63, 126),
    code: str = "",
) -> pd.DataFrame:
    """Month-end rows: causal bucket at d + forward BH over H bars (label only)."""
    px = pd.to_numeric(close, errors="coerce").copy()
    px.index = pd.DatetimeIndex(pd.to_datetime(px.index)).normalize()
    px = px[~px.index.duplicated(keep="last")].sort_index()
    buck = bucket.copy()
    buck.index = pd.DatetimeIndex(pd.to_datetime(buck.index)).normalize()
    buck = buck[~buck.index.duplicated(keep="last")].sort_index()
    dec = decision_dates(px.index, freq="M")
    rows: list[dict[str, Any]] = []
    for d in dec:
        if d not in px.index:
            continue
        lab = label_at(buck, d)
        pos = px.index.get_loc(d)
        if isinstance(pos, slice):
            continue
        p0 = float(px.iloc[pos])
        if not (np.isfinite(p0) and p0 > 0):
            continue
        row: dict[str, Any] = {
            "code": code,
            "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
            "bucket": lab,
        }
        for h in horizons:
            hi = int(h)
            if pos + hi >= len(px):
                row[f"fwd_{hi}"] = float("nan")
                continue
            p1 = float(px.iloc[pos + hi])
            row[f"fwd_{hi}"] = (
                float(p1 / p0 - 1.0) if np.isfinite(p1) and p1 > 0 else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def spread_metrics(
    panel: pd.DataFrame,
    *,
    horizon: int = 63,
    min_group_n: int = 5,
    pos_label: str = "hold_ok",
    neg_label: str = "swing_only",
    fwd_col: str | None = None,
) -> dict[str, Any]:
    """Pool ranking metrics: mean(fwd|pos) − mean(fwd|neg)."""
    col = fwd_col or f"fwd_{int(horizon)}"
    if panel.empty or "bucket" not in panel.columns or col not in panel.columns:
        return {
            "horizon": int(horizon),
            "fwd_col": col,
            "n": 0,
            "n_hold": 0,
            "n_swing": 0,
            "n_pos": 0,
            "n_neg": 0,
            "hold_frac": float("nan"),
            "pos_frac": float("nan"),
            "mean_fwd_hold": float("nan"),
            "mean_fwd_swing": float("nan"),
            "mean_fwd_pos": float("nan"),
            "mean_fwd_neg": float("nan"),
            "spread": float("nan"),
            "month_win_rate": float("nan"),
            "n_months": 0,
            "pass_min_group": False,
            "binary_acc": float("nan"),
        }
    sub = panel.dropna(subset=[col, "bucket"]).copy()
    hold = sub[sub["bucket"].astype(str) == str(pos_label)]
    swing = sub[sub["bucket"].astype(str) == str(neg_label)]
    mh = float(hold[col].mean()) if len(hold) else float("nan")
    ms = float(swing[col].mean()) if len(swing) else float("nan")
    spread = (
        float(mh - ms)
        if np.isfinite(mh) and np.isfinite(ms)
        else float("nan")
    )
    wins = 0
    n_months = 0
    for _, g in sub.groupby("date"):
        gh = g[g["bucket"].astype(str) == str(pos_label)][col]
        gs = g[g["bucket"].astype(str) == str(neg_label)][col]
        if len(gh) < 1 or len(gs) < 1:
            continue
        n_months += 1
        if float(gh.mean()) > float(gs.mean()):
            wins += 1
    pred = sub["bucket"].astype(str) == str(pos_label)
    truth = sub[col].astype(float) > 0.0
    binary_acc = float(np.mean(pred.to_numpy() == truth.to_numpy())) if len(sub) else float("nan")
    return {
        "horizon": int(horizon),
        "fwd_col": col,
        "n": int(len(sub)),
        "n_hold": int(len(hold)),
        "n_swing": int(len(swing)),
        "n_pos": int(len(hold)),
        "n_neg": int(len(swing)),
        "hold_frac": float(len(hold) / len(sub)) if len(sub) else float("nan"),
        "pos_frac": float(len(hold) / len(sub)) if len(sub) else float("nan"),
        "mean_fwd_hold": mh,
        "mean_fwd_swing": ms,
        "mean_fwd_pos": mh,
        "mean_fwd_neg": ms,
        "spread": spread,
        "month_win_rate": float(wins / n_months) if n_months else float("nan"),
        "n_months": int(n_months),
        "pass_min_group": bool(len(hold) >= int(min_group_n) and len(swing) >= int(min_group_n)),
        "binary_acc": binary_acc,
    }


__all__ = [
    "HOLD_BH_MIN",
    "HOLD_UP_FRAC_MIN",
    "ROLL_BH_EXIT",
    "ROLL_BH_MIN",
    "ROLL_UP_FRAC_EXIT",
    "ROLL_UP_FRAC_MIN",
    "ROLL_WINDOW",
    "Bucket",
    "attach_dual_benchmarks",
    "classify_hold_bucket",
    "decision_dates",
    "gate_edge_key",
    "ideal_up_compound",
    "label_at",
    "month_end_forward_panel",
    "pit_path_edge",
    "rolling_hold_features",
    "rolling_hold_worthy",
    "spread_metrics",
    "stability_stats",
    "trend_quality_metrics",
]
