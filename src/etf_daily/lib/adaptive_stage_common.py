"""Shared helpers for adaptive per-symbol stage rules (train / prior / OOS HTML)."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from etf_daily.lib.regime_transition_plot import VOL_REGIME_CLUSTER, VOL_REGIME_EXTREME
from etf_daily.lib.state_space_regime import (
    STATE_SPACE_METHODS,
    state_space_labels_or_fallback,
)
from etf_daily.lib.state_space_selection import (
    aggregate_candidate_metrics,
    candidate_grid,
    choose_candidate,
    evaluate_candidate_folds,
    expanding_splits as state_space_expanding_splits,
    fit_final_candidate,
)

# Bayes pseudo-count for posterior shrinkage
PRIOR_K = 5.0
MIN_TRAIN_BARS = 120
# One-way trading cost used when scoring by mean_ret_net
COST_PER_SIDE = 0.001
NO_TRADE_BUY = "NO_TRADE"
NO_TRADE_EXIT = "none"

# ---------------------------------------------------------------------------
# Rule libraries (buy name + exit name). Exit "leave_up" only for up/hold.
# ---------------------------------------------------------------------------
UP_BUY_SPECS: list[dict[str, Any]] = [
    dict(name="绿牛∨金叉∨红上MA20", green_bull=True, golden=True, red_ma20=True, vol_min=None),
    dict(name="绿牛∨金叉", green_bull=True, golden=True, red_ma20=False, vol_min=None),
    dict(name="绿牛∨金叉_vol≥0.5", green_bull=True, golden=True, red_ma20=False, vol_min=0.5),
    dict(name="金叉", green_bull=False, golden=True, red_ma20=False, vol_min=None),
    dict(name="绿牛", green_bull=True, golden=False, red_ma20=False, vol_min=None),
    dict(name="进入上涨态", hold_enter=True),
]
UP_EXIT_SPECS: list[dict[str, Any]] = [
    dict(name="红破MA10", mode="red_ma10"),
    dict(name="红破MA20", mode="red_ma20"),
    dict(name="破MA20", mode="break_ma20"),
    dict(name="死叉", mode="death"),
    dict(name="trail15", mode="trail", trail=0.15),
    dict(name="trail20", mode="trail", trail=0.20),
    dict(name="trail25", mode="trail", trail=0.25),
    dict(name="离上涨态", mode="leave_up"),
]

DOWN_BUY_SPECS: list[dict[str, Any]] = [
    dict(name="红近低", near_low=True, thr=0.2),
    dict(name="深红≤-5%", deep=True, thr=-0.05),
    dict(name="深红≤-7%", deep=True, thr=-0.07),
    dict(name="红地板", floor=True),
]
DOWN_EXIT_SPECS: list[dict[str, Any]] = [
    dict(name="tp10_sl6_h10", mode="short", tp=0.10, sl=-0.06, max_hold=10),
    dict(name="tp12_sl8_h10", mode="short", tp=0.12, sl=-0.08, max_hold=10),
    dict(name="tp15_sl8_h15", mode="short", tp=0.15, sl=-0.08, max_hold=15),
    dict(name="tp8_sl5_h8", mode="short", tp=0.08, sl=-0.05, max_hold=8),
    dict(name="trail10", mode="trail", trail=0.10),
    dict(name="trail15", mode="trail", trail=0.15),
    dict(name="破MA20", mode="break_ma20"),
    dict(name="红破MA10", mode="red_ma10"),
]

RANGE_BUY_SPECS: list[dict[str, Any]] = [
    dict(name="红近低∨绿牛", near_low=True, thr=0.25, green_bull=True),
    dict(name="红近低", near_low=True, thr=0.25, green_bull=False),
    dict(name="绿牛", near_low=False, green_bull=True),
    dict(name="金叉", golden=True),
]
RANGE_EXIT_SPECS: list[dict[str, Any]] = [
    dict(name="tp8_sl5_h10", mode="short", tp=0.08, sl=-0.05, max_hold=10),
    dict(name="tp6_sl4_h8", mode="short", tp=0.06, sl=-0.04, max_hold=8),
    dict(name="tp4_sl3_h5", mode="short", tp=0.04, sl=-0.03, max_hold=5),
    dict(name="trail10", mode="trail", trail=0.10),
    dict(name="trail15", mode="trail", trail=0.15),
    dict(name="红破MA10", mode="red_ma10"),
    dict(name="破MA20", mode="break_ma20"),
]

# Fixed SH588710 hybrid rules (for comparison column)
FIXED_HYBRID_RULES = {
    "method": "hybrid_ma_adx",
    "up": {"buy": "绿牛∨金叉∨红上MA20", "exit": "红破MA10"},
    "down": {"buy": "红近低", "exit": "tp10_sl6_h10"},
    "range": {"buy": "红近低∨绿牛", "exit": "tp8_sl5_h10"},
}

# Long-only: buy at up-regime start, sell when up ends; no trade in down/range.
REGIME_HOLD_UP_RULES: dict[str, dict[str, str]] = {
    "up": {"buy": "进入上涨态", "exit": "离上涨态"},
    "down": {"buy": NO_TRADE_BUY, "exit": NO_TRADE_EXIT},
    "range": {"buy": NO_TRADE_BUY, "exit": NO_TRADE_EXIT},
}
TRADE_MODE_HOLD_UP = "hold_up"
TRADE_MODE_BAYES = "bayes_rules"


def segments(regime: np.ndarray, dates: np.ndarray, close: np.ndarray) -> pd.DataFrame:
    labs = list(regime)
    rows: list[dict[str, Any]] = []
    s = 0
    for i in range(1, len(labs) + 1):
        if i == len(labs) or labs[i] != labs[s]:
            rows.append(
                dict(
                    regime=labs[s],
                    start=str(dates[s]),
                    end=str(dates[i - 1]),
                    n_bars=i - s,
                    ret=float(close[i - 1] / close[s] - 1),
                )
            )
            s = i
    return pd.DataFrame(rows)


TRUTH_VERSION = "retro_full+causal_confirmed_v1"
# Bump when METHOD label logic changes (invalidates adaptive config cache).
METHOD_IMPL_VERSION = "strict_vrev_nearstack_v1"
# Visual-only early marks: ±move from episode peak/trough (does not change labels/trades).
EARLY_REGIME_MOVE = 0.06
# When retrospective truth is almost all-range (short listing / no swings),
# label-agree scoring is meaningless — fall back to MA stack.
DEGENERATE_TRUTH_RANGE = 0.90
DEGENERATE_FALLBACK_METHOD = "ma_stack_hyst"
REGIME_SELECT_ENSEMBLE_CV = "ensemble_cv"  # B4 winner
REGIME_SELECT_LEGACY = "legacy_score_method"  # B0
REGIME_SELECT_STATE_SPACE_NESTED_CV = "state_space_nested_cv"
# backward-compat alias
AGGRESSIVE_VREVERSAL_BOUNCE = EARLY_REGIME_MOVE


def label_regime(
    ev,
    px: pd.DataFrame,
    method: str,
    method_params: dict[str, Any] | None = None,
    *,
    sticky_guard: bool = True,
) -> np.ndarray:
    """Apply a single METHOD or ensemble vote; optional sticky-up risk guard."""
    params = dict(method_params or {})
    if method in STATE_SPACE_METHODS:
        frozen = dict(params.get("frozen_model") or {})
        fallback_method = str(
            params.get("fallback_method")
            or frozen.get("fallback_method")
            or DEGENERATE_FALLBACK_METHOD
        )
        fallback = ev.run_method(fallback_method, px)
        labs, _diag = state_space_labels_or_fallback(
            px, frozen, fallback_labels=fallback
        )
    elif method == "ensemble":
        names = list(params.get("member_methods") or ev.METHODS.keys())
        weights = params.get("weights") or {n: 1.0 / len(names) for n in names}
        conf = float(params.get("conf_thr", 0.45))
        tuned = params.get("tuned_params") or {}
        mats = []
        for name in names:
            p = dict(tuned.get(name) or {})
            mats.append(ev.run_method(name, px, params=p))
        arr = np.stack(mats, axis=0)
        w = np.array([float(weights.get(n, 0.0)) for n in names], dtype=float)
        if w.sum() <= 0:
            w = np.ones(len(names), dtype=float) / len(names)
        else:
            w = w / w.sum()
        labs = ensemble_vote(arr, weights=w, conf_thr=conf)
    else:
        if method not in ev.METHODS:
            raise KeyError(f"unknown method {method}")
        labs = ev.run_method(method, px, params=params)
    if sticky_guard:
        labs = apply_sticky_up_guard(labs, px)
    return np.asarray(labs, dtype=object)


def prepare_features(
    ev,
    ohlcv: pd.DataFrame,
    *,
    method: str | None = None,
    method_params: dict[str, Any] | None = None,
    sticky_guard: bool | None = None,
) -> pd.DataFrame:
    """Build px features; optionally attach regime labels for `method`."""
    px = ev.build_px(ohlcv)
    px["as_of"] = pd.to_datetime(px["as_of"]).dt.normalize()
    close = px["$close"].astype(float)
    high = px["$high"].astype(float)
    low = px["$low"].astype(float)
    open_ = px["$open"].astype(float)
    px["MA5"] = close.rolling(5, min_periods=5).mean()
    px["MA10"] = close.rolling(10, min_periods=10).mean()
    px["is_green"] = close > open_
    px["is_red"] = close < open_
    px["ma_bull"] = px["MA20"] > px["MA60"]
    px["above_ma10"] = close > px["MA10"]
    px["above_ma20"] = close > px["MA20"]
    px["golden"] = (px["MA5"] > px["MA20"]) & (px["MA5"].shift(1) <= px["MA20"].shift(1))
    px["death"] = (px["MA5"] < px["MA20"]) & (px["MA5"].shift(1) >= px["MA20"].shift(1))
    px["ret1"] = close.pct_change(fill_method=None)
    px["close_eq_low"] = np.isclose(close, low, rtol=0, atol=1e-4)
    roll_lo = low.rolling(20, min_periods=5).min()
    roll_hi = high.rolling(20, min_periods=5).max()
    px["range_pos"] = (close - roll_lo) / (roll_hi - roll_lo).replace(0, np.nan)
    # vol percentile proxy (fast min-max rank of |ret| 5d over 120d)
    abs5 = close.pct_change(fill_method=None).abs().rolling(5, min_periods=3).mean()
    roll_lo_v = abs5.rolling(120, min_periods=40).min()
    roll_hi_v = abs5.rolling(120, min_periods=40).max()
    px["vol5_pct_120d"] = (abs5 - roll_lo_v) / (roll_hi_v - roll_lo_v).replace(0, np.nan)
    if method is not None:
        # sticky guard only for ensemble (B4); legacy B0 keeps raw method labels
        use_guard = (
            bool(sticky_guard)
            if sticky_guard is not None
            else (method == "ensemble")
        )
        if method in STATE_SPACE_METHODS:
            params = dict(method_params or {})
            frozen = dict(params.get("frozen_model") or {})
            fallback_method = str(
                params.get("fallback_method")
                or frozen.get("fallback_method")
                or DEGENERATE_FALLBACK_METHOD
            )
            fallback = ev.run_method(fallback_method, px)
            labels, diag = state_space_labels_or_fallback(
                px, frozen, fallback_labels=fallback
            )
            px["regime"] = labels
            diagnostics = diag.get("diagnostics")
            if diagnostics is not None:
                for col in diagnostics.columns:
                    if col != "regime":
                        px[col] = diagnostics[col].to_numpy()
            px["state_space_fallback"] = bool(diag.get("fallback", False))
            px["state_space_reliability"] = ",".join(
                str(x) for x in diag.get("flags", [])
            )
        else:
            px["regime"] = label_regime(
                ev, px, method, method_params, sticky_guard=use_guard
            )
    return px


def expanding_folds(
    n_train: int, n_folds: int = 3, min_train: int = 120, val_frac: float = 0.2
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window (train_idx, val_idx) folds inside [0, n_train)."""
    if n_train < min_train + 40:
        cut = max(min_train, int(n_train * 0.7))
        if cut >= n_train - 20:
            return [(np.arange(n_train), np.arange(n_train))]
        return [(np.arange(cut), np.arange(cut, n_train))]
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    val_len = max(40, int(n_train * val_frac / n_folds))
    for k in range(n_folds):
        val_end = n_train - (n_folds - 1 - k) * (val_len // 2)
        val_end = min(n_train, max(min_train + val_len, val_end))
        val_start = max(min_train, val_end - val_len)
        tr_end = val_start
        if tr_end < min_train:
            continue
        folds.append((np.arange(tr_end), np.arange(val_start, val_end)))
    if not folds:
        cut = int(n_train * 0.75)
        folds = [(np.arange(cut), np.arange(cut, n_train))]
    return folds


# Hold-up trading historically excluded ADX/prior60 (grind-miss). Per-code
# enter-up training may still pick them when their train score wins.
HOLD_UP_METHOD_EXCLUDE = frozenset()
# Legacy near-tie preference removed: each symbol's train score picks method+params.
HOLD_UP_TIEBREAK = ()
HOLD_UP_TIE_EPS = 0.05

# Per-symbol enter-up / label grids (train-only B0 joint search).
STRICT_P60_THR_GRID = (0.03, 0.06, 0.10)
STRICT_P20_VREV_GRID = (0.03, 0.06)
CONFIRM_GRID = (1, 3)
SMOOTH_GRID_B0 = (3, 5)
MIN_SEG_GRID_B0 = (5, 10)


def enter_up_param_grid(method: str) -> list[dict[str, Any]]:
    """Small per-method param grid for train-only B0 joint selection."""
    out: list[dict[str, Any]] = []
    if method == "ma_stack_strict":
        for p60 in STRICT_P60_THR_GRID:
            for p20 in STRICT_P20_VREV_GRID:
                for conf in CONFIRM_GRID:
                    for ms in MIN_SEG_GRID_B0:
                        out.append(
                            {
                                "p60_thr": float(p60),
                                "p20_vrev": float(p20),
                                "confirm": int(conf),
                                "min_seg": int(ms),
                            }
                        )
    elif method in ("ma_stack_hyst", "hybrid_ma_adx"):
        for conf in CONFIRM_GRID:
            for sm in SMOOTH_GRID_B0:
                for ms in MIN_SEG_GRID_B0:
                    out.append(
                        {
                            "confirm": int(conf),
                            "smooth": int(sm),
                            "min_seg": int(ms),
                        }
                    )
    elif method in ("ma_stack_struct", "dual_ma_cross"):
        for conf in CONFIRM_GRID:
            for sm in SMOOTH_GRID_B0:
                for ms in MIN_SEG_GRID_B0:
                    out.append(
                        {
                            "confirm": int(conf),
                            "confirm_enter": int(conf),
                            "confirm_leave": int(conf),
                            "smooth": int(sm),
                            "min_seg": int(ms),
                        }
                    )
    elif method == "adx_di":
        for thr in (18.0, 22.0):
            for sm in SMOOTH_GRID_B0:
                for ms in MIN_SEG_GRID_B0:
                    out.append(
                        {"adx_thr": float(thr), "smooth": int(sm), "min_seg": int(ms)}
                    )
    elif method == "prior60_band":
        for ms in MIN_SEG_GRID_B0:
            out.append({"min_seg": int(ms)})
    else:
        out.append({})
    # Always include default empty (= production defaults) as a candidate.
    if {} not in out:
        out.insert(0, {})
    return out


def select_regime_method_legacy(
    ev,
    px_train: pd.DataFrame,
    *,
    trade_mode: str | None = None,
) -> tuple[str, dict, dict]:
    """B0: per-symbol train argmax over method × enter-up params.

    Rejects range-collapse winners (pred range ≫ truth range) so methods like
    ``prior60_band`` cannot win by painting almost everything range
    (failure mode on SH520830 after chop-direction fix).

    When truth itself is degenerate (almost all range — typical of short-listed
    ETFs where swing truth finds no legs), agree-based scoring rewards
    do-nothing methods (e.g. SH589720 → prior60_band never enters up on the
    Jun-2026 rally). Fall back to ``ma_stack_hyst``.

    No pool-wide method tie-break: score alone decides. Params are frozen into
    ``method_params`` for OOS labeling.
    """
    close = px_train["$close"].to_numpy(float)
    truth = ev.retrospective_truth(close)
    truth_range = float((truth == "range").mean()) if len(truth) else 0.0
    truth_dir = (
        float(((truth == "up") | (truth == "down")).mean()) if len(truth) else 0.0
    )
    exclude = HOLD_UP_METHOD_EXCLUDE if trade_mode == TRADE_MODE_HOLD_UP else frozenset()

    # Default-params scores (for meta / backwards-compatible method_scores).
    scores: dict[str, float] = {}
    for name in ev.METHODS:
        if name in exclude:
            continue
        labs0 = ev.run_method(name, px_train)
        scores[name] = float(score_method(labs0, close, truth))

    degenerate = truth_range >= DEGENERATE_TRUTH_RANGE or truth_dir < 0.05
    if degenerate:
        fb = DEGENERATE_FALLBACK_METHOD
        return fb, {}, dict(
            scores=scores,
            score=float(scores.get(fb, 0.0)),
            vetoed_range_collapse=[],
            truth_range=truth_range,
            truth_dir=truth_dir,
            degenerate_truth_fallback=True,
            fallback_method=fb,
            hold_up_excluded=sorted(exclude),
            joint_param_search=False,
        )

    best_m: str | None = None
    best_p: dict[str, Any] = {}
    best_s = -1e18
    vetoed: list[str] = []
    grid_tried = 0
    grid_best_by_method: dict[str, dict[str, Any]] = {}

    for name in ev.METHODS:
        if name in exclude:
            continue
        local_best = -1e18
        local_p: dict[str, Any] = {}
        local_rf = 1.0
        local_df = 0.0
        for p in enter_up_param_grid(name):
            grid_tried += 1
            labs = ev.run_method(name, px_train, params=p)
            s = float(score_method(labs, close, truth))
            rf = float((labs == "range").mean()) if len(labs) else 1.0
            df_ = float(((labs == "up") | (labs == "down")).mean()) if len(labs) else 0.0
            # Same range-collapse vetoes as before, applied per (method, params).
            if rf >= 0.85 or (
                rf >= 0.75 and rf > truth_range + 0.35 and truth_range >= 0.30
            ):
                if name not in vetoed:
                    vetoed.append(name)
                continue
            if truth_dir >= 0.20 and df_ < truth_dir * 0.25 and rf >= 0.65:
                if name not in vetoed:
                    vetoed.append(name)
                continue
            if s > local_best:
                local_best = s
                local_p = dict(p)
                local_rf = rf
                local_df = df_
            if s > best_s:
                best_s = s
                best_m = name
                best_p = dict(p)
        if local_best > -1e17:
            grid_best_by_method[name] = {
                "score": local_best,
                "params": local_p,
                "range_frac": local_rf,
                "dir_frac": local_df,
            }
            scores[name] = float(local_best)

    if best_m is None:
        # All vetoed — fall back to default-params top score.
        if scores:
            best_m = max(scores, key=scores.get)
            best_s = float(scores[best_m])
            best_p = {}
        else:
            best_m, best_s, best_p = "ma_stack_strict", -1e18, {}

    return best_m, dict(best_p), dict(
        scores=scores,
        score=best_s,
        vetoed_range_collapse=vetoed,
        truth_range=truth_range,
        truth_dir=truth_dir,
        hold_up_excluded=sorted(exclude),
        hold_up_tiebreak=False,
        joint_param_search=True,
        grid_tried=grid_tried,
        grid_best_by_method=grid_best_by_method,
        method_params=dict(best_p),
    )



def select_regime_method_ensemble_cv(
    ev, px_train: pd.DataFrame
) -> tuple[str, dict, dict]:
    """B4: CV-weighted ensemble vote; low confidence → range; sticky-up guard."""
    close = px_train["$close"].to_numpy(float)
    truth_r = ev.retrospective_truth(close)
    truth_c = ev.causal_confirmed_swing_truth(close)
    folds = expanding_folds(len(px_train), n_folds=3)
    raw: dict[str, float] = {}
    for name in ev.METHODS:
        labs = apply_sticky_up_guard(ev.run_method(name, px_train), px_train)
        fold_scores: list[float] = []
        for _tr, va in folds:
            if len(va) < 15:
                continue
            fold_scores.append(
                float(
                    dual_truth_selection_score(
                        labs[va],
                        close[va],
                        truth_r[va],
                        truth_c[va],
                        eval_fn=ev.eval_labels,
                    )
                )
            )
        if fold_scores:
            arr = np.asarray(fold_scores, dtype=float)
            raw[name] = max(float(arr.mean() - 0.25 * arr.std(ddof=0)), 0.0)
        else:
            raw[name] = 0.0
    tot = sum(raw.values()) or 1.0
    weights = {k: v / tot for k, v in raw.items()}
    names = list(ev.METHODS.keys())
    mats = [
        apply_sticky_up_guard(ev.run_method(n, px_train), px_train) for n in names
    ]
    label_mat = np.stack(mats, axis=0)
    w = np.array([float(weights[n]) for n in names], dtype=float)
    conf_grid = (0.35, 0.40, 0.45, 0.50, 0.55)
    best_conf, best_s = 0.45, -1e18
    _tr, va_idx = folds[-1]
    for conf in conf_grid:
        voted = ensemble_vote(label_mat[:, va_idx], weights=w, conf_thr=conf)
        s = dual_truth_selection_score(
            voted,
            close[va_idx],
            truth_r[va_idx],
            truth_c[va_idx],
            eval_fn=ev.eval_labels,
        )
        if s > best_s:
            best_s, best_conf = s, conf
    params = dict(
        weights={n: float(weights[n]) for n in names},
        conf_thr=float(best_conf),
        member_methods=names,
    )
    return (
        "ensemble",
        params,
        dict(cv_raw=raw, score=float(best_s), truth_version=TRUTH_VERSION),
    )


def select_regime_method_state_space_nested_cv(
    ev,
    px_train: pd.DataFrame,
    *,
    trade_mode: str | None = None,
    prior_rate: float = 0.5,
    prior_strength: float = PRIOR_K,
    min_closed: int = 2,
) -> tuple[str, dict, dict]:
    """Select on development folds, gate once on the last chronological fold."""
    baseline, _, baseline_meta = select_regime_method_legacy(
        ev, px_train, trade_mode=trade_mode
    )
    candidates = candidate_grid()
    by_key = {str(c["key"]): c for c in candidates}
    folds = state_space_expanding_splits(len(px_train))
    if len(folds) < 2:
        return baseline, {}, {
            "score": float(baseline_meta.get("score") or 0.0),
            "accepted": False,
            "reason": "insufficient_nested_folds",
            "baseline": baseline,
        }
    fold_rows = evaluate_candidate_folds(
        ev, px_train, candidates, splits=folds, cost_per_side=COST_PER_SIDE
    )
    last_fold = int(fold_rows["fold"].max())
    development = fold_rows[fold_rows["fold"] < last_fold]
    acceptance = fold_rows[fold_rows["fold"] == last_fold]
    aggregate = aggregate_candidate_metrics(
        development,
        prior_rate=prior_rate,
        prior_strength=prior_strength,
    )
    winner_key, choice = choose_candidate(
        aggregate,
        baseline_candidate=baseline,
        min_closed=min_closed,
    )
    winner = acceptance[acceptance["candidate"] == winner_key]
    base = acceptance[acceptance["candidate"] == baseline]

    gate_reason = "candidate_is_baseline"
    gate_pass = winner_key == baseline
    if winner_key != baseline and len(winner) and len(base):
        w = winner.iloc[0]
        b = base.iloc[0]
        # Sparse last folds cannot establish superiority; retain production.
        gate_pass = bool(
            int(w["n_closed"]) > 0
            and float(w["mean_ret_net"]) >= float(b["mean_ret_net"])
            and float(w["edge"]) >= float(b["edge"])
            and float(w["max_drawdown"]) >= float(b["max_drawdown"]) - 0.02
        )
        gate_reason = (
            "acceptance_constraints_passed"
            if gate_pass
            else "acceptance_constraints_failed"
        )
    elif winner_key != baseline:
        gate_reason = "missing_acceptance_metrics"
    if not gate_pass:
        winner_key = baseline
    candidate = by_key.get(
        winner_key,
        {
            "key": baseline,
            "kind": "baseline",
            "method": baseline,
            "config": {},
        },
    )
    method, params = fit_final_candidate(ev, px_train, candidate)
    agg_records = aggregate.replace({np.nan: None}).to_dict(orient="records")
    return method, params, {
        "score": float(
            next(
                (
                    r["posterior_win"]
                    for r in agg_records
                    if r["candidate"] == winner_key
                ),
                0.0,
            )
            or 0.0
        ),
        "accepted": gate_pass,
        "reason": gate_reason,
        "baseline": baseline,
        "winner_candidate": winner_key,
        "prior_rate": prior_rate,
        "prior_strength": prior_strength,
        "fold_metrics": fold_rows.replace({np.nan: None}).to_dict(orient="records"),
        "candidate_metrics": agg_records,
        "baseline_meta": {
            k: v
            for k, v in baseline_meta.items()
            if k not in ("scores", "cv_raw")
        },
    }


def select_regime_method(
    ev,
    px_train: pd.DataFrame,
    *,
    mode: str = REGIME_SELECT_ENSEMBLE_CV,
    trade_mode: str | None = None,
) -> tuple[str, dict, dict]:
    if mode == REGIME_SELECT_LEGACY:
        return select_regime_method_legacy(ev, px_train, trade_mode=trade_mode)
    if mode == REGIME_SELECT_ENSEMBLE_CV:
        return select_regime_method_ensemble_cv(ev, px_train)
    if mode == REGIME_SELECT_STATE_SPACE_NESTED_CV:
        return select_regime_method_state_space_nested_cv(
            ev, px_train, trade_mode=trade_mode
        )
    raise ValueError(f"unknown regime select mode: {mode}")


def n_switches(labs: np.ndarray) -> int:
    return int(sum(1 for i in range(1, len(labs)) if labs[i] != labs[i - 1]))


def hold_up_compound(labs: np.ndarray, close: np.ndarray) -> tuple[float, int]:
    eq = 1.0
    pos = False
    entry = None
    n = 0
    for i in range(len(labs)):
        if (not pos) and labs[i] == "up":
            pos = True
            entry = i
        elif pos and labs[i] != "up":
            eq *= close[i] / close[entry]
            n += 1
            pos = False
    if pos:
        eq *= close[-1] / close[entry]
        n += 1
    return float(eq - 1), n


def score_method(labs: np.ndarray, close: np.ndarray, truth: np.ndarray) -> float:
    """Legacy B0 score: hold-up + agree + up_frac − switch penalty."""
    hu, _ = hold_up_compound(labs, close)
    agree = float((labs == truth).mean()) if len(labs) else 0.0
    up_frac = float((labs == "up").mean()) if len(labs) else 0.0
    sw = n_switches(labs)
    return 0.45 * hu + 0.25 * agree + 0.15 * up_frac - 0.015 * sw


def worst_up_drawdown(labs: np.ndarray, close: np.ndarray) -> float:
    """Worst peak-to-trough drawdown inside contiguous up segments (≤0)."""
    worst = 0.0
    i = 0
    n = len(labs)
    while i < n:
        if labs[i] != "up":
            i += 1
            continue
        j = i
        while j < n and labs[j] == "up":
            j += 1
        c = np.asarray(close[i:j], dtype=float)
        if len(c) >= 2:
            peak = np.maximum.accumulate(c)
            dd = float(((c / peak) - 1.0).min())
            worst = min(worst, dd)
        i = j
    return worst


def score_method_balanced(
    labs: np.ndarray,
    close: np.ndarray,
    truth: np.ndarray,
    *,
    eval_fn=None,
) -> float:
    """B1: macro-F1 + balanced_acc + switch_hit − false/delay − down-gap penalty."""
    if eval_fn is None:
        # lazy: caller should pass ev.eval_labels
        agree = float((labs == truth).mean()) if len(labs) else 0.0
        return agree
    m = eval_fn(truth, labs)
    mf1 = float(m.get("macro_f1") or 0.0)
    if not np.isfinite(mf1):
        mf1 = 0.0
    bal = float(m.get("balanced_acc") or 0.0)
    if not np.isfinite(bal):
        bal = 0.0
    hit = float(m.get("switch_hit") or 0.0)
    if not np.isfinite(hit):
        hit = 0.0
    false = float(m.get("switch_false") or 0.0)
    if not np.isfinite(false):
        false = 1.0
    delay = m.get("switch_delay_mean", 0.0)
    delay_pen = 0.0 if not np.isfinite(delay) else min(abs(float(delay)), 10) / 20.0
    pred_dn = float((labs == "down").mean()) if len(labs) else 0.0
    tru_dn = float((truth == "down").mean()) if len(truth) else 0.0
    down_gap = max(0.0, pred_dn - tru_dn)
    # risk: penalize sticky up
    wdd = worst_up_drawdown(labs, close)
    risk_pen = min(1.0, max(0.0, -wdd - 0.12) / 0.20)  # 0 at -12%, 1 at -32%
    return (
        0.40 * mf1
        + 0.25 * bal
        + 0.15 * hit
        - 0.10 * false
        - 0.05 * delay_pen
        - 0.15 * down_gap
        - 0.10 * risk_pen
    )


def dual_truth_selection_score(
    labs: np.ndarray,
    close: np.ndarray,
    truth_retro: np.ndarray,
    truth_causal: np.ndarray,
    *,
    eval_fn,
) -> float:
    """Default dual-truth weighted score for method selection (plan §2)."""
    mr = eval_fn(truth_retro, labs)
    mc = eval_fn(truth_causal, labs)
    mf1_r = float(mr.get("macro_f1") or 0.0)
    mf1_c = float(mc.get("macro_f1") or 0.0)
    if not np.isfinite(mf1_r):
        mf1_r = 0.0
    if not np.isfinite(mf1_c):
        mf1_c = 0.0
    hit = float(mr.get("switch_hit") or 0.0)
    if not np.isfinite(hit):
        hit = 0.0
    false = float(mr.get("switch_false") or 0.0)
    if not np.isfinite(false):
        false = 1.0
    delay = mr.get("switch_delay_mean", 0.0)
    delay_pen = 0.0 if not np.isfinite(delay) else min(abs(float(delay)), 10) / 20.0
    wdd = worst_up_drawdown(labs, close)
    risk = 1.0 - min(1.0, max(0.0, -wdd) / 0.25)  # 1 best (no dd), 0 at -25%
    # 40% retro F1 + 25% causal F1 + 15% switch hit − 10% false/delay + 10% risk
    return (
        0.40 * mf1_r
        + 0.25 * mf1_c
        + 0.15 * hit
        - 0.10 * (0.5 * false + 0.5 * delay_pen)
        + 0.10 * risk
    )


def ensemble_vote(
    label_matrix: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    conf_thr: float = 0.45,
) -> np.ndarray:
    """Weighted vote across methods (rows=methods, cols=days). Low conf → range."""
    k, n = label_matrix.shape
    if weights is None:
        weights = np.ones(k, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / max(weights.sum(), 1e-12)
    out = np.array(["range"] * n, dtype=object)
    labs = ("up", "down", "range")
    for j in range(n):
        scores = {lab: 0.0 for lab in labs}
        for i in range(k):
            scores[str(label_matrix[i, j])] = scores.get(str(label_matrix[i, j]), 0.0) + weights[i]
        best = max(labs, key=lambda lab: scores[lab])
        if scores[best] < conf_thr and best != "range":
            out[j] = "range"
        else:
            out[j] = best
    return out


def apply_sticky_up_guard(
    labs: np.ndarray,
    px: pd.DataFrame,
    *,
    below20_days: int = 3,
    drawdown: float = 0.18,
    reenter_guard: int = 2,
) -> np.ndarray:
    """Force up→range when structure soft-breaks (risk veto).

    Only soft-leave when price is **below MA20** and either:
      - sustained below for ``below20_days``, or
      - drawdown from peak-since-enter-up ≥ ``drawdown``.

    Pure peak-to-trough noise while still above MA20 (common on high-vol
    names like SH513310) must **not** punch 1-day range holes mid-uptrend.
    After a soft-leave, suppress up for ``reenter_guard`` bars.
    """
    out = np.asarray(labs, dtype=object).copy()
    close = px["$close"].to_numpy(float)
    ma20 = px["MA20"].to_numpy(float)
    below = 0
    peak = np.nan
    guard = 0
    for i in range(len(out)):
        c, m = close[i], ma20[i]
        above = np.isfinite(c) and np.isfinite(m) and c >= m
        if guard > 0:
            if out[i] == "up":
                out[i] = "range"
                if above:
                    guard -= 1
                # stay guarded while still weak / below MA20
                below = 0
                peak = np.nan
                continue
            guard = 0
        if out[i] != "up":
            below = 0
            peak = np.nan
            continue
        if np.isfinite(c):
            peak = c if (not np.isfinite(peak) or c > peak) else peak
        if np.isfinite(c) and np.isfinite(m) and c < m:
            below += 1
        else:
            below = 0
        dd = (c / peak - 1.0) if (np.isfinite(c) and np.isfinite(peak) and peak > 0) else 0.0
        # Structure break only: must be below MA20 (not DD-alone while extended).
        soft = below >= below20_days or (below >= 1 and dd <= -drawdown)
        if soft:
            out[i] = "range"
            below = 0
            peak = np.nan
            guard = max(int(reenter_guard), 0)
    return out


def detect_early_regime_marks(
    px: pd.DataFrame,
    *,
    move: float = EARLY_REGIME_MOVE,
    regimes: np.ndarray | list[str] | None = None,
) -> list[dict[str, Any]]:
    """Visual-only early warnings for regime enter/leave (does not change labels).

    Per conservative-label episode, fire at most one mark when price has moved
    ``move`` against the episode extremum while the official label has not yet
    flipped:

    - ``leave_up``   (X): still up, drawdown from peak ≥ move
    - ``leave_down`` (V): still down, bounce from trough ≥ move  (early off down)
    - ``enter_up``   (↑): in range, bounce from trough ≥ move
    - ``enter_down`` (↓): in range, drop from peak ≥ move
    """
    close = px["$close"].to_numpy(float)
    dates = (
        px["as_of"].dt.strftime("%Y-%m-%d").to_numpy()
        if "as_of" in px.columns
        else np.arange(len(px)).astype(str)
    )
    labs = (
        np.asarray(regimes, dtype=object)
        if regimes is not None
        else (
            px["regime"].to_numpy(dtype=object)
            if "regime" in px.columns
            else np.array(["range"] * len(px), dtype=object)
        )
    )
    marks: list[dict[str, Any]] = []
    prev: str | None = None
    peak = trough = np.nan
    marked_leave = False
    marked_enter_up = False
    marked_enter_down = False

    def _emit(i: int, kind: str, move_v: float, ref: float) -> None:
        labels = {
            "leave_up": ("X", "提前离上涨", "仍标上涨，高点回撤已达阈"),
            "leave_down": ("V", "提前离下跌", "仍标下跌，低点反弹已达阈"),
            "enter_up": ("↑", "提前进上涨", "仍标横盘，低点反弹已达阈"),
            "enter_down": ("↓", "提前进下跌", "仍标横盘，高点回撤已达阈"),
        }
        sym, title, tip = labels[kind]
        marks.append(
            dict(
                date=str(dates[i]),
                px=float(close[i]),
                kind=kind,
                symbol=sym,
                ref=float(ref),
                move=float(move_v),
                why=f"{title} {move_v:+.1%}(阈{move:.0%})｜{tip}｜非交易信号",
            )
        )

    for i in range(len(px)):
        c = close[i]
        lab = str(labs[i]) if i < len(labs) else "range"
        if lab not in ("up", "down", "range"):
            lab = "range"
        if lab != prev:
            prev = lab
            peak = c if np.isfinite(c) else np.nan
            trough = c if np.isfinite(c) else np.nan
            marked_leave = False
            marked_enter_up = False
            marked_enter_down = False
            continue
        if not np.isfinite(c):
            continue
        if lab == "up":
            if not np.isfinite(peak) or c > peak:
                peak = c
                marked_leave = False
            dd = c / peak - 1.0 if peak > 0 else 0.0
            if not marked_leave and dd <= -move:
                _emit(i, "leave_up", dd, peak)
                marked_leave = True
        elif lab == "down":
            if not np.isfinite(trough) or c < trough:
                trough = c
                marked_leave = False
            reb = c / trough - 1.0 if trough > 0 else 0.0
            if not marked_leave and reb >= move:
                _emit(i, "leave_down", reb, trough)
                marked_leave = True
        else:
            if not np.isfinite(peak) or c > peak:
                peak = c
            if not np.isfinite(trough) or c < trough:
                trough = c
            reb = c / trough - 1.0 if (np.isfinite(trough) and trough > 0) else 0.0
            drp = c / peak - 1.0 if (np.isfinite(peak) and peak > 0) else 0.0
            if not marked_enter_up and reb >= move:
                _emit(i, "enter_up", reb, trough)
                marked_enter_up = True
            if not marked_enter_down and drp <= -move:
                _emit(i, "enter_down", drp, peak)
                marked_enter_down = True
    return marks


def detect_aggressive_vreversal_marks(
    px: pd.DataFrame,
    *,
    bounce: float = EARLY_REGIME_MOVE,
    regimes: np.ndarray | list[str] | None = None,
) -> list[dict[str, Any]]:
    """Backward-compat: early leave-down / enter-up subset of ``detect_early_regime_marks``."""
    marks = detect_early_regime_marks(px, move=bounce, regimes=regimes)
    return [m for m in marks if m.get("kind") in ("leave_down", "enter_up")]


def _buy_mask_up(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    n = len(df)
    m = np.zeros(n, dtype=bool)
    why = np.array([""] * n, dtype=object)
    regime = df["regime"].to_numpy()
    in_up = regime == "up"
    if cfg.get("hold_enter"):
        # first day of up streak while previously not up — filled in simulator via enter_up
        return in_up, np.where(in_up, "进入上涨态", "")
    g = df["is_green"].to_numpy(bool)
    bull = df["ma_bull"].to_numpy(bool)
    gold = df["golden"].fillna(False).to_numpy(bool)
    red = df["is_red"].to_numpy(bool)
    a20 = df["above_ma20"].to_numpy(bool)
    vol = df["vol5_pct_120d"].to_numpy(float)
    used = np.zeros(n, dtype=bool)
    if cfg.get("green_bull"):
        x = g & bull & in_up & ~used
        why[x] = "绿牛"
        used |= x
        m |= x
    if cfg.get("golden"):
        x = gold & in_up & ~used
        why[x] = "金叉"
        used |= x
        m |= x
    if cfg.get("red_ma20"):
        x = red & a20 & in_up & ~used
        why[x] = "红上MA20"
        used |= x
        m |= x
    if cfg.get("vol_min") is not None:
        ok = np.isfinite(vol) & (vol >= float(cfg["vol_min"]))
        m &= ok
        why = np.where(m, why, "")
    return m, why


def _buy_mask_down(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    n = len(df)
    m = np.zeros(n, dtype=bool)
    why = np.array([""] * n, dtype=object)
    in_dn = df["regime"].to_numpy() == "down"
    red = df["is_red"].to_numpy(bool)
    ret = df["ret1"].to_numpy(float)
    fl = df["close_eq_low"].to_numpy(bool)
    rp = df["range_pos"].to_numpy(float)
    if cfg.get("floor"):
        x = red & (ret <= -0.07) & fl & in_dn
        why[x] = "红地板"
        m |= x
    if cfg.get("deep"):
        thr = float(cfg.get("thr", -0.05))
        x = red & (ret <= thr) & in_dn & ~m
        why[x] = "深红"
        m |= x
    if cfg.get("near_low"):
        thr = float(cfg.get("thr", 0.2))
        x = red & np.isfinite(rp) & (rp <= thr) & in_dn & ~m
        why[x] = "红近低"
        m |= x
    return m, why


def _buy_mask_range(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    n = len(df)
    m = np.zeros(n, dtype=bool)
    why = np.array([""] * n, dtype=object)
    in_rg = df["regime"].to_numpy() == "range"
    red = df["is_red"].to_numpy(bool)
    g = df["is_green"].to_numpy(bool)
    bull = df["ma_bull"].to_numpy(bool)
    gold = df["golden"].fillna(False).to_numpy(bool)
    rp = df["range_pos"].to_numpy(float)
    used = np.zeros(n, dtype=bool)
    if cfg.get("near_low"):
        thr = float(cfg.get("thr", 0.25))
        x = red & np.isfinite(rp) & (rp <= thr) & in_rg & ~used
        why[x] = "红近低"
        used |= x
        m |= x
    if cfg.get("green_bull"):
        x = g & bull & in_rg & ~used
        why[x] = "绿牛"
        used |= x
        m |= x
    if cfg.get("golden"):
        x = gold & in_rg & ~used
        why[x] = "金叉"
        used |= x
        m |= x
    return m, why


def _find_buy_spec(stage: str, name: str) -> dict[str, Any]:
    if name == NO_TRADE_BUY:
        return dict(name=NO_TRADE_BUY, no_trade=True)
    lib = {"up": UP_BUY_SPECS, "down": DOWN_BUY_SPECS, "range": RANGE_BUY_SPECS}[stage]
    for s in lib:
        if s["name"] == name:
            return s
    raise KeyError(f"buy spec {stage}/{name}")


def _find_exit_spec(stage: str, name: str) -> dict[str, Any]:
    if name == NO_TRADE_EXIT:
        return dict(name=NO_TRADE_EXIT, mode="none")
    lib = {"up": UP_EXIT_SPECS, "down": DOWN_EXIT_SPECS, "range": RANGE_EXIT_SPECS}[stage]
    for s in lib:
        if s["name"] == name:
            return s
    raise KeyError(f"exit spec {stage}/{name}")


def is_no_trade_rule(buy: str, exit_: str | None = None) -> bool:
    return buy == NO_TRADE_BUY or (exit_ is not None and exit_ == NO_TRADE_EXIT)


def _check_exit(
    *,
    exit_spec: dict[str, Any],
    entry_rg: str,
    rg: str,
    c: float,
    h: float,
    entry: float,
    peak: float,
    hold: int,
    is_red: bool,
    above10: bool,
    above20: bool,
    death: bool,
) -> tuple[str | None, float]:
    mode = exit_spec["mode"]
    px_e = c
    why = None
    if mode == "leave_up":
        if rg != "up":
            why = f"离上涨→{rg}"
    elif mode == "red_ma10":
        if is_red and (not above10):
            why = "红破MA10"
        elif entry_rg == "up" and rg != "up":
            why = f"离上涨→{rg}"
    elif mode == "red_ma20":
        if is_red and (not above20):
            why = "红破MA20"
        elif entry_rg == "up" and rg != "up":
            why = f"离上涨→{rg}"
    elif mode == "break_ma20":
        if not above20:
            why = "破MA20"
        elif entry_rg == "up" and rg != "up":
            why = f"离上涨→{rg}"
    elif mode == "death":
        if death:
            why = "死叉"
        elif entry_rg == "up" and rg != "up":
            why = f"离上涨→{rg}"
    elif mode == "trail":
        trail = float(exit_spec["trail"])
        if c <= peak * (1 - trail):
            why = f"trail{int(trail * 100)}"
        elif entry_rg == "up" and rg != "up":
            why = f"离上涨→{rg}"
    elif mode == "short":
        tp = float(exit_spec["tp"])
        sl = float(exit_spec["sl"])
        mh = int(exit_spec["max_hold"])
        if (h / entry - 1) >= tp:
            why = f"tp{int(tp * 100)}"
            px_e = entry * (1 + tp)
        elif (c / entry - 1) <= sl:
            why = f"sl{int(abs(sl) * 100)}"
        elif hold >= mh:
            why = f"hold{mh}"
    return why, px_e


def simulate_stage_rule(
    df: pd.DataFrame,
    *,
    stage: str,
    buy_name: str,
    exit_name: str,
    restrict_regime: bool = True,
    cost_per_side: float = COST_PER_SIDE,
) -> tuple[list[dict], dict]:
    """Backtest one (buy, exit) pair only inside `stage` segments (concatenated wealth)."""
    if is_no_trade_rule(buy_name, exit_name):
        return [], dict(
            stage=stage,
            buy=NO_TRADE_BUY,
            exit=NO_TRADE_EXIT,
            n_trades=0,
            win=float("nan"),
            compound=0.0,
            bh_segments=0.0,
            edge=0.0,
            mean_ret=0.0,
            mean_ret_net=0.0,
            sum_ret=0.0,
        )

    buy_spec = _find_buy_spec(stage, buy_name)
    exit_spec = _find_exit_spec(stage, exit_name)
    hold_enter = bool(buy_spec.get("hold_enter"))

    if stage == "up":
        buy_m, buy_why = _buy_mask_up(df, buy_spec)
    elif stage == "down":
        buy_m, buy_why = _buy_mask_down(df, buy_spec)
    else:
        buy_m, buy_why = _buy_mask_range(df, buy_spec)

    close = df["$close"].to_numpy(float)
    high = df["$high"].to_numpy(float)
    dates = df["as_of"].dt.strftime("%Y-%m-%d").to_numpy()
    regime = df["regime"].to_numpy()
    is_red = df["is_red"].to_numpy(bool)
    above10 = df["above_ma10"].to_numpy(bool)
    above20 = df["above_ma20"].to_numpy(bool)
    death = df["death"].fillna(False).to_numpy(bool)

    # Segment-wise compounding within stage only
    seg_df = segments(regime, dates, close)
    segs = seg_df[seg_df.regime == stage] if restrict_regime else seg_df
    wealth = 1.0
    all_ev: list[dict] = []
    n_trades = 0
    wins = 0
    sum_ret = 0.0
    bh_comp = 1.0

    for _, sg in segs.iterrows():
        mask = (df["as_of"] >= sg["start"]) & (df["as_of"] <= sg["end"])
        sub_idx = np.where(mask.to_numpy())[0]
        if len(sub_idx) < 2:
            continue
        bh_comp *= float(close[sub_idx[-1]] / close[sub_idx[0]])
        pos = 0.0
        entry = None
        ei = None
        peak = None
        prev_rg = None
        for j, i in enumerate(sub_idx):
            rg = regime[i]
            c = close[i]
            h = high[i]
            if pos == 0:
                enter = False
                ewhy = ""
                if hold_enter and stage == "up":
                    if j == 0 or (prev_rg != "up" and rg == "up"):
                        enter = True
                        ewhy = "进入上涨态"
                elif buy_m[i]:
                    enter = True
                    ewhy = str(buy_why[i])
                if enter:
                    entry = c
                    ei = i
                    peak = h
                    pos = 1.0  # unit notional per trade; compound via wealth
                    all_ev.append(
                        dict(side="BUY", date=dates[i], px=c, why=f"{stage}:{ewhy}", regime=stage)
                    )
                prev_rg = rg
                continue
            peak = max(peak, h)
            hold = i - ei
            if hold <= 0:
                prev_rg = rg
                continue
            why, px_e = _check_exit(
                exit_spec=exit_spec,
                entry_rg=stage,
                rg=rg,
                c=c,
                h=h,
                entry=entry,
                peak=peak,
                hold=hold,
                is_red=is_red[i],
                above10=above10[i],
                above20=above20[i],
                death=death[i],
            )
            if why:
                ret = px_e / entry - 1
                all_ev.append(
                    dict(
                        side="SELL",
                        date=dates[i],
                        px=px_e,
                        why=why,
                        entry_date=dates[ei],
                        fwd_ret=ret,
                        regime=stage,
                    )
                )
                n_trades += 1
                wins += int(ret > 0)
                sum_ret += ret
                wealth *= 1.0 + ret
                pos = 0.0
                entry = ei = peak = None
            prev_rg = rg
        if pos > 0:
            px_e = close[sub_idx[-1]]
            ret = px_e / entry - 1
            all_ev.append(
                dict(
                    side="SELL",
                    date=dates[sub_idx[-1]],
                    px=px_e,
                    why="段末强平",
                    entry_date=dates[ei],
                    fwd_ret=ret,
                    regime=stage,
                )
            )
            n_trades += 1
            wins += int(ret > 0)
            sum_ret += ret
            wealth *= 1.0 + ret

    compound = float(wealth - 1)
    mean_ret = float(sum_ret / n_trades) if n_trades else 0.0
    # round-trip cost = 2 * one-way
    mean_ret_net = mean_ret - 2.0 * float(cost_per_side) if n_trades else 0.0
    return all_ev, dict(
        stage=stage,
        buy=buy_name,
        exit=exit_name,
        n_trades=n_trades,
        win=(wins / n_trades) if n_trades else float("nan"),
        compound=compound,
        bh_segments=float(bh_comp - 1),
        edge=compound - float(bh_comp - 1),
        mean_ret=mean_ret,
        mean_ret_net=mean_ret_net,
        sum_ret=float(sum_ret),
    )


def simulate_adaptive_book(
    df: pd.DataFrame,
    *,
    rules: dict[str, dict[str, str]],
) -> tuple[list[dict], dict, dict | None]:
    """Single-capital adaptive book across regimes using frozen per-stage rules.

    rules = {up:{buy,exit}, down:{...}, range:{...}}
    Never emits 期末 sells; open marked separately.
    """
    close = df["$close"].to_numpy(float)
    high = df["$high"].to_numpy(float)
    dates = df["as_of"].dt.strftime("%Y-%m-%d").to_numpy()
    regime = df["regime"].to_numpy()
    is_red = df["is_red"].to_numpy(bool)
    above10 = df["above_ma10"].to_numpy(bool)
    above20 = df["above_ma20"].to_numpy(bool)
    death = df["death"].fillna(False).to_numpy(bool)
    bh = float(close[-1] / close[0] - 1) if len(close) > 1 else 0.0

    masks: dict[str, tuple[np.ndarray, np.ndarray, dict] | None] = {}
    for stage in ("up", "down", "range"):
        bname = rules[stage]["buy"]
        ename = rules[stage]["exit"]
        if is_no_trade_rule(bname, ename):
            masks[stage] = None
            continue
        bspec = _find_buy_spec(stage, bname)
        espec = _find_exit_spec(stage, ename)
        if stage == "up":
            bm, bw = _buy_mask_up(df, bspec)
        elif stage == "down":
            bm, bw = _buy_mask_down(df, bspec)
        else:
            bm, bw = _buy_mask_range(df, bspec)
        masks[stage] = (bm, bw, espec)

    cash = 1.0
    pos = 0.0
    entry = None
    ei = None
    peak = None
    entry_rg = ""
    exit_spec: dict | None = None
    aev: list[dict] = []
    open_mtm = None
    prev_rg = None

    for i in range(len(df)):
        rg = str(regime[i])
        c = close[i]
        h = high[i]
        if pos == 0:
            if rg in masks and masks[rg] is not None:
                bm, bw, espec = masks[rg]  # type: ignore[misc]
                hold_enter = rules[rg]["buy"] == "进入上涨态"
                enter = False
                ewhy = ""
                if hold_enter and rg == "up":
                    if prev_rg != "up":
                        enter = True
                        ewhy = "进入上涨态"
                elif bm[i]:
                    enter = True
                    ewhy = str(bw[i])
                if enter:
                    entry = c
                    ei = i
                    peak = h
                    entry_rg = rg
                    exit_spec = espec
                    pos = cash / c
                    cash = 0.0
                    aev.append(dict(side="BUY", date=dates[i], px=c, why=f"{rg}:{ewhy}", regime=rg))
            prev_rg = rg
            continue

        peak = max(peak, h)
        hold = i - ei
        if hold <= 0:
            prev_rg = rg
            continue
        assert exit_spec is not None and entry is not None
        why, px_e = _check_exit(
            exit_spec=exit_spec,
            entry_rg=entry_rg,
            rg=rg,
            c=c,
            h=h,
            entry=entry,
            peak=peak,
            hold=hold,
            is_red=is_red[i],
            above10=above10[i],
            above20=above20[i],
            death=death[i],
        )
        if why:
            cash = pos * px_e
            aev.append(
                dict(
                    side="SELL",
                    date=dates[i],
                    px=px_e,
                    why=why,
                    entry_date=dates[ei],
                    fwd_ret=px_e / entry - 1,
                    regime=entry_rg,
                )
            )
            pos = 0.0
            entry = ei = peak = None
            entry_rg = ""
            exit_spec = None
        prev_rg = rg

    if pos > 0:
        open_mtm = dict(
            entry_date=dates[ei],
            entry_px=float(entry),
            regime=entry_rg,
            last_date=dates[-1],
            last_px=float(close[-1]),
            unrealized=float(close[-1] / entry - 1),
        )
        equity = cash + pos * close[-1]
    else:
        equity = cash

    sells = [e for e in aev if e["side"] == "SELL"]
    stats = dict(
        compound=float(equity - 1),
        bh=bh,
        edge=float(equity - 1) - bh,
        n=len(sells),
        win=float(np.mean([e["fwd_ret"] > 0 for e in sells])) if sells else float("nan"),
        open_pos=bool(open_mtm),
        open_unrealized=(open_mtm["unrealized"] if open_mtm else None),
    )
    return aev, stats, open_mtm


def sweep_stage_rules(
    df: pd.DataFrame,
    stage: str,
    *,
    include_no_trade: bool = False,
    cost_per_side: float = COST_PER_SIDE,
) -> pd.DataFrame:
    buys = {"up": UP_BUY_SPECS, "down": DOWN_BUY_SPECS, "range": RANGE_BUY_SPECS}[stage]
    exits = {"up": UP_EXIT_SPECS, "down": DOWN_EXIT_SPECS, "range": RANGE_EXIT_SPECS}[stage]
    rows = []
    for b in buys:
        for e in exits:
            # hold_enter only pairs with leave_up or structural exits
            if b.get("hold_enter") and e["mode"] not in {
                "leave_up",
                "red_ma10",
                "red_ma20",
                "break_ma20",
                "death",
                "trail",
            }:
                continue
            if (not b.get("hold_enter")) and e["mode"] == "leave_up":
                continue
            _, sm = simulate_stage_rule(
                df,
                stage=stage,
                buy_name=b["name"],
                exit_name=e["name"],
                cost_per_side=cost_per_side,
            )
            rows.append(sm)
    if include_no_trade:
        _, sm = simulate_stage_rule(
            df,
            stage=stage,
            buy_name=NO_TRADE_BUY,
            exit_name=NO_TRADE_EXIT,
            cost_per_side=cost_per_side,
        )
        rows.append(sm)
    return pd.DataFrame(rows)


def bayes_posterior(own: float, n: int, prior: float, k: float = PRIOR_K) -> float:
    n = max(0, int(n))
    return float((n * own + k * prior) / (n + k))


def rule_own_score(
    row: dict | pd.Series,
    *,
    score_field: str = "compound",
) -> float:
    """Extract own score for Bayes selection from a sweep row."""
    if score_field == "compound":
        v = row.get("compound") if isinstance(row, dict) else row["compound"]
        return float(v) if pd.notna(v) else -1.0
    if score_field in ("mean_ret", "mean_ret_net"):
        v = row.get(score_field) if isinstance(row, dict) else row[score_field]
        if pd.isna(v):
            return 0.0
        return float(v)
    raise KeyError(f"unknown score_field {score_field}")


def select_stage_rules_bayes(
    *,
    own_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    k: float = PRIOR_K,
    score_field: str = "compound",
    include_no_trade: bool = False,
    min_trades: int = 1,
) -> dict[str, dict[str, Any]]:
    """Pick per-stage rules by posterior. own_df columns include stage/buy/exit/n_trades/scores."""
    prior_map: dict[str, float] = {}
    for _, r in prior_df.iterrows():
        prior_map[f"{r['stage']}|{r['buy']}|{r['exit']}"] = float(r["prior_score"])

    def _no_trade_pick(source: str = "no_trade") -> dict[str, Any]:
        return dict(
            buy=NO_TRADE_BUY,
            exit=NO_TRADE_EXIT,
            own_score=0.0,
            prior_score=0.0,
            posterior=0.0,
            n_trades=0,
            source=source,
        )

    selected: dict[str, dict[str, Any]] = {}
    for stage in ("up", "down", "range"):
        stage_prior = prior_df[prior_df.stage == stage].copy()
        # drop NO_TRADE from "best trading prior" look-ups
        stage_prior_trade = stage_prior[stage_prior.buy != NO_TRADE_BUY]

        if stage_prior_trade.empty and not include_no_trade:
            fb = FIXED_HYBRID_RULES[stage]
            selected[stage] = dict(
                buy=fb["buy"],
                exit=fb["exit"],
                own_score=None,
                prior_score=None,
                posterior=None,
                n_trades=0,
                source="fixed_fallback",
            )
            continue

        if own_df is None or own_df.empty:
            if include_no_trade:
                if stage_prior_trade.empty or float(stage_prior_trade.iloc[0]["prior_score"]) <= 0:
                    selected[stage] = _no_trade_pick("prior_only_no_trade")
                    continue
            if stage_prior_trade.empty:
                selected[stage] = _no_trade_pick("prior_only_no_trade") if include_no_trade else dict(
                    buy=FIXED_HYBRID_RULES[stage]["buy"],
                    exit=FIXED_HYBRID_RULES[stage]["exit"],
                    own_score=None,
                    prior_score=None,
                    posterior=None,
                    n_trades=0,
                    source="fixed_fallback",
                )
                continue
            top = stage_prior_trade.iloc[0]
            selected[stage] = dict(
                buy=top["buy"],
                exit=top["exit"],
                own_score=None,
                prior_score=float(top["prior_score"]),
                posterior=float(top["prior_score"]),
                n_trades=0,
                source="prior_only",
            )
            continue

        own_s = own_df[own_df.stage == stage].copy()
        # only rules that actually traded (plus optional NO_TRADE)
        traded = own_s[
            (own_s.buy != NO_TRADE_BUY) & (own_s.n_trades.fillna(0).astype(int) >= min_trades)
        ]
        candidates = traded
        if include_no_trade:
            nt_row = pd.DataFrame(
                [
                    dict(
                        stage=stage,
                        buy=NO_TRADE_BUY,
                        exit=NO_TRADE_EXIT,
                        n_trades=0,
                        compound=0.0,
                        mean_ret=0.0,
                        mean_ret_net=0.0,
                        edge=0.0,
                        win=np.nan,
                    )
                ]
            )
            candidates = pd.concat([traded, nt_row], ignore_index=True)

        if candidates.empty:
            selected[stage] = (
                _no_trade_pick("no_candidates")
                if include_no_trade
                else dict(
                    buy=FIXED_HYBRID_RULES[stage]["buy"],
                    exit=FIXED_HYBRID_RULES[stage]["exit"],
                    own_score=None,
                    prior_score=None,
                    posterior=None,
                    n_trades=0,
                    source="fixed_fallback",
                )
            )
            continue

        best = None
        best_post = -1e9
        prior_default = (
            float(stage_prior_trade["prior_score"].mean()) if len(stage_prior_trade) else 0.0
        )
        for _, r in candidates.iterrows():
            key = f"{stage}|{r['buy']}|{r['exit']}"
            if is_no_trade_rule(str(r["buy"]), str(r["exit"])):
                prior = 0.0
                own = 0.0
                n = 0
                post = 0.0  # hurdle: trading rules must beat 0
            else:
                prior = prior_map.get(key, prior_default)
                own = rule_own_score(r, score_field=score_field)
                n = int(r["n_trades"])
                post = bayes_posterior(own, n, prior, k=k)
            if post > best_post:
                best_post = post
                best = dict(
                    buy=r["buy"],
                    exit=r["exit"],
                    own_score=own,
                    prior_score=prior,
                    posterior=post,
                    n_trades=n,
                    source="posterior",
                )

        # expectation-negative → empty if allowed
        if include_no_trade and (best is None or best_post < 0):
            selected[stage] = _no_trade_pick("negative_posterior")
        elif best is None:
            top = stage_prior_trade.iloc[0] if len(stage_prior_trade) else None
            if top is not None:
                selected[stage] = dict(
                    buy=top["buy"],
                    exit=top["exit"],
                    own_score=None,
                    prior_score=float(top["prior_score"]),
                    posterior=float(top["prior_score"]),
                    n_trades=0,
                    source="prior_only",
                )
            else:
                selected[stage] = _no_trade_pick("no_candidates") if include_no_trade else dict(
                    buy=FIXED_HYBRID_RULES[stage]["buy"],
                    exit=FIXED_HYBRID_RULES[stage]["exit"],
                    own_score=None,
                    prior_score=None,
                    posterior=None,
                    n_trades=0,
                    source="fixed_fallback",
                )
        else:
            selected[stage] = best
    return selected


def build_pool_prior_from_rows(
    all_rule_rows: list[dict],
    *,
    score_field: str = "compound",
) -> pd.DataFrame:
    if not all_rule_rows:
        return pd.DataFrame(columns=["stage", "buy", "exit", "prior_score", "n_symbols", "mean_n"])
    df = pd.DataFrame(all_rule_rows)
    if score_field not in df.columns:
        score_field = "compound" if "compound" in df.columns else list(df.columns)[0]
    # exclude zero-trade rows from prior (they inflate mean_ret_net toward 0/noise);
    # keep NO_TRADE rows as explicit 0 anchors
    is_nt = df["buy"].astype(str) == NO_TRADE_BUY
    usable = df[is_nt | (df["n_trades"].fillna(0).astype(int) > 0)].copy()
    if usable.empty:
        usable = df.copy()
    fill = 0.0 if score_field != "compound" else -1.0
    usable["score"] = usable[score_field].fillna(fill)
    return (
        usable.groupby(["stage", "buy", "exit"], as_index=False)
        .agg(
            prior_score=("score", "mean"),
            n_symbols=("code", "nunique"),
            mean_n=("n_trades", "mean"),
        )
        .sort_values(["stage", "prior_score"], ascending=[True, False])
    )


def _apply_title_top_margin(
    fig: go.Figure,
    *,
    min_t: int = 110,
    px_per_line: int = 20,
    pad: int = 40,
) -> go.Figure:
    """Grow top margin so multi-line HTML titles stay above the price panel.

    Long lines without ``<br>`` still wrap in the browser; treat each ~56 chars
    as an extra visual line so wrapped text does not spill into candles.
    """
    text = ""
    if fig.layout.title is not None and fig.layout.title.text:
        text = str(fig.layout.title.text)
    if not text:
        return fig
    plain = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    plain = re.sub(r"<[^>]+>", "", plain)
    visual_lines = 0
    for line in plain.split("\n"):
        line = line.strip()
        if not line:
            visual_lines += 1
            continue
        # CJK titles wrap earlier than Latin; ~56 chars ≈ one visual line.
        visual_lines += max(1, (len(line) + 55) // 56)
    need = max(int(min_t), int(pad + visual_lines * px_per_line))
    cur = fig.layout.margin.to_plotly_json() if fig.layout.margin else {}
    old_t = int(cur.get("t") or min_t)
    if need <= old_t:
        return fig
    delta = need - old_t
    cur["t"] = need
    updates: dict = {"margin": cur}
    height = fig.layout.height
    if height:
        updates["height"] = int(height) + delta
    fig.update_layout(**updates)
    return fig


def annotate_figure(
    *,
    fig: go.Figure,
    ohlcv: pd.DataFrame,
    seg_df: pd.DataFrame,
    aev: list[dict],
    open_mtm: dict | None,
    stats: dict,
    start_date: str,
    end_date: str,
    title_note: str,
    aggressive_marks: list[dict] | None = None,
    early_down_marks: list[dict] | None = None,
    recovery_up_marks: list[dict] | None = None,
) -> go.Figure:
    vol_legend = {VOL_REGIME_CLUSTER, VOL_REGIME_EXTREME}
    fig.data = tuple(t for t in fig.data if getattr(t, "name", None) not in vol_legend)

    ohlcv_w = ohlcv[
        (pd.to_datetime(ohlcv["datetime"]) >= pd.Timestamp(start_date))
        & (pd.to_datetime(ohlcv["datetime"]) <= pd.Timestamp(end_date))
    ].copy()
    ohlcv_w["as_of"] = pd.to_datetime(ohlcv_w["datetime"]).dt.normalize()
    pxmap = ohlcv_w.set_index("as_of")
    lo = float(pd.to_numeric(ohlcv_w["$low"], errors="coerce").min())
    hi = float(pd.to_numeric(ohlcv_w["$high"], errors="coerce").max())
    span = max(hi - lo, 1e-6)
    buy_off = span * 0.035

    color_map = {
        "up": "rgba(18,161,80,0.12)",
        "down": "rgba(214,39,40,0.10)",
        "range": "rgba(148,148,148,0.14)",
    }
    shapes = [
        dict(
            type="rect",
            xref="x",
            yref="y",
            x0=srow["start"],
            x1=srow["end"],
            y0=lo - span * 0.02,
            y1=hi + span * 0.05,
            fillcolor=color_map.get(srow["regime"], "rgba(0,0,0,0.05)"),
            line=dict(width=0),
            layer="below",
        )
        for _, srow in seg_df.iterrows()
    ]
    fig.update_layout(shapes=shapes)
    for name, color in (
        ("上涨态", "rgba(18,161,80,0.55)"),
        ("下跌态", "rgba(214,39,40,0.55)"),
        ("横盘态", "rgba(120,120,120,0.55)"),
    ):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color=color, symbol="square"),
                name=name,
            ),
            row=1,
            col=1,
        )

    def y_buy(d: str) -> float | None:
        ts = pd.Timestamp(d)
        return None if ts not in pxmap.index else float(pxmap.loc[ts, "$low"]) - buy_off

    def y_sell(d: str) -> float | None:
        ts = pd.Timestamp(d)
        return None if ts not in pxmap.index else float(pxmap.loc[ts, "$high"]) + buy_off

    style = {
        "up": dict(bc="#12a150", bl="#064d1f", sc="#d62728", sl="#6b1010", bn="上涨买", sn="上涨卖"),
        "down": dict(bc="#2ca02c", bl="#145214", sc="#ff7f0e", sl="#8c4500", bn="下跌买", sn="下跌卖"),
        "range": dict(bc="#1f77b4", bl="#0b3d5c", sc="#9467bd", sl="#4b2e66", bn="横盘买", sn="横盘卖"),
    }
    tr = pd.DataFrame(aev)
    if not tr.empty:
        for rg, st in style.items():
            sub = tr[tr.regime.astype(str) == rg]
            if sub.empty:
                continue
            bx, by, bt = [], [], []
            for _, r in sub[sub.side == "BUY"].iterrows():
                y = y_buy(r["date"])
                if y is None:
                    continue
                bx.append(r["date"])
                by.append(y)
                bt.append(f"买 {r['date']}<br>{r['why']}<br>价={float(r['px']):.3f}")
            sx, sy, stxt = [], [], []
            for _, r in sub[sub.side == "SELL"].iterrows():
                y = y_sell(r["date"])
                if y is None:
                    continue
                ret = r.get("fwd_ret", np.nan)
                ret_s = f"<br>收益={float(ret):+.1%}" if pd.notna(ret) else ""
                sx.append(r["date"])
                sy.append(y)
                stxt.append(f"卖 {r['date']}<br>{r['why']}<br>价={float(r['px']):.3f}{ret_s}")
            if bx:
                fig.add_trace(
                    go.Scatter(
                        x=bx,
                        y=by,
                        mode="markers+text",
                        name=st["bn"],
                        text=["B"] * len(bx),
                        textposition="bottom center",
                        textfont=dict(size=11, color=st["bl"], family="Arial Black"),
                        marker=dict(
                            symbol="triangle-up",
                            size=14,
                            color=st["bc"],
                            line=dict(width=1.2, color=st["bl"]),
                        ),
                        hovertext=bt,
                        hoverinfo="text",
                    ),
                    row=1,
                    col=1,
                )
            if sx:
                fig.add_trace(
                    go.Scatter(
                        x=sx,
                        y=sy,
                        mode="markers+text",
                        name=st["sn"],
                        text=["S"] * len(sx),
                        textposition="top center",
                        textfont=dict(size=11, color=st["sl"], family="Arial Black"),
                        marker=dict(
                            symbol="triangle-down",
                            size=14,
                            color=st["sc"],
                            line=dict(width=1.2, color=st["sl"]),
                        ),
                        hovertext=stxt,
                        hoverinfo="text",
                    ),
                    row=1,
                    col=1,
                )

    if open_mtm is not None:
        fig.add_trace(
            go.Scatter(
                x=[open_mtm["last_date"]],
                y=[float(open_mtm["last_px"])],
                mode="markers+text",
                name="持仓未平",
                text=["持"],
                textposition="top center",
                textfont=dict(size=11, color="#333", family="Arial Black"),
                marker=dict(
                    symbol="diamond",
                    size=12,
                    color="#f0ad4e",
                    line=dict(width=1.2, color="#8a6d3b"),
                ),
                hovertext=[
                    f"持仓未平<br>买入 {open_mtm['entry_date']} @ {open_mtm['entry_px']:.3f}"
                    f"<br>状态={open_mtm['regime']}<br>浮盈={open_mtm['unrealized']:+.1%}"
                ],
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )

    if aggressive_marks:
        # Visual-only early regime warnings (by kind).
        styles = {
            "leave_down": dict(
                name="提前离↓(V)",
                text="V",
                color="#e67e22",
                line="#8a4b08",
                pos="bottom center",
                y="buy",
            ),
            "enter_up": dict(
                name="提前进↑",
                text="↑",
                color="#27ae60",
                line="#145a32",
                pos="bottom center",
                y="buy",
            ),
            "leave_up": dict(
                name="提前离↑(X)",
                text="X",
                color="#8e44ad",
                line="#4a235a",
                pos="top center",
                y="sell",
            ),
            "enter_down": dict(
                name="提前进↓",
                text="↓",
                color="#c0392b",
                line="#7b241c",
                pos="top center",
                y="sell",
            ),
        }
        by_kind: dict[str, list[dict]] = {}
        for m in aggressive_marks:
            d = str(m.get("date", ""))
            ts = pd.Timestamp(d)
            if ts < pd.Timestamp(start_date) or ts > pd.Timestamp(end_date):
                continue
            kind = str(m.get("kind") or "leave_down")
            by_kind.setdefault(kind, []).append(m)
        for kind, rows in by_kind.items():
            st = styles.get(kind) or styles["leave_down"]
            ax, ay, at, txts = [], [], [], []
            for m in rows:
                d = str(m["date"])
                y = y_buy(d) if st["y"] == "buy" else y_sell(d)
                if y is None:
                    continue
                mv = m.get("move", np.nan)
                mv_s = f"{float(mv):+.1%}" if pd.notna(mv) else "?"
                ax.append(d)
                ay.append(y)
                txts.append(str(m.get("symbol") or st["text"]))
                at.append(
                    f"{st['name']} {d}<br>变动{mv_s}（非交易）"
                    f"<br>价={float(m.get('px', np.nan)):.3f}"
                    f"<br>{m.get('why', '')}"
                )
            if not ax:
                continue
            fig.add_trace(
                go.Scatter(
                    x=ax,
                    y=ay,
                    mode="markers+text",
                    name=st["name"],
                    text=txts,
                    textposition=st["pos"],
                    textfont=dict(size=10, color=st["line"], family="Arial Black"),
                    marker=dict(
                        symbol="diamond-open",
                        size=13,
                        color=st["color"],
                        line=dict(width=2.0, color=st["line"]),
                    ),
                    hovertext=at,
                    hoverinfo="text",
                ),
                row=1,
                col=1,
            )

    if early_down_marks:
        # Diagnostic early-down layer (soft_bear/peak8); does not change trades.
        ed_styles = {
            "ed_alert": dict(
                name="ED告警(模型滞后)",
                color="#c0392b",
                line="#641e16",
                band="rgba(192,57,43,0.10)",
            ),
            "ed_watch": dict(
                name="ED观察(已对齐↓)",
                color="#7f8c8d",
                line="#2c3e50",
                band="rgba(127,140,141,0.08)",
            ),
        }
        # Keep existing regime shapes; append ED run bands then restore via layout shapes.
        shapes = list(fig.layout.shapes) if fig.layout.shapes else []
        for m in early_down_marks:
            d0 = str(m.get("date") or "")
            d1 = str(m.get("end") or d0)
            ts0, ts1 = pd.Timestamp(d0), pd.Timestamp(d1)
            if ts1 < pd.Timestamp(start_date) or ts0 > pd.Timestamp(end_date):
                continue
            kind = str(m.get("kind") or "ed_alert")
            st = ed_styles.get(kind) or ed_styles["ed_alert"]
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=max(ts0, pd.Timestamp(start_date)).strftime("%Y-%m-%d"),
                    x1=min(ts1, pd.Timestamp(end_date)).strftime("%Y-%m-%d"),
                    y0=lo - span * 0.02,
                    y1=hi + span * 0.05,
                    fillcolor=st["band"],
                    line=dict(width=1.0, color=st["line"], dash="dot"),
                    layer="below",
                )
            )
        fig.update_layout(shapes=shapes)

        by_kind_ed: dict[str, list[dict]] = {}
        for m in early_down_marks:
            d = str(m.get("date") or "")
            ts = pd.Timestamp(d)
            if ts < pd.Timestamp(start_date) or ts > pd.Timestamp(end_date):
                continue
            by_kind_ed.setdefault(str(m.get("kind") or "ed_alert"), []).append(m)
        for kind, rows in by_kind_ed.items():
            st = ed_styles.get(kind) or ed_styles["ed_alert"]
            ax, ay, at, txts = [], [], [], []
            for m in rows:
                d = str(m["date"])
                y = y_sell(d)
                if y is None:
                    continue
                ax.append(d)
                ay.append(y + span * 0.01)
                txts.append(str(m.get("symbol") or "ED"))
                end_s = str(m.get("end") or d)
                at.append(
                    f"{st['name']} {d}→{end_s}<br>"
                    f"n={int(m.get('n_bars') or 1)} · regime={m.get('regime')}<br>"
                    f"reasons={m.get('reasons') or '—'}<br>"
                    f"{m.get('why', '')}<br>诊断层·不改买卖"
                )
            if not ax:
                continue
            fig.add_trace(
                go.Scatter(
                    x=ax,
                    y=ay,
                    mode="markers+text",
                    name=st["name"],
                    text=txts,
                    textposition="top center",
                    textfont=dict(size=10, color=st["line"], family="Arial Black"),
                    marker=dict(
                        symbol="hexagon",
                        size=14,
                        color=st["color"],
                        line=dict(width=1.5, color=st["line"]),
                    ),
                    hovertext=at,
                    hoverinfo="text",
                ),
                row=1,
                col=1,
            )

    if recovery_up_marks:
        # Diagnostic recovery-up dual channel; does not change trades.
        ru_styles = {
            "ru_alert": dict(
                name="RU恢复上涨(诊断)",
                color="#1abc9c",
                line="#0e6655",
                band="rgba(26,188,156,0.12)",
            ),
        }
        shapes = list(fig.layout.shapes) if fig.layout.shapes else []
        for m in recovery_up_marks:
            d0 = str(m.get("date") or "")
            d1 = str(m.get("end") or d0)
            ts0, ts1 = pd.Timestamp(d0), pd.Timestamp(d1)
            if ts1 < pd.Timestamp(start_date) or ts0 > pd.Timestamp(end_date):
                continue
            kind = str(m.get("kind") or "ru_alert")
            st = ru_styles.get(kind) or ru_styles["ru_alert"]
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=max(ts0, pd.Timestamp(start_date)).strftime("%Y-%m-%d"),
                    x1=min(ts1, pd.Timestamp(end_date)).strftime("%Y-%m-%d"),
                    y0=lo - span * 0.02,
                    y1=hi + span * 0.05,
                    fillcolor=st["band"],
                    line=dict(width=1.0, color=st["line"], dash="dot"),
                    layer="below",
                )
            )
        fig.update_layout(shapes=shapes)

        by_kind_ru: dict[str, list[dict]] = {}
        for m in recovery_up_marks:
            d = str(m.get("date") or "")
            ts = pd.Timestamp(d)
            if ts < pd.Timestamp(start_date) or ts > pd.Timestamp(end_date):
                continue
            by_kind_ru.setdefault(str(m.get("kind") or "ru_alert"), []).append(m)
        for kind, rows in by_kind_ru.items():
            st = ru_styles.get(kind) or ru_styles["ru_alert"]
            ax, ay, at, txts = [], [], [], []
            for m in rows:
                d = str(m["date"])
                y = y_sell(d)
                if y is None:
                    continue
                ax.append(d)
                ay.append(y + span * 0.02)
                txts.append(str(m.get("symbol") or "RU"))
                end_s = str(m.get("end") or d)
                at.append(
                    f"{st['name']} {d}→{end_s}<br>"
                    f"n={int(m.get('n_bars') or 1)} · regime={m.get('regime')}<br>"
                    f"{m.get('reasons') or '—'}<br>"
                    f"{m.get('why', '')}<br>诊断双通道·不改买卖"
                )
            if not ax:
                continue
            fig.add_trace(
                go.Scatter(
                    x=ax,
                    y=ay,
                    mode="markers+text",
                    name=st["name"],
                    text=txts,
                    textposition="top center",
                    textfont=dict(size=10, color=st["line"], family="Arial Black"),
                    marker=dict(
                        symbol="diamond",
                        size=14,
                        color=st["color"],
                        line=dict(width=1.5, color=st["line"]),
                    ),
                    hovertext=at,
                    hoverinfo="text",
                ),
                row=1,
                col=1,
            )

    old = fig.layout.title.text if fig.layout.title else ""
    seg_txt = " | ".join(
        f"{r.regime} {str(r.start)[5:]}~{str(r.end)[5:]} {r.ret:+.0%}" for _, r in seg_df.iterrows()
    )
    win_s = f"{stats['win']:.0%}" if stats["win"] == stats["win"] else "n/a"
    open_note = ""
    if open_mtm:
        open_note = (
            f"｜窗口截止仍持仓（自{open_mtm['entry_date']}，"
            f"浮盈{open_mtm['unrealized']:+.1%}，非卖点）"
        )
    note = (
        f"<br><span style='font-size:12px;color:#333'>"
        f"{title_note}"
        f"<br>已平仓口径（含未平仓按收盘盯市）{stats['compound']:+.1%}｜持有 {stats['bh']:+.1%}｜"
        f"edge {stats['edge']:+.1%}｜已平仓笔数 n={stats['n']} 胜率 {win_s}{open_note}"
        f"<br><span style='font-size:11px;color:#555'>{seg_txt}</span></span>"
    )
    fig.update_layout(title=dict(text=(old or "") + note))
    return _apply_title_top_margin(fig)
