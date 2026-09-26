"""Leakage-safe walk-forward selection for ETF regime methods."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from decision_pack.src.state_space_regime import (
    STATE_SPACE_METHODS,
    StateSpaceRegimeConfig,
    fit_state_space_regime,
    state_space_labels_or_fallback,
)


DEFAULT_BASELINES = (
    "ma_stack_hyst",
    "ma_stack_struct",
    "ma_stack_strict",
    "dual_ma_cross",
    "hybrid_ma_adx",
)


def expanding_splits(
    n: int,
    *,
    n_folds: int = 4,
    min_train: int = 160,
    val_size: int = 60,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Non-overlapping expanding validation folds ending at n."""
    if n < min_train + 20:
        cut = max(30, int(n * 0.7))
        return [(np.arange(cut), np.arange(cut, n))] if cut < n else []
    usable = n - min_train
    size = max(20, min(int(val_size), usable // max(1, n_folds)))
    starts = list(range(min_train, n, size))
    starts = starts[-n_folds:]
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for start in starts:
        stop = min(n, start + size)
        if stop - start >= 10:
            out.append((np.arange(start), np.arange(start, stop)))
    return out


def beta_binomial_rate(
    wins: int,
    trades: int,
    *,
    prior_rate: float = 0.5,
    prior_strength: float = 5.0,
) -> float:
    a = max(float(prior_rate), 0.0) * max(float(prior_strength), 0.0)
    b = max(1.0 - float(prior_rate), 0.0) * max(float(prior_strength), 0.0)
    return float((wins + a) / max(trades + a + b, 1e-12))


def hold_up_trade_metrics(
    labels: np.ndarray,
    close: np.ndarray,
    *,
    dates: np.ndarray | None = None,
    cost_per_side: float = 0.001,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    """Evaluate enter-up/leave-up with closed-trade win and daily equity."""
    labs = np.asarray(labels, dtype=object)
    px = np.asarray(close, dtype=float)
    if len(labs) != len(px):
        raise ValueError("labels and close length mismatch")
    if dates is None:
        dates = np.arange(len(px)).astype(str)
    else:
        dates = np.asarray(dates).astype(str)
    if len(px) == 0:
        empty = pd.DataFrame(columns=["date", "equity", "drawdown", "exposed"])
        return {
            "n_closed": 0,
            "wins": 0,
            "win": np.nan,
            "mean_ret_net": 0.0,
            "compound": 0.0,
            "bh": 0.0,
            "edge": 0.0,
            "max_drawdown": 0.0,
            "exposure": 0.0,
            "switches": 0,
            "open_pos": False,
        }, [], empty

    wealth = 1.0
    entry_effective = np.nan
    entry_px = np.nan
    entry_i = -1
    trades: list[dict[str, Any]] = []
    equities = np.ones(len(px), dtype=float)
    exposed = np.zeros(len(px), dtype=bool)

    for i, lab in enumerate(labs):
        if entry_i < 0 and lab == "up" and np.isfinite(px[i]) and px[i] > 0:
            entry_i = i
            entry_px = float(px[i])
            entry_effective = entry_px * (1.0 + float(cost_per_side))
        elif entry_i >= 0 and lab != "up" and np.isfinite(px[i]) and px[i] > 0:
            exit_effective = float(px[i]) * (1.0 - float(cost_per_side))
            net = exit_effective / entry_effective - 1.0
            wealth *= 1.0 + net
            trades.append(
                {
                    "entry_date": str(dates[entry_i]),
                    "exit_date": str(dates[i]),
                    "entry_px": entry_px,
                    "exit_px": float(px[i]),
                    "ret_net": float(net),
                    "bars": int(i - entry_i),
                }
            )
            entry_i = -1
            entry_effective = entry_px = np.nan
        if entry_i >= 0:
            exposed[i] = True
            liquidation = float(px[i]) * (1.0 - float(cost_per_side))
            equities[i] = wealth * liquidation / entry_effective
        else:
            equities[i] = wealth

    peak = np.maximum.accumulate(equities)
    drawdown = equities / np.maximum(peak, 1e-12) - 1.0
    rets = np.asarray([t["ret_net"] for t in trades], dtype=float)
    bh = (
        float(px[-1] / px[0] - 1.0)
        if len(px) > 1 and np.isfinite(px[0]) and px[0] > 0 and np.isfinite(px[-1])
        else 0.0
    )
    metrics = {
        "n_closed": int(len(trades)),
        "wins": int(np.sum(rets > 0)),
        "win": float(np.mean(rets > 0)) if len(rets) else np.nan,
        "mean_ret_net": float(np.mean(rets)) if len(rets) else 0.0,
        "compound": float(equities[-1] - 1.0),
        "bh": bh,
        "edge": float(equities[-1] - 1.0 - bh),
        "max_drawdown": float(np.min(drawdown)),
        "exposure": float(np.mean(exposed)),
        "switches": int(np.sum(labs[1:] != labs[:-1])) if len(labs) > 1 else 0,
        "open_pos": bool(entry_i >= 0),
    }
    equity = pd.DataFrame(
        {
            "date": dates,
            "equity": equities,
            "drawdown": drawdown,
            "exposed": exposed,
        }
    )
    return metrics, trades, equity


def state_space_candidate_grid() -> list[dict[str, Any]]:
    """Small explicit grid; enough flexibility without per-symbol curve fitting."""
    out: list[dict[str, Any]] = []
    for method in STATE_SPACE_METHODS:
        for prob in (0.68, 0.74):
            q_slope = 0.001
            cfg = StateSpaceRegimeConfig(
                method=method,
                direction_prob=prob,
                high_vol_direction_prob=min(prob + 0.08, 0.90),
                q_slope_ratio=q_slope,
                confirm_days=3,
                hmm_max_iter=40,
            )
            raw = asdict(cfg)
            key = f"{method}:p{prob:.2f}:q{q_slope:g}"
            out.append({"key": key, "kind": "state_space", "config": raw})
    return out


def candidate_grid(
    baseline_methods: tuple[str, ...] = DEFAULT_BASELINES,
) -> list[dict[str, Any]]:
    base = [
        {"key": name, "kind": "baseline", "method": name, "config": {}}
        for name in baseline_methods
    ]
    return base + state_space_candidate_grid()


def _candidate_labels(
    ev,
    candidate: dict[str, Any],
    *,
    px_fit: pd.DataFrame,
    px_apply: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    if candidate["kind"] == "baseline":
        labs = ev.run_method(candidate["method"], px_apply)
        return np.asarray(labs, dtype=object), {"fallback": False}
    cfg = StateSpaceRegimeConfig(**dict(candidate["config"]))
    model = fit_state_space_regime(px_fit, cfg)
    fallback = ev.run_method(cfg.fallback_method, px_apply)
    labels, diag = state_space_labels_or_fallback(
        px_apply, model, fallback_labels=fallback
    )
    return labels, {"model": model, **diag}


def evaluate_candidate_folds(
    ev,
    px: pd.DataFrame,
    candidates: list[dict[str, Any]],
    *,
    splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
    cost_per_side: float = 0.001,
) -> pd.DataFrame:
    """Fit each candidate on fold-train and score only the subsequent fold."""
    folds = splits or expanding_splits(len(px))
    rows: list[dict[str, Any]] = []
    for fold, (tr, va) in enumerate(folds):
        if len(tr) == 0 or len(va) == 0:
            continue
        apply_end = int(va[-1]) + 1
        px_fit = px.iloc[tr].reset_index(drop=True)
        px_apply = px.iloc[:apply_end].reset_index(drop=True)
        close = px_apply["$close"].to_numpy(float)[va]
        dates = (
            px_apply["as_of"].dt.strftime("%Y-%m-%d").to_numpy()[va]
            if "as_of" in px_apply.columns
            else va.astype(str)
        )
        for candidate in candidates:
            labels, diag = _candidate_labels(
                ev, candidate, px_fit=px_fit, px_apply=px_apply
            )
            metrics, _, _ = hold_up_trade_metrics(
                labels[va], close, dates=dates, cost_per_side=cost_per_side
            )
            rows.append(
                {
                    "fold": fold,
                    "train_end_i": int(tr[-1]),
                    "val_start_i": int(va[0]),
                    "val_end_i": int(va[-1]),
                    "candidate": candidate["key"],
                    "kind": candidate["kind"],
                    "fallback": bool(diag.get("fallback", False)),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def aggregate_candidate_metrics(
    fold_rows: pd.DataFrame,
    *,
    prior_rate: float,
    prior_strength: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, g in fold_rows.groupby("candidate", sort=False):
        n = int(g["n_closed"].sum())
        wins = int(g["wins"].sum())
        weights = np.maximum(g["n_closed"].to_numpy(float), 1.0)
        rows.append(
            {
                "candidate": candidate,
                "kind": str(g["kind"].iloc[0]),
                "n_closed": n,
                "wins": wins,
                "posterior_win": beta_binomial_rate(
                    wins,
                    n,
                    prior_rate=prior_rate,
                    prior_strength=prior_strength,
                ),
                "mean_ret_net": float(np.average(g["mean_ret_net"], weights=weights)),
                "mean_edge": float(g["edge"].mean()),
                "worst_drawdown": float(g["max_drawdown"].min()),
                "mean_exposure": float(g["exposure"].mean()),
                "fallback_folds": int(g["fallback"].sum()),
            }
        )
    return pd.DataFrame(rows)


def choose_candidate(
    aggregate: pd.DataFrame,
    *,
    baseline_candidate: str,
    min_closed: int = 2,
    max_drawdown_floor: float = -0.25,
) -> tuple[str, dict[str, Any]]:
    """Win-first constrained selection with a deterministic baseline fallback."""
    if aggregate.empty:
        return baseline_candidate, {"accepted": False, "reason": "no_folds"}
    eligible = aggregate[
        (aggregate["n_closed"] >= int(min_closed))
        & (aggregate["mean_ret_net"] >= 0.0)
        & (aggregate["worst_drawdown"] >= float(max_drawdown_floor))
    ].copy()
    if eligible.empty:
        return baseline_candidate, {
            "accepted": False,
            "reason": "no_eligible_candidate",
        }
    eligible = eligible.sort_values(
        ["posterior_win", "mean_ret_net", "mean_edge", "n_closed"],
        ascending=[False, False, False, False],
    )
    winner = str(eligible.iloc[0]["candidate"])
    return winner, {
        "accepted": True,
        "reason": "win_first_constraints_passed",
        "winner_metrics": eligible.iloc[0].to_dict(),
    }


def fit_final_candidate(
    ev,
    px_train: pd.DataFrame,
    candidate: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    if candidate["kind"] == "baseline":
        return str(candidate["method"]), {}
    cfg = StateSpaceRegimeConfig(**dict(candidate["config"]))
    model = fit_state_space_regime(px_train, cfg)
    payload = {"frozen_model": model, "fallback_method": cfg.fallback_method}
    blob = json.dumps(payload, sort_keys=True, default=str)
    payload["model_hash"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    return str(cfg.method), payload


def grouped_block_bootstrap_delta(
    selected: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    metric: str,
    n_boot: int = 2000,
    seed: int = 20260808,
) -> dict[str, float]:
    """Bootstrap symbol/fold rows as blocks; inputs must share `block_id`."""
    a = selected.set_index("block_id")[metric]
    b = baseline.set_index("block_id")[metric]
    ids = a.index.intersection(b.index).unique().to_numpy()
    if len(ids) == 0:
        return {"delta": np.nan, "lo": np.nan, "hi": np.nan, "n_blocks": 0}
    diff = (a.loc[ids].to_numpy(float) - b.loc[ids].to_numpy(float))
    rng = np.random.default_rng(seed)
    sims = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        sims[i] = float(np.mean(rng.choice(diff, size=len(diff), replace=True)))
    return {
        "delta": float(np.mean(diff)),
        "lo": float(np.quantile(sims, 0.05)),
        "hi": float(np.quantile(sims, 0.95)),
        "n_blocks": int(len(ids)),
    }
