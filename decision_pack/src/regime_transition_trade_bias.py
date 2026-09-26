"""Walk-forward trade-bias labels for regime-transition triangles.

Execution contract (no lookahead for signal generation):
  - Signal is computed at EOD ``as_of`` (T) from causal features only.
  - Entry is the next trading day's open (T+1).
  - Primary exit is the close of T+hold_days (default hold_days=5 → T+5 close).
  - Labels use future prices only for historical training samples that are
    already mature before a prediction month starts.
  - Forbidden feature fields: reversal/diamond labels, post_ret, future returns.

Action space is three-way: ``buy_bias`` / ``sell_bias`` / ``observe_only``.
Sell bias measures avoidance / long-exit benefit; it does not assume ETF shorting.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decision_pack.src.regime_transition_validation import compute_causal_prior_context

TRADE_BIAS_SCHEMA = "regime_trade_bias_v1"
PRIMARY_HOLD_DAYS = 5
AUX_HOLD_DAYS = (1, 3, 10)
DEFAULT_COST_BPS = (0, 10, 20, 30)
PRIMARY_COST_BPS = 10
MIN_TRAIN_ROWS = 40
MIN_TRAIN_POS = 8
MIN_TRAIN_NEG = 8
MIN_ACTIONS_FOR_THRESHOLD = 20

# Features knowable at EOD T. Never include reversal / post / forward labels.
FEATURE_WHITELIST = (
    "pred_cdf",
    "tail_strength",
    "pred_mean_next",
    "pred_scale_next",
    "pred_mean",
    "pred_scale",
    "return_z",
    "p_switch_hmm",
    "p_into_reversal_hmm",
    "p_state_reversal",
    "transition_score",
    "switch_side_up",
    "regime_separated",
    "label_ambiguous",
    "causal_prior_ret",
    "causal_jump_ret",
    "causal_thr_prior",
    "causal_prior_strong",
)

FORBIDDEN_FEATURES = (
    "turn_kind",
    "jump_role",
    "post_ret",
    "label_switch_reversal",
    "early_label_switch_reversal",
    "early_turn_kind",
    "early_post_ret",
    "matched_event_onset",
    "is_fp",
    "is_duplicate",
    "label_trend_up",
    "fwd_ret_h",
    "fwd_ret_1",
    "fwd_ret_3",
    "fwd_ret_5",
    "fwd_ret_10",
    "trade_ret_h5",
    "trade_ret_h1",
    "y_up",
)

BUY_THR_GRID = (0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70)
SELL_THR_GRID = (0.30, 0.32, 0.35, 0.38, 0.40, 0.42, 0.45)


@dataclass(frozen=True)
class TradeBiasConfig:
    hold_days: int = PRIMARY_HOLD_DAYS
    aux_hold_days: tuple[int, ...] = AUX_HOLD_DAYS
    primary_cost_bps: int = PRIMARY_COST_BPS
    cost_bps_grid: tuple[int, ...] = DEFAULT_COST_BPS
    min_train_rows: int = MIN_TRAIN_ROWS
    min_train_pos: int = MIN_TRAIN_POS
    min_train_neg: int = MIN_TRAIN_NEG
    min_feature_coverage: float = 0.50
    buy_thr_grid: tuple[float, ...] = BUY_THR_GRID
    sell_thr_grid: tuple[float, ...] = SELL_THR_GRID
    random_state: int = 0


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _trading_loc(index: pd.DatetimeIndex, as_of: pd.Timestamp) -> int | None:
    as_of = pd.Timestamp(as_of).normalize()
    if as_of not in index:
        return None
    loc = index.get_loc(as_of)
    if isinstance(loc, slice):
        return int(loc.start)
    if isinstance(loc, np.ndarray):
        return int(np.flatnonzero(loc)[0]) if loc.any() else None
    return int(loc)


def open_to_close_simple_return(
    open_series: pd.Series,
    close_series: pd.Series,
    as_of: pd.Timestamp,
    *,
    hold_days: int,
) -> float | None:
    """T+1 open → T+hold_days close simple return. None if immature/missing."""
    as_of = pd.Timestamp(as_of).normalize()
    opens = pd.to_numeric(open_series, errors="coerce").dropna()
    closes = pd.to_numeric(close_series, errors="coerce").dropna()
    # Align on intersection so T+k refers to the same trading calendar.
    idx = opens.index.intersection(closes.index).sort_values()
    if len(idx) == 0:
        return None
    loc = _trading_loc(pd.DatetimeIndex(idx), as_of)
    if loc is None:
        return None
    entry_loc = loc + 1
    exit_loc = loc + int(hold_days)
    if entry_loc >= len(idx) or exit_loc >= len(idx):
        return None
    entry_date = pd.Timestamp(idx[entry_loc]).normalize()
    exit_date = pd.Timestamp(idx[exit_loc]).normalize()
    o = float(opens.loc[entry_date]) if entry_date in opens.index else float("nan")
    c = float(closes.loc[exit_date]) if exit_date in closes.index else float("nan")
    if o <= 0 or c <= 0 or not math.isfinite(o) or not math.isfinite(c):
        return None
    return float(c / o - 1.0)


def attach_trade_labels(
    signals: pd.DataFrame,
    open_panel: pd.DataFrame,
    close_panel: pd.DataFrame,
    *,
    hold_days: int = PRIMARY_HOLD_DAYS,
    aux_hold_days: tuple[int, ...] = AUX_HOLD_DAYS,
) -> pd.DataFrame:
    """Attach open→close trade returns and binary up label for switch rows.

    Non-switch rows keep NaN trade returns. Maturity requires T+hold_days close.
    """
    out = signals.copy()
    out["as_of"] = pd.to_datetime(out["as_of"]).dt.normalize()
    out["code"] = out["code"].astype(str).str.upper()
    if "month" not in out.columns:
        out["month"] = out["as_of"].dt.to_period("M").astype(str)
    if "quantile_switch" not in out.columns:
        out["quantile_switch"] = False
    out["quantile_switch"] = out["quantile_switch"].fillna(False).astype(bool)

    horizons = tuple(sorted({int(hold_days), *[int(h) for h in aux_hold_days]}))
    for h in horizons:
        out[f"trade_ret_h{h}"] = np.nan
    out["entry_open_date"] = pd.NaT
    out["exit_close_date"] = pd.NaT
    out["trade_label_mature"] = False
    out["y_up"] = np.nan

    open_panel = open_panel.copy()
    close_panel = close_panel.copy()
    open_panel.index = pd.to_datetime(open_panel.index).normalize()
    close_panel.index = pd.to_datetime(close_panel.index).normalize()
    open_panel.columns = [str(c).upper() for c in open_panel.columns]
    close_panel.columns = [str(c).upper() for c in close_panel.columns]

    for i, row in out.iterrows():
        code = str(row["code"])
        as_of = pd.Timestamp(row["as_of"]).normalize()
        if code not in open_panel.columns or code not in close_panel.columns:
            continue
        opens = open_panel[code]
        closes = close_panel[code]
        # Entry/exit calendar dates for primary horizon (diagnostic only).
        idx = opens.dropna().index.intersection(closes.dropna().index).sort_values()
        loc = _trading_loc(pd.DatetimeIndex(idx), as_of)
        if loc is not None and loc + 1 < len(idx):
            out.at[i, "entry_open_date"] = pd.Timestamp(idx[loc + 1]).normalize()
        if loc is not None and loc + int(hold_days) < len(idx):
            out.at[i, "exit_close_date"] = pd.Timestamp(
                idx[loc + int(hold_days)]
            ).normalize()

        primary = open_to_close_simple_return(
            opens, closes, as_of, hold_days=int(hold_days)
        )
        if primary is not None:
            out.at[i, f"trade_ret_h{int(hold_days)}"] = primary
            out.at[i, "trade_label_mature"] = True
            out.at[i, "y_up"] = 1.0 if primary > 0.0 else 0.0
        for h in horizons:
            if h == int(hold_days):
                continue
            ret = open_to_close_simple_return(opens, closes, as_of, hold_days=int(h))
            if ret is not None:
                out.at[i, f"trade_ret_h{int(h)}"] = ret
    return out


def attach_causal_context_from_close(
    frame: pd.DataFrame,
    close_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Fill causal prior/jump fields from close panel when missing."""
    out = frame.copy()
    for col in (
        "causal_prior_ret",
        "causal_jump_ret",
        "causal_thr_prior",
        "causal_prior_strong",
    ):
        if col not in out.columns:
            out[col] = np.nan if col != "causal_prior_strong" else False
    close_panel = close_panel.copy()
    close_panel.index = pd.to_datetime(close_panel.index).normalize()
    close_panel.columns = [str(c).upper() for c in close_panel.columns]
    for i, row in out.iterrows():
        if _safe_float(row.get("causal_prior_ret")) is not None:
            continue
        code = str(row["code"]).upper()
        if code not in close_panel.columns:
            continue
        ctx = compute_causal_prior_context(
            close_panel[code], pd.Timestamp(row["as_of"]).normalize()
        )
        if ctx is None:
            continue
        out.at[i, "causal_prior_ret"] = ctx["prior_ret"]
        out.at[i, "causal_jump_ret"] = ctx["jump_ret"]
        out.at[i, "causal_thr_prior"] = ctx["thr_prior"]
        out.at[i, "causal_prior_strong"] = bool(ctx["prior_strong"])
    return out


def build_trade_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive model feature columns from causal signal fields only."""
    out = frame.copy()
    cdf = pd.to_numeric(out.get("pred_cdf"), errors="coerce")
    out["tail_strength"] = (cdf - 0.5).abs()
    side = out.get("switch_side")
    if side is None:
        out["switch_side_up"] = np.nan
    else:
        out["switch_side_up"] = side.map(
            lambda v: 1.0 if str(v) == "up" else (0.0 if str(v) == "down" else np.nan)
        )
    for col in ("regime_separated", "label_ambiguous", "causal_prior_strong"):
        if col not in out.columns:
            out[col] = 0.0
        out[col] = (
            out[col]
            .map(lambda v: 1.0 if bool(v) and not (isinstance(v, float) and pd.isna(v)) else 0.0)
            .astype(float)
        )
    # Ensure whitelist columns exist.
    for col in FEATURE_WHITELIST:
        if col not in out.columns:
            out[col] = np.nan
    leaked = [c for c in FORBIDDEN_FEATURES if c in FEATURE_WHITELIST]
    if leaked:
        raise ValueError(f"forbidden features present in whitelist: {leaked}")
    return out


def available_feature_cols(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in FEATURE_WHITELIST:
        if col not in frame.columns:
            continue
        series = pd.to_numeric(frame[col], errors="coerce")
        if series.notna().any():
            cols.append(col)
    return cols


def select_usable_feature_cols(
    train: pd.DataFrame,
    feature_cols: list[str],
    *,
    min_coverage: float = 0.50,
) -> list[str]:
    """Keep features with enough non-null coverage on the training slice.

    Sparse columns such as ``pred_mean_next`` (~94% NaN in production) must not
    zero out the entire training set via an all-finite row mask.
    """
    usable: list[str] = []
    for col in feature_cols:
        if col not in train.columns:
            continue
        series = pd.to_numeric(train[col], errors="coerce")
        if float(series.notna().mean()) >= float(min_coverage):
            usable.append(col)
    return usable


def _median_impute_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fit medians on train and impute train/test feature matrices."""
    medians: dict[str, float] = {}
    train_cols: list[np.ndarray] = []
    test_cols: list[np.ndarray] = []
    for col in feature_cols:
        tr = pd.to_numeric(train[col], errors="coerce").to_numpy(dtype=float)
        te = pd.to_numeric(test[col], errors="coerce").to_numpy(dtype=float)
        finite = tr[np.isfinite(tr)]
        med = float(np.median(finite)) if finite.size else 0.0
        medians[col] = med
        train_cols.append(np.where(np.isfinite(tr), tr, med))
        test_cols.append(np.where(np.isfinite(te), te, med))
    x_train = np.column_stack(train_cols) if train_cols else np.zeros((len(train), 0))
    x_test = np.column_stack(test_cols) if test_cols else np.zeros((len(test), 0))
    return x_train, x_test, medians


def _fit_predict_month(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    *,
    cfg: TradeBiasConfig,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    meta: dict[str, Any] = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "trained": False,
        "reason": None,
        "feature_cols_used": [],
    }
    if len(train) < cfg.min_train_rows:
        meta["reason"] = "insufficient_train_rows"
        return None, meta
    y_all = pd.to_numeric(train["y_up"], errors="coerce")
    label_mask = y_all.notna()
    train_l = train.loc[label_mask].copy()
    y = y_all.loc[label_mask].astype(int)
    if len(train_l) < cfg.min_train_rows:
        meta["reason"] = "insufficient_labeled_train_rows"
        return None, meta

    usable = select_usable_feature_cols(
        train_l, feature_cols, min_coverage=cfg.min_feature_coverage
    )
    meta["feature_cols_used"] = list(usable)
    if not usable:
        meta["reason"] = "no_usable_features"
        return None, meta

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    meta["n_pos"] = n_pos
    meta["n_neg"] = n_neg
    if n_pos < cfg.min_train_pos or n_neg < cfg.min_train_neg:
        meta["reason"] = "single_class_or_sparse"
        return None, meta

    x_train, x_test, medians = _median_impute_matrix(train_l, test, usable)
    meta["feature_medians"] = {k: float(v) for k, v in medians.items()}
    if x_train.shape[1] == 0 or len(test) == 0:
        meta["reason"] = "empty_feature_matrix"
        return None, meta

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=300,
        random_state=int(cfg.random_state),
    )
    model.fit(x_train_s, y.to_numpy())
    proba = model.predict_proba(x_test_s)[:, 1]
    meta["trained"] = True
    meta["reason"] = "ok"
    meta["train_end"] = str(pd.Timestamp(train_l["as_of"].max()).date())
    return np.asarray(proba, dtype=float), meta


def _action_from_proba(
    p_up: float | None,
    *,
    buy_thr: float,
    sell_thr: float,
) -> str:
    if p_up is None or not math.isfinite(float(p_up)):
        return "observe_only"
    p = float(p_up)
    if p >= float(buy_thr):
        return "buy_bias"
    if p <= float(sell_thr):
        return "sell_bias"
    return "observe_only"


def _signed_net_return(action: str, trade_ret: float, cost_bps: float) -> float | None:
    cost = float(cost_bps) / 10000.0
    if action == "buy_bias":
        return float(trade_ret) - cost
    if action == "sell_bias":
        # Avoidance / long-exit benefit: positive when price falls after signal.
        return -float(trade_ret) - cost
    return None


def select_thresholds_on_history(
    history: pd.DataFrame,
    *,
    p_col: str = "p_up",
    ret_col: str = "trade_ret_h5",
    cost_bps: int = PRIMARY_COST_BPS,
    buy_thr_grid: tuple[float, ...] = BUY_THR_GRID,
    sell_thr_grid: tuple[float, ...] = SELL_THR_GRID,
    min_actions: int = MIN_ACTIONS_FOR_THRESHOLD,
) -> dict[str, Any]:
    """Pick buy/sell probability thresholds on historical OOS predictions only."""
    default = {
        "buy_thr": 0.60,
        "sell_thr": 0.40,
        "score": None,
        "n_actions": 0,
        "reason": "fallback_default",
    }
    if history is None or len(history) == 0:
        return default
    work = history.copy()
    if p_col not in work.columns or ret_col not in work.columns:
        return default
    work = work[pd.to_numeric(work[p_col], errors="coerce").notna()]
    work = work[pd.to_numeric(work[ret_col], errors="coerce").notna()]
    if work.empty:
        return default
    best = default
    best_score = -1e18
    for buy_thr in buy_thr_grid:
        for sell_thr in sell_thr_grid:
            if float(sell_thr) >= float(buy_thr):
                continue
            signed: list[float] = []
            for _, row in work.iterrows():
                action = _action_from_proba(
                    float(row[p_col]), buy_thr=buy_thr, sell_thr=sell_thr
                )
                sn = _signed_net_return(action, float(row[ret_col]), float(cost_bps))
                if sn is not None:
                    signed.append(sn)
            n_actions = len(signed)
            if n_actions < min_actions:
                continue
            score = float(np.mean(signed))
            # Prefer more actions on ties to avoid tiny overfit pockets.
            key = (score, n_actions, -abs(buy_thr - 0.6), -abs(sell_thr - 0.4))
            best_key = (
                best_score,
                int(best.get("n_actions") or 0),
                -abs(float(best["buy_thr"]) - 0.6),
                -abs(float(best["sell_thr"]) - 0.4),
            )
            if key > best_key:
                best_score = score
                best = {
                    "buy_thr": float(buy_thr),
                    "sell_thr": float(sell_thr),
                    "score": score,
                    "n_actions": n_actions,
                    "reason": "max_mean_signed_net_on_history",
                }
    return best


def run_trade_bias_walkforward(
    signals: pd.DataFrame,
    open_panel: pd.DataFrame,
    close_panel: pd.DataFrame,
    *,
    cfg: TradeBiasConfig | None = None,
) -> dict[str, Any]:
    """Expanding-month walk-forward for triangle trade bias.

    For prediction month M:
      - train only on switch rows with month < M and trade_label_mature
      - freeze scaler/model/thresholds on that history
      - score all switch rows in month M (including immature tails)
    """
    cfg = cfg or TradeBiasConfig()
    ret_col = f"trade_ret_h{int(cfg.hold_days)}"
    frame = attach_trade_labels(
        signals,
        open_panel,
        close_panel,
        hold_days=cfg.hold_days,
        aux_hold_days=cfg.aux_hold_days,
    )
    frame = attach_causal_context_from_close(frame, close_panel)
    frame = build_trade_feature_frame(frame)
    feature_cols = available_feature_cols(frame)
    switches = frame[frame["quantile_switch"]].copy()
    months = sorted(switches["month"].astype(str).unique().tolist())

    pred_rows: list[pd.DataFrame] = []
    month_meta: list[dict[str, Any]] = []
    history_scored: list[pd.DataFrame] = []

    for month in months:
        train = switches[
            (switches["month"].astype(str) < month)
            & switches["trade_label_mature"].fillna(False)
            & pd.to_numeric(switches["y_up"], errors="coerce").notna()
        ].copy()
        test = switches[switches["month"].astype(str) == month].copy()
        if test.empty:
            continue
        proba, fit_meta = _fit_predict_month(train, test, feature_cols, cfg=cfg)
        thr = {"buy_thr": 0.60, "sell_thr": 0.40, "reason": "default_untrained"}
        if history_scored:
            hist = pd.concat(history_scored, ignore_index=True)
            # Threshold selection must not see the current prediction month.
            thr = select_thresholds_on_history(
                hist,
                p_col="p_up",
                ret_col=ret_col,
                cost_bps=cfg.primary_cost_bps,
                buy_thr_grid=cfg.buy_thr_grid,
                sell_thr_grid=cfg.sell_thr_grid,
            )
        out = test.copy()
        if proba is None:
            out["p_up"] = np.nan
            out["trade_action"] = "observe_only"
            out["model_trained"] = False
            out["model_skip_reason"] = fit_meta.get("reason")
        else:
            out["p_up"] = proba
            out["trade_action"] = [
                _action_from_proba(
                    None if (isinstance(p, float) and not math.isfinite(p)) else p,
                    buy_thr=float(thr["buy_thr"]),
                    sell_thr=float(thr["sell_thr"]),
                )
                for p in proba
            ]
            out["model_trained"] = True
            out["model_skip_reason"] = None
        out["buy_thr"] = float(thr["buy_thr"])
        out["sell_thr"] = float(thr["sell_thr"])
        out["threshold_reason"] = thr.get("reason")
        out["model_train_end"] = fit_meta.get("train_end")
        out["feature_cols"] = ",".join(feature_cols)
        out["hold_days"] = int(cfg.hold_days)
        out["schema"] = TRADE_BIAS_SCHEMA
        pred_rows.append(out)
        month_meta.append(
            {
                "month": month,
                **fit_meta,
                "buy_thr": thr.get("buy_thr"),
                "sell_thr": thr.get("sell_thr"),
                "threshold_reason": thr.get("reason"),
                "n_pred": int(len(out)),
            }
        )
        # Only mature scored months feed future threshold selection.
        scored = out[
            out["model_trained"].fillna(False)
            & out["trade_label_mature"].fillna(False)
            & pd.to_numeric(out["p_up"], errors="coerce").notna()
            & pd.to_numeric(out[ret_col], errors="coerce").notna()
        ].copy()
        if not scored.empty:
            history_scored.append(scored)

    predictions = (
        pd.concat(pred_rows, ignore_index=True)
        if pred_rows
        else switches.iloc[0:0].copy()
    )
    metrics = summarize_trade_bias_metrics(
        predictions,
        hold_days=cfg.hold_days,
        cost_bps_grid=cfg.cost_bps_grid,
        primary_cost_bps=cfg.primary_cost_bps,
    )
    equity = build_equal_weight_event_equity(
        predictions,
        open_panel=open_panel,
        close_panel=close_panel,
        hold_days=cfg.hold_days,
        cost_bps=cfg.primary_cost_bps,
    )
    manifest = {
        "schema": TRADE_BIAS_SCHEMA,
        "config": asdict(cfg),
        "feature_whitelist": list(FEATURE_WHITELIST),
        "feature_cols_used": feature_cols,
        "forbidden_features": list(FORBIDDEN_FEATURES),
        "execution": {
            "signal_time": "EOD T",
            "entry": "T+1 open",
            "exit": f"T+{cfg.hold_days} close",
            "sell_semantics": "avoidance_or_long_exit_not_hard_short",
        },
        "n_switch_rows": int(len(switches)),
        "n_predictions": int(len(predictions)),
        "months": months,
        "month_meta": month_meta,
    }
    return {
        "predictions": predictions,
        "metrics": metrics,
        "equity": equity,
        "manifest": manifest,
        "frame": frame,
    }


def month_block_bootstrap_mean_ci(
    values: np.ndarray,
    months: np.ndarray | None = None,
    *,
    n_boot: int = 400,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float | None]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    point = float(np.mean(x))
    if months is None or len(months) != len(values):
        if x.size < 5:
            return {"mean": point, "ci_low": point, "ci_high": point, "n": int(x.size)}
        rng = np.random.default_rng(seed)
        boots = [
            float(np.mean(rng.choice(x, size=x.size, replace=True)))
            for _ in range(n_boot)
        ]
        return {
            "mean": point,
            "ci_low": float(np.quantile(boots, alpha / 2.0)),
            "ci_high": float(np.quantile(boots, 1.0 - alpha / 2.0)),
            "n": int(x.size),
        }
    month_arr = np.asarray(months)
    # Align finite mask.
    finite = np.isfinite(np.asarray(values, dtype=float))
    x = np.asarray(values, dtype=float)[finite]
    month_arr = month_arr[finite]
    uniq = sorted({str(m) for m in month_arr.tolist()})
    if len(uniq) < 3:
        return {"mean": point, "ci_low": point, "ci_high": point, "n": int(x.size)}
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.isin(month_arr.astype(str), chosen)
        if not mask.any():
            continue
        boots.append(float(np.mean(x[mask])))
    if not boots:
        return {"mean": point, "ci_low": None, "ci_high": None, "n": int(x.size)}
    return {
        "mean": point,
        "ci_low": float(np.quantile(boots, alpha / 2.0)),
        "ci_high": float(np.quantile(boots, 1.0 - alpha / 2.0)),
        "n": int(x.size),
    }


def _metric_row_for_slice(
    sub: pd.DataFrame,
    *,
    window: str,
    side: str,
    hold_days: int,
    cost_bps: int,
) -> dict[str, Any]:
    ret_col = f"trade_ret_h{int(hold_days)}"
    work = sub.copy()
    if side != "both":
        work = work[work["trade_action"] == side]
    else:
        work = work[work["trade_action"].isin(["buy_bias", "sell_bias"])]
    work = work[pd.to_numeric(work.get(ret_col), errors="coerce").notna()]
    n_alerts = int(len(sub))
    n_actions = int(len(work))
    if work.empty:
        return {
            "window": window,
            "side": side,
            "hold_days": int(hold_days),
            "cost_bps": int(cost_bps),
            "n_switch": n_alerts,
            "n_actions": 0,
            "coverage": 0.0 if n_alerts else None,
            "hit_rate": None,
            "mean_signed_net": None,
            "median_signed_net": None,
            "ci_low": None,
            "ci_high": None,
            "supports_positive_edge": False,
        }
    hits: list[float] = []
    signed: list[float] = []
    months: list[str] = []
    for _, row in work.iterrows():
        action = str(row["trade_action"])
        ret = float(row[ret_col])
        sn = _signed_net_return(action, ret, float(cost_bps))
        if sn is None:
            continue
        signed.append(sn)
        # Directional hit ignores cost (price moved the intended way).
        if action == "buy_bias":
            hits.append(1.0 if ret > 0 else (0.0 if ret < 0 else np.nan))
        else:
            hits.append(1.0 if ret < 0 else (0.0 if ret > 0 else np.nan))
        months.append(str(row.get("month")))
    hit_arr = np.asarray([h for h in hits if math.isfinite(h)], dtype=float)
    signed_arr = np.asarray(signed, dtype=float)
    ci = month_block_bootstrap_mean_ci(signed_arr, np.asarray(months))
    mean_signed = float(np.mean(signed_arr)) if signed_arr.size else None
    return {
        "window": window,
        "side": side,
        "hold_days": int(hold_days),
        "cost_bps": int(cost_bps),
        "n_switch": n_alerts,
        "n_actions": n_actions,
        "coverage": (n_actions / n_alerts) if n_alerts else None,
        "hit_rate": float(np.mean(hit_arr)) if hit_arr.size else None,
        "mean_signed_net": mean_signed,
        "median_signed_net": float(np.median(signed_arr)) if signed_arr.size else None,
        "ci_low": ci.get("ci_low"),
        "ci_high": ci.get("ci_high"),
        "supports_positive_edge": bool(
            mean_signed is not None
            and ci.get("ci_low") is not None
            and mean_signed > 0
            and float(ci["ci_low"]) >= 0
            and n_actions >= 100
        ),
    }


def summarize_trade_bias_metrics(
    predictions: pd.DataFrame,
    *,
    hold_days: int = PRIMARY_HOLD_DAYS,
    cost_bps_grid: tuple[int, ...] = DEFAULT_COST_BPS,
    primary_cost_bps: int = PRIMARY_COST_BPS,
) -> dict[str, Any]:
    if predictions.empty:
        return {
            "rows": pd.DataFrame(),
            "by_month": pd.DataFrame(),
            "by_etf": pd.DataFrame(),
            "summary": {
                "n_predictions": 0,
                "pass_gate": False,
                "note": "no predictions",
            },
        }
    work = predictions.copy()
    if "month" not in work.columns:
        work["month"] = pd.to_datetime(work["as_of"]).dt.to_period("M").astype(str)
    mature = work[work["trade_label_mature"].fillna(False)].copy()
    # Holdout = latest ~25% months (development OOS evidence only).
    months = sorted(mature["month"].astype(str).unique().tolist())
    n_hold = max(2, int(round(len(months) * 0.25))) if months else 0
    holdout_months = set(months[-n_hold:]) if n_hold else set()
    windows = {
        "full_mature": mature,
        "holdout": mature[mature["month"].astype(str).isin(holdout_months)].copy(),
    }
    rows: list[dict[str, Any]] = []
    for window_name, sub in windows.items():
        for side in ("buy_bias", "sell_bias", "both"):
            for cost in cost_bps_grid:
                rows.append(
                    _metric_row_for_slice(
                        sub,
                        window=window_name,
                        side=side,
                        hold_days=hold_days,
                        cost_bps=int(cost),
                    )
                )
    metrics_df = pd.DataFrame(rows)

    by_month_rows: list[dict[str, Any]] = []
    for month, grp in mature.groupby("month", sort=True):
        by_month_rows.append(
            _metric_row_for_slice(
                grp,
                window=str(month),
                side="both",
                hold_days=hold_days,
                cost_bps=primary_cost_bps,
            )
        )
    by_etf_rows: list[dict[str, Any]] = []
    for code, grp in mature.groupby("code", sort=True):
        by_etf_rows.append(
            _metric_row_for_slice(
                grp,
                window=str(code),
                side="both",
                hold_days=hold_days,
                cost_bps=primary_cost_bps,
            )
        )
    by_month = pd.DataFrame(by_month_rows)
    by_etf = pd.DataFrame(by_etf_rows)

    primary = metrics_df[
        (metrics_df["window"] == "holdout")
        & (metrics_df["cost_bps"] == primary_cost_bps)
    ] if not metrics_df.empty else metrics_df
    buy = primary[primary["side"] == "buy_bias"] if not primary.empty else primary
    sell = primary[primary["side"] == "sell_bias"] if not primary.empty else primary
    buy_ok = bool(buy["supports_positive_edge"].any()) if not buy.empty else False
    sell_ok = bool(sell["supports_positive_edge"].any()) if not sell.empty else False
    # Stability: share of ETF/months with positive mean signed net among those with actions.
    if not by_etf.empty and "n_actions" in by_etf.columns:
        etf_pos = by_etf[by_etf["n_actions"].fillna(0) > 0]
        etf_share = (
            float((etf_pos["mean_signed_net"] > 0).mean()) if not etf_pos.empty else None
        )
    else:
        etf_share = None
    if not by_month.empty and "n_actions" in by_month.columns:
        month_pos = by_month[by_month["n_actions"].fillna(0) > 0]
        month_share = (
            float((month_pos["mean_signed_net"] > 0).mean()) if not month_pos.empty else None
        )
    else:
        month_share = None
    pass_gate = bool(
        buy_ok
        and sell_ok
        and etf_share is not None
        and etf_share >= 0.60
        and month_share is not None
        and month_share >= 0.50
    )
    summary = {
        "holdout_months": sorted(holdout_months),
        "primary_hold_days": int(hold_days),
        "primary_cost_bps": int(primary_cost_bps),
        "n_predictions": int(len(work)),
        "n_mature": int(len(mature)),
        "buy_holdout": buy.iloc[0].to_dict() if not buy.empty else {},
        "sell_holdout": sell.iloc[0].to_dict() if not sell.empty else {},
        "etf_positive_share": etf_share,
        "month_positive_share": month_share,
        "pass_gate": pass_gate,
        "update_plot_labels": False,  # keep observe_only until lockbox passes
        "note": (
            "Development holdout is informational only; do not upgrade HTML bias "
            "wording until forward lockbox from 2026-07-21 passes."
        ),
    }
    return {
        "rows": metrics_df,
        "by_month": by_month,
        "by_etf": by_etf,
        "summary": summary,
    }


def build_equal_weight_event_equity(
    predictions: pd.DataFrame,
    *,
    open_panel: pd.DataFrame,
    close_panel: pd.DataFrame,
    hold_days: int = PRIMARY_HOLD_DAYS,
    cost_bps: int = PRIMARY_COST_BPS,
) -> pd.DataFrame:
    """Daily equal-weight equity from concurrent buy/sell event positions.

    Buy positions contribute +daily simple return from entry open through exit
    close; sell positions contribute the negative of that path (avoidance).
    Entry-day cost is charged once on the entry open date.
    """
    actions = predictions[
        predictions["trade_action"].isin(["buy_bias", "sell_bias"])
        & predictions["trade_label_mature"].fillna(False)
    ].copy()
    if actions.empty:
        return pd.DataFrame(
            columns=["date", "n_positions", "daily_ret", "equity", "drawdown"]
        )
    open_panel = open_panel.copy()
    close_panel = close_panel.copy()
    open_panel.index = pd.to_datetime(open_panel.index).normalize()
    close_panel.index = pd.to_datetime(close_panel.index).normalize()
    open_panel.columns = [str(c).upper() for c in open_panel.columns]
    close_panel.columns = [str(c).upper() for c in close_panel.columns]

    # Collect per-position daily signed returns keyed by date.
    daily_bags: dict[pd.Timestamp, list[float]] = {}
    cost = float(cost_bps) / 10000.0
    for _, row in actions.iterrows():
        code = str(row["code"]).upper()
        as_of = pd.Timestamp(row["as_of"]).normalize()
        action = str(row["trade_action"])
        if code not in open_panel.columns or code not in close_panel.columns:
            continue
        opens = pd.to_numeric(open_panel[code], errors="coerce").dropna()
        closes = pd.to_numeric(close_panel[code], errors="coerce").dropna()
        idx = opens.index.intersection(closes.index).sort_values()
        loc = _trading_loc(pd.DatetimeIndex(idx), as_of)
        if loc is None:
            continue
        entry_loc = loc + 1
        exit_loc = loc + int(hold_days)
        if entry_loc >= len(idx) or exit_loc >= len(idx):
            continue
        # Path: entry open → same-day close, then close-to-close until exit.
        sign = 1.0 if action == "buy_bias" else -1.0
        entry_date = pd.Timestamp(idx[entry_loc]).normalize()
        entry_open = float(opens.loc[entry_date])
        entry_close = float(closes.loc[entry_date])
        if entry_open <= 0 or entry_close <= 0:
            continue
        day0 = sign * (entry_close / entry_open - 1.0) - cost
        daily_bags.setdefault(entry_date, []).append(day0)
        prev_close = entry_close
        for j in range(entry_loc + 1, exit_loc + 1):
            d = pd.Timestamp(idx[j]).normalize()
            c = float(closes.loc[d])
            if prev_close <= 0 or c <= 0:
                break
            daily_bags.setdefault(d, []).append(sign * (c / prev_close - 1.0))
            prev_close = c

    if not daily_bags:
        return pd.DataFrame(
            columns=["date", "n_positions", "daily_ret", "equity", "drawdown"]
        )
    dates = sorted(daily_bags.keys())
    rows: list[dict[str, Any]] = []
    equity = 1.0
    peak = 1.0
    for d in dates:
        rets = daily_bags[d]
        daily = float(np.mean(rets)) if rets else 0.0
        equity *= 1.0 + daily
        peak = max(peak, equity)
        rows.append(
            {
                "date": d,
                "n_positions": int(len(rets)),
                "daily_ret": daily,
                "equity": equity,
                "drawdown": equity / peak - 1.0,
            }
        )
    return pd.DataFrame(rows)


def write_trade_bias_outputs(
    result: dict[str, Any],
    output_dir: Path | str,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    preds: pd.DataFrame = result["predictions"]
    keep_cols = [
        c
        for c in (
            "as_of",
            "code",
            "name",
            "month",
            "quantile_switch",
            "switch_side",
            "p_up",
            "trade_action",
            "buy_thr",
            "sell_thr",
            "threshold_reason",
            "model_trained",
            "model_train_end",
            "model_skip_reason",
            "hold_days",
            "trade_label_mature",
            "entry_open_date",
            "exit_close_date",
            "trade_ret_h1",
            "trade_ret_h3",
            "trade_ret_h5",
            "trade_ret_h10",
            "y_up",
            "schema",
            "feature_cols",
        )
        if c in preds.columns
    ]
    preds[keep_cols].to_csv(
        out / "trade_bias_predictions.csv", index=False, encoding="utf-8-sig"
    )
    metrics = result.get("metrics") or {}
    if isinstance(metrics.get("rows"), pd.DataFrame):
        metrics["rows"].to_csv(
            out / "trade_bias_metrics.csv", index=False, encoding="utf-8-sig"
        )
    if isinstance(metrics.get("by_month"), pd.DataFrame):
        metrics["by_month"].to_csv(
            out / "trade_bias_by_month.csv", index=False, encoding="utf-8-sig"
        )
    if isinstance(metrics.get("by_etf"), pd.DataFrame):
        metrics["by_etf"].to_csv(
            out / "trade_bias_by_etf.csv", index=False, encoding="utf-8-sig"
        )
    if metrics.get("summary") is not None:
        (out / "trade_bias_metrics.json").write_text(
            json.dumps(metrics["summary"], ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    equity = result.get("equity")
    if isinstance(equity, pd.DataFrame):
        equity.to_csv(out / "trade_bias_equity.csv", index=False, encoding="utf-8-sig")
    manifest = result.get("manifest") or {}
    (out / "trade_bias_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return out


def load_trade_bias_predictions(path: Path | str | None) -> pd.DataFrame | None:
    """Load OOS trade-bias predictions; return None when missing/invalid."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        frame = pd.read_csv(p)
    except Exception:
        return None
    if frame.empty:
        return None
    if "as_of" not in frame.columns or "code" not in frame.columns:
        return None
    if "trade_action" not in frame.columns:
        return None
    if "schema" in frame.columns:
        schemas = set(frame["schema"].dropna().astype(str).unique().tolist())
        if schemas and TRADE_BIAS_SCHEMA not in schemas:
            return None
    frame["as_of"] = pd.to_datetime(frame["as_of"]).dt.normalize()
    frame["code"] = frame["code"].astype(str).str.upper()
    return frame
