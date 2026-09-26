"""Gate-driven bias promotion: T+5 horse race + bias_promotion.json.

Triangle 偏买/偏卖 may appear only when promotion_source == \"lockbox\" and the
corresponding side passes hit_ci_low > 0.5 with n_scored >= 20 on
T+1 open → T+5 close. Holdout results are informational only.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from decision_pack.src.regime_transition_assessment import block_bootstrap_mean_ci
from decision_pack.src.regime_transition_trade_bias import (
    PRIMARY_HOLD_DAYS,
    attach_trade_labels,
)
from decision_pack.src.regime_transition_validation import (
    causal_bias_from_prior,
    causal_bias_from_switch_side,
)

SCHEMA_VERSION = 1
MIN_N_SCORED = 20
CANDIDATE_RULES = (
    "baseline_switch_side",
    "prior_bias_strong",
    "prior_bias",
    "trade_bias",
)
# Tie-break priority (earlier wins).
RULE_PRIORITY = {r: i for i, r in enumerate(CANDIDATE_RULES)}
DEFAULT_LOCKBOX_START = "2026-07-21"
MIN_LOCKBOX_TRADING_DAYS = 20
PROMOTION_FILENAME = "bias_promotion.json"
RACE_CSV_FILENAME = "candidate_race_t5.csv"


def eval_rule_bias(
    rule: str | None,
    *,
    quantile_switch: bool,
    switch_side: str | None = None,
    prior_ret: float | None = None,
    thr_prior: float | None = None,
    trade_action: str | None = None,
) -> str | None:
    """Map one candidate rule to buy_bias / sell_bias / None."""
    if not quantile_switch:
        return None
    r = str(rule or "")
    if r == "baseline_switch_side":
        return causal_bias_from_switch_side(
            quantile_switch=True, switch_side=switch_side
        )
    if r == "prior_bias":
        return causal_bias_from_prior(
            quantile_switch=True,
            prior_ret=prior_ret,
            thr_prior=thr_prior,
            require_strong=False,
        )
    if r == "prior_bias_strong":
        return causal_bias_from_prior(
            quantile_switch=True,
            prior_ret=prior_ret,
            thr_prior=thr_prior,
            require_strong=True,
        )
    if r == "trade_bias":
        action = str(trade_action or "")
        if action in {"buy_bias", "sell_bias"}:
            return action
        return None
    return None


def attach_forward_returns(
    switches: pd.DataFrame,
    open_panel: pd.DataFrame,
    close_panel: pd.DataFrame,
    *,
    hold_days: int = PRIMARY_HOLD_DAYS,
) -> pd.DataFrame:
    """Attach T+1 open→T+hold close return once for all candidates."""
    work = switches.copy()
    if "quantile_switch" not in work.columns:
        work["quantile_switch"] = True
    labeled = attach_trade_labels(
        work,
        open_panel,
        close_panel,
        hold_days=int(hold_days),
        aux_hold_days=(),
    )
    col = f"trade_ret_h{int(hold_days)}"
    out = work.copy()
    out["r_t1_open_t5_close"] = pd.to_numeric(labeled[col], errors="coerce")
    out["trade_label_mature"] = labeled["trade_label_mature"].fillna(False).astype(bool)
    if "month" not in out.columns:
        out["month"] = pd.to_datetime(out["as_of"]).dt.to_period("M").astype(str)
    return out


def _row_prior(row: Mapping[str, Any]) -> float | None:
    for key in ("causal_prior_ret", "prior_ret"):
        if key in row and row[key] is not None and not (
            isinstance(row[key], float) and pd.isna(row[key])
        ):
            try:
                v = float(row[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                return v
    return None


def _row_thr(row: Mapping[str, Any]) -> float | None:
    for key in ("causal_thr_prior", "thr_prior"):
        if key in row and row[key] is not None and not (
            isinstance(row[key], float) and pd.isna(row[key])
        ):
            try:
                v = float(row[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                return v
    return None


def score_candidate(
    switches: pd.DataFrame,
    *,
    rule: str,
    side: str,
    window: str,
    n_boot: int = 400,
    seed: int = 42,
) -> dict[str, Any]:
    """Score one rule/side on rows that already have r_t1_open_t5_close."""
    if side not in {"buy_bias", "sell_bias"}:
        raise ValueError(f"side must be buy_bias|sell_bias, got {side}")
    work = switches.copy()
    n_switch = int(len(work))
    mature = work[work.get("trade_label_mature", False).fillna(False)].copy()
    hits: list[float] = []
    for _, row in mature.iterrows():
        bias = eval_rule_bias(
            rule,
            quantile_switch=bool(row.get("quantile_switch", True)),
            switch_side=(
                None
                if pd.isna(row.get("switch_side"))
                else str(row.get("switch_side"))
            ),
            prior_ret=_row_prior(row),
            thr_prior=_row_thr(row),
            trade_action=(
                None
                if pd.isna(row.get("trade_action"))
                else str(row.get("trade_action"))
            ),
        )
        if bias != side:
            continue
        r = row.get("r_t1_open_t5_close")
        if r is None or (isinstance(r, float) and pd.isna(r)):
            continue
        rv = float(r)
        if not math.isfinite(rv):
            continue
        if side == "buy_bias":
            hits.append(1.0 if rv > 0.0 else 0.0)
        else:
            hits.append(1.0 if rv < 0.0 else 0.0)
    hit_arr = np.asarray(hits, dtype=float)
    n_scored = int(hit_arr.size)
    coverage = float(n_scored / n_switch) if n_switch else 0.0
    hit_ci = block_bootstrap_mean_ci(hit_arr, n_boot=n_boot, seed=seed)
    hit_rate = float(np.mean(hit_arr)) if n_scored else None
    ci_low = hit_ci.get("ci_low")
    ci_high = hit_ci.get("ci_high")
    passed = bool(
        n_scored >= MIN_N_SCORED
        and ci_low is not None
        and float(ci_low) > 0.5
    )
    return {
        "window": window,
        "rule": rule,
        "side": side,
        "n_switch": n_switch,
        "n_scored": n_scored,
        "coverage": coverage,
        "hit_rate": hit_rate,
        "hit_ci_low": None if ci_low is None else float(ci_low),
        "hit_ci_high": None if ci_high is None else float(ci_high),
        "pass": passed,
    }


def _filter_window(
    switches: pd.DataFrame,
    *,
    window: str,
    holdout_months: list[str] | None,
    lockbox_start: str | None,
) -> pd.DataFrame:
    work = switches.copy()
    work["as_of"] = pd.to_datetime(work["as_of"]).dt.normalize()
    if "month" not in work.columns:
        work["month"] = work["as_of"].dt.to_period("M").astype(str)
    if window == "holdout":
        months = {str(m) for m in (holdout_months or [])}
        if not months:
            return work.iloc[0:0].copy()
        return work[work["month"].astype(str).isin(months)].copy()
    if window == "lockbox":
        if not lockbox_start:
            return work.iloc[0:0].copy()
        start = pd.Timestamp(lockbox_start).normalize()
        return work[work["as_of"] >= start].copy()
    raise ValueError(f"unknown window {window}")


def run_candidate_race(
    switches_with_returns: pd.DataFrame,
    *,
    holdout_months: list[str],
    lockbox_start: str = DEFAULT_LOCKBOX_START,
    rules: tuple[str, ...] = CANDIDATE_RULES,
) -> pd.DataFrame:
    """Score all rules × sides for holdout and lockbox windows."""
    rows: list[dict[str, Any]] = []
    for window in ("holdout", "lockbox"):
        sub = _filter_window(
            switches_with_returns,
            window=window,
            holdout_months=holdout_months,
            lockbox_start=lockbox_start,
        )
        for rule in rules:
            for side in ("buy_bias", "sell_bias"):
                rows.append(
                    score_candidate(sub, rule=rule, side=side, window=window)
                )
    return pd.DataFrame(rows)


def pick_side_winner(race: pd.DataFrame, *, window: str, side: str) -> str | None:
    """Return winning rule id for a side in a window, or None."""
    if race.empty:
        return None
    sub = race[
        (race["window"].astype(str) == window)
        & (race["side"].astype(str) == side)
        & (race["pass"].fillna(False).astype(bool))
    ].copy()
    if sub.empty:
        return None
    sub = sub.sort_values(
        by=["hit_rate", "coverage"],
        ascending=[False, False],
        kind="stable",
    )
    best_hit = float(sub.iloc[0]["hit_rate"])
    best_cov = float(sub.iloc[0]["coverage"])
    tied = sub[
        (sub["hit_rate"].astype(float) >= best_hit - 1e-12)
        & (sub["coverage"].astype(float) >= best_cov - 1e-12)
    ]
    tied = tied.copy()
    tied["_prio"] = tied["rule"].map(lambda r: RULE_PRIORITY.get(str(r), 99))
    tied = tied.sort_values("_prio", kind="stable")
    return str(tied.iloc[0]["rule"])


def lockbox_eligible(
    switches_with_returns: pd.DataFrame,
    *,
    lockbox_start: str,
    min_trading_days: int = MIN_LOCKBOX_TRADING_DAYS,
    min_n_scored: int = MIN_N_SCORED,
    race: pd.DataFrame | None = None,
) -> bool:
    """True when lockbox has enough mature calendar span and evaluable n."""
    sub = _filter_window(
        switches_with_returns,
        window="lockbox",
        holdout_months=None,
        lockbox_start=lockbox_start,
    )
    mature = sub[sub.get("trade_label_mature", False).fillna(False)]
    if mature.empty:
        return False
    n_days = int(mature["as_of"].nunique())
    if n_days < int(min_trading_days):
        return False
    if race is not None and not race.empty:
        lock = race[race["window"].astype(str) == "lockbox"]
        if lock.empty:
            return False
        if int(lock["n_scored"].max()) < int(min_n_scored):
            # Still eligible to *evaluate*, but design says:
            # eligible = (days >= 20) AND (at least one side has a candidate with n>=20)
            # So require some rule/side reaches n>=20.
            return False
        return bool((lock["n_scored"] >= int(min_n_scored)).any())
    return True


def build_bias_promotion(
    race: pd.DataFrame,
    *,
    validation_dir: Path | str,
    holdout_months: list[str],
    lockbox_start: str = DEFAULT_LOCKBOX_START,
    lockbox_end: str | None = None,
    switches_with_returns: pd.DataFrame | None = None,
    hold_days: int = PRIMARY_HOLD_DAYS,
) -> dict[str, Any]:
    """Build bias_promotion.json payload (holdout informational; lockbox authoritative)."""
    holdout_buy_pass = bool(
        not race.empty
        and race[
            (race["window"] == "holdout")
            & (race["side"] == "buy_bias")
            & (race["pass"].fillna(False))
        ].shape[0]
        > 0
    )
    holdout_sell_pass = bool(
        not race.empty
        and race[
            (race["window"] == "holdout")
            & (race["side"] == "sell_bias")
            & (race["pass"].fillna(False))
        ].shape[0]
        > 0
    )
    eligible = False
    if switches_with_returns is not None:
        eligible = lockbox_eligible(
            switches_with_returns,
            lockbox_start=lockbox_start,
            race=race,
        )
    lockbox_buy_rule = pick_side_winner(race, window="lockbox", side="buy_bias") if eligible else None
    lockbox_sell_rule = pick_side_winner(race, window="lockbox", side="sell_bias") if eligible else None
    lockbox_buy_pass = lockbox_buy_rule is not None
    lockbox_sell_pass = lockbox_sell_rule is not None

    promotion_source = "none"
    buy_rule = None
    sell_rule = None
    buy_pass = False
    sell_pass = False
    if eligible and (lockbox_buy_pass or lockbox_sell_pass):
        promotion_source = "lockbox"
        buy_rule = lockbox_buy_rule
        sell_rule = lockbox_sell_rule
        buy_pass = lockbox_buy_pass
        sell_pass = lockbox_sell_pass

    holdout_preview_buy = pick_side_winner(race, window="holdout", side="buy_bias")
    holdout_preview_sell = pick_side_winner(race, window="holdout", side="sell_bias")

    hold_start = min(holdout_months) if holdout_months else None
    hold_end = max(holdout_months) if holdout_months else None

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "validation_dir": str(validation_dir),
        "primary_hold_days": int(hold_days),
        "execution": "t1_open_to_t5_close",
        "gate": {"metric": "hit_ci_low_gt_0.5", "min_n_scored": MIN_N_SCORED},
        "holdout": {
            "start": hold_start,
            "end": hold_end,
            "months": list(holdout_months),
            "buy_pass_gate": holdout_buy_pass,
            "sell_pass_gate": holdout_sell_pass,
        },
        "holdout_preview_buy_rule": holdout_preview_buy,
        "holdout_preview_sell_rule": holdout_preview_sell,
        "lockbox": {
            "start": lockbox_start,
            "end": lockbox_end,
            "min_trading_days_mature": MIN_LOCKBOX_TRADING_DAYS,
            "eligible": bool(eligible),
            "buy_pass_gate": lockbox_buy_pass,
            "sell_pass_gate": lockbox_sell_pass,
        },
        "promotion_source": promotion_source,
        "buy_rule": buy_rule,
        "sell_rule": sell_rule,
        "buy_pass_gate": buy_pass,
        "sell_pass_gate": sell_pass,
        "candidates_path": f"assessment/{RACE_CSV_FILENAME}",
        "notes": "Plot may promote only when promotion_source==lockbox",
    }


def write_bias_promotion_outputs(
    race: pd.DataFrame,
    promotion: dict[str, Any],
    output_dir: Path | str,
) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    race_path = out / RACE_CSV_FILENAME
    promo_path = out / PROMOTION_FILENAME
    race.to_csv(race_path, index=False, encoding="utf-8-sig")
    promo_path.write_text(
        json.dumps(promotion, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return race_path, promo_path


def load_bias_promotion(path: Path | str | None) -> dict[str, Any]:
    """Load promotion JSON; missing/bad → safe none source."""
    none_doc = {
        "schema_version": SCHEMA_VERSION,
        "promotion_source": "none",
        "buy_rule": None,
        "sell_rule": None,
        "buy_pass_gate": False,
        "sell_pass_gate": False,
    }
    if path is None:
        return dict(none_doc)
    p = Path(path)
    if not p.exists():
        return dict(none_doc)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(none_doc)
    if not isinstance(data, dict):
        return dict(none_doc)
    if int(data.get("schema_version") or 0) != SCHEMA_VERSION:
        return dict(none_doc)
    source = str(data.get("promotion_source") or "none")
    if source != "lockbox":
        data = dict(data)
        data["promotion_source"] = "none"
        data["buy_rule"] = None
        data["sell_rule"] = None
        data["buy_pass_gate"] = False
        data["sell_pass_gate"] = False
        return data
    # lockbox but require flags
    data = dict(data)
    if not data.get("buy_pass_gate"):
        data["buy_rule"] = None
    if not data.get("sell_pass_gate"):
        data["sell_rule"] = None
    return data


def default_promotion_path(validation_dir: Path | str) -> Path:
    return Path(validation_dir) / "assessment" / PROMOTION_FILENAME


def prepare_switch_universe(
    signals: pd.DataFrame,
    trade_bias_predictions: pd.DataFrame | None = None,
    prior_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """OOS quantile switches with optional trade_action / prior merges."""
    work = signals.copy()
    work["as_of"] = pd.to_datetime(work["as_of"]).dt.normalize()
    work["code"] = work["code"].astype(str).str.upper()
    if "quantile_switch" not in work.columns:
        raise ValueError("signals missing quantile_switch")
    work["quantile_switch"] = work["quantile_switch"].fillna(False).astype(bool)
    switches = work[work["quantile_switch"]].copy()

    if prior_frame is not None and not prior_frame.empty:
        pr = prior_frame.copy()
        pr["as_of"] = pd.to_datetime(pr["as_of"]).dt.normalize()
        pr["code"] = pr["code"].astype(str).str.upper()
        keep = [
            c
            for c in (
                "as_of",
                "code",
                "causal_prior_ret",
                "causal_thr_prior",
                "prior_ret",
                "thr_prior",
                "causal_prior_strong",
            )
            if c in pr.columns
        ]
        if len(keep) > 2:
            pr = pr[keep].drop_duplicates(subset=["as_of", "code"], keep="last")
            drop_overlap = [c for c in keep if c not in {"as_of", "code"} and c in switches.columns]
            if drop_overlap:
                switches = switches.drop(columns=drop_overlap)
            switches = switches.merge(pr, on=["as_of", "code"], how="left")

    if trade_bias_predictions is not None and not trade_bias_predictions.empty:
        tb = trade_bias_predictions.copy()
        tb["as_of"] = pd.to_datetime(tb["as_of"]).dt.normalize()
        tb["code"] = tb["code"].astype(str).str.upper()
        keep = [c for c in ("as_of", "code", "trade_action", "p_up") if c in tb.columns]
        tb = tb[keep].drop_duplicates(subset=["as_of", "code"], keep="last")
        switches = switches.drop(columns=["trade_action"], errors="ignore")
        switches = switches.merge(tb, on=["as_of", "code"], how="left")
    elif "trade_action" not in switches.columns:
        switches["trade_action"] = None
    return switches


def run_bias_promotion_pipeline(
    signals: pd.DataFrame,
    open_panel: pd.DataFrame,
    close_panel: pd.DataFrame,
    *,
    validation_dir: Path | str,
    output_dir: Path | str,
    holdout_months: list[str],
    lockbox_start: str = DEFAULT_LOCKBOX_START,
    trade_bias_predictions: pd.DataFrame | None = None,
    prior_frame: pd.DataFrame | None = None,
    hold_days: int = PRIMARY_HOLD_DAYS,
) -> dict[str, Any]:
    """End-to-end: attach returns → race → write JSON/CSV."""
    switches = prepare_switch_universe(
        signals,
        trade_bias_predictions,
        prior_frame=prior_frame,
    )
    with_ret = attach_forward_returns(
        switches, open_panel, close_panel, hold_days=hold_days
    )
    race = run_candidate_race(
        with_ret,
        holdout_months=holdout_months,
        lockbox_start=lockbox_start,
    )
    lockbox_end = None
    mature = with_ret[with_ret["trade_label_mature"].fillna(False)]
    if not mature.empty:
        lock_sub = mature[mature["as_of"] >= pd.Timestamp(lockbox_start).normalize()]
        if not lock_sub.empty:
            lockbox_end = str(pd.Timestamp(lock_sub["as_of"].max()).date())
    promotion = build_bias_promotion(
        race,
        validation_dir=validation_dir,
        holdout_months=holdout_months,
        lockbox_start=lockbox_start,
        lockbox_end=lockbox_end,
        switches_with_returns=with_ret,
        hold_days=hold_days,
    )
    race_path, promo_path = write_bias_promotion_outputs(race, promotion, output_dir)
    return {
        "race": race,
        "promotion": promotion,
        "race_path": race_path,
        "promotion_path": promo_path,
        "switches": with_ret,
    }
