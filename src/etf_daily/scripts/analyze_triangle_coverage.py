#!/usr/bin/env python3
"""Validate red/green triangle coverage vs T+10 reversal events and retune q*.

Selection rule (select window only): maximize event recall subject to
event precision >= precision_floor. Holdout is evaluated once.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.lib.regime_transition_assessment import (
    _event_metrics_for_switch,
    default_qstar_grid,
    evaluate_threshold_on_frame,
    split_time_windows,
    switch_mask_from_cdf,
)
from etf_daily.lib.extreme_drop_override import coverage_split_stats
from etf_daily.lib.regime_transition_validation import (
    cluster_positive_events,
    match_alerts_to_events,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--validation-dir",
        type=Path,
        default=Path(
            "runtime/decision_packs/20260720/regime_transition_validation_q90_to0720"
        ),
    )
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--lead", type=int, default=3)
    p.add_argument("--precision-floor", type=float, default=0.20)
    p.add_argument("--baseline-q", type=float, default=0.90)
    p.add_argument("--eval-end", type=str, default=None)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: <validation-dir>/triangle_coverage",
    )
    p.add_argument("--bootstrap-n", type=int, default=200)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    return p.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (_PROJECT_ROOT / path)


def _bool_series(series: pd.Series) -> pd.Series:
    return series.map(lambda v: False if pd.isna(v) else bool(v))


def load_frame(validation_dir: Path, eval_end: str | None) -> pd.DataFrame:
    oos_path = validation_dir / "signals_oos.csv"
    rev_path = validation_dir / "signals_with_reversal_labels.csv"
    if not oos_path.exists():
        raise FileNotFoundError(f"missing {oos_path}")
    if not rev_path.exists():
        raise FileNotFoundError(f"missing {rev_path}")

    oos = pd.read_csv(oos_path, low_memory=False)
    rev = pd.read_csv(rev_path, low_memory=False)
    oos["as_of"] = pd.to_datetime(oos["as_of"]).dt.normalize()
    rev["as_of"] = pd.to_datetime(rev["as_of"]).dt.normalize()
    oos["code"] = oos["code"].astype(str).str.upper()
    rev["code"] = rev["code"].astype(str).str.upper()

    keep_oos = [
        c
        for c in [
            "as_of",
            "code",
            "name",
            "pred_cdf",
            "quantile_switch",
            "model_quantile_switch",
            "extreme_drop_switch",
            "switch_side",
            "switch_source",
            "day_ret",
            "regime_hmm_raw",
            "regime_now",
            "return_z",
            "signal_level",
        ]
        if c in oos.columns
    ]
    keep_rev = [
        c
        for c in [
            "as_of",
            "code",
            "label_switch_reversal",
            "turn_kind",
            "jump_role",
            "prior_ret",
            "jump_ret",
            "post_ret",
            "final_confirmed_on",
        ]
        if c in rev.columns
    ]
    frame = oos[keep_oos].merge(rev[keep_rev], on=["as_of", "code"], how="left")
    if eval_end is not None:
        end = pd.Timestamp(eval_end).normalize()
        frame = frame[frame["as_of"] <= end].copy()

    frame["pred_cdf"] = pd.to_numeric(frame["pred_cdf"], errors="coerce")
    frame["label_mature"] = frame["label_switch_reversal"].notna()
    frame["label_switch_reversal"] = _bool_series(frame["label_switch_reversal"])
    if "quantile_switch" in frame.columns:
        frame["quantile_switch"] = _bool_series(frame["quantile_switch"])
    if "model_quantile_switch" in frame.columns:
        frame["model_quantile_switch"] = _bool_series(frame["model_quantile_switch"])
    if "extreme_drop_switch" in frame.columns:
        frame["extreme_drop_switch"] = _bool_series(frame["extreme_drop_switch"])
    frame["turn_kind"] = frame.get("turn_kind", "none")
    frame["turn_kind"] = frame["turn_kind"].fillna("none").astype(str)
    frame["month"] = frame["as_of"].dt.to_period("M").astype(str)
    frame["year"] = frame["as_of"].dt.year
    if "name" not in frame.columns:
        frame["name"] = frame["code"]
    if "regime_hmm_raw" not in frame.columns:
        frame["regime_hmm_raw"] = frame.get("regime_now")
    return frame.sort_values(["code", "as_of"], kind="stable").reset_index(drop=True)


def build_events(mature: pd.DataFrame) -> list:
    label_frame = mature[["code", "as_of", "label_switch_reversal"]].rename(
        columns={"label_switch_reversal": "label_trend_up"}
    )
    return cluster_positive_events(label_frame)


def metrics_for_mask(
    mature: pd.DataFrame,
    mask: pd.Series,
    *,
    events: list | None,
    horizon: int,
    lead: int,
) -> dict[str, Any]:
    return _event_metrics_for_switch(
        mature,
        mask,
        horizon=horizon,
        lead=lead,
        events=events,
    )


def dense_q_grid(
    coarse: tuple[float, ...] | None = None,
    fine_lo: float = 0.82,
    fine_hi: float = 0.94,
    fine_step: float = 0.005,
) -> tuple[float, ...]:
    base = list(coarse or default_qstar_grid())
    n = int(round((fine_hi - fine_lo) / fine_step)) + 1
    fine = [round(fine_lo + i * fine_step, 3) for i in range(n)]
    vals = sorted(set(base + fine))
    return tuple(vals)


def sweep_unified(
    mature: pd.DataFrame,
    *,
    q_values: tuple[float, ...],
    events: list,
    horizon: int,
    lead: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for q in q_values:
        for side in ("both", "up", "down"):
            mask = switch_mask_from_cdf(mature["pred_cdf"], q_star=float(q), side=side)
            stats = metrics_for_mask(
                mature, mask, events=events, horizon=horizon, lead=lead
            )
            rows.append(
                {
                    "mode": "unified",
                    "side": side,
                    "q_star": float(q),
                    "q_up": float(q) if side in {"both", "up"} else None,
                    "q_down": (1.0 - float(q)) if side in {"both", "down"} else None,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def sweep_asymmetric(
    mature: pd.DataFrame,
    *,
    q_values: tuple[float, ...],
    events: list,
    horizon: int,
    lead: int,
    precision_floor: float,
    top_n_per_side: int = 8,
) -> pd.DataFrame:
    """Evaluate asymmetric (q_up, q_down) from top recall candidates per side."""
    up_rows = []
    down_rows = []
    for q in q_values:
        up_mask = switch_mask_from_cdf(mature["pred_cdf"], q_star=float(q), side="up")
        dn_mask = switch_mask_from_cdf(mature["pred_cdf"], q_star=float(q), side="down")
        up_stats = metrics_for_mask(
            mature, up_mask, events=events, horizon=horizon, lead=lead
        )
        dn_stats = metrics_for_mask(
            mature, dn_mask, events=events, horizon=horizon, lead=lead
        )
        up_rows.append({"q_star": float(q), **up_stats})
        down_rows.append({"q_star": float(q), **dn_stats})
    up_df = pd.DataFrame(up_rows)
    dn_df = pd.DataFrame(down_rows)

    def _top(df: pd.DataFrame) -> pd.DataFrame:
        ok = df[
            (df["precision"].fillna(0) >= precision_floor)
            & (df["n_alerts"].fillna(0) >= 10)
        ].copy()
        if ok.empty:
            ok = df[df["n_alerts"].fillna(0) > 0].copy()
        return ok.sort_values(
            by=["recall", "precision", "fp_rate", "n_alerts"],
            ascending=[False, False, True, True],
            kind="stable",
        ).head(top_n_per_side)

    cand_up = _top(up_df)
    cand_dn = _top(dn_df)
    rows: list[dict[str, Any]] = []
    for _, ur in cand_up.iterrows():
        for _, dr in cand_dn.iterrows():
            q_up = float(ur["q_star"])
            q_down = 1.0 - float(dr["q_star"])
            mask = switch_mask_from_cdf(
                mature["pred_cdf"], q_up=q_up, q_down=q_down, side="both"
            )
            stats = metrics_for_mask(
                mature, mask, events=events, horizon=horizon, lead=lead
            )
            rows.append(
                {
                    "mode": "asymmetric",
                    "side": "both",
                    "q_star": None,
                    "q_up": q_up,
                    "q_down": q_down,
                    "q_down_star": float(dr["q_star"]),
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def pick_best(
    table: pd.DataFrame,
    *,
    precision_floor: float,
    min_alerts: int = 15,
) -> dict[str, Any] | None:
    if table.empty:
        return None
    both = table[table["side"] == "both"].copy()
    if both.empty:
        return None
    feasible = both[
        (both["precision"].fillna(0) >= precision_floor)
        & (both["n_alerts"].fillna(0) >= min_alerts)
        & both["recall"].notna()
    ].copy()
    if feasible.empty:
        return None
    ranked = feasible.sort_values(
        by=["recall", "precision", "fp_rate", "duplicate_rate", "n_alerts"],
        ascending=[False, False, True, True, True],
        kind="stable",
    )
    row = ranked.iloc[0]
    return {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}


def group_baseline(
    mature: pd.DataFrame,
    *,
    events: list,
    horizon: int,
    lead: int,
    baseline_q: float,
) -> dict[str, Any]:
    mask = switch_mask_from_cdf(mature["pred_cdf"], q_star=baseline_q, side="both")
    overall = metrics_for_mask(
        mature, mask, events=events, horizon=horizon, lead=lead
    )
    by_side = {}
    for side in ("up", "down"):
        m = switch_mask_from_cdf(mature["pred_cdf"], q_star=baseline_q, side=side)
        by_side[side] = metrics_for_mask(
            mature, m, events=events, horizon=horizon, lead=lead
        )

    # Direction of events via turn_kind on event onset days.
    onset_rows = []
    for ev in events:
        sub = mature[
            (mature["code"] == ev.code) & (mature["as_of"] == pd.Timestamp(ev.onset))
        ]
        turn = str(sub["turn_kind"].iloc[0]) if not sub.empty else "none"
        onset_rows.append(
            {
                "code": ev.code,
                "onset": pd.Timestamp(ev.onset),
                "turn_kind": turn,
            }
        )
    onset_df = pd.DataFrame(onset_rows)
    turn_counts = (
        onset_df["turn_kind"].value_counts(dropna=False).to_dict()
        if not onset_df.empty
        else {}
    )

    by_year = []
    for year, grp in mature.groupby("year", sort=True):
        evs = build_events(grp)
        m = switch_mask_from_cdf(grp["pred_cdf"], q_star=baseline_q, side="both")
        stats = metrics_for_mask(grp, m, events=evs, horizon=horizon, lead=lead)
        by_year.append({"year": int(year), **stats})

    by_regime = []
    if "regime_hmm_raw" in mature.columns:
        for regime, grp in mature.groupby(mature["regime_hmm_raw"].fillna("unknown"), sort=True):
            evs = build_events(grp)
            m = switch_mask_from_cdf(grp["pred_cdf"], q_star=baseline_q, side="both")
            stats = metrics_for_mask(grp, m, events=evs, horizon=horizon, lead=lead)
            by_regime.append({"regime": str(regime), **stats})

    by_etf = []
    for code, grp in mature.groupby("code", sort=True):
        evs = build_events(grp)
        m = switch_mask_from_cdf(grp["pred_cdf"], q_star=baseline_q, side="both")
        stats = metrics_for_mask(grp, m, events=evs, horizon=horizon, lead=lead)
        name = str(grp["name"].iloc[0]) if "name" in grp.columns else code
        by_etf.append({"code": code, "name": name, **stats})
    by_etf_df = pd.DataFrame(by_etf).sort_values(
        by=["recall", "n_events"], ascending=[True, False], kind="stable"
    )

    return {
        "overall": overall,
        "by_side": by_side,
        "event_turn_counts": {str(k): int(v) for k, v in turn_counts.items()},
        "by_year": by_year,
        "by_regime": by_regime,
        "by_etf_worst": by_etf_df.head(12).to_dict(orient="records"),
        "by_etf_best": by_etf_df.tail(8).iloc[::-1].to_dict(orient="records"),
    }


def _calendar_by_code(mature: pd.DataFrame) -> dict[str, pd.DatetimeIndex]:
    out: dict[str, pd.DatetimeIndex] = {}
    for code, grp in mature.groupby("code", sort=False):
        out[str(code)] = pd.DatetimeIndex(
            sorted(pd.to_datetime(grp["as_of"]).dt.normalize().unique())
        )
    return out


def classify_misses(
    mature: pd.DataFrame,
    *,
    events: list,
    baseline_q: float,
    candidate: dict[str, Any] | None,
    horizon: int,
    lead: int,
) -> pd.DataFrame:
    """For each event, record whether baseline/candidate cover it and near-window CDF."""
    cal = _calendar_by_code(mature)
    base_mask = switch_mask_from_cdf(mature["pred_cdf"], q_star=baseline_q, side="both")
    if candidate is None:
        cand_mask = pd.Series(False, index=mature.index)
        cand_label = "none"
    else:
        cand_mask = switch_mask_from_cdf(
            mature["pred_cdf"],
            q_star=candidate.get("q_star"),
            q_up=candidate.get("q_up"),
            q_down=candidate.get("q_down"),
            side="both",
        )
        if candidate.get("mode") == "asymmetric":
            cand_label = f"asym_up{candidate.get('q_up')}_dn{candidate.get('q_down')}"
        else:
            cand_label = f"q{candidate.get('q_star')}"

    base_alerts = mature.loc[base_mask].copy()
    cand_alerts = mature.loc[cand_mask].copy()
    for alerts in (base_alerts, cand_alerts):
        if alerts.empty:
            continue
        alerts["switch_score"] = (alerts["pred_cdf"] - 0.5).abs().fillna(0.0)

    base_matched = (
        match_alerts_to_events(
            base_alerts,
            events,
            lead=lead,
            horizon=horizon,
            score_col="switch_score",
            calendar_by_code=cal,
        )
        if not base_alerts.empty
        else pd.DataFrame()
    )
    cand_matched = (
        match_alerts_to_events(
            cand_alerts,
            events,
            lead=lead,
            horizon=horizon,
            score_col="switch_score",
            calendar_by_code=cal,
        )
        if not cand_alerts.empty
        else pd.DataFrame()
    )

    def _covered_onsets(matched: pd.DataFrame) -> set[tuple[str, pd.Timestamp]]:
        if matched.empty or "matched_event_onset" not in matched.columns:
            return set()
        hit = matched[matched["matched_event_onset"].notna()]
        return {
            (str(r["code"]), pd.Timestamp(r["matched_event_onset"]).normalize())
            for _, r in hit.iterrows()
        }

    base_hit = _covered_onsets(base_matched)
    cand_hit = _covered_onsets(cand_matched)

    rows: list[dict[str, Any]] = []
    for ev in events:
        code = str(ev.code)
        onset = pd.Timestamp(ev.onset).normalize()
        sub = mature[(mature["code"] == code)].copy()
        if sub.empty:
            continue
        dates = list(cal.get(code, pd.DatetimeIndex([])))
        if onset not in dates:
            continue
        i = dates.index(onset)
        lo = max(0, i - lead)
        hi = min(len(dates) - 1, i + horizon)
        window_dates = set(dates[lo : hi + 1])
        win = sub[sub["as_of"].isin(window_dates)]
        cdf = pd.to_numeric(win["pred_cdf"], errors="coerce")
        max_cdf = float(cdf.max()) if cdf.notna().any() else None
        min_cdf = float(cdf.min()) if cdf.notna().any() else None
        near = win[win["as_of"] == onset]
        turn = str(near["turn_kind"].iloc[0]) if not near.empty else "none"
        name = str(near["name"].iloc[0]) if not near.empty and "name" in near.columns else code
        covered_base = (code, onset) in base_hit
        covered_cand = (code, onset) in cand_hit

        # Miss taxonomy relative to candidate (or baseline if no candidate).
        thr_up = float(candidate.get("q_up") if candidate else baseline_q)
        thr_dn = float(
            candidate.get("q_down") if candidate else (1.0 - baseline_q)
        )
        if covered_cand if candidate is not None else covered_base:
            miss_class = "covered"
        elif max_cdf is None or min_cdf is None:
            miss_class = "no_cdf"
        elif max_cdf >= thr_up or min_cdf <= thr_dn:
            # Extreme CDF exists in window but matching/side failed.
            # Check if extremes are on the "wrong" side vs turn_kind.
            if turn == "bottom_reversal" and max_cdf < thr_up and min_cdf <= thr_dn:
                miss_class = "direction_mismatch"
            elif turn == "top_reversal" and min_cdf > thr_dn and max_cdf >= thr_up:
                miss_class = "direction_mismatch"
            else:
                miss_class = "match_window_miss"
        else:
            # How close?
            up_gap = thr_up - max_cdf if max_cdf is not None else None
            dn_gap = min_cdf - thr_dn if min_cdf is not None else None
            near_gap = None
            for g in (up_gap, dn_gap):
                if g is not None and g >= 0:
                    near_gap = g if near_gap is None else min(near_gap, g)
            if near_gap is not None and near_gap <= 0.05:
                miss_class = "threshold_near_miss"
            else:
                miss_class = "no_extreme_cdf"

        rows.append(
            {
                "code": code,
                "name": name,
                "onset": str(onset.date()),
                "turn_kind": turn,
                "covered_baseline": covered_base,
                "covered_candidate": covered_cand,
                "newly_covered": (not covered_base) and covered_cand,
                "still_missed": not (
                    covered_cand if candidate is not None else covered_base
                ),
                "window_max_cdf": max_cdf,
                "window_min_cdf": min_cdf,
                "miss_class": miss_class,
                "candidate": cand_label,
            }
        )
    return pd.DataFrame(rows)


def new_false_positives(
    mature: pd.DataFrame,
    *,
    events: list,
    baseline_q: float,
    candidate: dict[str, Any],
    horizon: int,
    lead: int,
) -> pd.DataFrame:
    cal = _calendar_by_code(mature)
    base_mask = switch_mask_from_cdf(mature["pred_cdf"], q_star=baseline_q, side="both")
    cand_mask = switch_mask_from_cdf(
        mature["pred_cdf"],
        q_star=candidate.get("q_star"),
        q_up=candidate.get("q_up"),
        q_down=candidate.get("q_down"),
        side="both",
    )
    new_mask = cand_mask & ~base_mask
    alerts = mature.loc[new_mask].copy()
    if alerts.empty:
        return alerts
    alerts["switch_score"] = (alerts["pred_cdf"] - 0.5).abs().fillna(0.0)
    matched = match_alerts_to_events(
        alerts,
        events,
        lead=lead,
        horizon=horizon,
        score_col="switch_score",
        calendar_by_code=cal,
    )
    out = matched.copy()
    keep = [
        c
        for c in [
            "as_of",
            "code",
            "name",
            "pred_cdf",
            "switch_side",
            "matched_event_onset",
            "detection_delay",
            "is_fp",
            "is_duplicate",
        ]
        if c in out.columns
    ]
    out = out[keep]
    out["as_of"] = pd.to_datetime(out["as_of"]).dt.strftime("%Y-%m-%d")
    if "matched_event_onset" in out.columns:
        out["matched_event_onset"] = pd.to_datetime(
            out["matched_event_onset"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
    return out.sort_values(["is_fp", "as_of", "code"], ascending=[False, True, True])


def bootstrap_metric_ci_for_config(
    mature: pd.DataFrame,
    *,
    q_star: float | None,
    q_up: float | None,
    q_down: float | None,
    metric: str,
    horizon: int,
    lead: int,
    n_boot: int,
    seed: int,
) -> dict[str, float | None]:
    if mature.empty or "month" not in mature.columns:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    months = sorted(mature["month"].astype(str).unique().tolist())

    def _metric(sub: pd.DataFrame) -> float | None:
        evs = build_events(sub)
        mask = switch_mask_from_cdf(
            sub["pred_cdf"], q_star=q_star, q_up=q_up, q_down=q_down, side="both"
        )
        stats = metrics_for_mask(sub, mask, events=evs, horizon=horizon, lead=lead)
        val = stats.get(metric)
        return float(val) if val is not None and math.isfinite(float(val)) else None

    point = _metric(mature)
    if len(months) < 3:
        return {"mean": point, "ci_low": point, "ci_high": point, "n": len(months)}
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        chosen = rng.choice(months, size=len(months), replace=True)
        sub = mature[mature["month"].astype(str).isin(set(chosen))]
        if sub.empty:
            continue
        m = _metric(sub)
        if m is not None:
            boots.append(m)
    if not boots:
        return {"mean": point, "ci_low": None, "ci_high": None, "n": len(months)}
    return {
        "mean": point,
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
        "n": len(months),
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        if obj is None or (isinstance(obj, float) and not math.isfinite(obj)):
            return None
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if pd.isna(obj):
        return None
    return obj


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validation_dir = _resolve(args.validation_dir)
    out_dir = _resolve(args.output_dir) if args.output_dir else (
        validation_dir / "triangle_coverage"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] loading frame from {validation_dir}")
    frame = load_frame(validation_dir, args.eval_end)
    mature = frame[frame["label_mature"]].copy()
    print(
        f"[INFO] rows={len(frame)} mature={len(mature)} "
        f"codes={frame['code'].nunique()} "
        f"signal={frame['as_of'].min().date()}→{frame['as_of'].max().date()} "
        f"mature_end={mature['as_of'].max().date() if not mature.empty else None}"
    )

    events = build_events(mature)
    print(f"[INFO] T+10 reversal events={len(events)}")

    baseline = group_baseline(
        mature,
        events=events,
        horizon=args.horizon,
        lead=args.lead,
        baseline_q=args.baseline_q,
    )
    print(
        "[INFO] baseline q*=%.2f recall=%.3f precision=%.3f alerts=%s events=%s"
        % (
            args.baseline_q,
            baseline["overall"].get("recall") or 0.0,
            baseline["overall"].get("precision") or 0.0,
            baseline["overall"].get("n_alerts"),
            baseline["overall"].get("n_events"),
        )
    )

    q_values = dense_q_grid()
    splits = split_time_windows(frame)
    select = splits["select"]
    holdout = splits["holdout"]
    select_events = build_events(select)
    holdout_events = build_events(holdout)
    print(
        f"[INFO] select months={splits['select_months']} "
        f"holdout months={splits['holdout_months']}"
    )

    print("[INFO] sweeping unified thresholds on select window")
    select_unified = sweep_unified(
        select,
        q_values=q_values,
        events=select_events,
        horizon=args.horizon,
        lead=args.lead,
    )
    print("[INFO] sweeping asymmetric thresholds on select window")
    select_asym = sweep_asymmetric(
        select,
        q_values=q_values,
        events=select_events,
        horizon=args.horizon,
        lead=args.lead,
        precision_floor=args.precision_floor,
    )
    parts = [df for df in (select_unified, select_asym) if not df.empty]
    select_all = (
        pd.concat(parts, ignore_index=True, sort=False)
        if parts
        else pd.DataFrame()
    )
    best_unified = pick_best(
        select_unified, precision_floor=args.precision_floor
    )
    best_asym = pick_best(select_asym, precision_floor=args.precision_floor)
    # Prefer higher recall; tie-break already inside pick_best.
    candidates = [c for c in (best_unified, best_asym) if c is not None]
    if candidates:
        recommended = sorted(
            candidates,
            key=lambda r: (
                float(r.get("recall") or 0.0),
                float(r.get("precision") or 0.0),
                -float(r.get("fp_rate") or 1.0),
            ),
            reverse=True,
        )[0]
    else:
        recommended = None

    # Full-mature frontier for charting (both-side unified).
    print("[INFO] sweeping full mature frontier")
    full_unified = sweep_unified(
        mature,
        q_values=q_values,
        events=events,
        horizon=args.horizon,
        lead=args.lead,
    )
    frontier = full_unified[full_unified["side"] == "both"].copy()

    # Holdout evaluation
    baseline_holdout = evaluate_threshold_on_frame(
        holdout,
        q_star=args.baseline_q,
        side="both",
        horizon=args.horizon,
        label="baseline_q90_holdout",
    )
    # Recompute with shared event clustering for consistency.
    baseline_holdout = {
        **baseline_holdout,
        **metrics_for_mask(
            holdout,
            switch_mask_from_cdf(holdout["pred_cdf"], q_star=args.baseline_q, side="both"),
            events=holdout_events,
            horizon=args.horizon,
            lead=args.lead,
        ),
        "label": "baseline_q90_holdout",
        "q_star": args.baseline_q,
        "q_up": args.baseline_q,
        "q_down": 1.0 - args.baseline_q,
    }

    if recommended is None:
        recommended_holdout = None
        holdout_improves_recall = False
        holdout_precision_ok = False
    else:
        rec_mask = switch_mask_from_cdf(
            holdout["pred_cdf"],
            q_star=recommended.get("q_star"),
            q_up=recommended.get("q_up"),
            q_down=recommended.get("q_down"),
            side="both",
        )
        recommended_holdout = {
            "label": "recommended_holdout",
            "mode": recommended.get("mode"),
            "q_star": recommended.get("q_star"),
            "q_up": recommended.get("q_up"),
            "q_down": recommended.get("q_down"),
            **metrics_for_mask(
                holdout,
                rec_mask,
                events=holdout_events,
                horizon=args.horizon,
                lead=args.lead,
            ),
        }
        holdout_improves_recall = float(
            recommended_holdout.get("recall") or 0.0
        ) > float(baseline_holdout.get("recall") or 0.0)
        holdout_precision_ok = float(
            recommended_holdout.get("precision") or 0.0
        ) >= args.precision_floor

    print("[INFO] bootstrap holdout CIs")
    boot_base_recall = bootstrap_metric_ci_for_config(
        holdout,
        q_star=args.baseline_q,
        q_up=args.baseline_q,
        q_down=1.0 - args.baseline_q,
        metric="recall",
        horizon=args.horizon,
        lead=args.lead,
        n_boot=args.bootstrap_n,
        seed=args.bootstrap_seed,
    )
    boot_base_prec = bootstrap_metric_ci_for_config(
        holdout,
        q_star=args.baseline_q,
        q_up=args.baseline_q,
        q_down=1.0 - args.baseline_q,
        metric="precision",
        horizon=args.horizon,
        lead=args.lead,
        n_boot=args.bootstrap_n,
        seed=args.bootstrap_seed + 1,
    )
    if recommended is not None:
        boot_rec_recall = bootstrap_metric_ci_for_config(
            holdout,
            q_star=recommended.get("q_star"),
            q_up=recommended.get("q_up"),
            q_down=recommended.get("q_down"),
            metric="recall",
            horizon=args.horizon,
            lead=args.lead,
            n_boot=args.bootstrap_n,
            seed=args.bootstrap_seed + 2,
        )
        boot_rec_prec = bootstrap_metric_ci_for_config(
            holdout,
            q_star=recommended.get("q_star"),
            q_up=recommended.get("q_up"),
            q_down=recommended.get("q_down"),
            metric="precision",
            horizon=args.horizon,
            lead=args.lead,
            n_boot=args.bootstrap_n,
            seed=args.bootstrap_seed + 3,
        )
    else:
        boot_rec_recall = {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
        boot_rec_prec = {"mean": None, "ci_low": None, "ci_high": None, "n": 0}

    print("[INFO] classifying misses / new false positives")
    miss_df = classify_misses(
        mature,
        events=events,
        baseline_q=args.baseline_q,
        candidate=recommended,
        horizon=args.horizon,
        lead=args.lead,
    )
    fp_df = (
        new_false_positives(
            mature,
            events=events,
            baseline_q=args.baseline_q,
            candidate=recommended,
            horizon=args.horizon,
            lead=args.lead,
        )
        if recommended is not None
        else pd.DataFrame()
    )

    promote = bool(
        recommended is not None
        and holdout_improves_recall
        and holdout_precision_ok
    )
    if recommended is None:
        decision = "keep_q90_no_feasible_candidate"
        decision_note = (
            f"No threshold met precision>={args.precision_floor:.0%} "
            "on the select window while raising recall."
        )
    elif promote:
        decision = "consider_update"
        decision_note = (
            "Holdout recall improved and precision stayed at/above the floor; "
            "consider updating production switch_quantile_q / asymmetric thresholds."
        )
    else:
        decision = "keep_q90"
        decision_note = (
            "Select-window candidate exists, but holdout did not jointly "
            f"improve recall while keeping precision>={args.precision_floor:.0%}."
        )

    summary = {
        "validation_dir": str(validation_dir),
        "signal_start": str(frame["as_of"].min().date()),
        "signal_end": str(frame["as_of"].max().date()),
        "label_mature_end": str(mature["as_of"].max().date()) if not mature.empty else None,
        "horizon": args.horizon,
        "lead": args.lead,
        "precision_floor": args.precision_floor,
        "baseline_q": args.baseline_q,
        "n_rows": int(len(frame)),
        "n_mature": int(len(mature)),
        "n_codes": int(frame["code"].nunique()),
        "n_events": int(len(events)),
        "coverage_model_vs_effective": coverage_split_stats(frame),
        "splits": {
            "select_months": splits["select_months"],
            "holdout_months": splits["holdout_months"],
        },
        "baseline": baseline,
        "recommended_select": recommended,
        "best_unified_select": best_unified,
        "best_asymmetric_select": best_asym,
        "holdout": {
            "baseline": baseline_holdout,
            "recommended": recommended_holdout,
            "improves_recall": holdout_improves_recall,
            "precision_ok": holdout_precision_ok,
            "bootstrap_baseline_recall": boot_base_recall,
            "bootstrap_baseline_precision": boot_base_prec,
            "bootstrap_recommended_recall": boot_rec_recall,
            "bootstrap_recommended_precision": boot_rec_prec,
        },
        "decision": decision,
        "decision_note": decision_note,
        "promote": promote,
        "miss_class_counts": (
            miss_df["miss_class"].value_counts().to_dict() if not miss_df.empty else {}
        ),
        "n_newly_covered": int(miss_df["newly_covered"].sum()) if not miss_df.empty else 0,
        "n_still_missed": int(miss_df["still_missed"].sum()) if not miss_df.empty else 0,
        "n_new_alerts": int(len(fp_df)),
        "n_new_false_positives": (
            int(fp_df["is_fp"].fillna(False).astype(bool).sum()) if not fp_df.empty else 0
        ),
    }

    # Persist tables
    frontier.to_csv(out_dir / "threshold_frontier_full.csv", index=False)
    select_all.to_csv(out_dir / "threshold_sweep_select.csv", index=False)
    miss_df.to_csv(out_dir / "event_coverage_detail.csv", index=False)
    if not fp_df.empty:
        fp_df.to_csv(out_dir / "new_alerts_vs_baseline.csv", index=False)
    pd.DataFrame(baseline["by_etf_worst"]).to_csv(
        out_dir / "baseline_by_etf_worst.csv", index=False
    )
    (out_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Compact markdown
    lines = [
        "# Triangle coverage vs T+10 reversal events",
        "",
        f"- signal: {summary['signal_start']} → {summary['signal_end']}",
        f"- mature end: `{summary['label_mature_end']}`",
        f"- events: {summary['n_events']} / mature rows: {summary['n_mature']}",
        f"- precision floor: {args.precision_floor:.0%}",
        f"- decision: **{decision}**",
        f"- note: {decision_note}",
        "",
        "## Model vs effective triangle coverage",
        (
            f"- model_quantile_switch rate="
            f"{summary['coverage_model_vs_effective'].get('model_switch_rate')}"
        ),
        (
            f"- extreme_drop_override n="
            f"{summary['coverage_model_vs_effective'].get('n_extreme_override')} "
            f"rate={summary['coverage_model_vs_effective'].get('extreme_override_rate')}"
        ),
        (
            f"- effective quantile_switch rate="
            f"{summary['coverage_model_vs_effective'].get('effective_switch_rate')}"
        ),
        (
            "- note: model q* audits use `model_quantile_switch` / pred_cdf; "
            "OPS/plots consume effective `quantile_switch`"
        ),
        "",
        "## Baseline q*=0.90 (full mature)",
        f"- alerts={baseline['overall'].get('n_alerts')} "
        f"precision={baseline['overall'].get('precision')} "
        f"recall={baseline['overall'].get('recall')} "
        f"f1={baseline['overall'].get('f1')}",
        f"- up: precision={baseline['by_side']['up'].get('precision')} "
        f"recall={baseline['by_side']['up'].get('recall')}",
        f"- down: precision={baseline['by_side']['down'].get('precision')} "
        f"recall={baseline['by_side']['down'].get('recall')}",
        "",
        "## Recommended (select window)",
    ]
    if recommended is None:
        lines.append("- none")
    else:
        lines.append(
            f"- mode={recommended.get('mode')} q_star={recommended.get('q_star')} "
            f"q_up={recommended.get('q_up')} q_down={recommended.get('q_down')}"
        )
        lines.append(
            f"- select precision={recommended.get('precision')} "
            f"recall={recommended.get('recall')} alerts={recommended.get('n_alerts')}"
        )
    lines.extend(
        [
            "",
            "## Holdout",
            f"- baseline recall={baseline_holdout.get('recall')} "
            f"precision={baseline_holdout.get('precision')}",
        ]
    )
    if recommended_holdout is not None:
        lines.append(
            f"- recommended recall={recommended_holdout.get('recall')} "
            f"precision={recommended_holdout.get('precision')}"
        )
        lines.append(
            f"- bootstrap recommended recall CI="
            f"[{boot_rec_recall.get('ci_low')}, {boot_rec_recall.get('ci_high')}]"
        )
    lines.extend(
        [
            "",
            "## Miss taxonomy",
            f"- counts: `{summary['miss_class_counts']}`",
            f"- newly covered vs baseline: {summary['n_newly_covered']}",
            f"- still missed: {summary['n_still_missed']}",
            f"- new alerts / new FP: {summary['n_new_alerts']} / {summary['n_new_false_positives']}",
            "",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[INFO] wrote outputs → {out_dir}")
    print(f"[INFO] decision={decision} promote={promote}")
    if recommended is not None:
        print(
            f"[INFO] recommended mode={recommended.get('mode')} "
            f"q_up={recommended.get('q_up')} q_down={recommended.get('q_down')} "
            f"select_recall={recommended.get('recall')} "
            f"select_precision={recommended.get('precision')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
