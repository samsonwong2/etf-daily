#!/usr/bin/env python3
"""Tune triangle alerts with pred_cdf soft tails + return_z gates.

Compares:
  - baseline q*=0.90
  - and_soft: soft_q=0.80 AND |return_z| gates
  - or_rescue: baseline OR (soft_q=0.80 AND |return_z| gates)

Selection: maximize event recall subject to precision >= floor on select months.
Holdout is evaluated once with month-block bootstrap CIs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from decision_pack.src.regime_transition_assessment import split_time_windows
from decision_pack.src.regime_transition_validation import match_alerts_to_events
from decision_pack.src.triangle_cdf_z_rules import (
    CdfZRule,
    build_rule_masks,
    default_z_grid,
    direction_aligned_event_stats,
    iter_cdf_z_rules,
)


def _load_coverage_helpers():
    path = Path(__file__).resolve().parent / "analyze_triangle_coverage.py"
    spec = importlib.util.spec_from_file_location("triangle_coverage_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load helpers from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_helpers = _load_coverage_helpers()
_calendar_by_code = _helpers._calendar_by_code
_jsonable = _helpers._jsonable
_resolve = _helpers._resolve
build_events = _helpers.build_events
load_frame = _helpers.load_frame
metrics_for_mask = _helpers.metrics_for_mask


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--validation-dir",
        type=Path,
        default=Path(
            "workspace/decision_packs/20260720/regime_transition_validation_q90_to0720"
        ),
    )
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--lead", type=int, default=3)
    p.add_argument("--precision-floor", type=float, default=0.20)
    p.add_argument("--baseline-q", type=float, default=0.90)
    p.add_argument("--soft-q", type=float, default=0.80)
    p.add_argument("--eval-end", type=str, default=None)
    p.add_argument("--z-start", type=float, default=0.5)
    p.add_argument("--z-stop", type=float, default=3.0)
    p.add_argument("--z-step", type=float, default=0.25)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: <validation-dir>/triangle_coverage_cdf_z",
    )
    p.add_argument("--bootstrap-n", type=int, default=200)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--min-alerts", type=int, default=15)
    return p.parse_args(argv)


def evaluate_rule(
    mature: pd.DataFrame,
    rule: CdfZRule,
    *,
    events: list,
    horizon: int,
    lead: int,
) -> dict[str, Any]:
    alert, up, down = build_rule_masks(mature, rule)
    stats = metrics_for_mask(
        mature, alert, events=events, horizon=horizon, lead=lead
    )
    direction = direction_aligned_event_stats(
        events=events,
        mature=mature,
        up_mask=up,
        down_mask=down,
        lead=lead,
        horizon=horizon,
    )
    return {
        **rule.to_dict(),
        **stats,
        **direction,
        "n_up_alerts": int(up.sum()),
        "n_down_alerts": int(down.sum()),
    }


def sweep_rules(
    mature: pd.DataFrame,
    rules: list[CdfZRule],
    *,
    events: list,
    horizon: int,
    lead: int,
) -> pd.DataFrame:
    rows = [
        evaluate_rule(mature, rule, events=events, horizon=horizon, lead=lead)
        for rule in rules
    ]
    return pd.DataFrame(rows)


def pick_best_rule(
    table: pd.DataFrame,
    *,
    precision_floor: float,
    min_alerts: int,
    exclude_baseline: bool = True,
) -> dict[str, Any] | None:
    if table.empty:
        return None
    work = table.copy()
    if exclude_baseline:
        work = work[work["mode"] != "baseline"]
    feasible = work[
        (work["precision"].fillna(0) >= precision_floor)
        & (work["n_alerts"].fillna(0) >= min_alerts)
        & work["recall"].notna()
    ].copy()
    if feasible.empty:
        return None
    ranked = feasible.sort_values(
        by=[
            "recall",
            "direction_recall",
            "precision",
            "fp_rate",
            "duplicate_rate",
            "n_alerts",
        ],
        ascending=[False, False, False, True, True, True],
        kind="stable",
    )
    row = ranked.iloc[0]
    return {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}


def rule_from_row(row: dict[str, Any]) -> CdfZRule:
    return CdfZRule(
        mode=str(row["mode"]),
        baseline_q=float(row.get("baseline_q") or 0.90),
        soft_q=float(row.get("soft_q") or 0.80),
        z_up=float(row.get("z_up") or 0.0),
        z_down=float(row.get("z_down") or 0.0),
    )


def bootstrap_rule_metric(
    mature: pd.DataFrame,
    rule: CdfZRule,
    *,
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
        stats = evaluate_rule(sub, rule, events=evs, horizon=horizon, lead=lead)
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


def classify_cdf_z_misses(
    mature: pd.DataFrame,
    *,
    events: list,
    baseline: CdfZRule,
    candidate: CdfZRule | None,
    horizon: int,
    lead: int,
) -> pd.DataFrame:
    cal = _calendar_by_code(mature)
    base_mask, base_up, base_dn = build_rule_masks(mature, baseline)
    if candidate is None:
        cand_mask = pd.Series(False, index=mature.index)
        cand_up = cand_mask
        cand_dn = cand_mask
        cand_label = "none"
        soft_q = float(baseline.soft_q)
        z_up = 0.0
        z_down = 0.0
    else:
        cand_mask, cand_up, cand_dn = build_rule_masks(mature, candidate)
        cand_label = candidate.label()
        soft_q = float(candidate.soft_q)
        z_up = float(candidate.z_up)
        z_down = float(candidate.z_down)

    def _hit_set(mask: pd.Series) -> set[tuple[str, pd.Timestamp]]:
        alerts = mature.loc[mask].copy()
        if alerts.empty:
            return set()
        alerts = alerts.copy()
        alerts["switch_score"] = (
            pd.to_numeric(alerts["pred_cdf"], errors="coerce") - 0.5
        ).abs().fillna(0.0)
        matched = match_alerts_to_events(
            alerts,
            events,
            lead=lead,
            horizon=horizon,
            score_col="switch_score",
            calendar_by_code=cal,
        )
        if matched.empty or "matched_event_onset" not in matched.columns:
            return set()
        hit = matched[matched["matched_event_onset"].notna()]
        return {
            (str(r["code"]), pd.Timestamp(r["matched_event_onset"]).normalize())
            for _, r in hit.iterrows()
        }

    base_hit = _hit_set(base_mask)
    cand_hit = _hit_set(cand_mask)

    rows: list[dict[str, Any]] = []
    for ev in events:
        code = str(ev.code)
        onset = pd.Timestamp(ev.onset).normalize()
        dates = list(cal.get(code, pd.DatetimeIndex([])))
        if onset not in dates:
            continue
        i = dates.index(onset)
        lo = max(0, i - lead)
        hi = min(len(dates) - 1, i + horizon)
        window_dates = set(dates[lo : hi + 1])
        win = mature[(mature["code"] == code) & (mature["as_of"].isin(window_dates))]
        near = mature[(mature["code"] == code) & (mature["as_of"] == onset)]
        turn = str(near["turn_kind"].iloc[0]) if not near.empty else "none"
        name = (
            str(near["name"].iloc[0])
            if not near.empty and "name" in near.columns
            else code
        )
        cdf = pd.to_numeric(win["pred_cdf"], errors="coerce")
        zz = pd.to_numeric(win["return_z"], errors="coerce")
        max_cdf = float(cdf.max()) if cdf.notna().any() else None
        min_cdf = float(cdf.min()) if cdf.notna().any() else None
        max_z = float(zz.max()) if zz.notna().any() else None
        min_z = float(zz.min()) if zz.notna().any() else None

        covered_base = (code, onset) in base_hit
        covered_cand = (code, onset) in cand_hit
        still_missed = not (covered_cand if candidate is not None else covered_base)

        if not still_missed:
            miss_class = "covered"
        elif max_cdf is None or min_cdf is None:
            miss_class = "no_cdf"
        else:
            soft_up = soft_q
            soft_dn = 1.0 - soft_q
            has_up_cdf = max_cdf >= soft_up
            has_dn_cdf = min_cdf <= soft_dn
            has_up_z = max_z is not None and max_z >= z_up
            has_dn_z = min_z is not None and min_z <= -z_down
            if turn == "top_reversal":
                if not has_up_cdf:
                    miss_class = "cdf_not_extreme"
                elif not has_up_z:
                    miss_class = "z_not_extreme"
                elif has_dn_cdf and not has_up_cdf:
                    miss_class = "direction_mismatch"
                else:
                    miss_class = "match_window_miss"
            elif turn == "bottom_reversal":
                if not has_dn_cdf:
                    miss_class = "cdf_not_extreme"
                elif not has_dn_z:
                    miss_class = "z_not_extreme"
                elif has_up_cdf and not has_dn_cdf:
                    miss_class = "direction_mismatch"
                else:
                    miss_class = "match_window_miss"
            else:
                if not (has_up_cdf or has_dn_cdf):
                    miss_class = "cdf_not_extreme"
                elif not (has_up_z or has_dn_z):
                    miss_class = "z_not_extreme"
                else:
                    miss_class = "match_window_miss"

        rows.append(
            {
                "code": code,
                "name": name,
                "onset": str(onset.date()),
                "turn_kind": turn,
                "covered_baseline": covered_base,
                "covered_candidate": covered_cand,
                "newly_covered": (not covered_base) and covered_cand,
                "still_missed": still_missed,
                "window_max_cdf": max_cdf,
                "window_min_cdf": min_cdf,
                "window_max_z": max_z,
                "window_min_z": min_z,
                "miss_class": miss_class,
                "candidate": cand_label,
            }
        )
    return pd.DataFrame(rows)


def new_false_positives(
    mature: pd.DataFrame,
    *,
    events: list,
    baseline: CdfZRule,
    candidate: CdfZRule,
    horizon: int,
    lead: int,
) -> pd.DataFrame:
    cal = _calendar_by_code(mature)
    base_mask, _, _ = build_rule_masks(mature, baseline)
    cand_mask, cand_up, cand_dn = build_rule_masks(mature, candidate)
    new_mask = cand_mask & ~base_mask
    alerts = mature.loc[new_mask].copy()
    if alerts.empty:
        return alerts
    side = pd.Series(pd.NA, index=alerts.index, dtype=object)
    side = side.mask(cand_dn.reindex(alerts.index).fillna(False), "down")
    side = side.mask(cand_up.reindex(alerts.index).fillna(False), "up")
    alerts = alerts.copy()
    alerts["switch_side"] = side
    alerts["switch_score"] = (
        pd.to_numeric(alerts["pred_cdf"], errors="coerce") - 0.5
    ).abs().fillna(0.0)
    matched = match_alerts_to_events(
        alerts,
        events,
        lead=lead,
        horizon=horizon,
        score_col="switch_score",
        calendar_by_code=cal,
    )
    keep = [
        c
        for c in [
            "as_of",
            "code",
            "name",
            "pred_cdf",
            "return_z",
            "switch_side",
            "matched_event_onset",
            "detection_delay",
            "is_fp",
            "is_duplicate",
        ]
        if c in matched.columns
    ]
    out = matched[keep].copy()
    out["as_of"] = pd.to_datetime(out["as_of"]).dt.strftime("%Y-%m-%d")
    if "matched_event_onset" in out.columns:
        out["matched_event_onset"] = pd.to_datetime(
            out["matched_event_onset"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
    return out.sort_values(["is_fp", "as_of", "code"], ascending=[False, True, True])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validation_dir = _resolve(args.validation_dir)
    out_dir = _resolve(args.output_dir) if args.output_dir else (
        validation_dir / "triangle_coverage_cdf_z"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] loading frame from {validation_dir}")
    frame = load_frame(validation_dir, args.eval_end)
    if "return_z" not in frame.columns:
        raise SystemExit("[ERROR] return_z missing from signals_oos merge")
    frame["return_z"] = pd.to_numeric(frame["return_z"], errors="coerce")
    mature = frame[frame["label_mature"]].copy()
    events = build_events(mature)
    print(
        f"[INFO] rows={len(frame)} mature={len(mature)} events={len(events)} "
        f"codes={frame['code'].nunique()} "
        f"{frame['as_of'].min().date()}→{frame['as_of'].max().date()} "
        f"mature_end={mature['as_of'].max().date()}"
    )

    baseline_rule = CdfZRule(
        mode="baseline",
        baseline_q=args.baseline_q,
        soft_q=args.soft_q,
        z_up=0.0,
        z_down=0.0,
    )
    z_values = default_z_grid(args.z_start, args.z_stop, args.z_step)
    rules = iter_cdf_z_rules(
        baseline_q=args.baseline_q,
        soft_q=args.soft_q,
        z_values=z_values,
        modes=("and_soft", "or_rescue"),
    )
    print(f"[INFO] rules to sweep={len(rules)} z_grid={z_values}")

    splits = split_time_windows(frame)
    select = splits["select"]
    holdout = splits["holdout"]
    select_events = build_events(select)
    holdout_events = build_events(holdout)
    print(
        f"[INFO] select={splits['select_months']} holdout={splits['holdout_months']}"
    )

    print("[INFO] evaluating baseline on full mature")
    baseline_full = evaluate_rule(
        mature, baseline_rule, events=events, horizon=args.horizon, lead=args.lead
    )
    print(
        "[INFO] baseline recall=%.3f precision=%.3f dir_recall=%s alerts=%s"
        % (
            baseline_full.get("recall") or 0.0,
            baseline_full.get("precision") or 0.0,
            baseline_full.get("direction_recall"),
            baseline_full.get("n_alerts"),
        )
    )

    print("[INFO] sweeping select window")
    select_table = sweep_rules(
        select,
        rules,
        events=select_events,
        horizon=args.horizon,
        lead=args.lead,
    )
    recommended = pick_best_rule(
        select_table,
        precision_floor=args.precision_floor,
        min_alerts=args.min_alerts,
        exclude_baseline=True,
    )
    # Also keep best per mode for reporting.
    best_and = pick_best_rule(
        select_table[select_table["mode"] == "and_soft"],
        precision_floor=args.precision_floor,
        min_alerts=args.min_alerts,
        exclude_baseline=False,
    )
    best_or = pick_best_rule(
        select_table[select_table["mode"] == "or_rescue"],
        precision_floor=args.precision_floor,
        min_alerts=args.min_alerts,
        exclude_baseline=False,
    )

    baseline_holdout = evaluate_rule(
        holdout,
        baseline_rule,
        events=holdout_events,
        horizon=args.horizon,
        lead=args.lead,
    )

    if recommended is None:
        recommended_holdout = None
        rec_rule = None
        holdout_improves_recall = False
        holdout_precision_ok = False
        alert_inflation_ok = True
    else:
        rec_rule = rule_from_row(recommended)
        recommended_holdout = evaluate_rule(
            holdout,
            rec_rule,
            events=holdout_events,
            horizon=args.horizon,
            lead=args.lead,
        )
        holdout_improves_recall = float(
            recommended_holdout.get("recall") or 0.0
        ) > float(baseline_holdout.get("recall") or 0.0)
        holdout_precision_ok = float(
            recommended_holdout.get("precision") or 0.0
        ) >= args.precision_floor
        # Guard against pure alert spam: allow at most 1.75x baseline alerts.
        base_alerts = max(int(baseline_holdout.get("n_alerts") or 1), 1)
        rec_alerts = int(recommended_holdout.get("n_alerts") or 0)
        alert_inflation_ok = rec_alerts <= int(math.ceil(1.75 * base_alerts))

    print("[INFO] bootstrap holdout")
    boot_base_recall = bootstrap_rule_metric(
        holdout,
        baseline_rule,
        metric="recall",
        horizon=args.horizon,
        lead=args.lead,
        n_boot=args.bootstrap_n,
        seed=args.bootstrap_seed,
    )
    boot_base_prec = bootstrap_rule_metric(
        holdout,
        baseline_rule,
        metric="precision",
        horizon=args.horizon,
        lead=args.lead,
        n_boot=args.bootstrap_n,
        seed=args.bootstrap_seed + 1,
    )
    if rec_rule is not None:
        boot_rec_recall = bootstrap_rule_metric(
            holdout,
            rec_rule,
            metric="recall",
            horizon=args.horizon,
            lead=args.lead,
            n_boot=args.bootstrap_n,
            seed=args.bootstrap_seed + 2,
        )
        boot_rec_prec = bootstrap_rule_metric(
            holdout,
            rec_rule,
            metric="precision",
            horizon=args.horizon,
            lead=args.lead,
            n_boot=args.bootstrap_n,
            seed=args.bootstrap_seed + 3,
        )
        boot_rec_dir = bootstrap_rule_metric(
            holdout,
            rec_rule,
            metric="direction_recall",
            horizon=args.horizon,
            lead=args.lead,
            n_boot=args.bootstrap_n,
            seed=args.bootstrap_seed + 4,
        )
    else:
        boot_rec_recall = {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
        boot_rec_prec = {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
        boot_rec_dir = {"mean": None, "ci_low": None, "ci_high": None, "n": 0}

    print("[INFO] full-mature frontier for charting (symmetric z only)")
    # Compact frontier: and_soft / or_rescue with z_up=z_down.
    frontier_rules = [baseline_rule]
    for mode in ("and_soft", "or_rescue"):
        for z in z_values:
            frontier_rules.append(
                CdfZRule(
                    mode=mode,
                    baseline_q=args.baseline_q,
                    soft_q=args.soft_q,
                    z_up=float(z),
                    z_down=float(z),
                )
            )
    frontier = sweep_rules(
        mature,
        frontier_rules,
        events=events,
        horizon=args.horizon,
        lead=args.lead,
    )

    print("[INFO] classifying misses / new FPs")
    miss_df = classify_cdf_z_misses(
        mature,
        events=events,
        baseline=baseline_rule,
        candidate=rec_rule,
        horizon=args.horizon,
        lead=args.lead,
    )
    fp_df = (
        new_false_positives(
            mature,
            events=events,
            baseline=baseline_rule,
            candidate=rec_rule,
            horizon=args.horizon,
            lead=args.lead,
        )
        if rec_rule is not None
        else pd.DataFrame()
    )

    promote = bool(
        rec_rule is not None
        and holdout_improves_recall
        and holdout_precision_ok
        and alert_inflation_ok
    )
    if recommended is None:
        decision = "keep_q90_no_feasible_cdf_z"
        decision_note = (
            f"No CDF+Z rule met precision>={args.precision_floor:.0%} "
            "on select while beating baseline recall."
        )
    elif promote:
        decision = "consider_update_cdf_z"
        decision_note = (
            "Holdout recall improved, precision stayed at/above floor, "
            "and alert count did not explode; consider production CDF+Z rule."
        )
    else:
        decision = "keep_q90"
        reasons = []
        if not holdout_improves_recall:
            reasons.append("holdout recall not higher")
        if not holdout_precision_ok:
            reasons.append("holdout precision below floor")
        if not alert_inflation_ok:
            reasons.append("holdout alert inflation >1.75x")
        decision_note = "Select candidate exists, but " + "; ".join(reasons) + "."

    summary = {
        "validation_dir": str(validation_dir),
        "signal_start": str(frame["as_of"].min().date()),
        "signal_end": str(frame["as_of"].max().date()),
        "label_mature_end": str(mature["as_of"].max().date()),
        "horizon": args.horizon,
        "lead": args.lead,
        "precision_floor": args.precision_floor,
        "baseline_q": args.baseline_q,
        "soft_q": args.soft_q,
        "z_grid": list(z_values),
        "n_rows": int(len(frame)),
        "n_mature": int(len(mature)),
        "n_codes": int(frame["code"].nunique()),
        "n_events": int(len(events)),
        "splits": {
            "select_months": splits["select_months"],
            "holdout_months": splits["holdout_months"],
        },
        "baseline_full": baseline_full,
        "recommended_select": recommended,
        "best_and_soft_select": best_and,
        "best_or_rescue_select": best_or,
        "holdout": {
            "baseline": baseline_holdout,
            "recommended": recommended_holdout,
            "improves_recall": holdout_improves_recall,
            "precision_ok": holdout_precision_ok,
            "alert_inflation_ok": alert_inflation_ok,
            "bootstrap_baseline_recall": boot_base_recall,
            "bootstrap_baseline_precision": boot_base_prec,
            "bootstrap_recommended_recall": boot_rec_recall,
            "bootstrap_recommended_precision": boot_rec_prec,
            "bootstrap_recommended_direction_recall": boot_rec_dir,
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

    select_table.to_csv(out_dir / "cdf_z_sweep_select.csv", index=False)
    frontier.to_csv(out_dir / "cdf_z_frontier_full.csv", index=False)
    miss_df.to_csv(out_dir / "event_coverage_detail_cdf_z.csv", index=False)
    if not miss_df.empty:
        miss_df.loc[miss_df["newly_covered"].astype(bool)].to_csv(
            out_dir / "newly_covered_events.csv", index=False
        )
        miss_df.loc[miss_df["still_missed"].astype(bool)].to_csv(
            out_dir / "still_missed_events.csv", index=False
        )
    if not fp_df.empty:
        fp_df.to_csv(out_dir / "new_alerts_vs_baseline_cdf_z.csv", index=False)
        fp_df.loc[fp_df["is_fp"].fillna(False).astype(bool)].to_csv(
            out_dir / "new_false_positives.csv", index=False
        )
    (out_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Triangle coverage: CDF + return_z",
        "",
        f"- signal: {summary['signal_start']} → {summary['signal_end']}",
        f"- mature end: `{summary['label_mature_end']}`",
        f"- events: {summary['n_events']}",
        f"- soft_q={args.soft_q} baseline_q={args.baseline_q} "
        f"precision_floor={args.precision_floor:.0%}",
        f"- decision: **{decision}**",
        f"- note: {decision_note}",
        "",
        "## Baseline q*=0.90",
        f"- alerts={baseline_full.get('n_alerts')} precision={baseline_full.get('precision')} "
        f"recall={baseline_full.get('recall')} direction_recall={baseline_full.get('direction_recall')}",
        "",
        "## Recommended (select)",
    ]
    if recommended is None:
        lines.append("- none")
    else:
        lines.append(
            f"- {recommended.get('label')} mode={recommended.get('mode')} "
            f"z_up={recommended.get('z_up')} z_down={recommended.get('z_down')}"
        )
        lines.append(
            f"- select precision={recommended.get('precision')} "
            f"recall={recommended.get('recall')} "
            f"direction_recall={recommended.get('direction_recall')} "
            f"alerts={recommended.get('n_alerts')}"
        )
    lines.append("")
    lines.append("## Holdout")
    lines.append(
        f"- baseline recall={baseline_holdout.get('recall')} "
        f"precision={baseline_holdout.get('precision')} "
        f"direction_recall={baseline_holdout.get('direction_recall')}"
    )
    if recommended_holdout is not None:
        lines.append(
            f"- recommended recall={recommended_holdout.get('recall')} "
            f"precision={recommended_holdout.get('precision')} "
            f"direction_recall={recommended_holdout.get('direction_recall')}"
        )
        lines.append(
            f"- bootstrap recommended recall CI="
            f"[{boot_rec_recall.get('ci_low')}, {boot_rec_recall.get('ci_high')}]"
        )
        lines.append(
            f"- bootstrap recommended precision CI="
            f"[{boot_rec_prec.get('ci_low')}, {boot_rec_prec.get('ci_high')}]"
        )
    lines.extend(
        [
            "",
            "## Miss taxonomy",
            f"- counts: `{summary['miss_class_counts']}`",
            f"- newly covered: {summary['n_newly_covered']}",
            f"- still missed: {summary['n_still_missed']}",
            f"- new alerts / new FP: {summary['n_new_alerts']} / {summary['n_new_false_positives']}",
            "",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[INFO] wrote → {out_dir}")
    print(f"[INFO] decision={decision} promote={promote}")
    if recommended is not None:
        print(
            f"[INFO] recommended {recommended.get('label')} "
            f"select_recall={recommended.get('recall')} "
            f"select_precision={recommended.get('precision')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
