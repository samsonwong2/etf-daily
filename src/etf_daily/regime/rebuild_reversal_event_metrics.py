#!/usr/bin/env python3
"""Rebuild event matching under switch/reversal labels (not trend-up).

Reads an existing regime_transition_validation*_q90 signals_oos.csv + qlib
closes, attaches ``label_switch_reversal``, clusters reversal events, and
matches quantile-switch alerts to those events.

Example:
  PY=~/etf-daily-output/python/envs/py312/bin/python
  $PY src/etf_daily/regime/rebuild_reversal_event_metrics.py \\
    --validation-dir runtime/decision_packs/20260720/regime_transition_validation_q90
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.lib.regime_transition_live_snapshot import (
    LiveSnapshotError,
    apply_live_snapshot_to_close_panel,
    codes_requiring_live_coverage,
    load_live_snapshot,
    snapshot_meta,
)
from etf_daily.lib.regime_transition_validation import (
    EARLY_REVERSAL_HORIZON,
    attach_instrument_names,
    cluster_positive_events,
    compute_causal_prior_context,
    compute_dual_horizon_reversal_detail,
    match_alerts_to_events,
)
from etf_daily.pipeline.price_adjustments import auto_qfq_adjust_close_panel
from etf_daily.pipeline.strategy_layer_audit import load_qlib_close
from etf_daily.paths import FUND_LIST_CSV, QLIB_PROVIDER_URI
from etf_daily.lib.generate_daily_mu_position_report import load_fund_name_map


def _early_confirmation_counts(labeled: pd.DataFrame) -> dict[str, Any]:
    """Summarize early→final migration on dual-mature rows."""
    if labeled.empty or "early_to_final_status" not in labeled.columns:
        return {
            "n_dual_mature": 0,
            "n_early_positive": 0,
            "n_final_positive": 0,
            "n_confirmed": 0,
            "n_revoked": 0,
            "n_direction_changed": 0,
            "n_final_only": 0,
            "confirm_rate_given_early": None,
            "revoke_rate_given_early": None,
            "final_only_rate_given_final": None,
        }
    status = labeled["early_to_final_status"].astype(str)
    dual = labeled[
        ~status.isin(["immature", "early_only_immature_final"])
        & labeled["early_label_switch_reversal"].notna()
        & labeled["label_switch_reversal"].notna()
    ].copy()
    n_dual = int(len(dual))
    early_pos = dual[dual["early_label_switch_reversal"].fillna(False).astype(bool)]
    final_pos = dual[dual["label_switch_reversal"].fillna(False).astype(bool)]
    n_early = int(len(early_pos))
    n_final = int(len(final_pos))
    n_confirmed = int((dual["early_to_final_status"] == "confirmed").sum())
    n_revoked = int((dual["early_to_final_status"] == "revoked").sum())
    n_dir = int((dual["early_to_final_status"] == "direction_changed").sum())
    n_final_only = int((dual["early_to_final_status"] == "final_only").sum())
    return {
        "n_dual_mature": n_dual,
        "n_early_positive": n_early,
        "n_final_positive": n_final,
        "n_confirmed": n_confirmed,
        "n_revoked": n_revoked,
        "n_direction_changed": n_dir,
        "n_final_only": n_final_only,
        "confirm_rate_given_early": (n_confirmed / n_early) if n_early else None,
        "revoke_rate_given_early": (
            ((n_revoked + n_dir) / n_early) if n_early else None
        ),
        "final_only_rate_given_final": (n_final_only / n_final) if n_final else None,
    }


def _attach_reversal_labels(
    signals: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    lookback: int,
    horizon: int,
    early_horizon: int,
    kappa_prior: float,
    kappa_post: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for code, grp in signals.groupby("code", sort=False):
        if code not in panel.columns:
            continue
        close = pd.to_numeric(panel[code], errors="coerce").dropna()
        for _, row in grp.iterrows():
            detail = compute_dual_horizon_reversal_detail(
                close,
                pd.Timestamp(row["as_of"]),
                lookback=lookback,
                early_horizon=early_horizon,
                final_horizon=horizon,
                kappa_prior=kappa_prior,
                kappa_post=kappa_post,
            )
            out = dict(row)
            out.update(detail)
            prior_ctx = compute_causal_prior_context(
                close,
                pd.Timestamp(row["as_of"]),
                lookback=lookback,
                kappa_prior=kappa_prior,
            )
            if prior_ctx is None:
                out["causal_prior_ret"] = np.nan
                out["causal_jump_ret"] = np.nan
                out["causal_thr_prior"] = np.nan
                out["causal_prior_strong"] = False
            else:
                out["causal_prior_ret"] = prior_ctx["prior_ret"]
                out["causal_jump_ret"] = prior_ctx["jump_ret"]
                out["causal_thr_prior"] = prior_ctx["thr_prior"]
                out["causal_prior_strong"] = bool(prior_ctx["prior_strong"])
            rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path(
            "runtime/decision_packs/20260720/regime_transition_validation_q90"
        ),
    )
    parser.add_argument("--lookback", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument(
        "--early-horizon",
        type=int,
        default=EARLY_REVERSAL_HORIZON,
        help="early confirmation horizon (trading days after origin)",
    )
    parser.add_argument("--kappa-prior", type=float, default=0.35)
    parser.add_argument("--kappa-post", type=float, default=0.35)
    parser.add_argument("--lead", type=int, default=3)
    parser.add_argument(
        "--live-snapshot",
        type=Path,
        default=None,
        help="frozen AkShare OHLCV snapshot CSV (same file used by backtest)",
    )
    args = parser.parse_args()

    out_dir = args.validation_dir.expanduser().resolve()
    signals_path = out_dir / "signals_oos.csv"
    if not signals_path.exists():
        raise SystemExit(f"missing {signals_path}")

    signals = pd.read_csv(signals_path)
    codes = sorted(signals["code"].astype(str).unique().tolist())
    start = (
        pd.Timestamp(signals["as_of"].min()) - pd.tseries.offsets.BDay(40)
    ).strftime("%Y-%m-%d")
    signal_end = pd.Timestamp(signals["as_of"].max()).normalize()
    end = (
        signal_end + pd.tseries.offsets.BDay(args.horizon + 5)
    ).strftime("%Y-%m-%d")
    panel = load_qlib_close(str(QLIB_PROVIDER_URI), codes, start, end)
    panel, _ = auto_qfq_adjust_close_panel(panel)

    live_meta = None
    if args.live_snapshot is not None:
        snap_path = Path(args.live_snapshot).expanduser().resolve()
        # Prefer the snapshot's own as_of (provisional intraday bar). Signals may
        # lag by one session when instruments end_date still equals Qlib calendar
        # last; still inject the bar for price continuity / HTML.
        try:
            snap_peek = pd.read_csv(snap_path, usecols=["as_of"])
            snap_as_of = pd.to_datetime(snap_peek["as_of"].iloc[0]).strftime("%Y-%m-%d")
        except Exception:
            snap_as_of = signal_end.strftime("%Y-%m-%d")
        as_of = snap_as_of
        if pd.Timestamp(as_of) < signal_end:
            raise SystemExit(
                f"[ERROR] live snapshot as_of={as_of} is before signals max "
                f"{signal_end.date()}"
            )
        if pd.Timestamp(as_of) > signal_end:
            print(
                f"[WARN] live snapshot as_of={as_of} ahead of signals max "
                f"{signal_end.date()}; injecting provisional bar anyway"
            )
        try:
            required, skipped = codes_requiring_live_coverage(signals, as_of=as_of)
            if skipped:
                print(
                    f"[WARN] live snapshot coverage skips {len(skipped)} "
                    f"historical-only codes (not on as_of={as_of}); "
                    f"examples={skipped[:10]}"
                )
            snapshot = load_live_snapshot(
                snap_path,
                as_of=as_of,
                required_codes=required,
                strict_coverage=True,
            )
            panel, apply_stats = apply_live_snapshot_to_close_panel(
                panel, snapshot, as_of=as_of
            )
        except LiveSnapshotError as exc:
            raise SystemExit(f"[ERROR] live snapshot: {exc}") from exc
        live_meta = snapshot_meta(snapshot, path=snap_path)
        live_meta.update(
            {
                "injected_codes": apply_stats.n_codes,
                "appended": apply_stats.appended,
                "replaced_existing": apply_stats.replaced_existing,
            }
        )
        print(
            f"[INFO] live snapshot applied as_of={apply_stats.as_of} "
            f"codes={apply_stats.n_codes} captured_at={apply_stats.captured_at}"
        )

    labeled = _attach_reversal_labels(
        signals,
        panel,
        lookback=int(args.lookback),
        horizon=int(args.horizon),
        early_horizon=int(args.early_horizon),
        kappa_prior=float(args.kappa_prior),
        kappa_post=float(args.kappa_post),
    )

    # Cluster reversal events from positive labels
    label_frame = labeled[["code", "as_of", "label_switch_reversal"]].copy()
    events = cluster_positive_events(
        label_frame.dropna(subset=["label_switch_reversal"]),
        label_col="label_switch_reversal",
    )

    # Alerts = locked switch definition days
    if "quantile_switch" in labeled.columns:
        alerts = labeled[labeled["quantile_switch"].astype(bool)].copy()
    else:
        alerts = labeled[labeled["signal_level"].isin(["OBSERVE", "CONFIRM"])].copy()

    matched = match_alerts_to_events(
        alerts,
        events,
        lead=int(args.lead),
        horizon=int(args.horizon),
        score_col="pred_cdf" if "pred_cdf" in alerts.columns else "transition_score",
    )

    # Ensure Chinese names on alerts/matched + labeled overlay
    name_map = load_fund_name_map(FUND_LIST_CSV) if FUND_LIST_CSV.exists() else {}
    matched = attach_instrument_names(matched, name_map)
    labeled = attach_instrument_names(labeled, name_map)

    # Summary stats
    n_alert = len(matched)
    n_hit = int(matched["matched_event_onset"].notna().sum()) if n_alert else 0
    n_fp = int(matched["is_fp"].astype(bool).sum()) if n_alert else 0
    n_dup = int(matched["is_duplicate"].astype(bool).sum()) if n_alert else 0
    n_rev = int(labeled["label_switch_reversal"].fillna(False).astype(bool).sum())
    n_events = len(events)

    # Focus rows for SZ159502
    focus = matched[matched["code"].astype(str) == "SZ159502"].copy()
    focus_0703 = focus[focus["as_of"].astype(str) == "2026-07-03"]

    out_csv = out_dir / "event_metrics_reversal.csv"
    matched.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # Also write labeled signal overlay (compact)
    overlay_cols = [
        c
        for c in [
            "as_of",
            "code",
            "name",
            "quantile_switch",
            "switch_side",
            "pred_cdf",
            "return_z",
            "signal_level",
            "label_trend_up",
            "label_switch_reversal",
            "turn_kind",
            "jump_role",
            "prior_ret",
            "jump_ret",
            "post_ret",
            "final_confirmed_on",
            "early_label_switch_reversal",
            "early_turn_kind",
            "early_jump_role",
            "early_prior_ret",
            "early_jump_ret",
            "early_post_ret",
            "early_horizon",
            "early_confirmed_on",
            "early_to_final_status",
            "causal_prior_ret",
            "causal_jump_ret",
            "causal_thr_prior",
            "causal_prior_strong",
        ]
        if c in labeled.columns
    ]
    labeled[overlay_cols].to_csv(
        out_dir / "signals_with_reversal_labels.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "label": "switch_reversal",
        "lookback": int(args.lookback),
        "horizon": int(args.horizon),
        "early_horizon": int(args.early_horizon),
        "kappa_prior": float(args.kappa_prior),
        "kappa_post": float(args.kappa_post),
        "n_reversal_label_days": n_rev,
        "n_reversal_events": n_events,
        "n_quantile_switch_alerts": n_alert,
        "n_matched": n_hit,
        "n_fp": n_fp,
        "n_duplicate": n_dup,
        "precision_alert": (n_hit / n_alert) if n_alert else None,
        "early_confirmation": _early_confirmation_counts(labeled),
        "sz159502_20260703": focus_0703[
            [
                c
                for c in [
                    "as_of",
                    "quantile_switch",
                    "pred_cdf",
                    "return_z",
                    "label_switch_reversal",
                    "turn_kind",
                    "jump_role",
                    "prior_ret",
                    "jump_ret",
                    "post_ret",
                    "final_confirmed_on",
                    "early_label_switch_reversal",
                    "early_turn_kind",
                    "early_post_ret",
                    "early_confirmed_on",
                    "early_to_final_status",
                    "matched_event_onset",
                    "detection_delay",
                    "is_fp",
                ]
                if c in focus_0703.columns
            ]
        ].to_dict(orient="records"),
    }
    if live_meta is not None:
        summary["live_snapshot"] = live_meta
        summary["provisional_intraday"] = True
    (out_dir / "reversal_match_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    md = [
        "# Switch / Reversal Event Matching",
        "",
        f"- label: prior({args.lookback}) vs post({args.horizon}) sign flip, κ={args.kappa_prior}/{args.kappa_post}",
        f"- early confirmation: H={args.early_horizon} → final H={args.horizon}",
        f"- reversal label days (final): **{n_rev}**",
        f"- clustered reversal events: **{n_events}**",
        f"- quantile_switch alerts: **{n_alert}**",
        f"- matched: **{n_hit}** | FP: **{n_fp}** | duplicate: **{n_dup}**",
        f"- alert precision: **{(n_hit / n_alert):.1%}**" if n_alert else "- alert precision: n/a",
        "",
        "## Early → Final confirmation",
    ]
    early_stats = summary["early_confirmation"]
    md += [
        f"- dual-mature rows: **{early_stats.get('n_dual_mature')}**",
        f"- early positives: **{early_stats.get('n_early_positive')}** "
        f"| final positives: **{early_stats.get('n_final_positive')}**",
        f"- confirmed: **{early_stats.get('n_confirmed')}** "
        f"| revoked: **{early_stats.get('n_revoked')}** "
        f"| direction_changed: **{early_stats.get('n_direction_changed')}** "
        f"| final_only: **{early_stats.get('n_final_only')}**",
        f"- confirm_rate|early: **{early_stats.get('confirm_rate_given_early')}**",
        f"- revoke_rate|early: **{early_stats.get('revoke_rate_given_early')}**",
        f"- final_only_rate|final: **{early_stats.get('final_only_rate_given_final')}**",
        "",
        "## SZ159502 @ 2026-07-03",
    ]
    if len(focus_0703):
        r = focus_0703.iloc[0]
        md += [
            f"- quantile_switch={r.get('quantile_switch')} pred_cdf={r.get('pred_cdf')}",
            f"- turn_kind=**{r.get('turn_kind')}** jump_role=**{r.get('jump_role')}**",
            f"- prior_ret={r.get('prior_ret'):.4f} jump_ret={r.get('jump_ret'):.4f} post_ret={r.get('post_ret'):.4f}",
            f"- matched_event_onset={r.get('matched_event_onset')} delay={r.get('detection_delay')} is_fp=**{r.get('is_fp')}**",
            "",
            "Under trend-up matching this day was FP; under switch/reversal it is a "
            "**climax top** (prior↑, jump↑, post↓) and should not be counted as a false alarm "
            "for switch detection.",
        ]
    else:
        md.append("- row missing from alert set")
    (out_dir / "REVERSAL_MATCH.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(
        f"[OK] alerts={n_alert} matched={n_hit} fp={n_fp} "
        f"reversal_days={n_rev} -> {out_csv}"
    )
    if len(focus_0703):
        r = focus_0703.iloc[0]
        print(
            f"[SZ159502 2026-07-03] turn={r.get('turn_kind')} role={r.get('jump_role')} "
            f"is_fp={r.get('is_fp')} matched={r.get('matched_event_onset')}"
        )


if __name__ == "__main__":
    main()
