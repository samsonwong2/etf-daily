#!/usr/bin/env python3
"""Assess HMM predictive distribution and regime/switch accuracy.

Example:
  python src/etf_daily/scripts/assess_regime_transition_validation.py \\
    --validation-dir runtime/decision_packs/20260720/regime_transition_validation_q90_to0720 \\
    --eval-end 2026-07-20 \\
    --horizon 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.lib.buyable_rank_backtest import load_qlib_open_close
from etf_daily.lib.regime_transition_assessment import (
    load_signals_oos,
    run_assessment,
    write_assessment_outputs,
)
from etf_daily.lib.regime_transition_trade_bias import (
    TradeBiasConfig,
    load_trade_bias_predictions,
    run_trade_bias_walkforward,
    write_trade_bias_outputs,
)
from etf_daily.lib.regime_transition_bias_promotion import (
    DEFAULT_LOCKBOX_START,
    run_bias_promotion_pipeline,
)
from etf_daily.lib.regime_transition_validation import load_cluster_instruments
from etf_daily.pipeline.price_adjustments import auto_qfq_adjust_close_panel
from etf_daily.pipeline.strategy_layer_audit import load_qlib_close
from etf_daily.paths import CLUSTER_MAPPING_SELECTED_TXT, QLIB_PROVIDER_URI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Assess HMM distribution calibration and regime/switch accuracy"
    )
    p.add_argument(
        "--validation-dir",
        type=Path,
        required=True,
        help="directory containing signals_oos.csv",
    )
    p.add_argument("--eval-end", type=str, default=None, help="optional upper as_of bound")
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--provider-uri", type=Path, default=None)
    p.add_argument(
        "--cluster-mapping",
        type=Path,
        default=None,
        help="instruments list (default: CLUSTER_MAPPING_SELECTED_TXT)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: <validation-dir>/assessment",
    )
    p.add_argument(
        "--skip-trade-bias",
        action="store_true",
        help="skip T+1 open / T+5 close walk-forward trade-bias study",
    )
    p.add_argument(
        "--trade-hold-days",
        type=int,
        default=5,
        help="primary open→close hold days for trade-bias labels (default 5)",
    )
    p.add_argument(
        "--lockbox-start",
        type=str,
        default=DEFAULT_LOCKBOX_START,
        help="forward lockbox start date for bias promotion (default 2026-07-21)",
    )
    p.add_argument(
        "--skip-bias-promotion",
        action="store_true",
        help="skip T+5 candidate race / bias_promotion.json",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validation_dir = Path(args.validation_dir)
    if not validation_dir.is_absolute():
        validation_dir = _PROJECT_ROOT / validation_dir
    if not (validation_dir / "signals_oos.csv").exists():
        print(f"[ERROR] missing signals_oos.csv under {validation_dir}", file=sys.stderr)
        return 2

    signals = load_signals_oos(validation_dir)
    codes = sorted(signals["code"].astype(str).unique().tolist())
    if not codes:
        # fallback to cluster mapping
        cluster_path = Path(args.cluster_mapping or CLUSTER_MAPPING_SELECTED_TXT)
        instruments = load_cluster_instruments(cluster_path)
        codes = [i.code for i in instruments]

    start = str(pd.Timestamp(signals["as_of"].min()).date())
    end = args.eval_end or str(pd.Timestamp(signals["as_of"].max()).date())
    # need history for baseline lookback + outcome horizon
    load_start = (pd.Timestamp(start) - pd.tseries.offsets.BDay(120)).strftime("%Y-%m-%d")
    load_end = (pd.Timestamp(end) + pd.tseries.offsets.BDay(args.horizon + 5)).strftime(
        "%Y-%m-%d"
    )
    uri = str(args.provider_uri or QLIB_PROVIDER_URI)
    print(f"[INFO] loading Qlib closes codes={len(codes)} {load_start}→{load_end}")
    panel = load_qlib_close(uri, codes, load_start, load_end)
    panel, _ = auto_qfq_adjust_close_panel(panel)

    result = run_assessment(
        validation_dir,
        panel,
        horizon=int(args.horizon),
        eval_end=args.eval_end,
    )
    out_dir = Path(args.output_dir) if args.output_dir else validation_dir / "assessment"
    if not out_dir.is_absolute():
        out_dir = _PROJECT_ROOT / out_dir
    write_assessment_outputs(result, out_dir)

    summary = result["summary"]
    verdict = result["verdict"]
    print(f"[INFO] wrote assessment → {out_dir}")
    print(
        f"[INFO] verdict={verdict.status} "
        f"signal_end={summary.get('signal_end')} "
        f"label_mature_end={summary.get('label_mature_end')}"
    )
    for reason in verdict.reasons:
        print(f"  - {reason}")
    study = summary.get("threshold_study") or {}
    rec = study.get("recommendation") or {}
    if rec:
        print(
            f"[INFO] q* recommendation: keep_q90={rec.get('keep_production_q90')} "
            f"mode={rec.get('mode')} "
            f"holdout_f1_base={rec.get('holdout_f1_baseline')} "
            f"holdout_f1_rec={rec.get('holdout_f1_recommended')}"
        )
        if rec.get("note"):
            print(f"  - {rec['note']}")
    causal = summary.get("causal_bias") or {}
    if causal:
        print(
            f"[INFO] causal bias holdout alerts={causal.get('n_alerts_holdout')} "
            f"months={causal.get('holdout_months')}"
        )
        for row in causal.get("holdout_forward") or []:
            if row.get("side") not in ("buy_bias", "sell_bias"):
                continue
            print(
                f"  - {row.get('side')} h={row.get('horizon')}: "
                f"hit_rate={row.get('hit_rate')} "
                f"CI=[{row.get('hit_ci_low')}, {row.get('hit_ci_high')}] "
                f"n={row.get('n_scored')}"
            )
        for row in causal.get("holdout_reversal") or []:
            if row.get("side") not in ("buy_bias", "sell_bias", "both"):
                continue
            print(
                f"  - reversal {row.get('side')}: "
                f"f1={row.get('f1')} precision={row.get('precision')} "
                f"recall={row.get('recall')} same_day={row.get('same_day_confirm_rate')}"
            )
        naming = causal.get("naming_recommendation") or {}
        for side in ("buy_bias", "sell_bias"):
            info = naming.get(side) or {}
            print(
                f"  - naming {side}: keep={info.get('keep_bias_wording')} "
                f"— {info.get('note')}"
            )
    early = summary.get("early_confirmation") or {}
    for row in early.get("windows") or []:
        if row.get("window") != "holdout":
            continue
        print(
            f"[INFO] early confirmation holdout: "
            f"confirm_rate|early={row.get('confirm_rate_given_early')} "
            f"revoke_rate|early={row.get('revoke_rate_given_early')} "
            f"final_only|final={row.get('final_only_rate_given_final')} "
            f"n_early={row.get('n_early_positive')} n_final={row.get('n_final_positive')}"
        )
    pba = summary.get("prior_bias_alignment") or {}
    prec = pba.get("recommendation") or {}
    if prec:
        print(
            f"[INFO] prior-bias alignment: update_plot={prec.get('update_plot_labels')} "
            f"rule={prec.get('rule')} baseline_hit={prec.get('baseline_same_day_hit_rate')}"
        )
        if prec.get("note"):
            print(f"  - {prec['note']}")
        for row in pba.get("holdout_same_day") or []:
            if row.get("side") != "both":
                continue
            print(
                f"  - {row.get('rule')}: hit={row.get('same_day_hit_rate')} "
                f"CI=[{row.get('hit_ci_low')}, {row.get('hit_ci_high')}] "
                f"n_bias={row.get('n_with_bias')} coverage={row.get('coverage')}"
            )

    open_panel = None
    close_panel = None
    if not args.skip_trade_bias:
        trade_load_end = (
            pd.Timestamp(end) + pd.tseries.offsets.BDay(int(args.trade_hold_days) + 5)
        ).strftime("%Y-%m-%d")
        print(
            f"[INFO] loading Qlib open/close for trade-bias "
            f"codes={len(codes)} {load_start}→{trade_load_end}"
        )
        open_panel, close_panel = load_qlib_open_close(
            uri, codes, load_start, trade_load_end
        )
        trade_signals = signals.copy()
        if args.eval_end is not None:
            trade_signals = trade_signals[
                trade_signals["as_of"] <= pd.Timestamp(args.eval_end).normalize()
            ].copy()
        trade_result = run_trade_bias_walkforward(
            trade_signals,
            open_panel,
            close_panel,
            cfg=TradeBiasConfig(hold_days=int(args.trade_hold_days)),
        )
        write_trade_bias_outputs(trade_result, out_dir)
        tb_summary = (trade_result.get("metrics") or {}).get("summary") or {}
        print(
            f"[INFO] trade-bias hold_days={args.trade_hold_days} "
            f"n_pred={tb_summary.get('n_predictions')} "
            f"pass_gate={tb_summary.get('pass_gate')} "
            f"update_plot={tb_summary.get('update_plot_labels')}"
        )
        for side in ("buy_holdout", "sell_holdout"):
            info = tb_summary.get(side) or {}
            if not info:
                continue
            print(
                f"  - {side}: n_actions={info.get('n_actions')} "
                f"hit={info.get('hit_rate')} "
                f"mean_net={info.get('mean_signed_net')} "
                f"CI=[{info.get('ci_low')}, {info.get('ci_high')}] "
                f"edge={info.get('supports_positive_edge')}"
            )
        if tb_summary.get("note"):
            print(f"  - {tb_summary['note']}")

    if not args.skip_bias_promotion:
        if open_panel is None or close_panel is None:
            trade_load_end = (
                pd.Timestamp(end) + pd.tseries.offsets.BDay(int(args.trade_hold_days) + 5)
            ).strftime("%Y-%m-%d")
            print(
                f"[INFO] loading Qlib open/close for bias-promotion "
                f"codes={len(codes)} {load_start}→{trade_load_end}"
            )
            open_panel, close_panel = load_qlib_open_close(
                uri, codes, load_start, trade_load_end
            )
        holdout_months = (
            ((summary.get("threshold_study") or {}).get("splits") or {}).get(
                "holdout_months"
            )
            or []
        )
        if not holdout_months:
            # Fallback: last 25% of months among switch as_ofs.
            months = sorted(
                pd.to_datetime(signals["as_of"]).dt.to_period("M").astype(str).unique()
            )
            n_hold = max(2, int(round(len(months) * 0.25))) if months else 0
            holdout_months = months[-n_hold:] if n_hold else []
        tb_pred = load_trade_bias_predictions(out_dir / "trade_bias_predictions.csv")
        prior_path = validation_dir / "signals_with_reversal_labels.csv"
        prior_frame = None
        if prior_path.exists():
            prior_frame = pd.read_csv(prior_path)
        promo_signals = signals.copy()
        if args.eval_end is not None:
            promo_signals = promo_signals[
                promo_signals["as_of"] <= pd.Timestamp(args.eval_end).normalize()
            ].copy()
        promo = run_bias_promotion_pipeline(
            promo_signals,
            open_panel,
            close_panel,
            validation_dir=validation_dir,
            output_dir=out_dir,
            holdout_months=[str(m) for m in holdout_months],
            lockbox_start=str(args.lockbox_start),
            trade_bias_predictions=tb_pred,
            prior_frame=prior_frame,
            hold_days=int(args.trade_hold_days),
        )
        promotion = promo["promotion"]
        print(
            f"[INFO] bias-promotion lockbox_start={args.lockbox_start} "
            f"source={promotion.get('promotion_source')} "
            f"eligible={((promotion.get('lockbox') or {}).get('eligible'))} "
            f"buy_rule={promotion.get('buy_rule')} sell_rule={promotion.get('sell_rule')}"
        )
        race = promo["race"]
        for _, row in race[race["window"] == "holdout"].iterrows():
            print(
                f"  - holdout {row['rule']} {row['side']}: "
                f"n={row['n_scored']} hit={row['hit_rate']} "
                f"CI=[{row['hit_ci_low']}, {row['hit_ci_high']}] pass={row['pass']}"
            )

    return 0 if verdict.status != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
