"""Validate causal mean-model grids with walk-forward IC diagnostics.

This utility is intentionally lighter than a full portfolio replay.  It scores
EWMA span/shrink grids and regime-switch variants by daily cross-sectional IC,
then reports train/test walk-forward stability so only a few candidates need the
expensive pipeline replay.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from etf_daily.pool import build_rolling_mean_route_a_override as builder
    from etf_daily.pool import audit_causal_mean_models as audit
except ModuleNotFoundError:
    import build_rolling_mean_route_a_override as builder  # type: ignore[no-redef]
    import audit_causal_mean_models as audit  # type: ignore[no-redef]


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    mean_model: str
    ewma_span: int = 60
    shrink_alpha: float = 1.0
    shrink_target: str = "zero"
    regime_risk_off_shrink_alpha: float = 0.50


@dataclass(frozen=True)
class WalkForwardFold:
    name: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str


DEFAULT_FOLDS = (
    "fold_2024:2023-05-01:2024-04-30:2024-05-01:2025-04-30,"
    "fold_2025:2024-05-01:2025-04-30:2025-05-01:2026-04-30"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward IC audit for causal EWMA/grid/regime mean models.")
    parser.add_argument("--analysis-start-date", default="2024-05-01")
    parser.add_argument("--analysis-end-date", default="2026-04-30")
    parser.add_argument("--history-start-date", default="2023-05-01")
    parser.add_argument("--cluster-mapping", default=builder.DEFAULT_CLUSTER_MAPPING)
    parser.add_argument("--provider-uri", default=builder.DEFAULT_PROVIDER_URI)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ewma-spans", default="20,40,60,90,120")
    parser.add_argument("--shrink-alphas", default="0.5,0.75,1.0")
    parser.add_argument("--multi-windows", default="20,60,120,252")
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--min-periods", type=int, default=20)
    parser.add_argument("--asof-lag", type=int, default=1)
    parser.add_argument("--regime-benchmark", default="SH510300")
    parser.add_argument("--regime-fast-ma", type=int, default=20)
    parser.add_argument("--regime-slow-ma", type=int, default=60)
    parser.add_argument("--regime-vol-window", type=int, default=20)
    parser.add_argument("--regime-vol-lookback", type=int, default=120)
    parser.add_argument("--folds", default=DEFAULT_FOLDS)
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def alpha_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def parse_folds(value: str) -> list[WalkForwardFold]:
    folds: list[WalkForwardFold] = []
    for item in str(value).split(","):
        if not item.strip():
            continue
        name, train_start, train_end, test_start, test_end = [part.strip() for part in item.split(":")]
        folds.append(WalkForwardFold(name, train_start, train_end, test_start, test_end))
    return folds


def candidate_specs(args: argparse.Namespace) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for span in parse_int_list(args.ewma_spans):
        for alpha in parse_float_list(args.shrink_alphas):
            specs.append(
                CandidateSpec(
                    name=f"ewma{span}_shrink{alpha_tag(alpha)}",
                    mean_model="ewma",
                    ewma_span=span,
                    shrink_alpha=alpha,
                )
            )
    for alpha in parse_float_list(args.shrink_alphas):
        specs.append(
            CandidateSpec(
                name=f"multi_window_shrink{alpha_tag(alpha)}",
                mean_model="multi_window_mean",
                shrink_alpha=alpha,
            )
        )
    specs.extend(
        [
            CandidateSpec(
                name="regime_switch_ewma60_multi_window_shrink0p75",
                mean_model="regime_switch_ewma_multi_window",
                ewma_span=60,
                shrink_alpha=0.75,
            ),
            CandidateSpec(
                name="regime_switch_ewma60_riskoff_shrink0p50",
                mean_model="regime_switch_ewma_shrink",
                ewma_span=60,
                shrink_alpha=0.75,
                regime_risk_off_shrink_alpha=0.50,
            ),
        ]
    )
    return specs


def namespace_for(args: argparse.Namespace, spec: CandidateSpec) -> argparse.Namespace:
    return argparse.Namespace(
        mean_model=spec.mean_model,
        ewma_span=spec.ewma_span,
        multi_windows=args.multi_windows,
        multi_window_weights="",
        momentum_window=60,
        vol_window=60,
        ensemble_ewma_spans="20,60,120,252",
        ensemble_momentum_windows="20,60,120",
        ensemble_vol_adjusted_windows="60,120",
        ic_lookback=60,
        ic_min_periods=20,
        regime_benchmark=args.regime_benchmark,
        regime_fast_ma=args.regime_fast_ma,
        regime_slow_ma=args.regime_slow_ma,
        regime_vol_window=args.regime_vol_window,
        regime_vol_lookback=args.regime_vol_lookback,
        regime_risk_off_shrink_alpha=spec.regime_risk_off_shrink_alpha,
        window=args.window,
        min_periods=args.min_periods,
        asof_lag=args.asof_lag,
        mu_scale=1.0,
        shrink_target=spec.shrink_target,
        shrink_alpha=spec.shrink_alpha,
        winsor_quantile=None,
    )


def summarize_window(daily: pd.DataFrame, start_date: str, end_date: str, label: str) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    work = daily.copy()
    work["forecast_date"] = pd.to_datetime(work["forecast_date"])
    mask = (work["forecast_date"] >= pd.Timestamp(start_date)) & (work["forecast_date"] <= pd.Timestamp(end_date))
    window = work.loc[mask].copy()
    if window.empty:
        return pd.DataFrame()
    summary = audit.summarize_scores(window)
    summary.insert(0, "window", label)
    summary.insert(1, "start_date", start_date)
    summary.insert(2, "end_date", end_date)
    return summary


def spearman_t_stat(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 3 or clean.std(ddof=1) <= 0:
        return math.nan
    return float(clean.mean() / clean.std(ddof=1) * math.sqrt(len(clean)))


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output dir exists; pass --overwrite to replace: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_tickers = builder.static_builder.read_cluster_tickers(Path(args.cluster_mapping).expanduser().resolve())
    if args.max_tickers is not None:
        selected_tickers = selected_tickers[: int(args.max_tickers)]
    load_tickers = list(selected_tickers)
    benchmark = str(args.regime_benchmark).upper().strip()
    if benchmark and benchmark not in load_tickers:
        load_tickers.append(benchmark)

    close = builder.load_close_panel(
        load_tickers,
        provider_uri=args.provider_uri,
        start_date=args.history_start_date,
        end_date=args.analysis_end_date,
    )
    daily_returns = close.pct_change(fill_method=None)
    forward_returns = builder.forward_return_panel(close)
    forecast_mask = (close.index >= pd.Timestamp(args.analysis_start_date)) & (close.index <= pd.Timestamp(args.analysis_end_date))

    daily_frames: list[pd.DataFrame] = []
    manifest_specs: list[dict[str, object]] = []
    for spec in candidate_specs(args):
        model_args = namespace_for(args, spec)
        signal, _count, _source, metadata = builder.build_mean_model_panel(daily_returns, close, model_args)
        signal = signal.loc[forecast_mask, selected_tickers]
        scored = audit.daily_scores(spec.name, signal, forward_returns.loc[forecast_mask, selected_tickers])
        daily_frames.append(scored)
        manifest_specs.append({"name": spec.name, "config": spec.__dict__, "metadata": metadata})

    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    daily_path = output_dir / "causal_mean_grid_daily_ic.csv"
    daily.to_csv(daily_path, index=False)

    windows: list[pd.DataFrame] = []
    windows.append(summarize_window(daily, args.analysis_start_date, args.analysis_end_date, "full_analysis"))
    folds = parse_folds(args.folds)
    for fold in folds:
        windows.append(summarize_window(daily, fold.train_start, fold.train_end, f"{fold.name}_train"))
        windows.append(summarize_window(daily, fold.test_start, fold.test_end, f"{fold.name}_test"))
    window_summary = pd.concat([frame for frame in windows if frame is not None and not frame.empty], ignore_index=True)
    window_summary_path = output_dir / "causal_mean_grid_window_summary.csv"
    window_summary.to_csv(window_summary_path, index=False)

    fold_rows: list[dict[str, object]] = []
    for fold in folds:
        train = window_summary[window_summary["window"].eq(f"{fold.name}_train")].copy()
        test = window_summary[window_summary["window"].eq(f"{fold.name}_test")].copy()
        if train.empty or test.empty:
            continue
        train = train.sort_values(["avg_spearman_ic", "avg_top_decile_forward_spread"], ascending=[False, False])
        champion = str(train.iloc[0]["candidate"])
        test_row = test[test["candidate"].eq(champion)]
        fold_rows.append(
            {
                "fold": fold.name,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "selected_candidate": champion,
                "train_avg_spearman_ic": float(train.iloc[0]["avg_spearman_ic"]),
                "train_rank": 1,
                "test_avg_spearman_ic": float(test_row.iloc[0]["avg_spearman_ic"]) if not test_row.empty else math.nan,
                "test_positive_spearman_rate": float(test_row.iloc[0]["positive_spearman_rate"]) if not test_row.empty else math.nan,
                "test_rank": int(test["avg_spearman_ic"].rank(ascending=False, method="min")[test["candidate"].eq(champion)].iloc[0]) if not test_row.empty else math.nan,
            }
        )
    fold_selection = pd.DataFrame(fold_rows)
    fold_selection_path = output_dir / "walk_forward_selection_summary.csv"
    fold_selection.to_csv(fold_selection_path, index=False)

    stability_rows: list[dict[str, object]] = []
    for candidate, group in daily.groupby("candidate", dropna=False):
        group = group.copy()
        group["forecast_date"] = pd.to_datetime(group["forecast_date"])
        monthly = group.groupby(group["forecast_date"].dt.to_period("M"))["spearman_ic"].mean()
        stability_rows.append(
            {
                "candidate": candidate,
                "days": int(group["forecast_date"].nunique()),
                "avg_spearman_ic": float(pd.to_numeric(group["spearman_ic"], errors="coerce").mean()),
                "median_spearman_ic": float(pd.to_numeric(group["spearman_ic"], errors="coerce").median()),
                "naive_t_stat": spearman_t_stat(group["spearman_ic"]),
                "positive_day_rate": float((pd.to_numeric(group["spearman_ic"], errors="coerce") > 0.0).mean()),
                "positive_months": int((monthly > 0.0).sum()),
                "months": int(monthly.notna().sum()),
                "worst_month_ic": float(monthly.min()),
                "best_month_ic": float(monthly.max()),
            }
        )
    stability = pd.DataFrame(stability_rows).sort_values("avg_spearman_ic", ascending=False)
    stability_path = output_dir / "causal_mean_grid_stability.csv"
    stability.to_csv(stability_path, index=False)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "analysis_start_date": args.analysis_start_date,
                "analysis_end_date": args.analysis_end_date,
                "history_start_date": args.history_start_date,
                "cluster_mapping": str(Path(args.cluster_mapping).expanduser().resolve()),
                "provider_uri": args.provider_uri,
                "daily_path": str(daily_path),
                "window_summary_path": str(window_summary_path),
                "fold_selection_path": str(fold_selection_path),
                "stability_path": str(stability_path),
                "candidates": manifest_specs,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"Wrote daily IC: {daily_path}")
    print(f"Wrote window summary: {window_summary_path}")
    print(f"Wrote walk-forward selection: {fold_selection_path}")
    print(stability.head(20).to_string(index=False))
    if not fold_selection.empty:
        print(fold_selection.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())