from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .shadow_alert import DEFAULT_FUND_LIST_CSV, build_shadow_alert_history

DEFAULT_CANDIDATES = {
    "2025Q3_expected_l3": "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/bridge_posterior_mu1d_softshrink_fullpanel_20250701_20250930_l3.csv",
    "2025Q4_expected_l3": "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/bridge_posterior_mu1d_softshrink_fullpanel_20251001_20251231_l3.csv",
    "2025Q3Q4_weight_only_observed": "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/w_20250701_20251104.csv",
}


def _parse_l3_csv_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path_text = value.split("=", 1)
        return label.strip(), Path(path_text).expanduser()
    path = Path(value).expanduser()
    return path.stem, path


def _compound_return(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if numeric.empty:
        return float("nan")
    return float((1.0 + numeric).prod() - 1.0)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in frame.columns) + " |")
    return "\n".join(lines)


def _summarize_history(label: str, path: Path, history: pd.DataFrame) -> dict[str, object]:
    if history.empty:
        return {"label": label, "path": str(path), "status": "empty_history"}
    days = int(history.shape[0])
    entropy_days = int(history["entropy_defense_trigger"].sum())
    entropy_threshold_days = int(history["entropy_threshold_trigger"].sum())
    fatigue_days = int(history["momentum_fatigue_warning"].sum())
    both_days = int((history["entropy_defense_trigger"] & history["momentum_fatigue_warning"]).sum())
    mainline_turnover = pd.to_numeric(history.get("mainline_effective_turnover_proxy", pd.Series(dtype=float)), errors="coerce")
    shadow_turnover = pd.to_numeric(history.get("entropy_shadow_effective_turnover_proxy", pd.Series(dtype=float)), errors="coerce")
    return {
        "label": label,
        "path": str(path),
        "status": "ok",
        "days": days,
        "start_date": str(history["datetime"].iloc[0]),
        "end_date": str(history["datetime"].iloc[-1]),
        "entropy_threshold_trigger_days": entropy_threshold_days,
        "entropy_threshold_trigger_rate": entropy_threshold_days / days if days else np.nan,
        "entropy_defense_trigger_days": entropy_days,
        "entropy_defense_trigger_rate": entropy_days / days if days else np.nan,
        "momentum_fatigue_warning_days": fatigue_days,
        "momentum_fatigue_warning_rate": fatigue_days / days if days else np.nan,
        "both_trigger_days": both_days,
        "mainline_cumulative_return_proxy": _compound_return(history["mainline_effective_return"]),
        "entropy_shadow_cumulative_return_proxy": _compound_return(history["entropy_shadow_effective_return"]),
        "entropy_shadow_delta_return_sum": float(pd.to_numeric(history["entropy_shadow_delta_return"], errors="coerce").fillna(0.0).sum()),
        "entropy_shadow_delta_return_worst_day": float(pd.to_numeric(history["entropy_shadow_delta_return"], errors="coerce").min()),
        "mainline_turnover_mean": float(mainline_turnover.dropna().mean()) if not mainline_turnover.dropna().empty else np.nan,
        "mainline_turnover_max": float(mainline_turnover.dropna().max()) if not mainline_turnover.dropna().empty else np.nan,
        "entropy_shadow_turnover_mean": float(shadow_turnover.dropna().mean()) if not shadow_turnover.dropna().empty else np.nan,
        "entropy_shadow_turnover_max": float(shadow_turnover.dropna().max()) if not shadow_turnover.dropna().empty else np.nan,
        "overtrigger_flag": bool(entropy_days / days > 0.20) if days else False,
        "turnover_stress_flag": bool(shadow_turnover.dropna().mean() > mainline_turnover.dropna().mean() * 1.25) if not shadow_turnover.dropna().empty and not mainline_turnover.dropna().empty else False,
    }


def run_stress_test(candidates: list[tuple[str, Path]], output_dir: Path, fund_list_csv: str | Path | None = DEFAULT_FUND_LIST_CSV) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    for label, path in candidates:
        if not path.exists():
            row = {"label": label, "path": str(path), "status": "missing_file"}
            rows.append(row)
            missing_rows.append(row)
            continue
        try:
            history, _ = build_shadow_alert_history(path, fund_list_csv=fund_list_csv)
            history.to_csv(output_dir / f"{label}_shadow_alert_history.csv", index=False)
            rows.append(_summarize_history(label, path, history))
        except Exception as exc:
            row = {"label": label, "path": str(path), "status": "unusable_input", "error": str(exc)}
            rows.append(row)
            missing_rows.append(row)
    summary = pd.DataFrame(rows)
    missing = pd.DataFrame(missing_rows)
    summary.to_csv(output_dir / "cross_regime_shadow_stress_summary.csv", index=False)
    missing.to_csv(output_dir / "cross_regime_shadow_stress_missing_inputs.csv", index=False)
    report_lines = [
        "# Cross-Regime Shadow Stress Test",
        "",
        "This audit reuses the daily shadow alert logic to evaluate short-window SQ plus industry entropy across older regimes.",
        "",
        "## Summary",
        "",
        _markdown_table(summary),
        "",
        "## Missing Or Unusable Inputs",
        "",
        _markdown_table(missing) if not missing.empty else "None.",
        "",
        "## Interpretation",
        "",
        "A regime is not promotable if entropy defense triggers too frequently, shadow turnover materially exceeds mainline turnover, or cumulative ex-post delta is materially negative.",
    ]
    (output_dir / "cross_regime_shadow_stress_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return summary, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-regime stress test for entropy defense and momentum fatigue shadow alerts.")
    parser.add_argument(
        "--l3-csv",
        action="append",
        default=None,
        help="Candidate input as label=/path/to/bridge_l3.csv. May be repeated. Defaults to expected 2025 Q3/Q4 paths.",
    )
    parser.add_argument(
        "--output-dir",
        default="archive/audits/20260504_cross_regime_stress",
        help="Directory for stress-test summary outputs.",
    )
    parser.add_argument(
        "--fund-list-csv",
        default=DEFAULT_FUND_LIST_CSV,
        help="fund_list CSV used to refine industry entropy labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.l3_csv:
        candidates = [_parse_l3_csv_arg(value) for value in args.l3_csv]
    else:
        candidates = [(label, Path(path)) for label, path in DEFAULT_CANDIDATES.items()]
    summary, missing = run_stress_test(candidates, Path(args.output_dir), fund_list_csv=args.fund_list_csv)
    print(f"[INFO] wrote {Path(args.output_dir) / 'cross_regime_shadow_stress_summary.csv'}")
    if not missing.empty:
        print(f"[WARN] missing/unusable inputs: {len(missing)}")


if __name__ == "__main__":
    main()
