"""Rolling admission audit for cluster_mapping_selected instruments.

This is an output-side diagnostic: it reads daily parameter-effect audit files
and writes rolling contribution/stability summaries. It does not modify the
cluster mapping or trading pipeline.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_GROUP_FOCUS = ("other", "semiconductor", "communication")


@dataclass(frozen=True)
class AuditSpec:
    label: str
    path: Path


def _parse_audit_spec(raw: str) -> AuditSpec:
    if "=" in raw:
        label, path_text = raw.split("=", 1)
        label = label.strip()
    else:
        path_text = raw
        label = Path(raw).name
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"audit directory not found: {path}")
    return AuditSpec(label=label or path.name, path=path)


def _numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _load_instrument_daily(spec: AuditSpec) -> pd.DataFrame:
    path = spec.path / "param_inst_daily.csv"
    if not path.exists():
        path = spec.path / "per_instrument_daily_parameter_effects.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing per-instrument audit CSV: {path}")
    frame = pd.read_csv(path, parse_dates=["datetime"], low_memory=False)
    frame["run_label"] = spec.label
    if "cluster_group" not in frame.columns:
        frame["cluster_group"] = "unknown"
    if "cluster_mapping_rank" not in frame.columns:
        frame["cluster_mapping_rank"] = np.nan
    _numeric(
        frame,
        [
            "cluster_mapping_rank",
            "effective_weight",
            "pre_l3_weight",
            "realized_return_contribution",
            "next_return_used_for_attribution",
            "optimizer_mean_used_for_audit",
            "mean_direction_hit",
            "variance_floor_trigger_flag",
            "l3_weight_delta",
            "active_after_l3_flag",
            "active_before_l3_flag",
        ],
    )
    return frame


def _iter_windows(dates: list[pd.Timestamp], window: int, step: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not dates:
        return []
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for start_idx in range(0, len(dates), step):
        end_idx = start_idx + window - 1
        if end_idx >= len(dates):
            if start_idx == 0 or len(dates) - start_idx < max(5, window // 2):
                break
            end_idx = len(dates) - 1
        windows.append((dates[start_idx], dates[end_idx]))
        if end_idx == len(dates) - 1:
            break
    return windows


def _window_instrument_rows(frame: pd.DataFrame, window: int, step: int) -> pd.DataFrame:
    rows: list[dict] = []
    for run_label, run_frame in frame.groupby("run_label", sort=True):
        dates = sorted(pd.to_datetime(run_frame["datetime"].dropna().unique()))
        for start_dt, end_dt in _iter_windows(dates, window, step):
            mask = (run_frame["datetime"] >= start_dt) & (run_frame["datetime"] <= end_dt)
            window_frame = run_frame.loc[mask].copy()
            n_days = int(window_frame["datetime"].nunique())
            grouped = window_frame.groupby("instrument", dropna=False)
            for instrument, group in grouped:
                active = group["active_after_l3_flag"].fillna(0.0) > 0.0
                rows.append(
                    {
                        "run_label": run_label,
                        "window_start": start_dt.date().isoformat(),
                        "window_end": end_dt.date().isoformat(),
                        "n_days": n_days,
                        "instrument": instrument,
                        "cluster_group": str(group["cluster_group"].dropna().iloc[-1]) if group["cluster_group"].notna().any() else "unknown",
                        "cluster_mapping_rank": group["cluster_mapping_rank"].dropna().iloc[-1] if group["cluster_mapping_rank"].notna().any() else np.nan,
                        "active_days": int(active.sum()),
                        "active_rate": float(active.mean()) if len(active) else np.nan,
                        "avg_effective_weight": float(group["effective_weight"].mean()),
                        "sum_realized_return_contribution": float(group["realized_return_contribution"].sum()),
                        "avg_next_return_active": float(group.loc[active, "next_return_used_for_attribution"].mean()) if active.any() else np.nan,
                        "mean_direction_hit_rate": float(group["mean_direction_hit"].mean()),
                        "avg_optimizer_mean_used": float(group["optimizer_mean_used_for_audit"].mean()),
                        "variance_floor_trigger_rate": float(group["variance_floor_trigger_flag"].mean()),
                        "l3_reduction_days": int((group["l3_weight_delta"].fillna(0.0) < -1e-12).sum()),
                    }
                )
    return pd.DataFrame(rows)


def _window_group_rows(instrument_windows: pd.DataFrame) -> pd.DataFrame:
    if instrument_windows.empty:
        return pd.DataFrame()
    grouped = instrument_windows.groupby(
        ["run_label", "window_start", "window_end", "cluster_group"], dropna=False
    )
    out = grouped.agg(
        n_instruments=("instrument", "nunique"),
        active_days=("active_days", "sum"),
        avg_active_rate=("active_rate", "mean"),
        avg_effective_weight=("avg_effective_weight", "mean"),
        sum_realized_return_contribution=("sum_realized_return_contribution", "sum"),
        mean_direction_hit_rate=("mean_direction_hit_rate", "mean"),
        variance_floor_trigger_rate=("variance_floor_trigger_rate", "mean"),
        l3_reduction_days=("l3_reduction_days", "sum"),
    )
    return out.reset_index()


def _stability_summary(instrument_windows: pd.DataFrame) -> pd.DataFrame:
    if instrument_windows.empty:
        return pd.DataFrame()
    frame = instrument_windows.copy()
    frame["negative_window_flag"] = frame["sum_realized_return_contribution"] < 0.0
    frame["positive_window_flag"] = frame["sum_realized_return_contribution"] > 0.0
    grouped = frame.groupby(["run_label", "instrument"], dropna=False)
    out = grouped.agg(
        cluster_group=("cluster_group", "last"),
        cluster_mapping_rank=("cluster_mapping_rank", "last"),
        windows=("window_start", "count"),
        negative_windows=("negative_window_flag", "sum"),
        positive_windows=("positive_window_flag", "sum"),
        active_days_total=("active_days", "sum"),
        total_realized_return_contribution=("sum_realized_return_contribution", "sum"),
        worst_window_contribution=("sum_realized_return_contribution", "min"),
        best_window_contribution=("sum_realized_return_contribution", "max"),
        avg_active_rate=("active_rate", "mean"),
        avg_effective_weight=("avg_effective_weight", "mean"),
        avg_direction_hit_rate=("mean_direction_hit_rate", "mean"),
        avg_variance_floor_trigger_rate=("variance_floor_trigger_rate", "mean"),
        l3_reduction_days=("l3_reduction_days", "sum"),
    ).reset_index()
    out["negative_window_rate"] = out["negative_windows"] / out["windows"].replace(0, np.nan)
    return out.sort_values(["run_label", "total_realized_return_contribution", "negative_window_rate"])


def _default_focus_codes(stability: pd.DataFrame, n: int = 6) -> list[str]:
    if stability.empty:
        return []
    totals = stability.groupby("instrument")["total_realized_return_contribution"].sum().sort_values()
    return totals.head(n).index.astype(str).tolist()


def _write_report(
    output_path: Path,
    stability: pd.DataFrame,
    group_windows: pd.DataFrame,
    focus_codes: list[str],
    group_focus: tuple[str, ...],
) -> None:
    lines: list[str] = []
    lines.append("# Cluster Admission Rolling Audit")
    lines.append("")
    lines.append("This is a read-only stability audit. It does not edit cluster_mapping_selected.txt.")
    lines.append("")
    lines.append("## Focus Codes")
    lines.append("")
    lines.append(", ".join(focus_codes) if focus_codes else "None")
    lines.append("")
    if not stability.empty:
        focus = stability[stability["instrument"].isin(focus_codes)].copy()
        if not focus.empty:
            lines.append("## Focus Code Stability")
            lines.append("")
            lines.append(focus.to_markdown(index=False))
            lines.append("")
        lines.append("## Worst Total Contribution")
        lines.append("")
        lines.append(stability.nsmallest(20, "total_realized_return_contribution").to_markdown(index=False))
        lines.append("")
    if not group_windows.empty:
        group_summary = group_windows.groupby(["run_label", "cluster_group"], dropna=False).agg(
            windows=("window_start", "count"),
            total_contribution=("sum_realized_return_contribution", "sum"),
            avg_window_contribution=("sum_realized_return_contribution", "mean"),
            negative_windows=("sum_realized_return_contribution", lambda x: int((x < 0.0).sum())),
            avg_active_rate=("avg_active_rate", "mean"),
        ).reset_index()
        group_summary["negative_window_rate"] = group_summary["negative_windows"] / group_summary["windows"].replace(0, np.nan)
        focus_groups = group_summary[group_summary["cluster_group"].isin(group_focus)]
        lines.append("## Focus Group Stability")
        lines.append("")
        lines.append(focus_groups.to_markdown(index=False) if not focus_groups.empty else "No focus group rows.")
        lines.append("")
        lines.append("## All Group Stability")
        lines.append("")
        lines.append(group_summary.sort_values(["run_label", "total_contribution"]).to_markdown(index=False))
        lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(
    audit_specs: list[AuditSpec],
    output_dir: Path,
    rolling_window: int = 20,
    rolling_step: int = 5,
    focus_codes: list[str] | None = None,
    group_focus: tuple[str, ...] = DEFAULT_GROUP_FOCUS,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = [_load_instrument_daily(spec) for spec in audit_specs]
    combined = pd.concat(frames, ignore_index=True)
    instrument_windows = _window_instrument_rows(combined, rolling_window, rolling_step)
    group_windows = _window_group_rows(instrument_windows)
    stability = _stability_summary(instrument_windows)
    if focus_codes is None:
        focus_codes = _default_focus_codes(stability)
    paths = {
        "instrument_windows": output_dir / "cluster_admission_instrument_rolling.csv",
        "group_windows": output_dir / "cluster_admission_group_rolling.csv",
        "stability": output_dir / "cluster_admission_instrument_stability.csv",
        "report": output_dir / "cluster_admission_rolling_report.md",
    }
    instrument_windows.to_csv(paths["instrument_windows"], index=False)
    group_windows.to_csv(paths["group_windows"], index=False)
    stability.to_csv(paths["stability"], index=False)
    _write_report(paths["report"], stability, group_windows, focus_codes, group_focus)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only rolling admission audit for cluster mapping candidates")
    parser.add_argument("--audit-dir", action="append", required=True, help="Audit dir, optionally label=/path/to/audit_dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--rolling-step", type=int, default=5)
    parser.add_argument("--focus-codes", default="", help="Optional comma-separated focus codes; default = worst 6 by total contribution")
    parser.add_argument("--group-focus", default=",".join(DEFAULT_GROUP_FOCUS))
    args = parser.parse_args()

    specs = [_parse_audit_spec(raw) for raw in args.audit_dir]
    focus_codes = [code.strip().upper() for code in args.focus_codes.split(",") if code.strip()] or None
    group_focus = tuple(group.strip() for group in args.group_focus.split(",") if group.strip())
    paths = run_audit(
        specs,
        Path(args.output_dir).expanduser().resolve(),
        rolling_window=args.rolling_window,
        rolling_step=args.rolling_step,
        focus_codes=focus_codes,
        group_focus=group_focus,
    )
    print(f"[INFO] Cluster admission rolling report: {paths['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
