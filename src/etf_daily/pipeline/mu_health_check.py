"""Read-only tail view of mu_monitor + daily_layer_quality for a pipeline temp_dir."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

DEFAULT_MU_PRESET = "simple_pulse_mu"
DEFAULT_TAIL = 5
PRESET_SLUG_ALIASES = {
    "simple_pulse_mu": "spm",
    "trend_mu_default": "trend",
    "legacy_raw_mu": "rawmu",
    "path_trend_default": "path",
    "mu_smooth_trend_default": "smooth",
    "posterior_mu1d_default": "pmu1d",
    "posterior_mu20_default": "pmu20",
    "posterior_mu1d_softshrink_default": "pmu1d_ss",
    "posterior_mu1d_softshrink_signgate": "pmu1d_sg",
    "posterior_mu1d_softshrink_stoploss": "pmu1d_sl",
}

MONITOR_PREFERRED = [
    "date",
    "overall_status",
    "raw_model_status",
    "trend_protection_status",
    "bridge_status",
    "portfolio_confirm_status",
    "rank_ic_raw",
    "pearson_ic_raw",
    "top_bottom_spread_raw",
    "dir_acc_raw",
    "trigger_metrics",
]

LAYER_QUALITY_PREFERRED = [
    "datetime",
    "label_status",
    "mean_ic_spearman",
    "mean_ic_pearson",
    "mean_top20_bottom20_next_return_spread",
    "mvo_selected_vs_unselected_spread",
    "mean_direction_hit_rate",
    "adaptive_realized_return_contribution_sum",
]


def _compact(date: str) -> str:
    return str(date).replace("-", "")


def _suffix(start_date: str, end_date: str) -> str:
    return f"{_compact(start_date)}_{_compact(end_date)}"


def _preset_slug(preset: str) -> str:
    if preset in PRESET_SLUG_ALIASES:
        return PRESET_SLUG_ALIASES[preset]
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in preset) or "preset"


def _legacy_preset_slug(preset: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in preset) or "preset"


def _pick_columns(frame: pd.DataFrame, preferred: list[str]) -> list[str]:
    return [column for column in preferred if column in frame.columns]


def _load_manifest_dates(temp_dir: Path) -> tuple[str | None, str | None]:
    manifests = sorted([
        *temp_dir.glob("audit_*/run_manifest.json"),
        *temp_dir.glob("parameter_effect_audit_*/run_manifest.json"),
    ])
    if not manifests:
        return None, None
    try:
        payload = json.loads(manifests[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    start = payload.get("start_date") or payload.get("window_start")
    end = payload.get("end_date") or payload.get("window_end")
    return (str(start) if start else None, str(end) if end else None)


def _infer_dates_from_artifacts(temp_dir: Path, mu_preset: str) -> tuple[str | None, str | None]:
    slug = _preset_slug(mu_preset)
    legacy_slug = _legacy_preset_slug(mu_preset)
    patterns = (
        re.compile(rf"monitor_{re.escape(slug)}_(\d{{8}})_(\d{{8}})$"),
        re.compile(rf"mu_monitor_report_{re.escape(legacy_slug)}_fullpanel_(\d{{8}})_(\d{{8}})$"),
    )
    candidates: list[tuple[str, str]] = []
    for path in temp_dir.iterdir():
        if not path.is_dir():
            continue
        for pattern in patterns:
            match = pattern.match(path.name)
            if match:
                candidates.append((match.group(1), match.group(2)))
    if not candidates:
        return None, None
    start, end = sorted(candidates)[-1]

    def _iso(compact: str) -> str:
        return f"{compact[:4]}-{compact[4:6]}-{compact[6:8]}"

    return _iso(start), _iso(end)


def resolve_window(
    temp_dir: Path,
    mu_preset: str,
    start_date: str | None,
    end_date: str | None,
) -> tuple[str, str]:
    if start_date and end_date:
        return start_date, end_date
    manifest_start, manifest_end = _load_manifest_dates(temp_dir)
    if manifest_start and manifest_end:
        return manifest_start, manifest_end
    inferred_start, inferred_end = _infer_dates_from_artifacts(temp_dir, mu_preset)
    if inferred_start and inferred_end:
        return inferred_start, inferred_end
    raise ValueError(
        "Could not infer start/end dates. Pass --start-date and --end-date, "
        "or ensure temp_dir contains monitor_*/audit_* or legacy mu_monitor_report_*/parameter_effect_audit_* artifacts."
    )


def resolve_paths(
    temp_dir: Path,
    mu_preset: str,
    start_date: str,
    end_date: str,
) -> dict[str, Path]:
    slug = _preset_slug(mu_preset)
    legacy_slug = _legacy_preset_slug(mu_preset)
    suffix = _suffix(start_date, end_date)
    monitor_csv_candidates = [
        temp_dir / f"monitor_{slug}_{suffix}" / "raw_mu_alert_daily_summary.csv",
        temp_dir / f"mu_monitor_report_{legacy_slug}_fullpanel_{suffix}" / "raw_mu_alert_daily_summary.csv",
    ]
    layer_quality_candidates = [
        temp_dir / f"audit_{slug}_{suffix}" / "layer_daily.csv",
        temp_dir / f"parameter_effect_audit_{legacy_slug}_fullpanel_{suffix}" / "daily_layer_quality_summary.csv",
    ]
    monitor_csv = next((path for path in monitor_csv_candidates if path.exists()), monitor_csv_candidates[0])
    layer_quality_csv = next((path for path in layer_quality_candidates if path.exists()), layer_quality_candidates[0])
    return {
        "monitor_csv": monitor_csv,
        "layer_quality_csv": layer_quality_csv,
    }


def _print_section(title: str, frame: pd.DataFrame, preferred: list[str], tail: int) -> None:
    print(f"\n=== {title} (last {tail} rows) ===")
    if frame.empty:
        print("(empty)")
        return
    subset = frame.tail(tail)
    columns = _pick_columns(subset, preferred)
    if not columns:
        print(subset.to_string(index=False))
        return
    print(subset[columns].to_string(index=False))


def run_check(
    temp_dir: Path,
    *,
    mu_preset: str = DEFAULT_MU_PRESET,
    start_date: str | None = None,
    end_date: str | None = None,
    tail: int = DEFAULT_TAIL,
) -> int:
    temp_dir = temp_dir.expanduser().resolve()
    if not temp_dir.is_dir():
        print(f"[ERROR] temp_dir not found: {temp_dir}", file=sys.stderr)
        return 1

    try:
        start, end = resolve_window(temp_dir, mu_preset, start_date, end_date)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    paths = resolve_paths(temp_dir, mu_preset, start, end)
    print(f"[INFO] temp_dir={temp_dir}")
    print(f"[INFO] window={start} .. {end}  mu_preset={mu_preset}")

    exit_code = 0

    monitor_path = paths["monitor_csv"]
    if monitor_path.is_file():
        monitor = pd.read_csv(monitor_path)
        if "date" in monitor.columns:
            monitor = monitor.sort_values("date")
        _print_section(f"mu_monitor: {monitor_path.name}", monitor, MONITOR_PREFERRED, tail)
    else:
        print(f"\n[WARN] missing mu_monitor: {monitor_path}", file=sys.stderr)
        exit_code = 1

    layer_path = paths["layer_quality_csv"]
    if layer_path.is_file():
        layer = pd.read_csv(layer_path)
        if "datetime" in layer.columns:
            layer = layer.sort_values("datetime")
        _print_section(f"layer_quality: {layer_path.name}", layer, LAYER_QUALITY_PREFERRED, tail)
    else:
        print(
            f"\n[WARN] missing layer_quality (need --audit-level full): {layer_path}",
            file=sys.stderr,
        )
        exit_code = 1

    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print last N rows of mu_monitor + layer quality summary for a pipeline temp_dir."
    )
    parser.add_argument(
        "--temp-dir",
        required=True,
        help="Pipeline --temp-dir (e.g. run folder with monitor_* and audit_* outputs).",
    )
    parser.add_argument("--mu-preset", default=DEFAULT_MU_PRESET)
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD; optional if manifest/artifacts exist.")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD; optional if manifest/artifacts exist.")
    parser.add_argument("--tail", type=int, default=DEFAULT_TAIL, help="Rows to show per table (default 5).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_check(
        Path(args.temp_dir),
        mu_preset=str(args.mu_preset),
        start_date=args.start_date,
        end_date=args.end_date,
        tail=int(args.tail),
    )


if __name__ == "__main__":
    raise SystemExit(main())
