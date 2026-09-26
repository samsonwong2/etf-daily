"""Discover pipeline artifacts for decision_pack (reuses position-report helpers)."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.lib.generate_daily_mu_position_report import (  # noqa: E402
    InputPaths,
    audit_outputs_ready,
    compact_to_iso,
    discover_bridge_l3_csv,
    discover_daily_nav_csv,
    ensure_audit_outputs,
    infer_preferred_start,
    normalize_report_date,
    resolve_input_paths,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_PATH = PACKAGE_ROOT / "state" / "recovery_state.json"


@dataclass(frozen=True)
class RunInputs:
    report_date: str
    compact_date: str
    run_dir: Path
    paths: InputPaths
    bridge_l3_csv: Path | None
    daily_nav_csv: Path | None
    preferred_start: str
    suffix: str


def _file_meta(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "size_bytes": stat.st_size,
    }


def resolve_run_inputs(
    run_dir: Path | str,
    report_date: str,
    *,
    preferred_start: str = "20260401",
    audit_dir_override: Path | None = None,
    ensure_audit: bool = True,
    audit_level: str = "full",
) -> RunInputs:
    base_dir = Path(run_dir).expanduser().resolve()
    iso_date, compact_date = normalize_report_date(report_date)
    preferred_start = infer_preferred_start(base_dir, compact_date, preferred_start)
    suffix = f"{preferred_start}_{compact_date}"
    paths = resolve_input_paths(
        base_dir,
        compact_date,
        preferred_start,
        audit_dir_override=audit_dir_override,
        ensure_audit=ensure_audit,
        audit_level=audit_level,
    )
    return RunInputs(
        report_date=iso_date,
        compact_date=compact_date,
        run_dir=base_dir,
        paths=paths,
        bridge_l3_csv=discover_bridge_l3_csv(base_dir, suffix),
        daily_nav_csv=discover_daily_nav_csv(base_dir, suffix),
        preferred_start=preferred_start,
        suffix=suffix,
    )


def build_inputs_manifest(
    inputs: RunInputs,
    *,
    live_holdings: Path | None = None,
    output_dir: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths = inputs.paths
    manifest: dict[str, Any] = {
        "as_of": inputs.report_date,
        "run_dir": str(inputs.run_dir),
        "suffix": inputs.suffix,
        "preferred_start": inputs.preferred_start,
        "preferred_start_iso": compact_to_iso(inputs.preferred_start),
        "files": {
            "audit_dir": _file_meta(paths.audit_dir),
            "layer_quality_csv": _file_meta(paths.layer_quality_csv),
            "portfolio_weights_csv": _file_meta(paths.portfolio_weights_csv),
            "bridge_mean_monitor_csv": _file_meta(paths.bridge_mean_monitor_csv),
            "with_name_csv": _file_meta(paths.with_name_csv),
            "rebalance_event_log_csv": _file_meta(paths.rebalance_event_log_csv),
            "bridge_l3_csv": _file_meta(inputs.bridge_l3_csv),
            "daily_nav_csv": _file_meta(inputs.daily_nav_csv),
            "live_holdings": _file_meta(live_holdings),
        },
    }
    if output_dir is not None:
        manifest["output_dir"] = str(output_dir.resolve())
    if extra:
        manifest.update(extra)
    return manifest


def write_inputs_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_STATE_PATH",
    "RunInputs",
    "audit_outputs_ready",
    "build_inputs_manifest",
    "ensure_audit_outputs",
    "resolve_run_inputs",
    "write_inputs_manifest",
]
