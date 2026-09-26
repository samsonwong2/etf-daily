"""Runtime path defaults for the portable ETF strategy project.

Machine-specific paths live in ``configs/production_regime_switch_ewma_shrink.json``
under the top-level ``paths`` object. Defaults are still derived from the copied
project location if that file is missing or a key is omitted.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "production_regime_switch_ewma_shrink.json"


def _default_paths() -> dict[str, str]:
    home = Path.home()
    provider_uri = home / ".qlib" / "qlib_data" / "all_fund_data"
    temp_dir = home / "etf-daily-output"
    workspace_dir = PROJECT_ROOT / "workspace"
    log_dir = workspace_dir / "logs"
    return {
        "provider_uri": str(provider_uri),
        "qlib_scripts_dir": "",
        "temp_dir": str(temp_dir),
        "workspace_dir": str(workspace_dir),
        "log_dir": str(log_dir),
        "plotly_outputs_dir": str(temp_dir / "plotly_outputs"),
        "local_library_dir": "",
        "fund_list_csv": str(temp_dir / "fund_list.csv"),
        "qlib_change_csv_dir": str(provider_uri / "change_csv"),
        "all_instruments_txt": str(provider_uri / "instruments" / "all.txt"),
        "cluster_mapping_path": str(provider_uri / "instruments" / "cluster_mapping_selected.txt"),
    }


def _optional_path(value: object) -> Path | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def _load_config_paths() -> dict[str, Any]:
    if not DEFAULT_CONFIG_PATH.exists():
        return {}
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    paths = payload.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError(f"Config paths must be a JSON object: {DEFAULT_CONFIG_PATH}")
    return paths


def _expand_paths(paths: dict[str, Any]) -> dict[str, str]:
    expanded: dict[str, str] = {
        **_default_paths(),
        **{str(key): str(value) for key, value in paths.items() if value is not None},
    }
    expanded["project_root"] = str(PROJECT_ROOT)
    expanded["workspace_root"] = str(WORKSPACE_ROOT)
    for _ in range(8):
        changed = False
        context = dict(expanded)
        for key, value in list(expanded.items()):
            formatted = value.format(**context)
            if formatted != value:
                expanded[key] = formatted
                changed = True
        if not changed:
            break
    return expanded


_PATHS = _expand_paths(_load_config_paths())

QLIB_PROVIDER_URI = Path(_PATHS["provider_uri"]).expanduser().resolve()
TEMP_DIR = Path(_PATHS["temp_dir"]).expanduser().resolve()
LOG_DIR = Path(_PATHS["log_dir"]).expanduser().resolve()
WORKSPACE_DIR = Path(_PATHS["workspace_dir"]).expanduser().resolve()
FUND_CRAWLER_LOG_DIR = LOG_DIR / "fund_crawler"
PIPELINE_RUN_LOG_DIR = LOG_DIR / "pipeline_runs"
HISTORY_DIR = WORKSPACE_DIR / "history"
DOCS_DIR = HISTORY_DIR
NOTES_DIR = WORKSPACE_DIR / "notes"
WORKSPACE_SCRIPTS_DIR = WORKSPACE_DIR / "scripts"
PLOTLY_OUTPUTS_DIR = Path(_PATHS["plotly_outputs_dir"]).expanduser().resolve()
FUND_LIST_CSV = Path(_PATHS["fund_list_csv"]).expanduser().resolve()
QLIB_CHANGE_CSV_DIR = Path(_PATHS["qlib_change_csv_dir"]).expanduser().resolve()
QLIB_INSTRUMENTS_DIR = QLIB_PROVIDER_URI / "instruments"
ALL_INSTRUMENTS_TXT = Path(_PATHS["all_instruments_txt"]).expanduser().resolve()
CLUSTER_MAPPING_SELECTED_TXT = Path(_PATHS["cluster_mapping_path"]).expanduser().resolve()
QLIB_SCRIPTS_DIR = _optional_path(_PATHS.get("qlib_scripts_dir"))
LOCAL_LIBRARY_DIR = _optional_path(_PATHS.get("local_library_dir"))


def as_str(path: str | Path) -> str:
    return str(Path(path).expanduser())
