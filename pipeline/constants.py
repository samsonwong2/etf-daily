"""Shared constants and mutable orchestrator flags.

``_DRY_RUN`` and ``_LOG_DIR`` are module-level globals set by ``cli.main()``
before any stage runs. Other modules MUST read them as
``constants._DRY_RUN`` / ``constants._LOG_DIR`` (fresh attribute lookup), NOT
via ``from .constants import _DRY_RUN`` (which would capture the initial
value at import time).
"""
from __future__ import annotations

from pathlib import Path

from runtime_paths import (
    CLUSTER_MAPPING_SELECTED_TXT,
    FUND_LIST_CSV,
    PROJECT_ROOT,
    QLIB_PROVIDER_URI,
    TEMP_DIR,
)

# Orchestrator repo root: parent of the ``pipeline/`` package.
REPO_ROOT = PROJECT_ROOT
DEFAULT_PROVIDER_URI = str(QLIB_PROVIDER_URI)
DEFAULT_CLUSTER_MAPPING = str(CLUSTER_MAPPING_SELECTED_TXT)
DEFAULT_TEMP_DIR = str(TEMP_DIR)
DEFAULT_FUND_LIST_CSV = str(FUND_LIST_CSV)
DEFAULT_MU_PRESET = "simple_pulse_mu"
DEFAULT_INIT_CASH = 1_000_000.0
PRODUCTION_FORECAST_MU_METHOD = "ewma60"
PRODUCTION_FORECAST_MU_EWMA_SPAN = 60
PRODUCTION_FORECAST_MU_SHRINK_TARGET = "zero"
PRODUCTION_FORECAST_MU_SHRINK_ALPHA = 0.75
PRODUCTION_FORECAST_MU_MIN_PERIODS = 20
PRODUCTION_FORECAST_MU_ASOF_LAG = 1
PRODUCTION_FORECAST_MU_MISSING_FILL = "zero"

# Module-level toggle set by cli.main() from --dry-run. When True, every
# _run_command() call prints its command and skips execution. All orchestrator
# Python code still runs (code list, raw panel, path resolution, risk controls),
# but all subprocess stages (Route A, bridge, backtest, monitor) are simulated.
_DRY_RUN: bool = False

# Directory where _run_command tees per-stage stdout/stderr. Set by cli.main()
# once the suffix + timestamp are known. None disables tee (stages write only
# to the orchestrator's stdout, matching legacy behavior).
_LOG_DIR: Path | None = None
