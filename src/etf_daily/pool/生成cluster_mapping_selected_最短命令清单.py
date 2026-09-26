#!/usr/bin/env python
"""Thin launcher: orchestrator logic lives in ``pool_builder/`` (see README.md).

Mirrors the top-level project style (see
``etf_strategy_clean/run.py``).
"""
import sys
from pathlib import Path

# Make the src tree importable when this launcher is run directly.
_SRC = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from etf_daily.pool.builder.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
