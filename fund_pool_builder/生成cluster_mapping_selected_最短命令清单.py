#!/usr/bin/env python
"""Thin launcher: orchestrator logic lives in ``pool_builder/`` (see README.md).

Mirrors the top-level project style (see
``etf_strategy_clean/run.py``).
"""
import sys
from pathlib import Path

# Make ``pool_builder`` importable when this launcher is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pool_builder.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
