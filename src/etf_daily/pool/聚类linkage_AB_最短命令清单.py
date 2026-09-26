#!/usr/bin/env python
"""Thin launcher: A/B audit for cluster linkage choices."""
import sys
from pathlib import Path

_SRC = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from etf_daily.pool.builder.linkage_ab import main


if __name__ == "__main__":
    raise SystemExit(main())
