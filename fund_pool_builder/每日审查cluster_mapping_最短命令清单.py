#!/usr/bin/env python
"""Thin launcher: daily hand-curated cluster pool review (see README §6).

Mirrors ``生成cluster_mapping_selected_最短命令清单.py`` style.

After pipeline production, also run μ health tail from the project root::

    python run.py health \\
      --temp-dir <pipeline_temp_dir>

See ``workspace/history/daily_layer_quality_audit_runbook_20260517.md`` §5.1.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pool_builder.daily_review import main


if __name__ == "__main__":
    raise SystemExit(main())
