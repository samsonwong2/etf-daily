#!/usr/bin/env python3
"""T0 spike: SZ159647 3-state μ separation @ 20260620–20260703."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.lib.regime_transition_board import (
    RegimeTransitionConfig,
    compute_regime_transition_metrics,
)
from etf_daily.pipeline.price_adjustments import auto_qfq_adjust_close_panel
from etf_daily.pipeline.strategy_layer_audit import load_qlib_close
from etf_daily.paths import QLIB_PROVIDER_URI

CODE = "SZ159647"
DATES = ["2026-06-26", "2026-06-29", "2026-07-03", "2026-07-15"]


def main() -> int:
    panel = load_qlib_close(str(QLIB_PROVIDER_URI), [CODE], "2024-01-01", "2026-07-16")
    panel, _ = auto_qfq_adjust_close_panel(panel)
    s = panel[CODE].dropna()
    cfg = RegimeTransitionConfig(emission="student_t")
    print(f"=== {CODE} regime transition spike ===")
    for d in DATES:
        end = pd.Timestamp(d)
        sub = s.loc[:end]
        m = compute_regime_transition_metrics(sub, end, cfg=cfg)
        print(
            f"{d} regime={m.regime_now} prev={m.prev_regime} "
            f"score={m.transition_score:.3f} dir={m.transition_direction} "
            f"sep={m.regime_separated} ratio={m.min_pairwise_sigma_ratio} "
            f"p_trans={m.p_transition} "
            f"p_switch={m.p_switch_hmm} p_into_rev={m.p_into_reversal_hmm} "
            f"amb={m.label_ambiguous}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
