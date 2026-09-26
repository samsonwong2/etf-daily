"""Unit tests for HRP cluster router (no qlib)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from etf_daily.hrp.hrp_cluster_router import (
    MODE_DEFAULT,
    MODE_IMPULSE_SHADOW,
    MODE_RECOVERY_UP,
    MODE_SHALLOW_UP,
    code_mode_map,
    load_cluster_membership,
    mode_allows_trade_overlay,
    paint_cluster_modes,
    recovery_up_codes,
    router_snapshot,
    shallow_up_codes,
)

FIX = Path("~/etf-daily-output/temp/decision_packs/20260811/cluster_representatives_20260811.csv")


@pytest.mark.skipif(not FIX.exists(), reason="membership CSV missing")
def test_load_and_paint_20260811():
    mem = load_cluster_membership(FIX)
    assert len(mem) == 46
    cm = paint_cluster_modes(mem)
    assert cm[8] == MODE_RECOVERY_UP
    assert cm[17] == MODE_SHALLOW_UP
    assert cm[18] != MODE_RECOVERY_UP
    assert cm[12] == MODE_IMPULSE_SHADOW
    by_code = code_mode_map(mem, cm)
    assert by_code["SH515220"] == MODE_RECOVERY_UP
    assert by_code["SH513080"] == MODE_SHALLOW_UP
    assert by_code["SZ159561"] == MODE_SHALLOW_UP
    assert by_code["SH513650"] != MODE_RECOVERY_UP
    assert recovery_up_codes(by_code) == {"SH515220"}
    assert shallow_up_codes(by_code) == {"SH513080", "SZ159561"}
    assert not mode_allows_trade_overlay(MODE_RECOVERY_UP)
    assert mode_allows_trade_overlay(MODE_SHALLOW_UP)
    from etf_daily.hrp.hrp_cluster_router import mode_allows_diagnostic_marks

    assert mode_allows_diagnostic_marks(MODE_RECOVERY_UP)
    assert not mode_allows_diagnostic_marks(MODE_DEFAULT)
    snap = router_snapshot(mem, source_path=str(FIX))
    assert snap["mode_counts"][MODE_RECOVERY_UP] == 1
    assert snap["mode_counts"][MODE_SHALLOW_UP] == 2


def test_conflict_raises():
    mem = pd.DataFrame(
        {
            "code": ["SH515220", "SH513650"],
            "cluster": [1, 1],
            "name": ["a", "b"],
        }
    )
    with pytest.raises(ValueError, match="conflict"):
        paint_cluster_modes(
            mem,
            {"SH515220": MODE_RECOVERY_UP, "SH513650": MODE_IMPULSE_SHADOW},
        )
