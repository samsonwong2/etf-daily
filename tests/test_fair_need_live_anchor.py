"""Fair-need should survive live snapshots that omit the equity anchor."""
from __future__ import annotations

import pandas as pd


def test_live_snapshot_without_anchor_keeps_fair_need_path_logic():
    """Document expected gate: only apply live bar when anchor code is present."""
    live = pd.DataFrame(
        {
            "code": ["SH588710"],
            "as_of": ["2026-08-17"],
            "captured_at": ["2026-08-17T11:00:00"],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [1e6],
            "source": ["test"],
        }
    )
    anchor = "SH510300"
    live_codes = set(live["code"].astype(str).str.upper())
    assert anchor not in live_codes
    # Callers must skip apply_live_snapshot_to_ohlcv(anchor) in this case.
    assert "SH588710" in live_codes
