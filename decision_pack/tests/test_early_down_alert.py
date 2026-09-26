"""Unit tests for early-down diagnostic alert layer."""
from __future__ import annotations

import numpy as np
import pandas as pd

from decision_pack.src.early_down_alert import (
    build_early_down_alert_row,
    build_early_down_frame,
    classify_early_down_level,
    detect_early_down_marks,
    detect_early_down_runs,
    early_down_flags_frame,
    flags_from_values,
    render_early_down_markdown,
    write_early_down_artifacts,
)


def test_soft_bear_pre_stack_sh515060_pattern():
    """C below MA20 with weak priors, but MA20 still above MA60."""
    flags = flags_from_values(
        close=0.65,
        ma20=0.68,
        ma60=0.67,  # still bull stack (m20 > m60)
        prior20=-0.02,
        prior60=-0.10,
        dd20_high=-0.12,
    )
    assert flags.soft_bear is True
    assert flags.peak8 is True
    assert flags.classic_bear is False
    assert flags.early_down is True
    assert flags.bear_stack is False
    assert classify_early_down_level(flags, "range") == "alert"
    assert classify_early_down_level(flags, "up") == "alert"
    assert classify_early_down_level(flags, "down") == "watch"


def test_classic_bear_aligned_is_watch():
    flags = flags_from_values(
        close=0.60,
        ma20=0.65,
        ma60=0.68,
        prior20=-0.05,
        prior60=-0.08,
        dd20_high=-0.10,
    )
    assert flags.classic_bear is True
    assert flags.soft_bear is True
    assert classify_early_down_level(flags, "down") == "watch"


def test_no_signal_when_still_above_ma20():
    flags = flags_from_values(
        close=0.70,
        ma20=0.68,
        ma60=0.69,
        prior20=-0.05,
        prior60=-0.10,
        dd20_high=-0.10,
    )
    assert flags.early_down is False
    assert classify_early_down_level(flags, "range") == "none"


def test_build_row_from_feature_frame():
    n = 30
    dates = pd.bdate_range("2026-05-01", periods=n)
    # Grind down under MA20 while MA20 > MA60 on last bar.
    close = np.linspace(0.75, 0.62, n)
    px = pd.DataFrame(
        {
            "as_of": dates,
            "$close": close,
            "MA20": close + 0.03,
            "MA60": close + 0.01,  # MA20 > MA60
            "prior20": np.linspace(0.05, -0.04, n),
            "prior60": np.linspace(-0.02, -0.12, n),
        }
    )
    row = build_early_down_alert_row(
        code="SH515060",
        name="房地产ETF华夏",
        method="ma_stack_strict",
        as_of="2026-05-22",
        regime_asof="range",
        px=px,
    )
    assert row["soft_bear"] is True
    assert row["classic_bear"] is False
    assert row["level"] == "alert"
    assert "soft_bear" in row["reasons"]
    assert "diagnostic_only" in row["note"]


def test_detect_early_down_runs_and_marks():
    n = 12
    dates = pd.bdate_range("2026-05-13", periods=n)
    # First 6 bars: soft_bear while regime=range → alert run
    # Next 4: soft_bear while regime=down → watch run
    # Last 2: recovered above MA20 → none
    close = np.array([0.70, 0.69, 0.68, 0.67, 0.66, 0.65, 0.64, 0.63, 0.62, 0.61, 0.72, 0.73])
    ma20 = close + 0.03
    ma20[-2:] = close[-2:] - 0.01
    ma60 = close + 0.01  # no bear stack
    ma60[-2:] = close[-2:] - 0.02
    regimes = ["range"] * 6 + ["down"] * 4 + ["range"] * 2
    px = pd.DataFrame(
        {
            "as_of": dates,
            "$close": close,
            "MA20": ma20,
            "MA60": ma60,
            "prior20": [-0.02] * n,
            "prior60": [-0.10] * n,
            "regime": regimes,
        }
    )
    # Force dd20 for peak8 on early bars via synthetic high history in rolling window
    flags = early_down_flags_frame(px)
    assert bool(flags.iloc[0]["soft_bear"])
    assert flags.iloc[0]["level"] == "alert"
    assert flags.iloc[6]["level"] == "watch"
    runs = detect_early_down_runs(px)
    assert any(r["level"] == "alert" for r in runs)
    marks = detect_early_down_marks(px)
    kinds = {m["kind"] for m in marks}
    assert "ed_alert" in kinds
    assert "ed_watch" in kinds


def test_write_artifacts_and_markdown(tmp_path):
    rows = [
        build_early_down_alert_row(
            code="AAA",
            name="a",
            method="ma_stack_strict",
            as_of="2026-08-07",
            regime_asof="range",
            px=pd.DataFrame(
                {
                    "as_of": pd.to_datetime(["2026-08-07"]),
                    "$close": [1.0],
                    "MA20": [1.1],
                    "MA60": [1.05],
                    "prior20": [-0.01],
                    "prior60": [-0.08],
                }
            ),
        )
    ]
    # Force peak8 via dd: only one bar → dd nan; soft_bear still true.
    frame = build_early_down_frame(rows)
    assert frame.iloc[0]["level"] == "alert"
    paths = write_early_down_artifacts(
        frame, tmp_path, meta={"as_of": "2026-08-07", "trade_date": "2026-08-10"}
    )
    assert paths["csv"].exists()
    md = render_early_down_markdown(frame, meta={"as_of": "2026-08-07"})
    assert "不改 regime" in md
    assert "AAA" in md
