from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from decision_pack.src.pack_live_prices import (
    UNIVERSE_LIVE_CLOSE_CSV,
    apply_pack_live_closes,
    build_universe_live_close_frame,
    load_live_close_from_pack,
    load_portfolio_live_closes,
    should_fetch_spot,
)


def test_load_portfolio_live_closes(tmp_path: Path):
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    pd.DataFrame(
        [
            {
                "code": "SHAAA",
                "market_value": 1000.0,
                "shares": 100.0,
                "close_price": 10.5,
            },
            {
                "code": "SHBBB",
                "market_value": 500.0,
                "shares": 200.0,
                "close_price": "",
            },
            {"code": "CASH", "market_value": 100.0, "close_price": 1.0},
        ]
    ).to_csv(pack_dir / "portfolio_snapshot.csv", index=False)

    closes = load_portfolio_live_closes(pack_dir)
    assert closes == {"SHAAA": 10.5, "SHBBB": 2.5}


def test_load_live_close_prefers_universe_file(tmp_path: Path):
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    pd.DataFrame(
        [{"code": "SHAAA", "close_price": 10.5, "market_value": 1000.0, "shares": 100.0}]
    ).to_csv(pack_dir / "portfolio_snapshot.csv", index=False)
    pd.DataFrame(
        [
            {
                "code": "SHAAA",
                "name": "A",
                "close_price": 11.0,
                "source": "spot",
                "bar_date": "2024-06-17",
                "as_of": "2024-06-17",
            },
            {
                "code": "SHBBB",
                "name": "B",
                "close_price": 2.5,
                "source": "spot",
                "bar_date": "2024-06-17",
                "as_of": "2024-06-17",
            },
        ]
    ).to_csv(pack_dir / UNIVERSE_LIVE_CLOSE_CSV, index=False)

    assert load_live_close_from_pack(pack_dir) == {"SHAAA": 11.0, "SHBBB": 2.5}


def test_build_universe_live_close_priority(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "decision_pack.src.pack_live_prices.fetch_qlib_closes_with_dates",
        lambda codes, as_of, provider_uri=None: (
            {"SHAAA": 1.0, "SHBBB": 2.0, "SHCCC": 3.0},
            {"SHAAA": "2024-06-14", "SHBBB": "2024-06-14", "SHCCC": "2024-06-14"},
        ),
    )
    frame = build_universe_live_close_frame(
        as_of="2024-06-17",
        universe_codes=["SHAAA", "SHBBB", "SHCCC"],
        names={"SHAAA": "A", "SHBBB": "B", "SHCCC": "C"},
        portfolio_closes={"SHAAA": 1.2},
        spot_closes={"SHAAA": 1.25, "SHBBB": 2.5, "SHCCC": 3.5},
    )
    by_code = frame.set_index("code")
    assert by_code.loc["SHAAA", "close_price"] == pytest.approx(1.25)
    assert by_code.loc["SHAAA", "source"] == "spot"
    assert by_code.loc["SHBBB", "close_price"] == pytest.approx(2.5)
    assert by_code.loc["SHBBB", "source"] == "spot"
    assert by_code.loc["SHCCC", "close_price"] == pytest.approx(3.5)
    assert by_code.loc["SHCCC", "source"] == "spot"


def test_build_universe_live_close_portfolio_fallback_without_spot(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "decision_pack.src.pack_live_prices.fetch_qlib_closes_with_dates",
        lambda codes, as_of, provider_uri=None: (
            {"SHAAA": 1.0},
            {"SHAAA": "2024-06-14"},
        ),
    )
    frame = build_universe_live_close_frame(
        as_of="2024-06-17",
        universe_codes=["SHAAA"],
        portfolio_closes={"SHAAA": 1.2},
        spot_closes={},
    )
    row = frame.set_index("code").loc["SHAAA"]
    assert row["close_price"] == pytest.approx(1.2)
    assert row["source"] == "portfolio_live"


def test_should_fetch_spot_only_on_today(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "decision_pack.src.pack_live_prices.pd.Timestamp.today",
        lambda: pd.Timestamp("2024-06-17"),
    )
    assert should_fetch_spot("2024-06-17") is True
    assert should_fetch_spot("2024-06-16") is False


def test_apply_pack_live_closes_appends_as_of_row():
    idx = pd.bdate_range("2024-06-10", periods=5)
    panel = pd.DataFrame({"SHAAA": [1.0, 1.1, 1.2, 1.3, 1.4]}, index=idx)
    panel, stats = apply_pack_live_closes(
        panel,
        "2024-06-17",
        {"SHAAA": 1.55},
    )
    assert stats.injected == 1
    assert panel.index[-1] == pd.Timestamp("2024-06-17")
    assert panel.loc[pd.Timestamp("2024-06-17"), "SHAAA"] == pytest.approx(1.55)


def test_apply_pack_live_closes_overwrites_existing_as_of_row():
    idx = pd.bdate_range("2024-06-10", periods=8)
    panel = pd.DataFrame({"SHAAA": range(8)}, index=idx, dtype=float)
    as_of = idx[-1]
    panel, stats = apply_pack_live_closes(
        panel,
        as_of.strftime("%Y-%m-%d"),
        {"SHAAA": 99.0},
    )
    assert stats.injected == 1
    assert len(panel) == len(idx)
    assert panel.loc[as_of, "SHAAA"] == pytest.approx(99.0)


def test_apply_pack_live_closes_skips_unknown_columns():
    panel = pd.DataFrame({"SHAAA": [1.0]}, index=pd.bdate_range("2024-06-10", periods=1))
    panel, stats = apply_pack_live_closes(
        panel,
        "2024-06-11",
        {"SHBBB": 2.0},
    )
    assert stats.injected == 0
    assert len(panel) == 1


def test_apply_pack_live_closes_skips_duplicate_flat_day():
    """Same live price as last qlib bar must not append a 0% return day."""
    idx = pd.bdate_range("2024-06-10", periods=5)
    panel = pd.DataFrame({"SHAAA": [1.0, 1.1, 1.2, 1.3, 1.4]}, index=idx)
    panel, stats = apply_pack_live_closes(
        panel,
        "2024-06-17",
        {"SHAAA": 1.4},
    )
    assert stats.injected == 0
    assert stats.skipped_same == 1
    assert len(panel) == 5
    assert panel.index[-1] == pd.Timestamp("2024-06-14")


def test_apply_pack_live_closes_appends_only_changed_codes():
    idx = pd.bdate_range("2024-06-10", periods=5)
    panel = pd.DataFrame(
        {"SHAAA": [1.0, 1.1, 1.2, 1.3, 1.4], "SHBBB": [2.0, 2.1, 2.2, 2.3, 2.4]},
        index=idx,
    )
    panel, stats = apply_pack_live_closes(
        panel,
        "2024-06-17",
        {"SHAAA": 1.4, "SHBBB": 2.55},
    )
    assert stats.injected == 1
    assert stats.skipped_same == 1
    assert len(panel) == 6
    assert panel.index[-1] == pd.Timestamp("2024-06-17")
    assert pd.isna(panel.loc[pd.Timestamp("2024-06-17"), "SHAAA"])
    assert panel.loc[pd.Timestamp("2024-06-17"), "SHBBB"] == pytest.approx(2.55)
