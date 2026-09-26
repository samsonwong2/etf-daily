"""Tests for frozen AkShare intraday OHLCV snapshots."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from decision_pack.src.regime_transition_live_snapshot import (
    LiveSnapshotError,
    apply_live_snapshot_to_close_panel,
    apply_live_snapshot_to_ohlcv,
    capture_live_snapshot,
    codes_with_equity_anchor,
    fetch_akshare_etf_spot_ohlcv,
    load_live_snapshot,
    validate_snapshot,
    write_live_snapshot,
)
from decision_pack.src.vol_need_return_board import EQUITY_ANCHOR


def _spot_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "代码": "513310",
                "最新价": 5.50,
                "开盘价": 5.20,
                "最高价": 5.60,
                "最低价": 5.10,
                "成交额": 1.2e8,
            },
            {
                "代码": "588710",
                "最新价": 1.25,
                "今开": 1.20,
                "最高": 1.28,
                "最低": 1.18,
                "成交量": 9.9e6,
            },
            {
                "代码": "515050",
                "最新价": 0.0,  # invalid — should be dropped
                "开盘价": 1.0,
                "最高价": 1.0,
                "最低价": 1.0,
                "成交额": 1.0,
            },
        ]
    )


def test_fetch_akshare_etf_spot_ohlcv_normalizes_and_filters():
    frame = fetch_akshare_etf_spot_ohlcv(
        ["SH513310", "SH588710", "SH515050"],
        as_of="2026-07-22",
        captured_at="2026-07-22T10:15:00",
        spot_df=_spot_frame(),
    )
    assert set(frame["code"]) == {"SH513310", "SH588710"}
    row = frame.set_index("code").loc["SH513310"]
    assert row["as_of"] == "2026-07-22"
    assert row["close"] == pytest.approx(5.50)
    assert row["open"] == pytest.approx(5.20)
    assert row["high"] == pytest.approx(5.60)
    assert row["low"] == pytest.approx(5.10)
    assert row["source"] == "akshare_sina_spot"


def test_validate_snapshot_requires_full_coverage():
    frame = fetch_akshare_etf_spot_ohlcv(
        ["SH513310", "SH588710"],
        as_of="2026-07-22",
        spot_df=_spot_frame(),
    )
    with pytest.raises(LiveSnapshotError, match="missing"):
        validate_snapshot(
            frame,
            as_of="2026-07-22",
            required_codes=["SH513310", "SH588710", "SH999999"],
            strict_coverage=True,
        )


def test_validate_snapshot_rejects_wrong_as_of():
    frame = fetch_akshare_etf_spot_ohlcv(
        ["SH513310"],
        as_of="2026-07-22",
        spot_df=_spot_frame(),
    )
    with pytest.raises(LiveSnapshotError, match="as_of mismatch"):
        validate_snapshot(frame, as_of="2026-07-21", required_codes=["SH513310"])


def test_validate_snapshot_require_today():
    frame = fetch_akshare_etf_spot_ohlcv(
        ["SH513310"],
        as_of="2026-07-22",
        spot_df=_spot_frame(),
    )
    with pytest.raises(LiveSnapshotError, match="as_of==today"):
        validate_snapshot(
            frame,
            as_of="2026-07-22",
            required_codes=["SH513310"],
            require_today=True,
            today="2026-07-21",
        )


def test_codes_with_equity_anchor_appends_missing_anchor():
    out = codes_with_equity_anchor(["SH588710", "SZ159985"])
    assert out[0] == "SH588710"
    assert EQUITY_ANCHOR in out
    assert out[-1] == EQUITY_ANCHOR
    # Idempotent when pool already includes the anchor.
    again = codes_with_equity_anchor([EQUITY_ANCHOR, "SH588710"])
    assert again.count(EQUITY_ANCHOR) == 1


def test_capture_and_load_roundtrip(tmp_path: Path):
    out = tmp_path / "live_snapshot.csv"
    path, frame = capture_live_snapshot(
        as_of="2026-07-22",
        codes=["SH513310", "SH588710"],
        output_path=out,
        require_today=False,
        strict_coverage=True,
        spot_df=_spot_frame(),
    )
    assert path.exists()
    loaded = load_live_snapshot(
        path,
        as_of="2026-07-22",
        required_codes=["SH513310", "SH588710"],
    )
    assert len(loaded) == 2
    assert list(loaded.columns) == list(frame.columns)


def test_apply_live_snapshot_to_close_panel_appends_all_codes_including_flat():
    idx = pd.bdate_range("2026-07-17", periods=3)  # Fri..Tue -> 17,20,21
    panel = pd.DataFrame(
        {
            "SH513310": [5.0, 5.1, 5.2],
            "SH588710": [1.1, 1.2, 1.25],
        },
        index=idx,
    )
    snap = fetch_akshare_etf_spot_ohlcv(
        ["SH513310", "SH588710"],
        as_of="2026-07-22",
        spot_df=_spot_frame(),
    )
    # Force one code flat vs last close to prove we still write the as_of row.
    snap.loc[snap["code"] == "SH588710", "close"] = 1.25
    snap.loc[snap["code"] == "SH588710", "open"] = 1.25
    snap.loc[snap["code"] == "SH588710", "high"] = 1.25
    snap.loc[snap["code"] == "SH588710", "low"] = 1.25

    out, stats = apply_live_snapshot_to_close_panel(panel, snap, as_of="2026-07-22")
    assert stats.appended is True
    assert stats.n_codes == 2
    assert pd.Timestamp("2026-07-22") in out.index
    assert out.loc[pd.Timestamp("2026-07-22"), "SH513310"] == pytest.approx(5.50)
    assert out.loc[pd.Timestamp("2026-07-22"), "SH588710"] == pytest.approx(1.25)


def test_apply_live_snapshot_replaces_existing_as_of_row():
    idx = pd.to_datetime(["2026-07-21", "2026-07-22"])
    panel = pd.DataFrame({"SH513310": [5.0, 9.9]}, index=idx)
    snap = fetch_akshare_etf_spot_ohlcv(
        ["SH513310"],
        as_of="2026-07-22",
        spot_df=_spot_frame(),
    )
    out, stats = apply_live_snapshot_to_close_panel(panel, snap, as_of="2026-07-22")
    assert stats.replaced_existing is True
    assert len(out) == 2
    assert out.loc[pd.Timestamp("2026-07-22"), "SH513310"] == pytest.approx(5.50)


def test_apply_live_snapshot_to_ohlcv_sets_pct_chg():
    ohlcv = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-07-21"]),
            "$open": [5.0],
            "$high": [5.1],
            "$low": [4.9],
            "$close": [5.0],
            "$volume": [1e6],
            "$pct_change": [0.0],
        }
    )
    snap = fetch_akshare_etf_spot_ohlcv(
        ["SH513310"],
        as_of="2026-07-22",
        spot_df=_spot_frame(),
    )
    out = apply_live_snapshot_to_ohlcv(
        ohlcv, snap, code="SH513310", as_of="2026-07-22"
    )
    assert len(out) == 2
    last = out.iloc[-1]
    assert pd.Timestamp(last["datetime"]).normalize() == pd.Timestamp("2026-07-22")
    assert last["$close"] == pytest.approx(5.50)
    assert last["$pct_change"] == pytest.approx(10.0)


def test_apply_live_snapshot_aligns_raw_spot_to_qlib_qfq_via_pre_close():
    """SH588710-class: Sina raw ~1.1 vs Qlib qfq ~3.16; scale by 昨收 bridge."""
    from decision_pack.src.regime_transition_live_snapshot import qlib_align_factor

    factor, reason = qlib_align_factor(
        3.16, raw_close=1.1, pre_close=1.053, pct_chg=4.463
    )
    assert reason == "pre_close"
    assert factor == pytest.approx(3.16 / 1.053, rel=1e-6)

    ohlcv = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-08-14"]),
            "$open": [3.16],
            "$high": [3.187],
            "$low": [3.09],
            "$close": [3.16],
            "$volume": [1e6],
        }
    )
    snap = pd.DataFrame(
        [
            {
                "code": "SH588710",
                "as_of": "2026-08-17",
                "captured_at": "2026-08-17T11:40:00",
                "open": 1.053,
                "high": 1.106,
                "low": 1.04,
                "close": 1.1,
                "volume": 1e9,
                "source": "akshare_sina_spot",
                "pre_close": 1.053,
                "pct_chg": 4.463,
            }
        ]
    )
    out = apply_live_snapshot_to_ohlcv(
        ohlcv, snap, code="SH588710", as_of="2026-08-17"
    )
    last = out.iloc[-1]
    assert last["$close"] == pytest.approx(1.1 * (3.16 / 1.053), rel=1e-6)
    assert last["$open"] == pytest.approx(1.053 * (3.16 / 1.053), rel=1e-6)
    # Continuity: open ≈ prior qlib close when spot open == 昨收
    assert last["$open"] == pytest.approx(3.16, rel=1e-6)
    assert last["$pct_change"] == pytest.approx(4.463, rel=1e-6)


def test_fetch_includes_pre_close_and_pct_chg():
    frame = fetch_akshare_etf_spot_ohlcv(
        ["SH588710"],
        as_of="2026-08-17",
        spot_df=pd.DataFrame(
            [
                {
                    "代码": "588710",
                    "最新价": 1.1,
                    "今开": 1.053,
                    "最高": 1.106,
                    "最低": 1.04,
                    "成交量": 1e9,
                    "昨收": 1.053,
                    "涨跌幅": 4.463,
                }
            ]
        ),
    )
    row = frame.iloc[0]
    assert row["pre_close"] == pytest.approx(1.053)
    assert row["pct_chg"] == pytest.approx(4.463)


def test_codes_requiring_live_coverage_skips_historical_only():
    from decision_pack.src.regime_transition_live_snapshot import (
        codes_requiring_live_coverage,
    )

    signals = pd.DataFrame(
        [
            {"code": "SH513310", "as_of": "2026-07-29"},
            {"code": "SZ159985", "as_of": "2026-07-29"},
            {"code": "SZ159616", "as_of": "2026-06-30"},  # rotated out
        ]
    )
    required, skipped = codes_requiring_live_coverage(signals, as_of="2026-07-29")
    assert required == ["SH513310", "SZ159985"]
    assert skipped == ["SZ159616"]


def test_write_live_snapshot_schema(tmp_path: Path):
    from decision_pack.src.regime_transition_live_snapshot import (
        SNAPSHOT_COLUMNS,
        SNAPSHOT_OPTIONAL_COLUMNS,
    )

    snap = fetch_akshare_etf_spot_ohlcv(
        ["SH513310"],
        as_of="2026-07-22",
        spot_df=_spot_frame(),
    )
    path = write_live_snapshot(snap, tmp_path / "s.csv")
    loaded = pd.read_csv(path)
    assert list(loaded.columns)[: len(SNAPSHOT_COLUMNS)] == list(SNAPSHOT_COLUMNS)
    for col in SNAPSHOT_OPTIONAL_COLUMNS:
        assert col in loaded.columns
