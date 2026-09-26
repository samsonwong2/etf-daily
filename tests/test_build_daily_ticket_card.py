"""Unit tests for daily ticket card assembly (no GARCH / Qlib)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from etf_daily.scripts.backtest_beat_or_hold_513650 import apply_new_veto
from etf_daily.plots.build_daily_ticket_card import (
    ANCHOR,
    META_KEYS,
    OUTPUT_COLUMNS,
    assemble_ticket_card,
    build_meta,
    write_outputs,
)


def _tickets(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=["code", "q05", "match"])


def _closes_for_mom(*, as_of: str, anchor_last: float, other_last: float) -> dict[str, pd.Series]:
    idx = pd.bdate_range(end=as_of, periods=8)
    base = np.ones(len(idx), dtype=float)
    other = base.copy()
    other[-1] = 1.0 + other_last
    anchor = base.copy()
    anchor[-1] = 1.0 + anchor_last
    return {
        ANCHOR: pd.Series(anchor, index=idx),
        "SH588710": pd.Series(other, index=idx),
        "SH515220": pd.Series(base, index=idx),
    }


def test_left_join_keeps_universe_when_tickets_incomplete():
    as_of = pd.Timestamp("2026-08-12")
    codes = ["SH513650", "SH588710", "SH515220"]
    tickets = _tickets(
        {"code": "SH513650", "q05": -0.02, "match": 1.1},
        {"code": "SH588710", "q05": -0.03, "match": 0.5},
    )
    _, floor = apply_new_veto(tickets)
    board = assemble_ticket_card(
        codes,
        {c: c for c in codes},
        as_of=as_of,
        tickets=tickets,
        regimes={},
        closes={},
        floor=floor,
    )
    assert list(board["code"]) == codes
    missing = board.set_index("code").loc["SH515220"]
    assert int(missing["garch_reliable"]) == 0
    assert int(missing["pass_q05_floor"]) == 0
    assert not np.isfinite(missing["q05_20d_garch"])


def test_omega_below_one_still_passes_q05_floor():
    as_of = pd.Timestamp("2026-08-12")
    codes = ["WEAK", "OK1", "OK2", "TAIL"]
    tickets = _tickets(
        {"code": "WEAK", "q05": -0.02, "match": 0.5},
        {"code": "OK1", "q05": -0.03, "match": 1.1},
        {"code": "OK2", "q05": -0.04, "match": 1.2},
        {"code": "TAIL", "q05": -0.20, "match": 1.0},
    )
    passed, floor = apply_new_veto(tickets)
    board = assemble_ticket_card(
        codes,
        {c: c for c in codes},
        as_of=as_of,
        tickets=tickets,
        regimes={},
        closes={},
        floor=floor,
    )
    weak = board.set_index("code").loc["WEAK"]
    assert "WEAK" in passed
    assert int(weak["pass_q05_floor"]) == 1
    assert float(weak["omega_match_20d"]) == pytest.approx(0.5)


def test_empty_tickets_keep_rows_and_nan_floor():
    as_of = pd.Timestamp("2026-08-12")
    codes = ["SH513650", "SH588710"]
    empty = pd.DataFrame(columns=["code", "q05", "match"])
    _, floor = apply_new_veto(empty)
    board = assemble_ticket_card(
        codes,
        {c: c for c in codes},
        as_of=as_of,
        tickets=empty,
        regimes={},
        closes={},
        floor=floor,
    )
    assert len(board) == 2
    assert not np.isfinite(floor)
    assert board["pass_q05_floor"].tolist() == [0, 0]
    assert not np.isfinite(board["q05_floor_20d"]).any()


def test_mom5_and_pass_up_and_q05():
    as_of = pd.Timestamp("2026-08-12")
    codes = [ANCHOR, "SH588710", "SH515220"]
    tickets = _tickets(
        {"code": ANCHOR, "q05": -0.02, "match": 1.0},
        {"code": "SH588710", "q05": -0.03, "match": 0.5},
        {"code": "SH515220", "q05": -0.20, "match": 1.0},
    )
    _, floor = apply_new_veto(tickets)
    idx = pd.bdate_range(end=as_of, periods=3)
    regimes = {
        "SH588710": pd.Series(["up", "up", "up"], index=idx),
        ANCHOR: pd.Series(["range", "range", "range"], index=idx),
    }
    board = assemble_ticket_card(
        codes,
        {c: c for c in codes},
        as_of=as_of,
        tickets=tickets,
        regimes=regimes,
        closes=_closes_for_mom(as_of=str(as_of.date()), anchor_last=0.01, other_last=0.04),
        floor=floor,
    )
    by = board.set_index("code")
    assert float(by.loc["SH588710", "mom5_vs_anchor"]) == pytest.approx(0.03)
    assert int(by.loc["SH588710", "in_up"]) == 1
    assert int(by.loc["SH588710", "pass_q05_floor"]) == 1
    assert int(by.loc["SH588710", "pass_up_and_q05"]) == 1
    assert int(by.loc[ANCHOR, "pass_up_and_q05"]) == 0


def test_column_contract_and_meta(tmp_path: Path):
    as_of = pd.Timestamp("2026-08-12")
    codes = ["SH513650"]
    tickets = _tickets({"code": "SH513650", "q05": -0.02, "match": 1.2})
    _, floor = apply_new_veto(tickets)
    board = assemble_ticket_card(
        codes,
        {"SH513650": "标普500ETF南方"},
        as_of=as_of,
        tickets=tickets,
        regimes={},
        closes={},
        floor=floor,
    )
    assert tuple(board.columns) == OUTPUT_COLUMNS
    assert "match" not in board.columns
    joined = "".join(board.columns)
    assert "风险" not in joined
    meta = build_meta(
        board,
        as_of=as_of,
        floor=floor,
        horizon=20,
        mc_samples=1000,
        sharpe_target=1.33,
    )
    assert tuple(meta) == META_KEYS
    out = tmp_path / "ticket_card.csv"
    write_outputs(board, meta, out)
    saved = json.loads((tmp_path / "ticket_card_meta.json").read_text(encoding="utf-8"))
    assert tuple(saved) == META_KEYS
    assert out.is_file()
