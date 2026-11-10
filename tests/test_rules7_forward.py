"""Forward ledger rows, segment labels, and the K1-K3 failure tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from etf_daily.lib import rules7  # noqa: E402
from etf_daily.lib.rules7_forward import (  # noqa: E402
    CRITERIA,
    _clean_json,
    assess,
    gate_open,
    k1_status,
    k2_false_fail_rate,
    k2_status,
    k3_status,
    overall_status,
    replay_frame,
    segment_for,
    update_daily,
    write_holdout,
    write_ledger,
)
from etf_daily.scripts.backtest_fig9_touch import prepare_fig9_touch_frame  # noqa: E402


def _frame(start: str, n: int = 8) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n)
    px = np.array([10.0, 10.0, 10.5, 12.0, 11.0, 10.9, 10.9, 10.9])
    frame = pd.DataFrame(index=idx)
    frame["px"] = px
    frame["ma20"] = 11.0
    frame["ma5"] = 10.5
    frame["gap_5d"] = -0.01
    frame["gap_20d"] = -0.01
    frame["fig9_lower"] = np.where(np.arange(n) == 0, 10.0, 8.0)
    frame["fig9_upper"] = 20.0
    frame["fig10_lower"] = 8.0
    frame["fig10_upper"] = 20.0
    frame["fig8_upper"] = 30.0
    frame["fig7_gap"] = 0.0
    frame["fig8_gap"] = 0.0
    for key in (7, 8, 9, 10):
        frame[f"fig{key}_g"] = 0.01
    frame["fig9_touch_hi"] = False
    frame["fig10_touch_hi"] = False
    frame["locmax"] = False
    frame["locmin"] = False
    frame["vol5_pct_120d"] = 0.1
    # Day 0 is near the fig9 lower rail and the aux picture is on, so aux_edge fires once.
    frame.loc[idx[0], "px"] = 10.0
    return prepare_fig9_touch_frame(frame)


def _spec() -> rules7.RuleSpec:
    return rules7.OVERRIDES["SH512530"][1]


def test_segment_boundaries():
    assert segment_for("2026-10-13") == "in_sample"
    assert segment_for("2026-10-14") == "pseudo_forward"
    assert segment_for("2026-11-10") == "pseudo_forward"
    assert segment_for("2026-11-11") == "forward"


def test_replay_matches_evaluate_and_labels_the_segment():
    frame = _frame("2026-09-01")
    spec = _spec()
    replay = replay_frame("SH512530", "红利", frame, typ="平稳震荡", spec=spec)
    evaluated = rules7.evaluate("SH512530", frame)
    assert len(evaluated["trades"]) == 1
    assert evaluated["trades"][0]["why"] == "回落"
    closed = [row for row in replay["rows"] if row["exit_reason"] == "回落"]
    assert len(closed) == 1
    assert closed[0]["segment"] == "in_sample"
    assert np.isclose(closed[0]["ret_gross"], evaluated["trades"][0]["ret"])
    assert closed[0]["fill_date"] == "2026-09-02"
    assert closed[0]["exit_signal_date"] == "2026-09-07"
    assert closed[0]["exit_fill_date"] == "2026-09-08"
    assert np.isclose(closed[0]["ret_gross"], closed[0]["exit_fill_px"] / closed[0]["fill_px"] - 1)
    assert closed[0]["ret_net"] < closed[0]["ret_gross"]

    later = replay_frame("SH512530", "红利", _frame("2026-10-20"), typ="平稳震荡", spec=spec)
    assert later["rows"][0]["segment"] == "pseudo_forward"
    ahead = replay_frame("SH512530", "红利", _frame("2026-11-12"), typ="平稳震荡", spec=spec)
    assert ahead["rows"][0]["segment"] == "forward"


def test_ledger_write_is_idempotent_and_daily_keeps_older_days(tmp_path: Path):
    frame = _frame("2026-09-01")
    rows = replay_frame("SH512530", "红利", frame, typ="平稳震荡", spec=_spec())["rows"]
    path = tmp_path / "ledger.csv"
    write_ledger(path, rows)
    first = path.read_bytes()
    write_ledger(path, list(reversed(rows)))
    assert path.read_bytes() == first

    daily = tmp_path / "daily.csv"
    update_daily(daily, [{"as_of": "2026-11-09", "code": "SH512530", "holding": "1", "action": "持有",
                          "equity": 1.1, "bh_equity": 1.2, "spec_fp": "aa"}], "2026-11-09")
    update_daily(daily, [{"as_of": "2026-11-10", "code": "SH512530", "holding": "0", "action": "观望",
                          "equity": 1.2, "bh_equity": 1.3, "spec_fp": "aa"}], "2026-11-10")
    update_daily(daily, [{"as_of": "2026-11-10", "code": "SH512530", "holding": "0", "action": "观望",
                          "equity": 9.9, "bh_equity": 1.3, "spec_fp": "bb"}], "2026-11-10")
    saved = pd.read_csv(daily, dtype=str)
    assert list(saved["as_of"]) == ["2026-11-09", "2026-11-10"]
    assert saved.loc[saved["as_of"] == "2026-11-10", "equity"].item() == "9.9"
    holdout = tmp_path / "holdout.json"
    write_holdout(holdout)
    text = holdout.read_text(encoding="utf-8")
    holdout.write_text("changed", encoding="utf-8")
    write_holdout(holdout)
    assert holdout.read_text(encoding="utf-8") == "changed"
    assert "33" in text


def test_k2_threshold_false_fail_rate_is_under_five_percent():
    assert k2_false_fail_rate() <= 0.05
    assert abs(k2_false_fail_rate() - 0.0426) < 0.001


def test_k1_k2_k3_boundaries():
    assert k1_status([{"bh_mdd": -0.01, "strat_mdd": -0.2}] * 20)["status"] == "未定"
    half = [{"bh_mdd": -0.10, "strat_mdd": -0.04}] * 5 + [{"bh_mdd": -0.10, "strat_mdd": -0.20}] * 5
    assert k1_status(half)["status"] == "通过"
    worse = [{"bh_mdd": -0.10, "strat_mdd": -0.04}] * 4 + [{"bh_mdd": -0.10, "strat_mdd": -0.20}] * 6
    assert k1_status(worse)["status"] == "不通过"

    short = k2_status([0.01] * 11 + [-0.01] * 18)
    assert short["status"] == "未定"
    assert k2_status([0.01] * 11 + [-0.01] * 19)["status"] == "不通过"
    # 16 wins of 30 clears 11/30, and the median of that mix is positive.
    assert k2_status([0.02] * 16 + [-0.01] * 14)["status"] == "通过"
    # 14 wins still clears 11/30, but the median is negative.
    assert k2_status([0.02] * 14 + [-0.01] * 16)["status"] == "不通过"

    assert k3_status(-0.20, -0.115, -0.05) == "不通过"
    assert k3_status(-0.20, -0.115, -0.30) == "通过"
    assert k3_status(-0.15, -0.115, -0.05) == "通过"
    assert k3_status(float("nan"), -0.115, -0.05) == "未定"


def test_overall_waits_unless_k3_trips():
    assert overall_status("通过", "通过", "通过", waiting=True) == "未定"
    assert overall_status("通过", "通过", "不通过", waiting=True) == "不通过"
    assert overall_status("通过", "通过", "通过", waiting=False) == "通过"
    assert overall_status("不通过", "通过", "通过", waiting=False) == "不通过"
    assert gate_open("2026-11-11", 30) is True
    assert gate_open(CRITERIA["gate_date"], 29) is True
    assert gate_open(CRITERIA["gate_date"], 30) is False


def test_forward_commands_reject_an_empty_listing(tmp_path: Path):
    from etf_daily.cli import _forward, _forward_page

    listing = tmp_path / "empty"
    listing.mkdir()
    assert _forward(["--listing-dir", str(listing), "--as-of", "2026-11-09"]) == 2
    assert _forward_page(["--listing-dir", str(listing), "--as-of", "2026-11-09", "--codes", "SH512530"]) == 2


def test_assess_uses_only_forward_closed_trades():
    idx = pd.bdate_range("2026-11-11", periods=5)
    flat = pd.Series(np.linspace(1.0, 0.80, 5), index=idx)
    bh = pd.Series(np.linspace(1.0, 0.95, 5), index=idx)
    specs = {"codes": {"SH512530": {"universe": "study", "in_sample_mdd": -0.115}}}
    curves = {"SH512530": {"equity": flat, "bh": bh}}
    rows = [
        {"segment": "in_sample", "exit_reason": "回落", "ret_net": 0.5},
        {"segment": "forward", "exit_reason": "持有中", "ret_net": ""},
        {"segment": "forward", "exit_reason": "卖点", "ret_net": -0.01},
    ]
    verdict = assess(curves, rows, specs, "2026-11-20")
    assert verdict["k2"]["n"] == 1
    assert verdict["k2"]["status"] == "未定"
    assert verdict["k3"]["status"] == "不通过"
    assert verdict["overall"] == "不通过"


def test_replay_carries_a_missing_close_forward():
    frame = _frame("2026-09-01")
    frame.loc[frame.index[4], "px"] = np.nan
    replay = replay_frame("SH512530", "红利", frame, typ="平稳震荡", spec=_spec())
    assert replay["result"]["equity"].notna().all()


def test_k3_is_undecided_before_any_forward_bar():
    idx = pd.bdate_range("2026-10-20", periods=5)
    curve = pd.Series(np.linspace(1.0, 1.1, 5), index=idx)
    specs = {"codes": {"SH512530": {"universe": "study", "in_sample_mdd": -0.115}}}
    verdict = assess({"SH512530": {"equity": curve, "bh": curve}}, [], specs, "2026-10-24")
    assert verdict["k3"]["codes"]["SH512530"]["status"] == "未定"


def test_verdict_json_has_no_bare_nan():
    import json

    text = json.dumps(_clean_json({"a": float("nan"), "b": [np.float64("inf"), 1.0]}), allow_nan=False)
    assert json.loads(text) == {"a": None, "b": [None, 1.0]}
