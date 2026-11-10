"""Tests for the rules7 state machine (synthetic series, no HTML)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from etf_daily.lib.rules7 import (  # noqa: E402
    TYPE_CYCLE,
    TYPE_HIVOL,
    TYPE_SHORT,
    TYPE_SMOOTH,
    TYPE_TREND,
    OVERRIDES,
    RuleSpec,
    buy_sell_masks,
    classify,
    flat_buy_quote,
    indicator_snapshot,
    run_state_machine,
    triggered,
)


def _series(values, dtype=float) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=dtype), index=pd.bdate_range("2024-01-01", periods=len(values)))


def _mask(n: int, on: list[int]) -> pd.Series:
    a = np.zeros(n, bool)
    a[on] = True
    return _series(a, bool)


def test_trail_peak_starts_at_fill_bar():
    close = _series([10.0, 8.0, 7.6, 9.0, 8.0, 8.0])
    res = run_state_machine(close, _mask(6, [0]), _mask(6, []), trail=0.10)
    # 7.6 is -24% from the signal close but only -5% from the fill; no exit there.
    assert len(res["trades"]) == 1
    t = res["trades"][0]
    assert t["sell"] == close.index[4] and t["why"] == "回落"
    assert np.isclose(t["ret"], 8.0 / 8.0 - 1)


def test_g1_suppresses_rail_sell():
    close = _series(np.linspace(10, 12, 8))
    buy, sell = _mask(8, [0]), _mask(8, [5])
    plain = run_state_machine(close, buy, sell)
    assert plain["trades"][0]["sell"] == close.index[5]
    held = run_state_machine(close, buy, sell, g1=_mask(8, [3]))
    assert held["trades"] == []
    assert held["state"]["holding"] and held["state"]["g1_seen"]
    assert held["state"]["action"] == "持有"


def test_buy_on_last_bar_is_today_action():
    close = _series([10.0, 10.1, 10.2, 10.3])
    res = run_state_machine(close, _mask(4, [3]), _mask(4, []), trail=0.15)
    st = res["state"]
    assert st["action"] == "买"
    assert not st["holding"]
    assert st["fill_px"] is None
    assert res["exposure"] == 0.0


def test_fast_pattern_blocks_buy():
    n = 5
    masks = {
        "ae_buy": _mask(n, [1, 3]),
        "b1": _mask(n, [4]),
        "fast": _mask(n, [1, 4]),
        "ae_sell": _mask(n, []),
        "fig8_sell": _mask(n, []),
    }
    buy, _ = buy_sell_masks(masks, RuleSpec(b1=True, fast_block=True, sell="fig9", g1_hold=True, trail=0.15))
    assert list(np.flatnonzero(buy.to_numpy())) == [3]
    buy, _ = buy_sell_masks(masks, RuleSpec(b1=True, fast_block=False, sell="fig9", g1_hold=True, trail=0.15))
    assert list(np.flatnonzero(buy.to_numpy())) == [1, 3, 4]


def test_classify_order():
    base = dict(bars=1000, vol=0.20, bh_cagr=0.12, bh_mdd=-0.25, cross_per_year=8)
    assert classify({**base, "bars": 100}) == TYPE_SHORT
    assert classify({**base, "vol": 0.35, "bh_mdd": -0.47}) == TYPE_HIVOL
    assert classify({**base, "bh_mdd": -0.48}) == TYPE_CYCLE
    assert classify({**base, "bh_cagr": 0.02}) == TYPE_CYCLE
    assert classify({**base, "vol": 0.16, "cross_per_year": 10}) == TYPE_SMOOTH
    assert classify(base) == TYPE_TREND


def _snap_frame(n: int = 120) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=n)
    px = np.linspace(10.0, 12.6, n)
    px[-1] = 9.9  # today: within 1% of the 10.0 lower rail; yesterday was not
    f = pd.DataFrame(index=idx)
    f["px"] = px
    f["fig9_lower"], f["fig9_upper"], f["fig8_upper"] = 10.0, 13.0, 14.0
    f["ma5"], f["ma20"] = 10.5, 11.0
    f["gap_5d"], f["gap_20d"] = -0.01, -0.02
    f["fig7_gap"], f["fig8_gap"] = -0.10, -0.05
    f["fig9_pos"], f["fig10_pos"] = 0.5, 0.5
    for k in (7, 8, 9, 10):
        f[f"fig{k}_g"] = 0.1
    f["vol5_pct_120d"] = 0.5
    for col in ("fig9_touch_hi", "fig10_touch_hi", "locmax"):
        f[col] = False
    f["aux_buy"] = (f["px"] < f["ma20"]) & (f["ma5"] < f["ma20"])
    f["aux_sell"] = False
    f["fig9_mid_stretch_sell"] = False
    return f


def test_indicator_snapshot_today_flags():
    f = _snap_frame()
    n = len(f)
    near_lo = f["px"] <= f["fig9_lower"] * 1.01
    edge = near_lo & ~near_lo.shift(fill_value=False)
    masks = {
        "ae_buy": edge & f["aux_buy"],
        "ae_sell": _mask(n, [n - 1]),
        "b1": near_lo & (f["fig7_gap"] <= -0.08),
        "fast": _mask(n, []),
        "fig8_sell": _mask(n, []),
        "g1": _mask(n, []),
    }
    spec = RuleSpec(b1=True, fast_block=True, sell="fig9", g1_hold=True, trail=0.15)
    state = {"trail_hit": False, "sell_masked_by_g1": True, "stop": 9.0}
    s = indicator_snapshot(f, masks, spec, state)
    assert s["near_lo"] and s["near_lo_edge"] and not s["near_lo_prev"]
    assert not np.isfinite(s["lo9_ewma"]) and not s["near_lo_ewma"] and not s["near_lo_ewma_edge"]
    assert s["aux_buy"] and s["ae_buy"] and s["b1"] and s["disc_ok"]
    assert s["buy_signal"] and not s["fast"]
    assert not s["near_hi"] and not s["aux_sell"]
    assert np.isclose(s["dist_lo9"], 9.9 / 10.0 - 1)
    assert np.isclose(s["dist_stop"], 9.9 / 9.0 - 1)
    labels = triggered(s)
    assert {"近下轨上跳沿", "辅助买像", "aux_edge买点", "B1", "买入信号", "卖点被G1屏蔽"} <= set(labels)
    assert "快形态" not in labels


def _gold_snap(**over) -> dict:
    snap = {
        "lo9": 7.930,
        "px": 8.633,
        "near_lo": False,
        "aux_buy": True,
        "fast": False,
        "disc_ok": False,
        "buy_signal": False,
    }
    snap.update(over)
    return snap


def test_flat_buy_quote_gold_still_short():
    spec = OVERRIDES["SH518880"][1]
    price, status = flat_buy_quote(_gold_snap(), spec)
    assert np.isclose(price, 7.930 * 1.01)
    assert status == "还差这个价"


def test_flat_buy_quote_already_inside_band():
    spec = OVERRIDES["SH518880"][1]
    _, status = flat_buy_quote(_gold_snap(near_lo=True), spec)
    assert status == "已在带内，明天同价不算"


def test_flat_buy_quote_fast_blocks_before_buy_signal():
    spec = OVERRIDES["SH518880"][1]
    _, status = flat_buy_quote(_gold_snap(fast=True, buy_signal=True, near_lo=True), spec)
    assert status == "快形态挡住"


def test_flat_buy_quote_fast_without_trigger_is_not_a_blocked_buy():
    spec = OVERRIDES["SH518880"][1]
    _, status = flat_buy_quote(_gold_snap(fast=True), spec)
    assert status == "快形态中，未到买点"


def test_flat_buy_quote_fast_blocks_raw_aux_edge_and_b1():
    spec = RuleSpec(b1=True, fast_block=True, sell="fig9", g1_hold=True, trail=0.15)
    _, by_ae = flat_buy_quote(_gold_snap(fast=True, ae_buy=True), spec)
    _, by_b1 = flat_buy_quote(_gold_snap(fast=True, b1=True), spec)
    assert by_ae == by_b1 == "快形态挡住"
    no_b1 = RuleSpec(b1=False, fast_block=True, sell="fig9", g1_hold=True, trail=0.15)
    _, ignored = flat_buy_quote(_gold_snap(fast=True, b1=True), no_b1)
    assert ignored == "快形态中，未到买点"


def test_flat_buy_quote_fast_ignored_when_rule_does_not_block():
    spec = RuleSpec(b1=True, fast_block=False, sell="fig9", g1_hold=True, trail=0.15)
    _, status = flat_buy_quote(_gold_snap(fast=True), spec)
    assert status == "还差这个价"


def test_fast_clear_bars():
    from etf_daily.lib.rules7 import fast_clear_bars

    assert fast_clear_bars(pd.Series(np.full(120, 1.0))) == 0
    # Jump to 1.3 at bar 80: the 40-bar run is >= 20% for bars 80..119, then drops out.
    jump = np.r_[np.full(80, 1.0), np.full(100, 1.3)]
    assert fast_clear_bars(pd.Series(jump[:120])) == 60  # still hitting on the last bar
    assert fast_clear_bars(pd.Series(jump[:150])) == 30  # flag clears at bar 179
    assert fast_clear_bars(pd.Series(jump[:180])) == 0


def test_flat_buy_quote_b1_needs_only_the_rail():
    spec = RuleSpec(b1=True, fast_block=False, sell="fig9", g1_hold=True, trail=0.15)
    _, status = flat_buy_quote(_gold_snap(disc_ok=True, aux_buy=False), spec)
    assert status == "贴轨即可"


def test_flat_buy_quote_aux_buy_missing():
    spec = OVERRIDES["SH518880"][1]
    _, status = flat_buy_quote(_gold_snap(aux_buy=False), spec)
    assert status == "辅助买像未齐"


def test_flat_buy_quote_buy_signal_today():
    spec = OVERRIDES["SH518880"][1]
    _, status = flat_buy_quote(_gold_snap(buy_signal=True, near_lo=True), spec)
    assert status == "今天已是买点"


def test_flat_buy_quote_not_applicable():
    spec = OVERRIDES["SH518880"][1]
    price, status = flat_buy_quote(_gold_snap(buy_signal=True), spec, applicable=False)
    assert np.isclose(price, 7.930 * 1.01)
    assert status == "规则不适用"


def test_flat_buy_quote_missing_lower_rail():
    spec = OVERRIDES["SH518880"][1]
    price, status = flat_buy_quote(_gold_snap(lo9=float("nan")), spec)
    assert price is None
    assert status == "无下轨"


def _ewma_snap(**over) -> dict:
    snap = _gold_snap(
        lo9_ewma=8.0,
        near_lo_ewma=False,
        near_lo_ewma_edge=False,
    )
    snap.update(over)
    return snap


def test_flat_buy_quote_ewma_ignores_p95_buy_signal():
    spec = OVERRIDES["SH518880"][1]
    price, status = flat_buy_quote(_ewma_snap(buy_signal=True), spec, rail="ewma")
    assert np.isclose(price, 8.0 * 1.01)
    assert status == "还差这个价"
    p95, _ = flat_buy_quote(_ewma_snap(lo9_ewma=1.0), spec)
    assert np.isclose(p95, 7.930 * 1.01)


def test_flat_buy_quote_ewma_edge_is_today_buy():
    spec = RuleSpec(b1=False, fast_block=True, sell="fig9", g1_hold=True, trail=0.15)
    snap = _ewma_snap(near_lo_ewma=True, near_lo_ewma_edge=True, aux_buy=True)
    _, status = flat_buy_quote(snap, spec, rail="ewma")
    assert status == "今天已是买点"
    _, blocked = flat_buy_quote({**snap, "fast": True}, spec, rail="ewma")
    assert blocked == "快形态挡住"


def test_flat_buy_quote_ewma_b1_uses_ewma_rail():
    spec = RuleSpec(b1=True, fast_block=False, sell="fig9", g1_hold=True, trail=0.15)
    on_rail = _ewma_snap(disc_ok=True, near_lo_ewma=True, aux_buy=False)
    _, status = flat_buy_quote(on_rail, spec, rail="ewma")
    assert status == "今天已是买点"
    _, waiting = flat_buy_quote({**on_rail, "near_lo_ewma": False}, spec, rail="ewma")
    assert waiting == "贴轨即可"


def test_flat_buy_quote_ewma_status_matches_p95_wording():
    spec = OVERRIDES["SH518880"][1]
    _, status = flat_buy_quote(_ewma_snap(aux_buy=False), spec, rail="ewma")
    assert status == "辅助买像未齐"
    _, inside = flat_buy_quote(_ewma_snap(near_lo_ewma=True), spec, rail="ewma")
    assert inside == "已在带内，明天同价不算"
    price, skipped = flat_buy_quote(_ewma_snap(buy_signal=True), spec, applicable=False, rail="ewma")
    assert np.isclose(price, 8.0 * 1.01)
    assert skipped == "规则不适用"
    missing, why = flat_buy_quote(_ewma_snap(lo9_ewma=float("nan")), spec, rail="ewma")
    assert missing is None and why == "无下轨"


def test_indicator_snapshot_ewma_lower_matches_envelope():
    from etf_daily.lib.fair_path_research import DEFAULT_MIN_BARS, ewma_scaled_envelope

    f = _snap_frame()
    f["fig9_path"] = 11.0
    n = len(f)
    masks = {k: _mask(n, []) for k in ("ae_buy", "ae_sell", "b1", "fast", "fig8_sell", "g1")}
    spec = RuleSpec(b1=True, fast_block=True, sell="fig9", g1_hold=True, trail=0.15)
    s = indicator_snapshot(f, masks, spec, {"trail_hit": False, "sell_masked_by_g1": False, "stop": None})
    _upper, lower, _delta = ewma_scaled_envelope(f["fig9_path"], f["px"], min_bars=int(DEFAULT_MIN_BARS))
    lo, lo_prev = float(lower.iloc[-1]), float(lower.iloc[-2])
    assert np.isfinite(lo) and np.isclose(s["lo9_ewma"], lo)
    assert np.isclose(s["dist_lo9_ewma"], float(f["px"].iloc[-1]) / lo - 1)
    near = bool(float(f["px"].iloc[-1]) <= lo * 1.01)
    near_prev = bool(float(f["px"].iloc[-2]) <= lo_prev * 1.01)
    assert s["near_lo_ewma"] is near
    assert s["near_lo_ewma_edge"] is (near and not near_prev)
    assert s["buy_signal"] is False


def test_md_table_missing_column_does_not_raise():
    from etf_daily.scripts.scan_rules7_checklist import _md_table

    frame = pd.DataFrame([{"代码": "SH510300", "名称": "沪深300"}])
    lines = _md_table(frame, ["代码", "快形态"])
    assert "SH510300" in lines[2]
    assert "—" in lines[2]


def test_flat_buy_section_lists_holdings_with_a_price(tmp_path):
    from etf_daily.scripts.scan_rules7_checklist import (
        _format_flat_buy_section,
        annotate_flat_buy_quotes,
    )

    frame = annotate_flat_buy_quotes(pd.DataFrame([
        {
            "代码": "SH513080", "名称": "法国ETF华安", "今日状态": "持仓", "今日动作": "持有",
            "类型": "慢趋势", "买规则": "aux_edge 或 B1", "T收盘": "1.6550",
            "图9下轨": "1.6377", "距图9下轨": "1.1%",
            "快形态(≥20%)": "", "买入信号": "", "图7或图8折价≤-8%": "是",
            "辅助买像": "是", "图9贴下轨(≤下轨×1.01)": "",
        },
        {
            "代码": "SH510300", "名称": "沪深300ETF华泰柏瑞", "今日状态": "空仓", "今日动作": "观望",
            "类型": "长期下跌或周期", "买规则": "aux_edge 或 B1", "T收盘": "4.3850",
            "图9下轨": "4.1171", "距图9下轨": "6.5%",
            "快形态(≥20%)": "", "买入信号": "", "图7或图8折价≤-8%": "",
            "辅助买像": "是", "图9贴下轨(≤下轨×1.01)": "",
        },
    ]))
    text = "\n".join(_format_flat_buy_section(frame, tmp_path))
    assert "SH513080" in text
    assert "SH510300" in text
    assert "1.6541" in text
    assert "持仓和空仓都填" in text
    assert "| 代码 | 名称 | 类型 | 今日状态 | 买点价 | 买点状态 | T收盘 | 图9下轨 | 距图9下轨 |" in text
    assert "| SH513080 | 法国ETF华安 | 慢趋势 | 持仓 |" in text
    assert "| SH510300 | 沪深300ETF华泰柏瑞 | 长期下跌或周期 | 空仓 |" in text
    assert "## 空仓买点价（图9 因果P95下轨）" in text
    assert "## 空仓买点价（图9 EWMA下轨）" in text


def test_flat_buy_section_ewma_sorts_apart_from_p95(tmp_path):
    from etf_daily.scripts.scan_rules7_checklist import (
        _format_flat_buy_section,
        annotate_flat_buy_quotes,
    )

    def _row(code, name, lo, lo_ewma, dist, dist_ewma):
        return {
            "代码": code, "名称": name, "今日状态": "空仓", "今日动作": "观望",
            "类型": "慢趋势", "买规则": "aux_edge", "T收盘": "10.0000",
            "图9下轨": lo, "距图9下轨": dist,
            "图9 EWMA下轨": lo_ewma, "距图9 EWMA下轨": dist_ewma,
            "快形态(≥20%)": "", "买入信号": "", "图7或图8折价≤-8%": "",
            "辅助买像": "是", "图9贴下轨(≤下轨×1.01)": "",
            "图9近EWMA下轨(≤下轨×1.01)": "", "近EWMA下轨上跳沿": "",
        }

    frame = annotate_flat_buy_quotes(pd.DataFrame([
        _row("SH510300", "沪深300", "9.0000", "9.9000", "11.1%", "1.0%"),
        _row("SH510500", "中证500", "9.8000", "8.0000", "2.0%", "25.0%"),
        _row("SH511880", "银华日利", "9.5000", "", "5.3%", ""),
    ]))
    text = "\n".join(_format_flat_buy_section(frame, tmp_path))
    assert "## 空仓买点价（图9 EWMA下轨）" in text
    assert "| 代码 | 名称 | 类型 | 今日状态 | EWMA买点价 | EWMA买点状态 | T收盘 | 图9 EWMA下轨 | 距图9 EWMA下轨 |" in text
    p95, ewma = text.split("## 空仓买点价（图9 EWMA下轨）", 1)
    assert p95.index("SH510500") < p95.index("SH511880") < p95.index("SH510300")
    assert ewma.index("SH510300") < ewma.index("SH510500") < ewma.index("SH511880")
    assert "9.9990" in ewma
    assert "8.0800" in ewma
    assert "无下轨" in ewma
    assert "只作观察" in ewma


def test_jiu_march_2020_names_the_trail_and_the_next_buy():
    """酒 2020-03-17 is the 回落 signal; 03-18 is a new 买点 and that close is the fill."""
    idx = pd.to_datetime(["2020-03-12", "2020-03-13", "2020-03-16", "2020-03-17", "2020-03-18"])
    close = pd.Series([100.0, 100.0, 100.0, 88.0, 90.0], index=idx)
    buy = pd.Series([True, False, False, False, True], index=idx)
    sell = pd.Series(False, index=idx)
    res = run_state_machine(close, buy, sell, trail=0.12)
    trade = res["trades"][0]
    assert trade["why"] == "回落"
    assert trade["sell"] == idx[3]
    assert np.isclose(trade["ret"], 90.0 / 100.0 - 1)
    assert res["state"]["action"] == "买"
    assert res["state"]["buy_date"] == idx[4]
    assert res["state"]["why"] != "回落"


def test_hongli_march_2020_does_not_buy_on_the_trail_fill():
    """红利 03-16 is the 回落 signal. The next session is the fill and not a 买点. 03-18 is."""
    idx = pd.to_datetime(["2020-03-12", "2020-03-13", "2020-03-16", "2020-03-17", "2020-03-18"])
    close = pd.Series([100.0, 100.0, 92.0, 93.0, 94.0], index=idx)
    buy = pd.Series([True, False, False, False, True], index=idx)
    sell = pd.Series(False, index=idx)
    mid = run_state_machine(close.iloc[:4], buy.iloc[:4], sell.iloc[:4], trail=0.08)
    assert mid["trades"][0]["why"] == "回落"
    assert mid["trades"][0]["sell"] == idx[2]
    assert np.isclose(mid["trades"][0]["ret"], 93.0 / 100.0 - 1)
    assert mid["state"]["action"] == "观望"
    full = run_state_machine(close, buy, sell, trail=0.08)
    assert full["state"]["action"] == "买"
    assert full["state"]["buy_date"] == idx[4]


def test_exact_eight_percent_counts_and_a_hair_less_does_not():
    from etf_daily.lib.rules7 import build_masks

    hit = _band_frame(px=10.10, lower=10.0, gap7=-0.08, gap8=0.0)
    miss = _band_frame(px=10.10, lower=10.0, gap7=-0.079, gap8=0.0)
    off_rail = _band_frame(px=10.11, lower=10.0, gap7=-0.08, gap8=0.0)
    assert bool(build_masks(hit)["b1"].iloc[-1])
    assert not bool(build_masks(miss)["b1"].iloc[-1])
    assert not bool(build_masks(off_rail)["b1"].iloc[-1])


def test_already_inside_the_lower_rail_is_not_aux_edge():
    from etf_daily.lib.rules7 import build_masks

    frame = _band_frame(px=10.05, lower=10.0, gap7=0.0, gap8=0.0, n=2)
    frame["fig9_near_lo"] = True
    frame["aux_buy"] = [False, True]
    assert not bool(build_masks(frame)["ae_buy"].any())


def test_half_percent_under_the_upper_rail_is_touching_not_piercing():
    from etf_daily.lib.rules7 import build_masks

    frame = _g1_frame(px=19.90, upper=20.0)
    masks = build_masks(frame)
    snap = indicator_snapshot(
        frame, masks,
        RuleSpec(True, False, "fig9", True, 0.15),
        {"trail_hit": False, "sell_masked_by_g1": False, "stop": None},
    )
    assert snap["near_hi"] and snap["not_touch9hi"]
    assert bool(masks["g1"].iloc[-1])


def test_g1_on_the_buy_signal_does_not_latch():
    close = _series([10.0, 10.0, 10.0])
    res = run_state_machine(close, _mask(3, [2]), _mask(3, []), g1=_mask(3, [2]), trail=0.08)
    assert res["state"]["action"] == "买"
    assert res["state"]["g1_seen"] is False


def test_g1_on_the_fill_day_blocks_the_rail_and_is_not_a_trail():
    close = _series([10.0, 8.0])
    res = run_state_machine(close, _mask(2, [0]), _mask(2, [1]), g1=_mask(2, [1]), trail=0.08)
    assert res["trades"] == []
    assert res["state"]["action"] == "持有"
    assert res["state"]["g1_seen"] is True
    assert res["state"]["why"] != "回落"
    assert res["state"]["sell_masked_by_g1"] is True


def test_same_day_rail_and_trail_is_named_rail_sell():
    close = _series([10.0, 10.0, 8.0, 8.0])
    res = run_state_machine(close, _mask(4, [0]), _mask(4, [2]), trail=0.08)
    assert res["trades"][0]["why"] == "卖点"


def test_checklist_uses_the_study_words():
    from etf_daily.scripts.scan_rules7_checklist import HEAD_COLS, POS_COLS, SNAP_COLS

    names = set(HEAD_COLS) | set(POS_COLS) | {col for col, _ in SNAP_COLS}
    assert "信号日" in names and "成交日" in names
    assert "G1后不卖" not in names
    assert "本笔已进入G1" in names
    assert any("贴下轨" in col for col in names)
    assert any("贴上轨" in col for col in names)
    assert any("未碰上轨" in col for col in names)


def _band_frame(*, px: float, lower: float, gap7: float, gap8: float, n: int = 1) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=n)
    frame = pd.DataFrame(index=idx)
    frame["px"] = px
    frame["fig9_lower"] = lower
    frame["fig9_upper"] = lower * 1.2
    frame["fig8_upper"] = lower * 1.3
    frame["fig7_gap"] = gap7
    frame["fig8_gap"] = gap8
    frame["fig9_near_lo"] = frame["px"] <= frame["fig9_lower"] * 1.01
    frame["fig9_near_hi"] = False
    frame["aux_buy"] = False
    frame["aux_sell"] = False
    frame["fig9_mid_stretch_sell"] = False
    frame["locmin"] = False
    frame["locmax"] = False
    frame["fig9_pos"] = 0.5
    frame["fig9_touch_hi"] = False
    for k in (7, 8, 9, 10):
        frame[f"fig{k}_g"] = 0.0
    return frame


def _g1_frame(*, px: float, upper: float) -> pd.DataFrame:
    frame = _band_frame(px=px, lower=upper / 1.2, gap7=0.0, gap8=0.0, n=12)
    frame["fig9_upper"] = upper
    frame["px"] = px
    frame["fig9_touch_hi"] = False
    frame["fig9_pos"] = 0.4
    frame.iloc[-1, frame.columns.get_loc("fig9_pos")] = 0.9
    frame["fig7_g"] = 0.1
    frame["fig8_g"] = 0.1
    frame["fig9_g"] = 0.05
    frame["fig10_g"] = 0.2
    return frame
