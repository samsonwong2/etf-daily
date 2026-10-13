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
            "辅助买像": "是", "图9近下轨(≤下轨×1.01)": "",
        },
        {
            "代码": "SH510300", "名称": "沪深300ETF华泰柏瑞", "今日状态": "空仓", "今日动作": "观望",
            "类型": "长期下跌或周期", "买规则": "aux_edge 或 B1", "T收盘": "4.3850",
            "图9下轨": "4.1171", "距图9下轨": "6.5%",
            "快形态(≥20%)": "", "买入信号": "", "图7或图8折价≤-8%": "",
            "辅助买像": "是", "图9近下轨(≤下轨×1.01)": "",
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
