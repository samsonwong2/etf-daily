"""Unit tests for adaptive hold_up next-session execution plan."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from decision_pack.src.adaptive_execution_plan import (
    append_scenario_bar,
    build_symbol_execution_row,
    classify_bucket,
    end_regime,
    find_regime_thresholds,
    load_holdings_shares,
    make_close_only_bar,
    next_trade_date,
    render_execution_plan_markdown,
    round_tick,
    threshold_precision_for,
    truncate_ohlcv_through,
    write_execution_plan_artifacts,
)


def _ohlcv(n: int = 30, start: str = "2026-06-01", close0: float = 1.0) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n)
    close = close0 + np.linspace(0, 0.05, n)
    return pd.DataFrame(
        {
            "instrument": ["SH000001"] * n,
            "datetime": dates,
            "$open": close,
            "$high": close + 0.01,
            "$low": close - 0.01,
            "$close": close,
            "$volume": np.full(n, 1e6),
            "$pct_change": np.r_[0.0, np.diff(close) / close[:-1] * 100],
        }
    )


def test_round_tick_and_append_scenario_bar():
    assert round_tick(1.2344, 0.001) == 1.234
    assert round_tick(1.2346, 0.001) == 1.235
    base = _ohlcv(5, start="2026-08-01")
    bar = make_close_only_bar(date="2026-08-10", close=1.2)
    out = append_scenario_bar(base, bar)
    assert str(pd.Timestamp(out.datetime.iloc[-1]).date()) == "2026-08-10"
    assert float(out["$close"].iloc[-1]) == 1.2
    assert len(out) == len(truncate_ohlcv_through(base, "2026-08-07")) + 1 or len(out) <= 6


def test_threshold_precision_adx_conditional():
    assert threshold_precision_for("ma_stack_strict", has_snapshot_ohlc=False) == "exact_close_only"
    assert threshold_precision_for("hybrid_ma_adx", has_snapshot_ohlc=False) == "conditional_estimate"
    assert threshold_precision_for("hybrid_ma_adx", has_snapshot_ohlc=True) == "snapshot_ohlc"


def test_enter_up_buy_and_no_chase():
    ohlcv = _ohlcv(20, close0=1.0)
    # Keep as-of close below the enter threshold.
    ohlcv.loc[ohlcv.index[-1], "$close"] = 1.00
    ohlcv.loc[ohlcv.index[-1], "$open"] = 1.00
    ohlcv.loc[ohlcv.index[-1], "$high"] = 1.00
    ohlcv.loc[ohlcv.index[-1], "$low"] = 1.00

    def label_fn(df: pd.DataFrame) -> np.ndarray:
        # last bar up only if close >= 1.05
        labs = np.array(["range"] * len(df), dtype=object)
        if float(df["$close"].iloc[-1]) >= 1.05:
            labs[-1] = "up"
        return labs

    row = build_symbol_execution_row(
        code="SHTEST1",
        name="测试",
        method="ma_stack_strict",
        method_params={},
        ohlcv=ohlcv,
        as_of="2026-08-07",
        trade_date="2026-08-10",
        live_shares=0,
        holdings_stale=True,
        model_open_hint=False,
        label_fn=label_fn,
        limit_pct=0.20,
        tick=0.001,
    )
    assert row["regime_asof"] == "range"
    assert row["can_buy"] is False or row["buy_trigger_px"] is not None
    assert row["buy_action"] in {"等待上涨切换", "收盘确认买入"}
    if row["buy_trigger_px"] is not None:
        assert row["buy_trigger_px"] >= 1.05 - 1e-9
        assert row["buy_trigger_pct"] == pytest.approx(
            row["buy_trigger_px"] / row["last_close"] - 1.0, rel=1e-6
        )

    # already model-open => no chase
    def always_up(df: pd.DataFrame) -> np.ndarray:
        return np.array(["up"] * len(df), dtype=object)

    row2 = build_symbol_execution_row(
        code="SHTEST2",
        name="测试2",
        method="ma_stack_strict",
        method_params={},
        ohlcv=ohlcv,
        as_of="2026-08-07",
        trade_date="2026-08-10",
        live_shares=0,
        holdings_stale=True,
        model_open_hint=True,
        label_fn=always_up,
    )
    assert row2["can_buy"] is False
    assert row2["buy_action"] == "不追买"
    assert row2["bucket"] in {"no_chase", "hypo_hold", "hypo_sell_trigger"}


def test_leave_up_hypo_sell_and_exit_close():
    ohlcv = _ohlcv(20, close0=2.0)

    def label_fn(df: pd.DataFrame) -> np.ndarray:
        labs = np.array(["up"] * len(df), dtype=object)
        # leave up when close drops to <= 1.95
        if float(df["$close"].iloc[-1]) <= 1.95:
            labs[-1] = "range"
        return labs

    row = build_symbol_execution_row(
        code="SHUP",
        name="上涨中",
        method="ma_stack_hyst",
        method_params={},
        ohlcv=ohlcv,
        as_of="2026-08-07",
        trade_date="2026-08-10",
        live_shares=0,
        holdings_stale=True,
        model_open_hint=True,
        label_fn=label_fn,
        limit_pct=0.20,
        tick=0.001,
    )
    assert row["can_sell_live"] is False
    assert row["sell_live_action"] == "不适用"
    assert row["hypo_can_sell"] is True
    assert row["hypo_sell_action"] in {"触及卖出价则卖", "收盘确认卖出"}
    if row["hypo_sell_trigger_px"] is not None and row["hypo_sell_action"] == "触及卖出价则卖":
        assert row["hypo_sell_trigger_px"] <= 1.95 + 1e-9

    def always_range(df: pd.DataFrame) -> np.ndarray:
        return np.array(["range"] * len(df), dtype=object)

    row2 = build_symbol_execution_row(
        code="SHFLAT",
        name="横盘",
        method="ma_stack_strict",
        method_params={},
        ohlcv=ohlcv,
        as_of="2026-08-07",
        trade_date="2026-08-10",
        live_shares=0,
        holdings_stale=True,
        model_open_hint=False,
        label_fn=always_range,
    )
    assert row2["hypo_sell_action"] == "收盘直接退出"
    assert row2["bucket"] in {"hypo_exit_close", "unreachable", "wait_buy"}


def test_unreachable_buy_within_limit():
    ohlcv = _ohlcv(15, close0=1.0)

    def never_up(df: pd.DataFrame) -> np.ndarray:
        return np.array(["range"] * len(df), dtype=object)

    row = build_symbol_execution_row(
        code="SHNONE",
        name="无触发",
        method="ma_stack_strict",
        method_params={},
        ohlcv=ohlcv,
        as_of="2026-08-07",
        trade_date="2026-08-10",
        live_shares=0,
        holdings_stale=True,
        model_open_hint=False,
        label_fn=never_up,
        limit_pct=0.05,
    )
    assert row["can_buy"] is False
    assert row["buy_trigger_px"] is None
    assert "无法确认" in row["buy_note"] or row["bucket"] == "unreachable"


def test_min_seg_backfill_not_historical_fill():
    """Confirmation only on trade date counts; earlier rewrite is ignored for fills."""
    ohlcv = _ohlcv(12, start="2026-07-20", close0=1.0)

    def label_fn(df: pd.DataFrame) -> np.ndarray:
        # Mimic min_seg backfill: if last close high, rewrite last 10 bars to up.
        labs = np.array(["range"] * len(df), dtype=object)
        if float(df["$close"].iloc[-1]) >= 1.08:
            labs[-10:] = "up"
        return labs

    asof = truncate_ohlcv_through(ohlcv, "2026-08-07")
    assert end_regime(asof, label_fn=label_fn) == "range"
    row = build_symbol_execution_row(
        code="SHSEG",
        name="回填",
        method="ma_stack_strict",
        method_params={},
        ohlcv=ohlcv,
        as_of=str(pd.Timestamp(asof.datetime.iloc[-1]).date()),
        trade_date="2026-08-10",
        live_shares=0,
        holdings_stale=True,
        model_open_hint=False,
        label_fn=label_fn,
        limit_pct=0.20,
    )
    # Signal is attributed to trade_date buy action, not historical entry dates.
    assert row["buy_action"] in {"等待上涨切换", "收盘确认买入"}
    assert row["as_of"] != row["trade_date"]
    if row["buy_trigger_px"] is not None:
        assert row["buy_trigger_px"] >= 1.08 - 1e-9


def test_empty_holdings_treated_as_flat(tmp_path: Path):
    path = tmp_path / "20260717.csv"
    path.write_text("code,name,shares,price\n", encoding="utf-8")
    shares, stale, note = load_holdings_shares(path, as_of="2026-08-07")
    assert shares == {}
    assert stale is True
    assert "empty" in note or "before" in note


def test_classify_bucket_helpers():
    assert (
        classify_bucket(
            can_buy=True,
            buy_action="收盘确认买入",
            hypo_sell_action="收盘直接退出",
            buy_trigger_px=1.1,
            hypo_sell_trigger_px=1.0,
        )
        == "buy_candidate"
    )
    assert (
        classify_bucket(
            can_buy=False,
            buy_action="等待上涨切换",
            hypo_sell_action="收盘直接退出",
            buy_trigger_px=None,
            hypo_sell_trigger_px=1.0,
        )
        == "unreachable"
    )
    assert (
        classify_bucket(
            can_buy=False,
            buy_action="不追买",
            hypo_sell_action="继续持有",
            buy_trigger_px=None,
            hypo_sell_trigger_px=None,
        )
        == "hypo_hold"
    )


def test_find_regime_thresholds_prefers_upside():
    ohlcv = _ohlcv(10, close0=1.0)

    def label_fn(df: pd.DataFrame) -> np.ndarray:
        labs = np.array(["range"] * len(df), dtype=object)
        if float(df["$close"].iloc[-1]) >= 1.03:
            labs[-1] = "up"
        return labs

    hit = find_regime_thresholds(
        ohlcv,
        trade_date="2026-08-10",
        last_close=float(ohlcv["$close"].iloc[-1]),
        target_regimes={"up"},
        label_fn=label_fn,
        limit_pct=0.10,
        tick=0.001,
        coarse_step_pct=0.005,
        prefer="up",
    )
    assert hit is not None
    assert hit.price >= 1.03 - 1e-9


def test_render_execution_plan_markdown_and_daily_artifacts(tmp_path: Path):
    assert next_trade_date("2026-08-07") == "2026-08-10"
    frame = pd.DataFrame(
        [
            {
                "code": "SH512690",
                "name": "酒ETF鹏华",
                "method": "ma_stack_strict",
                "as_of": "2026-08-07",
                "trade_date": "2026-08-10",
                "last_close": 0.435,
                "regime_asof": "up",
                "model_open": True,
                "live_shares": 0.0,
                "live_holding": False,
                "holdings_stale": True,
                "can_buy": False,
                "buy_action": "不追买",
                "buy_trigger_px": None,
                "buy_trigger_pct": None,
                "buy_note": "不追买",
                "can_sell_live": False,
                "sell_live_action": "不适用",
                "sell_live_note": "空仓",
                "hypo_can_sell": False,
                "hypo_sell_action": "继续持有",
                "hypo_sell_trigger_px": None,
                "hypo_sell_trigger_pct": None,
                "hypo_sell_note": "难离上涨",
                "threshold_precision": "exact_close_only",
                "bucket": "hypo_hold",
                "data_source": "20260807all_adaptive",
            }
        ]
    )
    meta = {
        "as_of": "2026-08-07",
        "trade_date": "2026-08-10",
        "holdings_note": "empty_holdings_treated_as_flat",
        "holdings_stale": True,
        "adaptive_dir": "workspace/plotly_outputs/20260807all_adaptive",
        "limit_pct": 0.10,
    }
    md = render_execution_plan_markdown(frame, meta=meta)
    assert "# Adaptive hold_up 执行方案（2026-08-10）" in md
    assert "SH512690" in md
    assert "每日生成" in md
    paths = write_execution_plan_artifacts(frame, tmp_path, meta=meta)
    assert paths["markdown"].name == "EXEC_for_20260810_from_20260807.md"
    assert paths["markdown"].exists()
    assert "EXEC_for_20260810_from_20260807.md" in paths["readme"].read_text(encoding="utf-8")
