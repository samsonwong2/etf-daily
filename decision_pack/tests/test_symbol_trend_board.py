"""v0 trend board: causal metrics and equity sleeve filtering."""
from __future__ import annotations

import math

import pandas as pd

from decision_pack.src.config import load_decision_pack_config
from decision_pack.src.indicators import (
    compute_trend_metrics,
    days_below_ma,
    ma_stack_label,
    vol_level_label,
)
from decision_pack.src.orders import build_display_only_orders
from decision_pack.src.portfolio import HoldingRow, PortfolioSnapshot
from decision_pack.src.render import _trend_row_mark
from decision_pack.src.symbol_trend_board import (
    TREND_BOARD_COLUMNS,
    build_cluster_trend_board,
    build_risk_flags,
    build_trend_board,
    load_cluster_mapping_codes,
)
from pipeline.price_adjustments import auto_qfq_adjust_close_panel


def _rising_close(days: int = 180, start: float = 1.0, step: float = 0.01) -> pd.Series:
    index = pd.bdate_range("2025-01-02", periods=days)
    values = [start + step * i for i in range(days)]
    return pd.Series(values, index=index, dtype=float)


def _portfolio_with_bond() -> PortfolioSnapshot:
    as_of = _rising_close().index[-1].strftime("%Y-%m-%d")
    holdings = (
        HoldingRow("SH510300", "HS300", 50000, 0.5, 0.5, "live", cost_price=1.0),
        HoldingRow("SH511260", "BOND", 20000, 0.2, 0.2, "live"),
        HoldingRow("SH512100", "ZZ1000", 30000, 0.3, 0.3, "live", cost_price=1.5),
    )
    return PortfolioSnapshot(
        as_of=as_of,
        source="live",
        holdings=holdings,
        total_assets=100000.0,
        cash_mv=0.0,
        stock_mv=100000.0,
        equity_exposure=0.8,
    )


def test_ma_stack_labels():
    assert ma_stack_label(12.0, 11.0, 10.0) == "bull"
    assert ma_stack_label(10.0, 11.0, 12.0) == "bear"
    assert ma_stack_label(11.0, 10.0, 12.0) == "mixed"


def test_vol_level_label_buckets():
    assert vol_level_label(0.20) == "low"
    assert vol_level_label(0.50) == "mid"
    assert vol_level_label(0.95) == "high"
    assert vol_level_label(None) is None


def test_compute_trend_metrics_bullish_series():
    close = _rising_close()
    end = close.index[-1]
    metrics = compute_trend_metrics(close, end, vol_window=20, vol_lookback=120)
    assert metrics["ma_stack"] == "bull"
    assert metrics["close"] is not None and metrics["ma20"] is not None
    assert metrics["dist_ma20_pct"] is not None and metrics["dist_ma20_pct"] > 0
    assert metrics["ret_5d"] is not None and metrics["ret_5d"] > 0
    assert metrics["vol_ann_20d"] is not None and metrics["vol_ann_20d"] >= 0
    assert metrics["vol_ann_5d"] is not None and metrics["vol_ann_5d"] >= 0
    assert metrics["vol_ratio_5_20"] is not None and metrics["vol_ratio_5_20"] > 0


def test_vol_ratio_5_20_is_ratio_of_ann_vols():
    close = _rising_close()
    end = close.index[-1]
    metrics = compute_trend_metrics(close, end, vol_window=20, vol_lookback=120)
    assert metrics["vol_ann_5d"] is not None and metrics["vol_ann_20d"] is not None
    assert metrics["vol_ratio_5_20"] is not None
    expected = metrics["vol_ann_5d"] / metrics["vol_ann_20d"]
    assert abs(metrics["vol_ratio_5_20"] - expected) < 1e-9


def test_load_cluster_mapping_codes_reads_first_column(tmp_path):
    mapping = tmp_path / "cluster_mapping_selected.txt"
    mapping.write_text(
        "SH513080\t2020-06-12\t2026-06-08\n"
        "SH515050\t2019-10-16\t2026-06-08\n"
        "\n"
        "# comment\n",
        encoding="utf-8",
    )
    assert load_cluster_mapping_codes(mapping) == ("SH513080", "SH515050")


def test_build_cluster_trend_board_includes_unheld_symbols(tmp_path):
    cfg = load_decision_pack_config()
    close = _rising_close()
    panel = pd.DataFrame(
        {
            "SH510300": close,
            "SH512100": close * 0.95,
            "SH999999": close * 1.02,
        },
        index=close.index,
    )
    mapping = tmp_path / "cluster_mapping_selected.txt"
    mapping.write_text(
        "SH510300\t2020-01-01\t2099-01-01\n"
        "SH512100\t2020-01-01\t2099-01-01\n"
        "SH999999\t2020-01-01\t2099-01-01\n",
        encoding="utf-8",
    )
    result = build_cluster_trend_board(
        _portfolio_with_bond(),
        mapping,
        config=cfg,
        close_panel=panel,
        benchmark_close=close * 0.98,
    )
    board = result.board
    assert list(board.columns) == TREND_BOARD_COLUMNS
    assert set(board["code"]) == {"SH510300", "SH512100", "SH999999"}
    assert "SH511260" not in board["code"].tolist()
    unheld = board.loc[board["code"] == "SH999999"].iloc[0]
    assert unheld["weight"] == 0.0
    assert unheld["pnl_pct"] is None or pd.isna(unheld["pnl_pct"])
    assert unheld["close"] is not None


def test_build_trend_board_excludes_defensive_codes():
    cfg = load_decision_pack_config()
    close = _rising_close()
    panel = pd.DataFrame(
        {
            "SH510300": close,
            "SH512100": close * 0.95,
        },
        index=close.index,
    )
    bench = close * 0.98
    result = build_trend_board(
        _portfolio_with_bond(),
        config=cfg,
        close_panel=panel,
        benchmark_close=bench,
    )
    board = result.board
    assert list(board.columns) == TREND_BOARD_COLUMNS
    assert "SH511260" not in board["code"].tolist()
    assert len(board) == 2
    assert result.vol_context.equity_count == 2
    dist = board["dist_ma20_pct"].tolist()
    assert dist == sorted(dist, key=lambda x: (x is None, x if x is not None else math.inf))
    hs300 = board.loc[board["code"] == "SH510300"].iloc[0]
    assert hs300["pnl_pct"] is not None and hs300["pnl_pct"] > 0
    assert hs300["days_below_ma20"] == 0
    assert hs300["rs_20d_vs_bench"] is not None
    assert hs300["vol_level"] in {"low", "mid", "high"}
    assert isinstance(hs300["risk_flags"], str)


def test_days_below_ma_counts_consecutive_breaks():
    index = pd.bdate_range("2025-01-02", periods=30)
    close = pd.Series(10.0, index=index, dtype=float)
    close.iloc[-3:] = 9.0
    end = index[-1]
    assert days_below_ma(close, end, 20) == 3


def test_build_risk_flags_composite():
    flags = build_risk_flags(
        dist_ma20_pct=-6.0,
        vol_pct_120d=0.85,
        weight=0.12,
        pnl_pct=-2.0,
        ma_stack="bear",
        rs_20d_vs_bench=-0.08,
        days_below_ma20=4,
        thresholds={
            "heavy_weight": 0.10,
            "high_vol_pct": 0.80,
            "deep_below_ma20_pct": -5.0,
            "weak_rs_20d": -0.05,
            "stale_below_ma20_days": 3,
            "red_pnl_pct": -7.0,
            "red_dd_20d_high": -0.05,
            "vol_accel_ratio": 1.2,
        },
    )
    for tag in (
        "below_ma20",
        "deep_below_ma20",
        "below_ma20_3d+",
        "high_vol",
        "heavy_wt",
        "loss",
        "bear_stack",
        "weak_rs",
    ):
        assert tag in flags.split("|")


def test_build_risk_flags_pnl7_and_dd5():
    flags = build_risk_flags(
        dist_ma20_pct=-2.0,
        vol_pct_120d=0.50,
        weight=0.05,
        pnl_pct=-8.0,
        ma_stack="mixed",
        rs_20d_vs_bench=0.0,
        days_below_ma20=1,
        dd_20d_high=-0.06,
        ret_5d=-0.02,
        ret_20d=0.01,
        vol_ratio_5_20=1.5,
        thresholds={
            "heavy_weight": 0.10,
            "high_vol_pct": 0.80,
            "deep_below_ma20_pct": -5.0,
            "weak_rs_20d": -0.05,
            "stale_below_ma20_days": 3,
            "red_pnl_pct": -7.0,
            "red_dd_20d_high": -0.05,
            "vol_accel_ratio": 1.2,
        },
    )
    parts = flags.split("|")
    assert "pnl7" in parts
    assert "dd5_20d" in parts
    assert "mom_fade" in parts
    assert "vol_accel" in parts


def test_trend_row_mark_below_ma20_and_high_vol():
    row = pd.Series(
        {
            "dist_ma20_pct": -1.0,
            "risk_flags": "below_ma20|high_vol|loss",
        }
    )
    assert _trend_row_mark(row) == "⚠⚠ "


def test_qfq_smooths_ex_dividend_jump_for_returns():
    index = pd.bdate_range("2025-01-02", periods=40)
    raw = pd.Series(2.0, index=index, dtype=float)
    ex_day = index[20]
    raw.loc[ex_day:] = 1.0  # 50% ex-dividend jump on raw close

    end = index[-1]
    raw_metrics = compute_trend_metrics(raw, end, vol_window=20, vol_lookback=30)
    adjusted, records = auto_qfq_adjust_close_panel(raw.to_frame(name="SH515050"))
    assert not records.empty
    qfq_metrics = compute_trend_metrics(adjusted["SH515050"], end, vol_window=20, vol_lookback=30)

    assert raw_metrics["ret_20d"] is not None and raw_metrics["ret_20d"] < -0.4
    assert qfq_metrics["ret_20d"] is not None and abs(qfq_metrics["ret_20d"]) < 0.05


def test_display_only_orders_hold_only():
    orders = build_display_only_orders(_portfolio_with_bond())
    assert (orders["action"] == "HOLD").all()
    assert orders["sell_mv"].sum() == 0.0
    assert (orders["reason"] == "display_only_v0").all()
