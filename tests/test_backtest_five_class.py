import numpy as np
import pandas as pd
import pytest

from etf_daily.scripts.backtest_triangle_five_class_inception import (
    GateConfig,
    _bucket_row,
    apply_gates,
    backtest_code,
    segment_stats,
    summarize_by_bucket,
    write_report,
)


def _make_ohlcv(n: int = 320, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    rets = rng.normal(0.0003, 0.013, size=n)
    close = 1.0 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    return pd.DataFrame(
        {
            "datetime": idx,
            "instrument": "SH000300",
            "$open": open_,
            "$high": high,
            "$low": low,
            "$close": close,
            "$volume": rng.integers(1e6, 5e6, n),
        }
    )


class _FakePlotMod:
    def __init__(self, ohlcv):
        self._ohlcv = ohlcv

    def load_qlib_ohlcv(self, code, start, end, **kw):
        return self._ohlcv.copy()

    @staticmethod
    def attach_sma_columns(df):
        df = df.copy()
        df["MA5"] = df["$close"].rolling(5).mean()
        df["MA20"] = df["$close"].rolling(20).mean()
        return df


def test_bucket_row_maps_five_class():
    assert _bucket_row("买观察", True)[0] == "buy"
    assert _bucket_row("卖预警", True)[0] == "sell"
    assert _bucket_row("趋势内交替观察", True)[0] == "up"
    assert _bucket_row("红三角不抄底", False)[0] == "down"
    assert _bucket_row("底部交替观察", None)[0] == "range"
    assert _bucket_row("底部交替观察", False)[0] == "down"
    assert _bucket_row("底部交替观察", True)[0] == "up"


def test_backtest_code_produces_events_and_buckets():
    ohlcv = _make_ohlcv()
    plot_mod = _FakePlotMod(ohlcv)
    res = backtest_code(
        "SH000300",
        plot_mod,
        __import__(
            "etf_daily.scripts.backtest_triangle_decision_rules",
            fromlist=["RuleConfig"],
        ).RuleConfig(),
        start_date="2024-01-02",
        end_date="2025-04-30",
        refit_days=63,
        min_warmup_days=120,
    )
    assert res["code"] == "SH000300"
    assert res["n_days"] > 0
    assert "events" in res and "zones" in res
    if res["events"]:
        ev = pd.DataFrame(res["events"])
        assert {"bucket", "bucket_label", "action", "edge"}.issubset(ev.columns)
        assert ev["bucket"].isin(["buy", "sell", "up", "down", "range"]).all()


def test_summarize_by_bucket_groups_five_classes():
    ev = pd.DataFrame(
        [
            {"bucket": "buy", "bucket_label": "潜在买入点", "executed": True, "edge": 0.02, "fwd_ret": 0.01, "fwd_mae": -0.01, "fwd_mfe": 0.03},
            {"bucket": "buy", "bucket_label": "潜在买入点", "executed": True, "edge": -0.01, "fwd_ret": -0.01, "fwd_mae": -0.02, "fwd_mfe": 0.01},
            {"bucket": "sell", "bucket_label": "潜在卖出点", "executed": True, "edge": 0.03, "fwd_ret": -0.03, "fwd_mae": -0.05, "fwd_mfe": 0.01},
            {"bucket": "up", "bucket_label": "趋势上涨中", "executed": False, "edge": np.nan, "fwd_ret": 0.04, "fwd_mae": -0.01, "fwd_mfe": 0.06},
        ]
    )
    s = summarize_by_bucket(ev)
    assert set(s["bucket"]) == {"buy", "sell", "up"}
    buy = s[s["bucket"] == "buy"].iloc[0]
    assert buy["n_signals"] == 2 and buy["n_executed"] == 2
    assert buy["win_rate"] == 0.5


def test_write_report_sections(tmp_path):
    meta = {
        "start": "inception",
        "end": "2026-07-28",
        "n_ok": 1,
        "n_failed": 0,
        "refit_days": 63,
        "min_warmup_days": 120,
    }
    by_action = pd.DataFrame(
        [{"action": "买观察", "n_signals": 2, "n_executed": 1, "exec_rate": 0.5, "win_rate": 1.0, "mean_edge": 0.02, "median_edge": 0.02}]
    )
    by_bucket = pd.DataFrame(
        [{"bucket": "buy", "bucket_label": "潜在买入点", "n_signals": 2, "n_executed": 1, "exec_rate": 0.5, "win_rate": 1.0, "mean_edge": 0.02, "median_edge": 0.02, "mean_fwd20_ret": 0.01, "mean_fwd20_mae": -0.01, "mean_fwd20_mfe": 0.03}]
    )
    zones = pd.DataFrame(
        [{"kind": "顶部交替区", "avoided_drawdown": 0.05}]
    )
    write_report(tmp_path, meta, by_action, by_bucket, zones)
    txt = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "Inception 5-class OPS backtest" in txt
    assert "Five-class buy/sell accuracy" in txt
    assert "Trend/range diagnostics" in txt
    assert "潜在买入点" in txt
    assert "synthetic walk-forward HMM" in txt


def _gate_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # 卖预警: 中 strength passes, 强 fails
            {"action": "卖预警", "strength": "中", "vol_pct": 0.85, "prior10": 0.08, "executed": True, "edge": 0.02, "signal_date": "2023-06-01"},
            {"action": "卖预警", "strength": "强", "vol_pct": 0.95, "prior10": 0.12, "executed": True, "edge": -0.03, "signal_date": "2024-06-01"},
            # 买观察: vol>=0.80 passes
            {"action": "买观察", "strength": "中", "vol_pct": 0.85, "prior10": -0.06, "executed": True, "edge": 0.01, "signal_date": "2023-06-01"},
            {"action": "买观察", "strength": "弱", "vol_pct": 0.72, "prior10": -0.06, "executed": True, "edge": -0.01, "signal_date": "2024-06-01"},
            # 顶区收复: deep or positive passes, middle fails
            {"action": "顶区失效-收复买观察", "strength": "中", "vol_pct": 0.85, "prior10": -0.09, "executed": True, "edge": 0.03, "signal_date": "2023-06-01"},
            {"action": "顶区失效-收复买观察", "strength": "中", "vol_pct": 0.85, "prior10": -0.04, "executed": True, "edge": -0.02, "signal_date": "2024-06-01"},
            # 买确认候选 always passes
            {"action": "买确认候选", "strength": "中", "vol_pct": 0.85, "prior10": -0.09, "executed": True, "edge": 0.04, "signal_date": "2024-06-01"},
        ]
    )


def test_apply_gates_flags():
    ev = apply_gates(_gate_events(), GateConfig())
    assert len(ev) == 7
    g = ev.set_index(["action", "strength", "prior10"]) if False else ev
    sell_mid = ev[(ev["action"] == "卖预警") & (ev["strength"] == "中")].iloc[0]
    sell_strong = ev[(ev["action"] == "卖预警") & (ev["strength"] == "强")].iloc[0]
    assert bool(sell_mid["gated"]) is True
    assert bool(sell_strong["gated"]) is False
    buy_hi = ev[(ev["action"] == "买观察") & (ev["vol_pct"] > 0.8)].iloc[0]
    buy_lo = ev[(ev["action"] == "买观察") & (ev["vol_pct"] < 0.8)].iloc[0]
    assert bool(buy_hi["gated"]) is True
    assert bool(buy_lo["gated"]) is False
    pt_deep = ev[(ev["action"] == "顶区失效-收复买观察") & (ev["prior10"] < -0.08)].iloc[0]
    pt_mid = ev[(ev["action"] == "顶区失效-收复买观察") & (ev["prior10"] > -0.08)].iloc[0]
    assert bool(pt_deep["gated"]) is True
    assert bool(pt_mid["gated"]) is False
    bc = ev[ev["action"] == "买确认候选"].iloc[0]
    assert bool(bc["gated"]) is True


def test_segment_stats_train_test():
    ev = apply_gates(_gate_events(), GateConfig())
    ss = segment_stats(ev, "2024-01-01")
    assert {"train", "test"} == set(ss["segment"])
    sell_train_base = ss[
        (ss["action"] == "卖预警") & (ss["segment"] == "train") & (ss["gate"] == "baseline")
    ].iloc[0]
    assert sell_train_base["n_exec"] == 1
    sell_test_gated = ss[
        (ss["action"] == "卖预警") & (ss["segment"] == "test") & (ss["gate"] == "gated")
    ].iloc[0]
    assert sell_test_gated["n_exec"] == 0  # strong sell filtered out in test


def test_apply_gates_empty():
    ev = apply_gates(pd.DataFrame(), GateConfig())
    assert ev.empty
    assert segment_stats(ev, "2024-01-01").empty
