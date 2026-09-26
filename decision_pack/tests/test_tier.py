"""P0 tier ladder tests — 2026-06-05 style replay without qlib."""
from __future__ import annotations

from decision_pack.src.tier import MarketSnapshot, decide_tier, evaluate_table_tier


def _snapshot_20260605() -> MarketSnapshot:
    return MarketSnapshot(
        as_of="2026-06-05",
        benchmark="SH510300",
        r_bench=-0.018,
        bench_3d=-0.025,
        bench_5d=-0.01,
        close_below_ma20=False,
        vol_percentile_120d=0.65,
        dd_port=0.032,
    )


def test_table_tier_20260605_example():
    snap = _snapshot_20260605()
    table_tier, triggers = evaluate_table_tier(snap)
    assert table_tier == 1
    assert any("r_bench" in t for t in triggers)
    assert any("dd_port" in t for t in triggers)


def test_decide_tier_20260605_full_position():
    decision = decide_tier(_snapshot_20260605(), current_equity_exposure=1.0)
    assert decision.table_tier == 1
    assert decision.executed_tier == 1
    assert decision.regime == "sentiment_shock"
    assert decision.target_equity_exposure == 0.8
    assert abs(decision.reduce_fraction - 0.2) < 1e-9


def test_sentiment_downgrade_only_from_tier2():
    snap = MarketSnapshot(
        as_of="2026-06-05",
        benchmark="SH510300",
        r_bench=-0.03,
        bench_3d=-0.05,
        bench_5d=-0.04,
        close_below_ma20=True,
        vol_percentile_120d=0.85,
        dd_port=0.06,
    )
    decision = decide_tier(snap, current_equity_exposure=1.0)
    assert decision.table_tier == 2
    assert decision.regime == "structural_weakness"
    assert decision.executed_tier == 2
    assert decision.target_equity_exposure == 0.6


def test_sentiment_shock_downgrades_table_tier2_to_executed1():
    snap = MarketSnapshot(
        as_of="2026-06-05",
        benchmark="SH510300",
        r_bench=-0.03,
        bench_3d=-0.05,
        bench_5d=-0.01,
        close_below_ma20=False,
        vol_percentile_120d=0.50,
        dd_port=0.06,
    )
    decision = decide_tier(snap, current_equity_exposure=1.0)
    assert decision.table_tier == 2
    assert decision.regime == "sentiment_shock"
    assert decision.executed_tier == 1
    assert decision.target_equity_exposure == 0.8


def test_no_reduce_when_already_below_target():
    decision = decide_tier(_snapshot_20260605(), current_equity_exposure=0.75)
    assert decision.target_equity_exposure == 0.75
    assert decision.reduce_fraction == 0.0


def test_circuit_breaker_tier4():
    snap = MarketSnapshot(
        as_of="2026-06-05",
        benchmark="SH510300",
        r_bench=-0.01,
        bench_3d=-0.02,
        bench_5d=-0.02,
        close_below_ma20=False,
        vol_percentile_120d=0.40,
        dd_port=0.13,
    )
    decision = decide_tier(snap, current_equity_exposure=1.0)
    assert decision.table_tier == 4
    assert decision.target_equity_exposure == 0.27
