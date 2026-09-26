"""P3 recovery state machine."""
from __future__ import annotations

from decision_pack.src.recovery import RecoveryState, update_recovery_state
from decision_pack.src.tier import MarketSnapshot, TierDecision


def _market() -> MarketSnapshot:
    return MarketSnapshot(
        as_of="2026-06-05",
        benchmark="SH510300",
        r_bench=-0.01,
        bench_3d=-0.01,
        bench_5d=-0.01,
        close_below_ma20=False,
        vol_percentile_120d=0.4,
        dd_port=0.01,
    )


def _decision_tier1() -> TierDecision:
    return TierDecision(
        as_of="2026-06-05",
        benchmark="SH510300",
        r_bench=-0.01,
        dd_port=0.01,
        vol_percentile_120d=0.4,
        table_tier=1,
        regime="sentiment_shock",
        executed_tier=1,
        current_equity_exposure=0.8,
        target_equity_exposure=0.8,
        reduce_fraction=0.0,
        triggers=(),
        structural_rule_hits=(),
    )


def test_recovery_increments_calm_days():
    state = RecoveryState(last_updated="2026-06-04", days_without_tier_ge_2=5, peak_tier=1)
    update = update_recovery_state(state, snapshot=_market(), decision=_decision_tier1(), exposure_scale=1.0)
    assert update.state.days_without_tier_ge_2 == 6


def test_recovery_resets_on_tier2():
    state = RecoveryState(last_updated="2026-06-04", days_without_tier_ge_2=9, peak_tier=1)
    decision = TierDecision(
        as_of="2026-06-05",
        benchmark="SH510300",
        r_bench=-0.03,
        dd_port=0.06,
        vol_percentile_120d=0.5,
        table_tier=2,
        regime="sentiment_shock",
        executed_tier=1,
        current_equity_exposure=1.0,
        target_equity_exposure=0.8,
        reduce_fraction=0.2,
        triggers=("dd_port>=0.05",),
        structural_rule_hits=(),
    )
    update = update_recovery_state(state, snapshot=_market(), decision=decision, exposure_scale=1.0)
    assert update.state.days_without_tier_ge_2 == 0
