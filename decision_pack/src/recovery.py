"""T+N recovery state machine."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from decision_pack.src.config import DecisionPackConfig, load_decision_pack_config
from decision_pack.src.inputs import DEFAULT_STATE_PATH
from decision_pack.src.tier import MarketSnapshot, TierDecision


@dataclass
class RecoveryState:
    last_updated: str
    peak_tier: int = 0
    days_without_tier_ge_2: int = 0
    last_addback_date: str | None = None
    quarter_tier2_manual_count: int = 0
    bench_5d_low_watermark: float | None = None
    quarter_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RecoveryState:
        return cls(
            last_updated=str(payload.get("last_updated", "")),
            peak_tier=int(payload.get("peak_tier", 0)),
            days_without_tier_ge_2=int(payload.get("days_without_tier_ge_2", 0)),
            last_addback_date=payload.get("last_addback_date"),
            quarter_tier2_manual_count=int(payload.get("quarter_tier2_manual_count", 0)),
            bench_5d_low_watermark=payload.get("bench_5d_low_watermark"),
            quarter_key=payload.get("quarter_key"),
        )


@dataclass(frozen=True)
class RecoveryUpdate:
    state: RecoveryState
    recovery_signal: bool
    addback_fraction: float
    reason: str
    watch_items: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self.state)
        payload.update(
            {
                "recovery_signal": self.recovery_signal,
                "addback_fraction": self.addback_fraction,
                "reason": self.reason,
                "watch_items": list(self.watch_items),
            }
        )
        return payload


def _quarter_key(report_date: str) -> str:
    ts = datetime.strptime(report_date, "%Y-%m-%d")
    quarter = (ts.month - 1) // 3 + 1
    return f"{ts.year}Q{quarter}"


def _recovery_rules(config: DecisionPackConfig) -> dict[str, Any]:
    return config.raw.get("recovery") or {}


def _l3_gap_rules(config: DecisionPackConfig) -> dict[str, Any]:
    return config.raw.get("l3_gap") or {}


def load_recovery_state(path: Path | None = None) -> RecoveryState:
    state_path = path or DEFAULT_STATE_PATH
    if not state_path.exists():
        return RecoveryState(last_updated="")
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid recovery state JSON: {state_path}")
    return RecoveryState.from_dict(payload)


def save_recovery_state(state: RecoveryState, path: Path | None = None) -> Path:
    state_path = path or DEFAULT_STATE_PATH
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(state.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return state_path


def _trading_days_since(last_date: str | None, current_date: str) -> int | None:
    if not last_date:
        return None
    last = datetime.strptime(last_date, "%Y-%m-%d").date()
    current = datetime.strptime(current_date, "%Y-%m-%d").date()
    return max(0, (current - last).days)


def update_recovery_state(
    previous: RecoveryState,
    *,
    snapshot: MarketSnapshot,
    decision: TierDecision,
    exposure_scale: float | None,
    config: DecisionPackConfig | None = None,
) -> RecoveryUpdate:
    cfg = config or load_decision_pack_config()
    rules = _recovery_rules(cfg)
    l3_rules = _l3_gap_rules(cfg)

    state = RecoveryState.from_dict(previous.to_dict())
    state.last_updated = snapshot.as_of
    state.peak_tier = max(state.peak_tier, decision.executed_tier)

    quarter = _quarter_key(snapshot.as_of)
    if state.quarter_key != quarter:
        state.quarter_key = quarter
        state.quarter_tier2_manual_count = 0

    exposure = exposure_scale if exposure_scale is not None else 1.0
    exposure_threshold = float(l3_rules.get("exposure_scale_threshold", 0.99))
    manual_threshold = int(l3_rules.get("manual_tier_threshold", 2))
    if decision.executed_tier >= manual_threshold and exposure >= exposure_threshold:
        state.quarter_tier2_manual_count += 1

    effective_tier = max(decision.table_tier, decision.executed_tier)
    if effective_tier >= 2:
        state.days_without_tier_ge_2 = 0
    else:
        state.days_without_tier_ge_2 += 1

    if state.bench_5d_low_watermark is None:
        state.bench_5d_low_watermark = snapshot.bench_5d
    else:
        state.bench_5d_low_watermark = min(state.bench_5d_low_watermark, snapshot.bench_5d)

    days_needed = int(rules.get("days_without_tier_ge_2", 10))
    max_addback = float(rules.get("max_addback_per_step", 0.10))
    min_gap_days = int(rules.get("min_days_between_addback", 3))
    require_ma = bool(rules.get("require_close_above_ma20", True))
    tier34_guard = bool(rules.get("tier34_require_bench_5d_not_new_low", True))

    watch: list[str] = [
        f"days_without_tier_ge_2={state.days_without_tier_ge_2}/{days_needed}",
        f"peak_tier={state.peak_tier}",
        f"quarter_tier2_manual_count={state.quarter_tier2_manual_count}",
    ]

    recovery_signal = False
    addback_fraction = 0.0
    reason = "no_recovery"

    from_tier1_path = state.peak_tier <= 1 and decision.executed_tier <= 1
    days_since_addback = _trading_days_since(state.last_addback_date, snapshot.as_of)
    ma_ok = (not require_ma) or (not snapshot.close_below_ma20)
    spacing_ok = days_since_addback is None or days_since_addback >= min_gap_days
    calm_enough = state.days_without_tier_ge_2 >= days_needed

    tier34_ok = True
    if tier34_guard and state.peak_tier >= 3:
        tier34_ok = snapshot.bench_5d > (state.bench_5d_low_watermark or snapshot.bench_5d)
        watch.append(f"bench_5d_not_new_low={tier34_ok}")

    if from_tier1_path and calm_enough and ma_ok and spacing_ok and tier34_ok:
        recovery_signal = True
        addback_fraction = max_addback
        reason = "tier1_recovery_addback"
        state.last_addback_date = snapshot.as_of
        watch.append("recovery_signal=BUY")

    return RecoveryUpdate(
        state=state,
        recovery_signal=recovery_signal,
        addback_fraction=addback_fraction,
        reason=reason,
        watch_items=tuple(watch),
    )


def build_system_vs_manual(
    *,
    report_date: str,
    decision: TierDecision,
    exposure_scale: float | None,
    risk_macro_budget: float | None,
    recovery_state: RecoveryState,
    config: DecisionPackConfig | None = None,
) -> dict[str, Any]:
    cfg = config or load_decision_pack_config()
    l3_rules = _l3_gap_rules(cfg)
    exposure = exposure_scale if exposure_scale is not None else 1.0
    manual_threshold = int(l3_rules.get("manual_tier_threshold", 2))
    exposure_threshold = float(l3_rules.get("exposure_scale_threshold", 0.99))
    quarter_threshold = int(l3_rules.get("quarter_count_threshold", 2))

    l3_gap = (
        decision.executed_tier >= manual_threshold
        and exposure >= exposure_threshold
    )
    return {
        "date": report_date,
        "exposure_scale": exposure,
        "risk_macro_budget": risk_macro_budget,
        "manual_tier": decision.executed_tier,
        "manual_target_exposure": decision.target_equity_exposure,
        "l3_gap": l3_gap,
        "quarter_tier2_manual_count": recovery_state.quarter_tier2_manual_count,
        "quarter_l3_gap_alert": recovery_state.quarter_tier2_manual_count >= quarter_threshold,
    }


__all__ = [
    "RecoveryState",
    "RecoveryUpdate",
    "build_system_vs_manual",
    "load_recovery_state",
    "save_recovery_state",
    "update_recovery_state",
]
