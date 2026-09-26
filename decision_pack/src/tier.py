"""Tier ladder and regime classification (Playbook steps 1–2)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from decision_pack.src.config import (
    DecisionPackConfig,
    DEFAULT_CONFIG_PATH,
    load_decision_pack_config,
    target_exposure_for_tier,
)

Regime = Literal["structural_weakness", "sentiment_shock"]


@dataclass(frozen=True)
class MarketSnapshot:
    """Inputs required to evaluate tier — usually from market.py + nav (P0: caller supplies)."""

    as_of: str
    benchmark: str
    r_bench: float
    bench_3d: float
    bench_5d: float
    close_below_ma20: bool
    vol_percentile_120d: float
    dd_port: float


@dataclass(frozen=True)
class TierDecision:
    as_of: str
    benchmark: str
    r_bench: float
    dd_port: float
    vol_percentile_120d: float
    table_tier: int
    regime: Regime
    executed_tier: int
    current_equity_exposure: float
    target_equity_exposure: float | None
    reduce_fraction: float
    triggers: tuple[str, ...]
    structural_rule_hits: tuple[str, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["triggers"] = list(self.triggers)
        payload["structural_rule_hits"] = list(self.structural_rule_hits)
        return payload


def _trigger_hit(rule_triggers: dict[str, float], snapshot: MarketSnapshot) -> list[str]:
    hits: list[str] = []
    for key, threshold in rule_triggers.items():
        if key == "r_bench_lte" and snapshot.r_bench <= threshold:
            hits.append(f"r_bench<={threshold:.4f}")
        elif key == "bench_3d_lte" and snapshot.bench_3d <= threshold:
            hits.append(f"bench_3d<={threshold:.4f}")
        elif key == "dd_port_gte" and snapshot.dd_port >= threshold:
            hits.append(f"dd_port>={threshold:.4f}")
        elif key == "below_ma20_and_bench_5d_lte":
            if snapshot.close_below_ma20 and snapshot.bench_5d <= threshold:
                hits.append(f"below_MA20_and_bench_5d<={threshold:.4f}")
    return hits


def evaluate_table_tier(
    snapshot: MarketSnapshot,
    config: DecisionPackConfig | None = None,
) -> tuple[int, tuple[str, ...]]:
    """Return highest triggered tier (0 = hold) and trigger labels."""
    cfg = config or load_decision_pack_config()
    tier_hits: list[tuple[int, str]] = []

    for rule in cfg.tiers:
        if rule.tier == 0 or not rule.triggers:
            continue
        for label in _trigger_hit(rule.triggers, snapshot):
            tier_hits.append((rule.tier, label))

    if not tier_hits:
        return 0, ()

    table_tier = max(tier for tier, _ in tier_hits)
    triggers = tuple(label for tier, label in tier_hits if tier == table_tier)
    # Include lower-tier triggers that also fired (audit trail)
    all_triggers = tuple(dict.fromkeys(label for _, label in sorted(tier_hits)))
    return table_tier, all_triggers


def classify_regime(
    snapshot: MarketSnapshot,
    config: DecisionPackConfig | None = None,
) -> tuple[Regime, tuple[str, ...]]:
    cfg = config or load_decision_pack_config()
    hits: list[str] = []

    if snapshot.close_below_ma20:
        hits.append("close_below_ma20")
    if snapshot.bench_5d <= cfg.regime.bench_5d_lte:
        hits.append(f"bench_5d<={cfg.regime.bench_5d_lte:.4f}")
    if snapshot.vol_percentile_120d >= cfg.regime.vol_percentile_gte:
        hits.append(f"vol_pct>={cfg.regime.vol_percentile_gte:.2f}")

    if len(hits) >= cfg.regime.structural_min_rules:
        return "structural_weakness", tuple(hits)
    return "sentiment_shock", tuple(hits)


def apply_executed_tier(
    table_tier: int,
    regime: Regime,
    config: DecisionPackConfig,
) -> int:
    if table_tier in config.regime.sentiment_exempt_tiers:
        return table_tier
    if (
        regime == "sentiment_shock"
        and config.regime.sentiment_downgrade_enabled
        and table_tier >= config.regime.sentiment_min_table_tier
    ):
        return max(0, table_tier - 1)
    return table_tier


def resolve_target_exposure(
    executed_tier: int,
    current_equity_exposure: float,
    config: DecisionPackConfig,
) -> float | None:
    target = target_exposure_for_tier(config, executed_tier)
    if target is None:
        return None
    if current_equity_exposure <= target:
        return current_equity_exposure
    return target


def compute_reduce_fraction(
    current_equity_exposure: float,
    target_equity_exposure: float | None,
) -> float:
    if target_equity_exposure is None:
        return 0.0
    if current_equity_exposure <= 0:
        return 0.0
    if target_equity_exposure >= current_equity_exposure:
        return 0.0
    return (current_equity_exposure - target_equity_exposure) / current_equity_exposure


def decide_tier(
    snapshot: MarketSnapshot,
    *,
    current_equity_exposure: float = 1.0,
    config: DecisionPackConfig | None = None,
) -> TierDecision:
    """Full tier path: table tier → regime → executed tier → target exposure."""
    cfg = config or load_decision_pack_config()
    table_tier, triggers = evaluate_table_tier(snapshot, cfg)
    regime, structural_hits = classify_regime(snapshot, cfg)
    executed_tier = apply_executed_tier(table_tier, regime, cfg)
    target = resolve_target_exposure(executed_tier, current_equity_exposure, cfg)
    reduce_fraction = compute_reduce_fraction(current_equity_exposure, target)

    return TierDecision(
        as_of=snapshot.as_of,
        benchmark=snapshot.benchmark,
        r_bench=snapshot.r_bench,
        dd_port=snapshot.dd_port,
        vol_percentile_120d=snapshot.vol_percentile_120d,
        table_tier=table_tier,
        regime=regime,
        executed_tier=executed_tier,
        current_equity_exposure=current_equity_exposure,
        target_equity_exposure=target,
        reduce_fraction=reduce_fraction,
        triggers=triggers,
        structural_rule_hits=structural_hits,
    )


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "MarketSnapshot",
    "Regime",
    "TierDecision",
    "apply_executed_tier",
    "classify_regime",
    "compute_reduce_fraction",
    "decide_tier",
    "evaluate_table_tier",
    "load_decision_pack_config",
    "resolve_target_exposure",
]
