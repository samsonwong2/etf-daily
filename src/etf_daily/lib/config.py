"""Load decision_pack YAML config into typed structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config" / "default.yaml"


@dataclass(frozen=True)
class TierRule:
    tier: int
    label: str
    target_equity_exposure: float | None
    triggers: dict[str, float]


@dataclass(frozen=True)
class RegimeConfig:
    structural_min_rules: int
    bench_5d_lte: float
    vol_percentile_gte: float
    sentiment_downgrade_enabled: bool
    sentiment_min_table_tier: int
    sentiment_exempt_tiers: frozenset[int]


OrdersMode = str  # display_only | proportional_legacy | ma_then_cap (future)


@dataclass(frozen=True)
class DecisionPackConfig:
    benchmark_code: str
    ma_window: int
    vol_window: int
    vol_lookback: int
    vol_percentile_threshold: float
    dd_lookback_days: int
    orders_mode: OrdersMode
    defensive_codes: frozenset[str]
    tiers: tuple[TierRule, ...]
    regime: RegimeConfig
    raw: dict[str, Any] = field(repr=False)


def _require_mapping(node: Any, key: str) -> dict[str, Any]:
    if not isinstance(node, dict):
        raise ValueError(f"Config key '{key}' must be a mapping")
    return node


def _parse_tiers(tiers_node: list[Any]) -> tuple[TierRule, ...]:
    rules: list[TierRule] = []
    for item in tiers_node:
        if not isinstance(item, dict):
            raise ValueError("Each tiers[] entry must be a mapping")
        tier = int(item["tier"])
        target = item.get("target_equity_exposure")
        triggers = item.get("triggers") or {}
        if triggers and not isinstance(triggers, dict):
            raise ValueError(f"tiers[{tier}].triggers must be a mapping")
        rules.append(
            TierRule(
                tier=tier,
                label=str(item.get("label", f"tier_{tier}")),
                target_equity_exposure=None if target is None else float(target),
                triggers={str(k): float(v) for k, v in triggers.items()},
            )
        )
    return tuple(sorted(rules, key=lambda r: r.tier))


def load_decision_pack_config(path: Path | str | None = None) -> DecisionPackConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyYAML is required: pip install pyyaml") from exc

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid config root in {config_path}")

    benchmark = _require_mapping(raw.get("benchmark"), "benchmark")
    portfolio = _require_mapping(raw.get("portfolio"), "portfolio")
    orders = _require_mapping(raw.get("orders"), "orders") if raw.get("orders") else {}
    regime_raw = _require_mapping(raw.get("regime"), "regime")
    regime_rules = _require_mapping(regime_raw.get("rules"), "regime.rules")
    sentiment = _require_mapping(regime_raw.get("sentiment_downgrade"), "regime.sentiment_downgrade")

    orders_mode = str(orders.get("mode", "proportional_legacy")).strip()
    if orders_mode not in {"display_only", "proportional_legacy", "ma_then_cap"}:
        raise ValueError(
            f"orders.mode must be display_only, proportional_legacy, or ma_then_cap; got {orders_mode!r}"
        )
    defensive_raw = raw.get("defensive_codes") or []
    if not isinstance(defensive_raw, list):
        raise ValueError("defensive_codes must be a list of instrument codes")

    return DecisionPackConfig(
        benchmark_code=str(benchmark.get("code", "SH510300")),
        ma_window=int(benchmark.get("ma_window", 20)),
        vol_window=int(benchmark.get("vol_window", 20)),
        vol_lookback=int(benchmark.get("vol_lookback", 120)),
        vol_percentile_threshold=float(benchmark.get("vol_percentile_threshold", 0.80)),
        dd_lookback_days=int(portfolio.get("dd_lookback_days", 20)),
        orders_mode=orders_mode,
        defensive_codes=frozenset(str(code).strip() for code in defensive_raw if str(code).strip()),
        tiers=_parse_tiers(raw.get("tiers") or []),
        regime=RegimeConfig(
            structural_min_rules=int(regime_raw.get("structural_min_rules", 2)),
            bench_5d_lte=float(regime_rules.get("bench_5d_lte", -0.03)),
            vol_percentile_gte=float(regime_rules.get("vol_percentile_gte", 0.80)),
            sentiment_downgrade_enabled=bool(sentiment.get("enabled", True)),
            sentiment_min_table_tier=int(sentiment.get("min_table_tier", 2)),
            sentiment_exempt_tiers=frozenset(int(x) for x in (sentiment.get("exempt_tiers") or [4])),
        ),
        raw=raw,
    )


def target_exposure_for_tier(config: DecisionPackConfig, tier: int) -> float | None:
    for rule in config.tiers:
        if rule.tier == tier:
            return rule.target_equity_exposure
    raise KeyError(f"Unknown tier: {tier}")
