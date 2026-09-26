"""Simulate reduce-on-alert strategies against replay timelines."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from etf_daily.lib.generate_daily_mu_position_report import normalize_code

SimMode = Literal["hold", "replay_alert", "metric_rule"]


@dataclass(frozen=True)
class StrategySpec:
    name: str
    mode: SimMode = "replay_alert"
    respond_yellow: bool = True
    respond_red: bool = True
    yellow_reduce_pct: float | None = None
    red_reduce_pct: float | None = None
    max_yellow_actions: int | None = None
    max_red_actions: int | None = None
    max_total_actions: int | None = None
    max_cumulative_reduce: float | None = None
    cooldown_days: int = 0
    require_pnl_lt: float | None = None
    red_reasons_allow: frozenset[str] | None = None
    skip_dates: frozenset[str] = field(default_factory=frozenset)
    one_shot_prefer_red: bool = False
    critical_min_suggest: float | None = None
    warning_b_branch: bool = False
    metric_pnl_lte: float | None = None
    metric_dd_lte: float | None = None
    metric_reduce_pct: float = 0.5
    metric_once: bool = True


@dataclass
class SymbolSimState:
    remaining_shares: float
    cash: float = 0.0
    cumulative_reduce_frac: float = 0.0
    yellow_actions: int = 0
    red_actions: int = 0
    total_actions: int = 0
    last_action_date: str | None = None
    metric_triggered: bool = False
    one_shot_done: bool = False
    warning_b_red_only: bool = False


@dataclass(frozen=True)
class SimAction:
    date: str
    reduce_pct: float
    shares_sold: float
    price: float
    alert_level: str
    reasons: str


@dataclass(frozen=True)
class SymbolSimResult:
    code: str
    cost: float
    shares: float
    cost_basis: float
    pnl: float
    hold_pnl: float
    final_position_pct: float
    min_market_value: float
    hold_min_market_value: float
    actions: tuple[SimAction, ...]


@dataclass(frozen=True)
class PortfolioSimResult:
    strategy: StrategySpec
    total_pnl: float
    total_hold_pnl: float
    vs_hold: float
    total_min_mv: float
    hold_total_min_mv: float
    symbols: tuple[SymbolSimResult, ...]
    win_count: int
    lose_count: int
    tie_count: int


def _read_live_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def load_holdings_for_sim(live_holdings: Path) -> dict[str, dict[str, float | str]]:
    frame = _read_live_csv(live_holdings.expanduser().resolve())
    frame["code"] = frame["code"].astype(str).str.strip().map(normalize_code)
    out: dict[str, dict[str, float | str]] = {}
    for _, row in frame.iterrows():
        code = str(row["code"])
        if not code or code == "CASH":
            continue
        out[code] = {
            "cost": float(row["price"]),
            "shares": float(row["shares"]),
            "name": str(row.get("name", code)),
        }
    return out


def load_replay_frames(replay_dir: Path, codes: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    replay_dir = replay_dir.expanduser().resolve()
    frames: dict[str, pd.DataFrame] = {}
    for code in codes:
        path = replay_dir / f"{code}_replay.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing replay CSV: {path}")
        frame = pd.read_csv(path)
        frame["date"] = frame["date"].astype(str)
        frames[code] = frame.sort_values("date").reset_index(drop=True)
    return frames


def price_from_pnl(cost: float, pnl_pct: float | None) -> float:
    if pnl_pct is None or (isinstance(pnl_pct, float) and pd.isna(pnl_pct)):
        return cost
    return cost * (1.0 + float(pnl_pct) / 100.0)


def _parse_reasons(reasons: str | None) -> set[str]:
    return {part for part in str(reasons or "").split("|") if part}


def _days_between(d1: str, d2: str) -> int:
    return abs((pd.Timestamp(d1) - pd.Timestamp(d2)).days)


def _row_passes_filters(row: pd.Series, spec: StrategySpec) -> bool:
    date = str(row.get("date") or "")
    if date in spec.skip_dates:
        return False
    pnl = row.get("pnl_pct")
    if spec.require_pnl_lt is not None:
        if pnl is None or pd.isna(pnl) or float(pnl) >= spec.require_pnl_lt:
            return False
    return True


def _red_allowed(row: pd.Series, spec: StrategySpec) -> bool:
    if spec.red_reasons_allow is None:
        return True
    reasons = _parse_reasons(row.get("reasons"))
    return bool(reasons & spec.red_reasons_allow)


def _cooldown_ok(state: SymbolSimState, date: str, spec: StrategySpec) -> bool:
    if spec.cooldown_days <= 0 or state.last_action_date is None:
        return True
    return _days_between(state.last_action_date, date) >= spec.cooldown_days


def _cap_reduce(state: SymbolSimState, reduce_pct: float, spec: StrategySpec) -> float:
    if reduce_pct <= 0:
        return 0.0
    if spec.max_cumulative_reduce is not None:
        room = spec.max_cumulative_reduce - state.cumulative_reduce_frac
        if room <= 0:
            return 0.0
        reduce_pct = min(reduce_pct, room)
    return max(0.0, min(1.0, reduce_pct))


def _action_limits_ok(level: str, state: SymbolSimState, spec: StrategySpec) -> bool:
    if spec.max_total_actions is not None and state.total_actions >= spec.max_total_actions:
        return False
    if level == "yellow" and spec.max_yellow_actions is not None:
        if state.yellow_actions >= spec.max_yellow_actions:
            return False
    if level == "red" and spec.max_red_actions is not None:
        if state.red_actions >= spec.max_red_actions:
            return False
    return True


def _first_alert_date(replay: pd.DataFrame, level: str) -> str | None:
    hits = replay.loc[replay["alert_level"] == level, "date"]
    if hits.empty:
        return None
    return str(hits.iloc[0])


def _init_warning_b_branch(replay: pd.DataFrame, state: SymbolSimState) -> None:
    first_yellow = _first_alert_date(replay, "yellow")
    first_red = _first_alert_date(replay, "red")
    if first_red and (first_yellow is None or first_red <= first_yellow):
        state.warning_b_red_only = True


def decide_reduce(row: pd.Series, state: SymbolSimState, spec: StrategySpec) -> float:
    if spec.mode == "hold":
        return 0.0

    if not _row_passes_filters(row, spec):
        return 0.0
    if not _cooldown_ok(state, str(row["date"]), spec):
        return 0.0

    if spec.mode == "metric_rule":
        if spec.metric_once and state.metric_triggered:
            return 0.0
        pnl = row.get("pnl_pct")
        dd = row.get("dd_20d_high")
        hit = False
        if spec.metric_pnl_lte is not None and pnl is not None and not pd.isna(pnl):
            if float(pnl) <= spec.metric_pnl_lte:
                hit = True
        if spec.metric_dd_lte is not None and dd is not None and not pd.isna(dd):
            if float(dd) <= spec.metric_dd_lte:
                hit = True
        if not hit:
            return 0.0
        return _cap_reduce(state, spec.metric_reduce_pct, spec)

    level = str(row.get("alert_level") or "green")
    suggest = float(row.get("suggest_reduce_pct") or 0.0)

    respond_yellow = spec.respond_yellow and not state.warning_b_red_only

    if spec.critical_min_suggest is not None:
        if suggest < spec.critical_min_suggest:
            return 0.0
        reduce_pct = suggest
        if not _action_limits_ok("red", state, spec):
            return 0.0
        return _cap_reduce(state, reduce_pct, spec)

    if spec.one_shot_prefer_red:
        if state.one_shot_done:
            return 0.0
        if level == "red" and spec.respond_red and _red_allowed(row, spec):
            reduce_pct = spec.red_reduce_pct if spec.red_reduce_pct is not None else suggest
            if reduce_pct > 0 and _action_limits_ok("red", state, spec):
                return _cap_reduce(state, reduce_pct, spec)
        if level == "yellow" and respond_yellow:
            reduce_pct = spec.yellow_reduce_pct if spec.yellow_reduce_pct is not None else suggest
            if reduce_pct > 0 and _action_limits_ok("yellow", state, spec):
                return _cap_reduce(state, reduce_pct, spec)
        return 0.0

    reduce_pct = 0.0
    if level == "red" and spec.respond_red and _red_allowed(row, spec):
        reduce_pct = spec.red_reduce_pct if spec.red_reduce_pct is not None else suggest
        if not _action_limits_ok("red", state, spec):
            reduce_pct = 0.0
    elif level == "yellow" and respond_yellow:
        reduce_pct = spec.yellow_reduce_pct if spec.yellow_reduce_pct is not None else suggest
        if not _action_limits_ok("yellow", state, spec):
            reduce_pct = 0.0

    return _cap_reduce(state, reduce_pct, spec)


def simulate_symbol(
    code: str,
    replay: pd.DataFrame,
    *,
    cost: float,
    shares: float,
    spec: StrategySpec,
) -> SymbolSimResult:
    initial = shares * cost
    state = SymbolSimState(remaining_shares=float(shares))
    if spec.warning_b_branch:
        _init_warning_b_branch(replay, state)
    actions: list[SimAction] = []

    hold_rem = float(shares)
    hold_cash = 0.0
    min_hold = initial
    min_strat = initial

    for _, row in replay.iterrows():
        price = price_from_pnl(cost, row.get("pnl_pct"))
        hold_mv = hold_rem * price + hold_cash
        min_hold = min(min_hold, hold_mv)

        reduce_pct = decide_reduce(row, state, spec)
        if reduce_pct > 0 and state.remaining_shares > 0:
            sold = state.remaining_shares * reduce_pct
            state.cash += sold * price
            state.remaining_shares -= sold
            if shares > 0:
                state.cumulative_reduce_frac = 1.0 - (state.remaining_shares / float(shares))
            state.total_actions += 1
            state.last_action_date = str(row["date"])
            level = str(row.get("alert_level") or "")
            if level == "yellow":
                state.yellow_actions += 1
            elif level == "red":
                state.red_actions += 1
            if spec.one_shot_prefer_red:
                state.one_shot_done = True
            if spec.mode == "metric_rule":
                state.metric_triggered = True
            actions.append(
                SimAction(
                    date=str(row["date"]),
                    reduce_pct=reduce_pct,
                    shares_sold=sold,
                    price=price,
                    alert_level=level,
                    reasons=str(row.get("reasons") or ""),
                )
            )

        strat_mv = state.remaining_shares * price + state.cash
        min_strat = min(min_strat, strat_mv)

    final_row = replay.iloc[-1]
    final_price = price_from_pnl(cost, final_row.get("pnl_pct"))
    hold_pnl = hold_rem * final_price + hold_cash - initial
    strat_pnl = state.remaining_shares * final_price + state.cash - initial
    final_pct = (state.remaining_shares / float(shares) * 100.0) if shares else 0.0

    return SymbolSimResult(
        code=code,
        cost=cost,
        shares=shares,
        cost_basis=initial,
        pnl=strat_pnl,
        hold_pnl=hold_pnl,
        final_position_pct=final_pct,
        min_market_value=min_strat,
        hold_min_market_value=min_hold,
        actions=tuple(actions),
    )


def simulate_portfolio(
    holdings: dict[str, dict[str, float | str]],
    replay_frames: dict[str, pd.DataFrame],
    spec: StrategySpec,
    *,
    tie_threshold: float = 100.0,
) -> PortfolioSimResult:
    symbols: list[SymbolSimResult] = []
    for code, meta in sorted(holdings.items()):
        replay = replay_frames[code]
        result = simulate_symbol(
            code,
            replay,
            cost=float(meta["cost"]),
            shares=float(meta["shares"]),
            spec=spec,
        )
        symbols.append(result)

    total_pnl = sum(s.pnl for s in symbols)
    total_hold = sum(s.hold_pnl for s in symbols)
    win = lose = tie = 0
    for s in symbols:
        diff = s.pnl - s.hold_pnl
        if abs(diff) <= tie_threshold:
            tie += 1
        elif diff > 0:
            win += 1
        else:
            lose += 1

    return PortfolioSimResult(
        strategy=spec,
        total_pnl=total_pnl,
        total_hold_pnl=total_hold,
        vs_hold=total_pnl - total_hold,
        total_min_mv=sum(s.min_market_value for s in symbols),
        hold_total_min_mv=sum(s.hold_min_market_value for s in symbols),
        symbols=tuple(symbols),
        win_count=win,
        lose_count=lose,
        tie_count=tie,
    )


def compare_vs_hold(
    strategy_result: PortfolioSimResult,
    hold_result: PortfolioSimResult,
) -> dict[str, float]:
    return {
        "vs_hold_pnl": strategy_result.total_pnl - hold_result.total_pnl,
        "vs_hold_min_mv": strategy_result.total_min_mv - hold_result.hold_total_min_mv,
    }


__all__ = [
    "PortfolioSimResult",
    "SimAction",
    "StrategySpec",
    "SymbolSimResult",
    "SymbolSimState",
    "decide_reduce",
    "load_holdings_for_sim",
    "load_replay_frames",
    "price_from_pnl",
    "simulate_portfolio",
    "simulate_symbol",
]
