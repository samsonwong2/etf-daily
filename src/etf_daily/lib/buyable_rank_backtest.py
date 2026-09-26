"""Causal walk-forward backtest for buyable Top-K selection rules."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from etf_daily.lib.buyable_rank import (
    BUYABLE_COLUMNS,
    BuyableRankConfig,
    GateName,
    build_buyable_frame,
    gate_to_buyable_config,
)
from etf_daily.lib.config import DecisionPackConfig, load_decision_pack_config
from etf_daily.lib.fat_tail_regime import enrich_regime_columns
from etf_daily.lib.fat_tail_risk import (
    FatTailConfig,
    build_fat_tail_frame,
    fat_tail_config_from_raw,
)
from etf_daily.lib.garch_short_horizon_board import garch_short_horizon_config_from_raw
from etf_daily.lib.hmm_short_horizon import hmm_short_horizon_config_from_raw
from etf_daily.lib.portfolio import PortfolioSnapshot
from etf_daily.lib.price_jump_flags import build_jump_diagnostics_frame
from etf_daily.lib.short_horizon_validation import build_eval_dates
from etf_daily.lib.symbol_trend_board import (
    build_cluster_trend_board,
    load_cluster_mapping_codes,
)
from etf_daily.lib.tomorrow_digest import (
    RankRule,
    TomorrowDigestConfig,
    build_tomorrow_digest_frame,
    select_topk_picks,
    tomorrow_digest_config_from_raw,
)
from etf_daily.pipeline.price_adjustments import auto_qfq_adjust_price_columns_by_group
from etf_daily.paths import CLUSTER_MAPPING_SELECTED_TXT, FUND_LIST_CSV, QLIB_PROVIDER_URI
from etf_daily.lib.generate_daily_mu_position_report import load_fund_name_map, normalize_code

SegmentName = Literal["train", "validation", "holdout", "full"]

_PATH_STATE_SHORT = {
    "near_high": "近高位",
    "pullback": "回撤中",
    "broken": "破位",
}

DEFAULT_TRAIN_END = "2026-05-31"
DEFAULT_VALIDATION_END = "2026-06-30"


@dataclass(frozen=True)
class StrategySpec:
    name: str
    rank_rule: RankRule
    k: int
    gate: GateName
    min_credibility: float
    is_benchmark: bool = False


@dataclass(frozen=True)
class DaySignalBundle:
    as_of: str
    trade_date: str
    digest: pd.DataFrame
    trend: pd.DataFrame
    pool_risk: pd.DataFrame
    jump: pd.DataFrame
    cluster_frame: pd.DataFrame


@dataclass(frozen=True)
class SegmentMetrics:
    segment: SegmentName
    strategy: str
    n_days: int
    n_trade_days: int
    avg_n_picks: float
    cash_day_rate: float
    cum_ret: float
    mean_daily_ret: float
    hit_rate: float
    max_dd: float
    sharpe: float
    vol_ann: float


def build_strategy_grid(
    *,
    include_benchmarks: bool = True,
    k_values: tuple[int, ...] = (1, 2, 3, 5, 8),
    rank_rules: tuple[RankRule, ...] = ("upside_x_cred", "dir_x_cred", "composite"),
    gates: tuple[GateName, ...] = ("strict", "no_path_gate", "core_risk", "positive_score"),
    min_creds: tuple[float, ...] = (0.0, 0.003, 0.005),
) -> list[StrategySpec]:
    specs: list[StrategySpec] = []
    if include_benchmarks:
        specs.append(
            StrategySpec(
                name="bench_tomorrow_topk_upside_k8",
                rank_rule="upside_x_cred",
                k=8,
                gate="none",
                min_credibility=0.0,
                is_benchmark=True,
            )
        )
        specs.append(
            StrategySpec(
                name="bench_buyable_strict_upside_k8_c003",
                rank_rule="upside_x_cred",
                k=8,
                gate="strict",
                min_credibility=0.003,
                is_benchmark=True,
            )
        )
    for rule in rank_rules:
        for k in k_values:
            for gate in gates:
                for cred in min_creds:
                    name = f"{gate}__{rule}__k{k}__c{cred:g}"
                    specs.append(
                        StrategySpec(
                            name=name,
                            rank_rule=rule,
                            k=int(k),
                            gate=gate,
                            min_credibility=float(cred),
                        )
                    )
    return specs


def load_qlib_open_close(
    provider_uri: str | Path,
    codes: list[str],
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return wide open/close panels (qfq-adjusted), indexed by datetime."""
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    provider_path = Path(provider_uri).expanduser()
    qlib.init(provider_uri=str(provider_path), region=REG_CN)
    raw = D.features(
        codes,
        ["$open", "$close"],
        start_time=start_date,
        end_time=end_date,
        freq="day",
    )
    if raw is None or raw.empty:
        raise RuntimeError("Qlib returned no OHLC data")
    raw = raw.sort_index().reset_index()
    raw["instrument"] = raw["instrument"].map(normalize_code)
    raw, _ = auto_qfq_adjust_price_columns_by_group(
        raw,
        group_col="instrument",
        date_col="datetime",
        price_cols=["$open", "$close"],
    )
    open_panel = raw.pivot(index="datetime", columns="instrument", values="$open").sort_index()
    close_panel = raw.pivot(index="datetime", columns="instrument", values="$close").sort_index()
    open_panel.columns = [normalize_code(c) for c in open_panel.columns]
    close_panel.columns = [normalize_code(c) for c in close_panel.columns]
    return open_panel, close_panel


def _empty_portfolio(as_of: str) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        as_of=as_of,
        source="backtest",
        holdings=(),
        total_assets=0.0,
        cash_mv=0.0,
        stock_mv=0.0,
        equity_exposure=0.0,
    )


def _flags_str(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "|".join(str(x) for x in value if x)
    return str(value)


def build_pool_risk_lite(fat_frame: pd.DataFrame) -> pd.DataFrame:
    """Minimal pool_risk columns consumed by build_buyable_frame."""
    if fat_frame.empty:
        return pd.DataFrame(columns=["code", "leg", "状态"])
    rows = []
    for _, row in fat_frame.iterrows():
        path = str(row.get("path_state") or "")
        rows.append(
            {
                "code": normalize_code(row["code"]),
                "leg": str(row.get("barbell_leg") or "—"),
                "状态": _PATH_STATE_SHORT.get(path, path or "—"),
            }
        )
    return pd.DataFrame(rows)


def reconstruct_day_signals(
    *,
    as_of: str,
    trade_date: str,
    close_panel: pd.DataFrame,
    cluster_codes: tuple[str, ...],
    names: dict[str, str],
    pack_cfg: DecisionPackConfig,
    fat_cfg: FatTailConfig,
    digest_cfg: TomorrowDigestConfig,
    garch_cfg: Any,
    hmm_cfg: Any,
    cluster_mapping_path: Path,
) -> DaySignalBundle:
    end = pd.Timestamp(as_of)
    panel = close_panel.loc[close_panel.index <= end].copy()
    if panel.empty:
        empty = pd.DataFrame()
        return DaySignalBundle(
            as_of=as_of,
            trade_date=trade_date,
            digest=empty,
            trend=empty,
            pool_risk=empty,
            jump=empty,
            cluster_frame=empty,
        )

    available_codes = [
        c for c in cluster_codes if c in panel.columns and int(panel[c].notna().sum()) >= 20
    ]
    fat_252 = replace(fat_cfg, lookback_days=252)
    fat_frame = build_fat_tail_frame(
        available_codes,
        panel,
        as_of,
        weights={c: 0.0 for c in available_codes},
        names=names,
        defensive_codes=pack_cfg.defensive_codes,
        cfg=fat_252,
        rng=np.random.default_rng(fat_252.kappa_seed),
    )
    fat_frame = enrich_regime_columns(fat_frame, panel, as_of, fat_252)
    pool_risk = build_pool_risk_lite(fat_frame)

    cluster_frame = fat_frame.copy()
    if "flags" in cluster_frame.columns:
        cluster_frame["flags"] = cluster_frame["flags"].map(_flags_str)
    else:
        cluster_frame["flags"] = ""

    portfolio = _empty_portfolio(as_of)
    trend_result = build_cluster_trend_board(
        portfolio,
        cluster_mapping_path,
        fund_list_csv=FUND_LIST_CSV if FUND_LIST_CSV.exists() else None,
        config=pack_cfg,
        close_panel=panel,
    )
    trend = trend_result.board

    jump = build_jump_diagnostics_frame(
        panel,
        as_of=as_of,
        codes=available_codes,
        names=names,
        live_closes=None,
    )

    digest = build_tomorrow_digest_frame(
        cluster_frame,
        panel,
        as_of,
        digest_cfg=digest_cfg,
        garch_board_cfg=garch_cfg,
        hmm_board_cfg=hmm_cfg,
        garch_cal=pd.DataFrame(),
        hmm_cal=pd.DataFrame(),
    )
    return DaySignalBundle(
        as_of=as_of,
        trade_date=trade_date,
        digest=digest,
        trend=trend,
        pool_risk=pool_risk,
        jump=jump,
        cluster_frame=cluster_frame,
    )


def select_strategy_picks(
    bundle: DaySignalBundle,
    strategy: StrategySpec,
    *,
    digest_cfg: TomorrowDigestConfig,
) -> tuple[list[str], pd.DataFrame]:
    """Return ordered pick codes and full buyable/audit frame for one strategy."""
    topk_cfg = replace(
        digest_cfg.topk,
        enabled=True,
        rank_all=True,
        k=int(strategy.k),
        rank_rule=strategy.rank_rule,
        exclude_avoid_leg=False,
        candidates_only=False,
        require_positive_direction=True,
        min_credibility=float(strategy.min_credibility),
        min_direction=0.0,
    )
    day_digest_cfg = replace(digest_cfg, topk=topk_cfg, candidates_only=False)
    topk = select_topk_picks(bundle.digest, bundle.cluster_frame, digest_cfg=day_digest_cfg)
    buyable_cfg = gate_to_buyable_config(strategy.gate, k=strategy.k)
    frame = build_buyable_frame(
        topk,
        bundle.trend,
        bundle.pool_risk,
        bundle.jump,
        cfg=buyable_cfg,
    )
    if frame.empty:
        return [], frame

    passed = frame.loc[frame["pass_filters"]].copy()
    if strategy.min_credibility > 0 and not passed.empty:
        cred = pd.to_numeric(passed["combined_cred"], errors="coerce")
        passed = passed.loc[cred.fillna(-1.0) >= float(strategy.min_credibility)]
    if passed.empty:
        return [], frame
    picked = passed.head(int(strategy.k))
    codes = [normalize_code(c) for c in picked["code"].tolist() if normalize_code(c)]
    return codes, frame


def open_to_close_return(
    open_px: float | None,
    close_px: float | None,
    *,
    cost_bps_per_side: float,
) -> float | None:
    if open_px is None or close_px is None:
        return None
    if not math.isfinite(open_px) or not math.isfinite(close_px) or open_px <= 0 or close_px <= 0:
        return None
    gross = close_px / open_px - 1.0
    cost = 2.0 * (float(cost_bps_per_side) / 10000.0)
    return gross - cost


def price_on(
    panel: pd.DataFrame,
    code: str,
    date: pd.Timestamp,
) -> float | None:
    if code not in panel.columns or date not in panel.index:
        return None
    value = pd.to_numeric(pd.Series([panel.at[date, code]]), errors="coerce").iloc[0]
    if pd.isna(value) or not math.isfinite(float(value)) or float(value) <= 0:
        return None
    return float(value)


def portfolio_day_return(
    codes: list[str],
    *,
    trade_date: pd.Timestamp,
    open_panel: pd.DataFrame,
    close_panel: pd.DataFrame,
    cost_bps_per_side: float,
) -> tuple[float, list[dict[str, Any]]]:
    if not codes:
        return 0.0, []
    rets: list[float] = []
    rows: list[dict[str, Any]] = []
    for code in codes:
        o = price_on(open_panel, code, trade_date)
        c = price_on(close_panel, code, trade_date)
        net = open_to_close_return(o, c, cost_bps_per_side=cost_bps_per_side)
        if net is None:
            continue
        rets.append(net)
        gross = (c / o - 1.0) if o and c and o > 0 else None
        rows.append(
            {
                "code": code,
                "open": o,
                "close": c,
                "gross_ret": gross,
                "net_ret": net,
            }
        )
    if not rets:
        return 0.0, rows
    return float(np.mean(rets)), rows


def max_drawdown(rets: pd.Series) -> float:
    if rets.empty:
        return 0.0
    eq = (1.0 + rets.fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def sharpe_ann(rets: pd.Series) -> float:
    if rets.empty or float(rets.std(ddof=0)) <= 0:
        return 0.0
    return float(rets.mean() / rets.std(ddof=0) * math.sqrt(252.0))


def summarize_segment(
    daily: pd.DataFrame,
    *,
    strategy: str,
    segment: SegmentName,
) -> SegmentMetrics:
    sub = daily.copy() if not daily.empty else daily
    if sub.empty:
        return SegmentMetrics(
            segment=segment,
            strategy=strategy,
            n_days=0,
            n_trade_days=0,
            avg_n_picks=0.0,
            cash_day_rate=0.0,
            cum_ret=0.0,
            mean_daily_ret=0.0,
            hit_rate=0.0,
            max_dd=0.0,
            sharpe=0.0,
            vol_ann=0.0,
        )
    rets = pd.to_numeric(sub["net_ret"], errors="coerce").fillna(0.0)
    n_picks = pd.to_numeric(sub["n_picks"], errors="coerce").fillna(0.0)
    cash_days = int((n_picks <= 0).sum())
    return SegmentMetrics(
        segment=segment,
        strategy=strategy,
        n_days=len(sub),
        n_trade_days=int((n_picks > 0).sum()),
        avg_n_picks=float(n_picks.mean()),
        cash_day_rate=float(cash_days / len(sub)),
        cum_ret=float((1.0 + rets).prod() - 1.0),
        mean_daily_ret=float(rets.mean()),
        hit_rate=float((rets > 0).mean()) if len(rets) else 0.0,
        max_dd=max_drawdown(rets),
        sharpe=sharpe_ann(rets),
        vol_ann=float(rets.std(ddof=0) * math.sqrt(252.0)) if len(rets) else 0.0,
    )


def assign_segment(signal_date: str) -> SegmentName:
    ts = pd.Timestamp(signal_date)
    if ts <= pd.Timestamp(DEFAULT_TRAIN_END):
        return "train"
    if ts <= pd.Timestamp(DEFAULT_VALIDATION_END):
        return "validation"
    return "holdout"


def select_best_strategy(
    summary: pd.DataFrame,
    *,
    exclude_benchmarks: bool = True,
) -> dict[str, Any] | None:
    if summary.empty:
        return None
    frame = summary.copy()
    if exclude_benchmarks and "is_benchmark" in frame.columns:
        frame = frame.loc[~frame["is_benchmark"].astype(bool)]
    val = frame.loc[frame["segment"] == "validation"].copy()
    if val.empty:
        return None
    val = val.sort_values(
        ["sharpe", "max_dd", "cum_ret", "strategy"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    best = val.iloc[0]

    def _seg(segment: str) -> dict[str, Any] | None:
        hold = frame.loc[(frame["strategy"] == best["strategy"]) & (frame["segment"] == segment)]
        if hold.empty:
            return None
        row = hold.iloc[0]
        return {
            "sharpe": float(row["sharpe"]),
            "cum_ret": float(row["cum_ret"]),
            "max_dd": float(row["max_dd"]),
            "hit_rate": float(row["hit_rate"]),
            "avg_n_picks": float(row["avg_n_picks"]),
            "cash_day_rate": float(row["cash_day_rate"]),
            "n_days": int(row["n_days"]),
        }

    return {
        "strategy": str(best["strategy"]),
        "rank_rule": str(best.get("rank_rule", "")),
        "gate": str(best.get("gate", "")),
        "k": int(best.get("k", 0)),
        "min_credibility": float(best.get("min_credibility", 0.0)),
        "selection_segment": "validation",
        "selection_metric": "sharpe",
        "validation": _seg("validation"),
        "holdout": _seg("holdout"),
        "full": _seg("full"),
        "train": _seg("train"),
    }


def format_backtest_report(
    *,
    summary: pd.DataFrame,
    best: dict[str, Any] | None,
    eval_start: str,
    eval_end: str,
    cost_bps_per_side: float,
    n_signal_days: int,
    pack_dir: str,
) -> str:
    lines = [
        "# Buyable Rank Backtest",
        "",
        f"**Pack dir:** `{pack_dir}`",
        f"**Signal window:** {eval_start} → {eval_end} ({n_signal_days} signal days)",
        (
            f"**Execution:** next open → same-day close; "
            f"cost {cost_bps_per_side:g} bp/side "
            f"(round-trip {2 * cost_bps_per_side:g} bp)"
        ),
        (
            f"**Selection:** validation Sharpe "
            f"(train ≤ {DEFAULT_TRAIN_END}, validation ≤ {DEFAULT_VALIDATION_END}, else holdout)"
        ),
        "",
        "## Caveats",
        "",
        "- Fixed current `cluster_mapping_selected` universe → survivorship bias.",
        "- No historical H=1 calibration cache → pooled median credibility (same as live fallback).",
        "- Short sample; holdout is indicative, not a production guarantee.",
        "",
    ]
    if best is None:
        lines.append("**No best strategy selected** (empty validation).")
        lines.append("")
        return "\n".join(lines)

    val = best.get("validation") or {}
    lines.extend(
        [
            "## Best strategy (by validation Sharpe)",
            "",
            f"- **name:** `{best['strategy']}`",
            f"- gate / rank_rule / k / min_cred: "
            f"`{best.get('gate')}` / `{best.get('rank_rule')}` / "
            f"{best.get('k')} / {best.get('min_credibility')}",
            f"- validation Sharpe: **{float(val.get('sharpe', 0.0)):.3f}**",
            f"- validation cum_ret: {float(val.get('cum_ret', 0.0)):.2%}",
            f"- validation max_dd: {float(val.get('max_dd', 0.0)):.2%}",
            f"- validation hit_rate: {float(val.get('hit_rate', 0.0)):.1%}",
            f"- validation avg_n_picks: {float(val.get('avg_n_picks', 0.0)):.2f}",
            f"- validation cash_day_rate: {float(val.get('cash_day_rate', 0.0)):.1%}",
            "",
        ]
    )
    if best.get("holdout"):
        h = best["holdout"]
        lines.extend(
            [
                "### Holdout (July)",
                "",
                f"- Sharpe: **{h['sharpe']:.3f}**",
                f"- cum_ret: {h['cum_ret']:.2%}",
                f"- max_dd: {h['max_dd']:.2%}",
                f"- hit_rate: {h['hit_rate']:.1%}",
                f"- avg_n_picks: {h['avg_n_picks']:.2f}",
                f"- cash_day_rate: {h['cash_day_rate']:.1%}",
                "",
            ]
        )

    def _top_table(segment: str, n: int = 8) -> list[str]:
        if "is_benchmark" in summary.columns:
            sub = summary.loc[
                (summary["segment"] == segment) & (~summary["is_benchmark"].astype(bool))
            ].copy()
        else:
            sub = summary.loc[summary["segment"] == segment].copy()
        if sub.empty:
            return ["_(empty)_", ""]
        sub = sub.sort_values("sharpe", ascending=False, kind="stable").head(n)
        out = [
            f"### Top {n} by {segment} Sharpe",
            "",
            "| strategy | sharpe | cum_ret | max_dd | hit | avg_n | cash% |",
            "|----------|--------|---------|--------|-----|-------|-------|",
        ]
        for _, row in sub.iterrows():
            out.append(
                f"| `{row['strategy']}` | {row['sharpe']:.3f} | {row['cum_ret']:.2%} | "
                f"{row['max_dd']:.2%} | {row['hit_rate']:.0%} | {row['avg_n_picks']:.1f} | "
                f"{row['cash_day_rate']:.0%} |"
            )
        out.append("")
        return out

    lines.extend(_top_table("validation"))
    lines.extend(_top_table("holdout"))

    if "is_benchmark" in summary.columns:
        benches = summary.loc[
            summary["is_benchmark"].astype(bool) & (summary["segment"] == "holdout")
        ]
        if not benches.empty:
            lines.extend(["## Benchmarks (holdout)", ""])
            lines.append("| strategy | sharpe | cum_ret | max_dd |")
            lines.append("|----------|--------|---------|--------|")
            for _, row in benches.iterrows():
                lines.append(
                    f"| `{row['strategy']}` | {row['sharpe']:.3f} | "
                    f"{row['cum_ret']:.2%} | {row['max_dd']:.2%} |"
                )
            lines.append("")

    lines.extend(
        [
            "*Generated by `src/etf_daily/scripts/backtest_buyable_rank.py`.*",
            "",
        ]
    )
    return "\n".join(lines)


def run_buyable_rank_backtest(
    *,
    pack_dir: Path,
    eval_start: str = "2026-04-01",
    eval_end: str = "2026-07-17",
    cost_bps_per_side: float = 10.0,
    strategies: list[StrategySpec] | None = None,
    provider_uri: Path | str | None = None,
    cluster_mapping_path: Path | None = None,
    write_daily_buyable: bool = False,
    output_dir: Path | None = None,
    progress_every: int = 5,
) -> dict[str, Any]:
    pack_dir = pack_dir.expanduser().resolve()
    out_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else pack_dir / "buyable_rank_backtest"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pack_cfg = load_decision_pack_config()
    fat_cfg = fat_tail_config_from_raw(pack_cfg.raw.get("fat_tail"))
    digest_cfg = tomorrow_digest_config_from_raw(pack_cfg.raw.get("tomorrow_digest"))
    digest_cfg = replace(digest_cfg, candidates_only=False, horizon=1)
    digest_cfg = replace(
        digest_cfg,
        topk=replace(digest_cfg.topk, candidates_only=False, exclude_avoid_leg=False),
    )
    garch_cfg = garch_short_horizon_config_from_raw(pack_cfg.raw.get("garch_short_horizon"))
    hmm_cfg = hmm_short_horizon_config_from_raw(pack_cfg.raw.get("hmm_short_horizon"))

    cluster_path = Path(cluster_mapping_path or CLUSTER_MAPPING_SELECTED_TXT)
    cluster_codes = load_cluster_mapping_codes(cluster_path) if cluster_path.exists() else ()
    names = load_fund_name_map(FUND_LIST_CSV) if FUND_LIST_CSV.exists() else {}
    uri = str(provider_uri or QLIB_PROVIDER_URI)

    warmup = (
        pd.Timestamp(eval_start)
        - pd.tseries.offsets.BDay(fat_cfg.lookback_days + fat_cfg.warmup_buffer_days + 40)
    ).strftime("%Y-%m-%d")
    panel_end = (pd.Timestamp(eval_end) + pd.tseries.offsets.BDay(5)).strftime("%Y-%m-%d")

    open_panel, close_panel = load_qlib_open_close(
        uri,
        list(cluster_codes),
        warmup,
        panel_end,
    )
    common_idx = open_panel.index.intersection(close_panel.index).sort_values()
    open_panel = open_panel.loc[common_idx]
    close_panel = close_panel.loc[common_idx]

    signal_dates = build_eval_dates(
        close_panel,
        eval_start=pd.Timestamp(eval_start),
        eval_end=pd.Timestamp(eval_end),
        step_days=1,
    )
    usable: list[pd.Timestamp] = []
    for d in signal_dates:
        loc = common_idx.get_loc(d)
        if isinstance(loc, (slice, np.ndarray)):
            continue
        if int(loc) + 1 < len(common_idx):
            usable.append(pd.Timestamp(d))
    signal_dates = pd.DatetimeIndex(usable)

    specs = strategies or build_strategy_grid()
    daily_rows: list[dict[str, Any]] = []
    pick_rows: list[dict[str, Any]] = []

    for i, signal_ts in enumerate(signal_dates):
        as_of = signal_ts.strftime("%Y-%m-%d")
        loc = int(common_idx.get_loc(signal_ts))
        trade_ts = pd.Timestamp(common_idx[loc + 1])
        trade_date = trade_ts.strftime("%Y-%m-%d")
        if progress_every > 0 and (i % progress_every == 0 or i == len(signal_dates) - 1):
            print(
                f"[INFO] signals {i + 1}/{len(signal_dates)} as_of={as_of} trade={trade_date}",
                flush=True,
            )

        bundle = reconstruct_day_signals(
            as_of=as_of,
            trade_date=trade_date,
            close_panel=close_panel,
            cluster_codes=cluster_codes,
            names=names,
            pack_cfg=pack_cfg,
            fat_cfg=fat_cfg,
            digest_cfg=digest_cfg,
            garch_cfg=garch_cfg,
            hmm_cfg=hmm_cfg,
            cluster_mapping_path=cluster_path,
        )

        if write_daily_buyable:
            day_dir = out_dir / "daily" / signal_ts.strftime("%Y%m%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            strict_spec = StrategySpec(
                name="audit_strict",
                rank_rule="upside_x_cred",
                k=8,
                gate="strict",
                min_credibility=0.0,
            )
            _, audit_frame = select_strategy_picks(bundle, strict_spec, digest_cfg=digest_cfg)
            audit_frame.reindex(columns=list(BUYABLE_COLUMNS)).to_csv(
                day_dir / "buyable_topk_picks.csv",
                index=False,
                encoding="utf-8-sig",
            )

        segment = assign_segment(as_of)
        for spec in specs:
            codes, _frame = select_strategy_picks(bundle, spec, digest_cfg=digest_cfg)
            port_ret, detail = portfolio_day_return(
                codes,
                trade_date=trade_ts,
                open_panel=open_panel,
                close_panel=close_panel,
                cost_bps_per_side=cost_bps_per_side,
            )
            gross_vals = [d["gross_ret"] for d in detail if d.get("gross_ret") is not None]
            daily_rows.append(
                {
                    "signal_date": as_of,
                    "trade_date": trade_date,
                    "segment": segment,
                    "strategy": spec.name,
                    "rank_rule": spec.rank_rule,
                    "gate": spec.gate,
                    "k": spec.k,
                    "min_credibility": spec.min_credibility,
                    "is_benchmark": spec.is_benchmark,
                    "n_picks": len(codes),
                    "codes": "|".join(codes),
                    "net_ret": port_ret,
                    "gross_ret": float(np.mean(gross_vals)) if gross_vals else 0.0,
                }
            )
            for d in detail:
                pick_rows.append(
                    {
                        "signal_date": as_of,
                        "trade_date": trade_date,
                        "segment": segment,
                        "strategy": spec.name,
                        "code": d["code"],
                        "open": d["open"],
                        "close": d["close"],
                        "gross_ret": d["gross_ret"],
                        "net_ret": d["net_ret"],
                        "n_picks": len(codes),
                    }
                )

    daily = pd.DataFrame(daily_rows)
    picks = pd.DataFrame(pick_rows)

    summary_rows: list[dict[str, Any]] = []
    for spec in specs:
        for segment in ("train", "validation", "holdout", "full"):
            if segment == "full":
                sub = daily.loc[daily["strategy"] == spec.name]
            else:
                sub = daily.loc[
                    (daily["strategy"] == spec.name) & (daily["segment"] == segment)
                ]
            metrics = summarize_segment(sub, strategy=spec.name, segment=segment)  # type: ignore[arg-type]
            row = asdict(metrics)
            row.update(
                {
                    "rank_rule": spec.rank_rule,
                    "gate": spec.gate,
                    "k": spec.k,
                    "min_credibility": spec.min_credibility,
                    "is_benchmark": spec.is_benchmark,
                }
            )
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    best = select_best_strategy(summary, exclude_benchmarks=True)

    daily.to_csv(out_dir / "daily_returns.csv", index=False, encoding="utf-8-sig")
    picks.to_csv(out_dir / "daily_picks.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "strategy_summary.csv", index=False, encoding="utf-8-sig")
    (out_dir / "best_strategy.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = format_backtest_report(
        summary=summary,
        best=best,
        eval_start=eval_start,
        eval_end=eval_end,
        cost_bps_per_side=cost_bps_per_side,
        n_signal_days=len(signal_dates),
        pack_dir=str(pack_dir),
    )
    (out_dir / "BUYABLE_RANK_BACKTEST.md").write_text(report, encoding="utf-8")

    return {
        "output_dir": out_dir,
        "daily": daily,
        "picks": picks,
        "summary": summary,
        "best": best,
        "n_signal_days": len(signal_dates),
    }


__all__ = [
    "DaySignalBundle",
    "SegmentMetrics",
    "StrategySpec",
    "assign_segment",
    "build_pool_risk_lite",
    "build_strategy_grid",
    "format_backtest_report",
    "gate_to_buyable_config",
    "load_qlib_open_close",
    "open_to_close_return",
    "portfolio_day_return",
    "reconstruct_day_signals",
    "run_buyable_rank_backtest",
    "select_best_strategy",
    "select_strategy_picks",
    "sharpe_ann",
    "summarize_segment",
]
