#!/usr/bin/env python3
"""Backtest core-4 equal-weight vs GARCH daily rotation (Jun 2026 window)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.scripts.generate_fat_tail_risk_report import (
    build_universe_codes,
    load_as_of,
    load_fat_tail_data,
    load_holdings_from_pack,
    warmup_start,
)
from etf_daily.scripts.generate_garch_daily_picks_csv import collect_daily_picks
from etf_daily.scripts.validate_garch_short_horizon_board import _flags_by_code
from etf_daily.lib.config import load_decision_pack_config
from etf_daily.lib.fat_tail_risk import fat_tail_config_from_raw
from etf_daily.lib.garch_short_horizon_board import garch_short_horizon_config_from_raw
from etf_daily.lib.symbol_trend_board import load_cluster_mapping_codes
from etf_daily.pipeline.price_adjustments import auto_qfq_adjust_close_panel
from etf_daily.pipeline.strategy_layer_audit import load_qlib_close
from etf_daily.paths import CLUSTER_MAPPING_SELECTED_TXT, QLIB_PROVIDER_URI

CORE4 = ("SH588170", "SH588710", "SH588200", "SZ159502")


def max_drawdown(rets: pd.Series) -> float:
    eq = (1 + rets.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def summarize(name: str, daily: pd.Series) -> dict:
    daily = daily.dropna()
    if daily.empty:
        return {"strategy": name, "cum_ret_pct": np.nan, "max_dd_pct": np.nan, "sharpe": np.nan, "n_days": 0}
    eq = (1 + daily).cumprod()
    vol = float(daily.std())
    sharpe = float(daily.mean() / vol * np.sqrt(252)) if vol > 0 else 0.0
    return {
        "strategy": name,
        "cum_ret_pct": round((eq.iloc[-1] - 1) * 100, 2),
        "max_dd_pct": round(max_drawdown(daily) * 100, 2),
        "sharpe": round(sharpe, 3),
        "win_rate_pct": round(float((daily > 0).mean()) * 100, 1),
        "n_days": len(daily),
    }


def daily_equal_weight(panel: pd.Series, codes: list[str]) -> float:
    codes = [c for c in codes if c in panel.index and pd.notna(panel[c])]
    if not codes:
        return 0.0
    return float(panel[codes].mean())


def backtest_fixed_equal(panel_rets: pd.DataFrame, codes: tuple[str, ...]) -> pd.Series:
    cols = [c for c in codes if c in panel_rets.columns]
    return panel_rets[cols].mean(axis=1)


def backtest_from_pick_csv(
    pick_df: pd.DataFrame,
    panel_rets: pd.DataFrame,
    *,
    strategy: str,
    k: int | None = None,
    universe_filter: frozenset[str] | None = None,
    min_overlap_hold: int | None = None,
    prev_codes: list[str] | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Signal at T close -> return on T+1 (panel_rets index aligned)."""
    rows: list[dict] = []
    out: list[float] = []
    dates = sorted(pick_df["as_of"].unique())
    held: list[str] = list(prev_codes or [])

    for i, d in enumerate(dates[:-1]):
        next_d = dates[i + 1]
        day = pick_df[(pick_df["as_of"] == d) & (pick_df["strategy"] == strategy)].sort_values("rank")
        if k is not None:
            day = day.head(k)
        codes = day["code"].astype(str).tolist()
        if universe_filter is not None:
            codes = [c for c in codes if c in universe_filter]
        if min_overlap_hold is not None and held:
            overlap = len(set(codes) & set(held))
            if overlap >= min_overlap_hold:
                codes = held
        if not codes:
            codes = list(CORE4)
        held = codes
        if next_d not in panel_rets.index:
            continue
        r = daily_equal_weight(panel_rets.loc[next_d], codes)
        out.append(r)
        rows.append({"as_of": d, "next_date": next_d, "codes": "|".join(codes), "n": len(codes), "ret": r})

    idx = [r["next_date"] for r in rows]
    return pd.Series(out, index=pd.to_datetime(idx)), pd.DataFrame(rows)


def backtest_hybrid_core_satellite(
    panel_rets: pd.DataFrame,
    pick_df: pd.DataFrame,
    *,
    core_w: float,
    strategy: str,
    k: int = 6,
) -> pd.Series:
    core = backtest_fixed_equal(panel_rets, CORE4)
    sat, _ = backtest_from_pick_csv(pick_df, panel_rets, strategy=strategy, k=k)
    aligned = pd.concat([core.rename("core"), sat.rename("sat")], axis=1).dropna()
    return core_w * aligned["core"] + (1 - core_w) * aligned["sat"]


def core4_coverage(pick_df: pd.DataFrame, strategy: str, k: int = 6) -> pd.DataFrame:
    rows = []
    for d, g in pick_df[pick_df["strategy"] == strategy].groupby("as_of"):
        top = set(g.sort_values("rank").head(k)["code"].astype(str))
        n = len(top & set(CORE4))
        rows.append({"as_of": d, "core4_in_topk": n, "pct": n / len(CORE4)})
    return pd.DataFrame(rows)


def write_core4_daily(path: Path, trading: pd.DatetimeIndex) -> pd.DataFrame:
    w = 1.0 / len(CORE4)
    rows = []
    for d in trading:
        for code in CORE4:
            rows.append(
                {
                    "as_of": d.strftime("%Y-%m-%d"),
                    "strategy": "core4_equal",
                    "code": code,
                    "weight": w,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest core-4 vs GARCH strategies")
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--eval-start", type=str, default="2026-06-01")
    parser.add_argument("--eval-end", type=str, default="2026-07-13")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    pack_dir = args.pack_dir.expanduser().resolve()
    out_dir = (args.output_dir or pack_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pack_cfg = load_decision_pack_config()
    fat_cfg = fat_tail_config_from_raw(pack_cfg.raw.get("fat_tail"))
    board_cfg = garch_short_horizon_config_from_raw(pack_cfg.raw.get("garch_short_horizon"))

    as_of = load_as_of(pack_dir)
    holdings = load_holdings_from_pack(pack_dir)
    cluster_path = Path(CLUSTER_MAPPING_SELECTED_TXT)
    cluster_codes = load_cluster_mapping_codes(cluster_path) if cluster_path.exists() else ()
    defensive = frozenset(pack_cfg.defensive_codes)
    universe = build_universe_codes(holdings, cluster_codes, defensive)
    _, _, cluster_frame, _ = load_fat_tail_data(
        pack_dir,
        cluster_mapping_path=cluster_path,
        defensive_codes=defensive,
        cfg=fat_cfg,
    )
    flags_map = _flags_by_code(cluster_frame)

    start = warmup_start(as_of, fat_cfg)
    eval_end = pd.Timestamp(args.eval_end)
    panel = load_qlib_close(str(QLIB_PROVIDER_URI), universe, start, eval_end)
    panel, _ = auto_qfq_adjust_close_panel(panel)

    eval_start = pd.Timestamp(args.eval_start)
    trading = panel.index[(panel.index >= eval_start) & (panel.index <= eval_end)]
    if len(trading) < 2:
        print("[ERR] insufficient trading days", file=sys.stderr)
        return 1

    upside_df, avoid_df = collect_daily_picks(
        panel,
        cluster_frame,
        trading,
        horizon=int(args.horizon),
        k=int(args.k),
        cfg=board_cfg,
        flags_map=flags_map,
    )
    upside_path = out_dir / "garch_daily_picks_upside_skew.csv"
    avoid_path = out_dir / "garch_daily_picks_avoid_max_q50.csv"
    upside_df.to_csv(upside_path, index=False, encoding="utf-8-sig")
    avoid_df.to_csv(avoid_path, index=False, encoding="utf-8-sig")
    core_path = out_dir / "core4_equal_daily.csv"
    write_core4_daily(core_path, trading)

    panel_rets = panel.pct_change().shift(-1)
    panel_rets = panel_rets.loc[trading[:-1]]

    results: list[dict] = []
    ledger: dict[str, pd.DataFrame] = {}

    # Benchmark
    s = backtest_fixed_equal(panel_rets, CORE4)
    results.append(summarize("benchmark_core4_equal_bh", s))

    # GARCH daily rotation
    for label, df, strat in [
        ("garch_upside_skew_k6", upside_df, "upside_skew"),
        ("garch_avoid_max_q50_k6", avoid_df, "avoid_max_q50"),
        ("garch_upside_skew_k4", upside_df, "upside_skew"),
        ("garch_avoid_max_q50_k4", avoid_df, "avoid_max_q50"),
    ]:
        k = 6 if "_k6" in label else 4
        ser, led = backtest_from_pick_csv(df, panel_rets, strategy=strat, k=k)
        results.append(summarize(label, ser))
        ledger[label] = led

    # Restrict universe to core4 only (rank within the 4 names)
    for label, df, strat, k in [
        ("garch_upside_skew_core4_only_k4", upside_df, "upside_skew", 4),
        ("garch_avoid_core4_only_k4", avoid_df, "avoid_max_q50", 4),
    ]:
        ser, led = backtest_from_pick_csv(
            df, panel_rets, strategy=strat, k=k, universe_filter=frozenset(CORE4)
        )
        results.append(summarize(label, ser))
        ledger[label] = led

    # Intersection: daily picks filtered to names in core4 (equal weight among hits)
    for label, df, strat in [
        ("garch_upside_intersect_core4", upside_df, "upside_skew"),
        ("garch_avoid_intersect_core4", avoid_df, "avoid_max_q50"),
    ]:
        ser, led = backtest_from_pick_csv(
            df, panel_rets, strategy=strat, k=6, universe_filter=frozenset(CORE4)
        )
        results.append(summarize(label, ser))
        ledger[label] = led

    # Low turnover: keep prior picks if >=3 still in today's top6
    for label, df, strat in [
        ("garch_upside_sticky3", upside_df, "upside_skew"),
        ("garch_avoid_sticky3", avoid_df, "avoid_max_q50"),
    ]:
        ser, led = backtest_from_pick_csv(
            df, panel_rets, strategy=strat, k=6, min_overlap_hold=3
        )
        results.append(summarize(label, ser))
        ledger[label] = led

    # Hybrid sleeves
    for cw, label in [(0.8, "hybrid80_core4_20_upside"), (0.9, "hybrid90_core4_10_upside")]:
        ser = backtest_hybrid_core_satellite(panel_rets, upside_df, core_w=cw, strategy="upside_skew", k=6)
        results.append(summarize(label, ser))

    summary = pd.DataFrame(results)
    bench_ret = summary.loc[summary["strategy"] == "benchmark_core4_equal_bh", "cum_ret_pct"].iloc[0]
    summary["gap_vs_benchmark_pct"] = summary["cum_ret_pct"] - bench_ret
    summary = summary.sort_values("cum_ret_pct", ascending=False)

    summary_path = out_dir / "core4_vs_garch_backtest_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    cov_up = core4_coverage(upside_df, "upside_skew", k=6)
    cov_av = core4_coverage(avoid_df, "avoid_max_q50", k=6)
    cov_path = out_dir / "core4_garch_coverage.csv"
    pd.concat(
        [
            cov_up.assign(strategy="upside_skew"),
            cov_av.assign(strategy="avoid_max_q50"),
        ],
        ignore_index=True,
    ).to_csv(cov_path, index=False, encoding="utf-8-sig")

    for name, led in ledger.items():
        led.to_csv(out_dir / f"ledger_{name}.csv", index=False, encoding="utf-8-sig")

    print(f"[OK] upside picks -> {upside_path}")
    print(f"[OK] avoid picks -> {avoid_path}")
    print(f"[OK] core4 daily -> {core_path}")
    print(f"[OK] summary -> {summary_path}")
    print(f"[OK] coverage -> {cov_path}")
    print("\n=== Backtest summary (sorted by cum return) ===")
    print(summary.to_string(index=False))
    print("\n=== Core4 in GARCH top6 (mean) ===")
    print(f"upside_skew: {cov_up['core4_in_topk'].mean():.2f}/4 avg")
    print(f"avoid_max_q50: {cov_av['core4_in_topk'].mean():.2f}/4 avg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
