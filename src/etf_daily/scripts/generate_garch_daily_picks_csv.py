#!/usr/bin/env python3
"""Daily GARCH top-K pick ledgers with forward 1-day returns."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

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
from etf_daily.scripts.validate_garch_short_horizon_board import _flags_by_code
from etf_daily.lib.config import load_decision_pack_config
from etf_daily.lib.fat_tail_risk import fat_tail_config_from_raw
from etf_daily.lib.garch_short_horizon_board import (
    GarchShortHorizonConfig,
    GarchShortHorizonMetrics,
    compute_garch_short_horizon_metrics,
    garch_short_horizon_config_from_raw,
)
from etf_daily.lib.short_horizon_mlfam import forward_log_returns
from etf_daily.lib.symbol_trend_board import load_cluster_mapping_codes
from etf_daily.pipeline.price_adjustments import auto_qfq_adjust_close_panel
from etf_daily.pipeline.strategy_layer_audit import load_qlib_close
from etf_daily.paths import CLUSTER_MAPPING_SELECTED_TXT, QLIB_PROVIDER_URI

DEFAULT_K = 6
DEFAULT_H = 10


def _pct(x: float | None, digits: int = 4) -> float | None:
    if x is None or not math.isfinite(float(x)):
        return None
    return round(float(x) * 100.0, digits)


def _next_day_simple_return(series: pd.Series, as_of: pd.Timestamp) -> float | None:
    fwd = forward_log_returns(series, as_of, horizon=1)
    if fwd is None or fwd.size != 1:
        return None
    return round(math.expm1(float(fwd[0])) * 100.0, 4)


def build_board_row(
    code: str,
    name: str,
    leg: str,
    metrics: GarchShortHorizonMetrics,
    *,
    as_of: pd.Timestamp,
    close: float,
    next_ret_pct: float | None,
    rank: int,
    strategy: str,
) -> dict:
    q05 = metrics.q05
    q95 = metrics.q95
    q50 = metrics.q50
    upside = None
    if q95 is not None and q05 is not None:
        upside = float(q95) - abs(float(q05))
    row = {
        "as_of": as_of.strftime("%Y-%m-%d"),
        "strategy": strategy,
        "rank": rank,
        "code": code,
        "name": name,
        "leg": leg,
        "H": metrics.horizon,
        "close": round(float(close), 4),
        "q05_H_pct": _pct(q05),
        "q50_H_pct": _pct(q50),
        "q95_H_pct": _pct(q95),
        "upside_skew_pct": _pct(upside),
        "p_tp_pct": _pct(metrics.p_up),
        "p_sl_pct": _pct(metrics.p_down),
        "p_timeout_pct": _pct(metrics.p_timeout),
        "next_day_ret_pct": next_ret_pct,
    }
    return row


def collect_daily_picks(
    panel: pd.DataFrame,
    cluster_frame: pd.DataFrame,
    eval_dates: pd.DatetimeIndex,
    *,
    horizon: int,
    k: int,
    cfg: GarchShortHorizonConfig,
    flags_map: dict[str, tuple[str, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_by_code = cluster_frame.set_index("code")
    name_map = cluster_frame.set_index("code")["name"].astype(str).to_dict()
    leg_map = cluster_frame.set_index("code")["barbell_leg"].astype(str).to_dict()

    upside_rows: list[dict] = []
    avoid_rows: list[dict] = []

    for end in eval_dates:
        ts_end = pd.Timestamp(end)
        candidates: list[dict] = []
        for code in panel.columns:
            if code not in cluster_by_code.index:
                continue
            series = pd.to_numeric(panel[code], errors="coerce").dropna()
            if ts_end not in series.index and ts_end > series.index.max():
                continue
            available = series.loc[series.index <= ts_end].dropna()
            if available.empty:
                continue
            close = float(available.iloc[-1])
            if close <= 0:
                continue
            metrics = compute_garch_short_horizon_metrics(
                series,
                ts_end,
                horizon=horizon,
                cfg=cfg,
                flags=flags_map.get(str(code), ()),
                max_horizon=horizon,
            )
            if not metrics.reliable or metrics.q50 is None or metrics.q05 is None or metrics.q95 is None:
                continue
            next_ret = _next_day_simple_return(series, ts_end)
            upside = float(metrics.q95) - abs(float(metrics.q05))
            candidates.append(
                {
                    "code": str(code),
                    "name": name_map.get(str(code), str(code)),
                    "leg": leg_map.get(str(code), "—"),
                    "metrics": metrics,
                    "close": close,
                    "next_ret": next_ret,
                    "upside_skew": upside,
                    "q50": float(metrics.q50),
                }
            )

        if not candidates:
            continue

        cand_df = pd.DataFrame(candidates)
        upside_pick = cand_df.sort_values("upside_skew", ascending=False).head(k)
        avoid_pool = cand_df[cand_df["leg"] == "avoid"].sort_values("q50", ascending=False).head(k)

        for i, row in enumerate(upside_pick.itertuples(), start=1):
            upside_rows.append(
                build_board_row(
                    row.code,
                    row.name,
                    row.leg,
                    row.metrics,
                    as_of=ts_end,
                    close=row.close,
                    next_ret_pct=row.next_ret,
                    rank=i,
                    strategy="upside_skew",
                )
            )
        for i, row in enumerate(avoid_pool.itertuples(), start=1):
            avoid_rows.append(
                build_board_row(
                    row.code,
                    row.name,
                    row.leg,
                    row.metrics,
                    as_of=ts_end,
                    close=row.close,
                    next_ret_pct=row.next_ret,
                    rank=i,
                    strategy="avoid_max_q50",
                )
            )

    upside_cols = [
        "as_of",
        "strategy",
        "rank",
        "code",
        "name",
        "leg",
        "H",
        "close",
        "q05_H_pct",
        "q95_H_pct",
        "upside_skew_pct",
        "q50_H_pct",
        "p_tp_pct",
        "p_sl_pct",
        "p_timeout_pct",
        "next_day_ret_pct",
    ]
    avoid_cols = [
        "as_of",
        "strategy",
        "rank",
        "code",
        "name",
        "leg",
        "H",
        "close",
        "q50_H_pct",
        "q05_H_pct",
        "q95_H_pct",
        "upside_skew_pct",
        "p_tp_pct",
        "p_sl_pct",
        "p_timeout_pct",
        "next_day_ret_pct",
    ]
    return pd.DataFrame(upside_rows).reindex(columns=upside_cols), pd.DataFrame(avoid_rows).reindex(
        columns=avoid_cols
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate daily GARCH pick CSVs")
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--eval-start", type=str, default="2026-06-01")
    parser.add_argument("--eval-end", type=str, default="2026-07-14")
    parser.add_argument("--horizon", type=int, default=DEFAULT_H)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--output-upside", type=Path, default=None)
    parser.add_argument("--output-avoid", type=Path, default=None)
    args = parser.parse_args(argv)

    pack_dir = args.pack_dir.expanduser().resolve()
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
    panel = load_qlib_close(str(QLIB_PROVIDER_URI), universe, start, as_of)
    panel, _ = auto_qfq_adjust_close_panel(panel)

    eval_start = pd.Timestamp(args.eval_start)
    eval_end = pd.Timestamp(args.eval_end)
    trading = panel.index[(panel.index >= eval_start) & (panel.index <= eval_end)]
    if trading.empty:
        print("[ERR] no trading dates", file=sys.stderr)
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

    out_up = args.output_upside or (pack_dir / "garch_daily_picks_upside_skew.csv")
    out_av = args.output_avoid or (pack_dir / "garch_daily_picks_avoid_max_q50.csv")
    upside_df.to_csv(out_up, index=False, encoding="utf-8-sig")
    avoid_df.to_csv(out_av, index=False, encoding="utf-8-sig")
    print(f"[OK] upside_skew -> {out_up} ({len(upside_df)} rows, {upside_df['as_of'].nunique()} days)")
    print(f"[OK] avoid_max_q50 -> {out_av} ({len(avoid_df)} rows, {avoid_df['as_of'].nunique()} days)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
