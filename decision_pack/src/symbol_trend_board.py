"""Per-holding trend board for manual exit decisions (v0.3 display-only)."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from decision_pack.src.config import DecisionPackConfig, load_decision_pack_config
from decision_pack.src.indicators import (
    compute_trend_metrics,
    days_below_ma,
    realized_vol_annualized,
    vol_level_label,
    vol_percentile,
    window_return,
)
from pipeline.benchmark_regime import load_hs300_close
from pipeline.price_adjustments import auto_qfq_adjust_close_panel
from decision_pack.src.portfolio import HoldingRow, PortfolioSnapshot
from pipeline.strategy_layer_audit import load_qlib_close
from runtime_paths import CLUSTER_MAPPING_SELECTED_TXT, QLIB_PROVIDER_URI
from workspace.scripts.generate_daily_mu_position_report import load_fund_name_map, normalize_code

TREND_BOARD_COLUMNS = [
    "code",
    "name",
    "weight",
    "close",
    "ma5",
    "ma10",
    "ma20",
    "dist_ma20_pct",
    "ret_5d",
    "ret_20d",
    "ma_stack",
    "vol_pct_120d",
    "vol_ann_20d",
    "vol_ann_5d",
    "vol_ratio_5_20",
    "vol_level",
    "dd_20d_high",
    "pnl_pct",
    "days_below_ma20",
    "rs_20d_vs_bench",
    "risk_flags",
]

_METRIC_COLUMNS = TREND_BOARD_COLUMNS[3:]


@dataclass(frozen=True)
class TrendBoardVolContext:
    bench_vol_pct: float | None
    bench_vol_ann: float | None
    bench_vol_level: str | None
    equity_count: int
    equity_high_vol_count: int
    equity_below_ma20_count: int


@dataclass(frozen=True)
class TrendBoardResult:
    board: pd.DataFrame
    vol_context: TrendBoardVolContext


def _trend_board_thresholds(config: DecisionPackConfig) -> dict[str, float]:
    raw = config.raw.get("trend_board") or {}
    if not isinstance(raw, dict):
        raw = {}
    ew = config.raw.get("early_warning") or {}
    if not isinstance(ew, dict):
        ew = {}
    return {
        "heavy_weight": float(raw.get("heavy_weight", 0.10)),
        "high_vol_pct": float(raw.get("high_vol_pct", 0.80)),
        "vol_level_low_max": float(raw.get("vol_level_low_max", 0.33)),
        "vol_level_mid_max": float(raw.get("vol_level_mid_max", 0.80)),
        "deep_below_ma20_pct": float(raw.get("deep_below_ma20_pct", -5.0)),
        "weak_rs_20d": float(raw.get("weak_rs_20d", -0.05)),
        "stale_below_ma20_days": float(raw.get("stale_below_ma20_days", 3)),
        "red_dd_20d_high": float(raw.get("red_dd_20d_high", ew.get("red_dd_20d_high", -0.05))),
        "red_pnl_pct": float(raw.get("red_pnl_pct", ew.get("red_pnl_pct", -7.0))),
        "vol_accel_ratio": float(raw.get("vol_accel_ratio", ew.get("vol_accel_ratio", 1.2))),
    }


def _apply_vol_level(
    metrics: dict[str, Any],
    thresholds: dict[str, float],
) -> None:
    metrics["vol_level"] = vol_level_label(
        metrics.get("vol_pct_120d"),
        low_max=thresholds["vol_level_low_max"],
        mid_max=thresholds["vol_level_mid_max"],
    )


def build_risk_flags(
    *,
    dist_ma20_pct: float | None,
    vol_pct_120d: float | None,
    weight: float,
    pnl_pct: float | None,
    ma_stack: str | None,
    rs_20d_vs_bench: float | None,
    days_below_ma20: int | None,
    ret_5d: float | None = None,
    ret_20d: float | None = None,
    vol_ratio_5_20: float | None = None,
    dd_20d_high: float | None = None,
    thresholds: dict[str, float],
) -> str:
    flags: list[str] = []
    if dist_ma20_pct is not None and dist_ma20_pct < 0:
        flags.append("below_ma20")
    if dist_ma20_pct is not None and dist_ma20_pct < thresholds["deep_below_ma20_pct"]:
        flags.append("deep_below_ma20")
    stale_days = int(thresholds["stale_below_ma20_days"])
    if days_below_ma20 is not None and days_below_ma20 >= stale_days:
        flags.append(f"below_ma20_{stale_days}d+")
    if vol_pct_120d is not None and vol_pct_120d >= thresholds["high_vol_pct"]:
        flags.append("high_vol")
    if weight >= thresholds["heavy_weight"]:
        flags.append("heavy_wt")
    if pnl_pct is not None and pnl_pct < 0:
        flags.append("loss")
    if pnl_pct is not None and pnl_pct <= thresholds["red_pnl_pct"]:
        flags.append("pnl7")
    if ma_stack == "bear":
        flags.append("bear_stack")
    if rs_20d_vs_bench is not None and rs_20d_vs_bench < thresholds["weak_rs_20d"]:
        flags.append("weak_rs")
    if dd_20d_high is not None and dd_20d_high <= thresholds["red_dd_20d_high"]:
        flags.append("dd5_20d")
    if ret_5d is not None and ret_20d is not None and ret_5d < ret_20d:
        flags.append("mom_fade")
    if vol_ratio_5_20 is not None and vol_ratio_5_20 > thresholds["vol_accel_ratio"]:
        flags.append("vol_accel")
    return "|".join(flags)


def _bench_vol_context(
    bench: pd.Series | None,
    end: pd.Timestamp,
    cfg: DecisionPackConfig,
    thresholds: dict[str, float],
) -> tuple[float | None, float | None, str | None]:
    if bench is None or bench.empty:
        return None, None, None
    raw_pct = vol_percentile(
        bench,
        end,
        vol_window=cfg.vol_window,
        lookback=cfg.vol_lookback,
    )
    raw_ann = realized_vol_annualized(bench, end, vol_window=cfg.vol_window)
    bench_pct = float(raw_pct) if pd.notna(raw_pct) else None
    bench_ann = float(raw_ann) if pd.notna(raw_ann) else None
    bench_level = vol_level_label(
        bench_pct,
        low_max=thresholds["vol_level_low_max"],
        mid_max=thresholds["vol_level_mid_max"],
    )
    return bench_pct, bench_ann, bench_level


def _summarize_board_vol(
    frame: pd.DataFrame,
    *,
    bench_vol_pct: float | None,
    bench_vol_ann: float | None,
    bench_vol_level: str | None,
    high_vol_pct: float,
) -> TrendBoardVolContext:
    if frame.empty:
        return TrendBoardVolContext(
            bench_vol_pct=bench_vol_pct,
            bench_vol_ann=bench_vol_ann,
            bench_vol_level=bench_vol_level,
            equity_count=0,
            equity_high_vol_count=0,
            equity_below_ma20_count=0,
        )
    vols = pd.to_numeric(frame["vol_pct_120d"], errors="coerce")
    dists = pd.to_numeric(frame["dist_ma20_pct"], errors="coerce")
    return TrendBoardVolContext(
        bench_vol_pct=bench_vol_pct,
        bench_vol_ann=bench_vol_ann,
        bench_vol_level=bench_vol_level,
        equity_count=len(frame),
        equity_high_vol_count=int((vols >= high_vol_pct).sum()),
        equity_below_ma20_count=int((dists < 0).sum()),
    )


def defensive_codes(config: DecisionPackConfig | None = None) -> frozenset[str]:
    cfg = config or load_decision_pack_config()
    return cfg.defensive_codes


def _equity_holdings(
    portfolio: PortfolioSnapshot,
    *,
    config: DecisionPackConfig | None = None,
) -> tuple[HoldingRow, ...]:
    skip = {normalize_code(code) for code in defensive_codes(config)}
    rows = []
    for holding in portfolio.holdings:
        code = normalize_code(holding.code)
        if code in skip or code.upper() == "CASH":
            continue
        rows.append(holding)
    return tuple(rows)


def _warmup_start(as_of: str, config: DecisionPackConfig) -> str:
    end = pd.Timestamp(as_of)
    days = config.vol_lookback + config.vol_window + config.ma_window + 40
    return (end - pd.tseries.offsets.BDay(days)).strftime("%Y-%m-%d")


def _pnl_pct(close: float | None, cost_price: float | None) -> float | None:
    if close is None or cost_price is None or cost_price <= 0:
        return None
    return (close / cost_price - 1.0) * 100.0


def load_cluster_mapping_codes(path: Path | str) -> tuple[str, ...]:
    """Read instrument codes from a qlib instruments txt (first tab column)."""
    codes: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.split("\t")[0].strip()
            if token:
                codes.append(normalize_code(token))
    seen: set[str] = set()
    ordered: list[str] = []
    for code in codes:
        if code not in seen:
            seen.add(code)
            ordered.append(code)
    return tuple(ordered)


def _holdings_for_cluster_universe(
    portfolio: PortfolioSnapshot,
    cluster_codes: tuple[str, ...],
    *,
    fund_list_csv: Path | None = None,
) -> tuple[HoldingRow, ...]:
    name_map = load_fund_name_map(fund_list_csv) if fund_list_csv and fund_list_csv.exists() else {}
    by_code: dict[str, HoldingRow] = {}
    for holding in portfolio.holdings:
        code = normalize_code(holding.code)
        if code.upper() == "CASH":
            continue
        by_code[code] = holding

    rows: list[HoldingRow] = []
    for code in cluster_codes:
        if code in by_code:
            rows.append(by_code[code])
        else:
            rows.append(
                HoldingRow(
                    code=code,
                    name=name_map.get(code, code),
                    market_value=0.0,
                    weight=0.0,
                    weight_total=0.0,
                    source=portfolio.source,
                )
            )
    return tuple(rows)


def _build_trend_board_from_holdings(
    holdings: tuple[HoldingRow, ...],
    as_of: str,
    *,
    config: DecisionPackConfig | None = None,
    provider_uri: str | Path | None = None,
    close_panel: pd.DataFrame | None = None,
    benchmark_close: pd.Series | None = None,
    sort_by: str = "dist_ma20_pct",
) -> TrendBoardResult:
    cfg = config or load_decision_pack_config()
    end = pd.Timestamp(as_of)
    thresholds = _trend_board_thresholds(cfg)

    if not holdings:
        empty = pd.DataFrame(columns=TREND_BOARD_COLUMNS)
        return TrendBoardResult(
            board=empty,
            vol_context=_summarize_board_vol(
                empty,
                bench_vol_pct=None,
                bench_vol_ann=None,
                bench_vol_level=None,
                high_vol_pct=thresholds["high_vol_pct"],
            ),
        )

    codes = [normalize_code(row.code) for row in holdings]
    panel = close_panel
    if panel is not None:
        panel, _ = auto_qfq_adjust_close_panel(panel)
    if panel is None:
        uri = str(provider_uri or QLIB_PROVIDER_URI)
        panel = load_qlib_close(uri, codes, _warmup_start(as_of, cfg), as_of)

    bench = benchmark_close
    if bench is None and close_panel is None:
        uri = str(provider_uri or QLIB_PROVIDER_URI)
        bench = load_hs300_close(uri, _warmup_start(as_of, cfg), as_of)

    bench_vol_pct, bench_vol_ann, bench_vol_level = _bench_vol_context(bench, end, cfg, thresholds)

    bench_ret_20d: float | None = None
    if bench is not None and not bench.empty:
        raw_bench_ret = window_return(bench, end, 20)
        bench_ret_20d = raw_bench_ret if pd.notna(raw_bench_ret) else None

    rows: list[dict[str, Any]] = []
    for holding in holdings:
        code = normalize_code(holding.code)
        metrics: dict[str, Any] = {key: None for key in _METRIC_COLUMNS}
        if code not in panel.columns:
            if holding.close_price is not None:
                metrics["close"] = holding.close_price
        else:
            series = pd.to_numeric(panel[code], errors="coerce").dropna()
            metrics = compute_trend_metrics(
                series,
                end,
                vol_window=cfg.vol_window,
                vol_lookback=cfg.vol_lookback,
                dd_lookback=20,
            )
            if metrics.get("close") is None and holding.close_price is not None:
                metrics["close"] = holding.close_price
            metrics["days_below_ma20"] = days_below_ma(series, end, cfg.ma_window)

        _apply_vol_level(metrics, thresholds)

        close = metrics.get("close")
        metrics["pnl_pct"] = _pnl_pct(close, holding.cost_price)

        ret_20d = metrics.get("ret_20d")
        if ret_20d is not None and bench_ret_20d is not None:
            metrics["rs_20d_vs_bench"] = ret_20d - bench_ret_20d

        metrics["risk_flags"] = build_risk_flags(
            dist_ma20_pct=metrics.get("dist_ma20_pct"),
            vol_pct_120d=metrics.get("vol_pct_120d"),
            weight=holding.weight_total,
            pnl_pct=metrics.get("pnl_pct"),
            ma_stack=metrics.get("ma_stack"),
            rs_20d_vs_bench=metrics.get("rs_20d_vs_bench"),
            days_below_ma20=metrics.get("days_below_ma20"),
            ret_5d=metrics.get("ret_5d"),
            ret_20d=metrics.get("ret_20d"),
            vol_ratio_5_20=metrics.get("vol_ratio_5_20"),
            dd_20d_high=metrics.get("dd_20d_high"),
            thresholds=thresholds,
        )

        rows.append(
            {
                "code": holding.code,
                "name": holding.name,
                "weight": holding.weight_total,
                **{key: metrics.get(key) for key in _METRIC_COLUMNS},
            }
        )

    frame = pd.DataFrame(rows, columns=TREND_BOARD_COLUMNS)
    if sort_by in frame.columns and not frame.empty:
        frame = frame.sort_values(sort_by, ascending=True, na_position="last").reset_index(drop=True)

    vol_context = _summarize_board_vol(
        frame,
        bench_vol_pct=bench_vol_pct,
        bench_vol_ann=bench_vol_ann,
        bench_vol_level=bench_vol_level,
        high_vol_pct=thresholds["high_vol_pct"],
    )
    return TrendBoardResult(board=frame, vol_context=vol_context)


def build_trend_board(
    portfolio: PortfolioSnapshot,
    *,
    config: DecisionPackConfig | None = None,
    provider_uri: str | Path | None = None,
    close_panel: pd.DataFrame | None = None,
    benchmark_close: pd.Series | None = None,
    sort_by: str = "dist_ma20_pct",
) -> TrendBoardResult:
    """Build holdings trend board for equity sleeve holdings."""
    cfg = config or load_decision_pack_config()
    holdings = _equity_holdings(portfolio, config=cfg)
    return _build_trend_board_from_holdings(
        holdings,
        portfolio.as_of,
        config=cfg,
        provider_uri=provider_uri,
        close_panel=close_panel,
        benchmark_close=benchmark_close,
        sort_by=sort_by,
    )


def build_cluster_trend_board(
    portfolio: PortfolioSnapshot,
    cluster_mapping_path: Path | str | None = None,
    *,
    fund_list_csv: Path | None = None,
    config: DecisionPackConfig | None = None,
    provider_uri: str | Path | None = None,
    close_panel: pd.DataFrame | None = None,
    benchmark_close: pd.Series | None = None,
    sort_by: str = "dist_ma20_pct",
) -> TrendBoardResult:
    """Build trend board for every symbol in cluster_mapping_selected.txt."""
    path = Path(cluster_mapping_path or CLUSTER_MAPPING_SELECTED_TXT)
    cluster_codes = load_cluster_mapping_codes(path)
    holdings = _holdings_for_cluster_universe(
        portfolio,
        cluster_codes,
        fund_list_csv=fund_list_csv,
    )
    return _build_trend_board_from_holdings(
        holdings,
        portfolio.as_of,
        config=config,
        provider_uri=provider_uri,
        close_panel=close_panel,
        benchmark_close=benchmark_close,
        sort_by=sort_by,
    )


__all__ = [
    "TREND_BOARD_COLUMNS",
    "TrendBoardResult",
    "TrendBoardVolContext",
    "build_cluster_trend_board",
    "build_risk_flags",
    "build_trend_board",
    "defensive_codes",
    "load_cluster_mapping_codes",
]
