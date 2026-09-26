"""GARCH short-horizon price targets (TP/SL/median) from board metrics."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from etf_daily.lib.garch_dynamics import GarchFitResult
from etf_daily.lib.garch_short_horizon_board import (
    GarchShortHorizonConfig,
    GarchShortHorizonMetrics,
    compute_garch_short_horizon_metrics,
)
from etf_daily.lib.indicators import slice_through

GARCH_PRICE_TARGET_CSV = "garch_price_targets.csv"

GARCH_PRICE_TARGET_COLUMNS: tuple[str, ...] = (
    "as_of",
    "code",
    "name",
    "持仓",
    "wt",
    "H",
    "close",
    "tp_price",
    "p_tp",
    "sl_price",
    "p_sl",
    "median_price",
    "p_median",
    "p_timeout",
    "barrier_log",
    "q05_price",
    "q95_price",
    "DGP",
    "依据",
)


def price_from_log_return(close: float, log_ret: float | None) -> float | None:
    if log_ret is None or not math.isfinite(log_ret) or not math.isfinite(close) or close <= 0:
        return None
    return float(close * math.exp(float(log_ret)))


def _flag_tuple(flags: object) -> tuple[str, ...]:
    if flags is None or (isinstance(flags, float) and math.isnan(flags)):
        return ()
    text = str(flags).strip()
    if not text or text == "—":
        return ()
    return tuple(f for f in text.split("|") if f)


def build_price_target_row(
    *,
    as_of: str,
    code: str,
    name: str,
    weight: float,
    close: float | None,
    horizon: int,
    metrics: GarchShortHorizonMetrics,
) -> dict[str, Any]:
    holding = "持" if weight > 0 else "候选"
    p0 = close
    barrier = metrics.barrier
    tp = price_from_log_return(p0, barrier) if p0 is not None and barrier is not None else None
    sl = price_from_log_return(p0, -barrier) if p0 is not None and barrier is not None else None
    median = price_from_log_return(p0, metrics.q50) if p0 is not None else None
    q05_p = price_from_log_return(p0, metrics.q05) if p0 is not None else None
    q95_p = price_from_log_return(p0, metrics.q95) if p0 is not None else None

    return {
        "as_of": as_of,
        "code": code,
        "name": name,
        "持仓": holding,
        "wt": round(float(weight), 4),
        "H": int(horizon),
        "close": round(float(p0), 4) if p0 is not None and math.isfinite(p0) else None,
        "tp_price": round(tp, 4) if tp is not None else None,
        "p_tp": metrics.p_up,
        "sl_price": round(sl, 4) if sl is not None else None,
        "p_sl": metrics.p_down,
        "median_price": round(median, 4) if median is not None else None,
        "p_median": 0.5 if metrics.q50 is not None else None,
        "p_timeout": metrics.p_timeout,
        "barrier_log": barrier,
        "q05_price": round(q05_p, 4) if q05_p is not None else None,
        "q95_price": round(q95_p, 4) if q95_p is not None else None,
        "DGP": metrics.dgp_basis or "—",
        "依据": metrics.basis,
    }


def build_garch_price_targets_frame(
    cluster_frame: pd.DataFrame,
    close_panel: pd.DataFrame,
    as_of: str,
    *,
    board_cfg: GarchShortHorizonConfig,
    fit_cache: dict[str, GarchFitResult] | None = None,
    horizons: tuple[int, ...] | list[int] | None = None,
    max_horizon: int | None = None,
) -> pd.DataFrame:
    """Long-format price targets: one row per (code, H)."""
    if cluster_frame.empty:
        return pd.DataFrame(columns=list(GARCH_PRICE_TARGET_COLUMNS))

    end = pd.Timestamp(as_of)
    horizon_list = tuple(int(h) for h in (horizons or board_cfg.price_target_horizons))
    max_h = int(max_horizon) if max_horizon is not None else max(horizon_list)
    fit_cache = fit_cache or {}
    rows: list[dict[str, Any]] = []

    for _, row in cluster_frame.iterrows():
        code = str(row["code"])
        weight = float(row["weight"])
        name = str(row["name"])
        flag_tuple = _flag_tuple(row.get("flags"))

        if code in close_panel.columns:
            series = pd.to_numeric(close_panel[code], errors="coerce").dropna()
            available = slice_through(end, series)
            close_val = float(available.iloc[-1]) if len(available) else None
        else:
            series = pd.Series(dtype=float)
            close_val = None

        for horizon in horizon_list:
            if code in close_panel.columns and len(series) > 0:
                metrics = compute_garch_short_horizon_metrics(
                    series,
                    end,
                    horizon=int(horizon),
                    cfg=board_cfg,
                    flags=flag_tuple,
                    fit=fit_cache.get(code),
                    max_horizon=max_h,
                )
            else:
                metrics = GarchShortHorizonMetrics(
                    horizon=int(horizon),
                    sigma_garch_1d=None,
                    sigma_garch_h=None,
                    vol_ann_5d=None,
                    vol_ann_20d=None,
                    vol_ratio_5_20=None,
                    nu_t=None,
                    arch_pvalue=None,
                    p_up=None,
                    p_down=None,
                    p_timeout=None,
                    barrier=None,
                    q05=None,
                    q50=None,
                    q95=None,
                    var95_cond=None,
                    cvar95_cond=None,
                    dgp_basis="none",
                    reliable=False,
                    basis="缺价格·仅展示",
                )
            rows.append(
                build_price_target_row(
                    as_of=str(as_of),
                    code=code,
                    name=name,
                    weight=weight,
                    close=close_val,
                    horizon=int(horizon),
                    metrics=metrics,
                )
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=list(GARCH_PRICE_TARGET_COLUMNS))
    frame["_hold"] = frame["持仓"].eq("持")
    frame = frame.sort_values(by=["_hold", "code", "H"], ascending=[False, True, True])
    frame = frame.drop(columns=["_hold"])
    return frame.reindex(columns=list(GARCH_PRICE_TARGET_COLUMNS))


__all__ = [
    "GARCH_PRICE_TARGET_COLUMNS",
    "GARCH_PRICE_TARGET_CSV",
    "build_garch_price_targets_frame",
    "build_price_target_row",
    "price_from_log_return",
]
