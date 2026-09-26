"""GARCH conditional-volatility short-horizon board metrics (ftsnotes §18–21, §24.1)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from decision_pack.src.fat_tail_risk import ANNUALIZE, slice_log_returns
from decision_pack.src.garch_dynamics import (
    GarchDynamicsConfig,
    GarchFitResult,
    conditional_var_cvar,
    fit_garch_dynamics,
    forecast_sigma_h,
    garch_dynamics_config_from_raw,
    mc_quantiles_garch,
    triple_barrier_mc_garch,
    vol_spike_prob_mc_garch,
)
from decision_pack.src.indicators import slice_through


@dataclass(frozen=True)
class GarchShortHorizonConfig:
    horizons: tuple[int, ...] = (5, 10, 20)
    price_target_horizons: tuple[int, ...] = (1, 5, 10, 20)
    garch: GarchDynamicsConfig = field(default_factory=GarchDynamicsConfig)
    min_samples: int = 60
    vol5_lookback: int = 120
    spike_mult: float = 1.2


@dataclass(frozen=True)
class GarchShortHorizonMetrics:
    horizon: int
    sigma_garch_1d: float | None
    sigma_garch_h: float | None
    vol_ann_5d: float | None
    vol_ann_20d: float | None
    vol_ratio_5_20: float | None
    nu_t: float | None
    arch_pvalue: float | None
    p_up: float | None
    p_down: float | None
    p_timeout: float | None
    barrier: float | None
    q05: float | None
    q50: float | None
    q95: float | None
    var95_cond: float | None
    cvar95_cond: float | None
    dgp_basis: str
    reliable: bool
    basis: str
    vol5_pct_120d: float | None = None
    p_vol_spike_h5: float | None = None
    n_samples: int = 0


def garch_short_horizon_config_from_raw(raw: dict[str, Any] | None) -> GarchShortHorizonConfig:
    if not raw or not isinstance(raw, dict):
        return GarchShortHorizonConfig()
    horizons_raw = raw.get("horizons")
    horizons = tuple(int(h) for h in horizons_raw) if horizons_raw else (5, 10, 20)
    pt_raw = raw.get("price_target_horizons")
    price_target_horizons = (
        tuple(int(h) for h in pt_raw) if pt_raw else (1, 5, 10, 20)
    )
    garch_cfg = garch_dynamics_config_from_raw(raw)
    min_samples = int(raw.get("min_samples", 60))
    vol5_lookback = int(raw.get("vol5_lookback", 120))
    spike_mult = float(raw.get("spike_mult", 1.2))
    return GarchShortHorizonConfig(
        horizons=horizons,
        price_target_horizons=price_target_horizons,
        garch=garch_cfg,
        min_samples=min_samples,
        vol5_lookback=vol5_lookback,
        spike_mult=spike_mult,
    )


def _vol_ratio(vol5: float | None, vol20: float | None) -> float | None:
    if vol5 is None or vol20 is None or vol20 <= 0:
        return None
    if not math.isfinite(vol5) or not math.isfinite(vol20):
        return None
    return vol5 / vol20


def realized_vol_annualized_log(
    close: pd.Series,
    end: pd.Timestamp,
    *,
    vol_window: int,
) -> float:
    """Annualized realized vol from log returns (aligned with GARCH return type)."""
    available = slice_through(end, close)
    if len(available) < vol_window + 1:
        return float("nan")
    prices = pd.to_numeric(available, errors="coerce").dropna()
    if len(prices) < vol_window + 1:
        return float("nan")
    log_rets = np.diff(np.log(prices.to_numpy(dtype=float)))
    if log_rets.size < vol_window:
        return float("nan")
    window = log_rets[-vol_window:]
    std = float(np.std(window, ddof=1))
    if not math.isfinite(std):
        return float("nan")
    return std * ANNUALIZE


def realized_vol_percentile_log(
    close: pd.Series,
    end: pd.Timestamp,
    *,
    vol_window: int = 5,
    lookback: int = 120,
) -> float:
    """Percentile of current rolling vol (log returns) within trailing history."""
    available = slice_through(end, close)
    if len(available) < vol_window + 5:
        return float("nan")
    prices = pd.to_numeric(available, errors="coerce").dropna()
    if len(prices) < vol_window + 5:
        return float("nan")
    log_rets = np.diff(np.log(prices.to_numpy(dtype=float)))
    if log_rets.size < vol_window + 5:
        return float("nan")
    series = pd.Series(log_rets)
    realized = series.rolling(vol_window).std(ddof=1) * ANNUALIZE
    realized = realized.dropna()
    if realized.empty:
        return float("nan")
    current = float(realized.iloc[-1])
    if not math.isfinite(current):
        return float("nan")
    history = realized.tail(lookback)
    if len(history) < 5:
        return float("nan")
    return float((history <= current).sum()) / float(len(history))


def metrics_from_fit(
    fit: GarchFitResult,
    *,
    horizon: int,
    cfg: GarchShortHorizonConfig,
    n_samples: int,
    vol_ann_5: float,
    vol_ann_20: float,
    vol_ratio: float | None,
    vol5_pct: float | None,
    flag_set: set[str],
) -> GarchShortHorizonMetrics:
    """Build horizon metrics from a precomputed GARCH fit (shared across H)."""
    gcfg = cfg.garch
    caution: str | None = None
    if "alpha_unreliable" in flag_set:
        caution = "α不可靠"
    elif "low_n" in flag_set or n_samples < cfg.min_samples:
        caution = "低样本"
    elif not fit.reliable:
        caution = "GARCH未收敛"

    reliable = caution is None and fit.sigma_1d is not None

    var95, cvar95 = conditional_var_cvar(fit, alpha=0.05)
    sigma_h = forecast_sigma_h(fit, horizon)

    p_up, p_down, p_timeout, barrier = triple_barrier_mc_garch(
        fit,
        horizon=horizon,
        barrier_mult=gcfg.barrier_mult,
        n_paths=gcfg.mc_samples,
        seed=gcfg.mc_seed + horizon,
    )
    q05, q50, q95 = mc_quantiles_garch(
        fit,
        horizon=horizon,
        n_paths=gcfg.mc_samples,
        seed=gcfg.mc_seed + horizon + 1000,
    )

    p_vol_spike: float | None = None
    if (
        reliable
        and vol_ann_5 is not None
        and math.isfinite(vol_ann_5)
        and vol_ann_5 > 0
        and horizon >= 2
    ):
        p_vol_spike = vol_spike_prob_mc_garch(
            fit,
            horizon=horizon,
            baseline_vol_ann=vol_ann_5,
            spike_mult=cfg.spike_mult,
            n_paths=gcfg.mc_samples,
            seed=gcfg.mc_seed + horizon + 2000,
        )

    basis = f"GARCH-DGP·{fit.dgp_basis or fit.model}"
    if caution:
        basis = f"{basis}·{caution}"

    return GarchShortHorizonMetrics(
        horizon=horizon,
        sigma_garch_1d=fit.sigma_1d,
        sigma_garch_h=sigma_h,
        vol_ann_5d=vol_ann_5 if math.isfinite(vol_ann_5) else None,
        vol_ann_20d=vol_ann_20 if math.isfinite(vol_ann_20) else None,
        vol_ratio_5_20=vol_ratio,
        nu_t=fit.nu,
        arch_pvalue=fit.arch_pvalue,
        p_up=p_up,
        p_down=p_down,
        p_timeout=p_timeout,
        barrier=barrier,
        q05=q05,
        q50=q50,
        q95=q95,
        var95_cond=var95,
        cvar95_cond=cvar95,
        dgp_basis=fit.dgp_basis or fit.model,
        reliable=reliable,
        basis=basis,
        vol5_pct_120d=vol5_pct if vol5_pct is not None and math.isfinite(vol5_pct) else None,
        p_vol_spike_h5=p_vol_spike,
        n_samples=n_samples,
    )


def compute_garch_short_horizon_metrics(
    close: pd.Series,
    end: pd.Timestamp,
    *,
    horizon: int,
    cfg: GarchShortHorizonConfig | None = None,
    flags: tuple[str, ...] | str = (),
    fit: GarchFitResult | None = None,
    max_horizon: int | None = None,
) -> GarchShortHorizonMetrics:
    cfg = cfg or GarchShortHorizonConfig()
    gcfg = cfg.garch
    if isinstance(flags, str):
        flag_set = {f for f in flags.split("|") if f}
    else:
        flag_set = {str(f) for f in (flags or ())}

    if "missing_price" in flag_set:
        return GarchShortHorizonMetrics(
            horizon=horizon,
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
            vol5_pct_120d=None,
            p_vol_spike_h5=None,
            n_samples=0,
        )

    returns = slice_log_returns(close, end, lookback_days=gcfg.train_window)
    n = int(returns.size)

    vol_ann_5 = realized_vol_annualized_log(close, end, vol_window=5)
    vol_ann_20 = realized_vol_annualized_log(close, end, vol_window=20)
    vol5 = vol_ann_5 / ANNUALIZE if math.isfinite(vol_ann_5) else None
    vol20 = vol_ann_20 / ANNUALIZE if math.isfinite(vol_ann_20) else None
    vol_ratio = _vol_ratio(vol5, vol20)
    vol5_pct = realized_vol_percentile_log(
        close,
        end,
        vol_window=5,
        lookback=cfg.vol5_lookback,
    )

    if n < 20:
        return GarchShortHorizonMetrics(
            horizon=horizon,
            sigma_garch_1d=None,
            sigma_garch_h=None,
            vol_ann_5d=vol_ann_5 if math.isfinite(vol_ann_5) else None,
            vol_ann_20d=vol_ann_20 if math.isfinite(vol_ann_20) else None,
            vol_ratio_5_20=vol_ratio,
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
            basis="数据不足·仅展示",
            vol5_pct_120d=vol5_pct if math.isfinite(vol5_pct) else None,
            p_vol_spike_h5=None,
            n_samples=n,
        )

    if fit is None:
        max_h = int(max_horizon) if max_horizon is not None else max(
            int(horizon), max(cfg.horizons, default=horizon)
        )
        fit = fit_garch_dynamics(returns, gcfg, max_horizon=max_h)

    return metrics_from_fit(
        fit,
        horizon=horizon,
        cfg=cfg,
        n_samples=n,
        vol_ann_5=vol_ann_5,
        vol_ann_20=vol_ann_20,
        vol_ratio=vol_ratio,
        vol5_pct=vol5_pct if math.isfinite(vol5_pct) else None,
        flag_set=flag_set,
    )


GARCH_SHORT_HORIZON_COLUMNS: tuple[str, ...] = (
    "code",
    "name",
    "持仓",
    "wt",
    "n",
    "H",
    "σ_garch_1d",
    "σ_garch_H",
    "vol_ann_5d",
    "vol_ann_20d",
    "vol_ratio_5_20",
    "vol5_pct_120d",
    "p_vol_spike_h5",
    "arch_p",
    "VaR95_cond",
    "CVaR95_cond",
    "触轨↑",
    "触轨↓",
    "到期",
    "q05_H",
    "q50_H",
    "q95_H",
    "障宽",
    "leg",
    "flags",
    "尾结构",
    "状态",
    "回撤长",
    "DGP",
    "依据",
)


def format_garch_row(
    *,
    code: str,
    name: str,
    weight: float,
    cluster_row: pd.Series,
    metrics: GarchShortHorizonMetrics,
    formatters: dict[str, Any],
) -> dict[str, str]:
    """Build one CSV row dict using injected formatters."""
    pct = formatters["pct"]
    prob = formatters["prob"]
    num = formatters["num"]
    pval = formatters["pval"]
    flags_str = formatters["flags_str"]
    tail_short = formatters["tail_short"](cluster_row.get("tail_regime"))
    state_short = formatters["state_short"](cluster_row.get("path_state"))

    return {
        "code": code,
        "name": name,
        "持仓": "持" if weight > 0 else "候选",
        "wt": f"{weight:.4f}",
        "n": str(metrics.n_samples),
        "H": str(metrics.horizon),
        "σ_garch_1d": pct(metrics.sigma_garch_1d),
        "σ_garch_H": pct(metrics.sigma_garch_h),
        "vol_ann_5d": pct(metrics.vol_ann_5d),
        "vol_ann_20d": pct(metrics.vol_ann_20d),
        "vol_ratio_5_20": num(metrics.vol_ratio_5_20, digits=2),
        "vol5_pct_120d": prob(metrics.vol5_pct_120d),
        "p_vol_spike_h5": prob(metrics.p_vol_spike_h5),
        "arch_p": pval(metrics.arch_pvalue),
        "VaR95_cond": pct(metrics.var95_cond),
        "CVaR95_cond": pct(metrics.cvar95_cond),
        "触轨↑": prob(metrics.p_up),
        "触轨↓": prob(metrics.p_down),
        "到期": prob(metrics.p_timeout),
        "q05_H": pct(metrics.q05),
        "q50_H": pct(metrics.q50),
        "q95_H": pct(metrics.q95),
        "障宽": pct(metrics.barrier),
        "leg": str(cluster_row.get("barbell_leg") or "—"),
        "flags": flags_str(cluster_row.get("flags")),
        "尾结构": tail_short,
        "状态": state_short,
        "回撤长": pct(cluster_row.get("dd_long"), digits=1),
        "DGP": metrics.dgp_basis or "—",
        "依据": metrics.basis,
    }


__all__ = [
    "GARCH_SHORT_HORIZON_COLUMNS",
    "GarchShortHorizonConfig",
    "GarchShortHorizonMetrics",
    "compute_garch_short_horizon_metrics",
    "format_garch_row",
    "garch_short_horizon_config_from_raw",
    "metrics_from_fit",
    "realized_vol_annualized_log",
    "realized_vol_percentile_log",
]
