"""Dual-track vol forecast → required return + downside bands.

Default need (research board)
-----------------------------
r_need_ann = 0.043 × σ̂ / σ̂_SH510300
           = λ × σ̂  where λ = 0.043 / σ̂_SH510300
r_need_Nd  = (1 + r_need_ann)^(N/252) − 1

This is the fair-return scale (CSI300 listing-era realized CAGR /
same-day SH510300 vol). Fixed sharpe_like=1.33 remains available for the
old personal-hurdle mode. Bond / defensive ETFs use the bond fair premium
vs SH511090 (30年国债) instead of the CSI300 scale.

Tracks
------
persist
    σ̂5 / σ̂20 = current realized rv5 / rv20
garch
    σ̂_H_ann = sigma_garch_h × sqrt(252/H) for H in {5, 20}

Need returns (fixed-Sharpe mode)
--------------------------------
r_need_ann = sharpe_like × σ̂
r_need_Nd  = (1 + r_need_ann)^(N/252) − 1

Downside (same σ̂, opposite side)
---------------------------------
sigma_H = σ̂_ann / sqrt(252/H)   (or GARCH sigma_garch_h directly)
down_z   = exp(−z · sigma_H) − 1   for z ∈ {1, 1.65≈VaR95, 2}

Ticket card (GARCH MC, optional)
--------------------------------
p_up          = P(R > 0)
e_up          = E[R | R > 0]
e_down        = E[R | R < 0]
q05           = 5% quantile of R
match         = E[max(R,0)] / E[max(-R,0)]
p_ge_need     = P(R >= r_need_Hd)

Research only — does not mutate regime / trade rules.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from etf_daily.lib.fat_tail_risk import slice_log_returns
from etf_daily.lib.garch_dynamics import (
    GarchDynamicsConfig,
    fit_garch_dynamics,
    forecast_sigma_h,
    simulate_h_day_log_returns,
)
from etf_daily.lib.garch_short_horizon_board import GarchShortHorizonConfig
from etf_daily.lib.regime_transition_plot import TRADING_DAYS_PER_YEAR
from etf_daily.lib.vol_return_board import (
    DEFAULT_BOND_ANCHOR,
    DEFAULT_DEFENSIVE_CODES,
    asof_metrics,
    slice_close_through,
)

TARGET_SHARPE_LIKE = 1.33
# CSI300 ETF (华泰柏瑞): listing 2012-05-28 → 2026-08-13 calendar CAGR ≈ 4.29%.
EQUITY_ANCHOR = "SH510300"
ANCHOR_ERP_ANN = 0.043
EQUITY_ANCHOR_LABEL = "沪深300"
# 30年国债ETF鹏扬: listing 2023-06-13 → 2026-09-02 calendar CAGR ≈ 5.14%.
BOND_ANCHOR = DEFAULT_BOND_ANCHOR
BOND_ERP_ANN = 0.051
BOND_ANCHOR_LABEL = "30年国债"
HORIZON_5 = 5
HORIZON_20 = 20
Z_VAR95 = 1.65


def is_defensive_code(code: str) -> bool:
    return str(code).upper() in {
        str(c).upper() for c in DEFAULT_DEFENSIVE_CODES
    }


def implied_sharpe_from_anchor(
    sigma_ann: float,
    *,
    erp_ann: float = ANCHOR_ERP_ANN,
) -> float:
    """Price of risk: slow equity premium divided by same-day anchor vol."""
    sig = float(sigma_ann)
    prem = float(erp_ann)
    if not np.isfinite(sig) or sig <= 0.0 or not np.isfinite(prem):
        return float("nan")
    return prem / sig


def fair_need_ann_series(
    rv20: pd.Series,
    rv20_anchor: pd.Series,
    *,
    erp_ann: float = ANCHOR_ERP_ANN,
) -> pd.Series:
    """Fair annualized need: ``erp_ann × rv20 / rv20_anchor`` (aligned by index).

    Persist-vol scale used by adaptive HTML row 4. Days with non-positive
    or non-finite anchor vol become NaN.
    """
    left = pd.to_numeric(rv20, errors="coerce")
    right = pd.to_numeric(rv20_anchor, errors="coerce")
    aligned = pd.concat([left.rename("rv"), right.rename("anchor")], axis=1, join="inner")
    if aligned.empty:
        return pd.Series(dtype=float, name="r_need_ann")
    prem = float(erp_ann)
    out = np.full(len(aligned), np.nan, dtype=float)
    if np.isfinite(prem):
        ok = (
            np.isfinite(aligned["rv"].to_numpy(dtype=float))
            & np.isfinite(aligned["anchor"].to_numpy(dtype=float))
            & (aligned["anchor"].to_numpy(dtype=float) > 0.0)
            & (aligned["rv"].to_numpy(dtype=float) >= 0.0)
        )
        out[ok] = prem * aligned["rv"].to_numpy(dtype=float)[ok] / aligned["anchor"].to_numpy(
            dtype=float
        )[ok]
    return pd.Series(out, index=aligned.index, name="r_need_ann")


def realized_simple_return(
    close: pd.Series,
    *,
    bars: int = HORIZON_20,
) -> pd.Series:
    """Causal N-bar simple return: ``close[t] / close[t-N] − 1``."""
    c = pd.to_numeric(close, errors="coerce")
    c = c[~c.index.duplicated(keep="last")].sort_index()
    n = int(bars)
    name = f"r_{n}d"
    if n <= 0 or c.empty:
        return pd.Series(dtype=float, name=name)
    return c.pct_change(n, fill_method=None).rename(name)


def period_return_from_ann(
    r_need_ann: pd.Series,
    *,
    bars: int = HORIZON_20,
) -> pd.Series:
    """``(1 + r_need_ann)^(N/252) − 1``. Values ``<= -1`` become NaN."""
    r = pd.to_numeric(r_need_ann, errors="coerce")
    n = int(bars)
    name = f"r_need_{n}d"
    out = np.full(len(r), np.nan, dtype=float)
    if n <= 0 or r.empty:
        return pd.Series(out, index=r.index, name=name)
    vals = r.to_numpy(dtype=float)
    ok = np.isfinite(vals) & (vals > -1.0)
    out[ok] = np.power(1.0 + vals[ok], float(n) / float(TRADING_DAYS_PER_YEAR)) - 1.0
    return pd.Series(out, index=r.index, name=name)


def gap_realized_minus_need(
    close: pd.Series,
    r_need_ann: pd.Series,
    *,
    bars: int = HORIZON_20,
) -> pd.DataFrame:
    """Daily gap: realized N-bar return minus fair period need.

    Negative gap = last N days did not earn the vol-scaled CSI300 compensation.
    Uses only closes through t (no future bars).
    """
    need_ann = pd.to_numeric(r_need_ann, errors="coerce").rename("r_need_ann")
    need_n = period_return_from_ann(need_ann, bars=bars)
    r_n = realized_simple_return(close, bars=bars)
    aligned = pd.concat(
        [need_ann, need_n.rename("r_need_period"), r_n.rename("r_realized")],
        axis=1,
        join="outer",
    )
    aligned["gap"] = aligned["r_realized"] - aligned["r_need_period"]
    return aligned


def garch_h_to_ann(sigma_h: float | None, *, bars: int) -> float:
    """Convert H-day cumulative GARCH sigma to annualized vol (rv-comparable)."""
    if sigma_h is None:
        return float("nan")
    s = float(sigma_h)
    h = int(bars)
    if not np.isfinite(s) or s <= 0 or h <= 0:
        return float("nan")
    return s * float(np.sqrt(float(TRADING_DAYS_PER_YEAR) / float(h)))


def ann_to_h_sigma(sigma_ann: float | None, *, bars: int) -> float:
    """Invert ``garch_h_to_ann``: annualized vol → H-day cumulative sigma."""
    if sigma_ann is None:
        return float("nan")
    s = float(sigma_ann)
    h = int(bars)
    if not np.isfinite(s) or s < 0 or h <= 0:
        return float("nan")
    return s / float(np.sqrt(float(TRADING_DAYS_PER_YEAR) / float(h)))


def need_returns(
    sigma_ann: float,
    *,
    bars: int,
    sharpe_like: float = TARGET_SHARPE_LIKE,
) -> dict[str, float]:
    """Map forecast annualized vol → required ann + period returns."""
    if not np.isfinite(sigma_ann) or float(sigma_ann) < 0:
        return {"r_need_ann": float("nan"), "r_need_period": float("nan")}
    r_ann = float(sharpe_like) * float(sigma_ann)
    n = int(bars)
    if n <= 0 or not np.isfinite(r_ann) or r_ann <= -1.0:
        return {"r_need_ann": r_ann, "r_need_period": float("nan")}
    period = float((1.0 + r_ann) ** (float(n) / float(TRADING_DAYS_PER_YEAR)) - 1.0)
    return {"r_need_ann": r_ann, "r_need_period": period}


def downside_bands(
    sigma_h: float | None,
    *,
    z_var: float = Z_VAR95,
) -> dict[str, float]:
    """Map H-day cumulative sigma → simple-return downside bands (mean≈0).

    Uses log-normal approx: simple = exp(−z · σ_H) − 1.
    """
    nan = float("nan")
    if sigma_h is None:
        return {
            "sigma_h": nan,
            "down_1sig": nan,
            "down_var95": nan,
            "down_2sig": nan,
        }
    s = float(sigma_h)
    if not np.isfinite(s) or s < 0:
        return {
            "sigma_h": nan,
            "down_1sig": nan,
            "down_var95": nan,
            "down_2sig": nan,
        }
    z = float(z_var)
    return {
        "sigma_h": s,
        "down_1sig": float(np.expm1(-1.0 * s)),
        "down_var95": float(np.expm1(-z * s)),
        "down_2sig": float(np.expm1(-2.0 * s)),
    }


def _empty_ticket() -> dict[str, float]:
    nan = float("nan")
    return {
        "p_up": nan,
        "e_up": nan,
        "e_down": nan,
        "q05": nan,
        "match": nan,
        "p_ge_need": nan,
    }


def ticket_stats(
    simple_returns: np.ndarray,
    *,
    need: float | None = None,
) -> dict[str, float]:
    """Six-metric ticket card from H-day simple-return MC paths."""
    out = _empty_ticket()
    r = np.asarray(simple_returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return out
    up = r[r > 0.0]
    down = r[r < 0.0]
    reward = float(np.mean(np.clip(r, 0.0, None)))
    risk = float(np.mean(np.clip(-r, 0.0, None)))
    out["p_up"] = float(np.mean(r > 0.0))
    out["e_up"] = float(np.mean(up)) if up.size else float("nan")
    out["e_down"] = float(np.mean(down)) if down.size else float("nan")
    out["q05"] = float(np.quantile(r, 0.05))
    out["match"] = (reward / risk) if risk > 0.0 else float("nan")
    if need is not None and np.isfinite(float(need)):
        out["p_ge_need"] = float(np.mean(r >= float(need)))
    return out


def _sleeve_for(code: str, defensive_codes: frozenset[str]) -> str:
    return "bond" if str(code).upper() in defensive_codes else "equity"


def _ticket_nan_fields(prefix: str) -> dict[str, float]:
    """Board column stubs for one horizon ticket (values NaN)."""
    nan = float("nan")
    return {
        f"p_up_{prefix}": nan,
        f"e_up_{prefix}": nan,
        f"e_down_{prefix}": nan,
        f"q05_{prefix}": nan,
        f"match_{prefix}": nan,
        f"p_ge_need_{prefix}": nan,
    }


def forecast_garch_ann_vols(
    close: pd.Series,
    as_of: str | pd.Timestamp,
    *,
    cfg: GarchShortHorizonConfig | None = None,
    include_mc: bool = True,
    sharpe_like: float = TARGET_SHARPE_LIKE,
) -> dict[str, Any]:
    """Fit GARCH once; return σ̂ + optional MC ticket card for H=5/20."""
    cfg = cfg or GarchShortHorizonConfig()
    end = pd.Timestamp(as_of).normalize()
    s = slice_close_through(close, end)
    nan = float("nan")
    out: dict[str, Any] = {
        "rv5_garch_ann": nan,
        "rv20_garch_ann": nan,
        "sigma5_garch": nan,
        "sigma20_garch": nan,
        "garch_reliable": False,
        "garch_basis": "",
        **_ticket_nan_fields("5d_garch"),
        **_ticket_nan_fields("20d_garch"),
    }
    if s.empty:
        return out
    try:
        gcfg = cfg.garch if isinstance(cfg.garch, GarchDynamicsConfig) else GarchDynamicsConfig()
        returns = slice_log_returns(s, end, lookback_days=gcfg.train_window)
        if returns.size < int(cfg.min_samples):
            out["garch_basis"] = "insufficient_samples"
            return out
        fit = fit_garch_dynamics(returns, gcfg, max_horizon=max(HORIZON_5, HORIZON_20))
        s5 = forecast_sigma_h(fit, HORIZON_5)
        s20 = forecast_sigma_h(fit, HORIZON_20)
        out["sigma5_garch"] = float(s5) if s5 is not None else nan
        out["sigma20_garch"] = float(s20) if s20 is not None else nan
        out["rv5_garch_ann"] = garch_h_to_ann(s5, bars=HORIZON_5)
        out["rv20_garch_ann"] = garch_h_to_ann(s20, bars=HORIZON_20)
        out["garch_reliable"] = bool(fit.reliable)
        out["garch_basis"] = str(fit.dgp_basis or fit.model or "")
        if include_mc and fit.sigma_1d is not None:
            target = float(sharpe_like)
            for h, prefix, ann_key in (
                (HORIZON_5, "5d_garch", "rv5_garch_ann"),
                (HORIZON_20, "20d_garch", "rv20_garch_ann"),
            ):
                logs = simulate_h_day_log_returns(
                    fit,
                    h,
                    gcfg.mc_samples,
                    gcfg.mc_seed + h + 1000,
                )
                simple = np.expm1(logs)
                out[f"_simple_{prefix}"] = simple
                need = (
                    need_returns(float(out[ann_key]), bars=h, sharpe_like=target)[
                        "r_need_period"
                    ]
                    if np.isfinite(target)
                    else None
                )
                ticket = ticket_stats(simple, need=need)
                out[f"p_up_{prefix}"] = ticket["p_up"]
                out[f"e_up_{prefix}"] = ticket["e_up"]
                out[f"e_down_{prefix}"] = ticket["e_down"]
                out[f"q05_{prefix}"] = ticket["q05"]
                out[f"match_{prefix}"] = ticket["match"]
                out[f"p_ge_need_{prefix}"] = ticket["p_ge_need"]
    except Exception as exc:  # noqa: BLE001 — per-symbol soft fail
        out["garch_basis"] = f"error:{type(exc).__name__}"
    return out


def _apply_p_ge_need(g: dict[str, Any], *, lam: float, apply: bool) -> None:
    """Overwrite p_ge_need from stored MC paths after λ is known."""
    if not apply:
        g["p_ge_need_5d_garch"] = float("nan")
        g["p_ge_need_20d_garch"] = float("nan")
        return
    for h, prefix, ann_key in (
        (HORIZON_5, "5d_garch", "rv5_garch_ann"),
        (HORIZON_20, "20d_garch", "rv20_garch_ann"),
    ):
        simple = g.get(f"_simple_{prefix}")
        if simple is None:
            g[f"p_ge_need_{prefix}"] = float("nan")
            continue
        need = need_returns(float(g[ann_key]), bars=h, sharpe_like=lam)["r_need_period"]
        g[f"p_ge_need_{prefix}"] = ticket_stats(simple, need=need)["p_ge_need"]


def build_vol_need_board(
    frames: Mapping[str, pd.Series | pd.DataFrame],
    as_of: str | pd.Timestamp,
    *,
    bond_anchor: str = DEFAULT_BOND_ANCHOR,
    defensive_codes: frozenset[str] | set[str] | None = None,
    names: Mapping[str, str] | None = None,
    sharpe_like: float | None = None,
    erp_ann: float = ANCHOR_ERP_ANN,
    equity_anchor: str = EQUITY_ANCHOR,
    garch_cfg: GarchShortHorizonConfig | None = None,
    include_mc: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build persist + GARCH need-return + downside board through ``as_of``.

    Default ``sharpe_like=None`` uses fair-return scale
    ``erp_ann / σ_equity_anchor`` (GARCH and persist tracks separately).
    Pass a finite ``sharpe_like`` to restore the old fixed-hurdle mode.
    Bond sleeve need columns are NaN under the default scale.
    """
    def_codes = frozenset(
        str(c).upper() for c in (defensive_codes or DEFAULT_DEFENSIVE_CODES)
    )
    anchor = str(bond_anchor).upper()
    eq_anchor = str(equity_anchor).upper()
    name_map = {str(k).upper(): str(v) for k, v in (names or {}).items()}
    spx_mode = sharpe_like is None
    prem = float(erp_ann)
    forecast_sharpe = float("nan") if spx_mode else float(sharpe_like)

    closes: dict[str, pd.Series] = {}
    for raw_code, frame in frames.items():
        code = str(raw_code).upper()
        if isinstance(frame, pd.DataFrame):
            close_col = "close" if "close" in frame.columns else "$close"
            if close_col not in frame.columns:
                raise KeyError(f"{code}: DataFrame missing close")
            ser = frame[close_col]
            if "as_of" in frame.columns:
                ser = ser.copy()
                ser.index = pd.to_datetime(frame["as_of"])
            elif "datetime" in frame.columns:
                ser = ser.copy()
                ser.index = pd.to_datetime(frame["datetime"])
        else:
            ser = frame
        closes[code] = slice_close_through(ser, as_of)

    if not closes:
        raise ValueError("no price frames provided")

    metrics: dict[str, dict[str, float]] = {}
    garch: dict[str, dict[str, Any]] = {}
    n_garch_ok = 0
    for code in sorted(closes):
        ser = closes[code]
        metrics[code] = asof_metrics(ser, as_of)
        g = forecast_garch_ann_vols(
            ser,
            as_of,
            cfg=garch_cfg,
            include_mc=include_mc,
            sharpe_like=forecast_sharpe,
        )
        garch[code] = g
        if np.isfinite(float(g["rv5_garch_ann"])) and np.isfinite(
            float(g["rv20_garch_ann"])
        ):
            n_garch_ok += 1

    lam_persist = float("nan")
    lam_garch = float("nan")
    sigma_spx_persist = float("nan")
    sigma_spx_garch = float("nan")
    if spx_mode:
        if eq_anchor not in closes:
            raise ValueError(
                f"equity anchor {eq_anchor} missing; pass sharpe_like for fixed mode"
            )
        sigma_spx_persist = float(metrics[eq_anchor]["rv20"])
        sigma_spx_garch = float(garch[eq_anchor]["rv20_garch_ann"])
        lam_persist = implied_sharpe_from_anchor(sigma_spx_persist, erp_ann=prem)
        lam_garch = implied_sharpe_from_anchor(sigma_spx_garch, erp_ann=prem)
        if not np.isfinite(lam_persist) or not np.isfinite(lam_garch):
            raise ValueError(
                f"cannot form λ from {eq_anchor}: persist={sigma_spx_persist} "
                f"garch={sigma_spx_garch}"
            )
        target_display = float(lam_garch)
        formula_need = "erp_ann * sigma_hat_ann / sigma_spx_ann"
    else:
        lam_persist = float(sharpe_like)
        lam_garch = float(sharpe_like)
        target_display = float(sharpe_like)
        formula_need = "sharpe_target * sigma_hat_ann"

    rows: list[dict[str, Any]] = []
    for code in sorted(closes):
        m = metrics[code]
        g = garch[code]
        rv5 = float(m["rv5"])
        rv20 = float(m["rv20"])
        r_5a = float(m["r_5a"])
        r_20a = float(m["r_20a"])
        sleeve = _sleeve_for(code, def_codes)
        equity_need = (not spx_mode) or sleeve != "bond"
        lam_p = lam_persist if equity_need else float("nan")
        lam_g = lam_garch if equity_need else float("nan")
        if spx_mode:
            _apply_p_ge_need(g, lam=lam_g, apply=equity_need and include_mc)

        need5_p = need_returns(rv5, bars=HORIZON_5, sharpe_like=lam_p)
        need20_p = need_returns(rv20, bars=HORIZON_20, sharpe_like=lam_p)
        down5_p = downside_bands(ann_to_h_sigma(rv5, bars=HORIZON_5))
        down20_p = downside_bands(ann_to_h_sigma(rv20, bars=HORIZON_20))
        rv5_g = float(g["rv5_garch_ann"])
        rv20_g = float(g["rv20_garch_ann"])
        need5_g = need_returns(rv5_g, bars=HORIZON_5, sharpe_like=lam_g)
        need20_g = need_returns(rv20_g, bars=HORIZON_20, sharpe_like=lam_g)
        s5_g = float(g["sigma5_garch"])
        s20_g = float(g["sigma20_garch"])
        if not np.isfinite(s5_g):
            s5_g = ann_to_h_sigma(rv5_g, bars=HORIZON_5)
        if not np.isfinite(s20_g):
            s20_g = ann_to_h_sigma(rv20_g, bars=HORIZON_20)
        down5_g = downside_bands(s5_g)
        down20_g = downside_bands(s20_g)
        row_sharpe = lam_g if equity_need else float("nan")

        rows.append(
            {
                "code": code,
                "name": name_map.get(code, ""),
                "sleeve": sleeve,
                "sharpe_target": row_sharpe,
                "rv5": rv5,
                "rv20": rv20,
                "r_5a": r_5a,
                "r_20a": r_20a,
                "r_need_ann_5_persist": need5_p["r_need_ann"],
                "r_need_5d_persist": need5_p["r_need_period"],
                "r_need_ann_20_persist": need20_p["r_need_ann"],
                "r_need_20d_persist": need20_p["r_need_period"],
                "sigma5_persist": down5_p["sigma_h"],
                "down1_5d_persist": down5_p["down_1sig"],
                "down_var95_5d_persist": down5_p["down_var95"],
                "down2_5d_persist": down5_p["down_2sig"],
                "sigma20_persist": down20_p["sigma_h"],
                "down1_20d_persist": down20_p["down_1sig"],
                "down_var95_20d_persist": down20_p["down_var95"],
                "down2_20d_persist": down20_p["down_2sig"],
                "rv5_garch_ann": rv5_g,
                "rv20_garch_ann": rv20_g,
                "r_need_ann_5_garch": need5_g["r_need_ann"],
                "r_need_5d_garch": need5_g["r_need_period"],
                "r_need_ann_20_garch": need20_g["r_need_ann"],
                "r_need_20d_garch": need20_g["r_need_period"],
                "sigma5_garch": down5_g["sigma_h"],
                "down1_5d_garch": down5_g["down_1sig"],
                "down_var95_5d_garch": down5_g["down_var95"],
                "down2_5d_garch": down5_g["down_2sig"],
                "sigma20_garch": down20_g["sigma_h"],
                "down1_20d_garch": down20_g["down_1sig"],
                "down_var95_20d_garch": down20_g["down_var95"],
                "down2_20d_garch": down20_g["down_2sig"],
                "p_up_5d_garch": float(g["p_up_5d_garch"]),
                "e_up_5d_garch": float(g["e_up_5d_garch"]),
                "e_down_5d_garch": float(g["e_down_5d_garch"]),
                "q05_5d_garch": float(g["q05_5d_garch"]),
                "match_5d_garch": float(g["match_5d_garch"]),
                "p_ge_need_5d_garch": float(g["p_ge_need_5d_garch"]),
                "p_up_20d_garch": float(g["p_up_20d_garch"]),
                "e_up_20d_garch": float(g["e_up_20d_garch"]),
                "e_down_20d_garch": float(g["e_down_20d_garch"]),
                "q05_20d_garch": float(g["q05_20d_garch"]),
                "match_20d_garch": float(g["match_20d_garch"]),
                "p_ge_need_20d_garch": float(g["p_ge_need_20d_garch"]),
                "garch_reliable": bool(g["garch_reliable"]),
                "garch_basis": g["garch_basis"],
                "is_bond_anchor": code == anchor,
            }
        )

    board = pd.DataFrame(rows)
    if not board.empty:
        board = board.sort_values(
            by=["sleeve", "rv20"],
            ascending=[True, False],
            kind="mergesort",
        ).reset_index(drop=True)

    meta: dict[str, Any] = {
        "as_of": str(pd.Timestamp(as_of).normalize().date()),
        "bond_anchor": anchor,
        "equity_anchor": eq_anchor,
        "sharpe_mode": "spx_erp" if spx_mode else "fixed",
        "erp_ann": prem if spx_mode else None,
        "sigma_spx_persist_20": sigma_spx_persist if spx_mode else None,
        "sigma_spx_garch_20": sigma_spx_garch if spx_mode else None,
        "sharpe_target": target_display,
        "n_codes": int(len(board)),
        "n_garch_ok": int(n_garch_ok),
        "include_mc": bool(include_mc),
        "z_var95": float(Z_VAR95),
        "formula_r_need_ann": formula_need,
        "formula_r_need_period": "(1 + r_need_ann)^(N/252) - 1",
        "formula_down": "expm1(-z * sigma_H); z in {1, 1.65, 2}",
        "formula_ticket": (
            "MC GARCH simple R: p_up=P(R>0); e_up=E[R|R>0]; e_down=E[R|R<0]; "
            "q05; match=E[max(R,0)]/E[max(-R,0)]; p_ge_need=P(R>=r_need)"
        ),
        "persist": "sigma_hat = current rv5 / rv20",
        "garch": "sigma_hat_ann = sigma_garch_h * sqrt(252/H)",
    }
    return board, meta


def render_readme(meta: dict[str, Any], board: pd.DataFrame) -> str:
    lines = [
        "# 双轨波动预判 → 目标收益 + 下行风险板",
        "",
        f"- as_of: `{meta.get('as_of')}`",
        f"- sharpe_mode: `{meta.get('sharpe_mode', 'fixed')}`",
        f"- equity_anchor: `{meta.get('equity_anchor', '')}`",
        f"- erp_ann: **{meta.get('erp_ann')}**",
        f"- sharpe_target (λ): **{meta.get('sharpe_target')}**",
        f"- n_codes: {meta.get('n_codes')} (garch_ok={meta.get('n_garch_ok')})",
        f"- include_mc: {meta.get('include_mc')}",
        "",
        "## Formulas",
        "",
        f"- persist: `{meta.get('persist')}`",
        f"- garch: `{meta.get('garch')}`",
        f"- need: `{meta.get('formula_r_need_ann')}`",
        f"- need period: `{meta.get('formula_r_need_period')}` （N=5 或 20）",
        f"- downside: `{meta.get('formula_down')}`",
        f"- ticket: `{meta.get('formula_ticket')}`",
        "",
        "## How to read",
        "",
        "- `r_need_*`：公平收益（默认 `4.3% × σ/σ_沪深300`；固定 1.33 模式才是个人效率门槛）",
        "- `down1_*` / `down_var95_*` / `down2_*`：同 σ̂ 下的 −1σ / ~95%VaR / −2σ 跌幅尺子",
        "- 赔率卡（GARCH MC）：`p_up` 涨概率；`e_up`/`e_down` 涨/亏条件均值；"
        "`q05` 倒霉分位；`match` 是 Omega 不是风险匹配；`p_ge_need` 达到 need 的概率",
        "- `r_5a` / `r_20a`：近窗已实现折年化，便于对照目标",
        "- 诊断尺，不是买卖信号",
        "",
    ]
    if board is not None and not board.empty:
        eq = board[board["sleeve"] == "equity"]
        if not eq.empty:
            top = eq.nlargest(5, "r_need_20d_persist", keep="all")
            lines.extend(["## Equity: highest persist 20d need (top 5)", ""])
            for _, r in top.iterrows():
                lines.append(
                    f"- `{r['code']}` need20d_g={_pct(r['r_need_20d_garch'])} "
                    f"p_up20={_pct(r['p_up_20d_garch'])} "
                    f"e_down20={_pct(r['e_down_20d_garch'])} "
                    f"q05_20={_pct(r['q05_20d_garch'])} "
                    f"match20={_num(r['match_20d_garch'])} "
                    f"p_ge_need20={_pct(r['p_ge_need_20d_garch'])}"
                )
            lines.append("")
    return "\n".join(lines)


def _pct(x: Any) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    return f"{v * 100:.2f}%"


def _num(x: Any) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.2f}"


__all__ = [
    "ANCHOR_ERP_ANN",
    "BOND_ANCHOR",
    "BOND_ANCHOR_LABEL",
    "BOND_ERP_ANN",
    "EQUITY_ANCHOR",
    "EQUITY_ANCHOR_LABEL",
    "TARGET_SHARPE_LIKE",
    "Z_VAR95",
    "ann_to_h_sigma",
    "build_vol_need_board",
    "downside_bands",
    "fair_need_ann_series",
    "gap_realized_minus_need",
    "forecast_garch_ann_vols",
    "period_return_from_ann",
    "realized_simple_return",
    "garch_h_to_ann",
    "implied_sharpe_from_anchor",
    "is_defensive_code",
    "need_returns",
    "render_readme",
    "ticket_stats",
]
