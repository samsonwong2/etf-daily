#!/usr/bin/env python3
"""Plot regime-transition overlays on K-line charts.

Default: export HTML for every code in the cluster_mapping_selected pool.
Optional ``--code`` limits the run to one symbol.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.lib.extreme_drop_override import enrich_oos_from_ohlcv_loader
from etf_daily.lib.regime_transition_walkforward import (
    extend_validation_with_pre_oos_walkforward,
)
from etf_daily.lib.regime_transition_plot import (
    DEFAULT_VALIDATION_DIR,
    DEFAULT_VOL_CALM_PCT,
    DEFAULT_VOL_CLUSTER_PCT,
    DEFAULT_VOL_EXTREME_PCT,
    VOL_REGIME_CALM,
    VOL_REGIME_CLUSTER,
    VOL_REGIME_EXTREME,
    VOL_REGIME_NORMAL,
    VOL_REGIMES_SHADED,
    build_confirmation_marker_frame,
    build_overlay_frame,
    compute_volatility_regime_frame,
    load_validation_csvs,
    overlay_marker_specs,
    volatility_regime_segments,
)
from etf_daily.lib.regime_transition_live_snapshot import (
    LiveSnapshotError,
    apply_live_snapshot_to_ohlcv,
    load_live_snapshot,
    snapshot_meta,
)
from etf_daily.lib.regime_transition_trade_bias import load_trade_bias_predictions
from etf_daily.lib.regime_transition_bias_promotion import (
    default_promotion_path,
    load_bias_promotion,
)
from etf_daily.lib.indicators import ma20_trend_state
from etf_daily.lib.symbol_trend_board import load_cluster_mapping_codes
from etf_daily.lib.vol_need_return_board import (
    ANCHOR_ERP_ANN,
    EQUITY_ANCHOR_LABEL,
)
from etf_daily.lib.plotly_html_autoscale import write_adaptive_html
from etf_daily.pipeline.price_adjustments import (
    DEFAULT_QFQ_THRESHOLD,
    auto_qfq_adjust_long_frame,
    log_qfq_adjustments,
)
from etf_daily.paths import (
    CLUSTER_MAPPING_SELECTED_TXT,
    PLOTLY_OUTPUTS_DIR,
    QLIB_PROVIDER_URI,
)

KLINE_MODULE_PATH = (
    _PROJECT_ROOT / "strategy" / "route_a_core" / "个股显示_2025104.py"
)
DEFAULT_START = "2026-06-01"
DEFAULT_END = "2026-07-03"
# ~120 trading-day percentile lookback + rv20 needs ~180 calendar days; keep buffer.
VOL_WARMUP_CALENDAR_DAYS = 220
# Fig7 fair-path mode (design: production default = trailing log trend):
# - own_log_trend_trail: per-bar OLS on log(close) over trailing W years (≤t only).
# - own_log_trend: full-window OLS on log(close) (optional; look-ahead level bias).
# - own_cagr_window: constant window endpoint CAGR from P0 (optional).
# - own_cagr_causal: daily g_t expanding→trail 5y (research only; water-level bias).
# - csi300_need: legacy 4.3%×σ/σCSI300 compounding (optional rebase).
FAIR_PATH_MODE = "own_log_trend_trail"
# Legacy only: restart P0 every N calendar years when FAIR_PATH_MODE == "csi300_need".
FAIR_PATH_REBASE_YEARS = 2
# Phase B: after this many trading bars, switch expanding → trailing window.
FAIR_PATH_CAUSAL_TRAIL_YEARS = 5
# Phase B: need at least this many bars in the CAGR window (else expand from P0).
FAIR_PATH_CAUSAL_MIN_BARS = 63
# own_log_trend / trail: min valid closes in the fit window.
FAIR_PATH_LOG_TREND_MIN_BARS = 63
# own_log_trend_trail: trailing window length in years (expanding until full).
FAIR_PATH_LOG_TREND_TRAIL_YEARS = 2
# Fig7–11: show same-window endpoint CAGR in path legend/hover (no price chord).
FAIR_PATH_SHOW_REALIZED_CAGR = True
# Extra calendar buffer beyond trail_years·365.25 so Qlib weekends/holidays
# still leave a full trail of trading bars before plot start.
FAIR_PATH_LOOKBACK_BUFFER_DAYS = 40
_UNSAFE_NAME_RE = re.compile(r'[\\/:*?"<>|\s]+')


def fair_path_lookback_calendar_days(
    extra_trail_years: float | list[float] | tuple[float, ...] | str | None = None,
) -> int:
    """Calendar lookback so fig7 trailing/causal modes see a full trail.

    ``daily_adaptive_stage_html`` uses a short OOS plot window (e.g. Apr→as_of).
    Fitting trail OLS only on that window labeled ``2年`` but annualized a
    ~4-month slope (hover g up to 10^4%). Lookback must cover the trail.
    """
    mode = str(FAIR_PATH_MODE or "").strip().lower()
    years = 0.0
    if mode == "own_log_trend_trail":
        years = float(FAIR_PATH_LOG_TREND_TRAIL_YEARS)
    elif mode == "own_cagr_causal":
        years = float(FAIR_PATH_CAUSAL_TRAIL_YEARS)
    for ey in _normalize_extra_trail_years(extra_trail_years):
        years = max(years, float(ey))
    trail_cal = 0
    if np.isfinite(years) and years > 0:
        trail_cal = int(np.ceil(years * 365.25)) + int(FAIR_PATH_LOOKBACK_BUFFER_DAYS)
    return max(int(VOL_WARMUP_CALENDAR_DAYS), trail_cal)


def _parse_trail_years_token(item: object) -> float | None:
    """Parse one trail length: float, or fraction string like ``1/12``."""
    if isinstance(item, (int, float)):
        y = float(item)
        return y if np.isfinite(y) and y > 0 else None
    s = str(item).strip()
    if not s:
        return None
    try:
        if "/" in s:
            a, b = s.split("/", 1)
            y = float(a.strip()) / float(b.strip())
        else:
            y = float(s)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return y if np.isfinite(y) and y > 0 else None


def _normalize_extra_trail_years(
    value: float | list[float] | tuple[float, ...] | str | None,
) -> list[float]:
    """Parse optional extra trail horizons (fig8+); keep positive finite order.

    Disable tokens: empty / ``off`` / ``none`` / ``0``.
    """
    if value is None:
        return []
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"", "off", "none", "0"}:
            return []
    raw: list[object]
    if isinstance(value, str):
        raw = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raw = [value]
    out: list[float] = []
    for item in raw:
        y = _parse_trail_years_token(item)
        if y is not None:
            out.append(float(y))
    return out


def _trail_envelope_policy(trail_years: float) -> str:
    """Which rails to draw on a log-trend trail panel.

    fig7 (2y) / fig8 (1y) / fig9 (6m) / fig10 (3m): expanding P95 plus
    EWMA overlay. Other horizons (fig11 1m) keep expanding P95.
    Does not change hold_up or production checklist defaults.
    """
    y = float(trail_years)
    if not np.isfinite(y):
        return "expanding"
    if (
        abs(y - 2.0) < 1e-9
        or abs(y - 1.0) < 1e-9
        or abs(y - 0.5) < 1e-9
        or abs(y - 0.25) < 1e-9
    ):
        return "expanding_plus_ewma"
    return "expanding"


def _ewma_scaled_envelope(path: pd.Series, close: pd.Series):
    """Lazy import so this plot module does not import research at load time."""
    from etf_daily.lib.fair_path_research import ewma_scaled_envelope

    return ewma_scaled_envelope(path, close)


def _daily_half_width_pct(upper: pd.Series, path: pd.Series) -> np.ndarray:
    u = pd.to_numeric(upper, errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(path, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(u), np.nan, dtype=float)
    ok = np.isfinite(u) & np.isfinite(p) & (p > 0)
    out[ok] = u[ok] / p[ok] - 1.0
    return out


def _add_symmetric_rail_traces(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    row: int,
    upper: pd.Series,
    lower: pd.Series,
    path: pd.Series,
    tag: str,
    kind: str,
    color_up: str,
    color_dn: str,
    width: float = 1.6,
    dash: str = "dot",
) -> None:
    """Draw upper/lower rails. ``kind`` is ``p95`` (as-of legend) or ``ewma``."""
    u = pd.to_numeric(upper, errors="coerce")
    lo = pd.to_numeric(lower, errors="coerce")
    if not u.notna().any() or not lo.notna().any():
        return
    daily_pct = _daily_half_width_pct(u, path)
    finite = daily_pct[np.isfinite(daily_pct)]
    asof = float(finite[-1]) if finite.size else float("nan")
    xs = df["datetime"]
    uy = u.to_numpy(dtype=float)
    ly = lo.to_numpy(dtype=float)
    if kind == "p95":
        q_txt = "因果P95"
        pct_txt = f" +{asof:.1%}" if np.isfinite(asof) else ""
        name_up = f"上轨{tag}{pct_txt} ({q_txt})"
        name_dn = f"下轨{tag}{pct_txt.replace('+', '-', 1) if pct_txt else ''} ({q_txt})"
        asof_lab = f"+{asof:.1%}" if np.isfinite(asof) else ""
        hover_up = [
            f"%{{x|%Y-%m-%d}}<br>上轨({asof_lab} {q_txt}·asof)=%{{y:.4f}}<extra></extra>"
        ]
        hover_dn = [
            f"%{{x|%Y-%m-%d}}<br>下轨({asof_lab.replace('+', '-', 1) if asof_lab else ''} {q_txt}·asof)=%{{y:.4f}}<extra></extra>"
        ]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=uy,
                mode="lines",
                name=name_up,
                line=dict(color=color_up, width=width, dash=dash),
                hovertemplate=hover_up[0],
                showlegend=True,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ly,
                mode="lines",
                name=name_dn,
                line=dict(color=color_dn, width=width, dash=dash),
                hovertemplate=hover_dn[0],
                showlegend=True,
            ),
            row=row,
            col=1,
        )
        return
    asof_leg = f" +{asof:.1%}(当日)" if np.isfinite(asof) else ""
    name_up = f"上轨{tag} EWMA残差轨{asof_leg}"
    name_dn = f"下轨{tag} EWMA残差轨{asof_leg.replace('+', '-', 1) if asof_leg else ''}"
    cd_up: list[str] = []
    cd_dn: list[str] = []
    for i, pv in enumerate(daily_pct):
        if np.isfinite(pv):
            cd_up.append(f"上轨(EWMA 当日{pv:+.1%})={uy[i]:.4f}")
            cd_dn.append(f"下轨(EWMA 当日{-pv:+.1%})={ly[i]:.4f}")
        else:
            y_u = uy[i] if i < len(uy) else float("nan")
            y_d = ly[i] if i < len(ly) else float("nan")
            cd_up.append(f"上轨(EWMA)={y_u:.4f}" if np.isfinite(y_u) else "上轨(EWMA)=na")
            cd_dn.append(f"下轨(EWMA)={y_d:.4f}" if np.isfinite(y_d) else "下轨(EWMA)=na")
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=uy,
            mode="lines",
            name=name_up,
            line=dict(color=color_up, width=width, dash=dash),
            hovertemplate="%{x|%Y-%m-%d}<br>%{customdata}<extra></extra>",
            customdata=cd_up,
            showlegend=True,
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ly,
            mode="lines",
            name=name_dn,
            line=dict(color=color_dn, width=width, dash=dash),
            hovertemplate="%{x|%Y-%m-%d}<br>%{customdata}<extra></extra>",
            customdata=cd_dn,
            showlegend=True,
        ),
        row=row,
        col=1,
    )


def _trail_horizon_label(years: float) -> str:
    """Human label for trail length (e.g. 0.5 → 6个月, 1/12 → 1个月)."""
    y = float(years)
    if not np.isfinite(y) or y <= 0:
        return "?"
    months = y * 12.0
    # Tolerate float noise (e.g. 0.083333 → ~1 month).
    if abs(months - round(months)) < 1e-3 and 0.5 <= months < 12:
        return f"{int(round(months))}个月"
    if abs(y - round(y)) < 1e-6:
        return f"{int(round(y))}年"
    return f"{y:g}年"


def _trail_fit_min_bars(
    trail_years: float,
    *,
    trading_days_per_year: float = 252.0,
    floor: int | None = None,
) -> int:
    """OLS min bars for a trail: never demand more points than the window holds.

    Default floor is ``FAIR_PATH_LOG_TREND_MIN_BARS`` (63). Short trails
    (e.g. 1个月 ≈ 21 bars) clamp to ``trail_bars`` so fig11 can fit.
    """
    tdy = float(trading_days_per_year)
    y = float(trail_years)
    trail_bars = int(round(y * tdy)) if np.isfinite(y) and y > 0 and tdy > 0 else 0
    base = int(FAIR_PATH_LOG_TREND_MIN_BARS) if floor is None else max(2, int(floor))
    if trail_bars <= 0:
        return max(2, base)
    return max(2, min(base, trail_bars))


def _asof_close_series(ohlcv: pd.DataFrame) -> pd.Series:
    """Normalized as_of-indexed close series (last wins on duplicate dates)."""
    dt = pd.to_datetime(ohlcv["datetime"]).dt.normalize()
    s = pd.Series(
        pd.to_numeric(ohlcv["$close"], errors="coerce").to_numpy(dtype=float),
        index=pd.DatetimeIndex(dt),
        dtype=float,
    )
    return s[~s.index.duplicated(keep="last")].sort_index()


def _align_series_to_plot(series: pd.Series, plot_df: pd.DataFrame) -> pd.Series:
    """Reindex a datetime-indexed series onto the plot window row order."""
    plot_dt = pd.to_datetime(plot_df["datetime"]).dt.normalize()
    aligned = series.reindex(plot_dt)
    return pd.Series(aligned.to_numpy(dtype=float), index=plot_df.index, dtype=float)


def _trailing_log_g_last(g_aligned: pd.Series, fallback: float) -> float:
    finite = g_aligned.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(finite[-1]) if len(finite) else float(fallback)


def _add_own_log_trail_kline_row(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    row: int,
    close_fit: pd.Series,
    trail_years: float,
    erp_ann: float,
    align_from_long_history: bool,
    line_color: str = "#d62728",
    band_up_color: str = "#2ca02c",
    band_dn_color: str = "#ff7f0e",
    dual_peer_years: float | None = None,
    dual_peer_g: pd.Series | None = None,
    dual_peers: list[tuple[float, pd.Series]] | None = None,
    show_candles: bool = True,
    hover_asset_name: str | None = None,
    precomputed_erp_g: tuple[pd.Series, pd.Series] | None = None,
) -> tuple[pd.Series, pd.Series, float]:
    """Draw clean K + trailing-log fair path (+ P95 / EWMA rails) on one subplot row.

    Dual-scale: ``dual_peers`` (or legacy single ``dual_peer_*``) adds peer
    trail annualized g to legend/hover (e.g. 2y panel also shows 1y / 6m g).

    ``precomputed_erp_g`` is optional ``(erp_fit, g_fit)`` from a prior
    ``trailing_log_trend_price_path`` call so multi-scale callers avoid a
    second OLS on the same close/trail.

    Rails (HTML only; hold_up unchanged): fig7–fig10 expanding P95 + EWMA
    overlay; fig11 and other horizons expanding P95. Percentiles and rails
    use the fit history (``erp_fit`` / ``close_fit``), then align onto the
    plot window so a short OOS chart does not restart the residual sample.
    """
    if show_candles:
        fig.add_trace(
            go.Candlestick(
                x=df["datetime"],
                open=df["$open"],
                high=df["$high"],
                low=df["$low"],
                close=df["$close"],
                name="K线(净)",
                showlegend=False,
            ),
            row=row,
            col=1,
        )
    if precomputed_erp_g is not None:
        erp_fit, g_fit = precomputed_erp_g
    else:
        erp_fit, g_fit = trailing_log_trend_price_path(
            close_fit,
            trail_years=float(trail_years),
            min_bars=_trail_fit_min_bars(float(trail_years)),
            fallback_erp=float(erp_ann),
        )
    if align_from_long_history:
        erp_path = _align_series_to_plot(erp_fit, df)
        g_aln = _align_series_to_plot(g_fit, df)
    else:
        erp_path = erp_fit
        g_aln = g_fit
    own_g = _trailing_log_g_last(g_aln, float(erp_ann))
    hz = _trail_horizon_label(trail_years)
    show_realized = _fair_path_show_realized_cagr()
    g_real_aln: pd.Series | None = None
    real_g = float("nan")
    if show_realized:
        # Causal per-bar endpoint CAGR for legend/hover only — no price chord
        # (an as-of-T geometric chord would look ahead on historical dates).
        min_b = _trail_fit_min_bars(float(trail_years))
        g_real_fit = trailing_realized_cagr_series(
            close_fit,
            trail_years=float(trail_years),
            min_bars=min_b,
        )
        if align_from_long_history:
            g_real_aln = _align_series_to_plot(g_real_fit, df)
        else:
            g_real_aln = g_real_fit
        real_g = _trailing_log_g_last(g_real_aln, float("nan"))
    peers: list[tuple[float, pd.Series]] = []
    if dual_peers:
        peers.extend([(float(y), g) for y, g in dual_peers if g is not None])
    elif (
        dual_peer_years is not None
        and dual_peer_g is not None
        and not dual_peer_g.empty
    ):
        peers.append((float(dual_peer_years), dual_peer_g))

    path_name = f"公平路径(自有log趋势·{hz} g≈{own_g:.1%}"
    for py, pg in peers:
        path_name += (
            f"｜{_trail_horizon_label(py)} g≈{_trailing_log_g_last(pg, float(erp_ann)):.1%}"
        )
    # Hover must not reuse as-of 实现CAGR inside path_name — that number is
    # end-of-series only and would look identical on every historical point.
    hover_path_label = path_name + ", 诊断)"
    if show_realized and np.isfinite(real_g):
        path_name += f"｜实现CAGR≈{real_g:.1%}"
    path_name += ", 诊断)"

    hover_need: list[str] = []
    peer_cols = [
        (py, pg.to_numpy(dtype=float))
        for py, pg in peers
        if len(pg) == len(df)
    ]
    close_plot = pd.to_numeric(df["$close"], errors="coerce")

    def _to_plot(series: pd.Series) -> pd.Series:
        if align_from_long_history:
            return _align_series_to_plot(series, df)
        return series

    pct_signed = _to_plot(
        causal_path_residual_signed_percentile(
            erp_fit,
            close_fit,
            min_bars=_trail_fit_min_bars(float(trail_years)),
        )
    )
    pct_arr = pct_signed.to_numpy(dtype=float)
    ewma_signed = _to_plot(
        ewma_path_residual_signed_percentile(erp_fit, close_fit)
    )
    ewma_arr = ewma_signed.to_numpy(dtype=float)
    for i, (d, pv, rn) in enumerate(
        zip(
            df["datetime"],
            erp_path.to_numpy(dtype=float),
            g_aln.to_numpy(dtype=float),
        )
    ):
        extra = f"<br>重锚=无（滚动{hz}log趋势）"
        dual_txt = ""
        for py, parr in peer_cols:
            dual_txt += (
                f"<br>双尺度·{_trail_horizon_label(py)}路径年化={_fmt_vol(parr[i])}"
            )
        real_txt = ""
        if g_real_aln is not None and len(g_real_aln) == len(df):
            rg = float(g_real_aln.iloc[i])
            if np.isfinite(rg):
                real_txt = f"<br>当日实现年化({hz})={_fmt_vol(rg)}"
        pct_txt = (
            f"<br>相对路径因果分位={_fmt_signed_path_percentile(pct_arr[i])}"
            f"<br>相对路径ewma分位={_fmt_signed_path_percentile(ewma_arr[i])}"
        )
        cv = float(close_plot.iloc[i]) if i < len(close_plot) else float("nan")
        gap_txt = (
            "<br>"
            + fmt_path_gap_hover_line(hover_asset_name, d, cv, pv)
        )
        hover_need.append(
            f"日期={pd.Timestamp(d).date()}<br>"
            f"{hover_path_label}={pv:.4f}<br>"
            f"当日路径年化({hz})={_fmt_vol(rn)}{real_txt}{pct_txt}{gap_txt}{dual_txt}{extra}"
        )
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=erp_path,
            mode="lines",
            name=path_name,
            line=dict(color=line_color, width=2.0, dash="dash"),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_need,
            showlegend=True,
        ),
        row=row,
        col=1,
    )
    policy = _trail_envelope_policy(float(trail_years))
    tag = hz if peers else ""
    if policy in {"expanding", "expanding_plus_ewma"}:
        upper_fit, lower_fit, delta = erp_path_symmetric_envelope(
            erp_fit, close_fit, q=0.95
        )
        upper = _to_plot(upper_fit)
        lower = _to_plot(lower_fit)
        if np.isfinite(delta):
            _add_symmetric_rail_traces(
                fig,
                df,
                row=row,
                upper=upper,
                lower=lower,
                path=erp_path,
                tag=tag,
                kind="p95",
                color_up=band_up_color,
                color_dn=band_dn_color,
            )
    if policy in {"ewma", "expanding_plus_ewma"}:
        ewma_up_fit, ewma_lo_fit, ewma_delta = _ewma_scaled_envelope(
            erp_fit, close_fit
        )
        ewma_up = _to_plot(ewma_up_fit)
        ewma_lo = _to_plot(ewma_lo_fit)
        if np.isfinite(ewma_delta) or ewma_up.notna().any():
            overlay = policy == "expanding_plus_ewma"
            _add_symmetric_rail_traces(
                fig,
                df,
                row=row,
                upper=ewma_up,
                lower=ewma_lo,
                path=erp_path,
                tag=tag,
                kind="ewma",
                color_up="#17becf" if overlay else band_up_color,
                color_dn="#e377c2" if overlay else band_dn_color,
                width=1.3 if overlay else 1.6,
                dash="dashdot" if overlay else "dot",
            )
    return erp_path, g_aln, own_g


# Background colors for elevated vol regimes only (calm/normal stay unshaded).
VOL_BG_CLUSTER = "rgba(214, 39, 40, 0.22)"
VOL_BG_EXTREME = "rgba(148, 0, 0, 0.32)"
VOL_LEGEND_CLUSTER = "rgba(214, 39, 40, 0.70)"
VOL_LEGEND_EXTREME = "rgba(148, 0, 0, 0.85)"
VOL_PCT_LINE = "#9467bd"

# MA20 趋势滞后带：|距MA20| < 2% 判缠绕，不确认趋势（与 OPS 五类口径一致）。
TREND_HUG_BAND_PCT = 2.0

# 极端单日跌幅的视觉标注阈值（视觉层，非信号层）。
# 极端大跌（单日 ≤ -7%）在模型未触发红三角时，由 extreme_drop_override
# 升为正式下行红三角并写入 signals_oos；此处灰标仅提示「已有红三角之外」
# 的极端跌幅日，避免与红三角重复。
EXTREME_DROP_PCT = -7.0
EXTREME_DROP_COLOR = "rgba(90, 90, 90, 0.9)"
EXTREME_DROP_LEGEND = "极端跌幅"


def compute_extreme_drop_dates(
    ohlcv: pd.DataFrame,
    overlay: pd.DataFrame | None = None,
    *,
    threshold_pct: float = EXTREME_DROP_PCT,
) -> set[pd.Timestamp]:
    """Dates with single-day pct <= threshold, excluding red-triangle days.

    Days that already carry a down quantile-switch (red triangle) are
    excluded: the bearish signal is shown there, so the neutral marker and
    its "勿当看跌信号" note would be redundant/misleading.
    """
    if ohlcv is None or ohlcv.empty or "$pct_change" not in ohlcv.columns:
        return set()
    df = ohlcv[["datetime", "$pct_change"]].copy()
    df["as_of"] = pd.to_datetime(df["datetime"]).dt.normalize()
    pct = pd.to_numeric(df["$pct_change"], errors="coerce")
    hits = set(df.loc[pct <= float(threshold_pct), "as_of"])
    if not hits or overlay is None or overlay.empty:
        return hits
    if "quantile_switch" not in overlay.columns or "switch_side" not in overlay.columns:
        return hits
    red = overlay[
        overlay["quantile_switch"].fillna(False).astype(bool)
        & overlay["switch_side"].eq("down")
    ]
    red_days = set(pd.to_datetime(red["as_of"]).dt.normalize())
    return hits - red_days


def annotate_extreme_drops(
    fig: go.Figure,
    ohlcv: pd.DataFrame,
    overlay: pd.DataFrame | None = None,
    *,
    threshold_pct: float = EXTREME_DROP_PCT,
) -> go.Figure:
    """Mark single-day drops ≤ threshold on the price panel (visual only).

    Neutral gray marker below the candle; red-triangle days are skipped.
    Marker is hover-silent (hoverinfo=skip) so it never masks candle hover;
    the explanatory note lives in the candle hover text instead
    (see enrich_candle_hover_with_pct_chg).
    """
    dates = compute_extreme_drop_dates(ohlcv, overlay, threshold_pct=threshold_pct)
    if not dates:
        return fig
    df = ohlcv.copy()
    df["as_of"] = pd.to_datetime(df["datetime"]).dt.normalize()
    df = df[df["as_of"].isin(dates)]
    lows = pd.to_numeric(ohlcv["$low"], errors="coerce")
    highs = pd.to_numeric(ohlcv["$high"], errors="coerce")
    y_span = max(float(highs.max()) - float(lows.min()), abs(float(highs.max())) * 0.02, 1e-6)
    xs = list(df["as_of"])
    ys = [float(lo) - y_span * 0.035 for lo in pd.to_numeric(df["$low"], errors="coerce")]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            name=EXTREME_DROP_LEGEND,
            marker=dict(
                symbol="triangle-down",
                size=13,
                color=EXTREME_DROP_COLOR,
                line=dict(width=1.0, color="#333"),
            ),
            hoverinfo="skip",
            cliponaxis=False,
        ),
        row=1,
        col=1,
    )
    return fig


def _trend_label_zh(state: str) -> str:
    return {
        "up": "趋势上涨",
        "down": "趋势下跌",
        "hug": "MA20缠绕",
    }.get(state, state)


def trend_state_series(
    close: pd.Series,
    *,
    band_pct: float = TREND_HUG_BAND_PCT,
) -> pd.DataFrame:
    """Per-day causal MA20 trend state over a full close series.

    Returns columns: ``ma20``, ``dist_pct``, ``state`` in
    {up, down, hug}; ``hug`` marks the ±band hysteresis zone.
    """
    series = pd.Series(pd.to_numeric(close, errors="coerce"), copy=False)
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    ma = series.rolling(window=20, min_periods=20).mean()
    dist = (series / ma - 1.0) * 100.0
    state = pd.Series("hug", index=series.index, dtype=object)
    state[ma.isna()] = None
    state[dist >= band_pct] = "up"
    state[dist <= -band_pct] = "down"
    return pd.DataFrame(
        {"ma20": ma, "dist_pct": dist, "state": state}, index=series.index
    )


def trend_state_summary(
    close: pd.Series,
    as_of: pd.Timestamp,
    *,
    band_pct: float = TREND_HUG_BAND_PCT,
) -> dict[str, object]:
    """Trend snapshot at ``as_of`` for title display (state, not prediction)."""
    st = ma20_trend_state(close, as_of)
    dist = st.get("dist_ma20_pct")
    dist_v = float(dist) if dist is not None and not pd.isna(dist) else 0.0
    intact = bool(st.get("trend_intact"))
    hugging = abs(dist_v) < band_pct
    if hugging:
        state = "hug"
    elif intact:
        state = "up"
    else:
        state = "down"
    return {
        "state": state,
        "state_zh": _trend_label_zh(state),
        "dist_pct": dist_v,
        "ma20": st.get("ma20"),
        "trend_since": st.get("trend_since"),
        "last_break": st.get("last_break"),
        "days_above": st.get("days_above"),
    }


def _load_kline_plotter_class():
    if not KLINE_MODULE_PATH.exists():
        raise FileNotFoundError(f"KlinePlotter module not found: {KLINE_MODULE_PATH}")
    spec = importlib.util.spec_from_file_location(
        "route_a_kline_display", KLINE_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {KLINE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Ensure sibling imports like runtime_paths resolve.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "KlinePlotter"):
        raise AttributeError("KlinePlotter not found in 个股显示_2025104.py")
    return module.KlinePlotter


def _init_qlib(provider_uri: str | None = None) -> str:
    import qlib
    from qlib.constant import REG_CN

    uri = str(provider_uri or QLIB_PROVIDER_URI)
    qlib.init(provider_uri=uri, region=REG_CN)
    return uri


def load_qlib_ohlcv(
    code: str,
    start_date: str,
    end_date: str,
    *,
    provider_uri: str | None = None,
    init_qlib: bool = True,
    lookback_calendar_days: int = VOL_WARMUP_CALENDAR_DAYS,
    clip_to_window: bool = False,
    apply_qfq: bool = True,
    qfq_threshold: float = DEFAULT_QFQ_THRESHOLD,
) -> pd.DataFrame:
    """Load Qlib OHLCV for plotting.

    When ``apply_qfq`` is True (default), apply the same post-hoc forward
    adjustment used by the signal pipeline (``auto_qfq_adjust_*``) so K-line
    gaps from ETF 除权/拆分 match continuous return series behind triangles.
    """
    from qlib.data import D

    if init_qlib:
        uri = _init_qlib(provider_uri)
    else:
        uri = str(provider_uri or QLIB_PROVIDER_URI)
    # Extra lookback so rv5 + 20d causal quantiles are defined on start_date.
    warmup_start = (
        pd.Timestamp(start_date) - pd.Timedelta(days=int(lookback_calendar_days))
    ).strftime("%Y-%m-%d")
    raw = D.features(
        [str(code).upper()],
        fields=["$open", "$high", "$low", "$close", "$volume"],
        start_time=warmup_start,
        end_time=end_date,
    )
    if raw is None or raw.empty:
        raise ValueError(f"no Qlib OHLCV for {code} in [{warmup_start}, {end_date}]")
    df = raw.reset_index()
    # Qlib MultiIndex usually yields instrument + datetime columns.
    if "datetime" not in df.columns:
        raise ValueError(f"unexpected Qlib columns: {list(df.columns)}")
    if "instrument" in df.columns:
        df = df[df["instrument"].astype(str).str.upper() == str(code).upper()].copy()
    else:
        df = df.copy()
        df["instrument"] = str(code).upper()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    for col in ("$open", "$high", "$low", "$close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    if apply_qfq:
        df, qfq_records = auto_qfq_adjust_long_frame(
            df,
            date_col="datetime",
            code_col="instrument",
            close_col="$close",
            threshold=float(qfq_threshold),
            extra_price_cols=["$open", "$high", "$low"],
        )
        if not qfq_records.empty:
            log_qfq_adjustments(qfq_records, stage=f"plot_ohlcv:{code}")

    df["$pct_change"] = df["$close"].pct_change() * 100.0
    if clip_to_window:
        window = df[
            (df["datetime"] >= pd.Timestamp(start_date))
            & (df["datetime"] <= pd.Timestamp(end_date))
        ].copy()
    else:
        window = df[df["datetime"] <= pd.Timestamp(end_date)].copy()
    if window.empty:
        raise ValueError(
            f"Qlib OHLCV empty for {code} in [{start_date}, {end_date}] "
            f"(provider={uri})"
        )
    # Ensure display window is non-empty when caller later clips.
    display = window[
        (window["datetime"] >= pd.Timestamp(start_date))
        & (window["datetime"] <= pd.Timestamp(end_date))
    ]
    if display.empty:
        raise ValueError(
            f"Qlib OHLCV empty for {code} in [{start_date}, {end_date}] "
            f"(provider={uri})"
        )
    return window.reset_index(drop=True)


def _price_lookup(ohlcv: pd.DataFrame) -> pd.DataFrame:
    out = ohlcv.copy()
    out["as_of"] = pd.to_datetime(out["datetime"]).dt.normalize()
    return out.set_index("as_of")[["$high", "$low", "$close"]]


def enrich_candle_hover_with_pct_chg(
    fig: go.Figure,
    ohlcv: pd.DataFrame,
    *,
    trend_frame: pd.DataFrame | None = None,
    extreme_dates: set[pd.Timestamp] | None = None,
) -> go.Figure:
    """Add 涨跌幅%/pct_chg (+ optional MA20 trend state) to candlestick hover.

    Older Plotly Candlestick traces do not support ``hovertemplate``; use
    per-bar ``text`` / ``hoverinfo='text'`` instead.
    """
    if ohlcv is None or ohlcv.empty or "$pct_change" not in ohlcv.columns:
        return fig
    prices = ohlcv.copy()
    prices["as_of"] = pd.to_datetime(prices["datetime"]).dt.normalize()
    by_date = prices.set_index("as_of")
    # Keep last row per day if duplicates exist.
    by_date = by_date[~by_date.index.duplicated(keep="last")]
    trend_by_date: pd.DataFrame | None = None
    if trend_frame is not None and not trend_frame.empty:
        tf = trend_frame.copy()
        tf.index = pd.to_datetime(tf.index).normalize()
        trend_by_date = tf[~tf.index.duplicated(keep="last")]

    for tr in fig.data:
        if getattr(tr, "type", None) != "candlestick":
            continue
        xs = getattr(tr, "x", None)
        if xs is None:
            continue
        opens = list(tr.open) if tr.open is not None else []
        highs = list(tr.high) if tr.high is not None else []
        lows = list(tr.low) if tr.low is not None else []
        closes = list(tr.close) if tr.close is not None else []
        texts: list[str] = []
        for i, x in enumerate(xs):
            key = pd.Timestamp(x).normalize()
            date_s = key.strftime("%Y-%m-%d")
            o = opens[i] if i < len(opens) else None
            h = highs[i] if i < len(highs) else None
            lo = lows[i] if i < len(lows) else None
            c = closes[i] if i < len(closes) else None
            pct = None
            if key in by_date.index:
                raw = by_date.loc[key, "$pct_change"]
                if not pd.isna(raw):
                    pct = float(raw)
            pct_s = "N/A" if pct is None else f"{pct:.2f}"
            lines = [
                f"日期={date_s}",
                f"open={_fmt_px(o)}",
                f"high={_fmt_px(h)}",
                f"low={_fmt_px(lo)}",
                f"close={_fmt_px(c)}",
                f"pct_chg={pct_s}",
            ]
            if extreme_dates is not None and key in extreme_dates:
                lines.append("极端跌幅标注（视觉层，非交易信号）")
                lines.append("回测参考：未触发红三角的极端大跌，")
                lines.append("20日反弹概率约70~79%，勿当看跌信号")
            if trend_by_date is not None and key in trend_by_date.index:
                trow = trend_by_date.loc[key]
                if isinstance(trow, pd.DataFrame):
                    trow = trow.iloc[-1]
                t_dist = trow.get("dist_pct")
                t_ma = trow.get("ma20")
                t_state = str(trow.get("state") or "")
                if t_state and t_state.lower() != "nan":
                    dist_t = (
                        "N/A"
                        if t_dist is None or pd.isna(t_dist)
                        else f"{float(t_dist):+.2f}%"
                    )
                    lines.append(f"MA20={_fmt_px(t_ma)}（距{dist_t}）")
                    lines.append(f"趋势={_trend_label_zh(t_state)}（状态≠预测）")
            texts.append("<br>".join(lines))
        tr.text = texts
        tr.hoverinfo = "text"
        tr.hovertext = texts
        break
    return fig


def _fmt_px(v: object) -> str:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "N/A"
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def annotate_regime_overlays(
    fig: go.Figure,
    overlay: pd.DataFrame,
    ohlcv: pd.DataFrame,
    *,
    title: str,
    vol_frame: pd.DataFrame | None = None,
) -> go.Figure:
    prices = _price_lookup(ohlcv)
    # Offset markers by a share of the full window range so they clear wicks.
    y_hi = float(prices["$high"].max())
    y_lo = float(prices["$low"].min())
    y_span = max(y_hi - y_lo, abs(y_hi) * 0.02, 1e-6)
    # Dedicated signal rails above/below the whole price window.
    # Avoids markers visually cutting through neighboring tall candles.
    rail_above = y_hi + y_span * 0.10
    rail_below = y_lo - y_span * 0.10
    rail_gap = y_span * 0.045

    # Keep candlestick/volume out of the legend; overlays are the focus.
    for tr in fig.data:
        if getattr(tr, "type", None) in {"candlestick", "bar"}:
            tr.showlegend = False

    # Volatility regime background on the price panel (elevated regimes only).
    vol_by_date: pd.DataFrame | None = None
    if vol_frame is not None and not vol_frame.empty:
        vol_by_date = vol_frame.copy()
        vol_by_date["as_of"] = pd.to_datetime(vol_by_date["as_of"]).dt.normalize()
        vol_by_date = vol_by_date.set_index("as_of", drop=False)
        start = pd.Timestamp(ohlcv["datetime"].min()).normalize()
        end = pd.Timestamp(ohlcv["datetime"].max()).normalize()
        segments = volatility_regime_segments(
            vol_frame,
            start_date=start,
            end_date=end,
            regimes=VOL_REGIMES_SHADED,
        )
        shown = {VOL_REGIME_CLUSTER: False, VOL_REGIME_EXTREME: False}
        for seg in segments:
            if seg["regime"] == VOL_REGIME_EXTREME:
                color = VOL_BG_EXTREME
                legend_color = VOL_LEGEND_EXTREME
            else:
                color = VOL_BG_CLUSTER
                legend_color = VOL_LEGEND_CLUSTER
            # Extend half-day so adjacent segments touch visually.
            x0 = pd.Timestamp(seg["start"]) - pd.Timedelta(hours=12)
            x1 = pd.Timestamp(seg["end"]) + pd.Timedelta(hours=12)
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=color,
                opacity=1.0,
                layer="below",
                line_width=0,
                row=1,
                col=1,
            )
            if not shown.get(seg["regime"], True):
                shown[seg["regime"]] = True
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker=dict(
                            size=14,
                            color=legend_color,
                            symbol="square",
                        ),
                        name=seg["regime"],
                        hoverinfo="skip",
                        showlegend=True,
                    ),
                    row=1,
                    col=1,
                )

    specs = overlay_marker_specs(overlay)
    above_slot = 0
    below_slot = 0
    for spec in specs:
        xs = []
        ys = []
        hovers = []
        if spec["anchor"] == "above":
            y_rail = rail_above + above_slot * rail_gap
            above_slot += 1
        else:
            y_rail = rail_below - below_slot * rail_gap
            below_slot += 1
        for x, hover, (_, row) in zip(spec["x"], spec["hover"], spec["rows"].iterrows()):
            as_of = pd.Timestamp(x).normalize()
            if as_of not in prices.index:
                continue
            xs.append(as_of)
            ys.append(y_rail)
            # Append vol-regime context when available.
            if vol_by_date is not None and as_of in vol_by_date.index:
                vr = vol_by_date.loc[as_of]
                if isinstance(vr, pd.DataFrame):
                    vr = vr.iloc[-1]
                hover = (
                    f"{hover}<br>vol_regime={vr.get('vol_regime')}"
                    f"<br>rv5={_fmt_vol(vr.get('rv5'))}"
                    f"<br>rv20={_fmt_vol(vr.get('rv20'))}"
                    f"<br>vol5_pct_120d={_fmt_pct_rank(vr.get('vol5_pct_120d'))}"
                )
            hovers.append(hover)
        if not xs:
            continue
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=spec["name"],
                marker=dict(
                    symbol=spec["symbol"],
                    size=16,
                    color=spec["color"],
                    line=dict(width=1.5, color="#222"),
                ),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hovers,
                cliponaxis=False,
            ),
            row=1,
            col=1,
        )

    n_switch = int(overlay["quantile_switch"].fillna(False).sum())
    confirm = build_confirmation_marker_frame(overlay)
    n_early = (
        int(confirm["confirm_stage"].eq("early").sum())
        if not confirm.empty and "confirm_stage" in confirm.columns
        else 0
    )
    n_final = (
        int(confirm["confirm_stage"].eq("final").sum())
        if not confirm.empty and "confirm_stage" in confirm.columns
        else 0
    )
    n_cluster = n_extreme = n_calm = n_normal = 0
    if vol_frame is not None and not vol_frame.empty:
        disp = vol_frame.copy()
        disp["as_of"] = pd.to_datetime(disp["as_of"]).dt.normalize()
        start = pd.Timestamp(ohlcv["datetime"].min()).normalize()
        end = pd.Timestamp(ohlcv["datetime"].max()).normalize()
        disp = disp[(disp["as_of"] >= start) & (disp["as_of"] <= end)]
        n_cluster = int((disp["vol_regime"] == VOL_REGIME_CLUSTER).sum())
        n_extreme = int((disp["vol_regime"] == VOL_REGIME_EXTREME).sum())
        n_calm = int((disp["vol_regime"] == VOL_REGIME_CALM).sum())
        n_normal = int((disp["vol_regime"] == VOL_REGIME_NORMAL).sum())

    # Drop subplot titles from candle_vol; one layout title is enough.
    if getattr(fig.layout, "annotations", None):
        kept = []
        for ann in fig.layout.annotations:
            text = str(getattr(ann, "text", "") or "")
            if (
                text in {
                    "价格",
                    "成交量",
                    "波动率",
                    "公平年化与20日差额",
                    "公平年化与5日差额",
                    "公平收益（7%×σ/σ标普）",
                    "自训练策略净值",
                    title,
                }
                or "分位切换与反转验证" in text
                or text.startswith("K线 vs ")
            ):
                continue
            kept.append(ann)
        fig.layout.annotations = tuple(kept)

    fig.update_layout(
        title=dict(
            text=(
                f"{title}<br>"
                f"<span style='font-size:12px;color:#555'>"
                f"三角=当日可观察分位切换（默认仅观察：只标上行/下行；"
                f"偏买/偏卖须 lockbox 门控通过后才标注，无未来数据）；"
                f"<br>浅色空心菱形=T+3早期确认日；深色空心菱形=T+10最终确认日"
                f"（标记在确认日而非拐点日，依赖未来post_ret，非买卖单）。"
                f"<br>背景色带=波动率聚集环境（rv5 相对近120日历史分位；"
                f"≥70%聚集、≥90%极端聚集；平稳/正常不着色；无未来数据）。"
                f"<br>切换点={n_switch}，早期确认={n_early}，最终确认={n_final}，"
                f"聚集日={n_cluster}，极端聚集日={n_extreme}，"
                f"正常日={n_normal}，平稳日={n_calm}"
                f"</span>"
            ),
            x=0.01,
            xanchor="left",
            y=0.99,
            yanchor="top",
        ),
        # Title on top; legend below volume/vol panel so they never collide.
        # Adaptive annotate_figure may further grow margin.t for long titles.
        margin=dict(t=160, b=130, l=60, r=70),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#ddd",
            borderwidth=1,
            font=dict(size=11),
            traceorder="normal",
            itemsizing="constant",
        ),
        hovermode="closest",
        # Keep multi-row heights from build_candle_volume_vol_figure (e.g. 2500
        # for 7 panels). Only default when the candle figure left height unset.
        height=(
            int(fig.layout.height)
            if fig.layout.height is not None
            else 1100
        ),
    )
    # Grow top margin if the (already long) base title would spill into candles.
    from etf_daily.lib.adaptive_stage_common import _apply_title_top_margin

    fig = _apply_title_top_margin(fig, min_t=160)
    # Give the volume subplot a named y-axis since subplot titles were removed.
    # Vol dual-axis titles are set in build_candle_volume_vol_figure.
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    # Extreme single-day drops: neutral visual marker (not a signal).
    # Red-triangle days are excluded inside (they already carry the signal).
    fig = annotate_extreme_drops(fig, ohlcv, overlay=overlay)
    # Include both rails in the y-range.
    n_above = max(above_slot, 1)
    n_below = max(below_slot, 1)
    fig.update_yaxes(
        range=[
            rail_below - (n_below + 0.8) * rail_gap,
            rail_above + (n_above + 0.8) * rail_gap,
        ],
        row=1,
        col=1,
    )
    return fig


def _fmt_vol(value: object) -> str:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "—"
        return f"{float(value) * 100.0:.2f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_pct_rank(value: object) -> str:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "—"
        return f"{float(value) * 100.0:.1f}%"
    except (TypeError, ValueError):
        return "—"


SMA_WINDOWS: tuple[int, ...] = (5, 10, 20)
SMA_COLORS: dict[str, str] = {
    "MA5": "#ff7f0e",
    "MA10": "#1f77b4",
    "MA20": "#9467bd",
}


def attach_sma_columns(
    ohlcv: pd.DataFrame,
    *,
    windows: tuple[int, ...] = SMA_WINDOWS,
    close_col: str = "$close",
) -> pd.DataFrame:
    """Add simple moving averages of ``close_col`` (trading-day rolling)."""
    out = ohlcv.copy()
    if close_col not in out.columns:
        return out
    close = pd.to_numeric(out[close_col], errors="coerce")
    for w in windows:
        col = f"MA{int(w)}"
        out[col] = close.rolling(window=int(w), min_periods=int(w)).mean()
    return out


def constant_erp_price_path(
    close: pd.Series,
    *,
    erp_ann: float = ANCHOR_ERP_ANN,
    trading_days_per_year: float = 252.0,
) -> pd.Series:
    """Constant-annualized fair path: ``P0 × (1+erp)^(i / 252)``.

    ``P0`` is the first finite close; ``i`` is the 0-based bar index in the
    series order (trading-day spacing, not calendar days).
    """
    y = pd.to_numeric(close, errors="coerce")
    out = pd.Series(np.nan, index=y.index, dtype=float)
    if y.empty:
        return out
    valid = y.notna()
    if not valid.any():
        return out
    p0 = float(y.loc[valid].iloc[0])
    if not np.isfinite(p0) or p0 <= 0:
        return out
    erp = float(erp_ann)
    tdy = float(trading_days_per_year)
    if not np.isfinite(erp) or not np.isfinite(tdy) or tdy <= 0:
        return out
    # Bar index across the full window so gaps keep calendar alignment of x
    # but compound at trading-bar pace from the first plotted bar.
    i = np.arange(len(y), dtype=float)
    path = p0 * np.power(1.0 + erp, i / tdy)
    out.iloc[:] = path
    return out


def window_own_cagr(
    close: pd.Series,
    *,
    trading_days_per_year: float = 252.0,
) -> float:
    """Geometric annualized CAGR of plot-window closes (Phase A fig7 ruler).

    ``g = (P_T / P_0)^(tdy / n_steps) - 1`` where ``P_0``/``P_T`` are the first
    and last strictly positive finite closes and ``n_steps`` is the positional
    bar distance between them in ``close`` (trading-bar pace).
    """
    y = pd.to_numeric(close, errors="coerce")
    if y.empty:
        return float("nan")
    valid = y.notna() & np.isfinite(y.to_numpy(dtype=float)) & (y > 0)
    if int(valid.sum()) < 2:
        return float("nan")
    idx = np.flatnonzero(valid.to_numpy())
    p0 = float(y.iloc[int(idx[0])])
    pt = float(y.iloc[int(idx[-1])])
    n_steps = int(idx[-1] - idx[0])
    tdy = float(trading_days_per_year)
    if n_steps <= 0 or p0 <= 0 or not np.isfinite(pt) or not np.isfinite(tdy) or tdy <= 0:
        return float("nan")
    return float((pt / p0) ** (tdy / float(n_steps)) - 1.0)


def log_trend_price_path(
    close: pd.Series,
    *,
    trading_days_per_year: float = 252.0,
    min_bars: int | None = None,
    fallback_erp: float = ANCHOR_ERP_ANN,
) -> tuple[pd.Series, float]:
    """OLS fair path on ``log(close)``: ``path_i = exp(a + b·i)`` (full window).

    Fits only strictly positive finite closes. Annualized slope
    ``g = exp(b · tdy) - 1`` is returned for the legend. When fewer than
    ``min_bars`` valid points, falls back to constant endpoint-CAGR path
    (``own_cagr_window``). Returns ``(path, g_ann)``.
    """
    y = pd.to_numeric(close, errors="coerce")
    out = pd.Series(np.nan, index=y.index, dtype=float)
    min_b = (
        int(FAIR_PATH_LOG_TREND_MIN_BARS)
        if min_bars is None
        else max(2, int(min_bars))
    )
    tdy = float(trading_days_per_year)
    if y.empty or not np.isfinite(tdy) or tdy <= 0:
        return out, float("nan")
    y_arr = y.to_numpy(dtype=float)
    valid = np.isfinite(y_arr) & (y_arr > 0)
    n_valid = int(valid.sum())
    if n_valid < 2:
        return out, float("nan")
    if n_valid < min_b:
        g = window_own_cagr(y, trading_days_per_year=tdy)
        if not np.isfinite(g):
            g = float(fallback_erp)
        return constant_erp_price_path(y, erp_ann=float(g), trading_days_per_year=tdy), float(
            g
        )
    idx = np.flatnonzero(valid)
    # Fit on positional bar index so gaps keep x-alignment with the plot window.
    i_all = np.arange(len(y_arr), dtype=float)
    i_fit = i_all[idx]
    log_y = np.log(y_arr[idx])
    # np.polyfit returns [b, a] for logP = a + b·i
    b, a = np.polyfit(i_fit, log_y, 1)
    if not np.isfinite(a) or not np.isfinite(b):
        g = window_own_cagr(y, trading_days_per_year=tdy)
        if not np.isfinite(g):
            g = float(fallback_erp)
        return constant_erp_price_path(y, erp_ann=float(g), trading_days_per_year=tdy), float(
            g
        )
    path = np.exp(a + b * i_all)
    # Only expose path on bars that had a usable close (or all bars — design
    # uses full window indices; fill everywhere for a continuous ruler).
    out.iloc[:] = path
    g_ann = float(np.exp(b * tdy) - 1.0)
    if not np.isfinite(g_ann):
        g_ann = float(fallback_erp) if np.isfinite(fallback_erp) else float("nan")
    return out, g_ann


def trailing_log_trend_price_path(
    close: pd.Series,
    *,
    trail_years: float | None = None,
    min_bars: int | None = None,
    trading_days_per_year: float = 252.0,
    fallback_erp: float = ANCHOR_ERP_ANN,
) -> tuple[pd.Series, pd.Series]:
    """Causal trailing OLS on ``log(close)`` (fig7 production default).

    For each bar ``t``, fit ``log P = a + b·j`` on closes in
    ``[max(0, t+1-trail_bars), t]`` only (expanding until the trail is full).
    ``path[t] = exp(a + b·j_last)`` at the last valid point in that window —
    never back-fills full-sample coefficients onto history.

    Returns ``(path, g_t)`` where ``g_t = exp(b_t · tdy) - 1``.
    """
    y = pd.to_numeric(close, errors="coerce")
    out = pd.Series(np.nan, index=y.index, dtype=float)
    g_out = pd.Series(np.nan, index=y.index, dtype=float)
    tdy = float(trading_days_per_year)
    if y.empty or not np.isfinite(tdy) or tdy <= 0:
        return out, g_out
    trail_y = (
        float(FAIR_PATH_LOG_TREND_TRAIL_YEARS)
        if trail_years is None
        else float(trail_years)
    )
    min_b = (
        int(FAIR_PATH_LOG_TREND_MIN_BARS)
        if min_bars is None
        else max(2, int(min_bars))
    )
    trail_bars = (
        int(round(trail_y * tdy)) if np.isfinite(trail_y) and trail_y > 0 else 0
    )
    y_arr = y.to_numpy(dtype=float)
    n = len(y_arr)
    path_arr = np.full(n, np.nan, dtype=float)
    g_arr = np.full(n, np.nan, dtype=float)
    fb = float(fallback_erp) if np.isfinite(fallback_erp) else float("nan")
    for t in range(n):
        start = 0
        if trail_bars > 0 and (t + 1) >= trail_bars:
            start = t + 1 - trail_bars
        sl = y_arr[start : t + 1]
        ok = np.isfinite(sl) & (sl > 0)
        n_ok = int(ok.sum())
        if n_ok < min_b:
            continue
        jj = np.flatnonzero(ok).astype(float)
        log_y = np.log(sl[ok])
        b, a = np.polyfit(jj, log_y, 1)
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        path_arr[t] = float(np.exp(a + b * float(jj[-1])))
        g_arr[t] = float(np.exp(b * tdy) - 1.0)
    # Warmup: before min_bars, fall back to endpoint CAGR on the expanding prefix
    # once we have ≥2 points (keeps a continuous ruler on young listings).
    first_fit = int(np.flatnonzero(np.isfinite(path_arr))[0]) if np.isfinite(path_arr).any() else n
    for t in range(first_fit):
        sl = y_arr[: t + 1]
        ok = np.isfinite(sl) & (sl > 0)
        if int(ok.sum()) < 2:
            continue
        g = _slice_own_cagr(y_arr, 0, t, trading_days_per_year=tdy)
        if not np.isfinite(g):
            g = fb
        if not np.isfinite(g):
            continue
        idx = np.flatnonzero(ok)
        p0 = float(sl[int(idx[0])])
        n_steps = int(idx[-1] - idx[0])
        path_arr[t] = p0 * ((1.0 + float(g)) ** (n_steps / tdy)) if n_steps > 0 else p0
        g_arr[t] = float(g)
    out.iloc[:] = path_arr
    g_out.iloc[:] = g_arr
    return out, g_out


def _fair_path_show_realized_cagr() -> bool:
    """Env ``FAIR_PATH_SHOW_REALIZED_CAGR=0`` hides fig7–11 legend/hover realized CAGR."""
    v = os.environ.get("FAIR_PATH_SHOW_REALIZED_CAGR", "").strip().lower()
    if v in {"0", "false", "no", "off"}:
        return False
    if v in {"1", "true", "yes", "on"}:
        return True
    return bool(FAIR_PATH_SHOW_REALIZED_CAGR)


def trailing_realized_cagr_series(
    close: pd.Series,
    *,
    trail_years: float | None = None,
    min_bars: int | None = None,
    trading_days_per_year: float = 252.0,
) -> pd.Series:
    """Per-bar endpoint CAGR in the same causal trail window as the fair path.

    For each bar ``t``, ``g_t = (P_T/P_0)^(tdy/n_steps)-1`` on closes in
    ``[max(0, t+1-trail_bars), t]`` (same window as ``trailing_log_trend_price_path``).
    """
    y = pd.to_numeric(close, errors="coerce")
    g_out = pd.Series(np.nan, index=y.index, dtype=float)
    tdy = float(trading_days_per_year)
    if y.empty or not np.isfinite(tdy) or tdy <= 0:
        return g_out
    trail_y = (
        float(FAIR_PATH_LOG_TREND_TRAIL_YEARS)
        if trail_years is None
        else float(trail_years)
    )
    min_b = (
        int(FAIR_PATH_LOG_TREND_MIN_BARS)
        if min_bars is None
        else max(2, int(min_bars))
    )
    trail_bars = (
        int(round(trail_y * tdy)) if np.isfinite(trail_y) and trail_y > 0 else 0
    )
    y_arr = y.to_numpy(dtype=float)
    n = len(y_arr)
    g_arr = np.full(n, np.nan, dtype=float)
    for t in range(n):
        start = 0
        if trail_bars > 0 and (t + 1) >= trail_bars:
            start = t + 1 - trail_bars
        sl = y_arr[start : t + 1]
        ok = np.isfinite(sl) & (sl > 0)
        if int(ok.sum()) < min_b:
            continue
        g_arr[t] = _slice_own_cagr(y_arr, start, t, trading_days_per_year=tdy)
    first_fit = (
        int(np.flatnonzero(np.isfinite(g_arr))[0]) if np.isfinite(g_arr).any() else n
    )
    for t in range(first_fit):
        sl = y_arr[: t + 1]
        ok = np.isfinite(sl) & (sl > 0)
        if int(ok.sum()) < 2:
            continue
        g = _slice_own_cagr(y_arr, 0, t, trading_days_per_year=tdy)
        if np.isfinite(g):
            g_arr[t] = float(g)
    g_out.iloc[:] = g_arr
    return g_out


def realized_cagr_chord(
    close: pd.Series,
    *,
    trail_years: float | None = None,
    min_bars: int | None = None,
    trading_days_per_year: float = 252.0,
) -> pd.Series:
    """Geometric endpoint-CAGR chord for the **last** trail window only.

    ``path[j] = P0 · (1+g)^((j-j0)/tdy)`` for ``j`` in the last window;
    outside the window values are NaN. ``g`` is endpoint CAGR on that window.

    Look-ahead if read as a historical price path (mid-window y uses close[T]).
    Kept for tests/tools; fig7–11 plot uses legend/hover numbers only.
    """
    y = pd.to_numeric(close, errors="coerce")
    out = pd.Series(np.nan, index=y.index, dtype=float)
    tdy = float(trading_days_per_year)
    if y.empty or not np.isfinite(tdy) or tdy <= 0:
        return out
    trail_y = (
        float(FAIR_PATH_LOG_TREND_TRAIL_YEARS)
        if trail_years is None
        else float(trail_years)
    )
    min_b = (
        int(FAIR_PATH_LOG_TREND_MIN_BARS)
        if min_bars is None
        else max(2, int(min_bars))
    )
    trail_bars = (
        int(round(trail_y * tdy)) if np.isfinite(trail_y) and trail_y > 0 else 0
    )
    y_arr = y.to_numpy(dtype=float)
    n = len(y_arr)
    if n < 2:
        return out
    t = n - 1
    start = 0
    if trail_bars > 0 and (t + 1) >= trail_bars:
        start = t + 1 - trail_bars
    sl = y_arr[start : t + 1]
    ok = np.isfinite(sl) & (sl > 0)
    if int(ok.sum()) < 2:
        return out
    g = _slice_own_cagr(y_arr, start, t, trading_days_per_year=tdy)
    if not np.isfinite(g):
        return out
    idx = np.flatnonzero(ok)
    j0 = start + int(idx[0])
    jT = t
    p0 = float(sl[int(idx[0])])
    path_arr = np.full(n, np.nan, dtype=float)
    for j in range(j0, jT + 1):
        n_steps = j - j0
        path_arr[j] = p0 * ((1.0 + float(g)) ** (n_steps / tdy))
    out.iloc[:] = path_arr
    return out


def _slice_own_cagr(
    y_arr: np.ndarray,
    start: int,
    end: int,
    *,
    trading_days_per_year: float,
) -> float:
    """CAGR on ``y_arr[start:end+1]`` using first/last positive finite closes."""
    if end < start:
        return float("nan")
    sl = y_arr[start : end + 1]
    ok = np.isfinite(sl) & (sl > 0)
    if int(ok.sum()) < 2:
        return float("nan")
    idx = np.flatnonzero(ok)
    p0 = float(sl[int(idx[0])])
    pt = float(sl[int(idx[-1])])
    n_steps = int(idx[-1] - idx[0])
    tdy = float(trading_days_per_year)
    if n_steps <= 0 or p0 <= 0 or not np.isfinite(pt) or not np.isfinite(tdy) or tdy <= 0:
        return float("nan")
    return float((pt / p0) ** (tdy / float(n_steps)) - 1.0)


def causal_own_cagr_series(
    close: pd.Series,
    *,
    trail_years: float | None = None,
    min_bars: int | None = None,
    trading_days_per_year: float = 252.0,
) -> pd.Series:
    """Causal daily own CAGR ``g_t`` (Phase B fig7).

    For each bar ``t``, ``g_t`` uses only closes at indices ``≤ t``:
    expanding from the first bar until ``trail_years * 252`` bars are available,
    then a trailing window of that length. Negative CAGR is kept as-is.
    """
    y = pd.to_numeric(close, errors="coerce")
    out = pd.Series(np.nan, index=y.index, dtype=float)
    if y.empty:
        return out
    tdy = float(trading_days_per_year)
    if not np.isfinite(tdy) or tdy <= 0:
        return out
    trail_y = (
        float(FAIR_PATH_CAUSAL_TRAIL_YEARS)
        if trail_years is None
        else float(trail_years)
    )
    min_b = (
        int(FAIR_PATH_CAUSAL_MIN_BARS) if min_bars is None else max(2, int(min_bars))
    )
    trail_bars = (
        int(round(trail_y * tdy)) if np.isfinite(trail_y) and trail_y > 0 else 0
    )
    y_arr = y.to_numpy(dtype=float)
    n = len(y_arr)
    g_arr = np.full(n, np.nan, dtype=float)
    for i in range(n):
        start = 0
        if trail_bars > 0 and (i + 1) >= trail_bars:
            start = i + 1 - trail_bars
        # Prefer trailing when long enough; else expanding from listing/window P0.
        # If the chosen window is shorter than min_bars, fall back to expanding.
        if (i - start + 1) < min_b:
            start = 0
        g_arr[i] = _slice_own_cagr(y_arr, start, i, trading_days_per_year=tdy)
    out.iloc[:] = g_arr
    return out


def causal_own_cagr_price_path(
    close: pd.Series,
    g_ann: pd.Series | None = None,
    *,
    trail_years: float | None = None,
    min_bars: int | None = None,
    trading_days_per_year: float = 252.0,
    fallback_erp: float = ANCHOR_ERP_ANN,
) -> tuple[pd.Series, pd.Series]:
    """Compound fair path with causal daily ``g_t`` (no P0 calendar rebase).

    ``path_0 = P0``, ``path_t = path_{t-1} · (1+g_t)^(1/252)``. Missing ``g_t``
    falls back to the last finite ``g`` then ``fallback_erp``. Returns
    ``(path, g_aligned)``.
    """
    y = pd.to_numeric(close, errors="coerce")
    out = pd.Series(np.nan, index=y.index, dtype=float)
    if g_ann is None:
        g_s = causal_own_cagr_series(
            y,
            trail_years=trail_years,
            min_bars=min_bars,
            trading_days_per_year=trading_days_per_year,
        )
    else:
        g_s = pd.to_numeric(g_ann, errors="coerce")
        if len(g_s) != len(y):
            g_s = pd.Series(np.nan, index=y.index, dtype=float)
        else:
            g_s = pd.Series(g_s.to_numpy(dtype=float), index=y.index, dtype=float)
    if y.empty:
        return out, g_s
    valid = y.notna() & np.isfinite(y.to_numpy(dtype=float)) & (y > 0)
    if not valid.any():
        return out, g_s
    p0 = float(y.loc[valid].iloc[0])
    tdy = float(trading_days_per_year)
    fb = float(fallback_erp)
    if not np.isfinite(p0) or p0 <= 0 or not np.isfinite(tdy) or tdy <= 0:
        return out, g_s
    g_arr = g_s.to_numpy(dtype=float).copy()
    y_arr = y.to_numpy(dtype=float)
    n = len(y_arr)
    path = np.full(n, np.nan, dtype=float)
    first = int(np.flatnonzero(valid.to_numpy())[0])
    path[first] = p0
    last_g = fb if np.isfinite(fb) else 0.0
    for i in range(first + 1, n):
        gi = g_arr[i]
        if not np.isfinite(gi):
            gi = last_g
        else:
            last_g = float(gi)
        # Clip absurd annualized rates for numerical stability (same band as need).
        gi = float(np.clip(gi, -0.99, 5.0))
        g_arr[i] = gi
        path[i] = path[i - 1] * ((1.0 + gi) ** (1.0 / tdy))
    out.iloc[:] = path
    g_s.iloc[:] = g_arr
    return out, g_s


def fair_need_compound_price_path(
    close: pd.Series,
    dates: pd.Series | pd.DatetimeIndex,
    r_need_ann: pd.Series,
    *,
    fallback_erp: float = ANCHOR_ERP_ANN,
    trading_days_per_year: float = 252.0,
    rebase_years: int | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Fair path from daily need: ``P0 × Π (1+r_need_k)^(1/252)``.

    ``r_need_ann`` is aligned onto ``dates`` (as_of index). Missing need days
    forward-fill then fall back to ``fallback_erp`` (default 4.3%).

    When ``rebase_years`` is a positive int, restart ``P0`` at the first bar of
    each calendar segment of that length from the first valid date (so long
    commodity windows stay a usable high/low ruler). ``None`` keeps a single
    listing-era path.

    Returns ``(path, r_need_aligned, rebase_anchor_dates)`` — the third series
    is the trading date of the current segment's ``P0`` (NaT on invalid bars).
    """
    y = pd.to_numeric(close, errors="coerce")
    out = pd.Series(np.nan, index=y.index, dtype=float)
    need_out = pd.Series(np.nan, index=y.index, dtype=float)
    rebase_anchor = pd.Series(pd.NaT, index=y.index, dtype="datetime64[ns]")
    if y.empty:
        return out, need_out, rebase_anchor
    valid = y.notna() & (y > 0) & np.isfinite(y.to_numpy(dtype=float))
    if not valid.any():
        return out, need_out, rebase_anchor
    p0 = float(y.loc[valid].iloc[0])
    if not np.isfinite(p0) or p0 <= 0:
        return out, need_out, rebase_anchor
    tdy = float(trading_days_per_year)
    fb = float(fallback_erp)
    if not np.isfinite(tdy) or tdy <= 0 or not np.isfinite(fb):
        return out, need_out, rebase_anchor

    n = len(y)
    dt = pd.to_datetime(pd.Series(dates)).dt.normalize()
    if len(dt) != n:
        # Fall back to positional index dates if lengths diverge.
        dt = pd.Series(pd.to_datetime(dates), index=y.index).dt.normalize()
        if len(dt) != n:
            return out, need_out, rebase_anchor
    need_s = pd.to_numeric(r_need_ann, errors="coerce")
    if len(need_s) == n and not isinstance(need_s.index, pd.DatetimeIndex):
        # Positional companion series (tests / already-aligned callers).
        need_arr = need_s.to_numpy(dtype=float).copy()
    else:
        need_s = need_s.copy()
        need_s.index = pd.to_datetime(need_s.index).normalize()
        need_s = need_s[~need_s.index.duplicated(keep="last")].sort_index()
        need_arr = (
            need_s.reindex(pd.DatetimeIndex(dt.to_numpy()))
            .ffill()
            .to_numpy(dtype=float)
            .copy()
        )
    bad = ~np.isfinite(need_arr)
    need_arr[bad] = fb
    # Guard absurd / negative annualized needs for compounding stability.
    need_arr = np.clip(need_arr, -0.99, 5.0)
    need_out.iloc[:] = need_arr

    y_arr = y.to_numpy(dtype=float)
    path_arr = np.full(n, np.nan, dtype=float)
    anchor_arr = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    valid_arr = valid.to_numpy()
    dt_arr = pd.DatetimeIndex(dt.to_numpy())

    rebase_n = int(rebase_years) if rebase_years is not None else 0
    if rebase_n <= 0:
        steps = np.ones(n, dtype=float)
        # Step from bar k-1 → k uses r_need at bar k (plan: Π_{k=1..t}).
        if n > 1:
            steps[1:] = np.power(1.0 + need_arr[1:], 1.0 / tdy)
        path_arr[:] = p0 * np.cumprod(steps)
        # Invalid closes still get a compounded path for continuity; anchors
        # only on originally valid bars.
        first_valid_ts = pd.Timestamp(dt_arr[int(np.flatnonzero(valid_arr)[0])])
        for i in range(n):
            if valid_arr[i]:
                anchor_arr[i] = np.datetime64(first_valid_ts.to_datetime64())
        out.iloc[:] = path_arr
        rebase_anchor.iloc[:] = anchor_arr
        return out, need_out, rebase_anchor

    first_i = int(np.flatnonzero(valid_arr)[0])
    t0 = pd.Timestamp(dt_arr[first_i]).normalize()
    last_ts = pd.Timestamp(dt_arr[int(np.flatnonzero(valid_arr)[-1])]).normalize()
    bounds: list[pd.Timestamp] = []
    cursor = t0
    # Cap iterations so pathological dates cannot loop forever.
    for _ in range(512):
        bounds.append(cursor)
        if cursor > last_ts:
            break
        nxt = (cursor + pd.DateOffset(years=rebase_n)).normalize()
        if nxt <= cursor:
            break
        cursor = nxt
    bound_ns = np.array([b.to_datetime64() for b in bounds], dtype="datetime64[ns]")

    current_seg_bound: pd.Timestamp | None = None
    seg_trade_anchor: pd.Timestamp | None = None
    prev_path = float("nan")
    for i in range(n):
        if not valid_arr[i]:
            continue
        d_ts = pd.Timestamp(dt_arr[i]).normalize()
        # Largest calendar boundary <= this bar.
        le = bound_ns[bound_ns <= np.datetime64(d_ts.to_datetime64())]
        seg_bound = pd.Timestamp(le[-1]) if le.size else t0
        if current_seg_bound is None or seg_bound != current_seg_bound:
            # New segment: path = today's close (visible jump is intended).
            path_arr[i] = float(y_arr[i])
            current_seg_bound = seg_bound
            seg_trade_anchor = d_ts
            prev_path = path_arr[i]
            anchor_arr[i] = np.datetime64(seg_trade_anchor.to_datetime64())
        else:
            step = float(np.power(1.0 + need_arr[i], 1.0 / tdy))
            path_arr[i] = prev_path * step
            prev_path = path_arr[i]
            assert seg_trade_anchor is not None
            anchor_arr[i] = np.datetime64(seg_trade_anchor.to_datetime64())

    out.iloc[:] = path_arr
    rebase_anchor.iloc[:] = anchor_arr
    return out, need_out, rebase_anchor


def erp_path_symmetric_envelope(
    path: pd.Series,
    close: pd.Series,
    *,
    q: float = 0.95,
    min_bars: int | None = None,
) -> tuple[pd.Series, pd.Series, float]:
    """Causal expanding P95 bands parallel to ``path`` (no future residuals).

    For each bar ``t`` with valid path/close, residuals are
    ``r_s = ln(close_s/path_s)`` for all valid ``s ≤ t`` only.
    ``δ_t = quantile(|r|_{≤t}, q)`` (default 0.95). Rails:
    ``upper[t] = path[t] · exp(δ_t)``, ``lower[t] = path[t] · exp(-δ_t)``.

    Warmup: fewer than ``min_bars`` valid residuals → NaN rails that day
    (default ``FAIR_PATH_LOG_TREND_MIN_BARS``). Returns
    ``(upper, lower, delta_asof)`` where ``delta_asof`` is the last finite
    ``δ_t`` (for legend ``+x%``).
    """
    p = pd.to_numeric(path, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    upper = pd.Series(np.nan, index=p.index, dtype=float)
    lower = pd.Series(np.nan, index=p.index, dtype=float)
    qq = float(q)
    min_b = (
        int(FAIR_PATH_LOG_TREND_MIN_BARS)
        if min_bars is None
        else max(2, int(min_bars))
    )
    if p.empty or not np.isfinite(qq) or not (0.0 < qq < 1.0):
        return upper, lower, float("nan")
    path_arr = p.to_numpy(dtype=float)
    close_arr = c.to_numpy(dtype=float)
    n = len(path_arr)
    upper_arr = np.full(n, np.nan, dtype=float)
    lower_arr = np.full(n, np.nan, dtype=float)
    delta_arr = np.full(n, np.nan, dtype=float)
    abs_hist: list[float] = []
    for t in range(n):
        pt = float(path_arr[t])
        ct = float(close_arr[t])
        if np.isfinite(pt) and pt > 0 and np.isfinite(ct) and ct > 0:
            rt = float(np.log(ct / pt))
            if np.isfinite(rt):
                abs_hist.append(abs(rt))
        if len(abs_hist) < min_b:
            continue
        if not (np.isfinite(pt) and pt > 0):
            continue
        delta_t = float(np.quantile(np.asarray(abs_hist, dtype=float), qq))
        if not np.isfinite(delta_t) or delta_t < 0:
            continue
        delta_arr[t] = delta_t
        upper_arr[t] = pt * float(np.exp(delta_t))
        lower_arr[t] = pt * float(np.exp(-delta_t))
    upper.iloc[:] = upper_arr
    lower.iloc[:] = lower_arr
    finite_d = delta_arr[np.isfinite(delta_arr)]
    delta_asof = float(finite_d[-1]) if finite_d.size else float("nan")
    return upper, lower, delta_asof


def causal_path_residual_signed_percentile(
    path: pd.Series,
    close: pd.Series,
    *,
    min_bars: int | None = None,
) -> pd.Series:
    """Causal signed percentile of ``|ln(close/path)|`` vs history ``≤ t``.

    Same residual definition as ``erp_path_symmetric_envelope``. For each ``t``,
    ``q_t = mean(|r_s| ≤ |r_t| for s≤t)`` in ``[0, 1]``, then
    ``signed = sign(r_t) · q_t`` (+ above path, − below). Warmup / invalid → NaN.
    """
    p = pd.to_numeric(path, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    min_b = (
        int(FAIR_PATH_LOG_TREND_MIN_BARS)
        if min_bars is None
        else max(2, int(min_bars))
    )
    if p.empty:
        return out
    path_arr = p.to_numpy(dtype=float)
    close_arr = c.to_numpy(dtype=float)
    n = len(path_arr)
    signed = np.full(n, np.nan, dtype=float)
    abs_hist: list[float] = []
    for t in range(n):
        pt = float(path_arr[t])
        ct = float(close_arr[t])
        rt = float("nan")
        if np.isfinite(pt) and pt > 0 and np.isfinite(ct) and ct > 0:
            rt = float(np.log(ct / pt))
            if np.isfinite(rt):
                abs_hist.append(abs(rt))
        if len(abs_hist) < min_b or not np.isfinite(rt):
            continue
        hist = np.asarray(abs_hist, dtype=float)
        q_t = float(np.mean(hist <= abs(rt)))
        if not np.isfinite(q_t):
            continue
        if rt > 0:
            signed[t] = q_t
        elif rt < 0:
            signed[t] = -q_t
        else:
            signed[t] = 0.0
    out.iloc[:] = signed
    return out


def ewma_path_residual_signed_percentile(
    path: pd.Series,
    close: pd.Series,
    *,
    min_bars: int | None = None,
    lam: float | None = None,
) -> pd.Series:
    """Causal signed percentile of ``|z|=|ln(close/path)|/σ_ewma`` vs history ``≤ t``.

    Same residual and σ as ``ewma_scaled_envelope``: ``σ_t`` uses ``r_{t-1}`` only.
    ``q_t = mean(|z_s| ≤ |z_t| for valid s≤t)``, then ``signed = sign(z_t) · q_t``
    (+ above path, − below). Warmup (fewer than ``min_bars`` finite ``|z|``,
    default 63) or invalid → NaN.
    """
    from etf_daily.lib.fair_path_research import (
        DEFAULT_EWMA_LAMBDA,
        DEFAULT_MIN_BARS,
        ewma_residual_sigma,
    )

    p = pd.to_numeric(path, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    min_b = int(DEFAULT_MIN_BARS) if min_bars is None else max(2, int(min_bars))
    lam_f = float(DEFAULT_EWMA_LAMBDA) if lam is None else float(lam)
    if p.empty or not (0.0 < lam_f < 1.0):
        return out
    # Positional pairing, same as causal_path_residual_signed_percentile.
    # Index alignment would scramble residuals when the two indexes differ.
    path_arr = p.to_numpy(dtype=float)
    close_arr = c.to_numpy(dtype=float)
    n = len(path_arr)
    r_arr = np.full(n, np.nan, dtype=float)
    n_c = len(close_arr)
    for t in range(n):
        if t >= n_c:
            continue
        pt = float(path_arr[t])
        ct = float(close_arr[t])
        if np.isfinite(pt) and pt > 0 and np.isfinite(ct) and ct > 0:
            rt = float(np.log(ct / pt))
            if np.isfinite(rt):
                r_arr[t] = rt
    r = pd.Series(r_arr, index=p.index, dtype=float)
    sigma = ewma_residual_sigma(r, lam=lam_f)
    r_arr = r.to_numpy(dtype=float)
    s_arr = sigma.to_numpy(dtype=float)
    n = len(r_arr)
    signed = np.full(n, np.nan, dtype=float)
    abs_z: list[float] = []
    for t in range(n):
        rt = float(r_arr[t])
        st = float(s_arr[t])
        zt = float("nan")
        if np.isfinite(st) and st > 0 and np.isfinite(rt):
            zt = rt / st
            if not np.isfinite(zt):
                zt = float("nan")
        if np.isfinite(zt):
            abs_z.append(abs(zt))
        if len(abs_z) < min_b or not np.isfinite(zt):
            continue
        hist = np.asarray(abs_z, dtype=float)
        q_t = float(np.mean(hist <= abs(zt)))
        if not np.isfinite(q_t):
            continue
        if zt > 0:
            signed[t] = q_t
        elif zt < 0:
            signed[t] = -q_t
        else:
            signed[t] = 0.0
    out.iloc[:] = signed
    return out


def _fmt_signed_path_percentile(value: object) -> str:
    """Format signed causal residual percentile as ``+P72.3`` / ``-P45.1``."""
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "—"
        v = float(value)
        if not np.isfinite(v):
            return "—"
        if v == 0.0:
            return "P0.0"
        sign = "+" if v > 0 else "-"
        return f"{sign}P{abs(v) * 100.0:.1f}"
    except (TypeError, ValueError):
        return "—"


def hover_asset_short_name(name: str | None) -> str:
    """``煤炭ETF国泰`` → ``煤炭``; ``沪深300ETF`` → ``沪深300``."""
    s = str(name or "").strip()
    if not s:
        return ""
    head = re.split(r"ETF", s, maxsplit=1)[0].strip()
    return head or s


def fmt_path_gap_hover_line(
    asset_name: str | None,
    when,
    close: object,
    path: object,
) -> str:
    """Hover line: ``煤炭 8/28 约 +13%`` (gap = close/path − 1, integer %)."""
    label = hover_asset_short_name(asset_name) or "标的"
    try:
        ts = pd.Timestamp(when)
        md = f"{int(ts.month)}/{int(ts.day)}"
    except (TypeError, ValueError):
        md = "—"
    try:
        c = float(close)
        p = float(path)
        if np.isfinite(c) and np.isfinite(p) and p != 0.0:
            pct = int(round((c / p - 1.0) * 100.0))
            return f"{label} {md} 约 {pct:+d}%"
    except (TypeError, ValueError):
        pass
    return f"{label} {md} 约 —"


def build_candle_volume_vol_figure(
    ohlcv: pd.DataFrame,
    vol_frame: pd.DataFrame,
    *,
    template: str = "plotly_white",
    height: int | None = None,
    need_frame: pd.DataFrame | None = None,
    erp_ann: float = ANCHOR_ERP_ANN,
    equity_anchor_label: str = EQUITY_ANCHOR_LABEL,
    policy_frame: pd.DataFrame | None = None,
    policy_meta: dict | None = None,
    train_cutoff: str | None = None,
    fair_path_ohlcv: pd.DataFrame | None = None,
    fair_path_extra_trail_years: float | list[float] | tuple[float, ...] | str | None = None,
    hover_asset_name: str | None = None,
) -> go.Figure:
    """Build shared-x figure: price / volume / realized-vol [ / fair need ] [ / policy ].

    Row 3 left axis = annualized rv5/rv20; right axis = vol5_pct_120d.
    Optional row 4 left = fair annualized need ``erp × rv20 / rv20_anchor``.
    Row 4 right = 20d gap ``realized_20d − fair_20d`` when ``gap_20d`` is present.
    Optional row 5 left = ``erp × rv5 / rv5_anchor`` (``r_need_ann_5``; falls
    back to rv20 annualized). Right = 5d gap from that same annualized series.
    Optional last-but-one row = per-code self-train NAV (hold_up / BH / cash).
    Optional last row = clean K-line (no markers) vs fair need path:
    when ``need_frame.r_need_ann`` is present, compound
    ``P0 × Π (1+r_need)^(1/252)`` (``r_need = erp × σ/σ_anchor``);
    otherwise constant ``erp_ann`` path. Plus causal expanding P95 rails.

    ``fair_path_ohlcv``: optional longer history (warmup + trail). When set,
    trailing/causal fig7 modes fit on that series and align onto the plot
    window so short OOS HTML (daily adaptive) still gets a real N-year trail.

    ``fair_path_extra_trail_years``: optional extra trail horizons with
    need+policy — a float, list, or comma string (e.g. ``1`` or ``1,0.5``).
    Adds **图8+** K vs trailing-log panels and multi-scale g annotations
    across fig7 and the extras (2年 / 1年 / 6个月 …).
    """
    df = ohlcv.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    if not all(f"MA{w}" in df.columns for w in SMA_WINDOWS):
        df = attach_sma_columns(df)
    # Prefer longer history for trail/causal fits; candles stay on ``df``.
    _fp_src = (
        fair_path_ohlcv
        if fair_path_ohlcv is not None and not fair_path_ohlcv.empty
        else None
    )
    if _fp_src is not None and len(_fp_src) > len(df):
        close_for_fair = _asof_close_series(_fp_src)
    else:
        close_for_fair = None

    has_need = need_frame is not None and not need_frame.empty
    has_policy = policy_frame is not None and not policy_frame.empty
    extra_trails = _normalize_extra_trail_years(fair_path_extra_trail_years)
    # Full adaptive HTML (need + policy): add clean K + fair path as row 7
    # (+ optional fig8/fig9… extra trails).
    if has_need and has_policy:
        mode = str(FAIR_PATH_MODE or "").strip().lower()
        trail_y = int(round(float(FAIR_PATH_LOG_TREND_TRAIL_YEARS)))
        realized_note = (
            "·实现CAGR=端点数字≠路径" if _fair_path_show_realized_cagr() else ""
        )
        if mode == "own_log_trend_trail":
            fig7_title = (
                f"图7 K线 vs 公平路径(自有log趋势·滚动{trail_y}年·因果P95+EWMA双轨；"
                f"≠图4公平年化{realized_note})"
            )
        elif mode == "own_log_trend":
            fig7_title = "图7 K线 vs 公平路径(自有log趋势·全样本诊断；≠图4公平年化)"
        elif mode == "own_cagr_causal":
            fig7_title = "图7 K线 vs 公平路径(自有CAGR·因果滚动；≠图4公平年化)"
        elif mode == "own_cagr_window":
            fig7_title = "图7 K线 vs 公平路径(自有CAGR·全样本诊断；≠图4公平年化)"
        else:
            rebase_note = (
                f", 每{int(FAIR_PATH_REBASE_YEARS)}年重锚"
                if FAIR_PATH_REBASE_YEARS and int(FAIR_PATH_REBASE_YEARS) > 0
                else ""
            )
            fig7_title = (
                f"图7 K线 vs 公平路径({float(erp_ann):.1%}×σ/σ"
                f"{equity_anchor_label or EQUITY_ANCHOR_LABEL}{rebase_note})"
            )
        if extra_trails:
            n_extra = len(extra_trails)
            n_rows = 7 + n_extra
            titles = [
                "价格",
                "成交量",
                "波动率",
                "公平年化与20日差额",
                "公平年化与5日差额",
                "自训练策略净值",
                fig7_title,
            ]
            for i, ey in enumerate(extra_trails):
                fig_n = 8 + i
                hz = _trail_horizon_label(ey)
                policy = _trail_envelope_policy(float(ey))
                if policy == "ewma":
                    rail_note = "·EWMA残差轨"
                elif policy == "expanding_plus_ewma":
                    rail_note = "·因果P95+EWMA双轨"
                else:
                    rail_note = ""
                titles.append(
                    f"图{fig_n} K线 vs 公平路径(自有log趋势·滚动{hz}{rail_note}；"
                    f"多尺度对照图7·{_trail_horizon_label(float(FAIR_PATH_LOG_TREND_TRAIL_YEARS))}{realized_note})"
                )
            subplot_titles = tuple(titles)
            # Keep early panels compact; split remaining height across trail rows.
            base = [0.16, 0.05, 0.08, 0.08, 0.09, 0.10]
            trail_share = 1.0 - sum(base)
            per = trail_share / (1 + n_extra)
            row_heights = base + [per] * (1 + n_extra)
            specs = [
                [{}],
                [{}],
                [{"secondary_y": True}],
                [{"secondary_y": True}],
                [{"secondary_y": True}],
                [{}],
            ] + [[{}] for _ in range(1 + n_extra)]
            v_space = 0.05
            if height is None:
                height = 2600 + 400 * n_extra
            policy_row = 6
            erp_path_row = 7
            erp_path_rows_extra = [8 + i for i in range(n_extra)]
        else:
            n_rows = 7
            subplot_titles = (
                "价格",
                "成交量",
                "波动率",
                "公平年化与20日差额",
                "公平年化与5日差额",
                "自训练策略净值",
                fig7_title,
            )
            row_heights = [0.24, 0.07, 0.11, 0.11, 0.12, 0.15, 0.20]
            specs = [
                [{}],
                [{}],
                [{"secondary_y": True}],
                [{"secondary_y": True}],
                [{"secondary_y": True}],
                [{}],
                [{}],
            ]
            # 7 panels share titles + dual-y labels; too-tight spacing overlaps fig5–7.
            v_space = 0.06
            if height is None:
                height = 2600
            policy_row = 6
            erp_path_row = 7
            erp_path_rows_extra = []
    elif has_need:
        n_rows = 5
        subplot_titles = (
            "价格",
            "成交量",
            "波动率",
            "公平年化与20日差额",
            "公平年化与5日差额",
        )
        row_heights = [0.34, 0.12, 0.18, 0.18, 0.18]
        specs = [
            [{}],
            [{}],
            [{"secondary_y": True}],
            [{"secondary_y": True}],
            [{"secondary_y": True}],
        ]
        v_space = 0.04
        if height is None:
            height = 1520
        policy_row = None
        erp_path_row = None
        erp_path_rows_extra = []
    elif has_policy:
        n_rows = 4
        subplot_titles = ("价格", "成交量", "波动率", "自训练策略净值")
        row_heights = [0.42, 0.14, 0.20, 0.24]
        specs = [[{}], [{}], [{"secondary_y": True}], [{}]]
        v_space = 0.05
        if height is None:
            height = 1300
        policy_row = 4
        erp_path_row = None
        erp_path_rows_extra = []
    else:
        n_rows = 3
        subplot_titles = ("价格", "成交量", "波动率")
        row_heights = [0.55, 0.20, 0.25]
        specs = [[{}], [{}], [{"secondary_y": True}]]
        v_space = 0.06
        if height is None:
            height = 1100
        policy_row = None
        erp_path_row = None
        erp_path_rows_extra = []
    erp_txt = f"{float(erp_ann):.1%}"
    anchor_label = str(equity_anchor_label or EQUITY_ANCHOR_LABEL)

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=v_space,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
        specs=specs,
    )
    fig.add_trace(
        go.Candlestick(
            x=df["datetime"],
            open=df["$open"],
            high=df["$high"],
            low=df["$low"],
            close=df["$close"],
            name="K线",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    for w in SMA_WINDOWS:
        col = f"MA{w}"
        if col not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df[col],
                mode="lines",
                name=col,
                line=dict(color=SMA_COLORS.get(col, "#333"), width=1.4),
                hoverinfo="skip",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Bar(
            x=df["datetime"],
            y=df["$volume"],
            name="成交量",
            marker_color="rgba(0,100,128,0.6)",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    vf = vol_frame.copy()
    vf["as_of"] = pd.to_datetime(vf["as_of"]).dt.normalize()
    start = pd.Timestamp(df["datetime"].min()).normalize()
    end = pd.Timestamp(df["datetime"].max()).normalize()
    vf = vf[(vf["as_of"] >= start) & (vf["as_of"] <= end)].copy()
    if not vf.empty:
        hover = [
            (
                f"日期={pd.Timestamp(r.as_of).date()}<br>"
                f"vol_regime={r.vol_regime}<br>"
                f"rv5={_fmt_vol(r.rv5)}<br>"
                f"rv20={_fmt_vol(getattr(r, 'rv20', None))}<br>"
                f"vol5_pct_120d={_fmt_pct_rank(getattr(r, 'vol5_pct_120d', None))}<br>"
                f"说明=分位用近120日rv5历史（含当日），无未来数据"
            )
            for r in vf.itertuples()
        ]
        fig.add_trace(
            go.Scatter(
                x=vf["as_of"],
                y=vf["rv5"],
                mode="lines",
                name="rv5",
                line=dict(color="#1f77b4", width=2),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover,
            ),
            row=3,
            col=1,
            secondary_y=False,
        )
        if "rv20" in vf.columns:
            fig.add_trace(
                go.Scatter(
                    x=vf["as_of"],
                    y=vf["rv20"],
                    mode="lines",
                    name="rv20",
                    line=dict(color="#7f7f7f", width=1.5, dash="dot"),
                    hovertemplate="rv20=%{y:.2%}<extra></extra>",
                ),
                row=3,
                col=1,
                secondary_y=False,
            )
        if "vol5_pct_120d" in vf.columns:
            fig.add_trace(
                go.Scatter(
                    x=vf["as_of"],
                    y=vf["vol5_pct_120d"],
                    mode="lines",
                    name="vol5_pct_120d",
                    line=dict(color=VOL_PCT_LINE, width=2),
                    hovertemplate="vol5_pct_120d=%{y:.1%}<extra></extra>",
                ),
                row=3,
                col=1,
                secondary_y=True,
            )
            # Threshold reference lines on the percentile axis.
            for y, name, dash, color in (
                (DEFAULT_VOL_CALM_PCT, "分位30%", "dot", "#2ca02c"),
                (DEFAULT_VOL_CLUSTER_PCT, "分位70%", "dash", "#d62728"),
                (DEFAULT_VOL_EXTREME_PCT, "分位90%", "dash", "#8b0000"),
            ):
                fig.add_trace(
                    go.Scatter(
                        x=[vf["as_of"].iloc[0], vf["as_of"].iloc[-1]],
                        y=[y, y],
                        mode="lines",
                        name=name,
                        line=dict(color=color, width=1.2, dash=dash),
                        hoverinfo="skip",
                        showlegend=True,
                    ),
                    row=3,
                    col=1,
                    secondary_y=True,
                )

    if has_need:
        nf = need_frame.copy()
        nf["as_of"] = pd.to_datetime(nf["as_of"]).dt.normalize()
        nf = nf[(nf["as_of"] >= start) & (nf["as_of"] <= end)].copy()
        if "r_need_ann" in nf.columns and not nf.empty:
            r_ann = pd.to_numeric(nf["r_need_ann"], errors="coerce")
            r_ann_5 = (
                pd.to_numeric(nf["r_need_ann_5"], errors="coerce")
                if "r_need_ann_5" in nf.columns
                else r_ann
            )
            x0 = nf["as_of"].iloc[0]
            x1 = nf["as_of"].iloc[-1]
            for bars, row, show_leg in ((20, 4, True), (5, 5, False)):
                r_ann_row = r_ann_5 if bars == 5 else r_ann
                use_rv5 = bars == 5 and "r_need_ann_5" in nf.columns
                vol_tag = "rv5" if use_rv5 else "rv20"
                line_name = "公平年化收益(rv5)" if use_rv5 else "公平年化收益"
                need_col = f"r_need_{bars}d"
                real_col = f"r_{bars}d"
                gap_col = f"gap_{bars}d"
                if need_col in nf.columns:
                    r_need = pd.to_numeric(nf[need_col], errors="coerce")
                else:
                    r_need = (1.0 + r_ann_row).pow(float(bars) / 252.0) - 1.0
                if real_col in nf.columns:
                    r_real = pd.to_numeric(nf[real_col], errors="coerce")
                else:
                    r_real = pd.Series(np.nan, index=nf.index)
                if gap_col in nf.columns:
                    gap = pd.to_numeric(nf[gap_col], errors="coerce")
                else:
                    gap = r_real - r_need
                has_gap = bool(np.isfinite(gap.to_numpy(dtype=float)).any())
                need_hover = [
                    (
                        f"日期={pd.Timestamp(d).date()}<br>"
                        f"公平年化={_fmt_vol(a)}<br>"
                        f"公平{bars}日={_fmt_vol(p)}<br>"
                        f"已实现{bars}日={_fmt_vol(rr)}<br>"
                        f"差额=已实现−公平{bars}日={_fmt_vol(g)}<br>"
                        f"公式={erp_txt}×{vol_tag}/{vol_tag}_{anchor_label}（实现波动）"
                    )
                    for d, a, p, rr, g in zip(
                        nf["as_of"], r_ann_row, r_need, r_real, gap
                    )
                ]
                fig.add_trace(
                    go.Scatter(
                        x=nf["as_of"],
                        y=r_ann_row,
                        mode="lines",
                        name=line_name,
                        line=dict(color="#e6550d", width=2),
                        hovertemplate="%{customdata}<extra></extra>",
                        customdata=need_hover,
                        showlegend=True if use_rv5 else show_leg,
                    ),
                    row=row,
                    col=1,
                    secondary_y=False,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[float(erp_ann), float(erp_ann)],
                        mode="lines",
                        name=f"{anchor_label}公平 {erp_txt}",
                        line=dict(color="#636363", width=1.2, dash="dash"),
                        hoverinfo="skip",
                        showlegend=show_leg,
                    ),
                    row=row,
                    col=1,
                    secondary_y=False,
                )
                if has_gap:
                    gap_colors = [
                        (
                            "rgba(214,39,40,0.55)"
                            if np.isfinite(g) and float(g) < 0
                            else "rgba(44,160,44,0.50)"
                        )
                        for g in gap.to_numpy(dtype=float)
                    ]
                    gap_hover = [
                        (
                            f"日期={pd.Timestamp(d).date()}<br>"
                            f"已实现{bars}日={_fmt_vol(rr)}<br>"
                            f"公平{bars}日={_fmt_vol(p)}<br>"
                            f"差额={_fmt_vol(g)}<br>"
                            f"{'近' + str(bars) + '日没赚够波动补偿' if np.isfinite(g) and float(g) < 0 else ('近' + str(bars) + '日覆盖了波动补偿' if np.isfinite(g) else '差额=—')}"
                        )
                        for d, rr, p, g in zip(nf["as_of"], r_real, r_need, gap)
                    ]
                    fig.add_trace(
                        go.Bar(
                            x=nf["as_of"],
                            y=gap,
                            name=f"{bars}日差额(实现−公平)",
                            marker_color=gap_colors,
                            hovertemplate="%{customdata}<extra></extra>",
                            customdata=gap_hover,
                            showlegend=True,
                        ),
                        row=row,
                        col=1,
                        secondary_y=True,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[x0, x1],
                            y=[0.0, 0.0],
                            mode="lines",
                            name="差额=0",
                            line=dict(color="#636363", width=1.0, dash="dot"),
                            hoverinfo="skip",
                            showlegend=show_leg,
                        ),
                        row=row,
                        col=1,
                        secondary_y=True,
                    )

    if has_policy and policy_row is not None:
        pf = policy_frame.copy()
        pf["as_of"] = pd.to_datetime(pf["as_of"]).dt.normalize()
        pf = pf[(pf["as_of"] >= start) & (pf["as_of"] <= end)].copy()
        meta = dict(policy_meta or {})
        chosen = str(meta.get("policy") or "")
        label = str(meta.get("policy_label") or chosen or "自选")
        tr_hu = meta.get("train_hold_up")
        tr_bh = meta.get("train_bh")

        def _pct(x: object) -> str:
            try:
                v = float(x)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return "n/a"
            if not np.isfinite(v):
                return "n/a"
            return f"{v:+.1%}"

        series = (
            ("nav_hold_up", "hold_up净值", "#1f77b4", chosen != "hold_up"),
            ("nav_bh", "BH净值", "#2ca02c", chosen != "bh"),
            ("nav_cash", "cash净值", "#7f7f7f", chosen != "cash"),
        )
        for col, name, color, thin in series:
            if col not in pf.columns:
                continue
            y = pd.to_numeric(pf[col], errors="coerce")
            fig.add_trace(
                go.Scatter(
                    x=pf["as_of"],
                    y=y,
                    mode="lines",
                    name=name,
                    line=dict(
                        color=color,
                        width=1.2 if thin else 2.8,
                        dash="dot" if thin else "solid",
                    ),
                    hovertemplate="%{x|%Y-%m-%d}<br>" + name + "=%{y:.3f}<extra></extra>",
                    showlegend=True,
                ),
                row=policy_row,
                col=1,
            )
        if "nav_policy" in pf.columns and chosen:
            fig.add_trace(
                go.Scatter(
                    x=pf["as_of"],
                    y=pd.to_numeric(pf["nav_policy"], errors="coerce"),
                    mode="lines",
                    name=f"本票自选·{label}",
                    line=dict(color="#d62728", width=2.4),
                    hovertemplate=(
                        "%{x|%Y-%m-%d}<br>自选净值=%{y:.3f}<extra></extra>"
                    ),
                    showlegend=True,
                ),
                row=policy_row,
                col=1,
            )
        cut = train_cutoff or meta.get("train_cutoff")
        if cut and not pf.empty:
            cut_ts = pd.Timestamp(cut).normalize()
            if pf["as_of"].min() <= cut_ts <= pf["as_of"].max():
                fig.add_vline(
                    x=cut_ts,
                    line_width=1.2,
                    line_dash="dash",
                    line_color="#9467bd",
                    row=policy_row,
                    col=1,
                )
                y_max = float(
                    np.nanmax(
                        pd.to_numeric(
                            pf[
                                [
                                    c
                                    for c in (
                                        "nav_hold_up",
                                        "nav_bh",
                                        "nav_cash",
                                        "nav_policy",
                                    )
                                    if c in pf.columns
                                ]
                            ].stack(),
                            errors="coerce",
                        )
                    )
                )
                if not np.isfinite(y_max):
                    y_max = 1.0
                fig.add_annotation(
                    x=cut_ts,
                    y=y_max,
                    text=(
                        f"train截止<br>自选={label}<br>"
                        f"train hold_up={_pct(tr_hu)} BH={_pct(tr_bh)}"
                    ),
                    showarrow=False,
                    yanchor="top",
                    bgcolor="rgba(255,255,255,0.75)",
                    font=dict(size=10, color="#333"),
                    row=policy_row,
                    col=1,
                )

    if erp_path_row is not None:
        mode = str(FAIR_PATH_MODE or "").strip().lower()
        own_g = float("nan")
        need_aligned: pd.Series | None = None
        rebase_anchors: pd.Series | None = None
        trail_y = int(round(float(FAIR_PATH_LOG_TREND_TRAIL_YEARS)))
        close_fit = (
            close_for_fair
            if close_for_fair is not None
            else pd.to_numeric(df["$close"], errors="coerce")
        )
        align_long = close_for_fair is not None
        # Trail panels (fig7 default years; optional fig8/fig9… + multi-scale).
        if mode == "own_log_trend_trail" or extra_trails:
            primary_years = float(FAIR_PATH_LOG_TREND_TRAIL_YEARS)
            # Raw (erp_fit, g_fit) once per horizon; aligned g for dual-scale peers.
            fit_by_years: dict[float, tuple[pd.Series, pd.Series]] = {}
            g_by_years: dict[float, pd.Series] = {}
            all_years = [primary_years] + list(extra_trails)
            for y in all_years:
                erp_fit, g_fit = trailing_log_trend_price_path(
                    close_fit,
                    trail_years=float(y),
                    min_bars=_trail_fit_min_bars(float(y)),
                    fallback_erp=float(erp_ann),
                )
                fit_by_years[float(y)] = (erp_fit, g_fit)
                g_by_years[float(y)] = (
                    _align_series_to_plot(g_fit, df) if align_long else g_fit
                )

            def _peers_for(y: float) -> list[tuple[float, pd.Series]]:
                return [
                    (oy, g_by_years[float(oy)])
                    for oy in all_years
                    if abs(float(oy) - float(y)) > 1e-12
                ]

            _erp7, g7_aln, own_g = _add_own_log_trail_kline_row(
                fig,
                df,
                row=int(erp_path_row),
                close_fit=close_fit,
                trail_years=primary_years,
                erp_ann=float(erp_ann),
                align_from_long_history=align_long,
                dual_peers=_peers_for(primary_years) if extra_trails else None,
                show_candles=True,
                hover_asset_name=hover_asset_name,
                precomputed_erp_g=fit_by_years[float(primary_years)],
            )
            need_aligned = g7_aln
            erp_path = _erp7
            extra_colors = [
                ("#1f77b4", "#17becf", "#e377c2"),
                ("#9467bd", "#8c564b", "#bcbd22"),
                # fig10 (3m): keep green path; darken rails (was too pale #98df8a/#dbdb8d).
                ("#2ca02c", "#145a32", "#7b241c"),
                # fig11 (1m): distinct dark slate path + rails.
                ("#37474f", "#00695c", "#bf360c"),
            ]
            for i, ey in enumerate(extra_trails):
                row_i = (
                    int(erp_path_rows_extra[i])
                    if i < len(erp_path_rows_extra)
                    else (8 + i)
                )
                lc, uc, dc = extra_colors[i % len(extra_colors)]
                _add_own_log_trail_kline_row(
                    fig,
                    df,
                    row=row_i,
                    close_fit=close_fit,
                    trail_years=float(ey),
                    erp_ann=float(erp_ann),
                    align_from_long_history=align_long,
                    line_color=lc,
                    band_up_color=uc,
                    band_dn_color=dc,
                    dual_peers=_peers_for(ey),
                    show_candles=True,
                    hover_asset_name=hover_asset_name,
                    precomputed_erp_g=fit_by_years[float(ey)],
                )
        else:
            # Clean K-line (no MA / markers / regime overlays) vs fair need path.
            fig.add_trace(
                go.Candlestick(
                    x=df["datetime"],
                    open=df["$open"],
                    high=df["$high"],
                    low=df["$low"],
                    close=df["$close"],
                    name="K线(净)",
                    showlegend=False,
                ),
                row=erp_path_row,
                col=1,
            )
            if mode == "own_log_trend":
                # Full-window OLS: intentionally on the visible plot window only.
                erp_path, own_g = log_trend_price_path(
                    df["$close"],
                    fallback_erp=float(erp_ann),
                )
                if not np.isfinite(own_g):
                    own_g = float(erp_ann)
                path_name = f"公平路径(自有log趋势 g≈{own_g:.1%}, 全样本诊断)"
                need_aligned = pd.Series(float(own_g), index=df.index, dtype=float)
                rebase_anchors = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
                if not df.empty:
                    first_dt = pd.Timestamp(df["datetime"].iloc[0]).normalize()
                    rebase_anchors.iloc[:] = first_dt
            elif mode == "own_cagr_causal":
                path_name = "公平路径(自有CAGR·因果滚动)"
                erp_path_fit, need_fit = causal_own_cagr_price_path(
                    close_fit,
                    fallback_erp=float(erp_ann),
                )
                if close_for_fair is not None:
                    erp_path = _align_series_to_plot(erp_path_fit, df)
                    need_aligned = _align_series_to_plot(need_fit, df)
                else:
                    erp_path = erp_path_fit
                    need_aligned = need_fit
                rebase_anchors = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
                if not df.empty:
                    first_dt = pd.Timestamp(df["datetime"].iloc[0]).normalize()
                    rebase_anchors.iloc[:] = first_dt
            elif mode == "own_cagr_window":
                own_g = window_own_cagr(df["$close"])
                if not np.isfinite(own_g):
                    own_g = float(erp_ann)
                path_name = f"公平路径(自有CAGR={own_g:.1%}, 全样本诊断)"
                erp_path = constant_erp_price_path(df["$close"], erp_ann=float(own_g))
                need_aligned = pd.Series(float(own_g), index=df.index, dtype=float)
                rebase_anchors = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
                if not df.empty:
                    first_dt = pd.Timestamp(df["datetime"].iloc[0]).normalize()
                    rebase_anchors.iloc[:] = first_dt
            else:
                path_name = f"公平路径({erp_txt}×σ/σ{anchor_label}"
                if FAIR_PATH_REBASE_YEARS and int(FAIR_PATH_REBASE_YEARS) > 0:
                    path_name += f", 每{int(FAIR_PATH_REBASE_YEARS)}年重锚"
                path_name += ")"
                if has_need and need_frame is not None and "r_need_ann" in need_frame.columns:
                    nf_path = need_frame.copy()
                    nf_path["as_of"] = pd.to_datetime(nf_path["as_of"]).dt.normalize()
                    need_s = (
                        pd.to_numeric(nf_path["r_need_ann"], errors="coerce")
                        .groupby(nf_path["as_of"])
                        .last()
                    )
                    erp_path, need_aligned, rebase_anchors = fair_need_compound_price_path(
                        df["$close"],
                        df["datetime"],
                        need_s,
                        fallback_erp=float(erp_ann),
                        rebase_years=FAIR_PATH_REBASE_YEARS,
                    )
                else:
                    erp_path = constant_erp_price_path(df["$close"], erp_ann=float(erp_ann))
                    need_aligned = pd.Series(
                        float(erp_ann), index=df.index, dtype=float
                    )
                    rebase_anchors = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
                    if not df.empty:
                        first_dt = pd.Timestamp(df["datetime"].iloc[0]).normalize()
                        rebase_anchors.iloc[:] = first_dt
            if need_aligned is not None:
                hover_need = []
                close_need = pd.to_numeric(df["$close"], errors="coerce").to_numpy(
                    dtype=float
                )
                for i, (d, pv, rn, ra) in enumerate(
                    zip(
                        df["datetime"],
                        erp_path.to_numpy(dtype=float),
                        need_aligned.to_numpy(dtype=float),
                        (
                            rebase_anchors
                            if rebase_anchors is not None
                            else pd.Series(pd.NaT, index=df.index)
                        ).to_numpy(),
                    )
                ):
                    if mode == "own_log_trend":
                        extra = "<br>重锚=无（log趋势诊断）"
                    elif mode == "own_cagr_causal":
                        extra = "<br>重锚=无（因果滚动）"
                    elif mode == "own_cagr_window":
                        extra = "<br>重锚=无（全样本诊断）"
                    else:
                        ra_ts = (
                            pd.Timestamp(ra) if ra is not None and not pd.isna(ra) else None
                        )
                        extra = (
                            f"<br>重锚日={ra_ts.date()}"
                            if ra_ts is not None and ra_ts is not pd.NaT
                            else ""
                        )
                    cv = float(close_need[i]) if i < len(close_need) else float("nan")
                    gap_txt = (
                        "<br>"
                        + fmt_path_gap_hover_line(hover_asset_name, d, cv, pv)
                    )
                    hover_need.append(
                        f"日期={pd.Timestamp(d).date()}<br>"
                        f"{path_name}={pv:.4f}<br>"
                        f"当日路径年化={_fmt_vol(rn)}{gap_txt}{extra}"
                    )
                fig.add_trace(
                    go.Scatter(
                        x=df["datetime"],
                        y=erp_path,
                        mode="lines",
                        name=path_name,
                        line=dict(color="#d62728", width=2.0, dash="dash"),
                        hovertemplate="%{customdata}<extra></extra>",
                        customdata=hover_need,
                        showlegend=True,
                    ),
                    row=erp_path_row,
                    col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=df["datetime"],
                        y=erp_path,
                        mode="lines",
                        name=path_name,
                        line=dict(color="#d62728", width=2.0, dash="dash"),
                        hovertemplate=(
                            "%{x|%Y-%m-%d}<br>"
                            + path_name
                            + "=%{y:.4f}<extra></extra>"
                        ),
                        showlegend=True,
                    ),
                    row=erp_path_row,
                    col=1,
                )
            upper, lower, delta = erp_path_symmetric_envelope(
                erp_path, df["$close"], q=0.95
            )
            if np.isfinite(delta):
                band_pct = float(np.exp(delta) - 1.0)
                q_txt = "因果P95"
                fig.add_trace(
                    go.Scatter(
                        x=df["datetime"],
                        y=upper,
                        mode="lines",
                        name=f"上轨 +{band_pct:.1%} ({q_txt})",
                        line=dict(color="#2ca02c", width=1.6, dash="dot"),
                        hovertemplate=(
                            "%{x|%Y-%m-%d}<br>"
                            + f"上轨(+{band_pct:.1%} {q_txt}·asof)=%{{y:.4f}}<extra></extra>"
                        ),
                        showlegend=True,
                    ),
                    row=erp_path_row,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=df["datetime"],
                        y=lower,
                        mode="lines",
                        name=f"下轨 -{band_pct:.1%} ({q_txt})",
                        line=dict(color="#ff7f0e", width=1.6, dash="dot"),
                        hovertemplate=(
                            "%{x|%Y-%m-%d}<br>"
                            + f"下轨(-{band_pct:.1%} {q_txt}·asof)=%{{y:.4f}}<extra></extra>"
                        ),
                        showlegend=True,
                    ),
                    row=erp_path_row,
                    col=1,
                )


    fig.update_layout(
        template=template,
        height=height,
        xaxis_rangeslider_visible=False,
        hovermode="closest",
        margin=dict(r=70),
    )
    # Candlestick defaults enable a rangeslider; disable on every subplot x-axis.
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_xaxes(tickformat="%Y-%m-%d", showticklabels=True, row=1, col=1)
    fig.update_xaxes(tickformat="%Y-%m-%d", showticklabels=True, row=2, col=1)
    last_row = n_rows
    for r in range(3, last_row):
        fig.update_xaxes(tickformat="%Y-%m-%d", showticklabels=True, row=r, col=1)
    fig.update_xaxes(
        title_text="日期",
        tickformat="%Y-%m-%d",
        tickangle=-35,
        showticklabels=True,
        row=last_row,
        col=1,
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_yaxes(
        title_text="年化波动率",
        tickformat=".1%",
        row=3,
        col=1,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="vol5历史分位",
        tickformat=".0%",
        range=[0.0, 1.05],
        row=3,
        col=1,
        secondary_y=True,
    )
    if has_need:
        fig.update_yaxes(
            title_text="公平年化收益",
            tickformat=".1%",
            row=4,
            col=1,
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text="20日差额",
            tickformat=".1%",
            row=4,
            col=1,
            secondary_y=True,
        )
        fig.update_yaxes(
            title_text="公平年化收益(rv5)",
            tickformat=".1%",
            row=5,
            col=1,
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text="5日差额",
            tickformat=".1%",
            row=5,
            col=1,
            secondary_y=True,
        )
    if has_policy and policy_row is not None:
        fig.update_yaxes(title_text="净值(起点=1)", row=policy_row, col=1)
    if erp_path_row is not None:
        fig.update_yaxes(title_text="价格", row=erp_path_row, col=1)
    for _erow in erp_path_rows_extra:
        fig.update_yaxes(title_text="价格", row=int(_erow), col=1)
    return fig


def sanitize_name_for_filename(name: str) -> str:
    cleaned = _UNSAFE_NAME_RE.sub("_", str(name).strip())
    cleaned = cleaned.strip("._")
    return cleaned or "unnamed"


def resolve_display_name(
    overlay: pd.DataFrame,
    code: str,
    override: str | None = None,
    name_map: dict[str, str] | None = None,
) -> str:
    if override:
        return override
    if name_map:
        mapped = name_map.get(str(code).upper())
        if mapped:
            return mapped
    if "name" in overlay.columns:
        names = overlay["name"].dropna().astype(str)
        if not names.empty:
            return names.iloc[0]
    return str(code).upper()


def name_map_from_oos(oos: pd.DataFrame) -> dict[str, str]:
    if "name" not in oos.columns:
        return {}
    frame = oos.copy()
    frame["code"] = frame["code"].astype(str).str.upper()
    frame["name"] = frame["name"].astype(str)
    frame = frame.dropna(subset=["code", "name"])
    frame = frame[frame["name"].str.strip() != ""]
    if frame.empty:
        return {}
    # Latest non-empty name per code.
    frame = frame.sort_values("as_of") if "as_of" in frame.columns else frame
    return (
        frame.drop_duplicates(subset=["code"], keep="last")
        .set_index("code")["name"]
        .to_dict()
    )


def resolve_pool_codes(
    *,
    oos: pd.DataFrame,
    cluster_mapping: Path | None = None,
    codes: list[str] | None = None,
) -> list[str]:
    """Return ordered codes to plot.

    Prefer explicit ``codes``; otherwise use cluster_mapping_selected intersected
    with codes present in validation signals_oos.
    """
    oos_codes = (
        oos["code"].astype(str).str.upper().drop_duplicates().tolist()
        if not oos.empty
        else []
    )
    oos_set = set(oos_codes)

    if codes:
        wanted = [str(c).upper() for c in codes]
        missing = [c for c in wanted if c not in oos_set]
        if missing:
            raise ValueError(
                f"codes not found in signals_oos.csv: {', '.join(missing)}"
            )
        return wanted

    mapping_path = Path(cluster_mapping or CLUSTER_MAPPING_SELECTED_TXT)
    if mapping_path.exists():
        pool = list(load_cluster_mapping_codes(mapping_path))
        selected = [c for c in pool if c in oos_set]
        if selected:
            return selected
        # Fall through if mapping exists but has no overlap.

    if not oos_codes:
        raise ValueError("no codes available in signals_oos.csv")
    return oos_codes


def build_figure(
    *,
    code: str,
    start_date: str,
    end_date: str,
    oos: pd.DataFrame,
    rev: pd.DataFrame,
    evt: pd.DataFrame,
    provider_uri: str | None = None,
    display_name: str | None = None,
    name_map: dict[str, str] | None = None,
    template: str = "plotly_white",
    init_qlib: bool = True,
    kline_plotter_cls: Any | None = None,
    live_snapshot: pd.DataFrame | None = None,
    trade_bias: pd.DataFrame | None = None,
    promotion: dict | None = None,
    need_frame: pd.DataFrame | None = None,
    policy_frame: pd.DataFrame | None = None,
    policy_meta: dict | None = None,
    train_cutoff: str | None = None,
    fair_path_extra_trail_years: float | list[float] | tuple[float, ...] | str | None = None,
    erp_ann: float = ANCHOR_ERP_ANN,
    equity_anchor_label: str = EQUITY_ANCHOR_LABEL,
    ohlcv_full: pd.DataFrame | None = None,
) -> tuple[go.Figure, pd.DataFrame, str]:
    if ohlcv_full is not None and not ohlcv_full.empty:
        ohlcv_full = ohlcv_full.copy()
    else:
        ohlcv_full = load_qlib_ohlcv(
            code,
            start_date,
            end_date,
            provider_uri=provider_uri,
            init_qlib=init_qlib,
            lookback_calendar_days=fair_path_lookback_calendar_days(
                extra_trail_years=fair_path_extra_trail_years
            ),
            clip_to_window=False,
        )
    provisional_note = ""
    if live_snapshot is not None and not live_snapshot.empty:
        ohlcv_full = apply_live_snapshot_to_ohlcv(
            ohlcv_full, live_snapshot, code=code, as_of=end_date
        )
        meta = snapshot_meta(live_snapshot)
        captured = meta.get("captured_at") or "unknown"
        provisional_note = (
            f"<br><span style='font-size:12px;color:#a60'>"
            f"盘中快照 provisional · captured_at={captured} · 非收盘正式数据"
            f"</span>"
        )

    close_full = ohlcv_full.set_index(pd.to_datetime(ohlcv_full["datetime"]).dt.normalize())[
        "$close"
    ]
    # Keep last close if duplicate dates after live snapshot append.
    close_full = close_full[~close_full.index.duplicated(keep="last")]
    oos_ext, rev_ext = extend_validation_with_pre_oos_walkforward(
        oos,
        rev,
        code=code,
        close=close_full,
        start_date=start_date,
        end_date=end_date,
    )
    n_pre = 0
    if not oos_ext.empty and "switch_source" in oos_ext.columns:
        n_pre = int(
            (
                oos_ext["code"].astype(str).str.upper().eq(str(code).upper())
                & oos_ext["switch_source"].eq("walkforward_pre_oos")
            ).sum()
        )
    if n_pre:
        print(
            f"[INFO] {code} pre-OOS walkforward triangles={n_pre} "
            f"(before official signals_oos)"
        )
    overlay = build_overlay_frame(
        oos_ext,
        rev_ext,
        evt,
        code=code,
        start_date=start_date,
        end_date=end_date,
        trade_bias=trade_bias,
        promotion=promotion,
    )
    vol_frame = compute_volatility_regime_frame(close_full)

    # Per-day causal MA20 trend state over the full warmup series.
    trend_frame = trend_state_series(close_full)
    trend_snap = trend_state_summary(close_full, pd.Timestamp(end_date))

    # SMA on full warmup series so MA20 is defined from plot start_date.
    ohlcv_full = attach_sma_columns(ohlcv_full)

    ohlcv = ohlcv_full[
        (pd.to_datetime(ohlcv_full["datetime"]) >= pd.Timestamp(start_date))
        & (pd.to_datetime(ohlcv_full["datetime"]) <= pd.Timestamp(end_date))
    ].copy()
    if ohlcv.empty:
        raise ValueError(
            f"Qlib OHLCV empty for {code} in [{start_date}, {end_date}] after clip"
        )

    name = resolve_display_name(overlay, code, display_name, name_map=name_map)
    title = f"{name}({code}) 分位切换与反转验证{provisional_note}"

    # Trend line in title: state recognition only, not direction prediction.
    _dist_txt = f"{trend_snap['dist_pct']:+.1f}%"
    if trend_snap["state"] == "up" and trend_snap.get("trend_since"):
        _state_txt = f"趋势上涨（距MA20 {_dist_txt}，站上自{trend_snap['trend_since']}）"
    elif trend_snap["state"] == "down" and trend_snap.get("last_break"):
        _state_txt = f"趋势下跌（距MA20 {_dist_txt}，跌破自{trend_snap['last_break']}）"
    else:
        _state_txt = f"{trend_snap['state_zh']}（距MA20 {_dist_txt}）"
    trend_note = (
        f"<br><span style='font-size:12px;color:#06c'>"
        f"趋势状态={_state_txt} · MA20={_fmt_px(trend_snap['ma20'])} · "
        f"状态≠预测（20日前瞻跌破后统计偏中性，勿当买卖信号）"
        f"</span>"
    )

    # kline_plotter_cls kept for backward-compatible signature / tests.
    _ = kline_plotter_cls
    fig = build_candle_volume_vol_figure(
        ohlcv,
        vol_frame,
        template=template,
        need_frame=need_frame,
        erp_ann=float(erp_ann),
        equity_anchor_label=str(equity_anchor_label or EQUITY_ANCHOR_LABEL),
        policy_frame=policy_frame,
        policy_meta=policy_meta,
        train_cutoff=train_cutoff,
        fair_path_ohlcv=ohlcv_full,
        fair_path_extra_trail_years=fair_path_extra_trail_years,
        hover_asset_name=name,
    )
    extreme_dates = compute_extreme_drop_dates(ohlcv, overlay)
    fig = enrich_candle_hover_with_pct_chg(
        fig, ohlcv, trend_frame=trend_frame, extreme_dates=extreme_dates
    )
    fig = annotate_regime_overlays(
        fig, overlay, ohlcv, title=f"{title}{trend_note}", vol_frame=vol_frame
    )
    return fig, overlay, name


def default_html_path(
    code: str,
    name: str,
    start_date: str,
    end_date: str,
    out_dir: Path,
) -> Path:
    start_tag = pd.Timestamp(start_date).strftime("%Y%m%d")
    end_tag = pd.Timestamp(end_date).strftime("%Y%m%d")
    safe_name = sanitize_name_for_filename(name)
    return (
        out_dir
        / f"regime_transition_{code.upper()}_{safe_name}_{start_tag}_{end_tag}.html"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export K-line HTML with regime-transition overlays. "
            "Default: all cluster_mapping_selected pool codes."
        )
    )
    p.add_argument(
        "--code",
        action="append",
        default=None,
        help="optional instrument code; repeatable. Omit to plot the whole pool.",
    )
    p.add_argument("--start-date", default=DEFAULT_START)
    p.add_argument("--end-date", default=DEFAULT_END)
    p.add_argument(
        "--validation-dir",
        type=Path,
        default=DEFAULT_VALIDATION_DIR,
        help="directory containing the three validation CSVs",
    )
    p.add_argument(
        "--cluster-mapping",
        type=Path,
        default=None,
        help="cluster_mapping_selected.txt (default: runtime_paths)",
    )
    p.add_argument(
        "--provider-uri",
        default=None,
        help="Qlib provider URI (default: runtime_paths.QLIB_PROVIDER_URI)",
    )
    p.add_argument(
        "--name",
        default=None,
        help="Chinese display name override (only used with a single --code)",
    )
    p.add_argument(
        "--html-out-dir",
        "--output-dir",
        dest="html_out_dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR,
        help="directory for auto-named HTML files (alias: --output-dir)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="explicit HTML path (single --code only)",
    )
    p.add_argument("--template", default="plotly_white")
    p.add_argument(
        "--dump-overlay-csv",
        type=Path,
        default=None,
        help="optional path to dump merged overlay table (single --code only)",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="skip failed symbols in batch mode instead of aborting",
    )
    p.add_argument(
        "--live-snapshot",
        type=Path,
        default=None,
        help="frozen AkShare OHLCV snapshot CSV to append provisional end-date candle",
    )
    p.add_argument(
        "--trade-bias-csv",
        type=Path,
        default=None,
        help=(
            "optional walk-forward trade-bias predictions CSV; "
            "default: <validation-dir>/assessment/trade_bias_predictions.csv"
        ),
    )
    p.add_argument(
        "--promotion-json",
        type=Path,
        default=None,
        help=(
            "optional bias_promotion.json; "
            "default: <validation-dir>/assessment/bias_promotion.json"
        ),
    )
    return p.parse_args(argv)


def _resolve_path(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


def export_one(
    *,
    code: str,
    start_date: str,
    end_date: str,
    oos: pd.DataFrame,
    rev: pd.DataFrame,
    evt: pd.DataFrame,
    out_dir: Path,
    provider_uri: str | None,
    display_name: str | None,
    name_map: dict[str, str],
    template: str,
    output: Path | None,
    dump_overlay_csv: Path | None,
    init_qlib: bool,
    kline_plotter_cls: Any,
    live_snapshot: pd.DataFrame | None = None,
    trade_bias: pd.DataFrame | None = None,
    promotion: dict | None = None,
) -> Path:
    fig, overlay, name = build_figure(
        code=code,
        start_date=start_date,
        end_date=end_date,
        oos=oos,
        rev=rev,
        evt=evt,
        provider_uri=provider_uri,
        display_name=display_name,
        name_map=name_map,
        template=template,
        init_qlib=init_qlib,
        kline_plotter_cls=kline_plotter_cls,
        live_snapshot=live_snapshot,
        trade_bias=trade_bias,
        promotion=promotion,
    )
    if output is not None:
        out_path = _resolve_path(output)
    else:
        out_path = default_html_path(code, name, start_date, end_date, out_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_adaptive_html(fig, out_path)
    print(f"[INFO] wrote HTML: {out_path}")

    if dump_overlay_csv is not None:
        dump = _resolve_path(dump_overlay_csv)
        dump.parent.mkdir(parents=True, exist_ok=True)
        overlay.to_csv(dump, index=False)
        print(f"[INFO] wrote overlay CSV: {dump}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validation_dir = _resolve_path(args.validation_dir)
    out_dir = _resolve_path(args.html_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_mapping = (
        _resolve_path(args.cluster_mapping)
        if args.cluster_mapping is not None
        else CLUSTER_MAPPING_SELECTED_TXT
    )

    try:
        oos, rev, evt = load_validation_csvs(validation_dir)
        codes = resolve_pool_codes(
            oos=oos,
            cluster_mapping=cluster_mapping,
            codes=args.code,
        )
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if args.output is not None and len(codes) != 1:
        print(
            "[ERROR] --output requires exactly one --code (or a one-code pool)",
            file=sys.stderr,
        )
        return 2
    if args.dump_overlay_csv is not None and len(codes) != 1:
        print(
            "[ERROR] --dump-overlay-csv requires exactly one --code",
            file=sys.stderr,
        )
        return 2
    if args.name is not None and len(codes) != 1:
        print(
            "[ERROR] --name only applies when plotting a single --code",
            file=sys.stderr,
        )
        return 2

    name_map = name_map_from_oos(oos)
    provider_uri = args.provider_uri
    live_snapshot = None
    if args.live_snapshot is not None:
        snap_path = _resolve_path(args.live_snapshot)
        try:
            live_snapshot = load_live_snapshot(
                snap_path,
                as_of=args.end_date,
                required_codes=codes,
                strict_coverage=True,
            )
        except LiveSnapshotError as exc:
            print(f"[ERROR] live snapshot: {exc}", file=sys.stderr)
            return 2
        meta = snapshot_meta(live_snapshot, path=snap_path)
        print(
            f"[INFO] live snapshot loaded as_of={meta.get('as_of')} "
            f"codes={meta.get('n_codes')} captured_at={meta.get('captured_at')}"
        )

    trade_bias_path = (
        _resolve_path(args.trade_bias_csv)
        if args.trade_bias_csv is not None
        else validation_dir / "assessment" / "trade_bias_predictions.csv"
    )
    trade_bias = load_trade_bias_predictions(trade_bias_path)
    if trade_bias is None:
        print(
            f"[INFO] trade-bias CSV missing/invalid → hover omits 5日模型参考 "
            f"({trade_bias_path}); triangle labels default observe_only"
        )
    else:
        print(
            f"[INFO] trade-bias predictions loaded rows={len(trade_bias)} "
            f"from {trade_bias_path} (hover 5日模型参考 only; "
            f"triangle labels gated by bias_promotion.json)"
        )

    promo_path = (
        _resolve_path(args.promotion_json)
        if args.promotion_json is not None
        else default_promotion_path(validation_dir)
    )
    promotion = load_bias_promotion(promo_path)
    print(
        f"[INFO] bias promotion source={promotion.get('promotion_source')} "
        f"buy_rule={promotion.get('buy_rule')} sell_rule={promotion.get('sell_rule')} "
        f"from {promo_path}"
    )

    try:
        _init_qlib(provider_uri)
        kline_plotter_cls = _load_kline_plotter_class()
    except Exception as exc:  # noqa: BLE001 - surface init failures cleanly
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    oos = enrich_oos_from_ohlcv_loader(oos, load_qlib_ohlcv)
    n_override = int(
        oos.get("extreme_drop_switch", pd.Series(dtype=bool)).fillna(False).sum()
    )
    if n_override:
        print(f"[INFO] extreme_drop_override reds={n_override}")

    print(
        f"[INFO] plotting {len(codes)} code(s) "
        f"[{args.start_date} -> {args.end_date}] -> {out_dir}"
    )

    ok = 0
    failed: list[tuple[str, str]] = []
    for code in codes:
        try:
            export_one(
                code=code,
                start_date=args.start_date,
                end_date=args.end_date,
                oos=oos,
                rev=rev,
                evt=evt,
                out_dir=out_dir,
                provider_uri=provider_uri,
                display_name=args.name,
                name_map=name_map,
                template=args.template,
                output=args.output,
                dump_overlay_csv=args.dump_overlay_csv,
                init_qlib=False,
                kline_plotter_cls=kline_plotter_cls,
                live_snapshot=live_snapshot,
                trade_bias=trade_bias,
                promotion=promotion,
            )
            ok += 1
        except (FileNotFoundError, ValueError, OSError, LiveSnapshotError) as exc:
            failed.append((code, str(exc)))
            print(f"[ERROR] {code}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                return 2

    print(f"[INFO] done: ok={ok} failed={len(failed)} total={len(codes)}")
    if failed:
        for code, msg in failed:
            print(f"[ERROR] failed {code}: {msg}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
