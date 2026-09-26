"""Fair-path band auxiliary buy/sell signals (fig7–11, research only).

Matches HTML diagnostics in plot_regime_transition_example.py:
  - trailing_log_trend_price_path (own log OLS, causal)
  - erp_path_symmetric_envelope (expanding P95 |ln(c/p)|)

Does NOT change production hold_up rules.
"""
from __future__ import annotations

import importlib.util
import json
import re
import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIG_PANELS: tuple[tuple[str, float, str], ...] = (
    ("fig7", 2.0, "2年"),
    ("fig8", 1.0, "1年"),
    ("fig9", 0.5, "6个月"),
    ("fig10", 0.25, "3个月"),
    ("fig11", 1.0 / 12.0, "1个月"),
)

DEFAULT_DEEP_GAP = -0.15
DEFAULT_MIN_RR = 2.0

ENTRY_MODES = (
    "baseline_touch",
    "B12",
    "B123",
    "B1235",
    "B12345",
)


def _decode_plotly_bdata(obj: Any) -> Any:
    if isinstance(obj, dict) and "bdata" in obj and "dtype" in obj:
        dtype = obj["dtype"]
        raw = base64.b64decode(obj["bdata"])
        mapping = {"f8": "<f8", "f4": "<f4", "i4": "<i4", "i8": "<i8", "u1": "<u1", "i1": "<i1"}
        return np.frombuffer(raw, dtype=np.dtype(mapping.get(dtype, dtype)))
    return obj


def _walk_plotly_json(o: Any) -> Any:
    if isinstance(o, list):
        return [_walk_plotly_json(x) for x in o]
    if isinstance(o, dict):
        if "bdata" in o and "dtype" in o:
            return _decode_plotly_bdata(o)
        return {k: _walk_plotly_json(v) for k, v in o.items()}
    return o


def _extract_plotly_traces(html: str) -> list[dict[str, Any]]:
    m = re.search(r'Plotly\.newPlot\(\s*"[^"]+"\s*,\s*(\[)', html)
    if not m:
        raise ValueError("Plotly.newPlot data array not found")
    start = m.start(1)
    i = start
    depth = 0
    in_str = False
    esc = False
    while i < len(html):
        c = html[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        i += 1
    else:
        raise ValueError("unclosed Plotly data array")
    return _walk_plotly_json(json.loads(html[start:end]))


def _plotly_x_to_index(x: Any) -> pd.DatetimeIndex:
    arr = np.asarray(x)
    if np.issubdtype(arr.dtype, np.number):
        return pd.to_datetime(arr, unit="ms")
    # Incremental HTML appends date-only last bars; Plotly history is ISO datetime.
    return pd.to_datetime(arr, format="ISO8601")


def load_ohlcv_from_regime_html(path: Path | str) -> tuple[pd.Series, pd.DataFrame]:
    """Load qfq OHLCV from regime_transition HTML fig7 candlestick trace."""
    html = Path(path).read_text(encoding="utf-8")
    data = _extract_plotly_traces(html)
    candle = None
    for tr in data:
        if tr.get("type") != "candlestick":
            continue
        if tr.get("yaxis") == "y10":
            candle = tr
            break
    if candle is None:
        for tr in data:
            if tr.get("type") == "candlestick" and str(tr.get("name", "")).startswith("K线"):
                yax = str(tr.get("yaxis", "y"))
                if yax in {"y10", "y11", "y12", "y13", "y14"}:
                    candle = tr
                    break
    if candle is None:
        raise ValueError(f"fig7 candlestick not found in {path}")
    idx = _plotly_x_to_index(candle["x"])
    ohlcv = pd.DataFrame(
        {
            "$open": np.asarray(candle["open"], dtype=float),
            "$high": np.asarray(candle["high"], dtype=float),
            "$low": np.asarray(candle["low"], dtype=float),
            "$close": np.asarray(candle["close"], dtype=float),
        },
        index=idx,
    ).sort_index()
    ohlcv.index.name = "datetime"
    close = ohlcv["$close"].astype(float)
    return close, ohlcv


def find_regime_html(listing_dir: Path, code: str) -> Path | None:
    """Resolve regime_transition HTML for a symbol under listing_dir."""
    code = code.upper()
    matches = sorted(listing_dir.glob(f"regime_transition_{code}_*_adaptive.html"))
    return matches[0] if matches else None


def _load_plot_mod():
    path = _PROJECT_ROOT / "decision_pack/scripts/plot_regime_transition_example.py"
    spec = importlib.util.spec_from_file_location("plot_regime_pm_fp", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_qfq_close(
    code: str,
    start: str,
    end: str,
    *,
    init_qlib: bool = True,
) -> tuple[pd.Series, pd.DataFrame]:
    """Load qfq close + ohlcv indexed by datetime."""
    pm = _load_plot_mod()
    ohlcv = pm.load_qlib_ohlcv(
        code,
        start,
        end,
        init_qlib=init_qlib,
        lookback_calendar_days=0,
        clip_to_window=False,
        apply_qfq=True,
    )
    ohlcv = ohlcv.copy()
    ohlcv["datetime"] = pd.to_datetime(ohlcv["datetime"])
    ohlcv = ohlcv.set_index("datetime").sort_index()
    close = pm._asof_close_series(ohlcv.reset_index())
    close.index = pd.to_datetime(close.index)
    return close, ohlcv


def atr20(ohlcv: pd.DataFrame) -> pd.Series:
    h = ohlcv["$high"].astype(float)
    l = ohlcv["$low"].astype(float)
    c = ohlcv["$close"].astype(float)
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(20).mean()


def compute_multi_scale_frame(
    close: pd.Series,
    *,
    min_bars_fn=None,
) -> pd.DataFrame:
    """One row per date; columns path/g/upper/lower/gap per fig panel."""
    pm = _load_plot_mod()
    if min_bars_fn is None:
        min_bars_fn = pm._trail_fit_min_bars
    idx = close.index
    out: dict[str, pd.Series] = {"px": close.astype(float)}
    for key, years, _ in FIG_PANELS:
        path, g = pm.trailing_log_trend_price_path(
            close,
            trail_years=float(years),
            min_bars=min_bars_fn(float(years)),
        )
        upper, lower, _delta = pm.erp_path_symmetric_envelope(path, close)
        out[f"{key}_path"] = path
        out[f"{key}_g"] = g
        out[f"{key}_upper"] = upper
        out[f"{key}_lower"] = lower
        gap = close / path - 1.0
        gap = gap.where(path > 0)
        out[f"{key}_gap"] = gap
        touch_lo = close <= lower
        touch_hi = close >= upper
        out[f"{key}_touch_lo"] = touch_lo
        out[f"{key}_touch_hi"] = touch_hi
        above_path = close >= path
        out[f"{key}_above_path"] = above_path
    df = pd.DataFrame(out, index=idx)
    df.index.name = "date"
    return df


def path_rising(path: pd.Series, *, window: int = 20) -> pd.Series:
    """True when path level rose over trailing window."""
    return path > path.shift(window)


@dataclass
class ChecklistResult:
    date: str
    code: str
    px: float
    regime: str | None = None
    has_ed: bool = False
    has_ru: bool = False
    production_buy: bool = False
    production_sell: bool = False
    flags: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    rr_to_fig8_path: float | None = None
    rr_to_fig10_path: float | None = None
    verdict: str = "neutral"

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "code": self.code,
            "px": self.px,
            "regime": self.regime,
            "has_ed": self.has_ed,
            "has_ru": self.has_ru,
            "production_buy": self.production_buy,
            "production_sell": self.production_sell,
            "verdict": self.verdict,
            "rr_fig8": self.rr_to_fig8_path,
            "rr_fig10": self.rr_to_fig10_path,
            **{f"flag_{k}": v for k, v in self.flags.items()},
            "notes": "|".join(self.notes),
        }


def _rr_reward_risk(
    px: float,
    target: float,
    stop: float,
) -> float | None:
    if not (np.isfinite(px) and np.isfinite(target) and np.isfinite(stop)):
        return None
    if px <= 0 or stop <= 0 or stop >= px or target <= px:
        return None
    reward = target / px - 1.0
    risk = px / stop - 1.0
    if risk <= 0 or reward <= 0:
        return None
    return float(reward / risk)


def evaluate_checklist_row(
    row: pd.Series,
    *,
    code: str,
    atr: float | None = None,
    regime: str | None = None,
    has_ed: bool = False,
    has_ru: bool = False,
    production_buy: bool = False,
    production_sell: bool = False,
    deep_gap: float = DEFAULT_DEEP_GAP,
    min_rr: float = DEFAULT_MIN_RR,
    listing_age_years: float | None = None,
    touch_panels: tuple[str, ...] = ("fig9", "fig10"),
    b3_mode: Literal["default", "fig10_only", "reclaim_from_lo"] = "default",
) -> ChecklistResult:
    """Score one day against B1–B6 / S1–S3."""
    px = float(row["px"])
    d = pd.Timestamp(row.name).strftime("%Y-%m-%d")
    res = ChecklistResult(
        date=d,
        code=code,
        px=px,
        regime=regime,
        has_ed=has_ed,
        has_ru=has_ru,
        production_buy=production_buy,
        production_sell=production_sell,
    )

    # B1: touch lower selected panels or deep discount fig8
    touch_lo = False
    for p in touch_panels:
        if bool(row.get(f"{p}_touch_lo", False)):
            touch_lo = True
            break
    deep_disc = bool(
        float(row.get("fig8_gap", 0)) <= deep_gap if pd.notna(row.get("fig8_gap")) else False
    )
    b1 = touch_lo or deep_disc
    res.flags["B1_position"] = b1

    # B2: long scale g>0 or path rising (prefer fig8 if young listing)
    use_fig7 = listing_age_years is None or listing_age_years >= 1.9
    g7 = float(row.get("fig7_g", np.nan))
    g8 = float(row.get("fig8_g", np.nan))
    path7_up = bool(row.get("fig7_path_rising", False)) if "fig7_path_rising" in row else False
    path8_up = bool(row.get("fig8_path_rising", False)) if "fig8_path_rising" in row else False
    b2 = bool(
        (use_fig7 and np.isfinite(g7) and g7 > 0)
        or (np.isfinite(g8) and g8 > 0)
        or path8_up
        or (use_fig7 and path7_up)
    )
    res.flags["B2_long_ok"] = b2

    # B3: short reclaim — above path or off lower band (fig10/11)
    prev_touch = bool(row.get("fig10_prev_touch_lo", False))
    reclaim = bool(
        prev_touch
        and pd.notna(row.get("fig10_lower"))
        and px > float(row["fig10_lower"])
    )
    if b3_mode == "default":
        b3 = bool(
            row.get("fig10_above_path", False)
            or row.get("fig11_above_path", False)
            or reclaim
        )
    elif b3_mode == "fig10_only":
        b3 = bool(row.get("fig10_above_path", False) or reclaim)
    else:  # reclaim_from_lo
        b3 = reclaim
    res.flags["B3_short_reclaim"] = b3

    # B4: conflict
    rg = (regime or "range").lower()
    b4 = not has_ed and (rg == "up" or has_ru or (rg == "range" and has_ru))
    if not has_ed and rg == "range" and not has_ru:
        b4 = False  # wait
    if rg == "down" and not has_ru:
        b4 = False
    res.flags["B4_no_conflict"] = b4

    # B5: R:R to fig8 or fig10 path
    stop = px - float(atr) if atr is not None and np.isfinite(atr) else px * 0.94
    t8 = float(row.get("fig8_path", np.nan))
    t10 = float(row.get("fig10_path", np.nan))
    res.rr_to_fig8_path = _rr_reward_risk(px, t8, stop) if np.isfinite(t8) else None
    res.rr_to_fig10_path = _rr_reward_risk(px, t10, stop) if np.isfinite(t10) else None
    b5 = bool(
        (res.rr_to_fig8_path is not None and res.rr_to_fig8_path >= min_rr)
        or (res.rr_to_fig10_path is not None and res.rr_to_fig10_path >= min_rr)
    )
    res.flags["B5_rr_ok"] = b5

    res.flags["B6_production_buy"] = production_buy

    # S1/S2/S3
    s1 = bool(row.get("fig10_touch_hi", False) or row.get("fig9_touch_hi", False))
    res.flags["S1_upper_band"] = s1
    g10 = float(row.get("fig10_g", np.nan))
    s2 = bool(np.isfinite(g10) and g10 < 0 and not row.get("fig10_above_path", True))
    res.flags["S2_repair_fail"] = s2
    res.flags["S3_production_sell"] = production_sell

    # Verdict
    if production_sell or s1:
        res.verdict = "trim_or_sell"
        res.notes.append("S1/S3")
    elif production_buy:
        res.verdict = "production_buy"
        res.notes.append("B6")
    elif b1 and b2 and b3 and b4 and b5:
        res.verdict = "strong_aux_buy"
    elif b1 and b2 and b3:
        res.verdict = "aux_buy_watch"
    elif b1 and not b2:
        res.verdict = "avoid_falling_knife"
        res.notes.append("B1 without B2")
    elif b1 and b2 and not b3:
        res.verdict = "observe_no_entry"
    else:
        res.verdict = "neutral"

    return res


def enrich_frame_for_checklist(ms: pd.DataFrame) -> pd.DataFrame:
    """Add path_rising and prev_touch_lo helpers."""
    out = ms.copy()
    for key, _, _ in FIG_PANELS:
        pcol = f"{key}_path"
        if pcol in out.columns:
            out[f"{key}_path_rising"] = path_rising(out[pcol], window=20)
    if "fig10_touch_lo" in out.columns:
        prev = out["fig10_touch_lo"].shift(1)
        out["fig10_prev_touch_lo"] = prev.eq(True).fillna(False).astype(bool)
    return out


def build_checklist_series(
    ms: pd.DataFrame,
    *,
    code: str,
    atr_s: pd.Series | None = None,
    regime_s: pd.Series | None = None,
    ed_dates: set[str] | None = None,
    ru_dates: set[str] | None = None,
    buy_dates: set[str] | None = None,
    sell_dates: set[str] | None = None,
    listing_start: str | None = None,
) -> pd.DataFrame:
    ms = enrich_frame_for_checklist(ms)
    ed_dates = ed_dates or set()
    ru_dates = ru_dates or set()
    buy_dates = buy_dates or set()
    sell_dates = sell_dates or set()
    listing_age_years = None
    if listing_start:
        listing_age_years = (
            pd.Timestamp(ms.index[-1]) - pd.Timestamp(listing_start)
        ).days / 365.25

    rows: list[dict[str, Any]] = []
    for dt, row in ms.iterrows():
        ds = pd.Timestamp(dt).strftime("%Y-%m-%d")
        atr = float(atr_s.loc[dt]) if atr_s is not None and dt in atr_s.index else None
        regime = str(regime_s.loc[dt]) if regime_s is not None and dt in regime_s.index else None
        cr = evaluate_checklist_row(
            row,
            code=code,
            atr=atr,
            regime=regime,
            has_ed=ds in ed_dates,
            has_ru=ds in ru_dates,
            production_buy=ds in buy_dates,
            production_sell=ds in sell_dates,
            listing_age_years=listing_age_years,
        )
        rows.append(cr.to_dict())
    return pd.DataFrame(rows)


def build_checklist_flags_frame(
    ms: pd.DataFrame,
    *,
    code: str,
    atr_s: pd.Series | None = None,
    regime_s: pd.Series | None = None,
    ed_dates: set[str] | None = None,
    ru_dates: set[str] | None = None,
    buy_dates: set[str] | None = None,
    sell_dates: set[str] | None = None,
    listing_start: str | None = None,
    deep_gap: float = DEFAULT_DEEP_GAP,
    min_rr: float = DEFAULT_MIN_RR,
    touch_panels: tuple[str, ...] = ("fig9", "fig10"),
    b3_mode: Literal["default", "fig10_only", "reclaim_from_lo"] = "default",
) -> pd.DataFrame:
    """Vector-friendly daily checklist flags for backtests."""
    ms_en = enrich_frame_for_checklist(ms)
    ed_dates = ed_dates or set()
    ru_dates = ru_dates or set()
    buy_dates = buy_dates or set()
    sell_dates = sell_dates or set()
    listing_age_years = None
    if listing_start and len(ms_en.index):
        listing_age_years = (
            pd.Timestamp(ms_en.index[-1]) - pd.Timestamp(listing_start)
        ).days / 365.25

    rows: list[dict[str, Any]] = []
    for dt, row in ms_en.iterrows():
        ds = pd.Timestamp(dt).strftime("%Y-%m-%d")
        atr = float(atr_s.loc[dt]) if atr_s is not None and dt in atr_s.index else None
        regime = str(regime_s.loc[dt]) if regime_s is not None and dt in regime_s.index else None
        cr = evaluate_checklist_row(
            row,
            code=code,
            atr=atr,
            regime=regime,
            has_ed=ds in ed_dates,
            has_ru=ds in ru_dates,
            production_buy=ds in buy_dates,
            production_sell=ds in sell_dates,
            deep_gap=deep_gap,
            min_rr=min_rr,
            listing_age_years=listing_age_years,
            touch_panels=touch_panels,
            b3_mode=b3_mode,
        )
        rows.append(
            {
                "date": ds,
                "B1": cr.flags.get("B1_position", False),
                "B2": cr.flags.get("B2_long_ok", False),
                "B3": cr.flags.get("B3_short_reclaim", False),
                "B4": cr.flags.get("B4_no_conflict", False),
                "B5": cr.flags.get("B5_rr_ok", False),
            }
        )
    out = pd.DataFrame(rows)
    out.index = ms_en.index
    return out


def entries_from_flags(flags: pd.DataFrame, *, mode: str) -> pd.Series:
    """Combine precomputed B1–B5 flags into entry series."""
    if mode == "baseline_touch":
        raise ValueError("use detect_lower_band_entries for baseline_touch")
    ok = flags["B1"] & flags["B2"]
    if mode in {"B123", "B1235", "B12345"}:
        ok = ok & flags["B3"]
    if mode in {"B1235", "B12345"}:
        ok = ok & flags["B5"]
    if mode == "B12345":
        ok = ok & flags["B4"]
    return ok.astype(bool)


def detect_checklist_entries(
    ms: pd.DataFrame,
    *,
    mode: str = "B123",
    deep_gap: float = DEFAULT_DEEP_GAP,
    min_rr: float = DEFAULT_MIN_RR,
    touch_panels: tuple[str, ...] = ("fig9", "fig10"),
    b3_mode: Literal["default", "fig10_only", "reclaim_from_lo"] = "default",
    code: str = "UNK",
    atr_s: pd.Series | None = None,
    regime_s: pd.Series | None = None,
    ed_dates: set[str] | None = None,
    ru_dates: set[str] | None = None,
    buy_dates: set[str] | None = None,
    sell_dates: set[str] | None = None,
    listing_start: str | None = None,
) -> pd.Series:
    """Research entry booleans aligned with checklist B1–B5 variants."""
    if mode == "baseline_touch":
        return detect_lower_band_entries(ms, panels=touch_panels)

    flags = build_checklist_flags_frame(
        ms,
        code=code,
        atr_s=atr_s,
        regime_s=regime_s,
        ed_dates=ed_dates,
        ru_dates=ru_dates,
        buy_dates=buy_dates,
        sell_dates=sell_dates,
        listing_start=listing_start,
        deep_gap=deep_gap,
        min_rr=min_rr,
        touch_panels=touch_panels,
        b3_mode=b3_mode,
    )
    return entries_from_flags(flags, mode=mode)


def detect_s1_trim_signals(ms: pd.DataFrame) -> pd.Series:
    """S1: close at/above fig9 or fig10 upper band."""
    s1 = pd.Series(False, index=ms.index)
    for p in ("fig9", "fig10"):
        col = f"{p}_touch_hi"
        if col in ms.columns:
            s1 = s1 | ms[col].fillna(False)
    return s1


def detect_lower_band_entries(
    ms: pd.DataFrame,
    *,
    require_long_g: bool = True,
    panels: tuple[str, ...] = ("fig9", "fig10"),
) -> pd.Series:
    """Research entry: touch lower band on fig9/10 + fig7 or fig8 g>0."""
    touch = pd.Series(False, index=ms.index)
    for p in panels:
        col = f"{p}_touch_lo"
        if col in ms.columns:
            touch = touch | ms[col].fillna(False)
    if not require_long_g:
        return touch
    long_ok = pd.Series(False, index=ms.index)
    for p in ("fig7", "fig8"):
        gcol = f"{p}_g"
        if gcol in ms.columns:
            long_ok = long_ok | (ms[gcol] > 0)
    return touch & long_ok


def build_quiet_discount_frame(
    close: pd.Series,
    ohlcv: pd.DataFrame,
    *,
    rv20_anchor: pd.Series,
    rv5_anchor: pd.Series | None = None,
    fig7_gap_max: float = -0.12,
    vol5_pct_max: float = 0.30,
) -> pd.DataFrame:
    """Combine fig3 vol + fig4 need/gap + fig7–11 path columns for quiet_discount.

    Does not use production buy/sell markers. Anchor vols should be CSI300
    (SH510300) series aligned by date.
    """
    from decision_pack.src.regime_transition_plot import compute_volatility_regime_frame
    from decision_pack.src.vol_need_return_board import (
        ANCHOR_ERP_ANN,
        fair_need_ann_series,
        gap_realized_minus_need,
    )

    s = pd.to_numeric(close, errors="coerce").copy()
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()

    ms = compute_multi_scale_frame(s)
    vol = compute_volatility_regime_frame(s)
    vol = vol.copy()
    vol["as_of"] = pd.to_datetime(vol["as_of"]).dt.normalize()
    vol = vol.set_index("as_of").reindex(s.index)

    a20 = pd.to_numeric(rv20_anchor, errors="coerce").copy()
    a20.index = pd.DatetimeIndex(pd.to_datetime(a20.index)).normalize()
    a20 = a20[~a20.index.duplicated(keep="last")].sort_index().reindex(s.index)
    if rv5_anchor is None:
        a5 = a20
    else:
        a5 = pd.to_numeric(rv5_anchor, errors="coerce").copy()
        a5.index = pd.DatetimeIndex(pd.to_datetime(a5.index)).normalize()
        a5 = a5[~a5.index.duplicated(keep="last")].sort_index().reindex(s.index)

    rv20 = pd.to_numeric(vol["rv20"], errors="coerce")
    rv5 = pd.to_numeric(vol["rv5"], errors="coerce")
    need20 = fair_need_ann_series(rv20, a20, erp_ann=ANCHOR_ERP_ANN)
    g20 = gap_realized_minus_need(s, need20, bars=20)

    out = ms.copy()
    out["rv5"] = rv5
    out["rv20"] = rv20
    out["vol5_pct_120d"] = pd.to_numeric(vol["vol5_pct_120d"], errors="coerce")
    out["r_need_ann"] = need20.reindex(s.index)
    out["gap_20d"] = pd.to_numeric(g20["gap"], errors="coerce").reindex(s.index)
    out["atr20"] = atr20(ohlcv).reindex(s.index)

    # Signal components (for diagnostics)
    out["qd_long_discount"] = (out["fig7_gap"] <= float(fig7_gap_max)) | out[
        "fig7_touch_lo"
    ].fillna(False)
    out["qd_long_ok"] = (out["fig7_g"] > 0) & (out["fig8_g"] > 0)
    out["qd_low_vol"] = out["vol5_pct_120d"] <= float(vol5_pct_max)
    out["qd_gap_weak"] = out["gap_20d"] <= 0
    out["qd_not_hot"] = ~(
        out["fig9_touch_hi"].fillna(False) | out["fig10_touch_hi"].fillna(False)
    )
    out["qd_entry"] = (
        out["qd_long_discount"].fillna(False)
        & out["qd_long_ok"].fillna(False)
        & out["qd_low_vol"].fillna(False)
        & out["qd_gap_weak"].fillna(False)
        & out["qd_not_hot"].fillna(False)
    )
    out["qd_exit_s1"] = (
        out["fig9_touch_hi"].fillna(False) | out["fig10_touch_hi"].fillna(False)
    )
    out["qd_exit_reclaim"] = out["fig7_gap"] >= 0
    out["qd_exit_trend_break"] = (out["fig7_g"] < 0) & (out["fig8_g"] < 0)
    return out


def quiet_discount_entries(frame: pd.DataFrame) -> pd.Series:
    """Boolean entry signals from ``build_quiet_discount_frame``."""
    if "qd_entry" not in frame.columns:
        raise KeyError("frame missing qd_entry; call build_quiet_discount_frame first")
    return frame["qd_entry"].fillna(False).astype(bool)


def simulate_quiet_discount_trades(
    frame: pd.DataFrame,
    ohlcv: pd.DataFrame,
    *,
    entries: pd.Series | None = None,
    max_hold: int = 40,
) -> list[dict]:
    """Long-only: next-bar entry; exit on S1 / fig7 reclaim / dual-g break / stop / max_hold."""
    if entries is None:
        entries = quiet_discount_entries(frame)
    close = frame["px"].astype(float)
    low = ohlcv["$low"].reindex(frame.index).astype(float)
    atr_s = frame["atr20"] if "atr20" in frame.columns else atr20(ohlcv).reindex(frame.index)
    trades: list[dict] = []
    i = 0
    n = len(frame)
    while i < n - 1:
        if not bool(entries.iloc[i]):
            i += 1
            continue
        entry_i = i + 1
        if entry_i >= n:
            break
        entry_px = float(close.iloc[entry_i])
        atr_v = (
            float(atr_s.iloc[entry_i])
            if pd.notna(atr_s.iloc[entry_i])
            else entry_px * 0.1
        )
        stop = entry_px - atr_v
        exit_reason = None
        exit_px = entry_px
        exit_i = entry_i
        for j in range(entry_i + 1, n):
            px = float(close.iloc[j])
            lo = float(low.iloc[j])
            hold_days = j - entry_i
            if np.isfinite(stop) and lo <= stop:
                exit_reason = "stop"
                exit_px = stop
                exit_i = j
                break
            if bool(frame["qd_exit_s1"].iloc[j]):
                exit_reason = "s1_upper"
                exit_px = px
                exit_i = j
                break
            reclaim = frame["qd_exit_reclaim"].iloc[j]
            if pd.notna(reclaim) and bool(reclaim):
                exit_reason = "fig7_reclaim"
                exit_px = px
                exit_i = j
                break
            if bool(frame["qd_exit_trend_break"].iloc[j]):
                exit_reason = "trend_break"
                exit_px = px
                exit_i = j
                break
            if hold_days >= max_hold:
                exit_reason = "max_hold"
                exit_px = px
                exit_i = j
                break
        if exit_reason is None:
            exit_reason = "eod"
            exit_px = float(close.iloc[n - 1])
            exit_i = n - 1
        fwd: dict[str, float | None] = {}
        for horizon in (5, 10, 20):
            jh = entry_i + horizon
            if jh < n:
                fwd[f"fwd_ret_{horizon}d"] = float(close.iloc[jh] / entry_px - 1.0)
            else:
                fwd[f"fwd_ret_{horizon}d"] = None
        trades.append(
            dict(
                signal_date=frame.index[i].strftime("%Y-%m-%d"),
                entry_date=frame.index[entry_i].strftime("%Y-%m-%d"),
                exit_date=frame.index[exit_i].strftime("%Y-%m-%d"),
                entry_px=entry_px,
                exit_px=float(exit_px),
                ret=float(exit_px) / entry_px - 1.0,
                hold_days=int(exit_i - entry_i),
                exit_reason=exit_reason,
                stop=float(stop),
                **fwd,
            )
        )
        i = exit_i + 1
    return trades
