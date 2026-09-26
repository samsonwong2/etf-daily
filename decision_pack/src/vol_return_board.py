"""Research board: ETF vol vs 46-pool market price of risk.

Horizons (matched):
- 60d: λ60 = median(r_60a / rv60); r_req_60 = λ60 × rv60; gap_60 = r_60a − r_req_60
- 20d: λ20 = median(r_20a / rv20); r_req_20 = λ20 × rv20; gap_20 = r_20a − r_req_20
- 5d:  λ5  = median(r_5a  / rv5);  r_req_5  = λ5  × rv5;  gap_5  = r_5a  − r_req_5
- inception: λ_incep = median(r_incep / rv_incep); columns r_1y/r_req/gap/sharpe_like

``r_*a`` = trailing N-day return annualized to 252 trading days.
``r_incep`` / ``rv_incep`` = since listing (all bars through as_of), annualized.
Bond is secondary (vol_vs_bond60 only) — not used in r_req.

Pure diagnostics — does not mutate regime labels or trade rules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from decision_pack.src.regime_transition_plot import (
    TRADING_DAYS_PER_YEAR,
    compute_realized_vol,
)

# 30年国债ETF鹏扬 — primary bond fair-need / vol_vs_bond anchor.
DEFAULT_BOND_ANCHOR = "SH511090"
DEFAULT_BOND_AUX = "SH511010"
DEFAULT_DEFENSIVE_CODES = frozenset(
    {DEFAULT_BOND_ANCHOR, DEFAULT_BOND_AUX, "SH511260"}
)
DEFAULT_QLIB_INSTRUMENTS = Path(
    "~/etf-daily-output/data/qlib_data/all_fund_data/instruments/all.txt"
)
RV5_WINDOW = 5
RV20_WINDOW = 20
RV60_WINDOW = 60
LAMBDA_FALLBACK = 0.3
MIN_SIGMA = 1e-4


def load_instrument_codes(
    path: Path | str,
    *,
    as_of: str | pd.Timestamp | None = None,
) -> list[str]:
    """Load Qlib instruments file (code start end). Optionally filter by as_of."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    end = pd.Timestamp(as_of).normalize() if as_of is not None else None
    codes: list[str] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", "\t").split()
            if not parts:
                continue
            code = str(parts[0]).upper()
            if end is not None and len(parts) >= 3:
                try:
                    start_d = pd.Timestamp(parts[1]).normalize()
                    end_d = pd.Timestamp(parts[2]).normalize()
                except Exception:
                    codes.append(code)
                    continue
                if start_d > end or end_d < end:
                    continue
            codes.append(code)
    return list(dict.fromkeys(codes))


def load_fund_name_map(path: Path | str) -> dict[str, str]:
    """Load code→name from ``temp/fund_list.csv`` (基金代码 / 基金简称)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    code_col = "基金代码" if "基金代码" in df.columns else None
    name_col = "基金简称" if "基金简称" in df.columns else None
    if code_col is None or name_col is None:
        # fallback english-ish headers
        for c in df.columns:
            cl = str(c).lower()
            if code_col is None and ("code" in cl or "代码" in str(c)):
                code_col = c
            if name_col is None and ("name" in cl or "简称" in str(c) or "名称" in str(c)):
                name_col = c
    if code_col is None or name_col is None:
        raise KeyError(f"fund_list missing code/name cols: {list(df.columns)}")
    out: dict[str, str] = {}
    for _, r in df.iterrows():
        raw = str(r[code_col]).strip()
        if not raw or raw.lower() == "nan":
            continue
        code = raw.upper()
        if not code.startswith(("SH", "SZ")) and len(code) >= 6:
            # sh588170 / sz159327
            if raw[:2].lower() in ("sh", "sz"):
                code = raw[:2].upper() + raw[2:]
            else:
                code = code
        name = str(r[name_col]).strip()
        if name and name.lower() != "nan":
            out[code] = name
    return out


def _as_of_ts(as_of: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(as_of).normalize()


def slice_close_through(close: pd.Series, as_of: str | pd.Timestamp) -> pd.Series:
    """Causal close series through ``as_of`` (inclusive), numeric, sorted."""
    end = _as_of_ts(as_of)
    s = pd.to_numeric(close, errors="coerce").dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s = s.loc[s.index.normalize() <= end]
    return s


def trailing_realized_annual_return(close: pd.Series, *, bars: int) -> float:
    """Compound annualized return over the last ``bars`` trading intervals."""
    s = pd.to_numeric(close, errors="coerce").dropna()
    if len(s) < 2:
        return float("nan")
    need = int(bars) + 1
    window = s.tail(need) if len(s) >= need else s
    start = float(window.iloc[0])
    end = float(window.iloc[-1])
    n = int(len(window) - 1)
    if start <= 0 or end <= 0 or n <= 0:
        return float("nan")
    total = end / start
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    return float(total ** (float(TRADING_DAYS_PER_YEAR) / float(n)) - 1.0)


def full_sample_realized_vol(close: pd.Series, *, min_bars: int = 5) -> float:
    """Annualized realized vol from all log returns in ``close`` (since listing)."""
    s = pd.to_numeric(close, errors="coerce").dropna()
    if len(s) < int(min_bars) + 1:
        return float("nan")
    log_ret = np.log(s / s.shift(1)).dropna()
    if len(log_ret) < int(min_bars):
        return float("nan")
    std = float(log_ret.std(ddof=0))
    if not np.isfinite(std):
        return float("nan")
    return std * float(np.sqrt(TRADING_DAYS_PER_YEAR))


def asof_metrics(
    close: pd.Series,
    as_of: str | pd.Timestamp,
    *,
    rv5_window: int = RV5_WINDOW,
    rv20_window: int = RV20_WINDOW,
    rv60_window: int = RV60_WINDOW,
) -> dict[str, float]:
    """Point-in-time 5/20/60d and since-inception annualized metrics through ``as_of``."""
    s = slice_close_through(close, as_of)
    empty = {
        "rv5": float("nan"),
        "rv20": float("nan"),
        "rv60": float("nan"),
        "rv_incep": float("nan"),
        "r_5a": float("nan"),
        "r_20a": float("nan"),
        "r_60a": float("nan"),
        "r_incep": float("nan"),
        "n_bars_incep": 0.0,
    }
    if s.empty:
        return empty

    def _last_rv(window: int) -> float:
        series = compute_realized_vol(s, window=int(window), ddof=0).dropna()
        return float(series.iloc[-1]) if not series.empty else float("nan")

    n_bars = float(max(len(s) - 1, 0))
    return {
        "rv5": _last_rv(rv5_window),
        "rv20": _last_rv(rv20_window),
        "rv60": _last_rv(rv60_window),
        "rv_incep": full_sample_realized_vol(s),
        "r_5a": trailing_realized_annual_return(s, bars=int(rv5_window)),
        "r_20a": trailing_realized_annual_return(s, bars=int(rv20_window)),
        "r_60a": trailing_realized_annual_return(s, bars=int(rv60_window)),
        "r_incep": trailing_realized_annual_return(s, bars=max(len(s) - 1, 1)),
        "n_bars_incep": n_bars,
    }

def compute_market_price_of_risk(
    equity_rows: list[Mapping[str, Any]],
    *,
    ret_key: str,
    vol_key: str,
    fallback: float = LAMBDA_FALLBACK,
    min_sigma: float = MIN_SIGMA,
) -> dict[str, float]:
    """Pool market price of risk: median(ret / vol) over equity rows."""
    vals: list[float] = []
    for row in equity_rows:
        r = float(row.get(ret_key, float("nan")))
        sig = float(row.get(vol_key, float("nan")))
        if not (np.isfinite(r) and np.isfinite(sig)):
            continue
        if abs(sig) < float(min_sigma):
            continue
        vals.append(r / sig)
    if not vals:
        lam = float(fallback)
        return {
            "lambda_mkt": lam,
            "lambda_mkt_p25": lam,
            "lambda_mkt_p75": lam,
            "n_lambda": 0,
        }
    arr = np.asarray(vals, dtype=float)
    return {
        "lambda_mkt": float(np.median(arr)),
        "lambda_mkt_p25": float(np.percentile(arr, 25)),
        "lambda_mkt_p75": float(np.percentile(arr, 75)),
        "n_lambda": int(arr.size),
    }


def required_annual_return(*, vol: float, lambda_mkt: float) -> float:
    """r_req = λ_mkt × vol (pool market price of risk)."""
    if not (np.isfinite(vol) and np.isfinite(lambda_mkt)):
        return float("nan")
    return float(lambda_mkt) * float(vol)


def _sleeve_for(code: str, defensive_codes: frozenset[str]) -> str:
    return "bond" if str(code).upper() in defensive_codes else "equity"


def _finite_median(vals: list[float]) -> float:
    return float(np.median(vals)) if vals else float("nan")


def _sharpe_like(ret: float, vol: float) -> float:
    if np.isfinite(ret) and np.isfinite(vol) and abs(vol) >= MIN_SIGMA:
        return ret / vol
    return float("nan")


def _gap(ret: float, req: float) -> float:
    if np.isfinite(ret) and np.isfinite(req):
        return ret - req
    return float("nan")


def _note_for(
    *,
    sleeve: str,
    rv5: float,
    rv20: float,
    rv60: float,
    vol_vs_bond60: float,
    gap_60: float,
    gap_20: float,
    gap_5: float,
) -> str:
    parts: list[str] = []
    if sleeve == "bond":
        parts.append("defensive_anchor_or_peer")
        if np.isfinite(rv20) and np.isfinite(rv60) and rv20 > rv60 * 1.25:
            parts.append("short_noisy_long_calmer")
    if np.isfinite(vol_vs_bond60) and vol_vs_bond60 >= 5.0:
        if np.isfinite(gap_60) and gap_60 < -0.05:
            parts.append("vol_expensive_gap60_neg")
    for tag, gap in (("r60", gap_60), ("r20", gap_20), ("r5", gap_5)):
        if np.isfinite(gap) and gap > 0.05:
            parts.append(f"{tag}_above_req")
        elif np.isfinite(gap) and gap < -0.05:
            parts.append(f"{tag}_below_req")
    return "|".join(parts) if parts else ""


def build_pool_board(
    frames: Mapping[str, pd.Series | pd.DataFrame],
    as_of: str | pd.Timestamp,
    *,
    bond_anchor: str = DEFAULT_BOND_ANCHOR,
    defensive_codes: frozenset[str] | set[str] | None = None,
    names: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build 5/20/60-horizon vol/return board for ``frames`` keyed by code."""
    def_codes = frozenset(
        str(c).upper() for c in (defensive_codes or DEFAULT_DEFENSIVE_CODES)
    )
    anchor = str(bond_anchor).upper()
    name_map = {str(k).upper(): str(v) for k, v in (names or {}).items()}

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

    metrics: dict[str, dict[str, float]] = {
        code: asof_metrics(ser, as_of) for code, ser in closes.items()
    }

    bond_present = anchor in metrics and not closes.get(anchor, pd.Series(dtype=float)).empty
    if bond_present:
        bond_m = metrics[anchor]
        r_bond_60a = float(bond_m["r_60a"])
        r_bond_20a = float(bond_m["r_20a"])
        r_bond_5a = float(bond_m["r_5a"])
        r_bond_incep = float(bond_m["r_incep"])
        sigma_bond60 = float(bond_m["rv60"])
        sigma_bond20 = float(bond_m["rv20"])
        sigma_bond5 = float(bond_m["rv5"])
        sigma_bond_incep = float(bond_m["rv_incep"])
    else:
        r_bond_60a = r_bond_20a = r_bond_5a = r_bond_incep = float("nan")
        sigma_bond60 = sigma_bond20 = sigma_bond5 = sigma_bond_incep = float("nan")

    equity_rows = [
        {"code": c, **metrics[c]}
        for c in metrics
        if _sleeve_for(c, def_codes) == "equity"
    ]
    por60 = compute_market_price_of_risk(equity_rows, ret_key="r_60a", vol_key="rv60")
    por20 = compute_market_price_of_risk(equity_rows, ret_key="r_20a", vol_key="rv20")
    por5 = compute_market_price_of_risk(equity_rows, ret_key="r_5a", vol_key="rv5")
    por_incep = compute_market_price_of_risk(
        equity_rows, ret_key="r_incep", vol_key="rv_incep"
    )
    lambda_60 = float(por60["lambda_mkt"])
    lambda_20 = float(por20["lambda_mkt"])
    lambda_5 = float(por5["lambda_mkt"])
    lambda_incep = float(por_incep["lambda_mkt"])

    def _eq_vals(key: str) -> list[float]:
        return [
            float(r[key])
            for r in equity_rows
            if np.isfinite(float(r.get(key, float("nan"))))
        ]

    rows: list[dict[str, Any]] = []
    for code, m in sorted(metrics.items()):
        sleeve = _sleeve_for(code, def_codes)
        rv5 = float(m["rv5"])
        rv20 = float(m["rv20"])
        rv60 = float(m["rv60"])
        rv_incep = float(m["rv_incep"])
        r_5a = float(m["r_5a"])
        r_20a = float(m["r_20a"])
        r_60a = float(m["r_60a"])
        r_incep = float(m["r_incep"])
        n_bars_incep = int(m.get("n_bars_incep", 0) or 0)
        if (
            np.isfinite(rv60)
            and np.isfinite(sigma_bond60)
            and abs(sigma_bond60) >= MIN_SIGMA
        ):
            vol_vs = rv60 / sigma_bond60
        else:
            vol_vs = float("nan")

        r_req_60 = required_annual_return(vol=rv60, lambda_mkt=lambda_60)
        r_req_20 = required_annual_return(vol=rv20, lambda_mkt=lambda_20)
        r_req_5 = required_annual_return(vol=rv5, lambda_mkt=lambda_5)
        r_req_incep = required_annual_return(vol=rv_incep, lambda_mkt=lambda_incep)
        gap_60 = _gap(r_60a, r_req_60)
        gap_20 = _gap(r_20a, r_req_20)
        gap_5 = _gap(r_5a, r_req_5)
        gap_incep = _gap(r_incep, r_req_incep)
        sharpe_60 = _sharpe_like(r_60a, rv60)
        sharpe_20 = _sharpe_like(r_20a, rv20)
        sharpe_5 = _sharpe_like(r_5a, rv5)
        sharpe_incep = _sharpe_like(r_incep, rv_incep)
        rows.append(
            {
                "code": code,
                "name": name_map.get(code, ""),
                "sleeve": sleeve,
                "rv5": rv5,
                "rv20": rv20,
                "rv60": rv60,
                "rv_incep": rv_incep,
                "vol_vs_bond60": vol_vs,
                "r_5a": r_5a,
                "r_20a": r_20a,
                "r_60a": r_60a,
                "r_incep": r_incep,
                "n_bars_incep": n_bars_incep,
                "sharpe_like_5": sharpe_5,
                "sharpe_like_20": sharpe_20,
                "sharpe_like_60": sharpe_60,
                "sharpe_like_incep": sharpe_incep,
                "r_req_5": r_req_5,
                "r_req_20": r_req_20,
                "r_req_60": r_req_60,
                "r_req_incep": r_req_incep,
                "gap_5": gap_5,
                "gap_20": gap_20,
                "gap_60": gap_60,
                "gap_incep": gap_incep,
                # Compat columns: since-inception annualized (not 60d).
                "r_1y": r_incep,
                "r_req": r_req_incep,
                "gap": gap_incep,
                "sharpe_like": sharpe_incep,
                "note": _note_for(
                    sleeve=sleeve,
                    rv5=rv5,
                    rv20=rv20,
                    rv60=rv60,
                    vol_vs_bond60=vol_vs,
                    gap_60=gap_60,
                    gap_20=gap_20,
                    gap_5=gap_5,
                ),
                "is_bond_anchor": code == anchor,
            }
        )

    board = pd.DataFrame(rows)
    if not board.empty:
        board = board.sort_values(
            by=["sleeve", "gap_60"],
            ascending=[True, True],
            kind="mergesort",
        ).reset_index(drop=True)

    meta: dict[str, Any] = {
        "as_of": str(_as_of_ts(as_of).date()),
        "bond_anchor": anchor,
        "bond_present": bool(bond_present),
        "r_bond_60a": r_bond_60a,
        "r_bond_20a": r_bond_20a,
        "r_bond_5a": r_bond_5a,
        "r_bond_incep": r_bond_incep,
        "r_bond_1y": r_bond_incep,
        "sigma_bond60": sigma_bond60,
        "sigma_bond20": sigma_bond20,
        "sigma_bond5": sigma_bond5,
        "sigma_bond_incep": sigma_bond_incep,
        "r_eq_med_60a": _finite_median(_eq_vals("r_60a")),
        "r_eq_med_20a": _finite_median(_eq_vals("r_20a")),
        "r_eq_med_5a": _finite_median(_eq_vals("r_5a")),
        "r_eq_med_incep": _finite_median(_eq_vals("r_incep")),
        "r_eq_med": _finite_median(_eq_vals("r_incep")),
        "rv60_eq_med": _finite_median(_eq_vals("rv60")),
        "rv20_eq_med": _finite_median(_eq_vals("rv20")),
        "rv5_eq_med": _finite_median(_eq_vals("rv5")),
        "rv_incep_eq_med": _finite_median(_eq_vals("rv_incep")),
        "lambda_mkt_60": lambda_60,
        "lambda_mkt_20": lambda_20,
        "lambda_mkt_5": lambda_5,
        "lambda_mkt_incep": lambda_incep,
        "lambda_mkt": lambda_incep,
        "lambda_price_of_risk": lambda_incep,
        "lambda_mkt_60_p25": float(por60["lambda_mkt_p25"]),
        "lambda_mkt_60_p75": float(por60["lambda_mkt_p75"]),
        "lambda_mkt_20_p25": float(por20["lambda_mkt_p25"]),
        "lambda_mkt_20_p75": float(por20["lambda_mkt_p75"]),
        "lambda_mkt_5_p25": float(por5["lambda_mkt_p25"]),
        "lambda_mkt_5_p75": float(por5["lambda_mkt_p75"]),
        "lambda_mkt_incep_p25": float(por_incep["lambda_mkt_p25"]),
        "lambda_mkt_incep_p75": float(por_incep["lambda_mkt_p75"]),
        "lambda_mkt_p25": float(por_incep["lambda_mkt_p25"]),
        "lambda_mkt_p75": float(por_incep["lambda_mkt_p75"]),
        "n_lambda_60": int(por60["n_lambda"]),
        "n_lambda_20": int(por20["n_lambda"]),
        "n_lambda_5": int(por5["n_lambda"]),
        "n_lambda_incep": int(por_incep["n_lambda"]),
        "n_lambda": int(por_incep["n_lambda"]),
        "n_codes": int(len(board)),
        "n_equity": int((board["sleeve"] == "equity").sum()) if not board.empty else 0,
        "formula_r_req_60": "lambda_mkt_60 * rv60",
        "formula_r_req_20": "lambda_mkt_20 * rv20",
        "formula_r_req_5": "lambda_mkt_5 * rv5",
        "formula_r_req_incep": "lambda_mkt_incep * rv_incep",
        "formula_r_req": "lambda_mkt_incep * rv_incep",
        "formula_lambda_mkt_60": "median(r_60a / rv60) over equity pool",
        "formula_lambda_mkt_20": "median(r_20a / rv20) over equity pool",
        "formula_lambda_mkt_5": "median(r_5a / rv5) over equity pool",
        "formula_lambda_mkt_incep": "median(r_incep / rv_incep) over equity pool",
        "formula_gap_60": "r_60a - r_req_60",
        "formula_gap_20": "r_20a - r_req_20",
        "formula_gap_5": "r_5a - r_req_5",
        "formula_gap_incep": "r_incep - r_req_incep",
        "formula_gap": "r_incep - r_req_incep",
    }
    return board, meta


def render_readme(meta: dict[str, Any], board: pd.DataFrame) -> str:
    """Markdown summary for the research board output."""
    lines = [
        "# ETF 市场风险定价诊断板（46 票 · 5/20/60 + 成立以来）",
        "",
        f"- as_of: `{meta.get('as_of')}`",
        f"- n_codes: {meta.get('n_codes')} (equity={meta.get('n_equity')})",
        f"- bond_anchor (secondary only): `{meta.get('bond_anchor')}` "
        f"(present={meta.get('bond_present')})",
        "",
        "## Market price of risk (primary)",
        "",
        f"- `λ_incep` = **{_num(meta.get('lambda_mkt_incep'))}** "
        f"(p25/p75={_num(meta.get('lambda_mkt_incep_p25'))}/"
        f"{_num(meta.get('lambda_mkt_incep_p75'))}, n={meta.get('n_lambda_incep')})",
        f"- `λ60` = **{_num(meta.get('lambda_mkt_60'))}** "
        f"(p25/p75={_num(meta.get('lambda_mkt_60_p25'))}/"
        f"{_num(meta.get('lambda_mkt_60_p75'))}, n={meta.get('n_lambda_60')})",
        f"- `λ20` = **{_num(meta.get('lambda_mkt_20'))}** "
        f"(p25/p75={_num(meta.get('lambda_mkt_20_p25'))}/"
        f"{_num(meta.get('lambda_mkt_20_p75'))}, n={meta.get('n_lambda_20')})",
        f"- `λ5`  = **{_num(meta.get('lambda_mkt_5'))}** "
        f"(p25/p75={_num(meta.get('lambda_mkt_5_p25'))}/"
        f"{_num(meta.get('lambda_mkt_5_p75'))}, n={meta.get('n_lambda_5')})",
        f"- equity median r_incep/r_60a/r_20a/r_5a = "
        f"{_pct(meta.get('r_eq_med_incep'))} / {_pct(meta.get('r_eq_med_60a'))} / "
        f"{_pct(meta.get('r_eq_med_20a'))} / {_pct(meta.get('r_eq_med_5a'))}",
        "",
        "## Bond reference (secondary)",
        "",
        f"- r_bond incep/60/20/5 = {_pct(meta.get('r_bond_incep'))} / "
        f"{_pct(meta.get('r_bond_60a'))} / {_pct(meta.get('r_bond_20a'))} / "
        f"{_pct(meta.get('r_bond_5a'))}",
        f"- σ_bond incep/60/20/5 = {_pct(meta.get('sigma_bond_incep'))} / "
        f"{_pct(meta.get('sigma_bond60'))} / {_pct(meta.get('sigma_bond20'))} / "
        f"{_pct(meta.get('sigma_bond5'))}",
        "",
        "## Formulas",
        "",
        "- `r_*a` = trailing N-day return annualized to 252d",
        "- `r_incep` / `rv_incep` = since listing (all bars through as_of), annualized",
        f"- incep: `{meta.get('formula_lambda_mkt_incep')}` → `{meta.get('formula_r_req_incep')}` → `{meta.get('formula_gap_incep')}`",
        f"- 60d: `{meta.get('formula_lambda_mkt_60')}` → `{meta.get('formula_r_req_60')}` → `{meta.get('formula_gap_60')}`",
        f"- 20d: `{meta.get('formula_lambda_mkt_20')}` → `{meta.get('formula_r_req_20')}` → `{meta.get('formula_gap_20')}`",
        f"- 5d:  `{meta.get('formula_lambda_mkt_5')}` → `{meta.get('formula_r_req_5')}` → `{meta.get('formula_gap_5')}`",
        "",
        "## How to read",
        "",
        "- `r_1y`/`r_req`/`gap`/`sharpe_like` = **成立以来**年化套（非 60 日）",
        "- 另看 `gap_60` / `gap_20` / `gap_5`：短中窗相对同窗市场定价",
        "- `vol_vs_bond60` 只是相对国债波动倍数，**不进** `r_req`",
        "- 本板是**诊断尺**，不是买卖信号",
        "",
    ]
    if board is not None and not board.empty:
        eq = board[board["sleeve"] == "equity"]
        worst = eq.nsmallest(5, "gap", keep="all")
        if not worst.empty:
            lines.extend(["## Equity: most negative inception gap (top 5)", ""])
            for _, r in worst.iterrows():
                lines.append(
                    f"- `{r['code']}` gap={_pct(r['gap'])} "
                    f"(r_1y={_pct(r['r_1y'])}, bars={int(r.get('n_bars_incep', 0))}) "
                    f"| gap60={_pct(r['gap_60'])} gap20={_pct(r['gap_20'])} "
                    f"gap5={_pct(r['gap_5'])}"
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
    return f"{v:.3f}"


__all__ = [
    "DEFAULT_BOND_ANCHOR",
    "DEFAULT_BOND_AUX",
    "DEFAULT_DEFENSIVE_CODES",
    "DEFAULT_QLIB_INSTRUMENTS",
    "LAMBDA_FALLBACK",
    "RV5_WINDOW",
    "asof_metrics",
    "build_pool_board",
    "compute_market_price_of_risk",
    "full_sample_realized_vol",
    "load_fund_name_map",
    "load_instrument_codes",
    "render_readme",
    "required_annual_return",
    "slice_close_through",
    "trailing_realized_annual_return",
]
