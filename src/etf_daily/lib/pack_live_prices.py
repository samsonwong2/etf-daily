"""Pack live / spot close prices for cluster universe injection."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from etf_daily.lib.config import DecisionPackConfig, load_decision_pack_config
from etf_daily.pipeline.strategy_layer_audit import load_qlib_close, normalize_code
from etf_daily.paths import QLIB_PROVIDER_URI

UNIVERSE_LIVE_CLOSE_CSV = "universe_live_close.csv"

LiveCloseSource = Literal["portfolio_live", "spot", "qlib"]

_UNIVERSE_LIVE_COLUMNS = ("code", "name", "close_price", "source", "bar_date", "as_of")


def _normalize_code(value: object) -> str:
    return normalize_code(value) if value is not None else ""


def _normalize_spot_code(code: object) -> str:
    text = str(code).strip()
    if text.startswith(("SH", "SZ")):
        return text.upper()
    if text[0] == "5":
        return f"SH{text}"
    return text.upper()


def load_portfolio_live_closes(pack_dir: Path) -> dict[str, float]:
    """Read ``close_price`` from ``portfolio_snapshot.csv``."""
    path = Path(pack_dir) / "portfolio_snapshot.csv"
    if not path.exists():
        return {}

    frame = pd.read_csv(path)
    if "code" not in frame.columns:
        return {}

    out: dict[str, float] = {}
    for _, row in frame.iterrows():
        code = _normalize_code(row["code"])
        if not code or code == "CASH":
            continue

        close = pd.to_numeric(row.get("close_price"), errors="coerce")
        if pd.isna(close):
            shares = pd.to_numeric(row.get("shares"), errors="coerce")
            market_value = pd.to_numeric(row.get("market_value"), errors="coerce")
            if pd.notna(shares) and pd.notna(market_value) and float(shares) > 0:
                close = float(market_value) / float(shares)

        if pd.isna(close):
            continue
        price = float(close)
        if math.isfinite(price) and price > 0:
            out[code] = price
    return out


def load_live_close_from_pack(pack_dir: Path) -> dict[str, float]:
    """Load pack live closes; prefer ``universe_live_close.csv`` when present."""
    universe_path = Path(pack_dir) / UNIVERSE_LIVE_CLOSE_CSV
    if universe_path.exists():
        frame = pd.read_csv(universe_path)
        if "code" in frame.columns and "close_price" in frame.columns:
            out: dict[str, float] = {}
            for _, row in frame.iterrows():
                code = _normalize_code(row["code"])
                close = pd.to_numeric(row.get("close_price"), errors="coerce")
                if code and pd.notna(close) and float(close) > 0:
                    out[code] = float(close)
            if out:
                return out
    return load_portfolio_live_closes(pack_dir)


def universe_live_include_spot(config: DecisionPackConfig | None = None) -> bool:
    cfg = config or load_decision_pack_config()
    raw = cfg.raw.get("universe_live_close") or {}
    return bool(raw.get("include_spot", True))


def should_fetch_spot(as_of: str, *, config: DecisionPackConfig | None = None) -> bool:
    if not universe_live_include_spot(config):
        return False
    return pd.Timestamp(as_of).normalize() == pd.Timestamp.today().normalize()


def fetch_qlib_closes_with_dates(
    codes: list[str],
    as_of: str,
    *,
    provider_uri: str | Path | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    if not codes:
        return {}, {}

    uri = str(provider_uri or QLIB_PROVIDER_URI)
    end = pd.Timestamp(as_of)
    start = (end - pd.tseries.offsets.BDay(10)).strftime("%Y-%m-%d")
    panel = load_qlib_close(uri, codes, start, as_of)

    prices: dict[str, float] = {}
    bar_dates: dict[str, str] = {}
    for code in codes:
        norm = _normalize_code(code)
        if norm not in panel.columns:
            continue
        series = pd.to_numeric(panel[norm], errors="coerce").dropna()
        if series.empty:
            continue
        available = series.loc[series.index <= end]
        if available.empty:
            continue
        value = float(available.iloc[-1])
        if math.isfinite(value) and value > 0:
            prices[norm] = value
            bar_dates[norm] = pd.Timestamp(available.index[-1]).strftime("%Y-%m-%d")
    return prices, bar_dates


def fetch_etf_spot_closes(
    codes: set[str] | frozenset[str] | None = None,
) -> dict[str, float]:
    """Fetch AkShare Sina ETF spot ``最新价``; filter to ``codes`` when provided."""
    try:
        import akshare as ak
    except ImportError:
        print("[WARN] universe_live_close: akshare not installed; skipping spot fetch")
        return {}

    try:
        spot_df = ak.fund_etf_category_sina(symbol="ETF基金")
    except Exception as exc:  # pragma: no cover - network/provider dependent
        print(f"[WARN] universe_live_close: AkShare spot fetch failed: {exc}")
        return {}

    required = ["代码", "最新价"]
    missing = [col for col in required if col not in spot_df.columns]
    if missing:
        print(f"[WARN] universe_live_close: spot data missing columns {missing}")
        return {}

    wanted = {_normalize_code(code) for code in (codes or ()) if _normalize_code(code)}
    out: dict[str, float] = {}
    for _, row in spot_df.iterrows():
        code = _normalize_spot_code(row["代码"])
        if wanted and code not in wanted:
            continue
        price = pd.to_numeric(row["最新价"], errors="coerce")
        if pd.isna(price):
            continue
        value = float(price)
        if math.isfinite(value) and value > 0:
            out[code] = value
    return out


def build_universe_live_close_frame(
    *,
    as_of: str,
    universe_codes: list[str],
    names: dict[str, str] | None = None,
    portfolio_closes: dict[str, float] | None = None,
    provider_uri: str | Path | None = None,
    config: DecisionPackConfig | None = None,
    spot_closes: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build full-universe live close table for pack output.

    Priority per code: spot (when available) > portfolio_live > qlib.
    Holdings use AkShare spot when present; portfolio snapshot is fallback only.
    """
    codes = [_normalize_code(code) for code in universe_codes]
    codes = [code for code in codes if code and code != "CASH"]
    if not codes:
        return pd.DataFrame(columns=_UNIVERSE_LIVE_COLUMNS)

    name_map = names or {}
    portfolio = portfolio_closes or {}
    qlib_prices, qlib_dates = fetch_qlib_closes_with_dates(
        codes,
        as_of,
        provider_uri=provider_uri,
    )

    spot: dict[str, float] = {}
    if spot_closes is not None:
        spot = { _normalize_code(k): float(v) for k, v in spot_closes.items() }
    elif should_fetch_spot(as_of, config=config):
        print(f"[INFO] universe_live_close: fetching AkShare spot for as_of={as_of}")
        spot = fetch_etf_spot_closes(frozenset(codes))

    rows: list[dict[str, object]] = []
    for code in codes:
        price: float | None = None
        source: LiveCloseSource | None = None
        bar_date = qlib_dates.get(code)

        if code in spot:
            price = float(spot[code])
            source = "spot"
            bar_date = as_of
        elif code in portfolio:
            price = float(portfolio[code])
            source = "portfolio_live"
            bar_date = as_of
        elif code in qlib_prices:
            price = float(qlib_prices[code])
            source = "qlib"

        if price is None or source is None:
            continue

        rows.append(
            {
                "code": code,
                "name": name_map.get(code, code),
                "close_price": price,
                "source": source,
                "bar_date": bar_date,
                "as_of": as_of,
            }
        )

    return pd.DataFrame(rows, columns=_UNIVERSE_LIVE_COLUMNS)


def _last_close_before(
    panel: pd.DataFrame,
    code: str,
    end: pd.Timestamp,
) -> tuple[pd.Timestamp | None, float | None]:
    series = pd.to_numeric(panel[code], errors="coerce").dropna()
    if series.empty:
        return None, None
    available = series.loc[series.index <= end]
    if available.empty:
        return None, None
    last_dt = pd.Timestamp(available.index[-1]).normalize()
    return last_dt, float(available.iloc[-1])


def _same_close_price(a: float, b: float, *, rtol: float = 1e-6, atol: float = 1e-4) -> bool:
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


@dataclass(frozen=True)
class LiveCloseInjectionStats:
    fetched: int
    injected: int
    skipped_same: int


def apply_pack_live_closes(
    panel: pd.DataFrame,
    as_of: str,
    live_closes: dict[str, float],
) -> tuple[pd.DataFrame, LiveCloseInjectionStats]:
    """Append or overwrite ``as_of`` row with pack live closes for matching columns.

    When ``as_of`` is after the last qlib bar and live price equals that bar,
    skip injection to avoid a spurious 0% return day (distorts short-horizon vol).
    """
    if panel.empty or not live_closes:
        return panel, LiveCloseInjectionStats(fetched=0, injected=0, skipped_same=0)

    end = pd.Timestamp(as_of).normalize()
    applicable = {
        code: price
        for code, price in live_closes.items()
        if code in panel.columns
    }
    if not applicable:
        return panel, LiveCloseInjectionStats(fetched=0, injected=0, skipped_same=0)

    out = panel.copy()
    updates: dict[str, float] = {}
    for code, price in applicable.items():
        last_dt, last_px = _last_close_before(out, code, end)
        if last_dt is None:
            updates[code] = price
            continue
        if last_dt >= end:
            if last_px is None or not _same_close_price(last_px, price):
                updates[code] = price
            continue
        if last_px is not None and _same_close_price(last_px, price):
            continue
        updates[code] = price

    if not updates:
        return panel, LiveCloseInjectionStats(
            fetched=len(applicable),
            injected=0,
            skipped_same=len(applicable),
        )

    if end not in out.index:
        out = out.reindex(out.index.union([end])).sort_index()

    for code, price in updates.items():
        out.loc[end, code] = price
    return out, LiveCloseInjectionStats(
        fetched=len(applicable),
        injected=len(updates),
        skipped_same=len(applicable) - len(updates),
    )


__all__ = [
    "UNIVERSE_LIVE_CLOSE_CSV",
    "LiveCloseInjectionStats",
    "apply_pack_live_closes",
    "build_universe_live_close_frame",
    "fetch_etf_spot_closes",
    "fetch_qlib_closes_with_dates",
    "load_live_close_from_pack",
    "load_portfolio_live_closes",
    "should_fetch_spot",
    "universe_live_include_spot",
]
