"""Portfolio snapshot: live holdings or pipeline weights."""
from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from decision_pack.src.config import DecisionPackConfig, load_decision_pack_config
from pipeline.strategy_layer_audit import load_qlib_close
from runtime_paths import QLIB_PROVIDER_URI
from workspace.scripts.generate_daily_mu_position_report import (  # noqa: E402
    first_existing_column,
    load_fund_name_map,
    load_portfolio_weights,
    normalize_code,
    read_csv,
)

PriceSemantics = Literal["cost", "market"]


@dataclass(frozen=True)
class HoldingRow:
    code: str
    name: str
    market_value: float
    weight: float
    weight_total: float
    source: str
    shares: float | None = None
    cost_price: float | None = None
    close_price: float | None = None


@dataclass(frozen=True)
class PortfolioSnapshot:
    as_of: str
    source: str
    holdings: tuple[HoldingRow, ...]
    total_assets: float
    cash_mv: float
    stock_mv: float
    equity_exposure: float

    def to_frame(self) -> pd.DataFrame:
        columns = list(HoldingRow.__dataclass_fields__.keys())
        rows = [asdict(row) for row in self.holdings]
        return pd.DataFrame(rows, columns=columns)

    def top_holdings(self, n: int = 10) -> tuple[HoldingRow, ...]:
        ranked = sorted(self.holdings, key=lambda row: row.market_value, reverse=True)
        return tuple(ranked[:n])


def _coerce_positive(value: object, default: float = 0.0) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number) or not math.isfinite(float(number)):
        return default
    return max(0.0, float(number))


def _resolve_name(code: str, name_map: dict[str, str]) -> str:
    return name_map.get(code, code)


def live_holdings_price_semantics(config: DecisionPackConfig | None = None) -> PriceSemantics:
    cfg = config or load_decision_pack_config()
    raw = cfg.raw.get("live_holdings") or {}
    value = str(raw.get("price_column_semantics", "cost")).strip().lower()
    if value not in {"cost", "market"}:
        raise ValueError(f"live_holdings.price_column_semantics must be 'cost' or 'market', got {value!r}")
    return value  # type: ignore[return-value]


def fetch_close_on_date(
    codes: list[str],
    as_of: str,
    *,
    provider_uri: str | Path | None = None,
) -> dict[str, float]:
    if not codes:
        return {}
    uri = str(provider_uri or QLIB_PROVIDER_URI)
    end = pd.Timestamp(as_of)
    start = (end - pd.tseries.offsets.BDay(10)).strftime("%Y-%m-%d")
    close_panel = load_qlib_close(uri, codes, start, as_of)
    out: dict[str, float] = {}
    for code in codes:
        norm = normalize_code(code)
        if norm not in close_panel.columns:
            continue
        series = pd.to_numeric(close_panel[norm], errors="coerce").dropna()
        if series.empty:
            continue
        available = series.loc[series.index <= end]
        if available.empty:
            continue
        value = float(available.iloc[-1])
        if math.isfinite(value) and value > 0:
            out[norm] = value
    return out


def _resolve_live_row_values(
    raw: pd.Series,
    *,
    code: str,
    shares_col: str | None,
    mv_col: str | None,
    market_price_col: str | None,
    cost_col: str | None,
    generic_price_col: str | None,
    price_semantics: PriceSemantics,
    close_prices: dict[str, float],
) -> tuple[float, float | None, float | None, float | None]:
    """Return market_value, shares, cost_price, close_price."""
    shares = _coerce_positive(raw[shares_col]) if shares_col is not None else None

    if mv_col is not None and not pd.isna(raw.get(mv_col)):
        return _coerce_positive(raw[mv_col]), shares, None, None

    cost_price = None
    if cost_col is not None and not pd.isna(raw.get(cost_col)):
        cost_price = _coerce_positive(raw[cost_col])
    elif generic_price_col is not None and price_semantics == "cost" and not pd.isna(raw.get(generic_price_col)):
        cost_price = _coerce_positive(raw[generic_price_col])

    close_price = close_prices.get(code)
    if market_price_col is not None and not pd.isna(raw.get(market_price_col)):
        close_price = _coerce_positive(raw[market_price_col])

    if shares is not None and shares > 0:
        if close_price is not None and close_price > 0:
            return shares * close_price, shares, cost_price, close_price
        if generic_price_col is not None and price_semantics == "market" and not pd.isna(raw.get(generic_price_col)):
            market_price = _coerce_positive(raw[generic_price_col])
            return shares * market_price, shares, cost_price, market_price

    if cost_price is not None and shares is not None and shares > 0:
        if close_price is None:
            raise ValueError(
                f"Missing qlib close for {code} on valuation date; "
                f"cannot compute market_value from shares × cost ({cost_price})."
            )
        return shares * close_price, shares, cost_price, close_price

    raise ValueError(
        f"Live holdings row for {code} needs market_value, shares+close/market_price, "
        f"or shares+cost with qlib close."
    )


def _read_live_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return read_csv(path)


def load_live_holdings_csv(
    path: Path,
    *,
    as_of: str,
    name_map: dict[str, str] | None = None,
    total_assets_override: float | None = None,
    price_semantics: PriceSemantics | None = None,
    config: DecisionPackConfig | None = None,
    provider_uri: str | Path | None = None,
) -> PortfolioSnapshot:
    cfg = config or load_decision_pack_config()
    semantics = price_semantics or live_holdings_price_semantics(cfg)

    frame = _read_live_csv(path)
    code_col = first_existing_column(frame, ["code", "instrument", "基金代码", "symbol"])
    if code_col is None:
        raise ValueError(f"No code column in live holdings CSV: {path}")

    mv_col = first_existing_column(frame, ["market_value", "市值", "market_val"])
    shares_col = first_existing_column(frame, ["shares", "份额", "volume"])
    market_price_col = first_existing_column(
        frame, ["market_price", "close_price", "last_price", "close", "收盘价", "现价", "最新价"]
    )
    cost_col = first_existing_column(frame, ["cost", "cost_price", "avg_cost", "成本", "持仓成本", "成本价"])
    generic_price_col = first_existing_column(frame, ["price"])
    name_col = first_existing_column(frame, ["name", "基金简称", "基金名称"])

    codes: list[str] = []
    for _, raw in frame.iterrows():
        code = normalize_code(raw[code_col])
        if code and code != "CASH":
            codes.append(code)

    need_qlib_close = (
        mv_col is None
        and market_price_col is None
        and shares_col is not None
        and (semantics == "cost" or cost_col is not None)
    )
    close_prices = fetch_close_on_date(codes, as_of, provider_uri=provider_uri) if need_qlib_close else {}

    if need_qlib_close:
        missing = sorted(set(codes) - set(close_prices))
        if missing:
            raise ValueError(
                f"Could not load qlib close for {len(missing)} live holdings on {as_of}: {missing[:8]}"
                + (" ..." if len(missing) > 8 else "")
            )

    names = name_map or {}
    rows: list[dict[str, object]] = []
    for _, raw in frame.iterrows():
        code = normalize_code(raw[code_col])
        if not code:
            continue
        name = ""
        if name_col is not None and not pd.isna(raw.get(name_col)):
            name = str(raw[name_col]).strip()
        if not name:
            name = _resolve_name(code, names)

        market_value, shares, cost_price, close_price = _resolve_live_row_values(
            raw,
            code=code,
            shares_col=shares_col,
            mv_col=mv_col,
            market_price_col=market_price_col,
            cost_col=cost_col,
            generic_price_col=generic_price_col,
            price_semantics=semantics,
            close_prices=close_prices,
        )
        rows.append(
            {
                "code": code,
                "name": name,
                "market_value": market_value,
                "shares": shares,
                "cost_price": cost_price,
                "close_price": close_price,
            }
        )

    if not rows:
        total_assets = total_assets_override if total_assets_override is not None else 0.0
        return PortfolioSnapshot(
            as_of=as_of,
            source="live",
            holdings=(),
            total_assets=total_assets,
            cash_mv=0.0,
            stock_mv=0.0,
            equity_exposure=0.0,
        )

    holdings_df = pd.DataFrame(rows)
    cash_rows = holdings_df.loc[holdings_df["code"].eq("CASH")]
    cash_mv = float(cash_rows["market_value"].sum()) if not cash_rows.empty else 0.0
    stock_df = holdings_df.loc[~holdings_df["code"].eq("CASH")].copy()
    stock_mv = float(stock_df["market_value"].sum())
    total_assets = total_assets_override if total_assets_override is not None else stock_mv + cash_mv
    if total_assets <= 0:
        raise ValueError(f"Total assets must be positive for {path}")

    equity_exposure = stock_mv / total_assets if total_assets > 0 else 0.0
    holding_rows: list[HoldingRow] = []
    for _, row in stock_df.iterrows():
        mv = float(row["market_value"])
        weight_total = mv / total_assets
        weight = mv / stock_mv if stock_mv > 0 else 0.0
        holding_rows.append(
            HoldingRow(
                code=str(row["code"]),
                name=str(row["name"]),
                market_value=mv,
                weight=weight,
                weight_total=weight_total,
                source="live",
                shares=None if pd.isna(row.get("shares")) else float(row["shares"]),
                cost_price=None if pd.isna(row.get("cost_price")) else float(row["cost_price"]),
                close_price=None if pd.isna(row.get("close_price")) else float(row["close_price"]),
            )
        )

    return PortfolioSnapshot(
        as_of=as_of,
        source="live",
        holdings=tuple(sorted(holding_rows, key=lambda item: item.market_value, reverse=True)),
        total_assets=total_assets,
        cash_mv=cash_mv,
        stock_mv=stock_mv,
        equity_exposure=equity_exposure,
    )


def load_pipeline_portfolio(
    portfolio_weights_csv: Path,
    *,
    as_of: str,
    fund_list_csv: Path | None = None,
    notional_total: float = 1.0,
) -> PortfolioSnapshot:
    frame, cash_weight, _ = load_portfolio_weights(portfolio_weights_csv, as_of)
    name_map = load_fund_name_map(fund_list_csv) if fund_list_csv else {}

    stock_df = frame.loc[frame["code_norm"].ne("CASH") & frame["code_norm"].ne("")].copy()
    if stock_df.empty:
        raise ValueError(f"No stock weights for {as_of} in {portfolio_weights_csv}")

    stock_weight_sum = float(stock_df["actual_weight"].sum())
    cash_mv = max(0.0, float(cash_weight)) * notional_total
    stock_mv = max(0.0, stock_weight_sum) * notional_total
    total_assets = stock_mv + cash_mv
    if total_assets <= 0:
        total_assets = notional_total
        stock_mv = max(stock_mv, stock_weight_sum * notional_total)

    equity_exposure = stock_mv / total_assets if total_assets > 0 else 0.0
    holding_rows: list[HoldingRow] = []
    for _, row in stock_df.iterrows():
        code = str(row["code_norm"])
        weight_total = float(row["actual_weight"])
        mv = weight_total * notional_total
        weight = mv / stock_mv if stock_mv > 0 else 0.0
        holding_rows.append(
            HoldingRow(
                code=code,
                name=_resolve_name(code, name_map),
                market_value=mv,
                weight=weight,
                weight_total=weight_total if notional_total == 1.0 else mv / total_assets,
                source="pipeline",
            )
        )

    return PortfolioSnapshot(
        as_of=as_of,
        source="pipeline",
        holdings=tuple(sorted(holding_rows, key=lambda item: item.market_value, reverse=True)),
        total_assets=total_assets,
        cash_mv=cash_mv,
        stock_mv=stock_mv,
        equity_exposure=equity_exposure,
    )


def load_portfolio_snapshot(
    *,
    as_of: str,
    live_holdings_path: Path | None = None,
    pipeline_weights_csv: Path | None = None,
    fund_list_csv: Path | None = None,
    config: DecisionPackConfig | None = None,
    provider_uri: str | Path | None = None,
) -> PortfolioSnapshot:
    if live_holdings_path is not None:
        name_map = load_fund_name_map(fund_list_csv) if fund_list_csv else {}
        return load_live_holdings_csv(
            live_holdings_path,
            as_of=as_of,
            name_map=name_map,
            config=config,
            provider_uri=provider_uri,
        )
    if pipeline_weights_csv is None:
        raise ValueError("Either live_holdings_path or pipeline_weights_csv is required")
    return load_pipeline_portfolio(
        pipeline_weights_csv,
        as_of=as_of,
        fund_list_csv=fund_list_csv,
    )


__all__ = [
    "HoldingRow",
    "PortfolioSnapshot",
    "fetch_close_on_date",
    "live_holdings_price_semantics",
    "load_live_holdings_csv",
    "load_pipeline_portfolio",
    "load_portfolio_snapshot",
]
