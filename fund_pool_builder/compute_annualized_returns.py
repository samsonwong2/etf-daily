"""Compute annualized mean daily return for every ticker in cluster_mapping_selected.txt.

Usage
-----
python compute_annualized_returns.py --start-date 2024-01-01 --end-date 2024-12-31
python compute_annualized_returns.py --start-date 2024-01-01 --end-date 2024-12-31 \
    --cluster-mapping /path/to/cluster_mapping_selected.txt \
    --output-dir ~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp

Output
------
CSV saved to <output_dir>/annualized_returns_<start>_<end>.csv
Columns: instrument, 基金简称, annualized_return, trading_days, mean_daily_return
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

PROVIDER_URI = "~/etf-daily-output/Documents/stock/data/qlib_data/all_fund_data"
DEFAULT_CLUSTER_MAPPING = (
    "~/etf-daily-output/Documents/stock/data/qlib_data/all_fund_data/instruments/cluster_mapping_selected.txt"
)
DEFAULT_FUND_LIST_CSV = "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/fund_list.csv"
DEFAULT_OUTPUT_DIR = "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp"
TRADING_DAYS_PER_YEAR = 252


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute annualized mean daily return for all tickers in cluster_mapping_selected.txt"
    )
    parser.add_argument("--start-date", required=True, help="Start date, e.g. 2024-01-01")
    parser.add_argument("--end-date", required=True, help="End date, e.g. 2024-12-31")
    parser.add_argument(
        "--cluster-mapping",
        default=DEFAULT_CLUSTER_MAPPING,
        help=f"Path to cluster_mapping_selected.txt (default: {DEFAULT_CLUSTER_MAPPING})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for CSV (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--fund-list-csv",
        default=DEFAULT_FUND_LIST_CSV,
        help=f"CSV with fund code and short-name columns (default: {DEFAULT_FUND_LIST_CSV})",
    )
    return parser.parse_args()


def read_tickers(mapping_path: str) -> list[str]:
    """Read instrument codes from a qlib instruments file (tab-separated, first column is ticker)."""
    tickers: list[str] = []
    with open(mapping_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ticker = line.split("\t")[0].strip()
            if ticker:
                tickers.append(ticker)
    return tickers


def normalize_code(code: object) -> str:
    return str(code).strip().upper()


def code_aliases(code: object) -> set[str]:
    norm = normalize_code(code)
    aliases = {norm}
    if norm.startswith(("SH", "SZ")) and len(norm) > 2:
        aliases.add(norm[2:])
    return {alias for alias in aliases if alias and alias.lower() != "nan"}


def load_fund_short_name_map(fund_list_csv: str) -> dict[str, str]:
    """Load fund short names keyed by normalized fund-code aliases."""
    if not os.path.isfile(fund_list_csv):
        print(f"WARNING: fund_list_csv not found, 基金简称 will be blank: {fund_list_csv}")
        return {}
    frame = pd.read_csv(fund_list_csv, dtype=str).fillna("")
    possible_code_names = ["基金代码", "fund_code", "代码", "code", "证券代码", "代码ID"]
    possible_name_names = ["基金简称", "简称", "名称", "name", "fund_name", "基金名称"]
    code_col = next((column for column in possible_code_names if column in frame.columns), None)
    name_col = next((column for column in possible_name_names if column in frame.columns), None)
    if code_col is None or name_col is None:
        raise ValueError(
            f"Could not find fund code/name columns in {fund_list_csv}; columns={list(frame.columns)}"
        )

    name_map: dict[str, str] = {}
    for _, row in frame[[code_col, name_col]].iterrows():
        name = str(row[name_col]).strip()
        for alias in code_aliases(row[code_col]):
            if alias and name and alias not in name_map:
                name_map[alias] = name
    return name_map


def attach_fund_short_names(result: pd.DataFrame, fund_name_map: dict[str, str]) -> pd.DataFrame:
    """Insert 基金简称 after instrument by matching normalized fund codes."""
    output = result.copy()
    names = output["instrument"].map(lambda code: fund_name_map.get(normalize_code(code), ""))
    if "基金简称" in output.columns:
        output["基金简称"] = names
    else:
        output.insert(1, "基金简称", names)
    return output


def load_close(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load adjusted close prices from qlib and return a date × instrument DataFrame."""
    try:
        import qlib
        from qlib.constant import REG_CN
        from qlib.data import D
    except ImportError as exc:
        sys.exit(f"ERROR: qlib is not available — {exc}")

    print(f"Initializing qlib: {PROVIDER_URI}")
    qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)

    print(f"Fetching $close for {len(tickers)} tickers [{start_date} → {end_date}] …")
    raw = D.features(
        tickers,
        fields=["$close"],
        start_time=start_date,
        end_time=end_date,
    )
    if raw.empty:
        sys.exit("ERROR: qlib returned no data for the given instruments/date range.")

    close: pd.DataFrame = raw["$close"].unstack(level="instrument")
    close.index.name = "date"
    close = close.sort_index()
    print(f"Loaded close price panel: {close.shape[0]} dates × {close.shape[1]} instruments")
    return close


def compute_annualized_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with annualized mean daily return per instrument."""
    # Percent change from previous trading day
    daily_ret = close.pct_change(fill_method=None)

    records = []
    for ticker in daily_ret.columns:
        series = daily_ret[ticker].dropna()
        n = len(series)
        if n == 0:
            records.append(
                {
                    "instrument": ticker,
                    "annualized_return": float("nan"),
                    "trading_days": 0,
                    "mean_daily_return": float("nan"),
                }
            )
        else:
            mean_daily = series.mean()
            annualized = mean_daily * TRADING_DAYS_PER_YEAR
            records.append(
                {
                    "instrument": ticker,
                    "annualized_return": annualized,
                    "trading_days": n,
                    "mean_daily_return": mean_daily,
                }
            )

    result = pd.DataFrame(records).sort_values("annualized_return", ascending=False).reset_index(drop=True)
    return result


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.cluster_mapping):
        sys.exit(f"ERROR: cluster_mapping file not found: {args.cluster_mapping}")

    tickers = read_tickers(args.cluster_mapping)
    print(f"Read {len(tickers)} tickers from {args.cluster_mapping}")
    fund_name_map = load_fund_short_name_map(args.fund_list_csv)
    print(f"Loaded {len(fund_name_map)} fund-code aliases from {args.fund_list_csv}")

    close = load_close(tickers, args.start_date, args.end_date)
    result = compute_annualized_returns(close)
    result = attach_fund_short_names(result, fund_name_map)

    os.makedirs(args.output_dir, exist_ok=True)
    start_tag = args.start_date.replace("-", "")
    end_tag = args.end_date.replace("-", "")
    out_path = os.path.join(args.output_dir, f"annualized_returns_{start_tag}_{end_tag}.csv")
    result.to_csv(out_path, index=False)
    print(f"Saved {len(result)} rows → {out_path}")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
