"""Build Route-A-style override files from per-instrument annualized returns.

The generated directory can be passed to the main pipeline with
``--mode reuse-route-a --route-a-output-dir <output-dir>``.  This is intended
for oracle / calibration experiments: the same daily mu is written for every
forecast date of an instrument.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_PROVIDER_URI = "~/etf-daily-output/Documents/stock/data/qlib_data/all_fund_data/"
DEFAULT_CLUSTER_MAPPING = "~/etf-daily-output/Documents/stock/data/qlib_data/all_fund_data/instruments/cluster_mapping_selected.txt"
DEFAULT_TEMP_DIR = "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp"
TRADING_DAYS_PER_YEAR = 252.0


PARAMS_COLUMNS = [
    "target_date",
    "forecast_date",
    "mu_raw",
    "mu",
    "mu_smooth",
    "mu_path",
    "posterior_mu_1d",
    "posterior_mu_20d",
    "posterior_mu_std",
    "pred_mean_20",
    "real_mean_20",
    "scale_20",
    "pred_ann20",
    "real_ann20",
    "sigma",
    "nu",
    "mu_shrink",
    "signal_strength",
    "fit_success",
    "fit_fallback",
    "fit_message",
    "skew",
    "kurt",
]

EVAL_COLUMNS = [
    "target_date",
    "forecast_date",
    "y_true",
    "nll",
    "nll_raw",
    "crps",
    "crps_raw",
    "cov90",
    "cov95",
    "cov90_raw",
    "cov95_raw",
    "mae",
    "mae_raw",
    "rmse",
    "rmse_raw",
    "dir_acc",
    "dir_acc_raw",
    "mu",
    "mu_raw",
    "mu_smooth",
    "mu_path",
    "posterior_mu_1d",
    "posterior_mu_20d",
    "posterior_mu_std",
    "pred_ann20",
    "real_ann20",
    "cum_gap_raw",
    "cum_gap_mu",
    "cum_gap_path",
    "signal_strength",
    "fit_success",
    "fit_fallback",
    "active_signal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert annualized mean returns into Route-A override files."
    )
    parser.add_argument("--start-date", required=True, help="Start date, e.g. 2025-05-01")
    parser.add_argument("--end-date", required=True, help="End date, e.g. 2026-04-30")
    parser.add_argument(
        "--annualized-csv",
        default=None,
        help=(
            "Input CSV with instrument and mean_daily_return/annualized_return. "
            "Default: ~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/annualized_returns_<start>_<end>.csv"
        ),
    )
    parser.add_argument("--cluster-mapping", default=DEFAULT_CLUSTER_MAPPING)
    parser.add_argument("--provider-uri", default=DEFAULT_PROVIDER_URI)
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output Route-A override directory. Default: "
            "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/route_a_oracle_mean_<start>_<end>"
        ),
    )
    parser.add_argument(
        "--mu-source",
        default="mean_daily_return",
        choices=["mean_daily_return", "annualized_arithmetic", "annualized_geometric"],
        help="How to derive daily mu from the input CSV.",
    )
    parser.add_argument(
        "--mu-scale",
        type=float,
        default=1.0,
        help="Multiply the derived daily mu before writing override files.",
    )
    parser.add_argument(
        "--winsor-quantile",
        type=float,
        default=None,
        help="Optional two-sided cross-sectional winsor quantile, e.g. 0.01.",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="Optional first-N ticker limit for smoke tests.",
    )
    parser.add_argument(
        "--date-source",
        default="qlib",
        choices=["qlib", "business"],
        help="Use qlib close-date coverage or pandas business dates for forecast_date rows.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete output dir first if it exists.")
    return parser.parse_args()


def compact_date(value: str) -> str:
    return value.replace("-", "")


def default_annualized_csv(start_date: str, end_date: str) -> Path:
    return Path(DEFAULT_TEMP_DIR) / f"annualized_returns_{compact_date(start_date)}_{compact_date(end_date)}.csv"


def default_output_dir(start_date: str, end_date: str) -> Path:
    return Path(DEFAULT_TEMP_DIR) / f"route_a_oracle_mean_{compact_date(start_date)}_{compact_date(end_date)}"


def read_cluster_tickers(path: Path) -> list[str]:
    tickers: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        ticker = stripped.split()[0].upper()
        if ticker:
            tickers.append(ticker)
    if not tickers:
        raise ValueError(f"No tickers found in {path}")
    return tickers


def load_daily_mu(csv_path: Path, mu_source: str, mu_scale: float) -> pd.Series:
    read_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            frame = pd.read_csv(csv_path, encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            read_error = exc
    else:
        raise read_error or UnicodeDecodeError("utf-8", b"", 0, 1, "unable to decode CSV")
    if "instrument" not in frame.columns:
        raise ValueError(f"Input CSV must contain an instrument column: {csv_path}")
    frame = frame.copy()
    frame["instrument"] = frame["instrument"].astype(str).str.upper().str.strip()

    if mu_source == "mean_daily_return":
        if "mean_daily_return" not in frame.columns:
            raise ValueError("--mu-source mean_daily_return requires column mean_daily_return")
        daily_mu = pd.to_numeric(frame["mean_daily_return"], errors="coerce")
    else:
        if "annualized_return" not in frame.columns:
            raise ValueError(f"--mu-source {mu_source} requires column annualized_return")
        annualized = pd.to_numeric(frame["annualized_return"], errors="coerce")
        if mu_source == "annualized_arithmetic":
            daily_mu = annualized / TRADING_DAYS_PER_YEAR
        else:
            daily_mu = annualized.map(
                lambda value: (1.0 + value) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
                if pd.notna(value) and value > -1.0
                else math.nan
            )

    result = pd.Series(daily_mu.to_numpy(dtype=float), index=frame["instrument"], name="daily_mu")
    result = result[~result.index.duplicated(keep="first")]
    return result * float(mu_scale)


def apply_winsor(daily_mu: pd.Series, quantile: float | None) -> pd.Series:
    if quantile is None:
        return daily_mu
    q = float(quantile)
    if q <= 0.0 or q >= 0.5:
        raise ValueError("--winsor-quantile must be in (0, 0.5)")
    finite = daily_mu.dropna()
    if finite.empty:
        return daily_mu
    lower = float(finite.quantile(q))
    upper = float(finite.quantile(1.0 - q))
    return daily_mu.clip(lower=lower, upper=upper)


def qlib_close_dates(
    tickers: list[str],
    *,
    provider_uri: str,
    start_date: str,
    end_date: str,
) -> dict[str, list[pd.Timestamp]]:
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=provider_uri, region="cn")
    raw = D.features(tickers, ["$close"], start_time=start_date, end_time=end_date)
    if raw.empty:
        raise ValueError("qlib returned no close data for the requested window")
    close = raw["$close"].unstack(level="instrument").sort_index()
    result: dict[str, list[pd.Timestamp]] = {}
    for ticker in tickers:
        if ticker not in close.columns:
            result[ticker] = []
            continue
        result[ticker] = [pd.Timestamp(value) for value in close.index[close[ticker].notna()]]
    return result


def business_dates(tickers: Iterable[str], start_date: str, end_date: str) -> dict[str, list[pd.Timestamp]]:
    dates = [pd.Timestamp(value) for value in pd.bdate_range(start_date, end_date)]
    return {ticker: dates for ticker in tickers}


def make_target_dates(forecast_dates: list[pd.Timestamp]) -> list[str]:
    targets: list[str] = []
    for idx, value in enumerate(forecast_dates):
        if idx + 1 < len(forecast_dates):
            targets.append(forecast_dates[idx + 1].strftime("%Y-%m-%d"))
        else:
            targets.append(value.strftime("%Y-%m-%d"))
    return targets


def build_params_frame(forecast_dates: list[pd.Timestamp], daily_mu: float) -> pd.DataFrame:
    date_text = [value.strftime("%Y-%m-%d") for value in forecast_dates]
    frame = pd.DataFrame(
        {
            "target_date": make_target_dates(forecast_dates),
            "forecast_date": date_text,
            "mu_raw": daily_mu,
            "mu": daily_mu,
            "mu_smooth": daily_mu,
            "mu_path": daily_mu,
            "posterior_mu_1d": daily_mu,
            "posterior_mu_20d": daily_mu,
            "posterior_mu_std": 0.0,
            "pred_mean_20": daily_mu,
            "real_mean_20": daily_mu,
            "scale_20": 1.0,
            "pred_ann20": daily_mu * TRADING_DAYS_PER_YEAR,
            "real_ann20": daily_mu * TRADING_DAYS_PER_YEAR,
            "sigma": pd.NA,
            "nu": 8.0,
            "mu_shrink": daily_mu,
            "signal_strength": abs(float(daily_mu)),
            "fit_success": True,
            "fit_fallback": False,
            "fit_message": "oracle_mean_override",
            "skew": pd.NA,
            "kurt": pd.NA,
        }
    )
    return frame[PARAMS_COLUMNS]


def build_eval_frame(params: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=params.index)
    frame["target_date"] = params["target_date"]
    frame["forecast_date"] = params["forecast_date"]
    frame["y_true"] = pd.NA
    for column in ["nll", "nll_raw", "crps", "crps_raw", "mae", "mae_raw", "rmse", "rmse_raw"]:
        frame[column] = pd.NA
    for column in ["cov90", "cov95", "cov90_raw", "cov95_raw"]:
        frame[column] = pd.NA
    frame["dir_acc"] = pd.NA
    frame["dir_acc_raw"] = pd.NA
    for column in [
        "mu",
        "mu_raw",
        "mu_smooth",
        "mu_path",
        "posterior_mu_1d",
        "posterior_mu_20d",
        "posterior_mu_std",
        "pred_ann20",
        "real_ann20",
        "signal_strength",
        "fit_success",
        "fit_fallback",
    ]:
        frame[column] = params[column]
    frame["cum_gap_raw"] = pd.NA
    frame["cum_gap_mu"] = pd.NA
    frame["cum_gap_path"] = pd.NA
    frame["active_signal"] = (pd.to_numeric(params["signal_strength"], errors="coerce") > 0.0).astype(float)
    return frame[EVAL_COLUMNS]


def write_override_files(
    *,
    tickers: list[str],
    daily_mu: pd.Series,
    dates_by_ticker: dict[str, list[pd.Timestamp]],
    output_dir: Path,
    start_label: str,
    end_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        mu_value = daily_mu.get(ticker, math.nan)
        dates = dates_by_ticker.get(ticker, [])
        status = "ok"
        message = ""
        if pd.isna(mu_value):
            status = "missing_mu"
            message = "missing daily mu in annualized CSV"
        elif not dates:
            status = "missing_dates"
            message = "no forecast dates available"
        else:
            params = build_params_frame(dates, float(mu_value))
            eval_frame = build_eval_frame(params)
            params_path = output_dir / f"route_a_params_{ticker}_{start_label}_{end_label}.csv"
            eval_path = output_dir / f"route_a_eval_daily_{ticker}_{start_label}_{end_label}.csv"
            params.to_csv(params_path, index=False)
            eval_frame.to_csv(eval_path, index=False)
            message = f"wrote {len(params)} forecast rows"
        rows.append(
            {
                "instrument": ticker,
                "status": status,
                "message": message,
                "daily_mu": mu_value,
                "annualized_mu_arithmetic": mu_value * TRADING_DAYS_PER_YEAR if pd.notna(mu_value) else math.nan,
                "forecast_rows": len(dates),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    start_label = compact_date(args.start_date)
    end_label = compact_date(args.end_date)
    annualized_csv = Path(args.annualized_csv).expanduser().resolve() if args.annualized_csv else default_annualized_csv(args.start_date, args.end_date)
    cluster_mapping = Path(args.cluster_mapping).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir(args.start_date, args.end_date)

    if not annualized_csv.exists():
        raise FileNotFoundError(f"annualized CSV not found: {annualized_csv}")
    if not cluster_mapping.exists():
        raise FileNotFoundError(f"cluster mapping not found: {cluster_mapping}")
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output dir exists; pass --overwrite to replace: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers = read_cluster_tickers(cluster_mapping)
    if args.max_tickers is not None:
        tickers = tickers[: int(args.max_tickers)]
    daily_mu = load_daily_mu(annualized_csv, args.mu_source, args.mu_scale)
    daily_mu = apply_winsor(daily_mu, args.winsor_quantile)

    if args.date_source == "qlib":
        dates_by_ticker = qlib_close_dates(
            tickers,
            provider_uri=args.provider_uri,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    else:
        dates_by_ticker = business_dates(tickers, args.start_date, args.end_date)

    status = write_override_files(
        tickers=tickers,
        daily_mu=daily_mu,
        dates_by_ticker=dates_by_ticker,
        output_dir=output_dir,
        start_label=start_label,
        end_label=end_label,
    )
    status_path = output_dir / "oracle_mean_override_status.csv"
    status.to_csv(status_path, index=False)

    manifest = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "annualized_csv": str(annualized_csv),
        "cluster_mapping": str(cluster_mapping),
        "provider_uri": str(args.provider_uri),
        "output_dir": str(output_dir),
        "mu_source": args.mu_source,
        "mu_scale": args.mu_scale,
        "winsor_quantile": args.winsor_quantile,
        "date_source": args.date_source,
        "tickers_requested": len(tickers),
        "tickers_ok": int((status["status"] == "ok").sum()),
        "params_files": int(len(list(output_dir.glob("route_a_params_*.csv")))),
        "eval_files": int(len(list(output_dir.glob("route_a_eval_daily_*.csv")))),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote oracle Route-A override to: {output_dir}")
    print(f"Status CSV: {status_path}")
    print(f"Manifest: {manifest_path}")
    print(status["status"].value_counts(dropna=False).to_string())
    if not (status["status"] == "ok").all():
        print(status[status["status"] != "ok"].head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())