"""Missed-winner audit for ETF cluster selection.

Refactored from ``2_filter/0.5_audit_missed_winners.py``. Instead of loading
the legacy monolith via ``importlib.util``, this module constructs a small
compatibility shim (:func:`_make_cluster_module_shim`) that exposes the exact
set of attributes the audit logic expects (constants + ``build_code_name_map``,
``load_data``, ``D``, ``qlib``, ``TEST_PERIOD`` …), all backed by the new
:mod:`pool_builder` modules.

Usage:

    python -m pool_builder.audit --future_end 2026-04-22
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import constants as C
from . import data_loading as _dl
from .config import load_shared_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.price_adjustments import auto_qfq_adjust_close_panel, log_qfq_adjustments


DEFAULT_INPUT_DIR = "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/etf_filter_checks"
DEFAULT_MAPPING_CSV = os.path.join(DEFAULT_INPUT_DIR, "cluster_mapping_etf.csv")
DEFAULT_OUT_DIR = DEFAULT_INPUT_DIR
DEFAULT_AUDIT_CSV = "missed_winner_audit_etf.csv"
DEFAULT_TOP_REGRET_CSV = "top_regret_same_cluster_etf.csv"
DEFAULT_TOP_CLUSTER_CSV = "top_missed_clusters_etf.csv"
DEFAULT_LOW_VOLUME_CSV = "low_volume_high_future_return_etf.csv"
DEFAULT_UNSTABLE_CSV = "unstable_cluster_cases_etf.csv"
DEFAULT_WINDOWS = "5,20,60,120"
DEFAULT_STRONG_PERCENTILE = 0.90
DEFAULT_WRONG_CLUSTER_CORR = 0.55
DEFAULT_WRONG_CLUSTER_CENTER_CORR = 0.55
DEFAULT_MIN_REGRET = 0.10
DEFAULT_FUTURE_END = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")


def _make_cluster_module_shim(shared_config_path: str | None = None) -> SimpleNamespace:
    """Build a back-compat shim that exposes everything the audit needs."""
    cfg = load_shared_config(shared_config_path)
    provider_uri = cfg["provider_uri"]
    test_period = tuple(cfg["test_period"])
    fund_list_csv = cfg["fund_list_csv"]

    def _build_code_name_map_shim(csv_path, exclude_types=None, inception_cutoff=None,
                                  manual_excludes=None, cluster_excludes=None,
                                  cluster_reference_map_csv=None):
        return _dl.build_code_name_map(
            csv_path,
            exclude_types=exclude_types,
            inception_cutoff=inception_cutoff,
            manual_excludes=manual_excludes,
            cluster_excludes=cluster_excludes,
            cluster_reference_map_csv=cluster_reference_map_csv,
        )

    def _load_data_shim(instruments, excluded_codes=None):
        return _dl.load_data(
            instruments,
            provider_uri=provider_uri,
            test_period=test_period,
            excluded_codes=excluded_codes,
        )

    shim = SimpleNamespace(
        # symbols the audit script reads
        qlib=_dl.qlib,
        D=_dl.D,
        REG_CN=_dl.REG_CN,
        FUND_LIST_CSV=fund_list_csv,
        TEST_PERIOD=test_period,
        INCEPTION_CUTOFF=test_period[1],
        EXCLUDE_TYPES=C.EXCLUDE_TYPES,
        ALWAYS_DROP_CODES=C.ALWAYS_DROP_CODES,
        ALWAYS_DROP_CLUSTER_CODES=C.ALWAYS_DROP_CLUSTER_CODES,
        TARGET_SELECTED_COUNT=C.TARGET_SELECTED_COUNT,
        RULE=C.RULE,
        DIST_T=C.DIST_T,
        PER_CLUSTER=C.PER_CLUSTER,
        # callables
        build_code_name_map=_build_code_name_map_shim,
        load_data=_load_data_shim,
    )
    return shim


# ────────────────────────────────────────────────────────────────────────────
# Original audit logic (ported verbatim; only module-loading layer changed).
# ────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Audit missed winners after ETF cluster selection")
    parser.add_argument("--mapping_csv", default=DEFAULT_MAPPING_CSV)
    parser.add_argument("--future_end", default=DEFAULT_FUTURE_END)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--windows", default=DEFAULT_WINDOWS)
    parser.add_argument("--strong_percentile", type=float, default=DEFAULT_STRONG_PERCENTILE)
    parser.add_argument("--wrong_cluster_corr", type=float, default=DEFAULT_WRONG_CLUSTER_CORR)
    parser.add_argument("--wrong_cluster_center_corr", type=float, default=DEFAULT_WRONG_CLUSTER_CENTER_CORR)
    parser.add_argument("--min_regret", type=float, default=DEFAULT_MIN_REGRET)
    parser.add_argument("--top_n", type=int, default=50)
    parser.add_argument("--shared-config", default=None, help="Override shared_filter_config.json path")
    return parser.parse_args()


def parse_windows(windows_text: str) -> List[int]:
    windows = [int(tok.strip()) for tok in str(windows_text).split(",") if tok.strip()]
    windows = sorted(set(windows))
    if not windows:
        raise ValueError("No valid windows were provided")
    return windows


def _normalize_selected(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes"}).fillna(False)


def load_mapping_csv(mapping_csv: str) -> pd.DataFrame:
    if not os.path.exists(mapping_csv):
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv}")
    df = pd.read_csv(mapping_csv)
    required = {"code", "cluster", "selected", "reason"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {mapping_csv}: {sorted(missing)}")
    df = df.copy()
    df["code"] = df["code"].astype(str).str.strip()
    df = df[df["code"] != ""].copy()
    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce")
    df = df[df["cluster"].notna()].copy()
    df["cluster"] = df["cluster"].astype(int)
    df["selected"] = _normalize_selected(df["selected"])
    for col in ("name", "n", "mean_corr", "mean_volatility", "mean_return", "avg_volume"):
        if col not in df.columns:
            df[col] = "" if col == "name" else np.nan
    return df


def resolve_fund_list_path(cluster_module) -> str:
    path = getattr(cluster_module, "FUND_LIST_CSV", None)
    if not path:
        raise RuntimeError("Cluster module does not expose FUND_LIST_CSV")
    return path


def build_code_map_state(cluster_module, mapping_df: pd.DataFrame):
    code_map, excluded_codes, selected_codes = cluster_module.build_code_name_map(
        resolve_fund_list_path(cluster_module),
        cluster_module.EXCLUDE_TYPES,
        cluster_module.INCEPTION_CUTOFF,
        manual_excludes=cluster_module.ALWAYS_DROP_CODES,
    )
    return code_map, excluded_codes, list(dict.fromkeys(selected_codes))


def load_cluster_data(cluster_module, selected_codes: List[str], excluded_codes):
    excluded_union = excluded_codes.union(cluster_module.ALWAYS_DROP_CODES)
    return cluster_module.load_data(selected_codes, excluded_union)


def build_runtime_state(cluster_module, mapping_csv: str):
    if getattr(cluster_module, "qlib", None) is None:
        raise RuntimeError("qlib is unavailable in the current environment")

    mapping_df = load_mapping_csv(mapping_csv)
    code_map, excluded_codes, selected_codes = build_code_map_state(cluster_module, mapping_df)
    close, volume_df = load_cluster_data(cluster_module, selected_codes, excluded_codes)
    keep_codes = [code for code in mapping_df["code"].tolist() if code in close.columns]
    if not keep_codes:
        raise RuntimeError("No mapping codes were matched in loaded qlib data")
    mapping_df = mapping_df[mapping_df["code"].isin(keep_codes)].copy()
    close = close[keep_codes]
    volume_df = volume_df[keep_codes]
    returns = close.pct_change().dropna(how="all")
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    name_map = dict(zip(mapping_df["code"], mapping_df["name"].fillna("")))
    for code in mapping_df["code"]:
        if not name_map.get(code):
            name_map[code] = code_map.get(code, "")
    final_selected = mapping_df.loc[mapping_df["selected"], "code"].tolist()
    final_reasons = dict(zip(mapping_df["code"], mapping_df["reason"].fillna("")))

    return {
        "code_map": code_map,
        "name_map": name_map,
        "mapping_df": mapping_df,
        "close": close,
        "volume_df": volume_df,
        "returns": returns,
        "final_selected": list(dict.fromkeys(final_selected)),
        "final_reasons": final_reasons,
    }


def safe_lookup_name(code, name_map):
    return name_map.get(code, "")


def build_cluster_member_map(mapping_df: pd.DataFrame) -> Dict[int, List[str]]:
    return mapping_df.groupby("cluster")["code"].apply(list).to_dict()


def build_selected_by_cluster(mapping_df: pd.DataFrame) -> Dict[int, List[str]]:
    selected_df = mapping_df[mapping_df["selected"]].copy()
    if selected_df.empty:
        return {}
    return selected_df.groupby("cluster")["code"].apply(list).to_dict()


def pick_primary_rep(codes: List[str], reason_map: Dict[str, str], cluster_id: int) -> Optional[str]:
    if not codes:
        return None
    suffix = f"cluster_{cluster_id}"
    for code in codes:
        reason = reason_map.get(code, "")
        if suffix in reason or reason == "forced_keep":
            return code
    return codes[0]


def _load_qlib_day_calendar(provider_uri: str | os.PathLike) -> list[str]:
    calendar_path = Path(provider_uri).expanduser() / "calendars" / "day.txt"
    if not calendar_path.is_file():
        raise RuntimeError(f"Qlib calendar not found: {calendar_path}")
    return [line.strip() for line in calendar_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_qlib_feature_bin(
    bin_path: Path,
    calendar: list[str],
    start_date: str,
    end_date: str,
) -> pd.Series:
    raw = bin_path.read_bytes()
    if len(raw) < 4:
        return pd.Series(dtype=float)
    start_idx = int(struct.unpack("<f", raw[:4])[0])
    values = np.frombuffer(raw[4:], dtype="<f")
    if values.size == 0:
        return pd.Series(dtype=float)
    end_idx = start_idx + len(values) - 1
    if end_idx >= len(calendar):
        raise RuntimeError(f"Qlib bin index out of range for {bin_path}")
    series = pd.Series(values, index=pd.to_datetime(calendar[start_idx : end_idx + 1]), dtype=float)
    series.index.name = "Date"
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    return series.loc[(series.index >= start_ts) & (series.index <= end_ts)]


def _qfq_adjust_close(close: pd.DataFrame, *, stage: str) -> pd.DataFrame:
    adjusted, records = auto_qfq_adjust_close_panel(close)
    if not records.empty:
        log_qfq_adjustments(records, stage=stage)
    return adjusted


def load_close_from_qlib_bins(
    provider_uri: str | os.PathLike,
    codes: List[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load close prices directly from on-disk qlib feature bins (no pyqlib import)."""
    calendar = _load_qlib_day_calendar(provider_uri)
    feature_root = Path(provider_uri).expanduser() / "features"
    if not feature_root.is_dir():
        raise RuntimeError(f"Qlib features directory not found: {feature_root}")

    chunks: dict[str, pd.Series] = {}
    for code in codes:
        code_key = str(code).strip().lower()
        if not code_key:
            continue
        bin_path = feature_root / code_key / "close.day.bin"
        if not bin_path.is_file():
            continue
        series = _read_qlib_feature_bin(bin_path, calendar, start_date, end_date)
        if series.empty:
            continue
        chunks[code_key] = series

    if not chunks:
        raise RuntimeError("No close prices loaded from local qlib feature bins")

    close = pd.DataFrame(chunks).sort_index()
    close.index.name = "Date"
    close = close.sort_index().ffill().bfill()
    return _qfq_adjust_close(close, stage="pool_builder.load_close_from_qlib_bins")


def build_review_metrics_from_local_qlib(
    provider_uri: str | os.PathLike,
    mapping_df: pd.DataFrame,
    build_period: tuple[str, str],
    future_end: str,
    windows: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build forward metrics and build-window returns when pyqlib import is unavailable."""
    build_start, build_end = build_period
    codes = mapping_df["code"].astype(str).tolist()
    close = load_close_from_qlib_bins(provider_uri, codes, build_start, future_end)
    keep_codes = [code for code in codes if str(code).strip().lower() in close.columns]
    if not keep_codes:
        raise RuntimeError("No mapping codes were matched in local qlib feature bins")
    close = close[[str(code).strip().lower() for code in keep_codes]]
    close.columns = keep_codes

    build_end_ts = pd.Timestamp(build_end)
    build_close = close.loc[close.index <= build_end_ts]
    returns = build_close.pct_change().dropna(how="all")
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)

    volume_chunks: dict[str, pd.Series] = {}
    calendar = _load_qlib_day_calendar(provider_uri)
    feature_root = Path(provider_uri).expanduser() / "features"
    for code in keep_codes:
        code_key = str(code).strip().lower()
        volume_path = feature_root / code_key / "volume.day.bin"
        if volume_path.is_file():
            volume_chunks[code] = _read_qlib_feature_bin(volume_path, calendar, build_start, build_end)
    if not volume_chunks:
        volume_df = pd.DataFrame(0.0, index=build_close.index, columns=keep_codes)
    else:
        volume_df = pd.DataFrame(volume_chunks).reindex(columns=keep_codes).fillna(0.0)

    member_map = build_cluster_member_map(mapping_df[mapping_df["code"].isin(keep_codes)])
    future_df = compute_trailing_return_metrics(close, future_end, windows)
    build_metrics, _ = build_build_window_metrics(returns, volume_df, member_map)
    filtered_mapping = mapping_df[mapping_df["code"].isin(keep_codes)].copy()
    return filtered_mapping, future_df, build_metrics, returns


def fetch_close_prices(cluster_module, codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    raw = cluster_module.D.features(
        codes,
        fields=["$close"],
        start_time=start_date,
        end_time=end_date,
    )
    if raw is None or raw.empty:
        raise RuntimeError("No close price data returned for the audit window")
    close = raw["$close"].unstack(level="instrument")
    close.index = pd.to_datetime(close.index)
    close.index.name = "Date"
    close = close.sort_index().fillna(method="ffill").fillna(method="bfill")
    return _qfq_adjust_close(close, stage="pool_builder.fetch_close_prices")


def compute_forward_metrics(close: pd.DataFrame, build_end: str, windows: List[int]) -> pd.DataFrame:
    build_end_ts = pd.Timestamp(build_end)
    close = close.sort_index()
    anchor_candidates = np.where(close.index <= build_end_ts)[0]
    if len(anchor_candidates) == 0:
        raise RuntimeError(f"No price found on or before build_end={build_end}")
    anchor_pos = int(anchor_candidates[-1])
    anchor_prices = close.iloc[anchor_pos]

    rows = []
    for code in close.columns:
        row = {"code": code}
        anchor_price = anchor_prices.get(code, np.nan)
        if pd.isna(anchor_price) or anchor_price == 0:
            rows.append(row)
            continue
        series = close[code]
        for window in windows:
            end_pos = anchor_pos + window
            if end_pos >= len(close.index):
                row[f"fwd_ret_{window}d"] = np.nan
                row[f"fwd_max_upside_{window}d"] = np.nan
                row[f"fwd_drawdown_{window}d"] = np.nan
                continue
            future_slice = series.iloc[anchor_pos + 1:end_pos + 1].dropna()
            if future_slice.empty:
                row[f"fwd_ret_{window}d"] = np.nan
                row[f"fwd_max_upside_{window}d"] = np.nan
                row[f"fwd_drawdown_{window}d"] = np.nan
                continue
            end_price = future_slice.iloc[-1]
            row[f"fwd_ret_{window}d"] = end_price / anchor_price - 1.0
            row[f"fwd_max_upside_{window}d"] = future_slice.max() / anchor_price - 1.0
            row[f"fwd_drawdown_{window}d"] = future_slice.min() / anchor_price - 1.0
        rows.append(row)

    future_df = pd.DataFrame(rows)
    for window in windows:
        col = f"fwd_ret_{window}d"
        future_df[f"market_ret_rank_{window}d"] = future_df[col].rank(pct=True, method="average")
    return future_df


def compute_trailing_return_metrics(close: pd.DataFrame, as_of: str, windows: List[int]) -> pd.DataFrame:
    """Trailing returns over N trading days ending on or before as_of."""
    as_of_ts = pd.Timestamp(as_of)
    close = close.sort_index()
    end_candidates = np.where(close.index <= as_of_ts)[0]
    if len(end_candidates) == 0:
        raise RuntimeError(f"No price found on or before as_of={as_of}")
    end_pos = int(end_candidates[-1])
    end_prices = close.iloc[end_pos]

    rows = []
    for code in close.columns:
        row = {"code": code}
        end_price = end_prices.get(code, np.nan)
        if pd.isna(end_price) or end_price == 0:
            rows.append(row)
            continue
        series = close[code]
        for window in windows:
            start_pos = end_pos - window
            if start_pos < 0:
                row[f"fwd_ret_{window}d"] = np.nan
                row[f"fwd_max_upside_{window}d"] = np.nan
                row[f"fwd_drawdown_{window}d"] = np.nan
                continue
            trail_slice = series.iloc[start_pos : end_pos + 1].dropna()
            if trail_slice.empty:
                row[f"fwd_ret_{window}d"] = np.nan
                row[f"fwd_max_upside_{window}d"] = np.nan
                row[f"fwd_drawdown_{window}d"] = np.nan
                continue
            start_price = trail_slice.iloc[0]
            if pd.isna(start_price) or start_price == 0:
                row[f"fwd_ret_{window}d"] = np.nan
                row[f"fwd_max_upside_{window}d"] = np.nan
                row[f"fwd_drawdown_{window}d"] = np.nan
                continue
            row[f"fwd_ret_{window}d"] = end_price / start_price - 1.0
            row[f"fwd_max_upside_{window}d"] = trail_slice.max() / start_price - 1.0
            row[f"fwd_drawdown_{window}d"] = trail_slice.min() / start_price - 1.0
        rows.append(row)

    future_df = pd.DataFrame(rows)
    for window in windows:
        col = f"fwd_ret_{window}d"
        future_df[f"market_ret_rank_{window}d"] = future_df[col].rank(pct=True, method="average")
    return future_df


def build_build_window_metrics(returns: pd.DataFrame, volume_df: pd.DataFrame, member_map: Dict[int, List[str]]):
    corr = returns.corr().fillna(0)
    rows = []
    avg_volume = volume_df.mean()
    mean_return = returns.mean()
    volatility = returns.std()

    for cluster_id, members in member_map.items():
        member_count = len(members)
        cluster_corr = corr.loc[members, members]
        cluster_center = cluster_corr.mean(axis=1)
        volume_rank = avg_volume[members].rank(method="min", ascending=False)
        return_rank = mean_return[members].rank(method="min", ascending=False)
        volatility_rank = volatility[members].rank(method="min", ascending=False)

        for code in members:
            if member_count <= 1:
                corr_to_center = 1.0
            else:
                corr_to_center = cluster_corr.loc[code].drop(code, errors="ignore").mean()
            rows.append(
                {
                    "code": code,
                    "stock_avg_volume": float(avg_volume.get(code, np.nan)),
                    "stock_mean_return": float(mean_return.get(code, np.nan)),
                    "stock_volatility": float(volatility.get(code, np.nan)),
                    "volume_rank_in_cluster": float(volume_rank.get(code, np.nan)),
                    "return_rank_in_cluster": float(return_rank.get(code, np.nan)),
                    "volatility_rank_in_cluster": float(volatility_rank.get(code, np.nan)),
                    "corr_to_cluster_center": float(corr_to_center),
                    "cluster_center_score": float(cluster_center.get(code, np.nan)),
                }
            )
    return pd.DataFrame(rows), corr


def attach_rep_fields(df: pd.DataFrame, rep_prefix: str, rep_code_col: str, metric_df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    merged = metric_df.rename(columns={"code": rep_code_col})
    cols = [rep_code_col, "stock_avg_volume", "stock_mean_return", "stock_volatility"]
    rename_map = {
        "stock_avg_volume": f"{rep_prefix}_avg_volume",
        "stock_mean_return": f"{rep_prefix}_mean_return",
        "stock_volatility": f"{rep_prefix}_volatility",
    }
    for window in windows:
        cols.extend(
            [
                f"fwd_ret_{window}d",
                f"fwd_max_upside_{window}d",
                f"fwd_drawdown_{window}d",
            ]
        )
        rename_map[f"fwd_ret_{window}d"] = f"{rep_prefix}_fwd_ret_{window}d"
        rename_map[f"fwd_max_upside_{window}d"] = f"{rep_prefix}_fwd_max_upside_{window}d"
        rename_map[f"fwd_drawdown_{window}d"] = f"{rep_prefix}_fwd_drawdown_{window}d"
    merged = merged[cols].rename(columns=rename_map)
    return df.merge(merged, on=rep_code_col, how="left")


def assign_miss_type(row, wrong_cluster_corr: float, wrong_cluster_center_corr: float) -> str:
    if bool(row.get("selected", False)):
        return "selected"
    if not bool(row.get("cluster_has_selected_rep", False)):
        return "cluster_cut_miss"
    corr_to_rep = row.get("corr_to_rep_in_build_window")
    corr_to_center = row.get("corr_to_cluster_center")
    if pd.notna(corr_to_rep) and pd.notna(corr_to_center):
        if corr_to_rep < wrong_cluster_corr and corr_to_center < wrong_cluster_center_corr:
            return "wrong_cluster_miss"
    return "same_cluster_miss"


def assign_primary_reason(row) -> str:
    if row["miss_type"] == "cluster_cut_miss":
        return "cluster_cut_by_target_count"
    if row["miss_type"] == "wrong_cluster_miss":
        return "wrong_cluster_assignment"
    if row.get("cluster_has_selected_rep", False):
        rep_avg_volume = row.get("selected_rep_avg_volume")
        stock_avg_volume = row.get("stock_avg_volume")
        if pd.notna(rep_avg_volume) and pd.notna(stock_avg_volume) and stock_avg_volume < rep_avg_volume:
            return "low_volume_lost"
        if row.get("volume_rank_in_cluster", np.inf) > 1:
            return "not_top1_in_cluster"
    return "weak_build_window_signal"


def assign_secondary_reason(row) -> str:
    if row["miss_type"] == "cluster_cut_miss":
        return "no_final_rep_in_cluster"
    if row.get("volume_rank_in_cluster", np.inf) > 1 and row.get("return_rank_in_cluster", np.inf) <= 3:
        return "build_return_good_but_volume_weaker"
    if row.get("corr_to_rep_in_build_window", 1.0) < 0.6:
        return "weak_similarity_to_rep"
    return ""


def add_regret_fields(df: pd.DataFrame, windows: List[int], portfolio_returns: Dict[int, float]) -> pd.DataFrame:
    for window in windows:
        own_ret = df[f"fwd_ret_{window}d"]
        if f"selected_rep_fwd_ret_{window}d" in df.columns:
            df[f"regret_{window}d"] = own_ret - df[f"selected_rep_fwd_ret_{window}d"]
        if f"pretrim_rep_fwd_ret_{window}d" in df.columns:
            df[f"pretrim_regret_{window}d"] = own_ret - df[f"pretrim_rep_fwd_ret_{window}d"]
        own_upside = df[f"fwd_max_upside_{window}d"]
        own_drawdown = df[f"fwd_drawdown_{window}d"]
        if f"selected_rep_fwd_max_upside_{window}d" in df.columns:
            df[f"upside_gap_{window}d"] = own_upside - df[f"selected_rep_fwd_max_upside_{window}d"]
        if f"selected_rep_fwd_drawdown_{window}d" in df.columns:
            df[f"drawdown_gap_{window}d"] = own_drawdown - df[f"selected_rep_fwd_drawdown_{window}d"]
        df[f"selected_portfolio_ret_{window}d"] = portfolio_returns.get(window, np.nan)
        df[f"portfolio_regret_{window}d"] = own_ret - df[f"selected_portfolio_ret_{window}d"]
    return df


def build_portfolio_returns(audit_base: pd.DataFrame, selected_codes: List[str], windows: List[int]) -> Dict[int, float]:
    selected_df = audit_base[audit_base["code"].isin(set(selected_codes))]
    result = {}
    for window in windows:
        col = f"fwd_ret_{window}d"
        if col in selected_df.columns and not selected_df.empty:
            result[window] = float(selected_df[col].mean())
        else:
            result[window] = np.nan
    return result


def build_audit_table(cluster_module, state, future_end: str, windows: List[int],
                      strong_percentile: float, wrong_cluster_corr: float,
                      wrong_cluster_center_corr: float, min_regret: float) -> pd.DataFrame:
    returns = state["returns"]
    volume_df = state["volume_df"]
    mapping_df = state["mapping_df"].copy()
    name_map = state["name_map"]
    final_reasons = state["final_reasons"]
    member_map = build_cluster_member_map(mapping_df)
    final_by_cluster = build_selected_by_cluster(mapping_df)
    build_metric_df, corr = build_build_window_metrics(returns, volume_df, member_map)
    future_start = (pd.Timestamp(cluster_module.TEST_PERIOD[1]) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    future_close = fetch_close_prices(
        cluster_module,
        mapping_df["code"].tolist(),
        future_start,
        future_end,
    )
    future_metric_df = compute_forward_metrics(future_close, cluster_module.TEST_PERIOD[1], windows)

    audit_df = mapping_df.merge(build_metric_df, on="code", how="left")
    audit_df = audit_df.merge(future_metric_df, on="code", how="left")

    selected_rep_codes = {}
    for cluster_id in sorted(member_map):
        final_codes = final_by_cluster.get(cluster_id, [])
        selected_rep_codes[cluster_id] = pick_primary_rep(final_codes, final_reasons, cluster_id)

    audit_df["cluster_size"] = audit_df["n"]
    audit_df["cluster_mean_corr"] = audit_df["mean_corr"]
    audit_df["cluster_mean_volatility"] = audit_df["mean_volatility"]
    audit_df["cluster_mean_return"] = audit_df["mean_return"]
    audit_df["cluster_avg_volume"] = audit_df["avg_volume"]
    audit_df["cluster_has_selected_rep"] = audit_df["cluster"].map(lambda c: bool(selected_rep_codes.get(int(c))))
    audit_df["selected_rep_code"] = audit_df["cluster"].map(lambda c: selected_rep_codes.get(int(c)))
    audit_df["selected_rep_name"] = audit_df["selected_rep_code"].map(lambda c: safe_lookup_name(c, name_map) if pd.notna(c) else "")

    corr_to_rep = []
    dist_to_rep = []
    for _, row in audit_df.iterrows():
        rep_code = row.get("selected_rep_code")
        if pd.isna(rep_code) or rep_code not in corr.index or row["code"] not in corr.index:
            corr_to_rep.append(np.nan)
            dist_to_rep.append(np.nan)
            continue
        value = float(corr.loc[row["code"], rep_code])
        corr_to_rep.append(value)
        dist_to_rep.append(1.0 - value)
    audit_df["corr_to_rep_in_build_window"] = corr_to_rep
    audit_df["distance_to_rep_in_build_window"] = dist_to_rep

    rep_source_df = audit_df[
        ["code", "stock_avg_volume", "stock_mean_return", "stock_volatility"]
        + [f"fwd_ret_{window}d" for window in windows]
        + [f"fwd_max_upside_{window}d" for window in windows]
        + [f"fwd_drawdown_{window}d" for window in windows]
    ].copy()
    audit_df = attach_rep_fields(audit_df, "selected_rep", "selected_rep_code", rep_source_df, windows)

    portfolio_returns = build_portfolio_returns(audit_df, state["final_selected"], windows)
    audit_df = add_regret_fields(audit_df, windows, portfolio_returns)

    for window in windows:
        audit_df[f"future_strong_{window}d"] = audit_df[f"market_ret_rank_{window}d"] >= strong_percentile

    strong_cols = [f"future_strong_{window}d" for window in windows]
    audit_df["is_strong_post"] = audit_df[strong_cols].any(axis=1)
    audit_df["miss_type"] = audit_df.apply(
        assign_miss_type,
        axis=1,
        wrong_cluster_corr=wrong_cluster_corr,
        wrong_cluster_center_corr=wrong_cluster_center_corr,
    )
    audit_df["miss_reason_primary"] = audit_df.apply(assign_primary_reason, axis=1)
    audit_df["miss_reason_secondary"] = audit_df.apply(assign_secondary_reason, axis=1)
    audit_df["likely_due_to_rule"] = (
        (audit_df["miss_type"] == "same_cluster_miss")
        & (audit_df["miss_reason_primary"] == "low_volume_lost")
    )
    audit_df["likely_due_to_target_count"] = audit_df["miss_type"] == "cluster_cut_miss"
    audit_df["likely_due_to_clustering"] = audit_df["miss_type"] == "wrong_cluster_miss"
    audit_df["review_comment"] = ""

    focus_window = max([window for window in windows if window <= 60], default=windows[0])
    regret_col = f"regret_{focus_window}d"
    portfolio_regret_col = f"portfolio_regret_{focus_window}d"
    audit_df["review_priority"] = np.where(
        audit_df["is_strong_post"] & (audit_df.get(regret_col, pd.Series(np.nan, index=audit_df.index)).fillna(-np.inf) >= min_regret),
        "high",
        np.where(audit_df["is_strong_post"], "medium", "low"),
    )
    audit_df["review_priority_score"] = audit_df["review_priority"].map({"high": 2, "medium": 1, "low": 0}).fillna(0)
    audit_df["audit_date"] = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    audit_df["build_window_start"] = cluster_module.TEST_PERIOD[0]
    audit_df["build_window_end"] = cluster_module.TEST_PERIOD[1]
    audit_df["target_selected_count"] = getattr(cluster_module, "TARGET_SELECTED_COUNT", None)
    audit_df["selection_rule"] = cluster_module.RULE
    audit_df["distance_threshold"] = cluster_module.DIST_T
    audit_df["per_cluster"] = cluster_module.PER_CLUSTER
    audit_df["focus_regret_window"] = focus_window
    audit_df["focus_regret_value"] = audit_df.get(regret_col, np.nan)
    audit_df["focus_portfolio_regret"] = audit_df.get(portfolio_regret_col, np.nan)

    unselected_df = audit_df[~audit_df["selected"]].copy()
    return unselected_df.sort_values(
        ["review_priority_score", "is_strong_post", "focus_regret_value"],
        ascending=[False, False, False],
    )


def save_focus_outputs(audit_df: pd.DataFrame, out_dir: str, windows: List[int], top_n: int):
    os.makedirs(out_dir, exist_ok=True)
    focus_window = int(audit_df["focus_regret_window"].iloc[0]) if not audit_df.empty else windows[0]
    regret_col = f"regret_{focus_window}d"

    audit_df.to_csv(os.path.join(out_dir, DEFAULT_AUDIT_CSV), index=False)

    same_cluster = audit_df[audit_df["miss_type"] == "same_cluster_miss"].sort_values(
        ["is_strong_post", regret_col],
        ascending=[False, False],
    )
    same_cluster.head(top_n).to_csv(os.path.join(out_dir, DEFAULT_TOP_REGRET_CSV), index=False)

    missed_clusters = audit_df[audit_df["miss_type"] == "cluster_cut_miss"].copy()
    if not missed_clusters.empty:
        missed_clusters = missed_clusters.sort_values(
            ["is_strong_post", f"fwd_ret_{focus_window}d"],
            ascending=[False, False],
        )
        missed_clusters = missed_clusters.drop_duplicates(subset=["cluster"], keep="first")
    missed_clusters.head(top_n).to_csv(os.path.join(out_dir, DEFAULT_TOP_CLUSTER_CSV), index=False)

    low_volume = audit_df[
        (audit_df["volume_rank_in_cluster"] > 1) & audit_df["is_strong_post"]
    ].sort_values([f"fwd_ret_{focus_window}d", regret_col], ascending=[False, False])
    low_volume.head(top_n).to_csv(os.path.join(out_dir, DEFAULT_LOW_VOLUME_CSV), index=False)

    unstable = audit_df[
        (audit_df["miss_type"] == "wrong_cluster_miss")
        | (audit_df["corr_to_rep_in_build_window"] < 0.55)
    ].sort_values(["is_strong_post", regret_col], ascending=[False, False])
    unstable.head(top_n).to_csv(os.path.join(out_dir, DEFAULT_UNSTABLE_CSV), index=False)


def main():
    args = parse_args()
    windows = parse_windows(args.windows)
    cluster_module = _make_cluster_module_shim(args.shared_config)
    state = build_runtime_state(cluster_module, args.mapping_csv)
    audit_df = build_audit_table(
        cluster_module,
        state,
        args.future_end,
        windows,
        args.strong_percentile,
        args.wrong_cluster_corr,
        args.wrong_cluster_center_corr,
        args.min_regret,
    )
    save_focus_outputs(audit_df, args.out_dir, windows, args.top_n)
    print(audit_df["miss_type"].value_counts().to_string())
    print("Saved audit outputs to", args.out_dir)


if __name__ == "__main__":
    main()
