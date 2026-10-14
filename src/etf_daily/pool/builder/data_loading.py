"""qlib initialization + fund_list CSV loader + close/volume panel loader."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from etf_daily.pipeline.price_adjustments import auto_qfq_adjust_close_panel, log_qfq_adjustments

try:
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D
except Exception as exc:  # pragma: no cover
    qlib = None  # type: ignore[assignment]
    D = None  # type: ignore[assignment]
    REG_CN = None  # type: ignore[assignment]
    print("Warning: qlib import failed:", exc)

from .code_utils import (
    build_normalized_code_set,
    code_in_normalized_set,
)


_QLIB_PROVIDER_URI: str | None = None


def should_reinit_qlib(active_uri: str | None, provider_uri: str) -> bool:
    """True when this process has not yet initialized qlib on ``provider_uri``."""
    return (active_uri or "") != str(provider_uri)


def panel_has_codes(loaded_columns: Iterable[str], requested: Iterable[str]) -> bool:
    """True when every non-empty requested code is already a loaded column."""
    loaded = {str(col) for col in loaded_columns}
    requested_codes = [str(code).strip() for code in requested if str(code).strip()]
    return all(code in loaded for code in requested_codes)


def filter_stockpool(
    instruments: Iterable[str],
    excluded_codes: Iterable[str] | None = None,
) -> list[str]:
    """Same instrument filter ``load_data`` applies before ``D.features``."""
    stockpool = [str(code).strip() for code in instruments if str(code).strip()]
    if excluded_codes:
        normalized_excluded = build_normalized_code_set(excluded_codes)
        stockpool = [code for code in stockpool if not code_in_normalized_set(code, normalized_excluded)]
    return stockpool


def slice_loaded_codes(
    loaded_columns: Iterable[str],
    requested: Iterable[str],
    first_requested: Iterable[str],
) -> list[str] | None:
    """Columns to reuse from an already loaded panel, in loaded order.

    Returns None when a requested code was never part of the first request, so a
    second qlib read might still find it. Codes that were requested the first
    time but missing from the panel were omitted by qlib and stay omitted.
    """
    loaded_list = [str(col) for col in loaded_columns]
    loaded = set(loaded_list)
    first = {str(code).strip() for code in first_requested if str(code).strip()}
    req = [str(code).strip() for code in requested if str(code).strip()]
    for code in req:
        if code not in loaded and code not in first:
            return None
    requested_set = set(req)
    return [col for col in loaded_list if col in requested_set]


def _ensure_qlib(provider_uri: str) -> None:
    """Initialize qlib once per provider URI in this process."""
    global _QLIB_PROVIDER_URI
    if qlib is None or D is None:
        raise RuntimeError("qlib is not available; cannot load data.")
    if not should_reinit_qlib(_QLIB_PROVIDER_URI, provider_uri):
        print("Reusing qlib provider:", provider_uri)
        return
    print("Initializing qlib provider:", provider_uri)
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    _QLIB_PROVIDER_URI = str(provider_uri)


POSSIBLE_TYPE_NAMES = ["基金类型", "类型", "fund_type", "type"]
POSSIBLE_INCEPTION_NAMES = ["成立日期", "inception_date", "上市日期", "成立时间"]
POSSIBLE_CODE_NAMES = ["代码", "code", "fund_code", "基金代码", "证券代码", "代码ID"]
POSSIBLE_NAME_NAMES = ["基金简称", "简称", "名称", "name", "fund_name", "基金名称"]


def _resolve_fund_list_columns(df: pd.DataFrame) -> tuple[str, str, str | None, str | None]:
    code_col = next((c for c in POSSIBLE_CODE_NAMES if c in df.columns), None)
    name_col = next((c for c in POSSIBLE_NAME_NAMES if c in df.columns), None)
    if code_col is None or name_col is None:
        code_col = code_col or df.columns[0]
        name_col = name_col or (df.columns[1] if len(df.columns) > 1 else df.columns[0])
    type_col = next((c for c in POSSIBLE_TYPE_NAMES if c in df.columns), None)
    inception_col = next((c for c in POSSIBLE_INCEPTION_NAMES if c in df.columns), None)
    return code_col, name_col, type_col, inception_col


def _build_augmented_code_name_map(base_map: dict[str, str]) -> dict[str, str]:
    aug: dict[str, str] = {}
    for k, v in base_map.items():
        ks = str(k).strip()
        aug[ks] = v
        aug[ks.upper()] = v
        if ks.upper().startswith("SH") or ks.upper().startswith("SZ"):
            aug[ks[2:]] = v
        try:
            aug[str(int(ks))] = v
        except Exception:
            pass
    return aug


def load_fund_list_universe(csv_path: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load the full fund list universe without applying any exclusions."""
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    code_col, name_col, type_col, inception_col = _resolve_fund_list_columns(df)
    universe = pd.DataFrame(
        {
            "code": df[code_col].astype(str).str.strip(),
            "name": df[name_col].astype(str).str.strip(),
        }
    )
    if type_col is not None:
        universe["fund_type"] = df[type_col].astype(str).str.strip()
    else:
        universe["fund_type"] = ""
    if inception_col is not None:
        universe["inception_date"] = df[inception_col].astype(str).str.strip()
    else:
        universe["inception_date"] = ""
    universe = universe[universe["code"] != ""].drop_duplicates(subset="code", keep="first").reset_index(drop=True)
    code_name_map = _build_augmented_code_name_map(dict(zip(universe["code"], universe["name"])))
    return universe, code_name_map


def _panels_from_features(raw: pd.DataFrame, excluded_codes: Iterable[str] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = raw["$close"].unstack(level="instrument")
    vol = raw["$volume"].unstack(level="instrument")
    close.index.name = "Date"
    vol.index.name = "Date"
    close = close.sort_index().ffill()
    vol = vol.sort_index()
    if excluded_codes:
        normalized_excluded = build_normalized_code_set(excluded_codes)
        keep_cols = [c for c in close.columns if not code_in_normalized_set(c, normalized_excluded)]
        dropped = set(close.columns) - set(keep_cols)
        if dropped:
            print(f"Removed {len(dropped)} codes due to exclude-types rule.")
            close = close[keep_cols]
            vol = vol[keep_cols]
    return close, vol


def load_raw_close_volume(
    instruments: Iterable[str],
    provider_uri: str,
    test_period: tuple[str, str],
    excluded_codes: Iterable[str] | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull unadjusted ``$close`` and ``$volume``. ``end_date`` overrides ``test_period[1]``."""
    _ensure_qlib(provider_uri)
    stockpool = filter_stockpool(instruments, excluded_codes)
    raw = D.features(
        stockpool,
        fields=["$close", "$volume"],
        start_time=test_period[0],
        end_time=end_date or test_period[1],
    )
    close, vol = _panels_from_features(raw, excluded_codes)
    print("Loaded raw close shape", close.shape, "volume shape", vol.shape)
    return close, vol


def load_data(
    instruments: Iterable[str],
    provider_uri: str,
    test_period: tuple[str, str],
    excluded_codes: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull ``$close`` and ``$volume`` from qlib for ``instruments``.

    Mirrors ``2_filter/0.2_cluster_select_from_dendrogram.py::load_data``.
    Qfq is applied to the requested ``test_period`` only.
    """
    close, vol = load_raw_close_volume(
        instruments,
        provider_uri=provider_uri,
        test_period=test_period,
        excluded_codes=excluded_codes,
    )
    close, qfq_records = auto_qfq_adjust_close_panel(close)
    if not qfq_records.empty:
        log_qfq_adjustments(qfq_records, stage="pool_builder.load_close_volume_from_qlib")
    print("Loaded close shape", close.shape, "volume shape", vol.shape)
    return close, vol


def build_code_name_map(
    csv_path: str,
    exclude_types: Iterable[str] | None = None,
    inception_cutoff: str | None = None,
    manual_excludes: Iterable[str] | None = None,
    cluster_excludes: Iterable[str] | None = None,
    cluster_reference_map_csv: str | None = None,
    force_keep_codes: Iterable[str] | None = None,
) -> tuple[dict[str, str], set[str], list[str]]:
    """Read ``fund_list.csv``, apply type/inception/manual/cluster exclusions.

    ``force_keep_codes`` (ALWAYS_KEEP / manual whitelist) are exempt from
    ``exclude_types`` and inception-cutoff filters so bond/gold anchors can
    survive ``EXCLUDE_TYPES`` such as ``指数型-固收``. Manual/cluster drops
    still apply (blacklist wins over whitelist).

    Returns ``(augmented_code_name_map, excluded_codes, selected_codes)``.
    """
    from .code_utils import load_cluster_seed_ids  # local import to avoid cycle

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    code_col, name_col, type_col, inception_col = _resolve_fund_list_columns(df)
    code_series = df[code_col].astype(str).str.strip()
    name_series = df[name_col].astype(str).str.strip()
    force_keep_set = build_normalized_code_set(force_keep_codes or [])

    def _is_force_keep(code: object) -> bool:
        return bool(force_keep_set) and code_in_normalized_set(str(code), force_keep_set)

    excluded_codes: set[str] = set()
    if exclude_types:
        if type_col:
            type_series = df[type_col].astype(str).str.strip()
            mask = type_series.isin({t.strip() for t in exclude_types})
            if force_keep_set and mask.any():
                mask = mask & ~code_series.map(_is_force_keep)
            if mask.any():
                excluded_codes = set(code_series[mask])
                code_series = code_series[~mask]
                name_series = name_series[~mask]

    recent_cutoff_codes: set[str] = set()
    if inception_col and inception_cutoff is not None:
        # Align inception mask to current code_series index (after type filter).
        inception_all = pd.to_datetime(df[inception_col], errors="coerce")
        cutoff = pd.to_datetime(inception_cutoff, errors="coerce")
        if cutoff is not pd.NaT:
            inception_series = inception_all.reindex(code_series.index)
            recent_mask = inception_series > cutoff
            if force_keep_set and recent_mask.any():
                recent_mask = recent_mask & ~code_series.map(_is_force_keep)
            if recent_mask.any():
                recent_cutoff_codes = set(code_series[recent_mask])
                code_series = code_series[~recent_mask]
                name_series = name_series[~recent_mask]

    manual_excludes = list(manual_excludes or [])
    if manual_excludes:
        normalized_manual = build_normalized_code_set(manual_excludes)
        manual_mask = code_series.apply(lambda c: code_in_normalized_set(c, normalized_manual))
        if manual_mask.any():
            excluded_codes.update(code_series[manual_mask])
            code_series = code_series[~manual_mask]
            name_series = name_series[~manual_mask]

    cluster_excludes = list(cluster_excludes or [])
    if cluster_excludes:
        cluster_seed_ids = load_cluster_seed_ids(cluster_reference_map_csv, cluster_excludes)
        if cluster_seed_ids:
            cluster_series = pd.Series("", index=code_series.index, dtype=str)
            if cluster_reference_map_csv and os.path.exists(cluster_reference_map_csv):
                ref_df = pd.read_csv(cluster_reference_map_csv, dtype=str).fillna("")
                if "code" in ref_df.columns and "cluster" in ref_df.columns:
                    ref_df["code"] = ref_df["code"].astype(str).str.strip()
                    ref_df["cluster"] = ref_df["cluster"].astype(str).str.strip()
                    ref_map = dict(zip(ref_df["code"], ref_df["cluster"]))
                    cluster_series = code_series.map(lambda c: ref_map.get(str(c).strip(), ""))
            cluster_mask = cluster_series.astype(str).isin(cluster_seed_ids)
            if cluster_mask.any():
                excluded_codes.update(code_series[cluster_mask])
                code_series = code_series[~cluster_mask]
                name_series = name_series[~cluster_mask]

    base_map = dict(zip(code_series, name_series))
    aug = _build_augmented_code_name_map(base_map)
    selected_codes = code_series.astype(str).str.strip().tolist()
    return aug, excluded_codes.union(recent_cutoff_codes), list(dict.fromkeys(selected_codes))
