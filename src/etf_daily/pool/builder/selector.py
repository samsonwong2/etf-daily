"""High-level :func:`select_pool` pipeline.

Mirrors ``2_filter/0.2_cluster_select_from_dendrogram.py::main``. Writes:

- ``cluster_mapping.csv`` (full mapping: identity + fund_list extras)
- ``cluster_mapping_selected.csv`` (selected rows only; or dated auto path)
- ``cluster_mapping_selected_metadata.json``
- ``dendrogram_selected_reps.png`` / ``.svg``
"""
from __future__ import annotations

from datetime import datetime
import re

import pandas as pd

from . import constants as C
from .clustering import (
    cluster_and_select,
    pairwise_overlap_matrices,
    prepare_recent_window,
    summarize_overlap_counts,
)
from .code_utils import build_normalized_code_set, code_in_normalized_set, load_manual_keep_codes
from .config import load_shared_config, write_json
from .data_loading import build_code_name_map, load_data, load_fund_list_universe
from .dendrogram import plot_selected_dendrogram
from .export import export_mapping


_SEED_CODE_PATTERN = re.compile(r"^(?:SH|SZ)?\d{6}$", re.IGNORECASE)


def _resolve_current_cluster_drop_codes(
    close: pd.DataFrame,
    vol: pd.DataFrame | None,
    cluster_seed_codes: list[str] | tuple[str, ...] | None,
    linkage_method: str = "ward",
) -> set[str]:
    """Resolve cluster-seed exclusions from the current run's own clustering universe.

    This removes the old one-run lag where ``ALWAYS_DROP_CLUSTER_CODES`` had to
    look up seed clusters from the previous ``cluster_mapping.csv`` output.
    """
    seed_codes = [
        str(code).strip().upper()
        for code in cluster_seed_codes or []
        if str(code).strip() and _SEED_CODE_PATTERN.match(str(code).strip())
    ]
    if not seed_codes or close.empty:
        return set()

    _, recent_vol, returns = prepare_recent_window(close, vol, C.CLUSTER_LOOKBACK_DAYS)
    recent_valid_days = returns.notna().sum().astype(int)
    eligible_codes = recent_valid_days[recent_valid_days >= C.MIN_CLUSTER_HISTORY_DAYS].index.tolist()
    if not eligible_codes:
        print("[cluster_seed_excludes] skipped because no ETF meets the clustering history requirement.")
        return set()

    returns = returns[eligible_codes]
    cluster_vol = recent_vol[eligible_codes] if recent_vol is not None else None
    corr, overlap, _ = pairwise_overlap_matrices(returns, C.MIN_PAIR_OVERLAP_DAYS, C.LOW_OVERLAP_DISTANCE)
    _, clusters, _, _, _ = cluster_and_select(
        returns,
        cluster_vol,
        t=C.DIST_T,
        per_cluster=C.PER_CLUSTER,
        rule=C.RULE,
        target_count=C.TARGET_SELECTED_COUNT,
        history_days=recent_valid_days.reindex(eligible_codes).fillna(0),
        corr=corr,
        overlap=overlap,
        linkage_method=linkage_method,
    )

    normalized_seed_codes = build_normalized_code_set(seed_codes)
    matched_seed_codes = [code for code in clusters.index if code_in_normalized_set(code, normalized_seed_codes)]
    if not matched_seed_codes:
        print("[cluster_seed_excludes] no configured seed codes were present in the current clustering universe.")
        return set()

    matched_cluster_ids = set(clusters.reindex(matched_seed_codes).dropna().tolist())
    cluster_drop_codes = set(clusters[clusters.isin(matched_cluster_ids)].index.tolist())
    normalized_matched_seed_codes = build_normalized_code_set(matched_seed_codes)
    unresolved_seed_codes = [
        code for code in seed_codes if not code_in_normalized_set(code, normalized_matched_seed_codes)
    ]

    print(
        "[cluster_seed_excludes] matched "
        f"{len(matched_seed_codes)}/{len(seed_codes)} seeds across {len(matched_cluster_ids)} current clusters; "
        f"dropping {len(cluster_drop_codes)} codes in the same run."
    )
    if unresolved_seed_codes:
        preview = ", ".join(unresolved_seed_codes[:10])
        suffix = " ..." if len(unresolved_seed_codes) > 10 else ""
        print(f"[cluster_seed_excludes] unresolved seeds outside the current clustering universe: {preview}{suffix}")
    return cluster_drop_codes


def _build_diagnostic_view(
    close: pd.DataFrame,
    vol: pd.DataFrame | None,
) -> dict[str, object]:
    """Build row-level diagnostics and clustering inputs for a given universe."""
    _, recent_vol, returns_raw = prepare_recent_window(close, vol, C.CLUSTER_LOOKBACK_DAYS)
    recent_valid_days = returns_raw.notna().sum().astype(int)
    available_history_days = close.notna().sum().astype(int)
    eligible_for_cluster = recent_valid_days >= C.MIN_CLUSTER_HISTORY_DAYS
    eligible_for_selection = recent_valid_days >= C.MIN_SELECTION_HISTORY_DAYS
    diagnostics_df = pd.DataFrame(
        {
            "code": close.columns,
            "available_history_days": close.columns.map(lambda c: int(available_history_days.get(c, 0))),
            "recent_valid_days": close.columns.map(lambda c: int(recent_valid_days.get(c, 0))),
            "eligible_for_cluster": close.columns.map(lambda c: bool(eligible_for_cluster.get(c, False))),
            "eligible_for_selection": close.columns.map(lambda c: bool(eligible_for_selection.get(c, False))),
        }
    )
    eligible_codes = diagnostics_df.loc[diagnostics_df["eligible_for_cluster"], "code"].tolist()
    overlap_columns = ["overlap_min_days", "overlap_median_days", "overlap_max_days"]

    if not eligible_codes:
        for col in overlap_columns:
            diagnostics_df[col] = 0
        empty_returns = returns_raw.iloc[:, 0:0]
        empty_vol = recent_vol.iloc[:, 0:0] if recent_vol is not None else None
        return {
            "diagnostics_df": diagnostics_df,
            "eligible_codes": eligible_codes,
            "returns": empty_returns,
            "cluster_vol": empty_vol,
            "recent_valid_days": recent_valid_days,
            "corr": None,
            "overlap": None,
        }

    returns = returns_raw[eligible_codes]
    cluster_vol = recent_vol[eligible_codes] if recent_vol is not None else None
    corr, overlap, _ = pairwise_overlap_matrices(returns, C.MIN_PAIR_OVERLAP_DAYS, C.LOW_OVERLAP_DISTANCE)
    overlap_summary = summarize_overlap_counts(overlap)
    diagnostics_df = diagnostics_df.merge(overlap_summary, on="code", how="left")
    diagnostics_df[overlap_columns] = diagnostics_df[overlap_columns].fillna(0)
    return {
        "diagnostics_df": diagnostics_df,
        "eligible_codes": eligible_codes,
        "returns": returns,
        "cluster_vol": cluster_vol,
        "recent_valid_days": recent_valid_days,
        "corr": corr,
        "overlap": overlap,
    }


def select_pool(
    *,
    shared_config_path: str | None = None,
    inception_cutoff: str | None = None,
    auto_selected_suffix: str | None = None,
) -> dict:
    """Run the full selection pipeline and return a metadata dict."""
    cfg = load_shared_config(shared_config_path)
    provider_uri = cfg["provider_uri"]
    test_period = tuple(cfg["test_period"])
    fund_list_csv = cfg["fund_list_csv"]

    if inception_cutoff is None:
        inception_cutoff = test_period[1]  # match legacy default

    effective_manual_excludes = list(dict.fromkeys(list(C.ALWAYS_DROP_CODES)))
    fund_universe_df, full_code_map = load_fund_list_universe(fund_list_csv)
    manual_keep_codes = load_manual_keep_codes(C.MANUAL_KEEP_CODES_FILE)
    force_keep_codes = list(dict.fromkeys(C.ALWAYS_KEEP_CODES + manual_keep_codes))
    if manual_keep_codes:
        print(f"Loaded {len(manual_keep_codes)} manual keep ETF codes from {C.MANUAL_KEEP_CODES_FILE}")

    _, full_universe_excluded_codes, full_universe_codes = build_code_name_map(
        fund_list_csv,
        C.EXCLUDE_TYPES,
        inception_cutoff,
        force_keep_codes=force_keep_codes,
    )
    full_close, full_vol = load_data(
        full_universe_codes,
        provider_uri=provider_uri,
        test_period=test_period,
        excluded_codes=full_universe_excluded_codes,
    )
    full_loaded_codes = set(full_close.columns.tolist())
    full_diag_view = _build_diagnostic_view(full_close, full_vol)
    full_clusters = None
    full_cluster_df = None
    if full_diag_view["eligible_codes"]:
        _, full_clusters, full_cluster_df, _, _ = cluster_and_select(
            full_diag_view["returns"],
            full_diag_view["cluster_vol"],
            t=C.DIST_T,
            per_cluster=C.PER_CLUSTER,
            rule=C.RULE,
            target_count=C.TARGET_SELECTED_COUNT,
            history_days=full_diag_view["recent_valid_days"].reindex(full_diag_view["eligible_codes"]).fillna(0),
            corr=full_diag_view["corr"],
            overlap=full_diag_view["overlap"],
        )

    code_map, excluded_codes, selected_codes = build_code_name_map(
        fund_list_csv,
        C.EXCLUDE_TYPES,
        inception_cutoff,
        manual_excludes=effective_manual_excludes,
        force_keep_codes=force_keep_codes,
    )

    close, vol = load_data(
        selected_codes,
        provider_uri=provider_uri,
        test_period=test_period,
        excluded_codes=excluded_codes.union(effective_manual_excludes),
    )
    loaded_codes_before_cluster_drop = set(close.columns.tolist())

    current_cluster_drop_codes = _resolve_current_cluster_drop_codes(
        close,
        vol,
        C.ALWAYS_DROP_CLUSTER_CODES,
    )
    if current_cluster_drop_codes:
        normalized_cluster_drops = build_normalized_code_set(current_cluster_drop_codes)
        keep_cols = [code for code in close.columns if not code_in_normalized_set(code, normalized_cluster_drops)]
        dropped_cols = [code for code in close.columns if code_in_normalized_set(code, normalized_cluster_drops)]
        if dropped_cols:
            close = close[keep_cols]
            vol = vol[keep_cols] if vol is not None else None
            excluded_codes = excluded_codes.union(dropped_cols)

    missing_provider_codes = sorted(set(full_universe_codes) - full_loaded_codes)

    diagnostics_reason_map: dict[str, str] = {}
    normalized_manual_excludes = build_normalized_code_set(effective_manual_excludes)
    normalized_cluster_drops = build_normalized_code_set(current_cluster_drop_codes)
    exclude_type_set = {str(t).strip() for t in (C.EXCLUDE_TYPES or set())}
    normalized_force_keep = build_normalized_code_set(force_keep_codes)
    cutoff_ts = pd.to_datetime(inception_cutoff, errors="coerce") if inception_cutoff is not None else pd.NaT
    for row in fund_universe_df.itertuples(index=False):
        code = str(row.code).strip()
        if not code:
            continue
        reasons: list[str] = []
        fund_type = str(getattr(row, "fund_type", "")).strip()
        inception_date = pd.to_datetime(str(getattr(row, "inception_date", "")).strip(), errors="coerce")
        is_force_keep = code_in_normalized_set(code, normalized_force_keep)
        if fund_type and fund_type in exclude_type_set and not is_force_keep:
            reasons.append("excluded_type")
        if (
            cutoff_ts is not pd.NaT
            and pd.notna(inception_date)
            and inception_date > cutoff_ts
            and not is_force_keep
        ):
            reasons.append("inception_after_cutoff")
        if code_in_normalized_set(code, normalized_manual_excludes):
            reasons.append("always_drop_code")
        if code_in_normalized_set(code, normalized_cluster_drops):
            reasons.append("always_drop_cluster_code")
        if not reasons and code in missing_provider_codes:
            reasons.append("missing_from_qlib_data")
        if reasons:
            diagnostics_reason_map[code] = "|".join(dict.fromkeys(reasons))

    diag_view = _build_diagnostic_view(close, vol)
    diagnostics_df = diag_view["diagnostics_df"]
    eligible_codes = diag_view["eligible_codes"]
    if not eligible_codes:
        raise RuntimeError("No ETF meets the minimum recent-history requirement for clustering")
    returns = diag_view["returns"]
    cluster_vol = diag_view["cluster_vol"]
    recent_valid_days = diag_view["recent_valid_days"]
    corr = diag_view["corr"]
    overlap = diag_view["overlap"]
    reasons = {
        code: "insufficient_history_for_clustering"
        for code in diagnostics_df.loc[~diagnostics_df["eligible_for_cluster"], "code"].tolist()
    }
    for code, reason in diagnostics_reason_map.items():
        reasons.setdefault(code, reason)
    print("Returns shape", returns.shape)

    Z, clusters, cluster_df, selected, cluster_reasons = cluster_and_select(
        returns,
        cluster_vol,
        t=C.DIST_T,
        per_cluster=C.PER_CLUSTER,
        rule=C.RULE,
        force_keep=force_keep_codes,
        target_count=C.TARGET_SELECTED_COUNT,
        history_days=recent_valid_days.reindex(eligible_codes).fillna(0),
        corr=corr,
        overlap=overlap,
        family_clusters=full_clusters,
    )
    reasons.update(cluster_reasons)
    full_diagnostics_df = fund_universe_df[["code"]].copy()
    full_diagnostics_df = full_diagnostics_df.merge(full_diag_view["diagnostics_df"], on="code", how="left")
    int_fill_columns = ["available_history_days", "recent_valid_days", "overlap_min_days", "overlap_max_days"]
    float_fill_columns = ["overlap_median_days"]
    bool_fill_columns = ["eligible_for_cluster", "eligible_for_selection"]
    for col in int_fill_columns:
        if col in full_diagnostics_df.columns:
            full_diagnostics_df[col] = pd.to_numeric(full_diagnostics_df[col], errors="coerce").fillna(0).astype(int)
    for col in float_fill_columns:
        if col in full_diagnostics_df.columns:
            full_diagnostics_df[col] = pd.to_numeric(full_diagnostics_df[col], errors="coerce").fillna(0.0)
    for col in bool_fill_columns:
        if col in full_diagnostics_df.columns:
            full_diagnostics_df[col] = full_diagnostics_df[col].eq(True)
    if auto_selected_suffix and str(auto_selected_suffix).strip():
        mainline_selected_csv = C.dated_selected_csv_path(str(auto_selected_suffix).strip())
        print(
            f"[INFO] Writing machine-selected pool to {mainline_selected_csv} "
            f"(hand-maintained {C.CSV_SELECTED_OUT} is not overwritten)."
        )
    else:
        mainline_selected_csv = C.CSV_SELECTED_OUT
    export_mapping(
        fund_universe_df["code"].tolist(),
        full_clusters,
        selected,
        reasons,
        full_cluster_df,
        full_code_map,
        diagnostics_df=full_diagnostics_df,
        out_csv=C.CSV_OUT,
        selected_csv=mainline_selected_csv,
        selection_clusters=clusters,
        selection_cluster_df=cluster_df,
        fund_list_csv=fund_list_csv,
    )

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "production_rule": C.RULE,
        "production_scheme": C.PRODUCTION_MULTI_REP_MAINLINE_SCHEME,
        "benchmark_rule": None,
        "benchmark_scheme": None,
        "deprecated_schemes": list(C.DEPRECATED_MULTI_REP_WEIGHT_SCHEMES),
        "dist_t": float(C.DIST_T),
        "per_cluster": int(C.PER_CLUSTER),
        "target_selected_count": int(C.TARGET_SELECTED_COUNT) if C.TARGET_SELECTED_COUNT is not None else None,
        "cluster_lookback_days": int(C.CLUSTER_LOOKBACK_DAYS),
        "min_cluster_history_days": int(C.MIN_CLUSTER_HISTORY_DAYS),
        "min_selection_history_days": int(C.MIN_SELECTION_HISTORY_DAYS),
        "min_pair_overlap_days": int(C.MIN_PAIR_OVERLAP_DAYS),
        "mainline_selected_csv": mainline_selected_csv,
        "benchmark_selected_csv": None,
        "mainline_selected_count": int(len(selected)),
        "benchmark_selected_count": 0,
    }
    write_json(C.METADATA_OUT, metadata)
    print("Wrote generation metadata to", C.METADATA_OUT)

    # 谱系图叶子顺序与聚类使用的标的集一致。
    codes = list(clusters.index)
    plot_selected_dendrogram(Z, codes, set(selected), code_map, C.IMG_OUT, C.SVG_OUT)
    return metadata


if __name__ == "__main__":
    select_pool()
