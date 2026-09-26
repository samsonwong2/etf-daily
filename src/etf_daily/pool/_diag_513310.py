#!/usr/bin/env python3
"""Diagnose why SH513310 was not selected in latest cluster pool run."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from etf_daily.pool.builder import constants as C
from etf_daily.pool.builder.clustering import (
    _build_multi_rep_cluster_pool,
    prepare_recent_window,
    pairwise_overlap_matrices,
    summarize_overlap_counts,
)
from etf_daily.pool.builder.data_loading import load_fund_universe, load_close_volume
from etf_daily.pool.builder.selector import _resolve_current_cluster_drop_codes, _build_diagnostic_view
from etf_daily.pool.builder.utils import build_normalized_code_set, code_in_normalized_set

TARGET = "SH513310"


def main() -> None:
    import json

    cfg_path = os.path.join(os.path.dirname(__file__), "shared_filter_config.json")
    cfg = json.loads(open(cfg_path, encoding="utf-8").read())
    fund_df = load_fund_universe(cfg["fund_list_csv"], C.EXCLUDE_TYPES, cfg["test_period"][1])
    codes = fund_df["code"].tolist()
    close, vol = load_close_volume(cfg["provider_uri"], cfg["market"], codes)
    drop_cluster = _resolve_current_cluster_drop_codes(close, vol, C.ALWAYS_DROP_CLUSTER_CODES)
    keep_cols = [
        c for c in close.columns
        if not code_in_normalized_set(c, build_normalized_code_set(C.ALWAYS_DROP_CODES))
        and not code_in_normalized_set(c, build_normalized_code_set(drop_cluster))
    ]
    close = close[keep_cols]
    vol = vol[keep_cols] if vol is not None else None

    diag = _build_diagnostic_view(close, vol)
    eligible = diag["eligible_codes"]
    returns = diag["returns"]
    corr = diag["corr"]
    history = diag["recent_valid_days"].reindex(eligible).fillna(0)

    force_keep = list(dict.fromkeys(C.ALWAYS_KEEP_CODES))
    norm_target = TARGET.upper()

    in_universe = any(code_in_normalized_set(c, {norm_target}) for c in returns.columns)
    actual_code = next((c for c in returns.columns if code_in_normalized_set(c, {norm_target})), None)
    print("in_universe", in_universe, "code", actual_code)
    print("force_keep", force_keep)
    print("in_ALWAYS_DROP", norm_target in [x.upper() for x in C.ALWAYS_DROP_CODES])
    print("in_ALWAYS_KEEP", norm_target in [x.upper() for x in C.ALWAYS_KEEP_CODES])

    # cluster id from saved mapping
    cm = pd.read_csv("~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/cluster_mapping.csv")
    row = cm[cm["code"].str.upper().str.replace(r"[^A-Z0-9]", "", regex=True) == norm_target]
    if not row.empty:
        print("saved mapping:", row[["name", "cluster", "selected", "reason", "mean_return", "n"]].to_string(index=False))

    # rebuild cluster labels quickly
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    dist = 1.0 - corr
    Z = linkage(squareform(dist.values, checks=False), method="ward")
    labels = fcluster(Z, t=C.DIST_T, criterion="distance")
    clusters = pd.Series(labels, index=returns.columns)
    cid = clusters.get(actual_code)
    print("cluster_id", cid, "cluster_size", int((clusters == cid).sum()))

    members = clusters[clusters == cid].index.tolist()
    print("cluster_members", members)

    # corr to force_keep and selected pool
    selected_df = pd.read_csv("~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/cluster_mapping_selected.csv")
    selected_codes = selected_df["code"].tolist()
    if actual_code and actual_code in corr.index:
        corrs = []
        for sc in selected_codes:
            sc_actual = next((c for c in corr.columns if c.upper().replace("SZ", "SZ").replace("SH", "SH") == sc.upper().replace("SH", "SH")), sc)
            if sc in corr.columns:
                corrs.append((sc, float(abs(corr.loc[actual_code, sc]))))
        corrs.sort(key=lambda x: -x[1])
        print("\nTop corr with selected pool:")
        for sc, v in corrs[:15]:
            flag = " **>0.70**" if v > 0.70 else (" *intra>0.60*" if v > 0.60 else "")
            print(f"  {sc}: {v:.4f}{flag}")

        print("\nCorr with ALWAYS_KEEP:")
        for fk in force_keep:
            if fk in corr.columns:
                print(f"  {fk}: {abs(corr.loc[actual_code, fk]):.4f}")

    # simulate selection rank
    cluster_rows = []
    for cluster_id, group in clusters.groupby(clusters):
        ms = list(group.index)
        cluster_rows.append({"cluster": cluster_id, "members": ms, "n": len(ms)})
    cluster_df = pd.DataFrame(cluster_rows)

    near_idx = int(os.environ.get("POOL_NEAR_CLONE_IDX", "0"))
    near_limit = C.FINAL_MAX_ABS_CORR_THRESHOLDS[near_idx]

    selected, reasons = _build_multi_rep_cluster_pool(
        cluster_df,
        returns,
        history,
        target_count=C.TARGET_SELECTED_COUNT,
        force_keep_codes=force_keep,
        weight_scheme="equal",
        corr=corr,
        near_clone_corr_limit=near_limit,
        intra_cluster_corr_limit=C.INTRA_CLUSTER_MAX_ABS_CORR,
    )
    print("\nnear_clone_limit", near_limit)
    print("final_selected_count", len(selected))
    print("513310 in selected?", actual_code in selected if actual_code else False)
    if actual_code in reasons:
        print("reason if selected:", reasons[actual_code])

    # check rank among rank-1 reps before truncation
    from etf_daily.pool.builder.clustering import _return_scores, _build_multi_rep_candidate_caps, MULTI_REP_EXP_BASE, _multi_rep_rank_weight
    import numpy as np

    work_df = cluster_df.copy()
    work_df["candidate_members"] = work_df["members"].apply(
        lambda members: [code for code in members if float(history.get(code, 0.0)) >= C.MIN_SELECTION_HISTORY_DAYS] or list(members)
    )
    cap_df = _build_multi_rep_candidate_caps(work_df, C.TARGET_SELECTED_COUNT)
    cap_map = dict(zip(cap_df["cluster"], cap_df["candidate_cap"]))
    global_scores = _return_scores(list(returns.columns), returns)
    if actual_code:
        score = float(global_scores.get(actual_code, 0.0))
        rank = int((global_scores > score).sum() + 1)
        print(f"\nreturn_score={score:.6f}, global_rank={rank}/{len(global_scores)}")
        print(f"cluster candidate_cap={cap_map.get(cid, 0)}")


if __name__ == "__main__":
    main()
