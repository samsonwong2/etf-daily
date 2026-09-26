"""Windowing, pair overlap, multi-rep pool, trimming and clustering.

Extracted verbatim from ``2_filter/0.2_cluster_select_from_dendrogram.py``.
Public entry point is :func:`cluster_and_select`.
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from .code_utils import build_normalized_code_set, code_in_normalized_set
from .constants import (
    CLUSTER_LOOKBACK_DAYS,
    DIST_T,
    FINAL_MAX_ABS_CORR_THRESHOLDS,
    INTRA_CLUSTER_MAX_ABS_CORR,
    LOW_OVERLAP_DISTANCE,
    MIN_PAIR_OVERLAP_DAYS,
    MIN_SELECTION_HISTORY_DAYS,
    MULTI_REP_EXP_BASE,
    PER_CLUSTER,
    RULE,
)


def _rank_score(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    series = series.astype(float)
    if series.empty:
        return series
    return series.rank(method="average", pct=True)


def _return_scores(codes: Iterable[str], returns: pd.DataFrame) -> pd.Series:
    """Score candidate ETFs by blend of annualized Sharpe and 12-1 momentum.

    Legacy behavior (rank of simple mean return) can be restored by setting
    env ``POOL_SCORING_MODE=legacy_mean``. Default ``sharpe_mom`` blends:
      - annualized Sharpe on the full lookback window (252d typically)
      - 12-1 momentum: cumulative return over the last ~231 days skipping the
        most recent ~21 days (suppresses short-term reversal noise)
    Each factor is converted to a percentile rank (0..1) then averaged 50/50.
    """
    codes = list(codes)
    if not codes:
        return pd.Series(dtype=float)
    code_list = list(dict.fromkeys(codes))
    panel = returns[code_list]

    mode = os.environ.get("POOL_SCORING_MODE", "sharpe_mom").lower()
    if mode == "legacy_mean":
        mean_return = panel.mean().reindex(code_list).fillna(0.0)
        return _rank_score(mean_return).sort_values(ascending=False)

    # Annualized Sharpe over the full panel window.
    mean_daily = panel.mean()
    std_daily = panel.std(ddof=0).replace(0.0, np.nan)
    sharpe = (mean_daily / std_daily * np.sqrt(252.0)).reindex(code_list).fillna(0.0)

    # 12-1 momentum: cumulative return over t-252 ... t-21 (skip last month).
    n_rows = len(panel)
    skip_recent = min(21, max(0, n_rows - 1))
    momentum_window = panel.iloc[: n_rows - skip_recent] if skip_recent > 0 else panel
    momentum = (momentum_window + 1.0).prod() - 1.0
    momentum = momentum.reindex(code_list).fillna(0.0)

    score = 0.5 * _rank_score(sharpe) + 0.5 * _rank_score(momentum)
    return score.sort_values(ascending=False)


def _multi_rep_rank_weight(rank_index: int, scheme: str, exp_base: float = MULTI_REP_EXP_BASE) -> float:
    rank_number = int(rank_index) + 1
    if scheme == "equal":
        return 1.0
    if scheme == "harmonic":
        return 1.0 / float(rank_number)
    if scheme == "exp":
        return float(exp_base) ** float(rank_index)
    raise ValueError(f"Unsupported multi-rep weight scheme: {scheme}")


def _families_with_force_keep(
    force_keep_codes: Iterable[str] | None,
    family_clusters: pd.Series | None,
) -> set:
    """Diagnostic families (族) that already have an ALWAYS_KEEP anchor."""
    if family_clusters is None or family_clusters.empty or not force_keep_codes:
        return set()
    normalized_keep = build_normalized_code_set(force_keep_codes)
    families: set = set()
    for code, family_id in family_clusters.items():
        if pd.isna(family_id):
            continue
        if code_in_normalized_set(str(code), normalized_keep):
            families.add(family_id)
    return families


def _code_blocked_by_family_force_keep(
    code: str,
    family_clusters: pd.Series | None,
    families_with_force_keep: set,
    normalized_keep_for_cluster: set[str],
) -> bool:
    """True when another force_keep already covers this code's diagnostic family."""
    if code_in_normalized_set(code, normalized_keep_for_cluster):
        return False
    if not families_with_force_keep or family_clusters is None:
        return False
    family_id = family_clusters.get(code)
    if family_id is None or pd.isna(family_id):
        return False
    return family_id in families_with_force_keep


def _build_multi_rep_candidate_caps(cluster_df: pd.DataFrame, target_count: int) -> pd.DataFrame:
    quota_rows = []
    for _, row in cluster_df.iterrows():
        quota_rows.append(
            {
                "cluster": row["cluster"],
                "member_count": len(row["candidate_members"]),
                "top_return": float(row["candidate_top_return"]),
            }
        )
    quota_df = pd.DataFrame(quota_rows)
    total_members = int(quota_df["member_count"].sum()) if not quota_df.empty else 0
    if total_members <= 0:
        return pd.DataFrame(columns=["cluster", "member_count", "top_return", "raw_quota", "candidate_cap"])

    quota_df["raw_quota"] = quota_df["member_count"] / total_members * target_count
    quota_df["candidate_cap"] = np.ceil(quota_df["raw_quota"]).astype(int)
    quota_df["candidate_cap"] = quota_df[["candidate_cap", "member_count"]].min(axis=1)
    quota_df["candidate_cap"] = quota_df["candidate_cap"].clip(lower=1)

    target_candidate_count = int(target_count) + min(
        max(5, int(np.ceil(target_count * 0.15))),
        max(0, total_members - int(target_count)),
    )
    current_candidates = int(quota_df["candidate_cap"].sum())
    quota_df["remainder"] = quota_df["raw_quota"] - np.floor(quota_df["raw_quota"])
    while current_candidates < target_candidate_count:
        available = quota_df.loc[quota_df["candidate_cap"] < quota_df["member_count"]].copy()
        if available.empty:
            break
        available = available.sort_values(
            ["remainder", "member_count", "top_return", "cluster"],
            ascending=[False, False, False, True],
        )
        pick_index = available.index[0]
        quota_df.loc[pick_index, "candidate_cap"] += 1
        current_candidates += 1
    return quota_df


def _build_multi_rep_cluster_pool(
    cluster_df: pd.DataFrame,
    returns: pd.DataFrame,
    history_days: pd.Series,
    target_count: int | None,
    force_keep_codes: Iterable[str] | None = None,
    weight_scheme: str = "equal",
    exp_base: float = MULTI_REP_EXP_BASE,
    corr: pd.DataFrame | None = None,
    near_clone_corr_limit: float | None = None,
    intra_cluster_corr_limit: float | None = None,
    family_clusters: pd.Series | None = None,
) -> tuple[list[str], dict[str, str]]:
    if cluster_df is None or cluster_df.empty:
        return [], {}

    target_count = int(target_count or len(returns.columns))
    history_days = history_days if history_days is not None else pd.Series(dtype=float)
    work_df = cluster_df.copy()
    work_df["candidate_members"] = work_df["members"].apply(
        lambda members: [code for code in members if float(history_days.get(code, 0.0)) >= MIN_SELECTION_HISTORY_DAYS]
        or list(members)
    )
    work_df["candidate_top_return"] = work_df["candidate_members"].apply(
        lambda members: float(_return_scores(members, returns).iloc[0]) if members else -np.inf
    )

    cap_df = _build_multi_rep_candidate_caps(work_df, target_count)
    cap_map = dict(zip(cap_df["cluster"], cap_df["candidate_cap"]))
    raw_quota_map = dict(zip(cap_df["cluster"], cap_df["raw_quota"]))
    global_return_scores = _return_scores(list(returns.columns), returns)

    # Pre-compute force_keep set so that a force_keep ETF which is already a
    # cluster member "consumes" that cluster's quota instead of letting the
    # cluster emit an extra (near-clone) representative on top of it.
    normalized_keep_for_cluster = build_normalized_code_set(force_keep_codes or [])
    families_with_force_keep = _families_with_force_keep(force_keep_codes, family_clusters)
    family_suppressed_clusters: set = set()

    selected_rows = []
    for _, row in work_df.iterrows():
        members = row["candidate_members"]
        candidate_cap = int(cap_map.get(row["cluster"], 0))
        if candidate_cap <= 0 or not members:
            continue
        member_scores = global_return_scores.reindex(members).fillna(0.0).sort_values(ascending=False)
        # If a force_keep ETF is a member of this cluster, treat the cluster as
        # already "covered" by that force_keep and skip emitting any additional
        # auto representatives — this prevents near-clone pairs like
        # SH518880 (force_keep gold) + SH518600 (auto-picked gold) ending up
        # together in the final pool.
        forced_in_cluster = [
            code for code in member_scores.index
            if code_in_normalized_set(code, normalized_keep_for_cluster)
        ]
        if forced_in_cluster:
            picks = forced_in_cluster
        else:
            eligible_members = [
                code
                for code in member_scores.index
                if not _code_blocked_by_family_force_keep(
                    code,
                    family_clusters,
                    families_with_force_keep,
                    normalized_keep_for_cluster,
                )
            ]
            if not eligible_members:
                family_suppressed_clusters.add(row["cluster"])
                continue
            picks = member_scores.reindex(eligible_members).dropna().head(candidate_cap).index.tolist()
        for rank_index, code in enumerate(picks):
            selected_rows.append(
                {
                    "code": code,
                    "cluster": row["cluster"],
                    "cluster_quota": float(raw_quota_map.get(row["cluster"], 0.0)),
                    "within_cluster_rank": rank_index + 1,
                    "return_score": float(member_scores.get(code, 0.0)),
                    "rep_weight": _multi_rep_rank_weight(rank_index, weight_scheme, exp_base),
                    "short_history": float(history_days.get(code, 0.0)) < MIN_SELECTION_HISTORY_DAYS,
                }
            )

    selected_df = pd.DataFrame(selected_rows)
    if selected_df.empty:
        selected: list[str] = []
        selected_reason_map: dict[str, str] = {}
    else:
        selected_df["priority_score"] = selected_df["return_score"] * selected_df["rep_weight"]
        selected_df = selected_df.sort_values(
            [
                "priority_score",
                "return_score",
                "cluster_quota",
                "rep_weight",
                "cluster",
                "within_cluster_rank",
                "code",
            ],
            ascending=[False, False, False, False, True, True, True],
        )
        unique_selected_df = selected_df.drop_duplicates(subset=["code"], keep="first").copy()
        # Plan B: guaranteed-coverage ordering. Every cluster contributes its
        # rank-1 representative FIRST (ordered by that rep's return_score so
        # higher-momentum clusters still show up early in the priority list).
        # Remaining seats (rank-2 and beyond) are appended afterwards using
        # the original priority_score ordering. This prevents lonely 1-member
        # clusters (e.g. 豆粕 ETF) from being starved out by multi-rep
        # allocations in high-return mega-clusters.
        # A/B 实验支持：POOL_PLAN_B_COVERAGE=0 可关闭该重排以复现 Fix#4 行为。
        _plan_b_coverage = os.environ.get("POOL_PLAN_B_COVERAGE", "1") == "1"
        if _plan_b_coverage:
            rank1 = unique_selected_df[unique_selected_df["within_cluster_rank"] == 1].copy()
            rank1 = rank1.sort_values(
                ["return_score", "cluster_quota", "cluster", "code"],
                ascending=[False, False, True, True],
            )
            rankN = unique_selected_df[unique_selected_df["within_cluster_rank"] > 1]
            ordered_df = pd.concat([rank1, rankN], ignore_index=False)
        else:
            ordered_df = unique_selected_df
        selected = ordered_df["code"].tolist()
        selected_reason_map = {}
        for _, picked in ordered_df.iterrows():
            short_history_suffix = "_short_history" if bool(picked["short_history"]) else ""
            coverage_tag = "_coverage" if (_plan_b_coverage and int(picked["within_cluster_rank"]) == 1) else ""
            selected_reason_map[picked["code"]] = (
                f"multi_rep_{weight_scheme}_cluster_{picked['cluster']}_rank_{int(picked['within_cluster_rank'])}{coverage_tag}{short_history_suffix}"
            )

    normalized_keep = build_normalized_code_set(force_keep_codes or [])
    forced_codes: list[str] = []
    if force_keep_codes:
        for code in returns.columns:
            if code_in_normalized_set(code, normalized_keep):
                forced_codes.append(code)
    forced_codes = list(dict.fromkeys(forced_codes))

    non_forced_selected = [code for code in selected if code not in set(forced_codes)]
    seats_left = max(0, int(target_count) - len(forced_codes))
    final_selected = (forced_codes + non_forced_selected[:seats_left])[:target_count]

    dedup_reason_overrides: dict[str, str] = {}

    # Fix #4: final near-clone dedup pass. Drops any non-force ETF whose
    # |corr| with an already-kept ETF exceeds ``near_clone_corr_limit``
    # (lowest rank in the priority order wins). Force_keep ETFs are anchors
    # and are never dropped. Runs AFTER quota filling so that eliminated
    # slots don't get back-filled with further near-clones.
    if (
        near_clone_corr_limit is not None
        and corr is not None
        and not corr.empty
    ):
        forced_set = set(forced_codes)
        # Map cluster_id -> ordered candidate members (highest return first), so
        # that if a cluster's rank-1 rep is trimmed we can try its rank-2/3/...
        # instead of leaving the cluster with zero representatives (regression
        # observed e.g. 煤炭/能源 cluster 7 getting zeroed out when its rank-1
        # correlated >0.75 with an earlier-picked 油气 rep).
        cluster_to_candidates: dict[object, list[str]] = {}
        cluster_of_code: dict[str, object] = {}
        for _, row in work_df.iterrows():
            members = list(row["candidate_members"] or [])
            if not members:
                continue
            ordered = global_return_scores.reindex(members).fillna(0.0)
            ordered = ordered.sort_values(ascending=False).index.tolist()
            cluster_to_candidates[row["cluster"]] = ordered
            for m in ordered:
                cluster_of_code[m] = row["cluster"]

        def _passes_near_clone(code: str, kept: list[str]) -> bool:
            if code not in corr.index:
                return True
            already = [c for c in kept if c in corr.columns]
            if not already:
                return True
            # Cross-cluster threshold (looser, e.g. 0.75).
            cross_limit = float(near_clone_corr_limit)
            # Intra-cluster threshold (stricter, e.g. 0.60) — same cluster means
            # same macro theme, so require stronger divergence to keep multiples.
            intra_limit = (
                float(intra_cluster_corr_limit)
                if intra_cluster_corr_limit is not None
                else cross_limit
            )
            code_cluster = cluster_of_code.get(code)
            for peer in already:
                peer_cluster = cluster_of_code.get(peer)
                limit = intra_limit if (
                    code_cluster is not None and peer_cluster == code_cluster
                ) else cross_limit
                val = float(abs(corr.loc[code, peer]))
                if val > limit:
                    return False
            return True

        dedup: list[str] = []
        clusters_with_rep: set[object] = set()
        dedup_reason_overrides: dict[str, str] = {}
        for code in final_selected:
            if code in forced_set:
                dedup.append(code)
                clusters_with_rep.add(cluster_of_code.get(code))
                continue
            if _passes_near_clone(code, dedup):
                dedup.append(code)
                clusters_with_rep.add(cluster_of_code.get(code))

        # Coverage rescue: for any cluster that lost its rep to near-clone trim,
        # try lower-ranked members until one passes the corr test. This keeps
        # Plan B's "≥1 per eligible cluster" guarantee intact when possible.
        for cluster_id, ordered_members in cluster_to_candidates.items():
            if cluster_id in clusters_with_rep:
                continue
            if cluster_id in family_suppressed_clusters:
                continue
            chosen_from_cluster = [c for c in final_selected if cluster_of_code.get(c) == cluster_id]
            chosen_set = set(chosen_from_cluster)
            for rank_index, candidate in enumerate(ordered_members):
                if candidate in chosen_set:
                    continue  # already tried and rejected above
                if candidate in dedup:
                    continue
                if candidate not in returns.columns:
                    continue
                if _passes_near_clone(candidate, dedup):
                    dedup.append(candidate)
                    clusters_with_rep.add(cluster_id)
                    short_history = float(history_days.get(candidate, 0.0)) < MIN_SELECTION_HISTORY_DAYS
                    short_suffix = "_short_history" if short_history else ""
                    dedup_reason_overrides[candidate] = (
                        f"multi_rep_{weight_scheme}_cluster_{cluster_id}_rank_{rank_index + 1}_coverage_rescue{short_suffix}"
                    )
                    break

        # Preserve original priority order for already-kept items, then append
        # any rescue picks at the end (they don't compete for rank-1 priority).
        final_selected = dedup

    reasons: dict[str, str] = {}
    for code in final_selected:
        if code in forced_codes and code not in selected_reason_map:
            reasons[code] = "forced_keep"
        elif code in dedup_reason_overrides:
            reasons[code] = dedup_reason_overrides[code]
        else:
            reasons[code] = selected_reason_map.get(code, f"multi_rep_{weight_scheme}")
    return final_selected, reasons


def prepare_recent_window(
    close: pd.DataFrame,
    volume_df: pd.DataFrame | None,
    lookback_days: int = CLUSTER_LOOKBACK_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    recent_rows = max(int(lookback_days) + 1, 2)
    recent_close = close.sort_index().tail(recent_rows)
    recent_volume = volume_df.sort_index().reindex(recent_close.index) if volume_df is not None else None
    returns = recent_close.pct_change().replace([np.inf, -np.inf], np.nan)
    return recent_close, recent_volume, returns


def pairwise_overlap_matrices(
    returns: pd.DataFrame,
    min_overlap_days: int = MIN_PAIR_OVERLAP_DAYS,
    low_overlap_distance: float = LOW_OVERLAP_DISTANCE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fallback_corr = 1.0 - float(low_overlap_distance)
    overlap = returns.notna().astype(int).T.dot(returns.notna().astype(int))
    corr = returns.corr(min_periods=min_overlap_days).fillna(fallback_corr)
    corr = corr.clip(lower=-1.0, upper=1.0)
    np.fill_diagonal(corr.values, 1.0)

    dist = 1.0 - corr
    np.fill_diagonal(dist.values, 0.0)
    return corr, overlap, dist


def summarize_overlap_counts(overlap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code in overlap.index:
        values = overlap.loc[code].drop(index=code, errors="ignore")
        values = values[values > 0]
        rows.append(
            {
                "code": code,
                "overlap_min_days": int(values.min()) if not values.empty else 0,
                "overlap_median_days": float(values.median()) if not values.empty else 0.0,
                "overlap_max_days": int(values.max()) if not values.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def _pick_cluster_representatives(
    members: list[str],
    returns: pd.DataFrame,
    per_cluster: int,
    history_days: pd.Series | None = None,
) -> list[str]:
    if not members:
        return []
    if history_days is None:
        history_days = pd.Series(dtype=float)
    eligible_members = [code for code in members if float(history_days.get(code, 0.0)) >= MIN_SELECTION_HISTORY_DAYS]
    candidate_members = eligible_members or members
    return _return_scores(candidate_members, returns).head(per_cluster).index.tolist()


def _trim_selected(
    selected: list[str],
    returns: pd.DataFrame,
    corr: pd.DataFrame,
    target_count: int | None,
    force_keep: Iterable[str] | None = None,
) -> list[str]:
    if target_count is None or target_count <= 0:
        return selected
    selected = list(dict.fromkeys(selected))
    if len(selected) <= target_count:
        return selected

    force_keep = list(force_keep or [])
    normalized_keep = build_normalized_code_set(force_keep)
    forced = [code for code in selected if code_in_normalized_set(code, normalized_keep)]
    forced = list(dict.fromkeys(forced))
    if len(forced) >= target_count:
        return forced

    candidates = [code for code in selected if code not in forced]
    mean_returns = returns[candidates].mean().reindex(candidates).fillna(0.0)
    final_selected = list(forced)

    for max_abs_corr_limit in FINAL_MAX_ABS_CORR_THRESHOLDS:
        if len(final_selected) >= target_count:
            break
        while len(final_selected) < target_count:
            available_selected = [picked for picked in final_selected if picked in returns.columns]
            ranked_candidates = []
            for code in candidates:
                if code in final_selected:
                    continue
                if available_selected and code in corr.index:
                    corr_slice = corr.loc[code, available_selected]
                    if isinstance(corr_slice, pd.Series):
                        max_abs_corr = float(corr_slice.abs().max()) if not corr_slice.empty else 0.0
                    else:
                        max_abs_corr = float(abs(corr_slice))
                else:
                    max_abs_corr = 0.0

                if max_abs_corr <= max_abs_corr_limit:
                    ranked_candidates.append((code, max_abs_corr, float(mean_returns.get(code, 0.0))))

            if not ranked_candidates:
                break

            ranked_candidates.sort(key=lambda item: (item[1], -item[2], item[0]))
            final_selected.append(ranked_candidates[0][0])

    return final_selected


def cluster_and_select(
    returns: pd.DataFrame,
    volume_df: pd.DataFrame | None,
    t: float = DIST_T,
    per_cluster: int = PER_CLUSTER,
    rule: str = RULE,
    force_keep: Iterable[str] | None = None,
    target_count: int | None = None,
    history_days: pd.Series | None = None,
    corr: pd.DataFrame | None = None,
    overlap: pd.DataFrame | None = None,
    linkage_method: str = "ward",
    family_clusters: pd.Series | None = None,
):
    """Run hierarchical clustering and selection.

    Pipeline (2026-04-24 简化版)：
      1. 对 ``returns`` 所有列做层级聚类（生产默认 Ward）
      2. 在每个簇内按 (Sharpe + 12-1 动量) 打分，挑 top-K 代表
      3. 最后一道 ``FINAL_MAX_ABS_CORR_THRESHOLDS`` 近克隆去冗余

    相关性由聚类保证，收益由簇内打分保证，两层互不干扰。

    Returns ``(Z, clusters, cluster_df, selected, reasons)``.
    """
    if corr is None or overlap is None:
        corr, overlap, _ = pairwise_overlap_matrices(returns, MIN_PAIR_OVERLAP_DAYS, LOW_OVERLAP_DISTANCE)

    codes = list(returns.columns)
    dist = 1.0 - corr
    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method=str(linkage_method).strip().lower())
    labels = fcluster(Z, t=t, criterion="distance")
    clusters = pd.Series(labels, index=codes, name="cluster")

    rows = []
    for cid, members in clusters.groupby(clusters).groups.items():
        members = list(members)
        n = len(members)
        if n <= 1:
            mean_corr = 1.0
        else:
            subcorr = corr.loc[members, members].values
            mean_corr = (subcorr.sum() - n) / (n * (n - 1))
        mean_volatility = returns[members].std().mean()
        mean_return = returns[members].mean().mean()
        try:
            avg_volume = volume_df[members].mean().mean()
        except Exception:
            avg_volume = None
        member_history = pd.Series(dtype=float) if history_days is None else history_days.reindex(members).fillna(0.0)
        rows.append(
            {
                "cluster": cid,
                "members": members,
                "n": n,
                "mean_corr": mean_corr,
                "mean_volatility": mean_volatility,
                "mean_return": mean_return,
                "avg_volume": avg_volume,
                "cluster_min_history_days": float(member_history.min()) if not member_history.empty else 0.0,
                "cluster_median_history_days": float(member_history.median()) if not member_history.empty else 0.0,
            }
        )
    cluster_df = pd.DataFrame(rows).sort_values(["mean_corr", "n"], ascending=[False, False])

    selection_rule = str(rule or RULE).strip().lower()
    multi_rep_rule_to_scheme = {
        "multi_rep_equal_mainline": "equal",
        "multi_rep_equal": "equal",
        "equal": "equal",
        "multi_rep_exp_benchmark": "exp",
        "multi_rep_exp": "exp",
        "exp": "exp",
        "multi_rep_harmonic_deprecated": "harmonic",
        "multi_rep_harmonic": "harmonic",
        "harmonic": "harmonic",
    }

    if selection_rule in multi_rep_rule_to_scheme:
        selected, reasons = _build_multi_rep_cluster_pool(
            cluster_df=cluster_df,
            returns=returns,
            history_days=history_days.reindex(codes).fillna(0.0) if history_days is not None else pd.Series(dtype=float),
            target_count=target_count,
            force_keep_codes=force_keep,
            weight_scheme=multi_rep_rule_to_scheme[selection_rule],
            exp_base=MULTI_REP_EXP_BASE,
            corr=corr,
            # Fix #4 + Plan B: near-clone dedup cap. User goal is a pool with
            # LOW internal correlation (portfolio holds only 5-6 names/day, so
            # high corr = concentrated systemic risk). Uses the *strictest*
            # (first) entry of FINAL_MAX_ABS_CORR_THRESHOLDS so any pair above
            # that threshold keeps only the higher-priority code.
            # A/B 实验支持：POOL_NEAR_CLONE_IDX 环境变量可指定索引；0=最严(Plan B)，
            # -1=最宽(Fix#4 前)。
            near_clone_corr_limit=(
                float(FINAL_MAX_ABS_CORR_THRESHOLDS[int(os.environ.get("POOL_NEAR_CLONE_IDX", 0))])
                if FINAL_MAX_ABS_CORR_THRESHOLDS
                else None
            ),
            # 方案 A: 簇内（同 cluster）成员之间使用更严的 0.60 阈值，避免
            # 同一宽基族里出现高度重叠的多只代表（如上证中盘 vs 上证超大盘）。
            intra_cluster_corr_limit=float(INTRA_CLUSTER_MAX_ABS_CORR),
            family_clusters=family_clusters,
        )
    else:
        selected = []
        reasons = {}
        for _, row in cluster_df.iterrows():
            members = row["members"]
            cid = row["cluster"]
            if len(members) == 1:
                pick = members[0]
                selected.append(pick)
                if history_days is not None and float(history_days.get(pick, 0.0)) < MIN_SELECTION_HISTORY_DAYS:
                    reasons[pick] = f"only_member_cluster_{cid}_short_history"
                else:
                    reasons[pick] = f"only_member_cluster_{cid}"
                continue
            picks = _pick_cluster_representatives(members, returns, per_cluster, history_days=history_days)
            for p in picks:
                selected.append(p)
                if history_days is not None and float(history_days.get(p, 0.0)) < MIN_SELECTION_HISTORY_DAYS:
                    reasons[p] = f"return_top_cluster_{cid}_short_history"
                else:
                    reasons[p] = f"return_top_cluster_{cid}"

        selected = list(dict.fromkeys(selected))
        if force_keep:
            normalized_keep = build_normalized_code_set(force_keep)
            for code in codes:
                if code_in_normalized_set(code, normalized_keep):
                    if code not in selected:
                        selected.append(code)
                    reasons.setdefault(code, "forced_keep")
        selected = _trim_selected(selected, returns, corr, target_count, force_keep)

    return Z, clusters, cluster_df, selected, reasons
