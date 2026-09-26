"""Quality review for cluster_mapping_selected.csv using full cluster_mapping.csv.

Primary use: after ``select_pool`` writes both CSVs, check whether each selected
code is still the best recent representative within its cluster (siblings live in
the full mapping with ``selected=False``).

Also supports a hand-edited ``--selected-csv``; optional diff vs a dated
machine snapshot (``cluster_mapping_selected_YYYYMMDD.csv``) is skipped unless
that file exists.

Usage::

    python -m pool_builder.daily_review --future-end 2026-05-20
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import squareform

from . import constants as C
from .audit import (
    _make_cluster_module_shim,
    build_audit_table,
    build_build_window_metrics,
    build_review_metrics_from_local_qlib,
    build_runtime_state,
    compute_forward_metrics,
    compute_trailing_return_metrics,
    fetch_close_prices,
    load_mapping_csv,
    parse_windows,
    save_focus_outputs,
)
from .code_utils import build_normalized_code_set, code_in_normalized_set, normalize_code
from .compare import _df_to_md_table, build_report, load_codes_from_csv
from .config import load_shared_config
from .data_loading import load_fund_list_universe


DEFAULT_MAPPING_CSV = C.CSV_OUT
DEFAULT_FUND_LIST_CSV = "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/fund_list.csv"
DEFAULT_SELECTED_CSV = C.CSV_SELECTED_OUT
DEFAULT_OUT_BASE = os.path.join(C.OUT_DIR, "etf_filter_checks")
DEFAULT_WINDOWS = "5,20,60"
DEFAULT_MIN_REGRET = 0.01
DEFAULT_MIN_CLUSTER_N = 2
DEFAULT_TOP_N = 15
DEFAULT_RELATED_SELECTED_TOP_K = 3

REASON_KIND_CN = {
    "always_drop_cluster_rule": "整簇规则剔除",
    "target_count_not_selected": "目标池裁剪（multi_rep 未选中该簇）",
    "insufficient_history": "历史数据不足",
    "always_drop_code_rule": "单码规则剔除",
    "excluded_type": "基金类型排除",
    "inception_after_cutoff": "成立日晚于截止日",
    "missing_from_qlib_data": "qlib 无行情",
    "mixed_diagnostics": "混合剔除原因",
}

MISSED_MD_COLUMN_CN = {
    "cluster": "簇编号",
    "n": "簇规模",
    "members_preview": "簇成员预览",
    "best_code": "最强代码",
    "best_name": "最强中文名",
    "best_20d_ret_pct": "最强近20日收益",
    "reason": "原因说明",
    "related_selected_clusters": "相关已选簇（选入标的/20日收益）",
}

MISSED_MD_COLUMN_CN_PROXY = {
    **MISSED_MD_COLUMN_CN,
    "best_20d_ret_pct": "簇均日收益(构建窗)",
}

REP_LAG_MD_COLUMN_CN = {
    "cluster": "簇编号",
    "selected_code": "入选代码",
    "selected_name": "入选中文名",
    "best_code": "最强代码",
    "best_name": "最强中文名",
    "selected_pct": "入选近20日收益",
    "best_pct": "最强近20日收益",
    "regret_pct": "落后幅度",
    "rank_20d": "簇内20日排名",
    "likely_due_to_low_volume": "可能因流动性保留",
}


def resolve_fund_list_csv(shared_config_path: str | None, explicit: str | None) -> str:
    if explicit:
        return str(Path(explicit).expanduser())
    cfg = load_shared_config(shared_config_path)
    return str(cfg.get("fund_list_csv") or DEFAULT_FUND_LIST_CSV)


def load_fund_name_map(fund_list_csv: str) -> dict[str, str]:
    _, code_name_map = load_fund_list_universe(fund_list_csv)
    return code_name_map


def lookup_fund_name(code: str, name_map: dict[str, str]) -> str:
    raw = str(code).strip()
    if not raw:
        return ""
    for key in (raw.upper(), raw, raw.lower()):
        if key in name_map:
            return str(name_map[key]).strip()
    if len(raw) > 2 and raw[2:].isdigit():
        tail = raw[2:]
        if tail in name_map:
            return str(name_map[tail]).strip()
    return ""


def format_code_with_cn(code: str, name_map: dict[str, str]) -> str:
    code_u = str(code).strip().upper()
    cn = lookup_fund_name(code_u, name_map)
    return f"{code_u} {cn}" if cn else code_u


def _split_reason_tags(reason_series: pd.Series) -> list[str]:
    tags: list[str] = []
    for raw in reason_series.fillna("").astype(str):
        for part in raw.split("|"):
            part = part.strip()
            if part:
                tags.append(part)
    return tags


def _find_cluster_drop_seeds(member_codes: list[str]) -> list[str]:
    configured = build_normalized_code_set(C.ALWAYS_DROP_CLUSTER_CODES)
    found: list[str] = []
    for code in member_codes:
        if code_in_normalized_set(code, configured):
            found.append(normalize_code(code))
    return list(dict.fromkeys(found))


def _find_single_drop_codes(member_codes: list[str]) -> list[str]:
    configured = build_normalized_code_set(C.ALWAYS_DROP_CODES)
    found: list[str] = []
    for code in member_codes:
        if code_in_normalized_set(code, configured):
            found.append(normalize_code(code))
    return list(dict.fromkeys(found))


def _infer_missed_cluster_reason(grp: pd.DataFrame) -> tuple[str, list[str]]:
    """Classify why a cluster has no selected representative in the hand pool."""
    member_codes = grp["code"].astype(str).str.upper().tolist()
    tags = _split_reason_tags(grp["reason"]) if "reason" in grp.columns else []
    drop_seeds = _find_cluster_drop_seeds(member_codes)

    if drop_seeds or any("always_drop_cluster_code" in tag for tag in tags):
        return "always_drop_cluster_rule", drop_seeds
    if tags and all("always_drop_code" in tag for tag in tags):
        return "always_drop_code_rule", _find_single_drop_codes(member_codes) or [
            normalize_code(code) for code in member_codes
        ]
    if any("insufficient_history" in tag for tag in tags):
        return "insufficient_history", []
    if any("excluded_type" in tag for tag in tags):
        return "excluded_type", []
    if any("inception_after_cutoff" in tag for tag in tags):
        return "inception_after_cutoff", []
    if any("missing_from_qlib_data" in tag for tag in tags):
        return "missing_from_qlib_data", []
    if tags:
        return "mixed_diagnostics", drop_seeds
    return "target_count_not_selected", []


def format_missed_reason_detail(reason_kind: str, drop_seed_codes: str, name_map: dict[str, str]) -> str:
    cn = REASON_KIND_CN.get(str(reason_kind), str(reason_kind))
    seeds = [part.strip() for part in str(drop_seed_codes or "").split("|") if part.strip()]
    if seeds:
        labels = [format_code_with_cn(seed, name_map) for seed in seeds]
        return f"{cn}；种子：{', '.join(labels)}"
    if reason_kind == "always_drop_cluster_rule":
        return f"{cn}（ALWAYS_DROP_CLUSTER_CODES；本簇 mapping 行内未命中配置种子）"
    return cn


def format_members_preview_cn(preview: str, name_map: dict[str, str]) -> str:
    if preview is None or (isinstance(preview, float) and np.isnan(preview)):
        return ""
    return "；".join(
        format_code_with_cn(part.strip(), name_map)
        for part in str(preview).split("|")
        if str(part).strip()
    )


def format_missed_df_for_markdown(missed_df: pd.DataFrame, name_map: dict[str, str], top_n: int, *, csv_proxy: bool = False) -> pd.DataFrame:
    if missed_df.empty:
        return missed_df
    show = missed_df.head(top_n).copy()
    if "best_20d_ret" in show.columns:
        show["best_20d_ret_pct"] = show["best_20d_ret"].map(
            lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "-"
        )
    if "members_preview" in show.columns:
        show["members_preview"] = show["members_preview"].map(
            lambda x: format_members_preview_cn(x, name_map)
        )
    if "best_code" in show.columns:
        show["best_code"] = show["best_code"].map(lambda x: str(x).strip().upper() if pd.notna(x) else "")
        show["best_name"] = show["best_code"].map(lambda c: lookup_fund_name(c, name_map) if c else "")
    if "reason_kind" in show.columns:
        show["reason"] = show.apply(
            lambda row: format_missed_reason_detail(
                row.get("reason_kind", ""),
                row.get("drop_seed_codes", ""),
                name_map,
            ),
            axis=1,
        )
    display_cols = [
        c
        for c in [
            "cluster",
            "n",
            "members_preview",
            "best_code",
            "best_name",
            "best_20d_ret_pct",
            "reason",
            "related_selected_clusters",
        ]
        if c in show.columns
    ]
    out = show[display_cols].rename(
        columns=MISSED_MD_COLUMN_CN_PROXY if csv_proxy else MISSED_MD_COLUMN_CN
    )
    return out


def format_rep_lag_df_for_markdown(rep_lag_df: pd.DataFrame, name_map: dict[str, str], top_n: int) -> pd.DataFrame:
    if rep_lag_df.empty:
        return rep_lag_df
    show = rep_lag_df.head(top_n).copy()
    pct_map = {
        "selected_20d_ret": "selected_pct",
        "best_20d_ret": "best_pct",
        "regret_20d": "regret_pct",
    }
    for src, dst in pct_map.items():
        if src in show.columns:
            show[dst] = show[src].map(lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "-")
    if "selected_code" in show.columns:
        show["selected_code"] = show["selected_code"].map(
            lambda x: str(x).strip().upper() if pd.notna(x) else ""
        )
        show["selected_name"] = show["selected_code"].map(
            lambda c: lookup_fund_name(c, name_map) if c else ""
        )
    if "best_code" in show.columns:
        show["best_code"] = show["best_code"].map(lambda x: str(x).strip().upper() if pd.notna(x) else "")
        show["best_name"] = show["best_code"].map(lambda c: lookup_fund_name(c, name_map) if c else "")
    display_cols = [
        c
        for c in [
            "cluster",
            "selected_code",
            "selected_name",
            "best_code",
            "best_name",
            "selected_pct" if "selected_pct" in show.columns else "selected_20d_ret",
            "best_pct" if "best_pct" in show.columns else "best_20d_ret",
            "regret_pct" if "regret_pct" in show.columns else "regret_20d",
            "rank_20d",
            "likely_due_to_low_volume",
        ]
        if c in show.columns
    ]
    return show[display_cols].rename(columns=REP_LAG_MD_COLUMN_CN)


def _return_source_label(csv_proxy: bool, bin_fallback: bool) -> str:
    if csv_proxy:
        return "mapping CSV 簇级 mean_return 代理（qlib 不可用；簇内成员同值，无法算代表落后）"
    if bin_fallback:
        return "本地 qlib bin 回看窗口（pyqlib 不可用）"
    return "qlib 回看窗口（审查日 anchor）"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review cluster_mapping_selected.csv quality against full cluster_mapping.csv"
    )
    parser.add_argument("--mapping-csv", default=DEFAULT_MAPPING_CSV, help="Full mapping from the same select run")
    parser.add_argument(
        "--selected-csv",
        default=DEFAULT_SELECTED_CSV,
        help="Pool to audit (default: generated cluster_mapping_selected.csv)",
    )
    parser.add_argument(
        "--auto-selected-csv",
        default=None,
        help="Today's machine pool CSV; default: cluster_mapping_selected_{future_end compact}.csv",
    )
    parser.add_argument("--future-end", default=pd.Timestamp.today().normalize().strftime("%Y-%m-%d"))
    parser.add_argument("--windows", default=DEFAULT_WINDOWS)
    parser.add_argument("--min-regret", type=float, default=DEFAULT_MIN_REGRET)
    parser.add_argument("--min-cluster-n", type=int, default=DEFAULT_MIN_CLUSTER_N)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--shared-config", default=None)
    parser.add_argument(
        "--fund-list-csv",
        default=None,
        help=f"fund_list.csv for Chinese names (default: shared_config fund_list_csv or {DEFAULT_FUND_LIST_CSV})",
    )
    parser.add_argument("--skip-audit", action="store_true", help="Skip extended missed-winner audit outputs")
    parser.add_argument(
        "--skip-diff",
        action="store_true",
        help="Never compare to cluster_mapping_selected_YYYYMMDD.csv (quality-only mode)",
    )
    return parser


def load_hand_selected_codes(path: str) -> list[str]:
    codes = load_codes_from_csv(path)
    return list(dict.fromkeys(codes))


def overlay_hand_selected(mapping_df: pd.DataFrame, hand_codes: list[str]) -> pd.DataFrame:
    hand_set = {str(c).strip().upper() for c in hand_codes if str(c).strip()}
    out = mapping_df.copy()
    out["selected"] = out["code"].astype(str).str.strip().str.upper().isin(hand_set)
    out["selection_source"] = np.where(out["selected"], "hand_curated", "not_in_hand_pool")
    return out


def resolve_auto_selected_path(future_end: str, explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None
    suffix = pd.Timestamp(future_end).strftime("%Y%m%d")
    path = Path(C.dated_selected_csv_path(suffix))
    return path if path.is_file() else None


def build_csv_proxy_future_df(mapping_df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Fallback when neither pyqlib nor local qlib bins are available.

    Uses cluster-level ``mean_return`` copied onto every member row in mapping CSV.
    All members inside a cluster therefore share the same proxy value.
    """
    proxy = pd.to_numeric(mapping_df.get("mean_return", np.nan), errors="coerce")
    rows = []
    for code in mapping_df["code"].astype(str):
        base = proxy[mapping_df["code"].astype(str) == code]
        val = float(base.iloc[0]) if not base.empty and pd.notna(base.iloc[0]) else np.nan
        row = {"code": code}
        for w in windows:
            row[f"fwd_ret_{w}d"] = val
        rows.append(row)
    return pd.DataFrame(rows)


def build_csv_proxy_build_metrics(mapping_df: pd.DataFrame) -> pd.DataFrame:
    out = mapping_df[["code"]].copy()
    out["volume_rank_in_cluster"] = np.nan
    if "avg_volume" in mapping_df.columns:
        out["avg_volume"] = mapping_df["avg_volume"]
    return out


def build_forward_metrics_from_state(
    cluster_module,
    state: dict,
    future_end: str,
    windows: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping_df = state["mapping_df"]
    codes = mapping_df["code"].astype(str).tolist()
    max_window = max(windows) if windows else 20
    future_start = (pd.Timestamp(future_end) - pd.tseries.offsets.BDay(max_window + 40)).strftime("%Y-%m-%d")
    future_close = fetch_close_prices(cluster_module, codes, future_start, future_end)
    future_df = compute_trailing_return_metrics(future_close, future_end, windows)
    member_map = mapping_df.groupby("cluster")["code"].apply(list).to_dict()
    build_metrics, _ = build_build_window_metrics(
        state["returns"],
        state["volume_df"],
        member_map,
    )
    return future_df, build_metrics


def _code_key(code: object) -> str:
    return str(code).strip().upper()


def _coerce_cluster_id(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _format_signed_pct(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    if pd.isna(numeric):
        return "-"
    return f"{numeric * 100:+.2f}%"


def _build_forward_return_map(future_df: pd.DataFrame, focus_window: int) -> dict[str, float]:
    ret_col = f"fwd_ret_{focus_window}d"
    if future_df.empty or ret_col not in future_df.columns or "code" not in future_df.columns:
        return {}
    ret_by_code: dict[str, float] = {}
    for _, future_row in future_df.iterrows():
        key = _code_key(future_row.get("code", ""))
        if not key:
            continue
        value = pd.to_numeric(future_row.get(ret_col), errors="coerce")
        ret_by_code[key] = float(value) if pd.notna(value) else np.nan
    return ret_by_code


def _format_selected_cluster_entry(
    cluster_id: int,
    selected_codes: list[str],
    name_map: dict[str, str],
    ret_by_code: dict[str, float],
) -> str:
    parts: list[str] = []
    for code in selected_codes:
        key = _code_key(code)
        label = format_code_with_cn(key, name_map)
        parts.append(f"{label} {_format_signed_pct(ret_by_code.get(key, np.nan))}")
    return f"簇 {cluster_id}：" + "；".join(parts)


def _build_cophenetic_distance_df(returns: pd.DataFrame, code_keys: list[str]) -> pd.DataFrame:
    if returns is None or returns.empty:
        return pd.DataFrame()

    returns_column_by_key: dict[str, object] = {}
    for returns_column in returns.columns:
        key = _code_key(returns_column)
        if key and key not in returns_column_by_key:
            returns_column_by_key[key] = returns_column

    matched_keys = [key for key in dict.fromkeys(code_keys) if key in returns_column_by_key]
    if len(matched_keys) < 2:
        return pd.DataFrame()

    matched_columns = [returns_column_by_key[key] for key in matched_keys]
    panel = returns[matched_columns].copy()
    panel.columns = matched_keys
    corr = panel.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    dist_values = np.clip(1.0 - corr.to_numpy(dtype=float), 0.0, 2.0)
    dist_values = (dist_values + dist_values.T) / 2.0
    np.fill_diagonal(dist_values, 0.0)
    condensed = squareform(dist_values, checks=False)
    linkage_matrix = linkage(condensed, method="ward")
    cophenetic_result = cophenet(linkage_matrix, condensed)
    cophenetic_condensed = cophenetic_result[1] if isinstance(cophenetic_result, tuple) else cophenetic_result
    cophenetic_values = squareform(cophenetic_condensed)
    np.fill_diagonal(cophenetic_values, 0.0)
    return pd.DataFrame(cophenetic_values, index=matched_keys, columns=matched_keys)


def add_related_selected_clusters(
    missed_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    hand_codes: list[str],
    future_df: pd.DataFrame,
    returns: pd.DataFrame | None,
    name_map: dict[str, str],
    *,
    focus_window: int = 20,
    top_k: int = DEFAULT_RELATED_SELECTED_TOP_K,
    unavailable_text: str = "-",
) -> pd.DataFrame:
    out = missed_df.copy()
    out["related_selected_clusters"] = "-"
    if out.empty:
        return out

    target_mask = out["reason_kind"].astype(str).eq("target_count_not_selected") if "reason_kind" in out.columns else pd.Series(False, index=out.index)
    if not target_mask.any():
        return out

    if returns is None or returns.empty:
        out.loc[target_mask, "related_selected_clusters"] = unavailable_text
        return out

    mapping_work = mapping_df.copy()
    mapping_work["code_key"] = mapping_work["code"].map(_code_key)
    mapping_work["cluster_id"] = pd.to_numeric(mapping_work["cluster"], errors="coerce")
    mapping_work = mapping_work[mapping_work["code_key"].ne("") & mapping_work["cluster_id"].notna()].copy()
    if mapping_work.empty:
        out.loc[target_mask, "related_selected_clusters"] = unavailable_text
        return out
    mapping_work["cluster_id"] = mapping_work["cluster_id"].astype(int)

    hand_set = {_code_key(code) for code in hand_codes if _code_key(code)}
    selected_rows = mapping_work[mapping_work["code_key"].isin(hand_set)].copy()
    if selected_rows.empty:
        out.loc[target_mask, "related_selected_clusters"] = unavailable_text
        return out

    cluster_members = mapping_work.groupby("cluster_id")["code_key"].apply(
        lambda series: list(dict.fromkeys(series.tolist()))
    ).to_dict()
    selected_by_cluster = selected_rows.groupby("cluster_id")["code_key"].apply(
        lambda series: list(dict.fromkeys(series.tolist()))
    ).to_dict()
    cophenetic_df = _build_cophenetic_distance_df(returns, mapping_work["code_key"].tolist())
    if cophenetic_df.empty:
        out.loc[target_mask, "related_selected_clusters"] = unavailable_text
        return out

    ret_by_code = _build_forward_return_map(future_df, focus_window)

    for missed_index, missed_row in out.loc[target_mask].iterrows():
        missed_cluster_id = _coerce_cluster_id(missed_row.get("cluster"))
        if missed_cluster_id is None:
            out.at[missed_index, "related_selected_clusters"] = unavailable_text
            continue

        missed_members = [
            code for code in cluster_members.get(missed_cluster_id, [])
            if code in cophenetic_df.index
        ]
        if not missed_members:
            out.at[missed_index, "related_selected_clusters"] = unavailable_text
            continue

        ranking: list[tuple[float, float, int, list[str]]] = []
        for selected_cluster_id, selected_codes in selected_by_cluster.items():
            if selected_cluster_id == missed_cluster_id:
                continue
            selected_members = [
                code for code in cluster_members.get(selected_cluster_id, [])
                if code in cophenetic_df.columns
            ]
            if not selected_members:
                continue
            distance_block = cophenetic_df.loc[missed_members, selected_members]
            min_distance = float(distance_block.min().min())
            if pd.isna(min_distance):
                continue
            selected_returns = [ret_by_code.get(_code_key(code), np.nan) for code in selected_codes]
            valid_selected_returns = [float(value) for value in selected_returns if pd.notna(value)]
            best_selected_return = max(valid_selected_returns) if valid_selected_returns else -np.inf
            ranking.append((min_distance, -best_selected_return, int(selected_cluster_id), selected_codes))

        if not ranking:
            out.at[missed_index, "related_selected_clusters"] = unavailable_text
            continue

        ranking.sort(key=lambda item: (item[0], item[1], item[2]))
        entries = [
            _format_selected_cluster_entry(cluster_id, selected_codes, name_map, ret_by_code)
            for _, _, cluster_id, selected_codes in ranking[: max(1, int(top_k))]
        ]
        out.at[missed_index, "related_selected_clusters"] = "<br>".join(entries)

    return out


def build_missed_clusters(
    mapping_df: pd.DataFrame,
    hand_codes: list[str],
    future_df: pd.DataFrame,
    *,
    min_cluster_n: int,
    focus_window: int = 20,
) -> pd.DataFrame:
    hand_set = {c.upper() for c in hand_codes}
    merged = mapping_df.merge(future_df, on="code", how="left")
    rows: list[dict] = []
    ret_col = f"fwd_ret_{focus_window}d"

    for cluster_id, grp in merged.groupby("cluster"):
        members = grp["code"].astype(str).str.upper().tolist()
        n = int(grp["n"].iloc[0]) if "n" in grp.columns and grp["n"].notna().any() else len(grp)
        if n < min_cluster_n:
            continue
        selected_in_cluster = [c for c in members if c in hand_set]
        if selected_in_cluster:
            continue

        grp_sorted = grp.sort_values(ret_col, ascending=False, na_position="last")
        best = grp_sorted.iloc[0] if not grp_sorted.empty else None
        reason_kind, drop_seeds = _infer_missed_cluster_reason(grp)

        rows.append(
            {
                "cluster": int(cluster_id),
                "n": n,
                "members_count": len(members),
                "members_preview": "|".join(members[:6]),
                "best_code": best["code"] if best is not None else "",
                "best_name": best.get("name", "") if best is not None else "",
                f"best_{focus_window}d_ret": float(best[ret_col]) if best is not None and pd.notna(best.get(ret_col)) else np.nan,
                "reason_kind": reason_kind,
                "drop_seed_codes": "|".join(drop_seeds),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "cluster",
                "n",
                "members_count",
                "members_preview",
                "best_code",
                "best_name",
                f"best_{focus_window}d_ret",
                "reason_kind",
                "drop_seed_codes",
            ]
        )
    out = pd.DataFrame(rows)
    sort_col = f"best_{focus_window}d_ret"
    return out.sort_values(sort_col, ascending=False, na_position="last")


def build_rep_lag(
    mapping_df: pd.DataFrame,
    hand_codes: list[str],
    future_df: pd.DataFrame,
    build_metrics: pd.DataFrame,
    *,
    min_regret: float,
    focus_window: int = 20,
) -> pd.DataFrame:
    hand_set = {c.upper() for c in hand_codes}
    merged = mapping_df.merge(future_df, on="code", how="left")
    if not build_metrics.empty:
        merged = merged.merge(build_metrics, on="code", how="left")

    ret_col = f"fwd_ret_{focus_window}d"
    rows: list[dict] = []

    for cluster_id, grp in merged.groupby("cluster"):
        members = grp.copy()
        if members.empty:
            continue
        members = members.sort_values(ret_col, ascending=False, na_position="last")
        best_row = members.iloc[0]
        best_code = str(best_row["code"]).upper()
        best_ret = float(best_row[ret_col]) if pd.notna(best_row.get(ret_col)) else np.nan

        selected_rows = members[members["code"].astype(str).str.upper().isin(hand_set)]
        if selected_rows.empty:
            continue

        for _, sel in selected_rows.iterrows():
            sel_code = str(sel["code"]).upper()
            sel_ret = float(sel[ret_col]) if pd.notna(sel.get(ret_col)) else np.nan
            regret = best_ret - sel_ret if pd.notna(best_ret) and pd.notna(sel_ret) else np.nan
            rank = int((members[ret_col] >= sel_ret).sum()) if pd.notna(sel_ret) else len(members)

            vol_rank = sel.get("volume_rank_in_cluster", np.nan)
            likely_low_volume = bool(pd.notna(vol_rank) and float(vol_rank) > 1.0)

            if rank <= 1:
                continue
            if pd.notna(regret) and regret < min_regret:
                continue

            rows.append(
                {
                    "cluster": int(cluster_id),
                    "n": int(sel.get("n", len(members)) if pd.notna(sel.get("n")) else len(members)),
                    "selected_code": sel_code,
                    "selected_name": sel.get("name", ""),
                    "best_code": best_code,
                    "best_name": best_row.get("name", ""),
                    f"selected_{focus_window}d_ret": sel_ret,
                    f"best_{focus_window}d_ret": best_ret,
                    f"regret_{focus_window}d": regret,
                    f"rank_{focus_window}d": rank,
                    "volume_rank_in_cluster": vol_rank,
                    "likely_due_to_low_volume": likely_low_volume,
                    "selected_reason": sel.get("reason", ""),
                }
            )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(f"regret_{focus_window}d", ascending=False, na_position="last")


def write_diff_vs_auto(
    hand_csv: str,
    auto_csv: Path,
    mapping_csv: str,
    out_dir: str,
    date_tag: str,
) -> tuple[int, int, str]:
    report_md, added_df, removed_df = build_report(
        old_path=hand_csv,
        new_path=str(auto_csv),
        cluster_map_path=mapping_csv,
        pool_summary_path=None,
        pool_meta_path=None,
        old_label="hand_curated",
        new_label=f"machine_{date_tag}",
    )
    diff_path = os.path.join(out_dir, f"diff_vs_auto_{date_tag}.md")
    with open(diff_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    added_df.to_csv(os.path.join(out_dir, f"diff_vs_auto_{date_tag}_added.csv"), index=False)
    removed_df.to_csv(os.path.join(out_dir, f"diff_vs_auto_{date_tag}_removed.csv"), index=False)
    return len(added_df), len(removed_df), diff_path


def build_daily_markdown(
    *,
    review_date: str,
    hand_count: int,
    mapping_count: int,
    missed_df: pd.DataFrame,
    rep_lag_df: pd.DataFrame,
    added_n: int,
    removed_n: int,
    diff_path: str | None,
    audit_summary: str,
    top_n: int,
    name_map: dict[str, str],
    csv_proxy: bool = False,
    bin_fallback: bool = False,
    generated_at: str | None = None,
) -> str:
    lines = [
        f"# 每日聚类池审查 {review_date}",
        "",
        "## 摘要",
        "",
        f"| 项目 | 值 |",
        f"| --- | --- |",
        f"| 审查日 | {review_date} |",
        f"| 报告生成时间 | {generated_at or '-'} |",
        f"| 漏族原因口径 | reason_kind + drop_seed_codes（v2） |",
        f"| 手工池标的数 | {hand_count} |",
        f"| 全量 mapping 行数 | {mapping_count} |",
        f"| 漏族（簇内无人选手工池） | {len(missed_df)} |",
        f"| 代表落后（{20}d 非簇内第1） | {len(rep_lag_df)} |",
        f"| 相对今日机器池 新增/移除 | {added_n} / {removed_n} |",
        f"| 收益数据来源 | {_return_source_label(csv_proxy, bin_fallback)} |",
        f"| 收益锚点日 | {review_date}（回看 {20} 个交易日） |",
        "",
    ]
    if diff_path:
        lines.append(f"机器对比报告：[`{diff_path}`]({diff_path})")
        lines.append("")

    lines.append("## 漏族（Top {})".format(top_n))
    lines.append("")
    lines.append("中文名来自 `fund_list.csv`（基金简称列）。")
    lines.append("")
    if missed_df.empty:
        lines.append("（无 — 每个 n≥2 的簇在入选池中至少有一只代表）")
    else:
        lines.append(_df_to_md_table(format_missed_df_for_markdown(missed_df, name_map, top_n, csv_proxy=csv_proxy)))
    lines.append("")

    lines.append("## 代表落后（Top {}，按 regret_20d）".format(top_n))
    lines.append("")
    if rep_lag_df.empty:
        lines.append("（无 — 入选标的均为各自簇内近20日收益前列，或落后未达阈值）")
    else:
        lines.append(_df_to_md_table(format_rep_lag_df_for_markdown(rep_lag_df, name_map, top_n)))
    lines.append("")

    lines.append("## 今日建议")
    lines.append("")
    suggestions: list[str] = []
    if not missed_df.empty:
        top_miss = missed_df.iloc[0]
        best_code = str(top_miss.get("best_code", "")).strip().upper()
        best_cn = lookup_fund_name(best_code, name_map) if best_code else ""
        best_label = f"{best_code} {best_cn}".strip() if best_cn else best_code
        reason_detail = format_missed_reason_detail(
            str(top_miss.get("reason_kind", "")),
            str(top_miss.get("drop_seed_codes", "")),
            name_map,
        )
        suggestions.append(
            f"- 漏族：簇 {top_miss['cluster']}（n={top_miss['n']}）无入选代表；"
            f"近20日最强为 {best_label}；原因：{reason_detail}。"
        )
    if not rep_lag_df.empty:
        top_lag = rep_lag_df.iloc[0]
        suggestions.append(
            f"- 代表落后：簇 {top_lag['cluster']} 选手 {top_lag['selected_code']}，"
            f"但 {top_lag['best_code']} 近20日更强（regret≈{top_lag.get('regret_20d', 0)*100:.2f}%）；"
            f"{'可能因流动性规则保留现代表' if top_lag.get('likely_due_to_low_volume') else '可考虑替换代表'}。"
        )
    if added_n or removed_n:
        suggestions.append(
            f"- 机器池与手工池差异 {added_n} 增 / {removed_n} 减；定稿手工 CSV 后执行 filter_txt 刷新 canonical txt。"
        )
    if not suggestions:
        suggestions.append("- 今日无明显漏族或代表落后；维持手工池即可。")
    lines.extend(suggestions)
    lines.append("")
    if audit_summary:
        lines.append("## Audit 扩展（强但未选）")
        lines.append("")
        lines.append("```")
        lines.append(audit_summary.strip())
        lines.append("```")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "说明：审查以 **手工 cluster_mapping_selected.csv** 为权威；"
        "全量 cluster_mapping.csv 仅用于同族对比。反事实换代表需人工决策后更新 CSV 并 `--skip-select` 刷 txt。"
    )
    return "\n".join(lines)


def run_daily_review(args: argparse.Namespace) -> int:
    windows = parse_windows(args.windows)
    focus_window = 20 if 20 in windows else windows[-1]
    review_date = pd.Timestamp(args.future_end).strftime("%Y-%m-%d")
    date_tag = pd.Timestamp(args.future_end).strftime("%Y%m%d")

    out_dir = args.out_dir or os.path.join(DEFAULT_OUT_BASE, f"daily_{date_tag}")
    os.makedirs(out_dir, exist_ok=True)

    hand_codes = load_hand_selected_codes(args.selected_csv)
    fund_list_csv = resolve_fund_list_csv(args.shared_config, args.fund_list_csv)
    name_map = load_fund_name_map(fund_list_csv)
    print(f"[INFO] Loaded fund names from {fund_list_csv}")

    cluster_module = _make_cluster_module_shim(args.shared_config)
    shared_cfg = load_shared_config(args.shared_config)
    csv_proxy = False
    bin_fallback = False
    state = None
    returns = None
    try:
        state = build_runtime_state(cluster_module, args.mapping_csv)
        state["mapping_df"] = overlay_hand_selected(state["mapping_df"], hand_codes)
        state["final_selected"] = list(dict.fromkeys(hand_codes))
        mapping_df = state["mapping_df"]
        future_df, build_metrics = build_forward_metrics_from_state(
            cluster_module, state, args.future_end, windows
        )
        returns = state["returns"]
    except RuntimeError as exc:
        msg = str(exc)
        if "qlib is unavailable" not in msg and "No mapping codes were matched" not in msg:
            raise
        try:
            print(f"[WARN] pyqlib path failed ({exc}); trying local qlib feature bins.")
            mapping_df, future_df, build_metrics, returns = build_review_metrics_from_local_qlib(
                shared_cfg["provider_uri"],
                overlay_hand_selected(load_mapping_csv(args.mapping_csv), hand_codes),
                tuple(shared_cfg["test_period"]),
                args.future_end,
                windows,
            )
            bin_fallback = True
            state = {"mapping_df": mapping_df, "returns": returns}
            print("[INFO] Loaded review metrics from local qlib feature bins.")
        except RuntimeError as bin_exc:
            print(f"[WARN] local qlib bin path failed ({bin_exc}); using CSV mean_return proxy.")
            csv_proxy = True
            mapping_df = overlay_hand_selected(load_mapping_csv(args.mapping_csv), hand_codes)
            future_df = build_csv_proxy_future_df(mapping_df, windows)
            build_metrics = build_csv_proxy_build_metrics(mapping_df)
            returns = None

    missed_df = build_missed_clusters(
        mapping_df,
        hand_codes,
        future_df,
        min_cluster_n=int(args.min_cluster_n),
        focus_window=focus_window,
    )
    rep_lag_df = build_rep_lag(
        mapping_df,
        hand_codes,
        future_df,
        build_metrics,
        min_regret=float(args.min_regret),
        focus_window=focus_window,
    )
    missed_df = add_related_selected_clusters(
        missed_df,
        mapping_df,
        hand_codes,
        future_df,
        returns,
        name_map,
        focus_window=focus_window,
        unavailable_text="qlib不可用，未计算树近邻" if csv_proxy else "-",
    )

    missed_df.to_csv(os.path.join(out_dir, "missed_clusters.csv"), index=False)
    rep_lag_df.to_csv(os.path.join(out_dir, "rep_lag.csv"), index=False)

    added_n = removed_n = 0
    diff_path = None
    auto_path = None if args.skip_diff else resolve_auto_selected_path(args.future_end, args.auto_selected_csv)
    if auto_path is not None:
        added_n, removed_n, diff_path = write_diff_vs_auto(
            args.selected_csv,
            auto_path,
            args.mapping_csv,
            out_dir,
            date_tag,
        )
        print(f"[INFO] Wrote diff vs machine pool: {diff_path}")
    else:
        print("[WARN] No auto-selected CSV found; skipped diff_vs_auto section.")

    audit_summary = ""
    if not args.skip_audit and not csv_proxy and not bin_fallback and state is not None:
        audit_df = build_audit_table(
            cluster_module,
            state,
            args.future_end,
            windows,
            strong_percentile=0.90,
            wrong_cluster_corr=0.55,
            wrong_cluster_center_corr=0.55,
            min_regret=float(args.min_regret),
        )
        save_focus_outputs(audit_df, out_dir, windows, int(args.top_n))
        if "miss_type" in audit_df.columns:
            audit_summary = audit_df["miss_type"].value_counts().to_string()
        print(f"[INFO] Audit outputs written under {out_dir}")
    elif not args.skip_audit and (csv_proxy or bin_fallback):
        print("[WARN] Skipped audit extension outside pyqlib mode.")

    report_md = build_daily_markdown(
        review_date=review_date,
        hand_count=len(hand_codes),
        mapping_count=len(mapping_df),
        missed_df=missed_df,
        rep_lag_df=rep_lag_df,
        added_n=added_n,
        removed_n=removed_n,
        diff_path=diff_path,
        audit_summary=audit_summary,
        top_n=int(args.top_n),
        name_map=name_map,
        csv_proxy=csv_proxy,
        bin_fallback=bin_fallback,
        generated_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    report_path = os.path.join(out_dir, f"daily_cluster_review_{date_tag}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"[INFO] Daily review report: {report_path}")
    if not missed_df.empty and "reason_kind" in missed_df.columns:
        preview = missed_df.head(3)[["cluster", "reason_kind", "drop_seed_codes"]]
        print("[INFO] missed_clusters reason preview (top 3):")
        print(preview.to_string(index=False))
    print(f"[INFO] missed_clusters={len(missed_df)} rep_lag={len(rep_lag_df)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_daily_review(args)


if __name__ == "__main__":
    raise SystemExit(main())
