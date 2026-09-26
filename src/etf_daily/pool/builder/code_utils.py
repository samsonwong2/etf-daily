"""Code normalization helpers and manual-keep / cluster-seed loaders."""
from __future__ import annotations

import os
from typing import Iterable

import pandas as pd


def normalize_code(code: str) -> str:
    return str(code).strip().upper()


def normalized_aliases(code: str) -> set[str]:
    norm = normalize_code(code)
    aliases = {norm}
    if norm.startswith(("SH", "SZ")) and len(norm) > 2:
        aliases.add(norm[2:])
    return aliases


def build_normalized_code_set(codes: Iterable[str] | None) -> set[str]:
    normalized: set[str] = set()
    for code in codes or []:
        for alias in normalized_aliases(code):
            normalized.add(alias)
    return normalized


def code_in_normalized_set(code: str, normalized_set: set[str]) -> bool:
    norm = normalize_code(code)
    if norm in normalized_set:
        return True
    if norm.startswith(("SH", "SZ")) and norm[2:] in normalized_set:
        return True
    return False


def load_manual_keep_codes(file_path: str | None) -> list[str]:
    if not file_path or not os.path.exists(file_path):
        return []
    df = pd.read_csv(file_path, dtype=str).fillna("")
    possible_code_names = ["code", "基金代码", "证券代码", "代码"]
    code_col = next((c for c in possible_code_names if c in df.columns), None)
    if code_col is None:
        code_col = df.columns[0]
    codes = [str(code).strip() for code in df[code_col].tolist() if str(code).strip()]
    return list(dict.fromkeys(codes))


def load_cluster_seed_ids(reference_map_csv: str | None, seed_codes: Iterable[str]) -> set[str]:
    if not reference_map_csv or not os.path.exists(reference_map_csv):
        return set()
    if not seed_codes:
        return set()
    df = pd.read_csv(reference_map_csv, dtype=str).fillna("")
    if "code" not in df.columns or "cluster" not in df.columns:
        return set()
    normalized_seeds = build_normalized_code_set(seed_codes)
    seed_clusters: set[str] = set()
    for _, row in df.iterrows():
        code = str(row.get("code", "")).strip()
        if code_in_normalized_set(code, normalized_seeds):
            cluster_val = str(row.get("cluster", "")).strip()
            if cluster_val and cluster_val.lower() != "nan":
                seed_clusters.add(cluster_val)
    return seed_clusters


# Legacy underscore-prefixed aliases (used inside the old monolith). Keep
# available so audit/test code that imported them by name still works.
_normalize_code = normalize_code
_normalized_aliases = normalized_aliases
_build_normalized_code_set = build_normalized_code_set
_code_in_normalized_set = code_in_normalized_set
_load_manual_keep_codes = load_manual_keep_codes
_load_cluster_seed_ids = load_cluster_seed_ids
