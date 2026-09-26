"""HRP cluster → overlay mode router.

Design: huangtuo-master-design-20260812-171530-hrp-cluster-router.md
C17 trade: huangtuo-master-design-20260813-104222-c17-shallow-up-trade.md
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

MODE_RECOVERY_UP = "recovery_up"
MODE_SHALLOW_UP = "shallow_up"
MODE_GRIND_SHADOW = "grind_shadow"
MODE_IMPULSE_SHADOW = "impulse_shadow"
MODE_DEFAULT = "default"

# C17 (exemplar SH513080) may change trade labels; C8 recovery_up stays diagnostic.
TRADE_MODES = frozenset({MODE_SHALLOW_UP})
# HRP recovery_up label (C8 etc.). HTML RU marks are full-pool predicate-based.
DIAG_MODES = frozenset({MODE_RECOVERY_UP})

# Exemplar → mode; whole HRP cluster inherits the mode.
DEFAULT_EXEMPLAR_SEEDS: dict[str, str] = {
    "SH515220": MODE_RECOVERY_UP,
    "SH513080": MODE_SHALLOW_UP,
    "SH513650": MODE_GRIND_SHADOW,
    "SH588710": MODE_IMPULSE_SHADOW,
    "SH515050": MODE_IMPULSE_SHADOW,
}

DEFAULT_MEMBERSHIP_CSV: Path | None = None


def normalize_code(code: str) -> str:
    c = str(code).strip().upper()
    if c.endswith(".SH"):
        return "SH" + c[:-3]
    if c.endswith(".SZ"):
        return "SZ" + c[:-3]
    return c


def load_cluster_membership(path: Path | str) -> pd.DataFrame:
    """Load representatives CSV → one row per code with cluster id."""
    path = Path(path)
    df = pd.read_csv(path)
    if "code" not in df.columns or "cluster" not in df.columns:
        raise ValueError(f"membership CSV missing code/cluster: {path}")
    out = df.copy()
    out["code"] = out["code"].map(normalize_code)
    # Prefer representative row when duplicates exist
    if "is_representative" in out.columns:
        out = out.sort_values(
            ["code", "is_representative"], ascending=[True, False]
        )
    out = out.drop_duplicates("code", keep="first")
    cols = ["code", "cluster"]
    for optional in ("name", "group", "direction", "n_members", "is_representative"):
        if optional in out.columns:
            cols.append(optional)
    return out[cols].reset_index(drop=True)


def paint_cluster_modes(
    membership: pd.DataFrame,
    exemplar_seeds: Mapping[str, str] | None = None,
) -> dict[int, str]:
    """Map cluster_id → mode from exemplar seeds. Conflicts raise."""
    seeds = {
        normalize_code(k): str(v)
        for k, v in (exemplar_seeds or DEFAULT_EXEMPLAR_SEEDS).items()
    }
    code_to_cluster = {
        str(r.code): int(r.cluster) for r in membership.itertuples(index=False)
    }
    cluster_mode: dict[int, str] = {}
    for code, mode in seeds.items():
        if code not in code_to_cluster:
            continue
        cid = code_to_cluster[code]
        prev = cluster_mode.get(cid)
        if prev is not None and prev != mode:
            raise ValueError(
                f"cluster {cid} mode conflict: {prev} vs {mode} (exemplar {code})"
            )
        cluster_mode[cid] = mode
    return cluster_mode


def code_mode_map(
    membership: pd.DataFrame,
    cluster_mode: Mapping[int, str] | None = None,
    exemplar_seeds: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Map code → overlay mode (default if cluster unpainted)."""
    cm = (
        dict(cluster_mode)
        if cluster_mode is not None
        else paint_cluster_modes(membership, exemplar_seeds)
    )
    out: dict[str, str] = {}
    for r in membership.itertuples(index=False):
        out[str(r.code)] = cm.get(int(r.cluster), MODE_DEFAULT)
    return out


def recovery_up_codes(mode_by_code: Mapping[str, str]) -> set[str]:
    return {c for c, m in mode_by_code.items() if m == MODE_RECOVERY_UP}


def shallow_up_codes(mode_by_code: Mapping[str, str]) -> set[str]:
    return {c for c, m in mode_by_code.items() if m == MODE_SHALLOW_UP}


def mode_allows_trade_overlay(mode: str) -> bool:
    return str(mode) in TRADE_MODES


def mode_allows_diagnostic_marks(mode: str) -> bool:
    return str(mode) in DIAG_MODES


def router_snapshot(
    membership: pd.DataFrame,
    *,
    exemplar_seeds: Mapping[str, str] | None = None,
    source_path: str | None = None,
) -> dict[str, Any]:
    cm = paint_cluster_modes(membership, exemplar_seeds)
    by_code = code_mode_map(membership, cm)
    counts: dict[str, int] = {}
    for m in by_code.values():
        counts[m] = counts.get(m, 0) + 1
    return {
        "source_path": source_path,
        "n_codes": len(by_code),
        "n_clusters": int(membership["cluster"].nunique()),
        "cluster_mode": {str(k): v for k, v in sorted(cm.items())},
        "mode_counts": counts,
        "recovery_up_codes": sorted(recovery_up_codes(by_code)),
        "shallow_up_codes": sorted(shallow_up_codes(by_code)),
        "exemplar_seeds": {
            normalize_code(k): v
            for k, v in (exemplar_seeds or DEFAULT_EXEMPLAR_SEEDS).items()
        },
    }
