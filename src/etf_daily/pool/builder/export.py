"""CSV mapping exporter and metadata writer."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_IDENTITY_COLS = [
    "name",
    "code",
    "cluster",
    "diagnostic_cluster",
    "selection_cluster",
    "selected",
    "reason",
    "n",
]

_FUND_LIST_DROP = {"序号", "基金代码", "基金简称"}
_FUND_LIST_CODE_ALIASES = ("基金代码", "code", "fund_code", "证券代码", "代码", "代码ID")


def _fund_list_extras(fund_list_csv: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Load fund_list rows keyed by code; return (frame with code + extras, extra col names)."""
    path = Path(fund_list_csv)
    raw = pd.read_csv(path, dtype=str).fillna("")
    code_col = next((c for c in _FUND_LIST_CODE_ALIASES if c in raw.columns), None)
    if code_col is None:
        raise KeyError(f"fund_list missing code column: {list(raw.columns)}")
    extra_cols = [c for c in raw.columns if c not in _FUND_LIST_DROP and c != code_col]
    out = pd.DataFrame({"code": raw[code_col].astype(str).str.strip()})
    for c in extra_cols:
        out[c] = raw[c].astype(str).str.strip()
    out = out[out["code"] != ""].drop_duplicates(subset="code", keep="first")
    return out, extra_cols


def export_mapping(
    all_codes: Iterable[str],
    clusters: pd.Series | None,
    selected: Iterable[str],
    reasons: dict[str, str],
    cluster_df: pd.DataFrame | None,
    code_name_map: dict[str, str],
    diagnostics_df: pd.DataFrame | None = None,
    out_csv: str | None = None,
    selected_csv: str | None = None,
    selection_clusters: pd.Series | None = None,
    selection_cluster_df: pd.DataFrame | None = None,
    fund_list_csv: str | None = None,
) -> str:
    """Write slim mapping CSVs: identity cols + fund_list extras (no metric dump).

    ``diagnostics_df`` and ``selection_cluster_df`` are accepted for call-site
    compatibility but are not written to the CSV.
    """
    del diagnostics_df, selection_cluster_df  # unused in slim export
    if out_csv is None:
        raise ValueError("out_csv is required")

    all_codes = list(dict.fromkeys(list(all_codes)))
    selected_set = set(selected)
    df = pd.DataFrame({"code": all_codes})
    if clusters is not None:
        df["cluster"] = df["code"].map(clusters)
    else:
        df["cluster"] = np.nan
    df["diagnostic_cluster"] = df["cluster"]
    if selection_clusters is not None:
        df["selection_cluster"] = df["code"].map(selection_clusters)
    else:
        df["selection_cluster"] = np.nan
    df["selected"] = df["code"].isin(selected_set)
    df["reason"] = df["code"].map(lambda c: reasons.get(c, ""))
    df["name"] = df["code"].map(
        lambda c: code_name_map.get(str(c).strip())
        or code_name_map.get(str(c).strip().upper())
        or (code_name_map.get(str(c)[2:]) if str(c).upper().startswith(("SH", "SZ")) else "")
    )
    df = df[["name", "code", "cluster", "diagnostic_cluster", "selection_cluster", "selected", "reason"]]
    if cluster_df is not None and not cluster_df.empty and "n" in cluster_df.columns:
        cluster_n = cluster_df.set_index("cluster")[["n"]]
        df = df.join(cluster_n, on="cluster")
    else:
        df["n"] = np.nan

    extra_cols: list[str] = []
    if fund_list_csv:
        extras, extra_cols = _fund_list_extras(fund_list_csv)
        df = df.merge(extras, on="code", how="left")
        for c in extra_cols:
            if c in df.columns:
                df[c] = df[c].fillna("")

    df = df[_IDENTITY_COLS + extra_cols]
    df.to_csv(out_csv, index=False)
    print("Wrote mapping to", out_csv)
    if selected_csv:
        df[df["selected"]].to_csv(selected_csv, index=False)
        print("Wrote selected-only mapping to", selected_csv)
    return out_csv
