"""Same-tracking-index deduplication.

For ETFs that track the same underlying index (by ``跟踪指数代码``), keep only
the one with the largest ``流通市值`` and treat the rest as drop codes.

Rationale: correlation-based clustering can fail for low-liquidity ETFs whose
return series have many near-zero / stale-price days.  Two ETFs tracking
``创业板指`` (e.g. ``SZ159971`` vs ``SZ159977``) can appear to have low daily
return correlation purely due to liquidity artifacts and slip through the
near-clone dedup.  Gating by ``跟踪指数代码`` from Wind metadata is a cleaner,
fundamentals-based safeguard.

Source files (xlsx, Wind export):
    ~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/上证ETF_2.xlsx
    ~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/深证ETF_2.xlsx

Required columns (name-substring matched for robustness):
    证券代码          e.g. "510050.SH" / "159915.SZ"
    跟踪指数代码
    基金简称
    流通市值...      circulating market cap column (contains "流通市值")
"""
from __future__ import annotations

import os
from typing import Iterable

import pandas as pd

SAME_INDEX_XLSX_FILES = (
    "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/上证ETF_2.xlsx",
    "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/深证ETF_2.xlsx",
)

# Index codes to skip (money-market placeholders, missing data, etc.)
_BLANK_INDEX_TOKENS = {"", "--", "-", "nan", "none", "null"}


def _normalize_code(raw: str) -> str | None:
    """Convert '510050.SH' → 'SH510050'. Return None if unparseable."""
    if raw is None:
        return None
    s = str(raw).strip().upper()
    if not s:
        return None
    if "." in s:
        num, suf = s.split(".", 1)
        if suf in ("SH", "SZ", "BJ"):
            return f"{suf}{num}"
    if s.startswith(("SH", "SZ", "BJ")):
        return s
    return None


def _pick_column(df: pd.DataFrame, substr: str) -> str | None:
    for c in df.columns:
        if substr in str(c):
            return c
    return None


def _load_same_index_table(xlsx_paths: Iterable[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in xlsx_paths:
        if not os.path.exists(p):
            continue
        df = pd.read_excel(p)
        code_col = _pick_column(df, "证券代码") or df.columns[0]
        idx_col = _pick_column(df, "跟踪指数代码")
        mv_col = _pick_column(df, "流通市值")
        name_col = _pick_column(df, "基金简称") or _pick_column(df, "证券名称")
        if idx_col is None or mv_col is None:
            continue
        sub = pd.DataFrame(
            {
                "code_raw": df[code_col].astype(str),
                "index_code": df[idx_col].astype(str).str.strip(),
                "market_cap": pd.to_numeric(df[mv_col], errors="coerce"),
                "name": df[name_col].astype(str) if name_col else "",
            }
        )
        sub["code"] = sub["code_raw"].map(_normalize_code)
        frames.append(sub.dropna(subset=["code"]))
    if not frames:
        return pd.DataFrame(columns=["code", "index_code", "market_cap", "name"])
    return pd.concat(frames, ignore_index=True)


def compute_same_index_drops(
    xlsx_paths: Iterable[str] = SAME_INDEX_XLSX_FILES,
    *,
    always_keep_codes: Iterable[str] = (),
    verbose: bool = True,
) -> tuple[list[str], pd.DataFrame]:
    """Return (drop_codes, decisions_df).

    For each ``跟踪指数代码`` group with >=2 ETFs, keep the one with the largest
    ``流通市值`` and add the rest to ``drop_codes``. Codes listed in
    ``always_keep_codes`` are never dropped (and, if present in a group, become
    the effective winner).

    ``decisions_df`` has columns: ``index_code``, ``code``, ``name``,
    ``market_cap``, ``decision`` (``keep`` / ``drop``).
    """
    df = _load_same_index_table(xlsx_paths)
    if df.empty:
        return [], df

    keep_set = {str(c).strip().upper() for c in (always_keep_codes or [])}

    df = df[~df["index_code"].str.lower().isin(_BLANK_INDEX_TOKENS)].copy()
    df["market_cap"] = df["market_cap"].fillna(0.0)

    decisions_rows = []
    drop_codes: list[str] = []
    for idx_code, grp in df.groupby("index_code"):
        grp = grp.drop_duplicates(subset="code")
        if len(grp) < 2:
            continue
        # Winner preference: ALWAYS_KEEP first; otherwise max market_cap.
        forced = grp[grp["code"].isin(keep_set)]
        if not forced.empty:
            winner_code = forced.sort_values("market_cap", ascending=False)["code"].iloc[0]
        else:
            winner_code = grp.sort_values("market_cap", ascending=False)["code"].iloc[0]
        for _, row in grp.iterrows():
            decision = "keep" if row["code"] == winner_code else "drop"
            if decision == "drop" and row["code"] in keep_set:
                decision = "keep"  # ALWAYS_KEEP never dropped
            decisions_rows.append(
                {
                    "index_code": idx_code,
                    "code": row["code"],
                    "name": row["name"],
                    "market_cap": float(row["market_cap"]),
                    "decision": decision,
                }
            )
            if decision == "drop":
                drop_codes.append(row["code"])

    decisions_df = pd.DataFrame(decisions_rows)
    drop_codes = list(dict.fromkeys(drop_codes))
    if verbose:
        n_groups = decisions_df["index_code"].nunique() if not decisions_df.empty else 0
        print(
            f"[same_index_dedup] {n_groups} 跟踪指数 groups with ≥2 ETFs; "
            f"dropping {len(drop_codes)} duplicates, "
            f"keeping {n_groups} winners."
        )
    return drop_codes, decisions_df
