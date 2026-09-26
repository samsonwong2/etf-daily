"""A/B audit for ETF pool hierarchical clustering linkage choices.

This module is intentionally read-mostly: it rebuilds candidate pools under a
small fixed set of linkage variants and writes comparison artifacts under
``~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/etf_filter_checks``. It does not overwrite production
``cluster_mapping_selected.csv`` or canonical txt files.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import constants as C
from .audit import _make_cluster_module_shim, compute_forward_metrics, fetch_close_prices, parse_windows
from .clustering import cluster_and_select
from .code_utils import build_normalized_code_set, code_in_normalized_set, load_manual_keep_codes
from .compare import _df_to_md_table
from .config import load_shared_config
from .data_loading import build_code_name_map, load_data, load_fund_list_universe
from .selector import _build_diagnostic_view, _resolve_current_cluster_drop_codes


DEFAULT_OUT_BASE = os.path.join(C.OUT_DIR, "etf_filter_checks")
DEFAULT_WINDOWS = "5,20,60"
NEAR_CLONE_THRESHOLDS = (0.70, 0.80, 0.90)
LARGE_CLUSTER_MIN_N = 20


@dataclass(frozen=True)
class LinkageVariant:
    name: str
    linkage_method: str
    distance_label: str = "1-corr"


VARIANTS = (
    LinkageVariant("ward_1corr", "ward"),
    LinkageVariant("average_corrdist", "average"),
    LinkageVariant("complete_corrdist", "complete"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A/B audit: Ward + 1-corr vs average/complete + correlation distance"
    )
    parser.add_argument("--future-end", default=pd.Timestamp.today().normalize().strftime("%Y-%m-%d"))
    parser.add_argument("--windows", default=DEFAULT_WINDOWS)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--shared-config", default=None)
    parser.add_argument("--inception-cutoff", default=None)
    parser.add_argument(
        "--ignore-cluster-drop-codes",
        action="store_true",
        help="A/B only: ignore ALWAYS_DROP_CLUSTER_CODES whole-cluster seed exclusions.",
    )
    return parser


def _code_key(code: object) -> str:
    return str(code).strip().upper()


def _lookup_name(code: object, name_map: dict[str, str]) -> str:
    raw = str(code).strip()
    if not raw:
        return ""
    for key in (raw, raw.upper(), raw.lower()):
        if key in name_map:
            return str(name_map[key]).strip()
    if raw.upper().startswith(("SH", "SZ")) and raw[2:] in name_map:
        return str(name_map[raw[2:]]).strip()
    return ""


def _theme_key(name: object, code: object) -> str:
    text = str(name or "").strip()
    if not text:
        return _code_key(code)
    cut_positions = [idx for token in ("ETF", "LOF", "基金") if (idx := text.upper().find(token)) > 0]
    if cut_positions:
        text = text[: min(cut_positions)]
    return text.strip() or _code_key(code)


def _pairwise_selected_corr(selected: Iterable[str], corr: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    selected_present = [code for code in dict.fromkeys(selected) if code in corr.index]
    pair_rows: list[dict[str, object]] = []
    for i, left in enumerate(selected_present):
        for right in selected_present[i + 1:]:
            value = float(corr.loc[left, right]) if right in corr.columns else np.nan
            if pd.isna(value):
                continue
            pair_rows.append(
                {
                    "code_left": left,
                    "code_right": right,
                    "corr": value,
                    "abs_corr": abs(value),
                }
            )
    pair_df = pd.DataFrame(pair_rows)
    summary: dict[str, float] = {
        "selected_corr_pair_count": float(len(pair_df)),
        "max_abs_corr_selected": np.nan,
        "mean_abs_corr_selected": np.nan,
    }
    if not pair_df.empty:
        summary["max_abs_corr_selected"] = float(pair_df["abs_corr"].max())
        summary["mean_abs_corr_selected"] = float(pair_df["abs_corr"].mean())
    for threshold in NEAR_CLONE_THRESHOLDS:
        key = f"selected_pairs_abs_corr_gt_{int(round(threshold * 100)):03d}".replace("0", "0p", 1)
        summary[key] = float((pair_df["abs_corr"] > threshold).sum()) if not pair_df.empty else 0.0
    return summary, pair_df.sort_values("abs_corr", ascending=False) if not pair_df.empty else pair_df


def _build_theme_fragmentation(
    clusters: pd.Series,
    selected: Iterable[str],
    name_map: dict[str, str],
) -> tuple[dict[str, float], pd.DataFrame]:
    selected_set = {_code_key(code) for code in selected}
    rows: list[dict[str, object]] = []
    for code, cluster_id in clusters.items():
        name = _lookup_name(code, name_map)
        rows.append(
            {
                "code": _code_key(code),
                "name": name,
                "theme": _theme_key(name, code),
                "cluster": int(cluster_id),
                "selected": _code_key(code) in selected_set,
            }
        )
    code_df = pd.DataFrame(rows)
    if code_df.empty:
        return {
            "theme_groups_ge2": 0.0,
            "fragmented_theme_groups": 0.0,
            "fragmented_theme_ratio": np.nan,
            "avg_clusters_per_theme": np.nan,
            "max_clusters_per_theme": np.nan,
        }, pd.DataFrame()

    theme_df = code_df.groupby("theme").agg(
        n=("code", "size"),
        cluster_count=("cluster", "nunique"),
        selected_count=("selected", "sum"),
        clusters=("cluster", lambda x: "|".join(str(v) for v in sorted(set(x)))),
        members_preview=("code", lambda x: "|".join(list(x)[:8])),
    ).reset_index()
    theme_df = theme_df[theme_df["n"] >= 2].copy()
    fragmented = theme_df[theme_df["cluster_count"] > 1].copy()
    summary = {
        "theme_groups_ge2": float(len(theme_df)),
        "fragmented_theme_groups": float(len(fragmented)),
        "fragmented_theme_ratio": float(len(fragmented) / len(theme_df)) if len(theme_df) else np.nan,
        "avg_clusters_per_theme": float(theme_df["cluster_count"].mean()) if len(theme_df) else np.nan,
        "max_clusters_per_theme": float(theme_df["cluster_count"].max()) if len(theme_df) else np.nan,
    }
    return summary, fragmented.sort_values(["cluster_count", "n"], ascending=[False, False])


def _summarize_forward(detail_df: pd.DataFrame, windows: list[int]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for window in windows:
        ret_col = f"fwd_ret_{window}d"
        drawdown_col = f"fwd_drawdown_{window}d"
        if ret_col in detail_df.columns:
            returns = pd.to_numeric(detail_df[ret_col], errors="coerce")
            summary[f"fwd_ret_{window}d_mean"] = float(returns.mean()) if returns.notna().any() else np.nan
            summary[f"fwd_ret_{window}d_median"] = float(returns.median()) if returns.notna().any() else np.nan
            summary[f"fwd_ret_{window}d_positive_ratio"] = float((returns > 0).mean()) if returns.notna().any() else np.nan
        if drawdown_col in detail_df.columns:
            drawdowns = pd.to_numeric(detail_df[drawdown_col], errors="coerce")
            summary[f"fwd_drawdown_{window}d_mean"] = float(drawdowns.mean()) if drawdowns.notna().any() else np.nan
    return summary


def _format_pct(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    if pd.isna(numeric):
        return "-"
    return f"{numeric * 100:.2f}%"


def _load_base_inputs(shared_config_path: str | None, inception_cutoff: str | None) -> dict[str, object]:
    cfg = load_shared_config(shared_config_path)
    provider_uri = cfg["provider_uri"]
    test_period = tuple(cfg["test_period"])
    fund_list_csv = cfg["fund_list_csv"]
    if inception_cutoff is None:
        inception_cutoff = test_period[1]

    _, full_name_map = load_fund_list_universe(fund_list_csv)
    effective_manual_excludes = list(dict.fromkeys(list(C.ALWAYS_DROP_CODES)))
    manual_keep_codes = load_manual_keep_codes(C.MANUAL_KEEP_CODES_FILE)
    force_keep_codes = list(dict.fromkeys(C.ALWAYS_KEEP_CODES + manual_keep_codes))
    code_map, excluded_codes, selected_codes = build_code_name_map(
        fund_list_csv,
        C.EXCLUDE_TYPES,
        inception_cutoff,
        manual_excludes=effective_manual_excludes,
        force_keep_codes=force_keep_codes,
    )
    full_name_map.update(code_map)

    close, vol = load_data(
        selected_codes,
        provider_uri=provider_uri,
        test_period=test_period,
        excluded_codes=excluded_codes.union(effective_manual_excludes),
    )
    return {
        "cfg": cfg,
        "test_period": test_period,
        "name_map": full_name_map,
        "close": close,
        "vol": vol,
        "force_keep_codes": force_keep_codes,
    }


def _run_variant(
    variant: LinkageVariant,
    base: dict[str, object],
    *,
    ignore_cluster_drop_codes: bool = False,
) -> dict[str, object]:
    close = base["close"]
    vol = base["vol"]
    force_keep_codes = base["force_keep_codes"]
    name_map = base["name_map"]

    if ignore_cluster_drop_codes:
        cluster_drop_codes = set()
    else:
        cluster_drop_codes = _resolve_current_cluster_drop_codes(
            close,
            vol,
            C.ALWAYS_DROP_CLUSTER_CODES,
            linkage_method=variant.linkage_method,
        )
    normalized_cluster_drops = build_normalized_code_set(cluster_drop_codes)
    keep_cols = [code for code in close.columns if not code_in_normalized_set(code, normalized_cluster_drops)]
    close_variant = close[keep_cols]
    vol_variant = vol[keep_cols] if vol is not None else None

    diag_view = _build_diagnostic_view(close_variant, vol_variant)
    if not diag_view["eligible_codes"]:
        raise RuntimeError(f"No eligible codes for variant {variant.name}")
    returns = diag_view["returns"]
    cluster_vol = diag_view["cluster_vol"]
    recent_valid_days = diag_view["recent_valid_days"]
    corr = diag_view["corr"]
    overlap = diag_view["overlap"]

    _, clusters, cluster_df, selected, reasons = cluster_and_select(
        returns,
        cluster_vol,
        t=C.DIST_T,
        per_cluster=C.PER_CLUSTER,
        rule=C.RULE,
        force_keep=force_keep_codes,
        target_count=C.TARGET_SELECTED_COUNT,
        history_days=recent_valid_days.reindex(diag_view["eligible_codes"]).fillna(0),
        corr=corr,
        overlap=overlap,
        linkage_method=variant.linkage_method,
    )
    selected_clusters = clusters.reindex(selected).dropna().astype(int)
    large_unselected = cluster_df[
        (pd.to_numeric(cluster_df["n"], errors="coerce") >= LARGE_CLUSTER_MIN_N)
        & (~cluster_df["cluster"].isin(set(selected_clusters.tolist())))
    ].copy()
    corr_summary, pair_df = _pairwise_selected_corr(selected, corr)
    theme_summary, theme_df = _build_theme_fragmentation(clusters, selected, name_map)

    detail_rows = []
    for code in selected:
        detail_rows.append(
            {
                "variant": variant.name,
                "code": code,
                "name": _lookup_name(code, name_map),
                "cluster": int(clusters.get(code)) if code in clusters.index else np.nan,
                "reason": reasons.get(code, ""),
            }
        )
    detail_df = pd.DataFrame(detail_rows)

    summary = {
        "variant": variant.name,
        "linkage_method": variant.linkage_method,
        "distance": variant.distance_label,
        "ignore_cluster_drop_codes": bool(ignore_cluster_drop_codes),
        "cluster_drop_code_count": len(cluster_drop_codes),
        "universe_count": int(returns.shape[1]),
        "cluster_count": int(len(cluster_df)),
        "selected_count": int(len(selected)),
        "selected_cluster_count": int(selected_clusters.nunique()),
        "singleton_cluster_count": int((pd.to_numeric(cluster_df["n"], errors="coerce") == 1).sum()),
        "median_cluster_size": float(pd.to_numeric(cluster_df["n"], errors="coerce").median()),
        "max_cluster_size": int(pd.to_numeric(cluster_df["n"], errors="coerce").max()),
        "large_unselected_cluster_count": int(len(large_unselected)),
    }
    summary.update(corr_summary)
    summary.update(theme_summary)

    pair_df = pair_df.head(100).copy()
    if not pair_df.empty:
        pair_df.insert(0, "variant", variant.name)
        pair_df["name_left"] = pair_df["code_left"].map(lambda code: _lookup_name(code, name_map))
        pair_df["name_right"] = pair_df["code_right"].map(lambda code: _lookup_name(code, name_map))
    theme_df = theme_df.head(100).copy()
    if not theme_df.empty:
        theme_df.insert(0, "variant", variant.name)

    return {
        "variant": variant,
        "selected": selected,
        "summary": summary,
        "detail_df": detail_df,
        "pair_df": pair_df,
        "theme_df": theme_df,
        "large_unselected_df": large_unselected.assign(variant=variant.name),
    }


def _attach_forward_metrics(results: list[dict[str, object]], shared_config_path: str | None, future_end: str, windows: list[int]) -> None:
    selected_union = sorted({code for result in results for code in result["selected"]})
    if not selected_union:
        return
    cluster_module = _make_cluster_module_shim(shared_config_path)
    future_start = (pd.Timestamp(cluster_module.TEST_PERIOD[1]) - pd.Timedelta(days=max(windows) + 20)).strftime("%Y-%m-%d")
    future_close = fetch_close_prices(cluster_module, selected_union, future_start, future_end)
    future_df = compute_forward_metrics(future_close, cluster_module.TEST_PERIOD[1], windows)
    for result in results:
        detail_df = result["detail_df"].merge(future_df, on="code", how="left")
        result["detail_df"] = detail_df
        result["summary"].update(_summarize_forward(detail_df, windows))


def _write_report(
    out_dir: Path,
    summary_df: pd.DataFrame,
    windows: list[int],
    *,
    ignore_cluster_drop_codes: bool = False,
) -> Path:
    display_cols = [
        "variant",
        "selected_count",
        "cluster_count",
        "singleton_cluster_count",
        "fragmented_theme_groups",
        "max_abs_corr_selected",
        "selected_pairs_abs_corr_gt_0p70",
    ]
    for window in windows:
        display_cols.append(f"fwd_ret_{window}d_mean")
        display_cols.append(f"fwd_ret_{window}d_positive_ratio")
    display_cols = [col for col in display_cols if col in summary_df.columns]
    display = summary_df[display_cols].copy()
    for col in display.columns:
        if col.startswith("fwd_ret_") or col.endswith("positive_ratio"):
            display[col] = display[col].map(_format_pct)
    if "max_abs_corr_selected" in display.columns:
        display["max_abs_corr_selected"] = display["max_abs_corr_selected"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
    report_path = out_dir / "linkage_ab_report.md"
    lines = [
        "# Linkage A/B Report",
        "",
        "Variants: `ward_1corr` (production baseline), `average_corrdist`, `complete_corrdist`.",
        f"ALWAYS_DROP_CLUSTER_CODES whole-cluster exclusions: {'ignored' if ignore_cluster_drop_codes else 'enabled'}.",
        "",
        "## Summary",
        "",
        _df_to_md_table(display),
        "",
        "## Artifacts",
        "",
        "| File | Content |",
        "| --- | --- |",
        "| `linkage_ab_summary.csv` | variant-level selected count, fragmentation, near-clone, forward metrics |",
        "| `linkage_ab_selected.csv` | selected constituents by variant with forward metrics |",
        "| `linkage_ab_near_clone_pairs.csv` | top selected pairs by absolute correlation |",
        "| `linkage_ab_theme_fragmentation.csv` | theme name groups split across multiple clusters |",
        "| `linkage_ab_large_unselected_clusters.csv` | clusters with n>=20 and no selected representative |",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_linkage_ab(args: argparse.Namespace) -> int:
    windows = parse_windows(args.windows)
    date_tag = pd.Timestamp(args.future_end).strftime("%Y%m%d")
    default_dir_name = f"linkage_ab_no_cluster_drop_{date_tag}" if args.ignore_cluster_drop_codes else f"linkage_ab_{date_tag}"
    out_dir = Path(args.out_dir or os.path.join(DEFAULT_OUT_BASE, default_dir_name))
    out_dir.mkdir(parents=True, exist_ok=True)

    base = _load_base_inputs(args.shared_config, args.inception_cutoff)
    results = []
    for variant in VARIANTS:
        print(f"[INFO] Running linkage variant: {variant.name}")
        results.append(
            _run_variant(
                variant,
                base,
                ignore_cluster_drop_codes=bool(args.ignore_cluster_drop_codes),
            )
        )

    _attach_forward_metrics(results, args.shared_config, args.future_end, windows)

    summary_df = pd.DataFrame([result["summary"] for result in results])
    selected_df = pd.concat([result["detail_df"] for result in results], ignore_index=True)
    pair_frames = [result["pair_df"] for result in results if not result["pair_df"].empty]
    theme_frames = [result["theme_df"] for result in results if not result["theme_df"].empty]
    large_unselected_frames = [result["large_unselected_df"] for result in results if not result["large_unselected_df"].empty]

    summary_df.to_csv(out_dir / "linkage_ab_summary.csv", index=False)
    selected_df.to_csv(out_dir / "linkage_ab_selected.csv", index=False)
    pd.concat(pair_frames, ignore_index=True).to_csv(out_dir / "linkage_ab_near_clone_pairs.csv", index=False) if pair_frames else pd.DataFrame().to_csv(out_dir / "linkage_ab_near_clone_pairs.csv", index=False)
    pd.concat(theme_frames, ignore_index=True).to_csv(out_dir / "linkage_ab_theme_fragmentation.csv", index=False) if theme_frames else pd.DataFrame().to_csv(out_dir / "linkage_ab_theme_fragmentation.csv", index=False)
    pd.concat(large_unselected_frames, ignore_index=True).to_csv(out_dir / "linkage_ab_large_unselected_clusters.csv", index=False) if large_unselected_frames else pd.DataFrame().to_csv(out_dir / "linkage_ab_large_unselected_clusters.csv", index=False)
    for result in results:
        result["detail_df"].to_csv(out_dir / f"selected_{result['variant'].name}.csv", index=False)

    report_path = _write_report(
        out_dir,
        summary_df,
        windows,
        ignore_cluster_drop_codes=bool(args.ignore_cluster_drop_codes),
    )
    print(f"[INFO] Wrote A/B summary: {out_dir / 'linkage_ab_summary.csv'}")
    print(f"[INFO] Wrote A/B report: {report_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_linkage_ab(args)


if __name__ == "__main__":
    raise SystemExit(main())
