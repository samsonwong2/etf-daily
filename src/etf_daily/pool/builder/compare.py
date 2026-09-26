"""
compare_cluster_pool.py
-----------------------
对比两个 cluster_mapping_selected.txt，生成新增/移除标的的详细报告。

典型用法：
    python 2_filter/compare_cluster_pool.py \
        --old ~/etf-daily-output/Documents/stock/data/qlib_data/all_fund_data/instruments/cluster_mapping_selected_20250101_20251231.txt \
        --new ~/etf-daily-output/Documents/stock/data/qlib_data/all_fund_data/instruments/cluster_mapping_selected.txt \
        --cluster-map ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean/temp/cluster_mapping.csv

    # 也可以同时指定 pool summary（production_cluster_pool_eval 的输出）
    python 2_filter/compare_cluster_pool.py \
        --old ~/etf-daily-output/Documents/stock/data/qlib_data/all_fund_data/instruments/cluster_mapping_selected_20250101_20251231.txt \
        --new ~/etf-daily-output/Documents/stock/data/qlib_data/all_fund_data/instruments/cluster_mapping_selected.txt \
        --cluster-map ~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean/temp/cluster_mapping.csv \
        --pool-summary ~/etf-daily-output/Documents/stock/gitee/skfolio_csi300/etf_strategy_cs/cluster_pool_rolling_eval/outputs/production_eval_20260422_172354/production_pool_summary.csv \
        --pool-meta ~/etf-daily-output/Documents/stock/gitee/skfolio_csi300/etf_strategy_cs/cluster_pool_rolling_eval/outputs/production_eval_20260422_172354/applied_pool_metadata.json

输出：
    ~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/cluster_pool_compare_reports/compare_<timestamp>.md
    ~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/cluster_pool_compare_reports/compare_<timestamp>_added.csv
    ~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/cluster_pool_compare_reports/compare_<timestamp>_removed.csv
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path("~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/cluster_pool_compare_reports")


def load_codes(path: str) -> list[str]:
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".csv":
        return load_codes_from_csv(path)
    codes = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                codes.append(parts[0].upper())
    return codes


def load_codes_from_csv(path: str, code_column: str = "code") -> list[str]:
    df = pd.read_csv(path)
    col = code_column if code_column in df.columns else df.columns[0]
    return (
        df[col]
        .astype(str)
        .str.strip()
        .str.upper()
        .loc[lambda s: s != ""]
        .tolist()
    )


def get_info(code: str, cmap: pd.DataFrame) -> dict:
    row = cmap[cmap["code_upper"] == code.upper()]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {
        "name": r.get("name", ""),
        "cluster": r.get("cluster", ""),
        "selected": r.get("selected", ""),
        "reason": r.get("reason", ""),
        "mean_return": r.get("mean_return", np.nan),
        "mean_volatility": r.get("mean_volatility", np.nan),
        "mean_corr": r.get("mean_corr", np.nan),
        "available_history_days": r.get("available_history_days", np.nan),
        "recent_valid_days": r.get("recent_valid_days", np.nan),
        "avg_volume": r.get("avg_volume", np.nan),
    }


def fmt(val, scale=1, decimals=3, suffix=""):
    if pd.isna(val):
        return "-"
    return f"{val * scale:.{decimals}f}{suffix}"


def build_detail_df(codes: list[str], cmap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code in codes:
        info = get_info(code, cmap)
        rows.append(
            {
                "代码": code,
                "名称": str(info.get("name", ""))[:20],
                "聚类": str(info.get("cluster", "")),
                "近1年均日收益%": fmt(info.get("mean_return", np.nan), scale=100, decimals=3),
                "日波动率%": fmt(info.get("mean_volatility", np.nan), scale=100, decimals=3),
                "均相关": fmt(info.get("mean_corr", np.nan), scale=1, decimals=3),
                "历史天数": fmt(info.get("available_history_days", np.nan), scale=1, decimals=0),
                "原因": str(info.get("reason", "")),
            }
        )
    return pd.DataFrame(rows)


def _escape_md_table_cell(value: object) -> str:
    """Escape cell text so embedded pipes/newlines do not break Markdown tables."""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")
    return text.replace("|", "&#124;")


def _df_to_md_table(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a Markdown table string."""
    if df.empty:
        return "（无）"
    header = "| " + " | ".join(_escape_md_table_cell(c) for c in df.columns) + " |"
    sep_row = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [
        "| " + " | ".join(_escape_md_table_cell(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([header, sep_row] + rows)


def build_report(
    old_path: str,
    new_path: str,
    cluster_map_path: str | None,
    pool_summary_path: str | None,
    pool_meta_path: str | None,
    old_label: str,
    new_label: str,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    old_codes = load_codes(old_path)
    new_codes = load_codes(new_path)
    old_set = set(old_codes)
    new_set = set(new_codes)

    added = sorted(new_set - old_set)
    removed = sorted(old_set - new_set)
    kept = sorted(old_set & new_set)

    # Load cluster mapping
    cmap = pd.DataFrame()
    if cluster_map_path and Path(cluster_map_path).exists():
        cmap = pd.read_csv(cluster_map_path)
        cmap["code_upper"] = cmap["code"].str.upper()

    added_df = build_detail_df(added, cmap) if not cmap.empty else pd.DataFrame({"代码": added})
    removed_df = build_detail_df(removed, cmap) if not cmap.empty else pd.DataFrame({"代码": removed})

    meta: dict = {}
    lines = []

    lines.append(f"# 标的池对比报告")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 基本信息")
    lines.append("")
    lines.append(f"| 项目 | 内容 |")
    lines.append(f"| --- | --- |")
    lines.append(f"| 旧池标签 | `{old_label}` |")
    lines.append(f"| 旧池文件 | `{old_path}` |")
    lines.append(f"| 新池标签 | `{new_label}` |")
    lines.append(f"| 新池文件 | `{new_path}` |")
    lines.append(f"| 旧池标的数 | {len(old_set)} |")
    lines.append(f"| 新池标的数 | {len(new_set)} |")
    lines.append(f"| 新增 | **{len(added)}** |")
    lines.append(f"| 移除 | **{len(removed)}** |")
    lines.append(f"| 保留 | {len(kept)} |")
    lines.append("")

    # Pool selection summary
    if pool_meta_path and Path(pool_meta_path).exists():
        with open(pool_meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        chosen = meta.get("chosen_pool", "-")
        sm = meta.get("selection_metrics", {})
        lines.append("## 评估框架选池结果")
        lines.append("")
        lines.append(f"| 项目 | 值 |")
        lines.append(f"| --- | --- |")
        lines.append(f"| 胜出候选池 | **{chosen}** |")
        lines.append(f"| 评估窗口 | {sm.get('eval_start','')} ~ {sm.get('eval_end','')} |")
        lines.append(f"| 未来均前向收益 | {sm.get('future_avg_forward_return', 0)*100:.2f}% |")
        lines.append(f"| 未来正收益占比 | {sm.get('future_positive_ratio', 0)*100:.2f}% |")
        lines.append(f"| 综合排名分 | rank_sum = {sm.get('rank_sum','-')} |")
        lines.append("")

    if pool_summary_path and Path(pool_summary_path).exists():
        summary = pd.read_csv(pool_summary_path)
        cols = [
            "pool_name", "pool_size", "covered_clusters",
            "future_avg_forward_return", "future_positive_ratio",
            "future_return_rank", "future_corr_rank", "positive_ratio_rank",
        ]
        display_cols = [c for c in cols if c in summary.columns]
        summary_display = summary[display_cols].copy()
        if "future_avg_forward_return" in summary_display:
            summary_display["future_avg_forward_return"] = summary_display["future_avg_forward_return"].map(
                lambda x: f"{x*100:.2f}%"
            )
        if "future_positive_ratio" in summary_display:
            summary_display["future_positive_ratio"] = summary_display["future_positive_ratio"].map(
                lambda x: f"{x*100:.1f}%"
            )
        # Rename columns for display
        rename_map = {
            "pool_name": "候选池", "pool_size": "标的数", "covered_clusters": "覆盖聚类",
            "future_avg_forward_return": "未来均收益", "future_positive_ratio": "正收益占比",
            "future_return_rank": "收益排名", "future_corr_rank": "相关排名",
            "positive_ratio_rank": "正收益排名",
        }
        summary_display = summary_display.rename(columns={k: v for k, v in rename_map.items() if k in summary_display})
        lines.append("## 各候选池评估对比")
        lines.append("")
        lines.append(_df_to_md_table(summary_display))
        lines.append("")

    # Added
    lines.append(f"## 新增 {len(added)} 只（新池有、旧池无）")
    lines.append("")
    if not added_df.empty:
        lines.append(_df_to_md_table(added_df))
    else:
        lines.append("（无）")
    lines.append("")

    # Theme summary for added
    if not cmap.empty and added:
        theme_rows = []
        for code in added:
            info = get_info(code, cmap)
            theme_rows.append({
                "聚类": str(info.get("cluster", "")),
                "mean_return": info.get("mean_return", np.nan),
            })
        tdf = pd.DataFrame(theme_rows)
        if not tdf.empty and "mean_return" in tdf.columns:
            cluster_theme = (
                tdf.groupby("聚类")
                .agg(标的数=("mean_return", "count"), 均日收益=("mean_return", "mean"))
                .sort_values("均日收益", ascending=False)
                .reset_index()
            )
            cluster_theme["均日收益%"] = cluster_theme["均日收益"].map(lambda x: f"{x*100:.3f}%")
            lines.append("### 新增主题分布（按聚类均日收益排序）")
            lines.append("")
            lines.append(_df_to_md_table(cluster_theme[["聚类", "标的数", "均日收益%"]]))
        lines.append("")

    # Removed
    lines.append(f"## 移除 {len(removed)} 只（旧池有、新池无）")
    lines.append("")
    if not removed_df.empty:
        lines.append(_df_to_md_table(removed_df))
    else:
        lines.append("（无）")
    lines.append("")

    # Removed reason summary
    if not cmap.empty and removed:
        reasons = removed_df["原因"].value_counts() if "原因" in removed_df.columns else pd.Series()
        if not reasons.empty:
            lines.append("### 移除原因分布")
            lines.append("")
            lines.append("| 原因 | 数量 |")
            lines.append("| --- | --- |")
            for reason, cnt in reasons.items():
                r_short = str(reason)[:80] if reason else "（无原因/规则未命中）"
                lines.append(f"| {r_short} | {cnt} |")
        lines.append("")

    # Kept
    lines.append(f"## 保留 {len(kept)} 只")
    lines.append("")
    # Group into rows of 8 per line as a code block
    chunks = [kept[i:i+8] for i in range(0, len(kept), 8)]
    lines.append("```")
    for chunk in chunks:
        lines.append("  ".join(chunk))
    lines.append("```")
    lines.append("")

    # ── 分析解读 ──────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## 分析解读")
    lines.append("")

    # 整体变化
    change_pct = len(added) / max(len(old_set), 1) * 100
    lines.append("### 整体变化")
    lines.append("")
    lines.append(f"变化幅度 **{change_pct:.0f}%**（{len(added)} 只替换），"
                 f"旧池 {len(old_set)} 只 → 新池 {len(new_set)} 只，保留 {len(kept)} 只不变。")
    lines.append("")

    # 为什么选出了胜出池
    if pool_meta_path and Path(pool_meta_path).exists() and pool_summary_path and Path(pool_summary_path).exists():
        chosen = meta.get("chosen_pool", "-")
        sm = meta.get("selection_metrics", {})
        summary2 = pd.read_csv(pool_summary_path)

        # Compute rank_sum per row if not present
        rank_cols = ["future_return_rank", "future_corr_rank", "positive_ratio_rank"]
        if "rank_sum" not in summary2.columns and all(c in summary2.columns for c in rank_cols):
            summary2["rank_sum"] = summary2[rank_cols].sum(axis=1)

        lines.append(f"### 为什么选出了 {chosen}")
        lines.append("")
        tbl_header = "| 候选池 | 未来均收益 | 正收益占比 | 综合排名 |"
        tbl_sep = "| --- | --- | --- | --- |"
        lines.append(tbl_header)
        lines.append(tbl_sep)
        for _, row in summary2.sort_values("future_avg_forward_return", ascending=False).iterrows():
            pname = str(row.get("pool_name", "-"))
            ret = f"{row.get('future_avg_forward_return', 0)*100:.1f}%"
            pos = f"{row.get('future_positive_ratio', 0)*100:.2f}%"
            rs_val = row.get("rank_sum", None)
            rs = f"rank_sum = {rs_val:.0f}" if rs_val is not None and not pd.isna(rs_val) else "-"
            if pname == chosen:
                lines.append(f"| **{pname}（胜出）** | **{ret}** | **{pos}** | **{rs}（最低）** |")
            else:
                lines.append(f"| {pname} | {ret} | {pos} | {rs} |")
        lines.append("")
        lines.append(f"`{chosen}` 在评估窗口 {sm.get('eval_start','')} ~ {sm.get('eval_end','')} "
                     f"的前向平均收益高达 {sm.get('future_avg_forward_return', 0)*100:.1f}%，"
                     f"远超其他候选池，因此被 apply 为新 canonical。")
        lines.append("")

    # 新增主题：按聚类收益排序，取 top 5 聚类并举例
    if not cmap.empty and added:
        lines.append("### 新增标的：主题分布")
        lines.append("")
        lines.append("主要是过去评估窗口内涨幅居前的板块代表：")
        lines.append("")
        theme_rows2 = []
        for code in added:
            info = get_info(code, cmap)
            theme_rows2.append({
                "代码": code,
                "名称": str(info.get("name", ""))[:16],
                "聚类": str(info.get("cluster", "")),
                "mean_return": info.get("mean_return", np.nan),
            })
        tdf2 = pd.DataFrame(theme_rows2)
        if not tdf2.empty and "mean_return" in tdf2.columns:
            cluster_agg = (
                tdf2.groupby("聚类")
                .agg(标的数=("mean_return", "count"), 均日收益=("mean_return", "mean"))
                .sort_values("均日收益", ascending=False)
                .reset_index()
            )
            # For each cluster, pick top-3 codes by return as examples
            top_per_cluster = (
                tdf2.sort_values("mean_return", ascending=False)
                .groupby("聚类")
                .apply(lambda g: "、".join(g.head(3)["代码"] + " " + g.head(3)["名称"]))
                .rename("代表标的（收益前3）")
                .reset_index()
            )
            cluster_agg = cluster_agg.merge(top_per_cluster, on="聚类", how="left")
            cluster_agg["均日收益%"] = cluster_agg["均日收益"].map(lambda x: f"{x*100:.3f}%")
            display_agg = cluster_agg[["聚类", "标的数", "均日收益%", "代表标的（收益前3）"]]
            lines.append(_df_to_md_table(display_agg))
        lines.append("")

    # 移除原因分析
    if not cmap.empty and removed and not removed_df.empty and "原因" in removed_df.columns:
        lines.append("### 移除标的：原因与收益分析")
        lines.append("")
        # Merge mean_return into removed_df for analysis
        removed_with_ret = removed_df.copy()
        removed_with_ret["_mean_return"] = [
            get_info(c, cmap).get("mean_return", np.nan) for c in removed_with_ret["代码"]
        ]
        reason_groups = removed_with_ret.groupby("原因")
        tbl_lines = ["| 移除原因 | 数量 | 均日收益% | 代表标的（收益最低前3） |",
                     "| --- | --- | --- | --- |"]
        for reason, grp in sorted(reason_groups, key=lambda x: -len(x[1])):
            cnt = len(grp)
            avg_ret = grp["_mean_return"].mean()
            avg_ret_str = f"{avg_ret*100:.3f}%" if not pd.isna(avg_ret) else "-"
            examples_grp = grp.sort_values("_mean_return").head(3)
            examples = "、".join(
                examples_grp["代码"] + " " + examples_grp["名称"].str[:8]
            )
            r_short = str(reason)[:60] if reason else "（无原因/规则未命中）"
            tbl_lines.append(f"| {r_short} | {cnt} | {avg_ret_str} | {examples} |")
        lines.extend(tbl_lines)
        lines.append("")

    # 核心结论
    lines.append("### 核心结论")
    lines.append("")
    if not cmap.empty and added:
        added_ret_vals = [get_info(c, cmap).get("mean_return", np.nan) for c in added]
        added_ret_vals = [v for v in added_ret_vals if not pd.isna(v)]
        avg_added_str = f"{np.mean(added_ret_vals)*100:.3f}%" if added_ret_vals else "-"
    else:
        avg_added_str = "-"
    if not cmap.empty and removed:
        removed_ret_vals = [get_info(c, cmap).get("mean_return", np.nan) for c in removed]
        removed_ret_vals = [v for v in removed_ret_vals if not pd.isna(v)]
        avg_removed_str = f"{np.mean(removed_ret_vals)*100:.3f}%" if removed_ret_vals else "-"
    else:
        avg_removed_str = "-"

    if avg_added_str != "-" and avg_removed_str != "-":
        lines.append(f"- **新池均日收益显著高于移除标的**：新增标的均日收益 {avg_added_str}，移除标的均日收益 {avg_removed_str}")
    lines.append(f"- **{change_pct:.0f}% 的大换血**说明评估窗口内市场风格发生了显著轮动，新池向高动量板块集中")
    if pool_meta_path and Path(pool_meta_path).exists():
        chosen_str = meta.get("chosen_pool", "胜出池")
        lines.append(f"- **{chosen_str}** 基于 one-shot 前向评估自动择优，不依赖人工判断")
    lines.append("- **注意动量暴露风险**：新池集中于近期强势板块，若市场风格切换（如大盘价值反攻），超额可能快速回撤")
    lines.append("")

    report_text = "\n".join(lines)
    return report_text, added_df, removed_df


def main():
    parser = argparse.ArgumentParser(
        description="对比两个 cluster_mapping_selected.txt，生成新增/移除标的报告"
    )
    parser.add_argument("--old", required=True, help="旧版 txt 路径（或旧标签，如 20250101_20251231）")
    parser.add_argument("--new", required=True, help="新版 txt 路径（默认为当前 canonical）")
    parser.add_argument(
        "--cluster-map",
        default="~/etf-daily-output/gitee/skfolio_csi300/etf_strategy_clean/temp/cluster_mapping.csv",
        help="cluster_mapping.csv 路径（含名称、聚类等信息）",
    )
    parser.add_argument(
        "--pool-summary",
        default=None,
        help="production_pool_summary.csv 路径（可选，显示各候选池评估对比）",
    )
    parser.add_argument(
        "--pool-meta",
        default=None,
        help="applied_pool_metadata.json 路径（可选，显示胜出池信息）",
    )
    parser.add_argument(
        "--old-label",
        default=None,
        help="旧版标签（用于报告标题，默认从文件名推断）",
    )
    parser.add_argument(
        "--new-label",
        default=None,
        help="新版标签（用于报告标题，默认从文件名推断）",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help=f"输出目录，默认: {OUTPUT_DIR}",
    )
    args = parser.parse_args()

    old_path = args.old
    new_path = args.new

    # Auto-detect canonical path for shorthand inputs
    instruments_dir = Path("~/etf-daily-output/Documents/stock/data/qlib_data/all_fund_data/instruments")
    if not Path(old_path).exists():
        candidate = instruments_dir / f"cluster_mapping_selected_{old_path}.txt"
        if candidate.exists():
            old_path = str(candidate)
        else:
            print(f"[ERROR] 旧版文件不存在: {old_path}", file=sys.stderr)
            sys.exit(1)
    if not Path(new_path).exists():
        candidate = instruments_dir / f"cluster_mapping_selected_{new_path}.txt"
        if candidate.exists():
            new_path = str(candidate)
        else:
            print(f"[ERROR] 新版文件不存在: {new_path}", file=sys.stderr)
            sys.exit(1)

    old_label = args.old_label or Path(old_path).stem.replace("cluster_mapping_selected_", "").replace("cluster_mapping_selected", "current")
    new_label = args.new_label or Path(new_path).stem.replace("cluster_mapping_selected_", "").replace("cluster_mapping_selected", "current")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"compare_{old_label}_vs_{new_label}_{ts}.md"
    added_csv = output_dir / f"compare_{old_label}_vs_{new_label}_{ts}_added.csv"
    removed_csv = output_dir / f"compare_{old_label}_vs_{new_label}_{ts}_removed.csv"

    report_text, added_df, removed_df = build_report(
        old_path=old_path,
        new_path=new_path,
        cluster_map_path=args.cluster_map,
        pool_summary_path=args.pool_summary,
        pool_meta_path=args.pool_meta,
        old_label=old_label,
        new_label=new_label,
    )

    report_path.write_text(report_text, encoding="utf-8")
    if not added_df.empty:
        added_df.to_csv(added_csv, index=False, encoding="utf-8-sig")
    if not removed_df.empty:
        removed_df.to_csv(removed_csv, index=False, encoding="utf-8-sig")

    print(report_text)
    print(f"\n报告（Markdown）已保存到: {report_path}")
    if not added_df.empty:
        print(f"新增明细CSV: {added_csv}")
    if not removed_df.empty:
        print(f"移除明细CSV: {removed_csv}")


if __name__ == "__main__":
    main()
