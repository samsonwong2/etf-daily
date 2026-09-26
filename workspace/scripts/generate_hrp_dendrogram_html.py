#!/usr/bin/env python3
"""Generate skfolio HRP-CVaR dendrogram HTML for holdings and cluster pool.

Requires ``pyqlib`` and ``skfolio`` (tested with conda env ``py312``).
"""

from __future__ import annotations

import argparse
import html
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from skfolio import RiskMeasure
from skfolio.optimization import HierarchicalRiskParity
from skfolio.preprocessing import prices_to_returns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fund_pool_builder.pool_builder.constants import CLUSTER_LOOKBACK_DAYS, DIST_T
from fund_pool_builder.pool_builder.data_loading import load_data
from runtime_paths import CLUSTER_MAPPING_SELECTED_TXT, FUND_LIST_CSV, QLIB_PROVIDER_URI
from workspace.scripts.generate_daily_mu_position_report import load_fund_name_map, normalize_code

@dataclass
class UniverseResult:
    name: str
    requested_codes: tuple[str, ...]
    used_codes: tuple[str, ...] = ()
    dropped_codes: tuple[str, ...] = ()
    error: str | None = None
    figures: list[tuple[str, object]] = field(default_factory=list)
    cluster_reps: pd.DataFrame | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build HRP-CVaR dendrogram HTML for holdings and cluster pool."
    )
    parser.add_argument(
        "--holdings-csv",
        type=Path,
        default=None,
        help=(
            "Live holdings CSV with a code column. "
            "Optional: omit to skip the holdings universe and only plot the cluster pool."
        ),
    )
    parser.add_argument(
        "--cluster-mapping-txt",
        type=Path,
        default=CLUSTER_MAPPING_SELECTED_TXT,
        help="Canonical cluster_mapping_selected.txt path.",
    )
    parser.add_argument(
        "--asof-date",
        required=True,
        help="End date for return window (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=CLUSTER_LOOKBACK_DAYS,
        help="Calendar lookback for qlib close panel.",
    )
    parser.add_argument(
        "--provider-uri",
        default=str(QLIB_PROVIDER_URI),
        help="qlib provider URI.",
    )
    parser.add_argument(
        "--fund-list-csv",
        type=Path,
        default=FUND_LIST_CSV,
        help="fund_list.csv used to map instrument codes to Chinese names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output HTML path.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help=(
            "强制簇数（criterion=maxclust）；缺省用 DIST_T 距离阈值自然切簇"
        ),
    )
    parser.add_argument(
        "--dist-t",
        type=float,
        default=DIST_T,
        help=(
            "自然切簇的 Ward(1-corr) 距离阈值（默认 0.4=corr>=0.6 并簇）；"
            "越小簇越多越细（如 0.3），越大越粗（如 0.5）。"
            "若同时给 --n-clusters，则 n-clusters 优先"
        ),
    )
    parser.add_argument(
        "--rep-window",
        type=int,
        default=20,
        help="代表选取的涨跌窗口（交易日，默认 20）",
    )
    parser.add_argument(
        "--min-rep-move",
        type=float,
        default=MIN_REP_MOVE_PCT,
        help=(
            "方向组出代表的最小 |区间涨跌%%|（默认 5.0；0=关闭门槛，每组都出代表）"
        ),
    )
    args = parser.parse_args()
    if not (0.0 < args.dist_t < 2.0):
        parser.error("--dist-t 需在 (0, 2) 范围内（1-corr 距离合法范围）")
    return args


def load_cluster_mapping_codes(path: Path | str) -> tuple[str, ...]:
    codes: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.split("\t")[0].strip()
            if token:
                codes.append(normalize_code(token))
    seen: set[str] = set()
    ordered: list[str] = []
    for code in codes:
        if code and code not in seen:
            seen.add(code)
            ordered.append(code)
    return tuple(ordered)


def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, dtype=str, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=str, encoding="latin-1")


def load_holdings_codes(path: Path) -> tuple[str, ...]:
    frame = _read_csv_with_fallback(path)
    if "code" not in frame.columns:
        raise ValueError(f"Holdings CSV missing code column: {path}")
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in frame["code"].tolist():
        code = normalize_code(raw)
        if code and code not in seen:
            seen.add(code)
            ordered.append(code)
    return tuple(ordered)


def build_name_lookup(
    fund_list_csv: Path,
    holdings_csv: Path | None = None,
) -> dict[str, str]:
    lookup = load_fund_name_map(fund_list_csv)
    if holdings_csv is None or not holdings_csv.exists():
        return lookup

    frame = _read_csv_with_fallback(holdings_csv)
    if "code" not in frame.columns or "name" not in frame.columns:
        return lookup

    for _, row in frame.iterrows():
        code = normalize_code(row["code"])
        name = "" if pd.isna(row["name"]) else str(row["name"]).strip()
        if code and name:
            lookup.setdefault(code, name)
    return lookup


def rename_returns_to_display_names(
    returns: pd.DataFrame,
    name_map: dict[str, str],
) -> pd.DataFrame:
    renamed = returns.copy()
    assigned: set[str] = set()
    labels: list[str] = []
    for code in renamed.columns:
        base = (name_map.get(str(code)) or str(code)).strip() or str(code)
        label = base
        if label in assigned:
            label = f"{base}({code})"
        assigned.add(label)
        labels.append(label)
    renamed.columns = labels
    return renamed


def build_test_period(asof_date: str, lookback_days: int) -> tuple[str, str]:
    end = datetime.strptime(asof_date, "%Y-%m-%d").date()
    start = end - timedelta(days=int(lookback_days) + 30)
    return start.isoformat(), end.isoformat()


def prepare_returns(close: pd.DataFrame, requested_codes: Iterable[str]) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    requested = tuple(normalize_code(code) for code in requested_codes)
    available = {normalize_code(col): col for col in close.columns}
    present = [available[code] for code in requested if code in available]
    missing = tuple(code for code in requested if code not in available)

    if not present:
        return pd.DataFrame(), tuple(), missing

    prices = close[present].copy()
    prices.columns = [normalize_code(col) for col in prices.columns]
    returns = prices_to_returns(prices)
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.fillna(returns.mean())

    var_series = returns.var(axis=0)
    zero_var = tuple(col for col, value in var_series.items() if value <= 0 or np.isnan(value))
    if zero_var:
        returns = returns.drop(columns=list(zero_var))

    used = tuple(returns.columns.tolist())
    dropped = tuple(dict.fromkeys([*missing, *zero_var]))
    return returns, used, dropped


# 组内最大 |retN| 低于该阈值 → 该方向组标「无明确方向」，不出代表。
MIN_REP_MOVE_PCT = 5.0


def compute_cluster_representatives(
    returns: pd.DataFrame,
    close: pd.DataFrame,
    *,
    name_map: dict[str, str],
    n_clusters: int | None = None,
    rep_window: int = 20,
    min_rep_move: float = MIN_REP_MOVE_PCT,
    dist_t: float = DIST_T,
) -> pd.DataFrame:
    """Cluster codes by (1 - corr) Ward linkage; pick max |retN| per direction group.

    Distance/linkage matches fund_pool_builder production (ward); the cutting
    threshold defaults to production ``DIST_T`` and can be overridden via
    ``dist_t`` (ignored when ``n_clusters`` forces maxclust).
    Each cluster is split into up/down groups by the sign of the cumulative
    move over the last ``rep_window`` trading days; each group gets its own
    representative (max |retN|). Groups whose max |retN| is below
    ``min_rep_move`` are marked ``no_direction`` and yield no representative.
    Returns one row per member with an ``is_representative`` flag.
    """
    codes = list(returns.columns)
    corr = returns.corr()
    dist = (1.0 - corr).clip(lower=0.0).fillna(1.0)
    np.fill_diagonal(dist.values, 0.0)
    condensed = squareform(dist.values, checks=False)
    z = linkage(condensed, method="ward")
    if n_clusters is not None and int(n_clusters) >= 2:
        labels = fcluster(z, t=int(n_clusters), criterion="maxclust")
    else:
        labels = fcluster(z, t=float(dist_t), criterion="distance")
    clusters = pd.Series(labels, index=codes, name="cluster")

    px = close.copy()
    px.columns = [normalize_code(col) for col in px.columns]
    w = max(int(rep_window), 1)
    base = px.iloc[-(w + 1)] if len(px) > w else px.iloc[0]
    ret_n = (px.iloc[-1] / base - 1.0) * 100.0

    records: list[dict[str, object]] = []
    for cid, members in clusters.groupby(clusters).groups.items():
        members = [str(m) for m in members]
        r_n = ret_n.reindex(members).astype(float)
        # Split cluster into up/down groups by sign of the window move.
        # 0/NaN joins the side with the larger |move| (default up).
        up_members = [c for c in members if pd.notna(r_n.get(c)) and r_n[c] > 0]
        dn_members = [c for c in members if pd.notna(r_n.get(c)) and r_n[c] < 0]
        flat = [c for c in members if c not in up_members and c not in dn_members]
        if flat:
            up_best = max((abs(float(r_n[c])) for c in up_members), default=0.0)
            dn_best = max((abs(float(r_n[c])) for c in dn_members), default=0.0)
            (up_members if up_best >= dn_best else dn_members).extend(flat)
        for group_label, grp_members, direction_zh in (
            ("up", up_members, "涨"),
            ("down", dn_members, "跌"),
        ):
            if not grp_members:
                continue
            g = r_n.reindex(grp_members)
            best = float(g.abs().max())
            has_dir = float(min_rep_move) <= 0 or best >= float(min_rep_move)
            rep = str(g.abs().idxmax()) if has_dir else None
            for code in grp_members:
                value = float(g[code])
                records.append(
                    {
                        "cluster": int(cid),
                        "group": group_label,
                        "code": code,
                        "name": name_map.get(code, code),
                        f"ret{w}_pct": value,
                        "direction": direction_zh,
                        "is_representative": code == rep,
                        "group_status": "representative" if has_dir else "no_direction",
                        "n_members": len(members),
                    }
                )
    frame = pd.DataFrame(records)
    abs_col = frame[f"ret{w}_pct"].abs()
    frame = frame.assign(_abs=abs_col)
    frame = frame.sort_values(
        ["cluster", "group", "is_representative", "_abs"],
        ascending=[True, True, False, False],
    ).drop(columns=["_abs"])
    return frame.reset_index(drop=True)


def fit_hrp_dendrograms(
    returns: pd.DataFrame,
    *,
    layout_width: int,
    layout_height: int,
) -> list[tuple[str, object]]:
    model = HierarchicalRiskParity(
        risk_measure=RiskMeasure.CVAR,
        portfolio_params={"name": "HRP-CVaR-Ward-Pearson"},
    )
    model.fit(returns)

    fig_no_heat = model.hierarchical_clustering_estimator_.plot_dendrogram(heatmap=False)
    fig_heat = model.hierarchical_clustering_estimator_.plot_dendrogram()

    for title, fig in (
        ("Dendrogram (heatmap=False)", fig_no_heat),
        ("Dendrogram (heatmap=True)", fig_heat),
    ):
        fig.update_layout(
            title=title,
            width=layout_width,
            height=layout_height,
            showlegend=False,
        )
    return [
        ("Dendrogram (heatmap=False)", fig_no_heat),
        ("Dendrogram (heatmap=True)", fig_heat),
    ]


def build_universe_result(
    name: str,
    codes: tuple[str, ...],
    *,
    provider_uri: str,
    test_period: tuple[str, str],
    layout_width: int,
    layout_height: int,
    name_map: dict[str, str],
    n_clusters: int | None = None,
    rep_window: int = 20,
    min_rep_move: float = MIN_REP_MOVE_PCT,
    dist_t: float = DIST_T,
) -> UniverseResult:
    result = UniverseResult(name=name, requested_codes=codes)
    if len(codes) < 2:
        result.error = f"{name}: fewer than 2 instruments requested."
        return result

    close, _ = load_data(codes, provider_uri=provider_uri, test_period=test_period)
    returns, used, dropped = prepare_returns(close, codes)
    result.used_codes = used
    result.dropped_codes = dropped

    if len(used) < 2:
        result.error = (
            f"{name}: fewer than 2 instruments with valid returns "
            f"(requested={len(codes)}, used={len(used)}, dropped={len(dropped)})."
        )
        return result

    result.figures = fit_hrp_dendrograms(
        rename_returns_to_display_names(returns, name_map),
        layout_width=layout_width,
        layout_height=layout_height,
    )
    result.cluster_reps = compute_cluster_representatives(
        returns,
        close,
        name_map=name_map,
        n_clusters=n_clusters,
        rep_window=rep_window,
        min_rep_move=min_rep_move,
        dist_t=dist_t,
    )
    return result


def figure_to_html(fig: object, *, include_plotlyjs: bool | str) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)


def _fmt_ret(v: object) -> str:
    try:
        if v is None or pd.isna(v):
            return "N/A"
        return f"{float(v):+.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def render_cluster_reps_html(
    frame: pd.DataFrame,
    rep_window: int,
    min_rep_move: float = MIN_REP_MOVE_PCT,
) -> str:
    """Render 代表清单 table: one row per cluster × direction group."""
    ret_col = f"ret{rep_window}_pct"
    group_zh = {"up": "涨组", "down": "跌组"}
    rows: list[str] = []
    for (cid, glabel), grp in frame.groupby(["cluster", "group"], sort=True):
        has_rep = bool(grp["is_representative"].any())
        members = grp.sort_values(ret_col, key=lambda s: s.abs(), ascending=False)
        detail = "、".join(
            f"{r.code}({html.escape(str(r.name))} {_fmt_ret(getattr(r, ret_col))})"
            for r in members.itertuples()
        )
        if has_rep:
            rep = grp[grp["is_representative"]].iloc[0]
            rep_cells = (
                f"<td><b>{html.escape(str(rep['code']))}</b></td>"
                f"<td>{html.escape(str(rep['name']))}</td>"
                f"<td>{_fmt_ret(rep[ret_col])}</td>"
            )
        else:
            rep_cells = (
                "<td colspan='3'>无明确方向"
                f"（组内最大|涨跌| &lt; {float(min_rep_move):.1f}%）</td>"
            )
        rows.append(
            "<tr>"
            f"<td>{int(cid)}</td>"
            f"<td>{group_zh.get(str(glabel), str(glabel))}</td>"
            f"{rep_cells}"
            f"<td>{len(members)}</td>"
            f"<td>{detail}</td>"
            "</tr>"
        )
    return (
        "<table border='1' cellspacing='0' cellpadding='4' "
        "style='border-collapse:collapse;font-size:13px;'>"
        "<thead><tr>"
        "<th>簇</th><th>方向组</th><th>代表代码</th><th>名称</th>"
        f"<th>近{rep_window}日涨跌</th><th>组内成员数</th>"
        "<th>成员明细（按 |涨跌| 降序）</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def render_html(
    *,
    output_path: Path,
    asof_date: str,
    test_period: tuple[str, str],
    universes: list[UniverseResult],
    rep_window: int = 20,
    min_rep_move: float = MIN_REP_MOVE_PCT,
    dist_t: float = DIST_T,
) -> None:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections: list[str] = []
    include_plotlyjs: bool | str = "cdn"

    sections.append(
        "<!DOCTYPE html>"
        "<html lang=\"zh-CN\"><head><meta charset=\"utf-8\" />"
        "<title>HRP Dendrogram Report</title>"
        "<style>"
        "body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#1f2937;}"
        "h1{margin-bottom:8px;} h2{margin-top:32px;} .meta{color:#4b5563;}"
        ".section{margin-top:28px;} .error{color:#b91c1c;}"
        "</style></head><body>"
    )
    sections.append("<h1>HRP 层次聚类 Dendrogram</h1>")
    sections.append(
        "<p class=\"meta\">"
        f"生成时间: {html.escape(generated_at)}<br />"
        f"数据区间: {html.escape(test_period[0])} ~ {html.escape(test_period[1])} "
        f"(asof={html.escape(asof_date)})"
        "</p>"
    )

    for universe in universes:
        sections.append(f"<h2>{html.escape(universe.name)}</h2>")
        sections.append(
            "<p class=\"meta\">"
            f"请求标的数: {len(universe.requested_codes)}; "
            f"有效标的数: {len(universe.used_codes)}; "
            f"剔除标的数: {len(universe.dropped_codes)}"
            "</p>"
        )
        if universe.dropped_codes:
            dropped_text = ", ".join(universe.dropped_codes)
            sections.append(f"<p class=\"meta\">剔除: {html.escape(dropped_text)}</p>")
        if universe.error:
            sections.append(f"<p class=\"error\">{html.escape(universe.error)}</p>")
            continue

        if universe.cluster_reps is not None and not universe.cluster_reps.empty:
            n_clusters = int(universe.cluster_reps["cluster"].nunique())
            sections.append(
                f"<div class=\"section\"><h3>聚类代表（近{rep_window}日涨跌最大；"
                f"共 {n_clusters} 簇）</h3>"
            )
            sections.append(
                f"<p class=\"meta\">切簇=Ward(1-corr) dist_t={float(dist_t):.2f}；"
                "簇内按涨跌方向拆组，代表=组内 |区间涨跌幅| 最大的一只；"
                f"组内最大|涨跌| &lt; {float(min_rep_move):.1f}% 标「无明确方向」不出代表</p>"
            )
            sections.append(
                render_cluster_reps_html(
                    universe.cluster_reps, rep_window, min_rep_move=min_rep_move
                )
            )
            sections.append("</div>")

        for subtitle, fig in universe.figures:
            sections.append(f"<div class=\"section\"><h3>{html.escape(subtitle)}</h3>")
            sections.append(figure_to_html(fig, include_plotlyjs=include_plotlyjs))
            sections.append("</div>")
            include_plotlyjs = False

    sections.append("</body></html>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(sections), encoding="utf-8")


def main() -> int:
    args = parse_args()
    test_period = build_test_period(args.asof_date, args.lookback_days)

    holdings_csv = args.holdings_csv
    if holdings_csv is not None and not holdings_csv.exists():
        raise FileNotFoundError(f"Holdings CSV not found: {holdings_csv}")

    holdings_codes = (
        load_holdings_codes(holdings_csv) if holdings_csv is not None else None
    )
    pool_codes = load_cluster_mapping_codes(args.cluster_mapping_txt)
    name_map = build_name_lookup(args.fund_list_csv, holdings_csv)

    pool_universe_name = "待选池 cluster_mapping_selected"
    universes: list[UniverseResult] = []
    if holdings_codes is not None:
        universes.append(
            build_universe_result(
                "持仓 ETF",
                holdings_codes,
                provider_uri=args.provider_uri,
                test_period=test_period,
                layout_width=1200,
                layout_height=500,
                name_map=name_map,
                n_clusters=args.n_clusters,
                rep_window=args.rep_window,
                min_rep_move=args.min_rep_move,
                dist_t=args.dist_t,
            )
        )
    else:
        print("No --holdings-csv given; skipping holdings universe.")
    universes.append(
        build_universe_result(
            pool_universe_name,
            pool_codes,
            provider_uri=args.provider_uri,
            test_period=test_period,
            layout_width=2400,
            layout_height=700,
            name_map=name_map,
            n_clusters=args.n_clusters,
            rep_window=args.rep_window,
            min_rep_move=args.min_rep_move,
            dist_t=args.dist_t,
        )
    )

    render_html(
        output_path=args.output,
        asof_date=args.asof_date,
        test_period=test_period,
        universes=universes,
        rep_window=args.rep_window,
        min_rep_move=args.min_rep_move,
        dist_t=args.dist_t,
    )
    print(f"Wrote {args.output}")
    for universe in universes:
        status = "ok" if not universe.error else universe.error
        print(
            f"{universe.name}: requested={len(universe.requested_codes)} "
            f"used={len(universe.used_codes)} dropped={len(universe.dropped_codes)} "
            f"figures={len(universe.figures)} status={status}"
        )

    # 待选清单 CSV（待选池）：与 HTML 同目录，便于下游直接读取。
    asof_tag = args.asof_date.replace("-", "")
    if args.n_clusters:
        cut_tag = f"_k{int(args.n_clusters)}"
    elif abs(float(args.dist_t) - float(DIST_T)) > 1e-9:
        # Non-default distance cut (e.g. 0.8 → _d080); keep untagged for production DIST_T.
        cut_tag = f"_d{int(round(float(args.dist_t) * 100.0)):03d}"
    else:
        cut_tag = ""
    for universe in universes:
        if universe.name != pool_universe_name:
            continue
        if universe.cluster_reps is None or universe.cluster_reps.empty:
            continue
        csv_path = (
            args.output.parent / f"cluster_representatives_{asof_tag}{cut_tag}.csv"
        )
        universe.cluster_reps.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
