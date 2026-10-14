#!/usr/bin/env python3
"""Generate skfolio HRP-CVaR dendrogram HTML for holdings and cluster pool.

Requires ``pyqlib`` and ``skfolio`` (tested with conda env ``py312``).
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from skfolio import RiskMeasure
from skfolio.optimization import HierarchicalRiskParity
from skfolio.preprocessing import prices_to_returns

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from etf_daily.pool.builder.constants import CLUSTER_LOOKBACK_DAYS, DIST_T
from etf_daily.pool.builder.data_loading import load_data
from etf_daily.paths import CLUSTER_MAPPING_SELECTED_TXT, FUND_LIST_CSV, QLIB_PROVIDER_URI
from etf_daily.lib.generate_daily_mu_position_report import load_fund_name_map, normalize_code

@dataclass
class UniverseResult:
    name: str
    requested_codes: tuple[str, ...]
    used_codes: tuple[str, ...] = ()
    dropped_codes: tuple[str, ...] = ()
    error: str | None = None
    figures: list[tuple[str, object]] = field(default_factory=list)
    cluster_reps: pd.DataFrame | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
            "方向组出极端动量代表的最小 |区间涨跌%%|（默认 5.0；0=关闭门槛）"
        ),
    )
    parser.add_argument(
        "--min-corr",
        type=float,
        default=DEFAULT_MIN_CORR,
        help=(
            "簇内最小两两相关硬门槛（默认 0.55）；低于则拆成连通分量。"
            "设为 0 关闭拆分。"
        ),
    )
    args = parser.parse_args(argv)
    if not (0.0 < args.dist_t < 2.0):
        parser.error("--dist-t 需在 (0, 2) 范围内（1-corr 距离合法范围）")
    if args.min_corr < 0.0 or args.min_corr >= 1.0:
        parser.error("--min-corr 需在 [0, 1) 范围内（0=关闭）")
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


# 组内最大 |retN| 低于该阈值 → 该方向组标「无明确方向」，不出极端动量代表。
MIN_REP_MOVE_PCT = 5.0
# 簇内两两相关低于此阈值则拆簇（0=关闭）。
DEFAULT_MIN_CORR = 0.55
DOMAIN_BOND = "bond"
DOMAIN_OVERSEAS = "overseas"
DOMAIN_DOMESTIC = "domestic"
DOMAIN_ORDER = (DOMAIN_BOND, DOMAIN_OVERSEAS, DOMAIN_DOMESTIC)

_OVERSEAS_NAME_RE = re.compile(
    r"纳指|标普|道琼斯|日经|德国|法国|巴西|沙特|东南亚|"
    r"美国|QDII|标普油气|道指"
)


def classify_cluster_domain(code: str, name: str) -> str:
    """Map an ETF to bond / overseas / domestic for domain-first clustering."""
    from etf_daily.scripts.backtest_fig12_six_states_pool import is_bond

    code_u = normalize_code(code)
    name_s = name or ""
    if is_bond(code_u, name_s):
        return DOMAIN_BOND
    # 港股通 / 恒生 / 中概 stay domestic even if they trade offshore listings
    if re.search(r"港股通|恒生|恒指|中概", name_s) and not re.search(
        r"纳指|标普|道琼斯|日经|德国|法国|巴西|沙特|东南亚|美国|QDII", name_s
    ):
        return DOMAIN_DOMESTIC
    if _OVERSEAS_NAME_RE.search(name_s) or "QDII" in name_s.upper():
        return DOMAIN_OVERSEAS
    return DOMAIN_DOMESTIC


def _aligned_corr(returns: pd.DataFrame, corr: pd.DataFrame | None) -> pd.DataFrame:
    """Use ``corr`` when it covers every return column; otherwise compute it."""
    cols = list(returns.columns)
    if corr is not None and all(col in corr.index and col in corr.columns for col in cols):
        return corr.loc[cols, cols]
    return returns.corr()


def ward_labels(
    returns: pd.DataFrame,
    *,
    dist_t: float = DIST_T,
    n_clusters: int | None = None,
    corr: pd.DataFrame | None = None,
) -> pd.Series:
    """Ward(1-corr) cluster labels for columns of ``returns``."""
    codes = list(returns.columns)
    if len(codes) == 0:
        return pd.Series(dtype=int)
    if len(codes) == 1:
        return pd.Series([1], index=codes, dtype=int)
    corr = _aligned_corr(returns, corr)
    dist = (1.0 - corr).clip(lower=0.0).fillna(1.0)
    np.fill_diagonal(dist.values, 0.0)
    condensed = squareform(dist.values, checks=False)
    z = linkage(condensed, method="ward")
    if n_clusters is not None and int(n_clusters) >= 2:
        labels = fcluster(z, t=min(int(n_clusters), len(codes)), criterion="maxclust")
    else:
        labels = fcluster(z, t=float(dist_t), criterion="distance")
    return pd.Series(labels, index=codes, name="cluster", dtype=int)


def split_by_min_corr(
    labels: pd.Series,
    corr: pd.DataFrame,
    *,
    min_corr: float = DEFAULT_MIN_CORR,
) -> pd.Series:
    """Split clusters whose min pairwise corr is below ``min_corr``.

    Within a weak cluster, keep connected components where an edge exists when
    pairwise corr >= ``min_corr``; isolated names become singleton clusters.
    ``min_corr <= 0`` disables splitting.
    """
    if float(min_corr) <= 0 or labels.empty:
        return labels.astype(int)

    out = labels.copy().astype(int)
    next_id = int(out.max()) + 1 if len(out) else 1
    for cid, members in labels.groupby(labels).groups.items():
        ms = [str(m) for m in members]
        if len(ms) < 2:
            continue
        sub = corr.reindex(index=ms, columns=ms)
        vals = []
        for i, a in enumerate(ms):
            for b in ms[i + 1 :]:
                v = sub.loc[a, b]
                if pd.notna(v):
                    vals.append(float(v))
        if not vals or min(vals) >= float(min_corr):
            continue
        # Union-find / BFS components on corr >= min_corr
        adj: dict[str, list[str]] = {c: [] for c in ms}
        for i, a in enumerate(ms):
            for b in ms[i + 1 :]:
                v = sub.loc[a, b]
                if pd.notna(v) and float(v) >= float(min_corr):
                    adj[a].append(b)
                    adj[b].append(a)
        seen: set[str] = set()
        components: list[list[str]] = []
        for c in ms:
            if c in seen:
                continue
            stack = [c]
            seen.add(c)
            comp = [c]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
                        comp.append(v)
            components.append(comp)
        if len(components) <= 1:
            # Fully disconnected below threshold → each name its own cluster
            # except we already have components of size 1 each
            pass
        # First component keeps original cid; rest get new ids
        for i, comp in enumerate(components):
            new_cid = int(cid) if i == 0 else next_id
            if i > 0:
                next_id += 1
            for code in comp:
                out.loc[code] = new_cid
    return out.astype(int)


def corr_medoid(members: list[str], corr: pd.DataFrame) -> str:
    """Code with highest mean correlation to other members (self excluded)."""
    ms = [str(m) for m in members]
    if len(ms) == 1:
        return ms[0]
    sub = corr.reindex(index=ms, columns=ms)
    best_code = ms[0]
    best_score = -np.inf
    for c in ms:
        others = [x for x in ms if x != c]
        score = float(pd.to_numeric(sub.loc[c, others], errors="coerce").mean())
        if np.isnan(score):
            score = -np.inf
        if score > best_score:
            best_score = score
            best_code = c
    return best_code


def domain_ward_labels(
    returns: pd.DataFrame,
    name_map: dict[str, str],
    *,
    dist_t: float = DIST_T,
    n_clusters: int | None = None,
    min_corr: float = DEFAULT_MIN_CORR,
    corr: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Domain-first Ward labels + min-corr split.

    Returns ``(labels, domains, corr)`` with globally renumbered cluster ids.
    When ``n_clusters`` is set, it is applied only inside the domestic domain
    (largest); bond/overseas still use ``dist_t``.
    """
    codes = [str(c) for c in returns.columns]
    corr = _aligned_corr(returns, corr)
    domains = pd.Series(
        {c: classify_cluster_domain(c, name_map.get(c, c)) for c in codes},
        name="domain",
    )
    pieces: list[pd.Series] = []
    next_id = 1
    for domain in DOMAIN_ORDER:
        members = [c for c in codes if domains[c] == domain]
        if not members:
            continue
        sub = returns[members]
        # n_clusters only for domestic when forcing maxclust
        n_c = n_clusters if domain == DOMAIN_DOMESTIC else None
        lab = ward_labels(sub, dist_t=dist_t, n_clusters=n_c, corr=corr.loc[members, members])
        lab = split_by_min_corr(lab, corr, min_corr=min_corr)
        # renumber contiguous from next_id
        mapping: dict[int, int] = {}
        for old in sorted(lab.unique()):
            mapping[int(old)] = next_id
            next_id += 1
        lab = lab.map(lambda x, m=mapping: m[int(x)]).astype(int)
        pieces.append(lab)
    if not pieces:
        empty = pd.Series(dtype=int, name="cluster")
        return empty, domains, corr
    labels = pd.concat(pieces).reindex(codes).astype(int)
    labels.name = "cluster"
    return labels, domains, corr


def compute_cluster_representatives(
    returns: pd.DataFrame,
    close: pd.DataFrame,
    *,
    name_map: dict[str, str],
    n_clusters: int | None = None,
    rep_window: int = 20,
    min_rep_move: float = MIN_REP_MOVE_PCT,
    dist_t: float = DIST_T,
    min_corr: float = DEFAULT_MIN_CORR,
    labels: pd.Series | None = None,
    corr: pd.DataFrame | None = None,
    domains: pd.Series | None = None,
) -> pd.DataFrame:
    """Domain-first Ward clusters; medoid representative + extreme movers.

    * ``is_representative``: correlation medoid of the whole cluster (one per cluster).
    * ``is_extreme_mover``: max |retN| within each up/down group when |move| >= min_rep_move.
    """
    if labels is None or corr is None or domains is None:
        labels, domains, corr = domain_ward_labels(
            returns,
            name_map,
            dist_t=dist_t,
            n_clusters=n_clusters,
            min_corr=min_corr,
            corr=corr,
        )
    if labels.empty:
        return pd.DataFrame()

    px = close.copy()
    px.columns = [normalize_code(col) for col in px.columns]
    w = max(int(rep_window), 1)
    base = px.iloc[-(w + 1)] if len(px) > w else px.iloc[0]
    ret_n = (px.iloc[-1] / base - 1.0) * 100.0

    medoids: dict[int, str] = {}
    for cid, members in labels.groupby(labels).groups.items():
        medoids[int(cid)] = corr_medoid([str(m) for m in members], corr)

    records: list[dict[str, object]] = []
    for cid, members_idx in labels.groupby(labels).groups.items():
        members = [str(m) for m in members_idx]
        r_n = ret_n.reindex(members).astype(float)
        medoid = medoids[int(cid)]
        domain = str(domains.get(members[0], DOMAIN_DOMESTIC))
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
            extreme = str(g.abs().idxmax()) if has_dir else None
            for code in grp_members:
                value = float(g[code])
                records.append(
                    {
                        "cluster": int(cid),
                        "domain": domain,
                        "group": group_label,
                        "code": code,
                        "name": name_map.get(code, code),
                        f"ret{w}_pct": value,
                        "direction": direction_zh,
                        "is_representative": code == medoid,
                        "is_extreme_mover": code == extreme,
                        "group_status": "representative" if has_dir else "no_direction",
                        "n_members": len(members),
                    }
                )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    abs_col = frame[f"ret{w}_pct"].abs()
    frame = frame.assign(_abs=abs_col)
    frame = frame.sort_values(
        ["cluster", "group", "is_representative", "is_extreme_mover", "_abs"],
        ascending=[True, True, False, False, False],
    ).drop(columns=["_abs"])
    return frame.reset_index(drop=True)



def plot_ward_dendrogram(
    returns: pd.DataFrame,
    labels: pd.Series,
    name_map: dict[str, str],
    *,
    layout_width: int,
    layout_height: int,
    title: str = "Ward(1-corr) dendrogram (matches CSV cut)",
    corr: pd.DataFrame | None = None,
) -> object:
    """Plotly dendrogram from the same Ward linkage used for cluster labels."""
    import plotly.graph_objects as go

    codes = [str(c) for c in returns.columns]
    if len(codes) < 2:
        fig = go.Figure()
        fig.update_layout(title=title, width=layout_width, height=layout_height)
        return fig
    display = []
    assigned: set[str] = set()
    for code in codes:
        base = (name_map.get(code) or code).strip() or code
        label = base if base not in assigned else f"{base}({code})"
        assigned.add(label)
        display.append(label)
    corr = _aligned_corr(returns, corr)
    dist = (1.0 - corr).clip(lower=0.0).fillna(1.0)
    np.fill_diagonal(dist.values, 0.0)
    condensed = squareform(dist.values, checks=False)
    z = linkage(condensed, method="ward")
    dendro = scipy_dendrogram(z, labels=display, no_plot=True, color_threshold=None)
    fig = go.Figure()
    icoord = np.array(dendro["icoord"])
    dcoord = np.array(dendro["dcoord"])
    # Color leaves by final cluster id
    lab_by_display = {
        display[i]: int(labels.loc[codes[i]]) for i in range(len(codes))
    }
    # Draw U-links
    for xs, ys in zip(icoord, dcoord):
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="#64748b", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    # Leaf labels on x
    leaf_pos = {lab: i * 10 for i, lab in enumerate(dendro["ivl"])}
    fig.update_layout(
        title=title,
        width=layout_width,
        height=layout_height,
        showlegend=False,
        xaxis=dict(
            tickmode="array",
            tickvals=[leaf_pos[lab] + 5 for lab in dendro["ivl"]],
            ticktext=[
                f"{lab} [C{lab_by_display.get(lab, '?')}]" if lab in lab_by_display else lab
                for lab in dendro["ivl"]
            ],
            tickangle=90,
            side="bottom",
        ),
        yaxis_title="Ward distance (1-corr)",
        margin=dict(b=160),
    )
    return fig


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
        ("HRP-CVaR reference (heatmap=False)", fig_no_heat),
        ("HRP-CVaR reference (heatmap=True)", fig_heat),
    ):
        fig.update_layout(
            title=title,
            width=layout_width,
            height=layout_height,
            showlegend=False,
        )
    return [
        ("HRP-CVaR reference (heatmap=False)", fig_no_heat),
        ("HRP-CVaR reference (heatmap=True)", fig_heat),
    ]


def close_cache_usable(meta: dict, asof: str, lookback: int, calendar_last: str) -> bool:
    """True when a saved close panel matches this as-of, lookback, and calendar."""
    if not calendar_last:
        return False
    try:
        cached_lookback = int(meta.get("lookback", -1))
    except (TypeError, ValueError):
        return False
    return (
        str(meta.get("asof") or "") == str(asof)
        and cached_lookback == int(lookback)
        and str(meta.get("calendar_last") or "") == str(calendar_last)
    )


def calendar_last_line(provider_uri: str) -> str:
    path = Path(provider_uri).expanduser() / "calendars" / "day.txt"
    if not path.is_file():
        return ""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def close_cache_paths(output: Path, asof: str, lookback: int) -> tuple[Path, Path]:
    tag = str(asof).replace("-", "")
    stem = output.parent / f"hrp_close_{tag}_lb{int(lookback)}"
    return stem.with_suffix(".pkl"), stem.with_suffix(".json")


def load_cached_close(
    codes: tuple[str, ...],
    *,
    provider_uri: str,
    test_period: tuple[str, str],
    cache_pkl: Path | None,
    cache_json: Path | None,
    asof: str,
    lookback: int,
) -> pd.DataFrame:
    """Reuse a close panel when the qlib calendar has not moved; otherwise load and save."""
    calendar_last = calendar_last_line(provider_uri)
    requested = [str(code) for code in codes]
    if cache_pkl is not None and cache_json is not None and cache_pkl.is_file() and cache_json.is_file():
        try:
            meta = json.loads(cache_json.read_text(encoding="utf-8"))
            cached = pd.read_pickle(cache_pkl)
        except (OSError, ValueError, json.JSONDecodeError):
            meta, cached = {}, pd.DataFrame()
        if (
            isinstance(cached, pd.DataFrame)
            and close_cache_usable(meta, asof, lookback, calendar_last)
            and all(code in cached.columns for code in requested)
        ):
            print(f"[INFO] Reusing HRP close cache {cache_pkl}")
            return cached.loc[:, requested]
    close, _ = load_data(requested, provider_uri=provider_uri, test_period=test_period)
    if cache_pkl is not None and cache_json is not None:
        cache_pkl.parent.mkdir(parents=True, exist_ok=True)
        close.to_pickle(cache_pkl)
        cache_json.write_text(
            json.dumps(
                {"asof": asof, "lookback": int(lookback), "calendar_last": calendar_last},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return close


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
    min_corr: float = DEFAULT_MIN_CORR,
    close_cache_pkl: Path | None = None,
    close_cache_json: Path | None = None,
    asof_date: str = "",
    lookback_days: int = CLUSTER_LOOKBACK_DAYS,
) -> UniverseResult:
    result = UniverseResult(name=name, requested_codes=codes)
    if len(codes) < 2:
        result.error = f"{name}: fewer than 2 instruments requested."
        return result

    close = load_cached_close(
        codes,
        provider_uri=provider_uri,
        test_period=test_period,
        cache_pkl=close_cache_pkl,
        cache_json=close_cache_json,
        asof=asof_date,
        lookback=lookback_days,
    )
    returns, used, dropped = prepare_returns(close, codes)
    result.used_codes = used
    result.dropped_codes = dropped

    if len(used) < 2:
        result.error = (
            f"{name}: fewer than 2 instruments with valid returns "
            f"(requested={len(codes)}, used={len(used)}, dropped={len(dropped)})."
        )
        return result

    labels, domains, corr = domain_ward_labels(
        returns,
        name_map,
        dist_t=dist_t,
        n_clusters=n_clusters,
        min_corr=min_corr,
    )
    result.cluster_reps = compute_cluster_representatives(
        returns,
        close,
        name_map=name_map,
        n_clusters=n_clusters,
        rep_window=rep_window,
        min_rep_move=min_rep_move,
        dist_t=dist_t,
        min_corr=min_corr,
        labels=labels,
        corr=corr,
        domains=domains,
    )
    ward_fig = plot_ward_dendrogram(
        returns,
        labels,
        name_map,
        layout_width=layout_width,
        layout_height=layout_height,
        corr=corr,
    )
    hrp_figs = fit_hrp_dendrograms(
        rename_returns_to_display_names(returns, name_map),
        layout_width=layout_width,
        layout_height=layout_height,
    )
    result.figures = [
        ("Ward(1-corr) dendrogram (matches CSV cut)", ward_fig),
        *hrp_figs,
    ]
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
    """Render 代表清单: medoid per cluster + extreme movers per direction group."""
    ret_col = f"ret{rep_window}_pct"
    group_zh = {"up": "涨组", "down": "跌组"}
    rows: list[str] = []
    # Cluster-level medoid rows first summary
    for cid, cgrp in frame.groupby("cluster", sort=True):
        med = cgrp[cgrp["is_representative"]].drop_duplicates("code")
        domain = str(cgrp["domain"].iloc[0]) if "domain" in cgrp.columns else ""
        n_members = int(cgrp["n_members"].iloc[0]) if "n_members" in cgrp.columns else int(cgrp["code"].nunique())
        if not med.empty:
            r0 = med.iloc[0]
            med_cells = (
                f"<td><b>{html.escape(str(r0['code']))}</b></td>"
                f"<td>{html.escape(str(r0['name']))}</td>"
                f"<td>{_fmt_ret(r0[ret_col])}</td>"
            )
        else:
            med_cells = "<td colspan='3'>—</td>"
        rows.append(
            "<tr style='background:#f8fafc;'>"
            f"<td>{int(cid)}</td>"
            f"<td>{html.escape(domain)}</td>"
            "<td>medoid</td>"
            f"{med_cells}"
            f"<td>{n_members}</td>"
            "<td>簇相关中心（is_representative）</td>"
            "</tr>"
        )
        for glabel, grp in cgrp.groupby("group", sort=True):
            members = grp.sort_values(ret_col, key=lambda s: s.abs(), ascending=False)
            detail = "、".join(
                f"{r.code}({html.escape(str(r.name))} {_fmt_ret(getattr(r, ret_col))}"
                f"{'★' if bool(getattr(r, 'is_extreme_mover', False)) else ''})"
                for r in members.itertuples()
            )
            movers = grp[grp["is_extreme_mover"].astype(bool)] if "is_extreme_mover" in grp.columns else grp.iloc[0:0]
            if not movers.empty:
                mv = movers.iloc[0]
                mv_cells = (
                    f"<td>{html.escape(str(mv['code']))}</td>"
                    f"<td>{html.escape(str(mv['name']))}</td>"
                    f"<td>{_fmt_ret(mv[ret_col])}</td>"
                )
            elif str(grp["group_status"].iloc[0]) == "no_direction":
                mv_cells = (
                    "<td colspan='3'>无明确方向"
                    f"（组内最大|涨跌| &lt; {float(min_rep_move):.1f}%）</td>"
                )
            else:
                mv_cells = "<td colspan='3'>—</td>"
            rows.append(
                "<tr>"
                f"<td>{int(cid)}</td>"
                f"<td>{html.escape(domain)}</td>"
                f"<td>{group_zh.get(str(glabel), str(glabel))}</td>"
                f"{mv_cells}"
                f"<td>{len(members.drop_duplicates('code'))}</td>"
                f"<td>{detail}</td>"
                "</tr>"
            )
    return (
        "<table border='1' cellspacing='0' cellpadding='4' "
        "style='border-collapse:collapse;font-size:13px;'>"
        "<thead><tr>"
        "<th>簇</th><th>域</th><th>角色</th><th>代码</th><th>名称</th>"
        f"<th>近{rep_window}日涨跌</th><th>成员数</th>"
        "<th>说明 / 成员明细（★=极端动量）</th>"
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
    min_corr: float = DEFAULT_MIN_CORR,
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
                f"<div class=\"section\"><h3>聚类代表（相关 medoid + 极端动量；"
                f"共 {n_clusters} 簇）</h3>"
            )
            sections.append(
                f"<p class=\"meta\">切簇=域内 Ward(1-corr) dist_t={float(dist_t):.2f}；"
                f"域=bond/overseas/domestic；min_corr={float(min_corr):.2f} 拆弱簇；"
                "is_representative=簇相关中心（medoid）；"
                "is_extreme_mover=方向组内 |区间涨跌| 最大；"
                f"组内最大|涨跌| &lt; {float(min_rep_move):.1f}% 标「无明确方向」</p>"
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
    cache_pkl, cache_json = close_cache_paths(args.output, args.asof_date, args.lookback_days)
    cache_kwargs = dict(
        close_cache_pkl=cache_pkl,
        close_cache_json=cache_json,
        asof_date=args.asof_date,
        lookback_days=int(args.lookback_days),
    )
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
                min_corr=args.min_corr,
                **cache_kwargs,
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
            min_corr=args.min_corr,
            **cache_kwargs,
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
        min_corr=args.min_corr,
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
