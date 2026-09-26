#!/usr/bin/env python
"""Walk-forward evaluation of the simplified pool-generation flow.

At each anchor date in ``2020-01-01 ... 2026-04-23``:
  1. Build a pool using the past 252 trading days (Ward clustering + per-cluster
     top-K by Sharpe+12-1 momentum), same logic as ``cluster_and_select``.
  2. Measure OOS performance over the next ``oos_days`` trading days.
  3. Compare against the full eligible universe's top 10/20/40% by the same
     blend metric (sanity check that clustering doesn't destroy alpha).

Outputs a CSV + console table.

Run:
    python fund_pool_builder/walk_forward_pool_eval.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

# Make pool_builder importable so we reuse the real scoring function.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pool_builder.clustering import _return_scores  # noqa: E402

import qlib  # noqa: E402
from qlib.data import D  # noqa: E402


PROVIDER_URI = "~/etf-daily-output/Documents/stock/data/qlib_data/all_fund_data"
FUND_LIST_CSV = "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/fund_list.csv"

START_DATE = "2020-01-01"
END_DATE = "2026-04-23"
LOOKBACK_DAYS = 252
OOS_DAYS = 60
MIN_HISTORY_FOR_CLUSTER = 120
DIST_T = 0.40
INTRA_CLUSTER_MAX_CORR = 0.60
NEAR_CLONE_CORR = 0.70

ANCHOR_FREQ_MONTHS = 6  # 每 6 个月跑一次 walk-forward
OUT_CSV = "~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/walk_forward_pool_eval.csv"


def _load_universe_close() -> pd.DataFrame:
    qlib.init(provider_uri=PROVIDER_URI, region="cn")
    fl = pd.read_csv(FUND_LIST_CSV)
    feat_dir = os.path.join(PROVIDER_URI, "features")
    avail = set(os.listdir(feat_dir))
    codes = [
        c for c in fl["基金代码"].astype(str).str.upper().tolist()
        if c.lower() in avail
    ]
    df = D.features(codes, ["$close"], start_time=START_DATE, end_time=END_DATE, freq="day")
    close = df["$close"].unstack("instrument").sort_index()
    return close


def _anchor_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    if len(index) == 0:
        return []
    # 每 ANCHOR_FREQ_MONTHS 月取一个锚点，要求锚点之前有 LOOKBACK_DAYS，之后有 OOS_DAYS
    first_eligible = index[LOOKBACK_DAYS]
    last_eligible = index[-OOS_DAYS - 1] if len(index) > OOS_DAYS else index[-1]
    anchors: list[pd.Timestamp] = []
    cursor = first_eligible
    while cursor <= last_eligible:
        idx = index.get_indexer([cursor], method="bfill")[0]
        if idx < 0 or idx >= len(index):
            break
        anchors.append(index[idx])
        # advance roughly ANCHOR_FREQ_MONTHS
        cursor = (cursor + pd.DateOffset(months=ANCHOR_FREQ_MONTHS)).normalize()
    return anchors


def _select_pool_at_anchor(close: pd.DataFrame, anchor: pd.Timestamp) -> list[str]:
    """Mirror the simplified cluster_and_select flow at a given anchor date."""
    # IS window: past LOOKBACK_DAYS trading days ending at anchor (inclusive)
    idx = close.index.get_indexer([anchor], method="ffill")[0]
    if idx < 0:
        return []
    start = max(0, idx - LOOKBACK_DAYS + 1)
    is_close = close.iloc[start:idx + 1]
    is_ret = is_close.pct_change().replace([np.inf, -np.inf], np.nan)

    # eligible = codes with >=MIN_HISTORY valid rows in IS window
    valid_days = is_ret.notna().sum()
    eligible = valid_days[valid_days >= MIN_HISTORY_FOR_CLUSTER].index.tolist()
    if len(eligible) < 10:
        return []
    ret = is_ret[eligible]

    # pairwise corr (same as pairwise_overlap_matrices, simplified)
    corr = ret.corr(min_periods=120).fillna(0.05)  # low-overlap fallback
    corr = corr.clip(-1, 1)
    np.fill_diagonal(corr.values, 1.0)
    dist = 1.0 - corr
    np.fill_diagonal(dist.values, 0.0)
    try:
        condensed = squareform(dist.values, checks=False)
        Z = linkage(condensed, method="ward")
    except Exception:
        return []
    labels = fcluster(Z, t=DIST_T, criterion="distance")
    clusters = pd.Series(labels, index=ret.columns)

    # per-cluster pick: top-1 by blend score
    global_scores = _return_scores(list(ret.columns), ret)
    selected: list[str] = []
    for cid, members in clusters.groupby(clusters).groups.items():
        members = list(members)
        member_scores = global_scores.reindex(members).fillna(0.0).sort_values(ascending=False)
        if len(member_scores) == 0:
            continue
        selected.append(member_scores.index[0])

    # near-clone dedup: greedy keep higher-priority, drop anything with |corr|>NEAR_CLONE
    selected_scores = global_scores.reindex(selected).fillna(0.0).sort_values(ascending=False)
    kept: list[str] = []
    for code in selected_scores.index:
        if not kept:
            kept.append(code)
            continue
        max_corr = float(corr.loc[code, kept].abs().max())
        if max_corr <= NEAR_CLONE_CORR:
            kept.append(code)
    return kept


def _oos_stats(close: pd.DataFrame, anchor: pd.Timestamp, pool: list[str]) -> dict:
    idx = close.index.get_indexer([anchor], method="ffill")[0]
    if idx < 0 or idx + OOS_DAYS >= len(close):
        return {}
    oos_close = close.iloc[idx + 1:idx + 1 + OOS_DAYS]
    if pool:
        cols = [c for c in pool if c in oos_close.columns]
        if not cols:
            return {}
        sub = oos_close[cols]
    else:
        sub = oos_close
    ret = sub.pct_change().where(sub.pct_change().abs() <= 0.15, 0.0)
    ann = (1 + ret.fillna(0)).prod() ** (252 / max(len(ret), 1)) - 1
    vol = ret.std() * np.sqrt(252)
    sh = ann / vol.replace(0, np.nan)
    cum_per_name = (1 + ret.fillna(0)).prod() - 1  # per-name OOS cum return
    cp = ret.corr()
    n = len(cp)
    if n > 1:
        mc = cp.values[np.triu_indices(n, 1)]
    else:
        mc = np.array([np.nan])
    # equal-weight portfolio OOS ret
    ew = ret.mean(axis=1).dropna()
    ew_ann = (1 + ew).prod() ** (252 / max(len(ew), 1)) - 1
    ew_vol = ew.std() * np.sqrt(252)
    ew_sh = ew_ann / ew_vol if ew_vol > 0 else np.nan
    ew_cum = (1 + ew).prod() - 1
    dd = (1 + ew).cumprod()
    dd = (dd / dd.cummax() - 1).min()
    return dict(
        n=len(sub.columns),
        ann_med=float(ann.median()),
        sh_med=float(sh.median()) if not sh.empty else np.nan,
        corr_med=float(np.nanmedian(mc)),
        corr_90p=float(np.nanquantile(mc, 0.9)),
        ew_cum=float(ew_cum),
        ew_sh=float(ew_sh),
        ew_dd=float(dd),
        cum_per_name=cum_per_name,
    )


def _market_topk_is(close: pd.DataFrame, anchor: pd.Timestamp, pct: float) -> list[str]:
    """Top-pct% by IS blend score (same metric as pool)."""
    idx = close.index.get_indexer([anchor], method="ffill")[0]
    if idx < 0:
        return []
    start = max(0, idx - LOOKBACK_DAYS + 1)
    is_close = close.iloc[start:idx + 1]
    is_ret = is_close.pct_change().replace([np.inf, -np.inf], np.nan)
    valid_days = is_ret.notna().sum()
    eligible = valid_days[valid_days >= MIN_HISTORY_FOR_CLUSTER].index.tolist()
    if not eligible:
        return []
    ret = is_ret[eligible]
    scores = _return_scores(list(ret.columns), ret)
    n = max(1, int(len(scores) * pct))
    return scores.head(n).index.tolist()


def main() -> int:
    print(f"Loading universe from {PROVIDER_URI} ({START_DATE} -> {END_DATE})")
    close = _load_universe_close()
    print(f"Loaded close shape: {close.shape}")
    anchors = _anchor_dates(close.index)
    print(f"Anchor dates ({len(anchors)}): {[str(a.date()) for a in anchors]}")

    rows = []
    winner_detail_lines: list[str] = []
    for anchor in anchors:
        pool = _select_pool_at_anchor(close, anchor)
        pool_stats = _oos_stats(close, anchor, pool)
        if not pool_stats:
            continue
        # compare against market top-10/20/40% by IS blend
        top10 = _market_topk_is(close, anchor, 0.10)
        top40 = _market_topk_is(close, anchor, 0.40)
        t10 = _oos_stats(close, anchor, top10)
        t40 = _oos_stats(close, anchor, top40)

        # ==== OOS winner capture: market top-20 by OOS return → how many in pool? ====
        full_oos = _oos_stats(close, anchor, [])
        full_cum = full_oos.get("cum_per_name") if full_oos else None
        market_top20 = full_cum.sort_values(ascending=False).head(20) if full_cum is not None else pd.Series(dtype=float)
        market_top50 = full_cum.sort_values(ascending=False).head(50) if full_cum is not None else pd.Series(dtype=float)
        pool_set = set(pool)
        hit20 = sum(1 for c in market_top20.index if c in pool_set)
        hit50 = sum(1 for c in market_top50.index if c in pool_set)

        # ==== Pool's best OOS member & avg of pool top-5 OOS ====
        pool_cum = pool_stats["cum_per_name"].sort_values(ascending=False)
        pool_best = float(pool_cum.iloc[0]) if len(pool_cum) > 0 else np.nan
        pool_top5_avg = float(pool_cum.head(5).mean()) if len(pool_cum) >= 1 else np.nan
        market_best = float(market_top20.iloc[0]) if len(market_top20) > 0 else np.nan

        winner_detail_lines.append(
            f"{anchor.date()}  pool_best={pool_best*100:+6.2f}%  "
            f"pool_top5_avg={pool_top5_avg*100:+6.2f}%  market_best={market_best*100:+6.2f}%  "
            f"hit_top20={hit20}/20  hit_top50={hit50}/50"
        )

        row = {
            "anchor": anchor.date(),
            "pool_n": pool_stats["n"],
            "pool_ew_cum": pool_stats["ew_cum"],
            "pool_ew_sh": pool_stats["ew_sh"],
            "pool_ew_dd": pool_stats["ew_dd"],
            "pool_corr_med": pool_stats["corr_med"],
            "pool_ann_med": pool_stats["ann_med"],
            "pool_best_oos": pool_best,
            "pool_top5_avg_oos": pool_top5_avg,
            "market_best_oos": market_best,
            "hit_market_top20": hit20,
            "hit_market_top50": hit50,
            "top10_ew_cum": t10.get("ew_cum"),
            "top10_ew_sh": t10.get("ew_sh"),
            "top10_corr_med": t10.get("corr_med"),
            "top40_ew_cum": t40.get("ew_cum"),
            "top40_ew_sh": t40.get("ew_sh"),
            "top40_corr_med": t40.get("corr_med"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No anchors produced results.")
        return 1
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote per-anchor results to {OUT_CSV}")

    # winner capture detail
    print("\n=== Per-anchor winner capture (OOS 60d) ===")
    for line in winner_detail_lines:
        print(f"  {line}")

    # console summary
    pd.options.display.float_format = lambda v: f"{v:+.3f}"
    print("\n=== Walk-forward results (EW portfolio, next 60 trading days) ===")
    disp = df.copy()
    for col in ("pool_ew_cum", "pool_ew_dd", "top10_ew_cum", "top40_ew_cum",
                "pool_best_oos", "pool_top5_avg_oos", "market_best_oos"):
        disp[col] = disp[col].apply(lambda v: f"{v*100:+6.2f}%" if pd.notna(v) else "n/a")
    for col in ("pool_ew_sh", "top10_ew_sh", "top40_ew_sh"):
        disp[col] = disp[col].apply(lambda v: f"{v:+.2f}" if pd.notna(v) else "n/a")
    for col in ("pool_corr_med", "top10_corr_med", "top40_corr_med", "pool_ann_med"):
        disp[col] = disp[col].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "n/a")
    print(disp[[
        "anchor", "pool_n",
        "pool_best_oos", "pool_top5_avg_oos", "market_best_oos",
        "hit_market_top20", "hit_market_top50",
        "pool_ew_cum", "pool_corr_med",
    ]].to_string(index=False))

    # aggregate
    print("\n=== Aggregate across anchors ===")
    agg = {
        "pool_ew_cum_avg":        df["pool_ew_cum"].mean(),
        "pool_ew_sh_avg":         df["pool_ew_sh"].mean(),
        "pool_ew_dd_worst":       df["pool_ew_dd"].min(),
        "pool_corr_med_avg":      df["pool_corr_med"].mean(),
        "pool_best_oos_avg":      df["pool_best_oos"].mean(),
        "pool_top5_avg_oos_avg":  df["pool_top5_avg_oos"].mean(),
        "market_best_oos_avg":    df["market_best_oos"].mean(),
        "hit_market_top20_avg":   df["hit_market_top20"].mean(),
        "hit_market_top50_avg":   df["hit_market_top50"].mean(),
        "top10_ew_cum_avg":       df["top10_ew_cum"].mean(),
        "top10_ew_sh_avg":        df["top10_ew_sh"].mean(),
        "top40_ew_cum_avg":       df["top40_ew_cum"].mean(),
        "top40_ew_sh_avg":        df["top40_ew_sh"].mean(),
        "pool_beats_top40_pct":   (df["pool_ew_cum"] > df["top40_ew_cum"]).mean(),
        "pool_beats_top10_pct":   (df["pool_ew_cum"] > df["top10_ew_cum"]).mean(),
    }
    for k, v in agg.items():
        print(f"  {k:<30}  {v:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
