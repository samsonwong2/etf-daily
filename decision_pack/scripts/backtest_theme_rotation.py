#!/usr/bin/env python3
"""Causal 4-theme momentum rotation vs SH510300 and EW46.

Rules (no HTML peaks, no lookahead):
1. At month-end, form 4 buckets either:
   - causal: agglomerative k=4 on past return correlation (lookback only)
   - fixed: static theme map from names (robustness check only)
2. Score each bucket by equal-weight past momentum (20d / 63d).
3. Hold Top-1 or Top-2 buckets, equal-weight names inside; T+1 + cost.
4. Gate: train excess>0 AND OOS excess>0 vs SH510300 (and report vs EW46).

Does not change production adaptive / hold_up or from_listing HTML.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

warnings.filterwarnings("ignore", category=FutureWarning)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runtime_paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from decision_pack.src.event_long_only_sim import COST_PER_SIDE  # noqa: E402
from decision_pack.src.hrp_cluster_router import (  # noqa: E402
    load_cluster_membership,
    normalize_code,
)
from decision_pack.src.vol_need_return_board import EQUITY_ANCHOR  # noqa: E402

DEFAULT_MEMBERSHIP = Path(
    "~/etf-daily-output/temp/decision_packs/20260814/cluster_representatives_20260814.csv"
)
DEFAULT_LISTING_CSV = (
    PLOTLY_OUTPUTS_DIR / "20260814_from_listing/listing_dates.csv"
)

# Fixed 4 themes from names (as-of pack labels). Research only; not causal.
FIXED_THEME: dict[str, str] = {
    # tech / semi / growth hardware
    "SH588850": "tech_semi",
    "SH513310": "tech_semi",
    "SH588710": "tech_semi",
    "SH515050": "tech_semi",
    "SZ159206": "tech_semi",
    "SH516770": "tech_semi",
    "SZ159512": "tech_semi",
    "SZ159237": "tech_semi",
    # resources / battery / energy commodities
    "SH561800": "resources",
    "SH561160": "resources",
    "SZ159320": "resources",
    "SH515220": "resources",
    "SH513350": "resources",
    "SZ159981": "resources",
    "SZ159985": "resources",
    "SZ159309": "resources",
    "SH516570": "resources",
    "SH517400": "resources",
    "SH562900": "resources",
    "SH561560": "resources",
    # overseas equity / thematic ADRs
    "SH520830": "overseas",
    "SH520870": "overseas",
    "SH513290": "overseas",
    "SZ159502": "overseas",
    "SZ159561": "overseas",
    "SH513080": "overseas",
    "SZ159509": "overseas",
    "SH513650": "overseas",
    "SH513400": "overseas",
    "SH513730": "overseas",
    "SZ159529": "overseas",
    "SH513050": "overseas",
    "SH513970": "overseas",
    # domestic cyclicals / defensive / other CN
    "SH560280": "domestic",
    "SH560880": "domestic",
    "SH516910": "domestic",
    "SH512690": "domestic",
    "SH516310": "domestic",
    "SH517120": "domestic",
    "SH589720": "domestic",
    "SZ159647": "domestic",
    "SZ159898": "domestic",
    "SH515060": "domestic",
    "SZ159745": "domestic",
    "SH513090": "domestic",
    "SH562510": "domestic",
}

THEME_ORDER = ("tech_semi", "resources", "overseas", "domestic")


def _load_pool_mod():
    import importlib.util

    path = _PROJECT_ROOT / "decision_pack/scripts/backtest_pool_vs_anchor.py"
    spec = importlib.util.spec_from_file_location("pool_vs_anchor_rot", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--as-of", default="2026-08-14")
    p.add_argument("--train-cutoff", default="2026-04-01")
    p.add_argument("--membership", type=Path, default=DEFAULT_MEMBERSHIP)
    p.add_argument("--listing-csv", type=Path, default=DEFAULT_LISTING_CSV)
    p.add_argument(
        "--html-out-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260814_theme_rotation",
    )
    p.add_argument("--cost-per-side", type=float, default=COST_PER_SIDE)
    p.add_argument("--cluster-lookback", type=int, default=63)
    p.add_argument("--mom-lookbacks", default="20,63")
    p.add_argument("--tops", default="1,2", help="hold top-K buckets")
    p.add_argument("--n-themes", type=int, default=4)
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--fail-fast", action="store_true")
    return p.parse_args(argv)


def past_total_return(prices: pd.DataFrame, code: str, asof: pd.Timestamp, lookback: int) -> float:
    if code not in prices.columns:
        return float("nan")
    s = prices[code].loc[:asof].dropna()
    if len(s) < lookback + 1:
        return float("nan")
    p0, p1 = float(s.iloc[-(lookback + 1)]), float(s.iloc[-1])
    if not (np.isfinite(p0) and p0 > 0 and np.isfinite(p1)):
        return float("nan")
    return float(p1 / p0 - 1.0)


def causal_k_buckets(
    prices: pd.DataFrame,
    codes: list[str],
    asof: pd.Timestamp,
    *,
    n_themes: int,
    lookback: int,
) -> dict[str, list[str]]:
    """Agglomerative k clusters on past return correlation distance (causal)."""
    use = [c for c in codes if c in prices.columns]
    if len(use) < n_themes:
        # Fall back: each name alone until we have enough; then lump remainder
        buckets: dict[str, list[str]] = {f"c{i}": [] for i in range(n_themes)}
        for i, c in enumerate(use):
            buckets[f"c{i % n_themes}"].append(c)
        return {k: v for k, v in buckets.items() if v}

    px = prices[use].loc[:asof].ffill().dropna(how="all")
    if len(px) < lookback + 2:
        buckets = {f"c{i}": [] for i in range(n_themes)}
        for i, c in enumerate(use):
            buckets[f"c{i % n_themes}"].append(c)
        return {k: v for k, v in buckets.items() if v}

    window = px.iloc[-(lookback + 1) :]
    rets = window.pct_change().iloc[1:].dropna(axis=1, how="any")
    cols = list(rets.columns)
    if len(cols) < n_themes:
        buckets = {f"c{i}": [] for i in range(n_themes)}
        for i, c in enumerate(cols):
            buckets[f"c{i % n_themes}"].append(c)
        return {k: v for k, v in buckets.items() if v}

    corr = rets.corr().fillna(0.0).values
    # Correlation distance in [0, 2]
    dist = np.clip(1.0 - corr, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    # Condensed-ish via precomputed affinity: sklearn wants distance matrix with metric=precomputed
    model = AgglomerativeClustering(
        n_clusters=n_themes,
        metric="precomputed",
        linkage="average",
    )
    labels = model.fit_predict(dist)
    buckets = {f"c{i}": [] for i in range(n_themes)}
    for code, lab in zip(cols, labels):
        buckets[f"c{int(lab)}"].append(code)
    return {k: v for k, v in buckets.items() if v}


def fixed_theme_buckets(codes: list[str]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {t: [] for t in THEME_ORDER}
    for c in codes:
        theme = FIXED_THEME.get(c, "domestic")
        buckets[theme].append(c)
    return {k: v for k, v in buckets.items() if v}


def score_buckets(
    prices: pd.DataFrame,
    buckets: dict[str, list[str]],
    asof: pd.Timestamp,
    mom_lookback: int,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for name, members in buckets.items():
        rets = [past_total_return(prices, c, asof, mom_lookback) for c in members]
        rets = [r for r in rets if np.isfinite(r)]
        scores[name] = float(np.mean(rets)) if rets else float("-inf")
    return scores


def top_bucket_weights(
    prices: pd.DataFrame,
    listed: list[str],
    asof: pd.Timestamp,
    *,
    mode: str,
    n_themes: int,
    cluster_lookback: int,
    mom_lookback: int,
    top_k: int,
    pick_log: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, float], str | None]:
    if not listed:
        return {}, "no_listed"
    if mode == "causal":
        buckets = causal_k_buckets(
            prices, listed, asof, n_themes=n_themes, lookback=cluster_lookback
        )
    elif mode == "fixed":
        buckets = fixed_theme_buckets(listed)
    else:
        return {}, f"unknown_mode:{mode}"

    scores = score_buckets(prices, buckets, asof, mom_lookback)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    chosen = [name for name, sc in ranked[: max(1, top_k)] if np.isfinite(sc) and sc > float("-inf")]
    if not chosen:
        # Fallback: EW all listed
        w = 1.0 / float(len(listed))
        return {c: w for c in listed}, "fallback_ew_all"

    members: list[str] = []
    for name in chosen:
        members.extend(buckets.get(name, []))
    members = sorted(set(m for m in members if m in listed))
    if not members:
        return {}, "empty_chosen"
    w = 1.0 / float(len(members))
    weights = {c: w for c in members}
    if pick_log is not None:
        pick_log.append(
            {
                "asof": asof.strftime("%Y-%m-%d"),
                "mode": mode,
                "mom_lookback": mom_lookback,
                "top_k": top_k,
                "scores": {k: (None if not np.isfinite(v) else float(v)) for k, v in scores.items()},
                "chosen": chosen,
                "n_members": len(members),
                "members": members,
            }
        )
    return weights, None


def write_readme(path: Path) -> None:
    path.write_text(
        """# Theme rotation gate (4 buckets + month-end momentum)

## Setup
- Buckets: causal agglomerative k=4 on past return correlation, or fixed name themes.
- Signal: month-end; hold Top-1/Top-2 buckets by past 20d/63d EW momentum.
- Execution: T+1 close; cost 0.001 × turnover.
- Benchmarks: SH510300 buy-and-hold; monthly EW46 (same sim engine).

## Gate
Pass only if **train excess > 0 and OOS excess > 0** vs SH510300.
Also report excess vs EW46 (informational).

HTML chart peaks are **not** evidence and are not used.
""",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    as_of = pd.Timestamp(args.as_of).strftime("%Y-%m-%d")
    train_cutoff = pd.Timestamp(args.train_cutoff).strftime("%Y-%m-%d")
    train_end = (pd.Timestamp(train_cutoff) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    out_dir = Path(args.html_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_readme(out_dir / "README.md")

    mom_lbs = [int(x) for x in str(args.mom_lookbacks).split(",") if x.strip()]
    tops = [int(x) for x in str(args.tops).split(",") if x.strip()]
    n_themes = int(args.n_themes)

    pool = _load_pool_mod()
    membership = load_cluster_membership(args.membership)
    listings = pool.load_listings(args.listing_csv)
    codes = sorted(membership["code"].unique())
    if args.max_codes is not None:
        codes = codes[: int(args.max_codes)]

    # Validate fixed map coverage
    missing_fixed = [c for c in codes if c not in FIXED_THEME]
    if missing_fixed:
        print(f"[WARN] fixed theme missing codes → domestic: {missing_fixed}", flush=True)

    print(
        f"[INFO] codes={len(codes)} themes={n_themes} mom={mom_lbs} tops={tops} "
        f"train_end={train_end} oos={train_cutoff}→{as_of}",
        flush=True,
    )
    print(f"[INFO] out={out_dir}", flush=True)

    pm = pool._load_pm()
    pm._init_qlib(None)

    ohlcv_a = pm.load_qlib_ohlcv(
        EQUITY_ANCHOR, "2010-01-01", as_of, init_qlib=False, clip_to_window=False
    )
    close_a = pool._norm_close(
        ohlcv_a.set_index(pd.to_datetime(ohlcv_a["datetime"]).dt.normalize())["$close"]
    )

    listing_dates = [pd.Timestamp(listings.get(c, "2015-01-01")) for c in codes]
    train_start = min(listing_dates).strftime("%Y-%m-%d") if listing_dates else "2015-01-01"

    series: dict[str, pd.Series] = {}
    for i, code in enumerate(codes, 1):
        listing = listings.get(code, "2015-01-01")
        try:
            print(f"[{i}/{len(codes)}] {code}", flush=True)
            ohlcv = pm.load_qlib_ohlcv(
                code, listing, as_of, init_qlib=False, clip_to_window=False
            )
            close = pool._norm_close(
                ohlcv.set_index(pd.to_datetime(ohlcv["datetime"]).dt.normalize())["$close"]
            )
            close = close[
                (close.index >= pd.Timestamp(listing).normalize())
                & (close.index <= pd.Timestamp(as_of).normalize())
            ]
            series[code] = close
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {code}: {exc}", flush=True)
            if args.fail_fast:
                raise
            traceback.print_exc()

    prices = pd.DataFrame(series).sort_index()
    union = pd.DatetimeIndex(sorted(set(prices.index) | set(close_a.index)))
    prices = prices.reindex(union).sort_index()
    prices = prices.loc[
        (prices.index >= pd.Timestamp(train_start))
        & (prices.index <= pd.Timestamp(as_of))
    ]
    prices = prices.ffill()

    pick_logs: dict[str, list[dict[str, Any]]] = {}
    summary_rows: list[dict[str, Any]] = []

    def make_fn(
        mode: str, mom_lb: int, top_k: int, key: str
    ) -> Callable[[list[str], pd.Timestamp], tuple[dict[str, float], str | None]]:
        log = pick_logs.setdefault(key, [])

        def _fn(listed: list[str], asof: pd.Timestamp):
            return top_bucket_weights(
                prices,
                listed,
                asof,
                mode=mode,
                n_themes=n_themes,
                cluster_lookback=int(args.cluster_lookback),
                mom_lookback=mom_lb,
                top_k=top_k,
                pick_log=log,
            )

        return _fn

    # Baseline EW46
    def ew_fn(listed: list[str], _asof: pd.Timestamp):
        return pool.equal_weights(listed), None

    for split, lo, hi in (
        ("train", train_start, train_end),
        ("oos", train_cutoff, as_of),
    ):
        eq, meta = pool.simulate_monthly_rebalance(
            prices,
            close_a,
            weight_fn=ew_fn,
            start=lo,
            end=hi,
            listings=listings,
            cost_per_side=float(args.cost_per_side),
            scheme="ew46",
        )
        meta["split"] = split
        meta["mode"] = "ew46"
        meta["mom_lookback"] = None
        meta["top_k"] = None
        summary_rows.append(meta)
        eq.to_csv(out_dir / f"bench_ew46_{split}.csv", index=False)
        print(
            f"[OK] EW46 {split}: bh={meta['bh']:.4f} "
            f"anchor={meta['anchor_bh']:.4f} excess_vs_300={meta['excess']:.4f}",
            flush=True,
        )

    ew_by_split = {
        r["split"]: r for r in summary_rows if r.get("scheme") == "ew46"
    }

    variants: list[tuple[str, str, int, int]] = []
    for mode in ("causal", "fixed"):
        for mom_lb in mom_lbs:
            for top_k in tops:
                key = f"{mode}_mom{mom_lb}_top{top_k}"
                variants.append((key, mode, mom_lb, top_k))

    for key, mode, mom_lb, top_k in variants:
        # Reset pick log for clean per-split appends
        pick_logs[key] = []
        for split, lo, hi in (
            ("train", train_start, train_end),
            ("oos", train_cutoff, as_of),
        ):
            # Separate log per split
            split_key = f"{key}__{split}"
            pick_logs[split_key] = []
            fn = make_fn(mode, mom_lb, top_k, split_key)
            eq, meta = pool.simulate_monthly_rebalance(
                prices,
                close_a,
                weight_fn=fn,
                start=lo,
                end=hi,
                listings=listings,
                cost_per_side=float(args.cost_per_side),
                scheme=key,
            )
            meta["split"] = split
            meta["mode"] = mode
            meta["mom_lookback"] = mom_lb
            meta["top_k"] = top_k
            ew_m = ew_by_split.get(split, {})
            ew_bh = float(ew_m.get("bh", float("nan")))
            port_bh = float(meta.get("bh", float("nan")))
            meta["excess_vs_ew46"] = (
                float(port_bh - ew_bh)
                if np.isfinite(port_bh) and np.isfinite(ew_bh)
                else float("nan")
            )
            summary_rows.append(meta)
            eq.to_csv(out_dir / f"{key}_{split}.csv", index=False)
            with (out_dir / f"{key}_{split}_picks.json").open("w", encoding="utf-8") as f:
                json.dump(pick_logs[split_key], f, ensure_ascii=False, indent=2)
            print(
                f"[OK] {key} {split}: bh={meta['bh']:.4f} "
                f"vs300={meta['excess']:.4f} vsEW46={meta['excess_vs_ew46']:.4f} "
                f"reb={meta.get('n_rebalances')}",
                flush=True,
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)

    # Gate table: one row per variant
    gate_rows: list[dict[str, Any]] = []
    for key, mode, mom_lb, top_k in variants:
        sub = summary[summary["scheme"] == key]
        tr = sub[sub["split"] == "train"]
        oo = sub[sub["split"] == "oos"]
        if tr.empty or oo.empty:
            continue
        tr0, oo0 = tr.iloc[0], oo.iloc[0]
        ex_tr = float(tr0["excess"])
        ex_oo = float(oo0["excess"])
        ex_ew_tr = float(tr0["excess_vs_ew46"])
        ex_ew_oo = float(oo0["excess_vs_ew46"])
        pass_300 = bool(np.isfinite(ex_tr) and np.isfinite(ex_oo) and ex_tr > 0 and ex_oo > 0)
        pass_ew = bool(
            np.isfinite(ex_ew_tr) and np.isfinite(ex_ew_oo) and ex_ew_tr > 0 and ex_ew_oo > 0
        )
        gate_rows.append(
            {
                "scheme": key,
                "mode": mode,
                "mom_lookback": mom_lb,
                "top_k": top_k,
                "train_bh": float(tr0["bh"]),
                "oos_bh": float(oo0["bh"]),
                "train_excess_vs_300": ex_tr,
                "oos_excess_vs_300": ex_oo,
                "train_excess_vs_ew46": ex_ew_tr,
                "oos_excess_vs_ew46": ex_ew_oo,
                "pass_vs_300": pass_300,
                "pass_vs_ew46": pass_ew,
            }
        )

    gate = pd.DataFrame(gate_rows)
    gate.to_csv(out_dir / "gate.csv", index=False)

    any_pass_300 = bool(gate["pass_vs_300"].any()) if not gate.empty else False
    any_pass_both = (
        bool((gate["pass_vs_300"] & gate["pass_vs_ew46"]).any()) if not gate.empty else False
    )
    best = None
    if not gate.empty:
        # Prefer causal; rank by min(train,oos) excess vs 300
        g2 = gate.copy()
        g2["min_ex"] = g2[["train_excess_vs_300", "oos_excess_vs_300"]].min(axis=1)
        g2 = g2.sort_values(
            ["pass_vs_300", "mode", "min_ex"],
            ascending=[False, True, False],
        )
        best = g2.iloc[0].to_dict()

    if any_pass_both:
        verdict = "PASS_theme_rotation_beats_300_and_ew46"
    elif any_pass_300:
        verdict = "WEAK_PASS_vs_300_only_check_vs_ew46"
    else:
        verdict = "REJECT_hold_anchor_or_ew46_do_not_rotate_on_momentum"

    payload = {
        "verdict": verdict,
        "as_of": as_of,
        "train_cutoff": train_cutoff,
        "train_start": train_start,
        "n_codes": len(codes),
        "n_themes": n_themes,
        "cluster_lookback": int(args.cluster_lookback),
        "any_pass_vs_300": any_pass_300,
        "any_pass_vs_300_and_ew46": any_pass_both,
        "best_variant": best,
        "note": (
            "Causal k=4 uses only past returns through signal date; "
            "fixed themes are a robustness label map, not lookahead peaks."
        ),
    }
    with (out_dir / "verdict.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    print("\n=== GATE ===", flush=True)
    if not gate.empty:
        cols = [
            "scheme",
            "train_excess_vs_300",
            "oos_excess_vs_300",
            "train_excess_vs_ew46",
            "oos_excess_vs_ew46",
            "pass_vs_300",
            "pass_vs_ew46",
        ]
        print(gate[cols].to_string(index=False), flush=True)
    print(f"\n[VERDICT] {verdict}", flush=True)
    print(f"[OK] wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
