#!/usr/bin/env python3
"""Search long-only entry/exit combos from from_listing chart indicators.

Train < train_cutoff selects pairs; freeze for OOS; then break down by HRP
cluster. Does not use hold_up / dual_ma / adaptive configs as signal rules.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from etf_daily.lib.event_long_only_sim import (  # noqa: E402
    COST_PER_SIDE,
    simulate_long_only_events,
)
from etf_daily.lib.html_indicator_catalog import (  # noqa: E402
    CATEGORIES,
    EVENT_BY_NAME,
    EVENT_DEFS,
    build_event_frame,
    catalog_markdown,
    combine_entry_signals,
    events_in_category,
)
from etf_daily.hrp.hrp_cluster_router import normalize_code  # noqa: E402
from etf_daily.lib.regime_transition_plot import (  # noqa: E402
    compute_volatility_regime_frame,
    load_validation_csvs,
)
from etf_daily.lib.vol_need_return_board import EQUITY_ANCHOR  # noqa: E402

DEFAULT_MEMBERSHIP = Path(
    "~/etf-daily-output/temp/decision_packs/20260814/cluster_representatives_20260814.csv"
)
DEFAULT_VALIDATION_DIR = (
    _PROJECT_ROOT
    / "runtime/decision_packs/20260720/regime_transition_validation_q90_to0720"
)
DEFAULT_LISTING_CSV = (
    PLOTLY_OUTPUTS_DIR / "20260814_from_listing/listing_dates.csv"
)
MIN_TRAIN_BARS = 60
TRAIN_MEDIAN_CLOSED = 3
TRAIN_POS_EDGE_FRAC = 0.40


def _load_pm():
    import importlib.util

    path = _PROJECT_ROOT / "src/etf_daily/plots/plot_regime_transition_example.py"
    spec = importlib.util.spec_from_file_location("plot_regime_pm_combo", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--as-of", default="2026-08-14")
    p.add_argument("--train-cutoff", default="2026-04-01")
    p.add_argument("--membership", type=Path, default=DEFAULT_MEMBERSHIP)
    p.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    p.add_argument("--listing-csv", type=Path, default=DEFAULT_LISTING_CSV)
    p.add_argument(
        "--html-out-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260814_html_indicator_combos",
    )
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--code", action="append", default=None)
    p.add_argument("--cost-per-side", type=float, default=COST_PER_SIDE)
    p.add_argument("--skip-stage-b", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    return p.parse_args(argv)


def load_membership(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"cluster", "code"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"membership missing columns: {sorted(missing)}")
    out = df.copy()
    out["code"] = out["code"].map(normalize_code)
    out["cluster"] = out["cluster"].astype(int)
    return out


def load_listings(path: Path, as_of: str) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "code" not in df.columns or "listing" not in df.columns:
        return {}
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        code = normalize_code(row["code"])
        listing = pd.Timestamp(row["listing"]).strftime("%Y-%m-%d")
        out[code] = listing
    return out


def stage_a_combos() -> list[dict[str, Any]]:
    combos: list[dict[str, Any]] = []
    for cat in CATEGORIES:
        names = events_in_category(cat)
        for entry, exit_name in itertools.product(names, names):
            if entry == exit_name:
                continue
            combos.append(
                {
                    "stage": "A",
                    "combo_id": f"A|{cat}|{entry}|{exit_name}",
                    "category": cat,
                    "entry_events": (entry,),
                    "exit_event": exit_name,
                    "entry_how": "single",
                }
            )
    return combos


def stage_b_combos(survivors: pd.DataFrame) -> list[dict[str, Any]]:
    """Cross-category AND of two survivor entries + one survivor exit."""
    if survivors is None or survivors.empty:
        return []
    entry_pool: list[tuple[str, str]] = []
    exit_pool: list[str] = []
    seen_e: set[str] = set()
    seen_x: set[str] = set()
    for _, row in survivors.iterrows():
        entries = str(row["entry_events"]).split("+")
        for e in entries:
            e = e.strip()
            if not e or e in seen_e or e not in EVENT_BY_NAME:
                continue
            seen_e.add(e)
            entry_pool.append((e, EVENT_BY_NAME[e].category))
        x = str(row["exit_event"]).strip()
        if x and x not in seen_x and x in EVENT_BY_NAME:
            seen_x.add(x)
            exit_pool.append(x)
    combos: list[dict[str, Any]] = []
    for (e1, c1), (e2, c2) in itertools.combinations(entry_pool, 2):
        if c1 == c2:
            continue
        for exit_name in exit_pool:
            if exit_name in (e1, e2):
                continue
            eid = "+".join(sorted([e1, e2]))
            combos.append(
                {
                    "stage": "B",
                    "combo_id": f"B|{eid}|{exit_name}",
                    "category": f"{c1}+{c2}",
                    "entry_events": tuple(sorted([e1, e2])),
                    "exit_event": exit_name,
                    "entry_how": "and",
                }
            )
    return combos


def _window_mask(idx: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    lo = pd.Timestamp(start).normalize()
    hi = pd.Timestamp(end).normalize()
    return (idx >= lo) & (idx <= hi)


def eval_combo_on_code(
    events: pd.DataFrame,
    close: pd.Series,
    combo: dict[str, Any],
    *,
    start: str,
    end: str,
    cost_per_side: float,
) -> dict[str, Any] | None:
    idx = events.index
    mask = _window_mask(pd.DatetimeIndex(idx), start, end)
    if int(mask.sum()) < 2:
        return None
    sub_ev = events.loc[mask]
    sub_px = close.reindex(sub_ev.index)
    if combo["entry_how"] == "and":
        entry = combine_entry_signals(sub_ev, combo["entry_events"], how="and")
    else:
        entry = sub_ev[combo["entry_events"][0]].fillna(False).astype(bool)
    exit_col = combo["exit_event"]
    if exit_col not in sub_ev.columns:
        return None
    exit_sig = sub_ev[exit_col].fillna(False).astype(bool)
    metrics, _trades, _eq = simulate_long_only_events(
        entry,
        exit_sig,
        sub_px,
        dates=sub_ev.index.strftime("%Y-%m-%d"),
        cost_per_side=cost_per_side,
    )
    return metrics


def aggregate_pool(
    per_code: list[dict[str, Any]],
    *,
    eligible_codes: set[str],
) -> dict[str, Any]:
    rows = [r for r in per_code if r["code"] in eligible_codes]
    if not rows:
        return {
            "n_codes": 0,
            "n_eligible": 0,
            "median_n_closed": 0.0,
            "mean_edge": float("nan"),
            "mean_compound": float("nan"),
            "mean_bh": float("nan"),
            "mean_win": float("nan"),
            "frac_pos_edge": float("nan"),
            "pass_train_gate": False,
        }
    edges = np.asarray([r["edge"] for r in rows], dtype=float)
    closed = np.asarray([r["n_closed"] for r in rows], dtype=float)
    compounds = np.asarray([r["compound"] for r in rows], dtype=float)
    bhs = np.asarray([r["bh"] for r in rows], dtype=float)
    wins = np.asarray(
        [r["win"] if np.isfinite(r["win"]) else np.nan for r in rows], dtype=float
    )
    frac = float(np.mean(edges > 0))
    med_closed = float(np.median(closed))
    mean_edge = float(np.mean(edges))
    pass_gate = (
        med_closed >= TRAIN_MEDIAN_CLOSED
        and mean_edge > 0.0
        and frac >= TRAIN_POS_EDGE_FRAC
    )
    return {
        "n_codes": int(len(rows)),
        "n_eligible": int(len(eligible_codes)),
        "median_n_closed": med_closed,
        "mean_edge": mean_edge,
        "mean_compound": float(np.mean(compounds)),
        "mean_bh": float(np.mean(bhs)),
        "mean_win": float(np.nanmean(wins)) if np.isfinite(wins).any() else float("nan"),
        "frac_pos_edge": frac,
        "pass_train_gate": bool(pass_gate),
    }


def combo_row(
    combo: dict[str, Any],
    agg: dict[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    return {
        "split": split,
        "stage": combo["stage"],
        "combo_id": combo["combo_id"],
        "category": combo["category"],
        "entry_events": "+".join(combo["entry_events"]),
        "exit_event": combo["exit_event"],
        "entry_how": combo["entry_how"],
        **agg,
    }


def cluster_breakdown(
    profitable: pd.DataFrame,
    per_code_oos: dict[str, list[dict[str, Any]]],
    membership: pd.DataFrame,
) -> pd.DataFrame:
    if profitable.empty:
        return pd.DataFrame(
            columns=[
                "combo_id",
                "cluster",
                "n_codes",
                "mean_edge",
                "mean_win",
                "mean_n_closed",
                "mean_compound",
                "codes",
            ]
        )
    code_cluster = {
        normalize_code(r.code): int(r.cluster)
        for r in membership.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for _, prow in profitable.iterrows():
        cid = str(prow["combo_id"])
        code_rows = per_code_oos.get(cid, [])
        by_c: dict[int, list[dict[str, Any]]] = {}
        for r in code_rows:
            cl = code_cluster.get(r["code"])
            if cl is None:
                continue
            by_c.setdefault(cl, []).append(r)
        for cl, items in sorted(by_c.items()):
            edges = [x["edge"] for x in items]
            wins = [x["win"] for x in items if np.isfinite(x["win"])]
            closed = [x["n_closed"] for x in items]
            compounds = [x["compound"] for x in items]
            rows.append(
                {
                    "combo_id": cid,
                    "cluster": cl,
                    "n_codes": len(items),
                    "mean_edge": float(np.mean(edges)),
                    "mean_win": float(np.mean(wins)) if wins else float("nan"),
                    "mean_n_closed": float(np.mean(closed)),
                    "mean_compound": float(np.mean(compounds)),
                    "codes": ",".join(sorted(x["code"] for x in items)),
                }
            )
    return pd.DataFrame(rows)


def write_readme(path: Path, *, train_cutoff: str, as_of: str, cost: float) -> None:
    text = f"""# HTML indicator combo backtest

## Split
- Train: listing → day before `{train_cutoff}` (codes with < {MIN_TRAIN_BARS} train bars excluded from selection)
- OOS freeze: `{train_cutoff}` → `{as_of}`
- Long-only; signal at T close → fill at T+1 close; cost `{cost:.4f}` per side

## Train gate (pool equal-weight over eligible codes)
- median `n_closed` ≥ {TRAIN_MEDIAN_CLOSED}
- mean `edge` (`compound − BH`) > 0
- ≥ {TRAIN_POS_EDGE_FRAC:.0%} of eligible codes with `edge` > 0

## Profit definition
Combo passes train gate **and** OOS pool mean `edge` > 0 → `profitable_oos.csv`.
Train-pass / OOS-fail → `overfit_watch.csv` (not a discovery).

## Excluded (not searched)
- T+3 / T+10 reversal diamonds, early-confirm revoke
- hold_up 上涨买/卖 / method regime bands
- ED / RU / early regime marks / trade-bias hover

## Risks
- OOS window is short (~4 months); weak statistical power
- Many HRP clusters have 1–2 names; cluster_breakdown is descriptive only
- Pre-2024-06 HMM triangles use walkforward refill (not official OOS pack fit)
- Pool equal-weight mean `edge` vs buy-and-hold is a hard bar: intermittent
  long-only pairs often lose to BH on average even when some names win
- When train gate yields zero survivors, see `leaderboard_train.csv` /
  `leaderboard_oos.csv` and diagnostic `oos_positive_ungated.csv`
"""
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    as_of = pd.Timestamp(args.as_of).strftime("%Y-%m-%d")
    train_cutoff = pd.Timestamp(args.train_cutoff).strftime("%Y-%m-%d")
    train_end = (pd.Timestamp(train_cutoff) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    out_dir = Path(args.html_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    membership = load_membership(args.membership)
    listings = load_listings(args.listing_csv, as_of)
    codes = sorted(membership["code"].unique())
    if args.code:
        want = {normalize_code(c) for c in args.code}
        codes = [c for c in codes if c in want]
    if args.max_codes is not None:
        codes = codes[: int(args.max_codes)]

    print(f"[INFO] codes={len(codes)} train_end={train_end} oos={train_cutoff}→{as_of}")
    print(f"[INFO] out={out_dir}")

    (out_dir / "event_catalog.md").write_text(catalog_markdown(), encoding="utf-8")
    write_readme(
        out_dir / "README.md",
        train_cutoff=train_cutoff,
        as_of=as_of,
        cost=float(args.cost_per_side),
    )

    oos_all, _rev, _evt = load_validation_csvs(args.validation_dir)
    pm = _load_pm()
    pm._init_qlib(None)

    # Anchor vol for fair-gap events.
    print(f"[INFO] loading anchor {EQUITY_ANCHOR}")
    ohlcv_anchor = pm.load_qlib_ohlcv(
        EQUITY_ANCHOR,
        "2010-01-01",
        as_of,
        init_qlib=False,
        clip_to_window=False,
    )
    close_a = ohlcv_anchor.set_index(
        pd.to_datetime(ohlcv_anchor["datetime"]).dt.normalize()
    )["$close"]
    close_a = close_a[~close_a.index.duplicated(keep="last")]
    vf_a = compute_volatility_regime_frame(close_a)
    vf_a["as_of"] = pd.to_datetime(vf_a["as_of"]).dt.normalize()
    vf_a = vf_a.set_index("as_of")
    rv20_anchor = vf_a["rv20"]
    rv5_anchor = vf_a["rv5"]

    # Per-code cache: events + close + eligibility.
    cache: dict[str, dict[str, Any]] = {}
    eligible: set[str] = set()
    for i, code in enumerate(codes, 1):
        listing = listings.get(code, "2015-01-01")
        try:
            print(f"[{i}/{len(codes)}] build events {code} listing={listing}")
            ohlcv = pm.load_qlib_ohlcv(
                code,
                listing,
                as_of,
                init_qlib=False,
                clip_to_window=False,
            )
            close = ohlcv.set_index(pd.to_datetime(ohlcv["datetime"]).dt.normalize())[
                "$close"
            ]
            close = close[~close.index.duplicated(keep="last")].sort_index()
            # Clip display window to listing→as_of for trading.
            close = close[
                (close.index >= pd.Timestamp(listing).normalize())
                & (close.index <= pd.Timestamp(as_of).normalize())
            ]
            events = build_event_frame(
                close,
                code=code,
                oos=oos_all,
                rv20_anchor=rv20_anchor,
                rv5_anchor=rv5_anchor,
                start_date=listing,
                end_date=as_of,
            )
            train_mask = _window_mask(
                pd.DatetimeIndex(close.index), listing, train_end
            )
            n_train = int(train_mask.sum())
            cache[code] = {
                "close": close,
                "events": events,
                "listing": listing,
                "n_train": n_train,
            }
            if n_train >= MIN_TRAIN_BARS:
                eligible.add(code)
            else:
                print(f"  [skip-select] {code} train_bars={n_train} < {MIN_TRAIN_BARS}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {code}: {exc}")
            if args.fail_fast:
                raise
            traceback.print_exc()

    print(f"[INFO] eligible_for_select={len(eligible)}/{len(cache)}")

    def run_stage(
        combos: list[dict[str, Any]],
        *,
        split: str,
        start: str,
        end: str,
    ) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
        rows: list[dict[str, Any]] = []
        per_code_map: dict[str, list[dict[str, Any]]] = {}
        for j, combo in enumerate(combos, 1):
            if j % 50 == 0 or j == len(combos):
                print(f"  [{split}] {j}/{len(combos)} {combo['combo_id']}")
            per_code: list[dict[str, Any]] = []
            for code, blob in cache.items():
                m = eval_combo_on_code(
                    blob["events"],
                    blob["close"],
                    combo,
                    start=max(start, blob["listing"]),
                    end=end,
                    cost_per_side=float(args.cost_per_side),
                )
                if m is None:
                    continue
                rec = {"code": code, **m}
                per_code.append(rec)
            agg = aggregate_pool(
                per_code, eligible_codes=eligible if split == "train" else set(cache)
            )
            # For OOS, use all cached codes with any OOS bars (not only train-eligible).
            if split == "oos":
                oos_codes = {
                    c
                    for c, b in cache.items()
                    if int(
                        _window_mask(
                            pd.DatetimeIndex(b["close"].index), train_cutoff, as_of
                        ).sum()
                    )
                    >= 2
                }
                agg = aggregate_pool(per_code, eligible_codes=oos_codes)
                # Gate flag is train-only; do not recompute on OOS metrics.
                agg["pass_train_gate"] = False
            rows.append(combo_row(combo, agg, split=split))
            per_code_map[combo["combo_id"]] = per_code
        return pd.DataFrame(rows), per_code_map

    combos_a = stage_a_combos()
    print(f"[INFO] Stage A combos={len(combos_a)}")
    train_a, _ = run_stage(combos_a, split="train", start="2005-01-01", end=train_end)
    train_a_path = out_dir / "pair_train.csv"
    survivors = train_a[train_a["pass_train_gate"] == True].copy()  # noqa: E712
    print(f"[INFO] Stage A train survivors={len(survivors)}")

    combos_b: list[dict[str, Any]] = []
    if not args.skip_stage_b and not survivors.empty:
        combos_b = stage_b_combos(survivors)
        print(f"[INFO] Stage B combos={len(combos_b)}")
        if combos_b:
            train_b, _ = run_stage(
                combos_b, split="train", start="2005-01-01", end=train_end
            )
            train_all = pd.concat([train_a, train_b], ignore_index=True)
            survivors = train_all[train_all["pass_train_gate"] == True].copy()  # noqa: E712
        else:
            train_all = train_a
    else:
        train_all = train_a
    train_all.to_csv(train_a_path, index=False)
    print(f"[OK] wrote {train_a_path} rows={len(train_all)}")

    # Always emit leaderboards (useful when train gate yields zero survivors).
    lb_cols = [
        "combo_id",
        "stage",
        "category",
        "entry_events",
        "exit_event",
        "mean_edge",
        "mean_compound",
        "mean_bh",
        "median_n_closed",
        "frac_pos_edge",
        "mean_win",
        "n_codes",
    ]
    train_all.sort_values("mean_edge", ascending=False).head(50)[lb_cols].to_csv(
        out_dir / "leaderboard_train.csv", index=False
    )

    # Freeze survivors (+ all Stage A for OOS table transparency? Plan: pair_oos for combos).
    # Evaluate OOS for all Stage A + Stage B that were searched; highlight profitable.
    oos_combos = combos_a + combos_b
    print(f"[INFO] OOS eval combos={len(oos_combos)}")
    oos_df, per_code_oos = run_stage(
        oos_combos, split="oos", start=train_cutoff, end=as_of
    )
    oos_path = out_dir / "pair_oos.csv"
    oos_df.to_csv(oos_path, index=False)
    print(f"[OK] wrote {oos_path} rows={len(oos_df)}")
    oos_df.sort_values("mean_edge", ascending=False).head(50)[lb_cols].to_csv(
        out_dir / "leaderboard_oos.csv", index=False
    )
    oos_ungated = oos_df[oos_df["mean_edge"] > 0].sort_values(
        "mean_edge", ascending=False
    )
    oos_ungated.to_csv(out_dir / "oos_positive_ungated.csv", index=False)

    train_pass_ids = set(survivors["combo_id"].astype(str))
    oos_pass = oos_df[
        oos_df["combo_id"].isin(train_pass_ids) & (oos_df["mean_edge"] > 0)
    ].copy()
    oos_fail = oos_df[
        oos_df["combo_id"].isin(train_pass_ids) & ~(oos_df["mean_edge"] > 0)
    ].copy()
    profitable_path = out_dir / "profitable_oos.csv"
    oos_pass.sort_values("mean_edge", ascending=False).to_csv(
        profitable_path, index=False
    )
    overfit_path = out_dir / "overfit_watch.csv"
    oos_fail.sort_values("mean_edge", ascending=True).to_csv(overfit_path, index=False)
    print(f"[OK] profitable_oos={len(oos_pass)} overfit_watch={len(oos_fail)}")

    br = cluster_breakdown(oos_pass, per_code_oos, membership)
    br_path = out_dir / "cluster_breakdown.csv"
    br.to_csv(br_path, index=False)
    print(f"[OK] cluster_breakdown rows={len(br)} → {br_path}")

    survivors_path = out_dir / "train_survivors.csv"
    survivors.to_csv(survivors_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
