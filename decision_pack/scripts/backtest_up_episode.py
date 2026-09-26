#!/usr/bin/env python3
"""Backtest causal 3-state machines: long while pred==up (enter/leave up).

Truth labels use H=20 forward return (±2%) for scoring only. Machines use
only causal chart series. Does not use hold_up / dual_ma adaptive configs.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runtime_paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from decision_pack.src.causal_trend_machines import (  # noqa: E402
    MACHINE_IDS,
    machine_descriptions,
    pred_to_entry_exit,
    run_machine,
)
from decision_pack.src.event_long_only_sim import (  # noqa: E402
    COST_PER_SIDE,
    simulate_long_only_events,
)
from decision_pack.src.fwd_return_trend_truth import (  # noqa: E402
    DEFAULT_HORIZON,
    classification_scores,
    fwd_return_trend_labels,
    turn_hit_rate,
    up_episode_turns,
)
from decision_pack.src.hrp_cluster_router import normalize_code  # noqa: E402
from decision_pack.src.regime_transition_plot import (  # noqa: E402
    compute_volatility_regime_frame,
    load_validation_csvs,
)
from decision_pack.src.vol_need_return_board import EQUITY_ANCHOR  # noqa: E402

DEFAULT_MEMBERSHIP = Path(
    "~/etf-daily-output/temp/decision_packs/20260814/cluster_representatives_20260814.csv"
)
DEFAULT_VALIDATION_DIR = (
    _PROJECT_ROOT
    / "workspace/decision_packs/20260720/regime_transition_validation_q90_to0720"
)
DEFAULT_LISTING_CSV = (
    PLOTLY_OUTPUTS_DIR / "20260814_from_listing/listing_dates.csv"
)
MIN_TRAIN_BARS = 60
TRAIN_MEDIAN_CLOSED = 3
TRAIN_POS_EDGE_FRAC = 0.40
TURN_TOLERANCE = 3


def _load_pm():
    import importlib.util

    path = _PROJECT_ROOT / "decision_pack/scripts/plot_regime_transition_example.py"
    spec = importlib.util.spec_from_file_location("plot_regime_pm_up_ep", path)
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
        default=PLOTLY_OUTPUTS_DIR / "20260814_up_episode",
    )
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--code", action="append", default=None)
    p.add_argument("--cost-per-side", type=float, default=COST_PER_SIDE)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
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


def load_listings(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "code" not in df.columns or "listing" not in df.columns:
        return {}
    return {
        normalize_code(r.code): pd.Timestamp(r.listing).strftime("%Y-%m-%d")
        for r in df.itertuples(index=False)
    }


def _window_slice(
    close: pd.Series, start: str, end: str
) -> tuple[pd.Series, pd.DatetimeIndex]:
    lo = pd.Timestamp(start).normalize()
    hi = pd.Timestamp(end).normalize()
    sub = close[(close.index >= lo) & (close.index <= hi)]
    return sub, pd.DatetimeIndex(sub.index)


def eval_machine_on_code(
    *,
    pred_full: pd.Series,
    close: pd.Series,
    truth_full: pd.Series,
    start: str,
    end: str,
    cost_per_side: float,
) -> dict[str, Any] | None:
    sub_px, idx = _window_slice(close, start, end)
    if len(sub_px) < 2:
        return None
    pred = pred_full.reindex(idx)
    truth = truth_full.reindex(idx)
    entry, exit_sig = pred_to_entry_exit(pred)
    metrics, _trades, _eq = simulate_long_only_events(
        entry,
        exit_sig,
        sub_px,
        dates=idx.strftime("%Y-%m-%d"),
        cost_per_side=cost_per_side,
    )
    scores = classification_scores(pred, truth)
    truth_turns = up_episode_turns(truth)
    pred_turns = up_episode_turns(pred.where(pred.notna(), other="range"))
    # For turn hits, treat NaN pred as range so turns are defined on scored window.
    enter_hit = turn_hit_rate(
        pred_turns["enter_up"],
        truth_turns["enter_up"],
        tolerance_bars=TURN_TOLERANCE,
        calendar=idx,
    )
    leave_hit = turn_hit_rate(
        pred_turns["leave_up"],
        truth_turns["leave_up"],
        tolerance_bars=TURN_TOLERANCE,
        calendar=idx,
    )
    return {
        **metrics,
        **scores,
        "enter_up_hit_rate": enter_hit["hit_rate"],
        "enter_up_n_truth": enter_hit["n_truth"],
        "leave_up_hit_rate": leave_hit["hit_rate"],
        "leave_up_n_truth": leave_hit["n_truth"],
    }


def aggregate_pool(
    per_code: list[dict[str, Any]],
    *,
    eligible_codes: set[str],
    compute_gate: bool,
) -> dict[str, Any]:
    rows = [r for r in per_code if r["code"] in eligible_codes]
    empty = {
        "n_codes": 0,
        "n_eligible": int(len(eligible_codes)),
        "median_n_closed": 0.0,
        "mean_edge": float("nan"),
        "mean_compound": float("nan"),
        "mean_bh": float("nan"),
        "mean_win": float("nan"),
        "frac_pos_edge": float("nan"),
        "mean_accuracy": float("nan"),
        "mean_up_precision": float("nan"),
        "mean_up_recall": float("nan"),
        "mean_enter_up_hit_rate": float("nan"),
        "mean_leave_up_hit_rate": float("nan"),
        "pass_train_gate": False,
    }
    if not rows:
        return empty
    edges = np.asarray([r["edge"] for r in rows], dtype=float)
    closed = np.asarray([r["n_closed"] for r in rows], dtype=float)
    compounds = np.asarray([r["compound"] for r in rows], dtype=float)
    bhs = np.asarray([r["bh"] for r in rows], dtype=float)
    wins = np.asarray(
        [r["win"] if np.isfinite(r.get("win", np.nan)) else np.nan for r in rows],
        dtype=float,
    )

    def _mean_key(key: str) -> float:
        vals = np.asarray(
            [r[key] for r in rows if np.isfinite(r.get(key, np.nan))], dtype=float
        )
        return float(np.mean(vals)) if len(vals) else float("nan")

    frac = float(np.mean(edges > 0))
    med_closed = float(np.median(closed))
    mean_edge = float(np.mean(edges))
    pass_gate = False
    if compute_gate:
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
        "mean_accuracy": _mean_key("accuracy"),
        "mean_up_precision": _mean_key("up_precision"),
        "mean_up_recall": _mean_key("up_recall"),
        "mean_enter_up_hit_rate": _mean_key("enter_up_hit_rate"),
        "mean_leave_up_hit_rate": _mean_key("leave_up_hit_rate"),
        "pass_train_gate": bool(pass_gate),
    }


def cluster_breakdown(
    profitable: pd.DataFrame,
    per_code_oos: dict[str, list[dict[str, Any]]],
    membership: pd.DataFrame,
) -> pd.DataFrame:
    if profitable.empty:
        return pd.DataFrame(
            columns=[
                "machine_id",
                "cluster",
                "n_codes",
                "mean_edge",
                "mean_win",
                "mean_n_closed",
                "mean_accuracy",
                "codes",
            ]
        )
    code_cluster = {
        normalize_code(r.code): int(r.cluster)
        for r in membership.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for _, prow in profitable.iterrows():
        mid = str(prow["machine_id"])
        for cl, items in _group_by_cluster(per_code_oos.get(mid, []), code_cluster):
            rows.append(
                {
                    "machine_id": mid,
                    "cluster": cl,
                    "n_codes": len(items),
                    "mean_edge": float(np.mean([x["edge"] for x in items])),
                    "mean_win": float(
                        np.nanmean(
                            [
                                x["win"]
                                for x in items
                                if np.isfinite(x.get("win", np.nan))
                            ]
                        )
                    ),
                    "mean_n_closed": float(np.mean([x["n_closed"] for x in items])),
                    "mean_accuracy": float(
                        np.nanmean(
                            [
                                x["accuracy"]
                                for x in items
                                if np.isfinite(x.get("accuracy", np.nan))
                            ]
                        )
                    ),
                    "codes": ",".join(sorted(x["code"] for x in items)),
                }
            )
    return pd.DataFrame(rows)


def _group_by_cluster(
    code_rows: list[dict[str, Any]], code_cluster: dict[str, int]
) -> list[tuple[int, list[dict[str, Any]]]]:
    by_c: dict[int, list[dict[str, Any]]] = {}
    for r in code_rows:
        cl = code_cluster.get(r["code"])
        if cl is None:
            continue
        by_c.setdefault(cl, []).append(r)
    return sorted(by_c.items())


def write_readme(path: Path, *, train_cutoff: str, as_of: str, cost: float, h: int) -> None:
    lines = [
        "# Up-episode trend backtest",
        "",
        "## Problem",
        "Three states {up, down, range}. Long-only → trade only enter-up / leave-up.",
        "",
        "## Truth (scoring only)",
        f"- Horizon H={h}: r = close[T+H]/close[T]−1",
        "- r > +2% → up; r < −2% → down; else range",
        "- Last H bars unlabeled (excluded from accuracy)",
        "",
        "## Machines (causal)",
    ]
    for d in machine_descriptions():
        lines.append(f"- `{d['id']}`: {d['description']}")
    lines.extend(
        [
            "",
            "## Split / cost",
            f"- Train: listing → day before `{train_cutoff}` "
            f"(exclude codes with < {MIN_TRAIN_BARS} train bars from selection)",
            f"- OOS freeze: `{train_cutoff}` → `{as_of}`",
            f"- Signal T close → fill T+1 close; cost {cost:.4f}/side",
            "",
            "## Train gate (pool equal-weight)",
            f"- median n_closed ≥ {TRAIN_MEDIAN_CLOSED}",
            "- mean edge (compound − BH) > 0",
            f"- ≥ {TRAIN_POS_EDGE_FRAC:.0%} codes with edge > 0",
            "",
            "## Profit definition",
            "Pass train gate AND OOS mean edge > 0 → profitable_oos.csv.",
            "Train-pass / OOS-fail → overfit_watch.csv.",
            "",
            "## Excluded",
            "- hold_up / dual_ma / adaptive method configs as signal source",
            "- T+3/T+10 reversal diamonds, ED/RU early marks",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    as_of = pd.Timestamp(args.as_of).strftime("%Y-%m-%d")
    train_cutoff = pd.Timestamp(args.train_cutoff).strftime("%Y-%m-%d")
    train_end = (pd.Timestamp(train_cutoff) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    out_dir = Path(args.html_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    horizon = int(args.horizon)

    membership = load_membership(args.membership)
    listings = load_listings(args.listing_csv)
    codes = sorted(membership["code"].unique())
    if args.code:
        want = {normalize_code(c) for c in args.code}
        codes = [c for c in codes if c in want]
    if args.max_codes is not None:
        codes = codes[: int(args.max_codes)]

    print(f"[INFO] codes={len(codes)} train_end={train_end} oos={train_cutoff}→{as_of}")
    print(f"[INFO] machines={list(MACHINE_IDS)} horizon={horizon}")
    print(f"[INFO] out={out_dir}")
    write_readme(
        out_dir / "README.md",
        train_cutoff=train_cutoff,
        as_of=as_of,
        cost=float(args.cost_per_side),
        h=horizon,
    )
    pd.DataFrame(machine_descriptions()).to_csv(out_dir / "machines.csv", index=False)

    oos_all, _rev, _evt = load_validation_csvs(args.validation_dir)
    pm = _load_pm()
    pm._init_qlib(None)

    print(f"[INFO] loading anchor {EQUITY_ANCHOR}")
    ohlcv_anchor = pm.load_qlib_ohlcv(
        EQUITY_ANCHOR, "2010-01-01", as_of, init_qlib=False, clip_to_window=False
    )
    close_a = ohlcv_anchor.set_index(
        pd.to_datetime(ohlcv_anchor["datetime"]).dt.normalize()
    )["$close"]
    close_a = close_a[~close_a.index.duplicated(keep="last")]
    vf_a = compute_volatility_regime_frame(close_a)
    vf_a["as_of"] = pd.to_datetime(vf_a["as_of"]).dt.normalize()
    vf_a = vf_a.set_index("as_of")
    rv20_anchor = vf_a["rv20"]

    cache: dict[str, dict[str, Any]] = {}
    eligible: set[str] = set()
    for i, code in enumerate(codes, 1):
        listing = listings.get(code, "2015-01-01")
        try:
            print(f"[{i}/{len(codes)}] {code} listing={listing}")
            ohlcv = pm.load_qlib_ohlcv(
                code, listing, as_of, init_qlib=False, clip_to_window=False
            )
            close = ohlcv.set_index(pd.to_datetime(ohlcv["datetime"]).dt.normalize())[
                "$close"
            ]
            close = close[~close.index.duplicated(keep="last")].sort_index()
            close = close[
                (close.index >= pd.Timestamp(listing).normalize())
                & (close.index <= pd.Timestamp(as_of).normalize())
            ]
            truth = fwd_return_trend_labels(close, horizon=horizon)
            preds: dict[str, pd.Series] = {}
            for mid in MACHINE_IDS:
                preds[mid] = run_machine(
                    mid,
                    close,
                    code=code,
                    oos=oos_all,
                    rv20_anchor=rv20_anchor,
                    start_date=listing,
                    end_date=as_of,
                )
            n_train = int(
                (
                    (close.index >= pd.Timestamp(listing).normalize())
                    & (close.index <= pd.Timestamp(train_end).normalize())
                ).sum()
            )
            cache[code] = {
                "close": close,
                "truth": truth,
                "preds": preds,
                "listing": listing,
                "n_train": n_train,
            }
            if n_train >= MIN_TRAIN_BARS:
                eligible.add(code)
            else:
                print(f"  [skip-select] train_bars={n_train} < {MIN_TRAIN_BARS}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {code}: {exc}")
            if args.fail_fast:
                raise
            traceback.print_exc()

    print(f"[INFO] eligible={len(eligible)}/{len(cache)}")

    def run_split(split: str, start: str, end: str) -> tuple[pd.DataFrame, dict[str, list]]:
        rows: list[dict[str, Any]] = []
        per_code_map: dict[str, list[dict[str, Any]]] = {}
        for mid in MACHINE_IDS:
            per_code: list[dict[str, Any]] = []
            for code, blob in cache.items():
                win_start = max(start, blob["listing"])
                m = eval_machine_on_code(
                    pred_full=blob["preds"][mid],
                    close=blob["close"],
                    truth_full=blob["truth"],
                    start=win_start,
                    end=end,
                    cost_per_side=float(args.cost_per_side),
                )
                if m is None:
                    continue
                per_code.append({"code": code, **m})
            if split == "train":
                agg = aggregate_pool(
                    per_code, eligible_codes=eligible, compute_gate=True
                )
            else:
                oos_codes = {
                    c
                    for c, b in cache.items()
                    if int(
                        (
                            (b["close"].index >= pd.Timestamp(train_cutoff).normalize())
                            & (b["close"].index <= pd.Timestamp(as_of).normalize())
                        ).sum()
                    )
                    >= 2
                }
                agg = aggregate_pool(
                    per_code, eligible_codes=oos_codes, compute_gate=False
                )
            rows.append({"split": split, "machine_id": mid, **agg})
            per_code_map[mid] = per_code
            print(
                f"  [{split}] {mid}: edge={agg['mean_edge']:.4f} "
                f"acc={agg['mean_accuracy']:.3f} "
                f"gate={agg['pass_train_gate']} n={agg['n_codes']}"
            )
        return pd.DataFrame(rows), per_code_map

    train_df, _ = run_split("train", "2005-01-01", train_end)
    train_path = out_dir / "machine_train.csv"
    train_df.to_csv(train_path, index=False)
    survivors = train_df[train_df["pass_train_gate"] == True].copy()  # noqa: E712
    survivors.to_csv(out_dir / "train_survivors.csv", index=False)
    print(f"[OK] train survivors={len(survivors)} → {train_path}")

    oos_df, per_code_oos = run_split("oos", train_cutoff, as_of)
    oos_path = out_dir / "machine_oos.csv"
    oos_df.to_csv(oos_path, index=False)
    print(f"[OK] oos → {oos_path}")

    lb_cols = [
        "machine_id",
        "mean_edge",
        "mean_compound",
        "mean_bh",
        "median_n_closed",
        "frac_pos_edge",
        "mean_accuracy",
        "mean_up_precision",
        "mean_up_recall",
        "mean_enter_up_hit_rate",
        "mean_leave_up_hit_rate",
        "n_codes",
    ]
    train_df.sort_values("mean_edge", ascending=False)[lb_cols].to_csv(
        out_dir / "leaderboard_train.csv", index=False
    )
    oos_df.sort_values("mean_edge", ascending=False)[lb_cols].to_csv(
        out_dir / "leaderboard_oos.csv", index=False
    )

    train_pass_ids = set(survivors["machine_id"].astype(str))
    oos_pass = oos_df[
        oos_df["machine_id"].isin(train_pass_ids) & (oos_df["mean_edge"] > 0)
    ].copy()
    oos_fail = oos_df[
        oos_df["machine_id"].isin(train_pass_ids) & ~(oos_df["mean_edge"] > 0)
    ].copy()
    oos_pass.sort_values("mean_edge", ascending=False).to_csv(
        out_dir / "profitable_oos.csv", index=False
    )
    oos_fail.sort_values("mean_edge", ascending=True).to_csv(
        out_dir / "overfit_watch.csv", index=False
    )
    print(f"[OK] profitable_oos={len(oos_pass)} overfit_watch={len(oos_fail)}")

    br = cluster_breakdown(oos_pass, per_code_oos, membership)
    br.to_csv(out_dir / "cluster_breakdown.csv", index=False)
    print(f"[OK] cluster_breakdown rows={len(br)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
