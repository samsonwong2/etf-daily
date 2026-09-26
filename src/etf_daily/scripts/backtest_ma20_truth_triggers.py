#!/usr/bin/env python3
"""Evaluate enter/leave-up triggers against MA20±2% chart truth.

Truth = same trend state as adaptive HTML row 1 (趋势上涨/下跌/缠绕).
Baseline trades truth enter-up / leave-up. Causal catalog events are searched
as alternate triggers that may fire earlier (lead 0–5 bars).

Does not use hold_up / dual_ma adaptive configs as signal rules.
"""
from __future__ import annotations

import argparse
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from etf_daily.lib.event_long_only_sim import (  # noqa: E402
    COST_PER_SIDE,
    simulate_long_only_events,
)
from etf_daily.lib.hold_vs_swing_bench import (  # noqa: E402
    HOLD_BH_MIN,
    HOLD_UP_FRAC_MIN,
    ROLL_BH_EXIT,
    ROLL_BH_MIN,
    ROLL_UP_FRAC_EXIT,
    ROLL_UP_FRAC_MIN,
    ROLL_WINDOW,
    attach_dual_benchmarks,
    gate_edge_key,
    label_at,
    pit_path_edge,
    rolling_hold_worthy,
)
from etf_daily.lib.html_indicator_catalog import (  # noqa: E402
    EVENT_DEFS,
    build_event_frame,
)
from etf_daily.hrp.hrp_cluster_router import normalize_code  # noqa: E402
from etf_daily.lib.ma20_truth_triggers import (  # noqa: E402
    ma20_hug_truth,
    score_trigger_pair,
    truth_entry_exit,
)
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
MAX_LEAD = 5
BASELINE_ENTRY = "trend_enter_up"
BASELINE_EXIT = "trend_leave_up"


def _load_pm():
    import importlib.util

    path = _PROJECT_ROOT / "src/etf_daily/plots/plot_regime_transition_example.py"
    spec = importlib.util.spec_from_file_location("plot_regime_pm_ma20t", path)
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
        default=PLOTLY_OUTPUTS_DIR / "20260814_up_episode_ma20truth_dualbench_pit",
    )
    p.add_argument("--hold-bh-min", type=float, default=HOLD_BH_MIN)
    p.add_argument("--hold-up-frac-min", type=float, default=HOLD_UP_FRAC_MIN)
    p.add_argument("--roll-window", type=int, default=ROLL_WINDOW)
    p.add_argument("--roll-bh-min", type=float, default=ROLL_BH_MIN)
    p.add_argument("--roll-up-frac-min", type=float, default=ROLL_UP_FRAC_MIN)
    p.add_argument("--roll-bh-exit", type=float, default=ROLL_BH_EXIT)
    p.add_argument("--roll-up-frac-exit", type=float, default=ROLL_UP_FRAC_EXIT)
    p.add_argument(
        "--no-hysteresis",
        action="store_true",
        help="disable hold_worthy hysteresis",
    )
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--code", action="append", default=None)
    p.add_argument("--cost-per-side", type=float, default=COST_PER_SIDE)
    p.add_argument("--max-lead-bars", type=int, default=MAX_LEAD)
    p.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="optional cap on stage-1 pairs (smoke)",
    )
    p.add_argument("--stage2-top-k", type=int, default=5)
    p.add_argument(
        "--walkforward-pre-oos",
        action="store_true",
        help="include HMM walkforward pre-OOS (slow)",
    )
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


def stage1_pairs() -> list[dict[str, Any]]:
    """Baseline + single-sided swaps (vary enter or exit vs baseline)."""
    names = [e.name for e in EVENT_DEFS]
    pairs: list[dict[str, Any]] = [
        {
            "combo_id": f"baseline|{BASELINE_ENTRY}|{BASELINE_EXIT}",
            "kind": "baseline",
            "entry_event": BASELINE_ENTRY,
            "exit_event": BASELINE_EXIT,
        }
    ]
    for entry in names:
        if entry == BASELINE_ENTRY:
            continue
        pairs.append(
            {
                "combo_id": f"enter|{entry}|{BASELINE_EXIT}",
                "kind": "enter_swap",
                "entry_event": entry,
                "exit_event": BASELINE_EXIT,
            }
        )
    for exit_name in names:
        if exit_name == BASELINE_EXIT:
            continue
        pairs.append(
            {
                "combo_id": f"exit|{BASELINE_ENTRY}|{exit_name}",
                "kind": "exit_swap",
                "entry_event": BASELINE_ENTRY,
                "exit_event": exit_name,
            }
        )
    return pairs


def stage2_pairs(
    train_df: pd.DataFrame, *, top_k: int = 5
) -> list[dict[str, Any]]:
    """Cross top enter swaps × top exit swaps by train enter-lead / leave-hit."""
    enters = train_df[train_df["kind"] == "enter_swap"].copy()
    exits = train_df[train_df["kind"] == "exit_swap"].copy()
    if enters.empty or exits.empty:
        return []
    enters = enters.sort_values(
        ["mean_enter_hit_rate", "mean_enter_lead"], ascending=[False, False]
    ).head(int(top_k))
    exits = exits.sort_values(
        ["mean_leave_hit_rate", "mean_leave_lead"], ascending=[False, False]
    ).head(int(top_k))
    pairs: list[dict[str, Any]] = []
    for _, er in enters.iterrows():
        for _, xr in exits.iterrows():
            entry = str(er["entry_event"])
            exit_name = str(xr["exit_event"])
            if entry == exit_name:
                continue
            pairs.append(
                {
                    "combo_id": f"cross|{entry}|{exit_name}",
                    "kind": "cross",
                    "entry_event": entry,
                    "exit_event": exit_name,
                }
            )
    return pairs


def candidate_pairs() -> list[dict[str, Any]]:
    """Backward-compatible alias used by older call sites / --max-pairs smoke. """
    return stage1_pairs()


def _window_mask(idx: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    lo = pd.Timestamp(start).normalize()
    hi = pd.Timestamp(end).normalize()
    return (idx >= lo) & (idx <= hi)


def eval_pair_on_code(
    *,
    events: pd.DataFrame,
    truth: pd.Series,
    close: pd.Series,
    pair: dict[str, Any],
    start: str,
    end: str,
    cost_per_side: float,
    max_lead_bars: int,
    hold_worthy: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    idx = pd.DatetimeIndex(close.index)
    mask = _window_mask(idx, start, end)
    if int(mask.sum()) < 2:
        return None
    sub_idx = idx[mask]
    sub_close = close.reindex(sub_idx)
    sub_truth = truth.reindex(sub_idx)
    sub_ev = events.reindex(sub_idx)

    entry_name = pair["entry_event"]
    exit_name = pair["exit_event"]
    if pair["kind"] == "baseline":
        entry, exit_sig = truth_entry_exit(sub_truth)
    else:
        if entry_name not in sub_ev.columns or exit_name not in sub_ev.columns:
            return None
        entry = sub_ev[entry_name].fillna(False).astype(bool)
        exit_sig = sub_ev[exit_name].fillna(False).astype(bool)

    metrics, _tr, eq = simulate_long_only_events(
        entry,
        exit_sig,
        sub_close,
        dates=sub_idx.strftime("%Y-%m-%d"),
        cost_per_side=cost_per_side,
    )
    metrics = attach_dual_benchmarks(
        metrics,
        close=close,
        truth=truth,
        start=start,
        end=end,
        cost_per_side=cost_per_side,
    )
    if hold_worthy is not None and "bucket" in hold_worthy.columns:
        pit = pit_path_edge(
            eq,
            close,
            hold_worthy["bucket"],
            start=start,
            end=end,
        )
        metrics.update(pit)
    else:
        metrics.setdefault("edge_pit", metrics.get("edge_cash", float("nan")))
        metrics.setdefault("pit_hold_frac", float("nan"))
        metrics.setdefault("pit_n_months", 0)
        metrics.setdefault("pit_mean_month_edge", float("nan"))
    trig_scores = score_trigger_pair(
        entry, exit_sig, sub_truth, max_lead_bars=max_lead_bars
    )
    return {**metrics, **trig_scores}


def aggregate_pool(
    per_code: list[dict[str, Any]],
    *,
    eligible_codes: set[str],
    compute_gate: bool,
    edge_key: str = "edge_bh",
    bucket: str = "all",
) -> dict[str, Any]:
    rows = [r for r in per_code if r["code"] in eligible_codes]
    empty = {
        "bucket": bucket,
        "edge_key": edge_key,
        "n_codes": 0,
        "n_eligible": int(len(eligible_codes)),
        "median_n_closed": 0.0,
        "mean_edge": float("nan"),
        "mean_edge_bh": float("nan"),
        "mean_edge_cash": float("nan"),
        "mean_edge_pit": float("nan"),
        "mean_up_capture": float("nan"),
        "mean_pit_hold_frac": float("nan"),
        "mean_compound": float("nan"),
        "mean_bh": float("nan"),
        "mean_win": float("nan"),
        "frac_pos_edge": float("nan"),
        "mean_enter_hit_rate": float("nan"),
        "mean_enter_lead": float("nan"),
        "mean_enter_precision": float("nan"),
        "mean_leave_hit_rate": float("nan"),
        "mean_leave_lead": float("nan"),
        "mean_leave_precision": float("nan"),
        "pass_train_gate": False,
    }
    if not rows:
        return empty

    def _mean(key: str) -> float:
        vals = np.asarray(
            [r[key] for r in rows if np.isfinite(r.get(key, np.nan))], dtype=float
        )
        return float(np.mean(vals)) if len(vals) else float("nan")

    edges = np.asarray(
        [r.get(edge_key, r.get("edge", np.nan)) for r in rows], dtype=float
    )
    closed = np.asarray([r["n_closed"] for r in rows], dtype=float)
    finite = np.isfinite(edges)
    frac = float(np.mean(edges[finite] > 0)) if finite.any() else float("nan")
    med_closed = float(np.median(closed))
    mean_edge = float(np.mean(edges[finite])) if finite.any() else float("nan")
    pass_gate = False
    if compute_gate and finite.any():
        pass_gate = (
            med_closed >= TRAIN_MEDIAN_CLOSED
            and mean_edge > 0.0
            and frac >= TRAIN_POS_EDGE_FRAC
        )
    return {
        "bucket": bucket,
        "edge_key": edge_key,
        "n_codes": int(len(rows)),
        "n_eligible": int(len(eligible_codes)),
        "median_n_closed": med_closed,
        "mean_edge": mean_edge,
        "mean_edge_bh": _mean("edge_bh"),
        "mean_edge_cash": _mean("edge_cash"),
        "mean_edge_pit": _mean("edge_pit"),
        "mean_up_capture": _mean("up_capture"),
        "mean_pit_hold_frac": _mean("pit_hold_frac"),
        "mean_compound": float(np.mean([r["compound"] for r in rows])),
        "mean_bh": float(np.mean([r["bh"] for r in rows])),
        "mean_win": _mean("win"),
        "frac_pos_edge": frac,
        "mean_enter_hit_rate": _mean("enter_hit_rate"),
        "mean_enter_lead": _mean("enter_mean_lead"),
        "mean_enter_precision": _mean("enter_precision"),
        "mean_leave_hit_rate": _mean("leave_hit_rate"),
        "mean_leave_lead": _mean("leave_mean_lead"),
        "mean_leave_precision": _mean("leave_precision"),
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
                "combo_id",
                "cluster",
                "n_codes",
                "mean_edge",
                "mean_enter_hit_rate",
                "mean_enter_lead",
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
        by_c: dict[int, list[dict[str, Any]]] = {}
        for r in per_code_oos.get(cid, []):
            cl = code_cluster.get(r["code"])
            if cl is None:
                continue
            by_c.setdefault(cl, []).append(r)
        for cl, items in sorted(by_c.items()):
            rows.append(
                {
                    "combo_id": cid,
                    "cluster": cl,
                    "n_codes": len(items),
                    "mean_edge": float(np.mean([x["edge"] for x in items])),
                    "mean_enter_hit_rate": float(
                        np.nanmean(
                            [
                                x["enter_hit_rate"]
                                for x in items
                                if np.isfinite(x.get("enter_hit_rate", np.nan))
                            ]
                        )
                    ),
                    "mean_enter_lead": float(
                        np.nanmean(
                            [
                                x["enter_mean_lead"]
                                for x in items
                                if np.isfinite(x.get("enter_mean_lead", np.nan))
                            ]
                        )
                    ),
                    "codes": ",".join(sorted(x["code"] for x in items)),
                }
            )
    return pd.DataFrame(rows)


def write_readme(
    path: Path,
    *,
    train_cutoff: str,
    as_of: str,
    cost: float,
    lead: int,
    roll_window: int,
    roll_bh_min: float,
    roll_up_frac_min: float,
    roll_bh_exit: float,
    roll_up_frac_exit: float,
    hysteresis: bool,
) -> None:
    text = f"""# MA20±2% truth trigger evaluation (PIT dual benchmark)

## Truth
Same as adaptive HTML chart-1 trend state (MA20±2%).
Baseline trades: truth enter-up / leave-up.

## Point-in-time hold_worthy(t)
Trailing `{roll_window}` bars (causal):
- Enter hold_ok: BH > {roll_bh_min} and up_frac ≥ {roll_up_frac_min:.0%}
- Hysteresis={'ON' if hysteresis else 'OFF'}: leave if BH ≤ {roll_bh_exit}
  or up_frac < {roll_up_frac_exit:.0%}

Pool membership for a split uses the label at the split **reference date**:
- Train → `{train_cutoff}`−1d (`train_end`)
- OOS → day before `{train_cutoff}` (known entering OOS)

## Dual benchmarks
- **hold_ok** codes → gate on `edge_bh` = compound − BH
- **swing_only** codes → gate on `edge_cash` = compound − 0
- **all** → gate on `edge_pit` (month-by-month: hold months vs BH, swing vs cash)

Also report `up_capture` = strategy / ideal MA20-up compound.

See `bucket_codes_pit.csv`.

## Triggers / split
Two-stage enter/exit search; lead ≤ {lead} bars.
Train: listing → day before `{train_cutoff}`; OOS: `{train_cutoff}` → `{as_of}`.
Cost {cost:.4f}/side; T→T+1 close.

## Train gate (per bucket)
median n_closed ≥ {TRAIN_MEDIAN_CLOSED}; mean bucket-edge > 0;
≥ {TRAIN_POS_EDGE_FRAC:.0%} codes with bucket-edge > 0.
"""
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    as_of = pd.Timestamp(args.as_of).strftime("%Y-%m-%d")
    train_cutoff = pd.Timestamp(args.train_cutoff).strftime("%Y-%m-%d")
    train_end = (pd.Timestamp(train_cutoff) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    out_dir = Path(args.html_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_lead = int(args.max_lead_bars)
    hysteresis = not bool(args.no_hysteresis)
    roll_window = int(args.roll_window)

    membership = load_membership(args.membership)
    listings = load_listings(args.listing_csv)
    codes = sorted(membership["code"].unique())
    if args.code:
        want = {normalize_code(c) for c in args.code}
        codes = [c for c in codes if c in want]
    if args.max_codes is not None:
        codes = codes[: int(args.max_codes)]

    pairs = stage1_pairs()
    if args.max_pairs is not None:
        pairs = pairs[: int(args.max_pairs)]

    print(
        f"[INFO] codes={len(codes)} stage1_pairs={len(pairs)} lead≤{max_lead}",
        flush=True,
    )
    print(f"[INFO] train_end={train_end} oos={train_cutoff}→{as_of}", flush=True)
    print(f"[INFO] out={out_dir}", flush=True)
    print(
        f"[INFO] walkforward_pre_oos={bool(args.walkforward_pre_oos)}",
        flush=True,
    )
    print(
        f"[INFO] PIT hold_worthy window={roll_window} hysteresis={hysteresis}",
        flush=True,
    )
    write_readme(
        out_dir / "README.md",
        train_cutoff=train_cutoff,
        as_of=as_of,
        cost=float(args.cost_per_side),
        lead=max_lead,
        roll_window=roll_window,
        roll_bh_min=float(args.roll_bh_min),
        roll_up_frac_min=float(args.roll_up_frac_min),
        roll_bh_exit=float(args.roll_bh_exit),
        roll_up_frac_exit=float(args.roll_up_frac_exit),
        hysteresis=hysteresis,
    )

    oos_all, _rev, _evt = load_validation_csvs(args.validation_dir)
    pm = _load_pm()
    pm._init_qlib(None)

    print(f"[INFO] loading anchor {EQUITY_ANCHOR}")
    ohlcv_a = pm.load_qlib_ohlcv(
        EQUITY_ANCHOR, "2010-01-01", as_of, init_qlib=False, clip_to_window=False
    )
    close_a = ohlcv_a.set_index(pd.to_datetime(ohlcv_a["datetime"]).dt.normalize())[
        "$close"
    ]
    close_a = close_a[~close_a.index.duplicated(keep="last")]
    vf_a = compute_volatility_regime_frame(close_a)
    vf_a["as_of"] = pd.to_datetime(vf_a["as_of"]).dt.normalize()
    vf_a = vf_a.set_index("as_of")
    rv20_anchor = vf_a["rv20"]
    rv5_anchor = vf_a["rv5"]

    cache: dict[str, dict[str, Any]] = {}
    eligible: set[str] = set()
    for i, code in enumerate(codes, 1):
        listing = listings.get(code, "2015-01-01")
        try:
            print(f"[{i}/{len(codes)}] {code} listing={listing}", flush=True)
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
            truth = ma20_hug_truth(close)
            events = build_event_frame(
                close,
                code=code,
                oos=oos_all,
                rv20_anchor=rv20_anchor,
                rv5_anchor=rv5_anchor,
                start_date=listing,
                end_date=as_of,
                walkforward_pre_oos=bool(args.walkforward_pre_oos),
            )
            hw = rolling_hold_worthy(
                close,
                truth,
                window=roll_window,
                bh_min=float(args.roll_bh_min),
                up_frac_min=float(args.roll_up_frac_min),
                hysteresis=hysteresis,
                bh_exit=float(args.roll_bh_exit),
                up_frac_exit=float(args.roll_up_frac_exit),
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
                "events": events,
                "hold_worthy": hw,
                "listing": listing,
                "n_train": n_train,
            }
            if n_train >= MIN_TRAIN_BARS:
                eligible.add(code)
            else:
                print(f"  [skip-select] train_bars={n_train}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {code}: {exc}")
            if args.fail_fast:
                raise
            traceback.print_exc()

    print(f"[INFO] eligible={len(eligible)}/{len(cache)}")

    # PIT labels at train_end (membership for train) and entering OOS.
    bucket_rows: list[dict[str, Any]] = []
    train_bucket: dict[str, str] = {}
    oos_bucket: dict[str, str] = {}
    for code in sorted(cache):
        blob = cache[code]
        hw = blob["hold_worthy"]
        b_train = label_at(hw["bucket"], train_end)
        b_oos = label_at(hw["bucket"], train_end)  # known at OOS start
        b_asof = label_at(hw["bucket"], as_of)
        train_bucket[code] = b_train
        oos_bucket[code] = b_oos
        feat_te = hw.loc[hw.index <= pd.Timestamp(train_end)]
        last = feat_te.iloc[-1] if len(feat_te) else None
        bucket_rows.append(
            {
                "code": code,
                "listing": blob["listing"],
                "n_train": blob["n_train"],
                "eligible": code in eligible,
                "bucket_train_end": b_train,
                "bucket_oos_entry": b_oos,
                "bucket_asof": b_asof,
                "bh_train_end": float(last["bh"]) if last is not None else np.nan,
                "up_frac_train_end": float(last["up_frac"])
                if last is not None
                else np.nan,
            }
        )
    bucket_df = pd.DataFrame(bucket_rows)
    bucket_df.to_csv(out_dir / "bucket_codes_pit.csv", index=False)
    hold_codes = {c for c in eligible if train_bucket.get(c) == "hold_ok"}
    swing_codes = {c for c in eligible if train_bucket.get(c) == "swing_only"}
    print(
        f"[INFO] PIT buckets @train_end: hold_ok={len(hold_codes)} "
        f"swing_only={len(swing_codes)}",
        flush=True,
    )
    if "SH515060" in train_bucket:
        row515 = bucket_df[bucket_df["code"] == "SH515060"].iloc[0]
        print(
            f"[INFO] SH515060 train_end={row515['bucket_train_end']} "
            f"asof={row515['bucket_asof']} "
            f"bh={row515['bh_train_end']:.3f} up_frac={row515['up_frac_train_end']:.3f}",
            flush=True,
        )

    def _oos_eligible() -> set[str]:
        return {
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

    def run_split(
        pairs_in: list[dict[str, Any]], split: str, start: str, end: str
    ) -> tuple[pd.DataFrame, dict]:
        rows: list[dict[str, Any]] = []
        per_map: dict[str, list[dict[str, Any]]] = {}
        oos_codes = _oos_eligible() if split == "oos" else set()
        buck_map = train_bucket if split == "train" else oos_bucket
        for j, pair in enumerate(pairs_in, 1):
            if j % 20 == 0 or j == len(pairs_in) or pair["kind"] == "baseline":
                print(f"  [{split}] {j}/{len(pairs_in)} {pair['combo_id']}", flush=True)
            per_code: list[dict[str, Any]] = []
            for code, blob in cache.items():
                m = eval_pair_on_code(
                    events=blob["events"],
                    truth=blob["truth"],
                    close=blob["close"],
                    pair=pair,
                    start=max(start, blob["listing"]),
                    end=end,
                    cost_per_side=float(args.cost_per_side),
                    max_lead_bars=max_lead,
                    hold_worthy=blob["hold_worthy"],
                )
                if m is None:
                    continue
                per_code.append(
                    {
                        "code": code,
                        "bucket": buck_map.get(code, "swing_only"),
                        **m,
                    }
                )
            if split == "train":
                specs = [
                    ("all", eligible, "edge_pit", True),
                    ("hold_ok", hold_codes, "edge_bh", True),
                    ("swing_only", swing_codes, "edge_cash", True),
                ]
            else:
                oos_hold = {c for c in oos_codes if oos_bucket.get(c) == "hold_ok"}
                oos_swing = {
                    c for c in oos_codes if oos_bucket.get(c) == "swing_only"
                }
                specs = [
                    ("all", oos_codes, "edge_pit", False),
                    ("hold_ok", oos_hold, "edge_bh", False),
                    ("swing_only", oos_swing, "edge_cash", False),
                ]
            for bucket, codes_set, edge_key, gate in specs:
                agg = aggregate_pool(
                    per_code,
                    eligible_codes=codes_set,
                    compute_gate=gate,
                    edge_key=edge_key,
                    bucket=bucket,
                )
                rows.append(
                    {
                        "split": split,
                        "combo_id": pair["combo_id"],
                        "kind": pair["kind"],
                        "entry_event": pair["entry_event"],
                        "exit_event": pair["exit_event"],
                        **agg,
                    }
                )
            per_map[pair["combo_id"]] = per_code
        return pd.DataFrame(rows), per_map

    train1, _ = run_split(pairs, "train", "2005-01-01", train_end)
    # Stage2 ranking uses full-pool (bucket=all) hit rates.
    train1_all = train1[train1["bucket"] == "all"]
    cross = stage2_pairs(train1_all, top_k=int(args.stage2_top_k))
    print(f"[INFO] stage2 cross pairs={len(cross)}", flush=True)
    if cross:
        train2, _ = run_split(cross, "train", "2005-01-01", train_end)
        train_df = pd.concat([train1, train2], ignore_index=True)
        all_pairs = pairs + cross
    else:
        train_df = train1
        all_pairs = pairs

    train_df.to_csv(out_dir / "pair_train.csv", index=False)

    lb_cols = [
        "bucket",
        "edge_key",
        "combo_id",
        "kind",
        "entry_event",
        "exit_event",
        "mean_edge",
        "mean_edge_bh",
        "mean_edge_cash",
        "mean_edge_pit",
        "mean_up_capture",
        "mean_pit_hold_frac",
        "mean_compound",
        "mean_bh",
        "median_n_closed",
        "frac_pos_edge",
        "mean_enter_hit_rate",
        "mean_enter_lead",
        "mean_enter_precision",
        "mean_leave_hit_rate",
        "mean_leave_lead",
        "mean_leave_precision",
        "n_codes",
        "pass_train_gate",
    ]

    for bucket in ("hold_ok", "swing_only", "all"):
        sub = train_df[train_df["bucket"] == bucket]
        survivors = sub[sub["pass_train_gate"] == True].copy()  # noqa: E712
        survivors.to_csv(out_dir / f"train_survivors_{bucket}.csv", index=False)
        edge_name = (
            "edge_pit"
            if bucket == "all"
            else gate_edge_key(bucket)  # type: ignore[arg-type]
        )
        print(
            f"[OK] train survivors[{bucket}]={len(survivors)} (edge={edge_name})",
            flush=True,
        )
        sub.sort_values("mean_edge", ascending=False).head(50)[
            [c for c in lb_cols if c in sub.columns]
        ].to_csv(out_dir / f"leaderboard_train_edge_{bucket}.csv", index=False)

    # Baseline snapshots per bucket
    for bucket in ("hold_ok", "swing_only", "all"):
        base = train_df[(train_df["kind"] == "baseline") & (train_df["bucket"] == bucket)]
        if not base.empty:
            cols = [
                c
                for c in [
                    "mean_edge",
                    "mean_edge_bh",
                    "mean_edge_cash",
                    "mean_edge_pit",
                    "mean_up_capture",
                    "mean_pit_hold_frac",
                    "mean_bh",
                    "frac_pos_edge",
                    "n_codes",
                ]
                if c in base.columns
            ]
            print(f"[BASELINE train {bucket}]", base.iloc[0][cols].to_dict(), flush=True)

    oos_df, per_oos = run_split(all_pairs, "oos", train_cutoff, as_of)
    oos_df.to_csv(out_dir / "pair_oos.csv", index=False)

    for bucket in ("hold_ok", "swing_only", "all"):
        sub_oos = oos_df[oos_df["bucket"] == bucket]
        sub_oos.sort_values("mean_edge", ascending=False).head(50)[
            [c for c in lb_cols if c in sub_oos.columns]
        ].to_csv(out_dir / f"leaderboard_oos_edge_{bucket}.csv", index=False)
        train_ids = set(
            train_df[
                (train_df["bucket"] == bucket)
                & (train_df["pass_train_gate"] == True)  # noqa: E712
            ]["combo_id"].astype(str)
        )
        oos_pass = sub_oos[
            sub_oos["combo_id"].isin(train_ids) & (sub_oos["mean_edge"] > 0)
        ].copy()
        oos_fail = sub_oos[
            sub_oos["combo_id"].isin(train_ids) & ~(sub_oos["mean_edge"] > 0)
        ].copy()
        oos_pass.sort_values("mean_edge", ascending=False).to_csv(
            out_dir / f"profitable_oos_{bucket}.csv", index=False
        )
        oos_fail.to_csv(out_dir / f"overfit_watch_{bucket}.csv", index=False)
        print(
            f"[OK] profitable_oos[{bucket}]={len(oos_pass)} "
            f"overfit_watch={len(oos_fail)}",
            flush=True,
        )

    # Early-enter on full pool train rows
    early = train_df[
        (train_df["bucket"] == "all")
        & (train_df["mean_enter_hit_rate"] >= 0.5)
        & (train_df["kind"].isin(["enter_swap", "cross"]))
    ].sort_values(
        ["mean_enter_lead", "mean_enter_hit_rate"], ascending=[False, False]
    )
    early.head(50)[[c for c in lb_cols if c in early.columns]].to_csv(
        out_dir / "leaderboard_train_early_enter.csv", index=False
    )

    hold_pass = pd.read_csv(out_dir / "profitable_oos_hold_ok.csv")
    br = cluster_breakdown(hold_pass, per_oos, membership)
    br.to_csv(out_dir / "cluster_breakdown_hold_ok.csv", index=False)
    swing_pass = pd.read_csv(out_dir / "profitable_oos_swing_only.csv")
    br_s = cluster_breakdown(swing_pass, per_oos, membership)
    br_s.to_csv(out_dir / "cluster_breakdown_swing_only.csv", index=False)

    for bucket in ("hold_ok", "swing_only", "all"):
        base_oos = oos_df[
            (oos_df["kind"] == "baseline") & (oos_df["bucket"] == bucket)
        ]
        if not base_oos.empty:
            cols = [
                c
                for c in [
                    "mean_edge",
                    "mean_edge_bh",
                    "mean_edge_cash",
                    "mean_edge_pit",
                    "mean_up_capture",
                    "mean_pit_hold_frac",
                    "frac_pos_edge",
                    "n_codes",
                ]
                if c in base_oos.columns
            ]
            print(f"[BASELINE oos {bucket}]", base_oos.iloc[0][cols].to_dict(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
