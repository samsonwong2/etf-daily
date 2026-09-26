#!/usr/bin/env python3
"""Layered gate: equal-weight 46-name pool vs SH510300 (CSI300 ETF).

Layer 1: Does buy-and-hold SH510300 beat cash and its fair need?
Layer 2 (veto): Monthly equal-weight of *listed* representatives vs SH510300.
Appendix: cluster-balanced EW and optional HRP (report only, not veto).

Does not change production adaptive rules or from_listing HTML.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runtime_paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from decision_pack.src.event_long_only_sim import COST_PER_SIDE  # noqa: E402
from decision_pack.src.hold_vs_swing_bench import decision_dates  # noqa: E402
from decision_pack.src.hrp_cluster_router import (  # noqa: E402
    load_cluster_membership,
    normalize_code,
)
from decision_pack.src.regime_transition_plot import (  # noqa: E402
    TRADING_DAYS_PER_YEAR,
    compute_volatility_regime_frame,
)
from decision_pack.src.vol_need_return_board import (  # noqa: E402
    ANCHOR_ERP_ANN,
    EQUITY_ANCHOR,
)

DEFAULT_MEMBERSHIP = Path(
    "~/etf-daily-output/temp/decision_packs/20260814/cluster_representatives_20260814.csv"
)
DEFAULT_LISTING_CSV = (
    PLOTLY_OUTPUTS_DIR / "20260814_from_listing/listing_dates.csv"
)
HRP_LOOKBACK = 126


def _load_pm():
    import importlib.util

    path = _PROJECT_ROOT / "decision_pack/scripts/plot_regime_transition_example.py"
    spec = importlib.util.spec_from_file_location("plot_regime_pm_pool", path)
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
        default=PLOTLY_OUTPUTS_DIR / "20260814_pool_vs_anchor",
    )
    p.add_argument("--cost-per-side", type=float, default=COST_PER_SIDE)
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--code", action="append", default=None)
    p.add_argument("--skip-hrp", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    return p.parse_args(argv)


def load_listings(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {
        normalize_code(r.code): pd.Timestamp(r.listing).strftime("%Y-%m-%d")
        for r in df.itertuples(index=False)
    }


def _norm_close(close: pd.Series) -> pd.Series:
    s = pd.to_numeric(close, errors="coerce").copy()
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).normalize()
    return s[~s.index.duplicated(keep="last")].sort_index()


def window_bh(close: pd.Series, start: str, end: str) -> dict[str, Any]:
    s = _norm_close(close)
    lo = pd.Timestamp(start).normalize()
    hi = pd.Timestamp(end).normalize()
    sub = s[(s.index >= lo) & (s.index <= hi)].dropna()
    n = int(len(sub))
    if n < 2:
        return {"n_bars": n, "bh": float("nan"), "start": start, "end": end}
    p0, p1 = float(sub.iloc[0]), float(sub.iloc[-1])
    bh = float(p1 / p0 - 1.0) if p0 > 0 and np.isfinite(p0) and np.isfinite(p1) else float("nan")
    return {
        "n_bars": n,
        "bh": bh,
        "start": sub.index[0].strftime("%Y-%m-%d"),
        "end": sub.index[-1].strftime("%Y-%m-%d"),
    }


def period_need_from_ann(r_ann: float, n_bars: int) -> float:
    if not np.isfinite(r_ann) or r_ann <= -1.0 or n_bars <= 0:
        return float("nan")
    return float((1.0 + float(r_ann)) ** (float(n_bars) / float(TRADING_DAYS_PER_YEAR)) - 1.0)


def layer1_anchor(
    close_a: pd.Series,
    *,
    train_start: str,
    train_end: str,
    oos_start: str,
    oos_end: str,
    erp_ann: float = ANCHOR_ERP_ANN,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, lo, hi in (
        ("train", train_start, train_end),
        ("oos", oos_start, oos_end),
    ):
        m = window_bh(close_a, lo, hi)
        need = period_need_from_ann(float(erp_ann), int(m["n_bars"]))
        bh = float(m["bh"]) if np.isfinite(m["bh"]) else float("nan")
        rows.append(
            {
                "split": split,
                "code": EQUITY_ANCHOR,
                "n_bars": m["n_bars"],
                "start": m["start"],
                "end": m["end"],
                "bh": bh,
                "cash": 0.0,
                "edge_vs_cash": bh,
                "need_period": need,
                "edge_vs_need": float(bh - need)
                if np.isfinite(bh) and np.isfinite(need)
                else float("nan"),
                "erp_ann": float(erp_ann),
            }
        )
    return pd.DataFrame(rows)


def listed_on(
    codes: list[str],
    listings: dict[str, str],
    asof: pd.Timestamp,
) -> list[str]:
    out = []
    for c in codes:
        listing = listings.get(c)
        if listing is None:
            out.append(c)
            continue
        if pd.Timestamp(listing).normalize() <= asof.normalize():
            out.append(c)
    return out


def equal_weights(codes: list[str]) -> dict[str, float]:
    if not codes:
        return {}
    w = 1.0 / float(len(codes))
    return {c: w for c in codes}


def cluster_balanced_weights(
    codes: list[str],
    membership: pd.DataFrame,
) -> dict[str, float]:
    """Equal weight across clusters, equal within cluster."""
    sub = membership[membership["code"].isin(codes)].copy()
    if sub.empty:
        return equal_weights(codes)
    clusters = sorted(sub["cluster"].astype(int).unique())
    if not clusters:
        return equal_weights(codes)
    w_c = 1.0 / float(len(clusters))
    out: dict[str, float] = {}
    for cl in clusters:
        members = sorted(sub.loc[sub["cluster"].astype(int) == cl, "code"].tolist())
        if not members:
            continue
        w_i = w_c / float(len(members))
        for c in members:
            out[c] = w_i
    # Renormalize in case of missing
    s = sum(out.values())
    if s > 0:
        out = {k: v / s for k, v in out.items()}
    return out


def hrp_weights(
    prices: pd.DataFrame,
    codes: list[str],
    asof: pd.Timestamp,
    *,
    lookback: int = HRP_LOOKBACK,
) -> tuple[dict[str, float], str | None]:
    """Fit skfolio HRP on past lookback returns through asof. Failure → message."""
    try:
        from skfolio.optimization import HierarchicalRiskParity
        from skfolio.preprocessing import prices_to_returns
    except Exception as exc:  # noqa: BLE001
        return {}, f"skfolio_import_failed: {exc}"
    use = [c for c in codes if c in prices.columns]
    if len(use) < 2:
        return {}, "need_at_least_2_codes"
    px = prices[use].loc[:asof].dropna(how="any")
    if len(px) < lookback + 2:
        px = prices[use].loc[:asof].ffill().dropna(how="any")
    if len(px) < max(lookback // 2, 40):
        return {}, f"insufficient_history n={len(px)}"
    px = px.iloc[-lookback:]
    try:
        rets = prices_to_returns(px)
        model = HierarchicalRiskParity()
        model.fit(rets)
        w = np.asarray(model.weights_, dtype=float)
        if len(w) != len(use) or not np.all(np.isfinite(w)) or float(np.sum(w)) <= 0:
            return {}, "invalid_weights"
        w = w / float(np.sum(w))
        return {c: float(wi) for c, wi in zip(use, w)}, None
    except Exception as exc:  # noqa: BLE001
        return {}, f"hrp_fit_failed: {exc}"


def simulate_monthly_rebalance(
    prices: pd.DataFrame,
    anchor: pd.Series,
    *,
    weight_fn,
    start: str,
    end: str,
    listings: dict[str, str],
    membership: pd.DataFrame | None = None,
    cost_per_side: float = COST_PER_SIDE,
    scheme: str = "equal",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Monthly signal → T+1 close rebalance; long-only fully invested.

    ``weight_fn(codes, asof) -> (weights dict, optional_error)``.
    """
    px = prices.copy()
    px.index = pd.DatetimeIndex(pd.to_datetime(px.index)).normalize()
    px = px.sort_index()
    ax = _norm_close(anchor).reindex(px.index).ffill()
    lo = pd.Timestamp(start).normalize()
    hi = pd.Timestamp(end).normalize()
    cal = px.index[(px.index >= lo) & (px.index <= hi)]
    if len(cal) < 2:
        empty = pd.DataFrame(columns=["date", "equity", "anchor", "excess_cum"])
        return empty, {"scheme": scheme, "n_bars": 0, "bh": float("nan"), "anchor_bh": float("nan"), "excess": float("nan")}

    dec = decision_dates(cal, freq="M")
    # Map decision date → next trading bar for execution
    exec_map: dict[pd.Timestamp, pd.Timestamp] = {}
    for d in dec:
        if d not in cal:
            prior = cal[cal <= d]
            if len(prior) == 0:
                continue
            d = prior[-1]
        pos = cal.get_loc(d)
        if isinstance(pos, slice):
            continue
        if pos + 1 < len(cal):
            exec_map[pd.Timestamp(d)] = pd.Timestamp(cal[pos + 1])

    wealth = 1.0
    weights: dict[str, float] = {}
    equities = []
    costs_paid = 0.0
    n_rebalances = 0
    hrp_errors: list[str] = []

    # Pending target from signal day
    pending_exec: pd.Timestamp | None = None
    pending_w: dict[str, float] | None = None

    # Seed: deploy on first bar so portfolio start matches anchor start date.
    listed0 = listed_on(list(px.columns), listings, cal[0])
    listed0 = [
        c
        for c in listed0
        if c in px.columns and np.isfinite(px.at[cal[0], c])
    ]
    w0, err0 = weight_fn(listed0, cal[0])
    if err0:
        hrp_errors.append(f"{cal[0].date()}:seed:{err0}")
    elif w0:
        turnover = sum(abs(w) for w in w0.values())
        cost = float(turnover) * float(cost_per_side)
        wealth *= 1.0 - cost
        costs_paid += cost
        weights = {c: w for c, w in w0.items() if w > 1e-12}
        n_rebalances += 1

    for i, dt in enumerate(cal):
        # Execute pending rebalance at today's close
        if pending_exec is not None and dt == pending_exec and pending_w is not None:
            # Mark portfolio to close before trade, then pay turnover cost
            if weights and i > 0:
                prev = cal[i - 1]
                port_ret = 0.0
                for c, w in weights.items():
                    if c not in px.columns:
                        continue
                    p0, p1 = px.at[prev, c], px.at[dt, c]
                    if np.isfinite(p0) and p0 > 0 and np.isfinite(p1):
                        port_ret += float(w) * (float(p1) / float(p0) - 1.0)
                wealth *= 1.0 + port_ret
            turnover = 0.0
            all_codes = set(weights) | set(pending_w)
            for c in all_codes:
                turnover += abs(pending_w.get(c, 0.0) - weights.get(c, 0.0))
            cost = float(turnover) * float(cost_per_side)
            wealth *= 1.0 - cost
            costs_paid += cost
            weights = {c: w for c, w in pending_w.items() if w > 1e-12}
            n_rebalances += 1
            pending_exec = None
            pending_w = None
        elif weights and i > 0:
            prev = cal[i - 1]
            port_ret = 0.0
            for c, w in weights.items():
                if c not in px.columns:
                    continue
                p0, p1 = px.at[prev, c], px.at[dt, c]
                if np.isfinite(p0) and p0 > 0 and np.isfinite(p1):
                    port_ret += float(w) * (float(p1) / float(p0) - 1.0)
            wealth *= 1.0 + port_ret

        # Signal on month-end decision dates
        if dt in exec_map and pending_exec is None:
            listed = listed_on(list(px.columns), listings, dt)
            listed = [c for c in listed if c in px.columns and np.isfinite(px.at[dt, c])]
            w_new, err = weight_fn(listed, dt)
            if err:
                hrp_errors.append(f"{dt.date()}:{err}")
                # keep previous weights
            elif w_new:
                pending_w = w_new
                pending_exec = exec_map[dt]

        a0 = float(ax.loc[cal[0]]) if cal[0] in ax.index else float(ax.iloc[0])
        a_t = float(ax.loc[dt]) if dt in ax.index else float("nan")
        anchor_nav = (
            float(a_t / a0) if np.isfinite(a0) and a0 > 0 and np.isfinite(a_t) else float("nan")
        )
        equities.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "equity": wealth,
                "anchor": anchor_nav,
                "excess_cum": wealth - anchor_nav
                if np.isfinite(anchor_nav)
                else float("nan"),
            }
        )

    eq = pd.DataFrame(equities)
    if eq.empty or len(eq) < 2:
        meta = {
            "scheme": scheme,
            "n_bars": int(len(eq)),
            "bh": float("nan"),
            "anchor_bh": float("nan"),
            "excess": float("nan"),
            "n_rebalances": n_rebalances,
            "total_cost_drag": costs_paid,
        }
        return eq, meta

    bh = float(eq["equity"].iloc[-1] - 1.0)
    a_bh = float(eq["anchor"].iloc[-1] - 1.0) if np.isfinite(eq["anchor"].iloc[-1]) else float("nan")
    excess = float(bh - a_bh) if np.isfinite(a_bh) else float("nan")

    # Month-end 63-bar forward excess (label only), for gap-style appendix
    fwd_rows = []
    for d in dec:
        if d not in eq.index and d.strftime("%Y-%m-%d") not in set(eq["date"]):
            # find date string
            pass
        ds = pd.Timestamp(d).strftime("%Y-%m-%d")
        hit = eq.index[eq["date"] == ds]
        if len(hit) == 0:
            continue
        i0 = int(hit[0])
        if i0 + 63 >= len(eq):
            continue
        e0, e1 = float(eq["equity"].iloc[i0]), float(eq["equity"].iloc[i0 + 63])
        a0v, a1v = float(eq["anchor"].iloc[i0]), float(eq["anchor"].iloc[i0 + 63])
        if not all(np.isfinite(x) and x > 0 for x in (e0, e1, a0v, a1v)):
            continue
        fwd_rows.append((e1 / e0 - 1.0) - (a1v / a0v - 1.0))

    meta = {
        "scheme": scheme,
        "n_bars": int(len(eq)),
        "start": eq["date"].iloc[0],
        "end": eq["date"].iloc[-1],
        "bh": bh,
        "anchor_bh": a_bh,
        "excess": excess,
        "mean_fwd63_excess": float(np.mean(fwd_rows)) if fwd_rows else float("nan"),
        "n_fwd63": int(len(fwd_rows)),
        "n_rebalances": int(n_rebalances),
        "total_cost_drag": float(costs_paid),
        "hrp_n_errors": int(len(hrp_errors)),
        "hrp_errors_sample": ";".join(hrp_errors[:5]),
    }
    return eq, meta


def write_readme(path: Path) -> None:
    path.write_text(
        """# Pool equal-weight vs SH510300 gate

## Layer 1
SH510300 buy-and-hold vs cash (0) and vs period fair need (4.3% ann).

## Layer 2 (veto)
Monthly equal-weight of *listed* cluster representatives, T+1 close rebalance,
cost 0.001 × turnover. Excess = portfolio BH − SH510300 BH.

**Pass only if train excess > 0 and OOS excess > 0.**
Else: hold SH510300; do not pick names from charts.

## Appendix
Cluster-balanced EW and optional HRP — report only, not veto.
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

    membership = load_cluster_membership(args.membership)
    listings = load_listings(args.listing_csv)
    codes = sorted(membership["code"].unique())
    if args.code:
        want = {normalize_code(c) for c in args.code}
        codes = [c for c in codes if c in want]
    if args.max_codes is not None:
        codes = codes[: int(args.max_codes)]
    mem = membership[membership["code"].isin(codes)].copy()

    print(f"[INFO] codes={len(codes)} train_end={train_end} oos={train_cutoff}→{as_of}")
    print(f"[INFO] out={out_dir}", flush=True)

    pm = _load_pm()
    pm._init_qlib(None)

    print(f"[INFO] loading anchor {EQUITY_ANCHOR}", flush=True)
    ohlcv_a = pm.load_qlib_ohlcv(
        EQUITY_ANCHOR, "2010-01-01", as_of, init_qlib=False, clip_to_window=False
    )
    close_a = _norm_close(
        ohlcv_a.set_index(pd.to_datetime(ohlcv_a["datetime"]).dt.normalize())["$close"]
    )

    # --- Layer 1 ---
    # Earliest listing among pool for train start
    listing_dates = [
        pd.Timestamp(listings.get(c, "2015-01-01")) for c in codes
    ]
    train_start = min(listing_dates).strftime("%Y-%m-%d") if listing_dates else "2015-01-01"
    l1 = layer1_anchor(
        close_a,
        train_start=train_start,
        train_end=train_end,
        oos_start=train_cutoff,
        oos_end=as_of,
    )
    l1.to_csv(out_dir / "layer1_anchor.csv", index=False)
    print("[OK] layer1:", flush=True)
    print(l1.to_string(index=False), flush=True)

    # --- Load price panel ---
    series: dict[str, pd.Series] = {}
    for i, code in enumerate(codes, 1):
        listing = listings.get(code, "2015-01-01")
        try:
            print(f"[{i}/{len(codes)}] {code}", flush=True)
            ohlcv = pm.load_qlib_ohlcv(
                code, listing, as_of, init_qlib=False, clip_to_window=False
            )
            close = _norm_close(
                ohlcv.set_index(pd.to_datetime(ohlcv["datetime"]).dt.normalize())[
                    "$close"
                ]
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
    # Union calendar with anchor; ffill after first valid (listed) print
    union = pd.DatetimeIndex(sorted(set(prices.index) | set(close_a.index)))
    prices = prices.reindex(union).sort_index()
    prices = prices.loc[
        (prices.index >= pd.Timestamp(train_start))
        & (prices.index <= pd.Timestamp(as_of))
    ]
    prices = prices.ffill()

    def ew_fn(listed: list[str], _asof: pd.Timestamp):
        return equal_weights(listed), None

    def cluster_fn(listed: list[str], _asof: pd.Timestamp):
        return cluster_balanced_weights(listed, mem), None

    def hrp_fn(listed: list[str], asof: pd.Timestamp):
        return hrp_weights(prices, listed, asof, lookback=HRP_LOOKBACK)

    summary_rows: list[dict[str, Any]] = []

    # Layer 2 EW on train and OOS separately (fresh start each split)
    for split, lo, hi in (
        ("train", train_start, train_end),
        ("oos", train_cutoff, as_of),
    ):
        eq, meta = simulate_monthly_rebalance(
            prices,
            close_a,
            weight_fn=ew_fn,
            start=lo,
            end=hi,
            listings=listings,
            cost_per_side=float(args.cost_per_side),
            scheme="equal",
        )
        meta["split"] = split
        summary_rows.append(meta)
        eq.to_csv(out_dir / f"layer2_ew_{split}.csv", index=False)
        print(
            f"[OK] EW {split}: bh={meta['bh']:.4f} "
            f"anchor={meta['anchor_bh']:.4f} excess={meta['excess']:.4f}",
            flush=True,
        )

    ew_df = pd.DataFrame([r for r in summary_rows if r["scheme"] == "equal"])
    ew_df.to_csv(out_dir / "layer2_ew.csv", index=False)

    # Appendix cluster EW
    cluster_rows = []
    for split, lo, hi in (
        ("train", train_start, train_end),
        ("oos", train_cutoff, as_of),
    ):
        eq, meta = simulate_monthly_rebalance(
            prices,
            close_a,
            weight_fn=cluster_fn,
            start=lo,
            end=hi,
            listings=listings,
            membership=mem,
            cost_per_side=float(args.cost_per_side),
            scheme="cluster_ew",
        )
        meta["split"] = split
        cluster_rows.append(meta)
        eq.to_csv(out_dir / f"appendix_cluster_ew_{split}.csv", index=False)
        print(
            f"[OK] cluster_ew {split}: excess={meta['excess']:.4f}",
            flush=True,
        )
    pd.DataFrame(cluster_rows).to_csv(out_dir / "appendix_cluster_ew.csv", index=False)

    # Appendix HRP
    hrp_rows = []
    if not args.skip_hrp:
        for split, lo, hi in (
            ("train", train_start, train_end),
            ("oos", train_cutoff, as_of),
        ):
            eq, meta = simulate_monthly_rebalance(
                prices,
                close_a,
                weight_fn=hrp_fn,
                start=lo,
                end=hi,
                listings=listings,
                cost_per_side=float(args.cost_per_side),
                scheme="hrp",
            )
            meta["split"] = split
            hrp_rows.append(meta)
            eq.to_csv(out_dir / f"appendix_hrp_{split}.csv", index=False)
            print(
                f"[OK] hrp {split}: excess={meta['excess']} "
                f"errors={meta.get('hrp_n_errors')}",
                flush=True,
            )
        pd.DataFrame(hrp_rows).to_csv(out_dir / "appendix_hrp.csv", index=False)
    else:
        pd.DataFrame(columns=["scheme", "split"]).to_csv(
            out_dir / "appendix_hrp.csv", index=False
        )

    def _ex(split: str) -> float:
        hit = ew_df[ew_df["split"] == split]
        if hit.empty:
            return float("nan")
        return float(hit.iloc[0]["excess"])

    train_ex = _ex("train")
    oos_ex = _ex("oos")
    gate_train = bool(np.isfinite(train_ex) and train_ex > 0.0)
    gate_oos = bool(np.isfinite(oos_ex) and oos_ex > 0.0)
    pass_all = bool(gate_train and gate_oos)

    l1_train = l1[l1["split"] == "train"].iloc[0].to_dict() if not l1.empty else {}
    l1_oos = l1[l1["split"] == "oos"].iloc[0].to_dict() if len(l1) > 1 else {}

    payload = {
        "layer1_anchor": {
            "train_bh": l1_train.get("bh"),
            "train_edge_vs_need": l1_train.get("edge_vs_need"),
            "oos_bh": l1_oos.get("bh"),
            "oos_edge_vs_need": l1_oos.get("edge_vs_need"),
        },
        "layer2_equal_weight": {
            "train_excess": train_ex if np.isfinite(train_ex) else None,
            "oos_excess": oos_ex if np.isfinite(oos_ex) else None,
            "train_mean_fwd63_excess": float(
                ew_df.loc[ew_df["split"] == "train", "mean_fwd63_excess"].iloc[0]
            )
            if (ew_df["split"] == "train").any()
            else None,
            "oos_mean_fwd63_excess": float(
                ew_df.loc[ew_df["split"] == "oos", "mean_fwd63_excess"].iloc[0]
            )
            if (ew_df["split"] == "oos").any()
            else None,
        },
        "appendix": {
            "cluster_ew": [
                {
                    "split": r["split"],
                    "excess": r["excess"],
                }
                for r in cluster_rows
            ],
            "hrp": [
                {
                    "split": r["split"],
                    "excess": r.get("excess"),
                    "hrp_n_errors": r.get("hrp_n_errors"),
                }
                for r in hrp_rows
            ],
        },
        "gates": {
            "train_excess_positive": gate_train,
            "oos_excess_positive": gate_oos,
            "pass_all": pass_all,
        },
        "recommendation": (
            "OK_discuss_weights"
            if pass_all
            else "REJECT_hold_anchor_do_not_pick_names"
        ),
        "note": (
            "Equal-weight is the only veto. Cluster-EW/HRP are appendix. "
            "Do not wire chart name-picking unless pass_all."
        ),
    }
    (out_dir / "verdict.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[GATE] train={gate_train} oos={gate_oos} → pass_all={pass_all}",
        flush=True,
    )
    print(f"[REC] {payload['recommendation']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
