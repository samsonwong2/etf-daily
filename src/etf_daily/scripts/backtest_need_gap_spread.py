#!/usr/bin/env python3
"""Test whether HTML row-4/5 fair-need gap can be an investability signal.

Hypothesis (user-chosen): gap > 0 at month-end ⇒ investable, else skip.
Control: high r_need_ann vs pool median (risk pricing, not alpha).

Metrics: train/OOS ranking spread on fwd BH and excess vs SH510300.
Does not change production adaptive rules or from_listing HTML.
"""
from __future__ import annotations

import argparse
import json
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

from etf_daily.lib.hold_vs_swing_bench import (  # noqa: E402
    decision_dates,
    label_at,
    spread_metrics,
)
from etf_daily.hrp.hrp_cluster_router import normalize_code  # noqa: E402
from etf_daily.lib.regime_transition_plot import (  # noqa: E402
    compute_volatility_regime_frame,
)
from etf_daily.lib.vol_need_return_board import (  # noqa: E402
    ANCHOR_ERP_ANN,
    EQUITY_ANCHOR,
    HORIZON_5,
    HORIZON_20,
    fair_need_ann_series,
    gap_realized_minus_need,
)

DEFAULT_MEMBERSHIP = Path(
    "~/etf-daily-output/temp/decision_packs/20260814/cluster_representatives_20260814.csv"
)
DEFAULT_LISTING_CSV = (
    PLOTLY_OUTPUTS_DIR / "20260814_from_listing/listing_dates.csv"
)
HORIZONS = (63, 126)
MIN_GROUP_N = 5
GAP_THRESHOLD = 0.0  # plan: only threshold {0}, no OOS-fitting


def _load_pm():
    import importlib.util

    path = _PROJECT_ROOT / "src/etf_daily/plots/plot_regime_transition_example.py"
    spec = importlib.util.spec_from_file_location("plot_regime_pm_gap", path)
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
        default=PLOTLY_OUTPUTS_DIR / "20260814_need_gap_spread",
    )
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--code", action="append", default=None)
    p.add_argument("--fail-fast", action="store_true")
    return p.parse_args(argv)


def load_membership(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df.copy()
    out["code"] = out["code"].map(normalize_code)
    return out


def load_listings(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {
        normalize_code(r.code): pd.Timestamp(r.listing).strftime("%Y-%m-%d")
        for r in df.itertuples(index=False)
    }


def build_need_gap_frame(
    close: pd.Series,
    *,
    rv20_anchor: pd.Series,
    rv5_anchor: pd.Series,
    erp_ann: float = ANCHOR_ERP_ANN,
) -> pd.DataFrame:
    """Causal daily need + gap series aligned to close index."""
    s = pd.to_numeric(close, errors="coerce").copy()
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    vol = compute_volatility_regime_frame(s)
    vol = vol.copy()
    vol["as_of"] = pd.to_datetime(vol["as_of"]).dt.normalize()
    vol = vol.set_index("as_of").reindex(s.index)
    rv20 = pd.to_numeric(vol["rv20"], errors="coerce")
    rv5 = pd.to_numeric(vol["rv5"], errors="coerce")
    a20 = pd.to_numeric(rv20_anchor, errors="coerce").copy()
    a20.index = pd.DatetimeIndex(pd.to_datetime(a20.index)).normalize()
    a20 = a20[~a20.index.duplicated(keep="last")].sort_index()
    a5 = pd.to_numeric(rv5_anchor, errors="coerce").copy()
    a5.index = pd.DatetimeIndex(pd.to_datetime(a5.index)).normalize()
    a5 = a5[~a5.index.duplicated(keep="last")].sort_index()
    need20 = fair_need_ann_series(rv20, a20, erp_ann=erp_ann).reindex(s.index)
    need5 = fair_need_ann_series(rv5, a5, erp_ann=erp_ann).reindex(s.index)
    g20 = gap_realized_minus_need(s, need20, bars=HORIZON_20)
    g5 = gap_realized_minus_need(s, need5, bars=HORIZON_5)
    out = pd.DataFrame(
        {
            "r_need_ann": need20,
            "r_need_ann_5": need5,
            "gap_20d": pd.to_numeric(g20["gap"], errors="coerce").reindex(s.index),
            "gap_5d": pd.to_numeric(g5["gap"], errors="coerce").reindex(s.index),
        },
        index=s.index,
    )
    return out


def month_end_gap_panel(
    close: pd.Series,
    need_gap: pd.DataFrame,
    anchor_close: pd.Series,
    *,
    code: str,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """Month-end features + forward BH / excess vs anchor (labels use future)."""
    px = pd.to_numeric(close, errors="coerce").copy()
    px.index = pd.DatetimeIndex(pd.to_datetime(px.index)).normalize()
    px = px[~px.index.duplicated(keep="last")].sort_index()
    ax = pd.to_numeric(anchor_close, errors="coerce").copy()
    ax.index = pd.DatetimeIndex(pd.to_datetime(ax.index)).normalize()
    ax = ax[~ax.index.duplicated(keep="last")].sort_index()
    ng = need_gap.reindex(px.index)
    dec = decision_dates(px.index, freq="M")
    rows: list[dict[str, Any]] = []
    for d in dec:
        if d not in px.index:
            continue
        pos = px.index.get_loc(d)
        if isinstance(pos, slice):
            continue
        p0 = float(px.iloc[pos])
        if not (np.isfinite(p0) and p0 > 0):
            continue
        ax_aligned = ax.reindex(px.index).ffill()
        a0 = float(ax_aligned.iloc[pos]) if pos < len(ax_aligned) else float("nan")
        feat = ng.iloc[pos] if pos < len(ng) else None
        row: dict[str, Any] = {
            "code": code,
            "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
            "gap_20d": float(feat["gap_20d"]) if feat is not None else float("nan"),
            "gap_5d": float(feat["gap_5d"]) if feat is not None else float("nan"),
            "r_need_ann": float(feat["r_need_ann"]) if feat is not None else float("nan"),
            "r_need_ann_5": float(feat["r_need_ann_5"])
            if feat is not None
            else float("nan"),
        }
        for h in horizons:
            hi = int(h)
            if pos + hi >= len(px):
                row[f"fwd_{hi}"] = float("nan")
                row[f"excess_{hi}"] = float("nan")
                continue
            p1 = float(px.iloc[pos + hi])
            fwd = float(p1 / p0 - 1.0) if np.isfinite(p1) and p1 > 0 else float("nan")
            row[f"fwd_{hi}"] = fwd
            if np.isfinite(a0) and a0 > 0 and pos + hi < len(ax_aligned):
                a1 = float(ax_aligned.iloc[pos + hi])
                if np.isfinite(a1) and a1 > 0 and np.isfinite(fwd):
                    row[f"excess_{hi}"] = float(fwd - (a1 / a0 - 1.0))
                else:
                    row[f"excess_{hi}"] = float("nan")
            else:
                row[f"excess_{hi}"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def assign_gap_bucket(panel: pd.DataFrame, *, bars: int, thr: float) -> pd.DataFrame:
    out = panel.copy()
    col = f"gap_{int(bars)}d"
    g = pd.to_numeric(out[col], errors="coerce")
    out["bucket"] = np.where(g > float(thr), "investable", "skip")
    out.loc[~np.isfinite(g), "bucket"] = "skip"
    return out


def assign_need_bucket(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional: above same-date median r_need_ann → high_need."""
    out = panel.copy()
    out["bucket"] = "low_need"
    for date, g in out.groupby("date"):
        need = pd.to_numeric(g["r_need_ann"], errors="coerce")
        med = float(need.median()) if need.notna().any() else float("nan")
        if not np.isfinite(med):
            continue
        mask = need > med
        out.loc[g.index[mask.fillna(False)], "bucket"] = "high_need"
        out.loc[g.index[(~mask).fillna(True)], "bucket"] = "low_need"
    return out


def filter_split(panel: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if panel.empty:
        return panel
    ts = pd.to_datetime(panel["date"])
    lo = pd.Timestamp(start).normalize()
    hi = pd.Timestamp(end).normalize()
    return panel.loc[(ts >= lo) & (ts <= hi)].copy()


def summarize_rule(
    panel: pd.DataFrame,
    *,
    pos_label: str,
    neg_label: str,
    rule_name: str,
    split: str,
) -> dict[str, Any]:
    m63 = spread_metrics(
        panel,
        horizon=63,
        min_group_n=MIN_GROUP_N,
        pos_label=pos_label,
        neg_label=neg_label,
        fwd_col="fwd_63",
    )
    m126 = spread_metrics(
        panel,
        horizon=126,
        min_group_n=MIN_GROUP_N,
        pos_label=pos_label,
        neg_label=neg_label,
        fwd_col="fwd_126",
    )
    e63 = spread_metrics(
        panel,
        horizon=63,
        min_group_n=MIN_GROUP_N,
        pos_label=pos_label,
        neg_label=neg_label,
        fwd_col="excess_63",
    )
    e126 = spread_metrics(
        panel,
        horizon=126,
        min_group_n=MIN_GROUP_N,
        pos_label=pos_label,
        neg_label=neg_label,
        fwd_col="excess_126",
    )
    return {
        "rule": rule_name,
        "split": split,
        "pos_label": pos_label,
        "neg_label": neg_label,
        "spread_fwd_63": m63["spread"],
        "spread_fwd_126": m126["spread"],
        "spread_excess_63": e63["spread"],
        "spread_excess_126": e126["spread"],
        "month_win_fwd_63": m63["month_win_rate"],
        "month_win_excess_63": e63["month_win_rate"],
        "pos_frac": m63["pos_frac"],
        "n_63": m63["n"],
        "n_pos_63": m63["n_pos"],
        "n_neg_63": m63["n_neg"],
        "mean_fwd_pos_63": m63["mean_fwd_pos"],
        "mean_fwd_neg_63": m63["mean_fwd_neg"],
        "mean_excess_pos_63": e63["mean_fwd_pos"],
        "mean_excess_neg_63": e63["mean_fwd_neg"],
        "pass_min_group_63": m63["pass_min_group"],
        "binary_acc_fwd_63": m63["binary_acc"],
    }


def write_readme(path: Path) -> None:
    path.write_text(
        """# Need / gap investability spread test

## Question
Can HTML row-4/5 fair-need gap be used as an investability switch?

## Rules
1. **gap_20d > 0** (and gap_5d > 0) → investable vs skip
2. **Control**: r_need_ann above same-month pool median → high_need vs low_need

## Metrics
- `spread_fwd_63` = mean(BH_63|pos) − mean(BH_63|neg)
- `spread_excess_63` = same on excess vs SH510300

Train: listing → 2026-03-31. OOS: 2026-04-01 → as-of (veto only).

## Gate
Train spread_fwd_63 > 0 and OOS spread_fwd_63 > 0 for the primary gap_20 rule.
If fail: do **not** wire into HTML / trading.
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

    membership = load_membership(args.membership)
    listings = load_listings(args.listing_csv)
    codes = sorted(membership["code"].unique())
    if args.code:
        want = {normalize_code(c) for c in args.code}
        codes = [c for c in codes if c in want]
    if args.max_codes is not None:
        codes = codes[: int(args.max_codes)]

    print(f"[INFO] codes={len(codes)} train_end={train_end} oos={train_cutoff}→{as_of}")
    print(f"[INFO] out={out_dir}", flush=True)

    pm = _load_pm()
    pm._init_qlib(None)

    print(f"[INFO] loading anchor {EQUITY_ANCHOR}", flush=True)
    ohlcv_a = pm.load_qlib_ohlcv(
        EQUITY_ANCHOR, "2010-01-01", as_of, init_qlib=False, clip_to_window=False
    )
    close_a = ohlcv_a.set_index(pd.to_datetime(ohlcv_a["datetime"]).dt.normalize())[
        "$close"
    ]
    close_a = close_a[~close_a.index.duplicated(keep="last")].sort_index()
    vf_a = compute_volatility_regime_frame(close_a)
    vf_a["as_of"] = pd.to_datetime(vf_a["as_of"]).dt.normalize()
    vf_a = vf_a.set_index("as_of")
    rv20_anchor = vf_a["rv20"]
    rv5_anchor = vf_a["rv5"]

    panels: list[pd.DataFrame] = []
    for i, code in enumerate(codes, 1):
        listing = listings.get(code, "2015-01-01")
        try:
            print(f"[{i}/{len(codes)}] {code}", flush=True)
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
            ng = build_need_gap_frame(
                close, rv20_anchor=rv20_anchor, rv5_anchor=rv5_anchor
            )
            panel = month_end_gap_panel(
                close, ng, close_a, code=code, horizons=HORIZONS
            )
            if not panel.empty:
                panels.append(panel)
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {code}: {exc}", flush=True)
            if args.fail_fast:
                raise
            traceback.print_exc()

    full = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()
    full.to_csv(out_dir / "month_end_panel.csv", index=False)
    print(f"[OK] panel rows={len(full)}", flush=True)

    rules = [
        ("gap_20d_gt0", lambda p: assign_gap_bucket(p, bars=20, thr=GAP_THRESHOLD), "investable", "skip"),
        ("gap_5d_gt0", lambda p: assign_gap_bucket(p, bars=5, thr=GAP_THRESHOLD), "investable", "skip"),
        ("high_need_vs_median", assign_need_bucket, "high_need", "low_need"),
    ]

    rows: list[dict[str, Any]] = []
    for rule_name, assign_fn, pos, neg in rules:
        labeled = assign_fn(full)
        for split, lo, hi in (
            ("train", "2005-01-01", train_end),
            ("oos", train_cutoff, as_of),
        ):
            sub = filter_split(labeled, lo, hi)
            m = summarize_rule(
                sub, pos_label=pos, neg_label=neg, rule_name=rule_name, split=split
            )
            rows.append(m)
            print(
                f"[OK] {rule_name} {split}: "
                f"spread_fwd63={m['spread_fwd_63']} "
                f"spread_xs63={m['spread_excess_63']} "
                f"pos_frac={m['pos_frac']}",
                flush=True,
            )

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "spread_by_rule.csv", index=False)

    # Primary gate: gap_20d
    def _get(rule: str, split: str, key: str) -> float:
        hit = res[(res["rule"] == rule) & (res["split"] == split)]
        if hit.empty:
            return float("nan")
        return float(hit.iloc[0][key])

    train_s = _get("gap_20d_gt0", "train", "spread_fwd_63")
    oos_s = _get("gap_20d_gt0", "oos", "spread_fwd_63")
    train_xs = _get("gap_20d_gt0", "train", "spread_excess_63")
    oos_xs = _get("gap_20d_gt0", "oos", "spread_excess_63")
    train_need = _get("high_need_vs_median", "train", "spread_fwd_63")
    oos_need = _get("high_need_vs_median", "oos", "spread_fwd_63")

    gate_train = bool(np.isfinite(train_s) and train_s > 0.0)
    gate_oos = bool(np.isfinite(oos_s) and oos_s > 0.0)
    pass_all = bool(gate_train and gate_oos)

    payload = {
        "primary_rule": "gap_20d > 0 → investable",
        "threshold": GAP_THRESHOLD,
        "gap_20d": {
            "train_spread_fwd_63": train_s if np.isfinite(train_s) else None,
            "oos_spread_fwd_63": oos_s if np.isfinite(oos_s) else None,
            "train_spread_excess_63": train_xs if np.isfinite(train_xs) else None,
            "oos_spread_excess_63": oos_xs if np.isfinite(oos_xs) else None,
        },
        "control_high_need": {
            "train_spread_fwd_63": train_need if np.isfinite(train_need) else None,
            "oos_spread_fwd_63": oos_need if np.isfinite(oos_need) else None,
            "note": "If high_need does not beat low_need on excess, risk pricing ≠ alpha",
        },
        "gates": {
            "train_spread_fwd_63_positive": gate_train,
            "oos_spread_fwd_63_positive": gate_oos,
            "pass_all": pass_all,
        },
        "recommendation": (
            "OK_as_config_layer_candidate"
            if pass_all
            else "REJECT_do_not_wire_HTML_or_trading"
        ),
        "note": (
            "Fair need is risk compensation vs SH510300, not alpha. "
            "Gap>0 is trailing outperformance of that hurdle; "
            "do not treat high need alone as investable."
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
