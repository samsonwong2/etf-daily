#!/usr/bin/env python3
"""Calibrate hold_worthy by train ranking spread (not binary accuracy).

Objective (train only): maximize mean(fwd_63 | hold_ok) − mean(fwd_63 | swing)
subject to month-end flip_rate ≤ 0.20 and hold_frac ∈ [0.15, 0.75].

OOS (2026-04-01 → as-of) is reported only — not used for selection.
Does not change production adaptive hold_up or from_listing HTML.
"""
from __future__ import annotations

import argparse
import itertools
import json
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

from decision_pack.src.hold_vs_swing_bench import (  # noqa: E402
    ROLL_BH_EXIT,
    ROLL_BH_MIN,
    ROLL_UP_FRAC_EXIT,
    ROLL_UP_FRAC_MIN,
    ROLL_WINDOW,
    decision_dates,
    label_at,
    month_end_forward_panel,
    rolling_hold_worthy,
    spread_metrics,
    stability_stats,
)
from decision_pack.src.hrp_cluster_router import normalize_code  # noqa: E402
from decision_pack.src.ma20_truth_triggers import ma20_hug_truth  # noqa: E402

DEFAULT_MEMBERSHIP = Path(
    "~/etf-daily-output/temp/decision_packs/20260814/cluster_representatives_20260814.csv"
)
DEFAULT_LISTING_CSV = (
    PLOTLY_OUTPUTS_DIR / "20260814_from_listing/listing_dates.csv"
)
FOCUS_CODE = "SH515060"
HORIZONS = (63, 126)
MAX_FLIP_RATE = 0.20
HOLD_FRAC_LO = 0.15
HOLD_FRAC_HI = 0.75
MIN_GROUP_N = 5
# Baseline default rule (current production defaults in hold_vs_swing_bench).
BASELINE_RULE = {
    "window": ROLL_WINDOW,
    "bh_min": ROLL_BH_MIN,
    "up_frac_min": ROLL_UP_FRAC_MIN,
    "hysteresis": True,
    "bh_exit": ROLL_BH_EXIT,
    "up_frac_exit": ROLL_UP_FRAC_EXIT,
}


def _load_pm():
    import importlib.util

    path = _PROJECT_ROOT / "decision_pack/scripts/plot_regime_transition_example.py"
    spec = importlib.util.spec_from_file_location("plot_regime_pm_cal_hw", path)
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
        default=PLOTLY_OUTPUTS_DIR / "20260814_hold_worthy_calibrate",
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


def rule_grid() -> list[dict[str, Any]]:
    """Small exhaustive grid from the plan."""
    windows = [63, 126, 252]
    bh_mins = [0.0, 0.05, 0.10]
    up_frac_mins = [0.15, 0.20, 0.30]
    rules: list[dict[str, Any]] = []
    # No hysteresis
    for w, bh, uf in itertools.product(windows, bh_mins, up_frac_mins):
        rules.append(
            {
                "window": w,
                "bh_min": bh,
                "up_frac_min": uf,
                "hysteresis": False,
                "bh_exit": ROLL_BH_EXIT,
                "up_frac_exit": ROLL_UP_FRAC_EXIT,
            }
        )
    # Hysteresis on
    for w, bh, uf, be, ue in itertools.product(
        windows, bh_mins, up_frac_mins, [-0.05, 0.0], [0.10, 0.15]
    ):
        rules.append(
            {
                "window": w,
                "bh_min": bh,
                "up_frac_min": uf,
                "hysteresis": True,
                "bh_exit": be,
                "up_frac_exit": ue,
            }
        )
    # Ensure baseline is present exactly once at front for reporting.
    out = [dict(BASELINE_RULE)]
    for r in rules:
        if all(r[k] == BASELINE_RULE[k] for k in BASELINE_RULE):
            continue
        out.append(r)
    return out


def rule_id(rule: dict[str, Any]) -> str:
    h = "hyst" if rule["hysteresis"] else "raw"
    return (
        f"w{int(rule['window'])}_bh{float(rule['bh_min']):.2f}_"
        f"up{float(rule['up_frac_min']):.2f}_{h}_"
        f"ex{float(rule['bh_exit']):.2f}_{float(rule['up_frac_exit']):.2f}"
    )


def eval_rule_on_cache(
    cache: dict[str, dict[str, Any]],
    rule: dict[str, Any],
    *,
    split_start: str,
    split_end: str,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, pd.DataFrame]]:
    """Apply rule to all codes; return metrics + panel + per-code hold_worthy frames."""
    panels: list[pd.DataFrame] = []
    hw_map: dict[str, pd.DataFrame] = {}
    flip_rates: list[float] = []
    hold_fracs: list[float] = []
    lo = pd.Timestamp(split_start).normalize()
    hi = pd.Timestamp(split_end).normalize()
    for code, blob in cache.items():
        hw = rolling_hold_worthy(
            blob["close"],
            blob["truth"],
            window=int(rule["window"]),
            bh_min=float(rule["bh_min"]),
            up_frac_min=float(rule["up_frac_min"]),
            hysteresis=bool(rule["hysteresis"]),
            bh_exit=float(rule["bh_exit"]),
            up_frac_exit=float(rule["up_frac_exit"]),
        )
        hw_map[code] = hw
        st = stability_stats(hw["bucket"], decision_freq="M")
        if np.isfinite(st.get("flip_rate", np.nan)):
            flip_rates.append(float(st["flip_rate"]))
        if np.isfinite(st.get("hold_frac", np.nan)):
            hold_fracs.append(float(st["hold_frac"]))
        panel = month_end_forward_panel(
            blob["close"], hw["bucket"], horizons=HORIZONS, code=code
        )
        if panel.empty:
            continue
        panel["date_ts"] = pd.to_datetime(panel["date"])
        panel = panel[(panel["date_ts"] >= lo) & (panel["date_ts"] <= hi)]
        panels.append(panel.drop(columns=["date_ts"]))
    full = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()
    m63 = spread_metrics(full, horizon=63, min_group_n=MIN_GROUP_N)
    m126 = spread_metrics(full, horizon=126, min_group_n=MIN_GROUP_N)
    mean_flip = float(np.mean(flip_rates)) if flip_rates else float("nan")
    mean_hold = float(np.mean(hold_fracs)) if hold_fracs else float("nan")
    # Pool hold_frac from panel if available
    if not full.empty and "bucket" in full.columns:
        pool_hold = float((full["bucket"].astype(str) == "hold_ok").mean())
    else:
        pool_hold = mean_hold
    pass_stab = (
        np.isfinite(mean_flip)
        and mean_flip <= MAX_FLIP_RATE
        and np.isfinite(pool_hold)
        and HOLD_FRAC_LO <= pool_hold <= HOLD_FRAC_HI
        and bool(m63["pass_min_group"])
    )
    metrics = {
        "rule_id": rule_id(rule),
        **{f"r_{k}": v for k, v in rule.items()},
        "mean_flip_rate": mean_flip,
        "mean_code_hold_frac": mean_hold,
        "pool_hold_frac": pool_hold,
        "pass_stability": bool(pass_stab),
        "spread_63": m63["spread"],
        "spread_126": m126["spread"],
        "month_win_63": m63["month_win_rate"],
        "month_win_126": m126["month_win_rate"],
        "mean_fwd_hold_63": m63["mean_fwd_hold"],
        "mean_fwd_swing_63": m63["mean_fwd_swing"],
        "n_63": m63["n"],
        "n_hold_63": m63["n_hold"],
        "n_swing_63": m63["n_swing"],
        "binary_acc_63": m63["binary_acc"],
        "spread_126_same_sign": bool(
            np.isfinite(m63["spread"])
            and np.isfinite(m126["spread"])
            and (m63["spread"] * m126["spread"] > 0)
        ),
        "score": float(m63["spread"])
        + (0.25 * float(m126["spread"]) if np.isfinite(m126["spread"]) else 0.0)
        if np.isfinite(m63["spread"])
        else float("-inf"),
    }
    return metrics, full, hw_map


def focus_sanity(
    hw: pd.DataFrame,
    *,
    as_of: str,
    year: int = 2026,
) -> dict[str, Any]:
    dec = decision_dates(hw.index, freq="M")
    sub = hw["bucket"].reindex(dec).dropna()
    asof_lab = label_at(hw["bucket"], as_of)
    y = sub[pd.to_datetime(sub.index).year == int(year)]
    swing_frac_y = (
        float((y.astype(str) == "swing_only").mean()) if len(y) else float("nan")
    )
    return {
        "asof_bucket": asof_lab,
        "year_swing_frac": swing_frac_y,
        "year_n_months": int(len(y)),
        "pass_swing_majority": bool(
            asof_lab == "swing_only"
            and np.isfinite(swing_frac_y)
            and swing_frac_y >= 0.5
        ),
    }


def write_readme(path: Path) -> None:
    path.write_text(
        """# hold_worthy recalibration (ranking spread)

## Objective
Train: maximize `mean(fwd_63|hold_ok) − mean(fwd_63|swing)` subject to
month-end flip_rate ≤ 0.20 and hold_frac ∈ [0.15, 0.75].

OOS is report-only. Binary accuracy is appendix only.

## Gate (must all pass to treat as config switch)
1. Train 63d spread better than baseline default rule
2. OOS 63d spread > 0 (same sign)
3. SH515060 as-of swing_only and ≥50% of 2026 months swing

If any fails: do **not** wire into HTML / trading.

## Outputs
- `grid_train.csv` — all rules on train
- `chosen_rule.json` — best feasible + gate flags
- `oos_spread.csv` — baseline vs chosen on OOS
- `timeline_SH515060.csv` — month-end labels under chosen rule
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

    cache: dict[str, dict[str, Any]] = {}
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
            truth = ma20_hug_truth(close)
            cache[code] = {"close": close, "truth": truth, "listing": listing}
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {code}: {exc}", flush=True)
            if args.fail_fast:
                raise
            traceback.print_exc()

    rules = rule_grid()
    print(f"[INFO] grid size={len(rules)}", flush=True)
    train_rows: list[dict[str, Any]] = []
    baseline_train: dict[str, Any] | None = None
    for j, rule in enumerate(rules, 1):
        if j == 1 or j % 20 == 0 or j == len(rules):
            print(f"  [train-grid] {j}/{len(rules)} {rule_id(rule)}", flush=True)
        m, _panel, _hw = eval_rule_on_cache(
            cache, rule, split_start="2005-01-01", split_end=train_end
        )
        m["is_baseline"] = all(rule[k] == BASELINE_RULE[k] for k in BASELINE_RULE)
        train_rows.append(m)
        if m["is_baseline"]:
            baseline_train = m

    train_df = pd.DataFrame(train_rows)
    train_df.to_csv(out_dir / "grid_train.csv", index=False)

    feasible = train_df[train_df["pass_stability"] == True].copy()  # noqa: E712
    if feasible.empty:
        print("[WARN] no rule passed stability constraints", flush=True)
        chosen_row = train_df.sort_values("score", ascending=False).iloc[0]
        feasible_ok = False
    else:
        # Prefer same-sign 126, then score
        feasible = feasible.sort_values(
            ["spread_126_same_sign", "score"], ascending=[False, False]
        )
        chosen_row = feasible.iloc[0]
        feasible_ok = True

    chosen_rule = {
        "window": int(chosen_row["r_window"]),
        "bh_min": float(chosen_row["r_bh_min"]),
        "up_frac_min": float(chosen_row["r_up_frac_min"]),
        "hysteresis": bool(chosen_row["r_hysteresis"]),
        "bh_exit": float(chosen_row["r_bh_exit"]),
        "up_frac_exit": float(chosen_row["r_up_frac_exit"]),
    }
    print(
        f"[OK] chosen={rule_id(chosen_rule)} "
        f"train_spread63={chosen_row['spread_63']:.4f} "
        f"feasible={feasible_ok}",
        flush=True,
    )

    # OOS for baseline + chosen
    oos_rows: list[dict[str, Any]] = []
    for label, rule in (("baseline", BASELINE_RULE), ("chosen", chosen_rule)):
        m, _p, hw_map = eval_rule_on_cache(
            cache, rule, split_start=train_cutoff, split_end=as_of
        )
        m["which"] = label
        oos_rows.append(m)
        if label == "chosen" and FOCUS_CODE in hw_map:
            hw = hw_map[FOCUS_CODE]
            dec = decision_dates(hw.index, freq="M")
            tl = hw.reindex(dec).dropna(subset=["bucket"]).reset_index()
            tl = tl.rename(columns={"index": "date"})
            tl["date"] = pd.to_datetime(tl["date"]).dt.strftime("%Y-%m-%d")
            tl["code"] = FOCUS_CODE
            tl.to_csv(out_dir / f"timeline_{FOCUS_CODE}.csv", index=False)
            sanity = focus_sanity(hw, as_of=as_of)
        else:
            sanity = {}

    oos_df = pd.DataFrame(oos_rows)
    oos_df.to_csv(out_dir / "oos_spread.csv", index=False)

    base_spread_train = (
        float(baseline_train["spread_63"])
        if baseline_train is not None and np.isfinite(baseline_train["spread_63"])
        else float("nan")
    )
    chosen_spread_train = float(chosen_row["spread_63"])
    oos_chosen = oos_df[oos_df["which"] == "chosen"].iloc[0]
    oos_base = oos_df[oos_df["which"] == "baseline"].iloc[0]
    oos_spread = float(oos_chosen["spread_63"])

    if FOCUS_CODE not in cache:
        sanity = {
            "asof_bucket": "",
            "year_swing_frac": float("nan"),
            "pass_swing_majority": False,
        }
    elif "pass_swing_majority" not in sanity:
        # recompute if chosen path skipped
        _m, _p, hw_map = eval_rule_on_cache(
            cache, chosen_rule, split_start=train_cutoff, split_end=as_of
        )
        sanity = focus_sanity(hw_map[FOCUS_CODE], as_of=as_of)
        dec = decision_dates(hw_map[FOCUS_CODE].index, freq="M")
        tl = hw_map[FOCUS_CODE].reindex(dec).dropna(subset=["bucket"]).reset_index()
        tl = tl.rename(columns={"index": "date"})
        tl["date"] = pd.to_datetime(tl["date"]).dt.strftime("%Y-%m-%d")
        tl["code"] = FOCUS_CODE
        tl.to_csv(out_dir / f"timeline_{FOCUS_CODE}.csv", index=False)

    gate_train_better = bool(
        np.isfinite(chosen_spread_train)
        and np.isfinite(base_spread_train)
        and chosen_spread_train > base_spread_train
    )
    gate_oos_pos = bool(np.isfinite(oos_spread) and oos_spread > 0.0)
    gate_515 = bool(sanity.get("pass_swing_majority", False))
    pass_all = bool(feasible_ok and gate_train_better and gate_oos_pos and gate_515)

    chosen_payload = {
        "rule": chosen_rule,
        "rule_id": rule_id(chosen_rule),
        "feasible_stability": feasible_ok,
        "train": {
            "spread_63": chosen_spread_train,
            "spread_126": float(chosen_row["spread_126"])
            if np.isfinite(chosen_row["spread_126"])
            else None,
            "baseline_spread_63": base_spread_train
            if np.isfinite(base_spread_train)
            else None,
            "month_win_63": float(chosen_row["month_win_63"])
            if np.isfinite(chosen_row["month_win_63"])
            else None,
            "mean_flip_rate": float(chosen_row["mean_flip_rate"])
            if np.isfinite(chosen_row["mean_flip_rate"])
            else None,
            "pool_hold_frac": float(chosen_row["pool_hold_frac"])
            if np.isfinite(chosen_row["pool_hold_frac"])
            else None,
        },
        "oos": {
            "spread_63": oos_spread if np.isfinite(oos_spread) else None,
            "baseline_spread_63": float(oos_base["spread_63"])
            if np.isfinite(oos_base["spread_63"])
            else None,
            "spread_126": float(oos_chosen["spread_126"])
            if np.isfinite(oos_chosen["spread_126"])
            else None,
        },
        "focus_SH515060": sanity,
        "gates": {
            "train_better_than_baseline": gate_train_better,
            "oos_spread_63_positive": gate_oos_pos,
            "SH515060_swing_sanity": gate_515,
            "pass_all": pass_all,
        },
        "recommendation": (
            "OK_as_config_layer"
            if pass_all
            else "REJECT_do_not_wire_HTML_or_trading"
        ),
        "note": (
            "Do not change production hold_up or from_listing HTML unless pass_all."
        ),
    }
    (out_dir / "chosen_rule.json").write_text(
        json.dumps(chosen_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"[GATE] train_better={gate_train_better} "
        f"oos_pos={gate_oos_pos} SH515060={gate_515} → pass_all={pass_all}",
        flush=True,
    )
    print(f"[REC] {chosen_payload['recommendation']}", flush=True)
    print(
        f"[BASELINE train spread63]={base_spread_train:.4f} "
        f"oos={float(oos_base['spread_63']):.4f}",
        flush=True,
    )
    print(
        f"[CHOSEN  train spread63]={chosen_spread_train:.4f} "
        f"oos={oos_spread:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
