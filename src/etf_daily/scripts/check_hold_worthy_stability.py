#!/usr/bin/env python3
"""Point-in-time hold_worthy(t) stability check on the 46-code pool.

Rule (causal, trailing window only):
- Features at t: BH and MA20-up fraction over the past ``--window`` bars
- Enter hold_ok: BH > bh_min and up_frac ≥ up_frac_min
- With hysteresis (default): leave hold_ok only if BH ≤ bh_exit or up_frac < up_frac_exit
- Decision frequency for flip stats: month-end (default); daily also reported

Does not change production adaptive trading defaults.
"""
from __future__ import annotations

import argparse
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
    ROLL_BH_EXIT,
    ROLL_BH_MIN,
    ROLL_UP_FRAC_EXIT,
    ROLL_UP_FRAC_MIN,
    ROLL_WINDOW,
    classify_hold_bucket,
    decision_dates,
    rolling_hold_worthy,
    stability_stats,
    trend_quality_metrics,
)
from etf_daily.hrp.hrp_cluster_router import normalize_code  # noqa: E402
from etf_daily.lib.ma20_truth_triggers import ma20_hug_truth  # noqa: E402

DEFAULT_MEMBERSHIP = Path(
    "~/etf-daily-output/temp/decision_packs/20260814/cluster_representatives_20260814.csv"
)
DEFAULT_LISTING_CSV = (
    PLOTLY_OUTPUTS_DIR / "20260814_from_listing/listing_dates.csv"
)
FOCUS_CODE = "SH515060"


def _load_pm():
    import importlib.util

    path = _PROJECT_ROOT / "src/etf_daily/plots/plot_regime_transition_example.py"
    spec = importlib.util.spec_from_file_location("plot_regime_pm_holdw", path)
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
        default=PLOTLY_OUTPUTS_DIR / "20260814_hold_worthy_stability",
    )
    p.add_argument("--window", type=int, default=ROLL_WINDOW)
    p.add_argument("--windows", default="63,126,252", help="sensitivity windows")
    p.add_argument("--bh-min", type=float, default=ROLL_BH_MIN)
    p.add_argument("--up-frac-min", type=float, default=ROLL_UP_FRAC_MIN)
    p.add_argument("--bh-exit", type=float, default=ROLL_BH_EXIT)
    p.add_argument("--up-frac-exit", type=float, default=ROLL_UP_FRAC_EXIT)
    p.add_argument("--no-hysteresis", action="store_true")
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


def write_readme(
    path: Path,
    *,
    window: int,
    bh_min: float,
    up_frac_min: float,
    bh_exit: float,
    up_frac_exit: float,
    hysteresis: bool,
    as_of: str,
) -> None:
    text = f"""# Point-in-time hold_worthy stability

## Rule (causal)
At each bar t, trailing `{window}` trading days:
- BH = close[t] / close[t-window+1] − 1
- up_frac = fraction of MA20±2% truth == up

Enter **hold_ok** when BH > {bh_min} and up_frac ≥ {up_frac_min}.
Hysteresis={'ON' if hysteresis else 'OFF'}: leave hold_ok only if
BH ≤ {bh_exit} or up_frac < {up_frac_exit}.

Decision frequency for flip stats: **month-end** (also daily in summary).

## Outputs
- `summary_by_code.csv` — flip rates, hold_frac, agreement with static train bucket
- `sensitivity_windows.csv` — pool flip stats across lookbacks
- `timeline_SH515060.csv` — month-end labels for focus code
- `as_of_labels.csv` — label at `{as_of}` under primary window
"""
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    as_of = pd.Timestamp(args.as_of).strftime("%Y-%m-%d")
    train_cutoff = pd.Timestamp(args.train_cutoff).strftime("%Y-%m-%d")
    train_end = (pd.Timestamp(train_cutoff) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    out_dir = Path(args.html_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hysteresis = not bool(args.no_hysteresis)
    primary_w = int(args.window)
    sens_windows = [int(x) for x in str(args.windows).split(",") if x.strip()]

    membership = load_membership(args.membership)
    listings = load_listings(args.listing_csv)
    codes = sorted(membership["code"].unique())
    if args.code:
        want = {normalize_code(c) for c in args.code}
        codes = [c for c in codes if c in want]
    if args.max_codes is not None:
        codes = codes[: int(args.max_codes)]

    print(
        f"[INFO] codes={len(codes)} window={primary_w} hysteresis={hysteresis}",
        flush=True,
    )
    print(f"[INFO] out={out_dir}", flush=True)
    write_readme(
        out_dir / "README.md",
        window=primary_w,
        bh_min=float(args.bh_min),
        up_frac_min=float(args.up_frac_min),
        bh_exit=float(args.bh_exit),
        up_frac_exit=float(args.up_frac_exit),
        hysteresis=hysteresis,
        as_of=as_of,
    )

    pm = _load_pm()
    pm._init_qlib(None)

    rows: list[dict[str, Any]] = []
    sens_rows: list[dict[str, Any]] = []
    focus_timeline: pd.DataFrame | None = None
    asof_rows: list[dict[str, Any]] = []

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

            # Static train bucket (evaluation reference, not causal at t).
            static_m = trend_quality_metrics(
                close, truth, start=listing, end=train_end
            )
            static_bucket = classify_hold_bucket(
                static_m,
                bh_min=float(args.bh_min),
                up_frac_min=float(args.up_frac_min),
            )

            hw = rolling_hold_worthy(
                close,
                truth,
                window=primary_w,
                bh_min=float(args.bh_min),
                up_frac_min=float(args.up_frac_min),
                hysteresis=hysteresis,
                bh_exit=float(args.bh_exit),
                up_frac_exit=float(args.up_frac_exit),
            )
            # Restrict stats to post-listing bars with enough history.
            st_m = stability_stats(hw["bucket"], decision_freq="M")
            st_d = stability_stats(hw["bucket"], decision_freq="D")

            # Agreement: month-end labels vs static train bucket on train months only
            dec = decision_dates(hw.index, freq="M")
            dec_train = dec[dec <= pd.Timestamp(train_end)]
            sub = hw["bucket"].reindex(dec_train).dropna()
            if len(sub):
                agree = float(np.mean(sub.astype(str) == static_bucket))
                last_train = str(sub.iloc[-1])
            else:
                agree = float("nan")
                last_train = ""

            asof_lab = (
                str(hw["bucket"].iloc[-1]) if len(hw) else "swing_only"
            )
            asof_feat = hw.iloc[-1] if len(hw) else None
            rows.append(
                {
                    "code": code,
                    "listing": listing,
                    "n_bars": int(len(close)),
                    "static_train_bucket": static_bucket,
                    "static_train_bh": static_m.get("bh"),
                    "static_train_up_frac": static_m.get("up_frac"),
                    "asof_bucket": asof_lab,
                    "asof_bh": float(asof_feat["bh"]) if asof_feat is not None else np.nan,
                    "asof_up_frac": float(asof_feat["up_frac"])
                    if asof_feat is not None
                    else np.nan,
                    "last_train_month_bucket": last_train,
                    "agree_static_train_months": agree,
                    "window": primary_w,
                    "hysteresis": hysteresis,
                    **{f"m_{k}": v for k, v in st_m.items()},
                    **{f"d_{k}": v for k, v in st_d.items()},
                }
            )
            asof_rows.append(
                {
                    "code": code,
                    "bucket": asof_lab,
                    "bh": float(asof_feat["bh"]) if asof_feat is not None else np.nan,
                    "up_frac": float(asof_feat["up_frac"])
                    if asof_feat is not None
                    else np.nan,
                    "static_train_bucket": static_bucket,
                }
            )

            if code == FOCUS_CODE:
                dec_f = decision_dates(hw.index, freq="M")
                focus_timeline = hw.reindex(dec_f).dropna(subset=["bucket"]).reset_index()
                focus_timeline = focus_timeline.rename(columns={"index": "date"})
                focus_timeline["date"] = pd.to_datetime(focus_timeline["date"]).dt.strftime(
                    "%Y-%m-%d"
                )
                focus_timeline["code"] = code

            for w in sens_windows:
                hw_w = rolling_hold_worthy(
                    close,
                    truth,
                    window=w,
                    bh_min=float(args.bh_min),
                    up_frac_min=float(args.up_frac_min),
                    hysteresis=hysteresis,
                    bh_exit=float(args.bh_exit),
                    up_frac_exit=float(args.up_frac_exit),
                )
                st = stability_stats(hw_w["bucket"], decision_freq="M")
                sens_rows.append(
                    {
                        "code": code,
                        "window": w,
                        "hold_frac": st["hold_frac"],
                        "n_flips": st["n_flips"],
                        "flip_rate": st["flip_rate"],
                        "mean_spell_decisions": st["mean_spell_decisions"],
                        "asof_bucket": str(hw_w["bucket"].iloc[-1])
                        if len(hw_w)
                        else "",
                    }
                )
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {code}: {exc}", flush=True)
            if args.fail_fast:
                raise
            traceback.print_exc()

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary_by_code.csv", index=False)
    pd.DataFrame(asof_rows).to_csv(out_dir / "as_of_labels.csv", index=False)
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(out_dir / "sensitivity_by_code.csv", index=False)
    if not sens.empty:
        pool = (
            sens.groupby("window", as_index=False)
            .agg(
                n_codes=("code", "count"),
                mean_hold_frac=("hold_frac", "mean"),
                mean_flip_rate=("flip_rate", "mean"),
                median_flip_rate=("flip_rate", "median"),
                mean_spell=("mean_spell_decisions", "mean"),
                n_asof_hold=("asof_bucket", lambda s: int((s == "hold_ok").sum())),
            )
            .sort_values("window")
        )
        pool.to_csv(out_dir / "sensitivity_windows.csv", index=False)
        print("[OK] sensitivity_windows:", flush=True)
        print(pool.to_string(index=False), flush=True)

    if focus_timeline is not None:
        focus_timeline.to_csv(out_dir / f"timeline_{FOCUS_CODE}.csv", index=False)
        print(
            f"[OK] {FOCUS_CODE} month-end flips="
            f"{int((focus_timeline['bucket'].shift() != focus_timeline['bucket']).sum())} "
            f"hold_frac={float((focus_timeline['bucket']=='hold_ok').mean()):.2%}",
            flush=True,
        )

    if not summary.empty:
        print(
            "[OK] pool month-end: "
            f"mean_hold_frac={summary['m_hold_frac'].mean():.2%} "
            f"mean_flip_rate={summary['m_flip_rate'].mean():.2%} "
            f"asof_hold={(summary['asof_bucket']=='hold_ok').sum()}/"
            f"{len(summary)}",
            flush=True,
        )
        print(
            "[OK] agree static train months: "
            f"mean={summary['agree_static_train_months'].mean():.2%}",
            flush=True,
        )
        if FOCUS_CODE in set(summary["code"]):
            r = summary[summary["code"] == FOCUS_CODE].iloc[0]
            print(
                f"[FOCUS] {FOCUS_CODE} static={r['static_train_bucket']} "
                f"asof={r['asof_bucket']} m_hold_frac={r['m_hold_frac']:.2%} "
                f"m_flips={r['m_n_flips']} m_flip_rate={r['m_flip_rate']:.2%}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
