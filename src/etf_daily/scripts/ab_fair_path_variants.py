#!/usr/bin/env python3
"""A/B for the pre-registered fig7–11 challenger.

``--score-v0`` scores the legacy trade list. ``--pool`` runs one pass over
the 20261019 listing and compares:

- V0: legacy B123 entry, ATR stop, path target
- V4: ex-ante residual + POT/split left rail, same exit
- V6: legacy entry, exit at max hold only (no ATR, no path target)

Thresholds are PROTOCOL. This script does not retune them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from etf_daily.lib.fair_path_tail_audit import (  # noqa: E402
    PROTOCOL,
    acceptance_vs_legacy,
    pnl_distribution,
)
from etf_daily.paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

OUT = PLOTLY_OUTPUTS_DIR / "research" / "fair_path_fat_tail_audit"
IMPLEMENTED = {"V0": "legacy ATR stop + path target, scored only"}


def _independent_days(trades: pd.DataFrame) -> int:
    if trades.empty or "entry_date" not in trades.columns:
        return 0
    return int(pd.to_datetime(trades["entry_date"]).nunique())


def score_v0(trades_csv: Path) -> dict:
    trades = pd.read_csv(trades_csv)
    # Development vs final split is by entry date. Final period is reported
    # and not used to choose a variant.
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    final_start = pd.Timestamp(PROTOCOL["final_start"])
    dev = trades[trades["entry_date"] <= pd.Timestamp(PROTOCOL["dev_end"])]
    final = trades[trades["entry_date"] >= final_start]
    report = {
        "variant": "V0",
        "implemented": sorted(IMPLEMENTED),
        "not_implemented": [v for v in PROTOCOL["variants"] if v not in IMPLEMENTED],
        "dev": pnl_distribution(dev["ret"]) if len(dev) else {"n": 0},
        "final": pnl_distribution(final["ret"]) if len(final) else {"n": 0},
        "dev_independent_days": _independent_days(dev),
        "final_independent_days": _independent_days(final),
        "protocol": PROTOCOL,
    }
    # V0 against itself must pass the numeric gates when the sample is large
    # enough. A thin final sample is "证据不足", not a pass.
    self_check = acceptance_vs_legacy(
        report["final"],
        {
            **report["final"],
            "median_diff_lo": 0.0,
            "median_diff_hi": 0.0,
            "shadow_mean_diff_lo": 0.0,
            "shadow_mean_diff_hi": 0.0,
        },
        independent_days=report["final_independent_days"],
    )
    report["v0_self_check"] = self_check
    return report


def _simulate_max_hold(ms: pd.DataFrame, entries: pd.Series, *, max_hold: int) -> list[dict]:
    close = ms["px"]
    trades: list[dict] = []
    i = 0
    n = len(ms)
    while i < n - 1:
        if not bool(entries.iloc[i]):
            i += 1
            continue
        entry_i = i + 1
        if entry_i >= n:
            break
        entry_px = float(close.iloc[entry_i])
        exit_i = min(n - 1, entry_i + int(max_hold))
        exit_px = float(close.iloc[exit_i])
        trades.append(
            {
                "entry_date": ms.index[entry_i].strftime("%Y-%m-%d"),
                "ret": exit_px / entry_px - 1.0,
                "code": "",
            }
        )
        i = exit_i + 1
    return trades


def run_pool() -> dict:
    from etf_daily.lib.fair_path_band_signals import (
        SKIP_CODES_NO_FIG7,
        compute_multi_scale_frame,
        detect_checklist_entries,
        find_regime_html,
        load_ohlcv_from_regime_html,
    )
    from etf_daily.lib.fair_path_tail_audit import date_block_mean_diff
    from etf_daily.lib.fair_path_variants import exante_pot_lower_touch
    from etf_daily.plots.plot_regime_transition_example import _trail_fit_min_bars
    from etf_daily.scripts.backtest_fair_path_band_signals import (
        simulate_trades,
    )

    listing_dir = PLOTLY_OUTPUTS_DIR / "20261019_from_listing"
    listing = pd.read_csv(listing_dir / "listing_dates.csv")
    codes = [str(c).upper() for c in listing["code"].tolist()]
    buckets: dict[str, list[dict]] = {"V0": [], "V4": [], "V6": []}
    for code in codes:
        if code in SKIP_CODES_NO_FIG7:
            continue
        html = find_regime_html(listing_dir, code)
        if html is None:
            continue
        close, ohlcv = load_ohlcv_from_regime_html(html)
        if len(close) < 120:
            continue
        sub = listing[listing["code"].astype(str).str.upper() == code]
        listing_start = str(sub.iloc[0]["listing"]) if len(sub) else None
        ms = compute_multi_scale_frame(close)
        legacy = detect_checklist_entries(
            ms, mode="B123", code=code, listing_start=listing_start
        )
        touch = np.zeros(len(close), dtype=bool)
        for years in (0.5, 0.25):
            touch |= exante_pot_lower_touch(
                close,
                trail_bars=int(round(years * 252)),
                min_bars=_trail_fit_min_bars(years),
            )
        deep = (ms["fig8_gap"] <= float(PROTOCOL["deep_gap"])).fillna(False).to_numpy()
        from etf_daily.lib.fair_path_band_signals import build_checklist_flags_frame

        flag_df = build_checklist_flags_frame(
            ms, code=code, listing_start=listing_start
        )
        b1 = touch | deep
        challenger = pd.Series(
            b1 & flag_df["B2"].to_numpy() & flag_df["B3"].to_numpy(),
            index=ms.index,
        )
        hold = int(PROTOCOL["max_hold_days"])
        for label, entries, hold_only in (
            ("V0", legacy, False),
            ("V4", challenger, False),
            ("V6", legacy, True),
        ):
            if hold_only:
                got = _simulate_max_hold(ms, entries, max_hold=hold)
            else:
                got = simulate_trades(
                    ms, ohlcv, entries=entries, target_panel="fig10", max_hold=hold
                )
            for tr in got:
                tr["code"] = code
            buckets[label].extend(got)
        print(f"[ab] {code} v0={int(legacy.sum())} v4={int(challenger.sum())}", flush=True)

    final_start = pd.Timestamp(PROTOCOL["final_start"])
    dev_end = pd.Timestamp(PROTOCOL["dev_end"])
    report: dict = {"variants": {}}
    frames = {}
    for label, rows in buckets.items():
        df = pd.DataFrame(rows)
        frames[label] = df
        if df.empty:
            report["variants"][label] = {"n": 0}
            continue
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        final = df[df["entry_date"] >= final_start]
        dev = df[df["entry_date"] <= dev_end]
        report["variants"][label] = {
            "all": pnl_distribution(df["ret"]),
            "dev": pnl_distribution(dev["ret"]) if len(dev) else {"n": 0},
            "final": pnl_distribution(final["ret"]) if len(final) else {"n": 0},
            "final_independent_days": int(final["entry_date"].nunique()) if len(final) else 0,
        }
    v0f = frames["V0"]
    v0f = v0f[pd.to_datetime(v0f["entry_date"]) >= final_start] if len(v0f) else v0f
    for label in ("V4", "V6"):
        chall = frames[label]
        chall = chall[pd.to_datetime(chall["entry_date"]) >= final_start] if len(chall) else chall
        if v0f.empty or chall.empty:
            report["variants"][label]["acceptance"] = {"pass": False, "reasons": ["no_trades"]}
            continue
        diff = date_block_mean_diff(
            v0f["entry_date"],
            v0f["ret"],
            chall["entry_date"],
            chall["ret"],
            block=int(PROTOCOL["block_len"]),
            n_boot=int(PROTOCOL["n_boot"]),
            seed=int(PROTOCOL["boot_seed"]),
        )
        final_metrics = dict(report["variants"][label]["final"])
        final_metrics["median_diff_lo"] = diff["median_lo"]
        final_metrics["median_diff_hi"] = diff["median_hi"]
        # Shadow mean moves with the same trades as the mean. Its interval is
        # the date-block interval of the mean difference: a negative mean
        # interval is also a negative shadow-mean interval, and the point
        # shadow means are reported beside it.
        final_metrics["shadow_mean_diff_lo"] = diff["lo"]
        final_metrics["shadow_mean_diff_hi"] = diff["hi"]
        report["variants"][label]["final_diff"] = diff
        report["variants"][label]["acceptance"] = acceptance_vs_legacy(
            report["variants"]["V0"]["final"],
            final_metrics,
            independent_days=int(report["variants"][label]["final_independent_days"]),
        )
    return report


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    trades = OUT / "v0_backtest" / "trades_B123.csv"
    if trades.is_file():
        payload = score_v0(trades)
        (OUT / "ab_v0.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if not args.pool:
        if trades.is_file():
            print(json.dumps(payload["final"], indent=2, default=str))
        return
    report = run_pool()
    (OUT / "ab_pool.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    brief = {
        k: {
            "final_n": v.get("final", {}).get("n"),
            "final_median": v.get("final", {}).get("median"),
            "final_worst5pct": v.get("final", {}).get("worst5pct_mean"),
            "acceptance": v.get("acceptance"),
        }
        for k, v in report["variants"].items()
    }
    print(json.dumps(brief, indent=2, default=str))


if __name__ == "__main__":
    main()
