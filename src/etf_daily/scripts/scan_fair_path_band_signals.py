#!/usr/bin/env python3
"""Scan all symbols in a from_listing dir for fig7–11 auxiliary buy/sell on one as-of day.

Research only — does not change production hold_up.
Default data: frozen regime HTML (same as backtest --data-source html).

Example::

    PYTHONPATH=. python src/etf_daily/scripts/scan_fair_path_band_signals.py \\
      --listing-dir ~/etf-daily-output/temp/plotly_outputs/20260821_from_listing \\
      --as-of 2026-08-21 \\
      --entry-mode B123
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from etf_daily.lib.fair_path_band_signals import (  # noqa: E402
    ENTRY_MODES,
    atr20,
    compute_multi_scale_frame,
    enrich_frame_for_checklist,
    evaluate_checklist_row,
    find_regime_html,
    load_ohlcv_from_regime_html,
    load_qfq_close,
)

SKIP_CODES_NO_FIG7 = {"SH511010", "SH511260"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--listing-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260821_from_listing",
    )
    p.add_argument("--as-of", required=True, help="YYYY-MM-DD")
    p.add_argument(
        "--data-source",
        default="html",
        choices=["html", "qlib"],
        help="html = read regime_transition_*.html (no Qlib)",
    )
    p.add_argument("--entry-mode", default="B123", choices=list(ENTRY_MODES))
    p.add_argument("--deep-gap", type=float, default=-0.15)
    p.add_argument("--touch-panels", default="fig9,fig10")
    p.add_argument(
        "--b3-mode",
        default="default",
        choices=["default", "fig10_only", "reclaim_from_lo"],
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: {listing-dir}/fair_path_scan/{as_of}",
    )
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--code", action="append", default=None)
    return p.parse_args(argv)


def _load_marks(trades_dir: Path, code: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {"ed": set(), "ru": set(), "buy": set(), "sell": set()}
    ed = trades_dir / f"{code}_early_down_marks.csv"
    if ed.exists():
        for _, r in pd.read_csv(ed).iterrows():
            out["ed"].add(pd.Timestamp(r["date"]).strftime("%Y-%m-%d"))
    ru = trades_dir / f"{code}_recovery_up_marks.csv"
    if ru.exists():
        for _, r in pd.read_csv(ru).iterrows():
            d0 = pd.Timestamp(r["date"]).strftime("%Y-%m-%d")
            d1 = pd.Timestamp(r["end"]).strftime("%Y-%m-%d")
            for d in pd.date_range(d0, d1, freq="B"):
                out["ru"].add(d.strftime("%Y-%m-%d"))
    tr = trades_dir / f"{code}_trades.csv"
    if tr.exists():
        for _, r in pd.read_csv(tr).iterrows():
            ds = pd.Timestamp(r["date"]).strftime("%Y-%m-%d")
            side = str(r["side"]).upper()
            if side == "BUY":
                out["buy"].add(ds)
            elif side == "SELL":
                out["sell"].add(ds)
    return out


def _load_regime_segments(trades_dir: Path, code: str) -> pd.Series:
    seg_path = trades_dir / f"{code}_segments.csv"
    if not seg_path.exists():
        return pd.Series(dtype=object)
    seg = pd.read_csv(seg_path)
    seg["start"] = pd.to_datetime(seg["start"])
    seg["end"] = pd.to_datetime(seg["end"])
    pieces: list[pd.Series] = []
    for _, r in seg.iterrows():
        idx = pd.date_range(r["start"], r["end"], freq="B")
        pieces.append(pd.Series(str(r["regime"]), index=idx))
    if not pieces:
        return pd.Series(dtype=object)
    s = pd.concat(pieces)
    return s[~s.index.duplicated(keep="last")]


def _name_from_html(listing_dir: Path, code: str) -> str:
    html = find_regime_html(listing_dir, code)
    if html is None:
        return ""
    m = re.match(rf"regime_transition_{code}_(.+?)_\d{{8}}_\d{{8}}_adaptive\.html", html.name)
    return m.group(1) if m else ""


def _action_from_row(cr_flags: dict[str, bool], *, entry_mode: str, production_buy: bool, production_sell: bool) -> str:
    if production_sell or cr_flags.get("S3_production_sell"):
        return "生产卖(S3)"
    if cr_flags.get("S1_upper_band"):
        return "减仓观察(S1)"
    if cr_flags.get("S2_repair_fail"):
        return "持仓减仓警惕(S2)"
    b1, b2, b3 = cr_flags.get("B1_position"), cr_flags.get("B2_long_ok"), cr_flags.get("B3_short_reclaim")
    b4, b5 = cr_flags.get("B4_no_conflict"), cr_flags.get("B5_rr_ok")
    if b1 and not b2:
        return "禁止(跌刀)"
    ok = bool(b1 and b2)
    if entry_mode in {"B123", "B1235", "B12345"}:
        ok = ok and bool(b3)
    if entry_mode in {"B1235", "B12345"}:
        ok = ok and bool(b5)
    if entry_mode == "B12345":
        ok = ok and bool(b4)
    if entry_mode == "B12":
        ok = bool(b1 and b2)
    if ok and production_buy:
        return "辅助买+生产BUY"
    if ok:
        return "辅助买观察"
    if b1 and b2 and not b3:
        return "观察清单(缺B3)"
    return "中性"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    listing_dir = args.listing_dir
    if not listing_dir.is_absolute():
        listing_dir = _PROJECT_ROOT / listing_dir
    as_of = pd.Timestamp(args.as_of).strftime("%Y-%m-%d")
    out_dir = args.out_dir or (listing_dir / "fair_path_scan" / as_of.replace("-", ""))
    if not out_dir.is_absolute():
        out_dir = _PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ld = listing_dir / "listing_dates.csv"
    if not ld.exists():
        raise SystemExit(f"missing {ld}")
    listing = pd.read_csv(ld)
    if args.code:
        codes = [c.upper() for c in args.code]
    else:
        codes = listing["code"].astype(str).str.upper().tolist()
    if args.max_codes:
        codes = codes[: int(args.max_codes)]

    touch_panels = tuple(p.strip() for p in args.touch_panels.split(",") if p.strip())
    trades_dir = listing_dir / "trades"
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    qlib_inited = False

    for code in codes:
        if code in SKIP_CODES_NO_FIG7:
            errors.append({"code": code, "skip": "no_fig7_panels"})
            continue
        sub = listing[listing["code"].astype(str).str.upper() == code]
        if sub.empty:
            continue
        listing_start = pd.Timestamp(sub.iloc[0]["listing"]).strftime("%Y-%m-%d")
        age_years = (pd.Timestamp(as_of) - pd.Timestamp(listing_start)).days / 365.25
        try:
            if args.data_source == "html":
                html_path = find_regime_html(listing_dir, code)
                if html_path is None:
                    errors.append({"code": code, "skip": "html_missing"})
                    continue
                close, ohlcv = load_ohlcv_from_regime_html(html_path)
                close = close[close.index <= pd.Timestamp(as_of)]
                ohlcv = ohlcv[ohlcv.index <= pd.Timestamp(as_of)]
            else:
                init = not qlib_inited
                close, ohlcv = load_qfq_close(code, listing_start, as_of, init_qlib=init)
                qlib_inited = True
            if len(close) < 60:
                errors.append({"code": code, "skip": "short_history"})
                continue
            snap_dt = close.index[close.index <= pd.Timestamp(as_of)]
            if len(snap_dt) == 0:
                errors.append({"code": code, "skip": "no_bar_on_asof"})
                continue
            snap_dt = snap_dt[-1]
            snap_date = pd.Timestamp(snap_dt).strftime("%Y-%m-%d")

            ms = enrich_frame_for_checklist(compute_multi_scale_frame(close))
            marks = _load_marks(trades_dir, code)
            regime_s = _load_regime_segments(trades_dir, code)
            atr_s = atr20(ohlcv).reindex(ms.index)
            row = ms.loc[snap_dt]
            atr = float(atr_s.loc[snap_dt]) if snap_dt in atr_s.index and pd.notna(atr_s.loc[snap_dt]) else None
            regime = None
            if not regime_s.empty and snap_dt in regime_s.index:
                regime = str(regime_s.loc[snap_dt])
            elif not regime_s.empty:
                # business-day segments may miss weekends; nearest prior
                prior = regime_s.index[regime_s.index <= snap_dt]
                if len(prior):
                    regime = str(regime_s.loc[prior[-1]])

            cr = evaluate_checklist_row(
                row,
                code=code,
                atr=atr,
                regime=regime,
                has_ed=snap_date in marks["ed"],
                has_ru=snap_date in marks["ru"],
                production_buy=snap_date in marks["buy"],
                production_sell=snap_date in marks["sell"],
                deep_gap=args.deep_gap,
                listing_age_years=age_years,
                touch_panels=touch_panels,
                b3_mode=args.b3_mode,  # type: ignore[arg-type]
            )
            action = _action_from_row(
                cr.flags,
                entry_mode=args.entry_mode,
                production_buy=cr.production_buy,
                production_sell=cr.production_sell,
            )
            rows.append(
                {
                    "code": code,
                    "name": _name_from_html(listing_dir, code),
                    "as_of": as_of,
                    "snap_date": snap_date,
                    "close": round(cr.px, 4),
                    "regime": regime,
                    "action": action,
                    "verdict": cr.verdict,
                    "B1": cr.flags.get("B1_position"),
                    "B2": cr.flags.get("B2_long_ok"),
                    "B3": cr.flags.get("B3_short_reclaim"),
                    "B4": cr.flags.get("B4_no_conflict"),
                    "B5": cr.flags.get("B5_rr_ok"),
                    "B6": cr.flags.get("B6_production_buy"),
                    "S1": cr.flags.get("S1_upper_band"),
                    "S2": cr.flags.get("S2_repair_fail"),
                    "S3": cr.flags.get("S3_production_sell"),
                    "has_ed": cr.has_ed,
                    "has_ru": cr.has_ru,
                    "fig8_gap": round(float(row["fig8_gap"]), 4) if pd.notna(row.get("fig8_gap")) else None,
                    "fig7_g": round(float(row["fig7_g"]), 4) if pd.notna(row.get("fig7_g")) else None,
                    "fig8_g": round(float(row["fig8_g"]), 4) if pd.notna(row.get("fig8_g")) else None,
                    "fig10_g": round(float(row["fig10_g"]), 4) if pd.notna(row.get("fig10_g")) else None,
                    "fig10_path": round(float(row["fig10_path"]), 4) if pd.notna(row.get("fig10_path")) else None,
                    "rr_fig8": cr.rr_to_fig8_path,
                    "rr_fig10": cr.rr_to_fig10_path,
                    "notes": "|".join(cr.notes),
                }
            )
        except Exception as exc:  # noqa: BLE001
            errors.append({"code": code, "skip": str(exc)})

    df = pd.DataFrame(rows)
    err_df = pd.DataFrame(errors)
    scan_path = out_dir / f"scan_{as_of.replace('-', '')}.csv"
    df.to_csv(scan_path, index=False, encoding="utf-8-sig")
    if not err_df.empty:
        err_df.to_csv(out_dir / "skipped.csv", index=False, encoding="utf-8-sig")

    interesting = {
        "辅助买观察",
        "辅助买+生产BUY",
        "观察清单(缺B3)",
        "禁止(跌刀)",
        "减仓观察(S1)",
        "持仓减仓警惕(S2)",
        "生产卖(S3)",
    }
    hits = df[df["action"].isin(interesting)].copy() if not df.empty else df
    hits_path = out_dir / f"signals_{as_of.replace('-', '')}.csv"
    if not hits.empty:
        order = [
            "辅助买+生产BUY",
            "辅助买观察",
            "观察清单(缺B3)",
            "禁止(跌刀)",
            "生产卖(S3)",
            "持仓减仓警惕(S2)",
            "减仓观察(S1)",
        ]
        hits["_ord"] = hits["action"].map({a: i for i, a in enumerate(order)}).fillna(99)
        hits = hits.sort_values(["_ord", "code"]).drop(columns=["_ord"])
    hits.to_csv(hits_path, index=False, encoding="utf-8-sig")

    summary: dict[str, Any] = {
        "as_of": as_of,
        "listing_dir": str(listing_dir),
        "entry_mode": args.entry_mode,
        "deep_gap": args.deep_gap,
        "data_source": args.data_source,
        "n_scanned": len(df),
        "n_skipped": len(err_df),
        "action_counts": df["action"].value_counts().to_dict() if not df.empty else {},
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[OK] scan as_of={as_of} n={len(df)} signals={len(hits)} out={out_dir}")
    if not df.empty:
        print("action counts:")
        for k, v in df["action"].value_counts().items():
            print(f"  {k}: {v}")
    if not hits.empty:
        print("\n非中性信号:")
        cols = ["code", "name", "action", "close", "regime", "B1", "B2", "B3", "fig8_gap"]
        print(hits[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
