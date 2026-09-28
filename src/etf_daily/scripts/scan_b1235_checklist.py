#!/usr/bin/env python3
"""B1235 checklist scan for a from_listing EOD batch (one as-of day).

For every equity ETF in --listing-dir, compute the B1/B2/B3/B5 auxiliary buy
conditions from 图7-11公平路径上下轨辅助买卖.md (same scoring as
``evaluate_checklist_row``: deep gap -15%, min R:R 2, stop = close - ATR20)
and write one CSV with the raw readings plus per-condition verdicts.

Research scan only (does not change production hold_up).

Daily use (same style as ``etf-daily triggers``; ``--as-of`` defaults to the
date in the directory name)::

    etf-daily b1235 --listing-dir "$PLOTLY_ROOT/20260928_from_listing"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.lib.fair_path_band_signals import (  # noqa: E402
    DEFAULT_DEEP_GAP,
    DEFAULT_MIN_RR,
    atr20,
    compute_multi_scale_frame,
    enrich_frame_for_checklist,
    evaluate_checklist_row,
    load_ohlcv_from_regime_html,
)
from etf_daily.scripts import backtest_fig12_six_states_pool as pool  # noqa: E402

COLS = (
    "代码",
    "名称",
    "T收盘",
    "图8折价",
    "图9触下轨",
    "图10触下轨",
    "B1达成",
    "上市年数",
    "图7年化",
    "图8年化",
    "图7路径20日抬升",
    "图8路径20日抬升",
    "B2达成",
    "图10路径",
    "站上图10",
    "图11路径",
    "站上图11",
    "图10昨触今收回",
    "B3达成",
    "ATR20",
    "止损价",
    "图8目标价",
    "盈亏比图8",
    "图10目标价",
    "盈亏比图10",
    "B5达成",
    "B1235达成",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--listing-dir", type=Path, required=True)
    p.add_argument("--as-of", required=True, help="YYYY-MM-DD (T close)")
    p.add_argument("--out", type=Path, default=None,
                   help="default: {listing-dir}/b1235_checklist_{as-of}.csv")
    p.add_argument("--deep-gap", type=float, default=DEFAULT_DEEP_GAP)
    p.add_argument("--min-rr", type=float, default=DEFAULT_MIN_RR)
    return p.parse_args(argv)


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _pct(v: Any, digits: int = 1) -> str:
    x = _f(v)
    return "—" if x is None else f"{x:.{digits}%}"


def _px(v: Any) -> str:
    x = _f(v)
    return "—" if x is None else f"{x:.4f}"


def _rr(v: Any) -> str:
    x = _f(v)
    return "—" if x is None else f"{x:.2f}"


def _yn(b: Any) -> str:
    return "是" if bool(b) else ""


def _load_listing_dates(listing: Path) -> dict[str, str]:
    ld = listing / "listing_dates.csv"
    if not ld.exists():
        return {}
    df = pd.read_csv(ld)
    return {
        str(r["code"]).upper(): pd.Timestamp(r["listing"]).strftime("%Y-%m-%d")
        for _, r in df.iterrows()
    }


def scan_one(
    html: Path,
    *,
    as_of: pd.Timestamp,
    listing_start: str | None,
    deep_gap: float,
    min_rr: float,
) -> dict[str, str]:
    code, name = pool.parse_html_label(html)
    close, ohlcv = load_ohlcv_from_regime_html(html)
    close = close.loc[close.index <= as_of]
    ohlcv = ohlcv.loc[ohlcv.index <= as_of]
    if close.empty:
        raise ValueError(f"no bars on/before {as_of.date()}")
    snap_dt = close.index[-1]
    if snap_dt != as_of:
        raise ValueError(f"last bar {snap_dt.date()} != as-of {as_of.date()}")

    age_years = None
    if listing_start:
        age_years = (as_of - pd.Timestamp(listing_start)).days / 365.25

    ms = enrich_frame_for_checklist(compute_multi_scale_frame(close))
    atr_s = atr20(ohlcv).reindex(ms.index)
    row = ms.loc[snap_dt]
    atr = _f(atr_s.loc[snap_dt]) if snap_dt in atr_s.index else None

    cr = evaluate_checklist_row(
        row,
        code=code,
        atr=atr,
        deep_gap=deep_gap,
        min_rr=min_rr,
        listing_age_years=age_years,
    )
    b1 = bool(cr.flags.get("B1_position"))
    b2 = bool(cr.flags.get("B2_long_ok"))
    b3 = bool(cr.flags.get("B3_short_reclaim"))
    b5 = bool(cr.flags.get("B5_rr_ok"))

    px = float(row["px"])
    fig10_lower = _f(row.get("fig10_lower"))
    reclaim = bool(
        bool(row.get("fig10_prev_touch_lo", False))
        and fig10_lower is not None
        and px > fig10_lower
    )
    stop = px - atr if atr is not None else None

    return {
        "代码": code,
        "名称": name,
        "T收盘": _px(px),
        "图8折价": _pct(row.get("fig8_gap")),
        "图9触下轨": _yn(row.get("fig9_touch_lo", False)),
        "图10触下轨": _yn(row.get("fig10_touch_lo", False)),
        "B1达成": _yn(b1),
        "上市年数": "—" if age_years is None else f"{age_years:.1f}",
        "图7年化": _pct(row.get("fig7_g")),
        "图8年化": _pct(row.get("fig8_g")),
        "图7路径20日抬升": _yn(row.get("fig7_path_rising", False)),
        "图8路径20日抬升": _yn(row.get("fig8_path_rising", False)),
        "B2达成": _yn(b2),
        "图10路径": _px(row.get("fig10_path")),
        "站上图10": _yn(row.get("fig10_above_path", False)),
        "图11路径": _px(row.get("fig11_path")),
        "站上图11": _yn(row.get("fig11_above_path", False)),
        "图10昨触今收回": _yn(reclaim),
        "B3达成": _yn(b3),
        "ATR20": _px(atr),
        "止损价": _px(stop),
        "图8目标价": _px(row.get("fig8_path")),
        "盈亏比图8": _rr(cr.rr_to_fig8_path),
        "图10目标价": _px(row.get("fig10_path")),
        "盈亏比图10": _rr(cr.rr_to_fig10_path),
        "B5达成": _yn(b5),
        "B1235达成": _yn(b1 and b2 and b3 and b5),
        "_gap": _f(row.get("fig8_gap")),
        "_pass": bool(b1 and b2 and b3 and b5),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    listing = args.listing_dir if args.listing_dir.is_absolute() else _PROJECT_ROOT / args.listing_dir
    as_of = pd.Timestamp(args.as_of).normalize()
    tag = as_of.strftime("%Y%m%d")
    out = args.out or (listing / f"b1235_checklist_{tag}.csv")

    listing_dates = _load_listing_dates(listing)
    htmls = pool.list_equity_html(listing)  # prints skipped bonds

    records: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    for html in htmls:
        code, name = pool.parse_html_label(html)
        try:
            records.append(
                scan_one(
                    html,
                    as_of=as_of,
                    listing_start=listing_dates.get(code.upper()),
                    deep_gap=float(args.deep_gap),
                    min_rr=float(args.min_rr),
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append({"code": code, "name": name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[error] {code} {name}: {exc}", flush=True)

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(
            by=["_pass", "_gap", "代码"],
            ascending=[False, True, True],
            na_position="last",
            kind="mergesort",
        )
    out_df = df.loc[:, list(COLS)] if not df.empty else pd.DataFrame(columns=list(COLS))
    out_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[OK] {out}", flush=True)

    if errors:
        err_path = out.with_name(out.stem + "_skipped.csv")
        pd.DataFrame(errors).to_csv(err_path, index=False, encoding="utf-8-sig")
        print(f"[OK] {err_path} ({len(errors)} skipped)", flush=True)

    if not df.empty:
        b1 = df["B1达成"].eq("是")
        b2 = df["B2达成"].eq("是")
        b3 = df["B3达成"].eq("是")
        b5 = df["B5达成"].eq("是")
        print(f"扫描 {len(df)} 只（as-of {as_of.date()}）:")
        print(f"  B1            : {int(b1.sum())}")
        print(f"  B1∧B2         : {int((b1 & b2).sum())}")
        print(f"  B1∧B2∧B3      : {int((b1 & b2 & b3).sum())}")
        print(f"  B1235         : {int((b1 & b2 & b3 & b5).sum())}")
        passed = df.loc[df["B1235达成"].eq("是"), ["代码", "名称"]]
        for _, r in passed.iterrows():
            print(f"    {r['代码']} {r['名称']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
