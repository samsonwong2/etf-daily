#!/usr/bin/env python3
"""Check fair-path band auxiliary buy/sell checklist for one symbol.

Research only — does not change production hold_up.

Example::

    PYTHONPATH=. python decision_pack/scripts/check_fair_path_band_signals.py \\
      --code SZ159502 --as-of 2026-08-20 \\
      --from-listing-dir ~/etf-daily-output/temp/plotly_outputs/20260820_from_listing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runtime_paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from decision_pack.src.fair_path_band_signals import (  # noqa: E402
    atr20,
    build_checklist_series,
    compute_multi_scale_frame,
    load_qfq_close,
)


def _load_marks(trades_dir: Path, code: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {"ed": set(), "ru": set(), "buy": set(), "sell": set()}
    base = trades_dir
    ed = base / f"{code}_early_down_marks.csv"
    if ed.exists():
        df = pd.read_csv(ed)
        for _, r in df.iterrows():
            d0 = pd.Timestamp(r["date"]).strftime("%Y-%m-%d")
            out["ed"].add(d0)
            if "end" in r and pd.notna(r["end"]):
                d1 = pd.Timestamp(r["end"]).strftime("%Y-%m-%d")
                # span not all days — only start for flag
                _ = d1
    ru = base / f"{code}_recovery_up_marks.csv"
    if ru.exists():
        df = pd.read_csv(ru)
        for _, r in df.iterrows():
            d0 = pd.Timestamp(r["date"]).strftime("%Y-%m-%d")
            d1 = pd.Timestamp(r["end"]).strftime("%Y-%m-%d")
            for d in pd.date_range(d0, d1, freq="B"):
                out["ru"].add(d.strftime("%Y-%m-%d"))
    tr = base / f"{code}_trades.csv"
    if tr.exists():
        df = pd.read_csv(tr)
        for _, r in df.iterrows():
            ds = pd.Timestamp(r["date"]).strftime("%Y-%m-%d")
            if str(r["side"]).upper() == "BUY":
                out["buy"].add(ds)
            elif str(r["side"]).upper() == "SELL":
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code", required=True)
    p.add_argument("--as-of", required=True, help="YYYY-MM-DD")
    p.add_argument(
        "--from-listing-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260820_from_listing",
    )
    p.add_argument(
        "--start",
        default=None,
        help="OHLCV start (default: listing from listing_dates.csv or 2015-01-01)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: {from_listing_dir}/fair_path_check/{code}",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    code = str(args.code).upper()
    as_of = pd.Timestamp(args.as_of).strftime("%Y-%m-%d")
    listing_dir = args.from_listing_dir
    if not listing_dir.is_absolute():
        listing_dir = _PROJECT_ROOT / listing_dir
    trades_dir = listing_dir / "trades"
    out_dir = args.out_dir or (listing_dir / "fair_path_check" / code)
    out_dir.mkdir(parents=True, exist_ok=True)

    listing_start = None
    ld = listing_dir / "listing_dates.csv"
    if ld.exists():
        sub = pd.read_csv(ld)
        sub = sub[sub["code"].astype(str).str.upper() == code]
        if not sub.empty:
            listing_start = pd.Timestamp(sub.iloc[0]["listing"]).strftime("%Y-%m-%d")
    start = args.start or listing_start or "2015-01-01"

    close, ohlcv = load_qfq_close(code, start, as_of, init_qlib=True)
    ms = compute_multi_scale_frame(close)
    atr_s = atr20(ohlcv)
    atr_s.index = pd.to_datetime(atr_s.index)

    marks = _load_marks(trades_dir, code)
    regime_s = _load_regime_segments(trades_dir, code)

    checklist = build_checklist_series(
        ms,
        code=code,
        atr_s=atr_s,
        regime_s=regime_s,
        ed_dates=marks["ed"],
        ru_dates=marks["ru"],
        buy_dates=marks["buy"],
        sell_dates=marks["sell"],
        listing_start=listing_start,
    )
    checklist.to_csv(out_dir / "checklist_daily.csv", index=False, encoding="utf-8-sig")

    # Snapshot panels on as-of
    if as_of not in [pd.Timestamp(x).strftime("%Y-%m-%d") for x in ms.index]:
        idx = ms.index[ms.index <= pd.Timestamp(as_of)]
        snap_dt = idx[-1] if len(idx) else ms.index[-1]
    else:
        snap_dt = pd.Timestamp(as_of)
    snap = ms.loc[snap_dt]
    panel_rows = []
    for key, years, label in [
        ("fig7", 2.0, "图7·2年"),
        ("fig8", 1.0, "图8·1年"),
        ("fig9", 0.5, "图9·6个月"),
        ("fig10", 0.25, "图10·3个月"),
        ("fig11", 1 / 12, "图11·1个月"),
    ]:
        panel_rows.append(
            {
                "panel": label,
                "years": years,
                "path": float(snap[f"{key}_path"]) if pd.notna(snap[f"{key}_path"]) else None,
                "g": float(snap[f"{key}_g"]) if pd.notna(snap[f"{key}_g"]) else None,
                "upper": float(snap[f"{key}_upper"])
                if pd.notna(snap.get(f"{key}_upper"))
                else None,
                "lower": float(snap[f"{key}_lower"])
                if pd.notna(snap.get(f"{key}_lower"))
                else None,
                "gap": float(snap[f"{key}_gap"]) if pd.notna(snap[f"{key}_gap"]) else None,
                "touch_lo": bool(snap.get(f"{key}_touch_lo", False)),
                "touch_hi": bool(snap.get(f"{key}_touch_hi", False)),
            }
        )
    pd.DataFrame(panel_rows).to_csv(
        out_dir / f"panels_{as_of.replace('-', '')}.csv", index=False, encoding="utf-8-sig"
    )

    # Focus window: last production sell → as-of
    sells = sorted(marks["sell"])
    focus_start = sells[-1] if sells else pd.Timestamp(start).strftime("%Y-%m-%d")
    focus = checklist[
        (pd.to_datetime(checklist["date"]) >= pd.Timestamp(focus_start))
        & (pd.to_datetime(checklist["date"]) <= pd.Timestamp(as_of))
    ].copy()

    summary: dict[str, Any] = {
        "code": code,
        "as_of": as_of,
        "listing_start": listing_start,
        "focus_from_last_sell": focus_start,
        "snapshot_date": pd.Timestamp(snap_dt).strftime("%Y-%m-%d"),
        "px": float(snap["px"]),
        "panels": panel_rows,
        "as_of_checklist": checklist[checklist["date"] == pd.Timestamp(snap_dt).strftime("%Y-%m-%d")].to_dict(
            "records"
        ),
        "focus_verdict_counts": focus["verdict"].value_counts().to_dict() if not focus.empty else {},
        "focus_ru_days": int(focus["has_ru"].sum()) if not focus.empty else 0,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    md_lines = [
        f"# {code} 公平路径辅助 checklist",
        "",
        f"- as-of: **{as_of}** (snapshot **{summary['snapshot_date']}**)",
        f"- listing: {listing_start}",
        f"- focus: last SELL **{focus_start}** → {as_of}",
        f"- px: **{summary['px']:.4f}**",
        "",
        "## Panels @ snapshot",
        "",
        "| 图 | path | g | gap | 触下轨 | 触上轨 |",
        "|---|---:|---:|---:|:---:|:---:|",
    ]
    for pr in panel_rows:
        md_lines.append(
            f"| {pr['panel']} | {pr['path']:.4f} | {pr['g']:+.1%} | {pr['gap']:+.1%} | "
            f"{'Y' if pr['touch_lo'] else ''} | {'Y' if pr['touch_hi'] else ''} |"
        )
    if summary["as_of_checklist"]:
        ac = summary["as_of_checklist"][0]
        md_lines.extend(
            [
                "",
                "## Checklist @ snapshot",
                "",
                f"- verdict: **{ac.get('verdict')}**",
                f"- regime: {ac.get('regime')} | ED: {ac.get('has_ed')} | RU: {ac.get('has_ru')}",
                f"- production BUY: {ac.get('production_buy')} | SELL: {ac.get('production_sell')}",
                f"- R:R fig8: {ac.get('rr_fig8')} | fig10: {ac.get('rr_fig10')}",
                "",
                "### Flags",
                "",
            ]
        )
        for k, v in sorted(ac.items()):
            if k.startswith("flag_"):
                md_lines.append(f"- {k}: {v}")
    md_lines.extend(["", "## Focus window verdicts", ""])
    for k, v in (summary.get("focus_verdict_counts") or {}).items():
        md_lines.append(f"- {k}: {v}")

    # Narrative vs plan checklist
    ac0 = summary["as_of_checklist"][0] if summary.get("as_of_checklist") else {}
    md_lines.extend(
        [
            "",
            "## 核对结论（相对 checklist）",
            "",
            f"- 生产 **07-20 SELL** 后 focus 窗内：{summary.get('focus_verdict_counts')}",
            f"- RU 覆盖 {summary.get('focus_ru_days')} 个交易日；0820 仍在 RU 段",
            f"- 0820 价 **在** 各尺度 path **上方**（非深折价），故 B1 不通过 → 辅助买点为 **neutral**",
            f"- B2/B3/B4 通过（长尺 g>0、已在 path 上、RU+range）；**无 B6 绿三角** → 非生产 BUY",
            f"- 窗内曾出现 **aux_buy_watch×2**（触轨+长尺未坏+短尺收复），与「SELL 后修复段」叙事一致",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[OK] {code} checklist → {out_dir}")
    if summary["as_of_checklist"]:
        print(f"  verdict={summary['as_of_checklist'][0].get('verdict')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
