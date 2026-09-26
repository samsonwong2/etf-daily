#!/usr/bin/env python3
"""Quiet-discount backtest from fig3–11 (no production fig1 triangles).

Rules (``quiet_discount``):
  Buy (close signal → next-bar fill):
    1. fig7_gap ≤ −12% OR fig7_touch_lo
    2. fig7_g > 0 AND fig8_g > 0
    3. vol5_pct_120d ≤ 0.30
    4. gap_20d ≤ 0
    5. not fig9/fig10 touch_hi
  Sell (any): S1 upper / fig7_gap≥0 / both g<0 / stop ATR20 / max_hold=40

Example::

    PYTHONPATH=. ~/etf-daily-output/python/envs/py312/bin/python \\
      src/etf_daily/scripts/backtest_fig3_11_quiet_discount.py \\
      --html ~/etf-daily-output/temp/plotly_outputs/20260824_from_listing/regime_transition_SH518880_黄金ETF华安_20130729_20260824_adaptive.html
"""
from __future__ import annotations

import argparse
import json
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
    build_quiet_discount_frame,
    find_regime_html,
    load_ohlcv_from_regime_html,
    quiet_discount_entries,
    simulate_quiet_discount_trades,
)
from etf_daily.lib.regime_transition_plot import (  # noqa: E402
    compute_volatility_regime_frame,
)

DEFAULT_HTML = (
    PLOTLY_OUTPUTS_DIR / "20260824_from_listing"
    / "regime_transition_SH518880_黄金ETF华安_20130729_20260824_adaptive.html"
)
DEFAULT_LISTING = PLOTLY_OUTPUTS_DIR / "20260824_from_listing"
DEFAULT_OUT = (
    PLOTLY_OUTPUTS_DIR / "20260824_from_listing"
    / "fig3_11_bt_SH518880"
)
CHECK_DATES = ("2026-08-04", "2026-08-05")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--html", type=Path, default=None, help="target symbol regime HTML")
    p.add_argument("--code", default="SH518880")
    p.add_argument("--listing-dir", type=Path, default=DEFAULT_LISTING)
    p.add_argument("--anchor-code", default="SH510300")
    p.add_argument("--oos-start", default="2025-01-01")
    p.add_argument("--max-hold", type=int, default=40)
    p.add_argument("--fig7-gap-max", type=float, default=-0.12)
    p.add_argument("--vol5-pct-max", type=float, default=0.30)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args(argv)


def _resolve_html(args: argparse.Namespace) -> Path:
    if args.html is not None:
        path = Path(args.html)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    found = find_regime_html(Path(args.listing_dir), str(args.code).upper())
    if found is None:
        if Path(DEFAULT_HTML).exists():
            return Path(DEFAULT_HTML)
        raise FileNotFoundError(
            f"no regime HTML for {args.code} under {args.listing_dir}"
        )
    return found


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _build_equity(
    close: pd.Series,
    trades: list[dict[str, Any]],
) -> pd.DataFrame:
    """Cash=1 flat; fully invested long between entry and exit (inclusive exit day)."""
    idx = close.index
    eq = pd.Series(1.0, index=idx, dtype=float)
    in_pos = pd.Series(False, index=idx)
    for t in trades:
        e0 = pd.Timestamp(t["entry_date"]).normalize()
        e1 = pd.Timestamp(t["exit_date"]).normalize()
        mask = (idx >= e0) & (idx <= e1)
        in_pos.loc[mask] = True
    ret = close.astype(float).pct_change().fillna(0.0)
    # Apply overnight return on days after entry through exit day.
    daily = ret.where(in_pos, 0.0)
    # On entry day we buy at close → no same-day return; shift position by 1
    # so first PnL day is entry+1. Exit day still earns close-to-close vs prior.
    held = in_pos.shift(1, fill_value=False)
    daily = ret.where(held, 0.0)
    eq = (1.0 + daily).cumprod()
    return pd.DataFrame({"equity": eq, "in_position": held.astype(int)}, index=idx)


def _summarize(
    trades_df: pd.DataFrame,
    equity: pd.Series,
    *,
    label: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "label": label,
        "n_trades": int(len(trades_df)),
        "max_drawdown": _max_drawdown(equity),
        "equity_final": float(equity.iloc[-1]) if len(equity) else float("nan"),
    }
    if trades_df.empty:
        out.update(
            win_rate=None,
            mean_ret=None,
            compound=None,
            mean_hold_days=None,
            exit_reason_counts={},
        )
        return out
    out.update(
        win_rate=float((trades_df["ret"] > 0).mean()),
        mean_ret=float(trades_df["ret"].mean()),
        compound=float((1.0 + trades_df["ret"]).prod() - 1.0),
        mean_hold_days=float(trades_df["hold_days"].mean()),
        median_hold_days=float(trades_df["hold_days"].median()),
        exit_reason_counts={
            str(k): int(v)
            for k, v in trades_df["exit_reason"].value_counts().to_dict().items()
        },
    )
    return out


def _check_window_rows(frame: pd.DataFrame, dates: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ds in dates:
        ts = pd.Timestamp(ds).normalize()
        if ts not in frame.index:
            rows.append({"date": ds, "present": False})
            continue
        r = frame.loc[ts]
        rows.append(
            {
                "date": ds,
                "present": True,
                "qd_entry": bool(r.get("qd_entry", False)),
                "fig7_gap": None if pd.isna(r.get("fig7_gap")) else float(r["fig7_gap"]),
                "fig7_touch_lo": bool(r.get("fig7_touch_lo", False)),
                "fig7_g": None if pd.isna(r.get("fig7_g")) else float(r["fig7_g"]),
                "fig8_g": None if pd.isna(r.get("fig8_g")) else float(r["fig8_g"]),
                "vol5_pct_120d": (
                    None
                    if pd.isna(r.get("vol5_pct_120d"))
                    else float(r["vol5_pct_120d"])
                ),
                "gap_20d": None if pd.isna(r.get("gap_20d")) else float(r["gap_20d"]),
                "fig9_touch_hi": bool(r.get("fig9_touch_hi", False)),
                "fig10_touch_hi": bool(r.get("fig10_touch_hi", False)),
                "qd_long_discount": bool(r.get("qd_long_discount", False)),
                "qd_long_ok": bool(r.get("qd_long_ok", False)),
                "qd_low_vol": bool(r.get("qd_low_vol", False)),
                "qd_gap_weak": bool(r.get("qd_gap_weak", False)),
                "qd_not_hot": bool(r.get("qd_not_hot", False)),
            }
        )
    return rows


def _fmt_pct(x: float | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{digits}%}"


def _write_report(
    out_dir: Path,
    *,
    code: str,
    html_path: Path,
    oos_start: str,
    full_sum: dict[str, Any],
    oos_sum: dict[str, Any],
    trades_df: pd.DataFrame,
    check_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    lines = [
        f"# {code} quiet_discount（图3–11）回测报告",
        "",
        f"- HTML: `{html_path}`",
        f"- 锚: {args.anchor_code}（fair need / gap20）",
        f"- OOS start: `{oos_start}`（此前仅预热，信号/成交仍可全样本诊断）",
        f"- 参数: fig7_gap_max={args.fig7_gap_max}, vol5_pct_max={args.vol5_pct_max}, max_hold={args.max_hold}",
        f"- **不用** 图1 生产三角 / hold_up / production BUY·SELL",
        "",
        "## OOS 表现",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| n_trades | {oos_sum.get('n_trades')} |",
        f"| win_rate | {_fmt_pct(oos_sum.get('win_rate'))} |",
        f"| mean_ret | {_fmt_pct(oos_sum.get('mean_ret'))} |",
        f"| compound | {_fmt_pct(oos_sum.get('compound'))} |",
        f"| mean_hold_days | {oos_sum.get('mean_hold_days')} |",
        f"| max_drawdown | {_fmt_pct(oos_sum.get('max_drawdown'))} |",
        f"| equity_final | {oos_sum.get('equity_final')} |",
        f"| exit_reasons | `{json.dumps(oos_sum.get('exit_reason_counts') or {}, ensure_ascii=False)}` |",
        "",
        "## 全样本诊断",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| n_trades | {full_sum.get('n_trades')} |",
        f"| win_rate | {_fmt_pct(full_sum.get('win_rate'))} |",
        f"| mean_ret | {_fmt_pct(full_sum.get('mean_ret'))} |",
        f"| compound | {_fmt_pct(full_sum.get('compound'))} |",
        f"| mean_hold_days | {full_sum.get('mean_hold_days')} |",
        f"| max_drawdown | {_fmt_pct(full_sum.get('max_drawdown'))} |",
        f"| exit_reasons | `{json.dumps(full_sum.get('exit_reason_counts') or {}, ensure_ascii=False)}` |",
        "",
        "## 2026-08-04 / 08-05 静默折价窗口",
        "",
        "收盘信号日是否满足 `qd_entry`（次日成交）：",
        "",
    ]
    for row in check_rows:
        if not row.get("present"):
            lines.append(f"- **{row['date']}**: 不在样本内")
            continue
        lines.append(
            f"- **{row['date']}**: qd_entry=`{row['qd_entry']}` | "
            f"fig7_gap={_fmt_pct(row.get('fig7_gap'))} touch_lo={row.get('fig7_touch_lo')} | "
            f"g7={row.get('fig7_g')} g8={row.get('fig8_g')} | "
            f"vol5_pct={row.get('vol5_pct_120d')} | gap20={_fmt_pct(row.get('gap_20d'))} | "
            f"fig9_hi={row.get('fig9_touch_hi')} fig10_hi={row.get('fig10_touch_hi')}"
        )
        lines.append(
            f"  - gates: discount={row.get('qd_long_discount')} long_ok={row.get('qd_long_ok')} "
            f"low_vol={row.get('qd_low_vol')} gap_weak={row.get('qd_gap_weak')} "
            f"not_hot={row.get('qd_not_hot')}"
        )

    # Trades overlapping the check window (signal / entry / exit / open span)
    lines.extend(["", "### 相关成交 / 仓位说明", ""])
    if trades_df.empty:
        lines.append("_全样本无成交。_")
    else:
        w0 = pd.Timestamp(CHECK_DATES[0]).normalize()
        w1 = pd.Timestamp(CHECK_DATES[-1]).normalize()
        near_rows: list[pd.Series] = []
        for _, t in trades_df.iterrows():
            s = pd.Timestamp(t["signal_date"]).normalize()
            e0 = pd.Timestamp(t["entry_date"]).normalize()
            e1 = pd.Timestamp(t["exit_date"]).normalize()
            overlap = (e0 <= w1) and (e1 >= w0)
            touch = s in {w0, w1} or e0 in {w0, w1} or e1 in {w0, w1}
            if overlap or touch:
                near_rows.append(t)
        for row in check_rows:
            if not row.get("present") or not row.get("qd_entry"):
                continue
            d = pd.Timestamp(row["date"]).normalize()
            blocked = False
            for _, t in trades_df.iterrows():
                e0 = pd.Timestamp(t["entry_date"]).normalize()
                e1 = pd.Timestamp(t["exit_date"]).normalize()
                # Signal at close of d → fill next bar; blocked if still holding through d.
                if e0 <= d < e1:
                    blocked = True
                    lines.append(
                        f"- `{row['date']}` 满足 qd_entry，但当日已持仓"
                        f"（{t['entry_date']}→{t['exit_date']}，{t['exit_reason']}），"
                        f"不新开仓；次日不会单独成交。"
                    )
                    break
            if not blocked:
                lines.append(
                    f"- `{row['date']}` 满足 qd_entry 且空仓 → 应于次日开仓。"
                )
        if not near_rows:
            lines.append("_窗口内无重叠持仓成交。_")
        else:
            lines.append("| signal | entry | exit | ret | hold | reason |")
            lines.append("|---|---|---|---:|---:|---|")
            for t in near_rows:
                lines.append(
                    f"| {t['signal_date']} | {t['entry_date']} | {t['exit_date']} | "
                    f"{float(t['ret']):.2%} | {int(t['hold_days'])} | {t['exit_reason']} |"
                )

    if not trades_df.empty:
        oos_cut = pd.Timestamp(oos_start).normalize()
        oos_tr = trades_df[pd.to_datetime(trades_df["entry_date"]) >= oos_cut]
        show = oos_tr if not oos_tr.empty else trades_df
        lines.extend(["", "## OOS 成交明细（或全样本若 OOS 为空）", ""])
        lines.append("| signal | entry | exit | ret | hold | reason |")
        lines.append("|---|---|---|---:|---:|---|")
        for _, t in show.iterrows():
            lines.append(
                f"| {t['signal_date']} | {t['entry_date']} | {t['exit_date']} | "
                f"{float(t['ret']):.2%} | {int(t['hold_days'])} | {t['exit_reason']} |"
            )

    lines.extend(
        [
            "",
            "## 规则备忘",
            "",
            "- 买入：长尺折价 + 双 g>0 + 低波 + gap20≤0 + 短尺未触上轨",
            "- 卖出：S1 短尺上轨 / fig7 回到 path / 双 g<0 / ATR 止损 / max_hold",
            "",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    html_path = _resolve_html(args)
    code = str(args.code).upper()
    listing_dir = Path(args.listing_dir)
    if not listing_dir.is_absolute():
        listing_dir = _PROJECT_ROOT / listing_dir

    close, ohlcv = load_ohlcv_from_regime_html(html_path)
    anchor_html = find_regime_html(listing_dir, str(args.anchor_code).upper())
    if anchor_html is None:
        raise FileNotFoundError(
            f"anchor HTML for {args.anchor_code} not found under {listing_dir}"
        )
    close_a, _ = load_ohlcv_from_regime_html(anchor_html)
    vf_a = compute_volatility_regime_frame(close_a)
    vf_a = vf_a.copy()
    vf_a["as_of"] = pd.to_datetime(vf_a["as_of"]).dt.normalize()
    vf_a = vf_a.set_index("as_of")
    rv20_anchor = pd.to_numeric(vf_a["rv20"], errors="coerce")
    rv5_anchor = pd.to_numeric(vf_a["rv5"], errors="coerce")

    frame = build_quiet_discount_frame(
        close,
        ohlcv,
        rv20_anchor=rv20_anchor,
        rv5_anchor=rv5_anchor,
        fig7_gap_max=float(args.fig7_gap_max),
        vol5_pct_max=float(args.vol5_pct_max),
    )
    entries = quiet_discount_entries(frame)
    trades = simulate_quiet_discount_trades(
        frame,
        ohlcv,
        entries=entries,
        max_hold=int(args.max_hold),
    )
    trades_df = pd.DataFrame(trades)
    eq_df = _build_equity(frame["px"], trades)
    oos_cut = pd.Timestamp(args.oos_start).normalize()

    if trades_df.empty:
        oos_trades = trades_df
    else:
        oos_trades = trades_df[
            pd.to_datetime(trades_df["entry_date"]) >= oos_cut
        ].reset_index(drop=True)

    # OOS equity: restart compounding from oos_cut using only OOS trades
    oos_eq = _build_equity(
        frame["px"].loc[frame.index >= oos_cut],
        oos_trades.to_dict("records") if not oos_trades.empty else [],
    )

    full_sum = _summarize(trades_df, eq_df["equity"], label="full")
    oos_sum = _summarize(oos_trades, oos_eq["equity"], label="oos")
    check_rows = _check_window_rows(frame, CHECK_DATES)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    trades_df.to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")
    eq_df.to_csv(out_dir / "equity.csv", encoding="utf-8-sig")
    summary = {
        "code": code,
        "html": str(html_path),
        "anchor_code": args.anchor_code,
        "oos_start": args.oos_start,
        "fig7_gap_max": args.fig7_gap_max,
        "vol5_pct_max": args.vol5_pct_max,
        "max_hold": args.max_hold,
        "full": full_sum,
        "oos": oos_sum,
        "check_20260804_05": check_rows,
        "n_entry_signals_full": int(entries.sum()),
        "n_entry_signals_oos": int(
            entries.loc[entries.index >= oos_cut].sum()
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(
        out_dir,
        code=code,
        html_path=html_path,
        oos_start=args.oos_start,
        full_sum=full_sum,
        oos_sum=oos_sum,
        trades_df=trades_df,
        check_rows=check_rows,
        args=args,
    )

    print(
        f"[OK] quiet_discount {code} n_trades={len(trades_df)} "
        f"oos={len(oos_trades)} out={out_dir}"
    )
    for row in check_rows:
        if row.get("present"):
            print(f"  check {row['date']}: qd_entry={row['qd_entry']}")
    if oos_sum.get("win_rate") is not None:
        print(
            f"  oos win_rate={oos_sum['win_rate']:.1%} "
            f"mean_ret={oos_sum['mean_ret']:.2%} "
            f"compound={oos_sum['compound']:.2%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
