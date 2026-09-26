#!/usr/bin/env python3
"""Replay per-symbol early warnings across ablation dates.

After --all-holdings, compare strategies vs buy-and-hold with:
  compare_replay_strategies.py --replay-dir <out> --live-holdings <csv>
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

from decision_pack.src.config import load_decision_pack_config
from decision_pack.src.early_warning import classify_symbol_alert
from decision_pack.src.inputs import resolve_run_inputs
from decision_pack.src.portfolio import load_portfolio_snapshot
from decision_pack.src.symbol_trend_board import build_trend_board, defensive_codes
from runtime_paths import FUND_LIST_CSV
from workspace.scripts.generate_daily_mu_position_report import load_fund_name_map, normalize_code


def _iso_from_compact(compact: str) -> str:
    return f"{compact[:4]}-{compact[4:6]}-{compact[6:8]}"


def _discover_ablation_dates(temp_root: Path, start: str, end: str) -> list[str]:
    dates: list[str] = []
    for path in sorted(temp_root.glob("trend_sleeve_ablation_*")):
        compact = path.name.replace("trend_sleeve_ablation_", "")
        if len(compact) != 8 or not compact.isdigit():
            continue
        if start <= compact <= end:
            dates.append(compact)
    return dates


def _run_subdir(ablation_dir: Path, subdir: str) -> Path:
    candidate = ablation_dir / subdir
    if not candidate.is_dir():
        raise FileNotFoundError(f"Missing run subdir: {candidate}")
    return candidate


def _read_live_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def load_holdings_name_map(live_holdings: Path) -> dict[str, str]:
    """仅 live CSV 中的持仓 code -> 中文名（fund_list 作补全）。"""
    path = live_holdings.expanduser().resolve()
    frame = _read_live_csv(path)
    if "code" not in frame.columns:
        raise ValueError(f"live holdings missing code column: {path}")
    name_col = "name" if "name" in frame.columns else None
    fund_map = load_fund_name_map(FUND_LIST_CSV) if FUND_LIST_CSV.exists() else {}
    names: dict[str, str] = {}
    for _, row in frame.iterrows():
        code = normalize_code(row["code"])
        if code == "CASH":
            continue
        raw_name = str(row[name_col]).strip() if name_col and pd.notna(row.get(name_col)) else ""
        names[code] = raw_name or fund_map.get(code, code)
    return names


def list_replay_symbols(live_holdings: Path, *, config=None) -> tuple[str, ...]:
    cfg = config or load_decision_pack_config()
    skip = {normalize_code(code) for code in defensive_codes(cfg)}
    names = load_holdings_name_map(live_holdings)
    return tuple(sorted(code for code in names.keys() if code not in skip))


def _resolve_symbol_name(code: str, name_map: dict[str, str]) -> str:
    return name_map.get(normalize_code(code), code)


def replay_md_filename(code: str, name: str) -> str:
    """{code}_{name}_replay.md — name 来自 live holdings 中文名。"""
    safe_name = str(name or code).strip()
    for ch in ("/", "\\", ":", "*", "?", '"', "<", ">", "|"):
        safe_name = safe_name.replace(ch, "_")
    safe_name = safe_name or code
    return f"{normalize_code(code)}_{safe_name}_replay.md"


def _empty_replay_row(iso: str, code: str) -> dict[str, Any]:
    return {
        "date": iso,
        "code": code,
        "alert_level": "missing",
        "reasons": "",
        "suggest_reduce_pct": 0.0,
        "pnl_pct": None,
        "dd_20d_high": None,
        "dist_ma20_pct": None,
        "risk_flags": "",
        "pipeline_weight": None,
        "weight": None,
    }


def _alert_row_from_board_row(
    iso: str,
    code: str,
    row: pd.Series,
    *,
    pipeline_weight: float,
    config,
) -> dict[str, Any]:
    alert = classify_symbol_alert(row, pipeline_weight=pipeline_weight, config=config)
    return {
        "date": iso,
        "code": code,
        "alert_level": alert.level,
        "reasons": "|".join(alert.reasons),
        "suggest_reduce_pct": alert.suggest_reduce_pct,
        "pnl_pct": row.get("pnl_pct"),
        "dd_20d_high": row.get("dd_20d_high"),
        "dist_ma20_pct": row.get("dist_ma20_pct"),
        "risk_flags": row.get("risk_flags"),
        "pipeline_weight": pipeline_weight,
        "weight": row.get("weight"),
    }


def replay_holdings(
    *,
    symbols: tuple[str, ...],
    temp_root: Path,
    live_holdings: Path,
    start: str,
    end: str,
    run_subdir: str,
    preferred_start: str,
) -> dict[str, pd.DataFrame]:
    cfg = load_decision_pack_config()
    fund_list = FUND_LIST_CSV if FUND_LIST_CSV.exists() else None
    codes = tuple(normalize_code(symbol) for symbol in symbols)
    rows_by_code: dict[str, list[dict[str, Any]]] = {code: [] for code in codes}

    for compact in _discover_ablation_dates(temp_root, start, end):
        iso = _iso_from_compact(compact)
        ablation_dir = temp_root / f"trend_sleeve_ablation_{compact}"
        run_dir = _run_subdir(ablation_dir, run_subdir)
        inputs = resolve_run_inputs(
            run_dir,
            iso,
            preferred_start=preferred_start,
            ensure_audit=False,
        )
        portfolio = load_portfolio_snapshot(
            as_of=iso,
            live_holdings_path=live_holdings,
            fund_list_csv=fund_list,
            config=cfg,
        )
        pipeline = load_portfolio_snapshot(
            as_of=iso,
            pipeline_weights_csv=inputs.paths.portfolio_weights_csv,
            fund_list_csv=fund_list,
        )
        trend = build_trend_board(portfolio, config=cfg)
        board = trend.board
        board = board.copy()
        board["_code_norm"] = board["code"].astype(str).map(normalize_code)
        pipeline_weights = {
            normalize_code(h.code): h.weight_total for h in pipeline.holdings
        }

        for code in codes:
            match = board.loc[board["_code_norm"] == code]
            if match.empty:
                rows_by_code[code].append(_empty_replay_row(iso, code))
                continue
            row = match.iloc[0]
            rows_by_code[code].append(
                _alert_row_from_board_row(
                    iso,
                    code,
                    row,
                    pipeline_weight=pipeline_weights.get(code),
                    config=cfg,
                )
            )

    return {code: pd.DataFrame(rows) for code, rows in rows_by_code.items()}


def replay_symbol(
    *,
    symbol: str,
    temp_root: Path,
    live_holdings: Path,
    start: str,
    end: str,
    run_subdir: str,
    output_dir: Path | None,
    preferred_start: str,
) -> pd.DataFrame:
    code = normalize_code(symbol)
    frames = replay_holdings(
        symbols=(code,),
        temp_root=temp_root,
        live_holdings=live_holdings,
        start=start,
        end=end,
        run_subdir=run_subdir,
        preferred_start=preferred_start,
    )
    frame = frames[code]
    if output_dir is not None:
        name_map = load_holdings_name_map(live_holdings)
        write_symbol_replay_outputs(
            output_dir,
            code,
            _resolve_symbol_name(code, name_map),
            frame,
            start=start,
            end=end,
        )
    return frame


def _first_date(frame: pd.DataFrame, *, column: str, op: str, threshold: float) -> str | None:
    for _, row in frame.iterrows():
        val = row.get(column)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        if op == "lte" and float(val) <= threshold:
            return str(row["date"])
        if op == "gte" and float(val) >= threshold:
            return str(row["date"])
    return None


def _first_alert(frame: pd.DataFrame, level: str) -> str | None:
    hits = frame.loc[frame["alert_level"] == level, "date"]
    if hits.empty:
        return None
    return str(hits.iloc[0])


def _fmt_bool(value: bool | None) -> str:
    if value is None:
        return "—"
    return "是" if value else "否"


def _fmt_num(value: Any, *, pct: bool = False, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    number = float(value)
    if pct:
        return f"{number * 100:.{digits}f}%"
    return f"{number:.{digits}f}"


def summarize_replay(
    frame: pd.DataFrame,
    *,
    pnl_threshold: float = -7.0,
    dd_threshold: float = -0.09,
) -> dict[str, Any]:
    first_yellow = _first_alert(frame, "yellow")
    first_red = _first_alert(frame, "red")
    first_pnl7 = _first_date(frame, column="pnl_pct", op="lte", threshold=pnl_threshold)
    first_dd9 = _first_date(frame, column="dd_20d_high", op="lte", threshold=dd_threshold)

    def _before(alert_date: str | None, ref_date: str | None) -> bool | None:
        if alert_date is None or ref_date is None:
            return None
        return alert_date < ref_date

    return {
        "first_yellow": first_yellow,
        "first_red": first_red,
        "first_pnl_lte_7pct": first_pnl7,
        "first_dd_20d_high_lte_9pct": first_dd9,
        "yellow_before_pnl7": _before(first_yellow, first_pnl7),
        "red_before_pnl7": _before(first_red, first_pnl7),
        "yellow_before_dd9": _before(first_yellow, first_dd9),
        "red_before_dd9": _before(first_red, first_dd9),
        "rows": len(frame),
    }


def render_symbol_replay_report_md(
    *,
    code: str,
    name: str,
    frame: pd.DataFrame,
    summary: dict[str, Any],
    start: str,
    end: str,
) -> str:
    title_name = name if name and name != code else code
    lines = [
        f"# {code} {title_name} · 预警回放",
        "",
        f"- 回放区间：{_iso_from_compact(start)} ~ {_iso_from_compact(end)}",
        f"- 交易日数：{summary.get('rows', len(frame))}",
        "",
        "## 结论摘要",
        "",
        "| 指标 | 日期 |",
        "|------|------|",
        f"| 首次黄灯 | {summary.get('first_yellow') or '—'} |",
        f"| 首次红灯 | {summary.get('first_red') or '—'} |",
        f"| 浮亏 ≤ -7% | {summary.get('first_pnl_lte_7pct') or '—'} |",
        f"| 20日高点回撤 ≤ -9% | {summary.get('first_dd_20d_high_lte_9pct') or '—'} |",
        "",
        "| 对比 | 结果 |",
        "|------|------|",
        f"| 黄灯早于 -7% 浮亏 | {_fmt_bool(summary.get('yellow_before_pnl7'))} |",
        f"| 红灯早于 -7% 浮亏 | {_fmt_bool(summary.get('red_before_pnl7'))} |",
        f"| 黄灯早于 -9% 回撤 | {_fmt_bool(summary.get('yellow_before_dd9'))} |",
        f"| 红灯早于 -9% 回撤 | {_fmt_bool(summary.get('red_before_dd9'))} |",
        "",
        "## 逐日时间线",
        "",
        "| 日期 | 预警 | 建议减仓 | pnl% | dd_20d | dist_ma20% | 原因 |",
        "|------|------|----------|------|--------|------------|------|",
    ]

    if frame.empty:
        lines.append("| — | — | — | — | — | — | — |")
    else:
        for _, row in frame.iterrows():
            level = str(row.get("alert_level") or "—")
            if level == "red":
                level = "🔴 red"
            elif level == "yellow":
                level = "🟡 yellow"
            elif level == "green":
                level = "green"
            suggest = row.get("suggest_reduce_pct")
            suggest_s = (
                "—"
                if suggest is None or pd.isna(suggest) or float(suggest) <= 0
                else f"{float(suggest) * 100:.0f}%"
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("date") or "—"),
                        level,
                        suggest_s,
                        _fmt_num(row.get("pnl_pct")),
                        _fmt_num(row.get("dd_20d_high"), pct=True),
                        _fmt_num(row.get("dist_ma20_pct")),
                        str(row.get("reasons") or "—"),
                    ]
                )
                + " |"
            )

    lines.append("")
    return "\n".join(lines)


def render_replay_index_md(
    entries: list[tuple[str, str, dict[str, Any]]],
    *,
    start: str,
    end: str,
    output_dir: Path,
) -> str:
    lines = [
        "# 持仓标的预警回放索引",
        "",
        f"- 回放区间：{_iso_from_compact(start)} ~ {_iso_from_compact(end)}",
        f"- 标的数量：{len(entries)}",
        f"- 输出目录：`{output_dir}`",
        "",
        "| code | 中文名 | 首次黄灯 | 首次红灯 | 黄灯早于-7% | 红灯早于-7% | 报告 |",
        "|------|--------|----------|----------|-------------|-------------|------|",
    ]
    for code, name, summary in sorted(entries, key=lambda item: item[0]):
        md_name = replay_md_filename(code, name)
        lines.append(
            f"| {code} | {name} | {summary.get('first_yellow') or '—'} | "
            f"{summary.get('first_red') or '—'} | "
            f"{_fmt_bool(summary.get('yellow_before_pnl7'))} | "
            f"{_fmt_bool(summary.get('red_before_pnl7'))} | "
            f"[{code}]({md_name}) |"
        )
    lines.append("")
    return "\n".join(lines)


def write_symbol_replay_outputs(
    output_dir: Path,
    code: str,
    name: str,
    frame: pd.DataFrame,
    *,
    start: str,
    end: str,
    pnl_threshold: float = -7.0,
    dd_threshold: float = -0.09,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_replay(frame, pnl_threshold=pnl_threshold, dd_threshold=dd_threshold)
    written = {
        "csv": output_dir / f"{code}_replay.csv",
        "json": output_dir / f"{code}_replay_summary.json",
        "md": output_dir / replay_md_filename(code, name),
    }
    frame.to_csv(written["csv"], index=False)
    written["json"].write_text(
        json.dumps({**summary, "code": code, "name": name}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    written["md"].write_text(
        render_symbol_replay_report_md(
            code=code,
            name=name,
            frame=frame,
            summary=summary,
            start=start,
            end=end,
        ),
        encoding="utf-8",
    )
    legacy_md = output_dir / f"{code}_replay.md"
    if legacy_md.exists() and legacy_md != written["md"]:
        legacy_md.unlink()
    return written


def replay_all_holdings(
    *,
    temp_root: Path,
    live_holdings: Path,
    start: str,
    end: str,
    run_subdir: str,
    output_dir: Path,
    preferred_start: str,
) -> dict[str, pd.DataFrame]:
    name_map = load_holdings_name_map(live_holdings)
    symbols = list_replay_symbols(live_holdings)
    frames = replay_holdings(
        symbols=symbols,
        temp_root=temp_root,
        live_holdings=live_holdings,
        start=start,
        end=end,
        run_subdir=run_subdir,
        preferred_start=preferred_start,
    )
    index_entries: list[tuple[str, str, dict[str, Any]]] = []
    for code, frame in frames.items():
        name = _resolve_symbol_name(code, name_map)
        write_symbol_replay_outputs(
            output_dir,
            code,
            name,
            frame,
            start=start,
            end=end,
        )
        index_entries.append((code, name, summarize_replay(frame)))

    index_path = output_dir / "REPLAY_INDEX.md"
    index_path.write_text(
        render_replay_index_md(index_entries, start=start, end=end, output_dir=output_dir),
        encoding="utf-8",
    )
    return frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay symbol early warnings across ablation dates")
    parser.add_argument("--symbol", default=None, help="Single ETF code; omit with --all-holdings")
    parser.add_argument(
        "--all-holdings",
        action="store_true",
        help="Replay all equity symbols in live-holdings CSV and write MD reports",
    )
    parser.add_argument("--temp-root", type=Path, default=Path.home() / "temp")
    parser.add_argument("--live-holdings", type=Path, required=True)
    parser.add_argument("--start", default="20260529", help="Compact start date YYYYMMDD")
    parser.add_argument("--end", default="20260610", help="Compact end date YYYYMMDD")
    parser.add_argument(
        "--run-subdir",
        default="2025_regime_switch_ewma_shrink_no_sleeve",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--preferred-start", default="20260401")
    args = parser.parse_args(argv)

    live_holdings = args.live_holdings.expanduser().resolve()
    temp_root = args.temp_root.expanduser().resolve()

    if args.all_holdings:
        out = args.output or (temp_root / "decision_packs_replay" / "all_holdings")
        frames = replay_all_holdings(
            temp_root=temp_root,
            live_holdings=live_holdings,
            start=args.start,
            end=args.end,
            run_subdir=args.run_subdir,
            output_dir=out,
            preferred_start=args.preferred_start,
        )
        print(f"[OK] replay reports -> {out}")
        print(f"     symbols={len(frames)} index=REPLAY_INDEX.md")
        return 0

    symbol = args.symbol or "SH561560"
    out = args.output or (temp_root / "decision_packs_replay" / symbol)
    frame = replay_symbol(
        symbol=symbol,
        temp_root=temp_root,
        live_holdings=live_holdings,
        start=args.start,
        end=args.end,
        run_subdir=args.run_subdir,
        output_dir=out,
        preferred_start=args.preferred_start,
    )
    summary = summarize_replay(frame)
    code = normalize_code(symbol)
    name = _resolve_symbol_name(code, load_holdings_name_map(live_holdings))
    md_path = out / replay_md_filename(code, name)
    print(json.dumps({**summary, "md": str(md_path)}, indent=2, ensure_ascii=False))
    if not frame.empty:
        print("\nTimeline:")
        print(
            frame[
                [
                    "date",
                    "alert_level",
                    "pnl_pct",
                    "dd_20d_high",
                    "dist_ma20_pct",
                    "reasons",
                ]
            ].to_string(index=False)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
