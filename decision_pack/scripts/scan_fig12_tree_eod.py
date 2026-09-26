#!/usr/bin/env python3
"""EOD snapshot: fig12 state × TREE_POS action + fig1 markers + fig7–11 g/percentile.

Reads from_listing adaptive HTML. Morphology comes from per_etf.csv (locked).
Position table is TREE_POS in backtest_decision_tree_winrate.py
(documents/图12决策树回测_形态x状态胜率.md).

T-close state → overnight position for T+1. Buy/sell mark = change vs
yesterday's TREE_POS (yesterday state). ①过热 stays full; the winrate
doc records 「不追」 as 持有.

Example::

    PYTHONPATH=. python decision_pack/scripts/scan_fig12_tree_eod.py \\
      --html-dir ~/etf-daily-output/temp/plotly_outputs/20260915_from_listing \\
      --morph-csv ~/etf-daily-output/temp/plotly_outputs/20260915_from_listing/fig12_six_state_backtest/per_etf.csv \\
      --as-of 2026-09-15 \\
      --write-md documents/图12决策树_20260915收盘快照.md
"""
from __future__ import annotations

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import backtest_decision_tree_winrate as tree  # noqa: E402
import backtest_fig12_six_states_pool as pool  # noqa: E402
from scan_fig9_aux_edge_eod import fig1_markers_on_asof  # noqa: E402
from decision_pack.src.fair_path_band_signals import (  # noqa: E402
    FIG_PANELS,
    _load_plot_mod,
    compute_multi_scale_frame,
    enrich_frame_for_checklist,
    load_ohlcv_from_regime_html,
)
from decision_pack.src.six_states import (  # noqa: E402
    STATE_CHOP,
    STATE_DOWN,
    STATE_KNIFE,
    STATE_ORDER,
    STATE_OVERHEAT,
    STATE_REPAIR,
    STATE_UP,
    STATE_WARMUP,
    classify_six_states,
)

COMMS_CODE = "SH515880"
CROSS_VETO_CODES = {"SH588850", "SH588170"}  # 科创机械 / 科创半导体
CROSS_VETO_STATES = {STATE_OVERHEAT, STATE_UP}

POS_CN = pool.ACTION_CN
EPS = 1e-9


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--html-dir", type=Path, required=True)
    p.add_argument("--morph-csv", type=Path, required=True)
    p.add_argument("--as-of", type=str, default=None)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--write-md", type=Path, default=None)
    p.add_argument("--write-csv", type=Path, default=None)
    p.add_argument(
        "--include-bonds",
        action="store_true",
        help="also scan bond ETFs at html-dir root (no TREE_POS)",
    )
    return p.parse_args(argv)


def _fmt_g(v: object) -> str:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        x = float(v)
        if not np.isfinite(x):
            return "—"
        return f"{x:+.1%}"
    except (TypeError, ValueError):
        return "—"


def _fmt_pct(v: object) -> str:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        x = float(v)
        if not np.isfinite(x):
            return "—"
        if x == 0.0:
            return "P0.0"
        sign = "+" if x > 0 else "-"
        return f"{sign}P{abs(x) * 100.0:.1f}"
    except (TypeError, ValueError):
        return "—"


def _pos_cn(pos: float | None) -> str:
    if pos is None or not np.isfinite(pos):
        return "—"
    for k, lab in POS_CN.items():
        if abs(float(pos) - float(k)) < EPS:
            return lab
    return f"{float(pos):.0%}"


def trade_mark(
    prev_pos: float | None,
    pos: float | None,
    state: str,
) -> str:
    if pos is None or not np.isfinite(pos):
        return "未入决策树"
    p = float(pos)
    if prev_pos is None or not np.isfinite(prev_pos):
        if p > EPS:
            return f"建仓·{_pos_cn(p)}"
        return "空仓待命"
    prev = float(prev_pos)
    if p > prev + EPS:
        if prev <= EPS and abs(p - 0.5) < EPS:
            return "买入·半仓"
        if prev <= EPS:
            return "买入·满仓"
        if abs(prev - 0.5) < EPS and abs(p - 1.0) < EPS:
            return "加仓·满仓"
        return f"买入·{_pos_cn(p)}"
    if p < prev - EPS:
        if abs(p) <= EPS:
            return "卖出·空仓"
        if abs(p - 0.5) < EPS:
            return "减仓·半仓"
        return f"卖出·{_pos_cn(p)}"
    if abs(p) <= EPS:
        return "保持空仓"
    if abs(p - 0.5) < EPS:
        return "持有半仓"
    if state == STATE_OVERHEAT:
        return "持有（不追）"
    return "持有"


def _is_actionable_mark(mark: str) -> bool:
    return any(k in mark for k in ("买入", "卖出", "加仓", "减仓", "建仓"))


def list_html(html_dir: Path, *, include_bonds: bool) -> list[Path]:
    html_dir = Path(html_dir)
    if include_bonds:
        out: list[Path] = []
        for p in sorted(html_dir.glob("regime_transition_*_adaptive.html")):
            if pool._FNAME_RE.match(p.name):
                out.append(p)
        return out
    return pool.list_equity_html(html_dir)


def scan_one(args: tuple[str, str, str, str, str | None]) -> dict[str, Any]:
    html_s, code, name, morph, asof_s = args
    html = Path(html_s)
    row: dict[str, Any] = {
        "ok": False,
        "code": code,
        "name": name,
        "morph": morph or "未分类",
        "path": str(html),
        "asof": asof_s or "",
        "last_date": "",
        "px": float("nan"),
        "state": "",
        "prev_state": "",
        "pos": float("nan"),
        "prev_pos": float("nan"),
        "pos_cn": "—",
        "prev_pos_cn": "—",
        "mark": "未入决策树",
        "fig1": "无",
        "error": "",
        "bond": pool.is_bond(code, name),
    }
    for key, _, _ in FIG_PANELS:
        row[f"{key}_g"] = float("nan")
        row[f"{key}_pct"] = float("nan")
        row[f"{key}_txt"] = "—"
    try:
        close, _ohlcv = load_ohlcv_from_regime_html(html)
        if close.empty:
            row["error"] = "empty close"
            return row
        last = pd.Timestamp(close.index[-1]).normalize()
        row["last_date"] = last.strftime("%Y-%m-%d")
        asof = pd.Timestamp(asof_s).normalize() if asof_s else last
        row["asof"] = asof.strftime("%Y-%m-%d")
        if last != asof:
            if asof not in close.index:
                row["error"] = f"last={row['last_date']} missing {asof.date()}"
                return row
        ms = enrich_frame_for_checklist(compute_multi_scale_frame(close))
        states = classify_six_states(ms)
        if asof not in ms.index:
            row["error"] = f"as-of {asof.date()} not in frame"
            return row
        i = ms.index.get_loc(asof)
        if isinstance(i, slice):
            i = i.stop - 1
        elif isinstance(i, (np.ndarray, pd.Index)):
            i = int(np.asarray(i).flat[-1])
        i = int(i)
        rec = ms.iloc[i]
        row["px"] = float(rec["px"]) if pd.notna(rec.get("px")) else float("nan")
        state = str(states.iloc[i]) if i < len(states) else STATE_WARMUP
        prev_state = (
            str(states.iloc[i - 1]) if i >= 1 else ""
        )
        row["state"] = state
        row["prev_state"] = prev_state

        pm = _load_plot_mod()
        close_s = pd.to_numeric(ms["px"], errors="coerce")
        for key, years, _lab in FIG_PANELS:
            g = rec.get(f"{key}_g")
            row[f"{key}_g"] = float(g) if pd.notna(g) else float("nan")
            path = pd.to_numeric(ms[f"{key}_path"], errors="coerce")
            signed = pm.causal_path_residual_signed_percentile(
                path,
                close_s,
                min_bars=pm._trail_fit_min_bars(float(years)),
            )
            pv = signed.iloc[i] if i < len(signed) else float("nan")
            row[f"{key}_pct"] = float(pv) if pd.notna(pv) else float("nan")
            row[f"{key}_txt"] = f"{_fmt_g(row[f'{key}_g'])} / {_fmt_pct(row[f'{key}_pct'])}"

        morph_key = morph if morph in tree.TREE_POS else ""
        pos: float | None = None
        prev_pos: float | None = None
        if morph_key and state in tree.TREE_POS[morph_key]:
            pos = float(tree.TREE_POS[morph_key][state])
        if morph_key and prev_state in tree.TREE_POS[morph_key]:
            prev_pos = float(tree.TREE_POS[morph_key][prev_state])
        row["pos"] = float(pos) if pos is not None else float("nan")
        row["prev_pos"] = float(prev_pos) if prev_pos is not None else float("nan")
        row["pos_cn"] = _pos_cn(pos)
        row["prev_pos_cn"] = _pos_cn(prev_pos)
        row["mark"] = trade_mark(prev_pos, pos, state)

        try:
            markers = fig1_markers_on_asof(html, asof)
            row["fig1"] = "、".join(markers) if markers else "无"
        except Exception as exc:
            row["fig1"] = f"解析失败: {type(exc).__name__}: {exc}"

        row["ok"] = True
        return row
    except Exception:
        row["error"] = traceback.format_exc(limit=4)
        return row


def _md_cell(s: object) -> str:
    return str(s or "").replace("|", "/")


def _state_flow(prev_state: str, state: str) -> str:
    if not prev_state:
        return state
    if prev_state == state:
        return state
    return f"{prev_state}→{state}"


def apply_cross_veto_notes(rows: list[dict[str, Any]]) -> None:
    comms = next((r for r in rows if r.get("code") == COMMS_CODE and r.get("ok")), None)
    if comms is None or comms.get("state") != STATE_KNIFE:
        for r in rows:
            r["note"] = r.get("note") or ""
        return
    for r in rows:
        notes: list[str] = []
        if r.get("code") in CROSS_VETO_CODES and r.get("state") in CROSS_VETO_STATES:
            notes.append("交叉否决：通信ETF处于②，①/⑤不追（六状态文档）")
        r["note"] = "；".join(notes)


def render_md(
    rows: list[dict[str, Any]],
    *,
    asof: str,
    html_dir: Path,
    morph_csv: Path,
) -> str:
    ok_rows = [r for r in rows if r.get("ok")]
    fail_rows = [r for r in rows if not r.get("ok")]
    n = len(ok_rows)
    asof_ok = sum(1 for r in ok_rows if r.get("last_date") == asof)
    actionable = [r for r in ok_rows if _is_actionable_mark(str(r.get("mark") or ""))]
    fig1_new = [r for r in ok_rows if str(r.get("fig1") or "无") not in ("无", "")]

    by_morph: dict[str, list[dict[str, Any]]] = {m: [] for m in pool.MORPH_ORDER}
    other: list[dict[str, Any]] = []
    for r in ok_rows:
        m = str(r.get("morph") or "")
        if m in by_morph:
            by_morph[m].append(r)
        else:
            other.append(r)

    by_state: dict[str, int] = {s: 0 for s in STATE_ORDER}
    by_state[STATE_WARMUP] = 0
    for r in ok_rows:
        st = str(r.get("state") or STATE_WARMUP)
        by_state[st] = by_state.get(st, 0) + 1

    lines: list[str] = [
        f"# 图12 决策树收盘快照（{asof}）",
        "",
        f"> 数据：`{html_dir}` 全部 adaptive HTML（截止 {asof}；债券不入决策树）",
        f"> 形态：`{morph_csv}` 的 `morph` 列（与胜率文档锁定同一标签）",
        "> 决策： [图12决策树回测_形态x状态胜率.md](图12决策树回测_形态x状态胜率.md) 的 TREE_POS；"
        "T 收盘状态 → 隔夜仓位；买卖标注 = 相对昨收状态仓位的变化。"
        "①「不追」在满仓基准下记为持有。",
        f"> 复现：`PYTHONPATH=. python decision_pack/scripts/scan_fig12_tree_eod.py "
        f"--html-dir {html_dir} --morph-csv {morph_csv} --as-of {asof} "
        f"--write-md documents/图12决策树_{asof.replace('-', '')}收盘快照.md`",
        "",
        "## 0. 口径",
        "",
        "- 图12 状态：因果 `classify_six_states`，与 HTML 图12 同一套规则。",
        "- 图1 新标识：当日 fig1 marker（不含「持仓未平」）；含 ED 告警/观察结束。",
        "- 图7–11：路径年化 g 与相对路径的因果有符号分位（`+P72.3` = 在路径上方、"
        "|ln(C/P)| 处于历史 P72.3；负号 = 在路径下方）。与 HTML hover 同一函数。",
        "- 仓位表：",
        "",
        "| 状态 | 深熊阴跌型 | V型急跌快修复型 | 高波动成长型 | 平稳慢牛型 |",
        "|---|---|---|---|---|",
        "| ①过热 | 满仓 | 满仓 | 满仓 | 满仓 |",
        "| ②急跌刀锋 | 空仓 | 半仓 | 空仓 | 满仓 |",
        "| ③趋势下行 | 空仓 | 空仓 | 满仓 | 满仓 |",
        "| ④底部修复 | 满仓 | 满仓 | 满仓 | 满仓 |",
        "| ⑤趋势上行 | 满仓 | 满仓 | 满仓 | 满仓 |",
        "| ⑥震荡 | 满仓 | 满仓 | 满仓 | 满仓 |",
        "",
        f"扫描 **{n}** 只；末日日期对齐 {asof} 的 **{asof_ok}** 只。"
        f"仓位变化（买入/卖出/加仓/减仓）**{len(actionable)}** 只；"
        f"图1当日有新标识 **{len(fig1_new)}** 只。",
        "",
        "## 1. 状态分布",
        "",
        "| 状态 | 只数 |",
        "|---|---|",
    ]
    for st in list(STATE_ORDER) + [STATE_WARMUP]:
        if by_state.get(st, 0):
            lines.append(f"| {st} | {by_state[st]} |")
    extra_states = [
        k for k in by_state if k not in STATE_ORDER and k != STATE_WARMUP and by_state[k]
    ]
    for st in extra_states:
        lines.append(f"| {st} | {by_state[st]} |")
    lines.extend(["", "按形态 × 状态：", "", "| 形态 | " + " | ".join(STATE_ORDER) + " | 合计 |", "|---|" + "---|" * len(STATE_ORDER) + "---|"])
    for morph in pool.MORPH_ORDER:
        grp = by_morph[morph]
        counts = [sum(1 for r in grp if r.get("state") == st) for st in STATE_ORDER]
        lines.append(
            "| "
            + morph
            + " | "
            + " | ".join(str(c) if c else "—" for c in counts)
            + f" | {len(grp)} |"
        )
    lines.append("")

    lines.extend(
        [
            "## 2. 买卖标注（仓位变化）",
            "",
        ]
    )
    if not actionable:
        lines.append("本日无仓位变化（全池相对昨收均为持有 / 保持空仓 / 持有半仓 / 不追）。")
        lines.append("")
    else:
        lines.extend(
            [
                "| 代码 | 名称 | 形态 | 状态变化 | 昨仓 | 今仓 | 买卖标注 | 图1新标识 |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        actionable_sorted = sorted(
            actionable,
            key=lambda r: (
                0 if "卖出" in str(r.get("mark")) or "减仓" in str(r.get("mark")) else 1,
                str(r.get("morph") or ""),
                str(r.get("code") or ""),
            ),
        )
        for r in actionable_sorted:
            lines.append(
                "| {code} | {name} | {morph} | {flow} | {prev} | {pos} | **{mark}** | {fig1} |".format(
                    code=_md_cell(r.get("code")),
                    name=_md_cell(r.get("name")),
                    morph=_md_cell(r.get("morph")),
                    flow=_md_cell(_state_flow(str(r.get("prev_state") or ""), str(r.get("state") or ""))),
                    prev=_md_cell(r.get("prev_pos_cn")),
                    pos=_md_cell(r.get("pos_cn")),
                    mark=_md_cell(r.get("mark")),
                    fig1=_md_cell(r.get("fig1")),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## 3. 图1当日新标识",
            "",
        ]
    )
    if not fig1_new:
        lines.append("本日全池图1无新标识。")
        lines.append("")
    else:
        lines.extend(
            [
                "| 代码 | 名称 | 形态 | 图12状态 | 买卖标注 | 图1新标识 |",
                "|---|---|---|---|---|---|",
            ]
        )
        for r in sorted(fig1_new, key=lambda x: str(x.get("code") or "")):
            lines.append(
                "| {code} | {name} | {morph} | {state} | {mark} | {fig1} |".format(
                    code=_md_cell(r.get("code")),
                    name=_md_cell(r.get("name")),
                    morph=_md_cell(r.get("morph")),
                    state=_md_cell(r.get("state")),
                    mark=_md_cell(r.get("mark")),
                    fig1=_md_cell(r.get("fig1")),
                )
            )
        lines.append("")

    lines.extend(["## 4. 逐只明细", ""])
    detail_header = (
        "| 代码 | 名称 | 收盘 | 图12状态 | 昨→今仓 | 买卖标注 | 图1新标识 |"
        " 图7 g/分位 | 图8 g/分位 | 图9 g/分位 | 图10 g/分位 | 图11 g/分位 | 备注 |"
    )
    detail_sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|"

    def _detail_row(r: dict[str, Any]) -> str:
        px = r.get("px")
        try:
            px_s = f"{float(px):.4f}" if px is not None and np.isfinite(float(px)) else "—"
        except (TypeError, ValueError):
            px_s = "—"
        flow_pos = f"{r.get('prev_pos_cn') or '—'}→{r.get('pos_cn') or '—'}"
        return (
            "| {code} | {name} | {px} | {flow} | {pos} | {mark} | {fig1} |"
            " {f7} | {f8} | {f9} | {f10} | {f11} | {note} |"
        ).format(
            code=_md_cell(r.get("code")),
            name=_md_cell(r.get("name")),
            px=px_s,
            flow=_md_cell(_state_flow(str(r.get("prev_state") or ""), str(r.get("state") or ""))),
            pos=_md_cell(flow_pos),
            mark=_md_cell(r.get("mark")),
            fig1=_md_cell(r.get("fig1")),
            f7=_md_cell(r.get("fig7_txt")),
            f8=_md_cell(r.get("fig8_txt")),
            f9=_md_cell(r.get("fig9_txt")),
            f10=_md_cell(r.get("fig10_txt")),
            f11=_md_cell(r.get("fig11_txt")),
            note=_md_cell(r.get("note") or ""),
        )

    for morph in pool.MORPH_ORDER:
        grp = sorted(by_morph[morph], key=lambda r: str(r.get("code") or ""))
        lines.append(f"### {morph}（{len(grp)} 只）")
        lines.append("")
        lines.append(detail_header)
        lines.append(detail_sep)
        for r in grp:
            lines.append(_detail_row(r))
        lines.append("")

    if other:
        lines.append(f"### 未入形态表（{len(other)} 只）")
        lines.append("")
        lines.append(detail_header)
        lines.append(detail_sep)
        for r in sorted(other, key=lambda x: str(x.get("code") or "")):
            lines.append(_detail_row(r))
        lines.append("")

    if fail_rows:
        lines.append("## 5. 失败")
        lines.append("")
        for r in fail_rows:
            err = str(r.get("error") or "").strip().splitlines()
            short = err[-1] if err else "unknown"
            lines.append(f"- `{r.get('code')}` {r.get('name')}: {short}")
        lines.append("")

    lines.extend(
        [
            "## 5. 读表注意" if not fail_rows else "## 6. 读表注意",
            "",
            "- 平稳慢牛型 ②/③ 仓位仍是满仓，决策树 ≡ 持有，不会出现买卖标注。",
            "- 只有深熊/V型的 ②③、高波动成长的 ② 会改仓位。",
            "- 图1「新标识」是当日新出现的诊断标记，不是决策树买卖指令。",
            "- 分位是相对**该图自己的公平路径**的因果残差分位，不是全市场截面分位。",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    cols = [
        "ok",
        "code",
        "name",
        "morph",
        "last_date",
        "asof",
        "px",
        "prev_state",
        "state",
        "prev_pos",
        "pos",
        "prev_pos_cn",
        "pos_cn",
        "mark",
        "fig1",
        "note",
        "fig7_g",
        "fig7_pct",
        "fig8_g",
        "fig8_pct",
        "fig9_g",
        "fig9_pct",
        "fig10_g",
        "fig10_pct",
        "fig11_g",
        "fig11_pct",
        "error",
    ]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    html_dir = args.html_dir.resolve()
    morph_map = tree.load_morph_map(args.morph_csv)
    files = list_html(html_dir, include_bonds=args.include_bonds)
    if not files:
        print(f"no html in {html_dir}", file=sys.stderr)
        return 1
    asof_s = str(args.as_of) if args.as_of else None
    jobs: list[tuple[str, str, str, str, str | None]] = []
    for p in files:
        code, name = pool.parse_html_label(p)
        jobs.append((str(p), code, name, morph_map.get(code, ""), asof_s))

    rows: list[dict[str, Any]] = []
    n_jobs = max(1, int(args.jobs))
    if n_jobs == 1 or len(jobs) == 1:
        for j in jobs:
            rows.append(scan_one(j))
            print(f"[{'ok' if rows[-1]['ok'] else 'fail'}] {j[1]} {j[2]}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = {ex.submit(scan_one, j): j for j in jobs}
            for fut in as_completed(futs):
                j = futs[fut]
                try:
                    row = fut.result()
                except Exception:
                    row = {
                        "ok": False,
                        "code": j[1],
                        "name": j[2],
                        "morph": j[3] or "未分类",
                        "error": traceback.format_exc(limit=4),
                    }
                rows.append(row)
                print(f"[{'ok' if row.get('ok') else 'fail'}] {j[1]} {j[2]}", flush=True)

    rows.sort(key=lambda r: (str(r.get("morph") or ""), str(r.get("code") or "")))
    apply_cross_veto_notes(rows)
    asof = asof_s or next((r.get("asof") for r in rows if r.get("asof")), "")
    asof = str(asof or "")

    md_path = args.write_md
    if md_path is None:
        md_path = html_dir / f"fig12_tree_eod_{asof.replace('-', '')}.md"
    csv_path = args.write_csv
    if csv_path is None:
        csv_path = html_dir / f"fig12_tree_eod_{asof.replace('-', '')}.csv"

    md = render_md(rows, asof=asof, html_dir=html_dir, morph_csv=args.morph_csv.resolve())
    md_path = Path(md_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
    frame = rows_to_frame(rows)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    n_ok = int(sum(1 for r in rows if r.get("ok")))
    print(f"wrote {md_path} ({n_ok}/{len(rows)} ok)")
    print(f"wrote {csv_path}")
    return 0 if n_ok == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
