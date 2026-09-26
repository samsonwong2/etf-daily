#!/usr/bin/env python3
"""P95 vs EWMA fig9 overlay on 煤炭 / 黄金 / 豆粕 (research only).

Fixed params (no grid): bank-selected EWMA λ=0.94, near=1.5%, leave=3%.
P95 baseline is documented L0+S0 (aux_edge), near=1%.

Does not change TREE_POS or production hold_up.

Example::

    PYTHONPATH=. python decision_pack/scripts/backtest_fig9_ewma_three_calm.py \\
      --write-md documents/平稳三只_图9_EWMA对照回测.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runtime_paths import PLOTLY_OUTPUTS_DIR  # noqa: E402
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import backtest_fig9_ewma_edge as ewma  # noqa: E402
from decision_pack.src.fair_path_band_signals import find_regime_html  # noqa: E402
from decision_pack.src.fair_path_research import ewma_scaled_envelope  # noqa: E402

LAM = 0.94
P95_NEAR = 0.01
EWMA_NEAR = 0.015
LEAVE_PCT = 0.03

CODES: tuple[tuple[str, str, str], ...] = (
    ("SH515220", "煤炭ETF国泰", "文档已改 EWMA L0+触上离开；立刻卖会砍 2021 主升"),
    ("SH518880", "黄金ETF华安", "震荡才 L0+S0；趋势须 G1/G2/G3 停图9"),
    ("SZ159985", "豆粕ETF华夏", "文档是 L2 穿回 + S1 骑轨，不是立刻买卖"),
)

LISTING_CANDIDATES = (
    "~/etf-daily-output/temp/plotly_outputs/20260915_from_listing",
    "~/etf-daily-output/temp/plotly_outputs/20260909_from_listing",
    "~/etf-daily-output/temp/plotly_outputs/20260824_from_listing",
)

VARIANT_ORDER = ("p95_l0s0", "ewma_l0s0", "ewma_l0_leave")
VARIANT_CN = {
    "p95_l0s0": "P95 轨 L0+S0（aux_edge）",
    "ewma_l0s0": "EWMA 轨 L0+S0",
    "ewma_l0_leave": "EWMA 轨 L0+触上离开",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write-md", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--skip-html", action="store_true")
    return p.parse_args(argv)


def _pct(x: float | None, digits: int = 1) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{100.0 * x:.{digits}f}%"


def resolve_html(code: str) -> tuple[Path, Path]:
    for rel in LISTING_CANDIDATES:
        listing = _PROJECT_ROOT / rel
        html = find_regime_html(listing, code)
        if html is not None and html.exists():
            return html, listing
    raise FileNotFoundError(code)


def closed_stats(trades: pd.DataFrame) -> dict[str, Any]:
    if trades is None or trades.empty:
        return {"n": 0, "n_open": 0, "win": float("nan"), "compound": float("nan"), "mean_hold": float("nan")}
    if "open" in trades.columns:
        open_mask = trades["open"].eq(True)
    elif "signal_sell" in trades.columns:
        open_mask = trades["signal_sell"].isna()
    else:
        open_mask = pd.Series(False, index=trades.index)
    closed = trades.loc[~open_mask]
    n_open = int(open_mask.sum())
    if closed.empty:
        return {"n": 0, "n_open": n_open, "win": float("nan"), "compound": float("nan"), "mean_hold": float("nan")}
    rets = pd.to_numeric(closed["ret"], errors="coerce").dropna()
    hold = pd.to_numeric(closed.get("hold_days"), errors="coerce")
    return {
        "n": int(len(rets)),
        "n_open": n_open,
        "win": float((rets > 0).mean()) if len(rets) else float("nan"),
        "compound": float((1.0 + rets).prod() - 1.0) if len(rets) else float("nan"),
        "mean_hold": float(hold.mean()) if hold.notna().any() else float("nan"),
    }


def run_variant(
    *,
    frame0: pd.DataFrame,
    ewma_lo: pd.Series,
    ewma_up: pd.Series,
    p95_lo: pd.Series,
    p95_up: pd.Series,
    name: str,
    touch: Any,
    sm: Any,
) -> dict[str, Any]:
    if name == "p95_l0s0":
        fr = ewma.apply_fig9_rails(
            frame0, lower=p95_lo, upper=p95_up, near_pct=P95_NEAR, touch_mod=touch, mid_stretch=True
        )
        buy, sell = ewma.build_candidates(
            fr,
            touch_mod=touch,
            use_aux_buy=True,
            use_aux_sell=True,
            use_stretch=True,
            locmin_days=0,
            holdup_gate=False,
            regime_up=pd.Series(False, index=fr.index),
        )
        trades, _, _ = sm.run_state_machine(fr, buy, sell)
        trades = touch.tag_sell_kind(trades, fr)
    elif name == "ewma_l0s0":
        fr = ewma.apply_fig9_rails(
            frame0, lower=ewma_lo, upper=ewma_up, near_pct=EWMA_NEAR, touch_mod=touch, mid_stretch=True
        )
        buy, sell = ewma.build_candidates(
            fr,
            touch_mod=touch,
            use_aux_buy=True,
            use_aux_sell=True,
            use_stretch=True,
            locmin_days=0,
            holdup_gate=False,
            regime_up=pd.Series(False, index=fr.index),
        )
        trades, _, _ = sm.run_state_machine(fr, buy, sell)
        trades = touch.tag_sell_kind(trades, fr)
    elif name == "ewma_l0_leave":
        fr = ewma.apply_fig9_rails(
            frame0, lower=ewma_lo, upper=ewma_up, near_pct=EWMA_NEAR, touch_mod=touch, mid_stretch=False
        )
        buy, _sell = ewma.build_candidates(
            fr,
            touch_mod=touch,
            use_aux_buy=True,
            use_aux_sell=False,
            use_stretch=False,
            locmin_days=0,
            holdup_gate=False,
            regime_up=pd.Series(False, index=fr.index),
        )
        trades, _, _ = ewma.run_touch_leave_state_machine(fr, buy, leave_pct=LEAVE_PCT)
    else:
        raise ValueError(name)
    st = closed_stats(trades)
    st["name"] = name
    st["trades"] = trades
    return st


def fmt_n(x: float, d: int = 1) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{d}f}"


def build_markdown(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# 煤炭 / 黄金 / 豆粕：图9 P95 vs EWMA 对照")
    lines.append("")
    lines.append(
        "> 目的：在三只已有拟合文档的平稳型上，把银行选定的 **EWMA 轨**（λ=0.94）"
        "接到统一 L0，看能否代替豆粕穿回、黄金过门、煤炭立刻卖，并提高相对持有的收益"
    )
    lines.append(
        "> 复现：`PYTHONPATH=. python decision_pack/scripts/backtest_fig9_ewma_three_calm.py "
        "--write-md documents/平稳三只_图9_EWMA对照回测.md`"
    )
    lines.append("")
    lines.append("## 0. 口径")
    lines.append("")
    lines.append("- 路径仍是图9 六个月 OLS；P95 轨 = 现 HTML expanding 包络；EWMA 轨 = `ewma_scaled_envelope`。")
    lines.append("- **p95_l0s0**：近下上跳沿 ∧ 辅助买 / 近上上跳沿 ∧ 辅助卖 ∨ 带内拉伸。near=1%（文档 aux_edge）。")
    lines.append(
        "- **ewma_l0s0**：同一套 L0+S0，轨换成 EWMA，near=1.5%（银行买档）。"
    )
    lines.append(
        "- **ewma_l0_leave**：EWMA 近下上跳沿 ∧ 辅助买；本轮硬触上轨后再离开 `dist_hi≤−3%` 才卖（银行卖档）。"
    )
    lines.append("- 只做多，T 收盘信号 → T+1 收盘成交。复合 = **已平仓**段收益连乘，未平另计。")
    lines.append("- 无网格。豆粕文档 L2/S1、黄金 G1/G2/G3 **不重跑**，只当对照目标。")
    lines.append("- 研究对照，不改决策树、不改生产 `hold_up`。")
    lines.append("")
    lines.append("## 1. 三只对照")
    lines.append("")
    lines.append(
        "| 代码 | 名称 | 样本 | 持有累计 | 变体 | 已平笔 | 胜率 | 已平复合 | vs持有 | 均持仓 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    by_code: dict[str, list[dict[str, Any]]] = {}
    for rec in rows:
        by_code.setdefault(rec["code"], []).append(rec)
        html_rel = rec["html"]
        try:
            html_rel = Path(rec["html"]).relative_to(_PROJECT_ROOT).as_posix()
        except ValueError:
            pass
        span = f"{rec['start']}→{rec['end']}"
        for v in VARIANT_ORDER:
            st = rec["variants"][v]
            d = (
                float(st["compound"]) - float(rec["bh"])
                if np.isfinite(st["compound"]) and np.isfinite(rec["bh"])
                else float("nan")
            )
            lines.append(
                f"| {rec['code']} | {rec['name']} | {span} | {_pct(rec['bh'])} | "
                f"{VARIANT_CN[v]} | {st['n']}"
                f"{' +未平' if st['n_open'] else ''} | {_pct(st['win'])} | "
                f"{_pct(st['compound'])} | {_pct(d, 1)} | {fmt_n(st['mean_hold'], 0)} |"
            )
    lines.append("")
    lines.append("## 2. 逐只读法")
    lines.append("")
    for code, _cname, why in CODES:
        recs = by_code.get(code, [])
        if not recs:
            continue
        rec = recs[0]
        lines.append(f"### {code} {rec['name']}")
        lines.append("")
        lines.append(f"文档难点：{why}")
        lines.append("")
        bh = rec["bh"]
        p95 = rec["variants"]["p95_l0s0"]
        es0 = rec["variants"]["ewma_l0s0"]
        elv = rec["variants"]["ewma_l0_leave"]
        lines.append(
            f"- 持有 {_pct(bh)}；P95 aux_edge 已平 {_pct(p95['compound'])}（{p95['n']} 笔）；"
            f"EWMA S0 {_pct(es0['compound'])}（{es0['n']} 笔）；"
            f"EWMA 离开 {_pct(elv['compound'])}（{elv['n']} 笔）。"
        )
        best = max(
            (("P95 L0+S0", p95), ("EWMA L0+S0", es0), ("EWMA 离开", elv)),
            key=lambda kv: kv[1]["compound"] if np.isfinite(kv[1]["compound"]) else -1e9,
        )
        beat_bh = np.isfinite(best[1]["compound"]) and best[1]["compound"] > bh + 1e-12
        lines.append(
            f"- 三门里最高是 **{best[0]}**。"
            + ("已平复合高于持有。" if beat_bh else "仍不高于持有——简化规则没有把这只做成该空仓波段的票。")
        )
        lines.append("")
        trades = rec["variants"]["ewma_l0_leave"]["trades"]
        if trades is not None and not trades.empty:
            lines.append("| 买信号 | 卖信号 | 卖因 | 段收益 | 持仓日 |")
            lines.append("|---|---|---|---:|---:|")
            for _, tr in trades.iterrows():
                kind = tr.get("sell_kind", "") or ("未平" if tr.get("open") else "")
                lines.append(
                    f"| {tr.get('signal_buy')} | {tr.get('signal_sell') or '—'} | "
                    f"{kind} | {_pct(float(tr['ret']))} | {int(tr['hold_days'])} |"
                )
            lines.append("")
    lines.append("## 3. 能否简化、能否提高收益")
    lines.append("")
    n_beat = 0
    n_best_ewma = 0
    for rec in rows:
        bh = rec["bh"]
        comps = {k: rec["variants"][k]["compound"] for k in VARIANT_ORDER}
        finite = {k: v for k, v in comps.items() if np.isfinite(v)}
        if not finite:
            continue
        best_k = max(finite, key=lambda k: finite[k])
        if finite[best_k] > bh + 1e-12:
            n_beat += 1
        if best_k.startswith("ewma"):
            n_best_ewma += 1
    lines.append(
        f"三只里，任一简化门（含 P95 aux_edge）已平复合高于持有的有 **{n_beat}/3**；"
        f"三门最高落在 EWMA 的有 **{n_best_ewma}/3**。"
    )
    lines.append("")
    gold = by_code.get("SH518880", [None])[0]
    soy = by_code.get("SZ159985", [None])[0]
    coal = by_code.get("SH515220", [None])[0]
    if gold:
        g_bh, g_p, g_l = gold["bh"], gold["variants"]["p95_l0s0"]["compound"], gold["variants"]["ewma_l0_leave"]["compound"]
        if np.isfinite(g_l) and np.isfinite(g_p) and g_l > g_p + 0.02:
            lines.append(
                f"黄金：EWMA 离开（{_pct(g_l)}）好于 P95 立刻卖（{_pct(g_p)}），"
                f"仍对照持有 {_pct(g_bh)}。"
                + (" 还没追上持有，G1/G2 过门不能删。" if g_l < g_bh else " 已追上持有，可用离开代替过门再复核。")
            )
        else:
            lines.append("黄金：EWMA 没有明显好过 P95 立刻卖，过门（趋势停图9）仍必要。")
    if soy:
        s_bh, s_p, s_e = soy["bh"], soy["variants"]["p95_l0s0"]["compound"], soy["variants"]["ewma_l0_leave"]["compound"]
        lines.append(
            f"豆粕：P95 立刻买 {_pct(s_p)}，EWMA 离开 {_pct(s_e)}，持有 {_pct(s_bh)}。"
            "文档 L2 就是因为第一次近下不可信；EWMA 若仍是边沿买，**不能**代替穿回。"
        )
    if coal:
        c_bh, c_p, c_l = coal["bh"], coal["variants"]["p95_l0s0"]["compound"], coal["variants"]["ewma_l0_leave"]["compound"]
        if np.isfinite(c_l) and np.isfinite(c_bh) and c_l > c_bh:
            lines.append(
                f"煤炭：P95 立刻卖 {_pct(c_p)} 低于持有；EWMA 离开 {_pct(c_l)} 高于持有 {_pct(c_bh)}。"
                "这只可以把文档 S0/拉伸简化成银行那套「触上再离开」。"
                "这是单票结果，不要套到黄金或豆粕。"
            )
        else:
            lines.append(
                f"煤炭：P95 aux_edge {_pct(c_p)}，EWMA 离开 {_pct(c_l)}，持有 {_pct(c_bh)}。"
                "复合仍不高于持有时，默认继续持有。"
            )
    lines.append("")
    lines.append(
        "**结论**：EWMA 轨能接到同一套 L0，判断比豆粕穿回、黄金三模式简单；"
        "固定银行参数下 **只有煤炭** 相对持有提高了收益。"
        "不能把三只收成一条生产规则，也不改 42 只平稳型决策树。"
        "煤炭可单列「EWMA L0 + 触上离开」对照；黄金仍要过门；豆粕仍要穿回。"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sm = ewma._load_sm()
    touch = ewma._load_touch()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else PLOTLY_OUTPUTS_DIR / "20260915_from_listing/fig9_ewma_three_calm"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for code, name, _why in CODES:
        html, listing = resolve_html(code)
        print(f"[INFO] {code} {html.name}", flush=True)
        frame0, _ohlcv = sm.build_feature_frame(html, listing, "SH510300", lookback=10)
        close = frame0["px"].astype(float)
        path9 = pd.to_numeric(frame0["fig9_path"], errors="coerce")
        ewma_up, ewma_lo, _ = ewma_scaled_envelope(path9, close, lam=LAM)
        p95_lo = pd.to_numeric(frame0["fig9_lower"], errors="coerce")
        p95_up = pd.to_numeric(frame0["fig9_upper"], errors="coerce")
        bh = float(close.iloc[-1] / close.iloc[0] - 1.0) if float(close.iloc[0]) else float("nan")
        variants: dict[str, dict[str, Any]] = {}
        for v in VARIANT_ORDER:
            st = run_variant(
                frame0=frame0,
                ewma_lo=ewma_lo,
                ewma_up=ewma_up,
                p95_lo=p95_lo,
                p95_up=p95_up,
                name=v,
                touch=touch,
                sm=sm,
            )
            trades = st.pop("trades")
            trades.to_csv(out_dir / f"trades_{code}_{v}.csv", index=False, encoding="utf-8-sig")
            st["trades"] = trades
            variants[v] = st
            print(
                f"  [{v}] n={st['n']} wr={_pct(st['win'])} cp={_pct(st['compound'])} bh={_pct(bh)}",
                flush=True,
            )
        rows.append(
            {
                "code": code,
                "name": name,
                "html": str(html),
                "start": pd.Timestamp(close.index[0]).strftime("%Y-%m-%d"),
                "end": pd.Timestamp(close.index[-1]).strftime("%Y-%m-%d"),
                "bh": bh,
                "variants": variants,
            }
        )

    md = build_markdown(rows)
    md_path = (
        args.write_md.expanduser().resolve()
        if args.write_md is not None
        else _PROJECT_ROOT / "documents" / "平稳三只_图9_EWMA对照回测.md"
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
    print(f"[OK] md={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
