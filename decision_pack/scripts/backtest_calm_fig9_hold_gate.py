#!/usr/bin/env python3
"""Calm-morphology fig9 channel hold-gate A/B (research only).

Does not change TREE_POS. T-close flags → T+1 position.

Gates
-----
GateStrict : hold iff inside fig9 bands and fig9_g > 0, else empty
GateSlow   : hold iff inside and 0 < fig9_g <= 0.50, else empty
GateBand   : hold iff inside (ignore slope), else empty
GateHiOnly : empty only on fig9 upper-band touch, else hold

Example::

    PYTHONPATH=. python decision_pack/scripts/backtest_calm_fig9_hold_gate.py \\
      --html-dir ~/etf-daily-output/temp/plotly_outputs/20260915_from_listing \\
      --morph-csv ~/etf-daily-output/temp/plotly_outputs/20260915_from_listing/fig12_six_state_backtest/per_etf.csv \\
      --write-md documents/平稳型_fig9通道持仓开关回测.md
"""
from __future__ import annotations

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import backtest_fig12_six_states_pool as pool  # noqa: E402
from decision_pack.src.fair_path_band_signals import (  # noqa: E402
    compute_multi_scale_frame,
    enrich_frame_for_checklist,
    load_ohlcv_from_regime_html,
)

EPS = 1e-12
G_SLOW_CAP = 0.50
FOCUS_CODE = "SH512530"
GATE_ORDER = ("GateStrict", "GateSlow", "GateBand", "GateHiOnly")
GATE_CN = {
    "GateStrict": "通道内且 fig9_g>0 才持有，否则空",
    "GateSlow": "通道内且 0<fig9_g≤0.50 才持有，否则空",
    "GateBand": "通道内持有，碰上/下轨才空",
    "GateHiOnly": "仅碰上轨空仓（不追），其余持有",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--html-dir", type=Path, required=True)
    p.add_argument("--morph-csv", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--write-md", type=Path, default=None)
    return p.parse_args(argv)


def equity_mdd(daily: pd.Series) -> float:
    eq = (1.0 + pd.to_numeric(daily, errors="coerce").fillna(0.0)).cumprod()
    return pool.max_drawdown(eq)


def cagr_from_total(total: float, start, end) -> float:
    years = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25, 1e-6)
    if total <= -1.0 or not np.isfinite(total):
        return float("nan")
    return float((1.0 + total) ** (1.0 / years) - 1.0)


def gate_weights(ms: pd.DataFrame) -> dict[str, pd.Series]:
    g9 = pd.to_numeric(ms["fig9_g"], errors="coerce")
    hi = ms["fig9_touch_hi"].fillna(False).astype(bool)
    lo = ms["fig9_touch_lo"].fillna(False).astype(bool)
    inside = (~hi) & (~lo)
    up = g9 > 0
    slow_up = (g9 > 0) & (g9 <= G_SLOW_CAP)
    ready = g9.notna()
    w: dict[str, pd.Series] = {}
    w["GateStrict"] = (inside & up & ready).astype(float)
    w["GateSlow"] = (inside & slow_up & ready).astype(float)
    w["GateBand"] = (inside & ready).astype(float)
    w["GateHiOnly"] = (~hi & ready).astype(float)
    for k, s in list(w.items()):
        w[k] = s.where(ready, np.nan)
    return w


def metrics_from_returns(r_bh: pd.Series, r_strat: pd.Series, weight: pd.Series) -> dict:
    start = pd.Timestamp(r_bh.index[0])
    end = pd.Timestamp(r_bh.index[-1])
    total_bh = float((1.0 + r_bh).prod() - 1.0)
    total = float((1.0 + r_strat).prod() - 1.0)
    sharpe_bh = pool.ann_sharpe(r_bh)
    sharpe = pool.ann_sharpe(r_strat)
    mdd_bh = equity_mdd(r_bh)
    mdd = equity_mdd(r_strat)
    cagr_bh = cagr_from_total(total_bh, start, end)
    cagr = cagr_from_total(total, start, end)
    hold_share = float(weight.mean()) if len(weight) else float("nan")
    return {
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "n": int(len(r_bh)),
        "hold_share": hold_share,
        "total_bh": total_bh,
        "cagr_bh": cagr_bh,
        "sharpe_bh": sharpe_bh,
        "mdd_bh": mdd_bh,
        "total": total,
        "cagr": cagr,
        "sharpe": sharpe,
        "mdd": mdd,
        "excess_cagr": (
            cagr - cagr_bh if np.isfinite(cagr) and np.isfinite(cagr_bh) else float("nan")
        ),
        "excess_sharpe": (
            sharpe - sharpe_bh if np.isfinite(sharpe) and np.isfinite(sharpe_bh) else float("nan")
        ),
        "beat_total": bool(total > total_bh + EPS),
    }


def process_one(path_str: str) -> dict:
    path = Path(path_str)
    code, name = pool.parse_html_label(path)
    close, _ohlcv = load_ohlcv_from_regime_html(path)
    ms = enrich_frame_for_checklist(compute_multi_scale_frame(close))
    px = pd.to_numeric(ms["px"], errors="coerce")
    weights = gate_weights(ms)
    r_full = px.pct_change(fill_method=None)
    rows = []
    for gate in GATE_ORDER:
        w = weights[gate]
        lagged = w.shift(1)
        valid = lagged.notna() & r_full.notna() & px.notna()
        if int(valid.sum()) < 60:
            raise ValueError(f"too few bars after warmup: {int(valid.sum())}")
        r_bh = r_full[valid].astype(float)
        w_lag = lagged[valid].astype(float)
        r_strat = (w_lag * r_bh).astype(float)
        rec = metrics_from_returns(r_bh, r_strat, w_lag)
        rec.update({"ok": True, "code": code, "name": name, "path": str(path), "gate": gate})
        rows.append(rec)
    return {"ok": True, "rows": rows, "error": ""}


def _worker(path_str: str) -> dict:
    try:
        return process_one(path_str)
    except Exception as exc:  # noqa: BLE001
        path = Path(path_str)
        code, name = pool.parse_html_label(path)
        return {
            "ok": False,
            "rows": [],
            "error": f"{code} {name}: {type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


def _focus_table(df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    sub = df[df["code"] == FOCUS_CODE]
    if sub.empty:
        lines.append("样本中没有 SH512530。")
        lines.append("")
        return lines
    name = str(sub["name"].iloc[0])
    start = str(sub["start"].iloc[0])
    end = str(sub["end"].iloc[0])
    bh = sub.iloc[0]
    lines.append(
        f"{FOCUS_CODE} {name}，{start}→{end}（{int(bh['n'])} 日）。"
        f"买入持有累计 {pool.fmt_pct(bh['total_bh'])}，"
        f"年化 {pool.fmt_pct(bh['cagr_bh'], 2)}，"
        f"夏普 {pool.fmt_n(bh['sharpe_bh'])}，"
        f"回撤 {pool.fmt_pct(bh['mdd_bh'])}。"
    )
    lines.append("")
    lines.append("| 门 | 含义 | 持仓天数 | 累计 | 年化 | 夏普 | 回撤 | Δ累计 | 超额年化 | Δ夏普 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for gate in GATE_ORDER:
        r = sub[sub["gate"] == gate]
        if r.empty:
            continue
        r = r.iloc[0]
        d_tot = float(r["total"]) - float(r["total_bh"])
        lines.append(
            f"| {gate} | {GATE_CN[gate]} | {pool.fmt_pct(r['hold_share'])} | "
            f"{pool.fmt_pct(r['total'])} | {pool.fmt_pct(r['cagr'], 2)} | "
            f"{pool.fmt_n(r['sharpe'])} | {pool.fmt_pct(r['mdd'])} | "
            f"{pool.fmt_pct(d_tot, 2)} | {pool.fmt_pct(r['excess_cagr'], 2)} | "
            f"{pool.fmt_n(r['excess_sharpe'])} |"
        )
    lines.append("")
    return lines


def _pool_table(df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    n_etf = int(df["code"].nunique())
    lines.append(
        f"共 **{n_etf}** 只平稳慢牛型。组内中位；「跑赢累计」= 该门累计 > 买入持有累计。"
    )
    lines.append("")
    lines.append("| 门 | 中位持仓天数 | 中位Δ夏普 | 中位超额年化 | 中位Δ累计 | 跑赢累计只数 |")
    lines.append("|---|---|---|---|---|---|")
    for gate in GATE_ORDER:
        sub = df[df["gate"] == gate]
        d_tot = sub["total"] - sub["total_bh"]
        n_beat = int(sub["beat_total"].sum())
        lines.append(
            f"| {gate} | {pool.fmt_pct(float(sub['hold_share'].median()))} | "
            f"{pool.fmt_n(float(sub['excess_sharpe'].median()))} | "
            f"{pool.fmt_pct(float(sub['excess_cagr'].median()), 2)} | "
            f"{pool.fmt_pct(float(d_tot.median()), 2)} | "
            f"{n_beat}/{len(sub)} |"
        )
    lines.append("")
    return lines


def _verdict(df: pd.DataFrame) -> list[str]:
    lines = ["## 3. 结论：能不能叠进决策树", ""]
    focus = df[df["code"] == FOCUS_CODE]
    n_etf = int(df["code"].nunique())

    def row(gate: str, src: pd.DataFrame) -> pd.Series | None:
        s = src[src["gate"] == gate]
        return None if s.empty else s.iloc[0]

    fs = row("GateStrict", focus)
    fslow = row("GateSlow", focus)
    fhi = row("GateHiOnly", focus)
    pool_s = df[df["gate"] == "GateStrict"]
    pool_hi = df[df["gate"] == "GateHiOnly"]
    med_s = float(pool_s["excess_sharpe"].median()) if not pool_s.empty else float("nan")
    med_hi = float(pool_hi["excess_sharpe"].median()) if not pool_hi.empty else float("nan")
    beat_s = int(pool_s["beat_total"].sum()) if not pool_s.empty else 0
    beat_hi = int(pool_hi["beat_total"].sum()) if not pool_hi.empty else 0

    strict_hurt = (
        fs is not None
        and np.isfinite(float(fs["excess_sharpe"]))
        and float(fs["excess_sharpe"]) < -0.02
    )
    slow_hurt = (
        fslow is not None
        and np.isfinite(float(fslow["excess_sharpe"]))
        and float(fslow["excess_sharpe"]) < -0.02
    )
    hi_helps_focus = (
        fhi is not None
        and np.isfinite(float(fhi["excess_sharpe"]))
        and float(fhi["excess_sharpe"]) > 0.02
    )
    hi_helps_pool = np.isfinite(med_hi) and med_hi > 0.02 and beat_hi > n_etf / 2

    if strict_hurt or slow_hurt:
        lines.append(
            "**GateStrict / GateSlow 不能叠进决策树。** "
            "通道内且红线向上才持有，等价于 fig9_g≤0（③类阴跌确认）时空仓；"
            f"在 {FOCUS_CODE} 上 Δ夏普为负，与八品种 §7.2「红利 ③空仓 0.56→0.13」同向。"
            "平稳型继续全程持有。"
        )
    elif (
        fs is not None
        and float(fs["excess_sharpe"]) > 0.02
        and np.isfinite(med_s)
        and med_s > 0.02
        and beat_s > n_etf / 2
    ):
        lines.append(
            "GateStrict 在红利单只与平稳组中位都优于持有，可作为候选叠加；"
            "本轮仍不改 TREE_POS，需另开样本外复核。"
        )
    else:
        lines.append(
            "**不改决策树。** GateStrict 没有在红利单只与平稳组中位上同时明显优于持有。"
        )

    if hi_helps_focus and hi_helps_pool:
        lines.append(
            "GateHiOnly（仅碰上轨空）在单只和组中位都有改善，最多把平稳型①从「不动作」改成「碰上轨减仓」；"
            "仍不要空③。本轮不改表，只记这条对照。"
        )
    elif hi_helps_focus and not hi_helps_pool:
        lines.append(
            "GateHiOnly 在红利上略有改善，但 43 只中位没有稳定超额，不够改平稳型①。"
        )
    else:
        lines.append(
            "GateHiOnly 也没有稳定超额：碰上轨空仓对平稳型仍是时间成本，保持「①不动作」。"
        )
    lines.append("")
    lines.append(
        "本回测 in-sample、不计手续费。通道开关是持仓过滤器，不是新买入信号。"
    )
    lines.append("")
    return lines


def build_markdown(df: pd.DataFrame, html_dir: Path, morph_csv: Path) -> str:
    try:
        rel = html_dir.relative_to(_PROJECT_ROOT)
    except ValueError:
        rel = html_dir
    try:
        morph_rel = morph_csv.relative_to(_PROJECT_ROOT)
    except ValueError:
        morph_rel = morph_csv
    as_of = str(df["end"].max()) if len(df) else ""
    n = int(df["code"].nunique())
    lines: list[str] = []
    lines.append("# 平稳型：fig9 通道持仓开关回测")
    lines.append("")
    lines.append(
        "> 目的：检验「图9 上下轨之间、红线缓慢向上」能否作为平稳慢牛型的持仓开关"
        "（通道内且向上才持有，否则空），不改现有 6×4 决策树"
    )
    lines.append(
        f"> 数据：`{rel.as_posix()}` adaptive HTML（前复权，截止 {as_of}）；"
        f"形态来自 `{morph_rel.as_posix()}` 中 `morph==平稳慢牛型`（{n} 只）"
    )
    lines.append(
        "> 复现：`PYTHONPATH=. python decision_pack/scripts/backtest_calm_fig9_hold_gate.py "
        "--html-dir ~/etf-daily-output/temp/plotly_outputs/20260915_from_listing "
        "--morph-csv ~/etf-daily-output/temp/plotly_outputs/20260915_from_listing/fig12_six_state_backtest/per_etf.csv "
        "--write-md documents/平稳型_fig9通道持仓开关回测.md`"
    )
    lines.append("")
    lines.append("## 0. 口径")
    lines.append("")
    lines.append("- T 日收盘的 fig9 触轨 / `fig9_g` → T+1 仓位，不计手续费。")
    lines.append("- 通道内 = 未触 fig9 上轨且未触 fig9 下轨（`not touch_hi` 且 `not touch_lo`）。")
    lines.append("- 向上 = `fig9_g > 0`；慢 = `0 < fig9_g ≤ 0.50`。")
    lines.append(
        "- 先验：[六状态趋势划分_八品种验证.md](六状态趋势划分_八品种验证.md) §7.2 "
        "SH512530 ③空仓夏普 0.56→0.13。GateStrict 在 g≤0 时空仓，接近空③。"
    )
    lines.append("- 本轮只出证据，不改 `TREE_POS`。")
    lines.append("")
    lines.append("## 1. SH512530 沪深300红利")
    lines.append("")
    lines.extend(_focus_table(df))
    lines.append("## 2. 43 只平稳慢牛型")
    lines.append("")
    lines.extend(_pool_table(df))
    lines.append("逐只（GateStrict）：")
    lines.append("")
    lines.append("| 代码 | 名称 | 持有累计 | Strict累计 | Δ累计 | Δ夏普 | 超额年化 | 持仓天数 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    strict = df[df["gate"] == "GateStrict"].sort_values("excess_sharpe")
    for _, r in strict.iterrows():
        d_tot = float(r["total"]) - float(r["total_bh"])
        lines.append(
            f"| {r['code']} | {r['name']} | {pool.fmt_pct(r['total_bh'])} | "
            f"{pool.fmt_pct(r['total'])} | {pool.fmt_pct(d_tot, 2)} | "
            f"{pool.fmt_n(r['excess_sharpe'])} | {pool.fmt_pct(r['excess_cagr'], 2)} | "
            f"{pool.fmt_pct(r['hold_share'])} |"
        )
    lines.append("")
    lines.extend(_verdict(df))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    html_dir = args.html_dir.expanduser().resolve()
    morph_csv = args.morph_csv.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else html_dir / "fig9_calm_hold_gate"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    morph = pd.read_csv(morph_csv)
    calm_codes = set(
        morph.loc[morph["morph"] == pool.MORPH_CALM, "code"].astype(str).str.strip()
    )
    files = [p for p in pool.list_equity_html(html_dir) if pool.parse_html_label(p)[0] in calm_codes]
    if not files:
        print("[ERROR] no calm HTML", file=sys.stderr)
        return 2

    # SH512530 first in logs
    files = sorted(files, key=lambda p: (pool.parse_html_label(p)[0] != FOCUS_CODE, p.name))

    jobs = max(1, int(args.jobs))
    results: list[dict] = []
    if jobs == 1 or len(files) == 1:
        results = [_worker(str(p)) for p in files]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_worker, str(p)): p for p in files}
            done = 0
            for fut in as_completed(futs):
                rec = fut.result()
                results.append(rec)
                done += 1
                p = futs[fut]
                code, name = pool.parse_html_label(p)
                flag = "OK" if rec.get("ok") else "FAIL"
                print(f"[{done}/{len(files)}] {flag} {code} {name}")
                if not rec.get("ok"):
                    print(rec.get("error", ""), file=sys.stderr)

    failed = [r for r in results if not r.get("ok")]
    ok = [r for r in results if r.get("ok")]
    pd.DataFrame([{"error": r.get("error", "")} for r in failed]).to_csv(
        out_dir / "errors.csv", index=False
    )
    if not ok:
        print("[ERROR] all failed", file=sys.stderr)
        return 1

    df = pd.DataFrame([row for r in ok for row in r["rows"]])
    df = df.sort_values(["code", "gate"]).reset_index(drop=True)
    df.to_csv(out_dir / "per_etf_gate.csv", index=False)

    md = build_markdown(df, html_dir, morph_csv)
    md_path = (
        args.write_md.expanduser().resolve()
        if args.write_md is not None
        else _PROJECT_ROOT / "documents" / "平稳型_fig9通道持仓开关回测.md"
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")

    print(f"[OK] n_etf={df['code'].nunique()} fail={len(failed)}")
    focus = df[df["code"] == FOCUS_CODE][["gate", "sharpe", "excess_sharpe", "total", "hold_share"]]
    if not focus.empty:
        print(focus.to_string(index=False))
    print(f"[OK] csv={out_dir / 'per_etf_gate.csv'}")
    print(f"[OK] md={md_path}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
