#!/usr/bin/env python3
"""Pool backtest of fig12 six-state overlays (documents/六状态趋势划分 §7).

For each from_listing adaptive HTML (bonds skipped):
  - recompute causal six states
  - buy-and-hold Sharpe / max drawdown
  - overlay empty/half/full while in each state (T close → T+1 close)

Then cluster ETFs into 深熊阴跌 / V型急跌 / 平稳慢牛 and pick the
median-Sharpe action per (state × morphology).

Example::

    PYTHONPATH=. python src/etf_daily/scripts/backtest_fig12_six_states_pool.py \\
      --html-dir ~/etf-daily-output/temp/plotly_outputs/20260915_from_listing --jobs 8
"""
from __future__ import annotations

import argparse
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.lib.fair_path_band_signals import (  # noqa: E402
    compute_multi_scale_frame,
    enrich_frame_for_checklist,
    load_ohlcv_from_regime_html,
)
from etf_daily.lib.six_states import (  # noqa: E402
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

_FNAME_RE = re.compile(
    r"^regime_transition_((?:SH|SZ)\d+)_(.+)_(\d{8})_(\d{8})_adaptive\.html$"
)
_BOND_NAME_RE = re.compile(r"国债|政金债|城投债|信用债|可转债|债券|货币")
_BOND_CODES = {"SH511010", "SH511030", "SH511090", "SH511260"}

TDY = 252.0
ACTIONS = (0.0, 0.5, 1.0)
ACTION_CN = {0.0: "空仓", 0.5: "半仓", 1.0: "满仓"}
MORPH_BEAR = "深熊阴跌型"
MORPH_V = "V型急跌快修复型"
MORPH_GROWTH = "高波动成长型"
MORPH_CALM = "平稳慢牛型"
MORPH_ORDER = (MORPH_BEAR, MORPH_V, MORPH_GROWTH, MORPH_CALM)
MDD_DEEP = 0.45
SHARPE_LIFT = 0.07
GROWTH_D2 = 0.20
GROWTH_MAX_BARS = 1000  # ~4 年：年轻、回撤未达 45% 但 ② 空仓显著改善



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--html-dir", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--write-md", type=Path, default=None)
    p.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="reuse per_etf.csv (skip HTML recompute); still re-clusters morph",
    )
    return p.parse_args(argv)


def parse_html_label(path: Path) -> tuple[str, str]:
    m = _FNAME_RE.match(path.name)
    if not m:
        return path.stem, path.stem
    return m.group(1), m.group(2)


def is_bond(code: str, name: str) -> bool:
    if code.upper() in _BOND_CODES:
        return True
    return bool(_BOND_NAME_RE.search(name or ""))


def list_equity_html(html_dir: Path) -> list[Path]:
    out: list[Path] = []
    skipped: list[str] = []
    for p in sorted(html_dir.glob("regime_transition_*_adaptive.html")):
        if not _FNAME_RE.match(p.name):
            continue
        code, name = parse_html_label(p)
        if is_bond(code, name):
            skipped.append(f"{code} {name}")
            continue
        out.append(p)
    if skipped:
        print("[skip bond]", ", ".join(skipped))
    return out


def ann_sharpe(daily: pd.Series) -> float:
    r = pd.to_numeric(daily, errors="coerce").dropna()
    if len(r) < 20:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd <= 0 or not np.isfinite(sd):
        return float("nan")
    return float(r.mean() / sd * np.sqrt(TDY))


def max_drawdown(px: pd.Series) -> float:
    c = pd.to_numeric(px, errors="coerce").dropna()
    if c.empty:
        return float("nan")
    peak = c.cummax()
    dd = c / peak - 1.0
    return float(dd.min())


def overlay_daily(px: pd.Series, states: pd.Series, target: str, pos: float) -> pd.Series:
    """T-day state → T+1 return. pos in target state, else 1.0."""
    r = px.pct_change()
    lagged = states.shift(1)
    weight = pd.Series(1.0, index=px.index)
    weight = weight.mask(lagged.eq(target), float(pos))
    weight = weight.fillna(1.0)
    return (weight * r).fillna(0.0)


def bh_daily(px: pd.Series) -> pd.Series:
    return px.pct_change().fillna(0.0)


def process_one(path_str: str) -> dict:
    path = Path(path_str)
    code, name = parse_html_label(path)
    close, _ohlcv = load_ohlcv_from_regime_html(path)
    ms = enrich_frame_for_checklist(compute_multi_scale_frame(close))
    states = classify_six_states(ms)
    px = pd.to_numeric(ms["px"], errors="coerce")
    mask = px.notna() & states.ne(STATE_WARMUP)
    px = px[mask]
    st = states[mask]
    if len(px) < 60:
        raise ValueError(f"too few bars after warmup: {len(px)}")

    r_bh = bh_daily(px)
    sharpe_bh = ann_sharpe(r_bh)
    mdd = max_drawdown(px)
    total = float(px.iloc[-1] / px.iloc[0] - 1.0)
    years = max((px.index[-1] - px.index[0]).days / 365.25, 1e-6)
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0) if total > -1 else float("nan")

    row: dict = {
        "ok": True,
        "code": code,
        "name": name,
        "path": str(path),
        "start": pd.Timestamp(px.index[0]).strftime("%Y-%m-%d"),
        "end": pd.Timestamp(px.index[-1]).strftime("%Y-%m-%d"),
        "n": int(len(px)),
        "total_ret": total,
        "cagr": cagr,
        "sharpe_bh": sharpe_bh,
        "mdd": mdd,
        "error": "",
    }
    vc = st.value_counts()
    for s in STATE_ORDER:
        row[f"share_{s}"] = float(vc.get(s, 0) / len(st))
        r0 = overlay_daily(px, st, s, 0.0)
        r05 = overlay_daily(px, st, s, 0.5)
        row[f"sharpe_{s}_empty"] = ann_sharpe(r0)
        row[f"sharpe_{s}_half"] = ann_sharpe(r05)
        row[f"sharpe_{s}_full"] = sharpe_bh  # full = hold
        d0 = row[f"sharpe_{s}_empty"] - sharpe_bh
        d05 = row[f"sharpe_{s}_half"] - sharpe_bh
        best = 1.0
        best_s = sharpe_bh
        if np.isfinite(row[f"sharpe_{s}_empty"]) and row[f"sharpe_{s}_empty"] > best_s + 1e-12:
            best, best_s = 0.0, row[f"sharpe_{s}_empty"]
        if np.isfinite(row[f"sharpe_{s}_half"]) and row[f"sharpe_{s}_half"] > best_s + 1e-12:
            best, best_s = 0.5, row[f"sharpe_{s}_half"]
        row[f"best_{s}"] = ACTION_CN[best]
        row[f"dsharpe_{s}_empty"] = d0
        row[f"dsharpe_{s}_half"] = d05
    return row


def _worker(path_str: str) -> dict:
    try:
        return process_one(path_str)
    except Exception as exc:  # noqa: BLE001
        path = Path(path_str)
        code, name = parse_html_label(path)
        return {
            "ok": False,
            "code": code,
            "name": name,
            "path": path_str,
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


def classify_morph(row: pd.Series) -> str:
    mdd = abs(float(row["mdd"])) if pd.notna(row["mdd"]) else 0.0
    d2 = float(row.get(f"dsharpe_{STATE_KNIFE}_empty", 0.0) or 0.0)
    d3 = float(row.get(f"dsharpe_{STATE_DOWN}_empty", 0.0) or 0.0)
    n_bars = int(row["n"]) if pd.notna(row.get("n")) else 0
    if not np.isfinite(d2):
        d2 = 0.0
    if not np.isfinite(d3):
        d3 = 0.0
    deep = mdd >= MDD_DEEP
    knife_helps = d2 >= SHARPE_LIFT
    down_helps = d3 >= SHARPE_LIFT
    knife_hurts = d2 <= -0.02
    if deep and (knife_helps or down_helps):
        if knife_hurts and down_helps:
            return MORPH_V
        if knife_helps:
            return MORPH_BEAR
        return MORPH_V
    if d2 >= GROWTH_D2 and 0 < n_bars < GROWTH_MAX_BARS:
        return MORPH_GROWTH
    return MORPH_CALM


def median_action(sub: pd.DataFrame, state: str) -> tuple[str, dict[str, float]]:
    cols = {
        "空仓": f"sharpe_{state}_empty",
        "半仓": f"sharpe_{state}_half",
        "满仓": f"sharpe_{state}_full",
    }
    meds = {k: float(sub[c].median()) for k, c in cols.items() if c in sub.columns}
    best = max(meds, key=lambda k: meds[k] if np.isfinite(meds[k]) else -1e9)
    return best, meds


def fmt_pct(x: float, digits: int = 1) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{100.0 * x:.{digits}f}%"


def fmt_n(x: float, digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{digits}f}"


def build_markdown(df: pd.DataFrame, html_dir: Path, skipped: list[str]) -> str:
    n = len(df)
    as_of = str(df["end"].max()) if n else ""
    lines: list[str] = []
    lines.append("# 图12 六状态：20260915 全品种回测（状态 × 形态）")
    lines.append("")
    lines.append(
        f"> 目的：把 [六状态趋势划分_八品种验证.md](六状态趋势划分_八品种验证.md) §7 的"
        f"「状态 × 下跌形态」二维映射扩到 20260915_from_listing 全池非债券品种"
    )
    rel = html_dir
    try:
        rel = html_dir.relative_to(_PROJECT_ROOT)
    except ValueError:
        pass
    lines.append(
        f"> 数据：`{rel.as_posix()}` 下 adaptive HTML（前复权，截止 {as_of}）；"
        f"状态重算 `classify_six_states`；动作 T 日收盘 → T+1 收盘生效，不计手续费"
    )
    lines.append(
        "> 复现：`PYTHONPATH=. python src/etf_daily/scripts/backtest_fig12_six_states_pool.py "
        "--html-dir ~/etf-daily-output/temp/plotly_outputs/20260915_from_listing --jobs 8 "
        "--write-md documents/图12六状态_20260915全品种回测_状态x形态.md`"
    )
    lines.append(
        "> 相关：[V3否决规则_七品种回测.md](V3否决规则_七品种回测.md)、"
        "[图12六状态_切换点买卖与三品种交叉验证.md](图12六状态_切换点买卖与三品种交叉验证.md)"
    )
    lines.append("")
    lines.append("## 0. 样本")
    lines.append("")
    lines.append(
        f"共 **{n}** 只非债券 ETF。剔除债券："
        + ("、".join(skipped) if skipped else "无")
        + "。"
    )
    lines.append("")
    lines.append("形态判定（前三类与 §7.2 / V3 §6.3 对齐；第四类为本轮扩样新增）：")
    lines.append("")
    lines.append("- **深熊阴跌型**：买入持有回撤 ≥ 45%，且 ②空仓夏普提升 ≥ 0.07")
    lines.append(
        "- **V 型急跌快修复型**：买入持有回撤 ≥ 45%，②空仓不升反降（Δ≤−0.02）而 ③空仓提升 ≥ 0.08；"
        "或深回撤但只有 ③ 空仓改善"
    )
    lines.append(
        "- **高波动成长型**（新增）：未达深熊/V 回撤门槛，但样本 < 1000 个交易日（约 4 年）"
        "且 ②空仓夏普提升 ≥ 0.20——年轻、波动大、刀锋日空仓有用"
    )
    lines.append("- **平稳慢牛型**：其余（②/③ 空仓几乎零和 / 负和，或长样本深回撤但空仓不赚）")
    lines.append("")

    counts = df["morph"].value_counts()
    lines.append("### 0.1 形态分组计数")
    lines.append("")
    lines.append("| 形态 | 只数 | 占比 |")
    lines.append("|---|---|---|")
    for m in MORPH_ORDER:
        k = int(counts.get(m, 0))
        lines.append(f"| {m} | {k} | {k / n:.1%} |")
    lines.append("")

    lines.append("### 0.2 分组名单")
    lines.append("")
    for m in MORPH_ORDER:
        sub = df[df["morph"] == m].sort_values("mdd")
        lines.append(f"**{m}**（{len(sub)} 只）")
        lines.append("")
        lines.append("| 代码 | 名称 | 区间 | 持有夏普 | 回撤 | ②空仓Δ夏普 | ③空仓Δ夏普 |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['code']} | {r['name']} | {r['start']}→{r['end']} | "
                f"{fmt_n(r['sharpe_bh'])} | {fmt_pct(r['mdd'])} | "
                f"{fmt_n(r[f'dsharpe_{STATE_KNIFE}_empty'])} | "
                f"{fmt_n(r[f'dsharpe_{STATE_DOWN}_empty'])} |"
            )
        lines.append("")

    mdd_pct = f"{100.0 * MDD_DEEP:.0f}"
    lift = f"{SHARPE_LIFT:.2f}"
    g_d2 = f"{GROWTH_D2:.2f}"
    g_bars = str(GROWTH_MAX_BARS)

    lines.append("## 1. 这张表怎么指导操作（每日决策树）")
    lines.append("")
    lines.append(
        "6×4 表动作不统一是结论本身：动作 = f（状态， 形态），不能压成一套买卖。"
        "落地方式是**两步查表**——形态标签一次性定死，每天只读图12 当前状态。"
    )
    lines.append("")
    lines.append(
        f"### 1.1 第 0 步（一次性，约每季度复核）：给持仓 ETF 定形态标签"
    )
    lines.append("")
    lines.append(
        "判定顺序与 `classify_morph` 一致（阈值来自代码常量，不要手改）："
    )
    lines.append("")
    lines.append(
        f"1. 买入持有回撤 ≥ {mdd_pct}% 且 ②空仓Δ夏普 ≥ {lift} → **深熊阴跌型**"
    )
    lines.append(
        f"2. 买入持有回撤 ≥ {mdd_pct}% 且 ②空仓亏（Δ≤−0.02）、③空仓赚（Δ≥{lift}）"
        " → **V 型急跌快修复型**（深回撤但只有 ③ 空仓改善也归此类）"
    )
    lines.append(
        f"3. 样本 < {g_bars} 个交易日（约 4 年）且 ②空仓Δ夏普 ≥ {g_d2} → **高波动成长型**"
    )
    lines.append("4. 其余 → **平稳慢牛型**")
    lines.append("")
    lines.append(
        "标签以 `~/etf-daily-output/temp/plotly_outputs/20260915_from_listing/fig12_six_state_backtest/per_etf.csv` "
        "的 `morph` 列为准（§0.2 名单即该列）。"
        "新上市品种满 500 个交易日前标签按「暂定」处理，满 500 日再跑一次回测定死。"
    )
    lines.append("")
    lines.append("### 1.2 第 1～2 步（每日收盘后）：读状态、查列、执行")
    lines.append("")
    lines.append(
        "1. 打开该 ETF 的 from_listing HTML，看图12 标题里的「当前状态」"
        "（或背景色最右一格）。"
    )
    lines.append(
        "2. 在下方 6×4 可执行动作表中，找 **该 ETF 形态列 × 当前状态行**，T+1 收盘生效。"
    )
    lines.append("")
    lines.append("```mermaid")
    lines.append("flowchart TD")
    lines.append("  startNode[每日收盘后打开图12]")
    lines.append("  startNode --> stateNode[读当前状态]")
    lines.append("  stateNode --> morphNode[查该ETF形态列]")
    lines.append("  morphNode --> s2{是刀锋或下行?}")
    lines.append("  s2 -->|否_过热修复上行震荡| holdNode[不追或不动作]")
    lines.append("  s2 -->|是_急跌刀锋| s2m{形态}")
    lines.append("  s2m -->|深熊或成长| empty2[空仓V3]")
    lines.append("  s2m -->|V型| half2[持有或半仓]")
    lines.append("  s2m -->|平稳| hold2[持有]")
    lines.append("  s2 -->|是_趋势下行| s3m{形态}")
    lines.append("  s3m -->|V型| empty3[空仓]")
    lines.append("  s3m -->|深熊| split3[视个股]")
    lines.append("  s3m -->|成长或平稳| hold3[持有]")
    lines.append("```")
    lines.append("")
    lines.append("### 1.3 第 3 步（纪律）：只有两行会改仓位")
    lines.append("")
    lines.append(
        "①过热 / ④底部修复 / ⑤趋势上行 / ⑥震荡 在四个形态里都是「不追」或「持有」。"
        "**日常只对 ②急跌刀锋、③趋势下行 动手。**"
        "看到①④⑤⑥不要加仓、也不要因为它们空仓。"
    )
    lines.append("")
    lines.append(
        "高相关组合额外否决（见 [图12六状态_切换点买卖与三品种交叉验证.md]"
        "(图12六状态_切换点买卖与三品种交叉验证.md) §6）："
        "SH515880 通信 / SH588850 科创机械 / SH588170 科创半导体 一起看时，"
        "若通信当天切到 ②急跌刀锋，另外两只即使自己是①或⑤，买入信号也要降级等待。"
    )
    lines.append("")
    lines.append(
        "按此表做状态段回测的胜率见 [图12决策树回测_形态x状态胜率.md]"
        "(图12决策树回测_形态x状态胜率.md)。"
    )
    lines.append("")

    lines.append("## 2. 状态 × 形态：组内中位夏普最优动作")
    lines.append("")
    lines.append(
        "对每个形态组、每个状态，比较该组内「该状态日空仓 / 半仓 / 满仓」的中位夏普，取最高者。"
        "满仓 = 始终持有（该状态不动作）。"
    )
    lines.append("")
    header = "| 状态 | " + " | ".join(MORPH_ORDER) + " |"
    lines.append(header)
    lines.append("|---|---|---|---|---|")
    for s in STATE_ORDER:
        cells = [s]
        for m in MORPH_ORDER:
            sub = df[df["morph"] == m]
            if sub.empty:
                cells.append("—")
                continue
            best, meds = median_action(sub, s)
            detail = "/".join(f"{ACTION_CN[a][0]}{fmt_n(meds[ACTION_CN[a]])}" for a in ACTIONS)
            cells.append(f"**{best}**（{detail}）")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("括号内为组内中位夏普：空 / 半 / 满。")
    lines.append("")
    lines.append("据此整理的**可执行动作表**（6 状态 × 4 形态；组中位 + 组内一致性）：")
    lines.append("")
    lines.append("| 状态 | 深熊阴跌型 | V 型急跌快修复型 | 高波动成长型 | 平稳慢牛型 |")
    lines.append("|---|---|---|---|---|")
    lines.append("| ①过热 | 不追、减仓 | 不追 | 不追 | 不动作 |")
    lines.append("| ②急跌刀锋 | **空仓（V3）** | 持有或半仓（可配 8% trailing 止损） | **空仓（V3）** | 持有 |")
    lines.append("| ③趋势下行 | 视个股（组内分裂） | **空仓** | 持有 | 持有 |")
    lines.append("| ④底部修复 | 持有 | 持有 | 持有 | 持有 |")
    lines.append("| ⑤趋势上行 | 持有 | 持有 | 持有 | 持有 |")
    lines.append("| ⑥震荡 | 持有 | 持有 | 持有 | 持有 |")
    lines.append("")
    lines.append(
        "高波动成长型与深熊型在 ② 上动作相同（空仓），差别是：回撤还没走到 45%，"
        "③ 空仓通常不赚（短端急跌后是 V 型反弹），所以只否决刀锋、不否决阴跌确认。"
    )
    lines.append("")

    lines.append("## 3. 各状态最优动作在组内的票数分布")
    lines.append("")
    lines.append("看组内是否一致：每只票自己的最优动作（夏普最高）计一票。")
    lines.append("")
    for m in MORPH_ORDER:
        sub = df[df["morph"] == m]
        lines.append(f"### {m}（n={len(sub)}）")
        lines.append("")
        lines.append("| 状态 | 空仓 | 半仓 | 满仓 | 组内多数 |")
        lines.append("|---|---|---|---|---|")
        for s in STATE_ORDER:
            vc = sub[f"best_{s}"].value_counts()
            n0 = int(vc.get("空仓", 0))
            n5 = int(vc.get("半仓", 0))
            n1 = int(vc.get("满仓", 0))
            ranked = sorted(
                (("空仓", n0), ("半仓", n5), ("满仓", n1)),
                key=lambda x: -x[1],
            )
            majority = ranked[0][0]
            if ranked[0][1] == ranked[1][1]:
                majority = f"{ranked[0][0]}/{ranked[1][0]}（平）"
            lines.append(f"| {s} | {n0} | {n5} | {n1} | {majority} |")
        lines.append("")

    lines.append("## 4. 与八品种 §7.3 对照")
    lines.append("")
    lines.append(
        "八品种表（[六状态趋势划分_八品种验证.md](六状态趋势划分_八品种验证.md) §7.3）建议："
        "深熊型 ②③空仓；V 型 ②持有（可加 8% 止损）、③空仓；平稳型全程持有。"
    )
    lines.append("")
    lines.append("本次 62 只扩样后：")
    lines.append("")
    # Compare narrative from actual matrix
    bear = df[df["morph"] == MORPH_BEAR]
    v = df[df["morph"] == MORPH_V]
    calm = df[df["morph"] == MORPH_CALM]
    notes = []
    if not bear.empty:
        b2, _ = median_action(bear, STATE_KNIFE)
        b3, _ = median_action(bear, STATE_DOWN)
        notes.append(f"- 深熊阴跌型（{len(bear)} 只）：②最优 **{b2}**，③最优 **{b3}**")
    if not v.empty:
        v2, _ = median_action(v, STATE_KNIFE)
        v3, _ = median_action(v, STATE_DOWN)
        notes.append(f"- V 型急跌快修复型（{len(v)} 只）：②最优 **{v2}**，③最优 **{v3}**")
    growth = df[df["morph"] == MORPH_GROWTH]
    if not growth.empty:
        g2, _ = median_action(growth, STATE_KNIFE)
        g3, _ = median_action(growth, STATE_DOWN)
        notes.append(f"- 高波动成长型（{len(growth)} 只）：②最优 **{g2}**，③最优 **{g3}**")
    if not calm.empty:
        c2, _ = median_action(calm, STATE_KNIFE)
        c5, _ = median_action(calm, STATE_UP)
        notes.append(f"- 平稳慢牛型（{len(calm)} 只）：②最优 **{c2}**，⑤最优 **{c5}**")
    lines.extend(notes)
    lines.append("")
    lines.append("八品种锚点在本组中的落点：")
    lines.append("")
    lines.append("- **中概互联 SH513050、通信 SH515880** → 深熊阴跌型（与 §7.2 一致）")
    lines.append("- **能源化工 SZ159981** → V 型急跌快修复型（②空仓亏、③空仓赚，与 §7.2 一致）")
    lines.append("- **黄金 / 银行 / 红利 / 豆粕 / 沪深300** → 平稳慢牛型（与 §7.2 一致；沪深300 回撤虽 −46%，但 ②③ 空仓都降夏普，故不进深熊）")
    lines.append("- 八品种里没有高波动成长型锚点——这是 62 只扩样后从原「平稳」里拆出来的第四类")
    lines.append("")
    lines.append(
        "和三分类的差别：原平稳组里科创机械、中韩半导体、汽车、航空航天等"
        "②空仓夏普能抬 0.20~0.58，但回撤不到 45%，被第四类收走；"
        "拆完之后剩余平稳型的 ② 更接近「不要乱动」。"
        "V 型仍是 2 只（能源化工、教育）；深熊型 ② 八票全空仓，③ 组内 4/4 分裂。"
    )
    lines.append("")

    lines.append("## 5. 局限")
    lines.append("")
    lines.append("1. 不计交易成本；空仓/半仓的夏普优势在实盘会打折。")
    lines.append("2. 年轻品种（上市不足 2 年）回撤和夏普噪声大，形态标签应打折。")
    lines.append("3. 形态阈值（回撤 45%、深熊 Δ夏普 0.07、成长型 Δ夏普 0.20 且 <1000 日）未再网格搜索。")
    lines.append("4. 「满仓最优」在平稳型上往往只是「不要乱动」，不是该状态有超额。")
    lines.append("5. 债券已剔除，商品 / 海外 / 行业混在同一池，形态组内仍有异质性。")
    lines.append(
        "6. 第四类「高波动成长型」样本短（<1000 日），②空仓Δ≥0.20 可能含噪声；"
        "家电 ETF（1076 日、②Δ=+0.27）因超过日数门槛仍留在平稳型。"
    )
    lines.append(
        "7. 红利 ETF（SH510880）买入持有回撤 −75% 仍归平稳型："
        "②③ 空仓几乎不提升夏普（2008 式长熊，V3 的 10 日确认躲不掉主跌）。"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    html_dir = args.html_dir.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else html_dir / "fig12_six_state_backtest"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    skipped = []
    for p in sorted(html_dir.glob("regime_transition_*_adaptive.html")):
        if not _FNAME_RE.match(p.name):
            continue
        code, name = parse_html_label(p)
        if is_bond(code, name):
            skipped.append(f"{code} {name}")

    if args.from_csv is not None:
        df = pd.read_csv(args.from_csv.expanduser().resolve())
        df["morph"] = df.apply(classify_morph, axis=1)
        df = df.sort_values(["morph", "mdd", "code"]).reset_index(drop=True)
        df.to_csv(out_dir / "per_etf.csv", index=False)
        md = build_markdown(df, html_dir, skipped)
        md_path = (
            args.write_md.expanduser().resolve()
            if args.write_md is not None
            else _PROJECT_ROOT / "documents" / "图12六状态_20260915全品种回测_状态x形态.md"
        )
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")
        print(f"[OK] n={len(df)} from-csv")
        print(df["morph"].value_counts().to_string())
        print(f"[OK] md={md_path}")
        return 0

    files = list_equity_html(html_dir)
    if not files:
        print("[ERROR] no equity HTML", file=sys.stderr)
        return 2

    jobs = max(1, int(args.jobs))
    rows: list[dict] = []
    if jobs == 1 or len(files) == 1:
        futs_results = [_worker(str(p)) for p in files]
        rows = futs_results
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(_worker, str(p)): p for p in files}
            done = 0
            for fut in as_completed(futs):
                rec = fut.result()
                rows.append(rec)
                done += 1
                flag = "OK" if rec.get("ok") else "FAIL"
                print(f"[{done}/{len(files)}] {flag} {rec.get('code')} {rec.get('name', '')}")
                if not rec.get("ok"):
                    print(rec.get("error", ""), file=sys.stderr)

    failed = [r for r in rows if not r.get("ok")]
    ok_rows = [r for r in rows if r.get("ok")]
    pd.DataFrame(failed).to_csv(out_dir / "errors.csv", index=False)
    if not ok_rows:
        print("[ERROR] all failed", file=sys.stderr)
        return 1

    df = pd.DataFrame(ok_rows)
    df["morph"] = df.apply(classify_morph, axis=1)
    df = df.sort_values(["morph", "mdd", "code"]).reset_index(drop=True)
    df.to_csv(out_dir / "per_etf.csv", index=False)

    md = build_markdown(df, html_dir, skipped)
    md_path = (
        args.write_md.expanduser().resolve()
        if args.write_md is not None
        else _PROJECT_ROOT / "documents" / "图12六状态_20260915全品种回测_状态x形态.md"
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
    print(f"[OK] n={len(df)} fail={len(failed)}")
    print(df["morph"].value_counts().to_string())
    print(f"[OK] per_etf={out_dir / 'per_etf.csv'}")
    print(f"[OK] md={md_path}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
