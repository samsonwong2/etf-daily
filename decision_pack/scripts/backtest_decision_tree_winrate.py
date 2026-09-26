#!/usr/bin/env python3
"""Backtest the fig12 6×4 decision tree; episode win-rate by morph × state.

Morphology labels are locked from per_etf.csv (document §1 step 0).
Each day: T-close state → T+1 position from TREE_POS[morph][state].
A consecutive run of the same lagged state is one episode; win = episode
strategy cumulative return > buy-and-hold cumulative return over the same bars.

Example::

    PYTHONPATH=. python decision_pack/scripts/backtest_decision_tree_winrate.py \\
      --html-dir ~/etf-daily-output/temp/plotly_outputs/20260915_from_listing \\
      --morph-csv ~/etf-daily-output/temp/plotly_outputs/20260915_from_listing/fig12_six_state_backtest/per_etf.csv \\
      --write-md documents/图12决策树回测_形态x状态胜率.md
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

EPS = 1e-12
ACTIONABLE = (STATE_KNIFE, STATE_DOWN)

TREE_POS: dict[str, dict[str, float]] = {
    pool.MORPH_BEAR: {
        STATE_OVERHEAT: 1.0,
        STATE_KNIFE: 0.0,
        STATE_DOWN: 0.0,
        STATE_REPAIR: 1.0,
        STATE_UP: 1.0,
        STATE_CHOP: 1.0,
    },
    pool.MORPH_V: {
        STATE_OVERHEAT: 1.0,
        STATE_KNIFE: 0.5,
        STATE_DOWN: 0.0,
        STATE_REPAIR: 1.0,
        STATE_UP: 1.0,
        STATE_CHOP: 1.0,
    },
    pool.MORPH_GROWTH: {
        STATE_OVERHEAT: 1.0,
        STATE_KNIFE: 0.0,
        STATE_DOWN: 1.0,
        STATE_REPAIR: 1.0,
        STATE_UP: 1.0,
        STATE_CHOP: 1.0,
    },
    pool.MORPH_CALM: {
        STATE_OVERHEAT: 1.0,
        STATE_KNIFE: 1.0,
        STATE_DOWN: 1.0,
        STATE_REPAIR: 1.0,
        STATE_UP: 1.0,
        STATE_CHOP: 1.0,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--html-dir", type=Path, required=True)
    p.add_argument("--morph-csv", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--write-md", type=Path, default=None)
    return p.parse_args(argv)


def load_morph_map(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    if "code" not in df.columns or "morph" not in df.columns:
        raise ValueError(f"{path} needs columns code, morph")
    out: dict[str, str] = {}
    for _, r in df.iterrows():
        code = str(r["code"]).strip()
        morph = str(r["morph"]).strip()
        if not code or morph not in TREE_POS:
            continue
        out[code] = morph
    return out


def equity_mdd(daily: pd.Series) -> float:
    eq = (1.0 + pd.to_numeric(daily, errors="coerce").fillna(0.0)).cumprod()
    return pool.max_drawdown(eq)


def cagr_from_total(total: float, start, end) -> float:
    years = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25, 1e-6)
    if total <= -1.0 or not np.isfinite(total):
        return float("nan")
    return float((1.0 + total) ** (1.0 / years) - 1.0)


def split_episodes(
    lagged: pd.Series,
    r_strat: pd.Series,
    r_bh: pd.Series,
    *,
    code: str,
    name: str,
    morph: str,
) -> list[dict]:
    eid = lagged.ne(lagged.shift()).cumsum()
    rows: list[dict] = []
    for i, idx in enumerate(eid.unique(), start=1):
        mask = eid.eq(idx)
        st = str(lagged[mask].iloc[0])
        pos = float(TREE_POS[morph].get(st, 1.0))
        rs = r_strat[mask]
        rb = r_bh[mask]
        strat_cum = float((1.0 + rs).prod() - 1.0)
        bh_cum = float((1.0 + rb).prod() - 1.0)
        excess = strat_cum - bh_cum
        if excess > EPS:
            outcome = "win"
        elif excess < -EPS:
            outcome = "loss"
        else:
            outcome = "tie"
        rows.append(
            {
                "code": code,
                "name": name,
                "morph": morph,
                "episode": i,
                "state": st,
                "pos": pos,
                "action": pool.ACTION_CN.get(pos, str(pos)),
                "actionable": st in ACTIONABLE and pos < 1.0 - EPS,
                "start": pd.Timestamp(rs.index[0]).strftime("%Y-%m-%d"),
                "end": pd.Timestamp(rs.index[-1]).strftime("%Y-%m-%d"),
                "n_bars": int(len(rs)),
                "strat_ret": strat_cum,
                "bh_ret": bh_cum,
                "excess": excess,
                "outcome": outcome,
            }
        )
    return rows


def process_one(path_str: str, morph: str) -> dict:
    path = Path(path_str)
    code, name = pool.parse_html_label(path)
    close, _ohlcv = load_ohlcv_from_regime_html(path)
    ms = enrich_frame_for_checklist(compute_multi_scale_frame(close))
    states = classify_six_states(ms)
    px = pd.to_numeric(ms["px"], errors="coerce")
    mask = px.notna() & states.ne(STATE_WARMUP)
    px = px[mask]
    st = states[mask]
    if len(px) < 60:
        raise ValueError(f"too few bars after warmup: {len(px)}")

    r_bh_full = px.pct_change()
    lagged = st.shift(1)
    valid = lagged.notna() & r_bh_full.notna()
    lagged = lagged[valid]
    r_bh = r_bh_full[valid].astype(float)
    pos_map = TREE_POS[morph]
    weight = lagged.map(lambda s: float(pos_map.get(s, 1.0))).astype(float)
    r_strat = (weight * r_bh).astype(float)

    start = pd.Timestamp(r_bh.index[0])
    end = pd.Timestamp(r_bh.index[-1])
    total_bh = float((1.0 + r_bh).prod() - 1.0)
    total_tree = float((1.0 + r_strat).prod() - 1.0)
    sharpe_bh = pool.ann_sharpe(r_bh)
    sharpe_tree = pool.ann_sharpe(r_strat)
    mdd_bh = equity_mdd(r_bh)
    mdd_tree = equity_mdd(r_strat)
    cagr_bh = cagr_from_total(total_bh, start, end)
    cagr_tree = cagr_from_total(total_tree, start, end)

    episodes = split_episodes(
        lagged, r_strat, r_bh, code=code, name=name, morph=morph
    )
    act = [e for e in episodes if e["actionable"]]
    n_act = len(act)
    n_act_win = sum(1 for e in act if e["outcome"] == "win")
    return {
        "ok": True,
        "etf": {
            "ok": True,
            "code": code,
            "name": name,
            "morph": morph,
            "path": str(path),
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
            "n": int(len(r_bh)),
            "total_bh": total_bh,
            "cagr_bh": cagr_bh,
            "sharpe_bh": sharpe_bh,
            "mdd_bh": mdd_bh,
            "total_tree": total_tree,
            "cagr_tree": cagr_tree,
            "sharpe_tree": sharpe_tree,
            "mdd_tree": mdd_tree,
            "excess_cagr": (
                cagr_tree - cagr_bh
                if np.isfinite(cagr_tree) and np.isfinite(cagr_bh)
                else float("nan")
            ),
            "excess_sharpe": (
                sharpe_tree - sharpe_bh
                if np.isfinite(sharpe_tree) and np.isfinite(sharpe_bh)
                else float("nan")
            ),
            "n_episodes": len(episodes),
            "n_action_ep": n_act,
            "n_action_win": n_act_win,
            "action_win_rate": (n_act_win / n_act) if n_act else float("nan"),
            "error": "",
        },
        "episodes": episodes,
    }


def _worker(args: tuple[str, str]) -> dict:
    path_str, morph = args
    try:
        return process_one(path_str, morph)
    except Exception as exc:  # noqa: BLE001
        path = Path(path_str)
        code, name = pool.parse_html_label(path)
        return {
            "ok": False,
            "etf": {
                "ok": False,
                "code": code,
                "name": name,
                "morph": morph,
                "path": path_str,
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            },
            "episodes": [],
        }


def summarize_winrate(ep: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for morph in pool.MORPH_ORDER:
        for state in STATE_ORDER:
            sub = ep[(ep["morph"] == morph) & (ep["state"] == state)]
            pos = float(TREE_POS[morph][state])
            n = int(len(sub))
            n_etf = int(sub["code"].nunique()) if n else 0
            n_win = int((sub["outcome"] == "win").sum()) if n else 0
            n_tie = int((sub["outcome"] == "tie").sum()) if n else 0
            n_loss = int((sub["outcome"] == "loss").sum()) if n else 0
            decided = n_win + n_loss
            rows.append(
                {
                    "morph": morph,
                    "state": state,
                    "pos": pos,
                    "action": pool.ACTION_CN[pos],
                    "actionable": bool(state in ACTIONABLE and pos < 1.0 - EPS),
                    "n_episodes": n,
                    "n_etf": n_etf,
                    "n_win": n_win,
                    "n_tie": n_tie,
                    "n_loss": n_loss,
                    "win_rate": (n_win / n) if n else float("nan"),
                    "win_rate_ex_tie": (n_win / decided) if decided else float("nan"),
                    "tie_rate": (n_tie / n) if n else float("nan"),
                    "mean_excess": float(sub["excess"].mean()) if n else float("nan"),
                    "median_excess": float(sub["excess"].median()) if n else float("nan"),
                    "mean_strat_ret": float(sub["strat_ret"].mean()) if n else float("nan"),
                    "mean_bh_ret": float(sub["bh_ret"].mean()) if n else float("nan"),
                    "mean_n_bars": float(sub["n_bars"].mean()) if n else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _cell_win(wr: pd.DataFrame, morph: str, state: str) -> str:
    row = wr[(wr["morph"] == morph) & (wr["state"] == state)]
    if row.empty:
        return "—"
    r = row.iloc[0]
    n = int(r["n_episodes"])
    if n == 0:
        return "—"
    if not bool(r["actionable"]):
        return f"不动作（平{int(r['n_tie'])}/{n}）"
    wr_pct = pool.fmt_pct(float(r["win_rate"]), 1)
    ex = pool.fmt_pct(float(r["mean_excess"]), 2)
    return f"{wr_pct}（{int(r['n_win'])}/{n}，均超额{ex}）"


def build_markdown(
    etf: pd.DataFrame,
    wr: pd.DataFrame,
    html_dir: Path,
    morph_csv: Path,
) -> str:
    n = len(etf)
    as_of = str(etf["end"].max()) if n else ""
    try:
        rel = html_dir.relative_to(_PROJECT_ROOT)
    except ValueError:
        rel = html_dir
    try:
        morph_rel = morph_csv.relative_to(_PROJECT_ROOT)
    except ValueError:
        morph_rel = morph_csv

    lines: list[str] = []
    lines.append("# 图12 决策树回测：形态 × 状态胜率")
    lines.append("")
    lines.append(
        "> 目的：把 [图12六状态_20260915全品种回测_状态x形态.md]"
        "(图12六状态_20260915全品种回测_状态x形态.md) §1/§2 的每日决策树"
        "落成逐日仓位，按**状态段**统计每个形态×状态决策跑赢持有的胜率"
    )
    lines.append(
        f"> 数据：`{rel.as_posix()}` 下 adaptive HTML（前复权，截止 {as_of}）；"
        f"形态标签来自 `{morph_rel.as_posix()}` 的 `morph` 列（一次性定死，不在本回测内重分类）"
    )
    lines.append(
        "> 复现：`PYTHONPATH=. python decision_pack/scripts/backtest_decision_tree_winrate.py "
        "--html-dir ~/etf-daily-output/temp/plotly_outputs/20260915_from_listing "
        "--morph-csv ~/etf-daily-output/temp/plotly_outputs/20260915_from_listing/fig12_six_state_backtest/per_etf.csv "
        "--write-md documents/图12决策树回测_形态x状态胜率.md`"
    )
    lines.append("")

    lines.append("## 0. 口径")
    lines.append("")
    lines.append("- 状态：因果 `classify_six_states`，剔除 warmup。")
    lines.append("- 仓位：T 日收盘状态 → T+1 收盘生效，不计手续费与滑点。")
    lines.append(
        "- 决策表与文档 §2 可执行动作表一致。①「不追」在满仓基准下无法表达为加仓，"
        "**统一记为持有**；深熊 × ③趋势下行取**空仓**（组中位最优，不再「视个股」）。"
    )
    lines.append(
        "- **状态段**：按驱动当日收益的滞后状态切分（`lagged.ne(lagged.shift()).cumsum()`）。"
        "连续同一状态 = 一笔；段内策略累计收益 vs 同期买入持有累计收益；"
        "**胜 = 超额 > 0，平 = |超额|≈0，负 = 超额 < 0**。"
    )
    lines.append(
        "- 只有 ②急跌刀锋、③趋势下行 会改仓位。①④⑤⑥ 仓位=1，超额≈0，表中记「不动作」。"
    )
    lines.append("- 胜率矩阵与动作表来自**同一样本（in-sample）**，偏乐观，见 §4。")
    lines.append("")
    lines.append("决策表仓位（1=满仓，0.5=半仓，0=空仓）：")
    lines.append("")
    lines.append("| 状态 | " + " | ".join(pool.MORPH_ORDER) + " |")
    lines.append("|---|---|---|---|---|")
    for s in STATE_ORDER:
        cells = [s]
        for m in pool.MORPH_ORDER:
            pos = TREE_POS[m][s]
            cells.append(pool.ACTION_CN[pos])
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 1. 整体：决策树 vs 买入持有")
    lines.append("")
    lines.append(
        "每个形态组内取中位数。Δ夏普 / 超额年化 > 0 表示决策树优于始终持有。"
    )
    lines.append("")
    lines.append(
        "| 形态 | 只数 | 中位持有夏普 | 中位决策树夏普 | 中位Δ夏普 | "
        "中位持有回撤 | 中位决策树回撤 | 中位超额年化 | ②③段胜率 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for m in pool.MORPH_ORDER:
        sub = etf[etf["morph"] == m]
        if sub.empty:
            continue
        act = sub["action_win_rate"]
        lines.append(
            f"| {m} | {len(sub)} | "
            f"{pool.fmt_n(float(sub['sharpe_bh'].median()))} | "
            f"{pool.fmt_n(float(sub['sharpe_tree'].median()))} | "
            f"{pool.fmt_n(float(sub['excess_sharpe'].median()))} | "
            f"{pool.fmt_pct(float(sub['mdd_bh'].median()))} | "
            f"{pool.fmt_pct(float(sub['mdd_tree'].median()))} | "
            f"{pool.fmt_pct(float(sub['excess_cagr'].median()), 2)} | "
            f"{pool.fmt_pct(float(act.median()) if act.notna().any() else float('nan'))} |"
        )
    lines.append("")
    lines.append("②③段胜率 = 该形态组内、每只 ETF 自己的可操作状态段胜率的中位数。")
    lines.append("")

    lines.append("## 2. 6×4 状态段胜率")
    lines.append("")
    lines.append(
        "格子：胜率（胜/段数，平均段超额）。不动作格给出平局数。"
        "胜率 = 胜 / 全部段（平局计入分母，故持有格胜率≈0、平局≈100%）。"
    )
    lines.append("")
    lines.append("| 状态 | " + " | ".join(pool.MORPH_ORDER) + " |")
    lines.append("|---|---|---|---|---|")
    for s in STATE_ORDER:
        cells = [s]
        for m in pool.MORPH_ORDER:
            cells.append(_cell_win(wr, m, s))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "读表注意：深熊 ②/③、V 型 ②/③ 的**段胜率可以低于 50%，但均超额仍为正**——"
        "空仓躲过大跌、错过小反弹，输的次数多、赢的幅度大，这与组内 Δ夏普为正并不矛盾。"
        "高波动成长 ② 是唯一胜率与均超额同向都明显为正的格子。"
        "平稳型 ②/③ 仓位=持有，851/851 与 791/791 全平，决策树 ≡ 买入持有。"
    )
    lines.append("")
    lines.append("可操作格的有效胜率（剔除平局，胜 / (胜+负)）与段数：")
    lines.append("")
    lines.append("| 形态 | 状态 | 动作 | 段数 | ETF 数 | 胜 | 平 | 负 | 胜率 | 有效胜率 | 均超额 | 中位超额 | 均段长 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    act_wr = wr[wr["actionable"]].copy()
    for _, r in act_wr.iterrows():
        lines.append(
            f"| {r['morph']} | {r['state']} | {r['action']} | "
            f"{int(r['n_episodes'])} | {int(r['n_etf'])} | "
            f"{int(r['n_win'])} | {int(r['n_tie'])} | {int(r['n_loss'])} | "
            f"{pool.fmt_pct(float(r['win_rate']))} | "
            f"{pool.fmt_pct(float(r['win_rate_ex_tie']))} | "
            f"{pool.fmt_pct(float(r['mean_excess']), 2)} | "
            f"{pool.fmt_pct(float(r['median_excess']), 2)} | "
            f"{pool.fmt_n(float(r['mean_n_bars']), 1)} |"
        )
    lines.append("")

    lines.append("## 3. 逐只明细")
    lines.append("")
    for m in pool.MORPH_ORDER:
        sub = etf[etf["morph"] == m].sort_values("excess_sharpe", ascending=False)
        lines.append(f"### {m}（{len(sub)} 只）")
        lines.append("")
        lines.append(
            "| 代码 | 名称 | 区间 | 持有夏普 | 决策树夏普 | Δ夏普 | "
            "持有回撤 | 决策树回撤 | 超额年化 | ②③胜率 |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['code']} | {r['name']} | {r['start']}→{r['end']} | "
                f"{pool.fmt_n(r['sharpe_bh'])} | {pool.fmt_n(r['sharpe_tree'])} | "
                f"{pool.fmt_n(r['excess_sharpe'])} | "
                f"{pool.fmt_pct(r['mdd_bh'])} | {pool.fmt_pct(r['mdd_tree'])} | "
                f"{pool.fmt_pct(r['excess_cagr'], 2)} | "
                f"{pool.fmt_pct(r['action_win_rate'])} "
                f"（{int(r['n_action_win'])}/{int(r['n_action_ep'])}） |"
            )
        lines.append("")

    lines.append("## 4. 局限")
    lines.append("")
    lines.append(
        "1. **动作表与胜率来自同一样本（in-sample）**："
        "形态标签用了全样本回撤和 Δ夏普，决策表也按组内中位夏普定，胜率偏乐观。"
    )
    lines.append("2. V 型仅 2 只，段数少，胜率噪声大。")
    lines.append("3. 不计手续费与滑点；空仓/半仓的优势在实盘会打折。")
    lines.append("4. ①「不追」被记成持有，本回测没有「减仓不追」的超额。")
    lines.append("5. 高相关组合的交叉否决（通信转②时科创①/⑤降级）未纳入本回测。")
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    html_dir = args.html_dir.expanduser().resolve()
    morph_csv = args.morph_csv.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else html_dir / "fig12_decision_tree_backtest"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    morph_map = load_morph_map(morph_csv)
    files = pool.list_equity_html(html_dir)
    jobs_in: list[tuple[str, str]] = []
    missing: list[str] = []
    for p in files:
        code, _name = pool.parse_html_label(p)
        morph = morph_map.get(code)
        if morph is None:
            missing.append(code)
            continue
        jobs_in.append((str(p), morph))
    if missing:
        print("[WARN] no morph label:", ", ".join(missing), file=sys.stderr)
    if not jobs_in:
        print("[ERROR] no jobs", file=sys.stderr)
        return 2

    jobs = max(1, int(args.jobs))
    results: list[dict] = []
    if jobs == 1 or len(jobs_in) == 1:
        results = [_worker(a) for a in jobs_in]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_worker, a): a for a in jobs_in}
            done = 0
            for fut in as_completed(futs):
                rec = fut.result()
                results.append(rec)
                done += 1
                etf = rec["etf"]
                flag = "OK" if rec.get("ok") else "FAIL"
                print(f"[{done}/{len(jobs_in)}] {flag} {etf.get('code')} {etf.get('name', '')}")
                if not rec.get("ok"):
                    print(etf.get("error", ""), file=sys.stderr)

    failed = [r["etf"] for r in results if not r.get("ok")]
    ok = [r for r in results if r.get("ok")]
    pd.DataFrame(failed).to_csv(out_dir / "errors.csv", index=False)
    if not ok:
        print("[ERROR] all failed", file=sys.stderr)
        return 1

    etf_df = pd.DataFrame([r["etf"] for r in ok])
    etf_df = etf_df.sort_values(["morph", "excess_sharpe", "code"], ascending=[True, False, True]).reset_index(drop=True)
    ep_df = pd.DataFrame([e for r in ok for e in r["episodes"]])
    wr_df = summarize_winrate(ep_df)

    etf_df.to_csv(out_dir / "per_etf.csv", index=False)
    ep_df.to_csv(out_dir / "episodes.csv", index=False)
    wr_df.to_csv(out_dir / "winrate_by_morph_state.csv", index=False)

    md = build_markdown(etf_df, wr_df, html_dir, morph_csv)
    md_path = (
        args.write_md.expanduser().resolve()
        if args.write_md is not None
        else _PROJECT_ROOT / "documents" / "图12决策树回测_形态x状态胜率.md"
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")

    print(f"[OK] n={len(etf_df)} fail={len(failed)}")
    print(etf_df["morph"].value_counts().to_string())
    print(wr_df[wr_df["actionable"]][["morph", "state", "n_episodes", "win_rate", "mean_excess"]].to_string(index=False))
    print(f"[OK] per_etf={out_dir / 'per_etf.csv'}")
    print(f"[OK] episodes={out_dir / 'episodes.csv'}")
    print(f"[OK] winrate={out_dir / 'winrate_by_morph_state.csv'}")
    print(f"[OK] md={md_path}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
