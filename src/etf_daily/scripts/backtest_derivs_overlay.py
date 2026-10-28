#!/usr/bin/env python3
"""Research pass: futures basis and option IV as filters on rules7.

Signal on close T, fill on close T+1, 5bp a side. The back half of each
ETF restarts flat. A filter is kept only when that half improves on at
least two of SH510300, SH510050, and SH510500.

    python src/etf_daily/scripts/backtest_derivs_overlay.py
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

from etf_daily.derivs.features import PAIRS, build_panel  # noqa: E402
from etf_daily.derivs.fetch import (  # noqa: E402
    close_series,
    load_chain,
    month_ends,
    refresh_chain,
    refresh_market,
)
from etf_daily.derivs.overlay import (  # noqa: E402
    HORIZONS,
    adopted_variants,
    block_masks,
    entry_fills,
    oos_improved,
    path_stats,
    run_window,
    tercile,
)
from etf_daily.lib import rules7  # noqa: E402

CODES = ("SH510300", "SH510050", "SH510500")
DOC_PATH = _PROJECT_ROOT / "docs" / "衍生品信号叠加rules7.md"
START = "2005-01-01"
END = "2026-10-20"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="download market series again")
    parser.add_argument("--skip-chain", action="store_true", help="do not download month-end option snapshots")
    parser.add_argument("--doc", type=Path, default=DOC_PATH)
    return parser.parse_args(argv)


def _load_frames(codes: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    from etf_daily.lib.fair_path_band_signals import load_qfq_close
    from etf_daily.scripts.backtest_fig9_touch import prepare_fig9_touch_frame
    from etf_daily.scripts.run_reusable_extrema_state_machine import build_feature_frame_from_series

    closes: dict[str, pd.Series] = {}
    ohlcvs: dict[str, pd.DataFrame] = {}
    for i, code in enumerate(codes):
        close, ohlcv = load_qfq_close(code, START, END, init_qlib=(i == 0))
        close.index = pd.DatetimeIndex(pd.to_datetime(close.index)).normalize()
        ohlcv.index = pd.DatetimeIndex(pd.to_datetime(ohlcv.index)).normalize()
        closes[code] = close[~close.index.duplicated(keep="last")].sort_index()
        ohlcvs[code] = ohlcv[~ohlcv.index.duplicated(keep="last")].sort_index()
    anchor = closes["SH510300"]
    frames = {}
    for code in codes:
        frame = build_feature_frame_from_series(
            closes[code], ohlcvs[code], anchor, lookback=10
        )
        frames[code] = prepare_fig9_touch_frame(frame)
    return frames


def _panel_for(code: str, close: pd.Series, market: dict[str, pd.DataFrame], chain: pd.DataFrame) -> pd.DataFrame:
    spec = PAIRS[code]
    return build_panel(
        close,
        close_series(market.get(f"future:{spec.future}", pd.DataFrame())),
        close_series(market.get(f"index:{spec.index}", pd.DataFrame())),
        close_series(market.get(f"qvix:{spec.qvix}", pd.DataFrame())),
        chain,
        underlying=code[-6:],
    )


def _metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "total": float(result["total"]),
        "mdd": float(result["mdd"]),
        "calmar": float(result["calmar"]),
        "n_trades": int(result["n_trades"]),
        "exposure": float(result["exposure"]),
    }


def _entry_table(result: dict[str, Any], panel: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    rows = []
    for signal, fill in entry_fills(result):
        if signal not in panel.index:
            continue
        row: dict[str, Any] = {
            "signal": signal,
            "fill": fill,
            "basis_pct": float(panel.at[signal, "basis_pct"]),
            "iv_pct": float(panel.at[signal, "iv_pct"]),
            "skew": float(panel.at[signal, "skew_carry"]),
        }
        row["basis_bucket"] = tercile(row["basis_pct"])
        row["iv_bucket"] = tercile(row["iv_pct"])
        for horizon in HORIZONS:
            stats = path_stats(close, fill, horizon)
            row[f"ret{horizon}"] = np.nan if stats is None else stats[0]
            row[f"mdd{horizon}"] = np.nan if stats is None else stats[1]
        rows.append(row)
    return pd.DataFrame(rows)


def _bucket_lines(entries: pd.DataFrame, column: str, half: pd.Series) -> list[str]:
    lines = []
    if entries.empty:
        return ["这一半没有买入点。"]
    view = entries.loc[half.reindex(entries.index).fillna(False)]
    if view.empty:
        return ["这一半没有买入点。"]
    missing = int(view[column].isna().sum())
    if missing:
        lines.append(f"- 无分位：{missing} 笔。信号日还不满 60 日窗口，或当天没有基差、隐含波动率。")
    for label in ("低", "中", "高"):
        part = view.loc[view[column] == label]
        if part.empty:
            lines.append(f"- {label}：0 笔")
            continue
        bits = [f"n={len(part)}"]
        for horizon in HORIZONS:
            rets = pd.to_numeric(part[f"ret{horizon}"], errors="coerce").dropna()
            mdds = pd.to_numeric(part[f"mdd{horizon}"], errors="coerce").dropna()
            ret_txt = "—" if rets.empty else f"{rets.mean():.1%}"
            mdd_txt = "—" if mdds.empty else f"{mdds.mean():.1%}"
            bits.append(f"{horizon}日均收益 {ret_txt}、均回撤 {mdd_txt}")
        lines.append(f"- {label}：" + "，".join(bits))
    return lines


def _fmt_pct(value: float) -> str:
    return "—" if not np.isfinite(value) else f"{value:.1%}"


def _fmt_num(value: float) -> str:
    return "—" if not np.isfinite(value) else f"{value:.2f}"


def _coverage(panel: pd.DataFrame, column: str) -> str:
    series = panel[column].dropna()
    if series.empty:
        return "无数据"
    return f"{series.index.min().date()} 至 {series.index.max().date()}，{len(series)}/{len(panel)} 日"


def _qvix_vs_atm(panel: pd.DataFrame) -> str:
    both = panel[["iv", "atm_iv"]].dropna()
    if len(both) < 3:
        return "重叠不足，没有比较。"
    gap = (both["atm_iv"] - both["iv"]).abs()
    corr = both["iv"].corr(both["atm_iv"])
    return (
        f"{len(both)} 个重叠日，绝对差中位数 {gap.median():.1%}，"
        f"相关系数 {corr:.2f}。"
    )


def render_doc(
    *,
    panels: dict[str, pd.DataFrame],
    tables: dict[str, list[dict[str, Any]]],
    entries: dict[str, pd.DataFrame],
    mids: dict[str, pd.Timestamp],
    improved: dict[str, dict[str, bool]],
    kept: list[str],
) -> str:
    oos_counts = []
    for code in CODES:
        base = next(row for row in tables[code] if row["variant"] == "基线" and row["window"] == "后半段")
        oos_counts.append(f"{code} {base['n_trades']} 笔")
    oos_text = "、".join(oos_counts)
    if kept:
        verdict = (
            "样本外有改善的过滤是："
            + "、".join(kept)
            + f"。后半段基线是 {oos_text}。改善的定义写在下面。本轮不改 `buy_sell_masks`。"
            "下一轮可以在 rules7 清单加上「基差分位」和「隐含波动率分位」两列，只作参考，不进买卖公式。"
        )
    else:
        verdict = (
            f"后半段基线只有 {oos_text}。过滤再去掉几笔之后，没有一只还留得住 5 笔，"
            "也没有两只同时把回撤或 Calmar 做得更好。全样本上有的过滤回撤更浅，"
            "那段和规则是同一段历史，不拿来决定是否接入。"
            "到这里停止，不把这些列写进 rules7 清单，也不改买卖公式。"
        )
    lines = [
        "# 衍生品信号叠加 rules7",
        "",
        "## 结论",
        "",
        verdict,
        "",
        "买卖点仍是图9下轨 aux_edge（规则含 B1 时加上 B1）、图9或图8上轨卖出、G1 持有、回落止损。"
        "期货基差和期权隐含波动率只作为风险状态。信号日收盘确认，次日收盘成交，每边成本 5 个基点。",
        "",
        "后半段从这只 ETF 行情的中点起，状态机重新空仓开始。这是样本外。"
        "改善指：最大回撤至少浅 1 个百分点，或 Calmar 至少高 0.05；"
        "同时交易次数不少于基线的一半，且不少于 5 笔。"
        "三只里至少两只成立，这个过滤才算通过。分位阈值是事先定的 80% 和 20%，没有搜索。",
        "",
        "## 数据",
        "",
        "基差是主力连续期货收盘除以指数收盘再减 1，再按距离当月第三个周五的天数年化。"
        "到期前 5 个自然日改用下一个月，当作主力已经换月。"
        "期货相对指数的对数收益差超过 1.5% 的日子视为换月跳空，当天基差留空，不进分位。",
        "",
        "日频隐含波动率是 QVIX 收盘除以 100。50ETF 从 2015-02 起，300ETF 从 2019-12 起，500ETF 更晚。"
        "分位是截至当日的 250 日窗口，最少 60 日。缺数据的日子不挡买入。",
        "",
        "近月平值 IV 和 25delta 偏斜来自上交所期权风险指标的月末快照。"
        "平值是 delta 最接近 0.5 或 -0.5 的合约。偏斜是 25delta 认沽 IV 减去 25delta 认购 IV。"
        "表里的 IV 是交易所给出的隐含波动率。Black-Scholes 反解（q=0）写在 `derivs/bs.py`，"
        "核对价是 S=K=100、r=0、σ=0.20、τ=1 的认购 7.9656±0.01，测试里反解回到 0.20。"
        "月末快照没有逐合约成交价，所以日频序列没有再反解一遍。",
        "",
    ]
    for code in CODES:
        panel = panels[code]
        lines.append(
            f"- {code}：基差 {_coverage(panel, 'basis_ann')}；"
            f"QVIX {_coverage(panel, 'iv')}。QVIX 对月末平值：{_qvix_vs_atm(panel)}"
            f"中点 {mids[code].date()}。"
        )
    lines.extend(["", "## 过滤变体", ""])
    for code in CODES:
        lines.append(f"### {code}")
        lines.append("")
        lines.append("| 变体 | 区间 | 总收益 | 最大回撤 | Calmar | 交易次数 | 样本外改善 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in tables[code]:
            flag = ""
            if row["window"] == "后半段" and row["variant"] != "基线":
                flag = "是" if improved[code][row["variant"]] else "否"
            lines.append(
                f"| {row['variant']} | {row['window']} | {_fmt_pct(row['total'])} | "
                f"{_fmt_pct(row['mdd'])} | {_fmt_num(row['calmar'])} | {row['n_trades']} | {flag} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 买入点分桶",
            "",
            "下面是基线状态机真正开仓的日子，按信号日的分位分成低、中、高。"
            "收益从成交日收盘起算。基差分位低是贴水更深的一段，隐含波动率分位高是期权更贵的一段。"
            "笔数少于 5 的格子只作记录。",
            "",
        ]
    )
    for code in CODES:
        frame = entries[code]
        mid = mids[code]
        if frame.empty:
            lines.extend([f"### {code}", "", "没有开仓。", ""])
            continue
        front = frame["signal"] < mid
        back = frame["signal"] >= mid
        lines.extend([f"### {code}", "", f"前半段，基差分位（{int(front.sum())} 笔）", ""])
        lines.extend(_bucket_lines(frame, "basis_bucket", front))
        lines.extend(["", f"前半段，隐含波动率分位", ""])
        lines.extend(_bucket_lines(frame, "iv_bucket", front))
        lines.extend(["", f"后半段，基差分位（{int(back.sum())} 笔）", ""])
        lines.extend(_bucket_lines(frame, "basis_bucket", back))
        lines.extend(["", f"后半段，隐含波动率分位", ""])
        lines.extend(_bucket_lines(frame, "iv_bucket", back))
        skew = pd.to_numeric(frame.loc[back, "skew"], errors="coerce").dropna()
        lines.extend(
            [
                "",
                f"后半段能读到偏斜的开仓 {len(skew)} 笔。"
                + ("" if skew.empty else f"中位数 {skew.median():.1%}。"),
                "",
            ]
        )
    lines.extend(
        [
            "## 不做的事",
            "",
            "- 不改 `buy_sell_masks`、aux_edge、B1、快形态阈值。",
            "- 不做盘中信号，不下期货或期权的单。",
            "- 不做 159915 对 IM、科创 50 这类跨指数配对。",
            "- 不用基差或隐含波动率预测涨跌方向。",
            "",
            "## 限制",
            "",
            "新浪的主力连续合约从 2017-01-17 才有，更早的基差是空的。年化时把这份连续价当成近月，分红和真实换月日都会把它带偏。"
            "换月跳空只按 1.5% 的价差来认。期权样本短于 ETF 本身，300ETF 期权从 2019 年底才有，500ETF 期权从 2022-09 才有。"
            "510500 有两天空的收盘，状态机按 rules7 的做法用前一天收盘补上，避免整段净值变成空值。"
            "三只宽基、后半段只有几笔交易，容易把一次碰巧读成规则，所以要求两只以上、并且每只还留得住 5 笔。",
            "",
        ]
    )
    return "\n".join(lines)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    market = refresh_market(refresh=args.refresh)
    chain = load_chain()
    if not args.skip_chain:
        index = close_series(market.get("index:000300", pd.DataFrame()))
        if not index.empty:
            chain = refresh_chain(month_ends(index.index))
    print("loading rules7 frames")
    frames = _load_frames(CODES)
    panels: dict[str, pd.DataFrame] = {}
    tables: dict[str, list[dict[str, Any]]] = {}
    entries: dict[str, pd.DataFrame] = {}
    mids: dict[str, pd.Timestamp] = {}
    improved: dict[str, dict[str, bool]] = {}
    for code in CODES:
        frame = frames[code]
        close = frame["px"].astype(float).ffill()
        panel = _panel_for(code, close, market, chain).reindex(close.index)
        panels[code] = panel
        evaluated = rules7.evaluate(code, frame)
        spec = evaluated["spec"]
        g1 = evaluated["masks"]["g1"] if spec.g1_hold else None
        mid = pd.Timestamp(close.index[len(close) // 2]).normalize()
        mids[code] = mid
        windows = {
            "全样本": (None, None),
            "前半段": (None, mid - pd.Timedelta(days=1)),
            "后半段": (mid, None),
        }
        code_rows: list[dict[str, Any]] = []
        flags: dict[str, bool] = {}
        masks = block_masks(panel)
        ordered = ["基线", *[name for name in masks if name != "基线"]]
        base_oos = None
        for name in ordered:
            block = masks[name]
            buy = evaluated["buy"] & ~block.reindex(close.index).fillna(False)
            for window, (start, end) in windows.items():
                result = run_window(
                    close, buy, evaluated["sell"], g1=g1, trail=spec.trail, start=start, end=end
                )
                metrics = _metrics(result)
                code_rows.append({"variant": name, "window": window, **metrics})
                if name == "基线" and window == "后半段":
                    base_oos = metrics
            if name != "基线":
                oos = next(row for row in code_rows if row["variant"] == name and row["window"] == "后半段")
                flags[name] = oos_improved(base_oos, oos)
            else:
                flags[name] = False
        tables[code] = code_rows
        improved[code] = flags
        entries[code] = _entry_table(
            run_window(close, evaluated["buy"], evaluated["sell"], g1=g1, trail=spec.trail),
            panel,
            close,
        )
        print(code, "oos", {name: flags[name] for name in flags if name != "基线"})
    kept = adopted_variants(improved)
    text = render_doc(
        panels=panels,
        tables=tables,
        entries=entries,
        mids=mids,
        improved=improved,
        kept=kept,
    )
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text(text, encoding="utf-8")
    print(f"wrote {args.doc}")
    print("kept", kept or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
