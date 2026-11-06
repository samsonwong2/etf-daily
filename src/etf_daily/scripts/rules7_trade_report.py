"""Daily P95 signal report. Each signal is its own trade; nothing is a live account."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from etf_daily.lib import rules7
from etf_daily.lib.rules7_trade_plan import (
    HOLD_DAYS,
    STOP_PCT,
    WATCH_Q,
    WATCH_TARGET_TRADES,
    breadth_stats,
    d080_cluster_csv,
    tier_history,
    trade_stats,
)

LEDGER_COLUMNS = [
    "date", "code", "name", "n_buy", "pool", "share", "tier",
    "action", "filtered", "entry_close", "ret_5", "ret_10", "ret_20",
]

SHANGHAI = ZoneInfo("Asia/Shanghai")
NEAR_AVG = 0.06


def _now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(SHANGHAI)
    if now.tzinfo is None:
        return now.replace(tzinfo=SHANGHAI)
    return now.astimezone(SHANGHAI)


def is_intraday(as_of: pd.Timestamp | str, now: datetime | None = None) -> bool:
    clock = _now(now)
    return pd.Timestamp(as_of).date() == clock.date() and clock.hour < 15


def report_path(listing: Path, as_of: pd.Timestamp | str, now: datetime | None = None) -> Path:
    tag = pd.Timestamp(as_of).strftime("%Y%m%d")
    clock = _now(now)
    if is_intraday(as_of, clock):
        return listing / f"rules7_trade_report_{tag}_{clock.strftime('%H%M')}.md"
    return listing / f"rules7_trade_report_{tag}.md"


def _px(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    return f"{number:.4f}"


def _pct(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    return f"{number:+.1%}"


def _num(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _close_px(row: pd.Series) -> float:
    return _num(row.get("T收盘"))


def _avg_dist(row: pd.Series) -> float:
    lo = _num(row.get("_dist_lo9"))
    ewma = _num(row.get("_dist_ewma"))
    if np.isfinite(lo) and np.isfinite(ewma):
        return (lo + ewma) / 2.0
    return lo if np.isfinite(lo) else ewma


def _dist_text(row: pd.Series) -> str:
    return f"距P95 {_pct(row.get('_dist_lo9'))}，距EWMA {_pct(row.get('_dist_ewma'))}"


def _load_cluster(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty or "code" not in frame.columns:
        return pd.DataFrame(columns=["code", "cluster", "is_representative"])
    out = frame.copy()
    out["code"] = out["code"].astype(str).str.upper()
    if "domain" in out.columns:
        out = out[out["domain"].astype(str) != "bond"]
    if "is_representative" in out.columns:
        out["is_representative"] = out["is_representative"].map(
            lambda x: x is True or str(x).strip().lower() in {"true", "1", "yes"}
        )
    else:
        out["is_representative"] = False
    return out.drop_duplicates("code")[["code", "cluster", "is_representative"]]


def _attach_cluster(df: pd.DataFrame, cluster: pd.DataFrame | None) -> pd.DataFrame:
    out = df.copy()
    out["代码"] = out["代码"].astype(str).str.upper()
    if cluster is None or cluster.empty:
        out["_cluster"] = np.nan
        out["_is_rep"] = False
        return out
    merged = out.drop(columns=[c for c in ("_cluster", "_is_rep") if c in out.columns], errors="ignore")
    merged = merged.merge(cluster, left_on="代码", right_on="code", how="left")
    merged["_cluster"] = merged["cluster"]
    merged["_is_rep"] = merged["is_representative"].fillna(False).astype(bool)
    return merged.drop(columns=[c for c in ("code", "cluster", "is_representative") if c in merged.columns])


def _watch_reason(row: pd.Series, demoted: str | None) -> str:
    if not bool(row.get("_plan_applicable", True)):
        return "规则不适用。文档里没有一套规则能赚钱，信号只供参考。"
    quotes = (str(row.get("买点状态") or ""), str(row.get("EWMA买点状态") or ""))
    if rules7.FAST_BLOCKED in quotes:
        return "快形态挡住。10 月这组平均 -5.1%，2 到 9 月也比全池差，不买。" + _dist_text(row)
    if rules7.FAST_ACTIVE in quotes:
        return "快形态中，今天没有买点。就算到价也不买。" + _dist_text(row)
    if demoted:
        return demoted + _dist_text(row)
    if str(row.get("EWMA买点状态") or "") == "今天已是买点":
        return "只有 EWMA 买点。这条信号胜率约 41%，不作为买入依据。" + _dist_text(row)
    avg = _avg_dist(row)
    if np.isfinite(avg) and avg <= NEAR_AVG:
        return "贴近下轨（两条平均距离 ≤ 6%）。这是回测里最差的一组，胜率约 41%，不抄底。" + _dist_text(row)
    return "离下轨较远，今天没有 P95 正式买点。" + _dist_text(row)


def _pick_buy(group: pd.DataFrame) -> pd.Series:
    ranked = group.copy()
    ranked["_rep_rank"] = np.where(ranked["_is_rep"].astype(bool), 0, 1)
    ranked["_dist_rank"] = ranked["_dist_lo9"].map(_num)
    ranked = ranked.sort_values(["_rep_rank", "_dist_rank", "代码"], kind="mergesort")
    return ranked.iloc[0]


def _share_txt(stats: dict) -> str:
    share = stats.get("share")
    return f"{share:.1%}" if isinstance(share, (int, float)) and np.isfinite(share) else "—"


def _breadth_line(stats: dict) -> str:
    note = "集中度按检查表的买入信号估算（缺少回放列）。" if stats.get("degraded") else ""
    if stats.get("tier") == "无买点" or int(stats.get("n_buy") or 0) <= 0:
        return note + "今日全池无买点，集中度不适用。"
    return (
        note
        + f"今日全池买点 {stats['n_buy']} 只 / 池子 {stats['pool']}，占 {_share_txt(stats)}，属于{stats['tier']}。"
    )


def _tier_note(tier: str, stats: dict) -> str:
    base = f"今日全池买点 {stats['n_buy']} 只 / {stats['pool']}，占 {_share_txt(stats)}，属于{tier}。"
    if tier == "一般":
        return base + "这一档的历史成绩见上面的全样本统计，仍列出，但不宜加仓。"
    return base + "这一档的历史成绩见上面的全样本统计。"


def _breadth_reason(stats: dict) -> str:
    return (
        f"P95 买点单独出现：今日全池买点 {stats['n_buy']} 只 / {stats['pool']}，占 {_share_txt(stats)}，低于 5%。"
        "这一档的历史成绩见上面的全样本统计，降到观察档，不自动买。只过滤今天的列表，不改买入信号。"
    )


def _assign(df: pd.DataFrame, *, breadth_filter: bool = True, stats: dict | None = None) -> pd.DataFrame:
    out = df.copy()
    out["_report_action"] = ""
    out["_report_reason"] = ""
    out["_report_tier"] = ""
    out["_breadth_filtered"] = False
    if "_plan_applicable" not in out.columns:
        out["_plan_applicable"] = True
    if "_dist_lo9" not in out.columns:
        out["_dist_lo9"] = np.nan
    if stats is None:
        stats = breadth_stats(out)
    if "_buy_raw" not in out.columns:
        out["_buy_raw"] = False
    applicable = out["_plan_applicable"].map(lambda v: bool(v) if pd.notna(v) else True)
    raw = out["_buy_raw"].map(_raw_buy)
    buys = raw & applicable
    # Drop the whole day before cluster picking, so a demoted buy does not take the slot.
    solo = buys & bool(breadth_filter and stats.get("tier") == "单独")
    buys = buys & ~solo
    chosen: set[str] = set()
    demoted: dict[str, str] = {}
    candidates = out[buys]
    clustered = candidates[candidates["_cluster"].notna()]
    for _cid, group in clustered.groupby("_cluster", sort=True):
        pick = _pick_buy(group)
        chosen.add(str(pick["代码"]))
        for code, name in zip(group["代码"], group["名称"]):
            if str(code) == str(pick["代码"]):
                continue
            demoted[str(code)] = (
                f"同簇今天已推荐 {pick['代码']} {pick['名称']}。你如果已经持有同簇的，自己跳过。"
            )
    for code in candidates.loc[candidates["_cluster"].isna(), "代码"]:
        chosen.add(str(code))
    out.loc[out["代码"].isin(chosen), "_report_action"] = "买"
    if breadth_filter and stats.get("tier") in {"集中", "一般"}:
        out.loc[out["代码"].isin(chosen), "_report_tier"] = stats["tier"]
    if bool(solo.any()):
        out.loc[solo, "_report_action"] = "观察"
        out.loc[solo, "_report_tier"] = "单独"
        out.loc[solo, "_report_reason"] = _breadth_reason(stats)
        out.loc[solo, "_breadth_filtered"] = True
    for col, default in (("_w_buy_raw", False), ("_w_lower", np.nan), ("_w_dist", np.nan)):
        if col not in out.columns:
            out[col] = default
    watch = out["_report_action"].eq("")
    for idx in out.index[watch]:
        code = str(out.at[idx, "代码"])
        row = out.loc[idx]
        is_watch = applicable.loc[idx] and _raw_buy(row.get("_w_buy_raw")) and not _raw_buy(row.get("_buy_raw"))
        if is_watch:
            out.at[idx, "_report_action"] = "观察"
            out.at[idx, "_report_reason"] = _p90_reason(row)
            continue
        out.at[idx, "_report_action"] = "观望"
        out.at[idx, "_report_reason"] = _watch_reason(row, demoted.get(code))
    return out


def _p90_reason(row: pd.Series) -> str:
    return (
        f"P90 买点触发，P95 没触发（距P95 {_pct(row.get('_dist_lo9'))}，距P90 {_pct(row.get('_w_dist'))}）。"
        "每个信号单独计一笔。全样本成绩见上面的统计，只观察，不自动买。"
    )


def _buy_reason() -> str:
    return "P95 正式买点。每个信号单独计一笔，持有 20 个交易日、止损 -10%。明日收盘买入。"


def _list_of(value: object) -> list:
    return list(value) if isinstance(value, list) else []


def _signal_status(trade: dict) -> str:
    status = trade.get("status")
    if status == "pending":
        return "待成交"
    if status == "closed":
        when = trade.get("exit_date") or "—"
        if trade.get("reason") == "stop":
            return f"已于 {when} 止损"
        return f"已于 {when} 到期"
    if trade.get("sell_tomorrow") and trade.get("reason") == "stop":
        return "明日止损卖"
    if trade.get("sell_tomorrow"):
        return "明日到期卖"
    left = trade.get("days_left")
    left_txt = "—" if left is None else str(int(left))
    return f"持有中，剩 {left_txt} 天"


def _in_recent(trade: dict) -> bool:
    if trade.get("status") == "pending":
        return True
    bars = trade.get("bars_from_end")
    return isinstance(bars, (int, np.integer)) and int(bars) <= HOLD_DAYS - 1


def _fmt_stat(stat: tuple[int, float, float]) -> str:
    n, win, mean = stat
    if n == 0:
        return "还没有已平仓的笔"
    return f"{n} 笔，胜率 {win:.0%}，平均 {mean:+.2%}"


def _cluster_label(row: pd.Series) -> str:
    value = row.get("_cluster")
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["（无）", ""]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _recent_cutoff(df: pd.DataFrame) -> str | None:
    if "_recent_start" not in df.columns:
        return None
    values = [str(v) for v in df["_recent_start"].dropna().tolist() if str(v) not in {"", "None", "nan"}]
    if not values:
        return None
    return max(set(values), key=values.count)


def _tier_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    applicable = df["_plan_applicable"] if "_plan_applicable" in df.columns else pd.Series(True, index=df.index)
    for idx, row in df.iterrows():
        flag = applicable.loc[idx]
        rows.append({
            "applicable": bool(flag) if pd.notna(flag) else True,
            "bar250_date": row.get("_bar250_date") or None,
            "last_date": row.get("_last_date") or None,
            "trades": _list_of(row.get("_sig_trades")),
        })
    return rows


def _p90_only_trades(df: pd.DataFrame) -> list[dict]:
    found: list[dict] = []
    for _, row in df.iterrows():
        p95_days = {str(trade.get("signal_date")) for trade in _list_of(row.get("_sig_trades"))}
        for trade in _list_of(row.get("_w_trades")):
            if str(trade.get("signal_date")) not in p95_days:
                found.append(trade)
    return found


def _etf_stat(row: pd.Series) -> str:
    n, win, mean = trade_stats(_list_of(row.get("_sig_trades")))
    if n == 0:
        return "这只还没有已平仓的 P95 买点"
    return f"{n} 笔，胜率 {win:.0%}，平均 {mean:+.2%}"


def _sc(row: pd.Series) -> dict:
    value = row.get("_sc")
    return value if isinstance(value, dict) else {}


def _closest_edge(ranges: list) -> float:
    edges = [edge for lo, hi in ranges for edge in (lo, hi)]
    return min(edges, key=abs) if edges else float("nan")


def _intersects(ranges: list, band: float) -> bool:
    return any(hi >= -band and lo <= band for lo, hi in ranges)


def _range_phrase(ranges: list, px: float) -> str:
    parts = []
    for lo, hi in ranges:
        lo_txt, hi_txt = _pct(lo), _pct(hi)
        if lo <= -0.0995 and hi >= 0.0995:
            parts.append("涨跌 ±10% 内都会触发")
        elif lo <= -0.0995:
            parts.append(f"跌到 {hi_txt} 或更低（收盘 ≤ {_px(px * (1 + hi))}）")
        elif hi >= 0.0995:
            parts.append(f"涨到 {lo_txt} 或更高（收盘 ≥ {_px(px * (1 + lo))}）")
        else:
            parts.append(f"涨跌在 {lo_txt} 到 {hi_txt} 之间")
    return "；".join(parts) if parts else "—"


def _scenario_lines(df: pd.DataFrame, *, breadth_filter: bool = True) -> list[str]:
    if "_sc" not in df.columns or not df["_sc"].map(lambda v: isinstance(v, dict)).any():
        return []
    bands = (0.02, 0.05, 0.10)
    counts = {b: {"买": 0, "卖": 0, "持有": 0} for b in bands}
    buy_rows, p90_rows, sell_rows, hold_rows, blocked_rows = [], [], [], [], []
    no_change = 0
    for _, row in df.iterrows():
        sc = _sc(row)
        if not sc:
            continue
        px = _close_px(row)
        raw_buy = list(sc.get("buy_ranges") or [])
        raw_p90 = list(sc.get("p90_ranges") or [])
        reason = sc.get("block")
        usable = not reason
        buy_ranges = raw_buy if usable else []
        p90_ranges = raw_p90 if usable else []
        exits = [item for item in _list_of(sc.get("exits")) if isinstance(item, dict)]
        sell_in_band = False
        holds_any = False
        for band in bands:
            if _intersects(buy_ranges, band):
                counts[band]["买"] += 1
            for ex in exits:
                kind, move = ex.get("sell_kind"), ex.get("sell_move")
                sells_here = kind == "expiry" or (
                    kind == "stop" and move is not None and move >= -band
                )
                if sells_here:
                    counts[band]["卖"] += 1
                if kind == "stop" and (move is None or move < -band):
                    counts[band]["持有"] += 1
        if buy_ranges:
            buy_rows.append((abs(_closest_edge(buy_ranges)), [
                str(row["代码"]), str(row["名称"]), _cluster_label(row), _px(px),
                _range_phrase(buy_ranges, px),
            ]))
        p90_edge, p95_edge = _closest_edge(p90_ranges), _closest_edge(buy_ranges)
        if p90_ranges and (not buy_ranges or abs(p90_edge) < abs(p95_edge) - 0.0005):
            p90_rows.append((abs(p90_edge), [
                str(row["代码"]), str(row["名称"]), _cluster_label(row), _px(px),
                _range_phrase(p90_ranges, px), "观察，不自动买",
            ]))
        for ex in exits:
            kind, move = ex.get("sell_kind"), ex.get("sell_move")
            stamp = str(ex.get("signal_date") or "—")
            if kind == "expiry":
                sell_in_band = True
                sell_rows.append([
                    str(row["代码"]), str(row["名称"]), stamp, "满20日", "—", "明天无论涨跌都卖",
                ])
            elif kind == "stop" and move is not None and move >= -0.10:
                sell_in_band = True
                sell_rows.append([
                    str(row["代码"]), str(row["名称"]), stamp, "止损",
                    _px(float(ex["entry_px"]) * (1.0 - STOP_PCT)) if ex.get("entry_px") else "—",
                    f"跌到 {_pct(move)} 或更低（收盘 ≤ {_px(px * (1 + move))}）",
                ])
            if kind == "stop" and move is not None:
                holds_any = True
                hold_rows.append([
                    str(row["代码"]), str(row["名称"]), stamp,
                    f"高于 {_pct(move)}（{_px(px * (1 + move))} 以上）到 +10%",
                ])
        if reason:
            # Same wording as the checklist: a block with no buy trigger today is not a blocked buy.
            if reason == rules7.FAST_BLOCKED and str(row.get("买点状态") or "") == rules7.FAST_ACTIVE:
                reason = rules7.FAST_ACTIVE
            blocked_rows.append([str(row["代码"]), str(row["名称"]), _cluster_label(row), str(reason)])
        if not buy_ranges and not p90_ranges and not sell_in_band and not holds_any:
            no_change += 1

    lines = ["## 明日情景（假设明天收盘涨跌 ±10%）", ""]
    lines.append(
        "这不是方向预测。假设明天收盘相对今天涨跌 x%（-10% 到 +10%，ETF 的涨跌停范围），"
        "用同一套规则重算。买入看全部适用 ETF，不因为别的信号还没结束就跳过。"
        "卖出和继续持有按下面「最近 20 个交易日」里仍未结束的每一笔单独算，只有你真的在那天买了才适用。"
        "明天收盘出的信号，后天收盘才成交。基准（沪深300）按明天不动计算。"
    )
    if breadth_filter:
        lines.append(
            "集中度过滤只对今日信号生效。明天全池有多少只一起出买点现在还不知道，下面的买入区间不按集中度降级。"
        )
    lines.append("")
    lines.append("| 假设明天收盘涨跌在 | 会出买入 | 会出卖出 | 继续持有 |")
    lines.append("| --- | ---: | ---: | ---: |")
    for band in bands:
        c = counts[band]
        lines.append(f"| ±{band:.0%} | {c['买']} 只 | {c['卖']} 笔 | {c['持有']} 笔 |")
    lines.append("")
    lines.append(f"±10% 内买卖和持有都不会变化的有 {no_change} 只，不逐只列出。")
    lines.append("")
    lines.append(f"### 买入机会（明天收盘出 P95 买点，后天成交）：{len(buy_rows)} 只")
    lines.append("")
    lines += _md_table(["代码", "名称", "簇", "今日收盘", "需要的涨跌"],
                       [r for _, r in sorted(buy_rows, key=lambda x: x[0])])
    lines.append(f"### P90 观察档机会（同样的涨跌下只到 P90，不自动买）：{len(p90_rows)} 只")
    lines.append("")
    lines += _md_table(["代码", "名称", "簇", "今日收盘", "需要的涨跌", "说明"],
                       [r for _, r in sorted(p90_rows, key=lambda x: x[0])])
    lines.append(f"### 卖出机会（未结束的信号里，明天收盘会触发卖出）：{len(sell_rows)} 笔")
    lines.append("")
    lines += _md_table(["代码", "名称", "信号日", "类型", "止损价", "需要的涨跌"], sell_rows)
    lines.append(f"### 持有区间（未结束的信号里，明天收盘仍未止损）：{len(hold_rows)} 笔")
    lines.append("")
    lines += _md_table(["代码", "名称", "信号日", "继续持有的区间"], hold_rows)
    lines.append(f"### 这几种情况不给触发价：{len(blocked_rows)} 只")
    lines.append("")
    lines += _md_table(["代码", "名称", "簇", "原因"], blocked_rows)
    return lines


def write_trade_report(
    listing: Path,
    df: pd.DataFrame,
    *,
    as_of: pd.Timestamp | str,
    now: datetime | None = None,
    cluster: pd.DataFrame | None = None,
    cluster_path: Path | None = None,
    cluster_date: pd.Timestamp | None = None,
    breadth_filter: bool = True,
    ledger_path: Path | None = None,
) -> Path:
    clock = _now(now)
    as_of_ts = pd.Timestamp(as_of).normalize()
    intraday = is_intraday(as_of_ts, clock)
    if cluster is None and cluster_path is None:
        cluster_path, cluster_date = d080_cluster_csv(as_of_ts)
    if cluster is None and cluster_path is not None:
        cluster = _load_cluster(cluster_path)
    attached = _attach_cluster(df, cluster)
    stats = breadth_stats(attached)
    assigned = _assign(attached, breadth_filter=breadth_filter, stats=stats)
    path = report_path(listing, as_of_ts, clock)
    all_trades: list[dict] = []
    for trades in assigned.get("_sig_trades", pd.Series(dtype=object)).tolist():
        all_trades.extend(_list_of(trades))
    full_stat = trade_stats(all_trades)
    recent_stat = trade_stats(all_trades, _recent_cutoff(assigned))
    tier_stat = tier_history(_tier_rows(assigned))
    p90_stat = trade_stats(_p90_only_trades(assigned))

    lines = [
        f"# 每日买卖报告 {as_of_ts.date()}",
        "",
        "> 研究清单，不推断持仓，不下单。每个 P95 买点单独计一笔，持有期内再出的买点也算。",
        "",
        f"生成时间: {clock.strftime('%Y-%m-%d %H:%M')}（上海）",
        f"读数: {'盘中。最后一根 K 线未收盘，收盘可能变。' if intraday else '收盘后。'}",
        f"as-of **{as_of_ts.date()}** 收盘出信号，买卖都在次日收盘成交。",
        f"规则: 只在 P95 正式买点买入；持有 {HOLD_DAYS} 个交易日；收盘 ≤ 买入价 × {1 - STOP_PCT:.2f} 则次日止损。",
    ]
    if breadth_filter:
        lines.append(_breadth_line(stats))
    if ledger_path is not None:
        lines.append(f"信号账本: `{ledger_path}`")
    if cluster_path is None:
        lines.append("聚类: 没有找到 d080 簇文件，本次不按簇去重。")
    else:
        used = pd.Timestamp(cluster_date).date() if cluster_date is not None else "未知"
        stale = ""
        if cluster_date is not None and pd.Timestamp(cluster_date).normalize() < as_of_ts:
            stale = "（早于 as-of，用最近一天）"
        lines.append(f"聚类: `{cluster_path}`，日期 {used}{stale}。")
    lines.append("")
    lines.append("## 全样本统计")
    lines.append("")
    lines.append(
        f"口径：每个 P95 买点单独一笔，信号次日收盘买入，持有 {HOLD_DAYS} 个交易日，"
        f"收盘跌破买入价 {STOP_PCT:.0%} 则次日止损。持有期内再出的买点也算。只统计已平仓。"
        "相邻信号的持仓会重叠，独立样本比笔数少。"
        "集中、一般、单独只计入信号当天该 ETF 已有 250 根 K 线的笔，所以三档加起来少于上市以来全部。"
    )
    lines.append("")
    lines += _md_table(
        ["分组", "结果"],
        [
            ["上市以来全部", _fmt_stat(full_stat)],
            ["最近 60 个交易日", _fmt_stat(recent_stat)],
            ["集中（当天占比不低于 8%）", _fmt_stat(tier_stat["集中"])],
            ["一般（5% 到 8%）", _fmt_stat(tier_stat["一般"])],
            ["单独（不到 5%）", _fmt_stat(tier_stat["单独"])],
            ["P90 有、P95 没有", _fmt_stat(p90_stat)],
        ],
    )
    _n_recent, win_recent, _mean_recent = recent_stat
    if _n_recent and win_recent < 0.40:
        lines.append("最近 60 个交易日买点失效偏多，新买入宜减量。这只是提示，不改变下面的列表。")
        lines.append("")

    buys = assigned[assigned["_report_action"].eq("买")].sort_values(["_cluster", "代码"], na_position="last")
    watches = assigned[assigned["_report_action"].eq("观望")].sort_values("代码")

    lines.append(f"## 今日 P95 买点（明日收盘买）：{len(buys)} 只")
    lines.append("")
    lines.append("同簇同日只留一只（优先簇代表，其次更贴近 P95 下轨）。你如果已经持有同簇的，自己跳过。")
    lines.append("")
    buy_rows = []
    for _, row in buys.iterrows():
        close = _close_px(row)
        reason = _buy_reason()
        if str(row.get("_report_tier") or "") in {"集中", "一般"}:
            reason += _tier_note(str(row.get("_report_tier")), stats)
        buy_rows.append([
            str(row["代码"]),
            str(row["名称"]),
            _cluster_label(row),
            _px(close),
            str(row.get("买点价") or "—"),
            _px(close * (1.0 - STOP_PCT)) if np.isfinite(close) else "—",
            _etf_stat(row),
            reason,
        ])
    lines += _md_table(
        ["代码", "名称", "簇", "T收盘", "P95买点价", "参考止损价", "这只上市以来", "原因"],
        buy_rows,
    )
    lines.append("参考止损价 = 今日收盘 × 0.9，只作对照。实际止损按明日成交价计算。")
    lines.append("")

    recent_rows = []
    for _, row in assigned.iterrows():
        for trade in _list_of(row.get("_sig_trades")):
            if not _in_recent(trade):
                continue
            recent_rows.append((str(trade.get("signal_date") or ""), str(row["代码"]), [
                str(row["代码"]),
                str(row["名称"]),
                str(trade.get("signal_date") or "—"),
                str(trade.get("entry_date") or "明日"),
                _px(trade.get("entry_px")),
                "—" if trade.get("days_held") is None else str(int(trade["days_held"])),
                _pct(trade.get("pnl")),
                _px(trade.get("stop_px")),
                _pct(trade.get("dist_stop")),
                _signal_status(trade),
            ]))
    recent_rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
    lines.append(f"## 最近 {HOLD_DAYS} 个交易日的 P95 买点：{len(recent_rows)} 笔")
    lines.append("")
    lines.append(
        "同一只 ETF 的多次信号分行。状态是按规则推的，不是你的账户。"
        "只有你实际在信号次日买了，止损价和到期日才适用。"
    )
    lines.append("")
    lines += _md_table(
        ["代码", "名称", "信号日", "假设买入日", "假设买入价", "第几天", "浮盈", "止损价", "距止损", "状态"],
        [item[2] for item in recent_rows],
    )

    observes = assigned[assigned["_report_action"].eq("观察")].sort_values(["_cluster", "代码"], na_position="last")
    p90_obs = observes[observes["_report_tier"].ne("单独")]
    solo_obs = observes[observes["_report_tier"].eq("单独")]
    lines.append(f"## P90 观察档（P90 触发、P95 未触发，不自动买入）：{len(p90_obs)} 只")
    lines.append("")
    lines.append(
        f"下轨分位从 P95 放宽到 P{int(round(WATCH_Q * 100))}，买入信号的其他条件不变（均线/gap、上跳沿、快形态过滤）。"
        "只做观察，不进入今日 P95 买点，也不占用同簇名额。"
    )
    lines.append("")
    lines.append(
        f"样本进度（上市以来，每个信号单独一笔，持有 {HOLD_DAYS} 天、止损 -{STOP_PCT:.0%}）："
        f"P{int(round(WATCH_Q * 100))} 有而 P95 没有 {_fmt_stat(p90_stat)}；P95 {_fmt_stat(full_stat)}。"
        f"升为正式买点的条件: P90 已平仓 ≥ {WATCH_TARGET_TRADES} 笔，且不输 P95，平均收益的区间下限在 0 以上。"
    )
    lines.append("")
    obs_rows = []
    for _, row in p90_obs.iterrows():
        obs_rows.append([
            str(row["代码"]),
            str(row["名称"]),
            _cluster_label(row),
            _px(_close_px(row)),
            _px(row.get("_w_lower")),
            _pct(row.get("_w_dist")),
            str(row.get("_report_reason") or ""),
        ])
    lines += _md_table(["代码", "名称", "簇", "T收盘", "P90下轨价", "距P90", "原因"], obs_rows)

    if breadth_filter:
        lines.append(f"## 单独出现（今日 P95 买点占比不到 5%，不自动买入）：{len(solo_obs)} 只")
        lines.append("")
        lines.append(
            "这些是今天的 P95 正式买点，但当天全池买点占比不到 5%。"
            "降到观察档，不进推荐买入，也不占用同簇名额。买入信号列本身不改。"
        )
        lines.append("")
        solo_rows = []
        for _, row in solo_obs.iterrows():
            solo_rows.append([
                str(row["代码"]),
                str(row["名称"]),
                _cluster_label(row),
                _px(_close_px(row)),
                _share_txt(stats),
                str(row.get("_report_reason") or ""),
            ])
        lines += _md_table(["代码", "名称", "簇", "T收盘", "全池占比", "原因"], solo_rows)

    lines += _scenario_lines(assigned, breadth_filter=breadth_filter)

    lines.append(f"## 观望：{len(watches)} 只")
    lines.append("")
    watch_rows = []
    for _, row in watches.iterrows():
        watch_rows.append([
            str(row["代码"]),
            str(row["名称"]),
            _cluster_label(row),
            str(row.get("_report_reason") or ""),
        ])
    lines += _md_table(["代码", "名称", "簇", "原因"], watch_rows)

    lines += [
        "## 规则来源",
        "",
        f"- 买入只认 P95 正式买点。每个信号单独一笔，持有 {HOLD_DAYS} 天、止损 -10%。"
        f"上市以来 {_fmt_stat(full_stat)}。",
        f"- 按信号当天的全池占比：集中 {_fmt_stat(tier_stat['集中'])}；"
        f"一般 {_fmt_stat(tier_stat['一般'])}；单独 {_fmt_stat(tier_stat['单独'])}。",
        "- 止损 -3% 或 -5% 会把胜率压低大约 10 到 20 个百分点，所以不用紧止损。",
        "- 快形态挡住：10 月平均约 -5.1%，2 到 9 月也比全池差。",
        "- 只有 EWMA 买点的胜率约 41%，不买。贴近下轨是回测里最差的一组，胜率约 41%。",
        "- 同簇同日多只出买点，只留一只（优先簇代表，其次更贴近 P95 下轨）。不推断你是否已持有。",
    ]
    if breadth_filter:
        lines += [
            "- 当天全池 P95 买点占比不到 5% 时，这些买点降到观察档，不占簇位。"
            "占比 5% 到 8% 标「一般」仍列出，不低于 8% 标「集中」。"
            "分母是适用且至少有 250 根 K 线的 ETF。",
        ]
    lines += [
        f"- P90 有而 P95 没有：{_fmt_stat(p90_stat)}。"
        f"升为正式买点要已平仓 ≥ {WATCH_TARGET_TRADES} 笔，并且不输 P95。目前只观察。",
        "",
        "## 边界",
        "",
        "- 相邻信号的持仓会重叠，独立样本比笔数少。下跌段里买点同样会失效。",
        "- 最近 60 个交易日胜率低于 40% 时只提示减量，不改当日列表。",
        "- 最近 20 日信号表不是账户。只有你实际在信号次日买了，止损和到期才适用。",
        "- 明日情景假设基准（沪深300）明天不动，只看收盘价，涨跌停按 ±10%。明天收盘出的信号后天才成交。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    if ledger_path is not None:
        update_ledger(ledger_path, assigned, as_of_ts, stats)
    return path


def _raw_buy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "是"}
    try:
        if value is None or pd.isna(value):
            return False
    except (TypeError, ValueError):
        return False
    return bool(value)


def _ret_filled(value: object) -> bool:
    return np.isfinite(_num(value))


def _px_tail(row: pd.Series) -> list[tuple[str, float]]:
    tail = row.get("_px_tail")
    if not isinstance(tail, (list, tuple)):
        return []
    parsed: list[tuple[str, float]] = []
    for item in tail:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            parsed.append((str(item[0]), _num(item[1])))
    return parsed


def _backfill_ledger(frame: pd.DataFrame, assigned: pd.DataFrame) -> pd.DataFrame:
    """Fill ret_5/10/20 from this scan's close tail. A filled cell is left as-is."""
    tails: dict[str, list[tuple[str, float]]] = {}
    if "_px_tail" in assigned.columns and "代码" in assigned.columns:
        for _, row in assigned.iterrows():
            parsed = _px_tail(row)
            if parsed:
                tails[str(row["代码"])] = parsed
    out = frame.copy()
    for col in ("ret_5", "ret_10", "ret_20"):
        if col not in out.columns:
            out[col] = np.nan
    for idx, row in out.iterrows():
        tail = tails.get(str(row.get("code")))
        if not tail:
            continue
        dates = [item[0] for item in tail]
        try:
            pos = dates.index(str(row.get("date")))
        except ValueError:
            continue
        entry = _num(row.get("entry_close"))
        if not np.isfinite(entry) or entry <= 0:
            continue
        for k, col in ((5, "ret_5"), (10, "ret_10"), (20, "ret_20")):
            if _ret_filled(row.get(col)):
                continue
            if pos + k >= len(tail):
                continue
            price = tail[pos + k][1]
            if np.isfinite(price):
                out.at[idx, col] = price / entry - 1.0
    return out


def update_ledger(
    ledger_path: Path,
    assigned: pd.DataFrame,
    as_of: pd.Timestamp | str,
    stats: dict | None = None,
) -> Path:
    """Append today's raw P95 buys and backfill forward returns.

    Re-running the same as-of replaces that day's rows (same date and code)
    and does not change a return that is already filled.
    """
    if stats is None:
        stats = breadth_stats(assigned)
    as_of_s = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    fresh_rows: list[dict] = []
    if "_buy_raw" in assigned.columns:
        for _, row in assigned.iterrows():
            if not _raw_buy(row.get("_buy_raw")):
                continue
            fresh_rows.append(
                {
                    "date": as_of_s,
                    "code": str(row.get("代码")),
                    "name": str(row.get("名称")),
                    "n_buy": stats.get("n_buy"),
                    "pool": stats.get("pool"),
                    "share": stats.get("share"),
                    "tier": stats.get("tier"),
                    "action": str(row.get("_report_action") or ""),
                    "filtered": bool(row.get("_breadth_filtered")),
                    "entry_close": _close_px(row),
                    "ret_5": "",
                    "ret_10": "",
                    "ret_20": "",
                }
            )
    fresh = pd.DataFrame(fresh_rows, columns=LEDGER_COLUMNS)
    path = Path(ledger_path)
    if path.is_file():
        old = pd.read_csv(path, dtype=str).fillna("")
    else:
        old = pd.DataFrame(columns=LEDGER_COLUMNS)
    for col in LEDGER_COLUMNS:
        if col not in old.columns:
            old[col] = ""
    old = old.loc[:, LEDGER_COLUMNS]
    if not fresh.empty and not old.empty:
        keys = set(zip(fresh["date"].astype(str), fresh["code"].astype(str)))
        keep = [
            (str(date), str(code)) not in keys
            for date, code in zip(old["date"].astype(str), old["code"].astype(str))
        ]
        old = old.loc[keep]
    if fresh.empty:
        merged = old
    elif old.empty:
        merged = fresh
    else:
        merged = pd.concat([old, fresh], ignore_index=True)
    merged = _backfill_ledger(merged, assigned)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.loc[:, LEDGER_COLUMNS].to_csv(path, index=False, encoding="utf-8-sig")
    return path
