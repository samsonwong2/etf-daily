#!/usr/bin/env python3
"""7-ETF buy/sell rules: today's key-indicator checklist for one from_listing batch.

Rules from ``docs/7只ETF_买卖规律对比.md``: the 7 documented codes use their own
spec, every other equity ETF is auto-typed (平稳震荡 / 慢趋势 / 高波动主题 /
长期下跌或周期 / 历史不足) and uses that type's spec. For --as-of the CSV lists
every rule component's reading, its threshold verdict, which key indicators fire,
and today's action (买/卖/持有/观望, filled at T+1 close). Position state
(持仓/空仓, peak close, trailing stop, G1 seen in the trade) is replayed from the
first bar, but only today's state is written; no backtest statistics.

HRP cluster_representatives (prefer ``*_d080.csv``) annotate each row with its
cluster. Among same-day buy signals in one cluster, only one row is marked
``推荐买入`` (prefer cluster representative, then larger |ret20|).

Research scan only (does not change production)::

    etf-daily rules7 --listing-dir "$PLOTLY_ROOT/20260929_from_listing"
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.lib import rules7  # noqa: E402
from etf_daily.scripts import backtest_fig12_six_states_pool as pool  # noqa: E402

TYPE_ORDER = [rules7.TYPE_SMOOTH, rules7.TYPE_TREND, rules7.TYPE_HIVOL, rules7.TYPE_CYCLE, rules7.TYPE_SHORT]
ACTION_ORDER = {"买": 0, "卖": 1, "持有": 2, "观望": 3}
NEAR_WARN = 0.03
CLUSTER_PACK_ROOTS = (
    Path.home() / "temp" / "decision_packs",
    Path.home() / "etf-daily-output" / "temp" / "decision_packs",
)
CLUSTER_COLS = ("聚类", "簇人数", "是否簇代表", "簇组", "ret20", "推荐买入")


def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _pct(v: Any, digits: int = 1) -> str:
    x = _f(v)
    return "—" if x is None else f"{x:.{digits}%}"


def _px(v: Any) -> str:
    x = _f(v)
    return "—" if x is None else f"{x:.4f}"


def _yn(b: Any) -> str:
    return "是" if bool(b) else ""


def _day(ts: Any) -> str:
    return "" if ts is None else pd.Timestamp(ts).strftime("%Y-%m-%d")


def _p(key: str, digits: int = 1) -> Callable[[dict], str]:
    return lambda s: _pct(s[key], digits)


def _x(key: str) -> Callable[[dict], str]:
    return lambda s: _px(s[key])


def _b(key: str) -> Callable[[dict], str]:
    return lambda s: _yn(s[key])


# (column, formatter over the snapshot) in CSV order, grouped by rule.
SNAP_COLS: list[tuple[str, Callable[[dict], str]]] = [
    # buy: aux_edge
    ("图9下轨", _x("lo9")), ("距图9下轨", _p("dist_lo9")),
    ("图9近下轨(≤下轨×1.01)", _b("near_lo")), ("昨日近下轨", _b("near_lo_prev")),
    ("近下轨上跳沿", _b("near_lo_edge")),
    ("MA5", _x("ma5")), ("MA20", _x("ma20")),
    ("收盘<MA20", _b("below_ma20")), ("MA5<MA20", _b("ma5_below")),
    ("gap5", _p("gap5", 2)), ("gap5≤0", _b("gap5_le0")),
    ("gap20", _p("gap20", 2)), ("gap20≤0", _b("gap20_le0")),
    ("图10未触上轨", _b("fig10_not_hi")), ("辅助买像", _b("aux_buy")),
    ("aux_edge买点", _b("ae_buy")),
    # buy: B1 / fast pattern
    ("图7折价", _p("gap7")), ("图8折价", _p("gap8")),
    ("图7或图8折价≤-8%", _b("disc_ok")), ("B1", _b("b1")),
    ("近60日最大40日涨幅", _p("run40")), ("快形态(≥20%)", _b("fast")),
    # sell: fig9
    ("图9上轨", _x("hi9")), ("距图9上轨", _p("dist_hi9")),
    ("图9近上轨(≥上轨×0.99)", _b("near_hi")), ("近上轨上跳沿", _b("near_hi_edge")),
    ("收盘>MA20", _b("above_ma20")), ("MA5>MA20", _b("ma5_above")),
    ("gap5≥0", _b("gap5_ge0")), ("gap20≥0", _b("gap20_ge0")),
    ("辅助卖像", _b("aux_sell")), ("卖点A", _b("sell_a")),
    ("10日局部高点", _b("locmax")),
    ("图9带内位置", _p("pos9", 0)), ("图9位置≥80%", _b("pos9_ok")),
    ("图9斜率", _p("g9")), ("图9斜率>0", _b("g9_pos")),
    ("图9图10未触上轨", _b("no_touch_hi")),
    ("vol5分位", _p("vol5", 0)), ("vol5分位≥70%", _b("vol5_ok")),
    ("图7溢价≥5%", _b("prem7_ok")),
    ("图10带内位置", _p("pos10", 0)), ("图10位置≥90%", _b("pos10_ok")),
    ("带内拉伸B", _b("stretch")), ("图9卖点", _b("fig9_sell")),
    # sell: fig8
    ("图8上轨", _x("hi8")), ("距图8上轨", _p("dist_hi8")),
    ("图8近上轨(≥上轨×0.99)", _b("near_hi8")), ("图8卖点", _b("fig8_sell")),
    # G1
    ("图7斜率", _p("g7")), ("图7斜率>0", _b("g7_pos")),
    ("图8斜率", _p("g8")), ("图8斜率>0", _b("g8_pos")),
    ("图9位置10日变化", _p("pos9_10", 0)), ("10日变化≥30pp", _b("pos9_10_ok")),
    ("未触图9上轨", _b("not_touch9hi")),
    ("图10斜率", _p("g10")), ("图10斜率>图9", _b("g10_gt_g9")),
    ("G1今日", _b("g1")),
]
HEAD_COLS = (
    "代码", "名称", "T收盘", "类型", "类型来源", "买规则", "卖规则", "G1后不卖", "回落幅度",
    "今日状态", "今日动作", "卖出原因", "触发指标", "买入信号", "买点价", "买点状态", "卖出信号",
    *CLUSTER_COLS,
)
POS_COLS = (
    "买入日", "买入价", "持仓最高收盘", "持仓浮盈", "回落止损价", "距止损",
    "回落触发", "本笔G1已触发", "卖点被G1屏蔽",
)
COLS = (*HEAD_COLS, *(c for c, _ in SNAP_COLS), *POS_COLS)

THRESHOLDS = [
    ("图9近下轨", "收盘 ≤ 图9下轨 × 1.01", "aux_edge买点、B1"),
    ("近下轨上跳沿", "今天近下轨、昨天不是", "aux_edge买点"),
    ("辅助买像", "收盘 < MA20、MA5 < MA20、gap5 ≤ 0、gap20 ≤ 0、图10 未触上轨", "aux_edge买点"),
    ("B1", "图7 或图8 折价 ≤ −8%，且图9近下轨", "买（规则含 B1 时）"),
    ("快形态", "近 60 日任一 40 日涨幅 ≥ 20%", "规则含“快形态不买”时拦买"),
    ("图9近上轨", "收盘 ≥ 图9上轨 × 0.99", "卖点A"),
    ("辅助卖像", "收盘 > MA20、MA5 > MA20、gap5 ≥ 0、gap20 ≥ 0", "卖点A、图8卖点、带内拉伸B"),
    ("卖点A", "图9近上轨上跳沿，且辅助卖像", "图9卖点"),
    ("带内拉伸B", "辅助卖像、10日局部高点、图9位置 ≥ 80%、图9斜率 > 0、图9图10未触上轨、"
                 "vol5分位 ≥ 70%、图7溢价 ≥ 5% 或图10位置 ≥ 90%", "图9卖点、图8卖点"),
    ("图8卖点", "收盘 ≥ 图8上轨 × 0.99 且辅助卖像，或带内拉伸B", "卖（规则为图8卖时）"),
    ("G1", "图7、图8 斜率 > 0，图9位置 10 日升 ≥ 30pp 且 ≥ 80%，未触图9上轨，图10斜率 > 图9斜率",
     "持仓中触发后本笔不再按图9/图8 卖"),
    ("回落触发", "收盘 ≤ 持仓最高收盘 × (1 − 回落幅度)", "卖（规则设回落时）"),
    ("买点价", "收盘 ≤ 图9下轨 × 1.01", "全部标的都填，不是新的成交规则"),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--listing-dir", type=Path, required=True)
    p.add_argument("--as-of", required=True, help="YYYY-MM-DD (T close)")
    p.add_argument("--out", type=Path, default=None,
                   help="default: {listing-dir}/rules7_checklist_{as-of}.csv (md next to it)")
    p.add_argument("--anchor-code", default="SH510300")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument(
        "--cluster-csv",
        type=Path,
        default=None,
        help=(
            "HRP cluster_representatives CSV. Default: prefer "
            "decision_packs/{asof}/*_d080.csv, then untagged; missing → skip"
        ),
    )
    p.add_argument(
        "--rewrite-from-csv",
        action="store_true",
        help="Skip HTML rescan; re-annotate existing checklist CSV with cluster columns and rewrite md",
    )
    return p.parse_args(argv)


def default_cluster_csv(as_of: pd.Timestamp | str) -> Path | None:
    """Prefer fine DIST_T cut (untagged); then d080 / k12."""
    tag = pd.Timestamp(as_of).strftime("%Y%m%d")
    names = (
        f"cluster_representatives_{tag}.csv",
        f"cluster_representatives_{tag}_d080.csv",
        f"cluster_representatives_{tag}_k12.csv",
    )
    for root in CLUSTER_PACK_ROOTS:
        pack = root / tag
        for name in names:
            cand = pack / name
            if cand.is_file():
                return cand
    return None


def load_cluster_frame(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    path = path.expanduser().resolve()
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if df.empty or "code" not in df.columns:
        return None
    out = df.copy()
    out["code"] = out["code"].astype(str).str.upper()
    if "is_representative" in out.columns:
        out["is_representative"] = out["is_representative"].map(
            lambda x: True
            if x is True or str(x).strip().lower() in {"true", "1", "yes"}
            else False
        )
    return out


def _parse_pct_cell(v: Any) -> float | None:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    s = str(v).strip()
    if not s or s in {"—", "-", "nan", "None"}:
        return None
    if s.endswith("%"):
        s = s[:-1]
    try:
        x = float(s)
    except ValueError:
        return None
    # CSV stores 7.4% as "7.4%"; ret20 in cluster file is already percent points
    return x


def attach_cluster_recommendations(
    df: pd.DataFrame,
    cluster: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add HRP cluster columns; mark one 推荐买入 per cluster among today's buys."""
    out = df.copy()
    for col in CLUSTER_COLS:
        out[col] = ""
    out["_cluster"] = pd.NA
    out["_is_rep"] = False
    out["_ret20"] = np.nan
    out["_recommend"] = False
    if out.empty or cluster is None or cluster.empty:
        return out

    # One membership row per code: prefer representative row, then larger |ret20|
    mem = cluster.copy()
    mem["code"] = mem["code"].astype(str).str.upper()
    mem["_abs"] = pd.to_numeric(mem.get("ret20_pct"), errors="coerce").abs()
    if "is_representative" not in mem.columns:
        mem["is_representative"] = False
    mem = mem.sort_values(
        ["code", "is_representative", "_abs"], ascending=[True, False, False], kind="mergesort"
    ).drop_duplicates("code", keep="first")

    by_code = mem.set_index("code")
    codes = out["代码"].astype(str).str.upper()
    out["_cluster"] = codes.map(by_code["cluster"] if "cluster" in by_code.columns else {})
    out["_is_rep"] = codes.map(by_code["is_representative"]).fillna(False).astype(bool)
    out["_ret20"] = codes.map(by_code["ret20_pct"] if "ret20_pct" in by_code.columns else {})
    n_map = by_code["n_members"] if "n_members" in by_code.columns else None
    g_map = by_code["group"] if "group" in by_code.columns else None
    out["聚类"] = out["_cluster"].map(lambda x: "" if pd.isna(x) else str(int(x)))
    out["簇人数"] = (
        codes.map(n_map).map(lambda x: "" if pd.isna(x) else str(int(x))) if n_map is not None else ""
    )
    out["是否簇代表"] = out["_is_rep"].map(lambda b: "是" if b else "")
    if g_map is not None:
        out["簇组"] = codes.map(g_map).map(
            lambda g: {"up": "涨", "down": "跌"}.get(str(g).lower(), str(g) if pd.notna(g) else "")
        )
    out["ret20"] = out["_ret20"].map(lambda x: "" if pd.isna(x) else f"{float(x):+.1f}%")

    # One recommended buy per cluster among 今日动作==买
    buys = out[out["今日动作"] == "买"].copy()
    if buys.empty:
        out["推荐买入"] = ""
        return out

    # Prefer clustered buys; singleton / unmatched codes each form their own bucket
    picks: list[Any] = []
    clustered = buys[buys["_cluster"].notna()]
    for _, sub in clustered.groupby("_cluster", sort=True):
        sub = sub.copy()
        sub["_abs"] = sub["_ret20"].abs()
        sub["_dlo"] = pd.to_numeric(sub.get("_dist_lo9"), errors="coerce")
        sub = sub.sort_values(
            ["_is_rep", "_abs", "_dlo"], ascending=[False, False, True], kind="mergesort"
        )
        picks.append(sub.index[0])
    for idx in buys[buys["_cluster"].isna()].index:
        picks.append(idx)

    out.loc[picks, "_recommend"] = True
    out["推荐买入"] = out["_recommend"].map(lambda b: "是" if b else "")
    # Clear 推荐买入 on non-buys
    out.loc[out["今日动作"] != "买", "推荐买入"] = ""
    out.loc[out["今日动作"] != "买", "_recommend"] = False
    return out


def build_frame(html: Path, listing: Path, anchor_code: str) -> pd.DataFrame:
    from etf_daily.scripts.backtest_fig9_touch import _load_sm, prepare_fig9_touch_frame

    frame, _ = _load_sm().build_feature_frame(html, listing, anchor_code, lookback=10)
    return prepare_fig9_touch_frame(frame)


def scan_one(html: Path, *, listing: Path, as_of: pd.Timestamp, anchor_code: str) -> dict[str, Any]:
    code, name = pool.parse_html_label(html)
    frame = build_frame(html, listing, anchor_code)
    frame = frame.loc[frame.index <= as_of]
    if frame.empty:
        raise ValueError(f"no bars on/before {as_of.date()}")
    if frame.index[-1] != as_of:
        raise ValueError(f"last bar {frame.index[-1].date()} != as-of {as_of.date()}")

    res = rules7.evaluate(code, frame)
    spec, st, snap = res["spec"], res["state"], res["snapshot"]
    px = snap["px"]
    fill_px = st["fill_px"]
    trig = rules7.triggered(snap)
    row: dict[str, Any] = {
        "代码": code,
        "名称": name,
        "T收盘": _px(px),
        "类型": res["type"] + ("（规则不适用）" if code.upper() in rules7.NOT_APPLICABLE else ""),
        "类型来源": res["source"],
        "买规则": spec.buy_text(),
        "卖规则": spec.sell_text(),
        "G1后不卖": _yn(spec.g1_hold),
        "回落幅度": spec.trail_text(),
        "今日状态": "持仓" if st["holding"] else "空仓",
        "今日动作": st["action"],
        "卖出原因": st["why"],
        "触发指标": "；".join(trig),
        "买入信号": _yn(snap["buy_signal"]),
        "卖出信号": _yn(snap["sell_signal"]),
    }
    quote, quote_status = rules7.flat_buy_quote(
        snap, spec, applicable=code.upper() not in rules7.NOT_APPLICABLE,
    )
    row["买点价"] = "" if quote is None else _px(quote)
    row["买点状态"] = quote_status
    row["_html"] = html.name
    row.update({col: fmt(snap) for col, fmt in SNAP_COLS})
    row.update({
        "买入日": _day(st["buy_date"]),
        "买入价": _px(fill_px),
        "持仓最高收盘": _px(st["peak"]),
        "持仓浮盈": _pct(px / fill_px - 1 if fill_px else None),
        "回落止损价": _px(st["stop"]),
        "距止损": _pct(snap["dist_stop"]),
        "回落触发": _yn(snap["trail_hit"]),
        "本笔G1已触发": _yn(st["g1_seen"]),
        "卖点被G1屏蔽": _yn(snap["sell_masked_by_g1"]),
    })
    row.update({
        "_action_rank": ACTION_ORDER.get(st["action"], 9),
        "_n_trig": len(trig),
        "_type": res["type"],
        "_holding": st["holding"],
        "_dist_lo9": snap["dist_lo9"],
        "_dist_stop": snap["dist_stop"],
        **{f"_t_{k}": bool(snap.get(k)) for k in rules7.TRIGGER_LABELS},
    })
    return row


def _worker(payload: dict[str, str]) -> dict[str, Any]:
    return scan_one(Path(payload["html"]), listing=Path(payload["listing"]),
                    as_of=pd.Timestamp(payload["as_of"]), anchor_code=payload["anchor_code"])


def _md_table(df: pd.DataFrame, cols: list[str]) -> list[str]:
    if df.empty:
        return ["（无）", ""]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) if c in r.index else "—" for c in cols) + " |")
    return lines + [""]


def _html_for(code: str, html_name: Any, listing: Path) -> str:
    name = str(html_name or "").strip()
    if name and name.lower() not in {"nan", "none"}:
        return name
    matches = sorted(listing.glob(f"regime_transition_{code}_*_adaptive.html"))
    return matches[0].name if matches else ""


def _gap_to_buy(px_cell: Any, buy_cell: Any) -> float:
    px, buy = _f(px_cell), _f(buy_cell)
    if px is None or buy is None or buy == 0:
        return float("inf")
    return px / buy - 1.0


def _yes_cell(v: Any) -> bool:
    return str(v).strip() == "是"


def _spec_from_buy_text(text: Any) -> rules7.RuleSpec:
    t = str(text or "")
    return rules7.RuleSpec(
        b1="B1" in t,
        fast_block="快形态不买" in t,
        sell="fig9",
        g1_hold=False,
        trail=None,
    )


def annotate_flat_buy_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """Fill 买点价 and 买点状态 on every row from that row's fig9 lower rail."""
    out = df.copy()
    prices: list[str] = []
    statuses: list[str] = []
    for _, row in out.iterrows():
        lo = _f(row.get("图9下轨"))
        snap = {
            "lo9": float("nan") if lo is None else lo,
            "fast": _yes_cell(row.get("快形态(≥20%)")),
            "buy_signal": _yes_cell(row.get("买入信号")),
            "disc_ok": _yes_cell(row.get("图7或图8折价≤-8%")),
            "aux_buy": _yes_cell(row.get("辅助买像")),
            "near_lo": _yes_cell(row.get("图9近下轨(≤下轨×1.01)")),
        }
        applicable = "规则不适用" not in str(row.get("类型", ""))
        quote, status = rules7.flat_buy_quote(
            snap, _spec_from_buy_text(row.get("买规则")), applicable=applicable,
        )
        prices.append("" if quote is None else _px(quote))
        statuses.append(status)
    out["买点价"] = prices
    out["买点状态"] = statuses
    return out


def _format_flat_buy_section(flat: pd.DataFrame, listing: Path) -> list[str]:
    n_buy = int(flat["今日动作"].astype(str).eq("买").sum()) if "今日动作" in flat.columns else 0
    lines = [
        "## 空仓买点价",
        "",
        f"今天买入信号 **{n_buy}** 只。买入信号为 0 则今天不买。",
        "买点价 = 图9下轨 × 1.01，持仓和空仓都填。持仓行的这个价格不产生新的买单。",
        "不是盘中委托。成交仍是信号出现之后的下一根收盘，且只在当时空仓时成交。",
        "已在带内的下一根收盘若仍在带内，不是新的 aux_edge。",
        "",
    ]
    if flat.empty:
        return lines + ["（无）", ""]
    show = flat.copy()
    if "买点价" not in show.columns:
        show["买点价"] = ""
    if "买点状态" not in show.columns:
        show["买点状态"] = ""
    show["_gap"] = [_gap_to_buy(px, buy) for px, buy in zip(show["T收盘"], show["买点价"])]
    show = show.sort_values("_gap", kind="mergesort")
    links = []
    for _, r in show.iterrows():
        html_name = _html_for(str(r["代码"]), r["_html"] if "_html" in show.columns else "", listing)
        links.append(f"[{r['代码']}]({html_name})" if html_name else str(r["代码"]))
    show = show.copy()
    show["代码"] = links
    lines += _md_table(show, ["代码", "名称", "类型", "今日状态", "买点价", "买点状态", "T收盘", "图9下轨", "距图9下轨"])
    return lines


def write_markdown(path: Path, df: pd.DataFrame, *, as_of: pd.Timestamp, listing: Path,
                   skipped_bonds: list[str], errors: list[dict[str, str]],
                   cluster_path: Path | None = None) -> None:
    n = len(df)
    holding = df[df["_holding"]]
    lines = [
        f"# 7 只规则当天指标清单 {as_of.date()}",
        "",
        "> **身份：研究清单，不改生产。** 规则来自 `docs/7只ETF_买卖规律对比.md`。"
        "本清单只看 as-of 当天的指标读数和是否达标；持仓状态按规则从上市推算，只写当天结果。"
        "相似标的按 HRP 层级聚类去重：同簇今日买入信号只标一只 **推荐买入**。",
        "",
        f"目录：`{listing}`。as-of **{as_of.date()}** 收盘出信号，次日收盘成交。"
        f"有效 **{n}** 只，其中持仓 **{len(holding)}** 只、空仓 **{n - len(holding)}** 只。",
    ]
    if cluster_path is not None:
        lines.append(f"聚类：`{cluster_path}`。")
    if skipped_bonds:
        lines.append(f"剔除债券：{'；'.join(skipped_bonds)}。")
    if errors:
        lines.append("跳过：" + "；".join(f"{e['code']} {e['name']}（{e['error']}）" for e in errors) + "。")
    lines.append("")

    buys = df[df["今日动作"] == "买"]
    rec = buys[buys.get("推荐买入", pd.Series(dtype=str)).fillna("").eq("是")] if "推荐买入" in buys.columns else buys.iloc[0:0]
    sells = df[df["今日动作"] == "卖"]
    lines += [f"## 今日推荐买入（同簇只留一只）：{len(rec)} 只", ""]
    lines += _md_table(rec, ["代码", "名称", "聚类", "簇人数", "是否簇代表", "簇组", "ret20",
                             "类型", "买规则", "T收盘", "距图9下轨", "触发指标"])
    lines += [f"## 今日买入信号（去重前）：{len(buys)} 只", ""]
    lines += _md_table(buys, ["代码", "名称", "推荐买入", "聚类", "是否簇代表", "类型", "买规则",
                              "T收盘", "距图9下轨", "触发指标"])
    lines += [f"## 今日卖出（次日收盘卖）：{len(sells)} 只", ""]
    lines += _md_table(sells, ["代码", "名称", "类型", "卖出原因", "买入日", "买入价", "T收盘", "持仓浮盈", "触发指标"])

    lines += _format_hrp_cluster_section(df, cluster_path=cluster_path)

    lines += ["## 关键指标触发统计", "", "| 指标 | 达标只数 | 标的 |", "| --- | ---: | --- |"]
    for key, label in rules7.TRIGGER_LABELS.items():
        hit = df[df[f"_t_{key}"]] if f"_t_{key}" in df.columns else df.iloc[0:0]
        names = "、".join(f"{r['代码']} {r['名称']}" for _, r in hit.iterrows())
        lines.append(f"| {label} | {len(hit)} | {names} |")
    lines.append("")

    hold = holding.sort_values("_dist_stop", na_position="last") if "_dist_stop" in holding.columns else holding
    lines += [f"## 持仓中：{len(hold)} 只", "", "按距回落止损从近到远排；不设回落的排在最后。", ""]
    lines += _md_table(hold, ["代码", "名称", "聚类", "是否簇代表", "类型", "今日动作", "买入日", "买入价",
                              "T收盘", "持仓浮盈", "回落止损价", "距止损", "本笔G1已触发", "卖点被G1屏蔽", "触发指标"])

    lines += _format_flat_buy_section(df, listing)
    near_stop = holding[holding["_dist_stop"] <= NEAR_WARN].sort_values("_dist_stop") if "_dist_stop" in holding.columns else holding.iloc[0:0]
    lines += [f"## 接近触发（{NEAR_WARN:.0%} 以内）", "",
              f"### 持仓、距回落止损 ≤ {NEAR_WARN:.0%}：{len(near_stop)} 只", ""]
    lines += _md_table(near_stop, ["代码", "名称", "聚类", "类型", "T收盘", "回落止损价", "距止损", "持仓浮盈"])

    counts = df["_type"].value_counts() if "_type" in df.columns else pd.Series(dtype=int)
    lines += ["## 类型与规则", "", "| 类型 | 买 | 卖 | G1 后不卖 | 回落卖出 | 只数 |", "| --- | --- | --- | --- | --- | ---: |"]
    for t in TYPE_ORDER:
        s = rules7.TYPE_PARAMS[t]
        lines.append(f"| {t} | {s.buy_text()} | {s.sell_text()} | {'是' if s.g1_hold else '否'} | "
                     f"{s.trail_text()} | {int(counts.get(t, 0))} |")
    lines += ["", "7 只文档标的用各自的专属规则，只数按其所属类型计入；每只实际所用规则见 CSV 的“买规则 / 卖规则 / 回落幅度”。", ""]
    lines += ["## 指标门槛", "", "| 指标 | 门槛 | 用于 |", "| --- | --- | --- |"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in THRESHOLDS]
    lines += [
        "",
        "## 注意",
        "",
        "- 今日动作看的是该标的所用规则：买入信号只在空仓时生效，卖出信号只在持仓且本笔未触发 G1 时生效（G1 后不卖的规则）。",
        "- 买点价 = 图9下轨 × 1.01，持仓和空仓都填。不是盘中委托。成交仍是信号日之后的下一根收盘，且只在当时空仓时成交。",
        "- 已在带内的下一根收盘若仍在带内，不是新的 aux_edge。",
        "- **推荐买入**：同一 HRP 簇内若多只同时出现买入信号，只标一只（优先簇代表，其次 |ret20| 更大，再次更贴近图9下轨）。未进聚类的买入信号各自保留。",
        "- 恒生医药 SZ159892 在文档里没有一套规则能真正赚钱，标为“规则不适用”，信号仅供参考。",
        "- 历史不足的标的归类不可靠，信号只当提示。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_hrp_cluster_section(df: pd.DataFrame, *, cluster_path: Path | None) -> list[str]:
    lines = [
        "## 按 HRP 簇（相似去重）",
        "",
        "默认读 `cluster_representatives_*_d080.csv`（Ward 距离阈值≈0.8）。"
        "同簇标的走势接近，买入时只取一只推荐。",
        "",
    ]
    if cluster_path is None or "_cluster" not in df.columns or df["_cluster"].isna().all():
        note = f"（未挂上聚类{f': {cluster_path}' if cluster_path else ''}）"
        return lines + [note, ""]

    lines += [
        "| 簇 | 人数 | 代表 | ret20 | 组内清单状态 | 今日买入 | 推荐 |",
        "| ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for cid, sub in df[df["_cluster"].notna()].groupby("_cluster", sort=True):
        n_members = ""
        if "簇人数" in sub.columns:
            nm = sub["簇人数"].replace("", pd.NA).dropna()
            if not nm.empty:
                n_members = str(nm.iloc[0])
        reps = sub[sub["_is_rep"]]
        if reps.empty:
            rep_txt = "—"
            ret_txt = "—"
        else:
            r0 = reps.assign(_abs=reps["_ret20"].abs()).sort_values("_abs", ascending=False).iloc[0]
            rep_txt = f"{r0['代码']} {r0['名称']}"
            ret_txt = r0.get("ret20") or "—"
        members = "、".join(
            f"{r['代码']}{r['名称']}({r['今日状态']}/{r['今日动作']})"
            for _, r in sub.sort_values("代码").iterrows()
        )
        buy_codes = "、".join(sub.loc[sub["今日动作"] == "买", "代码"].astype(str))
        rec_codes = "、".join(sub.loc[sub.get("推荐买入", pd.Series(dtype=str)).fillna("").eq("是"), "代码"].astype(str))
        lines.append(
            f"| {int(cid)} | {n_members or sub['代码'].nunique()} | {rep_txt} | {ret_txt} | "
            f"{members} | {buy_codes or '—'} | {rec_codes or '—'} |"
        )
    unmatched = df[df["_cluster"].isna()]
    if not unmatched.empty:
        lines += [
            "",
            f"未进聚类（{len(unmatched)} 只）："
            + "、".join(f"{r['代码']} {r['名称']}" for _, r in unmatched.iterrows())
            + "。",
        ]
    lines.append("")
    return lines


def _ensure_helper_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild private helper columns when rewriting from a public CSV."""
    out = df.copy()
    out["_holding"] = out["今日状态"].astype(str).eq("持仓")
    out["_action_rank"] = out["今日动作"].map(ACTION_ORDER).fillna(9).astype(int)
    out["_n_trig"] = out["触发指标"].fillna("").astype(str).map(
        lambda s: 0 if not s.strip() else len([x for x in s.split("；") if x.strip()])
    )
    if "_type" not in out.columns:
        out["_type"] = out["类型"].astype(str).str.replace("（规则不适用）", "", regex=False)
    out["_dist_lo9"] = out["距图9下轨"].map(_parse_pct_cell)
    out["_dist_lo9"] = out["_dist_lo9"].map(lambda x: np.nan if x is None else x / 100.0)
    out["_dist_stop"] = out["距止损"].map(_parse_pct_cell)
    out["_dist_stop"] = out["_dist_stop"].map(lambda x: np.nan if x is None else x / 100.0)
    for key in rules7.TRIGGER_LABELS:
        label = rules7.TRIGGER_LABELS[key]
        col = f"_t_{key}"
        if col not in out.columns:
            out[col] = out["触发指标"].fillna("").astype(str).map(
                lambda s, lab=label: lab in {x.strip() for x in s.split("；")}
            )
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    listing = args.listing_dir if args.listing_dir.is_absolute() else _PROJECT_ROOT / args.listing_dir
    as_of = pd.Timestamp(args.as_of).normalize()
    tag = as_of.strftime("%Y%m%d")
    out = args.out or (listing / f"rules7_checklist_{tag}.csv")
    out_md = out.with_suffix(".md")
    cluster_path = args.cluster_csv or default_cluster_csv(as_of)
    cluster = load_cluster_frame(cluster_path)
    if cluster_path is not None:
        print(f"[cluster] {cluster_path}" + (" (loaded)" if cluster is not None else " (missing)"), flush=True)

    if args.rewrite_from_csv:
        if not out.is_file():
            print(f"missing checklist CSV: {out}", file=sys.stderr)
            return 2
        df = _ensure_helper_cols(pd.read_csv(out, dtype=str).fillna(""))
        df = annotate_flat_buy_quotes(df)
        # Drop old cluster cols before re-attach
        drop = [c for c in CLUSTER_COLS if c in df.columns]
        df = df.drop(columns=drop, errors="ignore")
        df = attach_cluster_recommendations(df, cluster)
        out_df = df.loc[:, [c for c in COLS if c in df.columns]]
        # Keep any extra public cols from older CSV only if in COLS
        out_df.to_csv(out, index=False, encoding="utf-8-sig")
        write_markdown(out_md, df, as_of=as_of, listing=listing, skipped_bonds=[], errors=[],
                       cluster_path=cluster_path)
        print(f"[OK] {out}", flush=True)
        print(f"[OK] {out_md}", flush=True)
        rec = df[df.get("推荐买入", pd.Series(dtype=str)).fillna("").eq("是")]
        print(f"  推荐买入: {len(rec)}" + "".join(
            f"\n    {r['代码']} {r['名称']} 簇{r.get('聚类','')}" for _, r in rec.iterrows()))
        return 0

    skipped_bonds: list[str] = []
    htmls: list[Path] = []
    for p in sorted(listing.glob("regime_transition_*_adaptive.html")):
        code, name = pool.parse_html_label(p)
        if code == p.stem:
            continue
        if pool.is_bond(code, name) or "债" in name:
            skipped_bonds.append(f"{code} {name}")
        else:
            htmls.append(p)
    print("[skip bond]", ", ".join(skipped_bonds), flush=True)
    payloads = [{"html": str(h), "listing": str(listing), "as_of": as_of.strftime("%Y-%m-%d"),
                 "anchor_code": str(args.anchor_code).upper()} for h in htmls]
    labels = {str(h): pool.parse_html_label(h) for h in htmls}

    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    jobs = max(1, int(args.jobs))
    print(f"[INFO] {len(payloads)} names  jobs={jobs}  as-of {as_of.date()}", flush=True)

    def _collect(payload: dict[str, str], fn) -> None:
        code, name = labels[payload["html"]]
        try:
            row = fn()
            records.append(row)
            print(f"[scan] {code} {row['今日状态']} -> {row['今日动作']}  {row['触发指标']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            errors.append({"code": code, "name": name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[error] {code} {name}: {exc}", flush=True)

    if jobs == 1:
        for p in payloads:
            _collect(p, lambda p=p: _worker(p))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_worker, p): p for p in payloads}
            for fut in as_completed(futs):
                _collect(futs[fut], fut.result)

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(["_action_rank", "_n_trig", "代码"], ascending=[True, False, True],
                            kind="mergesort").reset_index(drop=True)
        df = attach_cluster_recommendations(df, cluster)
    out_df = df.loc[:, list(COLS)] if not df.empty else pd.DataFrame(columns=list(COLS))
    out_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[OK] {out}", flush=True)
    if errors:
        err_path = out.with_name(out.stem + "_skipped.csv")
        pd.DataFrame(errors).to_csv(err_path, index=False, encoding="utf-8-sig")
        print(f"[OK] {err_path} ({len(errors)} skipped)", flush=True)
    if not df.empty:
        write_markdown(out_md, df, as_of=as_of, listing=listing, skipped_bonds=skipped_bonds, errors=errors,
                       cluster_path=cluster_path)
        print(f"[OK] {out_md}", flush=True)
        for act in ("买", "卖"):
            sel = df[df["今日动作"] == act]
            print(f"  今日{act}: {len(sel)}" + "".join(f"\n    {r['代码']} {r['名称']} ({r['触发指标']})"
                                                   for _, r in sel.iterrows()))
        rec = df[df["推荐买入"].fillna("").eq("是")]
        print(f"  推荐买入: {len(rec)}" + "".join(
            f"\n    {r['代码']} {r['名称']} 簇{r.get('聚类','')}" for _, r in rec.iterrows()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
