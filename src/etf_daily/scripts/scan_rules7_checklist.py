#!/usr/bin/env python3
"""7-ETF buy/sell rules checklist for a from_listing EOD batch (one as-of day).

Rules from ``docs/7只ETF_买卖规律对比.md``: the 7 documented codes use their own
spec, every other equity ETF is auto-typed (平稳震荡 / 慢趋势 / 高波动主题 /
长期下跌或周期 / 历史不足) and uses that type's spec. Each name is walked from
its first bar to --as-of, so the CSV carries today's state (持仓/空仓), today's
action (买/卖/持有/观望) and the full-history backtest vs buy-and-hold.

Research scan only (does not change production)::

    etf-daily rules7 --listing-dir "$PLOTLY_ROOT/20260929_from_listing"
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.lib import rules7  # noqa: E402
from etf_daily.scripts import backtest_fig12_six_states_pool as pool  # noqa: E402

COLS = (
    "代码", "名称", "T收盘", "类型", "类型来源",
    "买规则", "卖规则", "G1后不卖", "回落幅度",
    "年化波动", "买入持有年化", "买入持有最大回撤", "每年穿图7路径",
    "今日状态", "今日动作", "卖出原因",
    "aux_edge买点", "B1", "快形态", "图9卖点", "图8卖点", "本笔G1已触发",
    "图9下轨", "距图9下轨", "图9带内位置",
    "买入日", "买入价", "持仓最高收盘", "回落止损价", "距止损", "持仓浮盈",
    "回测收益", "回测最大回撤", "回测卡玛", "买入持有收益", "买入持有最大回撤", "买入持有卡玛",
    "笔数", "持仓占比",
)
TYPE_ORDER = [rules7.TYPE_SMOOTH, rules7.TYPE_TREND, rules7.TYPE_HIVOL, rules7.TYPE_CYCLE, rules7.TYPE_SHORT]
ACTION_ORDER = {"买": 0, "卖": 1, "持有": 2, "观望": 3}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--listing-dir", type=Path, required=True)
    p.add_argument("--as-of", required=True, help="YYYY-MM-DD (T close)")
    p.add_argument("--out", type=Path, default=None,
                   help="default: {listing-dir}/rules7_checklist_{as-of}.csv (md next to it)")
    p.add_argument("--anchor-code", default="SH510300")
    p.add_argument("--jobs", type=int, default=8)
    return p.parse_args(argv)


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


def _num(v: Any, digits: int = 2) -> str:
    x = _f(v)
    return "—" if x is None else f"{x:.{digits}f}"


def _yn(b: Any) -> str:
    return "是" if bool(b) else ""


def _day(ts: Any) -> str:
    return "" if ts is None else pd.Timestamp(ts).strftime("%Y-%m-%d")


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
    spec, feat, st, m = res["spec"], res["features"], res["state"], res["masks"]
    last = frame.iloc[-1]
    px = float(frame["px"].ffill().iloc[-1])
    lo9 = _f(last.get("fig9_lower"))
    bh_cal = rules7.calmar(feat["bh_total"], feat["bh_mdd"], feat["bars"])
    fill_px = st["fill_px"]
    typ = res["type"] + ("（规则不适用）" if code.upper() in rules7.NOT_APPLICABLE else "")
    return {
        "代码": code,
        "名称": name,
        "T收盘": _px(px),
        "类型": typ,
        "类型来源": res["source"],
        "买规则": spec.buy_text(),
        "卖规则": spec.sell_text(),
        "G1后不卖": _yn(spec.g1_hold),
        "回落幅度": spec.trail_text(),
        "年化波动": _pct(feat["vol"]),
        "买入持有年化": _pct(feat["bh_cagr"]),
        "买入持有最大回撤": _pct(feat["bh_mdd"]),
        "每年穿图7路径": _num(feat["cross_per_year"], 1),
        "今日状态": "持仓" if st["holding"] else "空仓",
        "今日动作": st["action"],
        "卖出原因": st["why"],
        "aux_edge买点": _yn(m["ae_buy"].iloc[-1]),
        "B1": _yn(m["b1"].iloc[-1]),
        "快形态": _yn(m["fast"].iloc[-1]),
        "图9卖点": _yn(m["ae_sell"].iloc[-1]),
        "图8卖点": _yn(m["fig8_sell"].iloc[-1]),
        "本笔G1已触发": _yn(st["g1_seen"]),
        "图9下轨": _px(lo9),
        "距图9下轨": _pct(px / lo9 - 1 if lo9 else None),
        "图9带内位置": _pct(last.get("fig9_pos"), 0),
        "买入日": _day(st["buy_date"]),
        "买入价": _px(fill_px),
        "持仓最高收盘": _px(st["peak"]),
        "回落止损价": _px(st["stop"]),
        "距止损": _pct(px / st["stop"] - 1 if st["stop"] else None),
        "持仓浮盈": _pct(px / fill_px - 1 if fill_px else None),
        "回测收益": _pct(res["total"]),
        "回测最大回撤": _pct(res["mdd"]),
        "回测卡玛": _num(res["calmar"]),
        "买入持有收益": _pct(feat["bh_total"]),
        "买入持有最大回撤": _pct(feat["bh_mdd"]),
        "买入持有卡玛": _num(bh_cal),
        "笔数": str(res["n_trades"]),
        "持仓占比": _pct(res["exposure"], 0),
        "_action_rank": ACTION_ORDER.get(st["action"], 9),
        "_type_rank": TYPE_ORDER.index(res["type"]),
        "_total": res["total"], "_mdd": res["mdd"], "_calmar": res["calmar"],
        "_bh_total": feat["bh_total"], "_bh_mdd": feat["bh_mdd"], "_bh_calmar": bh_cal,
        "_auto_type": rules7.classify(feat),
        "_doc_type": res["type"],
    }


def _worker(payload: dict[str, str]) -> dict[str, Any]:
    return scan_one(Path(payload["html"]), listing=Path(payload["listing"]),
                    as_of=pd.Timestamp(payload["as_of"]), anchor_code=payload["anchor_code"])


def _md_table(df: pd.DataFrame, cols: list[str]) -> list[str]:
    if df.empty:
        return ["（无）", ""]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) if str(r[c]) else "" for c in cols) + " |")
    return lines + [""]


def write_markdown(path: Path, df: pd.DataFrame, *, as_of: pd.Timestamp, listing: Path,
                   skipped_bonds: list[str], errors: list[dict[str, str]]) -> None:
    n = len(df)
    lines = [
        f"# 7 只规则全池清单 {as_of.date()}",
        "",
        "> **身份：研究清单，不改生产。** 规则来自 `docs/7只ETF_买卖规律对比.md`，是在 7 只上看完全程结果后挑的，"
        "其余标的按自身历史自动归类，归类特征和回测用的是同一段历史，所以回测也带事后成分。",
        "",
        f"目录：`{listing}`。as-of **{as_of.date()}** 收盘出信号，次日收盘成交，每边成本 5 个基点。"
        f"有效 **{n}** 只。",
    ]
    if skipped_bonds:
        lines.append(f"剔除债券：{'；'.join(skipped_bonds)}。")
    if errors:
        lines.append("跳过：" + "；".join(f"{e['code']} {e['name']}（{e['error']}）" for e in errors) + "。")
    lines.append("")

    buy_cols = ["代码", "名称", "类型", "类型来源", "买规则", "T收盘", "aux_edge买点", "B1", "距图9下轨", "回落幅度"]
    sell_cols = ["代码", "名称", "类型", "卖出原因", "买入日", "买入价", "T收盘", "持仓浮盈", "本笔G1已触发"]
    hold_cols = ["代码", "名称", "类型", "买入日", "买入价", "T收盘", "持仓浮盈", "本笔G1已触发", "回落止损价", "距止损"]
    lines += ["## 今日买入（次日收盘买）", ""] + _md_table(df[df["今日动作"] == "买"], buy_cols)
    lines += ["## 今日卖出（次日收盘卖）", ""] + _md_table(df[df["今日动作"] == "卖"], sell_cols)
    lines += ["## 持仓中", ""] + _md_table(df[df["今日动作"] == "持有"], hold_cols)

    lines += ["## 类型与规则", "", "| 类型 | 买 | 卖 | G1 后不卖 | 回落卖出 | 只数 |", "| --- | --- | --- | --- | --- | ---: |"]
    counts = df["_doc_type"].value_counts()
    for t in TYPE_ORDER:
        s = rules7.TYPE_PARAMS[t]
        lines.append(f"| {t} | {s.buy_text()} | {s.sell_text()} | {'是' if s.g1_hold else '否'} | "
                     f"{s.trail_text()} | {int(counts.get(t, 0))} |")
    lines += [
        "",
        "7 只文档标的用各自的专属规则（见下方校验表），只数按其所属类型计入。",
        "",
        "自动归类（按顺序判定，命中即停）：",
        "",
        f"1. 有效 K 线少于 {rules7.MIN_HISTORY_BARS} 根：历史不足，用共性规则。",
        "2. 年化波动 ≥ 30% 且买入持有年化 ≥ 10%：高波动主题。",
        "3. 买入持有最大回撤 ≤ −40% 或买入持有年化 < 5%：长期下跌或周期。",
        "4. 年化波动 < 18% 且每年穿图7 路径 ≥ 9 次：平稳震荡。",
        "5. 其余：慢趋势。",
        "",
    ]

    ok = df.dropna(subset=["_calmar", "_bh_calmar"])
    beat_cal = int((ok["_calmar"] > ok["_bh_calmar"]).sum())
    beat_both = int(((df["_total"] > df["_bh_total"]) & (df["_mdd"] > df["_bh_mdd"])).sum())
    shallower = int((df["_mdd"] > df["_bh_mdd"]).sum())
    lines += [
        "## 全池回测：规则 vs 买入持有",
        "",
        f"卡玛（年化收益 / |最大回撤|）高于买入持有：**{beat_cal} / {len(ok)}** 只；"
        f"收益和回撤都好于买入持有：**{beat_both} / {n}** 只；回撤比买入持有浅：**{shallower} / {n}** 只。",
        "",
    ]
    by_type = []
    for t in TYPE_ORDER:
        g = df[df["_doc_type"] == t]
        if g.empty:
            continue
        gc = g.dropna(subset=["_calmar", "_bh_calmar"])
        by_type.append(f"| {t} | {len(g)} | {gc['_calmar'].median():.2f} | {gc['_bh_calmar'].median():.2f} | "
                       f"{int((gc['_calmar'] > gc['_bh_calmar']).sum())} |")
    lines += ["| 类型 | 只数 | 规则中位卡玛 | 买入持有中位卡玛 | 卡玛胜买入持有 |", "| --- | ---: | ---: | ---: | ---: |",
              *by_type, ""]
    bt_cols = ["代码", "名称", "类型", "类型来源", "回测收益", "回测最大回撤", "回测卡玛",
               "买入持有收益", "买入持有最大回撤", "买入持有卡玛", "笔数", "持仓占比"]
    bt = df.sort_values(["_type_rank", "_calmar"], ascending=[True, False], na_position="last")
    lines += _md_table(bt, bt_cols)

    doc = df[df["类型来源"] == "专属"]
    lines += ["## 7 只归类校验", "", "自动归类规则用在 7 只文档标的上，应与文档类型一致。", "",
              "| 代码 | 名称 | 文档类型 | 自动归类 | 一致 | 年化波动 | 买入持有年化 | 买入持有最大回撤 | 每年穿图7路径 |",
              "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for _, r in doc.iterrows():
        lines.append(f"| {r['代码']} | {r['名称']} | {r['_doc_type']} | {r['_auto_type']} | "
                     f"{'是' if r['_doc_type'] == r['_auto_type'] else '否'} | {r['年化波动']} | {r['买入持有年化']} | "
                     f"{r['买入持有最大回撤']} | {r['每年穿图7路径']} |")
    lines += [
        "",
        "## 注意",
        "",
        "- 回落幅度对结果很敏感，各类型的档位只当大致范围。",
        "- 恒生医药 SZ159892 在文档里没有一套规则能真正赚钱，标为“规则不适用”，信号仅供参考。",
        "- 历史不足的标的归类和回测都不可靠，信号只当提示。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    listing = args.listing_dir if args.listing_dir.is_absolute() else _PROJECT_ROOT / args.listing_dir
    as_of = pd.Timestamp(args.as_of).normalize()
    tag = as_of.strftime("%Y%m%d")
    out = args.out or (listing / f"rules7_checklist_{tag}.csv")
    out_md = out.with_suffix(".md")

    skipped_bonds = []
    htmls = []
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
            print(f"[scan] {code} {row['类型']} {row['今日状态']} -> {row['今日动作']}", flush=True)
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
        df = df.sort_values(["_action_rank", "_type_rank", "代码"], kind="mergesort").reset_index(drop=True)
    out_df = df.loc[:, list(COLS)] if not df.empty else pd.DataFrame(columns=list(COLS))
    out_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[OK] {out}", flush=True)
    if errors:
        err_path = out.with_name(out.stem + "_skipped.csv")
        pd.DataFrame(errors).to_csv(err_path, index=False, encoding="utf-8-sig")
        print(f"[OK] {err_path} ({len(errors)} skipped)", flush=True)
    if not df.empty:
        write_markdown(out_md, df, as_of=as_of, listing=listing, skipped_bonds=skipped_bonds, errors=errors)
        print(f"[OK] {out_md}", flush=True)
        for act in ("买", "卖"):
            sel = df[df["今日动作"] == act]
            print(f"  今日{act}: {len(sel)}" + "".join(f"\n    {r['代码']} {r['名称']} ({r['类型']})"
                                                   for _, r in sel.iterrows()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
