#!/usr/bin/env python3
"""Collect the rules7 facts for one ETF from the latest *_from_listing directory.

Read-only, stdlib only. Usage: ask_etf.py CODE [--dir DIR] [--root PLOTLY_ROOT]
CODE may be SH520870, sh520870 or 520870.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[4]
CHECKLIST_KEYS = [
    "名称", "T收盘", "类型", "类型来源", "买规则", "卖规则", "回落幅度",
    "今日状态", "今日动作", "卖出原因", "触发指标", "买入信号",
    "买点价", "买点状态", "EWMA买点价", "EWMA买点状态",
    "图9下轨", "距图9下轨", "图9 EWMA下轨", "距图9 EWMA下轨",
    "近下轨上跳沿", "辅助买像", "aux_edge买点", "B1",
    "近60日最大40日涨幅", "快形态(≥20%)", "快形态剩余K线",
    "ret20", "MA5", "MA20", "gap20", "图7折价", "图8折价",
    "聚类", "簇人数", "是否簇代表", "推荐买入",
    "买入日", "买入价", "持仓浮盈", "持仓最高收盘", "回落止损价", "距止损",
]
SUMMARY_KEYS = [
    "method", "train_bars", "skipped_train", "bars",
    "compound", "bh", "edge", "n_closed", "win", "open_pos",
]


def plotly_root(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    if os.environ.get("PLOTLY_ROOT"):
        return Path(os.environ["PLOTLY_ROOT"])
    cfg = REPO / "config.env"
    if cfg.exists():
        for line in cfg.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*(?:export\s+)?PLOTLY_ROOT=(.*)", line)
            if m:
                return Path(os.path.expandvars(m.group(1).strip().strip("\"'")))
    sys.exit("PLOTLY_ROOT not found: pass --root or set it in config.env")


def latest_dir(root: Path) -> Path:
    dirs = [
        p for p in root.glob("*_from_listing")
        if re.fullmatch(r"\d{8}_from_listing", p.name) and glob.glob(str(p / "rules7_checklist_*.csv"))
    ]
    if not dirs:
        sys.exit(f"no *_from_listing directory with a rules7 checklist under {root}; run `etf-daily rules7` first")
    return max(dirs, key=lambda p: p.name)


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize(code: str, rows: list[dict]) -> str:
    digits = re.sub(r"\D", "", code)
    hits = [r["代码"] for r in rows if r["代码"].upper() == code.upper()] or [
        r["代码"] for r in rows if digits and r["代码"].endswith(digits)
    ]
    if not hits:
        matches = [r["代码"] for r in rows if code in r["名称"]]
        if len(matches) == 1:
            return matches[0]
        names = [f'{r["代码"]} {r["名称"]}' for r in rows if code in r["名称"]]
        sys.exit(f"{code} is not in the checklist pool." + (f" Name matches: {names}" if names else ""))
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("code")
    ap.add_argument("--dir")
    ap.add_argument("--root")
    a = ap.parse_args()

    d = Path(a.dir) if a.dir else latest_dir(plotly_root(a.root))
    checklist = sorted(d.glob("rules7_checklist_*.csv"))[-1]
    rows = read_csv(checklist)
    code = normalize(a.code, rows)
    row = next(r for r in rows if r["代码"] == code)

    today = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
    print(f"目录: {d}")
    print(f"清单: {checklist.name}  修改时间 {datetime.fromtimestamp(checklist.stat().st_mtime):%Y-%m-%d %H:%M}")
    if d.name[:8] != today:
        print(f"警告: 目录日期 {d.name[:8]} 不是今天 {today}，数据可能过期")
    n_buy = sum(r["买入信号"] == "是" for r in rows)
    print(f"池内 {len(rows)} 只，今日买入信号 {n_buy} 只")

    reports = sorted(d.glob("rules7_trade_report_*.md"), key=lambda p: p.stat().st_mtime)
    report = reports[-1] if reports else None
    if report:
        text = report.read_text(encoding="utf-8")
        print(f"\n日报: {report.name}")
        for line in text.splitlines():
            if line.startswith(("读数:", "生成时间:", "今日全池买点")):
                print("  " + line)
        print("日报里提到该代码的行:")
        for line in text.splitlines():
            if line.startswith(f"| {code} ") or f"[{code}]" in line and line.startswith("|"):
                print("  " + line[:300])
    else:
        print("\n日报: 无")

    print(f"\n== {code} 清单 ==")
    for k in CHECKLIST_KEYS:
        if row.get(k):
            print(f"{k} = {row[k]}")

    summary = d / "batch_summary.csv"
    if summary.exists():
        s = next((r for r in read_csv(summary) if r.get("code") == code), None)
        print(f"\n== {code} 上市以来回测 (batch_summary) ==")
        if s:
            for k in SUMMARY_KEYS:
                print(f"{k} = {s.get(k)}")
        else:
            print("无")

    trades = d / "trades" / f"{code}_trades.csv"
    if trades.exists():
        t = read_csv(trades)
        print(f"\n== {code} 最近交易 (共 {len(t)} 行) ==")
        for r in t[-6:]:
            print(f'{r["side"]} {r["date"]} {float(r["px"]):.4f} {r["why"]}'
                  + (f' 收益 {float(r["fwd_ret"]):+.1%}' if r.get("fwd_ret") else ""))


if __name__ == "__main__":
    main()
