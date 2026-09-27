#!/usr/bin/env python3
"""20260923 snapshot: dividend-style fig9 swing vs uptrend already started.

Research labels only. Does not change production hold_up.

Rules (documents, not new thresholds except the documented midline example):

- Dividend fingerprint: 平稳44 §3 five structural conditions.
- Uptrend: G1 today, or the close-based unread of G2 (high pos / outside
  the upper rail and not back to 0.50 within 15 bars, or 5 closes outside),
  and fig7/fig8 g still positive.

「进入上涨趋势」means stop the fig9 oscillation ruler. It is not a buy.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.lib.fair_path_band_signals import (  # noqa: E402
    compute_multi_scale_frame,
    load_ohlcv_from_regime_html,
)

DEFAULT_LISTING = Path(
    "~/etf-daily-output/temp/plotly_outputs/20260923_from_listing"
)
_HTML_STEM = re.compile(
    r"^regime_transition_([A-Z]{2}\d+)_(.+)_(\d{8})_(\d{8})_adaptive$"
)

LABEL_DIV = "红利式带内摆动"
LABEL_TREND = "上涨趋势已启动"
LABEL_OTHER = "两者都不是"

# 平稳44 §3. 近上 = 收盘 ≥ 上轨×0.99（黄金文档）.
AGE_MIN_Y = 4.0
WIDTH_LO = 0.10
WIDTH_HI = 0.28
NHI_LO = 1.0
NHI_HI = 3.5
SLOW_SHARE_MIN = 0.45
G_P90_MAX = 0.50
IN_BAND_MIN = 0.88
NEAR_HI = 0.99

# G1. 中轴附近：黄金样例 49%，文档未另给数，用 ≤0.55.
POS_MID_MAX = 0.55
POS_JUMP = 0.30
POS_HIGH = 0.80
POS_BACK = 0.50
UNRECOVERED_BARS = 15
OUTSIDE_STREAK = 5


def parse_html_stem(path: Path) -> tuple[str, str] | None:
    m = _HTML_STEM.match(path.stem)
    if not m:
        return None
    return m.group(1), m.group(2)


def is_bond_name(name: str) -> bool:
    return "债" in name


def fig9_pos(close: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """(ln c − ln lower) / (ln upper − ln lower)."""
    out = np.full(close.shape, np.nan, dtype=float)
    ok = (
        np.isfinite(close)
        & np.isfinite(upper)
        & np.isfinite(lower)
        & (close > 0)
        & (upper > 0)
        & (lower > 0)
        & (upper > lower)
    )
    out[ok] = (np.log(close[ok]) - np.log(lower[ok])) / (
        np.log(upper[ok]) - np.log(lower[ok])
    )
    return out


def _near_hi_edges(close: np.ndarray, upper: np.ndarray) -> int:
    near = np.isfinite(close) & np.isfinite(upper) & (upper > 0) & (close >= upper * NEAR_HI)
    if near.size == 0:
        return 0
    prev = np.empty_like(near)
    prev[0] = False
    prev[1:] = near[:-1]
    return int(np.count_nonzero(near & ~prev))


def _episode_flags(pos: np.ndarray, close: np.ndarray, upper: np.ndarray) -> tuple[bool, bool]:
    """(15 bars after the latest high without a return to 0.50, 5 closes outside).

    The clock starts at the most recent bar with pos >= 80% or a close
    outside the upper rail, matching 「最近一次…之后」. Still sitting on
    that high bar does not start the 15-bar count. A later return to
    0.50 clears it.
    """
    n = int(pos.size)
    last_hi: int | None = None
    for i in range(n):
        p = float(pos[i])
        if not np.isfinite(p):
            continue
        u = float(upper[i])
        if p >= POS_HIGH or (np.isfinite(u) and np.isfinite(close[i]) and close[i] >= u):
            last_hi = i
    clock = False
    if last_hi is not None and (n - 1 - last_hi) >= UNRECOVERED_BARS:
        after = pos[last_hi + 1 :]
        clock = not bool(np.any(np.isfinite(after) & (after <= POS_BACK + 1e-6)))
    streak = 0
    for i in range(n - 1, -1, -1):
        u = float(upper[i])
        if np.isfinite(u) and np.isfinite(close[i]) and close[i] >= u:
            streak += 1
        else:
            break
    return bool(clock), streak >= OUTSIDE_STREAK


def classify_frame(ms: pd.DataFrame) -> dict[str, object]:
    """Label one causal multi-scale frame. Last row is the as-of close."""
    close = pd.to_numeric(ms["px"], errors="coerce").to_numpy(dtype=float)
    g7 = pd.to_numeric(ms["fig7_g"], errors="coerce").to_numpy(dtype=float)
    g8 = pd.to_numeric(ms["fig8_g"], errors="coerce").to_numpy(dtype=float)
    g9 = pd.to_numeric(ms["fig9_g"], errors="coerce").to_numpy(dtype=float)
    g10 = pd.to_numeric(ms["fig10_g"], errors="coerce").to_numpy(dtype=float)
    upper = pd.to_numeric(ms["fig9_upper"], errors="coerce").to_numpy(dtype=float)
    lower = pd.to_numeric(ms["fig9_lower"], errors="coerce").to_numpy(dtype=float)
    pos = fig9_pos(close, upper, lower)
    finite = np.isfinite(pos) & np.isfinite(g9)
    idx = ms.index
    if not np.any(finite):
        return {
            "asof": "",
            "age_y": np.nan,
            "n": 0,
            "w9_full": np.nan,
            "nhi_y": np.nan,
            "slow_share": np.nan,
            "g9_p90": np.nan,
            "in_band": np.nan,
            "pos": np.nan,
            "pos_lag10": np.nan,
            "dpos10": np.nan,
            "g7": np.nan,
            "g8": np.nan,
            "g9": np.nan,
            "g10": np.nan,
            "pass_age": False,
            "pass_width": False,
            "pass_nhi": False,
            "pass_slow": False,
            "pass_inband": False,
            "fingerprint": False,
            "g1": False,
            "unrecovered": False,
            "outside5": False,
            "label": LABEL_OTHER,
            "fail": "无有效图9",
        }
    fin_idx = np.flatnonzero(finite)
    i0 = int(fin_idx[0])
    i1 = int(fin_idx[-1])
    age_y = (idx[i1] - idx[i0]).days / 365.25
    g9_f = g9[finite]
    w = np.full(close.shape, np.nan, dtype=float)
    ok_w = finite & (lower > 0)
    w[ok_w] = upper[ok_w] / lower[ok_w] - 1.0
    w_med = float(np.nanmedian(w[finite]))
    n_hi = _near_hi_edges(close[finite], upper[finite])
    nhi_y = n_hi / age_y if age_y > 0 else float("nan")
    slow = (g9_f > 0.0) & (g9_f <= G_P90_MAX)
    slow_share = float(np.mean(slow))
    g9_p90 = float(np.nanpercentile(g9_f, 90))
    inside = finite & (close > lower) & (close < upper)
    in_band = float(np.sum(inside) / np.sum(finite))

    pass_age = bool(age_y >= AGE_MIN_Y)
    pass_width = bool(WIDTH_LO <= w_med <= WIDTH_HI)
    pass_nhi = bool(np.isfinite(nhi_y) and NHI_LO <= nhi_y <= NHI_HI)
    pass_slow = bool(slow_share >= SLOW_SHARE_MIN and g9_p90 <= G_P90_MAX)
    pass_inband = bool(in_band >= IN_BAND_MIN)
    fingerprint = pass_age and pass_width and pass_nhi and pass_slow and pass_inband

    t = i1
    pos_now = float(pos[t])
    pos_lag = float(pos[t - 10]) if t >= 10 else float("nan")
    dpos = pos_now - pos_lag if np.isfinite(pos_lag) else float("nan")
    g7_now = float(g7[t]) if np.isfinite(g7[t]) else float("nan")
    g8_now = float(g8[t]) if np.isfinite(g8[t]) else float("nan")
    g9_now = float(g9[t])
    g10_now = float(g10[t]) if np.isfinite(g10[t]) else float("nan")
    long_up = bool(g7_now > 0.0 and g8_now > 0.0)
    g1 = bool(
        long_up
        and np.isfinite(pos_lag)
        and pos_lag <= POS_MID_MAX
        and dpos >= POS_JUMP
        and pos_now >= POS_HIGH
        and close[t] < upper[t]
        and np.isfinite(g10_now)
        and g10_now > g9_now
    )
    clock, outside5 = _episode_flags(pos, close, upper)
    unrecovered = bool(long_up and (clock or outside5))
    trend = bool(g1 or unrecovered)
    if trend:
        label = LABEL_TREND
    elif fingerprint:
        label = LABEL_DIV
    else:
        label = LABEL_OTHER
    fails: list[str] = []
    if not pass_age:
        fails.append("年龄")
    if not pass_width:
        fails.append("带宽")
    if not pass_nhi:
        fails.append("近上密度")
    if not pass_slow:
        fails.append("慢斜率")
    if not pass_inband:
        fails.append("带内")
    return {
        "asof": pd.Timestamp(idx[t]).strftime("%Y-%m-%d"),
        "age_y": age_y,
        "n": int(np.sum(finite)),
        "w9_full": w_med,
        "nhi_y": nhi_y,
        "slow_share": slow_share,
        "g9_p90": g9_p90,
        "in_band": in_band,
        "pos": pos_now,
        "pos_lag10": pos_lag,
        "dpos10": dpos,
        "g7": g7_now,
        "g8": g8_now,
        "g9": g9_now,
        "g10": g10_now,
        "pass_age": pass_age,
        "pass_width": pass_width,
        "pass_nhi": pass_nhi,
        "pass_slow": pass_slow,
        "pass_inband": pass_inband,
        "fingerprint": fingerprint,
        "g1": g1,
        "unrecovered": unrecovered,
        "outside5": bool(outside5),
        "label": label,
        "fail": "" if fingerprint else "+".join(fails),
    }


def iter_equity_htmls(listing_dir: Path) -> list[tuple[str, str, Path]]:
    out: list[tuple[str, str, Path]] = []
    for path in sorted(listing_dir.glob("regime_transition_*_adaptive.html")):
        parsed = parse_html_stem(path)
        if parsed is None:
            continue
        code, name = parsed
        if is_bond_name(name):
            continue
        out.append((code, name, path))
    return out


def scan_listing(listing_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    files = iter_equity_htmls(listing_dir)
    for i, (code, name, path) in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {code} {name}", flush=True)
        try:
            close, _ohlcv = load_ohlcv_from_regime_html(path)
            ms = compute_multi_scale_frame(close)
            rec = classify_frame(ms)
            rec["error"] = ""
        except Exception as exc:  # noqa: BLE001 — one bad HTML must not drop the pool
            rec = {
                "asof": "",
                "age_y": np.nan,
                "n": 0,
                "w9_full": np.nan,
                "nhi_y": np.nan,
                "slow_share": np.nan,
                "g9_p90": np.nan,
                "in_band": np.nan,
                "pos": np.nan,
                "pos_lag10": np.nan,
                "dpos10": np.nan,
                "g7": np.nan,
                "g8": np.nan,
                "g9": np.nan,
                "g10": np.nan,
                "pass_age": False,
                "pass_width": False,
                "pass_nhi": False,
                "pass_slow": False,
                "pass_inband": False,
                "fingerprint": False,
                "g1": False,
                "unrecovered": False,
                "outside5": False,
                "label": "",
                "fail": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
        rec["code"] = code
        rec["name"] = name
        rows.append(rec)
    return pd.DataFrame(rows)


def _pct(v: object, digits: int = 1) -> str:
    try:
        x = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(x):
        return "—"
    return f"{x * 100:.{digits}f}%"


def _num(v: object, digits: int = 2) -> str:
    try:
        x = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(x):
        return "—"
    return f"{x:.{digits}f}"


def _yn(v: object) -> str:
    return "是" if bool(v) else ""


def write_markdown(df: pd.DataFrame, path: Path, *, listing_dir: Path) -> None:
    ok = df[df["error"].fillna("") == ""].copy()
    bad = df[df["error"].fillna("") != ""]
    lines = [
        "# 20260923：红利摆动还是已经进入上涨趋势",
        "",
        "研究对照，不是生产规则，也不是买入信号。「上涨趋势已启动」只表示图9 震荡尺停用。",
        "判据：[`平稳44_图9路径摆动画像_20260916.md`](../../../gitee/skfolio_csi300/etf_strategy_clean/documents/平稳44_图9路径摆动画像_20260916.md) §3 五条，",
        "加上 `docs/图9_G1G2活开关_趋势失效规则.md` 的 G1；",
        "G2 不做全池成交重建。未恢复从**最近一次**分位 ≥80% 或收在上轨外的那根 K 算起：满 15 根仍没有回到 50%，或当前连续 5 个收盘在上轨外；并且图7、图8 的 g 仍为正。",
        "中轴附近用 `pos[t−10] ≤ 0.55`（黄金样例 49%）。",
        "",
        f"目录：`{listing_dir}`。名称含「债」的已剔除。有效 {len(ok)} 只。",
        "",
        "| 档 | 只数 |",
        "|---|---:|",
    ]
    for label in (LABEL_DIV, LABEL_TREND, LABEL_OTHER):
        lines.append(f"| {label} | {int((ok['label'] == label).sum())} |")
    lines.append("")

    lines += [
        "## 对照",
        "",
        "红利应落在摆动。黄金只有活开关亮了才落在趋势。",
        "",
    ]
    ctrl = ok[ok["code"].isin(["SH512530", "SH518880"])]
    lines.append(_table(ctrl))
    lines.append("")

    for label, title in (
        (LABEL_DIV, "红利式带内摆动"),
        (LABEL_TREND, "上涨趋势已启动"),
        (LABEL_OTHER, "两者都不是"),
    ):
        block = ok[ok["label"] == label]
        lines += [f"## {title}", ""]
        if block.empty:
            lines += ["（无）", ""]
            continue
        lines.append(_table(block))
        lines.append("")

    if not bad.empty:
        lines += ["## 读失败", ""]
        for _, r in bad.iterrows():
            lines.append(f"- {r['code']} {r['name']}：{r['error']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _table(block: pd.DataFrame) -> str:
    header = (
        "| 代码 | 名称 | 截止 | 年龄 | 带宽 | 近上/年 | 慢斜率 | g P90 | 带内 | "
        "分位 | 10日分位变 | 图7g | 图8g | 图9g | 图10g | G1 | 未恢复 | 未过 |"
    )
    sep = "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|"
    lines = [header, sep]
    ordered = block.sort_values(["pos", "code"], ascending=[False, True])
    for _, r in ordered.iterrows():
        lines.append(
            "| {code} | {name} | {asof} | {age} | {w} | {nhi} | {slow} | {p90} | {ib} | "
            "{pos} | {dpos} | {g7} | {g8} | {g9} | {g10} | {g1} | {un} | {fail} |".format(
                code=r["code"],
                name=r["name"],
                asof=r["asof"] or "—",
                age=_num(r["age_y"], 1),
                w=_pct(r["w9_full"]),
                nhi=_num(r["nhi_y"], 2),
                slow=_pct(r["slow_share"], 0),
                p90=_pct(r["g9_p90"], 0),
                ib=_pct(r["in_band"], 0),
                pos=_pct(r["pos"], 0),
                dpos=_pct(r["dpos10"], 0),
                g7=_pct(r["g7"], 0),
                g8=_pct(r["g8"], 0),
                g9=_pct(r["g9"], 0),
                g10=_pct(r["g10"], 0),
                g1=_yn(r["g1"]),
                un=_yn(r["unrecovered"]),
                fail=r["fail"] or "",
            )
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--listing-dir", type=Path, default=DEFAULT_LISTING)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    listing = args.listing_dir
    df = scan_listing(listing)
    stamp = "20260923"
    csv_path = listing / f"trend_vs_dividend_{stamp}.csv"
    md_path = listing / f"trend_vs_dividend_{stamp}.md"
    df.to_csv(csv_path, index=False)
    write_markdown(df, md_path, listing_dir=listing)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    ok = df[df["error"].fillna("") == ""]
    print(ok["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
