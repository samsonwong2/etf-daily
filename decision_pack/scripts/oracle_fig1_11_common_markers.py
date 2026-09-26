#!/usr/bin/env python3
"""Look-ahead (oracle) buy/sell pivots + fig1–11 common-marker rates.

Same document type as
``~/etf-daily-output/temp/plotly_outputs/.../fig3_11_bt_SH518880/oracle_fig1_11_common_markers/``.

Rule (look-ahead, not tradable):
  ±10/10 local extremum + next 40 bars peak ≥ +8% (buy) / trough ≤ −8% (sell);
  thin same-side clusters with gap ≥ 8 bars.

Example::

    PYTHONPATH=. python decision_pack/scripts/oracle_fig1_11_common_markers.py --code SH512690
    PYTHONPATH=. python decision_pack/scripts/oracle_fig1_11_common_markers.py \\
      --html ~/etf-daily-output/temp/plotly_outputs/20260824_from_listing/regime_transition_SH512530_沪深300红利ETF建信_20190923_20260824_adaptive.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runtime_paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from decision_pack.src.fair_path_band_signals import (  # noqa: E402
    _decode_plotly_bdata,
    _extract_plotly_traces,
    _plotly_x_to_index,
    build_quiet_discount_frame,
    find_regime_html,
    load_ohlcv_from_regime_html,
)
from decision_pack.src.regime_transition_plot import (  # noqa: E402
    compute_volatility_regime_frame,
)
from decision_pack.src.vol_need_return_board import (  # noqa: E402
    ANCHOR_ERP_ANN,
    fair_need_ann_series,
    gap_realized_minus_need,
)

DEFAULT_LISTING = PLOTLY_OUTPUTS_DIR / "20260824_from_listing"
FIG1_NAMES = [
    "上行切换",
    "下行切换",
    "上涨买",
    "上涨卖",
    "提前进↑",
    "提前进↓",
    "提前离↑(X)",
    "ED告警(模型滞后)",
    "ED观察(已对齐↓)",
    "RU恢复上涨(诊断)",
    "早期底反确认(T+3)",
    "早期顶反确认(T+3)",
    "最终底反确认(T+10)",
    "最终顶反确认(T+10)",
]
PATH_FIGS = [
    ("fig7", "图7·2年"),
    ("fig8", "图8·1年"),
    ("fig9", "图9·6月"),
    ("fig10", "图10·3月"),
    ("fig11", "图11·1月"),
]
KEY_FEATS = [
    "图1|收盘<MA20",
    "图1|收盘>MA20",
    "图1|MA5<MA20",
    "图1|MA5>MA20",
    "图3|vol5_pct≤0.30(calm)",
    "图3|vol5_pct≥0.70(cluster+)",
    "图3|rv5<rv20",
    "图3|rv5>rv20",
    "图4|gap20≤0",
    "图4|gap20≥0",
    "图5|gap5≤0",
    "图5|gap5≥0",
    "图7·2年|g>0",
    "图7·2年|折价 gap≤-5%",
    "图7·2年|触下轨",
    "图7·2年|溢价 gap≥0",
    "图8·1年|g>0",
    "图9·6月|触下轨",
    "图9·6月|触上轨",
    "图10·3月|触下轨",
    "图10·3月|触上轨",
    "图11·1月|触下轨",
    "图11·1月|触上轨",
    "组合|长尺双g>0",
    "组合|短尺触上轨(S1)",
    "组合|短尺触下轨",
    "组合|quiet_discount入场",
    "图1|上涨买",
    "图1|上涨卖",
    "图1|上行切换",
    "图1|下行切换",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--html", type=Path, default=None, help="regime_transition HTML")
    p.add_argument(
        "--code",
        default=None,
        help="symbol (e.g. SH512690); used to find HTML under --listing-dir if --html omitted",
    )
    p.add_argument("--listing-dir", type=Path, default=DEFAULT_LISTING)
    p.add_argument("--anchor-code", default="SH510300")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--left", type=int, default=10)
    p.add_argument("--right", type=int, default=10)
    p.add_argument("--fwd", type=int, default=40)
    p.add_argument("--buy-thr", type=float, default=0.08)
    p.add_argument("--sell-thr", type=float, default=0.08)
    p.add_argument("--thin-gap", type=int, default=8)
    args = p.parse_args(argv)
    if args.html is None and not args.code:
        p.error("need --html or --code")
    return args


def _resolve_path(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


def _code_from_html(html_path: Path) -> str:
    name = html_path.name
    # regime_transition_SH512530_...
    parts = name.split("_")
    if len(parts) >= 3 and parts[0] == "regime" and parts[1] == "transition":
        return parts[2]
    raise ValueError(f"cannot parse code from {html_path.name}")


def resolve_html_and_code(args: argparse.Namespace) -> tuple[Path, str, Path]:
    """Return (html_path, CODE, listing_dir)."""
    listing = _resolve_path(args.listing_dir)
    if args.html is not None:
        html_path = _resolve_path(args.html)
        if not html_path.exists():
            raise FileNotFoundError(html_path)
        code = (args.code or _code_from_html(html_path)).upper()
        return html_path, code, listing
    code = str(args.code).upper()
    html_path = find_regime_html(listing, code)
    if html_path is None:
        raise FileNotFoundError(
            f"no regime_transition_{code}_*_adaptive.html under {listing}"
        )
    return html_path, code, listing


def _series_from_marker(tr: dict[str, Any]) -> pd.DatetimeIndex:
    x = tr.get("x")
    if isinstance(x, dict):
        x = _decode_plotly_bdata(x)
    idx = _plotly_x_to_index(x)
    return pd.DatetimeIndex(pd.to_datetime(idx)).normalize()


def _bool_fill(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(bool)


def _thin(mask: np.ndarray, gap: int) -> np.ndarray:
    out = np.zeros_like(mask)
    last = -(10**9)
    for i, v in enumerate(mask):
        if v and i - last >= gap:
            out[i] = True
            last = i
    return out


def _oracle_masks(
    px: np.ndarray,
    *,
    left: int,
    right: int,
    fwd: int,
    buy_thr: float,
    sell_thr: float,
    thin_gap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(px)
    is_buy = np.zeros(n, dtype=bool)
    is_sell = np.zeros(n, dtype=bool)
    fwd_up = np.full(n, np.nan)
    fwd_dn = np.full(n, np.nan)
    for i in range(left, n - max(right, 1)):
        j1 = min(n, i + 1 + fwd)
        if j1 <= i + 1 or not np.isfinite(px[i]):
            continue
        window = px[i - left : i + right + 1]
        fut = px[i + 1 : j1]
        if len(fut) == 0:
            continue
        up = float(np.nanmax(fut) / px[i] - 1.0)
        dn = float(np.nanmin(fut) / px[i] - 1.0)
        fwd_up[i] = up
        fwd_dn[i] = dn
        if px[i] == np.nanmin(window) and up >= buy_thr:
            is_buy[i] = True
        if px[i] == np.nanmax(window) and dn <= -sell_thr:
            is_sell[i] = True
    return _thin(is_buy, thin_gap), _thin(is_sell, thin_gap), fwd_up, fwd_dn


def _near_hit(dates_a: set[str], dates_b: set[str], index: pd.DatetimeIndex, tol: int) -> int:
    idx = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(index)}
    hit = 0
    n = len(index)
    for d in dates_a:
        i = idx.get(d)
        if i is None:
            continue
        for k in range(max(0, i - tol), min(n, i + tol + 1)):
            if index[k].strftime("%Y-%m-%d") in dates_b:
                hit += 1
                break
    return hit


def _fmt_row(r: pd.Series) -> str:
    return (
        f"| {r['feature']} | {r['base_rate']:.0%} | {r['buy_rate']:.0%} | "
        f"{r['sell_rate']:.0%} | {r['buy_lift']:+.0%} | {r['sell_lift']:+.0%} |"
    )


def _feat(stats: pd.DataFrame, name: str) -> pd.Series | None:
    hit = stats.loc[stats["feature"] == name]
    if hit.empty:
        return None
    return hit.iloc[0]


def _pct3(stats: pd.DataFrame, name: str) -> str:
    r = _feat(stats, name)
    if r is None:
        return "n/a"
    return f"买 {r['buy_rate']:.0%} / 卖 {r['sell_rate']:.0%} / 基线 {r['base_rate']:.0%}"


def build_conclusion(stats: pd.DataFrame, summary: dict[str, Any]) -> str:
    """Data-driven Chinese narrative; same section titles as SH518880 report."""
    n_buy = summary["n_oracle_buy"]
    n_sell = summary["n_oracle_sell"]
    exact_b = summary["oracle_buy_exact_prod_buy"]
    exact_s = summary["oracle_sell_exact_prod_sell"]
    near_b = summary["oracle_buy_near_prod_buy_±5d"]
    near_s = summary["oracle_sell_near_prod_sell_±5d"]

    def rate(name: str, side: str) -> float:
        r = _feat(stats, name)
        if r is None:
            return float("nan")
        return float(r["buy_rate"] if side == "buy" else r["sell_rate"])

    def lift(name: str, side: str) -> float:
        r = _feat(stats, name)
        if r is None:
            return float("nan")
        return float(r["buy_lift"] if side == "buy" else r["sell_lift"])

    below20_b = rate("图1|收盘<MA20", "buy")
    above20_s = rate("图1|收盘>MA20", "sell")
    gap5_b = rate("图5|gap5≤0", "buy")
    gap5_s = rate("图5|gap5≥0", "sell")
    gap20_b = rate("图4|gap20≤0", "buy")
    gap20_s = rate("图4|gap20≥0", "sell")
    g7_b_lift = lift("图7·2年|g>0", "buy")
    dual_g_b_lift = lift("组合|长尺双g>0", "buy")
    s1_s = rate("组合|短尺触上轨(S1)", "sell")
    lo_b = rate("组合|短尺触下轨", "buy")
    calm_b_lift = lift("图3|vol5_pct≤0.30(calm)", "buy")
    cluster_s_lift = lift("图3|vol5_pct≥0.70(cluster+)", "sell")

    buy_ma = "多在 **MA20 下方 / MA5&lt;MA20**" if below20_b >= 0.7 else "均线位置分化，不完全钉在 MA20 下方"
    sell_ma = "多在 **MA20 上方 / MA5&gt;MA20**" if above20_s >= 0.7 else "均线位置分化，不完全钉在 MA20 上方"
    buy_gap = "近端 **gap≤0**（没赚够公平补偿）更常见" if gap5_b >= 0.7 or gap20_b >= 0.6 else "近端 gap 作为买点标识没有黄金那么干净"
    sell_gap = "近端 **gap≥0 / 超额** 更常见" if gap5_s >= 0.7 or gap20_s >= 0.6 else "近端 gap 作为卖点标识弱于均线/短尺溢价"
    if g7_b_lift >= 0.05:
        buy_trend = "常见 **长尺 g&gt;0（趋势未坏）+ 对 path 折价/触下轨**"
    elif g7_b_lift <= -0.05:
        buy_trend = "长尺 **g&gt;0 低于基线**（谷底更常落在长尺已转弱 + 折价/触下轨），与黄金「趋势未坏再抄」不同"
    else:
        buy_trend = "长尺趋势方向区分度弱，更靠折价/触下轨"
    sell_s1 = "**触上轨（S1）与短尺溢价** 是更干净的卖侧标识" if s1_s >= 0.2 else "短尺触上轨抬升有限，更靠溢价/均线"
    buy_band = "短尺更容易 **触下轨 / 折价**，而不是触上轨" if lo_b >= 0.1 else "短尺触下轨不是多数买点的必要条件"
    if calm_b_lift >= 0.05:
        vol_buy = "更常落在 **低/中波**，短波未爆炸（`rv5&lt;rv20` 偏多）"
    else:
        vol_buy = "波动分位几乎无区分度（calm/cluster 买点日接近基线）"
    vol_sell = "更常伴随 **高分位波动** 或短波抬升" if cluster_s_lift >= 0.08 else "高波不是卖点的硬条件"

    if dual_g_b_lift >= 0.05:
        buy_reuse = "长尺未坏（g7/g8&gt;0）+ 价低于长/中尺 path + 近端 gap≤0 + 波动未极端 + 短尺未触上轨"
    else:
        buy_reuse = "近端 gap≤0 + 价在 MA20 下 + 长/短尺折价或触下；长尺双 g&gt;0 不是硬条件（买点日甚至低于基线）"
    sell_reuse = "短尺触上轨或明显溢价 + 近端 gap≥0；长尺双 g 转负是更晚、更重的确认（本标的卖点日长尺常仍 g&gt;0）"
    lines = [
        "",
        "## 结论（人话）",
        "",
        "### 买点公共画像（图1→11）",
        f"1. **图1**：价格{buy_ma}；生产绿三角「上涨买」与事后谷底精确重合 {exact_b}/{n_buy}（±5 日邻近 {near_b}）——hold_up 买 ≠ 最优谷底。",
        f"2. **图3**：{vol_buy}。",
        f"3. **图4/5**：{buy_gap}。",
        f"4. **图7–8**：{buy_trend}。",
        f"5. **图9–11**：{buy_band}。",
        "",
        "### 卖点公共画像（图1→11）",
        f"1. **图1**：价格{sell_ma}；红三角「上涨卖」精确重合 {exact_s}/{n_sell}（±5 日邻近 {near_s}）。",
        f"2. **图3**：{vol_sell}。",
        f"3. **图4/5**：{sell_gap}。",
        "4. **图7–8**：趋势未必立刻坏，但 **溢价 / 靠上轨** 增多。",
        f"5. **图9–11**：{sell_s1}。",
        "",
        "### 跨图可复用公共标识",
        f"- **买**：{buy_reuse}。",
        f"- **卖**：{sell_reuse}。",
        "",
        "图2（成交量）与图6（策略净值）未纳入布尔标识统计（量能需另定阈值；净值是结果不是条件）。",
        "",
        "### 关键数字（买点日% / 卖点日% / 全样本%）",
        "",
    ]
    for f in KEY_FEATS:
        if _feat(stats, f) is None:
            continue
        lines.append(f"- {f}：{_pct3(stats, f)}")
    lines.append("")
    return "\n".join(lines)


def run_study(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    html_path, code, listing = resolve_html_and_code(args)
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = listing / f"fig3_11_bt_{code}" / "oracle_fig1_11_common_markers"
    else:
        out_dir = _resolve_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    close, ohlcv = load_ohlcv_from_regime_html(html_path)
    anchor_html = find_regime_html(listing, args.anchor_code)
    if anchor_html is None:
        raise FileNotFoundError(f"anchor {args.anchor_code} HTML not found under {listing}")
    close_a, _ = load_ohlcv_from_regime_html(anchor_html)
    vf_a = compute_volatility_regime_frame(close_a)
    vf_a["as_of"] = pd.to_datetime(vf_a["as_of"]).dt.normalize()
    vf_a = vf_a.set_index("as_of")
    frame = build_quiet_discount_frame(
        close,
        ohlcv,
        rv20_anchor=pd.to_numeric(vf_a["rv20"], errors="coerce"),
        rv5_anchor=pd.to_numeric(vf_a["rv5"], errors="coerce"),
    )
    vol = compute_volatility_regime_frame(close)
    vol["as_of"] = pd.to_datetime(vol["as_of"]).dt.normalize()
    vol = vol.set_index("as_of").reindex(close.index)
    need5 = fair_need_ann_series(
        pd.to_numeric(vol["rv5"], errors="coerce"),
        pd.to_numeric(vf_a["rv5"], errors="coerce").reindex(close.index),
        erp_ann=ANCHOR_ERP_ANN,
    )
    g5 = gap_realized_minus_need(close, need5, bars=5)
    frame["r_need_ann_5"] = need5.reindex(close.index)
    frame["gap_5d"] = pd.to_numeric(g5["gap"], errors="coerce").reindex(close.index)
    frame["ma5"] = close.rolling(5).mean()
    frame["ma10"] = close.rolling(10).mean()
    frame["ma20"] = close.rolling(20).mean()
    frame["above_ma20"] = close > frame["ma20"]
    frame["below_ma20"] = close < frame["ma20"]

    html_txt = html_path.read_text(encoding="utf-8")
    traces = _extract_plotly_traces(html_txt)
    marker_sets: dict[str, set[str]] = {}
    for tr in traces:
        name = str(tr.get("name") or "")
        if tr.get("mode") and "markers" in str(tr.get("mode")):
            marker_sets[name] = set(_series_from_marker(tr).strftime("%Y-%m-%d"))
    for k, v in sorted(marker_sets.items(), key=lambda kv: -len(kv[1])):
        print(f"[INFO] marker {k}: {len(v)}")

    for name in FIG1_NAMES:
        s = marker_sets.get(name, set())
        frame[f"fig1_{name}"] = [d.strftime("%Y-%m-%d") in s for d in frame.index]

    px = frame["px"].astype(float).to_numpy()
    is_buy, is_sell, fwd_up, fwd_dn = _oracle_masks(
        px,
        left=args.left,
        right=args.right,
        fwd=args.fwd,
        buy_thr=args.buy_thr,
        sell_thr=args.sell_thr,
        thin_gap=args.thin_gap,
    )
    frame["oracle_buy"] = is_buy
    frame["oracle_sell"] = is_sell
    print(f"[INFO] oracle buys={int(is_buy.sum())} sells={int(is_sell.sum())}")

    flags = pd.DataFrame(index=frame.index)
    for name in FIG1_NAMES:
        flags[f"图1|{name}"] = frame[f"fig1_{name}"].astype(bool)
    flags["图1|收盘<MA20"] = _bool_fill(frame["below_ma20"])
    flags["图1|收盘>MA20"] = _bool_fill(frame["above_ma20"])
    flags["图1|MA5<MA20"] = _bool_fill(frame["ma5"] < frame["ma20"])
    flags["图1|MA5>MA20"] = _bool_fill(frame["ma5"] > frame["ma20"])
    flags["图3|vol5_pct≤0.30(calm)"] = _bool_fill(frame["vol5_pct_120d"] <= 0.30)
    flags["图3|vol5_pct≥0.70(cluster+)"] = _bool_fill(frame["vol5_pct_120d"] >= 0.70)
    flags["图3|vol5_pct≥0.90(extreme)"] = _bool_fill(frame["vol5_pct_120d"] >= 0.90)
    flags["图3|rv5<rv20"] = _bool_fill(frame["rv5"] < frame["rv20"])
    flags["图3|rv5>rv20"] = _bool_fill(frame["rv5"] > frame["rv20"])
    flags["图4|gap20≤0"] = _bool_fill(frame["gap_20d"] <= 0)
    flags["图4|gap20≥0"] = _bool_fill(frame["gap_20d"] >= 0)
    flags["图4|gap20≤-2%"] = _bool_fill(frame["gap_20d"] <= -0.02)
    flags["图4|gap20≥+2%"] = _bool_fill(frame["gap_20d"] >= 0.02)
    flags["图4|need20≤4.3%"] = _bool_fill(frame["r_need_ann"] <= ANCHOR_ERP_ANN)
    flags["图5|gap5≤0"] = _bool_fill(frame["gap_5d"] <= 0)
    flags["图5|gap5≥0"] = _bool_fill(frame["gap_5d"] >= 0)
    for fig, label in PATH_FIGS:
        flags[f"{label}|g>0"] = _bool_fill(frame[f"{fig}_g"] > 0)
        flags[f"{label}|g<0"] = _bool_fill(frame[f"{fig}_g"] < 0)
        flags[f"{label}|折价 gap≤-5%"] = _bool_fill(frame[f"{fig}_gap"] <= -0.05)
        flags[f"{label}|折价 gap≤-12%"] = _bool_fill(frame[f"{fig}_gap"] <= -0.12)
        flags[f"{label}|溢价 gap≥0"] = _bool_fill(frame[f"{fig}_gap"] >= 0)
        flags[f"{label}|溢价 gap≥+5%"] = _bool_fill(frame[f"{fig}_gap"] >= 0.05)
        flags[f"{label}|触下轨"] = _bool_fill(frame[f"{fig}_touch_lo"])
        flags[f"{label}|触上轨"] = _bool_fill(frame[f"{fig}_touch_hi"])
    flags["组合|长尺双g>0"] = _bool_fill((frame["fig7_g"] > 0) & (frame["fig8_g"] > 0))
    flags["组合|长尺双g<0"] = _bool_fill((frame["fig7_g"] < 0) & (frame["fig8_g"] < 0))
    flags["组合|短尺触上轨(S1)"] = _bool_fill(
        frame["fig9_touch_hi"].fillna(False) | frame["fig10_touch_hi"].fillna(False)
    )
    flags["组合|短尺触下轨"] = _bool_fill(
        frame["fig9_touch_lo"].fillna(False) | frame["fig10_touch_lo"].fillna(False)
    )
    flags["组合|quiet_discount入场"] = _bool_fill(frame["qd_entry"])

    valid = np.isfinite(fwd_up) & np.isfinite(fwd_dn)
    base = flags.loc[valid]
    buy_f = flags.loc[is_buy]
    sell_f = flags.loc[is_sell]
    rows = []
    for col in flags.columns:
        b_rate = float(base[col].mean()) if len(base) else np.nan
        buy_rate = float(buy_f[col].mean()) if len(buy_f) else np.nan
        sell_rate = float(sell_f[col].mean()) if len(sell_f) else np.nan
        rows.append(
            {
                "feature": col,
                "base_rate": b_rate,
                "buy_rate": buy_rate,
                "sell_rate": sell_rate,
                "buy_lift": buy_rate - b_rate,
                "sell_lift": sell_rate - b_rate,
                "buy_vs_sell": buy_rate - sell_rate,
                "n_buy_hit": int(buy_f[col].sum()) if len(buy_f) else 0,
                "n_sell_hit": int(sell_f[col].sum()) if len(sell_f) else 0,
                "n_buy": int(len(buy_f)),
                "n_sell": int(len(sell_f)),
            }
        )
    stats = pd.DataFrame(rows)
    stats.to_csv(out_dir / "feature_rates.csv", index=False, encoding="utf-8-sig")

    buy_top = stats.sort_values("buy_lift", ascending=False).head(25)
    sell_top = stats.sort_values("sell_lift", ascending=False).head(25)
    diff_top = stats.reindex(stats["buy_vs_sell"].abs().sort_values(ascending=False).index).head(30)
    buy_common = stats[(stats["buy_rate"] >= 0.50) & (stats["buy_lift"] >= 0.10)].sort_values(
        "buy_lift", ascending=False
    )
    sell_common = stats[(stats["sell_rate"] >= 0.50) & (stats["sell_lift"] >= 0.10)].sort_values(
        "sell_lift", ascending=False
    )

    def dump_pivots(mask: np.ndarray, kind: str) -> pd.DataFrame:
        out: list[dict[str, Any]] = []
        for i, d in enumerate(frame.index):
            if not mask[i]:
                continue
            r = frame.iloc[i]
            out.append(
                {
                    "kind": kind,
                    "date": d.strftime("%Y-%m-%d"),
                    "px": float(r["px"]),
                    "fwd_up_40": float(fwd_up[i]) if np.isfinite(fwd_up[i]) else None,
                    "fwd_dn_40": float(fwd_dn[i]) if np.isfinite(fwd_dn[i]) else None,
                    "vol5_pct": None if pd.isna(r["vol5_pct_120d"]) else float(r["vol5_pct_120d"]),
                    "gap20": None if pd.isna(r["gap_20d"]) else float(r["gap_20d"]),
                    "gap5": None if pd.isna(r["gap_5d"]) else float(r["gap_5d"]),
                    "fig7_gap": None if pd.isna(r["fig7_gap"]) else float(r["fig7_gap"]),
                    "fig7_g": None if pd.isna(r["fig7_g"]) else float(r["fig7_g"]),
                    "fig8_g": None if pd.isna(r["fig8_g"]) else float(r["fig8_g"]),
                    "fig10_gap": None if pd.isna(r["fig10_gap"]) else float(r["fig10_gap"]),
                    "fig10_touch_hi": bool(r["fig10_touch_hi"]),
                    "fig10_touch_lo": bool(r["fig10_touch_lo"]),
                    "fig1_上涨买": bool(r["fig1_上涨买"]),
                    "fig1_上涨卖": bool(r["fig1_上涨卖"]),
                    "fig1_上行切换": bool(r["fig1_上行切换"]),
                    "fig1_下行切换": bool(r["fig1_下行切换"]),
                    "qd_entry": bool(r["qd_entry"]),
                }
            )
        return pd.DataFrame(out)

    piv = pd.concat([dump_pivots(is_buy, "buy"), dump_pivots(is_sell, "sell")], ignore_index=True)
    piv.to_csv(out_dir / "oracle_pivots.csv", index=False, encoding="utf-8-sig")

    prod_buy = set(marker_sets.get("上涨买", set()))
    prod_sell = set(marker_sets.get("上涨卖", set()))
    ob = set(pd.DatetimeIndex(frame.index[is_buy]).strftime("%Y-%m-%d"))
    os_ = set(pd.DatetimeIndex(frame.index[is_sell]).strftime("%Y-%m-%d"))
    summary = {
        "html": str(html_path),
        "code": code,
        "oracle_rule": {
            "local_window": f"±{args.left}/{args.right}",
            "fwd_days": args.fwd,
            "buy_thr": args.buy_thr,
            "sell_thr": args.sell_thr,
            "thin_gap": args.thin_gap,
        },
        "n_oracle_buy": int(is_buy.sum()),
        "n_oracle_sell": int(is_sell.sum()),
        "n_prod_buy": len(prod_buy),
        "n_prod_sell": len(prod_sell),
        "oracle_buy_exact_prod_buy": len(ob & prod_buy),
        "oracle_sell_exact_prod_sell": len(os_ & prod_sell),
        "oracle_buy_near_prod_buy_±5d": _near_hit(ob, prod_buy, frame.index, 5),
        "oracle_sell_near_prod_sell_±5d": _near_hit(os_, prod_sell, frame.index, 5),
        "buy_mean_fwd_up40": float(np.nanmean(fwd_up[is_buy])) if is_buy.any() else None,
        "sell_mean_fwd_dn40": float(np.nanmean(fwd_dn[is_sell])) if is_sell.any() else None,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    pb = piv[piv["kind"] == "buy"].tail(15)
    ps = piv[piv["kind"] == "sell"].tail(15)
    header = (
        f"# {code}：用未来数据标定买卖点，并总结图1–11公共标识\n\n"
        f"- 数据：`{html_path.name}`\n"
        f"- 事后买卖点规则：**±{args.left}/{args.right} 日局部极值** + 未来 {args.fwd} 日内"
        f"最大涨幅≥{args.buy_thr:.0%}（买）/ 最大跌幅≤−{args.sell_thr:.0%}（卖）；"
        f"同向簇间隔≥{args.thin_gap} 日去重。\n"
        f"- 事后买点 **{summary['n_oracle_buy']}** 个（均值未来40日峰值涨幅 "
        f"{summary['buy_mean_fwd_up40']:.1%}）；卖点 **{summary['n_oracle_sell']}** 个"
        f"（均值未来40日谷值跌幅 {summary['sell_mean_fwd_dn40']:.1%}）。\n"
        f"- 与图1生产「上涨买/卖」：精确重合买 {summary['oracle_buy_exact_prod_buy']}/"
        f"{summary['n_oracle_buy']}，卖 {summary['oracle_sell_exact_prod_sell']}/"
        f"{summary['n_oracle_sell']}；±5 交易日邻近买 "
        f"{summary['oracle_buy_near_prod_buy_±5d']}、卖 "
        f"{summary['oracle_sell_near_prod_sell_±5d']}。\n\n"
        "> 注意：这是 **偷看未来** 的标注，只用于归纳图上公共形态，不能当可交易信号。\n"
    )
    body: list[str] = [
        "## 买点：相对全样本抬升最大的标识",
        "",
        "| 标识 | 全样本 | 买点日 | 卖点日 | 买−基 | 卖−基 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in buy_top.iterrows():
        body.append(_fmt_row(r))
    body += [
        "",
        "## 卖点：相对全样本抬升最大的标识",
        "",
        "| 标识 | 全样本 | 买点日 | 卖点日 | 买−基 | 卖−基 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in sell_top.iterrows():
        body.append(_fmt_row(r))
    body += [
        "",
        "## 买 vs 卖 区分度最高",
        "",
        "| 标识 | 全样本 | 买点日 | 卖点日 | 买−卖 |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in diff_top.iterrows():
        body.append(
            f"| {r['feature']} | {r['base_rate']:.0%} | {r['buy_rate']:.0%} | "
            f"{r['sell_rate']:.0%} | {r['buy_vs_sell']:+.0%} |"
        )
    body += [
        "",
        "## 归纳：买点公共标识（事后）",
        "",
        "买点日出现率≥50% 且相对全样本抬升≥10个百分点：",
        "",
    ]
    for _, r in buy_common.head(20).iterrows():
        body.append(
            f"- **{r['feature']}**：买点日 {r['buy_rate']:.0%}"
            f"（全样本 {r['base_rate']:.0%}，卖点日 {r['sell_rate']:.0%}）"
        )
    if buy_common.empty:
        body.append("- （无：没有标识同时满足买点日≥50% 且抬升≥10pp）")
    body += ["", "## 归纳：卖点公共标识（事后）", ""]
    for _, r in sell_common.head(20).iterrows():
        body.append(
            f"- **{r['feature']}**：卖点日 {r['sell_rate']:.0%}"
            f"（全样本 {r['base_rate']:.0%}，买点日 {r['buy_rate']:.0%}）"
        )
    if sell_common.empty:
        body.append("- （无：没有标识同时满足卖点日≥50% 且抬升≥10pp）")

    panels = [
        ("图1", [c for c in flags.columns if c.startswith("图1|")]),
        ("图3", [c for c in flags.columns if c.startswith("图3|")]),
        ("图4", [c for c in flags.columns if c.startswith("图4|")]),
        ("图5", [c for c in flags.columns if c.startswith("图5|")]),
        ("图7", [c for c in flags.columns if c.startswith("图7")]),
        ("图8", [c for c in flags.columns if c.startswith("图8")]),
        ("图9", [c for c in flags.columns if c.startswith("图9")]),
        ("图10", [c for c in flags.columns if c.startswith("图10")]),
        ("图11", [c for c in flags.columns if c.startswith("图11")]),
        ("组合", [c for c in flags.columns if c.startswith("组合|")]),
    ]
    body += ["", "## 分图对照（买点日 vs 卖点日出现率）", ""]
    for title, cols in panels:
        body.append(f"### {title}")
        body.append("")
        body.append("| 标识 | 买点 | 卖点 | 差值 |")
        body.append("|---|---:|---:|---:|")
        sub = stats[stats["feature"].isin(cols)].copy()
        sub["absdiff"] = sub["buy_vs_sell"].abs()
        sub = sub.sort_values("absdiff", ascending=False)
        for _, r in sub.iterrows():
            body.append(
                f"| {r['feature']} | {r['buy_rate']:.0%} | {r['sell_rate']:.0%} | "
                f"{r['buy_vs_sell']:+.0%} |"
            )
        body.append("")

    body += [
        "## 事后买卖点列表（最近 15 买 / 15 卖）",
        "",
        "### 买",
        "| date | px | fwd峰值 | vol5% | gap20 | fig7_gap | g7 | 触下10 | 上涨买 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, t in pb.iterrows():
        body.append(
            f"| {t['date']} | {t['px']:.3f} | {t['fwd_up_40']:.1%} | "
            f"{(t['vol5_pct'] if pd.notna(t['vol5_pct']) else float('nan')):.2f} | "
            f"{(t['gap20'] if pd.notna(t['gap20']) else float('nan')):.1%} | "
            f"{(t['fig7_gap'] if pd.notna(t['fig7_gap']) else float('nan')):.1%} | "
            f"{(t['fig7_g'] if pd.notna(t['fig7_g']) else float('nan')):.2f} | "
            f"{t['fig10_touch_lo']} | {t['fig1_上涨买']} |"
        )
    body += [
        "",
        "### 卖",
        "| date | px | fwd谷值 | vol5% | gap20 | fig7_gap | g7 | 触上10 | 上涨卖 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, t in ps.iterrows():
        body.append(
            f"| {t['date']} | {t['px']:.3f} | {t['fwd_dn_40']:.1%} | "
            f"{(t['vol5_pct'] if pd.notna(t['vol5_pct']) else float('nan')):.2f} | "
            f"{(t['gap20'] if pd.notna(t['gap20']) else float('nan')):.1%} | "
            f"{(t['fig7_gap'] if pd.notna(t['fig7_gap']) else float('nan')):.1%} | "
            f"{(t['fig7_g'] if pd.notna(t['fig7_g']) else float('nan')):.2f} | "
            f"{t['fig10_touch_hi']} | {t['fig1_上涨卖']} |"
        )

    report = header + "\n" + "\n".join(body) + build_conclusion(stats, summary)
    report_path = out_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] wrote {out_dir}")
    return {
        "html": html_path,
        "code": code,
        "listing": listing,
        "out_dir": out_dir,
        "report": report_path,
        "pivots": out_dir / "oracle_pivots.csv",
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    run_study(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
