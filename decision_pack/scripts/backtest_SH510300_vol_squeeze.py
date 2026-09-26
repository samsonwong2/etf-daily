#!/usr/bin/env python3
"""SH510300: causal vol-squeeze (left-side) + direction guess + expansion exit.

Compression (no future bars):
  rng40 = 40d close high/low range
  rng40_pct_252 = trailing 252d percentile of rng40 (window includes today)
  squeeze day = rng40_pct_252 <= 0.30
  episode = consecutive squeeze days; left-side buy on first day streak >= min_days

Long-only. Signal close → next-bar close fill. Research only (not hold_up).

Direction filters (evaluated on the buy-candidate day):
  always     — guess up whenever squeeze is mature
  below_mid  — fig9_pos < 0.50 (band lower half, mean-revert up)
  drift_dn   — episode-to-date return < 0 (fade the grind)
  drift_up   — episode-to-date return > 0 (continuation)

Exit (OR): fig9 near-hi ∧ aux_sell; 40d-range percentile back above 50%
then locmax or expand_give days; optional % stop from entry; max hold.

Example::

    PYTHONPATH=. ~/etf-daily-output/python/envs/py312/bin/python \\
      decision_pack/scripts/backtest_SH510300_vol_squeeze.py \\
      --listing-dir ~/etf-daily-output/temp/plotly_outputs/20260827_from_listing
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runtime_paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from decision_pack.src.fair_path_band_signals import find_regime_html  # noqa: E402
from decision_pack.src.regime_transition_plot import (  # noqa: E402
    compute_volatility_regime_frame,
)

DEFAULT_LISTING = PLOTLY_OUTPUTS_DIR / "20260827_from_listing"
CODE = "SH510300"
RNG_WIN = 40
PCT_WIN = 252
PCT_MIN = 60
SQUEEZE_PCT = 0.30
MIN_DAYS = 30
EXPAND_PCT = 0.70
EXPAND_GIVE = 8
MAX_HOLD = 80
STOP_PCT = 0.10
DIR_NAMES = ("always", "below_mid", "drift_dn", "drift_up")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--listing-dir", type=Path, default=DEFAULT_LISTING)
    p.add_argument("--html", type=Path, default=None)
    p.add_argument("--anchor-code", default="SH510300")
    p.add_argument("--lookback", type=int, default=10)
    p.add_argument("--min-days", type=int, default=MIN_DAYS)
    p.add_argument("--squeeze-pct", type=float, default=SQUEEZE_PCT)
    p.add_argument("--stop-pct", type=float, default=STOP_PCT)
    p.add_argument("--max-hold", type=int, default=MAX_HOLD)
    p.add_argument("--expand-give", type=int, default=EXPAND_GIVE)
    return p.parse_args(argv)


def _pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.1%}"


def add_squeeze_columns(frame: pd.DataFrame, *, squeeze_pct: float) -> pd.DataFrame:
    out = frame.copy()
    px = out["px"].astype(float)
    rng40 = px.rolling(RNG_WIN).max() / px.rolling(RNG_WIN).min() - 1.0
    rng40_pct = rng40.rolling(PCT_WIN, min_periods=PCT_MIN).rank(pct=True)
    sq = (rng40_pct <= float(squeeze_pct)) & rng40_pct.notna()
    n = len(out)
    streak = np.zeros(n, dtype=int)
    run_start = np.full(n, -1, dtype=int)
    run_low = np.full(n, np.nan)
    ep_ret = np.full(n, np.nan)
    c = 0
    rs = -1
    lo = np.nan
    px_a = px.to_numpy(dtype=float)
    sq_a = sq.fillna(False).to_numpy(dtype=bool)
    for i in range(n):
        if sq_a[i]:
            if c == 0:
                rs = i
                lo = px_a[i]
            else:
                lo = min(lo, px_a[i])
            c += 1
            streak[i] = c
            run_start[i] = rs
            run_low[i] = lo
            if np.isfinite(px_a[rs]) and px_a[rs] != 0:
                ep_ret[i] = px_a[i] / px_a[rs] - 1.0
        else:
            c = 0
            rs = -1
            lo = np.nan
    out["rng40"] = rng40
    out["rng40_pct_252"] = rng40_pct
    out["squeeze"] = sq.fillna(False)
    out["squeeze_streak"] = streak
    out["squeeze_start_i"] = run_start
    out["squeeze_low"] = run_low
    out["squeeze_ep_ret"] = ep_ret
    return out


def episode_table(frame: pd.DataFrame, *, min_days: int) -> pd.DataFrame:
    px = frame["px"].astype(float)
    sq = frame["squeeze"].to_numpy(dtype=bool)
    idx = frame.index
    rows: list[dict[str, Any]] = []
    i = 0
    n = len(frame)
    while i < n:
        if not sq[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and sq[j + 1]:
            j += 1
        length = j - i + 1
        if length >= int(min_days):
            px0 = float(px.iloc[i])
            px1 = float(px.iloc[j])
            after = px.iloc[j + 1 :]
            rec: dict[str, Any] = {
                "start": idx[i].strftime("%Y-%m-%d"),
                "end": idx[j].strftime("%Y-%m-%d"),
                "n": length,
                "px_start": round(px0, 4),
                "px_end": round(px1, 4),
                "ep_ret": px1 / px0 - 1.0,
                "rng40_end": float(frame["rng40"].iloc[j]),
                "rng40_pct_end": float(frame["rng40_pct_252"].iloc[j]),
                "fig9_pos_end": float(frame["fig9_pos"].iloc[j])
                if pd.notna(frame["fig9_pos"].iloc[j])
                else np.nan,
                "rv20_end": float(frame["rv20"].iloc[j])
                if "rv20" in frame.columns and pd.notna(frame["rv20"].iloc[j])
                else np.nan,
                "left_buy": idx[i + min_days - 1].strftime("%Y-%m-%d")
                if length >= min_days
                else "",
            }
            for h in (20, 40, 60):
                if len(after) < h:
                    rec[f"fwd{h}"] = np.nan
                else:
                    rec[f"fwd{h}"] = float(after.iloc[h - 1] / px1 - 1.0)
            rows.append(rec)
        i = j + 1
    return pd.DataFrame(rows)


def dir_mask(frame: pd.DataFrame, name: str, *, min_days: int) -> pd.Series:
    mature = frame["squeeze_streak"] >= int(min_days)
    edge = mature & ~mature.shift(1, fill_value=False)
    pos = pd.to_numeric(frame["fig9_pos"], errors="coerce")
    drift = pd.to_numeric(frame["squeeze_ep_ret"], errors="coerce")
    if name == "always":
        extra = pd.Series(True, index=frame.index)
    elif name == "below_mid":
        extra = pos < 0.50
    elif name == "drift_dn":
        extra = drift < 0
    elif name == "drift_up":
        extra = drift > 0
    else:
        raise ValueError(name)
    return (edge & extra).fillna(False)


def simulate(
    frame: pd.DataFrame,
    buy_cand: pd.Series,
    *,
    stop_pct: float,
    max_hold: int,
    expand_give: int,
) -> pd.DataFrame:
    close = frame["px"].astype(float)
    idx = frame.index
    n = len(frame)
    near_hi = frame["fig9_near_hi"].fillna(False).astype(bool)
    aux_s = frame["aux_sell"].fillna(False).astype(bool)
    locmax = frame["locmax"].fillna(False).astype(bool)
    rng_pct = pd.to_numeric(frame["rng40_pct_252"], errors="coerce")
    buy_a = buy_cand.fillna(False).to_numpy(dtype=bool)
    trades: list[dict[str, Any]] = []
    in_pos = False
    entry_i = None
    entry_sig_i = None
    expand_i = None

    for i in range(n - 1):
        if not in_pos:
            if buy_a[i]:
                entry_sig_i = i
                entry_i = i + 1
                in_pos = True
                expand_i = None
            continue
        assert entry_i is not None and entry_sig_i is not None
        hold = i - entry_i
        ep = float(close.iloc[entry_i])
        px = float(close.iloc[i])
        reason = ""
        if pd.notna(rng_pct.iloc[i]) and float(rng_pct.iloc[i]) >= 0.50:
            if expand_i is None:
                expand_i = i
        if stop_pct > 0 and px <= ep * (1.0 - float(stop_pct)):
            reason = f"止损{stop_pct:.0%}"
        elif hold >= int(max_hold):
            reason = f"持仓满{max_hold}日"
        elif bool(near_hi.iloc[i]) and bool(aux_s.iloc[i]):
            reason = "图9近上∧aux卖"
        elif expand_i is not None and (
            bool(locmax.iloc[i]) or (i - expand_i) >= int(expand_give)
        ):
            reason = "波动放大后了结"
        if not reason:
            continue
        exit_i = i + 1
        if exit_i >= n:
            break
        xp = float(close.iloc[exit_i])
        trades.append(
            {
                "signal_buy": idx[entry_sig_i].strftime("%Y-%m-%d"),
                "entry": idx[entry_i].strftime("%Y-%m-%d"),
                "signal_sell": idx[i].strftime("%Y-%m-%d"),
                "exit": idx[exit_i].strftime("%Y-%m-%d"),
                "entry_px": ep,
                "exit_px": xp,
                "ret": xp / ep - 1.0,
                "hold_days": int(exit_i - entry_i),
                "exit_reason": reason,
                "open": False,
            }
        )
        in_pos = False
        entry_i = None
        entry_sig_i = None
        expand_i = None

    if in_pos and entry_i is not None and entry_sig_i is not None:
        ep = float(close.iloc[entry_i])
        xp = float(close.iloc[-1])
        trades.append(
            {
                "signal_buy": idx[entry_sig_i].strftime("%Y-%m-%d"),
                "entry": idx[entry_i].strftime("%Y-%m-%d"),
                "signal_sell": None,
                "exit": None,
                "entry_px": ep,
                "exit_px": xp,
                "ret": xp / ep - 1.0,
                "hold_days": int(n - 1 - entry_i),
                "exit_reason": "未平",
                "open": True,
            }
        )
    return pd.DataFrame(trades)


def write_report(
    path: Path,
    *,
    html: Path,
    bh: float,
    min_days: int,
    squeeze_pct: float,
    stop_pct: float,
    max_hold: int,
    expand_give: int,
    episodes: pd.DataFrame,
    variants: list[dict[str, Any]],
) -> None:
    lines = [
        "# SH510300 波动压缩 / 左侧猜方向",
        "",
        "只做多。压缩用**因果** 40 日收盘振幅的过去 252 日分位（含当日，不含未来）。",
        f"压缩日：`rng40_pct_252 ≤ {squeeze_pct:.0%}`；连续满 **{min_days}** 日当天左侧买入（次日收盘成交）。",
        "猜方向只决定买不买涨：`always` 逢压缩就猜涨；`below_mid` 图9 分位 <50%；",
        "`drift_dn` 压缩段已下跌（逆势）；`drift_up` 压缩段已上涨（顺势）。",
        f"卖：图9 近上 ∧ 辅助卖，或 40 日振幅分位 ≥50%（区间打开）后局部高 / {expand_give} 日，"
        f"或入场止损 {stop_pct:.0%}，或满 {max_hold} 日。",
        "",
        f"样本 HTML：`{html}`",
        f"买入持有：**{_pct(bh)}**。研究回测，不是生产 `hold_up`。",
        "",
        "## 压缩段（描述：结束后的 fwd 用了未来，只做对照，不进成交）",
        "",
        "| 开始 | 结束 | 天数 | 段收益 | 结束图9分位 | 左买日 | 后20日 | 后40日 | 后60日 |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    if episodes.empty:
        lines.append("| — | — | — | — | — | — | — | — | — |")
    else:
        for _, r in episodes.iterrows():
            lines.append(
                f"| {r['start']} | {r['end']} | {int(r['n'])} | {_pct(float(r['ep_ret']))} | "
                f"{_pct(float(r['fig9_pos_end']) if pd.notna(r['fig9_pos_end']) else None)} | "
                f"{r.get('left_buy') or ''} | {_pct(float(r['fwd20']) if pd.notna(r.get('fwd20')) else None)} | "
                f"{_pct(float(r['fwd40']) if pd.notna(r.get('fwd40')) else None)} | "
                f"{_pct(float(r['fwd60']) if pd.notna(r.get('fwd60')) else None)} |"
            )
    lines += ["", "## 回测变体", "", "| 方向 | 已平 | 胜率 | 复合 | 均持仓日 |", "|---|---:|---:|---:|---:|"]
    for v in variants:
        s = v["summ"]
        mh = s.get("mean_hold")
        if mh is None:
            lines.append(f"| `{v['name']}` | {s.get('n_trades') or 0} | — | — | — |")
        else:
            lines.append(
                f"| `{v['name']}` | {s.get('n_trades')} | {_pct(s.get('win_rate'))} | "
                f"{_pct(s.get('compound'))} | {mh:.0f} |"
            )
    for v in variants:
        trades: pd.DataFrame = v["trades"]
        lines += ["", f"### `{v['name']}`", ""]
        if trades is None or trades.empty:
            lines.append("无成交")
            continue
        lines.append("| 买信号 | 成交买 | 卖信号 | 成交卖 | 原因 | 段收益 | 持仓日 |")
        lines.append("|---|---|---|---|---|---:|---:|")
        for _, tr in trades.iterrows():
            lines.append(
                f"| {tr.get('signal_buy')} | {tr.get('entry')} | {tr.get('signal_sell') or '—'} | "
                f"{tr.get('exit') or '—'} | {tr.get('exit_reason') or ''} | "
                f"{_pct(float(tr['ret']))} | {int(tr['hold_days'])} |"
            )
    lines += [
        "",
        "## 说明",
        "",
        "- 滚动 20% 分位时 2024 连续压缩只有 22 日，进不了 30 日门槛；默认 30% 才能扫到 4–7 月那段。",
        "- 「方向选完」用 40 日振幅分位回到 50% 以上，不用 vol5 毛刺，避免压缩期内被吓出。",
        "- 2024-05-30 左侧买、9/18 因 40 日振幅分位回到 50% 卖掉（阴跌把区间打开），成交在 **9/19**，正好在 9/24 拉升前一天，`always` 该笔 **−8.8%**。",
        "- 段表的 fwd20/40/60 是压缩**结束之后**的收益，和左侧入场点不是同一天。",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    listing = _resolve(args.listing_dir)
    sm = _load(_SCRIPTS / "run_reusable_extrema_state_machine.py", "extrema_sm")
    bt = _load(_SCRIPTS / "backtest_fig9_touch.py", "fig9_touch_bt")
    html = _resolve(args.html) if args.html else find_regime_html(listing, CODE)
    if html is None:
        raise FileNotFoundError(CODE)
    frame, _ohlcv = sm.build_feature_frame(
        html, listing, args.anchor_code, lookback=int(args.lookback)
    )
    vf = compute_volatility_regime_frame(frame["px"])
    vf["as_of"] = pd.to_datetime(vf["as_of"]).dt.normalize()
    vf = vf.set_index("as_of")
    frame["rv20"] = pd.to_numeric(vf["rv20"], errors="coerce").reindex(frame.index)
    if "vol5_pct_120d" not in frame.columns:
        frame["vol5_pct_120d"] = pd.to_numeric(vf["vol5_pct_120d"], errors="coerce").reindex(
            frame.index
        )
    frame = bt.prepare_fig9_touch_frame(frame)
    frame = add_squeeze_columns(frame, squeeze_pct=float(args.squeeze_pct))
    min_days = int(args.min_days)
    episodes = episode_table(frame, min_days=min_days)
    close = frame["px"].astype(float)
    bh = float(close.iloc[-1] / close.iloc[0] - 1.0) if float(close.iloc[0]) else float("nan")

    out_dir = listing / f"fig3_11_bt_{CODE}" / "vol_squeeze_bt"
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(out_dir / "episodes.csv", index=False, encoding="utf-8-sig")

    variants: list[dict[str, Any]] = []
    for name in DIR_NAMES:
        buy = dir_mask(frame, name, min_days=min_days)
        trades = simulate(
            frame,
            buy,
            stop_pct=float(args.stop_pct),
            max_hold=int(args.max_hold),
            expand_give=int(args.expand_give),
        )
        trades.to_csv(out_dir / f"trades_{name}.csv", index=False, encoding="utf-8-sig")
        summ = sm.summarize_trades(trades)
        variants.append({"name": name, "trades": trades, "summ": summ})
        print(
            f"[{name}] n={summ.get('n_trades')} wr={_pct(summ.get('win_rate'))} "
            f"compound={_pct(summ.get('compound'))}"
        )

    report = out_dir / "report.md"
    write_report(
        report,
        html=html,
        bh=bh,
        min_days=min_days,
        squeeze_pct=float(args.squeeze_pct),
        stop_pct=float(args.stop_pct),
        max_hold=int(args.max_hold),
        expand_give=int(args.expand_give),
        episodes=episodes,
        variants=variants,
    )
    summary = {
        "code": CODE,
        "html": str(html),
        "bh": bh,
        "squeeze": f"rng40_pct_252<={args.squeeze_pct}",
        "min_days": min_days,
        "stop_pct": float(args.stop_pct),
        "max_hold": int(args.max_hold),
        "expand_give": int(args.expand_give),
        "fill": "signal close → next bar close",
        "look_ahead": False,
        "n_episodes": int(len(episodes)),
        "variants": [{k: v for k, v in x.items() if k != "trades"} for x in variants],
        "report": str(report),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] {report}")
    return summary


def main(argv: list[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
