#!/usr/bin/env python3
"""Fig9 near-touch-lo buy / unified high-in-band sell.

Primary (causal, from fig9 6-month fair-path band):
  buy  = close ≤ fig9_lower * (1 + near_pct)   # default 1% slack
  sell = aux_sell ∧ locmax ∧ fig9 band-pos ≥ pos_min (default 80%) AND either
           (A) close within near_pct of fig9 upper  — rail / 近上轨
           (B) fig9 g>0 ∧ ¬S1 ∧ vol簇 ∧ (fig7溢价≥5% ∨ fig10 band-pos≥90%)
               — mid-band stretch (2025-07-22, 2026-03-03)

(B) does not steal (A): rail exits have fig9 g<0 or already S1.
1% rail slack keeps 2022-11-29 (not 2022-11-25). pos≥80% is the honest
threshold: 2025-07-22 = 80.5%, 2026-03-03 = 80.9% (chart ~85%).

Long-only. Signal close → next-bar close fill.

Example::

    PYTHONPATH=. python decision_pack/scripts/backtest_fig9_touch.py --code SH512530
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

DEFAULT_LISTING = PLOTLY_OUTPUTS_DIR / "20260824_from_listing"
VARIANTS = ("raw", "edge", "aux", "aux_edge", "aux_extrema", "aux_fail_lo", "aux_leave")
DEFAULT_NEAR_PCT = 0.01
DEFAULT_POS_MIN = 0.80
DEFAULT_FIG10_POS_MIN = 0.90
DEFAULT_FIG7_PREM = 0.05
DEFAULT_VOL_CLUSTER = 0.70
DEFAULT_LEAVE_G_MAX = 0.25
DEFAULT_LEAVE_OFF_DAYS = 3
DEFAULT_LEAVE_HI_MIN_NEAR = 4  # ignore 1–3 day upper kisses; need a real ride
DEFAULT_LEAVE_BUY_POS_MAX = 0.15  # band-floor buy when not yet inside 1% near_lo
DEFAULT_LEAVE_BUY_G_MAX = 0.50  # skip steep glued lower walks (Jan 2026 g≫50%)
DEFAULT_LEAVE_FAIL_POS = 0.50  # re-touch lower only counts after a real bounce
DEFAULT_LEAVE_SOFT_FAIL_PEAK = 0.40
DEFAULT_LEAVE_SOFT_FAIL_POS = 0.20


def _load_sm():
    path = _SCRIPTS / "run_reusable_extrema_state_machine.py"
    spec = importlib.util.spec_from_file_location("extrema_sm", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--code", default="SH512530")
    p.add_argument("--html", type=Path, default=None)
    p.add_argument("--listing-dir", type=Path, default=DEFAULT_LISTING)
    p.add_argument("--anchor-code", default="SH510300")
    p.add_argument("--lookback", type=int, default=10)
    p.add_argument(
        "--near-pct",
        type=float,
        default=DEFAULT_NEAR_PCT,
        help="count as touch if close is within this fraction of the rail (0.01=1%)",
    )
    p.add_argument(
        "--pos-min",
        type=float,
        default=DEFAULT_POS_MIN,
        help="fig9 log band position floor for the stretch sell (0.80=80%)",
    )
    p.add_argument(
        "--no-mid-stretch",
        action="store_true",
        help="disable the extra mid-band stretch sell family",
    )
    p.add_argument(
        "--html-variant",
        default="aux_edge",
        choices=list(VARIANTS),
        help="which variant to draw on HTML",
    )
    p.add_argument("--skip-html", action="store_true")
    return p.parse_args(argv)


def _resolve(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def _band_pos(px: pd.Series, lower: pd.Series, upper: pd.Series) -> pd.Series:
    lo = np.log(pd.to_numeric(lower, errors="coerce"))
    hi = np.log(pd.to_numeric(upper, errors="coerce"))
    return (np.log(px) - lo) / (hi - lo)


def prepare_fig9_touch_frame(
    frame: pd.DataFrame,
    *,
    near_pct: float = DEFAULT_NEAR_PCT,
    pos_min: float = DEFAULT_POS_MIN,
    fig10_pos_min: float = DEFAULT_FIG10_POS_MIN,
    fig7_prem: float = DEFAULT_FIG7_PREM,
    vol_cluster: float = DEFAULT_VOL_CLUSTER,
    mid_stretch: bool = True,
) -> pd.DataFrame:
    """Add near-rail, aux, band-pos, and mid-stretch sell columns (causal)."""
    out = frame.copy()
    close = out["px"].astype(float)
    near = float(near_pct)
    lower = pd.to_numeric(out["fig9_lower"], errors="coerce")
    upper = pd.to_numeric(out["fig9_upper"], errors="coerce")
    out["fig9_near_lo"] = (close <= lower * (1.0 + near)).fillna(False)
    out["fig9_near_hi"] = (close >= upper * (1.0 - near)).fillna(False)
    out["aux_buy"] = (
        (close < out["ma20"])
        & (out["ma5"] < out["ma20"])
        & (out["gap_5d"] <= 0)
        & (out["gap_20d"] <= 0)
        & (~out["fig10_touch_hi"].fillna(False))
    )
    out["aux_sell"] = (
        (close > out["ma20"])
        & (out["ma5"] > out["ma20"])
        & (out["gap_5d"] >= 0)
        & (out["gap_20d"] >= 0)
    )
    out["fig9_pos"] = _band_pos(close, lower, upper)
    out["fig10_pos"] = _band_pos(
        close,
        pd.to_numeric(out["fig10_lower"], errors="coerce"),
        pd.to_numeric(out["fig10_upper"], errors="coerce"),
    )
    if "fig7_lower" in out.columns and "fig7_upper" in out.columns:
        out["fig7_pos"] = _band_pos(
            close,
            pd.to_numeric(out["fig7_lower"], errors="coerce"),
            pd.to_numeric(out["fig7_upper"], errors="coerce"),
        )
    s1 = out["fig9_touch_hi"].fillna(False) | out["fig10_touch_hi"].fillna(False)
    if not mid_stretch:
        out["fig9_mid_stretch_sell"] = pd.Series(False, index=out.index)
    else:
        out["fig9_mid_stretch_sell"] = (
            out["aux_sell"].fillna(False)
            & out["locmax"].fillna(False)
            & (pd.to_numeric(out["vol5_pct_120d"], errors="coerce") >= float(vol_cluster))
            & (pd.to_numeric(out["fig9_g"], errors="coerce") > 0)
            & (~s1)
            & (out["fig9_pos"] >= float(pos_min))
            & (
                (pd.to_numeric(out["fig7_gap"], errors="coerce") >= float(fig7_prem))
                | (out["fig10_pos"] >= float(fig10_pos_min))
            )
        ).fillna(False)
    return out


def rising_edge(mask: pd.Series) -> pd.Series:
    m = np.asarray(mask.fillna(False), dtype=bool)
    out = np.zeros(len(m), dtype=bool)
    if len(m):
        out[0] = bool(m[0])
        if len(m) > 1:
            out[1:] = m[1:] & ~m[:-1]
    return pd.Series(out, index=mask.index)


def falling_edge(mask: pd.Series) -> pd.Series:
    m = np.asarray(mask.fillna(False), dtype=bool)
    out = np.zeros(len(m), dtype=bool)
    if len(m) > 1:
        out[1:] = (~m[1:]) & m[:-1]
    return pd.Series(out, index=mask.index)


def leave_lo_confirmed(
    near_lo: pd.Series,
    g: pd.Series | None = None,
    *,
    off_days: int = DEFAULT_LEAVE_OFF_DAYS,
    g_max: float = DEFAULT_LEAVE_G_MAX,
) -> pd.Series:
    """True on the day a lower-rail walk is confirmed over.

    Causal: falling edge of ``near_lo``, then ``off_days`` consecutive sessions
    still not near the lower rail. Optional ``0 < g < g_max`` skips the first
    pop off a steep path (Jan 2023 on SH513290) and keeps the last kiss
    (~2023-03-23) after the walk has flattened.
    """
    lo = near_lo.fillna(False).astype(bool)
    left = falling_edge(lo)
    off = ~lo
    n = max(1, int(off_days))
    persist = off.copy()
    for k in range(1, n):
        persist = persist & off.shift(k).eq(True)
    persist = persist & left.shift(n - 1).eq(True)
    if g is not None:
        gv = pd.to_numeric(g, errors="coerce")
        persist = persist & gv.gt(0) & gv.lt(float(g_max))
    return persist.fillna(False).astype(bool)


def leave_hi_confirmed(
    near_hi: pd.Series,
    *,
    off_days: int = 1,
    min_near_days: int = DEFAULT_LEAVE_HI_MIN_NEAR,
) -> pd.Series:
    """True when a real upper-rail ride has left.

    Default ``off_days=1``: sell on the leave day itself after a streak of at
    least ``min_near_days`` near_hi (option 2: ride then leave, no early
    rising-edge sell). ``off_days>1`` adds consecutive off-rail confirmation.
    """
    hi = near_hi.fillna(False).astype(bool)
    run = hi.groupby((~hi).cumsum()).cumsum().where(hi, 0)
    left = falling_edge(hi) & run.shift(1).ge(int(min_near_days))
    off = ~hi
    n = max(1, int(off_days))
    persist = off.copy()
    for k in range(1, n):
        persist = persist & off.shift(k).eq(True)
    persist = persist & left.shift(n - 1).eq(True)
    return persist.fillna(False).astype(bool)


def aux_leave_buy_mask(frame: pd.DataFrame) -> pd.Series:
    """Lower pierce (near_lo∧aux rising) or mild band-floor; skip steep g walks."""
    lo = frame["fig9_near_lo"].fillna(False).astype(bool)
    aux_b = frame["aux_buy"].fillna(False).astype(bool)
    pos = pd.to_numeric(frame["fig9_pos"], errors="coerce")
    g = pd.to_numeric(frame["fig9_g"], errors="coerce")
    g_ok_steep = g.lt(float(DEFAULT_LEAVE_BUY_G_MAX)).fillna(False)
    # Soft floor only on a mild path — catches 2024-11-18 near-bottom without
    # early-buying steep walks (2024-04-08 g≈45%, 2024-09-27 g≈34%).
    floor = (
        aux_b
        & pos.le(float(DEFAULT_LEAVE_BUY_POS_MAX)).fillna(False)
        & g.lt(float(DEFAULT_LEAVE_G_MAX)).fillna(False)
        & g.gt(-0.50).fillna(False)
    )
    # Rising edge of (near_lo ∧ aux_buy), not bare near_lo — keeps 2026-03-17
    # (aux turns on while already near_lo) and drops bare-rail chops.
    pierce = rising_edge(lo & aux_b) & g_ok_steep
    return (pierce | rising_edge(floor)).astype(bool)


def variant_masks(
    frame: pd.DataFrame, name: str
) -> tuple[pd.Series, pd.Series, str]:
    lo = frame["fig9_near_lo"].fillna(False).astype(bool)
    hi = frame["fig9_near_hi"].fillna(False).astype(bool)
    extra = frame["fig9_mid_stretch_sell"].fillna(False).astype(bool)
    aux_b = frame["aux_buy"].fillna(False).astype(bool)
    aux_s = frame["aux_sell"].fillna(False).astype(bool)
    locmin = frame["locmin"].fillna(False).astype(bool)
    locmax = frame["locmax"].fillna(False).astype(bool)
    notes = {
        "raw": "图9近下买 / 近上轨∨带内分位拉伸（状态日）",
        "edge": "图9近下/近上上跳沿 ∨ 带内分位拉伸",
        "aux": "图9近轨 ∧ 均线/gap；卖再∨带内分位拉伸",
        "aux_edge": "图9近轨上跳沿 ∧ 均线/gap；卖再∨带内分位拉伸",
        "aux_extrema": "图9近轨 ∧ 均线/gap ∧ 局部极值；卖再∨带内分位拉伸",
        "aux_fail_lo": "aux 买且图9 g>0；卖再∨再触近下；失败卖后须见近上才再买",
        "aux_leave": "贴/穿下轨上跳沿买；骑上轨≥4日离开再卖∨反弹失败再触下/回带底",
    }
    if name == "raw":
        return lo, hi | extra, notes[name]
    if name == "edge":
        return rising_edge(lo), rising_edge(hi) | extra, notes[name]
    if name == "aux":
        return lo & aux_b, (hi & aux_s) | extra, notes[name]
    if name == "aux_edge":
        return rising_edge(lo) & aux_b, (rising_edge(hi) & aux_s) | extra, notes[name]
    if name == "aux_extrema":
        return lo & aux_b & locmin, (hi & aux_s & locmax) | extra, notes[name]
    if name == "aux_fail_lo":
        re_lo = rising_edge(lo)
        g_up = pd.to_numeric(frame["fig9_g"], errors="coerce").gt(0).fillna(False)
        return lo & aux_b & g_up, (hi & aux_s) | extra | re_lo, notes[name]
    if name == "aux_leave":
        buy = aux_leave_buy_mask(frame)
        # Candidate sell mask for reporting; live path uses run_aux_leave_state_machine.
        sell = leave_hi_confirmed(hi) | rising_edge(lo)
        return buy, sell, notes[name]
    raise ValueError(name)


def tag_sell_kind(trades: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    extra = frame["fig9_mid_stretch_sell"].fillna(False).astype(bool)
    near_hi = frame["fig9_near_hi"].fillna(False).astype(bool)
    leave_hi = leave_hi_confirmed(near_hi)
    re_lo = rising_edge(frame["fig9_near_lo"].fillna(False).astype(bool))
    kinds: list[str] = []
    for _, tr in out.iterrows():
        d = pd.Timestamp(tr.get("signal_sell"))
        if pd.isna(d) or d not in frame.index:
            kinds.append("")
            continue
        kind = tr.get("sell_kind")
        if isinstance(kind, str) and kind:
            kinds.append(kind)
            continue
        mid = bool(extra.loc[d])
        rail = bool(near_hi.loc[d])
        leave = bool(leave_hi.loc[d])
        fail = bool(re_lo.loc[d])
        if leave:
            kinds.append("离开上轨")
        elif rail:
            kinds.append("图9近上轨")
        elif mid:
            kinds.append("带内拉伸")
        elif fail:
            kinds.append("再触下轨")
        else:
            kinds.append("其他")
    out["sell_kind"] = kinds
    return out


def run_aux_leave_state_machine(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """贴/穿下轨买；骑上轨离开卖；反弹失败再触下或回带底卖。不加带内拉伸。"""
    buy_c = aux_leave_buy_mask(frame)
    lo = frame["fig9_near_lo"].fillna(False).astype(bool)
    hi = frame["fig9_near_hi"].fillna(False).astype(bool)
    aux_b = frame["aux_buy"].fillna(False).astype(bool)
    pos = pd.to_numeric(frame["fig9_pos"], errors="coerce")
    leave_hi = leave_hi_confirmed(hi)
    re_lo = rising_edge(lo)
    soft_zone = (
        aux_b
        & pos.le(float(DEFAULT_LEAVE_SOFT_FAIL_POS)).fillna(False)
        & (~lo)
    )
    soft_edge = rising_edge(soft_zone)
    close = frame["px"].astype(float)
    idx = frame.index
    n = len(frame)
    buy_sig = np.zeros(n, dtype=bool)
    sell_sig = np.zeros(n, dtype=bool)
    trades: list[dict[str, Any]] = []
    in_pos = False
    entry_i: int | None = None
    entry_sig_i: int | None = None
    max_pos = float("-inf")

    for i in range(n - 1):
        if not in_pos:
            if bool(buy_c.iloc[i]):
                buy_sig[i] = True
                entry_sig_i = i
                entry_i = i + 1
                in_pos = True
                max_pos = float(pos.iloc[i]) if pd.notna(pos.iloc[i]) else float("-inf")
        else:
            if pd.notna(pos.iloc[i]):
                max_pos = max(max_pos, float(pos.iloc[i]))
            kind = ""
            if bool(leave_hi.iloc[i]):
                kind = "离开上轨"
            elif bool(re_lo.iloc[i]) and max_pos >= float(DEFAULT_LEAVE_FAIL_POS):
                kind = "再触下轨"
            elif (
                bool(soft_edge.iloc[i])
                and max_pos >= float(DEFAULT_LEAVE_SOFT_FAIL_PEAK)
            ):
                kind = "失败回带底"
            if kind:
                sell_sig[i] = True
                assert entry_i is not None and entry_sig_i is not None
                exit_i = i + 1
                if exit_i >= n:
                    break
                ep = float(close.iloc[entry_i])
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
                        "sell_kind": kind,
                    }
                )
                in_pos = False
                entry_i = None
                entry_sig_i = None
                max_pos = float("-inf")

    if in_pos and entry_i is not None and entry_sig_i is not None:
        trades.append(
            {
                "signal_buy": idx[entry_sig_i].strftime("%Y-%m-%d"),
                "entry": idx[entry_i].strftime("%Y-%m-%d"),
                "signal_sell": None,
                "exit": None,
                "entry_px": float(close.iloc[entry_i]),
                "exit_px": float(close.iloc[-1]),
                "ret": float(close.iloc[-1] / close.iloc[entry_i] - 1.0),
                "hold_days": int(n - 1 - entry_i),
                "open": True,
                "sell_kind": "",
            }
        )
    tdf = pd.DataFrame(trades)
    return (
        tdf,
        pd.Series(buy_sig, index=idx),
        pd.Series(sell_sig, index=idx),
    )

def run_aux_fail_lo_state_machine(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """aux 买且图9 g>0；A·B 卖，加上再触近下卖；该类卖出后须见到近上才再买。

    g>0 挡住下跌通道里的近下（2026-03 轨跟着价格掉，不会再出现再触下轨）。
    """
    buy_base, _, _ = variant_masks(frame, "aux_fail_lo")
    _, sell_ab, _ = variant_masks(frame, "aux")
    re_lo = rising_edge(frame["fig9_near_lo"].fillna(False).astype(bool))
    near_hi = frame["fig9_near_hi"].fillna(False).astype(bool)
    close = frame["px"].astype(float)
    idx = frame.index
    n = len(frame)
    buy_sig = np.zeros(n, dtype=bool)
    sell_sig = np.zeros(n, dtype=bool)
    trades: list[dict[str, Any]] = []
    in_pos = False
    armed = True
    entry_i: int | None = None
    entry_sig_i: int | None = None

    for i in range(n - 1):
        if (not in_pos) and (not armed) and bool(near_hi.iloc[i]):
            armed = True
        if not in_pos:
            if armed and bool(buy_base.iloc[i]):
                buy_sig[i] = True
                entry_sig_i = i
                entry_i = i + 1
                in_pos = True
        else:
            fail = bool(re_lo.iloc[i])
            ab = bool(sell_ab.iloc[i])
            if fail or ab:
                sell_sig[i] = True
                assert entry_i is not None and entry_sig_i is not None
                exit_i = i + 1
                if exit_i >= n:
                    break
                ep = float(close.iloc[entry_i])
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
                    }
                )
                in_pos = False
                if fail and not ab:
                    armed = False
                entry_i = None
                entry_sig_i = None

    if in_pos and entry_i is not None and entry_sig_i is not None:
        trades.append(
            {
                "signal_buy": idx[entry_sig_i].strftime("%Y-%m-%d"),
                "entry": idx[entry_i].strftime("%Y-%m-%d"),
                "signal_sell": None,
                "exit": None,
                "entry_px": float(close.iloc[entry_i]),
                "exit_px": float(close.iloc[-1]),
                "ret": float(close.iloc[-1] / close.iloc[entry_i] - 1.0),
                "hold_days": int(n - 1 - entry_i),
                "open": True,
            }
        )
    tdf = pd.DataFrame(trades)
    return (
        tdf,
        pd.Series(buy_sig, index=idx),
        pd.Series(sell_sig, index=idx),
    )


def fail_lo_eod_flags(
    frame: pd.DataFrame, trades: pd.DataFrame
) -> tuple[bool, bool, bool]:
    """Last-bar (in_pos, buy_candidate, sell_candidate) with the fail-lo latch."""
    tagged = tag_sell_kind(trades, frame) if trades is not None and not trades.empty else trades
    in_pos = False
    if tagged is not None and not tagged.empty:
        last = tagged.iloc[-1]
        in_pos = bool(last.get("open")) or (
            pd.isna(last.get("signal_sell")) if "signal_sell" in tagged.columns else False
        )
    buy_base, sell_ab, _ = variant_masks(frame, "aux_fail_lo")
    _, sell_ab_only, _ = variant_masks(frame, "aux")
    re_lo = rising_edge(frame["fig9_near_lo"].fillna(False).astype(bool))
    armed = True
    if tagged is not None and not tagged.empty and "sell_kind" in tagged.columns:
        hi = frame["fig9_near_hi"].fillna(False).astype(bool)
        for _, tr in tagged.iterrows():
            if str(tr.get("sell_kind") or "") != "再触下轨":
                continue
            sd = pd.Timestamp(tr.get("signal_sell"))
            if pd.isna(sd):
                continue
            armed = False
            later = hi.loc[hi.index > sd]
            if bool(later.any()):
                armed = True
    buy_today = (not in_pos) and armed and bool(buy_base.iloc[-1])
    sell_today = in_pos and (bool(sell_ab_only.iloc[-1]) or bool(re_lo.iloc[-1]))
    return in_pos, buy_today, sell_today


def _pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.1%}"


def write_report(
    *,
    path: Path,
    code: str,
    html_src: Path,
    html_out: Path | None,
    bh: float,
    rows: list[dict[str, Any]],
    html_variant: str,
    near_pct: float,
    pos_min: float,
    mid_stretch: bool,
) -> None:
    mid_line = (
        "- **统一卖点**：辅助卖像 ∧ 近10日局部高 ∧ **图9相对路径分位≥"
        f"{pos_min:.0%}**，并且 (A) 距上轨≤{near_pct:.0%}，或 (B) 图9斜率仍正 ∧ 短尺未触上 ∧ vol簇 "
        "∧（图7溢价≥5% ∨ 图10分位≥90%）。"
        "目测 2026-03-03 约 85%，实测 log 带内位置 80.9%（2025-07-22 为 80.5%），所以门槛用 80%。"
        "1% 近上轨保住 2022-11-29 等贴轨卖；分位拉伸覆盖 2025-07-22 / 2026-03-03，不提前卖掉前面的贴轨卖点。"
        if mid_stretch
        else "- **第二类卖**：关闭。"
    )
    lines = [
        f"# {code}：图9 触下买 / 触上卖 回测",
        "",
        "> 因果规则，不是偷看未来。只做多；信号日收盘确认，**次日收盘成交**。",
        "",
        f"- 数据：`{html_src}`",
        "- **主规则**：图9 收盘触及或几乎触及下轨=买、上轨=卖。"
        f"几乎触及 = 距轨 ≤{near_pct:.1%}（2024-09-11 距下轨仅 0.17%，硬触轨漏掉，这里算买）。",
        mid_line,
        "- **辅助**：买 close&lt;MA20 ∧ MA5&lt;MA20 ∧ gap5≤0 ∧ gap20≤0 ∧ ¬图10触上；"
        "卖 close&gt;MA20 ∧ MA5&gt;MA20 ∧ gap5≥0 ∧ gap20≥0。",
        "- 空仓才买、持仓才卖。图2 量、图6 净值不进规则。",
        "",
        f"买入持有（首收→末收）：**{_pct(bh)}**。",
        "",
        "| 变体 | 说明 | 买/卖候选 | 成交笔 | 胜率 | 复合 | 相对BH | 均持仓 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    shown: pd.DataFrame | None = None
    path_keys: set[tuple[str, ...]] = set()
    for r in rows:
        t = r["trades"]
        star = " ←图上" if r["name"] == html_variant else ""
        wr = _pct(t.get("win_rate"))
        cp = _pct(t.get("compound"))
        vs = _pct(None if t.get("compound") is None else t["compound"] - bh)
        mh = "—" if t.get("mean_hold") is None else f"{t['mean_hold']:.0f}"
        lines.append(
            f"| {r['name']}{star} | {r['note']} | {r['n_buy_cand']}/{r['n_sell_cand']} | "
            f"{t.get('n_trades', 0)} | {wr} | {cp} | {vs} | {mh} |"
        )
        df = r.get("trades_df")
        if df is not None and not df.empty:
            path_keys.add(
                tuple(
                    f"{str(a)[:10]}|{str(b)[:10]}"
                    for a, b in zip(df["signal_buy"], df["signal_sell"])
                )
            )
        if r["name"] == html_variant:
            shown = df
    n_tr = 0 if shown is None or shown.empty else int(len(shown))
    if len(path_keys) <= 1:
        path_note = (
            f"各变体成交路径相同：状态机做成 {n_tr} 笔；辅助只减少候选日，不改变成交日。"
        )
    else:
        path_note = (
            "无辅助的 raw/edge 与带辅助的路径不完全相同："
            "1% 近上轨会提前卖；均线/gap 辅助会挡掉更早、更差的买点。"
        )
    lines += [
        "",
        path_note,
        "",
        f"### {n_tr} 笔成交",
        "",
        "| 信号买 | 成交买 | 信号卖 | 成交卖 | 卖因 | 段收益 | 持仓日 |",
        "|---|---|---|---|---|---:|---:|",
    ]
    if shown is None or shown.empty:
        lines.append("| — | — | — | — | — | — | — |")
        rets: list[float] = []
        compound = None
    else:
        rets = [float(tr["ret"]) for _, tr in shown.iterrows()]
        for _, tr in shown.iterrows():
            kind = tr.get("sell_kind", "")
            lines.append(
                f"| {tr.get('signal_buy')} | {tr.get('entry')} | {tr.get('signal_sell')} | "
                f"{tr.get('exit')} | {kind} | {_pct(float(tr['ret']))} | {int(tr['hold_days'])} |"
            )
        compound = None
        for r in rows:
            if r["name"] == html_variant:
                compound = r["trades"].get("compound")
                break
    buy_dates = []
    sell_dates = []
    sell_kinds = []
    if shown is not None and not shown.empty:
        buy_dates = [str(x)[:10] for x in shown["signal_buy"].tolist()]
        sell_dates = [str(x)[:10] for x in shown["signal_sell"].tolist()]
        if "sell_kind" in shown.columns:
            sell_kinds = [str(x) for x in shown["sell_kind"].tolist()]
    notes: list[str] = []
    if any(d.startswith("2024-09") for d in buy_dates):
        notes.append(
            "2024-09 近下轨已计入买入（硬触轨 `close≤下轨` 漏掉 2024-09-11，距轨仅 0.17%）。"
        )
    else:
        notes.append("2024-09 近下轨未形成成交（当时可能已持仓，或辅助过滤挡掉）。")
    if any(d.startswith("2025-07") for d in sell_dates):
        notes.append(
            "2025-07-22 带内分位拉伸已计入卖出（图9 分位 80.5%，图7 溢价≥5%）。"
        )
    if any(d.startswith("2026-03") for d in sell_dates):
        notes.append(
            "2026-03-03 带内分位拉伸已计入卖出（图9 分位 80.9%，图10 分位 94%，距上轨 2.6%）。"
        )
    if rets:
        wr_all = all(x > 0 for x in rets)
        ret_txt = " / ".join(f"{_pct(x)}" for x in rets)
        win_txt = "都盈利" if wr_all else "不全盈利"
        cp_txt = _pct(compound)
        vs_bh = ""
        if compound is not None and np.isfinite(bh):
            vs_bh = "高于" if compound >= bh else "低于"
            vs_bh = f"，{vs_bh}买入持有 **{_pct(bh)}**。"
        else:
            vs_bh = f"，买入持有 **{_pct(bh)}**。"
        lines += [
            "",
            f"{n_tr} 笔{win_txt}（{ret_txt}），复合 **{cp_txt}**{vs_bh}"
            + "".join(notes),
            "",
        ]
    else:
        lines += ["", "".join(notes), ""]
    if html_out is not None:
        lines += [
            "## 叠图",
            "",
            f"`{html_out}`  蓝▲买 / 橙▼卖 = `{html_variant}` 状态机信号"
            "（近上轨卖 ∪ 带内拉伸卖 ∪ 再触下轨卖）。",
            "",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    sm = _load_sm()
    listing = _resolve(args.listing_dir)
    if args.html is not None:
        html = _resolve(args.html)
    else:
        html = find_regime_html(listing, str(args.code).upper())
        if html is None:
            raise FileNotFoundError(args.code)
    code = str(args.code).upper()
    out_dir = listing / f"fig3_11_bt_{code}" / "fig9_touch_bt"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame, ohlcv = sm.build_feature_frame(
        html, listing, args.anchor_code, lookback=args.lookback
    )
    near = float(args.near_pct)
    pos_min = float(args.pos_min)
    frame = prepare_fig9_touch_frame(
        frame,
        near_pct=near,
        pos_min=pos_min,
        mid_stretch=not bool(args.no_mid_stretch),
    )
    close = frame["px"].astype(float)
    win = frame.loc["2024-09-01":"2024-10-05", [
        "px", "fig9_lower", "fig9_upper",
        "fig9_touch_lo", "fig9_touch_hi", "fig9_near_lo", "fig9_near_hi",
    ]]
    if not win.empty:
        dist_lo = (win["px"] / win["fig9_lower"] - 1.0) * 100.0
        print(f"[diag] 2024-09 near_pct={near:.1%}  near_lo days: "
              f"{win.index[win['fig9_near_lo']].strftime('%Y-%m-%d').tolist()}")
        for dt, row in win.iterrows():
            if bool(row["fig9_near_lo"]) or bool(row["fig9_touch_lo"]) or (
                pd.notna(row["fig9_lower"]) and abs(float(dist_lo.loc[dt])) < 2.0
            ):
                print(
                    f"  {dt.strftime('%Y-%m-%d')} px={float(row['px']):.4f} "
                    f"lo={float(row['fig9_lower']):.4f} "
                    f"dist_lo={float(dist_lo.loc[dt]):+.2f}% "
                    f"exact_lo={bool(row['fig9_touch_lo'])} "
                    f"near_lo={bool(row['fig9_near_lo'])} "
                    f"exact_hi={bool(row['fig9_touch_hi'])} "
                    f"near_hi={bool(row['fig9_near_hi'])}"
                )
    extra_days = frame.index[frame["fig9_mid_stretch_sell"]].strftime("%Y-%m-%d").tolist()
    print(f"[diag] mid-stretch sell days ({len(extra_days)}): {extra_days}")
    for win_label, start, end in (
        ("2025-07-22", "2025-07-18", "2025-07-25"),
        ("2026-03-03", "2026-02-27", "2026-03-12"),
    ):
        jul = frame.loc[start:end]
        if jul.empty:
            continue
        print(f"[diag] {win_label} window:")
        for dt, row in jul.iterrows():
            print(
                f"  {dt.strftime('%Y-%m-%d')} px={float(row['px']):.4f} "
                f"f9pos={float(row['fig9_pos']):.3f} f10pos={float(row['fig10_pos']):.3f} "
                f"locmax={bool(row['locmax'])} mid={bool(row['fig9_mid_stretch_sell'])} "
                f"near_hi={bool(row['fig9_near_hi'])}"
            )

    bh = float(close.iloc[-1] / close.iloc[0] - 1.0) if float(close.iloc[0]) else float("nan")
    overlay_mod = None if args.skip_html else sm._load_overlay()
    html_out: Path | None = None
    results: list[dict[str, Any]] = []

    for name in VARIANTS:
        buy_c, sell_c, note = variant_masks(frame, name)
        if name == "aux_fail_lo":
            trades, buy_sig, sell_sig = run_aux_fail_lo_state_machine(frame)
        elif name == "aux_leave":
            trades, buy_sig, sell_sig = run_aux_leave_state_machine(frame)
        else:
            trades, buy_sig, sell_sig = sm.run_state_machine(frame, buy_c, sell_c)
        trades = tag_sell_kind(trades, frame)
        trades.to_csv(out_dir / f"trades_{name}.csv", index=False, encoding="utf-8-sig")
        summ = sm.summarize_trades(trades)
        rec = {
            "name": name,
            "note": note,
            "n_buy_cand": int(buy_c.sum()),
            "n_sell_cand": int(sell_c.sum()),
            "n_sig_buy": int(buy_sig.sum()),
            "n_sig_sell": int(sell_sig.sum()),
            "trades": summ,
            "trades_df": trades,
        }
        results.append(rec)
        cp = _pct(summ.get("compound"))
        print(
            f"[{code}/{name}] cand={rec['n_buy_cand']}/{rec['n_sell_cand']} "
            f"trades={summ.get('n_trades')} compound={cp} wr={_pct(summ.get('win_rate'))}"
        )
        if overlay_mod is not None and name == args.html_variant:
            html_out = html.with_name(html.stem + f"_fig9_touch_{name}.html")
            sm.overlay_signals_html(
                overlay_mod=overlay_mod,
                html_path=html,
                ohlcv=ohlcv,
                buy_sig=buy_sig,
                sell_sig=sell_sig,
                out_path=html_out,
                title_note=(
                    (
                        f"｜图9贴下买 / 骑上离开卖∨失败回撤 [{name} 上轨≥{DEFAULT_LEAVE_HI_MIN_NEAR}日离开当日卖]"
                        if name == "aux_leave"
                        else (
                            f"｜图9近下买 / 近上∨分位拉伸∨再触下轨卖 [{name} 近轨≤{near:.0%} 分位≥{pos_min:.0%}]"
                            if name == "aux_fail_lo"
                            else f"｜图9近下买 / 近上∨分位拉伸卖 [{name} 近轨≤{near:.0%} 分位≥{pos_min:.0%}]"
                        )
                    )
                    + f"｜买{int(buy_sig.sum())} 卖{int(sell_sig.sum())}｜蓝▲买 橙▼卖｜因果"
                ),
            )
            print(f"[OK] overlay {html_out.name}")

    doc = _PROJECT_ROOT / "documents" / f"{code}_图9触轨回测.md"
    write_report(
        path=doc,
        code=code,
        html_src=html,
        html_out=html_out,
        bh=bh,
        rows=results,
        html_variant=args.html_variant,
        near_pct=near,
        pos_min=pos_min,
        mid_stretch=not bool(args.no_mid_stretch),
    )
    summary = {
        "code": code,
        "html_src": str(html),
        "html_out": None if html_out is None else str(html_out),
        "bh": bh,
        "near_pct": near,
        "pos_min": pos_min,
        "fig10_pos_min": DEFAULT_FIG10_POS_MIN,
        "mid_stretch": not bool(args.no_mid_stretch),
        "mid_stretch_days": extra_days,
        "fill": "signal close → next bar close",
        "look_ahead": False,
        "aux_buy": "close<MA20 & MA5<MA20 & gap5≤0 & gap20≤0 & not fig10_touch_hi",
        "aux_sell": "close>MA20 & MA5>MA20 & gap5≥0 & gap20≥0",
        "unified_sell": (
            "aux_sell & locmax & fig9_pos>=pos_min & "
            "(near_hi_1% OR (fig9_g>0 & not S1 & vol>=0.70 & (fig7_gap>=5% OR fig10_pos>=90%)))"
        ),
        "variants": [{k: v for k, v in r.items() if k != "trades_df"} for r in results],
        "report": str(doc),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] report {doc}")
    return summary


def main(argv: list[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
