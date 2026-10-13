"""7-ETF fig9 buy/sell rules generalized to the whole pool (research only).

Rules come from ``docs/7只ETF_买卖规律对比.md``:

* buy   = ``aux_edge`` buy (optionally ∨ B1), optionally blocked on fast pattern
* sell  = fig9 ``aux_edge`` sell or fig8 near-upper ∧ aux sell (∨ mid-band stretch)
* G1    = once a trade sees the G1 trend-start switch, rail sells are ignored
* trail = close ≤ (1 − trail) × highest close since the fill bar

Long-only 0/1. Signal on close T, fill on close T+1, ``cost`` per side.
The 7 documented codes use their own spec; every other code is mapped to one of
four types from its own history (volatility, buy-and-hold CAGR / max drawdown,
fig7-path crossings per year).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from etf_daily.scripts.backtest_fig9_touch import (
    DEFAULT_FIG7_PREM,
    DEFAULT_FIG10_POS_MIN,
    DEFAULT_NEAR_PCT,
    DEFAULT_POS_MIN,
    DEFAULT_VOL_CLUSTER,
    variant_masks,
)

BARS_PER_YEAR = 250
MIN_HISTORY_BARS = BARS_PER_YEAR
B1_GAP = -0.08
FAST_RUN = 0.20
FAST_RUN_BARS = 40
FAST_LOOKBACK = 60


@dataclass(frozen=True)
class RuleSpec:
    b1: bool
    fast_block: bool
    sell: str  # "fig9" | "fig8"
    g1_hold: bool
    trail: float | None

    def buy_text(self) -> str:
        s = "aux_edge 或 B1" if self.b1 else "aux_edge"
        return s + ("，快形态不买" if self.fast_block else "")

    def sell_text(self) -> str:
        return "图9" if self.sell == "fig9" else "图8"

    def trail_text(self) -> str:
        return "不设" if not self.trail else f"{self.trail:.0%}"


TYPE_SMOOTH = "平稳震荡"
TYPE_TREND = "慢趋势"
TYPE_HIVOL = "高波动主题"
TYPE_CYCLE = "长期下跌或周期"
TYPE_SHORT = "历史不足"

TYPE_PARAMS: dict[str, RuleSpec] = {
    TYPE_SMOOTH: RuleSpec(b1=True, fast_block=False, sell="fig8", g1_hold=False, trail=0.08),
    TYPE_TREND: RuleSpec(b1=True, fast_block=False, sell="fig9", g1_hold=True, trail=0.15),
    TYPE_HIVOL: RuleSpec(b1=False, fast_block=True, sell="fig8", g1_hold=True, trail=0.30),
    TYPE_CYCLE: RuleSpec(b1=True, fast_block=False, sell="fig9", g1_hold=True, trail=0.12),
    TYPE_SHORT: RuleSpec(b1=True, fast_block=True, sell="fig9", g1_hold=True, trail=0.15),
}

OVERRIDES: dict[str, tuple[str, RuleSpec]] = {
    "SH512530": (TYPE_SMOOTH, RuleSpec(True, False, "fig8", False, 0.08)),
    "SH518880": (TYPE_TREND, RuleSpec(False, True, "fig9", True, 0.15)),
    "SH513100": (TYPE_TREND, RuleSpec(True, False, "fig9", True, 0.08)),
    "SH515880": (TYPE_HIVOL, RuleSpec(False, True, "fig8", True, 0.30)),
    "SH512690": (TYPE_CYCLE, RuleSpec(True, True, "fig9", True, 0.12)),
    "SZ159981": (TYPE_CYCLE, RuleSpec(True, False, "fig9", True, 0.08)),
    "SZ159892": (TYPE_CYCLE, RuleSpec(False, True, "fig8", False, None)),
}
NOT_APPLICABLE = {"SZ159892"}


def _bool(s: Any, index: pd.Index) -> pd.Series:
    return pd.Series(s, index=index).fillna(False).astype(bool)


def build_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Causal signal masks from a ``prepare_fig9_touch_frame`` output."""
    idx = frame.index
    px = frame["px"].astype(float)
    ae_buy, ae_sell, _ = variant_masks(frame, "aux_edge")
    aux_sell = _bool(frame["aux_sell"], idx)
    stretch = _bool(frame["fig9_mid_stretch_sell"], idx)
    pos9 = frame["fig9_pos"]
    run = px / px.shift(FAST_RUN_BARS) - 1
    return {
        "ae_buy": _bool(ae_buy, idx),
        "ae_sell": _bool(ae_sell, idx),
        "b1": _bool(((frame["fig7_gap"] <= B1_GAP) | (frame["fig8_gap"] <= B1_GAP))
                    & (px <= frame["fig9_lower"] * 1.01), idx),
        "fast": _bool(run.rolling(FAST_LOOKBACK, min_periods=1).max() >= FAST_RUN, idx),
        "fig8_sell": _bool(((px >= frame["fig8_upper"] * 0.99) & aux_sell) | stretch, idx),
        "g1": _bool((frame["fig7_g"] > 0) & (frame["fig8_g"] > 0)
                    & (pos9 - pos9.shift(10) >= 0.30) & (pos9 >= 0.80)
                    & ~_bool(frame["fig9_touch_hi"], idx) & (frame["fig10_g"] > frame["fig9_g"]), idx),
    }


def buy_sell_masks(masks: dict[str, pd.Series], spec: RuleSpec) -> tuple[pd.Series, pd.Series]:
    buy = masks["ae_buy"] | masks["b1"] if spec.b1 else masks["ae_buy"]
    if spec.fast_block:
        buy = buy & ~masks["fast"]
    sell = masks["ae_sell"] if spec.sell == "fig9" else masks["fig8_sell"]
    return buy, sell


def max_drawdown(curve: pd.Series | np.ndarray) -> float:
    a = np.asarray(curve, dtype=float)
    return float((a / np.maximum.accumulate(a) - 1).min()) if len(a) else 0.0


def cagr(total: float, bars: int) -> float:
    years = bars / BARS_PER_YEAR
    return (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else float("nan")


def calmar(total: float, mdd: float, bars: int) -> float:
    return cagr(total, bars) / -mdd if mdd < 0 else float("nan")


def classify_features(frame: pd.DataFrame) -> dict[str, float]:
    px = frame["px"].astype(float)
    n = len(px)
    total = float(px.iloc[-1] / px.iloc[0] - 1)
    above = frame["fig7_gap"] > 0
    crossings = float((above != above.shift()).iloc[1:].sum())
    return {
        "bars": n,
        "vol": float(px.pct_change().std() * np.sqrt(BARS_PER_YEAR)),
        "bh_total": total,
        "bh_cagr": cagr(total, n),
        "bh_mdd": max_drawdown(px),
        "cross_per_year": crossings / (n / BARS_PER_YEAR),
    }


def classify(feat: dict[str, float]) -> str:
    if feat["bars"] < MIN_HISTORY_BARS:
        return TYPE_SHORT
    if feat["vol"] >= 0.30 and feat["bh_cagr"] >= 0.10:
        return TYPE_HIVOL
    if feat["bh_mdd"] <= -0.40 or feat["bh_cagr"] < 0.05:
        return TYPE_CYCLE
    if feat["vol"] < 0.18 and feat["cross_per_year"] >= 9:
        return TYPE_SMOOTH
    return TYPE_TREND


def resolve_spec(code: str, feat: dict[str, float]) -> tuple[str, str, RuleSpec]:
    """Return (type, source, spec); source ∈ 专属 / 自动 / 历史不足."""
    code = code.upper()
    if code in OVERRIDES:
        typ, spec = OVERRIDES[code]
        return typ, "专属", spec
    typ = classify(feat)
    return typ, ("历史不足" if typ == TYPE_SHORT else "自动"), TYPE_PARAMS[typ]


def run_state_machine(
    close: pd.Series,
    buy: pd.Series,
    sell: pd.Series,
    *,
    g1: pd.Series | None = None,
    trail: float | None = None,
    cost: float = 0.0005,
) -> dict[str, Any]:
    """Walk every bar, including the last one (its decision fills tomorrow)."""
    idx = close.index
    c = close.astype(float).to_numpy()
    n = len(c)
    b = buy.reindex(idx).fillna(False).to_numpy(bool)
    s = sell.reindex(idx).fillna(False).to_numpy(bool)
    th = g1.reindex(idx).fillna(False).to_numpy(bool) if g1 is not None else np.zeros(n, bool)
    tgt = np.zeros(n + 1)
    pos = False
    trades: list[dict[str, Any]] = []
    bi = fill = -1
    pk: float | None = None
    in_trend = False
    action, why = "观望", ""
    masked = False
    for i in range(n):
        act, reason = ("观望", "")
        masked = False
        if not pos:
            if b[i]:
                pos, bi, fill, pk, in_trend = True, i, i + 1, None, False
                act = "买"
            tgt[i + 1] = 1.0 if pos else 0.0
        else:
            if i >= fill:
                pk = c[i] if pk is None else max(pk, c[i])
            if th[i]:
                in_trend = True
            reason = ""
            if i >= fill:
                masked = bool(s[i] and in_trend)
                if s[i] and not in_trend:
                    reason = "卖点"
                elif trail and i > fill and c[i] <= pk * (1 - trail):
                    reason = "回落"
            if reason:
                act = "卖"
                if i + 1 < n:
                    seg = c[fill:i + 2]
                    trades.append(dict(buy=idx[bi], fill=idx[fill], sell=idx[i], why=reason,
                                       ret=c[i + 1] / c[fill] - 1, mdd=max_drawdown(seg)))
                pos = False
                tgt[i + 1] = 0.0
            else:
                act = "持有"
                tgt[i + 1] = 1.0
        action, why = act, reason
    held = tgt[:n]
    w = np.r_[0.0, held[:-1]]
    r = np.r_[0.0, c[1:] / c[:-1] - 1]
    turn = np.abs(np.diff(np.r_[0.0, held]))
    eq = pd.Series(np.cumprod(1 + w * r - turn * cost), index=idx)

    holding_now = bool(held[-1]) if n else False
    state: dict[str, Any] = {
        "holding": holding_now,
        "action": action,
        "why": why,
        "buy_date": None, "fill_date": None, "fill_px": None,
        "peak": None, "stop": None, "g1_seen": False,
        "trail_hit": why == "回落",
        "sell_masked_by_g1": masked,
    }
    open_trade = action in ("持有", "卖") or (action == "买")
    if open_trade and bi >= 0:
        state["buy_date"] = idx[bi]
        if fill < n:
            state["fill_date"] = idx[fill]
            state["fill_px"] = float(c[fill])
        state["g1_seen"] = bool(in_trend)
        if pk is not None:
            state["peak"] = float(pk)
            if trail:
                state["stop"] = float(pk * (1 - trail))
    total = float(eq.iloc[-1] - 1) if n else 0.0
    mdd = max_drawdown(eq)
    n_trades = len(trades) + (1 if holding_now else 0)
    return {
        "equity": eq,
        "trades": trades,
        "state": state,
        "total": total,
        "mdd": mdd,
        "calmar": calmar(total, mdd, n),
        "n_trades": n_trades,
        "exposure": float(w.mean()) if n else 0.0,
    }


def _num(row: pd.Series, col: str) -> float:
    v = pd.to_numeric(row.get(col), errors="coerce")
    return float(v) if pd.notna(v) else float("nan")


def _flag(row: pd.Series, col: str) -> bool:
    v = row.get(col)
    return bool(v) if pd.notna(v) else False


def indicator_snapshot(frame: pd.DataFrame, masks: dict[str, pd.Series], spec: RuleSpec,
                       state: dict[str, Any]) -> dict[str, Any]:
    """Last-bar readings and pass/fail for every rule component (no history stats)."""
    row, prev = frame.iloc[-1], frame.iloc[-2] if len(frame) > 1 else frame.iloc[-1]
    px = _num(row, "px")
    near = DEFAULT_NEAR_PCT
    lo9, hi9, hi8 = _num(row, "fig9_lower"), _num(row, "fig9_upper"), _num(row, "fig8_upper")
    ma5, ma20 = _num(row, "ma5"), _num(row, "ma20")
    gap5, gap20 = _num(row, "gap_5d"), _num(row, "gap_20d")
    gap7, gap8 = _num(row, "fig7_gap"), _num(row, "fig8_gap")
    pos9, pos10 = _num(row, "fig9_pos"), _num(row, "fig10_pos")
    g7, g8, g9, g10 = (_num(row, f"fig{k}_g") for k in (7, 8, 9, 10))
    vol5 = _num(row, "vol5_pct_120d")
    tail = frame["px"].astype(float).iloc[-(FAST_LOOKBACK + FAST_RUN_BARS):]
    run40 = (tail / tail.shift(FAST_RUN_BARS) - 1).iloc[-FAST_LOOKBACK:].max()
    pos9_10 = pos9 - _num(frame.iloc[-11], "fig9_pos") if len(frame) > 10 else float("nan")

    d: dict[str, Any] = {"px": px}
    d["lo9"], d["dist_lo9"] = lo9, px / lo9 - 1
    d["near_lo"] = bool(px <= lo9 * (1 + near))
    d["near_lo_prev"] = bool(_num(prev, "px") <= _num(prev, "fig9_lower") * (1 + near))
    d["near_lo_edge"] = d["near_lo"] and not d["near_lo_prev"]
    d["ma5"], d["ma20"] = ma5, ma20
    d["below_ma20"], d["ma5_below"] = bool(px < ma20), bool(ma5 < ma20)
    d["gap5"], d["gap20"] = gap5, gap20
    d["gap5_le0"], d["gap20_le0"] = bool(gap5 <= 0), bool(gap20 <= 0)
    d["fig10_not_hi"] = not _flag(row, "fig10_touch_hi")
    d["aux_buy"] = _flag(row, "aux_buy")
    d["ae_buy"] = bool(masks["ae_buy"].iloc[-1])
    d["gap7"], d["gap8"] = gap7, gap8
    d["disc_ok"] = bool(gap7 <= B1_GAP or gap8 <= B1_GAP)
    d["b1"] = bool(masks["b1"].iloc[-1])
    d["run40"], d["fast"] = float(run40), bool(masks["fast"].iloc[-1])

    d["hi9"], d["dist_hi9"] = hi9, px / hi9 - 1
    d["near_hi"] = bool(px >= hi9 * (1 - near))
    d["near_hi_edge"] = d["near_hi"] and not bool(_num(prev, "px") >= _num(prev, "fig9_upper") * (1 - near))
    d["above_ma20"], d["ma5_above"] = bool(px > ma20), bool(ma5 > ma20)
    d["gap5_ge0"], d["gap20_ge0"] = bool(gap5 >= 0), bool(gap20 >= 0)
    d["aux_sell"] = _flag(row, "aux_sell")
    d["sell_a"] = d["near_hi_edge"] and d["aux_sell"]
    d["locmax"] = _flag(row, "locmax")
    d["pos9"], d["pos9_ok"] = pos9, bool(pos9 >= DEFAULT_POS_MIN)
    d["g9"], d["g9_pos"] = g9, bool(g9 > 0)
    d["no_touch_hi"] = not (_flag(row, "fig9_touch_hi") or _flag(row, "fig10_touch_hi"))
    d["vol5"], d["vol5_ok"] = vol5, bool(vol5 >= DEFAULT_VOL_CLUSTER)
    d["prem7_ok"] = bool(gap7 >= DEFAULT_FIG7_PREM)
    d["pos10"], d["pos10_ok"] = pos10, bool(pos10 >= DEFAULT_FIG10_POS_MIN)
    d["prem_or_pos10"] = d["prem7_ok"] or d["pos10_ok"]
    d["stretch"] = _flag(row, "fig9_mid_stretch_sell")
    d["fig9_sell"] = bool(masks["ae_sell"].iloc[-1])
    d["hi8"], d["dist_hi8"] = hi8, px / hi8 - 1
    d["near_hi8"] = bool(px >= hi8 * (1 - near))
    d["fig8_sell"] = bool(masks["fig8_sell"].iloc[-1])

    d["g7"], d["g8"], d["g10"] = g7, g8, g10
    d["g7_pos"], d["g8_pos"] = bool(g7 > 0), bool(g8 > 0)
    d["pos9_10"], d["pos9_10_ok"] = pos9_10, bool(pos9_10 >= 0.30)
    d["not_touch9hi"] = not _flag(row, "fig9_touch_hi")
    d["g10_gt_g9"] = bool(g10 > g9)
    d["g1"] = bool(masks["g1"].iloc[-1])

    buy, sell = buy_sell_masks(masks, spec)
    d["buy_signal"], d["sell_signal"] = bool(buy.iloc[-1]), bool(sell.iloc[-1])
    d["trail_hit"] = bool(state.get("trail_hit"))
    d["sell_masked_by_g1"] = bool(state.get("sell_masked_by_g1"))
    stop = state.get("stop")
    d["dist_stop"] = px / stop - 1 if stop else float("nan")
    return d


def flat_buy_quote(snap: dict[str, Any], spec: RuleSpec, *, applicable: bool = True) -> tuple[float | None, str]:
    """Price that counts as touching fig9's lower rail, plus why it is or isn't a buy.

    The price is ``fig9_lower × (1 + near)`` for this close only. It is filled for
    every name, including names that are already long. It is not an intraday order.
    """
    try:
        price = float(snap.get("lo9")) * (1.0 + DEFAULT_NEAR_PCT)
    except (TypeError, ValueError):
        price = float("nan")
    if not np.isfinite(price):
        return None, "无下轨"
    if not applicable:
        return price, "规则不适用"
    if spec.fast_block and bool(snap.get("fast")):
        return price, "快形态挡住"
    if bool(snap.get("buy_signal")):
        return price, "今天已是买点"
    if spec.b1 and bool(snap.get("disc_ok")) and not bool(snap.get("near_lo")):
        return price, "贴轨即可"
    if not bool(snap.get("aux_buy")):
        return price, "辅助买像未齐"
    if bool(snap.get("near_lo")):
        return price, "已在带内，明天同价不算"
    return price, "还差这个价"


TRIGGER_LABELS: dict[str, str] = {
    "near_lo_edge": "近下轨上跳沿",
    "aux_buy": "辅助买像",
    "ae_buy": "aux_edge买点",
    "b1": "B1",
    "fast": "快形态",
    "buy_signal": "买入信号",
    "near_hi": "图9近上轨",
    "aux_sell": "辅助卖像",
    "sell_a": "卖点A",
    "stretch": "带内拉伸B",
    "near_hi8": "图8近上轨",
    "sell_signal": "卖出信号",
    "g1": "G1今日",
    "trail_hit": "回落触发",
    "sell_masked_by_g1": "卖点被G1屏蔽",
}


def triggered(snap: dict[str, Any]) -> list[str]:
    return [label for key, label in TRIGGER_LABELS.items() if snap.get(key)]


def evaluate(code: str, frame: pd.DataFrame, *, cost: float = 0.0005) -> dict[str, Any]:
    """Classify, build masks, run the state machine to the last bar."""
    frame = frame.copy()
    frame["px"] = frame["px"].astype(float).ffill()
    feat = classify_features(frame)
    typ, source, spec = resolve_spec(code, feat)
    masks = build_masks(frame)
    buy, sell = buy_sell_masks(masks, spec)
    res = run_state_machine(frame["px"], buy, sell,
                            g1=masks["g1"] if spec.g1_hold else None,
                            trail=spec.trail, cost=cost)
    snap = indicator_snapshot(frame, masks, spec, res["state"])
    return {"type": typ, "source": source, "spec": spec, "features": feat,
            "masks": masks, "buy": buy, "sell": sell, "snapshot": snap, **res}
