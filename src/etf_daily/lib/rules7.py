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

from etf_daily.scripts.backtest_fig9_touch import variant_masks

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
    TYPE_HIVOL: RuleSpec(b1=False, fast_block=True, sell="fig8", g1_hold=True, trail=None),
    TYPE_CYCLE: RuleSpec(b1=True, fast_block=False, sell="fig9", g1_hold=True, trail=0.12),
    TYPE_SHORT: RuleSpec(b1=True, fast_block=True, sell="fig9", g1_hold=True, trail=0.15),
}

OVERRIDES: dict[str, tuple[str, RuleSpec]] = {
    "SH512530": (TYPE_SMOOTH, RuleSpec(True, False, "fig8", False, 0.08)),
    "SH518880": (TYPE_TREND, RuleSpec(False, True, "fig9", True, 0.15)),
    "SH513100": (TYPE_TREND, RuleSpec(True, False, "fig9", True, 0.08)),
    "SH515880": (TYPE_HIVOL, RuleSpec(False, True, "fig8", True, None)),
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
    for i in range(n):
        act, reason = ("观望", "")
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
    return {"type": typ, "source": source, "spec": spec, "features": feat,
            "masks": masks, "buy": buy, "sell": sell, **res}
