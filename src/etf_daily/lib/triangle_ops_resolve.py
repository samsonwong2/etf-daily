"""Normalize triangle candidates and resolve one main decision per code.

Used by ``scan_triangle_decision_ops.py`` for both intraday and EOD OPS markdown.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

_EXEC_AT_RE = re.compile(r"执行点@([\d.]+)（@(\d{4}-\d{2}-\d{2})）")
_OPEN_AT_RE = re.compile(r"(\d{4}-\d{2}-\d{2})开盘@([\d.]+)")
_RECLAIM_AT_RE = re.compile(r"已于(\d{4}-\d{2}-\d{2})收[复盘]")

# Lower rank = higher priority for the unique main-table row.
SECTION_RANK = {
    "immediate_risk": 1,
    "pending_confirm": 2,
    "conditional": 3,
    "position_mgmt": 4,
    "audit": 5,
    "no_action": 6,
}

DECISION_RANK = {
    "顶部交替减仓": 10,
    "卖确认": 20,
    "卖/看跌延续": 30,
    "卖预警待确认": 40,
    "红三角不抄底": 50,
    "高位回砸观察": 55,
    "顶部交替警戒": 58,
    "买确认待收盘": 60,
    "买确认下一开盘": 65,
    "买观察": 70,
    # 刚错过的收复买（含卖/看跌延续转买）优先于底部交替「持有/逢低」叙述
    "已过试多窗口": 71,
    # 深跌红未升买观察：仍公布收复观察位（非买点），优先于纯底部交替说明
    "收复观察": 72,
    # 近深跌红高点已收复：筑底（横盘），优先于仍破MA20的「底部交替→下跌」分流
    "筑底观察": 74,
    "底部交替观察": 75,
    "趋势内交替观察": 78,
    "绿三角观察": 80,
    "顶部区已失效": 95,
    "趋势上涨-无三角": 96,
    "趋势下跌-无三角": 96,
    "横盘-无三角": 97,
    "忽略": 100,
}

# OPS markdown 五类（回测整理后对人可读口径）
OPS_BUCKET_BUY = "buy"
OPS_BUCKET_SELL = "sell"
OPS_BUCKET_DOWN = "down"
OPS_BUCKET_UP = "up"
OPS_BUCKET_RANGE = "range"

OPS_BUCKET_LABEL = {
    OPS_BUCKET_BUY: "潜在买入点",
    OPS_BUCKET_SELL: "潜在卖出点",
    OPS_BUCKET_DOWN: "趋势下跌中",
    OPS_BUCKET_UP: "趋势上涨中",
    OPS_BUCKET_RANGE: "无趋势横盘震荡",
}

_BUY_DECISIONS = {
    "买观察",
    "买确认",
    "买确认下一开盘",
    "买确认待收盘",
    "已过试多窗口",
}
_SELL_DECISIONS = {
    "卖预警",
    "卖确认",
    "卖预警待确认",
    "顶部交替减仓",
    "卖/看跌延续",
    "高位回砸观察",
}
_UP_DECISIONS = {"趋势内交替观察", "顶部交替警戒", "趋势上涨-无三角"}
_RANGE_DECISIONS = {
    "底部交替观察",
    "筑底观察",
    "忽略",
    "绿三角观察",
    "横盘-无三角",
    "收复观察",
}
_DOWN_DECISIONS = {"红三角不抄底", "趋势下跌-无三角"}
# 顶部区已失效：结构破坏是历史事件，不落固定桶，按当前趋势分流。
# 收复观察：与底部交替一样按确认趋势分流，绝不进潜在买入桶。
# 筑底观察：近深跌红已收复，固定横盘桶（不因仍破MA20进下跌）。


@dataclass(frozen=True)
class SignalGateConfig:
    """Live signal quality gates (fitted on the inception 5-class backtest).

    Test-segment (2024+) validation: baseline 41.4%/-0.66% -> gated 55.2%/+2.24%.
      卖预警       41.2% -> 60.8% (only ≤ 中强度 kept; 强/极端波动是噪声)
      顶区收复买    37.9% -> 50.0% (only prior10 < -8% 深跌 or > 0% 翻红 kept)
      买观察       43.4% -> 50.0% (only vol_pct >= 0.80 kept)
    """

    sell_max_strength: str = "中"
    buy_watch_min_vol: float = 0.80
    post_top_prior_deep: float = -0.08
    post_top_allow_positive: bool = True
    # 收复执行点迟到参与带:现价距执行点 ≤ +2% 仍可轻仓参与(时间窗过严的修正,
    # 如 SH516770 07/28 现价仅距执行点+0.66% 却被一刀切"窗口已过")。
    late_entry_band: float = 0.02


_SIGNAL_GATE = SignalGateConfig()


def late_entry_state(
    last_px: Any,
    exec_px: Any,
    stop: Any,
    gate: SignalGateConfig = _SIGNAL_GATE,
) -> str:
    """Classify a passed buy execution point vs current price.

    Returns one of: ``接近执行点`` (still joinable), ``已远离执行点`` (don't
    chase), ``反弹失败`` (fell back below stop).
    """
    try:
        last = float(last_px)
        exe = float(exec_px)
    except (TypeError, ValueError):
        return "已远离执行点"
    if exe <= 0:
        return "已远离执行点"
    try:
        if stop is not None and not pd.isna(stop) and last <= float(stop):
            return "反弹失败"
    except (TypeError, ValueError):
        pass
    if last / exe - 1.0 <= gate.late_entry_band:
        return "接近执行点"
    return "已远离执行点"


def _strength_rank_local(s: str) -> int:
    return {"弱": 1, "中": 2, "强": 3}.get(str(s or ""), 0)


def sell_passes_gate(strength: str, gate: SignalGateConfig = _SIGNAL_GATE) -> bool:
    """卖预警 quality gate: 强/极端波动 sells are noise (test win 36-38%)."""
    return _strength_rank_local(strength) <= _strength_rank_local(gate.sell_max_strength)


def buy_watch_passes_gate(vol_pct: Any, gate: SignalGateConfig = _SIGNAL_GATE) -> bool:
    """买观察 quality gate: low vol_pct (<0.80) reclaim watches underperform."""
    try:
        return pd.notna(vol_pct) and float(vol_pct) >= gate.buy_watch_min_vol
    except (TypeError, ValueError):
        return False


def post_top_buy_passes_gate(prior10: Any, gate: SignalGateConfig = _SIGNAL_GATE) -> bool:
    """顶区失效-收复买 gate: keep 深跌 (<-8%) or 翻红 (>0%), drop 中段 (-8%~0%)."""
    try:
        if pd.isna(prior10):
            return False
        p = float(prior10)
    except (TypeError, ValueError):
        return False
    if p < gate.post_top_prior_deep:
        return True
    return bool(gate.post_top_allow_positive and p > 0)


def ops_bucket(
    decision: str,
    *,
    trend_intact: bool | None = None,
) -> str:
    """Map fine-grained decision → one of five OPS buckets."""
    d = str(decision or "")
    # Normalize numpy.bool_ / NA so identity checks below are reliable.
    if trend_intact is not None:
        if isinstance(trend_intact, float) and pd.isna(trend_intact):
            trend_intact = None
        else:
            trend_intact = bool(trend_intact)
    if d in _BUY_DECISIONS:
        return OPS_BUCKET_BUY
    if d in _SELL_DECISIONS:
        return OPS_BUCKET_SELL
    if d in _UP_DECISIONS:
        return OPS_BUCKET_UP
    if d in _DOWN_DECISIONS:
        return OPS_BUCKET_DOWN
    # 顶部区已失效：结构破坏属历史事件，五类桶按当前确认趋势分流
    # （trend_intact=None 表示 MA20 缠绕带，仍归横盘）
    if d == "顶部区已失效":
        if trend_intact is True:
            return OPS_BUCKET_UP
        if trend_intact is False:
            return OPS_BUCKET_DOWN
        return OPS_BUCKET_RANGE
    if d in _RANGE_DECISIONS:
        # 筑底：近深跌红高点已收复 → 固定横盘，不因仍破 MA20 进下跌。
        if d == "筑底观察":
            return OPS_BUCKET_RANGE
        # 绿三角 / 底部交替：细粒度决策名保留，五类桶按确认趋势分流
        # （trend_intact=None 表示 MA20 缠绕带，仍归横盘）
        if d in {"绿三角观察", "底部交替观察", "收复观察"} and trend_intact is True:
            return OPS_BUCKET_UP
        if d in {"绿三角观察", "底部交替观察", "收复观察"} and trend_intact is False:
            return OPS_BUCKET_DOWN
        return OPS_BUCKET_RANGE
    if trend_intact is True:
        return OPS_BUCKET_UP
    if trend_intact is False:
        return OPS_BUCKET_DOWN
    return OPS_BUCKET_RANGE


def _parse_buy_trigger_meta(op: str) -> tuple[float | None, str | None]:
    """Extract (exec_px, reached_date) from operation text when structured fields missing."""
    text = str(op or "")
    m = _EXEC_AT_RE.search(text)
    if m:
        try:
            return float(m.group(1)), m.group(2)
        except ValueError:
            pass
    m = _OPEN_AT_RE.search(text)
    if m:
        try:
            return float(m.group(2)), m.group(1)
        except ValueError:
            pass
    return None, None


def _reached_need_label(
    *,
    last: Any,
    trig: Any,
    reached_date: str | None,
) -> str:
    """Format 需涨幅 cell when trigger already hit: 已达@日期；已涨+x%."""
    rise_s = ""
    try:
        last_f = float(last)
        trig_f = float(trig)
        if trig_f > 0 and np.isfinite(last_f) and np.isfinite(trig_f):
            chg = last_f / trig_f - 1.0
            rise_s = f"已涨{chg:+.1%}" if chg >= 0 else f"较触发{chg:+.1%}"
    except (TypeError, ValueError):
        rise_s = ""
    if reached_date and rise_s:
        return f"已达@{reached_date}；{rise_s}"
    if reached_date:
        return f"已达@{reached_date}"
    if rise_s:
        return f"已达；{rise_s}"
    return "已达"


def buy_exec_summary(row: dict[str, Any] | pd.Series) -> tuple[str, str, str]:
    """Return (trigger_px, need_pct, exec_text) for buy bucket MD.

    When trigger already hit, ``need_pct`` becomes
    ``已达@YYYY-MM-DD；已涨+x%`` (date = exec/open day; rise vs trigger).
    """
    last = row.get("last_px")
    trig = row.get("trigger_price")
    status = str(row.get("status") or "")
    decision = str(row.get("decision") or "")
    op = str(row.get("operation") or "")
    entry = str(row.get("entry_state") or "")

    parsed_px, parsed_date = _parse_buy_trigger_meta(op)
    ep = row.get("exec_px")
    if ep is not None and isinstance(ep, float) and pd.isna(ep):
        ep = None

    # Prefer structured exec/trigger; fall back to operation parse.
    if trig is None or (isinstance(trig, float) and pd.isna(trig)):
        trig = ep if ep is not None else parsed_px
    reached_date = parsed_date
    if reached_date is None:
        m = _RECLAIM_AT_RE.search(op)
        if m:
            # reclaim close day is one session before typical next-open exec
            reached_date = m.group(1)

    trig_s = _fmt_px(trig)
    need_s = ""
    try:
        if trig is not None and last is not None and float(last) > 0:
            need = float(trig) / float(last) - 1.0
            need_s = f"{need:+.1%}" if need > 0 else _reached_need_label(
                last=last, trig=trig, reached_date=reached_date
            )
    except (TypeError, ValueError):
        need_s = ""

    already_hit = (
        status == "接近执行点仍可参与"
        or "参与带内" in op
        or decision == "已过试多窗口"
        or status == "收复窗口已过"
        or entry == "expired"
        or (need_s.startswith("已达") if need_s else False)
    )

    if status == "接近执行点仍可参与" or "参与带内" in op:
        use_px = ep if ep is not None else (trig if trig is not None else parsed_px)
        trig_s = _fmt_px(use_px)
        need_s = _reached_need_label(
            last=last, trig=use_px, reached_date=reached_date
        )
        exec_s = op or "现价接近执行点，可轻仓参与；跌回止损则离场"
        return trig_s, need_s, exec_s
    if decision == "已过试多窗口" or status == "收复窗口已过":
        use_px = ep if ep is not None else (trig if trig is not None else parsed_px)
        trig_s = _fmt_px(use_px)
        need_s = _reached_need_label(
            last=last, trig=use_px, reached_date=reached_date
        )
        exec_s = (
            op
            if ("涨离执行点" in op or "放弃参与" in op)
            else "买入窗口已过，不追买；有仓则管止损"
        )
        return trig_s, need_s, exec_s
    if status == "待下一交易日开盘" or decision in {
        "买确认下一开盘",
        "买确认",
    }:
        exec_s = "下一开盘试多（小仓）；低开破止损则取消"
    elif decision == "买确认待收盘":
        exec_s = "等收盘确认后，下一交易日开盘试多；禁止盘中追买"
    elif trig_s and not already_hit:
        exec_s = (
            f"涨至{trig_s}"
            + (f"（需{need_s}）" if need_s and not need_s.startswith("已达") else "")
            + "收复红三角高点后，次日开盘试多；未达不加仓"
        )
    else:
        exec_s = op or "条件满足后试多"
    return trig_s, need_s, exec_s


def sell_exec_summary(row: dict[str, Any] | pd.Series) -> str:
    """Short execution text for sell bucket."""
    decision = str(row.get("decision") or "")
    op = str(row.get("operation") or "")
    if decision in {"卖预警", "卖预警待确认"}:
        return "若持仓：盯盘跌破信号低点/MA5则卖；未破不机械卖"
    if decision == "卖确认":
        return "若持仓：卖点已确认，复核是否已减仓，未减则减"
    if decision == "顶部交替减仓":
        return "若持仓：借反弹减仓/清仓；跌破后反弹不加仓"
    if decision == "高位回砸观察":
        return "若持仓：高位红三角不抄底；配合减仓，不加仓"
    if decision == "卖/看跌延续":
        return "若持仓：偏减仓/不抄底"
    return op or "若持仓：减仓观望"



@dataclass
class Candidate:
    code: str
    name: str
    decision: str
    section: str
    side: str = ""
    strength: str = ""
    signal_date: str = ""
    signal_px: float | None = None
    last_px: float | None = None
    last_date: str = ""
    vol_pct: float | None = None
    prior10: float | None = None
    trigger_price: float | None = None
    invalidation: str = ""
    operation: str = ""
    decision_reason: str = ""
    provisional: bool = False
    entry_state: str = "forward"
    source: str = "history"  # today | history
    pred_cdf: float | None = None
    rv_expand: bool | None = None
    above_ma20: bool | None = None
    signal_high: float | None = None
    trend_intact: bool | None = None
    trend_since: str = ""
    trend_label: str = ""
    suppressed_by: str = ""
    is_main: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def sort_key(self) -> tuple[int, int, str]:
        return (
            SECTION_RANK.get(self.section, 99),
            DECISION_RANK.get(self.decision, 999),
            self.signal_date,
        )


def _fmt_px(x: Any) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return str(x)


def reclaim_buy_trigger_note(
    *,
    trigger_hi: Any,
    ref_px: Any,
    ref_label: str = "现价",
) -> str:
    """Explicit next-session buy trigger: price level + % rise from reference.

    Example: 「买入触发：涨至1.069（较现价需+2.6%）=收复红三角高点，趋势可能买入点」
    """
    try:
        hi = float(trigger_hi)
        ref = float(ref_px)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(hi) or not np.isfinite(ref) or ref <= 0:
        return f"买入触发：涨至{_fmt_px(hi)}=收复红三角高点，趋势可能买入点"
    need = hi / ref - 1.0
    if need <= 0:
        return (
            f"买入触发：已达/超过收复位{_fmt_px(hi)}（{ref_label}{_fmt_px(ref)}），"
            f"趋势可能买入点已触发，下一开盘试多"
        )
    return (
        f"买入触发：涨至{_fmt_px(hi)}（较{ref_label}需{need:+.1%}）"
        f"=收复红三角高点，趋势可能买入点"
    )


def reclaim_watch_note(
    *,
    trigger_hi: Any,
    ref_px: Any,
    signal_date: Any = None,
    ref_label: str = "现价",
) -> str:
    """Publish unreclaimed red-day high as a watch level — NOT a buy signal.

    Used for deep reds blocked from 「买观察」(prior10 floor / rv gate). Full-pool
    backtest showed extreme-deep reclaim is only marginal; keep wording strictly
    observational.
    """
    try:
        hi = float(trigger_hi)
        ref = float(ref_px)
    except (TypeError, ValueError):
        return ""
    if signal_date is not None and str(signal_date):
        try:
            date_s = f"{pd.Timestamp(signal_date).date()}红三角高点"
        except Exception:  # noqa: BLE001
            date_s = "红三角高点"
    else:
        date_s = "红三角高点"
    if not np.isfinite(hi) or not np.isfinite(ref) or ref <= 0:
        return f"收复观察位：{_fmt_px(hi)}={date_s}（观察位，非买点）"
    need = hi / ref - 1.0
    if need <= 0:
        return (
            f"收复观察位：{_fmt_px(hi)}={date_s}已触及/超过（{ref_label}{_fmt_px(ref)}）；"
            f"须收盘站上后次日开盘再评估，盘中触及不下手"
        )
    return (
        f"收复观察位：涨至{_fmt_px(hi)}（较{ref_label}需{need:+.1%}）={date_s}；"
        f"观察位，非买点；须收盘站上后次日开盘再评估，盘中触及不下手"
    )


def _prior_s(prior: Any) -> str:
    if prior is None or (isinstance(prior, float) and not np.isfinite(prior)):
        return "?"
    try:
        return f"{float(prior):+.1%}"
    except (TypeError, ValueError):
        return "?"


def _prior_label(prior: Any) -> str:
    """Human label: prior10 is ~10-session cumulative return, not same-day drop."""
    return f"近10日累计{_prior_s(prior)}"


def candidates_from_today_triangle(
    row: dict[str, Any] | pd.Series,
    *,
    trade_s: str,
    intraday: bool,
) -> Candidate | None:
    """Map today's annotated triangle into a forward-looking candidate."""
    side = str(row.get("side") or "")
    action = str(row.get("action") or "")
    code = str(row.get("code") or "")
    name = str(row.get("name") or "")
    prior = row.get("prior10")
    rv_exp = row.get("rv_expand")
    last_px = row.get("last_px")
    signal_px = row.get("signal_px")
    strength = str(row.get("strength") or "")
    signal_date = str(row.get("signal_date") or "")
    vol_pct = row.get("vol_pct")
    hi = row.get("signal_high")
    provisional = bool(intraday)

    # Trend context: route gated 绿三角观察 into 趋势上涨/下跌 instead of 横盘.
    # Hugging MA20 (±2% band) → None → 横盘 bucket.
    trend_intact = row.get("trend_intact")
    if isinstance(trend_intact, float) and pd.isna(trend_intact):
        trend_intact = None
    elif trend_intact is not None:
        trend_intact = bool(trend_intact)
    _hug = row.get("trend_hugging")
    if _hug is not None and not (isinstance(_hug, float) and pd.isna(_hug)) and bool(_hug):
        trend_intact = None
    trend_since = (
        str(row.get("trend_since") or "") if pd.notna(row.get("trend_since")) else ""
    )
    trend_label = str(row.get("trend_label") or "")
    if not trend_label:
        if trend_intact is True and trend_since:
            trend_label = f"未破@{trend_since}"
        elif trend_intact is False:
            trend_label = "已破MA20"

    base = dict(
        code=code,
        name=name,
        side=side,
        strength=strength,
        signal_date=signal_date,
        signal_px=None if signal_px is None else float(signal_px),
        last_px=None if last_px is None else float(last_px),
        last_date=signal_date,
        vol_pct=None if vol_pct is None or pd.isna(vol_pct) else float(vol_pct),
        prior10=None if prior is None or pd.isna(prior) else float(prior),
        provisional=provisional,
        source="today",
        pred_cdf=None
        if row.get("pred_cdf") is None or pd.isna(row.get("pred_cdf"))
        else float(row.get("pred_cdf")),
        rv_expand=None if rv_exp is None or (isinstance(rv_exp, float) and pd.isna(rv_exp)) else bool(rv_exp),
        above_ma20=row.get("above_ma20")
        if row.get("above_ma20") is None or not (isinstance(row.get("above_ma20"), float) and pd.isna(row.get("above_ma20")))
        else None,
        signal_high=None if hi is None or pd.isna(hi) else float(hi),
        entry_state="forward",
        trend_intact=trend_intact,
        trend_since=trend_since,
        trend_label=trend_label,
    )

    if side == "绿":
        if action == "卖预警" and sell_passes_gate(strength):
            return Candidate(
                **base,
                decision="卖预警待确认",
                section="conditional",
                trigger_price=base["signal_px"],
                invalidation="未跌破信号低点/MA5则不卖",
                operation=(
                    f"若持仓：{trade_s}盯盘跌破信号低点/MA5则卖；"
                    f"未破不机械卖（信号价{_fmt_px(signal_px)}）"
                ),
                decision_reason="今日绿三角达卖预警",
            )
        if action == "卖预警":
            # Gated out: 强/极端波动 sells are noise (test-seg win 36-38%).
            return Candidate(
                **base,
                decision="绿三角观察",
                section="pending_confirm",
                operation=(
                    f"绿三角强度{strength}超出gated卖出口径（极端波动噪声），"
                    f"{trade_s}不机械卖，若变GR交替再减（现价{_fmt_px(last_px)}）"
                ),
                decision_reason=f"今日绿三角卖预警被gated过滤（强度{strength}）",
            )
        if action == "买确认候选":
            if intraday:
                return Candidate(
                    **base,
                    decision="买确认待收盘",
                    section="pending_confirm",
                    invalidation="收盘前被红三角/破位则取消",
                    operation=(
                        f"盘中仅预警：等{signal_date}收盘确认后，"
                        f"下一交易日开盘试多；禁止盘中追买（现价{_fmt_px(last_px)}）"
                    ),
                    decision_reason="今日绿三角买确认（盘中待收盘）",
                )
            return Candidate(
                **base,
                decision="买确认下一开盘",
                section="pending_confirm",
                invalidation="下一开盘前若破位则取消",
                operation=(
                    f"{trade_s}开盘试多（小仓）；勿用已收盘价下单；"
                    f"信号收盘{_fmt_px(signal_px)}"
                ),
                decision_reason="今日绿三角买确认（EOD→次日开盘）",
            )
        if action == "忽略":
            return Candidate(
                **base,
                decision="忽略",
                section="no_action",
                operation=f"绿三角未进规则；{trade_s}无动作",
                decision_reason="绿三角规则=忽略",
            )
        return Candidate(
            **base,
            decision="绿三角观察",
            section="pending_confirm",
            operation=(
                f"绿三角未达卖预警；{trade_s}不机械卖，"
                f"若变GR交替再减（现价{_fmt_px(last_px)}）"
            ),
            decision_reason="今日绿三角观察",
        )

    # 红
    if action == "买观察" and not buy_watch_passes_gate(vol_pct):
        # Gated out: low-vol reclaim watches underperform (test-seg win 29%).
        return Candidate(
            **base,
            decision="红三角不抄底",
            section="pending_confirm",
            operation=(
                f"红三角买观察但vol_pct不足gated口径（<{_SIGNAL_GATE.buy_watch_min_vol:.2f}），"
                f"{trade_s}不抄底，等收盘复核"
            ),
            decision_reason=f"今日红三角买观察被gated过滤（vol_pct={vol_pct}）",
        )
    if action == "买观察":
        # Intraday: buy signals wait for close confirm before acting.
        section = "pending_confirm" if intraday else "conditional"
        trigger_note = reclaim_buy_trigger_note(
            trigger_hi=hi,
            ref_px=last_px,
            ref_label="快照现价" if intraday else "现价",
        )
        op = (
            (
                f"盘中预警：等{signal_date}收盘确认后，"
                f"{trigger_note}；未达不加仓（快照现价{_fmt_px(last_px)}）"
            )
            if intraday
            else (
                f"{trade_s}盯盘；{trigger_note}；"
                f"未达不加仓（现价{_fmt_px(last_px)}）"
            )
        )
        return Candidate(
            **base,
            decision="买观察",
            section=section,
            trigger_price=None if hi is None or pd.isna(hi) else float(hi),
            invalidation=f"未收复{_fmt_px(hi)}不加仓",
            operation=op,
            decision_reason=f"今日红三角买观察 {_prior_label(prior)}",
        )
    if action == "高位回砸观察":
        return Candidate(
            **base,
            decision="高位回砸观察",
            section="immediate_risk",
            operation=f"若持仓：{trade_s}高位红三角不抄底；配合顶部区减仓",
            decision_reason="今日红三角高位回砸",
        )
    if action == "卖/看跌延续":
        return Candidate(
            **base,
            decision="卖/看跌延续",
            section="immediate_risk",
            operation=(
                f"若持仓：{trade_s}红三角看跌延续（{_prior_label(prior)}），"
                f"偏减仓/不抄底"
            ),
            decision_reason="今日红三角卖/看跌延续",
        )
    if action == "忽略":
        return Candidate(
            **base,
            decision="忽略",
            section="no_action",
            operation=f"红三角波动/跌幅未进规则；{trade_s}无动作",
            decision_reason="红三角规则=忽略",
        )
    # action == 观察
    if prior is not None and not pd.isna(prior) and float(prior) < -0.08:
        too_deep = float(prior) < -0.08  # below buy_watch_floor band
        if too_deep and rv_exp is True:
            why = "近10日回撤过深(超出-5%~-8%买观察带)"
        elif rv_exp is False:
            why = "rv5>rv20未满足"
        else:
            why = "条件不全"
        return Candidate(
            **base,
            decision="红三角不抄底",
            section="pending_confirm",
            trigger_price=None if hi is None or pd.isna(hi) else float(hi),
            operation=(
                f"红三角，{_prior_label(prior)}，{why}→未升买观察；"
                f"{trade_s}不抄底；"
                f"{reclaim_watch_note(trigger_hi=hi, ref_px=last_px, signal_date=signal_date)}"
            ),
            decision_reason=f"今日红三角{_prior_label(prior)}但{why}",
        )
    return Candidate(
        **base,
        decision="红三角不抄底",
        section="pending_confirm",
        operation=f"红三角；{trade_s}先观察，不机械交易",
        decision_reason="今日红三角观察",
    )


def candidates_from_history_row(
    row: dict[str, Any] | pd.Series,
    *,
    trade_s: str,
    asof_s: str,
    intraday: bool,
) -> Candidate | None:
    """Map historical active states (sell warn / buy watch / zone / expired)."""
    decision = str(row.get("decision") or "")
    status = str(row.get("status") or "")
    entry = str(row.get("entry_state") or "forward")
    code = str(row.get("code") or "")
    name = str(row.get("name") or "")
    side = str(row.get("side") or "")
    signal_date = str(row.get("signal_date") or "")
    last_px = row.get("last_px")
    signal_px = row.get("signal_px")
    vol_pct = row.get("vol_pct")
    provisional = bool(intraday)
    px_label = "快照现价" if intraday else "收盘价"
    op_text = str(row.get("operation") or "")
    # Same hug-band routing as today triangles: ±2% → None → 横盘桶。
    trend_intact = row.get("trend_intact")
    if isinstance(trend_intact, float) and pd.isna(trend_intact):
        trend_intact = None
    elif trend_intact is not None:
        trend_intact = bool(trend_intact)
    _hug = row.get("trend_hugging")
    if _hug is not None and not (isinstance(_hug, float) and pd.isna(_hug)) and bool(_hug):
        trend_intact = None
    elif str(row.get("trend_label") or "") == "MA20缠绕":
        trend_intact = None
    trend_since = str(row.get("trend_since") or "") if pd.notna(row.get("trend_since")) else ""
    trend_label = str(row.get("trend_label") or "")
    if not trend_label:
        if trend_intact is True and trend_since:
            trend_label = f"未破@{trend_since}"
        elif trend_intact is False:
            trend_label = "已破MA20"

    base = dict(
        code=code,
        name=name,
        side=side,
        strength=str(row.get("strength") or ""),
        signal_date=signal_date,
        signal_px=None if signal_px is None or pd.isna(signal_px) else float(signal_px),
        last_px=None if last_px is None or pd.isna(last_px) else float(last_px),
        last_date=str(row.get("last_date") or asof_s),
        vol_pct=None if vol_pct is None or pd.isna(vol_pct) else float(vol_pct),
        prior10=None
        if row.get("prior10") is None or pd.isna(row.get("prior10"))
        else float(row.get("prior10")),
        provisional=provisional,
        source="history",
        entry_state=entry,
        trend_intact=trend_intact,
        trend_since=trend_since,
        trend_label=trend_label,
        above_ma20=row.get("above_ma20")
        if row.get("above_ma20") is None
        or not (isinstance(row.get("above_ma20"), float) and pd.isna(row.get("above_ma20")))
        else None,
    )

    if decision == "顶部交替警戒":
        # Legacy label → treat as mid-trend watch (no top clear wording).
        return Candidate(
            **base,
            decision="趋势内交替观察",
            section="pending_confirm",
            invalidation=str(row.get("invalidation") or "跌破MA20则退出上行中继解读"),
            operation=op_text or f"{trade_s}趋势上行交替，非顶部；趋势未破可持有/逢低",
            decision_reason="MA20趋势未破，交替视为上行中继而非顶部",
        )
    if decision == "趋势内交替观察":
        return Candidate(
            **base,
            decision="趋势内交替观察",
            section="pending_confirm",
            invalidation=str(row.get("invalidation") or "跌破MA20则退出上行中继解读"),
            operation=op_text
            or f"{trade_s}趋势上行中的交替，非顶部结构；趋势未破持有/可逢低",
            decision_reason="MA20趋势未破，交替视为上行中继而非顶部",
        )
    if decision == "顶部区已失效":
        return Candidate(
            **base,
            decision="顶部区已失效",
            section="position_mgmt",
            invalidation=str(row.get("invalidation") or "结构已破坏"),
            operation=op_text
            or f"原顶部交替区已失效；{trade_s}不再按借反弹清/顶部减仓执行",
            decision_reason="区后同色连红且破MA20，顶部交替结构已破坏",
        )
    if decision == "底部交替观察":
        return Candidate(
            **base,
            decision="底部交替观察",
            section="pending_confirm",
            invalidation=str(row.get("invalidation") or "放量跌破区低则取消偏多解读"),
            operation=op_text
            or (
                f"{trade_s}底部盘整/变盘观察；非顶部清仓；"
                f"未达买观察条件前不机械抄底"
            ),
            decision_reason="MA20下交替（含GR波动聚集）→底部变盘观察",
        )
    if decision == "筑底观察":
        hi = row.get("signal_high")
        trigger = None if hi is None or pd.isna(hi) else float(hi)
        return Candidate(
            **base,
            decision="筑底观察",
            section="pending_confirm",
            trigger_price=trigger,
            invalidation=str(
                row.get("invalidation")
                or "再度跌破最近深跌红低点或放量破位则取消筑底解读"
            ),
            operation=op_text
            or (
                f"{trade_s}筑底观察：近深跌红高点已收复；"
                f"虽仍破MA20，按筑底而非趋势下跌处理；不机械抄底"
            ),
            decision_reason="近深跌红高点已收复→筑底（非趋势下跌）",
        )
    if decision == "收复观察":
        hi = row.get("signal_high")
        trigger = None if hi is None or pd.isna(hi) else float(hi)
        note = reclaim_watch_note(
            trigger_hi=trigger,
            ref_px=last_px,
            signal_date=signal_date,
        )
        return Candidate(
            **base,
            decision="收复观察",
            section="pending_confirm",
            trigger_price=trigger,
            invalidation=str(row.get("invalidation") or "盘中触及不加仓；未收盘站上不加仓"),
            operation=op_text
            or (
                f"{trade_s}盯盘；{note}；"
                f"深跌红未升买观察，仅公布观察位"
            ),
            decision_reason="深跌红三角未收复高点→收复观察位（非买点）",
        )
    if decision == "顶部交替减仓":
        age_days = None
        try:
            start = str(signal_date).split("~", 1)[0]
            age_days = (pd.Timestamp(asof_s).normalize() - pd.Timestamp(start).normalize()).days
        except Exception:  # noqa: BLE001
            age_days = None
        # Stale top zones leave the "immediate" main table; keep as position note.
        section = "immediate_risk" if age_days is None or age_days <= 15 else "position_mgmt"
        return Candidate(
            **base,
            decision="顶部交替减仓",
            section=section,
            invalidation=str(row.get("invalidation") or "跌破后反弹不加仓"),
            operation=op_text or f"若持仓：{trade_s}顶部交替减仓/清仓",
            decision_reason=(
                "历史顶部交替区仍有效（趋势已破或未确认）"
                if section == "immediate_risk"
                else "顶部交替区偏旧，仅作持仓管理参考"
            ),
        )
    if decision == "卖预警":
        if status == "刚确认":
            return Candidate(
                **base,
                decision="卖确认",
                section="immediate_risk",
                operation=op_text or f"若持仓：{trade_s}复核减仓",
                decision_reason="卖预警已确认",
            )
        return Candidate(
            **base,
            decision="卖预警待确认",
            section="conditional",
            trigger_price=base["signal_px"],
            invalidation="未跌破信号低点/MA5则不卖",
            operation=op_text or f"若持仓：{trade_s}盯盘跌破再卖",
            decision_reason="卖预警待确认",
        )
    if decision == "买观察":
        if status == "接近执行点仍可参与":
            ep = row.get("exec_px")
            return Candidate(
                **base,
                decision="买观察",
                # pending_confirm：可参与买点不可被底部交替盖住
                section="pending_confirm",
                trigger_price=None if ep is None or pd.isna(ep) else float(ep),
                invalidation=str(row.get("invalidation") or "跌回止损则离场"),
                operation=op_text or f"{trade_s}现价接近执行点，可轻仓参与",
                decision_reason="收复执行点迟到但现价仍在参与带内",
            )
        if entry == "expired" or status == "收复窗口已过":
            ep = row.get("exec_px")
            if ep is None or (isinstance(ep, float) and pd.isna(ep)):
                parsed_ep, _ = _parse_buy_trigger_meta(op_text)
                ep = parsed_ep
            return Candidate(
                **base,
                decision="已过试多窗口",
                # pending_confirm：近期收复买窗口不可被底部交替「持有/逢低」盖住
                section="pending_confirm",
                trigger_price=None if ep is None or pd.isna(ep) else float(ep),
                operation=op_text or f"{trade_s}有仓则管止损，不追买",
                decision_reason="买观察收复窗口已过",
            )
        if status == "待下一交易日开盘":
            return Candidate(
                **base,
                decision="买观察",
                section="pending_confirm",
                trigger_price=None
                if row.get("signal_high") is None or pd.isna(row.get("signal_high"))
                else float(row.get("signal_high")),
                invalidation=str(row.get("invalidation") or "开盘低开破止损则取消"),
                operation=op_text or f"{trade_s}开盘试多（收复已确认）",
                decision_reason="顶部破坏后连红低点已收复，待开盘试多",
            )
        hi = row.get("signal_high")
        trigger = None if hi is None or pd.isna(hi) else float(hi)
        note = reclaim_buy_trigger_note(trigger_hi=trigger, ref_px=last_px)
        op = op_text
        if note and "买入触发" not in (op_text or ""):
            op = f"{op_text}；{note}" if op_text else f"{trade_s}盯盘；{note}"
        return Candidate(
            **base,
            decision="买观察",
            section="conditional",
            trigger_price=trigger,
            invalidation=f"未收复{_fmt_px(trigger)}不加仓" if trigger else "未收复不加仓",
            operation=op,
            decision_reason="历史买观察待收复",
        )
    if decision == "买确认":
        if status == "接近执行点仍可参与":
            ep = row.get("exec_px")
            return Candidate(
                **base,
                decision="买观察",
                section="pending_confirm",
                trigger_price=None if ep is None or pd.isna(ep) else float(ep),
                invalidation=str(row.get("invalidation") or "跌回止损则离场"),
                operation=op_text or f"{trade_s}现价接近买确认执行点，可轻仓参与",
                decision_reason="买确认执行点迟到但现价仍在参与带内",
            )
        if entry == "expired" or status in {"已过开盘窗口", "收复窗口已过"}:
            return Candidate(
                **base,
                decision="已过试多窗口",
                section="pending_confirm",
                operation=op_text or f"{trade_s}只管理持仓，不按旧开盘价追买",
                decision_reason="买确认开盘窗口已过",
            )
        if intraday:
            return Candidate(
                **base,
                decision="买确认待收盘",
                section="pending_confirm",
                operation=(
                    f"盘中仅预警：收盘确认后下一交易日开盘试多；"
                    f"禁止盘中追买（{px_label}{_fmt_px(last_px)}）"
                ),
                decision_reason="买确认盘中待收盘",
            )
        return Candidate(
            **base,
            decision="买确认下一开盘",
            section="pending_confirm",
            operation=op_text or f"{trade_s}开盘试多",
            decision_reason="买确认待次日开盘",
        )
    if decision == "高位回砸观察":
        return Candidate(
            **base,
            decision="高位回砸观察",
            section="immediate_risk",
            operation=op_text or f"若持仓：{trade_s}高位红三角不抄底",
            decision_reason="高位回砸观察",
        )
    if decision in {"趋势上涨-无三角", "趋势下跌-无三角", "横盘-无三角"}:
        return Candidate(
            **base,
            decision=decision,
            section="position_mgmt",
            operation=op_text or f"{trade_s}无三角信号，按MA20趋势归类",
            decision_reason="无红/绿三角信号，仅按MA20趋势归入五类",
        )
    return None


def _rank_tuple(c: Candidate) -> tuple:
    # Newer signal_date first when ranks tie (YYYY-MM-DD lexicographic reverse).
    return (
        SECTION_RANK.get(c.section, 99),
        DECISION_RANK.get(c.decision, 999),
        0 if c.source == "today" else 1,
        "".join(chr(255 - ord(ch)) for ch in (c.signal_date or "")),
    )


def resolve_main_per_code(candidates: list[Candidate]) -> list[Candidate]:
    """Pick one unique main decision per code; mark suppressed losers."""
    by_code: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_code.setdefault(c.code, []).append(c)

    resolved: list[Candidate] = []
    for _code, group in by_code.items():
        # Deduplicate buy watches: keep today / latest signal_date only.
        buys = [c for c in group if c.decision == "买观察"]
        if len(buys) > 1:
            buys_sorted = sorted(
                buys,
                key=lambda x: (
                    0 if x.source == "today" else 1,
                    "".join(chr(255 - ord(ch)) for ch in (x.signal_date or "")),
                ),
            )
            keep = buys_sorted[0]
            for old in buys_sorted[1:]:
                old.suppressed_by = f"{keep.decision}@{keep.signal_date}"
                old.is_main = False

        ranked = sorted(group, key=_rank_tuple)

        # Conflict: immediate risk suppresses buy观察
        risk_hit = next(
            (c for c in ranked if c.section == "immediate_risk" and not c.suppressed_by),
            None,
        )
        if risk_hit is not None:
            for c in ranked:
                if c.decision == "买观察" and not c.suppressed_by:
                    c.suppressed_by = f"{risk_hit.decision}@{risk_hit.signal_date}"
                    c.is_main = False

        # Today bearish / deep-red / buy-confirm covers historical buy观察
        today_cover = next(
            (
                c
                for c in ranked
                if c.source == "today"
                and not c.suppressed_by
                and c.decision
                in {
                    "买确认待收盘",
                    "买确认下一开盘",
                    "红三角不抄底",
                    "卖/看跌延续",
                    "高位回砸观察",
                }
            ),
            None,
        )
        if today_cover is not None:
            for c in ranked:
                if c.decision == "买观察" and c.source == "history" and not c.suppressed_by:
                    c.suppressed_by = f"{today_cover.decision}@{today_cover.signal_date}"
                    c.is_main = False

        # Prefer today over history duplicate of same decision on same date
        today_keys = {
            (c.decision, c.signal_date)
            for c in ranked
            if c.source == "today" and not c.suppressed_by
        }
        for c in ranked:
            if (
                c.source == "history"
                and (c.decision, c.signal_date) in today_keys
                and not c.suppressed_by
            ):
                c.suppressed_by = f"today:{c.decision}@{c.signal_date}"
                c.is_main = False
            # Also suppress history 买确认 when today already emitted 买确认*
            if (
                c.source == "history"
                and c.decision in {"买确认下一开盘", "买确认待收盘"}
                and any(
                    t.source == "today"
                    and t.decision in {"买确认待收盘", "买确认下一开盘"}
                    and not t.suppressed_by
                    for t in ranked
                )
                and not c.suppressed_by
            ):
                c.suppressed_by = "today_buy_confirm"
                c.is_main = False

        eligible = [c for c in ranked if not c.suppressed_by and c.section in SECTION_RANK]
        if not eligible:
            for c in group:
                c.is_main = False
                resolved.append(c)
            continue

        # Main table prefers actionable sections; fall back to no_action/audit.
        mains = [
            c
            for c in eligible
            if c.section
            in {"immediate_risk", "pending_confirm", "conditional", "position_mgmt"}
        ]
        main = mains[0] if mains else eligible[0]
        for c in group:
            c.is_main = c is main
            if (
                not c.is_main
                and not c.suppressed_by
                and c.decision == main.decision
                and c.signal_date != main.signal_date
            ):
                c.suppressed_by = f"{main.decision}@{main.signal_date}"
            resolved.append(c)
    return resolved


def candidates_to_frame(candidates: list[Candidate]) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame()
    rows = []
    for c in candidates:
        d = asdict(c)
        d.pop("extra", None)
        rows.append(d)
    return pd.DataFrame(rows)


def main_table(candidates: list[Candidate]) -> pd.DataFrame:
    mains = [c for c in candidates if c.is_main]
    return candidates_to_frame(mains)
