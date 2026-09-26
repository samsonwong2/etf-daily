"""Next-session execution plan from frozen adaptive hold_up configs.

Signals are close-confirmed enter-up / leave-up only. Historical label backfill
from ``min_seg`` merge is never treated as a past fill; only the trade-date bar
can produce an executable signal.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from etf_daily.lib.adaptive_stage_common import prepare_features
from etf_daily.lib.early_down_alert import (
    render_early_down_markdown,
    write_early_down_artifacts,
)
from etf_daily.lib.execution_signal_overlay import (
    build_signal_board,
    render_signal_board_markdown,
    write_signal_board_artifacts,
)
from etf_daily.lib.portfolio import load_live_holdings_csv

Precision = Literal["exact_close_only", "conditional_estimate", "snapshot_ohlc"]
Bucket = Literal[
    "buy_candidate",
    "wait_buy",
    "no_chase",
    "hypo_sell_trigger",
    "hypo_exit_close",
    "hypo_hold",
    "unreachable",
    "other",
]

ADX_DEPENDENT_METHODS = frozenset({"hybrid_ma_adx", "adx_di"})
DEFAULT_LIMIT_PCT = 0.10
DEFAULT_TICK = 0.001
DEFAULT_COARSE_STEP_PCT = 0.0025

EXECUTION_PLAN_COLUMNS: tuple[str, ...] = (
    "code",
    "name",
    "method",
    "as_of",
    "trade_date",
    "last_close",
    "regime_asof",
    "model_open",
    "live_shares",
    "live_holding",
    "holdings_stale",
    "can_buy",
    "buy_action",
    "buy_trigger_px",
    "buy_trigger_pct",
    "buy_note",
    "can_sell_live",
    "sell_live_action",
    "sell_live_note",
    "hypo_can_sell",
    "hypo_sell_action",
    "hypo_sell_trigger_px",
    "hypo_sell_trigger_pct",
    "hypo_sell_note",
    "threshold_precision",
    "bucket",
    "data_source",
)


LabelFn = Callable[[pd.DataFrame], np.ndarray]


@dataclass(frozen=True)
class ScenarioBar:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass(frozen=True)
class ThresholdHit:
    price: float
    pct: float
    regime: str


def round_tick(price: float, tick: float = DEFAULT_TICK) -> float:
    if not math.isfinite(price) or tick <= 0:
        return float(price)
    n = round(price / tick)
    return float(round(n * tick, 10))


def pct_move(price: float, last_close: float) -> float:
    return float(price / last_close - 1.0)


def make_close_only_bar(
    *,
    date: str,
    close: float,
    volume: float = 0.0,
) -> ScenarioBar:
    px = float(close)
    return ScenarioBar(date=date, open=px, high=px, low=px, close=px, volume=float(volume))


def make_snapshot_bar(
    *,
    date: str,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 0.0,
) -> ScenarioBar:
    return ScenarioBar(
        date=date,
        open=float(open_),
        high=float(high),
        low=float(low),
        close=float(close),
        volume=float(volume),
    )


def append_scenario_bar(ohlcv: pd.DataFrame, bar: ScenarioBar) -> pd.DataFrame:
    """Append or replace the trade-date row; history before that date is kept."""
    if ohlcv is None or ohlcv.empty:
        raise ValueError("empty ohlcv")
    work = ohlcv.copy()
    if "datetime" not in work.columns:
        raise ValueError("ohlcv missing datetime")
    work["datetime"] = pd.to_datetime(work["datetime"])
    trade_ts = pd.Timestamp(bar.date).normalize()
    work = work[work["datetime"].dt.normalize() < trade_ts].copy()
    instrument = str(work["instrument"].iloc[-1]) if "instrument" in work.columns else ""
    row: dict[str, Any] = {
        "datetime": trade_ts,
        "$open": float(bar.open),
        "$high": float(bar.high),
        "$low": float(bar.low),
        "$close": float(bar.close),
        "$volume": float(bar.volume),
    }
    if "instrument" in work.columns:
        row["instrument"] = instrument
    if "$pct_change" in work.columns:
        prev = float(work["$close"].iloc[-1])
        row["$pct_change"] = (float(bar.close) / prev - 1.0) * 100.0 if prev > 0 else 0.0
    out = pd.concat([work, pd.DataFrame([row])], ignore_index=True)
    return out


def truncate_ohlcv_through(ohlcv: pd.DataFrame, as_of: str) -> pd.DataFrame:
    end = pd.Timestamp(as_of).normalize()
    work = ohlcv.copy()
    work["datetime"] = pd.to_datetime(work["datetime"])
    return work[work["datetime"].dt.normalize() <= end].reset_index(drop=True)


def label_regimes(
    ohlcv: pd.DataFrame,
    *,
    ev: Any | None = None,
    method: str | None = None,
    method_params: dict[str, Any] | None = None,
    label_fn: LabelFn | None = None,
) -> np.ndarray:
    if label_fn is not None:
        return np.asarray(label_fn(ohlcv), dtype=object)
    if ev is None or method is None:
        raise ValueError("ev+method or label_fn required")
    px = prepare_features(ev, ohlcv, method=method, method_params=method_params or {})
    return px["regime"].to_numpy(object)


def end_regime(
    ohlcv: pd.DataFrame,
    *,
    ev: Any | None = None,
    method: str | None = None,
    method_params: dict[str, Any] | None = None,
    label_fn: LabelFn | None = None,
) -> str:
    labs = label_regimes(
        ohlcv,
        ev=ev,
        method=method,
        method_params=method_params,
        label_fn=label_fn,
    )
    if len(labs) == 0:
        raise ValueError("no regime labels")
    return str(labs[-1])


def evaluate_trade_date_regime(
    ohlcv_asof: pd.DataFrame,
    *,
    trade_date: str,
    close: float,
    ev: Any | None = None,
    method: str | None = None,
    method_params: dict[str, Any] | None = None,
    label_fn: LabelFn | None = None,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
) -> str:
    if open_ is None or high is None or low is None:
        bar = make_close_only_bar(date=trade_date, close=close)
    else:
        bar = make_snapshot_bar(
            date=trade_date,
            open_=open_,
            high=high,
            low=low,
            close=close,
        )
    ohlcv = append_scenario_bar(ohlcv_asof, bar)
    return end_regime(
        ohlcv,
        ev=ev,
        method=method,
        method_params=method_params,
        label_fn=label_fn,
    )


def _price_grid(
    last_close: float,
    *,
    limit_pct: float = DEFAULT_LIMIT_PCT,
    tick: float = DEFAULT_TICK,
    coarse_step_pct: float = DEFAULT_COARSE_STEP_PCT,
) -> list[float]:
    lo = round_tick(last_close * (1.0 - limit_pct), tick)
    hi = round_tick(last_close * (1.0 + limit_pct), tick)
    if lo <= 0:
        lo = tick
    step = max(tick, round_tick(last_close * coarse_step_pct, tick))
    prices = []
    x = lo
    while x <= hi + tick * 0.5:
        prices.append(round_tick(x, tick))
        x += step
    # ensure last close and limits are included
    for px in (last_close, lo, hi):
        rp = round_tick(px, tick)
        if rp not in prices:
            prices.append(rp)
    return sorted(set(prices))


def _nearest_hit(
    hits: list[ThresholdHit],
    *,
    last_close: float,
    prefer: Literal["up", "down", "any"] = "any",
) -> ThresholdHit | None:
    if not hits:
        return None
    filtered = hits
    if prefer == "up":
        up = [h for h in hits if h.price >= last_close]
        filtered = up or hits
    elif prefer == "down":
        down = [h for h in hits if h.price <= last_close]
        filtered = down or hits
    return min(filtered, key=lambda h: (abs(h.pct), abs(h.price - last_close), h.price))


def _eval_px(
    ohlcv_asof: pd.DataFrame,
    *,
    trade_date: str,
    close: float,
    ev: Any | None,
    method: str | None,
    method_params: dict[str, Any] | None,
    label_fn: LabelFn | None,
) -> str:
    return evaluate_trade_date_regime(
        ohlcv_asof,
        trade_date=trade_date,
        close=close,
        ev=ev,
        method=method,
        method_params=method_params,
        label_fn=label_fn,
    )


def find_regime_thresholds(
    ohlcv_asof: pd.DataFrame,
    *,
    trade_date: str,
    last_close: float,
    target_regimes: set[str],
    ev: Any | None = None,
    method: str | None = None,
    method_params: dict[str, Any] | None = None,
    label_fn: LabelFn | None = None,
    limit_pct: float = DEFAULT_LIMIT_PCT,
    tick: float = DEFAULT_TICK,
    coarse_step_pct: float = DEFAULT_COARSE_STEP_PCT,
    prefer: Literal["up", "down", "any"] = "any",
) -> ThresholdHit | None:
    """Find nearest trade-date close whose end regime is in ``target_regimes``.

    Uses endpoint checks + binary search on the preferred side (fast path),
    then a coarse fallback grid if the response is non-monotonic.
    """
    del coarse_step_pct  # reserved for fallback grid density
    lo = max(tick, round_tick(last_close * (1.0 - limit_pct), tick))
    hi = round_tick(last_close * (1.0 + limit_pct), tick)
    mid0 = round_tick(last_close, tick)

    def hit(px: float, rg: str) -> ThresholdHit:
        return ThresholdHit(price=px, pct=pct_move(px, last_close), regime=rg)

    rg_mid = _eval_px(
        ohlcv_asof,
        trade_date=trade_date,
        close=mid0,
        ev=ev,
        method=method,
        method_params=method_params,
        label_fn=label_fn,
    )
    if rg_mid in target_regimes:
        return hit(mid0, rg_mid)

    candidates: list[ThresholdHit] = []

    if prefer in {"up", "any"}:
        rg_hi = _eval_px(
            ohlcv_asof,
            trade_date=trade_date,
            close=hi,
            ev=ev,
            method=method,
            method_params=method_params,
            label_fn=label_fn,
        )
        if rg_hi in target_regimes:
            # lowest price in [mid0, hi] that hits target
            left, right = mid0, hi
            best = hi
            while right - left > tick * 1.5:
                mid = round_tick((left + right) / 2.0, tick)
                if mid <= left or mid >= right:
                    break
                rg = _eval_px(
                    ohlcv_asof,
                    trade_date=trade_date,
                    close=mid,
                    ev=ev,
                    method=method,
                    method_params=method_params,
                    label_fn=label_fn,
                )
                if rg in target_regimes:
                    best = mid
                    right = mid
                else:
                    left = mid
            rg_best = _eval_px(
                ohlcv_asof,
                trade_date=trade_date,
                close=best,
                ev=ev,
                method=method,
                method_params=method_params,
                label_fn=label_fn,
            )
            if rg_best in target_regimes:
                candidates.append(hit(best, rg_best))

    if prefer in {"down", "any"}:
        rg_lo = _eval_px(
            ohlcv_asof,
            trade_date=trade_date,
            close=lo,
            ev=ev,
            method=method,
            method_params=method_params,
            label_fn=label_fn,
        )
        if rg_lo in target_regimes:
            # highest price in [lo, mid0] that hits target
            left, right = lo, mid0
            best = lo
            while right - left > tick * 1.5:
                mid = round_tick((left + right) / 2.0, tick)
                if mid <= left or mid >= right:
                    break
                rg = _eval_px(
                    ohlcv_asof,
                    trade_date=trade_date,
                    close=mid,
                    ev=ev,
                    method=method,
                    method_params=method_params,
                    label_fn=label_fn,
                )
                if rg in target_regimes:
                    best = mid
                    left = mid
                else:
                    right = mid
            rg_best = _eval_px(
                ohlcv_asof,
                trade_date=trade_date,
                close=best,
                ev=ev,
                method=method,
                method_params=method_params,
                label_fn=label_fn,
            )
            if rg_best in target_regimes:
                candidates.append(hit(best, rg_best))

    nearest = _nearest_hit(candidates, last_close=last_close, prefer=prefer)
    if nearest is not None:
        return nearest

    # Fallback sparse grid for non-monotonic methods.
    hits: list[ThresholdHit] = []
    for px in _price_grid(
        last_close,
        limit_pct=limit_pct,
        tick=tick,
        coarse_step_pct=max(0.01, DEFAULT_COARSE_STEP_PCT * 4),
    ):
        rg = _eval_px(
            ohlcv_asof,
            trade_date=trade_date,
            close=px,
            ev=ev,
            method=method,
            method_params=method_params,
            label_fn=label_fn,
        )
        if rg in target_regimes:
            hits.append(hit(px, rg))
    return _nearest_hit(hits, last_close=last_close, prefer=prefer)


def threshold_precision_for(
    method: str,
    *,
    has_snapshot_ohlc: bool,
) -> Precision:
    if has_snapshot_ohlc:
        return "snapshot_ohlc"
    if method in ADX_DEPENDENT_METHODS:
        return "conditional_estimate"
    return "exact_close_only"


def classify_bucket(
    *,
    can_buy: bool,
    buy_action: str,
    hypo_sell_action: str,
    buy_trigger_px: float | None,
    hypo_sell_trigger_px: float | None,
) -> Bucket:
    """Primary UI bucket. Buy-side state wins over hypothetical sell labels."""
    if can_buy:
        return "buy_candidate"
    if buy_action == "不追买":
        # Still surface hypo-hold/sell when useful for the open model book.
        if hypo_sell_action == "触及卖出价则卖":
            return "hypo_sell_trigger"
        if hypo_sell_action == "继续持有":
            return "hypo_hold"
        return "no_chase"
    if buy_action == "等待上涨切换":
        if buy_trigger_px is None:
            return "unreachable"
        return "wait_buy"
    if hypo_sell_action == "收盘直接退出":
        return "hypo_exit_close"
    if hypo_sell_action == "触及卖出价则卖":
        return "hypo_sell_trigger"
    if hypo_sell_action == "继续持有":
        return "hypo_hold"
    return "other"


def build_symbol_execution_row(
    *,
    code: str,
    name: str,
    method: str,
    method_params: dict[str, Any] | None,
    ohlcv: pd.DataFrame,
    as_of: str,
    trade_date: str,
    live_shares: float,
    holdings_stale: bool,
    model_open_hint: bool | None = None,
    ev: Any | None = None,
    label_fn: LabelFn | None = None,
    snapshot: dict[str, float] | None = None,
    limit_pct: float = DEFAULT_LIMIT_PCT,
    tick: float = DEFAULT_TICK,
    data_source: str = "20260807all_adaptive",
) -> dict[str, Any]:
    ohlcv_asof = truncate_ohlcv_through(ohlcv, as_of)
    if ohlcv_asof.empty:
        raise ValueError(f"{code}: no bars through {as_of}")
    last_close = float(ohlcv_asof["$close"].iloc[-1])
    regime_asof = end_regime(
        ohlcv_asof,
        ev=ev,
        method=method,
        method_params=method_params,
        label_fn=label_fn,
    )
    model_open = bool(model_open_hint) if model_open_hint is not None else (regime_asof == "up")
    live_holding = float(live_shares or 0.0) > 0
    has_snapshot = bool(snapshot)
    precision = threshold_precision_for(method, has_snapshot_ohlc=has_snapshot)

    # Evaluate unchanged close as baseline scenario on trade date.
    if has_snapshot:
        baseline_close = float(snapshot["close"])
        baseline_regime = evaluate_trade_date_regime(
            ohlcv_asof,
            trade_date=trade_date,
            close=baseline_close,
            open_=float(snapshot["open"]),
            high=float(snapshot["high"]),
            low=float(snapshot["low"]),
            ev=ev,
            method=method,
            method_params=method_params,
            label_fn=label_fn,
        )
    else:
        baseline_close = last_close
        baseline_regime = evaluate_trade_date_regime(
            ohlcv_asof,
            trade_date=trade_date,
            close=baseline_close,
            ev=ev,
            method=method,
            method_params=method_params,
            label_fn=label_fn,
        )

    buy_trigger: ThresholdHit | None = None
    hypo_sell_trigger: ThresholdHit | None = None

    # Live buy
    if live_holding:
        can_buy = False
        buy_action = "已持仓不重复买"
        buy_note = "实盘已持有，hold_up 不加仓"
    elif model_open or regime_asof == "up":
        can_buy = False
        buy_action = "不追买"
        buy_note = "08-07 模型已处上涨态且实盘空仓：不追买，等下一次非up→up"
    else:
        if baseline_regime == "up":
            can_buy = True
            buy_action = "收盘确认买入"
            buy_trigger = ThresholdHit(
                price=round_tick(baseline_close, tick),
                pct=pct_move(baseline_close, last_close),
                regime=baseline_regime,
            )
            buy_note = "收盘确认进入上涨态后买入"
        else:
            can_buy = False
            buy_action = "等待上涨切换"
            buy_trigger = find_regime_thresholds(
                ohlcv_asof,
                trade_date=trade_date,
                last_close=last_close,
                target_regimes={"up"},
                ev=ev,
                method=method,
                method_params=method_params,
                label_fn=label_fn,
                limit_pct=limit_pct,
                tick=tick,
                prefer="up",
            )
            if buy_trigger is None:
                buy_note = f"±{limit_pct:.0%} 涨跌停范围内收盘无法确认进入上涨态"
            else:
                buy_note = (
                    f"收盘≥{buy_trigger.price:.3f}（较08-07 {buy_trigger.pct:+.2%}）"
                    "且确认进入上涨态后买入；盘中仅预警"
                )

    # Live sell
    if not live_holding:
        can_sell_live = False
        sell_live_action = "不适用"
        sell_live_note = "实盘空仓，无卖出"
    elif regime_asof != "up":
        can_sell_live = True
        sell_live_action = "收盘卖出"
        sell_live_note = "模型已离上涨态，持仓应在收盘退出"
    else:
        if baseline_regime != "up":
            can_sell_live = True
            sell_live_action = "收盘确认卖出"
            sell_live_note = "收盘确认离开上涨态后卖出"
        else:
            can_sell_live = False
            sell_live_action = "继续持有"
            hypo_sell_trigger = find_regime_thresholds(
                ohlcv_asof,
                trade_date=trade_date,
                last_close=last_close,
                target_regimes={"range", "down"},
                ev=ev,
                method=method,
                method_params=method_params,
                label_fn=label_fn,
                limit_pct=limit_pct,
                tick=tick,
                prefer="down",
            )
            if hypo_sell_trigger is None:
                sell_live_note = f"±{limit_pct:.0%} 内收盘仍维持上涨态，不卖"
            else:
                sell_live_note = (
                    f"收盘≤{hypo_sell_trigger.price:.3f}（{hypo_sell_trigger.pct:+.2%}）"
                    "离开上涨态则卖出；盘中仅预警"
                )

    # Hypothetical sell (always filled for empty portfolio workflow)
    if regime_asof != "up" and not model_open:
        hypo_can_sell = True
        hypo_sell_action = "收盘直接退出"
        hypo_sell_note = "08-07 已非上涨态；若误持仓则 08-10 收盘退出"
        hypo_sell_trigger = ThresholdHit(
            price=round_tick(baseline_close, tick),
            pct=pct_move(baseline_close, last_close),
            regime=baseline_regime,
        )
    else:
        if baseline_regime != "up":
            hypo_can_sell = True
            hypo_sell_action = "收盘确认卖出"
            hypo_sell_trigger = ThresholdHit(
                price=round_tick(baseline_close, tick),
                pct=pct_move(baseline_close, last_close),
                regime=baseline_regime,
            )
            hypo_sell_note = "收盘确认离开上涨态后卖出（假设持有）"
        else:
            if hypo_sell_trigger is None:
                hypo_sell_trigger = find_regime_thresholds(
                    ohlcv_asof,
                    trade_date=trade_date,
                    last_close=last_close,
                    target_regimes={"range", "down"},
                    ev=ev,
                    method=method,
                    method_params=method_params,
                    label_fn=label_fn,
                    limit_pct=limit_pct,
                    tick=tick,
                    prefer="down",
                )
            if hypo_sell_trigger is None:
                hypo_can_sell = False
                hypo_sell_action = "继续持有"
                hypo_sell_note = f"假设持有：±{limit_pct:.0%} 内收盘难离上涨态"
            else:
                hypo_can_sell = True
                hypo_sell_action = "触及卖出价则卖"
                hypo_sell_note = (
                    f"假设持有：收盘≤{hypo_sell_trigger.price:.3f}"
                    f"（{hypo_sell_trigger.pct:+.2%}）离上涨态则卖"
                )

    if precision == "conditional_estimate":
        buy_note = f"{buy_note}｜ADX依赖OHLC，缺快照时为条件估计"
        hypo_sell_note = f"{hypo_sell_note}｜ADX依赖OHLC，缺快照时为条件估计"

    bucket = classify_bucket(
        can_buy=can_buy,
        buy_action=buy_action,
        hypo_sell_action=hypo_sell_action,
        buy_trigger_px=buy_trigger.price if buy_trigger else None,
        hypo_sell_trigger_px=hypo_sell_trigger.price if hypo_sell_trigger else None,
    )

    return {
        "code": code,
        "name": name,
        "method": method,
        "as_of": as_of,
        "trade_date": trade_date,
        "last_close": round(last_close, 6),
        "regime_asof": regime_asof,
        "model_open": bool(model_open),
        "live_shares": float(live_shares or 0.0),
        "live_holding": bool(live_holding),
        "holdings_stale": bool(holdings_stale),
        "can_buy": bool(can_buy),
        "buy_action": buy_action,
        "buy_trigger_px": None if buy_trigger is None else round(buy_trigger.price, 6),
        "buy_trigger_pct": None if buy_trigger is None else round(buy_trigger.pct, 6),
        "buy_note": buy_note,
        "can_sell_live": bool(can_sell_live),
        "sell_live_action": sell_live_action,
        "sell_live_note": sell_live_note,
        "hypo_can_sell": bool(hypo_can_sell),
        "hypo_sell_action": hypo_sell_action,
        "hypo_sell_trigger_px": (
            None if hypo_sell_trigger is None else round(hypo_sell_trigger.price, 6)
        ),
        "hypo_sell_trigger_pct": (
            None if hypo_sell_trigger is None else round(hypo_sell_trigger.pct, 6)
        ),
        "hypo_sell_note": hypo_sell_note,
        "threshold_precision": precision,
        "bucket": bucket,
        "data_source": data_source,
        "baseline_regime_trade_date": baseline_regime,
    }


def load_holdings_shares(
    path: Path | None,
    *,
    as_of: str,
) -> tuple[dict[str, float], bool, str]:
    """Return shares by code, stale flag, and note."""
    if path is None or not Path(path).exists():
        return {}, True, "missing_holdings"
    path = Path(path)
    # empty CSV (header only) => flat
    try:
        snap = load_live_holdings_csv(path, as_of=as_of)
    except Exception as exc:  # pragma: no cover - defensive
        return {}, True, f"holdings_load_error:{exc}"
    shares = {h.code: float(h.shares or 0.0) for h in snap.holdings if float(h.shares or 0) > 0}
    # Compare filename date vs as_of when possible
    stale = True
    stem = path.stem
    if len(stem) == 8 and stem.isdigit():
        file_asof = f"{stem[:4]}-{stem[4:6]}-{stem[6:8]}"
        stale = file_asof < as_of or len(shares) == 0 and file_asof < as_of
        if file_asof < as_of:
            return shares, True, f"holdings_file_dated_{file_asof}_before_{as_of}"
    if not shares:
        return {}, True, "empty_holdings_treated_as_flat"
    return shares, stale, "holdings_loaded"


def load_batch_open_pos(batch_summary: Path | pd.DataFrame) -> dict[str, bool]:
    if isinstance(batch_summary, Path):
        df = pd.read_csv(batch_summary)
    else:
        df = batch_summary
    out: dict[str, bool] = {}
    for _, row in df.iterrows():
        code = str(row["code"]).upper()
        val = row.get("open_pos")
        if isinstance(val, str):
            out[code] = val.strip().lower() in {"1", "true", "yes", "y"}
        else:
            out[code] = bool(val)
    return out


def build_execution_plan_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=list(EXECUTION_PLAN_COLUMNS))
    extras = [c for c in frame.columns if c not in EXECUTION_PLAN_COLUMNS]
    ordered = list(EXECUTION_PLAN_COLUMNS) + extras
    return frame.reindex(columns=ordered)


def next_trade_date(as_of: str | pd.Timestamp) -> str:
    """First exchange session strictly after ``as_of`` (qlib calendar, else BDay)."""
    as_of_ts = pd.Timestamp(as_of).normalize()
    try:
        from qlib.data import D

        end = (as_of_ts + pd.Timedelta(days=20)).strftime("%Y-%m-%d")
        cal = D.calendar(start_time=as_of_ts.strftime("%Y-%m-%d"), end_time=end)
        future = [
            pd.Timestamp(d).normalize()
            for d in cal
            if pd.Timestamp(d).normalize() > as_of_ts
        ]
        if future:
            return future[0].strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        pass
    return (as_of_ts + pd.offsets.BDay(1)).normalize().strftime("%Y-%m-%d")


def _fmt_px(value: Any) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or pd.isna(value))):
        return "—"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or pd.isna(value))):
        return "—"
    try:
        return f"{float(value):+.2%}"
    except (TypeError, ValueError):
        return "—"


def _md_escape(text: Any) -> str:
    return str(text).replace("|", "\\|")


def render_execution_plan_markdown(
    frame: pd.DataFrame,
    *,
    meta: dict[str, Any],
    early_down: pd.DataFrame | None = None,
    signal_board: pd.DataFrame | None = None,
) -> str:
    """Full daily ops-style markdown for adaptive hold_up execution."""
    export = frame.copy()
    if export.empty:
        export = pd.DataFrame(columns=list(EXECUTION_PLAN_COLUMNS))

    as_of = str(meta.get("as_of") or "")
    trade_date = str(meta.get("trade_date") or "")
    limit_pct = float(meta.get("limit_pct") or DEFAULT_LIMIT_PCT)
    n = len(export)
    n_buy = int(export["can_buy"].sum()) if n and "can_buy" in export.columns else 0
    n_sell_live = (
        int(export["can_sell_live"].sum()) if n and "can_sell_live" in export.columns else 0
    )
    n_no_chase = int((export.get("buy_action") == "不追买").sum()) if n else 0
    n_wait = int((export.get("buy_action") == "等待上涨切换").sum()) if n else 0
    n_buy_trig = (
        int(export["buy_trigger_px"].notna().sum())
        if n and "buy_trigger_px" in export.columns
        else 0
    )
    n_unreach = int((export.get("bucket") == "unreachable").sum()) if n else 0
    n_hypo_exit = int((export.get("hypo_sell_action") == "收盘直接退出").sum()) if n else 0
    n_hypo_trig = int((export.get("hypo_sell_action") == "触及卖出价则卖").sum()) if n else 0
    n_hypo_hold = int((export.get("hypo_sell_action") == "继续持有").sum()) if n else 0
    n_cond = (
        int((export.get("threshold_precision") == "conditional_estimate").sum()) if n else 0
    )

    lines: list[str] = [
        f"# Adaptive hold_up 执行方案（{trade_date}）",
        "",
        f"> 数据截止 `{as_of}` · 执行日 `{trade_date}` · 收盘确认 · "
        f"涨跌停扫描 ±{limit_pct:.0%}",
        "",
        "## 摘要",
        "",
        f"- 持仓: `{meta.get('holdings_note')}`"
        + ("（过期/空仓告警）" if meta.get("holdings_stale") else ""),
        f"- 票数: **{n}**",
        f"- 可买: **{n_buy}** · 有买入触发价: **{n_buy_trig}** · "
        f"±{limit_pct:.0%} 内难买入: **{n_unreach}**",
        f"- 不追买（模型已上涨/空仓）: **{n_no_chase}**",
        f"- 等待上涨切换: **{n_wait}**",
        f"- 实盘可卖: **{n_sell_live}**",
        f"- 假设持有·收盘退出: **{n_hypo_exit}** · 触及卖出价: **{n_hypo_trig}** · "
        f"继续持有: **{n_hypo_hold}**",
        f"- ADX 条件估计: **{n_cond}**",
        f"- 自适应目录: `{meta.get('adaptive_dir')}`",
        "",
        "## 结论",
        "",
    ]

    if n_buy == 0 and n_sell_live == 0:
        lines.append(
            f"实盘口径：**{trade_date} 全部不买、全部不卖**"
            + ("（当前按空仓处理）。" if meta.get("holdings_stale") else "。")
        )
    else:
        lines.append(
            f"实盘口径：可买 **{n_buy}** 票，可卖 **{n_sell_live}** 票；"
            "未列触发价者见下方明细。"
        )
    lines.extend(
        [
            "",
            f"- 模型上涨且空仓不追买：{n_no_chase} 票。",
            f"- 横盘/非上涨：假设持有则收盘退出 {n_hypo_exit} 票；"
            f"单日 ±{limit_pct:.0%} 情景无法确认 enter-up 的有 {n_unreach} 票。",
            "",
            "## 规则",
            "",
            "1. 买入：仅 `非 up → up`，且信号在 trade_date **首次可见**；已上涨不追买。",
            "2. 实盘卖出：无持仓则不适用；持仓且离开上涨态则收盘卖出。",
            "3. 假设持有卖出：as_of 已非 up → 收盘退出；仍为 up → 扫描离上涨态阈值。",
            "4. 执行时机：盘中触价仅预警，14:50/收盘确认后执行。",
            "5. `min_seg` 历史回填不视为已成交。",
            "",
        ]
    )

    if "signals_compact" not in export.columns:
        if signal_board is not None and len(signal_board):
            sig = signal_board[["code", "signals_compact"]].copy()
            export = export.merge(sig, on="code", how="left")
            export["signals_compact"] = export["signals_compact"].fillna("—")
        else:
            export["signals_compact"] = "—"
    else:
        export["signals_compact"] = export["signals_compact"].fillna("—")

    def _section(title: str, mask: pd.Series, *, empty: str) -> None:
        lines.extend([f"## {title}", ""])
        sub = export.loc[mask].copy() if n else export.iloc[0:0].copy()
        if sub.empty:
            lines.extend([empty, ""])
            return
        lines.append(
            "| 代码 | 名称 | 方法 | as_of收盘 | 状态 | 信号 | 买入动作 | 买入触发价 | "
            "需涨跌幅 | 假设卖出 | 卖出触发价 | 卖出涨跌幅 |"
        )
        lines.append("|---|---|---|---:|---|---|---|---:|---:|---|---:|---:|")
        for _, r in sub.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md_escape(r.get("code")),
                        _md_escape(r.get("name")),
                        _md_escape(r.get("method")),
                        _fmt_px(r.get("last_close")),
                        _md_escape(r.get("regime_asof")),
                        _md_escape(r.get("signals_compact") or "—"),
                        _md_escape(r.get("buy_action")),
                        _fmt_px(r.get("buy_trigger_px")),
                        _fmt_pct(r.get("buy_trigger_pct")),
                        _md_escape(r.get("hypo_sell_action")),
                        _fmt_px(r.get("hypo_sell_trigger_px")),
                        _fmt_pct(r.get("hypo_sell_trigger_pct")),
                    ]
                )
                + " |"
            )
        lines.append("")

    if n:
        _section(
            "可买候选",
            export["can_buy"].fillna(False).astype(bool),
            empty="（无）",
        )
        _section(
            "不追买 / 模型上涨",
            export["buy_action"].eq("不追买"),
            empty="（无）",
        )
        _section(
            "有买入触发价（等待确认）",
            export["buy_trigger_px"].notna() & ~export["can_buy"].fillna(False).astype(bool),
            empty="（无；±涨跌停内无法确认 enter-up）",
        )
        _section(
            "实盘可卖",
            export["can_sell_live"].fillna(False).astype(bool),
            empty="（无；当前空仓则均为不适用）",
        )
        _section(
            "假设持有·需关注卖出",
            export["hypo_sell_action"].isin(["收盘直接退出", "触及卖出价则卖", "收盘确认卖出"]),
            empty="（无）",
        )
        _section(
            "全市场明细",
            pd.Series(True, index=export.index),
            empty="（无）",
        )

    if signal_board is not None:
        lines.append(render_signal_board_markdown(signal_board, meta=meta).rstrip())
        lines.append("")

    if early_down is not None:
        lines.append(render_early_down_markdown(early_down, meta=meta).rstrip())
        lines.append("")

    lines.extend(
        [
            "## 每日生成",
            "",
            "```bash",
            "# 默认：AS_OF=今天对应的上一交易数据日目录 {YYYYMMDD}all_adaptive",
            "# TRADE_DATE 自动取下一交易日",
            "AS_OF=2026-08-07 PY=~/etf-daily-output/python/envs/py312/bin/python \\",
            "  ./scripts/daily_adaptive_execution_plan.sh",
            "",
            "# 或直接调用：",
            "python src/etf_daily/scripts/generate_adaptive_execution_plan.py \\",
            "  --adaptive-dir ~/etf-daily-output/temp/plotly_outputs/20260807all_adaptive \\",
            "  --as-of 2026-08-07 --trade-date 2026-08-10 \\",
            "  --live-holdings ~/etf-daily-output/temp/live_holdings/20260717.csv",
            "```",
            "",
            "## 文件",
            "",
            f"- `EXEC_for_{trade_date.replace('-', '')}_from_{as_of.replace('-', '')}.md`（本文件）",
            "- `execution_plan.csv`",
            "- `execution_plan.json`",
            "- `signal_board.csv` / `.json` / `.md`（三角/菱形/ED 标注）",
            "- `early_down_alerts.csv` / `.json` / `.md`（诊断层，不改买卖）",
            "- `README.md`",
            "",
        ]
    )
    return "\n".join(lines)


def write_execution_plan_artifacts(
    frame: pd.DataFrame,
    out_dir: Path,
    *,
    meta: dict[str, Any],
    early_down: pd.DataFrame | None = None,
    signal_board: pd.DataFrame | None = None,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "execution_plan.csv"
    json_path = out_dir / "execution_plan.json"
    readme_path = out_dir / "README.md"
    as_of = str(meta.get("as_of") or "asof")
    trade_date = str(meta.get("trade_date") or "trade")
    md_name = (
        f"EXEC_for_{trade_date.replace('-', '')}_from_{as_of.replace('-', '')}.md"
    )
    md_path = out_dir / md_name

    export = frame.reindex(columns=[c for c in EXECUTION_PLAN_COLUMNS if c in frame.columns])
    # Keep ED→exec overlay fields if present (not in EXECUTION_PLAN_COLUMNS core).
    for extra in (
        "ed_level",
        "ed_trigger",
        "ed_trigger_source",
        "ed_lookback_k",
        "ed_overlay_mode",
        "ed_overlay_applied",
        "ed_overlay_note",
    ):
        if extra in frame.columns and extra not in export.columns:
            export[extra] = frame[extra].values

    # HTML-parity signal overlay (triangles / hollow diamonds / ED).
    adaptive_dir = Path(str(meta.get("adaptive_dir") or ""))
    window_start = str(meta.get("signal_window_start") or "2026-04-01")
    window_end = str(meta.get("signal_window_end") or trade_date or as_of)
    meta = dict(meta)
    meta["signal_window"] = f"{window_start}..{window_end}"
    meta.setdefault("signal_window_start", window_start)
    meta.setdefault("signal_window_end", window_end)

    if signal_board is None and adaptive_dir.is_dir() and len(export):
        signal_board = build_signal_board(
            export,
            adaptive_dir=adaptive_dir,
            window_start=window_start,
            window_end=window_end,
            as_of=as_of,
            early_down=early_down,
        )

    if signal_board is not None and len(signal_board):
        # Historical ED path marks — do not overwrite as-of ed_level used by overlay.
        rename = {
            "ed_level": "ed_hist_level",
            "ed_run": "ed_hist_run",
            "ed_reasons": "ed_hist_reasons",
        }
        board = signal_board.rename(columns=rename)
        sig_cols = [
            c
            for c in (
                "signals_compact",
                "triangle_buy",
                "triangle_sell",
                "diamond_V",
                "diamond_up",
                "diamond_X",
                "diamond_down",
                "ed_hist_level",
                "ed_hist_run",
                "ed_hist_reasons",
                "signals_detail",
            )
            if c in board.columns
        ]
        export = export.merge(
            board[["code", *sig_cols]],
            on="code",
            how="left",
        )

    export.to_csv(csv_path, index=False, encoding="utf-8-sig")

    rows = export.replace({np.nan: None}).to_dict(orient="records")
    payload = {"meta": meta, "rows": rows}
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_text = render_execution_plan_markdown(
        export,
        meta=meta,
        early_down=early_down,
        signal_board=signal_board,
    )
    md_path.write_text(md_text, encoding="utf-8")
    # README points at the dated EXEC md so daily folders stay discoverable.
    n = len(export)
    n_buy = int(export["can_buy"].sum()) if n and "can_buy" in export.columns else 0
    n_no_chase = int((export.get("buy_action") == "不追买").sum()) if n else 0
    n_hypo_exit = int((export.get("hypo_sell_action") == "收盘直接退出").sum()) if n else 0
    n_hypo_trig = int((export.get("hypo_sell_action") == "触及卖出价则卖").sum()) if n else 0
    n_unreach = int((export.get("bucket") == "unreachable").sum()) if n else 0
    n_ed_alert = 0
    if early_down is not None and len(early_down):
        n_ed_alert = int((early_down.get("level") == "alert").sum())
    n_sig = 0
    if signal_board is not None and len(signal_board):
        n_sig = int(signal_board["signals_compact"].fillna("—").ne("—").sum())
    readme_lines = [
        f"# Adaptive hold_up 执行方案（{trade_date}）",
        "",
        f"- 完整 Markdown：[`{md_name}`]({md_name})",
        f"- as_of: `{as_of}` · trade_date: `{trade_date}`",
        f"- 持仓: {meta.get('holdings_note')}",
        f"- 票数: {n} · 可买: {n_buy} · 不追买: {n_no_chase} · "
        f"难买入: {n_unreach} · 假设退出: {n_hypo_exit} · 假设卖阈值: {n_hypo_trig}",
        f"- Early-down alert（诊断）: {n_ed_alert}",
        f"- 信号标注（三角/菱形/ED）: {n_sig} 票有标记 · 窗口 `{meta.get('signal_window')}`",
        "",
        "## 每日命令",
        "",
        "```bash",
        f"AS_OF={as_of} PY=~/etf-daily-output/python/envs/py312/bin/python \\",
        "  ./scripts/daily_adaptive_execution_plan.sh",
        "```",
        "",
        "## 文件",
        "",
        f"- `{md_name}`",
        "- `execution_plan.csv`",
        "- `execution_plan.json`",
        "- `signal_board.csv` / `signal_board.json` / `signal_board.md`",
        "- `early_down_alerts.csv` / `early_down_alerts.json` / `early_down_alerts.md`",
        "",
    ]
    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    paths: dict[str, Path] = {
        "csv": csv_path,
        "json": json_path,
        "readme": readme_path,
        "markdown": md_path,
    }
    if signal_board is not None:
        sb_paths = write_signal_board_artifacts(signal_board, out_dir, meta=meta)
        paths["signal_board_csv"] = sb_paths["csv"]
        paths["signal_board_json"] = sb_paths["json"]
        paths["signal_board_markdown"] = sb_paths["markdown"]
    if early_down is not None:
        ed_paths = write_early_down_artifacts(early_down, out_dir, meta=meta)
        paths["early_down_csv"] = ed_paths["csv"]
        paths["early_down_json"] = ed_paths["json"]
        paths["early_down_markdown"] = ed_paths["markdown"]
    return paths


__all__ = [
    "ADX_DEPENDENT_METHODS",
    "EXECUTION_PLAN_COLUMNS",
    "ScenarioBar",
    "ThresholdHit",
    "append_scenario_bar",
    "build_execution_plan_frame",
    "build_symbol_execution_row",
    "classify_bucket",
    "end_regime",
    "evaluate_trade_date_regime",
    "find_regime_thresholds",
    "label_regimes",
    "load_batch_open_pos",
    "load_holdings_shares",
    "make_close_only_bar",
    "make_snapshot_bar",
    "next_trade_date",
    "pct_move",
    "render_execution_plan_markdown",
    "round_tick",
    "threshold_precision_for",
    "truncate_ohlcv_through",
    "write_execution_plan_artifacts",
]
