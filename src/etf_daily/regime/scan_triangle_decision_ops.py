#!/usr/bin/env python3
"""Scan pool ETFs for triangle decision ops → write OPS markdown + CSVs.

Two modes
---------
**EOD** (default): ``--as-of`` = last close; OPS guides the **next** trading day.

**Intraday** (``--intraday`` or auto when ``live_snapshot.csv`` in pack):
``--as-of`` = provisional bar date; OPS guides **same day** (rest of session).

Examples
--------
EOD (Friday close → Monday ops)::

    $PY src/etf_daily/regime/scan_triangle_decision_ops.py \\
      --as-of 2026-07-24 \\
      --continue-on-error

Intraday pack (one directory for signals + html + snapshot)::

    $PY src/etf_daily/regime/scan_triangle_decision_ops.py \\
      --pack-dir runtime/decision_packs/intraday/20260728/103630 \\
      --intraday \\
      --continue-on-error

Outputs under ``<validation-dir>/triangle_decision_scan_<asof>/``:
  - Intraday: ``OPS_intraday_<asof>.md`` + compat ``OPS_<asof>.md`` (no OPS_for_*)
  - EOD: ``OPS_<asof>.md`` + ``OPS_for_<trade_date>_from_<asof>.md``
  - CSVs: all_decision_points / ops_primary_fresh / today_triangles / active_zones
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from etf_daily.scripts.backtest_triangle_decision_rules import (  # noqa: E402
    RuleConfig,
    _is_below_ma20,
    _load_plot_mod,
    _next_open,
    annotate_switches,
    build_price_frame,
    detect_alternating_zones,
    simulate_buy_confirm,
    simulate_buy_watch,
    simulate_sell,
    top_zone_post_red_run,
    top_zone_superseded_by_red_run,
)
from etf_daily.lib.extreme_drop_override import (  # noqa: E402
    enrich_oos_from_ohlcv_loader,
)
from etf_daily.lib.indicators import ma20_trend_state  # noqa: E402
from etf_daily.lib.regime_transition_plot import (  # noqa: E402
    compute_volatility_regime_frame,
    load_validation_csvs,
)
from etf_daily.lib.triangle_ops_resolve import (  # noqa: E402
    Candidate,
    OPS_BUCKET_BUY,
    OPS_BUCKET_DOWN,
    OPS_BUCKET_LABEL,
    OPS_BUCKET_RANGE,
    OPS_BUCKET_SELL,
    OPS_BUCKET_UP,
    buy_exec_summary,
    buy_watch_passes_gate,
    candidates_from_history_row,
    candidates_from_today_triangle,
    candidates_to_frame,
    late_entry_state,
    main_table,
    ops_bucket,
    post_top_buy_passes_gate,
    reclaim_buy_trigger_note,
    reclaim_watch_note,
    resolve_main_per_code,
    sell_exec_summary,
    sell_passes_gate,
)


def next_trade_date(as_of: pd.Timestamp) -> pd.Timestamp:
    """First exchange session strictly after ``as_of`` (qlib calendar, else BDay)."""
    as_of = pd.Timestamp(as_of).normalize()
    try:
        from qlib.data import D

        end = (as_of + pd.Timedelta(days=20)).strftime("%Y-%m-%d")
        cal = D.calendar(start_time=as_of.strftime("%Y-%m-%d"), end_time=end)
        future = [
            pd.Timestamp(d).normalize()
            for d in cal
            if pd.Timestamp(d).normalize() > as_of
        ]
        if future:
            return future[0]
    except Exception:  # noqa: BLE001
        pass
    return (as_of + pd.offsets.BDay(1)).normalize()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (_PROJECT_ROOT / path)


def codes_from_html_dir(html_dir: Path) -> list[str]:
    codes: set[str] = set()
    for p in sorted(html_dir.glob("regime_transition_*.html")):
        m = re.search(r"regime_transition_([A-Z]{2}\d+)_", p.name)
        if m:
            codes.add(m.group(1))
    return sorted(codes)


def _live_row(live: pd.DataFrame | None, code: str) -> pd.Series | None:
    if live is None or live.empty:
        return None
    hit = live[live["code"].astype(str).str.upper() == code]
    return None if hit.empty else hit.iloc[0]


def _inject_live_bar(
    ohlcv: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    live_row: pd.Series,
    plot_mod: Any,
) -> pd.DataFrame:
    work = ohlcv.copy()
    work["as_of"] = pd.to_datetime(work["datetime"]).dt.normalize()
    asof_n = as_of.normalize()
    if asof_n in set(work["as_of"]):
        return work.drop(columns=["as_of"], errors="ignore")

    def _pick(*keys: str) -> float:
        for k in keys:
            if k in live_row.index and pd.notna(live_row[k]):
                return float(live_row[k])
        return float("nan")

    add: dict[str, Any] = {}
    for c in work.columns:
        if c == "as_of":
            continue
        if c == "datetime":
            add[c] = as_of
        elif c == "$open":
            add[c] = _pick("open", "$open")
        elif c == "$high":
            add[c] = _pick("high", "$high")
        elif c == "$low":
            add[c] = _pick("low", "$low")
        elif c == "$close":
            add[c] = _pick("close", "$close")
        elif c == "$volume":
            add[c] = _pick("volume", "$volume")
        else:
            add[c] = np.nan
    out = pd.concat(
        [work.drop(columns=["as_of"], errors="ignore"), pd.DataFrame([add])],
        ignore_index=True,
    )
    return plot_mod.attach_sma_columns(out)


def _fmt_px(x: Any) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return str(x)


def _latest_deep_red(
    ann: pd.DataFrame,
    *,
    buy_from: pd.Timestamp,
) -> pd.Series | None:
    deep_reds = ann[
        (ann["side"] == "down")
        & (ann["as_of"] >= buy_from)
        & (pd.to_numeric(ann["prior10"], errors="coerce") < -0.08)
    ].sort_values("as_of")
    if deep_reds.empty:
        return None
    return deep_reds.iloc[-1]


def _deep_red_high_reclaimed(
    s: pd.Series,
    px: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
) -> bool:
    d = pd.Timestamp(s["as_of"]).normalize()
    hi = float(s["high"])
    post = px[(px["as_of"] > d) & (px["as_of"] <= as_of)]
    return (not post.empty) and bool((post["$close"] > hi).any())


def pick_latest_deep_red_for_reclaim_watch(
    ann: pd.DataFrame,
    px: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    buy_from: pd.Timestamp,
    buy_watch_dates: set[pd.Timestamp] | set[Any],
) -> pd.Series | None:
    """Latest deep-red (prior10<-8%) unreclaimed high, or None.

    Only the most recent deep red in ``[buy_from, as_of]`` is eligible. If that
    bar already upgraded to 买观察 or a later close reclaimed its high, return
    None — do not fall back to older unreclaimed highs (zombie watches).
    """
    s = _latest_deep_red(ann, buy_from=buy_from)
    if s is None:
        return None
    d = pd.Timestamp(s["as_of"]).normalize()
    if d in buy_watch_dates:
        return None
    if _deep_red_high_reclaimed(s, px, as_of=as_of):
        return None
    return s


def pick_latest_deep_red_reclaimed(
    ann: pd.DataFrame,
    px: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    buy_from: pd.Timestamp,
    buy_watch_dates: set[pd.Timestamp] | set[Any],
) -> pd.Series | None:
    """Latest deep red whose high was later closed through → 筑底 (not 下跌).

    If the latest deep red upgraded to 买观察, return None (buy path owns it).
    """
    s = _latest_deep_red(ann, buy_from=buy_from)
    if s is None:
        return None
    d = pd.Timestamp(s["as_of"]).normalize()
    if d in buy_watch_dates:
        return None
    if not _deep_red_high_reclaimed(s, px, as_of=as_of):
        return None
    return s


def _is_soft_red_reclaim_eligible(row: pd.Series | dict[str, Any]) -> bool:
    """Down reds that are not formal 买观察 / 高位回砸, excluding deep 观察.

    Covers:
      - 卖/看跌延续（低波动 mid 跌）
      - 忽略（更浅的低波动红，如 SH513080 prior10≈−2%）
      - 观察 with prior10 ≥ −8%（中段；深跌交由收复观察/筑底路径）
    """
    if str(row.get("side") or "") != "down":
        return False
    action = str(row.get("action") or "")
    if action in {"卖/看跌延续", "忽略"}:
        return True
    if action != "观察":
        return False
    try:
        p = float(row.get("prior10"))
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(p) and p >= -0.08)


def plan_sell_continuation_reclaim_buys(
    ann: pd.DataFrame,
    px: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    buy_from: pd.Timestamp,
    cfg: RuleConfig,
    last_px: float | None = None,
) -> list[dict[str, Any]]:
    """软红（卖/看跌延续/忽略/中段观察）收盘收复高点 → 买观察执行计划。

    Returns dicts with: signal, reclaim_d, exec_d, exec_px, stop, hi, red_d, phase
    where phase ∈ {pending_open, near_exec, failed, expired}.

    Note: when reclaim prints on ``as_of`` itself, next open may be absent from
    ``px`` — still emit ``pending_open``. Stale filter follows ``buy_from`` on the
    red date (not a short 3-day exec window), so earlier reclaims like SH513080
    07-27→07-28 still surface as expired within the reclaim lookback.
    """
    out: list[dict[str, Any]] = []
    as_of_n = pd.Timestamp(as_of).normalize()
    if last_px is None:
        last_bar = px[px["as_of"] <= as_of_n]
        last_px = float(last_bar.iloc[-1]["$close"]) if not last_bar.empty else None
    soft = ann[ann["as_of"] >= buy_from].copy()
    if soft.empty:
        return out
    soft = soft[soft.apply(_is_soft_red_reclaim_eligible, axis=1)]
    for _, s in soft.sort_values("as_of").iterrows():
        red_d = pd.Timestamp(s["as_of"]).normalize()
        hi = float(s["high"])
        stop = float(s["low"])
        # Only bars after the red, known by as_of (incl. reclaim on as_of).
        post_px = px[(px["as_of"] > red_d) & (px["as_of"] <= as_of_n)].sort_values(
            "as_of"
        )
        post_px = post_px.head(int(cfg.buy_reclaim_days))
        # Use the start of the *current* close>high streak. A prior touch that
        # later closed back ≤ high is a false reclaim (SZ159509: 07-29 yes →
        # 07-30 lost → 07-31 real reclaim).
        reclaim_d: pd.Timestamp | None = None
        for _, r in post_px.iterrows():
            if float(r["$close"]) > hi:
                if reclaim_d is None:
                    reclaim_d = pd.Timestamp(r["as_of"]).normalize()
            else:
                reclaim_d = None
        if reclaim_d is None:
            continue
        exec_d, exec_px = _next_open(px, reclaim_d)
        if exec_d is None or pd.Timestamp(exec_d).normalize() > as_of_n:
            phase = "pending_open"
            # exec_px may be None when next bar not yet in px; keep stop/hi.
            out.append(
                {
                    "signal": s,
                    "reclaim_d": reclaim_d,
                    "exec_d": pd.Timestamp(exec_d).normalize() if exec_d is not None else None,
                    "exec_px": exec_px,
                    "stop": stop,
                    "hi": hi,
                    "red_d": red_d,
                    "phase": phase,
                }
            )
            continue
        exec_d = pd.Timestamp(exec_d).normalize()
        assert exec_px is not None
        ref = float(last_px) if last_px is not None else float(exec_px)
        phase = {
            "接近执行点": "near_exec",
            "反弹失败": "failed",
        }.get(late_entry_state(ref, float(exec_px), stop), "expired")
        out.append(
            {
                "signal": s,
                "reclaim_d": reclaim_d,
                "exec_d": exec_d,
                "exec_px": float(exec_px),
                "stop": stop,
                "hi": hi,
                "red_d": red_d,
                "phase": phase,
            }
        )
    return out


def scan_one_code(
    *,
    code: str,
    name: str,
    oos: pd.DataFrame,
    as_of: pd.Timestamp,
    trade_date: pd.Timestamp,
    cfg: RuleConfig,
    plot_mod: Any,
    live: pd.DataFrame | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build decision rows that guide ``trade_date`` using info known at ``as_of`` close."""
    rows: list[dict[str, Any]] = []
    today_rows: list[dict[str, Any]] = []
    zone_rows: list[dict[str, Any]] = []
    trade_s = str(trade_date.date())
    asof_s = str(as_of.date())

    sub = oos[oos["code"].astype(str).str.upper() == code].copy()
    if sub.empty:
        return rows, today_rows, zone_rows

    start = str(pd.Timestamp(sub["as_of"].min()).date())
    end = str(as_of.date())
    ohlcv = plot_mod.load_qlib_ohlcv(
        code,
        start,
        end,
        init_qlib=False,
        lookback_calendar_days=220,
        clip_to_window=False,
    )
    ohlcv = plot_mod.attach_sma_columns(ohlcv)
    lr = _live_row(live, code)
    if lr is not None:
        ohlcv = _inject_live_bar(ohlcv, as_of=as_of, live_row=lr, plot_mod=plot_mod)

    close = ohlcv.set_index(pd.to_datetime(ohlcv["datetime"]).dt.normalize())["$close"]
    close = close[~close.index.duplicated(keep="last")]
    vol = compute_volatility_regime_frame(close)
    px = build_price_frame(ohlcv, vol)
    # Keep full lookback series for MA20 trend; only clip the switch/price join window.
    px_trend = px[px["as_of"] <= as_of].copy()
    px = px[(px["as_of"] >= pd.Timestamp(start)) & (px["as_of"] <= as_of)].copy()

    sw = sub[sub["quantile_switch"].fillna(False).astype(bool)].copy()
    sw["as_of"] = pd.to_datetime(sw["as_of"]).dt.normalize()
    sw = sw[sw["as_of"] <= as_of]
    if sw.empty or px.empty:
        return rows, today_rows, zone_rows

    merge_cols = [
        c
        for c in [
            "as_of",
            "$close",
            "$high",
            "$low",
            "MA5",
            "MA20",
            "rv5",
            "rv20",
            "vol5_pct_120d",
            "vol_regime",
        ]
        if c in px.columns
    ]
    sw = sw.merge(px[merge_cols], on="as_of", how="left")
    ann = annotate_switches(sw, px, cfg)
    zones = detect_alternating_zones(ann, cfg)

    last_px = float(px.iloc[-1]["$close"])
    last_d = str(pd.Timestamp(px.iloc[-1]["as_of"]).date())

    sell_from = as_of - pd.Timedelta(days=cfg.sell_confirm_days + 2)
    buy_from = as_of - pd.Timedelta(days=cfg.buy_reclaim_days + 2)
    zone_from = as_of - pd.Timedelta(days=cfg.alt_window + 5)

    close_s = px_trend.set_index("as_of")["$close"]
    close_s = close_s[~close_s.index.duplicated(keep="last")].sort_index()
    trend = ma20_trend_state(close_s, as_of)
    trend_intact = bool(trend.get("trend_intact"))
    # Hysteresis: hugging MA20 within ±2% is NOT a confirmed trend (e.g.
    # SH517120 07/29 was +0.08% above MA20 and flipped sell→up in one day).
    TREND_HUG_BAND_PCT = 2.0
    _dist_raw = trend.get("dist_ma20_pct")
    trend_dist = (
        float(_dist_raw) if _dist_raw is not None and not pd.isna(_dist_raw) else 0.0
    )
    trend_hugging = abs(trend_dist) < TREND_HUG_BAND_PCT
    trend_confirmed_up = trend_intact and not trend_hugging
    trend_confirmed_down = (not trend_intact) and not trend_hugging
    trend_since = trend.get("trend_since")
    trend_above = trend.get("above_ma20")
    if trend_hugging:
        trend_note = f"MA20缠绕（{trend_dist:+.1f}%，±2%内不作趋势确认）"
        trend_label_main = "MA20缠绕"
    else:
        trend_note = (
            f"自{trend_since}起收盘仍站MA20（趋势未破）"
            if trend_intact and trend_since
            else (
                f"已破MA20"
                + (
                    f"（本轮跌破自@{trend.get('last_break')}）"
                    if trend.get("last_break")
                    else ""
                )
            )
        )
        trend_label_main = (
            f"未破@{trend_since}" if trend_intact and trend_since else "已破MA20"
        )

    for _, s in ann[ann["as_of"] == as_of.normalize()].iterrows():
        side_label = "绿" if s["side"] == "up" else "红"
        today_rows.append(
            {
                "code": code,
                "name": name,
                "kind": "今日三角",
                "signal_date": str(as_of.date()),
                "side": side_label,
                "action": s["action"],
                "strength": s["strength"],
                "signal_px": float(s["close"]),
                "signal_high": float(s["high"]),
                "last_px": last_px,
                "vol_pct": s["vol5_pct_120d"],
                "prior10": s["prior10"],
                "above_ma20": s["above_ma20"],
                "rv_expand": s["rv_expand"],
                "pred_cdf": s.get("pred_cdf"),
                # Trend context so today's gated 绿三角观察 lands in the right
                # five-class bucket (hugging → None → range).
                "trend_intact": trend_intact,
                "trend_since": trend_since,
                "trend_label": trend_label_main,
                "trend_hugging": trend_hugging,
            }
        )

    for _, s in ann[(ann["action"] == "卖预警") & (ann["as_of"] >= sell_from)].iterrows():
        # Gated: 强/极端波动 sells are noise (inception backtest test-seg win 36-38%).
        if not sell_passes_gate(s["strength"]):
            continue
        sim = simulate_sell(px, s, cfg)
        if sim["executed"] and pd.Timestamp(sim["exec_date"]) < as_of - pd.Timedelta(days=2):
            continue
        if not sim["executed"]:
            status = "待确认"
            op = (
                f"若持仓：{trade_s}盯盘跌破信号低点/MA5则卖；"
                f"未破不机械卖（信号价{_fmt_px(s['close'])}）"
            )
        elif pd.Timestamp(sim["exec_date"]).normalize() >= as_of - pd.Timedelta(days=2):
            status = "刚确认"
            op = (
                f"若持仓：卖确认已在{_fmt_px(sim['exec_px'])}触发（{sim['reason']}）；"
                f"{trade_s}复核是否已减仓，未减则减"
            )
        else:
            continue
        rows.append(
            {
                "code": code,
                "name": name,
                "priority": 2 if status == "刚确认" else 3,
                "decision": "卖预警",
                "status": status,
                "signal_date": str(pd.Timestamp(s["as_of"]).date()),
                "side": "绿",
                "strength": s["strength"],
                "signal_px": float(s["close"]),
                "last_px": last_px,
                "last_date": last_d,
                "vol_pct": s["vol5_pct_120d"],
                "operation": op,
                "detail": sim["reason"],
                "entry_state": "forward",
            }
        )

    for _, s in ann[(ann["action"] == "买观察") & (ann["as_of"] >= buy_from)].iterrows():
        # Gated: low-vol reclaim watches underperform (test-seg win 29%).
        if not buy_watch_passes_gate(s["vol5_pct_120d"]):
            continue
        sim = simulate_buy_watch(px, s, cfg)
        if sim["executed"] and pd.Timestamp(sim["exec_date"]) < as_of - pd.Timedelta(days=3):
            continue
        if not sim["executed"]:
            rows.append(
                {
                    "code": code,
                    "name": name,
                    "priority": 3,
                    "decision": "买观察",
                    "status": "待收复",
                    "signal_date": str(pd.Timestamp(s["as_of"]).date()),
                    "side": "红",
                    "strength": s["strength"],
                    "signal_px": float(s["close"]),
                    "signal_high": float(s["high"]),
                    "last_px": last_px,
                    "last_date": last_d,
                    "vol_pct": s["vol5_pct_120d"],
                    "prior10": s["prior10"],
                    "operation": (
                        f"{trade_s}盯盘；"
                        f"{reclaim_buy_trigger_note(trigger_hi=s['high'], ref_px=last_px)}；"
                        f"未达不加仓（现价{_fmt_px(last_px)}）"
                    ),
                    "detail": sim["reason"],
                    "entry_state": "forward",
                }
            )
        else:
            # Reclaim already happened on/before as_of — price-aware late entry:
            # near exec point → still joinable; run away → don't chase; below stop → failed.
            exec_px = float(sim["exec_px"])
            stop = sim["stop"]
            state = late_entry_state(last_px, exec_px, stop)
            late = last_px / exec_px - 1.0 if exec_px else 0.0
            if state == "接近执行点":
                rows.append(
                    {
                        "code": code,
                        "name": name,
                        "priority": 3,
                        "decision": "买观察",
                        "status": "接近执行点仍可参与",
                        "signal_date": str(pd.Timestamp(s["as_of"]).date()),
                        "side": "红",
                        "strength": s["strength"],
                        "signal_px": float(s["close"]),
                        "last_px": last_px,
                        "last_date": last_d,
                        "vol_pct": s["vol5_pct_120d"],
                        "exec_px": exec_px,
                        "stop": stop,
                        "operation": (
                            f"收复执行点@{_fmt_px(exec_px)}"
                            f"（@{pd.Timestamp(sim['exec_date']).date()}）；"
                            f"现价距执行点{late:+.1%}（参与带内）；"
                            f"{trade_s}可轻仓参与，止损{_fmt_px(stop)}；"
                            f"涨超执行点+2%不追，跌回止损离场"
                        ),
                        "detail": sim["reason"],
                        "entry_state": "forward",
                    }
                )
            elif state == "反弹失败":
                rows.append(
                    {
                        "code": code,
                        "name": name,
                        "priority": 5,
                        "decision": "买观察",
                        "status": "反弹失败",
                        "signal_date": str(pd.Timestamp(s["as_of"]).date()),
                        "side": "红",
                        "strength": s["strength"],
                        "signal_px": float(s["close"]),
                        "last_px": last_px,
                        "last_date": last_d,
                        "vol_pct": s["vol5_pct_120d"],
                        "operation": (
                            f"收复后已跌回止损{_fmt_px(stop)}下方；"
                            f"放弃参与，有仓按止损离场"
                        ),
                        "detail": sim["reason"],
                        "entry_state": "expired",
                    }
                )
            else:
                rows.append(
                    {
                        "code": code,
                        "name": name,
                        "priority": 4,
                        "decision": "买观察",
                        "status": "收复窗口已过",
                        "signal_date": str(pd.Timestamp(s["as_of"]).date()),
                        "side": "红",
                        "strength": s["strength"],
                        "signal_px": float(s["close"]),
                        "last_px": last_px,
                        "last_date": last_d,
                        "vol_pct": s["vol5_pct_120d"],
                        "operation": (
                            f"收复后执行点已过（曾@{_fmt_px(exec_px)}），"
                            f"现价已涨离执行点{late:+.1%}；"
                            f"不追买，等回踩执行点/MA5企稳再看；"
                            f"有仓则管止损{_fmt_px(stop)}"
                        ),
                        "detail": sim["reason"],
                        "entry_state": "expired",
                    }
                )

    # 软红（卖/看跌延续 / 忽略 / 中段观察）：收盘收复高点后转买观察。
    # 不套用买观察 vol 门控；深跌观察仍走收复观察位/筑底，不在此路径。
    for ev in plan_sell_continuation_reclaim_buys(
        ann, px, as_of=as_of, buy_from=buy_from, cfg=cfg, last_px=last_px
    ):
        s = ev["signal"]
        red_d = ev["red_d"]
        reclaim_d = ev["reclaim_d"]
        exec_d = ev["exec_d"]
        exec_px = ev["exec_px"]
        stop = ev["stop"]
        hi = ev["hi"]
        act = str(s.get("action") or "软红")
        late = (
            last_px / float(exec_px) - 1.0
            if exec_px is not None and float(exec_px) > 0
            else 0.0
        )
        exec_d_s = str(exec_d.date()) if exec_d is not None else trade_s
        base = {
            "code": code,
            "name": name,
            "decision": "买观察",
            "signal_date": str(red_d.date()),
            "side": "红",
            "strength": s.get("strength", ""),
            "signal_px": float(s["close"]),
            "signal_high": hi,
            "last_px": last_px,
            "last_date": last_d,
            "vol_pct": s.get("vol5_pct_120d"),
            "prior10": s.get("prior10"),
            "trend_intact": None if trend_hugging else trend_intact,
            "trend_hugging": trend_hugging,
            "trend_since": trend_since,
            "above_ma20": trend_above,
            "trend_label": trend_label_main,
        }
        if ev["phase"] == "pending_open":
            rows.append(
                {
                    **base,
                    "priority": 2,
                    "status": "待下一交易日开盘",
                    "operation": (
                        f"{act}红@{red_d.date()}高点{_fmt_px(hi)}"
                        f"已于{reclaim_d.date()}收盘收复；"
                        f"{trade_s}开盘试多（小仓），止损参考{_fmt_px(stop)}"
                    ),
                    "detail": f"{act}红高点收复→转买观察，待次日开盘",
                    "entry_state": "forward",
                    "invalidation": f"开盘跳空低开破{_fmt_px(stop)}则取消",
                }
            )
        elif ev["phase"] == "near_exec":
            rows.append(
                {
                    **base,
                    "priority": 3,
                    "status": "接近执行点仍可参与",
                    "exec_px": exec_px,
                    "stop": stop,
                    "operation": (
                        f"{act}红@{red_d.date()}高点{_fmt_px(hi)}"
                        f"已于{reclaim_d.date()}收复；"
                        f"执行点@{_fmt_px(exec_px)}（@{exec_d_s}）；"
                        f"现价距执行点{late:+.1%}（参与带内）；"
                        f"{trade_s}可轻仓参与，止损{_fmt_px(stop)}"
                    ),
                    "detail": f"{act}红高点收复→转买观察",
                    "entry_state": "forward",
                    "invalidation": f"跌回止损{_fmt_px(stop)}则离场",
                }
            )
        elif ev["phase"] == "failed":
            rows.append(
                {
                    **base,
                    "priority": 5,
                    "status": "反弹失败",
                    "operation": (
                        f"{act}红收复后已跌回止损{_fmt_px(stop)}下方；"
                        f"放弃参与，有仓按止损离场"
                    ),
                    "detail": f"{act}红收复后反弹失败",
                    "entry_state": "expired",
                }
            )
        else:
            rows.append(
                {
                    **base,
                    "priority": 4,
                    "status": "收复窗口已过",
                    "exec_px": exec_px,
                    "stop": stop,
                    "operation": (
                        f"{act}红@{red_d.date()}曾于{reclaim_d.date()}收复，"
                        f"执行点@{_fmt_px(exec_px)}（@{exec_d_s}）已过；"
                        f"现价已涨离{late:+.1%}；不追买，等回踩执行点/MA5企稳再看；"
                        f"有仓则管止损{_fmt_px(stop)}"
                    ),
                    "detail": f"{act}红收复买窗口已过",
                    "entry_state": "expired",
                }
            )

    # Deep reds blocked from 「买观察」: still publish unreclaimed high as 收复观察位
    # (watch only — not a buy signal; extreme-deep reclaim edge is marginal).
    buy_watch_dates = {
        pd.Timestamp(s["as_of"]).normalize()
        for _, s in ann[
            (ann["action"] == "买观察") & (ann["as_of"] >= buy_from)
        ].iterrows()
        if buy_watch_passes_gate(s["vol5_pct_120d"])
    }
    last_deep = pick_latest_deep_red_for_reclaim_watch(
        ann,
        px,
        as_of=as_of,
        buy_from=buy_from,
        buy_watch_dates=buy_watch_dates,
    )
    reclaimed_deep = pick_latest_deep_red_reclaimed(
        ann,
        px,
        as_of=as_of,
        buy_from=buy_from,
        buy_watch_dates=buy_watch_dates,
    )
    if last_deep is not None:
        hi = float(last_deep["high"])
        red_d = pd.Timestamp(last_deep["as_of"]).normalize()
        rows.append(
            {
                "code": code,
                "name": name,
                "priority": 4,
                "decision": "收复观察",
                "status": "待收复(观察位非买点)",
                "signal_date": str(red_d.date()),
                "side": "红",
                "strength": last_deep.get("strength", ""),
                "signal_px": float(last_deep["close"]),
                "signal_high": hi,
                "last_px": last_px,
                "last_date": last_d,
                "vol_pct": last_deep.get("vol5_pct_120d"),
                "prior10": last_deep.get("prior10"),
                "operation": (
                    f"{trade_s}盯盘；"
                    f"{reclaim_watch_note(trigger_hi=hi, ref_px=last_px, signal_date=red_d)}；"
                    f"深跌红未升买观察，仅公布观察位（现价{_fmt_px(last_px)}）"
                ),
                "detail": (
                    "深跌红三角未收复高点→收复观察位（非买点）"
                    + (
                        "；来源=极端跌幅兜底"
                        if str(last_deep.get("switch_source") or "")
                        == "extreme_drop_override"
                        else (
                            f"；来源={last_deep.get('switch_source')}"
                            if last_deep.get("switch_source")
                            else ""
                        )
                    )
                ),
                "entry_state": "forward",
                "invalidation": "盘中触及不加仓；未收盘站上不加仓",
                "trend_intact": None if trend_hugging else trend_intact,
                "trend_since": trend_since,
                "above_ma20": trend_above,
                "trend_hugging": trend_hugging,
                "trend_label": trend_label_main,
            }
        )

    for _, s in ann[(ann["action"] == "买确认候选") & (ann["as_of"] >= buy_from)].iterrows():
        sig_d = pd.Timestamp(s["as_of"]).normalize()
        nd, nop = _next_open(px, sig_d)
        exec_px: float | None = None
        stop: float | None = None
        # Forward only when the rule's "next open" is still in the future
        # relative to as_of close (typically: signal printed on as_of itself).
        if sig_d == as_of.normalize() or (
            nd is not None and pd.Timestamp(nd).normalize() > as_of.normalize()
        ):
            status = "待下一交易日开盘"
            entry_state = "forward"
            priority = 1
            op = (
                f"{trade_s}开盘试多（小仓）；"
                f"勿用已收盘的{asof_s}价下单；"
                f"信号日收盘{_fmt_px(s['close'])}，"
                f"{asof_s}参考价{_fmt_px(last_px)}"
            )
        else:
            # Next open already occurred on/before as_of — price-aware late entry.
            sim_c = simulate_buy_confirm(px, s)
            exec_px = (
                float(sim_c["exec_px"])
                if sim_c["executed"] and sim_c["exec_px"] is not None
                else (float(nop) if nop is not None else None)
            )
            stop = sim_c.get("stop")
            state = (
                late_entry_state(last_px, exec_px, stop)
                if exec_px is not None
                else "已远离执行点"
            )
            late = last_px / exec_px - 1.0 if exec_px else 0.0
            when = (
                f"{pd.Timestamp(nd).date()}开盘@{_fmt_px(exec_px)}"
                if nd is not None
                else "次日开盘"
            )
            if state == "接近执行点":
                status = "接近执行点仍可参与"
                entry_state = "forward"
                priority = 3
                op = (
                    f"买确认执行点（本应{when}）；"
                    f"现价距执行点{late:+.1%}（参与带内）；"
                    f"{trade_s}可轻仓参与，止损{_fmt_px(stop)}；"
                    f"涨超执行点+2%不追，跌回止损离场"
                )
            elif state == "反弹失败":
                status = "反弹失败"
                entry_state = "expired"
                priority = 5
                op = (
                    f"买确认后已跌回止损{_fmt_px(stop)}下方；"
                    f"放弃参与，有仓按止损离场"
                )
            else:
                status = "已过开盘窗口"
                entry_state = "expired"
                priority = 4
                op = (
                    f"试多窗口已过（本应{when}），现价已涨离执行点{late:+.1%}；"
                    f"{trade_s}不追买，等回踩执行点/MA5企稳再看；"
                    f"有仓则管止损{_fmt_px(stop)}"
                )
        rows.append(
            {
                "code": code,
                "name": name,
                "priority": priority,
                "decision": "买确认",
                "status": status,
                "signal_date": str(sig_d.date()),
                "side": "绿",
                "strength": s["strength"],
                "signal_px": float(s["close"]),
                "last_px": last_px,
                "last_date": last_d,
                "vol_pct": s["vol5_pct_120d"],
                "exec_px": exec_px,
                "stop": stop,
                "operation": op,
                "detail": "买确认候选",
                "entry_state": entry_state,
            }
        )

    if not zones.empty:
        for _, z in zones.iterrows():
            zend = pd.Timestamp(z["zone_end"])
            zstart = pd.Timestamp(z["zone_start"])
            active = zend >= zone_from or (
                zstart <= as_of and zend >= as_of - pd.Timedelta(days=cfg.alt_window)
            )
            if not active:
                continue
            zone_rows.append(
                {
                    "code": code,
                    "name": name,
                    **{
                        k: (str(v.date()) if hasattr(v, "date") else v)
                        for k, v in z.items()
                    },
                }
            )
            if z["kind"] == "底部交替区":
                morph = f"形态{z['pattern']}"
                vol_note = ""
                mv = z.get("max_vol_pct")
                if mv is not None and pd.notna(mv) and float(mv) >= cfg.vol_cluster:
                    vol_note = f"；波动聚集(vol%={float(mv):.0%})"
                # 近深跌红高点已收复 → 筑底（横盘），即使仍破 MA20 也不进趋势下跌。
                if reclaimed_deep is not None and trend_confirmed_down:
                    red_d = pd.Timestamp(reclaimed_deep["as_of"]).normalize()
                    hi = float(reclaimed_deep["high"])
                    op_body = (
                        f"{trade_s}筑底观察（{morph}{vol_note}）；"
                        f"{red_d.date()}深跌红高点{_fmt_px(hi)}已收复；"
                        f"虽仍破MA20，按筑底而非趋势下跌处理；"
                        f"等买观察/买确认或重新站上MA20后再评估，不机械抄底"
                    )
                    rows.append(
                        {
                            "code": code,
                            "name": name,
                            "priority": 5,
                            "decision": "筑底观察",
                            "status": "筑底（深跌红已收复）",
                            "signal_date": str(red_d.date()),
                            "side": z["pattern"],
                            "strength": "",
                            "signal_px": float(z["max_close"]),
                            "signal_high": hi,
                            "last_px": last_px,
                            "last_date": last_d,
                            "vol_pct": z.get("max_vol_pct"),
                            "prior10": reclaimed_deep.get("prior10"),
                            "operation": op_body,
                            "detail": (
                                f"近深跌红高点已收复→筑底；{z['action_hint']}；{trend_note}"
                            ),
                            "entry_state": "forward",
                            "invalidation": "再度跌破最近深跌红低点或放量破位则取消筑底解读",
                            "trend_intact": None if trend_hugging else trend_intact,
                            "trend_hugging": trend_hugging,
                            "trend_since": trend_since,
                            "above_ma20": trend_above,
                            "trend_label": trend_label_main,
                        }
                    )
                    continue
                # 确认跌破时底部交替只是结构备注，五类主桶走趋势下跌（见 ops_bucket）。
                if trend_confirmed_down:
                    op_body = (
                        f"{trade_s}趋势下跌中见底部交替（{morph}{vol_note}）；"
                        f"{trend_note}；交替不改下跌趋势，不机械抄底；"
                        f"等买观察/买确认或重新站上MA20后再评估"
                    )
                else:
                    op_body = (
                        f"{trade_s}底部盘整/变盘观察（{morph}{vol_note}）；"
                        f"价在MA20附近低位交替，非顶部清仓结构；"
                        f"{trend_note}；"
                        f"未出现买观察/买确认条件前不机械抄底，也不按顶部借反弹清"
                    )
                rows.append(
                    {
                        "code": code,
                        "name": name,
                        "priority": 5,
                        "decision": "底部交替观察",
                        "status": z["kind"],
                        "signal_date": f"{zstart.date()}~{zend.date()}",
                        "side": z["pattern"],
                        "strength": "",
                        "signal_px": float(z["max_close"]),
                        "last_px": last_px,
                        "last_date": last_d,
                        "vol_pct": z.get("max_vol_pct"),
                        "operation": op_body,
                        "detail": f"{z['action_hint']}；{trend_note}",
                        "entry_state": "forward",
                        "invalidation": "放量跌破区低或重新站上MA20后改判",
                        "trend_intact": trend_intact,
                        "trend_hugging": trend_hugging,
                        "trend_since": trend_since,
                        "above_ma20": trend_above,
                        "trend_label": trend_label_main,
                    }
                )
                continue

            if z["kind"] != "顶部交替区":
                continue
            legs = ann[
                (ann["as_of"] >= zstart)
                & (ann["as_of"] <= zend)
                & (ann["side"] == "down")
            ]
            zone_legs = ann[(ann["as_of"] >= zstart) & (ann["as_of"] <= zend)]
            # Safety: never treat all-below-MA20 alternating as top clear.
            if not zone_legs.empty and all(
                _is_below_ma20(v) for v in zone_legs["above_ma20"].tolist()
            ):
                continue
            # Zone pattern only (do not splice post-zone same-side reds into morph).
            morph = f"形态{z['pattern']}"
            superseded, supersede_why, last_red = top_zone_post_red_run(
                ann, zone_end=zend, as_of=as_of, min_reds=2
            )
            if superseded:
                rows.append(
                    {
                        "code": code,
                        "name": name,
                        "priority": 8,
                        "decision": "顶部区已失效",
                        "status": "顶部交替区已破坏",
                        "signal_date": f"{zstart.date()}~{zend.date()}",
                        "side": z["pattern"],
                        "strength": "",
                        "signal_px": float(z["max_close"]),
                        "last_px": last_px,
                        "last_date": last_d,
                        "vol_pct": z.get("max_vol_pct"),
                        "operation": (
                            f"原顶部区{zstart.date()}~{zend.date()}（{morph}）"
                            f"不再作为减仓主依据；{supersede_why}；"
                            f"{trend_note}；{trade_s}不按借反弹清执行"
                        ),
                        "detail": supersede_why,
                        "entry_state": "expired",
                        "invalidation": "结构已破坏",
                        "trend_intact": trend_intact,
                        "trend_since": trend_since,
                        "above_ma20": trend_above,
                        "trend_label": (
                            f"未破@{trend_since}"
                            if trend_intact and trend_since
                            else "已破MA20"
                        ),
                    }
                )
                # Post-top same-side reds below MA20: last red is a bounce watch
                # (reclaim high), even if generic prior10/rv gates blocked 买观察.
                # Gated: keep only 深跌 (prior10<-8%) or 翻红 (>0%); 中段 is a meat grinder.
                if last_red is not None and post_top_buy_passes_gate(last_red.get("prior10")):
                    red_d = pd.Timestamp(last_red["as_of"]).normalize()
                    if red_d >= buy_from:
                        hi = float(last_red["high"])
                        stop = float(last_red["low"])
                        post_px = px[
                            (px["as_of"] > red_d) & (px["as_of"] <= as_of)
                        ].sort_values("as_of")
                        reclaim_rows = post_px[post_px["$close"] > hi]
                        if reclaim_rows.empty:
                            rows.append(
                                {
                                    "code": code,
                                    "name": name,
                                    "priority": 3,
                                    "decision": "买观察",
                                    "status": "待收复",
                                    "signal_date": str(red_d.date()),
                                    "side": "红",
                                    "strength": last_red.get("strength", ""),
                                    "signal_px": float(last_red["close"]),
                                    "signal_high": hi,
                                    "last_px": last_px,
                                    "last_date": last_d,
                                    "vol_pct": last_red.get("vol5_pct_120d"),
                                    "prior10": last_red.get("prior10"),
                                    "operation": (
                                        f"顶部破坏后连红低点@{red_d.date()}；"
                                        f"{trade_s}盯盘；"
                                        f"{reclaim_buy_trigger_note(trigger_hi=hi, ref_px=last_px)}；"
                                        f"未达不加仓（现价{_fmt_px(last_px)}）"
                                    ),
                                    "detail": f"{supersede_why}；连红末脚作反弹观察",
                                    "entry_state": "forward",
                                    "invalidation": f"未收复{_fmt_px(hi)}不加仓",
                                    "trend_intact": trend_intact,
                                    "trend_since": trend_since,
                                    "above_ma20": trend_above,
                                    "trend_label": (
                                        f"未破@{trend_since}"
                                        if trend_intact and trend_since
                                        else "已破MA20"
                                    ),
                                }
                            )
                        else:
                            reclaim_d = pd.Timestamp(reclaim_rows.iloc[0]["as_of"]).normalize()
                            nd, nop = _next_open(px, reclaim_d)
                            # Next open still ahead of as_of (or missing in clipped px)
                            # → guide trade_date open entry.
                            if nd is None or pd.Timestamp(nd).normalize() > as_of.normalize():
                                rows.append(
                                    {
                                        "code": code,
                                        "name": name,
                                        "priority": 2,
                                        "decision": "买观察",
                                        "status": "待下一交易日开盘",
                                        "signal_date": str(red_d.date()),
                                        "side": "红",
                                        "strength": last_red.get("strength", ""),
                                        "signal_px": float(last_red["close"]),
                                        "signal_high": hi,
                                        "last_px": last_px,
                                        "last_date": last_d,
                                        "vol_pct": last_red.get("vol5_pct_120d"),
                                        "prior10": last_red.get("prior10"),
                                        "operation": (
                                            f"顶部破坏后连红低点@{red_d.date()}已收复高点"
                                            f"{_fmt_px(hi)}（@{reclaim_d.date()}）；"
                                            f"{trade_s}开盘试多（小仓），止损参考{_fmt_px(stop)}"
                                        ),
                                        "detail": "收复红三角高点，待次日开盘",
                                        "entry_state": "forward",
                                        "invalidation": f"开盘跳空低开破{_fmt_px(stop)}则取消",
                                        "trend_intact": trend_intact,
                                        "trend_since": trend_since,
                                        "above_ma20": trend_above,
                                        "trend_label": (
                                            f"未破@{trend_since}"
                                            if trend_intact and trend_since
                                            else "已破MA20"
                                        ),
                                    }
                                )
                            elif pd.Timestamp(nd).normalize() >= as_of - pd.Timedelta(
                                days=3
                            ):
                                exec_px = float(nop)
                                state = late_entry_state(last_px, exec_px, stop)
                                late = last_px / exec_px - 1.0 if exec_px else 0.0
                                if state == "接近执行点":
                                    rows.append(
                                        {
                                            "code": code,
                                            "name": name,
                                            "priority": 3,
                                            "decision": "买观察",
                                            "status": "接近执行点仍可参与",
                                            "signal_date": str(red_d.date()),
                                            "side": "红",
                                            "strength": last_red.get("strength", ""),
                                            "signal_px": float(last_red["close"]),
                                            "signal_high": hi,
                                            "last_px": last_px,
                                            "last_date": last_d,
                                            "vol_pct": last_red.get("vol5_pct_120d"),
                                            "prior10": last_red.get("prior10"),
                                            "exec_px": exec_px,
                                            "stop": stop,
                                            "operation": (
                                                f"顶部破坏后连红低点@{red_d.date()}收复执行点"
                                                f"@{_fmt_px(exec_px)}（@{pd.Timestamp(nd).date()}）；"
                                                f"现价距执行点{late:+.1%}（参与带内）；"
                                                f"{trade_s}可轻仓参与，止损{_fmt_px(stop)}；"
                                                f"涨超执行点+2%不追，跌回止损离场"
                                            ),
                                            "detail": "收复红三角高点后次日开盘",
                                            "entry_state": "forward",
                                            "trend_intact": trend_intact,
                                            "trend_since": trend_since,
                                            "above_ma20": trend_above,
                                            "trend_label": (
                                                f"未破@{trend_since}"
                                                if trend_intact and trend_since
                                                else "已破MA20"
                                            ),
                                        }
                                    )
                                elif state == "反弹失败":
                                    rows.append(
                                        {
                                            "code": code,
                                            "name": name,
                                            "priority": 5,
                                            "decision": "买观察",
                                            "status": "反弹失败",
                                            "signal_date": str(red_d.date()),
                                            "side": "红",
                                            "strength": last_red.get("strength", ""),
                                            "signal_px": float(last_red["close"]),
                                            "last_px": last_px,
                                            "last_date": last_d,
                                            "vol_pct": last_red.get("vol5_pct_120d"),
                                            "prior10": last_red.get("prior10"),
                                            "operation": (
                                                f"顶部破坏后连红低点@{red_d.date()}收复后"
                                                f"已跌回止损{_fmt_px(stop)}下方；"
                                                f"放弃参与，有仓按止损离场"
                                            ),
                                            "detail": "收复红三角高点后次日开盘",
                                            "entry_state": "expired",
                                            "trend_intact": trend_intact,
                                            "trend_since": trend_since,
                                            "above_ma20": trend_above,
                                            "trend_label": (
                                                f"未破@{trend_since}"
                                                if trend_intact and trend_since
                                                else "已破MA20"
                                            ),
                                        }
                                    )
                                else:
                                    rows.append(
                                        {
                                            "code": code,
                                            "name": name,
                                            "priority": 4,
                                            "decision": "买观察",
                                            "status": "收复窗口已过",
                                            "signal_date": str(red_d.date()),
                                            "side": "红",
                                            "strength": last_red.get("strength", ""),
                                            "signal_px": float(last_red["close"]),
                                            "signal_high": hi,
                                            "last_px": last_px,
                                            "last_date": last_d,
                                            "vol_pct": last_red.get("vol5_pct_120d"),
                                            "prior10": last_red.get("prior10"),
                                            "operation": (
                                                f"顶部破坏后连红低点@{red_d.date()}曾触发收复"
                                                f"（执行点@{_fmt_px(exec_px)}@{pd.Timestamp(nd).date()}），"
                                                f"现价已涨离执行点{late:+.1%}；"
                                                f"不追买，等回踩执行点/MA5企稳再看；"
                                                f"有仓则管止损{_fmt_px(stop)}"
                                            ),
                                            "detail": "收复红三角高点后次日开盘",
                                            "entry_state": "expired",
                                            "trend_intact": trend_intact,
                                            "trend_since": trend_since,
                                            "above_ma20": trend_above,
                                            "trend_label": (
                                                f"未破@{trend_since}"
                                                if trend_intact and trend_since
                                                else "已破MA20"
                                            ),
                                        }
                                    )
                continue

            if trend_confirmed_up:
                # Confirmed MA20 uptrend: GRG/RGR above MA20 is mid-trend noise, not a top.
                decision = "趋势内交替观察"
                pri = 6
                invalidation = "跌破MA20则退出上行中继解读"
                op = (
                    f"{trade_s}趋势上行中的交替（{morph}），非顶部结构；"
                    f"{trend_note}，持有/可逢低；"
                    f"跌破MA20后再评估减仓，不要按顶部借反弹清"
                )
                detail = f"趋势内交替；{trend_note}"
            else:
                if legs.empty:
                    exit_note = f"若持仓：{trade_s}绿后先减一半，等第一红主清"
                    pri = 2
                else:
                    exit_d = pd.Timestamp(legs.iloc[0]["as_of"])
                    exit_px = float(legs.iloc[0]["close"])
                    if exit_d >= as_of - pd.Timedelta(days=5):
                        exit_note = (
                            f"若持仓：{trade_s}主减仓/清仓仍有效"
                            f"（第一红@{exit_d.date()} {_fmt_px(exit_px)}）"
                        )
                        pri = 1
                    else:
                        exit_note = (
                            f"若持仓：{trade_s}借反弹清（第一红@{exit_d.date()}）"
                        )
                        pri = 2
                decision = "顶部交替减仓"
                invalidation = "跌破后反弹不加仓"
                op = f"{exit_note}；{morph}；{trend_note}；{trade_s}不加仓"
                detail = f"{z['action_hint']}；{trend_note}"

            rows.append(
                {
                    "code": code,
                    "name": name,
                    "priority": pri,
                    "decision": decision,
                    "status": (
                        "趋势交替区" if decision == "趋势内交替观察" else z["kind"]
                    ),
                    "signal_date": f"{zstart.date()}~{zend.date()}",
                    "side": z["pattern"],
                    "strength": "",
                    "signal_px": float(z["max_close"]),
                    "last_px": last_px,
                    "last_date": last_d,
                    "vol_pct": z.get("max_vol_pct"),
                    "operation": op,
                    "detail": detail,
                    "entry_state": "forward",
                    "invalidation": invalidation,
                    "trend_intact": trend_intact,
                    "trend_since": trend_since,
                    "above_ma20": trend_above,
                    "trend_label": trend_label_main,
                }
            )

    for _, s in ann[
        (ann["action"] == "高位回砸观察")
        & (ann["as_of"] >= as_of - pd.Timedelta(days=5))
    ].iterrows():
        rows.append(
            {
                "code": code,
                "name": name,
                "priority": 4,
                "decision": "高位回砸观察",
                "status": "仅诊断",
                "signal_date": str(pd.Timestamp(s["as_of"]).date()),
                "side": "红",
                "strength": s["strength"],
                "signal_px": float(s["close"]),
                "last_px": last_px,
                "last_date": last_d,
                "vol_pct": s["vol5_pct_120d"],
                "operation": f"若持仓：{trade_s}高位红三角不抄底；若在交替区内配合减仓",
                "detail": "高位回砸观察",
                "entry_state": "forward",
            }
        )

    if not rows:
        # No triangle-driven decision at all: classify by MA20 trend so every
        # scanned code lands in one of the five classes (附录B keeps failures only).
        dist_v = trend_dist
        if trend_confirmed_up:
            decision = "趋势上涨-无三角"
            op = (
                f"无红/绿三角信号；自{trend_since}起站MA20（{dist_v:+.1f}%）；"
                f"{trade_s}持有/可逢低，跌破MA20再评估减仓"
            )
            status = "无三角趋势跟踪"
            detail = "无三角信号，仅按MA20趋势归类"
            sig_d = asof_s
            sig_hi = None
            prior = None
        elif trend_confirmed_down and reclaimed_deep is not None:
            # 近深跌红已收复：按筑底，不进裸「趋势下跌-无三角」。
            red_d = pd.Timestamp(reclaimed_deep["as_of"]).normalize()
            hi = float(reclaimed_deep["high"])
            decision = "筑底观察"
            op = (
                f"{trade_s}筑底观察：{red_d.date()}深跌红高点{_fmt_px(hi)}已收复；"
                f"虽仍破MA20（{dist_v:+.1f}%），按筑底而非趋势下跌处理；"
                f"等买观察/买确认或重新站上MA20后再评估，不机械抄底"
            )
            status = "筑底（深跌红已收复）"
            detail = "近深跌红高点已收复→筑底（非趋势下跌）"
            sig_d = str(red_d.date())
            sig_hi = hi
            prior = reclaimed_deep.get("prior10")
        elif trend_confirmed_down:
            decision = "趋势下跌-无三角"
            op = (
                f"无红/绿三角信号；已破MA20"
                + (f"（本轮跌破自@{trend.get('last_break')}）" if trend.get("last_break") else "")
                + f"（{dist_v:+.1f}%）；{trade_s}不抄底不加仓，等结构好转"
            )
            status = "无三角趋势跟踪"
            detail = "无三角信号，仅按MA20趋势归类"
            sig_d = asof_s
            sig_hi = None
            prior = None
        else:
            decision = "横盘-无三角"
            op = (
                f"无红/绿三角信号；现价贴近MA20（{dist_v:+.1f}%），无明确趋势；"
                f"{trade_s}观望，等趋势选择方向或三角信号"
            )
            status = "无三角趋势跟踪"
            detail = "无三角信号，仅按MA20趋势归类"
            sig_d = asof_s
            sig_hi = None
            prior = None
        row_out: dict[str, Any] = {
            "code": code,
            "name": name,
            "priority": 9,
            "decision": decision,
            "status": status,
            "signal_date": sig_d,
            "side": "",
            "strength": "",
            "signal_px": last_px,
            "last_px": last_px,
            "last_date": last_d,
            "vol_pct": None,
            "operation": op,
            "detail": detail,
            "entry_state": "forward",
            "trend_intact": trend_intact,
            "trend_since": trend_since,
            "above_ma20": trend_above,
            "trend_label": trend_label_main,
        }
        if sig_hi is not None:
            row_out["signal_high"] = sig_hi
        if prior is not None:
            row_out["prior10"] = prior
        rows.append(row_out)

    return rows, today_rows, zone_rows


def _vol_s(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    try:
        return f"{float(v):.0%}"
    except (TypeError, ValueError):
        return ""


def _prior_s(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    try:
        return f"{float(v):+.1%}"
    except (TypeError, ValueError):
        return ""


def _cdf_s(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return ""


def _rv_s(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return "是" if bool(v) else "否"


def _section_rows(frame: pd.DataFrame, section: str) -> pd.DataFrame:
    if frame.empty or "section" not in frame.columns:
        return frame.iloc[0:0].copy()
    if "is_main" in frame.columns:
        out = frame[(frame["section"] == section) & (frame["is_main"])]
    else:
        out = frame[frame["section"] == section]
    return out.sort_values(["decision", "code"]) if not out.empty else out


def build_scanned_no_action_rows(
    *,
    main_df: pd.DataFrame,
    all_cand: pd.DataFrame,
    today: pd.DataFrame,
    scanned_codes: list[str] | None = None,
    name_map: dict[str, str] | None = None,
    failed: list[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Codes scanned but not in A–D actionable main decisions, with reasons."""
    name_map = name_map or {}
    failed = failed or []
    failed_map = {c: msg for c, msg in failed}
    scanned_codes = list(scanned_codes or [])

    actionable_sections = {
        "immediate_risk",
        "pending_confirm",
        "conditional",
        "position_mgmt",
    }
    # Codes already placed in any of the five OPS buckets (via main table).
    if not main_df.empty and "decision" in main_df.columns:
        work = main_df
        if "is_main" in work.columns:
            work = work[work["is_main"]]
        active_codes = set()
        for _, r in work.iterrows():
            bucket = ops_bucket(
                str(r.get("decision") or ""),
                trend_intact=None
                if r.get("trend_intact") is None
                or (
                    isinstance(r.get("trend_intact"), float)
                    and pd.isna(r.get("trend_intact"))
                )
                else bool(r.get("trend_intact")),
            )
            if bucket in {
                OPS_BUCKET_BUY,
                OPS_BUCKET_SELL,
                OPS_BUCKET_DOWN,
                OPS_BUCKET_UP,
                OPS_BUCKET_RANGE,
            }:
                active_codes.add(str(r["code"]))
    elif not main_df.empty and "section" in main_df.columns:
        mask = main_df["section"].isin(actionable_sections)
        if "is_main" in main_df.columns:
            mask = mask & main_df["is_main"]
        active_codes = set(main_df.loc[mask, "code"].astype(str).tolist())
    else:
        active_codes = set()

    today_by_code: dict[str, pd.Series] = {}
    if not today.empty:
        for _, r in today.iterrows():
            today_by_code[str(r["code"])] = r

    dec_codes = (
        set(all_cand["code"].astype(str))
        if not all_cand.empty and "code" in all_cand.columns
        else set()
    )
    hist_codes = set()
    if not all_cand.empty and "source" in all_cand.columns:
        hist_codes = set(
            all_cand.loc[all_cand["source"] == "history", "code"].astype(str)
        )

    quiet_rows: list[dict[str, str]] = []
    universe = scanned_codes or sorted(set(today_by_code) | dec_codes | active_codes)
    for code in universe:
        if code in active_codes:
            continue
        name = str(name_map.get(code, ""))
        if code in failed_map:
            quiet_rows.append(
                {"code": code, "name": name, "reason": f"扫描失败：{failed_map[code]}"}
            )
            continue
        tr = today_by_code.get(code)
        if tr is not None:
            action = str(tr.get("action") or "")
            side = str(tr.get("side") or "")
            nm = name or str(tr.get("name") or "")
            if action == "忽略":
                reason = (
                    f"今日{side}三角，规则=忽略（波动/幅度未进交易门槛）；无交易指令"
                )
            else:
                reason = (
                    f"今日{side}三角 action={action}，未进入 A–D 主操作（见 E 审计）"
                )
            quiet_rows.append({"code": code, "name": nm, "reason": reason})
            continue
        if code in hist_codes or code in dec_codes:
            quiet_rows.append(
                {
                    "code": code,
                    "name": name,
                    "reason": "近窗有历史候选但均被更高优先级覆盖或归入非主操作；本报告无独立动作",
                }
            )
            continue
        quiet_rows.append(
            {
                "code": code,
                "name": name,
                "reason": (
                    "已扫无动作：当日无红/绿三角，"
                    "近窗亦无活跃卖预警/买观察/买确认/顶部交替区"
                ),
            }
        )

    no_act = _section_rows(main_df, "no_action")
    listed = {r["code"] for r in quiet_rows}
    if not no_act.empty:
        for _, r in no_act.iterrows():
            code = str(r["code"])
            if code in listed:
                continue
            quiet_rows.append(
                {
                    "code": code,
                    "name": str(r.get("name") or name_map.get(code, "")),
                    "reason": str(
                        r.get("decision_reason") or r.get("operation") or "无动作"
                    ),
                }
            )
    return sorted(quiet_rows, key=lambda x: x["code"])


def build_ops_markdown(
    *,
    as_of: pd.Timestamp,
    trade_date: pd.Timestamp,
    html_dir: Path,
    val: Path,
    n_codes: int,
    live_captured: str | None,
    main_df: pd.DataFrame,
    all_cand: pd.DataFrame,
    today: pd.DataFrame,
    intraday: bool = False,
    scanned_codes: list[str] | None = None,
    name_map: dict[str, str] | None = None,
    failed: list[tuple[str, str]] | None = None,
) -> str:
    asof_s = as_of.strftime("%Y-%m-%d")
    trade_s = trade_date.strftime("%Y-%m-%d")
    src = f"`{val}`"
    if live_captured:
        src += f"（快照 {live_captured}）"
    px_col = "快照现价" if intraday else "收盘价(参考)"
    if intraday:
        title = f"# 盘中五类状态清单（{asof_s} provisional）"
        cutoff = f"- 信号截止：**{asof_s}** 盘中 provisional 价（含 live_snapshot）"
        guide = f"- 指导时段：**{trade_s} 当日剩余交易**（非下一交易日）"
        px_note = "- 表中价格统一为**快照现价**（禁止当作收盘价下单）"
        buy_note = "- 买入：盘中仅预警，须收盘确认后下一交易日再执行；禁止盘中追买"
    else:
        title = f"# 下一交易日五类状态清单（指导 {trade_s}）"
        cutoff = f"- 信号截止（收盘已发生）：**{asof_s}**"
        guide = f"- 指导交易日：**{trade_s}**（不回溯 {asof_s} 当日开盘）"
        px_note = f"- 表中「收盘价」= {asof_s} 收盘，仅供参考，**不是** {trade_s} 下单价"
        buy_note = (
            "- 买入：涨至触发价（收复红三角高点）后，**次日开盘**试多；未达不加仓"
        )

    lines = [
        title,
        "",
        "## 数据口径",
        "",
        cutoff,
        guide,
        f"- 扫描池：`{html_dir}`（{n_codes}只）",
        f"- 信号源：{src}",
        px_note,
        buy_note,
        "- 卖出：统一按「若持仓」理解；无持仓文件时仅作风控提示",
        "- 五类（回测整理）：①潜在买入 ②潜在卖出 ③趋势下跌 ④趋势上涨 ⑤横盘震荡",
        "- **收复观察位**（深跌红未升买观察时另列）：公布红三角高点与距现价涨幅；"
        "**非买点**，盘中触及不下手，须收盘站上后次日开盘再评估",
        "- 趋势：因果 MA20；确认站上=上涨，确认跌破=下跌；"
        "底部交替/绿三角在确认趋势下跟趋势归类，仅缠绕带内才归横盘/底部变盘",
        "- **Gated 口径**（成立以来全池回测，2024+测试段验证：胜率41.4%→55.2%、edge−0.66%→+2.24%）：",
        "  卖预警只保留≤中强度（强/极端波动为噪声）；买观察只保留vol_pct≥0.80；"
        "顶区失效收复买只保留深跌(prior10<−8%)或翻红(>0%)",
        "- **趋势确认滞后带**：距MA20在±2%以内为均线缠绕，不作趋势确认"
        "（顶部交替区维持减仓解读，避免一日卖一日买的反复横跳）；"
        "无三角信号的标的按MA20趋势归类进五类",
        "- 每个代码仅出现在一类（收复观察位表与五类互斥）",
        "",
    ]

    mains = (
        main_df[main_df["is_main"]].copy()
        if not main_df.empty and "is_main" in main_df.columns
        else (main_df.copy() if not main_df.empty else pd.DataFrame())
    )
    if not mains.empty:
        buckets: list[str] = []
        for _, r in mains.iterrows():
            ti = r.get("trend_intact")
            if isinstance(ti, float) and pd.isna(ti):
                ti = None
            elif ti is not None:
                ti = bool(ti)
            buckets.append(ops_bucket(str(r.get("decision") or ""), trend_intact=ti))
        mains = mains.assign()
        mains["ops_bucket"] = buckets

    def _cell(v: Any) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        return str(v)

    def _is_reclaim_watch_row(r: pd.Series) -> bool:
        d = str(r.get("decision") or "")
        if d == "收复观察":
            return True
        if d == "红三角不抄底":
            trig = r.get("trigger_price")
            return trig is not None and not (isinstance(trig, float) and pd.isna(trig))
        return False

    def _need_to_watch(last: Any, trig: Any) -> str:
        """现价到收复观察位的预计涨跌幅（trig/last - 1）。"""
        try:
            last_f = float(last)
            trig_f = float(trig)
        except (TypeError, ValueError):
            return ""
        if not np.isfinite(last_f) or not np.isfinite(trig_f) or last_f <= 0:
            return ""
        need = trig_f / last_f - 1.0
        if abs(need) < 1e-9:
            return "已达"
        if need < 0:
            return f"已超过({need:+.1%})"
        return f"{need:+.1%}"

    # 收复观察位单独成章，与五类互斥。
    if not mains.empty:
        reclaim_mask = mains.apply(_is_reclaim_watch_row, axis=1)
        reclaim_df = mains.loc[reclaim_mask].sort_values("code")
        mains_rest = mains.loc[~reclaim_mask].copy()
    else:
        reclaim_df = mains.iloc[0:0]
        mains_rest = mains

    def _bucket_df_rest(bucket: str) -> pd.DataFrame:
        if mains_rest.empty or "ops_bucket" not in mains_rest.columns:
            return mains_rest.iloc[0:0]
        return mains_rest[mains_rest["ops_bucket"] == bucket].sort_values("code")

    lines += [
        f"## 1. {OPS_BUCKET_LABEL[OPS_BUCKET_BUY]}及执行方式（{trade_s}）",
        "",
    ]
    buy_df = _bucket_df_rest(OPS_BUCKET_BUY)
    if buy_df.empty:
        lines += ["（无）", ""]
    else:
        lines.append(
            f"| 代码 | 名称 | {px_col} | 触发价 | 需涨幅/已达 | 执行方式 |"
        )
        lines.append("|---|---|---:|---:|---|---|")
        for _, r in buy_df.iterrows():
            trig_s, need_s, exec_s = buy_exec_summary(r)
            lines.append(
                f"| {r.get('code','')} | {r.get('name','')} | {_fmt_px(r.get('last_px'))} | "
                f"{trig_s} | {need_s} | {exec_s} |"
            )
        lines.append("")

    lines += [
        f"## 2. {OPS_BUCKET_LABEL[OPS_BUCKET_SELL]}及执行方式（若持仓；{trade_s}）",
        "",
    ]
    sell_df = _bucket_df_rest(OPS_BUCKET_SELL)
    if sell_df.empty:
        lines += ["（无）", ""]
    else:
        lines.append(f"| 代码 | 名称 | {px_col} | 信号依据 | 执行方式 |")
        lines.append("|---|---|---:|---|---|")
        for _, r in sell_df.iterrows():
            basis = str(r.get("decision") or "")
            lines.append(
                f"| {r.get('code','')} | {r.get('name','')} | {_fmt_px(r.get('last_px'))} | "
                f"{basis} | {sell_exec_summary(r)} |"
            )
        lines.append("")

    lines += [
        f"## 3. 收复观察位（非买点；{trade_s}）",
        "",
        "深跌红三角未升「买观察」时公布红日高点观察位。"
        "**盘中触及不下手**；须收盘站上后，次日开盘再评估。"
        "「距观察位」= 观察位/现价 − 1（正数=还需上涨）。",
        "",
    ]
    if reclaim_df.empty:
        lines += ["（无）", ""]
    else:
        lines.append(
            f"| 代码 | 名称 | {px_col} | 观察位 | 距观察位 | prior10 | 信号日 | MA20 | 说明 |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---|---|---|")
        for _, r in reclaim_df.iterrows():
            trig = r.get("trigger_price")
            need_s = _need_to_watch(r.get("last_px"), trig)
            sig_d = _cell(r.get("signal_date"))
            tl = r.get("trend_label") or ""
            prior_s = _prior_s(r.get("prior10"))
            note = (
                "观察位非买点；须收盘站上后次日开盘再评估，盘中触及不下手；不机械买入"
            )
            lines.append(
                f"| {r.get('code','')} | {r.get('name','')} | {_fmt_px(r.get('last_px'))} | "
                f"{_fmt_px(trig)} | {need_s} | {prior_s} | {sig_d} | {_cell(tl)} | {note} |"
            )
        lines.append("")

    lines += [f"## 4. {OPS_BUCKET_LABEL[OPS_BUCKET_DOWN]}", ""]
    down_df = _bucket_df_rest(OPS_BUCKET_DOWN)
    if down_df.empty:
        lines += ["（无）", ""]
    else:
        lines.append(f"| 代码 | 名称 | {px_col} | MA20 | 说明 |")
        lines.append("|---|---|---:|---|---|")
        for _, r in down_df.iterrows():
            note = "下跌趋势；不抄底、不加仓"
            d = str(r.get("decision") or "")
            if d == "红三角不抄底":
                note = "深回撤/条件不全，不进买入；等结构好转"
            elif d == "顶部区已失效":
                note = "原顶部结构已破坏；不按借反弹清；关注是否转出买入触发"
            elif d == "趋势下跌-无三角":
                note = "无三角信号；下跌趋势延续，不抄底、不加仓；等结构好转"
            elif d == "底部交替观察":
                note = "已确认跌破MA20；底部交替不改下跌趋势，不抄底、不加仓；等结构好转"
            elif d == "绿三角观察":
                note = "绿三角观察但趋势已破；不抄底、不加仓；等结构好转"
            tl = r.get("trend_label") or "已破MA20"
            lines.append(
                f"| {r.get('code','')} | {r.get('name','')} | {_fmt_px(r.get('last_px'))} | "
                f"{_cell(tl)} | {note} |"
            )
        lines.append("")

    lines += [f"## 5. {OPS_BUCKET_LABEL[OPS_BUCKET_UP]}", ""]
    up_df = _bucket_df_rest(OPS_BUCKET_UP)
    if up_df.empty:
        lines += ["（无）", ""]
    else:
        lines.append(f"| 代码 | 名称 | {px_col} | 站上MA20自 | 说明 |")
        lines.append("|---|---|---:|---|---|")
        for _, r in up_df.iterrows():
            since = r.get("trend_since") or ""
            if not since:
                tl = str(r.get("trend_label") or "")
                since = tl.replace("未破@", "") if tl.startswith("未破@") else tl
            note = "上行中继；持有/可逢低；跌破MA20再评估减仓"
            d = str(r.get("decision") or "")
            if d == "趋势上涨-无三角":
                note = "无三角信号；站上MA20，持有/可逢低；跌破MA20再评估减仓"
            elif d == "底部交替观察":
                note = "已确认站上MA20；原底部交替转为上行，持有/可逢低；跌破MA20再评估减仓"
            elif d == "绿三角观察":
                note = "绿三角观察且趋势未破；持有/可逢低；跌破MA20再评估减仓"
            elif d == "顶部区已失效":
                note = "原顶部结构已破坏；不按借反弹清；趋势已重新站上MA20，关注是否转出买入触发"
            lines.append(
                f"| {r.get('code','')} | {r.get('name','')} | {_fmt_px(r.get('last_px'))} | "
                f"{_cell(since)} | {note} |"
            )
        lines.append("")

    lines += [f"## 6. {OPS_BUCKET_LABEL[OPS_BUCKET_RANGE]}", ""]
    range_df = _bucket_df_rest(OPS_BUCKET_RANGE)
    if range_df.empty:
        lines += ["（无）", ""]
    else:
        lines.append(f"| 代码 | 名称 | {px_col} | 信号依据 | 说明 |")
        lines.append("|---|---|---:|---|---|")
        for _, r in range_df.iterrows():
            basis = str(r.get("decision") or "")
            note = "横盘/底部变盘观察；不机械抄底，也不按顶部清仓"
            if basis == "筑底观察":
                note = (
                    "近深跌红高点已收复，按筑底而非趋势下跌；"
                    "等买观察/买确认或重新站上MA20，不机械抄底"
                )
            elif "波动" in str(r.get("operation") or ""):
                note = "底部交替/波动聚集变盘；等明确买入触发或趋势选择方向"
            lines.append(
                f"| {r.get('code','')} | {r.get('name','')} | {_fmt_px(r.get('last_px'))} | "
                f"{basis} | {note} |"
            )
        lines.append("")

    lines += [f"## 附录A. {asof_s} 今日三角（审计）", ""]
    if today.empty:
        lines += ["（无今日三角）", ""]
    else:
        lines.append("| 代码 | 名称 | 颜色 | action | prior10 | 归入五类 |")
        lines.append("|---|---|---|---|---:|---|")
        main_by = {}
        if not mains.empty:
            for _, r in mains.iterrows():
                main_by[str(r["code"])] = r
        for _, r in today.sort_values(["side", "code"]).iterrows():
            code = str(r["code"])
            mr = main_by.get(code)
            bucket = ""
            if mr is not None:
                bucket = OPS_BUCKET_LABEL.get(str(mr.get("ops_bucket") or ""), "")
            lines.append(
                f"| {code} | {r.get('name', '')} | {r.get('side', '')} | {r.get('action', '')} | "
                f"{_prior_s(r.get('prior10'))} | {bucket} |"
            )
        lines.append("")

    lines += ["## 附录B. 已扫无动作 / 未入主表", ""]
    lines.append("扫描池中未进入收复观察位表或上述五类主表的标的。")
    lines.append("")
    quiet_rows = build_scanned_no_action_rows(
        main_df=main_df,
        all_cand=all_cand,
        today=today,
        scanned_codes=scanned_codes,
        name_map=name_map,
        failed=failed,
    )
    if not quiet_rows:
        lines += ["（无）", ""]
    else:
        lines.append("| 代码 | 名称 | 原因 |")
        lines.append("|---|---|---|")
        for r in quiet_rows:
            lines.append(f"| {r['code']} | {r.get('name', '')} | {r.get('reason', '')} |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def collect_candidates(
    *,
    today: pd.DataFrame,
    dec: pd.DataFrame,
    trade_s: str,
    asof_s: str,
    intraday: bool,
) -> list[Candidate]:
    """Normalize today triangles + historical active states into candidates."""
    cands: list[Candidate] = []
    if not today.empty:
        for _, r in today.iterrows():
            c = candidates_from_today_triangle(r, trade_s=trade_s, intraday=intraday)
            if c is not None:
                cands.append(c)
    if not dec.empty:
        for _, r in dec.iterrows():
            c = candidates_from_history_row(
                r, trade_s=trade_s, asof_s=asof_s, intraday=intraday
            )
            if c is not None:
                cands.append(c)
    return resolve_main_per_code(cands)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pack-dir",
        type=Path,
        default=None,
        help=(
            "intraday/EOD pack root: sets --validation-dir here and "
            "--html-dir to <pack-dir>/html when that folder exists"
        ),
    )
    p.add_argument(
        "--html-dir",
        type=Path,
        default=None,
        help="HTML pool dir used to resolve codes (default: plotly_outputs/<asof>all or pack/html)",
    )
    p.add_argument(
        "--validation-dir",
        type=Path,
        default=None,
        help="signals_oos + reversal labels dir (formal or intraday pack)",
    )
    p.add_argument(
        "--as-of",
        default=None,
        help="signal cutoff date YYYY-MM-DD (default: max as_of in signals_oos)",
    )
    p.add_argument(
        "--trade-date",
        default=None,
        help="session to guide YYYY-MM-DD (default: next day EOD / same day intraday)",
    )
    p.add_argument(
        "--intraday",
        action="store_true",
        help="intraday mode: guide same-day session; auto if live_snapshot.csv in pack",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: <validation-dir>/triangle_decision_scan_<asof>",
    )
    p.add_argument("--code", action="append", default=None, help="optional code filter")
    p.add_argument("--sell-min-strength", default="中", choices=["弱", "中", "强"])
    p.add_argument("--require-rv-expand-for-sell", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args(argv)


def resolve_paths(args: argparse.Namespace, as_of: pd.Timestamp) -> tuple[Path, Path, bool]:
    """Return (validation_dir, html_dir, intraday_mode)."""
    pack = _resolve(args.pack_dir) if args.pack_dir else None
    val = _resolve(args.validation_dir) if args.validation_dir else None
    if pack is not None:
        val = pack
    if val is None:
        val = _resolve(
            _PROJECT_ROOT
            / "runtime/decision_packs/20260720/regime_transition_validation_q90_to0720"
        )

    html = _resolve(args.html_dir) if args.html_dir else None
    if html is None and pack is not None and (pack / "html").is_dir():
        html = pack / "html"
    if html is None:
        tag = as_of.strftime("%Y%m%d")
        html = PLOTLY_OUTPUTS_DIR / f"{tag}all"

    intraday = bool(args.intraday)
    if not intraday and (val / "live_snapshot.csv").exists():
        intraday = "intraday" in val.as_posix()
    return val, html, intraday


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Peek as_of for default html dir naming before full load.
    peek_val = _resolve(args.pack_dir or args.validation_dir) if (
        args.pack_dir or args.validation_dir
    ) else _resolve(
        _PROJECT_ROOT
        / "runtime/decision_packs/20260720/regime_transition_validation_q90_to0720"
    )
    if not peek_val.exists():
        raise SystemExit(f"validation-dir not found: {peek_val}")

    oos_peek = pd.read_csv(peek_val / "signals_oos.csv", usecols=["as_of"])
    as_of = (
        pd.Timestamp(args.as_of).normalize()
        if args.as_of
        else pd.Timestamp(oos_peek["as_of"].max()).normalize()
    )

    val, html_dir, intraday = resolve_paths(args, as_of)
    if not val.exists():
        raise SystemExit(f"validation-dir not found: {val}")

    oos, _rev, _evt = load_validation_csvs(val)
    oos["as_of"] = pd.to_datetime(oos["as_of"]).dt.normalize()
    oos["code"] = oos["code"].astype(str).str.upper()

    as_of = (
        pd.Timestamp(args.as_of).normalize()
        if args.as_of
        else pd.Timestamp(oos["as_of"].max()).normalize()
    )
    out_dir = _resolve(args.out_dir) if args.out_dir else (
        val / f"triangle_decision_scan_{as_of.strftime('%Y%m%d')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.code:
        codes = [c.upper() for c in args.code]
    elif html_dir.exists():
        codes = codes_from_html_dir(html_dir)
    else:
        codes = sorted(oos["code"].unique().tolist())
    codes = [c for c in codes if c in set(oos["code"])]
    if not codes:
        raise SystemExit("no codes to scan")

    name_map = (
        oos[["code", "name"]]
        .drop_duplicates("code")
        .assign(code=lambda d: d["code"].astype(str).str.upper())
        .set_index("code")["name"]
        .to_dict()
        if "name" in oos.columns
        else {}
    )

    live = None
    live_captured = None
    live_path = val / "live_snapshot.csv"
    if live_path.exists():
        live = pd.read_csv(live_path)
        live["code"] = live["code"].astype(str).str.upper()
        if "captured_at" in live.columns and not live.empty:
            live_captured = str(live["captured_at"].iloc[0])
            if "T" in live_captured:
                live_captured = live_captured.split("T", 1)[1][:8]

    cfg = RuleConfig(
        sell_min_strength=args.sell_min_strength,
        require_rv_expand_for_sell=bool(args.require_rv_expand_for_sell),
    )
    plot_mod = _load_plot_mod()
    plot_mod._init_qlib(None)
    oos = enrich_oos_from_ohlcv_loader(oos, plot_mod.load_qlib_ohlcv)
    n_override = int(oos.get("extreme_drop_switch", pd.Series(dtype=bool)).fillna(False).sum())
    if n_override:
        print(f"[INFO] extreme_drop_override reds={n_override}")

    trade_date = (
        pd.Timestamp(args.trade_date).normalize()
        if args.trade_date
        else (as_of if intraday else next_trade_date(as_of))
    )

    mode = "INTRADAY" if intraday else "EOD"
    print(
        f"[INFO] mode={mode} scan codes={len(codes)} as_of={as_of.date()} "
        f"guide={trade_date.date()} val={val} html={html_dir}"
    )

    all_rows: list[dict[str, Any]] = []
    all_today: list[dict[str, Any]] = []
    all_zones: list[dict[str, Any]] = []
    failed: list[tuple[str, str]] = []

    for code in codes:
        try:
            rows, today_rows, zone_rows = scan_one_code(
                code=code,
                name=str(name_map.get(code, "")),
                oos=oos,
                as_of=as_of,
                trade_date=trade_date,
                cfg=cfg,
                plot_mod=plot_mod,
                live=live,
            )
            all_rows.extend(rows)
            all_today.extend(today_rows)
            all_zones.extend(zone_rows)
            print(f"[OK] {code} points={len(rows)} today={len(today_rows)}")
        except Exception as exc:  # noqa: BLE001
            failed.append((code, str(exc)))
            print(f"[FAIL] {code}: {exc}")
            if not args.continue_on_error:
                return 2

    dec = pd.DataFrame(all_rows)
    today = pd.DataFrame(all_today)
    zones = pd.DataFrame(all_zones)

    asof_s = str(as_of.date())
    trade_s = str(trade_date.date())
    if not dec.empty:
        dec = dec.sort_values(["priority", "code", "signal_date"]).drop_duplicates(
            ["code", "decision", "signal_date", "status"], keep="first"
        )

    resolved = collect_candidates(
        today=today,
        dec=dec,
        trade_s=trade_s,
        asof_s=asof_s,
        intraday=intraday,
    )
    all_cand = candidates_to_frame(resolved)
    best = main_table(resolved)
    if not best.empty:
        best = best.sort_values(["section", "decision", "code"])

    asof_tag = as_of.strftime("%Y%m%d")
    trade_tag = trade_date.strftime("%Y%m%d")
    dec.to_csv(out_dir / "all_decision_points.csv", index=False)
    if not all_cand.empty:
        all_cand.to_csv(out_dir / "all_candidates_resolved.csv", index=False)
    best.to_csv(out_dir / "ops_primary_fresh.csv", index=False)
    today.to_csv(out_dir / "today_triangles.csv", index=False)
    zones.to_csv(out_dir / "active_zones.csv", index=False)

    quiet = build_scanned_no_action_rows(
        main_df=best,
        all_cand=all_cand,
        today=today,
        scanned_codes=codes,
        name_map={str(k): str(v) for k, v in name_map.items()},
        failed=failed,
    )
    pd.DataFrame(quiet).to_csv(out_dir / "scanned_no_action.csv", index=False)

    md = build_ops_markdown(
        as_of=as_of,
        trade_date=trade_date,
        html_dir=html_dir,
        val=val,
        n_codes=len(codes),
        live_captured=live_captured,
        main_df=best,
        all_cand=all_cand,
        today=today,
        intraday=intraday,
        scanned_codes=codes,
        name_map={str(k): str(v) for k, v in name_map.items()},
        failed=failed,
    )
    ops_path = out_dir / f"OPS_{asof_tag}.md"
    if intraday:
        ops_for = out_dir / f"OPS_intraday_{asof_tag}.md"
        for stale in out_dir.glob("OPS_for_*_from_*.md"):
            stale.unlink()
    else:
        ops_for = out_dir / f"OPS_for_{trade_tag}_from_{asof_tag}.md"
    ops_path.write_text(md, encoding="utf-8")
    ops_for.write_text(md, encoding="utf-8")
    meta = {
        "mode": mode.lower(),
        "as_of_close": str(as_of.date()),
        "trade_date": str(trade_date.date()),
        "intraday": intraday,
        "n_codes": len(codes),
        "n_primary": int(len(best)),
        "validation_dir": str(val),
        "html_dir": str(html_dir),
    }
    (out_dir / "scan_meta.json").write_text(
        __import__("json").dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[OK] wrote {ops_path}")
    print(f"[OK] wrote {ops_for}")
    print(
        f"[OK] primary={len(best)} today={len(today)} points={len(dec)} "
        f"candidates={len(all_cand)} fail={len(failed)}"
    )
    if failed:
        print("[WARN] failures:", failed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
