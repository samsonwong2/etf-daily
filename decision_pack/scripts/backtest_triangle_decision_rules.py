#!/usr/bin/env python3
"""Backtest red/green triangle decision rules (incl. alternating top/bottom).

Important data constraint
-------------------------
``quantile_switch`` triangles come from walk-forward ``signals_oos.csv``.
In the current validation pack that series starts around 2024-06, so triangle
rules can only be backtested in the OOS window. Longer HTML charts (e.g. from
2018) can still show K-lines before the first signal, but have no triangles
there to trade.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from decision_pack.src.regime_transition_plot import (
    compute_volatility_regime_frame,
    load_validation_csvs,
)
from decision_pack.src.symbol_trend_board import load_cluster_mapping_codes
from runtime_paths import CLUSTER_MAPPING_SELECTED_TXT, PLOTLY_OUTPUTS_DIR


def _load_plot_mod():
    path = _PROJECT_ROOT / "decision_pack" / "scripts" / "plot_regime_transition_example.py"
    spec = importlib.util.spec_from_file_location("plot_regime_example_bt", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass(frozen=True)
class RuleConfig:
    prior_up: float = 0.05
    prior_down: float = -0.05
    prior_crash: float = -0.08
    # Deep crashes (prior10 below these floors) stay 观察/不抄底 — Apr–Jul 2026
    # showed 买观察/买确认 edges turn negative when the drawdown is too deep.
    buy_watch_floor: float = -0.08
    buy_confirm_floor: float = -0.10
    vol_cluster: float = 0.70
    vol_extreme: float = 0.90
    sell_min_strength: str = "中"  # 弱/中/强
    require_rv_expand_for_sell: bool = False  # False catches 06-29 style tops
    alt_window: int = 15
    buy_reclaim_days: int = 10
    sell_confirm_days: int = 10
    hold_eval_days: int = 20


def _strength(pct: float, *, cluster: float, extreme: float) -> str:
    if pct >= extreme:
        return "强"
    if pct >= 0.80:
        return "中"
    if pct >= cluster:
        return "弱"
    return ""


def _strength_rank(s: str) -> int:
    return {"弱": 1, "中": 2, "强": 3}.get(s, 0)


def build_price_frame(ohlcv: pd.DataFrame, vol_frame: pd.DataFrame) -> pd.DataFrame:
    px = ohlcv.copy()
    px["as_of"] = pd.to_datetime(px["datetime"]).dt.normalize()
    vf = vol_frame.copy()
    vf["as_of"] = pd.to_datetime(vf["as_of"]).dt.normalize()
    out = px.merge(
        vf[["as_of", "rv5", "rv20", "vol5_pct_120d", "vol_regime"]],
        on="as_of",
        how="left",
    )
    return out.sort_values("as_of").reset_index(drop=True)


def annotate_switches(
    switches: pd.DataFrame,
    px: pd.DataFrame,
    cfg: RuleConfig,
) -> pd.DataFrame:
    if switches.empty:
        return switches.copy()
    pxi = px.set_index("as_of")
    rows: list[dict[str, Any]] = []
    for _, r in switches.sort_values("as_of").iterrows():
        d = pd.Timestamp(r["as_of"]).normalize()
        if d not in pxi.index:
            continue
        loc = pxi.index.get_loc(d)
        if isinstance(loc, slice):
            loc = loc.start
        i0 = max(0, loc - 10)
        c0 = float(pxi.iloc[i0]["$close"])
        c1 = float(r.get("$close", pxi.iloc[loc]["$close"]))
        prior10 = (c1 / c0 - 1.0) if c0 else np.nan
        ma20 = pxi.iloc[loc].get("MA20")
        above = (
            None
            if ma20 is None or (isinstance(ma20, float) and pd.isna(ma20))
            else c1 > float(ma20)
        )
        rv5 = r.get("rv5", pxi.iloc[loc].get("rv5"))
        rv20 = r.get("rv20", pxi.iloc[loc].get("rv20"))
        pct = r.get("vol5_pct_120d", pxi.iloc[loc].get("vol5_pct_120d"))
        expand = (
            pd.notna(rv5)
            and pd.notna(rv20)
            and float(rv5) > float(rv20)
        )
        pct_v = float(pct) if pd.notna(pct) else np.nan
        side = str(r.get("switch_side") or "")
        action = "观察"
        strength = _strength(pct_v, cluster=cfg.vol_cluster, extreme=cfg.vol_extreme) if pd.notna(pct_v) else ""
        if side == "up":
            sell_ok = (
                prior10 > cfg.prior_up
                and pd.notna(pct_v)
                and pct_v >= cfg.vol_cluster
                and above is True
            )
            if cfg.require_rv_expand_for_sell:
                sell_ok = sell_ok and expand
            if sell_ok and _strength_rank(strength) >= _strength_rank(cfg.sell_min_strength):
                action = "卖预警"
            elif (
                prior10 < cfg.prior_crash
                and prior10 >= cfg.buy_confirm_floor
                and pd.notna(pct_v)
                and pct_v >= cfg.vol_cluster
            ):
                action = "买确认候选"
            elif pd.notna(pct_v) and pct_v < cfg.vol_cluster and prior10 > -0.03:
                action = "忽略"
        elif side == "down":
            if (
                prior10 < cfg.prior_down
                and prior10 >= cfg.buy_watch_floor
                and expand
                and pd.notna(pct_v)
                and pct_v >= cfg.vol_cluster
                and above is False
            ):
                action = "买观察"
            elif prior10 < -0.03 and pd.notna(pct_v) and pct_v < cfg.vol_cluster:
                action = "卖/看跌延续"
            elif above is True and pd.notna(pct_v) and pct_v >= cfg.vol_cluster:
                action = "高位回砸观察"
            elif prior10 > -0.03 and pd.notna(pct_v) and pct_v < cfg.vol_cluster:
                action = "忽略"
        rows.append(
            {
                "as_of": d,
                "code": str(r.get("code", "")).upper(),
                "side": side,
                "close": c1,
                "high": float(pxi.iloc[loc]["$high"]),
                "low": float(pxi.iloc[loc]["$low"]),
                "prior10": prior10,
                "above_ma20": above,
                "rv5": None if pd.isna(rv5) else float(rv5),
                "rv20": None if pd.isna(rv20) else float(rv20),
                "rv_expand": bool(expand),
                "vol5_pct_120d": None if pd.isna(pct_v) else pct_v,
                "vol_regime": pxi.iloc[loc].get("vol_regime"),
                "action": action,
                "strength": strength,
                "pred_cdf": None if pd.isna(r.get("pred_cdf")) else float(r.get("pred_cdf")),
                "switch_source": (
                    None
                    if r.get("switch_source") is None
                    or (
                        isinstance(r.get("switch_source"), float)
                        and pd.isna(r.get("switch_source"))
                    )
                    or str(r.get("switch_source") or "").strip() == ""
                    else str(r.get("switch_source"))
                ),
            }
        )
    return pd.DataFrame(rows)


def _ma20_flag(above_ma20: Any) -> bool | None:
    """Normalize above_ma20; avoid ``np.False_ is not False`` identity traps."""
    if above_ma20 is None or (isinstance(above_ma20, float) and pd.isna(above_ma20)):
        return None
    return bool(above_ma20)


def _is_below_ma20(above_ma20: Any) -> bool:
    return _ma20_flag(above_ma20) is False


def top_zone_superseded_by_red_run(
    annotated: pd.DataFrame,
    *,
    zone_end: pd.Timestamp,
    as_of: pd.Timestamp,
    min_reds: int = 2,
) -> tuple[bool, str]:
    """Invalidate a top alternating zone after same-side red continuation below MA20.

    Alternating-zone detection stops at same-color legs, so R…R after a top RGR/GRG
    never builds a new zone — but the old top zone must not keep driving
    「借反弹清 / 顶部交替减仓」once price has left the alternating top structure.
    """
    ok, why, _ = top_zone_post_red_run(
        annotated, zone_end=zone_end, as_of=as_of, min_reds=min_reds
    )
    return ok, why


def top_zone_post_red_run(
    annotated: pd.DataFrame,
    *,
    zone_end: pd.Timestamp,
    as_of: pd.Timestamp,
    min_reds: int = 2,
) -> tuple[bool, str, pd.Series | None]:
    """Return (superseded, why, last_red_row) for post-zone same-side reds below MA20."""
    zone_end = pd.Timestamp(zone_end).normalize()
    as_of = pd.Timestamp(as_of).normalize()
    if annotated.empty:
        return False, "", None
    post = annotated[
        (pd.to_datetime(annotated["as_of"]).dt.normalize() > zone_end)
        & (pd.to_datetime(annotated["as_of"]).dt.normalize() <= as_of)
    ].sort_values("as_of")
    if post.empty:
        return False, "", None

    run_dates: list[str] = []
    run_rows: list[pd.Series] = []
    last_hit: tuple[str, list[pd.Series]] | None = None
    for _, r in post.iterrows():
        side = str(r.get("side") or "")
        if side == "down" and _is_below_ma20(r.get("above_ma20")):
            run_dates.append(str(pd.Timestamp(r["as_of"]).date()))
            run_rows.append(r)
            if len(run_dates) >= min_reds:
                last_hit = ("/".join(run_dates), list(run_rows))
        else:
            run_dates = []
            run_rows = []
    if last_hit is None:
        return False, "", None
    dates_s, rows = last_hit
    return (
        True,
        f"区后同色连红{dates_s}且破MA20→顶部交替结构已破坏",
        rows[-1],
    )

def detect_alternating_zones(annotated: pd.DataFrame, cfg: RuleConfig) -> pd.DataFrame:
    """Detect green/red alternating clusters as top or bottom decision zones."""
    if annotated.empty:
        return annotated.iloc[0:0].copy()
    work = annotated.sort_values("as_of").reset_index(drop=True)
    zones: list[dict[str, Any]] = []
    i = 0
    while i < len(work) - 1:
        seq = [work.iloc[i]]
        j = i + 1
        while j < len(work):
            gap = (work.iloc[j]["as_of"] - seq[-1]["as_of"]).days
            if gap > cfg.alt_window:
                break
            if work.iloc[j]["side"] == seq[-1]["side"]:
                break
            seq.append(work.iloc[j])
            j += 1
            if len(seq) >= 4:
                break
        if len(seq) >= 3:
            sides = "".join("G" if r["side"] == "up" else "R" for r in seq)
            closes = [float(r["close"]) for r in seq]
            pcts = [
                r["vol5_pct_120d"]
                for r in seq
                if r["vol5_pct_120d"] is not None and pd.notna(r["vol5_pct_120d"])
            ]
            above_flags = [_ma20_flag(r["above_ma20"]) for r in seq]
            # All legs below MA20 → bottom consolidation / regime-change (never top clear).
            # Top only when at least one leg is explicitly above MA20 (peak/reclaim structure).
            # Avoid ``np.False_ is not False`` identity traps via _ma20_flag.
            if sides in {"RGR", "RGRG", "GRG", "GRGR"} and above_flags and all(
                a is False for a in above_flags
            ):
                kind = "底部交替区"
            elif sides in {"GRG", "GRGR", "RGR"} and all(
                a is True for a in above_flags[:2]
            ):
                kind = "顶部交替区"
            elif sides in {"GRG", "GRGR", "RGR"} and any(a is True for a in above_flags):
                kind = "顶部交替区"
            elif sides in {"RGR", "RGRG", "GRG"} and any(a is False for a in above_flags):
                kind = "底部交替区"
            else:
                kind = "交替观察区"
            max_pct = max(pcts) if pcts else np.nan
            zones.append(
                {
                    "zone_start": seq[0]["as_of"],
                    "zone_end": seq[-1]["as_of"],
                    "pattern": sides,
                    "kind": kind,
                    "n_legs": len(seq),
                    "first_close": closes[0],
                    "last_close": closes[-1],
                    "max_close": max(closes),
                    "min_close": min(closes),
                    "max_vol_pct": max_pct,
                    "action_hint": (
                        "减仓/清仓"
                        if kind == "顶部交替区"
                        else ("底部变盘观察" if kind == "底部交替区" else "观察")
                    ),
                }
            )
            i = j
        else:
            i += 1
    return pd.DataFrame(zones)


def _next_open(px: pd.DataFrame, d: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
    fut = px[px["as_of"] > d]
    if fut.empty:
        return None, None
    r = fut.iloc[0]
    return pd.Timestamp(r["as_of"]), float(r["$open"])


def simulate_sell(
    px: pd.DataFrame,
    signal: pd.Series,
    cfg: RuleConfig,
) -> dict[str, Any]:
    d = pd.Timestamp(signal["as_of"])
    close = float(signal["close"])
    sig_low = float(signal["low"])
    fut = px[px["as_of"] > d].head(cfg.sell_confirm_days)
    for _, r in fut.iterrows():
        if float(r["$close"]) < sig_low:
            return {
                "executed": True,
                "exec_date": pd.Timestamp(r["as_of"]),
                "exec_px": float(r["$close"]),
                "reason": "跌破信号日低点",
            }
        if float(r["$close"]) < float(r["MA5"]) and float(r["$close"]) < close * 0.98:
            return {
                "executed": True,
                "exec_date": pd.Timestamp(r["as_of"]),
                "exec_px": float(r["$close"]),
                "reason": "跌破MA5且弱于信号价2%",
            }
    if not fut.empty and float(fut["$high"].max()) > close * 1.05:
        return {"executed": False, "exec_date": None, "exec_px": None, "reason": "假信号续涨"}
    return {"executed": False, "exec_date": None, "exec_px": None, "reason": "未确认"}


def simulate_buy_watch(
    px: pd.DataFrame,
    signal: pd.Series,
    cfg: RuleConfig,
) -> dict[str, Any]:
    d = pd.Timestamp(signal["as_of"])
    sig_high = float(signal["high"])
    stop = float(signal["low"])
    fut = px[px["as_of"] > d].head(cfg.buy_reclaim_days)
    for _, r in fut.iterrows():
        if float(r["$close"]) > sig_high:
            ed, ep = _next_open(px, pd.Timestamp(r["as_of"]))
            if ed is None:
                break
            return {
                "executed": True,
                "exec_date": ed,
                "exec_px": ep,
                "stop": stop,
                "reason": "收复红三角高点后次日开盘",
            }
    return {
        "executed": False,
        "exec_date": None,
        "exec_px": None,
        "stop": stop,
        "reason": "未收复高点",
    }


def simulate_buy_confirm(px: pd.DataFrame, signal: pd.Series) -> dict[str, Any]:
    d = pd.Timestamp(signal["as_of"])
    ed, ep = _next_open(px, d)
    if ed is None:
        return {"executed": False, "exec_date": None, "exec_px": None, "stop": None, "reason": "无次日"}
    # Positional lookup: px may be a clipped view with non-contiguous index
    # (label-based iloc would slice the wrong rows and yield NaN stops).
    mask = (px["as_of"] <= d).to_numpy()
    if not mask.any():
        stop = float(signal["low"])
    else:
        i = int(np.flatnonzero(mask)[-1])
        stop = float(px["$low"].iloc[max(0, i - 5) : i + 1].min())
    return {
        "executed": True,
        "exec_date": ed,
        "exec_px": ep,
        "stop": stop,
        "reason": "绿三角次日开盘试多",
    }


def forward_stats(px: pd.DataFrame, d: pd.Timestamp, px0: float, days: int) -> dict[str, float]:
    fut = px[px["as_of"] > d].head(days)
    if fut.empty or not np.isfinite(px0) or px0 == 0:
        return {"ret": np.nan, "mfe": np.nan, "mae": np.nan}
    last = float(fut.iloc[-1]["$close"])
    hi = float(fut["$high"].max())
    lo = float(fut["$low"].min())
    return {
        "ret": last / px0 - 1.0,
        "mfe": hi / px0 - 1.0,
        "mae": lo / px0 - 1.0,
    }


def evaluate_code(
    code: str,
    oos: pd.DataFrame,
    plot_mod: Any,
    cfg: RuleConfig,
    *,
    end_date: str,
) -> dict[str, Any]:
    code_u = str(code).upper()
    sub = oos[oos["code"].astype(str).str.upper() == code_u].copy()
    if sub.empty:
        return {"code": code_u, "error": "no oos rows"}
    sig_start = pd.Timestamp(sub["as_of"].min()).strftime("%Y-%m-%d")
    # Load prices with warmup for vol percentile.
    ohlcv = plot_mod.load_qlib_ohlcv(
        code_u,
        sig_start,
        end_date,
        init_qlib=False,
        lookback_calendar_days=220,
        clip_to_window=False,
    )
    ohlcv = plot_mod.attach_sma_columns(ohlcv)
    close = ohlcv.set_index(pd.to_datetime(ohlcv["datetime"]).dt.normalize())["$close"]
    close = close[~close.index.duplicated(keep="last")]
    vol = compute_volatility_regime_frame(close)
    px = build_price_frame(ohlcv, vol)
    px = px[(px["as_of"] >= sig_start) & (px["as_of"] <= end_date)].copy()

    sw = sub[sub["quantile_switch"].fillna(False).astype(bool)].copy()
    sw["as_of"] = pd.to_datetime(sw["as_of"]).dt.normalize()
    sw = sw.merge(
        px[
            [
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
        ],
        on="as_of",
        how="left",
    )
    ann = annotate_switches(sw, px, cfg)
    zones = detect_alternating_zones(ann, cfg)

    events: list[dict[str, Any]] = []
    for _, s in ann.iterrows():
        action = s["action"]
        if action == "卖预警":
            sim = simulate_sell(px, s, cfg)
            ref = float(s["close"])
            base_d = pd.Timestamp(s["as_of"])
            if sim["executed"]:
                st = forward_stats(px, sim["exec_date"], sim["exec_px"], cfg.hold_eval_days)
                # For sells, positive "edge" if price fell after exit.
                edge = -st["ret"] if pd.notna(st["ret"]) else np.nan
            else:
                st = forward_stats(px, base_d, ref, cfg.hold_eval_days)
                edge = np.nan
            events.append(
                {
                    "code": code_u,
                    "signal_date": str(base_d.date()),
                    "side": "绿",
                    "action": action,
                    "strength": s["strength"],
                    "signal_px": ref,
                    "executed": bool(sim["executed"]),
                    "exec_date": None
                    if sim["exec_date"] is None
                    else str(pd.Timestamp(sim["exec_date"]).date()),
                    "exec_px": sim["exec_px"],
                    "reason": sim["reason"],
                    "fwd_ret": st["ret"],
                    "fwd_mfe": st["mfe"],
                    "fwd_mae": st["mae"],
                    "edge": edge,
                    "vol_pct": s["vol5_pct_120d"],
                    "prior10": s["prior10"],
                }
            )
        elif action == "买观察":
            sim = simulate_buy_watch(px, s, cfg)
            base_d = pd.Timestamp(s["as_of"])
            if sim["executed"]:
                # path until stop or hold_eval
                fut = px[px["as_of"] >= sim["exec_date"]].head(cfg.hold_eval_days)
                exit_px = float(fut.iloc[-1]["$close"]) if not fut.empty else sim["exec_px"]
                exit_why = "到期"
                for _, r in fut.iterrows():
                    if sim["stop"] is not None and float(r["$low"]) <= float(sim["stop"]):
                        exit_px = float(sim["stop"])
                        exit_why = "止损"
                        break
                ret = exit_px / float(sim["exec_px"]) - 1.0
                st = forward_stats(px, sim["exec_date"], float(sim["exec_px"]), cfg.hold_eval_days)
                events.append(
                    {
                        "code": code_u,
                        "signal_date": str(base_d.date()),
                        "side": "红",
                        "action": action,
                        "strength": s["strength"],
                        "signal_px": float(s["close"]),
                        "executed": True,
                        "exec_date": str(pd.Timestamp(sim["exec_date"]).date()),
                        "exec_px": sim["exec_px"],
                        "reason": f"{sim['reason']};{exit_why}",
                        "fwd_ret": ret,
                        "fwd_mfe": st["mfe"],
                        "fwd_mae": st["mae"],
                        "edge": ret,
                        "vol_pct": s["vol5_pct_120d"],
                        "prior10": s["prior10"],
                    }
                )
            else:
                st = forward_stats(px, base_d, float(s["close"]), cfg.hold_eval_days)
                events.append(
                    {
                        "code": code_u,
                        "signal_date": str(base_d.date()),
                        "side": "红",
                        "action": action,
                        "strength": s["strength"],
                        "signal_px": float(s["close"]),
                        "executed": False,
                        "exec_date": None,
                        "exec_px": None,
                        "reason": sim["reason"],
                        "fwd_ret": st["ret"],
                        "fwd_mfe": st["mfe"],
                        "fwd_mae": st["mae"],
                        "edge": np.nan,
                        "vol_pct": s["vol5_pct_120d"],
                        "prior10": s["prior10"],
                    }
                )
        elif action == "买确认候选":
            sim = simulate_buy_confirm(px, s)
            base_d = pd.Timestamp(s["as_of"])
            if not sim["executed"]:
                continue
            fut = px[px["as_of"] >= sim["exec_date"]].head(cfg.hold_eval_days)
            exit_px = float(fut.iloc[-1]["$close"]) if not fut.empty else sim["exec_px"]
            exit_why = "到期"
            for _, r in fut.iterrows():
                if sim["stop"] is not None and float(r["$low"]) <= float(sim["stop"]):
                    exit_px = float(sim["stop"])
                    exit_why = "止损"
                    break
            ret = exit_px / float(sim["exec_px"]) - 1.0
            st = forward_stats(px, sim["exec_date"], float(sim["exec_px"]), cfg.hold_eval_days)
            events.append(
                {
                    "code": code_u,
                    "signal_date": str(base_d.date()),
                    "side": "绿",
                    "action": action,
                    "strength": s["strength"],
                    "signal_px": float(s["close"]),
                    "executed": True,
                    "exec_date": str(pd.Timestamp(sim["exec_date"]).date()),
                    "exec_px": sim["exec_px"],
                    "reason": f"{sim['reason']};{exit_why}",
                    "fwd_ret": ret,
                    "fwd_mfe": st["mfe"],
                    "fwd_mae": st["mae"],
                    "edge": ret,
                    "vol_pct": s["vol5_pct_120d"],
                    "prior10": s["prior10"],
                }
            )
        elif action == "高位回砸观察":
            # Treat as soft sell risk: measure next 10d from signal close.
            base_d = pd.Timestamp(s["as_of"])
            st = forward_stats(px, base_d, float(s["close"]), 10)
            events.append(
                {
                    "code": code_u,
                    "signal_date": str(base_d.date()),
                    "side": "红",
                    "action": action,
                    "strength": s["strength"],
                    "signal_px": float(s["close"]),
                    "executed": False,
                    "exec_date": None,
                    "exec_px": None,
                    "reason": "高位红三角，仅诊断",
                    "fwd_ret": st["ret"],
                    "fwd_mfe": st["mfe"],
                    "fwd_mae": st["mae"],
                    "edge": -st["ret"] if pd.notna(st["ret"]) else np.nan,
                    "vol_pct": s["vol5_pct_120d"],
                    "prior10": s["prior10"],
                }
            )

    # Alternating zone evaluation: top zones -> sell at zone_end next confirm;
    # compare holding through zone vs exiting at first red after green.
    zone_evals: list[dict[str, Any]] = []
    for _, z in zones.iterrows():
        if z["kind"] != "顶部交替区":
            # still record
            st = forward_stats(px, pd.Timestamp(z["zone_end"]), float(z["last_close"]), 20)
            zone_evals.append(
                {
                    "code": code_u,
                    **{k: (str(v.date()) if hasattr(v, "date") else v) for k, v in z.items()},
                    "fwd20_ret": st["ret"],
                    "fwd20_mae": st["mae"],
                }
            )
            continue
        # Precise: exit on first red leg close in the zone (or zone end).
        legs = ann[
            (ann["as_of"] >= z["zone_start"])
            & (ann["as_of"] <= z["zone_end"])
            & (ann["side"] == "down")
        ]
        if legs.empty:
            exit_d = pd.Timestamp(z["zone_end"])
            exit_px = float(z["last_close"])
        else:
            exit_d = pd.Timestamp(legs.iloc[0]["as_of"])
            exit_px = float(legs.iloc[0]["close"])
        st = forward_stats(px, exit_d, exit_px, 20)
        zone_evals.append(
            {
                "code": code_u,
                **{k: (str(v.date()) if hasattr(v, "date") else v) for k, v in z.items()},
                "exit_date": str(exit_d.date()),
                "exit_px": exit_px,
                "fwd20_ret": st["ret"],
                "fwd20_mae": st["mae"],
                "avoided_drawdown": -st["mae"] if pd.notna(st["mae"]) else np.nan,
            }
        )

    bh0 = float(px.iloc[0]["$close"]) if not px.empty else np.nan
    bh1 = float(px.iloc[-1]["$close"]) if not px.empty else np.nan
    return {
        "code": code_u,
        "signal_start": sig_start,
        "signal_end": end_date,
        "n_switches": int(len(ann)),
        "n_events": int(len(events)),
        "n_zones": int(len(zones)),
        "buy_hold_ret": None if not np.isfinite(bh0) or bh0 == 0 else bh1 / bh0 - 1.0,
        "action_counts": ann["action"].value_counts().to_dict() if not ann.empty else {},
        "events": events,
        "zones": zone_evals,
        "annotated": ann,
    }


def summarize(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    rows = []
    for action, g in events.groupby("action"):
        ex = g[g["executed"] == True]  # noqa: E712
        rows.append(
            {
                "action": action,
                "n_signals": int(len(g)),
                "n_executed": int(len(ex)),
                "exec_rate": float(len(ex) / len(g)) if len(g) else np.nan,
                "mean_edge": float(ex["edge"].mean()) if len(ex) else np.nan,
                "median_edge": float(ex["edge"].median()) if len(ex) else np.nan,
                "win_rate": float((ex["edge"] > 0).mean()) if len(ex) else np.nan,
                "mean_fwd_ret_all": float(g["fwd_ret"].mean()),
                "mean_mae_all": float(g["fwd_mae"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("n_signals", ascending=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--validation-dir",
        type=Path,
        default=_PROJECT_ROOT
        / "workspace/decision_packs/20260720/regime_transition_validation_q90_to0720",
    )
    p.add_argument("--end-date", default="2026-07-24")
    p.add_argument("--code", action="append", default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--sell-min-strength",
        default="中",
        choices=["弱", "中", "强"],
        help="minimum strength to take sell warnings",
    )
    p.add_argument(
        "--require-rv-expand-for-sell",
        action="store_true",
        help="require rv5>rv20 for sell warnings (stricter; misses some tops)",
    )
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    val = args.validation_dir
    if not val.is_absolute():
        val = _PROJECT_ROOT / val
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = val / "triangle_decision_backtest"
    if not out_dir.is_absolute():
        out_dir = _PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    oos, _rev, _evt = load_validation_csvs(val)
    oos["as_of"] = pd.to_datetime(oos["as_of"]).dt.normalize()
    oos["code"] = oos["code"].astype(str).str.upper()

    if args.code:
        codes = [c.upper() for c in args.code]
    else:
        mapping = CLUSTER_MAPPING_SELECTED_TXT
        codes = [
            c
            for c in load_cluster_mapping_codes(mapping)
            if c in set(oos["code"])
        ]
        if not codes:
            codes = sorted(oos["code"].unique().tolist())

    cfg = RuleConfig(
        sell_min_strength=args.sell_min_strength,
        require_rv_expand_for_sell=bool(args.require_rv_expand_for_sell),
    )
    plot_mod = _load_plot_mod()
    plot_mod._init_qlib(None)

    all_events: list[pd.DataFrame] = []
    all_zones: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    failed: list[tuple[str, str]] = []

    print(
        f"[INFO] backtest codes={len(codes)} end={args.end_date} "
        f"sell_min={cfg.sell_min_strength} rv_expand={cfg.require_rv_expand_for_sell}"
    )
    print(
        f"[INFO] OOS triangle span global "
        f"{oos['as_of'].min().date()} -> {oos['as_of'].max().date()}"
    )

    for code in codes:
        try:
            res = evaluate_code(code, oos, plot_mod, cfg, end_date=args.end_date)
            if res.get("error"):
                raise ValueError(res["error"])
            summaries.append(
                {
                    "code": res["code"],
                    "signal_start": res["signal_start"],
                    "signal_end": res["signal_end"],
                    "n_switches": res["n_switches"],
                    "n_events": res["n_events"],
                    "n_zones": res["n_zones"],
                    "buy_hold_ret": res["buy_hold_ret"],
                    **{f"act_{k}": v for k, v in res["action_counts"].items()},
                }
            )
            if res["events"]:
                all_events.append(pd.DataFrame(res["events"]))
            if res["zones"]:
                all_zones.append(pd.DataFrame(res["zones"]))
            print(
                f"[OK] {code} switches={res['n_switches']} events={res['n_events']} "
                f"zones={res['n_zones']} bh={res['buy_hold_ret']}"
            )
        except Exception as exc:  # noqa: BLE001
            failed.append((code, str(exc)))
            print(f"[ERROR] {code}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                return 2

    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    zones_df = pd.concat(all_zones, ignore_index=True) if all_zones else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)
    by_action = summarize(events_df)

    events_df.to_csv(out_dir / "events.csv", index=False)
    zones_df.to_csv(out_dir / "alternating_zones.csv", index=False)
    summary_df.to_csv(out_dir / "per_code_summary.csv", index=False)
    by_action.to_csv(out_dir / "by_action_summary.csv", index=False)
    meta = {
        "validation_dir": str(val),
        "end_date": args.end_date,
        "n_codes": len(codes),
        "n_ok": len(summaries),
        "n_failed": len(failed),
        "failed": failed,
        "oos_start": str(oos["as_of"].min().date()),
        "oos_end": str(oos["as_of"].max().date()),
        "rule": asdict(cfg),
        "note": (
            "Triangles exist only inside signals_oos. "
            "HTML may start in 2018 for K-line context, but rule backtests "
            "cannot invent pre-OOS triangles."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Triangle decision-rule backtest",
        "",
        f"- OOS span: `{meta['oos_start']}` → `{meta['oos_end']}`",
        f"- codes ok/fail: {meta['n_ok']}/{meta['n_failed']}",
        f"- sell_min_strength={cfg.sell_min_strength}, "
        f"require_rv_expand_for_sell={cfg.require_rv_expand_for_sell}",
        "",
        "## By action",
        "",
    ]
    if not by_action.empty:
        lines.append(by_action.to_string(index=False))
    else:
        lines.append("(no events)")
    lines.extend(["", "## Note", "", meta["note"], ""])
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] wrote → {out_dir}")
    if not by_action.empty:
        print(by_action.to_string(index=False))
    return 0 if not failed or args.continue_on_error else 2


if __name__ == "__main__":
    raise SystemExit(main())
