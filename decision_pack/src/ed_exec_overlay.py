"""ED → execution overlay (diagnostic binding; does not mutate regime labels).

Modes
-----
off
    No change.
block_buy
    While flat, refuse enter-up on days with ED ``alert``.
block_buy_and_hypo_exit
    Also flatten when holding and ED ``alert`` fires (even if regime still ``up``).

Lookback (last-K)
-----------------
When ``lookback_k > 1``, an effective alert is true if **any** of the last K
bars (including today) was ``alert``. This keeps buy-blocks / forced exits
active for a few sessions after the soft_bear/peak8 point-check clears.

Production regime / METHOD / finalize stay untouched. Overlay only changes
hold_up *actions* (sim) or execution-plan row fields (ops).
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

EdOverlayMode = Literal["off", "block_buy", "block_buy_and_hypo_exit"]

ED_OVERLAY_MODES: tuple[str, ...] = ("off", "block_buy", "block_buy_and_hypo_exit")
DEFAULT_ED_LOOKBACK_K = 5


def normalize_ed_overlay_mode(value: Any) -> EdOverlayMode:
    text = str(value or "off").strip().lower()
    if text in {"0", "false", "no", "none", "off", ""}:
        return "off"
    if text in {"1", "true", "yes", "block_buy", "ed_block_buy"}:
        return "block_buy"
    if text in {
        "2",
        "block_buy_and_hypo_exit",
        "ed_block_buy_and_hypo_exit",
        "full",
    }:
        return "block_buy_and_hypo_exit"
    raise ValueError(f"unknown ed overlay mode: {value!r}")


def alert_mask_from_levels(levels: np.ndarray | pd.Series | list[str]) -> np.ndarray:
    arr = np.asarray(levels, dtype=object)
    return np.array([str(x) == "alert" for x in arr], dtype=bool)


def effective_alert_mask(
    raw_alert: np.ndarray | pd.Series | list[bool],
    *,
    lookback_k: int = 1,
) -> np.ndarray:
    """OR-reduce raw daily alerts over a trailing window of ``lookback_k`` bars.

    ``lookback_k<=1`` → identical to raw. ``lookback_k=5`` → alert if any of
    today and the previous 4 bars was alert.
    """
    raw = np.asarray(raw_alert, dtype=bool)
    k = int(lookback_k)
    if k <= 1 or len(raw) == 0:
        return raw.copy()
    kernel = np.ones(k, dtype=float)
    rolled = np.convolve(raw.astype(float), kernel, mode="full")[: len(raw)]
    return rolled > 0.0


def resolve_ed_trigger(
    levels: np.ndarray | pd.Series | list[str],
    *,
    lookback_k: int = 1,
) -> dict[str, Any]:
    """As-of ED trigger for execution rows from a level series ending at as_of."""
    arr = np.asarray(levels, dtype=object)
    if len(arr) == 0:
        return {
            "ed_level": "none",
            "ed_trigger": "none",
            "ed_trigger_source": "empty",
            "ed_lookback_k": int(lookback_k),
        }
    raw = str(arr[-1] or "none")
    raw_alert = alert_mask_from_levels(arr)
    eff = effective_alert_mask(raw_alert, lookback_k=lookback_k)
    if raw == "alert":
        source = "asof_alert"
        trigger = "alert"
    elif bool(eff[-1]):
        source = f"last_{int(lookback_k)}"
        trigger = "alert"
    else:
        source = "none"
        trigger = raw  # preserve watch/none for display
    return {
        "ed_level": raw,
        "ed_trigger": trigger if trigger == "alert" else raw,
        "ed_trigger_source": source,
        "ed_lookback_k": int(lookback_k),
    }


def simulate_hold_up_with_ed_overlay(
    labels: np.ndarray,
    close: np.ndarray,
    ed_alert: np.ndarray,
    *,
    mode: EdOverlayMode | str = "off",
    lookback_k: int = 1,
    dates: np.ndarray | None = None,
    cost_per_side: float = 0.001,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    """Enter-up / leave-up book with optional ED action overlay."""
    mode_n = normalize_ed_overlay_mode(mode)
    labs = np.asarray(labels, dtype=object)
    px = np.asarray(close, dtype=float)
    raw_alert = np.asarray(ed_alert, dtype=bool)
    alert = effective_alert_mask(raw_alert, lookback_k=lookback_k)
    if len(labs) != len(px) or len(alert) != len(px):
        raise ValueError("labels/close/ed_alert length mismatch")
    if dates is None:
        dates = np.arange(len(px)).astype(str)
    else:
        dates = np.asarray(dates).astype(str)

    if len(px) == 0:
        empty = pd.DataFrame(columns=["date", "equity", "drawdown", "exposed"])
        return {
            "n_closed": 0,
            "wins": 0,
            "win": float("nan"),
            "mean_ret_net": 0.0,
            "compound": 0.0,
            "bh": 0.0,
            "edge": 0.0,
            "max_drawdown": 0.0,
            "exposure": 0.0,
            "switches": 0,
            "open_pos": False,
            "n_blocked_entries": 0,
            "n_ed_forced_exits": 0,
            "n_alert_days": 0,
            "n_effective_alert_days": 0,
            "lookback_k": int(lookback_k),
        }, [], empty

    wealth = 1.0
    entry_effective = np.nan
    entry_px = np.nan
    entry_i = -1
    trades: list[dict[str, Any]] = []
    equities = np.ones(len(px), dtype=float)
    exposed = np.zeros(len(px), dtype=bool)
    n_blocked = 0
    n_forced = 0

    for i, lab in enumerate(labs):
        is_up = lab == "up"
        alert_i = bool(alert[i])
        finite = np.isfinite(px[i]) and px[i] > 0

        if entry_i < 0:
            if is_up and finite:
                if mode_n != "off" and alert_i:
                    n_blocked += 1
                else:
                    entry_i = i
                    entry_px = float(px[i])
                    entry_effective = entry_px * (1.0 + float(cost_per_side))
        else:
            leave = (not is_up) or (
                mode_n == "block_buy_and_hypo_exit" and alert_i
            )
            if leave and finite:
                if is_up and mode_n == "block_buy_and_hypo_exit" and alert_i:
                    n_forced += 1
                exit_effective = float(px[i]) * (1.0 - float(cost_per_side))
                net = exit_effective / entry_effective - 1.0
                wealth *= 1.0 + net
                trades.append(
                    {
                        "entry_date": str(dates[entry_i]),
                        "exit_date": str(dates[i]),
                        "entry_px": entry_px,
                        "exit_px": float(px[i]),
                        "ret_net": float(net),
                        "bars": int(i - entry_i),
                        "ed_forced_exit": bool(is_up and alert_i),
                    }
                )
                entry_i = -1
                entry_effective = entry_px = np.nan

        if entry_i >= 0:
            exposed[i] = True
            liquidation = float(px[i]) * (1.0 - float(cost_per_side))
            equities[i] = wealth * liquidation / entry_effective
        else:
            equities[i] = wealth

    peak = np.maximum.accumulate(equities)
    drawdown = equities / np.maximum(peak, 1e-12) - 1.0
    rets = np.asarray([t["ret_net"] for t in trades], dtype=float)
    bh = (
        float(px[-1] / px[0] - 1.0)
        if len(px) > 1 and np.isfinite(px[0]) and px[0] > 0 and np.isfinite(px[-1])
        else 0.0
    )
    metrics = {
        "n_closed": int(len(trades)),
        "wins": int(np.sum(rets > 0)),
        "win": float(np.mean(rets > 0)) if len(rets) else float("nan"),
        "mean_ret_net": float(np.mean(rets)) if len(rets) else 0.0,
        "compound": float(equities[-1] - 1.0),
        "bh": bh,
        "edge": float(equities[-1] - 1.0 - bh),
        "max_drawdown": float(np.min(drawdown)),
        "exposure": float(np.mean(exposed)),
        "switches": int(np.sum(labs[1:] != labs[:-1])) if len(labs) > 1 else 0,
        "open_pos": bool(entry_i >= 0),
        "n_blocked_entries": int(n_blocked),
        "n_ed_forced_exits": int(n_forced),
        "n_alert_days": int(raw_alert.sum()),
        "n_effective_alert_days": int(alert.sum()),
        "lookback_k": int(lookback_k),
    }
    equity = pd.DataFrame(
        {
            "date": dates,
            "equity": equities,
            "drawdown": drawdown,
            "exposed": exposed,
        }
    )
    return metrics, trades, equity


def apply_ed_overlay_to_execution_row(
    row: dict[str, Any],
    *,
    ed_level: str,
    mode: EdOverlayMode | str = "off",
    ed_trigger: str | None = None,
    ed_trigger_source: str | None = None,
    lookback_k: int = 1,
) -> dict[str, Any]:
    """Mutate a copy of an execution-plan row under ED overlay semantics.

    ``ed_level`` is the as-of point check. Overlay fires when
    ``ed_trigger`` (defaulting to ``ed_level``) equals ``alert`` — so last-K
    can raise a trigger while the point check is already ``none``.
    """
    mode_n = normalize_ed_overlay_mode(mode)
    out = dict(row)
    level = str(ed_level or "none")
    trigger = str(ed_trigger if ed_trigger is not None else level)
    source = str(ed_trigger_source or ("asof_alert" if trigger == "alert" else "none"))
    out["ed_level"] = level
    out["ed_trigger"] = trigger
    out["ed_trigger_source"] = source
    out["ed_lookback_k"] = int(lookback_k)
    out["ed_overlay_mode"] = mode_n
    if mode_n == "off" or trigger != "alert":
        out.setdefault("ed_overlay_applied", False)
        return out

    tag = "ED alert" if source == "asof_alert" else f"ED last-{int(lookback_k)}"
    notes: list[str] = []
    if bool(out.get("can_buy")) or str(out.get("buy_action") or "") in {
        "收盘确认买入",
        "等待上涨切换",
    }:
        out["can_buy"] = False
        out["buy_action"] = "ED禁买(模型滞后)"
        out["buy_trigger_px"] = None
        out["buy_trigger_pct"] = None
        notes.append(f"{tag} → 禁买")

    if mode_n == "block_buy_and_hypo_exit":
        regime = str(out.get("regime_asof") or "")
        if regime == "up" or str(out.get("hypo_sell_action") or "") == "继续持有":
            out["hypo_can_sell"] = True
            out["hypo_sell_action"] = "收盘确认卖出"
            prev = str(out.get("hypo_sell_note") or "")
            extra = f"{tag}：假设持有则收盘确认退出（regime 仍可能为 up）"
            out["hypo_sell_note"] = f"{prev}；{extra}" if prev else extra
            notes.append(f"{tag} → hypo 收盘确认卖出")
        if bool(out.get("live_holding")) and not bool(out.get("can_sell_live")):
            out["can_sell_live"] = True
            out["sell_live_action"] = "ED告警收盘卖出"
            out["sell_live_note"] = f"{tag} 且实盘持仓：收盘确认卖出（overlay）"
            notes.append(f"{tag} → 实盘可卖")

    out["ed_overlay_applied"] = True
    out["ed_overlay_note"] = "；".join(notes) if notes else f"{tag}（无动作字段可改）"
    try:
        from decision_pack.src.adaptive_execution_plan import classify_bucket

        out["bucket"] = classify_bucket(
            can_buy=bool(out.get("can_buy")),
            buy_action=str(out.get("buy_action") or ""),
            hypo_sell_action=str(out.get("hypo_sell_action") or ""),
            buy_trigger_px=out.get("buy_trigger_px"),
            hypo_sell_trigger_px=out.get("hypo_sell_trigger_px"),
        )
    except Exception:  # noqa: BLE001
        pass
    return out


__all__ = [
    "DEFAULT_ED_LOOKBACK_K",
    "ED_OVERLAY_MODES",
    "EdOverlayMode",
    "alert_mask_from_levels",
    "apply_ed_overlay_to_execution_row",
    "effective_alert_mask",
    "normalize_ed_overlay_mode",
    "resolve_ed_trigger",
    "simulate_hold_up_with_ed_overlay",
]
