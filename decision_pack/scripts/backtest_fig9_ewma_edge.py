#!/usr/bin/env python3
"""SH512700 fig9 EWMA L0+S0: parameter grid, walk-forward, overlay HTML.

Buy: rising-edge of EWMA near-lo (optional aux_buy / locmin).
Sell (default ``touch_leave``): this near-hi run must hard-touch EWMA
upper, then leave with ``dist_hi ≤ −leave_pct`` (not S0 first-kiss).
S0 (rising-edge near-hi) remains a grid baseline.
Fill: signal close → next-bar close. Long-only.

Tune on 2018-01-01..2024-12-31; validate 2025-01-01..2026-09-09.
2026-06-30 is reported after selection, never used to pick params.

Example::

    PYTHONPATH=. python decision_pack/scripts/backtest_fig9_ewma_edge.py
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
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

from decision_pack.src.fair_path_band_signals import (  # noqa: E402
    _extract_plotly_traces,
    find_regime_html,
    load_ohlcv_from_regime_html,
)
from decision_pack.src.fair_path_research import ewma_scaled_envelope  # noqa: E402
from decision_pack.src.plotly_html_autoscale import write_adaptive_html  # noqa: E402
from decision_pack.src.fair_path_research import ewma_scaled_envelope  # noqa: E402

DEFAULT_LISTING = (
    PLOTLY_OUTPUTS_DIR / "20260909_from_listing"
)
DEFAULT_HTML = DEFAULT_LISTING / (
    "regime_transition_SH512700_银行ETF南方_20170726_20260909_adaptive.html"
)
CODE = "SH512700"
TRAIN_END = pd.Timestamp("2024-12-31")
VALID_START = pd.Timestamp("2025-01-01")
EVAL_START = pd.Timestamp("2018-01-01")
DISCOVERY = pd.Timestamp("2026-06-30")
MIN_TRAIN_TRADES = 4
DEFAULT_LEAVE_PCT = 0.03
DEFAULT_LEAVE_MIN_NEAR = 1
GLIDE_NEAR_DAYS = 3
GLIDE_WINDOW = 10
GLIDE_DIST_MAX = 0.02
GLIDE_RET20_TH = -0.03

NEAR_GRID = (0.0, 0.005, 0.01, 0.015)
LAM_GRID = (0.94, 0.97)
BOOL_GRID = (True, False)
LOCMIN_GRID = (0, 10)


def _load_sm():
    path = _SCRIPTS / "run_reusable_extrema_state_machine.py"
    spec = importlib.util.spec_from_file_location("extrema_sm", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_touch():
    path = _SCRIPTS / "backtest_fig9_touch.py"
    spec = importlib.util.spec_from_file_location("fig9_touch", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_overlay():
    path = _SCRIPTS / "overlay_fig9_gap_rule_on_regime_html.py"
    spec = importlib.util.spec_from_file_location("fig9_overlay", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_bt():
    path = _SCRIPTS / "backtest_fair_path_band_signals.py"
    spec = importlib.util.spec_from_file_location("fair_path_bt", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stat(s: pd.Series) -> dict[str, Any]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    n = int(len(s))
    if n == 0:
        return {"n": 0, "win": None, "mean": None, "t": None, "compound": None}
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if n > 1 else float("nan")
    t = float(mean / (std / np.sqrt(n))) if n > 1 and std > 0 else None
    return {
        "n": n,
        "win": float((s > 0).mean()),
        "mean": mean,
        "t": t,
        "compound": float((1.0 + s).prod() - 1.0),
    }


def _fwd20(close: pd.Series, mask: pd.Series) -> pd.Series:
    entry = close.shift(-1)
    fut = close.shift(-21)
    return (fut / entry - 1.0)[mask.fillna(False)].dropna()


def apply_fig9_rails(
    frame: pd.DataFrame,
    *,
    lower: pd.Series,
    upper: pd.Series,
    near_pct: float,
    touch_mod: Any,
    mid_stretch: bool,
) -> pd.DataFrame:
    out = frame.copy()
    close = out["px"].astype(float)
    out["fig9_lower"] = pd.to_numeric(lower, errors="coerce")
    out["fig9_upper"] = pd.to_numeric(upper, errors="coerce")
    lo = out["fig9_lower"]
    up = out["fig9_upper"]
    out["fig9_touch_lo"] = (close <= lo).fillna(False)
    out["fig9_touch_hi"] = (close >= up).fillna(False)
    return touch_mod.prepare_fig9_touch_frame(
        out, near_pct=float(near_pct), mid_stretch=bool(mid_stretch)
    )


def build_candidates(
    frame: pd.DataFrame,
    *,
    touch_mod: Any,
    use_aux_buy: bool,
    use_aux_sell: bool,
    use_stretch: bool,
    locmin_days: int,
    holdup_gate: bool,
    regime_up: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    lo = frame["fig9_near_lo"].fillna(False).astype(bool)
    hi = frame["fig9_near_hi"].fillna(False).astype(bool)
    aux_b = frame["aux_buy"].fillna(False).astype(bool)
    aux_s = frame["aux_sell"].fillna(False).astype(bool)
    locmin = frame["locmin"].fillna(False).astype(bool)
    extra = (
        frame["fig9_mid_stretch_sell"].fillna(False).astype(bool)
        if use_stretch
        else pd.Series(False, index=frame.index)
    )
    buy = touch_mod.rising_edge(lo)
    sell = touch_mod.rising_edge(hi)
    if use_aux_buy:
        buy = buy & aux_b
    if locmin_days > 0:
        buy = buy & locmin
    if use_aux_sell:
        sell = sell & aux_s
    sell = sell | extra
    if holdup_gate:
        up = regime_up.reindex(frame.index).fillna(False).astype(bool)
        buy = buy & ~up
    return buy.fillna(False), sell.fillna(False)


def run_fail_c(
    frame: pd.DataFrame,
    buy_base: pd.Series,
    sell_ab: pd.Series,
    *,
    touch_mod: Any,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """S0/stretch plus fail C: left near-lo then rising-edge near-lo again.

    After a C-only sell, wait until near-hi before the next buy.
    """
    re_lo = touch_mod.rising_edge(frame["fig9_near_lo"].fillna(False).astype(bool))
    near_lo = frame["fig9_near_lo"].fillna(False).astype(bool)
    near_hi = frame["fig9_near_hi"].fillna(False).astype(bool)
    close = frame["px"].astype(float)
    idx = frame.index
    n = len(frame)
    buy_sig = np.zeros(n, dtype=bool)
    sell_sig = np.zeros(n, dtype=bool)
    trades: list[dict[str, Any]] = []
    in_pos = False
    armed = True
    left_lo = False
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
                left_lo = False
        else:
            if not bool(near_lo.iloc[i]):
                left_lo = True
            fail = left_lo and bool(re_lo.iloc[i])
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
                        "sell_kind": "失败C" if fail and not ab else "S0/拉伸",
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
                "sell_kind": "",
            }
        )
    return pd.DataFrame(trades), pd.Series(buy_sig, index=idx), pd.Series(sell_sig, index=idx)


def run_touch_leave_state_machine(
    frame: pd.DataFrame,
    buy_base: pd.Series,
    *,
    leave_pct: float = DEFAULT_LEAVE_PCT,
    min_near_days: int = DEFAULT_LEAVE_MIN_NEAR,
    require_hard_touch: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Sell only after this near-hi run hard-touches EWMA upper, then leaves.

    Leave day = falling edge of near_hi with ``dist_hi = close/upper − 1``
    at most ``-leave_pct``. Short kisses that never hard-touch, or leaves
    that stay within ``leave_pct`` of the rail, are ignored.
    """
    near_hi = frame["fig9_near_hi"].fillna(False).astype(bool)
    touch_hi = frame["fig9_touch_hi"].fillna(False).astype(bool)
    close = frame["px"].astype(float)
    upper = pd.to_numeric(frame["fig9_upper"], errors="coerce")
    dist_hi = close / upper - 1.0
    idx = frame.index
    n = len(frame)
    buy_sig = np.zeros(n, dtype=bool)
    sell_sig = np.zeros(n, dtype=bool)
    trades: list[dict[str, Any]] = []
    in_pos = False
    entry_i: int | None = None
    entry_sig_i: int | None = None
    run_len = 0
    run_touched = False
    leave = float(leave_pct)
    min_run = int(min_near_days)
    for i in range(n - 1):
        hi = bool(near_hi.iloc[i])
        if hi:
            run_len += 1
            run_touched = run_touched or bool(touch_hi.iloc[i])
        else:
            fe = i > 0 and bool(near_hi.iloc[i - 1])
            if in_pos and fe:
                armed = run_len >= min_run
                if require_hard_touch:
                    armed = armed and run_touched
                if armed and float(dist_hi.iloc[i]) <= -leave:
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
                            "sell_kind": "触上离开",
                            "sell_dist_hi": float(dist_hi.iloc[i]),
                            "run_len": int(run_len),
                        }
                    )
                    in_pos = False
                    entry_i = None
                    entry_sig_i = None
            run_len = 0
            run_touched = False
        if (not in_pos) and bool(buy_base.iloc[i]):
            buy_sig[i] = True
            entry_sig_i = i
            entry_i = i + 1
            in_pos = True
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
    return pd.DataFrame(trades), pd.Series(buy_sig, index=idx), pd.Series(sell_sig, index=idx)


def compute_glide_state(frame: pd.DataFrame) -> pd.Series:
    """Causal '贴 EWMA 下轨连续阴跌' warning mask.

    True when near the lower rail for at least ``GLIDE_NEAR_DAYS`` of the
    last ``GLIDE_WINDOW`` sessions, still within ``GLIDE_DIST_MAX`` of that
    rail, the worst 20d return over the window is ≤ ``GLIDE_RET20_TH``, and
    MA20 is falling. Diagnostic only — does not change buy/sell.
    """
    near_lo = frame["fig9_near_lo"].fillna(False).astype(bool)
    close = frame["px"].astype(float)
    lower = pd.to_numeric(frame["fig9_lower"], errors="coerce")
    dist_lo = close / lower - 1.0
    ret20 = close / close.shift(20) - 1.0
    ma20 = pd.to_numeric(frame["ma20"], errors="coerce")
    ma20_dn = ma20 < ma20.shift(5)
    run = near_lo.groupby((~near_lo).cumsum()).cumsum().where(near_lo, 0)
    cnt = near_lo.rolling(GLIDE_WINDOW).sum()
    min_ret20 = ret20.rolling(GLIDE_WINDOW).min()
    state = (run >= GLIDE_NEAR_DAYS) | (
        (cnt >= GLIDE_NEAR_DAYS)
        & (dist_lo <= GLIDE_DIST_MAX)
        & (min_ret20 <= GLIDE_RET20_TH)
        & ma20_dn
    )
    return state.fillna(False).astype(bool)


def glide_windows(mask: pd.Series) -> pd.DataFrame:
    """Collapse a boolean mask into [start, end] windows."""
    m = mask.fillna(False).astype(bool)
    rows: list[dict[str, Any]] = []
    start: pd.Timestamp | None = None
    prev: pd.Timestamp | None = None
    for dt, on in m.items():
        if on and start is None:
            start = dt
        if not on and start is not None:
            rows.append({"start": start, "end": prev})
            start = None
        prev = dt
    if start is not None and prev is not None:
        rows.append({"start": start, "end": prev})
    return pd.DataFrame(rows)


def slice_frame(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return frame[(frame.index >= start) & (frame.index <= end)].copy()


def eval_on(
    frame: pd.DataFrame,
    *,
    sm: Any,
    touch_mod: Any,
    params: dict[str, Any],
    regime_up: pd.Series,
) -> dict[str, Any]:
    buy_c, sell_c = build_candidates(
        frame,
        touch_mod=touch_mod,
        use_aux_buy=bool(params["use_aux_buy"]),
        use_aux_sell=bool(params["use_aux_sell"]),
        use_stretch=bool(params["use_stretch"]),
        locmin_days=int(params["locmin_days"]),
        holdup_gate=bool(params["holdup_gate"]),
        regime_up=regime_up,
    )
    if params.get("use_touch_leave"):
        trades, buy_sig, sell_sig = run_touch_leave_state_machine(
            frame,
            buy_c,
            leave_pct=float(params.get("leave_pct", DEFAULT_LEAVE_PCT)),
            min_near_days=int(params.get("leave_min_near", DEFAULT_LEAVE_MIN_NEAR)),
            require_hard_touch=bool(params.get("require_hard_touch", True)),
        )
    elif params["use_fail_c"]:
        trades, buy_sig, sell_sig = run_fail_c(frame, buy_c, sell_c, touch_mod=touch_mod)
    else:
        trades, buy_sig, sell_sig = sm.run_state_machine(frame, buy_c, sell_c)
    summ = sm.summarize_trades(trades)
    close = frame["px"].astype(float)
    ev = _stat(_fwd20(close, buy_c))
    un = _stat(_fwd20(close, pd.Series(True, index=frame.index)))
    xs = None
    if ev["mean"] is not None and un["mean"] is not None:
        xs = float(ev["mean"] - un["mean"])
    return {
        "n_buy_cand": int(buy_c.sum()),
        "n_sell_cand": int(sell_c.sum()),
        "n_sig_buy": int(buy_sig.sum()),
        "n_sig_sell": int(sell_sig.sum()),
        "n_trades": summ.get("n_trades"),
        "n_open": summ.get("n_open"),
        "win": summ.get("win_rate"),
        "mean": summ.get("mean_ret"),
        "compound": summ.get("compound"),
        "mean_hold": summ.get("mean_hold"),
        "evt20_n": ev["n"],
        "evt20_win": ev["win"],
        "evt20_mean": ev["mean"],
        "evt20_t": ev["t"],
        "evt20_xs": xs,
        "trades": trades,
        "buy_sig": buy_sig,
        "sell_sig": sell_sig,
        "buy_cand": buy_c,
        "sell_cand": sell_c,
    }


def rank_tuple(row: dict[str, Any]) -> tuple[int, float, float, float, int]:
    n = int(row.get("n_trades") or 0)
    ok = 1 if n >= MIN_TRAIN_TRADES else 0
    compound = float(row["compound"]) if row.get("compound") is not None else -9.0
    e20 = float(row["evt20_mean"]) if row.get("evt20_mean") is not None else -9.0
    wr = float(row["win"]) if row.get("win") is not None else 0.0
    return (ok, compound, e20, wr, n)


def cfg_id(p: dict[str, Any]) -> str:
    return (
        f"near{p['near_pct']:.3f}_lam{p['lam']:.2f}"
        f"_ab{int(p['use_aux_buy'])}_as{int(p['use_aux_sell'])}"
        f"_st{int(p['use_stretch'])}_c{int(p['use_fail_c'])}"
        f"_lm{int(p['locmin_days'])}_g0{int(p['holdup_gate'])}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code", default=CODE)
    p.add_argument("--html", type=Path, default=DEFAULT_HTML)
    p.add_argument("--listing-dir", type=Path, default=DEFAULT_LISTING)
    p.add_argument("--anchor-code", default="SH510300")
    p.add_argument("--skip-html", action="store_true")
    p.add_argument(
        "--skip-grid",
        action="store_true",
        help="reuse grid_results/selected.json buy params; only re-export overlay",
    )
    p.add_argument(
        "--sell-mode",
        choices=("s0", "touch_leave"),
        default="touch_leave",
        help="s0 = rising-edge near-hi (grid default); touch_leave = hard-touch then leave",
    )
    p.add_argument("--leave-pct", type=float, default=DEFAULT_LEAVE_PCT)
    p.add_argument("--leave-min-near", type=int, default=DEFAULT_LEAVE_MIN_NEAR)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sm = _load_sm()
    touch = _load_touch()
    bt = _load_bt()
    html = args.html if args.html.is_absolute() else _PROJECT_ROOT / args.html
    listing = (
        args.listing_dir
        if args.listing_dir.is_absolute()
        else _PROJECT_ROOT / args.listing_dir
    )
    if not html.exists():
        found = find_regime_html(listing, str(args.code).upper())
        if found is None:
            raise FileNotFoundError(html)
        html = found
    code = str(args.code).upper()
    out_dir = listing / f"fig9_ewma_l0s0_{code}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] building feature frame from {html.name}", flush=True)
    frame0, ohlcv = sm.build_feature_frame(html, listing, args.anchor_code, lookback=10)
    close = frame0["px"].astype(float)
    path9 = pd.to_numeric(frame0["fig9_path"], errors="coerce")
    regime_s = bt._load_regime_segments(listing / "trades", code)
    regime_up = pd.Series(False, index=frame0.index)
    if not regime_s.empty:
        aligned = regime_s.reindex(frame0.index)
        regime_up = aligned.astype(str).str.lower().eq("up").fillna(False)

    ewma_rails: dict[float, tuple[pd.Series, pd.Series]] = {}
    for lam in LAM_GRID:
        print(f"[INFO] EWMA envelope λ={lam}", flush=True)
        up, lo, _ = ewma_scaled_envelope(path9, close, lam=float(lam))
        ewma_rails[float(lam)] = (lo, up)

    p95_lo = pd.to_numeric(frame0["fig9_lower"], errors="coerce")
    p95_up = pd.to_numeric(frame0["fig9_upper"], errors="coerce")

    train_idx = (frame0.index >= EVAL_START) & (frame0.index <= TRAIN_END)
    valid_idx = frame0.index >= VALID_START

    prepared: dict[tuple[Any, ...], pd.DataFrame] = {}

    def prepared_frame(lam: float, near: float, stretch: bool, *, p95: bool) -> pd.DataFrame:
        key = (float(lam) if not p95 else -1.0, float(near), bool(stretch))
        cache_key = ("p95" if p95 else "ewma",) + key
        if cache_key in prepared:
            return prepared[cache_key]
        if p95:
            lo, up = p95_lo, p95_up
        else:
            lo, up = ewma_rails[float(lam)]
        fr = apply_fig9_rails(
            frame0, lower=lo, upper=up, near_pct=near, touch_mod=touch, mid_stretch=stretch
        )
        prepared[cache_key] = fr
        return fr

    rows: list[dict[str, Any]] = []
    p95_rec: dict[str, Any]
    best: dict[str, Any]
    if args.skip_grid:
        prev = json.loads((out_dir / "selected.json").read_text(encoding="utf-8"))
        buy_src = prev.get("grid_selected") or prev.get("selected")
        best = {
            "id": prev.get("selected_id") or "skip_grid",
            "near_pct": buy_src["near_pct"],
            "lam": buy_src["lam"],
            "use_aux_buy": buy_src["use_aux_buy"],
            "use_aux_sell": buy_src.get("use_aux_sell", True),
            "use_stretch": buy_src.get("use_stretch", False),
            "use_fail_c": buy_src.get("use_fail_c", False),
            "locmin_days": buy_src.get("locmin_days", 0),
            "holdup_gate": buy_src.get("holdup_gate", False),
            "train_n_trades": None,
            "train_win": None,
            "train_compound": None,
            "train_evt20_mean": None,
            "valid_n_trades": None,
            "valid_win": None,
            "valid_compound": None,
            "valid_evt20_mean": None,
        }
        p95_rec = prev.get("p95_baseline") or {"id": "p95_l0s0_default"}
        print("[INFO] skip-grid, buy params from selected.json", flush=True)
    else:
        grid = list(
            itertools.product(
                NEAR_GRID,
                LAM_GRID,
                BOOL_GRID,
                BOOL_GRID,
                BOOL_GRID,
                BOOL_GRID,
                LOCMIN_GRID,
                BOOL_GRID,
            )
        )
        print(f"[INFO] grid size={len(grid)}", flush=True)
        for near, lam, ab, asell, stretch, fail_c, locmin_d, g0 in grid:
            params = {
                "rail": "ewma",
                "near_pct": float(near),
                "lam": float(lam),
                "use_aux_buy": bool(ab),
                "use_aux_sell": bool(asell),
                "use_stretch": bool(stretch),
                "use_fail_c": bool(fail_c),
                "locmin_days": int(locmin_d),
                "holdup_gate": bool(g0),
            }
            fr = prepared_frame(float(lam), float(near), bool(stretch), p95=False)
            trn = eval_on(
                slice_frame(fr, EVAL_START, TRAIN_END),
                sm=sm,
                touch_mod=touch,
                params=params,
                regime_up=regime_up,
            )
            val = eval_on(
                slice_frame(fr, VALID_START, fr.index[-1]),
                sm=sm,
                touch_mod=touch,
                params=params,
                regime_up=regime_up,
            )
            rec = {"id": cfg_id(params), **params}
            for prefix, blob in (("train", trn), ("valid", val)):
                for k, v in blob.items():
                    if k in {"trades", "buy_sig", "sell_sig", "buy_cand", "sell_cand"}:
                        continue
                    rec[f"{prefix}_{k}"] = v
            rec["_train_rank"] = rank_tuple(
                {
                    "n_trades": trn["n_trades"],
                    "compound": trn["compound"],
                    "evt20_mean": trn["evt20_mean"],
                    "win": trn["win"],
                }
            )
            rows.append(rec)

        p95_params = {
            "rail": "p95",
            "near_pct": 0.01,
            "lam": None,
            "use_aux_buy": True,
            "use_aux_sell": True,
            "use_stretch": False,
            "use_fail_c": False,
            "locmin_days": 0,
            "holdup_gate": False,
        }
        fr_p95 = prepared_frame(0.94, 0.01, False, p95=True)
        p95_train = eval_on(
            slice_frame(fr_p95, EVAL_START, TRAIN_END),
            sm=sm,
            touch_mod=touch,
            params=p95_params,
            regime_up=regime_up,
        )
        p95_valid = eval_on(
            slice_frame(fr_p95, VALID_START, fr_p95.index[-1]),
            sm=sm,
            touch_mod=touch,
            params=p95_params,
            regime_up=regime_up,
        )
        p95_rec = {"id": "p95_l0s0_default", **p95_params}
        for prefix, blob in (("train", p95_train), ("valid", p95_valid)):
            for k, v in blob.items():
                if k in {"trades", "buy_sig", "sell_sig", "buy_cand", "sell_cand"}:
                    continue
                p95_rec[f"{prefix}_{k}"] = v

        order = sorted(range(len(rows)), key=lambda i: rows[i]["_train_rank"], reverse=True)
        grid_df = pd.DataFrame([rows[i] for i in order])
        drop_rank = grid_df.drop(columns=["_train_rank"])
        drop_rank.to_csv(out_dir / "grid_results.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([p95_rec]).to_csv(
            out_dir / "p95_baseline.csv", index=False, encoding="utf-8-sig"
        )
        best = rows[order[0]]

    selected = {
        "near_pct": best["near_pct"],
        "lam": best["lam"],
        "use_aux_buy": best["use_aux_buy"],
        "use_aux_sell": best["use_aux_sell"],
        "use_stretch": best["use_stretch"],
        "use_fail_c": best["use_fail_c"],
        "locmin_days": best["locmin_days"],
        "holdup_gate": best["holdup_gate"],
        "rail": "ewma",
    }
    grid_selected = dict(selected)
    if args.sell_mode == "touch_leave":
        selected = {
            **grid_selected,
            "use_stretch": False,
            "use_fail_c": False,
            "use_touch_leave": True,
            "leave_pct": float(args.leave_pct),
            "leave_min_near": int(args.leave_min_near),
            "require_hard_touch": True,
        }
    print("[INFO] selected", selected, flush=True)
    if not args.skip_grid:
        print(
            f"  train n={best['train_n_trades']} wr={best['train_win']} "
            f"compound={best['train_compound']} evt20={best['train_evt20_mean']}",
            flush=True,
        )
        print(
            f"  valid n={best['valid_n_trades']} wr={best['valid_win']} "
            f"compound={best['valid_compound']} evt20={best['valid_evt20_mean']}",
            flush=True,
        )

    fr_sel = prepared_frame(
        float(selected["lam"]),
        float(selected["near_pct"]),
        bool(selected.get("use_stretch", False)),
        p95=False,
    )
    full = eval_on(
        slice_frame(fr_sel, EVAL_START, fr_sel.index[-1]),
        sm=sm,
        touch_mod=touch,
        params=selected,
        regime_up=regime_up,
    )
    live_train = eval_on(
        slice_frame(fr_sel, EVAL_START, TRAIN_END),
        sm=sm,
        touch_mod=touch,
        params=selected,
        regime_up=regime_up,
    )
    live_valid = eval_on(
        slice_frame(fr_sel, VALID_START, fr_sel.index[-1]),
        sm=sm,
        touch_mod=touch,
        params=selected,
        regime_up=regime_up,
    )
    d = fr_sel.index[fr_sel.index <= DISCOVERY][-1]
    disc_buy = bool(full["buy_cand"].loc[d]) if d in full["buy_cand"].index else False
    print(f"[INFO] discovery {d.date()} buy_cand={disc_buy} px={float(close.loc[d]):.4f}")
    print(
        f"[INFO] live train n={live_train['n_trades']} wr={live_train['win']} "
        f"compound={live_train['compound']}",
        flush=True,
    )
    print(
        f"[INFO] live valid n={live_valid['n_trades']} wr={live_valid['win']} "
        f"compound={live_valid['compound']}",
        flush=True,
    )

    def _metrics(blob: dict[str, Any]) -> dict[str, Any]:
        return {
            k: blob[k]
            for k in (
                "n_buy_cand",
                "n_sell_cand",
                "n_sig_buy",
                "n_sig_sell",
                "n_trades",
                "n_open",
                "win",
                "mean",
                "compound",
                "mean_hold",
                "evt20_n",
                "evt20_win",
                "evt20_mean",
                "evt20_t",
                "evt20_xs",
            )
        }

    trades_full = full["trades"].copy()
    if not trades_full.empty:
        trades_full.to_csv(out_dir / "trades_full.csv", index=False, encoding="utf-8-sig")
        if args.sell_mode == "touch_leave":
            trades_full.to_csv(out_dir / "trades_leave.csv", index=False, encoding="utf-8-sig")
    sig_rows = []
    px = fr_sel["px"]
    path = fr_sel["fig9_path"]
    gap = fr_sel["fig9_gap"]
    pos = fr_sel["fig9_pos"]
    for dt in full["buy_sig"].index:
        is_b = bool(full["buy_sig"].loc[dt])
        is_s = bool(full["sell_sig"].loc[dt])
        if not (is_b or is_s):
            continue
        p = float(path.loc[dt]) if pd.notna(path.loc[dt]) else float("nan")
        g = float(gap.loc[dt]) if pd.notna(gap.loc[dt]) else float("nan")
        po = float(pos.loc[dt]) if pd.notna(pos.loc[dt]) else float("nan")
        sig_rows.append(
            {
                "kind": "buy" if is_b else "sell",
                "date": dt.strftime("%Y-%m-%d"),
                "px": float(px.loc[dt]),
                "path": p,
                "gap": g,
                "fig9_pos": po,
            }
        )
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(out_dir / "signals.csv", index=False, encoding="utf-8-sig")
    if args.sell_mode == "touch_leave" and not sig_df.empty:
        sig_df.to_csv(out_dir / "signals_leave.csv", index=False, encoding="utf-8-sig")

    payload = {
        "code": code,
        "html": str(html),
        "train": "2018-01-01..2024-12-31",
        "valid": "2025-01-01..as_of",
        "discovery": str(DISCOVERY.date()),
        "discovery_used_in_tuning": False,
        "min_train_trades": MIN_TRAIN_TRADES,
        "sell_mode": args.sell_mode,
        "grid_selected": grid_selected,
        "selected": selected,
        "selected_id": best.get("id"),
        "train_metrics": _metrics(live_train),
        "valid_metrics": _metrics(live_valid),
        "p95_baseline": p95_rec,
        "discovery_buy_cand": disc_buy,
        "n_grid": len(rows),
        "n_signals": len(sig_df),
    }
    (out_dir / "selected.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    html_out = None
    if not args.skip_html:
        ov = _load_overlay()
        html_txt = html.read_text(encoding="utf-8")
        traces = _extract_plotly_traces(html_txt)
        layout = ov._load_oracle_mod().extract_layout(html_txt)
        pivots = sig_df.copy()
        if args.sell_mode == "touch_leave":
            title_note = (
                "｜图1–11 图9 EWMA L0+触上离开："
                f"近下上跳沿买 / 硬触EWMA上轨后 dist_hi≤−{selected['leave_pct']:.0%} 离开才卖"
                f" λ={selected['lam']} near={selected['near_pct']:.1%}"
                f"（调参买档 2018–2024；卖侧按触上再离开，非 S0 立刻卖；非生产 hold_up）"
                f"｜买{int(full['n_sig_buy'])} 卖{int(full['n_sig_sell'])}"
            )
            html_suffix = "_fig9_ewma_l0_leave.html"
        else:
            title_note = (
                "｜图1–11 图9 EWMA L0+S0："
                f"近下上跳沿买 / 近上上跳沿卖 λ={selected['lam']} near={selected['near_pct']:.1%}"
                f" auxB={int(selected['use_aux_buy'])} auxS={int(selected['use_aux_sell'])}"
                f" stretch={int(selected['use_stretch'])} failC={int(selected['use_fail_c'])}"
                f" locmin={selected['locmin_days']} G0={int(selected['holdup_gate'])}"
                f"（调参 2018–2024，验证 2025+；非生产 hold_up）"
                f"｜买{int(full['n_sig_buy'])} 卖{int(full['n_sig_sell'])}"
            )
            html_suffix = "_fig9_ewma_l0s0.html"
        fig = ov.overlay_fig9_rule(
            traces=traces,
            layout=layout,
            pivots=pivots,
            ohlcv=ohlcv,
            gap_th=-0.05,
            title_note=title_note,
            glide_windows=glide_windows(compute_glide_state(fr_sel)),
        )
        html_out = html.with_name(html.stem + html_suffix)
        write_adaptive_html(fig, html_out, include_plotlyjs="cdn")
        print(f"[OK] html {html_out} ({html_out.stat().st_size / 1e6:.1f} MB)")

    payload["html_out"] = None if html_out is None else str(html_out)
    (out_dir / "selected.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
