#!/usr/bin/env python3
"""Daily research ticket card: same-day q05 floor + Omega diagnostic.

Writes one row per ``load_universe`` name into ``ticket_card.csv`` next to the
adaptive HTML batch. Does not change hold_up trade rules.

q05 is a same-day left-tail rank of zero-mean GARCH MC simple returns.
omega_match_20d is Omega (E[max(R,0)]/E[max(-R,0)]), diagnostic only.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from decision_pack.scripts.backtest_beat_or_hold_513650 import (  # noqa: E402
    ANCHOR,
    _load_mod,
    apply_new_veto,
    collect_tickets,
    load_symbol_config,
    load_universe,
    mom_at,
)
from decision_pack.src.adaptive_stage_common import prepare_features  # noqa: E402
from decision_pack.src.garch_dynamics import GarchDynamicsConfig  # noqa: E402
from decision_pack.src.garch_short_horizon_board import GarchShortHorizonConfig  # noqa: E402
from decision_pack.src.vol_need_return_board import TARGET_SHARPE_LIKE  # noqa: E402

OUTPUT_COLUMNS = (
    "as_of",
    "code",
    "name",
    "regime",
    "in_up",
    "q05_20d_garch",
    "q05_floor_20d",
    "pass_q05_floor",
    "omega_match_20d",
    "mom5",
    "mom5_anchor",
    "mom5_vs_anchor",
    "garch_reliable",
    "pass_up_and_q05",
)

META_KEYS = (
    "as_of",
    "n_codes",
    "n_garch_ok",
    "q05_floor_20d",
    "horizon",
    "mc_samples",
    "sharpe_target",
    "formula_q05",
    "formula_omega_match",
)

FORMULA_Q05 = (
    "5% quantile of zero-mean GARCH MC simple returns; "
    "same-day cross-section rank, not a 20-day loss forecast"
)
FORMULA_OMEGA = (
    "E[max(R,0)]/E[max(-R,0)] on those paths; Omega performance ratio; not a risk gate"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--as-of", required=True)
    p.add_argument("--adaptive-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--mc-samples", type=int, default=1000)
    p.add_argument("--min-history", type=int, default=252)
    p.add_argument("--sharpe-target", type=float, default=TARGET_SHARPE_LIKE)
    return p.parse_args()


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _json_float(value: Any) -> float | None:
    if not _finite(value):
        return None
    return float(value)


def last_regime(series: pd.Series | None, as_of: pd.Timestamp) -> str:
    if series is None or series.empty:
        return ""
    hist = series.loc[: pd.Timestamp(as_of)].dropna()
    if hist.empty:
        return ""
    return str(hist.iloc[-1]).strip().lower()


def assemble_ticket_card(
    codes: list[str],
    name_map: dict[str, str],
    *,
    as_of: pd.Timestamp,
    tickets: pd.DataFrame,
    regimes: dict[str, pd.Series],
    closes: dict[str, pd.Series],
    floor: float,
) -> pd.DataFrame:
    """Left-join universe onto collect_tickets; no GARCH in this function."""
    ticket_by_code: dict[str, dict[str, Any]] = {}
    if tickets is not None and not tickets.empty:
        for rec in tickets.to_dict("records"):
            ticket_by_code[str(rec["code"]).upper()] = rec

    mom5_anchor = float("nan")
    anchor_close = closes.get(ANCHOR)
    if anchor_close is not None:
        mom5_anchor = mom_at(anchor_close, as_of, 5)

    rows: list[dict[str, Any]] = []
    for raw in codes:
        code = str(raw).upper()
        rec = ticket_by_code.get(code)
        q05 = float(rec["q05"]) if rec is not None else float("nan")
        match = float(rec["match"]) if rec is not None else float("nan")
        reliable = int(rec is not None)
        pass_floor = int(_finite(q05) and _finite(floor) and float(q05) >= float(floor))
        regime = last_regime(regimes.get(code), as_of)
        in_up = int(regime == "up")
        close = closes.get(code)
        mom5 = mom_at(close, as_of, 5) if close is not None else float("nan")
        vs_anchor = (
            float(mom5) - float(mom5_anchor)
            if _finite(mom5) and _finite(mom5_anchor)
            else float("nan")
        )
        rows.append(
            {
                "as_of": pd.Timestamp(as_of).strftime("%Y-%m-%d"),
                "code": code,
                "name": str(name_map.get(code, code)),
                "regime": regime,
                "in_up": in_up,
                "q05_20d_garch": q05,
                "q05_floor_20d": float(floor) if _finite(floor) else float("nan"),
                "pass_q05_floor": pass_floor,
                "omega_match_20d": match if _finite(match) else float("nan"),
                "mom5": mom5,
                "mom5_anchor": mom5_anchor,
                "mom5_vs_anchor": vs_anchor,
                "garch_reliable": reliable,
                "pass_up_and_q05": int(in_up == 1 and pass_floor == 1),
            }
        )
    return pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))


def build_meta(
    board: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    floor: float,
    horizon: int,
    mc_samples: int,
    sharpe_target: float,
) -> dict[str, Any]:
    n_ok = int(board["garch_reliable"].sum()) if not board.empty else 0
    meta = {
        "as_of": pd.Timestamp(as_of).strftime("%Y-%m-%d"),
        "n_codes": int(len(board)),
        "n_garch_ok": n_ok,
        "q05_floor_20d": _json_float(floor),
        "horizon": int(horizon),
        "mc_samples": int(mc_samples),
        "sharpe_target": float(sharpe_target),
        "formula_q05": FORMULA_Q05,
        "formula_omega_match": FORMULA_OMEGA,
    }
    if tuple(meta) != META_KEYS:
        raise RuntimeError("ticket_card_meta keys drifted")
    return meta


def write_outputs(board: pd.DataFrame, meta: dict[str, Any], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(out_csv, index=False, float_format="%.6f")
    meta_path = out_csv.with_name("ticket_card_meta.json")
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_panel_and_regimes(
    codes: list[str],
    *,
    adaptive_dir: Path,
    as_of: pd.Timestamp,
    min_history: int,
    history_start: pd.Timestamp | None = None,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    pm = _load_mod("pm_ticket", _ROOT / "decision_pack/scripts/plot_regime_transition_example.py")
    ev = _load_mod("ev_ticket", _ROOT / "decision_pack/scripts/eval_causal_regime_switch_pool.py")
    pm._init_qlib(None)
    origin = pd.Timestamp(history_start).normalize() if history_start is not None else as_of
    load_start = (origin - pd.Timedelta(days=int(min_history * 2 + 120))).strftime("%Y-%m-%d")
    end = str(as_of.date())
    closes: dict[str, pd.Series] = {}
    regimes: dict[str, pd.Series] = {}
    for i, code in enumerate(codes, start=1):
        try:
            ohlcv = pm.load_qlib_ohlcv(
                code,
                load_start,
                end,
                init_qlib=False,
                lookback_calendar_days=220,
                clip_to_window=False,
            )
            if ohlcv is None or ohlcv.empty:
                print(f"[WARN] {code}: empty OHLCV", file=sys.stderr)
                continue
            close_col = "close" if "close" in ohlcv.columns else "$close"
            close = pd.to_numeric(ohlcv[close_col], errors="coerce")
            close.index = pd.to_datetime(
                ohlcv["datetime" if "datetime" in ohlcv.columns else "as_of"]
            )
            closes[code] = close[~close.index.duplicated(keep="last")].sort_index()
            cfg = load_symbol_config(adaptive_dir, code)
            if cfg.get("method"):
                px = prepare_features(
                    ev,
                    ohlcv,
                    method=str(cfg["method"]),
                    method_params=cfg.get("method_params") or {},
                    sticky_guard=False,
                )
                if "regime" in px.columns:
                    reg = px.set_index("as_of")["regime"].astype(str)
                    reg.index = pd.to_datetime(reg.index).normalize()
                    regimes[code] = reg[~reg.index.duplicated(keep="last")].sort_index()
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {code}: {exc}", file=sys.stderr)
        if i % 15 == 0 or i == len(codes):
            print(f"loaded {i}/{len(codes)}")
    if ANCHOR not in closes:
        warnings.warn(
            f"anchor {ANCHOR} missing from panel; mom5_anchor will be NaN",
            RuntimeWarning,
            stacklevel=1,
        )
    return closes, regimes


def main() -> int:
    args = parse_args()
    as_of = pd.Timestamp(args.as_of).normalize()
    adaptive_dir = (
        args.adaptive_dir if args.adaptive_dir.is_absolute() else _ROOT / args.adaptive_dir
    )
    if not (adaptive_dir / "batch_summary.csv").is_file():
        print(f"[ERROR] missing {adaptive_dir / 'batch_summary.csv'}", file=sys.stderr)
        return 2
    out_csv = args.out
    if out_csv is None:
        out_csv = adaptive_dir / "ticket_card.csv"
    elif not out_csv.is_absolute():
        out_csv = _ROOT / out_csv

    codes, name_map = load_universe(adaptive_dir)
    print(f"universe={len(codes)} as_of={as_of.date()} adaptive_dir={adaptive_dir}")
    closes, regimes = load_panel_and_regimes(
        codes,
        adaptive_dir=adaptive_dir,
        as_of=as_of,
        min_history=int(args.min_history),
    )
    panel = pd.DataFrame(closes).sort_index() if closes else pd.DataFrame()
    tcfg = GarchShortHorizonConfig(garch=GarchDynamicsConfig(mc_samples=int(args.mc_samples)))
    tickets = collect_tickets(
        panel,
        as_of,
        horizon=int(args.horizon),
        cfg=tcfg,
        sharpe_like=float(args.sharpe_target),
        min_history=int(args.min_history),
    )
    _passed, floor = apply_new_veto(tickets)
    board = assemble_ticket_card(
        codes,
        name_map,
        as_of=as_of,
        tickets=tickets,
        regimes=regimes,
        closes=closes,
        floor=floor,
    )
    if len(board) != len(codes):
        print("[ERROR] ticket card row count != universe", file=sys.stderr)
        return 2
    passed_codes = set(board.loc[board["pass_q05_floor"] == 1, "code"].astype(str))
    if passed_codes != set(_passed):
        print("[ERROR] pass_q05_floor set != apply_new_veto", file=sys.stderr)
        return 2
    meta = build_meta(
        board,
        as_of=as_of,
        floor=floor,
        horizon=int(args.horizon),
        mc_samples=int(args.mc_samples),
        sharpe_target=float(args.sharpe_target),
    )
    write_outputs(board, meta, out_csv)
    print(
        f"[OK] {out_csv} rows={len(board)} garch_ok={meta['n_garch_ok']} "
        f"floor={meta['q05_floor_20d']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
