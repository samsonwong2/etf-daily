#!/usr/bin/env python3
"""Backtest daily ticket-card q05 / Omega against realized 20-day simple returns.

Research only. Does not change hold_up.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from runtime_paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from decision_pack.scripts.backtest_beat_or_hold_513650 import (  # noqa: E402
    apply_new_veto,
    collect_tickets,
    load_universe,
)
from decision_pack.scripts.build_daily_ticket_card import (  # noqa: E402
    assemble_ticket_card,
    load_panel_and_regimes,
)
from decision_pack.src.garch_dynamics import GarchDynamicsConfig  # noqa: E402
from decision_pack.src.garch_short_horizon_board import GarchShortHorizonConfig  # noqa: E402
from decision_pack.src.vol_need_return_board import TARGET_SHARPE_LIKE  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--as-of", default="2026-08-12")
    p.add_argument("--start", default="2024-01-02")
    p.add_argument(
        "--adaptive-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260812all_adaptive",
    )
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--mc-samples", type=int, default=1000)
    p.add_argument("--min-history", type=int, default=252)
    p.add_argument("--sharpe-target", type=float, default=TARGET_SHARPE_LIKE)
    p.add_argument(
        "--snapshot",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260812_ticket_card_smoke/ticket_card.csv",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def fwd_simple(ser: pd.Series, t: pd.Timestamp, bars: int) -> float:
    s = ser.dropna().sort_index()
    if s.empty:
        return float("nan")
    loc = s.index.searchsorted(pd.Timestamp(t).normalize())
    if loc >= len(s) or s.index[loc] != pd.Timestamp(t).normalize():
        # as_of may not be a trading day; use last close <= t as T
        loc = int(s.index.searchsorted(pd.Timestamp(t).normalize(), side="right") - 1)
    if loc < 0 or loc + bars >= len(s):
        return float("nan")
    p0 = float(s.iloc[loc])
    p1 = float(s.iloc[loc + bars])
    if p0 <= 0 or not np.isfinite(p0) or not np.isfinite(p1):
        return float("nan")
    return p1 / p0 - 1.0


def spearman(a: pd.Series, b: pd.Series) -> float:
    d = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(d) < 8:
        return float("nan")
    return float(d["a"].corr(d["b"], method="spearman"))


def snapshot_integrity(path: Path) -> dict[str, Any]:
    board = pd.read_csv(path)
    finite = board.loc[np.isfinite(board["q05_20d_garch"])]
    floor = float(finite["q05_20d_garch"].quantile(0.25)) if not finite.empty else float("nan")
    file_floor = float(board["q05_floor_20d"].dropna().iloc[0]) if board["q05_floor_20d"].notna().any() else float("nan")
    expected_pass = (
        np.isfinite(board["q05_20d_garch"])
        & np.isfinite(board["q05_floor_20d"])
        & (board["q05_20d_garch"] >= board["q05_floor_20d"])
    )
    return {
        "path": str(path),
        "n_rows": int(len(board)),
        "n_garch_ok": int(board["garch_reliable"].sum()),
        "floor_file": file_floor,
        "floor_recomputed": floor,
        "floor_abs_diff": abs(file_floor - floor) if np.isfinite(file_floor) and np.isfinite(floor) else float("nan"),
        "pass_flag_mismatches": int((board["pass_q05_floor"].astype(int) != expected_pass.astype(int)).sum()),
        "unreliable_with_q05": int(((board["garch_reliable"] == 0) & np.isfinite(board["q05_20d_garch"])).sum()),
        "n_in_up": int(board["in_up"].sum()),
        "n_pass_q05": int(board["pass_q05_floor"].sum()),
        "n_pass_up_and_q05": int(board["pass_up_and_q05"].sum()),
        "has_match_column": "match" in board.columns,
    }


def main() -> int:
    args = parse_args()
    as_of = pd.Timestamp(args.as_of).normalize()
    start = pd.Timestamp(args.start).normalize()
    adaptive_dir = (
        args.adaptive_dir if args.adaptive_dir.is_absolute() else _ROOT / args.adaptive_dir
    )
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = PLOTLY_OUTPUTS_DIR / f"{as_of.strftime('%Y%m%d')}_ticket_card_eval"
    elif not out_dir.is_absolute():
        out_dir = _ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    snap_path = args.snapshot if args.snapshot.is_absolute() else _ROOT / args.snapshot
    integrity = snapshot_integrity(snap_path) if snap_path.is_file() else {"missing": True}

    codes, name_map = load_universe(adaptive_dir)
    closes, regimes = load_panel_and_regimes(
        codes,
        adaptive_dir=adaptive_dir,
        as_of=as_of,
        min_history=int(args.min_history),
        history_start=start,
    )
    panel = pd.DataFrame(closes).sort_index()
    calendar = panel.dropna(how="all").index
    cal = calendar[(calendar >= start) & (calendar <= as_of)]
    h = int(args.horizon)
    signals = list(cal[: -(h + 1) : h])
    tcfg = GarchShortHorizonConfig(garch=GarchDynamicsConfig(mc_samples=int(args.mc_samples)))

    rows: list[dict[str, Any]] = []
    for i, sig in enumerate(signals, start=1):
        tickets = collect_tickets(
            panel,
            pd.Timestamp(sig),
            horizon=h,
            cfg=tcfg,
            sharpe_like=float(args.sharpe_target),
            min_history=int(args.min_history),
        )
        _, floor = apply_new_veto(tickets)
        board = assemble_ticket_card(
            codes,
            name_map,
            as_of=pd.Timestamp(sig),
            tickets=tickets,
            regimes=regimes,
            closes=closes,
            floor=floor,
        )
        for rec in board.to_dict("records"):
            code = str(rec["code"])
            r20 = fwd_simple(panel[code], pd.Timestamp(sig), h) if code in panel.columns else float("nan")
            q05 = float(rec["q05_20d_garch"])
            rows.append(
                {
                    "signal": pd.Timestamp(sig).strftime("%Y-%m-%d"),
                    "code": code,
                    "garch_reliable": int(rec["garch_reliable"]),
                    "in_up": int(rec["in_up"]),
                    "pass_q05_floor": int(rec["pass_q05_floor"]),
                    "pass_up_and_q05": int(rec["pass_up_and_q05"]),
                    "q05_20d_garch": q05,
                    "omega_match_20d": rec["omega_match_20d"],
                    "r_fwd20": r20,
                    "breach_q05": int(np.isfinite(r20) and np.isfinite(q05) and r20 < q05),
                }
            )
        print(f"[{i}/{len(signals)}] {pd.Timestamp(sig).date()} tickets={len(tickets)}")

    obs = pd.DataFrame(rows)
    usable = obs.loc[(obs["garch_reliable"] == 1) & np.isfinite(obs["r_fwd20"]) & np.isfinite(obs["q05_20d_garch"])].copy()
    n = int(len(usable))
    breaches = int(usable["breach_q05"].sum()) if n else 0
    hit = breaches / n if n else float("nan")
    se = float(np.sqrt(hit * (1 - hit) / n)) if n and np.isfinite(hit) else float("nan")

    def mean_r(mask: pd.Series) -> float:
        sub = usable.loc[mask, "r_fwd20"]
        return float(sub.mean()) if len(sub) else float("nan")

    summary: dict[str, Any] = {
        "as_of": str(as_of.date()),
        "start": str(start.date()),
        "horizon": h,
        "n_signals": int(len(signals)),
        "n_obs": n,
        "q05_breach_rate": hit,
        "q05_breach_se": se,
        "q05_target": 0.05,
        "q05_breach_vs_5pp": (hit - 0.05) if np.isfinite(hit) else float("nan"),
        "spearman_q05_vs_r20": spearman(usable["q05_20d_garch"], usable["r_fwd20"]),
        "spearman_omega_vs_r20": spearman(usable["omega_match_20d"], usable["r_fwd20"]),
        "mean_r20_pass_q05": mean_r(usable["pass_q05_floor"] == 1),
        "mean_r20_fail_q05": mean_r(usable["pass_q05_floor"] == 0),
        "n_pass_q05": int((usable["pass_q05_floor"] == 1).sum()),
        "n_fail_q05": int((usable["pass_q05_floor"] == 0).sum()),
        "mean_r20_up_and_q05": mean_r(usable["pass_up_and_q05"] == 1),
        "mean_r20_in_up": mean_r(usable["in_up"] == 1),
        "n_up_and_q05": int((usable["pass_up_and_q05"] == 1).sum()),
        "n_in_up": int((usable["in_up"] == 1).sum()),
        "snapshot": integrity,
        "signal_first": signals[0].strftime("%Y-%m-%d") if signals else None,
        "signal_last": signals[-1].strftime("%Y-%m-%d") if signals else None,
    }

    by_sig = []
    for sig, g in usable.groupby("signal"):
        br = float(g["breach_q05"].mean()) if len(g) else float("nan")
        by_sig.append(
            {
                "signal": sig,
                "n": int(len(g)),
                "breach_rate": br,
                "mean_r20_pass": float(g.loc[g["pass_q05_floor"] == 1, "r_fwd20"].mean()) if (g["pass_q05_floor"] == 1).any() else float("nan"),
                "mean_r20_fail": float(g.loc[g["pass_q05_floor"] == 0, "r_fwd20"].mean()) if (g["pass_q05_floor"] == 0).any() else float("nan"),
                "spearman_q05": spearman(g["q05_20d_garch"], g["r_fwd20"]),
                "spearman_omega": spearman(g["omega_match_20d"], g["r_fwd20"]),
            }
        )
    per = pd.DataFrame(by_sig)

    obs.to_csv(out_dir / "observations.csv", index=False, float_format="%.6f")
    per.to_csv(out_dir / "per_signal.csv", index=False, float_format="%.6f")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: summary[k] for k in summary if k != "snapshot"}, indent=2, default=str))
    print("snapshot", json.dumps(integrity, indent=2, default=str))
    print(f"[OK] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
