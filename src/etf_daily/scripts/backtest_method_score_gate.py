#!/usr/bin/env python3
"""Research A/B: when train method_score < 0, do not use hold_up timing.

Does NOT change production defaults. Reads frozen from_listing configs + trades.

Variants per name:
  baseline: always hold_up (config oos compound/bh/edge)
  cash_gate: method_score < 0 → cash (compound=0)
  bh_gate:   method_score < 0 → buy&hold (compound=bh)  ← preferred fallback

Also reports post-cutoff (train_cutoff) trade-based slice when trades exist.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.paths import PLOTLY_OUTPUTS_DIR  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260814_from_listing/configs",
    )
    p.add_argument(
        "--trades-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260814_from_listing/trades",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260814_method_score_gate",
    )
    p.add_argument("--train-cutoff", default="2026-04-01")
    p.add_argument("--score-threshold", type=float, default=0.0)
    return p.parse_args(argv)


def trade_compound_after(
    trades_path: Path, *, cutoff: str
) -> dict[str, Any]:
    """Compound of closed trades with exit >= cutoff (approx OOS)."""
    if not trades_path.exists():
        return {"n_closed": 0, "compound": float("nan"), "win_rate": float("nan")}
    try:
        tr = pd.read_csv(trades_path)
    except pd.errors.EmptyDataError:
        return {"n_closed": 0, "compound": float("nan"), "win_rate": float("nan")}
    if tr.empty or "fwd_ret" not in tr.columns:
        return {"n_closed": 0, "compound": float("nan"), "win_rate": float("nan")}
    sells = tr[tr["side"].astype(str).str.upper() == "SELL"].copy()
    if sells.empty:
        return {"n_closed": 0, "compound": 0.0, "win_rate": float("nan")}
    sells["date"] = pd.to_datetime(sells["date"])
    cut = pd.Timestamp(cutoff)
    sub = sells[sells["date"] >= cut]
    if sub.empty:
        return {"n_closed": 0, "compound": 0.0, "win_rate": float("nan")}
    rets = pd.to_numeric(sub["fwd_ret"], errors="coerce").dropna()
    if rets.empty:
        return {"n_closed": 0, "compound": 0.0, "win_rate": float("nan")}
    return {
        "n_closed": int(len(rets)),
        "compound": float((1.0 + rets).prod() - 1.0),
        "win_rate": float((rets > 0).mean()),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thr = float(args.score_threshold)
    cut = str(args.train_cutoff)

    rows: list[dict[str, Any]] = []
    for path in sorted(Path(args.config_dir).glob("*.json")):
        j = json.loads(path.read_text(encoding="utf-8"))
        code = str(j.get("code") or path.stem)
        score = j.get("method_score")
        try:
            score_f = float(score) if score is not None else float("nan")
        except (TypeError, ValueError):
            score_f = float("nan")
        oos = j.get("oos") or {}
        bh = float(oos["bh"]) if oos.get("bh") is not None else float("nan")
        base_c = (
            float(oos["compound"]) if oos.get("compound") is not None else float("nan")
        )
        base_e = float(oos["edge"]) if oos.get("edge") is not None else float("nan")
        if not np.isfinite(base_e) and np.isfinite(base_c) and np.isfinite(bh):
            base_e = base_c - bh

        gated = bool(np.isfinite(score_f) and score_f < thr)
        # nan score: keep baseline (do not invent)
        if not np.isfinite(score_f):
            gated = False

        cash_c = 0.0 if gated else base_c
        cash_e = (0.0 - bh) if gated and np.isfinite(bh) else base_e
        bh_c = bh if gated and np.isfinite(bh) else base_c
        bh_e = 0.0 if gated and np.isfinite(bh) else base_e

        tmeta = trade_compound_after(
            Path(args.trades_dir) / f"{code}_trades.csv", cutoff=cut
        )
        # Post-cutoff: gated → cash 0 / or BH unknown without price panel → only cash
        post_base = float(tmeta["compound"])
        post_cash = 0.0 if gated else post_base

        rows.append(
            {
                "code": code,
                "method": j.get("method"),
                "method_score": score_f,
                "gated": gated,
                "n_closed_full": oos.get("n_closed"),
                "bh_full": bh,
                "baseline_compound": base_c,
                "baseline_edge": base_e,
                "cash_gate_compound": cash_c,
                "cash_gate_edge": cash_e,
                "bh_gate_compound": bh_c,
                "bh_gate_edge": bh_e,
                "post_cutoff_n_closed": tmeta["n_closed"],
                "post_cutoff_hold_up_compound": post_base,
                "post_cutoff_cash_if_gated": post_cash,
                "post_cutoff_win_rate": tmeta["win_rate"],
                "train_end": j.get("train_end"),
                "window_start": oos.get("start"),
                "window_end": oos.get("end"),
            }
        )

    df = pd.DataFrame(rows).sort_values("method_score", ascending=True)
    df.to_csv(out_dir / "per_code.csv", index=False)

    def _ew(col: str) -> float:
        s = pd.to_numeric(df[col], errors="coerce")
        s = s[np.isfinite(s)]
        return float(s.mean()) if len(s) else float("nan")

    summary = {
        "n_codes": int(len(df)),
        "score_threshold": thr,
        "n_gated": int(df["gated"].sum()),
        "n_ungated": int((~df["gated"]).sum()),
        "n_score_nan": int((~np.isfinite(df["method_score"])).sum()),
        "ew_baseline_compound": _ew("baseline_compound"),
        "ew_baseline_edge": _ew("baseline_edge"),
        "ew_cash_gate_compound": _ew("cash_gate_compound"),
        "ew_cash_gate_edge": _ew("cash_gate_edge"),
        "ew_bh_gate_compound": _ew("bh_gate_compound"),
        "ew_bh_gate_edge": _ew("bh_gate_edge"),
        "delta_ew_edge_bh_gate_vs_baseline": _ew("bh_gate_edge") - _ew("baseline_edge"),
        "delta_ew_edge_cash_gate_vs_baseline": _ew("cash_gate_edge")
        - _ew("baseline_edge"),
        "delta_ew_compound_bh_gate_vs_baseline": _ew("bh_gate_compound")
        - _ew("baseline_compound"),
    }

    # Classification among gated names using full-window edge
    g = df[df["gated"]].copy()
    ug = df[~df["gated"]].copy()
    summary["gated_mean_baseline_edge"] = (
        float(g["baseline_edge"].mean()) if len(g) else float("nan")
    )
    summary["ungated_mean_baseline_edge"] = (
        float(ug["baseline_edge"].mean()) if len(ug) else float("nan")
    )
    # True help for bh_gate: baseline_edge < 0 (timing lost to BH)
    if len(g):
        summary["gated_timing_lost_to_bh"] = int((g["baseline_edge"] < 0).sum())
        summary["gated_timing_beat_bh"] = int((g["baseline_edge"] > 0).sum())
        # False kill for cash: baseline_compound > 0 (timing still made money)
        summary["gated_hold_up_positive_compound"] = int(
            (g["baseline_compound"] > 0).sum()
        )
    if len(ug):
        summary["ungated_timing_lost_to_bh"] = int((ug["baseline_edge"] < 0).sum())
        summary["ungated_timing_beat_bh"] = int((ug["baseline_edge"] > 0).sum())

    callouts = {}
    for code in ("SZ159985", "SH588710", "SH515220", "SH512690"):
        sub = df[df["code"] == code]
        if not sub.empty:
            callouts[code] = sub.iloc[0].to_dict()
    summary["callouts"] = callouts

    # Prefer bh_gate if it lifts EW edge; else reject wiring
    d_edge = summary["delta_ew_edge_bh_gate_vs_baseline"]
    d_c = summary["delta_ew_compound_bh_gate_vs_baseline"]
    if np.isfinite(d_edge) and d_edge > 0 and np.isfinite(d_c) and d_c >= 0:
        verdict = "PROMISING_bh_fallback_when_method_score_lt_0"
    elif np.isfinite(d_edge) and d_edge > 0:
        verdict = "MIXED_bh_fallback_lifts_edge_check_compound"
    else:
        verdict = "REJECT_or_revise_gate_no_ew_edge_lift"

    # Cash gate almost never preferred for rising markets
    if summary["delta_ew_edge_cash_gate_vs_baseline"] > d_edge:
        cash_note = "cash_gate_better_than_bh_gate_on_ew_edge_unusual"
    else:
        cash_note = "bh_gate_preferred_over_cash_gate_on_ew_edge"

    payload = {
        "verdict": verdict,
        "cash_note": cash_note,
        "summary": {k: v for k, v in summary.items() if k != "callouts"},
        "callouts": callouts,
        "note": (
            "method_score is train B0 score (hold_up + agree − switches). "
            "Full-window oos.* in configs spans listing→as_of (includes train). "
            "bh_gate = skip timing, hold BH when score<0. "
            "Production hold_up defaults unchanged."
        ),
    }
    with (out_dir / "verdict.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    (out_dir / "README.md").write_text(
        """# method_score &lt; 0 gate A/B

Train `method_score < 0` → do not use hold_up timing.

| Variant | Gated names do |
|---------|----------------|
| baseline | hold_up |
| cash_gate | cash (0) |
| **bh_gate** | buy&hold (skip timing) |

Production defaults not changed.
""",
        encoding="utf-8",
    )

    # Print
    print("=== method_score gate A/B ===", flush=True)
    print(
        f"n={summary['n_codes']} gated={summary['n_gated']} "
        f"ungated={summary['n_ungated']} nan_score={summary['n_score_nan']}",
        flush=True,
    )
    print(
        f"EW edge  baseline={summary['ew_baseline_edge']:+.4f}  "
        f"cash_gate={summary['ew_cash_gate_edge']:+.4f}  "
        f"bh_gate={summary['ew_bh_gate_edge']:+.4f}",
        flush=True,
    )
    print(
        f"EW cmpd  baseline={summary['ew_baseline_compound']:+.4f}  "
        f"cash_gate={summary['ew_cash_gate_compound']:+.4f}  "
        f"bh_gate={summary['ew_bh_gate_compound']:+.4f}",
        flush=True,
    )
    print(
        f"delta EW edge bh_gate={summary['delta_ew_edge_bh_gate_vs_baseline']:+.4f}  "
        f"cash_gate={summary['delta_ew_edge_cash_gate_vs_baseline']:+.4f}",
        flush=True,
    )
    print(
        f"gated: timing_lost_to_bh={summary.get('gated_timing_lost_to_bh')} "
        f"timing_beat_bh={summary.get('gated_timing_beat_bh')}",
        flush=True,
    )
    print(
        f"ungated: timing_lost_to_bh={summary.get('ungated_timing_lost_to_bh')} "
        f"timing_beat_bh={summary.get('ungated_timing_beat_bh')}",
        flush=True,
    )
    print("\ncallouts:", flush=True)
    for code, r in callouts.items():
        print(
            f"  {code}: score={r['method_score']:.4f} gated={r['gated']} "
            f"base_edge={r['baseline_edge']:+.4f} "
            f"bh_gate_edge={r['bh_gate_edge']:+.4f} "
            f"base_c={r['baseline_compound']:+.4f}",
            flush=True,
        )
    # Top false kills: gated but baseline_edge > 0
    fk = df[df["gated"] & (df["baseline_edge"] > 0)].sort_values(
        "baseline_edge", ascending=False
    )
    print(f"\nfalse_kills_bh_gate (gated but hold_up beat BH): n={len(fk)}", flush=True)
    if not fk.empty:
        print(
            fk[
                ["code", "method_score", "baseline_edge", "baseline_compound", "bh_full"]
            ]
            .head(10)
            .to_string(index=False),
            flush=True,
        )
    tk = df[df["gated"] & (df["baseline_edge"] < 0)].sort_values("baseline_edge")
    print(f"\ntrue_blocks (gated & hold_up lost to BH): n={len(tk)}", flush=True)
    if not tk.empty:
        print(
            tk[
                ["code", "method_score", "baseline_edge", "baseline_compound", "bh_full"]
            ]
            .head(10)
            .to_string(index=False),
            flush=True,
        )

    print(f"\n[VERDICT] {verdict}", flush=True)
    print(f"[OK] wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
