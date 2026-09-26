#!/usr/bin/env python3
"""Per-code self-train policy (no pool-wide gate).

Each ticker independently picks among {hold_up, bh, cash} by **its own**
train-window compound (segment proxy). No single threshold is applied to the
whole pool. Production hold_up defaults are not changed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runtime_paths import PLOTLY_OUTPUTS_DIR  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--listing-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260814_from_listing",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260814_per_code_self_train",
    )
    p.add_argument("--train-cutoff", default="2026-04-01")
    return p.parse_args(argv)


def seg_compounds(seg: pd.DataFrame, mask: pd.Series) -> dict[str, float | int]:
    sub = seg[mask]
    if sub.empty:
        return {"hu": 0.0, "bh": 0.0, "cash": 0.0, "n_up": 0, "n_seg": 0, "bars": 0}
    bh = float((1.0 + sub["ret"]).prod() - 1.0)
    up = sub[sub["regime"] == "up"]
    hu = float((1.0 + up["ret"]).prod() - 1.0) if len(up) else 0.0
    return {
        "hu": hu,
        "bh": bh,
        "cash": 0.0,
        "n_up": int(len(up)),
        "n_seg": int(len(sub)),
        "bars": int(sub["n_bars"].sum()),
    }


def pick_policy(train: dict[str, float | int]) -> str:
    """Argmax train compound; ties prefer bh > hold_up > cash (less churn)."""
    cands = {
        "bh": float(train["bh"]),
        "hold_up": float(train["hu"]),
        "cash": float(train["cash"]),
    }
    rank = {"bh": 2, "hold_up": 1, "cash": 0}
    return max(rank, key=lambda k: (cands[k], rank[k]))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    listing = Path(args.listing_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cut = pd.Timestamp(args.train_cutoff)

    rows: list[dict[str, Any]] = []
    for path in sorted((listing / "configs").glob("*.json")):
        j = json.loads(path.read_text(encoding="utf-8"))
        code = str(j.get("code") or path.stem)
        score = j.get("method_score")
        try:
            score_f = float(score) if score is not None else float("nan")
        except (TypeError, ValueError):
            score_f = float("nan")
        oos_cfg = j.get("oos") or {}

        seg_path = listing / "trades" / f"{code}_segments.csv"
        try:
            seg = pd.read_csv(seg_path) if seg_path.exists() else pd.DataFrame()
        except Exception:
            seg = pd.DataFrame()
        if seg.empty:
            rows.append(
                {
                    "code": code,
                    "policy": "insufficient_data",
                    "method_score": score_f,
                }
            )
            continue

        seg = seg.copy()
        seg["start"] = pd.to_datetime(seg["start"])
        seg["end"] = pd.to_datetime(seg["end"])
        tr = seg_compounds(seg, seg["end"] < cut)
        oo = seg_compounds(seg, seg["end"] >= cut)
        policy = pick_policy(tr)

        oos_by = {"hold_up": float(oo["hu"]), "bh": float(oo["bh"]), "cash": float(oo["cash"])}
        oos_c = oos_by[policy]
        oos_edge = float(oos_c - float(oo["bh"]))
        oos_hu_edge = float(float(oo["hu"]) - float(oo["bh"]))

        full_hu = (
            float(oos_cfg["compound"])
            if oos_cfg.get("compound") is not None
            else float("nan")
        )
        full_bh = float(oos_cfg["bh"]) if oos_cfg.get("bh") is not None else float("nan")
        full_edge = (
            float(oos_cfg["edge"]) if oos_cfg.get("edge") is not None else float("nan")
        )
        full_policy_c = {"hold_up": full_hu, "bh": full_bh, "cash": 0.0}.get(
            policy, float("nan")
        )
        full_policy_e = (
            float(full_policy_c - full_bh)
            if np.isfinite(full_policy_c) and np.isfinite(full_bh)
            else float("nan")
        )

        rows.append(
            {
                "code": code,
                "method": j.get("method"),
                "method_score": score_f,
                "train_hold_up": tr["hu"],
                "train_bh": tr["bh"],
                "train_cash": 0.0,
                "train_edge_hu_vs_bh": float(tr["hu"]) - float(tr["bh"]),
                "policy": policy,
                "policy_reason": (
                    f"train argmax {{hold_up={tr['hu']:.4f}, "
                    f"bh={tr['bh']:.4f}, cash=0}}"
                ),
                "oos_hold_up": oo["hu"],
                "oos_bh": oo["bh"],
                "oos_policy_compound": oos_c,
                "oos_policy_edge_vs_bh": oos_edge,
                "oos_hold_up_edge_vs_bh": oos_hu_edge,
                "oos_lift_vs_forced_hold_up": float(oos_c - float(oo["hu"])),
                "full_window_hold_up": full_hu,
                "full_window_bh": full_bh,
                "full_window_hold_up_edge": full_edge,
                "full_window_policy_compound": full_policy_c,
                "full_window_policy_edge": full_policy_e,
                "train_bars": tr["bars"],
                "oos_bars": oo["bars"],
            }
        )

    df = pd.DataFrame(rows).sort_values(["policy", "code"])
    df.to_csv(out / "per_code_policy.csv", index=False)

    sub = df[
        np.isfinite(df.get("oos_policy_edge_vs_bh", np.nan))
        & np.isfinite(df.get("oos_hold_up_edge_vs_bh", np.nan))
    ]
    lift = sub["oos_lift_vs_forced_hold_up"] if not sub.empty else pd.Series(dtype=float)
    summary = {
        "n": int(len(df)),
        "policy_counts": df["policy"].value_counts().to_dict(),
        "n_switched_away_from_hold_up": int((df["policy"] != "hold_up").sum()),
        "oos_ew_forced_hold_up_edge": float(sub["oos_hold_up_edge_vs_bh"].mean())
        if len(sub)
        else None,
        "oos_ew_self_policy_edge": float(sub["oos_policy_edge_vs_bh"].mean())
        if len(sub)
        else None,
        "oos_ew_lift": float(lift.mean()) if len(lift) else None,
        "n_oos_improved": int((lift > 0).sum()) if len(lift) else 0,
        "n_oos_worse": int((lift < 0).sum()) if len(lift) else 0,
        "n_oos_same": int((lift == 0).sum()) if len(lift) else 0,
    }
    callouts = {}
    for code in ("SZ159985", "SH588710", "SH515220", "SH512690"):
        r = df[df["code"] == code]
        if not r.empty:
            callouts[code] = r.iloc[0].to_dict()

    payload = {
        "verdict": "PER_CODE_SELF_TRAIN_POLICY_TABLE",
        "summary": summary,
        "callouts": {
            k: {
                "policy": v.get("policy"),
                "train_hold_up": v.get("train_hold_up"),
                "train_bh": v.get("train_bh"),
                "oos_lift_vs_forced_hold_up": v.get("oos_lift_vs_forced_hold_up"),
            }
            for k, v in callouts.items()
        },
        "note": (
            "Each code picks hold_up/bh/cash from its own train only. "
            "No pool-wide standard. Production unchanged."
        ),
    }
    (out / "verdict.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (out / "README.md").write_text(
        """# Per-code self-train policy

Each ticker uses **only its own train window** to pick `hold_up` / `bh` / `cash`.
No pool-wide threshold. Production defaults unchanged.
""",
        encoding="utf-8",
    )

    print("policy counts:", summary["policy_counts"], flush=True)
    print(
        f"OOS EW edge forced_hold_up={summary['oos_ew_forced_hold_up_edge']} "
        f"self_policy={summary['oos_ew_self_policy_edge']} "
        f"lift={summary['oos_ew_lift']}",
        flush=True,
    )
    print(f"[OK] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
