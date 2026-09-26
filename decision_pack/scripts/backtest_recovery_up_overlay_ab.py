#!/usr/bin/env python3
"""A/B: recovery_up post-finalize overlay vs production finalize (no production commit).

Variants
--------
baseline
    Production finalize labels only.
recovery_alert_only
    Same trade labels as baseline; write recovery mask for diagnostics.
recovery_overlay
    apply_recovery_up after finalize (range→up where causal mask).

Selection: B0 once per code on train < cutoff with **no** overlay.
Design: ~/.gstack/projects/wongtoe-skfolio_csi300/huangtuo-master-design-20260812-161837-recovery-up-overlay.md
Eng:    ~/.gstack/projects/wongtoe-skfolio_csi300/huangtuo-master-eng-plan-20260812-165855-recovery-up-overlay.md
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from decision_pack.src.adaptive_stage_common import (  # noqa: E402
    TRADE_MODE_HOLD_UP,
    prepare_features,
    select_regime_method_legacy,
)
from decision_pack.src.early_down_alert import early_down_flags_frame  # noqa: E402
from decision_pack.src.recovery_up_overlay import (  # noqa: E402
    FOCUS_CODE,
    FOCUS_WINDOW_END,
    FOCUS_WINDOW_START,
    HUMAN_FLIP_DATE,
    KEEP,
    PROMOTE,
    apply_recovery_up,
    constants_snapshot,
    decide_recommendation,
    ed_alert_bool_from_frame,
    first_up_date,
    recovery_up_mask,
    trading_days_between,
    up_frac_in_window,
)
from decision_pack.src.state_space_selection import hold_up_trade_metrics  # noqa: E402
from decision_pack.src.symbol_trend_board import load_cluster_mapping_codes  # noqa: E402
from runtime_paths import CLUSTER_MAPPING_SELECTED_TXT, PLOTLY_OUTPUTS_DIR  # noqa: E402

VARIANTS = ("baseline", "recovery_alert_only", "recovery_overlay")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pool(rows: pd.DataFrame, prefix: str) -> dict[str, Any]:
    n = int(rows[f"{prefix}_n_closed"].sum())
    wins = int(rows[f"{prefix}_wins"].sum())
    traded = int((rows[f"{prefix}_n_closed"] > 0).sum())
    return {
        "n_closed": n,
        "wins": wins,
        "trade_win": float(wins / n) if n else float("nan"),
        "traded_symbols": traded,
        "zero_trade_rate": float(1.0 - traded / len(rows)) if len(rows) else 1.0,
        "mean_edge": float(pd.to_numeric(rows[f"{prefix}_edge"], errors="coerce").mean()),
        "mean_ret_net": float(
            pd.to_numeric(rows[f"{prefix}_mean_ret_net"], errors="coerce").mean()
        ),
        "mean_compound": float(
            pd.to_numeric(rows[f"{prefix}_compound"], errors="coerce").mean()
        ),
        "worst_drawdown": float(
            pd.to_numeric(rows[f"{prefix}_max_drawdown"], errors="coerce").min()
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", default="2026-04-01")
    p.add_argument("--end-date", default=FOCUS_WINDOW_END)
    p.add_argument("--train-cutoff", default="2026-04-01")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260811_recovery_up_overlay_ab",
    )
    p.add_argument("--code", action="append", default=[])
    p.add_argument("--max-codes", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = constants_snapshot()

    pm = _load(
        "pm_recovery",
        _ROOT / "decision_pack/scripts/plot_regime_transition_example.py",
    )
    pm._init_qlib(None)
    ev = _load(
        "ev_recovery",
        _ROOT / "decision_pack/scripts/eval_causal_regime_switch_pool.py",
    )

    codes = [str(x).upper() for x in args.code] or list(
        load_cluster_mapping_codes(CLUSTER_MAPPING_SELECTED_TXT)
    )
    if args.max_codes:
        codes = codes[: int(args.max_codes)]

    ohlcvs: dict[str, pd.DataFrame] = {}
    for i, code in enumerate(codes, 1):
        ohlcvs[code] = pm.load_qlib_ohlcv(
            code,
            "2015-01-01",
            args.end_date,
            init_qlib=False,
            lookback_calendar_days=400,
            clip_to_window=False,
        )
        print(f"[load {i}/{len(codes)}] {code}")

    per_variant_rows: dict[str, list[dict[str, Any]]] = {v: [] for v in VARIANTS}
    focus_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []

    for i, code in enumerate(codes, 1):
        px = prepare_features(ev, ohlcvs[code], method=None)
        px_train = px[px["as_of"] < args.train_cutoff].reset_index(drop=True)
        method, _, _meta = select_regime_method_legacy(
            ev, px_train, trade_mode=TRADE_MODE_HOLD_UP
        )
        labs0 = np.asarray(ev.run_method(method, px), dtype=object)

        try:
            ed_frame = early_down_flags_frame(px, regimes=labs0)
            ed_bool, ed_ok = ed_alert_bool_from_frame(ed_frame, len(px))
        except Exception as exc:  # noqa: BLE001 — fail-closed
            print(f"[ed-fail] {code}: {exc}")
            ed_bool, ed_ok = None, False

        mask = recovery_up_mask(px, labs0, ed_bool if ed_ok else None)
        labs_ov = apply_recovery_up(labs0, mask)

        audit = (px["as_of"] >= args.start_date) & (px["as_of"] <= args.end_date)
        close = px.loc[audit, "$close"].to_numpy(float)
        dates = px.loc[audit, "as_of"].dt.strftime("%Y-%m-%d").to_numpy()
        audit_idx = audit.to_numpy()

        arm_labs = {
            "baseline": labs0,
            "recovery_alert_only": labs0,
            "recovery_overlay": labs_ov,
        }
        for mode, labs in arm_labs.items():
            m, _trades, _ = hold_up_trade_metrics(
                labs[audit_idx], close, dates=dates
            )
            row = {
                "code": code,
                "variant": mode,
                "method": method,
                "ed_ok": bool(ed_ok),
                "up_frac": float((labs[audit_idx] == "up").mean()),
                "recovery_mask_frac_audit": float(mask[audit_idx].mean()),
                **m,
            }
            per_variant_rows[mode].append(row)

        # diagnostic mask dump (alert_only arm intent)
        as_of_s = px["as_of"].dt.strftime("%Y-%m-%d")
        for j in np.flatnonzero(mask):
            mask_rows.append(
                {
                    "code": code,
                    "as_of": as_of_s.iloc[int(j)],
                    "native_regime": str(labs0[int(j)]),
                    "overlay_regime": str(labs_ov[int(j)]),
                }
            )

        if code == FOCUS_CODE:
            flip = first_up_date(
                px["as_of"],
                labs_ov,
                start=HUMAN_FLIP_DATE,
                end=FOCUS_WINDOW_END,
            )
            delay = (
                trading_days_between(px["as_of"], HUMAN_FLIP_DATE, flip)
                if flip
                else None
            )
            focus_rows.append(
                {
                    "code": code,
                    "method": method,
                    "ed_ok": bool(ed_ok),
                    "baseline_up_frac_post_flip": up_frac_in_window(
                        px["as_of"],
                        labs0,
                        HUMAN_FLIP_DATE,
                        FOCUS_WINDOW_END,
                    ),
                    "overlay_up_frac_post_flip": up_frac_in_window(
                        px["as_of"],
                        labs_ov,
                        HUMAN_FLIP_DATE,
                        FOCUS_WINDOW_END,
                    ),
                    "overlay_up_frac_full_window": up_frac_in_window(
                        px["as_of"],
                        labs_ov,
                        FOCUS_WINDOW_START,
                        FOCUS_WINDOW_END,
                    ),
                    "first_overlay_up": flip,
                    "flip_delay_td": delay,
                    "human_flip_date": HUMAN_FLIP_DATE,
                    "recovery_mask_frac_audit": float(mask[audit_idx].mean()),
                }
            )

        print(
            f"[{i}/{len(codes)}] {code} -> {method} "
            f"mask={mask.mean():.2%} ed_ok={ed_ok}"
        )

    # --- wide symbol table ---
    base = pd.DataFrame(per_variant_rows["baseline"]).set_index("code")
    wide = base.add_prefix("baseline_").reset_index()
    for mode in VARIANTS[1:]:
        df = pd.DataFrame(per_variant_rows[mode]).set_index("code")
        wide = wide.merge(
            df.add_prefix(f"{mode}_").reset_index(), on="code", how="outer"
        )
    wide.to_csv(out_dir / "symbol_comparison.csv", index=False, encoding="utf-8-sig")

    pool_rows = []
    for mode in VARIANTS:
        df = pd.DataFrame(per_variant_rows[mode])
        tmp = df.rename(
            columns={
                "n_closed": f"{mode}_n_closed",
                "wins": f"{mode}_wins",
                "edge": f"{mode}_edge",
                "mean_ret_net": f"{mode}_mean_ret_net",
                "compound": f"{mode}_compound",
                "max_drawdown": f"{mode}_max_drawdown",
            }
        )
        s = _pool(tmp, mode)
        s["variant"] = mode
        s["method_mix"] = df["method"].value_counts().to_dict()
        s["mean_recovery_mask_frac"] = float(df["recovery_mask_frac_audit"].mean())
        s["ed_ok_rate"] = float(df["ed_ok"].mean())
        pool_rows.append(s)
    pd.DataFrame(pool_rows).to_csv(
        out_dir / "pool_comparison.csv", index=False, encoding="utf-8-sig"
    )

    focus_df = pd.DataFrame(focus_rows)
    focus_df.to_csv(out_dir / "focus_SH515220.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mask_rows).to_csv(
        out_dir / "recovery_up_marks.csv", index=False, encoding="utf-8-sig"
    )

    b = next(s for s in pool_rows if s["variant"] == "baseline")
    ov = next(s for s in pool_rows if s["variant"] == "recovery_overlay")
    gates = {
        "win_improved_or_equal": (
            ov["trade_win"] >= b["trade_win"] - 1e-9
            if np.isfinite(ov["trade_win"]) and np.isfinite(b["trade_win"])
            else False
        ),
        "enough_trades": ov["n_closed"] >= max(10, int(0.8 * b["n_closed"])),
        "mean_net_noninferior": ov["mean_ret_net"] >= b["mean_ret_net"] - 1e-6,
        "edge_noninferior": ov["mean_edge"] >= b["mean_edge"] - 1e-6,
        "drawdown_controlled": ov["worst_drawdown"] >= b["worst_drawdown"] - 0.02,
    }
    pool_ok = all(gates.values())

    focus_ok = False
    flip_ok = False
    focus_detail: dict[str, Any] = {}
    if len(focus_df):
        fr = focus_df.iloc[0].to_dict()
        focus_detail = fr
        up_frac = float(fr.get("overlay_up_frac_post_flip", float("nan")))
        delay = fr.get("flip_delay_td")
        focus_ok = bool(np.isfinite(up_frac) and up_frac >= 0.80)
        flip_ok = delay is not None and int(delay) <= int(snap["D_flip"])
        gates["focus_up_frac_ge_0_80"] = focus_ok
        gates["flip_delay_le_D"] = flip_ok
    else:
        gates["focus_up_frac_ge_0_80"] = False
        gates["flip_delay_le_D"] = False

    recommendation = decide_recommendation(pool_ok, focus_ok, flip_ok)

    summary = {
        "selection": f"B0 once on train < {args.train_cutoff} (no overlay)",
        "audit": f"{args.start_date}..{args.end_date}",
        "n_symbols": len(codes),
        "constants": snap,
        "design": "huangtuo-master-design-20260812-161837-recovery-up-overlay.md",
        "eng_plan": "huangtuo-master-eng-plan-20260812-165855-recovery-up-overlay.md",
        "baseline": {
            k: b[k]
            for k in (
                "trade_win",
                "n_closed",
                "mean_edge",
                "mean_ret_net",
                "worst_drawdown",
            )
        },
        "recovery_overlay": {
            k: ov[k]
            for k in (
                "trade_win",
                "n_closed",
                "mean_edge",
                "mean_ret_net",
                "worst_drawdown",
                "mean_recovery_mask_frac",
                "ed_ok_rate",
            )
        },
        "gates": gates,
        "pool_ok": pool_ok,
        "focus_ok": focus_ok,
        "flip_ok": flip_ok,
        "focus": focus_detail,
        "recommendation": recommendation,
        "production_modified": False,
    }
    (out_dir / "acceptance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    lines = [
        "# recovery_up overlay A/B",
        "",
        f"- Train/select: `< {args.train_cutoff}` (B0, hold_up, **no overlay**).",
        f"- Audit: `{args.start_date}` → `{args.end_date}`.",
        f"- human_flip_date: `{HUMAN_FLIP_DATE}`.",
        "- Production code **not** modified.",
        "",
        "## Pool",
        "",
        "| Variant | Trade win | n | Mean edge | Mean net | Worst DD | Mask frac |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in pool_rows:
        lines.append(
            f"| {s['variant']} | {s['trade_win']:.1%} | {s['n_closed']} | "
            f"{s['mean_edge']:+.2%} | {s['mean_ret_net']:+.2%} | "
            f"{s['worst_drawdown']:.1%} | {s['mean_recovery_mask_frac']:.1%} |"
        )
    lines += ["", f"## Recommendation: `{recommendation}`", ""]
    if focus_detail:
        lines += ["## Focus SH515220", "", "```json", json.dumps(focus_detail, ensure_ascii=False, indent=2, default=str), "```", ""]
    lines += ["## Gates", "", "```json", json.dumps(gates, ensure_ascii=False, indent=2), "```", ""]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print("DONE recommend=", recommendation, "gates=", gates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
