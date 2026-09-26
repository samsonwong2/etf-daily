#!/usr/bin/env python3
"""Small param grid for recovery_up overlay (expression fixed).

Sweeps N / p20_min / R with rec_min=R. Reuses per-code finalize+ED once.
Pool non-inferiority remains the promote primary; focus is confirmatory.
Does not modify production defaults/constants unless a combo promotes and
you explicitly promote later.

Design TODO-3 / eng plan follow-up.
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
    DEFAULT_D_FLIP,
    DEFAULT_K_ED,
    DEFAULT_K_PIT,
    DEFAULT_L,
    FOCUS_CODE,
    FOCUS_WINDOW_END,
    FOCUS_WINDOW_START,
    HUMAN_FLIP_DATE,
    KEEP,
    PROMOTE,
    apply_recovery_up,
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


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pool_from_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = int(sum(int(r["n_closed"]) for r in rows))
    wins = int(sum(int(r["wins"]) for r in rows))
    traded = int(sum(1 for r in rows if int(r["n_closed"]) > 0))
    return {
        "n_closed": n,
        "wins": wins,
        "trade_win": float(wins / n) if n else float("nan"),
        "traded_symbols": traded,
        "mean_edge": float(np.nanmean([r["edge"] for r in rows])),
        "mean_ret_net": float(np.nanmean([r["mean_ret_net"] for r in rows])),
        "worst_drawdown": float(np.nanmin([r["max_drawdown"] for r in rows])),
        "mean_mask_frac": float(np.nanmean([r["mask_frac"] for r in rows])),
    }


def _pool_gates(ov: dict[str, Any], b: dict[str, Any]) -> dict[str, bool]:
    return {
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", default="2026-04-01")
    p.add_argument("--end-date", default=FOCUS_WINDOW_END)
    p.add_argument("--train-cutoff", default="2026-04-01")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260811_recovery_up_overlay_grid",
    )
    p.add_argument("--code", action="append", default=[])
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--N", default="2,3,5")
    p.add_argument("--p20-min", default="0.02,0.03,0.05")
    p.add_argument("--R", default="0.06,0.08,0.10")
    return p.parse_args()


def _parse_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_ints(s: str) -> list[int]:
    return [int(float(x.strip())) for x in s.split(",") if x.strip()]


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    Ns = _parse_ints(args.N)
    p20s = _parse_floats(args.p20_min)
    Rs = _parse_floats(args.R)
    combos = [
        {"N": N, "p20_min": p20, "R": R, "rec_min": R}
        for N, p20, R in itertools.product(Ns, p20s, Rs)
    ]

    pm = _load(
        "pm_recovery_grid",
        _ROOT / "decision_pack/scripts/plot_regime_transition_example.py",
    )
    pm._init_qlib(None)
    ev = _load(
        "ev_recovery_grid",
        _ROOT / "decision_pack/scripts/eval_causal_regime_switch_pool.py",
    )

    codes = [str(x).upper() for x in args.code] or list(
        load_cluster_mapping_codes(CLUSTER_MAPPING_SELECTED_TXT)
    )
    if args.max_codes:
        codes = codes[: int(args.max_codes)]

    # Per-code frozen artifacts (select + finalize + ED once)
    cache: list[dict[str, Any]] = []
    for i, code in enumerate(codes, 1):
        ohlcv = pm.load_qlib_ohlcv(
            code,
            "2015-01-01",
            args.end_date,
            init_qlib=False,
            lookback_calendar_days=400,
            clip_to_window=False,
        )
        px = prepare_features(ev, ohlcv, method=None)
        px_train = px[px["as_of"] < args.train_cutoff].reset_index(drop=True)
        method, _, _ = select_regime_method_legacy(
            ev, px_train, trade_mode=TRADE_MODE_HOLD_UP
        )
        labs0 = np.asarray(ev.run_method(method, px), dtype=object)
        try:
            ed_frame = early_down_flags_frame(px, regimes=labs0)
            ed_bool, ed_ok = ed_alert_bool_from_frame(ed_frame, len(px))
        except Exception as exc:  # noqa: BLE001
            print(f"[ed-fail] {code}: {exc}")
            ed_bool, ed_ok = None, False
        audit = (px["as_of"] >= args.start_date) & (px["as_of"] <= args.end_date)
        cache.append(
            {
                "code": code,
                "method": method,
                "px": px,
                "labs0": labs0,
                "ed_bool": ed_bool if ed_ok else None,
                "ed_ok": bool(ed_ok),
                "audit": audit.to_numpy(),
                "close_audit": px.loc[audit, "$close"].to_numpy(float),
                "dates_audit": px.loc[audit, "as_of"].dt.strftime("%Y-%m-%d").to_numpy(),
            }
        )
        print(f"[prep {i}/{len(codes)}] {code} -> {method} ed_ok={ed_ok}")

    # Baseline pool once
    base_rows: list[dict[str, Any]] = []
    for c in cache:
        m, _, _ = hold_up_trade_metrics(
            c["labs0"][c["audit"]], c["close_audit"], dates=c["dates_audit"]
        )
        base_rows.append(
            {
                "code": c["code"],
                "mask_frac": 0.0,
                **m,
            }
        )
    baseline_pool = _pool_from_metrics(base_rows)

    grid_rows: list[dict[str, Any]] = []
    for ci, params in enumerate(combos, 1):
        ov_rows: list[dict[str, Any]] = []
        focus: dict[str, Any] = {}
        for c in cache:
            mask = recovery_up_mask(
                c["px"],
                c["labs0"],
                c["ed_bool"],
                L=DEFAULT_L,
                k=DEFAULT_K_PIT,
                N=int(params["N"]),
                R=float(params["R"]),
                rec_min=float(params["rec_min"]),
                p20_min=float(params["p20_min"]),
                k_ed=DEFAULT_K_ED,
            )
            labs = apply_recovery_up(c["labs0"], mask)
            m, _, _ = hold_up_trade_metrics(
                labs[c["audit"]], c["close_audit"], dates=c["dates_audit"]
            )
            ov_rows.append(
                {
                    "code": c["code"],
                    "mask_frac": float(mask[c["audit"]].mean()),
                    **m,
                }
            )
            if c["code"] == FOCUS_CODE:
                flip = first_up_date(
                    c["px"]["as_of"],
                    labs,
                    start=HUMAN_FLIP_DATE,
                    end=FOCUS_WINDOW_END,
                )
                delay = (
                    trading_days_between(c["px"]["as_of"], HUMAN_FLIP_DATE, flip)
                    if flip
                    else None
                )
                focus = {
                    "overlay_up_frac_post_flip": up_frac_in_window(
                        c["px"]["as_of"], labs, HUMAN_FLIP_DATE, FOCUS_WINDOW_END
                    ),
                    "overlay_up_frac_full_window": up_frac_in_window(
                        c["px"]["as_of"],
                        labs,
                        FOCUS_WINDOW_START,
                        FOCUS_WINDOW_END,
                    ),
                    "first_overlay_up": flip,
                    "flip_delay_td": delay,
                    "mask_frac_audit": float(mask[c["audit"]].mean()),
                }

        ov_pool = _pool_from_metrics(ov_rows)
        gates = _pool_gates(ov_pool, baseline_pool)
        pool_ok = all(gates.values())
        up_frac = float(focus.get("overlay_up_frac_post_flip", float("nan")))
        delay = focus.get("flip_delay_td")
        focus_ok = bool(np.isfinite(up_frac) and up_frac >= 0.80)
        flip_ok = delay is not None and int(delay) <= int(DEFAULT_D_FLIP)
        rec = decide_recommendation(pool_ok, focus_ok, flip_ok)
        row = {
            **params,
            **{f"ov_{k}": ov_pool[k] for k in ov_pool},
            **{f"gate_{k}": gates[k] for k in gates},
            "pool_ok": pool_ok,
            "focus_ok": focus_ok,
            "flip_ok": flip_ok,
            "recommendation": rec,
            **{f"focus_{k}": focus.get(k) for k in focus},
            # ranking helpers (do not replace gates)
            "delta_edge": float(ov_pool["mean_edge"] - baseline_pool["mean_edge"]),
            "delta_net": float(ov_pool["mean_ret_net"] - baseline_pool["mean_ret_net"]),
            "delta_win": float(ov_pool["trade_win"] - baseline_pool["trade_win"])
            if np.isfinite(ov_pool["trade_win"]) and np.isfinite(baseline_pool["trade_win"])
            else float("nan"),
        }
        grid_rows.append(row)
        print(
            f"[grid {ci}/{len(combos)}] N={params['N']} p20={params['p20_min']} R={params['R']} "
            f"-> {rec} pool={pool_ok} focus={focus_ok} flip={flip_ok} "
            f"up={up_frac if np.isfinite(up_frac) else float('nan'):.2%} "
            f"d_edge={row['delta_edge']:+.3%}"
        )

    gdf = pd.DataFrame(grid_rows)
    gdf.to_csv(out_dir / "grid_results.csv", index=False, encoding="utf-8-sig")

    promote = gdf[gdf["recommendation"] == PROMOTE].copy()
    if len(promote):
        # Prefer higher edge among promoters
        promote = promote.sort_values(
            ["delta_edge", "focus_overlay_up_frac_post_flip"],
            ascending=[False, False],
        )
        best = promote.iloc[0].to_dict()
        best_label = PROMOTE
    else:
        # Best effort: pool_ok first, then focus/flip, then least edge damage
        scored = gdf.copy()
        scored["_rank_pool"] = scored["pool_ok"].astype(int)
        scored["_rank_focus"] = scored["focus_ok"].astype(int)
        scored["_rank_flip"] = scored["flip_ok"].astype(int)
        scored = scored.sort_values(
            ["_rank_pool", "_rank_focus", "_rank_flip", "delta_edge", "focus_overlay_up_frac_post_flip"],
            ascending=[False, False, False, False, False],
        )
        best = scored.iloc[0].to_dict()
        best_label = KEEP

    summary = {
        "baseline_pool": baseline_pool,
        "n_combos": len(combos),
        "n_symbols": len(codes),
        "fixed": {
            "L": DEFAULT_L,
            "k": DEFAULT_K_PIT,
            "k_ed": DEFAULT_K_ED,
            "D_flip": DEFAULT_D_FLIP,
            "HUMAN_FLIP_DATE": HUMAN_FLIP_DATE,
            "rec_min": "=R",
        },
        "grid": {"N": Ns, "p20_min": p20s, "R": Rs},
        "n_promote": int((gdf["recommendation"] == PROMOTE).sum()),
        "best_label": best_label,
        "best": {
            k: best[k]
            for k in best
            if not str(k).startswith("_")
        },
        "note": (
            "If best_label!=promote_candidate, do not change production constants; "
            "optional: pick least-damaging combo for further research only."
        ),
    }
    (out_dir / "grid_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    lines = [
        "# recovery_up param grid",
        "",
        f"- Combos: {len(combos)} (N={Ns}, p20_min={p20s}, R={Rs}, rec_min=R)",
        f"- Symbols: {len(codes)}",
        f"- Promote count: {summary['n_promote']}",
        f"- Best label: `{best_label}`",
        "",
        "## Best row",
        "",
        "```json",
        json.dumps(summary["best"], ensure_ascii=False, indent=2, default=str),
        "```",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("DONE best_label=", best_label, "n_promote=", summary["n_promote"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
