#!/usr/bin/env python3
"""A/B: C17 shallow_up trade overlay vs production finalize (no production commit).

Design: huangtuo-master-design-20260813-104222-c17-shallow-up-trade.md

baseline
    Production finalize only.
shallow_up
    recovery_up_mask → range→up only for HRP mode==shallow_up (C17).
    Grid p20_min in {0, 1.5%, 3%}; N=2 locked. Pick first combo that
    passes focus delay then best C17/pool edge.
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
from decision_pack.src.hrp_cluster_router import (  # noqa: E402
    DEFAULT_MEMBERSHIP_CSV,
    code_mode_map,
    load_cluster_membership,
    mode_allows_trade_overlay,
    paint_cluster_modes,
    router_snapshot,
    shallow_up_codes,
)
from decision_pack.src.recovery_up_overlay import (  # noqa: E402
    KEEP,
    SHALLOW_FLIP_DATE,
    SHALLOW_FLIP_DEADLINE,
    SHALLOW_FOCUS_CODE,
    SHALLOW_N,
    SHALLOW_P20_GRID,
    SHALLOW_PEER_CODE,
    SHALLOW_TROUGH_DATE,
    apply_recovery_up,
    decide_recommendation,
    ed_alert_bool_from_frame,
    first_up_date,
    latch_recovery_up_mask,
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


def _pool_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = int(sum(int(r["n_closed"]) for r in rows))
    wins = int(sum(int(r["wins"]) for r in rows))
    return {
        "n_closed": n,
        "wins": wins,
        "trade_win": float(wins / n) if n else float("nan"),
        "traded_symbols": int(sum(1 for r in rows if int(r["n_closed"]) > 0)),
        "mean_edge": float(np.nanmean([r["edge"] for r in rows])),
        "mean_ret_net": float(np.nanmean([r["mean_ret_net"] for r in rows])),
        "worst_drawdown": float(np.nanmin([r["max_drawdown"] for r in rows])),
    }


def _pool_gates(ov: dict[str, Any], b: dict[str, Any]) -> dict[str, bool]:
    return {
        "win_improved_or_equal": (
            ov["trade_win"] >= b["trade_win"] - 1e-9
            if np.isfinite(ov["trade_win"]) and np.isfinite(b["trade_win"])
            else (not np.isfinite(b["trade_win"]))
        ),
        "enough_trades": ov["n_closed"] >= max(0, int(0.8 * b["n_closed"])),
        "mean_net_noninferior": ov["mean_ret_net"] >= b["mean_ret_net"] - 1e-6,
        "edge_noninferior": ov["mean_edge"] >= b["mean_edge"] - 1e-6,
        "drawdown_controlled": ov["worst_drawdown"] >= b["worst_drawdown"] - 0.02,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", default="2026-04-01")
    p.add_argument("--end-date", default="2026-08-12")
    p.add_argument("--train-cutoff", default="2026-04-01")
    p.add_argument("--membership-csv", type=Path, default=DEFAULT_MEMBERSHIP_CSV)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260812_c17_shallow_up_ab",
    )
    p.add_argument("--code", action="append", default=[])
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--N", type=int, default=SHALLOW_N)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mem = load_cluster_membership(args.membership_csv)
    cluster_mode = paint_cluster_modes(mem)
    mode_by_code = code_mode_map(mem, cluster_mode)
    su_codes = shallow_up_codes(mode_by_code)
    snap = router_snapshot(mem, source_path=str(args.membership_csv.resolve()))

    pm = _load(
        "pm_c17_ab",
        _ROOT / "decision_pack/scripts/plot_regime_transition_example.py",
    )
    pm._init_qlib(None)
    ev = _load(
        "ev_c17_ab",
        _ROOT / "decision_pack/scripts/eval_causal_regime_switch_pool.py",
    )

    codes = [str(x).upper() for x in args.code] or list(
        load_cluster_mapping_codes(CLUSTER_MAPPING_SELECTED_TXT)
    )
    if args.max_codes:
        codes = codes[: int(args.max_codes)]
    missing = sorted(su_codes - set(codes))
    if missing and not args.code:
        print(f"[warn] shallow_up codes not in universe: {missing}")

    # Cache per-code finalize + ED once
    cache: dict[str, dict[str, Any]] = {}
    for i, code in enumerate(codes, 1):
        mode = mode_by_code.get(code, "default")
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
        cache[code] = dict(
            px=px,
            labs0=labs0,
            method=method,
            mode=mode,
            ed_bool=ed_bool if ed_ok else None,
            ed_ok=ed_ok,
        )
        print(f"[cache {i}/{len(codes)}] {code} mode={mode} method={method}")

    grid_rows: list[dict[str, Any]] = []
    chosen: dict[str, Any] | None = None

    for p20 in SHALLOW_P20_GRID:
        per_base: list[dict[str, Any]] = []
        per_ov: list[dict[str, Any]] = []
        focus_detail: dict[str, Any] = {}
        peer_detail: dict[str, Any] = {}
        n_label_diff = 0
        n_non_su_diff = 0

        for code, c in cache.items():
            px, labs0, mode = c["px"], c["labs0"], c["mode"]
            mask = recovery_up_mask(
                px,
                labs0,
                c["ed_bool"],
                N=int(args.N),
                p20_min=float(p20),
            )
            if c["ed_bool"] is not None:
                mask = latch_recovery_up_mask(
                    px, labs0, mask, c["ed_bool"]
                )
            gated = mask if mode_allows_trade_overlay(mode) else np.zeros(len(px), dtype=bool)
            labs_ov = apply_recovery_up(labs0, gated)
            audit = (px["as_of"] >= args.start_date) & (px["as_of"] <= args.end_date)
            audit_idx = audit.to_numpy()
            close = px.loc[audit, "$close"].to_numpy(float)
            dates = px.loc[audit, "as_of"].dt.strftime("%Y-%m-%d").to_numpy()
            diff = int(np.sum(labs0[audit_idx] != labs_ov[audit_idx]))
            n_label_diff += diff
            if code not in su_codes:
                n_non_su_diff += diff
            for arm, labs in (("baseline", labs0), ("shallow_up", labs_ov)):
                m, _, _ = hold_up_trade_metrics(labs[audit_idx], close, dates=dates)
                row = {
                    "code": code,
                    "variant": arm,
                    "method": c["method"],
                    "hrp_mode": mode,
                    "p20_min": float(p20),
                    "mask_frac": float(gated[audit_idx].mean()),
                    **m,
                }
                (per_base if arm == "baseline" else per_ov).append(row)

            if code == SHALLOW_FOCUS_CODE:
                flip = first_up_date(
                    px["as_of"],
                    labs_ov,
                    start=SHALLOW_FLIP_DATE,
                    end=args.end_date,
                )
                delay = (
                    trading_days_between(px["as_of"], SHALLOW_FLIP_DATE, flip)
                    if flip
                    else None
                )
                focus_detail = {
                    "first_up": flip,
                    "delay_td": delay,
                    "up_frac_post_flip": up_frac_in_window(
                        px["as_of"], labs_ov, SHALLOW_FLIP_DATE, args.end_date
                    ),
                    "baseline_up_frac_post_flip": up_frac_in_window(
                        px["as_of"], labs0, SHALLOW_FLIP_DATE, args.end_date
                    ),
                    "up_frac_post_trough": up_frac_in_window(
                        px["as_of"], labs_ov, SHALLOW_TROUGH_DATE, args.end_date
                    ),
                }
            if code == SHALLOW_PEER_CODE:
                peer_flip = first_up_date(
                    px["as_of"],
                    labs_ov,
                    start=SHALLOW_FLIP_DATE,
                    end=args.end_date,
                )
                peer_detail = {
                    "first_up": peer_flip,
                    "up_frac_post_flip": up_frac_in_window(
                        px["as_of"], labs_ov, SHALLOW_FLIP_DATE, args.end_date
                    ),
                    "baseline_up_frac_post_flip": up_frac_in_window(
                        px["as_of"], labs0, SHALLOW_FLIP_DATE, args.end_date
                    ),
                }

        b = _pool_from_rows(per_base)
        ov = _pool_from_rows(per_ov)
        c17_b = _pool_from_rows([r for r in per_base if r["code"] in su_codes])
        c17_ov = _pool_from_rows([r for r in per_ov if r["code"] in su_codes])
        non_b = _pool_from_rows([r for r in per_base if r["code"] not in su_codes])
        non_ov = _pool_from_rows([r for r in per_ov if r["code"] not in su_codes])

        delay = focus_detail.get("delay_td")
        up_frac = float(focus_detail.get("up_frac_post_flip", float("nan")))
        focus_ok = bool(np.isfinite(up_frac) and up_frac >= 0.80)
        flip_ok = delay is not None and int(delay) <= 0
        peer_up = float(peer_detail.get("up_frac_post_flip", float("nan")))
        peer_base = float(peer_detail.get("baseline_up_frac_post_flip", float("nan")))
        peer_ok = bool(
            np.isfinite(peer_up)
            and (
                (np.isfinite(peer_base) and peer_up > peer_base + 1e-9)
                or peer_up > 1e-9
            )
        )
        non_eq = n_non_su_diff == 0
        gates_pool = _pool_gates(ov, b)
        gates_c17 = _pool_gates(c17_ov, c17_b)
        # C17 may have 0 baseline trades (all range) → enough_trades / win NaN
        if not np.isfinite(c17_b["trade_win"]):
            gates_c17["win_improved_or_equal"] = True
        if c17_b["n_closed"] == 0:
            gates_c17["enough_trades"] = True
        pool_ok = all(gates_pool.values())
        c17_ok = all(gates_c17.values())
        rec = decide_recommendation(
            pool_ok and c17_ok and non_eq and peer_ok, focus_ok, flip_ok
        )
        row = {
            "p20_min": float(p20),
            "N": int(args.N),
            "recommendation": rec,
            "focus_ok": focus_ok,
            "flip_ok": flip_ok,
            "peer_ok": peer_ok,
            "non_c17_labels_identical": non_eq,
            "n_non_su_diff": n_non_su_diff,
            "n_label_diff": n_label_diff,
            "pool_ok": pool_ok,
            "c17_ok": c17_ok,
            "focus": focus_detail,
            "peer": peer_detail,
            "baseline_pool": b,
            "overlay_pool": ov,
            "c17_baseline": c17_b,
            "c17_overlay": c17_ov,
            "non_c17_baseline": non_b,
            "non_c17_overlay": non_ov,
            "gates_pool": gates_pool,
            "gates_c17": gates_c17,
        }
        grid_rows.append(row)
        print(
            f"[grid] p20_min={p20:.3f} rec={rec} "
            f"first_up={focus_detail.get('first_up')} delay={delay} "
            f"up_frac={up_frac:.1%} peer_up={peer_up:.1%} "
            f"pool_edge {b['mean_edge']:+.2%}→{ov['mean_edge']:+.2%}"
        )
        if rec != KEEP and chosen is None:
            chosen = {**row, "per_ov": per_ov, "per_base": per_base}

    # If none promoted, keep the tightest p20 that at least hits flip (for diagnostics)
    if chosen is None:
        for row in grid_rows:
            if row["flip_ok"] and row["non_c17_labels_identical"]:
                chosen = row
                break
        if chosen is None:
            chosen = grid_rows[0]

    rec_final = chosen["recommendation"] if chosen.get("focus_ok") else KEEP
    # Recompute: only promote if the stored recommendation is PROMOTE
    rec_final = str(chosen.get("recommendation") or KEEP)

    pd.DataFrame(
        [
            {
                k: (json.dumps(v, default=str) if isinstance(v, dict) else v)
                for k, v in r.items()
            }
            for r in grid_rows
        ]
    ).to_csv(out_dir / "grid.csv", index=False, encoding="utf-8-sig")

    summary = {
        "design": "huangtuo-master-design-20260813-104222-c17-shallow-up-trade.md",
        "selection": f"B0 once on train < {args.train_cutoff}",
        "audit": f"{args.start_date}..{args.end_date}",
        "n_symbols": len(codes),
        "shallow_up_codes": sorted(su_codes),
        "hrp_router": snap,
        "N": int(args.N),
        "p20_grid": list(SHALLOW_P20_GRID),
        "trough": SHALLOW_TROUGH_DATE,
        "flip_date": SHALLOW_FLIP_DATE,
        "flip_deadline": SHALLOW_FLIP_DEADLINE,
        "chosen_p20_min": chosen["p20_min"],
        "recommendation": rec_final,
        "chosen": {k: v for k, v in chosen.items() if k not in ("per_ov", "per_base")},
        "grid": [
            {k: v for k, v in r.items() if k not in ("per_ov", "per_base")}
            for r in grid_rows
        ],
        "production_modified": False,
    }
    (out_dir / "acceptance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    lines = [
        "# C17 shallow_up A/B",
        "",
        f"- Membership: `{args.membership_csv}`",
        f"- shallow_up codes: `{sorted(su_codes)}`",
        f"- Chosen p20_min={chosen['p20_min']} N={args.N}",
        f"- Recommendation: `{rec_final}`",
        f"- Focus first_up: `{chosen.get('focus', {}).get('first_up')}`",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("DONE recommend=", rec_final, "p20_min=", chosen["p20_min"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
