#!/usr/bin/env python3
"""A/B: HRP-cluster-gated recovery_up vs production finalize (no production commit).

Arms
----
baseline
    Production finalize only.
hrp_shadow
    Membership/mode diagnostics; trade labels = baseline.
hrp_f2_overlay
    recovery_up only for codes whose HRP cluster mode == recovery_up.

Membership: cluster_representatives_{asof}.csv from generate_hrp_dendrogram_html.py
Design: huangtuo-master-design-20260812-171530-hrp-cluster-router.md
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
    MODE_RECOVERY_UP,
    code_mode_map,
    load_cluster_membership,
    mode_allows_diagnostic_marks,
    mode_allows_trade_overlay,
    paint_cluster_modes,
    recovery_up_codes,
    router_snapshot,
)
from decision_pack.src.recovery_up_overlay import (  # noqa: E402
    DIAGNOSTIC_N,
    DIAGNOSTIC_PROMOTE,
    FOCUS_CODE,
    FOCUS_WINDOW_END,
    FOCUS_WINDOW_START,
    HUMAN_FLIP_DATE,
    apply_recovery_up,
    constants_snapshot,
    decide_diagnostic_promote,
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

VARIANTS = ("baseline", "hrp_shadow", "hrp_f2_overlay")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pool_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
        "--membership-csv",
        type=Path,
        default=DEFAULT_MEMBERSHIP_CSV,
        help="cluster_representatives_*.csv from HRP dendrogram script",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260811_hrp_cluster_router_ab",
    )
    p.add_argument("--code", action="append", default=[])
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument(
        "--N",
        type=int,
        default=DIAGNOSTIC_N,
        help="recovery_up streak N (diagnostic default=2)",
    )
    p.add_argument("--p20-min", type=float, default=None)
    p.add_argument("--R", type=float, default=None)
    p.add_argument(
        "--enable-trade-overlay",
        action="store_true",
        help="research only: also apply recovery_up to trade labels for recovery_up clusters",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mem = load_cluster_membership(args.membership_csv)
    cluster_mode = paint_cluster_modes(mem)
    mode_by_code = code_mode_map(mem, cluster_mode)
    ru_codes = recovery_up_codes(mode_by_code)
    snap = router_snapshot(
        mem, source_path=str(args.membership_csv.resolve())
    )
    rec_consts = constants_snapshot()
    ru_kwargs: dict[str, Any] = {"N": int(args.N)}
    rec_consts = {**rec_consts, "N": int(args.N)}
    if args.p20_min is not None:
        ru_kwargs["p20_min"] = float(args.p20_min)
        rec_consts = {**rec_consts, "p20_min": float(args.p20_min)}
    if args.R is not None:
        ru_kwargs["R"] = float(args.R)
        ru_kwargs["rec_min"] = float(args.R)
        rec_consts = {**rec_consts, "R": float(args.R), "rec_min": float(args.R)}

    pm = _load(
        "pm_hrp_ab",
        _ROOT / "decision_pack/scripts/plot_regime_transition_example.py",
    )
    pm._init_qlib(None)
    ev = _load(
        "ev_hrp_ab",
        _ROOT / "decision_pack/scripts/eval_causal_regime_switch_pool.py",
    )

    codes = [str(x).upper() for x in args.code] or list(
        load_cluster_mapping_codes(CLUSTER_MAPPING_SELECTED_TXT)
    )
    if args.max_codes:
        codes = codes[: int(args.max_codes)]

    # Ensure recovery_up exemplars present when running full pool
    missing_ru = sorted(ru_codes - set(codes))
    if missing_ru and not args.code:
        print(f"[warn] recovery_up codes not in universe: {missing_ru}")

    per_variant: dict[str, list[dict[str, Any]]] = {v: [] for v in VARIANTS}
    focus_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []

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

        raw_mask = recovery_up_mask(
            px, labs0, ed_bool if ed_ok else None, **ru_kwargs
        )
        # v1 diagnostic: never change trade labels via mode gate (TRADE_MODES empty).
        # Optional research arm still applies overlay only when --enable-trade-overlay.
        if mode_allows_trade_overlay(mode) or (
            getattr(args, "enable_trade_overlay", False)
            and mode_allows_diagnostic_marks(mode)
        ):
            gated = raw_mask
        else:
            gated = np.zeros(len(px), dtype=bool)
        labs_ov = apply_recovery_up(labs0, gated)
        # Diagnostic mask for focus metrics (even when trades stay baseline)
        diag_mask = raw_mask if mode_allows_diagnostic_marks(mode) else gated
        labs_diag = apply_recovery_up(labs0, diag_mask)

        audit = (px["as_of"] >= args.start_date) & (px["as_of"] <= args.end_date)
        audit_idx = audit.to_numpy()
        close = px.loc[audit, "$close"].to_numpy(float)
        dates = px.loc[audit, "as_of"].dt.strftime("%Y-%m-%d").to_numpy()

        arm_labs = {
            "baseline": labs0,
            "hrp_shadow": labs0,
            "hrp_f2_overlay": labs_ov,
        }
        for arm, labs in arm_labs.items():
            m, _, _ = hold_up_trade_metrics(
                labs[audit_idx], close, dates=dates
            )
            per_variant[arm].append(
                {
                    "code": code,
                    "variant": arm,
                    "method": method,
                    "hrp_mode": mode,
                    "cluster": int(
                        mem.loc[mem.code == code, "cluster"].iloc[0]
                    )
                    if (mem.code == code).any()
                    else -1,
                    "ed_ok": bool(ed_ok),
                    "raw_mask_frac": float(raw_mask[audit_idx].mean()),
                    "gated_mask_frac": float(gated[audit_idx].mean()),
                    **m,
                }
            )

        diag_rows.append(
            {
                "code": code,
                "cluster": int(mem.loc[mem.code == code, "cluster"].iloc[0])
                if (mem.code == code).any()
                else -1,
                "hrp_mode": mode,
                "method": method,
                "trade_overlay": bool(
                    mode_allows_trade_overlay(mode) or args.enable_trade_overlay
                ),
                "diagnostic": mode_allows_diagnostic_marks(mode),
                "raw_mask_frac_audit": float(raw_mask[audit_idx].mean()),
                "gated_mask_frac_audit": float(gated[audit_idx].mean()),
            }
        )

        if code == FOCUS_CODE:
            flip = first_up_date(
                px["as_of"], labs_diag, start=HUMAN_FLIP_DATE, end=FOCUS_WINDOW_END
            )
            delay = (
                trading_days_between(px["as_of"], HUMAN_FLIP_DATE, flip)
                if flip
                else None
            )
            focus_rows.append(
                {
                    "code": code,
                    "hrp_mode": mode,
                    "method": method,
                    "baseline_up_frac_post_flip": up_frac_in_window(
                        px["as_of"], labs0, HUMAN_FLIP_DATE, FOCUS_WINDOW_END
                    ),
                    "diag_up_frac_post_flip": up_frac_in_window(
                        px["as_of"], labs_diag, HUMAN_FLIP_DATE, FOCUS_WINDOW_END
                    ),
                    "overlay_up_frac_post_flip": up_frac_in_window(
                        px["as_of"], labs_diag, HUMAN_FLIP_DATE, FOCUS_WINDOW_END
                    ),
                    "overlay_up_frac_full_window": up_frac_in_window(
                        px["as_of"],
                        labs_diag,
                        FOCUS_WINDOW_START,
                        FOCUS_WINDOW_END,
                    ),
                    "first_overlay_up": flip,
                    "flip_delay_td": delay,
                    "human_flip_date": HUMAN_FLIP_DATE,
                    "trade_labels_changed": bool(args.enable_trade_overlay),
                }
            )

        print(
            f"[{i}/{len(codes)}] {code} mode={mode} method={method} "
            f"gated_mask={gated.mean():.2%}"
        )

    pd.DataFrame(diag_rows).to_csv(
        out_dir / "code_mode_diag.csv", index=False, encoding="utf-8-sig"
    )

    # Wide comparison
    base_df = pd.DataFrame(per_variant["baseline"]).set_index("code")
    wide = base_df.add_prefix("baseline_").reset_index()
    for arm in VARIANTS[1:]:
        df = pd.DataFrame(per_variant[arm]).set_index("code")
        wide = wide.merge(df.add_prefix(f"{arm}_").reset_index(), on="code", how="outer")
    wide.to_csv(out_dir / "symbol_comparison.csv", index=False, encoding="utf-8-sig")

    pools = {arm: _pool_from_rows(per_variant[arm]) for arm in VARIANTS}
    pd.DataFrame([{**{"variant": k}, **v} for k, v in pools.items()]).to_csv(
        out_dir / "pool_comparison.csv", index=False, encoding="utf-8-sig"
    )

    # Non-recovery_up subset pools
    def _subset(arm: str, pred) -> dict[str, Any]:
        rows = [r for r in per_variant[arm] if pred(r["code"])]
        return _pool_from_rows(rows) if rows else {
            "n_closed": 0,
            "wins": 0,
            "trade_win": float("nan"),
            "traded_symbols": 0,
            "mean_edge": float("nan"),
            "mean_ret_net": float("nan"),
            "worst_drawdown": 0.0,
        }

    non_ru = lambda c: c not in ru_codes  # noqa: E731
    pool_non_ru_b = _subset("baseline", non_ru)
    pool_non_ru_ov = _subset("hrp_f2_overlay", non_ru)

    b = pools["baseline"]
    ov = pools["hrp_f2_overlay"]
    gates_pool = _pool_gates(ov, b)
    gates_non_ru = _pool_gates(pool_non_ru_ov, pool_non_ru_b)
    pool_ok = all(gates_pool.values())
    non_ru_ok = all(gates_non_ru.values())

    focus_df = pd.DataFrame(focus_rows)
    focus_df.to_csv(out_dir / "focus_SH515220.csv", index=False, encoding="utf-8-sig")
    focus_ok = False
    flip_ok = False
    focus_detail: dict[str, Any] = {}
    if len(focus_df):
        fr = focus_df.iloc[0].to_dict()
        focus_detail = fr
        up_frac = float(fr.get("overlay_up_frac_post_flip", float("nan")))
        delay = fr.get("flip_delay_td")
        focus_ok = bool(np.isfinite(up_frac) and up_frac >= 0.80)
        flip_ok = delay is not None and int(delay) <= int(rec_consts["D_flip"])

    gates = {
        **{f"pool_{k}": v for k, v in gates_pool.items()},
        **{f"non_ru_{k}": v for k, v in gates_non_ru.items()},
        "focus_up_frac_ge_0_80": focus_ok,
        "flip_delay_le_D": flip_ok,
    }
    # Design: pool + non-RU + focus confirmatory
    # Diagnostic promote: trades unchanged by default; focus uses N=DIAGNOSTIC_N mask.
    recommendation = decide_diagnostic_promote(focus_ok=focus_ok, flip_ok=flip_ok)
    trade_promote = None
    if args.enable_trade_overlay:
        trade_promote = decide_recommendation(
            pool_ok and non_ru_ok, focus_ok, flip_ok
        )

    summary = {
        "selection": f"B0 once on train < {args.train_cutoff} (no overlay)",
        "audit": f"{args.start_date}..{args.end_date}",
        "n_symbols": len(codes),
        "hrp_router": snap,
        "recovery_up_constants": rec_consts,
        "design": "huangtuo-master-design-20260812-171530-hrp-cluster-router.md",
        "baseline_pool": b,
        "overlay_pool": ov,
        "baseline_non_ru_pool": pool_non_ru_b,
        "overlay_non_ru_pool": pool_non_ru_ov,
        "gates": gates,
        "pool_ok": pool_ok,
        "non_ru_ok": non_ru_ok,
        "focus_ok": focus_ok,
        "flip_ok": flip_ok,
        "focus": focus_detail,
        "recommendation": recommendation,
        "trade_overlay_enabled": bool(args.enable_trade_overlay),
        "trade_recommendation": trade_promote,
        "production_modified": False,
        "channel": "diagnostic_dual",
    }
    (out_dir / "acceptance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    lines = [
        "# HRP cluster router A/B",
        "",
        f"- Membership: `{args.membership_csv}`",
        f"- recovery_up codes: `{sorted(ru_codes)}`",
        f"- Recommendation: `{recommendation}`",
        "",
        "## Pools",
        "",
        f"- baseline win={b['trade_win']:.1%} n={b['n_closed']} edge={b['mean_edge']:+.2%}",
        f"- overlay  win={ov['trade_win']:.1%} n={ov['n_closed']} edge={ov['mean_edge']:+.2%}",
        f"- non-RU baseline edge={pool_non_ru_b['mean_edge']:+.2%} / overlay edge={pool_non_ru_ov['mean_edge']:+.2%}",
        "",
        "```json",
        json.dumps(gates, ensure_ascii=False, indent=2),
        "```",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("DONE recommend=", recommendation, "gates=", gates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
