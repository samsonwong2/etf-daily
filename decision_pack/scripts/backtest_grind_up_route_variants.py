#!/usr/bin/env python3
"""A/B: grind-up family routing vs production finalize (no production commit).

Variants
--------
baseline
    Production finalize (lowvol escape 12%/8% + min_seg=10).
escape_relax
    Approach A control: global escape_p60=0.06, escape_p20=0.04.
grind_route
    Approach B: on causal grind_up bars skip lowvol downgrade; after min_seg,
    force up when grind + bull stack + prior20>=0.05 (thrust protect).

Selection: B0 on train < cutoff. Audit OOS descriptive only.
Design: ~/.gstack/projects/wongtoe-skfolio_csi300/huangtuo-master-design-20260809-grind-up-route.md
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
from decision_pack.src.state_space_selection import hold_up_trade_metrics  # noqa: E402
from decision_pack.src.symbol_trend_board import load_cluster_mapping_codes  # noqa: E402
from runtime_paths import CLUSTER_MAPPING_SELECTED_TXT, PLOTLY_OUTPUTS_DIR  # noqa: E402

FOCUS_THRUST = "SH513650"
FOCUS_GRIND = "SH513400"
THRUST_START = "2026-07-28"
THRUST_END = "2026-08-07"
WEAK_START = "2026-07-06"
WEAK_END = "2026-07-28"

VARIANTS = ("baseline", "escape_relax", "grind_route")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _grind_up_mask(px: pd.DataFrame) -> np.ndarray:
    """Causal grind-up mask (bar-level).

    Low-vol membership is **percentile OR absolute** ``rv20``: slow-trend ETFs
    can sit at high self-history percentiles while absolute vol stays muted
    vs the broader pool (SH513650 Jul–Aug 2026).
    """
    n = len(px)
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    vp = (
        px["rv20_pct_252"].to_numpy(float)
        if "rv20_pct_252" in px.columns
        else np.full(n, np.nan)
    )
    rv = px["rv20"].to_numpy(float) if "rv20" in px.columns else np.full(n, np.nan)
    close = px["$close"].to_numpy(float)
    m20 = px["MA20"].to_numpy(float)
    m60 = px["MA60"].to_numpy(float)
    p20 = px["prior20"].to_numpy(float)
    p60 = px["prior60"].to_numpy(float)
    for i in range(n):
        if not (
            np.isfinite(close[i]) and np.isfinite(m20[i]) and np.isfinite(m60[i])
        ):
            continue
        low_vol = (np.isfinite(vp[i]) and vp[i] < 0.40) or (
            np.isfinite(rv[i]) and rv[i] < 0.23
        )
        if not low_vol:
            continue
        if not (m20[i] > m60[i] and close[i] > m20[i]):
            continue
        p20i = p20[i] if np.isfinite(p20[i]) else 0.0
        p60i = p60[i] if np.isfinite(p60[i]) else 0.0
        if p60i >= 0.04 or p20i >= 0.03:
            out[i] = True
    return out


def _patch(ev, mode: str) -> None:
    if mode == "baseline":
        return

    if mode == "escape_relax":
        orig = ev.apply_lowvol_range_bias

        def apply_lowvol_range_bias(
            labels,
            vol_pct,
            prior60,
            prior20,
            *,
            low_pct=0.35,
            look=15,
            persist=10,
            escape_p60=0.06,
            escape_p20=0.04,
        ):
            return orig(
                labels,
                vol_pct,
                prior60,
                prior20,
                low_pct=low_pct,
                look=look,
                persist=persist,
                escape_p60=escape_p60,
                escape_p20=escape_p20,
            )

        ev.apply_lowvol_range_bias = apply_lowvol_range_bias  # type: ignore[assignment]
        return

    if mode != "grind_route":
        raise ValueError(mode)

    def finalize_regime_labels(labels, px, *, min_seg: int = 10):
        labs = np.asarray(labels, dtype=object).copy()
        grind = _grind_up_mask(px)
        # Recent grind: vol often rises on the thrust day itself, so protect
        # using a short causal lookback instead of requiring grind[i] today.
        recent_grind = grind.copy()
        last = -10**9
        for i in range(len(grind)):
            if grind[i]:
                last = i
            recent_grind[i] = (i - last) <= 15 and last >= 0
        if "rv20_pct_252" in px.columns:
            # Standard lowvol on a copy, then restore grind / recent-grind bars.
            before = labs.copy()
            labs = ev.apply_lowvol_range_bias(
                labs,
                px["rv20_pct_252"].to_numpy(float),
                px["prior60"].to_numpy(float),
                px["prior20"].to_numpy(float),
            )
            labs = np.where(recent_grind, before, labs)
        ms = int(min_seg or 0)
        if ms > 1:
            labs = ev.merge_short_segments(labs, min_bars=ms)
        # Thrust protect after min_seg: require recent grind (not same-day
        # low-vol), plus bull stack and p20>=0.05. Avoids global p20 chase.
        close = px["$close"].to_numpy(float)
        m20 = px["MA20"].to_numpy(float)
        m60 = px["MA60"].to_numpy(float)
        p20 = px["prior20"].to_numpy(float)
        out = np.asarray(labs, dtype=object).copy()
        for i in range(len(out)):
            if out[i] == "up" or not recent_grind[i]:
                continue
            if not (
                np.isfinite(m20[i])
                and np.isfinite(m60[i])
                and np.isfinite(close[i])
                and np.isfinite(p20[i])
            ):
                continue
            if m20[i] > m60[i] and close[i] > m20[i] and p20[i] >= 0.05:
                out[i] = "up"
        return out

    ev.finalize_regime_labels = finalize_regime_labels  # type: ignore[assignment]


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


def _window_up_frac(px: pd.DataFrame, labs: np.ndarray, start: str, end: str) -> float:
    m = (px["as_of"] >= start) & (px["as_of"] <= end)
    L = labs[m.to_numpy()]
    return float((L == "up").mean()) if len(L) else float("nan")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", default="2026-04-01")
    p.add_argument("--end-date", default="2026-08-07")
    p.add_argument("--train-cutoff", default="2026-04-01")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260807_grind_up_route_ab",
    )
    p.add_argument("--code", action="append", default=[])
    p.add_argument("--max-codes", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pm = _load(
        "pm_grind",
        _ROOT / "decision_pack/scripts/plot_regime_transition_example.py",
    )
    pm._init_qlib(None)
    codes = [str(x).upper() for x in args.code] or list(
        load_cluster_mapping_codes(CLUSTER_MAPPING_SELECTED_TXT)
    )
    if args.max_codes:
        codes = codes[: args.max_codes]

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

    for mode in VARIANTS:
        ev = _load(
            f"ev_grind_{mode}",
            _ROOT / "decision_pack/scripts/eval_causal_regime_switch_pool.py",
        )
        _patch(ev, mode)
        for i, code in enumerate(codes, 1):
            px = prepare_features(ev, ohlcvs[code], method=None)
            px_train = px[px["as_of"] < args.train_cutoff].reset_index(drop=True)
            method, _, _meta = select_regime_method_legacy(
                ev, px_train, trade_mode=TRADE_MODE_HOLD_UP
            )
            labs = np.asarray(ev.run_method(method, px), dtype=object)
            grind = _grind_up_mask(px)
            audit = (px["as_of"] >= args.start_date) & (px["as_of"] <= args.end_date)
            close = px.loc[audit, "$close"].to_numpy(float)
            dates = px.loc[audit, "as_of"].dt.strftime("%Y-%m-%d").to_numpy()
            m, _trades, _ = hold_up_trade_metrics(
                labs[audit.to_numpy()], close, dates=dates
            )
            row = {
                "code": code,
                "variant": mode,
                "method": method,
                "up_frac": float((labs[audit.to_numpy()] == "up").mean()),
                "grind_frac_audit": float(grind[audit.to_numpy()].mean()),
                **m,
            }
            per_variant_rows[mode].append(row)
            if code in {FOCUS_THRUST, FOCUS_GRIND}:
                focus_rows.append(
                    {
                        "code": code,
                        "variant": mode,
                        "method": method,
                        "weak_up_frac": _window_up_frac(
                            px, labs, WEAK_START, WEAK_END
                        ),
                        "thrust_up_frac": _window_up_frac(
                            px, labs, THRUST_START, THRUST_END
                        ),
                        "audit_up_frac": row["up_frac"],
                        "grind_frac_audit": row["grind_frac_audit"],
                        "edge": m["edge"],
                        "n_closed": m["n_closed"],
                        "win": m["win"],
                    }
                )
            print(
                f"[{mode}] {i}/{len(codes)} {code} -> {method} "
                f"win={m['win']} n={m['n_closed']} grind={row['grind_frac_audit']:.2f}"
            )

    base = pd.DataFrame(per_variant_rows["baseline"]).set_index("code")
    wide = base.add_prefix("baseline_").reset_index()
    for mode in VARIANTS[1:]:
        df = pd.DataFrame(per_variant_rows[mode]).set_index("code")
        wide = wide.merge(df.add_prefix(f"{mode}_").reset_index(), on="code", how="outer")
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
        s["mean_grind_frac"] = float(df["grind_frac_audit"].mean())
        pool_rows.append(s)
    pd.DataFrame(pool_rows).to_csv(
        out_dir / "pool_comparison.csv", index=False, encoding="utf-8-sig"
    )
    focus_df = pd.DataFrame(focus_rows)
    focus_df.to_csv(out_dir / "focus_codes.csv", index=False, encoding="utf-8-sig")

    b = pool_rows[0]
    base_focus = {
        c: next(r for r in focus_rows if r["variant"] == "baseline" and r["code"] == c)
        for c in (FOCUS_THRUST, FOCUS_GRIND)
        if any(r["code"] == c and r["variant"] == "baseline" for r in focus_rows)
    }
    decisions: dict[str, Any] = {}
    for s in pool_rows[1:]:
        mode = s["variant"]
        fr650 = next(
            r
            for r in focus_rows
            if r["variant"] == mode and r["code"] == FOCUS_THRUST
        )
        fr400 = next(
            (
                r
                for r in focus_rows
                if r["variant"] == mode and r["code"] == FOCUS_GRIND
            ),
            None,
        )
        gates = {
            "win_improved_or_equal": (
                s["trade_win"] >= b["trade_win"] - 1e-9
                if np.isfinite(s["trade_win"]) and np.isfinite(b["trade_win"])
                else False
            ),
            "enough_trades": s["n_closed"] >= max(10, int(0.8 * b["n_closed"])),
            "mean_net_noninferior": s["mean_ret_net"] >= b["mean_ret_net"] - 1e-6,
            "edge_noninferior": s["mean_edge"] >= b["mean_edge"] - 1e-6,
            "drawdown_controlled": s["worst_drawdown"]
            >= b["worst_drawdown"] - 0.02,
            "sh513650_thrust_improved": float(fr650["thrust_up_frac"])
            >= float(base_focus[FOCUS_THRUST]["thrust_up_frac"]) + 0.25,
        }
        if fr400 is not None and FOCUS_GRIND in base_focus:
            gates["sh513400_edge_noninferior"] = float(fr400["edge"]) >= float(
                base_focus[FOCUS_GRIND]["edge"]
            ) - 1e-6
        pool_ok = all(
            gates[k]
            for k in (
                "win_improved_or_equal",
                "enough_trades",
                "mean_net_noninferior",
                "edge_noninferior",
                "drawdown_controlled",
            )
        )
        focus_ok = gates["sh513650_thrust_improved"] and gates.get(
            "sh513400_edge_noninferior", True
        )
        decisions[mode] = {
            "gates": gates,
            "promote_candidate": bool(pool_ok and focus_ok),
            "pool": s,
            "focus": {"SH513650": fr650, "SH513400": fr400},
        }

    recommendation = "keep_baseline"
    if decisions.get("grind_route", {}).get("promote_candidate"):
        recommendation = "grind_route"
    elif decisions.get("escape_relax", {}).get("promote_candidate"):
        recommendation = "escape_relax"

    summary = {
        "selection": f"B0 reselect on train < {args.train_cutoff}",
        "audit": f"{args.start_date}..{args.end_date}",
        "audit_used_for_selection": False,
        "n_symbols": len(codes),
        "design": "huangtuo-master-design-20260809-grind-up-route.md",
        "baseline": b,
        "variants": {
            k: {
                "gates": v["gates"],
                "promote_candidate": v["promote_candidate"],
                "pool": {
                    kk: v["pool"][kk]
                    for kk in (
                        "trade_win",
                        "n_closed",
                        "mean_edge",
                        "mean_ret_net",
                        "worst_drawdown",
                        "mean_grind_frac",
                    )
                },
            }
            for k, v in decisions.items()
        },
        "focus": focus_rows,
        "recommendation": recommendation,
    }
    (out_dir / "acceptance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    lines = [
        "# Grind-up route A/B (audit descriptive)",
        "",
        f"- Train/select: `< {args.train_cutoff}` (B0 legacy, hold_up).",
        f"- Audit: `{args.start_date}` → `{args.end_date}` (not used to pick).",
        "- Production code **not** modified.",
        "- Design: `huangtuo-master-design-20260809-grind-up-route.md`.",
        "",
        "## Pool",
        "",
        "| Variant | Trade win | n | Mean edge | Mean net | Worst DD | Mean grind frac |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in pool_rows:
        lines.append(
            f"| {s['variant']} | {s['trade_win']:.1%} | {s['n_closed']} | "
            f"{s['mean_edge']:+.2%} | {s['mean_ret_net']:+.2%} | "
            f"{s['worst_drawdown']:.1%} | {s['mean_grind_frac']:.1%} |"
        )
    lines += ["", "## Focus", "", focus_df.to_string(index=False), ""]
    lines += [f"## Recommendation: `{recommendation}`", ""]
    lines += [
        "```json",
        json.dumps(
            {k: v["gates"] for k, v in decisions.items()},
            ensure_ascii=False,
            indent=2,
        ),
        "```",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        "DONE",
        {
            s["variant"]: (
                round(s["trade_win"], 3),
                s["n_closed"],
                round(s["mean_edge"], 4),
            )
            for s in pool_rows
        },
        "recommend=",
        recommendation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
