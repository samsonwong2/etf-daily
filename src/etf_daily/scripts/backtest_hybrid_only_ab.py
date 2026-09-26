#!/usr/bin/env python3
"""A/B: patch ONLY ``hybrid_ma_adx`` (SH513650-class late thrust / min_seg wipe).

Case
----
SH513650 hold_up edge ≈ −19% vs BH +20%. From 2026-08-04 the method layer is
already ``up``, but ``finalize min_seg=10`` (and even 5) merges a 4-bar thrust
back to range. ``p20`` peaks ~7% so thrust@8% never fires; @6% does.

Variants (monkeypatch; production untouched)
--------------------------------------------
baseline
hybrid_min_seg3
    ``run_method('hybrid_ma_adx')`` uses ``min_seg=3``.
hybrid_min_seg0
    No short-segment merge for hybrid.
hybrid_thrust_p06
    After finalize: ``C>MA20 & p20≥6%`` → force up.
hybrid_combo
    min_seg=3 + thrust_p06.

Scope: only ``hybrid_ma_adx``. hyst/strict/dual/struct untouched.
Selection: B0 on train < cutoff. Focus also reports **pinned hybrid**
(always evaluate patched hybrid on SH513650) so deselect does not hide the fix.
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

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from etf_daily.lib.adaptive_stage_common import (  # noqa: E402
    TRADE_MODE_HOLD_UP,
    prepare_features,
    select_regime_method_legacy,
)
from etf_daily.lib.state_space_selection import hold_up_trade_metrics  # noqa: E402
from etf_daily.lib.symbol_trend_board import load_cluster_mapping_codes  # noqa: E402
from etf_daily.paths import CLUSTER_MAPPING_SELECTED_TXT, PLOTLY_OUTPUTS_DIR  # noqa: E402

FOCUS = "SH513650"
AUG_START, AUG_END = "2026-07-28", "2026-08-07"
WEAK_UP_START, WEAK_UP_END = "2026-07-06", "2026-07-27"
EARLY_START, EARLY_END = "2026-04-08", "2026-05-05"
VARIANTS = (
    "baseline",
    "hybrid_min_seg3",
    "hybrid_min_seg0",
    "hybrid_thrust_p06",
    "hybrid_combo",
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _thrust_protect(px: pd.DataFrame, labs: np.ndarray, *, p20_thr: float) -> np.ndarray:
    out = np.asarray(labs, dtype=object).copy()
    close = px["$close"].to_numpy(float)
    m20 = px["MA20"].to_numpy(float)
    p20 = px["prior20"].to_numpy(float)
    for i in range(len(out)):
        if out[i] == "up":
            continue
        if (
            np.isfinite(close[i])
            and np.isfinite(m20[i])
            and np.isfinite(p20[i])
            and close[i] > m20[i]
            and p20[i] >= p20_thr
        ):
            out[i] = "up"
    return out


def _patch(ev, mode: str) -> None:
    if mode == "baseline":
        return

    use_seg3 = mode in {"hybrid_min_seg3", "hybrid_combo"}
    use_seg0 = mode == "hybrid_min_seg0"
    use_thrust06 = mode in {"hybrid_thrust_p06", "hybrid_combo"}

    orig_run = ev.run_method

    def run_method(name: str, px: pd.DataFrame, *, params=None, min_seg: int = 10):
        params = dict(params or {})
        if name == "hybrid_ma_adx":
            if use_seg0:
                params["min_seg"] = 0
                min_seg = 0
            elif use_seg3:
                params.setdefault("min_seg", 3)
                min_seg = int(params["min_seg"])
            labs = orig_run(name, px, params=params, min_seg=min_seg)
            if use_thrust06:
                labs = _thrust_protect(px, labs, p20_thr=0.06)
            return labs
        return orig_run(name, px, params=params, min_seg=min_seg)

    ev.run_method = run_method  # type: ignore[assignment]


def _pool(rows: pd.DataFrame, prefix: str) -> dict[str, Any]:
    n = int(rows[f"{prefix}_n_closed"].sum())
    wins = int(rows[f"{prefix}_wins"].sum())
    traded = int((rows[f"{prefix}_n_closed"] > 0).sum())
    return {
        "n_closed": n,
        "wins": wins,
        "trade_win": float(wins / n) if n else float("nan"),
        "traded_symbols": traded,
        "mean_edge": float(pd.to_numeric(rows[f"{prefix}_edge"], errors="coerce").mean()),
        "mean_ret_net": float(
            pd.to_numeric(rows[f"{prefix}_mean_ret_net"], errors="coerce").mean()
        ),
        "worst_drawdown": float(
            pd.to_numeric(rows[f"{prefix}_max_drawdown"], errors="coerce").min()
        ),
    }


def _wfrac(px, labs, a, b, lab: str = "up") -> float:
    m = (px["as_of"] >= a) & (px["as_of"] <= b)
    L = labs[m.to_numpy()]
    return float((L == lab).mean()) if len(L) else float("nan")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", default="2026-04-01")
    p.add_argument("--end-date", default="2026-08-07")
    p.add_argument("--train-cutoff", default="2026-04-01")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260807_hybrid_only_ab",
    )
    p.add_argument("--code", action="append", default=[])
    p.add_argument("--max-codes", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pm = _load(
        "pm_hybrid_only",
        _ROOT / "src/etf_daily/plots/plot_regime_transition_example.py",
    )
    pm._init_qlib(None)
    codes = [str(x).upper() for x in args.code] or list(
        load_cluster_mapping_codes(CLUSTER_MAPPING_SELECTED_TXT)
    )
    if args.max_codes:
        codes = codes[: args.max_codes]

    ohlcvs = {}
    for i, code in enumerate(codes, 1):
        ohlcvs[code] = pm.load_qlib_ohlcv(
            code,
            "2015-01-01",
            args.end_date,
            init_qlib=False,
            lookback_calendar_days=400,
            clip_to_window=False,
        )
        print(f"[load {i}/{len(codes)}] {code}", flush=True)

    per: dict[str, list[dict[str, Any]]] = {v: [] for v in VARIANTS}
    focus_rows: list[dict[str, Any]] = []

    for mode in VARIANTS:
        ev = _load(
            f"ev_hybrid_only_{mode}",
            _ROOT / "src/etf_daily/scripts/eval_causal_regime_switch_pool.py",
        )
        _patch(ev, mode)
        for i, code in enumerate(codes, 1):
            px = prepare_features(ev, ohlcvs[code], method=None)
            px_train = px[px["as_of"] < args.train_cutoff].reset_index(drop=True)
            method, _, _ = select_regime_method_legacy(
                ev, px_train, trade_mode=TRADE_MODE_HOLD_UP
            )
            labs = np.asarray(ev.run_method(method, px), dtype=object)
            audit = (px["as_of"] >= args.start_date) & (px["as_of"] <= args.end_date)
            m, _, _ = hold_up_trade_metrics(
                labs[audit.to_numpy()],
                px.loc[audit, "$close"].to_numpy(float),
                dates=px.loc[audit, "as_of"].dt.strftime("%Y-%m-%d").to_numpy(),
            )
            row = {
                "code": code,
                "variant": mode,
                "method": method,
                "picked_hybrid": method == "hybrid_ma_adx",
                "up_frac": float((labs[audit.to_numpy()] == "up").mean()),
                **m,
            }
            per[mode].append(row)
            if code == FOCUS:
                pin = np.asarray(ev.run_method("hybrid_ma_adx", px), dtype=object)
                m_pin, _, _ = hold_up_trade_metrics(
                    pin[audit.to_numpy()],
                    px.loc[audit, "$close"].to_numpy(float),
                    dates=px.loc[audit, "as_of"].dt.strftime("%Y-%m-%d").to_numpy(),
                )
                focus_rows.append(
                    {
                        "variant": mode,
                        "method": method,
                        "aug_up": _wfrac(px, labs, AUG_START, AUG_END),
                        "weak_up": _wfrac(px, labs, WEAK_UP_START, WEAK_UP_END),
                        "early_up": _wfrac(px, labs, EARLY_START, EARLY_END),
                        "edge": m["edge"],
                        "n_closed": m["n_closed"],
                        "win": m["win"],
                        "pin_aug_up": _wfrac(px, pin, AUG_START, AUG_END),
                        "pin_edge": m_pin["edge"],
                        "pin_n_closed": m_pin["n_closed"],
                        "pin_win": m_pin["win"],
                    }
                )
            print(
                f"[{mode}] {i}/{len(codes)} {code} -> {method} "
                f"win={m['win']} n={m['n_closed']} edge={m['edge']:+.2%}",
                flush=True,
            )

    base = pd.DataFrame(per["baseline"]).set_index("code")
    wide = base.add_prefix("baseline_").reset_index()
    for mode in VARIANTS[1:]:
        df = pd.DataFrame(per[mode]).set_index("code")
        wide = wide.merge(df.add_prefix(f"{mode}_").reset_index(), on="code", how="outer")
    wide.to_csv(out_dir / "symbol_comparison.csv", index=False, encoding="utf-8-sig")

    hybrid_codes = set(base.index[base["method"] == "hybrid_ma_adx"])

    pool_rows = []
    for mode in VARIANTS:
        df = pd.DataFrame(per[mode])
        tmp = df.rename(
            columns={
                "n_closed": f"{mode}_n_closed",
                "wins": f"{mode}_wins",
                "edge": f"{mode}_edge",
                "mean_ret_net": f"{mode}_mean_ret_net",
                "max_drawdown": f"{mode}_max_drawdown",
            }
        )
        s = _pool(tmp, mode)
        s["variant"] = mode
        s["method_mix"] = df["method"].value_counts().to_dict()
        s["n_hybrid"] = int((df["method"] == "hybrid_ma_adx").sum())
        sub = df[df["code"].isin(hybrid_codes)]
        if len(sub):
            n = int(sub["n_closed"].sum())
            w = int(sub["wins"].sum())
            s["hybrid_cohort_n"] = n
            s["hybrid_cohort_win"] = float(w / n) if n else float("nan")
            s["hybrid_cohort_edge"] = float(
                pd.to_numeric(sub["edge"], errors="coerce").mean()
            )
        else:
            s["hybrid_cohort_n"] = 0
            s["hybrid_cohort_win"] = float("nan")
            s["hybrid_cohort_edge"] = float("nan")
        pool_rows.append(s)
    pd.DataFrame(pool_rows).to_csv(
        out_dir / "pool_comparison.csv", index=False, encoding="utf-8-sig"
    )
    focus_df = pd.DataFrame(focus_rows)
    focus_df.to_csv(out_dir / f"{FOCUS}_focus.csv", index=False, encoding="utf-8-sig")
    (out_dir / "baseline_hybrid_codes.json").write_text(
        json.dumps(sorted(hybrid_codes), indent=2), encoding="utf-8"
    )

    b = pool_rows[0]
    fb = next(r for r in focus_rows if r["variant"] == "baseline")
    decisions = {}
    for s in pool_rows[1:]:
        mode = s["variant"]
        fr = next(r for r in focus_rows if r["variant"] == mode)
        gates = {
            "win_improved_or_equal": (
                s["trade_win"] >= b["trade_win"] - 1e-9
                if np.isfinite(s["trade_win"]) and np.isfinite(b["trade_win"])
                else False
            ),
            "enough_trades": s["n_closed"] >= max(10, int(0.8 * b["n_closed"])),
            "mean_net_noninferior": s["mean_ret_net"] >= b["mean_ret_net"] - 1e-6,
            "edge_noninferior": s["mean_edge"] >= b["mean_edge"] - 1e-6,
            "drawdown_controlled": s["worst_drawdown"] >= b["worst_drawdown"] - 0.02,
            "hybrid_cohort_win_noninferior": (
                s["hybrid_cohort_win"] >= b["hybrid_cohort_win"] - 1e-9
                if np.isfinite(s["hybrid_cohort_win"])
                and np.isfinite(b["hybrid_cohort_win"])
                else True
            ),
            "hybrid_cohort_edge_noninferior": (
                s["hybrid_cohort_edge"] >= b["hybrid_cohort_edge"] - 1e-6
                if np.isfinite(s["hybrid_cohort_edge"])
                and np.isfinite(b["hybrid_cohort_edge"])
                else True
            ),
            "pin_aug_up_ge_50": float(fr["pin_aug_up"]) >= 0.50,
            "pin_aug_improved": float(fr["pin_aug_up"])
            >= float(fb["pin_aug_up"]) + 0.25,
            "pin_edge_improved": float(fr["pin_edge"]) >= float(fb["pin_edge"]) + 0.02,
            "b0_still_hybrid": fr["method"] == "hybrid_ma_adx",
        }
        pool_ok = all(
            gates[k]
            for k in (
                "win_improved_or_equal",
                "enough_trades",
                "mean_net_noninferior",
                "edge_noninferior",
                "drawdown_controlled",
                "hybrid_cohort_win_noninferior",
                "hybrid_cohort_edge_noninferior",
            )
        )
        # Promote only if pinned hybrid actually fixes the case AND pool holds.
        focus_ok = gates["pin_aug_up_ge_50"] or gates["pin_aug_improved"] or gates[
            "pin_edge_improved"
        ]
        decisions[mode] = {
            "gates": gates,
            "promote_candidate": bool(pool_ok and focus_ok and gates["b0_still_hybrid"]),
            "pool": s,
            "focus": fr,
        }

    recommendation = "keep_baseline"
    for mode in (
        "hybrid_combo",
        "hybrid_thrust_p06",
        "hybrid_min_seg0",
        "hybrid_min_seg3",
    ):
        if decisions.get(mode, {}).get("promote_candidate"):
            recommendation = mode
            break

    summary = {
        "selection": f"B0 reselect on train < {args.train_cutoff}",
        "audit": f"{args.start_date}..{args.end_date}",
        "scope": "hybrid_ma_adx only; hyst/strict/dual/struct untouched",
        "case": (
            f"{FOCUS}: late thrust {AUG_START}..{AUG_END} wiped by min_seg; "
            "BH≈+20% vs hold_up≈0; pin_* = forced hybrid eval"
        ),
        "baseline_hybrid_codes": sorted(hybrid_codes),
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
                        "n_hybrid",
                        "hybrid_cohort_win",
                        "hybrid_cohort_edge",
                        "hybrid_cohort_n",
                    )
                },
                "focus": v["focus"],
            }
            for k, v in decisions.items()
        },
        "recommendation": recommendation,
    }
    (out_dir / "acceptance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    lines = [
        "# hybrid_ma_adx-only A/B (SH513650)",
        "",
        "- Scope: **only** `hybrid_ma_adx` patched.",
        f"- Baseline hybrid cohort ({len(hybrid_codes)}): `{sorted(hybrid_codes)}`.",
        f"- Focus `{FOCUS}` late window `{AUG_START}`..`{AUG_END}` (plus pin_* forced hybrid).",
        "- Production **not** modified.",
        "",
        "## Pool",
        "",
        "| Variant | Win | n | Edge | #hybrid | Hybrid-cohort win | Hybrid-cohort edge |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in pool_rows:
        hw = s["hybrid_cohort_win"]
        he = s["hybrid_cohort_edge"]
        hw_s = f"{hw:.1%}" if np.isfinite(hw) else "nan"
        he_s = f"{he:+.2%}" if np.isfinite(he) else "nan"
        lines.append(
            f"| {s['variant']} | {s['trade_win']:.1%} | {s['n_closed']} | "
            f"{s['mean_edge']:+.2%} | {s['n_hybrid']} | {hw_s} | {he_s} |"
        )
    lines += ["", "## Focus", "", focus_df.to_string(index=False), ""]
    lines += [f"## Recommendation: `{recommendation}`", ""]
    lines += [
        "```json",
        json.dumps({k: v["gates"] for k, v in decisions.items()}, indent=2),
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
                s["n_hybrid"],
            )
            for s in pool_rows
        },
        "recommend=",
        recommendation,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
