#!/usr/bin/env python3
"""A/B: patch ONLY ``ma_stack_hyst`` (not strict/dual/hybrid/…).

SH517120 currently selects ma_stack_hyst. Changing that method still affects
every symbol that B0 picks hyst (~10/46), but not the whole METHOD zoo.

Variants (monkeypatch; no production commit)
--------------------------------------------
baseline
hyst_vrev_fast
    Strong p20 V-rev sticky raw + enter confirm=1 when p20>=0.08.
hyst_thrust_protect
    After finalize for hyst only: C>MA20 & p20>=0.08 → force up.
hyst_combo
    Both.

hybrid_ma_adx is untouched (still calls production method_ma_stack).
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

FOCUS = "SH517120"
RALLY_START, RALLY_END = "2026-06-29", "2026-07-13"
LATE_START, LATE_END = "2026-07-14", "2026-07-31"
AUG_START, AUG_END = "2026-08-03", "2026-08-07"
VARIANTS = ("baseline", "hyst_vrev_fast", "hyst_thrust_protect", "hyst_combo")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _strong_vrev(c, m20, m60, p20, p60) -> bool:
    if not (np.isfinite(c) and np.isfinite(m20) and c > m20 and p20 >= 0.08):
        return False
    if np.isfinite(m60) and m20 > m60:
        return True
    return p60 <= -0.08 or p60 >= 0.08


def _thrust_protect(px: pd.DataFrame, labs: np.ndarray) -> np.ndarray:
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
            and p20[i] >= 0.08
        ):
            out[i] = "up"
    return out


def _hyst_vrev_fast(ev, px: pd.DataFrame) -> np.ndarray:
    raw: list[str] = []
    sticky = False
    for _, r in px.iterrows():
        if pd.isna(r["MA20"]) or pd.isna(r["MA60"]):
            raw.append("range")
            sticky = False
            continue
        c, m20, m60 = float(r["$close"]), float(r["MA20"]), float(r["MA60"])
        p20 = float(r["prior20"]) if pd.notna(r["prior20"]) else 0.0
        p60 = float(r["prior60"]) if pd.notna(r["prior60"]) else 0.0
        band = float(r["band60"]) if pd.notna(r["band60"]) else np.nan
        hug = abs(m20 - m60) / c < 0.02
        chop = (pd.notna(band) and band < 0.22 and abs(p60) < 0.10) or (
            hug and abs(p60) < 0.12
        )
        up = (m20 > m60 and c > m20 and p60 >= 0.03) or (
            m20 > m60 and c > m20 and p20 >= 0.03
        )
        if c > m20 and p20 >= 0.06 and (m20 > m60 or p60 <= -0.08 or p60 >= 0.08):
            up = True
        if _strong_vrev(c, m20, m60, p20, p60):
            up = True
            sticky = True
        elif sticky and c > m20 and p20 >= 0.04:
            up = True
        else:
            if sticky and not (c > m20 and p20 >= 0.04):
                sticky = False
        down = (m20 < m60 and c < m20 and p60 <= -0.03) or (
            m20 < m60 and c < m20 and p20 <= -0.03
        )
        if up and not down:
            lab = "up"
        elif down and not up:
            lab = "down"
            sticky = False
        else:
            lab = "range" if chop or True else "range"
        raw.append(lab)

    above20 = (px["$close"] > px["MA20"]).fillna(False).to_numpy(bool)
    ma20 = px["MA20"].to_numpy(float)
    ma60 = px["MA60"].to_numpy(float)
    close = px["$close"].to_numpy(float)
    p20a = px["prior20"].to_numpy(float)
    state = "range"
    below = above = 0
    trough = peak = np.nan
    soft_guard = 0
    out: list[str] = []
    confirm = 3
    for i, lab in enumerate(raw):
        if above20[i]:
            above += 1
            below = 0
        else:
            below += 1
            above = 0
        m20, m60 = ma20[i], ma60[i]
        c = float(close[i]) if np.isfinite(close[i]) else np.nan
        p20i = float(p20a[i]) if np.isfinite(p20a[i]) else 0.0
        enter_confirm = 1 if p20i >= 0.08 else confirm
        bull_stack = np.isfinite(m20) and np.isfinite(m60) and m20 > m60
        bear_stack = np.isfinite(m20) and np.isfinite(m60) and m20 < m60
        if state == "up":
            if np.isfinite(c):
                peak = c if (not np.isfinite(peak) or c > peak) else peak
            dd = (
                (c / peak - 1.0)
                if (np.isfinite(c) and np.isfinite(peak) and peak > 0)
                else 0.0
            )
            soft_up = below >= 2 or dd <= -0.06
            if lab == "down" and (below >= confirm or bear_stack):
                state = "down"
                trough = c if np.isfinite(c) else np.nan
                peak = np.nan
                soft_guard = 0
            elif soft_up or (lab == "range" and below >= confirm + 1):
                state = "range"
                trough = np.nan
                peak = np.nan
        elif state == "down":
            if np.isfinite(c):
                trough = c if (not np.isfinite(trough) or c < trough) else trough
            bounce = (
                (c / trough - 1.0)
                if (np.isfinite(c) and np.isfinite(trough) and trough > 0)
                else 0.0
            )
            soft_leave = above >= 2 or bounce >= 0.04
            if lab == "up" and (above >= enter_confirm or bull_stack):
                state = "up"
                trough = np.nan
                peak = c if np.isfinite(c) else np.nan
                soft_guard = 0
            elif soft_leave or (lab == "range" and above >= confirm):
                state = "range"
                soft_guard = 3
                peak = np.nan
        else:
            if soft_guard > 0:
                soft_guard -= 1
            if lab == "up" and (above >= enter_confirm or bull_stack):
                state = "up"
                trough = np.nan
                peak = c if np.isfinite(c) else np.nan
                soft_guard = 0
            elif lab == "down" and (below >= confirm or bear_stack):
                higher_low = np.isfinite(trough) and np.isfinite(c) and c > trough
                if higher_low:
                    pass
                elif soft_guard > 0 and below < max(confirm, 3):
                    pass
                else:
                    state = "down"
                    trough = c if np.isfinite(c) else np.nan
                    peak = np.nan
                    soft_guard = 0
        out.append(state)
    return np.array(ev.causal_smooth(out, 3), dtype=object)


def _patch(ev, mode: str) -> None:
    if mode == "baseline":
        return
    use_fast = mode in {"hyst_vrev_fast", "hyst_combo"}
    use_protect = mode in {"hyst_thrust_protect", "hyst_combo"}
    prod_hyst = ev.method_ma_stack

    def method_hyst(px: pd.DataFrame) -> np.ndarray:
        if use_fast:
            return _hyst_vrev_fast(ev, px)
        return prod_hyst(px)

    # Only swap the hyst entry in METHODS. hybrid keeps production base.
    methods = dict(ev.METHODS)
    methods["ma_stack_hyst"] = method_hyst
    ev.METHODS = methods
    ev.method_ma_stack = method_hyst  # B0 / callers that use the attribute

    if use_protect:
        orig_run = ev.run_method

        def run_method(name: str, px: pd.DataFrame, *, params=None, min_seg: int = 10):
            labs = orig_run(name, px, params=params, min_seg=min_seg)
            if name == "ma_stack_hyst":
                labs = _thrust_protect(px, labs)
            return labs

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


def _wfrac(px, labs, a, b) -> float:
    m = (px["as_of"] >= a) & (px["as_of"] <= b)
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
        default=PLOTLY_OUTPUTS_DIR / "20260807_hyst_only_vrev_ab",
    )
    p.add_argument("--code", action="append", default=[])
    p.add_argument("--max-codes", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pm = _load("pm_hyst_only", _ROOT / "decision_pack/scripts/plot_regime_transition_example.py")
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
        print(f"[load {i}/{len(codes)}] {code}")

    per: dict[str, list[dict[str, Any]]] = {v: [] for v in VARIANTS}
    focus_rows: list[dict[str, Any]] = []

    for mode in VARIANTS:
        ev = _load(
            f"ev_hyst_only_{mode}",
            _ROOT / "decision_pack/scripts/eval_causal_regime_switch_pool.py",
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
                "picked_hyst": method == "ma_stack_hyst",
                "up_frac": float((labs[audit.to_numpy()] == "up").mean()),
                **m,
            }
            per[mode].append(row)
            if code == FOCUS:
                focus_rows.append(
                    {
                        "variant": mode,
                        "method": method,
                        "rally_up": _wfrac(px, labs, RALLY_START, RALLY_END),
                        "late_up": _wfrac(px, labs, LATE_START, LATE_END),
                        "aug_up": _wfrac(px, labs, AUG_START, AUG_END),
                        "edge": m["edge"],
                        "n_closed": m["n_closed"],
                        "win": m["win"],
                    }
                )
            print(
                f"[{mode}] {i}/{len(codes)} {code} -> {method} "
                f"win={m['win']} n={m['n_closed']}"
            )

    base = pd.DataFrame(per["baseline"]).set_index("code")
    wide = base.add_prefix("baseline_").reset_index()
    for mode in VARIANTS[1:]:
        df = pd.DataFrame(per[mode]).set_index("code")
        wide = wide.merge(df.add_prefix(f"{mode}_").reset_index(), on="code", how="outer")
    wide.to_csv(out_dir / "symbol_comparison.csv", index=False, encoding="utf-8-sig")

    # Sub-pool: codes that used hyst under baseline (primary blast radius).
    hyst_codes = set(base.index[base["method"] == "ma_stack_hyst"])

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
        s["n_hyst"] = int((df["method"] == "ma_stack_hyst").sum())
        # metrics on baseline-hyst cohort only
        sub = df[df["code"].isin(hyst_codes)]
        if len(sub):
            n = int(sub["n_closed"].sum())
            w = int(sub["wins"].sum())
            s["hyst_cohort_n"] = n
            s["hyst_cohort_win"] = float(w / n) if n else float("nan")
            s["hyst_cohort_edge"] = float(pd.to_numeric(sub["edge"], errors="coerce").mean())
        else:
            s["hyst_cohort_n"] = 0
            s["hyst_cohort_win"] = float("nan")
            s["hyst_cohort_edge"] = float("nan")
        pool_rows.append(s)
    pd.DataFrame(pool_rows).to_csv(
        out_dir / "pool_comparison.csv", index=False, encoding="utf-8-sig"
    )
    focus_df = pd.DataFrame(focus_rows)
    focus_df.to_csv(out_dir / f"{FOCUS}_focus.csv", index=False, encoding="utf-8-sig")
    (out_dir / "baseline_hyst_codes.json").write_text(
        json.dumps(sorted(hyst_codes), indent=2), encoding="utf-8"
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
            "hyst_cohort_win_noninferior": (
                s["hyst_cohort_win"] >= b["hyst_cohort_win"] - 1e-9
                if np.isfinite(s["hyst_cohort_win"]) and np.isfinite(b["hyst_cohort_win"])
                else False
            ),
            "hyst_cohort_edge_noninferior": s["hyst_cohort_edge"]
            >= b["hyst_cohort_edge"] - 1e-6,
            "sh517120_rally_improved": float(fr["rally_up"])
            >= float(fb["rally_up"]) + 0.25,
            "sh517120_aug_improved": float(fr["aug_up"]) >= float(fb["aug_up"]) + 0.25,
            "sh517120_still_hyst": fr["method"] == "ma_stack_hyst",
        }
        pool_ok = all(
            gates[k]
            for k in (
                "win_improved_or_equal",
                "enough_trades",
                "mean_net_noninferior",
                "edge_noninferior",
                "drawdown_controlled",
                "hyst_cohort_win_noninferior",
                "hyst_cohort_edge_noninferior",
            )
        )
        focus_ok = gates["sh517120_still_hyst"] and (
            gates["sh517120_rally_improved"] or gates["sh517120_aug_improved"]
        )
        decisions[mode] = {
            "gates": gates,
            "promote_candidate": bool(pool_ok and focus_ok),
            "pool": s,
            "focus": fr,
        }

    recommendation = "keep_baseline"
    for mode in ("hyst_combo", "hyst_vrev_fast", "hyst_thrust_protect"):
        if decisions.get(mode, {}).get("promote_candidate"):
            recommendation = mode
            break

    summary = {
        "selection": f"B0 reselect on train < {args.train_cutoff}",
        "audit": f"{args.start_date}..{args.end_date}",
        "scope": "ma_stack_hyst only; hybrid/strict/dual untouched",
        "baseline_hyst_codes": sorted(hyst_codes),
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
                        "n_hyst",
                        "hyst_cohort_win",
                        "hyst_cohort_edge",
                        "hyst_cohort_n",
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
        "# ma_stack_hyst-only V-rev / thrust A/B",
        "",
        "- Scope: **only** `ma_stack_hyst` patched; hybrid/strict/dual unchanged.",
        f"- Baseline hyst cohort ({len(hyst_codes)}): `{sorted(hyst_codes)}`.",
        f"- Focus `{FOCUS}`.",
        "- Production **not** modified.",
        "",
        "## Pool",
        "",
        "| Variant | Win | n | Edge | #hyst | Hyst-cohort win | Hyst-cohort edge |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in pool_rows:
        lines.append(
            f"| {s['variant']} | {s['trade_win']:.1%} | {s['n_closed']} | "
            f"{s['mean_edge']:+.2%} | {s['n_hyst']} | "
            f"{s['hyst_cohort_win']:.1%} | {s['hyst_cohort_edge']:+.2%} |"
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
                s["n_hyst"],
            )
            for s in pool_rows
        },
        "recommend=",
        recommendation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
