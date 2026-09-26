#!/usr/bin/env python3
"""A/B: early ``down`` for ``ma_stack_strict`` (SH515060-class grind-down).

Case
----
SH515060 fell hard from 2026-05-13 but stayed ``range`` until 2026-06-08 because
strict down requires full bear stack ``MA20 < MA60`` plus ``C < MA20``. Price was
already below MA20 with ``prior60 ≤ -6%`` while MA20 was still above MA60.

Variants (monkeypatch; production untouched)
--------------------------------------------
baseline
    Production ``ma_stack_strict``.
strict_dn_soft_bear
    Drop stack: ``down = C < MA20 & p60 ≤ -6% & p20 ≤ 0``.
strict_dn_mom
    Soft-bear with tighter short momentum: also require ``p20 ≤ -3%``.
strict_dn_peak8
    Classic bear-stack down OR ``C < MA20 & p60 ≤ -6% & dd20_from_20d_high ≤ -8%``.
strict_dn_fast
    Soft-bear raw + enter-from-range confirm=1 when ``p20 ≤ -8%``.

Scope: only ``ma_stack_strict`` (and B0 reselect). hyst/struct/hybrid/dual unchanged.
Selection: B0 on train < cutoff. Audit descriptive only.
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

FOCUS = "SH515060"
MISS_START, MISS_END = "2026-05-13", "2026-06-05"
FULL_START, FULL_END = "2026-05-13", "2026-08-07"
VARIANTS = (
    "baseline",
    "strict_dn_soft_bear",
    "strict_dn_mom",
    "strict_dn_peak8",
    "strict_dn_fast",
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _apply_hyst_dn_fast(ev, raw: list[str], px: pd.DataFrame, *, confirm: int = 3) -> np.ndarray:
    """Like apply_hysteresis, but enter-from-range confirm=1 when p20 ≤ -8%."""
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
        enter_dn = 1 if p20i <= -0.08 else confirm
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
            if lab == "down" and (below >= enter_dn or bear_stack):
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
            if lab == "up" and (above >= confirm or bull_stack):
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
            if lab == "up" and (above >= confirm or bull_stack):
                state = "up"
                trough = np.nan
                peak = c if np.isfinite(c) else np.nan
                soft_guard = 0
            elif lab == "down" and (below >= enter_dn or bear_stack):
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


def _build_strict_method(ev, mode: str):
    def method_strict(px: pd.DataFrame) -> np.ndarray:
        close_arr = px["$close"].to_numpy(float)
        hi20 = pd.Series(close_arr).rolling(20, min_periods=5).max().to_numpy(float)
        raw: list[str] = []
        for i, (_, r) in enumerate(px.iterrows()):
            if pd.isna(r["MA20"]) or pd.isna(r["MA60"]):
                raw.append("range")
                continue
            c, m20, m60 = float(r["$close"]), float(r["MA20"]), float(r["MA60"])
            p20 = float(r["prior20"]) if pd.notna(r["prior20"]) else 0.0
            p60 = float(r["prior60"]) if pd.notna(r["prior60"]) else 0.0
            band = float(r["band60"]) if pd.notna(r["band60"]) else np.nan
            chop = pd.notna(band) and band < 0.26 and abs(p60) < 0.12
            up = (
                (m20 > m60 and c > m20 and p60 >= 0.06 and p20 >= 0.0)
                or (
                    c > m20
                    and p20 >= 0.06
                    and (m20 > m60 or p60 <= -0.08 or p60 >= 0.08)
                )
                or (m20 > m60 and c > m20 and p60 >= 0.18 and p20 >= -0.08)
            )
            classic_dn = m20 < m60 and c < m20 and p60 <= -0.06 and p20 <= 0.0
            soft_bear = c < m20 and p60 <= -0.06 and p20 <= 0.0
            soft_mom = c < m20 and p60 <= -0.06 and p20 <= -0.03
            dd20 = (
                (c / hi20[i] - 1.0)
                if (np.isfinite(c) and np.isfinite(hi20[i]) and hi20[i] > 0)
                else 0.0
            )
            peak_dn = c < m20 and p60 <= -0.06 and dd20 <= -0.08
            if mode == "strict_dn_soft_bear" or mode == "strict_dn_fast":
                down = soft_bear
            elif mode == "strict_dn_mom":
                down = soft_mom
            elif mode == "strict_dn_peak8":
                down = classic_dn or peak_dn
            else:
                down = classic_dn
            if up:
                lab = "up"
            elif down:
                lab = "down"
            elif chop:
                lab = "range"
            else:
                lab = "range"
            raw.append(lab)

        if mode == "strict_dn_fast":
            return _apply_hyst_dn_fast(ev, raw, px, confirm=3)
        above20 = (px["$close"] > px["MA20"]).fillna(False).to_numpy(bool)
        labs = ev.apply_hysteresis(
            raw,
            above20=above20,
            ma20=px["MA20"].to_numpy(float),
            ma60=px["MA60"].to_numpy(float),
            close=px["$close"].to_numpy(float),
            confirm=3,
        )
        return np.array(ev.causal_smooth(labs, 3), dtype=object)

    return method_strict


def _patch(ev, mode: str) -> None:
    if mode == "baseline":
        return
    method_strict = _build_strict_method(ev, mode)
    methods = dict(ev.METHODS)
    methods["ma_stack_strict"] = method_strict
    ev.METHODS = methods
    ev.method_ma_stack_strict = method_strict


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


def _wfrac(px, labs, a, b, lab: str = "down") -> float:
    m = (px["as_of"] >= a) & (px["as_of"] <= b)
    L = labs[m.to_numpy()]
    return float((L == lab).mean()) if len(L) else float("nan")


def _first_lab(px, labs, a, b, lab: str = "down"):
    m = (px["as_of"] >= a) & (px["as_of"] <= b)
    for d, L in zip(px.loc[m, "as_of"], labs[m.to_numpy()]):
        if L == lab:
            return pd.Timestamp(d).strftime("%Y-%m-%d")
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", default="2026-04-01")
    p.add_argument("--end-date", default="2026-08-07")
    p.add_argument("--train-cutoff", default="2026-04-01")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260807_strict_early_down_ab",
    )
    p.add_argument("--code", action="append", default=[])
    p.add_argument("--max-codes", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pm = _load(
        "pm_strict_early_dn",
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
            f"ev_strict_early_dn_{mode}",
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
                "picked_strict": method == "ma_stack_strict",
                "up_frac": float((labs[audit.to_numpy()] == "up").mean()),
                "down_frac": float((labs[audit.to_numpy()] == "down").mean()),
                **m,
            }
            per[mode].append(row)
            if code == FOCUS:
                focus_rows.append(
                    {
                        "variant": mode,
                        "method": method,
                        "miss_down": _wfrac(px, labs, MISS_START, MISS_END, "down"),
                        "full_down": _wfrac(px, labs, FULL_START, FULL_END, "down"),
                        "first_down": _first_lab(px, labs, FULL_START, FULL_END, "down"),
                        "edge": m["edge"],
                        "n_closed": m["n_closed"],
                        "win": m["win"],
                    }
                )
            print(
                f"[{mode}] {i}/{len(codes)} {code} -> {method} "
                f"win={m['win']} n={m['n_closed']} dn={row['down_frac']:.2%}",
                flush=True,
            )

    base = pd.DataFrame(per["baseline"]).set_index("code")
    wide = base.add_prefix("baseline_").reset_index()
    for mode in VARIANTS[1:]:
        df = pd.DataFrame(per[mode]).set_index("code")
        wide = wide.merge(df.add_prefix(f"{mode}_").reset_index(), on="code", how="outer")
    wide.to_csv(out_dir / "symbol_comparison.csv", index=False, encoding="utf-8-sig")

    strict_codes = set(base.index[base["method"] == "ma_stack_strict"])

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
        s["n_strict"] = int((df["method"] == "ma_stack_strict").sum())
        s["mean_down_frac"] = float(pd.to_numeric(df["down_frac"], errors="coerce").mean())
        sub = df[df["code"].isin(strict_codes)]
        if len(sub):
            n = int(sub["n_closed"].sum())
            w = int(sub["wins"].sum())
            s["strict_cohort_n"] = n
            s["strict_cohort_win"] = float(w / n) if n else float("nan")
            s["strict_cohort_edge"] = float(
                pd.to_numeric(sub["edge"], errors="coerce").mean()
            )
        else:
            s["strict_cohort_n"] = 0
            s["strict_cohort_win"] = float("nan")
            s["strict_cohort_edge"] = float("nan")
        pool_rows.append(s)
    pd.DataFrame(pool_rows).to_csv(
        out_dir / "pool_comparison.csv", index=False, encoding="utf-8-sig"
    )
    focus_df = pd.DataFrame(focus_rows)
    focus_df.to_csv(out_dir / f"{FOCUS}_focus.csv", index=False, encoding="utf-8-sig")
    (out_dir / "baseline_strict_codes.json").write_text(
        json.dumps(sorted(strict_codes), indent=2), encoding="utf-8"
    )

    b = pool_rows[0]
    fb = next(r for r in focus_rows if r["variant"] == "baseline")
    decisions = {}
    for s in pool_rows[1:]:
        mode = s["variant"]
        fr = next(r for r in focus_rows if r["variant"] == mode)
        base_first = fb.get("first_down") or "2099-01-01"
        cand_first = fr.get("first_down") or "2099-01-01"
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
            "strict_cohort_win_noninferior": (
                s["strict_cohort_win"] >= b["strict_cohort_win"] - 1e-9
                if np.isfinite(s["strict_cohort_win"])
                and np.isfinite(b["strict_cohort_win"])
                else False
            ),
            "strict_cohort_edge_noninferior": s["strict_cohort_edge"]
            >= b["strict_cohort_edge"] - 1e-6,
            "sh515060_miss_down_ge_50": float(fr["miss_down"]) >= 0.50,
            "sh515060_first_down_earlier": cand_first < base_first,
            "sh515060_still_strict": fr["method"] == "ma_stack_strict",
        }
        pool_ok = all(
            gates[k]
            for k in (
                "win_improved_or_equal",
                "enough_trades",
                "mean_net_noninferior",
                "edge_noninferior",
                "drawdown_controlled",
                "strict_cohort_win_noninferior",
                "strict_cohort_edge_noninferior",
            )
        )
        focus_ok = gates["sh515060_still_strict"] and (
            gates["sh515060_miss_down_ge_50"] or gates["sh515060_first_down_earlier"]
        )
        decisions[mode] = {
            "gates": gates,
            "promote_candidate": bool(pool_ok and focus_ok),
            "pool": s,
            "focus": fr,
        }

    recommendation = "keep_baseline"
    for mode in (
        "strict_dn_mom",
        "strict_dn_peak8",
        "strict_dn_soft_bear",
        "strict_dn_fast",
    ):
        if decisions.get(mode, {}).get("promote_candidate"):
            recommendation = mode
            break

    summary = {
        "selection": f"B0 reselect on train < {args.train_cutoff}",
        "audit": f"{args.start_date}..{args.end_date}",
        "scope": "ma_stack_strict only; hyst/struct/hybrid/dual untouched",
        "case": (
            f"{FOCUS}: miss-window {MISS_START}..{MISS_END} should gain down labels; "
            "baseline first down was 2026-06-08"
        ),
        "baseline_strict_codes": sorted(strict_codes),
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
                        "n_strict",
                        "mean_down_frac",
                        "strict_cohort_win",
                        "strict_cohort_edge",
                        "strict_cohort_n",
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
        "# ma_stack_strict early-down A/B (SH515060)",
        "",
        "- Scope: **only** `ma_stack_strict` patched.",
        f"- Baseline strict cohort ({len(strict_codes)}).",
        f"- Focus `{FOCUS}` miss-window `{MISS_START}`..`{MISS_END}`.",
        "- Production **not** modified.",
        "",
        "## Pool",
        "",
        "| Variant | Win | n | Edge | #strict | Strict-cohort win | Strict-cohort edge | mean down% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in pool_rows:
        lines.append(
            f"| {s['variant']} | {s['trade_win']:.1%} | {s['n_closed']} | "
            f"{s['mean_edge']:+.2%} | {s['n_strict']} | "
            f"{s['strict_cohort_win']:.1%} | {s['strict_cohort_edge']:+.2%} | "
            f"{s['mean_down_frac']:.1%} |"
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
                s["n_strict"],
                round(s["mean_down_frac"], 3),
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
