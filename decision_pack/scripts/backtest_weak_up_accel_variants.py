#!/usr/bin/env python3
"""A/B: tighten weak-ADX/weak-prior up entries + post-leave acceleration re-entry.

Target case: SH513650 Jul-2026 — false/weak up 7/6–7/27, missed thrust after 7/28.

Variants (monkeypatch; no production commit):
  - baseline
  - hybrid_tight: hybrid keeps up only if ADX>=18 or p20>=0.04 (block p60-only weak ADX)
  - prior_enter: hyst/strict classic enter needs p60>=0.04 and p20>=0.0 (drop bare p20=0.03)
  - accel_reenter: after leave-up, if bull stack + C>MA20 + p20>=0.06 → force up
  - combo: hybrid_tight + prior_enter + accel_reenter

Selection: B0 on train < cutoff. Audit OOS is descriptive only.
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

FOCUS_CODE = "SH513650"
FOCUS_START = "2026-07-06"
FOCUS_END = "2026-07-28"
THRUST_START = "2026-07-28"
THRUST_END = "2026-08-07"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _apply_accel_reenter(px: pd.DataFrame, labs: np.ndarray) -> np.ndarray:
    """Force up on post-leave acceleration (or cold thrust if no prior up).

    Runs **after** ``finalize_regime_labels`` so ``min_seg=10`` cannot erase
    short thrust segments. Cold-start (p20>=0.07) covers cases where a tight
    gate never entered the prior weak up, so there is no leave to re-enter from.
    """
    out = np.asarray(labs, dtype=object).copy()
    close = px["$close"].to_numpy(float)
    m20 = px["MA20"].to_numpy(float)
    m60 = px["MA60"].to_numpy(float)
    p20 = px["prior20"].to_numpy(float)
    left_up_recent = False
    since_leave = 999
    in_accel = False
    for i in range(len(out)):
        if i > 0 and out[i - 1] == "up" and out[i] != "up" and not in_accel:
            left_up_recent = True
            since_leave = 0
        if left_up_recent and not in_accel:
            since_leave += 1
            if since_leave > 25:
                left_up_recent = False
        ok = (
            np.isfinite(m20[i])
            and np.isfinite(m60[i])
            and np.isfinite(p20[i])
            and np.isfinite(close[i])
            and m20[i] > m60[i]
            and close[i] > m20[i]
        )
        if in_accel:
            if ok and p20[i] >= 0.04:
                out[i] = "up"
            else:
                in_accel = False
                left_up_recent = False
                since_leave = 999
            continue
        if out[i] == "up" or not ok:
            continue
        fire = (left_up_recent and p20[i] >= 0.06) or (p20[i] >= 0.07)
        if fire:
            out[i] = "up"
            in_accel = True
            left_up_recent = False
    return out


def _patch(ev, mode: str) -> None:
    """Monkeypatch METHODS (+ optional run_method for accel). No finalize here."""
    if mode == "baseline":
        return

    tight_hybrid = mode in {"hybrid_tight", "combo"}
    tight_prior = mode in {"prior_enter", "combo"}
    use_accel = mode in {"accel_reenter", "combo"}

    def raw_labels(px: pd.DataFrame, *, soft_hold60: bool) -> list[str]:
        raw: list[str] = []
        for _, r in px.iterrows():
            if pd.isna(r["MA20"]) or pd.isna(r["MA60"]):
                raw.append("range")
                continue
            c, m20, m60 = float(r["$close"]), float(r["MA20"]), float(r["MA60"])
            p20 = float(r["prior20"]) if pd.notna(r["prior20"]) else 0.0
            p60 = float(r["prior60"]) if pd.notna(r["prior60"]) else 0.0
            band = float(r["band60"]) if pd.notna(r["band60"]) else np.nan
            hug = abs(m20 - m60) / c < 0.02
            chop = (pd.notna(band) and band < 0.22 and abs(p60) < 0.10) or (
                hug and abs(p60) < 0.12
            )
            if soft_hold60:
                up = (
                    m20 > m60
                    and c > m20
                    and c >= m60
                    and (p60 >= 0.02 or p20 >= 0.02)
                )
                down = (
                    m20 < m60
                    and c < m20
                    and c <= m60
                    and (p60 <= -0.02 or p20 <= -0.02)
                )
            else:
                if tight_prior:
                    # Drop bare p20>=0.03 enter; need clearer intermediate trend.
                    up = m20 > m60 and c > m20 and p60 >= 0.04 and p20 >= 0.0
                else:
                    up = (m20 > m60 and c > m20 and p60 >= 0.03) or (
                        m20 > m60 and c > m20 and p20 >= 0.03
                    )
                if c > m20 and p20 >= 0.06 and (
                    m20 > m60 or p60 <= -0.08 or p60 >= 0.08
                ):
                    up = True
                down = (m20 < m60 and c < m20 and p60 <= -0.03) or (
                    m20 < m60 and c < m20 and p20 <= -0.03
                )
            if up and not down:
                lab = "up"
            elif down and not up:
                lab = "down"
            elif chop:
                lab = "range"
            else:
                lab = "range"
            raw.append(lab)
        return raw

    def method_hyst(px: pd.DataFrame) -> np.ndarray:
        raw = raw_labels(px, soft_hold60=False)
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

    def method_struct(px: pd.DataFrame) -> np.ndarray:
        raw = raw_labels(px, soft_hold60=True)
        labs = ev.apply_hysteresis_ma60(
            raw,
            close=px["$close"].to_numpy(float),
            ma20=px["MA20"].to_numpy(float),
            ma60=px["MA60"].to_numpy(float),
            confirm_leave=3,
            confirm_enter=3,
            soft_leave_below20=3,
            soft_leave_drawdown=0.12,
        )
        return np.array(ev.causal_smooth(labs, 3), dtype=object)

    def method_strict(px: pd.DataFrame) -> np.ndarray:
        raw = []
        for _, r in px.iterrows():
            if pd.isna(r["MA20"]) or pd.isna(r["MA60"]):
                raw.append("range")
                continue
            c, m20, m60 = float(r["$close"]), float(r["MA20"]), float(r["MA60"])
            p20 = float(r["prior20"]) if pd.notna(r["prior20"]) else 0.0
            p60 = float(r["prior60"]) if pd.notna(r["prior60"]) else 0.0
            band = float(r["band60"]) if pd.notna(r["band60"]) else np.nan
            chop = pd.notna(band) and band < 0.26 and abs(p60) < 0.12
            p60_thr = 0.08 if tight_prior else 0.06
            up = (
                (m20 > m60 and c > m20 and p60 >= p60_thr and p20 >= 0.0)
                or (
                    c > m20
                    and p20 >= 0.06
                    and (m20 > m60 or p60 <= -0.08 or p60 >= 0.08)
                )
                or (m20 > m60 and c > m20 and p60 >= 0.18 and p20 >= -0.08)
            )
            down = m20 < m60 and c < m20 and p60 <= -0.06 and p20 <= 0.0
            if up:
                lab = "up"
            elif down:
                lab = "down"
            elif chop:
                lab = "range"
            else:
                lab = "range"
            raw.append(lab)
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

    def method_hybrid(px: pd.DataFrame) -> np.ndarray:
        base = method_hyst(px)
        out = []
        for i, lab in enumerate(base):
            adx = float(px.iloc[i]["adx14"]) if pd.notna(px.iloc[i]["adx14"]) else 0.0
            p20 = float(px.iloc[i]["prior20"]) if pd.notna(px.iloc[i]["prior20"]) else 0.0
            p60 = float(px.iloc[i]["prior60"]) if pd.notna(px.iloc[i]["prior60"]) else 0.0
            c = float(px.iloc[i]["$close"])
            m20 = float(px.iloc[i]["MA20"]) if pd.notna(px.iloc[i]["MA20"]) else np.nan
            above = np.isfinite(m20) and c > m20
            if tight_hybrid:
                # Block weak-ADX ups propped up only by intermediate p60.
                keep_up = adx >= 18 or (above and p20 >= 0.04)
                keep_dn = adx >= 18 or abs(p20) >= 0.04 or abs(p60) >= 0.06
                if lab == "up" and not keep_up:
                    out.append("range")
                elif lab == "down" and not keep_dn:
                    out.append("range")
                else:
                    out.append(lab)
            else:
                mom = abs(p60) >= 0.06 or abs(p20) >= 0.04
                if lab in ("up", "down") and adx < 15 and not mom and not (
                    lab == "up" and above and p20 >= 0.02
                ):
                    out.append("range")
                else:
                    out.append(lab)
        return np.asarray(out, dtype=object)

    ev._ma_stack_raw_labels = raw_labels  # type: ignore[attr-defined]
    ev.method_ma_stack = method_hyst
    ev.method_ma_stack_struct = method_struct
    ev.method_ma_stack_strict = method_strict
    ev.method_hybrid_v1 = method_hybrid
    ev.METHODS = {
        "ma_stack_hyst": method_hyst,
        "ma_stack_struct": method_struct,
        "ma_stack_strict": method_strict,
        "dual_ma_cross": ev.method_dual_ma_cross,
        "adx_di": ev.method_adx_di,
        "prior60_band": ev.method_prior60_band,
        "hybrid_ma_adx": method_hybrid,
    }

    if use_accel:
        orig_run = ev.run_method

        def run_method(name: str, px: pd.DataFrame, *, params=None, min_seg: int = 10):
            # Finalize first (production), then accel so min_seg cannot erase thrust.
            labs = orig_run(name, px, params=params, min_seg=min_seg)
            return _apply_accel_reenter(px, labs)

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


def _focus_stats(px: pd.DataFrame, labs: np.ndarray) -> dict[str, Any]:
    weak = (px["as_of"] >= FOCUS_START) & (px["as_of"] <= FOCUS_END)
    thrust = (px["as_of"] >= THRUST_START) & (px["as_of"] <= THRUST_END)
    Lw = labs[weak.to_numpy()]
    Lt = labs[thrust.to_numpy()]
    cw = px.loc[weak, "$close"].to_numpy(float)
    ct = px.loc[thrust, "$close"].to_numpy(float)
    return {
        "weak_up_frac": float((Lw == "up").mean()) if len(Lw) else np.nan,
        "weak_n_up": int((Lw == "up").sum()),
        "weak_seg_ret": float(cw[-1] / cw[0] - 1) if len(cw) > 1 else 0.0,
        "thrust_up_frac": float((Lt == "up").mean()) if len(Lt) else np.nan,
        "thrust_n_up": int((Lt == "up").sum()),
        "thrust_seg_ret": float(ct[-1] / ct[0] - 1) if len(ct) > 1 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", default="2026-04-01")
    p.add_argument("--end-date", default="2026-08-07")
    p.add_argument("--train-cutoff", default="2026-04-01")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260807_weak_up_accel_ab",
    )
    p.add_argument("--code", action="append", default=[])
    p.add_argument("--max-codes", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pm = _load(
        "pm_weak_up",
        _ROOT / "decision_pack/scripts/plot_regime_transition_example.py",
    )
    pm._init_qlib(None)
    codes = [str(x).upper() for x in args.code] or list(
        load_cluster_mapping_codes(CLUSTER_MAPPING_SELECTED_TXT)
    )
    if args.max_codes:
        codes = codes[: args.max_codes]

    variants = (
        "baseline",
        "hybrid_tight",
        "prior_enter",
        "accel_reenter",
        "combo",
    )

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

    per_variant_rows: dict[str, list[dict[str, Any]]] = {v: [] for v in variants}
    focus_rows: list[dict[str, Any]] = []

    for mode in variants:
        ev = _load(
            f"ev_weak_{mode}",
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
                **m,
            }
            per_variant_rows[mode].append(row)
            if code == FOCUS_CODE:
                focus_rows.append(
                    {
                        "variant": mode,
                        "method": method,
                        **_focus_stats(px, labs),
                    }
                )
            print(
                f"[{mode}] {i}/{len(codes)} {code} -> {method} "
                f"win={m['win']} n={m['n_closed']}"
            )

    base = pd.DataFrame(per_variant_rows["baseline"]).set_index("code")
    frames = [base.add_prefix("baseline_").reset_index()]
    for mode in variants[1:]:
        df = pd.DataFrame(per_variant_rows[mode]).set_index("code")
        frames.append(df.add_prefix(f"{mode}_").reset_index())
    wide = frames[0]
    for f in frames[1:]:
        wide = wide.merge(f, on="code", how="outer")
    wide.to_csv(out_dir / "symbol_comparison.csv", index=False, encoding="utf-8-sig")

    pool_rows = []
    for mode in variants:
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
        pool_rows.append(s)
    pool = pd.DataFrame(pool_rows)
    pool.to_csv(out_dir / "pool_comparison.csv", index=False, encoding="utf-8-sig")
    focus_df = pd.DataFrame(focus_rows)
    focus_df.to_csv(out_dir / f"{FOCUS_CODE}_focus.csv", index=False, encoding="utf-8-sig")

    b = pool_rows[0]
    decisions: dict[str, Any] = {}
    for s in pool_rows[1:]:
        mode = s["variant"]
        fr = next(r for r in focus_rows if r["variant"] == mode)
        fb = next(r for r in focus_rows if r["variant"] == "baseline")
        gates = {
            "win_improved_or_equal": (
                s["trade_win"] >= b["trade_win"] - 1e-9
                if np.isfinite(s["trade_win"]) and np.isfinite(b["trade_win"])
                else False
            ),
            "enough_trades": s["n_closed"] >= max(8, int(0.75 * b["n_closed"])),
            "mean_net_noninferior": s["mean_ret_net"] >= b["mean_ret_net"] - 1e-6,
            "edge_noninferior": s["mean_edge"] >= b["mean_edge"] - 1e-6,
            "drawdown_controlled": s["worst_drawdown"]
            >= b["worst_drawdown"] - 0.02,
            "sh513650_weak_up_reduced": float(fr["weak_up_frac"])
            <= float(fb["weak_up_frac"]) - 0.25,
            "sh513650_thrust_up_improved": float(fr["thrust_up_frac"])
            >= float(fb["thrust_up_frac"]) + 0.25,
        }
        # Promote if pool non-inferior AND (fixes weak up OR catches thrust)
        focus_ok = gates["sh513650_weak_up_reduced"] or gates["sh513650_thrust_up_improved"]
        pool_ok = all(
            [
                gates["win_improved_or_equal"],
                gates["enough_trades"],
                gates["mean_net_noninferior"],
                gates["edge_noninferior"],
                gates["drawdown_controlled"],
            ]
        )
        decisions[mode] = {
            "gates": gates,
            "promote_candidate": bool(pool_ok and focus_ok),
            "pool": s,
            "focus": fr,
        }

    promoted = [m for m, d in decisions.items() if d["promote_candidate"]]
    # Prefer combo, then hybrid_tight, then others by mean_edge
    order = ["combo", "hybrid_tight", "prior_enter", "accel_reenter"]
    recommendation = "keep_baseline"
    for m in order:
        if m in promoted:
            recommendation = m
            break
    if recommendation == "keep_baseline" and promoted:
        recommendation = max(
            promoted, key=lambda m: decisions[m]["pool"]["mean_edge"]
        )

    summary = {
        "selection": f"B0 reselect on train < {args.train_cutoff}",
        "audit": f"{args.start_date}..{args.end_date}",
        "audit_used_for_selection": False,
        "n_symbols": len(codes),
        "focus": {
            "code": FOCUS_CODE,
            "weak_window": f"{FOCUS_START}..{FOCUS_END}",
            "thrust_window": f"{THRUST_START}..{THRUST_END}",
        },
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
                    )
                },
                "focus": v["focus"],
            }
            for k, v in decisions.items()
        },
        "recommendation": recommendation,
    }
    (out_dir / "acceptance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    lines = [
        "# Weak-up / accel re-enter A/B (audit descriptive)",
        "",
        f"- Train/select: `< {args.train_cutoff}` (B0 legacy, hold_up).",
        f"- Audit: `{args.start_date}` → `{args.end_date}` (not used to pick).",
        f"- Focus: `{FOCUS_CODE}` weak `{FOCUS_START}..{FOCUS_END}`, "
        f"thrust `{THRUST_START}..{THRUST_END}`.",
        "- Production code **not** modified.",
        "",
        "## Pool",
        "",
        "| Variant | Trade win | n | Mean edge | Mean net | Worst DD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in pool_rows:
        lines.append(
            f"| {s['variant']} | {s['trade_win']:.1%} | {s['n_closed']} | "
            f"{s['mean_edge']:+.2%} | {s['mean_ret_net']:+.2%} | {s['worst_drawdown']:.1%} |"
        )
    lines += [
        "",
        f"## {FOCUS_CODE} focus",
        "",
        focus_df.to_string(index=False),
        "",
        f"## Recommendation: `{recommendation}`",
        "",
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
            s["variant"]: (round(s["trade_win"], 3), s["n_closed"], round(s["mean_edge"], 4))
            for s in pool_rows
        },
        "recommend=",
        recommendation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
