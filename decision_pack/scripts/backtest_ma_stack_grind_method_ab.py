#!/usr/bin/env python3
"""A/B: add optional METHOD ``ma_stack_grind`` to the B0 pool (no production commit).

Variants
--------
baseline
    Production METHODS only.
with_grind
    Same pool + ``ma_stack_grind`` (soft prior enter, soft leave, min_seg=5).
    B0 reselect on train; symbols may or may not pick it.

Audit OOS is descriptive only.
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

FOCUS = ("SH513650", "SH513400", "SZ159561")
THRUST_START = "2026-07-28"
THRUST_END = "2026-08-07"
WEAK_START = "2026-07-06"
WEAK_END = "2026-07-28"
VARIANTS = ("baseline", "with_grind")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _patch_add_grind(ev) -> None:
    """Register ma_stack_grind and use min_seg=5 for that method only."""

    def method_ma_stack_grind(px: pd.DataFrame) -> np.ndarray:
        """Soft-prior MA stack for slow grind / low absolute vol trends.

        - Enter up on stack with milder p60/p20 than hyst/strict.
        - Soft leave: 3d below MA20 or 10% peak drawdown (less twitchy exits).
        - No ADX gate (hybrid's weak-ADX veto is what hurts SH513650-class).
        """
        raw: list[str] = []
        for _, r in px.iterrows():
            if pd.isna(r["MA20"]) or pd.isna(r["MA60"]):
                raw.append("range")
                continue
            c = float(r["$close"])
            m20, m60 = float(r["MA20"]), float(r["MA60"])
            p20 = float(r["prior20"]) if pd.notna(r["prior20"]) else 0.0
            p60 = float(r["prior60"]) if pd.notna(r["prior60"]) else 0.0
            band = float(r["band60"]) if pd.notna(r["band60"]) else np.nan
            hug = abs(m20 - m60) / c < 0.02
            chop = (pd.notna(band) and band < 0.22 and abs(p60) < 0.08) or (
                hug and abs(p60) < 0.10
            )
            up = (m20 > m60 and c > m20 and p60 >= 0.04) or (
                m20 > m60 and c > m20 and p20 >= 0.04
            )
            # Mild thrust while stacked (catch acceleration without ADX).
            if m20 > m60 and c > m20 and p20 >= 0.05:
                up = True
            down = (m20 < m60 and c < m20 and p60 <= -0.04) or (
                m20 < m60 and c < m20 and p20 <= -0.04
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

        above20 = (px["$close"] > px["MA20"]).fillna(False).to_numpy(bool)
        labs = ev.apply_hysteresis(
            raw,
            above20=above20,
            ma20=px["MA20"].to_numpy(float),
            ma60=px["MA60"].to_numpy(float),
            close=px["$close"].to_numpy(float),
            confirm=3,
            soft_leave_below20=3,
            soft_leave_drawdown=0.10,
        )
        return np.array(ev.causal_smooth(labs, 3), dtype=object)

    ev.method_ma_stack_grind = method_ma_stack_grind  # type: ignore[attr-defined]
    methods = dict(ev.METHODS)
    methods["ma_stack_grind"] = method_ma_stack_grind
    ev.METHODS = methods

    orig_run = ev.run_method

    def run_method(name: str, px: pd.DataFrame, *, params=None, min_seg: int = 10):
        if name == "ma_stack_grind":
            return orig_run(name, px, params=params, min_seg=5)
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
        default=PLOTLY_OUTPUTS_DIR / "20260807_ma_stack_grind_method_ab",
    )
    p.add_argument("--code", action="append", default=[])
    p.add_argument("--max-codes", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pm = _load(
        "pm_grind_method",
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
            f"ev_grind_method_{mode}",
            _ROOT / "decision_pack/scripts/eval_causal_regime_switch_pool.py",
        )
        if mode == "with_grind":
            _patch_add_grind(ev)
        for i, code in enumerate(codes, 1):
            px = prepare_features(ev, ohlcvs[code], method=None)
            px_train = px[px["as_of"] < args.train_cutoff].reset_index(drop=True)
            method, scores, meta = select_regime_method_legacy(
                ev, px_train, trade_mode=TRADE_MODE_HOLD_UP
            )
            labs = np.asarray(ev.run_method(method, px), dtype=object)
            audit = (px["as_of"] >= args.start_date) & (px["as_of"] <= args.end_date)
            close = px.loc[audit, "$close"].to_numpy(float)
            dates = px.loc[audit, "as_of"].dt.strftime("%Y-%m-%d").to_numpy()
            m, _trades, _ = hold_up_trade_metrics(
                labs[audit.to_numpy()], close, dates=dates
            )
            method_score = None
            ms = meta.get("score")
            if isinstance(ms, (int, float)) and ms == ms:
                method_score = float(ms)
            else:
                score_map = meta.get("scores") if isinstance(meta.get("scores"), dict) else scores
                if isinstance(score_map, dict):
                    v = score_map.get(method)
                    if isinstance(v, (int, float)) and v == v:
                        method_score = float(v)
            row = {
                "code": code,
                "variant": mode,
                "method": method,
                "method_score": method_score,
                "picked_grind": method == "ma_stack_grind",
                "up_frac": float((labs[audit.to_numpy()] == "up").mean()),
                **m,
            }
            per_variant_rows[mode].append(row)
            if code in FOCUS:
                focus_rows.append(
                    {
                        "code": code,
                        "variant": mode,
                        "method": method,
                        "picked_grind": method == "ma_stack_grind",
                        "weak_up_frac": _window_up_frac(
                            px, labs, WEAK_START, WEAK_END
                        ),
                        "thrust_up_frac": _window_up_frac(
                            px, labs, THRUST_START, THRUST_END
                        ),
                        "edge": m["edge"],
                        "n_closed": m["n_closed"],
                        "win": m["win"],
                    }
                )
            print(
                f"[{mode}] {i}/{len(codes)} {code} -> {method} "
                f"win={m['win']} n={m['n_closed']}"
            )

    base = pd.DataFrame(per_variant_rows["baseline"]).set_index("code")
    wide = base.add_prefix("baseline_").reset_index()
    wg = pd.DataFrame(per_variant_rows["with_grind"]).set_index("code")
    wide = wide.merge(wg.add_prefix("with_grind_").reset_index(), on="code", how="outer")
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
        s["n_picked_grind"] = int(df["picked_grind"].sum())
        pool_rows.append(s)
    pd.DataFrame(pool_rows).to_csv(
        out_dir / "pool_comparison.csv", index=False, encoding="utf-8-sig"
    )
    focus_df = pd.DataFrame(focus_rows)
    focus_df.to_csv(out_dir / "focus_codes.csv", index=False, encoding="utf-8-sig")

    b = pool_rows[0]
    w = pool_rows[1]
    base_thrust = next(
        r["thrust_up_frac"]
        for r in focus_rows
        if r["variant"] == "baseline" and r["code"] == "SH513650"
    )
    grind_thrust = next(
        r["thrust_up_frac"]
        for r in focus_rows
        if r["variant"] == "with_grind" and r["code"] == "SH513650"
    )
    focus_pick = [
        r["code"]
        for r in focus_rows
        if r["variant"] == "with_grind" and r["picked_grind"]
    ]
    switched = wide.loc[
        wide["baseline_method"] != wide["with_grind_method"], "code"
    ].tolist()

    gates = {
        "win_improved_or_equal": (
            w["trade_win"] >= b["trade_win"] - 1e-9
            if np.isfinite(w["trade_win"]) and np.isfinite(b["trade_win"])
            else False
        ),
        "enough_trades": w["n_closed"] >= max(10, int(0.8 * b["n_closed"])),
        "mean_net_noninferior": w["mean_ret_net"] >= b["mean_ret_net"] - 1e-6,
        "edge_noninferior": w["mean_edge"] >= b["mean_edge"] - 1e-6,
        "drawdown_controlled": w["worst_drawdown"] >= b["worst_drawdown"] - 0.02,
        "grind_selected_somewhere": w["n_picked_grind"] >= 1,
        "sh513650_thrust_improved": float(grind_thrust)
        >= float(base_thrust) + 0.25,
        "focus_or_switch_uses_grind": bool(focus_pick) or ("SH513650" in switched),
    }
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
    # Promote only if pool non-inferior AND grind is actually chosen AND
    # either focus thrust improves or a focus name picks grind.
    focus_ok = gates["grind_selected_somewhere"] and (
        gates["sh513650_thrust_improved"] or bool(focus_pick)
    )
    recommendation = (
        "with_grind" if pool_ok and focus_ok else "keep_baseline"
    )

    summary = {
        "selection": f"B0 reselect on train < {args.train_cutoff}",
        "audit": f"{args.start_date}..{args.end_date}",
        "audit_used_for_selection": False,
        "n_symbols": len(codes),
        "new_method": "ma_stack_grind",
        "baseline": b,
        "with_grind": w,
        "gates": gates,
        "promote_candidate": bool(pool_ok and focus_ok),
        "focus_picked_grind": focus_pick,
        "switched_codes": switched,
        "recommendation": recommendation,
        "focus": focus_rows,
    }
    (out_dir / "acceptance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    lines = [
        "# ma_stack_grind optional METHOD A/B",
        "",
        f"- Train/select: `< {args.train_cutoff}` (B0 legacy, hold_up).",
        f"- Audit: `{args.start_date}` → `{args.end_date}`.",
        "- Production code **not** modified.",
        "- `with_grind` adds `ma_stack_grind` to METHODS; B0 may ignore it.",
        "",
        "## Pool",
        "",
        "| Variant | Trade win | n | Mean edge | Mean net | Worst DD | #pick grind |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in pool_rows:
        lines.append(
            f"| {s['variant']} | {s['trade_win']:.1%} | {s['n_closed']} | "
            f"{s['mean_edge']:+.2%} | {s['mean_ret_net']:+.2%} | "
            f"{s['worst_drawdown']:.1%} | {s['n_picked_grind']} |"
        )
    lines += [
        "",
        "### Method mix",
        "",
        f"- baseline: `{b['method_mix']}`",
        f"- with_grind: `{w['method_mix']}`",
        f"- switched: {switched}",
        "",
        "## Focus (650 / 400 / 561)",
        "",
        focus_df.to_string(index=False),
        "",
        f"## Recommendation: `{recommendation}`",
        "",
        "```json",
        json.dumps(gates, ensure_ascii=False, indent=2),
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
                s["n_picked_grind"],
            )
            for s in pool_rows
        },
        "switched=",
        switched,
        "recommend=",
        recommendation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
