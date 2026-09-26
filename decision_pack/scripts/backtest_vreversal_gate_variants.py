#!/usr/bin/env python3
"""Compare V-reversal gate tightenings vs current production (no code commit).

Variants (applied to hyst / strict / hybrid via monkeypatch):
  - baseline: current production logic (unchanged module)
  - no_washout: drop ``p60 <= -0.08`` only; keep stack or ``p60 >= 0.08``
  - stack_only: V-reversal requires ``MA20 > MA60`` (no washout / bare thrust)

For each variant: re-run B0 selection on train (< cutoff), then hold_up OOS.
Audit window is descriptive; selection does not use OOS.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

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


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _patch_vreversal(ev, mode: str) -> None:
    """Replace METHODS that embed the gated V-reversal with a variant."""
    if mode == "baseline":
        return

    def vrev_ok(c, m20, m60, p20, p60) -> bool:
        if not (c > m20 and p20 >= 0.06):
            return False
        if mode == "stack_only":
            return m20 > m60
        if mode == "no_washout":
            return m20 > m60 or p60 >= 0.08
        raise ValueError(mode)

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
                up = (m20 > m60 and c > m20 and p60 >= 0.03) or (
                    m20 > m60 and c > m20 and p20 >= 0.03
                )
                if vrev_ok(c, m20, m60, p20, p60):
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
            up = (
                (m20 > m60 and c > m20 and p60 >= 0.06 and p20 >= 0.0)
                or vrev_ok(c, m20, m60, p20, p60)
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
            mom = abs(p60) >= 0.06 or abs(p20) >= 0.04
            above = np.isfinite(m20) and c > m20
            if lab in ("up", "down") and adx < 15 and not mom and not (
                lab == "up" and above and p20 >= 0.02
            ):
                out.append("range")
            else:
                out.append(lab)
        return np.asarray(out, dtype=object)

    # Keep patched raw helper consistent for any internal callers.
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
    p.add_argument("--end-date", default="2026-08-07")
    p.add_argument("--train-cutoff", default="2026-04-01")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260807_vreversal_gate_ab",
    )
    p.add_argument("--code", action="append", default=[])
    p.add_argument("--max-codes", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pm = _load(
        "pm_vrev",
        _ROOT / "decision_pack/scripts/plot_regime_transition_example.py",
    )
    pm._init_qlib(None)
    codes = [str(x).upper() for x in args.code] or list(
        load_cluster_mapping_codes(CLUSTER_MAPPING_SELECTED_TXT)
    )
    if args.max_codes:
        codes = codes[: args.max_codes]

    variants = ("baseline", "no_washout", "stack_only")
    # Preload OHLCV once
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
        # Fresh module each variant so patches don't leak.
        ev = _load(
            f"ev_vrev_{mode}",
            _ROOT / "decision_pack/scripts/eval_causal_regime_switch_pool.py",
        )
        _patch_vreversal(ev, mode)
        for i, code in enumerate(codes, 1):
            px = prepare_features(ev, ohlcvs[code], method=None)
            px_train = px[px["as_of"] < args.train_cutoff].reset_index(drop=True)
            method, _, meta = select_regime_method_legacy(
                ev, px_train, trade_mode=TRADE_MODE_HOLD_UP
            )
            labs = np.asarray(ev.run_method(method, px), dtype=object)
            audit = (px["as_of"] >= args.start_date) & (px["as_of"] <= args.end_date)
            close = px.loc[audit, "$close"].to_numpy(float)
            dates = px.loc[audit, "as_of"].dt.strftime("%Y-%m-%d").to_numpy()
            m, trades, _ = hold_up_trade_metrics(
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

            # Focus SH513090 false-up window
            if code == "SH513090":
                win = (px["as_of"] >= "2026-04-22") & (px["as_of"] <= "2026-05-25")
                L = labs[win.to_numpy()]
                cwin = px.loc[win, "$close"].to_numpy(float)
                focus_rows.append(
                    {
                        "variant": mode,
                        "method": method,
                        "up_frac": float((L == "up").mean()) if len(L) else np.nan,
                        "seg_ret": float(cwin[-1] / cwin[0] - 1) if len(cwin) > 1 else 0.0,
                        "n_up_days": int((L == "up").sum()),
                    }
                )
            print(f"[{mode}] {i}/{len(codes)} {code} -> {method} win={m['win']} n={m['n_closed']}")

    # Wide symbol comparison
    base = pd.DataFrame(per_variant_rows["baseline"]).set_index("code")
    frames = [base.add_prefix("baseline_").reset_index()]
    for mode in ("no_washout", "stack_only"):
        df = pd.DataFrame(per_variant_rows[mode]).set_index("code")
        frames.append(df.add_prefix(f"{mode}_").reset_index())
    wide = frames[0]
    for f in frames[1:]:
        wide = wide.merge(f, on="code", how="outer")
    wide.to_csv(out_dir / "symbol_comparison.csv", index=False, encoding="utf-8-sig")

    pool_rows = []
    for mode in variants:
        df = pd.DataFrame(per_variant_rows[mode])
        # reuse _pool with synthetic prefix by renaming
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
    pd.DataFrame(focus_rows).to_csv(
        out_dir / "SH513090_focus.csv", index=False, encoding="utf-8-sig"
    )

    # Acceptance vs baseline (audit descriptive)
    b = pool_rows[0]
    decisions = {}
    for s in pool_rows[1:]:
        mode = s["variant"]
        gates = {
            "win_improved_or_equal": (
                s["trade_win"] >= b["trade_win"]
                if np.isfinite(s["trade_win"]) and np.isfinite(b["trade_win"])
                else False
            ),
            "enough_trades": s["n_closed"] >= max(10, int(0.8 * b["n_closed"])),
            "mean_net_noninferior": s["mean_ret_net"] >= b["mean_ret_net"] - 1e-6,
            "edge_noninferior": s["mean_edge"] >= b["mean_edge"] - 1e-6,
            "drawdown_controlled": s["worst_drawdown"]
            >= b["worst_drawdown"] - 0.02,
            "sh513090_false_up_fixed": bool(
                next(
                    (
                        r["up_frac"] < 0.5
                        for r in focus_rows
                        if r["variant"] == mode
                    ),
                    False,
                )
            ),
        }
        decisions[mode] = {
            "gates": gates,
            "promote_candidate": bool(all(gates.values())),
            "pool": s,
        }

    summary = {
        "selection": f"B0 reselect on train < {args.train_cutoff}",
        "audit": f"{args.start_date}..{args.end_date}",
        "audit_used_for_selection": False,
        "n_symbols": len(codes),
        "baseline": b,
        "variants": decisions,
        "recommendation": (
            "stack_only"
            if decisions.get("stack_only", {}).get("promote_candidate")
            else (
                "no_washout"
                if decisions.get("no_washout", {}).get("promote_candidate")
                else "keep_baseline"
            )
        ),
    }
    (out_dir / "acceptance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # README
    lines = [
        "# V-reversal gate A/B (audit descriptive)",
        "",
        f"- Train/select: `< {args.train_cutoff}` (B0 legacy, hold_up exclusions).",
        f"- Audit: `{args.start_date}` → `{args.end_date}` (not used to pick).",
        "- Production code **not** modified by this script.",
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
        f"## SH513090 4/22–5/25 false-up check",
        "",
        pd.DataFrame(focus_rows).to_markdown(index=False)
        if hasattr(pd.DataFrame(focus_rows), "to_markdown")
        else str(pd.DataFrame(focus_rows)),
        "",
        f"## Recommendation: `{summary['recommendation']}`",
        "",
        "```json",
        json.dumps({k: v["gates"] for k, v in decisions.items()}, ensure_ascii=False, indent=2),
        "```",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        "DONE",
        {s["variant"]: (s["trade_win"], s["n_closed"], s["mean_edge"]) for s in pool_rows},
        "recommend=",
        summary["recommendation"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
