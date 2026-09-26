#!/usr/bin/env python3
"""A/B: earlier up-entry rules vs production B0 (no production code change).

Variants
--------
baseline
    Current METHODS / finalize(min_seg=10).
vrev_looser
    V-reversal: p20>=0.04 and (stack or p60<=-0.05 or p60>=0.05).
thrust_nostack
    Additionally allow C>MA20 & p20>=0.08 even without MA stack.
min_seg_5
    Same entries as baseline; finalize min_seg=5 (keep short ups).
bounce_up
    When leaving down with rebound from trough >=6%, label up (not range).
early_combo
    vrev_looser + thrust_nostack + min_seg_5 + bounce_up.

Protocol: B0 reselect on train < cutoff; hold_up metrics on audit window only.
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


VARIANTS = (
    "baseline",
    "vrev_looser",
    "thrust_nostack",
    "min_seg_5",
    "bounce_up",
    "early_combo",
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg(mode: str) -> dict[str, Any]:
    return {
        "vrev_p20": 0.04 if mode in {"vrev_looser", "early_combo"} else 0.06,
        "vrev_washout": -0.05 if mode in {"vrev_looser", "early_combo"} else -0.08,
        "vrev_thrust": 0.05 if mode in {"vrev_looser", "early_combo"} else 0.08,
        "thrust_nostack": mode in {"thrust_nostack", "early_combo"},
        "thrust_p20": 0.08,
        "min_seg": 5 if mode in {"min_seg_5", "early_combo"} else 10,
        "bounce_up": mode in {"bounce_up", "early_combo"},
        "bounce": 0.06,
    }


def _apply_bounce_up(labels: np.ndarray, close: np.ndarray, *, bounce: float) -> np.ndarray:
    """If a down episode rebounds >= bounce from trough, flip from that day to up.

    Causal: trough is tracked only inside the current down streak.
    """
    src = np.asarray(labels, dtype=object)
    out = src.copy()
    close = np.asarray(close, dtype=float)
    i = 0
    n = len(out)
    while i < n:
        if src[i] != "down":
            i += 1
            continue
        j = i
        trough = float(close[i]) if np.isfinite(close[i]) else np.nan
        rebound_at = None
        while j < n and src[j] == "down":
            c = float(close[j])
            if np.isfinite(c):
                if not np.isfinite(trough) or c < trough:
                    trough = c
                elif rebound_at is None and trough > 0 and (c / trough - 1.0) >= bounce:
                    rebound_at = j
            j += 1
        if rebound_at is not None:
            out[rebound_at:j] = "up"
        i = j
    return out


def _patch(ev, mode: str) -> None:
    if mode == "baseline":
        return
    cfg = _cfg(mode)

    def vrev_ok(c, m20, m60, p20, p60) -> bool:
        if not (c > m20 and p20 >= cfg["vrev_p20"]):
            return False
        return bool(m20 > m60 or p60 <= cfg["vrev_washout"] or p60 >= cfg["vrev_thrust"])

    def thrust_ok(c, m20, p20) -> bool:
        return bool(cfg["thrust_nostack"] and c > m20 and p20 >= cfg["thrust_p20"])

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
                if vrev_ok(c, m20, m60, p20, p60) or thrust_ok(c, m20, p20):
                    up = True
                down = (m20 < m60 and c < m20 and p60 <= -0.03) or (
                    m20 < m60 and c < m20 and p20 <= -0.03
                )
            if up and not down:
                lab = "up"
            elif down and not up:
                lab = "down"
            else:
                lab = "range" if chop or True else "range"
            raw.append(lab)
        return raw

    def finalize(labs: np.ndarray, px: pd.DataFrame) -> np.ndarray:
        x = np.asarray(labs, dtype=object)
        if cfg["bounce_up"]:
            x = _apply_bounce_up(x, px["$close"].to_numpy(float), bounce=cfg["bounce"])
        return ev.finalize_regime_labels(x, px, min_seg=int(cfg["min_seg"]))

    def method_hyst(px: pd.DataFrame) -> np.ndarray:
        raw = raw_labels(px, soft_hold60=False)
        if cfg["bounce_up"]:
            raw = list(
                _apply_bounce_up(
                    np.asarray(raw, dtype=object),
                    px["$close"].to_numpy(float),
                    bounce=cfg["bounce"],
                )
            )
        above20 = (px["$close"] > px["MA20"]).fillna(False).to_numpy(bool)
        labs = ev.apply_hysteresis(
            raw,
            above20=above20,
            ma20=px["MA20"].to_numpy(float),
            ma60=px["MA60"].to_numpy(float),
            close=px["$close"].to_numpy(float),
            confirm=3,
        )
        labs = np.array(ev.causal_smooth(labs, 3), dtype=object)
        return finalize(labs, px)

    def method_struct(px: pd.DataFrame) -> np.ndarray:
        raw = raw_labels(px, soft_hold60=True)
        if cfg["bounce_up"]:
            raw = list(
                _apply_bounce_up(
                    np.asarray(raw, dtype=object),
                    px["$close"].to_numpy(float),
                    bounce=cfg["bounce"],
                )
            )
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
        labs = np.array(ev.causal_smooth(labs, 3), dtype=object)
        return finalize(labs, px)

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
                or thrust_ok(c, m20, p20)
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
        if cfg["bounce_up"]:
            raw = list(
                _apply_bounce_up(
                    np.asarray(raw, dtype=object),
                    px["$close"].to_numpy(float),
                    bounce=cfg["bounce"],
                )
            )
        above20 = (px["$close"] > px["MA20"]).fillna(False).to_numpy(bool)
        labs = ev.apply_hysteresis(
            raw,
            above20=above20,
            ma20=px["MA20"].to_numpy(float),
            ma60=px["MA60"].to_numpy(float),
            close=px["$close"].to_numpy(float),
            confirm=3,
        )
        labs = np.array(ev.causal_smooth(labs, 3), dtype=object)
        return finalize(labs, px)

    def method_hybrid(px: pd.DataFrame) -> np.ndarray:
        # Build from patched hyst path but without double finalize: call pieces.
        raw = raw_labels(px, soft_hold60=False)
        if cfg["bounce_up"]:
            raw = list(
                _apply_bounce_up(
                    np.asarray(raw, dtype=object),
                    px["$close"].to_numpy(float),
                    bounce=cfg["bounce"],
                )
            )
        above20 = (px["$close"] > px["MA20"]).fillna(False).to_numpy(bool)
        labs = ev.apply_hysteresis(
            raw,
            above20=above20,
            ma20=px["MA20"].to_numpy(float),
            ma60=px["MA60"].to_numpy(float),
            close=px["$close"].to_numpy(float),
            confirm=3,
        )
        base = np.array(ev.causal_smooth(labs, 3), dtype=object)
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
        return finalize(np.asarray(out, dtype=object), px)

    # Bypass run_method's hardcoded finalize min_seg by wrapping METHODS to
    # already-finalized outputs, and make run_method a thin dispatcher.
    ev.METHODS = {
        "ma_stack_hyst": method_hyst,
        "ma_stack_struct": method_struct,
        "ma_stack_strict": method_strict,
        "dual_ma_cross": ev.method_dual_ma_cross,
        "adx_di": ev.method_adx_di,
        "prior60_band": ev.method_prior60_band,
        "hybrid_ma_adx": method_hybrid,
    }

    def run_method(name: str, px: pd.DataFrame, *, params=None, min_seg: int = 10):
        # For patched MA family, METHODS already finalize with variant min_seg.
        if name in {
            "ma_stack_hyst",
            "ma_stack_struct",
            "ma_stack_strict",
            "hybrid_ma_adx",
        }:
            return np.asarray(ev.METHODS[name](px), dtype=object)
        # Others: keep original finalize but honor variant min_seg.
        labs = ev.METHODS[name](px)
        return ev.finalize_regime_labels(
            labs, px, min_seg=int(cfg["min_seg"])
        )

    ev.run_method = run_method  # type: ignore[assignment]


def _pool(df: pd.DataFrame) -> dict[str, Any]:
    n = int(df["n_closed"].sum())
    wins = int(df["wins"].sum())
    traded = int((df["n_closed"] > 0).sum())
    return {
        "n_closed": n,
        "wins": wins,
        "trade_win": float(wins / n) if n else float("nan"),
        "traded_symbols": traded,
        "zero_trade_rate": float(1.0 - traded / len(df)) if len(df) else 1.0,
        "mean_edge": float(df["edge"].mean()),
        "mean_ret_net": float(df["mean_ret_net"].mean()),
        "mean_compound": float(df["compound"].mean()),
        "worst_drawdown": float(df["max_drawdown"].min()),
        "method_mix": df["method"].value_counts().to_dict(),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", default="2026-04-01")
    p.add_argument("--end-date", default="2026-08-07")
    p.add_argument("--train-cutoff", default="2026-04-01")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260807_early_up_entry_ab",
    )
    p.add_argument("--code", action="append", default=[])
    p.add_argument("--max-codes", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pm = _load("pm_early", _ROOT / "decision_pack/scripts/plot_regime_transition_example.py")
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

    rows_by: dict[str, list[dict[str, Any]]] = {v: [] for v in VARIANTS}
    focus: list[dict[str, Any]] = []

    for mode in VARIANTS:
        ev = _load(
            f"ev_early_{mode}",
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
            rows_by[mode].append(
                {
                    "code": code,
                    "variant": mode,
                    "method": method,
                    "up_frac": float((labs[audit.to_numpy()] == "up").mean()),
                    **m,
                }
            )
            if code == "SH517120":
                # first up on/after 6/15; up share 6/15-7/13 (missed rally window)
                after = px["as_of"] >= "2026-06-15"
                idx = np.where(after.to_numpy() & (labs == "up"))[0]
                first = str(px.as_of.iloc[idx[0]].date()) if len(idx) else None
                win = (px["as_of"] >= "2026-06-15") & (px["as_of"] <= "2026-07-13")
                Lw = labs[win.to_numpy()]
                focus.append(
                    {
                        "variant": mode,
                        "method": method,
                        "first_up_on_or_after_0615": first,
                        "up_frac_0615_0713": float((Lw == "up").mean()) if len(Lw) else 0.0,
                        "n_up_0615_0713": int((Lw == "up").sum()),
                        "oos_n_closed": m["n_closed"],
                        "oos_win": m["win"],
                        "oos_edge": m["edge"],
                    }
                )
            print(
                f"[{mode}] {i}/{len(codes)} {code} -> {method} "
                f"n={m['n_closed']} win={m['win']}"
            )

    # wide table
    wide = pd.DataFrame(rows_by["baseline"]).set_index("code").add_prefix("baseline_")
    wide = wide.reset_index()
    for mode in VARIANTS[1:]:
        df = pd.DataFrame(rows_by[mode]).set_index("code").add_prefix(f"{mode}_")
        wide = wide.merge(df.reset_index(), on="code", how="outer")
    wide.to_csv(out_dir / "symbol_comparison.csv", index=False, encoding="utf-8-sig")

    pool_rows = []
    for mode in VARIANTS:
        s = _pool(pd.DataFrame(rows_by[mode]))
        s["variant"] = mode
        pool_rows.append(s)
    pd.DataFrame(pool_rows).to_csv(
        out_dir / "pool_comparison.csv", index=False, encoding="utf-8-sig"
    )
    focus_df = pd.DataFrame(focus)
    focus_df.to_csv(out_dir / "SH517120_focus.csv", index=False, encoding="utf-8-sig")

    base = pool_rows[0]
    decisions = {}
    for s in pool_rows[1:]:
        mode = s["variant"]
        fr = next(r for r in focus if r["variant"] == mode)
        gates = {
            "win_improved_or_equal": (
                s["trade_win"] >= base["trade_win"]
                if np.isfinite(s["trade_win"]) and np.isfinite(base["trade_win"])
                else False
            ),
            "enough_trades": s["n_closed"] >= max(10, int(0.8 * base["n_closed"])),
            "mean_net_noninferior": s["mean_ret_net"] >= base["mean_ret_net"] - 1e-6,
            "edge_noninferior": s["mean_edge"] >= base["mean_edge"] - 1e-6,
            "drawdown_controlled": s["worst_drawdown"]
            >= base["worst_drawdown"] - 0.02,
            "sh517120_earlier_up": bool(
                fr["first_up_on_or_after_0615"] is not None
                and fr["first_up_on_or_after_0615"] <= "2026-06-30"
            ),
        }
        decisions[mode] = {
            "gates": gates,
            "promote_candidate": bool(all(gates.values())),
            "pool": {k: v for k, v in s.items() if k != "method_mix"},
            "sh517120": fr,
        }

    # Prefer promote if any; else best on (sh517120 earlier, then edge, then win)
    promote = [m for m, d in decisions.items() if d["promote_candidate"]]
    if promote:
        # pick highest edge among promoters
        rec = max(promote, key=lambda m: decisions[m]["pool"]["mean_edge"])
    else:
        scored = []
        for m, d in decisions.items():
            fr = d["sh517120"]
            scored.append(
                (
                    int(d["gates"]["sh517120_earlier_up"]),
                    d["pool"]["mean_edge"],
                    d["pool"]["trade_win"] if np.isfinite(d["pool"]["trade_win"]) else -1,
                    m,
                )
            )
        scored.sort(reverse=True)
        # only recommend non-baseline if earlier-up achieved AND edge not much worse
        top = scored[0][3]
        if (
            decisions[top]["gates"]["sh517120_earlier_up"]
            and decisions[top]["pool"]["mean_edge"] >= base["mean_edge"] - 0.005
            and decisions[top]["pool"]["trade_win"]
            >= (base["trade_win"] - 0.05 if np.isfinite(base["trade_win"]) else 0)
        ):
            rec = top
        else:
            rec = "keep_baseline"

    summary = {
        "selection": f"B0 reselect on train < {args.train_cutoff}",
        "audit": f"{args.start_date}..{args.end_date}",
        "audit_used_for_selection": False,
        "n_symbols": len(codes),
        "baseline": {k: v for k, v in base.items() if k != "method_mix"},
        "variants": decisions,
        "recommendation": rec,
    }
    (out_dir / "acceptance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    lines = [
        "# Early up-entry A/B (audit descriptive)",
        "",
        f"- Train/select: `< {args.train_cutoff}`",
        f"- Audit: `{args.start_date}` → `{args.end_date}` (not used to pick)",
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
        "## SH517120 focus (want up starting ~2026-06-15)",
        "",
        focus_df.to_string(index=False),
        "",
        f"## Recommendation: `{rec}`",
        "",
        "```json",
        json.dumps({k: v["gates"] for k, v in decisions.items()}, ensure_ascii=False, indent=2),
        "```",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("DONE recommend=", rec)
    for s in pool_rows:
        print(
            f"  {s['variant']}: win={s['trade_win']:.1%} n={s['n_closed']} "
            f"edge={s['mean_edge']:+.2%}"
        )
    print(focus_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
