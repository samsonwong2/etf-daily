#!/usr/bin/env python3
"""A/B: ED → hold_up execution overlay (no regime mutation).

Arms
----
off
    Production labels + plain enter-up / leave-up.
full_k1
    ``block_buy_and_hypo_exit`` on same-day ED alert only.
full_k5
    Same actions, but effective alert = any alert in last 5 bars.

Selection: freeze configs from ``--adaptive-dir`` (B0 production snapshot).
Audit: descriptive OOS window only. Regime labels untouched.
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

from decision_pack.src.adaptive_stage_common import prepare_features  # noqa: E402
from decision_pack.src.early_down_alert import early_down_flags_frame  # noqa: E402
from decision_pack.src.ed_exec_overlay import (  # noqa: E402
    ED_OVERLAY_MODES,
    alert_mask_from_levels,
    simulate_hold_up_with_ed_overlay,
)
from decision_pack.src.symbol_trend_board import load_cluster_mapping_codes  # noqa: E402
from runtime_paths import CLUSTER_MAPPING_SELECTED_TXT, PLOTLY_OUTPUTS_DIR  # noqa: E402

FOCUS = "SH515060"
MISS_START, MISS_END = "2026-05-13", "2026-06-05"
# (mode, lookback_k, arm_name)
ARMS = (
    ("off", 1, "off"),
    ("block_buy_and_hypo_exit", 1, "full_k1"),
    ("block_buy_and_hypo_exit", 5, "full_k5"),
)
VARIANTS = tuple(a[2] for a in ARMS)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pool(rows: pd.DataFrame, prefix: str) -> dict[str, Any]:
    n = int(rows[f"{prefix}_n_closed"].sum())
    wins = int(rows[f"{prefix}_wins"].sum())
    return {
        "variant": prefix,
        "n_closed": n,
        "wins": wins,
        "trade_win": float(wins / n) if n else float("nan"),
        "traded_symbols": int((rows[f"{prefix}_n_closed"] > 0).sum()),
        "mean_edge": float(pd.to_numeric(rows[f"{prefix}_edge"], errors="coerce").mean()),
        "mean_ret_net": float(
            pd.to_numeric(rows[f"{prefix}_mean_ret_net"], errors="coerce").mean()
        ),
        "worst_drawdown": float(
            pd.to_numeric(rows[f"{prefix}_max_drawdown"], errors="coerce").min()
        ),
        "mean_blocked_entries": float(
            pd.to_numeric(rows[f"{prefix}_n_blocked_entries"], errors="coerce").mean()
        ),
        "mean_forced_exits": float(
            pd.to_numeric(rows[f"{prefix}_n_ed_forced_exits"], errors="coerce").mean()
        ),
        "sum_blocked_entries": int(
            pd.to_numeric(rows[f"{prefix}_n_blocked_entries"], errors="coerce").fillna(0).sum()
        ),
        "sum_forced_exits": int(
            pd.to_numeric(rows[f"{prefix}_n_ed_forced_exits"], errors="coerce").fillna(0).sum()
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start-date", default="2026-04-01")
    p.add_argument("--end-date", default="2026-08-10")
    p.add_argument(
        "--adaptive-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260810all_adaptive",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260810_ed_exec_overlay_ab",
    )
    p.add_argument("--code", action="append", default=[])
    p.add_argument("--max-codes", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    adaptive_dir = (
        args.adaptive_dir
        if args.adaptive_dir.is_absolute()
        else _ROOT / args.adaptive_dir
    )
    cfg_dir = adaptive_dir / "configs"
    if not cfg_dir.is_dir():
        print(f"[ERROR] missing configs: {cfg_dir}", file=sys.stderr)
        return 2

    out_dir = args.out_dir if args.out_dir.is_absolute() else _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pm = _load(
        "pm_ed_exec",
        _ROOT / "decision_pack/scripts/plot_regime_transition_example.py",
    )
    ev = _load(
        "ev_ed_exec",
        _ROOT / "decision_pack/scripts/eval_causal_regime_switch_pool.py",
    )
    pm._init_qlib(None)

    codes = [c.upper() for c in args.code] if args.code else [
        str(c).upper() for c in load_cluster_mapping_codes(CLUSTER_MAPPING_SELECTED_TXT)
    ]
    if args.max_codes:
        codes = codes[: args.max_codes]

    print(
        f"codes={len(codes)} adaptive={adaptive_dir.name} "
        f"audit={args.start_date}→{args.end_date} out={out_dir}"
    )

    rows: list[dict[str, Any]] = []
    focus_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for i, code in enumerate(codes, 1):
        cfg_path = cfg_dir / f"{code}.json"
        try:
            if not cfg_path.exists():
                raise FileNotFoundError(cfg_path)
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            method = str(cfg["method"])
            method_params = dict(cfg.get("method_params") or {})
            ohlcv = pm.load_qlib_ohlcv(
                code,
                "2015-01-01",
                args.end_date,
                init_qlib=False,
                lookback_calendar_days=220,
                clip_to_window=False,
            )
            if ohlcv is None or ohlcv.empty:
                raise ValueError("empty ohlcv")
            px = prepare_features(
                ev, ohlcv, method=method, method_params=method_params
            )
            px = px[
                (px["as_of"] >= pd.Timestamp(args.start_date))
                & (px["as_of"] <= pd.Timestamp(args.end_date))
            ].reset_index(drop=True)
            if px.empty:
                raise ValueError("empty audit window")
            labs = px["regime"].to_numpy(object)
            close = px["$close"].to_numpy(float)
            dates = px["as_of"].dt.strftime("%Y-%m-%d").to_numpy()
            ed_frame = early_down_flags_frame(px, regimes=labs)
            alert = alert_mask_from_levels(ed_frame["level"].to_numpy())

            rec: dict[str, Any] = {
                "code": code,
                "method": method,
                "n_bars": len(px),
                "n_alert_days": int(alert.sum()),
            }
            for mode, lookback_k, arm in ARMS:
                m, trades, _eq = simulate_hold_up_with_ed_overlay(
                    labs,
                    close,
                    alert,
                    mode=mode,
                    lookback_k=lookback_k,
                    dates=dates,
                )
                p = arm
                rec[f"{p}_n_closed"] = m["n_closed"]
                rec[f"{p}_wins"] = m["wins"]
                rec[f"{p}_win"] = m["win"]
                rec[f"{p}_mean_ret_net"] = m["mean_ret_net"]
                rec[f"{p}_edge"] = m["edge"]
                rec[f"{p}_compound"] = m["compound"]
                rec[f"{p}_max_drawdown"] = m["max_drawdown"]
                rec[f"{p}_n_blocked_entries"] = m["n_blocked_entries"]
                rec[f"{p}_n_ed_forced_exits"] = m["n_ed_forced_exits"]
                rec[f"{p}_n_effective_alert_days"] = m["n_effective_alert_days"]
            rows.append(rec)

            if code == FOCUS:
                miss = (px["as_of"] >= MISS_START) & (px["as_of"] <= MISS_END)
                miss_alert = int(alert[miss.to_numpy()].sum())
                up_days = int(((labs == "up") & miss.to_numpy()).sum())
                for mode, lookback_k, arm in ARMS:
                    m, trades, _ = simulate_hold_up_with_ed_overlay(
                        labs,
                        close,
                        alert,
                        mode=mode,
                        lookback_k=lookback_k,
                        dates=dates,
                    )
                    focus_rows.append(
                        {
                            "variant": arm,
                            "code": code,
                            "method": method,
                            "lookback_k": lookback_k,
                            "miss_alert_days": miss_alert,
                            "miss_up_days": up_days,
                            "n_blocked_entries": m["n_blocked_entries"],
                            "n_ed_forced_exits": m["n_ed_forced_exits"],
                            "n_effective_alert_days": m["n_effective_alert_days"],
                            "edge": m["edge"],
                            "compound": m["compound"],
                            "max_drawdown": m["max_drawdown"],
                            "n_closed": m["n_closed"],
                            "win": m["win"],
                        }
                    )
            print(
                f"[{i}/{len(codes)}] {code} method={method} "
                f"alerts={int(alert.sum())} "
                f"edge_off={rec['off_edge']:+.2%} "
                f"edge_k1={rec['full_k1_edge']:+.2%} "
                f"edge_k5={rec['full_k5_edge']:+.2%}"
            )
        except Exception as exc:  # noqa: BLE001
            errors.append({"code": code, "error": str(exc)})
            print(f"[{i}/{len(codes)}] FAIL {code}: {exc}")

    sym = pd.DataFrame(rows)
    sym.to_csv(out_dir / "symbol_comparison.csv", index=False, encoding="utf-8-sig")
    focus_df = pd.DataFrame(focus_rows)
    focus_df.to_csv(out_dir / f"{FOCUS}_focus.csv", index=False, encoding="utf-8-sig")
    if errors:
        pd.DataFrame(errors).to_csv(
            out_dir / "errors.csv", index=False, encoding="utf-8-sig"
        )

    pool_rows = [_pool(sym, v) for v in VARIANTS]
    pd.DataFrame(pool_rows).to_csv(
        out_dir / "pool_comparison.csv", index=False, encoding="utf-8-sig"
    )

    b = pool_rows[0]
    decisions: dict[str, Any] = {}
    for s in pool_rows[1:]:
        mode = s["variant"]
        fr = next((r for r in focus_rows if r["variant"] == mode), {})
        acted = int(s.get("sum_blocked_entries") or 0) + int(
            s.get("sum_forced_exits") or 0
        )
        edge_delta = float(s["mean_edge"]) - float(b["mean_edge"])
        gates = {
            "win_improved_or_equal": (
                s["trade_win"] >= b["trade_win"] - 1e-9
                if np.isfinite(s["trade_win"]) and np.isfinite(b["trade_win"])
                else False
            ),
            "enough_trades": s["n_closed"] >= max(5, int(0.5 * max(b["n_closed"], 1))),
            "mean_net_noninferior": s["mean_ret_net"] >= b["mean_ret_net"] - 1e-6,
            "edge_noninferior": s["mean_edge"] >= b["mean_edge"] - 1e-6,
            "drawdown_controlled": s["worst_drawdown"] >= b["worst_drawdown"] - 0.02,
            "overlay_acted_pool": acted >= 1,
            "not_noop": acted >= 1 or abs(edge_delta) > 1e-12,
            "focus_has_ed_signal": int(fr.get("miss_alert_days") or 0) >= 1,
            "focus_overlay_acted": (
                int(fr.get("n_blocked_entries") or 0)
                + int(fr.get("n_ed_forced_exits") or 0)
                >= 1
            ),
        }
        pool_ok = all(
            gates[k]
            for k in (
                "win_improved_or_equal",
                "enough_trades",
                "mean_net_noninferior",
                "edge_noninferior",
                "drawdown_controlled",
                "overlay_acted_pool",
                "not_noop",
            )
        )
        # Focus: must see ED in miss window; overlay action on focus is
        # nice-to-have (miss window often has zero up-entries under hold_up).
        focus_ok = gates["focus_has_ed_signal"]
        decisions[mode] = {
            "gates": gates,
            "promote_candidate": bool(pool_ok and focus_ok),
            "pool": s,
            "focus": fr,
        }

    recommendation = "keep_baseline"
    # Prefer last-K full overlay when it passes; else same-day full.
    for mode in ("full_k5", "full_k1"):
        if decisions.get(mode, {}).get("promote_candidate"):
            recommendation = mode
            break

    summary = {
        "selection": f"frozen configs from {adaptive_dir}",
        "audit": f"{args.start_date}..{args.end_date}",
        "scope": "ED exec overlay only; regime labels untouched",
        "modes": list(ED_OVERLAY_MODES),
        "case": (
            f"{FOCUS}: miss-window {MISS_START}..{MISS_END} should show ED alerts; "
            "overlay must block buy / raise exit without changing labels"
        ),
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
                        "sum_blocked_entries",
                        "sum_forced_exits",
                    )
                },
                "focus": v["focus"],
            }
            for k, v in decisions.items()
        },
        "recommendation": recommendation,
        "n_errors": len(errors),
    }
    (out_dir / "acceptance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    readme = [
        "# ED exec overlay A/B",
        "",
        f"- audit: `{args.start_date}` → `{args.end_date}`",
        f"- adaptive: `{adaptive_dir}`",
        f"- recommendation: **`{recommendation}`**",
        "",
        "## Pool",
        "",
        "| variant | win | n | edge | net | DD | blocked | forced |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in pool_rows:
        readme.append(
            f"| `{s['variant']}` | {s['trade_win']:.1%} | {s['n_closed']} | "
            f"{s['mean_edge']:+.2%} | {s['mean_ret_net']:+.2%} | {s['worst_drawdown']:.2%} | "
            f"{s['sum_blocked_entries']} | {s['sum_forced_exits']} |"
        )
    readme.extend(
        [
            "",
            "## Decision",
            "",
            f"- promote: `{recommendation}`",
            "- production default stays off until promote ≠ keep_baseline",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"[OK] recommendation={recommendation} out={out_dir}")
    for mode, d in decisions.items():
        print(f"  {mode}: promote={d['promote_candidate']} gates={d['gates']}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
