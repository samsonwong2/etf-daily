#!/usr/bin/env python3
"""Backtest MA labels gated by causal Kalman slope confirmation.

Selection uses only data before ``--train-cutoff``. The 2026-04-01..0807
window is reported as an audit set and never used to pick thresholds.
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
from decision_pack.src.state_space_regime import (  # noqa: E402
    apply_ma_kalman_confirm,
    fit_kalman_confirm_model,
)
from decision_pack.src.state_space_selection import (  # noqa: E402
    beta_binomial_rate,
    expanding_splits,
    hold_up_trade_metrics,
)
from decision_pack.src.symbol_trend_board import load_cluster_mapping_codes  # noqa: E402
from runtime_paths import CLUSTER_MAPPING_SELECTED_TXT, PLOTLY_OUTPUTS_DIR  # noqa: E402


CONFIRM_PROBS = (0.55, 0.60, 0.65, 0.70)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _variant_labels(
    ev,
    px_fit: pd.DataFrame,
    px_apply: pd.DataFrame,
    *,
    base_method: str,
    variant: str,
) -> np.ndarray:
    ma = np.asarray(ev.run_method(base_method, px_apply), dtype=object)
    if variant == "baseline":
        return ma
    # variant: confirm_pXX[_soft|_hard][_bull|_nobull]
    parts = variant.split("|")
    confirm = float(parts[0].replace("confirm_p", ""))
    soft = "soft" in parts
    bull = "bull" in parts
    model = fit_kalman_confirm_model(px_fit)
    labels, _ = apply_ma_kalman_confirm(
        ma,
        px_apply,
        model,
        confirm_prob=confirm,
        require_ma_bull=bull,
        soft_exit=soft,
    )
    return labels


def _variants() -> list[str]:
    out = ["baseline"]
    for p in CONFIRM_PROBS:
        for soft in ("soft", "hard"):
            for bull in ("bull", "nobull"):
                out.append(f"confirm_p{p:.2f}|{soft}|{bull}")
    return out


def _pool_stats(rows: pd.DataFrame, prefix: str) -> dict[str, Any]:
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
        "worst_drawdown": float(
            pd.to_numeric(rows[f"{prefix}_max_drawdown"], errors="coerce").min()
        ),
        "mean_exposure": float(
            pd.to_numeric(rows[f"{prefix}_exposure"], errors="coerce").mean()
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
        default=PLOTLY_OUTPUTS_DIR / "20260807_ma_kalman_confirm",
    )
    p.add_argument("--code", action="append", default=[])
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--prior-strength", type=float, default=5.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ev = _load_module(
        "ev_ma_kalman",
        _ROOT / "decision_pack/scripts/eval_causal_regime_switch_pool.py",
    )
    pm = _load_module(
        "pm_ma_kalman",
        _ROOT / "decision_pack/scripts/plot_regime_transition_example.py",
    )
    pm._init_qlib(None)
    codes = [str(x).upper() for x in args.code] or list(
        load_cluster_mapping_codes(CLUSTER_MAPPING_SELECTED_TXT)
    )
    if args.max_codes:
        codes = codes[: args.max_codes]
    variants = _variants()

    fold_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    pick_rows: list[dict[str, Any]] = []

    for i, code in enumerate(codes, 1):
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
        baseline, _, _ = select_regime_method_legacy(
            ev, px_train, trade_mode=TRADE_MODE_HOLD_UP
        )
        splits = expanding_splits(len(px_train))
        # Per-variant OOF on development folds (all but last) for selection.
        oof: dict[str, dict[str, float]] = {
            v: {"wins": 0, "n": 0, "ret": 0.0, "edge": 0.0, "dd": 0.0, "w": 0.0}
            for v in variants
        }
        for fold, (tr, va) in enumerate(splits):
            px_fit = px_train.iloc[tr].reset_index(drop=True)
            apply_end = int(va[-1]) + 1
            px_apply = px_train.iloc[:apply_end].reset_index(drop=True)
            close = px_apply["$close"].to_numpy(float)[va]
            dates = px_apply["as_of"].dt.strftime("%Y-%m-%d").to_numpy()[va]
            for variant in variants:
                labs = _variant_labels(
                    ev,
                    px_fit,
                    px_apply,
                    base_method=baseline,
                    variant=variant,
                )
                metrics, _, _ = hold_up_trade_metrics(labs[va], close, dates=dates)
                fold_rows.append(
                    {
                        "code": code,
                        "fold": fold,
                        "baseline_method": baseline,
                        "variant": variant,
                        **metrics,
                    }
                )
                if fold < len(splits) - 1 or len(splits) == 1:
                    bucket = oof[variant]
                    bucket["wins"] += metrics["wins"]
                    bucket["n"] += metrics["n_closed"]
                    w = max(metrics["n_closed"], 1)
                    bucket["ret"] += metrics["mean_ret_net"] * w
                    bucket["edge"] += metrics["edge"]
                    bucket["dd"] += metrics["max_drawdown"]
                    bucket["w"] += w

        # Shrink win on development OOF; pick best gated variant vs baseline.
        prior_n = sum(oof[v]["n"] for v in variants if v == "baseline")
        prior_w = sum(oof[v]["wins"] for v in variants if v == "baseline")
        prior_rate = float(prior_w / prior_n) if prior_n else 0.5
        ranked = []
        for variant, b in oof.items():
            n_folds_used = max(len(splits) - 1, 1)
            ranked.append(
                {
                    "variant": variant,
                    "n_closed": int(b["n"]),
                    "wins": int(b["wins"]),
                    "posterior_win": beta_binomial_rate(
                        int(b["wins"]),
                        int(b["n"]),
                        prior_rate=prior_rate,
                        prior_strength=args.prior_strength,
                    ),
                    "mean_ret_net": float(b["ret"] / max(b["w"], 1.0)),
                    "mean_edge": float(b["edge"] / n_folds_used),
                    "mean_dd": float(b["dd"] / n_folds_used),
                }
            )
        rdf = pd.DataFrame(ranked)
        base_row = rdf[rdf.variant == "baseline"].iloc[0]
        gated = rdf[rdf.variant != "baseline"].copy()
        # Win-first with non-inferior net/edge and not much worse DD.
        eligible = gated[
            (gated.n_closed >= max(1, int(base_row.n_closed)))
            & (gated.mean_ret_net >= float(base_row.mean_ret_net) - 1e-12)
            & (gated.mean_edge >= float(base_row.mean_edge) - 1e-12)
            & (gated.mean_dd >= float(base_row.mean_dd) - 0.02)
            & (gated.posterior_win > float(base_row.posterior_win))
        ].sort_values(
            ["posterior_win", "mean_ret_net", "mean_edge"],
            ascending=[False, False, False],
        )
        if len(eligible):
            chosen = str(eligible.iloc[0].variant)
            reason = "gated_beats_baseline_oof"
        else:
            chosen = "baseline"
            reason = "keep_baseline"

        # Fit Kalman on full train; evaluate audit window for baseline vs chosen vs best fixed confirm.
        model = fit_kalman_confirm_model(px_train)
        audit = (px["as_of"] >= args.start_date) & (px["as_of"] <= args.end_date)
        close_a = px.loc[audit, "$close"].to_numpy(float)
        dates_a = px.loc[audit, "as_of"].dt.strftime("%Y-%m-%d").to_numpy()
        row: dict[str, Any] = {
            "code": code,
            "baseline_method": baseline,
            "chosen_variant": chosen,
            "reason": reason,
            "train_bars": len(px_train),
        }
        for tag, variant in (
            ("baseline", "baseline"),
            ("chosen", chosen),
            # fixed reference: most conservative soft+bull gate
            ("confirm060", "confirm_p0.60|soft|bull"),
            ("confirm065", "confirm_p0.65|soft|bull"),
        ):
            labs = _variant_labels(
                ev,
                px_train,
                px,
                base_method=baseline,
                variant=variant,
            )
            m, _, _ = hold_up_trade_metrics(
                labs[audit.to_numpy()], close_a, dates=dates_a
            )
            for k, v in m.items():
                row[f"{tag}_{k}"] = v
        audit_rows.append(row)
        pick_rows.append(
            {
                "code": code,
                "baseline_method": baseline,
                "chosen_variant": chosen,
                "reason": reason,
                "prior_rate": prior_rate,
                **{f"oof_{k}": v for k, v in rdf.set_index("variant").loc[chosen].items() if k != "variant"},
            }
        )
        print(
            f"[{i}/{len(codes)}] {code} base={baseline} chosen={chosen} "
            f"audit_base_win={row['baseline_win']} chosen_win={row['chosen_win']}"
        )

    fold_df = pd.DataFrame(fold_rows)
    audit_df = pd.DataFrame(audit_rows)
    pick_df = pd.DataFrame(pick_rows)
    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    audit_df.to_csv(out_dir / "symbol_comparison.csv", index=False, encoding="utf-8-sig")
    pick_df.to_csv(out_dir / "selection_reasons.csv", index=False, encoding="utf-8-sig")

    summaries = []
    for prefix in ("baseline", "chosen", "confirm060", "confirm065"):
        s = _pool_stats(audit_df, prefix)
        summaries.append({"variant": prefix, **s})
    pool = pd.DataFrame(summaries)
    pool.to_csv(out_dir / "pool_comparison.csv", index=False, encoding="utf-8-sig")

    base = summaries[0]
    chosen = summaries[1]
    gates = {
        "win_improved": (
            chosen["trade_win"] > base["trade_win"]
            if np.isfinite(chosen["trade_win"]) and np.isfinite(base["trade_win"])
            else False
        ),
        "enough_trades": chosen["n_closed"] >= max(10, int(0.8 * base["n_closed"])),
        "mean_net_noninferior": chosen["mean_ret_net"] >= base["mean_ret_net"],
        "edge_noninferior": chosen["mean_edge"] >= base["mean_edge"],
        "drawdown_controlled": chosen["worst_drawdown"]
        >= base["worst_drawdown"] - 0.02,
    }
    n_switched = int((pick_df.chosen_variant != "baseline").sum())
    acceptance = {
        "selection_period": f"< {args.train_cutoff}",
        "audit_period": f"{args.start_date}..{args.end_date}",
        "audit_used_for_selection": False,
        "n_symbols": len(audit_df),
        "n_switched_to_gate": n_switched,
        "gates": gates,
        "promote": bool(all(gates.values())),
        "pool": summaries,
    }
    (out_dir / "acceptance.json").write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # Also summarize fixed gates vs baseline on audit (descriptive).
    readme = f"""# MA + Kalman confirm backtest

- Base method: per-symbol `legacy_score_method` (production B0).
- Gate: keep MA `up` only if filtered Kalman `P(slope>0)` ≥ threshold
  (optional `MA20>MA60`, optional soft exit when `P<0.5`).
- Selection: train OOF folds before `{args.train_cutoff}` only.
- Audit: `{args.start_date}` → `{args.end_date}` (contaminated; not used to pick).

## Pool audit (descriptive)

| Variant | Trade win | n | Mean edge | Mean net | Zero-trade |
|---------|----------:|--:|----------:|---------:|-----------:|
| baseline | {base['trade_win']:.1%} | {base['n_closed']} | {base['mean_edge']:+.2%} | {base['mean_ret_net']:+.2%} | {base['zero_trade_rate']:.1%} |
| chosen (OOF pick) | {chosen['trade_win']:.1%} | {chosen['n_closed']} | {chosen['mean_edge']:+.2%} | {chosen['mean_ret_net']:+.2%} | {chosen['zero_trade_rate']:.1%} |
| fixed confirm_p0.60 soft+bull | {summaries[2]['trade_win']:.1%} | {summaries[2]['n_closed']} | {summaries[2]['mean_edge']:+.2%} | {summaries[2]['mean_ret_net']:+.2%} | {summaries[2]['zero_trade_rate']:.1%} |
| fixed confirm_p0.65 soft+bull | {summaries[3]['trade_win']:.1%} | {summaries[3]['n_closed']} | {summaries[3]['mean_edge']:+.2%} | {summaries[3]['mean_ret_net']:+.2%} | {summaries[3]['zero_trade_rate']:.1%} |

`promote={acceptance['promote']}` · switched symbols={n_switched}/46

```json
{json.dumps(gates, ensure_ascii=False, indent=2)}
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(
        f"DONE n={len(audit_df)} switched={n_switched} "
        f"base_win={base['trade_win']:.1%} chosen_win={chosen['trade_win']:.1%} "
        f"c060_win={summaries[2]['trade_win']:.1%} promote={acceptance['promote']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
