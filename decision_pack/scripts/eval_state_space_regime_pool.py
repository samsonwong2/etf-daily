#!/usr/bin/env python3
"""Full-pool, leakage-safe state-space regime tournament.

All selection uses data strictly before ``--train-cutoff``.  The requested
2026-04-01..2026-08-07 window is emitted as a retrospective audit only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
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
    StateSpaceRegimeConfig,
    state_space_labels_or_fallback,
)
from decision_pack.src.state_space_selection import (  # noqa: E402
    aggregate_candidate_metrics,
    candidate_grid,
    choose_candidate,
    evaluate_candidate_folds,
    expanding_splits,
    fit_final_candidate,
    grouped_block_bootstrap_delta,
    hold_up_trade_metrics,
)
from decision_pack.src.symbol_trend_board import load_cluster_mapping_codes  # noqa: E402
from runtime_paths import CLUSTER_MAPPING_SELECTED_TXT, PLOTLY_OUTPUTS_DIR  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_labels(ev, px: pd.DataFrame, method: str, params: dict[str, Any]) -> tuple[np.ndarray, bool]:
    if method not in ("kalman_local_trend", "hmm_kalman_gate"):
        return np.asarray(ev.run_method(method, px), dtype=object), False
    frozen = dict(params.get("frozen_model") or {})
    fallback_name = str(params.get("fallback_method") or "ma_stack_hyst")
    fallback = np.asarray(ev.run_method(fallback_name, px), dtype=object)
    labels, diag = state_space_labels_or_fallback(
        px, frozen, fallback_labels=fallback
    )
    return labels, bool(diag.get("fallback", False))


def _safe_mean(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.mean()) if len(clean) else float("nan")


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _pool_summary(rows: pd.DataFrame, prefix: str) -> dict[str, Any]:
    n = int(rows[f"{prefix}_n_closed"].sum())
    wins = int(rows[f"{prefix}_wins"].sum())
    traded = int((rows[f"{prefix}_n_closed"] > 0).sum())
    return {
        "n_closed": n,
        "wins": wins,
        "trade_win": float(wins / n) if n else float("nan"),
        "symbol_equal_win": _safe_mean(
            rows.loc[rows[f"{prefix}_n_closed"] > 0, f"{prefix}_win"]
        ),
        "traded_symbols": traded,
        "zero_trade_rate": float(1.0 - traded / len(rows)) if len(rows) else 1.0,
        "mean_edge": _safe_mean(rows[f"{prefix}_edge"]),
        "mean_ret_net": _safe_mean(rows[f"{prefix}_mean_ret_net"]),
        "worst_drawdown": float(rows[f"{prefix}_max_drawdown"].min()),
        "mean_exposure": _safe_mean(rows[f"{prefix}_exposure"]),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", default="2026-04-01")
    p.add_argument("--end-date", default="2026-08-07")
    p.add_argument("--train-cutoff", default="2026-04-01")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260807_state_space_audit",
    )
    p.add_argument("--code", action="append", default=[])
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--min-closed", type=int, default=2)
    p.add_argument("--prior-strength", type=float, default=5.0)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--continue-on-error", action="store_true", default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "configs").mkdir(exist_ok=True)
    (out_dir / "trades").mkdir(exist_ok=True)
    ev = _load_module(
        "ev_state_space",
        _ROOT / "decision_pack/scripts/eval_causal_regime_switch_pool.py",
    )
    pm = _load_module(
        "pm_state_space",
        _ROOT / "decision_pack/scripts/plot_regime_transition_example.py",
    )
    pm._init_qlib(None)
    codes = [str(x).upper() for x in args.code] or list(
        load_cluster_mapping_codes(CLUSTER_MAPPING_SELECTED_TXT)
    )
    if args.max_codes:
        codes = codes[: args.max_codes]
    candidates = candidate_grid()
    candidate_by_key = {c["key"]: c for c in candidates}

    per_symbol: dict[str, dict[str, Any]] = {}
    fold_parts: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []
    for i, code in enumerate(codes, 1):
        try:
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
            if len(px_train) < 60:
                raise ValueError(f"too few training bars: {len(px_train)}")
            baseline, _, baseline_meta = select_regime_method_legacy(
                ev, px_train, trade_mode=TRADE_MODE_HOLD_UP
            )
            splits = expanding_splits(len(px_train))
            folds = evaluate_candidate_folds(
                ev, px_train, candidates, splits=splits
            )
            folds.insert(0, "code", code)
            fold_parts.append(folds)
            per_symbol[code] = {
                "px": px,
                "px_train": px_train,
                "baseline": baseline,
                "baseline_meta": baseline_meta,
                "folds": folds,
            }
            print(
                f"[{i}/{len(codes)}] {code} train={len(px_train)} "
                f"folds={len(splits)} baseline={baseline}"
            )
        except Exception as exc:
            errors.append({"code": code, "error": str(exc)})
            print(f"[{i}/{len(codes)}] FAIL {code}: {exc}")
            if not args.continue_on_error:
                traceback.print_exc()
                return 1

    if not fold_parts:
        raise RuntimeError("no symbols evaluated")
    fold_all = pd.concat(fold_parts, ignore_index=True)
    fold_all.to_csv(out_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")

    # Empirical prior from each symbol's current production baseline on development folds.
    baseline_dev: list[pd.DataFrame] = []
    for code, info in per_symbol.items():
        g = info["folds"]
        dev = g[g["fold"] < g["fold"].max()]
        baseline_dev.append(dev[dev["candidate"] == info["baseline"]])
    prior_df = pd.concat(baseline_dev, ignore_index=True) if baseline_dev else pd.DataFrame()
    prior_n = int(prior_df["n_closed"].sum()) if len(prior_df) else 0
    prior_w = int(prior_df["wins"].sum()) if len(prior_df) else 0
    prior_rate = float(prior_w / prior_n) if prior_n else 0.5

    comparisons: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    equity_rows: list[pd.DataFrame] = []
    selected_accept_rows: list[dict[str, Any]] = []
    baseline_accept_rows: list[dict[str, Any]] = []

    for code, info in per_symbol.items():
        folds = info["folds"]
        max_fold = int(folds["fold"].max())
        dev = folds[folds["fold"] < max_fold]
        if dev.empty:
            dev = folds
        agg = aggregate_candidate_metrics(
            dev,
            prior_rate=prior_rate,
            prior_strength=args.prior_strength,
        )
        winner_key, select_meta = choose_candidate(
            agg,
            baseline_candidate=info["baseline"],
            min_closed=args.min_closed,
        )
        candidate = candidate_by_key.get(
            winner_key,
            {
                "key": info["baseline"],
                "kind": "baseline",
                "method": info["baseline"],
                "config": {},
            },
        )
        method, params = fit_final_candidate(ev, info["px_train"], candidate)

        px = info["px"]
        labels_selected, fallback = _load_labels(ev, px, method, params)
        labels_baseline, _ = _load_labels(ev, px, info["baseline"], {})
        audit = (px["as_of"] >= args.start_date) & (px["as_of"] <= args.end_date)
        close = px.loc[audit, "$close"].to_numpy(float)
        dates = px.loc[audit, "as_of"].dt.strftime("%Y-%m-%d").to_numpy()
        sm, strades, seq = hold_up_trade_metrics(
            labels_selected[audit.to_numpy()], close, dates=dates
        )
        bm, btrades, beq = hold_up_trade_metrics(
            labels_baseline[audit.to_numpy()], close, dates=dates
        )
        for frame, variant in ((seq, "selected"), (beq, "baseline")):
            frame.insert(0, "variant", variant)
            frame.insert(0, "code", code)
            equity_rows.append(frame)
        pd.DataFrame(
            [{**t, "variant": "selected"} for t in strades]
            + [{**t, "variant": "baseline"} for t in btrades],
            columns=[
                "entry_date",
                "exit_date",
                "entry_px",
                "exit_px",
                "ret_net",
                "bars",
                "variant",
            ],
        ).to_csv(
            out_dir / "trades" / f"{code}_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )

        comparison = {
            "code": code,
            "baseline_method": info["baseline"],
            "selected_candidate": winner_key,
            "selected_method": method,
            "selected_fallback": fallback,
            "train_bars": len(info["px_train"]),
        }
        for prefix, metrics in (("selected", sm), ("baseline", bm)):
            comparison.update({f"{prefix}_{k}": v for k, v in metrics.items()})
        comparisons.append(comparison)
        selection_rows.append(
            {
                "code": code,
                "winner": winner_key,
                "method": method,
                "baseline": info["baseline"],
                "prior_rate": prior_rate,
                **select_meta,
            }
        )
        cfg = {
            "code": code,
            "regime_mode": "state_space_nested_cv",
            "method": method,
            "method_params": params,
            "selected_candidate": winner_key,
            "baseline_method": info["baseline"],
            "train_cutoff": args.train_cutoff,
            "selection_meta": select_meta,
            "strict_audit_only": {
                "start": args.start_date,
                "end": args.end_date,
                "not_used_for_selection": True,
            },
        }
        (out_dir / "configs" / f"{code}.json").write_text(
            json.dumps(_json_clean(cfg), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        acceptance = folds[folds["fold"] == max_fold]
        sr = acceptance[acceptance["candidate"] == winner_key]
        br = acceptance[acceptance["candidate"] == info["baseline"]]
        if len(sr):
            selected_accept_rows.append(
                {"block_id": f"{code}:{max_fold}", **sr.iloc[0].to_dict()}
            )
        if len(br):
            baseline_accept_rows.append(
                {"block_id": f"{code}:{max_fold}", **br.iloc[0].to_dict()}
            )

    comp = pd.DataFrame(comparisons)
    comp.to_csv(out_dir / "symbol_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selection_rows).to_csv(
        out_dir / "selection_reasons.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(equity_rows, ignore_index=True).to_csv(
        out_dir / "daily_equity.csv", index=False, encoding="utf-8-sig"
    )

    selected_summary = _pool_summary(comp, "selected")
    baseline_summary = _pool_summary(comp, "baseline")
    acc_s = pd.DataFrame(selected_accept_rows)
    acc_b = pd.DataFrame(baseline_accept_rows)
    boot: dict[str, Any] = {}
    for metric in ("win", "mean_ret_net", "edge", "max_drawdown"):
        if len(acc_s) and len(acc_b):
            aa = acc_s.copy()
            bb = acc_b.copy()
            aa[metric] = pd.to_numeric(aa[metric], errors="coerce").fillna(0.0)
            bb[metric] = pd.to_numeric(bb[metric], errors="coerce").fillna(0.0)
            boot[metric] = grouped_block_bootstrap_delta(
                aa,
                bb,
                metric=metric,
                n_boot=args.bootstrap,
            )
    gates = {
        "win_improved": (
            selected_summary["trade_win"] > baseline_summary["trade_win"]
            if np.isfinite(selected_summary["trade_win"])
            and np.isfinite(baseline_summary["trade_win"])
            else False
        ),
        "enough_trades": selected_summary["n_closed"]
        >= max(10, int(0.8 * baseline_summary["n_closed"])),
        "mean_net_noninferior": selected_summary["mean_ret_net"]
        >= baseline_summary["mean_ret_net"],
        "edge_noninferior": selected_summary["mean_edge"]
        >= baseline_summary["mean_edge"],
        "drawdown_controlled": selected_summary["worst_drawdown"]
        >= baseline_summary["worst_drawdown"] - 0.02,
    }
    pool = {
        "selection_period": f"< {args.train_cutoff}",
        "audit_period": f"{args.start_date}..{args.end_date}",
        "audit_used_for_selection": False,
        "empirical_prior": {
            "rate": prior_rate,
            "strength": args.prior_strength,
            "wins": prior_w,
            "trades": prior_n,
        },
        "selected": selected_summary,
        "baseline": baseline_summary,
        "acceptance_fold_bootstrap": boot,
        "gates": gates,
        "promote": bool(all(gates.values())),
        "n_symbols": len(comp),
        "n_errors": len(errors),
    }
    pd.DataFrame(
        [
            {"variant": "selected", **selected_summary},
            {"variant": "baseline", **baseline_summary},
        ]
    ).to_csv(out_dir / "pool_comparison.csv", index=False, encoding="utf-8-sig")
    (out_dir / "acceptance.json").write_text(
        json.dumps(
            _json_clean(pool),
            ensure_ascii=False,
            indent=2,
            default=str,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    canary_start = (
        pd.Timestamp(args.end_date) + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")
    canary = {
        "start": canary_start,
        "status": "awaiting_future_data",
        "selection_frozen_at_train_cutoff": args.train_cutoff,
        "audit_end": args.end_date,
        "allow_reselection_from_canary": False,
        "symbols": sorted(comp["code"].astype(str).tolist()),
        "metrics_to_accumulate": [
            "closed_trades",
            "trade_win",
            "mean_ret_net",
            "compound",
            "edge",
            "max_drawdown",
            "fallback_rate",
        ],
    }
    (out_dir / "canary_baseline.json").write_text(
        json.dumps(canary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if errors:
        pd.DataFrame(errors).to_csv(
            out_dir / "errors.csv", index=False, encoding="utf-8-sig"
        )
    report = f"""# State-space regime audit

- Selection: data strictly before `{args.train_cutoff}`; expanding walk-forward folds.
- Audit: `{args.start_date}` → `{args.end_date}`; **not used for selection**.
- Symbols: {len(comp)}/{len(codes)}; errors: {len(errors)}.
- Empirical Beta prior: {prior_w}/{prior_n} = {prior_rate:.1%}, strength={args.prior_strength:g}.

## Audit metrics (descriptive, contaminated window)

- Selected trade win: {selected_summary['trade_win']:.1%} (n={selected_summary['n_closed']})
- Baseline trade win: {baseline_summary['trade_win']:.1%} (n={baseline_summary['n_closed']})
- Selected mean edge: {selected_summary['mean_edge']:+.2%}
- Baseline mean edge: {baseline_summary['mean_edge']:+.2%}
- Selected zero-trade rate: {selected_summary['zero_trade_rate']:.1%}
- Baseline zero-trade rate: {baseline_summary['zero_trade_rate']:.1%}

## Promotion gate

`promote={pool['promote']}`

```json
{json.dumps(gates, ensure_ascii=False, indent=2)}
```

The 2026-04-01..2026-08-07 values are audit-only.  A production claim requires
the frozen models to pass the post-2026-08-07 canary.
"""
    (out_dir / "README.md").write_text(report, encoding="utf-8")
    print(
        f"DONE symbols={len(comp)} errors={len(errors)} "
        f"selected_win={selected_summary['trade_win']:.1%} "
        f"baseline_win={baseline_summary['trade_win']:.1%} promote={pool['promote']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
