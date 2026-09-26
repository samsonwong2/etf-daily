#!/usr/bin/env python3
"""Walk-forward dual-truth method variants B0–B5 for regime labeling.

Selection uses only training folds (no OOS peeking). OOS is reported once for
acceptance after selection.

Variants
--------
B0  current score_method argmax on full train
B1  dual-truth balanced score on full train (no up_frac reward)
B2  expanding-window 3-fold CV mean−std selection
B3  per-symbol threshold grid on CV (ADX / strict / chop / smooth / min_seg)
B4  probability ensemble of 7 methods (CV weights; low-conf → range)
B5  B2 + B3 + B4

Outputs under ~/etf-daily-output/temp/plotly_outputs/20260806_regime_method_variants/
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

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from decision_pack.src.adaptive_stage_common import (  # noqa: E402
    apply_sticky_up_guard,
    dual_truth_selection_score,
    ensemble_vote,
    hold_up_compound,
    prepare_features,
    score_method,
    worst_up_drawdown,
)
from decision_pack.src.symbol_trend_board import load_cluster_mapping_codes  # noqa: E402
from runtime_paths import CLUSTER_MAPPING_SELECTED_TXT, PLOTLY_OUTPUTS_DIR  # noqa: E402

WARMUP_CALENDAR_DAYS = 220
DEFAULT_OUT = (
    PLOTLY_OUTPUTS_DIR / "20260806_regime_method_variants"
)
AUDIT_CODES = ("SH588710", "SZ159502", "SH562510", "SZ159320")

ADX_GRID = (18.0, 22.0, 26.0)
STRICT_THR_GRID = (0.03, 0.05, 0.08)
SMOOTH_GRID = (3, 5, 7)
MIN_SEG_GRID = (5, 10, 15)


def _load_ev():
    path = _PROJECT_ROOT / "decision_pack/scripts/eval_causal_regime_switch_pool.py"
    spec = importlib.util.spec_from_file_location("ev_var", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_pm():
    path = _PROJECT_ROOT / "decision_pack/scripts/plot_regime_transition_example.py"
    spec = importlib.util.spec_from_file_location("pm_var", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def expanding_folds(
    n_train: int, n_folds: int = 3, min_train: int = 120, val_frac: float = 0.2
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_idx, val_idx) within [0, n_train)."""
    if n_train < min_train + 40:
        cut = max(min_train, int(n_train * 0.7))
        if cut >= n_train - 20:
            return [(np.arange(n_train), np.arange(n_train))]
        return [(np.arange(cut), np.arange(cut, n_train))]

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    val_len = max(40, int(n_train * val_frac / n_folds))
    for k in range(n_folds):
        val_end = n_train - (n_folds - 1 - k) * (val_len // 2)
        val_end = min(n_train, max(min_train + val_len, val_end))
        val_start = max(min_train, val_end - val_len)
        tr_end = val_start
        if tr_end < min_train:
            continue
        folds.append((np.arange(tr_end), np.arange(val_start, val_end)))
    if not folds:
        cut = int(n_train * 0.75)
        folds = [(np.arange(cut), np.arange(cut, n_train))]
    return folds


def _metrics_pair(ev, labs, close, truth_r, truth_c) -> dict[str, float]:
    mr = ev.eval_labels(truth_r, labs)
    mc = ev.eval_labels(truth_c, labs)
    hu, _ = hold_up_compound(labs, close)
    wdd = worst_up_drawdown(labs, close)
    return dict(
        retro_agree=mr["agree"],
        retro_macro_f1=mr["macro_f1"],
        retro_balanced_acc=mr["balanced_acc"],
        retro_switch_hit=mr["switch_hit"],
        retro_switch_false=mr["switch_false"],
        retro_switch_delay=mr["switch_delay_mean"],
        retro_trend_f1=mr["trend_f1"],
        retro_seg_overlap=mr["seg_overlap"],
        retro_frac_pred_down=mr["frac_pred_down"],
        retro_frac_truth_down=mr["frac_truth_down"],
        retro_frac_pred_range=mr["frac_pred_range"],
        retro_frac_truth_range=mr["frac_truth_range"],
        causal_macro_f1=mc["macro_f1"],
        causal_balanced_acc=mc["balanced_acc"],
        causal_switch_hit=mc["switch_hit"],
        causal_switch_false=mc["switch_false"],
        causal_agree=mc["agree"],
        hold_up=hu,
        worst_up_dd=wdd,
        dual_score=dual_truth_selection_score(
            labs, close, truth_r, truth_c, eval_fn=ev.eval_labels
        ),
    )


def _run_with_params(ev, name: str, px, params: dict | None = None) -> np.ndarray:
    params = dict(params or {})
    labs = ev.run_method(name, px, params=params)
    return apply_sticky_up_guard(labs, px)


def _cv_dual_score(
    *,
    ev,
    name: str,
    px_train: pd.DataFrame,
    close_train: np.ndarray,
    truth_r: np.ndarray,
    truth_c: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    params: dict | None = None,
) -> tuple[float, list[float]]:
    labs_full = _run_with_params(ev, name, px_train, params)
    fold_scores: list[float] = []
    for _tr_idx, va_idx in folds:
        if len(va_idx) < 15:
            continue
        s = dual_truth_selection_score(
            labs_full[va_idx],
            close_train[va_idx],
            truth_r[va_idx],
            truth_c[va_idx],
            eval_fn=ev.eval_labels,
        )
        fold_scores.append(float(s))
    if not fold_scores:
        return -1e9, []
    arr = np.asarray(fold_scores, dtype=float)
    return float(arr.mean() - 0.25 * arr.std(ddof=0)), fold_scores


def select_b0(ev, px_train, close_train, truth_r) -> tuple[str, dict, dict]:
    best_m, best_s = None, -1e18
    scores = {}
    for name in ev.METHODS:
        labs = ev.run_method(name, px_train)
        s = score_method(labs, close_train, truth_r)
        scores[name] = float(s)
        if s > best_s:
            best_s, best_m = s, name
    return best_m or "ma_stack_strict", {}, dict(scores=scores, score=best_s)


def select_b1(ev, px_train, close_train, truth_r, truth_c) -> tuple[str, dict, dict]:
    best_m, best_s = None, -1e18
    scores = {}
    for name in ev.METHODS:
        labs = apply_sticky_up_guard(ev.run_method(name, px_train), px_train)
        s = dual_truth_selection_score(
            labs, close_train, truth_r, truth_c, eval_fn=ev.eval_labels
        )
        scores[name] = float(s)
        if s > best_s:
            best_s, best_m = s, name
    return best_m or "ma_stack_strict", {}, dict(scores=scores, score=best_s)


def select_b2(ev, px_train, close_train, truth_r, truth_c, folds) -> tuple[str, dict, dict]:
    best_m, best_s = None, -1e18
    scores = {}
    fold_detail = {}
    for name in ev.METHODS:
        s, fs = _cv_dual_score(
            ev=ev,
            name=name,
            px_train=px_train,
            close_train=close_train,
            truth_r=truth_r,
            truth_c=truth_c,
            folds=folds,
        )
        scores[name] = s
        fold_detail[name] = fs
        if s > best_s:
            best_s, best_m = s, name
    return best_m or "ma_stack_strict", {}, dict(
        scores=scores, fold_scores=fold_detail, score=best_s
    )


def _param_grid_for(name: str) -> list[dict]:
    base: list[dict]
    if name == "adx_di":
        base = [{"adx_thr": t, "smooth": s} for t in ADX_GRID for s in SMOOTH_GRID]
    elif name == "ma_stack_strict":
        base = [
            {"p60_thr": t, "smooth": s} for t in STRICT_THR_GRID for s in SMOOTH_GRID
        ]
    elif name in ("ma_stack_hyst", "ma_stack_struct", "hybrid_ma_adx", "prior60_band"):
        base = [{"smooth": s} for s in SMOOTH_GRID]
    else:
        base = [{}]
    out: list[dict] = []
    for g in base:
        for ms in MIN_SEG_GRID:
            gg = dict(g)
            gg["min_seg"] = ms
            out.append(gg)
    return out[:18]


def select_b3(ev, px_train, close_train, truth_r, truth_c, folds) -> tuple[str, dict, dict]:
    best_m, best_p, best_s = None, {}, -1e18
    detail: dict[str, Any] = {}
    for name in ev.METHODS:
        local_best = -1e18
        local_p: dict = {}
        for p in _param_grid_for(name):
            s, _fs = _cv_dual_score(
                ev=ev,
                name=name,
                px_train=px_train,
                close_train=close_train,
                truth_r=truth_r,
                truth_c=truth_c,
                folds=folds,
                params=p,
            )
            if s > local_best:
                local_best = s
                local_p = dict(p)
            if s > best_s:
                best_s, best_m, best_p = s, name, dict(p)
        detail[name] = dict(best_score=local_best, best_params=local_p)
    return best_m or "ma_stack_strict", best_p, dict(detail=detail, score=best_s)


def _ensemble_from_weights(
    ev,
    px_train,
    close_train,
    truth_r,
    truth_c,
    folds,
    weights: dict[str, float],
    tuned: dict[str, dict] | None = None,
) -> tuple[str, dict, dict]:
    names = list(ev.METHODS.keys())
    tuned = tuned or {}
    mats = []
    for name in names:
        mats.append(_run_with_params(ev, name, px_train, tuned.get(name)))
    label_mat = np.stack(mats, axis=0)
    w = np.array([float(weights.get(n, 0.0)) for n in names], dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(names)) / len(names)
    else:
        w = w / w.sum()
    conf_grid = (0.35, 0.40, 0.45, 0.50, 0.55)
    best_conf, best_s = 0.45, -1e18
    _tr, va_idx = folds[-1]
    for conf in conf_grid:
        voted = ensemble_vote(label_mat[:, va_idx], weights=w, conf_thr=conf)
        s = dual_truth_selection_score(
            voted,
            close_train[va_idx],
            truth_r[va_idx],
            truth_c[va_idx],
            eval_fn=ev.eval_labels,
        )
        if s > best_s:
            best_s, best_conf = s, conf
    params = dict(
        weights={n: float(weights.get(n, 0.0)) for n in names},
        conf_thr=best_conf,
        member_methods=names,
    )
    if tuned:
        params["tuned_params"] = tuned
    return "ensemble", params, dict(score=best_s)


def select_b4(ev, px_train, close_train, truth_r, truth_c, folds) -> tuple[str, dict, dict]:
    raw: dict[str, float] = {}
    for name in ev.METHODS:
        s, _ = _cv_dual_score(
            ev=ev,
            name=name,
            px_train=px_train,
            close_train=close_train,
            truth_r=truth_r,
            truth_c=truth_c,
            folds=folds,
        )
        raw[name] = max(float(s), 0.0)
    tot = sum(raw.values()) or 1.0
    weights = {k: v / tot for k, v in raw.items()}
    method, params, meta = _ensemble_from_weights(
        ev, px_train, close_train, truth_r, truth_c, folds, weights
    )
    meta["cv_raw"] = raw
    return method, params, meta


def select_b5(ev, px_train, close_train, truth_r, truth_c, folds) -> tuple[str, dict, dict]:
    tuned: dict[str, dict] = {}
    raw: dict[str, float] = {}
    for name in ev.METHODS:
        best_s, best_p = -1e18, {}
        for p in _param_grid_for(name)[:8]:
            s, _ = _cv_dual_score(
                ev=ev,
                name=name,
                px_train=px_train,
                close_train=close_train,
                truth_r=truth_r,
                truth_c=truth_c,
                folds=folds,
                params=p,
            )
            if s > best_s:
                best_s, best_p = s, dict(p)
        tuned[name] = best_p
        raw[name] = max(float(best_s), 0.0)
    tot = sum(raw.values()) or 1.0
    weights = {k: v / tot for k, v in raw.items()}
    method, params, meta = _ensemble_from_weights(
        ev, px_train, close_train, truth_r, truth_c, folds, weights, tuned=tuned
    )
    meta["cv_raw"] = raw
    return method, params, meta


def predict_variant(ev, method: str, params: dict, px) -> np.ndarray:
    if method == "ensemble":
        names = params.get("member_methods") or list(ev.METHODS.keys())
        weights = params.get("weights") or {n: 1.0 / len(names) for n in names}
        conf = float(params.get("conf_thr", 0.45))
        tuned = params.get("tuned_params") or {}
        mats = [_run_with_params(ev, name, px, tuned.get(name)) for name in names]
        arr = np.stack(mats, axis=0)
        w = np.array([float(weights.get(n, 0.0)) for n in names], dtype=float)
        if w.sum() <= 0:
            w = np.ones(len(names)) / len(names)
        else:
            w = w / w.sum()
        labs = ensemble_vote(arr, weights=w, conf_thr=conf)
        return apply_sticky_up_guard(labs, px)
    return _run_with_params(ev, method, px, params)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oos-start", default="2026-04-01")
    p.add_argument("--end-date", default="2026-08-06")
    p.add_argument("--train-cutoff", default=None)
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--code", action="append", default=None)
    p.add_argument("--variants", default="B0,B1,B2,B3,B4,B5")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args(argv)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _build_report(
    sdf: pd.DataFrame,
    out_dir: Path,
    oos_start: str,
    end: str,
    train_cutoff: str,
    variants: list[str],
) -> str:
    oos = sdf[(sdf["window"] == "oos") & (sdf["variant"] != "ERR")].copy()
    if oos.empty:
        return "# Regime method variants\n\nNo OOS rows.\n"

    lines = [
        "# Regime method variants (dual-truth)",
        "",
        f"> OOS **{oos_start}→{end}** · train < {train_cutoff} · selection on train CV only",
        f"> out=`{out_dir}`",
        "",
        "## Pool means (OOS)",
        "",
        "| Variant | n | retro macro-F1 | retro bal-acc | causal macro-F1 | switch_hit | switch_false | pred_down | sticky≤-12% | dual_score |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for var in variants:
        g = oos[oos["variant"] == var]
        if g.empty:
            continue
        sticky = int((g["worst_up_dd"] <= -0.12).sum())
        lines.append(
            f"| {var} | {len(g)} | {g['retro_macro_f1'].mean():.3f} | "
            f"{g['retro_balanced_acc'].mean():.3f} | {g['causal_macro_f1'].mean():.3f} | "
            f"{g['retro_switch_hit'].mean():.3f} | {g['retro_switch_false'].mean():.3f} | "
            f"{g['retro_frac_pred_down'].mean():.1%} | {sticky} | {g['dual_score'].mean():.3f} |"
        )

    train = sdf[(sdf["window"] == "train") & (sdf["variant"] != "ERR")]
    winner = None
    if not train.empty:
        train_means = (
            train.groupby("variant")["dual_score"].mean().sort_values(ascending=False)
        )
        lines += ["", "## Selection (train dual_score ranking)", ""]
        for var, sc in train_means.items():
            lines.append(f"- **{var}**: train dual_score={sc:.3f}")
        winner = str(train_means.index[0])
        lines += ["", f"**Train-selected winner: {winner}**", ""]

        b0 = oos[oos["variant"] == "B0"]
        w = oos[oos["variant"] == winner]
        if not b0.empty and not w.empty:
            lines += [
                "## OOS acceptance snapshot (winner vs B0)",
                "",
                f"| Gate | B0 | {winner} |",
                "|:--|--:|--:|",
                f"| retro macro-F1 | {b0['retro_macro_f1'].mean():.3f} | {w['retro_macro_f1'].mean():.3f} |",
                f"| retro bal-acc | {b0['retro_balanced_acc'].mean():.3f} | {w['retro_balanced_acc'].mean():.3f} |",
                f"| causal macro-F1 | {b0['causal_macro_f1'].mean():.3f} | {w['causal_macro_f1'].mean():.3f} |",
                f"| switch_false | {b0['retro_switch_false'].mean():.3f} | {w['retro_switch_false'].mean():.3f} |",
                f"| pred_down | {b0['retro_frac_pred_down'].mean():.1%} | {w['retro_frac_pred_down'].mean():.1%} |",
                f"| truth_down | {b0['retro_frac_truth_down'].mean():.1%} | {w['retro_frac_truth_down'].mean():.1%} |",
                f"| sticky≤-12% | {int((b0['worst_up_dd']<=-0.12).sum())} | {int((w['worst_up_dd']<=-0.12).sum())} |",
                f"| trend_f1 | {b0['retro_trend_f1'].mean():.3f} | {w['retro_trend_f1'].mean():.3f} |",
            ]

    if winner is not None:
        w = oos[oos["variant"] == winner].sort_values("retro_macro_f1")
        lines += ["", f"## Worst 8 by OOS retro macro-F1 ({winner})", ""]
        for _, r in w.head(8).iterrows():
            lines.append(
                f"- {r['code']} method={r['method']} mf1={r['retro_macro_f1']:.3f} "
                f"agree={r['retro_agree']:.1%} causal_mf1={r['causal_macro_f1']:.3f} "
                f"dd={r['worst_up_dd']:.1%}"
            )

    lines += [
        "",
        "## Audit symbols",
        "",
        f"Daily labels under `audits/` for: {', '.join(AUDIT_CODES)}",
        "",
        "## Files",
        "",
        "- `variant_summary.csv`",
        "- `method_configs/{code}.json`",
        "- `regime_accuracy_report.md`",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    oos_start = pd.Timestamp(args.oos_start).strftime("%Y-%m-%d")
    end = pd.Timestamp(args.end_date).strftime("%Y-%m-%d")
    train_cutoff = pd.Timestamp(args.train_cutoff or oos_start).strftime("%Y-%m-%d")
    out_dir = args.out_dir if args.out_dir.is_absolute() else _PROJECT_ROOT / args.out_dir
    cfg_dir = out_dir / "method_configs"
    audit_dir = out_dir / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    variants = [v.strip().upper() for v in args.variants.split(",") if v.strip()]
    if args.code:
        codes = [c.upper() for c in args.code]
    else:
        codes = list(load_cluster_mapping_codes(CLUSTER_MAPPING_SELECTED_TXT))
    if args.max_codes:
        codes = codes[: args.max_codes]

    ev = _load_ev()
    pm = _load_pm()
    pm._init_qlib(None)

    summary_rows: list[dict] = []

    for i, code in enumerate(codes, 1):
        try:
            ohlcv = pm.load_qlib_ohlcv(
                code,
                "2015-01-01",
                end,
                init_qlib=False,
                lookback_calendar_days=WARMUP_CALENDAR_DAYS,
                clip_to_window=False,
            )
            px = prepare_features(ev, ohlcv, method=None)
            as_of = px["as_of"]
            close_full = px["$close"].to_numpy(float)
            train_m = (as_of < train_cutoff).to_numpy()
            oos_m = ((as_of >= oos_start) & (as_of <= end)).to_numpy()
            if train_m.sum() < 80 or oos_m.sum() < 20:
                raise ValueError(
                    f"short history train={train_m.sum()} oos={oos_m.sum()}"
                )

            truth_r_full = ev.retrospective_truth(close_full)
            truth_c_full = ev.causal_confirmed_swing_truth(close_full)

            px_train = px.loc[train_m].reset_index(drop=True)
            close_train = close_full[train_m]
            truth_r_tr = truth_r_full[train_m]
            truth_c_tr = truth_c_full[train_m]
            folds = expanding_folds(len(px_train), n_folds=3)

            selections: dict[str, tuple[str, dict, dict]] = {}
            if "B0" in variants:
                selections["B0"] = select_b0(ev, px_train, close_train, truth_r_tr)
            if "B1" in variants:
                selections["B1"] = select_b1(
                    ev, px_train, close_train, truth_r_tr, truth_c_tr
                )
            if "B2" in variants:
                selections["B2"] = select_b2(
                    ev, px_train, close_train, truth_r_tr, truth_c_tr, folds
                )
            if "B3" in variants:
                selections["B3"] = select_b3(
                    ev, px_train, close_train, truth_r_tr, truth_c_tr, folds
                )
            if "B4" in variants:
                selections["B4"] = select_b4(
                    ev, px_train, close_train, truth_r_tr, truth_c_tr, folds
                )
            if "B5" in variants:
                selections["B5"] = select_b5(
                    ev, px_train, close_train, truth_r_tr, truth_c_tr, folds
                )

            code_cfg: dict[str, Any] = dict(
                code=code,
                train_cutoff=train_cutoff,
                oos_start=oos_start,
                end_date=end,
                truth_version="retro_full+causal_confirmed_v1",
                folds=[
                    (
                        int(a[0][0]) if len(a[0]) else 0,
                        int(a[1][0]),
                        int(a[1][-1]),
                    )
                    for a in folds
                ],
                variants={},
            )

            for var, (method, params, meta) in selections.items():
                labs_full = predict_variant(ev, method, params, px)
                for tag, mask in (("train", train_m), ("oos", oos_m)):
                    labs = labs_full[mask]
                    close = close_full[mask]
                    met = _metrics_pair(
                        ev, labs, close, truth_r_full[mask], truth_c_full[mask]
                    )
                    summary_rows.append(
                        dict(
                            code=code,
                            variant=var,
                            method=method,
                            window=tag,
                            **met,
                            params_json=json.dumps(params, ensure_ascii=False),
                        )
                    )

                code_cfg["variants"][var] = dict(
                    method=method, params=params, meta=_jsonable(meta)
                )

                if code in AUDIT_CODES:
                    oos_idx = np.where(oos_m)[0]
                    audit = pd.DataFrame(
                        {
                            "as_of": as_of.iloc[oos_idx].astype(str).values,
                            "close": close_full[oos_idx],
                            "pred": labs_full[oos_idx],
                            "truth_retro": truth_r_full[oos_idx],
                            "truth_causal": truth_c_full[oos_idx],
                        }
                    )
                    audit.to_csv(
                        audit_dir / f"{code}_{var}_oos_labels.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )

            (cfg_dir / f"{code}.json").write_text(
                json.dumps(code_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            oos_b0 = next(
                (
                    r
                    for r in summary_rows
                    if r["code"] == code
                    and r["variant"] == "B0"
                    and r["window"] == "oos"
                ),
                None,
            )
            oos_best = max(
                (
                    r
                    for r in summary_rows
                    if r["code"] == code and r["window"] == "oos"
                ),
                key=lambda r: r.get("dual_score", -1e9),
                default=None,
            )
            print(
                f"[{i}/{len(codes)}] {code} "
                f"B0_mf1={oos_b0['retro_macro_f1'] if oos_b0 else float('nan'):.3f} "
                f"best={oos_best['variant'] if oos_best else '?'} "
                f"mf1={oos_best['retro_macro_f1'] if oos_best else float('nan'):.3f}"
            )
        except Exception as e:
            print(f"[{i}/{len(codes)}] FAIL {code}: {e}")
            summary_rows.append(
                dict(code=code, variant="ERR", method="", window="", error=str(e))
            )

    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(out_dir / "variant_summary.csv", index=False, encoding="utf-8-sig")
    report = _build_report(sdf, out_dir, oos_start, end, train_cutoff, variants)
    (out_dir / "regime_accuracy_report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
