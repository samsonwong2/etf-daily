#!/usr/bin/env python3
"""Compare adaptive stage-rule scoring variants on the same OOS window.

Variants
--------
V0 baseline : Bayes on train compound (current production)
V1 expect   : Bayes on mean_ret_net (= mean_ret − 2×0.1% cost)
V2 no_trade : V1 + NO_TRADE candidate per stage
V3 holdout  : V1 + train 80/20 time holdout; negative holdout → own×0.5
V4 combo    : V1 + NO_TRADE + holdout

Usage::

    $PY src/etf_daily/scripts/backtest_adaptive_variants.py \\
      --end-date 2026-08-06 --start-date 2026-04-01
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.lib.adaptive_stage_common import (  # noqa: E402
    COST_PER_SIDE,
    FIXED_HYBRID_RULES,
    MIN_TRAIN_BARS,
    NO_TRADE_BUY,
    NO_TRADE_EXIT,
    PRIOR_K,
    build_pool_prior_from_rows,
    hold_up_compound,
    is_no_trade_rule,
    prepare_features,
    score_method,
    select_stage_rules_bayes,
    simulate_adaptive_book,
    simulate_stage_rule,
    sweep_stage_rules,
)
from etf_daily.lib.symbol_trend_board import load_cluster_mapping_codes  # noqa: E402
from etf_daily.paths import CLUSTER_MAPPING_SELECTED_TXT, PLOTLY_OUTPUTS_DIR  # noqa: E402

DEFAULT_START = "2026-04-01"
WARMUP_CALENDAR_DAYS = 220

VARIANTS: dict[str, dict[str, Any]] = {
    "V0_baseline": dict(score_field="compound", include_no_trade=False, holdout=False),
    "V1_expect": dict(score_field="mean_ret_net", include_no_trade=False, holdout=False),
    "V2_no_trade": dict(score_field="mean_ret_net", include_no_trade=True, holdout=False),
    "V3_holdout": dict(score_field="mean_ret_net", include_no_trade=False, holdout=True),
    "V4_combo": dict(score_field="mean_ret_net", include_no_trade=True, holdout=True),
}


def _load_ev():
    path = _PROJECT_ROOT / "src/etf_daily/scripts/eval_causal_regime_switch_pool.py"
    spec = importlib.util.spec_from_file_location("ev_variants", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_pm():
    path = _PROJECT_ROOT / "src/etf_daily/plots/plot_regime_transition_example.py"
    spec = importlib.util.spec_from_file_location("pm_variants", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_ohlcv(pm, code: str, end_date: str) -> pd.DataFrame:
    ohlcv = pm.load_qlib_ohlcv(
        code,
        "2015-01-01",
        end_date,
        init_qlib=False,
        lookback_calendar_days=WARMUP_CALENDAR_DAYS,
        clip_to_window=False,
    )
    if ohlcv is None or ohlcv.empty:
        raise ValueError("empty ohlcv")
    return ohlcv


def _pick_method(ev, px_tr: pd.DataFrame) -> tuple[str, dict[str, float], float]:
    close = px_tr["$close"].to_numpy(float)
    truth = ev.retrospective_truth(close)
    best_m = None
    best_sc = -1e9
    scores: dict[str, float] = {}
    for name, fn in ev.METHODS.items():
        labs = fn(px_tr)
        sc = score_method(labs, close, truth)
        scores[name] = float(sc)
        if sc > best_sc:
            best_sc = sc
            best_m = name
    assert best_m is not None
    return best_m, scores, float(best_sc)


def _sweep_rows(
    px: pd.DataFrame,
    *,
    code: str,
    include_no_trade: bool,
) -> list[dict]:
    rows: list[dict] = []
    for stage in ("up", "down", "range"):
        sw = sweep_stage_rules(px, stage, include_no_trade=include_no_trade)
        for _, r in sw.iterrows():
            rows.append(
                dict(
                    code=code,
                    stage=r["stage"],
                    buy=r["buy"],
                    exit=r["exit"],
                    n_trades=int(r["n_trades"]),
                    compound=float(r["compound"]),
                    edge=float(r["edge"]),
                    mean_ret=float(r.get("mean_ret", 0.0) or 0.0),
                    mean_ret_net=float(r.get("mean_ret_net", 0.0) or 0.0),
                    win=float(r["win"]) if pd.notna(r["win"]) else np.nan,
                )
            )
    return rows


def _apply_holdout_discount(
    px_tr: pd.DataFrame,
    rule_rows: list[dict],
    *,
    code: str,
    include_no_trade: bool,
    frac: float = 0.8,
) -> list[dict]:
    """Re-score with first `frac` for own, last 1-frac for validation discount."""
    n = len(px_tr)
    if n < MIN_TRAIN_BARS + 40:
        return rule_rows
    cut = int(n * frac)
    px_fit = px_tr.iloc[:cut].reset_index(drop=True)
    px_val = px_tr.iloc[cut:].reset_index(drop=True)
    if len(px_val) < 20:
        return rule_rows

    # rebuild scores on fit window
    fit_rows = _sweep_rows(px_fit, code=code, include_no_trade=include_no_trade)
    fit_map = {(r["stage"], r["buy"], r["exit"]): r for r in fit_rows}

    out: list[dict] = []
    for r in rule_rows:
        key = (r["stage"], r["buy"], r["exit"])
        base = dict(fit_map.get(key, r))
        # validate same rule on holdout
        if is_no_trade_rule(r["buy"], r["exit"]):
            base["holdout_mean_ret_net"] = 0.0
            base["holdout_n"] = 0
            base["score_discount"] = 1.0
        else:
            _, sm = simulate_stage_rule(
                px_val, stage=r["stage"], buy_name=r["buy"], exit_name=r["exit"]
            )
            hnet = float(sm.get("mean_ret_net", 0.0) or 0.0)
            hn = int(sm.get("n_trades", 0) or 0)
            base["holdout_mean_ret_net"] = hnet
            base["holdout_n"] = hn
            # discount if holdout expectation negative (and had at least 1 trade)
            disc = 0.5 if (hn > 0 and hnet < 0) else 1.0
            base["score_discount"] = disc
            base["mean_ret_net"] = float(base.get("mean_ret_net", 0.0)) * disc
            base["mean_ret"] = float(base.get("mean_ret", 0.0)) * disc
            base["compound"] = float(base.get("compound", 0.0)) * disc
            base["n_trades"] = int(base.get("n_trades", 0))
        base["code"] = code
        out.append(base)
    return out


def train_symbol_for_variant(
    *,
    code: str,
    ohlcv: pd.DataFrame,
    train_cutoff: str,
    ev,
    variant: dict[str, Any],
) -> dict[str, Any]:
    train_end = (pd.Timestamp(train_cutoff) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    px = prepare_features(ev, ohlcv, method=None)
    px_tr = px[px.as_of <= train_end].reset_index(drop=True)
    train_last = str(px_tr.as_of.iloc[-1].date()) if len(px_tr) else None
    out: dict[str, Any] = dict(
        code=code,
        train_end=train_last,
        train_cutoff=train_cutoff,
        train_bars=len(px_tr),
        skipped_train=False,
        method=None,
        rule_rows=[],
    )
    if len(px_tr) < MIN_TRAIN_BARS:
        out["skipped_train"] = True
        return out

    best_m, scores, best_sc = _pick_method(ev, px_tr)
    px_tr = px_tr.copy()
    px_tr["regime"] = ev.METHODS[best_m](px_tr)
    out["method"] = best_m
    out["method_scores"] = scores
    out["method_score"] = best_sc
    hu, hn = hold_up_compound(px_tr.regime.to_numpy(), px_tr["$close"].to_numpy(float))
    out["hold_up_compound"] = hu
    out["hold_up_n"] = hn

    include_nt = bool(variant["include_no_trade"])
    rule_rows = _sweep_rows(px_tr, code=code, include_no_trade=include_nt)
    if variant["holdout"]:
        rule_rows = _apply_holdout_discount(
            px_tr, rule_rows, code=code, include_no_trade=include_nt
        )
    out["rule_rows"] = rule_rows
    return out


def build_cfg(
    *,
    code: str,
    train_out: dict[str, Any],
    prior_df: pd.DataFrame,
    variant: dict[str, Any],
    k: float,
    modal_method: str,
) -> dict[str, Any]:
    own_df = pd.DataFrame(train_out.get("rule_rows") or [])
    skipped = bool(train_out.get("skipped_train"))
    selected = select_stage_rules_bayes(
        own_df=pd.DataFrame() if skipped else own_df,
        prior_df=prior_df,
        k=k,
        score_field=variant["score_field"],
        include_no_trade=bool(variant["include_no_trade"]),
    )
    method = train_out.get("method") or modal_method
    if skipped:
        method = modal_method
    return dict(
        code=code,
        method=method,
        train_end=train_out.get("train_end"),
        train_cutoff=train_out.get("train_cutoff"),
        train_bars=train_out.get("train_bars"),
        skipped_train=skipped,
        prior_k=k,
        score_field=variant["score_field"],
        include_no_trade=bool(variant["include_no_trade"]),
        holdout=bool(variant["holdout"]),
        rules=selected,
    )


def oos_eval(
    *,
    code: str,
    cfg: dict,
    ohlcv: pd.DataFrame,
    start_date: str,
    end_date: str,
    ev,
) -> dict[str, Any]:
    method = cfg["method"]
    px = prepare_features(ev, ohlcv, method=method)
    px_w = px[(px.as_of >= start_date) & (px.as_of <= end_date)].reset_index(drop=True)
    if len(px_w) < 5:
        raise ValueError(f"too few OOS bars {len(px_w)}")
    rules = {
        st: dict(buy=cfg["rules"][st]["buy"], exit=cfg["rules"][st]["exit"])
        for st in ("up", "down", "range")
    }
    aev, stats, _open = simulate_adaptive_book(px_w, rules=rules)

    # fixed hybrid comparison
    px_fx = prepare_features(ev, ohlcv, method=FIXED_HYBRID_RULES["method"])
    px_fx_w = px_fx[(px_fx.as_of >= start_date) & (px_fx.as_of <= end_date)].reset_index(drop=True)
    _, stats_fx, _ = simulate_adaptive_book(
        px_fx_w,
        rules={
            st: dict(buy=FIXED_HYBRID_RULES[st]["buy"], exit=FIXED_HYBRID_RULES[st]["exit"])
            for st in ("up", "down", "range")
        },
    )

    sells = [e for e in aev if e["side"] == "SELL"]
    stage_stats = {}
    for st in ("up", "down", "range"):
        sub = [e for e in sells if e.get("regime") == st]
        stage_stats[f"{st}_n"] = len(sub)
        stage_stats[f"{st}_win"] = (
            float(np.mean([e["fwd_ret"] > 0 for e in sub])) if sub else float("nan")
        )
        stage_stats[f"{st}_avg"] = float(np.mean([e["fwd_ret"] for e in sub])) if sub else float("nan")

    ru = cfg["rules"]
    return dict(
        code=code,
        method=method,
        up_buy=ru["up"]["buy"],
        up_exit=ru["up"]["exit"],
        down_buy=ru["down"]["buy"],
        down_exit=ru["down"]["exit"],
        range_buy=ru["range"]["buy"],
        range_exit=ru["range"]["exit"],
        compound=stats["compound"],
        bh=stats["bh"],
        edge=stats["edge"],
        n_closed=stats["n"],
        win=stats["win"],
        fixed_hybrid_edge=stats_fx["edge"],
        edge_vs_fixed=float(stats["edge"] - stats_fx["edge"]),
        **stage_stats,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code", action="append", default=None)
    p.add_argument("--start-date", default=DEFAULT_START)
    p.add_argument("--end-date", required=True)
    p.add_argument("--train-cutoff", default=None)
    p.add_argument("--prior-k", type=float, default=PRIOR_K)
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: ~/etf-daily-output/temp/plotly_outputs/{end}all_adaptive_variants",
    )
    p.add_argument(
        "--variants",
        default="V0_baseline,V1_expect,V2_no_trade,V3_holdout,V4_combo",
        help="comma-separated variant keys",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    end = pd.Timestamp(args.end_date).strftime("%Y-%m-%d")
    start = pd.Timestamp(args.start_date).strftime("%Y-%m-%d")
    train_cutoff = pd.Timestamp(args.train_cutoff or start).strftime("%Y-%m-%d")
    end_tag = pd.Timestamp(end).strftime("%Y%m%d")
    k = float(args.prior_k)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = PLOTLY_OUTPUTS_DIR / f"{end_tag}all_adaptive_variants"
    out_dir = out_dir if out_dir.is_absolute() else _PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    codes = [c.upper() for c in (args.code or [])] or list(
        load_cluster_mapping_codes(CLUSTER_MAPPING_SELECTED_TXT)
    )
    if args.max_codes:
        codes = codes[: args.max_codes]

    want = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in want:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v}; choose from {list(VARIANTS)}")

    ev = _load_ev()
    pm = _load_pm()
    pm._init_qlib(None)

    print(f"codes={len(codes)} train<{train_cutoff} OOS={start}→{end} variants={want}")

    # Preload OHLCV once
    ohlcv_map: dict[str, pd.DataFrame] = {}
    load_errors: list[dict] = []
    for i, code in enumerate(codes, 1):
        try:
            ohlcv_map[code] = _load_ohlcv(pm, code, end)
            print(f"  load [{i}/{len(codes)}] {code} ok")
        except Exception as e:
            load_errors.append(dict(code=code, error=str(e)))
            print(f"  load [{i}/{len(codes)}] {code} FAIL: {e}")

    all_rows: list[dict] = []
    report_blocks: list[str] = []

    for vname in want:
        variant = VARIANTS[vname]
        print(f"\n=== {vname} score={variant['score_field']} "
              f"no_trade={variant['include_no_trade']} holdout={variant['holdout']} ===")
        train_outs: dict[str, dict] = {}
        all_rule_rows: list[dict] = []
        for code, ohlcv in ohlcv_map.items():
            try:
                tout = train_symbol_for_variant(
                    code=code,
                    ohlcv=ohlcv,
                    train_cutoff=train_cutoff,
                    ev=ev,
                    variant=variant,
                )
                train_outs[code] = tout
                all_rule_rows.extend(tout.get("rule_rows") or [])
                tag = "SKIP" if tout["skipped_train"] else tout["method"]
                print(f"  train {code} bars={tout['train_bars']} method={tag}")
            except Exception as e:
                print(f"  train FAIL {code}: {e}")
                traceback.print_exc()

        prior_df = build_pool_prior_from_rows(
            all_rule_rows, score_field=variant["score_field"]
        )
        # ensure NO_TRADE in prior when enabled
        if variant["include_no_trade"]:
            extra = []
            for st in ("up", "down", "range"):
                if not ((prior_df.stage == st) & (prior_df.buy == NO_TRADE_BUY)).any():
                    extra.append(
                        dict(
                            stage=st,
                            buy=NO_TRADE_BUY,
                            exit=NO_TRADE_EXIT,
                            prior_score=0.0,
                            n_symbols=0,
                            mean_n=0.0,
                        )
                    )
            if extra:
                prior_df = pd.concat([prior_df, pd.DataFrame(extra)], ignore_index=True)
                prior_df = prior_df.sort_values(
                    ["stage", "prior_score"], ascending=[True, False]
                )

        prior_path = out_dir / f"prior_{vname}.csv"
        prior_df.to_csv(prior_path, index=False, encoding="utf-8-sig")

        methods = [t["method"] for t in train_outs.values() if t.get("method")]
        modal_method = (
            max(set(methods), key=methods.count) if methods else FIXED_HYBRID_RULES["method"]
        )

        v_rows: list[dict] = []
        for code, ohlcv in ohlcv_map.items():
            try:
                if code not in train_outs:
                    fake = dict(
                        code=code,
                        skipped_train=True,
                        method=None,
                        rule_rows=[],
                        train_end=None,
                        train_cutoff=train_cutoff,
                        train_bars=0,
                    )
                    cfg = build_cfg(
                        code=code,
                        train_out=fake,
                        prior_df=prior_df,
                        variant=variant,
                        k=k,
                        modal_method=modal_method,
                    )
                else:
                    cfg = build_cfg(
                        code=code,
                        train_out=train_outs[code],
                        prior_df=prior_df,
                        variant=variant,
                        k=k,
                        modal_method=modal_method,
                    )
                r = oos_eval(
                    code=code,
                    cfg=cfg,
                    ohlcv=ohlcv,
                    start_date=start,
                    end_date=end,
                    ev=ev,
                )
                r["variant"] = vname
                v_rows.append(r)
                all_rows.append(r)
                print(
                    f"  oos {code} edge={r['edge']:+.1%} win="
                    f"{(f'{r['win']:.0%}' if r['win'] == r['win'] else 'n/a')} "
                    f"n={r['n_closed']} down={r['down_buy']}→{r['down_exit']}"
                )
            except Exception as e:
                print(f"  oos FAIL {code}: {e}")
                traceback.print_exc()

        vs = pd.DataFrame(v_rows)
        vs.to_csv(out_dir / f"summary_{vname}.csv", index=False, encoding="utf-8-sig")

        # aggregate block
        mean_edge = float(vs["edge"].mean()) if len(vs) else float("nan")
        mean_win = float(vs["win"].mean()) if len(vs) else float("nan")
        mean_fx = float(vs["fixed_hybrid_edge"].mean()) if len(vs) else float("nan")
        edge_pos = float((vs["edge"] > 0).mean()) if len(vs) else float("nan")
        # pool-level down trades: recompute from per-symbol stage stats weighted by n
        dn_n = int(vs["down_n"].sum()) if "down_n" in vs else 0
        if dn_n > 0:
            # weighted win
            wsum = 0.0
            for _, row in vs.iterrows():
                if row["down_n"] > 0 and pd.notna(row["down_win"]):
                    wsum += row["down_win"] * row["down_n"]
            down_win = wsum / dn_n
        else:
            down_win = float("nan")
        no_trade_down = int((vs["down_buy"] == NO_TRADE_BUY).sum()) if len(vs) else 0
        block = (
            f"### {vname}\n\n"
            f"- mean edge: **{mean_edge:+.2%}** (fixed hybrid {mean_fx:+.2%})\n"
            f"- mean win: **{mean_win:.1%}** · edge>0: {edge_pos:.0%}\n"
            f"- down-stage trades: n={dn_n}, win={down_win:.1%} · "
            f"NO_TRADE down symbols: {no_trade_down}/{len(vs)}\n"
            f"- score_field=`{variant['score_field']}` · "
            f"no_trade={variant['include_no_trade']} · holdout={variant['holdout']}\n"
        )
        report_blocks.append(block)
        print(
            f"  SUMMARY {vname}: mean_edge={mean_edge:+.2%} mean_win={mean_win:.1%} "
            f"down_win={down_win:.1%} no_trade_down={no_trade_down}"
        )

    summary = pd.DataFrame(all_rows)
    summary.to_csv(out_dir / "variants_summary.csv", index=False, encoding="utf-8-sig")

    # pick best by mean_win then mean_edge
    ranks = []
    for vname in want:
        sub = summary[summary.variant == vname]
        ranks.append(
            dict(
                variant=vname,
                mean_edge=float(sub.edge.mean()),
                mean_win=float(sub.win.mean()) if sub.win.notna().any() else float("nan"),
                edge_pos=float((sub.edge > 0).mean()),
                mean_n=float(sub.n_closed.mean()),
            )
        )
    rank_df = pd.DataFrame(ranks).sort_values(
        ["mean_win", "mean_edge"], ascending=[False, False]
    )
    best = rank_df.iloc[0]["variant"] if len(rank_df) else "V0_baseline"
    rank_df.to_csv(out_dir / "variants_rank.csv", index=False, encoding="utf-8-sig")

    report = f"""# Adaptive stage-rule variants OOS report

> train < **{train_cutoff}** · OOS **{start} → {end}** · k={k} · cost/side={COST_PER_SIDE}
> codes={len(ohlcv_map)} (load fail {len(load_errors)})

## Ranking (by mean win, then mean edge)

| variant | mean_win | mean_edge | edge>0 | mean_n |
|:--|---:|---:|---:|---:|
"""
    for _, r in rank_df.iterrows():
        report += (
            f"| {r['variant']} | {r['mean_win']:.1%} | {r['mean_edge']:+.2%} | "
            f"{r['edge_pos']:.0%} | {r['mean_n']:.1f} |\n"
        )
    report += f"\n**Recommended: `{best}`**\n\n"
    report += "## Per-variant detail\n\n" + "\n".join(report_blocks)
    report += f"""
## Notes

- V0 uses train **compound** (baseline production).
- V1+ use **mean_ret_net** = avg trade return − 2×{COST_PER_SIDE:.1%} round-trip cost.
- NO_TRADE: stage allowed to stay flat when posterior of empty rule wins.
- Holdout: fit on first 80% of train bars; if holdout mean_ret_net < 0, own score ×0.5.

## Files

- `variants_summary.csv` · `variants_rank.csv` · `prior_{{variant}}.csv` · `summary_{{variant}}.csv`
"""
    (out_dir / "variants_report.md").write_text(report, encoding="utf-8")
    print(f"\nDONE best={best} out={out_dir}")
    print(rank_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
