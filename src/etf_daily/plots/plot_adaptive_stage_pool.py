#!/usr/bin/env python3
"""Per-symbol adaptive regime method + long-only up-regime trading.

Pass 1: train on history before TRAIN_CUTOFF; pick causal METHOD per symbol.
Pass 2: freeze trade rules. Default --trade-mode=hold_up:
  buy at up-regime start, sell when up ends; no trade in down/range.

Optional --trade-mode=bayes_rules keeps the old Bayesian per-stage rule sweep.

Daily::

    AS_OF=2026-08-06 PY=... ./scripts/daily_adaptive_stage_html.sh
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    METHOD_IMPL_VERSION,
    NO_TRADE_BUY,
    NO_TRADE_EXIT,
    PRIOR_K,
    REGIME_HOLD_UP_RULES,
    REGIME_SELECT_ENSEMBLE_CV,
    REGIME_SELECT_LEGACY,
    REGIME_SELECT_STATE_SPACE_NESTED_CV,
    TRADE_MODE_BAYES,
    TRADE_MODE_HOLD_UP,
    TRUTH_VERSION,
    annotate_figure,
    build_pool_prior_from_rows,
    detect_early_regime_marks,
    hold_up_compound,
    is_no_trade_rule,
    label_regime,
    prepare_features,
    segments,
    select_regime_method,
    select_stage_rules_bayes,
    simulate_adaptive_book,
    simulate_stage_rule,
    sweep_stage_rules,
)
from etf_daily.lib.early_down_alert import detect_early_down_marks  # noqa: E402
from etf_daily.lib.early_down_alert import early_down_flags_frame  # noqa: E402
from etf_daily.hrp.hrp_cluster_router import (  # noqa: E402
    DEFAULT_MEMBERSHIP_CSV,
    code_mode_map,
    load_cluster_membership,
    mode_allows_diagnostic_marks,
    mode_allows_trade_overlay,
    paint_cluster_modes,
)
from etf_daily.lib.recovery_up_overlay import (  # noqa: E402
    DIAGNOSTIC_N,
    SHALLOW_N,
    SHALLOW_P20_MIN_DEFAULT,
    apply_recovery_up,
    detect_recovery_up_marks,
    ed_alert_bool_from_frame,
    latch_recovery_up_mask,
    recovery_up_mask,
)
from etf_daily.lib.regime_transition_live_snapshot import (  # noqa: E402
    apply_live_snapshot_to_ohlcv,
    load_live_snapshot,
    snapshot_meta,
)
from etf_daily.lib.regime_transition_plot import (  # noqa: E402
    compute_volatility_regime_frame,
    load_validation_csvs,
)
from etf_daily.lib.vol_need_return_board import (  # noqa: E402
    ANCHOR_ERP_ANN,
    BOND_ANCHOR,
    BOND_ANCHOR_LABEL,
    BOND_ERP_ANN,
    EQUITY_ANCHOR,
    EQUITY_ANCHOR_LABEL,
    HORIZON_5,
    HORIZON_20,
    fair_need_ann_series,
    gap_realized_minus_need,
    is_defensive_code,
)
from etf_daily.lib.plotly_html_autoscale import write_adaptive_html  # noqa: E402
from etf_daily.lib.self_train_policy import (  # noqa: E402
    config_block as self_train_config_block,
    evaluate_self_train_policy,
    policy_frame_from_eval,
)
from etf_daily.lib.symbol_trend_board import load_cluster_mapping_codes  # noqa: E402
from etf_daily.paths import CLUSTER_MAPPING_SELECTED_TXT, PLOTLY_OUTPUTS_DIR  # noqa: E402

DEFAULT_VALIDATION_DIR = (
    _PROJECT_ROOT
    / "runtime/decision_packs/20260720/regime_transition_validation_q90_to0720"
)
DEFAULT_START = "2026-04-01"
WARMUP_CALENDAR_DAYS = 220

# Default = V2 from variants backtest (best OOS mean edge + win)
SCORE_MODES: dict[str, dict[str, Any]] = {
    "compound": dict(score_field="compound", include_no_trade=False, holdout=False),
    "expect": dict(score_field="mean_ret_net", include_no_trade=False, holdout=False),
    "expect_no_trade": dict(score_field="mean_ret_net", include_no_trade=True, holdout=False),
    "expect_holdout": dict(score_field="mean_ret_net", include_no_trade=False, holdout=True),
    "combo": dict(score_field="mean_ret_net", include_no_trade=True, holdout=True),
}
DEFAULT_SCORE_MODE = "expect_no_trade"
# Label-accuracy winner B4 fails trade-edge gate (~+2.9% → negative); keep B0 in prod.
DEFAULT_REGIME_MODE = REGIME_SELECT_LEGACY
# Default trading: long-only on up-regime (enter at up start, exit when up ends).
DEFAULT_TRADE_MODE = TRADE_MODE_HOLD_UP


def _hold_up_rules_cfg() -> dict[str, dict[str, Any]]:
    """Frozen long-only rules: trade only the up segment."""
    out: dict[str, dict[str, Any]] = {}
    for st, r in REGIME_HOLD_UP_RULES.items():
        out[st] = dict(
            buy=r["buy"],
            exit=r["exit"],
            own_score=None,
            prior_score=None,
            posterior=None,
            n_trades=None,
            source="regime_hold_up",
        )
    return out


def _load_ev():
    path = _PROJECT_ROOT / "src/etf_daily/scripts/eval_causal_regime_switch_pool.py"
    spec = importlib.util.spec_from_file_location("ev_adaptive_pool", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_pm():
    path = _PROJECT_ROOT / "src/etf_daily/plots/plot_regime_transition_example.py"
    spec = importlib.util.spec_from_file_location("pm_adaptive_pool", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _rule_key(stage: str, buy: str, exit_: str) -> str:
    return f"{stage}|{buy}|{exit_}"


def merge_batch_summary_df(
    existing: pd.DataFrame | None,
    new: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge batch summary rows; same ``code`` keeps the latest row.

    Needed because ``plot_adaptive_from_listing`` invokes ``main`` once per
    symbol into a shared ``html-out-dir``; a plain overwrite would leave only
    the last code in ``batch_summary.csv``.
    """
    if new is None or new.empty:
        return existing.copy() if existing is not None else pd.DataFrame()
    if existing is None or existing.empty:
        return new.copy()
    out = pd.concat([existing, new], ignore_index=True)
    if "code" in out.columns:
        out = out.drop_duplicates(subset=["code"], keep="last")
    return out.reset_index(drop=True)


def write_merged_batch_summary(out_dir: Path, summary: pd.DataFrame) -> pd.DataFrame:
    """Write ``batch_summary.csv``, merging with any prior file in ``out_dir``."""
    path = out_dir / "batch_summary.csv"
    prior: pd.DataFrame | None = None
    if path.exists():
        try:
            prior = pd.read_csv(path)
        except Exception:  # noqa: BLE001
            prior = None
    merged = merge_batch_summary_df(prior, summary)
    merged.to_csv(path, index=False, encoding="utf-8-sig")
    return merged


def write_merged_batch_errors(
    out_dir: Path,
    errors: pd.DataFrame | list[dict] | None,
    *,
    clear_codes: list[str] | None = None,
) -> pd.DataFrame:
    """Write ``batch_errors.csv``, merging by code (and phase when present).

    Same per-code listing pattern as ``batch_summary``: plain overwrite would
    keep only the last failure. Successful codes in this invocation are dropped
    from any prior error rows so a re-run that fixes a ticker clears its ghost.
    """
    path = out_dir / "batch_errors.csv"
    if isinstance(errors, list):
        new = pd.DataFrame(errors) if errors else pd.DataFrame()
    elif errors is None:
        new = pd.DataFrame()
    else:
        new = errors.copy()

    prior: pd.DataFrame | None = None
    if path.exists():
        try:
            prior = pd.read_csv(path)
        except Exception:  # noqa: BLE001
            prior = None
    if prior is not None and not prior.empty and clear_codes:
        prior = prior[~prior["code"].astype(str).isin({str(c) for c in clear_codes})]

    if prior is None or prior.empty:
        merged = new.copy()
    elif new is None or new.empty:
        merged = prior.copy()
    else:
        merged = pd.concat([prior, new], ignore_index=True)
        subset = (
            ["code", "phase"]
            if "phase" in merged.columns
            else ["code"]
            if "code" in merged.columns
            else None
        )
        if subset is not None:
            merged = merged.drop_duplicates(subset=subset, keep="last")
        merged = merged.reset_index(drop=True)

    if merged.empty:
        if path.exists():
            path.unlink()
        return pd.DataFrame()
    merged.to_csv(path, index=False, encoding="utf-8-sig")
    return merged


def _sweep_rule_rows(
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
    n = len(px_tr)
    if n < MIN_TRAIN_BARS + 40:
        return rule_rows
    cut = int(n * frac)
    px_fit = px_tr.iloc[:cut].reset_index(drop=True)
    px_val = px_tr.iloc[cut:].reset_index(drop=True)
    if len(px_val) < 20:
        return rule_rows
    fit_rows = _sweep_rule_rows(px_fit, code=code, include_no_trade=include_no_trade)
    fit_map = {(r["stage"], r["buy"], r["exit"]): r for r in fit_rows}
    out: list[dict] = []
    for r in rule_rows:
        key = (r["stage"], r["buy"], r["exit"])
        base = dict(fit_map.get(key, r))
        if is_no_trade_rule(r["buy"], r["exit"]):
            base["score_discount"] = 1.0
        else:
            _, sm = simulate_stage_rule(
                px_val, stage=r["stage"], buy_name=r["buy"], exit_name=r["exit"]
            )
            hnet = float(sm.get("mean_ret_net", 0.0) or 0.0)
            hn = int(sm.get("n_trades", 0) or 0)
            disc = 0.5 if (hn > 0 and hnet < 0) else 1.0
            base["score_discount"] = disc
            base["mean_ret_net"] = float(base.get("mean_ret_net", 0.0)) * disc
            base["mean_ret"] = float(base.get("mean_ret", 0.0)) * disc
            base["compound"] = float(base.get("compound", 0.0)) * disc
            base["n_trades"] = int(base.get("n_trades", 0))
        base["code"] = code
        out.append(base)
    return out


def _load_ohlcv(pm, code: str, end_date: str, *, train_cutoff: str) -> pd.DataFrame:
    """Load from listing through end_date with warmup; clip later."""
    # Far-back floor so pre-2015 listings (e.g. SH510300) still have full history.
    start_load = "2005-01-01"
    ohlcv = pm.load_qlib_ohlcv(
        code,
        start_load,
        end_date,
        init_qlib=False,
        lookback_calendar_days=WARMUP_CALENDAR_DAYS,
        clip_to_window=False,
    )
    if ohlcv is None or ohlcv.empty:
        raise ValueError("empty ohlcv")
    return ohlcv


def train_one_symbol(
    *,
    code: str,
    ohlcv: pd.DataFrame,
    train_cutoff: str,
    ev,
    score_mode: dict[str, Any],
    regime_mode: str = REGIME_SELECT_ENSEMBLE_CV,
    trade_mode: str = TRADE_MODE_HOLD_UP,
) -> dict[str, Any]:
    """Pass-1 per symbol: pick method; optionally sweep stage rules on train."""
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
        method_params={},
        method_scores={},
        rule_rows=[],
        regime_mode=regime_mode,
        trade_mode=trade_mode,
        truth_version=TRUTH_VERSION,
    )
    if len(px_tr) < MIN_TRAIN_BARS:
        out["skipped_train"] = True
        return out

    best_m, method_params, meta = select_regime_method(
        ev, px_tr, mode=regime_mode, trade_mode=trade_mode
    )
    px_tr = px_tr.copy()
    use_guard = regime_mode == REGIME_SELECT_ENSEMBLE_CV or best_m == "ensemble"
    px_tr["regime"] = label_regime(
        ev, px_tr, best_m, method_params, sticky_guard=use_guard
    )
    close = px_tr["$close"].to_numpy(float)
    out["method"] = best_m
    out["method_params"] = method_params
    out["method_scores"] = meta.get("scores") or meta.get("cv_raw") or {}
    out["method_score"] = float(meta.get("score") or 0.0)
    out["method_select_meta"] = {
        k: v for k, v in meta.items() if k not in ("scores", "cv_raw")
    }
    hu, hn = hold_up_compound(px_tr.regime.to_numpy(), close)
    out["hold_up_compound"] = hu
    out["hold_up_n"] = hn

    if trade_mode == TRADE_MODE_HOLD_UP:
        # No per-stage rule sweep: trading is fully determined by regime labels.
        out["rule_rows"] = []
        return out

    include_nt = bool(score_mode["include_no_trade"])
    rule_rows = _sweep_rule_rows(px_tr, code=code, include_no_trade=include_nt)
    if score_mode["holdout"]:
        rule_rows = _apply_holdout_discount(
            px_tr, rule_rows, code=code, include_no_trade=include_nt
        )
    out["rule_rows"] = rule_rows
    return out


def build_pool_prior(all_rule_rows: list[dict], *, score_field: str) -> pd.DataFrame:
    return build_pool_prior_from_rows(all_rule_rows, score_field=score_field)


def select_rules_bayes(
    *,
    code: str,
    train_out: dict[str, Any],
    prior_df: pd.DataFrame,
    k: float = PRIOR_K,
    score_mode: dict[str, Any],
    trade_mode: str = TRADE_MODE_HOLD_UP,
) -> dict[str, Any]:
    """Pass-2: freeze hold-up rules, or Bayesian pick per-stage rules."""
    skipped = bool(train_out.get("skipped_train"))
    method = train_out.get("method") or FIXED_HYBRID_RULES["method"]
    method_params = dict(train_out.get("method_params") or {})
    if skipped:
        method = FIXED_HYBRID_RULES["method"]
        method_params = {}

    if trade_mode == TRADE_MODE_HOLD_UP:
        rules = _hold_up_rules_cfg()
        return dict(
            code=code,
            method=method,
            method_params=method_params,
            method_scores=train_out.get("method_scores") or {},
            method_score=train_out.get("method_score"),
            method_select_meta=train_out.get("method_select_meta") or {},
            regime_mode=train_out.get("regime_mode"),
            trade_mode=trade_mode,
            truth_version=train_out.get("truth_version") or TRUTH_VERSION,
            train_end=train_out.get("train_end"),
            train_cutoff=train_out.get("train_cutoff"),
            train_bars=train_out.get("train_bars"),
            skipped_train=skipped,
            prior_k=k,
            score_field="regime_hold_up",
            include_no_trade=True,
            holdout=False,
            rules=rules,
        )

    own_df = pd.DataFrame(train_out.get("rule_rows") or [])
    selected = select_stage_rules_bayes(
        own_df=pd.DataFrame() if skipped else own_df,
        prior_df=prior_df,
        k=k,
        score_field=score_mode["score_field"],
        include_no_trade=bool(score_mode["include_no_trade"]),
    )

    return dict(
        code=code,
        method=method,
        method_params=method_params,
        method_scores=train_out.get("method_scores") or {},
        method_score=train_out.get("method_score"),
        method_select_meta=train_out.get("method_select_meta") or {},
        regime_mode=train_out.get("regime_mode"),
        trade_mode=trade_mode,
        truth_version=train_out.get("truth_version") or TRUTH_VERSION,
        train_end=train_out.get("train_end"),
        train_cutoff=train_out.get("train_cutoff"),
        train_bars=train_out.get("train_bars"),
        skipped_train=skipped,
        prior_k=k,
        score_field=score_mode["score_field"],
        include_no_trade=bool(score_mode["include_no_trade"]),
        holdout=bool(score_mode["holdout"]),
        rules={
            "up": {kk: selected["up"][kk] for kk in selected["up"]},
            "down": {kk: selected["down"][kk] for kk in selected["down"]},
            "range": {kk: selected["range"][kk] for kk in selected["range"]},
        },
    )


def _rules_for_sim(cfg: dict) -> dict[str, dict[str, str]]:
    return {
        st: dict(buy=cfg["rules"][st]["buy"], exit=cfg["rules"][st]["exit"])
        for st in ("up", "down", "range")
    }


def _close_series_from_ohlcv(ohlcv: pd.DataFrame) -> pd.Series:
    close = ohlcv.set_index(pd.to_datetime(ohlcv["datetime"]).dt.normalize())["$close"]
    return close[~close.index.duplicated(keep="last")].sort_index()


def _load_anchor_vol_series(
    pm,
    code: str,
    end: str,
    *,
    train_cutoff: str,
    live: pd.DataFrame | None,
) -> tuple[pd.Series | None, pd.Series | None]:
    """Load persist rv20/rv5 for a fair-need anchor code."""
    ohlcv_anchor = _load_ohlcv(pm, code, end, train_cutoff=train_cutoff)
    if live is not None and not live.empty:
        live_codes = set(live["code"].astype(str).str.upper())
        if str(code).upper() in live_codes:
            ohlcv_anchor = apply_live_snapshot_to_ohlcv(
                ohlcv_anchor, live, code=code, as_of=end
            )
    vf_anchor = compute_volatility_regime_frame(_close_series_from_ohlcv(ohlcv_anchor))
    if vf_anchor is None or vf_anchor.empty:
        return None, None
    vf_anchor = vf_anchor.copy()
    vf_anchor["as_of"] = pd.to_datetime(vf_anchor["as_of"]).dt.normalize()
    indexed_anchor = vf_anchor.set_index("as_of")
    rv20 = indexed_anchor["rv20"]
    rv5 = indexed_anchor["rv5"] if "rv5" in indexed_anchor.columns else None
    return rv20, rv5


def _need_frame_vs_anchor(
    ohlcv: pd.DataFrame,
    rv20_anchor: pd.Series,
    *,
    rv5_anchor: pd.Series | None = None,
    erp_ann: float = ANCHOR_ERP_ANN,
) -> pd.DataFrame | None:
    """Build as_of / fair-need / 20d+5d gap frame vs equity anchor.

    Row 4 uses persist rv20; row 5 uses persist rv5 when ``rv5_anchor`` is
    present, otherwise falls back to the rv20 annualized series.
    """
    if rv20_anchor is None or rv20_anchor.empty:
        return None
    close = _close_series_from_ohlcv(ohlcv)
    if close.empty:
        return None
    vf = compute_volatility_regime_frame(close)
    if vf is None or vf.empty or "rv20" not in vf.columns:
        return None
    vf = vf.copy()
    vf["as_of"] = pd.to_datetime(vf["as_of"]).dt.normalize()
    indexed = vf.set_index("as_of")
    rv20 = indexed["rv20"]
    need = fair_need_ann_series(rv20, rv20_anchor, erp_ann=erp_ann)
    if need.empty or not np.isfinite(need.to_numpy(dtype=float)).any():
        return None
    need5 = need
    if (
        rv5_anchor is not None
        and not rv5_anchor.empty
        and "rv5" in indexed.columns
    ):
        need5_try = fair_need_ann_series(
            indexed["rv5"], rv5_anchor, erp_ann=erp_ann
        )
        if need5_try is not None and not need5_try.empty:
            need5 = need5_try.reindex(need.index)
    gap20 = gap_realized_minus_need(close, need, bars=HORIZON_20)
    gap5 = gap_realized_minus_need(close, need5, bars=HORIZON_5)
    out = pd.DataFrame(
        {
            "as_of": need.index,
            "r_need_ann": need.reindex(need.index).to_numpy(dtype=float),
            "r_need_ann_5": need5.reindex(need.index).to_numpy(dtype=float),
            "r_need_20d": gap20["r_need_period"].reindex(need.index).to_numpy(dtype=float),
            "r_20d": gap20["r_realized"].reindex(need.index).to_numpy(dtype=float),
            "gap_20d": gap20["gap"].reindex(need.index).to_numpy(dtype=float),
            "r_need_5d": gap5["r_need_period"].reindex(need.index).to_numpy(dtype=float),
            "r_5d": gap5["r_realized"].reindex(need.index).to_numpy(dtype=float),
            "gap_5d": gap5["gap"].reindex(need.index).to_numpy(dtype=float),
        }
    )
    return out


# Backward-compatible alias for older call sites / notebooks.
_need_frame_vs_spx = _need_frame_vs_anchor


def oos_one(
    *,
    code: str,
    cfg: dict,
    ohlcv: pd.DataFrame,
    start_date: str,
    end_date: str,
    ev,
    pm,
    oos: pd.DataFrame,
    rev: pd.DataFrame,
    evt: pd.DataFrame,
    name_map: dict,
    out_dir: Path,
    live_snapshot: pd.DataFrame | None = None,
    hrp_mode_by_code: dict[str, str] | None = None,
    enable_ru_diag: bool = True,
    enable_shallow_up_trade: bool = False,
    rv20_anchor: pd.Series | None = None,
    rv5_anchor: pd.Series | None = None,
    rv20_bond_anchor: pd.Series | None = None,
    rv5_bond_anchor: pd.Series | None = None,
    fair_path_extra_trail_years: float | list[float] | tuple[float, ...] | str | None = None,
) -> dict:
    method = cfg["method"]
    method_params = cfg.get("method_params") or {}
    # Apply provisional intraday bar BEFORE regime / ED / trade simulation.
    # build_figure also injects the same bar for the K-line path; without this,
    # overlays stop at Qlib EOD while the candle extends to as_of.
    ohlcv_work = ohlcv
    provisional_note = ""
    if live_snapshot is not None and not live_snapshot.empty:
        ohlcv_work = apply_live_snapshot_to_ohlcv(
            ohlcv, live_snapshot, code=code, as_of=end_date
        )
        meta = snapshot_meta(live_snapshot)
        captured = meta.get("captured_at") or "unknown"
        provisional_note = (
            f"｜盘中快照 provisional captured_at={captured}（非收盘正式；"
            f"趋势/ED/买卖均含该日）"
        )
    px = prepare_features(ev, ohlcv_work, method=method, method_params=method_params)
    native_labs = np.asarray(px.regime.to_numpy(), dtype=object)
    mode = (hrp_mode_by_code or {}).get(str(code).upper(), "default")
    shallow_note = ""
    if enable_shallow_up_trade and mode_allows_trade_overlay(mode):
        try:
            ed_full = early_down_flags_frame(px, regimes=native_labs)
            ed_bool, ed_ok = ed_alert_bool_from_frame(ed_full, len(px))
            su_mask = recovery_up_mask(
                px,
                native_labs,
                ed_bool if ed_ok else None,
                N=SHALLOW_N,
                p20_min=SHALLOW_P20_MIN_DEFAULT,
            )
            if ed_ok:
                su_mask = latch_recovery_up_mask(
                    px, native_labs, su_mask, ed_bool
                )
            px = px.copy()
            px["regime"] = apply_recovery_up(native_labs, su_mask)
            n_up = int(su_mask.sum())
            shallow_note = (
                f"｜C17 shallow_up交易 overlay p20_min={SHALLOW_P20_MIN_DEFAULT:.1%} "
                f"N={SHALLOW_N} bars={n_up}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[shallow-up-skip] {code}: {exc}")
    px_w = px[(px.as_of >= start_date) & (px.as_of <= end_date)].reset_index(drop=True)
    if len(px_w) < 5:
        raise ValueError(f"too few OOS bars {len(px_w)}")

    close_w = px_w["$close"].to_numpy(float)
    dates = px_w.as_of.dt.strftime("%Y-%m-%d").to_numpy()
    seg_df = segments(px_w.regime.to_numpy(), dates, close_w)
    baseline_method = str(
        (cfg.get("method_select_meta") or {}).get("baseline") or method
    )
    if baseline_method in ev.METHODS:
        baseline_full = ev.run_method(baseline_method, px)
        window_mask = ((px.as_of >= start_date) & (px.as_of <= end_date)).to_numpy()
        baseline_w = np.asarray(baseline_full, dtype=object)[window_mask]
    else:
        baseline_w = px_w.regime.to_numpy(object)
    baseline_seg_df = segments(baseline_w, dates, close_w)
    baseline_diff = float(np.mean(baseline_w != px_w.regime.to_numpy(object)))
    aev, stats, open_mtm = simulate_adaptive_book(px_w, rules=_rules_for_sim(cfg))

    # Fixed hybrid comparison on same OOS window
    px_fix = prepare_features(ev, ohlcv_work, method=FIXED_HYBRID_RULES["method"])
    px_fx_w = px_fix[(px_fix.as_of >= start_date) & (px_fix.as_of <= end_date)].reset_index(drop=True)
    _, stats_fx, _ = simulate_adaptive_book(px_fx_w, rules=_rules_for_sim({"rules": FIXED_HYBRID_RULES}))

    train_cutoff = str(cfg.get("train_cutoff") or start_date)
    stp_eval = evaluate_self_train_policy(
        px["as_of"],
        px["$close"].to_numpy(dtype=float),
        px["regime"].to_numpy(dtype=object),
        train_cutoff=train_cutoff,
    )
    policy_meta = self_train_config_block(stp_eval)
    cfg["self_train_policy"] = policy_meta
    policy_frame = policy_frame_from_eval(stp_eval)
    lo = pd.Timestamp(start_date).normalize()
    hi = pd.Timestamp(end_date).normalize()
    policy_frame = policy_frame[
        (pd.to_datetime(policy_frame["as_of"]).dt.normalize() >= lo)
        & (pd.to_datetime(policy_frame["as_of"]).dt.normalize() <= hi)
    ].copy()

    if is_defensive_code(code):
        need_frame = _need_frame_vs_anchor(
            ohlcv_work,
            rv20_bond_anchor,
            rv5_anchor=rv5_bond_anchor,
            erp_ann=BOND_ERP_ANN,
        )
        need_erp = BOND_ERP_ANN
        need_anchor_label = BOND_ANCHOR_LABEL
    else:
        need_frame = _need_frame_vs_anchor(
            ohlcv_work, rv20_anchor, rv5_anchor=rv5_anchor
        )
        need_erp = ANCHOR_ERP_ANN
        need_anchor_label = EQUITY_ANCHOR_LABEL

    fig, _overlay, name = pm.build_figure(
        code=code,
        start_date=start_date,
        end_date=end_date,
        oos=oos,
        rev=rev,
        evt=evt,
        provider_uri=None,
        display_name=None,
        name_map=name_map,
        template="plotly_white",
        init_qlib=False,
        kline_plotter_cls=None,
        live_snapshot=None,
        need_frame=need_frame,
        policy_frame=policy_frame,
        policy_meta=policy_meta,
        train_cutoff=train_cutoff,
        fair_path_extra_trail_years=fair_path_extra_trail_years,
        erp_ann=need_erp,
        equity_anchor_label=need_anchor_label,
        ohlcv_full=ohlcv_work,
    )
    ru = cfg["rules"]

    def _rule_txt(st: str) -> str:
        b, e = ru[st]["buy"], ru[st]["exit"]
        if is_no_trade_rule(b, e):
            return "不交易"
        return f"{b} → {e}"

    def _pct_txt(x: object) -> str:
        try:
            v = float(x)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "n/a"
        if not np.isfinite(v):
            return "n/a"
        return f"{v:+.1%}"

    stp = policy_meta
    policy_note = (
        f"<br><span style='font-size:11px;color:#333'>"
        f"本票自训练策略=<b>{stp.get('policy_label') or stp.get('policy')}</b>"
        f"（仅用本票 train 截止 {train_cutoff}：hold_up={_pct_txt(stp.get('train_hold_up'))}"
        f" / BH={_pct_txt(stp.get('train_bh'))} / cash=0；"
        f"图6净值对照 + 图7 K线 vs 公平路径(自有log趋势·滚动2年；≠图4公平年化)，不改生产 hold_up 买卖标记）"
        f"</span>"
    )

    title_note = (
        f"<b>{method}</b> 自适应分段"
        + (
            " + 只做多（进上涨买 / 离上涨卖）"
            if cfg.get("trade_mode") == TRADE_MODE_HOLD_UP
            else " + 贝叶斯收缩规则"
        )
        + f"（{start_date}→{end_date}）"
        f"<br>涨：{_rule_txt('up')}｜跌：{_rule_txt('down')}｜盘：{_rule_txt('range')}"
        f"{policy_note}"
        f"<br><span style='font-size:11px;color:#666'>规则训练期截至 "
        f"{cfg.get('train_end') or 'n/a'}（cutoff={cfg.get('train_cutoff')}），本窗样本外；"
        f"trade={cfg.get('trade_mode', TRADE_MODE_HOLD_UP)} · "
        f"score={cfg.get('score_field', 'compound')} · k={cfg.get('prior_k', PRIOR_K)} · "
        f"cost/side={COST_PER_SIDE}"
        f"｜baseline={baseline_method} · 分段差异={baseline_diff:.1%}"
        f"｜提前标(≥6%不交易): V离↓ / X离↑ / ↑进涨 / ↓进跌"
        f"｜ED告警=诊断层(soft_bear/peak8，不改买卖)"
        f"｜RU恢复上涨=谓词诊断双通道(全池，不改买卖)"
        f"{shallow_note}"
        f"{provisional_note}</span>"
    )
    agg_marks = detect_early_regime_marks(px_w, regimes=px_w.regime.to_numpy())
    ed_marks = detect_early_down_marks(px_w, regimes=px_w.regime.to_numpy())
    ru_marks: list[dict] = []
    # Diagnostic dual-channel: emit RU whenever causal mask fires (all codes).
    # Trade overlay (shallow_up) already applied above when gated.
    if enable_ru_diag:
        try:
            # Full history for causal mask, then marks clipped by annotate window
            ed_full = early_down_flags_frame(px, regimes=native_labs)
            ed_bool, ed_ok = ed_alert_bool_from_frame(ed_full, len(px))
            ru_all = detect_recovery_up_marks(
                px,
                native_labs,
                ed_bool if ed_ok else None,
                N=DIAGNOSTIC_N,
            )
            # keep marks that overlap OOS window
            lo_ts, hi_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
            for m in ru_all:
                d0 = pd.Timestamp(str(m.get("date") or ""))
                d1 = pd.Timestamp(str(m.get("end") or m.get("date") or ""))
                if d1 < lo_ts or d0 > hi_ts:
                    continue
                m2 = dict(m)
                m2["hrp_mode"] = mode
                ru_marks.append(m2)
        except Exception as exc:  # noqa: BLE001
            print(f"[ru-diag-skip] {code}: {exc}")
    fig = annotate_figure(
        fig=fig,
        ohlcv=ohlcv_work,
        seg_df=seg_df,
        aev=aev,
        open_mtm=open_mtm,
        stats=stats,
        start_date=start_date,
        end_date=end_date,
        title_note=title_note,
        aggressive_marks=agg_marks,
        early_down_marks=ed_marks,
        recovery_up_marks=ru_marks or None,
    )
    # Filtered probabilities are scaled into the bottom 12% of the price panel;
    # hover values remain the true probabilities.  No smoother is used.
    prob_specs = (
        ("p_up_state", "P(up) filtered", "#2ca02c"),
        ("p_down_state", "P(down) filtered", "#d62728"),
        ("p_high_vol_state", "P(high vol) filtered", "#9467bd"),
    )
    lo, hi = float(np.nanmin(close_w)), float(np.nanmax(close_w))
    span = max(hi - lo, max(abs(lo), 1.0) * 0.01)
    for col, label, color in prob_specs:
        if col not in px_w.columns:
            continue
        prob = pd.to_numeric(px_w[col], errors="coerce").to_numpy(float)
        scaled = lo + np.nan_to_num(prob, nan=0.0) * span * 0.12
        fig.add_trace(
            dict(
                type="scatter",
                x=px_w.as_of,
                y=scaled,
                customdata=prob,
                mode="lines",
                name=label,
                line=dict(color=color, width=1),
                opacity=0.65,
                hovertemplate=f"{label}: %{{customdata:.1%}}<extra></extra>",
            )
        )

    start_tag = pd.Timestamp(start_date).strftime("%Y%m%d")
    end_tag = pd.Timestamp(end_date).strftime("%Y%m%d")
    safe = pm.sanitize_name_for_filename(name)
    out = out_dir / f"regime_transition_{code}_{safe}_{start_tag}_{end_tag}_adaptive.html"
    write_adaptive_html(fig, out)

    trades_dir = out_dir / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(aev).to_csv(trades_dir / f"{code}_trades.csv", index=False, encoding="utf-8-sig")
    seg_df.to_csv(trades_dir / f"{code}_segments.csv", index=False, encoding="utf-8-sig")
    baseline_seg_df.to_csv(
        trades_dir / f"{code}_baseline_segments.csv",
        index=False,
        encoding="utf-8-sig",
    )
    state_cols = [
        c
        for c in (
            "as_of",
            "regime",
            "state_candidate",
            "kalman_level",
            "kalman_slope",
            "kalman_slope_z",
            "p_up_state",
            "p_down_state",
            "p_high_vol_state",
            "state_threshold",
            "state_space_fallback",
            "state_space_reliability",
        )
        if c in px_w.columns
    ]
    if len(state_cols) > 2:
        px_w[state_cols].to_csv(
            trades_dir / f"{code}_filtered_state.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if agg_marks:
        pd.DataFrame(agg_marks).to_csv(
            trades_dir / f"{code}_early_regime_marks.csv", index=False, encoding="utf-8-sig"
        )
    if ed_marks:
        pd.DataFrame(ed_marks).to_csv(
            trades_dir / f"{code}_early_down_marks.csv", index=False, encoding="utf-8-sig"
        )
    if ru_marks:
        pd.DataFrame(ru_marks).to_csv(
            trades_dir / f"{code}_recovery_up_marks.csv", index=False, encoding="utf-8-sig"
        )

    cfg_dir = out_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{code}.json"
    cfg_out = dict(cfg)
    cfg_out["oos"] = dict(
        start=start_date,
        end=end_date,
        compound=stats["compound"],
        bh=stats["bh"],
        edge=stats["edge"],
        n_closed=stats["n"],
        fixed_hybrid_edge=stats_fx["edge"],
    )
    cfg_path.write_text(json.dumps(cfg_out, ensure_ascii=False, indent=2), encoding="utf-8")

    return dict(
        code=code,
        name=name,
        out=out.name,
        method=method,
        train_end=cfg.get("train_end"),
        train_bars=cfg.get("train_bars"),
        skipped_train=cfg.get("skipped_train"),
        up_buy=ru["up"]["buy"],
        up_exit=ru["up"]["exit"],
        up_n=ru["up"].get("n_trades"),
        up_post=ru["up"].get("posterior"),
        down_buy=ru["down"]["buy"],
        down_exit=ru["down"]["exit"],
        down_n=ru["down"].get("n_trades"),
        range_buy=ru["range"]["buy"],
        range_exit=ru["range"]["exit"],
        range_n=ru["range"].get("n_trades"),
        bars=len(px_w),
        compound=stats["compound"],
        bh=stats["bh"],
        edge=stats["edge"],
        n_closed=stats["n"],
        win=stats["win"],
        open_pos=stats["open_pos"],
        open_unrealized=stats["open_unrealized"],
        fixed_hybrid_edge=stats_fx["edge"],
        edge_vs_fixed=float(stats["edge"] - stats_fx["edge"]),
        n_seg=len(seg_df),
        baseline_method=baseline_method,
        baseline_regime_diff=baseline_diff,
    )


def _shared_train_cache_paths(
    train_cache_dir: Path, cache_tag: str
) -> tuple[Path, Path, Path, Path]:
    """Return ``(root, configs_dir, meta, prior)`` under a shared train-cache dir."""
    root = train_cache_dir / cache_tag
    return (
        root,
        root / "configs",
        root / "meta.json",
        root / "pool_prior.csv",
    )


def _train_cache_reusable(
    *,
    retrain: bool,
    cache_meta: Path,
    cfg_dir: Path,
    codes: list[str],
    trade_mode: str,
    prior_path: Path,
) -> bool:
    return (
        (not retrain)
        and cache_meta.exists()
        and all((cfg_dir / f"{c}.json").exists() for c in codes)
        and (trade_mode == TRADE_MODE_HOLD_UP or prior_path.exists())
    )


def _copy_train_cache(
    *,
    src_cfg_dir: Path,
    src_meta: Path,
    src_prior: Path,
    dst_cfg_dir: Path,
    dst_meta: Path,
    dst_prior: Path,
    codes: list[str],
) -> None:
    dst_cfg_dir.mkdir(parents=True, exist_ok=True)
    for c in codes:
        src = src_cfg_dir / f"{c}.json"
        shutil.copy2(src, dst_cfg_dir / f"{c}.json")
    if src_prior.exists():
        shutil.copy2(src_prior, dst_prior)
    if src_meta.exists():
        shutil.copy2(src_meta, dst_meta)


def prepare_adaptive_runtime(
    *,
    end: str,
    start: str,
    train_cutoff: str,
    out_dir: Path,
    codes: list[str],
    validation_dir: Path,
    cluster_mapping: Path | None,
    live_snapshot: Path | None,
    prior_k: float,
    score_mode_name: str,
    regime_mode: str,
    trade_mode: str,
    retrain: bool,
    fail_fast: bool,
    hrp_membership_csv: Path,
    disable_ru_diag: bool,
    enable_shallow_up_trade: bool = False,
    train_cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Load Qlib, validation, configs, and equity-anchor series once per process."""
    k = float(prior_k)
    score_mode = SCORE_MODES[score_mode_name]
    out_dir = out_dir if out_dir.is_absolute() else _PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = out_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cache_tag = (
        f"{train_cutoff.replace('-', '')}_{trade_mode}_{score_mode_name}_"
        f"{regime_mode}_{TRUTH_VERSION.replace('+', '-')}_{METHOD_IMPL_VERSION}"
    )
    prior_path = out_dir / f"pool_prior_train_lt_{cache_tag}.csv"
    cache_meta = out_dir / f".train_cache_{cache_tag}.json"
    shared_root: Path | None = None
    shared_cfg_dir: Path | None = None
    shared_meta: Path | None = None
    shared_prior: Path | None = None
    if train_cache_dir is not None:
        tc = train_cache_dir if train_cache_dir.is_absolute() else _PROJECT_ROOT / train_cache_dir
        shared_root, shared_cfg_dir, shared_meta, shared_prior = _shared_train_cache_paths(
            tc, cache_tag
        )

    val_dir = validation_dir if validation_dir.is_absolute() else _PROJECT_ROOT / validation_dir
    mapping = Path(cluster_mapping) if cluster_mapping else CLUSTER_MAPPING_SELECTED_TXT
    if not mapping.is_absolute():
        mapping = _PROJECT_ROOT / mapping

    ev = _load_ev()
    pm = _load_pm()
    pm._init_qlib(None)
    oos, rev, evt = load_validation_csvs(val_dir)
    name_map = pm.name_map_from_oos(oos)
    live = None
    if live_snapshot is not None:
        snap_path = live_snapshot if live_snapshot.is_absolute() else _PROJECT_ROOT / live_snapshot
        live = load_live_snapshot(snap_path, as_of=end)

    rv20_anchor: pd.Series | None = None
    rv5_anchor: pd.Series | None = None
    rv20_bond_anchor: pd.Series | None = None
    rv5_bond_anchor: pd.Series | None = None
    try:
        rv20_anchor, rv5_anchor = _load_anchor_vol_series(
            pm, EQUITY_ANCHOR, end, train_cutoff=train_cutoff, live=live
        )
        if rv20_anchor is not None:
            print(
                f"[fair-need] anchor={EQUITY_ANCHOR}({EQUITY_ANCHOR_LABEL}) "
                f"erp={ANCHOR_ERP_ANN:.1%} "
                f"rv20_days={int(rv20_anchor.notna().sum())} "
                f"rv5_days={int(rv5_anchor.notna().sum()) if rv5_anchor is not None else 0}"
            )
        else:
            print(f"[fair-need-skip] empty vol frame for {EQUITY_ANCHOR}")
    except Exception as exc:  # noqa: BLE001
        print(f"[fair-need-skip] failed to load {EQUITY_ANCHOR}: {exc}")
        rv20_anchor = None
        rv5_anchor = None
    try:
        rv20_bond_anchor, rv5_bond_anchor = _load_anchor_vol_series(
            pm, BOND_ANCHOR, end, train_cutoff=train_cutoff, live=live
        )
        if rv20_bond_anchor is not None:
            print(
                f"[fair-need-bond] anchor={BOND_ANCHOR}({BOND_ANCHOR_LABEL}) "
                f"erp={BOND_ERP_ANN:.1%} "
                f"rv20_days={int(rv20_bond_anchor.notna().sum())} "
                f"rv5_days="
                f"{int(rv5_bond_anchor.notna().sum()) if rv5_bond_anchor is not None else 0}"
            )
        else:
            print(f"[fair-need-bond-skip] empty vol frame for {BOND_ANCHOR}")
    except Exception as exc:  # noqa: BLE001
        print(f"[fair-need-bond-skip] failed to load {BOND_ANCHOR}: {exc}")
        rv20_bond_anchor = None
        rv5_bond_anchor = None

    enable_ru_diag = not disable_ru_diag
    hrp_mode_by_code: dict[str, str] | None = None
    if enable_ru_diag or enable_shallow_up_trade:
        mem_path = hrp_membership_csv
        if mem_path is None or not str(mem_path).strip():
            print("[hrp-meta-skip] membership unset")
        else:
            if not mem_path.is_absolute():
                mem_path = _PROJECT_ROOT / mem_path
            if mem_path.exists():
                try:
                    mem = load_cluster_membership(mem_path)
                    hrp_mode_by_code = code_mode_map(mem, paint_cluster_modes(mem))
                except Exception as exc:  # noqa: BLE001
                    print(f"[hrp-meta-skip] failed to load membership: {exc}")
                    hrp_mode_by_code = None
            else:
                print(f"[hrp-meta-skip] membership missing: {mem_path}")

    reuse = _train_cache_reusable(
        retrain=retrain,
        cache_meta=cache_meta,
        cfg_dir=cfg_dir,
        codes=codes,
        trade_mode=trade_mode,
        prior_path=prior_path,
    )
    shared_hit = False
    if (
        (not reuse)
        and (not retrain)
        and shared_cfg_dir is not None
        and shared_meta is not None
        and shared_prior is not None
        and _train_cache_reusable(
            retrain=False,
            cache_meta=shared_meta,
            cfg_dir=shared_cfg_dir,
            codes=codes,
            trade_mode=trade_mode,
            prior_path=shared_prior,
        )
    ):
        _copy_train_cache(
            src_cfg_dir=shared_cfg_dir,
            src_meta=shared_meta,
            src_prior=shared_prior,
            dst_cfg_dir=cfg_dir,
            dst_meta=cache_meta,
            dst_prior=prior_path,
            codes=codes,
        )
        reuse = True
        shared_hit = True
    configs: dict[str, dict] = {}
    train_errors: list[dict] = []
    if reuse:
        src = "shared" if shared_hit else "local"
        print(f"[cache] reuse configs for {cache_tag} ({src})")
        for c in codes:
            configs[c] = json.loads((cfg_dir / f"{c}.json").read_text(encoding="utf-8"))
    else:
        print("[pass1] train regime methods")
        all_rule_rows: list[dict] = []
        train_outs: dict[str, dict] = {}
        for i, code in enumerate(codes, 1):
            try:
                ohlcv = _load_ohlcv(pm, code, end, train_cutoff=train_cutoff)
                tout = train_one_symbol(
                    code=code,
                    ohlcv=ohlcv,
                    train_cutoff=train_cutoff,
                    ev=ev,
                    score_mode=score_mode,
                    regime_mode=regime_mode,
                    trade_mode=trade_mode,
                )
                train_outs[code] = tout
                all_rule_rows.extend(tout.get("rule_rows") or [])
            except Exception as e:
                train_errors.append(dict(code=code, phase="train", error=str(e)))
                print(f"  [{i}/{len(codes)}] FAIL train {code}: {e}")
                if fail_fast:
                    raise
        if trade_mode == TRADE_MODE_HOLD_UP:
            prior_df = pd.DataFrame(
                columns=["stage", "buy", "exit", "prior_score", "n_symbols", "mean_n"]
            )
        else:
            prior_df = build_pool_prior(
                all_rule_rows, score_field=score_mode["score_field"]
            )
        prior_df.to_csv(prior_path, index=False, encoding="utf-8-sig")
        methods = [t["method"] for t in train_outs.values() if t.get("method")]
        modal_method = (
            max(set(methods), key=methods.count) if methods else FIXED_HYBRID_RULES["method"]
        )
        for code in codes:
            if code not in train_outs:
                fake = dict(
                    code=code,
                    skipped_train=True,
                    method=None,
                    method_scores={},
                    rule_rows=[],
                    train_end=None,
                    train_cutoff=train_cutoff,
                    train_bars=0,
                    regime_mode=regime_mode,
                    trade_mode=trade_mode,
                )
                cfg = select_rules_bayes(
                    code=code,
                    train_out=fake,
                    prior_df=prior_df,
                    k=k,
                    score_mode=score_mode,
                    trade_mode=trade_mode,
                )
                cfg["method"] = modal_method
                cfg["method_params"] = {}
            else:
                cfg = select_rules_bayes(
                    code=code,
                    train_out=train_outs[code],
                    prior_df=prior_df,
                    k=k,
                    score_mode=score_mode,
                    trade_mode=trade_mode,
                )
            cfg["score_mode"] = score_mode_name
            cfg["regime_mode"] = regime_mode
            cfg["trade_mode"] = trade_mode
            cfg["truth_version"] = TRUTH_VERSION
            configs[code] = cfg
            (cfg_dir / f"{code}.json").write_text(
                json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        cache_meta.write_text(
            json.dumps(
                dict(
                    train_cutoff=train_cutoff,
                    n_codes=len(codes),
                    prior_k=k,
                    score_mode=score_mode_name,
                    regime_mode=regime_mode,
                    trade_mode=trade_mode,
                    truth_version=TRUTH_VERSION,
                    method_impl_version=METHOD_IMPL_VERSION,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        if shared_cfg_dir is not None and shared_meta is not None and shared_prior is not None:
            shared_cfg_dir.mkdir(parents=True, exist_ok=True)
            for code in codes:
                src = cfg_dir / f"{code}.json"
                if src.exists():
                    shutil.copy2(src, shared_cfg_dir / f"{code}.json")
            if prior_path.exists():
                shutil.copy2(prior_path, shared_prior)
            # Meta reflects union of configs present under the shared tag.
            n_shared = len(list(shared_cfg_dir.glob("*.json")))
            shared_meta.write_text(
                json.dumps(
                    dict(
                        train_cutoff=train_cutoff,
                        n_codes=n_shared,
                        prior_k=k,
                        score_mode=score_mode_name,
                        regime_mode=regime_mode,
                        trade_mode=trade_mode,
                        truth_version=TRUTH_VERSION,
                        method_impl_version=METHOD_IMPL_VERSION,
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[cache] wrote shared train cache {shared_root} (n={n_shared})")

    _ = start
    return dict(
        ev=ev,
        pm=pm,
        oos=oos,
        rev=rev,
        evt=evt,
        name_map=name_map,
        live=live,
        rv20_anchor=rv20_anchor,
        rv5_anchor=rv5_anchor,
        rv20_bond_anchor=rv20_bond_anchor,
        rv5_bond_anchor=rv5_bond_anchor,
        hrp_mode_by_code=hrp_mode_by_code,
        configs=configs,
        enable_ru_diag=enable_ru_diag,
        enable_shallow_up_trade=enable_shallow_up_trade,
        train_errors=train_errors,
        score_mode=score_mode,
        score_mode_name=score_mode_name,
        out_dir=out_dir,
        end=end,
        train_cutoff=train_cutoff,
        trade_mode=trade_mode,
        regime_mode=regime_mode,
    )


_PARALLEL_OOS_CTX: dict[str, Any] | None = None


def _init_parallel_oos_worker(ctx: dict[str, Any]) -> None:
    global _PARALLEL_OOS_CTX
    _PARALLEL_OOS_CTX = ctx
    pm = ctx["pm"]
    pm._init_qlib(None)


def _oos_one_worker(code: str) -> tuple[str, dict | None, str | None]:
    ctx = _PARALLEL_OOS_CTX
    if ctx is None:
        raise RuntimeError("parallel OOS worker context not initialized")
    try:
        ohlcv = _load_ohlcv(
            ctx["pm"], code, ctx["end"], train_cutoff=ctx["train_cutoff"]
        )
        cfg = ctx["configs"][code]
        r = oos_one(
            code=code,
            cfg=cfg,
            ohlcv=ohlcv,
            start_date=ctx["start"],
            end_date=ctx["end"],
            ev=ctx["ev"],
            pm=ctx["pm"],
            oos=ctx["oos"],
            rev=ctx["rev"],
            evt=ctx["evt"],
            name_map=ctx["name_map"],
            out_dir=ctx["out_dir"],
            live_snapshot=ctx["live"],
            hrp_mode_by_code=ctx["hrp_mode_by_code"],
            enable_ru_diag=ctx["enable_ru_diag"],
            enable_shallow_up_trade=ctx["enable_shallow_up_trade"],
            rv20_anchor=ctx["rv20_anchor"],
            rv5_anchor=ctx["rv5_anchor"],
            rv20_bond_anchor=ctx["rv20_bond_anchor"],
            rv5_bond_anchor=ctx["rv5_bond_anchor"],
            fair_path_extra_trail_years=ctx["fair_path_extra_trail_years"],
        )
        return code, r, None
    except Exception as exc:  # noqa: BLE001
        return code, None, str(exc)


def _run_oos_batch(
    *,
    codes: list[str],
    jobs: int,
    ctx: dict[str, Any],
    fail_fast: bool,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    errors: list[dict] = []
    n_codes = len(codes)
    if jobs <= 1 or n_codes <= 1:
        for i, code in enumerate(codes, 1):
            try:
                ohlcv = _load_ohlcv(
                    ctx["pm"], code, ctx["end"], train_cutoff=ctx["train_cutoff"]
                )
                cfg = ctx["configs"][code]
                r = oos_one(
                    code=code,
                    cfg=cfg,
                    ohlcv=ohlcv,
                    start_date=ctx["start"],
                    end_date=ctx["end"],
                    ev=ctx["ev"],
                    pm=ctx["pm"],
                    oos=ctx["oos"],
                    rev=ctx["rev"],
                    evt=ctx["evt"],
                    name_map=ctx["name_map"],
                    out_dir=ctx["out_dir"],
                    live_snapshot=ctx["live"],
                    hrp_mode_by_code=ctx["hrp_mode_by_code"],
                    enable_ru_diag=ctx["enable_ru_diag"],
                    enable_shallow_up_trade=ctx["enable_shallow_up_trade"],
                    rv20_anchor=ctx["rv20_anchor"],
                    rv5_anchor=ctx["rv5_anchor"],
                    rv20_bond_anchor=ctx["rv20_bond_anchor"],
                    rv5_bond_anchor=ctx["rv5_bond_anchor"],
                    fair_path_extra_trail_years=ctx["fair_path_extra_trail_years"],
                )
                rows.append(r)
                print(
                    f"[{i}/{n_codes}] OK {code} method={r['method']} "
                    f"edge={r['edge']:+.1%} vs_fixed={r['edge_vs_fixed']:+.1%}"
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(dict(code=code, phase="oos", error=str(exc)))
                print(f"[{i}/{n_codes}] FAIL oos {code}: {exc}")
                if fail_fast:
                    raise
        return rows, errors

    done = 0
    with ProcessPoolExecutor(
        max_workers=jobs, initializer=_init_parallel_oos_worker, initargs=(ctx,)
    ) as pool:
        futs = {pool.submit(_oos_one_worker, code): code for code in codes}
        for fut in as_completed(futs):
            code, r, err = fut.result()
            done += 1
            if err is None and r is not None:
                rows.append(r)
                print(
                    f"[{done}/{n_codes}] OK {code} method={r['method']} "
                    f"edge={r['edge']:+.1%} vs_fixed={r['edge_vs_fixed']:+.1%}"
                )
            else:
                errors.append(dict(code=code, phase="oos", error=err or "unknown"))
                print(f"[{done}/{n_codes}] FAIL oos {code}: {err}")
                if fail_fast:
                    raise RuntimeError(f"oos failed for {code}: {err}")
    return rows, errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code", action="append", default=None)
    p.add_argument("--start-date", default=DEFAULT_START, help="OOS / plot start (= train cutoff)")
    p.add_argument("--end-date", required=True)
    p.add_argument(
        "--train-cutoff",
        default=None,
        help="train uses dates < cutoff; default = start-date",
    )
    p.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    p.add_argument("--cluster-mapping", type=Path, default=None)
    p.add_argument("--html-out-dir", type=Path, default=None)
    p.add_argument("--live-snapshot", type=Path, default=None)
    p.add_argument("--prior-k", type=float, default=PRIOR_K)
    p.add_argument(
        "--score-mode",
        default=DEFAULT_SCORE_MODE,
        choices=list(SCORE_MODES),
        help="rule scoring mode; default expect_no_trade (V2 variants winner)",
    )
    p.add_argument(
        "--regime-mode",
        default=DEFAULT_REGIME_MODE,
        choices=[
            REGIME_SELECT_ENSEMBLE_CV,
            REGIME_SELECT_LEGACY,
            REGIME_SELECT_STATE_SPACE_NESTED_CV,
        ],
        help="regime method selection; default legacy_score_method "
        "(state_space_nested_cv is explicit opt-in and keeps baseline on gate failure)",
    )
    p.add_argument(
        "--trade-mode",
        default=DEFAULT_TRADE_MODE,
        choices=[TRADE_MODE_HOLD_UP, TRADE_MODE_BAYES],
        help="hold_up=long-only enter/exit with up regime; bayes_rules=old per-stage sweep",
    )
    p.add_argument("--retrain", action="store_true", help="rebuild configs even if cache exists")
    p.add_argument("--continue-on-error", action="store_true", default=True)
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument(
        "--hrp-membership-csv",
        type=Path,
        default=DEFAULT_MEMBERSHIP_CSV,
        help="HRP cluster_representatives CSV (optional metadata on RU marks)",
    )
    p.add_argument(
        "--disable-ru-diag",
        action="store_true",
        help="disable recovery_up diagnostic marks on HTML",
    )
    p.add_argument(
        "--enable-shallow-up-trade",
        action="store_true",
        help="research: apply C17 shallow_up range→up overlay (A/B 20260813 keep_baseline; off by default)",
    )
    p.add_argument(
        "--fair-path-extra-trail-years",
        type=str,
        default="1,0.5,0.25,1/12",
        help=(
            "comma-separated extra trail years for fig8+ after fig7=2y "
            "(production default: 1y,6m,3m,1m). Use off|none|0 to disable."
        ),
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="parallel workers for per-code OOS HTML (default 1 = sequential)",
    )
    p.add_argument(
        "--train-cache-dir",
        type=Path,
        default=None,
        help=(
            "optional shared dir for pass1 configs across AS_OF out dirs; "
            "same TRAIN_CUTOFF + version tag reuses without retraining"
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    end = pd.Timestamp(args.end_date).strftime("%Y-%m-%d")
    start = pd.Timestamp(args.start_date).strftime("%Y-%m-%d")
    train_cutoff = pd.Timestamp(args.train_cutoff or start).strftime("%Y-%m-%d")
    end_tag = pd.Timestamp(end).strftime("%Y%m%d")
    score_mode_name = args.score_mode
    score_mode = SCORE_MODES[score_mode_name]
    regime_mode = args.regime_mode
    trade_mode = args.trade_mode

    out_dir = args.html_out_dir
    if out_dir is None:
        out_dir = PLOTLY_OUTPUTS_DIR / f"{end_tag}all_adaptive"
    out_dir = out_dir if out_dir.is_absolute() else _PROJECT_ROOT / out_dir

    mapping = Path(args.cluster_mapping) if args.cluster_mapping else CLUSTER_MAPPING_SELECTED_TXT
    if not mapping.is_absolute():
        mapping = _PROJECT_ROOT / mapping
    codes = [c.upper() for c in (args.code or [])] or list(load_cluster_mapping_codes(mapping))
    if args.max_codes:
        codes = codes[: args.max_codes]

    print(
        f"codes={len(codes)} train< {train_cutoff} OOS={start}→{end} "
        f"k={args.prior_k} jobs={args.jobs} trade_mode={trade_mode} "
        f"score_mode={score_mode_name} regime_mode={regime_mode} out={out_dir}"
    )
    rt = prepare_adaptive_runtime(
        end=end,
        start=start,
        train_cutoff=train_cutoff,
        out_dir=out_dir,
        codes=codes,
        validation_dir=args.validation_dir,
        cluster_mapping=args.cluster_mapping,
        live_snapshot=args.live_snapshot,
        prior_k=float(args.prior_k),
        score_mode_name=score_mode_name,
        regime_mode=regime_mode,
        trade_mode=trade_mode,
        retrain=bool(args.retrain),
        fail_fast=bool(args.fail_fast),
        hrp_membership_csv=args.hrp_membership_csv,
        disable_ru_diag=bool(args.disable_ru_diag),
        enable_shallow_up_trade=bool(args.enable_shallow_up_trade),
        train_cache_dir=args.train_cache_dir,
    )
    ev = rt["ev"]
    pm = rt["pm"]
    oos, rev, evt = rt["oos"], rt["rev"], rt["evt"]
    name_map = rt["name_map"]
    live = rt["live"]
    rv20_anchor = rt["rv20_anchor"]
    rv5_anchor = rt["rv5_anchor"]
    rv20_bond_anchor = rt["rv20_bond_anchor"]
    rv5_bond_anchor = rt["rv5_bond_anchor"]
    hrp_mode_by_code = rt["hrp_mode_by_code"]
    configs = rt["configs"]
    enable_ru_diag = rt["enable_ru_diag"]
    enable_shallow_up_trade = rt["enable_shallow_up_trade"]
    train_errors = rt["train_errors"]
    out_dir = rt["out_dir"]

    oos_ctx = dict(
        pm=pm,
        ev=ev,
        oos=oos,
        rev=rev,
        evt=evt,
        name_map=name_map,
        live=live,
        rv20_anchor=rv20_anchor,
        rv5_anchor=rv5_anchor,
        rv20_bond_anchor=rv20_bond_anchor,
        rv5_bond_anchor=rv5_bond_anchor,
        hrp_mode_by_code=hrp_mode_by_code,
        configs=configs,
        enable_ru_diag=enable_ru_diag,
        enable_shallow_up_trade=enable_shallow_up_trade,
        out_dir=out_dir,
        start=start,
        end=end,
        train_cutoff=train_cutoff,
        fair_path_extra_trail_years=args.fair_path_extra_trail_years,
    )

    # OOS
    print("[oos] frozen execute + HTML")
    try:
        rows, oos_errors = _run_oos_batch(
            codes=codes,
            jobs=int(args.jobs),
            ctx=oos_ctx,
            fail_fast=bool(args.fail_fast),
        )
    except Exception:
        traceback.print_exc()
        return 1
    errors: list[dict] = list(train_errors)
    errors.extend(oos_errors)

    summary = pd.DataFrame(rows)
    # Merge with prior CSV so per-code listing runs accumulate (not overwrite).
    summary_all = write_merged_batch_summary(out_dir, summary)
    ok_codes = [str(r.get("code")) for r in rows if r.get("code")]
    write_merged_batch_errors(out_dir, errors, clear_codes=ok_codes)

    # README / DONE line: prefer merged pool file when this invocation is partial.
    stats_df = summary_all if len(summary_all) else summary
    mean_edge = float(stats_df["edge"].mean()) if len(stats_df) else float("nan")
    mean_fx = float(stats_df["fixed_hybrid_edge"].mean()) if len(stats_df) else float("nan")
    mean_delta = float(stats_df["edge_vs_fixed"].mean()) if len(stats_df) else float("nan")
    mean_win = float(stats_df["win"].mean()) if len(stats_df) and "win" in stats_df else float("nan")
    win_vs = (
        float((stats_df["edge_vs_fixed"] > 0).mean()) if len(stats_df) else float("nan")
    )
    no_trade_n = 0
    for st in ("up", "down", "range"):
        col = f"{st}_buy"
        if col in stats_df.columns:
            no_trade_n += int((stats_df[col] == NO_TRADE_BUY).sum())
    low_n = stats_df[
        (stats_df["up_n"].fillna(0) + stats_df["down_n"].fillna(0) + stats_df["range_n"].fillna(0)) < 8
    ]
    sf = score_mode["score_field"]
    if trade_mode == TRADE_MODE_HOLD_UP:
        method_section = f"""## 方法

1. **分段法自选**（`regime_mode={regime_mode}` · `truth={TRUTH_VERSION}`）：每标的训练期选因果 METHODS。
2. **交易规则固定为只做多**（`trade_mode=hold_up`）：
   - 上涨态开始 → **买入**（`进入上涨态`）
   - 上涨态结束 → **卖出**（`离上涨态`）
   - 下跌 / 横盘 → **不交易**
3. **样本外**：冻结 `configs/{{code}}.json`，图窗因果执行；**不画期末卖点**，持仓未平标菱形。
"""
    else:
        method_section = f"""## 方法

1. **分段法自选**（`regime_mode={regime_mode}` · `truth={TRUTH_VERSION}`）。
2. **规则库扫描 + 贝叶斯收缩**（`trade_mode=bayes_rules` · `score_mode={score_mode_name}`）：

```
posterior = (n · own_score + k · prior_score) / (n + k)
own_score = {sf}
```

3. **样本外**：冻结配置执行。
"""
    readme = f"""# 按标的自适应趋势阶段（{trade_mode}）

> 训练：各标的历史 **< {train_cutoff}** · 样本外图窗 **{start} → {end}**
> 成功 {len(rows)}/{len(codes)} · 累计汇总 {len(stats_df)} 票 · 失败 {len(errors)} · **trade_mode=`{trade_mode}`** · regime_mode=`{regime_mode}`

{method_section}
## OOS 对比（相对固定 hybrid_ma_adx + SH588710 规则）

| 指标 | 值 |
|:--|--:|
| 自适应 mean edge | {mean_edge:+.2%} |
| 自适应 mean win | {mean_win:.1%} |
| 固定 hybrid mean edge | {mean_fx:+.2%} |
| mean (adaptive − fixed) | {mean_delta:+.2%} |
| 自适应 edge 更高占比 | {win_vs:.0%} |
| 阶段选 NO_TRADE 次数（up+down+range） | {no_trade_n} |

## 输出

- HTML：`*_adaptive.html`
- 定稿：`configs/{{code}}.json`
- 汇总：`batch_summary.csv`

## 每日命令

```bash
AS_OF={end} PY=~/etf-daily-output/python/envs/py312/bin/python \\
  ./scripts/daily_adaptive_stage_html.sh
# 强制重训：RETRAIN=1 ...
# 旧贝叶斯规则：TRADE_MODE=bayes_rules RETRAIN=1 ...
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    n_train_fail = len(train_errors)
    n_oos_fail = len(errors)
    print(
        f"DONE ok={len(rows)} fail={n_oos_fail} train_fail={n_train_fail} "
        f"trade_mode={trade_mode} "
        f"mean_edge={mean_edge:+.2%} mean_win={mean_win:.1%} "
        f"vs_fixed_delta={mean_delta:+.2%} out={out_dir}"
    )
    # continue-on-error still processes siblings, but exit non-zero so wrappers
    # (from_listing) do not treat partial failure as success.
    return 1 if (n_oos_fail or n_train_fail) else 0


if __name__ == "__main__":
    raise SystemExit(main())
