"""Walk-forward synthetic triangle generator for full-history backtests.

Recomputes ``quantile_switch`` / ``switch_side`` / ``pred_cdf`` day-by-day using a
frozen HMM regime-transition model, so the five-class OPS backtest can run from
each ETF's inception instead of only the 2024+ ``signals_oos`` window.

Causality
---------
The triangle at day ``t`` uses a model fit only on returns ``<= t``. To bound the
cost of HMM refits, the model is refit every ``refit_days`` trading sessions and
frozen in between; on refit day the fit still uses data ``<=`` that day.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from decision_pack.src.regime_transition_board import (
    RegimeTransitionConfig,
    compute_frozen_transition_step,
    fit_frozen_regime_model,
)


def generate_walkforward_triangles(
    close: pd.Series,
    *,
    cfg: RegimeTransitionConfig | None = None,
    refit_days: int = 63,
    min_warmup_days: int = 260,
) -> pd.DataFrame:
    """Generate synthetic regime-transition triangles across the full history.

    Parameters
    ----------
    close
        Price series indexed by trading date (normalized).
    cfg
        HMM regime transition config (q* etc.). Defaults to 0.90.
    refit_days
        Number of trading sessions between HMM refits (frozen between).
    min_warmup_days
        Skip emitting triangles until at least this many price points exist, so
        the HMM training window is populated.

    Returns
    -------
    DataFrame with columns ``as_of, quantile_switch, switch_side, pred_cdf``,
    ``quantile_switch`` rows only where the rule fired.
    """
    cfg = cfg or RegimeTransitionConfig()
    if close is None or len(close) < 3:
        return pd.DataFrame(
            columns=["as_of", "quantile_switch", "switch_side", "pred_cdf"]
        )

    s = close.copy()
    s.index = pd.to_datetime(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()

    rows: list[dict[str, Any]] = []
    model = None
    alpha_prev: np.ndarray | None = None
    last_fit_i = -1

    for i, d in enumerate(s.index):
        if i < max(min_warmup_days - 1, 2):
            continue
        need_refit = (
            model is None
            or not model.fit.reliable
            or (i - last_fit_i) >= int(refit_days)
        )
        if need_refit:
            try:
                model = fit_frozen_regime_model(s, d, cfg=cfg, prev_model=model)
                alpha_prev = None  # reset forward filter under new params
                last_fit_i = i
            except Exception:  # noqa: BLE001
                model = None
                continue

        if model is None or not model.fit.reliable:
            continue

        try:
            metrics, alpha_t, _step = compute_frozen_transition_step(
                s,
                d,
                model,
                cfg=cfg,
                alpha_prev=alpha_prev,
            )
            if alpha_t is not None:
                alpha_prev = np.asarray(alpha_t, dtype=float).ravel()
        except Exception:  # noqa: BLE001
            alpha_prev = None
            continue

        pred_cdf = metrics.pred_cdf
        if pred_cdf is None or not math.isfinite(float(pred_cdf)):
            continue
        q_switch = bool(metrics.quantile_switch)
        if not q_switch:
            continue
        side = metrics.switch_side
        if side is None or (isinstance(side, float) and pd.isna(side)):
            continue
        rows.append(
            {
                "as_of": pd.Timestamp(d).normalize(),
                "quantile_switch": True,
                "switch_side": str(side),
                "pred_cdf": float(pred_cdf),
            }
        )

    return pd.DataFrame(rows)


def extend_validation_with_pre_oos_walkforward(
    oos: pd.DataFrame,
    reversal: pd.DataFrame,
    *,
    code: str,
    close: pd.Series,
    start_date: str,
    end_date: str,
    refit_days: int = 63,
    min_warmup_days: int = 260,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepend walk-forward triangles (and reversal labels) before official OOS.

    Official ``signals_oos`` for the 20260720 pack starts around 2024-06.
    Listing-to-now HTML would otherwise have K-lines but no red/green
    triangles in the earlier years. Overlap dates keep the official rows.
    """
    from decision_pack.src.regime_transition_validation import (
        compute_dual_horizon_reversal_detail,
    )

    code_u = str(code).upper()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    oos_n = oos.copy()
    if oos_n.empty or "as_of" not in oos_n.columns or "code" not in oos_n.columns:
        return oos, reversal
    oos_n["as_of"] = pd.to_datetime(oos_n["as_of"], errors="coerce").dt.normalize()
    oos_n["code"] = oos_n["code"].astype(str).str.upper()
    official = oos_n.loc[oos_n["code"] == code_u]
    if official.empty:
        return oos, reversal
    first_oos = pd.Timestamp(official["as_of"].min()).normalize()
    if first_oos <= start:
        return oos, reversal

    close_s = pd.to_numeric(close, errors="coerce")
    close_s.index = pd.to_datetime(close_s.index).normalize()
    close_s = close_s[~close_s.index.duplicated(keep="last")].sort_index()
    close_pre = close_s.loc[close_s.index < first_oos]
    if close_pre.empty:
        return oos, reversal

    tri = generate_walkforward_triangles(
        close_pre, refit_days=refit_days, min_warmup_days=min_warmup_days
    )
    if tri.empty:
        return oos, reversal
    tri = tri[(tri["as_of"] >= start) & (tri["as_of"] < first_oos) & (tri["as_of"] <= end)]
    if tri.empty:
        return oos, reversal

    name = None
    if "name" in official.columns and official["name"].notna().any():
        name = str(official["name"].dropna().iloc[0])

    extra_oos_rows: list[dict[str, Any]] = []
    extra_rev_rows: list[dict[str, Any]] = []
    for rec in tri.itertuples(index=False):
        as_of = pd.Timestamp(rec.as_of).normalize()
        extra_oos_rows.append(
            {
                "as_of": as_of,
                "code": code_u,
                "name": name,
                "quantile_switch": True,
                "switch_side": str(rec.switch_side),
                "pred_cdf": float(rec.pred_cdf),
                "switch_source": "walkforward_pre_oos",
            }
        )
        detail = compute_dual_horizon_reversal_detail(close_s, as_of)
        extra_rev_rows.append(
            {
                "as_of": as_of,
                "code": code_u,
                "label_switch_reversal": bool(detail.get("label_switch_reversal") or False),
                "turn_kind": detail.get("turn_kind") or "none",
                "jump_role": detail.get("jump_role") or "none",
                "prior_ret": detail.get("prior_ret"),
                "jump_ret": detail.get("jump_ret"),
                "post_ret": detail.get("post_ret"),
                "final_confirmed_on": detail.get("final_confirmed_on"),
                "early_label_switch_reversal": bool(
                    detail.get("early_label_switch_reversal") or False
                ),
                "early_turn_kind": detail.get("early_turn_kind") or "none",
                "early_jump_role": detail.get("early_jump_role") or "none",
                "early_prior_ret": detail.get("early_prior_ret"),
                "early_jump_ret": detail.get("early_jump_ret"),
                "early_post_ret": detail.get("early_post_ret"),
                "early_horizon": detail.get("early_horizon"),
                "early_confirmed_on": detail.get("early_confirmed_on"),
                "early_to_final_status": detail.get("early_to_final_status") or "none",
                "causal_prior_ret": detail.get("prior_ret"),
                "causal_jump_ret": detail.get("jump_ret"),
                "causal_thr_prior": detail.get("thr_prior"),
            }
        )

    extra_oos = pd.DataFrame(extra_oos_rows)
    extra_rev = pd.DataFrame(extra_rev_rows)
    oos_out = pd.concat([extra_oos, oos_n], ignore_index=True, sort=False)
    oos_out = oos_out.drop_duplicates(subset=["as_of", "code"], keep="last")
    rev_n = reversal.copy()
    if not rev_n.empty and "as_of" in rev_n.columns:
        rev_n["as_of"] = pd.to_datetime(rev_n["as_of"], errors="coerce").dt.normalize()
        if "code" in rev_n.columns:
            rev_n["code"] = rev_n["code"].astype(str).str.upper()
    rev_out = pd.concat([extra_rev, rev_n], ignore_index=True, sort=False)
    if not rev_out.empty:
        rev_out = rev_out.drop_duplicates(subset=["as_of", "code"], keep="last")
    return oos_out, rev_out


__all__ = [
    "generate_walkforward_triangles",
    "extend_validation_with_pre_oos_walkforward",
]
