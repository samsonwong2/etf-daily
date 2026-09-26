"""Walk-forward assessment of HMM predictive distribution and regime accuracy.

Distribution metrics use causal ``pred_cdf`` / ``pred_log_density`` and do not
require future bars. Outcome metrics (reversal / trend / forward vol) only use
rows with enough future horizon and report ``label_mature_end`` separately.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from decision_pack.src.regime_transition_board import CANONICAL_LABELS
from decision_pack.src.regime_transition_validation import (
    EARLY_REVERSAL_HORIZON,
    causal_bias_from_prior,
    causal_bias_from_switch_side,
    cluster_positive_events,
    compute_causal_prior_context,
    compute_dual_horizon_reversal_detail,
    compute_trend_up_label,
    match_alerts_to_events,
)

PIT_ALPHAS = (0.05, 0.10, 0.50, 0.90, 0.95)
BASELINE_LOOKBACK = 60
BASELINE_NU = 5.0


@dataclass(frozen=True)
class AssessmentVerdict:
    status: str  # PASS / WARN / FAIL
    reasons: tuple[str, ...]


def load_signals_oos(validation_dir: Path | str) -> pd.DataFrame:
    path = Path(validation_dir) / "signals_oos.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing signals_oos.csv: {path}")
    frame = pd.read_csv(path)
    if "as_of" not in frame.columns or "code" not in frame.columns:
        raise ValueError("signals_oos.csv must contain as_of and code")
    frame["as_of"] = pd.to_datetime(frame["as_of"]).dt.normalize()
    frame["code"] = frame["code"].astype(str).str.upper()
    return frame.sort_values(["code", "as_of"], kind="stable").reset_index(drop=True)


def attach_realized_returns(
    signals: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Add causal one-day log return ``r_t`` from close panel."""
    out = signals.copy()
    rets: list[float | None] = []
    for _, row in out.iterrows():
        code = str(row["code"])
        as_of = pd.Timestamp(row["as_of"]).normalize()
        if code not in panel.columns:
            rets.append(None)
            continue
        series = pd.to_numeric(panel[code], errors="coerce").dropna()
        if as_of not in series.index:
            rets.append(None)
            continue
        loc = int(series.index.get_loc(as_of))
        if loc < 1:
            rets.append(None)
            continue
        c0 = float(series.iloc[loc - 1])
        c1 = float(series.iloc[loc])
        if c0 <= 0 or c1 <= 0 or not math.isfinite(c0) or not math.isfinite(c1):
            rets.append(None)
            continue
        rets.append(float(math.log(c1 / c0)))
    out["r_t"] = rets
    return out


def ensure_pred_diagnostics(signals: pd.DataFrame) -> pd.DataFrame:
    """Fill missing diagnostic columns with NaN defaults when absent."""
    out = signals.copy()
    if "pred_cdf" not in out.columns:
        out["pred_cdf"] = np.nan
    if "pred_log_density" not in out.columns:
        out["pred_log_density"] = np.nan
    if "regime_hmm_raw" not in out.columns:
        out["regime_hmm_raw"] = out.get("regime_now")
    if "pred_mean" not in out.columns:
        out["pred_mean"] = np.nan
    if "pred_scale" not in out.columns:
        out["pred_scale"] = np.nan
    return out


def rolling_baseline_scores(
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    lookback: int = BASELINE_LOOKBACK,
    nu: float = BASELINE_NU,
) -> pd.DataFrame:
    """Causal rolling Gaussian / Student-t NLL and PIT for each signal day."""
    rows: list[dict[str, Any]] = []
    grouped = signals.groupby("code", sort=False)
    for code, grp in grouped:
        if code not in panel.columns:
            continue
        series = pd.to_numeric(panel[code], errors="coerce").dropna()
        if series.size < lookback + 2:
            continue
        log_px = np.log(series.to_numpy(dtype=float))
        rets = np.diff(log_px)
        ret_index = series.index[1:]
        ret_list = [
            (pd.Timestamp(ts).normalize(), float(r)) for ts, r in zip(ret_index, rets)
        ]
        loc_by_date = {d: i for i, (d, _) in enumerate(ret_list)}
        for _, row in grp.iterrows():
            as_of = pd.Timestamp(row["as_of"]).normalize()
            if as_of not in loc_by_date:
                continue
            i = loc_by_date[as_of]
            if i < lookback:
                continue
            window = np.array(
                [ret_list[j][1] for j in range(i - lookback, i)], dtype=float
            )
            r_t = ret_list[i][1]
            mu = float(np.mean(window))
            sigma = float(np.std(window, ddof=1))
            if not math.isfinite(sigma) or sigma <= 0:
                continue
            z = (r_t - mu) / sigma
            gauss_nll = 0.5 * math.log(2.0 * math.pi) + math.log(sigma) + 0.5 * z * z
            gauss_pit = float(stats.norm.cdf(z))
            t_pdf = float(stats.t.pdf(z, df=nu)) / sigma
            t_nll = -math.log(t_pdf) if t_pdf > 0 else float("inf")
            t_pit = float(stats.t.cdf(z, df=nu))
            rows.append(
                {
                    "code": code,
                    "as_of": as_of,
                    "baseline_gauss_nll": gauss_nll,
                    "baseline_gauss_pit": gauss_pit,
                    "baseline_t_nll": t_nll,
                    "baseline_t_pit": t_pit,
                    "baseline_mu": mu,
                    "baseline_sigma": sigma,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "code",
                "as_of",
                "baseline_gauss_nll",
                "baseline_gauss_pit",
                "baseline_t_nll",
                "baseline_t_pit",
                "baseline_mu",
                "baseline_sigma",
            ]
        )
    return pd.DataFrame(rows)


CAUSAL_BIAS_HORIZONS = (1, 3, 5, 10)


def _forward_log_return(
    series: pd.Series,
    as_of: pd.Timestamp,
    horizon: int,
) -> float | None:
    if as_of not in series.index:
        return None
    loc = int(series.index.get_loc(as_of))
    end_loc = loc + int(horizon)
    if end_loc >= len(series.index):
        return None
    c0 = float(series.iloc[loc])
    c1 = float(series.iloc[end_loc])
    if c0 <= 0 or c1 <= 0 or not math.isfinite(c0) or not math.isfinite(c1):
        return None
    return float(math.log(c1 / c0))


def attach_outcome_labels(
    signals: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    horizon: int = 10,
    lookback: int = 10,
    early_horizon: int = EARLY_REVERSAL_HORIZON,
    forward_horizons: tuple[int, ...] = CAUSAL_BIAS_HORIZONS,
) -> pd.DataFrame:
    """Attach future-dependent labels; leave null when horizon unavailable."""
    out = signals.copy()
    horizons = tuple(sorted({int(horizon), *[int(h) for h in forward_horizons]}))
    trend: list[bool | None] = []
    dual_rows: list[dict[str, Any]] = []
    causal_prior_rows: list[dict[str, Any]] = []
    fwd_by_h: dict[int, list[float | None]] = {h: [] for h in horizons}
    empty_dual = {
        "label_switch_reversal": None,
        "turn_kind": "none",
        "jump_role": "none",
        "prior_ret": None,
        "jump_ret": None,
        "post_ret": None,
        "final_confirmed_on": None,
        "early_label_switch_reversal": None,
        "early_turn_kind": "none",
        "early_jump_role": "none",
        "early_prior_ret": None,
        "early_jump_ret": None,
        "early_post_ret": None,
        "early_confirmed_on": None,
        "early_to_final_status": "immature",
        "early_horizon": int(early_horizon),
        "horizon": int(horizon),
    }
    empty_causal_prior = {
        "causal_prior_ret": None,
        "causal_jump_ret": None,
        "causal_thr_prior": None,
        "causal_prior_strong": False,
    }
    for _, row in out.iterrows():
        code = str(row["code"])
        as_of = pd.Timestamp(row["as_of"]).normalize()
        if code not in panel.columns:
            trend.append(None)
            dual_rows.append(dict(empty_dual))
            causal_prior_rows.append(dict(empty_causal_prior))
            for h in horizons:
                fwd_by_h[h].append(None)
            continue
        series = pd.to_numeric(panel[code], errors="coerce").dropna()
        tlab = compute_trend_up_label(series, as_of, horizon=horizon)
        dual = compute_dual_horizon_reversal_detail(
            series,
            as_of,
            lookback=lookback,
            early_horizon=early_horizon,
            final_horizon=horizon,
        )
        prior_ctx = compute_causal_prior_context(
            series, as_of, lookback=lookback, kappa_prior=0.35
        )
        trend.append(tlab)
        dual_rows.append(dual)
        if prior_ctx is None:
            causal_prior_rows.append(dict(empty_causal_prior))
        else:
            causal_prior_rows.append(
                {
                    "causal_prior_ret": prior_ctx["prior_ret"],
                    "causal_jump_ret": prior_ctx["jump_ret"],
                    "causal_thr_prior": prior_ctx["thr_prior"],
                    "causal_prior_strong": bool(prior_ctx["prior_strong"]),
                }
            )
        for h in horizons:
            fwd_by_h[h].append(_forward_log_return(series, as_of, h))
    out["label_trend_up_assess"] = trend
    dual_df = pd.DataFrame(dual_rows)
    for col in dual_df.columns:
        out[col] = dual_df[col].to_numpy()
    prior_df = pd.DataFrame(causal_prior_rows)
    for col in prior_df.columns:
        out[col] = prior_df[col].to_numpy()
    # Normalize string columns for downstream filters.
    out["turn_kind"] = out["turn_kind"].fillna("none").astype(str)
    out["jump_role"] = out["jump_role"].fillna("none").astype(str)
    out["early_turn_kind"] = out["early_turn_kind"].fillna("none").astype(str)
    out["early_jump_role"] = out["early_jump_role"].fillna("none").astype(str)
    for h in horizons:
        out[f"fwd_ret_{h}"] = fwd_by_h[h]
    # Keep legacy columns used by state/switch assessment.
    out["fwd_ret_h"] = out[f"fwd_ret_{int(horizon)}"]
    out["fwd_abs_ret_h"] = [
        abs(v) if v is not None and math.isfinite(float(v)) else None
        for v in out["fwd_ret_h"]
    ]
    out["label_mature"] = out["label_switch_reversal"].notna()
    out["early_label_mature"] = out["early_label_switch_reversal"].notna()
    return out


def label_mature_end(frame: pd.DataFrame) -> str | None:
    mature = frame.loc[frame.get("label_mature", False).fillna(False), "as_of"]
    if mature.empty:
        return None
    return str(pd.Timestamp(mature.max()).date())


def _summarize_early_final_migration(
    frame: pd.DataFrame,
    *,
    window: str,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "window": window,
            "n_rows": 0,
            "n_dual_mature": 0,
            "n_early_positive": 0,
            "n_final_positive": 0,
            "n_confirmed": 0,
            "n_revoked": 0,
            "n_direction_changed": 0,
            "n_final_only": 0,
            "confirm_rate_given_early": None,
            "revoke_rate_given_early": None,
            "direction_agree_rate_given_both_positive": None,
            "final_only_rate_given_final": None,
            "by_turn": {},
        }
    work = frame.copy()
    if "early_to_final_status" not in work.columns:
        return {
            "window": window,
            "n_rows": int(len(work)),
            "n_dual_mature": 0,
            "n_early_positive": 0,
            "n_final_positive": 0,
            "n_confirmed": 0,
            "n_revoked": 0,
            "n_direction_changed": 0,
            "n_final_only": 0,
            "confirm_rate_given_early": None,
            "revoke_rate_given_early": None,
            "direction_agree_rate_given_both_positive": None,
            "final_only_rate_given_final": None,
            "by_turn": {},
        }
    status = work["early_to_final_status"].astype(str)
    dual = work[
        work["early_label_switch_reversal"].notna()
        & work["label_switch_reversal"].notna()
        & ~status.isin(["immature", "early_only_immature_final"])
    ].copy()
    early_pos = dual[dual["early_label_switch_reversal"].fillna(False).astype(bool)]
    final_pos = dual[dual["label_switch_reversal"].fillna(False).astype(bool)]
    n_early = int(len(early_pos))
    n_final = int(len(final_pos))
    n_confirmed = int((dual["early_to_final_status"] == "confirmed").sum())
    n_revoked = int((dual["early_to_final_status"] == "revoked").sum())
    n_dir = int((dual["early_to_final_status"] == "direction_changed").sum())
    n_final_only = int((dual["early_to_final_status"] == "final_only").sum())
    both_pos = dual[
        dual["early_label_switch_reversal"].fillna(False).astype(bool)
        & dual["label_switch_reversal"].fillna(False).astype(bool)
    ]
    n_both = int(len(both_pos))
    n_agree = int(
        (
            both_pos["early_turn_kind"].astype(str) == both_pos["turn_kind"].astype(str)
        ).sum()
    ) if n_both else 0

    by_turn: dict[str, Any] = {}
    for turn in ("bottom_reversal", "top_reversal"):
        early_t = early_pos[early_pos["early_turn_kind"].astype(str) == turn]
        n_e = int(len(early_t))
        n_c = int((early_t["early_to_final_status"] == "confirmed").sum())
        n_r = int(
            early_t["early_to_final_status"].isin(["revoked", "direction_changed"]).sum()
        )
        by_turn[turn] = {
            "n_early_positive": n_e,
            "n_confirmed": n_c,
            "n_revoked_or_flipped": n_r,
            "confirm_rate": (n_c / n_e) if n_e else None,
            "revoke_rate": (n_r / n_e) if n_e else None,
        }
    return {
        "window": window,
        "n_rows": int(len(work)),
        "n_dual_mature": int(len(dual)),
        "n_early_positive": n_early,
        "n_final_positive": n_final,
        "n_confirmed": n_confirmed,
        "n_revoked": n_revoked,
        "n_direction_changed": n_dir,
        "n_final_only": n_final_only,
        "confirm_rate_given_early": (n_confirmed / n_early) if n_early else None,
        "revoke_rate_given_early": (
            ((n_revoked + n_dir) / n_early) if n_early else None
        ),
        "direction_agree_rate_given_both_positive": (
            (n_agree / n_both) if n_both else None
        ),
        "final_only_rate_given_final": (n_final_only / n_final) if n_final else None,
        "by_turn": by_turn,
    }


def run_early_confirmation_study(frame: pd.DataFrame) -> dict[str, Any]:
    """Evaluate early H=3 vs final H=10 reversal confirmation reliability."""
    work = frame.copy()
    if "month" not in work.columns:
        work["month"] = pd.to_datetime(work["as_of"]).dt.to_period("M").astype(str)
    if "year" not in work.columns:
        work["year"] = pd.to_datetime(work["as_of"]).dt.year
    splits = split_time_windows(work)
    holdout_months = set(str(m) for m in (splits.get("holdout_months") or []))
    if "label_mature" in work.columns:
        mature_mask = work["label_mature"].fillna(False)
    else:
        mature_mask = pd.Series(True, index=work.index)
    windows = {
        "full_mature": work[mature_mask].copy(),
        "holdout": work[
            work["month"].astype(str).isin(holdout_months) & mature_mask
        ].copy(),
    }
    # Prefer dual-mature rows for migration rates; full window still reports n_rows.
    summary_rows = [
        _summarize_early_final_migration(sub, window=name)
        for name, sub in windows.items()
    ]
    by_etf_rows: list[dict[str, Any]] = []
    if "code" in work.columns:
        for code, grp in work.groupby("code", sort=False):
            row = _summarize_early_final_migration(grp, window=str(code))
            row["group_col"] = "code"
            row["group"] = code
            by_etf_rows.append(row)
    by_year_rows: list[dict[str, Any]] = []
    if "year" in work.columns:
        for year, grp in work.groupby("year", sort=False):
            row = _summarize_early_final_migration(grp, window=str(year))
            row["group_col"] = "year"
            row["group"] = year
            by_year_rows.append(row)

    # Flatten by_turn for CSV friendliness.
    def _flatten(rows: list[dict[str, Any]]) -> pd.DataFrame:
        flat: list[dict[str, Any]] = []
        for r in rows:
            base = {k: v for k, v in r.items() if k != "by_turn"}
            flat.append(base)
            for turn, stats_ in (r.get("by_turn") or {}).items():
                item = dict(base)
                item["turn_kind"] = turn
                item.update(stats_)
                flat.append(item)
        return pd.DataFrame(flat)

    summary = {
        "holdout_months": sorted(holdout_months),
        "windows": summary_rows,
        "note": (
            "Early confirmation is provisional post-hoc labeling at T+3; "
            "it is not a trading return claim. Final labels remain T+10."
        ),
    }
    return {
        "summary": summary,
        "windows": pd.DataFrame(
            [{k: v for k, v in r.items() if k != "by_turn"} for r in summary_rows]
        ),
        "by_etf": _flatten(by_etf_rows),
        "by_year": _flatten(by_year_rows),
        "migration": _flatten(summary_rows),
    }

def pit_uniformity_summary(pit: np.ndarray) -> dict[str, Any]:
    u = np.asarray(pit, dtype=float)
    u = u[np.isfinite(u)]
    if u.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "ks_stat": None,
            "ks_pvalue": None,
            "cvm_stat": None,
            "hist_counts": [],
            "hist_edges": [],
        }
    ks_stat, ks_p = stats.kstest(u, "uniform")
    u_sorted = np.sort(u)
    n = u_sorted.size
    i = np.arange(1, n + 1)
    cvm = float(
        (1.0 / (12.0 * n)) + np.sum(((2 * i - 1) / (2.0 * n) - u_sorted) ** 2)
    )
    hist_counts, hist_edges = np.histogram(u, bins=10, range=(0.0, 1.0))
    return {
        "n": int(n),
        "mean": float(np.mean(u)),
        "std": float(np.std(u, ddof=1)) if n > 1 else 0.0,
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_p),
        "cvm_stat": cvm,
        "hist_counts": [int(x) for x in hist_counts.tolist()],
        "hist_edges": [float(x) for x in hist_edges.tolist()],
    }


def coverage_summary(
    pit: np.ndarray, alphas: tuple[float, ...] = PIT_ALPHAS
) -> dict[str, Any]:
    u = np.asarray(pit, dtype=float)
    u = u[np.isfinite(u)]
    out: dict[str, Any] = {"n": int(u.size)}
    for a in alphas:
        rate = float((u <= a).mean()) if u.size else None
        out[f"coverage_{a:g}"] = rate
        out[f"coverage_err_{a:g}"] = abs(rate - a) if rate is not None else None
    if u.size:
        tail = float(((u <= 0.10) | (u >= 0.90)).mean())
        out["bilateral_tail_rate"] = tail
        out["bilateral_tail_err"] = abs(tail - 0.20)
    else:
        out["bilateral_tail_rate"] = None
        out["bilateral_tail_err"] = None
    return out


def pit_autocorr_ljung_box(pit: np.ndarray, lags: int = 10) -> dict[str, Any]:
    u = np.asarray(pit, dtype=float)
    u = u[np.isfinite(u)]
    if u.size < lags + 5:
        return {"n": int(u.size), "lb_stat": None, "lb_pvalue": None, "acf1": None}
    centered = u - np.mean(u)
    acf1 = (
        float(np.corrcoef(centered[:-1], centered[1:])[0, 1]) if u.size > 2 else None
    )
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        lb = acorr_ljungbox(centered, lags=[lags], return_df=True)
        return {
            "n": int(u.size),
            "lb_stat": float(lb["lb_stat"].iloc[0]),
            "lb_pvalue": float(lb["lb_pvalue"].iloc[0]),
            "acf1": acf1,
        }
    except Exception:
        return {
            "n": int(u.size),
            "lb_stat": None,
            "lb_pvalue": None,
            "acf1": acf1,
        }


def block_bootstrap_mean_ci(
    values: np.ndarray,
    *,
    block_size: int = 20,
    n_boot: int = 400,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float | None]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    if x.size < block_size * 2:
        mean = float(np.mean(x))
        se = float(np.std(x, ddof=1) / math.sqrt(x.size)) if x.size > 1 else 0.0
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        return {
            "mean": mean,
            "ci_low": mean - z * se,
            "ci_high": mean + z * se,
            "n": int(x.size),
        }
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(x.size / block_size))
    means: list[float] = []
    max_start = x.size - block_size
    for _ in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sample = np.concatenate([x[s : s + block_size] for s in starts])[: x.size]
        means.append(float(np.mean(sample)))
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return {
        "mean": float(np.mean(x)),
        "ci_low": lo,
        "ci_high": hi,
        "n": int(x.size),
    }


def distribution_assessment(
    frame: pd.DataFrame,
    *,
    pit_col: str = "pred_cdf",
    nll_col: str = "pred_nll",
) -> dict[str, Any]:
    pit = pd.to_numeric(frame.get(pit_col), errors="coerce").to_numpy(dtype=float)
    uni = pit_uniformity_summary(pit)
    cov = coverage_summary(pit)
    lb = pit_autocorr_ljung_box(pit)
    nll = pd.to_numeric(frame.get(nll_col), errors="coerce").to_numpy(dtype=float)
    nll_ci = block_bootstrap_mean_ci(nll)
    return {
        "uniformity": uni,
        "coverage": cov,
        "ljung_box": lb,
        "nll": nll_ci,
        "mean_nll": nll_ci["mean"],
    }


def baseline_comparison(frame: pd.DataFrame) -> dict[str, Any]:
    hmm = pd.to_numeric(frame.get("pred_nll"), errors="coerce")
    g = pd.to_numeric(frame.get("baseline_gauss_nll"), errors="coerce")
    t = pd.to_numeric(frame.get("baseline_t_nll"), errors="coerce")
    mask = hmm.notna() & g.notna() & t.notna()
    if not mask.any():
        return {
            "n": 0,
            "delta_vs_gauss": None,
            "delta_vs_t": None,
            "gauss_ci": {"mean": None, "ci_low": None, "ci_high": None, "n": 0},
            "t_ci": {"mean": None, "ci_low": None, "ci_high": None, "n": 0},
        }
    d_g = (g[mask] - hmm[mask]).to_numpy(dtype=float)
    d_t = (t[mask] - hmm[mask]).to_numpy(dtype=float)
    return {
        "n": int(mask.sum()),
        "delta_vs_gauss": float(np.mean(d_g)),
        "delta_vs_t": float(np.mean(d_t)),
        "gauss_ci": block_bootstrap_mean_ci(d_g),
        "t_ci": block_bootstrap_mean_ci(d_t),
    }


def state_semantics_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Unsupervised + forward-outcome semantics for raw HMM states."""
    work = frame.copy()
    state_col = "regime_hmm_raw" if "regime_hmm_raw" in work.columns else "regime_now"
    work[state_col] = work[state_col].fillna("—").astype(str)
    mature = work[work.get("label_mature", False).fillna(False)].copy()
    occupancy = (
        work[state_col].value_counts(normalize=True).to_dict() if not work.empty else {}
    )
    dwell_stats: dict[str, float] = {}
    trans_rate = None
    if not work.empty:
        changes = 0
        runs: dict[str, list[int]] = {lab: [] for lab in CANONICAL_LABELS}
        for _, grp in work.groupby("code", sort=False):
            states = grp.sort_values("as_of")[state_col].tolist()
            if not states:
                continue
            run_lab = states[0]
            run_len = 1
            for s in states[1:]:
                if s == run_lab:
                    run_len += 1
                else:
                    changes += 1
                    if run_lab in runs:
                        runs[run_lab].append(run_len)
                    run_lab = s
                    run_len = 1
            if run_lab in runs:
                runs[run_lab].append(run_len)
        total_steps = max(len(work) - work["code"].nunique(), 1)
        trans_rate = changes / total_steps
        for lab, lengths in runs.items():
            dwell_stats[lab] = float(np.mean(lengths)) if lengths else 0.0

    forward: dict[str, Any] = {}
    if not mature.empty and "fwd_abs_ret_h" in mature.columns:
        for lab in CANONICAL_LABELS:
            sub = mature[mature[state_col] == lab]
            if sub.empty:
                forward[lab] = {"n": 0, "mean_abs_h": None, "mean_ret_h": None}
                continue
            forward[lab] = {
                "n": int(len(sub)),
                "mean_abs_h": float(
                    pd.to_numeric(sub["fwd_abs_ret_h"], errors="coerce").mean()
                ),
                "mean_ret_h": float(
                    pd.to_numeric(sub["fwd_ret_h"], errors="coerce").mean()
                ),
            }
    order_ok = None
    if forward:
        calm_abs = (forward.get("calm") or {}).get("mean_abs_h")
        vol_abs = (forward.get("volatile_reversal") or {}).get("mean_abs_h")
        calm_ret = (forward.get("calm") or {}).get("mean_ret_h")
        bear_ret = (forward.get("bear_grind") or {}).get("mean_ret_h")
        checks = []
        if calm_abs is not None and vol_abs is not None:
            checks.append(vol_abs >= calm_abs)
        if calm_ret is not None and bear_ret is not None:
            checks.append(bear_ret <= calm_ret)
        order_ok = all(checks) if checks else None

    ambiguous_rate = (
        float(work["label_ambiguous"].fillna(False).astype(bool).mean())
        if "label_ambiguous" in work.columns
        else None
    )
    separated_rate = (
        float(work["regime_separated"].fillna(False).astype(bool).mean())
        if "regime_separated" in work.columns
        else None
    )
    return {
        "state_col": state_col,
        "occupancy": {str(k): float(v) for k, v in occupancy.items()},
        "mean_dwell": dwell_stats,
        "transition_rate": trans_rate,
        "forward_by_state": forward,
        "semantic_order_ok": order_ok,
        "ambiguous_rate": ambiguous_rate,
        "separated_rate": separated_rate,
    }


def switch_reversal_assessment(
    frame: pd.DataFrame, *, horizon: int = 10
) -> dict[str, Any]:
    mature = frame[frame.get("label_mature", False).fillna(False)].copy()
    if mature.empty or "quantile_switch" not in mature.columns:
        return {
            "n_mature": 0,
            "n_alerts": 0,
            "n_events": 0,
            "precision": None,
            "recall": None,
            "event_match": {},
        }
    mature["quantile_switch"] = mature["quantile_switch"].map(
        lambda v: False if pd.isna(v) else bool(v)
    )
    mature["label_switch_reversal"] = mature["label_switch_reversal"].map(
        lambda v: False if pd.isna(v) else bool(v)
    )
    alerts = mature[mature["quantile_switch"]].copy()
    label_frame = mature[["code", "as_of", "label_switch_reversal"]].rename(
        columns={"label_switch_reversal": "label_trend_up"}
    )
    events = cluster_positive_events(label_frame)
    if not alerts.empty:
        alerts = alerts.copy()
        if "pred_cdf" in alerts.columns:
            alerts["switch_score"] = (alerts["pred_cdf"] - 0.5).abs().fillna(0.0)
        else:
            alerts["switch_score"] = 1.0
        matched = match_alerts_to_events(
            alerts,
            events,
            lead=3,
            horizon=horizon,
            score_col="switch_score",
        )
        n_matched = (
            int(matched["matched_event_onset"].notna().sum()) if not matched.empty else 0
        )
        n_fp = (
            int(matched["is_fp"].fillna(False).astype(bool).sum())
            if not matched.empty
            else 0
        )
        n_dup = (
            int(matched["is_duplicate"].fillna(False).astype(bool).sum())
            if not matched.empty
            else 0
        )
        delay = (
            pd.to_numeric(matched["detection_delay"], errors="coerce").dropna()
            if not matched.empty
            else pd.Series(dtype=float)
        )
    else:
        matched = pd.DataFrame()
        n_matched = 0
        n_fp = 0
        n_dup = 0
        delay = pd.Series(dtype=float)

    n_alerts = int(len(alerts))
    n_events = int(len(events))
    precision = (n_matched / n_alerts) if n_alerts else None
    recall = (n_matched / n_events) if n_events else None
    day_prec = (
        float(mature.loc[mature["quantile_switch"], "label_switch_reversal"].mean())
        if n_alerts
        else None
    )
    day_rec = (
        float(mature.loc[mature["label_switch_reversal"], "quantile_switch"].mean())
        if mature["label_switch_reversal"].any()
        else None
    )
    return {
        "n_mature": int(len(mature)),
        "n_alerts": n_alerts,
        "n_events": n_events,
        "precision": precision,
        "recall": recall,
        "day_precision": day_prec,
        "day_recall": day_rec,
        "event_match": {
            "n_matched": n_matched,
            "n_fp": n_fp,
            "n_duplicate": n_dup,
            "mean_detection_delay": float(delay.mean()) if not delay.empty else None,
        },
        "matched_table": matched,
    }


def default_qstar_grid(
    start: float = 0.80,
    stop: float = 0.99,
    step: float = 0.01,
) -> tuple[float, ...]:
    """Inclusive grid from start to stop (rounded to 2 decimals)."""
    if step <= 0 or stop < start:
        return ()
    n = int(round((stop - start) / step)) + 1
    return tuple(round(start + i * step, 2) for i in range(n))


def switch_mask_from_cdf(
    pred_cdf: pd.Series,
    *,
    q_up: float | None = None,
    q_down: float | None = None,
    q_star: float | None = None,
    side: str = "both",
) -> pd.Series:
    """Boolean alert mask from predictive CDF thresholds.

    ``side``: ``both`` | ``up`` | ``down``.
    Unified ``q_star`` sets ``q_up=q_star`` and ``q_down=1-q_star`` when
    explicit up/down thresholds are omitted.
    """
    cdf = pd.to_numeric(pred_cdf, errors="coerce")
    if q_star is not None:
        if q_up is None:
            q_up = float(q_star)
        if q_down is None:
            q_down = round(1.0 - float(q_star), 10)
    if q_up is None:
        q_up = 0.90
    if q_down is None:
        q_down = 0.10
    q_up = float(q_up)
    q_down = float(q_down)
    up = cdf >= q_up
    down = cdf <= q_down
    side_l = str(side).lower()
    if side_l == "up":
        return up.fillna(False)
    if side_l == "down":
        return down.fillna(False)
    return (up | down).fillna(False)


def _bool_col(series: pd.Series) -> pd.Series:
    return series.map(lambda v: False if pd.isna(v) else bool(v))


def _event_metrics_for_switch(
    mature: pd.DataFrame,
    switch_mask: pd.Series,
    *,
    horizon: int = 10,
    lead: int = 3,
    events: list | None = None,
) -> dict[str, Any]:
    """Event-level precision/recall/F1 for a fixed alert mask on mature rows."""
    if mature.empty:
        return {
            "n_alerts": 0,
            "n_events": 0,
            "n_matched": 0,
            "n_fp": 0,
            "n_duplicate": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "day_precision": None,
            "day_recall": None,
            "duplicate_rate": None,
            "fp_rate": None,
            "mean_detection_delay": None,
            "alerts_per_etf_year": None,
            "alert_rate": 0.0,
        }
    work = mature
    mask = switch_mask.reindex(work.index).fillna(False).astype(bool)
    if "label_switch_reversal" not in work.columns:
        labels = pd.Series(False, index=work.index)
    else:
        labels = _bool_col(work["label_switch_reversal"])

    alerts = work.loc[mask].copy()
    if events is None:
        label_frame = pd.DataFrame(
            {
                "code": work["code"],
                "as_of": work["as_of"],
                "label_trend_up": labels,
            }
        )
        events = cluster_positive_events(label_frame)
    n_alerts = int(len(alerts))
    n_events = int(len(events))
    alert_rate = float(mask.mean()) if len(work) else 0.0

    if n_alerts == 0:
        return {
            "n_alerts": 0,
            "n_events": n_events,
            "n_matched": 0,
            "n_fp": 0,
            "n_duplicate": 0,
            "precision": None,
            "recall": 0.0 if n_events else None,
            "f1": None,
            "day_precision": None,
            "day_recall": (0.0 if bool(labels.any()) else None),
            "duplicate_rate": None,
            "fp_rate": None,
            "mean_detection_delay": None,
            "alerts_per_etf_year": 0.0,
            "alert_rate": alert_rate,
        }

    if "pred_cdf" in alerts.columns:
        alerts["switch_score"] = (alerts["pred_cdf"] - 0.5).abs().fillna(0.0)
    else:
        alerts["switch_score"] = 1.0
    matched = match_alerts_to_events(
        alerts,
        events,
        lead=lead,
        horizon=horizon,
        score_col="switch_score",
    )
    n_matched = (
        int(matched["matched_event_onset"].notna().sum()) if not matched.empty else 0
    )
    n_fp = (
        int(matched["is_fp"].fillna(False).astype(bool).sum())
        if not matched.empty
        else 0
    )
    n_dup = (
        int(matched["is_duplicate"].fillna(False).astype(bool).sum())
        if not matched.empty
        else 0
    )
    delay = (
        pd.to_numeric(matched["detection_delay"], errors="coerce").dropna()
        if not matched.empty
        else pd.Series(dtype=float)
    )
    precision = (n_matched / n_alerts) if n_alerts else None
    recall = (n_matched / n_events) if n_events else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)

    day_prec = float(labels.loc[mask].mean()) if n_alerts else None
    day_rec = float(mask.loc[labels].mean()) if bool(labels.any()) else None

    n_codes = max(int(work["code"].nunique()), 1)
    as_of = pd.to_datetime(work["as_of"])
    span_days = max((as_of.max() - as_of.min()).days, 1)
    years = max(span_days / 365.25, 1.0 / 12.0)
    alerts_per_etf_year = n_alerts / (n_codes * years)

    return {
        "n_alerts": n_alerts,
        "n_events": n_events,
        "n_matched": n_matched,
        "n_fp": n_fp,
        "n_duplicate": n_dup,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "day_precision": day_prec,
        "day_recall": day_rec,
        "duplicate_rate": (n_dup / n_alerts) if n_alerts else None,
        "fp_rate": (n_fp / n_alerts) if n_alerts else None,
        "mean_detection_delay": float(delay.mean()) if not delay.empty else None,
        "alerts_per_etf_year": float(alerts_per_etf_year),
        "alert_rate": alert_rate,
    }


def month_block_bootstrap_metric_ci(
    mature: pd.DataFrame,
    *,
    q_star: float,
    side: str = "both",
    metric: str = "f1",
    horizon: int = 10,
    n_boot: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float | None]:
    """Block-bootstrap CI by resampling calendar months."""
    if mature.empty or "month" not in mature.columns:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    months = sorted(mature["month"].astype(str).unique().tolist())
    if not months:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}

    def _metric_on(sub: pd.DataFrame) -> float | None:
        mask = switch_mask_from_cdf(sub["pred_cdf"], q_star=q_star, side=side)
        stats_ = _event_metrics_for_switch(sub, mask, horizon=horizon)
        val = stats_.get(metric)
        return float(val) if val is not None and math.isfinite(float(val)) else None

    point = _metric_on(mature)
    if len(months) < 3:
        return {
            "mean": point,
            "ci_low": point,
            "ci_high": point,
            "n": int(len(months)),
        }

    rng = np.random.default_rng(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        chosen = rng.choice(months, size=len(months), replace=True)
        sub = mature[mature["month"].astype(str).isin(set(chosen))]
        if sub.empty:
            continue
        m = _metric_on(sub)
        if m is not None:
            boots.append(m)
    if not boots:
        return {"mean": point, "ci_low": None, "ci_high": None, "n": int(len(months))}
    return {
        "mean": point,
        "ci_low": float(np.quantile(boots, alpha / 2.0)),
        "ci_high": float(np.quantile(boots, 1.0 - alpha / 2.0)),
        "n": int(len(months)),
    }


def qstar_sensitivity(
    frame: pd.DataFrame,
    *,
    q_values: tuple[float, ...] | None = None,
    sides: tuple[str, ...] = ("both", "up", "down"),
    horizon: int = 10,
    lead: int = 3,
    include_bootstrap: bool = False,
    bootstrap_n: int = 100,
    bootstrap_seed: int = 42,
) -> pd.DataFrame:
    """Sweep predictive-quantile thresholds with day- and event-level metrics.

    Default grid: q* = 0.80 … 0.99 step 0.01. Rows include ``side`` =
    both / up / down so asymmetric thresholds can be compared later.
    """
    mature = frame[frame.get("label_mature", False).fillna(False)].copy()
    if mature.empty or "pred_cdf" not in mature.columns:
        return pd.DataFrame()
    if "month" not in mature.columns:
        mature["month"] = pd.to_datetime(mature["as_of"]).dt.to_period("M").astype(str)
    qs = q_values if q_values is not None else default_qstar_grid()
    rows: list[dict[str, Any]] = []
    # Cluster reversal events once; thresholds only change the alert mask.
    label_frame = mature[["code", "as_of", "label_switch_reversal"]].copy()
    label_frame["label_switch_reversal"] = _bool_col(label_frame["label_switch_reversal"])
    label_frame = label_frame.rename(columns={"label_switch_reversal": "label_trend_up"})
    events = cluster_positive_events(label_frame)
    for q in qs:
        for side in sides:
            mask = switch_mask_from_cdf(mature["pred_cdf"], q_star=float(q), side=side)
            stats_ = _event_metrics_for_switch(
                mature, mask, horizon=horizon, lead=lead, events=events
            )
            row: dict[str, Any] = {
                "q_star": float(q),
                "q_up": float(q),
                "q_down": round(1.0 - float(q), 10),
                "side": side,
                **stats_,
            }
            if include_bootstrap and side == "both":
                f1_ci = month_block_bootstrap_metric_ci(
                    mature,
                    q_star=float(q),
                    side=side,
                    metric="f1",
                    horizon=horizon,
                    n_boot=bootstrap_n,
                    seed=bootstrap_seed,
                )
                prec_ci = month_block_bootstrap_metric_ci(
                    mature,
                    q_star=float(q),
                    side=side,
                    metric="precision",
                    horizon=horizon,
                    n_boot=bootstrap_n,
                    seed=bootstrap_seed + 1,
                )
                row["f1_ci_low"] = f1_ci.get("ci_low")
                row["f1_ci_high"] = f1_ci.get("ci_high")
                row["precision_ci_low"] = prec_ci.get("ci_low")
                row["precision_ci_high"] = prec_ci.get("ci_high")
            rows.append(row)
    return pd.DataFrame(rows)


def split_time_windows(
    frame: pd.DataFrame,
    *,
    select_frac: float = 0.35,
    holdout_frac: float = 0.25,
    min_select_months: int = 3,
    min_holdout_months: int = 2,
) -> dict[str, Any]:
    """Split mature months into candidate-eval / select / holdout by calendar order.

    Early months → candidate evaluation pool (also used for selection metrics),
    middle → select, latest → holdout. Selection must not read holdout.
    """
    mature = frame[frame.get("label_mature", False).fillna(False)].copy()
    if mature.empty:
        return {
            "months": [],
            "candidate_months": [],
            "select_months": [],
            "holdout_months": [],
            "candidate": mature,
            "select": mature,
            "holdout": mature,
        }
    if "month" not in mature.columns:
        mature["month"] = pd.to_datetime(mature["as_of"]).dt.to_period("M").astype(str)
    months = sorted(mature["month"].astype(str).unique().tolist())
    n = len(months)
    n_hold = max(min_holdout_months, int(round(n * holdout_frac)))
    n_sel = max(min_select_months, int(round(n * select_frac)))
    # shrink if total months too few
    if n_hold + n_sel >= n:
        n_hold = max(1, n // 4)
        n_sel = max(1, n // 3)
        if n_hold + n_sel >= n:
            n_hold = 1 if n >= 2 else 0
            n_sel = max(1, n - n_hold - 1) if n >= 3 else max(0, n - n_hold)
    holdout_months = months[-n_hold:] if n_hold else []
    select_months = months[-(n_hold + n_sel) : -n_hold] if n_sel else []
    if not select_months and months:
        # fall back: use all non-holdout as select
        select_months = [m for m in months if m not in holdout_months]
    candidate_months = [m for m in months if m not in holdout_months]
    # selection window is the middle slice; candidate pool = all pre-holdout
    return {
        "months": months,
        "candidate_months": candidate_months,
        "select_months": select_months,
        "holdout_months": holdout_months,
        "candidate": mature[mature["month"].astype(str).isin(candidate_months)].copy(),
        "select": mature[mature["month"].astype(str).isin(select_months)].copy(),
        "holdout": mature[mature["month"].astype(str).isin(holdout_months)].copy(),
    }


def select_qstar_threshold(
    select_frame: pd.DataFrame,
    *,
    q_values: tuple[float, ...] | None = None,
    horizon: int = 10,
    min_alerts: int = 15,
    max_fp_rate: float = 0.85,
    max_alerts_per_etf_year: float = 40.0,
    baseline_q: float = 0.90,
) -> dict[str, Any]:
    """Pick q* on the selection window only (no holdout leakage).

    Rule: among feasible rows (min alerts, fp_rate, alert density), maximize
    event F1; ties → higher precision → lower duplicate_rate → higher (more
    conservative) q*. Also evaluates asymmetric up/down vs unified baseline.
    """
    qs = q_values if q_values is not None else default_qstar_grid()
    sens = qstar_sensitivity(
        select_frame,
        q_values=qs,
        sides=("both", "up", "down"),
        horizon=horizon,
        include_bootstrap=False,
    )
    if sens.empty:
        return {
            "recommended_mode": "unified",
            "q_star": baseline_q,
            "q_up": baseline_q,
            "q_down": 1.0 - baseline_q,
            "reason": "empty_select_frame",
            "baseline_q": baseline_q,
            "selection_table": sens,
            "best_unified": None,
            "best_asymmetric": None,
            "use_asymmetric": False,
        }

    both = sens[sens["side"] == "both"].copy()

    def _feasible(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["_ok"] = (
            (out["n_alerts"].fillna(0) >= min_alerts)
            & (out["fp_rate"].fillna(1.0) <= max_fp_rate)
            & (out["alerts_per_etf_year"].fillna(999.0) <= max_alerts_per_etf_year)
            & out["f1"].notna()
        )
        return out

    both_f = _feasible(both)
    feasible = both_f[both_f["_ok"]]
    if feasible.empty:
        # relax: pick max F1 among any both-side rows with alerts
        cand = both_f[both_f["n_alerts"].fillna(0) > 0].copy()
        if cand.empty:
            return {
                "recommended_mode": "unified",
                "q_star": baseline_q,
                "q_up": baseline_q,
                "q_down": 1.0 - baseline_q,
                "reason": "no_feasible_threshold_keep_baseline",
                "baseline_q": baseline_q,
                "selection_table": sens,
                "best_unified": None,
                "best_asymmetric": None,
                "use_asymmetric": False,
            }
        feasible = cand

    ranked = feasible.sort_values(
        by=["f1", "precision", "duplicate_rate", "q_star"],
        ascending=[False, False, True, False],
        kind="stable",
    )
    best_u = ranked.iloc[0]
    best_unified = {
        "q_star": float(best_u["q_star"]),
        "f1": float(best_u["f1"]) if pd.notna(best_u["f1"]) else None,
        "precision": float(best_u["precision"]) if pd.notna(best_u["precision"]) else None,
        "recall": float(best_u["recall"]) if pd.notna(best_u["recall"]) else None,
        "n_alerts": int(best_u["n_alerts"]),
        "fp_rate": float(best_u["fp_rate"]) if pd.notna(best_u["fp_rate"]) else None,
        "duplicate_rate": (
            float(best_u["duplicate_rate"]) if pd.notna(best_u["duplicate_rate"]) else None
        ),
    }

    # Asymmetric: independently pick best up and down q on selection window
    up = _feasible(sens[sens["side"] == "up"])
    down = _feasible(sens[sens["side"] == "down"])
    best_asymmetric = None
    use_asymmetric = False
    if not up[up["_ok"]].empty and not down[down["_ok"]].empty:
        bu = up[up["_ok"]].sort_values(
            by=["f1", "precision", "q_star"],
            ascending=[False, False, False],
            kind="stable",
        ).iloc[0]
        bd = down[down["_ok"]].sort_values(
            by=["f1", "precision", "q_star"],
            ascending=[False, False, False],
            kind="stable",
        ).iloc[0]
        q_up = float(bu["q_star"])
        q_down_star = float(bd["q_star"])  # this is the |tail| quantile; down thr=1-q
        q_down = 1.0 - q_down_star
        # evaluate joint asymmetric mask on select frame
        mask = switch_mask_from_cdf(
            select_frame["pred_cdf"], q_up=q_up, q_down=q_down, side="both"
        )
        joint = _event_metrics_for_switch(select_frame, mask, horizon=horizon)
        best_asymmetric = {
            "q_up": q_up,
            "q_down": q_down,
            "q_down_star": q_down_star,
            "f1": joint.get("f1"),
            "precision": joint.get("precision"),
            "recall": joint.get("recall"),
            "n_alerts": joint.get("n_alerts"),
            "fp_rate": joint.get("fp_rate"),
            "duplicate_rate": joint.get("duplicate_rate"),
            "up_row_f1": float(bu["f1"]) if pd.notna(bu["f1"]) else None,
            "down_row_f1": float(bd["f1"]) if pd.notna(bd["f1"]) else None,
        }
        u_f1 = best_unified["f1"] or 0.0
        a_f1 = best_asymmetric["f1"] or 0.0
        # require material and stable lift vs unified on select window
        if (
            a_f1 >= u_f1 + 0.02
            and (best_asymmetric.get("n_alerts") or 0) >= min_alerts
            and (best_asymmetric.get("fp_rate") or 1.0) <= max_fp_rate
            and (q_up != best_unified["q_star"] or q_down_star != best_unified["q_star"])
        ):
            use_asymmetric = True

    if use_asymmetric and best_asymmetric is not None:
        return {
            "recommended_mode": "asymmetric",
            "q_star": None,
            "q_up": best_asymmetric["q_up"],
            "q_down": best_asymmetric["q_down"],
            "reason": "asymmetric_f1_lift_on_select",
            "baseline_q": baseline_q,
            "selection_table": sens,
            "best_unified": best_unified,
            "best_asymmetric": best_asymmetric,
            "use_asymmetric": True,
        }
    return {
        "recommended_mode": "unified",
        "q_star": best_unified["q_star"],
        "q_up": best_unified["q_star"],
        "q_down": 1.0 - best_unified["q_star"],
        "reason": "max_event_f1_on_select",
        "baseline_q": baseline_q,
        "selection_table": sens,
        "best_unified": best_unified,
        "best_asymmetric": best_asymmetric,
        "use_asymmetric": False,
    }


def evaluate_threshold_on_frame(
    frame: pd.DataFrame,
    *,
    q_star: float | None = None,
    q_up: float | None = None,
    q_down: float | None = None,
    side: str = "both",
    horizon: int = 10,
    label: str = "",
) -> dict[str, Any]:
    """Evaluate one threshold configuration on a (usually mature) frame."""
    mature = frame[frame.get("label_mature", False).fillna(False)].copy()
    if mature.empty:
        mature = frame.copy()
    mask = switch_mask_from_cdf(
        mature["pred_cdf"],
        q_star=q_star,
        q_up=q_up,
        q_down=q_down,
        side=side,
    )
    stats_ = _event_metrics_for_switch(mature, mask, horizon=horizon)
    return {
        "label": label,
        "q_star": q_star,
        "q_up": q_up if q_up is not None else q_star,
        "q_down": q_down if q_down is not None else (1.0 - q_star if q_star else None),
        "side": side,
        "n_rows": int(len(mature)),
        **stats_,
    }


def run_qstar_threshold_study(
    frame: pd.DataFrame,
    *,
    horizon: int = 10,
    baseline_q: float = 0.90,
    q_values: tuple[float, ...] | None = None,
    include_ci: bool = False,
) -> dict[str, Any]:
    """Full walk-forward threshold study: sensitivity + select + holdout."""
    qs = q_values if q_values is not None else default_qstar_grid()
    splits = split_time_windows(frame)
    select_frame = splits["select"]
    holdout_frame = splits["holdout"]
    candidate_frame = splits["candidate"]

    # Full mature sensitivity: both tails for reporting; up/down for asymmetry checks
    sens_all = qstar_sensitivity(
        frame,
        q_values=qs,
        sides=("both", "up", "down"),
        horizon=horizon,
        include_bootstrap=False,
    )
    if include_ci:
        sens_boot = qstar_sensitivity(
            frame,
            q_values=(0.85, 0.90, 0.95),
            sides=("both",),
            horizon=horizon,
            include_bootstrap=True,
            bootstrap_n=30,
        )
        if not sens_all.empty and not sens_boot.empty:
            merge_cols = [
                "q_star",
                "side",
                "f1_ci_low",
                "f1_ci_high",
                "precision_ci_low",
                "precision_ci_high",
            ]
            sens_all = sens_all.merge(
                sens_boot[merge_cols],
                on=["q_star", "side"],
                how="left",
            )

    selection = select_qstar_threshold(
        select_frame if not select_frame.empty else candidate_frame,
        q_values=qs,
        horizon=horizon,
        baseline_q=baseline_q,
    )
    # Drop large table from JSON-friendly copy
    selection_meta = {
        k: v for k, v in selection.items() if k != "selection_table"
    }
    select_sens = selection.get("selection_table", pd.DataFrame())

    baseline_holdout = evaluate_threshold_on_frame(
        holdout_frame,
        q_star=baseline_q,
        side="both",
        horizon=horizon,
        label="baseline_q90_holdout",
    )
    if selection.get("use_asymmetric"):
        recommended_holdout = evaluate_threshold_on_frame(
            holdout_frame,
            q_up=selection["q_up"],
            q_down=selection["q_down"],
            side="both",
            horizon=horizon,
            label="recommended_asymmetric_holdout",
        )
    else:
        recommended_holdout = evaluate_threshold_on_frame(
            holdout_frame,
            q_star=selection.get("q_star") or baseline_q,
            side="both",
            horizon=horizon,
            label="recommended_unified_holdout",
        )

    baseline_select = evaluate_threshold_on_frame(
        select_frame if not select_frame.empty else candidate_frame,
        q_star=baseline_q,
        side="both",
        horizon=horizon,
        label="baseline_q90_select",
    )
    if selection.get("use_asymmetric"):
        recommended_select = evaluate_threshold_on_frame(
            select_frame if not select_frame.empty else candidate_frame,
            q_up=selection["q_up"],
            q_down=selection["q_down"],
            side="both",
            horizon=horizon,
            label="recommended_asymmetric_select",
        )
    else:
        recommended_select = evaluate_threshold_on_frame(
            select_frame if not select_frame.empty else candidate_frame,
            q_star=selection.get("q_star") or baseline_q,
            side="both",
            horizon=horizon,
            label="recommended_unified_select",
        )

    # Recommend config change only if holdout improves F1 vs baseline
    b_f1 = baseline_holdout.get("f1")
    r_f1 = recommended_holdout.get("f1")
    holdout_improves = (
        b_f1 is not None
        and r_f1 is not None
        and r_f1 >= b_f1 + 0.02
        and (recommended_holdout.get("n_alerts") or 0) >= 5
    )
    keep_baseline = not holdout_improves
    recommendation = {
        "keep_production_q90": keep_baseline,
        "production_q_star": baseline_q if keep_baseline else selection.get("q_star"),
        "production_q_up": (
            baseline_q if keep_baseline else selection.get("q_up")
        ),
        "production_q_down": (
            round(1.0 - baseline_q, 10)
            if keep_baseline
            else (
                round(float(selection.get("q_down")), 10)
                if selection.get("q_down") is not None
                else None
            )
        ),
        "mode": (
            "keep_baseline"
            if keep_baseline
            else selection.get("recommended_mode", "unified")
        ),
        "holdout_f1_baseline": b_f1,
        "holdout_f1_recommended": r_f1,
        "holdout_improves": holdout_improves,
        "note": (
            "Holdout F1 did not improve enough vs q*=0.90; keep production 0.90."
            if keep_baseline
            else "Holdout F1 improved vs baseline; consider updating switch_quantile_q."
        ),
    }

    comparison = pd.DataFrame(
        [baseline_select, recommended_select, baseline_holdout, recommended_holdout]
    )
    return {
        "splits": {
            "months": splits["months"],
            "candidate_months": splits["candidate_months"],
            "select_months": splits["select_months"],
            "holdout_months": splits["holdout_months"],
        },
        "sensitivity_all": sens_all,
        "sensitivity_select": select_sens,
        "selection": selection_meta,
        "comparison": comparison,
        "recommendation": recommendation,
        "baseline_q": baseline_q,
    }


def per_etf_distribution_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for code, grp in frame.groupby("code", sort=True):
        pit = pd.to_numeric(grp.get("pred_cdf"), errors="coerce").to_numpy(dtype=float)
        uni = pit_uniformity_summary(pit)
        cov = coverage_summary(pit)
        nll = pd.to_numeric(grp.get("pred_nll"), errors="coerce")
        base = baseline_comparison(grp)
        rows.append(
            {
                "code": code,
                "n": uni["n"],
                "pit_mean": uni["mean"],
                "ks_stat": uni["ks_stat"],
                "cvm_stat": uni["cvm_stat"],
                "bilateral_tail_rate": cov.get("bilateral_tail_rate"),
                "bilateral_tail_err": cov.get("bilateral_tail_err"),
                "mean_nll": float(nll.mean()) if nll.notna().any() else None,
                "delta_nll_vs_gauss": base.get("delta_vs_gauss"),
                "delta_nll_vs_t": base.get("delta_vs_t"),
            }
        )
    return pd.DataFrame(rows)


def leave_one_group_robustness(
    frame: pd.DataFrame,
    *,
    group_col: str,
) -> pd.DataFrame:
    if group_col not in frame.columns or frame.empty:
        return pd.DataFrame()
    groups = sorted(frame[group_col].dropna().unique().tolist())
    rows: list[dict[str, Any]] = []
    for g in groups:
        holdout = frame[frame[group_col] != g]
        if holdout.empty:
            continue
        pit = pd.to_numeric(holdout.get("pred_cdf"), errors="coerce").to_numpy(
            dtype=float
        )
        cov = coverage_summary(pit)
        nll = pd.to_numeric(holdout.get("pred_nll"), errors="coerce")
        rows.append(
            {
                "left_out": str(g),
                "n": int(len(holdout)),
                "bilateral_tail_err": cov.get("bilateral_tail_err"),
                "mean_nll": float(nll.mean()) if nll.notna().any() else None,
            }
        )
    return pd.DataFrame(rows)


def decide_verdict(
    dist: dict[str, Any],
    base: dict[str, Any],
    state: dict[str, Any],
    switch: dict[str, Any],
) -> AssessmentVerdict:
    reasons: list[str] = []
    status = "PASS"

    cov = dist.get("coverage") or {}
    tail_err = cov.get("bilateral_tail_err")
    if tail_err is not None and tail_err > 0.12:
        status = "FAIL"
        reasons.append(f"bilateral_tail_err={tail_err:.3f}>0.12")
    elif tail_err is not None and tail_err > 0.06:
        status = "WARN" if status == "PASS" else status
        reasons.append(f"bilateral_tail_err={tail_err:.3f}>0.06")

    uni = dist.get("uniformity") or {}
    ks = uni.get("ks_stat")
    if ks is not None and ks > 0.12:
        status = "WARN" if status == "PASS" else status
        reasons.append(f"pit_ks_stat={ks:.3f}>0.12 (effect size, not sole FAIL)")

    lb = dist.get("ljung_box") or {}
    acf1 = lb.get("acf1")
    if acf1 is not None and abs(acf1) > 0.25:
        status = "WARN" if status == "PASS" else status
        reasons.append(f"pit_acf1={acf1:.3f} indicates serial dependence")

    g_ci = base.get("gauss_ci") or {}
    t_ci = base.get("t_ci") or {}
    if (
        g_ci.get("ci_high") is not None
        and g_ci["ci_high"] < 0
        and t_ci.get("ci_high") is not None
        and t_ci["ci_high"] < 0
    ):
        status = "FAIL"
        reasons.append("HMM NLL worse than Gaussian and Student-t baselines (CI)")
    elif base.get("delta_vs_gauss") is not None and base["delta_vs_gauss"] < -0.05:
        status = "WARN" if status == "PASS" else status
        reasons.append(f"delta_nll_vs_gauss={base['delta_vs_gauss']:.4f}<0")

    if state.get("semantic_order_ok") is False:
        status = "WARN" if status == "PASS" else status
        reasons.append("raw-state forward semantics order violated")
    amb = state.get("ambiguous_rate")
    if amb is not None and amb > 0.60:
        status = "WARN" if status == "PASS" else status
        reasons.append(f"ambiguous_rate={amb:.2f}>0.60 (states hard to identify)")

    prec = switch.get("precision")
    if prec is not None and prec < 0.12 and (switch.get("n_alerts") or 0) >= 30:
        status = "FAIL"
        reasons.append(f"switch_reversal_precision={prec:.3f}<0.12")
    elif prec is not None and prec < 0.18 and (switch.get("n_alerts") or 0) >= 30:
        status = "WARN" if status == "PASS" else status
        reasons.append(f"switch_reversal_precision={prec:.3f}<0.18")

    if not reasons:
        reasons.append("no hard failures under joint distribution/state rules")
    return AssessmentVerdict(status=status, reasons=tuple(reasons))


def _causal_bias_side(row: pd.Series) -> str | None:
    if not bool(row.get("quantile_switch")):
        return None
    side = row.get("switch_side")
    if pd.isna(side):
        return None
    side_s = str(side)
    if side_s == "up":
        return "buy_bias"
    if side_s == "down":
        return "sell_bias"
    return None


def _direction_hit(side: str, fwd: float) -> bool | None:
    """Return True/False for directional hit; None if zero/undefined."""
    if not math.isfinite(float(fwd)):
        return None
    if float(fwd) == 0.0:
        return None
    if side == "buy_bias":
        return bool(float(fwd) > 0.0)
    if side == "sell_bias":
        return bool(float(fwd) < 0.0)
    return None


def _summarize_forward_hits(
    alerts: pd.DataFrame,
    *,
    horizon: int,
    window: str,
    side: str,
    n_boot: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    col = f"fwd_ret_{int(horizon)}"
    work = alerts.copy()
    if work.empty or "bias_side" not in work.columns:
        work = work.iloc[0:0]
    elif side != "both":
        work = work[work["bias_side"] == side]
    if col not in work.columns:
        work = work.iloc[0:0]
    else:
        work = work[pd.to_numeric(work[col], errors="coerce").notna()].copy()
    rets = (
        pd.to_numeric(work[col], errors="coerce") if not work.empty else pd.Series(dtype=float)
    )
    hits: list[float] = []
    signed: list[float] = []
    for idx, row in work.iterrows():
        hit = _direction_hit(str(row["bias_side"]), float(rets.loc[idx]))
        if hit is None:
            continue
        hits.append(1.0 if hit else 0.0)
        # positive signed return means bias-aligned move
        signed.append(
            float(rets.loc[idx])
            if str(row["bias_side"]) == "buy_bias"
            else -float(rets.loc[idx])
        )
    hit_arr = np.asarray(hits, dtype=float)
    ret_arr = np.asarray(signed, dtype=float)
    hit_ci = block_bootstrap_mean_ci(hit_arr, n_boot=n_boot, seed=seed)
    # Prefer month blocks when available.
    if not work.empty and "month" in work.columns and work["month"].nunique() >= 3:
        months = sorted(work["month"].astype(str).unique().tolist())
        rng = np.random.default_rng(seed)
        boots: list[float] = []
        for _ in range(n_boot):
            chosen = rng.choice(months, size=len(months), replace=True)
            sub = work[work["month"].astype(str).isin(set(chosen))]
            sub_hits: list[float] = []
            for _, row in sub.iterrows():
                h = _direction_hit(str(row["bias_side"]), float(pd.to_numeric(row[col])))
                if h is None:
                    continue
                sub_hits.append(1.0 if h else 0.0)
            if sub_hits:
                boots.append(float(np.mean(sub_hits)))
        if boots:
            hit_ci = {
                "mean": float(np.mean(hit_arr)) if hit_arr.size else None,
                "ci_low": float(np.quantile(boots, 0.025)),
                "ci_high": float(np.quantile(boots, 0.975)),
                "n": int(len(months)),
            }
    return {
        "window": window,
        "side": side,
        "horizon": int(horizon),
        "n_alerts": int(len(work)),
        "n_scored": int(hit_arr.size),
        "n_zero_or_nan": int(len(work) - hit_arr.size),
        "hit_rate": float(np.mean(hit_arr)) if hit_arr.size else None,
        "hit_ci_low": hit_ci.get("ci_low"),
        "hit_ci_high": hit_ci.get("ci_high"),
        "mean_signed_ret": float(np.mean(ret_arr)) if ret_arr.size else None,
        "median_signed_ret": float(np.median(ret_arr)) if ret_arr.size else None,
        "p25_signed_ret": float(np.quantile(ret_arr, 0.25)) if ret_arr.size else None,
        "p75_signed_ret": float(np.quantile(ret_arr, 0.75)) if ret_arr.size else None,
        "supports_bias_label": (
            bool(hit_ci.get("ci_low") is not None and float(hit_ci["ci_low"]) > 0.5)
            if hit_arr.size >= 20
            else False
        ),
    }


def _directional_reversal_events(frame: pd.DataFrame, turn: str) -> list:
    work = frame.copy()
    work["label_match"] = work["turn_kind"].astype(str) == turn
    label_frame = work[["code", "as_of", "label_match"]].rename(
        columns={"label_match": "label_trend_up"}
    )
    return cluster_positive_events(label_frame)


def _summarize_reversal_hits(
    alerts: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    window: str,
    side: str,
    horizon: int = 10,
    lead: int = 3,
) -> dict[str, Any]:
    turn = "bottom_reversal" if side == "buy_bias" else "top_reversal"
    work = alerts[alerts["bias_side"] == side].copy() if side != "both" else alerts.copy()
    # For "both", evaluate each alert against its own expected turn.
    if work.empty:
        return {
            "window": window,
            "side": side,
            "n_alerts": 0,
            "same_day_confirm_rate": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "n_events": 0,
            "n_matched": 0,
            "n_fp": 0,
            "n_duplicate": 0,
            "duplicate_rate": None,
            "fp_rate": None,
        }
    if side == "both":
        # Aggregate by evaluating buy and sell separately then combining counts.
        buy = _summarize_reversal_hits(
            alerts, frame, window=window, side="buy_bias", horizon=horizon, lead=lead
        )
        sell = _summarize_reversal_hits(
            alerts, frame, window=window, side="sell_bias", horizon=horizon, lead=lead
        )
        n_alerts = int(buy["n_alerts"] + sell["n_alerts"])
        n_matched = int((buy["n_matched"] or 0) + (sell["n_matched"] or 0))
        n_events = int((buy["n_events"] or 0) + (sell["n_events"] or 0))
        n_fp = int((buy["n_fp"] or 0) + (sell["n_fp"] or 0))
        n_dup = int((buy["n_duplicate"] or 0) + (sell["n_duplicate"] or 0))
        precision = (n_matched / n_alerts) if n_alerts else None
        recall = (n_matched / n_events) if n_events else None
        f1 = None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        same_day_n = 0
        same_day_hit = 0
        for _, row in alerts.iterrows():
            expected = (
                "bottom_reversal"
                if str(row["bias_side"]) == "buy_bias"
                else "top_reversal"
            )
            same_day_n += 1
            if str(row.get("turn_kind")) == expected:
                same_day_hit += 1
        return {
            "window": window,
            "side": "both",
            "n_alerts": n_alerts,
            "same_day_confirm_rate": (same_day_hit / same_day_n) if same_day_n else None,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_events": n_events,
            "n_matched": n_matched,
            "n_fp": n_fp,
            "n_duplicate": n_dup,
            "duplicate_rate": (n_dup / n_alerts) if n_alerts else None,
            "fp_rate": (n_fp / n_alerts) if n_alerts else None,
        }

    events = _directional_reversal_events(frame, turn)
    same_day_rate = float((work["turn_kind"].astype(str) == turn).mean())
    work = work.copy()
    work["switch_score"] = (pd.to_numeric(work["pred_cdf"], errors="coerce") - 0.5).abs()
    matched = match_alerts_to_events(
        work, events, lead=lead, horizon=horizon, score_col="switch_score"
    )
    n_alerts = int(len(work))
    n_events = int(len(events))
    n_matched = (
        int(matched["matched_event_onset"].notna().sum()) if not matched.empty else 0
    )
    n_fp = (
        int(matched["is_fp"].fillna(False).astype(bool).sum()) if not matched.empty else 0
    )
    n_dup = (
        int(matched["is_duplicate"].fillna(False).astype(bool).sum())
        if not matched.empty
        else 0
    )
    precision = (n_matched / n_alerts) if n_alerts else None
    recall = (n_matched / n_events) if n_events else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "window": window,
        "side": side,
        "n_alerts": n_alerts,
        "same_day_confirm_rate": same_day_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_events": n_events,
        "n_matched": n_matched,
        "n_fp": n_fp,
        "n_duplicate": n_dup,
        "duplicate_rate": (n_dup / n_alerts) if n_alerts else None,
        "fp_rate": (n_fp / n_alerts) if n_alerts else None,
    }


def _group_forward_accuracy(
    alerts: pd.DataFrame,
    *,
    group_col: str,
    horizons: tuple[int, ...] = CAUSAL_BIAS_HORIZONS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if alerts.empty or group_col not in alerts.columns:
        return pd.DataFrame()
    for group, grp in alerts.groupby(group_col, sort=False):
        for side in ("buy_bias", "sell_bias", "both"):
            for h in horizons:
                row = _summarize_forward_hits(
                    grp, horizon=h, window=str(group), side=side, n_boot=50, seed=17
                )
                row["group_col"] = group_col
                row["group"] = group
                rows.append(row)
    return pd.DataFrame(rows)


def run_causal_bias_study(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = CAUSAL_BIAS_HORIZONS,
    reversal_horizon: int = 10,
) -> dict[str, Any]:
    """Evaluate causal up=buy_bias / down=sell_bias accuracy without using labels to create signals."""
    work = frame.copy()
    if "month" not in work.columns:
        work["month"] = pd.to_datetime(work["as_of"]).dt.to_period("M").astype(str)
    if "year" not in work.columns:
        work["year"] = pd.to_datetime(work["as_of"]).dt.year
    work["bias_side"] = [_causal_bias_side(row) for _, row in work.iterrows()]
    alerts = work[work["bias_side"].notna()].copy()

    splits = split_time_windows(work)
    holdout_months = set(str(m) for m in (splits.get("holdout_months") or []))
    windows = {
        "full_mature": alerts,
        "holdout": alerts[alerts["month"].astype(str).isin(holdout_months)].copy(),
    }
    event_frame = work
    if "label_mature" in work.columns:
        event_frame = work[work["label_mature"].fillna(False)].copy()

    forward_rows: list[dict[str, Any]] = []
    reversal_rows: list[dict[str, Any]] = []
    for window_name, sub in windows.items():
        if window_name == "holdout":
            events_src = event_frame[
                event_frame["month"].astype(str).isin(holdout_months)
            ].copy()
        else:
            events_src = event_frame
        for side in ("buy_bias", "sell_bias", "both"):
            for h in horizons:
                # Only score rows mature for this horizon.
                mature_sub = sub
                col = f"fwd_ret_{int(h)}"
                if col in mature_sub.columns:
                    mature_sub = mature_sub[
                        pd.to_numeric(mature_sub[col], errors="coerce").notna()
                    ]
                forward_rows.append(
                    _summarize_forward_hits(
                        mature_sub, horizon=h, window=window_name, side=side
                    )
                )
            # Reversal labels require the main reversal horizon maturity.
            rev_sub = sub
            if "label_mature" in rev_sub.columns:
                rev_sub = rev_sub[rev_sub["label_mature"].fillna(False)]
            reversal_rows.append(
                _summarize_reversal_hits(
                    rev_sub,
                    events_src,
                    window=window_name,
                    side=side,
                    horizon=reversal_horizon,
                )
            )

    forward_df = pd.DataFrame(forward_rows)
    reversal_df = pd.DataFrame(reversal_rows)
    by_etf = _group_forward_accuracy(alerts, group_col="code", horizons=horizons)
    by_year = _group_forward_accuracy(alerts, group_col="year", horizons=horizons)

    # Naming recommendation from holdout CIs at h=10 (and support at shorter hs).
    naming: dict[str, Any] = {"buy_bias": {}, "sell_bias": {}}
    for side in ("buy_bias", "sell_bias"):
        side_rows = forward_df[
            (forward_df["window"] == "holdout") & (forward_df["side"] == side)
        ]
        support = {
            int(r["horizon"]): bool(r.get("supports_bias_label"))
            for _, r in side_rows.iterrows()
        }
        h10 = side_rows[side_rows["horizon"] == 10]
        keep = False
        note = "insufficient holdout support; prefer 上行/下行观察"
        if not h10.empty:
            row = h10.iloc[0]
            keep = bool(row.get("supports_bias_label"))
            note = (
                "holdout h=10 CI low > 0.5; bias wording supported"
                if keep
                else (
                    f"holdout h=10 hit_rate={row.get('hit_rate')} "
                    f"CI=[{row.get('hit_ci_low')}, {row.get('hit_ci_high')}]; "
                    "prefer 上行/下行观察"
                )
            )
        naming[side] = {
            "keep_bias_wording": keep,
            "support_by_horizon": support,
            "note": note,
        }

    summary = {
        "n_alerts_full": int(len(windows["full_mature"])),
        "n_alerts_holdout": int(len(windows["holdout"])),
        "holdout_months": sorted(holdout_months),
        "horizons": list(horizons),
        "naming_recommendation": naming,
        "holdout_forward": forward_df[forward_df["window"] == "holdout"].to_dict(
            orient="records"
        ),
        "holdout_reversal": reversal_df[reversal_df["window"] == "holdout"].to_dict(
            orient="records"
        ),
    }
    return {
        "forward": forward_df,
        "reversal": reversal_df,
        "by_etf": by_etf,
        "by_year": by_year,
        "summary": summary,
    }


PRIOR_BIAS_RULES = ("baseline_switch_side", "prior_bias", "prior_bias_strong")


def _assign_rule_bias(frame: pd.DataFrame, rule: str) -> pd.Series:
    """Assign buy_bias/sell_bias/None per row under a named causal rule."""
    out: list[str | None] = []
    for _, row in frame.iterrows():
        qsw = bool(row.get("quantile_switch"))
        if rule == "baseline_switch_side":
            out.append(
                causal_bias_from_switch_side(
                    quantile_switch=qsw,
                    switch_side=None if pd.isna(row.get("switch_side")) else row.get("switch_side"),
                )
            )
        elif rule == "prior_bias":
            out.append(
                causal_bias_from_prior(
                    quantile_switch=qsw,
                    prior_ret=row.get("causal_prior_ret"),
                    thr_prior=row.get("causal_thr_prior"),
                    require_strong=False,
                )
            )
        elif rule == "prior_bias_strong":
            out.append(
                causal_bias_from_prior(
                    quantile_switch=qsw,
                    prior_ret=row.get("causal_prior_ret"),
                    thr_prior=row.get("causal_thr_prior"),
                    require_strong=True,
                )
            )
        else:
            out.append(None)
    return pd.Series(out, index=frame.index, dtype=object)


def _diamond_side_from_turn(turn: Any) -> str | None:
    t = str(turn or "none")
    if t == "bottom_reversal":
        return "buy_bias"
    if t == "top_reversal":
        return "sell_bias"
    return None


def _same_day_alignment_stats(
    alerts: pd.DataFrame,
    *,
    window: str,
    rule: str,
    side: str,
    n_boot: int = 200,
    seed: int = 17,
) -> dict[str, Any]:
    """Same-day diamond direction hit rate for alerts with a bias wording."""
    work = alerts.copy()
    if work.empty or "rule_bias" not in work.columns:
        work = work.iloc[0:0]
    elif side != "both":
        work = work[work["rule_bias"] == side]
    else:
        work = work[work["rule_bias"].notna()]
    # Score only mature final labels.
    if "label_mature" in work.columns:
        work = work[work["label_mature"].fillna(False)].copy()
    hits: list[float] = []
    for _, row in work.iterrows():
        expected = _diamond_side_from_turn(row.get("turn_kind"))
        bias = row.get("rule_bias")
        if expected is None or bias is None:
            # No diamond on that day → miss for directional wording.
            hits.append(0.0)
            continue
        hits.append(1.0 if str(bias) == str(expected) else 0.0)
    hit_arr = np.asarray(hits, dtype=float)
    hit_ci = block_bootstrap_mean_ci(hit_arr, n_boot=n_boot, seed=seed)
    if not work.empty and "month" in work.columns and work["month"].nunique() >= 3:
        months = sorted(work["month"].astype(str).unique().tolist())
        rng = np.random.default_rng(seed)
        boots: list[float] = []
        for _ in range(n_boot):
            chosen = rng.choice(months, size=len(months), replace=True)
            sub = work[work["month"].astype(str).isin(set(chosen))]
            sub_hits: list[float] = []
            for _, row in sub.iterrows():
                expected = _diamond_side_from_turn(row.get("turn_kind"))
                bias = row.get("rule_bias")
                if expected is None or bias is None:
                    sub_hits.append(0.0)
                else:
                    sub_hits.append(1.0 if str(bias) == str(expected) else 0.0)
            if sub_hits:
                boots.append(float(np.mean(sub_hits)))
        if boots:
            hit_ci = {
                "mean": float(np.mean(hit_arr)) if hit_arr.size else None,
                "ci_low": float(np.quantile(boots, 0.025)),
                "ci_high": float(np.quantile(boots, 0.975)),
                "n": int(len(months)),
            }
    n_switch = int(alerts["quantile_switch"].fillna(False).sum()) if not alerts.empty else 0
    return {
        "window": window,
        "rule": rule,
        "side": side,
        "n_switch": n_switch,
        "n_with_bias": int(len(work)),
        "coverage": (len(work) / n_switch) if n_switch else None,
        "same_day_hit_rate": float(np.mean(hit_arr)) if hit_arr.size else None,
        "hit_ci_low": hit_ci.get("ci_low"),
        "hit_ci_high": hit_ci.get("ci_high"),
        "supports_ci": (
            bool(hit_ci.get("ci_low") is not None and float(hit_ci["ci_low"]) > 0.5)
            if hit_arr.size >= 20
            else False
        ),
    }


def _directional_event_stats_for_rule(
    frame: pd.DataFrame,
    alerts: pd.DataFrame,
    *,
    window: str,
    rule: str,
    horizon: int = 10,
    lead: int = 3,
) -> dict[str, Any]:
    """Event precision/recall when buy alerts match bottom and sell match top."""
    work = alerts[alerts["rule_bias"].notna()].copy() if not alerts.empty else alerts
    if work.empty:
        return {
            "window": window,
            "rule": rule,
            "n_alerts": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "n_events": 0,
            "n_matched": 0,
        }
    buy = work[work["rule_bias"] == "buy_bias"]
    sell = work[work["rule_bias"] == "sell_bias"]
    buy_events = _directional_reversal_events(frame, "bottom_reversal")
    sell_events = _directional_reversal_events(frame, "top_reversal")

    def _match(sub: pd.DataFrame, events: list) -> tuple[int, int, int]:
        if sub.empty:
            return 0, int(len(events)), 0
        scored = sub.copy()
        scored["switch_score"] = (
            pd.to_numeric(scored.get("pred_cdf"), errors="coerce") - 0.5
        ).abs()
        matched = match_alerts_to_events(
            scored, events, lead=lead, horizon=horizon, score_col="switch_score"
        )
        n_alerts = int(len(sub))
        n_matched = (
            int(matched["matched_event_onset"].notna().sum()) if not matched.empty else 0
        )
        return n_alerts, int(len(events)), n_matched

    n_buy, n_buy_evt, n_buy_hit = _match(buy, buy_events)
    n_sell, n_sell_evt, n_sell_hit = _match(sell, sell_events)
    n_alerts = n_buy + n_sell
    n_events = n_buy_evt + n_sell_evt
    n_matched = n_buy_hit + n_sell_hit
    precision = (n_matched / n_alerts) if n_alerts else None
    recall = (n_matched / n_events) if n_events else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "window": window,
        "rule": rule,
        "n_alerts": n_alerts,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_events": n_events,
        "n_matched": n_matched,
    }


def run_prior_bias_alignment_study(
    frame: pd.DataFrame,
    *,
    horizon: int = 10,
) -> dict[str, Any]:
    """Compare baseline switch_side bias vs prior-based bias for diamond alignment."""
    work = frame.copy()
    if "month" not in work.columns:
        work["month"] = pd.to_datetime(work["as_of"]).dt.to_period("M").astype(str)
    if "year" not in work.columns:
        work["year"] = pd.to_datetime(work["as_of"]).dt.year
    # Ensure causal prior columns exist (fallback to dual-horizon prior if needed).
    if "causal_prior_ret" not in work.columns:
        work["causal_prior_ret"] = work.get("prior_ret")
    if "causal_thr_prior" not in work.columns:
        work["causal_thr_prior"] = np.nan

    splits = split_time_windows(work)
    holdout_months = set(str(m) for m in (splits.get("holdout_months") or []))
    if "label_mature" in work.columns:
        mature_mask = work["label_mature"].fillna(False)
    else:
        mature_mask = pd.Series(True, index=work.index)
    windows = {
        "full_mature": work[mature_mask].copy(),
        "holdout": work[
            work["month"].astype(str).isin(holdout_months) & mature_mask
        ].copy(),
    }

    same_day_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    by_etf_rows: list[dict[str, Any]] = []
    by_year_rows: list[dict[str, Any]] = []

    for window_name, sub in windows.items():
        switches = sub[sub["quantile_switch"].fillna(False)].copy()
        for rule in PRIOR_BIAS_RULES:
            scored = switches.copy()
            scored["rule_bias"] = _assign_rule_bias(scored, rule)
            for side in ("buy_bias", "sell_bias", "both"):
                same_day_rows.append(
                    _same_day_alignment_stats(
                        scored, window=window_name, rule=rule, side=side
                    )
                )
            event_rows.append(
                _directional_event_stats_for_rule(
                    sub, scored, window=window_name, rule=rule, horizon=horizon
                )
            )

    # Group diagnostics on holdout for prior_bias_strong and prior_bias.
    hold = windows["holdout"]
    hold_sw = hold[hold["quantile_switch"].fillna(False)].copy()
    for rule in ("prior_bias", "prior_bias_strong"):
        scored = hold_sw.copy()
        scored["rule_bias"] = _assign_rule_bias(scored, rule)
        if "code" in scored.columns:
            for code, grp in scored.groupby("code", sort=False):
                row = _same_day_alignment_stats(
                    grp, window="holdout", rule=rule, side="both", n_boot=50
                )
                row["group_col"] = "code"
                row["group"] = code
                by_etf_rows.append(row)
        if "year" in scored.columns:
            for year, grp in scored.groupby("year", sort=False):
                row = _same_day_alignment_stats(
                    grp, window="holdout", rule=rule, side="both", n_boot=50
                )
                row["group_col"] = "year"
                row["group"] = year
                by_year_rows.append(row)

    same_day_df = pd.DataFrame(same_day_rows)
    event_df = pd.DataFrame(event_rows)

    # Recommendation from holdout: improve vs baseline AND both sides CI low > 0.5.
    hold_both = same_day_df[
        (same_day_df["window"] == "holdout") & (same_day_df["side"] == "both")
    ]
    baseline_row = hold_both[hold_both["rule"] == "baseline_switch_side"]
    baseline_rate = (
        float(baseline_row.iloc[0]["same_day_hit_rate"])
        if not baseline_row.empty and baseline_row.iloc[0]["same_day_hit_rate"] is not None
        else None
    )
    recommendation: dict[str, Any] = {
        "update_plot_labels": False,
        "rule": "observe_only",
        "note": "holdout insufficient; keep 上行/下行观察 without 偏买/偏卖",
        "baseline_same_day_hit_rate": baseline_rate,
    }
    candidates = []
    for rule in ("prior_bias_strong", "prior_bias"):
        both = hold_both[hold_both["rule"] == rule]
        buy = same_day_df[
            (same_day_df["window"] == "holdout")
            & (same_day_df["rule"] == rule)
            & (same_day_df["side"] == "buy_bias")
        ]
        sell = same_day_df[
            (same_day_df["window"] == "holdout")
            & (same_day_df["rule"] == rule)
            & (same_day_df["side"] == "sell_bias")
        ]
        if both.empty or buy.empty or sell.empty:
            continue
        b, bu, se = both.iloc[0], buy.iloc[0], sell.iloc[0]
        rate = b.get("same_day_hit_rate")
        improved = (
            baseline_rate is None
            or rate is None
            or (float(rate) > float(baseline_rate))
        )
        both_ok = bool(bu.get("supports_ci")) and bool(se.get("supports_ci"))
        candidates.append(
            {
                "rule": rule,
                "same_day_hit_rate": rate,
                "improved_vs_baseline": improved,
                "buy_supports_ci": bool(bu.get("supports_ci")),
                "sell_supports_ci": bool(se.get("supports_ci")),
                "buy_ci_low": bu.get("hit_ci_low"),
                "sell_ci_low": se.get("hit_ci_low"),
                "n_with_bias": int(b.get("n_with_bias") or 0),
                "passes": bool(improved and both_ok and (b.get("n_with_bias") or 0) >= 20),
            }
        )
    passing = [c for c in candidates if c["passes"]]
    # Prefer stronger gate when both pass.
    if passing:
        chosen = sorted(
            passing,
            key=lambda c: (
                1 if c["rule"] == "prior_bias_strong" else 0,
                float(c["same_day_hit_rate"] or 0.0),
            ),
            reverse=True,
        )[0]
        recommendation = {
            "update_plot_labels": True,
            "rule": chosen["rule"],
            "note": (
                f"holdout same-day hit={chosen['same_day_hit_rate']} "
                f"> baseline={baseline_rate}; buy/sell CI low > 0.5"
            ),
            "baseline_same_day_hit_rate": baseline_rate,
            "chosen": chosen,
            "candidates": candidates,
        }
    else:
        recommendation["candidates"] = candidates
        if candidates:
            best = max(
                candidates,
                key=lambda c: float(c["same_day_hit_rate"] or 0.0),
            )
            recommendation["note"] = (
                f"best {best['rule']} hit={best['same_day_hit_rate']} "
                f"baseline={baseline_rate}; buy_ci_ok={best['buy_supports_ci']} "
                f"sell_ci_ok={best['sell_supports_ci']}; keep 上行/下行观察"
            )

    summary = {
        "holdout_months": sorted(holdout_months),
        "recommendation": recommendation,
        "holdout_same_day": same_day_df[same_day_df["window"] == "holdout"].to_dict(
            orient="records"
        ),
        "holdout_events": event_df[event_df["window"] == "holdout"].to_dict(
            orient="records"
        ),
        "note": (
            "Bias wording is evaluated against final T+10 diamond direction only; "
            "not a trading-return claim."
        ),
    }
    return {
        "same_day": same_day_df,
        "events": event_df,
        "by_etf": pd.DataFrame(by_etf_rows),
        "by_year": pd.DataFrame(by_year_rows),
        "summary": summary,
    }


def build_assessment_frame(
    signals: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    horizon: int = 10,
) -> pd.DataFrame:
    frame = ensure_pred_diagnostics(signals)
    frame = attach_realized_returns(frame, panel)
    frame = attach_outcome_labels(frame, panel, horizon=horizon)
    base = rolling_baseline_scores(panel, frame)
    if not base.empty:
        frame = frame.merge(base, on=["code", "as_of"], how="left")
    if "pred_log_density" in frame.columns:
        dens = pd.to_numeric(frame["pred_log_density"], errors="coerce")
        frame["pred_nll"] = np.where(dens.notna(), -dens, np.nan)
    else:
        frame["pred_nll"] = np.nan
    frame["year"] = frame["as_of"].dt.year
    frame["month"] = frame["as_of"].dt.to_period("M").astype(str)
    return frame


def run_assessment(
    validation_dir: Path | str,
    panel: pd.DataFrame,
    *,
    horizon: int = 10,
    eval_end: str | None = None,
) -> dict[str, Any]:
    root = Path(validation_dir)
    signals = load_signals_oos(root)
    if eval_end is not None:
        end = pd.Timestamp(eval_end).normalize()
        signals = signals[signals["as_of"] <= end].copy()
    frame = build_assessment_frame(signals, panel, horizon=horizon)

    dist_rows = frame[frame["pred_cdf"].notna()].copy()
    dist = distribution_assessment(dist_rows)
    base = baseline_comparison(dist_rows)
    state = state_semantics_summary(frame)
    switch = switch_reversal_assessment(frame, horizon=horizon)
    matched_table = switch.pop("matched_table", pd.DataFrame())
    sens = qstar_sensitivity(
        frame,
        q_values=default_qstar_grid(),
        sides=("both",),
        horizon=horizon,
        include_bootstrap=False,
    )
    threshold_study = run_qstar_threshold_study(frame, horizon=horizon, baseline_q=0.90)
    causal_bias = run_causal_bias_study(frame, reversal_horizon=horizon)
    early_confirmation = run_early_confirmation_study(frame)
    prior_bias_alignment = run_prior_bias_alignment_study(frame, horizon=horizon)
    per_etf = per_etf_distribution_table(dist_rows)
    loo_etf = leave_one_group_robustness(dist_rows, group_col="code")
    loo_year = leave_one_group_robustness(dist_rows, group_col="year")

    split_tables: dict[str, Any] = {}
    if "label_ambiguous" in dist_rows.columns:
        for flag, name in ((True, "ambiguous"), (False, "separated_or_clear")):
            sub = dist_rows[
                dist_rows["label_ambiguous"].fillna(False).astype(bool) == flag
            ]
            split_tables[name] = distribution_assessment(sub) if not sub.empty else {}

    verdict = decide_verdict(dist, base, state, switch)
    mature_end = label_mature_end(frame)
    signal_end = (
        str(pd.Timestamp(frame["as_of"].max()).date()) if not frame.empty else None
    )
    signal_start = (
        str(pd.Timestamp(frame["as_of"].min()).date()) if not frame.empty else None
    )

    etf_pass = None
    if not per_etf.empty and per_etf["bilateral_tail_err"].notna().any():
        etf_pass = float((per_etf["bilateral_tail_err"] <= 0.10).mean())

    # Compact sensitivity for summary JSON (both-side only, key columns)
    sens_both = sens[sens["side"] == "both"] if not sens.empty and "side" in sens.columns else sens
    sens_records = []
    if not sens_both.empty:
        keep_cols = [
            c
            for c in [
                "q_star",
                "n_alerts",
                "alert_rate",
                "day_precision",
                "precision",
                "recall",
                "f1",
                "fp_rate",
                "duplicate_rate",
                "alerts_per_etf_year",
            ]
            if c in sens_both.columns
        ]
        sens_records = sens_both[keep_cols].to_dict(orient="records")

    summary = {
        "validation_dir": str(root),
        "signal_start": signal_start,
        "signal_end": signal_end,
        "label_mature_end": mature_end,
        "horizon": int(horizon),
        "n_rows": int(len(frame)),
        "n_codes": int(frame["code"].nunique()) if not frame.empty else 0,
        "n_dist_rows": int(len(dist_rows)),
        "distribution": dist,
        "baseline": base,
        "state": state,
        "switch": switch,
        "qstar_sensitivity": sens_records,
        "threshold_study": {
            "splits": threshold_study.get("splits"),
            "selection": threshold_study.get("selection"),
            "recommendation": threshold_study.get("recommendation"),
            "comparison": (
                threshold_study["comparison"].to_dict(orient="records")
                if isinstance(threshold_study.get("comparison"), pd.DataFrame)
                else []
            ),
        },
        "causal_bias": causal_bias.get("summary"),
        "early_confirmation": early_confirmation.get("summary"),
        "prior_bias_alignment": prior_bias_alignment.get("summary"),
        "split_ambiguous": split_tables,
        "per_etf_pass_rate_tail": etf_pass,
        "verdict": {"status": verdict.status, "reasons": list(verdict.reasons)},
    }
    return {
        "frame": frame,
        "per_etf": per_etf,
        "loo_etf": loo_etf,
        "loo_year": loo_year,
        "qstar_sensitivity": threshold_study.get("sensitivity_all", sens),
        "threshold_study": threshold_study,
        "causal_bias": causal_bias,
        "early_confirmation": early_confirmation,
        "prior_bias_alignment": prior_bias_alignment,
        "matched_table": matched_table,
        "summary": summary,
        "verdict": verdict,
    }


def write_assessment_outputs(
    result: dict[str, Any],
    output_dir: Path | str,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame: pd.DataFrame = result["frame"]
    frame.to_csv(out / "daily_assessment.csv", index=False, encoding="utf-8-sig")
    result["per_etf"].to_csv(
        out / "per_etf_distribution.csv", index=False, encoding="utf-8-sig"
    )
    result["loo_etf"].to_csv(
        out / "loo_etf_robustness.csv", index=False, encoding="utf-8-sig"
    )
    result["loo_year"].to_csv(
        out / "loo_year_robustness.csv", index=False, encoding="utf-8-sig"
    )
    result["qstar_sensitivity"].to_csv(
        out / "qstar_sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    study = result.get("threshold_study") or {}
    if isinstance(study.get("sensitivity_select"), pd.DataFrame):
        study["sensitivity_select"].to_csv(
            out / "qstar_sensitivity_select.csv", index=False, encoding="utf-8-sig"
        )
    if isinstance(study.get("comparison"), pd.DataFrame):
        study["comparison"].to_csv(
            out / "qstar_threshold_comparison.csv", index=False, encoding="utf-8-sig"
        )
    rec = {
        "splits": study.get("splits"),
        "selection": study.get("selection"),
        "recommendation": study.get("recommendation"),
        "baseline_q": study.get("baseline_q", 0.90),
    }
    (out / "qstar_threshold_recommendation.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    if result.get("matched_table") is not None and not result["matched_table"].empty:
        result["matched_table"].to_csv(
            out / "switch_reversal_matches.csv", index=False, encoding="utf-8-sig"
        )

    causal = result.get("causal_bias") or {}
    if isinstance(causal.get("forward"), pd.DataFrame):
        causal["forward"].to_csv(
            out / "causal_bias_forward_accuracy.csv", index=False, encoding="utf-8-sig"
        )
    if isinstance(causal.get("reversal"), pd.DataFrame):
        causal["reversal"].to_csv(
            out / "causal_bias_reversal_accuracy.csv", index=False, encoding="utf-8-sig"
        )
    if isinstance(causal.get("by_etf"), pd.DataFrame):
        causal["by_etf"].to_csv(
            out / "causal_bias_by_etf.csv", index=False, encoding="utf-8-sig"
        )
    if isinstance(causal.get("by_year"), pd.DataFrame):
        causal["by_year"].to_csv(
            out / "causal_bias_by_year.csv", index=False, encoding="utf-8-sig"
        )
    causal_summary = causal.get("summary") or result.get("summary", {}).get("causal_bias")
    if causal_summary is not None:
        (out / "causal_bias_summary.json").write_text(
            json.dumps(
                json.loads(json.dumps(causal_summary, ensure_ascii=False, default=str)),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    early = result.get("early_confirmation") or {}
    if isinstance(early.get("migration"), pd.DataFrame):
        early["migration"].to_csv(
            out / "early_confirmation_migration.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if isinstance(early.get("by_etf"), pd.DataFrame):
        early["by_etf"].to_csv(
            out / "early_confirmation_by_etf.csv", index=False, encoding="utf-8-sig"
        )
    if isinstance(early.get("by_year"), pd.DataFrame):
        early["by_year"].to_csv(
            out / "early_confirmation_by_year.csv", index=False, encoding="utf-8-sig"
        )
    early_summary = early.get("summary") or result.get("summary", {}).get(
        "early_confirmation"
    )
    if early_summary is not None:
        (out / "early_confirmation_summary.json").write_text(
            json.dumps(
                json.loads(json.dumps(early_summary, ensure_ascii=False, default=str)),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    prior_align = result.get("prior_bias_alignment") or {}
    if isinstance(prior_align.get("same_day"), pd.DataFrame):
        prior_align["same_day"].to_csv(
            out / "prior_bias_alignment_same_day.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if isinstance(prior_align.get("events"), pd.DataFrame):
        prior_align["events"].to_csv(
            out / "prior_bias_alignment_events.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if isinstance(prior_align.get("by_etf"), pd.DataFrame):
        prior_align["by_etf"].to_csv(
            out / "prior_bias_alignment_by_etf.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if isinstance(prior_align.get("by_year"), pd.DataFrame):
        prior_align["by_year"].to_csv(
            out / "prior_bias_alignment_by_year.csv",
            index=False,
            encoding="utf-8-sig",
        )
    prior_summary = prior_align.get("summary") or result.get("summary", {}).get(
        "prior_bias_alignment"
    )
    if prior_summary is not None:
        (out / "prior_bias_alignment_summary.json").write_text(
            json.dumps(
                json.loads(json.dumps(prior_summary, ensure_ascii=False, default=str)),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    summary = result["summary"]
    summary_json = json.loads(json.dumps(summary, ensure_ascii=False, default=str))
    (out / "assessment_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = render_assessment_markdown(summary_json)
    (out / "ASSESSMENT.md").write_text(md, encoding="utf-8")
    return out


def render_assessment_markdown(summary: dict[str, Any]) -> str:
    v = summary.get("verdict") or {}
    dist = summary.get("distribution") or {}
    uni = dist.get("uniformity") or {}
    cov = dist.get("coverage") or {}
    base = summary.get("baseline") or {}
    state = summary.get("state") or {}
    switch = summary.get("switch") or {}
    study = summary.get("threshold_study") or {}
    rec = study.get("recommendation") or {}
    sel = study.get("selection") or {}
    splits = study.get("splits") or {}
    lines = [
        "# ETF HMM 分布与状态准确性评估",
        "",
        f"- 状态: **{v.get('status', '—')}**",
        f"- 信号区间: {summary.get('signal_start')} → {summary.get('signal_end')}",
        f"- 标签成熟截止: `{summary.get('label_mature_end')}`（horizon={summary.get('horizon')}）",
        f"- 行数/标的: {summary.get('n_rows')} / {summary.get('n_codes')}",
        f"- 分布可用行: {summary.get('n_dist_rows')}",
        "",
        "## 判定理由",
    ]
    for r in v.get("reasons") or []:
        lines.append(f"- {r}")
    lines.extend(
        [
            "",
            "## 分布正确性（实时 PIT，不依赖未来）",
            f"- PIT mean={uni.get('mean')}, KS={uni.get('ks_stat')}, CvM={uni.get('cvm_stat')}",
            f"- bilateral tail rate={cov.get('bilateral_tail_rate')} (期望≈0.20), err={cov.get('bilateral_tail_err')}",
            f"- mean NLL={dist.get('mean_nll')}",
            f"- vs Gaussian ΔNLL={base.get('delta_vs_gauss')} (正=HMM更好)",
            f"- vs Student-t ΔNLL={base.get('delta_vs_t')}",
            "",
            "## 原始状态语义",
            f"- state列: `{state.get('state_col')}`",
            f"- occupancy: `{state.get('occupancy')}`",
            f"- mean dwell: `{state.get('mean_dwell')}`",
            f"- transition_rate={state.get('transition_rate')}",
            f"- ambiguous_rate={state.get('ambiguous_rate')}, separated_rate={state.get('separated_rate')}",
            f"- semantic_order_ok={state.get('semantic_order_ok')}",
            f"- forward_by_state: `{state.get('forward_by_state')}`",
            "",
            "## Operational switch × 反转代理（仅成熟样本）",
            f"- n_alerts={switch.get('n_alerts')}, n_events={switch.get('n_events')}",
            f"- event precision={switch.get('precision')}, recall={switch.get('recall')}",
            f"- day precision={switch.get('day_precision')}, day recall={switch.get('day_recall')}",
            f"- event_match={switch.get('event_match')}",
            "",
            "## q* 阈值 walk-forward 研究",
            f"- select months: `{splits.get('select_months')}`",
            f"- holdout months: `{splits.get('holdout_months')}`",
            f"- selection mode: `{sel.get('recommended_mode')}` reason=`{sel.get('reason')}`",
            f"- select best unified: `{sel.get('best_unified')}`",
            f"- select asymmetric: `{sel.get('best_asymmetric')}`",
            f"- **production recommendation:** keep_q90={rec.get('keep_production_q90')} "
            f"mode={rec.get('mode')} q_up={rec.get('production_q_up')} "
            f"q_down={rec.get('production_q_down')}",
            f"- holdout F1 baseline={rec.get('holdout_f1_baseline')} "
            f"recommended={rec.get('holdout_f1_recommended')} "
            f"improves={rec.get('holdout_improves')}",
            f"- note: {rec.get('note')}",
            "",
            "### select vs holdout comparison",
        ]
    )
    for row in study.get("comparison") or []:
        lines.append(
            f"- {row.get('label')}: q_up={row.get('q_up')} q_down={row.get('q_down')} "
            f"n_alerts={row.get('n_alerts')} precision={row.get('precision')} "
            f"recall={row.get('recall')} f1={row.get('f1')} fp_rate={row.get('fp_rate')}"
        )
    # Highlight baseline 0.90 and neighbors from sensitivity summary
    sens = summary.get("qstar_sensitivity") or []
    highlight = [
        r
        for r in sens
        if isinstance(r, dict) and r.get("q_star") in (0.85, 0.90, 0.95)
    ]
    if highlight:
        lines.append("")
        lines.append("### baseline neighborhood (both tails, full mature)")
        for r in highlight:
            lines.append(
                f"- q*={r.get('q_star')}: alerts={r.get('n_alerts')} "
                f"alert_rate={r.get('alert_rate')} day_prec={r.get('day_precision')} "
                f"event_prec={r.get('precision')} recall={r.get('recall')} f1={r.get('f1')}"
            )
    cb = summary.get("causal_bias") or {}
    naming = cb.get("naming_recommendation") or {}
    lines.extend(
        [
            "",
            "## 因果偏买/偏卖准确性（q*=0.90 信号，不反向改标签）",
            f"- full alerts={cb.get('n_alerts_full')}, holdout alerts={cb.get('n_alerts_holdout')}",
            f"- holdout months: `{cb.get('holdout_months')}`",
            f"- horizons: `{cb.get('horizons')}`",
            "",
            "### holdout 前向方向命中率",
        ]
    )
    for row in cb.get("holdout_forward") or []:
        lines.append(
            f"- {row.get('side')} h={row.get('horizon')}: "
            f"n={row.get('n_scored')} hit_rate={row.get('hit_rate')} "
            f"CI=[{row.get('hit_ci_low')}, {row.get('hit_ci_high')}] "
            f"mean_signed_ret={row.get('mean_signed_ret')} "
            f"supports_bias={row.get('supports_bias_label')}"
        )
    lines.append("")
    lines.append("### holdout 方向一致反转事件")
    for row in cb.get("holdout_reversal") or []:
        lines.append(
            f"- {row.get('side')}: n_alerts={row.get('n_alerts')} "
            f"same_day={row.get('same_day_confirm_rate')} "
            f"precision={row.get('precision')} recall={row.get('recall')} "
            f"f1={row.get('f1')} dup={row.get('duplicate_rate')} fp={row.get('fp_rate')}"
        )
    lines.append("")
    lines.append("### 命名建议")
    for side in ("buy_bias", "sell_bias"):
        info = naming.get(side) or {}
        lines.append(
            f"- {side}: keep_bias_wording={info.get('keep_bias_wording')} — {info.get('note')}"
        )
    early = summary.get("early_confirmation") or {}
    lines.extend(
        [
            "",
            "## 两阶段反转确认（T+3 早期 → T+10 最终）",
            f"- holdout months: `{early.get('holdout_months')}`",
            f"- note: {early.get('note')}",
            "",
            "### window summaries",
        ]
    )
    for row in early.get("windows") or []:
        lines.append(
            f"- {row.get('window')}: dual={row.get('n_dual_mature')} "
            f"early+={row.get('n_early_positive')} final+={row.get('n_final_positive')} "
            f"confirmed={row.get('n_confirmed')} revoked={row.get('n_revoked')} "
            f"dir_changed={row.get('n_direction_changed')} final_only={row.get('n_final_only')} "
            f"confirm_rate|early={row.get('confirm_rate_given_early')} "
            f"revoke_rate|early={row.get('revoke_rate_given_early')}"
        )
    pba = summary.get("prior_bias_alignment") or {}
    prec = pba.get("recommendation") or {}
    lines.extend(
        [
            "",
            "## 先验偏买对齐菱形（prior_ret vs switch_side）",
            f"- holdout months: `{pba.get('holdout_months')}`",
            f"- note: {pba.get('note')}",
            f"- **recommendation:** update_plot={prec.get('update_plot_labels')} "
            f"rule={prec.get('rule')} — {prec.get('note')}",
            "",
            "### holdout same-day diamond alignment",
        ]
    )
    for row in pba.get("holdout_same_day") or []:
        lines.append(
            f"- {row.get('rule')} {row.get('side')}: n_bias={row.get('n_with_bias')} "
            f"coverage={row.get('coverage')} hit={row.get('same_day_hit_rate')} "
            f"CI=[{row.get('hit_ci_low')}, {row.get('hit_ci_high')}] "
            f"supports_ci={row.get('supports_ci')}"
        )
    lines.append("")
    lines.append("### holdout directional event metrics")
    for row in pba.get("holdout_events") or []:
        lines.append(
            f"- {row.get('rule')}: n_alerts={row.get('n_alerts')} "
            f"precision={row.get('precision')} recall={row.get('recall')} "
            f"f1={row.get('f1')}"
        )
    lines.extend(
        [
            "",
            "## 稳健性",
            f"- per-ETF bilateral-tail pass rate (err≤0.10): {summary.get('per_etf_pass_rate_tail')}",
            f"- q* sensitivity rows: {len(sens)} (see `qstar_sensitivity.csv`)",
            "- causal bias by ETF/year: see `causal_bias_by_etf.csv` / `causal_bias_by_year.csv`",
            "- early confirmation by ETF/year: see `early_confirmation_by_etf.csv` / `early_confirmation_by_year.csv`",
            "- prior bias alignment: see `prior_bias_alignment_*.csv`",
            "",
            "## 注意",
            "- `pred_cdf` / NLL / PIT 可评估到 signal_end。",
            "- 反转/趋势/未来波动指标只覆盖 label_mature_end 之前的日期。",
            "- 阈值选择只用 select months；holdout 只评估一次，不回调参数。",
            "- 不以单一 KS p-value 判定 FAIL；需联合尾部覆盖、基准 NLL 与状态语义。",
            "- 当前生产默认 `switch_quantile_q=0.90`；仅当 holdout 稳定优于基线时才建议改配置。",
            "- 偏买/偏卖仅为因果方向倾向命名；若 holdout CI 不支持 >50%，应退回“上行/下行观察”。",
            "- 前向命中率不等于扣费后可交易收益；不用本次回测反向改历史标签。",
            "- T+3 早期确认为事后临时标签；最终口径仍是 T+10，早期确认不是可交易收益承诺。",
            "- 三角偏买/偏卖若改用 prior_ret，仅在 holdout 对齐菱形达标后改图；仍不使用未来 post_ret。",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "AssessmentVerdict",
    "attach_outcome_labels",
    "attach_realized_returns",
    "baseline_comparison",
    "block_bootstrap_mean_ci",
    "build_assessment_frame",
    "coverage_summary",
    "decide_verdict",
    "default_qstar_grid",
    "distribution_assessment",
    "evaluate_threshold_on_frame",
    "label_mature_end",
    "load_signals_oos",
    "month_block_bootstrap_metric_ci",
    "pit_autocorr_ljung_box",
    "pit_uniformity_summary",
    "qstar_sensitivity",
    "rolling_baseline_scores",
    "run_assessment",
    "run_causal_bias_study",
    "run_early_confirmation_study",
    "run_prior_bias_alignment_study",
    "run_qstar_threshold_study",
    "select_qstar_threshold",
    "split_time_windows",
    "state_semantics_summary",
    "switch_mask_from_cdf",
    "switch_reversal_assessment",
    "write_assessment_outputs",
]
