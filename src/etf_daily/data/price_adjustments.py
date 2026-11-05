"""Shared ex-dividend / split price adjustment (post-hoc qfq).

Detects large single-day close jumps (e.g. ETF 除权) and rescales pre-jump
history so return series stay continuous. Round splits (1/2, 1/3, 1/10, …)
use that multiple, leaving the ex-day's own move on the bar. Other cliffs
still use the close ratio. Used across pipeline, Route A/B, forecast μ,
pool builder, and backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_QFQ_THRESHOLD = 0.25
# Price multipliers (new/old), not simple returns. 5% relative tolerance.
ROUND_SPLIT_RATIOS: tuple[float, ...] = (0.1, 0.2, 0.25, 1.0 / 3.0, 0.5, 2.0, 3.0, 4.0, 5.0, 10.0)
SPLIT_RATIO_TOL = 0.05


def _snap_split_ratio(observed: float, *, tol: float = SPLIT_RATIO_TOL) -> float | None:
    """Nearest common split multiple, or None when ``observed`` is not within ``tol``."""
    if not np.isfinite(observed) or observed <= 0.0:
        return None
    best = min(ROUND_SPLIT_RATIOS, key=lambda ratio: abs(observed / ratio - 1.0))
    if abs(observed / best - 1.0) <= tol:
        return float(best)
    return None


def _step_split_ratio(close_ratio: float, open_ratio: float | None) -> tuple[float, bool]:
    """One-step multiplier for a detected cliff, and whether it snapped to a round multiple.

    Prefer the open/previous-close ratio when that print sits on a round split.
    Otherwise snap the close ratio. A non-round cliff keeps the close ratio.
    """
    if open_ratio is not None:
        snapped = _snap_split_ratio(open_ratio)
        if snapped is not None:
            return snapped, True
    snapped = _snap_split_ratio(close_ratio)
    if snapped is not None:
        return snapped, True
    return close_ratio, False


def _compute_cum_factors(
    close: np.ndarray,
    threshold: float,
    open_: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Return per-row cumulative qfq factors and jump records for one price series.

    Walks backward over consecutive *valid* closes (skips NaN / non-positive holes)
    so an ETF 除权 cliff still fires when a missing bar sits between the pre/post
    prints (e.g. SH512690: 2021-05-14 NaN, 2021-05-17 −48% split).

    A cliff is still detected from the close ratio. When that ratio (or the open
    ratio) sits within 5% of a round split, history before the ex-day is scaled by
    the round multiple so the ex-day's own move stays on the bar. Otherwise the
    close ratio is used and the ex-day return is flattened.
    """
    n = len(close)
    cum_factor = np.ones(n, dtype=float)
    jumps: list[dict[str, object]] = []
    if n < 2 or threshold is None or threshold <= 0:
        return cum_factor, jumps

    valid = [
        i
        for i in range(n)
        if np.isfinite(close[i]) and float(close[i]) > 0.0
    ]
    if len(valid) < 2:
        return cum_factor, jumps

    for k in range(len(valid) - 1, 0, -1):
        newer_i = int(valid[k])
        older_i = int(valid[k - 1])
        newer_c = float(close[newer_i])
        older_c = float(close[older_i])
        newer_factor = float(cum_factor[newer_i])
        close_ratio = newer_c / older_c
        rel = close_ratio - 1.0
        if abs(rel) > threshold:
            open_ratio = None
            if open_ is not None and newer_i < len(open_):
                ov = float(open_[newer_i])
                if np.isfinite(ov) and ov > 0.0:
                    open_ratio = ov / older_c
            step, snapped = _step_split_ratio(close_ratio, open_ratio)
            factor_here = float(step) * newer_factor
            jumps.append(
                {
                    "close_prev_raw": older_c,
                    "close_today_raw": newer_c,
                    "raw_ratio_minus_1": float(rel),
                    "split_ratio": float(step),
                    "snapped": bool(snapped),
                    "applied_factor": float(factor_here),
                    "_index": newer_i,
                }
            )
            # Scale everything strictly before the post-jump bar (holes included).
            cum_factor[:newer_i] = factor_here
        else:
            cum_factor[:newer_i] = newer_factor
    return cum_factor, jumps


def _open_array(frame: pd.DataFrame) -> np.ndarray | None:
    for name in ("$open", "open"):
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    return None


def auto_qfq_adjust_long_frame(
    data: pd.DataFrame,
    *,
    date_col: str,
    code_col: str,
    close_col: str,
    next_close_col: str | None = None,
    threshold: float = DEFAULT_QFQ_THRESHOLD,
    extra_price_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Post-hoc qfq on a long-format panel (one row per code × date).

    Returns (adjusted_df, records_df).
    """
    if threshold is None or threshold <= 0:
        return data.copy(), pd.DataFrame()

    out = data.copy()
    all_records: list[dict[str, object]] = []
    price_cols = [close_col]
    if extra_price_cols:
        price_cols.extend(col for col in extra_price_cols if col in out.columns and col != close_col)

    for code, grp in out.groupby(code_col, sort=False):
        grp_sorted = grp.sort_values(date_col)
        idx = grp_sorted.index.to_numpy()
        close = grp_sorted[close_col].astype(float).to_numpy()
        cum_factor, jumps = _compute_cum_factors(close, threshold, _open_array(grp_sorted))
        if np.allclose(cum_factor, 1.0):
            continue

        for jump in jumps:
            t = int(jump.pop("_index"))
            all_records.append(
                {
                    "code": str(code),
                    "datetime": grp_sorted.iloc[t][date_col],
                    **jump,
                }
            )

        for col in price_cols:
            raw = grp_sorted[col].astype(float).to_numpy()
            out.loc[idx, col] = (raw * cum_factor).astype(float)

        if next_close_col and next_close_col in out.columns:
            raw_next = grp_sorted[next_close_col].astype(float).to_numpy()
            shifted_factor = np.concatenate([cum_factor[1:], [cum_factor[-1]]])
            out.loc[idx, next_close_col] = raw_next * shifted_factor

    records_df = pd.DataFrame(all_records)
    return out, records_df


def auto_qfq_adjust_close_panel(
    close: pd.DataFrame,
    threshold: float = DEFAULT_QFQ_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Post-hoc qfq on a wide close panel (index=datetime, columns=ticker).

    Returns (adjusted_close, records_df).
    """
    if close.empty or threshold is None or threshold <= 0:
        return close.copy(), pd.DataFrame()

    adjusted = close.copy()
    all_records: list[dict[str, object]] = []
    for col in adjusted.columns:
        series = adjusted[col].astype(float)
        values = series.to_numpy()
        cum_factor, jumps = _compute_cum_factors(values, threshold)
        if np.allclose(cum_factor, 1.0):
            continue
        adjusted[col] = (values * cum_factor).astype(float)
        index = series.index
        for jump in jumps:
            t = int(jump.pop("_index"))
            all_records.append(
                {
                    "code": str(col),
                    "datetime": index[t],
                    **jump,
                }
            )

    return adjusted, pd.DataFrame(all_records)


def auto_qfq_adjust_price_columns_by_group(
    df: pd.DataFrame,
    *,
    group_col: str,
    date_col: str,
    price_cols: list[str],
    threshold: float = DEFAULT_QFQ_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Qfq-adjust multiple OHLC columns per group using close as the jump detector."""
    if df.empty or not price_cols:
        return df.copy(), pd.DataFrame()

    close_col = "close" if "close" in price_cols else price_cols[0]
    return auto_qfq_adjust_long_frame(
        df,
        date_col=date_col,
        code_col=group_col,
        close_col=close_col,
        threshold=threshold,
        extra_price_cols=[c for c in price_cols if c != close_col],
    )


def detect_ex_dividend_mask(
    next_return: pd.Series,
    next2_return: pd.Series,
    *,
    ex_div_thresh: float = -0.25,
    stabilization_thresh: float = -0.15,
) -> pd.Series:
    """Heuristic mask for ex-dividend rows (matches legacy data_prep logic)."""
    return (next_return < ex_div_thresh) & (next2_return > stabilization_thresh)


def log_qfq_adjustments(records: pd.DataFrame, *, stage: str, max_samples: int = 5) -> None:
    """Print a short summary of qfq adjustments."""
    if records is None or records.empty:
        return
    head = records.head(max_samples)
    samples = ", ".join(
        f"{row['code']}@{pd.Timestamp(row['datetime']).date()}"
        f"(raw_ret={row['raw_ratio_minus_1']:+.2%}, "
        f"ratio={float(row['split_ratio']):.4g}, "
        f"{'snap' if bool(row.get('snapped')) else 'close'}, "
        f"f={row['applied_factor']:.3f})"
        for _, row in head.iterrows()
    )
    print(f"[INFO] {stage}: qfq adjusted {len(records)} jump(s). Sample: {samples}")
