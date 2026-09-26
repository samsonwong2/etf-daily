"""Extreme single-day drop → formal down-triangle override.

Keeps the model quantile-tail rule (pred_cdf tails) unchanged. When the model
misses a large crash day, emit an effective red triangle so OPS reclaim/buy
rules can see it, while preserving provenance for coverage audits.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

DEFAULT_EXTREME_DROP_THRESHOLD = -0.07
SOURCE_QUANTILE_TAIL = "quantile_tail"
SOURCE_EXTREME_DROP = "extreme_drop_override"
SOURCE_NONE = ""

MODEL_SWITCH_COL = "model_quantile_switch"
EXTREME_SWITCH_COL = "extreme_drop_switch"
EFFECTIVE_SWITCH_COL = "quantile_switch"
SWITCH_SIDE_COL = "switch_side"
SWITCH_SOURCE_COL = "switch_source"
DAY_RET_COL = "day_ret"


def day_returns_from_close(close: pd.Series) -> pd.Series:
    """Causal simple return close_t / close_{t-1} - 1 (no fill)."""
    s = pd.to_numeric(close, errors="coerce")
    s.index = pd.to_datetime(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.pct_change(fill_method=None)


def day_returns_frame_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Wide close panel (columns=codes, index=dates) → long day_ret frame."""
    rows: list[pd.DataFrame] = []
    for code in panel.columns:
        ret = day_returns_from_close(panel[code])
        if ret.empty:
            continue
        rows.append(
            pd.DataFrame(
                {
                    "code": str(code).upper(),
                    "as_of": pd.to_datetime(ret.index).normalize(),
                    DAY_RET_COL: ret.to_numpy(dtype=float),
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=["code", "as_of", DAY_RET_COL])
    return pd.concat(rows, ignore_index=True)


def day_returns_frame_from_ohlcv(ohlcv: pd.DataFrame, *, code: str) -> pd.DataFrame:
    """OHLCV with datetime + $close → day_ret frame for one code."""
    if ohlcv is None or ohlcv.empty or "$close" not in ohlcv.columns:
        return pd.DataFrame(columns=["code", "as_of", DAY_RET_COL])
    df = ohlcv.copy()
    df["as_of"] = pd.to_datetime(df["datetime"]).dt.normalize()
    df = df.sort_values("as_of")
    close = df.set_index("as_of")["$close"]
    ret = day_returns_from_close(close)
    return pd.DataFrame(
        {
            "code": str(code).upper(),
            "as_of": pd.to_datetime(ret.index).normalize(),
            DAY_RET_COL: ret.to_numpy(dtype=float),
        }
    )


def _bool_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        s = values
    else:
        s = pd.Series(values)
    return s.map(
        lambda v: False
        if v is None or (isinstance(v, float) and pd.isna(v))
        else bool(v)
    ).astype(bool)


def _side_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        s = values
    else:
        s = pd.Series(values)
    out = s.map(lambda v: "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
    out = out.replace({"nan": "", "None": "", "NaN": ""})
    return out


def apply_extreme_drop_override(
    signals: pd.DataFrame,
    day_ret: pd.DataFrame | Mapping[str, Any] | None = None,
    *,
    threshold: float = DEFAULT_EXTREME_DROP_THRESHOLD,
    inplace: bool = False,
) -> pd.DataFrame:
    """Synthesize model / extreme / effective triangle fields.

    Parameters
    ----------
    signals:
        Must include ``as_of``, ``code``, and the *original* model
        ``quantile_switch`` / ``switch_side`` (or already-split model cols).
    day_ret:
        Long frame with ``code``, ``as_of``, ``day_ret``. If omitted and
        ``signals`` already has ``day_ret``, that column is used.
    threshold:
        Simple return cutoff (default -7%). Override fires only when
        ``day_ret <= threshold`` and there is no model *down* triangle.
    """
    if signals is None or signals.empty:
        out = signals.copy() if signals is not None and not inplace else signals
        return out

    out = signals if inplace else signals.copy()
    out["as_of"] = pd.to_datetime(out["as_of"]).dt.normalize()
    out["code"] = out["code"].astype(str).str.upper()

    # Snapshot original model columns once (idempotent if already present).
    if MODEL_SWITCH_COL not in out.columns:
        out[MODEL_SWITCH_COL] = _bool_series(out.get(EFFECTIVE_SWITCH_COL, False))
    else:
        out[MODEL_SWITCH_COL] = _bool_series(out[MODEL_SWITCH_COL])

    model_side_col = "model_switch_side"
    if model_side_col not in out.columns:
        out[model_side_col] = _side_series(out.get(SWITCH_SIDE_COL))
    else:
        out[model_side_col] = _side_series(out[model_side_col])

    # Attach day returns.
    if day_ret is not None:
        ret = pd.DataFrame(day_ret).copy()
        if ret.empty:
            out[DAY_RET_COL] = np.nan
        else:
            need = {"code", "as_of", DAY_RET_COL}
            missing = need - set(ret.columns)
            if missing:
                raise ValueError(f"day_ret missing columns: {sorted(missing)}")
            ret["as_of"] = pd.to_datetime(ret["as_of"]).dt.normalize()
            ret["code"] = ret["code"].astype(str).str.upper()
            ret[DAY_RET_COL] = pd.to_numeric(ret[DAY_RET_COL], errors="coerce")
            ret = ret.drop_duplicates(subset=["code", "as_of"], keep="last")
            if DAY_RET_COL in out.columns:
                out = out.drop(columns=[DAY_RET_COL])
            out = out.merge(ret[["code", "as_of", DAY_RET_COL]], on=["code", "as_of"], how="left")
    elif DAY_RET_COL not in out.columns:
        out[DAY_RET_COL] = np.nan
    else:
        out[DAY_RET_COL] = pd.to_numeric(out[DAY_RET_COL], errors="coerce")

    model_down = out[MODEL_SWITCH_COL] & out[model_side_col].eq("down")
    model_up = out[MODEL_SWITCH_COL] & out[model_side_col].eq("up")
    extreme = out[DAY_RET_COL].notna() & (out[DAY_RET_COL] <= float(threshold))
    out[EXTREME_SWITCH_COL] = extreme & ~model_down

    # Effective: model OR extreme override; one red triangle per day.
    # Prefer model side when model fired; else extreme → down.
    effective = out[MODEL_SWITCH_COL] | out[EXTREME_SWITCH_COL]
    side = np.where(
        model_down,
        "down",
        np.where(
            model_up,
            "up",
            np.where(out[EXTREME_SWITCH_COL], "down", ""),
        ),
    )
    source = np.where(
        model_down | model_up,
        SOURCE_QUANTILE_TAIL,
        np.where(out[EXTREME_SWITCH_COL], SOURCE_EXTREME_DROP, SOURCE_NONE),
    )

    out[EFFECTIVE_SWITCH_COL] = effective.astype(bool)
    out[SWITCH_SIDE_COL] = pd.Series(side, index=out.index).replace({"": pd.NA})
    out[SWITCH_SOURCE_COL] = pd.Series(source, index=out.index)
    out.loc[~out[EFFECTIVE_SWITCH_COL], SWITCH_SOURCE_COL] = SOURCE_NONE
    return out


def ensure_extreme_drop_override(
    signals: pd.DataFrame,
    day_ret: pd.DataFrame | None = None,
    *,
    threshold: float = DEFAULT_EXTREME_DROP_THRESHOLD,
) -> pd.DataFrame:
    """Apply override unless model/extreme fields are already present."""
    if signals is None or signals.empty:
        return signals
    if (
        MODEL_SWITCH_COL in signals.columns
        and EXTREME_SWITCH_COL in signals.columns
        and SWITCH_SOURCE_COL in signals.columns
    ):
        return signals
    return apply_extreme_drop_override(signals, day_ret, threshold=threshold)


def coverage_split_stats(signals: pd.DataFrame) -> dict[str, Any]:
    """Separate model q* coverage from effective (model ∪ extreme) coverage."""
    if signals is None or signals.empty:
        return {
            "n_rows": 0,
            "model_switch_rate": None,
            "model_down_rate": None,
            "extreme_override_rate": None,
            "effective_switch_rate": None,
            "effective_down_rate": None,
            "n_extreme_override": 0,
        }
    frame = signals.copy()
    if MODEL_SWITCH_COL not in frame.columns:
        frame[MODEL_SWITCH_COL] = _bool_series(frame.get(EFFECTIVE_SWITCH_COL, False))
    else:
        frame[MODEL_SWITCH_COL] = _bool_series(frame[MODEL_SWITCH_COL])
    if EXTREME_SWITCH_COL not in frame.columns:
        frame[EXTREME_SWITCH_COL] = False
    else:
        frame[EXTREME_SWITCH_COL] = _bool_series(frame[EXTREME_SWITCH_COL])
    if EFFECTIVE_SWITCH_COL not in frame.columns:
        frame[EFFECTIVE_SWITCH_COL] = frame[MODEL_SWITCH_COL] | frame[EXTREME_SWITCH_COL]
    else:
        frame[EFFECTIVE_SWITCH_COL] = _bool_series(frame[EFFECTIVE_SWITCH_COL])
    side = _side_series(frame.get(SWITCH_SIDE_COL))
    n = len(frame)
    model = frame[MODEL_SWITCH_COL]
    extreme = frame[EXTREME_SWITCH_COL]
    effective = frame[EFFECTIVE_SWITCH_COL]
    # Model down uses preserved model side when available.
    model_side = _side_series(frame.get("model_switch_side", side))
    return {
        "n_rows": int(n),
        "model_switch_rate": float(model.mean()) if n else None,
        "model_down_rate": float((model & model_side.eq("down")).mean()) if n else None,
        "extreme_override_rate": float(extreme.mean()) if n else None,
        "effective_switch_rate": float(effective.mean()) if n else None,
        "effective_down_rate": float((effective & side.eq("down")).mean()) if n else None,
        "n_extreme_override": int(extreme.sum()),
    }


def enrich_oos_from_ohlcv_loader(
    signals: pd.DataFrame,
    load_ohlcv,
    *,
    threshold: float = DEFAULT_EXTREME_DROP_THRESHOLD,
    lookback_calendar_days: int = 40,
) -> pd.DataFrame:
    """Ensure override fields; load day returns via ``load_ohlcv(code, start, end)``."""
    if signals is None or signals.empty:
        return signals
    if (
        MODEL_SWITCH_COL in signals.columns
        and EXTREME_SWITCH_COL in signals.columns
        and SWITCH_SOURCE_COL in signals.columns
    ):
        return signals
    work = signals.copy()
    work["as_of"] = pd.to_datetime(work["as_of"]).dt.normalize()
    work["code"] = work["code"].astype(str).str.upper()
    parts: list[pd.DataFrame] = []
    for code, grp in work.groupby("code", sort=False):
        start = str(grp["as_of"].min().date())
        end = str(grp["as_of"].max().date())
        ohlcv = load_ohlcv(
            code,
            start,
            end,
            init_qlib=False,
            lookback_calendar_days=lookback_calendar_days,
            clip_to_window=False,
        )
        parts.append(day_returns_frame_from_ohlcv(ohlcv, code=str(code)))
    day_ret = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["code", "as_of", DAY_RET_COL])
    )
    return apply_extreme_drop_override(work, day_ret, threshold=threshold)


__all__ = [
    "DEFAULT_EXTREME_DROP_THRESHOLD",
    "SOURCE_EXTREME_DROP",
    "SOURCE_NONE",
    "SOURCE_QUANTILE_TAIL",
    "apply_extreme_drop_override",
    "coverage_split_stats",
    "day_returns_frame_from_ohlcv",
    "day_returns_frame_from_panel",
    "day_returns_from_close",
    "enrich_oos_from_ohlcv_loader",
    "ensure_extreme_drop_override",
]
