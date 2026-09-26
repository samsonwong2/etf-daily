"""Detect corporate splits vs genuine large return jumps (sidecar diagnostics).

Does not alter GARCH short-horizon CSV columns. Used by GARCH REPORT / diagnostics
CSV and by the HMM board for optional split-day filtering.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pipeline.price_adjustments import DEFAULT_QFQ_THRESHOLD, _compute_cum_factors

# Common ETF split ratios (new/old - 1): 1:2 -> -0.5, 1:3 -> -2/3, 2:1 -> +1, ...
_SPLIT_RATIO_TARGETS: tuple[float, ...] = (
    -0.5,
    -2.0 / 3.0,
    -0.75,
    -0.8,
    1.0,
    2.0,
    3.0,
)


@dataclass(frozen=True)
class JumpFlagConfig:
    split_threshold: float = DEFAULT_QFQ_THRESHOLD
    jump_threshold: float = 0.08
    recent_lookback: int = 20
    split_ratio_tol: float = 0.05
    live_gap_threshold: float = 0.25


@dataclass
class SymbolJumpDiagnostics:
    code: str
    flags: tuple[str, ...] = ()
    n_jumps_20d: int = 0
    last_jump_ret: float | None = None
    last_jump_date: str | None = None
    n_splits: int = 0
    last_split_date: str | None = None
    last_split_raw_ret: float | None = None
    live_price_gap: float | None = None
    jump_dates: tuple[str, ...] = ()
    split_dates: tuple[str, ...] = ()


def jump_flag_config_from_raw(raw: dict[str, Any] | None) -> JumpFlagConfig:
    if not raw or not isinstance(raw, dict):
        return JumpFlagConfig()
    kwargs: dict[str, Any] = {}
    for key in JumpFlagConfig.__dataclass_fields__:
        if key in raw and raw[key] is not None:
            kwargs[key] = raw[key]
    return JumpFlagConfig(**kwargs)


def _is_near_split_ratio(raw_simple_ret: float, tol: float) -> bool:
    return any(abs(raw_simple_ret - target) <= tol for target in _SPLIT_RATIO_TARGETS)


def _date_str(value: object) -> str:
    return str(pd.Timestamp(value).date())


def diagnose_symbol_jumps(
    close: pd.Series,
    *,
    code: str,
    end: pd.Timestamp | None = None,
    cfg: JumpFlagConfig | None = None,
    live_close: float | None = None,
) -> SymbolJumpDiagnostics:
    """Classify split vs jump events on a single close series (raw or pre-qfq)."""
    cfg = cfg or JumpFlagConfig()
    series = pd.to_numeric(close, errors="coerce").dropna().astype(float)
    if end is not None:
        series = series.loc[series.index <= pd.Timestamp(end)]
    if series.size < 2:
        return SymbolJumpDiagnostics(code=code)

    values = series.to_numpy(dtype=float)
    index = series.index
    _, qfq_jumps = _compute_cum_factors(values, cfg.split_threshold)

    split_dates: list[str] = []
    split_rets: list[float] = []
    for jump in qfq_jumps:
        t = int(jump["_index"])
        raw_ret = float(jump["raw_ratio_minus_1"])
        if _is_near_split_ratio(raw_ret, cfg.split_ratio_tol) or abs(raw_ret) > cfg.split_threshold:
            split_dates.append(_date_str(index[t]))
            split_rets.append(raw_ret)

    # Genuine jumps on log returns; ignore days already tagged as splits.
    split_idx = {int(j["_index"]) for j in qfq_jumps if abs(float(j["raw_ratio_minus_1"])) > cfg.split_threshold}
    log_rets = np.diff(np.log(values))
    jump_dates: list[str] = []
    jump_rets: list[float] = []
    for i, lr in enumerate(log_rets):
        day_idx = i + 1  # return from t-1 -> t uses close[t]
        if day_idx in split_idx:
            continue
        if abs(float(lr)) >= cfg.jump_threshold:
            jump_dates.append(_date_str(index[day_idx]))
            jump_rets.append(float(lr))

    end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp(index[-1])
    lookback_start = end_ts - pd.tseries.offsets.BDay(cfg.recent_lookback)
    recent_jumps = [
        (d, r)
        for d, r in zip(jump_dates, jump_rets)
        if pd.Timestamp(d) >= lookback_start and pd.Timestamp(d) <= end_ts
    ]

    flags: list[str] = []
    if split_dates:
        flags.append("corp_split")
    if recent_jumps:
        flags.append("recent_jump")
    if jump_dates and not recent_jumps:
        flags.append("jump_day_hist")

    live_gap: float | None = None
    if live_close is not None and math.isfinite(live_close) and live_close > 0:
        last = float(values[-1])
        if last > 0:
            live_gap = live_close / last - 1.0
            if abs(live_gap) > cfg.live_gap_threshold:
                flags.append("live_price_gap")

    last_jump_ret = recent_jumps[-1][1] if recent_jumps else (jump_rets[-1] if jump_rets else None)
    last_jump_date = recent_jumps[-1][0] if recent_jumps else (jump_dates[-1] if jump_dates else None)

    return SymbolJumpDiagnostics(
        code=code,
        flags=tuple(flags),
        n_jumps_20d=len(recent_jumps),
        last_jump_ret=last_jump_ret,
        last_jump_date=last_jump_date,
        n_splits=len(split_dates),
        last_split_date=split_dates[-1] if split_dates else None,
        last_split_raw_ret=split_rets[-1] if split_rets else None,
        live_price_gap=live_gap,
        jump_dates=tuple(jump_dates),
        split_dates=tuple(split_dates),
    )


def build_jump_diagnostics_frame(
    close_panel: pd.DataFrame,
    *,
    as_of: str,
    codes: list[str] | tuple[str, ...] | None = None,
    names: dict[str, str] | None = None,
    live_closes: dict[str, float] | None = None,
    cfg: JumpFlagConfig | None = None,
) -> pd.DataFrame:
    """Build sidecar diagnostics table for a universe panel."""
    cfg = cfg or JumpFlagConfig()
    names = names or {}
    live_closes = live_closes or {}
    end = pd.Timestamp(as_of)
    cols = list(codes) if codes is not None else [str(c) for c in close_panel.columns]
    rows: list[dict[str, Any]] = []
    for code in cols:
        if code not in close_panel.columns:
            rows.append(
                {
                    "code": code,
                    "name": names.get(code, code),
                    "flags": "missing_price",
                    "n_jumps_20d": 0,
                    "last_jump_ret": None,
                    "last_jump_date": None,
                    "n_splits": 0,
                    "last_split_date": None,
                    "last_split_raw_ret": None,
                    "live_price_gap": None,
                }
            )
            continue
        series = pd.to_numeric(close_panel[code], errors="coerce").dropna()
        diag = diagnose_symbol_jumps(
            series,
            code=code,
            end=end,
            cfg=cfg,
            live_close=live_closes.get(code),
        )
        rows.append(
            {
                "code": code,
                "name": names.get(code, code),
                "flags": "|".join(diag.flags) if diag.flags else "",
                "n_jumps_20d": diag.n_jumps_20d,
                "last_jump_ret": diag.last_jump_ret,
                "last_jump_date": diag.last_jump_date,
                "n_splits": diag.n_splits,
                "last_split_date": diag.last_split_date,
                "last_split_raw_ret": diag.last_split_raw_ret,
                "live_price_gap": diag.live_price_gap,
            }
        )
    return pd.DataFrame(rows)


def split_day_mask(close: pd.Series, *, threshold: float = DEFAULT_QFQ_THRESHOLD) -> pd.Series:
    """Boolean mask (aligned to close index) True on corporate-split days."""
    series = pd.to_numeric(close, errors="coerce").astype(float)
    values = series.to_numpy()
    mask = np.zeros(len(values), dtype=bool)
    if len(values) < 2:
        return pd.Series(mask, index=series.index)
    _, jumps = _compute_cum_factors(values, threshold)
    for jump in jumps:
        t = int(jump["_index"])
        if 0 <= t < len(mask):
            mask[t] = True
    return pd.Series(mask, index=series.index)


def drop_split_log_returns(
    close: pd.Series,
    *,
    end: pd.Timestamp,
    lookback_days: int,
    split_threshold: float = DEFAULT_QFQ_THRESHOLD,
) -> np.ndarray:
    """Log returns with corporate-split days removed (for HMM training only)."""
    available = pd.to_numeric(close, errors="coerce").dropna()
    available = available.loc[available.index <= pd.Timestamp(end)]
    if available.size < 2:
        return np.array([], dtype=float)
    if lookback_days > 0 and available.size > lookback_days + 1:
        available = available.iloc[-(lookback_days + 1) :]
    mask = split_day_mask(available, threshold=split_threshold)
    # return at t uses close[t-1], close[t]; drop when day t is a split day
    keep = ~mask.to_numpy()[1:]
    log_rets = np.diff(np.log(available.to_numpy(dtype=float)))
    if keep.size != log_rets.size:
        return log_rets
    return log_rets[keep]


def format_jump_diagnostics_report_section(diag_frame: pd.DataFrame, *, max_rows: int = 12) -> str:
    """Markdown subsection for GARCH REPORT (does not change CSV schema)."""
    lines = [
        "## 拆分 / 跳跃诊断（旁路，不影响主 CSV 列）",
        "",
        "拆分（`corp_split`）与真实大幅波动（`recent_jump`）分开标注；",
        "主 GARCH CSV 列集不变。完整表见同目录 `garch_jump_diagnostics.csv`。",
        "",
    ]
    if diag_frame is None or diag_frame.empty:
        lines.append("*无诊断行。*")
        lines.append("")
        return "\n".join(lines)

    flagged = diag_frame.copy()
    flagged["_has"] = flagged["flags"].astype(str).str.len() > 0
    flagged = flagged.loc[flagged["_has"]].copy()
    if flagged.empty:
        lines.append("*本 universe 未检出拆分或近期跳跃。*")
        lines.append("")
        return "\n".join(lines)

    lines.append("| code | name | flags | n_jumps_20d | last_jump_ret | last_split_date |")
    lines.append("|------|------|-------|-------------|---------------|-----------------|")
    for _, row in flagged.head(max_rows).iterrows():
        lj = row.get("last_jump_ret")
        lj_s = f"{float(lj):.2%}" if lj is not None and pd.notna(lj) else "—"
        lines.append(
            f"| {row['code']} | {row.get('name', '')} | {row.get('flags') or '—'} | "
            f"{int(row.get('n_jumps_20d') or 0)} | {lj_s} | {row.get('last_split_date') or '—'} |"
        )
    if len(flagged) > max_rows:
        lines.append(f"| … | 另有 {len(flagged) - max_rows} 只 | | | | |")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "JumpFlagConfig",
    "SymbolJumpDiagnostics",
    "build_jump_diagnostics_frame",
    "diagnose_symbol_jumps",
    "drop_split_log_returns",
    "format_jump_diagnostics_report_section",
    "jump_flag_config_from_raw",
    "split_day_mask",
]
