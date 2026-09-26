"""Early-down diagnostic / alert layer (display-only).

Does **not** mutate regime labels or trading actions. Complements
``ma_stack_strict`` when price is already grinding lower under MA20 with
weak prior60, but the production model still waits for a full bear stack
(``MA20 < MA60``) — e.g. SH515060 2026-05-13→06-05.

Rules (from 46-ETF A/B focus winners; production unchanged)
----------------------------------------------------------
- soft_bear: ``C < MA20 & prior60 ≤ -6% & prior20 ≤ 0``
- peak8:     ``C < MA20 & prior60 ≤ -6% & dd20_from_20d_high ≤ -8%``
- classic:   ``MA20 < MA60 & C < MA20 & prior60 ≤ -6% & prior20 ≤ 0``
             (same as production strict down; for lag comparison only)

Severity
--------
- alert: early rule fires and production regime is still ``range``/``up``
- watch: early rule fires and production regime is already ``down``
- none:  no early rule
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

EarlyDownLevel = Literal["none", "watch", "alert"]

EARLY_DOWN_COLUMNS: tuple[str, ...] = (
    "code",
    "name",
    "method",
    "as_of",
    "regime_asof",
    "last_close",
    "soft_bear",
    "peak8",
    "classic_bear",
    "early_down",
    "level",
    "reasons",
    "prior20",
    "prior60",
    "dd20_high",
    "ma20",
    "ma60",
    "c_below_ma20",
    "bear_stack",
    "note",
)


@dataclass(frozen=True)
class EarlyDownFlags:
    soft_bear: bool
    peak8: bool
    classic_bear: bool
    prior20: float
    prior60: float
    dd20_high: float
    close: float
    ma20: float
    ma60: float

    @property
    def early_down(self) -> bool:
        return bool(self.soft_bear or self.peak8)

    @property
    def c_below_ma20(self) -> bool:
        return (
            np.isfinite(self.close)
            and np.isfinite(self.ma20)
            and self.close < self.ma20
        )

    @property
    def bear_stack(self) -> bool:
        return (
            np.isfinite(self.ma20)
            and np.isfinite(self.ma60)
            and self.ma20 < self.ma60
        )

    def reasons(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.soft_bear:
            out.append("soft_bear")
        if self.peak8:
            out.append("peak8")
        if self.classic_bear:
            out.append("classic_bear")
        return tuple(out)


def _f(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def rolling_dd20_high(close: np.ndarray | pd.Series) -> np.ndarray:
    c = np.asarray(close, dtype=float)
    hi = pd.Series(c).rolling(20, min_periods=5).max().to_numpy(float)
    out = np.full(len(c), np.nan, dtype=float)
    ok = np.isfinite(c) & np.isfinite(hi) & (hi > 0)
    out[ok] = c[ok] / hi[ok] - 1.0
    return out


def flags_from_values(
    *,
    close: float,
    ma20: float,
    ma60: float,
    prior20: float,
    prior60: float,
    dd20_high: float,
) -> EarlyDownFlags:
    c, m20, m60 = _f(close), _f(ma20), _f(ma60)
    p20, p60 = _f(prior20, 0.0), _f(prior60, 0.0)
    dd = _f(dd20_high)
    below = np.isfinite(c) and np.isfinite(m20) and c < m20
    bear = np.isfinite(m20) and np.isfinite(m60) and m20 < m60
    soft = below and p60 <= -0.06 and p20 <= 0.0
    peak = below and p60 <= -0.06 and np.isfinite(dd) and dd <= -0.08
    classic = bear and below and p60 <= -0.06 and p20 <= 0.0
    return EarlyDownFlags(
        soft_bear=bool(soft),
        peak8=bool(peak),
        classic_bear=bool(classic),
        prior20=float(p20),
        prior60=float(p60),
        dd20_high=float(dd) if np.isfinite(dd) else float("nan"),
        close=float(c) if np.isfinite(c) else float("nan"),
        ma20=float(m20) if np.isfinite(m20) else float("nan"),
        ma60=float(m60) if np.isfinite(m60) else float("nan"),
    )


def flags_at_index(px: pd.DataFrame, idx: int = -1) -> EarlyDownFlags:
    """Compute flags for one bar of a ``prepare_features`` frame."""
    if px is None or len(px) == 0:
        return flags_from_values(
            close=np.nan,
            ma20=np.nan,
            ma60=np.nan,
            prior20=0.0,
            prior60=0.0,
            dd20_high=np.nan,
        )
    i = idx if idx >= 0 else len(px) + idx
    i = max(0, min(i, len(px) - 1))
    dd = rolling_dd20_high(px["$close"].to_numpy(float))
    row = px.iloc[i]
    return flags_from_values(
        close=row["$close"],
        ma20=row["MA20"],
        ma60=row["MA60"],
        prior20=row["prior20"] if pd.notna(row.get("prior20")) else 0.0,
        prior60=row["prior60"] if pd.notna(row.get("prior60")) else 0.0,
        dd20_high=dd[i],
    )


def classify_early_down_level(
    flags: EarlyDownFlags,
    regime_asof: str | None,
) -> EarlyDownLevel:
    if not flags.early_down:
        return "none"
    regime = str(regime_asof or "range").lower()
    if regime == "down":
        return "watch"
    return "alert"


def note_for(level: EarlyDownLevel, flags: EarlyDownFlags, regime_asof: str | None) -> str:
    if level == "none":
        return "no_early_down"
    regime = str(regime_asof or "range")
    if level == "alert":
        if flags.early_down and not flags.classic_bear:
            return (
                f"price_grind_down_pre_stack; model_regime={regime}; "
                "diagnostic_only_no_trade_change"
            )
        return (
            f"early_down_while_model_{regime}; "
            "diagnostic_only_no_trade_change"
        )
    return "early_down_aligned_with_model_down; diagnostic_only"


def build_early_down_alert_row(
    *,
    code: str,
    name: str,
    method: str,
    as_of: str,
    regime_asof: str,
    px: pd.DataFrame,
    idx: int = -1,
) -> dict[str, Any]:
    flags = flags_at_index(px, idx=idx)
    level = classify_early_down_level(flags, regime_asof)
    return {
        "code": str(code).upper(),
        "name": str(name),
        "method": str(method),
        "as_of": str(as_of),
        "regime_asof": str(regime_asof),
        "last_close": flags.close,
        "soft_bear": bool(flags.soft_bear),
        "peak8": bool(flags.peak8),
        "classic_bear": bool(flags.classic_bear),
        "early_down": bool(flags.early_down),
        "level": level,
        "reasons": "|".join(flags.reasons()),
        "prior20": flags.prior20,
        "prior60": flags.prior60,
        "dd20_high": flags.dd20_high,
        "ma20": flags.ma20,
        "ma60": flags.ma60,
        "c_below_ma20": bool(flags.c_below_ma20),
        "bear_stack": bool(flags.bear_stack),
        "note": note_for(level, flags, regime_asof),
    }


def build_early_down_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=list(EARLY_DOWN_COLUMNS))
    out = frame.reindex(columns=list(EARLY_DOWN_COLUMNS) + [
        c for c in frame.columns if c not in EARLY_DOWN_COLUMNS
    ])
    level_order = {"alert": 0, "watch": 1, "none": 2}
    out["_sort"] = out["level"].map(level_order).fillna(9)
    out = out.sort_values(["_sort", "code"], kind="stable").drop(columns=["_sort"])
    return out.reset_index(drop=True)


def early_down_flags_frame(
    px: pd.DataFrame,
    *,
    regimes: np.ndarray | list[str] | None = None,
) -> pd.DataFrame:
    """Per-bar early-down flags aligned to a ``prepare_features`` frame."""
    n = len(px)
    if n == 0:
        return pd.DataFrame(
            columns=[
                "as_of",
                "regime",
                "soft_bear",
                "peak8",
                "classic_bear",
                "early_down",
                "level",
                "reasons",
                "prior20",
                "prior60",
                "dd20_high",
                "close",
                "ma20",
                "ma60",
            ]
        )
    close = px["$close"].to_numpy(float)
    ma20 = px["MA20"].to_numpy(float)
    ma60 = px["MA60"].to_numpy(float)
    p20 = pd.to_numeric(px.get("prior20"), errors="coerce").fillna(0.0).to_numpy(float)
    p60 = pd.to_numeric(px.get("prior60"), errors="coerce").fillna(0.0).to_numpy(float)
    dd = rolling_dd20_high(close)
    if regimes is not None:
        labs = np.asarray(regimes, dtype=object)
    elif "regime" in px.columns:
        labs = px["regime"].to_numpy(object)
    else:
        labs = np.array(["range"] * n, dtype=object)
    if "as_of" in px.columns:
        as_of = pd.to_datetime(px["as_of"]).dt.normalize()
    else:
        as_of = pd.RangeIndex(n)

    below = np.isfinite(close) & np.isfinite(ma20) & (close < ma20)
    bear = np.isfinite(ma20) & np.isfinite(ma60) & (ma20 < ma60)
    soft = below & (p60 <= -0.06) & (p20 <= 0.0)
    peak = below & (p60 <= -0.06) & np.isfinite(dd) & (dd <= -0.08)
    classic = bear & below & (p60 <= -0.06) & (p20 <= 0.0)
    early = soft | peak
    levels: list[str] = []
    reasons: list[str] = []
    for i in range(n):
        fl = EarlyDownFlags(
            soft_bear=bool(soft[i]),
            peak8=bool(peak[i]),
            classic_bear=bool(classic[i]),
            prior20=float(p20[i]),
            prior60=float(p60[i]),
            dd20_high=float(dd[i]) if np.isfinite(dd[i]) else float("nan"),
            close=float(close[i]) if np.isfinite(close[i]) else float("nan"),
            ma20=float(ma20[i]) if np.isfinite(ma20[i]) else float("nan"),
            ma60=float(ma60[i]) if np.isfinite(ma60[i]) else float("nan"),
        )
        levels.append(classify_early_down_level(fl, str(labs[i])))
        reasons.append("|".join(fl.reasons()))
    return pd.DataFrame(
        {
            "as_of": as_of,
            "regime": labs,
            "soft_bear": soft,
            "peak8": peak,
            "classic_bear": classic,
            "early_down": early,
            "level": levels,
            "reasons": reasons,
            "prior20": p20,
            "prior60": p60,
            "dd20_high": dd,
            "close": close,
            "ma20": ma20,
            "ma60": ma60,
        }
    )


def detect_early_down_runs(
    px: pd.DataFrame,
    *,
    regimes: np.ndarray | list[str] | None = None,
    levels: tuple[str, ...] = ("alert", "watch"),
) -> list[dict[str, Any]]:
    """Collapse consecutive early-down days into runs (for HTML bands / marks)."""
    frame = early_down_flags_frame(px, regimes=regimes)
    if frame.empty:
        return []
    want = set(levels)
    runs: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    for _, r in frame.iterrows():
        level = str(r["level"])
        d = pd.Timestamp(r["as_of"]).strftime("%Y-%m-%d")
        if level in want:
            if active is None or active["level"] != level:
                if active is not None:
                    runs.append(active)
                active = {
                    "level": level,
                    "start": d,
                    "end": d,
                    "n_bars": 1,
                    "reasons": str(r["reasons"] or ""),
                    "px_start": float(r["close"]) if pd.notna(r["close"]) else float("nan"),
                    "regime_start": str(r["regime"]),
                }
            else:
                active["end"] = d
                active["n_bars"] = int(active["n_bars"]) + 1
        elif active is not None:
            runs.append(active)
            active = None
    if active is not None:
        runs.append(active)
    return runs


def detect_early_down_marks(
    px: pd.DataFrame,
    *,
    regimes: np.ndarray | list[str] | None = None,
) -> list[dict[str, Any]]:
    """Visual-only marks at the start of each alert/watch run.

    Does not change regime labels or trades. ``kind`` is ``ed_alert`` / ``ed_watch``.
    """
    marks: list[dict[str, Any]] = []
    for run in detect_early_down_runs(px, regimes=regimes):
        level = str(run["level"])
        kind = "ed_alert" if level == "alert" else "ed_watch"
        title = "ED告警(模型滞后)" if kind == "ed_alert" else "ED观察(已对齐down)"
        tip = (
            "soft_bear/peak8 命中且模型仍非 down；诊断层，不改买卖"
            if kind == "ed_alert"
            else "soft_bear/peak8 命中且模型已是 down；诊断层"
        )
        marks.append(
            {
                "date": run["start"],
                "end": run["end"],
                "kind": kind,
                "level": level,
                "symbol": "ED",
                "title": title,
                "why": tip,
                "reasons": run.get("reasons") or "",
                "n_bars": int(run.get("n_bars") or 1),
                "px": run.get("px_start"),
                "regime": run.get("regime_start"),
            }
        )
    return marks


def render_early_down_markdown(
    frame: pd.DataFrame,
    *,
    meta: dict[str, Any] | None = None,
) -> str:
    """Standalone or embeddable markdown section."""
    meta = meta or {}
    as_of = str(meta.get("as_of") or "")
    trade_date = str(meta.get("trade_date") or "")
    export = frame.copy() if frame is not None else pd.DataFrame()
    if export.empty:
        export = pd.DataFrame(columns=list(EARLY_DOWN_COLUMNS))

    n_alert = int((export.get("level") == "alert").sum()) if len(export) else 0
    n_watch = int((export.get("level") == "watch").sum()) if len(export) else 0
    n_early = int(export.get("early_down", pd.Series(dtype=bool)).fillna(False).sum()) if len(export) else 0

    lines = [
        "## Early-down 诊断（告警层 · 不改 regime · 不产生买卖）",
        "",
        f"> soft_bear / peak8 仅作诊断；production `regime_asof` 与执行计划动作不变。"
        + (f" as_of=`{as_of}`" if as_of else "")
        + (f" · trade=`{trade_date}`" if trade_date else ""),
        "",
        f"- early_down 命中: **{n_early}** · alert（模型尚未 down）: **{n_alert}** · "
        f"watch（模型已是 down）: **{n_watch}**",
        "",
        "规则：`soft_bear = C<MA20 & p60≤-6% & p20≤0`；"
        "`peak8 = C<MA20 & p60≤-6% & 20日高点回撤≤-8%`。"
        "当命中且 `regime_asof∈{range,up}` → **alert**（典型：下跌通道但未形成空头排列）。",
        "",
    ]
    alerts = export.loc[export["level"] == "alert"] if len(export) else export.iloc[0:0]
    if alerts.empty:
        lines.extend(["### Alert（模型滞后）", "", "（无）", ""])
    else:
        lines.extend(
            [
                "### Alert（模型滞后）",
                "",
                "| 代码 | 名称 | 方法 | regime | close | p20 | p60 | dd20 | reasons | note |",
                "|---|---|---|---|---:|---:|---:|---:|---|---|",
            ]
        )
        for _, r in alerts.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(r.get("code")),
                        str(r.get("name")),
                        str(r.get("method")),
                        str(r.get("regime_asof")),
                        f"{float(r['last_close']):.3f}" if pd.notna(r.get("last_close")) else "—",
                        f"{float(r['prior20']):+.1%}" if pd.notna(r.get("prior20")) else "—",
                        f"{float(r['prior60']):+.1%}" if pd.notna(r.get("prior60")) else "—",
                        f"{float(r['dd20_high']):+.1%}" if pd.notna(r.get("dd20_high")) else "—",
                        str(r.get("reasons") or "—"),
                        str(r.get("note") or ""),
                    ]
                )
                + " |"
            )
        lines.append("")

    watches = export.loc[export["level"] == "watch"] if len(export) else export.iloc[0:0]
    if watches.empty:
        lines.extend(["### Watch（已与 down 对齐）", "", "（无）", ""])
    else:
        lines.extend(
            [
                "### Watch（已与 down 对齐）",
                "",
                "| 代码 | 名称 | regime | reasons |",
                "|---|---|---|---|",
            ]
        )
        for _, r in watches.iterrows():
            lines.append(
                f"| {r.get('code')} | {r.get('name')} | {r.get('regime_asof')} | "
                f"{r.get('reasons') or '—'} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_early_down_artifacts(
    frame: pd.DataFrame,
    out_dir: Path,
    *,
    meta: dict[str, Any] | None = None,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})
    meta.setdefault("layer", "early_down_diagnostic")
    meta.setdefault("mutates_regime", False)
    meta.setdefault("mutates_trades", False)

    csv_path = out_dir / "early_down_alerts.csv"
    json_path = out_dir / "early_down_alerts.json"
    md_path = out_dir / "early_down_alerts.md"

    export = build_early_down_frame(
        frame.to_dict(orient="records") if frame is not None and len(frame) else []
    )
    export.to_csv(csv_path, index=False, encoding="utf-8-sig")
    payload = {
        "meta": meta,
        "summary": {
            "n": int(len(export)),
            "n_early_down": int(export["early_down"].fillna(False).sum()) if len(export) else 0,
            "n_alert": int((export["level"] == "alert").sum()) if len(export) else 0,
            "n_watch": int((export["level"] == "watch").sum()) if len(export) else 0,
        },
        "rows": export.replace({np.nan: None}).to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(
        "# Early-down 诊断告警\n\n" + render_early_down_markdown(export, meta=meta),
        encoding="utf-8",
    )
    return {"csv": csv_path, "json": json_path, "markdown": md_path}


__all__ = [
    "EARLY_DOWN_COLUMNS",
    "EarlyDownFlags",
    "EarlyDownLevel",
    "build_early_down_alert_row",
    "build_early_down_frame",
    "classify_early_down_level",
    "detect_early_down_marks",
    "detect_early_down_runs",
    "early_down_flags_frame",
    "flags_at_index",
    "flags_from_values",
    "render_early_down_markdown",
    "rolling_dd20_high",
    "write_early_down_artifacts",
]
