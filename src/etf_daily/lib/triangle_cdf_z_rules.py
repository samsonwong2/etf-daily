"""Causal triangle rules combining pred_cdf tails with return_z filters.

Rules (long-only observational):
- Baseline q*: cdf >= q* (up) or cdf <= 1-q* (down)
- Strict AND at soft_q: up requires cdf>=soft_q AND return_z>=z_up;
  down requires cdf<=1-soft_q AND return_z<=-z_down
- Rescue OR: baseline OR (soft-q tail AND |return_z| gate)

Direction diagnostic (post-hoc, not a trade signal):
- up/green alerts align with top_reversal events
- down/red alerts align with bottom_reversal events
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_BASELINE_Q = 0.90
DEFAULT_SOFT_Q = 0.80
DEFAULT_Z_GRID = tuple(round(0.5 + 0.25 * i, 2) for i in range(11))  # 0.5..3.0


@dataclass(frozen=True)
class CdfZRule:
    """Joint CDF + return_z triangle rule."""

    mode: str  # baseline | and_soft | or_rescue
    baseline_q: float = DEFAULT_BASELINE_Q
    soft_q: float = DEFAULT_SOFT_Q
    z_up: float = 1.5
    z_down: float = 1.5

    def label(self) -> str:
        if self.mode == "baseline":
            return f"baseline_q{self.baseline_q:g}"
        if self.mode == "and_soft":
            return (
                f"and_soft_q{self.soft_q:g}_zup{self.z_up:g}_zdn{self.z_down:g}"
            )
        if self.mode == "or_rescue":
            return (
                f"or_rescue_q{self.baseline_q:g}|soft{self.soft_q:g}"
                f"_zup{self.z_up:g}_zdn{self.z_down:g}"
            )
        return f"unknown_{self.mode}"

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["label"] = self.label()
        return out


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def cdf_up_mask(pred_cdf: pd.Series, q_up: float) -> pd.Series:
    return _num(pred_cdf).ge(float(q_up)).fillna(False)


def cdf_down_mask(pred_cdf: pd.Series, q_down: float) -> pd.Series:
    return _num(pred_cdf).le(float(q_down)).fillna(False)


def z_up_mask(return_z: pd.Series, z_up: float) -> pd.Series:
    return _num(return_z).ge(float(z_up)).fillna(False)


def z_down_mask(return_z: pd.Series, z_down: float) -> pd.Series:
    return _num(return_z).le(-float(z_down)).fillna(False)


def switch_sides_from_masks(
    up: pd.Series,
    down: pd.Series,
) -> pd.Series:
    """Return switch_side labels; prefer up when both true (rare)."""
    side = pd.Series(pd.NA, index=up.index, dtype=object)
    side = side.mask(down.fillna(False), "down")
    side = side.mask(up.fillna(False), "up")
    return side


def build_rule_masks(
    frame: pd.DataFrame,
    rule: CdfZRule,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (alert_mask, up_mask, down_mask) for ``rule``."""
    if "pred_cdf" not in frame.columns:
        raise ValueError("frame missing pred_cdf")
    if rule.mode != "baseline" and "return_z" not in frame.columns:
        raise ValueError("frame missing return_z")

    cdf = frame["pred_cdf"]
    z = frame["return_z"] if "return_z" in frame.columns else pd.Series(np.nan, index=frame.index)

    base_up = cdf_up_mask(cdf, rule.baseline_q)
    base_dn = cdf_down_mask(cdf, 1.0 - rule.baseline_q)
    soft_up = cdf_up_mask(cdf, rule.soft_q) & z_up_mask(z, rule.z_up)
    soft_dn = cdf_down_mask(cdf, 1.0 - rule.soft_q) & z_down_mask(z, rule.z_down)

    if rule.mode == "baseline":
        up = base_up
        down = base_dn
    elif rule.mode == "and_soft":
        up = soft_up
        down = soft_dn
    elif rule.mode == "or_rescue":
        up = base_up | soft_up
        down = base_dn | soft_dn
    else:
        raise ValueError(f"unknown mode: {rule.mode}")

    # Mutually prefer stronger absolute z when both sides fire; else prefer up.
    both = up & down
    if both.any() and "return_z" in frame.columns:
        zz = _num(z)
        prefer_down = both & zz.notna() & (zz.abs() >= 0) & (zz < 0)
        up = up & ~prefer_down
        down = down | prefer_down
        # remaining both (z nan or zero): keep up
        both2 = up & down
        down = down & ~both2

    alert = (up | down).fillna(False)
    return alert, up.fillna(False), down.fillna(False)


def annotate_rule_columns(
    frame: pd.DataFrame,
    rule: CdfZRule,
    *,
    prefix: str = "rule",
) -> pd.DataFrame:
    """Copy frame and attach rule alert / side columns."""
    out = frame.copy()
    alert, up, down = build_rule_masks(out, rule)
    out[f"{prefix}_alert"] = alert
    out[f"{prefix}_up"] = up
    out[f"{prefix}_down"] = down
    out[f"{prefix}_side"] = switch_sides_from_masks(up, down)
    out[f"{prefix}_label"] = rule.label()
    return out


def default_z_grid(start: float = 0.5, stop: float = 3.0, step: float = 0.25) -> tuple[float, ...]:
    if step <= 0 or stop < start:
        return ()
    n = int(round((stop - start) / step)) + 1
    return tuple(round(start + i * step, 2) for i in range(n))


def iter_cdf_z_rules(
    *,
    baseline_q: float = DEFAULT_BASELINE_Q,
    soft_q: float = DEFAULT_SOFT_Q,
    z_values: Sequence[float] | None = None,
    modes: Iterable[str] = ("and_soft", "or_rescue"),
) -> list[CdfZRule]:
    """Cartesian product of modes × z_up × z_down (includes asymmetric)."""
    zs = list(z_values) if z_values is not None else list(default_z_grid())
    rules: list[CdfZRule] = [
        CdfZRule(mode="baseline", baseline_q=baseline_q, soft_q=soft_q, z_up=0.0, z_down=0.0)
    ]
    for mode in modes:
        if mode == "baseline":
            continue
        for z_up in zs:
            for z_down in zs:
                rules.append(
                    CdfZRule(
                        mode=str(mode),
                        baseline_q=float(baseline_q),
                        soft_q=float(soft_q),
                        z_up=float(z_up),
                        z_down=float(z_down),
                    )
                )
    return rules


def direction_aligned_event_stats(
    *,
    events: Sequence[Any],
    mature: pd.DataFrame,
    up_mask: pd.Series,
    down_mask: pd.Series,
    lead: int = 3,
    horizon: int = 10,
) -> dict[str, Any]:
    """Directional coverage of top/bottom events by up/down alerts.

    An event is direction-covered if any same-side alert falls in
    [onset-lead, onset+horizon] on that code's calendar.
    """
    if mature.empty or not events:
        return {
            "n_top_events": 0,
            "n_bottom_events": 0,
            "n_top_covered_by_up": 0,
            "n_bottom_covered_by_down": 0,
            "top_up_recall": None,
            "bottom_down_recall": None,
            "direction_recall": None,
            "n_direction_events": 0,
            "n_direction_covered": 0,
        }

    work = mature.copy()
    work["as_of"] = pd.to_datetime(work["as_of"]).dt.normalize()
    up_idx = set(
        zip(
            work.loc[up_mask.reindex(work.index).fillna(False), "code"].astype(str),
            work.loc[up_mask.reindex(work.index).fillna(False), "as_of"],
        )
    )
    dn_idx = set(
        zip(
            work.loc[down_mask.reindex(work.index).fillna(False), "code"].astype(str),
            work.loc[down_mask.reindex(work.index).fillna(False), "as_of"],
        )
    )
    calendars: dict[str, list[pd.Timestamp]] = {}
    for code, grp in work.groupby("code", sort=False):
        calendars[str(code)] = sorted(pd.to_datetime(grp["as_of"]).dt.normalize().unique())

    n_top = n_bot = 0
    n_top_hit = n_bot_hit = 0
    for ev in events:
        code = str(ev.code)
        onset = pd.Timestamp(ev.onset).normalize()
        sub = work[(work["code"] == code) & (work["as_of"] == onset)]
        turn = str(sub["turn_kind"].iloc[0]) if not sub.empty else "none"
        dates = calendars.get(code, [])
        if onset not in dates:
            continue
        i = dates.index(onset)
        lo = max(0, i - int(lead))
        hi = min(len(dates) - 1, i + int(horizon))
        window = dates[lo : hi + 1]
        if turn == "top_reversal":
            n_top += 1
            if any((code, d) in up_idx for d in window):
                n_top_hit += 1
        elif turn == "bottom_reversal":
            n_bot += 1
            if any((code, d) in dn_idx for d in window):
                n_bot_hit += 1

    n_dir = n_top + n_bot
    n_hit = n_top_hit + n_bot_hit
    return {
        "n_top_events": n_top,
        "n_bottom_events": n_bot,
        "n_top_covered_by_up": n_top_hit,
        "n_bottom_covered_by_down": n_bot_hit,
        "top_up_recall": (n_top_hit / n_top) if n_top else None,
        "bottom_down_recall": (n_bot_hit / n_bot) if n_bot else None,
        "direction_recall": (n_hit / n_dir) if n_dir else None,
        "n_direction_events": n_dir,
        "n_direction_covered": n_hit,
    }


__all__ = [
    "CdfZRule",
    "DEFAULT_BASELINE_Q",
    "DEFAULT_SOFT_Q",
    "DEFAULT_Z_GRID",
    "annotate_rule_columns",
    "build_rule_masks",
    "cdf_down_mask",
    "cdf_up_mask",
    "default_z_grid",
    "direction_aligned_event_stats",
    "iter_cdf_z_rules",
    "switch_sides_from_masks",
    "z_down_mask",
    "z_up_mask",
]
