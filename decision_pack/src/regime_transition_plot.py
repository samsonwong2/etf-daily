"""Helpers to overlay regime-transition validation signals on K-line charts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

DEFAULT_VALIDATION_DIR = Path(
    "workspace/decision_packs/20260720/regime_transition_validation_q90"
)

OOS_KEEP = (
    "as_of",
    "code",
    "name",
    "signal_level",
    "quantile_switch",
    "model_quantile_switch",
    "extreme_drop_switch",
    "switch_side",
    "switch_source",
    "day_ret",
    "pred_cdf",
    "pred_mean",
    "pred_mean_next",
    "pred_scale_next",
    "return_z",
    "p_switch_hmm",
    "p_into_reversal_hmm",
    "transition_score",
    "regime_now",
)

REVERSAL_KEEP = (
    "as_of",
    "code",
    "label_switch_reversal",
    "turn_kind",
    "jump_role",
    "prior_ret",
    "jump_ret",
    "post_ret",
    "final_confirmed_on",
    "early_label_switch_reversal",
    "early_turn_kind",
    "early_jump_role",
    "early_prior_ret",
    "early_jump_ret",
    "early_post_ret",
    "early_horizon",
    "early_confirmed_on",
    "early_to_final_status",
    "causal_prior_ret",
    "causal_jump_ret",
    "causal_thr_prior",
    "causal_prior_strong",
)

# Triangle bias wording mode.
# Default is observe_only: 上行/下行分位切换 with no 偏买/偏卖.
# Tradable (偏买)/(偏卖) suffixes appear only via lockbox bias_promotion.json.
# Walk-forward trade_action is hover-only reference, never triangle wording.
DEFAULT_TRIANGLE_BIAS_RULE = "observe_only"


EVENT_KEEP = (
    "as_of",
    "code",
    "matched_event_onset",
    "detection_delay",
    "is_duplicate",
    "is_fp",
)


def _require_columns(frame: pd.DataFrame, cols: tuple[str, ...], label: str) -> None:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _normalize_as_of(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["as_of"] = pd.to_datetime(out["as_of"]).dt.normalize()
    out["code"] = out["code"].astype(str).str.upper()
    return out


def load_validation_csvs(
    validation_dir: Path | str,
    *,
    oos_name: str = "signals_oos.csv",
    reversal_name: str = "signals_with_reversal_labels.csv",
    event_name: str = "event_metrics_reversal.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(validation_dir)
    oos_path = root / oos_name
    rev_path = root / reversal_name
    evt_path = root / event_name
    for path in (oos_path, rev_path, evt_path):
        if not path.exists():
            raise FileNotFoundError(f"missing validation CSV: {path}")
    oos = pd.read_csv(oos_path)
    rev = pd.read_csv(rev_path)
    evt = pd.read_csv(evt_path)
    _require_columns(oos, ("as_of", "code", "quantile_switch"), "signals_oos")
    _require_columns(rev, ("as_of", "code", "turn_kind"), "signals_with_reversal_labels")
    _require_columns(evt, ("as_of", "code"), "event_metrics_reversal")
    return _normalize_as_of(oos), _normalize_as_of(rev), _normalize_as_of(evt)


def dedupe_event_rows(events: pd.DataFrame) -> pd.DataFrame:
    """Keep one event-match row per (as_of, code). Prefer matched non-duplicate hits."""
    if events.empty:
        return events.copy()
    frame = events.copy()
    if "matched_event_onset" not in frame.columns:
        frame["matched_event_onset"] = pd.NaT
    if "is_duplicate" not in frame.columns:
        frame["is_duplicate"] = False
    if "is_fp" not in frame.columns:
        frame["is_fp"] = False
    frame["is_duplicate"] = frame["is_duplicate"].fillna(False).astype(bool)
    frame["is_fp"] = frame["is_fp"].fillna(False).astype(bool)
    frame["_matched"] = frame["matched_event_onset"].notna().astype(int)
    frame["_not_dup"] = (~frame["is_duplicate"]).astype(int)
    frame["_not_fp"] = (~frame["is_fp"]).astype(int)
    frame = frame.sort_values(
        ["code", "as_of", "_matched", "_not_dup", "_not_fp"],
        ascending=[True, True, False, False, False],
        kind="stable",
    )
    out = frame.drop_duplicates(subset=["as_of", "code"], keep="first")
    return out.drop(columns=["_matched", "_not_dup", "_not_fp"], errors="ignore")


def causal_action_label(
    *,
    quantile_switch: bool,
    switch_side: str | None,
    prior_ret: float | None = None,
    thr_prior: float | None = None,
    bias_rule: str | None = None,
    trade_action: str | None = None,
    promotion: Mapping[str, Any] | None = None,
) -> str:
    """EOD-causal action from quantile switch (+ lockbox-gated bias wording).

    Default is direction-only (``observe_only``). Tradable ``(偏买)`` / ``(偏卖)``
    suffixes appear only when ``promotion["promotion_source"] == "lockbox"`` and
    the corresponding side passes. ``bias_rule`` is ignored for main wording
    (kept for API compatibility). ``trade_action`` is hover-only.
    Future ``turn_kind``/``post_ret`` are never used here.
    """
    from decision_pack.src.regime_transition_bias_promotion import eval_rule_bias

    _ = bias_rule  # no longer drives triangle main wording
    side = str(switch_side or "")
    if not quantile_switch or side not in {"up", "down"}:
        return "无切换"
    direction = "上行分位切换" if side == "up" else "下行分位切换"
    promo = promotion or {}
    if str(promo.get("promotion_source") or "none") != "lockbox":
        return direction

    labels: set[str] = set()
    if promo.get("buy_pass_gate") and promo.get("buy_rule"):
        bias = eval_rule_bias(
            str(promo.get("buy_rule")),
            quantile_switch=True,
            switch_side=side,
            prior_ret=prior_ret,
            thr_prior=thr_prior,
            trade_action=trade_action,
        )
        if bias == "buy_bias":
            labels.add("buy")
    if promo.get("sell_pass_gate") and promo.get("sell_rule"):
        bias = eval_rule_bias(
            str(promo.get("sell_rule")),
            quantile_switch=True,
            switch_side=side,
            prior_ret=prior_ret,
            thr_prior=thr_prior,
            trade_action=trade_action,
        )
        if bias == "sell_bias":
            labels.add("sell")
    if labels == {"buy"}:
        return f"{direction}(偏买)"
    if labels == {"sell"}:
        return f"{direction}(偏卖)"
    return direction


def posthoc_reversal_label(
    *,
    turn_kind: str | None,
    jump_role: str | None = None,
    stage: str = "final",
) -> str:
    """Post-hoc reversal label for diamond markers (may use future path)."""
    turn = str(turn_kind or "none")
    role = str(jump_role or "none")
    stage_s = str(stage or "final")
    prefix = {
        "early": "早期确认",
        "final": "最终确认",
        "revoked": "早期确认已撤销",
    }.get(stage_s, "事后验证")
    if stage_s == "revoked":
        return f"{prefix}(非最终反转)"
    if turn == "bottom_reversal":
        if role == "bounce":
            return f"{prefix}:底部反转(偏买)"
        return f"{prefix}:底部反转"
    if turn == "top_reversal":
        if role in {"break", "climax_top"}:
            return f"{prefix}:顶部反转(偏卖)"
        return f"{prefix}:顶部反转"
    return "无反转标签"


def validation_action_label(
    *,
    quantile_switch: bool,
    switch_side: str | None,
    turn_kind: str | None = None,
    jump_role: str | None = None,
) -> str:
    """Causal action label only.

    ``turn_kind`` / ``jump_role`` are accepted for backward compatibility but
    ignored — post-hoc reversal text belongs on diamond markers via
    ``posthoc_reversal_label``.
    """
    del turn_kind, jump_role
    return causal_action_label(
        quantile_switch=quantile_switch, switch_side=switch_side
    )


def triangle_legend_name(switch_side: str) -> str:
    """Legend text for causal triangles (direction only; bias is per-point)."""
    side = str(switch_side)
    if side == "up":
        return "上行切换"
    if side == "down":
        return "下行切换"
    return "无切换"

def _switch_marker_kind(quantile_switch: bool, switch_side: Any) -> str:
    if not quantile_switch:
        return "none"
    side = "" if pd.isna(switch_side) else str(switch_side)
    if side == "up":
        return "switch_up"
    if side == "down":
        return "switch_down"
    return "none"


def _reversal_marker_kind(turn_kind: Any) -> str:
    turn = "none" if pd.isna(turn_kind) else str(turn_kind)
    if turn == "bottom_reversal":
        return "reversal_bottom"
    if turn == "top_reversal":
        return "reversal_top"
    return "none"


def merge_trade_bias_predictions(
    overlay: pd.DataFrame,
    trade_bias: pd.DataFrame | None,
) -> pd.DataFrame:
    """Left-merge OOS trade-bias actions onto overlay rows by (as_of, code)."""
    out = overlay.copy()
    if trade_bias is None or trade_bias.empty:
        if "trade_action" not in out.columns:
            out["trade_action"] = None
        if "p_up" not in out.columns:
            out["p_up"] = np.nan
        if "model_train_end" not in out.columns:
            out["model_train_end"] = None
        if "hold_days" not in out.columns:
            out["hold_days"] = np.nan
        return out
    tb = trade_bias.copy()
    tb["as_of"] = pd.to_datetime(tb["as_of"]).dt.normalize()
    tb["code"] = tb["code"].astype(str).str.upper()
    keep = [
        c
        for c in (
            "as_of",
            "code",
            "trade_action",
            "p_up",
            "model_train_end",
            "hold_days",
            "buy_thr",
            "sell_thr",
            "model_trained",
        )
        if c in tb.columns
    ]
    tb = tb[keep].drop_duplicates(subset=["as_of", "code"], keep="last")
    # Drop existing merge targets so re-merge is idempotent.
    drop_cols = [
        c
        for c in (
            "trade_action",
            "p_up",
            "model_train_end",
            "hold_days",
            "buy_thr",
            "sell_thr",
            "model_trained",
        )
        if c in out.columns
    ]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    out = out.merge(tb, on=["as_of", "code"], how="left")
    return out


def build_overlay_frame(
    oos: pd.DataFrame,
    reversal: pd.DataFrame,
    events: pd.DataFrame,
    *,
    code: str,
    start_date: str,
    end_date: str,
    trade_bias: pd.DataFrame | None = None,
    promotion: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Merge three CSVs for one symbol/window into a plot-ready overlay table.

    Causal switch markers and post-hoc reversal markers are independent columns
    so the same day can show both a triangle and a hollow diamond.
    """
    code_u = str(code).upper()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    oos_n = _normalize_as_of(oos)
    rev_n = _normalize_as_of(reversal)
    evt_n = _normalize_as_of(events) if not events.empty else events

    oos_cols = [c for c in OOS_KEEP if c in oos_n.columns]
    rev_cols = [c for c in REVERSAL_KEEP if c in rev_n.columns]
    evt_cols = [c for c in EVENT_KEEP if c in evt_n.columns]

    base = oos_n.loc[oos_n["code"] == code_u, oos_cols].copy()
    if base.empty:
        raise ValueError(f"code {code_u} not found in signals_oos.csv")
    base = base[(base["as_of"] >= start) & (base["as_of"] <= end)]
    if base.empty:
        raise ValueError(
            f"no signals_oos rows for {code_u} in [{start.date()}, {end.date()}]"
        )

    rev = rev_n.loc[rev_n["code"] == code_u, rev_cols].copy()
    rev = rev[(rev["as_of"] >= start) & (rev["as_of"] <= end)]
    evt = dedupe_event_rows(
        evt_n.loc[evt_n["code"] == code_u, evt_cols].copy()
        if not evt_n.empty
        else evt_n
    )
    if not evt.empty:
        evt = evt[(evt["as_of"] >= start) & (evt["as_of"] <= end)]

    out = base.merge(rev, on=["as_of", "code"], how="left")
    if not evt.empty:
        out = out.merge(evt, on=["as_of", "code"], how="left")
    else:
        for col in ("matched_event_onset", "detection_delay", "is_duplicate", "is_fp"):
            if col not in out.columns:
                out[col] = np.nan if col != "is_duplicate" and col != "is_fp" else False

    out = merge_trade_bias_predictions(out, trade_bias)

    out["quantile_switch"] = out.get("quantile_switch", False)
    out["quantile_switch"] = out["quantile_switch"].fillna(False).astype(bool)
    out["label_switch_reversal"] = out.get("label_switch_reversal", False)
    out["label_switch_reversal"] = out["label_switch_reversal"].fillna(False).astype(bool)
    out["turn_kind"] = out.get("turn_kind", "none").fillna("none").astype(str)
    out["jump_role"] = out.get("jump_role", "none").fillna("none").astype(str)
    out["switch_side"] = out.get("switch_side")
    if "is_duplicate" not in out.columns:
        out["is_duplicate"] = False
    out["is_duplicate"] = out["is_duplicate"].map(
        lambda v: False if pd.isna(v) else bool(v)
    )
    if "is_fp" not in out.columns:
        out["is_fp"] = False
    out["is_fp"] = out["is_fp"].map(lambda v: False if pd.isna(v) else bool(v))

    action_labels: list[str] = []
    for _, row in out.iterrows():
        prior_val = row.get("causal_prior_ret")
        if prior_val is None or (isinstance(prior_val, float) and pd.isna(prior_val)):
            prior_val = row.get("prior_ret")
        thr_val = row.get("causal_thr_prior")
        trade_action = row.get("trade_action")
        if trade_action is not None and (
            isinstance(trade_action, float) and pd.isna(trade_action)
        ):
            trade_action = None
        action_labels.append(
            causal_action_label(
                quantile_switch=bool(row.get("quantile_switch")),
                switch_side=(
                    None
                    if pd.isna(row.get("switch_side"))
                    else str(row.get("switch_side"))
                ),
                prior_ret=(
                    None
                    if prior_val is None
                    or (isinstance(prior_val, float) and pd.isna(prior_val))
                    else float(prior_val)
                ),
                thr_prior=(
                    None
                    if thr_val is None
                    or (isinstance(thr_val, float) and pd.isna(thr_val))
                    else float(thr_val)
                ),
                trade_action=None if trade_action is None else str(trade_action),
                promotion=promotion,
            )
        )
    out["action_label"] = action_labels
    # Normalize optional early/final confirmation columns.
    if "early_turn_kind" not in out.columns:
        out["early_turn_kind"] = "none"
    out["early_turn_kind"] = out["early_turn_kind"].fillna("none").astype(str)
    if "early_jump_role" not in out.columns:
        out["early_jump_role"] = "none"
    out["early_jump_role"] = out["early_jump_role"].fillna("none").astype(str)
    if "early_label_switch_reversal" not in out.columns:
        out["early_label_switch_reversal"] = False
    out["early_label_switch_reversal"] = (
        out["early_label_switch_reversal"].fillna(False).astype(bool)
    )
    if "early_to_final_status" not in out.columns:
        out["early_to_final_status"] = "none"
    out["early_to_final_status"] = out["early_to_final_status"].fillna("none").astype(str)
    for col in ("early_confirmed_on", "final_confirmed_on"):
        if col not in out.columns:
            out[col] = pd.NaT
        out[col] = pd.to_datetime(out[col], errors="coerce")

    out["reversal_label"] = [
        posthoc_reversal_label(turn_kind=turn, jump_role=role, stage="final")
        for turn, role in zip(out["turn_kind"], out["jump_role"])
    ]
    out["early_reversal_label"] = [
        posthoc_reversal_label(turn_kind=turn, jump_role=role, stage="early")
        for turn, role in zip(out["early_turn_kind"], out["early_jump_role"])
    ]
    # Independent marker tracks: same day may keep both triangle and diamond.
    out["switch_marker"] = [
        _switch_marker_kind(bool(qsw), side)
        for qsw, side in zip(out["quantile_switch"], out["switch_side"])
    ]
    # Origin-day aliases keep final turn_kind for backward-compatible filters.
    out["reversal_marker"] = [
        _reversal_marker_kind(turn) for turn in out["turn_kind"]
    ]
    out["early_reversal_marker"] = [
        _reversal_marker_kind(turn) if bool(lab) else "none"
        for lab, turn in zip(out["early_label_switch_reversal"], out["early_turn_kind"])
    ]
    # Backward-compatible alias: primary causal marker only (never mixes future).
    out["marker_kind"] = out["switch_marker"]

    return out.sort_values("as_of", kind="stable").reset_index(drop=True)


def build_confirmation_marker_frame(overlay: pd.DataFrame) -> pd.DataFrame:
    """Expand origin-day labels into confirmation-day marker rows.

    Marker ``as_of`` is the date the label becomes knowable (T+3 / T+10).
    ``origin_as_of`` keeps the pivot day for hover text.
    """
    if overlay.empty:
        return overlay.iloc[0:0].copy()
    rows: list[dict[str, Any]] = []
    for _, row in overlay.iterrows():
        origin = pd.Timestamp(row["as_of"]).normalize()
        early_on = row.get("early_confirmed_on")
        final_on = row.get("final_confirmed_on")
        early_kind = str(row.get("early_reversal_marker") or "none")
        final_kind = str(row.get("reversal_marker") or "none")
        status = str(row.get("early_to_final_status") or "none")

        if early_kind in {"reversal_bottom", "reversal_top"} and pd.notna(early_on):
            rows.append(
                {
                    "as_of": pd.Timestamp(early_on).normalize(),
                    "origin_as_of": origin,
                    "confirm_stage": "early",
                    "reversal_marker": f"early_{early_kind}",
                    "reversal_label": row.get("early_reversal_label"),
                    "turn_kind": row.get("early_turn_kind"),
                    "jump_role": row.get("early_jump_role"),
                    "prior_ret": row.get("early_prior_ret"),
                    "jump_ret": row.get("early_jump_ret"),
                    "post_ret": row.get("early_post_ret"),
                    "early_to_final_status": status,
                    "matched_event_onset": row.get("matched_event_onset"),
                    "detection_delay": row.get("detection_delay"),
                    "is_duplicate": row.get("is_duplicate"),
                    "is_fp": row.get("is_fp"),
                }
            )
        if final_kind in {"reversal_bottom", "reversal_top"} and pd.notna(final_on):
            # Only plot when confirmation date is knowable; never fall back to
            # origin day (that would visually leak a future-dependent label).
            rows.append(
                {
                    "as_of": pd.Timestamp(final_on).normalize(),
                    "origin_as_of": origin,
                    "confirm_stage": "final",
                    "reversal_marker": f"final_{final_kind}",
                    "reversal_label": row.get("reversal_label"),
                    "turn_kind": row.get("turn_kind"),
                    "jump_role": row.get("jump_role"),
                    "prior_ret": row.get("prior_ret"),
                    "jump_ret": row.get("jump_ret"),
                    "post_ret": row.get("post_ret"),
                    "early_to_final_status": status,
                    "matched_event_onset": row.get("matched_event_onset"),
                    "detection_delay": row.get("detection_delay"),
                    "is_duplicate": row.get("is_duplicate"),
                    "is_fp": row.get("is_fp"),
                }
            )
        if status == "revoked" and pd.notna(final_on):
            rows.append(
                {
                    "as_of": pd.Timestamp(final_on).normalize(),
                    "origin_as_of": origin,
                    "confirm_stage": "revoked",
                    "reversal_marker": "reversal_revoked",
                    "reversal_label": posthoc_reversal_label(
                        turn_kind=row.get("early_turn_kind"),
                        jump_role=row.get("early_jump_role"),
                        stage="revoked",
                    ),
                    "turn_kind": row.get("early_turn_kind"),
                    "jump_role": row.get("early_jump_role"),
                    "prior_ret": row.get("early_prior_ret"),
                    "jump_ret": row.get("early_jump_ret"),
                    "post_ret": row.get("early_post_ret"),
                    "early_to_final_status": status,
                    "matched_event_onset": row.get("matched_event_onset"),
                    "detection_delay": row.get("detection_delay"),
                    "is_duplicate": row.get("is_duplicate"),
                    "is_fp": row.get("is_fp"),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "as_of",
                "origin_as_of",
                "confirm_stage",
                "reversal_marker",
                "reversal_label",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["as_of", "origin_as_of", "confirm_stage"], kind="stable"
    ).reset_index(drop=True)

def collapse_reversal_marker_runs(overlay: pd.DataFrame) -> pd.DataFrame:
    """Keep only the first diamond in consecutive same-kind reversal runs.

    Switch triangles are untouched. Consecutive post-hoc reversal labels
    otherwise paint a diamond on every bar and clutter the candle chart.
    """
    if overlay.empty:
        return overlay.copy()
    out = overlay.sort_values("as_of", kind="stable").copy()
    diamond_kinds = {
        "reversal_bottom",
        "reversal_top",
        "early_reversal_bottom",
        "early_reversal_top",
        "final_reversal_bottom",
        "final_reversal_top",
        "reversal_revoked",
    }
    if "reversal_marker" not in out.columns:
        # Legacy frames that only had marker_kind with mixed types.
        out["reversal_marker"] = [
            kind
            if str(kind) in diamond_kinds
            else "none"
            for kind in out.get("marker_kind", pd.Series("none", index=out.index))
            .astype(str)
        ]
    keep: list[bool] = []
    prev_rev: str | None = None
    for kind in out["reversal_marker"].astype(str):
        if kind in diamond_kinds:
            keep.append(kind != prev_rev)
            prev_rev = kind
        else:
            keep.append(True)
            prev_rev = None
    # Drop subsequent diamonds in a run; leave switch_marker alone.
    drop = ~pd.Series(keep, index=out.index) & out["reversal_marker"].isin(diamond_kinds)
    out.loc[drop, "reversal_marker"] = "none"
    return out


def _causal_hover(row: pd.Series) -> str:
    parts = [
        f"日期={pd.Timestamp(row['as_of']).date()}",
        f"动作={row.get('action_label')}",
        f"signal={row.get('signal_level')}",
        f"pred_mean={_fmt_log_pct(row.get('pred_mean'))}",
        f"pred_mean_next={_fmt_log_pct(row.get('pred_mean_next'))}",
        f"pred_cdf={_fmt(row.get('pred_cdf'))}",
        f"return_z={_fmt(row.get('return_z'))}",
        f"p_switch_hmm={_fmt(row.get('p_switch_hmm'))}",
        f"p_into_reversal_hmm={_fmt(row.get('p_into_reversal_hmm'))}",
    ]
    switch_source = row.get("switch_source")
    if switch_source is not None and not (
        isinstance(switch_source, float) and pd.isna(switch_source)
    ):
        src = str(switch_source).strip()
        if src == "extreme_drop_override":
            parts.append("来源=极端跌幅兜底")
            day_ret = row.get("day_ret")
            if day_ret is not None and not (
                isinstance(day_ret, float) and pd.isna(day_ret)
            ):
                parts.append(f"day_ret={float(day_ret):+.2%}")
        elif src:
            parts.append(f"来源={src}")
    trade_action = row.get("trade_action")
    if trade_action is not None and not (
        isinstance(trade_action, float) and pd.isna(trade_action)
    ):
        parts.append(f"5日模型参考={trade_action}")
        parts.append(f"p_up={_fmt(row.get('p_up'))}")
        hold = row.get("hold_days")
        if hold is not None and not (isinstance(hold, float) and pd.isna(hold)):
            parts.append(f"hold_days={int(hold)}")
        train_end = row.get("model_train_end")
        if train_end is not None and not (
            isinstance(train_end, float) and pd.isna(train_end)
        ):
            parts.append(f"model_train_end={train_end}")
        parts.append(
            "说明=三角默认仅观察（上行/下行）；偏买/偏卖须 lockbox 门控通过；"
            "颜色只表示上行/下行切换；5日模型参考独立显示"
        )
    else:
        parts.append(
            "说明=三角默认仅观察（上行/下行）；偏买/偏卖须 lockbox 门控通过"
            "（无未来标签，非买卖单）"
        )
    return "<br>".join(parts)


def _posthoc_hover(row: pd.Series) -> str:
    stage = str(row.get("confirm_stage") or "final")
    stage_name = {
        "early": "早期确认(T+3，事后临时)",
        "final": "最终确认(T+10，事后)",
        "revoked": "早期确认已撤销(最终未成立)",
    }.get(stage, "事后反转验证")
    origin = row.get("origin_as_of", row.get("as_of"))
    parts = [
        f"确认日={pd.Timestamp(row['as_of']).date()}",
        f"拐点日={pd.Timestamp(origin).date()}",
        f"类型={stage_name}",
        f"验证={row.get('reversal_label')}",
        f"turn={row.get('turn_kind')}",
        f"jump_role={row.get('jump_role')}",
        f"prior={_fmt(row.get('prior_ret'))}",
        f"jump={_fmt(row.get('jump_ret'))}",
        f"post={_fmt(row.get('post_ret'))}",
        f"status={row.get('early_to_final_status')}",
    ]
    onset = row.get("matched_event_onset")
    if pd.notna(onset):
        parts.append(f"matched_onset={pd.Timestamp(onset).date()}")
    delay = row.get("detection_delay")
    if pd.notna(delay):
        parts.append(f"delay={delay}")
    if "is_duplicate" in row.index:
        parts.append(f"duplicate={bool(row.get('is_duplicate'))}")
    if "is_fp" in row.index:
        parts.append(f"is_fp={bool(row.get('is_fp'))}")
    return "<br>".join(parts)


def overlay_marker_specs(overlay: pd.DataFrame) -> list[dict[str, Any]]:
    """Build Plotly scatter specs with causal triangles and post-hoc diamonds."""
    specs: list[dict[str, Any]] = []
    plot_overlay = overlay.copy()
    if "switch_marker" not in plot_overlay.columns:
        plot_overlay["switch_marker"] = plot_overlay.get(
            "marker_kind", pd.Series("none", index=plot_overlay.index)
        )
    switch_groups = (
        ("switch_up", triangle_legend_name("up"), "triangle-up", "#2ca02c", "above"),
        ("switch_down", triangle_legend_name("down"), "triangle-down", "#d62728", "below"),
    )
    # Confirmation diamonds are placed on knowable dates (T+3 / T+10).
    confirm = collapse_reversal_marker_runs(build_confirmation_marker_frame(plot_overlay))
    reversal_groups = (
        (
            "early_reversal_bottom",
            "早期底反确认(T+3)",
            "diamond-open",
            "#9ecae1",
            "above",
        ),
        (
            "early_reversal_top",
            "早期顶反确认(T+3)",
            "diamond-open",
            "#fdae6b",
            "below",
        ),
        (
            "final_reversal_bottom",
            "最终底反确认(T+10)",
            "diamond-open",
            "#1f77b4",
            "above",
        ),
        (
            "final_reversal_top",
            "最终顶反确认(T+10)",
            "diamond-open",
            "#ff7f0e",
            "below",
        ),
        (
            "reversal_revoked",
            "早期确认已撤销",
            "diamond-open",
            "#7f7f7f",
            "below",
        ),
    )
    for kind, name, symbol, color, anchor in switch_groups:
        sub = plot_overlay[plot_overlay["switch_marker"] == kind]
        if sub.empty:
            continue
        hover = [_causal_hover(row) for _, row in sub.iterrows()]
        specs.append(
            {
                "name": name,
                "symbol": symbol,
                "color": color,
                "anchor": anchor,
                "x": pd.to_datetime(sub["as_of"]).tolist(),
                "hover": hover,
                "rows": sub,
            }
        )
    for kind, name, symbol, color, anchor in reversal_groups:
        if confirm.empty or "reversal_marker" not in confirm.columns:
            continue
        sub = confirm[confirm["reversal_marker"] == kind]
        if sub.empty:
            continue
        hover = [_posthoc_hover(row) for _, row in sub.iterrows()]
        specs.append(
            {
                "name": name,
                "symbol": symbol,
                "color": color,
                "anchor": anchor,
                "x": pd.to_datetime(sub["as_of"]).tolist(),
                "hover": hover,
                "rows": sub,
            }
        )
    return specs

def _fmt_log_pct(value: Any, digits: int = 2) -> str:
    """Format log-return expectation as a percentage string."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except Exception:
        pass
    try:
        return f"{float(value) * 100.0:.{digits}f}%"
    except Exception:
        return str(value)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except Exception:
        pass
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


# Causal realized-vol regime labels for K-line background / vol panel.
VOL_REGIME_EXTREME = "波动率极端聚集"
VOL_REGIME_CLUSTER = "波动率聚集"
VOL_REGIME_NORMAL = "波动率正常"
VOL_REGIME_CALM = "波动率平稳"
VOL_REGIME_UNKNOWN = "未判定"
# Background shading only highlights elevated regimes (not calm/normal).
VOL_REGIMES_SHADED = frozenset({VOL_REGIME_CLUSTER, VOL_REGIME_EXTREME})
VOL_REGIMES_KNOWN = frozenset(
    {VOL_REGIME_CALM, VOL_REGIME_NORMAL, VOL_REGIME_CLUSTER, VOL_REGIME_EXTREME}
)
DEFAULT_RV_WINDOW = 5
DEFAULT_RV20_WINDOW = 20
DEFAULT_VOL_PCT_LOOKBACK = 120
DEFAULT_VOL_PCT_MIN_HISTORY = 5
DEFAULT_VOL_CALM_PCT = 0.30
DEFAULT_VOL_CLUSTER_PCT = 0.70
DEFAULT_VOL_EXTREME_PCT = 0.90
# Legacy aliases kept for older callers/tests that still pass lookback/q args.
DEFAULT_VOL_LOOKBACK = DEFAULT_VOL_PCT_LOOKBACK
DEFAULT_VOL_CLUSTER_Q = DEFAULT_VOL_CLUSTER_PCT
DEFAULT_VOL_CALM_Q = DEFAULT_VOL_CALM_PCT
TRADING_DAYS_PER_YEAR = 252.0


def compute_realized_vol(
    close: pd.Series,
    *,
    window: int = DEFAULT_RV_WINDOW,
    ann_factor: float = TRADING_DAYS_PER_YEAR,
    ddof: int = 0,
) -> pd.Series:
    """Annualized rolling realized vol from log returns (causal through T)."""
    px = pd.to_numeric(close, errors="coerce")
    log_ret = np.log(px / px.shift(1))
    rv = log_ret.rolling(int(window), min_periods=int(window)).std(ddof=int(ddof))
    return rv * float(np.sqrt(ann_factor))


def compute_vol5_pct_120d(
    rv: pd.Series,
    *,
    lookback: int = DEFAULT_VOL_PCT_LOOKBACK,
    min_history: int = DEFAULT_VOL_PCT_MIN_HISTORY,
) -> pd.Series:
    """Historical percentile of current rv within the trailing lookback window.

    Matches ``garch_short_horizon_board.realized_vol_percentile_log``:
    for each T, take the last ``lookback`` finite rv values ending at T
    (inclusive) and return ``mean(history <= rv[T])``.
    """
    values = pd.to_numeric(rv, errors="coerce").to_numpy(dtype=float)
    out = np.full(values.shape[0], np.nan, dtype=float)
    lb = int(lookback)
    min_hist = int(min_history)
    for i in range(values.shape[0]):
        cur = values[i]
        if not np.isfinite(cur):
            continue
        start = max(0, i + 1 - lb)
        window = values[start : i + 1]
        finite = window[np.isfinite(window)]
        if finite.size < min_hist:
            continue
        out[i] = float(np.sum(finite <= cur)) / float(finite.size)
    return pd.Series(out, index=rv.index, name="vol5_pct_120d")


def classify_vol_pct_regime(
    pct: float | None,
    *,
    calm_pct: float = DEFAULT_VOL_CALM_PCT,
    cluster_pct: float = DEFAULT_VOL_CLUSTER_PCT,
    extreme_pct: float = DEFAULT_VOL_EXTREME_PCT,
) -> str:
    """Map a causal vol percentile into a four-tier regime label."""
    if pct is None or not np.isfinite(float(pct)):
        return VOL_REGIME_UNKNOWN
    p = float(pct)
    if p <= float(calm_pct):
        return VOL_REGIME_CALM
    if p < float(cluster_pct):
        return VOL_REGIME_NORMAL
    if p < float(extreme_pct):
        return VOL_REGIME_CLUSTER
    return VOL_REGIME_EXTREME


def compute_volatility_regime_frame(
    close: pd.Series,
    *,
    rv_window: int = DEFAULT_RV_WINDOW,
    rv20_window: int = DEFAULT_RV20_WINDOW,
    lookback: int = DEFAULT_VOL_PCT_LOOKBACK,
    min_history: int = DEFAULT_VOL_PCT_MIN_HISTORY,
    calm_pct: float = DEFAULT_VOL_CALM_PCT,
    cluster_pct: float = DEFAULT_VOL_CLUSTER_PCT,
    extreme_pct: float = DEFAULT_VOL_EXTREME_PCT,
    # Legacy kwargs accepted but ignored (replaced by percentile tiers).
    cluster_q: float | None = None,
    calm_q: float | None = None,
) -> pd.DataFrame:
    """Build causal vol-regime labels from close prices.

    Uses annualized ``rv5`` / ``rv20`` plus ``vol5_pct_120d`` (trailing
    historical percentile of rv5, aligned with GARCH board). Four-tier
    states: calm ≤30%, normal 30–70%, cluster 70–90%, extreme ≥90%.
    """
    del cluster_q, calm_q  # legacy API compatibility
    if close is None:
        raise ValueError("close series is required")
    series = pd.Series(pd.to_numeric(close, errors="coerce"), copy=False)
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    series.index = pd.DatetimeIndex(series.index).normalize()

    # Display lines keep ddof=0 (existing HTML series). Percentile uses
    # ddof=1 so vol5_pct_120d matches realized_vol_percentile_log.
    rv5 = compute_realized_vol(series, window=rv_window, ddof=0)
    rv20 = compute_realized_vol(series, window=rv20_window, ddof=0)
    rv5_for_pct = compute_realized_vol(series, window=rv_window, ddof=1)
    vol_pct = compute_vol5_pct_120d(
        rv5_for_pct, lookback=lookback, min_history=min_history
    )
    regimes = [
        classify_vol_pct_regime(
            None if (v is None or not np.isfinite(float(v))) else float(v),
            calm_pct=calm_pct,
            cluster_pct=cluster_pct,
            extreme_pct=extreme_pct,
        )
        for v in vol_pct.to_numpy(dtype=float)
    ]

    # Legacy threshold columns: rolling q of shifted rv5 (diagnostic only).
    hist = rv5.shift(1)
    q_hi = hist.rolling(20, min_periods=20).quantile(float(cluster_pct))
    q_lo = hist.rolling(20, min_periods=20).quantile(float(calm_pct))

    out = pd.DataFrame(
        {
            "as_of": series.index,
            "close": series.to_numpy(dtype=float),
            "rv5": rv5.to_numpy(dtype=float),
            "rv20": rv20.to_numpy(dtype=float),
            "vol5_pct_120d": vol_pct.to_numpy(dtype=float),
            "vol_q_cluster": q_hi.to_numpy(dtype=float),
            "vol_q_calm": q_lo.to_numpy(dtype=float),
            "vol_regime": regimes,
        }
    )
    return out.reset_index(drop=True)


def volatility_regime_segments(
    vol_frame: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    regimes: frozenset[str] | set[str] | None = None,
) -> list[dict[str, Any]]:
    """Collapse consecutive same-regime days into [start, end] segments.

    By default only elevated regimes (cluster / extreme) are returned for
    K-line background shading.
    """
    if vol_frame is None or vol_frame.empty:
        return []
    keep = frozenset(regimes) if regimes is not None else VOL_REGIMES_SHADED
    work = vol_frame.copy()
    work["as_of"] = pd.to_datetime(work["as_of"]).dt.normalize()
    if start_date is not None:
        work = work[work["as_of"] >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        work = work[work["as_of"] <= pd.Timestamp(end_date).normalize()]
    work = work.sort_values("as_of")
    if work.empty:
        return []

    segments: list[dict[str, Any]] = []
    cur_regime = None
    seg_start = None
    seg_end = None
    for _, row in work.iterrows():
        regime = str(row.get("vol_regime") or VOL_REGIME_UNKNOWN)
        day = pd.Timestamp(row["as_of"]).normalize()
        if regime not in keep:
            if cur_regime is not None:
                segments.append(
                    {"regime": cur_regime, "start": seg_start, "end": seg_end}
                )
                cur_regime = None
                seg_start = None
                seg_end = None
            continue
        if cur_regime is None:
            cur_regime = regime
            seg_start = day
            seg_end = day
        elif regime == cur_regime:
            seg_end = day
        else:
            segments.append(
                {"regime": cur_regime, "start": seg_start, "end": seg_end}
            )
            cur_regime = regime
            seg_start = day
            seg_end = day
    if cur_regime is not None:
        segments.append({"regime": cur_regime, "start": seg_start, "end": seg_end})
    return segments


__all__ = [
    "DEFAULT_TRIANGLE_BIAS_RULE",
    "DEFAULT_VALIDATION_DIR",
    "DEFAULT_RV_WINDOW",
    "DEFAULT_RV20_WINDOW",
    "DEFAULT_VOL_LOOKBACK",
    "DEFAULT_VOL_PCT_LOOKBACK",
    "DEFAULT_VOL_PCT_MIN_HISTORY",
    "DEFAULT_VOL_CALM_PCT",
    "DEFAULT_VOL_CLUSTER_PCT",
    "DEFAULT_VOL_EXTREME_PCT",
    "DEFAULT_VOL_CLUSTER_Q",
    "DEFAULT_VOL_CALM_Q",
    "VOL_REGIME_CALM",
    "VOL_REGIME_NORMAL",
    "VOL_REGIME_CLUSTER",
    "VOL_REGIME_EXTREME",
    "VOL_REGIME_UNKNOWN",
    "VOL_REGIMES_KNOWN",
    "VOL_REGIMES_SHADED",
    "build_confirmation_marker_frame",
    "build_overlay_frame",
    "causal_action_label",
    "classify_vol_pct_regime",
    "collapse_reversal_marker_runs",
    "compute_realized_vol",
    "compute_vol5_pct_120d",
    "compute_volatility_regime_frame",
    "dedupe_event_rows",
    "load_validation_csvs",
    "overlay_marker_specs",
    "posthoc_reversal_label",
    "triangle_legend_name",
    "validation_action_label",
    "volatility_regime_segments",
]
