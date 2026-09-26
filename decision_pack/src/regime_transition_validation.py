"""Pure-function validation helpers for regime-transition walk-forward.

No qlib / filesystem side effects. Scripts own I/O and orchestration.
Universe for this MVP is the cluster_mapping_selected instruments list.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

UniverseMode = Literal["cluster_selected", "survivor_exploratory", "invalid"]


@dataclass(frozen=True)
class InstrumentMembership:
    code: str
    list_date: pd.Timestamp
    end_date: pd.Timestamp | None


@dataclass(frozen=True)
class UniverseAuditResult:
    mode: UniverseMode
    n_instruments: int
    n_codes_with_dates: int
    missing_list_date: int
    missing_end_date: int
    messages: tuple[str, ...]
    instruments: tuple[InstrumentMembership, ...]

    @property
    def ok_for_formal(self) -> bool:
        return self.mode == "cluster_selected" and self.n_instruments > 0


def canonical_config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_cluster_instruments(path: str | Any) -> list[InstrumentMembership]:
    """Parse qlib instruments txt: code\\tstart\\tend."""
    from pathlib import Path

    from pipeline.strategy_layer_audit import normalize_code

    rows: list[InstrumentMembership] = []
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if not parts:
            continue
        code = normalize_code(parts[0].strip())
        if not code:
            continue
        list_date = pd.Timestamp(parts[1]) if len(parts) > 1 and parts[1].strip() else pd.NaT
        end_raw = parts[2].strip() if len(parts) > 2 else ""
        end_date = pd.Timestamp(end_raw) if end_raw else None
        if pd.isna(list_date):
            continue
        rows.append(
            InstrumentMembership(
                code=code,
                list_date=pd.Timestamp(list_date).normalize(),
                end_date=pd.Timestamp(end_date).normalize() if end_date is not None else None,
            )
        )
    # de-dupe keeping first
    seen: set[str] = set()
    out: list[InstrumentMembership] = []
    for row in rows:
        if row.code in seen:
            continue
        seen.add(row.code)
        out.append(row)
    return out


def audit_cluster_universe(
    instruments: list[InstrumentMembership] | tuple[InstrumentMembership, ...],
    *,
    allow_survivor_exploratory: bool = False,
) -> UniverseAuditResult:
    """Audit the selected-cluster universe.

    For this MVP the source of truth is ``cluster_mapping_selected.txt``.
    Formal mode requires every row to have a list_date. Missing end_date is
    allowed (treated as still listed). This is intentionally *not* an all-ETF
    point-in-time claim; mode is ``cluster_selected``.
    """
    instruments = list(instruments)
    missing_list = sum(1 for i in instruments if pd.isna(i.list_date))
    missing_end = sum(1 for i in instruments if i.end_date is None)
    messages: list[str] = []
    if not instruments:
        messages.append("cluster instruments empty")
        return UniverseAuditResult(
            mode="invalid",
            n_instruments=0,
            n_codes_with_dates=0,
            missing_list_date=0,
            missing_end_date=0,
            messages=tuple(messages),
            instruments=(),
        )
    if missing_list:
        messages.append(f"{missing_list} instruments missing list_date")
        if allow_survivor_exploratory:
            mode: UniverseMode = "survivor_exploratory"
            messages.append("running with --allow-survivor-exploratory")
        else:
            mode = "invalid"
            messages.append("formal cluster_selected mode hard-stop")
    else:
        mode = "cluster_selected"
        messages.append(
            f"cluster_selected universe with {len(instruments)} codes "
            "(not all.txt; membership from instruments start/end dates)"
        )
        if missing_end:
            messages.append(
                f"{missing_end} codes have open end_date; treated as still listed"
            )
    return UniverseAuditResult(
        mode=mode,
        n_instruments=len(instruments),
        n_codes_with_dates=len(instruments) - missing_list,
        missing_list_date=missing_list,
        missing_end_date=missing_end,
        messages=tuple(messages),
        instruments=tuple(instruments),
    )


def is_eligible_on_date(
    membership: InstrumentMembership,
    as_of: pd.Timestamp,
    close: pd.Series | None,
    *,
    min_history: int = 252,
    min_valid_in_20: int = 18,
) -> bool:
    """Point-in-time eligibility for one code on ``as_of``.

    ``membership.end_date`` is treated as the last *known* instruments/Qlib
    listing day. Dates beyond that are still allowed when ``close`` already
    carries a finite bar on ``as_of`` (provisional intraday live snapshot /
    calendar lag). Truly missing bars after end_date remain ineligible.
    """
    ts = pd.Timestamp(as_of).normalize()
    if pd.isna(membership.list_date) or ts < membership.list_date.normalize():
        return False
    if close is None:
        return False
    series = pd.to_numeric(close, errors="coerce")
    if membership.end_date is not None and ts > membership.end_date.normalize():
        # Provisional extension: require an explicit finite close on as_of.
        if ts not in series.index:
            return False
        try:
            px = float(series.loc[ts])
        except Exception:
            return False
        if not np.isfinite(px):
            return False
    series = series.dropna()
    if series.empty:
        return False
    hist = series.loc[series.index <= ts]
    if hist.size < min_history:
        return False
    # last 20 trading observations up to as_of
    tail = hist.iloc[-20:]
    if int(np.isfinite(tail.to_numpy(dtype=float)).sum()) < min_valid_in_20:
        return False
    return True


def eligible_codes_on_date(
    audit: UniverseAuditResult,
    as_of: pd.Timestamp,
    panel: pd.DataFrame,
    *,
    min_history: int = 252,
    min_valid_in_20: int = 18,
) -> list[str]:
    out: list[str] = []
    for membership in audit.instruments:
        series = panel[membership.code] if membership.code in panel.columns else None
        if is_eligible_on_date(
            membership,
            as_of,
            series,
            min_history=min_history,
            min_valid_in_20=min_valid_in_20,
        ):
            out.append(membership.code)
    return out


def _trading_loc(close: pd.Series, as_of: pd.Timestamp) -> tuple[int, pd.Timestamp] | None:
    idx = close.index
    as_of = pd.Timestamp(as_of)
    if as_of not in idx:
        pos = int(idx.searchsorted(as_of, side="right")) - 1
        if pos < 0:
            return None
        as_of = pd.Timestamp(idx[pos])
    loc = idx.get_loc(as_of)
    if isinstance(loc, slice):
        loc = loc.start
    return int(loc), pd.Timestamp(as_of)


def realized_sigma_20(close: pd.Series, as_of: pd.Timestamp) -> float | None:
    loc_pair = _trading_loc(close, as_of)
    if loc_pair is None:
        return None
    loc, _ = loc_pair
    if loc < 20:
        return None
    window = close.iloc[loc - 20 : loc + 1]
    prices = pd.to_numeric(window, errors="coerce").dropna()
    if prices.size < 19:
        return None
    rets = np.log(prices.to_numpy(dtype=float)[1:] / prices.to_numpy(dtype=float)[:-1])
    rets = rets[np.isfinite(rets)]
    if rets.size < 18:
        return None
    sigma = float(np.std(rets, ddof=1))
    if not math.isfinite(sigma) or sigma <= 0:
        return None
    return sigma


def compute_trend_up_label(
    close: pd.Series,
    origin: pd.Timestamp,
    *,
    horizon: int = 10,
    kappa_ret: float = 0.35,
    kappa_eff: float = 0.28,
    kappa_mae: float = 1.0,
) -> bool | None:
    """Independent price-path trend label. None = unavailable."""
    loc_pair = _trading_loc(close, origin)
    if loc_pair is None:
        return None
    loc, origin = loc_pair
    end_loc = loc + int(horizon)
    if end_loc >= len(close.index):
        return None
    path = pd.to_numeric(close.iloc[loc : end_loc + 1], errors="coerce")
    if path.isna().any() or path.size != int(horizon) + 1:
        return None
    px = path.to_numpy(dtype=float)
    if np.any(px <= 0) or not np.all(np.isfinite(px)):
        return None
    r_th = float(np.log(px[-1] / px[0]))
    step = np.diff(np.log(px))
    path_len = float(np.sum(np.abs(step)))
    efficiency = abs(r_th) / path_len if path_len > 0 else 0.0
    cum = np.cumsum(step)
    mae = float(np.min(cum)) if cum.size else 0.0
    sigma = realized_sigma_20(close, origin)
    if sigma is None:
        return None
    scale = sigma * math.sqrt(float(horizon))
    return bool(
        r_th >= float(kappa_ret) * scale
        and efficiency >= float(kappa_eff)
        and mae >= -float(kappa_mae) * scale
    )


def compute_switch_reversal_label(
    close: pd.Series,
    origin: pd.Timestamp,
    *,
    lookback: int = 10,
    horizon: int = 10,
    kappa_prior: float = 0.35,
    kappa_post: float = 0.35,
) -> bool | None:
    """True if day ``origin`` sits at a pre/post momentum flip (switch/reversal).

    Definition (auditable):
    - ``prior`` = log return over ``lookback`` days ending at T-1 (excludes jump day)
    - ``post``  = log return from close[T] to close[T+H] (path after the jump day)
    - Positive when signs(prior, post) oppose and both exceed scaled σ thresholds.

    Captures bounce pivots *and* climax tops/bottoms (e.g. blow-off up day then drop).
    None = not enough history/future.
    """
    detail = compute_switch_reversal_detail(
        close,
        origin,
        lookback=lookback,
        horizon=horizon,
        kappa_prior=kappa_prior,
        kappa_post=kappa_post,
    )
    if detail is None:
        return None
    return bool(detail["label_switch_reversal"])


def compute_switch_reversal_detail(
    close: pd.Series,
    origin: pd.Timestamp,
    *,
    lookback: int = 10,
    horizon: int = 10,
    kappa_prior: float = 0.35,
    kappa_post: float = 0.35,
) -> dict[str, Any] | None:
    """Return reversal diagnostics at ``origin``, or None if unavailable."""
    loc_pair = _trading_loc(close, origin)
    if loc_pair is None:
        return None
    loc, origin = loc_pair
    L = int(lookback)
    H = int(horizon)
    if loc < L + 1 or loc + H >= len(close.index):
        return None
    px = pd.to_numeric(close.iloc[loc - L - 1 : loc + H + 1], errors="coerce")
    if px.isna().any():
        return None
    vals = px.to_numpy(dtype=float)
    # local indexing: [0]=T-L-1, [L]=T-1, [L+1]=T, [L+1+H]=T+H
    if np.any(vals <= 0) or not np.all(np.isfinite(vals)):
        return None
    c_tm1 = float(vals[L])
    c_t = float(vals[L + 1])
    c_prior_start = float(vals[0])
    c_post_end = float(vals[L + 1 + H])
    prior_ret = float(np.log(c_tm1 / c_prior_start))
    jump_ret = float(np.log(c_t / c_tm1))
    post_ret = float(np.log(c_post_end / c_t))
    sigma = realized_sigma_20(close, origin)
    if sigma is None:
        return None
    thr_prior = float(kappa_prior) * float(sigma) * math.sqrt(float(L))
    thr_post = float(kappa_post) * float(sigma) * math.sqrt(float(H))
    strong = abs(prior_ret) >= thr_prior and abs(post_ret) >= thr_post
    flipped = prior_ret * post_ret < 0.0
    label = bool(strong and flipped)
    if prior_ret < 0 and post_ret > 0:
        turn_kind = "bottom_reversal"
    elif prior_ret > 0 and post_ret < 0:
        turn_kind = "top_reversal"
    else:
        turn_kind = "none"
    # jump role relative to the turn
    if turn_kind == "bottom_reversal" and jump_ret > 0:
        jump_role = "bounce"
    elif turn_kind == "top_reversal" and jump_ret < 0:
        jump_role = "break"
    elif turn_kind == "top_reversal" and jump_ret > 0:
        jump_role = "climax_top"
    elif turn_kind == "bottom_reversal" and jump_ret < 0:
        jump_role = "climax_bottom"
    else:
        jump_role = "other"
    return {
        "label_switch_reversal": label,
        "turn_kind": turn_kind if label else "none",
        "jump_role": jump_role if label else "none",
        "prior_ret": prior_ret,
        "jump_ret": jump_ret,
        "post_ret": post_ret,
        "thr_prior": thr_prior,
        "thr_post": thr_post,
        "lookback": L,
        "horizon": H,
    }


def compute_causal_prior_context(
    close: pd.Series,
    as_of: pd.Timestamp,
    *,
    lookback: int = 10,
    kappa_prior: float = 0.35,
) -> dict[str, Any] | None:
    """Return prior_ret / thr_prior / jump_ret knowable at EOD ``as_of`` (no future bars).

    ``prior_ret`` uses closes through T-1 only; ``jump_ret`` uses close[T].
    """
    loc_pair = _trading_loc(close, as_of)
    if loc_pair is None:
        return None
    loc, as_of = loc_pair
    L = int(lookback)
    # Need T-L-1 .. T inclusive.
    if loc < L + 1:
        return None
    px = pd.to_numeric(close.iloc[loc - L - 1 : loc + 1], errors="coerce")
    if px.isna().any() or len(px) < L + 2:
        return None
    vals = px.to_numpy(dtype=float)
    # local: [0]=T-L-1, [L]=T-1, [L+1]=T
    if np.any(vals <= 0) or not np.all(np.isfinite(vals)):
        return None
    c_prior_start = float(vals[0])
    c_tm1 = float(vals[L])
    c_t = float(vals[L + 1])
    prior_ret = float(np.log(c_tm1 / c_prior_start))
    jump_ret = float(np.log(c_t / c_tm1))
    sigma = realized_sigma_20(close, as_of)
    if sigma is None:
        return None
    thr_prior = float(kappa_prior) * float(sigma) * math.sqrt(float(L))
    return {
        "prior_ret": prior_ret,
        "jump_ret": jump_ret,
        "thr_prior": thr_prior,
        "lookback": L,
        "kappa_prior": float(kappa_prior),
        "prior_strong": bool(abs(prior_ret) >= thr_prior),
    }


def causal_bias_from_prior(
    *,
    quantile_switch: bool,
    prior_ret: float | None,
    thr_prior: float | None = None,
    require_strong: bool = False,
) -> str | None:
    """Map switch + prior sign to buy_bias/sell_bias without using future fields.

    Returns None when there is no switch, prior is missing/zero, or (if
    ``require_strong``) ``|prior_ret| < thr_prior``.
    """
    if not bool(quantile_switch):
        return None
    if prior_ret is None:
        return None
    try:
        pr = float(prior_ret)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(pr) or pr == 0.0:
        return None
    if require_strong:
        if thr_prior is None:
            return None
        try:
            thr = float(thr_prior)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(thr) or abs(pr) < thr:
            return None
    if pr < 0.0:
        return "buy_bias"
    return "sell_bias"


def causal_bias_from_switch_side(
    *,
    quantile_switch: bool,
    switch_side: str | None,
) -> str | None:
    """Baseline bias: up→buy_bias, down→sell_bias (jump/quantile side)."""
    if not bool(quantile_switch):
        return None
    if switch_side is None or (isinstance(switch_side, float) and math.isnan(switch_side)):
        return None
    side = str(switch_side)
    if side == "up":
        return "buy_bias"
    if side == "down":
        return "sell_bias"
    return None


EARLY_REVERSAL_HORIZON = 3
FINAL_REVERSAL_HORIZON = 10


def confirmation_trading_date(
    close: pd.Series,
    origin: pd.Timestamp,
    horizon: int,
) -> pd.Timestamp | None:
    """Return the trading day when a horizon-H post path becomes knowable (T+H)."""
    loc_pair = _trading_loc(close, origin)
    if loc_pair is None:
        return None
    loc, _ = loc_pair
    end_loc = loc + int(horizon)
    if end_loc < 0 or end_loc >= len(close.index):
        return None
    return pd.Timestamp(close.index[end_loc]).normalize()


def _early_to_final_status(
    *,
    early_mature: bool,
    final_mature: bool,
    early_label: bool,
    final_label: bool,
    early_turn: str,
    final_turn: str,
) -> str:
    if not early_mature and not final_mature:
        return "immature"
    if early_mature and not final_mature:
        return "early_only_immature_final"
    if not early_mature and final_mature:
        # Should be rare (final needs more history than early).
        return "final_only" if final_label else "none"
    if early_label and final_label:
        if early_turn == final_turn and early_turn != "none":
            return "confirmed"
        return "direction_changed"
    if early_label and not final_label:
        return "revoked"
    if (not early_label) and final_label:
        return "final_only"
    return "none"


def compute_dual_horizon_reversal_detail(
    close: pd.Series,
    origin: pd.Timestamp,
    *,
    lookback: int = 10,
    early_horizon: int = EARLY_REVERSAL_HORIZON,
    final_horizon: int = FINAL_REVERSAL_HORIZON,
    kappa_prior: float = 0.35,
    kappa_post: float = 0.35,
) -> dict[str, Any]:
    """Compute early (H=3) and final (H=10) reversal labels plus confirmation dates.

    Unprefixed fields keep the existing final-horizon contract used by assessment
    and event matching. ``early_*`` fields are the faster provisional layer.
    """
    early_h = int(early_horizon)
    final_h = int(final_horizon)
    early = compute_switch_reversal_detail(
        close,
        origin,
        lookback=lookback,
        horizon=early_h,
        kappa_prior=kappa_prior,
        kappa_post=kappa_post,
    )
    final = compute_switch_reversal_detail(
        close,
        origin,
        lookback=lookback,
        horizon=final_h,
        kappa_prior=kappa_prior,
        kappa_post=kappa_post,
    )
    early_on = confirmation_trading_date(close, origin, early_h)
    final_on = confirmation_trading_date(close, origin, final_h)
    early_mature = early is not None
    final_mature = final is not None
    early_label = bool(early["label_switch_reversal"]) if early_mature else False
    final_label = bool(final["label_switch_reversal"]) if final_mature else False
    early_turn = str(early["turn_kind"]) if early_mature else "none"
    final_turn = str(final["turn_kind"]) if final_mature else "none"
    status = _early_to_final_status(
        early_mature=early_mature,
        final_mature=final_mature,
        early_label=early_label,
        final_label=final_label,
        early_turn=early_turn,
        final_turn=final_turn,
    )
    out: dict[str, Any] = {
        "label_switch_reversal": final["label_switch_reversal"] if final_mature else None,
        "turn_kind": final_turn if final_mature else None,
        "jump_role": (str(final["jump_role"]) if final_mature else None),
        "prior_ret": (float(final["prior_ret"]) if final_mature else None),
        "jump_ret": (float(final["jump_ret"]) if final_mature else None),
        "post_ret": (float(final["post_ret"]) if final_mature else None),
        "thr_prior": (float(final["thr_prior"]) if final_mature else None),
        "thr_post": (float(final["thr_post"]) if final_mature else None),
        "lookback": int(lookback),
        "horizon": final_h,
        "final_confirmed_on": final_on,
        "early_label_switch_reversal": (
            early["label_switch_reversal"] if early_mature else None
        ),
        "early_turn_kind": early_turn if early_mature else None,
        "early_jump_role": (str(early["jump_role"]) if early_mature else None),
        "early_prior_ret": (float(early["prior_ret"]) if early_mature else None),
        "early_jump_ret": (float(early["jump_ret"]) if early_mature else None),
        "early_post_ret": (float(early["post_ret"]) if early_mature else None),
        "early_thr_prior": (float(early["thr_prior"]) if early_mature else None),
        "early_thr_post": (float(early["thr_post"]) if early_mature else None),
        "early_horizon": early_h,
        "early_confirmed_on": early_on,
        "early_to_final_status": status,
    }
    return out


def attach_instrument_names(
    frame: pd.DataFrame,
    name_map: dict[str, str] | None,
    *,
    code_col: str = "code",
    name_col: str = "name",
) -> pd.DataFrame:
    """Insert/refresh Chinese ``name`` immediately after ``code``."""
    if frame is None or frame.empty or code_col not in frame.columns:
        return frame
    out = frame.copy()
    mapping = name_map or {}
    resolved = out[code_col].astype(str).map(lambda c: mapping.get(c, ""))
    if name_col in out.columns:
        existing = out[name_col].astype(str).replace({"nan": "", "None": ""})
        out[name_col] = resolved.where(resolved.ne(""), existing)
        cols = [c for c in out.columns if c != name_col]
        code_i = cols.index(code_col)
        out = out.reindex(columns=cols[: code_i + 1] + [name_col] + cols[code_i + 1 :])
    else:
        code_i = list(out.columns).index(code_col)
        out.insert(code_i + 1, name_col, resolved)
    return out


def month_key(ts: pd.Timestamp) -> str:
    t = pd.Timestamp(ts)
    return f"{t.year:04d}-{t.month:02d}"


def last_trading_day_before_month(
    calendar: pd.DatetimeIndex,
    month: str,
) -> pd.Timestamp | None:
    """Last calendar trading day strictly before YYYY-MM."""
    year, mon = [int(x) for x in month.split("-")]
    start = pd.Timestamp(year=year, month=mon, day=1)
    cal = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values()
    prior = cal[cal < start]
    if prior.empty:
        return None
    return pd.Timestamp(prior[-1])


def trading_days_in_month(
    calendar: pd.DatetimeIndex,
    month: str,
) -> pd.DatetimeIndex:
    year, mon = [int(x) for x in month.split("-")]
    start = pd.Timestamp(year=year, month=mon, day=1)
    if mon == 12:
        end = pd.Timestamp(year=year + 1, month=1, day=1)
    else:
        end = pd.Timestamp(year=year, month=mon + 1, day=1)
    cal = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values()
    return cal[(cal >= start) & (cal < end)]


def shift_trading_days(
    calendar: pd.DatetimeIndex,
    as_of: pd.Timestamp,
    n: int,
) -> pd.Timestamp | None:
    cal = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values()
    as_of = pd.Timestamp(as_of)
    if as_of not in cal:
        pos = int(cal.searchsorted(as_of, side="right")) - 1
        if pos < 0:
            return None
        as_of = pd.Timestamp(cal[pos])
    loc = int(cal.get_loc(as_of))
    target = loc + int(n)
    if target < 0 or target >= len(cal):
        return None
    return pd.Timestamp(cal[target])


def build_expanding_crossfit_predictions(
    rows: pd.DataFrame,
    *,
    feature_cols: list[str],
    label_col: str = "label_trend_up",
    month_col: str = "month",
    min_train_rows: int = 30,
) -> pd.DataFrame:
    """Predict each month with logistic fit on earlier mature months only."""
    from sklearn.linear_model import LogisticRegression

    if rows.empty:
        return rows.copy()
    frame = rows.copy()
    frame = frame[frame[label_col].notna()].copy()
    if frame.empty:
        return frame
    frame[label_col] = frame[label_col].astype(bool).astype(int)
    months = sorted(frame[month_col].astype(str).unique())
    preds: list[pd.DataFrame] = []
    for i, m in enumerate(months):
        train = frame[frame[month_col].astype(str) < m]
        test = frame[frame[month_col].astype(str) == m]
        if len(train) < min_train_rows or train[label_col].nunique() < 2:
            continue
        x_train = train[feature_cols].astype(float).to_numpy()
        y_train = train[label_col].to_numpy()
        x_test = test[feature_cols].astype(float).to_numpy()
        if not np.all(np.isfinite(x_train)) or not np.all(np.isfinite(x_test)):
            mask_tr = np.all(np.isfinite(x_train), axis=1)
            mask_te = np.all(np.isfinite(x_test), axis=1)
            if mask_tr.sum() < min_train_rows or mask_te.sum() == 0:
                continue
            x_train, y_train = x_train[mask_tr], y_train[mask_tr]
            test = test.iloc[np.where(mask_te)[0]]
            x_test = x_test[mask_te]
            if y_train.min() == y_train.max():
                continue
        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=200,
            random_state=0,
        )
        model.fit(x_train, y_train)
        score = model.predict_proba(x_test)[:, 1]
        out = test.copy()
        out["p_trend_raw"] = score
        out["crossfit_train_months"] = i
        preds.append(out)
    if not preds:
        return frame.iloc[0:0].copy()
    return pd.concat(preds, ignore_index=True)


def maybe_isotonic_calibrate(
    train_scores: np.ndarray,
    train_labels: np.ndarray,
    apply_scores: np.ndarray,
    *,
    min_rows: int = 500,
    min_pos: int = 50,
    min_top_bin: int = 30,
) -> tuple[np.ndarray, bool]:
    """Optional isotonic on 1-d scores. Returns (calibrated, used_isotonic)."""
    train_scores = np.asarray(train_scores, dtype=float)
    train_labels = np.asarray(train_labels, dtype=float)
    apply_scores = np.asarray(apply_scores, dtype=float)
    if (
        train_scores.size < min_rows
        or int(train_labels.sum()) < min_pos
        or train_labels.min() == train_labels.max()
    ):
        return apply_scores, False
    order = np.argsort(train_scores)
    sorted_s = train_scores[order]
    # two highest score bins by tercile of top half
    q80 = float(np.quantile(sorted_s, 0.8))
    q90 = float(np.quantile(sorted_s, 0.9))
    n_top = int((train_scores >= q90).sum())
    n_mid = int(((train_scores >= q80) & (train_scores < q90)).sum())
    if n_top < min_top_bin or n_mid < min_top_bin:
        return apply_scores, False
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(train_scores, train_labels)
    return iso.predict(apply_scores), True


def select_fdr_threshold(
    crossfit_rows: pd.DataFrame,
    *,
    score_col: str = "p_trend_cal",
    label_col: str = "label_trend_up",
    target_fdr: float = 0.20,
    min_confirm: int = 50,
    n_bootstrap: int = 200,
    block_col: str = "month",
    seed: int = 42,
) -> dict[str, Any]:
    """Choose lowest threshold whose bootstrap FDR upper CI <= target."""
    if crossfit_rows.empty or score_col not in crossfit_rows.columns:
        return {
            "q_confirm": None,
            "confirm_suppressed_reason": "no_crossfit_rows",
            "n_confirm_support": 0,
        }
    frame = crossfit_rows.dropna(subset=[score_col, label_col]).copy()
    if frame.empty:
        return {
            "q_confirm": None,
            "confirm_suppressed_reason": "empty_after_dropna",
            "n_confirm_support": 0,
        }
    scores = frame[score_col].astype(float).to_numpy()
    labels = frame[label_col].astype(bool).to_numpy()
    candidates = np.unique(np.round(scores, 6))
    candidates = candidates[::-1]
    rng = np.random.default_rng(seed)
    blocks = frame[block_col].astype(str).to_numpy() if block_col in frame.columns else None
    unique_blocks = np.unique(blocks) if blocks is not None else None

    best: float | None = None
    best_n = 0
    for thr in candidates:
        mask = scores >= float(thr)
        # include all ties at thr
        n = int(mask.sum())
        if n < min_confirm:
            continue
        # point FDR
        fp = int((~labels[mask]).sum())
        fdr_point = fp / max(n, 1)
        # block bootstrap upper CI
        if unique_blocks is not None and unique_blocks.size >= 2:
            boot = []
            for _ in range(n_bootstrap):
                chosen = rng.choice(unique_blocks, size=unique_blocks.size, replace=True)
                sel = np.isin(blocks, chosen) & mask
                nn = int(sel.sum())
                if nn == 0:
                    continue
                boot.append(float((~labels[sel]).sum()) / nn)
            if not boot:
                continue
            upper = float(np.quantile(boot, 0.95))
        else:
            upper = fdr_point
        if upper <= target_fdr:
            best = float(thr)
            best_n = n
            break
    if best is None:
        return {
            "q_confirm": None,
            "confirm_suppressed_reason": "no_threshold_meets_fdr",
            "n_confirm_support": 0,
            "target_fdr": target_fdr,
        }
    return {
        "q_confirm": best,
        "confirm_suppressed_reason": None,
        "n_confirm_support": best_n,
        "target_fdr": target_fdr,
    }


@dataclass(frozen=True)
class TrendEvent:
    code: str
    direction: str
    onset: pd.Timestamp
    event_end: pd.Timestamp


def cluster_positive_events(
    labels: pd.DataFrame,
    *,
    max_gap: int = 5,
    code_col: str = "code",
    date_col: str = "as_of",
    label_col: str = "label_trend_up",
    direction: str = "up",
) -> list[TrendEvent]:
    """Merge contiguous positive labels with gap <= max_gap trading rows per code."""
    if labels.empty:
        return []
    frame = labels[labels[label_col].astype(bool)].copy()
    if frame.empty:
        return []
    frame[date_col] = pd.to_datetime(frame[date_col])
    events: list[TrendEvent] = []
    for code, grp in frame.groupby(code_col, sort=False):
        dates = sorted(pd.Timestamp(d) for d in grp[date_col])
        if not dates:
            continue
        # use positional gaps on the code's full calendar if provided via all dates
        all_dates = sorted(
            pd.Timestamp(d)
            for d in labels.loc[labels[code_col] == code, date_col]
        )
        index = {d: i for i, d in enumerate(all_dates)}
        cluster = [dates[0]]
        for d in dates[1:]:
            prev = cluster[-1]
            gap = index.get(d, 0) - index.get(prev, 0)
            if gap <= max_gap:
                cluster.append(d)
            else:
                events.append(
                    TrendEvent(
                        code=str(code),
                        direction=direction,
                        onset=cluster[0],
                        event_end=cluster[-1],
                    )
                )
                cluster = [d]
        events.append(
            TrendEvent(
                code=str(code),
                direction=direction,
                onset=cluster[0],
                event_end=cluster[-1],
            )
        )
    events.sort(key=lambda e: (e.code, e.onset))
    return events


def match_alerts_to_events(
    alerts: pd.DataFrame,
    events: list[TrendEvent],
    *,
    lead: int = 3,
    horizon: int = 10,
    code_col: str = "code",
    date_col: str = "as_of",
    score_col: str = "p_trend_cal",
    calendar_by_code: dict[str, pd.DatetimeIndex] | None = None,
) -> pd.DataFrame:
    """Greedy event matching. Returns alert-level match table."""
    if alerts.empty:
        return pd.DataFrame(
            columns=[
                code_col,
                date_col,
                "matched_event_onset",
                "detection_delay",
                "is_duplicate",
                "is_fp",
            ]
        )
    frame = alerts.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.sort_values([code_col, date_col], kind="stable").reset_index(drop=True)
    frame["matched_event_onset"] = pd.NaT
    frame["detection_delay"] = np.nan
    frame["is_duplicate"] = False
    frame["is_fp"] = False

    # Pre-index alerts by code for O(1) candidate lookup instead of full scans.
    by_code: dict[str, pd.DataFrame] = {
        str(code): grp for code, grp in frame.groupby(code_col, sort=False)
    }
    used_alert: set[int] = set()

    for event in sorted(events, key=lambda e: (e.onset, e.code)):
        code = str(event.code)
        sub = by_code.get(code)
        if sub is None or sub.empty:
            continue
        cal = None
        if calendar_by_code and code in calendar_by_code:
            cal = pd.DatetimeIndex(calendar_by_code[code])
        if cal is not None and event.onset in cal:
            onset_loc = int(cal.get_loc(event.onset))
            lo = cal[max(0, onset_loc - lead)]
            hi = cal[min(len(cal) - 1, onset_loc + horizon)]
        else:
            lo = event.onset - pd.tseries.offsets.BDay(lead)
            hi = event.onset + pd.tseries.offsets.BDay(horizon)

        dates = sub[date_col]
        in_window = (dates >= lo) & (dates <= hi)
        cand_idx = [int(i) for i in sub.index[in_window] if int(i) not in used_alert]
        if not cand_idx:
            continue

        def _key(i: int) -> tuple:
            d = pd.Timestamp(frame.at[i, date_col])
            if cal is not None and event.onset in cal and d in cal:
                dist = abs(int(cal.get_loc(d)) - int(cal.get_loc(event.onset)))
            else:
                dist = abs((d - event.onset).days)
            score = float(frame.at[i, score_col]) if score_col in frame.columns else 0.0
            if not math.isfinite(score):
                score = 0.0
            return (dist, d, -score)

        best = sorted(cand_idx, key=_key)[0]
        d_best = pd.Timestamp(frame.at[best, date_col])
        if cal is not None and event.onset in cal and d_best in cal:
            delay = int(cal.get_loc(d_best)) - int(cal.get_loc(event.onset))
        else:
            delay = int((d_best - event.onset).days)
        frame.at[best, "matched_event_onset"] = event.onset
        frame.at[best, "detection_delay"] = delay
        used_alert.add(best)
        for i in cand_idx:
            if i == best:
                continue
            frame.at[i, "is_duplicate"] = True
            used_alert.add(i)

    unmatched = frame["matched_event_onset"].isna() & ~frame["is_duplicate"].astype(bool)
    frame.loc[unmatched, "is_fp"] = True
    return frame


def summarize_calibration(rows: pd.DataFrame, *, score_col: str, label_col: str) -> dict[str, float]:
    if rows.empty:
        return {"n": 0, "brier": float("nan"), "ece": float("nan")}
    p = rows[score_col].astype(float).to_numpy()
    y = rows[label_col].astype(float).to_numpy()
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = p[mask], y[mask]
    if p.size == 0:
        return {"n": 0, "brier": float("nan"), "ece": float("nan")}
    brier = float(np.mean((p - y) ** 2))
    # 10-bin ECE
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if not m.any():
            continue
        ece += abs(float(y[m].mean()) - float(p[m].mean())) * (m.sum() / p.size)
    return {"n": int(p.size), "brier": brier, "ece": float(ece)}


__all__ = [
    "InstrumentMembership",
    "TrendEvent",
    "UniverseAuditResult",
    "UniverseMode",
    "attach_instrument_names",
    "audit_cluster_universe",
    "build_expanding_crossfit_predictions",
    "canonical_config_hash",
    "cluster_positive_events",
    "EARLY_REVERSAL_HORIZON",
    "FINAL_REVERSAL_HORIZON",
    "causal_bias_from_prior",
    "causal_bias_from_switch_side",
    "compute_causal_prior_context",
    "compute_dual_horizon_reversal_detail",
    "compute_switch_reversal_detail",
    "compute_switch_reversal_label",
    "compute_trend_up_label",
    "confirmation_trading_date",
    "eligible_codes_on_date",
    "is_eligible_on_date",
    "last_trading_day_before_month",
    "load_cluster_instruments",
    "match_alerts_to_events",
    "maybe_isotonic_calibrate",
    "month_key",
    "realized_sigma_20",
    "select_fdr_threshold",
    "shift_trading_days",
    "summarize_calibration",
    "trading_days_in_month",
]
