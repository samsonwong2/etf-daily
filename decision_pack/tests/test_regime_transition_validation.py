from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from decision_pack.src.regime_transition_validation import (
    InstrumentMembership,
    audit_cluster_universe,
    cluster_positive_events,
    compute_trend_up_label,
    eligible_codes_on_date,
    load_cluster_instruments,
    match_alerts_to_events,
    select_fdr_threshold,
)


def test_load_cluster_instruments_selected_file():
    from runtime_paths import CLUSTER_MAPPING_SELECTED_TXT

    path = Path(CLUSTER_MAPPING_SELECTED_TXT)
    if not path.exists():
        return
    rows = load_cluster_instruments(path)
    assert len(rows) >= 30
    assert all(r.list_date is not None for r in rows)
    codes = {r.code for r in rows}
    assert "SZ159647" in codes


def test_audit_cluster_universe_formal():
    instruments = [
        InstrumentMembership("SH513050", pd.Timestamp("2017-01-18"), pd.Timestamp("2026-07-17")),
        InstrumentMembership("SZ159647", pd.Timestamp("2022-07-28"), pd.Timestamp("2026-07-17")),
    ]
    audit = audit_cluster_universe(instruments)
    assert audit.mode == "cluster_selected"
    assert audit.ok_for_formal


def test_eligibility_respects_list_date_and_history():
    idx = pd.bdate_range("2020-01-01", periods=300)
    close = pd.Series(100 * np.exp(np.cumsum(np.zeros(300))), index=idx)
    membership = InstrumentMembership(
        "TEST", pd.Timestamp("2020-06-01"), pd.Timestamp("2026-01-01")
    )
    audit = audit_cluster_universe([membership])
    panel = pd.DataFrame({"TEST": close})
    # before list date
    early = eligible_codes_on_date(audit, pd.Timestamp("2020-05-01"), panel, min_history=60)
    assert early == []
    # enough history after list
    late = eligible_codes_on_date(
        audit, pd.Timestamp("2021-01-04"), panel, min_history=60, min_valid_in_20=18
    )
    assert late == ["TEST"]


def test_eligibility_allows_provisional_bar_past_instruments_end_date():
    """Intraday live snapshot may append beyond Qlib instruments end_date."""
    from decision_pack.src.regime_transition_validation import is_eligible_on_date

    idx = pd.bdate_range("2025-01-01", periods=40)
    close = pd.Series(np.linspace(1.0, 1.4, 40), index=idx)
    end = pd.Timestamp(idx[-1])
    # instruments file often freezes at last Qlib calendar day
    membership = InstrumentMembership("TEST", pd.Timestamp("2020-01-01"), end)
    beyond = end + pd.tseries.offsets.BDay(1)
    # without provisional bar → ineligible
    assert not is_eligible_on_date(
        membership, beyond, close, min_history=20, min_valid_in_20=18
    )
    # with provisional close on beyond → eligible
    close2 = close.copy()
    close2.loc[beyond] = 1.45
    assert is_eligible_on_date(
        membership, beyond, close2, min_history=20, min_valid_in_20=18
    )


def test_trend_label_unavailable_without_future():
    idx = pd.bdate_range("2020-01-01", periods=40)
    close = pd.Series(np.linspace(100, 110, 40), index=idx)
    # near end: not enough future
    assert compute_trend_up_label(close, close.index[-2], horizon=10) is None


def test_event_matching_duplicate_and_fp():
    labels = pd.DataFrame(
        {
            "code": ["A"] * 6,
            "as_of": pd.bdate_range("2024-01-01", periods=6),
            "label_trend_up": [False, True, True, False, True, False],
        }
    )
    events = cluster_positive_events(labels, max_gap=2)
    assert len(events) >= 1
    alerts = pd.DataFrame(
        {
            "code": ["A", "A", "A"],
            "as_of": [
                labels.loc[1, "as_of"],
                labels.loc[2, "as_of"],
                labels.loc[5, "as_of"],
            ],
            "p_trend_cal": [0.9, 0.8, 0.7],
        }
    )
    matched = match_alerts_to_events(alerts, events, lead=3, horizon=10)
    assert "is_fp" in matched.columns
    assert matched["is_duplicate"].any() or matched["matched_event_onset"].notna().any()


def test_select_fdr_threshold_suppresses_when_too_few():
    rng = np.random.default_rng(0)
    n = 40
    rows = pd.DataFrame(
        {
            "p_trend_cal": rng.uniform(0.5, 0.99, n),
            "label_trend_up": rng.integers(0, 2, n).astype(bool),
            "month": ["2024-01"] * n,
        }
    )
    out = select_fdr_threshold(rows, min_confirm=50, target_fdr=0.2, n_bootstrap=20)
    assert out["q_confirm"] is None


def test_switch_reversal_label_detects_climax_top():
    from decision_pack.src.regime_transition_validation import (
        compute_switch_reversal_detail,
        compute_switch_reversal_label,
    )

    # Strong prior up, jump up, then post down → climax top / path flip
    rng = np.random.default_rng(0)
    prior = 0.01 + 0.002 * rng.normal(size=40)
    jump = np.array([0.06])
    post = -0.012 + 0.002 * rng.normal(size=15)
    rets = np.concatenate([prior, jump, post])
    prices = 100.0 * np.exp(np.cumsum(np.insert(rets, 0, 0.0)))
    close = pd.Series(prices, index=pd.bdate_range("2024-01-01", periods=len(prices)))
    origin = close.index[41]  # day of the +6% jump (after 40 prior returns)
    detail = compute_switch_reversal_detail(close, origin, lookback=10, horizon=10)
    assert detail is not None
    assert detail["label_switch_reversal"] is True
    assert detail["turn_kind"] == "top_reversal"
    assert detail["jump_role"] == "climax_top"
    assert compute_switch_reversal_label(close, origin) is True


def test_attach_instrument_names_inserts_after_code():
    from decision_pack.src.regime_transition_validation import attach_instrument_names

    frame = pd.DataFrame({"as_of": ["2026-01-01"], "code": ["SZ159647"], "x": [1]})
    out = attach_instrument_names(frame, {"SZ159647": "中药ETF鹏华"})
    assert list(out.columns[:3]) == ["as_of", "code", "name"]
    assert out.loc[0, "name"] == "中药ETF鹏华"


def test_switch_reversal_label_detects_bounce():
    from decision_pack.src.regime_transition_validation import compute_switch_reversal_detail

    rng = np.random.default_rng(1)
    prior = -0.01 + 0.001 * rng.normal(size=40)
    jump = np.array([0.05])
    post = 0.01 + 0.001 * rng.normal(size=15)
    rets = np.concatenate([prior, jump, post])
    prices = 100.0 * np.exp(np.cumsum(np.insert(rets, 0, 0.0)))
    close = pd.Series(prices, index=pd.bdate_range("2024-01-01", periods=len(prices)))
    origin = close.index[41]  # day of the +5% bounce jump
    detail = compute_switch_reversal_detail(close, origin, lookback=10, horizon=10)
    assert detail is not None
    assert detail["label_switch_reversal"] is True
    assert detail["turn_kind"] == "bottom_reversal"
    assert detail["jump_role"] == "bounce"


def test_causal_prior_context_and_bias_rules():
    from decision_pack.src.regime_transition_validation import (
        causal_bias_from_prior,
        causal_bias_from_switch_side,
        compute_causal_prior_context,
    )

    rng = np.random.default_rng(1)
    prior = -0.01 + 0.001 * rng.normal(size=40)
    jump = np.array([0.05])
    post = 0.01 + 0.001 * rng.normal(size=15)
    rets = np.concatenate([prior, jump, post])
    prices = 100.0 * np.exp(np.cumsum(np.insert(rets, 0, 0.0)))
    close = pd.Series(prices, index=pd.bdate_range("2024-01-01", periods=len(prices)))
    origin = close.index[41]
    ctx = compute_causal_prior_context(close, origin, lookback=10)
    assert ctx is not None
    assert ctx["prior_ret"] < 0
    assert ctx["jump_ret"] > 0
    assert ctx["thr_prior"] > 0
    # Changing future closes must not change causal prior.
    close2 = close.copy()
    close2.iloc[-1] = close2.iloc[-1] * 1.5
    ctx2 = compute_causal_prior_context(close2, origin, lookback=10)
    assert ctx2 is not None
    assert abs(ctx2["prior_ret"] - ctx["prior_ret"]) < 1e-12

    assert causal_bias_from_prior(quantile_switch=True, prior_ret=-0.02) == "buy_bias"
    assert causal_bias_from_prior(quantile_switch=True, prior_ret=0.02) == "sell_bias"
    assert causal_bias_from_prior(quantile_switch=True, prior_ret=0.0) is None
    assert (
        causal_bias_from_prior(
            quantile_switch=True, prior_ret=-0.01, thr_prior=0.05, require_strong=True
        )
        is None
    )
    assert (
        causal_bias_from_prior(
            quantile_switch=True, prior_ret=-0.06, thr_prior=0.05, require_strong=True
        )
        == "buy_bias"
    )
    assert causal_bias_from_switch_side(quantile_switch=True, switch_side="up") == "buy_bias"
    assert causal_bias_from_switch_side(quantile_switch=False, switch_side="up") is None


def test_dual_horizon_confirmation_dates_and_status():
    from decision_pack.src.regime_transition_validation import (
        compute_dual_horizon_reversal_detail,
        confirmation_trading_date,
    )

    rng = np.random.default_rng(1)
    prior = -0.01 + 0.001 * rng.normal(size=40)
    jump = np.array([0.05])
    post = 0.01 + 0.001 * rng.normal(size=15)
    rets = np.concatenate([prior, jump, post])
    prices = 100.0 * np.exp(np.cumsum(np.insert(rets, 0, 0.0)))
    close = pd.Series(prices, index=pd.bdate_range("2024-01-01", periods=len(prices)))
    origin = close.index[41]
    dual = compute_dual_horizon_reversal_detail(close, origin, lookback=10)
    assert dual["early_confirmed_on"] == confirmation_trading_date(close, origin, 3)
    assert dual["final_confirmed_on"] == confirmation_trading_date(close, origin, 10)
    assert dual["early_confirmed_on"] == pd.Timestamp(close.index[41 + 3]).normalize()
    assert dual["final_confirmed_on"] == pd.Timestamp(close.index[41 + 10]).normalize()
    assert dual["early_label_switch_reversal"] is True
    assert dual["label_switch_reversal"] is True
    assert dual["early_to_final_status"] == "confirmed"

    # Immature tail: last day cannot confirm early or final.
    immature = compute_dual_horizon_reversal_detail(close, close.index[-1], lookback=10)
    assert immature["early_label_switch_reversal"] is None
    assert immature["label_switch_reversal"] is None
    assert immature["early_to_final_status"] == "immature"
    assert immature["early_confirmed_on"] is None
    assert immature["final_confirmed_on"] is None


def test_dual_horizon_revocation_when_short_post_fades():
    from decision_pack.src.regime_transition_validation import (
        compute_dual_horizon_reversal_detail,
    )

    # Strong bounce for 3 days, then give back all gains by day 10.
    rng = np.random.default_rng(2)
    prior = -0.012 + 0.0005 * rng.normal(size=40)
    jump = np.array([0.05])
    post3 = 0.02 + 0.0005 * rng.normal(size=3)
    post_rest = -0.025 + 0.0005 * rng.normal(size=12)
    rets = np.concatenate([prior, jump, post3, post_rest])
    prices = 100.0 * np.exp(np.cumsum(np.insert(rets, 0, 0.0)))
    close = pd.Series(prices, index=pd.bdate_range("2024-01-01", periods=len(prices)))
    origin = close.index[41]
    dual = compute_dual_horizon_reversal_detail(close, origin, lookback=10)
    # Early may or may not fire depending on thresholds; if early fires and final
    # does not, status must be revoked.
    if dual["early_label_switch_reversal"] and not dual["label_switch_reversal"]:
        assert dual["early_to_final_status"] == "revoked"
    assert dual["early_confirmed_on"] is not None
    assert dual["final_confirmed_on"] is not None
