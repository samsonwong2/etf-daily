"""Tests for causal T+1 open → T+H close trade-bias walk-forward."""
from __future__ import annotations

import numpy as np
import pandas as pd

from etf_daily.lib.regime_transition_trade_bias import (
    FEATURE_WHITELIST,
    FORBIDDEN_FEATURES,
    TRADE_BIAS_SCHEMA,
    TradeBiasConfig,
    attach_trade_labels,
    build_trade_feature_frame,
    load_trade_bias_predictions,
    open_to_close_simple_return,
    run_trade_bias_walkforward,
    select_thresholds_on_history,
    write_trade_bias_outputs,
)


def _panel_pair(n_days: int = 120, n_codes: int = 3, seed: int = 7):
    idx = pd.bdate_range("2024-01-02", periods=n_days)
    rng = np.random.default_rng(seed)
    opens = {}
    closes = {}
    for i in range(n_codes):
        code = f"SH{513310 + i}"
        rets = rng.normal(0.0005, 0.015, n_days)
        # Inject a few directional stretches so logistic can learn something.
        rets[30:35] = 0.02
        rets[60:65] = -0.02
        px = 10.0 * np.exp(np.cumsum(rets))
        closes[code] = px
        # Open ≈ previous close with small gap.
        op = np.r_[px[0], px[:-1] * (1.0 + rng.normal(0.0, 0.002, n_days - 1))]
        opens[code] = op
    open_panel = pd.DataFrame(opens, index=idx)
    close_panel = pd.DataFrame(closes, index=idx)
    return open_panel, close_panel


def _switch_signals(open_panel: pd.DataFrame, *, switch_frac: float = 0.35) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows: list[dict] = []
    for code in open_panel.columns:
        for as_of in open_panel.index[15:]:
            is_switch = bool(rng.random() < switch_frac)
            side = "up" if rng.random() < 0.5 else "down"
            cdf = 0.92 if side == "up" else 0.08
            if not is_switch:
                cdf = float(rng.uniform(0.2, 0.8))
                side = None
            rows.append(
                {
                    "as_of": as_of,
                    "code": code,
                    "name": code,
                    "month": pd.Timestamp(as_of).to_period("M").strftime("%Y-%m"),
                    "quantile_switch": is_switch,
                    "switch_side": side,
                    "pred_cdf": cdf,
                    "pred_mean": float(rng.normal(0, 0.01)),
                    "pred_mean_next": float(rng.normal(0, 0.01)),
                    "pred_scale": 0.02,
                    "pred_scale_next": 0.02,
                    "return_z": float(rng.normal(0, 1)),
                    "p_switch_hmm": float(rng.uniform(0, 1)),
                    "p_into_reversal_hmm": float(rng.uniform(0, 1)),
                    "p_state_reversal": float(rng.uniform(0, 1)),
                    "transition_score": float(rng.uniform(0, 1)),
                    "regime_separated": bool(rng.random() > 0.3),
                    "label_ambiguous": bool(rng.random() > 0.6),
                }
            )
    return pd.DataFrame(rows)


def test_open_to_close_simple_return_entry_exit_and_maturity():
    open_panel, close_panel = _panel_pair(n_days=20, n_codes=1)
    code = open_panel.columns[0]
    as_of = open_panel.index[5]
    ret = open_to_close_simple_return(
        open_panel[code], close_panel[code], as_of, hold_days=5
    )
    assert ret is not None
    entry = float(open_panel[code].iloc[6])  # T+1 open
    exit_px = float(close_panel[code].iloc[10])  # T+5 close
    assert abs(ret - (exit_px / entry - 1.0)) < 1e-12

    # Last days cannot mature for hold=5.
    last = open_panel.index[-2]
    assert (
        open_to_close_simple_return(
            open_panel[code], close_panel[code], last, hold_days=5
        )
        is None
    )


def test_attach_trade_labels_only_matures_with_future_bars():
    open_panel, close_panel = _panel_pair(n_days=40, n_codes=1)
    code = open_panel.columns[0]
    signals = pd.DataFrame(
        {
            "as_of": [open_panel.index[10], open_panel.index[-1]],
            "code": [code, code],
            "quantile_switch": [True, True],
            "switch_side": ["up", "down"],
            "pred_cdf": [0.95, 0.05],
        }
    )
    out = attach_trade_labels(signals, open_panel, close_panel, hold_days=5)
    assert bool(out.iloc[0]["trade_label_mature"]) is True
    assert pd.notna(out.iloc[0]["trade_ret_h5"])
    assert out.iloc[0]["y_up"] in (0.0, 1.0)
    assert bool(out.iloc[1]["trade_label_mature"]) is False
    assert pd.isna(out.iloc[1]["trade_ret_h5"])


def test_feature_whitelist_excludes_forbidden_fields():
    overlap = set(FEATURE_WHITELIST) & set(FORBIDDEN_FEATURES)
    assert not overlap
    frame = pd.DataFrame(
        {
            "pred_cdf": [0.9],
            "switch_side": ["up"],
            "turn_kind": ["bottom_reversal"],
            "post_ret": [0.05],
            "trade_ret_h5": [0.01],
            "y_up": [1.0],
        }
    )
    feat = build_trade_feature_frame(frame)
    assert "tail_strength" in feat.columns
    assert "switch_side_up" in feat.columns
    assert feat.loc[0, "switch_side_up"] == 1.0
    # Forbidden raw columns may exist on frame but must not be used as features.
    for col in ("turn_kind", "post_ret", "trade_ret_h5", "y_up"):
        assert col not in FEATURE_WHITELIST


def test_walkforward_freezes_train_before_predict_month():
    open_panel, close_panel = _panel_pair(n_days=160, n_codes=3)
    signals = _switch_signals(open_panel)
    result = run_trade_bias_walkforward(
        signals,
        open_panel,
        close_panel,
        cfg=TradeBiasConfig(
            hold_days=5,
            min_train_rows=20,
            min_train_pos=3,
            min_train_neg=3,
        ),
    )
    preds = result["predictions"]
    assert not preds.empty
    assert "p_up" in preds.columns
    assert "trade_action" in preds.columns
    assert set(preds["schema"].dropna().unique().tolist()) == {TRADE_BIAS_SCHEMA}
    # Every scored month with a train_end must have train_end < month start.
    trained = preds[preds["model_trained"].fillna(False) & preds["model_train_end"].notna()]
    for _, row in trained.iterrows():
        train_end = pd.Timestamp(row["model_train_end"]).normalize()
        month_start = pd.Timestamp(str(row["month"]) + "-01").normalize()
        assert train_end < month_start
    assert "equity" in result
    assert "manifest" in result
    assert result["manifest"]["execution"]["entry"] == "T+1 open"


def test_poison_future_prices_do_not_change_same_day_features():
    """Future closes must not alter EOD-causal features on as_of."""
    open_panel, close_panel = _panel_pair(n_days=160, n_codes=2)
    signals = _switch_signals(open_panel, switch_frac=0.4)
    labeled = attach_trade_labels(signals, open_panel, close_panel, hold_days=5)
    from etf_daily.lib.regime_transition_trade_bias import (
        attach_causal_context_from_close,
    )

    base = attach_causal_context_from_close(labeled, close_panel)
    base = build_trade_feature_frame(base)
    # Pick a mature switch day with room after it.
    cand = base[
        base["quantile_switch"]
        & base["trade_label_mature"].fillna(False)
        & base["as_of"].lt(base["as_of"].max() - pd.Timedelta(days=20))
    ]
    assert not cand.empty
    row = cand.sort_values("as_of").iloc[len(cand) // 2]
    as_of = pd.Timestamp(row["as_of"]).normalize()
    code = str(row["code"])

    poisoned_close = close_panel.copy()
    poisoned_close.loc[poisoned_close.index > as_of, :] *= 3.0
    poisoned = attach_causal_context_from_close(labeled, poisoned_close)
    poisoned = build_trade_feature_frame(poisoned)
    left = base[(base["as_of"] == as_of) & (base["code"] == code)].iloc[0]
    right = poisoned[(poisoned["as_of"] == as_of) & (poisoned["code"] == code)].iloc[0]
    for col in FEATURE_WHITELIST:
        lv = left.get(col)
        rv = right.get(col)
        if pd.isna(lv) and pd.isna(rv):
            continue
        assert abs(float(lv) - float(rv)) < 1e-12, col


def test_walkforward_train_ignores_current_and_future_months():
    open_panel, close_panel = _panel_pair(n_days=160, n_codes=2)
    signals = _switch_signals(open_panel, switch_frac=0.4)
    result = run_trade_bias_walkforward(
        signals,
        open_panel,
        close_panel,
        cfg=TradeBiasConfig(hold_days=5, min_train_rows=20, min_train_pos=3, min_train_neg=3),
    )
    preds = result["predictions"]
    trained = preds[preds["model_trained"].fillna(False) & preds["model_train_end"].notna()]
    assert not trained.empty
    for _, row in trained.iterrows():
        assert pd.Timestamp(row["model_train_end"]) < pd.Timestamp(
            str(row["month"]) + "-01"
        )

def test_threshold_selection_ignores_empty_history():
    thr = select_thresholds_on_history(pd.DataFrame())
    assert thr["buy_thr"] == 0.60
    assert thr["sell_thr"] == 0.40
    assert thr["reason"] == "fallback_default"


def test_write_and_load_trade_bias_predictions(tmp_path):
    open_panel, close_panel = _panel_pair(n_days=120, n_codes=2)
    signals = _switch_signals(open_panel, switch_frac=0.4)
    result = run_trade_bias_walkforward(
        signals,
        open_panel,
        close_panel,
        cfg=TradeBiasConfig(hold_days=5, min_train_rows=15, min_train_pos=2, min_train_neg=2),
    )
    out = write_trade_bias_outputs(result, tmp_path)
    assert (out / "trade_bias_predictions.csv").exists()
    assert (out / "trade_bias_metrics.csv").exists()
    assert (out / "trade_bias_metrics.json").exists()
    assert (out / "trade_bias_equity.csv").exists()
    assert (out / "trade_bias_manifest.json").exists()
    loaded = load_trade_bias_predictions(out / "trade_bias_predictions.csv")
    assert loaded is not None
    assert "trade_action" in loaded.columns
    assert load_trade_bias_predictions(tmp_path / "missing.csv") is None


def test_sparse_next_features_do_not_block_training():
    """pred_mean_next-like sparse columns must be dropped, not wipe all rows."""
    open_panel, close_panel = _panel_pair(n_days=140, n_codes=2)
    signals = _switch_signals(open_panel, switch_frac=0.45)
    # Mimic production: almost all pred_mean_next / pred_scale_next are NaN.
    signals["pred_mean_next"] = np.nan
    signals["pred_scale_next"] = np.nan
    signals.loc[signals.index[::20], "pred_mean_next"] = 0.01
    signals.loc[signals.index[::20], "pred_scale_next"] = 0.02
    result = run_trade_bias_walkforward(
        signals,
        open_panel,
        close_panel,
        cfg=TradeBiasConfig(hold_days=5, min_train_rows=20, min_train_pos=3, min_train_neg=3),
    )
    preds = result["predictions"]
    assert preds["model_trained"].fillna(False).any()
    assert preds["p_up"].notna().any()
    assert set(preds["trade_action"]) & {"buy_bias", "sell_bias", "observe_only"}
    # Sparse next-step features should not appear in usable set for late months.
    meta = result["manifest"]["month_meta"]
    trained = [m for m in meta if m.get("trained")]
    assert trained
    for m in trained[-3:]:
        used = m.get("feature_cols_used") or []
        assert "pred_mean_next" not in used
        assert "pred_scale_next" not in used
