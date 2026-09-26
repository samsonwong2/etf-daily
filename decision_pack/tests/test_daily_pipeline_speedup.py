from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _FakePm:
    def _init_qlib(self, _uri) -> None:
        return None


def _fake_oos_one_stub(*, code: str, **kwargs):  # noqa: ANN003
    return dict(
        code=code,
        method="hybrid_ma_adx",
        edge=0.01,
        edge_vs_fixed=0.0,
        fixed_hybrid_edge=0.01,
        win=0.5,
        up_buy="x",
        down_buy="x",
        range_buy="x",
        up_n=1,
        down_n=1,
        range_n=1,
    )


def _fake_load_ohlcv(*args, **kwargs):  # noqa: ANN002, ANN003
    return pd.DataFrame()


def _load_backtest_mod():
    path = _PROJECT_ROOT / "decision_pack/scripts/backtest_regime_transition_signals.py"
    spec = importlib.util.spec_from_file_location("bt_speedup_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_adaptive_pool_mod():
    path = _PROJECT_ROOT / "decision_pack/scripts/plot_adaptive_stage_pool.py"
    name = "plot_adaptive_stage_pool"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_plot_mod():
    path = _PROJECT_ROOT / "decision_pack/scripts/plot_regime_transition_example.py"
    spec = importlib.util.spec_from_file_location("plot_speedup_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_parse_args_jobs_default_is_one():
    mod = _load_adaptive_pool_mod()
    args = mod.parse_args(["--end-date", "2026-09-03"])
    assert args.jobs == 1


def test_build_elig_by_day_matches_naive():
    mod = _load_backtest_mod()
    from decision_pack.src.regime_transition_validation import (
        InstrumentMembership,
        UniverseAuditResult,
        eligible_codes_on_date,
    )

    dates = pd.date_range("2026-09-01", periods=5, freq="B")
    panel = pd.DataFrame(
        {
            "SHAAA": [1.0, 1.1, 1.2, 1.3, 1.4],
            "SHBBB": [2.0, 2.1, 2.2, 2.3, 2.4],
        },
        index=dates,
    )
    audit = UniverseAuditResult(
        mode="cluster_selected",
        n_instruments=2,
        n_codes_with_dates=2,
        missing_list_date=0,
        missing_end_date=0,
        messages=(),
        instruments=(
            InstrumentMembership(
                code="SHAAA",
                list_date=pd.Timestamp("2020-01-01"),
                end_date=None,
            ),
            InstrumentMembership(
                code="SHBBB",
                list_date=pd.Timestamp("2020-01-01"),
                end_date=None,
            ),
        ),
    )
    days = pd.DatetimeIndex(dates)
    cached = mod._build_elig_by_day(
        audit, days, panel, min_history=2, min_valid_in_20=2
    )
    for d in days:
        naive = set(
            eligible_codes_on_date(
                audit, d, panel, min_history=2, min_valid_in_20=2
            )
        )
        assert cached[d] == naive


def test_run_oos_batch_jobs_parity(monkeypatch: pytest.MonkeyPatch):
    mod = _load_adaptive_pool_mod()
    monkeypatch.setattr(mod, "oos_one", _fake_oos_one_stub)
    monkeypatch.setattr(mod, "_load_ohlcv", _fake_load_ohlcv)

    ctx = dict(
        pm=_FakePm(),
        ev=object(),
        oos=pd.DataFrame(),
        rev=pd.DataFrame(),
        evt=pd.DataFrame(),
        name_map={},
        live=None,
        rv20_anchor=None,
        rv5_anchor=None,
        rv20_bond_anchor=None,
        rv5_bond_anchor=None,
        hrp_mode_by_code=None,
        configs={
            "SHAAA": {"method": "hybrid_ma_adx", "rules": {}},
            "SHBBB": {"method": "hybrid_ma_adx", "rules": {}},
        },
        enable_ru_diag=False,
        enable_shallow_up_trade=False,
        out_dir=Path("/tmp"),
        start="2026-04-01",
        end="2026-09-03",
        train_cutoff="2026-04-01",
        fair_path_extra_trail_years="off",
    )
    codes = ["SHAAA", "SHBBB"]
    rows1, err1 = mod._run_oos_batch(codes=codes, jobs=1, ctx=ctx, fail_fast=False)
    rows2, err2 = mod._run_oos_batch(codes=codes, jobs=2, ctx=ctx, fail_fast=False)

    assert not err1 and not err2
    assert {r["code"] for r in rows1} == {r["code"] for r in rows2} == set(codes)
    for c in codes:
        r1 = next(r for r in rows1 if r["code"] == c)
        r2 = next(r for r in rows2 if r["code"] == c)
        assert r1["edge"] == r2["edge"]
        assert r1["method"] == r2["method"]


def test_build_figure_ohlcv_full_skips_qlib_load(monkeypatch: pytest.MonkeyPatch):
    mod = _load_plot_mod()
    oos = pd.DataFrame(
        {
            "as_of": ["2026-06-01"],
            "code": ["SH513310"],
            "name": ["test"],
            "signal_level": ["NONE"],
            "quantile_switch": [False],
            "pred_cdf": [0.5],
            "return_z": [0.0],
            "p_switch_hmm": [0.1],
            "p_into_reversal_hmm": [0.1],
        }
    )
    rev = pd.DataFrame(
        {
            "as_of": ["2026-06-01"],
            "code": ["SH513310"],
            "label_switch_reversal": [False],
            "turn_kind": ["none"],
            "jump_role": ["none"],
            "prior_ret": [0.0],
            "jump_ret": [0.0],
            "post_ret": [0.0],
            "final_confirmed_on": [None],
            "early_label_switch_reversal": [False],
            "early_turn_kind": ["none"],
            "early_jump_role": ["none"],
            "early_prior_ret": [0.0],
            "early_jump_ret": [0.0],
            "early_post_ret": [0.0],
            "early_confirmed_on": [None],
            "early_to_final_status": ["none"],
            "causal_prior_ret": [0.0],
            "causal_jump_ret": [0.0],
            "causal_thr_prior": [0.01],
            "causal_prior_strong": [False],
        }
    )
    evt = pd.DataFrame(columns=["as_of", "code"])
    dates = pd.date_range("2026-01-01", periods=120, freq="B")
    ohlcv = pd.DataFrame(
        {
            "datetime": dates,
            "instrument": "SH513310",
            "$open": 1.0,
            "$high": 1.1,
            "$low": 0.9,
            "$close": 1.0 + pd.Series(range(len(dates))) * 0.001,
            "$volume": 1000.0,
            "$pct_change": 0.0,
        }
    )
    for w in (5, 10, 20, 60):
        ohlcv[f"MA{w}"] = ohlcv["$close"]

    loader = mock.Mock(side_effect=AssertionError("load_qlib_ohlcv should not run"))
    monkeypatch.setattr(mod, "load_qlib_ohlcv", loader)

    fig, overlay, name = mod.build_figure(
        code="SH513310",
        start_date="2026-04-01",
        end_date="2026-06-01",
        oos=oos,
        rev=rev,
        evt=evt,
        init_qlib=False,
        fair_path_extra_trail_years="off",
        ohlcv_full=ohlcv,
    )
    assert fig is not None
    assert not overlay.empty
    assert name
    loader.assert_not_called()
