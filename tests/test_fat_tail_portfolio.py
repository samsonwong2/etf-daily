from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from etf_daily.scripts.generate_fat_tail_risk_report import (
    build_cluster_pool_risk_csv,
    build_fat_tail_report,
    lookback_output_suffix,
    suffixed_output_path,
)
from etf_daily.lib.fat_tail_portfolio import (
    build_mean_signal_table,
    build_return_max_proposal,
    concavity_flags,
    crash_cvar,
    kappa_to_n,
    kelly_cap,
    propose_from_holdings,
    return_tilt_weights,
    solve_barbell_w,
)
from etf_daily.lib.fat_tail_risk import FatTailConfig, gpd_loss_at_prob


def _frame(rows: list[dict]) -> pd.DataFrame:
    cols = [
        "code",
        "name",
        "weight",
        "barbell_leg",
        "alpha_left",
        "kappa",
        "cvar_left",
        "raw_mean",
        "shadow_mean",
        "shrink_factor",
        "shrunk_mean",
        "xi_left",
        "beta_left",
        "threshold_left",
        "nu_left",
        "flags",
    ]
    out = pd.DataFrame(rows)
    for c in cols:
        if c not in out.columns:
            out[c] = None
    return out[cols]


# --------------------------------------------------------------------------- #
# kappa -> n
# --------------------------------------------------------------------------- #
def test_kappa_to_n_monotonic_and_capped():
    cfg = FatTailConfig(n_min=12, n_max=40)
    assert kappa_to_n(0.1, cfg) < kappa_to_n(0.5, cfg) <= kappa_to_n(0.9, cfg)
    assert kappa_to_n(0.1, cfg) >= cfg.n_min
    assert kappa_to_n(0.99, cfg) == cfg.n_max
    assert kappa_to_n(None, cfg) >= cfg.n_min


# --------------------------------------------------------------------------- #
# barbell w solve
# --------------------------------------------------------------------------- #
def test_solve_barbell_w_tighter_budget_more_safe():
    w_loose = solve_barbell_w(0.03, 0.06, 0.0)
    w_tight = solve_barbell_w(0.02, 0.06, 0.0)
    assert w_loose == pytest.approx(0.5)
    assert w_tight > w_loose


def test_solve_barbell_w_fatter_risk_more_safe():
    w_small = solve_barbell_w(0.03, 0.06, 0.0)
    w_big = solve_barbell_w(0.03, 0.09, 0.0)
    assert w_big > w_small


def test_solve_barbell_w_risk_within_budget_no_safe():
    assert solve_barbell_w(0.06, 0.04, 0.0) == 0.0
    assert 0.0 <= solve_barbell_w(0.01, 0.5, 0.001) <= 1.0


# --------------------------------------------------------------------------- #
# crash CVaR (correlation -> 1, conservative fill)
# --------------------------------------------------------------------------- #
def test_crash_cvar_no_diversification_average():
    frame = _frame(
        [
            {"code": "A", "name": "A", "weight": 0.5, "cvar_left": -0.04},
            {"code": "B", "name": "B", "weight": 0.5, "cvar_left": -0.08},
        ]
    )
    val, fill = crash_cvar({"A": 1.0, "B": 1.0}, frame)
    assert val == pytest.approx(0.06)
    assert fill == 0.0


def test_crash_cvar_conservative_fill_for_missing():
    frame = _frame(
        [
            {"code": "A", "name": "A", "weight": 0.5, "cvar_left": -0.04},
            {"code": "B", "name": "B", "weight": 0.5, "cvar_left": None},
        ]
    )
    val, fill = crash_cvar({"A": 1.0, "B": 1.0}, frame)
    assert val == pytest.approx(0.04)  # B filled with worst-available 0.04
    assert fill == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# concavity / fragility
# --------------------------------------------------------------------------- #
def test_gpd_loss_at_prob_guards():
    assert gpd_loss_at_prob(xi=0.3, beta=0.01, threshold=0.02, nu=0.075, tail_prob=0.1) is None
    assert gpd_loss_at_prob(xi=None, beta=0.01, threshold=0.02, nu=0.075, tail_prob=0.01) is None
    loss = gpd_loss_at_prob(xi=0.3, beta=0.01, threshold=0.02, nu=0.075, tail_prob=0.01)
    assert loss is not None and loss > 0.02


def test_concavity_flags_fat_vs_thin():
    cfg = FatTailConfig()
    frame = _frame(
        [
            {
                "code": "FAT",
                "name": "fat",
                "weight": 0.5,
                "alpha_left": 1.1,
                "xi_left": 0.9,
                "beta_left": 0.01,
                "threshold_left": 0.02,
                "nu_left": 0.075,
            },
            {
                "code": "THIN",
                "name": "thin",
                "weight": 0.5,
                "alpha_left": 10.0,
                "xi_left": 0.1,
                "beta_left": 0.01,
                "threshold_left": 0.02,
                "nu_left": 0.075,
            },
        ]
    )
    out = concavity_flags(frame, cfg).set_index("code")
    assert bool(out.loc["FAT", "fragile"]) is True
    assert bool(out.loc["THIN", "fragile"]) is False
    assert out.loc["FAT", "concavity_ratio"] > out.loc["THIN", "concavity_ratio"]


# --------------------------------------------------------------------------- #
# Kelly cap
# --------------------------------------------------------------------------- #
def test_kelly_cap_shrinks_with_fatter_tail_and_bounded():
    cfg = FatTailConfig(kelly_fraction=0.25, max_name_weight=0.20)
    frame = _frame(
        [
            {"code": "THIN", "name": "t", "weight": 0.1, "shadow_mean": 0.001, "cvar_left": -0.03},
            {"code": "FAT", "name": "f", "weight": 0.1, "shadow_mean": 0.001, "cvar_left": -0.06},
            {"code": "NEG", "name": "n", "weight": 0.1, "shadow_mean": -0.001, "cvar_left": -0.03},
        ]
    )
    out = kelly_cap(frame, cfg).set_index("code")
    assert out.loc["FAT", "kelly_cap"] < out.loc["THIN", "kelly_cap"]
    assert out.loc["THIN", "kelly_cap"] <= cfg.max_name_weight
    assert out.loc["NEG", "kelly_cap"] == 0.0  # no positive edge -> no size


# --------------------------------------------------------------------------- #
# mean signal table
# --------------------------------------------------------------------------- #
def test_mean_signal_table_ratio():
    frame = _frame(
        [
            {
                "code": "A",
                "name": "A",
                "weight": 1.0,
                "raw_mean": 0.001,
                "shadow_mean": 0.0033,
                "shrink_factor": 0.5,
                "shrunk_mean": 0.0005,
            }
        ]
    )
    tbl = build_mean_signal_table(frame)
    assert tbl.loc[0, "shadow_to_sample"] == pytest.approx(3.3)


# --------------------------------------------------------------------------- #
# barbell proposal from holdings
# --------------------------------------------------------------------------- #
def test_propose_from_holdings_structure():
    cfg = FatTailConfig(loss_budget_k=0.03)
    frame = _frame(
        [
            {"code": "SH511010", "name": "bond1", "weight": 0.0, "barbell_leg": "safe_leg", "cvar_left": -0.005, "kappa": 0.2, "alpha_left": 3.3, "flags": ()},
            {"code": "SH511260", "name": "bond2", "weight": 0.0, "barbell_leg": "safe_leg", "cvar_left": -0.006, "kappa": 0.2, "alpha_left": 7.0, "flags": ()},
            {"code": "EQ1", "name": "eq1", "weight": 0.4, "barbell_leg": "risk_leg", "cvar_left": -0.05, "kappa": 0.1, "alpha_left": 5.0, "flags": ()},
            {"code": "EQ2", "name": "eq2", "weight": 0.4, "barbell_leg": "risk_leg", "cvar_left": -0.06, "kappa": 0.15, "alpha_left": 4.0, "flags": ()},
            {"code": "BAD", "name": "bad", "weight": 0.2, "barbell_leg": "avoid", "cvar_left": -0.20, "kappa": 0.4, "alpha_left": 1.2, "flags": ()},
        ]
    )
    prop = propose_from_holdings(frame, {"SH511010", "SH511260"}, cfg)
    assert "BAD" in prop.excluded_avoid
    assert prop.n_risk == 2 and prop.n_safe == 2
    assert 0.0 <= prop.w_safe <= 1.0
    total = sum(prop.weights.values())
    assert total == pytest.approx(1.0, abs=1e-9)
    # avoid name not in generated weights
    assert "BAD" not in prop.weights
    # portfolio crash CVaR should not exceed the (larger) risk-leg crash CVaR
    assert prop.portfolio_crash_cvar <= prop.cvar_risk_crash + 1e-9


# --------------------------------------------------------------------------- #
# return-max tilt + proposal (no safe leg)
# --------------------------------------------------------------------------- #
def test_return_tilt_weights_orders_by_kelly_and_zeros_no_edge():
    cfg = FatTailConfig(max_name_weight=0.8)
    frame = _frame(
        [
            {"code": "A", "name": "A", "weight": 0.0, "shrunk_mean": 0.002, "cvar_left": -0.04},
            {"code": "B", "name": "B", "weight": 0.0, "shrunk_mean": 0.001, "cvar_left": -0.04},
            {"code": "C", "name": "C", "weight": 0.0, "shrunk_mean": -0.001, "cvar_left": -0.04},
        ]
    )
    w, no_edge = return_tilt_weights(frame, ["A", "B", "C"], cfg)
    assert no_edge is False
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)
    assert w["A"] > w["B"] > 0.0
    assert w["C"] == pytest.approx(0.0)


def test_return_tilt_weights_respects_cap_and_sums_one():
    cfg = FatTailConfig(max_name_weight=0.2)
    rows = [{"code": "BIG", "name": "BIG", "weight": 0.0, "shrunk_mean": 0.10, "cvar_left": -0.04}]
    rows += [
        {"code": f"S{i}", "name": f"S{i}", "weight": 0.0, "shrunk_mean": 0.001, "cvar_left": -0.04}
        for i in range(9)
    ]
    w, no_edge = return_tilt_weights(_frame(rows), [r["code"] for r in rows], cfg)
    assert no_edge is False
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)
    assert max(w.values()) <= cfg.max_name_weight + 1e-9


def test_return_tilt_weights_equal_fallback_when_no_edge():
    cfg = FatTailConfig(max_name_weight=0.8)
    frame = _frame(
        [
            {"code": "A", "name": "A", "weight": 0.0, "shrunk_mean": -0.002, "cvar_left": -0.04},
            {"code": "B", "name": "B", "weight": 0.0, "shrunk_mean": -0.001, "cvar_left": -0.04},
        ]
    )
    w, no_edge = return_tilt_weights(frame, ["A", "B"], cfg)
    assert no_edge is True
    assert w["A"] == pytest.approx(0.5)
    assert w["B"] == pytest.approx(0.5)


def test_build_return_max_proposal_fully_invested_no_safe_leg():
    cfg = FatTailConfig(max_name_weight=0.8)
    frame = _frame(
        [
            {"code": "EQ1", "name": "eq1", "weight": 0.5, "shrunk_mean": 0.002, "cvar_left": -0.04, "kappa": 0.1, "alpha_left": 5.0},
            {"code": "EQ2", "name": "eq2", "weight": 0.5, "shrunk_mean": 0.001, "cvar_left": -0.05, "kappa": 0.12, "alpha_left": 4.0},
        ]
    )
    prop = build_return_max_proposal(frame, candidate_codes=["EQ1", "EQ2"], cfg=cfg, label="X")
    assert prop.objective == "return_max"
    assert prop.w_safe == 0.0 and prop.w_risk == 1.0
    assert prop.n_safe == 0
    assert sum(prop.weights.values()) == pytest.approx(1.0, abs=1e-9)
    assert prop.portfolio_crash_cvar == pytest.approx(prop.cvar_risk_crash)


def test_propose_from_holdings_return_max_excludes_bonds():
    cfg = FatTailConfig(
        proposal_objective="return_max",
        barbell_exclude_codes=("SH511010", "SH511260"),
        max_name_weight=0.8,
    )
    frame = _frame(
        [
            {"code": "SH511010", "name": "bond1", "weight": 0.2, "barbell_leg": "safe_leg", "cvar_left": -0.005, "shrunk_mean": 0.0001, "kappa": 0.2, "alpha_left": 3.3, "flags": ()},
            {"code": "SH511260", "name": "bond2", "weight": 0.2, "barbell_leg": "safe_leg", "cvar_left": -0.006, "shrunk_mean": 0.0001, "kappa": 0.2, "alpha_left": 7.0, "flags": ()},
            {"code": "EQ1", "name": "eq1", "weight": 0.3, "barbell_leg": "risk_leg", "cvar_left": -0.05, "shrunk_mean": 0.002, "kappa": 0.1, "alpha_left": 5.0, "flags": ()},
            {"code": "EQ2", "name": "eq2", "weight": 0.3, "barbell_leg": "risk_leg", "cvar_left": -0.06, "shrunk_mean": 0.001, "kappa": 0.15, "alpha_left": 4.0, "flags": ()},
        ]
    )
    prop = propose_from_holdings(frame, {"SH511010", "SH511260"}, cfg)
    assert prop.objective == "return_max"
    assert prop.w_safe == 0.0
    assert "SH511010" not in prop.weights and "SH511260" not in prop.weights
    assert set(prop.weights) == {"EQ1", "EQ2"}
    assert sum(prop.weights.values()) == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# report smoke (markdown + proposals df + new sections)
# --------------------------------------------------------------------------- #
def _make_synthetic_pack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "tier_decision.json").write_text(
        json.dumps({"as_of": "2024-06-15"}, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"code": "SHAAA", "name": "Test A", "market_value": 1000.0, "weight": 0.7, "weight_total": 0.7, "source": "live"},
            {"code": "SHBBB", "name": "Test B", "market_value": 300.0, "weight": 0.3, "weight_total": 0.3, "source": "live"},
        ]
    ).to_csv(pack_dir / "portfolio_snapshot.csv", index=False)

    dates = pd.bdate_range("2022-01-01", periods=600)
    rng = np.random.default_rng(6)
    panel = pd.DataFrame(
        {
            "SHAAA": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, size=len(dates)))),
            "SHBBB": 50 * np.exp(np.cumsum(rng.normal(0, 0.013, size=len(dates)))),
        },
        index=dates,
    )
    cluster_file = tmp_path / "cluster.txt"
    cluster_file.write_text("SHAAA\t2022-01-01\t2024-06-15\nSHBBB\t2022-01-01\t2024-06-15\n", encoding="utf-8")

    monkeypatch.setattr(
        "etf_daily.scripts.generate_fat_tail_risk_report.load_qlib_close",
        lambda uri, codes, start, end: panel,
    )
    monkeypatch.setattr(
        "etf_daily.scripts.generate_fat_tail_risk_report.auto_qfq_adjust_close_panel",
        lambda p: (p, {}),
    )
    monkeypatch.setattr(
        "etf_daily.scripts.generate_fat_tail_risk_report.load_fund_name_map",
        lambda *_args, **_kwargs: {},
    )
    return pack_dir, cluster_file


def test_report_budget_barbell_sections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pack_dir, cluster_file = _make_synthetic_pack(tmp_path, monkeypatch)
    md, proposals, _ = build_fat_tail_report(
        pack_dir=pack_dir,
        cluster_mapping_path=cluster_file,
        cfg=FatTailConfig(min_samples=120, lookback_days=0, proposal_objective="budget_barbell"),
    )
    assert "均值预测（弱信号" in md
    assert "barbell 组合生成" in md
    assert "组合风险控制" in md
    assert "硬止损 K" in md
    assert isinstance(proposals, pd.DataFrame)
    assert {"label", "leg", "code", "target_weight"} <= set(proposals.columns)
    assert not proposals.empty


def test_report_return_max_excludes_and_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pack_dir, cluster_file = _make_synthetic_pack(tmp_path, monkeypatch)
    md, proposals, _ = build_fat_tail_report(
        pack_dir=pack_dir,
        cluster_mapping_path=cluster_file,
        cfg=FatTailConfig(
            min_samples=120,
            lookback_days=0,
            proposal_objective="return_max",
            barbell_exclude_codes=("SHBBB",),
            max_name_weight=0.8,
        ),
    )
    assert "满仓进攻" in md
    assert "已剔除" in md and "SHBBB" in md
    assert "组合风险控制" in md
    assert not proposals.empty
    assert "SHBBB" not in set(proposals["code"].astype(str))
    assert "SHAAA" in set(proposals["code"].astype(str))


def test_lookback_output_suffix_and_paths():
    assert lookback_output_suffix(756) == "_756"
    assert lookback_output_suffix(252) == "_252"
    path = suffixed_output_path(Path("/tmp/FAT_TAIL_RISK_REPORT.md"), "_252")
    assert path.name == "FAT_TAIL_RISK_REPORT_252.md"
    csv_path = suffixed_output_path(
        Path("/tmp/cluster_mapping_selected_pool_risk.csv"),
        "_756",
    )
    assert csv_path.name == "cluster_mapping_selected_pool_risk_756.csv"


def test_build_cluster_pool_risk_csv_matches_report_columns():
    from etf_daily.lib.fat_tail_portfolio import build_exit_tier_table

    frame = pd.DataFrame(
        [
            {
                "code": "SHAAA",
                "name": "A",
                "weight": 0.0,
                "n_samples": 200,
                "mad_ann": 0.1351,
                "alpha_left": 1.26,
                "alpha_right": 3.2,
                "kappa": 0.324,
                "shadow_mean": -0.001315,
                "shrink_factor": 0.06,
                "shrunk_mean": -2.19e-5,
                "var_left": -0.0434,
                "cvar_left": -0.1857,
                "barbell_leg": "avoid",
                "flags": "",
                "tail_regime": "stable",
                "path_state": "near_high",
                "dd_near": -0.014,
                "dd_long": -0.072,
                "xi_left": -0.1,
                "beta_left": 0.01,
                "threshold_left": 0.02,
                "nu_left": 0.075,
            }
        ]
    )
    cfg = FatTailConfig()
    tiers = build_exit_tier_table(frame, cfg)
    out = build_cluster_pool_risk_csv(frame, tiers, cfg)
    assert len(out) == 1
    assert out.loc[0, "code"] == "SHAAA"
    assert out.loc[0, "wt"] == "0.0000"
    assert out.loc[0, "MAD_ann"] == "13.51%"
    assert out.loc[0, "α_L"] == "1.26"
    assert out.loc[0, "尾结构"] == "稳定"
    assert out.loc[0, "状态"] == "近高位"
    assert "档1≥%(VaR95)" in out.columns
    assert out.loc[0, "依据"] in {"左尾VaR", "尾部不可估·固定档"}
