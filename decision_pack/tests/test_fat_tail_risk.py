from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from decision_pack.scripts.generate_fat_tail_risk_report import build_fat_tail_risk_markdown
from decision_pack.src.fat_tail_risk import (
    FatTailConfig,
    barbell_leg,
    build_fat_tail_frame,
    compute_fat_tail_metrics,
    compute_kappa,
    compute_mad,
    effective_min_exceed,
    fit_tail,
    portfolio_weighted_summary,
    shrink_factor,
    slice_log_returns,
)


def _make_close_from_returns(returns: np.ndarray, start: str = "2020-01-01") -> pd.Series:
    prices = 100.0 * np.exp(np.cumsum(np.insert(returns, 0, 0.0)))
    index = pd.bdate_range(start, periods=len(prices))
    return pd.Series(prices, index=index)


def test_mad_positive_and_finite():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.01, size=500)
    mad, mad_ann = compute_mad(returns)
    assert mad is not None and mad > 0
    assert mad_ann is not None and mad_ann > mad


def test_kappa_in_unit_interval_and_fatter_is_higher():
    rng = np.random.default_rng(1)
    thin = rng.normal(0.0, 0.008, size=2000)
    # df=2.2 -> tail index ~2.2 (near-infinite variance), clearly fat.
    fat = stats.t.rvs(df=2.2, size=2000, random_state=rng) * 0.008
    kappa_thin = compute_kappa(thin, n=30)
    kappa_fat = compute_kappa(fat, n=30)
    assert kappa_thin is not None and 0.0 <= kappa_thin <= 1.0
    assert kappa_fat is not None and 0.0 <= kappa_fat <= 1.0
    assert kappa_fat > kappa_thin


def test_kappa_is_deterministic():
    rng = np.random.default_rng(7)
    returns = stats.t.rvs(df=3, size=1500, random_state=rng) * 0.01
    assert compute_kappa(returns, n=30) == compute_kappa(returns, n=30)


def test_fit_tail_recovers_pareto_alpha_roughly():
    rng = np.random.default_rng(2)
    true_alpha = 2.5
    xi = 1.0 / true_alpha
    scale = 0.01
    sample = stats.genpareto.rvs(xi, scale=scale, size=2000, random_state=rng)
    fit = fit_tail(sample, side="right", tail_quantile=0.90, min_exceed=25)
    assert fit.reliable
    assert fit.alpha is not None
    assert abs(fit.alpha - true_alpha) < 0.8


def test_short_sample_flags_low_n():
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0, 0.01, size=80)
    close = _make_close_from_returns(returns)
    metrics = compute_fat_tail_metrics(
        "TEST",
        close=close,
        end=close.index[-1],
        cfg=FatTailConfig(min_samples=120, lookback_days=0),
    )
    assert "low_n" in metrics.flags


def test_shadow_mean_adjusts_for_fat_left_tail():
    rng = np.random.default_rng(4)
    body = rng.normal(0.0005, 0.006, size=2400)
    crashes = -rng.pareto(2.2, size=80) * 0.02
    returns = np.concatenate([body, crashes])
    rng.shuffle(returns)
    close = _make_close_from_returns(returns)
    metrics = compute_fat_tail_metrics(
        "TEST",
        close=close,
        end=close.index[-1],
        cfg=FatTailConfig(min_samples=120, lookback_days=0, min_exceed=15),
    )
    assert metrics.raw_mean is not None
    assert metrics.shadow_mean is not None
    assert metrics.shadow_mean != metrics.raw_mean
    assert metrics.cvar_left is not None
    assert metrics.cvar_left < metrics.var_left


def test_shrink_factor_decreases_with_fatter_tail():
    cfg = FatTailConfig()
    s_thin = shrink_factor(alpha_left=5.0, kappa=0.2, n_samples=500, cfg=cfg)
    s_fat = shrink_factor(alpha_left=1.8, kappa=0.7, n_samples=500, cfg=cfg)
    assert s_thin > s_fat


def test_barbell_defensive_anchors_safe_leg():
    cfg = FatTailConfig()
    # Defensive asset that is not fat-tailed -> safe leg.
    leg = barbell_leg(
        alpha_left=None,
        kappa=None,
        cvar_left=-0.005,
        flags=("alpha_unreliable",),
        cfg=cfg,
        is_defensive=True,
    )
    assert leg == "safe_leg"


def test_barbell_bounded_left_tail_is_thin():
    cfg = FatTailConfig()
    leg = barbell_leg(
        alpha_left=None,
        kappa=0.0,
        cvar_left=-0.015,
        flags=("bounded_tail_left",),
        cfg=cfg,
    )
    assert leg == "safe_leg"


def test_barbell_bounded_right_tail_does_not_make_left_thin():
    # A bounded RIGHT tail must not route a moderately fat LEFT tail to safe_leg.
    cfg = FatTailConfig()
    leg = barbell_leg(
        alpha_left=3.0,  # in (alpha_fat, alpha_thin): not avoid, not thin
        kappa=0.0,
        cvar_left=-0.015,
        flags=("bounded_tail_right",),
        cfg=cfg,
    )
    assert leg == "risk_leg"


def test_barbell_fat_left_tail_avoids():
    cfg = FatTailConfig()
    leg = barbell_leg(
        alpha_left=1.8,
        kappa=0.1,
        cvar_left=-0.02,
        flags=(),
        cfg=cfg,
    )
    assert leg == "avoid"


def test_hill_alpha_fallback_when_gpd_bounded():
    rng = np.random.default_rng(0)
    returns = rng.uniform(-0.008, 0.012, size=252)
    fit = fit_tail(
        returns,
        side="left",
        tail_quantile=0.925,
        min_exceed=effective_min_exceed(252, tail_quantile=0.925, min_exceed=25),
    )
    assert fit.reliable
    assert "bounded_tail_left" in fit.flags
    assert fit.alpha is not None and fit.alpha > 0


def test_effective_min_exceed_scales_with_sample_size():
    cfg = FatTailConfig(tail_quantile=0.925, min_exceed=25)
    assert effective_min_exceed(756, tail_quantile=cfg.tail_quantile, min_exceed=cfg.min_exceed) == 25
    assert effective_min_exceed(252, tail_quantile=cfg.tail_quantile, min_exceed=cfg.min_exceed) == 19


def test_252_lookback_still_estimates_left_tail():
    rng = np.random.default_rng(12)
    body = rng.normal(0.0003, 0.008, size=230)
    crashes = -rng.pareto(2.5, size=22) * 0.015
    returns = np.concatenate([body, crashes])
    rng.shuffle(returns)
    close = _make_close_from_returns(returns)
    metrics = compute_fat_tail_metrics(
        "TEST",
        close=close,
        end=close.index[-1],
        cfg=FatTailConfig(min_samples=120, lookback_days=252, min_exceed=25),
    )
    assert "alpha_unreliable" not in metrics.flags
    assert metrics.alpha_left is not None
    assert metrics.var_left is not None
    assert metrics.cvar_left is not None


def test_left_tail_uses_full_loss_distribution():
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0, 0.01, size=1000)
    fit = fit_tail(returns, side="left", tail_quantile=0.925, min_exceed=25)
    # ~7.5% of all observations should exceed the threshold, not ~3.75%.
    assert fit.n_exceed >= 60


def test_portfolio_summary_xi_space_and_coverage():
    frame = pd.DataFrame(
        {
            "code": ["A", "B", "C"],
            "name": ["A", "B", "C"],
            "weight": [0.5, 0.4, 0.1],
            "kappa": [0.1, 0.2, 0.3],
            # C has no estimable left tail (short history / unreliable fit).
            "alpha_left": [2.0, 50.0, None],
            "cvar_left": [-0.08, -0.03, None],
            "barbell_leg": ["avoid", "risk_leg", "risk_leg"],
        }
    )
    summary = portfolio_weighted_summary(frame)

    # Coverage excludes the 10% with no alpha/CVaR; kappa is fully covered.
    assert summary["coverage_alpha_left"] == pytest.approx(0.9)
    assert summary["coverage_cvar_left"] == pytest.approx(0.9)
    assert summary["coverage_kappa"] == pytest.approx(1.0)

    # Fattest left tail = smallest alpha among held names.
    assert summary["fattest_code"] == "A"
    assert summary["fattest_alpha_left"] == pytest.approx(2.0)

    # xi-space aggregation must NOT be dominated by the degenerate alpha=50
    # (a naive linear weighted mean would be ~23).
    assert summary["weighted_alpha_left"] is not None
    assert summary["weighted_alpha_left"] < 10.0


def test_build_fat_tail_frame_smoke():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2022-01-01", periods=300)
    panel = pd.DataFrame(
        {
            "AAA": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=len(dates)))),
            "BBB": 50 * np.exp(np.cumsum(rng.normal(0, 0.012, size=len(dates)))),
        },
        index=dates,
    )
    frame = build_fat_tail_frame(
        ["AAA", "BBB"],
        panel,
        str(dates[-1].date()),
        weights={"AAA": 0.6, "BBB": 0.4},
        names={"AAA": "Alpha", "BBB": "Beta"},
        cfg=FatTailConfig(min_samples=120, lookback_days=0),
    )
    assert len(frame) == 2
    assert set(frame.columns) >= {"alpha_left", "kappa", "mad_ann", "barbell_leg"}


def test_render_smoke_with_synthetic_pack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "tier_decision.json").write_text(
        json.dumps({"as_of": "2024-06-15"}, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "code": "SHAAA",
                "name": "Test A",
                "market_value": 1000.0,
                "weight": 1.0,
                "weight_total": 1.0,
                "source": "live",
            }
        ]
    ).to_csv(pack_dir / "portfolio_snapshot.csv", index=False)

    dates = pd.bdate_range("2022-01-01", periods=320)
    rng = np.random.default_rng(6)
    panel = pd.DataFrame(
        {"SHAAA": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=len(dates))))},
        index=dates,
    )
    cluster_file = tmp_path / "cluster.txt"
    cluster_file.write_text("SHAAA\t2022-01-01\t2024-06-15\n", encoding="utf-8")

    def fake_load_qlib_close(uri, codes, start, end):
        return panel

    monkeypatch.setattr(
        "decision_pack.scripts.generate_fat_tail_risk_report.load_qlib_close",
        fake_load_qlib_close,
    )
    monkeypatch.setattr(
        "decision_pack.scripts.generate_fat_tail_risk_report.auto_qfq_adjust_close_panel",
        lambda p: (p, {}),
    )
    monkeypatch.setattr(
        "decision_pack.scripts.generate_fat_tail_risk_report.load_fund_name_map",
        lambda *_args, **_kwargs: {},
    )

    md = build_fat_tail_risk_markdown(
        pack_dir=pack_dir,
        cluster_mapping_path=cluster_file,
        cfg=FatTailConfig(min_samples=120, lookback_days=0),
    )
    assert "# Fat-Tail Risk Report" in md
    assert "方法学声明" in md
    assert "标准差" in md
    assert "SHAAA" in md
    assert "vol_ann" not in md
    assert "dd_20d" not in md


def test_render_live_pack_if_available():
    pack_dir = Path("~/etf-daily-output/temp/decision_packs/20260615")
    if not (pack_dir / "portfolio_snapshot.csv").exists():
        pytest.skip("live pack dir unavailable")
    md = build_fat_tail_risk_markdown(pack_dir=pack_dir)
    assert "Fat-Tail Risk Report" in md
    assert "持仓肥尾风险板" in md
