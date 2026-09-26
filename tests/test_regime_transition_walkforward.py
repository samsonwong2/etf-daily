import numpy as np
import pandas as pd

from etf_daily.lib.regime_transition_walkforward import (
    extend_validation_with_pre_oos_walkforward,
    generate_walkforward_triangles,
)


def _make_close(n: int = 320, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    rets = rng.normal(0.0002, 0.012, size=n)
    px = 100 * np.exp(np.cumsum(rets))
    return pd.Series(px, index=idx)


def test_walkforward_emits_triangle_columns():
    close = _make_close()
    tri = generate_walkforward_triangles(
        close, refit_days=63, min_warmup_days=120
    )
    assert list(tri.columns) == ["as_of", "quantile_switch", "switch_side", "pred_cdf"]
    if not tri.empty:
        assert set(tri["switch_side"]).issubset({"up", "down"})
        assert tri["quantile_switch"].all()
        assert ((tri["pred_cdf"] >= 0.90) | (tri["pred_cdf"] <= 0.10)).all()


def test_walkforward_respects_warmup():
    close = _make_close()
    tri = generate_walkforward_triangles(
        close, refit_days=63, min_warmup_days=200
    )
    if not tri.empty:
        first = pd.to_datetime(tri["as_of"].min())
        assert first >= close.index[199]


def test_walkforward_deterministic():
    close = _make_close()
    a = generate_walkforward_triangles(close, refit_days=63, min_warmup_days=120)
    b = generate_walkforward_triangles(close, refit_days=63, min_warmup_days=120)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_walkforward_empty_on_short_series():
    close = pd.Series([1.0, 1.1, 1.2], index=pd.bdate_range("2024-01-02", periods=3))
    tri = generate_walkforward_triangles(close)
    assert tri.empty


def test_extend_pre_oos_keeps_official_and_fills_earlier_window():
    close = _make_close(n=280, seed=3)
    cut = close.index[200]
    official = pd.DataFrame(
        {
            "as_of": [cut, cut + pd.Timedelta(days=3)],
            "code": ["SZ159985", "SZ159985"],
            "name": ["豆粕ETF", "豆粕ETF"],
            "quantile_switch": [True, False],
            "switch_side": ["down", None],
            "pred_cdf": [0.04, 0.50],
        }
    )
    rev = pd.DataFrame(
        {
            "as_of": [cut],
            "code": ["SZ159985"],
            "turn_kind": ["bottom_reversal"],
            "label_switch_reversal": [True],
        }
    )
    oos2, rev2 = extend_validation_with_pre_oos_walkforward(
        official,
        rev,
        code="SZ159985",
        close=close,
        start_date=str(close.index[0].date()),
        end_date=str(close.index[-1].date()),
        refit_days=40,
        min_warmup_days=80,
    )
    pre = oos2[oos2["as_of"] < cut]
    post = oos2[oos2["as_of"] >= cut]
    assert not pre.empty
    assert (pre["switch_source"] == "walkforward_pre_oos").all()
    assert set(post["as_of"]) >= set(official["as_of"])
    official_down = post[
        (post["as_of"] == cut) & (post["code"] == "SZ159985")
    ].iloc[0]
    assert bool(official_down["quantile_switch"]) is True
    assert str(official_down["switch_side"]) == "down"
    assert (rev2["as_of"] == cut).any()
