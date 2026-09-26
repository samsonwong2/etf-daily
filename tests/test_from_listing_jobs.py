from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


class _FakePm:
    def _init_qlib(self, _uri) -> None:
        return None


def _load_listing_mod():
    path = Path("src/etf_daily/plots/plot_adaptive_from_listing.py")
    name = "plot_adaptive_from_listing"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_parse_args_jobs_default_is_one():
    mod = _load_listing_mod()
    args = mod.parse_args(["--as-of", "2026-09-03"])
    assert args.jobs == 1


def test_run_listing_batch_jobs_parity(monkeypatch: pytest.MonkeyPatch):
    mod = _load_listing_mod()

    def _fake_process(code: str, **kwargs):  # noqa: ANN003
        listing_row = dict(
            code=code,
            listing="2020-01-01",
            plot_start="2020-01-01",
            plot_end=kwargs["end_date"],
            as_of=kwargs["as_of"],
        )
        summary_row = dict(
            code=code,
            method="hybrid_ma_adx",
            edge=0.01,
            edge_vs_fixed=0.0,
        )
        return code, listing_row, summary_row, None

    monkeypatch.setattr(mod, "_process_one_listing_code", _fake_process)

    ctx = dict(
        pool=object(),
        pm=_FakePm(),
        ev=object(),
        rt={},
        out_dir=Path("/tmp"),
        end_date="2026-09-03",
        as_of="2026-09-03",
        train_cutoff="2026-04-01",
        start_override=None,
        listing_lookup={},
        incremental_mode="off",
        prev_listing=None,
        incremental_close_tol=1e-6,
        fair_path_extra_trail_years="off",
    )
    codes = ["SHAAA", "SHBBB"]
    rows1, sum1, err1 = mod._run_listing_batch(
        codes=codes, jobs=1, ctx=ctx, fail_fast=False
    )
    rows2, sum2, err2 = mod._run_listing_batch(
        codes=codes, jobs=2, ctx=ctx, fail_fast=False
    )

    assert not err1 and not err2
    assert {r["code"] for r in rows1} == {r["code"] for r in rows2} == set(codes)
    assert {r["code"] for r in sum1} == {r["code"] for r in sum2} == set(codes)
    for c in codes:
        s1 = next(r for r in sum1 if r["code"] == c)
        s2 = next(r for r in sum2 if r["code"] == c)
        assert s1["edge"] == s2["edge"]
        assert s1["method"] == s2["method"]
