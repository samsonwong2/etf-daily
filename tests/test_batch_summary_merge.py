"""Regression: from_listing per-code pool calls must accumulate batch_summary."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_pool_mod():
    path = Path("src/etf_daily/plots/plot_adaptive_stage_pool.py")
    spec = importlib.util.spec_from_file_location("plot_adaptive_pool_batch_summary", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_merge_batch_summary_df_keeps_latest_per_code():
    mod = _load_pool_mod()
    prior = pd.DataFrame(
        [
            {"code": "SH510300", "edge": 0.1},
            {"code": "SZ159985", "edge": -0.2},
        ]
    )
    new = pd.DataFrame([{"code": "SZ159985", "edge": -0.5}])
    out = mod.merge_batch_summary_df(prior, new)
    assert list(out["code"]) == ["SH510300", "SZ159985"]
    assert float(out.loc[out["code"] == "SZ159985", "edge"].iloc[0]) == -0.5


def test_write_merged_batch_summary_accumulates(tmp_path: Path):
    mod = _load_pool_mod()
    r1 = pd.DataFrame([{"code": "SH510300", "edge": 0.11, "win": 0.5}])
    r2 = pd.DataFrame([{"code": "SZ159985", "edge": -0.6, "win": 0.56}])
    m1 = mod.write_merged_batch_summary(tmp_path, r1)
    assert len(m1) == 1
    m2 = mod.write_merged_batch_summary(tmp_path, r2)
    assert len(m2) == 2
    disk = pd.read_csv(tmp_path / "batch_summary.csv")
    assert set(disk["code"]) == {"SH510300", "SZ159985"}
    # re-run same code updates in place
    r2b = pd.DataFrame([{"code": "SZ159985", "edge": -0.1, "win": 0.6}])
    m3 = mod.write_merged_batch_summary(tmp_path, r2b)
    assert len(m3) == 2
    disk2 = pd.read_csv(tmp_path / "batch_summary.csv")
    assert float(disk2.loc[disk2["code"] == "SZ159985", "edge"].iloc[0]) == -0.1


def test_write_merged_batch_errors_accumulates_and_clears(tmp_path: Path):
    mod = _load_pool_mod()
    e1 = [{"code": "SH510300", "phase": "oos", "error": "boom"}]
    mod.write_merged_batch_errors(tmp_path, e1, clear_codes=[])
    e2 = [{"code": "SZ159985", "phase": "oos", "error": "nope"}]
    mod.write_merged_batch_errors(tmp_path, e2, clear_codes=[])
    disk = pd.read_csv(tmp_path / "batch_errors.csv")
    assert set(disk["code"]) == {"SH510300", "SZ159985"}
    # Successful re-run of SH510300 clears its ghost error.
    mod.write_merged_batch_errors(tmp_path, [], clear_codes=["SH510300"])
    disk2 = pd.read_csv(tmp_path / "batch_errors.csv")
    assert list(disk2["code"]) == ["SZ159985"]
