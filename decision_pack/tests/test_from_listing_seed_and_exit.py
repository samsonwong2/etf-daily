"""Regression: seed refreshes stale configs; pool exit reflects OOS failures."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_listing_mod():
    path = Path("decision_pack/scripts/plot_adaptive_from_listing.py")
    spec = importlib.util.spec_from_file_location("plot_adaptive_from_listing_seed", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_pool_mod():
    path = Path("decision_pack/scripts/plot_adaptive_stage_pool.py")
    spec = importlib.util.spec_from_file_location("plot_adaptive_pool_exit", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_seed_out_dir_refreshes_existing_dst(tmp_path: Path):
    mod = _load_listing_mod()
    source = tmp_path / "src"
    out = tmp_path / "out"
    (source / "configs").mkdir(parents=True)
    (out / "configs").mkdir(parents=True)
    code = "SH515880"
    src = source / "configs" / f"{code}.json"
    dst = out / "configs" / f"{code}.json"
    src.write_text(json.dumps({"method": "adx_di", "train_cutoff": "2026-04-01"}), encoding="utf-8")
    dst.write_text(json.dumps({"method": "stale_old", "train_cutoff": "2020-01-01"}), encoding="utf-8")
    tag = "20260401_hold_up_expect_no_trade_legacy_score_method_dummy"
    (source / f".train_cache_{tag}.json").write_text('{"ok":true}\n', encoding="utf-8")
    (out / f".train_cache_{tag}.json").write_text('{"ok":false}\n', encoding="utf-8")

    mod.seed_out_dir(
        out,
        source,
        [code],
        tag=tag,
        train_cutoff="2026-04-01",
        retrain=False,
    )
    refreshed = json.loads(dst.read_text(encoding="utf-8"))
    assert refreshed["method"] == "adx_di"
    assert refreshed["train_cutoff"] == "2026-04-01"
    cache = json.loads((out / f".train_cache_{tag}.json").read_text(encoding="utf-8"))
    assert cache["ok"] is True


def test_pool_exit_code_doc_contract():
    """Guard: pool main must end with non-zero when OOS/train errors exist.

    Full main() needs qlib; we assert the helper contract via source scan of the
    return expression that was introduced for from_listing wrappers.
    """
    src = Path("decision_pack/scripts/plot_adaptive_stage_pool.py").read_text(encoding="utf-8")
    assert "return 1 if (n_oos_fail or n_train_fail) else 0" in src
    listing = Path("decision_pack/scripts/plot_adaptive_from_listing.py").read_text(
        encoding="utf-8"
    )
    assert "Only record success after pool + HTML exist" in listing
    assert "no HTML for" in listing
