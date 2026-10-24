"""Regression: seed refreshes stale configs; pool exit reflects OOS failures."""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest


def _load_listing_mod():
    path = Path("src/etf_daily/plots/plot_adaptive_from_listing.py")
    spec = importlib.util.spec_from_file_location("plot_adaptive_from_listing_seed", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_pool_mod():
    path = Path("src/etf_daily/plots/plot_adaptive_stage_pool.py")
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
    src = Path("src/etf_daily/plots/plot_adaptive_stage_pool.py").read_text(encoding="utf-8")
    assert "return 1 if (n_oos_fail or n_train_fail) else 0" in src
    listing = Path("src/etf_daily/plots/plot_adaptive_from_listing.py").read_text(
        encoding="utf-8"
    )
    assert "Only record success after pool + HTML exist" in listing
    assert "no HTML for" in listing


def test_seed_allow_missing_copies_present_only(tmp_path: Path):
    mod = _load_listing_mod()
    source = tmp_path / "cache_tag"
    out = tmp_path / "out"
    (source / "configs").mkdir(parents=True)
    (source / "configs" / "SH515880.json").write_text(
        json.dumps({"method": "adx_di"}), encoding="utf-8"
    )
    tag = "20260401_hold_up_expect_no_trade_legacy_score_method_dummy"
    mod.seed_out_dir(
        out,
        source,
        ["SH515880", "SH999999"],
        tag=tag,
        train_cutoff="2026-04-01",
        retrain=False,
        allow_missing=True,
    )
    assert json.loads((out / "configs" / "SH515880.json").read_text(encoding="utf-8"))[
        "method"
    ] == "adx_di"
    assert not (out / "configs" / "SH999999.json").exists()
    assert (out / f".train_cache_{tag}.json").is_file()


def test_seed_missing_without_allow_raises(tmp_path: Path):
    mod = _load_listing_mod()
    source = tmp_path / "src"
    out = tmp_path / "out"
    (source / "configs").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="SH999999"):
        mod.seed_out_dir(
            out,
            source,
            ["SH999999"],
            tag="tag",
            train_cutoff="2026-04-01",
            retrain=False,
        )


def test_resolve_config_source_uses_train_cache(tmp_path: Path):
    mod = _load_listing_mod()
    cache = tmp_path / "_train_cache"
    tag = "20260401_hold_up_x"
    (cache / tag / "configs").mkdir(parents=True)
    (tmp_path / "20261023all_adaptive" / "configs").mkdir(parents=True)
    source, allow_missing = mod.resolve_config_source(
        None, train_cache_dir=cache, cache_tag_value=tag
    )
    assert source == cache / tag
    assert allow_missing is True
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    source, allow_missing = mod.resolve_config_source(
        explicit, train_cache_dir=cache, cache_tag_value=tag
    )
    assert source == explicit
    assert allow_missing is False


def test_resolve_explicit_missing_raises(tmp_path: Path):
    mod = _load_listing_mod()
    with pytest.raises(FileNotFoundError, match="config source missing"):
        mod.resolve_config_source(
            tmp_path / "nope", train_cache_dir=tmp_path, cache_tag_value="t"
        )


def test_resolve_without_cache_or_explicit_raises():
    mod = _load_listing_mod()
    with pytest.raises(FileNotFoundError, match="shared train cache is disabled"):
        mod.resolve_config_source(None, train_cache_dir=None, cache_tag_value="t")


def test_hardcoded_all_adaptive_candidates_removed():
    src = Path("src/etf_daily/plots/plot_adaptive_from_listing.py").read_text(
        encoding="utf-8"
    )
    assert "20260813all_adaptive" not in src
    assert "20260812all_adaptive" not in src
    assert 'f"{tag}all_adaptive"' not in src
    shell = Path("scripts/daily_adaptive_from_listing.sh").read_text(encoding="utf-8")
    assert "${AS_OF_TAG}all_adaptive" not in shell


def _train_kwargs(pool, **over):
    kwargs = dict(
        missing=["SH999999"],
        existing_configs={"SH515880": {"method": "adx_di"}},
        pm=object(),
        end="2026-10-23",
        train_cutoff="2026-04-01",
        ev=object(),
        score_mode={},
        score_mode_name="expect_no_trade",
        regime_mode="legacy_score_method",
        trade_mode=pool.TRADE_MODE_HOLD_UP,
        k=1.0,
        fail_fast=False,
    )
    kwargs.update(over)
    return kwargs


def test_partial_hold_up_trains_only_missing(tmp_path: Path, monkeypatch):
    pool = _load_pool_mod()
    shared_cfg = tmp_path / "shared" / "configs"
    shared_cfg.mkdir(parents=True)
    kept = {"code": "SH515880", "method": "adx_di", "rules": {"up": 1}}
    kept_path = shared_cfg / "SH515880.json"
    kept_path.write_text(json.dumps(kept), encoding="utf-8")
    before = kept_path.read_bytes()
    calls: list[str] = []

    def fake_load(_pm, code, _end, *, train_cutoff):
        calls.append(code)
        assert train_cutoff == "2026-04-01"
        return object()

    def fake_train(**kwargs):
        return dict(
            code=kwargs["code"],
            method="ema_cross",
            method_params={"n": 3},
            method_scores={},
            rule_rows=[],
            skipped_train=False,
            train_end="2026-03-31",
            train_cutoff="2026-04-01",
            train_bars=100,
            regime_mode="legacy_score_method",
            trade_mode=pool.TRADE_MODE_HOLD_UP,
        )

    monkeypatch.setattr(pool, "_load_ohlcv", fake_load)
    monkeypatch.setattr(pool, "train_one_symbol", fake_train)
    present, missing = pool.missing_config_codes(["SH515880", "SH999999"], shared_cfg)
    assert present == ["SH515880"]
    assert missing == ["SH999999"]
    existing = {"SH515880": json.loads(kept_path.read_text(encoding="utf-8"))}
    new_cfgs, errors = pool.train_missing_hold_up_configs(
        **_train_kwargs(pool, missing=missing, existing_configs=existing)
    )
    assert calls == ["SH999999"]
    assert errors == []
    assert set(new_cfgs) == {"SH999999"}
    assert new_cfgs["SH999999"]["method"] == "ema_cross"
    assert kept_path.read_bytes() == before
    assert existing["SH515880"]["method"] == "adx_di"

    local_cfg = tmp_path / "local" / "configs"
    local_cfg.mkdir(parents=True)
    shutil.copy2(kept_path, local_cfg / "SH515880.json")
    (local_cfg / "SH999999.json").write_text(
        json.dumps(new_cfgs["SH999999"]), encoding="utf-8"
    )
    prior = tmp_path / "local" / "prior.csv"
    prior.write_text("stage\n", encoding="utf-8")
    shared_meta = tmp_path / "shared" / "meta.json"
    shared_meta.write_text("{}\n", encoding="utf-8")
    pool._publish_train_cache(
        cache_meta=tmp_path / "local" / ".train_cache.json",
        cfg_dir=local_cfg,
        prior_path=prior,
        codes=["SH515880", "SH999999"],
        train_cutoff="2026-04-01",
        prior_k=1.0,
        score_mode_name="expect_no_trade",
        regime_mode="legacy_score_method",
        trade_mode=pool.TRADE_MODE_HOLD_UP,
        shared_cfg_dir=shared_cfg,
        shared_meta=shared_meta,
        shared_prior=tmp_path / "shared" / "pool_prior.csv",
        shared_root=tmp_path / "shared",
    )
    assert kept_path.read_bytes() == before
    assert json.loads((shared_cfg / "SH999999.json").read_text(encoding="utf-8"))[
        "method"
    ] == "ema_cross"
    assert json.loads(shared_meta.read_text(encoding="utf-8"))["n_codes"] == 2


def test_failed_missing_code_uses_existing_modal(monkeypatch):
    pool = _load_pool_mod()

    def fake_load(_pm, _code, _end, *, train_cutoff):
        raise RuntimeError("no bars")

    monkeypatch.setattr(pool, "_load_ohlcv", fake_load)
    existing = {
        "AAA": {"method": "adx_di"},
        "BBB": {"method": "adx_di"},
        "CCC": {"method": "other"},
    }
    new_cfgs, errors = pool.train_missing_hold_up_configs(
        **_train_kwargs(pool, existing_configs=existing)
    )
    assert errors[0]["code"] == "SH999999"
    assert new_cfgs["SH999999"]["method"] == "adx_di"
    assert set(new_cfgs) == {"SH999999"}
