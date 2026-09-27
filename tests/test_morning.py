"""Hermetic tests for the morning receipt. No qlib and no network."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from etf_daily.morning import main, numeric_one, ok_is_true

AS_OF = "2026-09-24"
NEXT_DAY = "2026-09-25"
EQUITY = "SH510300"
OTHER = "SH510500"
BOND = "SH511010"
ANCHOR = "SH513650"


def _candle(day: str) -> str:
    return (
        "<html><body><script>"
        f'Plotly.newPlot("g", [{{"type":"candlestick","yaxis":"y10",'
        f'"x":["{day}"],"open":[1],"high":[2],"low":[0.5],"close":[1.5]}}]);'
        "</script></body></html>\n"
    )


def _html_name(code: str, name: str = "X") -> str:
    return f"regime_transition_{code}_{name}_20240101_20260924_adaptive.html"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _world(tmp: Path) -> dict[str, Path]:
    listing = tmp / "listing"
    adaptive = tmp / "adaptive"
    validation = tmp / "validation"
    pool = tmp / "pool.txt"
    hrp = tmp / "hrp" / "hrp_dendrogram_20260924_d080.html"
    listing.mkdir()
    adaptive.mkdir()
    validation.mkdir()
    _write(pool, f"{EQUITY}\t2020-01-01\t2026-09-24\n{BOND}\t2020-01-01\t2026-09-24\n")
    _write(listing / _html_name(EQUITY, "沪深300"), _candle(AS_OF))
    _write(listing / _html_name(BOND, "国债"), _candle(AS_OF))
    _write(
        adaptive / "batch_summary.csv",
        f"code,name\n{EQUITY},沪深300\n{BOND},国债\n",
    )
    _write(
        adaptive / "ticket_card.csv",
        "as_of,code,name,regime,pass_up_and_q05\n"
        f"{AS_OF},{EQUITY},沪深300,up,1\n"
        f"{AS_OF},{ANCHOR},标普500ETF南方,up,0\n",
    )
    _write(adaptive / "ticket_card_meta.json", json.dumps({"as_of": AS_OF}) + "\n")
    _write(
        listing / "next_day_triggers_20260925.csv",
        "code,name,ok,error,px,buy_px,buy_ret,sell_px,sell_ret\n"
        f"{EQUITY},沪深300,True,,1.5,1.4,-0.0667,,\n",
    )
    _write(validation / "go_nogo.json", json.dumps({"pass": True}) + "\n")
    _write(validation / "signals_with_reversal_labels.csv", f"as_of\n{AS_OF}\n")
    _write(hrp, "<html>d080</html>\n")
    return {
        "listing": listing,
        "adaptive": adaptive,
        "validation": validation,
        "pool": pool,
        "hrp": hrp,
    }


def _argv(world: dict[str, Path], **over: str) -> list[str]:
    listing = over.get("listing", str(world["listing"]))
    return [
        "--as-of",
        over.get("as_of", AS_OF),
        "--next-day",
        over.get("next_day", NEXT_DAY),
        "--listing-dir",
        listing,
        "--adaptive-dir",
        over.get("adaptive", str(world["adaptive"])),
        "--pool-txt",
        over.get("pool", str(world["pool"])),
        "--validation-dir",
        over.get("validation", str(world["validation"])),
        "--hrp-file",
        over.get("hrp", str(world["hrp"])),
    ]


def _run(world: dict[str, Path], **over: str) -> tuple[int, dict, str]:
    listing = Path(over.get("listing", world["listing"]))
    rc = main(_argv(world, **over))
    receipt = json.loads((listing / "morning_receipt.json").read_text(encoding="utf-8"))
    page = (listing / "morning.html").read_text(encoding="utf-8")
    return rc, receipt, page


def test_numeric_one_rejects_zero_strings():
    assert numeric_one(1) is True
    assert numeric_one(1.0) is True
    assert numeric_one("1") is True
    assert numeric_one("0") is False
    assert numeric_one("0.0") is False
    assert numeric_one(0) is False
    assert numeric_one(True) is False
    assert numeric_one(None) is False
    assert ok_is_true(True) is True
    assert ok_is_true(1) is True
    assert ok_is_true("true") is True
    assert ok_is_true("0") is False
    assert ok_is_true(False) is False
    assert ok_is_true("false") is False


def test_clean_day_is_today_and_bond_is_not_required(tmp_path: Path):
    rc, receipt, page = _run(_world(tmp_path))
    assert rc == 0
    assert receipt["label"] == "TODAY"
    assert "文件齐，最后一根是 2026-09-24" in page
    assert "<table>" in page
    assert EQUITY in page
    assert BOND not in receipt.get("missing_triggers", [])
    assert "missing_triggers" not in receipt
    assert ".." in page  # d080 lives outside the listing dir


def test_missing_pool_stops_without_table(tmp_path: Path):
    world = _world(tmp_path)
    rc, _receipt, page = _run(world, pool=str(tmp_path / "no-such.txt"))
    assert rc == 1
    assert "STOP 这页不是今天" in page
    assert "<table>" not in page


def test_empty_pool_stops(tmp_path: Path):
    world = _world(tmp_path)
    world["pool"].write_text("\n\n", encoding="utf-8")
    rc, receipt, page = _run(world)
    assert rc == 1
    assert receipt["label"] == "STOP"
    assert "<table>" not in page


def test_missing_html_names_the_code(tmp_path: Path):
    world = _world(tmp_path)
    (world["listing"] / _html_name(EQUITY, "沪深300")).unlink()
    rc, receipt, _page = _run(world)
    assert rc == 1
    assert EQUITY in receipt["missing_html"]


def test_two_html_files_names_both(tmp_path: Path):
    world = _world(tmp_path)
    extra = world["listing"] / _html_name(EQUITY, "另一份")
    extra.write_text(_candle(AS_OF), encoding="utf-8")
    rc, receipt, page = _run(world)
    assert rc == 1
    detail = next(item["detail"] for item in receipt["checks"] if item["id"] == "html")
    assert _html_name(EQUITY, "沪深300") in detail
    assert extra.name in detail
    assert "<table>" not in page


def test_stale_last_bar_stops(tmp_path: Path):
    world = _world(tmp_path)
    (world["listing"] / _html_name(EQUITY, "沪深300")).write_text(
        _candle("2026-09-23"), encoding="utf-8"
    )
    rc, receipt, _page = _run(world)
    assert rc == 1
    detail = next(item["detail"] for item in receipt["checks"] if item["id"] == "last_bar")
    assert "2026-09-23" in detail


def test_unreadable_candle_stops(tmp_path: Path):
    world = _world(tmp_path)
    (world["listing"] / _html_name(EQUITY, "沪深300")).write_text("<html>nope</html>", encoding="utf-8")
    rc, receipt, _page = _run(world)
    assert rc == 1
    detail = next(item["detail"] for item in receipt["checks"] if item["id"] == "last_bar")
    assert "读不出 K 线" in detail


def test_meta_as_of_mismatch(tmp_path: Path):
    world = _world(tmp_path)
    (world["adaptive"] / "ticket_card_meta.json").write_text(
        json.dumps({"as_of": "2026-09-01"}), encoding="utf-8"
    )
    rc, receipt, _page = _run(world)
    assert rc == 1
    detail = next(item["detail"] for item in receipt["checks"] if item["id"] == "ticket_card")
    assert "meta as_of" in detail


def test_row_as_of_mismatch(tmp_path: Path):
    world = _world(tmp_path)
    text = (world["adaptive"] / "ticket_card.csv").read_text(encoding="utf-8")
    text = text.replace(f"{AS_OF},{ANCHOR}", f"2026-08-01,{ANCHOR}", 1)
    (world["adaptive"] / "ticket_card.csv").write_text(text, encoding="utf-8")
    rc, receipt, _page = _run(world)
    assert rc == 1
    detail = next(item["detail"] for item in receipt["checks"] if item["id"] == "ticket_card")
    assert "行 as_of" in detail


def test_missing_meta_stops(tmp_path: Path):
    world = _world(tmp_path)
    (world["adaptive"] / "ticket_card_meta.json").unlink()
    rc, receipt, _page = _run(world)
    assert rc == 1
    detail = next(item["detail"] for item in receipt["checks"] if item["id"] == "ticket_card")
    assert "ticket_card_meta.json 缺失" in detail


def test_missing_ticket_row_for_universe_code(tmp_path: Path):
    world = _world(tmp_path)
    (world["adaptive"] / "ticket_card.csv").write_text(
        "as_of,code,name,regime,pass_up_and_q05\n" f"{AS_OF},{EQUITY},沪深300,up,1\n",
        encoding="utf-8",
    )
    rc, receipt, _page = _run(world)
    assert rc == 1
    detail = next(item["detail"] for item in receipt["checks"] if item["id"] == "ticket_card")
    assert ANCHOR in detail


def test_trigger_ok_false_stops_with_error(tmp_path: Path):
    world = _world(tmp_path)
    (world["listing"] / "next_day_triggers_20260925.csv").write_text(
        "code,name,ok,error,px,buy_px,buy_ret,sell_px,sell_ret\n"
        f"{EQUITY},沪深300,False,boom,1.5,,,,\n",
        encoding="utf-8",
    )
    rc, receipt, page = _run(world)
    assert rc == 1
    detail = next(item["detail"] for item in receipt["checks"] if item["id"] == "triggers")
    assert "boom" in detail
    assert "<table>" not in page


def test_missing_trigger_row_stops(tmp_path: Path):
    world = _world(tmp_path)
    (world["listing"] / "next_day_triggers_20260925.csv").write_text(
        "code,name,ok,error,px,buy_px,buy_ret,sell_px,sell_ret\n",
        encoding="utf-8",
    )
    rc, receipt, _page = _run(world)
    assert rc == 1
    assert EQUITY in receipt["missing_triggers"]


def test_missing_trigger_file_stops(tmp_path: Path):
    world = _world(tmp_path)
    (world["listing"] / "next_day_triggers_20260925.csv").unlink()
    rc, _receipt, page = _run(world)
    assert rc == 1
    assert "<table>" not in page


def test_missing_d080_stops(tmp_path: Path):
    world = _world(tmp_path)
    world["hrp"].unlink()
    rc, receipt, _page = _run(world)
    assert rc == 1
    detail = next(item["detail"] for item in receipt["checks"] if item["id"] == "hrp_d080")
    assert "缺失" in detail


def test_go_nogo_false_is_warn_with_table(tmp_path: Path):
    world = _world(tmp_path)
    (world["validation"] / "go_nogo.json").write_text(json.dumps({"pass": False}), encoding="utf-8")
    rc, receipt, page = _run(world)
    assert rc == 0
    assert receipt["label"] == "WARN"
    assert "<table>" in page
    assert "模型校准" in page


def test_stale_reversal_is_warn(tmp_path: Path):
    world = _world(tmp_path)
    (world["validation"] / "signals_with_reversal_labels.csv").write_text(
        "as_of\n2026-07-20\n", encoding="utf-8"
    )
    rc, receipt, page = _run(world)
    assert rc == 0
    assert receipt["label"] == "WARN"
    assert "悬停里的验证不是今天" in page
    assert "<table>" in page


def test_both_warns_join_with_semicolon(tmp_path: Path):
    world = _world(tmp_path)
    (world["validation"] / "go_nogo.json").write_text("{}", encoding="utf-8")
    (world["validation"] / "signals_with_reversal_labels.csv").unlink()
    rc, _receipt, page = _run(world)
    assert rc == 0
    assert "模型校准" in page
    assert "；" in page
    assert "缺失" in page
    assert page.index("模型校准") < page.index("悬停里的验证不是今天")


def test_pass_flag_zero_is_left_off_the_list(tmp_path: Path):
    world = _world(tmp_path)
    (world["adaptive"] / "ticket_card.csv").write_text(
        "as_of,code,name,regime,pass_up_and_q05\n"
        f"{AS_OF},{EQUITY},沪深300,up,0\n"
        f"{AS_OF},{ANCHOR},标普500ETF南方,up,0\n",
        encoding="utf-8",
    )
    rc, receipt, page = _run(world)
    assert rc == 0
    assert receipt["label"] == "TODAY"
    assert "<table>" not in page
    assert "今天没有同时过门槛又有触发价的名字" in page


def test_pass_flag_one_is_on_the_list(tmp_path: Path):
    _rc, _receipt, page = _run(_world(tmp_path))
    assert f"<td>{EQUITY}</td>" in page
    assert "-6.67%" in page


def test_name_is_escaped(tmp_path: Path):
    world = _world(tmp_path)
    (world["listing"] / "next_day_triggers_20260925.csv").write_text(
        "code,name,ok,error,px,buy_px,buy_ret,sell_px,sell_ret\n"
        f'{EQUITY},<script>,True,,1.5,1.4,-0.01,,\n',
        encoding="utf-8",
    )
    _rc, _receipt, page = _run(world)
    assert "&lt;script&gt;" in page
    assert "<script>" not in page


def test_extra_html_outside_the_pool_is_ignored(tmp_path: Path):
    world = _world(tmp_path)
    (world["listing"] / _html_name("SH999999", "局外")).write_text(_candle("2020-01-01"), encoding="utf-8")
    rc, receipt, _page = _run(world)
    assert rc == 0
    assert receipt["label"] == "TODAY"
    assert "SH999999" not in receipt["expected_codes"]


def test_validation_dir_flag_selects_go_nogo(tmp_path: Path):
    world = _world(tmp_path)
    other = tmp_path / "other_validation"
    other.mkdir()
    (other / "go_nogo.json").write_text(json.dumps({"pass": False}), encoding="utf-8")
    (other / "signals_with_reversal_labels.csv").write_text(f"as_of\n{AS_OF}\n", encoding="utf-8")
    rc, receipt, page = _run(world, validation=str(other))
    assert rc == 0
    assert receipt["label"] == "WARN"
    assert "模型校准" in page


def test_validation_dir_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    world = _world(tmp_path)
    other = tmp_path / "env_validation"
    other.mkdir()
    (other / "go_nogo.json").write_text(json.dumps({"pass": False}), encoding="utf-8")
    (other / "signals_with_reversal_labels.csv").write_text(f"as_of\n{AS_OF}\n", encoding="utf-8")
    monkeypatch.setenv("VALIDATION_DIR", str(other))
    argv = _argv(world)
    # Drop the explicit flag so the env var is the one that applies.
    del argv[argv.index("--validation-dir") : argv.index("--validation-dir") + 2]
    rc = main(argv)
    page = (world["listing"] / "morning.html").read_text(encoding="utf-8")
    assert rc == 0
    assert "模型校准" in page


def test_accept_eod_calls_morning():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
    text = (root / "scripts" / "accept_eod.sh").read_text(encoding="utf-8")
    assert "-m etf_daily.morning" in text


def test_real_listing_smoke():
    root = os.environ.get("PLOTLY_ROOT")
    if root:
        base = Path(root)
    else:
        from etf_daily.paths import PLOTLY_OUTPUTS_DIR

        base = PLOTLY_OUTPUTS_DIR
    dirs = sorted(base.glob("*_from_listing")) if base.is_dir() else []
    if not dirs:
        pytest.skip("no listing dir")
    htmls = sorted(dirs[-1].glob("regime_transition_*_adaptive.html"))
    if not htmls:
        pytest.skip("listing has no html")
    from etf_daily.lib.fair_path_band_signals import load_ohlcv_from_regime_html

    close, _ohlcv = load_ohlcv_from_regime_html(htmls[0])
    assert len(close) > 0
