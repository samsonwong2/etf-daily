"""Offline tests for skipping from-listing HTML that already exists."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from etf_daily.plots.plot_adaptive_from_listing import (
    listing_codes_to_render,
    listing_page_exists,
)


def _page(out_dir: Path, name: str, body: str = "<html></html>") -> Path:
    path = out_dir / name
    path.write_text(body, encoding="utf-8")
    return path


def test_listing_page_exists(tmp_path: Path):
    _page(tmp_path, "regime_transition_SH510300_NAME_20100101_20260930_adaptive.html")
    assert listing_page_exists(tmp_path, "sh510300", "2026-09-30") is True
    assert listing_page_exists(tmp_path, "SH510300", "2026-09-29") is False
    assert listing_page_exists(tmp_path, "SZ159915", "2026-09-30") is False


def test_listing_page_ignores_empty_and_side_pages(tmp_path: Path):
    _page(tmp_path, "regime_transition_SH510300_NAME_20100101_20260930_adaptive.html", body="")
    _page(tmp_path, "regime_transition_SZ159915_NAME_20100101_20260930_fig9_adaptive.html")
    _page(tmp_path, "regime_transition_SH518880_NAME_20100101_20260930_fig10_adaptive.html")
    _page(tmp_path, "regime_transition_SH511010_NAME_20100101_20260930_oracle_adaptive.html")
    assert listing_page_exists(tmp_path, "SH510300", "2026-09-30") is False
    assert listing_page_exists(tmp_path, "SZ159915", "2026-09-30") is False
    assert listing_page_exists(tmp_path, "SH518880", "2026-09-30") is False
    assert listing_page_exists(tmp_path, "SH511010", "2026-09-30") is False


def test_listing_codes_to_render_keeps_csv_rows(tmp_path: Path):
    _page(tmp_path, "regime_transition_SH510300_NAME_20100101_20260930_adaptive.html")
    csv_path = tmp_path / "listing_dates.csv"
    pd.DataFrame(
        [{"code": "SH510300", "listing": "2010-01-01"}, {"code": "SH511010", "listing": "2013-01-01"}]
    ).to_csv(csv_path, index=False)
    codes = ["SH510300", "SZ159915", "SH511010"]
    render, kept = listing_codes_to_render(codes, tmp_path, "2026-09-30", csv_path, False)
    assert render == ["SZ159915", "SH511010"]
    assert kept == [{"code": "SH510300", "listing": "2010-01-01"}]
    forced, kept_forced = listing_codes_to_render(codes, tmp_path, "2026-09-30", csv_path, True)
    assert forced == codes
    assert kept_forced == []
