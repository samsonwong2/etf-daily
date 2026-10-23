"""Offline tests for skipping adaptive HTML that already exists."""
from __future__ import annotations

from pathlib import Path

from etf_daily.plots.plot_adaptive_stage_pool import adaptive_page_exists, codes_to_render


def _page(out_dir: Path, code: str, start: str, end: str, body: str = "<html></html>") -> Path:
    path = out_dir / f"regime_transition_{code}_NAME_{start}_{end}_adaptive.html"
    path.write_text(body, encoding="utf-8")
    return path


def test_adaptive_page_exists(tmp_path: Path):
    _page(tmp_path, "SH510300", "20260401", "20260930")
    assert adaptive_page_exists(tmp_path, "sh510300", "2026-04-01", "2026-09-30") is True
    assert adaptive_page_exists(tmp_path, "SH510300", "2026-04-01", "2026-09-29") is False
    assert adaptive_page_exists(tmp_path, "SZ159915", "2026-04-01", "2026-09-30") is False


def test_empty_file_does_not_count(tmp_path: Path):
    _page(tmp_path, "SH510300", "20260401", "20260930", body="")
    assert adaptive_page_exists(tmp_path, "SH510300", "2026-04-01", "2026-09-30") is False


def test_codes_to_render_skips_existing_unless_forced(tmp_path: Path):
    _page(tmp_path, "SH510300", "20260401", "20260930")
    codes = ["SH510300", "SZ159915"]
    assert codes_to_render(codes, tmp_path, "2026-04-01", "2026-09-30", False) == ["SZ159915"]
    assert codes_to_render(codes, tmp_path, "2026-04-01", "2026-09-30", True) == codes
    assert codes_to_render(
        codes, tmp_path, "2026-04-01", "2026-09-30", False, stale_codes={"sh510300"}
    ) == codes
