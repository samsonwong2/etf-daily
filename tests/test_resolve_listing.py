"""listing-dir resolution when PLOTLY_ROOT is unset in the shell."""
from __future__ import annotations

from pathlib import Path

import pytest

from etf_daily import cli


def test_resolve_listing_uses_root_when_shell_path_is_root_relative(tmp_path: Path, monkeypatch):
    day = tmp_path / "20260930_from_listing"
    day.mkdir()
    monkeypatch.setattr(cli, "_config_value", lambda key, default="": str(tmp_path) if key == "PLOTLY_ROOT" else default)
    found = cli._resolve_listing("/20260930_from_listing")
    assert found == day.resolve()


def test_resolve_listing_missing_names_latest(tmp_path: Path, monkeypatch):
    (tmp_path / "20260929_from_listing").mkdir()
    monkeypatch.setattr(cli, "_config_value", lambda key, default="": str(tmp_path) if key == "PLOTLY_ROOT" else default)
    with pytest.raises(SystemExit) as exc:
        cli._resolve_listing("/20260930_from_listing")
    message = str(exc.value)
    assert "20260930_from_listing" in message
    assert "20260929_from_listing" in message
