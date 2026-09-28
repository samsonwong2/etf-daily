"""Intraday flags stay on the same etf-daily commands. No stamp is passed."""
from __future__ import annotations

import subprocess

from etf_daily.cli import main


def _capture(monkeypatch):
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(cmd, cwd, env, check):
        del cwd, check
        calls.append((list(cmd), dict(env)))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("etf_daily.cli.subprocess.run", fake_run)
    return calls


def test_regime_intraday_skips_html_and_sets_no_stamp(monkeypatch):
    calls = _capture(monkeypatch)
    assert main(["regime", "--skip-html", "--intraday", "--as-of", "2026-09-28"]) == 0
    cmd, env = calls[-1]
    assert cmd[0].endswith("daily_regime_transition_validation.sh")
    assert env["INTRADAY"] == "1"
    assert env["SKIP_HTML"] == "1"
    assert env["SKIP_REBUILD"] == "0"
    assert env["AS_OF"] == "2026-09-28"
    assert "OUT_STAMP" not in env


def test_adaptive_intraday_uses_intraday_shell(monkeypatch):
    calls = _capture(monkeypatch)
    assert main(["adaptive", "--intraday", "--as-of", "2026-09-28"]) == 0
    cmd, env = calls[-1]
    assert cmd[0].endswith("daily_adaptive_stage_html_intraday.sh")
    assert env["AS_OF"] == "2026-09-28"
    assert "OUT_STAMP" not in env


def test_listing_intraday_reuses_latest_pack(monkeypatch):
    calls = _capture(monkeypatch)
    assert main(["listing", "--intraday", "--as-of", "2026-09-28"]) == 0
    cmd, env = calls[-1]
    assert cmd[0].endswith("daily_adaptive_from_listing.sh")
    assert env["INTRADAY"] == "1"
    assert env["AS_OF"] == "2026-09-28"
    assert "VALIDATION_DIR" not in env
    assert "LIVE_SNAPSHOT" not in env
    assert "OUT_STAMP" not in env
