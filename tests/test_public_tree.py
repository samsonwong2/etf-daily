"""Machine checks for the public tree. These do not call qlib or the eight commands."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEEDLE_A = "/home/" + "huangtuo"
NEEDLE_B = "/home/" + "pu"


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_git_grep_prefixes_empty() -> None:
    proc = _git("grep", "-n", "-e", NEEDLE_A, "-e", NEEDLE_B)
    assert proc.returncode == 1, proc.stdout


def test_decision_packs_ignored_and_untracked() -> None:
    ignored = _git("check-ignore", "-q", "workspace/decision_packs/20260720")
    assert ignored.returncode == 0
    listed = _git("ls-files", "--", "workspace/decision_packs")
    assert listed.returncode == 0
    assert listed.stdout.strip() == ""


def test_shells_source_daily_env() -> None:
    for name in (
        "daily_regime_transition_validation.sh",
        "daily_adaptive_stage_html.sh",
        "daily_adaptive_from_listing.sh",
        "daily_adaptive_stage_html_intraday.sh",
    ):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "daily_env.sh" in text
        proc = subprocess.run(["bash", "-n", str(ROOT / "scripts" / name)], check=False)
        assert proc.returncode == 0


def test_optional_paths_and_public_defaults() -> None:
    sys.path.insert(0, str(ROOT))
    import runtime_paths

    assert runtime_paths._optional_path("") is None
    assert runtime_paths._optional_path(None) is None
    defaults = runtime_paths._default_paths()
    assert defaults["qlib_scripts_dir"] == ""
    assert defaults["local_library_dir"] == ""
    assert defaults["provider_uri"].endswith(".qlib/qlib_data/all_fund_data")
    assert defaults["temp_dir"].endswith("etf-daily-output")
    assert "etf-daily-output/plotly_outputs" in defaults["plotly_outputs_dir"]


def test_missing_config_file_uses_none(tmp_path: Path) -> None:
    del tmp_path
    cfg = ROOT / "configs" / "production_regime_switch_ewma_shrink.json"
    hidden = cfg.with_suffix(".json.hidden-by-test")
    moved = False
    if cfg.exists():
        cfg.rename(hidden)
        moved = True
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import runtime_paths; "
                "assert runtime_paths.QLIB_SCRIPTS_DIR is None; "
                "assert runtime_paths.LOCAL_LIBRARY_DIR is None",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
    finally:
        if moved:
            hidden.rename(cfg)
