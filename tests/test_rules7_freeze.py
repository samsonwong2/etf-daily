"""The frozen rules7 knobs. A trade-changing edit must fail here on purpose."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from etf_daily.lib.rules7_forward import (  # noqa: E402
    FREEZE_DATE,
    FREEZE_TAG,
    FORWARD_START,
    LOGIC_EQUIVALENT_TO,
    PSEUDO_FORWARD_START,
    SPECS_PATH,
    spec_fingerprint,
)

PINNED_FINGERPRINT = "e0d372b555b733248b7e55bfcf751bf3c86f711d9b7f9fe8d91c9ef0dea97bfa"


def test_spec_fingerprint_is_pinned():
    assert spec_fingerprint() == PINNED_FINGERPRINT
    assert spec_fingerprint() == spec_fingerprint()


def test_freeze_clock():
    assert FREEZE_TAG == "rules7-freeze"
    assert LOGIC_EQUIVALENT_TO == "0263d0d"
    assert PSEUDO_FORWARD_START == "2026-10-14"
    assert FREEZE_DATE == "2026-11-10"
    assert FORWARD_START == "2026-11-11"


def test_freeze_specs_match_the_fingerprint():
    data = json.loads(SPECS_PATH.read_text(encoding="utf-8"))
    assert data["spec_fp"] == PINNED_FINGERPRINT
    assert data["logic"] == LOGIC_EQUIVALENT_TO
    assert data["as_of"] == "2026-11-09"
    codes = data["codes"]
    holdout = [row for row in codes.values() if row["universe"] == "holdout"]
    study = [row for row in codes.values() if row["universe"] == "study"]
    assert len(holdout) == 46
    assert len(study) == 7
    dividend = codes["SH512530"]
    assert dividend["type"] == "平稳震荡"
    assert dividend["applicable"] is True
    assert dividend["in_sample_closed"] == 8
    assert dividend["in_sample_open"] is True
    assert abs(dividend["in_sample_total"] - 1.71) < 0.02
    assert abs(dividend["in_sample_mdd"] - (-0.115)) < 0.005
    assert codes["SZ159892"]["applicable"] is False
