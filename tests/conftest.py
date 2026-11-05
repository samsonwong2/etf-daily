"""Skip tests that call Qlib when Qlib is not installed."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

_QLIB = importlib.util.find_spec("qlib") is not None
_NEEDS_QLIB = re.compile(r"(?m)^(?:import qlib|from qlib |.*load_qlib_close)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config
    skip = pytest.mark.skip(reason="qlib is not installed")
    seen: dict[str, bool] = {}
    for item in items:
        path = str(item.fspath)
        if path not in seen:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
            seen[path] = _NEEDS_QLIB.search(text) is not None
        if not seen[path]:
            continue
        item.add_marker(pytest.mark.qlib)
        if not _QLIB:
            item.add_marker(skip)
