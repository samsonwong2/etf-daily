#!/usr/bin/env python3
"""Freeze the legacy fig7–11 baseline for the fat-tail audit.

Reads HTML only. Writes parquet + timing under
``$PLOTLY_ROOT/research/fair_path_fat_tail_audit/golden``. Does not change
production calculations.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from etf_daily.lib.fair_path_band_signals import (  # noqa: E402
    atr20,
    build_checklist_flags_frame,
    compute_multi_scale_frame,
    find_regime_html,
    load_ohlcv_from_regime_html,
)
from etf_daily.lib.fair_path_research import ewma_scaled_envelope  # noqa: E402
from etf_daily.lib.fair_path_tail_audit import GOLDEN_CODES, PANELS  # noqa: E402
from etf_daily.paths import PLOTLY_OUTPUTS_DIR  # noqa: E402
from etf_daily.plots.plot_regime_transition_example import (  # noqa: E402
    causal_path_residual_signed_percentile,
)

LISTING = PLOTLY_OUTPUTS_DIR / "20261019_from_listing"
OUT = PLOTLY_OUTPUTS_DIR / "research" / "fair_path_fat_tail_audit" / "golden"


def _one(code: str) -> dict[str, float]:
    html = find_regime_html(LISTING, code)
    if html is None:
        raise FileNotFoundError(code)
    t0 = time.perf_counter()
    close, ohlcv = load_ohlcv_from_regime_html(html)
    t_html = time.perf_counter()
    ms = compute_multi_scale_frame(close)
    t_frame = time.perf_counter()
    extra: dict[str, pd.Series] = {}
    for key, years in PANELS:
        path = ms[f"{key}_path"]
        extra[f"{key}_p95_pct"] = causal_path_residual_signed_percentile(path, close)
        if years >= 0.25 - 1e-9:
            up, lo, _ = ewma_scaled_envelope(path, close)
            extra[f"{key}_ewma_upper"] = up
            extra[f"{key}_ewma_lower"] = lo
    t_extra = time.perf_counter()
    flags = build_checklist_flags_frame(ms, code=code, atr_s=atr20(ohlcv))
    t_flags = time.perf_counter()
    frame = ms.copy()
    for col, ser in extra.items():
        frame[col] = ser.to_numpy(dtype=float)
    for col in ("B1", "B2", "B3", "B4", "B5"):
        frame[col] = flags[col].to_numpy()
    frame.index.name = "date"
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT / f"{code}.parquet")
    return {
        "code": code,
        "n": float(len(close)),
        "html_s": t_html - t0,
        "frame_s": t_frame - t_html,
        "extra_s": t_extra - t_frame,
        "flags_s": t_flags - t_extra,
        "total_s": t_flags - t0,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for code in GOLDEN_CODES:
        row = _one(code)
        rows.append(row)
        print(f"[golden] {code} n={int(row['n'])} total={row['total_s']:.2f}s")
    timing = pd.DataFrame(rows)
    timing.to_csv(OUT / "timing.csv", index=False)
    meta = {
        "listing": str(LISTING),
        "codes": list(GOLDEN_CODES),
        "median_frame_s": float(timing["frame_s"].median()),
        "max_total_s": float(timing["total_s"].max()),
    }
    (OUT / "timing.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] {OUT}")


if __name__ == "__main__":
    main()
