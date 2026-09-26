#!/usr/bin/env python3
"""Replay early-down diagnostic alerts on adaptive codes (no regime mutation).

Example::

    python decision_pack/scripts/replay_early_down_alerts.py \
      --adaptive-dir ~/etf-daily-output/temp/plotly_outputs/20260807all_adaptive \
      --as-of 2026-05-22 --code SH515060
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runtime_paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from decision_pack.src.adaptive_execution_plan import truncate_ohlcv_through  # noqa: E402
from decision_pack.src.adaptive_stage_common import prepare_features  # noqa: E402
from decision_pack.src.early_down_alert import (  # noqa: E402
    build_early_down_alert_row,
    build_early_down_frame,
    write_early_down_artifacts,
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--adaptive-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260807all_adaptive",
    )
    p.add_argument("--as-of", required=True)
    p.add_argument("--code", action="append", default=[])
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: ~/etf-daily-output/temp/plotly_outputs/early_down_replay/<as_of tag>",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    adaptive_dir = (
        args.adaptive_dir
        if args.adaptive_dir.is_absolute()
        else _PROJECT_ROOT / args.adaptive_dir
    )
    cfg_dir = adaptive_dir / "configs"
    summary_path = adaptive_dir / "batch_summary.csv"
    if not cfg_dir.is_dir() or not summary_path.exists():
        print(f"[ERROR] adaptive batch incomplete: {adaptive_dir}", file=sys.stderr)
        return 2

    summary = pd.read_csv(summary_path)
    codes = [c.upper() for c in args.code] if args.code else [
        str(c).upper() for c in summary["code"].tolist()
    ]
    if args.max_codes:
        codes = codes[: args.max_codes]
    name_map = {
        str(r.code).upper(): str(r.name)
        for r in summary.itertuples(index=False)
        if hasattr(r, "name")
    }

    tag = args.as_of.replace("-", "")
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = PLOTLY_OUTPUTS_DIR / "early_down_replay" / tag
    elif not out_dir.is_absolute():
        out_dir = _PROJECT_ROOT / out_dir

    ev = _load("ev_ed_replay", _PROJECT_ROOT / "decision_pack/scripts/eval_causal_regime_switch_pool.py")
    pm = _load("pm_ed_replay", _PROJECT_ROOT / "decision_pack/scripts/plot_regime_transition_example.py")
    pm._init_qlib(None)

    rows = []
    for i, code in enumerate(codes, 1):
        cfg = json.loads((cfg_dir / f"{code}.json").read_text(encoding="utf-8"))
        method = str(cfg["method"])
        ohlcv = pm.load_qlib_ohlcv(
            code,
            "2015-01-01",
            args.as_of,
            init_qlib=False,
            lookback_calendar_days=220,
            clip_to_window=False,
        )
        ohlcv_asof = truncate_ohlcv_through(ohlcv, args.as_of)
        px = prepare_features(ev, ohlcv_asof, method=method)
        regime = str(px["regime"].iloc[-1]) if "regime" in px.columns else "range"
        row = build_early_down_alert_row(
            code=code,
            name=name_map.get(code, code),
            method=method,
            as_of=args.as_of,
            regime_asof=regime,
            px=px,
        )
        rows.append(row)
        print(
            f"[{i}/{len(codes)}] {code} regime={regime} level={row['level']} "
            f"reasons={row['reasons'] or '-'}",
            flush=True,
        )

    frame = build_early_down_frame(rows)
    paths = write_early_down_artifacts(
        frame,
        out_dir,
        meta={
            "as_of": args.as_of,
            "adaptive_dir": str(adaptive_dir),
            "layer": "early_down_diagnostic_replay",
            "mutates_regime": False,
            "mutates_trades": False,
        },
    )
    n_alert = int((frame["level"] == "alert").sum())
    print(f"[OK] alerts={n_alert} -> {paths['csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
