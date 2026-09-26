#!/usr/bin/env python3
"""One-off: SZ159981 adaptive HTML with multi-trail fair paths.

Default panels: fig7=2y, fig8=1y, fig9=6m, fig10=3m, fig11=1m.
Does not change the pool default. Example::

    PYTHONPATH=. ~/etf-daily-output/python/envs/py312/bin/python \\
      decision_pack/scripts/plot_sz159981_dual_trail_fig8.py \\
      --as-of 2026-08-18
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runtime_paths import PLOTLY_OUTPUTS_DIR  # noqa: E402
from decision_pack.scripts import plot_adaptive_stage_pool as pool  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--as-of", default="2026-08-18")
    p.add_argument("--listing-start", default="2020-01-17", help="plot/OOS start")
    p.add_argument("--train-cutoff", default="2026-04-01")
    p.add_argument(
        "--config-source-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260818_from_listing",
        help="reuse frozen configs (+ train cache) from from_listing run",
    )
    p.add_argument(
        "--html-out-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260818_sz159981_dual_trail_fig8",
    )
    p.add_argument(
        "--extra-trail-years",
        type=str,
        default="1,0.5,0.25,1/12",
        help="comma-separated extra trails after fig7=2y "
        "(default: 1y + 6m + 3m + 1m)",
    )
    args = p.parse_args(argv)

    as_of = str(args.as_of)
    out_dir = Path(args.html_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(args.config_source_dir)
    cfg_src = src / "configs" / "SZ159981.json"
    if not cfg_src.exists():
        raise SystemExit(f"missing frozen config: {cfg_src}")

    # Seed config into out_dir so pool reuses method without retrain.
    cfg_dst = out_dir / "configs"
    cfg_dst.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(cfg_src, cfg_dst / "SZ159981.json")
    for cache in src.glob(".train_cache_*.json"):
        shutil.copy2(cache, out_dir / cache.name)

    argv_pool = [
        "--code",
        "SZ159981",
        "--start-date",
        str(args.listing_start),
        "--end-date",
        as_of,
        "--train-cutoff",
        str(args.train_cutoff),
        "--html-out-dir",
        str(out_dir),
        "--fair-path-extra-trail-years",
        str(args.extra_trail_years),
        "--continue-on-error",
    ]
    print(
        f"[multi-trail] SZ159981 fig7=2y + extras={args.extra_trail_years} "
        f"→ {out_dir}"
    )
    return int(pool.main(argv_pool))


if __name__ == "__main__":
    raise SystemExit(main())
