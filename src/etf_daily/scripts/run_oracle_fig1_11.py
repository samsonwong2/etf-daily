#!/usr/bin/env python3
"""One-shot: fig1–11 事后买卖点 md 包 + 叠图 HTML。

对每个标的写出：

  <listing-dir>/fig3_11_bt_<CODE>/oracle_fig1_11_common_markers/
      report.md  summary.json  oracle_pivots.csv  feature_rates.csv
  <listing-dir>/regime_transition_<CODE>_..._adaptive_oracle_pivots.html

默认 listing 目录是 ``~/etf-daily-output/temp/plotly_outputs/20260824_from_listing``，
脚本按 ``regime_transition_<CODE>_*.html`` 自动找源图。

Examples（在仓库根目录执行）::

    PYTHONPATH=. python src/etf_daily/scripts/run_oracle_fig1_11.py --code SH512690

    PYTHONPATH=. python src/etf_daily/scripts/run_oracle_fig1_11.py \\
        --code SH512690 SH515880 SH513050

    PYTHONPATH=. python src/etf_daily/scripts/run_oracle_fig1_11.py \\
        --listing-dir ~/etf-daily-output/temp/plotly_outputs/20260824_from_listing \\
        --code SH518880

    PYTHONPATH=. python src/etf_daily/scripts/run_oracle_fig1_11.py \\
        --html ~/etf-daily-output/temp/plotly_outputs/20260824_from_listing/regime_transition_SH512530_沪深300红利ETF建信_20190923_20260824_adaptive.html

    # 只要 md，不要叠图 HTML
    PYTHONPATH=. python src/etf_daily/scripts/run_oracle_fig1_11.py --code SH512690 --skip-overlay
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
_SCRIPTS = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.paths import PLOTLY_OUTPUTS_DIR  # noqa: E402


def _load_sibling(filename: str, modname: str) -> Any:
    path = _SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(modname, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--code",
        nargs="+",
        default=None,
        help="one or more symbols, e.g. SH512690 SH515880",
    )
    p.add_argument(
        "--html",
        nargs="+",
        default=None,
        type=Path,
        help="explicit regime_transition HTML path(s)",
    )
    p.add_argument(
        "--listing-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260824_from_listing",
        help="directory that contains regime_transition_*_adaptive.html",
    )
    p.add_argument("--anchor-code", default="SH510300")
    p.add_argument("--skip-overlay", action="store_true", help="write md/csv only")
    p.add_argument("--left", type=int, default=10)
    p.add_argument("--right", type=int, default=10)
    p.add_argument("--fwd", type=int, default=40)
    p.add_argument("--buy-thr", type=float, default=0.08)
    p.add_argument("--sell-thr", type=float, default=0.08)
    p.add_argument("--thin-gap", type=int, default=8)
    args = p.parse_args(argv)
    if not args.code and not args.html:
        p.error("need --code and/or --html")
    return args


def _study_argv(args: argparse.Namespace, *, code: str | None, html: Path | None) -> list[str]:
    argv = [
        "--listing-dir",
        str(args.listing_dir),
        "--anchor-code",
        str(args.anchor_code),
        "--left",
        str(args.left),
        "--right",
        str(args.right),
        "--fwd",
        str(args.fwd),
        "--buy-thr",
        str(args.buy_thr),
        "--sell-thr",
        str(args.sell_thr),
        "--thin-gap",
        str(args.thin_gap),
    ]
    if html is not None:
        argv += ["--html", str(html)]
    if code is not None:
        argv += ["--code", str(code)]
    return argv


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    oracle = _load_sibling("oracle_fig1_11_common_markers.py", "oracle_fig1_11_common_markers")
    overlay = None if args.skip_overlay else _load_sibling(
        "overlay_oracle_pivots_on_regime_html.py", "overlay_oracle_pivots_on_regime_html"
    )

    jobs: list[tuple[str | None, Path | None]] = []
    if args.html:
        for h in args.html:
            jobs.append((None, Path(h)))
    if args.code:
        for c in args.code:
            jobs.append((str(c).upper(), None))

    failed: list[str] = []
    for code, html in jobs:
        label = str(html) if html is not None else str(code)
        print(f"\n========== {label} ==========", flush=True)
        try:
            result = oracle.run_study(_study_argv(args, code=code, html=html))
            print(f"[OK] md  {result['report']}")
            if overlay is not None:
                out_html = overlay.run_overlay(
                    html=result["html"],
                    pivots=result["pivots"],
                )
                print(f"[OK] html {out_html}")
        except Exception as exc:
            print(f"[FAIL] {label}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed.append(label)

    if failed:
        print(f"\n[DONE] failed {len(failed)}/{len(jobs)}: {failed}", file=sys.stderr)
        return 1
    print(f"\n[DONE] {len(jobs)} symbol(s) ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
