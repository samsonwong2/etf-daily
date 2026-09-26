#!/usr/bin/env python3
"""Append fig12 (clean K + six-state vrects) to from_listing adaptive HTML.

Idempotent: existing ``<!-- FIG12_SIX_STATES_START/END -->`` blocks are stripped
and redrawn. Intended as a post-processor after
``daily_adaptive_from_listing.sh`` because incremental ``write_adaptive_html``
drops extra trailing divs.

Example::

    PYTHONPATH=. python src/etf_daily/plots/add_fig12_six_states.py \\
      --html-dir ~/etf-daily-output/temp/plotly_outputs/20260914_from_listing --jobs 8
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.lib.fair_path_band_signals import (  # noqa: E402
    compute_multi_scale_frame,
    enrich_frame_for_checklist,
    find_regime_html,
    load_ohlcv_from_regime_html,
)
from etf_daily.lib.six_states import (  # noqa: E402
    build_fig12_figure,
    classify_six_states,
    current_state,
)

FIG12_START = "<!-- FIG12_SIX_STATES_START -->"
FIG12_END = "<!-- FIG12_SIX_STATES_END -->"
FIG12_BLOCK_RE = re.compile(
    r"\s*<!--\s*FIG12_SIX_STATES_START\s*-->.*?<!--\s*FIG12_SIX_STATES_END\s*-->",
    re.DOTALL,
)
_FNAME_RE = re.compile(
    r"^regime_transition_((?:SH|SZ)\d+)_(.+)_(\d{8})_(\d{8})_adaptive\.html$"
)
_HTML_GLOB = "regime_transition_*_adaptive.html"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--html-dir", type=Path, required=True, help="from_listing HTML directory")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument(
        "--code",
        action="append",
        default=[],
        help="limit to this ticker (repeatable)",
    )
    p.add_argument(
        "--errors-csv",
        type=Path,
        default=None,
        help="default: <html-dir>/fig12_errors.csv",
    )
    return p.parse_args(argv)


def parse_html_label(path: Path) -> tuple[str, str]:
    """Return (code, 'CODE 中文名') from a regime_transition filename."""
    m = _FNAME_RE.match(path.name)
    if not m:
        stem = path.stem
        return stem, stem
    code, name, _start, _end = m.group(1), m.group(2), m.group(3), m.group(4)
    return code, f"{code} {name}"


def strip_fig12_block(html: str) -> str:
    return FIG12_BLOCK_RE.sub("", html)


def _inject_before_body_end(html: str, block: str) -> str:
    m = re.search(r"</body\s*>", html, flags=re.IGNORECASE)
    if m is None:
        return html.rstrip() + "\n" + block + "\n"
    return html[: m.start()] + block + "\n" + html[m.start() :]


def build_fig12_block(fig_html: str, *, state: str) -> str:
    heading = (
        f'<h3 style="font-family:sans-serif;padding-left:12px">图12 六状态背景色'
        f'<span style="font-size:13px;color:#666">（背景=六状态趋势划分，'
        f"判定方法见 documents/六状态趋势划分_八品种验证.md §8；"
        f"当前状态：{state}）</span></h3>"
    )
    return (
        f"\n{FIG12_START}\n"
        f'<hr style="margin:24px 0">\n'
        f"{heading}\n"
        f"{fig_html}\n"
        f"{FIG12_END}\n"
    )


def list_html_files(html_dir: Path, codes: list[str] | None) -> list[Path]:
    html_dir = html_dir.resolve()
    if not html_dir.is_dir():
        raise FileNotFoundError(f"--html-dir not a directory: {html_dir}")
    wanted = {c.strip().upper() for c in (codes or []) if c and c.strip()}
    if wanted:
        files: list[Path] = []
        seen: set[Path] = set()
        for code in sorted(wanted):
            hit = find_regime_html(html_dir, code)
            if hit is None:
                continue
            resolved = hit.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(resolved)
        return files
    out: list[Path] = []
    for p in sorted(html_dir.glob(_HTML_GLOB)):
        if not _FNAME_RE.match(p.name):
            continue
        out.append(p)
    return out


def add_fig12_to_html(html_path: Path) -> dict[str, str]:
    """Strip + redraw fig12 on one adaptive HTML. Returns status dict."""
    html_path = Path(html_path)
    code, code_name = parse_html_label(html_path)
    close, ohlcv = load_ohlcv_from_regime_html(html_path)
    ms = enrich_frame_for_checklist(compute_multi_scale_frame(close))
    states = classify_six_states(ms)
    state = current_state(states)
    last_dt = pd_last_date(close)
    fig = build_fig12_figure(ohlcv, states, code_name)
    fig_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id="fig12_six_states")
    raw = html_path.read_text(encoding="utf-8")
    cleaned = strip_fig12_block(raw)
    block = build_fig12_block(fig_html, state=state)
    html_path.write_text(_inject_before_body_end(cleaned, block), encoding="utf-8")
    return {
        "ok": "1",
        "path": str(html_path),
        "code": code,
        "state": state,
        "last_date": last_dt,
        "error": "",
    }


def pd_last_date(close) -> str:
    try:
        return str(close.index[-1])[:10]
    except Exception:  # noqa: BLE001
        return ""


def _worker(path_str: str) -> dict[str, str]:
    try:
        return add_fig12_to_html(Path(path_str))
    except Exception as exc:  # noqa: BLE001
        path = Path(path_str)
        code, _ = parse_html_label(path)
        return {
            "ok": "0",
            "path": path_str,
            "code": code,
            "state": "",
            "last_date": "",
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


def write_errors_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["code", "path", "error"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    html_dir = args.html_dir.expanduser().resolve()
    files = list_html_files(html_dir, args.code)
    errors_csv = (
        args.errors_csv.expanduser().resolve()
        if args.errors_csv is not None
        else html_dir / "fig12_errors.csv"
    )
    if not files:
        write_errors_csv(errors_csv, [])
        print(f"[fig12] no matching HTML in {html_dir}")
        return 0

    jobs = max(1, int(args.jobs))
    results: list[dict[str, str]] = []
    if jobs == 1 or len(files) == 1:
        for p in files:
            results.append(_worker(str(p)))
            rec = results[-1]
            flag = "OK" if rec["ok"] == "1" else "FAIL"
            print(f"[fig12] {flag} {rec['code']} {rec.get('state', '')} {rec.get('last_date', '')}")
            if rec["ok"] != "1":
                print(rec["error"], file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(_worker, str(p)): p for p in files}
            done = 0
            n = len(files)
            for fut in as_completed(futs):
                rec = fut.result()
                results.append(rec)
                done += 1
                flag = "OK" if rec["ok"] == "1" else "FAIL"
                print(
                    f"[fig12] {done}/{n} {flag} {rec['code']} "
                    f"{rec.get('state', '')} {rec.get('last_date', '')}"
                )
                if rec["ok"] != "1":
                    print(rec["error"], file=sys.stderr)

    failed = [r for r in results if r.get("ok") != "1"]
    write_errors_csv(errors_csv, failed)
    n_ok = len(results) - len(failed)
    print(f"[fig12] done ok={n_ok} fail={len(failed)} errors={errors_csv}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
