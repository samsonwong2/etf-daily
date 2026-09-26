#!/usr/bin/env python3
"""Overlay look-ahead retrospective_truth regime bands on a cleaned fig1 HTML.

Presets:
  A — pool defaults (prom=0.18, mindist=25, ±22%, min_seg=30)
  B — relaxed for mid-vol ETFs (prom=0.12, mindist=15, ±12%, min_seg=20)

Example::

    PYTHONPATH=. python decision_pack/scripts/overlay_retro_truth_bands.py \\
      --html ~/etf-daily-output/temp/plotly_outputs/20260824_from_listing/regime_transition_SH512530_沪深300红利ETF建信_20190923_20260824_adaptive_no_fig1_marks.html \\
      --preset B
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from decision_pack.src.fair_path_band_signals import (  # noqa: E402
    _extract_plotly_traces,
    load_ohlcv_from_regime_html,
)
from decision_pack.src.plotly_html_autoscale import write_adaptive_html  # noqa: E402

PRESETS: dict[str, dict[str, Any]] = {
    "A": {
        "prom": 0.18,
        "mindist": 25,
        "up_ret": 0.22,
        "down_ret": -0.22,
        "min_seg": 30,
        "label": "池默认 A(±22%/prom18%/min_seg30)",
    },
    "B": {
        "prom": 0.12,
        "mindist": 15,
        "up_ret": 0.12,
        "down_ret": -0.12,
        "min_seg": 20,
        "label": "放宽 B(±12%/prom12%/min_seg20)",
    },
}

COLOR_MAP = {
    "up": "rgba(18,161,80,0.12)",
    "down": "rgba(214,39,40,0.10)",
    "range": "rgba(148,148,148,0.14)",
}
LEGEND = (
    ("上涨态(事后真值)", "rgba(18,161,80,0.55)"),
    ("下跌态(事后真值)", "rgba(214,39,40,0.55)"),
    ("横盘态(事后真值)", "rgba(120,120,120,0.55)"),
)
CN = {"up": "上涨", "down": "下跌", "range": "横盘"}


def _load_overlay():
    path = _SCRIPTS / "overlay_oracle_pivots_on_regime_html.py"
    spec = importlib.util.spec_from_file_location("overlay_oracle_pivots", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_retro_truth_fn():
    path = _SCRIPTS / "eval_causal_regime_switch_pool.py"
    spec = importlib.util.spec_from_file_location("eval_causal_regime_switch_pool", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    # Avoid running CLI if any — module is importable as library
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.retrospective_truth


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--html", type=Path, required=True)
    p.add_argument("--preset", choices=sorted(PRESETS), default="B")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None, help="also write segments.csv here")
    return p.parse_args(argv)


def _resolve(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def labels_to_segments(
    index: pd.DatetimeIndex, labels: np.ndarray, close: pd.Series
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = len(labels)
    if n == 0:
        return pd.DataFrame()
    i0 = 0
    for i in range(1, n + 1):
        if i < n and labels[i] == labels[i0]:
            continue
        lab = str(labels[i0])
        i1 = i - 1
        p0 = float(close.iloc[i0])
        p1 = float(close.iloc[i1])
        rows.append(
            {
                "start": pd.Timestamp(index[i0]).strftime("%Y-%m-%d"),
                "end": pd.Timestamp(index[i1]).strftime("%Y-%m-%d"),
                "label": lab,
                "label_cn": CN.get(lab, lab),
                "n_bars": int(i1 - i0 + 1),
                "start_px": p0,
                "end_px": p1,
                "ret": p1 / p0 - 1.0 if p0 else float("nan"),
            }
        )
        i0 = i
    return pd.DataFrame(rows)


def build_shapes(
    seg_df: pd.DataFrame, ohlcv: pd.DataFrame
) -> list[dict[str, Any]]:
    lo = float(pd.to_numeric(ohlcv["$low"], errors="coerce").min())
    hi = float(pd.to_numeric(ohlcv["$high"], errors="coerce").max())
    span = max(hi - lo, 1e-6)
    y0 = lo - span * 0.02
    y1 = hi + span * 0.05
    shapes: list[dict[str, Any]] = []
    for _, r in seg_df.iterrows():
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=r["start"],
                x1=r["end"],
                y0=y0,
                y1=y1,
                fillcolor=COLOR_MAP.get(str(r["label"]), "rgba(0,0,0,0.05)"),
                line=dict(width=0),
                layer="below",
            )
        )
    return shapes


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    html_path = _resolve(args.html)
    if not html_path.exists():
        raise FileNotFoundError(html_path)
    preset = PRESETS[args.preset]
    overlay = _load_overlay()
    retro_truth = _load_retro_truth_fn()

    close, ohlcv = load_ohlcv_from_regime_html(html_path)
    close = pd.to_numeric(close, errors="coerce")
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index)).normalize()
    px = close.astype(float).to_numpy()

    labels = retro_truth(
        px,
        prom=float(preset["prom"]),
        mindist=int(preset["mindist"]),
        up_ret=float(preset["up_ret"]),
        down_ret=float(preset["down_ret"]),
        min_seg=int(preset["min_seg"]),
    )
    seg_df = labels_to_segments(close.index, labels, close)

    html_txt = html_path.read_text(encoding="utf-8")
    traces = _extract_plotly_traces(html_txt)
    layout = overlay.extract_layout(html_txt)

    # Keep existing non-fig1 shapes; replace fig1 (y/y1 + x/x1) shapes with truth bands.
    old_shapes = layout.get("shapes") if isinstance(layout.get("shapes"), list) else []
    kept_shapes = [
        s
        for s in old_shapes
        if not (
            isinstance(s, dict)
            and str(s.get("yref") or "y") in {"y", "y1"}
            and str(s.get("xref") or "x") in {"x", "x1"}
        )
    ]
    new_shapes = kept_shapes + build_shapes(seg_df, ohlcv)
    layout = {**layout, "shapes": new_shapes}

    # Drop old legend-only fig1 state squares if any remain; add our legend.
    kept_traces = []
    for tr in traces:
        name = str(tr.get("name") or "")
        if name in {"上涨态", "下跌态", "横盘态"} and tr.get("x") in (None, [], [None]):
            continue
        # also skip null-x legend squares
        if name in {"上涨态", "下跌态", "横盘态"}:
            x = tr.get("x")
            if x is None or (isinstance(x, list) and all(v is None for v in x)):
                continue
        kept_traces.append(tr)

    for name, color in LEGEND:
        kept_traces.append(
            dict(
                type="scatter",
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color=color, symbol="square"),
                name=name,
                yaxis="y",
                xaxis="x",
                showlegend=True,
            )
        )

    n_up = int((labels == "up").sum())
    n_dn = int((labels == "down").sum())
    n_rg = int((labels == "range").sum())
    n = len(labels)
    note = (
        f"｜事后真值色带 {preset['label']}｜偷看未来｜"
        f"上涨{n_up}/{n}({n_up/n:.0%}) 下跌{n_dn}/{n}({n_dn/n:.0%}) "
        f"横盘{n_rg}/{n}({n_rg/n:.0%})｜段数{len(seg_df)}"
    )
    title = layout.get("title")
    if isinstance(title, dict):
        text = str(title.get("text") or "")
        # strip prior retro-band notes
        import re

        text = re.sub(
            r"<br><span style='font-size:12px;color:#333'>.*?事后真值色带.*?</span>",
            "",
            text,
        )
        layout = {
            **layout,
            "title": {
                **title,
                "text": text
                + f"<br><span style='font-size:12px;color:#333'>{note}</span>",
            },
        }
    elif isinstance(title, str):
        layout = {**layout, "title": title + note}
    else:
        layout = {**layout, "title": note}

    out_path = args.out
    if out_path is None:
        out_path = html_path.with_name(
            html_path.stem + f"_retro_bands_{args.preset}.html"
        )
    else:
        out_path = _resolve(out_path)

    fig = go.Figure(data=kept_traces, layout=layout)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_adaptive_html(fig, out_path, include_plotlyjs="cdn")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = out_path.parent / f"retro_bands_{args.preset}_{html_path.stem[:40]}"
        # Prefer beside listing under a short folder if stem is long
        code_guess = None
        for part in html_path.name.split("_"):
            if part.startswith(("SH", "SZ")) and len(part) >= 8:
                code_guess = part
                break
        if code_guess:
            out_dir = html_path.parent / f"fig3_11_bt_{code_guess}" / f"retro_bands_{args.preset}"
    else:
        out_dir = _resolve(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_path = out_dir / "segments.csv"
    seg_df.to_csv(seg_path, index=False, encoding="utf-8-sig")
    summary = {
        "html_src": str(html_path),
        "html_out": str(out_path),
        "preset": args.preset,
        "params": {k: v for k, v in preset.items() if k != "label"},
        "n_bars": n,
        "n_up": n_up,
        "n_down": n_dn,
        "n_range": n_rg,
        "frac_up": n_up / n if n else None,
        "frac_down": n_dn / n if n else None,
        "frac_range": n_rg / n if n else None,
        "n_segments": int(len(seg_df)),
        "segments_csv": str(seg_path),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[OK] wrote {seg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
