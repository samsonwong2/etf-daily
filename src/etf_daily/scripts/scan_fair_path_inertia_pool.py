#!/usr/bin/env python3
"""Pool scan: expanding vs 3y/5y rolling P95 vs EWMA vs calm-clamped EWMA.

Isolated inertia experiment. Same OLS path, production B2. Production
P95/checklist untouched. Dual gate plus a history-length split (n>=1260)
because 3–5y rolling ≈ expanding on young ETFs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from etf_daily.lib.fair_path_band_signals import (  # noqa: E402
    FIG_PANELS,
    compute_multi_scale_frame,
    load_ohlcv_from_regime_html,
)
from etf_daily.lib.fair_path_research import (  # noqa: E402
    COVERAGE_PASS_LO,
    LOOKBACK_5Y,
    b1_entries_variant,
    build_inertia_panels,
    coverage_report,
    regime_pass_table,
)
from etf_daily.lib.regime_transition_plot import (  # noqa: E402
    compute_volatility_regime_frame,
)

DEFAULT_LISTING = (
    PLOTLY_OUTPUTS_DIR / "20260908_from_listing"
)
DEFAULT_OUT = PLOTLY_OUTPUTS_DIR / "research/pool_scan_inertia"
SKIP_CODES = {"SH511010", "SH511260"}

# (variant, band key on the inertia panel)
VARIANTS: tuple[tuple[str, str], ...] = (
    ("roll3y", "roll3y"),
    ("roll5y", "roll5y"),
    ("ewma", "ewma"),
    ("ewma_clamp", "ewma_clamp"),
)
_BAND_COLS = {
    "prod_p95": ("upper", "lower"),
    "roll3y": ("roll3y_upper", "roll3y_lower"),
    "roll5y": ("roll5y_upper", "roll5y_lower"),
    "ewma": ("ewma_upper", "ewma_lower"),
    "ewma_clamp": ("ewma_clamp_upper", "ewma_clamp_lower"),
}


def _fwd_stats(close: pd.Series, mask: pd.Series) -> dict[str, float | int]:
    px = pd.to_numeric(close, errors="coerce")
    out: dict[str, float | int] = {"n_sig": int(mask.fillna(False).sum())}
    idx = px.index
    pos = {d: i for i, d in enumerate(idx)}
    arr = px.to_numpy(dtype=float)
    n = len(arr)
    for h in (5, 10, 20):
        rets: list[float] = []
        for d, flag in mask.items():
            if not bool(flag):
                continue
            i = pos.get(d)
            if i is None:
                continue
            j = i + 1 + h
            e = i + 1
            if e >= n or j >= n:
                continue
            if not (np.isfinite(arr[e]) and np.isfinite(arr[j]) and arr[e] > 0):
                continue
            rets.append(float(arr[j] / arr[e] - 1.0))
        if rets:
            s = np.asarray(rets, dtype=float)
            out[f"fwd_n_{h}d"] = int(s.size)
            out[f"fwd_mean_{h}d"] = float(s.mean())
            out[f"fwd_hit_{h}d"] = float((s > 0).mean())
        else:
            out[f"fwd_n_{h}d"] = 0
            out[f"fwd_mean_{h}d"] = float("nan")
            out[f"fwd_hit_{h}d"] = float("nan")
    return out


def _touch_lo_prod(ms: pd.DataFrame) -> pd.Series:
    return ms["fig9_touch_lo"].fillna(False) | ms["fig10_touch_lo"].fillna(False)


def _b1_prod(ms: pd.DataFrame) -> pd.Series:
    return b1_entries_variant(
        ms, touch_lo=_touch_lo_prod(ms), kalman_g=None, b2_mode="ols"
    )


def scan_one(code: str, html_path: Path, *, as_of: str | None) -> dict[str, Any]:
    close, _ohlcv = load_ohlcv_from_regime_html(html_path)
    close = close.sort_index()
    if as_of:
        close = close[close.index <= pd.Timestamp(as_of)]
    if len(close) < 120:
        return {"code": code, "skip": "short_history"}

    vol = compute_volatility_regime_frame(close)
    regime_s = vol.set_index("as_of")["vol_regime"].reindex(close.index)
    ms = compute_multi_scale_frame(close)
    panels = build_inertia_panels(close)
    n_hist = int(len(close))

    cov_parts: list[pd.DataFrame] = []
    for key, _, label in FIG_PANELS:
        pf = panels[key]
        bands = {
            name: (pf[hi], pf[lo]) for name, (hi, lo) in _BAND_COLS.items()
        }
        rep = coverage_report(close, bands, vol_regime=regime_s)
        rep.insert(0, "code", code)
        rep.insert(1, "panel", f"{key}({label})")
        cov_parts.append(rep)
    coverage = pd.concat(cov_parts, ignore_index=True)
    pass_tbl = regime_pass_table(coverage)
    if "code" not in pass_tbl.columns:
        pass_tbl.insert(0, "code", code)

    px = pd.to_numeric(close, errors="coerce").reindex(ms.index)
    signals: dict[str, Any] = {
        "code": code,
        "n_hist": n_hist,
        "long_hist": bool(n_hist >= LOOKBACK_5Y),
    }
    signals["prod"] = _fwd_stats(close, _b1_prod(ms))
    for name, band in VARIANTS:
        _hi, lo = _BAND_COLS[band]
        lo9 = panels["fig9"][lo].reindex(ms.index)
        lo10 = panels["fig10"][lo].reindex(ms.index)
        touch = ((px <= lo9) | (px <= lo10)).fillna(False)
        ent = b1_entries_variant(
            ms, touch_lo=touch, kalman_g=None, b2_mode="ols"
        )
        signals[name] = _fwd_stats(close, ent)
    return {
        "code": code,
        "n_hist": n_hist,
        "long_hist": n_hist >= LOOKBACK_5Y,
        "coverage": coverage,
        "pass": pass_tbl,
        "signal": signals,
    }


def _flatten_signals(sig: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "code": sig["code"],
        "n_hist": sig.get("n_hist"),
        "long_hist": sig.get("long_hist"),
    }
    for name, stats in sig.items():
        if name in {"code", "n_hist", "long_hist"}:
            continue
        if not isinstance(stats, dict):
            continue
        for k, v in stats.items():
            row[f"{name}_{k}"] = v
    return row


def _pool_b1(sig_df: pd.DataFrame, prefix: str) -> dict[str, Any]:
    n = pd.to_numeric(sig_df[f"{prefix}_n_sig"], errors="coerce").fillna(0)
    out: dict[str, Any] = {"n_sig": int(n.sum())}
    for h in (5, 10, 20):
        nc = pd.to_numeric(sig_df[f"{prefix}_fwd_n_{h}d"], errors="coerce").fillna(0)
        mean = pd.to_numeric(sig_df[f"{prefix}_fwd_mean_{h}d"], errors="coerce")
        hit = pd.to_numeric(sig_df[f"{prefix}_fwd_hit_{h}d"], errors="coerce")
        w = nc.to_numpy(dtype=float)
        m = mean.to_numpy(dtype=float)
        hh = hit.to_numpy(dtype=float)
        ok = np.isfinite(m) & np.isfinite(hh) & (w > 0)
        out[f"fwd_mean_{h}d"] = (
            float(np.average(m[ok], weights=w[ok])) if ok.any() else float("nan")
        )
        out[f"fwd_hit_{h}d"] = (
            float(np.average(hh[ok], weights=w[ok])) if ok.any() else float("nan")
        )
    return out


def _coverage_gate(pass_tbl: pd.DataFrame, band: str) -> dict[str, Any]:
    fig7 = pass_tbl[pass_tbl["panel"].astype(str).str.startswith("fig7")]
    sub = fig7[fig7["band"] == band]
    elig = sub[sub["eligible"]]
    if elig.empty:
        return {"codes_eligible": 0, "codes_all_pass": 0, "share": float("nan")}
    grouped = elig.groupby("code")["pass"].agg(["all", "size"])
    n_elig = int(len(grouped))
    n_pass = int(grouped["all"].sum())
    cov = pd.to_numeric(elig["coverage"], errors="coerce")
    by_rg = {
        str(rg): float(pd.to_numeric(gg["coverage"], errors="coerce").mean())
        for rg, gg in elig.groupby("bucket")
    }
    return {
        "codes_eligible": n_elig,
        "codes_all_pass": n_pass,
        "share": n_pass / n_elig if n_elig else float("nan"),
        "cell_pass_rate": float(elig["pass"].mean()),
        "cells_below_lo": int((cov < COVERAGE_PASS_LO).sum()),
        "mean_coverage_by_regime": by_rg,
    }


def _ablation_row(
    variant: str,
    band: str,
    b1: dict[str, Any],
    coverage: dict[str, Any],
    prod_b1: dict[str, Any],
    *,
    is_prod: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant": variant,
        "band": band,
        "b2": "ols",
        "b1": b1,
        "coverage": coverage,
    }
    if is_prod:
        return row
    cov_share = row["coverage"].get("share") or 0.0
    cov_ok = bool(cov_share >= 0.80)
    b1_ok = bool(
        np.isfinite(b1.get("fwd_mean_20d", np.nan))
        and np.isfinite(b1.get("fwd_hit_20d", np.nan))
        and b1["fwd_mean_20d"] >= prod_b1.get("fwd_mean_20d", np.inf)
        and b1["fwd_hit_20d"] >= prod_b1.get("fwd_hit_20d", np.inf)
    )
    row["gate_coverage"] = cov_ok
    row["gate_b1_not_worse"] = b1_ok
    row["replace_ok"] = bool(cov_ok and b1_ok)
    return row


def _format_table(ablation: list[dict[str, Any]]) -> list[str]:
    lines = [
        "variant   band      B1_n  fwd20     hit20   cov_share  replace"
    ]
    for a in ablation:
        b1 = a["b1"]
        lines.append(
            f"{a['variant']:<9} {a['band']:<9} "
            f"{b1.get('n_sig', 0):<5} "
            f"{b1.get('fwd_mean_20d', float('nan')):+.4f}  "
            f"{b1.get('fwd_hit_20d', float('nan')):.3f}   "
            f"{a['coverage'].get('share', float('nan')):.3f}      "
            f"{'OK' if a.get('replace_ok') else '-'}"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--listing-dir", type=Path, default=DEFAULT_LISTING)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--as-of", default="2026-09-08")
    ap.add_argument("--max-codes", type=int, default=None)
    ap.add_argument("--code", action="append", default=None)
    args = ap.parse_args(argv)

    listing_dir = args.listing_dir
    if not listing_dir.is_absolute():
        listing_dir = _PROJECT_ROOT / listing_dir
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = _PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    htmls = sorted(listing_dir.glob("regime_transition_*_adaptive.html"))
    htmls = [p for p in htmls if "_fig9_aux" not in p.name]
    by_code: dict[str, Path] = {}
    for p in htmls:
        parts = p.stem.split("_")
        code = parts[2].upper() if len(parts) > 2 else ""
        if code.startswith("SH") or code.startswith("SZ"):
            by_code[code] = p
    if args.code:
        want = {c.upper() for c in args.code}
        by_code = {c: p for c, p in by_code.items() if c in want}
    codes = sorted(c for c in by_code if c not in SKIP_CODES)
    if args.max_codes:
        codes = codes[: int(args.max_codes)]

    cov_all: list[pd.DataFrame] = []
    pass_all: list[pd.DataFrame] = []
    sig_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for i, code in enumerate(codes, 1):
        print(f"[{i}/{len(codes)}] {code}", flush=True)
        try:
            res = scan_one(code, by_code[code], as_of=args.as_of)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"code": code, "skip": str(exc)[:200]})
            continue
        if "skip" in res:
            skipped.append({"code": code, "skip": str(res["skip"])})
            continue
        cov_all.append(res["coverage"])
        pass_all.append(res["pass"])
        sig_rows.append(_flatten_signals(res["signal"]))

    coverage = pd.concat(cov_all, ignore_index=True) if cov_all else pd.DataFrame()
    pass_tbl = pd.concat(pass_all, ignore_index=True) if pass_all else pd.DataFrame()
    sig_df = pd.DataFrame(sig_rows)
    coverage.to_csv(out_dir / "coverage_by_code.csv", index=False, encoding="utf-8-sig")
    pass_tbl.to_csv(out_dir / "regime_pass.csv", index=False, encoding="utf-8-sig")
    sig_df.to_csv(out_dir / "signal_compare.csv", index=False, encoding="utf-8-sig")
    if skipped:
        pd.DataFrame(skipped).to_csv(
            out_dir / "skipped.csv", index=False, encoding="utf-8-sig"
        )

    gate_rule = (
        "forget diagnostic (post-storm width < expanding) AND "
        "calm/normal in [0.90, 0.98]; cluster/extreme >= 0.90; "
        "dual gate = coverage share >= 0.80 AND B1 fwd20 mean+hit >= prod"
    )

    def _build_ablation(sdf: pd.DataFrame, ptbl: pd.DataFrame) -> list[dict[str, Any]]:
        if sdf.empty:
            return []
        prod_b1 = _pool_b1(sdf, "prod")
        rows = [
            _ablation_row(
                "prod", "prod_p95", prod_b1, _coverage_gate(ptbl, "prod_p95"),
                prod_b1, is_prod=True,
            )
        ]
        for name, band in VARIANTS:
            rows.append(
                _ablation_row(
                    name, band, _pool_b1(sdf, name), _coverage_gate(ptbl, band),
                    prod_b1, is_prod=False,
                )
            )
        return rows

    ablation = _build_ablation(sig_df, pass_tbl)
    long_mask = (
        sig_df["long_hist"].fillna(False).astype(bool) if not sig_df.empty else pd.Series(dtype=bool)
    )
    long_codes = set(sig_df.loc[long_mask, "code"].astype(str)) if not sig_df.empty else set()
    pass_long = (
        pass_tbl[pass_tbl["code"].astype(str).isin(long_codes)]
        if not pass_tbl.empty
        else pass_tbl
    )
    ablation_long = _build_ablation(sig_df.loc[long_mask].copy(), pass_long)

    summary = {
        "n_codes_scanned": len(codes),
        "n_codes_ok": int(sig_df["code"].nunique()) if not sig_df.empty else 0,
        "n_codes_long_hist": int(long_mask.sum()) if not sig_df.empty else 0,
        "long_hist_min_bars": LOOKBACK_5Y,
        "gate": gate_rule,
        "replace_threshold": (
            "coverage share >= 0.80 AND B1 fwd20 mean+hit >= prod "
            "(replace only if forget diagnostic also passes)"
        ),
        "prod_b1": _pool_b1(sig_df, "prod") if not sig_df.empty else {},
        "ablation": ablation,
        "ablation_long_hist": ablation_long,
    }
    (out_dir / "ablation_gates.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = ["# all codes"] + _format_table(ablation)
    lines += ["", f"# n >= {LOOKBACK_5Y} bars"] + _format_table(ablation_long)
    (out_dir / "ablation_report.md").write_text(
        "# Expanding-inertia band ablation\n\n```\n" + "\n".join(lines) + "\n```\n",
        encoding="utf-8",
    )
    print("\n".join(lines))
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
