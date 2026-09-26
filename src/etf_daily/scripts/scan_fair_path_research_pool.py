#!/usr/bin/env python3
"""Pool ablation scan: fair-path band variants vs production, dual gate.

Reads K-lines from existing ``*_from_listing`` regime HTML (no Qlib).
Production P95 and the production checklist are untouched; every variant
is research-only.

Band variants (vol-scaled rails keep the per-ticker causal calm P95
floor; B2 = long-side filter). Rolling-|z| and extreme-block were
dropped after the previous round. This round pins calm coverage with
a causal online scale ``alpha_t`` (conformal scores on past calm days):

- ``prod`` — production P95 rails, production B2 (reference baseline)
- ``vs_ols_noalpha`` — vol-scaled rails, residual P95 floor only, B2 OLS
- ``vs_ols`` — same rails + online calm α, B2 OLS
- ``vs_kalman`` — online-α rails, Kalman B2
- ``vs_both`` — online-α rails, B2 = OLS AND Kalman

Outputs per variant: coverage gate share, B1 count / fwd20 / hit20, and
whether the dual gate passes (coverage share >= 0.80 AND B1 20d mean and
hit >= production's).
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
    detect_s1_trim_signals,
    load_ohlcv_from_regime_html,
)
from etf_daily.lib.fair_path_research import (  # noqa: E402
    COVERAGE_PASS_LO,
    b1_entries_variant,
    build_research_panels,
    coverage_report,
    kalman_trend_path,
    regime_pass_table,
)
from etf_daily.lib.regime_transition_plot import (  # noqa: E402
    compute_volatility_regime_frame,
)

DEFAULT_LISTING = (
    PLOTLY_OUTPUTS_DIR / "20260908_from_listing"
)
DEFAULT_OUT = PLOTLY_OUTPUTS_DIR / "research/pool_scan"
SKIP_CODES = {"SH511010", "SH511260"}

# (variant, b2_mode, block_extreme, calm_online)
VARIANTS: tuple[tuple[str, str, bool, bool], ...] = (
    ("vs_ols_noalpha", "ols", False, False),
    ("vs_ols", "ols", False, True),
    ("vs_kalman", "kalman", False, True),
    ("vs_both", "both", False, True),
)


def _fwd_stats(close: pd.Series, mask: pd.Series) -> dict[str, float | int]:
    """Mean / hit-rate of next-bar 5/10/20d returns after True days."""
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
            j = i + 1 + h  # next-bar entry, then h days
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
    return (ms["fig9_touch_lo"].fillna(False) | ms["fig10_touch_lo"].fillna(False))


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
    _, k_g, _, _ = kalman_trend_path(close)
    k_g = k_g.reindex(ms.index)

    # Band sets for coverage: production, residual-floor, online-α.
    panel_bands: dict[str, dict[str, tuple[pd.Series, pd.Series]]] = {}
    panels_on = build_research_panels(close, vol_regime=regime_s, z_lookback=0)
    panels_off = build_research_panels(
        close, vol_regime=regime_s, z_lookback=0, calm_online=False
    )
    touch_cache: dict[bool, pd.Series] = {}

    cov_parts: list[pd.DataFrame] = []
    for key, _, label in FIG_PANELS:
        pe = panels_on[key]
        po = panels_off[key]
        bands = {
            "prod_p95": (pe["upper"], pe["lower"]),
            "vs_expanding": (pe["vs_upper"], pe["vs_lower"]),
            "vs_expanding_noalpha": (po["vs_upper"], po["vs_lower"]),
        }
        panel_bands[key] = bands
        rep = coverage_report(close, bands, vol_regime=regime_s)
        rep.insert(0, "code", code)
        rep.insert(1, "panel", f"{key}({label})")
        cov_parts.append(rep)
    coverage = pd.concat(cov_parts, ignore_index=True)
    pass_tbl = regime_pass_table(coverage)
    if "code" not in pass_tbl.columns:
        pass_tbl.insert(0, "code", code)

    def _touch_lo(online: bool) -> pd.Series:
        if online not in touch_cache:
            src = panels_on if online else panels_off
            lo9 = src["fig9"]["vs_lower"].reindex(ms.index)
            lo10 = src["fig10"]["vs_lower"].reindex(ms.index)
            px = pd.to_numeric(close, errors="coerce").reindex(ms.index)
            touch_cache[online] = ((px <= lo9) | (px <= lo10)).fillna(False)
        return touch_cache[online]

    signals: dict[str, Any] = {"code": code}
    # Production reference (production rails + production B2).
    signals["prod"] = _fwd_stats(close, _b1_prod(ms))
    for name, b2_mode, block_ext, online in VARIANTS:
        touch = _touch_lo(online)
        ent = b1_entries_variant(
            ms,
            touch_lo=touch,
            kalman_g=k_g,
            b2_mode=b2_mode,
            vol_regime=regime_s,
            block_extreme=block_ext,
        )
        signals[name] = _fwd_stats(close, ent)

    # S1 reference: production vs expanding research upper rails.
    s1_p = detect_s1_trim_signals(ms)
    px = pd.to_numeric(close, errors="coerce").reindex(ms.index)
    hi9 = panel_bands["fig9"]["vs_expanding"][0].reindex(ms.index)
    hi10 = panel_bands["fig10"]["vs_expanding"][0].reindex(ms.index)
    s1_r = ((px >= hi9) | (px >= hi10)).fillna(False)
    signals["s1_prod"] = _fwd_stats(close, s1_p)
    signals["s1_vs"] = _fwd_stats(close, s1_r)

    return {"code": code, "coverage": coverage, "pass": pass_tbl, "signal": signals}


def _b1_prod(ms: pd.DataFrame) -> pd.Series:
    """Production B1: touch fig9/fig10 production lower + production B2."""
    return b1_entries_variant(
        ms, touch_lo=_touch_lo_prod(ms), kalman_g=None, b2_mode="ols"
    )


def _flatten_signals(sig: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"code": sig["code"]}
    for name, stats in sig.items():
        if name == "code":
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
        out[f"fwd_mean_{h}d"] = float(np.average(m[ok], weights=w[ok])) if ok.any() else float("nan")
        out[f"fwd_hit_{h}d"] = float(np.average(hh[ok], weights=w[ok])) if ok.any() else float("nan")
    return out


def _coverage_gate(pass_tbl: pd.DataFrame, band: str) -> dict[str, Any]:
    """Share of codes whose ALL eligible fig7 cells pass the relaxed gate."""
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


# Map B1 variants to the band that drives their rails.
_VARIANT_BAND = {
    "vs_ols_noalpha": "vs_expanding_noalpha",
    "vs_ols": "vs_expanding",
    "vs_kalman": "vs_expanding",
    "vs_both": "vs_expanding",
}


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
    codes = sorted(by_code)
    codes = [c for c in codes if c not in SKIP_CODES]
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
        except Exception as exc:  # noqa: BLE001 — pool scan continues
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
        "calm/normal in [0.90, 0.98]; cluster/extreme >= 0.90 (no upper); "
        "dual gate = coverage share >= 0.80 AND B1 fwd20 mean+hit >= prod"
    )
    prod_b1 = _pool_b1(sig_df, "prod")
    ablation: list[dict[str, Any]] = [
        {
            "variant": "prod",
            "band": "prod_p95",
            "b2": "ols",
            "z_lookback": 0,
            "block_extreme": False,
            "calm_online": False,
            "b1": prod_b1,
            "coverage": _coverage_gate(pass_tbl, "prod_p95"),
        }
    ]
    for name, b2_mode, block_ext, online in VARIANTS:
        band = _VARIANT_BAND[name]
        row = {
            "variant": name,
            "band": band,
            "b2": b2_mode,
            "z_lookback": 0,
            "block_extreme": block_ext,
            "calm_online": online,
            "b1": _pool_b1(sig_df, name),
            "coverage": _coverage_gate(pass_tbl, band),
        }
        b1 = row["b1"]
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
        ablation.append(row)

    summary = {
        "n_codes_scanned": len(codes),
        "n_codes_ok": int(sig_df["code"].nunique()) if not sig_df.empty else 0,
        "gate": gate_rule,
        "replace_threshold": "coverage share >= 0.80 AND B1 fwd20 mean+hit >= prod",
        "prod_b1": prod_b1,
        "s1_prod": _pool_b1(sig_df, "s1_prod"),
        "s1_vs": _pool_b1(sig_df, "s1_vs"),
        "ablation": ablation,
    }
    (out_dir / "ablation_gates.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # Compact per-variant table.
    lines = [
        "variant            band                 b2      alpha  B1_n  fwd20     hit20   cov_share  replace"
    ]
    for a in ablation:
        b1 = a["b1"]
        lines.append(
            f"{a['variant']:<18} {a['band']:<20} {a['b2']:<7} "
            f"{str(a.get('calm_online', False)):<5} "
            f"{b1.get('n_sig', 0):<5} "
            f"{b1.get('fwd_mean_20d', float('nan')):+.4f}  "
            f"{b1.get('fwd_hit_20d', float('nan')):.3f}   "
            f"{a['coverage'].get('share', float('nan')):.3f}      "
            f"{'OK' if a.get('replace_ok') else '-'}"
        )
    (out_dir / "ablation_report.md").write_text(
        "# Fair-path band ablation\n\n```\n" + "\n".join(lines) + "\n```\n",
        encoding="utf-8",
    )
    print("\n".join(lines))
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
