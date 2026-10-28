#!/usr/bin/env python3
"""H1–H6 read-only diagnosis of fig7–11 on one from-listing directory.

Writes CSV + report.md under
``$PLOTLY_ROOT/research/fair_path_fat_tail_audit/``. Does not change
production rails or signals.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from etf_daily.lib.fair_path_band_signals import (  # noqa: E402
    find_regime_html,
    load_ohlcv_from_regime_html,
)
from etf_daily.lib.fair_path_research import ewma_scaled_envelope  # noqa: E402
from etf_daily.lib.fair_path_tail_audit import (  # noqa: E402
    PROTOCOL,
    breach_rates,
    evaluate_gate,
    forward_return,
    g_sign_lift,
    insample_and_exante_residual,
    log_residual,
)
from etf_daily.lib.fat_tail_risk import fit_tail  # noqa: E402
from etf_daily.paths import PLOTLY_OUTPUTS_DIR  # noqa: E402
from etf_daily.plots.plot_regime_transition_example import (  # noqa: E402
    _trail_fit_min_bars,
    erp_path_symmetric_envelope,
    trailing_log_trend_price_path,
)

LISTING = PLOTLY_OUTPUTS_DIR / "20261019_from_listing"
OUT = PLOTLY_OUTPUTS_DIR / "research" / "fair_path_fat_tail_audit"
PANELS = (
    ("fig7", 2.0),
    ("fig8", 1.0),
    ("fig9", 0.5),
    ("fig10", 0.25),
    ("fig11", 1.0 / 12.0),
)


def _codes() -> list[str]:
    summary = LISTING / "batch_summary.csv"
    df = pd.read_csv(summary, encoding="utf-8-sig")
    return [str(c).upper() for c in df["code"].tolist()]


def _clusters() -> dict[str, str]:
    path = Path("/home/huangtuo/temp/decision_packs/20261019/cluster_representatives_20261019.csv")
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    code_col = cols.get("code") or cols.get("symbol")
    cluster_col = cols.get("cluster") or cols.get("cluster_id") or cols.get("group")
    if code_col is None or cluster_col is None:
        return {}
    return {
        str(r[code_col]).upper(): str(r[cluster_col])
        for _, r in df.iterrows()
        if pd.notna(r[code_col])
    }


def _ratio_ci(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    """Block bootstrap CI for (mean left) / (mean right)."""
    m = np.isfinite(left) & np.isfinite(right)
    a = left[m].astype(float)
    b = right[m].astype(float)
    n = int(a.size)
    if n < 40 or float(b.mean()) <= 0:
        return float("nan"), float("nan"), float("nan")
    point = float(a.mean() / b.mean())
    block = int(PROTOCOL["block_len"])
    n_boot = int(PROTOCOL["n_boot"])
    gen = np.random.default_rng(int(PROTOCOL["boot_seed"]))
    n_blocks = int(np.ceil(n / block))
    starts = gen.integers(0, n, size=(n_boot, n_blocks))
    offs = np.arange(block)
    idx = ((starts[:, :, None] + offs[None, None, :]) % n).reshape(n_boot, n_blocks * block)[:, :n]
    denom = b[idx].mean(axis=1)
    numer = a[idx].mean(axis=1)
    stats = np.divide(numer, denom, out=np.full(n_boot, np.nan), where=denom > 0)
    finite = stats[np.isfinite(stats)]
    if finite.size < 20:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(finite, 0.05)), float(np.quantile(finite, 0.95))


def _dist(vals: list[float]) -> dict[str, float]:
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0.0, "median": float("nan"), "worst5pct_mean": float("nan"), "mad": float("nan")}
    med = float(np.median(x))
    q = float(np.quantile(x, 0.05))
    tail = x[x <= q]
    return {
        "n": float(x.size),
        "median": med,
        "worst5pct_mean": float(np.mean(tail)) if tail.size else float(x.min()),
        "mad": float(np.median(np.abs(x - med))),
    }


def _stratum(n: int) -> str:
    if n < 504:
        return "lt_2y"
    if n < 740:
        return "lt_740"
    return "ge_740"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clusters = _clusters()
    h1_rows: list[dict[str, Any]] = []
    h3_rows: list[dict[str, Any]] = []
    h4_rows: list[dict[str, Any]] = []
    h6_rows: list[dict[str, Any]] = []
    h2: dict[tuple[str, str], list[float]] = {}
    b1_by_code: dict[str, pd.Series] = {}
    errors: list[str] = []

    for code in _codes():
        html = find_regime_html(LISTING, code)
        if html is None:
            errors.append(f"{code}: no html")
            continue
        try:
            close, _ohlcv = load_ohlcv_from_regime_html(html)
        except Exception as exc:  # noqa: BLE001 — record and continue the pool
            errors.append(f"{code}: {exc}")
            continue
        n = int(close.size)
        stratum = _stratum(n)
        touch_lo: dict[str, pd.Series] = {}
        gap8 = None
        for key, years in PANELS:
            min_bars = _trail_fit_min_bars(years)
            trail_bars = int(round(years * 252))
            path, g = trailing_log_trend_price_path(
                close, trail_years=years, min_bars=min_bars
            )
            upper, lower, _ = erp_path_symmetric_envelope(path, close, q=0.95, min_bars=min_bars)
            resid = log_residual(close, path)
            half = np.full(n, np.nan)
            p = path.to_numpy(dtype=float)
            u = upper.to_numpy(dtype=float)
            ok_w = np.isfinite(p) & (p > 0) & np.isfinite(u)
            half[ok_w] = np.log(u[ok_w] / p[ok_w])
            rates = breach_rates(resid, half_width=half)
            c = close.to_numpy(dtype=float)
            lo = lower.to_numpy(dtype=float)
            left_ind = np.zeros(n, dtype=float)
            right_ind = np.zeros(n, dtype=float)
            valid = np.isfinite(resid) & np.isfinite(half) & np.isfinite(lo) & np.isfinite(u)
            left_ind[valid] = (c[valid] < lo[valid]).astype(float)
            right_ind[valid] = (c[valid] > u[valid]).astype(float)
            # Only bootstrap days that have a rail, so the rate matches `rates`.
            point, lo_ci, hi_ci = _ratio_ci(left_ind[valid], right_ind[valid])
            excludes = bool(np.isfinite(lo_ci) and np.isfinite(hi_ci) and (hi_ci < 1.0 or lo_ci > 1.0))
            h1_rows.append(
                {
                    "code": code,
                    "panel": key,
                    "stratum": stratum,
                    "n": rates["n"],
                    "left_rate": rates["left_rate"],
                    "right_rate": rates["right_rate"],
                    "ratio": point,
                    "ratio_lo": lo_ci,
                    "ratio_hi": hi_ci,
                    "ci_excludes_1": excludes,
                    "left_dominant": bool(excludes and np.isfinite(point) and point > 1.0),
                }
            )
            if years >= 0.25 - 1e-9:
                e_up, e_lo, _ = ewma_scaled_envelope(path, close, min_bars=min_bars)
                px = close.to_numpy(dtype=float)
                p95_lo = lower.to_numpy(dtype=float)
                ew_lo = e_lo.to_numpy(dtype=float)
                both = (px <= p95_lo) & np.isfinite(p95_lo) & (px <= ew_lo) & np.isfinite(ew_lo)
                ewma_only = (px <= ew_lo) & np.isfinite(ew_lo) & ~((px <= p95_lo) & np.isfinite(p95_lo))
                p95_only = (px <= p95_lo) & np.isfinite(p95_lo) & ~((px <= ew_lo) & np.isfinite(ew_lo))
                fwd = forward_return(close, 20).to_numpy(dtype=float)
                for label, mask in (("both", both), ("ewma_only", ewma_only), ("p95_only", p95_only)):
                    h2.setdefault((key, label), []).extend(fwd[mask & np.isfinite(fwd)].tolist())

            rin, rex = insample_and_exante_residual(
                close, trail_bars=trail_bars, min_bars=min_bars
            )
            both_r = np.isfinite(rin) & np.isfinite(rex)
            med_ratio = (
                float(np.median(np.abs(rin[both_r]) / np.maximum(np.abs(rex[both_r]), 1e-12)))
                if int(both_r.sum())
                else float("nan")
            )
            # Touch using in-sample residual vs the published half-width, then
            # again with the ex-ante residual and the same width.
            touch_in = both_r & (rin <= -half) & np.isfinite(half)
            touch_ex = both_r & (rex <= -half) & np.isfinite(half)
            touched = touch_in | touch_ex
            flips = int(np.sum(touch_in != touch_ex))
            n_touch = int(touched.sum())
            h3_rows.append(
                {
                    "code": code,
                    "panel": key,
                    "stratum": stratum,
                    "median_abs_in_over_ex": med_ratio,
                    "n_touch": n_touch,
                    "flip_rate": (flips / n_touch) if n_touch else float("nan"),
                }
            )
            if key in {"fig9", "fig10"}:
                touch_lo[key] = pd.Series(px_le(close, lower), index=close.index)
            if key == "fig8":
                gap8 = close / path - 1.0

            fit_l = fit_tail(resid[np.isfinite(resid)], side="left", tail_quantile=0.90, min_exceed=25)
            fit_r = fit_tail(resid[np.isfinite(resid)], side="right", tail_quantile=0.90, min_exceed=25)
            if key == "fig8":
                h4_rows.append(
                    {
                        "code": code,
                        "cluster": clusters.get(code, ""),
                        "n": n,
                        "stratum": stratum,
                        "n_exceed_left": fit_l.n_exceed,
                        "n_exceed_right": fit_r.n_exceed,
                        "alpha_left": fit_l.alpha,
                        "alpha_right": fit_r.alpha,
                        "left_reliable": fit_l.reliable,
                        "low_exceed": fit_l.n_exceed < int(PROTOCOL["min_exceed"]),
                    }
                )
            lift = g_sign_lift(
                g,
                forward_return(close, 20),
                block=int(PROTOCOL["block_len"]),
                n_boot=100,
                seed=int(PROTOCOL["boot_seed"]),
            )
            h6_rows.append({"code": code, "panel": key, "stratum": stratum, **lift})

        if gap8 is not None and touch_lo:
            b1 = pd.Series(False, index=close.index)
            for ser in touch_lo.values():
                b1 = b1 | ser.fillna(False)
            b1 = b1 | (gap8 <= float(PROTOCOL["deep_gap"])).fillna(False)
            b1_by_code[code] = b1
        print(f"[diag] {code} n={n}", flush=True)

    h1 = pd.DataFrame(h1_rows)
    h3 = pd.DataFrame(h3_rows)
    h4 = pd.DataFrame(h4_rows)
    h6 = pd.DataFrame(h6_rows)
    h1.to_csv(OUT / "h1_breach.csv", index=False)
    h3.to_csv(OUT / "h3_flip.csv", index=False)
    h4.to_csv(OUT / "h4_alpha.csv", index=False)
    h6.to_csv(OUT / "h6_g_lift.csv", index=False)

    h2_rows = []
    for (panel, label), vals in sorted(h2.items()):
        h2_rows.append({"panel": panel, "class": label, **_dist(vals)})
    h2_df = pd.DataFrame(h2_rows)
    h2_df.to_csv(OUT / "h2_ewma.csv", index=False)

    if b1_by_code:
        b1_df = pd.DataFrame(b1_by_code).fillna(False)
        daily = b1_df.sum(axis=1)
        daily.to_csv(OUT / "h5_daily_b1.csv", header=["n_b1"])
    else:
        daily = pd.Series(dtype=float)

    summary = _summarize(h1, h2_df, h3, h4, h6, daily)
    gate = evaluate_gate(summary)
    (OUT / "gate.json").write_text(
        json.dumps({"summary": summary, "gate": gate, "errors": errors}, indent=2, default=str),
        encoding="utf-8",
    )
    (OUT / "report.md").write_text(_report(summary, gate, errors), encoding="utf-8")
    print(json.dumps(gate, indent=2))
    print(f"[OK] {OUT / 'report.md'}")


def px_le(close: pd.Series, lower: pd.Series) -> np.ndarray:
    c = close.to_numpy(dtype=float)
    lo = lower.to_numpy(dtype=float)
    return (c <= lo) & np.isfinite(lo)


def _summarize(h1, h2, h3, h4, h6, daily) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if len(h1):
        out["h1_left_dominant_frac"] = float(h1["left_dominant"].mean())
        out["h1_ci_excludes_1_frac"] = float(h1["ci_excludes_1"].mean())
        out["h1_n_panels"] = int(len(h1))
    # H2 on fig9, the panel the checklist actually uses for touches.
    def _row(panel: str, label: str) -> dict[str, float] | None:
        if h2.empty:
            return None
        hit = h2[(h2["panel"] == panel) & (h2["class"] == label)]
        if hit.empty:
            return None
        return hit.iloc[0].to_dict()

    both = _row("fig9", "both")
    only = _row("fig9", "ewma_only")
    if both is None or only is None or both["n"] < PROTOCOL["h2_min_events"] or only["n"] < PROTOCOL["h2_min_events"]:
        out["h2_inconclusive"] = True
        out["h2_ewma_only_not_better"] = False
    else:
        out["h2_inconclusive"] = False
        better_med = float(only["median"]) > float(both["median"])
        better_tail = float(only["worst5pct_mean"]) > float(both["worst5pct_mean"])
        out["h2_ewma_only_not_better"] = not (better_med and better_tail)
        out["h2_both"] = {"n": both["n"], "median": both["median"], "worst5pct_mean": both["worst5pct_mean"]}
        out["h2_ewma_only"] = {"n": only["n"], "median": only["median"], "worst5pct_mean": only["worst5pct_mean"]}
    if len(h3):
        short = h3[h3["panel"].isin(["fig10", "fig11"])]
        # Pool touch-weighted flip rate per panel.
        flips = []
        for panel, g in short.groupby("panel"):
            w = g["n_touch"].to_numpy(dtype=float)
            r = g["flip_rate"].to_numpy(dtype=float)
            m = np.isfinite(r) & (w > 0)
            rate = float(np.sum(r[m] * w[m]) / np.sum(w[m])) if m.any() and np.sum(w[m]) else float("nan")
            flips.append(rate)
            out[f"h3_flip_{panel}"] = rate
        out["h3_short_window_flip"] = any(
            np.isfinite(v) and v > float(PROTOCOL["h3_flip_min"]) for v in flips
        )
    if len(h4):
        out["h4_low_exceed_frac"] = float(h4["low_exceed"].mean())
        out["h4_n"] = int(len(h4))
        pool_possible = False
        if h4["cluster"].astype(str).str.len().gt(0).any():
            for _, g in h4.groupby("cluster"):
                if len(g) >= 2 and float(g["n_exceed_left"].sum()) >= float(PROTOCOL["min_exceed"]):
                    pool_possible = True
                    break
        out["h4_pool_possible"] = pool_possible
    if len(daily):
        out["h5_max_b1"] = int(daily.max())
        out["h5_mean_b1"] = float(daily.mean())
        out["h5_days_with_b1"] = int((daily > 0).sum())
        out["h5_top_dates"] = [
            str(pd.Timestamp(i).date())
            for i in daily.sort_values(ascending=False).head(8).index
        ]
    if len(h6):
        # Fraction of codes whose lift CI excludes 0, by panel.
        for panel, g in h6.groupby("panel"):
            lo = g["lo"].to_numpy(dtype=float)
            hi = g["hi"].to_numpy(dtype=float)
            excludes = np.isfinite(lo) & np.isfinite(hi) & ((hi < 0) | (lo > 0))
            out[f"h6_{panel}_excludes0_frac"] = float(np.mean(excludes)) if len(g) else float("nan")
            out[f"h6_{panel}_median_lift"] = float(np.nanmedian(g["lift"]))
    return out


def _report(summary: dict[str, Any], gate: dict[str, Any], errors: list[str]) -> str:
    lines = [
        "# 图7–11 肥尾诊断（只读）",
        "",
        f"动作：`{gate.get('action')}`。H1={gate.get('h1')} H2={gate.get('h2')} "
        f"(证据不足={gate.get('h2_inconclusive')}) H3={gate.get('h3')} "
        f"改轨道={gate.get('change_rails')} 用POT={gate.get('use_pot')}。",
        "",
        "## 汇总",
        "",
        "```",
        json.dumps(summary, indent=2, default=str),
        "```",
        "",
    ]
    if errors:
        lines.append(f"跳过 {len(errors)} 只：")
        lines.extend(f"- {e}" for e in errors[:20])
    lines.append("")
    lines.append("明细：`h1_breach.csv` `h2_ewma.csv` `h3_flip.csv` `h4_alpha.csv` `h5_daily_b1.csv` `h6_g_lift.csv`。")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
