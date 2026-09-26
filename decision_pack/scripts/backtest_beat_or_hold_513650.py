#!/usr/bin/env python3
"""Beat-or-hold SH513650: only leave the anchor when something looks stronger.

User goal
---------
Always holding 标普500ETF南方 (SH513650) is the default. A rotation only
happens when, at the signal date, we can point to names that are *currently
stronger than 513650* on a relative rule. Otherwise stay 100% in 513650.

Variants
--------
anchor_only
    Buy-and-hold SH513650 on the same T+1 windows (benchmark).
switch_mom5 / switch_mom10 / switch_mom20
    If any eligible name has momH > momH(513650): hold equal-weight Top-K of those;
    else 100% 513650.
switch_resid5
    Same, but compare momH − momH(513650) > 0 (equivalent to momH > momH(513650)).
switch_up_mom5
    Eligible = adaptive regime==up + same-day q05 not worst quartile;
    then same switch_mom5 rule. Match/Omega is diagnostic only.
blend_mom5
    Stronger names get weight; 513650 always keeps residual weight so never fully off
    unless K stronger names fill gross=1.

Execution: signal close T → hold T+1 close to next_signal+1 close, H=20.
Research only.
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

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from runtime_paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from decision_pack.src.adaptive_stage_common import prepare_features  # noqa: E402
from decision_pack.src.garch_dynamics import GarchDynamicsConfig  # noqa: E402
from decision_pack.src.garch_short_horizon_board import GarchShortHorizonConfig  # noqa: E402
from decision_pack.src.vol_need_return_board import (  # noqa: E402
    TARGET_SHARPE_LIKE,
    forecast_garch_ann_vols,
)
from decision_pack.src.vol_return_board import (  # noqa: E402
    DEFAULT_DEFENSIVE_CODES,
    load_fund_name_map,
    slice_close_through,
)

ANCHOR = "SH513650"
SEGMENTS = (
    ("2024H1", "2024-01-01", "2024-06-30"),
    ("2024H2", "2024-07-01", "2024-12-31"),
    ("2025H1", "2025-01-01", "2025-06-30"),
    ("2025H2", "2025-07-01", "2025-12-31"),
    ("2026YTD", "2026-01-01", "2026-12-31"),
)


def _load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--as-of", default="2026-08-10")
    p.add_argument("--start", default="2024-01-02")
    p.add_argument(
        "--adaptive-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260810all_adaptive",
    )
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--sharpe-target", type=float, default=TARGET_SHARPE_LIKE)
    p.add_argument("--mc-samples", type=int, default=1000)
    p.add_argument("--min-history", type=int, default=252)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def load_universe(adaptive_dir: Path) -> tuple[list[str], dict[str, str]]:
    summary = pd.read_csv(adaptive_dir / "batch_summary.csv")
    name_map: dict[str, str] = {}
    codes: list[str] = []
    for _, r in summary.iterrows():
        code = str(r["code"]).upper()
        if code in {str(x).upper() for x in DEFAULT_DEFENSIVE_CODES}:
            continue
        codes.append(code)
        if "name" in summary.columns:
            name_map[code] = str(r["name"])
    if ANCHOR not in codes:
        codes.append(ANCHOR)
    name_map.setdefault(ANCHOR, "标普500ETF南方")
    fund_list = _ROOT / "temp/fund_list.csv"
    if fund_list.is_file():
        name_map = {**name_map, **load_fund_name_map(fund_list)}
    return sorted(set(codes)), name_map


def picks_codes_to_names(cell: str, name_map: dict[str, str]) -> str:
    if cell is None or (isinstance(cell, float) and not np.isfinite(cell)):
        return ""
    text = str(cell).strip()
    if not text:
        return ""
    parts: list[str] = []
    for tok in text.split("|"):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            code, w = tok.split(":", 1)
            code = code.upper()
            parts.append(f"{name_map.get(code, code)}:{w}")
        else:
            code = tok.upper()
            parts.append(name_map.get(code, code))
    return "|".join(parts)

def load_symbol_config(adaptive_dir: Path, code: str) -> dict[str, Any]:
    path = adaptive_dir / "configs" / f"{code}.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def next_session(calendar: pd.DatetimeIndex, t: pd.Timestamp) -> pd.Timestamp | None:
    t = pd.Timestamp(t).normalize()
    idx = int(calendar.searchsorted(t))
    if idx < len(calendar) and calendar[idx] == t:
        nxt = idx + 1
    else:
        nxt = idx
    if nxt >= len(calendar):
        return None
    return pd.Timestamp(calendar[nxt])


def period_return(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    s = series.dropna()
    s = s[(s.index >= start) & (s.index <= end)]
    if len(s) < 2:
        return None
    p0, p1 = float(s.iloc[0]), float(s.iloc[-1])
    if p0 <= 0 or p1 <= 0 or not np.isfinite(p0) or not np.isfinite(p1):
        return None
    return p1 / p0 - 1.0


def weighted_period_return(
    panel: pd.DataFrame,
    weights: dict[str, float],
    entry: pd.Timestamp,
    exit_: pd.Timestamp,
) -> float:
    if not weights:
        return 0.0
    total = 0.0
    for code, w in weights.items():
        if w <= 0 or code not in panel.columns:
            continue
        r = period_return(panel[code], entry, exit_)
        if r is None:
            continue
        total += float(w) * float(r)
    return float(total)


def mom_at(ser: pd.Series, as_of: pd.Timestamp, bars: int) -> float:
    s = ser.dropna().loc[:as_of]
    if len(s) < bars + 1:
        return float("nan")
    return float(s.iloc[-1] / s.iloc[-(bars + 1)] - 1.0)


PUBLISHED_BASELINE_DIR = PLOTLY_OUTPUTS_DIR / "20260810_beat_or_hold_513650"
UNGATED_STRATEGIES = (
    "anchor_only",
    "switch_mom5",
    "switch_mom10",
    "switch_mom20",
    "blend_mom5",
)

# collect_tickets (once per T)
#     rows: garch_reliable & finite q05; match may be NaN
#     |
#     +-- apply_old_veto -> (old_set, floor_old)  # finite match; match>=1 & q05>=p25
#     +-- apply_new_veto -> (new_set, floor_new)  # all finite q05; q05>=p25
# allow = {regime==up} ∩ new_set ∪ {ANCHOR}


def published_baseline_dir() -> Path:
    return PUBLISHED_BASELINE_DIR


def resolve_out_dir(out_dir: Path | None, *, as_of: str | pd.Timestamp, root: Path = _ROOT) -> Path:
    tag = pd.Timestamp(as_of).strftime("%Y%m%d")
    if root == _ROOT:
        published = (PLOTLY_OUTPUTS_DIR / "20260810_beat_or_hold_513650").resolve()
        default_parent = PLOTLY_OUTPUTS_DIR
    else:
        published = (root / "workspace/plotly_outputs/20260810_beat_or_hold_513650").resolve()
        default_parent = root / "workspace/plotly_outputs"
    if out_dir is None:
        resolved = default_parent / f"{tag}_beat_or_hold_513650_q05floor"
    else:
        resolved = Path(out_dir) if Path(out_dir).is_absolute() else root / out_dir
    resolved = resolved.resolve()
    if resolved == published:
        raise SystemExit(2)
    return resolved


def ticket_row_from_forecast(code: str, g: dict[str, Any], *, horizon: int) -> dict[str, Any] | None:
    if not g.get("garch_reliable"):
        return None
    prefix = f"{horizon}d_garch"
    q = float(g.get(f"q05_{prefix}", float("nan")))
    if not np.isfinite(q):
        return None
    m = float(g.get(f"match_{prefix}", float("nan")))
    return {"code": str(code), "q05": q, "match": m if np.isfinite(m) else float("nan")}


def collect_tickets(
    panel: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    horizon: int,
    cfg: GarchShortHorizonConfig,
    sharpe_like: float,
    min_history: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for code in panel.columns:
        ser = slice_close_through(panel[code], as_of)
        if ser.dropna().shape[0] < min_history:
            continue
        try:
            g = forecast_garch_ann_vols(
                ser, as_of, cfg=cfg, include_mc=True, sharpe_like=sharpe_like
            )
        except Exception:  # noqa: BLE001
            continue
        row = ticket_row_from_forecast(str(code), g, horizon=horizon)
        if row is not None:
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["code", "q05", "match"])
    return pd.DataFrame(rows)


def apply_old_veto(df: pd.DataFrame) -> tuple[set[str], float]:
    if df is None or df.empty:
        return set(), float("nan")
    sub = df.loc[np.isfinite(df["match"].to_numpy(dtype=float))].copy()
    if sub.empty:
        return set(), float("nan")
    floor = float(sub["q05"].quantile(0.25))
    if not np.isfinite(floor):
        return set(), float("nan")
    ok = sub.loc[(sub["match"] >= 1.0) & (sub["q05"] >= floor), "code"]
    return set(ok.astype(str).tolist()), floor


def apply_new_veto(df: pd.DataFrame) -> tuple[set[str], float]:
    if df is None or df.empty:
        return set(), float("nan")
    sub = df.loc[np.isfinite(df["q05"].to_numpy(dtype=float))].copy()
    if sub.empty:
        return set(), float("nan")
    floor = float(sub["q05"].quantile(0.25))
    if not np.isfinite(floor):
        return set(), float("nan")
    ok = sub.loc[sub["q05"] >= floor, "code"]
    return set(ok.astype(str).tolist()), floor


def compare_ungated(
    published: pd.DataFrame,
    new: pd.DataFrame,
    *,
    atol: float = 1e-6,
    strategies: tuple[str, ...] = UNGATED_STRATEGIES,
) -> list[str]:
    mismatches: list[str] = []
    pub = published.set_index("strategy") if "strategy" in published.columns else published
    nxt = new.set_index("strategy") if "strategy" in new.columns else new
    for name in strategies:
        if name not in pub.index or name not in nxt.index:
            mismatches.append(name)
            continue
        a = float(pub.loc[name, "cum_ret"])
        b = float(nxt.loc[name, "cum_ret"])
        if not (np.isfinite(a) and np.isfinite(b) and abs(a - b) <= float(atol)):
            mismatches.append(name)
    return mismatches


def build_mom_table(
    panel: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    bars: int,
    min_history: int,
    allow: set[str] | None = None,
) -> pd.DataFrame:
    rows = []
    for code in panel.columns:
        code = str(code)
        if allow is not None and code not in allow and code != ANCHOR:
            continue
        ser = slice_close_through(panel[code], as_of)
        if ser.dropna().shape[0] < min_history:
            continue
        m = mom_at(panel[code], as_of, bars)
        if not np.isfinite(m):
            continue
        rows.append({"code": code, "mom": m})
    return pd.DataFrame(rows)


def weights_switch(
    mom_df: pd.DataFrame,
    *,
    k: int,
    anchor: str = ANCHOR,
) -> tuple[dict[str, float], str]:
    """If any non-anchor mom > anchor mom → EW Top-K of beaters; else 100% anchor."""
    if mom_df.empty or anchor not in set(mom_df["code"]):
        return {anchor: 1.0}, "fallback_anchor"
    a_mom = float(mom_df.loc[mom_df["code"] == anchor, "mom"].iloc[0])
    beaters = mom_df[(mom_df["code"] != anchor) & (mom_df["mom"] > a_mom)].copy()
    if beaters.empty:
        return {anchor: 1.0}, "hold_anchor"
    beaters = beaters.sort_values("mom", ascending=False, kind="mergesort").head(int(k))
    w = 1.0 / float(len(beaters))
    return {str(c): w for c in beaters["code"].tolist()}, "rotate"


def weights_blend(
    mom_df: pd.DataFrame,
    *,
    k: int,
    anchor: str = ANCHOR,
) -> tuple[dict[str, float], str]:
    """Keep anchor; fill remaining weight with stronger names (EW among beaters)."""
    if mom_df.empty or anchor not in set(mom_df["code"]):
        return {anchor: 1.0}, "fallback_anchor"
    a_mom = float(mom_df.loc[mom_df["code"] == anchor, "mom"].iloc[0])
    beaters = mom_df[(mom_df["code"] != anchor) & (mom_df["mom"] > a_mom)].copy()
    if beaters.empty:
        return {anchor: 1.0}, "hold_anchor"
    beaters = beaters.sort_values("mom", ascending=False, kind="mergesort").head(int(k))
    n = len(beaters)
    # split: half anchor floor + half to beaters, or proportional
    # User wants beat SPX — give beaters 60%, anchor 40% when rotating
    rot = 0.60
    w = {anchor: 1.0 - rot}
    each = rot / float(n)
    for c in beaters["code"].tolist():
        w[str(c)] = each
    return w, "blend"


def summarize(x: pd.Series, *, horizon: int, label: str) -> dict[str, Any]:
    r = x.dropna().astype(float)
    if r.empty:
        return {"strategy": label, "n_periods": 0}
    ppy = 252.0 / float(horizon)
    mu, sig = float(r.mean()), float(r.std(ddof=1)) if len(r) >= 2 else float("nan")
    sharpe = mu / sig * np.sqrt(ppy) if np.isfinite(sig) and sig > 0 else float("nan")
    eq = (1.0 + r.fillna(0.0)).cumprod()
    return {
        "strategy": label,
        "n_periods": int(len(r)),
        "cum_ret": float(eq.iloc[-1] - 1.0),
        "sharpe_ann": float(sharpe),
        "hit_rate": float((r > 0).mean()),
        "max_dd": float((eq / eq.cummax() - 1.0).min()),
    }


def info_ratio(ex: pd.Series, *, horizon: int) -> float:
    ex = ex.dropna().astype(float)
    if len(ex) < 2 or float(ex.std(ddof=1)) <= 0:
        return float("nan")
    return float(ex.mean() / ex.std(ddof=1) * np.sqrt(252.0 / float(horizon)))


def main() -> int:
    args = parse_args()
    adaptive_dir = (
        args.adaptive_dir if args.adaptive_dir.is_absolute() else _ROOT / args.adaptive_dir
    )
    try:
        out_dir = resolve_out_dir(args.out_dir, as_of=args.as_of)
    except SystemExit as exc:
        print(
            f"[ERROR] refusing to overwrite published baseline {published_baseline_dir()}",
            file=sys.stderr,
        )
        return int(exc.code) if isinstance(exc.code, int) else 2
    out_dir.mkdir(parents=True, exist_ok=True)

    codes, name_map = load_universe(adaptive_dir)
    print(f"universe={len(codes)} anchor={ANCHOR} target_sharpe={args.sharpe_target}")

    pm = _load_mod("pm_bh", _ROOT / "decision_pack/scripts/plot_regime_transition_example.py")
    ev = _load_mod("ev_bh", _ROOT / "decision_pack/scripts/eval_causal_regime_switch_pool.py")
    pm._init_qlib(None)
    load_start = (
        pd.Timestamp(args.start) - pd.Timedelta(days=int(args.min_history * 2 + 120))
    ).strftime("%Y-%m-%d")
    end = str(pd.Timestamp(args.as_of).date())

    closes: dict[str, pd.Series] = {}
    regimes: dict[str, pd.Series] = {}
    for i, code in enumerate(codes, start=1):
        try:
            ohlcv = pm.load_qlib_ohlcv(
                code, load_start, end, init_qlib=False, lookback_calendar_days=220, clip_to_window=False
            )
            if ohlcv is None or ohlcv.empty:
                continue
            close_col = "close" if "close" in ohlcv.columns else "$close"
            close = pd.to_numeric(ohlcv[close_col], errors="coerce")
            close.index = pd.to_datetime(ohlcv["datetime" if "datetime" in ohlcv.columns else "as_of"])
            closes[code] = close[~close.index.duplicated(keep="last")].sort_index()
            cfg = load_symbol_config(adaptive_dir, code)
            if cfg.get("method"):
                px = prepare_features(
                    ev,
                    ohlcv,
                    method=str(cfg["method"]),
                    method_params=cfg.get("method_params") or {},
                    sticky_guard=False,
                )
                if "regime" in px.columns:
                    reg = px.set_index("as_of")["regime"].astype(str)
                    reg.index = pd.to_datetime(reg.index).normalize()
                    regimes[code] = reg[~reg.index.duplicated(keep="last")].sort_index()
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {code}: {exc}")
        if i % 15 == 0 or i == len(codes):
            print(f"loaded {i}/{len(codes)}")

    panel = pd.DataFrame(closes).sort_index()
    if ANCHOR not in panel.columns:
        print(f"[ERROR] missing anchor {ANCHOR}", file=sys.stderr)
        return 2
    calendar = panel.dropna(how="all").index
    cal = calendar[(calendar >= pd.Timestamp(args.start)) & (calendar <= pd.Timestamp(args.as_of))]
    signals = list(cal[:: int(args.horizon)])
    if len(signals) < 2:
        return 2

    tcfg = GarchShortHorizonConfig(garch=GarchDynamicsConfig(mc_samples=int(args.mc_samples)))
    h, k = int(args.horizon), int(args.k)

    # (name, mom_bars, mode, use_up_ticket)
    # mode: switch | blend
    specs = [
        ("anchor_only", 5, "anchor", False),
        ("switch_mom5", 5, "switch", False),
        ("switch_mom10", 10, "switch", False),
        ("switch_mom20", 20, "switch", False),
        ("switch_up_mom5", 5, "switch", True),
        ("switch_up_mom10", 10, "switch", True),
        ("blend_mom5", 5, "blend", False),
        ("blend_up_mom5", 5, "blend", True),
    ]

    period_rows: list[dict[str, Any]] = []
    pick_rows: list[dict[str, Any]] = []
    allow_diff_rows: list[dict[str, Any]] = []

    for i, sig in enumerate(signals[:-1]):
        entry = next_session(calendar, sig)
        exit_ = next_session(calendar, signals[i + 1])
        if entry is None or exit_ is None or entry >= exit_:
            continue

        tickets = collect_tickets(
            panel,
            pd.Timestamp(sig),
            horizon=h,
            cfg=tcfg,
            sharpe_like=float(args.sharpe_target),
            min_history=int(args.min_history),
        )
        old_veto, floor_old = apply_old_veto(tickets)
        new_veto, floor_new = apply_new_veto(tickets)
        up_now: set[str] = set()
        for code, reg in regimes.items():
            hist = reg.loc[: pd.Timestamp(sig)].dropna()
            if not hist.empty and str(hist.iloc[-1]).lower() == "up":
                up_now.add(code)
        for rec in tickets.to_dict("records"):
            code = str(rec["code"])
            allow_diff_rows.append(
                {
                    "signal": sig.strftime("%Y-%m-%d"),
                    "code": code,
                    "in_old_veto": int(code in old_veto),
                    "in_new_veto": int(code in new_veto),
                    "regime_up": int(code in up_now),
                    "q05": rec["q05"],
                    "match": rec["match"],
                    "q05_floor_old": floor_old,
                    "q05_floor_new": floor_new,
                }
            )

        row: dict[str, Any] = {
            "signal": sig.strftime("%Y-%m-%d"),
            "entry": entry.strftime("%Y-%m-%d"),
            "exit": exit_.strftime("%Y-%m-%d"),
        }
        pick_row: dict[str, Any] = {"signal": row["signal"]}

        for name, bars, mode, use_gate in specs:
            allow = None
            if use_gate:
                allow = set(up_now)
                allow &= new_veto
                allow.add(ANCHOR)

            mom_df = build_mom_table(
                panel,
                pd.Timestamp(sig),
                bars=bars,
                min_history=int(args.min_history),
                allow=allow,
            )
            if mode == "anchor":
                weights, reason = {ANCHOR: 1.0}, "anchor_only"
            elif mode == "blend":
                weights, reason = weights_blend(mom_df, k=k)
            else:
                weights, reason = weights_switch(mom_df, k=k)

            ret = weighted_period_return(panel, weights, entry, exit_)
            row[name] = ret
            row[f"{name}__rotated"] = int(reason in ("rotate", "blend"))
            row[f"{name}__n"] = len(weights)
            pick_row[name] = "|".join(f"{c}:{weights[c]:.2f}" for c in weights)
            pick_row[f"{name}__reason"] = reason

        period_rows.append(row)
        pick_rows.append(pick_row)
        print(
            f"[{i+1}/{len(signals)-1}] {sig.date()} "
            f"anchor={row['anchor_only']*100:+.2f}% "
            f"sw5={row['switch_mom5']*100:+.2f}% "
            f"up5={row['switch_up_mom5']*100:+.2f}% "
            f"blend5={row['blend_mom5']*100:+.2f}%"
        )

    periods = pd.DataFrame(period_rows)
    if periods.empty:
        return 3

    summary_rows = []
    for name, *_ in specs:
        s = summarize(periods[name], horizon=h, label=name)
        s["cum_vs_anchor"] = float(s["cum_ret"] - summarize(periods["anchor_only"], horizon=h, label="a")["cum_ret"])
        s["info_ratio_vs_anchor"] = info_ratio(
            periods[name] - periods["anchor_only"], horizon=h
        )
        s["rotate_rate"] = float(periods[f"{name}__rotated"].mean()) if name != "anchor_only" else 0.0
        s["beat_sharpe_target"] = bool(s["sharpe_ann"] >= float(args.sharpe_target))
        s["beat_anchor_cum"] = bool(s["cum_vs_anchor"] > 0)
        summary_rows.append(s)
    summary = pd.DataFrame(summary_rows).sort_values(
        "info_ratio_vs_anchor", ascending=False, kind="mergesort"
    )

    seg_rows = []
    for seg_name, lo, hi in SEGMENTS:
        sub = periods[(periods["signal"] >= lo) & (periods["signal"] <= hi)]
        if sub.empty:
            continue
        for name, *_ in specs:
            ex = sub[name] - sub["anchor_only"]
            seg_rows.append(
                {
                    "segment": seg_name,
                    "strategy": name,
                    "n": len(sub),
                    "cum_ret": float((1 + sub[name]).prod() - 1),
                    "cum_anchor": float((1 + sub["anchor_only"]).prod() - 1),
                    "cum_vs_anchor": float((1 + sub[name]).prod() - (1 + sub["anchor_only"]).prod()),
                    "info_ratio_vs_anchor": info_ratio(ex, horizon=h),
                    "sharpe_ann": summarize(sub[name], horizon=h, label=name)["sharpe_ann"],
                }
            )
    seg = pd.DataFrame(seg_rows)

    published_summary_path = published_baseline_dir() / "summary.csv"
    if not published_summary_path.is_file():
        print(f"[ERROR] missing published summary {published_summary_path}", file=sys.stderr)
        return 2
    published_summary = pd.read_csv(published_summary_path)
    drifted = compare_ungated(published_summary, summary)
    if drifted:
        print(
            f"[ERROR] ungated strategies drifted vs published: {drifted}",
            file=sys.stderr,
        )
        return 2

    periods.to_csv(out_dir / "periods.csv", index=False, float_format="%.6f")
    if allow_diff_rows:
        pd.DataFrame(allow_diff_rows).to_csv(
            out_dir / "allow_diff.csv", index=False, float_format="%.6f"
        )
    picks_df = pd.DataFrame(pick_rows)
    if "switch_mom5" in picks_df.columns:
        zh = picks_df["switch_mom5"].map(lambda x: picks_codes_to_names(x, name_map))
        idx = list(picks_df.columns).index("switch_mom5")
        picks_df.insert(idx, "switch_mom5_name", zh)
    picks_df.to_csv(out_dir / "picks.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False, float_format="%.6f")
    seg.to_csv(out_dir / "segment_ir.csv", index=False, float_format="%.6f")
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "anchor": ANCHOR,
                "target_sharpe": float(args.sharpe_target),
                "rule": "hold 513650 unless other names have higher momH at signal; then rotate/blend",
                "execution": "T+1",
                "goal": "cum > anchor AND sharpe >= 1.33",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n=== summary vs SH513650 (target Sharpe>=1.33) ===")
    cols = [
        "strategy",
        "cum_ret",
        "cum_vs_anchor",
        "sharpe_ann",
        "beat_sharpe_target",
        "info_ratio_vs_anchor",
        "rotate_rate",
        "max_dd",
        "hit_rate",
    ]
    print(summary[cols].to_string(index=False))
    ok = summary[summary["beat_anchor_cum"] & summary["beat_sharpe_target"]]
    print("\nMeet BOTH beat-anchor-cum AND Sharpe>=1.33:")
    if ok.empty:
        print("  (none)")
    else:
        print(ok[cols].to_string(index=False))
    print(f"\n[OK] out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
