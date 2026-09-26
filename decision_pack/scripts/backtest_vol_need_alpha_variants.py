#!/usr/bin/env python3
"""Alpha variants on regime+mom (no extra ticket ranking weights).

Baseline B from A/B:
  regime==up + ticket veto (match>=1, q05>=p25) + mom rank Top-K, T+1, cash OK.

Variants (ticket only as veto — not used in score mix)
----------------------------------------------------
B_mom20_ew / B_mom10_ew / B_mom5_ew
    equal-weight Top-K by raw mom{H}
B_resid20_ew / B_resid10_ew / B_resid5_ew
    equal-weight Top-K by residual mom = mom − cross-section median (eligible pool)
B_mom20_str / B_resid10_str / B_resid5_str
    same ranking, but weights ∝ strength; gross exposure scales with strength → cash

Strength (per name, then portfolio)
-----------------------------------
days_in_up = consecutive up bars ending at signal
name_score = max(rank_signal, 0) * (1 + log1p(days_in_up))
gross = clip(median(days_in_up)/15, 0.25, 1.0)   # weak up → under-invest
weights sum to gross (remainder cash)

Also reports IR vs EW by calendar segment for stability.
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
    slice_close_through,
)

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
    name_map = {
        str(r["code"]).upper(): str(r["name"]) if "name" in summary.columns else ""
        for _, r in summary.iterrows()
    }
    codes = [
        c
        for c in name_map
        if c not in {str(x).upper() for x in DEFAULT_DEFENSIVE_CODES}
    ]
    return sorted(codes), name_map


def load_symbol_config(adaptive_dir: Path, code: str) -> dict[str, Any]:
    return json.loads((adaptive_dir / "configs" / f"{code}.json").read_text(encoding="utf-8"))


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
    wsum = 0.0
    for code, w in weights.items():
        if w <= 0 or code not in panel.columns:
            continue
        r = period_return(panel[code], entry, exit_)
        if r is None:
            continue
        total += float(w) * float(r)
        wsum += float(w)
    if wsum <= 0:
        return 0.0
    # weights already include gross; cash earns 0
    return float(total)


def days_in_regime(reg: pd.Series, as_of: pd.Timestamp, label: str = "up") -> int:
    hist = reg.loc[:as_of].dropna().astype(str).str.lower()
    if hist.empty:
        return 0
    n = 0
    for v in hist.iloc[::-1]:
        if v == label:
            n += 1
        else:
            break
    return int(n)


def mom_at(ser: pd.Series, as_of: pd.Timestamp, bars: int) -> float:
    s = ser.dropna().loc[:as_of]
    if len(s) < bars + 1:
        return float("nan")
    return float(s.iloc[-1] / s.iloc[-(bars + 1)] - 1.0)


def ticket_board_at(
    panel: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    horizon: int,
    cfg: GarchShortHorizonConfig,
    sharpe_like: float,
    min_history: int,
) -> pd.DataFrame:
    prefix = f"{horizon}d_garch"
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
        if not g.get("garch_reliable"):
            continue
        keys = [f"q05_{prefix}", f"match_{prefix}"]
        if any(not np.isfinite(float(g[k])) for k in keys):
            continue
        rows.append(
            {
                "code": str(code),
                "match": float(g[f"match_{prefix}"]),
                "q05": float(g[f"q05_{prefix}"]),
            }
        )
    return pd.DataFrame(rows)


def eligible_up_frame(
    *,
    as_of: pd.Timestamp,
    panel: pd.DataFrame,
    regimes: dict[str, pd.Series],
    ticket: pd.DataFrame,
    mom_bars: int,
) -> pd.DataFrame:
    if ticket.empty:
        return pd.DataFrame()
    q05_floor = float(ticket["q05"].quantile(0.25))
    veto_ok = set(
        ticket.loc[(ticket["match"] >= 1.0) & (ticket["q05"] >= q05_floor), "code"]
        .astype(str)
        .tolist()
    )
    rows: list[dict[str, Any]] = []
    for code in panel.columns:
        code = str(code)
        if code not in veto_ok:
            continue
        reg = regimes.get(code)
        if reg is None or reg.empty:
            continue
        hist = reg.loc[:as_of].dropna()
        if hist.empty or str(hist.iloc[-1]).lower() != "up":
            continue
        mom = mom_at(panel[code], as_of, mom_bars)
        if not np.isfinite(mom):
            continue
        rows.append(
            {
                "code": code,
                "mom": mom,
                "days_up": days_in_regime(reg, as_of, "up"),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["resid"] = df["mom"] - float(df["mom"].median())
    return df


def picks_to_weights(
    df: pd.DataFrame,
    *,
    k: int,
    signal_col: str,
    strength: bool,
) -> dict[str, float]:
    if df.empty:
        return {}
    ranked = df.sort_values(signal_col, ascending=False, kind="mergesort").head(int(k))
    if ranked.empty:
        return {}
    if not strength:
        w = 1.0 / float(len(ranked))
        return {str(c): w for c in ranked["code"].tolist()}

    # score from non-negative signal rank position + days_up
    sig = ranked[signal_col].astype(float).to_numpy()
    # shift so best stays positive even if all residual negative
    sig_pos = sig - float(np.min(sig)) + 1e-6
    days = ranked["days_up"].astype(float).to_numpy()
    score = sig_pos * (1.0 + np.log1p(np.maximum(days, 0.0)))
    if not np.isfinite(score).any() or float(score.sum()) <= 0:
        w = 1.0 / float(len(ranked))
        return {str(c): w for c in ranked["code"].tolist()}
    med_days = float(np.median(days)) if days.size else 0.0
    gross = float(np.clip(med_days / 15.0, 0.25, 1.0))
    score = score / float(score.sum()) * gross
    return {str(c): float(w) for c, w in zip(ranked["code"].tolist(), score, strict=True)}


def summarize(
    period_rets: pd.Series,
    *,
    horizon: int,
    label: str,
    n_picks: pd.Series | None = None,
    gross: pd.Series | None = None,
) -> dict[str, Any]:
    r = period_rets.dropna().astype(float)
    if r.empty:
        return {"strategy": label, "n_periods": 0}
    ppy = 252.0 / float(horizon)
    mu = float(r.mean())
    sig = float(r.std(ddof=1)) if len(r) >= 2 else float("nan")
    sharpe = mu / sig * np.sqrt(ppy) if np.isfinite(sig) and sig > 0 else float("nan")
    eq = (1.0 + r.fillna(0.0)).cumprod()
    max_dd = float((eq / eq.cummax() - 1.0).min())
    out: dict[str, Any] = {
        "strategy": label,
        "n_periods": int(len(r)),
        "cum_ret": float(eq.iloc[-1] - 1.0),
        "mean_period_ret": mu,
        "sharpe_ann": float(sharpe),
        "hit_rate": float((r > 0).mean()),
        "max_dd": max_dd,
    }
    if n_picks is not None:
        out["avg_n_picks"] = float(n_picks.mean())
        out["cash_rate"] = float((n_picks == 0).mean())
    if gross is not None:
        out["avg_gross"] = float(gross.mean())
    return out


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
    tag = pd.Timestamp(args.as_of).strftime("%Y%m%d")
    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else PLOTLY_OUTPUTS_DIR / f"{tag}_vol_need_alpha_variants"
    )
    out_dir = out_dir if out_dir.is_absolute() else _ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    codes, _name_map = load_universe(adaptive_dir)
    print(f"universe={len(codes)} variants run start={args.start} as_of={args.as_of}")

    pm = _load_mod("pm_av", _ROOT / "decision_pack/scripts/plot_regime_transition_example.py")
    ev = _load_mod("ev_av", _ROOT / "decision_pack/scripts/eval_causal_regime_switch_pool.py")
    pm._init_qlib(None)

    load_start = (
        pd.Timestamp(args.start) - pd.Timedelta(days=int(args.min_history * 2 + 120))
    ).strftime("%Y-%m-%d")
    end = str(pd.Timestamp(args.as_of).date())

    closes: dict[str, pd.Series] = {}
    regimes: dict[str, pd.Series] = {}
    for i, code in enumerate(codes, start=1):
        try:
            cfg = load_symbol_config(adaptive_dir, code)
            ohlcv = pm.load_qlib_ohlcv(
                code,
                load_start,
                end,
                init_qlib=False,
                lookback_calendar_days=220,
                clip_to_window=False,
            )
            if ohlcv is None or ohlcv.empty:
                continue
            close_col = "close" if "close" in ohlcv.columns else "$close"
            close = pd.to_numeric(ohlcv[close_col], errors="coerce")
            if "datetime" in ohlcv.columns:
                close.index = pd.to_datetime(ohlcv["datetime"])
            elif "as_of" in ohlcv.columns:
                close.index = pd.to_datetime(ohlcv["as_of"])
            close = close[~close.index.duplicated(keep="last")].sort_index()
            closes[code] = close
            px = prepare_features(
                ev,
                ohlcv,
                method=str(cfg.get("method") or ""),
                method_params=cfg.get("method_params") or {},
                sticky_guard=False,
            )
            if "regime" not in px.columns:
                continue
            reg = px.set_index("as_of")["regime"].astype(str)
            reg.index = pd.to_datetime(reg.index).normalize()
            regimes[code] = reg[~reg.index.duplicated(keep="last")].sort_index()
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {code}: {exc}")
        if i % 10 == 0 or i == len(codes):
            print(f"loaded {i}/{len(codes)} close={len(closes)} regime={len(regimes)}")

    panel = pd.DataFrame(closes).sort_index()
    calendar = panel.dropna(how="all").index
    cal = calendar[
        (calendar >= pd.Timestamp(args.start)) & (calendar <= pd.Timestamp(args.as_of))
    ]
    signals = list(cal[:: int(args.horizon)])
    if len(signals) < 2:
        print("[ERROR] need >=2 signals", file=sys.stderr)
        return 2

    tcfg = GarchShortHorizonConfig(
        garch=GarchDynamicsConfig(mc_samples=int(args.mc_samples))
    )
    h = int(args.horizon)
    k = int(args.k)

    # variant specs: (name, mom_bars, signal_col, strength)
    variants = [
        ("B_mom20_ew", 20, "mom", False),
        ("B_mom10_ew", 10, "mom", False),
        ("B_mom5_ew", 5, "mom", False),
        ("B_resid20_ew", 20, "resid", False),
        ("B_resid10_ew", 10, "resid", False),
        ("B_resid5_ew", 5, "resid", False),
        ("B_mom20_str", 20, "mom", True),
        ("B_resid10_str", 10, "resid", True),
        ("B_resid5_str", 5, "resid", True),
    ]

    period_rows: list[dict[str, Any]] = []
    pick_rows: list[dict[str, Any]] = []

    for i, sig in enumerate(signals[:-1]):
        sig_next = signals[i + 1]
        entry = next_session(calendar, sig)
        exit_ = next_session(calendar, sig_next)
        if entry is None or exit_ is None or entry >= exit_:
            continue

        ticket = ticket_board_at(
            panel,
            pd.Timestamp(sig),
            horizon=h,
            cfg=tcfg,
            sharpe_like=float(args.sharpe_target),
            min_history=int(args.min_history),
        )
        ew_codes = [
            c
            for c in panel.columns
            if slice_close_through(panel[c], sig).dropna().shape[0] >= int(args.min_history)
        ]
        ew_w = {c: 1.0 / len(ew_codes) for c in ew_codes} if ew_codes else {}
        ew_ret = weighted_period_return(panel, ew_w, entry, exit_)

        row: dict[str, Any] = {
            "signal": sig.strftime("%Y-%m-%d"),
            "entry": entry.strftime("%Y-%m-%d"),
            "exit": exit_.strftime("%Y-%m-%d"),
            "EW_universe": ew_ret,
        }
        pick_row: dict[str, Any] = {"signal": row["signal"]}

        # cache eligible frames by mom_bars
        elig_cache: dict[int, pd.DataFrame] = {}
        for name, mom_bars, signal_col, strength in variants:
            if mom_bars not in elig_cache:
                elig_cache[mom_bars] = eligible_up_frame(
                    as_of=pd.Timestamp(sig),
                    panel=panel,
                    regimes=regimes,
                    ticket=ticket,
                    mom_bars=mom_bars,
                )
            elig = elig_cache[mom_bars]
            weights = picks_to_weights(
                elig, k=k, signal_col=signal_col, strength=strength
            )
            ret = weighted_period_return(panel, weights, entry, exit_)
            row[name] = ret
            row[f"{name}__n"] = len(weights)
            row[f"{name}__gross"] = float(sum(weights.values())) if weights else 0.0
            pick_row[name] = "|".join(weights.keys())
            pick_row[f"{name}__w"] = "|".join(f"{w:.3f}" for w in weights.values())

        period_rows.append(row)
        pick_rows.append(pick_row)
        best = max(variants, key=lambda v: row[v[0]])
        print(
            f"[{i+1}/{len(signals)-1}] {sig.date()} EW={ew_ret*100:+.2f}% "
            f"mom20={row['B_mom20_ew']*100:+.2f}% resid10_str={row['B_resid10_str']*100:+.2f}% "
            f"best={best[0]}:{row[best[0]]*100:+.2f}%"
        )

    periods = pd.DataFrame(period_rows)
    if periods.empty:
        print("[ERROR] no periods", file=sys.stderr)
        return 3

    summary_rows = []
    for name, _, _, _ in variants:
        s = summarize(
            periods[name],
            horizon=h,
            label=name,
            n_picks=periods[f"{name}__n"],
            gross=periods[f"{name}__gross"],
        )
        s["cum_ret_minus_ew"] = float(s["cum_ret"] - summarize(periods["EW_universe"], horizon=h, label="EW")["cum_ret"])
        s["info_ratio_vs_ew"] = info_ratio(periods[name] - periods["EW_universe"], horizon=h)
        summary_rows.append(s)
    ew_s = summarize(periods["EW_universe"], horizon=h, label="EW_universe")
    ew_s["cum_ret_minus_ew"] = 0.0
    ew_s["info_ratio_vs_ew"] = float("nan")
    ew_s["avg_n_picks"] = float("nan")
    ew_s["cash_rate"] = 0.0
    ew_s["avg_gross"] = 1.0
    summary_rows.append(ew_s)
    summary = pd.DataFrame(summary_rows).sort_values(
        "info_ratio_vs_ew", ascending=False, kind="mergesort"
    )

    # segment IR stability
    seg_rows: list[dict[str, Any]] = []
    for seg_name, seg_lo, seg_hi in SEGMENTS:
        mask = (periods["signal"] >= seg_lo) & (periods["signal"] <= seg_hi)
        sub = periods.loc[mask]
        if sub.empty:
            continue
        for name, _, _, _ in variants:
            ex = sub[name] - sub["EW_universe"]
            seg_rows.append(
                {
                    "segment": seg_name,
                    "strategy": name,
                    "n_periods": int(len(sub)),
                    "cum_ret": float((1.0 + sub[name]).prod() - 1.0),
                    "cum_ret_ew": float((1.0 + sub["EW_universe"]).prod() - 1.0),
                    "cum_excess": float((1.0 + sub[name]).prod() - (1.0 + sub["EW_universe"]).prod()),
                    "info_ratio_vs_ew": info_ratio(ex, horizon=h),
                    "hit_rate": float((sub[name] > 0).mean()),
                    "avg_gross": float(sub[f"{name}__gross"].mean()),
                }
            )
    seg = pd.DataFrame(seg_rows)

    # stability score: mean IR across segments with n>=3, penalize negative segments
    if not seg.empty:
        stab = (
            seg.groupby("strategy")
            .agg(
                ir_mean=("info_ratio_vs_ew", "mean"),
                ir_std=("info_ratio_vs_ew", "std"),
                ir_min=("info_ratio_vs_ew", "min"),
                n_pos_seg=("info_ratio_vs_ew", lambda s: int((s > 0).sum())),
                n_seg=("info_ratio_vs_ew", "count"),
            )
            .reset_index()
        )
        stab["stable_score"] = stab["ir_mean"] - 0.5 * stab["ir_std"].fillna(0) + 0.1 * stab["n_pos_seg"]
        summary = summary.merge(stab, how="left", left_on="strategy", right_on="strategy")

    periods.to_csv(out_dir / "periods.csv", index=False, float_format="%.6f")
    pd.DataFrame(pick_rows).to_csv(out_dir / "picks.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False, float_format="%.6f")
    seg.to_csv(out_dir / "segment_ir.csv", index=False, float_format="%.6f")
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "as_of": args.as_of,
                "start": args.start,
                "horizon": h,
                "k": k,
                "execution": "T+1",
                "ticket": "veto only (match>=1, q05>=p25)",
                "variants": [v[0] for v in variants],
                "segments": [s[0] for s in SEGMENTS],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n=== summary (sorted by IR vs EW) ===")
    cols = [
        "strategy",
        "cum_ret",
        "cum_ret_minus_ew",
        "sharpe_ann",
        "info_ratio_vs_ew",
        "max_dd",
        "avg_gross",
        "cash_rate",
        "ir_mean",
        "ir_min",
        "n_pos_seg",
        "stable_score",
    ]
    show = [c for c in cols if c in summary.columns]
    print(summary[show].to_string(index=False))
    print("\n=== segment IR (top strategies) ===")
    if not seg.empty and "stable_score" in summary.columns:
        top = summary.dropna(subset=["stable_score"]).head(3)["strategy"].tolist()
        for st in top:
            print(f"\n-- {st} --")
            print(
                seg[seg.strategy == st][
                    ["segment", "n_periods", "cum_ret", "cum_excess", "info_ratio_vs_ew", "avg_gross"]
                ].to_string(index=False)
            )
    print(f"\n[OK] out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
