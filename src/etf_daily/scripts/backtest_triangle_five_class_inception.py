#!/usr/bin/env python3
"""Inception-to-date five-class OPS backtest.

Recomputes regime-transition triangles on each ETF's **full Qlib history** via a
frozen walk-forward HMM (see ``regime_transition_walkforward.py``), replays the
existing triangle decision engine, tags every decision with the five-class
``ops_bucket`` used by the live MD report, and simulates buy/sell execution.

Accuracy is quantified **only for buy/sell classes** (win rate / edge). Trend /
range classes get occurrence counts + forward-20d return/MAE distributions as
diagnostics.

Outputs under ``<out-dir>/inception_five_class_backtest/``:
  events.csv  zones.csv  per_code_summary.csv  summary.csv  REPORT.md
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from backtest_triangle_decision_rules import (  # noqa: E402
    RuleConfig,
    _strength_rank,
    annotate_switches,
    build_price_frame,
    detect_alternating_zones,
    forward_stats,
    simulate_buy_confirm,
    simulate_buy_watch,
    simulate_sell,
    top_zone_post_red_run,
)
from etf_daily.lib.regime_transition_plot import (  # noqa: E402
    compute_volatility_regime_frame,
)
from etf_daily.lib.regime_transition_walkforward import (  # noqa: E402
    generate_walkforward_triangles,
)
from etf_daily.lib.symbol_trend_board import (  # noqa: E402
    load_cluster_mapping_codes,
)
from etf_daily.lib.triangle_ops_resolve import (  # noqa: E402
    OPS_BUCKET_LABEL,
    SignalGateConfig,
    ops_bucket,
)
from etf_daily.paths import CLUSTER_MAPPING_SELECTED_TXT, PLOTLY_OUTPUTS_DIR  # noqa: E402

BUCKET_LABELS = OPS_BUCKET_LABEL

INCEPTION_DEFAULT = "2005-01-01"


@dataclass(frozen=True)
class GateConfig(SignalGateConfig):
    """Backtest alias of the shared live signal gates (see triangle_ops_resolve).

    Baseline = all events; gated = events passing these filters. The report
    compares both on train/test segments to check the gates generalize.
    """


_GATE_DIAGNOSTIC_ACTIONS = {"高位回砸观察"}


def apply_gates(events: pd.DataFrame, gate: GateConfig) -> pd.DataFrame:
    """Flag each event row ``gated`` (passes quality gates) + ``gate_reason``."""
    if events.empty:
        return events.assign(gated=pd.Series(dtype=bool), gate_reason=pd.Series(dtype=str))
    out = events.copy()
    max_rank = _strength_rank(gate.sell_max_strength)
    gated: list[bool] = []
    reasons: list[str] = []
    for _, r in out.iterrows():
        action = str(r.get("action") or "")
        if action in _GATE_DIAGNOSTIC_ACTIONS:
            gated.append(True)
            reasons.append("")
            continue
        if action == "卖预警":
            ok = _strength_rank(str(r.get("strength") or "")) <= max_rank
            gated.append(ok)
            reasons.append("" if ok else f"强度>{gate.sell_max_strength}")
        elif action == "买观察":
            vp = r.get("vol_pct")
            ok = pd.notna(vp) and float(vp) >= gate.buy_watch_min_vol
            gated.append(ok)
            reasons.append("" if ok else f"vol_pct<{gate.buy_watch_min_vol:.2f}")
        elif action == "顶区失效-收复买观察":
            p10 = r.get("prior10")
            deep = pd.notna(p10) and float(p10) < gate.post_top_prior_deep
            pos = gate.post_top_allow_positive and pd.notna(p10) and float(p10) > 0
            ok = bool(deep or pos)
            gated.append(ok)
            reasons.append("" if ok else "prior10中段(-8%~0%)")
        else:
            gated.append(True)
            reasons.append("")
    out["gated"] = gated
    out["gate_reason"] = reasons
    return out


def _exec_stats(g: pd.DataFrame) -> dict[str, Any]:
    ex = g[g["executed"] == True]  # noqa: E712
    if ex.empty:
        return {"n_exec": 0, "win_rate": np.nan, "mean_edge": np.nan, "median_edge": np.nan}
    return {
        "n_exec": int(len(ex)),
        "win_rate": float((ex["edge"] > 0).mean()),
        "mean_edge": float(ex["edge"].mean()),
        "median_edge": float(ex["edge"].median()),
    }


def segment_stats(events: pd.DataFrame, split_date: str) -> pd.DataFrame:
    """Baseline vs gated stats per action, split into train/test by signal_date."""
    if events.empty:
        return pd.DataFrame()
    work = events.copy()
    split = pd.Timestamp(split_date)
    work["segment"] = np.where(
        pd.to_datetime(work["signal_date"]) < split, "train", "test"
    )
    rows: list[dict[str, Any]] = []
    for (segment, action), g in work.groupby(["segment", "action"]):
        base = _exec_stats(g)
        rows.append({"segment": segment, "gate": "baseline", "action": action, **base})
        gd = g[g["gated"] == True]  # noqa: E712
        rows.append({"segment": segment, "gate": "gated", "action": action, **(_exec_stats(gd) | {"n_signals": int(len(gd))})})
    return pd.DataFrame(rows).sort_values(["action", "segment", "gate"])


def _load_plot_mod():
    path = _PROJECT_ROOT / "decision_pack" / "scripts" / "plot_regime_transition_example.py"
    spec = importlib.util.spec_from_file_location("plot_regime_example_inception", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _px_from_full_history(ohlcv: pd.DataFrame, plot_mod: Any) -> pd.DataFrame:
    """Build price frame with MA/vol over the full loaded history."""
    ohlcv = plot_mod.attach_sma_columns(ohlcv)
    close = ohlcv.set_index(pd.to_datetime(ohlcv["datetime"]).dt.normalize())["$close"]
    close = close[~close.index.duplicated(keep="last")]
    vol = compute_volatility_regime_frame(close)
    return build_price_frame(ohlcv, vol)


def _bucket_row(decision: str, above_ma20: Any) -> tuple[str, str]:
    trend_intact = None
    if above_ma20 is True or (isinstance(above_ma20, (bool, np.bool_)) and bool(above_ma20)):
        trend_intact = True
    elif above_ma20 is False:
        trend_intact = False
    bucket = ops_bucket(decision, trend_intact=trend_intact)
    return bucket, BUCKET_LABELS.get(bucket, bucket)


def backtest_code(
    code: str,
    plot_mod: Any,
    cfg: RuleConfig,
    *,
    start_date: str | None,
    end_date: str,
    refit_days: int = 63,
    min_warmup_days: int = 260,
) -> dict[str, Any]:
    code_u = str(code).upper()
    start = start_date or INCEPTION_DEFAULT
    ohlcv = plot_mod.load_qlib_ohlcv(
        code_u,
        start,
        end_date,
        init_qlib=False,
        lookback_calendar_days=0,
        clip_to_window=False,
    )
    if ohlcv.empty:
        return {"code": code_u, "error": "no ohlcv"}

    px = _px_from_full_history(ohlcv, plot_mod)
    px = px[px["as_of"] <= pd.Timestamp(end_date)].copy()
    if px.empty:
        return {"code": code_u, "error": "no px frame"}

    close = px.set_index("as_of")["$close"]
    tri = generate_walkforward_triangles(
        close, refit_days=refit_days, min_warmup_days=min_warmup_days
    )
    if tri.empty:
        return {
            "code": code_u,
            "inception": str(px["as_of"].min().date()),
            "n_days": int(len(px)),
            "n_triangles": 0,
            "n_events": 0,
            "n_zones": 0,
            "action_counts": {},
            "events": [],
            "zones": [],
            "annotated": pd.DataFrame(),
        }

    sw = tri.copy()
    sw["code"] = code_u
    sw = sw.merge(
        px[
            [
                "as_of",
                "$close",
                "$high",
                "$low",
                "MA5",
                "MA20",
                "rv5",
                "rv20",
                "vol5_pct_120d",
                "vol_regime",
            ]
        ],
        on="as_of",
        how="left",
    )
    ann = annotate_switches(sw, px, cfg)
    zones = detect_alternating_zones(ann, cfg)

    events: list[dict[str, Any]] = []

    def _ev(action: str, side: str, s: pd.Series, sim: dict[str, Any], extra: dict[str, Any]):
        bucket, bucket_label = _bucket_row(action, s.get("above_ma20"))
        events.append(
            {
                "code": code_u,
                "signal_date": str(pd.Timestamp(s["as_of"]).date()),
                "side": side,
                "action": action,
                "bucket": bucket,
                "bucket_label": bucket_label,
                "strength": s.get("strength", ""),
                "signal_px": float(s["close"]),
                "prior10": s.get("prior10"),
                "vol_pct": s.get("vol5_pct_120d"),
                **sim,
                **extra,
            }
        )

    for _, s in ann.iterrows():
        action = s["action"]
        base_d = pd.Timestamp(s["as_of"])
        if action == "卖预警":
            sim = simulate_sell(px, s, cfg)
            if sim["executed"]:
                st = forward_stats(px, sim["exec_date"], sim["exec_px"], cfg.hold_eval_days)
                edge = -st["ret"] if pd.notna(st["ret"]) else np.nan
            else:
                st = forward_stats(px, base_d, float(s["close"]), cfg.hold_eval_days)
                edge = np.nan
            _ev(
                action,
                "绿",
                s,
                {
                    "executed": bool(sim["executed"]),
                    "exec_date": None if sim["exec_date"] is None else str(pd.Timestamp(sim["exec_date"]).date()),
                    "exec_px": sim["exec_px"],
                    "stop": None,
                    "reason": sim["reason"],
                    "fwd_ret": st["ret"],
                    "fwd_mfe": st["mfe"],
                    "fwd_mae": st["mae"],
                    "edge": edge,
                },
                {},
            )
        elif action == "买观察":
            sim = simulate_buy_watch(px, s, cfg)
            if sim["executed"]:
                fut = px[px["as_of"] >= sim["exec_date"]].head(cfg.hold_eval_days)
                exit_px = float(fut.iloc[-1]["$close"]) if not fut.empty else sim["exec_px"]
                exit_why = "到期"
                for _, r in fut.iterrows():
                    if sim["stop"] is not None and float(r["$low"]) <= float(sim["stop"]):
                        exit_px = float(sim["stop"])
                        exit_why = "止损"
                        break
                ret = exit_px / float(sim["exec_px"]) - 1.0
                st = forward_stats(px, sim["exec_date"], float(sim["exec_px"]), cfg.hold_eval_days)
                _ev(
                    action,
                    "红",
                    s,
                    {
                        "executed": True,
                        "exec_date": str(pd.Timestamp(sim["exec_date"]).date()),
                        "exec_px": sim["exec_px"],
                        "stop": sim["stop"],
                        "reason": f"{sim['reason']};{exit_why}",
                        "fwd_ret": ret,
                        "fwd_mfe": st["mfe"],
                        "fwd_mae": st["mae"],
                        "edge": ret,
                    },
                    {},
                )
            else:
                st = forward_stats(px, base_d, float(s["close"]), cfg.hold_eval_days)
                _ev(
                    action,
                    "红",
                    s,
                    {
                        "executed": False,
                        "exec_date": None,
                        "exec_px": None,
                        "stop": sim["stop"],
                        "reason": sim["reason"],
                        "fwd_ret": st["ret"],
                        "fwd_mfe": st["mfe"],
                        "fwd_mae": st["mae"],
                        "edge": np.nan,
                    },
                    {},
                )
        elif action == "买确认候选":
            sim = simulate_buy_confirm(px, s)
            if not sim["executed"]:
                continue
            fut = px[px["as_of"] >= sim["exec_date"]].head(cfg.hold_eval_days)
            exit_px = float(fut.iloc[-1]["$close"]) if not fut.empty else sim["exec_px"]
            exit_why = "到期"
            for _, r in fut.iterrows():
                if sim["stop"] is not None and float(r["$low"]) <= float(sim["stop"]):
                    exit_px = float(sim["stop"])
                    exit_why = "止损"
                    break
            ret = exit_px / float(sim["exec_px"]) - 1.0
            st = forward_stats(px, sim["exec_date"], float(sim["exec_px"]), cfg.hold_eval_days)
            _ev(
                "买确认候选",
                "绿",
                s,
                {
                    "executed": True,
                    "exec_date": str(pd.Timestamp(sim["exec_date"]).date()),
                    "exec_px": sim["exec_px"],
                    "stop": sim["stop"],
                    "reason": f"{sim['reason']};{exit_why}",
                    "fwd_ret": ret,
                    "fwd_mfe": st["mfe"],
                    "fwd_mae": st["mae"],
                    "edge": ret,
                },
                {},
            )
        elif action == "高位回砸观察":
            st = forward_stats(px, base_d, float(s["close"]), 10)
            _ev(
                action,
                "红",
                s,
                {
                    "executed": False,
                    "exec_date": None,
                    "exec_px": None,
                    "stop": None,
                    "reason": "高位红三角，仅诊断",
                    "fwd_ret": st["ret"],
                    "fwd_mfe": st["mfe"],
                    "fwd_mae": st["mae"],
                    "edge": -st["ret"] if pd.notna(st["ret"]) else np.nan,
                },
                {},
            )

    # Post-top-zone reclaim-high buy watch (top superseded by red run below MA20).
    if not zones.empty and not ann.empty:
        for _, z in zones.iterrows():
            if z["kind"] != "顶部交替区":
                continue
            zone_end = pd.Timestamp(z["zone_end"])
            tail = ann[ann["as_of"] > zone_end]
            if tail.empty:
                continue
            # evaluate at each post-zone triangle day; take first supersede hit
            for _, t in tail.iterrows():
                ok, _why, last_red = top_zone_post_red_run(
                    ann, zone_end=zone_end, as_of=pd.Timestamp(t["as_of"])
                )
                if ok and last_red is not None:
                    sim = simulate_buy_watch(px, last_red, cfg)
                    st = forward_stats(px, pd.Timestamp(last_red["as_of"]), float(last_red["close"]), cfg.hold_eval_days)
                    if sim["executed"]:
                        fut = px[px["as_of"] >= sim["exec_date"]].head(cfg.hold_eval_days)
                        exit_px = float(fut.iloc[-1]["$close"]) if not fut.empty else sim["exec_px"]
                        for _, r in fut.iterrows():
                            if sim["stop"] is not None and float(r["$low"]) <= float(sim["stop"]):
                                exit_px = float(sim["stop"])
                                break
                        ret = exit_px / float(sim["exec_px"]) - 1.0
                    else:
                        ret = np.nan
                    _ev(
                        "顶区失效-收复买观察",
                        "红",
                        last_red,
                        {
                            "executed": bool(sim["executed"]),
                            "exec_date": None if sim["exec_date"] is None else str(pd.Timestamp(sim["exec_date"]).date()),
                            "exec_px": sim["exec_px"],
                            "stop": sim.get("stop"),
                            "reason": "顶部区失效后收复红三角高点",
                            "fwd_ret": ret,
                            "fwd_mfe": st["mfe"],
                            "fwd_mae": st["mae"],
                            "edge": ret if sim["executed"] else np.nan,
                        },
                        {},
                    )
                    break

    # Zones evaluation (same as existing backtest for top avoided drawdown).
    zone_evals: list[dict[str, Any]] = []
    for _, z in zones.iterrows():
        if z["kind"] != "顶部交替区":
            st = forward_stats(px, pd.Timestamp(z["zone_end"]), float(z["last_close"]), 20)
            zone_evals.append(
                {
                    "code": code_u,
                    **{k: (str(v.date()) if hasattr(v, "date") else v) for k, v in z.items()},
                    "fwd20_ret": st["ret"],
                    "fwd20_mae": st["mae"],
                }
            )
            continue
        legs = ann[
            (ann["as_of"] >= z["zone_start"])
            & (ann["as_of"] <= z["zone_end"])
            & (ann["side"] == "down")
        ]
        if legs.empty:
            exit_d = pd.Timestamp(z["zone_end"])
            exit_px = float(z["last_close"])
        else:
            exit_d = pd.Timestamp(legs.iloc[0]["as_of"])
            exit_px = float(legs.iloc[0]["close"])
        st = forward_stats(px, exit_d, exit_px, 20)
        zone_evals.append(
            {
                "code": code_u,
                **{k: (str(v.date()) if hasattr(v, "date") else v) for k, v in z.items()},
                "exit_date": str(exit_d.date()),
                "exit_px": exit_px,
                "fwd20_ret": st["ret"],
                "fwd20_mae": st["mae"],
                "avoided_drawdown": -st["mae"] if pd.notna(st["mae"]) else np.nan,
            }
        )

    return {
        "code": code_u,
        "inception": str(px["as_of"].min().date()),
        "n_days": int(len(px)),
        "n_triangles": int(len(ann)),
        "n_events": int(len(events)),
        "n_zones": int(len(zones)),
        "action_counts": ann["action"].value_counts().to_dict() if not ann.empty else {},
        "events": events,
        "zones": zone_evals,
        "annotated": ann,
    }


def summarize_by_action(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    for action, g in events.groupby("action"):
        ex = g[g["executed"] == True]  # noqa: E712
        rows.append(
            {
                "action": action,
                "n_signals": int(len(g)),
                "n_executed": int(len(ex)),
                "exec_rate": float(len(ex) / len(g)) if len(g) else np.nan,
                "win_rate": float((ex["edge"] > 0).mean()) if len(ex) else np.nan,
                "mean_edge": float(ex["edge"].mean()) if len(ex) else np.nan,
                "median_edge": float(ex["edge"].median()) if len(ex) else np.nan,
                "mean_fwd_ret_all": float(g["fwd_ret"].mean()),
                "mean_mae_all": float(g["fwd_mae"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("n_signals", ascending=False)


def summarize_by_bucket(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    for bucket, g in events.groupby("bucket"):
        ex = g[g["executed"] == True]  # noqa: E712
        rows.append(
            {
                "bucket": bucket,
                "bucket_label": g["bucket_label"].iloc[0],
                "n_signals": int(len(g)),
                "n_executed": int(len(ex)),
                "exec_rate": float(len(ex) / len(g)) if len(g) else np.nan,
                "win_rate": float((ex["edge"] > 0).mean()) if len(ex) else np.nan,
                "mean_edge": float(ex["edge"].mean()) if len(ex) else np.nan,
                "median_edge": float(ex["edge"].median()) if len(ex) else np.nan,
                "mean_fwd20_ret": float(g["fwd_ret"].mean()),
                "mean_fwd20_mae": float(g["fwd_mae"].mean()),
                "mean_fwd20_mfe": float(g["fwd_mfe"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("n_signals", ascending=False)


def write_report(
    out_dir: Path,
    meta: dict[str, Any],
    by_action: pd.DataFrame,
    by_bucket: pd.DataFrame,
    zones_df: pd.DataFrame,
    seg_stats: pd.DataFrame | None = None,
    gate: GateConfig | None = None,
) -> None:
    L: list[str] = []
    L.append("# Inception 5-class OPS backtest")
    L.append("")
    L.append(f"- window: `{meta['start']}` → `{meta['end']}` (per code: from inception)")
    L.append(f"- codes ok/fail: {meta['n_ok']}/{meta['n_failed']}")
    L.append(f"- triangles: **synthetic walk-forward HMM** (q*=0.90, refit every {meta['refit_days']}d, warmup {meta['min_warmup_days']}d)")
    L.append(f"- causal caveat: each day's triangle uses model fit only on data ≤ that day; frozen between refits.")
    L.append("")

    L.append("## Five-class buy/sell accuracy")
    L.append("")
    if not by_bucket.empty:
        L.append("| 五类 | signals | exec | exec_rate | win_rate | mean_edge | median_edge |")
        L.append("|---|---|---|---|---|---|---|")
        for _, r in by_bucket.iterrows():
            L.append(
                f"| {r['bucket_label']} | {int(r['n_signals'])} | {int(r['n_executed'])} | "
                f"{r['exec_rate']:.1%} | {r['win_rate']:.1%} | {r['mean_edge']:+.2%} | {r['median_edge']:+.2%} |"
            )
    else:
        L.append("(no events)")
    L.append("")

    L.append("## Buy/sell sub-actions")
    L.append("")
    if not by_action.empty:
        L.append("| action | signals | exec | win_rate | mean_edge | median_edge |")
        L.append("|---|---|---|---|---|---|")
        for _, r in by_action.iterrows():
            if int(r["n_executed"]) == 0:
                L.append(
                    f"| {r['action']} | {int(r['n_signals'])} | 0 | — | — | — |"
                )
            else:
                L.append(
                    f"| {r['action']} | {int(r['n_signals'])} | {int(r['n_executed'])} | "
                    f"{r['win_rate']:.1%} | {r['mean_edge']:+.2%} | {r['median_edge']:+.2%} |"
                )
    else:
        L.append("(no events)")
    L.append("")

    # Sell sub-class: top zone avoided drawdown
    top = zones_df[zones_df["kind"] == "顶部交替区"] if not zones_df.empty else pd.DataFrame()
    if not top.empty:
        L.append("## 顶部交替区 sell diagnostics")
        L.append("")
        L.append(f"- n top zones: {len(top)}")
        if "avoided_drawdown" in top:
            ad = pd.to_numeric(top["avoided_drawdown"], errors="coerce").dropna()
            if len(ad):
                L.append(f"- avoided drawdown (mean/median): {ad.mean():.2%} / {ad.median():.2%}")
                L.append(f"- zones avoiding ≥3% drawdown: {(ad >= 0.03).mean():.1%}")
        L.append("")

    L.append("## Trend/range diagnostics (forward 20d)")
    L.append("")
    if not by_bucket.empty:
        L.append("| 五类 | fwd20_ret | fwd20_mae | fwd20_mfe |")
        L.append("|---|---|---|---|")
        for _, r in by_bucket.iterrows():
            L.append(
                f"| {r['bucket_label']} | {r['mean_fwd20_ret']:+.2%} | "
                f"{r['mean_fwd20_mae']:+.2%} | {r['mean_fwd20_mfe']:+.2%} |"
            )
    else:
        L.append("(no events)")
    L.append("")

    if seg_stats is not None and not seg_stats.empty and gate is not None:
        L.append("## Gated filter validation (train/test)")
        L.append("")
        L.append(
            f"Gates: 卖预警≤{gate.sell_max_strength}强度；买观察 vol_pct≥{gate.buy_watch_min_vol:.2f}；"
            f"顶区收复 prior10<{gate.post_top_prior_deep:.0%} 或 >0%。"
            f"Split: train < {meta.get('split_date')} ≤ test。"
        )
        L.append("")
        L.append("| action | segment | gate | n_exec | win_rate | mean_edge | median_edge |")
        L.append("|---|---|---|---|---|---|---|")
        for _, r in seg_stats.iterrows():
            if int(r["n_exec"]) == 0:
                L.append(
                    f"| {r['action']} | {r['segment']} | {r['gate']} | 0 | — | — | — |"
                )
            else:
                L.append(
                    f"| {r['action']} | {r['segment']} | {r['gate']} | {int(r['n_exec'])} | "
                    f"{r['win_rate']:.1%} | {r['mean_edge']:+.2%} | {r['median_edge']:+.2%} |"
                )
        L.append("")

    L.append("## Notes")
    L.append("")
    L.append("- Buy/sell accuracy uses executed trades only (edge = trade return).")
    L.append("- Trend/range classes are diagnostic only; no trade simulation.")
    L.append("- Synthetic triangles may differ slightly from production `signals_oos` in overlap windows.")
    L.append("")
    (out_dir / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


def _codes_from_html(html_dir: Path) -> list[str]:
    """Derive codes from ``regime_transition_<CODE>_<name>_<start>_<end>.html``."""
    import re

    if not html_dir.exists():
        return []
    codes: list[str] = []
    for p in html_dir.glob("*.html"):
        m = re.match(r"regime_transition_((?:SH|SZ)\d{6})_", p.name)
        if m:
            codes.append(m.group(1).upper())
    return sorted(set(codes))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--codes-from-html", type=Path, default=None,
                   help="derive code pool from html filenames in dir")
    p.add_argument("--code", action="append", default=None)
    p.add_argument("--start-date", default=None,
                   help="override inception (default: first Qlib bar per code)")
    p.add_argument("--end-date", default="2026-07-28")
    p.add_argument("--refit-days", type=int, default=63)
    p.add_argument("--min-warmup-days", type=int, default=260)
    p.add_argument("--split-date", default="2024-01-01",
                   help="train/test split for gated filter validation")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.code:
        codes = [c.upper() for c in args.code]
    elif args.codes_from_html:
        codes = _codes_from_html(args.codes_from_html)
    else:
        codes = [c.upper() for c in load_cluster_mapping_codes(CLUSTER_MAPPING_SELECTED_TXT)]
    if not codes:
        print("[ERROR] no codes", file=sys.stderr)
        return 2

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = _PROJECT_ROOT / "runtime" / "backtests"
    if not out_dir.is_absolute():
        out_dir = _PROJECT_ROOT / out_dir
    out_dir = out_dir / "inception_five_class_backtest"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = RuleConfig()
    plot_mod = _load_plot_mod()
    plot_mod._init_qlib(None)

    all_events: list[pd.DataFrame] = []
    all_zones: list[pd.DataFrame] = []
    per_code: list[dict[str, Any]] = []
    failed: list[tuple[str, str]] = []

    print(f"[INFO] inception 5-class backtest codes={len(codes)} end={args.end_date} refit={args.refit_days}")
    for code in codes:
        try:
            res = backtest_code(
                code,
                plot_mod,
                cfg,
                start_date=args.start_date,
                end_date=args.end_date,
                refit_days=args.refit_days,
                min_warmup_days=args.min_warmup_days,
            )
            if res.get("error"):
                raise ValueError(res["error"])
            per_code.append(
                {
                    "code": res["code"],
                    "inception": res["inception"],
                    "n_days": res["n_days"],
                    "n_triangles": res["n_triangles"],
                    "n_events": res["n_events"],
                    "n_zones": res["n_zones"],
                    **{f"act_{k}": v for k, v in res["action_counts"].items()},
                }
            )
            if res["events"]:
                all_events.append(pd.DataFrame(res["events"]))
            if res["zones"]:
                all_zones.append(pd.DataFrame(res["zones"]))
            print(f"[OK] {code} days={res['n_days']} tri={res['n_triangles']} ev={res['n_events']} zones={res['n_zones']}")
        except Exception as exc:  # noqa: BLE001
            failed.append((code, str(exc)))
            print(f"[ERROR] {code}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                return 2

    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    zones_df = pd.concat(all_zones, ignore_index=True) if all_zones else pd.DataFrame()
    per_code_df = pd.DataFrame(per_code)

    gate = GateConfig()
    events_df = apply_gates(events_df, gate)
    seg_stats = segment_stats(events_df, args.split_date)

    by_action = summarize_by_action(events_df)
    by_bucket = summarize_by_bucket(events_df)
    summary = pd.concat(
        [
            by_action.assign(group="action").rename(columns={"action": "name"}),
            by_bucket.assign(group="bucket").rename(columns={"bucket_label": "name"}),
        ],
        ignore_index=True,
        sort=False,
    )

    events_df.to_csv(out_dir / "events.csv", index=False)
    zones_df.to_csv(out_dir / "zones.csv", index=False)
    per_code_df.to_csv(out_dir / "per_code_summary.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    seg_stats.to_csv(out_dir / "gated_segment_stats.csv", index=False)

    meta = {
        "start": args.start_date or "inception",
        "end": args.end_date,
        "n_codes": len(codes),
        "n_ok": len(per_code),
        "n_failed": len(failed),
        "failed": failed,
        "refit_days": args.refit_days,
        "min_warmup_days": args.min_warmup_days,
        "split_date": args.split_date,
        "rule": asdict(cfg),
        "gate": asdict(gate),
    }
    (out_dir / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir, meta, by_action, by_bucket, zones_df, seg_stats, gate)
    print(f"[INFO] wrote → {out_dir}")
    return 0 if not failed or args.continue_on_error else 2


if __name__ == "__main__":
    raise SystemExit(main())
