#!/usr/bin/env python3
"""Backtest: vol-need ticket-card vote → equal-weight Top-K, rebalance every H days.

Selection (H=20 default, matches chat vote rule)
-----------------------------------------------
1. garch_reliable
2. match_Hd >= 1
3. p_up_Hd >= cross-sectional median
4. q05_Hd >= cross-sectional 25% quantile (drop worst-quartile tails)
5. score = 0.35·rank(match) + 0.25·rank(p_up) + 0.20·rank(p_ge_need) + 0.20·rank(q05)
6. hold Top-K equal weight until next rebalance

Research only — does not mutate production regime / trade rules.
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

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from etf_daily.paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from etf_daily.lib.garch_dynamics import GarchDynamicsConfig  # noqa: E402
from etf_daily.lib.garch_short_horizon_board import GarchShortHorizonConfig  # noqa: E402
from etf_daily.lib.vol_need_return_board import (  # noqa: E402
    TARGET_SHARPE_LIKE,
    forecast_garch_ann_vols,
)
from etf_daily.lib.vol_return_board import (  # noqa: E402
    DEFAULT_DEFENSIVE_CODES,
    slice_close_through,
)

SCORE_W_MATCH = 0.35
SCORE_W_P_UP = 0.25
SCORE_W_P_GE_NEED = 0.20
SCORE_W_Q05 = 0.20


def _load_pm():
    path = _ROOT / "src/etf_daily/plots/plot_regime_transition_example.py"
    spec = importlib.util.spec_from_file_location("pm_vol_need_bt", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--as-of", default="2026-08-10")
    p.add_argument("--start", default="2024-01-02", help="first eligible rebalance date")
    p.add_argument(
        "--adaptive-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260810all_adaptive",
    )
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--horizon", type=int, default=20, help="rebalance / ticket horizon")
    p.add_argument("--sharpe-target", type=float, default=TARGET_SHARPE_LIKE)
    p.add_argument("--mc-samples", type=int, default=1000, help="MC paths per fit (speed)")
    p.add_argument("--min-history", type=int, default=252)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )
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


def load_close_panel(
    codes: list[str],
    *,
    start: str,
    end: str,
    min_history: int,
) -> pd.DataFrame:
    pm = _load_pm()
    pm._init_qlib(None)
    # pad history for GARCH train window
    load_start = (pd.Timestamp(start) - pd.Timedelta(days=int(min_history * 2 + 80))).strftime(
        "%Y-%m-%d"
    )
    frames: dict[str, pd.Series] = {}
    for i, code in enumerate(codes, start=1):
        try:
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
            frames[code] = close
        except Exception:  # noqa: BLE001
            continue
        if i % 20 == 0 or i == len(codes):
            print(f"loaded prices {i}/{len(codes)} ok={len(frames)}")
    if not frames:
        raise RuntimeError("no price series loaded")
    panel = pd.DataFrame(frames).sort_index()
    return panel


def rebalance_dates(
    calendar: pd.DatetimeIndex,
    *,
    start: str,
    end: str,
    horizon: int,
) -> list[pd.Timestamp]:
    cal = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    if cal.empty:
        return []
    dates = list(cal[:: int(horizon)])
    # ensure last eval can form a full period to as_of (drop last incomplete if needed)
    if len(dates) < 2:
        return dates
    return dates


def ticket_row_for_code(
    close: pd.Series,
    as_of: pd.Timestamp,
    *,
    horizon: int,
    cfg: GarchShortHorizonConfig,
    sharpe_like: float,
) -> dict[str, Any] | None:
    g = forecast_garch_ann_vols(
        close,
        as_of,
        cfg=cfg,
        include_mc=True,
        sharpe_like=sharpe_like,
    )
    if not g.get("garch_reliable"):
        return None
    prefix = f"{horizon}d_garch"
    need_keys = (
        f"p_up_{prefix}",
        f"e_up_{prefix}",
        f"e_down_{prefix}",
        f"q05_{prefix}",
        f"match_{prefix}",
        f"p_ge_need_{prefix}",
    )
    if any(not np.isfinite(float(g[k])) for k in need_keys):
        return None
    return {
        "code": None,  # filled by caller
        "garch_reliable": True,
        "p_up": float(g[f"p_up_{prefix}"]),
        "e_up": float(g[f"e_up_{prefix}"]),
        "e_down": float(g[f"e_down_{prefix}"]),
        "q05": float(g[f"q05_{prefix}"]),
        "match": float(g[f"match_{prefix}"]),
        "p_ge_need": float(g[f"p_ge_need_{prefix}"]),
    }


def select_topk(
    board: pd.DataFrame,
    *,
    k: int,
) -> tuple[list[str], pd.DataFrame]:
    """Apply vote screens + composite rank; return codes and scored frame."""
    if board.empty:
        return [], board
    df = board.copy()
    med_up = float(df["p_up"].median())
    q05_floor = float(df["q05"].quantile(0.25))
    screened = df[
        (df["match"] >= 1.0)
        & (df["p_up"] >= med_up)
        & (df["q05"] >= q05_floor)
    ].copy()
    pool = screened if len(screened) >= max(1, k) else df.copy()
    pool["vote_score"] = (
        SCORE_W_MATCH * pool["match"].rank(pct=True)
        + SCORE_W_P_UP * pool["p_up"].rank(pct=True)
        + SCORE_W_P_GE_NEED * pool["p_ge_need"].rank(pct=True)
        + SCORE_W_Q05 * pool["q05"].rank(pct=True)
    )
    pool = pool.sort_values("vote_score", ascending=False, kind="mergesort")
    picks = pool.head(int(k))["code"].astype(str).tolist()
    return picks, pool


def period_return(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    s = series.dropna()
    s = s[(s.index >= start) & (s.index <= end)]
    if len(s) < 2:
        return None
    p0 = float(s.iloc[0])
    p1 = float(s.iloc[-1])
    if p0 <= 0 or p1 <= 0 or not np.isfinite(p0) or not np.isfinite(p1):
        return None
    return p1 / p0 - 1.0


def max_drawdown_from_rets(period_rets: pd.Series) -> float:
    eq = (1.0 + period_rets.fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def summarize_strategy(
    period_rets: pd.Series,
    *,
    horizon: int,
    label: str,
) -> dict[str, Any]:
    r = period_rets.dropna().astype(float)
    if r.empty:
        return {"strategy": label, "n_periods": 0}
    periods_per_year = float(252.0 / float(horizon))
    mu = float(r.mean())
    sig = float(r.std(ddof=1)) if len(r) >= 2 else float("nan")
    sharpe = (
        mu / sig * np.sqrt(periods_per_year)
        if np.isfinite(sig) and sig > 0
        else float("nan")
    )
    return {
        "strategy": label,
        "n_periods": int(len(r)),
        "cum_ret": float((1.0 + r).prod() - 1.0),
        "mean_period_ret": mu,
        "vol_period": sig,
        "sharpe_ann": float(sharpe),
        "hit_rate": float((r > 0).mean()),
        "max_dd": max_drawdown_from_rets(r),
        "periods_per_year": periods_per_year,
    }


def main() -> int:
    args = parse_args()
    adaptive_dir = (
        args.adaptive_dir
        if args.adaptive_dir.is_absolute()
        else _ROOT / args.adaptive_dir
    )
    tag = pd.Timestamp(args.as_of).strftime("%Y%m%d")
    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else PLOTLY_OUTPUTS_DIR / f"{tag}_vol_need_ticket_bt"
    )
    out_dir = out_dir if out_dir.is_absolute() else _ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    codes, name_map = load_universe(adaptive_dir)
    print(f"universe equity={len(codes)} start={args.start} as_of={args.as_of} H={args.horizon} K={args.k}")

    panel = load_close_panel(
        codes, start=args.start, end=args.as_of, min_history=args.min_history
    )
    calendar = panel.dropna(how="all").index
    dates = rebalance_dates(
        calendar, start=args.start, end=args.as_of, horizon=args.horizon
    )
    if len(dates) < 2:
        print("[ERROR] need >=2 rebalance dates", file=sys.stderr)
        return 2
    print(f"rebalance dates={len(dates)} first={dates[0].date()} last={dates[-1].date()}")

    gcfg = GarchDynamicsConfig(mc_samples=int(args.mc_samples))
    cfg = GarchShortHorizonConfig(garch=gcfg)
    h = int(args.horizon)
    k = int(args.k)

    period_rows: list[dict[str, Any]] = []
    pick_rows: list[dict[str, Any]] = []

    for i, start in enumerate(dates[:-1]):
        end = dates[i + 1]
        rows: list[dict[str, Any]] = []
        for code in panel.columns:
            ser = slice_close_through(panel[code], start)
            if ser.dropna().shape[0] < int(args.min_history):
                continue
            try:
                tr = ticket_row_for_code(
                    ser,
                    pd.Timestamp(start),
                    horizon=h,
                    cfg=cfg,
                    sharpe_like=float(args.sharpe_target),
                )
            except Exception:  # noqa: BLE001
                continue
            if tr is None:
                continue
            tr["code"] = str(code)
            tr["name"] = name_map.get(str(code), "")
            rows.append(tr)
        board = pd.DataFrame(rows)
        if board.empty:
            print(f"[{i+1}/{len(dates)-1}] {start.date()} empty board — skip")
            continue

        picks, scored = select_topk(board, k=k)
        if not picks:
            print(f"[{i+1}/{len(dates)-1}] {start.date()} no picks — skip")
            continue

        # strategy return
        strat_rets = []
        for code in picks:
            r = period_return(panel[code], pd.Timestamp(start), pd.Timestamp(end))
            if r is not None:
                strat_rets.append(r)
        # equal-weight all eligible on board
        ew_rets = []
        for code in board["code"].tolist():
            r = period_return(panel[code], pd.Timestamp(start), pd.Timestamp(end))
            if r is not None:
                ew_rets.append(r)
        if not strat_rets:
            continue
        port = float(np.mean(strat_rets))
        ew = float(np.mean(ew_rets)) if ew_rets else float("nan")
        period_rows.append(
            {
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d"),
                "n_board": int(len(board)),
                "n_picks": len(picks),
                "ticket_top5": port,
                "equal_weight_universe": ew,
            }
        )
        pick_rows.append(
            {
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d"),
                "codes": "|".join(picks),
                "names": "|".join(name_map.get(c, "") for c in picks),
                "scores": "|".join(
                    f"{float(scored.set_index('code').loc[c, 'vote_score']):.3f}"
                    for c in picks
                    if c in set(scored["code"])
                ),
            }
        )
        print(
            f"[{i+1}/{len(dates)-1}] {start.date()}→{end.date()} "
            f"board={len(board)} picks={','.join(picks)} "
            f"ret={port*100:+.2f}% ew={ew*100:+.2f}%"
        )

    periods = pd.DataFrame(period_rows)
    picks_df = pd.DataFrame(pick_rows)
    if periods.empty:
        print("[ERROR] no periods produced", file=sys.stderr)
        return 3

    summary_rows = [
        summarize_strategy(periods["ticket_top5"], horizon=h, label=f"ticket_top{k}"),
        summarize_strategy(
            periods["equal_weight_universe"], horizon=h, label="equal_weight_universe"
        ),
    ]
    summary = pd.DataFrame(summary_rows)

    periods.to_csv(out_dir / "periods.csv", index=False, float_format="%.6f")
    picks_df.to_csv(out_dir / "picks.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False, float_format="%.6f")
    meta = {
        "as_of": args.as_of,
        "start": args.start,
        "horizon": h,
        "k": k,
        "sharpe_target": float(args.sharpe_target),
        "mc_samples": int(args.mc_samples),
        "min_history": int(args.min_history),
        "n_codes": len(codes),
        "score_weights": {
            "match": SCORE_W_MATCH,
            "p_up": SCORE_W_P_UP,
            "p_ge_need": SCORE_W_P_GE_NEED,
            "q05": SCORE_W_Q05,
        },
        "screens": ["match>=1", "p_up>=median", "q05>=p25"],
        "note": "Signal at rebalance close; hold to next rebalance close (research timing).",
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n=== summary ===")
    print(summary.to_string(index=False))
    print(f"\n[OK] out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
