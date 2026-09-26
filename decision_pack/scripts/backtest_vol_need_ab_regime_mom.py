#!/usr/bin/env python3
"""A/B backtest: pure ticket Top5 vs regime-gate + relative momentum (T+1, cash OK).

Strategies
----------
A_ticket_top5
    Same vote card as chat: match>=1, p_up>=median, q05>=p25, composite score Top-K.
    Always invests in up to K names (cash only if board empty).
B_regime_mom
    1) adaptive B0 regime == ``up`` only (frozen per-symbol method)
    2) rank by 20d simple momentum vs pool
    3) ticket veto: match>=1 and q05>=cross-section p25 (on full ticket board)
    4) take up to K; if 0 names → 100% cash
EW_universe
    Equal-weight all codes with enough history (invested benchmark).

Execution: signal at rebalance close T; hold from T+1 close to next_signal+1 close.
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

SCORE_W_MATCH = 0.35
SCORE_W_P_UP = 0.25
SCORE_W_P_GE_NEED = 0.20
SCORE_W_Q05 = 0.20


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
    p.add_argument("--mom-bars", type=int, default=20)
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
    path = adaptive_dir / "configs" / f"{code}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def next_session(calendar: pd.DatetimeIndex, t: pd.Timestamp) -> pd.Timestamp | None:
    idx = int(calendar.searchsorted(pd.Timestamp(t).normalize()))
    if idx < 0 or idx >= len(calendar):
        return None
    # if t is on calendar, next is idx+1; if t not on calendar, searchsorted gives insertion
    if idx < len(calendar) and calendar[idx] == pd.Timestamp(t).normalize():
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
    p0 = float(s.iloc[0])
    p1 = float(s.iloc[-1])
    if p0 <= 0 or p1 <= 0 or not np.isfinite(p0) or not np.isfinite(p1):
        return None
    return p1 / p0 - 1.0


def port_period_return(
    panel: pd.DataFrame,
    codes: list[str],
    entry: pd.Timestamp,
    exit_: pd.Timestamp,
) -> float:
    if not codes:
        return 0.0
    rets = []
    for code in codes:
        if code not in panel.columns:
            continue
        r = period_return(panel[code], entry, exit_)
        if r is not None:
            rets.append(r)
    if not rets:
        return 0.0
    return float(np.mean(rets))


def max_drawdown_from_rets(period_rets: pd.Series) -> float:
    eq = (1.0 + period_rets.fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def summarize(period_rets: pd.Series, *, horizon: int, label: str) -> dict[str, Any]:
    r = period_rets.dropna().astype(float)
    if r.empty:
        return {"strategy": label, "n_periods": 0}
    ppy = float(252.0 / float(horizon))
    mu = float(r.mean())
    sig = float(r.std(ddof=1)) if len(r) >= 2 else float("nan")
    sharpe = mu / sig * np.sqrt(ppy) if np.isfinite(sig) and sig > 0 else float("nan")
    return {
        "strategy": label,
        "n_periods": int(len(r)),
        "cum_ret": float((1.0 + r).prod() - 1.0),
        "mean_period_ret": mu,
        "vol_period": sig,
        "sharpe_ann": float(sharpe),
        "hit_rate": float((r > 0).mean()),
        "cash_rate": float((r == 0.0).mean()),
        "max_dd": max_drawdown_from_rets(r),
        "avg_n_picks": float("nan"),
    }


def ticket_board_at(
    panel: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    horizon: int,
    cfg: GarchShortHorizonConfig,
    sharpe_like: float,
    min_history: int,
    name_map: dict[str, str],
) -> pd.DataFrame:
    prefix = f"{horizon}d_garch"
    rows: list[dict[str, Any]] = []
    for code in panel.columns:
        ser = slice_close_through(panel[code], as_of)
        if ser.dropna().shape[0] < min_history:
            continue
        try:
            g = forecast_garch_ann_vols(
                ser,
                as_of,
                cfg=cfg,
                include_mc=True,
                sharpe_like=sharpe_like,
            )
        except Exception:  # noqa: BLE001
            continue
        if not g.get("garch_reliable"):
            continue
        keys = [
            f"p_up_{prefix}",
            f"q05_{prefix}",
            f"match_{prefix}",
            f"p_ge_need_{prefix}",
        ]
        if any(not np.isfinite(float(g[k])) for k in keys):
            continue
        rows.append(
            {
                "code": str(code),
                "name": name_map.get(str(code), ""),
                "p_up": float(g[f"p_up_{prefix}"]),
                "q05": float(g[f"q05_{prefix}"]),
                "match": float(g[f"match_{prefix}"]),
                "p_ge_need": float(g[f"p_ge_need_{prefix}"]),
            }
        )
    return pd.DataFrame(rows)


def select_ticket_top(
    board: pd.DataFrame,
    *,
    k: int,
) -> list[str]:
    if board.empty:
        return []
    df = board.copy()
    med_up = float(df["p_up"].median())
    q05_floor = float(df["q05"].quantile(0.25))
    screened = df[
        (df["match"] >= 1.0) & (df["p_up"] >= med_up) & (df["q05"] >= q05_floor)
    ].copy()
    pool = screened if len(screened) >= max(1, min(k, 3)) else df.copy()
    pool["vote_score"] = (
        SCORE_W_MATCH * pool["match"].rank(pct=True)
        + SCORE_W_P_UP * pool["p_up"].rank(pct=True)
        + SCORE_W_P_GE_NEED * pool["p_ge_need"].rank(pct=True)
        + SCORE_W_Q05 * pool["q05"].rank(pct=True)
    )
    pool = pool.sort_values("vote_score", ascending=False, kind="mergesort")
    return pool.head(int(k))["code"].astype(str).tolist()


def select_regime_mom(
    *,
    as_of: pd.Timestamp,
    panel: pd.DataFrame,
    regimes: dict[str, pd.Series],
    ticket: pd.DataFrame,
    mom_bars: int,
    k: int,
) -> list[str]:
    """up-only + mom rank + ticket veto; may return fewer than k (cash remainder)."""
    if ticket.empty:
        return []
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
        # last regime on/before as_of
        hist = reg.loc[:as_of].dropna()
        if hist.empty or str(hist.iloc[-1]).lower() != "up":
            continue
        ser = panel[code].dropna()
        ser = ser.loc[:as_of]
        if len(ser) < mom_bars + 1:
            continue
        mom = float(ser.iloc[-1] / ser.iloc[-(mom_bars + 1)] - 1.0)
        if not np.isfinite(mom):
            continue
        rows.append({"code": code, "mom": mom})
    if not rows:
        return []
    df = pd.DataFrame(rows).sort_values("mom", ascending=False, kind="mergesort")
    return df.head(int(k))["code"].astype(str).tolist()


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
        else PLOTLY_OUTPUTS_DIR / f"{tag}_vol_need_ab_regime_mom"
    )
    out_dir = out_dir if out_dir.is_absolute() else _ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    codes, name_map = load_universe(adaptive_dir)
    print(f"universe={len(codes)} start={args.start} as_of={args.as_of} H={args.horizon} K={args.k}")

    pm = _load_mod("pm_ab", _ROOT / "decision_pack/scripts/plot_regime_transition_example.py")
    ev = _load_mod("ev_ab", _ROOT / "decision_pack/scripts/eval_causal_regime_switch_pool.py")
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
            reg = reg[~reg.index.duplicated(keep="last")].sort_index()
            regimes[code] = reg
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
        print("[ERROR] need >=2 signal dates", file=sys.stderr)
        return 2
    print(f"signals={len(signals)} first={signals[0].date()} last={signals[-1].date()}")

    gcfg = GarchDynamicsConfig(mc_samples=int(args.mc_samples))
    tcfg = GarchShortHorizonConfig(garch=gcfg)
    h = int(args.horizon)
    k = int(args.k)

    period_rows: list[dict[str, Any]] = []
    pick_rows: list[dict[str, Any]] = []

    for i, sig in enumerate(signals[:-1]):
        sig_next = signals[i + 1]
        entry = next_session(calendar, sig)
        exit_ = next_session(calendar, sig_next)
        if entry is None or exit_ is None or entry >= exit_:
            print(f"[{i+1}] {sig.date()} missing T+1 window — skip")
            continue

        ticket = ticket_board_at(
            panel,
            pd.Timestamp(sig),
            horizon=h,
            cfg=tcfg,
            sharpe_like=float(args.sharpe_target),
            min_history=int(args.min_history),
            name_map=name_map,
        )
        a_picks = select_ticket_top(ticket, k=k)
        b_picks = select_regime_mom(
            as_of=pd.Timestamp(sig),
            panel=panel,
            regimes=regimes,
            ticket=ticket,
            mom_bars=int(args.mom_bars),
            k=k,
        )
        ew_codes = [
            c
            for c in panel.columns
            if slice_close_through(panel[c], sig).dropna().shape[0] >= int(args.min_history)
        ]

        a_ret = port_period_return(panel, a_picks, entry, exit_)
        b_ret = port_period_return(panel, b_picks, entry, exit_)
        ew_ret = port_period_return(panel, ew_codes, entry, exit_)

        period_rows.append(
            {
                "signal": sig.strftime("%Y-%m-%d"),
                "entry": entry.strftime("%Y-%m-%d"),
                "exit": exit_.strftime("%Y-%m-%d"),
                "n_ticket": int(len(ticket)),
                "n_a": len(a_picks),
                "n_b": len(b_picks),
                "A_ticket_top5": a_ret,
                "B_regime_mom": b_ret,
                "EW_universe": ew_ret,
                "B_minus_EW": b_ret - ew_ret,
                "A_minus_EW": a_ret - ew_ret,
            }
        )
        pick_rows.append(
            {
                "signal": sig.strftime("%Y-%m-%d"),
                "A_codes": "|".join(a_picks),
                "B_codes": "|".join(b_picks),
                "B_n": len(b_picks),
            }
        )
        print(
            f"[{i+1}/{len(signals)-1}] sig={sig.date()} entry={entry.date()}→{exit_.date()} "
            f"A({len(a_picks)})={a_ret*100:+.2f}% "
            f"B({len(b_picks)})={b_ret*100:+.2f}% "
            f"EW={ew_ret*100:+.2f}%"
        )

    periods = pd.DataFrame(period_rows)
    picks_df = pd.DataFrame(pick_rows)
    if periods.empty:
        print("[ERROR] no periods", file=sys.stderr)
        return 3

    summary = pd.DataFrame(
        [
            summarize(periods["A_ticket_top5"], horizon=h, label="A_ticket_top5"),
            summarize(periods["B_regime_mom"], horizon=h, label="B_regime_mom"),
            summarize(periods["EW_universe"], horizon=h, label="EW_universe"),
        ]
    )
    summary.loc[0, "avg_n_picks"] = float(periods["n_a"].mean())
    summary.loc[1, "avg_n_picks"] = float(periods["n_b"].mean())
    summary.loc[1, "cash_rate"] = float((periods["n_b"] == 0).mean())
    summary.loc[0, "cash_rate"] = float((periods["n_a"] == 0).mean())
    summary.loc[2, "cash_rate"] = 0.0
    summary["cum_ret_minus_ew"] = [
        float(summary.loc[0, "cum_ret"] - summary.loc[2, "cum_ret"]),
        float(summary.loc[1, "cum_ret"] - summary.loc[2, "cum_ret"]),
        0.0,
    ]
    ex_a = periods["A_ticket_top5"] - periods["EW_universe"]
    ex_b = periods["B_regime_mom"] - periods["EW_universe"]
    ppy = 252.0 / float(h)

    def _ex_sharpe(ex: pd.Series) -> float:
        if len(ex) < 2 or float(ex.std(ddof=1)) <= 0:
            return float("nan")
        return float(ex.mean() / ex.std(ddof=1) * np.sqrt(ppy))

    summary["info_ratio_vs_ew"] = [_ex_sharpe(ex_a), _ex_sharpe(ex_b), float("nan")]

    periods.to_csv(out_dir / "periods.csv", index=False, float_format="%.6f")
    picks_df.to_csv(out_dir / "picks.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False, float_format="%.6f")
    meta = {
        "as_of": args.as_of,
        "start": args.start,
        "horizon": h,
        "k": k,
        "mom_bars": int(args.mom_bars),
        "execution": "T+1 close to next_signal+1 close",
        "A": "ticket vote TopK",
        "B": "regime==up + mom20 rank + ticket veto (match>=1, q05>=p25); cash if empty",
        "mc_samples": int(args.mc_samples),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n=== summary ===")
    cols = [
        "strategy",
        "n_periods",
        "cum_ret",
        "cum_ret_minus_ew",
        "sharpe_ann",
        "info_ratio_vs_ew",
        "hit_rate",
        "cash_rate",
        "max_dd",
        "avg_n_picks",
    ]
    print(summary[cols].to_string(index=False))
    print(f"\n[OK] out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
