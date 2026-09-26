"""A→B peak-sell scanner on regime-transition quantile switches.

Manual rule distilled from SH588710 high-zone sells:

- **A 冲高提醒**: up switch with extreme upside (`pred_cdf` high, `return_z` high)
- **B 回落确认**: down switch with extreme downside within ``max_gap_days`` after A

This is an observational screen, not lockbox-gated tradable bias wording.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

DEFAULT_A_MIN_PRED_CDF = 0.90
DEFAULT_A_MIN_RETURN_Z = 2.0
DEFAULT_B_MAX_PRED_CDF = 0.10
DEFAULT_B_MAX_RETURN_Z = -1.5
DEFAULT_MAX_GAP_DAYS = 5


@dataclass(frozen=True)
class ABSellScanConfig:
    a_min_pred_cdf: float = DEFAULT_A_MIN_PRED_CDF
    a_min_return_z: float = DEFAULT_A_MIN_RETURN_Z
    b_max_pred_cdf: float = DEFAULT_B_MAX_PRED_CDF
    b_max_return_z: float = DEFAULT_B_MAX_RETURN_Z
    max_gap_days: int = DEFAULT_MAX_GAP_DAYS
    require_strong_prior: bool = False


def _norm_frame(signals: pd.DataFrame) -> pd.DataFrame:
    need = {"as_of", "code", "quantile_switch", "switch_side", "pred_cdf", "return_z"}
    missing = sorted(need - set(signals.columns))
    if missing:
        raise ValueError(f"signals missing columns: {missing}")
    out = signals.copy()
    out["as_of"] = pd.to_datetime(out["as_of"]).dt.normalize()
    out["code"] = out["code"].astype(str).str.upper()
    out["quantile_switch"] = out["quantile_switch"].fillna(False).astype(bool)
    out["switch_side"] = out["switch_side"].astype(str).str.lower()
    out["pred_cdf"] = pd.to_numeric(out["pred_cdf"], errors="coerce")
    out["return_z"] = pd.to_numeric(out["return_z"], errors="coerce")
    if "name" not in out.columns:
        out["name"] = ""
    if "causal_prior_strong" in out.columns:
        out["causal_prior_strong"] = (
            out["causal_prior_strong"].fillna(False).astype(bool)
        )
    else:
        out["causal_prior_strong"] = False
    return out


def flag_a_candidates(frame: pd.DataFrame, cfg: ABSellScanConfig) -> pd.Series:
    """Boolean mask for 冲高提醒 (A) rows."""
    m = (
        frame["quantile_switch"]
        & frame["switch_side"].eq("up")
        & frame["pred_cdf"].ge(cfg.a_min_pred_cdf)
        & frame["return_z"].ge(cfg.a_min_return_z)
    )
    if cfg.require_strong_prior:
        m = m & frame["causal_prior_strong"]
    return m.fillna(False)


def flag_b_candidates(frame: pd.DataFrame, cfg: ABSellScanConfig) -> pd.Series:
    """Boolean mask for 回落确认 (B) rows."""
    m = (
        frame["quantile_switch"]
        & frame["switch_side"].eq("down")
        & frame["pred_cdf"].le(cfg.b_max_pred_cdf)
        & frame["return_z"].le(cfg.b_max_return_z)
    )
    if cfg.require_strong_prior:
        m = m & frame["causal_prior_strong"]
    return m.fillna(False)


def _pair_one_code(
    a_rows: pd.DataFrame,
    b_rows: pd.DataFrame,
    *,
    max_gap_days: int,
    pair_role: str = "A→B",
) -> list[dict]:
    """Greedy earliest-B pairing per A within calendar gap."""
    if a_rows.empty or b_rows.empty:
        return []
    a_sorted = a_rows.sort_values("as_of")
    b_sorted = b_rows.sort_values("as_of")
    used_b: set[pd.Timestamp] = set()
    pairs: list[dict] = []
    gap = pd.Timedelta(days=int(max_gap_days))
    for _, a in a_sorted.iterrows():
        a_d = pd.Timestamp(a["as_of"])
        cand = None
        for _, b in b_sorted.iterrows():
            b_d = pd.Timestamp(b["as_of"])
            if b_d in used_b:
                continue
            if b_d <= a_d:
                continue
            if b_d > a_d + gap:
                break
            cand = b
            break
        if cand is None:
            continue
        b_d = pd.Timestamp(cand["as_of"])
        used_b.add(b_d)
        pairs.append(
            {
                "code": a["code"],
                "name": a.get("name") or cand.get("name") or "",
                "a_as_of": a_d,
                "b_as_of": b_d,
                "gap_days": int((b_d - a_d).days),
                "a_pred_cdf": float(a["pred_cdf"]),
                "a_return_z": float(a["return_z"]),
                "b_pred_cdf": float(cand["pred_cdf"]),
                "b_return_z": float(cand["return_z"]),
                "a_causal_prior_strong": bool(a.get("causal_prior_strong", False)),
                "b_causal_prior_strong": bool(cand.get("causal_prior_strong", False)),
                "pair_role": pair_role,
            }
        )
    return pairs


# ---------------------------------------------------------------------------
# Buy mirror: A'=杀跌提醒 (extreme down) → B'=反抽确认 (extreme up)
# ---------------------------------------------------------------------------

DEFAULT_BUY_A_MAX_PRED_CDF = 0.10
DEFAULT_BUY_A_MAX_RETURN_Z = -2.0
DEFAULT_BUY_B_MIN_PRED_CDF = 0.90
DEFAULT_BUY_B_MIN_RETURN_Z = 1.5
DEFAULT_BUY_MAX_GAP_DAYS = 7


@dataclass(frozen=True)
class ABBuyScanConfig:
    """Bottom-buy pair: extreme down then extreme up."""

    a_max_pred_cdf: float = DEFAULT_BUY_A_MAX_PRED_CDF
    a_max_return_z: float = DEFAULT_BUY_A_MAX_RETURN_Z
    b_min_pred_cdf: float = DEFAULT_BUY_B_MIN_PRED_CDF
    b_min_return_z: float = DEFAULT_BUY_B_MIN_RETURN_Z
    max_gap_days: int = DEFAULT_BUY_MAX_GAP_DAYS
    require_strong_prior: bool = False


def flag_buy_a_candidates(frame: pd.DataFrame, cfg: ABBuyScanConfig) -> pd.Series:
    """杀跌提醒 A'：极端下行分位切换。"""
    m = (
        frame["quantile_switch"]
        & frame["switch_side"].eq("down")
        & frame["pred_cdf"].le(cfg.a_max_pred_cdf)
        & frame["return_z"].le(cfg.a_max_return_z)
    )
    if cfg.require_strong_prior:
        m = m & frame["causal_prior_strong"]
    return m.fillna(False)


def flag_buy_b_candidates(frame: pd.DataFrame, cfg: ABBuyScanConfig) -> pd.Series:
    """反抽确认 B'：极端上行分位切换。"""
    m = (
        frame["quantile_switch"]
        & frame["switch_side"].eq("up")
        & frame["pred_cdf"].ge(cfg.b_min_pred_cdf)
        & frame["return_z"].ge(cfg.b_min_return_z)
    )
    if cfg.require_strong_prior:
        m = m & frame["causal_prior_strong"]
    return m.fillna(False)


def scan_ab_buy_candidates(
    signals: pd.DataFrame,
    *,
    cfg: ABBuyScanConfig | None = None,
    codes: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scan pool for buy A'→B' pairs (杀跌提醒 → 反抽确认)."""
    cfg = cfg or ABBuyScanConfig()
    frame = _norm_frame(signals)
    if codes is not None:
        wanted = {str(c).upper() for c in codes}
        frame = frame[frame["code"].isin(wanted)]
    if start_date:
        frame = frame[frame["as_of"] >= pd.Timestamp(start_date)]
    if end_date:
        frame = frame[frame["as_of"] <= pd.Timestamp(end_date)]

    a_mask = flag_buy_a_candidates(frame, cfg)
    b_mask = flag_buy_b_candidates(frame, cfg)
    keep_cols = [
        c
        for c in (
            "as_of",
            "code",
            "name",
            "switch_side",
            "pred_cdf",
            "return_z",
            "causal_prior_strong",
            "signal_level",
            "p_switch_hmm",
            "p_into_reversal_hmm",
        )
        if c in frame.columns
    ]
    a_hits = frame.loc[a_mask, keep_cols].copy()
    b_hits = frame.loc[b_mask, keep_cols].copy()
    a_hits.insert(0, "role", "杀跌提醒A'")
    b_hits.insert(0, "role", "反抽确认B'")
    a_hits = a_hits.sort_values(["code", "as_of"]).reset_index(drop=True)
    b_hits = b_hits.sort_values(["code", "as_of"]).reset_index(drop=True)

    pair_rows: list[dict] = []
    for code, a_sub in a_hits.groupby("code", sort=True):
        b_sub = b_hits[b_hits["code"] == code]
        pair_rows.extend(
            _pair_one_code(
                a_sub,
                b_sub,
                max_gap_days=cfg.max_gap_days,
                pair_role="A'→B'(buy)",
            )
        )
    pairs = pd.DataFrame(pair_rows)
    if not pairs.empty:
        pairs = pairs.sort_values(["code", "a_as_of", "b_as_of"]).reset_index(drop=True)
        paired_a = set(
            zip(pairs["code"].astype(str), pd.to_datetime(pairs["a_as_of"]))
        )
        paired_b = set(
            zip(pairs["code"].astype(str), pd.to_datetime(pairs["b_as_of"]))
        )
        a_hits["paired"] = [
            (str(r.code), pd.Timestamp(r.as_of)) in paired_a
            for r in a_hits.itertuples(index=False)
        ]
        b_hits["paired"] = [
            (str(r.code), pd.Timestamp(r.as_of)) in paired_b
            for r in b_hits.itertuples(index=False)
        ]
    else:
        a_hits["paired"] = False
        b_hits["paired"] = False
    return a_hits, b_hits, pairs


def forward_pair_metrics(
    pairs: pd.DataFrame,
    close_by_code: dict[str, pd.Series],
    *,
    horizons: tuple[int, ...] = (1, 3, 5, 10),
    side: str = "sell",
) -> pd.DataFrame:
    """Attach forward close-to-close returns after B (entry = B close).

    ``side``:
      - ``sell``: hit if horizon return < 0
      - ``buy``: hit if horizon return > 0
    """
    if pairs.empty:
        return pairs.copy()
    rows = []
    for _, p in pairs.iterrows():
        code = str(p["code"]).upper()
        close = close_by_code.get(code)
        out = dict(p)
        if close is None or close.empty:
            for h in horizons:
                out[f"ret_{h}d"] = float("nan")
                out[f"hit_{h}d"] = False
            out["mfe_10d"] = float("nan")
            out["mae_10d"] = float("nan")
            rows.append(out)
            continue
        close = close.sort_index()
        close.index = pd.to_datetime(close.index).normalize()
        b_d = pd.Timestamp(p["b_as_of"]).normalize()
        if b_d not in close.index:
            # snap to previous available session
            prev = close.index[close.index <= b_d]
            if len(prev) == 0:
                for h in horizons:
                    out[f"ret_{h}d"] = float("nan")
                    out[f"hit_{h}d"] = False
                out["mfe_10d"] = float("nan")
                out["mae_10d"] = float("nan")
                rows.append(out)
                continue
            b_d = prev[-1]
        px0 = float(close.loc[b_d])
        idx = close.index.get_loc(b_d)
        if not isinstance(idx, int):
            idx = int(idx)
        for h in horizons:
            j = idx + h
            if j >= len(close):
                out[f"ret_{h}d"] = float("nan")
                out[f"hit_{h}d"] = False
            else:
                ret = float(close.iloc[j] / px0 - 1.0)
                out[f"ret_{h}d"] = ret
                out[f"hit_{h}d"] = (ret < 0) if side == "sell" else (ret > 0)
        # MFE/MAE over next 10 sessions from B close
        j1 = idx + 1
        jn = min(idx + 10, len(close) - 1)
        if j1 <= jn:
            window = close.iloc[j1 : jn + 1].astype(float)
            rets = window / px0 - 1.0
            out["mfe_10d"] = float(rets.max())
            out["mae_10d"] = float(rets.min())
        else:
            out["mfe_10d"] = float("nan")
            out["mae_10d"] = float("nan")
        # local extreme context: A near 20d high/low?
        a_d = pd.Timestamp(p["a_as_of"]).normalize()
        if a_d in close.index:
            ai = close.index.get_loc(a_d)
            if not isinstance(ai, int):
                ai = int(ai)
            lo = max(0, ai - 19)
            win = close.iloc[lo : ai + 1]
            out["a_near_20d_high"] = bool(float(close.iloc[ai]) >= float(win.max()) * 0.98)
            out["a_near_20d_low"] = bool(float(close.iloc[ai]) <= float(win.min()) * 1.02)
        else:
            out["a_near_20d_high"] = False
            out["a_near_20d_low"] = False
        rows.append(out)
    return pd.DataFrame(rows)


def summarize_pair_metrics(scored: pd.DataFrame, *, side: str) -> dict:
    """Aggregate hit rates / mean returns for scored pairs."""
    out: dict = {"n_pairs": int(len(scored)), "side": side}
    if scored.empty:
        return out
    for h in (1, 3, 5, 10):
        col = f"ret_{h}d"
        hit = f"hit_{h}d"
        if col not in scored.columns:
            continue
        s = pd.to_numeric(scored[col], errors="coerce")
        valid = s.dropna()
        out[f"n_{h}d"] = int(len(valid))
        out[f"mean_ret_{h}d"] = float(valid.mean()) if len(valid) else float("nan")
        if hit in scored.columns and len(valid):
            out[f"hit_rate_{h}d"] = float(scored.loc[valid.index, hit].mean())
        else:
            out[f"hit_rate_{h}d"] = float("nan")
    return out


def scan_ab_sell_candidates(
    signals: pd.DataFrame,
    *,
    cfg: ABSellScanConfig | None = None,
    codes: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scan pool for A/B candidates and greedy A→B pairs.

    Returns
    -------
    a_hits, b_hits, pairs
        Candidate tables and paired events.
    """
    cfg = cfg or ABSellScanConfig()
    frame = _norm_frame(signals)
    if codes is not None:
        wanted = {str(c).upper() for c in codes}
        frame = frame[frame["code"].isin(wanted)]
    if start_date:
        frame = frame[frame["as_of"] >= pd.Timestamp(start_date)]
    if end_date:
        frame = frame[frame["as_of"] <= pd.Timestamp(end_date)]

    a_mask = flag_a_candidates(frame, cfg)
    b_mask = flag_b_candidates(frame, cfg)
    keep_cols = [
        "as_of",
        "code",
        "name",
        "switch_side",
        "pred_cdf",
        "return_z",
        "causal_prior_strong",
        "signal_level",
        "p_switch_hmm",
        "p_into_reversal_hmm",
    ]
    keep_cols = [c for c in keep_cols if c in frame.columns]

    a_hits = frame.loc[a_mask, keep_cols].copy()
    b_hits = frame.loc[b_mask, keep_cols].copy()
    a_hits.insert(0, "role", "冲高提醒A")
    b_hits.insert(0, "role", "回落确认B")
    a_hits = a_hits.sort_values(["code", "as_of"]).reset_index(drop=True)
    b_hits = b_hits.sort_values(["code", "as_of"]).reset_index(drop=True)

    pair_rows: list[dict] = []
    for code, a_sub in a_hits.groupby("code", sort=True):
        b_sub = b_hits[b_hits["code"] == code]
        pair_rows.extend(
            _pair_one_code(a_sub, b_sub, max_gap_days=cfg.max_gap_days)
        )
    pairs = pd.DataFrame(pair_rows)
    if not pairs.empty:
        pairs = pairs.sort_values(["code", "a_as_of", "b_as_of"]).reset_index(drop=True)
        # Mark which A/B rows were paired.
        paired_a = set(
            zip(pairs["code"].astype(str), pd.to_datetime(pairs["a_as_of"]))
        )
        paired_b = set(
            zip(pairs["code"].astype(str), pd.to_datetime(pairs["b_as_of"]))
        )
        a_hits["paired"] = [
            (str(r.code), pd.Timestamp(r.as_of)) in paired_a
            for r in a_hits.itertuples(index=False)
        ]
        b_hits["paired"] = [
            (str(r.code), pd.Timestamp(r.as_of)) in paired_b
            for r in b_hits.itertuples(index=False)
        ]
    else:
        a_hits["paired"] = False
        b_hits["paired"] = False

    return a_hits, b_hits, pairs


def merge_oos_with_causal_prior(
    oos: pd.DataFrame,
    reversal: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach causal_prior_strong from reversal labels when available."""
    out = oos.copy()
    if reversal is None or reversal.empty:
        if "causal_prior_strong" not in out.columns:
            out["causal_prior_strong"] = False
        return out
    rev = reversal.copy()
    rev["as_of"] = pd.to_datetime(rev["as_of"]).dt.normalize()
    rev["code"] = rev["code"].astype(str).str.upper()
    cols = ["as_of", "code"]
    if "causal_prior_strong" in rev.columns:
        cols.append("causal_prior_strong")
    elif "causal_prior_ret" in rev.columns and "causal_thr_prior" in rev.columns:
        rev = rev.copy()
        rev["causal_prior_strong"] = (
            rev["causal_prior_ret"].abs() >= rev["causal_thr_prior"].abs()
        )
        cols.append("causal_prior_strong")
    else:
        out["causal_prior_strong"] = False
        return out
    rev = rev[cols].drop_duplicates(subset=["as_of", "code"], keep="first")
    out["as_of"] = pd.to_datetime(out["as_of"]).dt.normalize()
    out["code"] = out["code"].astype(str).str.upper()
    out = out.drop(columns=["causal_prior_strong"], errors="ignore")
    return out.merge(rev, on=["as_of", "code"], how="left")


def write_ab_sell_scan_outputs(
    output_dir: Path | str,
    *,
    a_hits: pd.DataFrame,
    b_hits: pd.DataFrame,
    pairs: pd.DataFrame,
    cfg: ABSellScanConfig,
) -> dict[str, Path]:
    """Write candidate / pair CSVs and a short markdown summary."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "a_hits": root / "ab_sell_a_冲高提醒.csv",
        "b_hits": root / "ab_sell_b_回落确认.csv",
        "pairs": root / "ab_sell_pairs.csv",
        "summary": root / "ab_sell_summary.md",
    }
    a_hits.to_csv(paths["a_hits"], index=False, encoding="utf-8-sig")
    b_hits.to_csv(paths["b_hits"], index=False, encoding="utf-8-sig")
    pairs.to_csv(paths["pairs"], index=False, encoding="utf-8-sig")

    lines = [
        "# A→B 高点卖出筛子",
        "",
        "## 规则",
        "",
        f"- A 冲高提醒: up switch, `pred_cdf` ≥ {cfg.a_min_pred_cdf}, "
        f"`return_z` ≥ {cfg.a_min_return_z}",
        f"- B 回落确认: down switch, `pred_cdf` ≤ {cfg.b_max_pred_cdf}, "
        f"`return_z` ≤ {cfg.b_max_return_z}",
        f"- 配对: A 后 {cfg.max_gap_days} 个日历日内最早未占用 B（贪心）",
        f"- require_strong_prior={cfg.require_strong_prior}",
        "",
        "## 计数",
        "",
        f"- A 候选: {len(a_hits)}（已配对 {int(a_hits['paired'].sum()) if not a_hits.empty else 0}）",
        f"- B 候选: {len(b_hits)}（已配对 {int(b_hits['paired'].sum()) if not b_hits.empty else 0}）",
        f"- A→B 配对: {len(pairs)}",
        "",
    ]
    if not pairs.empty:
        lines.extend(["## 配对一览", "", "```", pairs.to_string(index=False), "```", ""])
    paths["summary"].write_text("\n".join(lines), encoding="utf-8")
    return paths


def config_to_dict(cfg: ABSellScanConfig) -> dict:
    return asdict(cfg)


__all__ = [
    "ABBuyScanConfig",
    "ABSellScanConfig",
    "DEFAULT_A_MIN_PRED_CDF",
    "DEFAULT_A_MIN_RETURN_Z",
    "DEFAULT_B_MAX_PRED_CDF",
    "DEFAULT_B_MAX_RETURN_Z",
    "DEFAULT_BUY_A_MAX_PRED_CDF",
    "DEFAULT_BUY_A_MAX_RETURN_Z",
    "DEFAULT_BUY_B_MIN_PRED_CDF",
    "DEFAULT_BUY_B_MIN_RETURN_Z",
    "DEFAULT_BUY_MAX_GAP_DAYS",
    "DEFAULT_MAX_GAP_DAYS",
    "config_to_dict",
    "flag_a_candidates",
    "flag_b_candidates",
    "flag_buy_a_candidates",
    "flag_buy_b_candidates",
    "forward_pair_metrics",
    "merge_oos_with_causal_prior",
    "scan_ab_buy_candidates",
    "scan_ab_sell_candidates",
    "summarize_pair_metrics",
    "write_ab_sell_scan_outputs",
]
