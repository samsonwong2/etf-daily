"""Frozen AkShare intraday OHLCV snapshot for regime-transition overlays.

Captures a point-in-time Sina ETF spot bar and injects it on top of Qlib history
without mutating the Qlib binary store. Bars are provisional (not official EOD).

Spot OHLC is **raw**; Qlib plot/signal series are post-hoc qfq. When applying a
live bar, scale OHLC onto the Qlib level via ``昨收`` (preferred) or ``涨跌幅``.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from pipeline.price_adjustments import DEFAULT_QFQ_THRESHOLD
from pipeline.strategy_layer_audit import normalize_code
from runtime_paths import CLUSTER_MAPPING_SELECTED_TXT

SNAPSHOT_COLUMNS = (
    "code",
    "as_of",
    "captured_at",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
)
# Optional; present on new captures. Used to align raw spot → Qlib qfq level.
SNAPSHOT_OPTIONAL_COLUMNS = (
    "pre_close",
    "pct_chg",
)
SNAPSHOT_SOURCE = "akshare_sina_spot"


class LiveSnapshotError(ValueError):
    """Invalid or incomplete live snapshot."""


def _normalize_spot_code(code: object) -> str:
    text = str(code).strip()
    if not text:
        return ""
    if text.upper().startswith(("SH", "SZ")):
        return text.upper()
    if text[0] == "5":
        return f"SH{text}"
    if text[0] in {"1", "0"}:
        return f"SZ{text}"
    return normalize_code(text)


def _finite_positive(value: object) -> float | None:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return None
    out = float(num)
    if not (math.isfinite(out) and out > 0):
        return None
    return out


def _pick_col(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    return None


@dataclass(frozen=True)
class SnapshotApplyStats:
    as_of: str
    n_codes: int
    appended: bool
    replaced_existing: bool
    captured_at: str | None = None
    source_path: str | None = None


def fetch_akshare_etf_spot_ohlcv(
    codes: Iterable[str],
    *,
    as_of: str,
    captured_at: str | None = None,
    spot_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fetch (or parse) AkShare Sina ETF spot rows for ``codes`` on ``as_of``."""
    wanted = sorted({normalize_code(c) for c in codes if normalize_code(c)})
    if not wanted:
        raise LiveSnapshotError("no codes requested for live snapshot")

    if spot_df is None:
        try:
            import akshare as ak
        except ImportError as exc:  # pragma: no cover - env dependent
            raise LiveSnapshotError("akshare not installed") from exc
        try:
            spot_df = ak.fund_etf_category_sina(symbol="ETF基金")
        except Exception as exc:  # pragma: no cover - network dependent
            raise LiveSnapshotError(f"AkShare spot fetch failed: {exc}") from exc

    if spot_df is None or spot_df.empty:
        raise LiveSnapshotError("AkShare spot frame is empty")

    required = ["代码", "最新价"]
    missing = [c for c in required if c not in spot_df.columns]
    if missing:
        raise LiveSnapshotError(f"AkShare spot missing columns: {missing}")

    open_col = _pick_col(spot_df, ("开盘价", "今开"))
    high_col = _pick_col(spot_df, ("最高价", "最高"))
    low_col = _pick_col(spot_df, ("最低价", "最低"))
    volume_col = _pick_col(spot_df, ("成交额", "成交量"))
    pre_close_col = _pick_col(spot_df, ("昨收", "前收盘", "昨收价"))
    pct_col = _pick_col(spot_df, ("涨跌幅",))

    stamp = captured_at or pd.Timestamp.now().isoformat(timespec="seconds")
    as_of_s = pd.Timestamp(as_of).normalize().strftime("%Y-%m-%d")
    wanted_set = set(wanted)
    by_code: dict[str, dict[str, Any]] = {}

    for _, row in spot_df.iterrows():
        code = _normalize_spot_code(row["代码"])
        if code not in wanted_set:
            continue
        close = _finite_positive(row["最新价"])
        if close is None:
            continue
        open_px = _finite_positive(row[open_col]) if open_col else None
        high_px = _finite_positive(row[high_col]) if high_col else None
        low_px = _finite_positive(row[low_col]) if low_col else None
        vol = pd.to_numeric(row[volume_col], errors="coerce") if volume_col else float("nan")
        pre_close = (
            _finite_positive(row[pre_close_col]) if pre_close_col else None
        )
        pct_chg = None
        if pct_col is not None:
            pct_raw = pd.to_numeric(row[pct_col], errors="coerce")
            if pd.notna(pct_raw) and math.isfinite(float(pct_raw)):
                pct_chg = float(pct_raw)
        if open_px is None:
            open_px = close
        if high_px is None:
            high_px = max(open_px, close)
        if low_px is None:
            low_px = min(open_px, close)
        # Keep OHLC consistent with last price when spot fields are partial.
        high_px = max(high_px, open_px, close)
        low_px = min(low_px, open_px, close)
        by_code[code] = {
            "code": code,
            "as_of": as_of_s,
            "captured_at": stamp,
            "open": float(open_px),
            "high": float(high_px),
            "low": float(low_px),
            "close": float(close),
            "volume": float(vol) if pd.notna(vol) else float("nan"),
            "source": SNAPSHOT_SOURCE,
            "pre_close": float(pre_close) if pre_close is not None else float("nan"),
            "pct_chg": float(pct_chg) if pct_chg is not None else float("nan"),
        }

    if not by_code:
        raise LiveSnapshotError("AkShare spot returned no rows for requested codes")

    cols = list(SNAPSHOT_COLUMNS) + list(SNAPSHOT_OPTIONAL_COLUMNS)
    frame = pd.DataFrame([by_code[c] for c in wanted if c in by_code], columns=cols)
    return frame.reset_index(drop=True)


def validate_snapshot(
    frame: pd.DataFrame,
    *,
    as_of: str,
    required_codes: Iterable[str] | None = None,
    strict_coverage: bool = True,
    require_today: bool = False,
    today: str | None = None,
) -> pd.DataFrame:
    """Validate schema/date/coverage; return normalized copy."""
    if frame is None or frame.empty:
        raise LiveSnapshotError("live snapshot is empty")

    missing_cols = [c for c in SNAPSHOT_COLUMNS if c not in frame.columns]
    if missing_cols:
        raise LiveSnapshotError(f"live snapshot missing columns: {missing_cols}")

    out = frame.copy()
    out["code"] = out["code"].map(normalize_code)
    out["as_of"] = pd.to_datetime(out["as_of"]).dt.normalize().dt.strftime("%Y-%m-%d")
    as_of_s = pd.Timestamp(as_of).normalize().strftime("%Y-%m-%d")
    bad_dates = sorted({d for d in out["as_of"].astype(str).unique() if d != as_of_s})
    if bad_dates:
        raise LiveSnapshotError(
            f"live snapshot as_of mismatch: expected {as_of_s}, found {bad_dates}"
        )

    if require_today:
        today_s = pd.Timestamp(today or pd.Timestamp.today()).normalize().strftime("%Y-%m-%d")
        if as_of_s != today_s:
            raise LiveSnapshotError(
                f"live snapshot requires as_of==today ({today_s}), got {as_of_s}"
            )

    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
        bad = out[col].isna() | (out[col] <= 0)
        if bool(bad.any()):
            codes = out.loc[bad, "code"].astype(str).tolist()
            raise LiveSnapshotError(f"invalid {col} for codes: {codes[:10]}")

    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    for opt in SNAPSHOT_OPTIONAL_COLUMNS:
        if opt in out.columns:
            out[opt] = pd.to_numeric(out[opt], errors="coerce")
    out = out.drop_duplicates(subset=["code"], keep="last")
    out = out.sort_values("code").reset_index(drop=True)

    if required_codes is not None:
        required = sorted({normalize_code(c) for c in required_codes if normalize_code(c)})
        have = set(out["code"].astype(str))
        missing = [c for c in required if c not in have]
        if missing and strict_coverage:
            raise LiveSnapshotError(
                f"live snapshot missing {len(missing)}/{len(required)} required codes; "
                f"examples: {missing[:10]}"
            )
        if missing and not strict_coverage:
            print(
                f"[WARN] live snapshot missing {len(missing)} codes "
                f"(strict_coverage=False); examples={missing[:10]}"
            )
    keep = list(SNAPSHOT_COLUMNS) + [
        c for c in SNAPSHOT_OPTIONAL_COLUMNS if c in out.columns
    ]
    return out[keep]


def qlib_align_factor(
    qlib_prev_close: float | None,
    *,
    raw_close: float,
    pre_close: float | None = None,
    pct_chg: float | None = None,
    cliff_threshold: float = DEFAULT_QFQ_THRESHOLD,
) -> tuple[float, str]:
    """Multiplier mapping raw spot OHLC onto Qlib qfq level.

    Prefer ``pre_close`` (新浪昨收): ``factor = qlib_prev / pre_close``.
    Else use ``pct_chg`` (percent units): ``qfq_close = qlib_prev * (1+pct/100)``.
    If neither is available and raw close would cliff vs qlib_prev, return
    factor 1.0 with reason ``cliff_unaligned`` (caller should warn).
    """
    if (
        qlib_prev_close is None
        or not math.isfinite(float(qlib_prev_close))
        or float(qlib_prev_close) <= 0
        or not math.isfinite(float(raw_close))
        or float(raw_close) <= 0
    ):
        return 1.0, "no_qlib_prev"

    q_prev = float(qlib_prev_close)
    raw = float(raw_close)

    if pre_close is not None and math.isfinite(float(pre_close)) and float(pre_close) > 0:
        return float(q_prev / float(pre_close)), "pre_close"

    if pct_chg is not None and math.isfinite(float(pct_chg)):
        qfq_close = q_prev * (1.0 + float(pct_chg) / 100.0)
        if qfq_close > 0 and raw > 0:
            return float(qfq_close / raw), "pct_chg"

    rel = raw / q_prev - 1.0
    if abs(rel) > float(cliff_threshold):
        return 1.0, "cliff_unaligned"
    return 1.0, "same_scale"


def write_live_snapshot(frame: pd.DataFrame, path: Path | str) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def codes_requiring_live_coverage(
    signals: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp,
) -> tuple[list[str], list[str]]:
    """Split signal universe into live-bar required vs historical-only.

    Live snapshots are captured for the *current* pool / as_of session. Historical
    shards may still contain delisted or rotated-out codes (e.g. SZ159616). Those
    must not fail ``strict_coverage`` against a current-pool snapshot.

    Returns ``(required_codes, historical_only_skipped)``.
    """
    if signals is None or signals.empty:
        return [], []
    work = signals.copy()
    work["code"] = work["code"].map(normalize_code)
    work = work[work["code"].astype(str).str.len() > 0]
    work["as_of"] = pd.to_datetime(work["as_of"]).dt.normalize()
    as_of_n = pd.Timestamp(as_of).normalize()
    on_day = work.loc[work["as_of"] == as_of_n, "code"]
    if on_day.empty:
        max_d = work["as_of"].max()
        on_day = work.loc[work["as_of"] == max_d, "code"]
    required = sorted({str(c) for c in on_day.tolist() if c})
    all_codes = sorted({str(c) for c in work["code"].tolist() if c})
    skipped = sorted(set(all_codes) - set(required))
    return required, skipped


def load_live_snapshot(
    path: Path | str,
    *,
    as_of: str | None = None,
    required_codes: Iterable[str] | None = None,
    strict_coverage: bool = True,
) -> pd.DataFrame:
    snap_path = Path(path)
    if not snap_path.exists():
        raise LiveSnapshotError(f"live snapshot not found: {snap_path}")
    frame = pd.read_csv(snap_path)
    if as_of is None:
        if "as_of" not in frame.columns or frame.empty:
            raise LiveSnapshotError(f"cannot infer as_of from {snap_path}")
        as_of = str(pd.to_datetime(frame["as_of"].iloc[0]).date())
    return validate_snapshot(
        frame,
        as_of=as_of,
        required_codes=required_codes,
        strict_coverage=strict_coverage,
        require_today=False,
    )


def capture_live_snapshot(
    *,
    as_of: str,
    codes: Iterable[str],
    output_path: Path | str,
    require_today: bool = True,
    strict_coverage: bool = True,
    today: str | None = None,
    spot_df: pd.DataFrame | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Fetch, validate, and write an immutable live snapshot CSV."""
    raw = fetch_akshare_etf_spot_ohlcv(codes, as_of=as_of, spot_df=spot_df)
    frame = validate_snapshot(
        raw,
        as_of=as_of,
        required_codes=codes,
        strict_coverage=strict_coverage,
        require_today=require_today,
        today=today,
    )
    path = write_live_snapshot(frame, output_path)
    return path, frame


def apply_live_snapshot_to_close_panel(
    panel: pd.DataFrame,
    snapshot: pd.DataFrame,
    *,
    as_of: str,
) -> tuple[pd.DataFrame, SnapshotApplyStats]:
    """Append or replace ``as_of`` closes for every code in the snapshot.

    Unlike ``apply_pack_live_closes``, unchanged prices are still written so the
    calendar date exists for all covered symbols (no NaN holes on as_of).
    Raw spot closes are scaled onto each column's last Qlib (qfq) close via
    ``pre_close`` / ``pct_chg`` when available.
    """
    if panel is None or panel.empty:
        raise LiveSnapshotError("close panel is empty; cannot apply live snapshot")
    as_of_ts = pd.Timestamp(as_of).normalize()
    snap = validate_snapshot(snapshot, as_of=as_of, strict_coverage=False)
    out = panel.copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).normalize()
    replaced = as_of_ts in out.index
    if not replaced:
        out = out.reindex(out.index.union([as_of_ts])).sort_index()

    n = 0
    cliff_warn: list[str] = []
    for _, row in snap.iterrows():
        code = str(row["code"])
        if code not in out.columns:
            continue
        hist = pd.to_numeric(out[code], errors="coerce").dropna()
        # Exclude as_of itself when replacing
        hist = hist[hist.index < as_of_ts]
        qlib_prev = float(hist.iloc[-1]) if len(hist) else None
        raw_close = float(row["close"])
        pre = row["pre_close"] if "pre_close" in snap.columns else None
        pct = row["pct_chg"] if "pct_chg" in snap.columns else None
        pre_f = float(pre) if pre is not None and pd.notna(pre) else None
        pct_f = float(pct) if pct is not None and pd.notna(pct) else None
        factor, reason = qlib_align_factor(
            qlib_prev, raw_close=raw_close, pre_close=pre_f, pct_chg=pct_f
        )
        if reason == "cliff_unaligned":
            cliff_warn.append(code)
        out.loc[as_of_ts, code] = raw_close * factor
        n += 1
    if cliff_warn:
        print(
            f"[WARN] live close not qfq-aligned (no 昨收/涨跌幅), "
            f"cliff vs Qlib for: {cliff_warn[:10]}"
        )
    if n == 0:
        raise LiveSnapshotError("live snapshot matched zero panel columns")

    captured = None
    if "captured_at" in snap.columns and not snap["captured_at"].isna().all():
        captured = str(snap["captured_at"].iloc[0])
    stats = SnapshotApplyStats(
        as_of=as_of_ts.strftime("%Y-%m-%d"),
        n_codes=n,
        appended=not replaced,
        replaced_existing=replaced,
        captured_at=captured,
    )
    return out.sort_index(), stats


def apply_live_snapshot_to_ohlcv(
    ohlcv: pd.DataFrame,
    snapshot: pd.DataFrame,
    *,
    code: str,
    as_of: str,
    datetime_col: str = "datetime",
) -> pd.DataFrame:
    """Append/replace one provisional OHLCV bar for ``code`` on ``as_of``.

    Scales raw spot OHLC onto the Qlib qfq level using snapshot ``pre_close``
    (昨收) or ``pct_chg`` so the bar does not create a fake ex-div cliff.
    """
    if ohlcv is None or ohlcv.empty:
        raise LiveSnapshotError("ohlcv frame is empty; cannot apply live snapshot")
    code_u = normalize_code(code)
    snap = validate_snapshot(snapshot, as_of=as_of, strict_coverage=False)
    rows = snap[snap["code"].astype(str) == code_u]
    if rows.empty:
        raise LiveSnapshotError(f"live snapshot has no row for {code_u}")
    row = rows.iloc[-1]
    as_of_ts = pd.Timestamp(as_of).normalize()

    out = ohlcv.copy()
    out[datetime_col] = pd.to_datetime(out[datetime_col])
    mask = out[datetime_col].dt.normalize() == as_of_ts
    out = out.loc[~mask].copy()

    prev = out.sort_values(datetime_col)
    prev_close = None
    if not prev.empty and "$close" in prev.columns:
        prev_close = pd.to_numeric(prev["$close"], errors="coerce").dropna()
        prev_close = float(prev_close.iloc[-1]) if not prev_close.empty else None

    raw_close = float(row["close"])
    raw_open = float(row["open"])
    raw_high = float(row["high"])
    raw_low = float(row["low"])
    pre = row["pre_close"] if "pre_close" in snap.columns else None
    pct = row["pct_chg"] if "pct_chg" in snap.columns else None
    pre_f = float(pre) if pre is not None and pd.notna(pre) else None
    pct_f = float(pct) if pct is not None and pd.notna(pct) else None
    factor, reason = qlib_align_factor(
        prev_close, raw_close=raw_close, pre_close=pre_f, pct_chg=pct_f
    )
    if reason == "cliff_unaligned":
        print(
            f"[WARN] {code_u}: live bar not qfq-aligned (no 昨收/涨跌幅); "
            f"raw_close={raw_close} qlib_prev={prev_close}"
        )

    close = raw_close * factor
    open_px = raw_open * factor
    high_px = raw_high * factor
    low_px = raw_low * factor
    pct_out = (
        (close / prev_close - 1.0) * 100.0
        if prev_close is not None and prev_close > 0
        else float("nan")
    )
    # Prefer spot 涨跌幅 when we used it / have it (more stable than recomputed).
    if pct_f is not None and reason in {"pre_close", "pct_chg", "same_scale"}:
        pct_out = float(pct_f)

    new_row = {
        datetime_col: as_of_ts,
        "$open": float(open_px),
        "$high": float(high_px),
        "$low": float(low_px),
        "$close": float(close),
        "$volume": float(row["volume"]) if pd.notna(row["volume"]) else float("nan"),
        "$pct_change": pct_out,
    }
    if "instrument" in out.columns:
        new_row["instrument"] = code_u
    out = pd.concat([out, pd.DataFrame([new_row])], ignore_index=True)
    out = out.sort_values(datetime_col).reset_index(drop=True)
    return out


def snapshot_meta(snapshot: pd.DataFrame, *, path: Path | str | None = None) -> dict[str, Any]:
    if snapshot is None or snapshot.empty:
        return {"provisional": True, "n_codes": 0}
    as_of = str(snapshot["as_of"].iloc[0])
    captured = (
        str(snapshot["captured_at"].iloc[0])
        if "captured_at" in snapshot.columns
        else None
    )
    return {
        "provisional": True,
        "as_of": as_of,
        "captured_at": captured,
        "n_codes": int(snapshot["code"].nunique()),
        "source": str(snapshot["source"].iloc[0]) if "source" in snapshot.columns else SNAPSHOT_SOURCE,
        "path": str(path) if path is not None else None,
    }


def _load_cluster_codes(path: Path | None) -> list[str]:
    from decision_pack.src.symbol_trend_board import load_cluster_mapping_codes

    mapping = Path(path or CLUSTER_MAPPING_SELECTED_TXT)
    return list(load_cluster_mapping_codes(mapping))


def codes_with_equity_anchor(codes: Iterable[str]) -> list[str]:
    """Ensure fair-need equity anchor is in the live snapshot code list.

    Trading pools often omit SH510300; rows 4–5 still need its live bar so
    ``r_need`` extends to as_of instead of stopping at the last Qlib EOD.
    """
    from decision_pack.src.vol_need_return_board import EQUITY_ANCHOR

    out: list[str] = []
    seen: set[str] = set()
    for raw in list(codes) + [str(EQUITY_ANCHOR)]:
        code = normalize_code(raw)
        if not code or code in seen:
            continue
        seen.add(code)
        out.append(code)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Capture a frozen AkShare ETF spot OHLCV snapshot for intraday regime runs"
    )
    p.add_argument("--as-of", required=True, help="YYYY-MM-DD (must equal today when capturing)")
    p.add_argument("--output", type=Path, required=True, help="CSV output path")
    p.add_argument(
        "--cluster-mapping",
        type=Path,
        default=None,
        help="instrument list (default: CLUSTER_MAPPING_SELECTED_TXT)",
    )
    p.add_argument(
        "--allow-stale-as-of",
        action="store_true",
        help="allow as_of != machine today (for replay/tests only)",
    )
    p.add_argument(
        "--allow-partial-coverage",
        action="store_true",
        help="do not fail when some pool codes are missing from AkShare",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    codes = codes_with_equity_anchor(_load_cluster_codes(args.cluster_mapping))
    path, frame = capture_live_snapshot(
        as_of=args.as_of,
        codes=codes,
        output_path=args.output,
        require_today=not bool(args.allow_stale_as_of),
        strict_coverage=not bool(args.allow_partial_coverage),
    )
    meta = snapshot_meta(frame, path=path)
    print(
        f"[OK] live snapshot wrote {path} "
        f"as_of={meta['as_of']} codes={meta['n_codes']} captured_at={meta['captured_at']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
