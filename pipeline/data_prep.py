"""Code list + qlib raw panel + AkShare spot-row append."""
from __future__ import annotations

from pathlib import Path

import qlib
from qlib.data import D

from . import constants
from .price_adjustments import (
    auto_qfq_adjust_price_columns_by_group,
    detect_ex_dividend_mask,
    log_qfq_adjustments,
)


def generate_code_list(cluster_mapping_path: Path, codes_out: Path) -> list[str]:
    lines = cluster_mapping_path.read_text(encoding="utf-8").splitlines()
    codes = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        code = stripped.split()[0].upper()
        if code:
            codes.append(code)
    if not codes:
        raise ValueError(f"No codes found in {cluster_mapping_path}")
    codes_out.parent.mkdir(parents=True, exist_ok=True)
    codes_out.write_text("\n".join(codes) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote {len(codes)} codes to {codes_out}")
    return codes


def generate_raw_panel(
    *,
    provider_uri: str,
    codes: list[str],
    start_date: str,
    end_date: str,
    output_path: Path,
) -> list[str]:
    if constants._DRY_RUN:
        print(
            f"[DRY-RUN] Skipping generate_raw_panel (would fetch {len(codes)} codes "
            f"from qlib {provider_uri} {start_date}..{end_date} → {output_path})"
        )
        return list(codes)
    qlib.init(provider_uri=provider_uri, region="cn")
    fields = ["$open", "$high", "$low", "$close", "$volume"]
    raw = D.features(codes, fields, start_time=start_date, end_time=end_date)
    raw.columns = ["open", "high", "low", "close", "volume"]
    raw = raw.sort_index()
    raw_long = raw.reset_index()
    raw_long, qfq_records = auto_qfq_adjust_price_columns_by_group(
        raw_long,
        group_col="instrument",
        date_col="datetime",
        price_cols=["open", "high", "low", "close"],
    )
    if not qfq_records.empty:
        log_qfq_adjustments(qfq_records, stage="generate_raw_panel")
    raw = raw_long.set_index(["instrument", "datetime"]).sort_index()
    raw["next_close"] = raw.groupby(level=0)["close"].shift(-1)
    raw["next_open"] = raw.groupby(level=0)["open"].shift(-1)
    raw["next_return"] = raw["next_close"] / raw["close"] - 1.0
    raw["return"] = raw.groupby(level=0)["close"].pct_change()
    raw["next_rank"] = raw.groupby(level=1)["next_return"].rank(ascending=False, method="average")

    # ── 除权检测：next_return < -0.25 时判断是否为除权日，若是则置 NaN ──
    # 除权日特征：次日收盘价相对当日大幅低于前收，但再下一日价格企稳（不是真实崩盘）。
    # 判断标准：next_return < -0.25 AND 后第2日涨幅 > -0.15
    ex_div_thresh = -0.25
    if (raw["next_return"] < ex_div_thresh).any():
        raw_sorted = raw.reset_index().sort_values(["instrument", "datetime"]).copy()
        raw_sorted["_next2_close"] = raw_sorted.groupby("instrument")["close"].shift(-2)
        raw_sorted["_next2_return"] = raw_sorted["_next2_close"] / raw_sorted["next_close"] - 1.0
        ex_div_flag = detect_ex_dividend_mask(
            raw_sorted["next_return"],
            raw_sorted["_next2_return"],
            ex_div_thresh=ex_div_thresh,
        )
        n_ex_div = ex_div_flag.sum()
        if n_ex_div > 0:
            affected = raw_sorted.loc[ex_div_flag, ["instrument", "datetime", "close", "next_close", "next_return"]].copy()
            print(f"[INFO] Ex-dividend detection: {n_ex_div} rows suspected, setting next_return=NaN")
            print(affected.to_string(index=False))
            raw_sorted.loc[ex_div_flag, "next_return"] = float("nan")
            # Also null out the day-after 'return' column (close_{t+1} / close_t - 1)
            # which carries the same ex-dividend jump and would contaminate Route B
            # GARCH residuals.  raw_sorted is already sorted by (instrument, datetime);
            # using shift(-1) gives us the flag aligned to the polluted row (t+1).
            next_day_flag = ex_div_flag.groupby(raw_sorted["instrument"]).shift(1).fillna(False).astype(bool)
            n_next = int(next_day_flag.sum())
            if n_next > 0:
                raw_sorted.loc[next_day_flag, "return"] = float("nan")
                print(f"[INFO] Ex-dividend detection: also zeroed 'return' on {n_next} next-day rows")
        raw_sorted = raw_sorted.drop(columns=["_next2_close", "_next2_return"])
        # Recompute next_rank after NaN injection
        raw_sorted["next_rank"] = raw_sorted.groupby("datetime")["next_return"].rank(
            ascending=False, method="average"
        )
        raw = raw_sorted.set_index(["instrument", "datetime"])
    # ── 除权检测结束 ─────────────────────────────────────────────────────
    raw = raw.dropna(subset=["close"]).reset_index()
    available_codes = sorted(raw["instrument"].astype(str).str.upper().unique().tolist())
    if not available_codes:
        raise ValueError("No raw panel rows were generated for the requested window")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_path, index=False)
    print(
        f"[INFO] Wrote raw panel to {output_path}, shape={raw.shape}, "
        f"available_codes={len(available_codes)}/{len(codes)}"
    )
    return available_codes


def _normalize_spot_code(code: str) -> str:
    """Convert AkShare numeric code to uppercase qlib-style instrument id."""
    code = str(code).strip()
    if not code:
        return code
    if code[0] == "1":
        return f"SZ{code}"
    if code[0] == "5":
        return f"SH{code}"
    return code.upper()


def append_spot_to_raw_panel(
    raw_panel_path: Path,
    spot_csv_path: Path,
    end_date: str,
) -> None:
    """Append today's realtime AkShare spot rows to an existing raw panel CSV.

    Only executes when end_date matches today's date. Skips silently if today's
    rows are already present. Sets next_close/next_return/next_rank to NaN for
    the terminal row; backtest engine has an existing fallback for this case.
    """
    import pandas as pd
    import akshare as ak

    today = pd.Timestamp.today().normalize()
    target = pd.Timestamp(end_date).normalize()
    if today != target:
        print(f"[INFO] --include-spot skipped: today={today.date()} != end_date={end_date}")
        return

    if constants._DRY_RUN:
        print(
            f"[DRY-RUN] Skipping append_spot_to_raw_panel "
            f"(would fetch AkShare spot for {today.date()} → {raw_panel_path})"
        )
        return

    # Pre-check: dedupe before any network call to AkShare.
    panel = pd.read_csv(raw_panel_path)
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    if (panel["datetime"].dt.normalize() == today).any():
        print(f"[INFO] --include-spot skipped: today={today.date()} already in raw panel (no AkShare call)")
        return

    print(f"[INFO] Fetching spot data from AkShare -> {spot_csv_path}")
    spot_df = ak.fund_etf_category_sina(symbol="ETF基金")
    spot_csv_path.parent.mkdir(parents=True, exist_ok=True)
    spot_df.to_csv(spot_csv_path, index=False)

    instruments = set(panel["instrument"].str.upper().unique())

    required_cols = ["代码", "最新价"]
    missing = [c for c in required_cols if c not in spot_df.columns]
    if missing:
        print(f"[WARN] --include-spot: spot CSV missing columns {missing}, skipping")
        return

    open_col = next((c for c in ["开盘价", "今开"] if c in spot_df.columns), None)
    high_col = next((c for c in ["最高价", "最高"] if c in spot_df.columns), None)
    low_col = next((c for c in ["最低价", "最低"] if c in spot_df.columns), None)
    volume_col = next((c for c in ["成交额", "成交量"] if c in spot_df.columns), None)

    spot_df["instrument"] = spot_df["代码"].map(_normalize_spot_code)
    spot_df = spot_df[spot_df["instrument"].isin(instruments)].copy()
    if spot_df.empty:
        print("[WARN] --include-spot: no matching instruments in spot data, skipping")
        return

    last_close = panel.sort_values("datetime").groupby("instrument")["close"].last()

    rows = []
    for _, row in spot_df.iterrows():
        inst = row["instrument"]
        close_val = pd.to_numeric(row["最新价"], errors="coerce")
        open_val = pd.to_numeric(row[open_col], errors="coerce") if open_col else float("nan")
        high_val = pd.to_numeric(row[high_col], errors="coerce") if high_col else float("nan")
        low_val = pd.to_numeric(row[low_col], errors="coerce") if low_col else float("nan")
        vol_val = pd.to_numeric(row[volume_col], errors="coerce") if volume_col else float("nan")
        prev_close = last_close.get(inst, float("nan"))
        ret_val = (close_val / prev_close - 1.0) if (prev_close and prev_close != 0) else float("nan")
        rows.append({
            "instrument": inst,
            "datetime": today,
            "open": open_val,
            "high": high_val,
            "low": low_val,
            "close": close_val,
            "volume": vol_val,
            "next_close": float("nan"),
            "next_open": float("nan"),
            "next_return": float("nan"),
            "return": ret_val,
            "next_rank": float("nan"),
        })

    if not rows:
        print("[WARN] --include-spot: no rows built, skipping")
        return

    today_df = pd.DataFrame(rows)
    panel = pd.concat([panel, today_df], ignore_index=True)
    panel.to_csv(raw_panel_path, index=False)
    print(f"[INFO] Appended {len(rows)} spot rows for {today.date()} to {raw_panel_path}")
