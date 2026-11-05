"""Fund code normalization and name lookup used by the daily commands."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


def normalize_code(value: object) -> str:
    """Normalize a fund code to ``SH######`` / ``SZ######``.

    Accepts bare digits, ``510300.SH``, and ``SH.510300``. ``CASH`` stays ``CASH``.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip().upper()
    if not text:
        return ""
    if text == "CASH":
        return "CASH"

    text = text.replace("_", ".").replace("-", ".")
    match = re.fullmatch(r"(\d{6})\.(SH|SZ)", text)
    if match:
        return f"{match.group(2)}{match.group(1)}"
    match = re.fullmatch(r"(SH|SZ)\.?([0-9]{6})", text)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    match = re.search(r"(\d{6})", text)
    if match:
        code = match.group(1)
        if code.startswith(("5", "6")):
            return f"SH{code}"
        if code.startswith(("0", "1", "2", "3")):
            return f"SZ{code}"
        return code
    return text


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def day_rows(frame: pd.DataFrame, report_date: str) -> pd.DataFrame:
    if "datetime" not in frame.columns:
        return frame.copy()
    mask = frame["datetime"].astype(str).str.startswith(report_date)
    return frame.loc[mask].copy()


def first_existing_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def load_fund_name_map(fund_list_csv: Path | None) -> dict[str, str]:
    if fund_list_csv is None or not fund_list_csv.exists():
        return {}
    frame = read_csv(fund_list_csv)
    code_col = first_existing_column(frame, ["基金代码", "code", "instrument", "symbol"])
    name_col = first_existing_column(frame, ["基金简称", "基金名称", "name", "short_name"])
    if code_col is None or name_col is None:
        return {}
    mapping: dict[str, str] = {}
    for _, row in frame[[code_col, name_col]].dropna(subset=[code_col]).iterrows():
        code = normalize_code(row[code_col])
        name = "" if pd.isna(row[name_col]) else str(row[name_col]).strip()
        if code and name:
            mapping.setdefault(code, name)
    return mapping


def load_portfolio_weights(path: Path, report_date: str) -> tuple[pd.DataFrame, float, str | None]:
    frame = day_rows(read_csv(path), report_date)
    if frame.empty:
        raise ValueError(f"No portfolio weight rows for {report_date} in {path}")
    weight_col = first_existing_column(frame, ["adaptive_final_weight", "target_weight", "当日实际持仓权重"])
    if weight_col is None:
        raise ValueError(f"No actual weight column found in {path}")
    frame = frame.copy()
    frame["code_norm"] = frame["instrument"].map(normalize_code) if "instrument" in frame else ""
    frame["actual_weight"] = pd.to_numeric(frame[weight_col], errors="coerce").fillna(0.0)
    cash_rows = frame.loc[frame["code_norm"].eq("CASH")]
    cash_weight = float(cash_rows["actual_weight"].iloc[0]) if not cash_rows.empty else 0.0
    return frame, cash_weight, weight_col
