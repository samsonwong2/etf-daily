"""Filter qlib ``all.txt`` by a CSV of fund codes → write canonical instruments txt.

Refactored from ``2_filter/0_all_filter_txt.py``. The main logic is exposed as
an importable function :func:`filter_all_txt` so the orchestrator can call it
in-process; the original script's CLI is preserved as ``python -m
pool_builder.filter_txt`` for back-compat.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Iterable

import pandas as pd

from . import constants as C

DEFAULT_CSV_PATH = pathlib.Path(C.CSV_SELECTED_OUT)
DEFAULT_ALL_PATH = pathlib.Path(C.DEFAULT_ALL_TXT)
DEFAULT_OUTPUT_PATH = pathlib.Path(C.DEFAULT_CANONICAL_TXT)


def _detect_encoding(csv_path: pathlib.Path) -> str:
    with csv_path.open("rb") as f:
        raw = f.read(10000)
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8"
    if b"\xbb" in raw or b"\xbc" in raw:
        return "gbk"
    return "utf-8"


def _read_csv_with_fallback(csv_path: pathlib.Path) -> tuple[pd.DataFrame, str]:
    encoding = _detect_encoding(csv_path)
    for enc_try in (encoding, "utf-8", "gbk", "latin-1"):
        try:
            df = pd.read_csv(csv_path, encoding=enc_try, dtype=str)
            return df, enc_try
        except Exception:
            continue
    raise SystemExit(f"Failed to read CSV {csv_path} with tried encodings")


def _detect_code_column(df: pd.DataFrame, preferred: str | None) -> str:
    if preferred:
        if preferred not in df.columns:
            raise SystemExit(
                f"Specified column '{preferred}' not in CSV columns {list(df.columns)}"
            )
        return preferred
    for cand in ("基金代码", "fund_code", "代码", "code"):
        if cand in df.columns:
            return cand
    # heuristic: column that looks like sh/sz prefix or 6-digit numeric
    for c in df.columns:
        sample = df[c].dropna().astype(str).head(50).tolist()
        n_pref = sum(1 for v in sample if v.strip().lower().startswith(("sh", "sz")))
        n_num6 = sum(1 for v in sample if v.strip().isdigit() and len(v.strip()) == 6)
        if n_pref >= 3 or n_num6 >= 3:
            return c
    raise SystemExit(f"Could not find fund code column in {df.columns}")


def _read_all_txt_lines(all_path: pathlib.Path, preferred_encoding: str) -> list[str]:
    for enc_try in (preferred_encoding, "utf-8", "gbk", "latin-1"):
        try:
            return all_path.read_text(encoding=enc_try).splitlines()
        except Exception:
            continue
    raise SystemExit(f"Failed to read all.txt at {all_path} with tried encodings")


def filter_all_txt(
    *,
    csv_path: pathlib.Path | str = DEFAULT_CSV_PATH,
    all_path: pathlib.Path | str = DEFAULT_ALL_PATH,
    output_path: pathlib.Path | str | None = None,
    code_column: str | None = None,
) -> pathlib.Path:
    """Filter ``all_path`` using codes from ``csv_path``.

    Returns the resolved output path.
    """
    csv_path = pathlib.Path(csv_path)
    all_path = pathlib.Path(all_path)

    default_csv_resolved = DEFAULT_CSV_PATH.resolve()
    if output_path is None:
        if csv_path.resolve() == default_csv_resolved:
            output_path = DEFAULT_OUTPUT_PATH
        else:
            output_path = csv_path.with_suffix(".txt")
    output_path = pathlib.Path(output_path)

    df_in, used_encoding = _read_csv_with_fallback(csv_path)
    print(f"Read CSV {csv_path} using encoding: {used_encoding}")
    code_col = _detect_code_column(df_in, code_column)
    print("Detected fund code column ->", code_col)

    codes_series = df_in[code_col].astype(str).str.strip()
    codes: list[str] = [c for c in codes_series if c and c.lower() != "nan"]

    all_lines = _read_all_txt_lines(all_path, used_encoding)
    codes_set = {c.strip().upper() for c in codes if c and str(c).strip()}
    all_codes_set: set[str] = set()
    filtered: list[str] = []
    for line in all_lines:
        parts = line.split("\t")
        if not parts:
            continue
        code_in_all = parts[0].strip().upper()
        all_codes_set.add(code_in_all)
        if code_in_all in codes_set:
            filtered.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(filtered)
    output_path.write_text((content + "\n") if content else "", encoding="utf-8")

    not_found = [code.upper() for code in codes if code.upper() not in all_codes_set]
    print(f"Done! 共筛出 {len(filtered)} 行，结果已写入 {output_path}")
    if not_found:
        print("\n以下 code 在 all.txt 中未找到：")
        for c in not_found:
            print(c)
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter all.txt using codes from a CSV.")
    parser.add_argument("--csv-path", type=pathlib.Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--code-column", default=None)
    parser.add_argument("--all-path", type=pathlib.Path, default=DEFAULT_ALL_PATH)
    parser.add_argument("--output-path", type=pathlib.Path, default=None)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    filter_all_txt(
        csv_path=args.csv_path,
        all_path=args.all_path,
        output_path=args.output_path,
        code_column=args.code_column,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
