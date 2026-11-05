"""Load close panels from a local Qlib provider."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from etf_daily.data.price_adjustments import auto_qfq_adjust_close_panel


def normalize_code(value: object) -> str:
    """Upper-case an instrument id that is already in Qlib form."""
    return str(value).strip().upper()


def load_qlib_close(
    provider_uri: str,
    codes: list[str],
    start_date: str,
    end_date: str,
    batch_size: int = 250,
    apply_qfq: bool = True,
) -> pd.DataFrame:
    provider_path = Path(provider_uri).expanduser()
    feature_dir = provider_path / "features"
    available_codes = {entry.name.upper() for entry in feature_dir.iterdir()} if feature_dir.exists() else set()
    requested_codes = [code for code in codes if not available_codes or code in available_codes]
    if not requested_codes:
        raise RuntimeError("No requested instruments are available in qlib features")

    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    qlib.init(provider_uri=str(provider_path), region=REG_CN)
    chunks: list[pd.DataFrame] = []
    for start_index in range(0, len(requested_codes), batch_size):
        batch_codes = requested_codes[start_index:start_index + batch_size]
        raw = D.features(batch_codes, ["$close"], start_time=start_date, end_time=end_date, freq="day")
        if raw is None or raw.empty:
            continue
        close_chunk = raw["$close"].unstack("instrument")
        close_chunk.columns = [normalize_code(column) for column in close_chunk.columns]
        chunks.append(close_chunk)
    if not chunks:
        raise RuntimeError("Qlib returned no close data")
    close = pd.concat(chunks, axis=1).sort_index()
    close = close.loc[:, ~close.columns.duplicated()]
    if apply_qfq:
        close, _ = auto_qfq_adjust_close_panel(close)
    return close
