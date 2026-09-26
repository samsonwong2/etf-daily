#!/usr/bin/env python3
"""Generate a daily forecast-mu position report from audit CSV outputs."""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_paths import CLUSTER_MAPPING_SELECTED_TXT, FUND_LIST_CSV, NOTES_DIR, QLIB_PROVIDER_URI, TEMP_DIR


DEFAULT_BASE_DIR = Path(
    TEMP_DIR / "trend_sleeve_ablation_20260529" / "2025_regime_switch_ewma_shrink_no_sleeve"
)
DEFAULT_FUND_LIST_CSV = FUND_LIST_CSV
DEFAULT_PREFERRED_START = "20260401"
HOLDING_WEIGHT_EPS = 1e-6
RANGE_SUFFIX_RE = re.compile(r"_(20\d{6})_(20\d{6})(?:$|_|\.csv)")


@dataclass(frozen=True)
class InputPaths:
    base_dir: Path
    audit_dir: Path
    layer_quality_csv: Path
    portfolio_weights_csv: Path
    bridge_mean_monitor_csv: Path | None
    with_name_csv: Path | None
    rebalance_event_log_csv: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 20260527.md-style daily mu/position report."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Trend-sleeve run output directory. Defaults to the 20260528 no-sleeve run.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Report date, accepted as YYYY-MM-DD or YYYYMMDD. Defaults to latest audit date.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown output path. Defaults to <project_root>/workspace/notes/<YYYYMMDD>.md.",
    )
    parser.add_argument(
        "--fund-list-csv",
        type=Path,
        default=DEFAULT_FUND_LIST_CSV,
        help="Optional fund list CSV for code-to-name fallback.",
    )
    parser.add_argument(
        "--preferred-start",
        default=DEFAULT_PREFERRED_START,
        help="Preferred audit/run start date when several ranges end on the same date.",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=None,
        help="Optional explicit audit output directory.",
    )
    parser.add_argument(
        "--ensure-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate missing full audit outputs from existing pipeline artifacts in --base-dir.",
    )
    parser.add_argument(
        "--audit-level",
        choices=("simple", "full"),
        default="full",
        help="Audit level used when --ensure-audit creates missing outputs.",
    )
    return parser.parse_args()


def normalize_report_date(value: str) -> tuple[str, str]:
    text = str(value).strip()
    if re.fullmatch(r"\d{8}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:]}", text
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text, text.replace("-", "")
    raise ValueError(f"Unsupported date format: {value!r}; use YYYY-MM-DD or YYYYMMDD")


def normalize_code(value: object) -> str:
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


def truthy_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "1.0", "true", "t", "yes", "y"})


def first_existing_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def coerce_float(value: object, default: float = 0.0) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return default
    return float(number)


def finite_or_none(value: object) -> float | None:
    number = coerce_float(value, default=float("nan"))
    return number if math.isfinite(number) else None


def compact_to_iso(compact_date: str) -> str:
    return f"{compact_date[:4]}-{compact_date[4:6]}-{compact_date[6:]}"


def first_existing_path(candidates: Iterable[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def iter_range_suffixes(base_dir: Path, compact_date: str) -> list[str]:
    if not base_dir.is_dir():
        return []
    suffixes: set[str] = set()
    for path in base_dir.iterdir():
        for match in RANGE_SUFFIX_RE.finditer(path.name):
            start, end = match.group(1), match.group(2)
            if end == compact_date:
                suffixes.add(f"{start}_{end}")
    return sorted(suffixes)


def infer_preferred_start(base_dir: Path, compact_date: str, preferred_start: str) -> str:
    suffixes = iter_range_suffixes(base_dir, compact_date)
    if not suffixes:
        return preferred_start
    preferred_suffix = f"{preferred_start}_{compact_date}"
    if preferred_suffix in suffixes:
        return preferred_start
    return suffixes[-1].split("_", 1)[0]


def infer_latest_compact_date(base_dir: Path) -> str:
    dates: set[str] = set()
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")
    for path in base_dir.iterdir():
        for match in RANGE_SUFFIX_RE.finditer(path.name):
            dates.add(match.group(2))
        if path.is_dir() and path.name.startswith(("audit_", "parameter_effect_audit_", "monitor_", "mu_monitor_report_")):
            match = re.search(r"(20\d{6})$", path.name)
            if match:
                dates.add(match.group(1))
        if path.is_dir() and path.name.startswith(("audit_", "parameter_effect_audit_")):
            for weights_csv in path.glob("latest_*weights*.csv"):
                match = re.search(r"latest_(?:portfolio_)?weights_(20\d{6})\.csv$", weights_csv.name)
                if match:
                    dates.add(match.group(1))
    if not dates:
        raise FileNotFoundError(f"No audit date could be inferred under {base_dir}")
    return sorted(dates)[-1]


def select_preferred_path(paths: Iterable[Path], compact_date: str, preferred_start: str) -> Path | None:
    candidates = [path for path in paths if path.exists()]
    if not candidates:
        return None
    exact = [path for path in candidates if f"_{preferred_start}_{compact_date}" in path.name]
    if exact:
        return sorted(exact)[-1]
    dated = [
        path
        for path in candidates
        if path.name.endswith(compact_date) or f"_{compact_date}." in path.name or f"_{compact_date}_" in path.name
    ]
    if dated:
        return sorted(dated, key=lambda path: path.name, reverse=True)[0]
    return sorted(candidates, key=lambda path: path.name, reverse=True)[0]


def collect_dated_dirs(base_dir: Path, prefixes: tuple[str, ...], compact_date: str) -> list[Path]:
    if not base_dir.is_dir():
        return []
    candidates: list[Path] = []
    for path in base_dir.iterdir():
        if not path.is_dir():
            continue
        if not any(path.name.startswith(prefix) for prefix in prefixes):
            continue
        if path.name.endswith(compact_date) or f"_{compact_date}_" in path.name:
            candidates.append(path)
    return candidates


def audit_dir_candidates(base_dir: Path, suffix: str, compact_date: str) -> list[Path]:
    return [
        base_dir / f"audit_spm_{suffix}",
        base_dir / f"parameter_effect_audit_simple_pulse_mu_fullpanel_{suffix}",
        *collect_dated_dirs(base_dir, ("audit_", "parameter_effect_audit_"), compact_date),
    ]


def monitor_dir_candidates(base_dir: Path, suffix: str, compact_date: str) -> list[Path]:
    return [
        base_dir / f"monitor_spm_{suffix}",
        base_dir / f"mu_monitor_report_simple_pulse_mu_fullpanel_{suffix}",
        *collect_dated_dirs(base_dir, ("monitor_", "mu_monitor_report_"), compact_date),
    ]


def with_name_csv_candidates(base_dir: Path, suffix: str, compact_date: str) -> list[Path]:
    return [
        base_dir / f"bt_spm_{suffix}.csv",
        base_dir / f"simple_pulse_mu_close_to_next_close_with_name_{suffix}.csv",
        *sorted(base_dir.glob(f"bt_*_{suffix}.csv")),
        *sorted(base_dir.glob(f"simple_pulse_mu_close_to_next_close_with_name_{suffix}.csv")),
        *sorted(base_dir.glob(f"*with_name*_{compact_date}.csv")),
    ]


def discover_bridge_l3_csv(base_dir: Path, suffix: str) -> Path | None:
    return first_existing_path(
        [
            base_dir / f"bridge_spm_{suffix}_l3.csv",
            base_dir / f"bridge_simple_pulse_mu_fullpanel_{suffix}_l3.csv",
            *sorted(base_dir.glob(f"bridge_*_{suffix}_l3.csv")),
        ]
    )


def discover_raw_panel_csv(base_dir: Path, suffix: str) -> Path | None:
    return first_existing_path(
        [
            base_dir / f"raw_{suffix}.csv",
            base_dir / f"new_raw_data_fullpanel_{suffix}.csv",
            *sorted(base_dir.glob(f"raw_*_{suffix}.csv")),
            *sorted(base_dir.glob(f"new_raw_data_fullpanel_{suffix}.csv")),
        ]
    )


def discover_daily_nav_csv(base_dir: Path, suffix: str) -> Path | None:
    return first_existing_path(
        [
            base_dir / f"nav_spm_{suffix}.csv",
            base_dir / f"simple_pulse_mu_close_to_next_close_daily_nav_{suffix}.csv",
            *sorted(base_dir.glob(f"nav_*_{suffix}.csv")),
            *sorted(base_dir.glob(f"*daily_nav*_{suffix}.csv")),
        ]
    )


def discover_weights_csv(base_dir: Path, suffix: str) -> Path | None:
    return first_existing_path(
        [
            base_dir / f"weights_spm_{suffix}.csv",
            base_dir / f"simple_pulse_mu_close_to_next_close_weights_{suffix}.csv",
            *sorted(base_dir.glob(f"weights_*_{suffix}.csv")),
            *sorted(base_dir.glob(f"*close_to_next_close_weights_{suffix}.csv")),
        ]
    )


def discover_weights_full_csv(base_dir: Path, suffix: str) -> Path | None:
    return first_existing_path(
        [
            base_dir / f"weights_full_spm_{suffix}.csv",
            base_dir / f"weights_full_backtrader_{suffix}.csv",
            *sorted(base_dir.glob(f"weights_full_*_{suffix}.csv")),
        ]
    )


def discover_route_a_output_dir(base_dir: Path, suffix: str) -> Path | None:
    return first_existing_path(
        [
            *sorted(base_dir.glob(f"route_a_*_{suffix}")),
            *sorted(base_dir.glob("route_a_forecast_mu_*")),
        ]
    )


def resolve_layer_quality_csv(audit_dir: Path) -> Path:
    for candidate in (
        audit_dir / "layer_latest_pending.csv",
        audit_dir / "latest_layer_quality_pending.csv",
    ):
        if candidate.exists():
            return candidate
    return audit_dir / "layer_latest_pending.csv"


def resolve_portfolio_weights_csv(audit_dir: Path, compact_date: str) -> Path:
    for candidate in (
        audit_dir / f"latest_weights_{compact_date}.csv",
        audit_dir / f"latest_portfolio_weights_{compact_date}.csv",
        audit_dir / "latest_weights.csv",
        audit_dir / "latest_portfolio_weights.csv",
        audit_dir / "portfolio_latest.csv",
    ):
        if candidate.exists():
            return candidate
    return audit_dir / f"latest_weights_{compact_date}.csv"


def audit_outputs_ready(audit_dir: Path, compact_date: str) -> bool:
    layer_quality_csv = resolve_layer_quality_csv(audit_dir)
    if not layer_quality_csv.exists():
        return False
    return resolve_portfolio_weights_csv(audit_dir, compact_date).exists()


def ensure_audit_outputs(
    base_dir: Path,
    compact_date: str,
    preferred_start: str,
    *,
    audit_level: str,
) -> Path:
    preferred_start = infer_preferred_start(base_dir, compact_date, preferred_start)
    suffix = f"{preferred_start}_{compact_date}"
    audit_dir = select_preferred_path(audit_dir_candidates(base_dir, suffix, compact_date), compact_date, preferred_start)
    if audit_dir is not None and audit_outputs_ready(audit_dir, compact_date):
        return audit_dir

    bridge_csv = discover_bridge_l3_csv(base_dir, suffix)
    if bridge_csv is None:
        hints = sorted(
            path.name
            for path in base_dir.iterdir()
            if path.is_dir() and ("audit" in path.name or "monitor" in path.name or "bridge" in path.name)
        )[:12]
        raise FileNotFoundError(
            f"Missing audit outputs and could not locate bridge L3 CSV for suffix {suffix} under {base_dir}. "
            f"Seen related entries: {hints or '(none)'}"
        )

    output_dir = base_dir / f"audit_spm_{suffix}"
    if audit_dir is not None:
        output_dir = audit_dir

    from pipeline.daily_parameter_effect_audit import generate_daily_parameter_effect_audit

    print(f"[INFO] Generating missing audit outputs -> {output_dir}")
    cluster_full_csv = TEMP_DIR / "cluster_mapping.csv"
    generate_daily_parameter_effect_audit(
        bridge_csv_path=bridge_csv,
        output_dir=output_dir,
        cluster_mapping_path=CLUSTER_MAPPING_SELECTED_TXT,
        cluster_mapping_full_csv_path=cluster_full_csv if cluster_full_csv.exists() else None,
        route_a_output_dir=discover_route_a_output_dir(base_dir, suffix),
        raw_panel_path=discover_raw_panel_csv(base_dir, suffix),
        daily_nav_path=discover_daily_nav_csv(base_dir, suffix),
        weights_path=discover_weights_csv(base_dir, suffix),
        weights_full_path=discover_weights_full_csv(base_dir, suffix),
        start_date=compact_to_iso(preferred_start),
        end_date=compact_to_iso(compact_date),
        provider_uri=str(QLIB_PROVIDER_URI),
        audit_level=audit_level,
    )
    return output_dir


def resolve_input_paths(
    base_dir: Path,
    compact_date: str,
    preferred_start: str,
    *,
    audit_dir_override: Path | None = None,
    ensure_audit: bool = False,
    audit_level: str = "full",
) -> InputPaths:
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    preferred_start = infer_preferred_start(base_dir, compact_date, preferred_start)
    suffix = f"{preferred_start}_{compact_date}"

    audit_dir = audit_dir_override.expanduser().resolve() if audit_dir_override is not None else None
    if audit_dir is None:
        audit_dir = select_preferred_path(
            audit_dir_candidates(base_dir, suffix, compact_date),
            compact_date,
            preferred_start,
        )
    if audit_dir is None or not audit_outputs_ready(audit_dir, compact_date):
        if ensure_audit:
            audit_dir = ensure_audit_outputs(
                base_dir,
                compact_date,
                preferred_start,
                audit_level=audit_level,
            )
        elif audit_dir is None:
            hints = sorted(
                path.name
                for path in base_dir.iterdir()
                if path.is_dir() and path.name.startswith(("audit_", "parameter_effect_audit_"))
            )
            raise FileNotFoundError(
                f"No audit directory ending on {compact_date} under {base_dir}. "
                f"Found audit dirs: {hints or '(none)'}. "
                f"Re-run with --ensure-audit to generate missing outputs."
            )
        else:
            raise FileNotFoundError(
                f"Audit directory {audit_dir} is missing layer quality / portfolio weight CSVs. "
                f"Re-run with --ensure-audit --audit-level full."
            )

    layer_quality_csv = resolve_layer_quality_csv(audit_dir)
    if not layer_quality_csv.exists():
        raise FileNotFoundError(f"Missing layer quality pending CSV under {audit_dir}")

    portfolio_weights_csv = resolve_portfolio_weights_csv(audit_dir, compact_date)
    if not portfolio_weights_csv.exists():
        raise FileNotFoundError(f"Missing portfolio weights CSV under {audit_dir}")

    with_name_csv = select_preferred_path(
        with_name_csv_candidates(base_dir, suffix, compact_date),
        compact_date,
        preferred_start,
    )
    monitor_dir = select_preferred_path(
        monitor_dir_candidates(base_dir, suffix, compact_date),
        compact_date,
        preferred_start,
    )
    bridge_mean_monitor_csv = None
    if monitor_dir is not None:
        candidate = monitor_dir / "layer2_bridge_mean_monitor_panel.csv"
        if candidate.exists():
            bridge_mean_monitor_csv = candidate
    rebalance_event_log_csv = audit_dir / "rebalance_log.csv"
    if not rebalance_event_log_csv.exists():
        rebalance_event_log_csv = audit_dir / "rebalance_event_log.csv"
    return InputPaths(
        base_dir=base_dir,
        audit_dir=audit_dir,
        layer_quality_csv=layer_quality_csv,
        portfolio_weights_csv=portfolio_weights_csv,
        bridge_mean_monitor_csv=bridge_mean_monitor_csv,
        with_name_csv=with_name_csv,
        rebalance_event_log_csv=rebalance_event_log_csv if rebalance_event_log_csv.exists() else None,
    )


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


def load_name_and_theme_maps(
    with_name_csv: Path | None,
    report_date: str,
    fund_list_csv: Path | None,
) -> tuple[dict[str, str], dict[str, str]]:
    name_map = load_fund_name_map(fund_list_csv)
    theme_map: dict[str, str] = {}
    if with_name_csv is None or not with_name_csv.exists():
        return name_map, theme_map

    frame = day_rows(read_csv(with_name_csv), report_date)
    if frame.empty:
        frame = read_csv(with_name_csv)
    code_col = first_existing_column(frame, ["code", "instrument", "基金代码"])
    name_col = first_existing_column(frame, ["基金简称", "基金名称", "name", "short_name"])
    theme_col = first_existing_column(frame, ["factor_industry_theme", "theme"])
    if code_col is None:
        return name_map, theme_map

    for _, row in frame.iterrows():
        code = normalize_code(row[code_col])
        if not code or code == "CASH":
            continue
        if name_col is not None and not pd.isna(row.get(name_col)):
            name = str(row[name_col]).strip()
            if name:
                name_map[code] = name
        if theme_col is not None and not pd.isna(row.get(theme_col)):
            theme = str(row[theme_col]).strip()
            if theme:
                theme_map[code] = theme
    return name_map, theme_map


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


def build_candidate_panel(
    layer_quality_csv: Path,
    bridge_mean_monitor_csv: Path | None,
    portfolio_frame: pd.DataFrame,
    report_date: str,
    name_map: dict[str, str],
    theme_map: dict[str, str],
) -> pd.DataFrame:
    layer = day_rows(read_csv(layer_quality_csv), report_date)
    if layer.empty:
        raise ValueError(f"No layer quality rows for {report_date} in {layer_quality_csv}")
    if "instrument" not in layer.columns:
        raise ValueError(f"Missing instrument column in {layer_quality_csv}")
    if "mean_prediction" not in layer.columns:
        raise ValueError(f"Missing mean_prediction column in {layer_quality_csv}")

    layer = layer.copy()
    layer["code_norm"] = layer["instrument"].map(normalize_code)
    layer = layer.loc[layer["code_norm"].ne("CASH") & layer["code_norm"].ne("")].copy()
    if "pool_selected_flag" in layer.columns and truthy_series(layer["pool_selected_flag"]).any():
        layer = layer.loc[truthy_series(layer["pool_selected_flag"])].copy()

    actual_by_code = (
        portfolio_frame.loc[portfolio_frame["code_norm"].ne("CASH")]
        .groupby("code_norm")["actual_weight"]
        .sum()
    )
    layer["actual_weight"] = layer["code_norm"].map(actual_by_code).fillna(0.0)
    if "adaptive_final_weight" in layer.columns:
        fallback_weight = pd.to_numeric(layer["adaptive_final_weight"], errors="coerce").fillna(0.0)
        layer["actual_weight"] = layer["actual_weight"].where(layer["actual_weight"].abs() > 0, fallback_weight)

    layer["mean_prediction"] = pd.to_numeric(layer["mean_prediction"], errors="coerce")
    layer = layer.dropna(subset=["mean_prediction"]).copy()
    # 缩放后 μ：bridge 经 transform/overlay 后送入 MVO 的日度均值（forecast_weight_mean_used）。
    layer["scaled_forecast_mu"] = layer["mean_prediction"]
    layer["forecast_mu_raw"] = layer["scaled_forecast_mu"]
    if bridge_mean_monitor_csv is not None and bridge_mean_monitor_csv.exists():
        bridge = day_rows(read_csv(bridge_mean_monitor_csv), report_date)
        if not bridge.empty and "instrument" in bridge.columns:
            bridge = bridge.copy()
            bridge["code_norm"] = bridge["instrument"].map(normalize_code)
            raw_col = first_existing_column(bridge, ["forecast_mean_return", "forecast_weight_mean_raw"])
            used_col = first_existing_column(bridge, ["forecast_weight_mean_used"])
            if raw_col is not None:
                raw_by_code = pd.to_numeric(bridge[raw_col], errors="coerce").groupby(bridge["code_norm"]).first()
                layer["forecast_mu_raw"] = layer["code_norm"].map(raw_by_code).fillna(layer["forecast_mu_raw"])
            if used_col is not None:
                used_by_code = pd.to_numeric(bridge[used_col], errors="coerce").groupby(bridge["code_norm"]).first()
                layer["scaled_forecast_mu"] = layer["code_norm"].map(used_by_code).fillna(layer["scaled_forecast_mu"])
    layer = layer.sort_values(["mean_prediction", "code_norm"], ascending=[False, True]).reset_index(drop=True)
    layer["mu_rank"] = range(1, len(layer) + 1)
    fallback_rank_pct = layer["mu_rank"] / max(len(layer), 1)
    if "mean_rank_pct_desc" in layer.columns:
        rank_pct = pd.to_numeric(layer["mean_rank_pct_desc"], errors="coerce")
        if rank_pct.notna().any():
            layer["rank_pct"] = rank_pct.fillna(fallback_rank_pct)
        else:
            layer["rank_pct"] = fallback_rank_pct
    else:
        layer["rank_pct"] = fallback_rank_pct

    if "factor_industry_theme" not in layer.columns:
        layer["factor_industry_theme"] = ""
    layer["theme"] = layer.apply(
        lambda row: str(row.get("factor_industry_theme") or theme_map.get(row["code_norm"], "") or "other"),
        axis=1,
    )
    layer["theme"] = layer["theme"].replace({"nan": "other", "None": "other", "": "other"})
    layer["name"] = layer["code_norm"].map(name_map).fillna("")
    if "cluster_mapping_name_latest" in layer.columns:
        fallback_name = layer["cluster_mapping_name_latest"].fillna("").astype(str).str.strip()
        layer["name"] = layer["name"].where(layer["name"].ne(""), fallback_name)
    return layer


def format_signed_percent(value: object, decimals: int) -> str:
    number = finite_or_none(value)
    if number is None:
        return ""
    percent = number * 100.0
    sign = "−" if percent < 0 else ""
    return f"{sign}{abs(percent):.{decimals}f}%"


def format_pct(value: object, decimals: int = 1) -> str:
    number = finite_or_none(value)
    if number is None:
        return ""
    sign = "−" if number < 0 else ""
    return f"{sign}{abs(number) * 100.0:.{decimals}f}%"


def format_weight_pct(value: object, decimals: int = 2) -> str:
    return format_pct(value, decimals=decimals)


def format_compact_weight_pct(value: object) -> str:
    number = finite_or_none(value)
    if number is None:
        return ""
    percent = number * 100.0
    if abs(percent - round(percent)) < 1e-8:
        return f"{int(round(percent))}%"
    return f"{percent:.2f}%"


def markdown_table(rows: list[dict[str, str]], columns: list[tuple[str, str]]) -> str:
    headers = [name for name, _ in columns]
    aligns = [align for _, align in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def rows_for_table(frame: pd.DataFrame, include_actual_weight: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for _, row in frame.iterrows():
        item = {
            "μ排名": str(int(row["mu_rank"])),
            "代码": row["code_norm"],
            "基金简称": row.get("name", ""),
            "theme": row.get("theme", "other"),
            "forecast μ": format_signed_percent(row.get("forecast_mu_raw", row["mean_prediction"]), 4),
            "缩放后 μ": format_signed_percent(row.get("scaled_forecast_mu", row["mean_prediction"]), 4),
            "rank_pct": format_pct(row["rank_pct"], 1),
        }
        if include_actual_weight:
            item["实际权重"] = format_weight_pct(row["actual_weight"], 2)
        rows.append(item)
    return rows


def get_rebalance_metadata(
    portfolio_frame: pd.DataFrame,
    rebalance_event_log_csv: Path | None,
    report_date: str,
) -> tuple[bool, str, int | None, str | None]:
    is_rebalance = False
    if "is_rebalance_day" in portfolio_frame.columns:
        is_rebalance = bool(truthy_series(portfolio_frame["is_rebalance_day"]).any())
    reason = ""
    if "rebalance_reason" in portfolio_frame.columns:
        reasons = [str(value).strip() for value in portfolio_frame["rebalance_reason"].dropna() if str(value).strip()]
        reason = reasons[0] if reasons else ""
    gap_days: int | None = None
    if "rebalance_gap_days" in portfolio_frame.columns:
        gaps = pd.to_numeric(portfolio_frame["rebalance_gap_days"], errors="coerce").dropna()
        if not gaps.empty:
            gap_days = int(round(float(gaps.iloc[0])))

    last_rebalance_date: str | None = None
    if rebalance_event_log_csv is not None and rebalance_event_log_csv.exists():
        events = read_csv(rebalance_event_log_csv)
        if "datetime" in events.columns and not events.empty:
            events = events.copy()
            events["date_str"] = events["datetime"].astype(str).str.slice(0, 10)
            events = events.loc[events["date_str"].le(report_date)].copy()
            if "is_rebalance_day" in events.columns:
                events = events.loc[truthy_series(events["is_rebalance_day"])]
            if not events.empty:
                last_rebalance_date = str(events.sort_values("date_str")["date_str"].iloc[-1])
    return is_rebalance, reason, gap_days, last_rebalance_date


def build_holdings_note(
    holdings: pd.DataFrame,
    cash_weight: float,
    is_rebalance: bool,
    reason: str,
    gap_days: int | None,
    last_rebalance_date: str | None,
) -> str:
    weight_sum = float(holdings["actual_weight"].sum()) if not holdings.empty else 0.0
    if is_rebalance:
        status = "当日**调仓日**"
        if reason:
            status += f"（原因：{reason}）"
    else:
        if last_rebalance_date and gap_days is not None:
            status = f"当日**非调仓日**（距上次调仓 {last_rebalance_date} 已 {gap_days} 个交易日），持仓为 carry-forward"
        elif gap_days is not None:
            status = f"当日**非调仓日**（rebalance_gap_days={gap_days}），持仓为 carry-forward"
        else:
            status = "当日**非调仓日**，持仓为 carry-forward"

    note = (
        f"> {len(holdings)} 只持仓权重合计 ≈ {format_weight_pct(weight_sum, 2)}"
        f"（现金约 {format_weight_pct(cash_weight, 3)}）。{status}。"
    )

    if not holdings.empty:
        top = holdings.sort_values("mean_prediction", ascending=False).iloc[0]
        positive = holdings.loc[holdings["mean_prediction"] > 0].sort_values("mean_prediction")
        fragments: list[str] = []
        if len(positive) >= 2:
            lowest_positive = positive.iloc[0]
            ratio = top["mean_prediction"] / lowest_positive["mean_prediction"]
            if math.isfinite(ratio) and ratio > 1:
                ratio_text = f"{ratio:.0f}" if ratio >= 10 else f"{ratio:.1f}"
                fragments.append(
                    f"μ 最高的 {top['code_norm']}（{top.get('name', '')}）与持仓中最低正 μ 的 "
                    f"{lowest_positive['code_norm']}（{lowest_positive.get('name', '')}）之间，缩放后 μ 相差约 **{ratio_text} 倍**"
                )
        negative = holdings.loc[holdings["mean_prediction"] < 0].sort_values("mean_prediction")
        if not negative.empty:
            worst = negative.iloc[0]
            fragments.append(
                f"{worst['code_norm']} 当日 μ 为负（排名第 {int(worst['mu_rank'])}）仍持有 "
                f"{format_weight_pct(worst['actual_weight'], 2)}，存在明显 μ–持仓错位"
            )
        if fragments:
            note += "" + "；".join(fragments) + "。"
    return note


def build_nonholdings_note(
    candidates: pd.DataFrame,
    nonholdings: pd.DataFrame,
    input_paths: InputPaths,
) -> str:
    note = (
        f"> **数据来源**：`{input_paths.audit_dir}/` + bridge monitor；"
        f"`forecast μ` = bridge `forecast_mean_return`（**日度**稳健均值，不是近 20 日累计 realized）；"
        f"`缩放后 μ` = bridge `forecast_weight_mean_used`（经 shrink/overlay 后送入 MVO 的日度 μ，与 audit `mean_prediction` 同源）；"
        f"μ 排名按 audit `mean_prediction`；`rank_pct` = `mean_rank_pct_desc`（{len(candidates)} 只候选池内百分位，越小表示 μ 越高）。"
        f"近 20 日**已实现**收益不在本表，见 `fund_pool_builder` 选池审查 `rep_lag.csv` 的 `selected_20d_ret` / `best_20d_ret`。"
    )
    focus_parts: list[str] = []
    top = candidates.sort_values("mu_rank").iloc[0] if not candidates.empty else None
    if top is not None and top["actual_weight"] <= HOLDING_WEIGHT_EPS:
        target = finite_or_none(top.get("mvo_target_weight_pre_l3"))
        if target is None or abs(target) <= HOLDING_WEIGHT_EPS:
            target = finite_or_none(top.get("l3_model_target_weight"))
        if target is not None and abs(target) > HOLDING_WEIGHT_EPS:
            focus_parts.append(
                f"μ 第 1 名 {top['code_norm']}（MVO 目标 {format_compact_weight_pct(target)} 但未执行调仓）"
            )
        else:
            focus_parts.append(f"μ 第 1 名 {top['code_norm']} 未持仓")

    top_positive_nonheld = nonholdings.loc[nonholdings["mean_prediction"] > 0].sort_values("mu_rank")
    if not top_positive_nonheld.empty:
        extras = [
            f"第 {int(row['mu_rank'])} 名 {row['code_norm']}"
            for _, row in top_positive_nonheld.head(3).iterrows()
            if top is None or row["code_norm"] != top["code_norm"]
        ]
        if extras:
            focus_parts.append("、".join(extras) + " 均未持仓")

    mvo_flag = truthy_series(candidates["mvo_selected_flag"]) if "mvo_selected_flag" in candidates.columns else pd.Series(False, index=candidates.index)
    mvo_not_held = candidates.loc[mvo_flag & (candidates["actual_weight"] <= HOLDING_WEIGHT_EPS)].sort_values("mu_rank")
    if not mvo_not_held.empty:
        codes = [str(code) for code in mvo_not_held["code_norm"].head(4)]
        focus_parts.append("MVO 选中但 adaptive 未跟仓的还有 " + "、".join(codes) + " 等")

    if focus_parts:
        note += "**重点关注**：" + "；".join(focus_parts) + "。"
    return note


def render_report(
    candidates: pd.DataFrame,
    holdings: pd.DataFrame,
    nonholdings: pd.DataFrame,
    cash_weight: float,
    input_paths: InputPaths,
    portfolio_frame: pd.DataFrame,
    report_date: str,
) -> str:
    table1_columns = [
        ("μ排名", ":---:"),
        ("代码", ":-------"),
        ("基金简称", ":--------------------"),
        ("theme", ":------------"),
        ("forecast μ", "---------:"),
        ("缩放后 μ", "---------:"),
        ("rank_pct", ":------:"),
        ("实际权重", ":------:"),
    ]
    table2_columns = table1_columns[:-1]

    is_rebalance, reason, gap_days, last_rebalance_date = get_rebalance_metadata(
        portfolio_frame,
        input_paths.rebalance_event_log_csv,
        report_date,
    )

    parts = [
        f"## 表 1：{len(holdings)} 只持仓标的",
        "",
        "按 μ 排名降序排列：",
        "",
        markdown_table(rows_for_table(holdings, include_actual_weight=True), table1_columns),
        "",
        build_holdings_note(
            holdings,
            cash_weight,
            is_rebalance,
            reason,
            gap_days,
            last_rebalance_date,
        ),
        "",
        "------",
        "",
        f"## 表 2：{len(nonholdings)} 只未持仓标的",
        "",
        f"按 μ 排名降序排列（候选池 {len(candidates)} 只）：",
        "",
        markdown_table(rows_for_table(nonholdings, include_actual_weight=False), table2_columns),
        "",
        build_nonholdings_note(candidates, nonholdings, input_paths),
        "",
    ]
    return "\n".join(parts)


def candidate_base_dirs(base_dir: Path) -> list[Path]:
    resolved = base_dir.expanduser().resolve()
    candidates = [resolved]
    try:
        rel = resolved.relative_to(TEMP_DIR)
        alt_roots = [Path.home() / "etf-daily-output", TEMP_DIR.parent / "temp"]
        for alt_root in alt_roots:
            alt = alt_root.expanduser().resolve() / rel
            if alt not in candidates:
                candidates.append(alt)
    except ValueError:
        pass
    return candidates


def resolve_existing_base_dir(base_dir: Path) -> Path:
    candidates = candidate_base_dirs(base_dir)
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if any(
            path.name.startswith(("bridge_", "audit_", "parameter_effect_audit_", "monitor_", "mu_monitor_report_"))
            for path in candidate.iterdir()
        ):
            if candidate != candidates[0]:
                print(f"[INFO] Using alternate base dir: {candidate}")
            return candidate
    return candidates[0]


def main() -> None:
    args = parse_args()
    base_dir = resolve_existing_base_dir(args.base_dir)
    if args.date is None:
        compact_date = infer_latest_compact_date(base_dir)
        report_date, compact_date = normalize_report_date(compact_date)
    else:
        report_date, compact_date = normalize_report_date(args.date)

    input_paths = resolve_input_paths(
        base_dir,
        compact_date,
        args.preferred_start,
        audit_dir_override=args.audit_dir,
        ensure_audit=args.ensure_audit,
        audit_level=args.audit_level,
    )
    name_map, theme_map = load_name_and_theme_maps(
        input_paths.with_name_csv,
        report_date,
        args.fund_list_csv.expanduser().resolve() if args.fund_list_csv else None,
    )
    portfolio_frame, cash_weight, weight_col = load_portfolio_weights(input_paths.portfolio_weights_csv, report_date)
    candidates = build_candidate_panel(
        input_paths.layer_quality_csv,
        input_paths.bridge_mean_monitor_csv,
        portfolio_frame,
        report_date,
        name_map,
        theme_map,
    )
    holdings = candidates.loc[candidates["actual_weight"] > HOLDING_WEIGHT_EPS].sort_values(
        "mean_prediction", ascending=False
    )
    nonholdings = candidates.loc[candidates["actual_weight"] <= HOLDING_WEIGHT_EPS].sort_values(
        "mean_prediction", ascending=False
    )

    output = args.output
    if output is None:
        output = NOTES_DIR / f"{compact_date}.md"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    report = render_report(
        candidates=candidates,
        holdings=holdings,
        nonholdings=nonholdings,
        cash_weight=cash_weight,
        input_paths=input_paths,
        portfolio_frame=portfolio_frame,
        report_date=report_date,
    )
    output.write_text(report, encoding="utf-8")

    print(f"wrote: {output}")
    print(f"date: {report_date}")
    print(f"audit_dir: {input_paths.audit_dir}")
    print(f"actual_weight_column: {weight_col}")
    print(f"candidate_count: {len(candidates)}")
    print(f"holding_count: {len(holdings)}")
    print(f"nonholding_count: {len(nonholdings)}")
    print(f"cash_weight: {cash_weight:.8f}")


if __name__ == "__main__":
    main()
