from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import DEFAULT_FUND_LIST_CSV


EPS = 1e-12
FOCUS_DATES = ("2026-01-12", "2026-01-14", "2026-01-29", "2026-03-06")


@dataclass(frozen=True)
class AuditOutputPaths:
    output_dir: Path
    simple_latest_weights: Path
    simple_daily_summary: Path
    simple_instrument_drilldown: Path
    simple_manifest_json: Path
    readme_md: Path
    instrument_daily: Path
    decision_panel: Path
    portfolio_weights: Path
    latest_portfolio_weights: Path
    rebalance_event_log: Path
    daily_summary: Path
    instrument_summary: Path
    covariance_daily: Path
    cluster_mapping_coverage: Path
    layer_quality_panel: Path
    layer_quality_daily_summary: Path
    layer_quality_instrument_summary: Path
    layer_quality_latest_pending: Path
    layer_quality_schema: Path
    layer_quality_report: Path
    metadata_json: Path
    schema_json: Path
    report_md: Path


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _safe_numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column in frame.columns:
        return _numeric(frame[column])
    return pd.Series(default, index=frame.index, dtype="float64")


def _coalesce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for column in columns:
        if column in frame.columns:
            result = result.combine_first(_numeric(frame[column]))
    return result


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype="bool")
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    text = values.astype(str).str.strip().str.lower()
    return text.isin({"1", "1.0", "true", "t", "yes", "y"})


def _normalize_code(value: object) -> str:
    text = str(value).strip().upper()
    if text.startswith(("SH", "SZ")):
        return text
    if len(text) == 6 and text.isdigit():
        prefix = "SH" if text.startswith(("5", "6")) else "SZ"
        return f"{prefix}{text}"
    return text


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        numeric_value = value.item()
        if isinstance(numeric_value, float) and not np.isfinite(numeric_value):
            return None
        return numeric_value
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _markdown_table(frame: pd.DataFrame, max_rows: int = 12) -> str:
    if frame.empty:
        return "No rows."
    working = frame.head(max_rows).copy()
    columns = [str(column) for column in working.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in working.iterrows():
        rendered = []
        for column in working.columns:
            value = row.get(column, "")
            if isinstance(value, float):
                rendered.append("" if not np.isfinite(value) else f"{value:.6g}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def _format_pct(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "待回填"
    return f"{value * 100:.3f}%"


def _format_count(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "待回填"
    return str(int(numeric))


def _format_instrument_label(row: pd.Series) -> str:
    instrument = str(row.get("instrument", "")).strip()
    name = str(row.get("name", "")).strip() or str(row.get("cluster_mapping_name_latest", "")).strip()
    return f"{instrument}({name})" if name else instrument


def _load_fund_list_name_map(fund_list_csv: str | Path | None) -> dict[str, str]:
    if not fund_list_csv:
        return {}
    path = Path(fund_list_csv).expanduser()
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception:
        return {}
    code_col = next((column for column in ["基金代码", "code", "instrument", "symbol"] if column in frame.columns), None)
    name_col = next((column for column in ["基金简称", "基金名称", "name", "short_name"] if column in frame.columns), None)
    if code_col is None or name_col is None:
        return {}
    name_map: dict[str, str] = {}
    for code, name in zip(frame[code_col], frame[name_col]):
        normalized = _normalize_code(code)
        label = str(name).strip()
        if normalized and label:
            name_map[normalized] = label
    return name_map


def _add_name_column_after_instrument(frame: pd.DataFrame, name_map: dict[str, str]) -> pd.DataFrame:
    if frame.empty or "instrument" not in frame.columns:
        return frame
    working = frame.copy()
    fallback = (
        working.get("cluster_mapping_name_latest", pd.Series("", index=working.index, dtype=object)).fillna("").astype(str)
    )
    mapped = working["instrument"].map(lambda value: name_map.get(_normalize_code(value), ""))
    name_values = mapped.fillna("").astype(str)
    name_values = name_values.where(name_values.str.strip() != "", fallback)
    if "name" in working.columns:
        working = working.drop(columns=["name"])
    insert_at = list(working.columns).index("instrument") + 1
    working.insert(insert_at, "name", name_values)
    return working


def _build_layer_quality_narrative(daily_summary: pd.DataFrame, instrument_summary: pd.DataFrame) -> list[str]:
    if daily_summary.empty:
        return ["- 当前没有可写入的层级质量数据。"]

    lines: list[str] = []
    recent_days = daily_summary.sort_values("datetime").tail(2)
    for _, row in recent_days.iterrows():
        date_value = pd.Timestamp(row["datetime"]).strftime("%Y-%m-%d")
        universe_count = _format_count(row.get("pool_universe_count"))
        selected_count = _format_count(row.get("pool_selected_count"))
        unselected_count = _format_count(row.get("pool_unselected_count"))
        selected_transmission = _format_pct(float(row.get("pool_selected_active_transmitted_rate", np.nan)))
        unselected_transmission = _format_pct(float(row.get("pool_unselected_active_transmitted_rate", np.nan)))
        spread = float(row.get("pool_selected_vs_unselected_spread", np.nan))
        selected_return = float(row.get("pool_selected_equal_weight_next_return", np.nan))
        unselected_return = float(row.get("pool_unselected_equal_weight_next_return", np.nan))
        realized_rows = float(row.get("realized_rows", np.nan))

        if np.isfinite(realized_rows) and realized_rows > 0 and np.isfinite(spread):
            if spread > EPS:
                conclusion = "selected 池事后优于未选池"
            elif spread < -EPS:
                conclusion = "selected 池事后并未优于未选池"
            else:
                conclusion = "selected 池与未选池基本持平"
            lines.append(
                f"- {date_value}：full universe 共 {universe_count} 只，其中 selected {selected_count} 只、unselected {unselected_count} 只。"
                f"selected 等权次日收益为 {_format_pct(selected_return)}，unselected 为 {_format_pct(unselected_return)}，spread 为 {_format_pct(spread)}，"
                f"说明这一天 {conclusion}。同时，selected 的 active transmission rate 为 {selected_transmission}，"
                f"显著高于 unselected 的 {unselected_transmission}。"
            )
        else:
            lines.append(
                f"- {date_value}：当前仍处于 pending 状态，尚不能比较 selected 与 unselected 的 realized next return。"
                f"这一天只能先看覆盖与传导：full universe 共 {universe_count} 只，其中 selected {selected_count} 只、"
                f"unselected {unselected_count} 只；selected transmission rate 为 {selected_transmission}，"
                f"unselected 为 {unselected_transmission}。"
            )

    if instrument_summary.empty:
        lines.append("- 当前没有可写入的标的级 summary，无法补充最强未选池和最弱已选池。")
        return lines

    weakest_selected = instrument_summary[instrument_summary["pool_selected_flag"].eq(True)].sort_values("avg_next_return").head(3)
    strongest_unselected = instrument_summary[instrument_summary["pool_selected_flag"].eq(False)].sort_values(
        "avg_next_return", ascending=False
    ).head(3)
    if not strongest_unselected.empty:
        labels = [
            _format_instrument_label(row)
            for _, row in strongest_unselected.iterrows()
        ]
        lines.append(f"- 当前最强未选池代表主要是：{'、'.join(labels)}。")
    if not weakest_selected.empty:
        labels = [
            _format_instrument_label(row)
            for _, row in weakest_selected.iterrows()
        ]
        lines.append(f"- 当前最弱已选池代表主要是：{'、'.join(labels)}。")
    return lines


def _first_finite(values: pd.Series) -> float:
    finite = _numeric(values).dropna()
    return float(finite.iloc[0]) if not finite.empty else float("nan")


def _first_text(values: pd.Series) -> str:
    if values is None:
        return ""
    text = values.dropna().astype(str).str.strip()
    text = text[text != ""]
    return str(text.iloc[0]) if not text.empty else ""


def _safe_corr(first: pd.Series, second: pd.Series, method: str = "pearson") -> float:
    pair = pd.concat([_numeric(first), _numeric(second)], axis=1).dropna()
    if len(pair) < 2:
        return float("nan")
    if float(pair.iloc[:, 0].std(ddof=0)) <= EPS or float(pair.iloc[:, 1].std(ddof=0)) <= EPS:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def _effective_holdings(weights: pd.Series) -> float:
    positive = _numeric(weights).fillna(0.0).clip(lower=0.0)
    total = float(positive.sum())
    if total <= EPS:
        return float("nan")
    normalized = positive / total
    herfindahl = float(np.square(normalized.to_numpy(dtype=float)).sum())
    return 1.0 / max(herfindahl, EPS)


def _format_top_contributors(day_frame: pd.DataFrame, column: str, ascending: bool, limit: int = 5) -> str:
    if column not in day_frame.columns:
        return ""
    working = day_frame[["instrument", column]].copy()
    working[column] = _numeric(working[column])
    working = working.dropna().sort_values(column, ascending=ascending).head(limit)
    return "|".join(f"{row['instrument']}:{row[column]:.6g}" for _, row in working.iterrows())


def _resolve_output_paths(output_dir: str | Path) -> AuditOutputPaths:
    output_path = Path(output_dir).expanduser().resolve()
    return AuditOutputPaths(
        output_dir=output_path,
        simple_latest_weights=output_path / "portfolio_latest.csv",
        simple_daily_summary=output_path / "decision_daily.csv",
        simple_instrument_drilldown=output_path / "decision_inst.csv",
        simple_manifest_json=output_path / "run_manifest.json",
        readme_md=output_path / "README.md",
        instrument_daily=output_path / "param_inst_daily.csv",
        decision_panel=output_path / "decision_panel.csv",
        portfolio_weights=output_path / "portfolio_daily.csv",
        latest_portfolio_weights=output_path / "latest_weights.csv",
        rebalance_event_log=output_path / "rebalance_log.csv",
        daily_summary=output_path / "param_daily.csv",
        instrument_summary=output_path / "param_inst.csv",
        covariance_daily=output_path / "cov_daily.csv",
        cluster_mapping_coverage=output_path / "cluster_coverage.csv",
        layer_quality_panel=output_path / "layer_panel.csv",
        layer_quality_daily_summary=output_path / "layer_daily.csv",
        layer_quality_instrument_summary=output_path / "layer_inst.csv",
        layer_quality_latest_pending=output_path / "layer_latest_pending.csv",
        layer_quality_schema=output_path / "layer_schema.json",
        layer_quality_report=output_path / "layer_report.md",
        metadata_json=output_path / "audit_metadata.json",
        schema_json=output_path / "decision_schema.json",
        report_md=output_path / "param_report.md",
    )


def _load_cluster_mapping(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    mapping_path = Path(path).expanduser().resolve()
    if not mapping_path.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for rank, line in enumerate(mapping_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        code = _normalize_code(parts[0])
        if not code:
            continue
        rows.append(
            {
                "instrument": code,
                "cluster_mapping_rank": rank,
                "cluster_mapping_selected_flag": True,
                "cluster_mapping_start_date": parts[1] if len(parts) > 1 else "",
                "cluster_mapping_end_date": parts[2] if len(parts) > 2 else "",
            }
        )
    return pd.DataFrame(rows).drop_duplicates("instrument", keep="first")


def _load_cluster_mapping_full_csv(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    mapping_path = Path(path).expanduser().resolve()
    if not mapping_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(mapping_path, low_memory=False)
    code_column = next((column for column in ["instrument", "code", "symbol", "ticker"] if column in frame.columns), None)
    if code_column is None:
        raise ValueError(f"cluster_mapping full CSV missing code column: {mapping_path}")
    frame = frame.copy()
    frame["instrument"] = frame[code_column].map(_normalize_code)
    frame = frame[frame["instrument"].astype(str).str.strip() != ""].copy()
    if "selected" in frame.columns:
        frame["cluster_mapping_full_selected_flag"] = _bool_series(frame, "selected")
    elif "cluster_mapping_selected_flag" in frame.columns:
        frame["cluster_mapping_full_selected_flag"] = _bool_series(frame, "cluster_mapping_selected_flag")
    else:
        frame["cluster_mapping_full_selected_flag"] = False
    frame["cluster_mapping_cluster"] = _numeric(frame["cluster"]) if "cluster" in frame.columns else np.nan
    frame["cluster_mapping_reason"] = frame["reason"].fillna("").astype(str) if "reason" in frame.columns else ""
    frame["cluster_mapping_name"] = frame["name"].fillna("").astype(str) if "name" in frame.columns else ""
    keep_columns = [
        "instrument",
        "cluster_mapping_full_selected_flag",
        "cluster_mapping_cluster",
        "cluster_mapping_reason",
        "cluster_mapping_name",
    ]
    return (
        frame[keep_columns]
        .sort_values(["cluster_mapping_full_selected_flag", "instrument"], ascending=[False, True])
        .drop_duplicates("instrument", keep="first")
        .reset_index(drop=True)
    )


def _combine_cluster_mappings(selected_mapping: pd.DataFrame, full_mapping: pd.DataFrame) -> pd.DataFrame:
    if full_mapping.empty:
        return selected_mapping.copy()

    base = full_mapping.copy()
    base["cluster_mapping_full_selected_flag"] = base["cluster_mapping_full_selected_flag"].eq(True)
    if selected_mapping.empty:
        base["cluster_mapping_rank"] = np.nan
        base["cluster_mapping_start_date"] = ""
        base["cluster_mapping_end_date"] = ""
        base["cluster_mapping_selected_from_txt_flag"] = False
    else:
        selected_meta = selected_mapping[
            [
                "instrument",
                "cluster_mapping_rank",
                "cluster_mapping_start_date",
                "cluster_mapping_end_date",
            ]
        ].copy()
        selected_meta["cluster_mapping_selected_from_txt_flag"] = True
        base = base.merge(selected_meta, on="instrument", how="left")

        known_instruments = set(base["instrument"].astype(str))
        missing_selected = selected_mapping.loc[~selected_mapping["instrument"].astype(str).isin(known_instruments)].copy()
        if not missing_selected.empty:
            missing_selected["cluster_mapping_full_selected_flag"] = True
            missing_selected["cluster_mapping_selected_from_txt_flag"] = True
            missing_selected["cluster_mapping_cluster"] = np.nan
            missing_selected["cluster_mapping_reason"] = ""
            missing_selected["cluster_mapping_name"] = ""
            base = pd.concat(
                [base, missing_selected.reindex(columns=base.columns)],
                ignore_index=True,
                sort=False,
            )

    base["cluster_mapping_selected_flag"] = (
        base["cluster_mapping_full_selected_flag"] | base["cluster_mapping_selected_from_txt_flag"].eq(True)
    )
    base["cluster_mapping_reason"] = base["cluster_mapping_reason"].fillna("").astype(str)
    base["cluster_mapping_name"] = base["cluster_mapping_name"].fillna("").astype(str)
    base["cluster_mapping_start_date"] = base["cluster_mapping_start_date"].fillna("").astype(str)
    base["cluster_mapping_end_date"] = base["cluster_mapping_end_date"].fillna("").astype(str)
    return base.sort_values(["cluster_mapping_selected_flag", "cluster_mapping_rank", "instrument"], ascending=[False, True, True]).drop_duplicates(
        "instrument", keep="first"
    )


def _load_bridge_frame(path: str | Path, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    bridge_path = Path(path).expanduser().resolve()
    if not bridge_path.exists():
        raise FileNotFoundError(f"bridge CSV not found: {bridge_path}")
    frame = pd.read_csv(bridge_path, parse_dates=["datetime"], low_memory=False)
    if "instrument" not in frame.columns or "datetime" not in frame.columns:
        raise ValueError(f"bridge CSV missing instrument/datetime columns: {bridge_path}")
    frame["instrument"] = frame["instrument"].map(_normalize_code)
    if start_date:
        frame = frame[frame["datetime"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        frame = frame[frame["datetime"] <= pd.Timestamp(end_date)].copy()
    frame = frame.sort_values(["datetime", "instrument"]).reset_index(drop=True)
    frame["_bridge_row_id"] = np.arange(len(frame), dtype="int64")
    return frame


def _load_raw_presence(path: str | Path | None, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    raw_path = Path(path).expanduser().resolve()
    if not raw_path.exists():
        return pd.DataFrame()
    raw_frame = pd.read_csv(raw_path, parse_dates=["datetime"], usecols=lambda column: column in {"instrument", "datetime"})
    if "instrument" not in raw_frame.columns or "datetime" not in raw_frame.columns:
        return pd.DataFrame()
    raw_frame["instrument"] = raw_frame["instrument"].map(_normalize_code)
    if start_date:
        raw_frame = raw_frame[raw_frame["datetime"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        raw_frame = raw_frame[raw_frame["datetime"] <= pd.Timestamp(end_date)].copy()
    raw_frame = raw_frame.drop_duplicates(["datetime", "instrument"])
    raw_frame["in_raw_panel_flag"] = True
    return raw_frame


def _load_backtest_positions(path: str | Path | None, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    positions_path = Path(path).expanduser().resolve()
    if not positions_path.exists():
        return pd.DataFrame()
    header = pd.read_csv(positions_path, nrows=0).columns.tolist()
    keep_columns = [
        "code",
        "instrument",
        "datetime",
        "w",
        "当日收盘价市值",
        "当日股份",
        "明日收盘价市值",
        "当日实际买入金额",
        "当日剩余现金",
        "当期起始总资产",
        "当日组合总资产",
        "估值日期",
        "is_rebalance_day",
        "rebalance_reason",
        "rebalance_gap_days",
        "trigger_portfolio_drift",
        "trigger_exposure_change",
        "trigger_macro_budget_change",
        "trigger_macro_state_flip",
        "trigger_risk_scale_change",
        "trigger_adaptive_score",
        "trigger_expected_return_gain",
        "trigger_variance_change",
        "trigger_correlation_improvement",
        "trigger_turnover",
        "adaptive_score_pass",
        "rebalance_candidate_reason",
        "当日实际持仓权重",
        "is_synthetic_carry",
        "当日组合盈亏",
    ]
    usecols = [column for column in keep_columns if column in header]
    if "datetime" not in usecols or not ({"code", "instrument"} & set(usecols)):
        return pd.DataFrame()
    parse_dates = [column for column in ["datetime", "估值日期"] if column in usecols]
    frame = pd.read_csv(positions_path, usecols=usecols, parse_dates=parse_dates, low_memory=False)
    if "instrument" not in frame.columns and "code" in frame.columns:
        frame = frame.rename(columns={"code": "instrument"})
    if "instrument" not in frame.columns:
        return pd.DataFrame()
    frame["instrument"] = frame["instrument"].map(_normalize_code)
    if start_date:
        frame = frame[frame["datetime"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        frame = frame[frame["datetime"] <= pd.Timestamp(end_date)].copy()
    rename_map = {
        "w": "backtest_input_model_weight_pre_l3",
        "当日收盘价市值": "adaptive_position_market_value_close",
        "当日股份": "adaptive_shares",
        "明日收盘价市值": "adaptive_position_market_value_next_close",
        "当日实际买入金额": "adaptive_actual_buy_amount",
        "当日剩余现金": "adaptive_row_cash_amount",
        "当期起始总资产": "adaptive_start_nav_from_positions",
        "当日组合总资产": "adaptive_end_nav_from_positions",
        "估值日期": "adaptive_valuation_date",
        "当日实际持仓权重": "adaptive_actual_weight_after_trade",
        "当日组合盈亏": "adaptive_position_pnl",
    }
    frame = frame.rename(columns=rename_map)
    numeric_columns = [
        "backtest_input_model_weight_pre_l3",
        "adaptive_position_market_value_close",
        "adaptive_shares",
        "adaptive_position_market_value_next_close",
        "adaptive_actual_buy_amount",
        "adaptive_row_cash_amount",
        "adaptive_start_nav_from_positions",
        "adaptive_end_nav_from_positions",
        "rebalance_gap_days",
        "trigger_portfolio_drift",
        "trigger_exposure_change",
        "trigger_macro_budget_change",
        "trigger_macro_state_flip",
        "trigger_risk_scale_change",
        "trigger_adaptive_score",
        "trigger_expected_return_gain",
        "trigger_variance_change",
        "trigger_correlation_improvement",
        "trigger_turnover",
        "adaptive_actual_weight_after_trade",
        "adaptive_position_pnl",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = _numeric(frame[column])
    for column in ["is_rebalance_day", "adaptive_score_pass", "is_synthetic_carry"]:
        if column in frame.columns:
            frame[column] = _bool_series(frame, column)
    return frame.sort_values(["datetime", "instrument"]).drop_duplicates(["datetime", "instrument"], keep="last")


def _load_daily_nav(path: str | Path | None, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    nav_path = Path(path).expanduser().resolve()
    if not nav_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(nav_path, parse_dates=["datetime"], low_memory=False)
    if "datetime" not in frame.columns:
        return pd.DataFrame()
    if start_date:
        frame = frame[frame["datetime"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        frame = frame[frame["datetime"] <= pd.Timestamp(end_date)].copy()
    rename_map = {
        "start_nav": "adaptive_daily_start_nav",
        "nav": "adaptive_daily_nav",
        "next_nav": "adaptive_daily_next_nav",
        "holding_mv": "adaptive_daily_holding_mv",
        "cash": "adaptive_cash_amount",
        "ret": "adaptive_daily_return",
    }
    frame = frame.rename(columns=rename_map)
    for column in frame.columns:
        if column == "datetime" or column in {"rebalance_reason", "rebalance_candidate_reason"}:
            continue
        if frame[column].dtype == bool:
            continue
        converted = _numeric(frame[column])
        if converted.notna().sum() > 0 or frame[column].isna().all():
            frame[column] = converted
    if "is_rebalance_day" in frame.columns:
        frame["daily_is_rebalance_day"] = _bool_series(frame, "is_rebalance_day")
    if "adaptive_score_pass" in frame.columns:
        frame["daily_adaptive_score_pass"] = _bool_series(frame, "adaptive_score_pass")
    nav = _safe_numeric(frame, "adaptive_daily_nav")
    cash = _safe_numeric(frame, "adaptive_cash_amount")
    frame["adaptive_cash_weight_by_date"] = np.where(nav.abs() > EPS, cash / nav, np.nan)
    keep = [
        "datetime",
        "adaptive_daily_start_nav",
        "adaptive_daily_nav",
        "adaptive_daily_next_nav",
        "adaptive_daily_holding_mv",
        "adaptive_cash_amount",
        "adaptive_cash_weight_by_date",
        "adaptive_daily_return",
        "daily_is_rebalance_day",
        "daily_adaptive_score_pass",
        "trigger_regime_flip",
        "trigger_expected_return_current",
        "trigger_expected_return_target",
        "trigger_variance_current",
        "trigger_variance_target",
        "trigger_avg_corr_current",
        "trigger_avg_corr_target",
        "trigger_risk_adjustment",
        "trigger_diversification_bonus",
        "trigger_turnover_penalty",
    ]
    keep = [column for column in keep if column in frame.columns]
    return frame[keep].sort_values("datetime").drop_duplicates("datetime", keep="last")


def _extract_route_a_parts(path: Path, prefix: str) -> tuple[str, str, str] | None:
    stem = path.stem
    if not stem.startswith(prefix):
        return None
    parts = stem.rsplit("_", 2)
    if len(parts) != 3:
        return None
    instrument = parts[0][len(prefix) :]
    return _normalize_code(instrument), parts[1], parts[2]


def _prefix_route_a_columns(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    working = frame.copy()
    if "forecast_date" not in working.columns or "target_date" not in working.columns:
        return pd.DataFrame()
    working["forecast_date"] = pd.to_datetime(working["forecast_date"], errors="coerce")
    working["target_date"] = pd.to_datetime(working["target_date"], errors="coerce")
    rename_map = {"forecast_date": "datetime", "target_date": "route_a_target_date"}
    for column in working.columns:
        if column in {"instrument", "forecast_date", "target_date"}:
            continue
        rename_map[column] = f"route_a_{source}_{column}"
    return working.rename(columns=rename_map)


def _load_route_a_panel(route_a_output_dir: str | Path | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if route_a_output_dir is None:
        return pd.DataFrame(), {"status": "route_a_output_dir_not_provided"}
    output_path = Path(route_a_output_dir).expanduser().resolve()
    if not output_path.exists():
        return pd.DataFrame(), {"status": "route_a_output_dir_missing", "path": str(output_path)}

    manifest_path = output_path / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {"manifest_error": "invalid_json"}

    params_frames: list[pd.DataFrame] = []
    for csv_path in sorted(output_path.glob("route_a_params_*.csv")):
        parts = _extract_route_a_parts(csv_path, "route_a_params_")
        if parts is None:
            continue
        instrument, route_start_label, route_end_label = parts
        frame = pd.read_csv(csv_path, parse_dates=["forecast_date", "target_date"])
        frame["instrument"] = instrument
        frame["route_a_file_start_label"] = route_start_label
        frame["route_a_file_end_label"] = route_end_label
        params_frames.append(_prefix_route_a_columns(frame, "params"))

    eval_frames: list[pd.DataFrame] = []
    for csv_path in sorted(output_path.glob("route_a_eval_daily_*.csv")):
        parts = _extract_route_a_parts(csv_path, "route_a_eval_daily_")
        if parts is None:
            continue
        instrument, route_start_label, route_end_label = parts
        frame = pd.read_csv(csv_path, parse_dates=["forecast_date", "target_date"])
        frame["instrument"] = instrument
        frame["route_a_file_start_label"] = route_start_label
        frame["route_a_file_end_label"] = route_end_label
        eval_frames.append(_prefix_route_a_columns(frame, "eval"))

    params_panel = pd.concat(params_frames, ignore_index=True) if params_frames else pd.DataFrame()
    eval_panel = pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame()
    join_keys = ["instrument", "datetime", "route_a_target_date"]
    if params_panel.empty and eval_panel.empty:
        return pd.DataFrame(), {
            "status": "route_a_files_missing",
            "path": str(output_path),
            "manifest": manifest,
        }
    if params_panel.empty:
        panel = eval_panel
    elif eval_panel.empty:
        panel = params_panel
    else:
        panel = params_panel.merge(eval_panel, on=join_keys, how="outer", suffixes=("", "_eval_dup"))
    panel = panel.sort_values(join_keys).drop_duplicates(["instrument", "datetime"], keep="last")
    return panel, {
        "status": "ok",
        "path": str(output_path),
        "manifest": manifest,
        "params_files": len(params_frames),
        "eval_files": len(eval_frames),
        "rows": int(panel.shape[0]),
    }


def _parse_float_vector(value: object) -> list[float]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        return [float(item) for item in value.split(",") if item != ""]
    except ValueError:
        return []


def _reconstruct_covariance(day_frame: pd.DataFrame) -> tuple[list[str], np.ndarray | None, str, str]:
    order_values = day_frame.get("forecast_cov_order", pd.Series(dtype=object)).dropna()
    if order_values.empty:
        return [], None, "", "missing_order"
    order = [item for item in str(order_values.iloc[0]).split("|") if item]
    if len(order) < 2:
        return order, None, "", "too_few_instruments"

    for matrix_column in ["forecast_cov_matrix_robust", "forecast_cov_matrix"]:
        if matrix_column not in day_frame.columns:
            continue
        matrix = np.full((len(order), len(order)), np.nan, dtype=float)
        index_by_instrument = {instrument: index for index, instrument in enumerate(order)}
        for row in day_frame.itertuples(index=False):
            instrument = str(getattr(row, "instrument"))
            row_index = index_by_instrument.get(instrument)
            if row_index is None:
                continue
            values = _parse_float_vector(getattr(row, matrix_column, ""))
            if len(values) == len(order):
                matrix[row_index, :] = values
        finite_rows = np.isfinite(matrix).all(axis=1)
        finite_cols = np.isfinite(matrix).all(axis=0)
        valid_mask = finite_rows & finite_cols
        if valid_mask.sum() < 2:
            continue
        valid_indices = np.where(valid_mask)[0]
        valid_order = [order[index] for index in valid_indices]
        valid_matrix = matrix[np.ix_(valid_indices, valid_indices)]
        valid_matrix = 0.5 * (valid_matrix + valid_matrix.T)
        diagonal = np.diag(valid_matrix).copy()
        np.fill_diagonal(valid_matrix, np.maximum(diagonal, EPS))
        return valid_order, valid_matrix, matrix_column, "ok"
    return order, None, "", "missing_matrix"


def _compute_covariance_outputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    instrument_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    bridge_only = frame[frame["in_bridge_flag"]].copy()
    for date_value, day_frame in bridge_only.groupby("datetime", sort=True):
        date_timestamp = pd.Timestamp(date_value)
        order, matrix, matrix_column, status = _reconstruct_covariance(day_frame)
        daily_row: dict[str, Any] = {
            "datetime": date_timestamp,
            "covariance_status": status,
            "covariance_source_col": matrix_column,
            "covariance_n_instruments": len(order),
        }
        if matrix is None:
            daily_rows.append(daily_row)
            continue

        diagonal = np.diag(matrix)
        std_values = np.sqrt(np.maximum(diagonal, EPS))
        corr = matrix / np.outer(std_values, std_values)
        corr = np.clip(corr, -1.0, 1.0)
        try:
            eigenvalues = np.linalg.eigvalsh(matrix)
            lambda_min = float(eigenvalues[0])
            lambda_max = float(eigenvalues[-1])
            cond_number = lambda_max / max(abs(lambda_min), EPS)
            effective_rank = float(np.square(eigenvalues.sum()) / max(np.square(eigenvalues).sum(), EPS))
        except np.linalg.LinAlgError:
            lambda_min = lambda_max = cond_number = effective_rank = float("nan")

        indexed_day = day_frame.set_index("instrument")
        forecast_weights = _numeric(indexed_day.get("forecast_w", pd.Series(dtype=float))).reindex(order).fillna(0.0)
        exposure_scale = _numeric(indexed_day.get("exposure_scale", pd.Series(dtype=float))).reindex(order).fillna(1.0)
        effective_weights = forecast_weights * exposure_scale
        weights_array = forecast_weights.to_numpy(dtype=float)
        effective_weights_array = effective_weights.to_numpy(dtype=float)
        matrix_weight_vector = matrix @ weights_array
        active_indices = np.where(weights_array > EPS)[0]
        port_var_pre_l3 = float(weights_array @ matrix @ weights_array)
        port_var_effective = float(effective_weights_array @ matrix @ effective_weights_array)

        daily_row.update(
            {
                "covariance_lambda_min": lambda_min,
                "covariance_lambda_max": lambda_max,
                "covariance_condition_number": cond_number,
                "covariance_effective_rank": effective_rank,
                "sigma_port_pre_l3": float(np.sqrt(max(port_var_pre_l3, 0.0))),
                "sigma_port_effective_l3": float(np.sqrt(max(port_var_effective, 0.0))),
                "covariance_trace": float(np.trace(matrix)),
            }
        )
        daily_rows.append(daily_row)

        for row_index, instrument in enumerate(order):
            partner_indices = np.array([index for index in range(len(order)) if index != row_index], dtype=int)
            active_partner_indices = np.array([index for index in active_indices if index != row_index], dtype=int)
            if active_partner_indices.size:
                avg_cov_to_active = float(np.nanmean(matrix[row_index, active_partner_indices]))
                avg_corr_to_active = float(np.nanmean(corr[row_index, active_partner_indices]))
            else:
                avg_cov_to_active = float("nan")
                avg_corr_to_active = float("nan")
            if partner_indices.size:
                partner_abs_corr = np.abs(corr[row_index, partner_indices])
                top_partner_index = int(partner_indices[int(np.nanargmax(partner_abs_corr))])
                top_partner = order[top_partner_index]
                max_corr_to_other = float(corr[row_index, top_partner_index])
                max_cov_to_other = float(matrix[row_index, top_partner_index])
            else:
                top_partner = ""
                max_corr_to_other = float("nan")
                max_cov_to_other = float("nan")
            instrument_rows.append(
                {
                    "datetime": date_timestamp,
                    "instrument": instrument,
                    "covariance_row_status": "ok",
                    "covariance_diag_from_matrix": float(diagonal[row_index]),
                    "covariance_row_norm": float(np.linalg.norm(matrix[row_index, :])),
                    "covariance_avg_cov_to_active": avg_cov_to_active,
                    "covariance_avg_corr_to_active": avg_corr_to_active,
                    "covariance_max_corr_partner": top_partner,
                    "covariance_max_corr_to_other": max_corr_to_other,
                    "covariance_max_cov_to_other": max_cov_to_other,
                    "covariance_weight_contribution": float(weights_array[row_index] * matrix_weight_vector[row_index]),
                }
            )
    return pd.DataFrame(instrument_rows), pd.DataFrame(daily_rows)


def _build_base_frame(
    bridge_frame: pd.DataFrame,
    cluster_mapping: pd.DataFrame,
    raw_presence: pd.DataFrame,
) -> pd.DataFrame:
    dates = sorted(bridge_frame["datetime"].dropna().unique())
    bridge_instruments = sorted(bridge_frame["instrument"].dropna().astype(str).unique())
    if cluster_mapping.empty:
        universe = bridge_instruments
    else:
        selected_instruments = cluster_mapping["instrument"].dropna().astype(str).tolist()
        bridge_extra = [instrument for instrument in bridge_instruments if instrument not in set(selected_instruments)]
        universe = selected_instruments + sorted(bridge_extra)
    base_index = pd.MultiIndex.from_product([dates, universe], names=["datetime", "instrument"])
    base = base_index.to_frame(index=False)
    if not cluster_mapping.empty:
        base = base.merge(cluster_mapping, on="instrument", how="left")
    else:
        base["cluster_mapping_rank"] = np.nan
        base["cluster_mapping_selected_flag"] = False
        base["cluster_mapping_start_date"] = ""
        base["cluster_mapping_end_date"] = ""
    base["cluster_mapping_selected_flag"] = base["cluster_mapping_selected_flag"].eq(True)

    working = base.merge(bridge_frame, on=["datetime", "instrument"], how="left")
    working["in_bridge_flag"] = working["_bridge_row_id"].notna()
    if raw_presence.empty:
        working["in_raw_panel_flag"] = working["in_bridge_flag"]
    else:
        working = working.merge(raw_presence, on=["datetime", "instrument"], how="left")
        working["in_raw_panel_flag"] = working["in_raw_panel_flag"].eq(True)
    return working


def _attach_route_a(working: pd.DataFrame, route_a_panel: pd.DataFrame, route_a_meta: dict[str, Any]) -> pd.DataFrame:
    if route_a_panel.empty:
        working["route_a_join_status"] = str(route_a_meta.get("status", "route_a_unavailable"))
        return working
    route_instruments = set(route_a_panel["instrument"].dropna().astype(str))
    route_keys = route_a_panel[["instrument", "datetime"]].drop_duplicates().copy()
    route_keys["_route_a_key_exists"] = True
    merged = working.merge(route_a_panel, on=["instrument", "datetime"], how="left")
    merged = merged.merge(route_keys, on=["instrument", "datetime"], how="left")
    joined = merged["_route_a_key_exists"].eq(True)
    known_instrument = merged["instrument"].isin(route_instruments)
    merged["route_a_join_status"] = np.select(
        [joined, known_instrument],
        ["joined", "missing_route_a_date"],
        default="missing_route_a_instrument",
    )
    merged = merged.drop(columns=["_route_a_key_exists"], errors="ignore")
    return merged


def _add_observed_effect_columns(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    forecast_weight = _safe_numeric(working, "forecast_w", 0.0).fillna(0.0)
    exposure_scale = _safe_numeric(working, "exposure_scale", 1.0).fillna(1.0)
    next_return = _safe_numeric(working, "next_return")
    optimizer_mean = _coalesce_numeric(
        working,
        ["forecast_weight_mean_used", "mu_used", "forecast_mean_return_robust", "forecast_mean_return"],
    )
    working["forecast_w"] = forecast_weight
    working["exposure_scale"] = exposure_scale
    working["optimizer_mean_used_for_audit"] = optimizer_mean
    working["pre_l3_weight"] = forecast_weight
    working["effective_weight"] = forecast_weight * exposure_scale
    working["next_return_used_for_attribution"] = next_return.fillna(0.0)
    working["expected_mean_contribution"] = working["effective_weight"] * optimizer_mean.fillna(0.0)
    working["realized_return_contribution"] = working["effective_weight"] * working["next_return_used_for_attribution"]
    working["forecast_error_next_return_minus_mean"] = next_return - optimizer_mean
    working["mean_signal_x_next_return"] = optimizer_mean * next_return
    valid_direction = optimizer_mean.notna() & next_return.notna() & (optimizer_mean.abs() > EPS) & (next_return.abs() > EPS)
    working["mean_direction_hit"] = np.where(valid_direction, np.sign(optimizer_mean) == np.sign(next_return), np.nan)

    mean_rank = optimizer_mean.groupby(working["datetime"]).rank(pct=True, ascending=False)
    mean_center = optimizer_mean - optimizer_mean.groupby(working["datetime"]).transform("mean")
    mean_std = optimizer_mean.groupby(working["datetime"]).transform("std").replace(0.0, np.nan)
    working["mean_used_rank_pct_desc"] = mean_rank
    working["mean_used_zscore_by_date"] = mean_center / mean_std

    working["mean_raw_to_slow_delta"] = _safe_numeric(working, "forecast_weight_mean_slow") - _safe_numeric(
        working, "forecast_weight_mean_raw"
    )
    working["mean_slow_to_used_delta"] = _safe_numeric(working, "forecast_weight_mean_used") - _safe_numeric(
        working, "forecast_weight_mean_slow"
    )
    working["mean_overlay_delta_observed"] = _safe_numeric(working, "mean_overlay_delta", 0.0).fillna(0.0)

    variance = _safe_numeric(working, "forecast_variance")
    variance_raw = _safe_numeric(working, "forecast_variance_raw")
    variance_robust = _safe_numeric(working, "forecast_variance_robust")
    working["variance_raw_to_used_delta"] = variance - variance_raw
    working["variance_used_to_robust_delta"] = variance_robust - variance
    working["variance_floor_trigger_flag"] = _bool_series(working, "forecast_variance_floor_triggered")

    working["l3_weight_delta"] = working["effective_weight"] - working["pre_l3_weight"]
    working["l3_return_delta_proxy"] = working["pre_l3_weight"] * (exposure_scale - 1.0) * working[
        "next_return_used_for_attribution"
    ]
    working["active_before_l3_flag"] = working["pre_l3_weight"] > EPS
    working["active_after_l3_flag"] = working["effective_weight"] > EPS
    for source_column, flag_column in [
        ("risk_regime_factor", "risk_regime_reduction_flag"),
        ("risk_vol_scale", "risk_vol_reduction_flag"),
        ("risk_cluster_scale", "risk_cluster_reduction_flag"),
        ("risk_signal_quality_scale", "risk_signal_quality_reduction_flag"),
    ]:
        working[flag_column] = _safe_numeric(working, source_column, 1.0).fillna(1.0) < 1.0 - 1e-9

    cluster_group = working.get("factor_industry_theme", pd.Series("", index=working.index)).fillna("").astype(str)
    if "cluster_mapping_cluster" in working.columns:
        fallback_group = working["cluster_mapping_cluster"].map(lambda value: "" if pd.isna(value) else f"cluster_{value:g}")
        cluster_group = cluster_group.where(cluster_group.str.strip() != "", fallback_group)
    working["cluster_group"] = cluster_group.where(cluster_group.str.strip() != "", "unknown")
    working["cluster_selected_rank"] = working["cluster_mapping_rank"]
    working = working.sort_values(["instrument", "datetime"]).copy()
    previous_pre_l3 = working.groupby("instrument")["pre_l3_weight"].shift(1).fillna(0.0)
    previous_effective = working.groupby("instrument")["effective_weight"].shift(1).fillna(0.0)
    working["turnover_contribution_pre_l3_abs"] = (working["pre_l3_weight"] - previous_pre_l3).abs() / 2.0
    working["turnover_contribution_effective_abs"] = (working["effective_weight"] - previous_effective).abs() / 2.0
    previous_active = previous_pre_l3 > EPS
    working["active_entry_flag"] = working["active_before_l3_flag"] & ~previous_active
    working["active_exit_flag"] = ~working["active_before_l3_flag"] & previous_active
    return working.sort_values(["datetime", "instrument"]).reset_index(drop=True)


def _attach_adaptive_backtest(
    frame: pd.DataFrame,
    backtest_positions: pd.DataFrame,
    daily_nav: pd.DataFrame,
) -> pd.DataFrame:
    working = frame.copy()
    if backtest_positions.empty:
        working["adaptive_join_status"] = "backtest_positions_missing"
    else:
        positions = backtest_positions.copy()
        positions["_adaptive_key_exists"] = True
        working = working.merge(positions, on=["datetime", "instrument"], how="left")
        joined = working["_adaptive_key_exists"].eq(True)
        working["adaptive_join_status"] = np.where(joined, "joined", "missing_backtest_row")
        working = working.drop(columns=["_adaptive_key_exists"], errors="ignore")

    if not daily_nav.empty:
        daily_extra = daily_nav.copy()
        duplicate_columns = [column for column in daily_extra.columns if column != "datetime" and column in working.columns]
        if duplicate_columns:
            daily_extra = daily_extra.drop(columns=duplicate_columns)
        working = working.merge(daily_extra, on="datetime", how="left")

    working["mvo_target_weight_pre_l3"] = _safe_numeric(working, "pre_l3_weight", 0.0).fillna(0.0)
    working["l3_model_target_weight"] = _safe_numeric(working, "effective_weight", 0.0).fillna(0.0)
    working["adaptive_actual_weight_after_trade"] = _safe_numeric(
        working, "adaptive_actual_weight_after_trade", 0.0
    ).fillna(0.0)
    working["adaptive_final_weight"] = working["adaptive_actual_weight_after_trade"]
    working["adaptive_shares"] = _safe_numeric(working, "adaptive_shares", 0.0).fillna(0.0)
    working["adaptive_position_market_value_close"] = _safe_numeric(
        working, "adaptive_position_market_value_close", 0.0
    ).fillna(0.0)
    working["adaptive_actual_buy_amount"] = _safe_numeric(working, "adaptive_actual_buy_amount", 0.0).fillna(0.0)
    working["is_rebalance_day"] = _bool_series(working, "is_rebalance_day") | _bool_series(
        working, "daily_is_rebalance_day"
    )
    working["adaptive_score_pass"] = _bool_series(working, "adaptive_score_pass") | _bool_series(
        working, "daily_adaptive_score_pass"
    )
    working["is_synthetic_carry"] = _bool_series(working, "is_synthetic_carry")
    for column in ["rebalance_reason", "rebalance_candidate_reason"]:
        if column not in working.columns:
            working[column] = ""
        working[column] = working[column].fillna("").astype(str)

    l3_weight_sum = working.groupby("datetime")["l3_model_target_weight"].transform("sum")
    working["l3_model_weight_sum_by_date"] = l3_weight_sum
    working["l3_cash_weight_by_date"] = (1.0 - l3_weight_sum).clip(lower=0.0)

    adaptive_weight_sum = working.groupby("datetime")["adaptive_final_weight"].transform("sum")
    working["adaptive_final_weight_sum_by_date"] = adaptive_weight_sum
    fallback_cash_weight = (1.0 - adaptive_weight_sum).clip(lower=0.0)
    if "adaptive_cash_weight_by_date" in working.columns:
        cash_weight = _safe_numeric(working, "adaptive_cash_weight_by_date")
        working["adaptive_cash_weight_by_date"] = cash_weight.fillna(fallback_cash_weight)
    else:
        working["adaptive_cash_weight_by_date"] = fallback_cash_weight
    working["adaptive_portfolio_weight_reconciliation_error"] = (
        working["adaptive_final_weight_sum_by_date"] + working["adaptive_cash_weight_by_date"] - 1.0
    )

    working = working.sort_values(["instrument", "datetime"]).copy()
    previous_weight = working.groupby("instrument")["adaptive_final_weight"].shift(1).fillna(0.0)
    working["adaptive_weight_change_from_previous_day"] = working["adaptive_final_weight"] - previous_weight
    working["adaptive_trade_delta_weight_proxy"] = np.where(
        working["is_rebalance_day"], working["adaptive_weight_change_from_previous_day"], 0.0
    )
    working["adaptive_trade_turnover_contribution_proxy_abs"] = working["adaptive_trade_delta_weight_proxy"].abs() / 2.0
    working["adaptive_no_trade_carry_forward_flag"] = ~working["is_rebalance_day"]
    working["model_vs_actual_drift_weight"] = working["l3_model_target_weight"] - working["adaptive_final_weight"]
    working["adaptive_realized_return_contribution"] = (
        working["adaptive_final_weight"] * _safe_numeric(working, "next_return", 0.0).fillna(0.0)
    )
    working["adaptive_final_weight_source"] = np.select(
        [working["adaptive_join_status"].ne("joined"), working["is_rebalance_day"]],
        ["missing_backtest_position", "adaptive_rebalance_executed"],
        default="adaptive_carry_forward",
    )
    working["adaptive_active_flag"] = working["adaptive_final_weight"] > EPS
    return working.sort_values(["datetime", "instrument"]).reset_index(drop=True)


def _build_decision_panel(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["schema_version"] = 2
    preferred_columns = [
        "schema_version",
        "datetime",
        "instrument",
        "cluster_mapping_rank",
        "cluster_mapping_selected_flag",
        "cluster_mapping_start_date",
        "cluster_mapping_end_date",
        "cluster_mapping_cluster",
        "cluster_mapping_reason",
        "cluster_mapping_name",
        "factor_industry_theme",
        "cluster_group",
        "in_raw_panel_flag",
        "in_bridge_flag",
        "adaptive_join_status",
        "route_a_join_status",
        "route_a_status",
        "route_b_status",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "next_close",
        "next_return",
        "return",
        "is_synthetic_carry",
        "forecast_mean_return",
        "forecast_mean_source",
        "forecast_mean_return_direct",
        "forecast_mean_return_direct_source",
        "forecast_weight_mean_raw",
        "forecast_weight_mean_slow",
        "forecast_weight_mean_used",
        "forecast_weight_mean_base_source",
        "forecast_weight_mean_transform",
        "optimizer_mean_used_for_audit",
        "mean_used_rank_pct_desc",
        "mean_used_zscore_by_date",
        "mean_direction_hit",
        "mean_signal_x_next_return",
        "route_a_mu_raw",
        "route_a_mu",
        "route_a_sigma",
        "route_a_nu",
        "route_a_mu_path_refined",
        "route_a_used_simple_pulse",
        "mu_raw",
        "mu_shadow_patched",
        "mu_used",
        "mean_overlay_mode",
        "mean_overlay_trigger_flag",
        "mean_overlay_delta",
        "forecast_variance_raw",
        "forecast_variance",
        "forecast_variance_robust",
        "forecast_variance_floor_value",
        "forecast_variance_floor_triggered",
        "forecast_variance_floor_quantile_used",
        "forecast_variance_floor_regime_on",
        "forecast_variance_floor_opportunity_strength",
        "variance_raw_to_used_delta",
        "variance_used_to_robust_delta",
        "covariance_source_col",
        "covariance_status",
        "covariance_n_instruments",
        "covariance_diag_from_matrix",
        "covariance_row_norm",
        "covariance_avg_cov_to_active",
        "covariance_avg_corr_to_active",
        "covariance_max_corr_partner",
        "covariance_max_corr_to_other",
        "covariance_max_cov_to_other",
        "covariance_weight_contribution",
        "covariance_condition_number",
        "covariance_effective_rank",
        "sigma_port_pre_l3",
        "sigma_port_effective_l3",
        "forecast_w_mode",
        "forecast_mu_preset",
        "forecast_weight_status",
        "forecast_weight_selected_n",
        "forecast_weight_inertia_alpha",
        "forecast_active_set_crowded_score",
        "mvo_target_weight_pre_l3",
        "l3_model_target_weight",
        "l3_model_weight_sum_by_date",
        "l3_cash_weight_by_date",
        "exposure_scale",
        "effective_exposure",
        "risk_base_exposure_scale",
        "risk_regime_on",
        "risk_regime_factor",
        "risk_macro_budget",
        "risk_bench_state_label",
        "risk_bench_trend_on",
        "risk_bench_vol_stress",
        "risk_bench_trend_score",
        "risk_portfolio_scale",
        "risk_signal_scale",
        "risk_vol_scale",
        "risk_cluster_scale",
        "risk_signal_quality_scale",
        "risk_regime_reduction_flag",
        "risk_vol_reduction_flag",
        "risk_cluster_reduction_flag",
        "risk_signal_quality_reduction_flag",
        "l3_weight_delta",
        "l3_return_delta_proxy",
        "is_rebalance_day",
        "rebalance_reason",
        "rebalance_gap_days",
        "trigger_portfolio_drift",
        "trigger_exposure_change",
        "trigger_macro_budget_change",
        "trigger_macro_state_flip",
        "trigger_risk_scale_change",
        "trigger_adaptive_score",
        "trigger_expected_return_gain",
        "trigger_variance_change",
        "trigger_correlation_improvement",
        "trigger_turnover",
        "adaptive_score_pass",
        "rebalance_candidate_reason",
        "adaptive_daily_nav",
        "adaptive_cash_amount",
        "adaptive_cash_weight_by_date",
        "adaptive_actual_weight_after_trade",
        "adaptive_final_weight",
        "adaptive_final_weight_sum_by_date",
        "adaptive_portfolio_weight_reconciliation_error",
        "adaptive_shares",
        "adaptive_position_market_value_close",
        "adaptive_position_market_value_next_close",
        "adaptive_actual_buy_amount",
        "adaptive_weight_change_from_previous_day",
        "adaptive_trade_delta_weight_proxy",
        "adaptive_trade_turnover_contribution_proxy_abs",
        "adaptive_no_trade_carry_forward_flag",
        "model_vs_actual_drift_weight",
        "adaptive_final_weight_source",
        "adaptive_realized_return_contribution",
        "expected_mean_contribution",
        "realized_return_contribution",
        "forecast_error_next_return_minus_mean",
    ]
    excluded = {"forecast_cov_matrix", "forecast_cov_matrix_robust"}
    columns = [column for column in preferred_columns if column in working.columns and column not in excluded]
    remaining_diagnostics = [
        column
        for column in working.columns
        if column.startswith(("risk_trend_sleeve", "anchor_", "risk_anchor")) and column not in columns and column not in excluded
    ]
    return working[columns + remaining_diagnostics].sort_values(["datetime", "instrument"]).reset_index(drop=True)


def _build_portfolio_weights(decision_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if decision_panel.empty:
        return pd.DataFrame(), pd.DataFrame()
    base_columns = [
        "datetime",
        "instrument",
        "adaptive_final_weight",
        "l3_model_target_weight",
        "mvo_target_weight_pre_l3",
        "adaptive_cash_weight_by_date",
        "is_rebalance_day",
        "rebalance_reason",
        "rebalance_gap_days",
        "adaptive_final_weight_source",
        "model_vs_actual_drift_weight",
        "adaptive_trade_delta_weight_proxy",
        "adaptive_shares",
        "close",
        "adaptive_position_market_value_close",
        "forecast_weight_mean_used",
        "forecast_variance",
        "forecast_variance_floor_triggered",
        "covariance_avg_corr_to_active",
        "exposure_scale",
        "effective_exposure",
        "risk_macro_budget",
        "risk_bench_state_label",
        "risk_portfolio_scale",
        "risk_signal_scale",
        "risk_regime_factor",
        "risk_vol_scale",
        "risk_cluster_scale",
        "risk_signal_quality_scale",
    ]
    keep = [column for column in base_columns if column in decision_panel.columns]
    weights = decision_panel[keep].copy()
    weights = weights.rename(columns={"adaptive_final_weight": "target_weight"})
    weights["target_weight"] = _numeric(weights["target_weight"]).fillna(0.0)
    weights["weight_type"] = "instrument"
    active = weights[weights["target_weight"].abs() > EPS].copy()

    cash_rows: list[dict[str, Any]] = []
    for date_value, day_frame in decision_panel.groupby("datetime", sort=True):
        target_sum = float(_numeric(day_frame["adaptive_final_weight"]).fillna(0.0).sum())
        cash_weight = _first_finite(day_frame.get("adaptive_cash_weight_by_date", pd.Series(dtype=float)))
        if not np.isfinite(cash_weight):
            cash_weight = max(0.0, 1.0 - target_sum)
        cash_rows.append(
            {
                "datetime": pd.Timestamp(date_value),
                "instrument": "CASH",
                "target_weight": max(0.0, float(cash_weight)),
                "l3_model_target_weight": 0.0,
                "mvo_target_weight_pre_l3": 0.0,
                "adaptive_cash_weight_by_date": max(0.0, float(cash_weight)),
                "is_rebalance_day": bool(day_frame["is_rebalance_day"].iloc[0]) if "is_rebalance_day" in day_frame else False,
                "rebalance_reason": _first_text(day_frame.get("rebalance_reason", pd.Series(dtype=object))),
                "rebalance_gap_days": _first_finite(day_frame.get("rebalance_gap_days", pd.Series(dtype=float))),
                "adaptive_final_weight_source": "cash_residual",
                "model_vs_actual_drift_weight": 0.0,
                "adaptive_trade_delta_weight_proxy": 0.0,
                "adaptive_shares": np.nan,
                "close": np.nan,
                "adaptive_position_market_value_close": np.nan,
                "weight_type": "cash",
            }
        )
    cash = pd.DataFrame(cash_rows)
    portfolio = pd.concat([active, cash], ignore_index=True, sort=False).sort_values(
        ["datetime", "weight_type", "target_weight", "instrument"], ascending=[True, True, False, True]
    )
    latest_date = portfolio["datetime"].max()
    latest = portfolio[portfolio["datetime"].eq(latest_date)].copy()
    return portfolio.reset_index(drop=True), latest.reset_index(drop=True)


def _build_rebalance_event_log(decision_panel: pd.DataFrame) -> pd.DataFrame:
    if decision_panel.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for date_value, day_frame in decision_panel.groupby("datetime", sort=True):
        final_weight = _numeric(day_frame["adaptive_final_weight"]).fillna(0.0)
        l3_weight = _numeric(day_frame["l3_model_target_weight"]).fillna(0.0)
        trade_delta = _numeric(day_frame.get("adaptive_trade_delta_weight_proxy", pd.Series(0.0, index=day_frame.index))).fillna(0.0)
        drift = _numeric(day_frame.get("model_vs_actual_drift_weight", pd.Series(0.0, index=day_frame.index))).fillna(0.0)
        rows.append(
            {
                "datetime": pd.Timestamp(date_value),
                "is_rebalance_day": bool(day_frame["is_rebalance_day"].iloc[0]) if "is_rebalance_day" in day_frame else False,
                "rebalance_reason": _first_text(day_frame.get("rebalance_reason", pd.Series(dtype=object))),
                "rebalance_candidate_reason": _first_text(day_frame.get("rebalance_candidate_reason", pd.Series(dtype=object))),
                "rebalance_gap_days": _first_finite(day_frame.get("rebalance_gap_days", pd.Series(dtype=float))),
                "trigger_portfolio_drift": _first_finite(day_frame.get("trigger_portfolio_drift", pd.Series(dtype=float))),
                "trigger_exposure_change": _first_finite(day_frame.get("trigger_exposure_change", pd.Series(dtype=float))),
                "trigger_macro_budget_change": _first_finite(day_frame.get("trigger_macro_budget_change", pd.Series(dtype=float))),
                "trigger_macro_state_flip": bool(day_frame["trigger_macro_state_flip"].iloc[0]) if "trigger_macro_state_flip" in day_frame else False,
                "trigger_risk_scale_change": _first_finite(day_frame.get("trigger_risk_scale_change", pd.Series(dtype=float))),
                "trigger_adaptive_score": _first_finite(day_frame.get("trigger_adaptive_score", pd.Series(dtype=float))),
                "trigger_expected_return_gain": _first_finite(day_frame.get("trigger_expected_return_gain", pd.Series(dtype=float))),
                "trigger_variance_change": _first_finite(day_frame.get("trigger_variance_change", pd.Series(dtype=float))),
                "trigger_correlation_improvement": _first_finite(day_frame.get("trigger_correlation_improvement", pd.Series(dtype=float))),
                "trigger_turnover": _first_finite(day_frame.get("trigger_turnover", pd.Series(dtype=float))),
                "adaptive_score_pass": bool(day_frame["adaptive_score_pass"].iloc[0]) if "adaptive_score_pass" in day_frame else False,
                "adaptive_final_weight_sum": float(final_weight.sum()),
                "adaptive_cash_weight": _first_finite(day_frame.get("adaptive_cash_weight_by_date", pd.Series(dtype=float))),
                "l3_model_target_weight_sum": float(l3_weight.sum()),
                "l3_cash_weight": _first_finite(day_frame.get("l3_cash_weight_by_date", pd.Series(dtype=float))),
                "adaptive_active_names": int((final_weight > EPS).sum()),
                "l3_model_active_names": int((l3_weight > EPS).sum()),
                "model_vs_actual_abs_drift_sum": float(drift.abs().sum()),
                "adaptive_trade_turnover_proxy": float(trade_delta.abs().sum() / 2.0),
                "adaptive_weight_reconciliation_error": _first_finite(
                    day_frame.get("adaptive_portfolio_weight_reconciliation_error", pd.Series(dtype=float))
                ),
            }
        )
    return pd.DataFrame(rows)


def _format_weight_list(day_frame: pd.DataFrame, weight_col: str, limit: int = 6) -> str:
    if weight_col not in day_frame.columns:
        return ""
    work = day_frame[["instrument", weight_col]].copy()
    work[weight_col] = _numeric(work[weight_col]).fillna(0.0)
    work = work[work[weight_col] > EPS].sort_values(weight_col, ascending=False).head(limit)
    return "|".join(f"{row['instrument']}:{row[weight_col]:.4f}" for _, row in work.iterrows())


def _build_simple_daily_summary(decision_panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date_value, day_frame in decision_panel.groupby("datetime", sort=True):
        adaptive_weight = _numeric(day_frame["adaptive_final_weight"]).fillna(0.0)
        l3_weight = _numeric(day_frame["l3_model_target_weight"]).fillna(0.0)
        contribution = _numeric(day_frame.get("adaptive_realized_return_contribution", pd.Series(0.0, index=day_frame.index))).fillna(0.0)
        drift = _numeric(day_frame.get("model_vs_actual_drift_weight", pd.Series(0.0, index=day_frame.index))).fillna(0.0)
        trade_delta = _numeric(day_frame.get("adaptive_trade_delta_weight_proxy", pd.Series(0.0, index=day_frame.index))).fillna(0.0)
        rows.append(
            {
                "datetime": pd.Timestamp(date_value),
                "is_rebalance_day": bool(day_frame["is_rebalance_day"].iloc[0]) if "is_rebalance_day" in day_frame else False,
                "rebalance_reason": _first_text(day_frame.get("rebalance_reason", pd.Series(dtype=object))),
                "rebalance_gap_days": _first_finite(day_frame.get("rebalance_gap_days", pd.Series(dtype=float))),
                "target_weight_sum": float(adaptive_weight.sum()),
                "cash_weight": _first_finite(day_frame.get("adaptive_cash_weight_by_date", pd.Series(dtype=float))),
                "active_names": int((adaptive_weight > EPS).sum()),
                "l3_model_weight_sum": float(l3_weight.sum()),
                "l3_cash_weight": _first_finite(day_frame.get("l3_cash_weight_by_date", pd.Series(dtype=float))),
                "avg_exposure_scale": float(_numeric(day_frame.get("exposure_scale", pd.Series(dtype=float))).mean()),
                "risk_regime_factor": _first_finite(day_frame.get("risk_regime_factor", pd.Series(dtype=float))),
                "risk_macro_budget": _first_finite(day_frame.get("risk_macro_budget", pd.Series(dtype=float))),
                "risk_bench_state_label": _first_text(day_frame.get("risk_bench_state_label", pd.Series(dtype=object))),
                "effective_exposure": _first_finite(day_frame.get("effective_exposure", pd.Series(dtype=float))),
                "risk_portfolio_scale": _first_finite(day_frame.get("risk_portfolio_scale", pd.Series(dtype=float))),
                "risk_signal_scale": _first_finite(day_frame.get("risk_signal_scale", pd.Series(dtype=float))),
                "trigger_adaptive_score": _first_finite(day_frame.get("trigger_adaptive_score", pd.Series(dtype=float))),
                "trigger_expected_return_gain": _first_finite(day_frame.get("trigger_expected_return_gain", pd.Series(dtype=float))),
                "trigger_correlation_improvement": _first_finite(day_frame.get("trigger_correlation_improvement", pd.Series(dtype=float))),
                "adaptive_trade_turnover_proxy": float(trade_delta.abs().sum() / 2.0),
                "model_vs_actual_abs_drift_sum": float(drift.abs().sum()),
                "adaptive_realized_return_contribution_sum": float(contribution.sum()),
                "worst_contributors": _format_top_contributors(day_frame, "adaptive_realized_return_contribution", ascending=True),
                "best_contributors": _format_top_contributors(day_frame, "adaptive_realized_return_contribution", ascending=False),
                "top_weights": _format_weight_list(day_frame, "adaptive_final_weight"),
            }
        )
    return pd.DataFrame(rows)


def _build_simple_instrument_drilldown(decision_panel: pd.DataFrame) -> pd.DataFrame:
    if decision_panel.empty:
        return pd.DataFrame()
    work = decision_panel.copy()
    contribution = _numeric(work.get("adaptive_realized_return_contribution", pd.Series(0.0, index=work.index))).fillna(0.0)
    work["_contribution_rank_asc"] = contribution.groupby(work["datetime"]).rank(method="first", ascending=True)
    active = _numeric(work.get("adaptive_final_weight", pd.Series(0.0, index=work.index))).fillna(0.0) > EPS
    model_active = _numeric(work.get("l3_model_target_weight", pd.Series(0.0, index=work.index))).fillna(0.0) > EPS
    large_drift = _numeric(work.get("model_vs_actual_drift_weight", pd.Series(0.0, index=work.index))).fillna(0.0).abs() >= 0.03
    worst = work["_contribution_rank_asc"] <= 5
    missing = work.get("adaptive_join_status", pd.Series("", index=work.index)).astype(str).ne("joined")
    floor_trigger = _bool_series(work, "forecast_variance_floor_triggered") & (active | model_active)
    selected = work[active | model_active | large_drift | worst | missing | floor_trigger].copy()
    selected["simple_reason"] = ""
    selected.loc[active, "simple_reason"] += "active|"
    selected.loc[model_active & ~active, "simple_reason"] += "model_target_not_held|"
    selected.loc[large_drift, "simple_reason"] += "large_model_actual_drift|"
    selected.loc[worst, "simple_reason"] += "daily_bottom5_contributor|"
    selected.loc[missing, "simple_reason"] += "missing_adaptive_join|"
    selected.loc[floor_trigger, "simple_reason"] += "variance_floor_triggered|"
    selected["simple_reason"] = selected["simple_reason"].str.strip("|")
    columns = [
        "datetime",
        "instrument",
        "factor_industry_theme",
        "simple_reason",
        "adaptive_final_weight",
        "adaptive_cash_weight_by_date",
        "l3_model_target_weight",
        "mvo_target_weight_pre_l3",
        "model_vs_actual_drift_weight",
        "is_rebalance_day",
        "rebalance_reason",
        "forecast_weight_mean_used",
        "mean_used_rank_pct_desc",
        "forecast_variance_floor_triggered",
        "covariance_avg_corr_to_active",
        "exposure_scale",
        "next_return",
        "adaptive_realized_return_contribution",
        "adaptive_final_weight_source",
        "adaptive_join_status",
    ]
    columns = [column for column in columns if column in selected.columns]
    return selected[columns].sort_values(["datetime", "adaptive_final_weight", "instrument"], ascending=[True, False, True]).reset_index(drop=True)


def _write_simple_readme(
    output_paths: AuditOutputPaths,
    metadata: dict[str, Any],
    simple_daily_summary: pd.DataFrame,
) -> None:
    latest_date = ""
    if not simple_daily_summary.empty:
        latest_date = pd.Timestamp(simple_daily_summary["datetime"].max()).strftime("%Y-%m-%d")
    lines = [
        "# Simple Audit",
        "",
        "这是日常使用的精简审计入口。默认只回答三个问题：今天拿什么权重、今天为什么调仓/没调仓、哪些标的值得追一下。",
        "",
        "## 先看这三个文件",
        "",
        f"1. 最新组合权重: `{output_paths.simple_latest_weights.name}`",
        f"2. 每日决策摘要: `{output_paths.simple_daily_summary.name}`",
        f"3. 标的级 drilldown: `{output_paths.simple_instrument_drilldown.name}`",
        "",
        "## 权重语义",
        "",
        "- `target_weight`: adaptive 后最终组合权重，可直接用于组合权重输入。",
        "- `CASH`: 现金行，和标的权重加总约等于 1。",
        "- `l3_model_target_weight`: 每日模型目标权重，不代表非调仓日实际执行。",
        "- `adaptive_final_weight`: 实际持仓权重，非调仓日为 carry-forward。",
        "",
        "## 本次运行",
        "",
        f"- run_profile: `{metadata.get('run_profile', '')}`",
        f"- audit_level: `{metadata.get('audit_level', '')}`",
        f"- start/end: `{metadata.get('inputs', {}).get('start_date', '')}` -> `{metadata.get('inputs', {}).get('end_date', '')}`",
        f"- latest trading date in summary: `{latest_date}`",
        "",
        "## 需要工程级排查时",
        "",
        "重新运行审计时加 `--audit-level full`，会输出完整 `decision_panel.csv` 和 `param_inst_daily.csv`。",
    ]
    output_paths.readme_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _decision_panel_schema() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "primary_key": ["datetime", "instrument"],
        "weight_semantics": {
            "mvo_target_weight_pre_l3": "MVO/bridge target before Layer-3 exposure scaling; source forecast_w.",
            "l3_model_target_weight": "Daily model target after Layer-3 scaling; source forecast_w * exposure_scale.",
            "adaptive_final_weight": "Actual portfolio weight after adaptive rebalance execution or carry-forward.",
            "daily_portfolio_weights_from_adaptive.target_weight": "Direct portfolio-weight output. Non-rebalance days use carry-forward actual weights.",
            "CASH": "Residual cash row emitted in portfolio-weight files so daily weights add to one.",
        },
        "layers": {
            "identity_pool": ["cluster_mapping_rank", "cluster_mapping_selected_flag", "route_a_join_status"],
            "market_tradeability": ["close", "next_close", "next_return", "is_synthetic_carry"],
            "mean_prediction": ["forecast_weight_mean_used", "route_a_mu", "route_a_used_simple_pulse", "mean_used_rank_pct_desc"],
            "variance_prediction": ["forecast_variance_raw", "forecast_variance", "forecast_variance_floor_triggered"],
            "covariance": ["covariance_avg_corr_to_active", "covariance_weight_contribution", "covariance_condition_number"],
            "mvo": ["forecast_weight_status", "forecast_weight_selected_n", "mvo_target_weight_pre_l3"],
            "l3": [
                "exposure_scale",
                "effective_exposure",
                "risk_macro_budget",
                "risk_bench_state_label",
                "risk_portfolio_scale",
                "risk_signal_scale",
                "risk_regime_factor",
                "risk_vol_scale",
                "l3_model_target_weight",
            ],
            "adaptive_rebalance": ["is_rebalance_day", "rebalance_reason", "trigger_adaptive_score", "adaptive_final_weight"],
        },
    }


def _compute_cluster_mapping_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date_value, day_frame in frame.groupby("datetime", sort=True):
        selected = day_frame[day_frame["cluster_mapping_selected_flag"]]
        unselected = day_frame[~day_frame["cluster_mapping_selected_flag"]]
        selected_missing_raw = selected[~selected["in_raw_panel_flag"]]
        selected_missing_bridge = selected[~selected["in_bridge_flag"]]
        selected_never_active = selected[~selected["active_before_l3_flag"]]
        unselected_missing_raw = unselected[~unselected["in_raw_panel_flag"]]
        unselected_missing_bridge = unselected[~unselected["in_bridge_flag"]]
        rows.append(
            {
                "datetime": pd.Timestamp(date_value),
                "cluster_universe_count": int(day_frame.shape[0]),
                "cluster_selected_count": int(selected.shape[0]),
                "cluster_unselected_count": int(unselected.shape[0]),
                "cluster_selected_in_raw_count": int(selected["in_raw_panel_flag"].sum()),
                "cluster_unselected_in_raw_count": int(unselected["in_raw_panel_flag"].sum()),
                "cluster_selected_in_bridge_count": int(selected["in_bridge_flag"].sum()),
                "cluster_unselected_in_bridge_count": int(unselected["in_bridge_flag"].sum()),
                "cluster_selected_active_count": int(selected["active_before_l3_flag"].sum()),
                "cluster_selected_missing_raw_count": int(selected_missing_raw.shape[0]),
                "cluster_selected_missing_bridge_count": int(selected_missing_bridge.shape[0]),
                "cluster_unselected_missing_raw_count": int(unselected_missing_raw.shape[0]),
                "cluster_unselected_missing_bridge_count": int(unselected_missing_bridge.shape[0]),
                "cluster_selected_zero_weight_count": int(selected_never_active.shape[0]),
                "cluster_selected_missing_raw_codes": "|".join(selected_missing_raw["instrument"].astype(str).tolist()[:30]),
                "cluster_selected_missing_bridge_codes": "|".join(selected_missing_bridge["instrument"].astype(str).tolist()[:30]),
                "cluster_selected_active_weight_sum": float(selected["pre_l3_weight"].sum()),
                "cluster_selected_effective_weight_sum": float(selected["effective_weight"].sum()),
                "cluster_selected_next_return_mean": float(_numeric(selected["next_return"]).mean()),
                "cluster_unselected_next_return_mean": float(_numeric(unselected["next_return"]).mean()),
                "cluster_selected_vs_unselected_next_return_spread": float(
                    _numeric(selected["next_return"]).mean() - _numeric(unselected["next_return"]).mean()
                ),
                "cluster_selected_next_return_median": float(_numeric(selected["next_return"]).median()),
            }
        )
    return pd.DataFrame(rows)


def _compute_daily_summary(frame: pd.DataFrame, covariance_daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    previous_pre_l3: pd.Series | None = None
    previous_effective: pd.Series | None = None
    for date_value, day_frame in frame.groupby("datetime", sort=True):
        indexed = day_frame.set_index("instrument")
        pre_l3 = _numeric(indexed["pre_l3_weight"]).fillna(0.0)
        effective = _numeric(indexed["effective_weight"]).fillna(0.0)
        if previous_pre_l3 is None:
            turnover_pre_l3 = float("nan")
            turnover_effective = float("nan")
        else:
            full_index = pre_l3.index.union(previous_pre_l3.index)
            turnover_pre_l3 = float((pre_l3.reindex(full_index).fillna(0.0) - previous_pre_l3.reindex(full_index).fillna(0.0)).abs().sum() / 2.0)
            turnover_effective = float(
                (effective.reindex(full_index).fillna(0.0) - previous_effective.reindex(full_index).fillna(0.0)).abs().sum()
                / 2.0
            )
        sorted_weights = pre_l3.sort_values(ascending=False)
        selected = day_frame[day_frame["cluster_mapping_selected_flag"]]
        unselected = day_frame[~day_frame["cluster_mapping_selected_flag"]]
        adaptive_final = _safe_numeric(day_frame, "adaptive_final_weight", 0.0).fillna(0.0)
        adaptive_trade_delta = _safe_numeric(day_frame, "adaptive_trade_delta_weight_proxy", 0.0).fillna(0.0)
        model_actual_drift = _safe_numeric(day_frame, "model_vs_actual_drift_weight", 0.0).fillna(0.0)
        rows.append(
            {
                "datetime": pd.Timestamp(date_value),
                "n_rows": int(day_frame.shape[0]),
                "n_bridge_rows": int(day_frame["in_bridge_flag"].sum()),
                "cluster_selected_count": int(selected.shape[0]),
            "cluster_unselected_count": int(unselected.shape[0]),
                "active_names_pre_l3": int((pre_l3 > EPS).sum()),
                "active_names_effective": int((effective > EPS).sum()),
                "weight_sum_pre_l3": float(pre_l3.sum()),
                "weight_sum_effective": float(effective.sum()),
                "effective_holdings_pre_l3": _effective_holdings(pre_l3),
                "effective_holdings_l3": _effective_holdings(effective),
                "weight_top1": float(sorted_weights.iloc[0]) if not sorted_weights.empty else float("nan"),
                "weight_top3": float(sorted_weights.head(3).sum()),
                "weight_top5": float(sorted_weights.head(5).sum()),
                "turnover_pre_l3": turnover_pre_l3,
                "turnover_effective": turnover_effective,
                "portfolio_expected_mean_contribution": float(_numeric(day_frame["expected_mean_contribution"]).sum()),
                "portfolio_realized_return_contribution": float(_numeric(day_frame["realized_return_contribution"]).sum()),
                "l3_gross_weight_delta": float(_numeric(day_frame["l3_weight_delta"]).abs().sum()),
                "l3_return_delta_proxy_sum": float(_numeric(day_frame["l3_return_delta_proxy"]).sum()),
                "cluster_selected_equal_weight_next_return": float(_numeric(selected["next_return"]).mean()),
                "cluster_unselected_equal_weight_next_return": float(_numeric(unselected["next_return"]).mean()),
                "cluster_selected_vs_unselected_spread": float(
                    _numeric(selected["next_return"]).mean() - _numeric(unselected["next_return"]).mean()
                ),
                "mean_ic_pearson": _safe_corr(day_frame["optimizer_mean_used_for_audit"], day_frame["next_return"], "pearson"),
                "mean_ic_spearman": _safe_corr(day_frame["optimizer_mean_used_for_audit"], day_frame["next_return"], "spearman"),
                "variance_floor_trigger_rate": float(day_frame.loc[day_frame["in_bridge_flag"], "variance_floor_trigger_flag"].mean())
                if day_frame["in_bridge_flag"].any()
                else float("nan"),
                "risk_base_exposure_scale": _first_finite(day_frame.get("risk_base_exposure_scale", pd.Series(dtype=float))),
                "risk_regime_factor": _first_finite(day_frame.get("risk_regime_factor", pd.Series(dtype=float))),
                "risk_vol_scale": _first_finite(day_frame.get("risk_vol_scale", pd.Series(dtype=float))),
                "risk_cluster_rho_bar": _first_finite(day_frame.get("risk_cluster_rho_bar", pd.Series(dtype=float))),
                "risk_cluster_scale": _first_finite(day_frame.get("risk_cluster_scale", pd.Series(dtype=float))),
                "risk_signal_quality_rolling_ic": _first_finite(day_frame.get("risk_signal_quality_rolling_ic", pd.Series(dtype=float))),
                "risk_signal_quality_scale": _first_finite(day_frame.get("risk_signal_quality_scale", pd.Series(dtype=float))),
                "adaptive_final_weight_sum": float(adaptive_final.sum()),
                "adaptive_cash_weight_by_date": _first_finite(day_frame.get("adaptive_cash_weight_by_date", pd.Series(dtype=float))),
                "adaptive_active_names": int((adaptive_final > EPS).sum()),
                "adaptive_trade_turnover_proxy": float(adaptive_trade_delta.abs().sum() / 2.0),
                "model_vs_actual_abs_drift_sum": float(model_actual_drift.abs().sum()),
                "adaptive_weight_reconciliation_error": _first_finite(
                    day_frame.get("adaptive_portfolio_weight_reconciliation_error", pd.Series(dtype=float))
                ),
                "top_positive_contributors": _format_top_contributors(day_frame, "realized_return_contribution", ascending=False),
                "top_negative_contributors": _format_top_contributors(day_frame, "realized_return_contribution", ascending=True),
            }
        )
        previous_pre_l3 = pre_l3.copy()
        previous_effective = effective.copy()
    summary = pd.DataFrame(rows)
    if not covariance_daily.empty:
        summary = summary.merge(covariance_daily, on="datetime", how="left")
    return summary


def _compute_instrument_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for instrument, instrument_frame in frame.groupby("instrument", sort=True):
        direction_values = instrument_frame["mean_direction_hit"].dropna()
        rows.append(
            {
                "instrument": instrument,
                "cluster_mapping_selected_flag": bool(instrument_frame["cluster_mapping_selected_flag"].any()),
                "cluster_mapping_rank": _first_finite(instrument_frame["cluster_mapping_rank"]),
                "cluster_group_latest": str(instrument_frame["cluster_group"].dropna().iloc[-1]) if instrument_frame["cluster_group"].notna().any() else "",
                "rows": int(instrument_frame.shape[0]),
                "bridge_rows": int(instrument_frame["in_bridge_flag"].sum()),
                "raw_panel_rows": int(instrument_frame["in_raw_panel_flag"].sum()),
                "active_days_pre_l3": int(instrument_frame["active_before_l3_flag"].sum()),
                "active_days_effective": int(instrument_frame["active_after_l3_flag"].sum()),
                "avg_pre_l3_weight": float(_numeric(instrument_frame["pre_l3_weight"]).mean()),
                "avg_effective_weight": float(_numeric(instrument_frame["effective_weight"]).mean()),
                "max_pre_l3_weight": float(_numeric(instrument_frame["pre_l3_weight"]).max()),
                "sum_realized_return_contribution": float(_numeric(instrument_frame["realized_return_contribution"]).sum()),
                "avg_optimizer_mean_used": float(_numeric(instrument_frame["optimizer_mean_used_for_audit"]).mean()),
                "mean_direction_hit_rate": float(direction_values.mean()) if not direction_values.empty else float("nan"),
                "variance_floor_trigger_rate": float(instrument_frame["variance_floor_trigger_flag"].mean()),
                "l3_reduction_days": int((_numeric(instrument_frame["l3_weight_delta"]) < -EPS).sum()),
                "route_a_join_rate": float((instrument_frame["route_a_join_status"] == "joined").mean()),
            }
        )
    return pd.DataFrame(rows)


def _top_bottom_spread_by_score(day_frame: pd.DataFrame, score_col: str, return_col: str, max_count: int = 20) -> float:
    if score_col not in day_frame.columns or return_col not in day_frame.columns:
        return float("nan")
    working = day_frame[[score_col, return_col]].copy()
    working[score_col] = _numeric(working[score_col])
    working[return_col] = _numeric(working[return_col])
    working = working.dropna()
    if len(working) < 2:
        return float("nan")
    group_count = min(max_count, max(1, len(working) // 2))
    ordered = working.sort_values(score_col, ascending=False)
    return float(ordered.head(group_count)[return_col].mean() - ordered.tail(group_count)[return_col].mean())


def _build_layer_quality_panel(decision_panel: pd.DataFrame) -> pd.DataFrame:
    if decision_panel.empty:
        return pd.DataFrame()
    working = decision_panel.copy()
    next_return = _safe_numeric(working, "next_return")
    mean_prediction = _coalesce_numeric(
        working,
        ["optimizer_mean_used_for_audit", "forecast_weight_mean_used", "mu_used", "forecast_mean_return"],
    )
    forecast_variance = _safe_numeric(working, "forecast_variance")
    forecast_variance_raw = _safe_numeric(working, "forecast_variance_raw")
    mvo_weight = _safe_numeric(working, "mvo_target_weight_pre_l3", 0.0).fillna(0.0)
    l3_weight = _safe_numeric(working, "l3_model_target_weight", 0.0).fillna(0.0)
    adaptive_weight = _safe_numeric(working, "adaptive_final_weight", 0.0).fillna(0.0)
    realized_sq_return = np.square(next_return)

    working["schema_version"] = 1
    working["label_status"] = np.where(next_return.notna(), "realized", "pending")
    working["tradable_status"] = np.select(
        [~_bool_series(working, "in_raw_panel_flag"), ~_bool_series(working, "in_bridge_flag"), next_return.isna()],
        ["missing_raw", "missing_bridge", "label_pending"],
        default="ok",
    )
    working["pool_selected_flag"] = _bool_series(working, "cluster_mapping_selected_flag")
    working["pool_unselected_flag"] = ~working["pool_selected_flag"]
    working["pool_bucket"] = np.where(working["pool_selected_flag"], "selected", "unselected")
    working["pool_selected_rank"] = _safe_numeric(working, "cluster_mapping_rank")
    working["pool_selected_next_return"] = np.where(working["pool_selected_flag"], next_return, np.nan)
    working["pool_unselected_next_return"] = np.where(working["pool_unselected_flag"], next_return, np.nan)
    working["pool_active_transmitted_flag"] = working["pool_selected_flag"] & ((mvo_weight > EPS) | (adaptive_weight > EPS))
    working["pool_unselected_active_transmitted_flag"] = working["pool_unselected_flag"] & ((mvo_weight > EPS) | (adaptive_weight > EPS))
    working["pool_selected_but_zero_weight_flag"] = working["pool_selected_flag"] & (mvo_weight.abs() <= EPS) & (adaptive_weight.abs() <= EPS)
    working["pool_unselected_but_zero_weight_flag"] = working["pool_unselected_flag"] & (mvo_weight.abs() <= EPS) & (adaptive_weight.abs() <= EPS)
    selected_return = working["pool_selected_next_return"]
    unselected_return = working["pool_unselected_next_return"]
    working["pool_selected_next_return_rank_pct_desc"] = selected_return.groupby(working["datetime"]).rank(
        pct=True,
        ascending=False,
    )
    working["pool_unselected_next_return_rank_pct_desc"] = unselected_return.groupby(working["datetime"]).rank(
        pct=True,
        ascending=False,
    )

    working["mean_prediction"] = mean_prediction
    working["mean_rank_pct_desc"] = _safe_numeric(working, "mean_used_rank_pct_desc")
    working["mean_zscore_by_date"] = _safe_numeric(working, "mean_used_zscore_by_date")
    working["mean_direction_hit"] = _safe_numeric(working, "mean_direction_hit")
    working["mean_signed_error"] = next_return - mean_prediction
    working["mean_abs_error"] = working["mean_signed_error"].abs()
    working["mean_signal_x_next_return"] = mean_prediction * next_return
    working["mean_weighted_error"] = adaptive_weight * working["mean_signed_error"]
    working["mean_active_transmitted_flag"] = (l3_weight > EPS) | (adaptive_weight > EPS)

    working["forecast_variance_used"] = forecast_variance
    working["realized_sq_return_next"] = realized_sq_return
    working["variance_error"] = realized_sq_return - forecast_variance
    working["variance_abs_error"] = working["variance_error"].abs()
    working["variance_calibration_ratio"] = realized_sq_return / forecast_variance.where(forecast_variance.abs() > EPS)
    working["variance_floor_trigger_flag"] = _bool_series(working, "forecast_variance_floor_triggered")
    raw_abs_error = (realized_sq_return - forecast_variance_raw).abs()
    used_abs_error = (realized_sq_return - forecast_variance).abs()
    working["variance_floor_helped_proxy"] = working["variance_floor_trigger_flag"] & (used_abs_error < raw_abs_error)
    working = working.sort_values(["instrument", "datetime"]).copy()
    working["realized_var_5d_trailing"] = working.groupby("instrument")["realized_sq_return_next"].transform(
        lambda values: values.rolling(5, min_periods=2).mean()
    )
    working["realized_var_20d_trailing"] = working.groupby("instrument")["realized_sq_return_next"].transform(
        lambda values: values.rolling(20, min_periods=5).mean()
    )
    working = working.sort_values(["datetime", "instrument"]).copy()

    working["covariance_marginal_to_mvo"] = _safe_numeric(working, "covariance_weight_contribution") / mvo_weight.where(
        mvo_weight.abs() > EPS
    )
    basket_return_pre_l3 = (mvo_weight * next_return.fillna(0.0)).groupby(working["datetime"]).transform("sum")
    working["realized_active_basket_return_pre_l3"] = basket_return_pre_l3
    working["realized_active_basket_cross_return_proxy"] = next_return * (basket_return_pre_l3 - mvo_weight * next_return.fillna(0.0))
    working["covariance_realized_proxy_error"] = (
        working["realized_active_basket_cross_return_proxy"] - working["covariance_marginal_to_mvo"]
    )

    working["mvo_selected_flag"] = mvo_weight > EPS
    working["mvo_weight_rank_pct_desc"] = mvo_weight.groupby(working["datetime"]).rank(pct=True, ascending=False)
    working["mvo_expected_contribution"] = mvo_weight * mean_prediction.fillna(0.0)
    working["mvo_realized_contribution_pre_l3"] = mvo_weight * next_return.fillna(0.0)
    weight_status = working.get("forecast_weight_status", pd.Series("", index=working.index)).fillna("").astype(str)
    working["mvo_cap_floor_binding_flag"] = weight_status.str.contains("cap|floor|band", case=False, regex=True)
    working["mvo_infeasible_or_fallback_flag"] = weight_status.str.contains("infeasible|fallback|error", case=False, regex=True)

    working["l3_return_delta_proxy"] = _safe_numeric(working, "l3_return_delta_proxy", 0.0).fillna(0.0)
    working["l3_cut_helped_flag"] = (_safe_numeric(working, "l3_weight_delta", 0.0) < -EPS) & (next_return < 0.0)
    working["l3_cut_cost_flag"] = (_safe_numeric(working, "l3_weight_delta", 0.0) < -EPS) & (next_return > 0.0)

    working["adaptive_no_trade_opportunity_cost_proxy"] = (l3_weight - adaptive_weight) * next_return.fillna(0.0)
    working["adaptive_trade_delta_weight"] = _safe_numeric(working, "adaptive_trade_delta_weight_proxy", 0.0).fillna(0.0)
    working["adaptive_trade_helped_proxy_flag"] = working["adaptive_trade_delta_weight"] * next_return > 0.0
    working["adaptive_trade_cost_proxy_flag"] = working["adaptive_trade_delta_weight"] * next_return < 0.0

    columns = [
        "schema_version",
        "datetime",
        "instrument",
        "label_status",
        "tradable_status",
        "factor_industry_theme",
        "cluster_mapping_cluster",
        "cluster_mapping_reason",
        "cluster_mapping_name",
        "pool_bucket",
        "pool_selected_flag",
        "pool_unselected_flag",
        "pool_selected_rank",
        "pool_selected_next_return",
        "pool_unselected_next_return",
        "pool_selected_next_return_rank_pct_desc",
        "pool_unselected_next_return_rank_pct_desc",
        "pool_active_transmitted_flag",
        "pool_unselected_active_transmitted_flag",
        "pool_selected_but_zero_weight_flag",
        "pool_unselected_but_zero_weight_flag",
        "in_raw_panel_flag",
        "in_bridge_flag",
        "next_return",
        "mean_prediction",
        "mean_rank_pct_desc",
        "mean_zscore_by_date",
        "mean_direction_hit",
        "mean_signed_error",
        "mean_abs_error",
        "mean_signal_x_next_return",
        "mean_weighted_error",
        "mean_active_transmitted_flag",
        "forecast_variance_used",
        "forecast_variance_raw",
        "forecast_variance_robust",
        "realized_sq_return_next",
        "realized_var_5d_trailing",
        "realized_var_20d_trailing",
        "variance_error",
        "variance_abs_error",
        "variance_calibration_ratio",
        "variance_floor_trigger_flag",
        "variance_floor_helped_proxy",
        "covariance_status",
        "covariance_avg_corr_to_active",
        "covariance_weight_contribution",
        "covariance_marginal_to_mvo",
        "realized_active_basket_return_pre_l3",
        "realized_active_basket_cross_return_proxy",
        "covariance_realized_proxy_error",
        "covariance_condition_number",
        "covariance_effective_rank",
        "sigma_port_pre_l3",
        "sigma_port_effective_l3",
        "forecast_weight_status",
        "forecast_weight_selected_n",
        "mvo_selected_flag",
        "mvo_weight_rank_pct_desc",
        "mvo_target_weight_pre_l3",
        "mvo_expected_contribution",
        "mvo_realized_contribution_pre_l3",
        "mvo_cap_floor_binding_flag",
        "mvo_infeasible_or_fallback_flag",
        "l3_model_target_weight",
        "exposure_scale",
        "effective_exposure",
        "risk_macro_budget",
        "risk_bench_state_label",
        "risk_bench_trend_on",
        "risk_bench_vol_stress",
        "risk_bench_trend_score",
        "risk_portfolio_scale",
        "risk_signal_scale",
        "risk_regime_factor",
        "risk_vol_scale",
        "risk_cluster_scale",
        "risk_signal_quality_scale",
        "l3_weight_delta",
        "l3_return_delta_proxy",
        "l3_cut_helped_flag",
        "l3_cut_cost_flag",
        "is_rebalance_day",
        "rebalance_reason",
        "rebalance_gap_days",
        "trigger_portfolio_drift",
        "trigger_exposure_change",
        "trigger_macro_budget_change",
        "trigger_macro_state_flip",
        "trigger_risk_scale_change",
        "trigger_adaptive_score",
        "trigger_expected_return_gain",
        "trigger_variance_change",
        "trigger_correlation_improvement",
        "trigger_turnover",
        "adaptive_final_weight",
        "adaptive_cash_weight_by_date",
        "model_vs_actual_drift_weight",
        "adaptive_trade_delta_weight",
        "adaptive_no_trade_carry_forward_flag",
        "adaptive_no_trade_opportunity_cost_proxy",
        "adaptive_realized_return_contribution",
        "adaptive_trade_helped_proxy_flag",
        "adaptive_trade_cost_proxy_flag",
        "adaptive_join_status",
        "route_a_join_status",
    ]
    return working[[column for column in columns if column in working.columns]].reset_index(drop=True)


def _compute_layer_quality_daily_summary(layer_panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date_value, day_frame in layer_panel.groupby("datetime", sort=True):
        realized = day_frame[day_frame["label_status"].eq("realized")]
        selected = realized[realized["pool_selected_flag"]]
        unselected = realized[realized["pool_unselected_flag"]]
        mvo_selected = realized[realized["mvo_selected_flag"]]
        mvo_unselected = realized[~realized["mvo_selected_flag"]]
        rows.append(
            {
                "datetime": pd.Timestamp(date_value),
                "rows": int(day_frame.shape[0]),
                "realized_rows": int(realized.shape[0]),
                "pending_rows": int(day_frame["label_status"].eq("pending").sum()),
                "pool_universe_count": int(day_frame.shape[0]),
                "pool_selected_count": int(day_frame["pool_selected_flag"].sum()),
                "pool_unselected_count": int(day_frame["pool_unselected_flag"].sum()),
                "pool_missing_raw_count": int(day_frame["tradable_status"].eq("missing_raw").sum()),
                "pool_missing_bridge_count": int(day_frame["tradable_status"].eq("missing_bridge").sum()),
                "pool_selected_equal_weight_next_return": float(_numeric(selected["next_return"]).mean()),
                "pool_unselected_equal_weight_next_return": float(_numeric(unselected["next_return"]).mean()),
                "pool_selected_vs_unselected_spread": float(
                    _numeric(selected["next_return"]).mean() - _numeric(unselected["next_return"]).mean()
                ),
                "pool_selected_active_transmitted_rate": float(day_frame.loc[day_frame["pool_selected_flag"], "pool_active_transmitted_flag"].mean())
                if day_frame["pool_selected_flag"].any()
                else float("nan"),
                "pool_unselected_active_transmitted_rate": float(
                    day_frame.loc[day_frame["pool_unselected_flag"], "pool_unselected_active_transmitted_flag"].mean()
                )
                if day_frame["pool_unselected_flag"].any()
                else float("nan"),
                "pool_selected_but_zero_weight_rate": float(
                    day_frame.loc[day_frame["pool_selected_flag"], "pool_selected_but_zero_weight_flag"].mean()
                )
                if day_frame["pool_selected_flag"].any()
                else float("nan"),
                "pool_unselected_but_zero_weight_rate": float(
                    day_frame.loc[day_frame["pool_unselected_flag"], "pool_unselected_but_zero_weight_flag"].mean()
                )
                if day_frame["pool_unselected_flag"].any()
                else float("nan"),
                "mean_ic_pearson": _safe_corr(realized["mean_prediction"], realized["next_return"], "pearson"),
                "mean_ic_spearman": _safe_corr(realized["mean_prediction"], realized["next_return"], "spearman"),
                "mean_direction_hit_rate": float(_numeric(realized["mean_direction_hit"]).mean()),
                "mean_abs_error_mean": float(_numeric(realized["mean_abs_error"]).mean()),
                "mean_top20_bottom20_next_return_spread": _top_bottom_spread_by_score(realized, "mean_prediction", "next_return"),
                "variance_abs_error_mean": float(_numeric(realized["variance_abs_error"]).mean()),
                "variance_calibration_ratio_median": float(_numeric(realized["variance_calibration_ratio"]).median()),
                "variance_floor_trigger_rate": float(_numeric(day_frame["variance_floor_trigger_flag"]).mean()),
                "variance_floor_helped_rate": float(_numeric(realized["variance_floor_helped_proxy"]).mean()),
                "covariance_condition_number": _first_finite(day_frame.get("covariance_condition_number", pd.Series(dtype=float))),
                "covariance_effective_rank": _first_finite(day_frame.get("covariance_effective_rank", pd.Series(dtype=float))),
                "sigma_port_pre_l3": _first_finite(day_frame.get("sigma_port_pre_l3", pd.Series(dtype=float))),
                "sigma_port_effective_l3": _first_finite(day_frame.get("sigma_port_effective_l3", pd.Series(dtype=float))),
                "mvo_selected_count": int(day_frame["mvo_selected_flag"].sum()),
                "mvo_selected_equal_weight_next_return": float(_numeric(mvo_selected["next_return"]).mean()),
                "mvo_unselected_equal_weight_next_return": float(_numeric(mvo_unselected["next_return"]).mean()),
                "mvo_selected_vs_unselected_spread": float(_numeric(mvo_selected["next_return"]).mean() - _numeric(mvo_unselected["next_return"]).mean()),
                "mvo_realized_contribution_pre_l3_sum": float(_numeric(realized["mvo_realized_contribution_pre_l3"]).sum()),
                "mvo_infeasible_or_fallback_count": int(day_frame["mvo_infeasible_or_fallback_flag"].sum()),
                "l3_return_delta_proxy_sum": float(_numeric(realized["l3_return_delta_proxy"]).sum()),
                "l3_cut_helped_count": int(realized["l3_cut_helped_flag"].sum()),
                "l3_cut_cost_count": int(realized["l3_cut_cost_flag"].sum()),
                "adaptive_realized_return_contribution_sum": float(_numeric(realized["adaptive_realized_return_contribution"]).sum()),
                "adaptive_no_trade_opportunity_cost_proxy_sum": float(_numeric(realized["adaptive_no_trade_opportunity_cost_proxy"]).sum()),
                "adaptive_model_actual_abs_drift_sum": float(_numeric(day_frame["model_vs_actual_drift_weight"]).abs().sum()),
                "adaptive_trade_turnover_proxy": float(_numeric(day_frame["adaptive_trade_delta_weight"]).abs().sum() / 2.0),
            }
        )
    return pd.DataFrame(rows)


def _compute_layer_quality_instrument_summary(layer_panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for instrument, instrument_frame in layer_panel.groupby("instrument", sort=True):
        realized = instrument_frame[instrument_frame["label_status"].eq("realized")]
        rows.append(
            {
                "instrument": instrument,
                "pool_bucket": "selected" if bool(instrument_frame["pool_selected_flag"].any()) else "unselected",
                "pool_selected_flag": bool(instrument_frame["pool_selected_flag"].any()),
                "pool_selected_rank": _first_finite(instrument_frame.get("pool_selected_rank", pd.Series(dtype=float))),
                "cluster_mapping_cluster": _first_finite(instrument_frame.get("cluster_mapping_cluster", pd.Series(dtype=float))),
                "cluster_mapping_reason_latest": _first_text(instrument_frame.get("cluster_mapping_reason", pd.Series(dtype=object))),
                "cluster_mapping_name_latest": _first_text(instrument_frame.get("cluster_mapping_name", pd.Series(dtype=object))),
                "factor_industry_theme_latest": _first_text(instrument_frame.get("factor_industry_theme", pd.Series(dtype=object))),
                "rows": int(instrument_frame.shape[0]),
                "realized_rows": int(realized.shape[0]),
                "pool_missing_raw_days": int(instrument_frame["tradable_status"].eq("missing_raw").sum()),
                "pool_missing_bridge_days": int(instrument_frame["tradable_status"].eq("missing_bridge").sum()),
                "avg_next_return": float(_numeric(realized["next_return"]).mean()),
                "mean_direction_hit_rate": float(_numeric(realized["mean_direction_hit"]).mean()),
                "mean_abs_error_mean": float(_numeric(realized["mean_abs_error"]).mean()),
                "avg_mean_rank_pct_desc": float(_numeric(realized["mean_rank_pct_desc"]).mean()),
                "variance_abs_error_mean": float(_numeric(realized["variance_abs_error"]).mean()),
                "variance_calibration_ratio_median": float(_numeric(realized["variance_calibration_ratio"]).median()),
                "mvo_selected_days": int(instrument_frame["mvo_selected_flag"].sum()),
                "adaptive_active_days": int((_numeric(instrument_frame["adaptive_final_weight"]) > EPS).sum()),
                "l3_return_delta_proxy_sum": float(_numeric(realized["l3_return_delta_proxy"]).sum()),
                "adaptive_realized_return_contribution_sum": float(_numeric(realized["adaptive_realized_return_contribution"]).sum()),
                "adaptive_no_trade_opportunity_cost_proxy_sum": float(_numeric(realized["adaptive_no_trade_opportunity_cost_proxy"]).sum()),
            }
        )
    return pd.DataFrame(rows)


def _layer_quality_schema() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "primary_key": ["datetime", "instrument"],
        "label_semantics": {
            "label_status=pending": "The row has prediction/weight diagnostics but no next close label yet.",
            "label_status=realized": "The row has next_return and can be used for accuracy metrics.",
            "next_return": "Close-to-next-close realized return for the forecast date.",
        },
        "layers": {
            "candidate_pool": [
                "pool_bucket",
                "pool_selected_flag",
                "pool_unselected_flag",
                "pool_selected_rank",
                "tradable_status",
                "pool_active_transmitted_flag",
                "pool_unselected_active_transmitted_flag",
            ],
            "mean_prediction": ["mean_prediction", "mean_direction_hit", "mean_abs_error", "mean_top20_bottom20_next_return_spread"],
            "variance_prediction": ["forecast_variance_used", "realized_sq_return_next", "variance_calibration_ratio"],
            "covariance": ["covariance_marginal_to_mvo", "covariance_realized_proxy_error", "sigma_port_effective_l3"],
            "mvo": ["mvo_selected_flag", "mvo_realized_contribution_pre_l3", "mvo_infeasible_or_fallback_flag"],
            "l3": [
                "l3_weight_delta",
                "l3_return_delta_proxy",
                "l3_cut_helped_flag",
                "l3_cut_cost_flag",
                "risk_macro_budget",
                "risk_bench_state_label",
                "effective_exposure",
                "risk_portfolio_scale",
                "risk_signal_scale",
            ],
            "adaptive_rebalance": ["adaptive_final_weight", "model_vs_actual_drift_weight", "adaptive_no_trade_opportunity_cost_proxy"],
        },
        "interpretation_boundary": "Layer quality is observed attribution. Counterfactual parameter effects still require rerunning downstream stages.",
    }


def _write_layer_quality_report(
    output_paths: AuditOutputPaths,
    daily_summary: pd.DataFrame,
    instrument_summary: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    fund_list_csv = (
        metadata.get("run_config", {}).get("mean_overlay_factor_fund_list_csv")
        or metadata.get("inputs", {}).get("fund_list_csv_path")
        or DEFAULT_FUND_LIST_CSV
    )
    fund_list_name_map = _load_fund_list_name_map(fund_list_csv)
    instrument_summary_named = _add_name_column_after_instrument(instrument_summary, fund_list_name_map)
    if daily_summary.empty:
        body = "暂无层级质量数据。"
        selected_vs_unselected = "暂无 selected vs unselected 对照数据。"
    else:
        issue_columns = [
            "datetime",
            "mean_ic_spearman",
            "mean_top20_bottom20_next_return_spread",
            "variance_calibration_ratio_median",
            "l3_return_delta_proxy_sum",
            "adaptive_no_trade_opportunity_cost_proxy_sum",
        ]
        issue_columns = [column for column in issue_columns if column in daily_summary.columns]
        body = _markdown_table(daily_summary[issue_columns].tail(12))
        compare_columns = [
            "datetime",
            "pool_universe_count",
            "pool_selected_count",
            "pool_unselected_count",
            "pool_selected_equal_weight_next_return",
            "pool_unselected_equal_weight_next_return",
            "pool_selected_vs_unselected_spread",
            "pool_selected_active_transmitted_rate",
            "pool_unselected_active_transmitted_rate",
        ]
        compare_columns = [column for column in compare_columns if column in daily_summary.columns]
        selected_vs_unselected = _markdown_table(daily_summary[compare_columns].tail(12))
    worst_instruments = instrument_summary_named.sort_values("adaptive_realized_return_contribution_sum").head(12)
    weakest_selected = instrument_summary_named[instrument_summary_named["pool_selected_flag"].eq(True)].sort_values("avg_next_return").head(12)
    strongest_unselected = instrument_summary_named[instrument_summary_named["pool_selected_flag"].eq(False)].sort_values(
        "avg_next_return", ascending=False
    ).head(12)
    narrative_lines = _build_layer_quality_narrative(daily_summary, instrument_summary_named)
    lines = [
        "# 每日层级质量审计",
        "",
        f"生成时间：{metadata['generated_at']}",
        "",
        "## 输出文件",
        "",
        f"- 层级主表：{output_paths.layer_quality_panel}",
        f"- 每日层级汇总：{output_paths.layer_quality_daily_summary}",
        f"- 标的层级汇总：{output_paths.layer_quality_instrument_summary}",
        f"- 最新 pending/realized 切片：{output_paths.layer_quality_latest_pending}",
        f"- Schema：{output_paths.layer_quality_schema}",
        "",
        "## 最近重点结论",
        "",
        *narrative_lines,
        "",
        "## 最近每日质量概览",
        "",
        body,
        "",
        "## 已选池 vs 未选池",
        "",
        selected_vs_unselected,
        "",
        "## 最弱已选标的",
        "",
        _markdown_table(weakest_selected),
        "",
        "## 最强未选标的",
        "",
        _markdown_table(strongest_unselected),
        "",
        "## 主要拖累标的",
        "",
        _markdown_table(worst_instruments),
        "",
        "## 解释边界",
        "",
        "T 日生产行在 T+1 close 到来前可能仍是 pending。只有 label_status=realized 的日期，才能比较 selected vs unselected 的 realized next return；pending 日期只能先看覆盖和传导，不应提前下收益结论。",
    ]
    output_paths.layer_quality_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    output_paths: AuditOutputPaths,
    daily_summary: pd.DataFrame,
    instrument_summary: pd.DataFrame,
    cluster_coverage: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    focus = daily_summary[daily_summary["datetime"].dt.strftime("%Y-%m-%d").isin(FOCUS_DATES)].copy()
    issue_columns = [
        "datetime",
        "portfolio_realized_return_contribution",
        "cluster_selected_equal_weight_next_return",
        "cluster_unselected_equal_weight_next_return",
        "cluster_selected_vs_unselected_spread",
        "l3_gross_weight_delta",
        "active_names_pre_l3",
        "effective_holdings_pre_l3",
        "covariance_condition_number",
        "risk_signal_quality_rolling_ic",
        "top_negative_contributors",
    ]
    issue_columns = [column for column in issue_columns if column in daily_summary.columns]
    issue_days = daily_summary.sort_values("l3_gross_weight_delta", ascending=False)[issue_columns].head(10)
    top_instruments = instrument_summary.sort_values("sum_realized_return_contribution").head(12)
    report_lines = [
        "# Daily Parameter Effect Audit",
        "",
        f"Generated: {metadata['generated_at']}",
        "",
        "## Outputs",
        "",
        f"- Per-instrument daily: {output_paths.instrument_daily}",
        f"- Daily weight decision panel: {output_paths.decision_panel}",
        f"- Adaptive portfolio weights: {output_paths.portfolio_weights}",
        f"- Latest portfolio weights: {output_paths.latest_portfolio_weights}",
        f"- Rebalance event log: {output_paths.rebalance_event_log}",
        f"- Daily summary: {output_paths.daily_summary}",
        f"- Instrument summary: {output_paths.instrument_summary}",
        f"- Covariance diagnostics: {output_paths.covariance_daily}",
        f"- Cluster mapping coverage: {output_paths.cluster_mapping_coverage}",
        f"- Metadata: {output_paths.metadata_json}",
        f"- Decision panel schema: {output_paths.schema_json}",
        "",
        "## Run Summary",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {
                        "days": metadata["row_counts"].get("days"),
                        "instrument_daily_rows": metadata["row_counts"].get("instrument_daily_rows"),
                        "decision_panel_rows": metadata["row_counts"].get("decision_panel_rows"),
                        "portfolio_weight_rows": metadata["row_counts"].get("portfolio_weight_rows"),
                        "cluster_mapping_codes": metadata["row_counts"].get("cluster_mapping_codes"),
                        "cluster_mapping_universe_codes": metadata["row_counts"].get("cluster_mapping_universe_codes"),
                        "bridge_rows": metadata["row_counts"].get("bridge_rows"),
                        "route_a_status": metadata.get("route_a", {}).get("status"),
                    }
                ]
            )
        ),
        "",
        "## Focus Dates",
        "",
        _markdown_table(focus[issue_columns] if not focus.empty else focus),
        "",
        "## Selected vs Unselected Tail",
        "",
        _markdown_table(
            daily_summary[
                [
                    column
                    for column in [
                        "datetime",
                        "cluster_selected_count",
                        "cluster_unselected_count",
                        "cluster_selected_equal_weight_next_return",
                        "cluster_unselected_equal_weight_next_return",
                        "cluster_selected_vs_unselected_spread",
                    ]
                    if column in daily_summary.columns
                ]
            ].tail(10)
        ),
        "",
        "## Largest L3 Cuts",
        "",
        _markdown_table(issue_days),
        "",
        "## Cluster Coverage Tail",
        "",
        _markdown_table(cluster_coverage.tail(10)),
        "",
        "## Worst Instrument Attribution",
        "",
        _markdown_table(top_instruments),
        "",
        "## Interpretation Boundary",
        "",
        "This artifact is a read-only observed decomposition. Exact causal effects for a parameter require a counterfactual replay with that parameter changed and all downstream stages rerun.",
    ]
    output_paths.report_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def generate_daily_parameter_effect_audit(
    *,
    bridge_csv_path: str | Path,
    output_dir: str | Path,
    cluster_mapping_path: str | Path | None = None,
    cluster_mapping_full_csv_path: str | Path | None = None,
    route_a_output_dir: str | Path | None = None,
    raw_panel_path: str | Path | None = None,
    daily_nav_path: str | Path | None = None,
    weights_path: str | Path | None = None,
    weights_full_path: str | Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    provider_uri: str | None = None,
    run_config: dict[str, Any] | None = None,
    audit_level: str = "simple",
) -> dict[str, Path]:
    audit_level = str(audit_level or "simple").lower()
    if audit_level not in {"simple", "full"}:
        raise ValueError(f"audit_level must be one of simple/full, got {audit_level}")
    output_paths = _resolve_output_paths(output_dir)
    output_paths.output_dir.mkdir(parents=True, exist_ok=True)

    bridge_path = Path(bridge_csv_path).expanduser().resolve()
    cluster_path = Path(cluster_mapping_path).expanduser().resolve() if cluster_mapping_path else None
    cluster_full_path = Path(cluster_mapping_full_csv_path).expanduser().resolve() if cluster_mapping_full_csv_path else None
    route_a_path = Path(route_a_output_dir).expanduser().resolve() if route_a_output_dir else None
    raw_path = Path(raw_panel_path).expanduser().resolve() if raw_panel_path else None
    daily_nav_resolved = Path(daily_nav_path).expanduser().resolve() if daily_nav_path else None
    weights_resolved = Path(weights_path).expanduser().resolve() if weights_path else None
    weights_full_resolved = Path(weights_full_path).expanduser().resolve() if weights_full_path else None

    bridge_frame = _load_bridge_frame(bridge_path, start_date, end_date)
    selected_cluster_mapping = _load_cluster_mapping(cluster_path)
    full_cluster_mapping = _load_cluster_mapping_full_csv(cluster_full_path)
    cluster_mapping = _combine_cluster_mappings(selected_cluster_mapping, full_cluster_mapping)
    raw_presence = _load_raw_presence(raw_path, start_date, end_date)
    route_a_panel, route_a_meta = _load_route_a_panel(route_a_path)
    backtest_positions = _load_backtest_positions(weights_full_resolved or weights_resolved, start_date, end_date)
    daily_nav = _load_daily_nav(daily_nav_resolved, start_date, end_date)

    working = _build_base_frame(bridge_frame, cluster_mapping, raw_presence)
    working = _attach_route_a(working, route_a_panel, route_a_meta)
    working = _add_observed_effect_columns(working)
    cov_instrument, cov_daily = _compute_covariance_outputs(working)
    if not cov_instrument.empty:
        working = working.merge(cov_instrument, on=["datetime", "instrument"], how="left")
    else:
        working["covariance_row_status"] = "unavailable"
    if not cov_daily.empty:
        working = working.merge(cov_daily, on="datetime", how="left")
    working = _attach_adaptive_backtest(working, backtest_positions, daily_nav)

    decision_panel = _build_decision_panel(working)
    portfolio_weights, latest_portfolio_weights = _build_portfolio_weights(decision_panel)
    rebalance_event_log = _build_rebalance_event_log(decision_panel)
    simple_daily_summary = _build_simple_daily_summary(decision_panel)
    simple_instrument_drilldown = _build_simple_instrument_drilldown(decision_panel)
    schema = _decision_panel_schema()
    latest_portfolio_weights_dated = output_paths.latest_portfolio_weights
    if end_date:
        latest_portfolio_weights_dated = output_paths.output_dir / f"latest_weights_{end_date.replace('-', '')}.csv"

    cluster_coverage = pd.DataFrame()
    daily_summary = pd.DataFrame()
    instrument_summary = pd.DataFrame()
    layer_quality_panel = pd.DataFrame()
    layer_quality_daily_summary = pd.DataFrame()
    layer_quality_instrument_summary = pd.DataFrame()
    layer_quality_latest = pd.DataFrame()
    if audit_level == "full":
        cluster_coverage = _compute_cluster_mapping_coverage(working)
        daily_summary = _compute_daily_summary(working, cov_daily)
        instrument_summary = _compute_instrument_summary(working)
        layer_quality_panel = _build_layer_quality_panel(decision_panel)
        layer_quality_daily_summary = _compute_layer_quality_daily_summary(layer_quality_panel)
        layer_quality_instrument_summary = _compute_layer_quality_instrument_summary(layer_quality_panel)
        if not layer_quality_panel.empty:
            latest_quality_date = layer_quality_panel["datetime"].max()
            layer_quality_latest = layer_quality_panel[layer_quality_panel["datetime"].eq(latest_quality_date)].copy()

    output_manifest = {
        "simple_latest_weights": str(output_paths.simple_latest_weights),
        "simple_daily_summary": str(output_paths.simple_daily_summary),
        "simple_instrument_drilldown": str(output_paths.simple_instrument_drilldown),
        "simple_manifest_json": str(output_paths.simple_manifest_json),
        "readme_md": str(output_paths.readme_md),
    }
    if audit_level == "full":
        output_manifest.update(
            {
                "instrument_daily": str(output_paths.instrument_daily),
                "decision_panel": str(output_paths.decision_panel),
                "portfolio_weights": str(output_paths.portfolio_weights),
                "latest_portfolio_weights": str(output_paths.latest_portfolio_weights),
                "latest_portfolio_weights_dated": str(latest_portfolio_weights_dated),
                "rebalance_event_log": str(output_paths.rebalance_event_log),
                "daily_summary": str(output_paths.daily_summary),
                "instrument_summary": str(output_paths.instrument_summary),
                "covariance_daily": str(output_paths.covariance_daily),
                "cluster_mapping_coverage": str(output_paths.cluster_mapping_coverage),
                "layer_quality_panel": str(output_paths.layer_quality_panel),
                "layer_quality_daily_summary": str(output_paths.layer_quality_daily_summary),
                "layer_quality_instrument_summary": str(output_paths.layer_quality_instrument_summary),
                "layer_quality_latest_pending": str(output_paths.layer_quality_latest_pending),
                "layer_quality_schema": str(output_paths.layer_quality_schema),
                "layer_quality_report": str(output_paths.layer_quality_report),
                "metadata_json": str(output_paths.metadata_json),
                "schema_json": str(output_paths.schema_json),
                "report_md": str(output_paths.report_md),
            }
        )

    metadata = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "simple_decision_audit" if audit_level == "simple" else "read_only_observed_parameter_effect_decomposition",
        "audit_level": audit_level,
        "run_profile": str((run_config or {}).get("run_profile", "")),
        "run_profile_overrides": _json_ready((run_config or {}).get("run_profile_overrides", {})),
        "inputs": {
            "bridge_csv_path": str(bridge_path),
            "bridge_csv_sha256": _sha256_file(bridge_path),
            "cluster_mapping_path": str(cluster_path) if cluster_path else "",
            "cluster_mapping_sha256": _sha256_file(cluster_path),
            "cluster_mapping_full_csv_path": str(cluster_full_path) if cluster_full_path else "",
            "cluster_mapping_full_csv_sha256": _sha256_file(cluster_full_path),
            "route_a_output_dir": str(route_a_path) if route_a_path else "",
            "raw_panel_path": str(raw_path) if raw_path else "",
            "raw_panel_sha256": _sha256_file(raw_path),
            "daily_nav_path": str(daily_nav_resolved) if daily_nav_resolved else "",
            "daily_nav_sha256": _sha256_file(daily_nav_resolved),
            "weights_path": str(weights_resolved) if weights_resolved else "",
            "weights_sha256": _sha256_file(weights_resolved),
            "weights_full_path": str(weights_full_resolved) if weights_full_resolved else "",
            "weights_full_sha256": _sha256_file(weights_full_resolved),
            "provider_uri": provider_uri or "",
            "start_date": start_date or "",
            "end_date": end_date or "",
        },
        "row_counts": {
            "days": int(working["datetime"].nunique()),
            "instrument_daily_rows": int(working.shape[0]),
            "decision_panel_rows": int(decision_panel.shape[0]),
            "portfolio_weight_rows": int(portfolio_weights.shape[0]),
            "rebalance_event_rows": int(rebalance_event_log.shape[0]),
            "simple_daily_summary_rows": int(simple_daily_summary.shape[0]),
            "simple_instrument_drilldown_rows": int(simple_instrument_drilldown.shape[0]),
            "bridge_rows": int(bridge_frame.shape[0]),
            "cluster_mapping_codes": int(selected_cluster_mapping.shape[0]),
            "cluster_mapping_universe_codes": int(cluster_mapping.shape[0]),
            "cluster_mapping_full_codes": int(full_cluster_mapping.shape[0]),
            "route_a_rows": int(route_a_panel.shape[0]) if not route_a_panel.empty else 0,
            "backtest_position_rows": int(backtest_positions.shape[0]) if not backtest_positions.empty else 0,
            "layer_quality_rows": int(layer_quality_panel.shape[0]) if audit_level == "full" else 0,
            "layer_quality_daily_rows": int(layer_quality_daily_summary.shape[0]) if audit_level == "full" else 0,
            "layer_quality_instrument_rows": int(layer_quality_instrument_summary.shape[0]) if audit_level == "full" else 0,
        },
        "route_a": route_a_meta,
        "run_config": _json_ready(run_config or {}),
        "outputs": output_manifest,
        "interpretation_boundary": (
            "Observed decomposition attributes actual values and realized contribution. "
            "Exact causal effects require counterfactual reruns."
        ),
    }

    simple_daily_summary.to_csv(output_paths.simple_daily_summary, index=False)
    simple_instrument_drilldown.to_csv(output_paths.simple_instrument_drilldown, index=False)
    latest_portfolio_weights.to_csv(output_paths.simple_latest_weights, index=False)
    output_paths.simple_manifest_json.write_text(json.dumps(_json_ready(metadata), indent=2, ensure_ascii=False), encoding="utf-8")
    _write_simple_readme(output_paths, metadata, simple_daily_summary)

    outputs: dict[str, Path] = {
        "simple_latest_weights": output_paths.simple_latest_weights,
        "simple_daily_summary": output_paths.simple_daily_summary,
        "simple_instrument_drilldown": output_paths.simple_instrument_drilldown,
        "simple_manifest_json": output_paths.simple_manifest_json,
        "readme_md": output_paths.readme_md,
    }
    if audit_level == "full":
        working.to_csv(output_paths.instrument_daily, index=False)
        decision_panel.to_csv(output_paths.decision_panel, index=False)
        portfolio_weights.to_csv(output_paths.portfolio_weights, index=False)
        latest_portfolio_weights.to_csv(output_paths.latest_portfolio_weights, index=False)
        if latest_portfolio_weights_dated != output_paths.latest_portfolio_weights:
            latest_portfolio_weights.to_csv(latest_portfolio_weights_dated, index=False)
        rebalance_event_log.to_csv(output_paths.rebalance_event_log, index=False)
        daily_summary.to_csv(output_paths.daily_summary, index=False)
        instrument_summary.to_csv(output_paths.instrument_summary, index=False)
        cov_daily.to_csv(output_paths.covariance_daily, index=False)
        cluster_coverage.to_csv(output_paths.cluster_mapping_coverage, index=False)
        layer_quality_panel.to_csv(output_paths.layer_quality_panel, index=False)
        layer_quality_daily_summary.to_csv(output_paths.layer_quality_daily_summary, index=False)
        layer_quality_instrument_summary.to_csv(output_paths.layer_quality_instrument_summary, index=False)
        layer_quality_latest.to_csv(output_paths.layer_quality_latest_pending, index=False)
        output_paths.layer_quality_schema.write_text(
            json.dumps(_json_ready(_layer_quality_schema()), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        output_paths.schema_json.write_text(json.dumps(_json_ready(schema), indent=2, ensure_ascii=False), encoding="utf-8")
        output_paths.metadata_json.write_text(json.dumps(_json_ready(metadata), indent=2, ensure_ascii=False), encoding="utf-8")
        _write_report(output_paths, daily_summary, instrument_summary, cluster_coverage, metadata)
        _write_layer_quality_report(output_paths, layer_quality_daily_summary, layer_quality_instrument_summary, metadata)
        outputs.update(
            {
                "instrument_daily": output_paths.instrument_daily,
                "decision_panel": output_paths.decision_panel,
                "portfolio_weights": output_paths.portfolio_weights,
                "latest_portfolio_weights": output_paths.latest_portfolio_weights,
                "latest_portfolio_weights_dated": latest_portfolio_weights_dated,
                "rebalance_event_log": output_paths.rebalance_event_log,
                "daily_summary": output_paths.daily_summary,
                "instrument_summary": output_paths.instrument_summary,
                "covariance_daily": output_paths.covariance_daily,
                "cluster_mapping_coverage": output_paths.cluster_mapping_coverage,
                "layer_quality_panel": output_paths.layer_quality_panel,
                "layer_quality_daily_summary": output_paths.layer_quality_daily_summary,
                "layer_quality_instrument_summary": output_paths.layer_quality_instrument_summary,
                "layer_quality_latest_pending": output_paths.layer_quality_latest_pending,
                "layer_quality_schema": output_paths.layer_quality_schema,
                "layer_quality_report": output_paths.layer_quality_report,
                "metadata_json": output_paths.metadata_json,
                "schema_json": output_paths.schema_json,
                "report_md": output_paths.report_md,
            }
        )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a daily per-instrument parameter effect audit.")
    parser.add_argument("--bridge-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cluster-mapping-path", default=None)
    parser.add_argument("--cluster-mapping-full-csv", default=None)
    parser.add_argument("--route-a-output-dir", default=None)
    parser.add_argument("--raw-panel-csv", default=None)
    parser.add_argument("--daily-nav-csv", default=None)
    parser.add_argument("--weights-csv", default=None)
    parser.add_argument("--weights-full-csv", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--provider-uri", default=None)
    parser.add_argument("--audit-level", default="simple", choices=["simple", "full"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_daily_parameter_effect_audit(
        bridge_csv_path=args.bridge_csv,
        output_dir=args.output_dir,
        cluster_mapping_path=args.cluster_mapping_path,
        cluster_mapping_full_csv_path=args.cluster_mapping_full_csv,
        route_a_output_dir=args.route_a_output_dir,
        raw_panel_path=args.raw_panel_csv,
        daily_nav_path=args.daily_nav_csv,
        weights_path=args.weights_csv,
        weights_full_path=args.weights_full_csv,
        start_date=args.start_date,
        end_date=args.end_date,
        provider_uri=args.provider_uri,
        audit_level=args.audit_level,
    )
    print(f"[INFO] Simple audit README: {outputs['readme_md']}")


if __name__ == "__main__":
    main()
