#!/usr/bin/env python3
"""Generate 3-state regime transition board CSV (frozen monthly HMM MVP).

Example:
  PY=~/etf-daily-output/python/envs/py312/bin/python
  $PY src/etf_daily/scripts/generate_regime_transition_board.py \\
    --pack-dir ~/temp/decision_packs/20260629
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from etf_daily.scripts.generate_fat_tail_risk_report import load_fat_tail_data
from etf_daily.lib.config import load_decision_pack_config
from etf_daily.lib.fat_tail_risk import FatTailConfig, fat_tail_config_from_raw
from etf_daily.lib.pack_live_prices import apply_pack_live_closes, load_live_close_from_pack
from etf_daily.lib.regime_transition_board import (
    REGIME_TRANSITION_COLUMNS,
    RegimeTransitionConfig,
    compute_frozen_transition_step,
    fit_frozen_regime_model,
    format_regime_transition_row,
    regime_transition_config_from_raw,
)
from etf_daily.lib.regime_transition_validation import last_trading_day_before_month, month_key


# Extended columns for MVP board (compat columns kept first).
MVP_EXTRA_COLUMNS: tuple[str, ...] = (
    "p_switch_hmm",
    "p_into_reversal_hmm",
    "p_into_reversal_adj",
    "p_state_reversal",
    "return_z",
    "pred_cdf",
    "quantile_switch",
    "switch_side",
    "label_ambiguous",
    "ambiguity_reasons",
    "model_train_end",
    "signal_hint",
)


def regime_transition_csv_path(pack_dir: Path) -> Path:
    return pack_dir / "cluster_mapping_selected_regime_transition.csv"


def _month_train_end(calendar: pd.DatetimeIndex, as_of: pd.Timestamp) -> pd.Timestamp:
    mk = month_key(as_of)
    end = last_trading_day_before_month(calendar, mk)
    if end is None:
        # first month in panel: use as_of itself minus lookback is handled by fit window
        prior = calendar[calendar < as_of]
        return pd.Timestamp(prior[-1]) if len(prior) else as_of
    return pd.Timestamp(end)


def build_regime_transition_csv(
    cluster_frame: pd.DataFrame,
    close_panel: pd.DataFrame,
    as_of: str,
    *,
    board_cfg: RegimeTransitionConfig,
) -> pd.DataFrame:
    cols = list(REGIME_TRANSITION_COLUMNS) + list(MVP_EXTRA_COLUMNS)
    if cluster_frame.empty:
        return pd.DataFrame(columns=cols)

    end = pd.Timestamp(as_of)
    calendar = pd.DatetimeIndex(close_panel.index).sort_values()
    train_end = _month_train_end(calendar, end)

    rows: list[dict[str, Any]] = []
    for _, row in cluster_frame.iterrows():
        code = str(row["code"])
        name = str(row.get("name") or code)
        if code not in close_panel.columns:
            continue
        series = pd.to_numeric(close_panel[code], errors="coerce").dropna()
        if series.empty:
            continue
        model = fit_frozen_regime_model(series, train_end, cfg=board_cfg)
        metrics, _alpha, _step = compute_frozen_transition_step(
            series, end, model, cfg=board_cfg
        )
        close_val = float(series.iloc[-1]) if len(series) else None
        out = format_regime_transition_row(
            code=code,
            name=name,
            close=close_val,
            as_of=as_of,
            metrics=metrics,
        )
        # MVP fields (numeric, not percent-formatted)
        # Locked: predictive-quantile switch (bilateral q*=0.90) → OBSERVE.
        hint = "OBSERVE" if metrics.quantile_switch else "NONE"
        strong_jump = (
            metrics.transition_score >= 0.70
            and metrics.transition_direction == "up"
            and metrics.return_z is not None
            and metrics.return_z >= 2.0
        )
        if strong_jump:
            hint = "OBSERVE"
        if (
            not metrics.label_ambiguous
            and metrics.p_into_reversal_adj is not None
            and metrics.p_into_reversal_adj >= 0.20
            and metrics.transition_direction == "up"
        ):
            hint = "OBSERVE"
        out.update(
            {
                "p_switch_hmm": metrics.p_switch_hmm,
                "p_into_reversal_hmm": metrics.p_into_reversal_hmm,
                "p_into_reversal_adj": metrics.p_into_reversal_adj,
                "p_state_reversal": metrics.p_state_reversal,
                "return_z": metrics.return_z,
                "pred_cdf": metrics.pred_cdf,
                "quantile_switch": metrics.quantile_switch,
                "switch_side": metrics.switch_side,
                "label_ambiguous": metrics.label_ambiguous,
                "ambiguity_reasons": "|".join(metrics.ambiguity_reasons),
                "model_train_end": metrics.model_train_end or train_end.strftime("%Y-%m-%d"),
                "signal_hint": hint,
            }
        )
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=cols)

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["_transition_score", "code"],
        ascending=[False, True],
        kind="stable",
        na_position="last",
    )
    return frame.drop(columns=["_transition_score"], errors="ignore").reindex(columns=cols)


def build_regime_transition_board(
    *,
    pack_dir: Path,
    cluster_mapping_path: Path | None = None,
    provider_uri: str | Path | None = None,
    fat_cfg: FatTailConfig | None = None,
    board_cfg: RegimeTransitionConfig | None = None,
) -> tuple[str, pd.DataFrame, int]:
    pack_cfg = load_decision_pack_config()
    fat_cfg = fat_cfg or fat_tail_config_from_raw(pack_cfg.raw.get("fat_tail"))
    board_cfg = board_cfg or regime_transition_config_from_raw(
        pack_cfg.raw.get("regime_transition")
    )
    defensive_codes = frozenset(c for c in pack_cfg.defensive_codes)

    as_of, _, cluster_frame, _ = load_fat_tail_data(
        pack_dir,
        cluster_mapping_path=cluster_mapping_path,
        provider_uri=provider_uri,
        defensive_codes=defensive_codes,
        cfg=fat_cfg,
    )

    from etf_daily.scripts.generate_fat_tail_risk_report import (
        build_universe_codes,
        load_holdings_from_pack,
        warmup_start,
    )
    from etf_daily.lib.symbol_trend_board import load_cluster_mapping_codes
    from etf_daily.pipeline.price_adjustments import auto_qfq_adjust_close_panel
    from etf_daily.pipeline.strategy_layer_audit import load_qlib_close
    from etf_daily.paths import CLUSTER_MAPPING_SELECTED_TXT, QLIB_PROVIDER_URI

    holdings = load_holdings_from_pack(pack_dir)
    cluster_path = Path(cluster_mapping_path or CLUSTER_MAPPING_SELECTED_TXT)
    cluster_codes = load_cluster_mapping_codes(cluster_path) if cluster_path.exists() else ()
    # Prefer selected-cluster universe for the board (holdings still allowed as extras).
    universe = build_universe_codes(holdings, cluster_codes, defensive_codes)
    uri = str(provider_uri or QLIB_PROVIDER_URI)
    start = warmup_start(as_of, fat_cfg)
    panel = load_qlib_close(uri, universe, start, as_of)
    panel, _ = auto_qfq_adjust_close_panel(panel)
    live_closes = load_live_close_from_pack(pack_dir)
    panel, live_stats = apply_pack_live_closes(panel, as_of, live_closes)

    csv_frame = build_regime_transition_csv(
        cluster_frame, panel, as_of, board_cfg=board_cfg
    )
    return as_of, csv_frame, live_stats.injected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate regime transition board CSV")
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cluster-mapping", type=Path, default=None)
    parser.add_argument("--provider-uri", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pack_dir = args.pack_dir.expanduser().resolve()
    as_of, frame, live_n = build_regime_transition_board(
        pack_dir=pack_dir,
        cluster_mapping_path=args.cluster_mapping,
        provider_uri=args.provider_uri,
    )
    out = args.output or regime_transition_csv_path(pack_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[OK] regime transition board as_of={as_of} rows={len(frame)} live_injected={live_n}")
    print(f"[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
