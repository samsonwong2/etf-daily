#!/usr/bin/env python3
"""Generate next-session adaptive hold_up execution plan from a frozen HTML batch.

Example::

    PY=.../python python decision_pack/scripts/generate_adaptive_execution_plan.py \
      --adaptive-dir ~/etf-daily-output/temp/plotly_outputs/20260807all_adaptive \
      --as-of 2026-08-07 --trade-date 2026-08-10 \
      --live-holdings ~/etf-daily-output/temp/live_holdings/20260717.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runtime_paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from decision_pack.src.adaptive_execution_plan import (  # noqa: E402
    build_execution_plan_frame,
    build_symbol_execution_row,
    load_batch_open_pos,
    load_holdings_shares,
    next_trade_date,
    truncate_ohlcv_through,
    write_execution_plan_artifacts,
)
from decision_pack.src.adaptive_stage_common import prepare_features  # noqa: E402
from decision_pack.src.early_down_alert import (  # noqa: E402
    build_early_down_alert_row,
    build_early_down_frame,
    early_down_flags_frame,
)
from decision_pack.src.ed_exec_overlay import (  # noqa: E402
    DEFAULT_ED_LOOKBACK_K,
    apply_ed_overlay_to_execution_row,
    normalize_ed_overlay_mode,
    resolve_ed_trigger,
)

WARMUP_CALENDAR_DAYS = 220


def _load_ev():
    path = _PROJECT_ROOT / "decision_pack/scripts/eval_causal_regime_switch_pool.py"
    spec = importlib.util.spec_from_file_location("ev_exec_plan", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_pm():
    path = _PROJECT_ROOT / "decision_pack/scripts/plot_regime_transition_example.py"
    spec = importlib.util.spec_from_file_location("pm_exec_plan", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_ohlcv(pm, code: str, end_date: str) -> pd.DataFrame:
    ohlcv = pm.load_qlib_ohlcv(
        code,
        "2015-01-01",
        end_date,
        init_qlib=False,
        lookback_calendar_days=WARMUP_CALENDAR_DAYS,
        clip_to_window=False,
    )
    if ohlcv is None or ohlcv.empty:
        raise ValueError("empty ohlcv")
    return ohlcv


def _load_snapshot(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    need = {"code", "open", "high", "low", "close"}
    missing = need - set(c.lower() for c in df.columns)
    # tolerate original case
    colmap = {c.lower(): c for c in df.columns}
    if missing:
        raise ValueError(f"snapshot missing columns: {sorted(missing)}")
    out: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        code = str(row[colmap["code"]]).upper()
        out[code] = {
            "open": float(row[colmap["open"]]),
            "high": float(row[colmap["high"]]),
            "low": float(row[colmap["low"]]),
            "close": float(row[colmap["close"]]),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--adaptive-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260807all_adaptive",
    )
    p.add_argument("--as-of", default="2026-08-07")
    p.add_argument(
        "--trade-date",
        default=None,
        help="default: next exchange session after --as-of",
    )
    p.add_argument(
        "--live-holdings",
        type=Path,
        default=Path("~/etf-daily-output/temp/live_holdings/20260717.csv"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: workspace/decision_packs/adaptive_execution/<trade_date tag>",
    )
    p.add_argument("--snapshot", type=Path, default=None, help="optional trade-date OHLC CSV")
    p.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="rebuild markdown/json/readme from an existing execution_plan.csv",
    )
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--code", action="append", default=[])
    p.add_argument("--limit-pct", type=float, default=0.10)
    p.add_argument(
        "--ed-overlay",
        default="block_buy_and_hypo_exit",
        help=(
            "ED→exec overlay: off | block_buy | block_buy_and_hypo_exit. "
            "Default full overlay after 20260810 A/B promote_candidate; "
            "does not mutate regime labels."
        ),
    )
    p.add_argument(
        "--ed-lookback-k",
        type=int,
        default=DEFAULT_ED_LOOKBACK_K,
        help="Trailing bars for effective ED alert (1=as-of only; 5=last-K). Default 5.",
    )
    p.add_argument("--fail-fast", action="store_true")
    args = p.parse_args(argv)

    ed_overlay_mode = normalize_ed_overlay_mode(args.ed_overlay)
    ed_lookback_k = max(1, int(args.ed_lookback_k))

    trade_date = args.trade_date or next_trade_date(args.as_of)
    adaptive_dir = (
        args.adaptive_dir
        if args.adaptive_dir.is_absolute()
        else _PROJECT_ROOT / args.adaptive_dir
    )
    cfg_dir = adaptive_dir / "configs"
    summary_path = adaptive_dir / "batch_summary.csv"

    trade_tag = trade_date.replace("-", "")
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = _PROJECT_ROOT / "workspace/decision_packs/adaptive_execution" / trade_tag
    elif not out_dir.is_absolute():
        out_dir = _PROJECT_ROOT / out_dir

    shares, holdings_stale, holdings_note = load_holdings_shares(
        args.live_holdings, as_of=args.as_of
    )

    if args.from_csv is not None:
        csv_path = (
            args.from_csv if args.from_csv.is_absolute() else _PROJECT_ROOT / args.from_csv
        )
        frame = pd.read_csv(csv_path)
        ed_path = csv_path.parent / "early_down_alerts.csv"
        early_down = pd.read_csv(ed_path) if ed_path.exists() else None
        meta = {
            "as_of": args.as_of,
            "trade_date": trade_date,
            "adaptive_dir": str(adaptive_dir),
            "holdings_path": str(args.live_holdings) if args.live_holdings else None,
            "holdings_note": holdings_note,
            "holdings_stale": holdings_stale,
            "n_codes": int(len(frame)),
            "n_errors": 0,
            "execution_timing": "close_confirm",
            "limit_pct": float(args.limit_pct),
            "snapshot": str(args.snapshot) if args.snapshot else None,
            "rebuilt_from_csv": str(csv_path),
            "early_down_layer": "diagnostic_labels",
            "ed_overlay_mode": ed_overlay_mode,
            "ed_lookback_k": ed_lookback_k,
            "signal_window_start": "2026-04-01",
            "signal_window_end": trade_date,
        }
        if early_down is not None and len(early_down) and ed_overlay_mode != "off":
            ed_map = {
                str(r.get("code") or "").upper(): str(r.get("level") or "none")
                for _, r in early_down.iterrows()
            }
            patched = [
                apply_ed_overlay_to_execution_row(
                    r.to_dict(),
                    ed_level=ed_map.get(str(r.get("code") or "").upper(), "none"),
                    mode=ed_overlay_mode,
                )
                for _, r in frame.iterrows()
            ]
            frame = build_execution_plan_frame(patched)
        paths = write_execution_plan_artifacts(
            frame, out_dir, meta=meta, early_down=early_down
        )
        print(f"[OK] rebuilt from {csv_path}")
        print(f"[OK] markdown={paths['markdown']}")
        print(f"[OK] csv={paths['csv']}")
        print(f"[OK] json={paths['json']}")
        print(f"[OK] readme={paths['readme']}")
        if "signal_board_csv" in paths:
            print(f"[OK] signal_board={paths['signal_board_csv']}")
        return 0

    if not cfg_dir.is_dir() or not summary_path.exists():
        print(f"[ERROR] adaptive batch incomplete: {adaptive_dir}", file=sys.stderr)
        return 2

    summary = pd.read_csv(summary_path)
    open_pos = load_batch_open_pos(summary)
    snapshots = _load_snapshot(args.snapshot)

    codes = [c.upper() for c in args.code] if args.code else [
        str(c).upper() for c in summary["code"].tolist()
    ]
    if args.max_codes:
        codes = codes[: args.max_codes]

    name_map = {
        str(r.code).upper(): str(r.name)
        for r in summary.itertuples(index=False)
        if hasattr(r, "name")
    }

    ev = _load_ev()
    pm = _load_pm()
    pm._init_qlib(None)

    rows: list[dict[str, Any]] = []
    early_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    print(
        f"codes={len(codes)} as_of={args.as_of} trade={trade_date} "
        f"holdings={holdings_note} out={out_dir}"
    )
    for i, code in enumerate(codes, 1):
        cfg_path = cfg_dir / f"{code}.json"
        try:
            if not cfg_path.exists():
                raise FileNotFoundError(cfg_path)
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            method = str(cfg["method"])
            method_params = dict(cfg.get("method_params") or {})
            ohlcv = _load_ohlcv(pm, code, args.as_of)
            row = build_symbol_execution_row(
                code=code,
                name=name_map.get(code, code),
                method=method,
                method_params=method_params,
                ohlcv=ohlcv,
                as_of=args.as_of,
                trade_date=trade_date,
                live_shares=float(shares.get(code, 0.0)),
                holdings_stale=holdings_stale,
                model_open_hint=open_pos.get(code),
                ev=ev,
                snapshot=snapshots.get(code),
                limit_pct=float(args.limit_pct),
                data_source=adaptive_dir.name,
            )
            rows.append(row)
            # ED diagnostic labels (soft_bear/peak8); regime unchanged.
            ohlcv_asof = truncate_ohlcv_through(ohlcv, args.as_of)
            px = prepare_features(
                ev, ohlcv_asof, method=method, method_params=method_params
            )
            ed = build_early_down_alert_row(
                code=code,
                name=name_map.get(code, code),
                method=method,
                as_of=args.as_of,
                regime_asof=str(row["regime_asof"]),
                px=px,
            )
            early_rows.append(ed)
            ed_hist = early_down_flags_frame(px, regimes=px["regime"].to_numpy(object))
            trig = resolve_ed_trigger(ed_hist["level"].to_numpy(), lookback_k=ed_lookback_k)
            row = apply_ed_overlay_to_execution_row(
                row,
                ed_level=str(ed.get("level") or "none"),
                mode=ed_overlay_mode,
                ed_trigger=str(trig.get("ed_trigger") or "none"),
                ed_trigger_source=str(trig.get("ed_trigger_source") or "none"),
                lookback_k=ed_lookback_k,
            )
            rows[-1] = row
            print(
                f"[{i}/{len(codes)}] {code} regime={row['regime_asof']} "
                f"buy={row['buy_action']} hypo={row['hypo_sell_action']} "
                f"early_down={ed['level']} trigger={row.get('ed_trigger')} "
                f"src={row.get('ed_trigger_source')} overlay={row.get('ed_overlay_mode')}"
            )
        except Exception as exc:
            errors.append({"code": code, "error": str(exc)})
            print(f"[{i}/{len(codes)}] FAIL {code}: {exc}")
            if args.fail_fast:
                traceback.print_exc()
                return 1

    frame = build_execution_plan_frame(rows)
    early_down = build_early_down_frame(early_rows)
    meta = {
        "as_of": args.as_of,
        "trade_date": trade_date,
        "adaptive_dir": str(adaptive_dir),
        "holdings_path": str(args.live_holdings) if args.live_holdings else None,
        "holdings_note": holdings_note,
        "holdings_stale": holdings_stale,
        "n_codes": len(rows),
        "n_errors": len(errors),
        "execution_timing": "close_confirm",
        "limit_pct": float(args.limit_pct),
        "snapshot": str(args.snapshot) if args.snapshot else None,
        "early_down_layer": "diagnostic_labels",
        "ed_overlay_mode": ed_overlay_mode,
        "ed_lookback_k": ed_lookback_k,
        "signal_window_start": "2026-04-01",
        "signal_window_end": trade_date,
    }
    paths = write_execution_plan_artifacts(
        frame, out_dir, meta=meta, early_down=early_down
    )
    if errors:
        err_path = out_dir / "execution_errors.csv"
        pd.DataFrame(errors).to_csv(err_path, index=False, encoding="utf-8-sig")
        print(f"[WARN] errors={len(errors)} -> {err_path}")
    n_alert = int((early_down["level"] == "alert").sum()) if len(early_down) else 0
    print(f"[OK] markdown={paths['markdown']}")
    print(f"[OK] csv={paths['csv']}")
    print(f"[OK] json={paths['json']}")
    print(f"[OK] readme={paths['readme']}")
    if "signal_board_csv" in paths:
        print(f"[OK] signal_board={paths['signal_board_csv']}")
    if "early_down_csv" in paths:
        print(f"[OK] early_down={paths['early_down_csv']} alerts={n_alert}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
