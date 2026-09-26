#!/usr/bin/env python3
"""Build dual-track vol forecast → need-return research board.

Default need is fair-return scale: r_need_ann = 0.043 × σ / σ_SH510300.
Pass --sharpe-target 1.33 for the old personal-hurdle mode.

Universe sources (priority):
  1. ``--code`` repeats
  2. ``--instruments`` Qlib all.txt
  3. ``--adaptive-dir`` batch_summary / configs

Example::

    python src/etf_daily/scripts/build_vol_need_return_board.py \\
      --as-of 2026-08-10 \\
      --instruments ~/etf-daily-output/data/qlib_data/all_fund_data/instruments/all.txt \\
      --out-dir ~/etf-daily-output/temp/plotly_outputs/20260810_all_fund_data
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from etf_daily.paths import PLOTLY_OUTPUTS_DIR  # noqa: E402

from etf_daily.lib.config import load_decision_pack_config  # noqa: E402
from etf_daily.lib.vol_need_return_board import (  # noqa: E402
    ANCHOR_ERP_ANN,
    EQUITY_ANCHOR,
    build_vol_need_board,
    render_readme,
)
from etf_daily.lib.vol_return_board import (  # noqa: E402
    DEFAULT_BOND_ANCHOR,
    DEFAULT_BOND_AUX,
    DEFAULT_DEFENSIVE_CODES,
    load_fund_name_map,
    load_instrument_codes,
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--as-of", default="2026-08-10")
    p.add_argument(
        "--adaptive-dir",
        type=Path,
        default=PLOTLY_OUTPUTS_DIR / "20260810all_adaptive",
    )
    p.add_argument(
        "--instruments",
        type=Path,
        default=None,
        help="Qlib instruments all.txt; when set, uses full universe (filtered by as-of)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: ~/etf-daily-output/temp/plotly_outputs/<as_of tag>_vol_need_return_board",
    )
    p.add_argument(
        "--fund-list",
        type=Path,
        default=_ROOT / "temp/fund_list.csv",
        help="name map CSV (基金代码/基金简称)",
    )
    p.add_argument("--bond-anchor", default=DEFAULT_BOND_ANCHOR)
    p.add_argument("--code", action="append", default=[])
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument(
        "--sharpe-target",
        type=float,
        default=None,
        help=(
            "fixed sharpe_like (old personal hurdle). "
            f"Default: none → λ = erp_ann / σ_{EQUITY_ANCHOR}"
        ),
    )
    p.add_argument(
        "--erp-ann",
        type=float,
        default=ANCHOR_ERP_ANN,
        help=f"CSI300 listing-era fair premium for need scale (default {ANCHOR_ERP_ANN})",
    )
    p.add_argument(
        "--equity-anchor",
        default=EQUITY_ANCHOR,
        help=f"vol scale name (default {EQUITY_ANCHOR})",
    )
    p.add_argument(
        "--skip-mc",
        action="store_true",
        help="skip GARCH MC q05 columns (faster full-universe runs)",
    )
    return p.parse_args()


def _resolve_codes(
    adaptive_dir: Path | None,
    *,
    explicit: list[str],
    instruments: Path | None,
    as_of: str,
    bond_anchor: str,
    max_codes: int | None,
) -> tuple[list[str], dict[str, str]]:
    name_map: dict[str, str] = {}
    if adaptive_dir is not None:
        summary_path = adaptive_dir / "batch_summary.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            if "code" in summary.columns:
                for _, r in summary.iterrows():
                    code = str(r["code"]).upper()
                    name_map[code] = str(r["name"]) if "name" in summary.columns else ""

    if explicit:
        codes = [c.upper() for c in explicit]
    elif instruments is not None:
        codes = load_instrument_codes(instruments, as_of=as_of)
    else:
        codes = list(name_map.keys()) if name_map else []
        if not codes and adaptive_dir is not None:
            cfg_dir = adaptive_dir / "configs"
            if cfg_dir.is_dir():
                codes = sorted(p.stem.upper() for p in cfg_dir.glob("*.json"))

    for extra, default_name in (
        (bond_anchor, "十年国债ETF国泰"),
        (DEFAULT_BOND_AUX, "国债ETF国泰"),
    ):
        c = str(extra).upper()
        if c not in codes:
            codes.append(c)
        name_map.setdefault(c, default_name)

    if max_codes is not None:
        anchors = {str(bond_anchor).upper(), DEFAULT_BOND_AUX}
        rest = [c for c in codes if c not in anchors][: max(0, int(max_codes))]
        codes = list(dict.fromkeys([*rest, *sorted(anchors)]))
    return codes, name_map


def main() -> int:
    args = parse_args()
    instruments = args.instruments
    if instruments is not None and not instruments.is_absolute():
        instruments = _ROOT / instruments

    adaptive_dir = (
        args.adaptive_dir
        if args.adaptive_dir.is_absolute()
        else _ROOT / args.adaptive_dir
    )
    if instruments is None and not adaptive_dir.is_dir():
        print(f"[ERROR] missing adaptive-dir: {adaptive_dir}", file=sys.stderr)
        return 2

    tag = pd.Timestamp(args.as_of).strftime("%Y%m%d")
    default_out = (
        f"{tag}_all_fund_data"
        if instruments is not None
        else f"{tag}_vol_need_return_board"
    )
    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else PLOTLY_OUTPUTS_DIR / default_out
    )
    out_dir = out_dir if out_dir.is_absolute() else _ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    codes, name_map = _resolve_codes(
        adaptive_dir if adaptive_dir.is_dir() else None,
        explicit=args.code,
        instruments=instruments,
        as_of=args.as_of,
        bond_anchor=args.bond_anchor,
        max_codes=args.max_codes,
    )
    fund_list = args.fund_list if args.fund_list.is_absolute() else _ROOT / args.fund_list
    if fund_list.is_file():
        fund_names = load_fund_name_map(fund_list)
        name_map = {**name_map, **fund_names}
        print(f"fund_list names={len(fund_names)} from {fund_list}")
    else:
        print(f"[WARN] fund_list missing: {fund_list}", file=sys.stderr)
    if not codes:
        print("[ERROR] no codes resolved", file=sys.stderr)
        return 2
    print(f"universe={len(codes)} instruments={instruments} as_of={args.as_of}")

    cfg = load_decision_pack_config()
    defensive = frozenset(
        {*(str(c).upper() for c in cfg.defensive_codes), *DEFAULT_DEFENSIVE_CODES}
    )

    pm = _load(
        "pm_vol_need",
        _ROOT / "src/etf_daily/plots/plot_regime_transition_example.py",
    )
    pm._init_qlib(None)
    end = str(pd.Timestamp(args.as_of).date())

    frames: dict[str, pd.Series] = {}
    errors: list[dict[str, str]] = []
    for i, code in enumerate(codes, start=1):
        try:
            ohlcv = pm.load_qlib_ohlcv(
                code,
                "2015-01-01",
                end,
                init_qlib=False,
                lookback_calendar_days=220,
                clip_to_window=False,
            )
            if ohlcv is None or ohlcv.empty:
                raise ValueError("empty ohlcv")
            close_col = "close" if "close" in ohlcv.columns else "$close"
            if close_col not in ohlcv.columns:
                raise ValueError(f"missing close col in {list(ohlcv.columns)}")
            close = pd.to_numeric(ohlcv[close_col], errors="coerce")
            if "datetime" in ohlcv.columns:
                close.index = pd.to_datetime(ohlcv["datetime"])
            elif "as_of" in ohlcv.columns:
                close.index = pd.to_datetime(ohlcv["as_of"])
            frames[code] = close
            if i % 50 == 0 or i == len(codes):
                print(f"[{i}/{len(codes)}] loaded ok={len(frames)} err={len(errors)}")
        except Exception as exc:  # noqa: BLE001
            errors.append({"code": code, "error": str(exc)})
            if i % 50 == 0 or i == len(codes):
                print(f"[{i}/{len(codes)}] loaded ok={len(frames)} err={len(errors)}")

    if not frames:
        print("[ERROR] no price frames loaded", file=sys.stderr)
        return 3

    include_mc = not bool(args.skip_mc)
    sharpe_like = None if args.sharpe_target is None else float(args.sharpe_target)
    print(
        f"building need board sharpe_like={sharpe_like} erp_ann={args.erp_ann} "
        f"equity_anchor={args.equity_anchor} include_mc={include_mc} n={len(frames)} ..."
    )
    board, meta = build_vol_need_board(
        frames,
        args.as_of,
        bond_anchor=args.bond_anchor,
        defensive_codes=defensive,
        names=name_map,
        sharpe_like=sharpe_like,
        erp_ann=float(args.erp_ann),
        equity_anchor=str(args.equity_anchor),
        include_mc=include_mc,
    )
    meta["adaptive_dir"] = str(adaptive_dir) if adaptive_dir.is_dir() else None
    meta["instruments"] = str(instruments) if instruments is not None else None
    meta["n_requested"] = len(codes)
    meta["n_loaded"] = len(frames)
    meta["n_load_errors"] = len(errors)
    meta["errors"] = errors[:200]

    csv_path = out_dir / "vol_need_return_board.csv"
    md_path = out_dir / "README_need.md"
    meta_path = out_dir / "meta_need.json"
    board.to_csv(csv_path, index=False, float_format="%.4f")
    md_path.write_text(render_readme(meta, board), encoding="utf-8")
    meta_path.write_text(
        json.dumps(_jsonable(meta), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[OK] csv={csv_path} rows={len(board)}")
    print(f"[OK] readme={md_path}")
    print(
        f"sharpe_target={meta['sharpe_target']} n={meta['n_codes']} "
        f"garch_ok={meta['n_garch_ok']} loaded={len(frames)}/{len(codes)}"
    )
    return 0


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        if obj != obj:
            return None
        return obj
    try:
        if hasattr(obj, "item"):
            return _jsonable(obj.item())
    except Exception:
        pass
    return obj


if __name__ == "__main__":
    raise SystemExit(main())
