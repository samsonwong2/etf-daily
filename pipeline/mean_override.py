"""External mean override preparation for the orchestrator.

This module converts a per-instrument expected-mean CSV into Route-A-style
params/eval files before the bridge runs.  It lets the production pipeline use
full-date mean coverage without treating the overlay patch as the main signal.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import pandas as pd


def _compact_date(value: str) -> str:
    return value.replace("-", "")


def _default_output_dir(args: argparse.Namespace, start_date: str, end_date: str) -> Path:
    temp_dir = Path(args.temp_dir).expanduser().resolve()
    return temp_dir / f"route_a_mean_override_{_compact_date(start_date)}_{_compact_date(end_date)}"


def _format_compact_date(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def _infer_window_from_csv_name(csv_path: str | Path | None) -> tuple[str | None, str | None]:
    if not csv_path:
        return None, None
    match = re.search(r"(\d{8})_(\d{8})", Path(csv_path).name)
    if not match:
        return None, None
    return _format_compact_date(match.group(1)), _format_compact_date(match.group(2))


def normalize_mean_override_args(args: argparse.Namespace) -> None:
    """Normalize CLI args before path resolution.

    If --mean-override-csv is supplied, Route A is intentionally bypassed and
    the generated override directory becomes the Route A source for bridge.
    """
    if not getattr(args, "mean_override_csv", None):
        return

    inferred_start, inferred_end = _infer_window_from_csv_name(args.mean_override_csv)
    mean_start = args.mean_override_start_date or args.route_a_start_date or inferred_start or args.start_date
    mean_end = args.mean_override_end_date or inferred_end or args.end_date
    output_dir = (
        Path(args.mean_override_output_dir).expanduser().resolve()
        if args.mean_override_output_dir
        else Path(args.route_a_output_dir).expanduser().resolve()
        if args.route_a_output_dir
        else _default_output_dir(args, mean_start, mean_end)
    )

    args.mean_override_start_date = mean_start
    args.mean_override_end_date = mean_end
    args.mean_override_output_dir = str(output_dir)
    args.route_a_start_date = mean_start
    args.route_a_output_dir = str(output_dir)
    if args.mode != "reuse-route-a":
        print(
            f"[INFO] --mean-override-csv supplied; switching --mode {args.mode} -> reuse-route-a "
            "so the bridge consumes the generated full-coverage mean files."
        )
        args.mode = "reuse-route-a"


def _validate_status(status: pd.DataFrame, *, allow_incomplete: bool) -> None:
    if "status" not in status.columns:
        raise ValueError("mean override status file is missing the status column")
    bad = status.loc[status["status"].astype(str) != "ok"].copy()
    if bad.empty:
        return
    if allow_incomplete:
        print(f"[WARN] Mean override has {len(bad)} incomplete instruments; continuing by request.")
        return
    preview = bad[["instrument", "status", "message"]].head(20).to_string(index=False)
    raise ValueError(
        "Mean override coverage is incomplete. Pass --mean-override-allow-incomplete "
        f"to continue anyway. First bad rows:\n{preview}"
    )


def _write_manifest(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    status: pd.DataFrame,
    annualized_csv: Path,
    cluster_mapping: Path,
) -> None:
    manifest = {
        "mode": "external_mean_override",
        "trading_start_date": args.start_date,
        "trading_end_date": args.end_date,
        "mean_override_start_date": args.mean_override_start_date,
        "mean_override_end_date": args.mean_override_end_date,
        "annualized_csv": str(annualized_csv),
        "cluster_mapping": str(cluster_mapping),
        "provider_uri": str(args.provider_uri),
        "output_dir": str(output_dir),
        "mu_source": args.mean_override_mu_source,
        "mu_scale": float(args.mean_override_mu_scale),
        "winsor_quantile": args.mean_override_winsor_quantile,
        "date_source": args.mean_override_date_source,
        "tickers_requested": int(len(status)),
        "tickers_ok": int((status["status"].astype(str) == "ok").sum()),
        "mean_overlay_mode": args.mean_overlay_mode,
        "mu_preset": args.mu_preset,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def prepare_mean_override_route_a(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    """Create or validate the Route-A override directory for external means."""
    if not getattr(args, "mean_override_csv", None):
        return

    from fund_pool_builder import build_oracle_route_a_override as builder

    output_dir = paths["route_a_output_dir"]
    annualized_csv = Path(args.mean_override_csv).expanduser().resolve()
    cluster_mapping = Path(args.cluster_mapping_path).expanduser().resolve()
    start_label = _compact_date(args.mean_override_start_date)
    end_label = _compact_date(args.mean_override_end_date)

    print(
        "[INFO] Mean override input: "
        f"csv={annualized_csv} labels={start_label}_{end_label} output={output_dir}"
    )

    if getattr(args, "dry_run", False):
        print("[DRY-RUN] Skipping mean override generation.")
        return
    if not annualized_csv.exists():
        raise FileNotFoundError(f"mean override CSV not found: {annualized_csv}")
    if not cluster_mapping.exists():
        raise FileNotFoundError(f"cluster_mapping file not found: {cluster_mapping}")

    status_path = output_dir / "oracle_mean_override_status.csv"
    if output_dir.exists() and not args.mean_override_overwrite:
        if not status_path.exists():
            raise FileExistsError(
                f"mean override output dir exists without status CSV: {output_dir}. "
                "Pass --mean-override-overwrite to rebuild it."
            )
        status = pd.read_csv(status_path)
        _validate_status(status, allow_incomplete=args.mean_override_allow_incomplete)
        print(
            f"[INFO] Reusing existing mean override dir: {output_dir} "
            f"ok={(status['status'].astype(str) == 'ok').sum()}/{len(status)}"
        )
        return

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers = builder.read_cluster_tickers(cluster_mapping)
    daily_mu = builder.load_daily_mu(
        annualized_csv,
        args.mean_override_mu_source,
        float(args.mean_override_mu_scale),
    )
    daily_mu = builder.apply_winsor(daily_mu, args.mean_override_winsor_quantile)

    if args.mean_override_date_source == "qlib":
        dates_by_ticker = builder.qlib_close_dates(
            tickers,
            provider_uri=args.provider_uri,
            start_date=args.mean_override_start_date,
            end_date=args.mean_override_end_date,
        )
    else:
        dates_by_ticker = builder.business_dates(
            tickers,
            args.mean_override_start_date,
            args.mean_override_end_date,
        )

    status = builder.write_override_files(
        tickers=tickers,
        daily_mu=daily_mu,
        dates_by_ticker=dates_by_ticker,
        output_dir=output_dir,
        start_label=start_label,
        end_label=end_label,
    )
    status.to_csv(status_path, index=False)
    _validate_status(status, allow_incomplete=args.mean_override_allow_incomplete)
    _write_manifest(
        args=args,
        output_dir=output_dir,
        status=status,
        annualized_csv=annualized_csv,
        cluster_mapping=cluster_mapping,
    )
    print(
        f"[INFO] Wrote mean override Route-A files: {output_dir} "
        f"ok={(status['status'].astype(str) == 'ok').sum()}/{len(status)}"
    )