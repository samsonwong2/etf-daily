"""Route A full / rolling-refresh execution + manifest."""
from __future__ import annotations

import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import constants
from .constants import REPO_ROOT
from .infra import _run_command


def _write_route_a_manifest(
    output_dir: Path,
    *,
    start_label: str,
    end_label: str,
    codes: list[str],
    train_window: int,
    retrain_every: int,
    training_objective: str,
    mode: str,
    extra: dict | None = None,
) -> None:
    """Write a manifest.json describing how this Route A output dir was produced.

    Downstream bridge/monitor consumers may prefer this over parsing filenames via
    `rsplit("_", 2)`. Safe to overwrite on rerun; fields are intentionally minimal.
    """
    import json
    from datetime import datetime

    git_sha = ""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=3,
        )
        if completed.returncode == 0:
            git_sha = completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass

    manifest = {
        "schema_version": 1,
        "start_label": start_label,
        "end_label": end_label,
        "codes": sorted(codes),
        "n_codes": len(codes),
        "train_window": int(train_window),
        "retrain_every": int(retrain_every),
        "training_objective": training_objective,
        "mode": mode,
        "git_sha": git_sha,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        manifest.update(extra)

    if constants._DRY_RUN:
        print(f"[DRY-RUN] Skipping manifest write (would go to {output_dir / 'manifest.json'})")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Wrote Route A manifest → {manifest_path}")


def run_route_a_full(
    *,
    codes: list[str],
    output_dir: Path,
    start_date: str,
    end_date: str,
    train_window: int,
    retrain_every: int,
    training_objective: str,
    keep_output_dir: bool,
    route_max_workers: int = 1,
) -> None:
    if output_dir.exists() and not keep_output_dir:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_workers = max(1, int(route_max_workers or 1))
    if max_workers == 1:
        for code in codes:
            _train_one_code_into(
                code=code,
                output_dir=output_dir,
                start_date=start_date,
                end_date=end_date,
                train_window=train_window,
                retrain_every=retrain_every,
                training_objective=training_objective,
                stage_label=f"Route A {code}",
            )
    else:
        print(f"[INFO] Route A full: running {len(codes)} instruments with max_workers={max_workers}")
        failures: list[tuple[str, BaseException]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_code = {
                executor.submit(
                    _train_one_code_into,
                    code=code,
                    output_dir=output_dir,
                    start_date=start_date,
                    end_date=end_date,
                    train_window=train_window,
                    retrain_every=retrain_every,
                    training_objective=training_objective,
                    stage_label=f"Route A {code}",
                ): code
                for code in codes
            }
            for future in as_completed(future_to_code):
                code = future_to_code[future]
                try:
                    future.result()
                except BaseException as exc:
                    failures.append((code, exc))
        if failures:
            failed_codes = ", ".join(code for code, _ in failures[:10])
            raise RuntimeError(
                f"Route A full failed for {len(failures)} instruments: {failed_codes}"
                + (" ..." if len(failures) > 10 else "")
            )

    _write_route_a_manifest(
        output_dir,
        start_label=start_date.replace("-", ""),
        end_label=end_date.replace("-", ""),
        codes=list(codes),
        train_window=train_window,
        retrain_every=retrain_every,
        training_objective=training_objective,
        mode="full",
        extra={"route_max_workers": int(max_workers)},
    )


def _train_one_code_into(
    *,
    code: str,
    output_dir: Path,
    start_date: str,
    end_date: str,
    train_window: int,
    retrain_every: int,
    training_objective: str,
    stage_label: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(REPO_ROOT / "strategy" / "route_a_core" / "route_a_student_t.py"),
        "--start-date", start_date,
        "--end-date", end_date,
        "--benchmark", code,
        "--runtime-profile", "quick-calibration",
        "--train-window", str(train_window),
        "--retrain-every", str(retrain_every),
        "--training-objective", training_objective,
        "--output-dir", str(output_dir),
    ]
    _run_command(command, REPO_ROOT, stage_label)


def _find_prior_route_a_dir(
    base_dir: Path,
    start_label: str,
    end_label: str,
) -> tuple[Path, str] | None:
    """Find the most recent Route A output dir with same start_label and prev_end < end_label."""
    pattern = f"outputs_route_a_quick_nll_fullpanel_{start_label}_*"
    candidates = sorted(base_dir.glob(pattern))
    best: tuple[Path, str] | None = None
    for cand in candidates:
        if not cand.is_dir():
            continue
        prev_end = cand.name.rsplit("_", 1)[-1]
        if len(prev_end) != 8 or not prev_end.isdigit():
            continue
        if prev_end >= end_label:
            continue
        if best is None or prev_end > best[1]:
            best = (cand, prev_end)
    return best


def run_route_a_rolling_refresh(
    *,
    codes: list[str],
    output_dir: Path,
    start_date: str,
    end_date: str,
    train_window: int,
    retrain_every: int,
    training_objective: str,
    keep_output_dir: bool,
    route_max_workers: int = 1,
) -> None:
    """Incremental Route A refresh: reuse prior run for most history, only re-train
    a short-window trailing segment for each code, then merge.

    Logic:
    1. Find the most recent prior `outputs_route_a_quick_nll_fullpanel_{start_label}_{prev_end}`
       dir under `strategy/route_a_core/`.
    2. If none found, fall back to run_route_a_full.
    3. Else, for each code:
       a. Short-window retrain in a temp dir using start =
          max(original_start, prev_end - buffer_days), end = new end_date.
          `buffer_days = train_window + retrain_every + 15` calendar days, which
          guarantees the last retrain step of the short run sees the same ~37-day
          window as a full run would.
       b. Merge: rows (target_date <= prev_end) from the prior file + rows
          (target_date > prev_end) from the short file, relabelled with the
          original start_label and new end_label.
       c. Copy eval_summary / readme from the prior run (metrics over the
          historical portion are still valid).
    4. Fall back to full retrain per code if prior files are missing or buffer
       would collapse (short_start <= original_start).
    """
    import pandas as pd

    base_dir = REPO_ROOT / "strategy" / "route_a_core"
    start_label = start_date.replace("-", "")
    end_label = end_date.replace("-", "")

    prior = _find_prior_route_a_dir(base_dir, start_label, end_label)
    if prior is None:
        print(
            f"[INFO] rolling-refresh: no prior Route A dir with start_label={start_label} "
            f"and prev_end<{end_label}; falling back to full mode"
        )
        run_route_a_full(
            codes=codes,
            output_dir=output_dir,
            start_date=start_date,
            end_date=end_date,
            train_window=train_window,
            retrain_every=retrain_every,
            training_objective=training_objective,
            keep_output_dir=keep_output_dir,
            route_max_workers=route_max_workers,
        )
        return

    prior_dir, prev_end_label = prior
    prev_end_ts = pd.to_datetime(prev_end_label, format="%Y%m%d")
    new_end_ts = pd.to_datetime(end_label, format="%Y%m%d")
    orig_start_ts = pd.to_datetime(start_label, format="%Y%m%d")
    # Short retrain needs enough rows after feature engineering to clear
    # train_window + 5 samples. Feature engineering typically drops ~20 warmup
    # rows for rolling stats, so aim for ~3 * train_window calendar days of
    # leeway. Trading days ~ 0.7 * calendar days.
    buffer_days = 3 * train_window + retrain_every + 30
    short_start_ts = max(orig_start_ts, prev_end_ts - pd.Timedelta(days=buffer_days))
    short_start = short_start_ts.strftime("%Y-%m-%d")
    short_start_label = short_start.replace("-", "")
    short_end_label = end_label
    short_suffix = f"{short_start_label}_{short_end_label}"
    new_suffix = f"{start_label}_{end_label}"

    print(
        f"[INFO] rolling-refresh: prior_dir={prior_dir.name} (prev_end={prev_end_ts.date()}); "
        f"new end={new_end_ts.date()}; short-retrain window [{short_start}..{end_date}] "
        f"(buffer={buffer_days} cal days)"
    )

    if output_dir.exists() and not keep_output_dir:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_root = output_dir / ".tmp_rolling"
    tmp_root.mkdir(parents=True, exist_ok=True)

    full_retrain_codes: list[str] = []
    merged_ok = 0
    merged_fail = 0

    if short_start_ts <= orig_start_ts:
        print(
            f"[INFO] rolling-refresh: short_start ({short_start}) <= original start "
            f"({start_date}); buffer too small to save work, falling back to full mode"
        )
        shutil.rmtree(tmp_root, ignore_errors=True)
        run_route_a_full(
            codes=codes,
            output_dir=output_dir,
            start_date=start_date,
            end_date=end_date,
            train_window=train_window,
            retrain_every=retrain_every,
            training_objective=training_objective,
            keep_output_dir=True,
            route_max_workers=route_max_workers,
        )
        return

    def _merge_by_date(prev_csv: Path, new_csv: Path, out_csv: Path, date_col: str) -> bool:
        if not prev_csv.exists() or not new_csv.exists():
            return False
        prev_df = pd.read_csv(prev_csv)
        new_df = pd.read_csv(new_csv)
        if date_col not in prev_df.columns or date_col not in new_df.columns:
            return False
        prev_df[date_col] = pd.to_datetime(prev_df[date_col])
        new_df[date_col] = pd.to_datetime(new_df[date_col])
        head = prev_df[prev_df[date_col] <= prev_end_ts]
        tail = new_df[new_df[date_col] > prev_end_ts]
        merged = pd.concat([head, tail], ignore_index=True)
        merged.to_csv(out_csv, index=False)
        return True

    for code in codes:
        prev_params = prior_dir / f"route_a_params_{code}_{start_label}_{prev_end_label}.csv"
        prev_eval = prior_dir / f"route_a_eval_daily_{code}_{start_label}_{prev_end_label}.csv"
        prev_quant = prior_dir / f"route_a_quantiles_{code}_{start_label}_{prev_end_label}.csv"
        prev_summary = prior_dir / f"route_a_eval_summary_{code}_{start_label}_{prev_end_label}.csv"
        prev_readme = prior_dir / f"route_a_outputs_readme_{code}_{start_label}_{prev_end_label}.txt"

        if not prev_params.exists() or not prev_eval.exists():
            print(f"[INFO] rolling-refresh: prior files missing for {code}, scheduling full retrain")
            full_retrain_codes.append(code)
            continue

        code_tmp = tmp_root / code
        try:
            _train_one_code_into(
                code=code,
                output_dir=code_tmp,
                start_date=short_start,
                end_date=end_date,
                train_window=train_window,
                retrain_every=retrain_every,
                training_objective=training_objective,
                stage_label=f"Route A rolling-refresh {code}",
            )
        except RuntimeError as exc:
            print(f"[WARN] rolling-refresh: short retrain failed for {code} ({exc}); falling back to full retrain for this code")
            full_retrain_codes.append(code)
            shutil.rmtree(code_tmp, ignore_errors=True)
            continue

        new_params = code_tmp / f"route_a_params_{code}_{short_suffix}.csv"
        new_eval = code_tmp / f"route_a_eval_daily_{code}_{short_suffix}.csv"
        new_quant = code_tmp / f"route_a_quantiles_{code}_{short_suffix}.csv"
        new_summary = code_tmp / f"route_a_eval_summary_{code}_{short_suffix}.csv"
        new_readme = code_tmp / f"route_a_outputs_readme_{code}_{short_suffix}.txt"

        ok_params = _merge_by_date(
            prev_params, new_params,
            output_dir / f"route_a_params_{code}_{new_suffix}.csv",
            "target_date",
        )
        ok_eval = _merge_by_date(
            prev_eval, new_eval,
            output_dir / f"route_a_eval_daily_{code}_{new_suffix}.csv",
            "target_date",
        )
        if prev_quant.exists() and new_quant.exists():
            _merge_by_date(
                prev_quant, new_quant,
                output_dir / f"route_a_quantiles_{code}_{new_suffix}.csv",
                "target_date",
            )

        # Summary: prefer new (reflects merged history is approximately the same
        # rolling model state; simple to just use the latest short-run summary
        # values for the trailing window). Copy the new summary but relabel; fall
        # back to prev if new is missing.
        summary_src = new_summary if new_summary.exists() else prev_summary
        if summary_src.exists():
            shutil.copy(summary_src, output_dir / f"route_a_eval_summary_{code}_{new_suffix}.csv")
        readme_src = new_readme if new_readme.exists() else prev_readme
        if readme_src.exists():
            shutil.copy(readme_src, output_dir / f"route_a_outputs_readme_{code}_{new_suffix}.txt")

        if ok_params and ok_eval:
            merged_ok += 1
        else:
            merged_fail += 1
            full_retrain_codes.append(code)

    shutil.rmtree(tmp_root, ignore_errors=True)

    if full_retrain_codes:
        print(
            f"[INFO] rolling-refresh: full-retraining {len(full_retrain_codes)} codes "
            f"without mergeable prior files"
        )
        for code in full_retrain_codes:
            _train_one_code_into(
                code=code,
                output_dir=output_dir,
                start_date=start_date,
                end_date=end_date,
                train_window=train_window,
                retrain_every=retrain_every,
                training_objective=training_objective,
                stage_label=f"Route A (fallback full) {code}",
            )

    print(
        f"[INFO] rolling-refresh summary: merged_ok={merged_ok}, merged_fail={merged_fail}, "
        f"full_fallback={len(full_retrain_codes)}"
    )

    _write_route_a_manifest(
        output_dir,
        start_label=start_label,
        end_label=end_label,
        codes=list(codes),
        train_window=train_window,
        retrain_every=retrain_every,
        training_objective=training_objective,
        mode="rolling-refresh",
        extra={
            "prior_dir": prior_dir.name,
            "prev_end_label": prev_end_label,
            "short_start_label": short_start_label,
            "merged_ok_count": int(merged_ok),
            "merged_fail_count": int(merged_fail),
            "full_fallback_count": int(len(full_retrain_codes)),
            "full_fallback_codes": sorted(full_retrain_codes),
        },
    )
