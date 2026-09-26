"""File-lock + stage-filename + subprocess runner used by every stage."""
from __future__ import annotations

import datetime as _dt
import errno
import fcntl
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from . import constants


def _safe_stage_filename(stage_name: str) -> str:
    """Return a filesystem-safe filename fragment for a stage label.

    Replaces runs of non-alphanumeric chars with single underscores so labels
    like 'Route A SH510300' become 'Route_A_SH510300'.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stage_name).strip("_")
    return cleaned or "stage"


@contextmanager
def _pipeline_lock(lock_path: Path, wait: bool = False):
    """Exclusive file lock keyed by (temp_dir, preset, start, end).

    Two concurrent orchestrator runs with the same (preset, start_date, end_date)
    would overwrite each other's temp CSVs and Route A output dir. This advisory
    lock ensures only one runs at a time. By default fails fast if another run
    already holds the lock; set wait=True to block.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        flags = fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB
        try:
            fcntl.flock(fd, flags)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EACCES):
                try:
                    with open(lock_path, "r") as f:
                        holder = f.read().strip()
                except OSError:
                    holder = "<unknown>"
                raise RuntimeError(
                    f"Another pipeline run already holds the lock at {lock_path} "
                    f"(holder={holder}). Refusing to run concurrently. "
                    f"If stale, delete the file manually."
                ) from None
            raise
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"pid={os.getpid()} argv={' '.join(sys.argv)}\n".encode())
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            lock_path.unlink()
        except OSError:
            pass


def _run_command(command: list[str], cwd: Path, stage_name: str) -> None:
    print(f"\n== {stage_name} ==")
    print(" ".join(command))
    if constants._DRY_RUN:
        print(f"[DRY-RUN] Skipping execution of stage: {stage_name}")
        return
    if constants._LOG_DIR is None:
        completed = subprocess.run(command, cwd=str(cwd), check=False)
        rc = completed.returncode
    else:
        log_path = constants._LOG_DIR / f"stage_{_safe_stage_filename(stage_name)}.log"
        # Append mode: same stage can be invoked multiple times (e.g. if the
        # user reruns after partial failure). A header line separates invocations.
        with log_path.open("a", encoding="utf-8", errors="replace") as log_f:
            header = (
                f"===== {_dt.datetime.now().isoformat(timespec='seconds')} "
                f"stage={stage_name} cwd={cwd} =====\n"
                f"{' '.join(command)}\n"
            )
            log_f.write(header)
            log_f.flush()
            proc = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
                errors="replace",
            )
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_f.write(line)
                rc = proc.wait()
            except BaseException:
                proc.kill()
                proc.wait()
                raise
            log_f.write(f"===== exit_code={rc} =====\n")
        print(f"[LOG] {stage_name} → {log_path}")
    if rc != 0:
        raise RuntimeError(f"Stage failed: {stage_name}, exit_code={rc}")
