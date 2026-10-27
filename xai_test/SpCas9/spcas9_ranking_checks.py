"""Run disposable #83 checks with an external cumulative time limit.

Usage: python -m SpCas9.spcas9_ranking_checks --ledger PATH
Never supplies published data or starts a full fit.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from SpCas9.spcas9_data import BASE_DIR, library_versions
from SpCas9.spcas9_split import write_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args(argv)
    path = args.ledger.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    # Reserve before reading: simultaneous parents must not both spend the
    # same remaining allowance. A killed parent leaves a fail-closed lock.
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "started_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        )
    try:
        return _run_checks(path)
    finally:
        lock.unlink()


def _run_checks(path: Path) -> int:
    record: dict = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {
            "optimizer_updates_charged": 0,
            "training_seconds_charged": 0.0,
            "sessions": [],
        }
    )
    if any(session["status"] == "running" for session in record["sessions"]):
        raise ValueError(
            "unfinished check: reconcile elapsed upper bound before further training"
        )
    remaining = 300 - record["training_seconds_charged"]
    if remaining <= 1 or record["optimizer_updates_charged"] + 6 > 10:
        raise ValueError("synthetic check allowance exhausted; do not reset the ledger")
    session_id = uuid.uuid4().hex
    session = {
        "id": session_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_limit_seconds": remaining - 1,
        "optimizer_updates_reserved": 0,
        "interpreter": sys.executable,
        "versions": library_versions(("tensorflow", "keras", "numpy", "pytest")),
    }
    record["sessions"].append(session)
    write_json(record, path)
    environment = dict(os.environ)
    environment.update(
        SPCAS9_SYNTHETIC_LEDGER=str(path), SPCAS9_SYNTHETIC_SESSION=session_id
    )
    started = time.monotonic()
    status = "failed"
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(BASE_DIR / "test_spcas9_ranking.py"),
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            cwd=BASE_DIR,
            env=environment,
            timeout=remaining - 1,
            check=False,
        )
        return_code = result.returncode
        status = "finished" if return_code == 0 else "failed"
    except subprocess.TimeoutExpired:
        status, return_code = "timed_out", 124
    except KeyboardInterrupt:
        status, return_code = "interrupted", 130
    finally:
        # subprocess.run kills/reaps the child on timeout or interruption.
        # Reserve one second above for termination/accounting overhead.
        record = json.loads(path.read_text(encoding="utf-8"))
        elapsed = time.monotonic() - started
        record["training_seconds_charged"] += elapsed
        record["sessions"][-1].update(
            status=status,
            duration_seconds=elapsed,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        write_json(record, path)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
