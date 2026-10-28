"""One explicitly authorized replacement of #88's interrupted first fit.

The original failed attempt and its time remain immutable. This exception grants
one new six-run execution, at most seven aggregate attempts and the same six-hour
cumulative budget. A new failure still stops everything; retries cannot nest.
"""

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

from SpCas9.spcas9_budget import BudgetLedger, read, synthetic_record
from SpCas9.spcas9_split import file_sha256


def _parent_history(path: Path, synthetic: dict) -> tuple[dict, dict]:
    from SpCas9.spcas9_study import StudyProtocol

    parent = StudyProtocol(path)
    config = parent.execution_configuration()
    status = BudgetLedger(path).status()
    attempts = status["attempts"]
    if (
        "retry" in config
        or len(attempts) != 1
        or attempts[0]["recipe"] != "baseline"
        or attempts[0]["seed"] != 42
        or attempts[0]["status"] not in ("failed", "interrupted", "timed_out")
        or status["training_seconds"] is None
        or status["remaining_seconds"] <= 0
        or config["synthetic"] != synthetic
    ):
        raise ValueError(
            "retry requires one accounted failed first fit and original synthetic history"
        )
    owner = read(Path(synthetic["path"] + ".study-owner.json"))
    if owner != {
        "study": str(path.resolve()),
        "config_sha256": file_sha256(path / "execution-config.json"),
    }:
        raise ValueError("original synthetic budget ownership differs")
    return config, status


@contextmanager
def retry_handoff(parent: Path, synthetic_path: Path, authorization: str):
    parent, synthetic_path = parent.resolve(), synthetic_path.resolve()
    if not authorization.strip():
        raise ValueError("explicit retry authorization is required")
    lock = synthetic_path.with_suffix(synthetic_path.suffix + ".lock")
    with BudgetLedger(parent).locked():
        with lock.open("x", encoding="utf-8") as handle:
            json.dump(
                {"pid": os.getpid(), "purpose": "authorized one-time retry"}, handle
            )
        try:
            if (parent / "authorized-retry.json").exists():
                raise FileExistsError("authorized retry already reserved")
            synthetic = synthetic_record(synthetic_path, lock_owned=True)
            config, status = _parent_history(parent, synthetic)
            journal = (parent / "budget.jsonl").read_bytes()
            yield (
                synthetic,
                {
                    "authorization": authorization,
                    "parent_study": str(parent),
                    "parent_config_sha256": file_sha256(
                        parent / "execution-config.json"
                    ),
                    "parent_protocol_sha256": file_sha256(parent / "protocol.json"),
                    "parent_journal_bytes": len(journal),
                    "parent_journal_sha256": hashlib.sha256(journal).hexdigest(),
                    "parent_event_count": len(BudgetLedger(parent).events()),
                    "parent_attempts": status["attempts"],
                    "carried_training_seconds": status["training_seconds"],
                    "parent_mode": config["mode"],
                    "aggregate_attempt_cap": 7,
                    "policy": "one user-authorized retry; six fresh runs; previous failed attempt and all elapsed time retained; further failure stops all later fits; no nested retry",
                },
            )
        finally:
            lock.unlink()


def verify_retry(config: dict, study_path: Path) -> None:
    retry = config["retry"]
    parent = Path(retry["parent_study"])
    if (
        retry["aggregate_attempt_cap"] != 7
        or not retry["authorization"].strip()
        or file_sha256(parent / "execution-config.json")
        != retry["parent_config_sha256"]
        or file_sha256(parent / "protocol.json") != retry["parent_protocol_sha256"]
    ):
        raise ValueError("changed authorized retry identity")
    journal = (parent / "budget.jsonl").read_bytes()
    if (
        hashlib.sha256(journal[: retry["parent_journal_bytes"]]).hexdigest()
        != retry["parent_journal_sha256"]
    ):
        raise ValueError("changed original attempt history")
    _, status = _parent_history(parent, config["synthetic"])
    events = BudgetLedger(parent).events()
    if (
        status["attempts"] != retry["parent_attempts"]
        or status["training_seconds"] != retry["carried_training_seconds"]
        or config["mode"] != retry["parent_mode"]
        or any(
            e["kind"] not in ("evaluation", "bootstrap")
            for e in events[retry["parent_event_count"] :]
        )
    ):
        raise ValueError("changed retry training accounting")
    if read(parent / "authorized-retry.json") != {
        "study": str(study_path.resolve()),
        "config_sha256": file_sha256(study_path / "execution-config.json"),
        "authorization": retry["authorization"],
    }:
        raise ValueError("retry belongs to another execution")
    child_events = BudgetLedger(study_path).events()
    if (
        child_events[0]["synthetic_seconds"] != retry["carried_training_seconds"]
        or len(retry["parent_attempts"])
        + sum(e["kind"] == "started" for e in child_events)
        > 7
    ):
        raise ValueError("retry attempt/time budget cannot be reset")
