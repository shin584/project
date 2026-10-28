"""Immutable #79 execution identity and crash-persistent training accounting.

Only StudyProtocol exposes orchestration. This module owns its on-disk journal;
it never imports a training framework or authorizes a full experiment.
"""

import hashlib
import json
import logging
import math
import multiprocessing
import os
import pickle
import platform
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from SpCas9.spcas9_acceptance import BOOTSTRAP_SETTINGS, GATE_SETTINGS
from SpCas9.spcas9_data import library_versions
from SpCas9.spcas9_execution import source_record
from SpCas9.spcas9_split import file_sha256

RUN_ORDER = [
    {"recipe": recipe, "seed": seed}
    for recipe in ("baseline", "candidate")
    for seed in (42, 43, 44)
]
BUDGETS = {
    "full_fits": 6,
    "training_seconds": 21600,
    "synthetic_updates": 10,
    "synthetic_seconds": 300,
}
ENVIRONMENT_KEYS = (
    "KERAS_BACKEND",
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
    "TF_DETERMINISTIC_OPS",
    "TF_CUDNN_DETERMINISTIC",
    "PYTHONHASHSEED",
    "CUBLAS_WORKSPACE_CONFIG",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def environment_record() -> dict:
    return {
        "interpreter": sys.executable,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": library_versions(),
        "settings": {
            key: os.environ.get(key, "unknown/default") for key in ENVIRONMENT_KEYS
        },
        "device": "unknown until backend initialization; no historical equivalence asserted",
        "determinism": "only recorded settings; effective runtime determinism unverified",
    }


def _seconds(value) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("missing or invalid elapsed bound")
    return float(value)


def synthetic_record(path: Path, *, lock_owned: bool = False) -> dict:
    if not lock_owned and path.with_suffix(path.suffix + ".lock").exists():
        raise ValueError("synthetic ledger has an active or unreconciled owner")
    raw = path.read_bytes()
    record = json.loads(raw)
    seconds = _seconds(record["training_seconds_charged"])
    updates = record["optimizer_updates_charged"]
    if type(updates) is not int or not 0 <= updates <= 10 or not record["sessions"]:
        raise ValueError("missing synthetic update/session evidence")
    charged = 0.0
    reserved = 0
    for session in record["sessions"]:
        if session["status"] not in (
            "finished",
            "completed",
            "interrupted",
            "timed_out",
            "failed",
        ):
            raise ValueError("unfinished synthetic session; elapsed bound unknown")
        charged += _seconds(
            session.get("elapsed_upper_bound_seconds", session.get("duration_seconds"))
        )
        count = session["optimizer_updates_reserved"]
        if type(count) is not int or count < 0:
            raise ValueError("invalid synthetic update evidence")
        reserved += count
    if reserved != updates or not math.isclose(charged, seconds, abs_tol=1e-6):
        raise ValueError("synthetic cumulative accounting mismatch")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "record": record,
        "training_seconds": seconds,
        "optimizer_updates": updates,
        "original_allowance_exceeded": seconds > 300,
    }


@contextmanager
def synthetic_handoff(path: Path):
    """Share the check runner's reservation through snapshot and ownership transfer."""
    path = path.resolve()
    lock = path.with_suffix(path.suffix + ".lock")
    with lock.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "purpose": "fixed study handoff"}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if Path(str(path) + ".study-owner.json").exists():
            raise FileExistsError(
                "synthetic history already belongs to a study; budget reset refused"
            )
        yield synthetic_record(path, lock_owned=True)
    finally:
        lock.unlink()


def configuration(protocol: dict, mode: str, synthetic: dict) -> dict:
    if (
        mode not in ("full", "verification")
        or protocol["checkpoint_selection"] != "ngg"
    ):
        raise ValueError(
            "fixed #79 execution requires NGG selection and a known execution mode"
        )
    return {
        "schema_version": 1,
        "issue": 85,
        "mode": mode,
        "protocol": protocol,
        "source": source_record(),
        "environment": environment_record(),
        "synthetic": synthetic,
        "run_order": RUN_ORDER,
        "representative_seed": 42,
        "budgets": BUDGETS,
        "recipes": {
            "baseline": "original binary-masked MSE, sum measured SSE / actual batch rows, sum nine heads",
            "candidate": {
                "base": "baseline",
                "weight": 0.01,
                "temperature": 0.1,
                "pairs": "all unordered within batch; same measured head; both NGG; distinct 30mers; unequal labels",
                "normalization": "gap-weighted softplus / sum gaps per head; empty=0; fixed nine-head mean",
            },
            "base_rows": "batch32, original row sampling/shuffle; paired-seed initial weights and row order",
        },
        "acceptance": {
            **GATE_SETTINGS,
            "comparisons": [
                "three paired seeds vs refitted baseline",
                "candidate seed42 vs stored v2",
            ],
            "stages": [
                "validation first",
                "Test only after both pass and finalist freeze",
            ],
            "per_head_populations": ["NGG", "non-NGG"],
            "boundaries": "inclusive; baseline MAE zero requires candidate zero",
            "aggregation": "per-head metric mean over three seeds then equal-head mean; relative change of means",
            "missing_or_undefined": "incomplete",
            "pooled": "context only; not a head-average correlation",
            "historical_checkpoint_sha256": "ece5009215482d3fd2506688dc8e2423437abdcd28dfe10e4ac21c5bed3d19d6",
            "historical_test_predictions_sha256": "2a6d9a70993787c839afe48d0489b8d1542003f0c817a56ac94f35f97bbe296b",
            "test_exposure": "historically inspected; not new external confirmation",
        },
        "ci": {
            **BOOTSTRAP_SETTINGS,
            "method": "percentile protospacer-paired NGG bootstrap",
            "joint_draws": "same sampled clusters for both methods, all heads and all seeds; retain all cluster rows and multiplicity",
            "statistics": [
                "relative equal-head mean NGG MAE change",
                "equal-head mean NGG Spearman change",
            ],
            "invalid_draw": "reject jointly when either required summary is undefined",
            "insufficient_draws": "CI unavailable; report attempted/valid counts and reason",
            "scope": "supplementary, fixed three seeds; no CI adoption gate or arbitrary-seed claim",
        },
        "fixture_specification": [
            "measured zero",
            "missing/nonfinite masked labels",
            "different per-head masks",
            "mixed NGG/non-NGG",
            "repeated exact inputs",
            "tied/near-tied labels",
            "head without eligible pairs",
            "partial batch",
            "MSE objective/gradient parity rtol1e-5 atol1e-6",
            "authoritative NGG MAE/Spearman/earlier-epoch selection and reload parity",
            "exact improvement/protection boundaries and either side; zero MAE denominator",
            "seed-before-head means; ratios of means; missing seed/metric; representative model failure",
            "shared protospacer cluster multiplicity across methods/heads/seeds; repeatable CI",
            "valid joint draw quota shortfall never changes point-estimate eligibility",
        ],
        "reporting": {
            "validation": "StudyProtocol.validation_report; reconstruct hash-verified persisted predictions without candidate Test",
            "training_ngg": "selected checkpoint on measured training NGG rows; diagnostic only",
            "history": "preserve complete per-run epoch history and selected epoch",
        },
        "failure_policy": "any failed/interrupted/timed-out fit consumes attempt and stops all later fits; no retry or intermediate resume",
        "readiness": {
            "final_pre_fit_frozen": False,
            "blocked_by": [
                "#86 validation acceptance and paired-bootstrap CI",
                "#87 finalist freeze and guarded Test evaluation",
                "#88 report readiness before final pre-fit freeze",
            ],
            "verification_only": mode == "verification",
        },
    }


class BudgetLedger:
    def __init__(self, path: Path):
        self.path = path
        self.journal = path / "budget.jsonl"
        self._tail_hash = ""
        self._journal_size: int | None = None

    @contextmanager
    def locked(self):
        # Kernel locks release on process death; a leftover file is not evidence
        # of a live owner. Never remove a lock file another process could hold.
        with (self.path / "budget.lock").open("r+b") as handle:
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError(
                    "study budget is owned by another process"
                ) from error
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def initialize(self, config_sha256: str, seconds: float) -> None:
        (self.path / "budget.lock").write_bytes(b"0")
        with self.journal.open("x", encoding="utf-8"):
            pass
        self.append(
            {
                "kind": "initialized",
                "config_sha256": config_sha256,
                "synthetic_seconds": seconds,
            }
        )

    def events(self) -> list[dict]:
        events = []
        previous = ""
        with self.journal.open(encoding="utf-8") as handle:
            for line in handle:
                envelope = json.loads(line)
                event = envelope["event"]
                if (
                    not line.endswith("\n")
                    or envelope["previous"] != previous
                    or envelope["sha256"]
                    != digest({"previous": previous, "event": event})
                ):
                    raise ValueError("corrupted budget journal")
                previous = envelope["sha256"]
                events.append(event)
        if not events or events[0]["kind"] != "initialized":
            raise ValueError("missing budget history")
        if events[0]["config_sha256"] != file_sha256(
            self.path / "execution-config.json"
        ):
            raise ValueError("changed execution configuration")
        self._tail_hash = previous
        self._journal_size = self.journal.stat().st_size
        return events

    def append(self, event: dict) -> None:
        size = self.journal.stat().st_size
        if size and size != self._journal_size:
            # Validate before extending; partial writes are never repaired.
            self.events()
        envelope = {"previous": self._tail_hash if size else "", "event": event}
        event_digest = digest(envelope)
        envelope["sha256"] = event_digest
        with self.journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(envelope, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._tail_hash = event_digest
        self._journal_size = self.journal.stat().st_size

    def status(self) -> dict:
        events = self.events()
        attempts: list[dict] = []
        evaluation = 0.0
        bootstrap = 0.0
        blocked = False
        for event in events[1:]:
            kind = event["kind"]
            if kind == "started":
                attempts.append({**event, "status": "running", "elapsed_seconds": 0.0})
            elif kind in ("heartbeat", "finished", "recovered"):
                attempt = attempts[-1]
                elapsed = event["elapsed_seconds"]
                if (
                    elapsed is not None
                    and _seconds(elapsed) < attempt["elapsed_seconds"]
                ):
                    raise ValueError("budget elapsed time decreased")
                attempt.update(event)
                if kind != "heartbeat":
                    attempt["status"] = event["status"]
                    blocked |= event["status"] != "completed"
            elif kind == "evaluation":
                evaluation += _seconds(event["seconds"])
            elif kind == "bootstrap":
                bootstrap += _seconds(event["seconds"])
            else:
                raise ValueError("invalid budget event")
        # Immutable start receipts reveal a truncated/reset journal.
        receipts = list(self.path.glob("attempt-*.json"))
        if len(receipts) != len(attempts):
            raise ValueError("missing attempt history; training blocked")
        for index, attempt in enumerate(attempts):
            if read(self.path / f"attempt-{index + 1}.json") != {
                key: attempt[key] for key in ("run_id", "recipe", "seed")
            }:
                raise ValueError("attempt identity mismatch")
        unknown = any(a["elapsed_seconds"] is None for a in attempts)
        seconds = (
            None
            if unknown
            else events[0]["synthetic_seconds"]
            + sum(a["elapsed_seconds"] for a in attempts)
        )
        return {
            "attempts": attempts,
            "training_seconds": seconds,
            "evaluation_seconds": evaluation,
            "bootstrap_seconds": bootstrap,
            "blocked": blocked
            or unknown
            or any(a["status"] == "running" for a in attempts),
            "remaining_seconds": None if seconds is None else max(0.0, 21600 - seconds),
        }


def _fit_worker(
    backend, training, validation, seed, checkpoint_path, started, deadline
):
    # The child also has a watchdog: loss of the parent must not leave an
    # unbounded orphan training process. Missing termination evidence still
    # means unknown elapsed, never permission to restart.
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        os._exit(124)
    watchdog = threading.Timer(remaining, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    receipt = checkpoint_path.parent / "fit-ended.json"
    try:
        result = backend.fit(
            training, validation, seed=seed, checkpoint_path=checkpoint_path
        )
        atomic_write(
            receipt,
            {"status": "fit_finished", "elapsed_seconds": time.monotonic() - started},
        )
        with (checkpoint_path.parent / "fit-result.pkl").open("xb") as handle:
            pickle.dump(result, handle)
    except (Exception, KeyboardInterrupt, SystemExit) as error:
        logging.getLogger(__name__).exception("training worker failed")
        atomic_write(
            receipt,
            {
                "status": "failed",
                "elapsed_seconds": time.monotonic() - started,
                "error": f"{type(error).__name__}: {error}",
                "error_type": type(error).__name__,
            },
        )
    finally:
        watchdog.cancel()


class BudgetedBackend:
    """Isolate fits so the scheduler can enforce a deadline even on a hung backend."""

    def __init__(self, backend, ledger: BudgetLedger, started: float, limit: float):
        self.backend = backend
        self.ledger = ledger
        self.started = started
        self.limit = limit
        self.fit_seconds: float | None = None

    def fit(self, training, validation, *, seed, checkpoint_path):
        process = multiprocessing.get_context("spawn").Process(
            target=_fit_worker,
            args=(
                self.backend,
                training,
                validation,
                seed,
                checkpoint_path,
                self.started,
                self.started + self.limit,
            ),
            daemon=True,
        )
        try:
            process.start()
            while process.is_alive():
                elapsed = time.monotonic() - self.started
                if elapsed >= self.limit:
                    raise TimeoutError("cumulative training deadline exhausted")
                self.ledger.append({"kind": "heartbeat", "elapsed_seconds": elapsed})
                process.join(min(1.0, self.limit - elapsed))
            if process.exitcode == 124:
                raise TimeoutError("training worker deadline exhausted")
            if process.exitcode != 0:
                raise RuntimeError(f"training worker abnormal exit: {process.exitcode}")
            receipt = read(checkpoint_path.parent / "fit-ended.json")
            # Charge worker startup/serialization/termination overhead too. This
            # also bounds every heartbeat emitted while awaiting worker exit.
            self.fit_seconds = max(
                _seconds(receipt["elapsed_seconds"]), time.monotonic() - self.started
            )
            if receipt["status"] != "fit_finished":
                error_class = {
                    "TimeoutError": TimeoutError,
                    "KeyboardInterrupt": KeyboardInterrupt,
                }.get(receipt.get("error_type"), RuntimeError)
                raise error_class(receipt.get("error", "training failed"))
            if self.fit_seconds >= self.limit:
                raise TimeoutError("cumulative training deadline exhausted")
            # Our worker alone creates this local result, under exclusive study
            # ownership. Recovery never unpickles earlier or external results.
            with (checkpoint_path.parent / "fit-result.pkl").open("rb") as handle:
                return pickle.load(handle)
        finally:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                    process.join(2)
                    if process.is_alive():
                        process.kill()
                process.join()

    def predict(self, checkpoint_path, inputs):
        return self.backend.predict(checkpoint_path, inputs)
