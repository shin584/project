"""Sequential, append-only study accounting over an acknowledged filesystem.

Receipts are trust anchors supplied by the caller, retained outside this root.
This journal provides integrity and conservative accounting, not writer locks
or proof that a mounted Drive has replicated bytes to Google's remote service.
"""

import copy
import hashlib
import json
import math
import os
import re
import time
import uuid
from pathlib import Path

from cas12a.nt_data import canonical_bytes, digest

LIMIT_SECONDS = 21_600
STUDY_ID = "cas12a-nt-138"


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class WallClock:
    monotonic = staticmethod(time.monotonic)
    utc = staticmethod(time.time)


class DirectoryStore:
    """Immutable writes with fsync, rename, and exact-byte readback."""

    def __init__(self, root):
        self.root = Path(root).resolve()

    def path(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError("storage path escapes study root")
        return path

    def read(self, name):
        return self.path(name).read_bytes()

    def put(self, name, data):
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError(f"will not replace existing artifact: {name}")
        temporary = path.with_name(path.name + ".pending-" + uuid.uuid4().hex)
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.rename(path)
        if self.read(name) != data:
            raise OSError(f"storage readback mismatch: {name}")
        return hashlib.sha256(data).hexdigest()


def _positive(value, description):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{description} must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{description} must be a finite positive number")
    return float(value)


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", value
    ):
        raise ValueError("run/session ID must be a simple nonempty name")
    return value


class Study:
    """A single sequential study; every run shares its charges and open job."""

    def __init__(self, root, *, clock=None, store=None):
        self.store = store or DirectoryStore(root)
        if self.store.root != Path(root).resolve():
            raise ValueError("storage root mismatch")
        self.clock = clock or WallClock()
        self.state = None
        self.head = None
        self.index = -1
        self.poisoned = False
        self.started = None
        self.close_ack = None

    @classmethod
    def create(cls, root, *, fixture=False, clock=None, store=None):
        study = cls(root, clock=clock, store=store)
        if study.store.root.exists() and any(study.store.root.iterdir()):
            raise ValueError("initialize requires a new empty study root")
        study.state = {
            "schema": 1,
            "study_id": STUDY_ID,
            "fixture": bool(fixture),
            "limit_seconds": LIMIT_SECONDS,
            "charged_seconds": 0.0,
            "open_job": None,
            "last_closed": None,
            "runs": {},
            "evidence": {},
        }
        study._append(study.state, "initialize")
        return study

    @classmethod
    def reopen(cls, root, *, receipt, clock=None, store=None):
        study = cls(root, clock=clock, store=store)
        study._load()
        if (
            not isinstance(receipt, dict)
            or receipt.get("head") != study.head
            or receipt.get("index") != study.index
        ):
            raise ValueError("expected acknowledged ledger tip is missing or stale")
        closed = study.state["last_closed"]
        if closed:
            ack = receipt.get("close_ack")
            if not isinstance(ack, dict) or ack.get("job_id") != closed["job_id"]:
                raise ValueError("missing final close acknowledgement")
            elapsed = _positive(
                ack.get("elapsed_seconds"), "close acknowledgement elapsed"
            )
            if elapsed > closed["charged_seconds"]:
                raise ValueError("budget overrun during final storage acknowledgement")
            study.close_ack = copy.deepcopy(ack)
        return study

    def _load(self):
        if list(self.store.root.rglob("*.pending-*")):
            raise ValueError(
                "ambiguous pending storage write; manual resolution required"
            )
        paths = sorted((self.store.root / "ledger").glob("*.json"))
        if not paths:
            raise ValueError("missing ledger; refusing automatic initialization")
        previous, charged = None, 0.0
        for index, path in enumerate(paths):
            if path.name != f"{index:08d}.json":
                raise ValueError("ledger sequence gap")
            envelope = json.loads(path.read_bytes())
            payload = envelope["payload"]
            if (
                envelope.get("sha256") != digest(payload)
                or payload["previous"] != previous
                or payload["index"] != index
            ):
                raise ValueError("ledger hash/chain mismatch")
            state = payload["state"]
            if (
                state["study_id"] != STUDY_ID
                or state["schema"] != 1
                or state["limit_seconds"] != LIMIT_SECONDS
            ):
                raise ValueError("ledger study identity mismatch")
            if (
                not math.isfinite(state["charged_seconds"])
                or state["charged_seconds"] < charged
            ):
                raise ValueError("ledger charge regression")
            charged, previous = state["charged_seconds"], envelope["sha256"]
        self.state, self.head, self.index = state, previous, len(paths) - 1
        for run in state["runs"].values():
            for name, record in run["files"].items():
                if (
                    hashlib.sha256(self.store.read(name)).hexdigest()
                    != record["sha256"]
                ):
                    raise ValueError(f"artifact hash mismatch: {name}")
        for name, expected in state["evidence"].items():
            if hashlib.sha256(self.store.read(name)).hexdigest() != expected:
                raise ValueError("termination evidence hash mismatch")

    def _ensure(self):
        if self.poisoned:
            raise ValueError(
                "storage/budget failure: manager is poisoned; reopen required"
            )
        old_head = self.head
        try:
            self._load()
        except Exception:
            self.poisoned = True
            raise
        if self.head != old_head:
            self.poisoned = True
            raise ValueError("ledger tip changed; explicit reopen required")
        if self.state["charged_seconds"] > LIMIT_SECONDS:
            raise ValueError("study budget exceeded; explicit budget review required")
        if self.state.get("blocked_reason"):
            raise ValueError(
                "study budget reservation overrun; explicit review required"
            )

    def _overrun(self, elapsed):
        state = self.status()
        state["blocked_reason"] = "reservation overrun"
        if state["open_job"]:
            state["open_job"]["heartbeat_seconds"] = max(
                elapsed, state["open_job"]["heartbeat_seconds"]
            )
        else:
            closed = state["last_closed"]
            extra = max(0.0, elapsed - closed["charged_seconds"])
            state["charged_seconds"] += extra
            closed["charged_seconds"] += extra
        # Best effort evidence; any failure still leaves the manager poisoned
        # and prevents a receipt acknowledging a known budget overrun.
        self.started = None
        try:
            self._append(state, "budget-overrun-blocked")
        finally:
            self.poisoned = True
        raise ValueError(
            "budget reservation overrun, including storage acknowledgement"
        )

    def _append(self, state, event):
        payload = {
            "index": self.index + 1,
            "previous": self.head,
            "event": event,
            "state": state,
        }
        new_head = digest(payload)
        try:
            self.store.put(
                f"ledger/{self.index + 1:08d}.json",
                canonical_bytes({"payload": payload, "sha256": new_head}),
            )
            self.state, self.head, self.index = (
                copy.deepcopy(state),
                new_head,
                self.index + 1,
            )
            if self.started is not None:
                elapsed = self.clock.monotonic() - self.started
                job = state["open_job"] or state["last_closed"]
                if elapsed > job["reserved_seconds"]:
                    self._overrun(elapsed)
        except Exception:
            self.poisoned = True
            raise

    @property
    def receipt(self):
        if self.poisoned:
            raise ValueError("cannot acknowledge failed storage or budget overrun")
        return {
            "head": self.head,
            "index": self.index,
            "close_ack": copy.deepcopy(self.close_ack),
        }

    def status(self):
        return copy.deepcopy(self.state)

    def artifacts(self, run_id):
        self._ensure()
        run = self.state["runs"][run_id]
        try:
            return [
                (record, json.loads(self.store.read(name)))
                for name, record in sorted(run["files"].items())
                if record["kind"] == "predictions"
            ]
        except Exception:
            self.poisoned = True
            raise

    def register(self, run_id, identity, *, snapshots=None):
        self._ensure()
        _name(run_id)
        if self.state["open_job"]:
            raise ValueError(
                "unresolved job: termination evidence required before changing runs"
            )
        if run_id in self.state["runs"]:
            if self.state["runs"][run_id]["identity"] != identity:
                raise ValueError("run identity mismatch")
            return
        state = self.status()
        files = {}
        try:
            for name, data in (snapshots or {}).items():
                path = f"runs/{run_id}/snapshot/{name}"
                files[path] = {
                    "sha256": self.store.put(path, data),
                    "kind": "snapshot",
                    "coverage": [],
                }
        except Exception:
            self.poisoned = True
            raise
        state["runs"][run_id] = {
            "identity": copy.deepcopy(identity),
            "files": files,
            "sessions": [],
        }
        self._append(state, "register-run")

    def start(self, run_id, session_id, reserved_seconds, environment):
        self._ensure()
        _name(session_id)
        reserved_seconds = _positive(reserved_seconds, "reservation")
        if self.state["open_job"]:
            raise ValueError("unresolved job: termination evidence required")
        if self.state["charged_seconds"] + reserved_seconds > LIMIT_SECONDS:
            raise ValueError("study budget cannot cover reservation")
        if run_id not in self.state["runs"]:
            raise ValueError("unknown run identity")
        self.started = self.clock.monotonic()
        state = self.status()
        state["open_job"] = {
            "job_id": uuid.uuid4().hex,
            "run_id": run_id,
            "session_id": session_id,
            "started_utc": self.clock.utc(),
            "reserved_seconds": reserved_seconds,
            "heartbeat_seconds": 0.0,
        }
        state["runs"][run_id]["sessions"].append(
            {"session_id": session_id, "environment": environment}
        )
        self._append(state, "start-before-model-load")

    def guard(self):
        """Check elapsed budget at each GPU boundary without a storage write."""
        if self.poisoned:
            raise ValueError(
                "storage/budget failure: manager is poisoned; reopen required"
            )
        if self.started is None or not self.state["open_job"]:
            raise ValueError(
                "job belongs to another session; termination evidence required"
            )
        elapsed = max(0.0, self.clock.monotonic() - self.started)
        if elapsed > self.state["open_job"]["reserved_seconds"]:
            self._overrun(elapsed)
        return elapsed

    def checkpoint(self):
        self._ensure()
        elapsed = self.guard()
        state = self.status()
        state["open_job"]["heartbeat_seconds"] = max(
            elapsed, state["open_job"]["heartbeat_seconds"]
        )
        self._append(state, "heartbeat")

    def record_execution(self, environment, parity):
        self.checkpoint()
        state = self.status()
        run = state["runs"][state["open_job"]["run_id"]]
        run["sessions"][-1].update(environment=environment, parity=parity)
        self._append(state, "verified-session-execution")

    def record_failure(self, error):
        self.checkpoint()
        state = self.status()
        state["open_job"]["failure"] = str(error)
        self._append(state, "failed-job-awaiting-termination-evidence")

    def save_artifact(self, name, payload, coverage):
        self.checkpoint()
        state = self.status()
        run_id = state["open_job"]["run_id"]
        run = state["runs"][run_id]
        covered = {
            row
            for record in run["files"].values()
            if record["kind"] == "predictions"
            for row in record["coverage"]
        }
        if (
            not coverage
            or list(coverage) != sorted(set(coverage))
            or covered.intersection(coverage)
        ):
            raise ValueError("overlapping or unordered shard coverage")
        path = f"runs/{run_id}/artifacts/{_name(name)}.json"
        try:
            checksum = self.store.put(path, canonical_bytes(payload))
        except Exception:
            self.poisoned = True
            raise
        self.checkpoint()
        state = self.status()
        state["runs"][run_id]["files"][path] = {
            "sha256": checksum,
            "coverage": list(coverage),
            "kind": "predictions",
        }
        self._append(state, "commit-shard")

    def finish(self, *, outcome="completed"):
        self.checkpoint()
        state = self.status()
        job = state["open_job"]
        state["charged_seconds"] += job["reserved_seconds"]
        state["last_closed"] = {
            **job,
            "charged_seconds": job["reserved_seconds"],
            "outcome": outcome,
        }
        state["open_job"] = None
        self._append(state, "finish")
        elapsed = max(1e-9, self.clock.monotonic() - self.started)
        if elapsed > job["reserved_seconds"]:
            self._overrun(elapsed)
        self.close_ack = {"job_id": job["job_id"], "elapsed_seconds": elapsed}
        self.started = None

    def settle(self, evidence_path, *, terminated_utc, upper_seconds):
        """Manual recovery after externally confirmed termination, never heartbeat expiry."""
        self._ensure()
        job = self.state["open_job"]
        if not job:
            raise ValueError("no unresolved job; duplicate settlement rejected")
        if self.started is not None:
            raise ValueError("settlement requires a new session after termination")
        upper_seconds = _positive(upper_seconds, "termination upper bound")
        terminated_utc = _positive(terminated_utc, "termination UTC")
        if terminated_utc > self.clock.utc() or terminated_utc < job["started_utc"]:
            raise ValueError("invalid termination UTC")
        if upper_seconds < max(
            job["heartbeat_seconds"], terminated_utc - job["started_utc"]
        ):
            raise ValueError("upper bound does not cover the recorded interval")
        evidence = Path(evidence_path).read_bytes()
        if not evidence.strip():
            raise ValueError("empty termination evidence")
        name = f"evidence/{job['job_id']}.bin"
        state = self.status()
        try:
            state["evidence"][name] = self.store.put(name, evidence)
        except Exception:
            self.poisoned = True
            raise
        charge = max(job["reserved_seconds"], upper_seconds)
        state["charged_seconds"] += charge
        state["last_closed"] = {
            **job,
            "charged_seconds": charge,
            "terminated_utc": terminated_utc,
            "outcome": "interrupted",
        }
        state["open_job"] = None
        self._append(state, "settle-with-external-termination-evidence")
        self.close_ack = {"job_id": job["job_id"], "elapsed_seconds": upper_seconds}
        return self.receipt
