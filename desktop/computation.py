"""Spawned computation owner. This module is safe to import in the UI process.

The scheduler retains callables locally; only JSON renderer operations cross
the private pipe. Request and generation checks apply even to preparation.
"""

from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import threading
import time

from desktop.api import Cancelled, Dispatcher

OPERATIONS = frozenset({"warm_up", "sites_fragment", "prediction_fragments",
                        "explanation_fragment", "csv_table"})
FAMILY_FACTORIES = {
    "non_nt": "desktop.worker_renderer:NonNTComputationRenderer",
    "nt": "desktop.worker_renderer:ComputationRenderer",
}

#: Cancel grace in seconds (#218). A child that does not answer a cancel or a
#: newer request within it is force-restarted in the same family. Provisional:
#: it only has to exceed one normal batch (a Rollout counts as one); measuring
#: the final value is a separate ticket. Tests inject `cancel_grace`.
CANCEL_GRACE = 15.0

#: No-progress limit in seconds (#219). A request that nobody cancelled and
#: whose child sends no progress report for this long is a stalled computation
#: and is force-restarted. Measured from the request and from each report, so a
#: long job that keeps reporting (Cas12a SHAP, about 62 s) never hits it; the
#: job's total time has no limit. It also covers non-NT's lazy model load
#: inside the first prediction. Provisional; tests inject `no_progress_limit`.
NO_PROGRESS_LIMIT = 120.0

#: Load limit in seconds (#219). Model loading sends no progress from inside
#: the load, so the whole preparation has this limit. Past it the child is
#: stopped and preparation fails, so the screen offers "모델 준비 다시 시도"
#: (`retry-prepare`), which replaces the child. Provisional;
#: tests inject `load_limit`.
LOAD_LIMIT = 300.0


def _send(connection, message):
    connection.send_bytes(json.dumps(message, ensure_ascii=False).encode("utf-8"))


def _receive(connection):
    return json.loads(connection.recv_bytes().decode("utf-8"))


def _compute(connection, cancellation, factory_name):
    """Child entry: all renderer, runtime and predictor references live here."""
    renderer = None
    try:
        while True:
            request = _receive(connection)
            identity = {k: request[k] for k in ("generation", "request_id")}

            def reply(kind, *, identity=identity, **fields):
                _send(connection, {**identity, "type": kind, **fields})

            def progress(done, total):
                if cancellation.is_set():
                    raise Cancelled()
                reply("progress", done=int(done), total=int(total))

            try:
                if request["operation"] not in OPERATIONS:
                    raise ValueError("Unknown computation operation")
                if renderer is None:
                    module, name = factory_name.split(":")
                    renderer = getattr(importlib.import_module(module), name)()
                value = getattr(renderer, request["operation"])(
                    **request["args"], on_batch=progress)
                reply("result", value=value)
            except Cancelled:
                reply("cancelled")
            except Exception as exc:  # noqa: BLE001 -- report renderer failures over IPC
                reply("error", value=str(exc) or type(exc).__name__)
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        connection.close()


class ProcessRenderer:
    """Renderer protocol proxy; it holds no model or rendering dependencies."""

    def __init__(self, owner, factory_name):
        self._owner = owner
        self._factory_name = factory_name
        self._state = threading.Lock()
        self._process = None
        self._connection = None
        self._cancellation = None
        self._broken = False

    @property
    def pid(self):
        with self._state:
            return self._process.pid if self._process is not None else None

    @property
    def exited(self):
        with self._state:
            return self._broken or (self._process is not None and not self._process.is_alive())

    def stop(self):
        with self._state:
            if self._process is None:
                return
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(3)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(3)
            if self._process.is_alive():
                raise RuntimeError("Computation process did not exit")
            self._connection.close()
            self._process.close()
            self._process = self._connection = self._cancellation = None
            self._broken = False

    def _call(self, operation, args, on_batch, *, switching=False):
        context = self._owner._context
        generation, rid = context.generation, context.request_id
        if switching:
            # Selection has already waited for the previous process to exit.
            on_batch(1, 3)
        with self._state:
            if generation != self._owner.generation:
                raise Cancelled()
            if self._process is None:
                spawn = mp.get_context("spawn")
                parent, child = spawn.Pipe()
                cancellation = spawn.Event()
                process = spawn.Process(target=_compute,
                                        args=(child, cancellation, self._factory_name),
                                        name="dashboard-v3-computation", daemon=True)
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                self._process, self._connection = process, parent
                self._cancellation = cancellation
            connection, process = self._connection, self._process
            cancellation = self._cancellation
        identity = {"generation": generation, "request_id": rid}
        # Preparation is never cancelled, so only requests start the grace.
        cancellable = not getattr(context, "lifecycle", False)
        cancelled = False
        deadline = None  # cancel grace: the child must answer before this
        # A request must keep reporting (no-progress limit); preparation must
        # finish as a whole (load limit), whatever it reports on the way.
        limit = self._owner.no_progress_limit if cancellable else self._owner.load_limit
        silence_deadline = time.monotonic() + limit
        try:
            cancellation.clear()
            _send(connection, {**identity, "operation": operation, "args": args})
            if switching:
                on_batch(2, 3)
            while True:
                if generation != self._owner.generation:
                    raise Cancelled()
                if not cancelled and cancellable and rid != self._owner._latest:
                    # Cancelled or superseded: tell the child now, without
                    # waiting for a progress report that may never come.
                    cancelled = True
                    cancellation.set()
                if cancelled and deadline is None:
                    deadline = time.monotonic() + self._owner.cancel_grace
                if not connection.poll(0.05):
                    if not process.is_alive():
                        self._broken = True
                        raise RuntimeError("Computation process exited; retry the request")
                    now = time.monotonic()
                    if deadline is not None and now >= deadline:
                        self._owner._force_restart(generation)
                        raise Cancelled()
                    if deadline is None and now >= silence_deadline:
                        if not cancellable:
                            # Stop the stuck load; the failed preparation offers
                            # "모델 준비 다시 시도", whose submission replaces the
                            # child in the same family with a new generation.
                            self._broken = True
                            process.kill()
                            process.join(3)
                            raise RuntimeError(
                                f"모델 적재가 적재 한도({limit:g}초)를 넘겨 계산 프로세스를 "
                                "멈췄습니다. 모델 준비를 다시 시도하세요")
                        self._owner._force_restart(generation, reason="no_progress")
                        raise Cancelled()
                    continue
                message = _receive(connection)
                if any(message.get(k) != v for k, v in identity.items()):
                    continue
                if message["type"] == "progress":
                    if cancellable:
                        silence_deadline = time.monotonic() + limit
                    if cancelled:
                        # The child still answers; give it until its next report.
                        deadline = time.monotonic() + self._owner.cancel_grace
                    elif not switching:
                        try:
                            on_batch(message["done"], message["total"])
                        except Cancelled:
                            cancelled = True
                            cancellation.set()
                    continue
                if cancelled or message["type"] == "cancelled":
                    raise Cancelled()
                if message["type"] == "error":
                    raise RuntimeError(message["value"])
                return message["value"]
        except (EOFError, BrokenPipeError, OSError, ValueError) as exc:
            if generation != self._owner.generation:
                raise Cancelled() from exc
            self._broken = True
            raise RuntimeError("Computation process exited; retry the request") from exc

    def warm_up(self, on_batch, *, switching=False):
        return self._call("warm_up", {}, on_batch, switching=switching)

    def sites_fragment(self, dna, on_batch):
        return self._call("sites_fragment", {"dna": dna}, on_batch)

    def prediction_fragments(self, dna, on_batch):
        return self._call("prediction_fragments", {"dna": dna}, on_batch)

    def explanation_fragment(self, dna, site_key, head, method, params, size, on_batch):
        return self._call("explanation_fragment", {
            "dna": dna, "site_key": site_key, "head": head,
            "method": method, "params": params, "size": size}, on_batch)

    def csv_table(self, dna, on_batch, filters=None):
        return self._call("csv_table", {"dna": dna, "filters": filters}, on_batch)


class ProcessDispatcher(Dispatcher):
    """Existing latest-request scheduler with a replaceable child lifetime."""

    def __init__(self, emit, *, preview=False, factory_name=None, family="non_nt",
                 cancel_grace=None, no_progress_limit=None, load_limit=None):
        if family not in FAMILY_FACTORIES:
            raise ValueError(f"Unknown model family: {family}")
        self.family = family
        self._factory_override = factory_name or ("desktop.fake:FakeRenderer" if preview else None)
        self.generation = 1
        self.cancel_grace = CANCEL_GRACE if cancel_grace is None else float(cancel_grace)
        self.no_progress_limit = (NO_PROGRESS_LIMIT if no_progress_limit is None
                                  else float(no_progress_limit))
        self.load_limit = LOAD_LIMIT if load_limit is None else float(load_limit)
        self._context = threading.local()
        self._submissions = threading.Lock()
        self._sink = emit
        self.renderer = ProcessRenderer(self, self._factory_override or FAMILY_FACTORIES[family])
        super().__init__(self._emit_current)

    @property
    def pid(self):
        return self.renderer.pid

    def _emit_current(self, event):
        # Dispatcher calls this while holding its lock; restart uses that lock
        # too, so no old ready/progress/result/error can follow invalidation.
        if self._context.generation == self.generation and not self._closed:
            self._sink({**event, "generation": self.generation, "family": self.family})

    def _tag(self, job):
        generation = self.generation

        def tagged(on_batch):
            self._context.generation = generation
            if generation != self.generation:
                raise Cancelled()
            return job(on_batch)

        return tagged

    def _submit_process(self, job, *, lifecycle, expected_generation=None,
                        expected_revision=None, receipt=False):
        with self._submissions:
            if self._closed:
                raise RuntimeError("Computation dispatcher is closed")
            if expected_generation is not None and expected_generation != self.generation:
                return self.restarted_refusal()
            if expected_revision is not None and expected_revision != self._latest:
                return {"request_id": None}
            if self.renderer.exited:
                self._restart()
                if expected_generation is not None:
                    return self.restarted_refusal()
            submit = super().submit_lifecycle if lifecycle else super().submit
            rid = submit(self._tag(job))
            if receipt:
                return {"request_id": rid, "generation": self.generation, "family": self.family}
            return rid

    def restarted_refusal(self):
        """Refuse work bound to a replaced generation and name the current one.

        Callers must not retry it in the new generation: invalidation has no
        exceptions.
        """
        return {"request_id": None, "reason": "restarted",
                "generation": self.generation, "family": self.family}

    def submit(self, job):
        return self._submit_process(job, lifecycle=False)

    def submit_request(self, job, *, lifecycle=False, generation=None, revision=None):
        """Return ID and generation atomically with scheduling, even on bridge threads.

        A request bound to a replaced generation is refused with
        `{"request_id": None, "reason": "restarted", "generation", "family"}`;
        one bound to a superseded revision gets a bare `{"request_id": None}`.
        """
        return self._submit_process(job, lifecycle=lifecycle,
                                    expected_generation=generation,
                                    expected_revision=revision, receipt=True)

    def submission_identity(self):
        """Bind a save dialog to the current generation and cancellation state.

        A child that died while idle is replaced here, before any dialog opens,
        so a save bound to its generation is refused without asking for a path.
        """
        with self._submissions:
            if not self._closed and self.renderer.exited:
                self._restart()
            return self.generation, self._latest

    def cancel(self):
        with self._submissions:
            super().cancel()

    def submit_lifecycle(self, job):
        return self._submit_process(job, lifecycle=True)

    def _execute(self, rid, job):
        self._context.request_id, self._context.lifecycle = rid, False
        return super()._execute(rid, job)

    def _execute_lifecycle(self, rid, job):
        self._context.request_id, self._context.lifecycle = rid, True
        return super()._execute_lifecycle(rid, job)

    def _force_restart(self, generation, reason="cancel_grace"):
        """Replace an unresponsive child in the same family (#218, #219).

        `reason` is "cancel_grace" (it did not answer a cancel) or "no_progress"
        (it stopped reporting with nobody cancelling: a stalled computation).

        Runs on the worker thread. Everything of the old generation is
        invalidated, including a request queued behind the unresponsive one;
        nothing is re-run here. The screen learns of the restart from a
        `restarting` event whose `request_id` is the new preparation, so it
        can keep analysis locked until that preparation's `ready`.
        """
        while not self._submissions.acquire(timeout=0.05):
            # A switch, restart or close already replaced this generation.
            if generation != self.generation:
                return
        try:
            if self._closed or generation != self.generation:
                return
            self._restart()
            rid = super().submit_lifecycle(self._tag(self.renderer.warm_up))
            with self._lock:
                # The worker's own context still names the old generation, so
                # this bypasses `_emit_current`; the ready that follows is current.
                self._sink({"type": "restarting", "reason": reason, "request_id": rid,
                            "generation": self.generation, "family": self.family})
        finally:
            self._submissions.release()

    def _restart(self):
        with self._lock:
            self.generation += 1
            if self._pending:
                self._record(self._pending[0], "superseded")
            for rid, _ in self._lifecycle:
                self._record(rid, "superseded")
            self._pending = None
            self._lifecycle.clear()
            self._latest = next(self._ids)
            # Do not allow a new child to start before exit is confirmed.
            self.renderer.stop()
            self._cv.notify_all()

    def restart(self):
        with self._submissions:
            if self._closed:
                raise RuntimeError("Computation dispatcher is closed")
            self._restart()
            return self.generation

    def select_family(self, family):
        """Invalidate, confirm exit, choose the factory, then queue preparation.

        Re-selecting the current family only asks the existing renderer for its
        ready signal; its predictors and explanation caches remain alive.
        """
        if family not in FAMILY_FACTORIES:
            raise ValueError(f"Unknown model family: {family}")
        with self._submissions:
            if self._closed:
                raise RuntimeError("Computation dispatcher is closed")
            changed = family != self.family
            if changed or self.renderer.exited:
                self._restart()
            self.family = family
            self.renderer._factory_name = self._factory_override or FAMILY_FACTORIES[family]
            rid = super().submit_lifecycle(self._tag(
                lambda on_batch: self.renderer.warm_up(on_batch, switching=changed)
            ))
            return {"request_id": rid, "generation": self.generation,
                    "family": self.family, "changed": changed}

    def commit_if_current(self, action):
        """Publish a CSV under the same invalidation lock as result events."""
        with self._lock:
            if (self._context.generation != self.generation
                    or self._context.request_id != self._latest or self._closed):
                raise Cancelled()
            action()

    def close(self):
        with self._submissions:
            self._restart()
            super().close()
