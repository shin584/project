"""Colab validation phases, checked replay, coverage freeze and evidence delivery."""

import hashlib
import json
import math
import time
import uuid
import zipfile
from dataclasses import replace
from pathlib import Path

from cas12a.nt_data import canonical_bytes, digest

from .nt_artifacts import prediction_rows, read_results
from .nt_ig import compute_ig
from .nt_ig_artifacts import plan_ig, read_ig, save_ig, save_ig_plan
from .nt_inputs import check_export_files, encode_sequences
from .nt_ism import compute_ism, read_ism, save_ism
from .nt_predict import NativePredictor, installed_environment
from .nt_rollout import compute_rollout
from .nt_rollout_artifacts import (
    plan_rollout,
    read_rollout,
    save_rollout,
    save_rollout_plan,
)
from .nt_run import DirectoryStore, Study, file_hash
from .nt_selection import CASE_PROTOCOL, freeze_coverage, read_coverage
from .nt_validation_protocol import (
    code_snapshot,
    forecast,
    prepare_protocol,
    snapshot_digest,
)


def _outside(path, study_root):
    path, study_root = Path(path).resolve(), Path(study_root).resolve()
    if (
        path == study_root
        or path.is_relative_to(study_root)
        or study_root.is_relative_to(path)
    ):
        raise ValueError(
            "evidence and study directories must be separate, non-nested roots"
        )
    return path


def _immutable_json(store, name, value, *, require_existing=False):
    data = canonical_bytes(value)
    if store.path(name).exists():
        if store.read(name) != data:
            raise ValueError(f"immutable evidence changed: {name}")
        return hashlib.sha256(data).hexdigest()
    if require_existing:
        raise ValueError(f"missing frozen evidence: {name}")
    return store.put(name, data)


class ReceiptJournal:
    """Independently retained immutable acknowledgements; never infer from ledger."""

    def __init__(self, directory, study_root):
        self.store = DirectoryStore(_outside(directory, study_root))
        self.study_root = Path(study_root).resolve()

    def __call__(self, receipt):
        _immutable_json(
            self.store, f"{receipt['index']:08d}-{receipt['head']}.json", receipt
        )

    def latest(self):
        if list(self.store.root.glob("*.pending-*")):
            raise ValueError("ambiguous receipt write; manual investigation required")
        paths = sorted(self.store.root.glob("*.json"))
        if not paths:
            raise ValueError(
                "no independently retained receipt; refusing recovery from ledger"
            )
        value = json.loads(paths[-1].read_bytes())
        if paths[-1].name != f"{value['index']:08d}-{value['head']}.json":
            raise ValueError("receipt filename/content mismatch")
        return value

    def completion(self, job_id):
        for path in sorted(self.store.root.glob("*.json"), reverse=True):
            value = json.loads(path.read_bytes())
            ack = value.get("close_ack")
            if ack and ack.get("job_id") == job_id:
                envelope = json.loads(
                    (
                        self.study_root / "ledger" / f"{value['index']:08d}.json"
                    ).read_bytes()
                )
                if envelope["payload"]["event"] not in (
                    "finish",
                    "settle-with-external-termination-evidence",
                ):
                    continue
                closed = envelope["payload"]["state"]["last_closed"]
                if envelope["sha256"] != value["head"] or closed["job_id"] != job_id:
                    raise ValueError("phase completion receipt/ledger mismatch")
                elapsed = ack["elapsed_seconds"]
                if (
                    not math.isfinite(elapsed)
                    or not 0 < elapsed <= closed["charged_seconds"]
                ):
                    raise ValueError("invalid acknowledged phase elapsed time")
                return {"receipt": value, "closed": closed}
        raise ValueError(
            "missing independently retained phase completion acknowledgement"
        )


def open_study(study_root, evidence_root, *, initialize=False, fixture=False):
    """Explicit first-ever initialization or strict resume using the external tip."""
    if not fixture:
        from .analyze import require_drive

        require_drive(study_root)
    evidence_root = _outside(evidence_root, study_root)
    journal = ReceiptJournal(evidence_root / "receipts", study_root)
    if initialize:
        if evidence_root.exists() and any(evidence_root.iterdir()):
            raise ValueError("first-study evidence directory must be new and empty")
        study = Study.create(study_root, fixture=fixture, receipt_sink=journal)
    else:
        study = Study.reopen(study_root, receipt=journal.latest(), receipt_sink=journal)
    if study.status()["fixture"] != fixture:
        raise ValueError("fixture/production study mismatch")
    return study


def freeze_protocol(
    inputs,
    study,
    partition_path,
    protocol,
    evidence_root,
    *,
    approved_sha256,
    single_executor=False,
    no_new_fp32_test=False,
):
    """Require human acceptance of exact pre-observation protocol bytes."""
    from .analyze import require_drive

    if single_executor is not True or no_new_fp32_test is not True:
        raise ValueError(
            "explicit single-executor and no-new-FP32-Test acknowledgements required"
        )
    if approved_sha256 != digest(protocol):
        raise ValueError("accept the exact displayed protocol hash before validation")
    expected = prepare_protocol(partition_path, fixture=inputs.fixture)
    if (
        protocol != expected
        or protocol["partition_sha256"] != inputs.identity["partition_sha256"]
    ):
        raise ValueError("protocol code/partition/ID binding changed")
    if inputs.fixture != study.status()["fixture"]:
        raise ValueError("fixture/production protocol mismatch")
    if not inputs.fixture:
        require_drive(study.store.root)
    evidence = DirectoryStore(_outside(evidence_root, study.store.root))
    if study.receipt_sink is None:
        raise ValueError("validation requires automatic external receipt retention")
    environment = installed_environment("cpu" if inputs.fixture else "cuda:0")
    frozen = {
        "schema": 1,
        "protocol": protocol,
        "protocol_sha256": digest(protocol),
        "input_identity": inputs.identity,
        "packages": environment["packages"],
        "python": environment["python"],
        "single_executor_acknowledged": True,
        "no_new_fp32_test_acknowledged": True,
    }
    run_id = "v146-" + digest(protocol)[:16]
    snapshots = code_snapshot()
    snapshots["validation-protocol.json"] = canonical_bytes(protocol)
    snapshots["validation-partition.json"] = Path(partition_path).read_bytes()
    snapshots["validation-freeze.json"] = canonical_bytes(frozen)
    # Keep the 1,500-row input binding once in the immutable snapshot rather
    # than duplicating it into every subsequent journal state.
    study.register(
        run_id,
        {
            "schema": 1,
            "kind": "validation-freeze",
            "protocol_sha256": digest(protocol),
            "frozen_sha256": digest(frozen),
        },
        snapshots=snapshots,
    )
    _immutable_json(evidence, f"protocols/{digest(protocol)}.json", frozen)
    return frozen


def _check_frozen(study, frozen, evidence_root):
    checksum = digest(frozen["protocol"])
    if checksum != frozen["protocol_sha256"]:
        raise ValueError("protocol hash mismatch")
    stored = json.loads(
        DirectoryStore(_outside(evidence_root, study.store.root)).read(
            f"protocols/{checksum}.json"
        )
    )
    if stored != frozen or study.status()["runs"]["v146-" + checksum[:16]][
        "identity"
    ] != {
        "schema": 1,
        "kind": "validation-freeze",
        "protocol_sha256": checksum,
        "frozen_sha256": digest(frozen),
    }:
        raise ValueError("protocol freeze differs from acknowledged study")
    # Triggers the ordinary ledger, committed-byte and stale-manager checks.
    study.artifacts("v146-" + checksum[:16], kind="validation")
    if study.status()["open_job"]:
        raise ValueError(
            "unresolved job; confirm termination and settle before continuing"
        )


def _phase_runs(study, frozen, phase):
    return [
        run_id
        for run_id, run in study.status()["runs"].items()
        if run["identity"].get("validation_phase", {}).get("protocol_sha256")
        == frozen["protocol_sha256"]
        and run["identity"]["validation_phase"]["phase"] == phase
    ]


def _check_phase_identity(study, frozen, run_id):
    state = study.status()
    identity = state["runs"][run_id]["identity"]
    phase = identity["validation_phase"]["phase"]
    ids = frozen["protocol"]["groups"][phase]
    for key, value in frozen["input_identity"].items():
        if key == "validation_source_rows":
            value = ids
        elif key == "validation_rows":
            value = [row for row in value if row["source_row"] in ids]
        if identity.get(key) != value:
            raise ValueError("phase input identity differs from protocol freeze")
    freeze_id = "v146-" + frozen["protocol_sha256"][:16]
    prefix = f"runs/{freeze_id}/snapshot/"
    metadata = {
        "validation-protocol.json",
        "validation-partition.json",
        "validation-freeze.json",
    }
    code = {
        name.removeprefix(prefix): record["sha256"]
        for name, record in state["runs"][freeze_id]["files"].items()
        if record["kind"] == "snapshot" and name.removeprefix(prefix) not in metadata
    }
    if digest(code) != frozen["protocol"]["code_sha256"]:
        raise ValueError("frozen code snapshot differs from scientific protocol")
    phase_prefix = f"runs/{run_id}/snapshot/"
    code.update(
        {
            name.removeprefix(phase_prefix): record["sha256"]
            for name, record in state["runs"][run_id]["files"].items()
            if record["kind"] == "snapshot"
        }
    )
    if digest(code) != identity["code_sha256"]:
        raise ValueError("phase code/config identity differs from frozen snapshot")


def read_phase(study, frozen, run_id, evidence_root):
    """Recompute numerical verdicts from committed arrays without model imports."""
    run = study.status()["runs"][run_id]
    context = run["identity"]["validation_phase"]
    if context["protocol_sha256"] != frozen["protocol_sha256"]:
        raise ValueError("phase protocol mismatch")
    records = study.artifacts(run_id, kind="validation")
    _check_phase_identity(study, frozen, run_id)
    if not records:
        failed = bool(
            read_ig(study, run_id)["failed"] or read_rollout(study, run_id)["failed"]
        )
        return {
            "complete": False,
            "accepted": False,
            "run_id": run_id,
            "numerical_failure": failed,
        }
    if len(records) != 1:
        raise ValueError("exactly one measurement summary per phase required")
    record, summary = records[0]
    expected_ids = frozen["protocol"]["groups"][context["phase"]]
    if (
        summary["identity"] != run["identity"]
        or record["coverage"] != expected_ids
        or summary["source_ids"] != expected_ids
    ):
        raise ValueError("phase measurement identity/coverage mismatch")
    if summary.get("numerical_failure"):
        journal = ReceiptJournal(Path(evidence_root) / "receipts", study.store.root)
        completion = journal.completion(summary["job_id"])
        if (
            completion["closed"]["run_id"] != run_id
            or completion["closed"]["outcome"] != "quality-failed"
        ):
            raise ValueError(
                "numerical diagnostic lacks its failed-phase acknowledgement"
            )
        return {
            "complete": True,
            "accepted": False,
            "numerical_failure": True,
            "run_id": run_id,
            "phase": context["phase"],
            "summary": summary,
            "completion": completion,
            "gpu_executed": not frozen["protocol"]["fixture"]
            and summary["environment"]["device"].startswith("cuda"),
        }
    reports = {
        "prediction": read_results(study, run_id),
        "ism": read_ism(study, run_id),
        "ig": read_ig(study, run_id),
        "rollout": read_rollout(study, run_id),
    }
    accepted = all(
        r["complete"] and r["coverage"] == expected_ids for r in reports.values()
    )
    if summary["accepted"] != accepted:
        raise ValueError("phase acceptance differs from saved-method quality replay")
    journal = ReceiptJournal(Path(evidence_root) / "receipts", study.store.root)
    completion = journal.completion(summary["job_id"])
    if completion["closed"]["run_id"] != run_id:
        raise ValueError("phase close receipt belongs to another run")
    if completion["closed"]["outcome"] == "interrupted":
        return {
            "complete": False,
            "accepted": False,
            "run_id": run_id,
            "numerical_failure": not summary["accepted"],
        }
    config = frozen["protocol"]["config"]
    if (
        reports["ig"]["plan"]["policy"] != config["ig_policy"]
        or reports["ig"]["plan"]["integration_batch_size"]
        != config["integration_batch_size"]
        or reports["rollout"]["plan"]["policy"] != config["rollout_policy"]
        or run["identity"]["protocol"]["batch_size"] != config["batch_size"]
    ):
        raise ValueError("method settings differ from frozen scientific protocol")
    for key, value in frozen["input_identity"].items():
        if (
            key not in ("validation_rows", "validation_source_rows")
            and run["identity"].get(key) != value
        ):
            raise ValueError("phase input identity differs from protocol freeze")
    if (
        run["identity"]["packages"] != frozen["packages"]
        or run["identity"]["python"] != frozen["python"]
    ):
        raise ValueError("phase environment differs from protocol freeze")
    if (
        len(run["sessions"]) != 1
        or run["sessions"][0]["parity"]["samples"] != len(expected_ids)
        or run["sessions"][0]["parity"].get("production_batch_size")
        != min(len(expected_ids), config["batch_size"])
    ):
        raise ValueError("phase lacks actual full production batch parity")
    for value in summary["timings"].values():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid measured phase timing")
    ig_times = summary["ig_seconds_by_source"]
    if accepted and (
        set(ig_times) != {str(i) for i in expected_ids}
        or any(not math.isfinite(v) or v < 0 for v in ig_times.values())
    ):
        raise ValueError("IG per-source measurement coverage mismatch")
    max_steps = frozen["protocol"]["config"]["ig_policy"]["max_steps"]
    ladder = sum(200 * 2**i for i in range(int(math.log2(max_steps / 200)) + 1))
    scaled = []
    for result in reports["ig"]["results"]:
        if result["accepted"]:
            steps = sum(attempt["steps"] for attempt in result["attempts"])
            scaled.append(ig_times[str(result["row"]["source_row"])] * ladder / steps)
    times = summary["timings"]
    count = len(expected_ids)
    method_time = sum(times[k] for k in ("prediction", "ism", "ig", "rollout"))
    elapsed = completion["receipt"]["close_ack"]["elapsed_seconds"]
    rates = {name: times[name] / count for name in ("prediction", "ism", "rollout")}
    rates.update(
        ig=sum(scaled) / count,
        common=max(0, elapsed - times["loading"] - times["parity"] - method_time)
        / count,
        loading=times["loading"] + times["parity"],
    )
    return {
        "complete": True,
        "accepted": accepted,
        "run_id": run_id,
        "phase": context["phase"],
        "summary": summary,
        "rates": rates,
        "gpu_executed": any(r["gpu_executed"] for r in reports.values()),
        "max_accepted_steps": max(
            (
                r["attempts"][-1]["steps"]
                for r in reports["ig"]["results"]
                if r["accepted"]
            ),
            default=None,
        ),
        "completion": completion,
    }


class _NumericalFailure(ValueError):
    def __init__(self, stage, reason):
        super().__init__(reason)
        self.stage, self.reason = stage, reason


class _Meter:
    def __init__(self, study):
        self.study = study
        self.timings = dict.fromkeys(
            (
                "loading",
                "parity",
                "prediction",
                "ism",
                "ig",
                "rollout",
                "verification",
                "storage",
                "tokenization",
                "synchronization",
            ),
            0.0,
        )
        self.sync = lambda: None

    def measure(self, name, operation, *args, **kwargs):
        self.sync()
        start = time.perf_counter()
        try:
            return operation(*args, **kwargs)
        except ValueError as error:
            if name in ("parity", "prediction", "ism", "ig", "rollout", "verification"):
                # A budget/storage failure must retain its own recovery path.
                # Numerical ValueErrors only arise at these calculation seams.
                self.study.guard()
                raise _NumericalFailure(name, str(error)) from error
            raise
        finally:
            self.sync()
            self.timings[name] += time.perf_counter() - start


class _TimedTokenizer:
    def __init__(self, tokenizer, meter):
        self.tokenizer, self.meter = tokenizer, meter

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            return self.tokenizer(*args, **kwargs)
        finally:
            self.meter.timings["tokenization"] += time.perf_counter() - start


def run_phase(inputs, study, frozen, phase, evidence_root, *, attempt=0, loader=None):
    """Run one fresh measured phase; reuse whole completed phases on resume."""
    from .analyze import _prepare_execution

    _check_frozen(study, frozen, evidence_root)
    protocol, config = frozen["protocol"], frozen["protocol"]["config"]
    if (
        phase not in protocol["groups"]
        or type(attempt) is not int
        or not 0 <= attempt <= 99
    ):
        raise ValueError("unknown phase or invalid explicit replacement attempt")
    if (
        inputs.identity != frozen["input_identity"]
        or snapshot_digest(code_snapshot()) != protocol["code_sha256"]
    ):
        raise ValueError("frozen source/code identity changed")
    # No later or replacement phase can bypass recorded numerical failures.
    for name in protocol["phase_order"]:
        prior_assessments = []
        for prior in _phase_runs(study, frozen, name):
            assessed = read_phase(study, frozen, prior, evidence_root)
            prior_assessments.append(assessed)
            if assessed.get("numerical_failure") or (
                assessed["complete"] and not assessed["accepted"]
            ):
                raise ValueError("saved numerical failure blocks this frozen protocol")
            if name == phase and assessed["complete"]:
                return assessed
        if name == phase:
            break
        if not any(r["accepted"] for r in prior_assessments):
            raise ValueError(f"previous phase {name} is incomplete")
    device = "cpu" if inputs.fixture else "cuda:0"
    ids = protocol["groups"][phase]
    selected = replace(
        inputs,
        rows=[r for r in inputs.rows if r["source_row"] in ids],
        identity={
            **inputs.identity,
            "validation_source_rows": ids,
            "validation_rows": [
                r for r in inputs.identity["validation_rows"] if r["source_row"] in ids
            ],
        },
    )
    environment, snapshots, identity = _prepare_execution(
        selected, study, device, config["batch_size"]
    )
    if (
        environment["packages"] != frozen["packages"]
        or environment["python"] != frozen["python"]
    ):
        raise ValueError("installed environment differs from frozen protocol")
    identity["validation_phase"] = {
        "protocol_sha256": frozen["protocol_sha256"],
        "phase": phase,
        "attempt": attempt,
    }
    run_id = f"v146-{frozen['protocol_sha256'][:12]}-{phase}-{attempt}"
    # Code is already snapshotted once in the frozen protocol run. Each phase
    # needs only its saved export metadata for the existing model-free readers.
    study.register(
        run_id,
        identity,
        snapshots={
            k: v for k, v in snapshots.items() if k.startswith("selected_export/")
        },
    )
    if study.status()["runs"][run_id]["sessions"]:
        raise ValueError(
            "partial measured phase requires settlement and an explicit replacement attempt"
        )
    reservation = config["reservations"]["cost" if phase.startswith("cost-") else phase]
    study.start(run_id, uuid.uuid4().hex, reservation, environment)
    meter, ig_times, numerical_failure = _Meter(study), {}, False
    try:
        import torch

        environment = installed_environment(device, query_gpu=True)
        prior_gpus = {
            r["sessions"][0]["environment"].get("gpu")
            for r in study.status()["runs"].values()
            if r["identity"].get("validation_phase", {}).get("protocol_sha256")
            == frozen["protocol_sha256"]
            and r["sessions"]
            and r["sessions"][0].get("parity")
        }
        if prior_gpus and prior_gpus != {environment["gpu"]}:
            raise ValueError(
                "GPU changed; fresh frozen probe protocol required in the same study"
            )
        study.record_execution(
            environment, {"validated": False, "stage": "before-model-load"}
        )
        if not inputs.fixture:
            torch.cuda.reset_peak_memory_stats()
        predictor = meter.measure(
            "loading",
            lambda: (loader or NativePredictor.load)(
                inputs.export_dir,
                device=device,
                budget_guard=study.guard,
                expected_files=inputs.identity["export_files"],
            ),
        )
        sync = predictor.synchronize

        def timed_sync():
            started = time.perf_counter()
            sync()
            meter.timings["synchronization"] += time.perf_counter() - started

        predictor.synchronize = timed_sync
        predictor.tokenizer = _TimedTokenizer(predictor.tokenizer, meter)
        meter.sync = timed_sync
        sequences = [r["sequence"] for r in selected.rows]
        parity = meter.measure(
            "parity",
            lambda: predictor.parity(sequences, batch_size=config["batch_size"]),
        )
        parity["production_batch_size"] = min(len(sequences), config["batch_size"])
        study.record_execution(environment, parity)
        peak_loading = (
            None
            if inputs.fixture
            else {
                "allocated": torch.cuda.max_memory_allocated(),
                "reserved": torch.cuda.max_memory_reserved(),
            }
        )
        if not inputs.fixture:
            torch.cuda.reset_peak_memory_stats()
        raw = meter.measure(
            "prediction",
            lambda: predictor.predict(sequences, batch_size=config["batch_size"]),
        )
        tokens = meter.measure(
            "verification",
            lambda: encode_sequences(
                predictor.tokenizer, sequences, predictor.embedding_rows
            ),
        )
        rows = prediction_rows(selected.rows, raw, tokens)
        meter.measure("verification", predictor.assert_unchanged)
        meter.measure(
            "storage",
            lambda: study.save_artifact(
                "predictions", {"schema": 1, "identity": identity, "rows": rows}, ids
            ),
        )
        predictions = read_results(study, run_id)
        ig_plan = plan_ig(
            predictions,
            run_id=run_id,
            policy=config["ig_policy"],
            integration_batch_size=config["integration_batch_size"],
            probe_ids=ids,
        )
        rollout_plan = plan_rollout(
            predictions, run_id=run_id, policy=config["rollout_policy"]
        )
        meter.measure("storage", lambda: save_ig_plan(study, run_id, ig_plan))
        meter.measure("storage", lambda: save_rollout_plan(study, run_id, rollout_plan))
        ism = meter.measure(
            "ism",
            lambda: compute_ism(
                rows,
                lambda batch: predictor.predict(batch, batch_size=config["batch_size"]),
                predictor.tokenizer,
                predictor.embedding_rows,
                batch_size=config["batch_size"],
                guard=study.guard,
            ),
        )
        meter.measure("verification", predictor.assert_unchanged)
        meter.measure("storage", lambda: save_ism(study, run_id, ism))
        for row in rows:
            previous_time = meter.timings["ig"]
            ig = meter.measure(
                "ig",
                compute_ig,
                row,
                predictor,
                policy=config["ig_policy"],
                integration_batch_size=config["integration_batch_size"],
            )
            ig_times[str(row["source_row"])] = meter.timings["ig"] - previous_time
            meter.measure("verification", predictor.assert_unchanged)
            meter.measure("storage", save_ig, study, run_id, ig)
            if not ig["accepted"]:
                numerical_failure = True
                break
            rollout = meter.measure(
                "rollout",
                compute_rollout,
                row,
                predictor,
                policy=config["rollout_policy"],
            )
            meter.measure("verification", predictor.assert_unchanged)
            meter.measure("storage", save_rollout, study, run_id, rollout)
            if not rollout["accepted"]:
                numerical_failure = True
                break
        reports = meter.measure(
            "storage",
            lambda: (
                read_results(study, run_id),
                read_ism(study, run_id),
                read_ig(study, run_id),
                read_rollout(study, run_id),
            ),
        )
        accepted = all(r["coverage_complete"] and not r.get("failed") for r in reports)
        if not accepted and not numerical_failure:
            raise ValueError("unexpected incomplete method coverage")
        meter.measure("verification", predictor.assert_unchanged)
        meter.measure(
            "verification",
            lambda: check_export_files(
                inputs.export_dir, inputs.identity["export_files"]
            ),
        )
        peak_production = (
            None
            if inputs.fixture
            else {
                "allocated": torch.cuda.max_memory_allocated(),
                "reserved": torch.cuda.max_memory_reserved(),
            }
        )
        summary = {
            "schema": 1,
            "identity": identity,
            "source_ids": ids,
            "accepted": accepted,
            "job_id": study.status()["open_job"]["job_id"],
            "timings": meter.timings.copy(),
            "ig_seconds_by_source": ig_times,
            "peak_loading_bytes": peak_loading,
            "peak_production_bytes": peak_production,
            "explicit_compilation": False,
            "nested_timings": ["tokenization", "synchronization"],
            "timing_note": "tokenizer/synchronization overlap method times. Final external receipt latency is budget-guarded but excluded from close_ack elapsed; the fixed forecast margin covers that unmeasured tail.",
        }
        study.save_artifact("validation-summary", summary, ids, kind="validation")
        predictor.synchronize()
        study.guard()
        meter.sync = lambda: None
        predictor = None
        sync = None
        study.finish(outcome="quality-failed" if numerical_failure else "completed")
    except _NumericalFailure as error:
        diagnosis = {
            "schema": 1,
            "identity": identity,
            "source_ids": ids,
            "accepted": False,
            "job_id": study.status()["open_job"]["job_id"],
            "environment": environment,
            "numerical_failure": {"stage": error.stage, "reason": error.reason},
        }
        study.save_artifact("validation-summary", diagnosis, ids, kind="validation")
        meter.sync = lambda: None
        predictor = None
        sync = None
        study.finish(outcome="quality-failed")
    except Exception as error:
        if not study.poisoned and study.status()["open_job"]:
            study.record_failure(error)
        raise
    return read_phase(study, frozen, run_id, evidence_root)


def freeze_ig_coverage(
    study, frozen, partition_path, evidence_root, *, verify_only=False
):
    """Replay real probes, choose full-first coverage and retain an immutable receipt."""
    _check_frozen(study, frozen, evidence_root)
    if file_hash(partition_path) != frozen["protocol"]["partition_file_sha256"]:
        raise ValueError("frozen partition bytes changed")
    evidence = DirectoryStore(evidence_root)
    reports = []
    for phase in frozen["protocol"]["phase_order"]:
        accepted = [
            r
            for run in _phase_runs(study, frozen, phase)
            if (r := read_phase(study, frozen, run, evidence_root))["complete"]
            and r["accepted"]
        ]
        if len(accepted) != 1:
            raise ValueError(
                f"exactly one complete accepted {phase} measurement required"
            )
        reports.append(accepted[0])
    if not frozen["protocol"]["fixture"] and not all(
        r["gpu_executed"] for r in reports
    ):
        raise ValueError("CPU fixture success cannot authorize production coverage")
    cost = [r for r in reports if r["phase"].startswith("cost-")]
    rates = {key: max(r["rates"][key] for r in cost) for key in cost[0]["rates"]}
    decision = forecast(
        rates,
        population=frozen["protocol"]["test_count"],
        charged_seconds=study.status()["charged_seconds"],
        policy=frozen["protocol"]["config"]["forecast"],
    )
    measurements = {
        "protocol_sha256": frozen["protocol_sha256"],
        "phases": reports,
        "forecast": decision,
        "ledger_receipt": study.receipt,
        "max_observed_accepted_steps": max(r["max_accepted_steps"] for r in reports),
    }
    measurement_sha = _immutable_json(
        evidence, "measurements.json", measurements, require_existing=verify_only
    )
    if decision["mode"] is None:
        _immutable_json(
            evidence,
            "infeasible.json",
            {
                "measurement_sha256": measurement_sha,
                "forecast": decision,
                "test_authorized": False,
            },
            require_existing=verify_only,
        )
        return {
            "feasible": False,
            "forecast": decision,
            "measurement_sha256": measurement_sha,
        }
    prediction_protocol = next(r["summary"]["identity"]["protocol"] for r in reports)
    identity = {
        "study_id": study.status()["study_id"],
        "run_id": "test-" + frozen["protocol_sha256"][:16],
        "model_sha256": frozen["input_identity"]["export_files"]["model.safetensors"],
        "protocol_sha256": digest(prediction_protocol),
    }
    selection_decision = {
        "schema": 1,
        "partition": "validation",
        "partition_sha256": frozen["protocol"]["partition_sha256"],
        "analysis_identity": identity,
        "test_prediction_protocol": prediction_protocol,
        "scientific_protocol_sha256": frozen["protocol_sha256"],
        "full_rejected": decision["full_rejected"],
        "reason": "Maximum repeated production rates, full retry ladder and minute-rounded charged reservations",
        "measurement_reference": "measurements.json",
        "measurement_sha256": measurement_sha,
    }
    checksum = _immutable_json(
        evidence,
        "coverage-decision.json",
        selection_decision,
        require_existing=verify_only,
    )
    coverage_root = evidence.root / "coverage"
    if coverage_root.exists():
        # Never recover a missing trusted checksum from an untrusted manifest.
        prior = json.loads(evidence.read("freeze.json"))
        coverage_sha = prior["coverage_sha256"]
    else:
        if verify_only:
            raise ValueError("missing frozen coverage")
        coverage_sha = freeze_coverage(
            partition_path,
            evidence.path("coverage-decision.json"),
            coverage_root,
            decision_checksum=checksum,
            mode=decision["mode"],
            size=decision["selected"]["size"],
            fixture=frozen["protocol"]["fixture"],
        )
    coverage = read_coverage(coverage_root, checksum=coverage_sha)
    receipt = {
        "schema": 1,
        "feasible": True,
        "gpu_executed": all(r["gpu_executed"] for r in reports),
        "test_authorized": not frozen["protocol"]["fixture"],
        "coverage_sha256": coverage_sha,
        "measurement_sha256": measurement_sha,
        "analysis_identity": identity,
        "test_prediction_protocol": prediction_protocol,
        "code_sha256": frozen["protocol"]["code_sha256"],
        "protocol_sha256": frozen["protocol_sha256"],
        "membership_sha256": coverage["membership_sha256"],
        "selection_ids": coverage["selection_ids"],
        "storage_ids": coverage["storage_ids"],
        "case_rules": CASE_PROTOCOL,
        "forecast": decision,
        "ledger_receipt": study.receipt,
        "historical_fp16_test_known": True,
    }
    freeze_sha = _immutable_json(
        evidence, "freeze.json", receipt, require_existing=verify_only
    )
    return {**receipt, "freeze_sha256": freeze_sha}


def export_evidence(study, frozen, evidence_root, output_path):
    """Export successes or failures with all ledger/shards and retained receipts."""
    output = Path(output_path).resolve()
    roots = {"study": study.store.root, "evidence": Path(evidence_root).resolve()}
    if any(output == root or output.is_relative_to(root) for root in roots.values()):
        raise ValueError("archive must be outside study and evidence roots")
    # Preserve diagnostic evidence even when storage/receipt failure makes
    # ordinary reopen impossible. Such archives can never establish completion.
    journal = ReceiptJournal(roots["evidence"] / "receipts", roots["study"])
    receipt, verification_error = None, None
    try:
        receipt = journal.latest()
        Study.reopen(roots["study"], receipt=receipt)
    except (OSError, ValueError, KeyError, TypeError) as error:
        verification_error = str(error)
    entries = {}
    for prefix, root in roots.items():
        for path in sorted(root.rglob("*")):
            if path.is_file():
                if path.is_symlink():
                    raise ValueError("symlink evidence is not supported")
                if ".pending-" in path.name:
                    verification_error = (
                        "ambiguous pending write; manual investigation required"
                    )
                entries[f"{prefix}/{path.relative_to(root).as_posix()}"] = path
    manifest = {
        "schema": 1,
        "protocol_sha256": frozen["protocol_sha256"],
        "receipt": receipt,
        "history_verified": verification_error is None,
        "verification_error": verification_error,
        "files": {name: file_hash(path) for name, path in entries.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        output.open("xb") as target,
        zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive,
    ):
        for name, path in entries.items():
            archive.write(path, name)
        archive.writestr("manifest.json", canonical_bytes(manifest))
    return {
        "path": str(output),
        "sha256": file_hash(output),
        "files": len(entries),
        "receipt": receipt,
        "history_verified": verification_error is None,
        "verification_error": verification_error,
    }


def audit_evidence(archive_path, *, checksum, directory):
    """Verify a returned archive and replay its evidence without model/GPU access."""
    archive_path, directory = Path(archive_path), Path(directory).resolve()
    if file_hash(archive_path) != checksum:
        raise ValueError("returned archive SHA256 mismatch")
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("audit extraction requires a new empty directory")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names) or "manifest.json" not in names:
            raise ValueError("duplicate or missing archive entries")
        manifest = json.loads(archive.read("manifest.json"))
        if set(names) != {"manifest.json", *manifest["files"]}:
            raise ValueError("archive file set differs from manifest")
        store = DirectoryStore(directory)
        for name in names:
            path = store.path(name)
            if "\\" in name or name.startswith("/") or ".." in name.split("/"):
                raise ValueError("unsafe archive entry")
            data = archive.read(name)
            if (
                name != "manifest.json"
                and hashlib.sha256(data).hexdigest() != manifest["files"][name]
            ):
                raise ValueError("archive member hash mismatch")
            _ = path
        for name in names:
            store.put(name, archive.read(name))
    result = {
        "history_verified": manifest["history_verified"],
        "complete": False,
        "verification_error": manifest["verification_error"],
        "gpu_executed": None,
    }
    if not manifest["history_verified"]:
        return result
    study_root, evidence_root = directory / "study", directory / "evidence"
    journal = ReceiptJournal(evidence_root / "receipts", study_root)
    if journal.latest() != manifest["receipt"]:
        raise ValueError("archive receipt differs from retained journal tip")
    study = Study.reopen(study_root, receipt=journal.latest())
    frozen = json.loads(
        (
            evidence_root / "protocols" / f"{manifest['protocol_sha256']}.json"
        ).read_bytes()
    )
    # An unresolved job is valid diagnostic evidence, never a completed study.
    if study.status()["open_job"]:
        return {**result, "unresolved_job": study.status()["open_job"]}
    _check_frozen(study, frozen, evidence_root)
    phases = [
        read_phase(study, frozen, run, evidence_root)
        for phase in frozen["protocol"]["phase_order"]
        for run in _phase_runs(study, frozen, phase)
    ]
    observed_gpu = [p["gpu_executed"] for p in phases if "gpu_executed" in p]
    result.update(
        phases=phases,
        charged_seconds=study.status()["charged_seconds"],
        gpu_executed=False
        if frozen["protocol"]["fixture"]
        else (any(observed_gpu) if observed_gpu else None),
    )
    if (evidence_root / "freeze.json").exists() or (
        evidence_root / "infeasible.json"
    ).exists():
        partition = (
            study_root
            / "runs"
            / ("v146-" + frozen["protocol_sha256"][:16])
            / "snapshot"
            / "validation-partition.json"
        )
        freeze = freeze_ig_coverage(
            study, frozen, partition, evidence_root, verify_only=True
        )
        result.update(
            complete=freeze.get("test_authorized", False),
            gpu_executed=freeze.get("gpu_executed", result["gpu_executed"]),
            freeze=freeze,
        )
    return result
