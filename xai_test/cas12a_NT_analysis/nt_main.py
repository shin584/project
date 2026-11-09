"""Full Test analysis: one native model, durable per-source method outputs."""

import gc
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from functools import partial

from cas12a.nt_data import canonical_bytes, digest

from .nt_artifacts import prediction_rows
from .nt_inputs import check_export_files, encode_sequences, row_identity
from .nt_main_artifacts import (
    read_main,
    read_plan,
    save_cases,
    save_shard,
    shard_identity,
    source_roles,
)
from .nt_run import DirectoryStore
from .nt_selection import read_coverage
from .nt_validation_protocol import PRESET, code_snapshot, snapshot_digest


def prepare_main(inputs, directory, *, coverage_directory=None, coverage_checksum=None):
    """Fix IG statistics membership before prediction; never infer affordability."""
    identity = inputs.identity
    ids = [r["source_row"] for r in inputs.rows]
    if (
        identity["partition"] != "test"
        or ids != identity.get("test_source_rows")
        or [row_identity(r) for r in inputs.rows] != identity.get("test_rows")
        or not ids
        or ids != sorted(set(ids))
        or (not inputs.fixture and len(ids) != 1500)
    ):
        raise ValueError("main analysis requires verified original Test membership")
    coverage = None
    if coverage_directory is not None:
        coverage = read_coverage(coverage_directory, checksum=coverage_checksum)
        partition = coverage["partition_manifest"]
        if (
            coverage["population_ids"] != ids
            or coverage["fixture"] != inputs.fixture
            or partition["partition_sha256"] != identity["partition_sha256"]
            or partition["source_file_sha256"] != identity["source_sha256"]
            or partition["table_sha256"] != identity["table_sha256"]
            or coverage["analysis_identity"]["model_sha256"]
            != identity["export_files"]["model.safetensors"]
        ):
            raise ValueError(
                "frozen IG coverage differs from Test input/model identity"
            )
    elif coverage_checksum is not None:
        raise ValueError("coverage checksum requires a coverage directory")
    code = code_snapshot()
    plan = {
        "schema": 1,
        "kind": "cas12a-test-main-analysis",
        "input_identity": identity,
        "code_sha256": snapshot_digest(code),
        "code_files": {
            name: hashlib.sha256(data).hexdigest() for name, data in code.items()
        },
        "ig_mode": coverage["mode"] if coverage else "full",
        "statistics_ids": coverage["storage_ids"] if coverage else ids,
        "coverage_evidence": coverage,
        "ig_policy": PRESET["ig_policy"],
        "rollout_policy": PRESET["rollout_policy"],
        "batch_size": PRESET["batch_size"],
        "integration_batch_size": PRESET["integration_batch_size"],
        "historical_budget_status": "unresolved; this bundle neither settles nor resets historical spending",
        "production_execution_policy": "pending explicit decision; no affordability or remaining-budget claim",
        "fixture": inputs.fixture,
    }
    store = DirectoryStore(directory)
    if store.path("plan.json").exists():
        if read_plan(directory) != plan:
            raise ValueError("incompatible input/code/IG plan on main analysis resume")
    else:
        if store.root.exists() and any(store.root.iterdir()):
            raise ValueError("new main analysis requires an empty directory")
        checksum = store.put("plan.json", canonical_bytes(plan))
        store.put("plan.sha256", checksum.encode())
    return plan


def run_main(
    inputs,
    directory,
    *,
    device="cuda:0",
    stages=("ism", "rollout", "ig"),
    on_progress=None,
    study=None,
    reserved_seconds=None,
):
    """Run/resume Test work using the existing cumulative accounting for production.

    No new allowance is manufactured here. An unresolved/blocked historical study
    stays blocked; preparation and model-free reports require no GPU allowance.
    """
    from .nt_ig import compute_ig
    from .nt_ism import compute_ism
    from .nt_predict import NativePredictor, installed_environment
    from .nt_rollout import compute_rollout

    store, plan = DirectoryStore(directory), read_plan(directory)
    binding = shard_identity(plan)
    if (
        inputs.identity != plan["input_identity"]
        or snapshot_digest(code_snapshot()) != plan["code_sha256"]
    ):
        raise ValueError("incompatible input/code identity on main analysis resume")
    if [row_identity(r) for r in inputs.rows] != plan["input_identity"]["test_rows"]:
        raise ValueError("Test source rows changed since preparation")
    if not set(stages) <= {"ism", "rollout", "ig"} or len(set(stages)) != len(stages):
        raise ValueError("stages must be unique ISM, rollout or IG methods")
    check_export_files(inputs.export_dir, inputs.identity["export_files"])
    saved = read_main(directory)
    if saved["case_selection_complete"] and all(
        not saved["coverage"][method]["not_computed"]
        for method in ("prediction", *stages)
    ):
        return {
            "complete": saved["complete"],
            "coverage": saved["coverage"],
            "output_dir": str(store.root),
        }
    if inputs.fixture:
        if device != "cpu":
            raise ValueError("synthetic fixtures run on CPU")
    else:
        from .analyze import require_drive

        if study is None or reserved_seconds is None:
            raise ValueError(
                "production requires the existing study and an explicit reservation; execution policy is unresolved"
            )
        require_drive(store.root)
        require_drive(study.store.root)
        if not device.startswith("cuda"):
            raise ValueError("production main analysis requires CUDA")
    if study is not None and study.status()["fixture"] != inputs.fixture:
        raise ValueError("fixture/production accounting mismatch")
    started, predictor, p, error = time.monotonic(), None, None, None
    session = (
        "sessions/"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex
    )
    if study is not None:
        run_id = "main-" + digest(plan)[:20]
        reference = {"study_root": str(study.store.root), "run_id": run_id}
        if store.path("accounting.json").exists():
            if json.loads(store.read("accounting.json")) != reference:
                raise ValueError(
                    "main analysis cannot switch cumulative accounting roots"
                )
        else:
            store.put("accounting.json", canonical_bytes(reference))
    guard = study.guard if study is not None else lambda: None
    environment = installed_environment(device)
    execution = {
        "python": environment["python"],
        "packages": {
            name: environment["packages"].get(name)
            for name in ("numpy", "scipy", "torch", "transformers", "safetensors")
        },
        "device_type": device.split(":")[0],
        "precision": "float32",
        "backend": "eager",
        "autocast": False,
        "tf32": False,
        "batch_size": plan["batch_size"],
        "integration_batch_size": plan["integration_batch_size"],
    }
    if store.path("execution.json").exists():
        if json.loads(store.read("execution.json")) != execution:
            raise ValueError(
                "incompatible execution environment on main analysis resume"
            )
    else:
        store.put("execution.json", canonical_bytes(execution))
    binding.update(session=session, execution_sha256=digest(execution))
    if study is not None:
        study.register(run_id, {"kind": "test-main-analysis", "plan": plan})
        study.start(run_id, uuid.uuid4().hex, reserved_seconds, environment)
    store.put(
        session + "/start.json",
        canonical_bytes(
            {
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "plan_sha256": digest(plan),
                "device": device,
                "fixture": inputs.fixture,
                "accounting_receipt": study.receipt if study is not None else None,
                "reserved_seconds": reserved_seconds,
            }
        ),
    )

    def model():
        nonlocal predictor
        if predictor is None:
            predictor = NativePredictor.load(
                inputs.export_dir,
                device=device,
                budget_guard=guard,
                expected_files=inputs.identity["export_files"],
            )
            store.put(
                session + "/environment.json",
                canonical_bytes(
                    {
                        "environment": installed_environment(device, query_gpu=True),
                        "model_state_sha256": predictor.initial_state,
                        "execution_sha256": digest(execution),
                    }
                ),
            )
        return predictor

    def progress(method, done, total):
        if on_progress is not None:
            on_progress(
                {
                    "method": method,
                    "completed": done,
                    "requested": total,
                    "output_dir": str(store.root),
                }
            )

    try:
        predictions = {r["source_row"]: r for r in saved["predictions"]}
        missing = [r for r in inputs.rows if r["source_row"] not in predictions]
        for start in range(0, len(missing), plan["batch_size"]):
            batch, p = missing[start : start + plan["batch_size"]], model()
            sequences = [r["sequence"] for r in batch]
            raw = p.predict(sequences, batch_size=plan["batch_size"])
            tokens = encode_sequences(p.tokenizer, sequences, p.embedding_rows)
            p.assert_versions_unchanged()
            for row in prediction_rows(batch, raw, tokens):
                guard()
                i = row["source_row"]
                save_shard(
                    store, binding, "prediction", i, row, roles=["test_population"]
                )
                predictions[i] = row
                progress("prediction", len(predictions), len(inputs.rows))
        rows = [predictions[i] for i in sorted(predictions)]
        selection = save_cases(store, plan, rows)
        cases = selection["cases"]
        work = set(plan["statistics_ids"]) | {c["source_row"] for c in cases}
        for method in stages:
            requested = sorted(work) if method == "ig" else sorted(predictions)
            existing = saved["shards"][method]
            done = len(existing)
            progress(method, done, len(requested))
            for i in requested:
                if i in existing:
                    continue
                row, p = predictions[i], model()
                p.assert_versions_unchanged()
                guard()
                try:
                    if method == "ism":
                        result = compute_ism(
                            [row],
                            partial(p.predict, batch_size=plan["batch_size"]),
                            p.tokenizer,
                            p.embedding_rows,
                            batch_size=plan["batch_size"],
                        )
                    elif method == "rollout":
                        result = compute_rollout(row, p, policy=plan["rollout_policy"])
                    else:
                        result = compute_ig(
                            row,
                            p,
                            policy=plan["ig_policy"],
                            integration_batch_size=plan["integration_batch_size"],
                            disable_parameter_gradients=True,
                        )
                except ValueError as numerical_error:
                    failed = {
                        "row": {
                            **{
                                k: row[k]
                                for k in (
                                    "source_row",
                                    "sequence",
                                    "sequence_sha256",
                                    "raw_prediction",
                                )
                            },
                            "prediction_sha256": digest(row),
                        },
                        "accepted": False,
                        "execution_failure": True,
                        "failure": {
                            "stage": method,
                            "reason": str(numerical_error),
                            "type": type(numerical_error).__name__,
                        },
                    }
                    try:
                        save_shard(
                            store,
                            binding,
                            method,
                            i,
                            failed,
                            roles=source_roles(plan, cases, i),
                        )
                    except (OSError, ValueError) as storage_error:
                        numerical_error.add_note(
                            f"Could not persist method failure: {storage_error}"
                        )
                    raise
                p.assert_versions_unchanged()
                save_shard(
                    store,
                    binding,
                    method,
                    i,
                    result,
                    roles=source_roles(plan, cases, i),
                )
                done += 1
                progress(method, done, len(requested))
            if predictor is not None:
                predictor.assert_unchanged()
        if predictor is not None:
            if not stages:
                predictor.assert_unchanged()
            predictor.assert_versions_unchanged()
            predictor.synchronize()
        guard()
    except BaseException as caught:
        error = {"type": type(caught).__name__, "message": str(caught)}
        if study is not None and not study.poisoned:
            try:
                study.record_failure(caught)
            except (OSError, ValueError) as accounting_error:
                caught.add_note(
                    f"Could not persist accounting failure: {accounting_error}"
                )
        raise
    finally:
        predictor, p = None, None
        gc.collect()
        try:
            store.put(
                session + "/end.json",
                canonical_bytes(
                    {
                        "elapsed_seconds": time.monotonic() - started,
                        "ended_utc": datetime.now(timezone.utc).isoformat(),
                        "error": error,
                        "historical_budget_settled": False,
                    }
                ),
            )
        except (OSError, ValueError):
            if error is None:
                raise
    if study is not None:
        study.finish()
    result = read_main(directory)
    return {
        "complete": result["complete"],
        "coverage": result["coverage"],
        "output_dir": str(store.root),
    }
