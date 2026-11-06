"""Immutable IG plans, complete-source shards and model-free quality readback."""

import hashlib
import io
import json
import math
import uuid

import numpy as np

from cas12a.nt_data import canonical_bytes, digest

from .nt_artifacts import read_checked, read_results, saved_vocabulary
from .nt_ig import PROTOCOL, quality, summarize, validate_policy
from .nt_inputs import DNA_SPANS, LAYOUT
from .nt_predict import ATOL, RTOL
from .nt_run import STUDY_ID
from .nt_selection import ig_work


def _ids(values, available):
    values = list(values)
    if (
        not values
        or any(type(i) is not int for i in values)
        or len(set(values)) != len(values)
        or not set(values) <= set(available)
    ):
        raise ValueError("IG IDs must be unique and available in predictions")
    return sorted(values)


def plan_ig(
    predictions,
    *,
    run_id,
    policy,
    integration_batch_size=16,
    probe_ids=None,
    coverage=None,
    cases=None,
):
    """Bind explicit validation probes or a checked #140 union to saved predictions."""
    policy = validate_policy(policy)
    if type(integration_batch_size) is not int or integration_batch_size < 1:
        raise ValueError("integration batch size must be a positive integer")
    if not predictions["coverage_complete"]:
        raise ValueError("IG requires complete prediction coverage")
    identity, rows = predictions["identity"], predictions["rows"]
    available = [row["source_row"] for row in rows]
    if _ids(available, available) != available or available != predictions["coverage"]:
        raise ValueError("prediction ID order/coverage mismatch")
    if coverage is None:
        if (
            cases is not None
            or probe_ids is None
            or identity["partition"] != "validation"
        ):
            raise ValueError(
                "explicit validation probe IDs or frozen coverage/cases required"
            )
        ids = _ids(probe_ids, available)
        work = {
            "ids": ids,
            "statistics_ids": [],
            "additional_case_ids": [],
            "rows": [{"source_row": i, "roles": ["validation_probe"]} for i in ids],
        }
        kind = "validation_probe"
    else:
        if probe_ids is not None or cases is None:
            raise ValueError("frozen coverage requires cases and forbids probe IDs")
        work = ig_work(coverage, cases)
        expected = {
            "study_id": STUDY_ID,
            "run_id": run_id,
            "model_sha256": identity["export_files"]["model.safetensors"],
            "protocol_sha256": digest(identity["protocol"]),
        }
        partition = coverage["partition_manifest"]
        if (
            identity["partition"] != "test"
            or coverage["population_ids"] != available
            or coverage["analysis_identity"] != expected
            or partition["partition_sha256"] != identity["partition_sha256"]
            or partition["source_file_sha256"] != identity["source_sha256"]
            or partition["table_sha256"] != identity["table_sha256"]
            or coverage["fixture"] != identity["fixture"]
        ):
            raise ValueError(
                "frozen IG population/provenance differs from prediction identity"
            )
        normalized = [
            {
                key: row[key]
                for key in ("source_row", "sequence", "label", "raw_prediction")
            }
            for row in rows
        ]
        if canonical_bytes(cases["predictions"]) != canonical_bytes(normalized):
            raise ValueError("case predictions differ from stored predictions")
        _ids(work["ids"], available)
        kind = coverage["mode"]
    return {
        "schema": 1,
        "method": "ig-plan",
        "run_id": run_id,
        "identity": identity,
        "protocol": PROTOCOL,
        "policy": policy,
        "integration_batch_size": integration_batch_size,
        "kind": kind,
        "prediction_rows_sha256": digest(rows),
        "work": work,
        "coverage_manifest": coverage,
        "case_manifest": cases,
    }


def _rebuild_plan(plan, predictions, run_id):
    rebuilt = plan_ig(
        predictions,
        run_id=run_id,
        policy=plan["policy"],
        integration_batch_size=plan["integration_batch_size"],
        probe_ids=plan["work"]["ids"] if plan["kind"] == "validation_probe" else None,
        coverage=plan["coverage_manifest"],
        cases=plan["case_manifest"],
    )
    if canonical_bytes(plan) != canonical_bytes(rebuilt):
        raise ValueError("IG plan policy/identity/coverage mismatch")


def read_ig_plan(study, run_id, *, predictions=None):
    predictions = read_results(study, run_id) if predictions is None else predictions
    plans = study.artifacts(run_id, kind="ig-plan")
    if not plans:
        return None
    if len(plans) != 1:
        raise ValueError("exactly one immutable IG plan per run is required")
    record, plan = plans[0]
    _rebuild_plan(plan, predictions, run_id)
    if record["coverage"] != plan["work"]["ids"]:
        raise ValueError("IG plan committed coverage mismatch")
    return plan


def save_ig_plan(study, run_id, plan):
    study.checkpoint()
    if study.status()["open_job"]["run_id"] != run_id:
        raise ValueError("IG plan must be saved in its active run")
    predictions = read_results(study, run_id)
    if predictions["identity"]["protocol"].get("ig") != PROTOCOL:
        raise ValueError("run does not bind the IG protocol")
    _rebuild_plan(plan, predictions, run_id)
    existing = read_ig_plan(study, run_id, predictions=predictions)
    if existing is not None:
        if canonical_bytes(existing) != canonical_bytes(plan):
            raise ValueError("immutable IG plan cannot be changed")
        return
    if study.artifacts(run_id, kind="ig"):
        raise ValueError("IG shards without an immutable plan")
    study.save_artifact(
        "ig-plan-" + uuid.uuid4().hex, plan, plan["work"]["ids"], kind="ig-plan"
    )


def _model_width(study, run_id, identity):
    data = read_checked(
        study,
        f"runs/{run_id}/snapshot/selected_export/config.json",
        identity["export_files"]["config.json"],
    )
    width = json.loads(data)["hidden_size"]
    if type(width) is not int or width < 1:
        raise ValueError("invalid saved model embedding width")
    return width


def _check_result(result, plan, bindings, vocabulary, model_width):
    row = result["row"]
    binding = bindings.get(row["source_row"])
    if binding is None or row["source_row"] not in plan["work"]["ids"]:
        raise ValueError("unexpected IG source ID")
    expected_row = {
        **{
            key: binding[key]
            for key in ("source_row", "sequence", "sequence_sha256", "raw_prediction")
        },
        "prediction_sha256": digest(binding),
    }
    if row != expected_row or result["tokens"] != binding["tokens"]:
        raise ValueError("IG source/prediction/token binding mismatch")
    tokens = result["tokens"]
    ids = [
        3,
        *[vocabulary[row["sequence"][start:end]] for start, end in DNA_SPANS],
        1,
        1,
    ]
    if tokens["input_ids"] != ids or any(not 0 <= i < 4105 for i in ids):
        raise ValueError("IG actual token identity/span mismatch")
    if (
        result["schema"] != 1
        or result["protocol"] != PROTOCOL
        or result["policy"] != plan["policy"]
        or result["integration_batch_size"] != plan["integration_batch_size"]
        or result["vocabulary"]
        != {
            "vocabulary_sha256": digest(vocabulary),
            "vocabulary_size": 4107,
            "embedding_rows": 4105,
        }
    ):
        raise ValueError("IG protocol/policy/batch/vocabulary mismatch")
    execution = result["execution"]
    if execution != {
        "parameter_dtype": "float32",
        "output_dtype": "float32",
        "autocast": False,
        "tf32": False,
        "backend": "eager",
        "eval": True,
        "input_gradient_enabled": True,
        "device": execution["device"],
    }:
        raise ValueError("invalid IG FP32 execution record")
    reference = result["reference"]
    shape = reference["shape"]
    if (
        len(shape) != 2
        or shape != [12, model_width]
        or reference
        != {
            "kind": PROTOCOL["reference"],
            "shape": shape,
            "dtype": "float32",
            "input_ids": ids,
            "attention_mask": tokens["attention_mask"],
        }
    ):
        raise ValueError("IG reference/mask mismatch")
    arrays = result["arrays"]
    mass = arrays["signed_mass"]
    attempts, outputs = result["attempts"], result["outputs"]
    if (
        set(arrays) != {"signed_mass"}
        or mass.dtype != np.dtype("float64")
        or mass.shape != (len(attempts), 12)
        or not np.isfinite(mass).all()
    ):
        raise ValueError("invalid finite IG signed-mass array")
    if set(outputs) != {"native", "endpoint", "reference"}:
        raise ValueError("invalid IG output schema")
    for value in outputs.values():
        if value is not None and (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or float(np.float32(value)) != value
        ):
            raise ValueError("invalid saved FP32 IG output")
    endpoint_ok = (
        all(value is not None for value in outputs.values())
        and np.isclose(outputs["native"], row["raw_prediction"], atol=ATOL, rtol=RTOL)
        and np.isclose(outputs["endpoint"], outputs["native"], atol=ATOL, rtol=RTOL)
    )
    if len(attempts) and not endpoint_ok:
        raise ValueError("IG endpoint parity failure before integration")
    accepted = False
    for index, attempt in enumerate(attempts):
        steps = 200 * 2**index
        batch_size = result["integration_batch_size"]
        batches = [
            min(batch_size, steps - start) for start in range(0, steps, batch_size)
        ]
        assessed = quality(
            mass[index], mass[index - 1] if index else None, outputs, plan["policy"]
        )
        if (
            accepted
            or steps > plan["policy"]["max_steps"]
            or attempt != {"steps": steps, "batch_sizes": batches, "quality": assessed}
        ):
            raise ValueError("IG resolution/convergence/quality mismatch")
        accepted = (
            assessed["completeness_pass"] and assessed["convergence_pass"] is True
        )
    failure = result["failure"]
    if type(result["accepted"]) is not bool or result["accepted"] != bool(accepted):
        raise ValueError("IG acceptance differs from reconstructed quality")
    if accepted:
        if failure is not None:
            raise ValueError("accepted IG has a failure")
    else:
        if (
            not isinstance(failure, dict)
            or set(failure) != {"stage", "reason", "steps"}
            or not isinstance(failure["reason"], str)
            or not failure["reason"]
        ):
            raise ValueError("failed IG requires a stage and reason")
        stage = failure["stage"]
        if stage not in (
            "embedding",
            "native",
            "endpoint",
            "reference",
            "gradient",
            "integration",
            "quality",
        ):
            raise ValueError("unknown IG failure stage")
        if stage in ("embedding", "native", "endpoint", "reference"):
            if len(attempts) or failure["steps"] is not None:
                raise ValueError("invalid pre-integration failure")
        elif stage == "quality" and failure["reason"] == "retry limit exceeded":
            if (
                not attempts
                or attempts[-1]["steps"] != plan["policy"]["max_steps"]
                or failure["steps"] != plan["policy"]["max_steps"]
            ):
                raise ValueError("invalid IG retry exhaustion")
        elif failure["steps"] != 200 * 2 ** len(attempts) or not endpoint_ok:
            raise ValueError("invalid failed integration resolution")
    rebuilt = summarize(mass, attempts, outputs)
    if result["summary"] != rebuilt:
        raise ValueError("IG summary/reconstruction mismatch")


def _provenance(identity, session, plan):
    return {
        "model_sha256": identity["export_files"]["model.safetensors"],
        "tokenizer_sha256": digest(
            {
                name: checksum
                for name, checksum in identity["export_files"].items()
                if name in ("vocab.txt", "tokenizer_config.json")
            }
        ),
        "protocol_sha256": digest(
            {"prediction": identity["protocol"], "ig_plan": digest(plan)}
        ),
        "code_sha256": identity["code_sha256"],
        "environment_sha256": digest(session["environment"]),
    }


def _schema(array):
    return {
        "signed_mass": {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "axes": ["attempt", "token"],
        }
    }


def save_ig(study, run_id, result):
    """Commit successful or failed whole-source diagnostics, never partial coverage."""
    study.checkpoint()
    state = study.status()
    if state["open_job"]["run_id"] != run_id:
        raise ValueError("IG must be saved in its active run")
    predictions = read_results(study, run_id)
    plan = read_ig_plan(study, run_id, predictions=predictions)
    if plan is None:
        raise ValueError("IG requires an immutable plan")
    identity = predictions["identity"]
    _check_result(
        result,
        plan,
        {row["source_row"]: row for row in predictions["rows"]},
        saved_vocabulary(study, run_id, identity),
        _model_width(study, run_id, identity),
    )
    session = state["runs"][run_id]["sessions"][-1]
    if (
        session.get("parity", {}).get("validated") is not True
        or result["execution"]["device"] != session["environment"]["device"]
    ):
        raise ValueError("IG requires verified matching execution session")
    buffer = io.BytesIO()
    study.guard()
    np.savez_compressed(buffer, **result["arrays"])
    data = buffer.getvalue()
    study.guard()
    name = "ig-" + uuid.uuid4().hex
    payload = {
        **result,
        "arrays": _schema(result["arrays"]["signed_mass"]),
        "identity": identity,
        "session": session,
        "plan_sha256": digest(plan),
        "layout": LAYOUT,
        "provenance": _provenance(identity, session, plan),
        "npz_file": f"runs/{run_id}/artifacts/{name}.npz",
        "npz_sha256": hashlib.sha256(data).hexdigest(),
    }
    study.save_artifact(
        name, payload, [result["row"]["source_row"]], kind="ig", npz=data
    )


def read_ig(study, run_id, *, source_ids=None, require_complete=False):
    """Verify attempts and recompute verdicts without torch, Transformers or a model."""
    predictions = read_results(study, run_id)
    plan = read_ig_plan(study, run_id, predictions=predictions)
    shards = study.artifacts(run_id, kind="ig")
    if shards and plan is None:
        raise ValueError("IG shards without an immutable plan")
    identity = predictions["identity"]
    vocabulary = saved_vocabulary(study, run_id, identity) if shards else None
    model_width = _model_width(study, run_id, identity) if shards else None
    run = study.status()["runs"][run_id]
    bindings = {row["source_row"]: row for row in predictions["rows"]}
    results = {}
    for record, sidecar in shards:
        session = sidecar["session"]
        if (
            sidecar["identity"] != identity
            or sidecar["plan_sha256"] != digest(plan)
            or sidecar["layout"] != LAYOUT
            or session not in run["sessions"]
            or session.get("parity", {}).get("validated") is not True
            or sidecar["execution"]["device"] != session["environment"]["device"]
            or sidecar["provenance"] != _provenance(identity, session, plan)
        ):
            raise ValueError("IG sidecar identity/plan/session mismatch")
        committed = run["files"].get(sidecar["npz_file"])
        if (
            not committed
            or committed["kind"] != "ig-array"
            or committed["sha256"] != sidecar["npz_sha256"]
        ):
            raise ValueError("IG NPZ is not a committed matching artifact")
        data = read_checked(study, sidecar["npz_file"], sidecar["npz_sha256"])
        with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        result = {**sidecar, "arrays": arrays}
        _check_result(result, plan, bindings, vocabulary, model_width)
        if sidecar["arrays"] != _schema(arrays["signed_mass"]):
            raise ValueError("IG array axes/dtype/shape mismatch")
        source_id = result["row"]["source_row"]
        if record["coverage"] != [source_id] or source_id in results:
            raise ValueError("IG coverage mismatch or overlap")
        results[source_id] = result
    covered = sorted(results)
    expected = plan["work"]["ids"] if plan else []
    missing = sorted(set(expected) - set(covered))
    failed = sorted(i for i, result in results.items() if not result["accepted"])
    complete = (
        plan is not None
        and not missing
        and not failed
        and not predictions["job_unresolved"]
    )
    if require_complete and not complete:
        raise ValueError("incomplete/failed IG coverage or unresolved job")
    selected = covered if source_ids is None else list(source_ids)
    if selected:
        _ids(selected, covered)
    return {
        "schema": 1,
        "method": "ig",
        "run_id": run_id,
        "identity": identity,
        "plan": plan,
        "coverage": covered,
        "missing": missing,
        "failed": failed,
        "accepted_coverage": sorted(set(covered) - set(failed)),
        "coverage_complete": plan is not None and not missing,
        "complete": complete,
        "job_unresolved": predictions["job_unresolved"],
        "results": [results[i] for i in selected],
        "gpu_executed": not study.status()["fixture"]
        and any(
            result["execution"]["device"].startswith("cuda")
            for result in results.values()
        ),
    }
