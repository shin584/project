"""Immutable rollout plans and checked, model-free attention-flow readback."""

import hashlib
import io
import json
import math
import uuid

import numpy as np

from cas12a.nt_data import canonical_bytes, digest

from .nt_artifacts import read_checked, read_results, saved_vocabulary
from .nt_inputs import DNA_SPANS, LAYOUT
from .nt_rollout import PROTOCOL, derive, parity_quality, validate_policy


def plan_rollout(predictions, *, run_id, policy):
    policy = validate_policy(policy)
    if not predictions["coverage_complete"] or not predictions["coverage"]:
        raise ValueError("rollout requires complete prediction coverage")
    ids = [row["source_row"] for row in predictions["rows"]]
    if ids != sorted(set(ids)) or ids != predictions["coverage"]:
        raise ValueError("rollout prediction ID order/coverage mismatch")
    if predictions["identity"]["protocol"].get("rollout") != PROTOCOL:
        raise ValueError("run does not bind the rollout protocol")
    return {
        "schema": 1,
        "method": "rollout-plan",
        "run_id": run_id,
        "identity": predictions["identity"],
        "protocol": PROTOCOL,
        "policy": policy,
        "ids": ids,
        "prediction_rows_sha256": digest(predictions["rows"]),
    }


def _verify_plan(plan, predictions, run_id):
    rebuilt = plan_rollout(predictions, run_id=run_id, policy=plan["policy"])
    if canonical_bytes(rebuilt) != canonical_bytes(plan):
        raise ValueError("rollout plan identity/policy/coverage mismatch")


def read_rollout_plan(study, run_id, *, predictions=None):
    predictions = read_results(study, run_id) if predictions is None else predictions
    plans = study.artifacts(run_id, kind="rollout-plan")
    if not plans:
        return None
    if len(plans) != 1:
        raise ValueError("exactly one immutable rollout plan required")
    record, plan = plans[0]
    _verify_plan(plan, predictions, run_id)
    if record["coverage"] != plan["ids"]:
        raise ValueError("rollout plan committed coverage mismatch")
    return plan


def save_rollout_plan(study, run_id, plan):
    study.checkpoint()
    if study.status()["open_job"]["run_id"] != run_id:
        raise ValueError("rollout plan must be saved in its active run")
    predictions = read_results(study, run_id)
    _verify_plan(plan, predictions, run_id)
    existing = read_rollout_plan(study, run_id, predictions=predictions)
    if existing is not None:
        if canonical_bytes(existing) != canonical_bytes(plan):
            raise ValueError("immutable rollout plan cannot be changed")
        return
    if study.artifacts(run_id, kind="rollout"):
        raise ValueError("rollout shards without immutable plan")
    study.save_artifact(
        "rollout-plan-" + uuid.uuid4().hex, plan, plan["ids"], kind="rollout-plan"
    )


def _model_config(study, run_id, identity):
    path = f"runs/{run_id}/snapshot/selected_export/config.json"
    return json.loads(
        read_checked(study, path, identity["export_files"]["config.json"])
    )


def _check_result(result, plan, bindings, vocabulary, config, session):
    row = result["row"]
    source = bindings.get(row["source_row"])
    if source is None or row != {
        **{
            key: source[key]
            for key in ("source_row", "sequence", "sequence_sha256", "raw_prediction")
        },
        "prediction_sha256": digest(source),
    }:
        raise ValueError("rollout prediction/source identity mismatch")
    expected_ids = (
        [vocabulary["<cls>"]]
        + [vocabulary[source["sequence"][start:end]] for start, end in DNA_SPANS]
        + [vocabulary["<pad>"]] * 2
    )
    if (
        result["tokens"]
        != {
            "record": source["tokens"],
            "valid_positions": list(range(10)),
            "dna_positions": list(range(1, 10)),
        }
        or source["tokens"]["input_ids"] != expected_ids
    ):
        raise ValueError("rollout token identity/span mismatch")
    if (
        result["schema"] != 1
        or result["protocol"] != PROTOCOL
        or result["policy"] != plan["policy"]
    ):
        raise ValueError("rollout protocol/policy mismatch")
    execution = result["execution"]
    expected_execution = {
        "backend": "eager",
        "device": session["environment"]["device"],
        "layers": config["num_hidden_layers"],
        "heads": config["num_attention_heads"],
        "layer_order": list(range(config["num_hidden_layers"])),
        "attention_classes": ["transformers.models.esm.modeling_esm.EsmSelfAttention"]
        * config["num_hidden_layers"],
        "packages": {
            name: session["environment"]["packages"][name]
            for name in ("torch", "transformers")
        },
        "parameter_dtype": "float32",
        "output_dtype": "float32",
        "autocast": False,
        "tf32": False,
        "eval": True,
        "gradient_enabled": False,
    }
    if (
        config["model_type"] != "esm"
        or execution != expected_execution
        or session.get("parity", {}).get("validated") is not True
    ):
        raise ValueError("rollout backend/config/version/session mismatch")
    outputs = result["outputs"]
    if set(outputs) != {"native", "attention_enabled"}:
        raise ValueError("rollout output schema mismatch")
    for value in outputs.values():
        if value is not None and (
            type(value) not in (float, int)
            or not math.isfinite(value)
            or float(np.float32(value)) != value
        ):
            raise ValueError("invalid saved FP32 rollout output")
    quality = parity_quality(outputs, source["raw_prediction"])
    if result["quality"] != quality or type(result["accepted"]) is not bool:
        raise ValueError("rollout parity/quality mismatch")
    if result["accepted"]:
        if (
            result["failure"] is not None
            or quality["saved_prediction_parity"] is not True
            or quality["attention_parity"] is not True
        ):
            raise ValueError("accepted rollout has parity failure")
        arrays = result["arrays"]
        if set(arrays) != {"attention_mean", "full", "sequence"} or arrays[
            "attention_mean"
        ].shape != (execution["layers"], 10, 10):
            raise ValueError("rollout layer/array schema mismatch")
        rebuilt, summary = derive(arrays["attention_mean"], plan["policy"])
        for name, array in arrays.items():
            if array.dtype != np.dtype("float64") or not np.array_equal(
                array, rebuilt[name]
            ):
                raise ValueError("rollout derived array mismatch")
        if result["summary"] != summary:
            raise ValueError("rollout mass/query summary mismatch")
    else:
        failure = result["failure"]
        if (
            result["arrays"] is not None
            or result["summary"] is not None
            or not isinstance(failure, dict)
            or set(failure) != {"stage", "reason"}
            or not isinstance(failure["reason"], str)
            or not failure["reason"]
        ):
            raise ValueError("failed rollout requires null arrays and explicit reason")
        stage = failure["stage"]
        valid = {
            "native_output": outputs["native"] is None
            and outputs["attention_enabled"] is None,
            "attention_output": outputs["native"] is not None
            and outputs["attention_enabled"] is None,
            "parity": quality["saved_prediction_parity"] is False
            or quality["attention_parity"] is False,
            "attention": quality["saved_prediction_parity"] is True
            and quality["attention_parity"] is True,
        }
        if not valid.get(stage):
            raise ValueError("rollout failure stage/output mismatch")


def _schema(arrays):
    if arrays is None:
        return None
    axes = {
        "attention_mean": ["layer", "valid_query_token", "valid_key_token"],
        "full": ["valid_query_token", "valid_key_token"],
        "sequence": ["dna_query_token", "dna_key_token"],
    }
    return {
        name: {
            "axes": axes[name],
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "quantity": "head-mean attention probability"
            if name == "attention_mean"
            else PROTOCOL["quantity"],
            "normalization": "rows sum to one within tolerance"
            if name != "sequence"
            else "CLS removed; no renormalization",
        }
        for name, array in arrays.items()
    }


def _provenance(identity, session, plan):
    return {
        "model_sha256": identity["export_files"]["model.safetensors"],
        "tokenizer_sha256": digest(
            {
                name: value
                for name, value in identity["export_files"].items()
                if name in ("vocab.txt", "tokenizer_config.json")
            }
        ),
        "protocol_sha256": digest(
            {"prediction": identity["protocol"], "rollout_plan": digest(plan)}
        ),
        "code_sha256": identity["code_sha256"],
        "environment_sha256": digest(session["environment"]),
    }


def save_rollout(study, run_id, result):
    """Commit one source's successful or failed evidence as a whole shard."""
    study.checkpoint()
    state = study.status()
    if state["open_job"]["run_id"] != run_id:
        raise ValueError("rollout must be saved in its active run")
    predictions = read_results(study, run_id)
    plan = read_rollout_plan(study, run_id, predictions=predictions)
    if plan is None:
        raise ValueError("rollout requires an immutable plan")
    identity = predictions["identity"]
    session = state["runs"][run_id]["sessions"][-1]
    _check_result(
        result,
        plan,
        {row["source_row"]: row for row in predictions["rows"]},
        saved_vocabulary(study, run_id, identity),
        _model_config(study, run_id, identity),
        session,
    )
    data = None
    if result["arrays"] is not None:
        buffer = io.BytesIO()
        study.guard()
        np.savez_compressed(buffer, **result["arrays"])
        data = buffer.getvalue()
        study.guard()
    name = "rollout-" + uuid.uuid4().hex
    payload = {
        **result,
        "arrays": None,
        "array_schema": _schema(result["arrays"]),
        "identity": identity,
        "session": session,
        "plan_sha256": digest(plan),
        "layout": LAYOUT,
        "provenance": _provenance(identity, session, plan),
        "npz_file": f"runs/{run_id}/artifacts/{name}.npz" if data is not None else None,
        "npz_sha256": hashlib.sha256(data).hexdigest() if data is not None else None,
    }
    study.save_artifact(
        name, payload, [result["row"]["source_row"]], kind="rollout", npz=data
    )


def read_rollout(study, run_id, *, source_ids=None, require_complete=False):
    """Verify committed bytes and reconstruct flow/mass/quality without a model."""
    predictions = read_results(study, run_id)
    plan = read_rollout_plan(study, run_id, predictions=predictions)
    shards = study.artifacts(run_id, kind="rollout")
    if shards and plan is None:
        raise ValueError("rollout shards without immutable plan")
    identity = predictions["identity"]
    run = study.status()["runs"][run_id]
    bindings = {row["source_row"]: row for row in predictions["rows"]}
    vocabulary = saved_vocabulary(study, run_id, identity) if shards else None
    config = _model_config(study, run_id, identity) if shards else None
    results = {}
    for record, sidecar in shards:
        session = sidecar["session"]
        if (
            sidecar["identity"] != identity
            or sidecar["plan_sha256"] != digest(plan)
            or sidecar["layout"] != LAYOUT
            or session not in run["sessions"]
            or sidecar["provenance"] != _provenance(identity, session, plan)
            or sidecar["arrays"] is not None
        ):
            raise ValueError("rollout identity/plan/session mismatch")
        arrays = None
        if sidecar["npz_file"] is not None:
            committed = run["files"].get(sidecar["npz_file"])
            if (
                not committed
                or committed["kind"] != "rollout-array"
                or committed["sha256"] != sidecar["npz_sha256"]
            ):
                raise ValueError("rollout NPZ is not a committed matching artifact")
            data = read_checked(study, sidecar["npz_file"], sidecar["npz_sha256"])
            with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
                arrays = {name: loaded[name] for name in loaded.files}
        elif sidecar["npz_sha256"] is not None:
            raise ValueError("rollout NPZ hash without file")
        result = {**sidecar, "arrays": arrays}
        _check_result(result, plan, bindings, vocabulary, config, session)
        if sidecar["array_schema"] != _schema(arrays):
            raise ValueError("rollout axes/quantity/normalization schema mismatch")
        source_id = result["row"]["source_row"]
        if record["coverage"] != [source_id] or source_id in results:
            raise ValueError("rollout source coverage mismatch or overlap")
        results[source_id] = result
    coverage = sorted(results)
    missing = sorted(set(plan["ids"] if plan else []) - set(coverage))
    failed = sorted(i for i, result in results.items() if not result["accepted"])
    complete = (
        plan is not None
        and not missing
        and not failed
        and not predictions["job_unresolved"]
    )
    if require_complete and not complete:
        raise ValueError("incomplete/failed rollout coverage or unresolved job")
    selected = coverage if source_ids is None else list(source_ids)
    if (
        any(type(i) is not int for i in selected)
        or len(set(selected)) != len(selected)
        or not set(selected) <= set(coverage)
    ):
        raise ValueError("selected rollout source IDs are duplicate or unavailable")
    return {
        "schema": 1,
        "method": "rollout",
        "run_id": run_id,
        "identity": identity,
        "plan": plan,
        "coverage": coverage,
        "missing": missing,
        "failed": failed,
        "accepted_coverage": sorted(set(coverage) - set(failed)),
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
