"""Durable Test shards and a model-free, identity-checked analysis reader."""

import hashlib
import io
import json
import uuid

import numpy as np

from cas12a.nt_data import canonical_bytes, digest

from .nt_artifacts import prediction_rows
from .nt_inputs import DNA_SPANS, LAYOUT, row_identity
from .nt_run import DirectoryStore
from .nt_selection import CASE_PROTOCOL, select_technical_cases

METHODS = ("prediction", "ism", "rollout", "ig")


def array_schema(arrays):
    return {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in arrays.items()
    }


def _check_numerics(method, shard, row, plan):
    """Reconstruct small derived quantities from saved arrays, without inference."""
    result, arrays = shard["result"], shard["arrays"]
    if result.get("execution_failure"):
        if (
            shard["status"] != "failed"
            or result["accepted"] is not False
            or not result["failure"]["reason"]
            or arrays is not None
        ):
            raise ValueError("invalid method execution failure record")
        return
    if method == "ism":
        from .nt_ism import PROTOCOL, _derived

        if result["protocol"] != PROTOCOL or arrays is None:
            raise ValueError("incompatible ISM protocol/arrays")
        rebuilt = _derived([row], arrays["prediction"])
        if set(rebuilt) != set(arrays) or any(
            not np.array_equal(rebuilt[k], arrays[k], equal_nan=True) for k in rebuilt
        ):
            raise ValueError("saved ISM derived arrays mismatch")
        if result["rows"][0]["tokens"][0] != row["tokens"]:
            raise ValueError("ISM WT token identity mismatch")
    elif method == "ig":
        from .nt_ig import PROTOCOL, quality, summarize
        from .nt_predict import ATOL, RTOL

        if result["protocol"] != PROTOCOL or result["policy"] != plan["ig_policy"]:
            raise ValueError("incompatible IG protocol/policy")
        mass = arrays["signed_mass"] if arrays is not None else None
        attempts = result["attempts"]
        if (
            mass is None
            or mass.shape != (len(attempts), 12)
            or mass.dtype != np.dtype("float64")
            or not np.isfinite(mass).all()
            or result["tokens"] != row["tokens"]
        ):
            raise ValueError("invalid IG arrays/actual token identity")
        outputs = result["outputs"]
        for index, attempt in enumerate(attempts):
            if attempt["steps"] != 200 * 2**index or attempt["quality"] != quality(
                mass[index],
                mass[index - 1] if index else None,
                outputs,
                result["policy"],
            ):
                raise ValueError("saved IG quality/retry sequence mismatch")
        if summarize(mass, attempts, outputs) != result["summary"]:
            raise ValueError("saved IG summary mismatch")
        if result["accepted"]:
            if (
                not attempts
                or attempts[-1]["steps"] > plan["ig_policy"]["max_steps"]
                or not attempts[-1]["quality"]["completeness_pass"]
                or not attempts[-1]["quality"]["convergence_pass"]
                or any(v is None or not np.isfinite(v) for v in outputs.values())
                or not np.isclose(
                    outputs["native"], row["raw_prediction"], atol=ATOL, rtol=RTOL
                )
                or not np.isclose(
                    outputs["endpoint"], outputs["native"], atol=ATOL, rtol=RTOL
                )
                or result["failure"] is not None
            ):
                raise ValueError(
                    "IG accepted without valid endpoint/completeness/convergence"
                )
        elif result["failure"] is None:
            raise ValueError("failed IG requires an explicit reason")
    else:
        from .nt_rollout import PROTOCOL, derive, parity_quality

        if result["protocol"] != PROTOCOL or result["policy"] != plan["rollout_policy"]:
            raise ValueError("incompatible rollout protocol/policy")
        assessed = parity_quality(result["outputs"], row["raw_prediction"])
        if assessed != result["quality"] or result["tokens"]["record"] != row["tokens"]:
            raise ValueError("saved rollout parity/actual token mismatch")
        if result["accepted"]:
            rebuilt, summary = derive(arrays["attention_mean"], result["policy"])
            if (
                not assessed["saved_prediction_parity"]
                or not assessed["attention_parity"]
                or summary != result["summary"]
                or any(
                    not np.array_equal(arrays[k], value) for k, value in rebuilt.items()
                )
                or result["failure"] is not None
            ):
                raise ValueError("saved rollout arrays/quality mismatch")
        elif result["failure"] is None:
            raise ValueError("failed rollout requires an explicit reason")


def read_plan(directory):
    store = DirectoryStore(directory)
    raw = store.read("plan.json")
    if hashlib.sha256(raw).hexdigest() != store.read("plan.sha256").decode():
        raise ValueError("main analysis plan checksum mismatch")
    plan = json.loads(raw)
    if plan["schema"] != 1 or plan["kind"] != "cas12a-test-main-analysis":
        raise ValueError("incompatible main analysis plan")
    return plan


def shard_identity(plan):
    return {
        "plan_sha256": digest(plan),
        "input_identity_sha256": digest(plan["input_identity"]),
        "code_sha256": plan["code_sha256"],
    }


def save_shard(store, identity, method, source_id, result, *, roles):
    """Commit JSON last; an orphan NPZ is never counted as completed work."""
    arrays = result.get("arrays")
    payload = {k: v for k, v in result.items() if k != "arrays"}
    manifest = {
        "schema": 1,
        "method": method,
        "source_row": source_id,
        **identity,
        "output_unit": "native raw fraction; clipped view separately identified",
        "coordinates": "0-based half-open",
        "layout": LAYOUT,
        "dna_spans": DNA_SPANS,
        "roles": roles,
        "status": "available" if result.get("accepted", True) else "failed",
        "result": payload,
        "npz": None,
        "array_schema": {},
    }
    if arrays is not None:
        arrays = {**arrays, "source_row": np.asarray([source_id], dtype=np.int64)}
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        data = buffer.getvalue()
        name = f"shards/{method}/{source_id}-{uuid.uuid4().hex}.npz"
        store.put(name, data)
        manifest["npz"] = {"file": name, "sha256": hashlib.sha256(data).hexdigest()}
        manifest["array_schema"] = array_schema(arrays)
    store.put(f"shards/{method}/{source_id}.json", canonical_bytes(manifest))
    return {**manifest, "arrays": arrays}


def _read_shard(store, identity, method, source_id):
    path = store.path(f"shards/{method}/{source_id}.json")
    if not path.exists():
        return None
    manifest = json.loads(path.read_bytes())
    if (
        manifest["schema"] != 1
        or manifest["method"] != method
        or manifest["source_row"] != source_id
        or any(manifest[k] != v for k, v in identity.items())
        or manifest["status"] not in ("available", "failed")
        or manifest["dna_spans"] != DNA_SPANS
        or manifest["layout"] != LAYOUT
    ):
        raise ValueError("incompatible main shard identity/schema")
    arrays = None
    if manifest["npz"] is not None:
        data = store.read(manifest["npz"]["file"])
        if hashlib.sha256(data).hexdigest() != manifest["npz"]["sha256"]:
            raise ValueError("main shard NPZ checksum mismatch")
        with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        actual = array_schema(arrays)
        if actual != manifest["array_schema"] or arrays["source_row"].tolist() != [
            source_id
        ]:
            raise ValueError("main shard source axis/schema mismatch")
    return {**manifest, "arrays": arrays}


def save_cases(store, plan, predictions):
    selection = {
        "plan_sha256": digest(plan),
        "prediction_sha256": digest(predictions),
        "protocol": CASE_PROTOCOL,
        **select_technical_cases(predictions),
    }
    if store.path("cases.json").exists():
        if json.loads(store.read("cases.json")) != selection:
            raise ValueError("frozen technical cases differ from saved predictions")
    else:
        store.put("cases.json", canonical_bytes(selection))
    return selection


def source_roles(plan, cases, source_id):
    roles = []
    if source_id in plan["statistics_ids"]:
        roles.append(
            "full_population" if plan["ig_mode"] == "full" else "random_subset"
        )
    roles.extend(c["role"] for c in cases if c["source_row"] == source_id)
    return roles


def read_main(directory):
    """Read each completed shard once; report absent and numerically failed IDs."""
    store, plan = DirectoryStore(directory), read_plan(directory)
    identity = shard_identity(plan)
    execution = (
        json.loads(store.read("execution.json"))
        if store.path("execution.json").exists()
        else None
    )
    sessions = {}
    population = plan["input_identity"]["test_source_rows"]
    for method in METHODS:
        unexpected = [
            p.name
            for p in store.path(f"shards/{method}").glob("*.json")
            if not p.stem.isdecimal() or int(p.stem) not in population
        ]
        if unexpected:
            raise ValueError(f"unexpected {method} source coverage: {unexpected}")
    shards = {
        method: {
            i: item
            for i in population
            if (item := _read_shard(store, identity, method, i)) is not None
        }
        for method in METHODS
    }
    for method_shards in shards.values():
        for shard in method_shards.values():
            session = shard["session"]
            if execution is None or shard["execution_sha256"] != digest(execution):
                raise ValueError("main shard execution environment mismatch")
            if session not in sessions:
                start = json.loads(store.read(session + "/start.json"))
                environment = json.loads(store.read(session + "/environment.json"))
                if start["plan_sha256"] != identity["plan_sha256"] or environment[
                    "execution_sha256"
                ] != digest(execution):
                    raise ValueError("main shard session/environment binding mismatch")
                sessions[session] = environment
    predictions = [
        shards["prediction"][i]["result"] for i in sorted(shards["prediction"])
    ]
    bindings = {r["source_row"]: r for r in plan["input_identity"]["test_rows"]}
    for row in predictions:
        source = {
            "source_row": row["source_row"],
            "sequence": row["sequence"],
            "sequence_sha256": row["sequence_sha256"],
            "target": row["label"],
            "raw": row["source_identity"],
        }
        if (
            row_identity(source) != bindings.get(row["source_row"])
            or prediction_rows([source], [row["raw_prediction"]], [row["tokens"]])[0]
            != row
        ):
            raise ValueError("saved prediction source/derived fields mismatch")
    selection, cases = None, []
    if store.path("cases.json").exists():
        selection = json.loads(store.read("cases.json"))
        expected = {
            "plan_sha256": digest(plan),
            "prediction_sha256": digest(predictions),
            "protocol": CASE_PROTOCOL,
            **select_technical_cases(predictions),
        }
        if len(predictions) != len(population) or selection != expected:
            raise ValueError("frozen case selection mismatch")
        cases = selection["cases"]
    work = sorted(set(plan["statistics_ids"]) | {c["source_row"] for c in cases})
    requested = {method: population for method in METHODS}
    requested["ig"] = work
    by_id = {r["source_row"]: r for r in predictions}
    for method in ("ism", "ig", "rollout"):
        for i, shard in shards[method].items():
            if i not in by_id or i not in requested[method]:
                raise ValueError("unexpected main method source coverage")
            result, prediction = shard["result"], by_id[i]
            if (
                method == "ism"
                and not result.get("execution_failure")
                and len(result["rows"]) != 1
            ):
                raise ValueError("ISM shard must contain exactly one source")
            binding = (
                result["rows"][0]
                if method == "ism" and not result.get("execution_failure")
                else result["row"]
            )
            if binding["prediction_sha256"] != digest(prediction) or any(
                binding[k] != prediction[k]
                for k in ("source_row", "sequence", "sequence_sha256", "raw_prediction")
            ):
                raise ValueError("main method prediction binding mismatch")
            if shard["roles"] != source_roles(plan, cases, i):
                raise ValueError("main method selection roles mismatch")
            if (method in ("ig", "rollout")) and (
                shard["status"] == "available"
            ) != result["accepted"]:
                raise ValueError("method acceptance/status mismatch")
            _check_numerics(method, shard, prediction, plan)
    coverage = {}
    for method in METHODS:
        coverage[method] = {
            "requested": requested[method],
            "available": sorted(
                i for i, s in shards[method].items() if s["status"] == "available"
            ),
            "failed": sorted(
                i for i, s in shards[method].items() if s["status"] == "failed"
            ),
            "not_computed": sorted(set(requested[method]) - shards[method].keys()),
        }
    complete = selection is not None and all(
        c["requested"] == c["available"] for c in coverage.values()
    )
    return {
        "schema": 1,
        "plan": plan,
        "predictions": predictions,
        "cases": selection,
        "coverage": coverage,
        "shards": shards,
        "complete": complete,
        "case_selection_complete": selection is not None,
        "execution": execution,
        "sessions": sessions,
    }
