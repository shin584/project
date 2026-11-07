"""Checked source-ID joins and immutable, independently replayable rank exports."""

import hashlib
import json
import platform

import numpy as np
import scipy

from cas12a.nt_data import canonical_bytes, digest

from .nt_agreement import PROTOCOL, sequence_agreement, summarize_population
from .nt_artifacts import read_results
from .nt_ig_artifacts import plan_ig, read_ig
from .nt_ism import read_ism
from .nt_run import DirectoryStore


def _index(rows):
    indexed = {row["source_row"]: row for row in rows}
    if len(indexed) != len(rows) or any(type(i) is not int for i in indexed):
        raise ValueError("duplicate or invalid source ID")
    return indexed


def _binding(row, prediction):
    if any(
        row[key] != prediction[key]
        for key in (
            "source_row",
            "sequence",
            "sequence_sha256",
            "raw_prediction",
        )
    ) or row["prediction_sha256"] != digest(prediction):
        raise ValueError("IG/ISM source/prediction identity mismatch")


def _capture(study, run_id):
    predictions = read_results(study, run_id)
    ig, ism = read_ig(study, run_id), read_ism(study, run_id)
    if predictions["run_id"] != run_id:
        raise ValueError("prediction run ID mismatch")
    return _join(predictions, ig, ism, study.receipt)


def _join(predictions, ig, ism, receipt):
    run_id = predictions["run_id"]
    if any(
        report["identity"] != predictions["identity"] or report["run_id"] != run_id
        for report in (ig, ism)
    ):
        raise ValueError("IG/ISM run/model/tokenizer/protocol identity mismatch")
    plan = ig["plan"]
    if plan is None:
        raise ValueError("agreement requires an immutable IG population plan")
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
        raise ValueError("agreement IG plan population/protocol mismatch")
    bindings = _index(predictions["rows"])
    ig_sources = _index([{**item["row"], "result": item} for item in ig["results"]])
    ism_sources = _index(ism["rows"])
    if sorted(ig_sources) != ig["coverage"] or sorted(ism_sources) != ism["coverage"]:
        raise ValueError("method source IDs differ from stored coverage")
    array_ids = ism["arrays"].get("source_row", np.array([], dtype=int)).tolist()
    if array_ids != [row["source_row"] for row in ism["rows"]]:
        raise ValueError("ISM array source ID order mismatch")
    ism_offsets = {source_id: index for index, source_id in enumerate(array_ids)}
    roles = {row["source_row"]: row["roles"] for row in plan["work"]["rows"]}
    rows = []
    for source_id in plan["work"]["ids"]:
        prediction = bindings[source_id]
        ig_row, ism_row = ig_sources.get(source_id), ism_sources.get(source_id)
        reasons, mass, delta, valid = [], None, None, None
        if ig_row is None:
            reasons.append("missing IG")
        else:
            _binding(ig_row, prediction)
            result = ig_row["result"]
            if result["tokens"] != prediction["tokens"]:
                raise ValueError("IG actual token/span identity mismatch")
            if not result["accepted"]:
                reasons.append(
                    "failed IG: "
                    + result["failure"]["stage"]
                    + ": "
                    + result["failure"]["reason"]
                )
            else:
                mass = result["arrays"]["signed_mass"][-1, 1:10].tolist()
        if ism_row is None:
            reasons.append("missing ISM")
        else:
            _binding(ism_row, prediction)
            if ism_row["tokens"][0] != prediction["tokens"]:
                raise ValueError("ISM actual WT token/span identity mismatch")
            offset = ism_offsets[source_id]
            delta = ism["arrays"]["raw_delta"][offset].tolist()
            valid = ism["arrays"]["base_valid_mask"][offset].tolist()
        rows.append(
            {
                "source_row": source_id,
                "prediction": prediction,
                "roles": roles[source_id],
                "exclusion_reasons": reasons,
                "signed_mass": mass,
                "raw_delta": delta,
                "base_valid_mask": valid,
            }
        )
    return {
        "schema": 1,
        "method": "ig-ism-agreement-inputs",
        "protocol": PROTOCOL,
        "run_id": run_id,
        "identity": predictions["identity"],
        "study_receipt": receipt,
        "plan": plan,
        "coverage": {
            "prediction": {
                key: predictions[key]
                for key in ("coverage", "missing", "complete", "job_unresolved")
            },
            "ig": {
                key: ig[key]
                for key in (
                    "coverage",
                    "accepted_coverage",
                    "missing",
                    "failed",
                    "complete",
                    "job_unresolved",
                )
            },
            "ism": {
                key: ism[key]
                for key in ("coverage", "missing", "complete", "job_unresolved")
            },
        },
        "rows": rows,
    }


def _recompute(inputs):
    if (
        inputs["schema"] != 1
        or inputs["method"] != "ig-ism-agreement-inputs"
        or inputs["protocol"] != PROTOCOL
    ):
        raise ValueError("incompatible agreement input schema/protocol")
    plan, identity = inputs["plan"], inputs["identity"]
    if plan["identity"] != identity or plan["run_id"] != inputs["run_id"]:
        raise ValueError("agreement plan/run identity mismatch")
    work = plan["work"]
    if [row["source_row"] for row in inputs["rows"]] != work["ids"]:
        raise ValueError("agreement expected source coverage mismatch")
    indexed = _index(inputs["rows"])
    rows = []
    for item in inputs["rows"]:
        prediction = item["prediction"]
        if item["source_row"] != prediction["source_row"]:
            raise ValueError("agreement source identity mismatch")
        agreement = None
        if not item["exclusion_reasons"]:
            agreement = sequence_agreement(
                prediction["sequence"],
                item["signed_mass"],
                item["raw_delta"],
                spans=prediction["tokens"]["dna_spans"],
                base_valid_mask=item["base_valid_mask"],
            )
        rows.append(
            {
                "source_row": item["source_row"],
                "sequence": prediction["sequence"],
                "sequence_sha256": prediction["sequence_sha256"],
                "prediction_sha256": digest(prediction),
                "tokens": prediction["tokens"],
                "roles": item["roles"],
                "exclusion_reasons": item["exclusion_reasons"],
                "agreement": agreement,
            }
        )
    result_by_id = _index(rows)
    populations = {
        "statistics": work["statistics_ids"],
        "additional_cases": work["additional_case_ids"],
        "validation_probes": work["ids"] if plan["kind"] == "validation_probe" else [],
    }
    if any(not set(ids) <= indexed.keys() for ids in populations.values()):
        raise ValueError("agreement population IDs outside planned union")
    unavailable = [row["source_row"] for row in rows if row["agreement"] is None]
    return {
        "schema": 1,
        "method": "ig-ism-agreement",
        "protocol": PROTOCOL,
        "run_id": inputs["run_id"],
        "identity": identity,
        "input_sha256": digest(inputs),
        "plan_sha256": digest(plan),
        "population_kind": plan["kind"],
        "coverage": inputs["coverage"],
        "complete": not unavailable
        and all(method["complete"] for method in inputs["coverage"].values()),
        "unavailable_ids": unavailable,
        "rows": rows,
        "populations": {
            name: summarize_population([result_by_id[i] for i in ids])
            for name, ids in populations.items()
        },
        "limitations": [
            "Five-token n=5 midranks are coarse; primary covers [0,30) and omits t6–t9.",
            "PAM exclusion retains full token IG; t1/t2 ISM support is only 4bp/5bp.",
            "Global rank partitions are separate diagnostics, not rho9-rho5 confound fractions or causal effects.",
            "IG path and local ISM probe different quantities; low rho is a valid result.",
            "Exports preserve checked provenance at export time, not independent scientific validation.",
        ],
    }


def read_agreement(study, run_id):
    """Verify committed IG/ISM and join by ID without a model or study mutation."""
    return _recompute(_capture(study, run_id))


def compare_saved(predictions, ig, ism, *, receipt=None):
    """Compare checked-reader reports, preserving their source IDs and coverage.

    Callers must supply read_results/read_ig/read_ism outputs. The study-facing
    read_agreement additionally verifies committed bytes and method quality.
    """
    return _recompute(_join(predictions, ig, ism, receipt))


def save_agreement(study, run_id, directory):
    """Export inputs and recomputed results to a new directory outside the study."""
    store = DirectoryStore(directory)
    if store.root.is_relative_to(study.store.root):
        raise ValueError("agreement output must be outside the study root")
    if store.root.exists() and any(store.root.iterdir()):
        raise ValueError("agreement requires a new empty directory")
    inputs = _capture(study, run_id)
    result = _recompute(inputs)
    files = {
        name: canonical_bytes(value)
        for name, value in (("inputs.json", inputs), ("results.json", result))
    }
    manifest = {
        "schema": 1,
        "method": "ig-ism-agreement-export",
        "protocol": PROTOCOL,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "files": {
            name: hashlib.sha256(data).hexdigest() for name, data in files.items()
        },
    }
    for name, data in files.items():
        store.put(name, data)
    return store.put("manifest.json", canonical_bytes(manifest))


def load_agreement(directory, *, checksum):
    """Check an independently retained manifest hash and replay every statistic."""
    store = DirectoryStore(directory)
    expected = {"inputs.json", "results.json", "manifest.json"}
    actual = {path.relative_to(store.root).as_posix() for path in store.root.rglob("*")}
    if actual != expected:
        raise ValueError("agreement export file set is incomplete or unexpected")
    raw = store.read("manifest.json")
    if hashlib.sha256(raw).hexdigest() != checksum:
        raise ValueError("agreement manifest checksum mismatch")
    manifest = json.loads(raw)
    if (
        manifest["schema"] != 1
        or manifest["method"] != "ig-ism-agreement-export"
        or manifest["protocol"] != PROTOCOL
        or set(manifest["files"]) != {"inputs.json", "results.json"}
    ):
        raise ValueError("incompatible agreement manifest schema/protocol")
    values = {}
    for name, expected_hash in manifest["files"].items():
        data = store.read(name)
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError("agreement export file checksum mismatch: " + name)
        values[name] = json.loads(data)
        if canonical_bytes(values[name]) != data:
            raise ValueError("agreement export requires canonical JSON")
    result = _recompute(values["inputs.json"])
    if canonical_bytes(result) != canonical_bytes(values["results.json"]):
        raise ValueError("agreement stored results differ from recomputation")
    return result
