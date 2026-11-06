"""Immutable IG membership and prediction-only case lists; no model imports.

Scientific cost acceptance belongs to #146 and verified FP32 prediction inputs
to #147. Independently retained manifest checksums are the integrity anchors.
"""

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from cas12a.cnn_manifest import verify_manifest
from cas12a.nt_data import canonical_bytes, digest

from .nt_run import DirectoryStore

COVERAGE_PROTOCOL = {
    "version": "cas12a-ig-coverage-v1",
    "population_order": "source_row ascending",
    "selection_order": "full: source_row ascending; subset: PCG64(42) permutation prefix",
    "storage_order": "source_row ascending",
    "seed": 42,
    "fallback_floor": 400,
}
CASE_PROTOCOL = {
    "version": "cas12a-technical-cases-v1",
    "quantile_method": "linear",
    "quantile_population": "all Test raw absolute errors before exclusions",
    "category_order": ["out_of_range", "high_error", "low_error"],
    "limit_per_category": 5,
    "out_of_range_order": "clipping distance descending, raw absolute error descending, source_row ascending",
    "high_error_rule": "raw absolute error >= Q75 and > Q25; error descending, source_row ascending",
    "low_error_rule": "raw absolute error <= Q25; error ascending, source_row ascending",
    "exclude_previous": True,
    "pam_quota": None,
    "force_fill": False,
}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _checksum(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("expected a lowercase SHA256 checksum")
    return value


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected nonempty provenance text")
    return value


def _identity(value):
    if not isinstance(value, dict):
        raise TypeError("analysis identity must bind study, run, model and protocol")
    for key in ("study_id", "run_id"):
        _text(value.get(key))
    for key in ("model_sha256", "protocol_sha256"):
        _checksum(value.get(key))
    canonical_bytes(value)


def _write(directory, manifest, inputs=None):
    store = DirectoryStore(directory)
    if store.root.exists() and any(store.root.iterdir()):
        raise ValueError("selection requires a new empty directory")
    for name, data in (inputs or {}).items():
        store.put(name, data)
    checksum = store.put("manifest.json", canonical_bytes(manifest))
    store.put("checksum.json", canonical_bytes({"sha256": checksum}))
    return checksum


def _read(directory, checksum, *, input_names=()):
    _checksum(checksum)
    store = DirectoryStore(directory)
    expected = {"manifest.json", "checksum.json", *input_names}
    actual = {p.relative_to(store.root).as_posix() for p in store.root.rglob("*")}
    if actual != expected:
        raise ValueError("selection file set is incomplete or unexpected")
    raw = store.read("manifest.json")
    if _sha(raw) != checksum or json.loads(store.read("checksum.json")) != {
        "sha256": checksum
    }:
        raise ValueError("selection manifest checksum mismatch")
    manifest = json.loads(raw)
    if canonical_bytes(manifest) != raw:
        raise ValueError("selection manifest must use canonical JSON")
    for name in input_names:
        if _sha(store.read(name)) != manifest["files"][name]:
            raise ValueError(f"selection input checksum mismatch: {name}")
    return manifest


def _coverage(partition, decision, *, mode, size, fixture, files, numpy_version):
    verify_manifest(partition)
    if type(fixture) is not bool or (partition["run_kind"] == "fixture") != fixture:
        raise ValueError("fixture/production membership mismatch")
    if (
        decision.get("schema") != 1
        or decision.get("partition") != "validation"
        or decision.get("partition_sha256") != partition["partition_sha256"]
        or type(decision.get("full_rejected")) is not bool
    ):
        raise ValueError("invalid validation cost decision or partition binding")
    _identity(decision.get("analysis_identity"))
    for key in ("reason", "measurement_reference"):
        _text(decision.get(key))
    _checksum(decision.get("measurement_sha256"))
    _text(numpy_version)
    if set(files) != {"partition.json", "decision.json"}:
        raise ValueError("invalid coverage input file set")
    for value in files.values():
        _checksum(value)
    population = sorted(partition["partitions"]["test"])
    if mode not in ("full", "subset") or decision["full_rejected"] != (
        mode == "subset"
    ):
        raise ValueError("invalid coverage mode or cost decision")
    if mode == "full" and size is None:
        size = len(population)
    if type(size) is not int:
        raise ValueError("coverage size must be an integer")
    if mode == "full":
        if size != len(population):
            raise ValueError("full coverage size must equal the population")
        selected = population
    else:
        if not 400 <= size < len(population):
            raise ValueError(
                "subset size must meet the 400 floor and be below full coverage"
            )
        permutation = np.random.Generator(np.random.PCG64(42)).permutation(
            len(population)
        )
        selected = [population[int(i)] for i in permutation[:size]]
    storage = sorted(selected)
    return {
        "schema": 1,
        "kind": "ig_coverage",
        "protocol": COVERAGE_PROTOCOL,
        "fixture": fixture,
        "mode": mode,
        "size": size,
        "numpy_version": numpy_version,
        "population_ids": population,
        "population_sha256": digest(population),
        "selection_ids": selected,
        "storage_ids": storage,
        "membership_sha256": digest(storage),
        "analysis_identity": decision["analysis_identity"],
        "partition_manifest": partition,
        "cost_decision": decision,
        "files": files,
        "historical_fp16_test_known": True,
    }


def _check_coverage(value):
    rebuilt = _coverage(
        value["partition_manifest"],
        value["cost_decision"],
        mode=value["mode"],
        size=value["size"],
        fixture=value["fixture"],
        files=value["files"],
        numpy_version=value["numpy_version"],
    )
    if canonical_bytes(rebuilt) != canonical_bytes(value):
        raise ValueError("coverage membership, ordering or protocol mismatch")


def freeze_coverage(
    partition_path,
    decision_path,
    directory,
    *,
    decision_checksum,
    mode,
    size=None,
    fixture=False,
):
    """Copy an existing partition and accepted cost decision into a new manifest."""
    inputs = {
        "partition.json": Path(partition_path).read_bytes(),
        "decision.json": Path(decision_path).read_bytes(),
    }
    if _sha(inputs["decision.json"]) != _checksum(decision_checksum):
        raise ValueError("validation decision checksum mismatch")
    manifest = _coverage(
        json.loads(inputs["partition.json"]),
        json.loads(inputs["decision.json"]),
        mode=mode,
        size=size,
        fixture=fixture,
        files={name: _sha(data) for name, data in inputs.items()},
        numpy_version=np.__version__,
    )
    return _write(directory, manifest, inputs)


def read_coverage(directory, *, checksum):
    """Verify copied evidence, trusted checksum and reconstructed membership."""
    value = _read(directory, checksum, input_names=("partition.json", "decision.json"))
    for name, key in (
        ("partition.json", "partition_manifest"),
        ("decision.json", "cost_decision"),
    ):
        if canonical_bytes(
            json.loads(DirectoryStore(directory).read(name))
        ) != canonical_bytes(value[key]):
            raise ValueError("coverage snapshot differs from manifest")
    _check_coverage(value)
    return value


def _prediction_inputs(coverage, predictions):
    bindings = {
        row["source_row"]: row["sequence_sha256"]
        for row in coverage["partition_manifest"]["rows"]
    }
    rows, seen = [], set()
    for row in predictions:
        source_row = row["source_row"]
        if type(source_row) is not int or source_row < 0 or source_row in seen:
            raise ValueError("prediction IDs must be unique nonnegative integers")
        seen.add(source_row)
        sequence = row["sequence"]
        if (
            not isinstance(sequence, str)
            or len(sequence) != 34
            or set(sequence) - set("ACGT")
            or sequence[4:7] != "TTT"
            or bindings.get(source_row) != _sha(sequence.encode("ascii"))
        ):
            raise ValueError("prediction sequence differs from frozen source identity")
        if any(
            isinstance(row[key], (bool, str)) for key in ("label", "raw_prediction")
        ):
            raise ValueError("prediction and label must be finite numeric values")
        label, raw = float(row["label"]), float(row["raw_prediction"])
        if not math.isfinite(label) or not 0 <= label <= 1 or not math.isfinite(raw):
            raise ValueError("nonfinite prediction or invalid fraction label")
        if not math.isfinite(abs(raw - label)):
            raise ValueError("nonfinite raw absolute error")
        rows.append(
            {
                "source_row": source_row,
                "sequence": sequence,
                "label": label,
                "raw_prediction": raw,
            }
        )
    if seen != set(coverage["population_ids"]):
        raise ValueError(
            "prediction coverage must equal the complete frozen population"
        )
    return sorted(rows, key=lambda row: row["source_row"])


def _cases(coverage, predictions, prediction_identity, numpy_version):
    _check_coverage(coverage)
    _identity(prediction_identity)
    if canonical_bytes(prediction_identity) != canonical_bytes(
        coverage["analysis_identity"]
    ):
        raise ValueError("prediction analysis identity differs from frozen coverage")
    _text(numpy_version)
    rows = _prediction_inputs(coverage, predictions)
    details = []
    for row in rows:
        raw, label = row["raw_prediction"], row["label"]
        details.append(
            {
                "source_row": row["source_row"],
                "raw_prediction": raw,
                "label": label,
                "raw_absolute_error": abs(raw - label),
                "clipping_distance": abs(raw - min(1.0, max(0.0, raw))),
                "pam_stratum": "TTTT" if row["sequence"][7] == "T" else "TTTV",
            }
        )
    q25, q75 = (
        float(v)
        for v in np.quantile(
            np.asarray(
                [row["raw_absolute_error"] for row in details], dtype=np.float64
            ),
            [0.25, 0.75],
            method="linear",
        )
    )
    if not math.isfinite(q25) or not math.isfinite(q75):
        raise ValueError("nonfinite case quantile")
    ordered = {
        "out_of_range": sorted(
            [r for r in details if r["raw_prediction"] < 0 or r["raw_prediction"] > 1],
            key=lambda r: (
                -r["clipping_distance"],
                -r["raw_absolute_error"],
                r["source_row"],
            ),
        ),
        "high_error": sorted(
            [
                r
                for r in details
                if r["raw_absolute_error"] >= q75 and r["raw_absolute_error"] > q25
            ],
            key=lambda r: (-r["raw_absolute_error"], r["source_row"]),
        ),
        "low_error": sorted(
            [r for r in details if r["raw_absolute_error"] <= q25],
            key=lambda r: (r["raw_absolute_error"], r["source_row"]),
        ),
    }
    selected, used, categories = [], set(), {}
    for role, eligible in ordered.items():
        available = [row for row in eligible if row["source_row"] not in used]
        chosen = available[:5]
        for rank, row in enumerate(chosen, 1):
            selected.append(
                {
                    **row,
                    "role": role,
                    "rank_in_category": rank,
                    "selection_reason": CASE_PROTOCOL[f"{role}_rule"]
                    if role != "out_of_range"
                    else "raw < 0 or raw > 1; " + CASE_PROTOCOL["out_of_range_order"],
                }
            )
            used.add(row["source_row"])
        categories[role] = {
            "eligible": len(eligible),
            "available": len(available),
            "selected": len(chosen),
            "unfilled": 5 - len(chosen),
        }
    return {
        "schema": 1,
        "kind": "technical_cases",
        "protocol": CASE_PROTOCOL,
        "numpy_version": numpy_version,
        "fixture": coverage["fixture"],
        "coverage_checksum": digest(coverage),
        "prediction_identity": prediction_identity,
        "predictions": rows,
        "predictions_sha256": digest(rows),
        "thresholds": {"q25": q25, "q75": q75, "degenerate": q25 == q75},
        "categories": categories,
        "cases": selected,
        "historical_fp16_test_known": True,
    }


def _check_cases(coverage, cases):
    rebuilt = _cases(
        coverage,
        cases["predictions"],
        cases["prediction_identity"],
        cases["numpy_version"],
    )
    if canonical_bytes(rebuilt) != canonical_bytes(cases):
        raise ValueError("case selection, coverage binding or protocol mismatch")


def save_cases(
    coverage_directory,
    directory,
    *,
    coverage_checksum,
    predictions,
    prediction_identity,
):
    """Select from complete verified FP32 input rows, without explanation inputs."""
    coverage = read_coverage(coverage_directory, checksum=coverage_checksum)
    return _write(
        directory, _cases(coverage, predictions, prediction_identity, np.__version__)
    )


def read_cases(coverage_directory, directory, *, coverage_checksum, checksum):
    """Reconstruct the entire case selection from its saved prediction snapshot."""
    coverage = read_coverage(coverage_directory, checksum=coverage_checksum)
    cases = _read(directory, checksum)
    _check_cases(coverage, cases)
    return cases


def ig_work(coverage, cases):
    """Return one IG task per ID and preserve the original statistics denominator."""
    _check_cases(coverage, cases)
    statistics = set(coverage["storage_ids"])
    roles = {
        i: ["full_population" if coverage["mode"] == "full" else "random_subset"]
        for i in statistics
    }
    for row in cases["cases"]:
        roles.setdefault(row["source_row"], []).append(row["role"])
    ids = sorted(roles)
    return {
        "ids": ids,
        "statistics_ids": sorted(statistics),
        "additional_case_ids": sorted(set(ids) - statistics),
        "rows": [{"source_row": i, "roles": roles[i]} for i in ids],
        "coverage_checksum": digest(coverage),
        "case_checksum": digest(cases),
    }


def _annotations(cases, annotations):
    selected = {row["source_row"] for row in cases["cases"]}
    seen, notes = set(), []
    for row in annotations:
        source_row = row["source_row"]
        if (
            type(source_row) is not int
            or source_row not in selected
            or source_row in seen
        ):
            raise ValueError("notes require unique selected case IDs")
        seen.add(source_row)
        notes.append({"source_row": source_row, "note": _text(row["note"])})
    return {
        "schema": 1,
        "kind": "case_annotations",
        "selection_checksum": digest(cases),
        "annotations": sorted(notes, key=lambda row: row["source_row"]),
    }


def save_annotations(
    coverage_directory,
    selection_directory,
    directory,
    *,
    coverage_checksum,
    selection_checksum,
    annotations,
):
    """Write separate notes referring to the immutable original case checksum."""
    cases = read_cases(
        coverage_directory,
        selection_directory,
        coverage_checksum=coverage_checksum,
        checksum=selection_checksum,
    )
    return _write(directory, _annotations(cases, annotations))


def read_annotations(
    coverage_directory,
    selection_directory,
    directory,
    *,
    coverage_checksum,
    selection_checksum,
    checksum,
):
    """Check notes and their original selection without changing that selection."""
    cases = read_cases(
        coverage_directory,
        selection_directory,
        coverage_checksum=coverage_checksum,
        checksum=selection_checksum,
    )
    notes = _read(directory, checksum)
    if canonical_bytes(_annotations(cases, notes["annotations"])) != canonical_bytes(
        notes
    ):
        raise ValueError("annotation schema or original selection checksum mismatch")
    return notes
