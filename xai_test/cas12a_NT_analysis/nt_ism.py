"""Single-output Cas12a saturation mutagenesis and model-free checked artifacts."""

import hashlib
import io
import uuid

import numpy as np

from cas12a.nt_data import digest

from .nt_artifacts import read_checked as _read_checked
from .nt_artifacts import read_results
from .nt_artifacts import saved_vocabulary as _vocabulary
from .nt_inputs import (
    DNA_SPANS,
    LAYOUT,
    encode_sequences,
    sequence_hash,
    validate_vocabulary,
)
from .nt_predict import ATOL, RTOL

BASES = "ACGT"
RELATIVE_FLOOR = 1e-4
PROTOCOL = {
    "schema": 1,
    "axes": ["source_row", "position", "alternative"],
    "positions": list(range(34)),
    "alternatives": list(BASES),
    "raw_delta": "mutant - WT",
    "clipped_delta": "clip(mutant,0,1) - clip(WT,0,1)",
    "relative_delta": "raw_delta / abs(WT)",
    "relative_floor": RELATIVE_FLOOR,
    "pam_breaking_positions": [4, 5, 6],
    "non_wt_per_source": 102,
    "output_unit": "fraction",
    "undefined_relative": "NaN in NPZ; JSON null with reason and validity mask",
}
QUALITY = {
    "finite_defined_outputs": True,
    "lossless_mutants": True,
    "non_wt_per_source": 102,
    "wt_parity_atol": ATOL,
    "wt_parity_rtol": RTOL,
}


def _sources(rows):
    ids = [row["source_row"] for row in rows]
    if (
        not ids
        or any(type(i) is not int or not 0 <= i <= np.iinfo(np.int64).max for i in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("source IDs must be unique nonnegative int64 values")
    for row in rows:
        sequence = row["sequence"]
        if (
            not isinstance(sequence, str)
            or len(sequence) != 34
            or set(sequence) - set(BASES)
            or sequence[4:7] != "TTT"
            or sequence_hash(sequence) != row["sequence_sha256"]
        ):
            raise ValueError("invalid source sequence/hash/PAM")
        raw = row["raw_prediction"]
        if not np.isfinite(raw) or float(np.float32(raw)) != raw:
            raise ValueError("saved WT must be a finite FP32 prediction")
    return sorted(rows, key=lambda row: row["source_row"])


def _mutations(sequence):
    return [
        (position, alternative, sequence[:position] + base + sequence[position + 1 :])
        for position, reference in enumerate(sequence)
        for alternative, base in enumerate(BASES)
        if base != reference
    ]


def _derived(rows, predictions):
    n = len(rows)
    if (
        predictions.dtype != np.dtype("float32")
        or predictions.shape != (n, 34, 4)
        or not np.isfinite(predictions).all()
    ):
        raise ValueError("expected finite FP32 ISM prediction array")
    reference = np.asarray(
        [[BASES.index(base) for base in row["sequence"]] for row in rows], dtype=np.int8
    )
    wt = np.asarray([row["raw_prediction"] for row in rows], dtype=np.float32)
    wt_mask = reference[:, :, None] == np.arange(4)
    if not np.array_equal(predictions[wt_mask], np.repeat(wt, 34)):
        raise ValueError("WT slots differ from saved original predictions")
    raw = predictions.astype(np.float64) - wt[:, None, None].astype(np.float64)
    clipped = np.clip(predictions.astype(np.float64), 0, 1) - np.clip(
        wt[:, None, None].astype(np.float64), 0, 1
    )
    relative_valid = np.abs(wt.astype(np.float64)) > RELATIVE_FLOOR
    relative = np.full(raw.shape, np.nan, dtype=np.float64)
    np.divide(
        raw,
        np.abs(wt[:, None, None].astype(np.float64)),
        out=relative,
        where=relative_valid[:, None, None],
    )
    pam = np.zeros(raw.shape, dtype=bool)
    pam[:, 4:7] = ~wt_mask[:, 4:7]
    return {
        "source_row": np.asarray([row["source_row"] for row in rows], dtype=np.int64),
        "wt_prediction": wt,
        "prediction": predictions,
        "reference_base": reference,
        "wt_mask": wt_mask,
        "non_wt_mask": ~wt_mask,
        "base_valid_mask": np.ones(raw.shape, dtype=bool),
        "raw_delta": raw,
        "clipped_delta": clipped,
        "relative_delta": relative,
        "relative_valid": relative_valid,
        "pam_breaking": pam,
        "tttn_preserving": ~pam,
        "denominator": (~wt_mask).sum(axis=2, dtype=np.int64),
        "preserving_denominator": (~wt_mask & ~pam).sum(axis=2, dtype=np.int64),
    }


def compute_ism(rows, forward, tokenizer, embedding_rows, *, batch_size=32, guard=None):
    """Retokenize 102 full mutants per source, preserving the checked saved WT."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch size must be a positive integer")
    rows = _sources(rows)
    guard = guard or (lambda: None)
    guard()
    vocabulary = validate_vocabulary(tokenizer, embedding_rows)
    predictions = np.empty((len(rows), 34, 4), dtype=np.float32)
    metadata = []
    for index, row in enumerate(rows):
        sequence = row["sequence"]
        mutations = _mutations(sequence)
        sequences = [sequence, *[item[2] for item in mutations]]
        tokens, outputs = [], []
        for start in range(0, len(sequences), batch_size):
            guard()
            batch = sequences[start : start + batch_size]
            tokens.extend(encode_sequences(tokenizer, batch, embedding_rows))
            guard()
            values = np.asarray(forward(batch))
            guard()
            if (
                values.dtype != np.dtype("float32")
                or values.shape != (len(batch),)
                or not np.isfinite(values).all()
            ):
                raise ValueError("forward must return finite scalar FP32 outputs")
            outputs.extend(values)
        if not np.isclose(outputs[0], row["raw_prediction"], atol=ATOL, rtol=RTOL):
            raise ValueError("saved WT/native prediction parity failure")
        predictions[index].fill(row["raw_prediction"])
        for (position, alternative, _), value in zip(
            mutations, outputs[1:], strict=True
        ):
            predictions[index, position, alternative] = value
        metadata.append(
            {
                "source_row": row["source_row"],
                "sequence": sequence,
                "sequence_sha256": row["sequence_sha256"],
                "raw_prediction": row["raw_prediction"],
                "prediction_sha256": digest(row),
                "tokens": tokens,
                "relative_undefined_reason": "abs(WT)<=1e-4"
                if abs(row["raw_prediction"]) <= RELATIVE_FLOOR
                else None,
            }
        )
    guard()
    return {
        "arrays": _derived(rows, predictions),
        "rows": metadata,
        "vocabulary": vocabulary,
        "protocol": PROTOCOL,
        "batch_size": batch_size,
    }


def _check_result(result, bindings, vocabulary):
    rows, arrays = result["rows"], result["arrays"]
    if _sources(rows) != rows or result["protocol"] != PROTOCOL:
        raise ValueError("incompatible ISM order/protocol")
    expected_vocabulary = {
        "vocabulary_sha256": digest(vocabulary),
        "vocabulary_size": 4107,
        "embedding_rows": 4105,
    }
    if result["vocabulary"] != expected_vocabulary:
        raise ValueError("incompatible ISM vocabulary identity")
    if type(result["batch_size"]) is not int or result["batch_size"] < 1:
        raise ValueError("invalid ISM batch size")
    for row in rows:
        binding = bindings.get(row["source_row"])
        if binding is None or row["prediction_sha256"] != digest(binding):
            raise ValueError("ISM source/prediction identity mismatch")
        if any(
            row[key] != binding[key]
            for key in ("sequence", "sequence_sha256", "raw_prediction")
        ):
            raise ValueError("ISM source/prediction fields mismatch")
        reason = (
            "abs(WT)<=1e-4" if abs(row["raw_prediction"]) <= RELATIVE_FLOOR else None
        )
        if row["relative_undefined_reason"] != reason:
            raise ValueError("relative undefined reason mismatch")
        sequences = [
            row["sequence"],
            *[item[2] for item in _mutations(row["sequence"])],
        ]
        if len(row["tokens"]) != 103:
            raise ValueError("missing actual mutant token records")
        for sequence, record in zip(sequences, row["tokens"], strict=True):
            ids = [
                3,
                *[vocabulary[sequence[start:end]] for start, end in DNA_SPANS],
                1,
                1,
            ]
            expected = {
                "input_ids": ids,
                "attention_mask": [1] * 10 + [0, 0],
                "special_mask": [True] + [False] * 9 + [True, True],
                "dna_spans": DNA_SPANS,
            }
            if record != expected or any(not 0 <= token < 4105 for token in ids):
                raise ValueError("mutant token identity/span/range mismatch")
    derived = _derived(rows, arrays["prediction"])
    if set(arrays) != set(derived):
        raise ValueError("ISM array schema mismatch")
    for name, expected in derived.items():
        actual = arrays[name]
        if (
            actual.dtype != expected.dtype
            or actual.shape != expected.shape
            or not np.array_equal(actual, expected, equal_nan=True)
        ):
            raise ValueError(f"ISM derived array mismatch: {name}")


def _array_schema(arrays):
    return {
        name: {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "axes": ["source_row"]
            + (["position"] if array.ndim >= 2 else [])
            + (["alternative"] if array.ndim == 3 else []),
        }
        for name, array in arrays.items()
    }


def _provenance(identity, session):
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
            {"prediction": identity["protocol"], "ism": PROTOCOL}
        ),
        "code_sha256": identity["code_sha256"],
        "environment_sha256": digest(session["environment"]),
    }


def save_ism(study, run_id, result):
    """Validate against saved predictions and commit one whole-source NPZ shard."""
    study.checkpoint()
    state = study.status()
    if state["open_job"]["run_id"] != run_id:
        raise ValueError("ISM must be saved in its active run")
    predictions = read_results(study, run_id)
    if not predictions["coverage_complete"]:
        raise ValueError("ISM requires complete original prediction coverage")
    identity = predictions["identity"]
    if identity["protocol"].get("ism") != PROTOCOL:
        raise ValueError("run does not bind the ISM protocol")
    if result["batch_size"] != identity["protocol"]["batch_size"]:
        raise ValueError("ISM batch size differs from run protocol")
    vocabulary = _vocabulary(study, run_id, identity)
    _check_result(
        result, {row["source_row"]: row for row in predictions["rows"]}, vocabulary
    )
    session = state["runs"][run_id]["sessions"][-1]
    if session.get("parity", {}).get("validated") is not True:
        raise ValueError("ISM requires verified session parity")
    study.guard()
    name = "ism-" + uuid.uuid4().hex
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **result["arrays"])
    data = buffer.getvalue()
    study.guard()
    payload = {
        "schema": 1,
        "method": "ism",
        "identity": identity,
        "session": session,
        "protocol": PROTOCOL,
        "batch_size": result["batch_size"],
        "vocabulary": result["vocabulary"],
        "rows": result["rows"],
        "summaries": [
            _summary(row, result["arrays"], index)
            for index, row in enumerate(result["rows"])
        ],
        "arrays": _array_schema(result["arrays"]),
        "layout": LAYOUT,
        "provenance": _provenance(identity, session),
        "quality": QUALITY,
        "npz_file": f"runs/{run_id}/artifacts/{name}.npz",
        "npz_sha256": hashlib.sha256(data).hexdigest(),
    }
    ids = result["arrays"]["source_row"].tolist()
    study.save_artifact(name, payload, ids, kind="ism", npz=data)


def _summary(row, arrays, index):
    raw = arrays["raw_delta"][index]
    mask = arrays["non_wt_mask"][index] & arrays["tttn_preserving"][index]
    denominator = arrays["preserving_denominator"][index]
    preserving = np.full(34, np.nan, dtype=np.float64)
    np.divide(
        np.where(mask, np.abs(raw), 0).sum(axis=1),
        denominator,
        out=preserving,
        where=denominator > 0,
    )
    return {
        "source_row": row["source_row"],
        "relative_delta": arrays["relative_delta"][index].tolist()
        if arrays["relative_valid"][index]
        else None,
        "relative_undefined_reason": row["relative_undefined_reason"],
        "sensitivity": (
            np.abs(raw).sum(axis=1) / arrays["denominator"][index]
        ).tolist(),
        "preserving_sensitivity": [
            float(value) if np.isfinite(value) else None for value in preserving
        ],
        "preserving_undefined_reason": [
            None if value else "no eligible non-WT alternatives"
            for value in denominator
        ],
    }


def read_ism(study, run_id, *, source_ids=None, require_complete=False):
    """Read only committed shards, verify every derived field, and select by ID."""
    predictions = read_results(study, run_id)
    identity = predictions["identity"]
    bindings = {row["source_row"]: row for row in predictions["rows"]}
    shards = study.artifacts(run_id, kind="ism")
    vocabulary = _vocabulary(study, run_id, identity) if shards else None
    rows, array_chunks = [], []
    run = study.status()["runs"][run_id]
    for record, sidecar in shards:
        session = sidecar["session"]
        if (
            sidecar["schema"] != 1
            or sidecar["method"] != "ism"
            or sidecar["identity"] != identity
            or sidecar["protocol"] != PROTOCOL
            or identity["protocol"].get("ism") != PROTOCOL
            or sidecar["layout"] != LAYOUT
            or sidecar["quality"] != QUALITY
            or session not in run["sessions"]
            or session.get("parity", {}).get("validated") is not True
            or sidecar["provenance"] != _provenance(identity, session)
            or sidecar["batch_size"] != identity["protocol"]["batch_size"]
        ):
            raise ValueError("incompatible ISM sidecar identity/protocol/session")
        path = sidecar["npz_file"]
        committed = run["files"].get(path)
        if (
            not committed
            or committed["kind"] != "ism-array"
            or committed["sha256"] != sidecar["npz_sha256"]
        ):
            raise ValueError("ISM NPZ is not a committed matching artifact")
        data = _read_checked(study, path, sidecar["npz_sha256"])
        with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        result = {**sidecar, "arrays": arrays}
        _check_result(result, bindings, vocabulary)
        if sidecar["arrays"] != _array_schema(arrays):
            raise ValueError("ISM array axes/dtype/shape mismatch")
        if sidecar["summaries"] != [
            _summary(row, arrays, index) for index, row in enumerate(sidecar["rows"])
        ]:
            raise ValueError("ISM derived summaries mismatch")
        if arrays["source_row"].tolist() != record["coverage"]:
            raise ValueError("ISM shard coverage mismatch")
        rows.extend(sidecar["rows"])
        array_chunks.append(arrays)
    ids = [row["source_row"] for row in rows]
    if len(set(ids)) != len(ids) or not set(ids) <= set(predictions["coverage"]):
        raise ValueError("overlapping or unexpected ISM source coverage")
    coverage = sorted(ids)
    missing = sorted(set(identity["validation_source_rows"]) - set(ids))
    complete = not missing and not predictions["job_unresolved"]
    if require_complete and not complete:
        raise ValueError("incomplete ISM coverage or unresolved job")
    selected = coverage if source_ids is None else list(source_ids)
    if (
        any(type(value) is not int for value in selected)
        or len(set(selected)) != len(selected)
        or not set(selected) <= set(ids)
    ):
        raise ValueError("selected ISM source IDs are duplicate or unavailable")
    order = [ids.index(value) for value in selected]
    arrays = (
        {
            name: np.concatenate([chunk[name] for chunk in array_chunks], axis=0)[order]
            for name in array_chunks[0]
        }
        if array_chunks
        else {}
    )
    rows = [rows[index] for index in order]
    return {
        "schema": 1,
        "method": "ism",
        "run_id": run_id,
        "identity": identity,
        "complete": complete,
        "coverage_complete": not missing,
        "job_unresolved": predictions["job_unresolved"],
        "coverage": coverage,
        "missing": missing,
        "mutant_count": len(ids) * 102,
        "gpu_executed": not study.status()["fixture"]
        and any(
            sidecar["session"]["environment"]["device"].startswith("cuda")
            for _, sidecar in shards
        ),
        "arrays": arrays,
        "rows": rows,
        "summaries": [_summary(row, arrays, index) for index, row in enumerate(rows)],
    }
