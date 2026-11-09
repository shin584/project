"""Prediction shards and metric reconstruction without torch or model loading."""

import hashlib
import math

import numpy as np
from scipy.stats import pearsonr, spearmanr

from cas12a.data import SEQUENCE_COLUMN, TARGET_COLUMN
from cas12a.nt_data import digest

from .nt_inputs import DNA_SPANS, LAYOUT, sequence_hash


def read_checked(study, path, checksum):
    """Check the immediate read bytes, including failures after a ledger scan."""
    try:
        data = study.store.read(path)
        if hashlib.sha256(data).hexdigest() != checksum:
            raise ValueError(f"artifact hash mismatch: {path}")
        return data
    except (OSError, ValueError):
        study.poisoned = True
        raise


def saved_vocabulary(study, run_id, identity):
    path = f"runs/{run_id}/snapshot/selected_export/vocab.txt"
    data = read_checked(study, path, identity["export_files"]["vocab.txt"])
    tokens = data.decode("utf-8").splitlines()
    vocabulary = {token: index for index, token in enumerate(tokens)}
    if len(tokens) != 4107 or len(vocabulary) != 4107:
        raise ValueError("invalid snapshotted vocabulary")
    return vocabulary


def prediction_rows(source_rows, predictions, tokens):
    result = []
    for row, prediction, token in zip(source_rows, predictions, tokens, strict=True):
        sequence, label, raw = row["sequence"], float(row["target"]), float(prediction)
        if len(sequence) != 34 or set(sequence) - set("ACGT") or sequence[4:7] != "TTT":
            raise ValueError("invalid source sequence/PAM")
        if row["sequence_sha256"] != sequence_hash(sequence):
            raise ValueError("sequence hash mismatch")
        if not math.isfinite(raw) or not math.isfinite(label) or not 0 <= label <= 1:
            raise ValueError("nonfinite prediction or invalid label")
        if (
            str(row["raw"][SEQUENCE_COLUMN]).strip().upper() != sequence
            or float(np.clip(float(row["raw"][TARGET_COLUMN]) / 100, 0, 1)) != label
        ):
            raise ValueError("source identity does not reproduce sequence/label")
        ids, mask = token["input_ids"], token["attention_mask"]
        if (
            len(ids) != 12
            or ids[0] != 3
            or ids[10:] != [1, 1]
            or mask != [1] * 10 + [0, 0]
            or token["dna_spans"] != DNA_SPANS
            or token["special_mask"] != [True] + [False] * 9 + [True, True]
        ):
            raise ValueError("invalid token span/mask record")
        if any(type(value) is not int or not 0 <= value < 4105 for value in ids) or any(
            value in (0, 1, 2, 3) for value in ids[1:10]
        ):
            raise ValueError("invalid actual DNA token IDs")
        clipped = float(np.clip(raw, 0, 1))
        errors = {}
        for name, value in (("raw", raw), ("clipped", clipped)):
            error = value - label
            squared = error * error
            if not math.isfinite(squared):
                raise ValueError("nonfinite prediction error")
            errors[name] = {"signed": error, "absolute": abs(error), "squared": squared}
        result.append(
            {
                "source_row": row["source_row"],
                "source_identity": row["raw"],
                "sequence": sequence,
                "sequence_sha256": sequence_hash(sequence),
                "label": label,
                "raw_prediction": raw,
                "clipped_prediction": clipped,
                "errors": errors,
                "pam_stratum": "TTTT" if sequence[7] == "T" else "TTTV",
                "layout": LAYOUT,
                "output_unit": "fraction",
                "tokens": token,
            }
        )
    return result


def prediction_metrics(rows, field):
    labels = np.asarray([row["label"] for row in rows], dtype=np.float64)
    predictions = np.asarray([row[field] for row in rows], dtype=np.float64)
    reason = None
    if not len(rows):
        reason = "empty population"
    elif len(rows) < 2:
        reason = "fewer than two targets"
    elif np.ptp(labels) == 0:
        reason = "constant labels"
    elif np.ptp(predictions) == 0:
        reason = "constant predictions"
    result = {"count": len(rows)}
    for name, function in (("spearman", spearmanr), ("pearson", pearsonr)):
        value = (
            float(function(labels, predictions).statistic) if reason is None else None
        )
        undefined = reason
        if value is not None and not math.isfinite(value):
            value, undefined = None, "numerically undefined correlation"
        result[name], result[f"{name}_undefined_reason"] = value, undefined
    with np.errstate(over="ignore", invalid="ignore"):
        error = predictions - labels
        for name, value in (
            ("mae", np.mean(np.abs(error)) if len(rows) else None),
            ("mse", np.mean(error**2) if len(rows) else None),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError("nonfinite aggregate error")
            result[name] = float(value) if value is not None else None
            result[f"{name}_undefined_reason"] = (
                None if value is not None else "empty population"
            )
    return result


def read_results(study, run_id, *, require_complete=False):
    artifacts = study.artifacts(run_id)
    identity = study.status()["runs"][run_id]["identity"]
    expected = identity["validation_source_rows"]
    bindings = {row["source_row"]: row for row in identity["validation_rows"]}
    rows = []
    for record, payload in artifacts:
        if payload.get("schema") != 1 or payload.get("identity") != identity:
            raise ValueError("incompatible shard identity/schema")
        shard_rows = payload["rows"]
        if [row["source_row"] for row in shard_rows] != record["coverage"]:
            raise ValueError("shard coverage differs from committed manifest")
        for row in shard_rows:
            binding = {
                "source_row": row["source_row"],
                "sequence_sha256": row["sequence_sha256"],
                "label": row["label"],
                "source_identity_sha256": digest(row["source_identity"]),
            }
            if bindings.get(row["source_row"]) != binding:
                raise ValueError("prediction source identity mismatch")
            source = {
                "source_row": row["source_row"],
                "raw": row["source_identity"],
                "sequence": row["sequence"],
                "sequence_sha256": row["sequence_sha256"],
                "target": row["label"],
            }
            rebuilt = prediction_rows(
                [source], [row["raw_prediction"]], [row["tokens"]]
            )[0]
            if rebuilt != row:
                raise ValueError("prediction derived fields mismatch")
        rows.extend(shard_rows)
    rows.sort(key=lambda row: row["source_row"])
    coverage = [row["source_row"] for row in rows]
    if len(set(coverage)) != len(coverage) or not set(coverage) <= set(expected):
        raise ValueError("overlapping or unexpected prediction coverage")
    coverage_complete = coverage == expected
    open_job = study.status()["open_job"]
    unresolved = bool(open_job and open_job["run_id"] == run_id)
    complete = coverage_complete and not unresolved
    if require_complete and not complete:
        raise ValueError("incomplete validation prediction coverage")
    groups = {
        "all": rows,
        "TTTV": [row for row in rows if row["pam_stratum"] == "TTTV"],
        "TTTT": [row for row in rows if row["pam_stratum"] == "TTTT"],
    }
    return {
        "schema": 1,
        "run_id": run_id,
        "identity": identity,
        "complete": complete,
        "coverage_complete": coverage_complete,
        "job_unresolved": unresolved,
        "coverage": coverage,
        "missing": sorted(set(expected) - set(coverage)),
        "gpu_executed": not study.status()["fixture"]
        and bool(study.status()["runs"][run_id]["sessions"]),
        "rows": rows,
        "metrics": {
            group: {
                view: prediction_metrics(members, f"{view}_prediction")
                for view in ("raw", "clipped")
            }
            for group, members in groups.items()
        },
    }
