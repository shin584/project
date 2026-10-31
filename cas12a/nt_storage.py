"""Hash-verified NT artifact publication, independent of the training runtime."""

import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from uuid import uuid4

from cas12a.nt_data import canonical_bytes, digest

EXPORT_EVIDENCE = (
    "configuration.json",
    "data.json",
    "partition.json",
    "execution.json",
    "preflight.json",
    "tokenization.json",
    "training.json",
    "history.json",
    "export.json",
)


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def files(directory):
    return {
        path.relative_to(directory).as_posix(): file_hash(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def write_verified(path, value):
    """Replace small evidence atomically after verifying its staged bytes."""
    encoded = canonical_bytes(value)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.incomplete")
    with temporary.open("xb") as handle:
        handle.write(encoded)
    if temporary.read_bytes() != encoded:
        raise OSError(f"artifact verification failed: {temporary}")
    # Windows indexers can briefly hold a closed evidence file without delete
    # sharing. Retry that transient rename lock; never fall back to partial writes.
    for attempt in range(6):
        try:
            temporary.replace(path)
            break
        except PermissionError as error:
            if os.name != "nt" or error.winerror not in (5, 32, 33) or attempt == 5:
                raise
            time.sleep(0.01 * 2**attempt)
    if path.read_bytes() != encoded:
        raise OSError(f"artifact verification failed: {path}")


def record_status(run, status):
    journal = status_journal(run, status)
    # Local evidence survives a failed Drive write. Status is the commit marker.
    for directory in (run.local_dir, run.durable_dir):
        write_verified(directory / "lifecycle.json", journal)
        write_verified(directory / "status.json", status)


def status_journal(run, status):
    journal_path = run.local_dir / "lifecycle.json"
    if not journal_path.exists():
        journal_path = run.durable_dir / "lifecycle.json"
    journal = (
        json.loads(journal_path.read_bytes())
        if journal_path.exists()
        else {
            "run_id": run.run_id,
            "events": [],
        }
    )
    if (
        not isinstance(journal, dict)
        or journal.get("run_id") != run.run_id
        or not isinstance(journal.get("events"), list)
    ):
        raise ValueError("lifecycle history identity or structure is invalid")
    journal["events"].append(
        {
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "status": dict(status),
        }
    )
    return journal


def record_failure(run, status, error, *, directories=None):
    """Try both stores independently and retain the primary exception."""
    try:
        journal = status_journal(run, status)
    except (OSError, ValueError, TypeError, KeyError) as storage_error:
        error.add_note(f"Could not read lifecycle history: {storage_error}")
        journal = {
            "run_id": run.run_id,
            "events": [
                {
                    "recorded_utc": datetime.now(timezone.utc).isoformat(),
                    "status": dict(status),
                }
            ],
        }
    for directory in (
        directories if directories is not None else (run.local_dir, run.durable_dir)
    ):
        for name, value in (("lifecycle.json", journal), ("status.json", status)):
            try:
                write_verified(directory / name, value)
            except (OSError, ValueError, TypeError, KeyError) as storage_error:
                error.add_note(
                    f"Could not record {name} in {directory}: {storage_error}"
                )


def invalidate_receipt(run, filename, error):
    for directory in (run.local_dir, run.durable_dir):
        path = directory / filename
        try:
            if path.exists():
                receipt = json.loads(path.read_bytes())
                receipt["validated"] = False
                write_verified(path, receipt)
        except (OSError, ValueError, TypeError, KeyError) as storage_error:
            error.add_note(
                f"Could not invalidate {filename} in {directory}: {storage_error}"
            )


def copy_verified(source, destination, expected):
    """Reserve a fresh bundle; publish its final name only after all hashes match."""
    temporary = destination.with_name(f".{destination.name}.incomplete")
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"artifact destination already exists: {destination}")
    if files(source) != expected:
        raise ValueError("source artifact identity changed")
    temporary.mkdir()
    for name in sorted(expected):
        target = temporary / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    if files(source) != expected or files(temporary) != expected:
        raise ValueError("copied artifact identity differs from source")
    # The run owns this destination; mkdir above reserves concurrent publishers.
    temporary.rename(destination)
    if files(destination) != expected:
        raise ValueError("published artifact identity differs from source")


def export_files(report):
    return {f"selected_export/{name}": value for name, value in report["files"].items()}


def publish_export(run, report):
    documents = {
        name: json.loads((run.local_dir / name).read_bytes())
        for name in EXPORT_EVIDENCE
    }
    configuration = documents["configuration.json"]
    execution = documents["execution.json"]
    history = documents["history.json"]
    if (
        digest(configuration) != run.status["configuration_sha256"]
        or digest(documents["data.json"]) != configuration["data_sha256"]
        or digest(documents["partition.json"])
        != configuration["partition_manifest_sha256"]
        or digest(execution) != report["execution_sha256"]
        or digest(configuration) != execution["configuration_sha256"]
        or digest(documents["preflight.json"]) != execution["preflight_sha256"]
        or digest(documents["tokenization.json"]) != execution["tokenization_sha256"]
        or digest(documents["training.json"]) != history["training_sha256"]
        or digest(history) != report["history_sha256"]
        or digest(documents["export.json"]) != digest(report)
    ):
        raise ValueError("durable export metadata identity changed")
    copy_verified(
        run.local_dir / "selected_export",
        run.durable_dir / "selected_export",
        report["files"],
    )
    expected = export_files(report)
    for name in EXPORT_EVIDENCE:
        expected[name] = file_hash(run.local_dir / name)
    for name, expected_hash in expected.items():
        if file_hash(run.durable_dir / name) != expected_hash:
            raise ValueError(f"durable evidence identity differs: {name}")
    return {
        "validated": True,
        "run_id": run.run_id,
        "run_kind": run.status["run_kind"],
        "selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
        "selected_artifact_sha256": report["selected_artifact_sha256"],
        "export_sha256": digest(report),
        "files": expected,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
    }


def verify_export(run, report, receipt):
    expected_names = set(EXPORT_EVIDENCE) | set(export_files(report))
    if (
        not receipt["validated"]
        or receipt["run_id"] != run.run_id
        or receipt["run_kind"] != run.status["run_kind"]
        or receipt["export_sha256"] != digest(report)
        or receipt["selected_checkpoint_sha256"] != report["selected_checkpoint_sha256"]
        or receipt["selected_artifact_sha256"] != report["selected_artifact_sha256"]
        or set(receipt["files"]) != expected_names
        or files(run.durable_dir / "selected_export") != report["files"]
    ):
        raise ValueError("durable export identity changed or incomplete")
    for name, expected in receipt["files"].items():
        if file_hash(run.durable_dir / name) != expected:
            raise ValueError(f"durable artifact identity changed: {name}")


def evaluation_identity(run, export_receipt, configuration):
    return {
        "run_id": run.run_id,
        "run_kind": run.status["run_kind"],
        "selected_checkpoint_sha256": export_receipt["selected_checkpoint_sha256"],
        "selected_artifact_sha256": export_receipt["selected_artifact_sha256"],
        "partition_manifest_sha256": configuration["partition_manifest_sha256"],
    }


def evaluation_files(run, directory, export_receipt, configuration):
    """Validate the later evaluator's signed-off bundle, not its scientific result."""
    report = json.loads((directory / "evaluation.json").read_bytes())
    identity = evaluation_identity(run, export_receipt, configuration)
    if (
        not isinstance(report, dict)
        or report.get("validated") is not True
        or any(report.get(key) != value for key, value in identity.items())
        or set(report["files"]) != {"metrics.json", "predictions.csv"}
    ):
        raise ValueError("final evaluation identity changed or not validated")
    expected = {
        **report["files"],
        "evaluation.json": file_hash(directory / "evaluation.json"),
    }
    if files(directory) != expected:
        raise ValueError("final evaluation artifact identity changed or incomplete")
    return expected


def publish_evaluation(run, export_receipt, configuration):
    source = run.local_dir / "final_evaluation"
    expected = evaluation_files(run, source, export_receipt, configuration)
    copy_verified(source, run.durable_dir / "final_evaluation", expected)
    return {
        **evaluation_identity(run, export_receipt, configuration),
        "validated": True,
        "files": expected,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
    }


def verify_evaluation(run, receipt, export_receipt, configuration):
    expected = evaluation_files(
        run, run.durable_dir / "final_evaluation", export_receipt, configuration
    )
    identity = evaluation_identity(run, export_receipt, configuration)
    if (
        receipt["validated"] is not True
        or any(receipt.get(key) != value for key, value in identity.items())
        or receipt["files"] != expected
    ):
        raise ValueError("durable evaluation receipt identity changed")
