"""Thin validation driver and separate model-free inspection/recovery commands.

Run from the repository root with python -m xai_test.cas12a_NT_analysis.analyze.
Actual production validation requires the scientific freeze described by #146.
"""

import argparse
import csv
import io
import json
import logging
import os
import uuid
from contextlib import contextmanager
from functools import partial
from pathlib import Path

from cas12a.nt_data import canonical_bytes

from .nt_artifacts import prediction_rows, read_results
from .nt_ig import compute_ig
from .nt_ig_artifacts import plan_ig, read_ig, save_ig, save_ig_plan
from .nt_inputs import check_export_files, encode_sequences, verify_inputs
from .nt_ism import compute_ism, read_ism, save_ism
from .nt_predict import (
    NativePredictor,
    execution_identity,
    installed_environment,
    source_snapshot,
)
from .nt_run import DirectoryStore, Study


@contextmanager
def _explanation_session(
    inputs,
    study,
    run_id,
    environment,
    *,
    reserved_seconds,
    batch_size,
    device,
    loader,
    predictions,
    before_load=None,
):
    """Share session parity, final identity checks and unresolved-fault handling."""
    study.start(run_id, uuid.uuid4().hex, reserved_seconds, environment)
    try:
        study.checkpoint()
        if before_load is not None:
            before_load()
        environment = installed_environment(device, query_gpu=True)
        predictor = (loader or NativePredictor.load)(
            inputs.export_dir,
            device=device,
            budget_guard=study.guard,
            expected_files=inputs.identity["export_files"],
        )
        predictor.synchronize()
        study.checkpoint()
        parity = predictor.parity(
            [row["sequence"] for row in predictions["rows"][:3]], batch_size=batch_size
        )
        predictor.synchronize()
        study.record_execution(environment, parity)
        yield predictor
        predictor.assert_unchanged()
        predictor.synchronize()
        study.guard()
        check_export_files(inputs.export_dir, inputs.identity["export_files"])
        study.guard()
        del predictor
        study.finish()
    except Exception as error:
        if not study.poisoned:
            try:
                study.record_failure(error)
            except (OSError, ValueError, KeyError, TypeError) as record_error:
                logging.getLogger(__name__).warning(
                    "Could not persist explanation failure: %s", record_error
                )
        raise


def require_drive(root):
    root = Path(root).resolve()
    if not root.is_relative_to(
        Path("/content/drive/MyDrive").resolve()
    ) or not os.path.ismount("/content/drive"):
        raise ValueError(
            "production ledger requires the external mounted Drive study root"
        )


def _prepare_execution(inputs, study, device, batch_size):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch size must be a positive integer")
    if inputs.fixture != study.status()["fixture"]:
        raise ValueError("fixture/production study mismatch")
    if inputs.fixture and device != "cpu":
        raise ValueError("fixture analysis is CPU-only")
    if not inputs.fixture:
        require_drive(study.store.root)
        if not device.startswith("cuda"):
            raise ValueError(
                "production analysis requires an explicitly selected CUDA device"
            )
    # This collects package metadata but does not initialize a CUDA context.
    check_export_files(inputs.export_dir, inputs.identity["export_files"])
    environment = installed_environment(device)
    snapshots = source_snapshot()
    snapshots.update(
        {
            f"selected_export/{name}": (inputs.export_dir / name).read_bytes()
            for name in inputs.identity["export_files"]
            if name != "model.safetensors"
        }
    )
    identity = execution_identity(inputs, environment, snapshots, batch_size)
    return environment, snapshots, identity


def run_validation(
    inputs, study, run_id, *, reserved_seconds, batch_size=32, device="cpu", loader=None
):
    """Verify/reuse coverage, acknowledge start, then load and export validation only."""
    environment, snapshots, identity = _prepare_execution(
        inputs, study, device, batch_size
    )
    study.register(run_id, identity, snapshots=snapshots)
    existing = read_results(study, run_id)
    if existing["complete"]:
        return existing
    study.start(run_id, uuid.uuid4().hex, reserved_seconds, environment)
    try:
        study.checkpoint()
        environment = installed_environment(device, query_gpu=True)
        predictor = (loader or NativePredictor.load)(
            inputs.export_dir,
            device=device,
            budget_guard=study.guard,
            expected_files=inputs.identity["export_files"],
        )
        predictor.synchronize()
        study.checkpoint()
        sequences = [row["sequence"] for row in inputs.rows]
        parity = predictor.parity(sequences[:3], batch_size=batch_size)
        predictor.synchronize()
        study.record_execution(environment, parity)
        covered = set(existing["coverage"])
        remaining = [row for row in inputs.rows if row["source_row"] not in covered]
        for start in range(0, len(remaining), batch_size):
            study.checkpoint()
            source_rows = remaining[start : start + batch_size]
            sequences = [row["sequence"] for row in source_rows]
            raw = predictor.predict(sequences, batch_size=batch_size)
            predictor.synchronize()
            study.checkpoint()
            tokens = encode_sequences(
                predictor.tokenizer, sequences, predictor.embedding_rows
            )
            rows = prediction_rows(source_rows, raw, tokens)
            predictor.assert_unchanged()
            payload = {"schema": 1, "identity": identity, "rows": rows}
            study.save_artifact(
                uuid.uuid4().hex, payload, [row["source_row"] for row in source_rows]
            )
        predictor.assert_unchanged()
        predictor.synchronize()
        study.guard()
        check_export_files(inputs.export_dir, inputs.identity["export_files"])
        study.guard()
        report = read_results(study, run_id)
        if not report["coverage_complete"]:
            raise ValueError("incomplete validation prediction coverage")
        del predictor
        study.finish()
        return read_results(study, run_id, require_complete=True)
    except Exception as error:
        # A known Python exception does not prove the lifetime of every GPU
        # operation. Preserve the open job for explicit termination settlement.
        if not study.poisoned:
            try:
                study.record_failure(error)
            except (OSError, ValueError, KeyError, TypeError) as record_error:
                logging.getLogger(__name__).warning(
                    "Could not persist the failure record: %s", record_error
                )
        raise


def run_ism(
    inputs, study, run_id, *, reserved_seconds, batch_size=32, device="cpu", loader=None
):
    """Calculate missing validation ISM in the existing prediction run and budget."""
    environment, _, identity = _prepare_execution(inputs, study, device, batch_size)
    if run_id not in study.status()["runs"]:
        raise ValueError("ISM requires an existing prediction run")
    study.register(run_id, identity)
    predictions = read_results(study, run_id, require_complete=True)
    existing = read_ism(study, run_id)
    if existing["complete"]:
        return existing
    with _explanation_session(
        inputs,
        study,
        run_id,
        environment,
        reserved_seconds=reserved_seconds,
        batch_size=batch_size,
        device=device,
        loader=loader,
        predictions=predictions,
    ) as predictor:
        covered = set(existing["coverage"])
        for row in predictions["rows"]:
            if row["source_row"] in covered:
                continue
            study.checkpoint()
            result = compute_ism(
                [row],
                partial(predictor.predict, batch_size=batch_size),
                predictor.tokenizer,
                predictor.embedding_rows,
                batch_size=batch_size,
                guard=study.guard,
            )
            predictor.synchronize()
            predictor.assert_unchanged()
            save_ism(study, run_id, result)
        if not read_ism(study, run_id)["coverage_complete"]:
            raise ValueError("incomplete ISM coverage")
    return read_ism(study, run_id, require_complete=True)


def run_ig(
    inputs,
    study,
    run_id,
    *,
    reserved_seconds,
    policy,
    integration_batch_size=16,
    probe_ids=None,
    coverage=None,
    cases=None,
    batch_size=32,
    device="cpu",
    loader=None,
):
    """Resume an immutable IG work plan in the existing prediction run/budget."""
    environment, _, identity = _prepare_execution(inputs, study, device, batch_size)
    if run_id not in study.status()["runs"]:
        raise ValueError("IG requires an existing prediction run")
    study.register(run_id, identity)
    predictions = read_results(study, run_id, require_complete=True)
    plan = plan_ig(
        predictions,
        run_id=run_id,
        policy=policy,
        integration_batch_size=integration_batch_size,
        probe_ids=probe_ids,
        coverage=coverage,
        cases=cases,
    )
    existing = read_ig(study, run_id)
    if existing["plan"] is not None and canonical_bytes(
        existing["plan"]
    ) != canonical_bytes(plan):
        raise ValueError("immutable IG plan cannot be changed")
    if existing["failed"]:
        raise ValueError(
            "saved IG quality failure blocks this plan; preserve diagnostics"
        )
    if existing["complete"]:
        return existing
    with _explanation_session(
        inputs,
        study,
        run_id,
        environment,
        reserved_seconds=reserved_seconds,
        batch_size=batch_size,
        device=device,
        loader=loader,
        predictions=predictions,
        before_load=lambda: save_ig_plan(study, run_id, plan),
    ) as predictor:
        covered = set(existing["coverage"])
        by_id = {row["source_row"]: row for row in predictions["rows"]}
        for source_id in plan["work"]["ids"]:
            if source_id in covered:
                continue
            study.checkpoint()
            result = compute_ig(
                by_id[source_id],
                predictor,
                policy=policy,
                integration_batch_size=integration_batch_size,
            )
            predictor.synchronize()
            predictor.assert_unchanged()
            save_ig(study, run_id, result)
            if not result["accepted"]:
                break
    report = read_ig(study, run_id)
    if report["failed"]:
        raise ValueError(
            "saved IG quality failure blocks completion; inspect diagnostics"
        )
    if not report["complete"]:
        raise ValueError("incomplete IG coverage")
    return report


def export_report(report, directory):
    """Export checked rows and freshly recomputed metrics into a new directory."""
    store = DirectoryStore(directory)
    if store.root.exists() and any(store.root.iterdir()):
        raise ValueError("report export requires a new empty directory")
    rows = report["rows"]
    buffer = io.StringIO(newline="")
    fields = [
        "source_row",
        "source_identity",
        "sequence",
        "sequence_sha256",
        "label",
        "raw_prediction",
        "clipped_prediction",
        "errors",
        "pam_stratum",
        "layout",
        "output_unit",
        "tokens",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: json.dumps(value, sort_keys=True, allow_nan=False)
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            }
        )
    files = {
        "predictions.csv": store.put(
            "predictions.csv", buffer.getvalue().encode("utf-8")
        ),
        "metrics.json": store.put("metrics.json", canonical_bytes(report["metrics"])),
    }
    manifest = {
        key: value for key, value in report.items() if key not in ("rows", "metrics")
    }
    manifest["files"] = files
    store.put("manifest.json", canonical_bytes(manifest))
    return manifest


def _receipt(path, root):
    if Path(path).resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("trusted receipt must be retained outside the study directory")
    value = json.loads(Path(path).read_bytes())
    return value.get("receipt", value.get("unresolved_job_receipt", value))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["initialize", "predict-validation", "inspect", "settle"]
    )
    parser.add_argument("--study-root", required=True)
    parser.add_argument(
        "--receipt",
        help="independently retained latest acknowledgement JSON, outside study root",
    )
    parser.add_argument(
        "--fixture", action="store_true", help="initialize synthetic CPU study"
    )
    parser.add_argument("--run-id")
    parser.add_argument("--source")
    parser.add_argument("--partition")
    parser.add_argument("--selected-export")
    parser.add_argument("--export-receipt")
    parser.add_argument("--reserve-seconds", type=float)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", help="new empty report export directory")
    parser.add_argument("--termination-evidence")
    parser.add_argument("--terminated-utc", type=float)
    parser.add_argument("--upper-seconds", type=float)
    args = parser.parse_args(argv)
    if args.command == "initialize":
        if not args.fixture:
            require_drive(args.study_root)
        study = Study.create(args.study_root, fixture=args.fixture)
        result = {"receipt": study.receipt, "status": study.status()}
    else:
        if not args.receipt:
            parser.error(
                "--receipt is required to reopen; no automatic ledger initialization"
            )
        study = Study.reopen(
            args.study_root, receipt=_receipt(args.receipt, args.study_root)
        )
        try:
            if args.command == "settle":
                if (
                    not args.termination_evidence
                    or args.terminated_utc is None
                    or args.upper_seconds is None
                ):
                    parser.error(
                        "settle requires evidence, termination UTC and conservative upper seconds"
                    )
                study.settle(
                    args.termination_evidence,
                    terminated_utc=args.terminated_utc,
                    upper_seconds=args.upper_seconds,
                )
                result = {"status": study.status()}
            else:
                if not args.run_id:
                    parser.error("--run-id is required")
                if args.command == "predict-validation":
                    if (
                        not all(
                            (
                                args.source,
                                args.partition,
                                args.selected_export,
                                args.export_receipt,
                            )
                        )
                        or args.reserve_seconds is None
                    ):
                        parser.error(
                            "prediction requires source, partition, selected export, export receipt and reservation"
                        )
                    inputs = verify_inputs(
                        args.source,
                        args.partition,
                        args.selected_export,
                        args.export_receipt,
                        fixture=study.status()["fixture"],
                    )
                    result = run_validation(
                        inputs,
                        study,
                        args.run_id,
                        reserved_seconds=args.reserve_seconds,
                        batch_size=args.batch_size,
                        device=args.device,
                    )
                else:
                    result = read_results(study, args.run_id)
                if args.output:
                    # This is a model-free report copy, outside the GPU job.
                    export_report(result, args.output)
            result["receipt"] = study.receipt
        except Exception:
            if not study.poisoned:
                print(
                    json.dumps(
                        {"unresolved_job_receipt": study.receipt}, sort_keys=True
                    ),
                    flush=True,
                )
            raise
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
