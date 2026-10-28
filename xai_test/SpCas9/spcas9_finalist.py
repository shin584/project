"""Finalist identity and Test evidence behind the StudyProtocol boundary (#87)."""

import json
from io import BytesIO
from pathlib import Path

import pandas as pd

from SpCas9.spcas9_acceptance import GATE_SETTINGS, _build_report, bootstrap_intervals
from SpCas9.spcas9_budget import RUN_ORDER, BudgetLedger, environment_record, read
from SpCas9.spcas9_eval import (
    _read_predictions,
    build_predictions,
    validate_prediction_rows,
)
from SpCas9.spcas9_execution import reserve_destination, source_record
from SpCas9.spcas9_split import file_sha256, write_json
from SpCas9.spcas9_study_inputs import StudyInputs, checked_bytes
from SpCas9.spcas9_train import CHECKPOINT_FILENAME, encode_inputs, split_record


def _sealed(path: Path, name: str) -> dict:
    checksum = read(path / "checksum.json")
    if checksum.get("file") != name:
        raise ValueError("corrupted evidence checksum")
    return json.loads(checked_bytes(path / name, checksum["sha256"]))


def _seal(record: dict, path: Path, name: str) -> None:
    write_json(record, path / name)
    write_json(
        {"file": name, "sha256": file_sha256(path / name)}, path / "checksum.json"
    )


def _validation_snapshot(study, validation: Path, stored_checkpoint: Path) -> dict:
    """Recompute required gates; a saved 'pass' flag alone never grants access."""
    from SpCas9.spcas9_study import _validation_run_evidence

    config = study.execution_configuration()
    if (
        config["representative_seed"] != 42
        or config["run_order"] != RUN_ORDER
        or any(config["acceptance"][k] != v for k, v in GATE_SETTINGS.items())
        or source_record() != config["source"]
        or environment_record() != config["environment"]
    ):
        raise ValueError("changed recipe/seed/source/environment configuration")
    inputs = StudyInputs.from_record(config["protocol"]["inputs"])
    rows = inputs.validate()["val"]
    report = _sealed(validation, "validation.json")
    if report["unavailable_evidence"]:
        raise PermissionError("passing complete validation evidence is required")
    identity = report["identity"]
    expected = {
        "protocol_sha256": file_sha256(study.path / "protocol.json"),
        "execution_config_sha256": file_sha256(study.path / "execution-config.json"),
        "source": config["source"],
        "environment": config["environment"],
        "inputs": config["protocol"]["inputs"],
    }
    if any(identity[key] != value for key, value in expected.items()):
        raise ValueError("validation identity differs from fixed study")
    status = BudgetLedger(study.path).status()
    attempts = status["attempts"]
    if status["blocked"] or len(attempts) != 6:
        raise PermissionError("six completed validation runs are required")
    runs: dict = {"baseline": {}, "candidate": {}}
    frozen_runs = []
    for attempt, planned in zip(attempts, RUN_ORDER):
        if attempt["status"] != "completed" or any(
            attempt[k] != v for k, v in planned.items()
        ):
            raise ValueError("recipe/seed attempt identity mismatch")
        run_id = attempt["run_id"]
        run = study._report_only(run_id)
        recipe, seed = planned["recipe"], planned["seed"]
        if (
            not run["eligible"]
            or any(run["plan"][k] != v for k, v in planned.items())
            or run["training"]["seed"] != seed
            or run["source"] != config["source"]
            or run["inputs"] != config["protocol"]["inputs"]
            or run.get("training_ngg") is None
            or not run["training"]["parity"]["passed"]
        ):
            raise ValueError("incomplete or changed required run evidence")
        path = study.path / "runs" / run_id
        evidence = _validation_run_evidence(run_id, run)
        if report["run_evidence"][recipe][str(seed)] != evidence:
            raise ValueError("changed validation run evidence")
        frame = pd.read_csv(
            BytesIO(
                checked_bytes(path / "predictions.csv", evidence["prediction_sha256"])
            )
        )
        validate_prediction_rows(frame, rows)
        runs[recipe][seed] = frame
        frozen_runs.append(
            {
                **planned,
                "run_id": run_id,
                "checkpoint": str((path / CHECKPOINT_FILENAME).resolve()),
                "checkpoint_sha256": evidence["checkpoint_sha256"],
                "evidence_sha256": file_sha256(path / "evidence.json"),
                "state_sha256": file_sha256(path / "state.json"),
            }
        )
    stored_identity = identity["stored_v2"]
    if stored_identity is None:
        raise PermissionError("stored-v2 validation evidence is required")
    for key in ("input_predictions", "input_sidecar"):
        checked_bytes(Path(stored_identity[key]), stored_identity[f"{key}_sha256"])
    stored_hash = config["acceptance"]["historical_checkpoint_sha256"]
    checked_bytes(stored_checkpoint, stored_hash)
    stored, verified_identity = _read_predictions(
        Path(stored_identity["input_predictions"]),
        Path(stored_identity["input_sidecar"]),
        rows,
        stored_hash,
        split_record(inputs.split_dir),
    )
    if stored_identity != {**verified_identity, "checkpoint_sha256": stored_hash}:
        raise ValueError("changed stored-v2 identity")
    rebuilt = _build_report(runs, stored, split="val", bootstrap=False)
    # Compare JSON representations: persisted seed keys are strings.
    for key in (
        "runs",
        "stored_v2",
        "comparisons",
        "decision",
        "settings",
        "split",
        "candidate_test_eligible",
    ):
        if report[key] != json.loads(json.dumps(rebuilt[key])):
            raise ValueError(
                f"validation report differs from verified predictions: {key}"
            )
    if rebuilt["decision"] != "pass":
        raise PermissionError(
            "both validation comparisons must pass before finalist freeze"
        )
    return {
        "schema_version": 1,
        "issue": 87,
        **expected,
        "recipes": config["recipes"],
        "acceptance": config["acceptance"],
        "ci": config["ci"],
        "representative_seed": 42,
        "execution_mode": config["mode"],
        "runs": frozen_runs,
        "stored_v2": {
            "checkpoint": str(stored_checkpoint.resolve()),
            "checkpoint_sha256": stored_hash,
            "validation": stored_identity,
        },
        "validation": {
            "path": str(validation.resolve()),
            "sha256": file_sha256(validation / "validation.json"),
            "checksum_sha256": file_sha256(validation / "checksum.json"),
        },
        "validation_comparisons": rebuilt["comparisons"],
    }


def freeze_selection(study, validation: Path, stored_checkpoint: Path) -> dict:
    destination = study.path / "finalist"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    with study.evaluation_timing():
        snapshot = _validation_snapshot(
            study, Path(validation), Path(stored_checkpoint)
        )
        destination.mkdir(exist_ok=False)
        _seal(snapshot, destination, "freeze.json")
    return snapshot


def _verify_selection(study) -> dict:
    path = study.path / "finalist"
    if not (path / "freeze.json").is_file() or not (path / "checksum.json").is_file():
        raise PermissionError("valid finalist freeze is required before Test")
    frozen = _sealed(path, "freeze.json")
    # Fail changed bytes before the more expensive metric reconstruction.
    config = study.execution_configuration()
    if frozen["representative_seed"] != 42 or frozen["recipes"] != config["recipes"]:
        raise ValueError("changed finalist recipe/representative seed")
    inputs = frozen["inputs"]
    checked_bytes(Path(inputs["source_path"]), inputs["dataset_sha256"])
    checked_bytes(
        Path(inputs["split_dir"]) / "split_checksum.json", inputs["sidecar_sha256"]
    )
    stored = frozen["stored_v2"]
    checked_bytes(Path(stored["checkpoint"]), stored["checkpoint_sha256"])
    for key in ("input_predictions", "input_sidecar"):
        checked_bytes(
            Path(stored["validation"][key]), stored["validation"][f"{key}_sha256"]
        )
    for name, expected in frozen["inputs"]["files"].items():
        checked_bytes(Path(frozen["inputs"]["split_dir"]) / name, expected)
    validation = Path(frozen["validation"]["path"])
    checked_bytes(validation / "validation.json", frozen["validation"]["sha256"])
    checked_bytes(validation / "checksum.json", frozen["validation"]["checksum_sha256"])
    for run in frozen["runs"]:
        run_path = study.path / "runs" / run["run_id"]
        for name, expected in (
            ("evidence.json", run["evidence_sha256"]),
            ("state.json", run["state_sha256"]),
        ):
            checked_bytes(run_path / name, expected)
        for name, expected in read(run_path / "evidence.json")["files"].items():
            if name != Path(name).name:
                raise ValueError("corrupted run evidence path")
            checked_bytes(run_path / name, expected)
    current = _validation_snapshot(
        study,
        Path(frozen["validation"]["path"]),
        Path(frozen["stored_v2"]["checkpoint"]),
    )
    if frozen != current:
        raise ValueError("changed finalist freeze/model/input/evidence/config identity")
    return frozen


def _test_decision(runs: dict, stored: pd.DataFrame) -> dict:
    report = _build_report(runs, stored, split="test", bootstrap=False)
    report.update(
        issue=87,
        adoption_eligible=report["decision"] == "pass",
        selected_checkpoint="candidate seed42"
        if report["decision"] == "pass"
        else "stored v2",
        test_exposure="historically inspected; not new external confirmation",
        test_access="frozen six checkpoints and stored v2 on identical published Test rows",
        deployment_authorized=False,
    )
    del report["candidate_test_eligible"]
    return report


def evaluate_test(study, destination: Path | None, backend=None) -> dict:
    frozen = _verify_selection(study)
    if destination is None:
        raise ValueError("a fresh Test result destination is required")
    from SpCas9.spcas9_study_backend import KerasV2Backend

    if frozen["execution_mode"] == "verification" and (
        backend is None or isinstance(backend, KerasV2Backend)
    ):
        raise PermissionError("verification mode requires a disposable backend double")
    backend = backend or KerasV2Backend()
    inputs = StudyInputs.from_record(frozen["inputs"])
    destination = Path(destination)
    stored_checkpoint = Path(frozen["stored_v2"]["checkpoint"])
    reserve_destination(
        destination,
        (
            study.path,
            inputs.source_path,
            inputs.split_dir,
            stored_checkpoint,
            Path(frozen["validation"]["path"]),
            Path(frozen["stored_v2"]["validation"]["input_predictions"]),
            Path(frozen["stored_v2"]["validation"]["input_sidecar"]),
        ),
    )
    with study.evaluation_timing():
        rows = inputs.validate()["test"]
        encoded = encode_inputs(rows)
        runs: dict = {"baseline": {}, "candidate": {}}
        files = {}
        models = [(f"{r['recipe']}-{r['seed']}", r) for r in frozen["runs"]] + [
            ("stored-v2", frozen["stored_v2"])
        ]
        for name, model in models:
            checkpoint = Path(model["checkpoint"])
            checked_bytes(checkpoint, model["checkpoint_sha256"])
            # Reuse the same complete rows for each explicitly frozen model.
            from SpCas9.spcas9_study import _predictions

            frame = build_predictions(
                rows,
                _predictions(backend.predict(checkpoint, encoded.copy()), len(rows)),
            )
            checked_bytes(checkpoint, model["checkpoint_sha256"])
            filename = f"{name}.csv"
            frame.to_csv(destination / filename, index=False, lineterminator="\n")
            files[filename] = file_sha256(destination / filename)
            persisted = pd.read_csv(destination / filename)
            validate_prediction_rows(persisted, rows)
            if name == "stored-v2":
                stored = persisted
            else:
                runs[model["recipe"]][model["seed"]] = persisted
        if _verify_selection(study) != frozen:
            raise ValueError("finalist identity changed during Test evaluation")
        report = _test_decision(runs, stored)
    with study.evaluation_timing(bootstrap=True):
        report["bootstrap"] = bootstrap_intervals(runs, frozen["ci"])
    report["execution_mode"] = frozen["execution_mode"]
    report["identity"] = {
        "freeze_sha256": file_sha256(study.path / "finalist" / "freeze.json"),
        "finalist": frozen,
        "predictions": files,
    }
    report = json.loads(json.dumps(report))
    _seal(report, destination, "results.json")
    return report


def test_report(study, destination: Path) -> dict:
    """Reopen results without a backend, verifying gates and all consumed bytes."""
    frozen = _verify_selection(study)
    destination = Path(destination)
    report = _sealed(destination, "results.json")
    identity = report["identity"]
    if identity["finalist"] != frozen or identity["freeze_sha256"] != file_sha256(
        study.path / "finalist" / "freeze.json"
    ):
        raise ValueError("Test result identity differs from finalist freeze")
    required = {f"{r['recipe']}-{r['seed']}.csv" for r in frozen["runs"]} | {
        "stored-v2.csv"
    }
    if set(identity["predictions"]) != required:
        raise ValueError("missing or unexpected Test prediction evidence")
    rows = StudyInputs.from_record(frozen["inputs"]).validate()["test"]
    frames = {}
    for name in required:
        frames[name] = pd.read_csv(
            BytesIO(checked_bytes(destination / name, identity["predictions"][name]))
        )
        validate_prediction_rows(frames[name], rows)
    runs = {
        r: {s: frames[f"{r}-{s}.csv"] for s in (42, 43, 44)}
        for r in ("baseline", "candidate")
    }
    rebuilt = _test_decision(runs, frames["stored-v2.csv"])
    for key in rebuilt.keys() - {"bootstrap"}:
        if report[key] != json.loads(json.dumps(rebuilt[key])):
            raise ValueError(f"corrupted Test report: {key}")
    return report
