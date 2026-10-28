"""#87's approved study boundary; disposable fits and prediction doubles only."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_spcas9_budget import synthetic_ledger as _synthetic_ledger
from test_spcas9_study import inputs as _inputs

from SpCas9.spcas9_data import HEAD_NAMES, output_name
from SpCas9.spcas9_eval import build_predictions
from SpCas9.spcas9_selection import NGG_MAE, NGG_SPEARMAN, ngg_stopping
from SpCas9.spcas9_split import CHECKSUM_FILENAME, file_sha256, write_json
from SpCas9.spcas9_study import InputPins, StudyProtocol
from SpCas9.spcas9_study_backend import FitResult
from SpCas9.spcas9_train import split_record

inputs = _inputs
synthetic_ledger = _synthetic_ledger


class FinalistBackend:
    def __init__(self, recipe="baseline", *, fail_test=False):
        self.recipe = recipe
        self.fail_test = fail_test
        self.calls = []

    def fit(self, training, validation, *, seed, checkpoint_path):
        checkpoint_path.write_text(f"{self.recipe}-{seed}")
        history = {
            "recipe_optimization_loss": [1.0],
            "val_recipe_optimization_loss": [1.0],
            "learning_rate": [0.001],
            NGG_MAE: [0.9 if self.recipe == "candidate" else 1.0],
            NGG_SPEARMAN: [1.0 if self.recipe == "candidate" else -1.0],
        }
        return FitResult(
            history,
            self.predict(checkpoint_path, validation.inputs),
            {"backend": "fixture"},
            ngg_stopping(history),
        )

    def predict(self, checkpoint_path, inputs):
        checkpoint = checkpoint_path.read_text()
        self.calls.append((checkpoint, inputs.copy()))
        candidate = checkpoint.startswith("candidate") and not self.fail_test
        y = (np.arange(len(inputs)) % 25) / 100
        values = y + 0.9 if candidate else 0.24 - y + 1.0
        return {output_name(h): values[:, None] for h in reversed(HEAD_NAMES)}


@pytest.fixture
def completed_study(inputs, synthetic_ledger, tmp_path):
    files = {}
    for role, prefix in (("train", "AAAA"), ("val", "CCCC"), ("test", "TTTT")):
        path = inputs.split_dir / f"{role}.csv"
        frame = pd.read_csv(path)
        frame["sequence"] = [
            prefix + "ACGT" * 5 + ("AGG" if i < 25 else "AAA") + "AAA"
            for i in range(len(frame))
        ]
        frame["protospacer"] = frame["sequence"].str.slice(4, 24)
        # Disjoint protospacers across splits, preserving each target window.
        frame["sequence"] = (
            frame["sequence"].str.slice(0, 4) + prefix + frame["sequence"].str.slice(8)
        )
        frame["protospacer"] = frame["sequence"].str.slice(4, 24)
        for head in HEAD_NAMES:
            frame[head] = np.arange(len(frame)) % 25
        frame.to_csv(path, index=False, lineterminator="\n")
        files[path.name] = file_sha256(path)
    sidecar = inputs.split_dir / CHECKSUM_FILENAME
    record = json.loads(sidecar.read_text())
    record["files"] = files
    write_json(record, sidecar)
    inputs = replace(
        inputs, pins=InputPins(inputs.pins.dataset_sha256, file_sha256(sidecar), files)
    )
    stored_checkpoint = tmp_path / "stored.keras"
    stored_checkpoint.write_text("stored-v2")
    study = StudyProtocol.plan(
        tmp_path / "studies",
        inputs,
        checkpoint_selection="ngg",
        execution_mode="verification",
        synthetic_ledger=synthetic_ledger,
        verification_stored_checkpoint=stored_checkpoint,
    )
    for recipe in ("baseline", "candidate"):
        for seed in (42, 43, 44):
            run = study.plan_run(recipe=recipe, seed=seed, run_id=f"{recipe}-{seed}")
            study.train_validation(run, backend=FinalistBackend(recipe))
    rows = inputs.validate()["val"]
    stored = tmp_path / "stored-validation.csv"
    backend = FinalistBackend()
    predictions = backend.predict(stored_checkpoint, np.zeros((len(rows), 30, 4)))
    build_predictions(
        rows, {h: predictions[output_name(h)] for h in HEAD_NAMES}
    ).to_csv(
        stored,
        index=False,
        lineterminator="\n",
    )
    sidecar = tmp_path / "stored-validation.json"
    write_json(
        {
            "rows": len(rows),
            "file_sha256": file_sha256(stored),
            "checkpoint_sha256": file_sha256(stored_checkpoint),
            "split": split_record(inputs.split_dir),
        },
        sidecar,
    )
    validation = tmp_path / "validation"
    report = study.validation_report(
        validation, stored_predictions=stored, stored_sidecar=sidecar
    )
    assert report["decision"] == "pass"
    return study, validation, stored_checkpoint


def test_freeze_guards_seven_models_and_reopens_verified_results(
    completed_study, tmp_path
):
    study, validation, stored_checkpoint = completed_study
    backend = FinalistBackend()
    with pytest.raises(ValueError, match="verification mode"):
        StudyProtocol.plan(
            tmp_path / "invalid-full",
            execution_mode="full",
            verification_stored_checkpoint=stored_checkpoint,
        )
    with pytest.raises(PermissionError, match="freeze"):
        study.evaluate_test(tmp_path / "early", backend=backend)
    assert backend.calls == []

    # A passing paired mean cannot override a failed seed42 vs stored-v2 gate.
    stored = tmp_path / "stored-validation.csv"
    sidecar = tmp_path / "stored-validation.json"
    original_stored, original_sidecar = stored.read_bytes(), sidecar.read_bytes()
    frame = pd.read_csv(stored)
    for head in HEAD_NAMES:
        frame[f"{head}_pred"] = frame[f"{head}_true"]
    frame.to_csv(stored, index=False, lineterminator="\n")
    record = json.loads(sidecar.read_text())
    record["file_sha256"] = file_sha256(stored)
    write_json(record, sidecar)
    failed_validation = tmp_path / "failed-validation"
    failed = study.validation_report(
        failed_validation, stored_predictions=stored, stored_sidecar=sidecar
    )
    assert failed["comparisons"]["paired_seeds"]["decision"] == "pass"
    assert failed["comparisons"]["representative"]["decision"] == "fail"
    with pytest.raises(PermissionError, match="both validation"):
        study.freeze_selection(failed_validation, stored_checkpoint=stored_checkpoint)
    assert not (study.path / "finalist").exists()
    stored.write_bytes(original_stored)
    sidecar.write_bytes(original_sidecar)

    checkpoint = study.path / "runs" / "candidate-44" / "9hydra_v2.keras"
    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint.unlink()
    incomplete_validation = tmp_path / "incomplete-validation"
    incomplete = study.validation_report(
        incomplete_validation, stored_predictions=stored, stored_sidecar=sidecar
    )
    assert incomplete["decision"] == "incomplete"
    with pytest.raises(PermissionError, match="complete validation"):
        study.freeze_selection(
            incomplete_validation, stored_checkpoint=stored_checkpoint
        )
    checkpoint.write_bytes(checkpoint_bytes)

    frozen = study.freeze_selection(validation, stored_checkpoint=stored_checkpoint)
    assert frozen["representative_seed"] == 42
    assert len(frozen["runs"]) == 6
    with pytest.raises(FileExistsError):
        study.freeze_selection(validation, stored_checkpoint=stored_checkpoint)
    study = StudyProtocol(study.path)

    # Every changed input is rejected before the backend receives Test rows.
    protected = [
        study.path / "finalist" / "freeze.json",
        study.path / "finalist" / "checksum.json",
        study.path / "execution-config.json",
        study.path / "protocol.json",
        study.path / "runs" / "baseline-42" / "9hydra_v2.keras",
        study.path / "runs" / "candidate-44" / "predictions.csv",
        study.path / "runs" / "candidate-42" / "training.json",
        validation / "validation.json",
        stored,
        sidecar,
        stored_checkpoint,
        tmp_path / "source.xlsx",
        tmp_path / "split" / "test.csv",
        tmp_path / "split" / CHECKSUM_FILENAME,
    ]
    for path in protected:
        original = path.read_bytes()
        path.write_bytes(original + b"\nchanged")
        try:
            with pytest.raises((ValueError, PermissionError)):
                study.evaluate_test(tmp_path / "blocked", backend=backend)
            assert backend.calls == [], path
            assert not (tmp_path / "blocked").exists()
        finally:
            path.write_bytes(original)

    freeze = study.path / "finalist" / "freeze.json"
    seal = study.path / "finalist" / "checksum.json"
    original_freeze, original_seal = freeze.read_bytes(), seal.read_bytes()
    for field, value in (
        ("representative_seed", 43),
        ("recipes", {"candidate": "reselected"}),
    ):
        altered = {**frozen, field: value}
        write_json(altered, freeze)
        write_json({"file": "freeze.json", "sha256": file_sha256(freeze)}, seal)
        with pytest.raises(ValueError, match="changed finalist"):
            study.evaluate_test(tmp_path / "blocked", backend=backend)
    freeze.write_bytes(original_freeze)
    seal.write_bytes(original_seal)

    report = study.evaluate_test(tmp_path / "test-results", backend=backend)
    assert report["decision"] == "pass"
    assert report["adoption_eligible"] is True
    assert report["split"] == "test"
    assert {name for name, _ in backend.calls} == {
        f"{recipe}-{seed}"
        for recipe in ("baseline", "candidate")
        for seed in (42, 43, 44)
    } | {"stored-v2"}
    assert len(backend.calls) == 7
    assert all(np.array_equal(backend.calls[0][1], rows) for _, rows in backend.calls)
    assert backend.calls[0][1].shape == (5186, 30, 4)
    assert StudyProtocol(study.path).test_report(tmp_path / "test-results") == report
    with pytest.raises(FileExistsError):
        study.evaluate_test(tmp_path / "test-results", backend=backend)
    assert len(backend.calls) == 7
    failed = study.evaluate_test(
        tmp_path / "failed-test", backend=FinalistBackend(fail_test=True)
    )
    assert failed["decision"] == "fail"
    assert failed["adoption_eligible"] is False
    assert failed["selected_checkpoint"] == "stored v2"
    assert freeze.read_bytes() == original_freeze
    assert seal.read_bytes() == original_seal
    prediction = tmp_path / "test-results" / "candidate-42.csv"
    prediction.write_bytes(prediction.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256|corrupt"):
        study.test_report(tmp_path / "test-results")
