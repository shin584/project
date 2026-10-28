"""The approved protocol seam: pinned split bytes to persisted run evidence."""

import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from SpCas9.spcas9_data import HEAD_NAMES, output_name
from SpCas9.spcas9_split import CHECKSUM_FILENAME, SPLIT_FILENAMES, write_json
from SpCas9.spcas9_study import InputPins, StudyInputs, StudyProtocol


class FixtureBackend:
    """Replace training/serialization at the approved external backend seam."""

    def fit(self, training, validation, *, seed, checkpoint_path):
        from SpCas9.spcas9_study_backend import FitResult

        assert training.inputs.shape == (29134, 30, 4)
        assert validation.inputs.shape == (3676, 30, 4)
        assert training.weights[output_name("SpCas9-HF.1")].tolist()[:2] == [0, 1]
        assert validation.labels[output_name("eSpCas9(1.1)")][1] == 0.2
        assert validation.labels[output_name("SpCas9-HF.1")][1] == 0.3
        assert validation.labels[output_name("SpCas9")][0] == -0.075
        checkpoint_path.write_bytes(b"fixture best checkpoint")
        return FitResult(
            history={
                "recipe_optimization_loss": [0.8, 0.4, 0.6],
                "val_recipe_optimization_loss": [0.9, 0.5, 0.7],
                # 8.5 measured heads per row at error .1, with first SpCas9
                # error .775 replacing .1: total SSE = 313.050625.
                "val_reference_masked_mse": [0.9, 313.050625 / 3676, 0.7],
                "learning_rate": [0.001] * 3,
            },
            validation_predictions=self.predict(checkpoint_path, validation.inputs),
            environment={"backend": "fixture", "device": "CPU"},
        )

    def predict(self, checkpoint_path, inputs):
        assert inputs.shape == (3676, 30, 4)  # No Test sequences, including parity.
        assert np.all(inputs[:, 0, 0] == 1)  # strip/uppercase before encoding.
        return {
            output_name(h): np.full((len(inputs), 1), i / 10 - 0.1)
            for i, h in enumerate(reversed(HEAD_NAMES))
        }


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def inputs(tmp_path):
    """Full split sizes, repeated measurements, reordered heads, missing labels."""
    source = tmp_path / "source.xlsx"
    source.write_bytes(b"fixture source, never interpreted or regenerated")
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    rows = {"train": 29134, "val": 3676, "test": 5186}
    for name, count in rows.items():
        index = np.arange(count)
        sequence = {"train": "a" * 30, "val": "acgt" * 7 + "ac", "test": "t" * 30}[name]
        frame = pd.DataFrame(
            {
                "row_index": index + {"train": 0, "val": 29134, "test": 32810}[name],
                "sequence": ["  " + sequence + "  "] * count,
                "protospacer": [sequence[4:24].upper()] * count,
                "in_pam_library": index % 2 == 0,
                "trna_feature": index % 2,
                "use": "Test" if name == "test" else "Training",
                "split": name,
                **{
                    head: np.full(count, float(i * 10))
                    for i, head in enumerate(reversed(HEAD_NAMES))
                },
            }
        )
        frame.loc[index % 2 == 0, "SpCas9-HF.1"] = np.nan
        frame.loc[0, "SpCas9"] = -7.5
        frame.to_csv(
            split_dir / SPLIT_FILENAMES[name], index=False, lineterminator="\n"
        )
    files = {
        filename: digest(split_dir / filename) for filename in SPLIT_FILENAMES.values()
    }
    sidecar = split_dir / CHECKSUM_FILENAME
    write_json(
        {"seed": 42, "dataset_sha256": digest(source), "rows": rows, "files": files},
        sidecar,
    )
    pins = InputPins(digest(source), digest(sidecar), files)
    return StudyInputs(source, split_dir, pins)


def test_planning_validates_pinned_bytes_and_reserves_new_locations(inputs, tmp_path):
    study = StudyProtocol.plan(tmp_path / "studies", inputs, study_id="first")
    run_id = study.plan_run(seed=42, run_id="baseline-42")

    report = study.report_only(run_id)
    assert report["status"] == "planned"
    assert report["eligible"] is False
    assert report["plan"]["seed"] == 42
    assert report["protocol"]["rows"] == {"train": 29134, "val": 3676, "test": 5186}
    with pytest.raises(FileExistsError):
        StudyProtocol.plan(tmp_path / "studies", inputs, study_id="first")
    with pytest.raises(FileExistsError):
        study.plan_run(seed=42, run_id=run_id)


@pytest.mark.parametrize("target", ["source", "sidecar", "train", "val", "test"])
def test_planning_refuses_changed_input_bytes(inputs, tmp_path, target):
    path = (
        inputs.source_path
        if target == "source"
        else inputs.split_dir
        / (CHECKSUM_FILENAME if target == "sidecar" else SPLIT_FILENAMES[target])
    )
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        StudyProtocol.plan(tmp_path / "studies", inputs)
    assert not (tmp_path / "studies").exists()


def test_validation_run_persists_named_measured_predictions_and_reopens_without_backend(
    inputs, tmp_path
):
    study = StudyProtocol.plan(tmp_path / "studies", inputs)
    run_id = study.plan_run(seed=42)
    result = study.train_validation(run_id, backend=FixtureBackend())

    assert result["status"] == "completed"
    assert (
        result["eligible"] is True
    )  # Complete run evidence, not an adoption decision.
    assert result["training"]["stopping"]["best_epoch"] == 2
    assert result["training"]["stopping"]["epochs_run"] == 3
    assert result["training"]["parity"]["split"] == "val"
    assert result["training"]["environment"]["device"] == "CPU"
    assert result["source"]["revision"]
    assert result["source"]["snapshot_sha256"]
    assert result["inputs"]["files"] == inputs.pins.files
    predictions = pd.read_csv(study.path / "runs" / run_id / "predictions.csv")
    assert len(predictions) == 3676  # Repeated sequences are retained as rows.
    assert predictions["sequence"].nunique() == 1
    assert predictions["row_index"].tolist()[:2] == [29134, 29135]
    assert predictions.loc[0, "SpCas9_true"] == -0.075
    assert (
        predictions.loc[0, "SpCas9-HF.1_true"] != predictions.loc[0, "SpCas9-HF.1_true"]
    )
    assert predictions.loc[0, "eSpCas9(1.1)_pred"] == pytest.approx(0.1)
    assert predictions.loc[0, "SpCas9-HF.1_pred"] == pytest.approx(0.2)
    assert predictions.loc[0, "evoCas9_pred"] == -0.1  # No clipping.
    heads = result["validation"]["heads"]
    assert heads["SpCas9-HF.1"]["n"] == 1838
    assert heads["evoCas9"]["mae"] == pytest.approx(0.1)
    assert StudyProtocol(study.path).report_only(run_id) == result
    # Fresh interpreter: report-only can inspect evidence even with framework
    # and training backend imports unavailable.
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.modules.update({name: None for name in "
                "('keras', 'tensorflow', 'SpCas9.spcas9_study_backend')}); "
                "from pathlib import Path; "
                "from SpCas9.spcas9_study import StudyProtocol; "
                "assert StudyProtocol(Path(sys.argv[1])).report_only(sys.argv[2])['eligible']"
            ),
            str(study.path),
            run_id,
        ],
        check=True,
        capture_output=True,
    )
    with pytest.raises(FileExistsError):
        study.train_validation(run_id, backend=FixtureBackend())


def test_report_only_reopens_intact_legacy_validation_evidence(inputs, tmp_path):
    study = StudyProtocol.plan(tmp_path / "studies", inputs)
    run_id = study.plan_run(seed=42)
    result = study.train_validation(run_id, backend=FixtureBackend())
    path = study.path / "runs" / run_id

    # Recreate the pre-#82 persisted schema, keeping all original metric values.
    # These fixture manifests represent an intact historical run, not a request
    # to migrate or rewrite real evidence.
    def legacy_metrics(metrics):
        return {key: metrics[key] for key in ("n", "mae", "mse", "spearman", "pearson")}

    report = result["validation"]
    legacy = {
        key: report[key]
        for key in ("model", "scale", "rows", "empty_strata", "split", "population")
    }
    legacy["heads"] = {
        head: {
            **legacy_metrics(scored),
            "strata": {
                name: {
                    value: legacy_metrics(metrics) for value, metrics in groups.items()
                }
                for name, groups in scored["strata"].items()
            },
        }
        for head, scored in report["heads"].items()
    }
    write_json(legacy, path / "validation.json")
    evidence = json.loads((path / "evidence.json").read_text())
    evidence["files"]["validation.json"] = digest(path / "validation.json")
    write_json(evidence, path / "evidence.json")
    state = json.loads((path / "state.json").read_text())
    state["evidence_sha256"] = digest(path / "evidence.json")
    write_json(state, path / "state.json")
    before = {file: digest(file) for file in path.iterdir()}

    reopened = StudyProtocol(study.path).report_only(run_id)
    assert reopened["eligible"] is True
    assert reopened["validation"] == legacy
    assert {file: digest(file) for file in path.iterdir()} == before


@pytest.mark.parametrize(
    "failure",
    ["nonfinite", "parity", "checkpoint", "history", "wrong_best", "interrupted"],
)
def test_invalid_backend_evidence_never_completes_a_run(inputs, tmp_path, failure):
    class BrokenBackend(FixtureBackend):
        def fit(self, training, validation, *, seed, checkpoint_path):
            if failure == "interrupted":
                checkpoint_path.write_bytes(b"partial")
                raise KeyboardInterrupt("fixture interruption")
            result = super().fit(
                training, validation, seed=seed, checkpoint_path=checkpoint_path
            )
            if failure == "nonfinite":
                result.validation_predictions[output_name("SpCas9")][0] = np.inf
            if failure == "parity":
                result.validation_predictions[output_name("SpCas9")][0] += 1
            if failure == "history":
                result.history["val_reference_masked_mse"][1] = np.nan
            if failure == "wrong_best":
                result.history["val_reference_masked_mse"][1] = 0.01
            return result

        def predict(self, checkpoint_path, inputs):
            outputs = super().predict(checkpoint_path, inputs)
            if failure == "checkpoint":
                checkpoint_path.write_bytes(
                    checkpoint_path.read_bytes() + b"corruption"
                )
            return outputs

    study = StudyProtocol.plan(tmp_path / "studies", inputs)
    run_id = study.plan_run(seed=42)
    with pytest.raises((ValueError, KeyboardInterrupt)):
        study.train_validation(run_id, backend=BrokenBackend())
    report = study.report_only(run_id)
    assert report["status"] == "failed"
    assert report["eligible"] is False
    assert report["error"]
    with pytest.raises(FileExistsError):
        study.train_validation(run_id, backend=FixtureBackend())


@pytest.mark.parametrize(
    "filename",
    [
        "predictions.csv",
        "9hydra_v2.keras",
        "training.json",
        "validation.json",
        "plan.json",
        "evidence.json",
    ],
)
def test_report_only_refuses_corrupted_completed_evidence(inputs, tmp_path, filename):
    study = StudyProtocol.plan(tmp_path / "studies", inputs)
    run_id = study.plan_run(seed=42)
    study.train_validation(run_id, backend=FixtureBackend())
    path = study.path / "runs" / run_id / filename
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="corrupted"):
        study.report_only(run_id)


def test_inputs_are_rechecked_before_backend_use_and_test_prediction_remains_closed(
    inputs, tmp_path
):
    class UnusedBackend:
        def fit(self, *args, **kwargs):
            pytest.fail("backend must not run after failed integrity inspection")

    study = StudyProtocol.plan(tmp_path / "studies", inputs)
    run_id = study.plan_run(seed=42)
    with pytest.raises(PermissionError, match="freeze"):
        study.evaluate_test()
    with pytest.raises(PermissionError, match="validation evidence"):
        study.freeze_selection()
    test_path = inputs.split_dir / "test.csv"
    original = test_path.read_bytes()
    test_path.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        study.train_validation(run_id, backend=UnusedBackend())
    assert study.report_only(run_id)["eligible"] is False
    assert test_path.read_bytes() == original + b"\n"  # Never repair a partition.


def test_metadata_does_not_change_predictor_inputs_or_training_weights(
    inputs, tmp_path
):
    class ObservingBackend(FixtureBackend):
        def fit(self, training, validation, **kwargs):
            self.training, self.validation = training, validation
            return super().fit(training, validation, **kwargs)

    first = StudyProtocol.plan(tmp_path / "studies", inputs)
    first_backend = ObservingBackend()
    first.train_validation(first.plan_run(seed=42), backend=first_backend)
    for filename in ("train.csv", "val.csv", "test.csv"):
        path = inputs.split_dir / filename
        frame = pd.read_csv(path)
        frame["trna_feature"] = 1 - frame["trna_feature"]
        frame["in_pam_library"] = ~frame["in_pam_library"]
        frame.to_csv(path, index=False, lineterminator="\n")
    # A separate fixture with independently pinned changed metadata, not a repair.
    files = {name: digest(inputs.split_dir / name) for name in inputs.pins.files}
    sidecar = inputs.split_dir / CHECKSUM_FILENAME
    record = json.loads(sidecar.read_text())
    record["files"] = files
    write_json(record, sidecar)
    changed_inputs = StudyInputs(
        inputs.source_path,
        inputs.split_dir,
        InputPins(inputs.pins.dataset_sha256, digest(sidecar), files),
    )
    second = StudyProtocol.plan(tmp_path / "studies", changed_inputs)
    second_backend = ObservingBackend()
    second.train_validation(second.plan_run(seed=42), backend=second_backend)
    for left, right in (
        (first_backend.training, second_backend.training),
        (first_backend.validation, second_backend.validation),
    ):
        np.testing.assert_array_equal(left.inputs, right.inputs)
        for head in left.weights:
            np.testing.assert_array_equal(left.weights[head], right.weights[head])
            np.testing.assert_array_equal(left.labels[head], right.labels[head])


def test_exact_required_row_counts_cannot_be_relaxed_by_repinning_the_sidecar(
    inputs, tmp_path
):
    path = inputs.split_dir / "train.csv"
    pd.read_csv(path).iloc[:-1].to_csv(path, index=False, lineterminator="\n")
    files = {**inputs.pins.files, "train.csv": digest(path)}
    sidecar = inputs.split_dir / CHECKSUM_FILENAME
    record = json.loads(sidecar.read_text())
    record["rows"]["train"] -= 1
    record["files"] = files
    write_json(record, sidecar)
    changed = StudyInputs(
        inputs.source_path,
        inputs.split_dir,
        InputPins(inputs.pins.dataset_sha256, digest(sidecar), files),
    )
    with pytest.raises(ValueError, match="29134 rows"):
        StudyProtocol.plan(tmp_path / "studies", changed)


def test_default_pins_accept_the_existing_published_artifacts():
    frames = StudyInputs().validate()
    assert {name: len(frame) for name, frame in frames.items()} == {
        "train": 29134,
        "val": 3676,
        "test": 5186,
    }


@pytest.mark.parametrize(
    "failure", [None, "history", "wrong_mae", "wrong_spearman", "wrong_epoch"]
)
def test_ngg_run_persists_selected_epoch_and_both_histories_or_refuses_invalid_evidence(
    inputs, tmp_path, failure
):
    from SpCas9.spcas9_selection import NGG_MAE, NGG_SPEARMAN, ngg_stopping
    from SpCas9.spcas9_study_backend import FitResult

    # A new pinned fixture; historical input bytes and evidence are untouched.
    path = inputs.split_dir / "val.csv"
    frame = pd.read_csv(path)
    frame["sequence"] = "ACGT" * 6 + "AGGAAA"
    for head in HEAD_NAMES:
        frame[head] = (np.arange(len(frame)) % 5) * 100.0
    frame.to_csv(path, index=False, lineterminator="\n")
    files = {**inputs.pins.files, "val.csv": digest(path)}
    sidecar = inputs.split_dir / CHECKSUM_FILENAME
    record = json.loads(sidecar.read_text())
    record["files"] = files
    write_json(record, sidecar)
    ngg_inputs = StudyInputs(
        inputs.source_path,
        inputs.split_dir,
        InputPins(inputs.pins.dataset_sha256, digest(sidecar), files),
    )

    class NggBackend:
        def fit(self, training, validation, *, seed, checkpoint_path):
            checkpoint_path.write_bytes(b"selected epoch2")
            history = {
                "recipe_optimization_loss": [3.0, 2.0, 1.0],
                "val_recipe_optimization_loss": [3.0, 2.0, 1.0],
                "learning_rate": [0.001] * 3,
                NGG_MAE: [5.0, 5.0, 5.0],
                NGG_SPEARMAN: [-1.0, 1.0, 1.0],
            }
            selection = ngg_stopping(history)
            if failure == "history":
                del history[NGG_SPEARMAN]
            elif failure == "wrong_mae":
                history[NGG_MAE][1] = 4.0
            elif failure == "wrong_spearman":
                history[NGG_SPEARMAN] = [-1.0] * 3
            elif failure == "wrong_epoch":
                selection["best_epoch"] = 3
            return FitResult(
                history,
                self.predict(checkpoint_path, validation.inputs),
                {"backend": "fixture"},
                selection,
            )

        def predict(self, checkpoint_path, inputs):
            return {
                output_name(h): (5.0 + np.arange(len(inputs)) % 5)[:, None]
                for h in HEAD_NAMES
            }

    study = StudyProtocol.plan(
        tmp_path / "studies", ngg_inputs, checkpoint_selection="ngg"
    )
    run_id = study.plan_run(seed=42)
    if failure:
        with pytest.raises(ValueError):
            study.train_validation(run_id, backend=NggBackend())
        assert study.report_only(run_id)["eligible"] is False
    else:
        result = study.train_validation(run_id, backend=NggBackend())
        assert result["training"]["stopping"]["best_epoch"] == 2
        assert result["training"]["selection"]["history"][NGG_MAE] == [5.0, 5.0, 5.0]
        assert result["training"]["selection"]["history"][NGG_SPEARMAN] == [
            -1.0,
            1.0,
            1.0,
        ]
        assert (
            result["protocol"]["hyperparameters"]["early_stopping"][
                "restore_best_weights"
            ]
            is False
        )
        assert StudyProtocol(study.path).report_only(run_id) == result
        with pytest.raises(PermissionError):
            study.evaluate_test()
