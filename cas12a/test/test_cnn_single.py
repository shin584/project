"""The single manifest-driven fit: what it fits, once, and what it saves (#116).

Nothing here asserts a performance number. The fixture library is 11 synthetic
rows and the fits are two epochs, so every number this file touches is
plumbing - the same stance `test_train.py` takes, and for the same reason:
training is stochastic, op determinism is deliberately off, and the metrics
that matter come from a real run's artifacts rather than a test.

What is pinned is what #116 asked to be provable:

- the recipe is `cas12a/train.py`'s, not a second copy of it;
- one model is built and one `fit` is called, and a second of either raises;
- `best_epoch` describes the weights that are actually returned;
- the saved checkpoint reloads to bit-identical weights and reproduces its own
  validation predictions;
- a fixture run is not recorded as confirmed training;
- the five-fold entry point, checkpoint and metrics JSON are untouched.

TensorFlow is imported, so this module skips where it is absent - the TF-free
half of the path is tested in `test_cnn_manifest.py`.
"""

import csv
import json

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from cas12a import cnn_single
from cas12a import train as train_module
from cas12a.cnn_manifest import ManifestMismatch, fitting_set
from cas12a.cnn_single import (
    BATCH_SIZE,
    CHECKPOINT_EVIDENCE_FILENAME,
    CHECKPOINT_FILENAME,
    CONFIGURATION_FILENAME,
    HISTORY_FILENAME,
    PREDICTIONS_FILENAME,
    STATUS_FILENAME,
    ReloadMismatch,
    SingleFit,
    SingleFitViolation,
    comparable_identity,
    compare_forward,
    fit_single,
    forward_tolerance,
    run,
    weight_identities,
)
from cas12a.nt_data import (
    PRODUCTION_COUNTS,
    canonical_bytes,
    digest,
    partition_library,
    read_library,
)
from cas12a.test.test_cnn_manifest import FIXTURE_SPLIT, write_source

#: Two epochs on the fixture library: enough to exercise the plumbing, and
#: deliberately not the recipe's ceiling, which `status.json` must then refuse
#: to call confirmed training.
FIXTURE_EPOCHS = 2


def stub_builder(val_loss):
    """A model whose `fit` reports exactly this per-epoch validation loss.

    The only way to test how a history is read: a real two-epoch fit on 8 rows
    cannot be made to produce a chosen loss curve, let alone a NaN one.
    """

    class Stub:
        def fit(self, inputs, targets, **kwargs):
            history = {"loss": list(val_loss), "val_loss": list(val_loss)}
            return type("History", (), {"history": history})()

        def predict(self, inputs, **kwargs):
            return np.zeros((len(inputs[0]), 1), dtype=np.float32)

    return Stub


@pytest.fixture
def library(tmp_path):
    """The same synthetic 8/2/1 library `test_cnn_manifest.py` builds.

    Rebuilt here rather than imported as a fixture: `from ... import library`
    and a `library` parameter are the same name twice, which ruff reads as a
    redefinition, and the three lines it saves are not worth a `noqa` on every
    test that takes it.
    """
    source = write_source(tmp_path / "source.csv")
    return source, partition_library(read_library(source, FIXTURE_SPLIT), FIXTURE_SPLIT)


@pytest.fixture(scope="module")
def recipe_names():
    return (
        "epochs_max",
        "batch_size",
        "seed",
        "early_stopping",
        "reduce_lr_on_plateau",
        "op_determinism_enabled",
    )


@pytest.fixture
def prepared(library):
    source, manifest = library
    return manifest, fitting_set(source, manifest)


@pytest.fixture
def fitted(prepared):
    """One two-epoch fit on the fixture library, with its record."""
    _, prepared_set = prepared
    return fit_single(prepared_set, epochs=FIXTURE_EPOCHS, verbose=0)


@pytest.fixture
def completed_run(library, tmp_path):
    """A whole run, end to end, into a fresh directory."""
    source, manifest = library
    partition = tmp_path / "partition.json"
    partition.write_bytes(canonical_bytes(manifest))
    return run(
        source,
        partition,
        tmp_path / "runs" / "fixture-run",
        epochs=FIXTURE_EPOCHS,
        verbose=0,
    )


class TestRecipe:
    """One recipe, in one place: #115's defaults are the retune's, unchanged."""

    def test_the_recipe_is_imported_from_the_five_fold_path(self, recipe_names):
        """A copied recipe is two recipes that look like one.

        Seed, batch size, the epoch ceiling and both callbacks come from
        `cas12a.train`, so a later change there reaches this path instead of
        leaving it quietly on the old numbers.
        """
        assert cnn_single.SEED == train_module.SEED == 42
        assert cnn_single.BATCH_SIZE == train_module.BATCH_SIZE == 32
        assert cnn_single.EPOCHS == train_module.EPOCHS == 100
        assert cnn_single.make_callbacks is train_module.make_callbacks
        assert set(train_module.training_configuration()) == set(recipe_names)

    def test_the_accepted_defaults_are_the_ones_the_design_froze(self):
        """#115's listed numbers, read back off the shared configuration."""
        recipe = train_module.training_configuration()

        assert recipe["seed"] == 42
        assert recipe["batch_size"] == 32
        assert recipe["epochs_max"] == 100
        assert recipe["early_stopping"] == {
            "monitor": "val_loss",
            "patience": 10,
            "restore_best_weights": True,
        }
        assert recipe["reduce_lr_on_plateau"] == {
            "monitor": "val_loss",
            "factor": 0.5,
            "patience": 5,
            "min_lr": 1e-5,
        }
        assert cnn_single.LEARNING_RATE == 0.0005

    def test_the_five_fold_path_is_never_called(self, prepared, monkeypatch):
        """#115: `train_5fold` and its CLI are not eligible entry points.

        Replacing both with explosives is the cheapest way to state that this
        path shares the recipe and nothing else - if it quietly fell back to
        the five-fold loop, the fixture run would have five models.
        """
        _, prepared_set = prepared

        def forbidden(*args, **kwargs):
            raise AssertionError("the single-fit path must not fit five folds")

        monkeypatch.setattr(train_module, "train_5fold", forbidden)
        monkeypatch.setattr(train_module, "make_kfold", forbidden)
        monkeypatch.setattr(train_module, "save_artifacts", forbidden)

        _, record = fit_single(prepared_set, epochs=1, verbose=0)

        assert record["model_builds"] == 1


class TestSingleFit:
    """ "Exactly one model, fitted once" as code rather than as a claim."""

    def test_one_model_and_one_fit_are_counted(self, fitted):
        _, record = fitted

        assert record["model_builds"] == 1
        assert record["fit_calls"] == 1

    def test_a_second_model_raises(self):
        guard = SingleFit(build=lambda: object())
        guard.model()

        with pytest.raises(SingleFitViolation, match="builds one model"):
            guard.model()

    def test_a_second_fit_raises(self, prepared):
        """The two guards are separate, so the second one is exercised alone.

        Calling `fit_single` twice would stop at the model guard, which says
        nothing about whether a caller that reuses one model can keep fitting
        it - and that is the shape a resumed or extended fit would take.
        """
        _, prepared_set = prepared
        partitions = prepared_set["partitions"]
        guard = SingleFit()
        model = guard.model()
        guard.fit(
            model, partitions["train"], partitions["validation"], epochs=1, verbose=0
        )

        with pytest.raises(SingleFitViolation, match="fits once"):
            guard.fit(
                model,
                partitions["train"],
                partitions["validation"],
                epochs=1,
                verbose=0,
            )
        assert guard.counts == {"model_builds": 1, "fit_calls": 2}

    def test_validation_rows_are_passed_explicitly_not_split_off_training(
        self, prepared, monkeypatch
    ):
        """`validation_split` would select on a tail of the training array.

        The rows a model is chosen on have to be the manifest's validation
        membership; this captures the kwargs `fit` actually received and
        checks the validation arrays against that membership row for row.

        The recorder arrives through the patched module attribute, which
        `SingleFit` resolves per build; a default argument would have bound the
        real builder at definition time and made the patch a silent no-op.
        """
        manifest, prepared_set = prepared
        seen = {}

        class Recorder:
            def fit(self, inputs, targets, **kwargs):
                seen["inputs"] = inputs
                seen["targets"] = targets
                seen.update(kwargs)
                return type("H", (), {"history": {"val_loss": [1.0, 0.5]}})()

            def predict(self, inputs, **kwargs):
                return np.zeros((len(inputs[0]), 1), dtype=np.float32)

        monkeypatch.setattr(cnn_single, "create_cas12a_model", Recorder)
        fit_single(prepared_set, epochs=FIXTURE_EPOCHS, verbose=0, guard=SingleFit())

        validation = prepared_set["partitions"]["validation"]
        assert "validation_split" not in seen
        assert seen["batch_size"] == BATCH_SIZE
        assert seen["epochs"] == FIXTURE_EPOCHS
        assert np.array_equal(seen["validation_data"][0][0], validation["X_seq"])
        assert np.array_equal(seen["validation_data"][1], validation["y"])
        assert len(validation["source_rows"]) == FIXTURE_SPLIT.counts[1]
        assert validation["source_rows"] == manifest["partitions"]["validation"]

    def test_the_callbacks_are_attached_with_restored_best_weights(self):
        """P2's half that mattered: the scored weights are the best epoch's."""
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        callbacks = {type(c): c for c in cnn_single.make_callbacks()}

        assert set(callbacks) == {EarlyStopping, ReduceLROnPlateau}
        assert callbacks[EarlyStopping].restore_best_weights is True
        assert callbacks[EarlyStopping].patience == 10


class TestRecord:
    """`best_epoch` apart from `epochs_ran`, and the history behind both."""

    def test_the_best_epoch_describes_the_weights_that_survived(self, fitted):
        _, record = fitted

        assert record["epochs_ran"] == FIXTURE_EPOCHS
        assert 1 <= record["best_epoch"] <= record["epochs_ran"]
        assert record["best_val_loss"] <= record["final_val_loss"]
        assert (
            record["best_val_loss"]
            == record["history"]["val_loss"][record["best_epoch"] - 1]
        )

    def test_a_nan_validation_loss_cannot_become_the_reported_best_epoch(
        self, prepared
    ):
        """`np.argmin` would name the NaN epoch; Keras would never restore it.

        Keras decides improvement with `current < best - min_delta`, which is
        False against NaN, so a NaN epoch is never the one `restore_best_weights`
        puts back. Plain `argmin` returns its index anyway - which would report
        an epoch whose weights nothing restored, the exact defect #116 asks to
        be ruled out, arriving by a route a healthy fixture fit never shows.
        """
        _, prepared_set = prepared
        nan = float("nan")

        _, record = fit_single(
            prepared_set,
            epochs=4,
            verbose=0,
            guard=SingleFit(stub_builder([0.9, 0.4, nan, 0.8])),
        )

        assert record["best_epoch"] == 2
        assert record["best_val_loss"] == pytest.approx(0.4)
        assert record["nonfinite_val_loss_epochs"] == [3]

    def test_a_fit_with_no_finite_validation_loss_has_no_best_epoch(self, prepared):
        """Nothing was selected, so there is no epoch to report.

        Reporting epoch 1 here would describe a selection that never happened.
        """
        _, prepared_set = prepared
        nan = float("nan")

        with pytest.raises(ValueError, match="no epoch produced a finite"):
            fit_single(
                prepared_set,
                epochs=2,
                verbose=0,
                guard=SingleFit(stub_builder([nan, nan])),
            )

    def test_epochs_run_is_recorded_separately_from_the_ceiling(self, fitted):
        """Two numbers, because they answer different questions.

        `epochs_max` is what was allowed, `epochs_ran` is what happened, and
        `early_stopped` is whether the two differ - a single "epochs" field
        cannot say whether early stopping fired.
        """
        _, record = fitted

        assert record["epochs_max"] == FIXTURE_EPOCHS
        assert record["early_stopped"] is False

    def test_the_whole_history_is_kept_and_is_json_safe(self, fitted):
        _, record = fitted

        assert {"loss", "val_loss"}.issubset(record["history"])
        assert all(
            len(values) == record["epochs_ran"] for values in record["history"].values()
        )
        digest(record)  # canonical serialization forbids NaN/Infinity

    def test_the_validation_predictions_are_one_per_validation_row(self, fitted):
        _, record = fitted

        assert len(record["validation_predictions"]) == FIXTURE_SPLIT.counts[1]
        assert record["validation_rows"] == FIXTURE_SPLIT.counts[1]
        assert record["train_rows"] == FIXTURE_SPLIT.counts[0]


class TestReloadEvidence:
    """Two different failures, two different judgements (#111)."""

    def test_weight_identity_is_exact_and_the_forward_bound_is_separate(self):
        tolerance = forward_tolerance()

        assert tolerance["precision"] == "float32"
        assert tolerance["rtol"] == pytest.approx(128 * 2.0**-24)
        assert tolerance["atol"] == pytest.approx(tolerance["rtol"] / 10)
        assert "forward reproducibility only" in tolerance["basis"]
        assert "weight identity, which is checked separately" in tolerance["basis"]

    def test_a_renamed_variable_is_not_reported_as_a_missing_one(self, fitted):
        """Keras renames the wrapped LSTM cell's variables on reload.

        A live model reports `bidirectional/forward_lstm/lstm_cell/kernel`;
        the same model loaded back reports `forward_lstm/lstm_cell/kernel`,
        with identical bytes. Keying identity on the name would turn six
        renamed tensors into twelve missing and unexpected ones - noise loud
        enough to hide the one defect this check exists to find. So identity
        is positional content and the name travels beside it as evidence.
        """
        model, _ = fitted
        identities = weight_identities(model)
        renamed = [
            {**entry, "name": f"wrapped/{entry['name']}"} for entry in identities
        ]

        assert cnn_single._identity_mismatches(renamed, identities) == []
        assert any("bidirectional" in entry["name"] for entry in identities)
        assert all("sha256" in entry for entry in identities)

    def test_a_changed_weight_fails_identity_however_small(self, fitted):
        """Exact, so a one-ULP edit is a loading defect and not drift.

        A tolerance that covered this would be a tolerance that could hide a
        tensor the loader re-initialised, which is the conflation #111 split.
        """
        model, _ = fitted
        identities = weight_identities(model)
        variable = model.weights[0]
        variable.assign(variable.numpy() + np.float32(1e-7))

        with pytest.raises(ReloadMismatch, match="reloaded weights differ") as error:
            cnn_single.verify_weight_identity(model, identities, "tampered")

        assert error.value.evidence["kind"] == "weight_identity"
        assert error.value.evidence["mismatches"][0]["reason"] == "content_differs"
        assert error.value.evidence["mismatches"][0]["index"] == 0

    def test_a_forward_difference_beyond_the_bound_is_reported_with_evidence(self):
        with pytest.raises(ReloadMismatch, match="predictions differ") as error:
            compare_forward([0.5, 0.5], [0.5, 0.9], "tampered")

        evidence = error.value.evidence
        assert evidence["kind"] == "forward_reproducibility"
        assert evidence["deviation"]["max_absolute_difference"] == pytest.approx(0.4)
        assert evidence["tolerance"] == forward_tolerance()

    def test_a_shape_change_is_not_comparable_rather_than_close(self):
        with pytest.raises(ReloadMismatch) as error:
            compare_forward([0.5], [0.5, 0.5], "tampered")

        assert error.value.evidence["deviation"]["comparable"] is False

    def test_drift_inside_the_bound_passes_and_is_still_quantified(self):
        report = compare_forward([0.5, 0.25], [0.5, 0.25 + 1e-9], "nearby")

        assert report["deviation"]["max_absolute_difference"] == pytest.approx(1e-9)


class TestRunArtifacts:
    """What a completed run leaves behind, and what it refuses to claim."""

    def test_the_run_directory_holds_the_checkpoint_and_its_evidence(
        self, completed_run
    ):
        run_dir = completed_run["run_dir"]

        assert sorted(path.name for path in run_dir.iterdir()) == sorted(
            [
                CHECKPOINT_FILENAME,
                CONFIGURATION_FILENAME,
                HISTORY_FILENAME,
                CHECKPOINT_EVIDENCE_FILENAME,
                PREDICTIONS_FILENAME,
                STATUS_FILENAME,
            ]
        )

    def test_the_configuration_records_the_inputs_it_is_answerable_to(
        self, completed_run
    ):
        config = json.loads(
            (completed_run["run_dir"] / CONFIGURATION_FILENAME).read_bytes()
        )

        assert config["issue"] == 116 and config["study"] == 115
        assert config["partition"]["reused_without_resplitting"] is True
        assert config["partition"]["counts"] == {
            "train": FIXTURE_SPLIT.counts[0],
            "validation": FIXTURE_SPLIT.counts[1],
            "test": FIXTURE_SPLIT.counts[2],
        }
        assert config["membership"]["held_out_used_for_fitting_or_selection"] is False
        assert config["training"]["folds"] == 1
        assert config["library_versions"]["tensorflow"]
        assert config["cpu_settings"]["visible_devices"]

    def test_the_checkpoint_reloads_to_identical_weights_and_predictions(
        self, completed_run
    ):
        """The save/reload evidence, and the file it describes, both checked.

        Re-hashing the `.keras` here is the half the run cannot do for itself:
        it proves the digest in `checkpoint.json` is of the file that is
        actually on disk beside it.
        """
        import hashlib

        import tensorflow as tf

        run_dir = completed_run["run_dir"]
        evidence = json.loads((run_dir / CHECKPOINT_EVIDENCE_FILENAME).read_bytes())
        checkpoint = run_dir / CHECKPOINT_FILENAME

        with checkpoint.open("rb") as handle:
            assert (
                hashlib.file_digest(handle, "sha256").hexdigest() == evidence["sha256"]
            )
        assert evidence["weight_identity_verified"] is True
        assert evidence["weight_tensors"] == len(evidence["weight_identities"])
        assert evidence["forward_reproducibility"]["deviation"]["comparable"] is True

        reloaded = tf.keras.models.load_model(checkpoint)
        assert [
            comparable_identity(entry) for entry in weight_identities(reloaded)
        ] == [comparable_identity(entry) for entry in evidence["weight_identities"]]
        prediction = reloaded.predict(
            [np.zeros((2, 34, 4), np.float32), np.zeros((2, 354), np.float32)],
            verbose=0,
        )
        assert prediction.shape == (2, 1)
        assert np.all((prediction >= 0) & (prediction <= 1))

    def test_the_predictions_csv_joins_on_source_row_not_on_offset(
        self, completed_run, library
    ):
        _, manifest = library
        rows = list(
            csv.DictReader(
                (completed_run["run_dir"] / PREDICTIONS_FILENAME)
                .read_text(encoding="utf-8")
                .splitlines()
            )
        )

        assert [int(row["source_row"]) for row in rows] == manifest["partitions"][
            "validation"
        ]
        assert all(len(row["sequence_sha256"]) == 64 for row in rows)
        for row in rows:
            assert float(row["absolute_difference"]) == pytest.approx(
                abs(float(row["reload_prediction"]) - float(row["prediction"]))
            )
            assert 0.0 <= float(row["target"]) <= 1.0

    def test_a_fixture_run_is_not_recorded_as_confirmed_training(self, completed_run):
        """The requirement in #116's last acceptance line, held by the artifact.

        An 11-row two-epoch rehearsal produces every file a real run does, so
        the file has to be the thing that says it is not one.
        """
        report = json.loads((completed_run["run_dir"] / STATUS_FILENAME).read_bytes())

        assert report["run_kind"] == "fixture"
        assert report["records_confirmed_training"] is False
        assert any(
            "fixture manifest" in reason for reason in report["not_confirmed_because"]
        )
        assert any(
            str(FIXTURE_EPOCHS) in reason for reason in report["not_confirmed_because"]
        )
        assert "#117 performs the confirmed" in report["note"]
        assert all(report["stages"].values())

    def test_the_status_artifact_digests_match_the_files_it_names(self, completed_run):
        run_dir = completed_run["run_dir"]
        report = json.loads((run_dir / STATUS_FILENAME).read_bytes())

        for name in (
            CONFIGURATION_FILENAME,
            HISTORY_FILENAME,
            CHECKPOINT_EVIDENCE_FILENAME,
        ):
            assert report["artifacts"][name] == digest(
                json.loads((run_dir / name).read_bytes())
            )

    def test_a_run_never_lands_on_top_of_another(self, library, tmp_path):
        source, manifest = library
        partition = tmp_path / "partition.json"
        partition.write_bytes(canonical_bytes(manifest))
        existing = tmp_path / "taken"
        existing.mkdir()

        with pytest.raises(FileExistsError, match="already exists"):
            run(source, partition, existing, epochs=1, verbose=0)

    def test_a_manifest_mismatch_stops_the_run_before_anything_is_fitted(
        self, library, tmp_path, monkeypatch
    ):
        """Verification is a gate, not a note written beside a trained model."""
        source, manifest = library
        partition = tmp_path / "partition.json"
        partition.write_bytes(
            canonical_bytes({**json.loads(canonical_bytes(manifest)), "seed": 7})
        )

        def forbidden(*args, **kwargs):
            raise AssertionError("no model may be built before verification passes")

        monkeypatch.setattr(cnn_single, "create_cas12a_model", forbidden)
        run_dir = tmp_path / "runs" / "never"

        with pytest.raises(ManifestMismatch, match="pinned NT split"):
            run(source, partition, run_dir, epochs=1, verbose=0)
        assert not run_dir.exists()

    def test_the_five_fold_artifacts_are_left_alone(self, completed_run):
        """#115 preserves the historical checkpoint and metrics.

        The run directory is under `artifacts/cnn_manifest/<run_id>/`, so the
        five-fold `.keras` and `cas12a_5fold_metrics.json` cannot be reached
        by a default-path run, let alone overwritten.
        """
        assert cnn_single.DEFAULT_OUTPUT_ROOT.name == "cnn_manifest", (
            "runs must not land in artifacts/ itself"
        )
        assert cnn_single.DEFAULT_OUTPUT_ROOT.parent == train_module.DEFAULT_OUTPUT_DIR
        assert train_module.METRICS_FILENAME not in [
            path.name for path in completed_run["run_dir"].iterdir()
        ]


def test_the_production_default_points_at_the_nt_run_s_manifest():
    """The CLI's defaults are the study's inputs, not an example path."""
    assert cnn_single.DEFAULT_PARTITION.name == "partition.json"
    assert cnn_single.DEFAULT_PARTITION.parent.name == "20261101-nt500m-full"
    assert cnn_single.DEFAULT_SOURCE == train_module.DEFAULT_DATASET
    assert PRODUCTION_COUNTS == (12000, 1500, 1500)


def test_the_cli_parses_its_overrides():
    args = cnn_single.parse_args(
        ["--epochs", "3", "--run-id", "smoke", "--output-root", "out"]
    )

    assert (args.epochs, args.run_id) == (3, "smoke")
    assert args.output_root.name == "out"
    assert args.partition == cnn_single.DEFAULT_PARTITION
