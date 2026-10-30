"""Export contract tests at the agreed public save/reload seam (#96)."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import keras
import numpy as np
import pytest

from SpCas9.spcas9_export import export_checkpoint, main

# Independent consumer order, deliberately HF1 before eSpCas9.
LAYERS = (
    "Output_SpCas9",
    "Output_SpCas9_NG",
    "Output_VRQR_variant",
    "Output_xCas",
    "Output_Sniper_Cas9",
    "Output_SpCas9_HF_1",
    "Output_eSpCas9_1_1_",
    "Output_HypaCas9",
    "Output_evoCas9",
)
VALUES = (0.0, -0.25, 1.5, 0.3, 0.4, 0.56, 0.67, 0.8, 0.9)
WINDOWS = ("ACGT" + "ACGT" * 5 + "AGG" + "TCA", "TGCA" + "TGCA" * 5 + "TGG" + "ATC")


def controlled_source(path, *, overrides=None, missing=None, input_kwargs=None):
    inputs = keras.Input(
        **(
            input_kwargs
            or {
                "shape": (30, 4),
                "dtype": "float32",
                "name": "In_Seq",
            }
        )
    )
    shared = keras.layers.Flatten()(inputs)
    outputs = {}
    for index in reversed(range(9)):
        name = LAYERS[index]
        if name == missing:
            continue
        options = {
            "units": 1,
            "name": name,
            "dtype": "float32",
            "kernel_initializer": "zeros",
            "bias_initializer": keras.initializers.Constant(VALUES[index]),
        }
        options.update((overrides or {}).get(name, {}))
        outputs[name] = keras.layers.Dense(**options)(shared)
    model = keras.Model(inputs, outputs)
    model.save(path)
    return model


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_export_reloads_as_consumer_ordered_list_preserving_linear_values(tmp_path):
    source = tmp_path / "source.keras"
    controlled_source(source)
    before = digest(source)
    destination = tmp_path / "export"

    export_checkpoint(source, destination)

    reloaded = keras.models.load_model(destination / "model.keras", compile=False)
    for windows in (WINDOWS[:1], WINDOWS):
        encoded = np.eye(4, dtype=np.float32)[
            [["ACGT".index(base) for base in window] for window in windows]
        ]
        predictions = reloaded.predict(encoded, verbose=0)
        assert type(predictions) is list
        assert len(predictions) == 9
        for prediction, value in zip(predictions, VALUES, strict=True):
            assert prediction.shape == (len(windows), 1)
            assert prediction.dtype == np.float32
            np.testing.assert_allclose(prediction, value, rtol=1e-5, atol=1e-6)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["validation"]["status"] == "passed"
    assert manifest["source"]["sha256"] == before == digest(source)
    assert manifest["artifact"]["sha256"] == digest(destination / "model.keras")
    assert [head["layer"] for head in manifest["heads"]] == list(LAYERS)
    assert manifest["validation"]["rtol"] == 1e-5
    assert manifest["validation"]["atol"] == 1e-6
    assert [batch["n"] for batch in manifest["validation"]["batches"]] == [1, 3]
    assert set(manifest["versions"]) == {"python", "tensorflow", "keras", "numpy"}
    assert len(manifest["verification_code"]["sha256"]) == 64


@pytest.mark.parametrize(
    "head_options,missing,reason",
    [
        ({}, LAYERS[5], "Output_SpCas9_HF_1.*found 0"),
        ({LAYERS[5]: {"units": 2}}, None, "Output_SpCas9_HF_1.*float32"),
        ({LAYERS[5]: {"dtype": "float64"}}, None, "Output_SpCas9_HF_1.*float32"),
        ({LAYERS[5]: {"activation": "sigmoid"}}, None, "Output_SpCas9_HF_1.*linear"),
        (
            {
                LAYERS[5]: {
                    "bias_initializer": keras.initializers.Constant(float("nan"))
                }
            },
            None,
            "Output_SpCas9_HF_1.*nonfinite",
        ),
        (
            {
                LAYERS[5]: {
                    "bias_initializer": keras.initializers.Constant(float("inf"))
                }
            },
            None,
            "Output_SpCas9_HF_1.*nonfinite",
        ),
    ],
)
def test_invalid_head_contract_never_publishes_success(
    tmp_path, head_options, missing, reason
):
    source = tmp_path / "source.keras"
    controlled_source(source, overrides=head_options, missing=missing)
    before = digest(source)
    destination = tmp_path / "export"
    with pytest.raises(ValueError, match=reason):
        export_checkpoint(source, destination)
    assert digest(source) == before
    assert not (destination / "manifest.json").exists()


@pytest.mark.parametrize(
    "input_kwargs",
    [
        {"shape": (29, 4), "name": "In_Seq"},
        {"shape": (30, 4), "name": "wrong_input"},
        {"shape": (30, 4), "name": "In_Seq", "dtype": "float64"},
    ],
)
def test_wrong_input_contract_is_rejected(tmp_path, input_kwargs):
    source = tmp_path / "source.keras"
    controlled_source(source, input_kwargs=input_kwargs)
    with pytest.raises(ValueError, match="Input contract"):
        export_checkpoint(source, tmp_path / "export")
    assert not (tmp_path / "export/manifest.json").exists()


def test_ambiguous_nested_layer_identity_is_rejected(tmp_path):
    nested_input = keras.Input((30, 4))
    nested_output = keras.layers.Dense(1, name=LAYERS[0])(
        keras.layers.Flatten()(nested_input)
    )
    nested = keras.Model(nested_input, nested_output, name="nested")
    inputs = keras.Input((30, 4), name="In_Seq")
    shared = nested(inputs)
    outputs = {name: keras.layers.Dense(1, name=name)(shared) for name in LAYERS}
    source = tmp_path / "source.keras"
    keras.Model(inputs, outputs).save(source)
    before = digest(source)
    with pytest.raises(ValueError, match="Output_SpCas9.*found 2"):
        export_checkpoint(source, tmp_path / "export")
    assert digest(source) == before
    assert not (tmp_path / "export/manifest.json").exists()


def test_existing_destination_and_source_aliases_are_preserved(tmp_path):
    source = tmp_path / "source.keras"
    controlled_source(source)
    before = digest(source)
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "model.keras").write_bytes(b"previous artifact")
    (destination / "manifest.json").write_bytes(b"previous evidence")
    alias = tmp_path / "source_alias.keras"
    os.link(source, alias)
    for target, exception in (
        (source, ValueError),
        (alias, ValueError),
        (destination, FileExistsError),
    ):
        with pytest.raises(exception):
            export_checkpoint(source, target)
    assert digest(source) == digest(alias) == before
    assert (destination / "model.keras").read_bytes() == b"previous artifact"
    assert (destination / "manifest.json").read_bytes() == b"previous evidence"


@pytest.mark.parametrize(
    "failure", ["save", "reload", "parity", "container", "interrupt", "artifact_drift"]
)
def test_failed_or_interrupted_io_has_no_success_marker(tmp_path, monkeypatch, failure):
    source = tmp_path / "source.keras"
    controlled_source(source)
    before = digest(source)
    destination = tmp_path / "export"
    save = keras.Model.save
    load = keras.models.load_model

    if failure in {"save", "interrupt"}:

        def failed_save(model, filepath, *args, **kwargs):
            Path(filepath).write_bytes(b"incomplete")
            if failure == "interrupt":
                raise KeyboardInterrupt("interrupted save")
            raise OSError("save failed")

        monkeypatch.setattr(keras.Model, "save", failed_save)
    else:

        def failed_reload(filepath, *args, **kwargs):
            if Path(filepath).resolve() == source:
                return load(filepath, *args, **kwargs)
            if failure == "reload":
                raise OSError("reload failed")
            model = load(filepath, *args, **kwargs)
            if failure == "artifact_drift":
                Path(filepath).write_bytes(b"artifact changed after loading")
                return model
            predict = model.predict

            def faulty_predictions(*predict_args, **predict_kwargs):
                result = predict(*predict_args, **predict_kwargs)
                if failure == "container":
                    return tuple(result)
                # HF1/eSpCas9 swap must fail even though both shapes are valid.
                result[5], result[6] = result[6], result[5]
                return result

            model.predict = faulty_predictions
            return model

        monkeypatch.setattr(keras.models, "load_model", failed_reload)

    exception = KeyboardInterrupt if failure == "interrupt" else (OSError, ValueError)
    with pytest.raises(exception):
        export_checkpoint(source, destination)
    assert digest(source) == before
    assert not (destination / "manifest.json").exists()
    assert not list(tmp_path.glob(".export.staging-*"))
    # The public operation remains usable with a fresh destination after failure.
    monkeypatch.setattr(keras.Model, "save", save)
    monkeypatch.setattr(keras.models, "load_model", load)
    export_checkpoint(source, tmp_path / "retry")
    assert (tmp_path / "retry/manifest.json").exists()


def test_source_identity_change_during_export_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "source.keras"
    controlled_source(source)
    save = keras.Model.save

    def concurrent_source_change(model, filepath, *args, **kwargs):
        result = save(model, filepath, *args, **kwargs)
        source.write_bytes(source.read_bytes() + b"external change")
        return result

    monkeypatch.setattr(keras.Model, "save", concurrent_source_change)
    with pytest.raises(ValueError, match="Source identity changed"):
        export_checkpoint(source, tmp_path / "export")
    assert source.read_bytes().endswith(b"external change")
    assert not (tmp_path / "export/manifest.json").exists()


def test_input_dependent_learned_weights_survive_export(tmp_path):
    source = tmp_path / "source.keras"
    kernel = np.zeros((120, 1), dtype=np.float32)
    kernel[0, 0] = 2.0  # First base is A in the first fixture, T in the second.
    original = controlled_source(
        source,
        overrides={
            LAYERS[0]: {"kernel_initializer": keras.initializers.Constant(kernel)},
        },
    )
    destination = tmp_path / "export"
    export_checkpoint(source, destination)
    reloaded = keras.models.load_model(destination / "model.keras", compile=False)
    encoded = np.eye(4, dtype=np.float32)[
        [["ACGT".index(base) for base in window] for window in WINDOWS]
    ]
    np.testing.assert_array_equal(
        reloaded.predict(encoded, verbose=0)[0], [[2.0], [0.0]]
    )
    for name in LAYERS:
        for before, after in zip(
            original.get_layer(name).get_weights(),
            reloaded.get_layer(name).get_weights(),
            strict=True,
        ):
            np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize(
    "failure", ["artifact_corruption", "manifest_write", "publish_interrupt"]
)
def test_publication_failure_leaves_only_incomplete_destination(
    tmp_path, monkeypatch, failure
):
    source = tmp_path / "source.keras"
    controlled_source(source)
    before = digest(source)
    destination = tmp_path / "export"
    rename = Path.rename
    write_text = Path.write_text

    def corrupt_artifact(path, target):
        result = rename(path, target)
        if Path(target) == destination / "model.keras":
            Path(target).write_bytes(b"externally corrupted artifact")
        return result

    def failed_manifest(path, *args, **kwargs):
        if path.name == "manifest.pending":
            if failure == "publish_interrupt":
                raise KeyboardInterrupt("interrupted publication")
            raise OSError("manifest write failed")
        return write_text(path, *args, **kwargs)

    if failure == "artifact_corruption":
        monkeypatch.setattr(Path, "rename", corrupt_artifact)
    else:
        monkeypatch.setattr(Path, "write_text", failed_manifest)
    exception = (
        KeyboardInterrupt if failure == "publish_interrupt" else (OSError, ValueError)
    )
    with pytest.raises(exception):
        export_checkpoint(source, destination)
    assert digest(source) == before
    assert (destination / "model.keras").exists()
    assert not (destination / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        export_checkpoint(source, destination)


def test_real_selected_checkpoint_save_reload_parity(tmp_path):
    source = Path(
        os.environ.get(
            "SPCAS9_ACCEPTANCE_CHECKPOINT",
            Path(__file__).resolve().parents[1] / "test_run/9hydra_v2.keras",
        )
    )
    assert source.is_file(), (
        "Real acceptance requires the agreed checkpoint; set "
        "SPCAS9_ACCEPTANCE_CHECKPOINT to its path"
    )
    # A missing acceptance input fails explicitly, never becomes a skip/pass.
    assert (
        digest(source)
        == "24e31cc6b9df8520e2bf761d33f7bb475c4658a9a479242b67db882946d0d2b8"
    )
    manifest = export_checkpoint(source, tmp_path / "real_export")
    for batch in manifest["validation"]["batches"]:
        assert len(batch["heads"]) == 9
        for head in batch["heads"]:
            np.testing.assert_allclose(
                head["reloaded_values"], head["source_values"], rtol=1e-5, atol=1e-6
            )


def test_staging_cleanup_failure_cannot_leave_a_success_marker(tmp_path, monkeypatch):
    source = tmp_path / "source.keras"
    controlled_source(source)
    cleanup = tempfile.TemporaryDirectory.__exit__

    def failed_cleanup(directory, *args):
        cleanup(directory, *args)
        raise OSError("staging cleanup failed")

    monkeypatch.setattr(tempfile.TemporaryDirectory, "__exit__", failed_cleanup)
    with pytest.raises(OSError, match="staging cleanup failed"):
        export_checkpoint(source, tmp_path / "export")
    assert not (tmp_path / "export/manifest.json").exists()


def test_cli_requires_explicit_paths_and_reports_failure(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2
    assert "--source" in capsys.readouterr().err
    assert (
        main(
            [
                "--source",
                str(tmp_path / "absent.keras"),
                "--destination",
                str(tmp_path / "export"),
            ]
        )
        == 1
    )
    assert "Export failed:" in capsys.readouterr().err
    assert not (tmp_path / "export/manifest.json").exists()


def test_cli_exports_the_explicit_checkpoint(tmp_path, capsys):
    source = tmp_path / "source.keras"
    controlled_source(source)
    destination = tmp_path / "export"
    assert main(["--source", str(source), "--destination", str(destination)]) == 0
    assert "Verified export:" in capsys.readouterr().out
    assert json.loads((destination / "manifest.json").read_text())["source"][
        "sha256"
    ] == digest(source)
