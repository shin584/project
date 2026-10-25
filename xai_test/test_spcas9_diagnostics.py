"""The approved diagnostic seam: explicit artifacts to preserved run evidence."""

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from SpCas9.spcas9_diagnostics import (
    ArtifactLocations,
    DiagnosticConfig,
    run_diagnostics,
)


@pytest.fixture
def artifacts(tmp_path):
    source = tmp_path / "originals"
    source.mkdir()
    for name in (
        "dataset.xlsx",
        "model.keras",
        "training_log.json",
        "predictions.csv",
        "predictions_checksum.json",
    ):
        (source / name).write_bytes(b"original evidence\n")
    split = source / "split"
    split.mkdir()
    (split / "train.csv").write_bytes(b"sequence,label\nACGT,0\n")
    (split / "split_checksum.json").write_bytes(b"{}\n")
    return ArtifactLocations(
        dataset=source / "dataset.xlsx",
        split_dir=split,
        checkpoint=source / "model.keras",
        training_log=source / "training_log.json",
        predictions=source / "predictions.csv",
        prediction_sidecar=source / "predictions_checksum.json",
    )


class UnavailableBackend:
    def environment(self):
        raise ModuleNotFoundError("fixture: TensorFlow unavailable")


def test_unavailable_backend_preserves_sources_and_writes_incomplete_report(
    artifacts, tmp_path
):
    before = {
        p: p.read_bytes() for p in (tmp_path / "originals").rglob("*") if p.is_file()
    }
    destination = tmp_path / "diagnostic"
    report = run_diagnostics(
        artifacts,
        destination,
        DiagnosticConfig(interpreter=Path(sys.executable)),
        backend=UnavailableBackend(),
    )

    assert report["outcome"] == "incomplete"
    assert report["checks"]["backend_environment"]["state"] == "blocked"
    assert report["checks"]["checkpoint_parity"]["state"] == "blocked"
    assert report["checks"]["source_preservation"]["state"] == "passed"
    assert before == {p: p.read_bytes() for p in before}
    assert json.loads((destination / "report.json").read_text()) == report
    assert "incomplete" in (destination / "summary.txt").read_text()
    config = json.loads((destination / "configuration.json").read_text())
    assert config["rtol"] == 1e-5
    assert config["atol"] == 1e-6
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["inputs"]
    assert all(entry["unchanged"] is True for entry in manifest["inputs"].values())


@pytest.mark.parametrize(
    "operation", ["fit", "train_on_batch", "apply_gradients", "update_step"]
)
def test_diagnostic_boundary_rejects_training_before_any_side_effect(
    artifacts, tmp_path, operation
):
    sentinel = tmp_path / "optimizer-was-called"

    class UnsafeOperations:
        def fit(self):
            sentinel.write_text("updated")

        def train_on_batch(self):
            sentinel.write_text("updated")

        def apply_gradients(self):
            sentinel.write_text("updated")

        def update_step(self):
            sentinel.write_text("updated")

    class UnsafeBackend:
        def environment(self):
            getattr(UnsafeOperations(), operation)()
            return {"backend": "unsafe"}

    report = run_diagnostics(
        artifacts,
        tmp_path / "run",
        DiagnosticConfig(Path(sys.executable)),
        backend=UnsafeBackend(),
    )
    assert report["outcome"] == "failed"
    assert report["checks"]["backend_environment"]["state"] == "failed"
    assert "forbidden" in report["checks"]["backend_environment"]["message"]
    assert not sentinel.exists()
    assert report["checks"]["source_preservation"]["state"] == "passed"


@pytest.mark.parametrize(
    "location",
    [
        "existing",
        "dataset",
        "split",
        "nested_split",
        "checkpoint",
        "log",
        "predictions",
        "sidecar",
    ],
)
def test_destination_collisions_never_modify_original_evidence(
    artifacts, tmp_path, location
):
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep.txt").write_bytes(b"keep partial evidence")
    targets = {
        "existing": existing,
        "dataset": artifacts.dataset,
        "split": artifacts.split_dir,
        "nested_split": artifacts.split_dir / "new",
        "checkpoint": artifacts.checkpoint,
        "log": artifacts.training_log,
        "predictions": artifacts.predictions,
        "sidecar": artifacts.prediction_sidecar,
    }
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises((FileExistsError, ValueError)):
        run_diagnostics(
            artifacts,
            targets[location],
            DiagnosticConfig(Path(sys.executable)),
            backend=UnavailableBackend(),
        )
    assert before == {p: p.read_bytes() for p in before}
    assert not (artifacts.split_dir / "new").exists()


def test_configuration_is_frozen_and_manifest_records_reproducibility(
    artifacts, tmp_path, monkeypatch
):
    monkeypatch.delenv("TF_ENABLE_ONEDNN_OPTS", raising=False)
    commands = ["explicit-interpreter -m SpCas9.spcas9_diagnostics --fixture"]
    config = DiagnosticConfig(Path(sys.executable), commands=commands)
    commands.append("changed after freeze")
    with pytest.raises(FrozenInstanceError):
        config.rtol = 0.1
    with pytest.raises((TypeError, ValueError)):
        replace(config, rtol=0.1)
    destination = tmp_path / "run"
    report = run_diagnostics(
        artifacts, destination, config, backend=UnavailableBackend()
    )
    frozen = json.loads((destination / "configuration.json").read_text())
    assert frozen["commands"] == [commands[0]]
    assert frozen["environment"]["TF_ENABLE_ONEDNN_OPTS"] == {
        "value": None,
        "status": "unset/default",
    }
    assert frozen["device"] == "/CPU:0"
    assert frozen["inference_batch_size"] == 1024
    assert frozen["fixture_specification"]
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["source"]["revision"]
    assert isinstance(manifest["source"]["dirty"], bool)
    assert manifest["source"]["files"]
    assert manifest["environment"]["interpreter"] == str(Path(sys.executable).resolve())
    assert manifest["environment"]["python"] == sys.version
    assert {"tensorflow", "keras", "numpy", "pandas"} <= manifest["environment"][
        "libraries"
    ].keys()
    assert manifest["environment"]["effective_onednn"].startswith("unknown")
    assert manifest["started_at"] <= manifest["finished_at"]
    assert report["finished_at"] == manifest["finished_at"]
    import hashlib

    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == digest
    assert set(manifest["outputs"]) == {
        "configuration.json",
        "report.json",
        "summary.txt",
    }


def test_changed_environment_is_rejected_before_reserving_destination(
    artifacts, tmp_path, monkeypatch
):
    config = DiagnosticConfig(Path(sys.executable))
    monkeypatch.setenv("TF_ENABLE_ONEDNN_OPTS", "changed-after-freeze")
    with pytest.raises(ValueError, match="frozen"):
        run_diagnostics(
            artifacts, tmp_path / "run", config, backend=UnavailableBackend()
        )
    assert not (tmp_path / "run").exists()


def test_stage_failure_preserves_partial_results_and_blocks_only_dependencies(
    artifacts, tmp_path
):
    destination = tmp_path / "run"
    before = {
        p: p.read_bytes() for p in (tmp_path / "originals").rglob("*") if p.is_file()
    }

    class PartialBackend:
        def environment(self):
            (destination / "partial.txt").write_text("partial diagnostic evidence")
            raise ValueError("fixture stage failed")

    report = run_diagnostics(
        artifacts,
        destination,
        DiagnosticConfig(Path(sys.executable)),
        backend=PartialBackend(),
    )
    assert report["outcome"] == "failed"
    checks = report["checks"]
    assert checks["backend_environment"]["state"] == "failed"
    assert checks["backend_environment"]["disposition"] == "no defect established"
    assert checks["checkpoint_parity"]["state"] == "blocked"
    assert checks["mask_metric_gradient"]["state"] == "blocked"
    assert "backend_environment" in checks["mask_metric_gradient"]["message"]
    assert checks["copy_reload"]["state"] == "blocked"
    assert "checkpoint_parity" in checks["copy_reload"]["message"]
    assert checks["input_snapshot"]["state"] == "passed"
    assert checks["source_provenance"]["state"] == "passed"
    assert checks["source_preservation"]["state"] == "passed"
    assert (destination / "partial.txt").read_text() == "partial diagnostic evidence"
    assert before == {p: p.read_bytes() for p in before}


def test_available_backend_does_not_verify_deferred_required_checks(
    artifacts, tmp_path
):
    class AvailableBackend:
        def environment(self):
            return {"backend": "fixture", "effective_onednn": "unknown"}

    report = run_diagnostics(
        artifacts,
        tmp_path / "run",
        DiagnosticConfig(Path(sys.executable)),
        backend=AvailableBackend(),
    )
    assert report["checks"]["backend_environment"]["state"] == "passed"
    assert (
        report["checks"]["backend_environment"]["disposition"]
        == "no defect established"
    )
    assert report["checks"]["source_preservation"]["state"] == "passed"
    assert report["checks"]["artifact_validation"]["state"] == "blocked"
    assert report["checks"]["checkpoint_parity"]["state"] == "blocked"
    assert report["outcome"] == "incomplete"


def test_diagnostic_boundary_cannot_create_a_candidate_study(artifacts, tmp_path):
    from SpCas9.spcas9_study import StudyProtocol

    class UnsafeBackend:
        def environment(self):
            StudyProtocol.plan(tmp_path / "candidate-study")

    report = run_diagnostics(
        artifacts,
        tmp_path / "run",
        DiagnosticConfig(Path(sys.executable)),
        backend=UnsafeBackend(),
    )
    assert report["outcome"] == "failed"
    assert "forbidden" in report["checks"]["backend_environment"]["message"]
    assert not (tmp_path / "candidate-study").exists()


def test_invalid_stage_json_does_not_erase_independent_checks(artifacts, tmp_path):
    class InvalidBackend:
        def environment(self):
            return {"observed": float("nan")}

    destination = tmp_path / "run"
    report = run_diagnostics(
        artifacts,
        destination,
        DiagnosticConfig(Path(sys.executable)),
        backend=InvalidBackend(),
    )
    assert report["outcome"] == "failed"
    assert report["checks"]["input_snapshot"]["state"] == "passed"
    assert report["checks"]["source_preservation"]["state"] == "passed"
    assert "NaN" not in (destination / "report.json").read_text()


def test_backend_failure_retains_before_and_after_hashes(artifacts, tmp_path):
    class FailedBackend:
        def environment(self):
            raise RuntimeError("fixture: incompatible runtime")

    destination = tmp_path / "run"
    before = {
        p: p.read_bytes() for p in (tmp_path / "originals").rglob("*") if p.is_file()
    }
    report = run_diagnostics(
        artifacts,
        destination,
        DiagnosticConfig(Path(sys.executable)),
        backend=FailedBackend(),
    )
    assert report["outcome"] == "failed"
    assert before == {p: p.read_bytes() for p in before}
    manifest = json.loads((destination / "manifest.json").read_text())
    assert all(
        v["before_sha256"] == v["after_sha256"] for v in manifest["inputs"].values()
    )


def test_source_mutation_is_a_failed_execution_without_automatic_repair(
    artifacts, tmp_path
):
    class MutatingBackend:
        def environment(self):
            artifacts.dataset.write_bytes(b"simulated external mutation")
            return {"backend": "fixture"}

    destination = tmp_path / "run"
    report = run_diagnostics(
        artifacts,
        destination,
        DiagnosticConfig(Path(sys.executable)),
        backend=MutatingBackend(),
    )
    assert report["outcome"] == "failed"
    assert report["checks"]["source_preservation"]["disposition"] == "unresolved defect"
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["inputs"][str(artifacts.dataset)]["unchanged"] is False
    assert artifacts.dataset.read_bytes() == b"simulated external mutation"


def test_missing_artifacts_are_reported_without_attempting_dependent_checks(
    artifacts, tmp_path
):
    missing = replace(artifacts, checkpoint=tmp_path / "unavailable.keras")
    destination = tmp_path / "run"
    report = run_diagnostics(
        missing,
        destination,
        DiagnosticConfig(Path(sys.executable)),
        backend=UnavailableBackend(),
    )
    assert report["outcome"] == "incomplete"
    assert report["checks"]["input_snapshot"]["state"] == "blocked"
    assert report["checks"]["artifact_validation"]["state"] == "blocked"
    assert "unavailable.keras" in report["checks"]["input_snapshot"]["message"]


def test_cli_retains_report_and_exits_unsuccessfully_without_backend(
    artifacts, tmp_path
):
    destination = tmp_path / "cli-run"
    arguments = [
        "--dataset",
        str(artifacts.dataset),
        "--split-dir",
        str(artifacts.split_dir),
        "--checkpoint",
        str(artifacts.checkpoint),
        "--training-log",
        str(artifacts.training_log),
        "--predictions",
        str(artifacts.predictions),
        "--prediction-sidecar",
        str(artifacts.prediction_sidecar),
        "--destination",
        str(destination),
    ]
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy, sys; sys.modules.update({'tensorflow': None, 'keras': None}); "
                "runpy.run_module('SpCas9.spcas9_diagnostics', run_name='__main__')"
            ),
            *arguments,
        ],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 1
    assert "incomplete" in process.stdout
    assert "Traceback" not in process.stderr
    assert (
        json.loads((destination / "report.json").read_text())["outcome"] == "incomplete"
    )
    frozen = json.loads((destination / "configuration.json").read_text())
    assert "--destination" in frozen["commands"][0]


def test_interruption_keeps_partial_evidence_and_finishes_input_hashes(
    artifacts, tmp_path
):
    destination = tmp_path / "run"

    class InterruptedBackend:
        def environment(self):
            (destination / "partial").mkdir()
            (destination / "partial" / "probe.txt").write_text("interrupted probe")
            raise KeyboardInterrupt

    report = run_diagnostics(
        artifacts,
        destination,
        DiagnosticConfig(Path(sys.executable)),
        backend=InterruptedBackend(),
    )
    assert report["outcome"] == "incomplete"
    assert report["checks"]["backend_environment"]["state"] == "blocked"
    assert report["checks"]["source_preservation"]["state"] == "passed"
    manifest = json.loads((destination / "manifest.json").read_text())
    assert "partial/probe.txt" in manifest["outputs"]
    assert all(entry["unchanged"] for entry in manifest["inputs"].values())


def test_destination_alias_into_source_directory_is_rejected(artifacts, tmp_path):
    alias = tmp_path / "source-alias"
    try:
        alias.symlink_to(artifacts.split_dir, target_is_directory=True)
    except OSError:
        pytest.skip("creating symlinks requires OS privileges")
    with pytest.raises(ValueError, match="overlaps"):
        run_diagnostics(
            artifacts,
            alias / "new",
            DiagnosticConfig(Path(sys.executable)),
            backend=UnavailableBackend(),
        )
    assert not (artifacts.split_dir / "new").exists()


@pytest.mark.parametrize("source_name", ["split_dir", "dataset"])
def test_destination_cannot_create_a_missing_source_location(
    artifacts, tmp_path, source_name
):
    missing = tmp_path / "missing-source"
    locations = replace(artifacts, **{source_name: missing})
    with pytest.raises(ValueError, match="overlaps"):
        run_diagnostics(
            locations,
            missing / "run",
            DiagnosticConfig(Path(sys.executable)),
            backend=UnavailableBackend(),
        )
    assert not missing.exists()
