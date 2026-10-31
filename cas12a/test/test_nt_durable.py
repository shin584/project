"""Durable lifecycle artifacts through the public NT workflow boundary."""

import hashlib
import json
from pathlib import Path

import pytest

from cas12a.nt_workflow import FitFixture, ModelSource
from cas12a.test.test_nt_preflight import artifact, prepared


def local_export(tmp_path, tiny_nt):
    workflow, run = prepared(tmp_path)
    run = workflow.preflight(run, model=ModelSource.fixture(tiny_nt))
    run = workflow.fit(run, fixture=FitFixture(validation_losses=(0.1, 0.2, 0.2, 0.2)))
    return workflow, workflow.export_selected(run)


def test_selected_export_is_durable_only_after_verified_copy(tmp_path, tiny_nt):
    workflow, run = local_export(tmp_path, tiny_nt)
    assert not (run.durable_dir / "selected_export").exists()
    assert run.status["durable_export_validated"] is False

    run = workflow.persist_export(run)

    assert run.status["state"] == "durable_export_validated"
    assert run.status["local_export_validated"] is True
    assert run.status["durable_export_validated"] is True
    assert run.status["final_evaluation_completed"] is False
    assert run.status["model_completed"] is False
    receipt = artifact(run, "durable_export.json")
    report = artifact(run, "export.json")
    assert receipt["validated"] is True
    assert receipt["run_kind"] == "fixture"
    assert receipt["selected_checkpoint_sha256"] == report["selected_checkpoint_sha256"]
    assert receipt["selected_artifact_sha256"] == report["selected_artifact_sha256"]
    for name, expected in receipt["files"].items():
        assert (
            hashlib.sha256((run.local_dir / name).read_bytes()).hexdigest() == expected
        )
        assert (
            hashlib.sha256((run.durable_dir / name).read_bytes()).hexdigest()
            == expected
        )
    assert {
        "configuration.json",
        "data.json",
        "partition.json",
        "execution.json",
        "preflight.json",
        "tokenization.json",
        "history.json",
        "training.json",
        "export.json",
        "selected_export/config.json",
        "selected_export/vocab.txt",
        "selected_export/model.safetensors",
    } <= receipt["files"].keys()
    assert not list(run.durable_dir.rglob("*.pt"))
    assert not (run.durable_dir / "checkpoints").exists()
    assert workflow.verify_durable_export(run).status == run.status
    events = artifact(run, "lifecycle.json")["events"]
    states = [event["status"]["state"] for event in events]
    assert states == [
        "preparing",
        "prepared",
        "preflighting",
        "preflight_passed",
        "training",
        "training_completed",
        "exporting",
        "export_validated",
        "persisting_export",
        "durable_export_validated",
    ]
    assert all(event["recorded_utc"] for event in events)


@pytest.mark.parametrize(
    "failure", ["corrupt", "copy_error", "interrupted", "status_error"]
)
def test_incomplete_copy_cannot_claim_durable_export(
    tmp_path, tiny_nt, monkeypatch, failure
):
    import shutil

    workflow, run = local_export(tmp_path, tiny_nt)
    copy = shutil.copyfile

    def faulty_copy(source, destination, *args, **kwargs):
        if Path(destination).name == "model.safetensors":
            if failure == "copy_error":
                raise OSError("fixture copy failure")
            if failure == "interrupted":
                raise KeyboardInterrupt("fixture interrupted copy")
            result = copy(source, destination, *args, **kwargs)
            Path(destination).write_bytes(b"corrupt copied weights")
            return result
        return copy(source, destination, *args, **kwargs)

    if failure == "status_error":
        # Fail creation at the filesystem API rather than mocking workflow helpers.
        open_file = Path.open

        def failed_open(path, mode="r", *args, **kwargs):
            if path.parent == run.durable_dir and path.name.startswith(".status.json."):
                status = json.loads((run.local_dir / "status.json").read_bytes())
                if status["state"] == "durable_export_validated":
                    raise OSError("fixture final status failure")
            return open_file(path, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", failed_open)
    else:
        monkeypatch.setattr(shutil, "copyfile", faulty_copy)
    expected = KeyboardInterrupt if failure == "interrupted" else (ValueError, OSError)
    with pytest.raises(expected, match="identity|fixture"):
        workflow.persist_export(run)
    status = artifact(run, "status.json")
    assert status["state"] == "durable_export_failed"
    assert status["local_export_validated"] is True
    assert status["durable_export_validated"] is False
    assert status["model_completed"] is False
    receipt = run.durable_dir / "durable_export.json"
    assert (
        not receipt.exists() or json.loads(receipt.read_bytes())["validated"] is False
    )
    if failure != "status_error":
        assert not (run.durable_dir / "selected_export").exists()
    assert (
        run.local_dir / "selected_export" / "model.safetensors"
    ).stat().st_size > 100


@pytest.mark.parametrize(
    "damage", ["weights", "tokenizer", "missing", "receipt", "configuration"]
)
def test_durable_reverification_revokes_corrupted_or_missing_evidence(
    tmp_path, tiny_nt, damage
):
    workflow, run = local_export(tmp_path, tiny_nt)
    run = workflow.persist_export(run)
    if damage == "receipt":
        path = run.durable_dir / "durable_export.json"
    elif damage == "configuration":
        path = run.durable_dir / "configuration.json"
    else:
        name = "vocab.txt" if damage == "tokenizer" else "model.safetensors"
        path = run.durable_dir / "selected_export" / name
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"damaged")
    with pytest.raises((ValueError, OSError)):
        workflow.verify_durable_export(run)
    status = json.loads((run.local_dir / "status.json").read_bytes())
    assert status["durable_export_validated"] is False
    assert status["state"] == "durable_export_failed"
    assert status["final_evaluation_completed"] is False


def test_durable_export_and_run_destinations_cannot_be_overwritten(tmp_path, tiny_nt):
    workflow, run = local_export(tmp_path, tiny_nt)
    destination = run.durable_dir / "selected_export"
    destination.mkdir()
    (destination / "prior.txt").write_text("preserve previous output")
    with pytest.raises(FileExistsError, match="already exists"):
        workflow.persist_export(run)
    assert (destination / "prior.txt").read_text() == "preserve previous output"
    assert artifact(run, "status.json")["state"] == "export_validated"
    with pytest.raises(FileExistsError, match="already exists"):
        workflow.prepare(run_id=run.run_id)


@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("journal", ["intact", "damaged"])
def test_training_failure_survives_unavailable_durable_evidence(
    tmp_path, tiny_nt, monkeypatch, interrupted, journal
):
    import torch

    workflow, run = prepared(tmp_path)
    run = workflow.preflight(run, model=ModelSource.fixture(tiny_nt))
    save = torch.save
    open_file = Path.open
    storage_offline = False

    def broken_save(value, path, *args, **kwargs):
        nonlocal storage_offline
        if Path(path).parent.name == ".checkpoint-2.incomplete":
            storage_offline = True
            if journal == "damaged":
                (run.local_dir / "lifecycle.json").write_text('{"events":{}}')
            if interrupted:
                raise KeyboardInterrupt("original training interruption")
            raise RuntimeError("original training failure")
        return save(value, path, *args, **kwargs)

    def offline_drive(path, *args, **kwargs):
        if storage_offline and path.is_relative_to(run.durable_dir):
            raise OSError("secondary Drive storage failure")
        return open_file(path, *args, **kwargs)

    monkeypatch.setattr(torch, "save", broken_save)
    monkeypatch.setattr(Path, "open", offline_drive)
    expected = KeyboardInterrupt if interrupted else RuntimeError
    with pytest.raises(expected, match="original training") as caught:
        workflow.fit(run, fixture=FitFixture(validation_losses=(0.1, 0.2, 0.2, 0.2)))
    status = json.loads((run.local_dir / "status.json").read_bytes())
    assert status["state"] == (
        "training_interrupted" if interrupted else "training_failed"
    )
    assert status["durable_export_validated"] is False
    assert status["model_completed"] is False
    assert any("secondary Drive" in note for note in caught.value.__notes__)
    journal = json.loads((run.local_dir / "lifecycle.json").read_bytes())
    assert journal["events"][-1]["status"]["state"] == status["state"]


def evaluation_evidence(run):
    """Stand in for the later evaluator's validated output, not numerical evidence."""
    directory = run.local_dir / "final_evaluation"
    directory.mkdir()
    (directory / "metrics.json").write_text('{"fixture":true,"raw":{"mae":0.1}}')
    (directory / "predictions.csv").write_text("source_row,raw_prediction\n8,0.25\n")
    receipt = artifact(run, "durable_export.json")
    report = {
        "validated": True,
        "run_id": run.run_id,
        "run_kind": "fixture",
        "selected_checkpoint_sha256": receipt["selected_checkpoint_sha256"],
        "selected_artifact_sha256": receipt["selected_artifact_sha256"],
        "partition_manifest_sha256": artifact(run, "configuration.json")[
            "partition_manifest_sha256"
        ],
        "files": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("metrics.json", "predictions.csv")
        },
    }
    (directory / "evaluation.json").write_text(json.dumps(report))
    return directory


def test_later_evaluation_publishes_against_same_selected_model(tmp_path, tiny_nt):
    workflow, run = local_export(tmp_path, tiny_nt)
    run = workflow.persist_export(run)
    evidence = evaluation_evidence(run)

    run = workflow.persist_evaluation(run)

    assert run.status["state"] == "final_evaluation_completed"
    assert run.status["final_evaluation_completed"] is True
    # Offline fixture publication is never production model completion.
    assert run.status["model_completed"] is False
    receipt = artifact(run, "durable_evaluation.json")
    assert (
        receipt["selected_artifact_sha256"]
        == artifact(run, "export.json")["selected_artifact_sha256"]
    )
    for name, expected in receipt["files"].items():
        assert hashlib.sha256((evidence / name).read_bytes()).hexdigest() == expected
        assert (
            hashlib.sha256(
                (run.durable_dir / "final_evaluation" / name).read_bytes()
            ).hexdigest()
            == expected
        )
    assert (
        workflow.verify_durable_export(run).status["durable_export_validated"] is True
    )
    assert (
        workflow.verify_durable_evaluation(run).status["final_evaluation_completed"]
        is True
    )
    with pytest.raises((ValueError, FileExistsError)):
        workflow.persist_evaluation(run)


@pytest.mark.parametrize(
    "damage", ["identity", "partition", "not_validated", "missing", "copy"]
)
def test_invalid_or_incomplete_evaluation_cannot_claim_completion(
    tmp_path, tiny_nt, monkeypatch, damage
):
    import shutil

    workflow, run = local_export(tmp_path, tiny_nt)
    run = workflow.persist_export(run)
    directory = evaluation_evidence(run)
    report_path = directory / "evaluation.json"
    report = json.loads(report_path.read_bytes())
    if damage == "identity":
        report["selected_artifact_sha256"] = "wrong-model"
    elif damage == "partition":
        report["partition_manifest_sha256"] = "wrong-partition"
    elif damage == "not_validated":
        report["validated"] = False
    elif damage == "missing":
        (directory / "predictions.csv").unlink()
    else:
        copy = shutil.copyfile

        def corrupt_copy(source, destination, *args, **kwargs):
            result = copy(source, destination, *args, **kwargs)
            if Path(destination).name == "predictions.csv":
                Path(destination).write_bytes(b"truncated copy")
            return result

        monkeypatch.setattr(shutil, "copyfile", corrupt_copy)
    report_path.write_text(json.dumps(report))
    with pytest.raises((ValueError, OSError), match="identity|incomplete|validated"):
        workflow.persist_evaluation(run)
    status = artifact(run, "status.json")
    assert status["state"] == "final_evaluation_failed"
    assert status["durable_export_validated"] is True
    assert status["final_evaluation_completed"] is False
    assert status["model_completed"] is False
    assert not (run.durable_dir / "durable_evaluation.json").exists()


@pytest.mark.parametrize("damage", ["metrics", "predictions", "weights"])
def test_completed_evaluation_is_revoked_if_either_bundle_is_damaged(
    tmp_path, tiny_nt, damage
):
    workflow, run = local_export(tmp_path, tiny_nt)
    run = workflow.persist_export(run)
    evaluation_evidence(run)
    run = workflow.persist_evaluation(run)
    name = {
        "metrics": "final_evaluation/metrics.json",
        "predictions": "final_evaluation/predictions.csv",
        "weights": "selected_export/model.safetensors",
    }[damage]
    (run.durable_dir / name).write_bytes(b"damaged after publication")
    with pytest.raises(ValueError, match="identity"):
        workflow.verify_durable_evaluation(run)
    status = artifact(run, "status.json")
    assert status["final_evaluation_completed"] is False
    assert status["durable_evaluation_validated"] is False
    assert status["model_completed"] is False


@pytest.mark.parametrize(
    "name", ["training.json", "preflight.json", "tokenization.json"]
)
def test_durable_export_rejects_changed_metadata_even_if_both_copies_match(
    tmp_path, tiny_nt, name
):
    workflow, run = local_export(tmp_path, tiny_nt)
    value = artifact(run, name)
    value["unexpected_change"] = True
    for directory in (run.local_dir, run.durable_dir):
        (directory / name).write_text(json.dumps(value))
    with pytest.raises(ValueError, match="identity"):
        workflow.persist_export(run)
    assert artifact(run, "status.json")["durable_export_validated"] is False
    assert not (run.durable_dir / "durable_export.json").exists()


def test_local_checkpoints_cannot_be_placed_within_a_custom_drive_mount(tmp_path):
    from cas12a.nt_workflow import NTWorkflow

    mount = tmp_path / "mounted-drive"
    with pytest.raises(ValueError, match="Drive mount"):
        NTWorkflow(
            tmp_path / "source.csv",
            tmp_path / "partition.json",
            mount / "local",
            mount / "outputs",
            drive_mount_root=mount,
        )
