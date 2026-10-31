"""Same-VM recovery through the public workflow and persisted artifact contract."""

import json
import shutil
from pathlib import Path

import pytest

from cas12a.nt_data import canonical_bytes
from cas12a.nt_workflow import FitFixture, ModelSource
from cas12a.test.test_nt_preflight import artifact, prepared

LOSSES = FitFixture(validation_losses=(0.2, 0.4, 0.3, 0.2))


def interrupted(
    tmp_path, tiny_nt, monkeypatch, *, counts=(96, 4, 2), step=3, device=None
):
    import torch

    workflow, run = prepared(tmp_path, counts=counts)
    run = workflow.preflight(run, model=ModelSource.fixture(tiny_nt), device=device)
    save = torch.save

    def interrupt_save(value, path, *args, **kwargs):
        if Path(path).parent.name == f".checkpoint-{step}.incomplete":
            raise KeyboardInterrupt("fixture interruption")
        return save(value, path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", interrupt_save)
        with pytest.raises(KeyboardInterrupt, match="fixture interruption"):
            workflow.fit(run, fixture=LOSSES)
    assert artifact(run, "status.json")["state"] == "training_interrupted"
    return workflow, run


@pytest.mark.parametrize("counts", [(96, 4, 2), (64, 4, 2), (33, 4, 2)])
def test_resume_preserves_updates_selection_patience_rng_and_export(
    tmp_path, tiny_nt, monkeypatch, counts
):
    import torch

    # Dropout makes lost model RNG visible in weights and validation predictions.
    config = json.loads((tiny_nt / "config.json").read_bytes())
    config.update(hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.1)
    (tiny_nt / "config.json").write_bytes(canonical_bytes(config))
    reference_path = tmp_path / "reference"
    reference_path.mkdir()
    reference, full = prepared(reference_path, counts=counts)
    full = reference.preflight(full, model=ModelSource.fixture(tiny_nt))
    full = reference.fit(full, fixture=LOSSES)
    full = reference.export_selected(full)
    reference_history = artifact(full, "history.json")

    recovery_path = tmp_path / "recovery"
    recovery_path.mkdir()
    workflow, run = interrupted(recovery_path, tiny_nt, monkeypatch, counts=counts)
    best_before = (
        run.checkpoint_dir / "checkpoint-1" / "model.safetensors"
    ).read_bytes()
    partition_before = workflow.partition_path.read_bytes()
    run = workflow.resume(run.run_id)  # Works without the original in-memory run.
    assert run.status["state"] == "training_completed"
    assert run.status["resumed_from_global_step"] == 2
    assert run.status["model_completed"] is False
    assert "error" not in run.status
    history = artifact(run, "history.json")
    for key in ("training", "validation", "selected_global_step", "stopping_reason"):
        assert history[key] == reference_history[key]
    assert history["optimizer_updates"] == 4
    assert history["selected_global_step"] == 1
    assert history["initial_global_step"] == 0
    assert workflow.partition_path.read_bytes() == partition_before
    assert sorted(p.name for p in run.checkpoint_dir.glob("checkpoint-*")) == [
        "checkpoint-1",
        "checkpoint-4",
    ]
    latest = torch.load(
        run.checkpoint_dir / "checkpoint-4" / "training_state.pt", weights_only=True
    )
    assert latest["scheduler"]["last_epoch"] == 4
    assert latest["progress"]["non_improving_validations"] == 3
    assert (
        run.checkpoint_dir / "checkpoint-1" / "model.safetensors"
    ).read_bytes() == best_before
    run = workflow.export_selected(run)
    run = workflow.persist_export(run)
    report = artifact(run, "export.json")
    assert report["files"] == artifact(full, "export.json")["files"]
    assert report["selected_checkpoint_sha256"] == history["selected_checkpoint_sha256"]
    events = artifact(run, "lifecycle.json")["events"]
    assert "resuming" in [e["status"]["state"] for e in events]
    assert not list(run.durable_dir.rglob("training_state.pt"))
    protected = (run.durable_dir / "selected_export" / "model.safetensors").read_bytes()
    with pytest.raises(ValueError, match="incompatible"):
        workflow.resume(run)
    assert (
        run.durable_dir / "selected_export" / "model.safetensors"
    ).read_bytes() == protected


@pytest.mark.parametrize(
    "damage",
    [
        "optimizer",
        "rng",
        "rng.python",
        "rng.numpy",
        "rng.torch",
        "rng.cuda",
        "rng.data_order",
        "scaler",
        "progress",
        "progress.non_improving_validations",
        "history",
        "history.selected_predictions",
        "history.reload_validation_predictions",
        "history.selected_fractional_epoch",
        "scheduler",
        "scheduler.last_epoch",
        "progress.next_example_offset",
        "optimizer.state",
        "model",
        "manifest",
        "partial-model",
        "optimizer-lr",
        "scheduler-incomplete",
    ],
)
def test_resume_rejects_missing_corrupt_or_inconsistent_checkpoint_state(
    tmp_path, tiny_nt, monkeypatch, damage
):
    import torch

    from cas12a.nt_storage import files

    workflow, run = interrupted(tmp_path, tiny_nt, monkeypatch)
    checkpoint = run.checkpoint_dir / "checkpoint-2"
    if damage == "model":
        (checkpoint / "model.safetensors").write_bytes(b"corrupt")
    elif damage == "manifest":
        manifest = json.loads((checkpoint / "checkpoint.json").read_bytes())
        manifest["execution_sha256"] = "changed"
        (checkpoint / "checkpoint.json").write_bytes(canonical_bytes(manifest))
    else:
        if damage == "partial-model":
            from safetensors.torch import load_file, save_file

            weights = {
                key: value.clone()
                for key, value in load_file(checkpoint / "model.safetensors").items()
            }
            weights.pop(next(iter(weights)))
            save_file(
                weights, checkpoint / "model.safetensors", metadata={"format": "pt"}
            )
        else:
            payload = torch.load(checkpoint / "training_state.pt", weights_only=True)
            if damage == "optimizer-lr":
                payload["optimizer"]["param_groups"][0]["lr"] = 0.5
            elif damage == "scheduler-incomplete":
                del payload["scheduler"]["base_lrs"]
            else:
                keys = damage.split(".")
                target = payload
                for key in keys[:-1]:
                    target = target[key]
                del target[keys[-1]]
            torch.save(payload, checkpoint / "training_state.pt")
        # Repair the file checksum to exercise required state validation, beyond
        # the independent ordinary-corruption hash check above.
        manifest = json.loads((checkpoint / "checkpoint.json").read_bytes())
        contents = files(checkpoint)
        del contents["checkpoint.json"]
        manifest["files"] = contents
        (checkpoint / "checkpoint.json").write_bytes(canonical_bytes(manifest))
    before = {p.name for p in run.checkpoint_dir.iterdir()}
    with pytest.raises(ValueError, match="resume"):
        workflow.resume(run)
    assert artifact(run, "status.json")["state"] == "resume_failed"
    assert before == {p.name for p in run.checkpoint_dir.iterdir()}
    assert not (run.local_dir / "selected_export").exists()


@pytest.mark.parametrize(
    "damage",
    ["source", "partition", "missing-partition", "execution", "training", "backbone"],
)
def test_resume_rejects_changed_experiment_without_rewriting_partition(
    tmp_path, tiny_nt, monkeypatch, damage
):
    workflow, run = interrupted(tmp_path, tiny_nt, monkeypatch)
    if damage == "source":
        with workflow.source.open("ab") as handle:
            handle.write(b"\n")
    elif damage == "partition":
        partition = json.loads(workflow.partition_path.read_bytes())
        partition["partitions"]["train"].reverse()
        workflow.partition_path.write_bytes(canonical_bytes(partition))
    elif damage == "missing-partition":
        workflow.partition_path.unlink()
    elif damage == "backbone":
        with (tiny_nt / "vocab.txt").open("a") as handle:
            handle.write("CHANGED\n")
    else:
        filename = damage + ".json"
        document = artifact(run, filename)
        if damage == "execution":
            document["source"]["revision"] = "changed"
        else:
            document["patience"] = 100
        for directory in (run.local_dir, run.durable_dir):
            (directory / filename).write_bytes(canonical_bytes(document))
    partition_bytes = (
        workflow.partition_path.read_bytes()
        if workflow.partition_path.exists()
        else None
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        workflow.resume(run)
    if partition_bytes is None:
        assert not workflow.partition_path.exists()
    else:
        assert workflow.partition_path.read_bytes() == partition_bytes
    assert artifact(run, "status.json")["state"] == "resume_failed"
    with pytest.raises(ValueError):
        workflow.fit(run)


@pytest.mark.parametrize("lost", ["checkpoints", "local"])
def test_recovery_unavailable_requires_an_explicit_new_run_on_verified_partition(
    tmp_path, tiny_nt, monkeypatch, lost
):
    workflow, run = interrupted(tmp_path, tiny_nt, monkeypatch)
    partition_bytes = workflow.partition_path.read_bytes()
    target = run.checkpoint_dir if lost == "checkpoints" else run.local_dir
    assert target.resolve().is_relative_to(tmp_path.resolve())
    shutil.rmtree(target)
    prior_events = json.loads((run.durable_dir / "lifecycle.json").read_bytes())[
        "events"
    ]
    with pytest.raises(FileNotFoundError, match="recovery unavailable"):
        workflow.resume(run.run_id)
    status = json.loads((run.durable_dir / "status.json").read_bytes())
    assert status["state"] == "recovery_unavailable"
    events = json.loads((run.durable_dir / "lifecycle.json").read_bytes())["events"]
    assert events[:-1] == prior_events
    assert not (run.durable_dir / "checkpoints").exists()
    assert workflow.partition_path.read_bytes() == partition_bytes
    fresh = workflow.prepare(run_id="explicit-fresh")
    assert fresh.status["state"] == "prepared"
    assert workflow.partition_path.read_bytes() == partition_bytes


@pytest.mark.parametrize("step", [1, 4])
def test_resume_after_checkpoint_commit_restores_self_selected_or_terminal_state(
    tmp_path, tiny_nt, monkeypatch, step
):
    workflow, run = prepared(tmp_path, counts=(64, 4, 2))
    run = workflow.preflight(run, model=ModelSource.fixture(tiny_nt))
    rename = Path.rename

    def commit_then_interrupt(path, destination):
        result = rename(path, destination)
        if Path(destination).name == f"checkpoint-{step}":
            raise KeyboardInterrupt("after checkpoint commit")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", commit_then_interrupt)
        with pytest.raises(KeyboardInterrupt):
            workflow.fit(run, fixture=LOSSES)
    run = workflow.resume(run)
    history = artifact(run, "history.json")
    assert run.status["resumed_from_global_step"] == step
    assert history["optimizer_updates"] == 4
    assert history["selected_global_step"] == 1
    assert history["stopping_reason"] == "early_stopping"
    assert sorted(p.name for p in run.checkpoint_dir.glob("checkpoint-*")) == [
        "checkpoint-1",
        "checkpoint-4",
    ]
    run = workflow.export_selected(run)
    assert run.status["local_export_validated"] is True


@pytest.mark.parametrize("damage", [None, "missing", "partial"])
def test_fp16_resume_restores_scaler_progress_and_requires_complete_state(
    tmp_path, tiny_nt, monkeypatch, damage
):
    import torch

    from cas12a import nt_runtime
    from cas12a.nt_storage import files

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; FP16 recovery was not executed")
    if nt_runtime.runtime_environment("cuda", "fixture")["precision"] != "float16":
        pytest.skip("resolved runtime uses BF16; FP16 scaler is not applicable")
    workflow, run = interrupted(tmp_path, tiny_nt, monkeypatch, device="cuda")
    checkpoint = run.checkpoint_dir / "checkpoint-2"
    payload = torch.load(
        checkpoint / "training_state.pt", map_location="cpu", weights_only=True
    )
    scaler_before = payload["scaler"].copy()
    assert scaler_before["scale"] > 0
    if damage is not None:
        if damage == "missing":
            del payload["scaler"]
        else:
            del payload["scaler"]["scale"]
        torch.save(payload, checkpoint / "training_state.pt")
        manifest = json.loads((checkpoint / "checkpoint.json").read_bytes())
        contents = files(checkpoint)
        del contents["checkpoint.json"]
        manifest["files"] = contents
        (checkpoint / "checkpoint.json").write_bytes(canonical_bytes(manifest))
        with pytest.raises(ValueError, match="scaler"):
            workflow.resume(run)
        assert artifact(run, "status.json")["state"] == "resume_failed"
    else:
        run = workflow.resume(run)
        history = artifact(run, "history.json")
        latest = torch.load(
            run.checkpoint_dir / "checkpoint-4" / "training_state.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert latest["scaler"]["scale"] == scaler_before["scale"]
        assert (
            latest["scaler"]["_growth_tracker"] == scaler_before["_growth_tracker"] + 2
        )
        assert history["optimizer_updates"] == 4
        assert history["stopping_reason"] == "early_stopping"
