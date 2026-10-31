"""Training and local selected export through the public NT lifecycle."""

import hashlib
import json
from pathlib import Path

import pytest

from cas12a.nt_workflow import FitFixture, ModelSource
from cas12a.test.test_nt_preflight import artifact, prepared


def test_fit_selects_earlier_checkpoint_stops_and_reloads_its_predictions(
    tmp_path, tiny_nt
):
    import torch
    from transformers import EsmForSequenceClassification, EsmTokenizer

    from cas12a.nt_workflow import FitFixture

    workflow, run = prepared(tmp_path, counts=(64, 4, 2))
    run = workflow.preflight(run, model=ModelSource.fixture(tiny_nt))
    run = workflow.fit(run, fixture=FitFixture(validation_losses=(0.2, 0.4, 0.3, 0.2)))

    history = artifact(run, "history.json")
    assert run.status["state"] == "training_completed"
    assert history["selected_global_step"] == 1
    assert history["selected_fractional_epoch"] == 0.5
    assert history["optimizer_updates"] == 4
    assert history["epochs_completed"] == 2
    assert history["fractional_epochs_run"] == 2.0
    assert history["stopping_reason"] == "early_stopping"
    assert [v["global_step"] for v in history["validation"]] == [1, 2, 3, 4]
    assert [v["loss"] for v in history["validation"]] == [0.2, 0.4, 0.3, 0.2]
    assert all(v["measured_loss"] >= 0 for v in history["validation"])
    assert len(history["training"]) == 4
    recipe = artifact(run, "training.json")
    assert recipe["validation_interval_updates"] == 1
    assert recipe["production_interval_updates"] == 125
    assert recipe["fixture"]["validation_losses"] == [0.2, 0.4, 0.3, 0.2]
    assert recipe["schedule"]["planned_optimizer_updates"] == 20
    assert recipe["schedule"]["warmup_steps"] == 1
    assert recipe["optimizer"]["learning_rate"] == 2e-5
    assert recipe["optimizer"]["weight_decay"] == 0.01
    assert recipe["schedule"]["seed"] == 42
    assert recipe["schedule"]["max_epochs"] == 10
    assert history["initial_global_step"] == 0
    assert history["initial_optimizer_entries"] == 0
    assert (
        history["initialization_sha256"]
        == artifact(run, "execution.json")["initialization_sha256"]
    )
    checkpoints = sorted(p.name for p in run.checkpoint_dir.iterdir())
    assert checkpoints == ["checkpoint-1", "checkpoint-4"]
    latest = torch.load(
        run.checkpoint_dir / "checkpoint-4" / "training_state.pt", weights_only=True
    )
    assert latest["progress"]["non_improving_validations"] == 3
    assert latest["progress"]["global_step"] == 4
    assert latest["scheduler"]["last_epoch"] == 4
    assert latest["optimizer"]["state"]
    assert {"python", "numpy", "torch", "cuda", "data_order"} <= latest["rng"].keys()
    assert latest["scaler"] == {}  # CPU fixture has no FP16 scaler.
    assert not list(run.durable_dir.rglob("*.pt"))
    assert not list(run.durable_dir.rglob("*.safetensors"))

    run = workflow.export_selected(run)

    assert run.status["state"] == "export_validated"
    assert run.status["model_completed"] is False  # Final Testset is a later stage.
    report = artifact(run, "export.json")
    assert report["validated"] is True
    assert report["durable_export_validated"] is False
    assert report["selected_global_step"] == 1
    partition = artifact(run, "partition.json")
    assert report["validation_source_rows"] == partition["partitions"]["validation"]
    assert set(report["validation_source_rows"]).isdisjoint(
        partition["partitions"]["test"]
    )
    assert report["tokenization_matches"] is True
    assert report["max_absolute_difference"] <= report["atol"]
    export = run.local_dir / "selected_export"
    for name, expected in report["files"].items():
        assert hashlib.sha256((export / name).read_bytes()).hexdigest() == expected
    model = EsmForSequenceClassification.from_pretrained(export, local_files_only=True)
    tokenizer = EsmTokenizer.from_pretrained(export, local_files_only=True)
    data = artifact(run, "data.json")
    by_id = {r["source_row"]: r for r in data["rows"]}
    batch = tokenizer(
        [by_id[i]["sequence"] for i in report["validation_source_rows"]],
        padding="max_length",
        max_length=12,
        truncation=False,
        return_tensors="pt",
    )
    model.eval()
    with torch.no_grad():
        predictions = model(**batch).logits.flatten().tolist()
    assert predictions == pytest.approx(report["selected_predictions"], abs=1e-6)
    last_model = EsmForSequenceClassification.from_pretrained(
        run.checkpoint_dir / "checkpoint-4", local_files_only=True
    )
    last_model.eval()
    with torch.no_grad():
        last_predictions = last_model(**batch).logits.flatten().tolist()
    assert max(abs(a - b) for a, b in zip(predictions, last_predictions)) > 1e-8


@pytest.mark.parametrize("damage", ["weights", "tokenizer", "report", "report-array"])
def test_reverification_rejects_damaged_export_and_revokes_validated_state(
    tmp_path, tiny_nt, damage
):
    from cas12a.nt_workflow import FitFixture

    workflow, run = prepared(tmp_path)
    run = workflow.preflight(run, model=ModelSource.fixture(tiny_nt))
    run = workflow.fit(run, fixture=FitFixture(validation_losses=(0.1, 0.2, 0.2, 0.2)))
    run = workflow.export_selected(run)
    original_report = artifact(run, "export.json")
    run = workflow.verify_export(run)
    assert run.status["local_export_validated"] is True
    if damage in ("report", "report-array"):
        report = (
            []
            if damage == "report-array"
            else dict(original_report, selected_global_step=4)
        )
        for directory in (run.local_dir, run.durable_dir):
            (directory / "export.json").write_text(json.dumps(report))
    else:
        name = "model.safetensors" if damage == "weights" else "vocab.txt"
        (run.local_dir / "selected_export" / name).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="identity"):
        workflow.verify_export(run)
    status = artifact(run, "status.json")
    assert status["state"] == "export_failed"
    assert status["local_export_validated"] is False
    assert status["model_completed"] is False
    assert artifact(run, "export.json")["validated"] is False


def test_intervals_count_updates_after_accumulation(tmp_path, tiny_nt, monkeypatch):
    import torch

    workflow, run = prepared(tmp_path, counts=(64, 4, 2))
    allocate = torch.tensor

    def limited_allocator(values, *args, **kwargs):
        if isinstance(values, list) and len(values) > 4 and isinstance(values[0], list):
            raise torch.OutOfMemoryError("fixture allocation limit")
        return allocate(values, *args, **kwargs)

    monkeypatch.setattr(torch, "tensor", limited_allocator)
    run = workflow.preflight(run, model=ModelSource.fixture(tiny_nt))
    run = workflow.fit(
        run,
        fixture=FitFixture(
            interval_updates=2,
            validation_losses=(0.1, 0.2, 0.2, 0.2),
        ),
    )
    recipe = artifact(run, "training.json")
    assert recipe["settings"]["train_microbatch"] == 4
    assert recipe["settings"]["gradient_accumulation"] == 8
    history = artifact(run, "history.json")
    assert history["optimizer_updates"] == 8
    assert [v["global_step"] for v in history["validation"]] == [2, 4, 6, 8]
    assert sorted(p.name for p in run.checkpoint_dir.iterdir()) == [
        "checkpoint-2",
        "checkpoint-8",
    ]
    assert (
        history["training_source_rows"]
        == artifact(run, "partition.json")["partitions"]["train"]
    )
    assert set(
        history["training_source_rows"] + history["validation_source_rows"]
    ).isdisjoint(artifact(run, "partition.json")["partitions"]["test"])


def test_production_interval_and_schedule_stay_fixed_when_stopping_early(
    tmp_path, tiny_nt
):
    workflow, run = prepared(tmp_path, counts=(12000, 2, 1))
    run = workflow.preflight(run, model=ModelSource.fixture(tiny_nt))
    run = workflow.fit(
        run,
        fixture=FitFixture(
            interval_updates=125,
            validation_losses=(0.1, 0.2, 0.2, 0.2),
        ),
    )
    recipe = artifact(run, "training.json")
    assert recipe["updates_per_epoch"] == 375
    assert (
        recipe["validation_interval_updates"]
        == recipe["checkpoint_interval_updates"]
        == 125
    )
    assert recipe["schedule"]["planned_optimizer_updates"] == 3750
    assert recipe["schedule"]["warmup_steps"] == 188
    history = artifact(run, "history.json")
    assert [v["global_step"] for v in history["validation"]] == [125, 250, 375, 500]
    assert history["optimizer_updates"] == 500
    assert history["selected_fractional_epoch"] == pytest.approx(1 / 3)
    assert history["fractional_epochs_run"] == pytest.approx(4 / 3)
    assert history["stopping_reason"] == "early_stopping"
    assert history["training"][0]["learning_rate"] == 0
    assert history["training"][188]["learning_rate"] == 2e-5
    assert history["training"][-1]["learning_rate"] == pytest.approx(
        1.825379000561482e-5
    )


@pytest.mark.parametrize("failure", ["headroom", "interruption"])
def test_failed_checkpoint_preserves_complete_best_latest_and_original_error(
    tmp_path, tiny_nt, monkeypatch, failure
):
    import shutil

    import torch

    workflow, run = prepared(tmp_path)
    run = workflow.preflight(run, model=ModelSource.fixture(tiny_nt))
    disk_usage = shutil.disk_usage
    save = torch.save
    calls = 0

    def disk_limit(path):
        nonlocal calls
        calls += 1
        if calls == 3:
            return type(disk_usage(path))(100, 100, 0)
        return disk_usage(path)

    def interrupted_save(value, path, *args, **kwargs):
        if Path(path).parent.name == ".checkpoint-3.incomplete":
            raise KeyboardInterrupt("fixture interruption during state write")
        return save(value, path, *args, **kwargs)

    if failure == "headroom":
        monkeypatch.setattr(shutil, "disk_usage", disk_limit)
    else:
        monkeypatch.setattr(torch, "save", interrupted_save)
    expected_error = OSError if failure == "headroom" else KeyboardInterrupt
    with pytest.raises(expected_error, match="headroom|fixture interruption"):
        workflow.fit(run, fixture=FitFixture(validation_losses=(0.1, 0.2, 0.2, 0.2)))
    status = artifact(run, "status.json")
    assert status["state"] == (
        "training_failed" if failure == "headroom" else "training_interrupted"
    )
    assert status["model_completed"] is False
    assert sorted(p.name for p in run.checkpoint_dir.glob("checkpoint-*")) == [
        "checkpoint-1",
        "checkpoint-2",
    ]
    assert artifact(run, "history.json")["optimizer_updates"] == 3
    with pytest.raises(ValueError, match="state/configuration"):
        workflow.fit(run)
    with pytest.raises(ValueError, match="state/configuration"):
        workflow.export_selected(run)


@pytest.mark.parametrize(
    "damage", ["checkpoint", "reload", "mismatch", "nonfinite", "tokenizer"]
)
def test_failed_local_reload_cannot_claim_validated_export(
    tmp_path, tiny_nt, monkeypatch, damage
):
    import torch
    from transformers import EsmForSequenceClassification, EsmTokenizer

    workflow, run = prepared(tmp_path)
    run = workflow.preflight(run, model=ModelSource.fixture(tiny_nt))
    run = workflow.fit(run, fixture=FitFixture(validation_losses=(0.1, 0.2, 0.2, 0.2)))
    load = EsmForSequenceClassification.from_pretrained
    save_tokenizer = EsmTokenizer.save_pretrained

    def damaged_reload(path, *args, **kwargs):
        if Path(path).name == "selected_export":
            if damage == "reload":
                raise OSError("fixture reload failure")
            model = load(path, *args, **kwargs)
            with torch.no_grad():
                model.classifier.out_proj.bias.add_(
                    float("nan") if damage == "nonfinite" else 0.5
                )
            return model
        return load(path, *args, **kwargs)

    def damaged_tokenizer(tokenizer, path, *args, **kwargs):
        result = save_tokenizer(tokenizer, path, *args, **kwargs)
        if Path(path).name == "selected_export":
            (Path(path) / "vocab.txt").write_text("<cls>\n<pad>\n<mask>\n<unk>\n")
        return result

    if damage == "checkpoint":
        (run.checkpoint_dir / "checkpoint-1" / "training_state.pt").unlink()
    elif damage == "tokenizer":
        monkeypatch.setattr(EsmTokenizer, "save_pretrained", damaged_tokenizer)
    else:
        monkeypatch.setattr(
            EsmForSequenceClassification, "from_pretrained", damaged_reload
        )
    with pytest.raises((ValueError, OSError, FloatingPointError)):
        workflow.export_selected(run)
    status = artifact(run, "status.json")
    assert status["state"] == "export_failed"
    assert status["local_export_validated"] is False
    assert status["model_completed"] is False
    assert not (run.local_dir / "export.json").exists()


def test_max_epochs_partial_batch_and_existing_export_preservation(tmp_path, tiny_nt):
    workflow, run = prepared(tmp_path, counts=(35, 2, 1))
    run = workflow.preflight(run, model=ModelSource.fixture(tiny_nt))
    run = workflow.fit(
        run, fixture=FitFixture(validation_losses=tuple(range(20, 0, -1)))
    )
    history = artifact(run, "history.json")
    assert history["optimizer_updates"] == 20
    assert history["epochs_completed"] == 10
    assert history["stopping_reason"] == "max_epochs"
    assert [v["examples"] for v in history["training"]] == [32, 3] * 10
    export = run.local_dir / "selected_export"
    export.mkdir()
    (export / "prior.txt").write_text("preserve me")
    with pytest.raises(FileExistsError, match="already exists"):
        workflow.export_selected(run)
    assert (export / "prior.txt").read_text() == "preserve me"


def test_production_rejects_fixture_training_controls_before_runtime(tmp_path):
    workflow, run = prepared(tmp_path, production=True)
    with pytest.raises(ValueError, match="fixture controls"):
        workflow.fit(run, fixture=FitFixture())
    assert artifact(run, "status.json")["state"] == "prepared"


def test_reload_uses_the_selected_sample_even_when_batch_shape_changes_rounding(
    tmp_path, tiny_nt, monkeypatch
):
    from transformers import EsmForSequenceClassification

    workflow, run = prepared(tmp_path, counts=(8, 16, 1))
    run = workflow.preflight(run, model=ModelSource.fixture(tiny_nt))
    forward = EsmForSequenceClassification.forward

    def shape_sensitive_forward(model, *args, **kwargs):
        output = forward(model, *args, **kwargs)
        # Numerical fault injection at the external model boundary: represent
        # batch-shape-dependent device rounding without a CUDA dependency.
        if not model.training and kwargs["input_ids"].shape[0] > 8:
            output.logits = output.logits + 0.001
        return output

    monkeypatch.setattr(
        EsmForSequenceClassification, "forward", shape_sensitive_forward
    )
    run = workflow.fit(run, fixture=FitFixture(validation_losses=(0.1, 0.2, 0.2, 0.2)))
    run = workflow.export_selected(run)
    history = artifact(run, "history.json")
    report = artifact(run, "export.json")
    assert report["validated"] is True
    assert len(report["validation_source_rows"]) == 8
    assert report["selected_predictions"] == pytest.approx(
        history["reload_validation_predictions"],
        abs=1e-6,
    )
    assert (
        max(
            abs(a - b)
            for a, b in zip(
                report["selected_predictions"],
                history["selected_predictions"][:8],
            )
        )
        > 0.0009
    )
