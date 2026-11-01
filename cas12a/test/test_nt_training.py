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
    assert (
        report["reload_deviation"]["max_absolute_difference"]
        <= report["forward_tolerance"]["atol"]
    )
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


def force_precision(monkeypatch, precision):
    """Relabel the frozen precision of an offline CPU run.

    The fixture always computes in float32; only the criterion derived from the
    recorded precision changes, which is what these cases exercise. Production
    float16 is forced by the GPU itself and cannot be reached offline.
    """
    from cas12a import nt_runtime as runtime

    environment = runtime.runtime_environment

    def relabeled(device, run_kind):
        return dict(environment(device, run_kind), precision=precision)

    monkeypatch.setattr(runtime, "runtime_environment", relabeled)


def shift_logits(monkeypatch, target, deviation):
    """Vary a reloaded model's forward pass while leaving every weight identical."""
    import torch
    from transformers import EsmForSequenceClassification

    load = EsmForSequenceClassification.from_pretrained

    def perturbed(path, *args, **kwargs):
        model = load(path, *args, **kwargs)
        if Path(path).name == target:
            inner = model.forward

            def shifted(*arguments, **keywords):
                output = inner(*arguments, **keywords)
                output.logits = output.logits + deviation
                return output

            model.forward = shifted
        return model

    monkeypatch.setattr(EsmForSequenceClassification, "from_pretrained", perturbed)
    return torch


def test_forward_tolerance_is_derived_per_precision_and_records_its_basis():
    from cas12a import nt_training

    derived = {
        precision: nt_training.forward_tolerance(precision)
        for precision in ("float32", "float16", "bfloat16")
    }
    assert [derived[p]["unit_roundoff"] for p in derived] == [2**-24, 2**-11, 2**-9]
    for precision, tolerance in derived.items():
        assert tolerance["precision"] == precision
        assert tolerance["ulp_budget"] == 128
        assert tolerance["rtol"] == 128 * tolerance["unit_roundoff"]
        assert tolerance["atol"] == tolerance["rtol"] / 10
        assert "unit_roundoff" in tolerance["basis"]
    # A loosened global constant is not the mechanism: each precision's bound
    # follows its own representable-value spacing, strictly ordered.
    assert derived["float32"]["atol"] < derived["float16"]["atol"]
    assert derived["float16"]["atol"] < derived["bfloat16"]["atol"]
    # Two FP16 steps near 1.0 are structurally unjudgeable at the float32 bound.
    two_fp16_steps = 4 * 2**-11
    for precision, acceptable in (("float32", False), ("float16", True)):
        tolerance = derived[precision]
        assert (two_fp16_steps <= tolerance["atol"] + tolerance["rtol"]) is acceptable
    with pytest.raises(ValueError, match="unsupported execution precision"):
        nt_training.forward_tolerance("float8")


@pytest.mark.parametrize(
    "comparison",
    ["selected_checkpoint_vs_selection_history", "export_reload_vs_selection_state"],
)
def test_fp16_forward_variation_passes_where_the_float32_bound_cannot(
    tmp_path, tiny_nt, monkeypatch, comparison
):
    """Identical weights, a 1e-3 forward shift: judged only by the precision."""
    from cas12a import nt_training

    selection = comparison == "selected_checkpoint_vs_selection_history"
    force_precision(monkeypatch, "float16")
    workflow, run = prepared(tmp_path)
    run = workflow.preflight(run, model=ModelSource.fixture(tiny_nt))
    run = workflow.fit(run, fixture=FitFixture(validation_losses=(0.1, 0.2, 0.2, 0.2)))
    assert artifact(run, "execution.json")["settings"]["precision"] == "float16"
    shift_logits(monkeypatch, "checkpoint-1" if selection else "selected_export", 1e-3)

    run = workflow.export_selected(run)

    report = artifact(run, "export.json")
    assert run.status["state"] == "export_validated"
    assert report["validated"] is True
    assert report["precision"] == "float16"
    assert report["forward_tolerance"] == nt_training.forward_tolerance("float16")
    assert report["forward_tolerance"]["atol"] == pytest.approx(0.00625)
    deviation = report[
        "selection_history_deviation" if selection else "reload_deviation"
    ]
    assert deviation["max_absolute_difference"] == pytest.approx(1e-3, rel=1e-3)
    assert deviation["comparable"] is True
    # Weight identity carries loading correctness, so it stays exact even here.
    assert (
        nt_training.digest(report["weight_identities"])
        == (report["weight_identity_sha256"])
    )
    assert "classifier.out_proj.weight" in report["weight_identities"]
    run = workflow.verify_export(run)
    assert run.status["local_export_validated"] is True


@pytest.mark.parametrize(
    "comparison",
    ["selected_checkpoint_vs_selection_history", "export_reload_vs_selection_state"],
)
def test_float32_reload_mismatch_preserves_both_arrays_and_the_applied_bound(
    tmp_path, tiny_nt, monkeypatch, comparison
):
    from cas12a import nt_training

    selection = comparison == "selected_checkpoint_vs_selection_history"
    workflow, run = prepared(tmp_path)
    run = workflow.preflight(run, model=ModelSource.fixture(tiny_nt))
    run = workflow.fit(run, fixture=FitFixture(validation_losses=(0.1, 0.2, 0.2, 0.2)))
    shift_logits(monkeypatch, "checkpoint-1" if selection else "selected_export", 1e-3)

    with pytest.raises(nt_training.ReloadMismatch, match="differ from"):
        workflow.export_selected(run)

    status = artifact(run, "status.json")
    assert status["state"] == "export_failed"
    assert status["local_export_validated"] is False
    assert status["model_completed"] is False
    assert not (run.local_dir / "export.json").exists()
    journal = artifact(run, "export_failure.json")
    assert journal["run_id"] == run.run_id
    assert len(journal["failures"]) == 1
    evidence = journal["failures"][0]["evidence"]
    assert evidence["kind"] == "forward_reproducibility"
    assert evidence["comparison"] == comparison
    assert evidence["tolerance"] == nt_training.forward_tolerance("float32")
    assert evidence["deviation"]["max_absolute_difference"] == pytest.approx(
        1e-3, rel=1e-3
    )
    assert evidence["deviation"]["max_relative_difference"] > 0
    # Both sides survive the checkpoint they came from.
    sample = artifact(run, "partition.json")["partitions"]["validation"][:8]
    assert len(evidence["actual_predictions"]) == len(sample)
    assert len(evidence["expected_predictions"]) == len(sample)
    assert evidence["deviation"]["count"] == len(sample)
    assert max(
        abs(a - b)
        for a, b in zip(
            evidence["actual_predictions"], evidence["expected_predictions"]
        )
    ) == pytest.approx(1e-3, rel=1e-3)
    assert "ReloadMismatch" in journal["failures"][0]["error"]
    assert [e["status"]["state"] for e in artifact(run, "lifecycle.json")["events"]][
        -1
    ] == "export_failed"


def damage_loaded_weights(monkeypatch, target, damage):
    """Alter a loaded model's weights while its forward pass keeps agreeing.

    The replayed forward comes from a second, pristine load, so every case here
    is judged by weight identity alone and nothing else can reject it.
    """
    import torch
    from transformers import EsmForSequenceClassification

    load = EsmForSequenceClassification.from_pretrained

    def damaged(path, *args, **kwargs):
        model = load(path, *args, **kwargs)
        if Path(path).name != target:
            return model
        model.forward = load(path, *args, **kwargs).forward
        with torch.no_grad():
            if damage == "head":
                model.classifier.out_proj.weight.normal_()
            elif damage == "encoder":
                model.esm.encoder.layer[0].attention.self.query.weight.add_(0.25)
            elif damage == "missing":
                model.state_dict = lambda *a, **k: {
                    name: value
                    for name, value in type(model).state_dict(model).items()
                    if name != "classifier.dense.bias"
                }
            elif damage == "unexpected":
                original = type(model).state_dict(model)
                model.state_dict = lambda *a, **k: {
                    **original,
                    "classifier.smuggled": torch.zeros(1),
                }
        return model

    monkeypatch.setattr(EsmForSequenceClassification, "from_pretrained", damaged)


@pytest.mark.parametrize(
    "damage,tensor,reason",
    [
        ("head", "classifier.out_proj.weight", "content_differs"),
        (
            "encoder",
            "esm.encoder.layer.0.attention.self.query.weight",
            "content_differs",
        ),
        ("missing", "classifier.dense.bias", "missing_after_reload"),
        ("unexpected", "classifier.smuggled", "unexpected_after_reload"),
    ],
)
@pytest.mark.parametrize("target", ["selected_export", "checkpoint-1"])
def test_weight_identity_rejects_each_defect_even_when_the_forward_agrees(
    tmp_path, tiny_nt, monkeypatch, damage, tensor, reason, target
):
    """Both load paths are anchored on the live selection state, not each other."""
    from cas12a import nt_training

    workflow, run = prepared(tmp_path)
    run = workflow.preflight(run, model=ModelSource.fixture(tiny_nt))
    run = workflow.fit(run, fixture=FitFixture(validation_losses=(0.1, 0.2, 0.2, 0.2)))
    damage_loaded_weights(monkeypatch, target, damage)

    with pytest.raises(nt_training.ReloadMismatch, match="parameters/buffers differ"):
        workflow.export_selected(run)

    assert artifact(run, "status.json")["state"] == "export_failed"
    assert artifact(run, "status.json")["local_export_validated"] is False
    evidence = artifact(run, "export_failure.json")["failures"][0]["evidence"]
    assert evidence["kind"] == "weight_identity"
    assert evidence["comparison"] == (
        "selected_checkpoint_vs_selection_state"
        if target == "checkpoint-1"
        else "export_reload_vs_selection_state"
    )
    assert evidence["mismatching_tensor_count"] == 1
    mismatch = evidence["mismatches"][0]
    assert mismatch["tensor"] == tensor
    assert mismatch["reason"] == reason
    if reason == "content_differs":
        assert mismatch["expected"]["sha256"] != mismatch["reloaded"]["sha256"]
        assert mismatch["expected"]["shape"] == mismatch["reloaded"]["shape"]


def test_selection_anchor_is_recorded_from_the_live_model_and_binds_the_export(
    tmp_path, tiny_nt
):
    from cas12a import nt_training

    workflow, run = prepared(tmp_path)
    run = workflow.preflight(run, model=ModelSource.fixture(tiny_nt))
    run = workflow.fit(run, fixture=FitFixture(validation_losses=(0.1, 0.2, 0.2, 0.2)))
    recorded = artifact(run, "history.json")["selected_weight_identities"]
    # Taken from the live training model at selection, before any save/reload.
    assert "classifier.out_proj.weight" in recorded
    assert set(recorded["classifier.out_proj.weight"]) == {"shape", "dtype", "sha256"}
    assert recorded["classifier.out_proj.weight"]["dtype"] == "torch.float32"

    run = workflow.export_selected(run)

    report = artifact(run, "export.json")
    assert report["weight_identities"] == recorded
    assert report["weight_identity_sha256"] == nt_training.digest(recorded)
    # A self-consistent report that swaps in its own anchor is still refused:
    # the recorded history, not the report, decides what the export must match.
    swapped = dict(report)
    swapped["weight_identities"] = {
        name: dict(value, sha256="0" * 64) for name, value in recorded.items()
    }
    swapped["weight_identity_sha256"] = nt_training.digest(swapped["weight_identities"])
    status = artifact(run, "status.json")
    status["export_sha256"] = nt_training.digest(swapped)
    for directory in (run.local_dir, run.durable_dir):
        (directory / "export.json").write_text(json.dumps(swapped))
        (directory / "status.json").write_text(json.dumps(status))
    with pytest.raises(ValueError, match="artifact identity changed"):
        workflow.verify_export(run)
    assert artifact(run, "status.json")["state"] == "export_failed"


def test_retry_export_revalidates_the_same_training_without_refitting(
    tmp_path, tiny_nt, monkeypatch
):
    from cas12a import nt_training

    workflow, run = prepared(tmp_path)
    run = workflow.preflight(run, model=ModelSource.fixture(tiny_nt))
    run = workflow.fit(run, fixture=FitFixture(validation_losses=(0.1, 0.2, 0.2, 0.2)))
    completed = artifact(run, "status.json")
    checkpoints = {
        path.name: sorted(p.name for p in path.iterdir())
        for path in run.checkpoint_dir.glob("checkpoint-*")
    }
    with monkeypatch.context() as patched:
        shift_logits(patched, "checkpoint-1", 1e-3)
        with pytest.raises(nt_training.ReloadMismatch):
            workflow.export_selected(run)
    failed = artifact(run, "status.json")
    assert failed["state"] == "export_failed"
    with pytest.raises(ValueError, match="state/configuration"):
        workflow.export_selected(run)

    run = workflow.retry_export(run)

    assert run.status["state"] == "export_validated"
    assert run.status["export_retry"] is True
    assert "error" not in artifact(run, "status.json")
    report = artifact(run, "export.json")
    assert report["validated"] is True
    assert report["selected_global_step"] == completed["selected_global_step"]
    # No refit: the same selected checkpoint and history back the retried export.
    assert report["history_sha256"] == nt_training.digest(artifact(run, "history.json"))
    assert (
        artifact(run, "status.json")["history_sha256"] == (completed["history_sha256"])
    )
    assert (
        report["selected_checkpoint_sha256"]
        == (completed["selected_checkpoint_sha256"])
    )
    assert {
        path.name: sorted(p.name for p in path.iterdir())
        for path in run.checkpoint_dir.glob("checkpoint-*")
    } == checkpoints
    # The failure that forced the retry is still diagnosable afterwards.
    assert len(artifact(run, "export_failure.json")["failures"]) == 1
    states = [e["status"]["state"] for e in artifact(run, "lifecycle.json")["events"]]
    assert "export_failed" in states and states[-1] == "export_validated"
    run = workflow.verify_export(run)
    assert run.status["local_export_validated"] is True


def test_retry_export_refuses_a_partial_destination_and_a_wrong_state(
    tmp_path, tiny_nt, monkeypatch
):
    from cas12a import nt_training

    workflow, run = prepared(tmp_path)
    run = workflow.preflight(run, model=ModelSource.fixture(tiny_nt))
    run = workflow.fit(run, fixture=FitFixture(validation_losses=(0.1, 0.2, 0.2, 0.2)))
    with pytest.raises(ValueError, match="state/configuration"):
        workflow.retry_export(run)
    with monkeypatch.context() as patched:
        shift_logits(patched, "selected_export", 1e-3)
        with pytest.raises(nt_training.ReloadMismatch):
            workflow.export_selected(run)
    export = run.local_dir / "selected_export"
    assert export.is_dir()  # The failure left weights at the destination.
    weights = (export / "model.safetensors").read_bytes()

    with pytest.raises(FileExistsError, match="remove the previous export attempt"):
        workflow.retry_export(run)

    assert (export / "model.safetensors").read_bytes() == weights
    assert artifact(run, "status.json")["state"] == "export_failed"
    import shutil

    shutil.rmtree(export)
    run = workflow.retry_export(run)
    assert run.status["state"] == "export_validated"
    with monkeypatch.context() as patched:
        shift_logits(patched, "selected_export", 1e-3)
        with pytest.raises(nt_training.ReloadMismatch):
            workflow.verify_export(run)
    # A second failure appends rather than replacing the first one's evidence.
    journal = artifact(run, "export_failure.json")
    assert len(journal["failures"]) == 2
    assert [f["evidence"]["comparison"] for f in journal["failures"]] == [
        "export_reload_vs_selection_state"
    ] * 2
    assert artifact(run, "export.json")["validated"] is False


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
