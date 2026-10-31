"""Lazy fixed-recipe fit, local resume checkpoints and selected export.

Only NTWorkflow calls this runtime module. Training and reload validation never
build a target mapping containing final-Testset labels.
"""

import gc
import json
import math
import random
import re
import shutil
from dataclasses import asdict
from uuid import uuid4

import numpy as np
import torch
from transformers import AutoTokenizer, EsmForSequenceClassification

from cas12a import nt_runtime as runtime
from cas12a.nt_data import canonical_bytes, digest

INTERVAL_UPDATES = 125
PATIENCE = 3
RELOAD_ATOL = 1e-6
RELOAD_RTOL = 1e-5


def _files(directory):
    return {
        path.relative_to(directory).as_posix(): runtime._file_hash(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _predict(model, encoded, identities, settings, *, require_finite=True):
    model.eval()
    predictions = []
    with torch.no_grad(), runtime._autocast(settings):
        for offset in range(0, len(identities), settings["evaluation_microbatch"]):
            ids = identities[offset : offset + settings["evaluation_microbatch"]]
            batch = {
                key: torch.tensor(
                    [encoded[i][key] for i in ids], device=settings["device"]
                )
                for key in ("input_ids", "attention_mask")
            }
            values = model(**batch).logits.float().flatten()
            if require_finite:
                runtime._finite(values, "validation/reload prediction")
            predictions.extend(values.cpu().tolist())
    return predictions


def predict_testset(run, rows, execution, report):
    """Inference from the frozen export; no optimizer, selection or fitting."""
    settings = execution["settings"]
    if (
        runtime.runtime_environment(settings["device"], run.status["run_kind"])
        != execution["environment"]
    ):
        raise ValueError("runtime configuration changed since fit")
    export = run.local_dir / "selected_export"
    tokenizer = AutoTokenizer.from_pretrained(export, local_files_only=True)
    if digest(tokenizer.get_vocab()) != report["vocabulary_sha256"]:
        raise ValueError("selected tokenizer identity changed")
    encoded, _ = runtime.tokenize_library(tokenizer, rows)
    model = EsmForSequenceClassification.from_pretrained(export, local_files_only=True)
    try:
        model.to(settings["device"])
        return _predict(
            model,
            encoded,
            [r["source_row"] for r in rows],
            settings,
            require_finite=False,
        )
    finally:
        del model
        gc.collect()
        if settings["device"] == "cuda":
            torch.cuda.empty_cache()


def _update(state, identities, encoded, targets, settings):
    """One successful optimizer update; overflow retries do not advance time."""
    model, optimizer = state.model, state.optimizer
    model.train()
    for retry in range(33):
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for offset in range(0, len(identities), settings["train_microbatch"]):
            ids = identities[offset : offset + settings["train_microbatch"]]
            with runtime._autocast(settings):
                output = model(
                    **runtime._batch(ids, encoded, targets, settings["device"])
                )
                runtime._finite(output.logits, "training output")
                runtime._finite(output.loss, "training loss")
                loss = output.loss * len(ids) / len(identities)
                loss_value += float(loss.detach())
            state.scaler.scale(loss).backward()
        state.scaler.unscale_(optimizer)
        if not all(
            torch.isfinite(p.grad).all().item()
            for p in model.parameters()
            if p.grad is not None
        ):
            if not state.scaler.is_enabled():
                raise FloatingPointError("nonfinite training gradient")
            state.scaler.step(optimizer)
            state.scaler.update()
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        learning_rate = optimizer.param_groups[0]["lr"]
        state.scaler.step(optimizer)
        state.scaler.update()
        state.scheduler.step()
        state.global_step += 1
        for parameter in model.parameters():
            runtime._finite(parameter, "updated parameter")
        return {
            "loss": loss_value,
            "learning_rate": learning_rate,
            "overflow_retries": retry,
        }
    raise FloatingPointError("FP16 gradients remain nonfinite after scale backoff")


def _rng_state(generator):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "data_order": generator.bit_generator.state,
    }


def _tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def _save_checkpoint(state, run, progress, history, generator, recipe):
    root = run.checkpoint_dir
    # Retention alone cannot make room for the next full model + Adam state.
    payload_bytes = _tensor_bytes(state.model.state_dict()) + _tensor_bytes(
        state.optimizer.state_dict()
    )
    required_bytes = math.ceil(payload_bytes * 1.1) + 64 * 1024**2
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < required_bytes:
        raise OSError(
            f"insufficient checkpoint disk headroom: need {required_bytes}, free {free_bytes}"
        )
    step = state.global_step
    temporary = root / f".checkpoint-{step}.incomplete"
    destination = root / f"checkpoint-{step}"
    if temporary.exists():
        temporary = root / f".checkpoint-{step}.{uuid4().hex}.incomplete"
    if destination.exists():
        raise FileExistsError(f"checkpoint destination already exists: {destination}")
    temporary.mkdir()
    state.model.save_pretrained(temporary, safe_serialization=True)
    torch.save(
        {
            "optimizer": state.optimizer.state_dict(),
            "scheduler": state.scheduler.state_dict(),
            "scaler": state.scaler.state_dict(),
            "rng": _rng_state(generator),
            "progress": progress,
            "history": history,
            "training_sha256": digest(recipe),
            "execution_sha256": digest(state.execution),
        },
        temporary / "training_state.pt",
    )
    manifest = {
        "complete": True,
        "global_step": step,
        "configuration_sha256": state.execution["configuration_sha256"],
        "execution_sha256": digest(state.execution),
        "training_sha256": digest(recipe),
        "files": _files(temporary),
        "disk_headroom": {
            "free_bytes_before": free_bytes,
            "required_bytes": required_bytes,
        },
    }
    (temporary / "checkpoint.json").write_bytes(canonical_bytes(manifest))
    temporary.rename(destination)
    retain_checkpoints(run, progress["selected_global_step"], step)
    return digest(manifest)


def retain_checkpoints(run, selected_step, latest_step):
    """Finish retention even when publication succeeded just before interruption."""
    root = run.checkpoint_dir
    # The completed best and latest remain, even when the best is much earlier.
    keep = {f"checkpoint-{selected_step}", f"checkpoint-{latest_step}"}
    for path in root.iterdir():
        if re.fullmatch(r"checkpoint-\d+", path.name) and path.name not in keep:
            if path.is_symlink() or path.resolve().parent != root.resolve():
                raise ValueError("checkpoint pruning target escaped the run")
            shutil.rmtree(path)


def training_recipe(execution, partition, fixture):
    train = partition["partitions"]["train"]
    updates_per_epoch = math.ceil(len(train) / runtime.EFFECTIVE_BATCH)
    interval = fixture.interval_updates if fixture else INTERVAL_UPDATES
    return {
        "run_kind": partition["run_kind"],
        "execution_sha256": digest(execution),
        "configuration_sha256": execution["configuration_sha256"],
        "optimizer": execution["optimizer"],
        "schedule": execution["schedule"],
        "settings": execution["settings"],
        "gradient_scaling": execution["gradient_scaling"],
        "updates_per_epoch": updates_per_epoch,
        "validation_interval_updates": interval,
        "checkpoint_interval_updates": interval,
        "production_interval_updates": INTERVAL_UPDATES,
        "patience": PATIENCE,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "load_best_model_at_end": True,
        "trainer": "cas12a-fixed-recipe-v1",
        "data_order": "numpy.default_rng(42), successive epoch permutations",
        "loss_reduction": "per-example MSE; partial batches weighted by example count",
        "fixture": asdict(fixture) if fixture else None,
    }


def fit(state, data, partition, run, fixture, *, record, recovered=None):
    execution = state.execution
    settings = execution["settings"]
    train = partition["partitions"]["train"]
    validation = partition["partitions"]["validation"]
    reload_ids = validation[:8]
    allowed = set(train + validation)
    rows = [r for r in data["rows"] if r["source_row"] in allowed]
    targets = {r["source_row"]: r["target"] for r in rows}
    encoded, _ = runtime.tokenize_library(state.tokenizer, rows)
    recipe = training_recipe(execution, partition, fixture)
    interval = recipe["validation_interval_updates"]
    record("training.json", recipe)
    history = {
        "training_sha256": digest(recipe),
        "initialization_sha256": state.initialization_sha256,
        "initial_global_step": state.global_step,
        "initial_optimizer_entries": len(state.optimizer.state),
        "training_source_rows": train,
        "validation_source_rows": validation,
        "reload_validation_source_rows": reload_ids,
        "training": [],
        "validation": [],
        "optimizer_updates": 0,
        "epochs_completed": 0,
        "fractional_epochs_run": 0.0,
        "selected_global_step": None,
        "selected_fractional_epoch": None,
        "stopping_reason": "running",
    }
    progress = {
        "global_step": 0,
        "epoch_index": 0,
        "next_example_offset": 0,
        "epoch_order": [],
        "selected_global_step": None,
        "best_validation_loss": None,
        "non_improving_validations": 0,
    }
    generator = np.random.default_rng(runtime.SEED)
    if recovered is not None:
        history, progress = recovered["history"], recovered["progress"]
        generator.bit_generator.state = recovered["rng"]["data_order"]
        history["stopping_reason"] = "running"
    record("history.json", history)
    try:
        if progress["non_improving_validations"] >= PATIENCE:
            history["stopping_reason"] = "early_stopping"
            return history
        for epoch in range(
            progress["epoch_index"], execution["schedule"]["max_epochs"]
        ):
            continuing = (
                bool(progress["epoch_order"]) and epoch == progress["epoch_index"]
            )
            order = (
                progress["epoch_order"]
                if continuing
                else generator.permutation(train).tolist()
            )
            start = progress["next_example_offset"] if continuing else 0
            for offset in range(start, len(order), runtime.EFFECTIVE_BATCH):
                ids = order[offset : offset + runtime.EFFECTIVE_BATCH]
                update = _update(state, ids, encoded, targets, settings)
                fractional_epoch = epoch + min(offset + len(ids), len(order)) / len(
                    order
                )
                history["training"].append(
                    {
                        "global_step": state.global_step,
                        "fractional_epoch": fractional_epoch,
                        "examples": len(ids),
                        **update,
                    }
                )
                history.update(
                    optimizer_updates=state.global_step,
                    epochs_completed=math.floor(fractional_epoch),
                    fractional_epochs_run=fractional_epoch,
                )
                progress.update(
                    global_step=state.global_step,
                    epoch_index=epoch,
                    next_example_offset=offset + len(ids),
                    epoch_order=order,
                )
                if state.global_step % interval:
                    continue
                predictions = _predict(state.model, encoded, validation, settings)
                measured_loss = float(
                    np.mean(
                        [
                            (prediction - targets[i]) ** 2
                            for i, prediction in zip(validation, predictions)
                        ]
                    )
                )
                if not math.isfinite(measured_loss):
                    raise FloatingPointError("nonfinite validation loss")
                loss = measured_loss
                if fixture and fixture.validation_losses is not None:
                    index = len(history["validation"])
                    if index >= len(fixture.validation_losses):
                        raise ValueError(
                            "controlled fixture validation history exhausted"
                        )
                    loss = fixture.validation_losses[index]
                improved = (
                    progress["best_validation_loss"] is None
                    or loss < progress["best_validation_loss"]
                )
                if improved:
                    progress.update(
                        selected_global_step=state.global_step,
                        best_validation_loss=loss,
                        non_improving_validations=0,
                    )
                    history.update(
                        selected_global_step=state.global_step,
                        selected_fractional_epoch=fractional_epoch,
                        selected_predictions=predictions,
                        reload_validation_predictions=_predict(
                            state.model, encoded, reload_ids, settings
                        ),
                    )
                else:
                    progress["non_improving_validations"] += 1
                history["validation"].append(
                    {
                        "global_step": state.global_step,
                        "fractional_epoch": fractional_epoch,
                        "loss": loss,
                        "measured_loss": measured_loss,
                        "improved": improved,
                    }
                )
                checkpoint_hash = _save_checkpoint(
                    state, run, progress, history, generator, recipe
                )
                if improved:
                    history["selected_checkpoint_sha256"] = checkpoint_hash
                history["latest_checkpoint_sha256"] = checkpoint_hash
                record("history.json", history)
                if progress["non_improving_validations"] >= PATIENCE:
                    history["stopping_reason"] = "early_stopping"
                    return history
            progress.update(
                epoch_index=epoch + 1, next_example_offset=0, epoch_order=[]
            )
        if history["selected_global_step"] is None:
            raise ValueError(
                "no validation checkpoint; use a shorter explicit fixture interval"
            )
        history["stopping_reason"] = "max_epochs"
        return history
    except (Exception, KeyboardInterrupt):
        history["stopping_reason"] = "interrupted_or_failed"
        raise
    finally:
        # Keep the original training error if evidence storage also fails.
        import sys

        active_error = sys.exception()
        try:
            record("history.json", history)
        except OSError as storage_error:
            if active_error is None:
                raise
            active_error.add_note(f"Could not record training history: {storage_error}")


def _verified_checkpoint(run, history, execution):
    checkpoint = run.checkpoint_dir / f"checkpoint-{history['selected_global_step']}"
    manifest = json.loads((checkpoint / "checkpoint.json").read_bytes())
    files = _files(checkpoint)
    del files["checkpoint.json"]
    if (
        digest(manifest) != history["selected_checkpoint_sha256"]
        or not manifest["complete"]
        or manifest["files"] != files
        or manifest["global_step"] != history["selected_global_step"]
        or manifest["execution_sha256"] != digest(execution)
        or manifest["training_sha256"] != history["training_sha256"]
    ):
        raise ValueError("selected checkpoint identity changed or incomplete")
    return checkpoint


def _reload_predictions(
    export, rows, ids, settings, tokenization, vocabulary_hash, expected
):
    """Verify a fresh model and tokenizer against the selected sample reference."""
    tokenizer = AutoTokenizer.from_pretrained(export, local_files_only=True)
    encoded, _ = runtime.tokenize_library(tokenizer, rows)
    if {str(i): encoded[i] for i in ids} != tokenization or digest(
        tokenizer.get_vocab()
    ) != vocabulary_hash:
        raise ValueError("selected export tokenizer identity changed")
    model = EsmForSequenceClassification.from_pretrained(export, local_files_only=True)
    try:
        model.to(settings["device"])
        predictions = _predict(model, encoded, ids, settings)
    finally:
        del model
        gc.collect()
    if not np.allclose(predictions, expected, atol=RELOAD_ATOL, rtol=RELOAD_RTOL):
        raise ValueError("export reload predictions differ from selected checkpoint")
    return predictions


def export_selected(run, data, partition, execution, history):
    settings = execution["settings"]
    if (
        runtime.runtime_environment(settings["device"], partition["run_kind"])
        != execution["environment"]
    ):
        raise ValueError("runtime configuration changed since fit")
    checkpoint = _verified_checkpoint(run, history, execution)
    export = run.local_dir / "selected_export"
    if export.exists():
        raise FileExistsError(f"selected export destination already exists: {export}")
    tokenizer = runtime.load_tokenizer(execution["source"])
    sample_ids = partition["partitions"]["validation"][:8]
    if sample_ids != history["reload_validation_source_rows"]:
        raise ValueError("selected reload sample identity changed")
    by_id = {r["source_row"]: r for r in data["rows"]}
    rows = [by_id[i] for i in sample_ids]
    encoded, _ = runtime.tokenize_library(tokenizer, rows)
    model = EsmForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True
    )
    model.to(settings["device"])
    selected = _predict(model, encoded, sample_ids, settings)
    # Exact sample/batch shape avoids false BF16/FP16 mismatches with full eval.
    expected = history["reload_validation_predictions"]
    if not np.allclose(selected, expected, atol=RELOAD_ATOL, rtol=RELOAD_RTOL):
        raise ValueError(
            "selected checkpoint predictions differ from selection history"
        )
    export.mkdir()
    model.save_pretrained(export, safe_serialization=True)
    tokenizer.save_pretrained(export)
    files = _files(export)
    del model
    gc.collect()
    if settings["device"] == "cuda":
        torch.cuda.empty_cache()
    # Reload both components from the export, independently of the source model.
    tokenization = {str(i): encoded[i] for i in sample_ids}
    vocabulary_hash = digest(tokenizer.get_vocab())
    reloaded = _reload_predictions(
        export,
        rows,
        sample_ids,
        settings,
        tokenization,
        vocabulary_hash,
        selected,
    )
    if _files(export) != files:
        raise ValueError("export artifact identity changed during reload")
    return {
        "validated": True,
        "durable_export_validated": False,
        "selected_global_step": history["selected_global_step"],
        "selected_fractional_epoch": history["selected_fractional_epoch"],
        "selected_checkpoint_sha256": history["selected_checkpoint_sha256"],
        "execution_sha256": digest(execution),
        "history_sha256": digest(history),
        "source": execution["source"],
        "validation_source_rows": sample_ids,
        "validation_tokenization": tokenization,
        "vocabulary_sha256": vocabulary_hash,
        "tokenization_matches": True,
        "selected_predictions": selected,
        "reloaded_predictions": reloaded,
        "atol": RELOAD_ATOL,
        "rtol": RELOAD_RTOL,
        "max_absolute_difference": max(abs(a - b) for a, b in zip(selected, reloaded)),
        "files": files,
        "selected_artifact_sha256": digest(files),
    }


def verify_export(run, data, partition, execution, history, report):
    export = run.local_dir / "selected_export"
    if (
        not report["validated"]
        or _files(export) != report["files"]
        or digest(report["files"]) != report["selected_artifact_sha256"]
        or report["selected_global_step"] != history["selected_global_step"]
        or report["selected_checkpoint_sha256"] != history["selected_checkpoint_sha256"]
        or report["execution_sha256"] != digest(execution)
        or report["history_sha256"] != digest(history)
        or report["validation_source_rows"] != partition["partitions"]["validation"][:8]
    ):
        raise ValueError("selected export artifact identity changed")
    ids = report["validation_source_rows"]
    by_id = {r["source_row"]: r for r in data["rows"]}
    _reload_predictions(
        export,
        [by_id[i] for i in ids],
        ids,
        execution["settings"],
        report["validation_tokenization"],
        report["vocabulary_sha256"],
        history["reload_validation_predictions"],
    )
