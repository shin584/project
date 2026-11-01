"""Lazy fixed-recipe fit, local resume checkpoints and selected export.

Only NTWorkflow calls this runtime module. Training and reload validation never
build a target mapping containing final-Testset labels.
"""

import gc
import hashlib
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
# Unit roundoff of each autocast compute dtype: half the gap between 1.0 and the
# next representable value. FP16 reaches 4.9e-4, so a single representable step
# near 1.0 is three orders of magnitude wider than any 1e-6 absolute bound.
UNIT_ROUNDOFF = {"float32": 2.0**-24, "float16": 2.0**-11, "bfloat16": 2.0**-9}
# Identical weights do not guarantee an identical kernel or reduction order
# between a live instance and a freshly loaded one, and the drift compounds over
# the backbone's reduction depth. This budget bounds that drift only; loading
# correctness is asserted exactly and separately by parameter/buffer identity,
# so the forward bound never has to carry it.
FORWARD_ULP_BUDGET = 128
# How many mismatching tensors a single failure report names.
MISMATCH_EVIDENCE_LIMIT = 16


class ReloadMismatch(ValueError):
    """A reload comparison failure that carries its own diagnosable evidence.

    The evidence must survive the checkpoint it describes: once a VM is
    reclaimed, this payload is the only way to tell a numerical variation apart
    from a real loading defect.
    """

    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def forward_tolerance(precision):
    """Derive the reload forward tolerance from the frozen execution precision."""
    if precision not in UNIT_ROUNDOFF:
        raise ValueError(f"unsupported execution precision: {precision}")
    roundoff = UNIT_ROUNDOFF[precision]
    rtol = FORWARD_ULP_BUDGET * roundoff
    return {
        "precision": precision,
        "unit_roundoff": roundoff,
        "ulp_budget": FORWARD_ULP_BUDGET,
        "rtol": rtol,
        "atol": rtol / 10.0,
        "basis": (
            "rtol = ulp_budget * unit_roundoff of the autocast compute dtype; "
            "atol anchors near-zero logits at one tenth of rtol. The budget is "
            "a deliberate margin against reduction-order drift, not a measured "
            "error bound. It bounds forward reproducibility only: loading "
            "correctness rests on exact parameter/buffer identity against the "
            "live selection state, so this bound never carries it."
        ),
    }


def _json_number(value):
    """Keep a non-finite magnitude legible: canonical evidence forbids NaN/Inf."""
    value = float(value)
    return value if math.isfinite(value) else repr(value)


def _deviation(actual, expected):
    """Numerical evidence for a prediction comparison, finite and JSON-safe."""
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    if left.shape != right.shape:
        return {
            "comparable": False,
            "actual_shape": list(left.shape),
            "expected_shape": list(right.shape),
        }
    difference = np.abs(left - right)
    nonzero = right != 0
    relative = difference[nonzero] / np.abs(right[nonzero])
    return {
        "comparable": True,
        "count": int(left.size),
        "zero_reference_count": int((~nonzero).sum()),
        "max_absolute_difference": (
            _json_number(difference.max()) if left.size else 0.0
        ),
        "max_relative_difference": (
            _json_number(relative.max()) if relative.size else 0.0
        ),
    }


def _compare_forward(message, comparison, actual, expected, tolerance):
    """Judge forward reproducibility alone, at the precision-derived bound."""
    deviation = _deviation(actual, expected)
    acceptable = deviation["comparable"] and np.allclose(
        actual, expected, atol=tolerance["atol"], rtol=tolerance["rtol"]
    )
    if not acceptable:
        raise ReloadMismatch(
            message,
            {
                "kind": "forward_reproducibility",
                "comparison": comparison,
                "tolerance": tolerance,
                "deviation": deviation,
                "actual_predictions": [_json_number(v) for v in actual],
                "expected_predictions": [_json_number(v) for v in expected],
            },
        )
    return deviation


def weight_identities(model):
    """Exact per-tensor content digests, independent of dtype and of kernels.

    Raw bytes are hashed through a uint8 view, so no dtype needs a NumPy
    equivalent and no value is ever cast or rounded on the way in.
    """
    identities = {}
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        identities[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(
                tensor.flatten().view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }
    return identities


def _identity_mismatches(reloaded, expected):
    mismatches = []
    for name in sorted(set(expected) | set(reloaded)):
        if name not in reloaded:
            mismatches.append({"tensor": name, "reason": "missing_after_reload"})
        elif name not in expected:
            mismatches.append({"tensor": name, "reason": "unexpected_after_reload"})
        elif reloaded[name] != expected[name]:
            mismatches.append(
                {
                    "tensor": name,
                    "reason": "content_differs",
                    "expected": expected[name],
                    "reloaded": reloaded[name],
                }
            )
    return mismatches


def _verify_weight_identity(model, expected, comparison):
    """Reject any load whose parameters or buffers are not bit-identical."""
    identities = weight_identities(model)
    mismatches = _identity_mismatches(identities, expected)
    if mismatches:
        raise ReloadMismatch(
            "loaded parameters/buffers differ from the selection state",
            {
                "kind": "weight_identity",
                "comparison": comparison,
                "expected_tensor_count": len(expected),
                "reloaded_tensor_count": len(identities),
                "mismatching_tensor_count": len(mismatches),
                "mismatches": mismatches[:MISMATCH_EVIDENCE_LIMIT],
            },
        )
    return identities


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
                        # The live model is the only anchor that does not come
                        # from the load path export has to verify. Without it, a
                        # checkpoint load that silently re-initializes a head
                        # would agree with every later reload of itself.
                        selected_weight_identities=weight_identities(state.model),
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
    export,
    rows,
    ids,
    settings,
    tokenization,
    vocabulary_hash,
    expected,
    expected_identities,
):
    """Verify a fresh model and tokenizer against the selected sample reference.

    Weight identity is judged exactly; only the forward pass is judged at the
    precision-derived tolerance.
    """
    tokenizer = AutoTokenizer.from_pretrained(export, local_files_only=True)
    encoded, _ = runtime.tokenize_library(tokenizer, rows)
    if {str(i): encoded[i] for i in ids} != tokenization or digest(
        tokenizer.get_vocab()
    ) != vocabulary_hash:
        raise ValueError("selected export tokenizer identity changed")
    model = EsmForSequenceClassification.from_pretrained(export, local_files_only=True)
    try:
        # On CPU, before any device copy: a re-initialized or partially restored
        # head fails here however its forward pass happens to land.
        _verify_weight_identity(
            model, expected_identities, "export_reload_vs_selection_state"
        )
        model.to(settings["device"])
        predictions = _predict(model, encoded, ids, settings)
    finally:
        del model
        gc.collect()
    deviation = _compare_forward(
        "export reload predictions differ from selected checkpoint",
        "export_reload_vs_selection_state",
        predictions,
        expected,
        forward_tolerance(settings["precision"]),
    )
    return predictions, deviation


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
    tolerance = forward_tolerance(settings["precision"])
    model = EsmForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True
    )
    # Exact, before any forward pass: a missing or renamed key that the loader
    # quietly re-initializes fails here, against the live selection state.
    selected_identities = _verify_weight_identity(
        model,
        history["selected_weight_identities"],
        "selected_checkpoint_vs_selection_state",
    )
    model.to(settings["device"])
    selected = _predict(model, encoded, sample_ids, settings)
    # Exact sample/batch shape avoids false BF16/FP16 mismatches with full eval.
    # The live selection model is gone by now, so only its recorded predictions
    # remain to compare against: this one comparison is forward-only and is
    # judged at the precision's own resolution, never at a global constant.
    selection_deviation = _compare_forward(
        "selected checkpoint predictions differ from selection history",
        "selected_checkpoint_vs_selection_history",
        selected,
        history["reload_validation_predictions"],
        tolerance,
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
    reloaded, reload_deviation = _reload_predictions(
        export,
        rows,
        sample_ids,
        settings,
        tokenization,
        vocabulary_hash,
        selected,
        selected_identities,
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
        "weight_identities": selected_identities,
        "weight_identity_sha256": digest(selected_identities),
        "precision": settings["precision"],
        "forward_tolerance": tolerance,
        "selection_history_deviation": selection_deviation,
        "reload_deviation": reload_deviation,
        "files": files,
        "selected_artifact_sha256": digest(files),
    }


def verify_export(run, data, partition, execution, history, report):
    export = run.local_dir / "selected_export"
    settings = execution["settings"]
    # Each pair is one recorded identity beside the value it must still equal.
    bound = [
        (report["validated"], True),
        (_files(export), report["files"]),
        (digest(report["files"]), report["selected_artifact_sha256"]),
        (report["selected_global_step"], history["selected_global_step"]),
        (report["selected_checkpoint_sha256"], history["selected_checkpoint_sha256"]),
        (report["execution_sha256"], digest(execution)),
        (report["history_sha256"], digest(history)),
        (report["validation_source_rows"], partition["partitions"]["validation"][:8]),
        (digest(report["weight_identities"]), report["weight_identity_sha256"]),
        # The report cannot substitute its own anchor for the recorded one, nor
        # loosen the criterion its precision implies.
        (report["weight_identities"], history["selected_weight_identities"]),
        (report["forward_tolerance"], forward_tolerance(settings["precision"])),
    ]
    if any(actual != expected for actual, expected in bound):
        raise ValueError("selected export artifact identity changed")
    ids = report["validation_source_rows"]
    by_id = {r["source_row"]: r for r in data["rows"]}
    _reload_predictions(
        export,
        [by_id[i] for i in ids],
        ids,
        settings,
        report["validation_tokenization"],
        report["vocabulary_sha256"],
        history["reload_validation_predictions"],
        report["weight_identities"],
    )
