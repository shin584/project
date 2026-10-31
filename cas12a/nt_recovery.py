"""Strict same-VM checkpoint validation and restoration, loaded only on resume."""

import json
import math
import random
import re

import numpy as np
import torch

from cas12a import nt_runtime as runtime
from cas12a import nt_training as training
from cas12a.nt_data import digest
from cas12a.nt_workflow import RecoveryUnavailable


def _manifest(run, step, execution, recipe):
    path = run.checkpoint_dir / f"checkpoint-{step}"
    if path.is_symlink() or path.resolve().parent != run.checkpoint_dir.resolve():
        raise ValueError("resume checkpoint escaped the run")
    manifest = json.loads((path / "checkpoint.json").read_bytes())
    files = training._files(path)
    del files["checkpoint.json"]
    if (
        manifest["complete"] is not True
        or manifest["global_step"] != step
        or manifest["configuration_sha256"] != execution["configuration_sha256"]
        or manifest["execution_sha256"] != digest(execution)
        or manifest["training_sha256"] != digest(recipe)
        or manifest["files"] != files
        or not {"config.json", "model.safetensors", "training_state.pt"} <= files.keys()
    ):
        raise ValueError("resume checkpoint manifest is incomplete or incompatible")
    return path, manifest


def _validate_progress(payload, step, partition, recipe):
    progress, history = payload["progress"], payload["history"]
    if (
        not {
            "selected_predictions",
            "reload_validation_predictions",
            "selected_fractional_epoch",
            "initial_global_step",
            "initial_optimizer_entries",
            "epochs_completed",
            "fractional_epochs_run",
            "stopping_reason",
        }
        <= history.keys()
    ):
        raise ValueError("resume selection/export history is incomplete")
    train = partition["partitions"]["train"]
    updates = recipe["updates_per_epoch"]
    epoch, within = divmod(step - 1, updates)
    offset = min((within + 1) * runtime.EFFECTIVE_BATCH, len(train))
    generator = np.random.default_rng(runtime.SEED)
    for _ in range(epoch + 1):
        order = generator.permutation(train).tolist()
    validations = history["validation"]
    best, selected, patience = None, None, 0
    for entry in validations:
        loss = entry["loss"]
        if not math.isfinite(loss) or loss < 0:
            raise ValueError("resume validation history is invalid")
        improved = best is None or loss < best
        if entry["improved"] != improved:
            raise ValueError("resume selection history is inconsistent")
        if improved:
            best, selected, patience = loss, entry["global_step"], 0
        else:
            patience += 1
    interval = recipe["validation_interval_updates"]
    selected_entry = next(
        (v for v in validations if v["global_step"] == selected), None
    )
    if (
        selected_entry is None
        or history["selected_fractional_epoch"] != selected_entry["fractional_epoch"]
        or len(history["selected_predictions"])
        != len(partition["partitions"]["validation"])
        or len(history["reload_validation_predictions"])
        != min(8, len(partition["partitions"]["validation"]))
        or not all(
            math.isfinite(v)
            for v in history["selected_predictions"]
            + history["reload_validation_predictions"]
        )
        or history["initial_global_step"] != 0
        or history["initial_optimizer_entries"] != 0
        or history["fractional_epochs_run"] != epoch + offset / len(train)
        or history["epochs_completed"] != math.floor(epoch + offset / len(train))
    ):
        raise ValueError("resume selection/export history is inconsistent")
    if (
        not 0 < step <= recipe["schedule"]["planned_optimizer_updates"]
        or step % interval
        or progress["global_step"] != step
        or progress["epoch_index"] != epoch
        or progress["next_example_offset"] != offset
        or progress["epoch_order"] != order
        or payload["rng"]["data_order"] != generator.bit_generator.state
        or progress["selected_global_step"] != selected
        or progress["best_validation_loss"] != best
        or progress["non_improving_validations"] != patience
        or history["selected_global_step"] != selected
        or history["optimizer_updates"] != step
        or history["training_sha256"] != digest(recipe)
        or history["training_source_rows"] != train
        or history["validation_source_rows"] != partition["partitions"]["validation"]
        or history["reload_validation_source_rows"]
        != partition["partitions"]["validation"][:8]
        or [v["global_step"] for v in validations]
        != list(range(interval, step + 1, interval))
        or [u["global_step"] for u in history["training"]] != list(range(1, step + 1))
        or payload["scheduler"]["last_epoch"] != step
        or not payload["optimizer"]["state"]
    ):
        raise ValueError("resume trainer/callback/update progress is inconsistent")


def restore(run, execution, tokenizer, partition, recipe):
    candidates = [
        int(path.name.split("-")[1])
        for path in run.checkpoint_dir.glob("checkpoint-*")
        if re.fullmatch(r"checkpoint-\d+", path.name)
    ]
    if not candidates:
        raise RecoveryUnavailable("recovery unavailable: no completed local checkpoint")
    step = max(candidates)
    try:
        checkpoint, manifest = _manifest(run, step, execution, recipe)
        payload = torch.load(
            checkpoint / "training_state.pt", map_location="cpu", weights_only=True
        )
        if (
            not {"optimizer", "scheduler", "scaler", "rng", "progress", "history"}
            <= payload.keys()
            or payload["execution_sha256"] != digest(execution)
            or payload["training_sha256"] != digest(recipe)
            or not {"python", "numpy", "torch", "cuda", "data_order"}
            <= payload["rng"].keys()
            or bool(payload["scaler"]) != execution["gradient_scaling"]["enabled"]
        ):
            raise ValueError(
                "resume checkpoint is missing required training/RNG/scaler state"
            )
        _validate_progress(payload, step, partition, recipe)
        history = payload["history"]
        selected = history["selected_global_step"]
        _, best = _manifest(run, selected, execution, recipe)
        if selected != step and digest(best) != history["selected_checkpoint_sha256"]:
            raise ValueError("resume best-validation checkpoint identity changed")
        history["selected_checkpoint_sha256"] = digest(best)
        history["latest_checkpoint_sha256"] = digest(manifest)
        state = runtime.initialize(
            execution["source"],
            execution["settings"],
            train_rows=len(partition["partitions"]["train"]),
            tokenizer=tokenizer,
            checkpoint=checkpoint,
        )
        state.execution = execution
        state.initialization_sha256 = execution["initialization_sha256"]
        state.global_step = step
        expected_lr = [
            base * schedule(step)
            for base, schedule in zip(
                state.scheduler.base_lrs, state.scheduler.lr_lambdas
            )
        ]
        scheduler = state.scheduler.state_dict()
        scheduler.update(last_epoch=step, _step_count=step + 1, _last_lr=expected_lr)
        groups = state.optimizer.state_dict()["param_groups"]
        for group, learning_rate in zip(groups, expected_lr):
            group["lr"] = learning_rate
        if (
            payload["scheduler"] != scheduler
            or payload["optimizer"]["param_groups"] != groups
            or history["initialization_sha256"] != execution["initialization_sha256"]
        ):
            raise ValueError(
                "resume optimizer/scheduler settings or initialization changed"
            )
        scaler = state.scaler.state_dict()
        saved_scaler = payload["scaler"]
        if saved_scaler.keys() != scaler.keys() or any(
            saved_scaler[key] != value
            for key, value in scaler.items()
            if key not in {"scale", "_growth_tracker"}
        ):
            raise ValueError("resume FP16 scaler state is incomplete or incompatible")
        if scaler and (
            not math.isfinite(saved_scaler["scale"])
            or saved_scaler["scale"] <= 0
            or type(saved_scaler["_growth_tracker"]) is not int
            or not 0
            <= saved_scaler["_growth_tracker"]
            < saved_scaler["growth_interval"]
        ):
            raise ValueError("resume FP16 scaler progress is invalid")
        state.optimizer.load_state_dict(payload["optimizer"])
        state.scheduler.load_state_dict(payload["scheduler"])
        state.scaler.load_state_dict(payload["scaler"])
        # ESM's contact-prediction head is registered but never participates in
        # sequence regression, so AdamW legitimately has no moments for it.
        parameters = [
            parameter
            for name, parameter in state.model.named_parameters()
            if not name.startswith("esm.contact_head.")
        ]
        if len(state.optimizer.state) != len(parameters):
            raise ValueError("resume optimizer state is incomplete")
        for parameter in parameters:
            values = state.optimizer.state[parameter]
            if float(values["step"]) != step:
                raise ValueError("resume optimizer update position changed")
            for key in ("exp_avg", "exp_avg_sq"):
                if values[key].shape != parameter.shape:
                    raise ValueError("resume optimizer moment shape changed")
                runtime._finite(values[key], "resume optimizer moment")
        for parameter in state.model.parameters():
            runtime._finite(parameter, "resume model parameter")
        rng = payload["rng"]
        random.setstate(rng["python"])
        numpy = rng["numpy"]
        np.random.set_state(
            (numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:])
        )
        torch.set_rng_state(rng["torch"])
        if execution["settings"]["device"] == "cuda":
            if len(rng["cuda"]) != torch.cuda.device_count():
                raise ValueError("resume CUDA RNG state is incomplete")
            torch.cuda.set_rng_state_all(rng["cuda"])
        # Optimizer tensors have moved to their runtime device; don't retain a
        # second CPU copy throughout a long resumed GPU fit.
        recovered = {key: payload[key] for key in ("history", "progress", "rng")}
        return state, recovered, digest(manifest)
    except (OSError, KeyError, TypeError, RuntimeError, IndexError) as error:
        raise ValueError(
            f"resume checkpoint is missing, corrupt or incompatible: {error}"
        ) from error
