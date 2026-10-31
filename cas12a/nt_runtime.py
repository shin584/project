"""Lazy NT runtime: original initialization, lossless tokens and resource probes.

Only the workflow imports this module, when a runtime lifecycle operation is
requested. No final-Testset labels enter a resource probe.
"""

import gc
import hashlib
import importlib.metadata
import math
import platform
import random
import re
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, EsmForSequenceClassification

from cas12a.nt_data import canonical_bytes, digest

BACKBONE = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
MAX_LENGTH = 12
EFFECTIVE_BATCH = 32
SEED = 42


class ResourceConstraint(RuntimeError):
    """The agreed full fine-tuning recipe cannot execute on this runtime."""


@dataclass
class FitInitialization:
    """Fresh, unstepped state; callers must retain the frozen execution recipe."""

    model: object
    tokenizer: object
    optimizer: object
    scheduler: object
    scaler: object
    execution: dict
    initialization_sha256: str
    global_step: int = 0


def _directory_identity(directory):
    paths = sorted(path for path in directory.rglob("*") if path.is_file())
    return digest(
        {str(path.relative_to(directory)): _file_hash(path) for path in paths}
    )


def _file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def resolve_source(source, run_kind):
    if source.fixture_dir is not None:
        if run_kind != "fixture":
            raise ValueError("production cannot use a local fixture model/tokenizer")
        directory = Path(source.fixture_dir).resolve()
        return {
            "kind": "fixture",
            "path": str(directory),
            "revision": _directory_identity(directory),
        }
    from huggingface_hub import HfApi

    revision = HfApi().model_info(BACKBONE, revision=source.revision).sha
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("backbone revision must resolve to an immutable commit")
    return {"kind": "production", "repository": BACKBONE, "revision": revision}


def _load_arguments(source):
    if source["kind"] == "fixture":
        path = Path(source["path"])
        if _directory_identity(path) != source["revision"]:
            raise ValueError("fixture model/tokenizer identity changed")
        return str(path), {"local_files_only": True}
    if source["repository"] != BACKBONE or not re.fullmatch(
        r"[0-9a-f]{40}", source["revision"]
    ):
        raise ValueError("require original generic NT and immutable revision")
    return BACKBONE, {"revision": source["revision"], "trust_remote_code": False}


def load_tokenizer(source):
    location, options = _load_arguments(source)
    return AutoTokenizer.from_pretrained(location, **options)


def tokenize_library(tokenizer, rows):
    """Check every usable input, including Testset sequences but no test labels."""
    encoded, lengths = {}, Counter()
    special_ids = set(tokenizer.all_special_ids)
    for row in rows:
        ids = tokenizer(
            row["sequence"], add_special_tokens=True, padding=False, truncation=False
        )["input_ids"]
        if tokenizer.unk_token_id in ids:
            raise ValueError(f"unknown token at source row {row['source_row']}")
        tokens = tokenizer.convert_ids_to_tokens(
            [i for i in ids if i not in special_ids]
        )
        if "".join(tokens) != row["sequence"]:
            raise ValueError(
                f"incomplete token reconstruction at source row {row['source_row']}"
            )
        if len(ids) > MAX_LENGTH:
            raise ValueError(
                f"encoded input exceeds 12 positions at source row {row['source_row']}"
            )
        lengths[len(ids)] += 1
        padded = tokenizer(
            row["sequence"],
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=False,
            return_attention_mask=True,
        )
        if (
            len(padded["input_ids"]) != MAX_LENGTH
            or [
                i
                for i, mask in zip(padded["input_ids"], padded["attention_mask"])
                if mask
            ]
            != ids
        ):
            raise ValueError("padding changed the complete encoded sequence")
        encoded[row["source_row"]] = {
            key: padded[key] for key in ("input_ids", "attention_mask")
        }
    metadata = {
        "checked_sequences": len(rows),
        "max_length": MAX_LENGTH,
        "truncation": False,
        "unpadded_length_distribution": {str(k): v for k, v in sorted(lengths.items())},
        "vocabulary_size": len(tokenizer),
        "vocabulary_sha256": digest(tokenizer.get_vocab()),
        "special_tokens": {k: str(v) for k, v in tokenizer.special_tokens_map.items()},
        "special_token_ids": sorted(special_ids),
        "tokenizer_class": type(tokenizer).__name__,
    }
    return encoded, metadata


def runtime_environment(device, run_kind):
    if device is None:
        device = "cpu" if run_kind == "fixture" else "cuda"
    if device not in ("cpu", "cuda") or (run_kind == "production" and device != "cuda"):
        raise ResourceConstraint("production preflight requires a CUDA GPU")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ResourceConstraint(
                "CUDA GPU unavailable; GPU preflight was not executed"
            )
        torch.cuda.set_device(0)
        name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        # Native BF16 requires Ampere or later; emulation is not sufficient.
        bf16 = capability[0] >= 8 and torch.cuda.is_bf16_supported()
        if capability < (7, 0):
            raise ResourceConstraint("GPU lacks the supported mixed-precision runtime")
        precision = "bfloat16" if bf16 and "T4" not in name else "float16"
        gpu = {
            "name": name,
            "capability": list(capability),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "native_bfloat16": bf16,
        }
    else:
        precision, gpu = "float32", None
    packages = {}
    for package in (
        "torch",
        "transformers",
        "tokenizers",
        "huggingface-hub",
        "numpy",
        "safetensors",
        "pandas",
    ):
        packages[package] = importlib.metadata.version(package)
    return {
        "device": device,
        "precision": precision,
        "gpu": gpu,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "packages": packages,
    }


def _seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _weight_identity(model):
    hasher = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        hasher.update(canonical_bytes([name, list(value.shape), str(value.dtype)]))
        hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def initialize(source, settings, *, train_rows, tokenizer=None, checkpoint=None):
    """Build runtime components from generic or verified recovery weights."""
    gc.collect()
    if settings["device"] == "cuda":
        torch.cuda.empty_cache()
    _seed()
    location, options = _load_arguments(source)
    config = AutoConfig.from_pretrained(location, **options)
    if config.model_type != "esm":
        raise ValueError("require an NT-compatible ESM backbone")
    if source["kind"] == "production" and (
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.vocab_size,
    ) != (1280, 24, 20, 4105):
        raise ValueError("original NT 500m model/config contract mismatch")
    config.num_labels = 1
    config.problem_type = "regression"
    if checkpoint is None:
        model = EsmForSequenceClassification.from_pretrained(
            location, config=config, **options
        )
    else:
        model, loading = EsmForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True, output_loading_info=True
        )
        # Serialization changes metadata and the task architecture name. All
        # operational config, including dropout and token IDs, must still match.
        metadata = {
            "architectures",
            "_name_or_path",
            "transformers_version",
            "dtype",
            "torch_dtype",
        }
        expected = {k: v for k, v in config.to_dict().items() if k not in metadata}
        actual = {k: v for k, v in model.config.to_dict().items() if k not in metadata}
        if any(loading.values()) or actual != expected:
            raise ValueError(
                "resume checkpoint model/config is incomplete or incompatible"
            )
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("all encoder/head parameters must be trainable")
    if settings["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    initial_hash = _weight_identity(model)
    model.to(settings["device"])
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
    )
    planned_updates = math.ceil(train_rows / EFFECTIVE_BATCH) * 10
    warmup_steps = math.ceil(0.05 * planned_updates)

    def linear_schedule(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(
            0.0, (planned_updates - step) / max(1, planned_updates - warmup_steps)
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, linear_schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=settings["precision"] == "float16")
    return FitInitialization(
        model,
        tokenizer or load_tokenizer(source),
        optimizer,
        scheduler,
        scaler,
        settings,
        initial_hash,
    )


def _autocast(settings):
    if settings["device"] == "cpu":
        return nullcontext()
    return torch.autocast("cuda", dtype=getattr(torch, settings["precision"]))


def _batch(identities, encoded, targets, device):
    values = {
        key: torch.tensor([encoded[i][key] for i in identities], device=device)
        for key in ("input_ids", "attention_mask")
    }
    values["labels"] = torch.tensor(
        [targets[i] for i in identities], dtype=torch.float32, device=device
    )
    return values


def _finite(value, description):
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"nonfinite {description}")


def _probe(source, settings, encoded, targets, train, validation, attempt):
    state = initialize(source, settings, train_rows=len(train))
    attempt["initialization_sha256"] = state.initialization_sha256
    attempt["optimizer_defaults"] = state.optimizer.defaults
    attempt["scheduler"] = {
        "planned_optimizer_updates": math.ceil(len(train) / EFFECTIVE_BATCH) * 10,
        "warmup_steps": math.ceil(0.05 * math.ceil(len(train) / EFFECTIVE_BATCH) * 10),
        "initial_last_epoch": state.scheduler.last_epoch,
    }
    attempt["gradient_scaling"] = {
        "enabled": state.scaler.is_enabled(),
        "initial_state": state.scaler.state_dict(),
    }
    attempt["training_losses"] = []
    model, optimizer = state.model, state.optimizer
    attempt["phase"] = "training"
    train_started = time.perf_counter()
    model.train()
    order = np.random.default_rng(SEED).permutation(train).tolist()
    update = 0
    attempt["scaler_overflow_retries"] = 0
    while update < 2:
        update_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        # Cycle tiny fixtures; production uses 64 distinct ordered training rows.
        identities = [
            order[(update * EFFECTIVE_BATCH + i) % len(order)]
            for i in range(EFFECTIVE_BATCH)
        ]
        for offset in range(0, EFFECTIVE_BATCH, settings["train_microbatch"]):
            ids = identities[offset : offset + settings["train_microbatch"]]
            with _autocast(settings):
                output = model(**_batch(ids, encoded, targets, settings["device"]))
                _finite(output.logits, "training output")
                _finite(output.loss, "training loss")
                # Weight by examples, never average microbatch means equally.
                loss = output.loss * len(ids) / EFFECTIVE_BATCH
                update_loss += float(loss.detach())
            state.scaler.scale(loss).backward()
        state.scaler.unscale_(optimizer)
        gradients_finite = all(
            torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        if not gradients_finite:
            if not state.scaler.is_enabled():
                raise FloatingPointError("nonfinite gradient")
            # Native FP16 scaling can overflow before its scale stabilizes.
            # GradScaler skips this step and reduces the scale; neither Adam
            # progress nor the scheduler may count the skipped update.
            state.scaler.step(optimizer)
            state.scaler.update()
            attempt["scaler_overflow_retries"] += 1
            if attempt["scaler_overflow_retries"] >= 32:
                raise FloatingPointError(
                    "FP16 gradients remain nonfinite after scale backoff"
                )
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        state.scaler.step(optimizer)
        state.scaler.update()
        state.scheduler.step()
        for parameter in model.parameters():
            _finite(parameter, "updated parameter")
        update += 1
        attempt["optimizer_updates"] = update
        attempt["training_losses"].append(update_loss)
        if update == 2:
            attempt["post_adam_backward_finite"] = True
    attempt["gradient_scaling"]["final_state"] = state.scaler.state_dict()
    if settings["device"] == "cuda":
        torch.cuda.synchronize()
    attempt["training_seconds"] = time.perf_counter() - train_started
    attempt["phase"] = "validation"
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    validation_started = time.perf_counter()
    # One full evaluation microbatch, or the complete smaller fixture partition.
    validation = validation[: settings["evaluation_microbatch"]]
    with torch.no_grad(), _autocast(settings):
        output = model(**_batch(validation, encoded, targets, settings["device"]))
        _finite(output.logits, "validation output")
        _finite(output.loss, "validation loss")
    attempt["validation_loss"] = float(output.loss)
    if settings["device"] == "cuda":
        torch.cuda.synchronize()
    attempt["validation_seconds"] = time.perf_counter() - validation_started
    attempt["validation_examples"] = len(validation)
    attempt["phase"] = "complete"
    return {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "problem_type": model.config.problem_type,
        "num_labels": model.config.num_labels,
        "class": type(model).__name__,
        "output": "unconstrained_raw",
        "loss": "MSE",
    }


def probe_candidates(source, environment, encoded, rows, partitions, evidence):
    settings = {
        "device": environment["device"],
        "precision": environment["precision"],
        "train_microbatch": 32,
        "gradient_accumulation": 1,
        "evaluation_microbatch": 32,
        "gradient_checkpointing": False,
        "effective_batch": EFFECTIVE_BATCH,
    }
    # Testset labels are never placed in the runtime's target mapping.
    allowed = set(partitions["train"] + partitions["validation"])
    targets = {r["source_row"]: r["target"] for r in rows if r["source_row"] in allowed}
    while True:
        gc.collect()
        if settings["device"] == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        attempt = {
            "settings": dict(settings),
            "optimizer_updates": 0,
            "post_adam_backward_finite": False,
            "phase": "initialization",
        }
        evidence["attempts"].append(attempt)
        started = time.perf_counter()
        try:
            model_metadata = _probe(
                source,
                settings,
                encoded,
                targets,
                partitions["train"],
                partitions["validation"],
                attempt,
            )
            attempt["state"] = "passed"
        except torch.OutOfMemoryError:
            attempt["state"] = "out_of_memory"
        except (Exception, KeyboardInterrupt) as error:
            attempt.update(state="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            attempt["elapsed_seconds"] = time.perf_counter() - started
            if settings["device"] == "cuda":
                attempt["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                attempt["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            else:
                attempt["peak_allocated_bytes"] = None
                attempt["peak_reserved_bytes"] = None
        if attempt["state"] == "passed":
            return dict(settings), model_metadata
        if attempt["phase"] == "validation" and settings["evaluation_microbatch"] > 1:
            settings["evaluation_microbatch"] //= 2
        elif settings["train_microbatch"] > 1:
            settings["train_microbatch"] //= 2
            settings["gradient_accumulation"] = (
                EFFECTIVE_BATCH // settings["train_microbatch"]
            )
        elif not settings["gradient_checkpointing"]:
            settings["gradient_checkpointing"] = True
        else:
            raise ResourceConstraint(
                "full NT fine-tuning cannot fit at microbatch 1 with gradient checkpointing"
            )
