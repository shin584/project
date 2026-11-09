"""Native-ID word-embedding IG for the complete raw scalar prediction function."""

import math
from contextlib import contextmanager

import numpy as np

from cas12a.nt_data import digest

from .nt_inputs import encode_sequences, sequence_hash, validate_vocabulary
from .nt_predict import ATOL, RTOL, fp32_context

PROTOCOL = {
    "schema": 1,
    "path": "replace word-embedding output; retain native input_ids and mask",
    "reference": "zero all word embeddings including CLS and padding",
    "quadrature": "midpoint",
    "initial_steps": 200,
    "refinement": "double steps; require consecutive-resolution convergence",
    "output": "native scalar raw",
    "unit": "fraction",
    "token_positions": list(range(12)),
    "dna_positions": list(range(1, 10)),
    "special_positions": [0, 10, 11],
    "completeness": "abs(sum(mass)-delta)<=atol+rtol*abs(delta)",
    "near_zero": "abs(delta)<=near_zero: absolute tolerance only; relative=null",
    "convergence": "L1(current-previous)<=atol+rtol*L1(current)",
    "accumulation_dtype": "float64",
    "endpoint_atol": ATOL,
    "endpoint_rtol": RTOL,
}


def validate_policy(policy):
    """No production quality defaults: the caller must freeze every number."""
    fields = {
        "completeness_atol",
        "completeness_rtol",
        "near_zero",
        "convergence_atol",
        "convergence_rtol",
    }
    if not isinstance(policy, dict) or set(policy) != fields | {"max_steps"}:
        raise ValueError("IG policy requires all explicit quality fields")
    for key in fields:
        value = policy[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid IG policy: {key}")
    if type(policy["max_steps"]) is not int or policy["max_steps"] not in (
        400,
        800,
        1600,
    ):
        raise ValueError("IG max_steps must be 400, 800 or 1600")
    return dict(policy)


def quality(mass, previous, outputs, policy):
    """Reconstruct raw completeness and consecutive-vector convergence model-free."""
    delta = outputs["endpoint"] - outputs["reference"]
    residual = float(mass.sum(dtype=np.float64) - delta)
    near_zero = abs(delta) <= policy["near_zero"]
    limit = policy["completeness_atol"] + (
        0 if near_zero else policy["completeness_rtol"] * abs(delta)
    )
    norm = float(np.abs(mass).sum(dtype=np.float64))
    difference = None if previous is None else float(np.abs(mass - previous).sum())
    convergence_limit = policy["convergence_atol"] + policy["convergence_rtol"] * norm
    values = [delta, residual, limit, norm, convergence_limit]
    if difference is not None:
        values.append(difference)
    relative = None if near_zero else abs(residual) / abs(delta)
    if relative is not None:
        values.append(relative)
    if not all(math.isfinite(value) for value in values):
        raise NumericalFailure("quality", "nonfinite derived quality")
    return {
        "output_difference": delta,
        "signed_residual": residual,
        "absolute_residual": abs(residual),
        "relative_residual": relative,
        "relative_undefined_reason": "near-zero output difference"
        if near_zero
        else None,
        "near_zero": near_zero,
        "completeness_limit": limit,
        "completeness_pass": abs(residual) <= limit,
        "convergence_l1": difference,
        "convergence_limit": convergence_limit,
        "convergence_pass": None
        if difference is None
        else difference <= convergence_limit,
        "convergence_undefined_reason": "no previous resolution"
        if previous is None
        else None,
        "l1_norm": norm,
    }


def summarize(masses, attempts, outputs):
    if not len(masses):
        return None
    mass = masses[-1]
    norm = float(np.abs(mass).sum())
    return {
        "steps": attempts[-1]["steps"],
        "dna_signed_mass": mass[1:10].tolist(),
        "special_signed_mass": mass[[0, 10, 11]].tolist(),
        "total_signed_mass": float(mass.sum()),
        "reconstructed_output": outputs["reference"] + float(mass.sum()),
        "quality": attempts[-1]["quality"],
        "zero_norm": norm == 0,
        "normalized_mass": (mass / norm).tolist() if norm else None,
        "normalized_undefined_reason": None if norm else "zero attribution L1 norm",
    }


class NumericalFailure(ValueError):
    """Only numerical/model-path defects become saved quality failures."""

    def __init__(self, stage, reason):
        super().__init__(reason)
        self.stage, self.reason = stage, reason


def _forward(predictor, encoded, *, replacement=None, stage):
    import torch

    predictor.budget_guard()
    calls = 0

    def replace(module, arguments, output):
        nonlocal calls
        calls += 1
        if output.shape != replacement.shape or output.dtype != replacement.dtype:
            raise NumericalFailure(
                stage, "word-embedding replacement shape/dtype mismatch"
            )
        return replacement

    hook = None
    try:
        if replacement is not None:
            hook = predictor.model.get_input_embeddings().register_forward_hook(replace)
        logits = predictor.model(**encoded).logits
        predictor.synchronize()
        predictor.budget_guard()
        if replacement is not None and calls != 1:
            raise NumericalFailure(
                stage, "word-embedding path was not used exactly once"
            )
        if logits.dtype != torch.float32 or tuple(logits.shape) != (
            len(encoded["input_ids"]),
            1,
        ):
            raise NumericalFailure(stage, "expected scalar FP32 output")
        if not torch.isfinite(logits).all():
            raise NumericalFailure(stage, "nonfinite output")
        return logits[:, 0]
    finally:
        if hook is not None:
            hook.remove()


def _integrate(predictor, encoded, embedding, steps, batch_size):
    import torch

    total = torch.zeros_like(embedding[0], dtype=torch.float64)
    batches = []
    for start in range(0, steps, batch_size):
        predictor.budget_guard()
        count = min(batch_size, steps - start)
        alpha = (
            torch.arange(
                start, start + count, device=embedding.device, dtype=torch.float32
            )
            + 0.5
        ) / steps
        interpolant = (embedding * alpha[:, None, None]).detach().requires_grad_(True)
        batch = {key: value.expand(count, -1) for key, value in encoded.items()}
        with torch.enable_grad():
            outputs = _forward(
                predictor, batch, replacement=interpolant, stage="integration"
            )
            if not outputs.requires_grad:
                raise NumericalFailure("gradient", "output disconnected from embedding")
            gradient = torch.autograd.grad(
                outputs.sum(), interpolant, allow_unused=True
            )[0]
        predictor.synchronize()
        predictor.budget_guard()
        if gradient is None:
            raise NumericalFailure("gradient", "output disconnected from embedding")
        if not torch.isfinite(gradient).all():
            raise NumericalFailure("gradient", "nonfinite gradient")
        total += gradient.detach().double().sum(0)
        batches.append(count)
    mass = (embedding[0].double() * (total / steps)).sum(-1).cpu().numpy()
    predictor.budget_guard()
    if not np.isfinite(mass).all():
        raise NumericalFailure("integration", "nonfinite attribution")
    return mass, batches


@contextmanager
def _parameter_gradients(model, disabled):
    parameters = [
        (parameter, parameter.requires_grad) for parameter in model.parameters()
    ]
    try:
        if disabled:
            for parameter, _ in parameters:
                parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in parameters:
            parameter.requires_grad_(requires_grad)


def compute_ig(
    row,
    predictor,
    *,
    policy,
    integration_batch_size=16,
    disable_parameter_gradients=False,
):
    """Compute one source, retaining quality failures without hiding execution faults."""
    import torch

    policy = validate_policy(policy)
    if type(integration_batch_size) is not int or integration_batch_size < 1:
        raise ValueError("integration batch size must be a positive integer")
    if (
        type(row["source_row"]) is not int
        or not 0 <= row["source_row"] <= np.iinfo(np.int64).max
    ):
        raise ValueError("invalid source ID")
    if sequence_hash(row["sequence"]) != row["sequence_sha256"]:
        raise ValueError("source sequence hash mismatch")
    raw = row["raw_prediction"]
    if not math.isfinite(raw) or float(np.float32(raw)) != raw:
        raise ValueError("expected saved finite FP32 raw prediction")
    predictor.budget_guard()
    model = predictor.model
    if any(module.training for module in model.modules()) or any(
        p.dtype != torch.float32 for p in model.parameters()
    ):
        raise ValueError("IG requires eval FP32 model")
    if model.config._attn_implementation != "eager":
        raise ValueError("IG requires eager attention")
    vocabulary = validate_vocabulary(predictor.tokenizer, predictor.embedding_rows)
    tokens = encode_sequences(
        predictor.tokenizer, [row["sequence"]], predictor.embedding_rows
    )[0]
    device = next(model.parameters()).device
    encoded = {
        key: torch.tensor([tokens[key]], device=device)
        for key in ("input_ids", "attention_mask")
    }
    result = {
        "schema": 1,
        "protocol": PROTOCOL,
        "policy": policy,
        "integration_batch_size": integration_batch_size,
        "row": {
            **{
                key: row[key]
                for key in (
                    "source_row",
                    "sequence",
                    "sequence_sha256",
                    "raw_prediction",
                )
            },
            "prediction_sha256": digest(row),
        },
        "tokens": tokens,
        "vocabulary": vocabulary,
        "reference": {
            "kind": PROTOCOL["reference"],
            "shape": [12, model.get_input_embeddings().weight.shape[1]],
            "dtype": "float32",
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        },
        "outputs": {"native": None, "endpoint": None, "reference": None},
        "attempts": [],
        "failure": None,
        "accepted": False,
        "execution": {
            "parameter_dtype": "float32",
            "output_dtype": "float32",
            "autocast": False,
            "tf32": False,
            "backend": "eager",
            "eval": True,
            "input_gradient_enabled": True,
            "device": str(device),
        },
    }
    masses = []
    active_steps = None
    try:
        with (
            _parameter_gradients(model, disable_parameter_gradients),
            fp32_context(device),
        ):
            embedding = model.get_input_embeddings()(encoded["input_ids"]).detach()
            if not torch.isfinite(embedding).all():
                raise NumericalFailure("embedding", "nonfinite word embedding")
            for name, replacement in (
                ("native", None),
                ("endpoint", embedding),
                ("reference", torch.zeros_like(embedding)),
            ):
                result["outputs"][name] = float(
                    _forward(predictor, encoded, replacement=replacement, stage=name)[0]
                )
            outputs = result["outputs"]
            if not np.isclose(
                outputs["native"], raw, atol=ATOL, rtol=RTOL
            ) or not np.isclose(
                outputs["endpoint"], outputs["native"], atol=ATOL, rtol=RTOL
            ):
                raise NumericalFailure(
                    "endpoint", "saved/native/embedding endpoint parity failure"
                )
            steps = 200
            while steps <= policy["max_steps"]:
                active_steps = steps
                mass, batches = _integrate(
                    predictor, encoded, embedding, steps, integration_batch_size
                )
                assessed = quality(
                    mass, masses[-1] if masses else None, outputs, policy
                )
                masses.append(mass)
                result["attempts"].append(
                    {"steps": steps, "batch_sizes": batches, "quality": assessed}
                )
                if assessed["completeness_pass"] and assessed["convergence_pass"]:
                    result["accepted"] = True
                    break
                steps *= 2
            if not result["accepted"]:
                raise NumericalFailure("quality", "retry limit exceeded")
    except NumericalFailure as error:
        result["failure"] = {
            "stage": error.stage,
            "reason": error.reason,
            "steps": active_steps,
        }
    result["arrays"] = {
        "signed_mass": np.asarray(masses, dtype=np.float64).reshape(-1, 12)
    }
    result["summary"] = summarize(
        result["arrays"]["signed_mass"], result["attempts"], result["outputs"]
    )
    predictor.synchronize()
    predictor.budget_guard()
    return result
