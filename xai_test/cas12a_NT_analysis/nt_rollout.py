"""Checked native attention flow, retaining CLS throughout encoder propagation."""

import importlib.metadata
import itertools
import math

import numpy as np

from cas12a.nt_data import digest

from .nt_inputs import encode_sequences, sequence_hash
from .nt_predict import ATOL, RTOL, fp32_context

PROTOCOL = {
    "schema": 1,
    "quantity": "encoder token attention flow",
    "head_reduction": "arithmetic mean",
    "residual": "0.5*A+0.5*I; row normalize",
    "product": "encoder order; R=B_layer@R; R_initial=I",
    "padding": "excluded before propagation; CLS retained",
    "sequence_block": "full[1:,1:]; no renormalization",
    "accumulation_dtype": "float64",
    "query_diagnostic": "all unordered DNA query pairs; L1 across key columns",
    "degenerate": "max pair L1 <= explicit query_l1_threshold",
    "parity_atol": ATOL,
    "parity_rtol": RTOL,
    "interpretation": "attention flow; not output attribution or selective token-pair interaction",
}


def validate_policy(policy):
    if not isinstance(policy, dict) or set(policy) != {"query_l1_threshold"}:
        raise ValueError("rollout requires an explicit query_l1_threshold policy")
    value = policy["query_l1_threshold"]
    if (
        type(value) not in (float, int)
        or not math.isfinite(value)
        or not 0 <= value <= 2
    ):
        raise ValueError("query_l1_threshold must be finite in [0,2]")
    return dict(policy)


def _query_diagnostic(matrix, threshold):
    pairs = list(itertools.combinations(range(9), 2))
    distances = np.asarray([np.abs(matrix[i] - matrix[j]).sum() for i, j in pairs])
    return {
        "query_pairs": [[i + 1, j + 1] for i, j in pairs],
        "l1_distances": distances.tolist(),
        "minimum": float(distances.min()),
        "mean": float(distances.mean()),
        "maximum": float(distances.max()),
        "near_identical_fraction": float(np.mean(distances <= threshold)),
        "status": "near_identical_queries"
        if distances.max() <= threshold
        else "distinct_queries",
    }


def derive(attention_mean, policy):
    """Reconstruct full/block matrices and query/mass diagnostics without a model."""
    validate_policy(policy)
    if (
        attention_mean.dtype != np.dtype("float64")
        or attention_mean.ndim != 3
        or attention_mean.shape[1:] != (10, 10)
        or not len(attention_mean)
        or not np.isfinite(attention_mean).all()
        or np.any(attention_mean < 0)
        or not np.allclose(attention_mean.sum(-1), 1, atol=ATOL, rtol=RTOL)
    ):
        raise ValueError("invalid valid-token attention means")
    full = np.eye(10, dtype=np.float64)
    for attention in attention_mean:
        residual = 0.5 * attention + 0.5 * np.eye(10)
        residual /= residual.sum(-1, keepdims=True)
        full = residual @ full
    sequence = full[1:, 1:].copy()
    summary = {
        "removed_cls_mass": full[1:, 0].tolist(),
        "sequence_row_mass": sequence.sum(-1).tolist(),
        "full_row_mass": full.sum(-1).tolist(),
        "sequence_renormalized": False,
        "full_query_diagnostic": _query_diagnostic(
            full[1:], policy["query_l1_threshold"]
        ),
        "sequence_query_diagnostic": _query_diagnostic(
            sequence, policy["query_l1_threshold"]
        ),
        "interpretation": PROTOCOL["interpretation"],
    }
    return {
        "attention_mean": attention_mean,
        "full": full,
        "sequence": sequence,
    }, summary


def parity_quality(outputs, saved):
    """Recompute both comparisons; absent/invalid forwards have no verdict."""
    native, enabled = outputs["native"], outputs["attention_enabled"]
    return {
        "saved_prediction_parity": None
        if native is None
        else bool(np.isclose(native, saved, atol=ATOL, rtol=RTOL)),
        "attention_parity": None
        if native is None or enabled is None
        else bool(np.isclose(enabled, native, atol=ATOL, rtol=RTOL)),
        "native_saved_absolute_difference": None
        if native is None
        else abs(native - saved),
        "attention_native_absolute_difference": None
        if native is None or enabled is None
        else abs(enabled - native),
    }


class NumericalFailure(ValueError):
    def __init__(self, stage, reason):
        super().__init__(reason)
        self.stage, self.reason = stage, reason


def _scalar(tensor, stage):
    import torch

    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.dtype != torch.float32
        or tuple(tensor.shape) != (1, 1)
        or not torch.isfinite(tensor).all()
    ):
        raise NumericalFailure(
            stage, "expected finite scalar FP32 logits with shape (1,1)"
        )
    return float(tensor.detach().cpu()[0, 0])


def _attention_means(attentions, layers, heads, guard):
    import torch

    if not isinstance(attentions, (list, tuple)) or len(attentions) != layers:
        raise NumericalFailure(
            "attention", "missing probabilities or encoder layer count mismatch"
        )
    means = []
    for index, tensor in enumerate(attentions):
        guard()
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.float32
            or tuple(tensor.shape) != (1, heads, 12, 12)
        ):
            raise NumericalFailure(
                "attention",
                f"layer {index}: expected FP32 (1,{heads},12,12) probabilities",
            )
        values = tensor.detach().cpu().numpy()
        guard()
        if (
            not np.isfinite(values).all()
            or np.any(values < 0)
            or not np.allclose(values.sum(-1), 1, atol=ATOL, rtol=RTOL)
            or not np.allclose(values[..., 10:], 0, atol=ATOL, rtol=0)
        ):
            raise NumericalFailure(
                "attention",
                f"layer {index}: nonfinite, negative, non-stochastic or padding-key probability",
            )
        means.append(values[0, :, :10, :10].astype(np.float64).mean(0))
    return np.stack(means)


def compute_rollout(row, predictor, *, policy):
    """Compute one checked source in native eval FP32, with no model updates."""
    import torch

    policy = validate_policy(policy)
    if sequence_hash(row["sequence"]) != row["sequence_sha256"]:
        raise ValueError("rollout source sequence hash mismatch")
    model = predictor.model
    if any(module.training for module in model.modules()) or any(
        parameter.dtype != torch.float32 for parameter in model.parameters()
    ):
        raise ValueError("rollout requires eval FP32 weights")
    if model.config._attn_implementation != "eager":
        raise ValueError("rollout requires native eager attention backend")
    records = encode_sequences(
        predictor.tokenizer, [row["sequence"]], predictor.embedding_rows
    )
    device = next(model.parameters()).device
    encoded = {
        key: torch.tensor([records[0][key]], dtype=torch.long, device=device)
        for key in ("input_ids", "attention_mask")
    }
    if (
        not math.isfinite(row["raw_prediction"])
        or float(np.float32(row["raw_prediction"])) != row["raw_prediction"]
    ):
        raise ValueError("saved rollout prediction must be finite FP32")
    result = {
        "schema": 1,
        "protocol": PROTOCOL,
        "policy": policy,
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
        "tokens": {
            "record": records[0],
            "valid_positions": list(range(10)),
            "dna_positions": list(range(1, 10)),
        },
        "outputs": {"native": None, "attention_enabled": None},
        "execution": {
            "backend": "eager",
            "device": str(device),
            "layers": model.config.num_hidden_layers,
            "heads": model.config.num_attention_heads,
            "layer_order": list(range(model.config.num_hidden_layers)),
            "attention_classes": [
                type(layer.attention.self).__module__
                + "."
                + type(layer.attention.self).__qualname__
                for layer in model.esm.encoder.layer
            ],
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers")
            },
            "parameter_dtype": "float32",
            "output_dtype": "float32",
            "autocast": False,
            "tf32": False,
            "eval": True,
            "gradient_enabled": False,
        },
        "accepted": False,
        "failure": None,
        "arrays": None,
        "summary": None,
    }
    try:
        predictor.budget_guard()
        with fp32_context(device):
            native = model(**encoded, output_attentions=False)
            predictor.synchronize()
            predictor.budget_guard()
            result["outputs"]["native"] = _scalar(
                getattr(native, "logits", None), "native_output"
            )
            enabled = model(**encoded, output_attentions=True)
            predictor.synchronize()
            predictor.budget_guard()
            result["outputs"]["attention_enabled"] = _scalar(
                getattr(enabled, "logits", None), "attention_output"
            )
            quality = parity_quality(result["outputs"], row["raw_prediction"])
            if (
                not quality["saved_prediction_parity"]
                or not quality["attention_parity"]
            ):
                raise NumericalFailure(
                    "parity",
                    "native/saved or attention-enabled/native FP32 parity failure",
                )
            means = _attention_means(
                getattr(enabled, "attentions", None),
                result["execution"]["layers"],
                result["execution"]["heads"],
                predictor.budget_guard,
            )
        result["arrays"], result["summary"] = derive(means, policy)
        result["accepted"] = True
    except NumericalFailure as error:
        result["failure"] = {"stage": error.stage, "reason": error.reason}
    result["quality"] = parity_quality(result["outputs"], row["raw_prediction"])
    predictor.budget_guard()
    return result
