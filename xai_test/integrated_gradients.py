"""Integrated Gradients: Model A vs Model B sequence-level comparison (Step 2, issue #15).

Model A is attributed directly against its one-hot input layer with an
all-zero one-hot baseline. Model B is attributed against
`Model_B_XAIPredictor`'s `(batch, 7, 1280)` last-hidden-state embedding
tensor (see ADR-0001 / issue #10) with a fixed zero-embedding baseline of
the same shape, running the integration only through `nt_model.classifier`
(the ESM sequence-classification head that consumes this exact tensor - see
`EsmForSequenceClassification.forward`) rather than re-deriving gradients
through the whole encoder.

Model B's 6-mer token attributions are projected back to the 36bp sequence
via the Deterministic Projection Rule (CONTEXT.md): each token's attribution
divided evenly across the 6 nucleotides it spans. This is an explicit
analysis assumption, not a claim that the model perceives individual
nucleotides independently.

Both models' final attribution arrays are within-sequence L1-normalized
before any comparison - comparisons are relative-distribution based, never
raw magnitude - especially at the PAM (`NNGRRT`) and Seed (1-8bp) positions.
Seed boundaries (0-7) reuse mismatch_profiling.py's existing convention;
PAM boundaries (25-30bp inclusive) are this dataset's confirmed fixed
sequence-construction alignment (36bp = ... + 6bp PAM at 25-30 + 5bp 3'
flank at 31-35).
"""

import numpy as np
import torch
from ism_sweep import dispatch_in_batches
from mismatch_profiling import SEED_REGION_END, SEED_REGION_START
from model_a_wrapper import Model_A_Predictor
from model_b_xai_wrapper import Model_B_XAIPredictor

DEFAULT_IG_STEPS = 50

# Lower than ism_sweep.DEFAULT_PREDICT_BATCH_SIZE (256): unlike a single
# predict() forward pass, IG retains a full autograd graph per step across
# `steps` sequential backward passes, so it is far more memory-hungry per
# sequence - the same class of CUDA OOM ism_sweep.py's own batching guards
# against, just reached at a smaller batch size.
DEFAULT_IG_BATCH_SIZE = 32

CLS_TOKEN_COUNT = 1
TOKEN_NT_SPAN = (
    6  # 6-mer tokenizer: each non-CLS token covers 6 consecutive nucleotides
)

# PAM (NNGRRT), confirmed dataset alignment: 25-30 inclusive (25:31 as a slice).
PAM_REGION_START, PAM_REGION_END = 25, 31


def _integrated_gradients(
    forward_fn, inputs: torch.Tensor, baseline: torch.Tensor, steps: int
) -> torch.Tensor:
    """Riemann-sum Integrated Gradients (Sundararajan et al., 2017).

    `forward_fn` must map a tensor shaped like `inputs` to a `(batch,)`
    scalar-per-sample output. Summing that output before `backward()` is
    safe because every sample's output depends only on its own slice of
    `inputs` (true for both Model A's per-sequence CNN+RNN forward pass and
    Model B's per-token classifier head) - the batch-summed gradient still
    equals each sample's own gradient, so one backward pass per step covers
    the whole batch instead of one per sample.
    """
    diff = inputs - baseline
    accumulated_grad = torch.zeros_like(inputs)
    for step in range(1, steps + 1):
        alpha = step / steps
        interpolated = (baseline + alpha * diff).detach().clone().requires_grad_(True)
        output = forward_fn(interpolated)
        output.sum().backward()
        accumulated_grad += interpolated.grad
    avg_grad = accumulated_grad / steps
    return (diff * avg_grad).detach()


def integrated_gradients_model_a(
    predictor: Model_A_Predictor,
    sequences: list[str],
    steps: int = DEFAULT_IG_STEPS,
    batch_size: int = DEFAULT_IG_BATCH_SIZE,
) -> np.ndarray:
    """Per-nucleotide IG attribution for Model A, shape `(batch, 36)`.

    Runs against Model A's one-hot input layer with an all-zero one-hot
    baseline. Since the baseline is zero and a one-hot input is zero
    everywhere except the observed base's channel, only that channel ever
    carries a nonzero `(x - baseline)` term - summing over the 4-channel
    axis collapses to the attribution of the base actually present at each
    position, discarding nothing. `sequences` is dispatched in
    `batch_size`-sized chunks (see `DEFAULT_IG_BATCH_SIZE`) to bound peak
    memory the same way `ism_sweep.py` bounds `predict()` calls.
    """

    def _attribute_chunk(chunk: list[str]) -> np.ndarray:
        inputs = predictor.encode_one_hot(chunk).to(predictor.device)
        baseline = torch.zeros_like(inputs)
        attributions = _integrated_gradients(predictor.model, inputs, baseline, steps)
        return attributions.sum(dim=-1).cpu().numpy()

    return dispatch_in_batches(_attribute_chunk, list(sequences), batch_size)


def integrated_gradients_model_b_tokens(
    xai_predictor: Model_B_XAIPredictor,
    sequences: list[str],
    steps: int = DEFAULT_IG_STEPS,
    batch_size: int = DEFAULT_IG_BATCH_SIZE,
) -> np.ndarray:
    """Raw per-token IG attribution for Model B, shape `(batch, 7, 1280)`, CLS included.

    Attributes against the `(batch, 7, 1280)` last-hidden-state embedding
    tensor `Model_B_XAIPredictor` exposes, using a fixed zero-embedding
    baseline of the same shape, backpropagating only through
    `classify_from_embeddings` (the head that consumes this exact tensor -
    see `EsmForSequenceClassification.forward`) rather than replaying the
    whole encoder. The shape is validated before attribution runs; a
    mismatch raises here rather than silently attributing against a
    differently-shaped tensor. `get_token_embeddings` already asserts this
    on its own, but that assertion compiles out entirely under Python's
    `-O` flag - this check does not, so the guarantee holds independent of
    how the caller runs. `sequences` is dispatched in `batch_size`-sized
    chunks for the same reason `integrated_gradients_model_a` is.
    """

    def _attribute_chunk(chunk: list[str]) -> np.ndarray:
        embeddings = xai_predictor.get_token_embeddings(chunk).detach()

        expected_shape = (
            len(chunk),
            Model_B_XAIPredictor.EXPECTED_SEQ_LEN,
            Model_B_XAIPredictor.EXPECTED_HIDDEN_DIM,
        )
        if tuple(embeddings.shape) != expected_shape:
            raise ValueError(
                f"Model B embedding shape {tuple(embeddings.shape)} does not match "
                f"expected {expected_shape}; refusing to attribute against a "
                "differently-shaped tensor."
            )

        baseline = torch.zeros_like(embeddings)
        attributions = _integrated_gradients(
            xai_predictor.classify_from_embeddings, embeddings, baseline, steps
        )
        return attributions.cpu().numpy()

    return dispatch_in_batches(_attribute_chunk, list(sequences), batch_size)


def project_model_b_attributions_to_nucleotides(
    token_attributions: np.ndarray,
) -> np.ndarray:
    """Deterministic Projection Rule: 6-mer token attribution -> 36bp nucleotide attribution.

    `token_attributions` is `(batch, 7, 1280)` (CLS + 6 tokens, hidden
    dimension not yet collapsed). Collapses the hidden dimension by summing
    (mirroring Model A's one-hot-channel collapse), drops the CLS token,
    then divides each of the 6 remaining tokens' attribution evenly across
    the 6 nucleotides it spans - an explicit analysis assumption (see
    CONTEXT.md's Deterministic Projection Rule), not evidence the model
    perceives individual nucleotides independently. Conservation holds:
    each nucleotide block sums back to exactly its source token's
    attribution.
    """
    _batch, n_tokens, _hidden = token_attributions.shape
    if n_tokens != Model_B_XAIPredictor.EXPECTED_SEQ_LEN:
        raise ValueError(
            f"Expected {Model_B_XAIPredictor.EXPECTED_SEQ_LEN} tokens, got {n_tokens}"
        )

    per_token = token_attributions.sum(axis=-1)  # (batch, 7)
    per_kmer_token = per_token[:, CLS_TOKEN_COUNT:]  # drop CLS -> (batch, 6)

    return (
        np.repeat(per_kmer_token, TOKEN_NT_SPAN, axis=1) / TOKEN_NT_SPAN
    )  # (batch, 36)


def l1_normalize(attribution: np.ndarray) -> np.ndarray:
    """Within-sequence L1 normalization: each row's absolute values sum to 1.

    Model A and Model B's attribution magnitudes are on unrelated scales -
    comparisons are only meaningful as relative distributions (per
    plan_realize.md Step 2), never raw magnitude. All-zero rows stay
    all-zero (dividing by zero L1 mass would otherwise produce NaN) since
    there is no distribution to normalize.
    """
    l1_mass = np.abs(attribution).sum(axis=-1, keepdims=True)
    safe_mass = np.where(l1_mass == 0, 1.0, l1_mass)
    return attribution / safe_mass


def region_attribution_share(
    normalized_attribution: np.ndarray, start: int, end: int
) -> np.ndarray:
    """Fraction of a sequence's (already L1-normalized) attribution mass in `[start, end)`."""
    return np.abs(normalized_attribution[:, start:end]).sum(axis=-1)


def run_integrated_gradients(
    sequences: list[str],
    predictor_a: Model_A_Predictor,
    xai_predictor_b: Model_B_XAIPredictor,
    steps: int = DEFAULT_IG_STEPS,
    batch_size: int = DEFAULT_IG_BATCH_SIZE,
) -> dict:
    """Run IG for both models and the PAM/Seed relative-distribution comparison.

    Returns raw and L1-normalized per-nucleotide attribution arrays for each
    model, Model B's pre-projection per-token attribution (CLS dropped), and
    each model's PAM (25-30bp) / Seed (0-7bp) attribution-mass share - the
    relative-distribution comparison issue #15's AC4 calls for.
    """
    ig_a = integrated_gradients_model_a(
        predictor_a, sequences, steps=steps, batch_size=batch_size
    )
    ig_a_norm = l1_normalize(ig_a)

    ig_b_tokens = integrated_gradients_model_b_tokens(
        xai_predictor_b, sequences, steps=steps, batch_size=batch_size
    )
    ig_b = project_model_b_attributions_to_nucleotides(ig_b_tokens)
    ig_b_norm = l1_normalize(ig_b)

    return {
        "ig_model_a": ig_a,
        "ig_model_a_normalized": ig_a_norm,
        "ig_model_b_tokens": ig_b_tokens[:, CLS_TOKEN_COUNT:, :].sum(axis=-1),
        "ig_model_b": ig_b,
        "ig_model_b_normalized": ig_b_norm,
        "pam_share_model_a": region_attribution_share(
            ig_a_norm, PAM_REGION_START, PAM_REGION_END
        ),
        "pam_share_model_b": region_attribution_share(
            ig_b_norm, PAM_REGION_START, PAM_REGION_END
        ),
        "seed_share_model_a": region_attribution_share(
            ig_a_norm, SEED_REGION_START, SEED_REGION_END
        ),
        "seed_share_model_b": region_attribution_share(
            ig_b_norm, SEED_REGION_START, SEED_REGION_END
        ),
    }
