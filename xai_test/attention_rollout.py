"""Attention Rollout: Model B's internal "Information Dependency" matrix (Step 4-4, issue #16).

Recursively rolls out `Model_B_XAIPredictor`'s per-layer NT transformer
attention (see ADR-0001 / issue #10) into a single nucleotide-coordinate
dependency matrix, per plan_realize.md Step 4-4's fixed operation order:
for every layer, `0.5*A + 0.5*I` is applied first (mixing in the residual
connection's implicit identity contribution) and row-normalization is
applied second - never the other order, which produces a numerically
different (and wrong) result (see test_attention_rollout.py's synthetic
order-sensitivity check).

Per-layer attention is averaged over heads before rollout (Abnar &
Zuidema, 2020, "Quantifying Attention Flow in Transformers"). `[CLS]` is
stripped only after the full cross-layer rollout - it still participates
in the attention flow the rollout mixes through the encoder - then the
remaining 6x6 token matrix is projected to a 36x36 nucleotide matrix: each
row is repeated across the 6 nucleotides its source token spans (the same
outgoing distribution copied, not divided), while each column is repeated
and divided by 6 (spreading a token's incoming dependency evenly across
the nucleotides it spans). This extends CONTEXT.md's Deterministic
Projection Rule to a 2D matrix and, by dividing only on the column
expansion, conserves each row's total mass exactly through the projection.
Note that mass itself is generally < 1 by this point: stripping `[CLS]`
already discarded whatever share of each row's mass the token-level
rollout sent to it, so only the pre-strip token-to-token rollout is
guaranteed row-stochastic.

This is explicitly "Information Dependency" - the internal attention
connectivity NT's transformer layers construct - not a physical DNA/protein
binding claim (CONTEXT.md).
"""

import numpy as np
import torch
from integrated_gradients import CLS_TOKEN_COUNT, TOKEN_NT_SPAN
from ism_sweep import DEFAULT_PREDICT_BATCH_SIZE, dispatch_in_batches
from model_b_xai_wrapper import Model_B_XAIPredictor

INFORMATION_DEPENDENCY_NOTE = (
    "Represents internal information dependency within the NT transformer's "
    "attention structure, not a physical DNA/protein binding claim."
)


def _rollout_single_layer(head_averaged_attention: torch.Tensor) -> torch.Tensor:
    """Fixed order for one layer: `0.5*A + 0.5*I` first, row-normalize second.

    `head_averaged_attention` is `(batch, seq, seq)`. Adding the identity
    models the residual connection every transformer layer also carries
    information through unchanged; row-normalizing afterwards keeps each
    row a probability distribution over the *combined* (attention +
    residual) mass. Normalizing first, then adding the identity, would
    instead produce a numerically different matrix - see
    test_attention_rollout.py's synthetic check confirming this order
    matters.
    """
    seq_len = head_averaged_attention.shape[-1]
    identity = torch.eye(
        seq_len,
        dtype=head_averaged_attention.dtype,
        device=head_averaged_attention.device,
    )
    residual_attention = 0.5 * head_averaged_attention + 0.5 * identity
    row_sums = residual_attention.sum(dim=-1, keepdim=True)
    return residual_attention / row_sums


def compute_attention_rollout(attentions: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Combine every NT transformer layer's attention into one rollout matrix.

    `attentions` is `Model_B_XAIPredictor.get_attentions`'s output: one
    `(batch, num_heads, seq, seq)` tensor per layer, `[CLS]` still present,
    ordered from the first encoder layer to the last. Each layer is
    head-averaged then passed through `_rollout_single_layer`; layers are
    combined by left-multiplying the running rollout with each new layer's
    matrix (Abnar & Zuidema, 2020), so the returned `(batch, seq, seq)`
    matrix reflects information flow from the input tokens through to the
    final layer. Each layer's attention is checked square before it's
    consumed, so a malformed layer raises immediately instead of silently
    rolling out a wrongly-shaped matrix; this stays agnostic to the actual
    sequence length so it also works against small synthetic attention
    matrices in tests - the NT-specific `EXPECTED_SEQ_LEN` check lives in
    `attention_rollout_nucleotide_matrix`, the same split
    `integrated_gradients.py` draws between its generic `_integrated_gradients`
    core and the NT-specific `integrated_gradients_model_b_tokens` wrapper.
    """
    rollout = None
    for layer_attention in attentions:
        _batch, _num_heads, seq_len, seq_len_cols = layer_attention.shape
        assert seq_len_cols == seq_len, (
            f"Expected a square attention matrix, got {seq_len}x{seq_len_cols}"
        )

        head_averaged = layer_attention.mean(dim=1)  # (batch, seq, seq)
        layer_rollout = _rollout_single_layer(head_averaged)
        rollout = (
            layer_rollout if rollout is None else torch.bmm(layer_rollout, rollout)
        )
    return rollout


def strip_cls_token(rollout: torch.Tensor) -> torch.Tensor:
    """Drop the `[CLS]` row and column, leaving the `(batch, 6, 6)` token matrix."""
    return rollout[:, CLS_TOKEN_COUNT:, CLS_TOKEN_COUNT:]


def project_rollout_to_nucleotides(token_rollout: np.ndarray) -> np.ndarray:
    """Project a `(batch, 6, 6)` token rollout matrix to `(batch, 36, 36)` nucleotide coordinates.

    Extends CONTEXT.md's Deterministic Projection Rule to a 2D dependency
    matrix: a nucleotide row inherits its source token's full outgoing
    distribution unchanged (rows are simply repeated 6x, once per
    nucleotide the token spans), while each token's incoming dependency is
    divided evenly across the 6 nucleotide columns it spans - dividing only
    on the column expansion is what conserves each row's total mass exactly
    (whatever that mass is) through the projection. This is an explicit
    analysis assumption, not evidence the model perceives individual
    nucleotides independently.
    """
    _batch, n_rows, n_cols = token_rollout.shape
    expected_tokens = Model_B_XAIPredictor.EXPECTED_SEQ_LEN - CLS_TOKEN_COUNT
    if n_rows != expected_tokens or n_cols != expected_tokens:
        raise ValueError(
            f"Expected a ({expected_tokens}, {expected_tokens}) token rollout matrix, "
            f"got ({n_rows}, {n_cols})"
        )

    row_expanded = np.repeat(token_rollout, TOKEN_NT_SPAN, axis=1)  # (batch, 36, 6)
    return (
        np.repeat(row_expanded, TOKEN_NT_SPAN, axis=2) / TOKEN_NT_SPAN
    )  # (batch, 36, 36)


def attention_rollout_nucleotide_matrix(
    xai_predictor: Model_B_XAIPredictor,
    sequences: list[str],
    batch_size: int = DEFAULT_PREDICT_BATCH_SIZE,
) -> np.ndarray:
    """Per-sample Information Dependency matrix, shape `(batch, 36, 36)`.

    Runs Attention Rollout over every NT transformer layer, strips `[CLS]`,
    and projects the result to 36bp nucleotide coordinates. Uses
    `ism_sweep.DEFAULT_PREDICT_BATCH_SIZE` (not Integrated Gradients'
    smaller `DEFAULT_IG_BATCH_SIZE`) since this is a single forward pass
    with no backward-pass graph to retain, the same memory profile
    `predict()` itself has. `sequences` is dispatched in `batch_size`-sized
    chunks, same chunking precedent as ism_sweep.py / integrated_gradients.py.
    The first layer's attention shape is validated against
    `Model_B_XAIPredictor.EXPECTED_SEQ_LEN` before rollout runs - the same
    "check before a consumer uses it" precedent
    `integrated_gradients_model_b_tokens` follows for its own embedding
    tensor - so a malformed `get_attentions` result raises here rather than
    silently rolling out the wrong number of tokens.
    """

    def _rollout_chunk(chunk: list[str]) -> np.ndarray:
        with torch.no_grad():
            attentions = xai_predictor.get_attentions(chunk)

            first_layer_seq_len = attentions[0].shape[-1]
            if first_layer_seq_len != Model_B_XAIPredictor.EXPECTED_SEQ_LEN:
                raise ValueError(
                    f"Expected {Model_B_XAIPredictor.EXPECTED_SEQ_LEN} tokens, "
                    f"got {first_layer_seq_len}"
                )

            rollout = compute_attention_rollout(attentions)
            token_rollout = strip_cls_token(rollout).cpu().numpy()
        return project_rollout_to_nucleotides(token_rollout)

    return dispatch_in_batches(_rollout_chunk, list(sequences), batch_size)


def run_attention_rollout(
    xai_predictor: Model_B_XAIPredictor,
    sequences: list[str],
    batch_size: int = DEFAULT_PREDICT_BATCH_SIZE,
) -> dict:
    """Run Attention Rollout and package the result for a `stepN_*.py` runner.

    Mirrors `integrated_gradients.run_integrated_gradients` /
    `shap_analysis.run_shap_analysis`: the analysis module owns result
    packaging (including the "Information Dependency" labeling this issue's
    AC4 requires), so the runner script stays thin plumbing.
    """
    matrix = attention_rollout_nucleotide_matrix(
        xai_predictor, sequences, batch_size=batch_size
    )
    return {
        "attention_rollout": matrix,
        "token_matrix_dim": [Model_B_XAIPredictor.EXPECTED_SEQ_LEN - CLS_TOKEN_COUNT]
        * 2,
        "projected_nucleotide_matrix_dim": [matrix.shape[-1]] * 2,
        "note": INFORMATION_DEPENDENCY_NOTE,
    }
