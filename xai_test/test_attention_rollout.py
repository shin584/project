import numpy as np
import pytest
import torch
from attention_rollout import (
    CLS_TOKEN_COUNT,
    attention_rollout_nucleotide_matrix,
    compute_attention_rollout,
    project_rollout_to_nucleotides,
    run_attention_rollout,
    strip_cls_token,
)
from model_b_xai_wrapper import Model_B_XAIPredictor

NT_MODEL_DIR = "./NT_sacas9_fintuned_model"

# Same fixture sequences as test_model_b_xai_wrapper.py (sample_id 0-2).
TESTSET_SEQUENCES = [
    "CCCGGCTGACCTGCCCAAGCTGGTGGAGGGGCTGAA",
    "TTGAAGGAAGGGAATCCAGGTGTGTAAGGGTCACCT",
    "GGAGGAGGAAGAAGAACTGGAAGAGGTGGAAGACCT",
]


# ---------------------------------------------------------------------------
# Fixed operation order: 0.5*A + 0.5*I first, row-normalize second.
# ---------------------------------------------------------------------------


def _wrong_order_single_layer(head_averaged_attention: torch.Tensor) -> torch.Tensor:
    """Row-normalize first, then add the identity - the order the acceptance
    criteria explicitly rules out. Used only to prove the two orderings are
    numerically distinguishable."""
    row_sums = head_averaged_attention.sum(dim=-1, keepdim=True)
    normalized = head_averaged_attention / row_sums
    seq_len = normalized.shape[-1]
    identity = torch.eye(seq_len, dtype=normalized.dtype, device=normalized.device)
    return 0.5 * normalized + 0.5 * identity


def test_operation_order_is_numerically_distinguishable_from_the_wrong_order():
    # A small synthetic, deliberately non-row-stochastic attention matrix,
    # so the two orderings' results actually differ (a row already summing
    # to 1 would make both orderings coincide, masking the bug this guards
    # against).
    synthetic_attention = torch.tensor(
        [[[0.1, 0.4, 0.1], [0.3, 0.1, 0.2], [0.5, 0.5, 0.5]]]
    )

    correct_order = compute_attention_rollout((synthetic_attention.unsqueeze(1),))
    wrong_order = _wrong_order_single_layer(synthetic_attention)

    assert not torch.allclose(correct_order, wrong_order)


def test_single_layer_rollout_rows_sum_to_one():
    rng = np.random.default_rng(0)
    synthetic_attention = torch.tensor(
        rng.uniform(size=(2, 1, 7, 7)).astype(np.float32)
    )

    rollout = compute_attention_rollout((synthetic_attention,))

    row_sums = rollout.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums))


# ---------------------------------------------------------------------------
# compute_attention_rollout: head averaging + cross-layer combination
# ---------------------------------------------------------------------------


def test_compute_attention_rollout_averages_over_heads():
    # Two heads whose average is a row-stochastic matrix -> single-layer
    # rollout should match treating that average as the only head.
    head_a = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    head_b = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    two_heads = torch.stack([head_a, head_b], dim=1)  # (1, 2, 2, 2)
    averaged = torch.stack([(head_a + head_b) / 2], dim=1)  # (1, 1, 2, 2)

    from_two_heads = compute_attention_rollout((two_heads,))
    from_averaged = compute_attention_rollout((averaged,))

    torch.testing.assert_close(from_two_heads, from_averaged)


def test_compute_attention_rollout_raises_on_non_square_attention():
    non_square_attention = torch.zeros(1, 4, 7, 5)

    with pytest.raises(AssertionError):
        compute_attention_rollout((non_square_attention,))


def test_compute_attention_rollout_combines_layers_by_matrix_multiplication():
    layer_1 = torch.tensor([[[0.6, 0.4], [0.3, 0.7]]]).unsqueeze(1)  # (1, 1, 2, 2)
    layer_2 = torch.tensor([[[0.9, 0.1], [0.2, 0.8]]]).unsqueeze(1)

    rollout = compute_attention_rollout((layer_1, layer_2))

    a_hat_1 = 0.5 * layer_1[:, 0] + 0.5 * torch.eye(2)
    a_hat_1 = a_hat_1 / a_hat_1.sum(dim=-1, keepdim=True)
    a_hat_2 = 0.5 * layer_2[:, 0] + 0.5 * torch.eye(2)
    a_hat_2 = a_hat_2 / a_hat_2.sum(dim=-1, keepdim=True)
    expected = torch.bmm(a_hat_2, a_hat_1)

    torch.testing.assert_close(rollout, expected)


# ---------------------------------------------------------------------------
# strip_cls_token
# ---------------------------------------------------------------------------


def test_strip_cls_token_drops_first_row_and_column():
    rollout = torch.arange(2 * 3 * 3, dtype=torch.float32).reshape(2, 3, 3)

    stripped = strip_cls_token(rollout)

    assert stripped.shape == (2, 3 - CLS_TOKEN_COUNT, 3 - CLS_TOKEN_COUNT)
    torch.testing.assert_close(stripped, rollout[:, 1:, 1:])


# ---------------------------------------------------------------------------
# project_rollout_to_nucleotides
# ---------------------------------------------------------------------------


def test_project_rollout_rejects_wrong_token_count():
    token_rollout = np.zeros((1, 5, 5), dtype=np.float32)

    with pytest.raises(ValueError):
        project_rollout_to_nucleotides(token_rollout)


def test_project_rollout_shape():
    token_rollout = np.zeros((3, 6, 6), dtype=np.float32)

    nucleotide_matrix = project_rollout_to_nucleotides(token_rollout)

    assert nucleotide_matrix.shape == (3, 36, 36)


def test_project_rollout_preserves_row_stochastic_property():
    rng = np.random.default_rng(1)
    token_rollout = rng.uniform(size=(2, 6, 6)).astype(np.float32)
    token_rollout /= token_rollout.sum(axis=-1, keepdims=True)

    nucleotide_matrix = project_rollout_to_nucleotides(token_rollout)

    np.testing.assert_allclose(
        nucleotide_matrix.sum(axis=-1), np.ones((2, 36)), atol=1e-6
    )


def test_project_rollout_repeats_each_token_block_across_its_six_nucleotides():
    token_rollout = np.zeros((1, 6, 6), dtype=np.float32)
    token_rollout[0, 0, 1] = 0.6  # token 0 -> token 1

    nucleotide_matrix = project_rollout_to_nucleotides(token_rollout)

    block = nucleotide_matrix[0, 0:6, 6:12]
    np.testing.assert_allclose(block, np.full((6, 6), 0.1))  # 0.6 / 6
    np.testing.assert_allclose(nucleotide_matrix[0, 0:6, 0:6], np.zeros((6, 6)))


# ---------------------------------------------------------------------------
# Full pipeline against the real NT checkpoint
# ---------------------------------------------------------------------------


class _WrongShapeXAIPredictor:
    """Stand-in exposing only `get_attentions`, returning deliberately
    mis-shaped attention (5x5, not 7x7) - proves
    `attention_rollout_nucleotide_matrix`'s own explicit shape check fires
    independently of the real predictor, mirroring
    test_integrated_gradients.py's identical pattern for its embedding
    tensor."""

    def get_attentions(self, sequences: list[str]) -> tuple[torch.Tensor, ...]:
        return (torch.zeros(len(sequences), 4, 5, 5),)


def test_attention_rollout_nucleotide_matrix_raises_on_mismatched_attention_shape():
    with pytest.raises(ValueError, match="Expected 7 tokens"):
        attention_rollout_nucleotide_matrix(_WrongShapeXAIPredictor(), ["seq"])


@pytest.fixture(scope="module")
def xai_predictor():
    return Model_B_XAIPredictor(nt_model_dir=NT_MODEL_DIR)


def test_attention_rollout_nucleotide_matrix_shape(xai_predictor):
    matrix = attention_rollout_nucleotide_matrix(xai_predictor, TESTSET_SEQUENCES)

    assert matrix.shape == (len(TESTSET_SEQUENCES), 36, 36)


def test_attention_rollout_nucleotide_matrix_row_mass_is_bounded_by_one(xai_predictor):
    # Stripping [CLS] removes whatever row-mass each token sent to it, so a
    # nucleotide row's mass is <= 1, not exactly 1 - only the pre-strip
    # token-to-token rollout is guaranteed row-stochastic (see
    # test_single_layer_rollout_rows_sum_to_one).
    matrix = attention_rollout_nucleotide_matrix(xai_predictor, TESTSET_SEQUENCES)

    row_sums = matrix.sum(axis=-1)
    assert np.all(row_sums <= 1.0 + 1e-5)
    assert np.all(row_sums > 0.0)


def test_attention_rollout_nucleotide_matrix_is_non_negative(xai_predictor):
    matrix = attention_rollout_nucleotide_matrix(xai_predictor, TESTSET_SEQUENCES)

    assert np.all(matrix >= 0.0)


def test_attention_rollout_nucleotide_matrix_batch_size_does_not_change_values(
    xai_predictor,
):
    one_shot = attention_rollout_nucleotide_matrix(
        xai_predictor, TESTSET_SEQUENCES, batch_size=10_000
    )
    batched = attention_rollout_nucleotide_matrix(
        xai_predictor, TESTSET_SEQUENCES, batch_size=1
    )

    np.testing.assert_allclose(one_shot, batched, atol=1e-5)


# ---------------------------------------------------------------------------
# run_attention_rollout: the packaged bundle a stepN_*.py runner consumes
# ---------------------------------------------------------------------------


def test_run_attention_rollout_returns_the_expected_bundle(xai_predictor):
    result = run_attention_rollout(xai_predictor, TESTSET_SEQUENCES)

    assert set(result.keys()) == {
        "attention_rollout",
        "token_matrix_dim",
        "projected_nucleotide_matrix_dim",
        "note",
    }
    assert result["attention_rollout"].shape == (len(TESTSET_SEQUENCES), 36, 36)
    assert result["token_matrix_dim"] == [6, 6]
    assert result["projected_nucleotide_matrix_dim"] == [36, 36]
    assert "not a physical DNA/protein binding claim" in result["note"]
