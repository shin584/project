import numpy as np
import pytest
import torch
from integrated_gradients import (
    DEFAULT_IG_STEPS,
    PAM_REGION_END,
    PAM_REGION_START,
    _integrated_gradients,
    integrated_gradients_model_a,
    integrated_gradients_model_b_tokens,
    l1_normalize,
    project_model_b_attributions_to_nucleotides,
    region_attribution_share,
    run_integrated_gradients,
)
from mismatch_profiling import SEED_REGION_END, SEED_REGION_START
from model_a_wrapper import Model_A_Predictor
from model_b_xai_wrapper import Model_B_XAIPredictor

NT_MODEL_DIR = "./NT_sacas9_fintuned_model"

# Same fixture sequences as test_model_b_xai_wrapper.py (sample_id 0-2).
TESTSET_SEQUENCES = [
    "CCCGGCTGACCTGCCCAAGCTGGTGGAGGGGCTGAA",
    "TTGAAGGAAGGGAATCCAGGTGTGTAAGGGTCACCT",
    "GGAGGAGGAAGAAGAACTGGAAGAGGTGGAAGACCT",
]


# ---------------------------------------------------------------------------
# _integrated_gradients: the completeness axiom (Sundararajan et al., 2017)
# holds exactly for a linear function regardless of step count, since the
# gradient is constant along the whole straight-line path.
# ---------------------------------------------------------------------------


def test_integrated_gradients_satisfies_completeness_for_a_linear_function():
    torch.manual_seed(0)
    weights = torch.randn(5)

    def linear_fn(x):
        return x @ weights

    inputs = torch.randn(4, 5)
    baseline = torch.zeros(4, 5)

    attributions = _integrated_gradients(linear_fn, inputs, baseline, steps=1)

    expected_delta = linear_fn(inputs) - linear_fn(baseline)
    torch.testing.assert_close(
        attributions.sum(dim=-1), expected_delta, atol=1e-5, rtol=1e-5
    )


def test_integrated_gradients_completeness_is_independent_of_step_count():
    torch.manual_seed(1)
    weights = torch.randn(3)

    def linear_fn(x):
        return x @ weights

    inputs = torch.randn(2, 3)
    baseline = torch.randn(2, 3)

    few_steps = _integrated_gradients(linear_fn, inputs, baseline, steps=1)
    many_steps = _integrated_gradients(linear_fn, inputs, baseline, steps=50)

    torch.testing.assert_close(few_steps, many_steps, atol=1e-5, rtol=1e-5)


def test_integrated_gradients_is_zero_when_input_equals_baseline():
    weights = torch.randn(6)

    def linear_fn(x):
        return x @ weights

    inputs = torch.randn(3, 6)

    attributions = _integrated_gradients(linear_fn, inputs, inputs.clone(), steps=10)

    torch.testing.assert_close(attributions, torch.zeros_like(attributions))


# ---------------------------------------------------------------------------
# Model A: one-hot input layer, all-zero one-hot baseline
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_a_predictor():
    # Random-initialized weights: IG's algorithmic correctness (shape,
    # completeness) doesn't depend on trained weights, and skipping the
    # weight file keeps this test module fast and independent of it.
    return Model_A_Predictor(weight_path=None)


def test_integrated_gradients_model_a_shape(model_a_predictor):
    attributions = integrated_gradients_model_a(model_a_predictor, TESTSET_SEQUENCES)

    assert attributions.shape == (len(TESTSET_SEQUENCES), 36)


def test_integrated_gradients_model_a_approximately_satisfies_completeness(
    model_a_predictor,
):
    sequences = TESTSET_SEQUENCES[:1]
    attributions = integrated_gradients_model_a(model_a_predictor, sequences, steps=200)

    inputs = model_a_predictor.encode_one_hot(sequences).to(model_a_predictor.device)
    baseline = torch.zeros_like(inputs)
    with torch.no_grad():
        expected_delta = (
            (model_a_predictor.model(inputs) - model_a_predictor.model(baseline))
            .cpu()
            .numpy()
        )

    np.testing.assert_allclose(attributions.sum(axis=-1), expected_delta, atol=1e-3)


def test_integrated_gradients_model_a_zero_sequence_yields_zero_attribution(
    model_a_predictor,
):
    # A sequence whose one-hot encoding *is* the baseline (all-zero) has no
    # (x - baseline) term anywhere, so attribution must be exactly zero -
    # this exercises the actual encoder path, unlike the pure-tensor tests
    # above which build the zero input directly.
    inputs = torch.zeros(1, 36, 4)
    baseline = torch.zeros_like(inputs)

    attributions = _integrated_gradients(
        model_a_predictor.model, inputs, baseline, steps=5
    )

    torch.testing.assert_close(attributions, torch.zeros_like(attributions))


def test_integrated_gradients_model_a_batch_size_does_not_change_values(
    model_a_predictor,
):
    one_shot = integrated_gradients_model_a(
        model_a_predictor, TESTSET_SEQUENCES, steps=5, batch_size=10_000
    )
    batched = integrated_gradients_model_a(
        model_a_predictor, TESTSET_SEQUENCES, steps=5, batch_size=1
    )

    np.testing.assert_allclose(one_shot, batched, atol=1e-6)


# ---------------------------------------------------------------------------
# Model B: (batch, 7, 1280) embedding tensor, zero-embedding baseline
# ---------------------------------------------------------------------------


class _WrongShapeXAIPredictor:
    """Stand-in exposing only `get_token_embeddings`, returning a
    deliberately mis-shaped tensor without raising - proves
    `integrated_gradients_model_b_tokens`'s own explicit shape check fires
    independently of `Model_B_XAIPredictor.get_token_embeddings`'s internal
    `assert` (which would otherwise always fire first against a real
    predictor, per `test_integrated_gradients_model_b_tokens_malformed_input_raises`
    below, and which compiles out entirely under Python's `-O` flag).
    """

    def get_token_embeddings(self, sequences: list[str]) -> torch.Tensor:
        return torch.zeros(len(sequences), 5, 1280, requires_grad=True)


@pytest.fixture(scope="module")
def xai_predictor():
    return Model_B_XAIPredictor(nt_model_dir=NT_MODEL_DIR)


def test_integrated_gradients_model_b_tokens_shape(xai_predictor):
    attributions = integrated_gradients_model_b_tokens(
        xai_predictor, TESTSET_SEQUENCES, steps=5
    )

    assert attributions.shape == (len(TESTSET_SEQUENCES), 7, 1280)


def test_integrated_gradients_model_b_tokens_malformed_input_raises(xai_predictor):
    # 12bp sequence tokenizes to 3 tokens (<cls> + two 6-mers), not 7 - see
    # test_model_b_xai_wrapper.py's identical malformed-input case. Against
    # a real predictor, `get_token_embeddings`'s own internal assert always
    # fires before our explicit check ever runs; that check is covered
    # separately below via `_WrongShapeXAIPredictor`.
    with pytest.raises(AssertionError):
        integrated_gradients_model_b_tokens(xai_predictor, ["ATCGATCGATCG"], steps=5)


def test_integrated_gradients_model_b_tokens_raises_on_mismatched_embedding_shape():
    with pytest.raises(ValueError, match="does not match expected"):
        integrated_gradients_model_b_tokens(
            _WrongShapeXAIPredictor(), TESTSET_SEQUENCES, steps=5
        )


def test_integrated_gradients_model_b_tokens_batch_size_does_not_change_values(
    xai_predictor,
):
    one_shot = integrated_gradients_model_b_tokens(
        xai_predictor, TESTSET_SEQUENCES, steps=5, batch_size=10_000
    )
    batched = integrated_gradients_model_b_tokens(
        xai_predictor, TESTSET_SEQUENCES, steps=5, batch_size=1
    )

    np.testing.assert_allclose(one_shot, batched, atol=1e-5)


def test_integrated_gradients_model_b_tokens_approximately_satisfies_completeness(
    xai_predictor,
):
    sequences = TESTSET_SEQUENCES[:1]
    attributions = integrated_gradients_model_b_tokens(
        xai_predictor, sequences, steps=100
    )

    embeddings = xai_predictor.get_token_embeddings(sequences).detach()
    baseline = torch.zeros_like(embeddings)
    with torch.no_grad():
        expected_delta = (
            (
                xai_predictor.classify_from_embeddings(embeddings)
                - xai_predictor.classify_from_embeddings(baseline)
            )
            .cpu()
            .numpy()
        )

    # Looser tolerance than Model A's completeness check: the classifier
    # head's tanh nonlinearity curves over this path (zero embedding -> a
    # real, large-magnitude NT embedding), so a finite-step Riemann sum
    # leaves a small residual even at steps=100.
    np.testing.assert_allclose(
        attributions.sum(axis=(-1, -2)), expected_delta, atol=5e-3, rtol=1e-2
    )


# ---------------------------------------------------------------------------
# Deterministic Projection Rule
# ---------------------------------------------------------------------------


def test_project_model_b_attributions_conserves_mass_per_token_block():
    rng = np.random.default_rng(0)
    token_attributions = rng.normal(size=(2, 7, 1280)).astype(np.float32)

    nucleotide_attr = project_model_b_attributions_to_nucleotides(token_attributions)

    assert nucleotide_attr.shape == (2, 36)
    per_token = token_attributions.sum(axis=-1)[:, 1:]  # drop CLS -> (2, 6)
    for token_idx in range(6):
        block = nucleotide_attr[:, token_idx * 6 : (token_idx + 1) * 6]
        np.testing.assert_allclose(
            block.sum(axis=-1), per_token[:, token_idx], atol=1e-5
        )


def test_project_model_b_attributions_divides_evenly_within_a_token_block():
    token_attributions = np.zeros((1, 7, 4), dtype=np.float32)
    token_attributions[0, 1, :] = [6.0, 0.0, 0.0, 0.0]  # first 6-mer token, sum = 6

    nucleotide_attr = project_model_b_attributions_to_nucleotides(token_attributions)

    np.testing.assert_allclose(nucleotide_attr[0, :6], np.full(6, 1.0))
    np.testing.assert_allclose(nucleotide_attr[0, 6:], np.zeros(30))


def test_project_model_b_attributions_rejects_wrong_token_count():
    token_attributions = np.zeros((1, 5, 1280), dtype=np.float32)

    with pytest.raises(ValueError):
        project_model_b_attributions_to_nucleotides(token_attributions)


# ---------------------------------------------------------------------------
# L1 normalization + region share
# ---------------------------------------------------------------------------


def test_l1_normalize_rows_sum_to_one_in_absolute_value():
    rng = np.random.default_rng(2)
    attribution = rng.normal(size=(5, 36)).astype(np.float32)

    normalized = l1_normalize(attribution)

    np.testing.assert_allclose(np.abs(normalized).sum(axis=-1), np.ones(5), atol=1e-6)


def test_l1_normalize_preserves_sign_and_relative_proportion():
    attribution = np.array([[2.0, -2.0, 4.0, 0.0]])

    normalized = l1_normalize(attribution)

    np.testing.assert_allclose(normalized, [[0.25, -0.25, 0.5, 0.0]])


def test_l1_normalize_leaves_all_zero_rows_as_zero():
    attribution = np.zeros((1, 36))

    normalized = l1_normalize(attribution)

    np.testing.assert_allclose(normalized, np.zeros((1, 36)))
    assert not np.isnan(normalized).any()


def test_region_attribution_share_sums_absolute_mass_in_range():
    normalized = np.zeros((1, 36))
    normalized[0, PAM_REGION_START:PAM_REGION_END] = [0.1, -0.1, 0.1, -0.1, 0.1, -0.1]

    pam_share = region_attribution_share(normalized, PAM_REGION_START, PAM_REGION_END)

    np.testing.assert_allclose(pam_share, [0.6])


def test_region_attribution_share_is_bounded_by_total_l1_mass():
    rng = np.random.default_rng(3)
    attribution = rng.normal(size=(4, 36)).astype(np.float32)
    normalized = l1_normalize(attribution)

    pam_share = region_attribution_share(normalized, PAM_REGION_START, PAM_REGION_END)
    seed_share = region_attribution_share(
        normalized, SEED_REGION_START, SEED_REGION_END
    )

    assert np.all(pam_share >= 0.0) and np.all(pam_share <= 1.0)
    assert np.all(seed_share >= 0.0) and np.all(seed_share <= 1.0)
    assert np.all(pam_share + seed_share <= 1.0 + 1e-6)  # regions don't overlap


# ---------------------------------------------------------------------------
# Full bundle
# ---------------------------------------------------------------------------


def test_run_integrated_gradients_returns_the_full_comparison_bundle(
    model_a_predictor, xai_predictor
):
    result = run_integrated_gradients(
        TESTSET_SEQUENCES, model_a_predictor, xai_predictor, steps=5
    )

    assert set(result.keys()) == {
        "ig_model_a",
        "ig_model_a_normalized",
        "ig_model_b_tokens",
        "ig_model_b",
        "ig_model_b_normalized",
        "pam_share_model_a",
        "pam_share_model_b",
        "seed_share_model_a",
        "seed_share_model_b",
    }

    n = len(TESTSET_SEQUENCES)
    assert result["ig_model_a"].shape == (n, 36)
    assert result["ig_model_a_normalized"].shape == (n, 36)
    assert result["ig_model_b_tokens"].shape == (n, 6)
    assert result["ig_model_b"].shape == (n, 36)
    assert result["ig_model_b_normalized"].shape == (n, 36)
    for key in (
        "pam_share_model_a",
        "pam_share_model_b",
        "seed_share_model_a",
        "seed_share_model_b",
    ):
        assert result[key].shape == (n,)
        assert np.all(result[key] >= 0.0) and np.all(result[key] <= 1.0)


def test_default_ig_steps_is_a_positive_int():
    assert isinstance(DEFAULT_IG_STEPS, int)
    assert DEFAULT_IG_STEPS > 0
