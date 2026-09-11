from typing import ClassVar

import numpy as np
import pytest
from ism_sweep import (
    alt_bases_for,
    compute_ism_delta,
    generate_ism_mutants,
    run_ism_sweep,
)


class FakePredictor:
    """Duck-typed stand-in for Model_A_Predictor / Model_B_Predictor.

    Score is a deterministic function of sequence content so mutants
    predictably diverge from the wild-type score, and every call is
    recorded so tests can assert exactly what was sent to `predict()`.
    """

    BASE_VALUE: ClassVar[dict[str, float]] = {"A": 0.0, "C": 1.0, "G": 2.0, "T": 3.0}

    def __init__(self, score_overrides: dict[str, float] | None = None):
        self.score_overrides = score_overrides or {}
        self.calls: list[list[str]] = []

    def predict(self, sequences: list[str]) -> np.ndarray:
        self.calls.append(list(sequences))
        return np.array([self._score(seq) for seq in sequences])

    def _score(self, sequence: str) -> float:
        if sequence in self.score_overrides:
            return self.score_overrides[sequence]
        return sum(self.BASE_VALUE[base] for base in sequence) / len(sequence)


def test_alt_bases_for_excludes_wild_type_and_is_lexicographic():
    assert alt_bases_for("A") == ["C", "G", "T"]
    assert alt_bases_for("C") == ["A", "G", "T"]
    assert alt_bases_for("G") == ["A", "C", "T"]
    assert alt_bases_for("T") == ["A", "C", "G"]


def test_alt_bases_for_rejects_non_acgt_base():
    with pytest.raises(ValueError):
        alt_bases_for("N")


def test_generate_ism_mutants_covers_every_position_and_alt_base():
    sequence = "ACGT"
    mutants = generate_ism_mutants(sequence)

    assert len(mutants) == len(sequence) * 3

    # Position-major, lexicographic-alt-base-minor ordering.
    expected = []
    for pos, wt_base in enumerate(sequence):
        for alt_base in alt_bases_for(wt_base):
            expected.append(sequence[:pos] + alt_base + sequence[pos + 1 :])
    assert mutants == expected

    for pos, mutant in enumerate(mutants):
        source_pos = pos // 3
        # Every mutant differs from the wild-type at exactly one position.
        diffs = [i for i in range(len(sequence)) if mutant[i] != sequence[i]]
        assert diffs == [source_pos]
        # The wild-type base is never reintroduced as its own "alternative".
        assert mutant[source_pos] != sequence[source_pos]


def test_compute_ism_delta_shape_matches_documented_axes():
    sequences = ["ACGT" * 9, "TGCA" * 9]  # 36bp each, like the real Testset
    predictor = FakePredictor()

    ism_delta = compute_ism_delta(sequences, predictor)

    assert ism_delta.shape == (2, 36, 3)


def test_compute_ism_delta_values_match_manual_relative_delta():
    sequences = ["ACGT"]
    predictor = FakePredictor()

    ism_delta = compute_ism_delta(sequences, predictor)

    wt_score = predictor._score("ACGT")
    mutants = generate_ism_mutants("ACGT")
    expected = np.array(
        [(predictor._score(m) - wt_score) / abs(wt_score) for m in mutants]
    ).reshape(4, 3)

    np.testing.assert_allclose(ism_delta[0], expected)


def test_compute_ism_delta_masks_near_zero_wild_type_to_nan():
    sequences = ["AAAA", "TTTT"]
    predictor = FakePredictor(score_overrides={"AAAA": 1e-4, "TTTT": 1.5e-4})

    ism_delta = compute_ism_delta(sequences, predictor)

    assert np.isnan(ism_delta[0]).all()  # |1e-4| <= 1e-4 -> masked
    assert not np.isnan(ism_delta[1]).any()  # |1.5e-4| > 1e-4 -> not masked


def test_compute_ism_delta_sends_full_mutant_sequences_to_predict():
    sequences = ["ACGT"]
    predictor = FakePredictor()

    compute_ism_delta(sequences, predictor)

    # First call is the wild-type batch, second is the full mutant sweep.
    assert predictor.calls[0] == sequences
    assert predictor.calls[1] == generate_ism_mutants("ACGT")
    # Every dispatched item is a complete candidate sequence, never a raw
    # delta or a reused wild-type value, so a real predictor recomputes its
    # own per-sequence features (e.g. Model B's physical features) fresh.
    for mutant in predictor.calls[1]:
        assert len(mutant) == len("ACGT")
        assert mutant != "ACGT"


def test_compute_ism_delta_rejects_mismatched_sequence_lengths():
    predictor = FakePredictor()
    with pytest.raises(ValueError):
        compute_ism_delta(["ACGT", "ACG"], predictor)


def test_compute_ism_delta_batches_predict_calls_to_bound_memory():
    # 3 x 36bp sequences -> 3 WT + 324 mutants.
    sequences = ["ACGT" * 9, "TGCA" * 9, "GATC" * 9]
    predictor = FakePredictor()

    compute_ism_delta(sequences, predictor, predict_batch_size=50)

    wt_call, *mutant_calls = predictor.calls
    # The 3-sequence WT batch fits in one chunk...
    assert len(wt_call) == 3
    # ...but neither predictor batches internally, so a predict() call sized
    # to the full 324-mutant sweep would risk exhausting GPU memory (this is
    # what produced the reported CUDA OOM inside Model A's LSTM forward
    # pass) - the 324 mutants must be split into bounded chunks instead.
    assert len(mutant_calls) > 1
    assert all(len(chunk) <= 50 for chunk in mutant_calls)
    assert sum(len(chunk) for chunk in mutant_calls) == 324


def test_compute_ism_delta_batch_size_does_not_change_values():
    sequences = ["ACGT" * 9]
    predictor_one_shot = FakePredictor()
    predictor_batched = FakePredictor()

    one_shot = compute_ism_delta(
        sequences, predictor_one_shot, predict_batch_size=10_000
    )
    batched = compute_ism_delta(sequences, predictor_batched, predict_batch_size=7)

    np.testing.assert_allclose(one_shot, batched)


def test_run_ism_sweep_returns_both_model_arrays():
    sequences = ["ACGT" * 9]
    predictor_a = FakePredictor()
    predictor_b = FakePredictor()

    result = run_ism_sweep(sequences, predictor_a, predictor_b)

    assert set(result.keys()) == {"ism_delta_model_a", "ism_delta_model_b"}
    assert result["ism_delta_model_a"].shape == (1, 36, 3)
    assert result["ism_delta_model_b"].shape == (1, 36, 3)
    # Each model is queried independently against its own predictor.
    assert predictor_a.calls and predictor_b.calls


def test_run_ism_sweep_forwards_predict_batch_size_to_both_models():
    sequences = ["ACGT" * 9]  # 36 positions x 3 alt bases = 108 mutants
    predictor_a = FakePredictor()
    predictor_b = FakePredictor()

    run_ism_sweep(sequences, predictor_a, predictor_b, predict_batch_size=40)

    for predictor in (predictor_a, predictor_b):
        mutant_calls = predictor.calls[1:]
        assert len(mutant_calls) > 1
        assert all(len(chunk) <= 40 for chunk in mutant_calls)
