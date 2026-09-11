from typing import ClassVar

import numpy as np
import pytest
from ism_sweep import alt_bases_for
from mismatch_profiling import (
    COMPLEX_MISMATCH_SEED,
    DISTAL_REGION_END,
    DISTAL_REGION_START,
    SEED_REGION_END,
    SEED_REGION_START,
    apply_scenario,
    compute_mismatch_complex_delta,
    compute_mismatch_single,
    generate_mismatch_complex_scenarios,
    run_mismatch_profiling,
    scenario_to_metadata,
)


class FakePredictor:
    """Same duck-typed stand-in used by test_ism_sweep.py."""

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


def _make_testset(n: int) -> list[str]:
    bases = "ACGT"
    return ["".join(bases[(i + j) % 4] for j in range(36)) for i in range(n)]


# --- generate_mismatch_complex_scenarios -----------------------------------


def test_generates_exactly_15_scenarios_5_per_region():
    scenarios = generate_mismatch_complex_scenarios()

    assert len(scenarios) == 15
    regions = [s.region for s in scenarios]
    assert regions.count("Seed") == 5
    assert regions.count("Distal") == 5
    assert regions.count("Intermittent") == 5


def test_is_reproducible_under_the_documented_seed():
    first = generate_mismatch_complex_scenarios(seed=COMPLEX_MISMATCH_SEED)
    second = generate_mismatch_complex_scenarios(seed=COMPLEX_MISMATCH_SEED)

    assert first == second


def test_different_seed_produces_different_scenarios():
    default_seed = generate_mismatch_complex_scenarios(seed=42)
    other_seed = generate_mismatch_complex_scenarios(seed=7)

    assert default_seed != other_seed


def test_seed_region_scenarios_stay_within_pam_adjacent_positions():
    scenarios = generate_mismatch_complex_scenarios()
    seed_scenarios = [s for s in scenarios if s.region == "Seed"]

    assert len(seed_scenarios) == 5
    for scenario in seed_scenarios:
        assert all(SEED_REGION_START <= p < SEED_REGION_END for p in scenario.positions)


def test_distal_region_scenarios_stay_within_the_far_end_positions():
    scenarios = generate_mismatch_complex_scenarios()
    distal_scenarios = [s for s in scenarios if s.region == "Distal"]

    assert len(distal_scenarios) == 5
    for scenario in distal_scenarios:
        assert all(
            DISTAL_REGION_START <= p < DISTAL_REGION_END for p in scenario.positions
        )


def test_intermittent_scenarios_are_never_all_consecutive():
    scenarios = generate_mismatch_complex_scenarios()
    intermittent_scenarios = [s for s in scenarios if s.region == "Intermittent"]

    assert len(intermittent_scenarios) == 5
    for scenario in intermittent_scenarios:
        positions = scenario.positions
        assert len(positions) >= 2
        assert not all(
            positions[i + 1] - positions[i] == 1 for i in range(len(positions) - 1)
        )


def test_every_scenario_has_1_to_3_unique_positions_within_the_sequence():
    scenarios = generate_mismatch_complex_scenarios()

    for scenario in scenarios:
        assert 1 <= len(scenario.positions) <= 3
        assert len(set(scenario.positions)) == len(scenario.positions)
        assert all(0 <= p < 36 for p in scenario.positions)
        assert len(scenario.alt_base_indices) == len(scenario.positions)
        assert all(0 <= idx <= 2 for idx in scenario.alt_base_indices)


def test_mutation_ids_are_unique_across_all_15_scenarios():
    scenarios = generate_mismatch_complex_scenarios()

    mutation_ids = [s.mutation_id for s in scenarios]
    assert len(set(mutation_ids)) == len(mutation_ids)


# --- apply_scenario ----------------------------------------------------------


def test_apply_scenario_always_produces_a_genuine_mismatch_at_every_position():
    scenarios = generate_mismatch_complex_scenarios()
    sequences = _make_testset(5)

    for scenario in scenarios:
        for sequence in sequences:
            mutant = apply_scenario(scenario, sequence)
            assert len(mutant) == len(sequence)
            for pos in scenario.positions:
                assert mutant[pos] != sequence[pos]
            # Every position outside the scenario is untouched.
            for pos in range(len(sequence)):
                if pos not in scenario.positions:
                    assert mutant[pos] == sequence[pos]


def test_apply_scenario_uses_the_fixed_alt_base_index_relative_to_each_sequence():
    scenario = generate_mismatch_complex_scenarios()[0]
    sequence = _make_testset(1)[0]

    mutant = apply_scenario(scenario, sequence)

    for pos, alt_idx in zip(scenario.positions, scenario.alt_base_indices):
        assert mutant[pos] == alt_bases_for(sequence[pos])[alt_idx]


# --- scenario_to_metadata -----------------------------------------------------


def test_scenario_to_metadata_carries_all_required_fields():
    scenario = generate_mismatch_complex_scenarios()[0]
    example_sequence = _make_testset(1)[0]

    metadata = scenario_to_metadata(
        scenario, scenario_index=0, example_sequence=example_sequence
    )

    assert metadata["mutation_id"] == scenario.mutation_id
    assert metadata["region"] == scenario.region
    assert metadata["positions"] == list(scenario.positions)
    assert len(metadata["ref_bases"]) == len(scenario.positions)
    assert len(metadata["alt_bases"]) == len(scenario.positions)
    for ref, alt in zip(metadata["ref_bases"], metadata["alt_bases"]):
        assert ref != alt


# --- compute_mismatch_single --------------------------------------------------


def test_compute_mismatch_single_delta_shape_matches_ism_delta():
    sequences = _make_testset(3)
    predictor = FakePredictor()

    single_delta, single_min = compute_mismatch_single(sequences, predictor)

    assert single_delta.shape == (3, 36, 3)
    assert single_min.shape == (3, 36)


def test_compute_mismatch_single_min_is_the_worst_case_across_alternatives():
    sequences = _make_testset(2)
    predictor = FakePredictor()

    single_delta, single_min = compute_mismatch_single(sequences, predictor)

    np.testing.assert_allclose(single_min, np.min(single_delta, axis=-1))


def test_compute_mismatch_single_masks_near_zero_wild_type_to_nan():
    sequences = ["AAAA" * 9, "TTTT" * 9]
    predictor = FakePredictor(
        score_overrides={sequences[0]: 1e-4, sequences[1]: 1.5e-4}
    )

    single_delta, single_min = compute_mismatch_single(sequences, predictor)

    assert np.isnan(single_delta[0]).all()
    assert np.isnan(single_min[0]).all()
    assert not np.isnan(single_delta[1]).any()
    assert not np.isnan(single_min[1]).any()


# --- compute_mismatch_complex_delta -------------------------------------------


def test_compute_mismatch_complex_delta_shape_is_samples_by_scenarios():
    sequences = _make_testset(4)
    predictor = FakePredictor()

    mismatch_complex = compute_mismatch_complex_delta(sequences, predictor)

    assert mismatch_complex.shape == (4, 15)


def test_compute_mismatch_complex_delta_values_match_manual_relative_delta():
    sequences = _make_testset(1)
    predictor = FakePredictor()
    scenarios = generate_mismatch_complex_scenarios()

    mismatch_complex = compute_mismatch_complex_delta(
        sequences, predictor, scenarios=scenarios
    )

    wt_score = predictor._score(sequences[0])
    expected = np.array(
        [
            (predictor._score(apply_scenario(s, sequences[0])) - wt_score)
            / abs(wt_score)
            for s in scenarios
        ]
    )
    np.testing.assert_allclose(mismatch_complex[0], expected)


def test_compute_mismatch_complex_delta_masks_near_zero_wild_type_to_nan():
    sequences = ["AAAA" * 9, "TTTT" * 9]
    predictor = FakePredictor(
        score_overrides={sequences[0]: 1e-4, sequences[1]: 1.5e-4}
    )

    mismatch_complex = compute_mismatch_complex_delta(sequences, predictor)

    assert np.isnan(mismatch_complex[0]).all()
    assert not np.isnan(mismatch_complex[1]).any()


def test_compute_mismatch_complex_delta_rejects_mismatched_sequence_lengths():
    predictor = FakePredictor()
    with pytest.raises(ValueError):
        compute_mismatch_complex_delta(["ACGT" * 9, "ACG"], predictor)


def test_compute_mismatch_complex_delta_batches_predict_calls_to_bound_memory():
    sequences = _make_testset(3)
    predictor = FakePredictor()

    compute_mismatch_complex_delta(sequences, predictor, predict_batch_size=10)

    wt_call, *mutant_calls = predictor.calls
    assert len(wt_call) == 3
    assert len(mutant_calls) > 1
    assert all(len(chunk) <= 10 for chunk in mutant_calls)
    assert sum(len(chunk) for chunk in mutant_calls) == 15 * 3


def test_compute_mismatch_complex_delta_batch_size_does_not_change_values():
    sequences = _make_testset(2)

    one_shot = compute_mismatch_complex_delta(
        sequences, FakePredictor(), predict_batch_size=10_000
    )
    batched = compute_mismatch_complex_delta(
        sequences, FakePredictor(), predict_batch_size=4
    )

    np.testing.assert_allclose(one_shot, batched)


# --- run_mismatch_profiling ---------------------------------------------------


def test_run_mismatch_profiling_returns_all_arrays_for_both_models():
    sequences = _make_testset(2)
    predictor_a = FakePredictor()
    predictor_b = FakePredictor()

    result = run_mismatch_profiling(sequences, predictor_a, predictor_b)

    assert set(result.keys()) == {
        "mismatch_single_delta_model_a",
        "mismatch_single_min_model_a",
        "mismatch_complex_model_a",
        "mismatch_single_delta_model_b",
        "mismatch_single_min_model_b",
        "mismatch_complex_model_b",
    }
    assert result["mismatch_single_delta_model_a"].shape == (2, 36, 3)
    assert result["mismatch_single_min_model_a"].shape == (2, 36)
    assert result["mismatch_complex_model_a"].shape == (2, 15)
    assert result["mismatch_single_delta_model_b"].shape == (2, 36, 3)
    assert result["mismatch_single_min_model_b"].shape == (2, 36)
    assert result["mismatch_complex_model_b"].shape == (2, 15)
    assert predictor_a.calls and predictor_b.calls
