"""Mismatch tolerance profiling: single + complex scenarios (Step 3, issue #12).

Single mismatch: a single mismatch *is* a single ISM mutation, so
`compute_mismatch_single` reuses `ism_sweep.compute_ism_delta` unchanged for
`mismatch_single_delta`, and reduces it to the worst-case (minimum relative
delta) per position for `mismatch_single_min`.

Complex mismatch: a fixed set of 15 reproducible (`random_seed=42`)
multi-position (1-3bp) scenarios - 5 Seed-region, 5 Distal-region, 5
Intermittent - applied identically (same positions, same per-position
lexicographic-alternate-base selection, see `alt_bases_for`) to every Testset
sequence. Substitutions stay self-relative to each sequence's own base at
each position (the same convention `ism_sweep` uses), which guarantees every
substitution is a genuine mismatch regardless of the underlying sequence,
while keeping one scenario definition comparable ("profiled") across the
whole population. Region boundaries: Seed = positions 17-24 (the 8bp of the
protospacer immediately 5' of - i.e. PAM-proximal to - the confirmed PAM at
25-30, see PAM_REGION_START/END below), Distal =
positions 0-7 (the farthest 8bp from the PAM), Intermittent = non-consecutive
positions scattered across the full 0-35 span.

Earlier versions of this module (issue #12) fixed Seed at positions 0-7,
before the PAM's exact position was pinned down (issue #15 confirmed
25-30). That range was mislabeled "PAM-adjacent" - positions 0-7 are in fact
the farthest possible point from the PAM - and the old "Distal" range
(28-35) actually overlapped the PAM itself. Corrected here so "Seed" is
biologically PAM-proximal and "Distal" is biologically PAM-distal, matching
plan.md/plan_realize.md's stated hypothesis that cleavage sensitivity rises
closer to the PAM.

`ref_bases`/`alt_bases` in a scenario's exported metadata are illustrative:
computed against one `example_sequence` (by convention, Testset sample_id 0)
because the literal ref base at a scenario's positions differs per sequence -
the same position + alt-base-index rule is applied to every sequence's own
base, exactly as ISM does.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from ism_sweep import (
    DEFAULT_PREDICT_BATCH_SIZE,
    RELATIVE_DELTA_NAN_THRESHOLD,
    alt_bases_for,
    compute_ism_delta,
    predict_in_batches,
    relative_delta,
)

SEQ_LEN = 36
# PAM window (NNGRRN: position 30 not fixed, see CONTEXT.md), confirmed
# dataset alignment: 25-30 inclusive (25:31 as a slice).
# Kept here, beside Seed/Distal, so torch-free code (the demo CLI's cached-only
# mode) can share one set of region constants with integrated_gradients.py.
PAM_REGION_START, PAM_REGION_END = 25, 31
# PAM-proximal 8bp of the protospacer, immediately 5' of the confirmed PAM.
SEED_REGION_START, SEED_REGION_END = 17, 25  # positions 17-24
DISTAL_REGION_START, DISTAL_REGION_END = 0, 8  # positions 0-7, farthest from the PAM
SCENARIOS_PER_REGION = 5
COMPLEX_MISMATCH_SEED = 42


@dataclass(frozen=True)
class MismatchScenario:
    """One fixed complex-mismatch scenario, applied identically across the Testset."""

    mutation_id: str
    region: str
    positions: tuple[int, ...]
    alt_base_indices: tuple[int, ...]  # index into alt_bases_for(wt) per position
    description: str


def _draw_alt_indices(rng: np.random.Generator, n: int) -> tuple[int, ...]:
    return tuple(int(x) for x in rng.integers(0, 3, size=n))


def _generate_consecutive_scenarios(
    rng: np.random.Generator, region: str, region_start: int, region_end: int
) -> list[MismatchScenario]:
    scenarios = []
    region_prefix = region.lower()
    seen_positions: set[tuple[int, ...]] = set()
    for _ in range(SCENARIOS_PER_REGION):
        while True:
            length = int(rng.integers(1, 4))  # 1..3 bp, per plan.md
            max_start = region_end - length
            start = int(rng.integers(region_start, max_start + 1))
            positions = tuple(range(start, start + length))
            if positions not in seen_positions:
                seen_positions.add(positions)
                break
        alt_base_indices = _draw_alt_indices(rng, length)
        pos_label = "_".join(str(p) for p in positions)
        scenarios.append(
            MismatchScenario(
                mutation_id=f"{region_prefix}_consecutive_{length}bp_pos{pos_label}",
                region=region,
                positions=positions,
                alt_base_indices=alt_base_indices,
                description=f"{region} consecutive {length}bp mismatch at position(s) {list(positions)}",
            )
        )
    return scenarios


def _generate_intermittent_scenarios(
    rng: np.random.Generator,
) -> list[MismatchScenario]:
    scenarios = []
    seen_positions: set[tuple[int, ...]] = set()
    for _ in range(SCENARIOS_PER_REGION):
        while True:
            count = int(rng.integers(2, 4))  # 2..3 non-consecutive positions
            positions = tuple(
                sorted(int(x) for x in rng.choice(SEQ_LEN, size=count, replace=False))
            )
            is_consecutive = all(
                positions[i + 1] - positions[i] == 1 for i in range(len(positions) - 1)
            )
            if not is_consecutive and positions not in seen_positions:
                seen_positions.add(positions)
                break
        alt_base_indices = _draw_alt_indices(rng, count)
        pos_label = "_".join(str(p) for p in positions)
        scenarios.append(
            MismatchScenario(
                mutation_id=f"intermittent_nonconsecutive_{count}bp_pos{pos_label}",
                region="Intermittent",
                positions=positions,
                alt_base_indices=alt_base_indices,
                description=f"Intermittent non-consecutive {count}bp mismatch at position(s) {list(positions)}",
            )
        )
    return scenarios


def generate_mismatch_complex_scenarios(
    seed: int = COMPLEX_MISMATCH_SEED,
) -> list[MismatchScenario]:
    """The fixed 15 complex mismatch scenarios: 5 Seed + 5 Distal + 5 Intermittent.

    Fully determined by `seed` - drawn in this order (Seed, then Distal, then
    Intermittent) from one `np.random.default_rng(seed)` stream, so the same
    seed always reproduces the same 15 scenarios.
    """
    rng = np.random.default_rng(seed)
    scenarios = []
    scenarios.extend(
        _generate_consecutive_scenarios(rng, "Seed", SEED_REGION_START, SEED_REGION_END)
    )
    scenarios.extend(
        _generate_consecutive_scenarios(
            rng, "Distal", DISTAL_REGION_START, DISTAL_REGION_END
        )
    )
    scenarios.extend(_generate_intermittent_scenarios(rng))
    return scenarios


def apply_scenario(scenario: MismatchScenario, sequence: str) -> str:
    """Apply `scenario`'s positions/alt-indices to `sequence`'s own bases."""
    chars = list(sequence.upper())
    for pos, alt_idx in zip(scenario.positions, scenario.alt_base_indices):
        chars[pos] = alt_bases_for(chars[pos])[alt_idx]
    return "".join(chars)


def scenario_to_metadata(
    scenario: MismatchScenario, scenario_index: int, example_sequence: str
) -> dict:
    """Export metadata for one scenario (see module docstring on ref/alt bases)."""
    example_sequence = example_sequence.upper()
    ref_bases = [example_sequence[p] for p in scenario.positions]
    alt_bases = [
        alt_bases_for(example_sequence[p])[idx]
        for p, idx in zip(scenario.positions, scenario.alt_base_indices)
    ]
    return {
        "scenario_index": scenario_index,
        "mutation_id": scenario.mutation_id,
        "region": scenario.region,
        "positions": list(scenario.positions),
        "ref_bases": ref_bases,
        "alt_bases": alt_bases,
        "description": scenario.description,
    }


def compute_mismatch_single(
    sequences: Sequence[str],
    predictor,
    nan_threshold: float = RELATIVE_DELTA_NAN_THRESHOLD,
    predict_batch_size: int = DEFAULT_PREDICT_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Single mismatch profiling for one predictor.

    Returns `(mismatch_single_delta, mismatch_single_min)`:
    - `mismatch_single_delta`, shape `(len(sequences), seq_len, 3)`: identical
      to `ism_sweep.compute_ism_delta` (a single mismatch is a single ISM
      mutation) - all 3 substitutions preserved per position.
    - `mismatch_single_min`, shape `(len(sequences), seq_len)`: the worst-case
      (minimum) relative delta across the 3 alternatives at each position.
    """
    mismatch_single_delta = compute_ism_delta(
        sequences,
        predictor,
        nan_threshold=nan_threshold,
        predict_batch_size=predict_batch_size,
    )
    mismatch_single_min = np.min(mismatch_single_delta, axis=-1)
    return mismatch_single_delta, mismatch_single_min


def compute_mismatch_complex_delta(
    sequences: Sequence[str],
    predictor,
    scenarios: list[MismatchScenario] | None = None,
    nan_threshold: float = RELATIVE_DELTA_NAN_THRESHOLD,
    predict_batch_size: int = DEFAULT_PREDICT_BATCH_SIZE,
) -> np.ndarray:
    """Complex mismatch profiling for one predictor.

    Returns `mismatch_complex`, shape `(len(sequences), len(scenarios))`: the
    relative score delta `(Score_mutant - Score_WT) / |Score_WT|` per
    scenario, NaN-masked wherever `|Score_WT| <= nan_threshold` (same
    guardrail as the ISM sweep).
    """
    if scenarios is None:
        scenarios = generate_mismatch_complex_scenarios()

    sequences = list(sequences)
    seq_len = len(sequences[0])
    if any(len(seq) != seq_len for seq in sequences):
        raise ValueError(
            "All sequences must share the same length for mismatch profiling."
        )

    wt_scores = predict_in_batches(predictor, sequences, predict_batch_size)

    # Scenario-major, sequence-minor ordering, matching the reshape below.
    mutants: list[str] = []
    for scenario in scenarios:
        for sequence in sequences:
            mutants.append(apply_scenario(scenario, sequence))

    mutant_scores = predict_in_batches(predictor, mutants, predict_batch_size)
    mutant_scores = mutant_scores.reshape(len(scenarios), len(sequences)).T

    return relative_delta(wt_scores, mutant_scores, nan_threshold)


def run_mismatch_profiling(
    sequences: Sequence[str],
    predictor_a,
    predictor_b,
    scenarios: list[MismatchScenario] | None = None,
    predict_batch_size: int = DEFAULT_PREDICT_BATCH_SIZE,
) -> dict[str, np.ndarray]:
    """Run single + complex mismatch profiling for both models over the same Testset."""
    if scenarios is None:
        scenarios = generate_mismatch_complex_scenarios()

    result: dict[str, np.ndarray] = {}
    for model_name, predictor in (("model_a", predictor_a), ("model_b", predictor_b)):
        single_delta, single_min = compute_mismatch_single(
            sequences, predictor, predict_batch_size=predict_batch_size
        )
        result[f"mismatch_single_delta_{model_name}"] = single_delta
        result[f"mismatch_single_min_{model_name}"] = single_min
        result[f"mismatch_complex_{model_name}"] = compute_mismatch_complex_delta(
            sequences,
            predictor,
            scenarios=scenarios,
            predict_batch_size=predict_batch_size,
        )
    return result
