"""In-silico saturation mutagenesis (ISM) sweep (Step 3, see CONTEXT.md / plan_realize.md).

Generates every position x alternative-base substitution for a set of 36bp
Testset sequences, scores wild-type and mutants through a predictor's own
`predict()`, and reduces the result to a relative delta-score sensitivity
array. Works against any predictor exposing `predict(sequences) -> array`
(duck-typed to `Model_A_Predictor` / `Model_B_Predictor`), so Model B's
per-mutant physical-feature recomputation (MFE, dG, Tm, GC) happens for free:
every mutant is dispatched as a full sequence, never a cached wild-type value
or a bare delta.
"""

from collections.abc import Sequence

import numpy as np

BASES = ("A", "C", "G", "T")

# Relative delta-Score is undefined (unbounded) as Score_WT -> 0; scores at or
# below this magnitude are masked to NaN instead of producing a spurious ratio.
RELATIVE_DELTA_NAN_THRESHOLD = 1e-4


def alt_bases_for(wt_base: str) -> list[str]:
    """The 3 non-wild-type bases, in fixed lexicographic order."""
    wt_base = wt_base.upper()
    if wt_base not in BASES:
        raise ValueError(f"Not a valid base: {wt_base!r}")
    return sorted(set(BASES) - {wt_base})


def generate_ism_mutants(sequence: str) -> list[str]:
    """All position x alternative-base substitutions for `sequence`.

    Ordered position-major (0..len-1), alternative-base lexicographic-minor,
    matching the `ism_delta` array's (position, alternative_base) axes.
    """
    sequence = sequence.upper()
    mutants = []
    for pos, wt_base in enumerate(sequence):
        for alt_base in alt_bases_for(wt_base):
            mutants.append(sequence[:pos] + alt_base + sequence[pos + 1 :])
    return mutants


def compute_ism_delta(
    sequences: Sequence[str],
    predictor,
    nan_threshold: float = RELATIVE_DELTA_NAN_THRESHOLD,
) -> np.ndarray:
    """Run the full ISM sweep for one predictor.

    Returns `ism_delta`, shape `(len(sequences), seq_len, 3)`: the relative
    score delta `(Score_mutant - Score_WT) / |Score_WT|` per position x
    lexicographic alternative base, NaN-masked wherever `|Score_WT| <=
    nan_threshold`.
    """
    sequences = list(sequences)
    seq_len = len(sequences[0])
    if any(len(seq) != seq_len for seq in sequences):
        raise ValueError("All sequences must share the same length for the ISM sweep.")

    wt_scores = np.asarray(predictor.predict(sequences), dtype=np.float64)

    mutants: list[str] = []
    for sequence in sequences:
        mutants.extend(generate_ism_mutants(sequence))

    mutant_scores = np.asarray(predictor.predict(mutants), dtype=np.float64)
    mutant_scores = mutant_scores.reshape(len(sequences), seq_len, 3)

    wt_scores_broadcast = wt_scores[:, None, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        ism_delta = (mutant_scores - wt_scores_broadcast) / np.abs(wt_scores_broadcast)

    ism_delta[np.abs(wt_scores) <= nan_threshold] = np.nan

    return ism_delta


def run_ism_sweep(
    sequences: Sequence[str], predictor_a, predictor_b
) -> dict[str, np.ndarray]:
    """Run the ISM sweep for both models over the same Testset sequences."""
    return {
        "ism_delta_model_a": compute_ism_delta(sequences, predictor_a),
        "ism_delta_model_b": compute_ism_delta(sequences, predictor_b),
    }
