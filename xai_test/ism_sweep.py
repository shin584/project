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

# Neither Model_A_Predictor nor Model_B_Predictor batches internally - predict()
# runs whatever list it's given as one forward pass. An ISM sweep can hand it
# tens of thousands of mutants at once, which is large enough to exhaust GPU
# memory (observed: CUDA OOM inside Model A's LSTM forward pass on a 14.56GB
# T4 trying to allocate 20.71 GiB for one such call). Chunking dispatch here
# keeps peak memory bounded regardless of sweep size, without changing either
# predictor's own predict() contract.
DEFAULT_PREDICT_BATCH_SIZE = 256


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


def predict_in_batches(predictor, sequences: list[str], batch_size: int) -> np.ndarray:
    """Dispatch `predictor.predict()` in bounded chunks (shared with mismatch_profiling.py)."""
    if not sequences:
        return np.array([], dtype=np.float64)
    score_chunks = [
        np.asarray(predictor.predict(sequences[i : i + batch_size]), dtype=np.float64)
        for i in range(0, len(sequences), batch_size)
    ]
    return np.concatenate(score_chunks)


def relative_delta(
    wt_scores: np.ndarray, mutant_scores: np.ndarray, nan_threshold: float
) -> np.ndarray:
    """`(mutant - wt) / |wt|` per sample, NaN-masked wherever `|wt| <= nan_threshold`.

    `wt_scores` is 1D (one value per sample); `mutant_scores` broadcasts against
    it along its leading axis - shape `(n, positions, alt_bases)` for the ISM
    sweep, `(n, scenarios)` for complex mismatch profiling. Shared so both
    modules apply the exact same guardrail (see `RELATIVE_DELTA_NAN_THRESHOLD`).
    """
    broadcast_shape = (wt_scores.shape[0],) + (1,) * (mutant_scores.ndim - 1)
    wt_broadcast = wt_scores.reshape(broadcast_shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = (mutant_scores - wt_broadcast) / np.abs(wt_broadcast)
    delta[np.abs(wt_scores) <= nan_threshold] = np.nan
    return delta


def compute_ism_delta(
    sequences: Sequence[str],
    predictor,
    nan_threshold: float = RELATIVE_DELTA_NAN_THRESHOLD,
    predict_batch_size: int = DEFAULT_PREDICT_BATCH_SIZE,
) -> np.ndarray:
    """Run the full ISM sweep for one predictor.

    Returns `ism_delta`, shape `(len(sequences), seq_len, 3)`: the relative
    score delta `(Score_mutant - Score_WT) / |Score_WT|` per position x
    lexicographic alternative base, NaN-masked wherever `|Score_WT| <=
    nan_threshold`. `predict_batch_size` bounds how many sequences reach
    `predictor.predict()` per call; lower it if you still hit an
    out-of-memory error.
    """
    sequences = list(sequences)
    seq_len = len(sequences[0])
    if any(len(seq) != seq_len for seq in sequences):
        raise ValueError("All sequences must share the same length for the ISM sweep.")

    wt_scores = predict_in_batches(predictor, sequences, predict_batch_size)

    mutants: list[str] = []
    for sequence in sequences:
        mutants.extend(generate_ism_mutants(sequence))

    mutant_scores = predict_in_batches(predictor, mutants, predict_batch_size)
    mutant_scores = mutant_scores.reshape(len(sequences), seq_len, 3)

    return relative_delta(wt_scores, mutant_scores, nan_threshold)


def run_ism_sweep(
    sequences: Sequence[str],
    predictor_a,
    predictor_b,
    predict_batch_size: int = DEFAULT_PREDICT_BATCH_SIZE,
) -> dict[str, np.ndarray]:
    """Run the ISM sweep for both models over the same Testset sequences."""
    return {
        "ism_delta_model_a": compute_ism_delta(
            sequences, predictor_a, predict_batch_size=predict_batch_size
        ),
        "ism_delta_model_b": compute_ism_delta(
            sequences, predictor_b, predict_batch_size=predict_batch_size
        ),
    }
