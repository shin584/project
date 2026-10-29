"""Value checks for the canonical Cas12a feature functions.

These run without TensorFlow: `cas12a.features` is pure NumPy.
"""

import numpy as np
import pytest

from cas12a.features import (
    DINUCLEOTIDE_INDEX,
    DINUCLEOTIDE_POSITIONS,
    FEATURE_DIM,
    PAM_SLICE,
    PROTOSPACER_LENGTH,
    PROTOSPACER_SLICE,
    SEED_LENGTH,
    SEQUENCE_LENGTH,
    extract_seq_features,
    one_hot_encode,
    protospacer_of,
)

BASES_PER_POSITION = len(DINUCLEOTIDE_INDEX)
PROTOSPACER = "ACGTACGTACGTACGTACGTACG"


def sequence_with(protospacer):
    """Wrap a protospacer in the 4 bp + PAM + ... + 3 bp row layout."""
    return "ACGT" + "TTTA" + protospacer + "TTA"


SEQUENCE = sequence_with(PROTOSPACER)


def test_sequence_layout_is_self_consistent():
    assert len(SEQUENCE) == SEQUENCE_LENGTH
    # The protospacer starts where the PAM ends; a gap or overlap would mean
    # `extract_seq_features` reads the wrong 23 bp.
    assert PAM_SLICE.stop == PROTOSPACER_SLICE.start
    assert PROTOSPACER_SLICE.stop - PROTOSPACER_SLICE.start == PROTOSPACER_LENGTH


def test_protospacer_of_takes_the_declared_window():
    assert protospacer_of(SEQUENCE) == PROTOSPACER
    assert len(protospacer_of(SEQUENCE)) == PROTOSPACER_LENGTH


def test_one_hot_encode_maps_each_base_to_its_own_channel():
    assert one_hot_encode("ACGT").tolist() == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]


def test_one_hot_encode_upper_cases_input():
    assert one_hot_encode("acgt").tolist() == one_hot_encode("ACGT").tolist()


def test_one_hot_encode_zeroes_unknown_bases():
    # The real Kim 2018 file has no non-ACGT base, so this fallback is only
    # ever reached by padded or malformed sequences.
    assert one_hot_encode("AN").tolist() == [[1, 0, 0, 0], [0, 0, 0, 0]]


def test_one_hot_encode_shape_for_a_full_sequence():
    assert one_hot_encode(SEQUENCE).shape == (SEQUENCE_LENGTH, 4)


def test_dinucleotide_index_is_the_acgt_product_order():
    assert DINUCLEOTIDE_INDEX["AA"] == 0
    assert DINUCLEOTIDE_INDEX["AC"] == 1
    assert DINUCLEOTIDE_INDEX["CA"] == 4
    assert DINUCLEOTIDE_INDEX["TT"] == 15
    assert len(DINUCLEOTIDE_INDEX) == 16


def test_gc_features_count_over_protospacer_and_seed():
    gc, _ = extract_seq_features(SEQUENCE)
    # ACGT repeated: 12 of 23 protospacer bases are G or C.
    assert gc[0] == pytest.approx(12 / PROTOSPACER_LENGTH)
    # Last 8 protospacer bases "GTACGTAC": 4 of 8.
    assert gc[1] == pytest.approx(4 / SEED_LENGTH)


def test_gc_features_at_the_extremes():
    all_gc = sequence_with("G" * PROTOSPACER_LENGTH)
    gc, _ = extract_seq_features(all_gc)
    assert gc.tolist() == [1.0, 1.0]

    all_at = sequence_with("A" * PROTOSPACER_LENGTH)
    gc, _ = extract_seq_features(all_at)
    assert gc.tolist() == [0.0, 0.0]


def test_dinucleotide_vector_sets_one_bit_per_position():
    _, dinuc = extract_seq_features(SEQUENCE)
    assert dinuc.shape == (DINUCLEOTIDE_POSITIONS * BASES_PER_POSITION,)
    # One bit per overlapping pair in the protospacer.
    assert dinuc.sum() == pytest.approx(float(DINUCLEOTIDE_POSITIONS))
    for position in range(DINUCLEOTIDE_POSITIONS):
        block = dinuc[
            position * BASES_PER_POSITION : (position + 1) * BASES_PER_POSITION
        ]
        pair = PROTOSPACER[position : position + 2]
        assert block.sum() == pytest.approx(1.0)
        assert block[DINUCLEOTIDE_INDEX[pair]] == pytest.approx(1.0)


def test_dinucleotide_vector_skips_pairs_with_unknown_bases():
    unknown_at = (3, 4, 5)
    _, dinuc = extract_seq_features(sequence_with("ACGTNNACGTACGTACGTACGTA"))
    # "TN", "NN" and "NA" at those positions are not in the index.
    for position in unknown_at:
        block = dinuc[
            position * BASES_PER_POSITION : (position + 1) * BASES_PER_POSITION
        ]
        assert block.sum() == pytest.approx(0.0)
    assert dinuc.sum() == pytest.approx(float(DINUCLEOTIDE_POSITIONS - len(unknown_at)))


def test_feature_dim_matches_the_concatenated_width():
    gc, dinuc = extract_seq_features(SEQUENCE)
    assert FEATURE_DIM == gc.size + dinuc.size


def test_input_widths_are_pinned_to_the_trained_checkpoint():
    # `models/Cas12a_Only.keras` was fitted with a (34, 4) sequence input and a
    # 354-wide feature input. Changing either number invalidates it, so these
    # two literals are deliberately not derived from the constants above.
    assert (SEQUENCE_LENGTH, FEATURE_DIM) == (34, 354)


def test_features_are_float32():
    gc, dinuc = extract_seq_features(SEQUENCE)
    assert gc.dtype == np.float32
    assert dinuc.dtype == np.float32
