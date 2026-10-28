"""Loader contract for the Kim 2018 Cas12a table.

The real dataset is clean (34 bp everywhere, no non-ACGT base, two missing
targets), so every defensive branch below is driven by
`fixtures/synthetic_cas12a.csv` instead. See `cas12a/docs/cas12a-model-review.md` §1.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from cas12a.data import read_cas12a_table
from cas12a.features import (
    FEATURE_DIM,
    PAM_SLICE,
    SEQUENCE_LENGTH,
    extract_seq_features,
)

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_cas12a.csv"
REPO_ROOT = Path(__file__).resolve().parents[2]

# Row positions in the fixture, which doubles as the list of branches covered.
NORMAL_ROWS = list(range(10))
TARGET_MISSING = 10
TARGET_UNPARSEABLE = 11
TARGET_NEGATIVE = 12
TARGET_ABOVE_100 = 13
SEQUENCE_TOO_LONG = 14
SEQUENCE_TOO_SHORT = 15
SEQUENCE_UNKNOWN_BASE = 16
SEQUENCE_LOWERCASE = 17
SEQUENCE_WHITESPACE = 18
SEQUENCE_MISSING = 19

KEPT_ROWS = [
    *NORMAL_ROWS,
    TARGET_NEGATIVE,
    TARGET_ABOVE_100,
    SEQUENCE_TOO_LONG,
    SEQUENCE_TOO_SHORT,
    SEQUENCE_UNKNOWN_BASE,
    SEQUENCE_LOWERCASE,
    SEQUENCE_WHITESPACE,
]


@pytest.fixture(scope="module")
def table():
    return read_cas12a_table(FIXTURE)


def test_loading_the_table_does_not_pull_in_a_deep_learning_framework():
    """`cas12a/test/` has to pass without TensorFlow installed (#91).

    Asserting on *this* process's `sys.modules` would pass here and fail in a
    full-suite run, where an `xai_test` module has already imported TensorFlow
    and torch for its own reasons. A clean interpreter is the only honest way
    to ask what `cas12a` itself pulls in.
    """
    probe = textwrap.dedent(f"""
        import sys

        from cas12a.data import read_cas12a_table

        read_cas12a_table(r"{FIXTURE}")
        print([m for m in ("tensorflow", "torch") if m in sys.modules])
    """)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )
    assert result.stdout.strip() == "[]"


def test_rows_without_a_usable_sequence_or_target_are_dropped(table):
    assert list(table.index) == KEPT_ROWS
    for dropped in (TARGET_MISSING, TARGET_UNPARSEABLE, SEQUENCE_MISSING):
        assert dropped not in table.index


def test_every_sequence_is_exactly_34_bp(table):
    assert set(table["sequence"].str.len()) == {SEQUENCE_LENGTH}


def test_pam_sits_at_positions_4_to_7(table):
    # The column header declares "4 bp + PAM + 23 bp protospacer + 3 bp", and
    # the Kim 2018 library is TTTN - so TTT is fixed and position 7 varies.
    pams = table["sequence"].str.slice(PAM_SLICE.start, PAM_SLICE.stop)
    assert set(pams.str.slice(0, 3)) == {"TTT"}
    assert set(pams.str.len()) == {4}


def test_targets_are_ratios_in_the_unit_interval(table):
    assert table["target"].between(0.0, 1.0).all()


def test_percent_targets_are_divided_by_100(table):
    assert table.loc[0, "target"] == pytest.approx(0.734)


def test_negative_targets_are_clipped_to_zero(table):
    # Background subtraction pushes 5.7% of the real rows below zero.
    assert table.loc[TARGET_NEGATIVE, "target"] == pytest.approx(0.0)


def test_targets_above_100_percent_are_clipped_to_one(table):
    assert table.loc[TARGET_ABOVE_100, "target"] == pytest.approx(1.0)


def test_over_long_sequences_are_truncated_to_the_first_34_bp(table):
    sequence = table.loc[SEQUENCE_TOO_LONG, "sequence"]
    assert sequence == "ATCGTTTCAGGAGGAGGAGGAGGAGGAGGAGTCC"
    assert len(sequence) == SEQUENCE_LENGTH


def test_short_sequences_are_padded_with_n(table):
    sequence = table.loc[SEQUENCE_TOO_SHORT, "sequence"]
    assert sequence.endswith("NNNN")
    assert len(sequence) == SEQUENCE_LENGTH


def test_sequences_are_upper_cased_and_stripped(table):
    assert table.loc[SEQUENCE_LOWERCASE, "sequence"].isupper()
    assert (
        table.loc[SEQUENCE_WHITESPACE, "sequence"]
        == table.loc[SEQUENCE_WHITESPACE, "sequence"].strip()
    )


class TestArrays:
    """`load_cas12a_dataset` is the array view the notebook trains on."""

    @pytest.fixture(scope="class")
    def arrays(self):
        from cas12a.data import load_cas12a_dataset

        return load_cas12a_dataset(FIXTURE)

    def test_shapes_follow_the_row_count(self, arrays, table):
        x_seq, x_feat, y = arrays
        rows = len(table)
        assert x_seq.shape == (rows, SEQUENCE_LENGTH, 4)
        assert x_feat.shape == (rows, FEATURE_DIM)
        assert y.shape == (rows,)

    def test_arrays_are_float32(self, arrays):
        for array in arrays:
            assert array.dtype == np.float32

    def test_targets_match_the_table(self, arrays, table):
        _, _, y = arrays
        assert y.tolist() == pytest.approx(table["target"].tolist())

    def test_feature_rows_are_the_gc_and_dinucleotide_features_concatenated(
        self, arrays, table
    ):
        _, x_feat, _ = arrays
        for position, sequence in enumerate(table["sequence"]):
            gc, dinuc = extract_seq_features(sequence)
            assert x_feat[position].tolist() == pytest.approx(
                np.concatenate([gc, dinuc]).tolist()
            )

    def test_padded_positions_one_hot_to_all_zeros(self, arrays, table):
        x_seq, _, _ = arrays
        position = list(table.index).index(SEQUENCE_TOO_SHORT)
        assert x_seq[position][-4:].sum() == pytest.approx(0.0)

    def test_every_known_base_sets_exactly_one_channel(self, arrays, table):
        x_seq, _, _ = arrays
        position = list(table.index).index(0)
        assert x_seq[position].sum(axis=1).tolist() == [1.0] * SEQUENCE_LENGTH
