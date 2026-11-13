"""Raw-sequence lookup, independent of the nuclease (#172).

The lookup knows rows by ID and sequences as text, nothing else, so SpCas9's
30bp input (#173) reuses it unchanged with its own unit and population.
"""

from __future__ import annotations

import pytest

from xai_test.nt_xai_demo import sequence_lookup as sl

UNIT = sl.SequenceUnit(length=8, name="8bp 서열")

#: Insertion order is not row order, so "first in the mapping" and "smallest
#: row ID" pick different rows for the duplicated sequence.
ROWS = {
    40: "AAAACCCC",
    7: "ACGTACGT",
    30: "AAAACCCC",
    12: "AAAACCCC",
    3: "GGGGTTTT",
}


@pytest.fixture
def index():
    return sl.SequenceIndex(ROWS, UNIT)


def test_a_unique_sequence_finds_its_row_with_no_duplicate_line(index):
    hit = index.find("ACGTACGT")
    assert hit.row == 7
    assert hit.rows == (7,)
    assert hit.duplicate_line() is None


def test_a_duplicated_sequence_shows_the_smallest_row_and_says_so(index):
    hit = index.find("AAAACCCC")
    assert hit.row == 12
    assert hit.rows == (12, 30, 40)
    assert hit.duplicate_line() == (
        "이 타깃은 3행 중복(세포주 반복): row 12, 30, 40 — 가장 작은 row 12를 보여준다"
    )


def test_the_duplicate_line_uses_the_callers_row_id_name():
    index = sl.SequenceIndex(ROWS, UNIT, id_name="source_row")
    assert (
        index.find("AAAACCCC")
        .duplicate_line()
        .startswith("이 타깃은 3행 중복(세포주 반복): source_row 12, 30, 40")
    )


def test_lower_case_and_surrounding_space_are_read_as_the_same_sequence(index):
    assert index.find("  acgtacgt ").row == 7


def test_a_miss_states_the_stored_count_and_gives_examples(index):
    with pytest.raises(sl.SequenceNotStored) as miss:
        index.find("TTTTTTTT")
    assert str(miss.value) == (
        "저장된 5행에 없음 (서로 다른 서열 3개): TTTTTTTT. "
        "예: GGGGTTTT, ACGTACGT, AAAACCCC"
    )


def test_a_miss_without_duplicates_does_not_repeat_the_count():
    index = sl.SequenceIndex({1: "AAAAAAAA", 2: "CCCCCCCC"}, UNIT)
    with pytest.raises(sl.SequenceNotStored) as miss:
        index.find("GGGGGGGG")
    assert str(miss.value) == "저장된 2행에 없음: GGGGGGGG. 예: AAAAAAAA, CCCCCCCC"


@pytest.mark.parametrize(
    ("typed", "reason"),
    [
        ("ACGTACG", "길이 7 (8이어야 한다)"),
        ("ACGTACGTA", "길이 9 (8이어야 한다)"),
        ("ACGTNCGT", "ACGT 외 문자 'N' (위치 4)"),
        ("ACGT-CGT", "ACGT 외 문자 '-' (위치 4)"),
    ],
)
def test_a_wrong_unit_is_refused_naming_the_unit(index, typed, reason):
    with pytest.raises(sl.NotASequence) as refusal:
        index.find(typed)
    assert str(refusal.value) == f"8bp 서열이 아니다: {reason}"


def test_the_length_is_checked_before_the_letters(index):
    with pytest.raises(sl.NotASequence) as refusal:
        index.find("hello")
    assert "길이 5" in str(refusal.value)


def test_a_stored_sequence_of_the_wrong_unit_is_refused_at_build():
    with pytest.raises(ValueError, match="row 2"):
        sl.SequenceIndex({1: "AAAAAAAA", 2: "AAAA"}, UNIT)


def test_an_empty_population_is_refused_at_build():
    with pytest.raises(ValueError):
        sl.SequenceIndex({}, UNIT)
