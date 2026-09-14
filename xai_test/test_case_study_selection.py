import numpy as np
import pytest
from case_study_selection import (
    CONCORDANT,
    PRIMARY_DISCORDANT,
    REVERSE_PRIMARY_DISCORDANT,
    REVERSE_SECONDARY_DISCORDANT,
    SECONDARY_DISCORDANT,
    select_case_studies,
)


def test_primary_discordant_selects_top5_ranked_by_gap_descending():
    # e_a = 1..100, e_b = 100..1 (inversely related): Q75(e_a) = 75.25,
    # Q25(e_b) = 25.25 (numpy's default linear-interpolation quantile), so
    # indices 75-99 (25 candidates) satisfy the Primary condition. Only the
    # top 5 by (e_a - e_b) descending should be selected.
    n = 100
    e_a = np.arange(1, n + 1, dtype=np.float64)
    e_b = np.arange(n, 0, -1, dtype=np.float64)

    entries = select_case_studies(e_a, e_b)
    primary = [e for e in entries if e["case_type"] == PRIMARY_DISCORDANT]

    assert [e["index"] for e in primary] == [99, 98, 97, 96, 95]
    assert [e["case_id"] for e in primary] == [
        "DISCORDANT_P01",
        "DISCORDANT_P02",
        "DISCORDANT_P03",
        "DISCORDANT_P04",
        "DISCORDANT_P05",
    ]
    # Primary filled its target, so Secondary must not fire at all.
    assert not any(e["case_type"] == SECONDARY_DISCORDANT for e in entries)


def test_concordant_selects_top5_ranked_by_combined_error_ascending():
    # e_a == e_b == 1..100: Q25 = 25.25, so indices 0-24 (values 1-25)
    # satisfy Concordant. Top 5 by (e_a + e_b) ascending are the 5 smallest
    # values, i.e. indices 0-4. Primary/Secondary can never fire here since
    # e_a == e_b makes the Primary/Secondary "B much better than A" gap zero.
    n = 100
    e_a = np.arange(1, n + 1, dtype=np.float64)
    e_b = np.arange(1, n + 1, dtype=np.float64)

    entries = select_case_studies(e_a, e_b)
    concordant = [e for e in entries if e["case_type"] == CONCORDANT]

    assert [e["index"] for e in concordant] == [0, 1, 2, 3, 4]
    assert [e["case_id"] for e in concordant] == [
        "CONCORDANT_C01",
        "CONCORDANT_C02",
        "CONCORDANT_C03",
        "CONCORDANT_C04",
        "CONCORDANT_C05",
    ]
    assert not any(e["case_type"] == PRIMARY_DISCORDANT for e in entries)
    assert not any(e["case_type"] == SECONDARY_DISCORDANT for e in entries)


def test_secondary_discordant_fills_remaining_slots_when_primary_below_five():
    # Hand-built residuals (see case_study_selection.py docstring for the
    # rule): indices 0,1 are unambiguous Primary Discordant; indices 2,3,4
    # are Secondary-eligible only (large e_a, e_b small enough for a >=0.5
    # relative gap, but not small enough to clear Q25(e_b)); indices 5-9 are
    # low-e_a filler that keeps the quantiles from being distorted.
    e_a = np.array([100, 95, 90, 85, 80, 10, 9, 8, 7, 6], dtype=np.float64)
    e_b = np.array([1, 2, 40, 38, 35, 9, 8, 7, 6, 5], dtype=np.float64)

    # Sanity-check the hand-derived quantiles this scenario relies on.
    assert np.quantile(e_a, 0.75) == pytest.approx(88.75)
    assert np.quantile(e_b, 0.25) == pytest.approx(5.25)
    assert np.median(e_a) == pytest.approx(45.0)

    entries = select_case_studies(e_a, e_b)
    primary = [e for e in entries if e["case_type"] == PRIMARY_DISCORDANT]
    secondary = [e for e in entries if e["case_type"] == SECONDARY_DISCORDANT]

    assert [e["index"] for e in primary] == [0, 1]
    # Relative gaps: idx4=0.5625, idx2=0.5556, idx3=0.5529 -> descending order.
    assert [e["index"] for e in secondary] == [4, 2, 3]
    assert [e["case_id"] for e in secondary] == [
        "DISCORDANT_S01",
        "DISCORDANT_S02",
        "DISCORDANT_S03",
    ]
    # Exactly 5 Discordant-family entries total (2 Primary + 3 Secondary).
    assert len(primary) + len(secondary) == 5


def test_secondary_discordant_never_duplicates_a_primary_index():
    e_a = np.array([100, 95, 90, 85, 80, 10, 9, 8, 7, 6], dtype=np.float64)
    e_b = np.array([1, 2, 40, 38, 35, 9, 8, 7, 6, 5], dtype=np.float64)

    entries = select_case_studies(e_a, e_b)
    indices = [e["index"] for e in entries]

    assert len(indices) == len(set(indices))


def test_discordant_family_can_fall_short_of_five_when_candidates_are_scarce():
    # Same scenario as the fallback test but with only 2 Secondary
    # candidates available (idx4 dropped into the low-e_a filler group) -
    # the Discordant family should end up with 2 Primary + 2 Secondary = 4,
    # not be padded out to 5 with ineligible samples.
    e_a = np.array([100, 95, 90, 85, 10, 10, 9, 8, 7, 6], dtype=np.float64)
    e_b = np.array([1, 2, 40, 38, 9, 9, 8, 7, 6, 5], dtype=np.float64)

    entries = select_case_studies(e_a, e_b)
    discordant = [
        e
        for e in entries
        if e["case_type"] in (PRIMARY_DISCORDANT, SECONDARY_DISCORDANT)
    ]

    assert len(discordant) == 4


def test_reverse_primary_discordant_selects_top5_ranked_by_gap_descending():
    # Mirror image of the forward Primary test: e_a = 100..1, e_b = 1..100,
    # so indices 75-99 (25 candidates) satisfy the Reverse Primary condition
    # (e_b >= Q75(e_b) AND e_a <= Q25(e_a)). Only the top 5 by
    # (e_b - e_a) descending should be selected.
    n = 100
    e_a = np.arange(n, 0, -1, dtype=np.float64)
    e_b = np.arange(1, n + 1, dtype=np.float64)

    entries = select_case_studies(e_a, e_b)
    reverse_primary = [
        e for e in entries if e["case_type"] == REVERSE_PRIMARY_DISCORDANT
    ]

    assert [e["index"] for e in reverse_primary] == [99, 98, 97, 96, 95]
    assert [e["case_id"] for e in reverse_primary] == [
        "DISCORDANT_R01",
        "DISCORDANT_R02",
        "DISCORDANT_R03",
        "DISCORDANT_R04",
        "DISCORDANT_R05",
    ]
    # Reverse Primary filled its target, so Reverse Secondary must not fire.
    assert not any(e["case_type"] == REVERSE_SECONDARY_DISCORDANT for e in entries)
    # This same inversely-related shape also produces a forward Primary
    # Discordant population at the opposite end of the index range (indices
    # 0-24, vs. Reverse Primary's 75-99) - both families coexist without
    # overlapping indices.
    primary = [e for e in entries if e["case_type"] == PRIMARY_DISCORDANT]
    assert [e["index"] for e in primary] == [0, 1, 2, 3, 4]
    indices = [e["index"] for e in entries]
    assert len(indices) == len(set(indices))


def test_reverse_secondary_discordant_fills_remaining_slots_when_reverse_primary_below_five():
    # Mirror image of the forward Secondary-fallback test (e_a/e_b swapped):
    # indices 0,1 are unambiguous Reverse Primary; indices 2,3,4 are Reverse-
    # Secondary-eligible only.
    e_b = np.array([100, 95, 90, 85, 80, 10, 9, 8, 7, 6], dtype=np.float64)
    e_a = np.array([1, 2, 40, 38, 35, 9, 8, 7, 6, 5], dtype=np.float64)

    entries = select_case_studies(e_a, e_b)
    reverse_primary = [
        e for e in entries if e["case_type"] == REVERSE_PRIMARY_DISCORDANT
    ]
    reverse_secondary = [
        e for e in entries if e["case_type"] == REVERSE_SECONDARY_DISCORDANT
    ]

    assert [e["index"] for e in reverse_primary] == [0, 1]
    # Relative gaps: idx4=0.5625, idx2=0.5556, idx3=0.5529 -> descending order.
    assert [e["index"] for e in reverse_secondary] == [4, 2, 3]
    assert [e["case_id"] for e in reverse_secondary] == [
        "DISCORDANT_RS01",
        "DISCORDANT_RS02",
        "DISCORDANT_RS03",
    ]
    # Exactly 5 Reverse-Discordant-family entries total (2 Primary + 3 Secondary).
    assert len(reverse_primary) + len(reverse_secondary) == 5


def test_reverse_secondary_discordant_never_duplicates_a_reverse_primary_index():
    e_b = np.array([100, 95, 90, 85, 80, 10, 9, 8, 7, 6], dtype=np.float64)
    e_a = np.array([1, 2, 40, 38, 35, 9, 8, 7, 6, 5], dtype=np.float64)

    entries = select_case_studies(e_a, e_b)
    indices = [e["index"] for e in entries]

    assert len(indices) == len(set(indices))


def test_reverse_discordant_family_can_fall_short_of_five_when_candidates_are_scarce():
    # Mirror image of the forward fall-short test - only 2 Reverse Secondary
    # candidates available, so the Reverse-Discordant family ends up with
    # 2 Primary + 2 Secondary = 4, not padded out to 5.
    e_b = np.array([100, 95, 90, 85, 10, 10, 9, 8, 7, 6], dtype=np.float64)
    e_a = np.array([1, 2, 40, 38, 9, 9, 8, 7, 6, 5], dtype=np.float64)

    entries = select_case_studies(e_a, e_b)
    reverse_discordant = [
        e
        for e in entries
        if e["case_type"] in (REVERSE_PRIMARY_DISCORDANT, REVERSE_SECONDARY_DISCORDANT)
    ]

    assert len(reverse_discordant) == 4


def test_reverse_discordant_never_duplicates_forward_discordant_selection():
    # Degenerate case engineered so the same indices trivially qualify for
    # both the forward and reverse Primary conditions (e_a == e_b == a
    # constant collapses every quantile to that same constant, so
    # `e_a >= Q75(e_a) AND e_b <= Q25(e_b)` and its reverse mirror are both
    # satisfied everywhere) - this exercises the guard that excludes
    # already-forward-selected indices from the reverse candidate pool.
    n = 10
    e_a = np.full(n, 50.0)
    e_b = np.full(n, 50.0)

    entries = select_case_studies(e_a, e_b)
    primary = [e for e in entries if e["case_type"] == PRIMARY_DISCORDANT]
    reverse_primary = [
        e for e in entries if e["case_type"] == REVERSE_PRIMARY_DISCORDANT
    ]

    assert [e["index"] for e in primary] == [0, 1, 2, 3, 4]
    assert [e["index"] for e in reverse_primary] == [5, 6, 7, 8, 9]

    discordant_indices = [e["index"] for e in primary + reverse_primary]
    assert len(discordant_indices) == len(set(discordant_indices))


def test_total_case_study_count_is_capped_at_fifteen():
    # Abundant candidates in every bucket - selection must still cap at 15.
    n = 200
    rng = np.random.default_rng(0)
    e_a = rng.uniform(0, 1, size=n)
    e_b = rng.uniform(0, 1, size=n)

    entries = select_case_studies(e_a, e_b)

    assert len(entries) <= 15
    indices = [e["index"] for e in entries]
    assert len(indices) == len(set(indices))


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        select_case_studies(np.array([1.0, 2.0]), np.array([1.0]))


def test_rejects_non_1d_input():
    with pytest.raises(ValueError):
        select_case_studies(np.zeros((2, 2)), np.zeros((2, 2)))
