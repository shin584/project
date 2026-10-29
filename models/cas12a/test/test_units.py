"""The output-unit contract fixed by #89: scores are 0-1 ratios.

Percent is a display concern. If this file is deleted the convention silently
reverts the next time someone multiplies by 100 one layer too early.
"""

import pytest

from cas12a.units import (
    SCORE_MAX,
    SCORE_MIN,
    clip_score,
    final_score,
    to_percent,
)


def test_score_bounds_are_a_ratio_not_a_percentage():
    assert (SCORE_MIN, SCORE_MAX) == (0.0, 1.0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (-0.3, 0.0),  # SaCas9's head is linear and unbounded.
        (1.42, 1.0),
    ],
)
def test_clip_score_keeps_scores_in_range(raw, expected):
    assert clip_score(raw) == pytest.approx(expected)


def test_to_percent_is_the_only_place_the_factor_100_appears():
    assert to_percent(0.85) == pytest.approx(85.0)
    assert to_percent(0.0) == pytest.approx(0.0)
    assert to_percent(1.0) == pytest.approx(100.0)


def test_to_percent_of_a_clipped_score_never_exceeds_100():
    assert to_percent(clip_score(1.9)) == pytest.approx(100.0)


def test_final_score_clips_before_applying_the_penalty():
    # Clipping after the multiplication would give 0.75 and let an unbounded
    # raw score leak through a penalty that is supposed to only ever lower it.
    assert final_score(1.5, 0.5) == pytest.approx(0.5)


def test_final_score_with_no_penalty_is_the_clipped_raw_score():
    assert final_score(0.42, 1.0) == pytest.approx(0.42)
    assert final_score(-0.1, 1.0) == pytest.approx(0.0)


def test_final_score_stays_in_range_for_any_penalty_at_most_one():
    for penalty in (0.0, 0.25, 1.0):
        for raw in (-2.0, 0.3, 7.0):
            assert SCORE_MIN <= final_score(raw, penalty) <= SCORE_MAX


def test_final_score_rejects_a_penalty_above_one():
    # A penalty > 1 is a gain, not a penalty, and would break the range.
    with pytest.raises(ValueError):
        final_score(0.5, 1.5)
