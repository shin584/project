"""The Cas12a `report`: what the analysis asked, and what each number divides by.

The two nucleases' reports are not symmetric - SpCas9 has a replicate noise
floor and a head-order reversal, Cas12a's single output has neither - so this
report opens with the questions this analysis answered, and says which ones it
did not ask. The performance section shows all three PAM populations because
the pooled `all` Spearman carries the mean efficiency difference between the
TTTV and TTTT groups; within-group ranking is the lower pair.
"""

from __future__ import annotations

import pytest

from xai_test.nt_xai_demo import cas12a_views
from xai_test.nt_xai_demo import reader as rd
from xai_test.nt_xai_demo.test import reader_fixtures as fx


@pytest.fixture
def reader(tmp_path):
    return rd.Cas12aReader(fx.build_data_dir(tmp_path / "data"))


@pytest.fixture
def report(reader):
    return cas12a_views.report(reader)


def test_it_opens_with_the_questions_the_analysis_answered(report):
    head = report.split("[")[0]  # everything before the first section

    assert "이 분석이 답한 질문" in head
    assert "예측 성능" not in head


def test_it_says_which_questions_this_analysis_did_not_ask(report):
    assert "노이즈 바닥" in report and "순서 역전" in report
    # ... and says why, so the asymmetry does not read as a missing section.
    assert "출력이 하나" in report


def test_all_three_pam_populations_are_shown(report):
    for stratum in ("all", "TTTV", "TTTT"):
        assert stratum in report
    assert "0.7007" in report and "0.5423" in report and "0.2995" in report


def test_the_pooled_population_states_what_the_pooling_does(report):
    pooled = [line for line in report.splitlines() if "풀링" in line]

    assert pooled, report
    assert any("0.5423" in line and "0.2995" in line for line in pooled)


def test_each_pooled_figure_states_its_own_statistics_group_values(report):
    """A pooling note is about the value it sits on, never copied from another.

    The Spearman triple describes ranking; pasting it onto a rollout count or
    an error magnitude would annotate one statistic with another's numbers -
    the misread this frame exists to prevent.
    """
    spearman_note = "군별 값은 TTTV 0.5423 / TTTT 0.2995"
    for line in report.splitlines():
        if spearman_note in line:
            assert "spearman" in line, line

    rollout = [
        line
        for line in report.splitlines()
        if "층=" in line and "near_identical_query_count" in line and "풀링" in line
    ]
    assert rollout
    assert "0.5423" not in rollout[0]
    assert "1행" in rollout[0] or "2행" in rollout[0]  # the groups' own counts


def test_the_pooled_population_is_declared_pooled_in_every_section(report):
    all_lines = [line for line in report.splitlines() if "층=all" in line]

    assert all_lines
    for line in all_lines:
        assert "풀링" in line, line


def test_raw_comes_first_and_clipped_is_the_second_line(report):
    lines = [line for line in report.splitlines() if "Spearman" in line]

    assert lines
    for index in range(0, len(lines), 2):
        assert "raw" in lines[index]
        assert "clipped" in lines[index + 1]


def test_every_displayed_value_carries_its_four_fields(report):
    for line in report.splitlines():
        if line.strip().startswith("층="):
            for field in ("층=", "head=", "통계=", "분모="):
                assert field in line, line


def test_each_figure_carries_the_count_actually_divided_by(report):
    spearman = [
        line
        for line in report.splitlines()
        if "층=" in line and "spearman" in line.lower()
    ]

    # The fixture's three populations have three different counts.
    assert any("분모=3행" in line for line in spearman)
    assert any("분모=2행" in line for line in spearman)
    assert any("분모=1행" in line for line in spearman)


def test_the_rollout_degeneracy_is_shown_as_an_interpretation_limit(report):
    section = report.split("[rollout")[1]

    assert "해석 제한" in section
    assert "실패" in section  # stated as *not* a failure
    assert "저장" in section or "수치" in section


def test_the_claim_limit_is_printed_at_the_bottom(report):
    assert fx.CAS12A_LIMIT_TEXT in report
    assert report.index(fx.CAS12A_LIMIT_TEXT) > report.index("[예측 성능")


def test_it_ends_with_the_one_line_pointer_to_cross(report):
    pointers = [line for line in report.splitlines() if "`cross`" in line]

    assert len(pointers) == 1
    assert "교차 소견" in pointers[0]


def test_the_report_names_no_other_view_as_a_pointer(report):
    assert "`agreement`" not in report
    assert "`cases`" not in report


def test_nothing_in_the_report_calls_the_model_good_or_bad(report):
    for word in ("합격", "불합격", "통과 판정", "우수", "충분히"):
        assert word not in report
