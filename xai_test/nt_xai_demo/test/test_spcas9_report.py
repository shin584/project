"""The SpCas9 `report`: three analysed heads, and the noise floor kept in its context.

9-Hydra v3 stores nine heads, but XAI was run on three (`SpCas9`,
`SpCas9-NG`, `evoCas9`); the other six must not appear anywhere. The report
opens with the questions the analysis answered, because its sections differ
from the Cas12a report's.

The replicate noise floor is a spread of *measured efficiency* (fraction
scale), so it belongs only where the head-ordering reversal reads a measured
label gap. It has no unit in common with the IG-ISM agreement, a
dimensionless rank correlation, and that agreement carries no pass/fail
verdict: it is shown as measured, beside the artifact's `mismatch_note`.
"""

from __future__ import annotations

import json

import pytest

from xai_test.nt_xai_demo import reader as rd
from xai_test.nt_xai_demo import spcas9_views
from xai_test.nt_xai_demo.test import reader_fixtures as fx

UNANALYSED_HEADS = tuple(h for h in fx.ISM_HEADS if h not in fx.ANALYSED_HEADS)


@pytest.fixture
def reader(tmp_path):
    return rd.SpCas9Reader(fx.build_data_dir(tmp_path / "data"))


@pytest.fixture
def report(reader):
    return spcas9_views.report(reader)


def _section(report: str, title: str) -> str:
    """The text of one `[title ...]` section, up to the next section or the footer."""
    start = report.index(f"[{title}")
    rest = report[start + 1 :]
    ends = [i for i in (rest.find("\n["), rest.find("\n-- ")) if i >= 0]
    return report[start : start + 1 + min(ends)]


def test_it_opens_with_the_questions_the_analysis_answered(report):
    head = report.split("[")[0]

    assert "이 분석이 답한 질문" in head
    assert "순서 역전" in head and "노이즈 바닥" in head
    assert "IG" in head and "ISM" in head


def test_only_the_three_analysed_heads_appear(report):
    for head in fx.ANALYSED_HEADS:
        assert head in report
    for head in UNANALYSED_HEADS:
        assert head not in report, head


def test_the_primary_head_is_named(report):
    assert "primary" in report
    assert "SpCas9" in report


def test_the_noise_floor_is_shown_per_head(report):
    for head, median in fx.NOISE_FLOOR_MEDIAN.items():
        lines = [
            line
            for line in report.splitlines()
            if "층=" in line and f"head={head} " in line and "노이즈 바닥" in line
        ]
        assert lines, head
    text = report
    for median in fx.NOISE_FLOOR_MEDIAN.values():
        assert f"{median:.4g}" in text


def test_the_pooled_noise_floor_is_never_shown(report):
    assert f"{fx.NOISE_FLOOR_POOLED_MEDIAN:.4g}" not in report
    assert "7.06" not in report  # head_floor_ratio only exists to motivate by_head


def test_the_noise_floor_appears_only_inside_the_head_order_reversal(report):
    reversal = _section(report, "head 순서 역전")
    outside = report.replace(reversal, "")
    head = report.split("[")[0]
    body_outside = outside.replace(head, "")

    assert "노이즈 바닥" in reversal
    assert "노이즈 바닥" not in body_outside
    for median in fx.NOISE_FLOOR_MEDIAN.values():
        assert f"{median:.4g}" not in outside


def test_the_noise_floor_is_stated_as_measured_efficiency_spread(report):
    reversal = _section(report, "head 순서 역전")

    assert "fraction" in reversal
    assert "%p" in reversal
    assert "실측" in reversal


def test_the_noise_floor_is_not_tied_to_the_ig_ism_agreement(report):
    reversal = _section(report, "head 순서 역전")

    # Stated, so nobody reads 0.70 against 0.0637.
    assert "IG-ISM" in reversal
    assert "연결할 수 없다" in reversal or "비교할 수 없다" in reversal


def test_the_reversal_section_says_why_three_heads_are_kept(report):
    reversal = _section(report, "head 순서 역전")

    assert "3개 head를 유지" in reversal


def test_the_reversal_is_shown_per_stratum_never_pooled(report):
    reversal = _section(report, "head 순서 역전")

    for stratum, (rows, targets, reversals, clearing) in fx.REVERSAL_STRATA.items():
        assert f"층={stratum}" in reversal
        assert f"분모={rows:,}행·{targets:,}타깃" in reversal
    total = sum(shape[0] for shape in fx.REVERSAL_STRATA.values())
    assert f"{total:,}" not in report
    assert "풀링:" not in report


def test_the_rows_clearing_the_floor_divide_by_the_reversals(report):
    reversal = _section(report, "head 순서 역전")
    _, _, reversals, _ = fx.REVERSAL_STRATA["generalisation"]
    lines = [
        line
        for line in reversal.splitlines()
        if "층=generalisation" in line and "clearing_floor" in line
    ]

    assert lines
    assert f"분모={reversals:,}행" in lines[0]


def test_the_pam_library_stratum_repeats_its_reversals_per_target(report):
    """Its rows are PAM variants of 5 targets, so the pool can hide one target."""
    reversal = _section(report, "head 순서 역전")

    for target in fx.REVERSAL_PER_TARGET:
        lines = [
            line
            for line in reversal.splitlines()
            if target["protospacer"] in line and "역전 행" in line
        ]
        assert lines, target["protospacer"]
    # The stratum's own qualifier_note is quoted, not restated in the demo's words.
    assert "원문(artifact 기록)" in reversal
    assert "n is 5 targets" in reversal
    # The per-target figures divide by that target's own rows, not the pool's.
    assert "분모=268행·1타깃" in reversal


def test_the_distribution_against_the_floor_is_shown_not_only_the_count(report):
    """The artifact's own headline: the median inverted pair sits under its floor."""
    reversal = _section(report, "head 순서 역전")
    lines = [line for line in reversal.splitlines() if "gap_over_floor" in line]

    assert lines
    for line in lines:
        assert "분모=" in line and "쌍" in line
    assert "0.42" in reversal and "0.15" in reversal


def test_the_tie_counts_are_reconciled_rather_than_left_implicit(report):
    """`tie_policy`: the difference between the two counts must be visible."""
    reversal = _section(report, "head 순서 역전")

    assert "tau가 음수인 행 전체" in reversal
    assert "그중 동률 쌍이 있어 제외한 행" in reversal


def test_the_measured_gap_and_its_floor_are_printed_in_the_same_unit(report):
    """The speaker compares the two numbers, so both carry percentage points."""
    reversal = _section(report, "head 순서 역전")
    pair_lines = [line for line in reversal.splitlines() if "이 쌍의 바닥" in line]

    assert pair_lines
    for line in pair_lines:
        assert line.count("%p") == 2, line


def test_the_floor_limit_states_both_pairs_of_counts(report):
    """`limit_note`: the cut against the intersection is the sharpest limit."""
    reversal = _section(report, "head 순서 역전")

    assert "79" in reversal and "379" in reversal
    assert "3,851" in reversal
    assert "779" in reversal and "2,247" in reversal


def test_the_floor_belongs_to_the_analysis_intersection_not_a_third_stratum(report):
    """The floor is not one of the two strata, and does not invent a layer."""
    floor_lines = [
        line for line in report.splitlines() if "replicate 노이즈 바닥 (실측" in line
    ]

    assert floor_lines
    for line in floor_lines:
        assert "층=analysis_intersection" in line, line


def test_the_report_leaves_the_profile_views_figures_to_the_agreement_view(report):
    """#161 owns the position profile and the mean-profile correlation."""
    agreement = _section(report, "IG-ISM 일치도")

    assert "#161" in agreement
    # Named in the pointer, but no figure here is one.
    for line in report.splitlines():
        if "층=" in line:
            assert "평균 프로파일" not in line, line


def test_each_pair_shows_its_own_floor(report):
    reversal = _section(report, "head 순서 역전")

    for pair in fx.PAIR_FLOORS:
        assert pair.replace("_vs_", " vs ") in reversal
    assert "더 큰" in reversal  # the pair floor rule, stated


def test_the_ig_ism_agreement_has_no_verdict_and_carries_the_mismatch_note(report):
    agreement = _section(report, "IG-ISM 일치도")

    assert "0.7" in agreement
    assert "낮은 상관은 토큰 수준 attribution의 한계" in agreement
    # The only mention of a verdict is the sentence refusing to make one.
    assert "합격·불합격 판정을 붙이지 않는다" in agreement

    # The ban is on the displayed values: a verdict beside a number is what
    # would read as the model passing or failing. The notes may well say the
    # word, since what they say about it is that no verdict is attached.
    for line in agreement.splitlines():
        if not ("층=" in line or "IG-ISM 순위 상관" in line):
            continue
        for word in ("합격", "불합격", "통과", "실패", "충분", "기준"):
            assert word not in line, line


def test_the_ig_ism_agreement_is_per_stratum_and_per_head(report):
    agreement = _section(report, "IG-ISM 일치도")

    for stratum in ("generalisation", "pam_library"):
        for head in fx.ANALYSED_HEADS:
            assert any(
                f"층={stratum}" in line and f"head={head} " in line
                for line in agreement.splitlines()
            ), (stratum, head)


def test_every_displayed_value_carries_its_four_fields(report):
    provenance = [
        line for line in report.splitlines() if line.strip().startswith("층=")
    ]

    assert provenance
    for line in provenance:
        for field in ("층=", "head=", "통계=", "분모="):
            assert field in line, line


def test_the_claim_limit_is_printed_at_the_bottom(report):
    assert fx.CLAIM_LIMIT in report
    assert report.index(fx.CLAIM_LIMIT) > report.index("[head 순서 역전")


def test_it_ends_with_the_one_line_pointer_to_cross(report):
    pointers = [line for line in report.splitlines() if "`cross`" in line]

    assert len(pointers) == 1
    assert "교차 소견" in pointers[0]


def test_a_floor_that_disagrees_with_the_noise_floor_artifact_is_refused(
    reader, tmp_path
):
    """The reversal quotes the floor it applied; it must be the per-head one."""
    reversal = reader.head_order_reversal
    reversal["floor"]["by_head"]["evoCas9"] = fx.NOISE_FLOOR_POOLED_MEDIAN

    with pytest.raises(rd.ArtifactMismatch):
        spcas9_views.report(reader)


def test_no_line_leaks_a_python_repr(report):
    """A note built from a comprehension once printed as a tuple repr."""
    for line in report.splitlines():
        for leak in ('",)', "',)", "{'", '{"'):
            assert leak not in line, line


def test_a_validation_row_count_that_disagrees_with_the_ig_manifest_is_refused(reader):
    """The figure's value and its denominator come from two artifacts.

    `_population` shows the IG manifest's row count against the validation
    report's declared rows. If the two ever disagreed, the demo would print one
    artifact's count under the other's denominator.
    """
    reader.validation["strata"]["generalisation"]["rows"] = 999

    with pytest.raises(rd.ArtifactMismatch):
        spcas9_views.report(reader)


def test_an_artifact_that_offers_a_case_pool_is_refused(tmp_path):
    """`selection_pool: forbidden` is what makes the reversal a result, not a pool.

    Checked when the artifact loads, so the file itself is what must say it.
    """
    data_dir = fx.build_data_dir(tmp_path / "data")
    path = data_dir / "spcas9/v3_report/head_order_reversal.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selection_pool"] = "allowed"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(rd.ArtifactMismatch):
        spcas9_views.report(rd.SpCas9Reader(data_dir))


def test_every_quoted_artifact_note_carries_a_korean_gloss(report):
    """The quote is the authority; the gloss is what the speaker reads aloud."""
    lines = report.splitlines()
    quotes = [i for i, line in enumerate(lines) if "원문(artifact 기록)" in line]

    assert quotes
    for index in quotes:
        assert "우리말 (데모가 붙인 설명)" in lines[index + 1], lines[index]


def test_the_gloss_is_labelled_as_the_demos_words_not_the_artifacts(report):
    for line in report.splitlines():
        if "우리말" in line:
            assert "데모가 붙인 설명" in line, line
            assert "원문" not in line, line
