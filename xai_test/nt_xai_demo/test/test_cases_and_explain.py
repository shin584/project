"""`cases` and `explain`: the source data's own IDs, and absence stated as absence.

Two things are load-bearing here and both are about not inventing information.

The demo never mints an identifier. SpCas9 cases are addressed by the
`case_id` the case table holds (`GEN-ERRHI-01` ...) and Cas12a cases by
`source_row` + `role`, because that is how each source identifies them. A demo
ID would be a new fact about the analysis.

`explain` names the methods a row does *not* carry, with the reason. On the
real data exactly one of the 18 SpCas9 cases (`GEN-HEADGAP-02`, row 17068)
falls inside the frozen 400-row IG list; the other 17 carry no IG, and the
analysis records that as `ig: null` rather than a failure. A blank where IG
would go reads as a gap in the demo; "이 행은 IG 고정 명단 400행에 없음" reads
as what it is.

SpCas9 lookups always print the row's split role. Measured, every ISM and IG
row is `test`, so this is a cheap safeguard rather than a warning expected to
fire: it blocks reasoning about a train row's error, and shows immediately if
the premise ever breaks.
"""

from __future__ import annotations

import pytest

from xai_test.nt_xai_demo import cas12a_views, spcas9_views
from xai_test.nt_xai_demo import reader as rd
from xai_test.nt_xai_demo.test import reader_fixtures as fx

#: The fixture's case on a `val` row with no prediction: the safeguard's case.
NON_TEST_CASE = "LIB-ERRLO-01"
#: Inside the IG list, as row 17068 is on the real data.
CASE_WITH_IG = "GEN-HEADGAP-02"
#: Outside it, like the other 17 real cases.
CASE_WITHOUT_IG = "GEN-ERRHI-01"


@pytest.fixture
def data_dir(tmp_path):
    return fx.build_data_dir(tmp_path / "data")


@pytest.fixture
def spcas9(data_dir):
    return rd.SpCas9Reader(data_dir)


@pytest.fixture
def cas12a(data_dir):
    return rd.Cas12aReader(data_dir)


def provenance_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip().startswith("층=")]


# --------------------------------------------------------------------------- #
# the reader's case lookup
# --------------------------------------------------------------------------- #


def test_spcas9_cases_are_addressed_by_their_own_case_id(spcas9):
    assert spcas9.case_ids() == tuple(case_id for case_id, *_ in fx.SPCAS9_CASES)

    case = spcas9.case(CASE_WITH_IG)
    assert case["case_id"] == CASE_WITH_IG
    assert case["row_index"] == 11


def test_an_unknown_spcas9_case_id_names_the_ones_that_exist(spcas9):
    with pytest.raises(KeyError) as refusal:
        spcas9.case("GEN-ERRHI-99")

    assert CASE_WITH_IG in str(refusal.value)


def test_a_spcas9_case_can_also_be_found_by_its_row_index(spcas9):
    """`row_index` is the artifacts' shared axis, so a row number has to work."""
    assert spcas9.case("11")["case_id"] == CASE_WITH_IG


def test_cas12a_cases_are_addressed_by_source_row_and_role(cas12a):
    assert cas12a.case_ids() == tuple(
        f"{row}:{role}" for row, role, *_ in fx.CAS12A_CASES
    )

    case = cas12a.case("4:out_of_range")
    assert case["source_row"] == 4
    assert case["role"] == "out_of_range"
    # The row alone is enough while it names one case.
    assert cas12a.case("4")["role"] == "out_of_range"


def test_a_cas12a_row_outside_the_case_list_is_refused_by_name(cas12a):
    with pytest.raises(KeyError) as refusal:
        cas12a.case("99999")

    assert "99999" in str(refusal.value)


def test_spcas9_method_presence_states_the_reason_for_each_absence(spcas9):
    present = {m.method: m for m in spcas9.case_methods(CASE_WITH_IG)}
    absent = {m.method: m for m in spcas9.case_methods(CASE_WITHOUT_IG)}

    assert present["IG"].present and present["IG"].reason is None
    assert not absent["IG"].present
    assert str(spcas9.ig_rows.rows) in absent["IG"].reason
    assert "IG 고정 명단" in absent["IG"].reason
    # Both carry ISM: the case table is drawn from the swept population.
    assert present["ISM"].present and absent["ISM"].present
    # Every entry names the population that decides it, with its size.
    for entry in (*present.values(), *absent.values()):
        assert entry.population


def test_rollout_covers_the_row_but_carries_its_no_head_axis_limit(spcas9):
    """Rollout is not an absence: it covers the row and gives no per-head read.

    Calling it "없음" would understate real coverage (the rollout archive holds
    these rows); calling it plainly "있음" would invite the per-head reading it
    cannot support, since it has no head axis at all (ADR 0008).
    """
    rollout = {m.method: m for m in spcas9.case_methods(CASE_WITH_IG)}["rollout"]

    assert rollout.present
    assert rollout.reason is None
    assert "head" in rollout.limit
    assert "ADR 0008" in rollout.limit
    assert str(spcas9.rollout_rows.rows) in rollout.population
    # The line a view prints says both things at once.
    assert "있음" in rollout.text() and "다만" in rollout.text()


def test_cas12a_method_presence_quotes_the_artifacts_own_failure(cas12a):
    failed_case = f"{fx.CAS12A_IG_FAILED_ROW}:low_error"
    methods = {m.method: m for m in cas12a.case_methods(failed_case)}

    assert not methods["IG"].present
    assert fx.CAS12A_IG_FAILURE in methods["IG"].reason
    assert methods["ISM"].present
    assert methods["rollout"].present


# --------------------------------------------------------------------------- #
# `cases`
# --------------------------------------------------------------------------- #


def test_spcas9_cases_view_lists_every_case_by_its_source_id(spcas9):
    text = spcas9_views.cases(spcas9)

    for case_id, stratum, criterion, _ in fx.SPCAS9_CASES:
        assert case_id in text
        assert stratum in text
        assert criterion in text


def test_the_cases_view_invents_no_identifier_of_its_own(spcas9, cas12a):
    """Every ID on screen has to be one the artifact holds."""
    import re

    text = spcas9_views.cases(spcas9)
    known = set(spcas9.case_ids())
    for token in re.findall(r"\b(?:GEN|LIB)-[A-Z]+-\d+\b", text):
        assert token in known, token

    cas12a_text = cas12a_views.cases(cas12a)
    for row, role, *_ in fx.CAS12A_CASES:
        assert str(row) in cas12a_text and role in cas12a_text


def test_the_cases_view_says_which_cases_carry_ig(spcas9):
    text = spcas9_views.cases(spcas9)

    carries_ig = [
        case_id
        for case_id in spcas9.case_ids()
        if spcas9.case_annotation(case_id).get("ig") is not None
    ]
    assert CASE_WITH_IG in carries_ig and CASE_WITHOUT_IG not in carries_ig

    # The count *and* which cases, so nobody has to count them off screen.
    assert (
        f"사례 {len(spcas9.case_ids()):,}개 중 IG를 가진 사례 {len(carries_ig):,}개"
        in text
    )
    for case_id in carries_ig:
        assert case_id in text
    # Each case without it is marked on its own line, and the reason - the
    # same sentence for all of them - is stated once for the list.
    assert text.count("없음: IG") == len(spcas9.case_ids()) - len(carries_ig)
    assert f"IG 고정 명단 {spcas9.ig_rows.rows:,}행에 없다" in text


def test_cas12a_cases_view_prints_the_pool_it_actually_drew_from(cas12a):
    """`available`, not `eligible`: earlier categories already took their rows.

    The two differ for `low_error` (372 drawn from, 375 met the criterion), so
    a view quoting `eligible` as the denominator is caught by value here.
    """
    text = cas12a_views.cases(cas12a)

    for row, role, *_ in fx.CAS12A_CASES:
        available, eligible = fx.CAS12A_CATEGORY_POOLS[role]
        assert role in text
        assert f"분모={available:,}행" in text, role
        if eligible != available:
            # The gap is the exclusion, and it is stated rather than hidden.
            assert f"{eligible:,}행 중" in text, role


def test_both_cases_views_carry_the_four_fields_and_the_claim_limit(spcas9, cas12a):
    for text, limit in (
        (spcas9_views.cases(spcas9), fx.CLAIM_LIMIT),
        (cas12a_views.cases(cas12a), fx.CAS12A_LIMIT_TEXT),
    ):
        assert provenance_lines(text)
        for line in provenance_lines(text):
            for field in ("층=", "head=", "통계=", "분모="):
                assert field in line, line
        assert limit in text


def test_the_cases_view_keeps_the_library_qualifier_on_a_library_case(spcas9):
    """A library case is one of 5 targets, and the artifact says so per case.

    The qualifier is the artifact's own English wording, so it is printed
    verbatim and labelled as a quote rather than paraphrased into a claim the
    demo would then own.
    """
    text = spcas9_views.cases(spcas9)

    assert fx.LIB_QUALIFIER in text
    assert f"기록된 한정(원문): {fx.LIB_QUALIFIER}" in text


# --------------------------------------------------------------------------- #
# `explain`
# --------------------------------------------------------------------------- #


def test_explain_shows_the_methods_a_row_has_with_their_values(spcas9):
    text = spcas9_views.explain(spcas9, CASE_WITH_IG)

    assert CASE_WITH_IG in text
    assert "ISM" in text and "IG" in text
    # The recorded per-head figures, for the analysed heads only.
    for head in fx.ANALYSED_HEADS:
        assert head in text
    for head in fx.ISM_HEADS:
        if head not in fx.ANALYSED_HEADS:
            assert head not in text, head


def test_explain_states_the_missing_method_with_its_reason(spcas9):
    text = spcas9_views.explain(spcas9, CASE_WITHOUT_IG)

    assert "IG 고정 명단" in text
    assert str(spcas9.ig_rows.rows) in text
    assert "없" in text


def test_the_ig_bearing_case_shows_ig_values_the_others_do_not(spcas9):
    with_ig = spcas9_views.explain(spcas9, CASE_WITH_IG)
    without_ig = spcas9_views.explain(spcas9, CASE_WITHOUT_IG)

    recorded = f"{fx.CASE_IG_TOKENS['SpCas9'][-1]:.4g}"
    assert recorded in with_ig
    assert recorded not in without_ig


def test_explain_always_prints_the_split_role(spcas9):
    for case_id, _stratum, _criterion, row in fx.SPCAS9_CASES:
        text = spcas9_views.explain(spcas9, case_id)

        assert f"split 역할: {fx.SPLIT_ROLE[row]}" in text, case_id


def test_a_case_outside_the_test_split_is_visible_as_such(spcas9):
    """The safeguard's whole point: a non-test row cannot pass unlabelled."""
    text = spcas9_views.explain(spcas9, NON_TEST_CASE)

    assert f"split 역할: {fx.SPLIT_ROLE[30]}" in text
    assert fx.SPLIT_ROLE[30] == "val"  # the fixture's non-test row
    # A row off the test split has no prediction, and that is said, not blanked.
    assert "예측 없음" in text


def test_explain_carries_the_four_fields_and_the_claim_limit(spcas9):
    text = spcas9_views.explain(spcas9, CASE_WITH_IG)

    assert provenance_lines(text)
    for line in provenance_lines(text):
        for field in ("층=", "head=", "통계=", "분모="):
            assert field in line, line
    assert fx.CLAIM_LIMIT in text


def test_explain_describes_this_rows_own_head_order(spcas9):
    text = spcas9_views.explain(spcas9, CASE_WITH_IG)
    entry = next(c for c in fx.SPCAS9_CASES if c[0] == CASE_WITH_IG)
    order = fx.spcas9_annotation(entry)["head_order"]

    assert "head 순서 (이 행)" in text
    # The row's own tau, not the stratum's count of reversed rows.
    assert f"tau={float(order['tau']):.4g}" in text
    # Each inverted pair is named with its measured gap against its own floor.
    for pair in order["inverted_pairs"]:
        assert pair["pair"].replace("_vs_", " vs ") in text
        assert f"{float(pair['floor']):.4g}" in text


def test_the_row_head_order_section_does_not_read_as_a_selection(spcas9):
    text = spcas9_views.explain(spcas9, CASE_WITH_IG)

    # #155 puts the population figures in `report`; this section says so rather
    # than letting one row stand in for the stratum.
    assert "선정 기준이 아니다" in text
    assert "`report`" in text


def test_the_row_head_order_figures_carry_the_four_fields(spcas9):
    text = spcas9_views.explain(spcas9, CASE_WITH_IG)
    section = text.split("head 순서 (이 행)", 1)[1]

    lines = provenance_lines(section)
    assert lines
    for line in lines:
        for field in ("층=", "head=", "통계=", "분모="):
            assert field in line, line


def test_cas12a_explain_shows_its_methods_and_quotes_a_failure(cas12a):
    text = cas12a_views.explain(cas12a, f"{fx.CAS12A_IG_FAILED_ROW}:low_error")

    assert str(fx.CAS12A_IG_FAILED_ROW) in text
    assert fx.CAS12A_IG_FAILURE in text
    assert "ISM" in text
    assert fx.CAS12A_LIMIT_TEXT in text


def test_cas12a_explain_carries_the_four_fields(cas12a):
    text = cas12a_views.explain(cas12a, "4:out_of_range")

    assert provenance_lines(text)
    for line in provenance_lines(text):
        for field in ("층=", "head=", "통계=", "분모="):
            assert field in line, line


def test_cas12a_explain_states_the_rollout_interpretation_limit(cas12a):
    text = cas12a_views.explain(cas12a, "4:out_of_range")

    assert "rollout query 진단" in text
    assert "near_identical_queries" in text  # the artifact's recorded status
    assert "거의 동일" in text  # and the demo's Korean reading of it


def test_explain_refuses_an_unknown_id_by_naming_the_known_ones(spcas9, cas12a):
    with pytest.raises(KeyError):
        spcas9_views.explain(spcas9, "GEN-NOPE-01")
    with pytest.raises(KeyError):
        cas12a_views.explain(cas12a, "99999")


def test_no_view_here_passes_a_verdict_on_the_model(spcas9, cas12a):
    texts = (
        spcas9_views.cases(spcas9),
        spcas9_views.explain(spcas9, CASE_WITH_IG),
        cas12a_views.cases(cas12a),
        cas12a_views.explain(cas12a, "4:out_of_range"),
    )
    for text in texts:
        for word in ("합격", "불합격", "통과 판정", "우수", "충분히"):
            assert word not in text
